const std = @import("std");
const Allocator = std.mem.Allocator;
const Tensor = @import("../core/tensor.zig").Tensor;
const memory = @import("../core/memory.zig");
const accel = @import("../hw/accel/accel_interface.zig");
const Thread = std.Thread;

pub const RSFLayerConfig = struct {
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    seed_offset: u64 = 0,
    grad_mean: bool = true,
};

pub const RSFConfig = struct {
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    grad_mean: bool = true,
    max_dim: usize = 1 << 20,
    max_layers: usize = 1 << 20,
};

const SAVE_VERSION: u32 = 4;

fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch return error.Overflow;
}

fn checkedMulU64(a: u64, b: u64) !u64 {
    return std.math.mul(u64, a, b) catch return error.Overflow;
}

fn checkedAddU64(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch return error.Overflow;
}

fn validateTensor2D(t: *const Tensor) !void {
    if (t.shape.dims.len != 2) return error.ShapeMismatch;
    const expected = try checkedMul(t.shape.dims[0], t.shape.dims[1]);
    if (t.data.len != expected) return error.DataLengthMismatch;
}

fn validateTensor2DShape(t: *const Tensor, rows: usize, cols: usize) !void {
    if (t.shape.dims.len != 2 or t.shape.dims[0] != rows or t.shape.dims[1] != cols) return error.ShapeMismatch;
    const expected = try checkedMul(rows, cols);
    if (t.data.len != expected) return error.DataLengthMismatch;
}

fn ensureFiniteSlice(data: []const f32) !void {
    for (data) |v| {
        if (!std.math.isFinite(v)) return error.NonFinite;
    }
}

fn zeroTensor(t: *Tensor) void {
    @memset(t.data, 0.0);
}

fn tensorsOverlap(a: *const Tensor, b: *const Tensor) bool {
    if (a.data.len == 0 or b.data.len == 0) return false;
    const a_start: usize = @intFromPtr(a.data.ptr);
    const b_start: usize = @intFromPtr(b.data.ptr);
    const a_bytes = std.math.mul(usize, a.data.len, @sizeOf(f32)) catch return true;
    const b_bytes = std.math.mul(usize, b.data.len, @sizeOf(f32)) catch return true;
    const a_end = std.math.add(usize, a_start, a_bytes) catch return true;
    const b_end = std.math.add(usize, b_start, b_bytes) catch return true;
    return a_start < b_end and b_start < a_end;
}

fn allocTensorArray(allocator: Allocator, count: usize, rows: usize, cols: usize) ![]Tensor {
    var arr = try allocator.alloc(Tensor, count);
    errdefer allocator.free(arr);

    var initialized: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < initialized) : (i += 1) arr[i].deinit();
    }

    var i: usize = 0;
    while (i < count) : (i += 1) {
        arr[i] = try Tensor.init(allocator, &.{ rows, cols });
        initialized += 1;
    }

    return arr;
}

fn freeTensorArray(allocator: Allocator, arr: []Tensor) void {
    for (arr) |*t| t.deinit();
    allocator.free(arr);
}

pub const LayerCore = struct {
    s_weight: Tensor,
    t_weight: Tensor,
    s_bias: Tensor,
    t_bias: Tensor,
    s_weight_grad: ?Tensor,
    t_weight_grad: ?Tensor,
    s_bias_grad: ?Tensor,
    t_bias_grad: ?Tensor,
    dim: usize,
    allocator: Allocator,
    clip_min: f32,
    clip_max: f32,
    grad_mean: bool,
    rwlock: Thread.RwLock,

    fn initOwned(allocator: Allocator, dim: usize, config: RSFLayerConfig) !LayerCore {
        if (dim == 0) return error.InvalidDimension;
        if (!std.math.isFinite(config.clip_min) or !std.math.isFinite(config.clip_max)) return error.NonFinite;
        if (!(config.clip_min < config.clip_max)) return error.InvalidConfig;
        if (config.clip_max > 20.0 or config.clip_min < -20.0) return error.InvalidConfig;

        _ = try checkedMul(dim, dim);

        const fan_in: f32 = @floatFromInt(dim);
        const fan_out: f32 = @floatFromInt(dim);
        const fan_sum = fan_in + fan_out;
        if (!(fan_sum > 0.0)) return error.InvalidDimension;

        const xavier_bound: f32 = @sqrt(6.0 / fan_sum);
        const weight_shape = [_]usize{ dim, dim };
        const bias_shape = [_]usize{ 1, dim };

        const seed1 = try checkedAddU64(42, config.seed_offset);
        const seed2 = try checkedAddU64(43, config.seed_offset);

        var s_w = try Tensor.randomUniform(allocator, &weight_shape, -xavier_bound, xavier_bound, seed1);
        errdefer s_w.deinit();

        var t_w = try Tensor.randomUniform(allocator, &weight_shape, -xavier_bound, xavier_bound, seed2);
        errdefer t_w.deinit();

        var s_b = try Tensor.zeros(allocator, &bias_shape);
        errdefer s_b.deinit();

        var t_b = try Tensor.zeros(allocator, &bias_shape);
        errdefer t_b.deinit();

        return LayerCore{
            .s_weight = s_w,
            .t_weight = t_w,
            .s_bias = s_b,
            .t_bias = t_b,
            .s_weight_grad = null,
            .t_weight_grad = null,
            .s_bias_grad = null,
            .t_bias_grad = null,
            .dim = dim,
            .allocator = allocator,
            .clip_min = config.clip_min,
            .clip_max = config.clip_max,
            .grad_mean = config.grad_mean,
            .rwlock = .{},
        };
    }

    fn deinitOwned(self: *LayerCore) void {
        self.s_weight.deinit();
        self.t_weight.deinit();
        self.s_bias.deinit();
        self.t_bias.deinit();
        if (self.s_weight_grad) |*g| g.deinit();
        if (self.t_weight_grad) |*g| g.deinit();
        if (self.s_bias_grad) |*g| g.deinit();
        if (self.t_bias_grad) |*g| g.deinit();
        self.s_weight_grad = null;
        self.t_weight_grad = null;
        self.s_bias_grad = null;
        self.t_bias_grad = null;
    }

    fn ensureGradients(self: *LayerCore) !void {
        const need_swg = self.s_weight_grad == null;
        const need_twg = self.t_weight_grad == null;
        const need_sbg = self.s_bias_grad == null;
        const need_tbg = self.t_bias_grad == null;

        if (!(need_swg or need_twg or need_sbg or need_tbg)) return;

        const weight_shape = [_]usize{ self.dim, self.dim };
        const bias_shape = [_]usize{ 1, self.dim };

        var swg_new: ?Tensor = null;
        var twg_new: ?Tensor = null;
        var sbg_new: ?Tensor = null;
        var tbg_new: ?Tensor = null;

        errdefer {
            if (swg_new) |*t| t.deinit();
            if (twg_new) |*t| t.deinit();
            if (sbg_new) |*t| t.deinit();
            if (tbg_new) |*t| t.deinit();
        }

        if (need_swg) swg_new = try Tensor.zeros(self.allocator, &weight_shape);
        if (need_twg) twg_new = try Tensor.zeros(self.allocator, &weight_shape);
        if (need_sbg) sbg_new = try Tensor.zeros(self.allocator, &bias_shape);
        if (need_tbg) tbg_new = try Tensor.zeros(self.allocator, &bias_shape);

        if (swg_new) |t| self.s_weight_grad = t;
        if (twg_new) |t| self.t_weight_grad = t;
        if (sbg_new) |t| self.s_bias_grad = t;
        if (tbg_new) |t| self.t_bias_grad = t;
    }

    fn zeroGradients(self: *LayerCore) void {
        if (self.s_weight_grad) |*g| zeroTensor(g);
        if (self.t_weight_grad) |*g| zeroTensor(g);
        if (self.s_bias_grad) |*g| zeroTensor(g);
        if (self.t_bias_grad) |*g| zeroTensor(g);
    }

    fn validatePair(self: *const LayerCore, a: *const Tensor, b: *const Tensor) !usize {
        try validateTensor2D(a);
        try validateTensor2D(b);
        if (a.shape.dims[1] != self.dim or b.shape.dims[1] != self.dim) return error.ShapeMismatch;
        if (a.shape.dims[0] != b.shape.dims[0]) return error.ShapeMismatch;
        const batch_size = a.shape.dims[0];
        if (batch_size == 0) return error.InvalidBatchSize;
        _ = try checkedMul(batch_size, self.dim);
        return batch_size;
    }

    fn gradScale(self: *const LayerCore, batch_size: usize) f32 {
        if (!self.grad_mean) return 1.0;
        const scale = 1.0 / @as(f32, @floatFromInt(batch_size));
        return if (std.math.isFinite(scale)) scale else 1.0;
    }

    fn computeTranslationInto(self: *const LayerCore, input: *const Tensor, out: []f32) void {
        const batch_size = input.shape.dims[0];
        const dim = self.dim;
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row_in = input.data[b * dim .. b * dim + dim];
            const row_out = out[b * dim .. b * dim + dim];
            var d: usize = 0;
            while (d < dim) : (d += 1) {
                var sum: f32 = self.t_bias.data[d];
                const w_row = self.t_weight.data[d * dim .. d * dim + dim];
                var j: usize = 0;
                while (j < dim) : (j += 1) sum += w_row[j] * row_in[j];
                row_out[d] = sum;
            }
        }
    }

    fn computeScaleInto(self: *const LayerCore, input: *const Tensor, out: []f32) void {
        const batch_size = input.shape.dims[0];
        const dim = self.dim;
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const row_in = input.data[b * dim .. b * dim + dim];
            const row_out = out[b * dim .. b * dim + dim];
            var d: usize = 0;
            while (d < dim) : (d += 1) {
                var sum: f32 = self.s_bias.data[d];
                const w_row = self.s_weight.data[d * dim .. d * dim + dim];
                var j: usize = 0;
                while (j < dim) : (j += 1) sum += w_row[j] * row_in[j];
                const clipped = if (sum < self.clip_min) self.clip_min else if (sum > self.clip_max) self.clip_max else sum;
                row_out[d] = clipped;
            }
            d = 0;
            while (d < dim) : (d += 1) row_out[d] = @exp(row_out[d]);
        }
    }

    fn forwardInPlace(self: *const LayerCore, x1: *Tensor, x2: *Tensor) !void {
        if (tensorsOverlap(x1, x2)) return error.AliasedBuffers;
        const batch_size = try self.validatePair(x1, x2);
        const bd = try checkedMul(batch_size, self.dim);

        const scale = try self.allocator.alloc(f32, bd);
        defer self.allocator.free(scale);
        self.computeScaleInto(x2, scale);

        var i: usize = 0;
        while (i < bd) : (i += 1) x1.data[i] *= scale[i];

        const trans = try self.allocator.alloc(f32, bd);
        defer self.allocator.free(trans);
        self.computeTranslationInto(x1, trans);

        i = 0;
        while (i < bd) : (i += 1) x2.data[i] += trans[i];
    }

    fn inverseInPlace(self: *const LayerCore, y1: *Tensor, y2: *Tensor) !void {
        if (tensorsOverlap(y1, y2)) return error.AliasedBuffers;
        const batch_size = try self.validatePair(y1, y2);
        const bd = try checkedMul(batch_size, self.dim);

        const trans = try self.allocator.alloc(f32, bd);
        defer self.allocator.free(trans);
        self.computeTranslationInto(y1, trans);

        var i: usize = 0;
        while (i < bd) : (i += 1) y2.data[i] -= trans[i];

        const scale = try self.allocator.alloc(f32, bd);
        defer self.allocator.free(scale);
        self.computeScaleInto(y2, scale);

        i = 0;
        while (i < bd) : (i += 1) y1.data[i] /= scale[i];
    }

    fn backwardFromOutputs(
        self: *LayerCore,
        y1: *const Tensor,
        y2: *const Tensor,
        dy1_in: *const Tensor,
        dy2_in: *const Tensor,
        x1_out: *Tensor,
        x2_out: *Tensor,
        dx1_out: *Tensor,
        dx2_out: *Tensor,
        dy1_total: []f32,
        ds: []f32,
    ) !void {
        const batch_size = y1.shape.dims[0];
        try self.ensureGradients();

        const dim = self.dim;
        const grad_scale = self.gradScale(batch_size);

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const y1_row = y1.data[b * dim .. b * dim + dim];
            const dy1_row = dy1_in.data[b * dim .. b * dim + dim];
            const dy2_row = dy2_in.data[b * dim .. b * dim + dim];
            const dy1_total_row = dy1_total[b * dim .. b * dim + dim];

            @memcpy(dy1_total_row, dy1_row);
            var d: usize = 0;
            while (d < dim) : (d += 1) {
                const dy2_val = dy2_row[d];
                const t_row = self.t_weight.data[d * dim .. d * dim + dim];
                var j: usize = 0;
                while (j < dim) : (j += 1) dy1_total_row[j] += t_row[j] * dy2_val;
            }

            if (self.t_weight_grad) |*twg| {
                d = 0;
                while (d < dim) : (d += 1) {
                    const dyv = dy2_row[d] * grad_scale;
                    var j2: usize = 0;
                    while (j2 < dim) : (j2 += 1) twg.data[d * dim + j2] += dyv * y1_row[j2];
                }
            }

            if (self.t_bias_grad) |*tbg| {
                d = 0;
                while (d < dim) : (d += 1) tbg.data[d] += dy2_row[d] * grad_scale;
            }
        }

        b = 0;
        while (b < batch_size) : (b += 1) {
            const y1_row = y1.data[b * dim .. b * dim + dim];
            const y2_row = y2.data[b * dim .. b * dim + dim];
            const x2_row = x2_out.data[b * dim .. b * dim + dim];
            const x1_row = x1_out.data[b * dim .. b * dim + dim];
            const dx1_row = dx1_out.data[b * dim .. b * dim + dim];
            const ds_row = ds[b * dim .. b * dim + dim];
            const dy1_total_row = dy1_total[b * dim .. b * dim + dim];

            var d: usize = 0;
            while (d < dim) : (d += 1) {
                var trans_sum: f32 = self.t_bias.data[d];
                const t_row = self.t_weight.data[d * dim .. d * dim + dim];
                var j: usize = 0;
                while (j < dim) : (j += 1) trans_sum += t_row[j] * y1_row[j];
                x2_row[d] = y2_row[d] - trans_sum;
            }

            var d2: usize = 0;
            while (d2 < dim) : (d2 += 1) {
                var pre_sum: f32 = self.s_bias.data[d2];
                const s_row = self.s_weight.data[d2 * dim .. d2 * dim + dim];
                var j2: usize = 0;
                while (j2 < dim) : (j2 += 1) pre_sum += s_row[j2] * x2_row[j2];

                const clipped = if (pre_sum < self.clip_min) self.clip_min else if (pre_sum > self.clip_max) self.clip_max else pre_sum;
                const scale = @exp(clipped);

                x1_row[d2] = y1_row[d2] / scale;
                dx1_row[d2] = dy1_total_row[d2] * scale;
                ds_row[d2] = if (pre_sum < self.clip_min or pre_sum > self.clip_max) 0.0 else dy1_total_row[d2] * y1_row[d2];
            }

            if (self.s_weight_grad) |*swg| {
                var d3: usize = 0;
                while (d3 < dim) : (d3 += 1) {
                    const dsv = ds_row[d3] * grad_scale;
                    var j3: usize = 0;
                    while (j3 < dim) : (j3 += 1) swg.data[d3 * dim + j3] += dsv * x2_row[j3];
                }
            }

            if (self.s_bias_grad) |*sbg| {
                var d4: usize = 0;
                while (d4 < dim) : (d4 += 1) sbg.data[d4] += ds_row[d4] * grad_scale;
            }

            const dx2_row = dx2_out.data[b * dim .. b * dim + dim];
            const dy2_row = dy2_in.data[b * dim .. b * dim + dim];
            @memcpy(dx2_row, dy2_row);
            var d5: usize = 0;
            while (d5 < dim) : (d5 += 1) {
                const ds_val = ds_row[d5];
                const s_row = self.s_weight.data[d5 * dim .. d5 * dim + dim];
                var j4: usize = 0;
                while (j4 < dim) : (j4 += 1) dx2_row[j4] += s_row[j4] * ds_val;
            }
        }
    }

    fn backwardFromActivations(
        self: *LayerCore,
        x2: *const Tensor,
        y1: *const Tensor,
        dy1_in: *const Tensor,
        dy2_in: *const Tensor,
        dx1_out: *Tensor,
        dx2_out: *Tensor,
        dy1_total: []f32,
        ds: []f32,
    ) !void {
        const batch_size = x2.shape.dims[0];
        try self.ensureGradients();

        const dim = self.dim;
        const grad_scale = self.gradScale(batch_size);

        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            const y1_row = y1.data[b * dim .. b * dim + dim];
            const dy1_row = dy1_in.data[b * dim .. b * dim + dim];
            const dy2_row = dy2_in.data[b * dim .. b * dim + dim];
            const dy1_total_row = dy1_total[b * dim .. b * dim + dim];

            @memcpy(dy1_total_row, dy1_row);
            var d: usize = 0;
            while (d < dim) : (d += 1) {
                const dy2_val = dy2_row[d];
                const t_row = self.t_weight.data[d * dim .. d * dim + dim];
                var j: usize = 0;
                while (j < dim) : (j += 1) dy1_total_row[j] += t_row[j] * dy2_val;
            }

            if (self.t_weight_grad) |*twg| {
                d = 0;
                while (d < dim) : (d += 1) {
                    const dyv = dy2_row[d] * grad_scale;
                    var j2: usize = 0;
                    while (j2 < dim) : (j2 += 1) twg.data[d * dim + j2] += dyv * y1_row[j2];
                }
            }

            if (self.t_bias_grad) |*tbg| {
                d = 0;
                while (d < dim) : (d += 1) tbg.data[d] += dy2_row[d] * grad_scale;
            }
        }

        b = 0;
        while (b < batch_size) : (b += 1) {
            const x2_row = x2.data[b * dim .. b * dim + dim];
            const y1_row = y1.data[b * dim .. b * dim + dim];
            const dy2_row = dy2_in.data[b * dim .. b * dim + dim];
            const dy1_total_row = dy1_total[b * dim .. b * dim + dim];
            const ds_row = ds[b * dim .. b * dim + dim];
            const dx1_row = dx1_out.data[b * dim .. b * dim + dim];
            const dx2_row = dx2_out.data[b * dim .. b * dim + dim];

            var d: usize = 0;
            while (d < dim) : (d += 1) {
                var pre_sum: f32 = self.s_bias.data[d];
                const s_row = self.s_weight.data[d * dim .. d * dim + dim];
                var j: usize = 0;
                while (j < dim) : (j += 1) pre_sum += s_row[j] * x2_row[j];

                const clipped = if (pre_sum < self.clip_min) self.clip_min else if (pre_sum > self.clip_max) self.clip_max else pre_sum;
                const scale = @exp(clipped);

                dx1_row[d] = dy1_total_row[d] * scale;
                ds_row[d] = if (pre_sum < self.clip_min or pre_sum > self.clip_max) 0.0 else dy1_total_row[d] * y1_row[d];
            }

            if (self.s_weight_grad) |*swg| {
                var d2: usize = 0;
                while (d2 < dim) : (d2 += 1) {
                    const dsv = ds_row[d2] * grad_scale;
                    var j2: usize = 0;
                    while (j2 < dim) : (j2 += 1) swg.data[d2 * dim + j2] += dsv * x2_row[j2];
                }
            }

            if (self.s_bias_grad) |*sbg| {
                var d3: usize = 0;
                while (d3 < dim) : (d3 += 1) sbg.data[d3] += ds_row[d3] * grad_scale;
            }

            @memcpy(dx2_row, dy2_row);
            var d4: usize = 0;
            while (d4 < dim) : (d4 += 1) {
                const ds_val = ds_row[d4];
                const s_row = self.s_weight.data[d4 * dim .. d4 * dim + dim];
                var j3: usize = 0;
                while (j3 < dim) : (j3 += 1) dx2_row[j3] += s_row[j3] * ds_val;
            }
        }
    }
};

const LayerRegistryEntry = struct {
    core: *LayerCore,
    active_ops: usize,
    destroyed: bool,
};

var g_layer_registry_mutex: Thread.Mutex = .{};
var g_layer_registry = std.AutoHashMap(u64, LayerRegistryEntry).init(std.heap.page_allocator);
var g_layer_next_id = std.atomic.Value(u64).init(1);

fn destroyLayerCore(core: *LayerCore) void {
    const allocator = core.allocator;
    core.deinitOwned();
    allocator.destroy(core);
}

fn registerLayerCore(core: *LayerCore) !u64 {
    g_layer_registry_mutex.lock();
    defer g_layer_registry_mutex.unlock();
    var id: u64 = 0;
    while (id == 0 or g_layer_registry.contains(id)) {
        id = g_layer_next_id.fetchAdd(1, .monotonic);
    }
    try g_layer_registry.put(id, .{ .core = core, .active_ops = 0, .destroyed = false });
    return id;
}

fn acquireLayerCore(id: u64) !*LayerCore {
    if (id == 0) return error.NotInitialized;
    g_layer_registry_mutex.lock();
    defer g_layer_registry_mutex.unlock();
    const entry = g_layer_registry.getPtr(id) orelse return error.NotInitialized;
    if (entry.destroyed) return error.NotInitialized;
    entry.active_ops += 1;
    return entry.core;
}

fn releaseLayerCore(id: u64) void {
    if (id == 0) return;
    var core_to_destroy: ?*LayerCore = null;
    g_layer_registry_mutex.lock();
    if (g_layer_registry.getPtr(id)) |entry| {
        if (entry.active_ops > 0) entry.active_ops -= 1;
        if (entry.destroyed and entry.active_ops == 0) {
            if (g_layer_registry.fetchRemove(id)) |kv| core_to_destroy = kv.value.core;
        }
    }
    g_layer_registry_mutex.unlock();
    if (core_to_destroy) |core| destroyLayerCore(core);
}

fn requestDestroyLayerCore(id: u64) void {
    if (id == 0) return;
    var core_to_destroy: ?*LayerCore = null;
    g_layer_registry_mutex.lock();
    if (g_layer_registry.getPtr(id)) |entry| {
        entry.destroyed = true;
        if (entry.active_ops == 0) {
            if (g_layer_registry.fetchRemove(id)) |kv| core_to_destroy = kv.value.core;
        }
    }
    g_layer_registry_mutex.unlock();
    if (core_to_destroy) |core| destroyLayerCore(core);
}

pub const RSFLayer = struct {
    id: u64 = 0,

    pub fn init(allocator: Allocator, dim: usize) !RSFLayer {
        return initWithConfig(allocator, dim, .{});
    }

    pub fn initWithArena(arena: *memory.ArenaAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(arena.allocator(), dim, config);
    }

    pub fn initWithPool(pool: *memory.PoolAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(pool.allocator(), dim, config);
    }

    pub fn initWithSlab(slab: *memory.SlabAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(slab.allocator(), dim, config);
    }

    pub fn initWithBuddy(buddy: *memory.BuddyAllocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        return initWithConfig(buddy.allocator(), dim, config);
    }

    pub fn initWithConfig(allocator: Allocator, dim: usize, config: RSFLayerConfig) !RSFLayer {
        var core = try allocator.create(LayerCore);
        errdefer allocator.destroy(core);

        core.* = try LayerCore.initOwned(allocator, dim, config);
        errdefer core.deinitOwned();

        const id = try registerLayerCore(core);
        return RSFLayer{ .id = id };
    }

    pub fn ensureGradients(self: *RSFLayer) !void {
        const id = self.id;
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try core.ensureGradients();
    }

    pub fn deinit(self: *RSFLayer) void {
        const id = self.id;
        if (id == 0) return;
        self.id = 0;
        requestDestroyLayerCore(id);
    }

    pub fn zeroGradients(self: *RSFLayer) void {
        const id = self.id;
        const core = acquireLayerCore(id) catch return;
        defer releaseLayerCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        core.zeroGradients();
    }

    pub fn forward(self: *const RSFLayer, x1: *Tensor, x2: *Tensor) !void {
        const id = self.id;
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try core.forwardInPlace(x1, x2);
    }

    pub fn inverse(self: *const RSFLayer, y1: *Tensor, y2: *Tensor) !void {
        const id = self.id;
        const core = try acquireLayerCore(id);
        defer releaseLayerCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try core.inverseInPlace(y1, y2);
    }
};

pub const RSFCore = struct {
    allocator: Allocator,
    dim: usize,
    num_layers: usize,
    layers: []LayerCore,
    cfg: RSFConfig,
    rwlock: Thread.RwLock,
    gpu_accel: ?accel.RSFAccelerator,
    gpu_available: std.atomic.Value(u8),
    gpu_weight_version: u64,
    cpu_weight_version: u64,
    f16_buf: ?[]f16,
};

const ModelRegistryEntry = struct {
    core: *RSFCore,
    active_ops: usize,
    destroyed: bool,
};

var g_model_registry_mutex: Thread.Mutex = .{};
var g_model_registry = std.AutoHashMap(u64, ModelRegistryEntry).init(std.heap.page_allocator);
var g_model_next_id = std.atomic.Value(u64).init(1);

fn destroyModelCore(core: *RSFCore) void {
    if (core.gpu_accel) |*ga| {
        ga.deinit();
        core.gpu_accel = null;
    }
    if (core.f16_buf) |buf| {
        core.allocator.free(buf);
        core.f16_buf = null;
    }
    core.gpu_available.store(0, .monotonic);

    const allocator = core.allocator;
    var i: usize = 0;
    while (i < core.num_layers) : (i += 1) core.layers[i].deinitOwned();
    allocator.free(core.layers);
    allocator.destroy(core);
}

fn registerModelCore(core: *RSFCore) !u64 {
    g_model_registry_mutex.lock();
    defer g_model_registry_mutex.unlock();
    var id: u64 = 0;
    while (id == 0 or g_model_registry.contains(id)) {
        id = g_model_next_id.fetchAdd(1, .monotonic);
    }
    try g_model_registry.put(id, .{ .core = core, .active_ops = 0, .destroyed = false });
    return id;
}

fn acquireModelCore(id: u64) !*RSFCore {
    if (id == 0) return error.NotInitialized;
    g_model_registry_mutex.lock();
    defer g_model_registry_mutex.unlock();
    const entry = g_model_registry.getPtr(id) orelse return error.NotInitialized;
    if (entry.destroyed) return error.NotInitialized;
    entry.active_ops += 1;
    return entry.core;
}

fn releaseModelCore(id: u64) void {
    if (id == 0) return;
    var core_to_destroy: ?*RSFCore = null;
    g_model_registry_mutex.lock();
    if (g_model_registry.getPtr(id)) |entry| {
        if (entry.active_ops > 0) entry.active_ops -= 1;
        if (entry.destroyed and entry.active_ops == 0) {
            if (g_model_registry.fetchRemove(id)) |kv| core_to_destroy = kv.value.core;
        }
    }
    g_model_registry_mutex.unlock();
    if (core_to_destroy) |core| destroyModelCore(core);
}

fn requestDestroyModelCore(id: u64) void {
    if (id == 0) return;
    var core_to_destroy: ?*RSFCore = null;
    g_model_registry_mutex.lock();
    if (g_model_registry.getPtr(id)) |entry| {
        entry.destroyed = true;
        if (entry.active_ops == 0) {
            if (g_model_registry.fetchRemove(id)) |kv| core_to_destroy = kv.value.core;
        }
    }
    g_model_registry_mutex.unlock();
    if (core_to_destroy) |core| destroyModelCore(core);
}

const GradSnapshot = struct {
    had_s_weight: bool,
    had_t_weight: bool,
    had_s_bias: bool,
    had_t_bias: bool,
    s_weight: ?[]f32,
    t_weight: ?[]f32,
    s_bias: ?[]f32,
    t_bias: ?[]f32,
};

fn captureModelGradSnapshots(allocator: Allocator, layers: []LayerCore) ![]GradSnapshot {
    var snaps = try allocator.alloc(GradSnapshot, layers.len);
    errdefer allocator.free(snaps);

    var initialized: usize = 0;
    errdefer {
        var i: usize = 0;
        while (i < initialized) : (i += 1) {
            if (snaps[i].s_weight) |s| allocator.free(s);
            if (snaps[i].t_weight) |s| allocator.free(s);
            if (snaps[i].s_bias) |s| allocator.free(s);
            if (snaps[i].t_bias) |s| allocator.free(s);
        }
    }

    var i: usize = 0;
    while (i < layers.len) : (i += 1) {
        snaps[i] = .{
            .had_s_weight = layers[i].s_weight_grad != null,
            .had_t_weight = layers[i].t_weight_grad != null,
            .had_s_bias = layers[i].s_bias_grad != null,
            .had_t_bias = layers[i].t_bias_grad != null,
            .s_weight = null,
            .t_weight = null,
            .s_bias = null,
            .t_bias = null,
        };

        if (layers[i].s_weight_grad) |g| snaps[i].s_weight = try allocator.dupe(f32, g.data);
        if (layers[i].t_weight_grad) |g| snaps[i].t_weight = try allocator.dupe(f32, g.data);
        if (layers[i].s_bias_grad) |g| snaps[i].s_bias = try allocator.dupe(f32, g.data);
        if (layers[i].t_bias_grad) |g| snaps[i].t_bias = try allocator.dupe(f32, g.data);

        initialized += 1;
    }

    return snaps;
}

fn restoreModelGradSnapshots(layers: []LayerCore, snaps: []const GradSnapshot) void {
    var i: usize = 0;
    while (i < layers.len and i < snaps.len) : (i += 1) {
        var layer = &layers[i];
        const snap = snaps[i];

        if (!snap.had_s_weight) {
            if (layer.s_weight_grad) |*g| {
                g.deinit();
                layer.s_weight_grad = null;
            }
        } else if (snap.s_weight) |saved| {
            if (layer.s_weight_grad) |*g| @memcpy(g.data, saved);
        }

        if (!snap.had_t_weight) {
            if (layer.t_weight_grad) |*g| {
                g.deinit();
                layer.t_weight_grad = null;
            }
        } else if (snap.t_weight) |saved| {
            if (layer.t_weight_grad) |*g| @memcpy(g.data, saved);
        }

        if (!snap.had_s_bias) {
            if (layer.s_bias_grad) |*g| {
                g.deinit();
                layer.s_bias_grad = null;
            }
        } else if (snap.s_bias) |saved| {
            if (layer.s_bias_grad) |*g| @memcpy(g.data, saved);
        }

        if (!snap.had_t_bias) {
            if (layer.t_bias_grad) |*g| {
                g.deinit();
                layer.t_bias_grad = null;
            }
        } else if (snap.t_bias) |saved| {
            if (layer.t_bias_grad) |*g| @memcpy(g.data, saved);
        }
    }
}

fn freeModelGradSnapshots(allocator: Allocator, snaps: []GradSnapshot) void {
    for (snaps) |snap| {
        if (snap.s_weight) |s| allocator.free(s);
        if (snap.t_weight) |s| allocator.free(s);
        if (snap.s_bias) |s| allocator.free(s);
        if (snap.t_bias) |s| allocator.free(s);
    }
    allocator.free(snaps);
}

fn splitInto(core: *const RSFCore, x: *const Tensor, x1: *Tensor, x2: *Tensor) !usize {
    const dim2 = try checkedMul(core.dim, 2);
    if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = x.shape.dims[0];

    const bd = try checkedMul(batch_size, core.dim);
    const bd2 = try checkedMul(batch_size, dim2);
    if (x1.data.len != bd or x2.data.len != bd or x.data.len != bd2) return error.DataLengthMismatch;

    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const src_offset = b * dim2;
        const dst_offset = b * core.dim;
        @memcpy(x1.data[dst_offset .. dst_offset + core.dim], x.data[src_offset .. src_offset + core.dim]);
        @memcpy(x2.data[dst_offset .. dst_offset + core.dim], x.data[src_offset + core.dim .. src_offset + dim2]);
    }

    return batch_size;
}

fn mergeFrom(core: *const RSFCore, x1: *const Tensor, x2: *const Tensor, out: *Tensor) !void {
    const dim2 = try checkedMul(core.dim, 2);
    const batch_size = x1.shape.dims[0];
    const bd = try checkedMul(batch_size, core.dim);
    const bd2 = try checkedMul(batch_size, dim2);
    if (x1.data.len != bd or x2.data.len != bd or out.data.len != bd2) return error.DataLengthMismatch;

    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const src_offset = b * core.dim;
        const dst_offset = b * dim2;
        @memcpy(out.data[dst_offset .. dst_offset + core.dim], x1.data[src_offset .. src_offset + core.dim]);
        @memcpy(out.data[dst_offset + core.dim .. dst_offset + dim2], x2.data[src_offset .. src_offset + core.dim]);
    }
}

fn forwardOnCore(core: *const RSFCore, x: *Tensor) !void {
    try validateTensor2D(x);
    const dim2 = try checkedMul(core.dim, 2);
    if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = x.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;

    var x1 = try Tensor.init(core.allocator, &.{ batch_size, core.dim });
    defer x1.deinit();
    var x2 = try Tensor.init(core.allocator, &.{ batch_size, core.dim });
    defer x2.deinit();

    _ = try splitInto(core, x, &x1, &x2);

    var i: usize = 0;
    while (i < core.num_layers) : (i += 1) try core.layers[i].forwardInPlace(&x1, &x2);

    try mergeFrom(core, &x1, &x2, x);
}

fn inverseOnCore(core: *const RSFCore, y: *Tensor) !void {
    try validateTensor2D(y);
    const dim2 = try checkedMul(core.dim, 2);
    if (y.shape.dims[1] != dim2) return error.ShapeMismatch;
    const batch_size = y.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;

    var y1 = try Tensor.init(core.allocator, &.{ batch_size, core.dim });
    defer y1.deinit();
    var y2 = try Tensor.init(core.allocator, &.{ batch_size, core.dim });
    defer y2.deinit();

    _ = try splitInto(core, y, &y1, &y2);

    var idx = core.num_layers;
    while (idx > 0) : (idx -= 1) try core.layers[idx - 1].inverseInPlace(&y1, &y2);

    try mergeFrom(core, &y1, &y2, y);
}

fn backwardOnCore(core: *RSFCore, grad_output: *const Tensor, input: *const Tensor, grad_input_out: *Tensor) !void {
    try validateTensor2D(grad_output);
    try validateTensor2D(input);
    try validateTensor2D(grad_input_out);

    const dim2 = try checkedMul(core.dim, 2);
    if (input.shape.dims[1] != dim2) return error.ShapeMismatch;
    if (grad_output.shape.dims[0] != input.shape.dims[0] or grad_output.shape.dims[1] != input.shape.dims[1]) return error.ShapeMismatch;
    if (grad_input_out.shape.dims[0] != input.shape.dims[0] or grad_input_out.shape.dims[1] != input.shape.dims[1]) return error.ShapeMismatch;

    const batch_size = input.shape.dims[0];
    if (batch_size == 0) return error.InvalidBatchSize;

    var stage_x1 = try allocTensorArray(core.allocator, core.num_layers + 1, batch_size, core.dim);
    defer freeTensorArray(core.allocator, stage_x1);

    var stage_x2 = try allocTensorArray(core.allocator, core.num_layers + 1, batch_size, core.dim);
    defer freeTensorArray(core.allocator, stage_x2);

    _ = try splitInto(core, input, &stage_x1[0], &stage_x2[0]);

    var l: usize = 0;
    while (l < core.num_layers) : (l += 1) {
        @memcpy(stage_x1[l + 1].data, stage_x1[l].data);
        @memcpy(stage_x2[l + 1].data, stage_x2[l].data);
        try core.layers[l].forwardInPlace(&stage_x1[l + 1], &stage_x2[l + 1]);
    }

    var cur_dy1 = try Tensor.init(core.allocator, &.{ batch_size, core.dim });
    defer cur_dy1.deinit();

    var cur_dy2 = try Tensor.init(core.allocator, &.{ batch_size, core.dim });
    defer cur_dy2.deinit();

    _ = try splitInto(core, grad_output, &cur_dy1, &cur_dy2);

    var next_dx1 = try Tensor.init(core.allocator, &.{ batch_size, core.dim });
    defer next_dx1.deinit();

    var next_dx2 = try Tensor.init(core.allocator, &.{ batch_size, core.dim });
    defer next_dx2.deinit();

    const bd = try checkedMul(batch_size, core.dim);
    const dy1_total = try core.allocator.alloc(f32, bd);
    defer core.allocator.free(dy1_total);
    const ds = try core.allocator.alloc(f32, bd);
    defer core.allocator.free(ds);

    const snaps = try captureModelGradSnapshots(core.allocator, core.layers);
    errdefer restoreModelGradSnapshots(core.layers, snaps);
    defer freeModelGradSnapshots(core.allocator, snaps);

    var idx = core.num_layers;
    while (idx > 0) : (idx -= 1) {
        try core.layers[idx - 1].backwardFromActivations(&stage_x2[idx - 1], &stage_x1[idx], &cur_dy1, &cur_dy2, &next_dx1, &next_dx2, dy1_total, ds);
        std.mem.swap(Tensor, &cur_dy1, &next_dx1);
        std.mem.swap(Tensor, &cur_dy2, &next_dx2);
    }

    try mergeFrom(core, &cur_dy1, &cur_dy2, grad_input_out);
}

fn layerGPUCompatible(layer: *const LayerCore) bool {
    if (layer.clip_min != -5.0 or layer.clip_max != 5.0) return false;
    for (layer.s_bias.data) |v| if (v != 0.0) return false;
    for (layer.t_bias.data) |v| if (v != 0.0) return false;
    return true;
}

fn modelGPUCompatible(core: *const RSFCore) bool {
    if (comptime !accel.gpu_enabled) return false;
    if (core.num_layers != 1 or core.layers.len != 1) return false;
    return layerGPUCompatible(&core.layers[0]);
}

fn disableGPU(core: *RSFCore) void {
    core.gpu_available.store(0, .monotonic);
    if (core.gpu_accel) |*ga| {
        ga.deinit();
        core.gpu_accel = null;
    }
    if (core.f16_buf) |buf| {
        core.allocator.free(buf);
        core.f16_buf = null;
    }
    core.gpu_weight_version = 0;
}

fn ensureGPUInitialized(core: *RSFCore) !void {
    if (comptime !accel.gpu_enabled) return error.GPUUnsupportedConfiguration;
    if (!modelGPUCompatible(core)) return error.GPUUnsupportedConfiguration;
    if (core.gpu_accel == null) core.gpu_accel = accel.RSFAccelerator.init(core.dim) catch return error.NoGPUAvailable;
    if (core.f16_buf == null) {
        const dim_sq = try checkedMul(core.dim, core.dim);
        core.f16_buf = try core.allocator.alloc(f16, dim_sq);
    }
}

fn validateF16Convertible(data: []const f32) !void {
    const max_f16 = std.math.floatMax(f16);
    for (data) |v| {
        if (!std.math.isFinite(v)) return error.NonFinite;
        if ((if (v >= 0) v else -v) > max_f16) return error.NumericFailure;
    }
}

fn syncAllLayersGPU(core: *RSFCore) !void {
    try ensureGPUInitialized(core);
    if (core.gpu_accel) |*ga| {
        const dim_sq = try checkedMul(core.dim, core.dim);
        const f16_buf = core.f16_buf orelse return error.GPUSyncFailed;
        if (f16_buf.len < dim_sq) return error.DataLengthMismatch;

        const layer = &core.layers[0];
        try validateTensor2DShape(&layer.s_weight, core.dim, core.dim);
        try validateTensor2DShape(&layer.t_weight, core.dim, core.dim);
        try ensureFiniteSlice(layer.s_weight.data);
        try ensureFiniteSlice(layer.t_weight.data);
        try validateF16Convertible(layer.s_weight.data);
        try validateF16Convertible(layer.t_weight.data);

        var i: usize = 0;
        while (i < dim_sq) : (i += 1) f16_buf[i] = @floatCast(layer.s_weight.data[i]);
        try ga.setWeightsS(f16_buf, core.dim, core.dim);

        i = 0;
        while (i < dim_sq) : (i += 1) f16_buf[i] = @floatCast(layer.t_weight.data[i]);
        try ga.setWeightsT(f16_buf, core.dim, core.dim);

        core.gpu_weight_version = core.cpu_weight_version;
        core.gpu_available.store(1, .monotonic);
    } else return error.NoGPUAvailable;
}

fn invalidateGPUForMismatch(core: *RSFCore) void {
    core.gpu_available.store(0, .monotonic);
}

pub const RSF = struct {
    id: u64 = 0,
    ctrl: ?*RSFCore = null,

    pub fn init(allocator: Allocator, dim: usize, num_layers: usize) !RSF {
        return initWithConfig(allocator, dim, num_layers, .{});
    }

    pub fn initWithConfig(allocator: Allocator, dim: usize, num_layers: usize, cfg: RSFConfig) !RSF {
        if (dim == 0) return error.InvalidDimension;
        if (num_layers == 0) return error.InvalidLayerCount;
        if (dim > cfg.max_dim or num_layers > cfg.max_layers) return error.TooLarge;
        if (!std.math.isFinite(cfg.clip_min) or !std.math.isFinite(cfg.clip_max)) return error.NonFinite;
        if (!(cfg.clip_min < cfg.clip_max)) return error.InvalidConfig;
        if (cfg.clip_max > 20.0 or cfg.clip_min < -20.0) return error.InvalidConfig;

        _ = try checkedMul(dim, dim);
        _ = try checkedMul(dim, 2);

        var core = try allocator.create(RSFCore);
        errdefer allocator.destroy(core);

        core.* = .{
            .allocator = allocator,
            .dim = dim,
            .num_layers = num_layers,
            .layers = try allocator.alloc(LayerCore, num_layers),
            .cfg = cfg,
            .rwlock = .{},
            .gpu_accel = null,
            .gpu_available = std.atomic.Value(u8).init(0),
            .gpu_weight_version = 0,
            .cpu_weight_version = 1,
            .f16_buf = null,
        };
        errdefer {
            if (core.gpu_accel) |*ga| {
                ga.deinit();
                core.gpu_accel = null;
            }
            if (core.f16_buf) |buf| {
                allocator.free(buf);
                core.f16_buf = null;
            }
            core.gpu_available.store(0, .monotonic);
        }
        errdefer allocator.free(core.layers);

        var initialized: usize = 0;
        errdefer {
            var j: usize = 0;
            while (j < initialized) : (j += 1) core.layers[j].deinitOwned();
        }

        var l: usize = 0;
        while (l < num_layers) : (l += 1) {
            const seed_base = try checkedMulU64(@as(u64, @intCast(l)), 10007);
            const layer_cfg = RSFLayerConfig{
                .clip_min = cfg.clip_min,
                .clip_max = cfg.clip_max,
                .seed_offset = seed_base,
                .grad_mean = cfg.grad_mean,
            };
            core.layers[l] = try LayerCore.initOwned(allocator, dim, layer_cfg);
            initialized += 1;
        }

        if (modelGPUCompatible(core)) {
            syncAllLayersGPU(core) catch disableGPU(core);
        }

        const id = try registerModelCore(core);
        return RSF{ .id = id, .ctrl = core };
    }

    pub fn deinit(self: *RSF) void {
        const id = self.id;
        if (id == 0) return;
        self.id = 0;
        self.ctrl = null;
        requestDestroyModelCore(id);
    }

    pub fn isGPUAvailable(self: *const RSF) bool {
        const id = self.id;
        const core = acquireModelCore(id) catch return false;
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        return core.gpu_available.load(.monotonic) != 0;
    }

    pub fn syncWeightsToGPU(self: *RSF) !void {
        const id = self.id;
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try syncAllLayersGPU(core);
    }

    pub fn zeroGradients(self: *RSF) !void {
        const id = self.id;
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        var i: usize = 0;
        while (i < core.num_layers) : (i += 1) core.layers[i].zeroGradients();
    }

    pub fn forwardCPU(self: *RSF, x: *Tensor) !void {
        const id = self.id;
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try forwardOnCore(core, x);
    }

    pub fn forward(self: *RSF, x: *Tensor) !void {
        const id = self.id;
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();

        try validateTensor2D(x);
        const dim2 = try checkedMul(core.dim, 2);
        if (x.shape.dims[1] != dim2) return error.ShapeMismatch;
        if (x.shape.dims[0] == 0) return error.InvalidBatchSize;

        if (comptime accel.gpu_enabled) {
            if (core.gpu_available.load(.monotonic) != 0 and core.gpu_weight_version == core.cpu_weight_version and core.gpu_accel != null and modelGPUCompatible(core)) {
                if (core.gpu_accel) |*ga| {
                    if (ga.forwardFromTensor(x, core.allocator)) |result| {
                        var gpu_result = result;
                        defer gpu_result.deinit();

                        if (gpu_result.shape.dims.len != 2 or gpu_result.shape.dims[0] != x.shape.dims[0] or gpu_result.shape.dims[1] != x.shape.dims[1] or gpu_result.data.len != x.data.len) {
                            invalidateGPUForMismatch(core);
                            try forwardOnCore(core, x);
                            return;
                        }
                        @memcpy(x.data, gpu_result.data);
                        return;
                    } else |_| invalidateGPUForMismatch(core);
                } else invalidateGPUForMismatch(core);
            }
        }
        try forwardOnCore(core, x);
    }

    pub fn inverse(self: *RSF, y: *Tensor) !void {
        const id = self.id;
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();
        try inverseOnCore(core, y);
    }

    pub fn backward(self: *RSF, grad_output: *const Tensor, input: *const Tensor, grad_input_out: *Tensor) !void {
        const id = self.id;
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();
        try backwardOnCore(core, grad_output, input, grad_input_out);
    }

    pub fn save(self: *const RSF, path: []const u8) !void {
        const id = self.id;
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lockShared();
        defer core.rwlock.unlockShared();

        const parent_path = if (std.fs.path.dirname(path)) |p| p else ".";
        const base_name = std.fs.path.basename(path);
        var parent_dir = if (std.fs.path.isAbsolute(parent_path)) try std.fs.openDirAbsolute(parent_path, .{}) else try std.fs.cwd().openDir(parent_path, .{});
        defer parent_dir.close();

        const temp = try createUniqueTempFile(&parent_dir, core.allocator, base_name);
        defer core.allocator.free(temp.tmp_name);

        var file = temp.file;
        var file_open = true;
        var tmp_exists = true;
        errdefer {
            if (file_open) file.close();
            if (tmp_exists) parent_dir.deleteFile(temp.tmp_name) catch {};
        }

        var buffered = std.io.bufferedWriter(file.writer());
        const w = buffered.writer();
        var hasher = std.hash.Crc32.init();

        try w.writeAll("RSF0");
        hasher.update("RSF0");
        try w.writeInt(u32, SAVE_VERSION, .little);
        crcUpdateU32LE(&hasher, SAVE_VERSION);
        try w.writeInt(u64, @intCast(core.num_layers), .Little);
        crcUpdateU64LE(&hasher, @intCast(core.num_layers));
        try w.writeInt(u64, @intCast(core.dim), .Little);
        crcUpdateU64LE(&hasher, @intCast(core.dim));

        const clip_min_bits = @as(u32, @bitCast(core.cfg.clip_min));
        const clip_max_bits = @as(u32, @bitCast(core.cfg.clip_max));
        try w.writeInt(u32, clip_min_bits, .little);
        try w.writeInt(u32, clip_max_bits, .little);
        crcUpdateU32LE(&hasher, clip_min_bits);
        crcUpdateU32LE(&hasher, clip_max_bits);

        const gm_byte: u8 = if (core.cfg.grad_mean) 1 else 0;
        try w.writeByte(gm_byte);
        crcUpdateU8(&hasher, gm_byte);

        try w.writeInt(u64, @intCast(core.cfg.max_dim), .Little);
        try w.writeInt(u64, @intCast(core.cfg.max_layers), .Little);
        crcUpdateU64LE(&hasher, @intCast(core.cfg.max_dim));
        crcUpdateU64LE(&hasher, @intCast(core.cfg.max_layers));

        var i: usize = 0;
        while (i < core.num_layers) : (i += 1) {
            const layer = &core.layers[i];
            try validateTensor2DShape(&layer.s_weight, core.dim, core.dim);
            try validateTensor2DShape(&layer.t_weight, core.dim, core.dim);
            try validateTensor2DShape(&layer.s_bias, 1, core.dim);
            try validateTensor2DShape(&layer.t_bias, 1, core.dim);
            try ensureFiniteSlice(layer.s_weight.data);
            try ensureFiniteSlice(layer.t_weight.data);
            try ensureFiniteSlice(layer.s_bias.data);
            try ensureFiniteSlice(layer.t_bias.data);
            const lmin_bits = @as(u32, @bitCast(layer.clip_min));
            const lmax_bits = @as(u32, @bitCast(layer.clip_max));
            try w.writeInt(u32, lmin_bits, .little);
            try w.writeInt(u32, lmax_bits, .little);
            crcUpdateU32LE(&hasher, lmin_bits);
            crcUpdateU32LE(&hasher, lmax_bits);

            const lgm: u8 = if (layer.grad_mean) 1 else 0;
            try w.writeByte(lgm);
            crcUpdateU8(&hasher, lgm);

            try writeTensorDataVersion4(w, &hasher, &layer.s_weight);
            try writeTensorDataVersion4(w, &hasher, &layer.t_weight);
            try writeTensorDataVersion4(w, &hasher, &layer.s_bias);
            try writeTensorDataVersion4(w, &hasher, &layer.t_bias);
        }

        try w.writeInt(u32, hasher.final(), .Little);
        try buffered.flush();
        try file.sync();
        file.close();
        file_open = false;

        try parent_dir.rename(temp.tmp_name, base_name);
        tmp_exists = false;
        try parent_dir.sync();
    }

    pub fn load(allocator: Allocator, path: []const u8) !RSF {
        return loadWithConfig(allocator, path, null);
    }

    pub fn loadWithConfig(allocator: Allocator, path: []const u8, policy: ?RSFConfig) !RSF {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        var r = file.reader();
        var magic: [4]u8 = undefined;
        try r.readNoEof(&magic);
        if (!std.mem.eql(u8, &magic, "RSF0")) return error.BadFileFormat;

        const version = try r.readInt(u32, .little);
        if (version != 1 and version != 2 and version != 3 and version != SAVE_VERSION) return error.UnsupportedVersion;

        const num_layers_u64 = try r.readInt(u64, .little);
        const dim_u64 = try r.readInt(u64, .little);
        if (num_layers_u64 == 0) return error.InvalidLayerCount;
        if (dim_u64 == 0) return error.InvalidDimension;

        const default_max_dim_u64: u64 = 1 << 20;
        const default_max_layers_u64: u64 = 1 << 20;
        const effective_max_dim: u64 = if (policy) |p| @intCast(p.max_dim) else default_max_dim_u64;
        const effective_max_layers: u64 = if (policy) |p| @intCast(p.max_layers) else default_max_layers_u64;

        if (policy != null or version == 1) {
            if (num_layers_u64 > effective_max_layers or dim_u64 > effective_max_dim) return error.TooLarge;
        }

        const num_layers = try checkedCastU64ToUsize(num_layers_u64);
        const dim = try checkedCastU64ToUsize(dim_u64);
        _ = try checkedMul(dim, dim);
        _ = try checkedMul(dim, 2);

        var hasher = std.hash.Crc32.init();
        if (version == SAVE_VERSION) {
            hasher.update("RSF0");
            crcUpdateU32LE(&hasher, version);
            crcUpdateU64LE(&hasher, num_layers_u64);
            crcUpdateU64LE(&hasher, dim_u64);
        }

        const clip_min: f32 = @bitCast(try r.readInt(u32, .little));
        const clip_max: f32 = @bitCast(try r.readInt(u32, .little));
        const grad_mean = (try r.readByte()) != 0;

        if (!std.math.isFinite(clip_min) or !std.math.isFinite(clip_max) or !(clip_min < clip_max)) return error.InvalidConfig;
        if (clip_max > 20.0 or clip_min < -20.0) return error.InvalidConfig;

        const clip_min_bits = @as(u32, @bitCast(clip_min));
        const clip_max_bits = @as(u32, @bitCast(clip_max));

        if (version == 2 or version == 3 or version == SAVE_VERSION) {
            crcUpdateU32LE(&hasher, clip_min_bits);
            crcUpdateU32LE(&hasher, clip_max_bits);
            crcUpdateU8(&hasher, if (grad_mean) @as(u8, 1) else @as(u8, 0));
        }

        var load_max_dim: usize = if (policy) |p| p.max_dim else (1 << 20);
        var load_max_layers: usize = if (policy) |p| p.max_layers else (1 << 20);

        if (version >= 2) {
            const saved_max_dim = try r.readInt(u64, .little);
            const saved_max_layers = try r.readInt(u64, .little);
            if (version == SAVE_VERSION) {
                crcUpdateU64LE(&hasher, saved_max_dim);
                crcUpdateU64LE(&hasher, saved_max_layers);
            }
            if (policy == null) {
                load_max_dim = @max(load_max_dim, @as(usize, @intCast(@min(saved_max_dim, @as(u64, std.math.maxInt(usize))))));
                load_max_layers = @max(load_max_layers, @as(usize, @intCast(@min(saved_max_layers, @as(u64, std.math.maxInt(usize))))));
            }
        }

        if (policy == null and version >= 2) {
            if (num_layers_u64 > @as(u64, @intCast(load_max_layers)) or dim_u64 > @as(u64, @intCast(load_max_dim))) return error.TooLarge;
        }

        var rsf = try RSF.initWithConfig(allocator, dim, num_layers, .{
            .clip_min = clip_min,
            .clip_max = clip_max,
            .grad_mean = grad_mean,
            .max_dim = load_max_dim,
            .max_layers = load_max_layers,
        });
        errdefer rsf.deinit();

        const id = rsf.id;
        const core = try acquireModelCore(id);
        defer releaseModelCore(id);
        core.rwlock.lock();
        defer core.rwlock.unlock();

        var i: usize = 0;
        while (i < core.num_layers) : (i += 1) {
            const layer_clip_min: f32 = @bitCast(try r.readInt(u32, .little));
            const layer_clip_max: f32 = @bitCast(try r.readInt(u32, .little));
            const layer_grad_mean = (try r.readByte()) != 0;

            if (!std.math.isFinite(layer_clip_min) or !std.math.isFinite(layer_clip_max) or !(layer_clip_min < layer_clip_max)) return error.InvalidConfig;
            if (layer_clip_max > 20.0 or layer_clip_min < -20.0) return error.InvalidConfig;

            if (version == 2 or version == 3 or version == SAVE_VERSION) {
                crcUpdateU32LE(&hasher, @as(u32, @bitCast(layer_clip_min)));
                crcUpdateU32LE(&hasher, @as(u32, @bitCast(layer_clip_max)));
                crcUpdateU8(&hasher, if (layer_grad_mean) @as(u8, 1) else @as(u8, 0));
            }

            var s_w_new = try readTensorData(allocator, r);
            errdefer s_w_new.deinit();
            var t_w_new = try readTensorData(allocator, r);
            errdefer t_w_new.deinit();
            var s_b_new = try readTensorData(allocator, r);
            errdefer s_b_new.deinit();
            var t_b_new = try readTensorData(allocator, r);
            errdefer t_b_new.deinit();

            try validateTensor2DShape(&s_w_new, dim, dim);
            try validateTensor2DShape(&t_w_new, dim, dim);
            try validateTensor2DShape(&s_b_new, 1, dim);
            try validateTensor2DShape(&t_b_new, 1, dim);
            try ensureFiniteSlice(s_w_new.data);
            try ensureFiniteSlice(t_w_new.data);
            try ensureFiniteSlice(s_b_new.data);
            try ensureFiniteSlice(t_b_new.data);

            if (version == 2 or version == 3) {
                hashTensorDataOld(&hasher, s_w_new.data);
                hashTensorDataOld(&hasher, t_w_new.data);
                hashTensorDataOld(&hasher, s_b_new.data);
                hashTensorDataOld(&hasher, t_b_new.data);
            } else if (version == SAVE_VERSION) {
                hashTensorDataVersion4(&hasher, &s_w_new);
                hashTensorDataVersion4(&hasher, &t_w_new);
                hashTensorDataVersion4(&hasher, &s_b_new);
                hashTensorDataVersion4(&hasher, &t_b_new);
            }

            var layer = &core.layers[i];
            layer.s_weight.deinit();
            layer.t_weight.deinit();
            layer.s_bias.deinit();
            layer.t_bias.deinit();
            layer.s_weight = s_w_new;
            layer.t_weight = t_w_new;
            layer.s_bias = s_b_new;
            layer.t_bias = t_b_new;
            layer.clip_min = layer_clip_min;
            layer.clip_max = layer_clip_max;
            layer.grad_mean = layer_grad_mean;

            if (layer.s_weight_grad) |*g| zeroTensor(g);
            if (layer.t_weight_grad) |*g| zeroTensor(g);
            if (layer.s_bias_grad) |*g| zeroTensor(g);
            if (layer.t_bias_grad) |*g| zeroTensor(g);
        }

        if (version >= 2) {
            if (try r.readInt(u32, .little) != hasher.final()) return error.ChecksumMismatch;
        }

        var eof_buf: [1]u8 = undefined;
        if ((try r.read(&eof_buf)) != 0) return error.TrailingData;

        core.cfg.clip_min = clip_min;
        core.cfg.clip_max = clip_max;
        core.cfg.grad_mean = grad_mean;
        core.cfg.max_dim = load_max_dim;
        core.cfg.max_layers = load_max_layers;
        core.cpu_weight_version = std.math.add(u64, core.cpu_weight_version, 1) catch return error.Overflow;

        if (modelGPUCompatible(core)) {
            syncAllLayersGPU(core) catch |err| {
                disableGPU(core);
                return err;
            };
        } else disableGPU(core);

        return rsf;
    }
};

fn crcUpdateU32LE(hasher: *std.hash.Crc32, v: u32) void {
    var le = std.mem.nativeToLittle(u32, v);
    hasher.update(std.mem.asBytes(&le));
}

fn crcUpdateU64LE(hasher: *std.hash.Crc32, v: u64) void {
    var le = std.mem.nativeToLittle(u64, v);
    hasher.update(std.mem.asBytes(&le));
}

fn crcUpdateU8(hasher: *std.hash.Crc32, v: u8) void {
    hasher.update(&.{v});
}

fn writeTensorDataVersion4(w: anytype, hasher: *std.hash.Crc32, t: *const Tensor) !void {
    try validateTensor2D(t);
    try ensureFiniteSlice(t.data);
    const rows = t.shape.dims[0];
    const cols = t.shape.dims[1];
    try w.writeInt(u64, 2, .little);
    crcUpdateU64LE(hasher, 2);
    try w.writeInt(u64, @intCast(rows), .Little);
    try w.writeInt(u64, @intCast(cols), .Little);
    crcUpdateU64LE(hasher, @intCast(rows));
    crcUpdateU64LE(hasher, @intCast(cols));
    for (t.data) |v| {
        const bits = @as(u32, @bitCast(v));
        try w.writeInt(u32, bits, .little);
        crcUpdateU32LE(hasher, bits);
    }
}

fn hashTensorDataVersion4(hasher: *std.hash.Crc32, t: *const Tensor) void {
    crcUpdateU64LE(hasher, 2);
    crcUpdateU64LE(hasher, @intCast(t.shape.dims[0]));
    crcUpdateU64LE(hasher, @intCast(t.shape.dims[1]));
    for (t.data) |v| crcUpdateU32LE(hasher, @as(u32, @bitCast(v)));
}

fn hashTensorDataOld(hasher: *std.hash.Crc32, data: []const f32) void {
    for (data) |v| crcUpdateU32LE(hasher, @as(u32, @bitCast(v)));
}

fn readTensorData(allocator: Allocator, r: anytype) !Tensor {
    if ((try r.readInt(u64, .little)) != 2) return error.BadFileFormat;
    const d0 = try checkedCastU64ToUsize(try r.readInt(u64, .little));
    const d1 = try checkedCastU64ToUsize(try r.readInt(u64, .little));
    var t = try Tensor.init(allocator, &.{ d0, d1 });
    errdefer t.deinit();
    const expected = try checkedMul(d0, d1);
    var i: usize = 0;
    while (i < expected) : (i += 1) t.data[i] = @bitCast(try r.readInt(u32, .little));
    return t;
}

fn checkedCastU64ToUsize(v: u64) !usize {
    if (v > std.math.maxInt(usize)) return error.TooLarge;
    return @intCast(v);
}

const TempFile = struct {
    file: std.fs.File,
    tmp_name: []u8,
};

fn hexEncodeLower(dst: []u8, src: []const u8) []u8 {
    const alphabet = "0123456789abcdef";
    var i: usize = 0;
    while (i < src.len) : (i += 1) {
        dst[i * 2] = alphabet[(src[i] >> 4) & 0x0f];
        dst[i * 2 + 1] = alphabet[src[i] & 0x0f];
    }
    return dst[0 .. src.len * 2];
}

fn createUniqueTempFile(dir: *std.fs.Dir, allocator: Allocator, base_name: []const u8) !TempFile {
    var attempt: usize = 0;
    while (attempt < 64) : (attempt += 1) {
        var rnd: [16]u8 = undefined;
        std.crypto.random.bytes(&rnd);
        var hex_buf: [32]u8 = undefined;
        const hex = hexEncodeLower(&hex_buf, &rnd);
        const tmp_name = try std.fmt.allocPrint(allocator, ".{s}.tmp.{s}", .{ base_name, hex });
        errdefer allocator.free(tmp_name);
        const file = dir.createFile(tmp_name, .{ .exclusive = true, .mode = 0o600 }) catch |e| switch (e) {
            error.PathAlreadyExists => {
                allocator.free(tmp_name);
                continue;
            },
            else => return e,
        };
        return .{ .file = file, .tmp_name = tmp_name };
    }
    return error.TempFileCollision;
}

