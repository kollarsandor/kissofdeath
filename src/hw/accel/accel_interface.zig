const std = @import("std");
const futhark = @import("futhark_bindings.zig");
const core_tensor = @import("../../core/tensor.zig");

pub const AccelError = error{
    FutharkInitFailed,
    FutharkSyncFailed,
    FutharkForwardFailed,
    FutharkGradFailed,
    FutharkTrainingStepFailed,
    NullPointer,
    InvalidDimensions,
    AllocationFailed,
};

pub const FutharkContext = struct {
    ctx: ?*futhark.struct_futhark_context,
    cfg: ?*futhark.struct_futhark_context_config,

    const Self = @This();

    pub fn init() AccelError!Self {
        const cfg = futhark.futhark_context_config_new() orelse return AccelError.FutharkInitFailed;
        const ctx = futhark.futhark_context_new(cfg) orelse {
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkInitFailed;
        };
        return Self{ .ctx = ctx, .cfg = cfg };
    }

    pub fn deinit(self: *Self) void {
        if (self.ctx) |ctx| futhark.futhark_context_free(ctx);
        if (self.cfg) |cfg| futhark.futhark_context_config_free(cfg);
        self.ctx = null;
        self.cfg = null;
    }

    pub fn sync(self: *Self) AccelError!void {
        if (futhark.futhark_context_sync(self.ctx) != 0) return AccelError.FutharkSyncFailed;
    }
};

pub const FutharkArray1DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_1d,
    len: usize,

    const Self = @This();

    pub fn newZeros(ctx: *FutharkContext, len: usize) AccelError!Self {
        const zeros = std.heap.page_allocator.alloc(u16, len) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f16_1d(ctx.ctx, zeros.ptr, @intCast(len)) orelse return AccelError.NullPointer;
        return Self{ .arr = arr, .len = len };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| _ = futhark.futhark_free_f16_1d(ctx.ctx, arr);
        self.arr = null;
    }
};

pub const FutharkArray2DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize) AccelError!Self {
        const total = rows * cols;
        const zeros = std.heap.page_allocator.alloc(u16, total) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f16_2d(ctx.ctx, zeros.ptr, @intCast(rows), @intCast(cols)) orelse return AccelError.NullPointer;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| _ = futhark.futhark_free_f16_2d(ctx.ctx, arr);
        self.arr = null;
    }
};

pub const FutharkArray3DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_3d,
    dim0: usize,
    dim1: usize,
    dim2: usize,

    const Self = @This();

    pub fn newZeros(ctx: *FutharkContext, dim0: usize, dim1: usize, dim2: usize) AccelError!Self {
        const total = dim0 * dim1 * dim2;
        const zeros = std.heap.page_allocator.alloc(u16, total) catch return AccelError.AllocationFailed;
        defer std.heap.page_allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f16_3d(ctx.ctx, zeros.ptr, @intCast(dim0), @intCast(dim1), @intCast(dim2)) orelse return AccelError.NullPointer;
        return Self{ .arr = arr, .dim0 = dim0, .dim1 = dim1, .dim2 = dim2 };
    }

    pub fn fromSlice(ctx: *FutharkContext, data: []const u16, dim0: usize, dim1: usize, dim2: usize) AccelError!Self {
        const arr = futhark.futhark_new_f16_3d(ctx.ctx, data.ptr, @intCast(dim0), @intCast(dim1), @intCast(dim2)) orelse return AccelError.NullPointer;
        return Self{ .arr = arr, .dim0 = dim0, .dim1 = dim1, .dim2 = dim2 };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| _ = futhark.futhark_free_f16_3d(ctx.ctx, arr);
        self.arr = null;
    }
};

pub const FutharkArray1DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_1d,
    len: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, t: *const core_tensor.Tensor) AccelError!Self {
        const len = t.shape.dims[0];
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, t.data.ptr, @intCast(len)) orelse return AccelError.NullPointer;
        return Self{ .arr = arr, .len = len };
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        const data = allocator.alloc(f32, self.len) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, data.ptr) != 0) {
            allocator.free(data);
            return AccelError.FutharkForwardFailed;
        }
        return core_tensor.Tensor{ .data = data, .shape = .{ .dims = &[_]usize{self.len} } };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| futhark.futhark_free_f32_1d(ctx.ctx, arr);
        self.arr = null;
    }
};

pub const RSFAccelerator = struct {
    ctx: FutharkContext,
    weights_s: FutharkArray2DF16,
    weights_t: FutharkArray2DF16,
    s_bias: FutharkArray1DF16,
    t_bias: FutharkArray1DF16,
    velocity_s: FutharkArray2DF16,
    velocity_t: FutharkArray2DF16,
    model_dim: usize,
    clip_min: f16,
    clip_max: f16,
    initialized: bool,

    const Self = @This();

    pub fn init(model_dim: usize) AccelError!Self {
        if (model_dim == 0) return AccelError.InvalidDimensions;
        const half = model_dim / 2;
        if (half == 0) return AccelError.InvalidDimensions;

        var ctx = try FutharkContext.init();
        errdefer ctx.deinit();

        var weights_s = try FutharkArray2DF16.newZeros(&ctx, half, half);
        errdefer weights_s.free(&ctx);

        var weights_t = try FutharkArray2DF16.newZeros(&ctx, half, half);
        errdefer weights_t.free(&ctx);

        var s_bias = try FutharkArray1DF16.newZeros(&ctx, half);
        errdefer s_bias.free(&ctx);

        var t_bias = try FutharkArray1DF16.newZeros(&ctx, half);
        errdefer t_bias.free(&ctx);

        var velocity_s = try FutharkArray2DF16.newZeros(&ctx, half, half);
        errdefer velocity_s.free(&ctx);

        var velocity_t = try FutharkArray2DF16.newZeros(&ctx, half, half);
        errdefer velocity_t.free(&ctx);

        return Self{
            .ctx = ctx,
            .weights_s = weights_s,
            .weights_t = weights_t,
            .s_bias = s_bias,
            .t_bias = t_bias,
            .velocity_s = velocity_s,
            .velocity_t = velocity_t,
            .model_dim = model_dim,
            .clip_min = -5.0,
            .clip_max = 5.0,
            .initialized = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        self.velocity_t.free(&self.ctx);
        self.velocity_s.free(&self.ctx);
        self.t_bias.free(&self.ctx);
        self.s_bias.free(&self.ctx);
        self.weights_t.free(&self.ctx);
        self.weights_s.free(&self.ctx);
        self.ctx.deinit();
        self.initialized = false;
    }

    pub fn forward(self: *Self, input: *FutharkArray2DF16) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (input.arr == null) return AccelError.NullPointer;
        if (self.weights_s.arr == null or self.weights_t.arr == null) return AccelError.NullPointer;
        if (self.s_bias.arr == null or self.t_bias.arr == null) return AccelError.NullPointer;

        var output: ?*futhark.struct_futhark_f16_2d = null;
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        const result = futhark.futhark_entry_rsf_forward(
            self.ctx.ctx,
            &output,
            input.arr,
            self.weights_s.arr,
            self.weights_t.arr,
            self.s_bias.arr,
            self.t_bias.arr,
            clip_min_bits,
            clip_max_bits,
        );

        if (result != 0) return AccelError.FutharkForwardFailed;
        if (output == null) return AccelError.NullPointer;

        return FutharkArray2DF16{
            .arr = output,
            .rows = input.rows,
            .cols = input.cols,
        };
    }

    pub fn trainingStep(
        self: *Self,
        inputs: *FutharkArray3DF16,
        targets: *FutharkArray3DF16,
        learning_rate: f16,
        momentum: f16,
    ) AccelError!f16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (inputs.arr == null or targets.arr == null) return AccelError.NullPointer;
        if (self.weights_s.arr == null or self.weights_t.arr == null) return AccelError.NullPointer;
        if (self.s_bias.arr == null or self.t_bias.arr == null) return AccelError.NullPointer;
        if (self.velocity_s.arr == null or self.velocity_t.arr == null) return AccelError.NullPointer;

        var new_ws: ?*futhark.struct_futhark_f16_2d = null;
        var new_wt: ?*futhark.struct_futhark_f16_2d = null;
        var new_sbs: ?*futhark.struct_futhark_f16_1d = null;
        var new_tbs: ?*futhark.struct_futhark_f16_1d = null;
        var new_vs: ?*futhark.struct_futhark_f16_2d = null;
        var new_vt: ?*futhark.struct_futhark_f16_2d = null;
        var loss: u16 = 0;

        const lr_bits: u16 = @bitCast(learning_rate);
        const momentum_bits: u16 = @bitCast(momentum);
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        const result = futhark.futhark_entry_training_step(
            self.ctx.ctx,
            &new_ws,
            &new_wt,
            &new_sbs,
            &new_tbs,
            &new_vs,
            &new_vt,
            &loss,
            inputs.arr,
            targets.arr,
            self.weights_s.arr,
            self.weights_t.arr,
            self.s_bias.arr,
            self.t_bias.arr,
            self.velocity_s.arr,
            self.velocity_t.arr,
            lr_bits,
            momentum_bits,
            clip_min_bits,
            clip_max_bits,
        );

        if (result != 0) return AccelError.FutharkTrainingStepFailed;
        if (new_ws == null or new_wt == null or new_vs == null or new_vt == null) return AccelError.NullPointer;
        if (new_sbs == null or new_tbs == null) return AccelError.NullPointer;

        const old_ws = self.weights_s.arr;
        const old_wt = self.weights_t.arr;
        const old_sbs = self.s_bias.arr;
        const old_tbs = self.t_bias.arr;
        const old_vs = self.velocity_s.arr;
        const old_vt = self.velocity_t.arr;

        self.weights_s.arr = new_ws;
        self.weights_t.arr = new_wt;
        self.s_bias.arr = new_sbs;
        self.t_bias.arr = new_tbs;
        self.velocity_s.arr = new_vs;
        self.velocity_t.arr = new_vt;

        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_ws);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_wt);
        _ = futhark.futhark_free_f16_1d(self.ctx.ctx, old_sbs);
        _ = futhark.futhark_free_f16_1d(self.ctx.ctx, old_tbs);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_vs);
        _ = futhark.futhark_free_f16_2d(self.ctx.ctx, old_vt);

        const loss_f16: f16 = @bitCast(loss);
        return loss_f16;
    }

    pub fn grad(
        self: *Self,
        y_out: *FutharkArray2DF16,
        dy: *FutharkArray2DF16,
    ) AccelError!struct { grad_s: FutharkArray2DF16, grad_t: FutharkArray2DF16, grad_bs: FutharkArray1DF16, grad_bt: FutharkArray1DF16 } {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (y_out.arr == null or dy.arr == null) return AccelError.NullPointer;
        if (self.weights_s.arr == null or self.weights_t.arr == null) return AccelError.NullPointer;
        if (self.s_bias.arr == null or self.t_bias.arr == null) return AccelError.NullPointer;

        const half = self.model_dim / 2;
        var out_grad_s: ?*futhark.struct_futhark_f16_2d = null;
        var out_grad_t: ?*futhark.struct_futhark_f16_2d = null;
        var out_grad_bs: ?*futhark.struct_futhark_f16_1d = null;
        var out_grad_bt: ?*futhark.struct_futhark_f16_1d = null;
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        const result = futhark.futhark_entry_rsf_grad(
            self.ctx.ctx,
            &out_grad_s,
            &out_grad_t,
            &out_grad_bs,
            &out_grad_bt,
            y_out.arr,
            dy.arr,
            self.weights_s.arr,
            self.weights_t.arr,
            self.s_bias.arr,
            self.t_bias.arr,
            clip_min_bits,
            clip_max_bits,
        );

        if (result != 0) return AccelError.FutharkGradFailed;
        if (out_grad_s == null or out_grad_t == null or out_grad_bs == null or out_grad_bt == null) return AccelError.NullPointer;

        return .{
            .grad_s = FutharkArray2DF16{ .arr = out_grad_s, .rows = half, .cols = half },
            .grad_t = FutharkArray2DF16{ .arr = out_grad_t, .rows = half, .cols = half },
            .grad_bs = FutharkArray1DF16{ .arr = out_grad_bs, .len = half },
            .grad_bt = FutharkArray1DF16{ .arr = out_grad_bt, .len = half },
        };
    }
};

pub const GPUOps = struct {
    ctx: FutharkContext,

    const Self = @This();

    pub fn init() AccelError!Self {
        const ctx = try FutharkContext.init();
        return Self{ .ctx = ctx };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.deinit();
    }

    pub fn matmul(self: *Self, a: *const core_tensor.Tensor, b: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        const rows_a = a.shape.dims[0];
        const cols_a = a.shape.dims[1];
        const cols_b = b.shape.dims[1];
        const fa = futhark.futhark_new_f32_2d(self.ctx.ctx, a.data.ptr, @intCast(rows_a), @intCast(cols_a)) orelse return AccelError.NullPointer;
        defer _ = futhark.futhark_free_f32_2d(self.ctx.ctx, fa);
        const fb = futhark.futhark_new_f32_2d(self.ctx.ctx, b.data.ptr, @intCast(cols_a), @intCast(cols_b)) orelse return AccelError.NullPointer;
        defer _ = futhark.futhark_free_f32_2d(self.ctx.ctx, fb);
        var out_arr: ?*futhark.struct_futhark_f32_2d = null;
        if (futhark.futhark_entry_matmul(self.ctx.ctx, &out_arr, fa, fb) != 0) return AccelError.FutharkForwardFailed;
        if (out_arr == null) return AccelError.NullPointer;
        defer _ = futhark.futhark_free_f32_2d(self.ctx.ctx, out_arr);
        const out_len = rows_a * cols_b;
        const data = allocator.alloc(f32, out_len) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_2d(self.ctx.ctx, out_arr, data.ptr) != 0) {
            allocator.free(data);
            return AccelError.FutharkForwardFailed;
        }
        return core_tensor.Tensor{ .data = data, .shape = .{ .dims = &[_]usize{ rows_a, cols_b } } };
    }
};
