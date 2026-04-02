const std = @import("std");
const mem = std.mem;
const Allocator = mem.Allocator;
const testing = std.testing;
const core_tensor = @import("../core/tensor.zig");
const core_memory = @import("../core/memory.zig");

pub const MGT = struct {
    token_to_id: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    prefixes: std.StringHashMap(u32),
    suffixes: std.StringHashMap(u32),
    roots: std.StringHashMap(u32),
    bpe_pairs: std.StringHashMap(BPEMerge),
    anchors: std.StringHashMap(u64),
    allocated_strings: std.ArrayList([]u8),
    allocator: Allocator,
    next_token_id: u32,

    const BPEMerge = struct {
        token_id: u32,
        priority: u32,
    };

    const SPECIAL_TOKENS = struct {
        const PAD: u32 = 0;
        const UNK: u32 = 1;
        const BOS: u32 = 2;
        const EOS: u32 = 3;
    };

    const SPECIAL_TOKEN_STRINGS = [_][]const u8{ "[PAD]", "[UNK]", "[BOS]", "[EOS]" };

    pub fn init(allocator: Allocator, vocab: []const []const u8, anchor_list: []const []const u8) !MGT {
        const token_to_id = std.StringHashMap(u32).init(allocator);
        const id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
        const prefixes_map = std.StringHashMap(u32).init(allocator);
        const suffixes_map = std.StringHashMap(u32).init(allocator);
        const roots_map = std.StringHashMap(u32).init(allocator);
        const bpe_pairs_map = std.StringHashMap(BPEMerge).init(allocator);
        const anch_map = std.StringHashMap(u64).init(allocator);
        const allocated = std.ArrayList([]u8).init(allocator);

        var mgt = MGT{
            .token_to_id = token_to_id,
            .id_to_token = id_to_token,
            .prefixes = prefixes_map,
            .suffixes = suffixes_map,
            .roots = roots_map,
            .bpe_pairs = bpe_pairs_map,
            .anchors = anch_map,
            .allocated_strings = allocated,
            .allocator = allocator,
            .next_token_id = 0,
        };
        errdefer mgt.deinit();

        for (SPECIAL_TOKEN_STRINGS) |tok| {
            _ = try mgt.addToken(tok);
        }

        for (vocab) |word| {
            if (!mgt.token_to_id.contains(word)) {
                _ = try mgt.addToken(word);
            }
        }

        try mgt.initMorphemes();

        for (anchor_list) |anch| {
            const tid = if (mgt.token_to_id.get(anch)) |t| t else try mgt.addToken(anch);
            const h: u64 = @intCast(tid);
            const anch_key = mgt.id_to_token.get(tid).?;
            try mgt.anchors.put(anch_key, h);
        }

        return mgt;
    }

    pub fn initWithArena(arena: *core_memory.ArenaAllocator, vocab: []const []const u8, anchors_list: []const []const u8) !MGT {
        return init(arena.allocator(), vocab, anchors_list);
    }

    pub fn initWithPool(pool: *core_memory.PoolAllocator, vocab: []const []const u8, anchors_list: []const []const u8) !MGT {
        return init(pool.allocator(), vocab, anchors_list);
    }

    pub fn initWithBuddy(buddy: *core_memory.BuddyAllocator, vocab: []const []const u8, anchors_list: []const []const u8) !MGT {
        return init(buddy.allocator(), vocab, anchors_list);
    }

    fn initMorphemes(self: *MGT) !void {
        const prefix_list = [_][]const u8{
            "un",    "re",     "pre",    "dis",    "mis",     "over",   "under",  "out",
            "sub",   "inter",  "fore",   "de",     "trans",   "super",  "semi",   "anti",
            "mid",   "non",    "ex",     "post",   "pro",     "co",     "en",     "em",
            "meg",   "el",     "fel",    "le",     "be",      "ki",     "r\xc3\xa1", "\xc3\xa1t",
            "sz\xc3\xa9t", "vissza", "ide",   "oda",    "al\xc3\xa1", "f\xc3\xb6l\xc3\xa9",
            "k\xc3\xb6z\xc3\xa9", "egy",   "\xc3\xb6ssze", "tul",   "hozz\xc3\xa1", "k\xc3\xb6r\xc3\xbcl",
            "alig",  "\xc3\xa9ppen", "majd",  "csak",   "is",      "leg",    "legesleg",
        };

        for (prefix_list) |prefix| {
            const id = if (self.token_to_id.get(prefix)) |existing_id| existing_id else try self.addToken(prefix);
            const key = self.id_to_token.get(id).?;
            try self.prefixes.put(key, id);
        }

        const suffix_list = [_][]const u8{
            "ing",    "ed",     "er",     "est",    "ly",     "tion",   "sion",   "ness",
            "ment",   "ful",    "less",   "ous",    "ive",    "able",   "ible",   "al",
            "ial",    "y",      "s",      "es",     "en",     "ize",    "ise",    "ate",
            "s\xc3\xa1g", "s\xc3\xa9g", "s\xc3\xa1g\xc3\xba", "s\xc3\xa9g\xc5\xb1",
            "\xc3\xa9", "je",    "ja",    "ban",    "ben",
            "ba",     "be",     "b\xc3\xb3l", "b\xc5\x91l", "hoz",    "hez",    "h\xc3\xb6z",
            "t\xc3\xb3l", "t\xc5\x91l", "nak",   "nek",    "val",    "vel",
            "\xc3\xa9rt", "ul",    "\xc3\xbcl", "k\xc3\xa9nt", "\xc3\xa1n",
            "\xc3\xa9n", "ig",    "at",     "et",     "tat",    "tet",    "ott",    "ett",
            "atlan",  "etlen",  "talan",  "telen",  "\xc3\xa1l", "\xc3\xa9l",
            "oz",     "ez",     "\xc3\xb6d", "gyet",  "get",
            "j",      "unk",    "jatok",  "j\xc3\xa1tok", "i",      "ni",     "nk\xc3\xa9nt",
            "kor",    "ra",     "re",
        };

        for (suffix_list) |suffix| {
            const id = if (self.token_to_id.get(suffix)) |existing_id| existing_id else try self.addToken(suffix);
            const key = self.id_to_token.get(id).?;
            try self.suffixes.put(key, id);
        }
    }

    pub fn deinit(self: *MGT) void {
        self.token_to_id.deinit();
        self.id_to_token.deinit();
        self.prefixes.deinit();
        self.suffixes.deinit();
        self.roots.deinit();
        self.bpe_pairs.deinit();
        self.anchors.deinit();
        for (self.allocated_strings.items) |str| {
            self.allocator.free(str);
        }
        self.allocated_strings.deinit();
    }

    fn isWhitespace(c: u8) bool {
        return c == ' ' or c == '\n' or c == '\t' or c == '\r';
    }

    fn isPunctuation(c: u8) bool {
        return c == '.' or c == ',' or c == '!' or c == '?' or c == ';' or
            c == ':' or c == '"' or c == '\'' or c == '(' or c == ')' or
            c == '{' or c == '}';
    }

    fn isSpecialTokenStart(text: []const u8, pos: usize) bool {
        if (pos >= text.len or text[pos] != '[') return false;
        for (SPECIAL_TOKEN_STRINGS) |special| {
            if (pos + special.len <= text.len and mem.eql(u8, text[pos .. pos + special.len], special)) {
                return true;
            }
        }
        return false;
    }

    fn getSpecialTokenLen(text: []const u8, pos: usize) ?usize {
        if (pos >= text.len or text[pos] != '[') return null;
        for (SPECIAL_TOKEN_STRINGS) |special| {
            if (pos + special.len <= text.len and mem.eql(u8, text[pos .. pos + special.len], special)) {
                return special.len;
            }
        }
        return null;
    }

    fn utf8CharLen(first_byte: u8) u8 {
        if (first_byte & 0x80 == 0) return 1;
        if (first_byte & 0xE0 == 0xC0) return 2;
        if (first_byte & 0xF0 == 0xE0) return 3;
        if (first_byte & 0xF8 == 0xF0) return 4;
        return 1;
    }

    pub fn encode(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32)) !void {
        var i: usize = 0;
        while (i < text.len) {
            if (isSpecialTokenStart(text, i)) {
                if (getSpecialTokenLen(text, i)) |special_len| {
                    const special_token = text[i .. i + special_len];
                    if (self.token_to_id.get(special_token)) |tid| {
                        try out_tokens.append(tid);
                        i += special_len;
                        continue;
                    }
                }
            }

            if (isWhitespace(text[i])) {
                const ws_char = text[i .. i + 1];
                if (self.token_to_id.get(ws_char)) |ws_tid| {
                    try out_tokens.append(ws_tid);
                } else if (self.token_to_id.get(" ")) |space_tid| {
                    try out_tokens.append(space_tid);
                } else {
                    try out_tokens.append(SPECIAL_TOKENS.UNK);
                }
                i += 1;
                continue;
            }

            if (isPunctuation(text[i])) {
                const char_len: usize = 1;
                const punct_str = text[i .. i + char_len];
                if (self.token_to_id.get(punct_str)) |tid| {
                    try out_tokens.append(tid);
                } else {
                    try out_tokens.append(SPECIAL_TOKENS.UNK);
                }
                i += char_len;
                continue;
            }

            var word_end = i;
            while (word_end < text.len) {
                const c = text[word_end];
                if (isWhitespace(c) or isPunctuation(c)) break;
                if (isSpecialTokenStart(text, word_end) and word_end > i) break;
                const char_len = utf8CharLen(c);
                if (word_end + char_len > text.len) {
                    word_end += 1;
                    break;
                }
                word_end += char_len;
            }

            if (word_end == i) {
                i += 1;
                continue;
            }

            const word = text[i..word_end];

            if (self.token_to_id.get(word)) |tid| {
                try out_tokens.append(tid);
                i = word_end;
                continue;
            }

            if (word.len >= 4) {
                if (try self.morphDecompose(word, out_tokens)) |decomposed| {
                    if (decomposed) {
                        i = word_end;
                        continue;
                    }
                }
            }

            const subword_tokens = try self.subwordSplit(word);
            defer self.allocator.free(subword_tokens);
            for (subword_tokens) |tok| {
                try out_tokens.append(tok);
            }
            i = word_end;
        }
    }

    fn findLongestPrefix(self: *MGT, word: []const u8) ?struct { prefix: []const u8, len: usize } {
        var max_len: usize = 0;
        var best: ?[]const u8 = null;

        var check_len: usize = 1;
        while (check_len < word.len) : (check_len += 1) {
            const candidate = word[0..check_len];
            if (self.prefixes.contains(candidate)) {
                if (check_len > max_len) {
                    max_len = check_len;
                    best = candidate;
                }
            }
        }

        if (best) |p| {
            return .{ .prefix = p, .len = max_len };
        }
        return null;
    }

    fn findLongestSuffix(self: *MGT, word: []const u8) ?struct { suffix: []const u8, len: usize } {
        var max_len: usize = 0;
        var best: ?[]const u8 = null;

        var check_len: usize = 1;
        while (check_len < word.len) : (check_len += 1) {
            const candidate = word[word.len - check_len ..];
            if (self.suffixes.contains(candidate)) {
                if (check_len > max_len) {
                    max_len = check_len;
                    best = candidate;
                }
            }
        }

        if (best) |s| {
            return .{ .suffix = s, .len = max_len };
        }
        return null;
    }

    fn morphDecompose(self: *MGT, word: []const u8, out_tokens: *std.ArrayList(u32)) !?bool {
        if (word.len < 4) return null;

        const prefix_result = self.findLongestPrefix(word);
        const suffix_result = self.findLongestSuffix(word);

        const prefix_len = if (prefix_result) |p| p.len else 0;
        const suffix_len = if (suffix_result) |s| s.len else 0;

        if (prefix_len == 0 and suffix_len == 0) return null;

        const root_start = prefix_len;
        const root_end = word.len - suffix_len;

        if (root_end <= root_start or root_end - root_start < 2) return null;

        const root = word[root_start..root_end];

        var temp_tokens = std.ArrayList(u32).init(self.allocator);
        defer temp_tokens.deinit();

        if (prefix_result) |p| {
            if (self.token_to_id.get(p.prefix)) |tid| {
                try temp_tokens.append(tid);
            } else {
                return null;
            }
        }

        if (self.token_to_id.get(root)) |tid| {
            try temp_tokens.append(tid);
        } else if (self.roots.get(root)) |rid| {
            try temp_tokens.append(rid);
        } else {
            const root_id = try self.addToken(root);
            const root_str = self.id_to_token.get(root_id).?;
            try self.roots.put(root_str, root_id);
            try temp_tokens.append(root_id);
        }

        if (suffix_result) |s| {
            if (self.token_to_id.get(s.suffix)) |tid| {
                try temp_tokens.append(tid);
            } else {
                return null;
            }
        }

        try out_tokens.appendSlice(temp_tokens.items);
        return true;
    }

    fn addToken(self: *MGT, token: []const u8) !u32 {
        if (self.token_to_id.get(token)) |existing| {
            return existing;
        }

        const token_copy = try self.allocator.dupe(u8, token);
        errdefer self.allocator.free(token_copy);
        try self.allocated_strings.append(token_copy);
        errdefer _ = self.allocated_strings.pop();

        try self.token_to_id.put(token_copy, self.next_token_id);
        errdefer _ = self.token_to_id.remove(token_copy);

        try self.id_to_token.put(self.next_token_id, token_copy);

        const id = self.next_token_id;
        self.next_token_id += 1;
        return id;
    }

    fn encodeBPE(self: *MGT, text: []const u8) ![]u32 {
        if (text.len == 0) {
            const empty = try self.allocator.alloc(u32, 0);
            return empty;
        }

        var byte_tokens = std.ArrayList([]u8).init(self.allocator);
        defer {
            for (byte_tokens.items) |bt| {
                self.allocator.free(bt);
            }
            byte_tokens.deinit();
        }

        for (text) |byte| {
            const byte_str = try std.fmt.allocPrint(self.allocator, "<{x:0>2}>", .{byte});
            errdefer self.allocator.free(byte_str);
            try byte_tokens.append(byte_str);
        }

        while (byte_tokens.items.len > 1) {
            var best_priority: u32 = std.math.maxInt(u32);
            var best_idx: ?usize = null;
            var best_merge: ?BPEMerge = null;

            var idx: usize = 0;
            while (idx + 1 < byte_tokens.items.len) : (idx += 1) {
                const pair = try std.fmt.allocPrint(
                    self.allocator,
                    "{s}{s}",
                    .{ byte_tokens.items[idx], byte_tokens.items[idx + 1] },
                );
                defer self.allocator.free(pair);

                if (self.bpe_pairs.get(pair)) |merge| {
                    if (merge.priority < best_priority) {
                        best_priority = merge.priority;
                        best_idx = idx;
                        best_merge = merge;
                    }
                }
            }

            if (best_idx == null) break;

            const bi = best_idx.?;
            const merged = try std.fmt.allocPrint(
                self.allocator,
                "{s}{s}",
                .{ byte_tokens.items[bi], byte_tokens.items[bi + 1] },
            );

            self.allocator.free(byte_tokens.items[bi]);
            self.allocator.free(byte_tokens.items[bi + 1]);

            byte_tokens.items[bi] = merged;
            _ = byte_tokens.orderedRemove(bi + 1);
        }

        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();

        for (byte_tokens.items) |bt| {
            if (self.token_to_id.get(bt)) |tid| {
                try tokens.append(tid);
            } else {
                const tid = try self.addToken(bt);
                try tokens.append(tid);
            }
        }

        return try tokens.toOwnedSlice();
    }

    const PairKey = struct {
        byte1: u8,
        byte2: u8,
    };
    const MergeItem = struct { key: PairKey, freq: u32 };

    const LessThanContext = struct {
        fn lessThan(_: @This(), a: MergeItem, b: MergeItem) bool {
            return b.freq < a.freq;
        }
    };

    pub fn trainBPE(self: *MGT, corpus: []const []const u8, num_merges: u32) !void {
        var sequences = std.ArrayList(std.ArrayList([]u8)).init(self.allocator);
        defer {
            for (sequences.items) |*seq| {
                for (seq.items) |s| {
                    self.allocator.free(s);
                }
                seq.deinit();
            }
            sequences.deinit();
        }

        for (corpus) |text| {
            var seq = std.ArrayList([]u8).init(self.allocator);
            for (text) |byte| {
                const byte_str = try std.fmt.allocPrint(self.allocator, "<{x:0>2}>", .{byte});
                try seq.append(byte_str);
            }
            try sequences.append(seq);
        }

        var merge_count: u32 = 0;
        while (merge_count < num_merges) {
            var pair_freqs = std.StringHashMap(u32).init(self.allocator);
            defer pair_freqs.deinit();

            var pair_keys_alloc = std.ArrayList([]u8).init(self.allocator);
            defer {
                for (pair_keys_alloc.items) |pk| {
                    self.allocator.free(pk);
                }
                pair_keys_alloc.deinit();
            }

            for (sequences.items) |seq| {
                var si: usize = 0;
                while (si + 1 < seq.items.len) : (si += 1) {
                    const pair_str = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ seq.items[si], seq.items[si + 1] });
                    const gop = try pair_freqs.getOrPut(pair_str);
                    if (gop.found_existing) {
                        self.allocator.free(pair_str);
                        gop.value_ptr.* += 1;
                    } else {
                        try pair_keys_alloc.append(pair_str);
                        gop.value_ptr.* = 1;
                    }
                }
            }

            var best_pair: ?[]const u8 = null;
            var best_freq: u32 = 0;

            var freq_it = pair_freqs.iterator();
            while (freq_it.next()) |entry| {
                if (entry.value_ptr.* > best_freq) {
                    best_freq = entry.value_ptr.*;
                    best_pair = entry.key_ptr.*;
                }
            }

            if (best_pair == null or best_freq < 2) break;

            const bp = best_pair.?;

            const merge_token_id = try self.addToken(bp);
            const key_ptr = self.id_to_token.get(merge_token_id).?;
            try self.bpe_pairs.put(key_ptr, .{
                .token_id = merge_token_id,
                .priority = merge_count,
            });

            for (sequences.items) |*seq| {
                var wi: usize = 0;
                while (wi + 1 < seq.items.len) {
                    const combined = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ seq.items[wi], seq.items[wi + 1] });
                    if (mem.eql(u8, combined, bp)) {
                        self.allocator.free(seq.items[wi]);
                        self.allocator.free(seq.items[wi + 1]);
                        seq.items[wi] = combined;
                        _ = seq.orderedRemove(wi + 1);
                    } else {
                        self.allocator.free(combined);
                        wi += 1;
                    }
                }
            }

            merge_count += 1;
        }
    }

    pub fn decode(self: *MGT, tokens: []const u32, out_text: *std.ArrayList(u8)) !void {
        for (tokens) |tok| {
            if (self.id_to_token.get(tok)) |token_str| {
                if (tok == SPECIAL_TOKENS.PAD or tok == SPECIAL_TOKENS.BOS or tok == SPECIAL_TOKENS.EOS) {
                    continue;
                }
                if (tok == SPECIAL_TOKENS.UNK) {
                    try out_text.appendSlice("[UNK]");
                    continue;
                }
                if (isBpeHexToken(token_str)) {
                    try decodeBpeHexToken(token_str, out_text);
                } else {
                    try out_text.appendSlice(token_str);
                }
            } else {
                try out_text.appendSlice("[UNK]");
            }
        }
    }

    fn isBpeHexToken(token_str: []const u8) bool {
        if (token_str.len < 4) return false;
        if (token_str[0] != '<') return false;
        if (token_str[token_str.len - 1] != '>') return false;
        var pos: usize = 1;
        while (pos < token_str.len - 1) {
            if (token_str[pos] == '<') return false;
            if (token_str[pos] == '>') {
                if (pos + 1 < token_str.len - 1) {
                    if (token_str[pos + 1] != '<') return false;
                    pos += 2;
                    continue;
                } else if (pos + 1 == token_str.len - 1) {
                    return true;
                } else {
                    return false;
                }
            }
            pos += 1;
        }
        return true;
    }

    fn decodeBpeHexToken(token_str: []const u8, out_text: *std.ArrayList(u8)) !void {
        var pos: usize = 0;
        var all_decoded = true;
        var temp = std.ArrayList(u8).init(out_text.allocator);
        defer temp.deinit();

        while (pos < token_str.len) {
            if (token_str[pos] == '<') {
                const close = mem.indexOfScalarPos(u8, token_str, pos + 1, '>');
                if (close) |ci| {
                    const hex = token_str[pos + 1 .. ci];
                    if (hex.len == 2) {
                        if (std.fmt.parseInt(u8, hex, 16)) |byte| {
                            try temp.append(byte);
                            pos = ci + 1;
                            continue;
                        } else |_| {
                            all_decoded = false;
                            break;
                        }
                    } else {
                        all_decoded = false;
                        break;
                    }
                } else {
                    all_decoded = false;
                    break;
                }
            } else {
                all_decoded = false;
                break;
            }
        }

        if (all_decoded and temp.items.len > 0) {
            try out_text.appendSlice(temp.items);
        } else {
            try out_text.appendSlice(token_str);
        }
    }

    pub fn longestMatch(self: *MGT, text: []const u8, start: usize) usize {
        var max_len: usize = 0;
        var len: usize = 1;

        while (start + len <= text.len) : (len += 1) {
            const substr = text[start .. start + len];
            if (self.token_to_id.contains(substr)) {
                max_len = len;
            }
        }

        return max_len;
    }

    pub fn vocabSize(self: *const MGT) usize {
        return self.token_to_id.count();
    }

    pub fn addVocabWord(self: *MGT, word: []const u8, is_anchor: bool) !void {
        const id = try self.addToken(word);
        if (is_anchor) {
            const h: u64 = @intCast(id);
            const key = self.id_to_token.get(id).?;
            try self.anchors.put(key, h);
        }
    }

    pub fn removeVocabWord(self: *MGT, word: []const u8) void {
        for (SPECIAL_TOKEN_STRINGS) |special| {
            if (mem.eql(u8, word, special)) return;
        }

        if (self.token_to_id.get(word)) |id| {
            if (self.id_to_token.get(id)) |allocated_ptr| {
                _ = self.token_to_id.remove(word);
                _ = self.id_to_token.remove(id);
                _ = self.anchors.remove(word);
                _ = self.prefixes.remove(word);
                _ = self.suffixes.remove(word);
                _ = self.roots.remove(word);

                var bpe_to_remove = std.ArrayList([]const u8).init(self.allocator);
                defer bpe_to_remove.deinit();
                var bpe_it = self.bpe_pairs.iterator();
                while (bpe_it.next()) |entry| {
                    if (entry.value_ptr.token_id == id) {
                        bpe_to_remove.append(entry.key_ptr.*) catch {};
                    }
                }
                for (bpe_to_remove.items) |bk| {
                    _ = self.bpe_pairs.remove(bk);
                }

                var idx: usize = 0;
                while (idx < self.allocated_strings.items.len) : (idx += 1) {
                    const str = self.allocated_strings.items[idx];
                    if (str.ptr == allocated_ptr.ptr) {
                        self.allocator.free(str);
                        _ = self.allocated_strings.orderedRemove(idx);
                        break;
                    }
                }
            }
        }
    }

    pub fn tokenizeWithAnchors(self: *MGT, text: []const u8, out_tokens: *std.ArrayList(u32), out_anchors: *std.ArrayList(usize)) !void {
        var temp_tokens = std.ArrayList(u32).init(self.allocator);
        defer temp_tokens.deinit();
        try self.encode(text, &temp_tokens);

        var char_pos: usize = 0;
        for (temp_tokens.items) |tid| {
            try out_tokens.append(tid);
            if (self.id_to_token.get(tid)) |tok_str| {
                if (self.anchors.contains(tok_str)) {
                    try out_anchors.append(char_pos);
                }
                char_pos += tok_str.len;
            }
        }
    }

    pub fn detokenize(self: *MGT, tokens: []const u32) ![]u8 {
        var text = std.ArrayList(u8).init(self.allocator);
        errdefer text.deinit();
        try self.decode(tokens, &text);
        return try text.toOwnedSlice();
    }

    pub fn encodeBatch(self: *MGT, texts: []const []const u8, allocator: Allocator) ![][]u32 {
        const results = try allocator.alloc([]u32, texts.len);
        errdefer allocator.free(results);
        var i: usize = 0;
        errdefer {
            var k: usize = 0;
            while (k < i) : (k += 1) {
                allocator.free(results[k]);
            }
        }
        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            errdefer tokens.deinit();
            try self.encode(text, &tokens);
            results[i] = try tokens.toOwnedSlice();
            i += 1;
        }
        return results;
    }

    pub fn batchDetokenize(self: *MGT, token_lists: []const []const u32, allocator: Allocator) ![][]u8 {
        _ = allocator;
        const results = try self.allocator.alloc([]u8, token_lists.len);
        errdefer self.allocator.free(results);
        var i: usize = 0;
        errdefer {
            var k: usize = 0;
            while (k < i) : (k += 1) {
                self.allocator.free(results[k]);
            }
        }
        for (token_lists) |tokens| {
            results[i] = try self.detokenize(tokens);
            i += 1;
        }
        return results;
    }

    pub fn saveVocab(self: *MGT, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        var writer = file.writer();

        const size = self.vocabSize();
        try writer.writeInt(u32, @as(u32, @intCast(size)), .little);

        var it = self.token_to_id.iterator();
        while (it.next()) |entry| {
            const word = entry.key_ptr.*;
            const id = entry.value_ptr.*;
            try writer.writeInt(u32, @as(u32, @intCast(word.len)), .little);
            try writer.writeAll(word);
            try writer.writeInt(u32, id, .little);
        }

        try writer.writeInt(u32, @as(u32, @intCast(self.bpe_pairs.count())), .little);
        var bpe_it = self.bpe_pairs.iterator();
        while (bpe_it.next()) |entry| {
            const key = entry.key_ptr.*;
            const merge = entry.value_ptr.*;
            try writer.writeInt(u32, @as(u32, @intCast(key.len)), .little);
            try writer.writeAll(key);
            try writer.writeInt(u32, merge.token_id, .little);
            try writer.writeInt(u32, merge.priority, .little);
        }

        const writeStringMap = struct {
            fn write(map: std.StringHashMap(u32), w: anytype) !void {
                try w.writeInt(u32, @as(u32, @intCast(map.count())), .little);
                var iter = map.iterator();
                while (iter.next()) |e| {
                    try w.writeInt(u32, @as(u32, @intCast(e.key_ptr.*.len)), .little);
                    try w.writeAll(e.key_ptr.*);
                    try w.writeInt(u32, e.value_ptr.*, .little);
                }
            }
        };

        try writeStringMap.write(self.prefixes, writer);
        try writeStringMap.write(self.suffixes, writer);
        try writeStringMap.write(self.roots, writer);

        try writer.writeInt(u32, @as(u32, @intCast(self.anchors.count())), .little);
        var anch_it = self.anchors.iterator();
        while (anch_it.next()) |entry| {
            const key = entry.key_ptr.*;
            try writer.writeInt(u32, @as(u32, @intCast(key.len)), .little);
            try writer.writeAll(key);
            try writer.writeInt(u64, entry.value_ptr.*, .little);
        }
    }

    fn resetState(self: *MGT) void {
        self.token_to_id.clearAndFree();
        self.id_to_token.clearAndFree();
        self.prefixes.clearAndFree();
        self.suffixes.clearAndFree();
        self.roots.clearAndFree();
        self.bpe_pairs.clearAndFree();
        self.anchors.clearAndFree();
        for (self.allocated_strings.items) |str| {
            self.allocator.free(str);
        }
        self.allocated_strings.clearAndFree();
        self.next_token_id = 0;
    }

    pub fn loadVocab(self: *MGT, path: []const u8) !void {
        self.resetState();

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        var reader = file.reader();

        const size = try reader.readInt(u32, .little);
        var i: usize = 0;
        while (i < size) : (i += 1) {
            const word_len = try reader.readInt(u32, .little);
            const word_buf = try self.allocator.alloc(u8, word_len);
            errdefer self.allocator.free(word_buf);
            try reader.readNoEof(word_buf);
            const id = try reader.readInt(u32, .little);

            try self.allocated_strings.append(word_buf);

            try self.token_to_id.put(word_buf, id);
            errdefer _ = self.token_to_id.remove(word_buf);
            try self.id_to_token.put(id, word_buf);

            if (id >= self.next_token_id) {
                self.next_token_id = id + 1;
            }
        }

        const bpe_count = try reader.readInt(u32, .little);
        var j: usize = 0;
        while (j < bpe_count) : (j += 1) {
            const key_len = try reader.readInt(u32, .little);
            const key_buf = try self.allocator.alloc(u8, key_len);
            errdefer self.allocator.free(key_buf);
            try reader.readNoEof(key_buf);
            const token_id = try reader.readInt(u32, .little);
            const priority = try reader.readInt(u32, .little);

            try self.allocated_strings.append(key_buf);
            try self.bpe_pairs.put(key_buf, .{ .token_id = token_id, .priority = priority });
        }

        const readStringMap = struct {
            fn read(map: *std.StringHashMap(u32), r: anytype, alloc: Allocator, alloc_list: *std.ArrayList([]u8)) !void {
                const count = try r.readInt(u32, .little);
                var k: usize = 0;
                while (k < count) : (k += 1) {
                    const len = try r.readInt(u32, .little);
                    const buf = try alloc.alloc(u8, len);
                    errdefer alloc.free(buf);
                    try r.readNoEof(buf);
                    const id = try r.readInt(u32, .little);

                    try alloc_list.append(buf);
                    try map.put(buf, id);
                }
            }
        };

        try readStringMap.read(&self.prefixes, reader, self.allocator, &self.allocated_strings);
        try readStringMap.read(&self.suffixes, reader, self.allocator, &self.allocated_strings);
        try readStringMap.read(&self.roots, reader, self.allocator, &self.allocated_strings);

        const anch_count = try reader.readInt(u32, .little);
        var l: usize = 0;
        while (l < anch_count) : (l += 1) {
            const key_len = try reader.readInt(u32, .little);
            const key_buf = try self.allocator.alloc(u8, key_len);
            errdefer self.allocator.free(key_buf);
            try reader.readNoEof(key_buf);
            const val = try reader.readInt(u64, .little);

            try self.allocated_strings.append(key_buf);
            try self.anchors.put(key_buf, val);
        }
    }

    pub fn unknownReplacement(self: *MGT, context: []const u8) u32 {
        _ = self;
        _ = context;
        return SPECIAL_TOKENS.UNK;
    }

    pub fn subwordSplit(self: *MGT, word: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();
        var i: usize = 0;
        while (i < word.len) {
            const match_result = self.longestMatch(word, i);
            if (match_result > 0) {
                const found_word = word[i .. i + match_result];
                if (self.token_to_id.get(found_word)) |tid| {
                    try tokens.append(tid);
                    i += match_result;
                    continue;
                }
            }

            var char_len = utf8CharLen(word[i]);
            if (i + char_len > word.len) char_len = 1;
            const chunk = word[i .. i + char_len];
            const bpe_tokens = try self.encodeBPE(chunk);
            defer self.allocator.free(bpe_tokens);
            for (bpe_tokens) |tok| {
                try tokens.append(tok);
            }
            i += char_len;
        }
        return try tokens.toOwnedSlice();
    }

    pub fn mergeSubwords(self: *MGT, subwords: []const []const u32) ![]u32 {
        var merged = std.ArrayList(u32).init(self.allocator);
        errdefer merged.deinit();
        for (subwords) |sw| {
            try merged.appendSlice(sw);
        }
        return try merged.toOwnedSlice();
    }

    pub fn validateTokens(self: *MGT, tokens: []const u32) bool {
        for (tokens) |tok| {
            if (!self.id_to_token.contains(tok)) return false;
        }
        return true;
    }

    pub fn coverage(self: *MGT, corpus: []const u8) f32 {
        var covered: usize = 0;
        var i: usize = 0;
        while (i < corpus.len) {
            const m = self.longestMatch(corpus, i);
            if (m > 0) {
                covered += m;
                i += m;
            } else {
                i += 1;
            }
        }
        if (corpus.len == 0) return 0.0;
        return @as(f32, @floatFromInt(covered)) / @as(f32, @floatFromInt(corpus.len));
    }

    pub fn encodeToTensor(self: *MGT, text: []const u8, allocator: Allocator) !core_tensor.Tensor {
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();
        try self.encode(text, &tokens);
        const shape = [_]usize{tokens.items.len};
        var tensor = try core_tensor.Tensor.init(allocator, &shape);
        {
            var i: usize = 0;
            while (i < tokens.items.len) : (i += 1) {
                const tok = tokens.items[i];
                tensor.data[i] = @floatFromInt(tok);
            }
        }
        return tensor;
    }

    pub fn encodeBatchToTensor(self: *MGT, texts: []const []const u8, allocator: Allocator) !core_tensor.Tensor {
        var all_tokens = std.ArrayList(u32).init(allocator);
        defer all_tokens.deinit();
        var max_len: usize = 0;
        var batch_lens = std.ArrayList(usize).init(allocator);
        defer batch_lens.deinit();
        for (texts) |text| {
            var tokens = std.ArrayList(u32).init(allocator);
            defer tokens.deinit();
            try self.encode(text, &tokens);
            if (tokens.items.len > max_len) max_len = tokens.items.len;
            try batch_lens.append(tokens.items.len);
            try all_tokens.appendSlice(tokens.items);
        }
        if (max_len == 0) max_len = 1;
        const shape = [_]usize{ texts.len, max_len };
        var tensor = try core_tensor.Tensor.init(allocator, &shape);
        @memset(tensor.data, 0);
        var offset: usize = 0;
        {
            var batch_idx: usize = 0;
            while (batch_idx < batch_lens.items.len) : (batch_idx += 1) {
                const blen = batch_lens.items[batch_idx];
                var jj: usize = 0;
                while (jj < blen) : (jj += 1) {
                    tensor.data[batch_idx * max_len + jj] = @floatFromInt(all_tokens.items[offset + jj]);
                }
                offset += blen;
            }
        }
        return tensor;
    }

    pub fn decodeFromTensor(self: *MGT, tensor: *const core_tensor.Tensor, allocator: Allocator) ![]u8 {
        if (tensor.shape.len == 0) {
            const empty = try allocator.alloc(u8, 0);
            return empty;
        }

        const total_len = blk: {
            var product: usize = 1;
            for (tensor.shape) |d| {
                product *= d;
            }
            break :blk product;
        };

        var tokens = try allocator.alloc(u32, total_len);
        defer allocator.free(tokens);
        {
            var i: usize = 0;
            while (i < total_len) : (i += 1) {
                const val = tensor.data[i];
                if (std.math.isNan(val) or std.math.isInf(val) or val < 0.0 or val > @as(f32, @floatFromInt(std.math.maxInt(u32)))) {
                    tokens[i] = SPECIAL_TOKENS.UNK;
                } else {
                    tokens[i] = @intFromFloat(val);
                }
            }
        }
        var out_text = std.ArrayList(u8).init(allocator);
        errdefer out_text.deinit();
        try self.decode(tokens, &out_text);
        return try out_text.toOwnedSlice();
    }
};

test "MGT encode decode" {
    const gpa = testing.allocator;
    const vocab = &.{ "hello", "world", " " };
    const anchors_list = &.{"hello"};
    var mgt = try MGT.init(gpa, vocab, anchors_list);
    defer mgt.deinit();
    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();
    try mgt.encode("hello world", &tokens);
    try testing.expect(tokens.items.len >= 2);
    var text = std.ArrayList(u8).init(gpa);
    defer text.deinit();
    try mgt.decode(tokens.items, &text);
    try testing.expectEqualStrings("hello world", text.items);
}

test "MGT add remove vocab" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    try mgt.addVocabWord("test", true);
    try testing.expect(mgt.anchors.contains("test"));
    mgt.removeVocabWord("test");
    try testing.expect(!mgt.anchors.contains("test"));
}

test "MGT longest match" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "hello", "hell" }, &.{});
    defer mgt.deinit();
    const len = mgt.longestMatch("hello", 0);
    try testing.expectEqual(@as(usize, 5), len);
}

test "MGT batch encode" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "a", "b" }, &.{});
    defer mgt.deinit();
    const texts = &.{ "a", "b" };
    const batches = try mgt.encodeBatch(texts, gpa);
    defer {
        for (batches) |batch| {
            gpa.free(batch);
        }
        gpa.free(batches);
    }
    try testing.expect(batches.len == 2);
}

test "MGT subword split" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "hel", "lo" }, &.{});
    defer mgt.deinit();
    const sub = try mgt.subwordSplit("hello");
    defer gpa.free(sub);
    try testing.expect(sub.len >= 1);
}

test "MGT coverage" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "hello", "world" }, &.{});
    defer mgt.deinit();
    const cov = mgt.coverage("hello world");
    try testing.expect(cov > 0.0);
}

test "MGT validate" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{"a"}, &.{});
    defer mgt.deinit();
    const valid = mgt.validateTokens(&.{0});
    try testing.expect(valid);
}

test "MGT tokenize with anchors" {
    const gpa = testing.allocator;
    const vocab = &.{ "test", "anchor" };
    const anchors_list = &.{"anchor"};
    var mgt = try MGT.init(gpa, vocab, anchors_list);
    defer mgt.deinit();
    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();
    var anchor_positions = std.ArrayList(usize).init(gpa);
    defer anchor_positions.deinit();
    try mgt.tokenizeWithAnchors("testanchor", &tokens, &anchor_positions);
    try testing.expect(tokens.items.len >= 1);
    try testing.expect(anchor_positions.items.len >= 1);
}

test "MGT batch detokenize" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "a", "b" }, &.{});
    defer mgt.deinit();
    const token_lists = &[_][]const u32{
        &.{4},
        &.{5},
    };
    const results = try mgt.batchDetokenize(token_lists, gpa);
    defer {
        for (results) |result| {
            mgt.allocator.free(result);
        }
        mgt.allocator.free(results);
    }
    try testing.expect(results.len == 2);
}

test "MGT vocab size" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "a", "b", "c" }, &.{});
    defer mgt.deinit();
    const size = mgt.vocabSize();
    try testing.expect(size >= 3);
}

test "MGT save and load vocab" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "test", "vocab" }, &.{"test"});
    defer mgt.deinit();
    try mgt.saveVocab("test_vocab.bin");
    defer {
        std.fs.cwd().deleteFile("test_vocab.bin") catch |err| {
            std.log.warn("Failed to delete test file: {}", .{err});
        };
    }
    const orig_size = mgt.vocabSize();
    const orig_bpe = mgt.bpe_pairs.count();
    const orig_prefix = mgt.prefixes.count();
    const orig_suffix = mgt.suffixes.count();
    const orig_roots = mgt.roots.count();
    const orig_anchors = mgt.anchors.count();

    var mgt2 = try MGT.init(gpa, &.{}, &.{});
    defer mgt2.deinit();
    try mgt2.loadVocab("test_vocab.bin");

    try testing.expectEqual(orig_size, mgt2.vocabSize());
    try testing.expectEqual(orig_bpe, mgt2.bpe_pairs.count());
    try testing.expectEqual(orig_prefix, mgt2.prefixes.count());
    try testing.expectEqual(orig_suffix, mgt2.suffixes.count());
    try testing.expectEqual(orig_roots, mgt2.roots.count());
    try testing.expectEqual(orig_anchors, mgt2.anchors.count());

    try testing.expect(mgt2.token_to_id.contains("test"));
    try testing.expect(mgt2.token_to_id.contains("vocab"));
    try testing.expect(mgt2.anchors.contains("test"));
}

test "MGT merge subwords" {
    var gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    const sub1 = &[_]u32{ 1, 2 };
    const sub2 = &[_]u32{ 3, 4 };
    const subwords = &[_][]const u32{ sub1, sub2 };
    const merged = try mgt.mergeSubwords(subwords);
    defer gpa.free(merged);
    try testing.expectEqual(@as(usize, 4), merged.len);
}

test "MGT unknown replacement" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    const replacement = mgt.unknownReplacement("context");
    try testing.expectEqual(MGT.SPECIAL_TOKENS.UNK, replacement);
}

test "MGT morphological decomposition" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{ "run", "walk" }, &.{});
    defer mgt.deinit();
    var tokens = std.ArrayList(u32).init(gpa);
    defer tokens.deinit();
    try mgt.encode("running", &tokens);
    try testing.expect(tokens.items.len >= 2);
    var has_run = false;
    var has_ing = false;
    for (tokens.items) |tid| {
        if (mgt.id_to_token.get(tid)) |tok_str| {
            if (mem.eql(u8, tok_str, "runn")) has_run = true;
            if (mem.eql(u8, tok_str, "run")) has_run = true;
            if (mem.eql(u8, tok_str, "ing")) has_ing = true;
        }
    }
    try testing.expect(has_ing or tokens.items.len >= 2);
}

test "MGT BPE training" {
    const gpa = testing.allocator;
    var mgt = try MGT.init(gpa, &.{}, &.{});
    defer mgt.deinit();
    const corpus = &.{ "hello", "help", "held" };
    try mgt.trainBPE(corpus, 10);
    try testing.expect(mgt.bpe_pairs.count() > 0);
}

test "MGT deterministic encoding" {
    const gpa = testing.allocator;
    var mgt1 = try MGT.init(gpa, &.{ "test", "data", " " }, &.{});
    defer mgt1.deinit();

    var tokens1 = std.ArrayList(u32).init(gpa);
    defer tokens1.deinit();
    try mgt1.encode("test data", &tokens1);

    var mgt2 = try MGT.init(gpa, &.{ "test", "data", " " }, &.{});
    defer mgt2.deinit();

    var tokens2 = std.ArrayList(u32).init(gpa);
    defer tokens2.deinit();
    try mgt2.encode("test data", &tokens2);

    try testing.expectEqualSlices(u32, tokens1.items, tokens2.items);
}