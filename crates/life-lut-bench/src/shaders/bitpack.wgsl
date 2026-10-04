// Bit-packed kernel: 32 cells per u32 (WGSL has no u64), one thread per
// word. Counts the 8 toroidal neighbours of every one of the 32 cells in a
// word simultaneously with a bitwise full-adder network (classic
// "bit-sliced"/SWAR Life counter: sum 8 one-bit planes into a 4-bit count
// per bit-lane using 5 half/full adders), then applies the rule as a
// runtime birth/survive bitmask so this kernel is not Conway-only.
//
// Row wrap: toroidal, matching `life::grid::Grid`. Requires
// `width % 32 == 0` (word-aligned row wrap) — see `Params::words_per_row`;
// grids not aligned to a 32-cell boundary are not covered by this kernel
// (16x16 in this benchmark's grid list falls in that gap and is reported
// N/A for the bit-packed rows). Workgroup size 256, no subgroups.

struct Params {
    words_per_row: u32,
    height: u32,
    birth_mask: u32,
    survive_mask: u32,
    // See lut_byte.wgsl's Params.dispatch_x doc comment: folds a 2D
    // workgroup grid back into one linear word index when the word count
    // needs more than 65535 workgroups in one dimension.
    dispatch_x: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

// The word holding the columns just left (west) of `self_w`'s columns: bit k
// takes self_w's bit k-1, or prev_w's top bit (31) when k == 0.
fn west(self_w: u32, prev_w: u32) -> u32 {
    return (self_w << 1u) | (prev_w >> 31u);
}

// The word holding the columns just right (east): bit k takes self_w's bit
// k+1, or next_w's bottom bit (0) when k == 31.
fn east(self_w: u32, next_w: u32) -> u32 {
    return (self_w >> 1u) | ((next_w & 1u) << 31u);
}

fn half_adder(a: u32, b: u32) -> vec2<u32> {
    return vec2<u32>(a ^ b, a & b);
}

fn full_adder(a: u32, b: u32, c: u32) -> vec2<u32> {
    return vec2<u32>(a ^ b ^ c, (a & b) | (b & c) | (a & c));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let wpr = params.words_per_row;
    let h = params.height;
    let total = wpr * h;
    let idx = (wgid.y * params.dispatch_x + wgid.x) * 256u + lid.x;
    if (idx >= total) {
        return;
    }
    let row = idx / wpr;
    let col = idx % wpr;
    let row_above = (row + h - 1u) % h;
    let row_below = (row + 1u) % h;
    let col_prev = (col + wpr - 1u) % wpr;
    let col_next = (col + 1u) % wpr;

    let self_c = src[row * wpr + col];
    let self_w = west(self_c, src[row * wpr + col_prev]);
    let self_e = east(self_c, src[row * wpr + col_next]);

    let above_c = src[row_above * wpr + col];
    let above_w = west(above_c, src[row_above * wpr + col_prev]);
    let above_e = east(above_c, src[row_above * wpr + col_next]);

    let below_c = src[row_below * wpr + col];
    let below_w = west(below_c, src[row_below * wpr + col_prev]);
    let below_e = east(below_c, src[row_below * wpr + col_next]);

    // 8 neighbour bit-planes: above_w, above_c, above_e, self_w, self_e,
    // below_w, below_c, below_e (self_c is the cell's own state, not a
    // neighbour). Sum them bitwise into a 4-bit count per lane.
    let l1a = full_adder(above_w, above_c, above_e);
    let l1b = full_adder(self_w, self_e, below_w);
    let l1c = half_adder(below_c, below_e);

    let l2a = full_adder(l1a.x, l1b.x, l1c.x);
    let b0 = l2a.x; // weight 1, final
    let l2b = full_adder(l1a.y, l1b.y, l1c.y);

    let l3a = half_adder(l2a.y, l2b.x);
    let b1 = l3a.x; // weight 2, final
    let l3b = half_adder(l2b.y, l3a.y);
    let b2 = l3b.x; // weight 4, final
    let b3 = l3b.y; // weight 8, final (only set when count == 8)

    var survive_or: u32 = 0u;
    var birth_or: u32 = 0u;
    for (var i: u32 = 0u; i < 9u; i = i + 1u) {
        let e0 = select(~b0, b0, (i & 1u) == 1u);
        let e1 = select(~b1, b1, ((i >> 1u) & 1u) == 1u);
        let e2 = select(~b2, b2, ((i >> 2u) & 1u) == 1u);
        let e3 = select(~b3, b3, ((i >> 3u) & 1u) == 1u);
        let eq = e0 & e1 & e2 & e3;
        if (((params.survive_mask >> i) & 1u) == 1u) {
            survive_or = survive_or | eq;
        }
        if (((params.birth_mask >> i) & 1u) == 1u) {
            birth_or = birth_or | eq;
        }
    }

    dst[row * wpr + col] = (self_c & survive_or) | (~self_c & birth_or);
}
