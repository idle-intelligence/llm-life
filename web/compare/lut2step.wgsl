// Ground truth from the rule, not from a trained model: a 2^25-entry, 1-bit
// table over every possible 5x5 neighbourhood (33,554,432 total cells, but
// the table only needs 2^25 = 33,554,432... see note below), filled on the
// GPU by build_table and then consumed by a resident ping-pong run
// (main_lookup2step) exactly like the other resident rows on this page.
//
// Bit convention: pattern index i in [0, 2^25); bit k of i is the state of
// the 5x5 window cell at row-major offset k, for (dy, dx) with dy, dx both
// ranging -2..2 and k = (dy+2)*5 + (dx+2) -- so k=12 is the centre cell.
// Same convention bitslice_life2_exhaustive.py uses for its independent
// ground-truth check (window_offsets(5), row-major), chosen so this table
// and the compiled 2-step circuit's own exhaustive check describe the same
// 2^25 patterns, even though this table is built from the rule directly and
// never touches the trained model.
//
// build_table computes, for each of the 2^25 patterns, the centre cell's
// state after Conway's rule is applied twice: first it derives the 3x3
// block of generation-t+1 states (each cell's own 3x3 neighbourhood sits
// entirely inside the 5x5 window), then applies the rule once more to that
// 3x3 block's centre to get the generation-t+2 state. One thread per
// pattern, plain integer popcount/compare, no bit-slicing -- an independent
// reference, not a copy of the compiled circuit's logic.
//
// main_lookup2step: one thread per board cell, same ping-pong/one-readback
// shape as every other resident row. Each thread gathers its own 25
// toroidal neighbours (same bit convention as above), builds the pattern
// index, and reads one bit out of the table -- the table was built once, at
// first use, and is reused across every generation and every run.

struct BuildParams {
    total: u32,
    dispatch_x: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> build_params: BuildParams;
@group(0) @binding(1) var<storage, read_write> table: array<u32>;

fn life_rule(center: u32, n: u32) -> u32 {
    if (center == 1u) {
        return select(0u, 1u, n == 2u || n == 3u);
    }
    return select(0u, 1u, n == 3u);
}

// Same 2D workgroup-grid fold as every other kernel on this page
// (dispatchShape's doc comment): 2^25 / 256 = 131,072 workgroups, well over
// WebGPU's 65535-per-dimension cap in a 1D dispatch.
@compute @workgroup_size(256)
fn build_table(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wgid.y * build_params.dispatch_x + wgid.x) * 256u + lid.x;
    if (i >= build_params.total) {
        return;
    }
    var cell: array<u32, 25>;
    for (var k: u32 = 0u; k < 25u; k = k + 1u) {
        cell[k] = (i >> k) & 1u;
    }
    // Generation t+1 over the inner 3x3 block (rows/cols 1..3 of the 5x5
    // window, i.e. offset -1..1 from the window's own centre at (2,2)).
    var mid: array<u32, 9>;
    for (var ry: i32 = -1; ry <= 1; ry = ry + 1) {
        for (var rx: i32 = -1; rx <= 1; rx = rx + 1) {
            let cy = 2 + ry;
            let cx = 2 + rx;
            var n: u32 = 0u;
            for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    n = n + cell[u32((cy + dy) * 5 + (cx + dx))];
                }
            }
            mid[u32((ry + 1) * 3 + (rx + 1))] = life_rule(cell[u32(cy * 5 + cx)], n);
        }
    }
    // Generation t+2: apply the rule once more to mid's own centre.
    var n2: u32 = 0u;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            n2 = n2 + mid[u32((1 + dy) * 3 + (1 + dx))];
        }
    }
    table[i] = life_rule(mid[4], n2);
}

struct Params {
    width: u32,
    height: u32,
    dispatch_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lut25: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256)
fn main_lookup2step(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wgid.y * params.dispatch_x + wgid.x) * 256u + lid.x;
    let w = params.width;
    let h = params.height;
    if (i >= w * h) {
        return;
    }
    let x = i32(i % w);
    let y = i32(i / w);
    var idx: u32 = 0u;
    var k: u32 = 0u;
    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
        let yy = u32((y + dy + 2 * i32(h)) % i32(h));
        for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
            let xx = u32((x + dx + 2 * i32(w)) % i32(w));
            idx = idx | (src[yy * w + xx] << k);
            k = k + 1u;
        }
    }
    dst[i] = lut25[idx];
}
