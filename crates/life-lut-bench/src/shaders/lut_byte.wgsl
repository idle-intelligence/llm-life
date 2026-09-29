// Byte-per-cell LUT kernel: one thread per cell, 9 reads (self + 8
// toroidal neighbours), pack a 9-bit index (bit0 = self, bits1..8 =
// neighbours), look up the 512-entry table. The table only depends on
// neighbour *count*, so any fixed neighbour-to-bit assignment is correct —
// this one matches `life::grid::Grid::neighbor_indices`'s NW,N,NE,W,E,SW,S,SE
// order for readability, not for correctness.
//
// One u32 per cell (not a packed byte): WGSL storage buffers have no native
// 1-byte element, and packing 4 cells/u32 here would need atomic
// read-modify-write since 4 threads would share an output word — out of
// scope for this benchmark (see docs/runs/2026-09-29-life-lut-bench.md). The
// CPU LUT baseline is a true `u8`-per-cell buffer, matching `life::Grid`.
// Workgroup size 256: under WebGPU's browser cap, portable to a wasm build.

struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> lut: array<u32, 512>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let w = params.width;
    let h = params.height;
    let n = w * h;
    if (i >= n) {
        return;
    }
    let x = i % w;
    let y = i / w;
    let xm = (x + w - 1u) % w;
    let xp = (x + 1u) % w;
    let ym = (y + h - 1u) % h;
    let yp = (y + 1u) % h;

    let center = src[y * w + x];
    let nw = src[ym * w + xm];
    let no = src[ym * w + x];
    let ne = src[ym * w + xp];
    let we = src[y * w + xm];
    let ea = src[y * w + xp];
    let sw = src[yp * w + xm];
    let so = src[yp * w + x];
    let se = src[yp * w + xp];

    let idx = center
        | (nw << 1u)
        | (no << 2u)
        | (ne << 3u)
        | (we << 4u)
        | (ea << 5u)
        | (sw << 6u)
        | (so << 7u)
        | (se << 8u);

    dst[i] = lut[idx];
}
