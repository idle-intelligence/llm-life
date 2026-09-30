// Reads the packed board (32 cells per u32, same packing as
// ../compare/bitpack.wgsl) directly out of a read-only storage buffer and
// paints it through a view (an origin in cells, plus a cells-per-pixel
// zoom) with toroidal wraparound in both dimensions. No readback: the board
// never leaves the GPU between the compute pass that advances it and the
// draw call that shows it.
//
// cells_per_pixel <= 1 (zoomed in, or the default 1:1 "screen" board): one
// cell per sample, nearest neighbour -- crisp.
// cells_per_pixel > 1 (zoomed out): each pixel covers more than one cell,
// so up to 8x8 samples are taken across the pixel's block (strided so the
// sample count never grows past 64 regardless of how far zoomed out) and
// averaged into a grey level.

struct Params {
    origin_x: f32,
    origin_y: f32,
    cells_per_pixel: f32,
    _pad0: f32,
    board_w: u32,
    board_h: u32,
    words_per_row: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> board: array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(p[idx], 0.0, 1.0);
}

fn wrap_u(v: i32, m: i32) -> u32 {
    return u32(((v % m) + m) % m);
}

fn cell_alive(cx: i32, cy: i32) -> u32 {
    let x = wrap_u(cx, i32(params.board_w));
    let y = wrap_u(cy, i32(params.board_h));
    let word = board[y * params.words_per_row + (x / 32u)];
    return (word >> (x % 32u)) & 1u;
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let cpp = params.cells_per_pixel;
    let baseX = params.origin_x + fragCoord.x * cpp;
    let baseY = params.origin_y + fragCoord.y * cpp;

    if (cpp <= 1.0) {
        let alive = cell_alive(i32(floor(baseX)), i32(floor(baseY)));
        let g = 1.0 - f32(alive);
        return vec4<f32>(g, g, g, 1.0);
    }

    let stride = max(1.0, cpp / 8.0);
    var count: u32 = 0u;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        for (var j: u32 = 0u; j < 8u; j = j + 1u) {
            let sx = baseX + f32(i) * stride;
            let sy = baseY + f32(j) * stride;
            count = count + cell_alive(i32(floor(sx)), i32(floor(sy)));
        }
    }
    let density = f32(count) / 64.0;
    let g = 1.0 - density;
    return vec4<f32>(g, g, g, 1.0);
}
