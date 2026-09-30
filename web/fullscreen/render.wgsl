// Full-screen triangle + a fragment shader that reads the packed board
// (32 cells per u32, same packing as ../compare/bitpack.wgsl) directly out of
// a read-only storage buffer. No readback: the board never leaves the GPU
// between the compute pass that advances it and the draw call that shows it.
// One cell per screen pixel: @builtin(position) is already in canvas pixel
// coordinates, so it maps directly onto the board.

struct Params {
    width: u32,
    height: u32,
    words_per_row: u32,
    _pad: u32,
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

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    var x = u32(fragCoord.x);
    var y = u32(fragCoord.y);
    x = min(x, params.width - 1u);
    y = min(y, params.height - 1u);
    let word = board[y * params.words_per_row + (x / 32u)];
    let alive = (word >> (x % 32u)) & 1u;
    let g = 1.0 - f32(alive); // alive = black, dead = white, same convention as the other pages
    return vec4<f32>(g, g, g, 1.0);
}
