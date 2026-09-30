// Full-screen triangle + a fragment shader that reads the packed board
// (32 cells per u32, same packing as ../compare/bitpack.wgsl) directly out of
// a read-only storage buffer. No readback: the board never leaves the GPU
// between the compute pass that advances it and the draw call that shows it.

struct Params {
    width: u32,
    height: u32,
    words_per_row: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> board: array<u32>;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOut;
    out.pos = vec4<f32>(p[idx], 0.0, 1.0);
    out.uv = vec2<f32>((p[idx].x + 1.0) * 0.5, 1.0 - (p[idx].y + 1.0) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var x = u32(in.uv.x * f32(params.width));
    var y = u32(in.uv.y * f32(params.height));
    x = min(x, params.width - 1u);
    y = min(y, params.height - 1u);
    let word = board[y * params.words_per_row + (x / 32u)];
    let alive = (word >> (x % 32u)) & 1u;
    let g = 1.0 - f32(alive); // alive = black, dead = white, same convention as the other pages
    return vec4<f32>(g, g, g, 1.0);
}
