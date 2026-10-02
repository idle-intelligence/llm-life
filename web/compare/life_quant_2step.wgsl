// Ternary CNN that predicts two Life generations ahead in one forward pass
// (9,601 parameters: conv1 3x3 circular 1->32, conv2 3x3 circular 32->32,
// readout 1x1 32->1, each followed by a step(0/1) activation -- the same
// architecture as the compiled row's source model, bitslice-2step:
// pytorch/bitslice_models/life_2step_cnn_c32_seed3_exact.json, seed 3, run
// here as an ordinary (uncompiled) network instead of a bit-sliced circuit.
//
// Two dispatches per pass, same board resolution as life_quant.wgsl's single-
// layer rows:
//   layer1: one thread per cell, reads the 3x3 neighbourhood of the board
//     (0/1 per cell), computes all 32 conv1 output channels, steps each to a
//     bit, and packs the 32 bits into one u32 per cell (hidden buffer).
//   layer2: one thread per cell, reads the 3x3 neighbourhood of the hidden
//     buffer (9 packed words = 9*32 input bits), computes all 32 conv2
//     output channels, steps each to a bit (kept in a local array, never
//     written out), then the 1x1 readout over those 32 bits, steps once
//     more, and writes the final 0/1 result to the board output buffer.
//
// _float entry points dequantize every weight to a float (int weight * that
// layer's scale) and multiply; _int entry points fold the +-1/0 weight
// alphabet into adds/subtracts and apply each layer's one scale as a single
// float multiply at the end of that layer, same split as life_quant.wgsl's
// main_float/main_int. Both variants are bit-identical to each other and to
// the compiled circuit's output (same forward, same step() convention: step
// activation fires at y >= 0).

struct Params {
    width: u32,
    height: u32,
    channels: u32, // always 32; kept for layout parity with life_quant.wgsl
    dispatch_x: u32,
}

struct Scales1 {
    conv1_scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Scales2 {
    conv2_scale: f32,
    readout_scale: f32,
    readout_bias: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> scales1: Scales1;
@group(0) @binding(2) var<storage, read> conv1_weight: array<i32>; // 32 * 9, row-major 3x3 per channel
@group(0) @binding(3) var<storage, read> conv1_bias: array<f32>;   // 32
@group(0) @binding(4) var<storage, read> src: array<u32>;          // board, 0/1 per cell
@group(0) @binding(5) var<storage, read_write> hidden: array<u32>; // 32 bits packed per cell

fn cell_neighbourhood(i: u32) -> array<i32, 9> {
    let w = params.width;
    let h = params.height;
    let x = i % w;
    let y = i / w;
    let xm = (x + w - 1u) % w;
    let xp = (x + 1u) % w;
    let ym = (y + h - 1u) % h;
    let yp = (y + 1u) % h;
    var g: array<i32, 9>;
    g[0] = i32(src[ym * w + xm]);
    g[1] = i32(src[ym * w + x]);
    g[2] = i32(src[ym * w + xp]);
    g[3] = i32(src[y * w + xm]);
    g[4] = i32(src[y * w + x]);
    g[5] = i32(src[y * w + xp]);
    g[6] = i32(src[yp * w + xm]);
    g[7] = i32(src[yp * w + x]);
    g[8] = i32(src[yp * w + xp]);
    return g;
}

@compute @workgroup_size(256)
fn main_float_layer1(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wgid.y * params.dispatch_x + wgid.x) * 256u + lid.x;
    if (i >= params.width * params.height) {
        return;
    }
    let g = cell_neighbourhood(i);
    var packed: u32 = 0u;
    for (var c: u32 = 0u; c < 32u; c = c + 1u) {
        var sum1: f32 = 0.0;
        for (var k: u32 = 0u; k < 9u; k = k + 1u) {
            let wq1 = f32(conv1_weight[c * 9u + k]) * scales1.conv1_scale;
            sum1 = sum1 + wq1 * f32(g[k]);
        }
        if (sum1 + conv1_bias[c] >= 0.0) {
            packed = packed | (1u << c);
        }
    }
    hidden[i] = packed;
}

@compute @workgroup_size(256)
fn main_int_layer1(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wgid.y * params.dispatch_x + wgid.x) * 256u + lid.x;
    if (i >= params.width * params.height) {
        return;
    }
    let g = cell_neighbourhood(i);
    var packed: u32 = 0u;
    for (var c: u32 = 0u; c < 32u; c = c + 1u) {
        var sum1: i32 = 0;
        for (var k: u32 = 0u; k < 9u; k = k + 1u) {
            sum1 = sum1 + conv1_weight[c * 9u + k] * g[k];
        }
        if (scales1.conv1_scale * f32(sum1) + conv1_bias[c] >= 0.0) {
            packed = packed | (1u << c);
        }
    }
    hidden[i] = packed;
}

@group(0) @binding(0) var<uniform> params2: Params;
@group(0) @binding(1) var<uniform> scales2: Scales2;
@group(0) @binding(2) var<storage, read> conv2_weight: array<i32>;   // 32 * 32 * 9, [c2][c1][k]
@group(0) @binding(3) var<storage, read> conv2_bias: array<f32>;     // 32
@group(0) @binding(4) var<storage, read> readout_weight: array<i32>; // 32
@group(0) @binding(5) var<storage, read> hidden_in: array<u32>;      // 32 bits packed per cell
@group(0) @binding(6) var<storage, read_write> dst: array<u32>;      // board, 0/1 per cell

fn hidden_neighbourhood(i: u32) -> array<u32, 9> {
    let w = params2.width;
    let h = params2.height;
    let x = i % w;
    let y = i / w;
    let xm = (x + w - 1u) % w;
    let xp = (x + 1u) % w;
    let ym = (y + h - 1u) % h;
    let yp = (y + 1u) % h;
    var g: array<u32, 9>;
    g[0] = hidden_in[ym * w + xm];
    g[1] = hidden_in[ym * w + x];
    g[2] = hidden_in[ym * w + xp];
    g[3] = hidden_in[y * w + xm];
    g[4] = hidden_in[y * w + x];
    g[5] = hidden_in[y * w + xp];
    g[6] = hidden_in[yp * w + xm];
    g[7] = hidden_in[yp * w + x];
    g[8] = hidden_in[yp * w + xp];
    return g;
}

@compute @workgroup_size(256)
fn main_float_layer2(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wgid.y * params2.dispatch_x + wgid.x) * 256u + lid.x;
    if (i >= params2.width * params2.height) {
        return;
    }
    let g = hidden_neighbourhood(i);
    var h2: array<f32, 32>;
    for (var c2: u32 = 0u; c2 < 32u; c2 = c2 + 1u) {
        var sum2: f32 = 0.0;
        for (var k: u32 = 0u; k < 9u; k = k + 1u) {
            let word = g[k];
            for (var c1: u32 = 0u; c1 < 32u; c1 = c1 + 1u) {
                let bit = f32((word >> c1) & 1u);
                let wq2 = f32(conv2_weight[c2 * 288u + c1 * 9u + k]) * scales2.conv2_scale;
                sum2 = sum2 + wq2 * bit;
            }
        }
        h2[c2] = select(0.0, 1.0, sum2 + conv2_bias[c2] >= 0.0);
    }
    var sum3: f32 = 0.0;
    for (var c2: u32 = 0u; c2 < 32u; c2 = c2 + 1u) {
        let wqo = f32(readout_weight[c2]) * scales2.readout_scale;
        sum3 = sum3 + wqo * h2[c2];
    }
    dst[i] = select(0u, 1u, sum3 + scales2.readout_bias >= 0.0);
}

@compute @workgroup_size(256)
fn main_int_layer2(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wgid.y * params2.dispatch_x + wgid.x) * 256u + lid.x;
    if (i >= params2.width * params2.height) {
        return;
    }
    let g = hidden_neighbourhood(i);
    var h2: array<u32, 32>;
    for (var c2: u32 = 0u; c2 < 32u; c2 = c2 + 1u) {
        var sum2: i32 = 0;
        for (var k: u32 = 0u; k < 9u; k = k + 1u) {
            let word = g[k];
            for (var c1: u32 = 0u; c1 < 32u; c1 = c1 + 1u) {
                let bit = i32((word >> c1) & 1u);
                sum2 = sum2 + conv2_weight[c2 * 288u + c1 * 9u + k] * bit;
            }
        }
        h2[c2] = select(0u, 1u, scales2.conv2_scale * f32(sum2) + conv2_bias[c2] >= 0.0);
    }
    var sum3: i32 = 0;
    for (var c2: u32 = 0u; c2 < 32u; c2 = c2 + 1u) {
        sum3 = sum3 + readout_weight[c2] * i32(h2[c2]);
    }
    dst[i] = select(0u, 1u, scales2.readout_scale * f32(sum3) + scales2.readout_bias >= 0.0);
}
