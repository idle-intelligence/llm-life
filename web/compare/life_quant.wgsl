// Ternary/binary CNN kernel, two entry points sharing the same bindings and
// weight buffers: one thread per cell, resident ping-pong like lut_byte.wgsl's
// rows. Both match minimal_life_1bit.py's QuantCNN forward exactly: conv3x3
// (1->channels, circular padding, ternary/binary weights in {-1,0,1} or
// {-1,1}) -> per-layer scale + bias -> relu -> conv1x1 (channels->1, same
// weight alphabet) -> per-layer scale + bias -> sigmoid >= 0.5 (equivalent to
// logit >= 0, used here to avoid computing sigmoid).
//
// main_int: both weighted sums are integer adds/subtracts only. Layer 1 sums
// 0/1 cell values with +-1/0 integer weights (no float multiply until the one
// per-layer scale is applied); layer 2 sums the (already-float, post-relu)
// hidden values with the same +-1/0 weight alphabet, so it is also pure
// add/subtract of floats, never a multiply, before its own one per-layer
// scale.
//
// main_float: an ordinary float forward over the same dequantized weights
// (int weight * its layer's scale, computed once per tap/channel as a plain
// float), i.e. what a normal (non-quantization-aware) float convolution
// would do with these weight values -- every tap and every channel is a
// float multiply, not an add/subtract. Produces bit-identical output to
// main_int (both implement the same forward exactly, just with the +-1/0
// multiplies spelled out or folded into the per-layer scale), and is the
// "ordinary" row this page compares the integer kernel against.
//
// Channel count is runtime (4 for the ternary 45-param model, 16 for the
// binary 177-param model); both models share this one shader and host
// buffer layout.

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    // See lut_byte.wgsl's Params.dispatch_x doc comment: folds a 2D
    // workgroup grid back into one linear cell index when the cell count
    // needs more than 65535 workgroups in one dimension.
    dispatch_x: u32,
}

struct Scales {
    conv1_scale: f32,
    conv2_scale: f32,
    conv2_bias: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> scales: Scales;
@group(0) @binding(2) var<storage, read> conv1_weight: array<i32>; // channels * 9, row-major 3x3 per channel
@group(0) @binding(3) var<storage, read> conv1_bias: array<f32>;   // channels
@group(0) @binding(4) var<storage, read> conv2_weight: array<i32>; // channels, in {-1, 0, 1}
@group(0) @binding(5) var<storage, read> src: array<u32>;
@group(0) @binding(6) var<storage, read_write> dst: array<u32>;

fn neighbourhood(i: u32) -> array<i32, 9> {
    let w = params.width;
    let h = params.height;
    let x = i % w;
    let y = i / w;
    let xm = (x + w - 1u) % w;
    let xp = (x + 1u) % w;
    let ym = (y + h - 1u) % h;
    let yp = (y + 1u) % h;
    // Row-major 3x3, same (dy, dx) order as the flattened conv1_weight rows:
    // (-1,-1) (-1,0) (-1,1) (0,-1) (0,0) (0,1) (1,-1) (1,0) (1,1).
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
fn main_int(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wgid.y * params.dispatch_x + wgid.x) * 256u + lid.x;
    if (i >= params.width * params.height) {
        return;
    }
    let g = neighbourhood(i);

    var sum2: f32 = 0.0;
    for (var c: u32 = 0u; c < params.channels; c = c + 1u) {
        var sum1: i32 = 0;
        for (var k: u32 = 0u; k < 9u; k = k + 1u) {
            sum1 = sum1 + conv1_weight[c * 9u + k] * g[k];
        }
        let hidden = max(scales.conv1_scale * f32(sum1) + conv1_bias[c], 0.0);
        let w2 = conv2_weight[c];
        if (w2 == 1) {
            sum2 = sum2 + hidden;
        } else if (w2 == -1) {
            sum2 = sum2 - hidden;
        }
    }
    let logit = scales.conv2_scale * sum2 + scales.conv2_bias;
    dst[i] = select(0u, 1u, logit >= 0.0);
}

// Same forward, computed as an ordinary float network: every tap and every
// channel is a float multiply against the dequantized weight (int weight *
// that layer's scale), instead of folding the +-1/0 weight alphabet into
// adds/subtracts plus one scale multiply per layer.
@compute @workgroup_size(256)
fn main_float(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = (wgid.y * params.dispatch_x + wgid.x) * 256u + lid.x;
    if (i >= params.width * params.height) {
        return;
    }
    let g = neighbourhood(i);

    var logit: f32 = scales.conv2_bias;
    for (var c: u32 = 0u; c < params.channels; c = c + 1u) {
        var sum1: f32 = 0.0;
        for (var k: u32 = 0u; k < 9u; k = k + 1u) {
            let wq1 = f32(conv1_weight[c * 9u + k]) * scales.conv1_scale;
            sum1 = sum1 + wq1 * f32(g[k]);
        }
        let hidden = max(sum1 + conv1_bias[c], 0.0);
        let wq2 = f32(conv2_weight[c]) * scales.conv2_scale;
        logit = logit + wq2 * hidden;
    }
    dst[i] = select(0u, 1u, logit >= 0.0);
}
