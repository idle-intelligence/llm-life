// Seeds a random soup straight into the packed board on the GPU -- one
// invocation per 32-bit word, a bit-mixing hash of the cell index plus a
// seed uniform standing in for a real PRNG, so even a 1B-cell board seeds
// in a single dispatch instead of building and uploading a CPU array.

struct SeedParams {
    words_per_row: u32,
    height: u32,
    dispatch_x: u32,
    seed: u32,
    // Cell is alive when hash(index) < threshold; threshold = density *
    // 0xffffffff, computed on the CPU side.
    threshold: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: SeedParams;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

fn hash(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let total = params.words_per_row * params.height;
    let idx = (wgid.y * params.dispatch_x + wgid.x) * 256u + lid.x;
    if (idx >= total) {
        return;
    }
    var word: u32 = 0u;
    for (var k: u32 = 0u; k < 32u; k = k + 1u) {
        let cellIdx = idx * 32u + k;
        let h = hash(cellIdx * 0x9e3779b1u + params.seed);
        if (h < params.threshold) {
            word = word | (1u << k);
        }
    }
    dst[idx] = word;
}
