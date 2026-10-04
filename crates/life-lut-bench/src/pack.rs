//! Byte-grid <-> bit-packed-word conversions, shared by the correctness
//! checks and the benchmark harness. Bit `k` of a word represents column
//! `word_index * WORD_BITS + k` (`k = 0` is the word's least-significant
//! bit) — `gpu.rs`'s `bitpack.wgsl` and `cpu.rs`'s `west`/`east` helpers both
//! assume this convention.

pub fn pack_u32(cells: &[u8], width: usize, height: usize) -> Vec<u32> {
    assert_eq!(width % 32, 0, "bit-packed (u32) requires width % 32 == 0");
    let wpr = width / 32;
    let mut out = vec![0u32; wpr * height];
    for y in 0..height {
        for word in 0..wpr {
            let mut w = 0u32;
            for k in 0..32 {
                w |= (cells[y * width + word * 32 + k] as u32) << k;
            }
            out[y * wpr + word] = w;
        }
    }
    out
}

pub fn unpack_u32(words: &[u32], width: usize, height: usize) -> Vec<u8> {
    let wpr = width / 32;
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        for word in 0..wpr {
            let w = words[y * wpr + word];
            for k in 0..32 {
                out[y * width + word * 32 + k] = ((w >> k) & 1) as u8;
            }
        }
    }
    out
}

pub fn pack_u64(cells: &[u8], width: usize, height: usize) -> Vec<u64> {
    assert_eq!(width % 64, 0, "bit-packed (u64) requires width % 64 == 0");
    let wpr = width / 64;
    let mut out = vec![0u64; wpr * height];
    for y in 0..height {
        for word in 0..wpr {
            let mut w = 0u64;
            for k in 0..64 {
                w |= (cells[y * width + word * 64 + k] as u64) << k;
            }
            out[y * wpr + word] = w;
        }
    }
    out
}

pub fn unpack_u64(words: &[u64], width: usize, height: usize) -> Vec<u8> {
    let wpr = width / 64;
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        for word in 0..wpr {
            let w = words[y * wpr + word];
            for k in 0..64 {
                out[y * width + word * 64 + k] = ((w >> k) & 1) as u8;
            }
        }
    }
    out
}
