//! CPU baselines: the same 512-entry byte-per-cell LUT as the GPU kernel,
//! and a bit-packed 64-cells-per-`u64` bitwise-adder version. Both come in a
//! single-thread and a rayon-threaded variant. No explicit SIMD intrinsics —
//! only whatever the compiler auto-vectorizes from the plain scalar/bitwise
//! loops below.

use rayon::prelude::*;

/// Byte-per-cell LUT, single thread. Toroidal wrap, matching
/// `life::grid::Grid`.
pub fn lut_step_scalar(src: &[u8], dst: &mut [u8], width: usize, height: usize, table: &[u32; 512]) {
    for y in 0..height {
        let ym = if y == 0 { height - 1 } else { y - 1 };
        let yp = if y + 1 == height { 0 } else { y + 1 };
        let row = y * width;
        let row_m = ym * width;
        let row_p = yp * width;
        for x in 0..width {
            let xm = if x == 0 { width - 1 } else { x - 1 };
            let xp = if x + 1 == width { 0 } else { x + 1 };
            let idx = (src[row + x] as usize)
                | ((src[row_m + xm] as usize) << 1)
                | ((src[row_m + x] as usize) << 2)
                | ((src[row_m + xp] as usize) << 3)
                | ((src[row + xm] as usize) << 4)
                | ((src[row + xp] as usize) << 5)
                | ((src[row_p + xm] as usize) << 6)
                | ((src[row_p + x] as usize) << 7)
                | ((src[row_p + xp] as usize) << 8);
            dst[row + x] = table[idx] as u8;
        }
    }
}

/// Same kernel, rayon over rows.
pub fn lut_step_rayon(src: &[u8], dst: &mut [u8], width: usize, height: usize, table: &[u32; 512]) {
    dst.par_chunks_mut(width).enumerate().for_each(|(y, dst_row)| {
        let ym = if y == 0 { height - 1 } else { y - 1 };
        let yp = if y + 1 == height { 0 } else { y + 1 };
        let row = y * width;
        let row_m = ym * width;
        let row_p = yp * width;
        for x in 0..width {
            let xm = if x == 0 { width - 1 } else { x - 1 };
            let xp = if x + 1 == width { 0 } else { x + 1 };
            let idx = (src[row + x] as usize)
                | ((src[row_m + xm] as usize) << 1)
                | ((src[row_m + x] as usize) << 2)
                | ((src[row_m + xp] as usize) << 3)
                | ((src[row + xm] as usize) << 4)
                | ((src[row + xp] as usize) << 5)
                | ((src[row_p + xm] as usize) << 6)
                | ((src[row_p + x] as usize) << 7)
                | ((src[row_p + xp] as usize) << 8);
            dst_row[x] = table[idx] as u8;
        }
    });
}

#[inline(always)]
fn west(self_w: u64, prev_w: u64) -> u64 {
    (self_w << 1) | (prev_w >> 63)
}

#[inline(always)]
fn east(self_w: u64, next_w: u64) -> u64 {
    (self_w >> 1) | ((next_w & 1) << 63)
}

#[inline(always)]
fn half_adder(a: u64, b: u64) -> (u64, u64) {
    (a ^ b, a & b)
}

#[inline(always)]
fn full_adder(a: u64, b: u64, c: u64) -> (u64, u64) {
    (a ^ b ^ c, (a & b) | (b & c) | (a & c))
}

/// One word's worth (64 cells) of the bit-packed adder-network step. Same
/// algorithm as `shaders/bitpack.wgsl`'s `main`, word size 64 instead of 32.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn bitpack_word(self_c: u64, self_p: u64, self_n: u64, above_c: u64, above_p: u64, above_n: u64, below_c: u64, below_p: u64, below_n: u64, birth_mask: u32, survive_mask: u32) -> u64 {
    let self_w = west(self_c, self_p);
    let self_e = east(self_c, self_n);
    let above_w = west(above_c, above_p);
    let above_e = east(above_c, above_n);
    let below_w = west(below_c, below_p);
    let below_e = east(below_c, below_n);

    let (s1, c1) = full_adder(above_w, above_c, above_e);
    let (s2, c2) = full_adder(self_w, self_e, below_w);
    let (s3, c3) = half_adder(below_c, below_e);

    let (b0, ca) = full_adder(s1, s2, s3);
    let (t1, t2) = full_adder(c1, c2, c3);

    let (b1, cb) = half_adder(ca, t1);
    let (b2, b3) = half_adder(t2, cb);

    let mut survive_or: u64 = 0;
    let mut birth_or: u64 = 0;
    for i in 0..9u32 {
        let e0 = if i & 1 == 1 { b0 } else { !b0 };
        let e1 = if (i >> 1) & 1 == 1 { b1 } else { !b1 };
        let e2 = if (i >> 2) & 1 == 1 { b2 } else { !b2 };
        let e3 = if (i >> 3) & 1 == 1 { b3 } else { !b3 };
        let eq = e0 & e1 & e2 & e3;
        if (survive_mask >> i) & 1 == 1 {
            survive_or |= eq;
        }
        if (birth_mask >> i) & 1 == 1 {
            birth_or |= eq;
        }
    }
    (self_c & survive_or) | (!self_c & birth_or)
}

fn bitpack_row(src: &[u64], dst_row: &mut [u64], row: usize, wpr: usize, height: usize, birth_mask: u32, survive_mask: u32) {
    let row_above = if row == 0 { height - 1 } else { row - 1 };
    let row_below = if row + 1 == height { 0 } else { row + 1 };
    let above = &src[row_above * wpr..row_above * wpr + wpr];
    let this = &src[row * wpr..row * wpr + wpr];
    let below = &src[row_below * wpr..row_below * wpr + wpr];
    for col in 0..wpr {
        let cp = if col == 0 { wpr - 1 } else { col - 1 };
        let cn = if col + 1 == wpr { 0 } else { col + 1 };
        dst_row[col] = bitpack_word(this[col], this[cp], this[cn], above[col], above[cp], above[cn], below[col], below[cp], below[cn], birth_mask, survive_mask);
    }
}

/// Bit-packed step, single thread. `width % 64 == 0` required (word-aligned
/// toroidal row wrap, same requirement as the GPU kernel at word size 32).
pub fn bitpack_step_scalar(src: &[u64], dst: &mut [u64], words_per_row: usize, height: usize, birth_mask: u32, survive_mask: u32) {
    for row in 0..height {
        let start = row * words_per_row;
        bitpack_row(src, &mut dst[start..start + words_per_row], row, words_per_row, height, birth_mask, survive_mask);
    }
}

pub fn bitpack_step_rayon(src: &[u64], dst: &mut [u64], words_per_row: usize, height: usize, birth_mask: u32, survive_mask: u32) {
    dst.par_chunks_mut(words_per_row).enumerate().for_each(|(row, dst_row)| {
        bitpack_row(src, dst_row, row, words_per_row, height, birth_mask, survive_mask);
    });
}
