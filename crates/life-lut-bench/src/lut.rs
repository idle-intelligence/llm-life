//! Shared rule encodings: the 512-entry outer-totalistic lookup table (used
//! by the byte-per-cell LUT kernels, CPU and GPU) and the 9-bit birth/survive
//! masks (used by the bit-packed bitwise-adder kernels, CPU and GPU). Both
//! are derived once from `life::rule::Rule` so every implementation scores
//! against the same rule as `llm-life`'s reference engine.

use life::rule::Rule;

/// 512-entry table keyed by a 9-bit neighbourhood: bit 0 is the cell's own
/// state, bits 1..=8 are its 8 neighbours (any fixed order — the rule only
/// depends on their popcount, not their identity). `table[idx]` is the next
/// state of the cell.
pub fn build_lut(rule: &Rule) -> [u32; 512] {
    let mut table = [0u32; 512];
    for (idx, out) in table.iter_mut().enumerate() {
        let center = idx & 1 != 0;
        let neighbor_bits = (idx >> 1) as u32;
        let count = neighbor_bits.count_ones() as usize;
        *out = rule.next(center, count) as u32;
    }
    table
}

/// 9-bit `birth`/`survive` masks (bit `n` set means "n live neighbours
/// triggers this transition"), as used by the bit-packed adder-network
/// kernels — they select which of the 9 popcount outcomes make a cell live,
/// without ever materializing a byte grid.
pub fn birth_survive_masks(rule: &Rule) -> (u32, u32) {
    let mut birth = 0u32;
    let mut survive = 0u32;
    for n in 0..9 {
        if rule.birth[n] {
            birth |= 1 << n;
        }
        if rule.survive[n] {
            survive |= 1 << n;
        }
    }
    (birth, survive)
}
