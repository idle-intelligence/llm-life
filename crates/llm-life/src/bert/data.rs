//! Training data for BERT of Life: the exhaustive 512 neighbourhood cases
//! (CONCEPT.md §11/§12: "the smallest model that fits the 512 cases") plus
//! real 16x16 grids for the IoU rollout, reusing variant A/B's generators
//! (`life::Grid`, `train::data::sample_grid`).
//!
//! Token order matches `Grid::neighbor_indices` (NW,N,NE,W,E,SW,S,SE) with
//! the cell's own state appended last — the same bit layout the native
//! driver's `case_index` and variant A's per-cell prompt already use, so a
//! case index `k` here means the same neighbourhood everywhere in this repo.

use life::{Grid, Rule};

/// One `[u8; 9]` case: 8 neighbour bits (fixed order) then the cell's own
/// state, decoded from a 9-bit index `0..512`.
pub fn case_from_index(k: usize) -> [u8; 9] {
    let mut c = [0u8; 9];
    for (b, slot) in c.iter_mut().take(8).enumerate() {
        *slot = ((k >> b) & 1) as u8;
    }
    c[8] = ((k >> 8) & 1) as u8;
    c
}

/// The label the classical rule gives one case: `1` (alive next generation)
/// or `0`.
pub fn label(case: &[u8; 9], rule: &Rule) -> u8 {
    let n = case[..8].iter().filter(|&&b| b != 0).count();
    rule.next(case[8] != 0, n) as u8
}

/// All 512 exhaustive cases and their true-Life labels.
pub fn all_cases(rule: &Rule) -> Vec<([u8; 9], u8)> {
    (0..512).map(|k| {
        let c = case_from_index(k);
        let y = label(&c, rule);
        (c, y)
    }).collect()
}

/// One neighbourhood case and its true-Life label.
pub type Case = ([u8; 9], u8);

/// A deterministic train/held-out split of the 512 cases: `held_out_n` cases
/// (evenly spaced, not random, so the split is reproducible without a seed)
/// held out for the generalisation row, the rest for training.
pub fn split_512(rule: &Rule, held_out_n: usize) -> (Vec<Case>, Vec<Case>) {
    let all = all_cases(rule);
    let stride = 512 / held_out_n.max(1);
    let mut train = Vec::with_capacity(512 - held_out_n);
    let mut held = Vec::with_capacity(held_out_n);
    for (k, case) in all.into_iter().enumerate() {
        if k % stride == 0 && held.len() < held_out_n {
            held.push(case);
        } else {
            train.push(case);
        }
    }
    (train, held)
}

/// The 9-token case at cell `i` of `grid`, in the same order `case_from_index`
/// decodes.
pub fn case_at(grid: &Grid, i: usize) -> [u8; 9] {
    let mut c = [0u8; 9];
    for (b, &j) in grid.neighbor_indices(i).iter().enumerate() {
        c[b] = grid.cells()[j];
    }
    c[8] = grid.cells()[i];
    c
}

/// Every cell of `grid` as a `(case, label)` pair, for the IoU rollout —
/// dense supervision, reusing `Rule::next` directly rather than the LLM
/// prompt/packing machinery variants A/B use.
pub fn grid_cases(grid: &Grid, rule: &Rule) -> Vec<([u8; 9], u8)> {
    (0..grid.cells().len())
        .map(|i| {
            let c = case_at(grid, i);
            let y = label(&c, rule);
            (c, y)
        })
        .collect()
}
