//! Training data for the vector-space variants (CONCEPT.md §12).
//!
//! (i) reuses `bert::data`'s exhaustive 512-case generator directly — same
//! 9-slot layout (8 neighbours, `Grid::neighbor_indices` order, then self
//! last), just consumed as floats instead of token ids.
//!
//! (ii) needs whole grids, not single cases: a density sweep of random
//! `life::Grid`s and their true-Life next state, for the grid-to-grid
//! stencil model.

use life::{Grid, Rule};

/// One `(width*height)`-cell grid and its true-Life next state, as flat f32
/// (input) / u8 (target) vectors, row-major (`Grid::cells` order).
pub fn grid_pair(rule: &Rule, width: usize, height: usize, seed: u64, density: f64) -> (Vec<f32>, Vec<u8>) {
    let g = Grid::random(width, height, seed, density);
    let next = g.step(rule);
    let cells: Vec<f32> = g.cells().iter().map(|&c| c as f32).collect();
    (cells, next.cells().to_vec())
}

/// A training batch of `n_seeds` random 16x16 grids at each of the given
/// densities (CONCEPT.md §12's "density sweep 0.1-0.5"), flattened into one
/// `[batch, cells]` pair of vectors ready to hand to a tensor constructor.
/// Seeds are offset by `seed_base` so a held-out eval can pick a disjoint
/// range.
pub fn density_sweep_batch(
    rule: &Rule,
    width: usize,
    height: usize,
    densities: &[f64],
    n_seeds: u64,
    seed_base: u64,
) -> (Vec<f32>, Vec<u8>, usize) {
    let n = width * height;
    let batch = densities.len() * n_seeds as usize;
    let mut xs = Vec::with_capacity(batch * n);
    let mut ys = Vec::with_capacity(batch * n);
    for &d in densities {
        for s in 0..n_seeds {
            let seed = seed_base + (d * 1000.0) as u64 * 10_000 + s;
            let (x, y) = grid_pair(rule, width, height, seed, d);
            xs.extend(x);
            ys.extend(y);
        }
    }
    (xs, ys, batch)
}
