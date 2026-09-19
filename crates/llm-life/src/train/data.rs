//! Training data for variant B: random Life grids and their true next state.
//!
//! The supervision is dense — every cell of every grid is one classification
//! example — so a 32x32 grid is 1024 labelled cells per forward pass and the
//! data is free (CONCEPT.md §5: "generated from true Life ... millions of
//! (neighborhood, next state) pairs").
//!
//! Density is sampled in `[0.1, 0.4]` rather than fixed: the six
//! neighbourhood classes (`score::LifeCase`) are wildly imbalanced at any one
//! density, and the sparse end is where `birth` is common while the dense end
//! is where `death-crowded` is.

use life::{Grid, Rule};

/// xorshift64 — the same shape of PRNG `life::Grid::random` uses, kept here
/// so a training run is reproducible from one seed.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One training grid: mostly uniform-random at a sampled density, with a
/// small share of seeded objects (glider, blinker) so the model sees the
/// sparse, structured regime the pictures are scored on.
pub fn sample_grid(rng: &mut Rng, size: usize) -> Grid {
    let roll = rng.unit();
    if roll < 0.10 {
        let mut g = Grid::new(size, size);
        let x = (rng.unit() * size as f64) as usize;
        let y = (rng.unit() * size as f64) as usize;
        g.place_glider(x.min(size - 3), y.min(size - 3));
        return g;
    }
    if roll < 0.15 {
        let mut g = Grid::new(size, size);
        let x = (rng.unit() * (size - 3) as f64) as usize + 1;
        let y = (rng.unit() * (size - 3) as f64) as usize + 1;
        g.set(x - 1, y, 1);
        g.set(x, y, 1);
        g.set(x + 1, y, 1);
        return g;
    }
    let density = 0.1 + rng.unit() * 0.3;
    Grid::random(size, size, rng.next_u64(), density)
}

/// A fixed evaluation set: never drawn from the training RNG, so the per-case
/// recalls logged during a run are held-out numbers.
///
/// Seeds `1..=n-2` at density 0.28 (the picture pipeline's default), plus a
/// glider and a blinker, which are the sparse cases where `birth` and
/// `survive-*` are rare enough to be invisible in an aggregate number.
pub fn held_out(size: usize, n: usize) -> Vec<Grid> {
    let mut out = Vec::with_capacity(n);
    let mut g = Grid::new(size, size);
    g.place_glider(size / 2, size / 2);
    out.push(g);
    let mut b = Grid::new(size, size);
    b.set(size / 2 - 1, size / 2, 1);
    b.set(size / 2, size / 2, 1);
    b.set(size / 2 + 1, size / 2, 1);
    out.push(b);
    for seed in 0..n.saturating_sub(2) {
        out.push(Grid::random(size, size, 1_000_000 + seed as u64, 0.28));
    }
    out
}

/// The target class per cell: `1` if true Life says the cell is alive next
/// generation, else `0` — the index into the sliced head's `[dead, alive]`.
pub fn targets(grid: &Grid, rule: &Rule) -> Vec<u8> {
    grid.step(rule).cells().to_vec()
}
