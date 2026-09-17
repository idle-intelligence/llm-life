//! The grid and the classical update.
//!
//! Boundary is **toroidal** (wrap in both axes). Chosen so that every cell has
//! exactly 8 neighbors: the LLM variants build one attention stencil per cell
//! and a uniform 9-token neighborhood keeps the mask (and the per-cell prompt
//! in variant A) the same shape everywhere, with no edge special case.

use crate::rule::Rule;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grid {
    width: usize,
    height: usize,
    /// Row-major, one byte per cell, 0 or 1. Row-major is also the
    /// serialization order variant B feeds the model.
    cells: Vec<u8>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Self {
        Grid {
            width,
            height,
            cells: vec![0; width * height],
        }
    }

    pub fn from_cells(width: usize, height: usize, cells: Vec<u8>) -> Self {
        assert_eq!(cells.len(), width * height);
        Grid {
            width,
            height,
            cells,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn cells(&self) -> &[u8] {
        &self.cells
    }

    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.cells[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        self.cells[y * self.width + x] = v;
    }

    pub fn toggle(&mut self, x: usize, y: usize) {
        let i = y * self.width + x;
        self.cells[i] ^= 1;
    }

    pub fn clear(&mut self) {
        self.cells.fill(0);
    }

    pub fn live_count(&self) -> usize {
        self.cells.iter().filter(|&&c| c != 0).count()
    }

    /// The 8 neighbor indices of cell `i`, in a fixed order: NW, N, NE, W, E,
    /// SW, S, SE. Variant A's per-cell prompt lists them in this order and
    /// variant B's stencil mask opens exactly these keys, so the order is part
    /// of the contract between this crate and the model glue.
    pub fn neighbor_indices(&self, i: usize) -> [usize; 8] {
        let w = self.width;
        let h = self.height;
        let x = i % w;
        let y = i / w;
        let xm = (x + w - 1) % w;
        let xp = (x + 1) % w;
        let ym = (y + h - 1) % h;
        let yp = (y + 1) % h;
        [
            ym * w + xm,
            ym * w + x,
            ym * w + xp,
            y * w + xm,
            y * w + xp,
            yp * w + xm,
            yp * w + x,
            yp * w + xp,
        ]
    }

    pub fn live_neighbors(&self, i: usize) -> usize {
        self.neighbor_indices(i)
            .iter()
            .filter(|&&j| self.cells[j] != 0)
            .count()
    }

    /// One generation under `rule`.
    pub fn step(&self, rule: &Rule) -> Grid {
        let mut next = vec![0u8; self.cells.len()];
        for (i, out) in next.iter_mut().enumerate() {
            let n = self.live_neighbors(i);
            *out = rule.next(self.cells[i] != 0, n) as u8;
        }
        Grid {
            width: self.width,
            height: self.height,
            cells: next,
        }
    }

    /// Cells that differ, as a 0/1 map — the diff panel of the demo and the
    /// per-generation error count of the evals.
    pub fn diff(&self, other: &Grid) -> Vec<u8> {
        self.cells
            .iter()
            .zip(other.cells.iter())
            .map(|(a, b)| (a != b) as u8)
            .collect()
    }

    /// Deterministic pseudo-random fill (xorshift64), `density` in 0.0..=1.0.
    /// Deterministic so the demo, the native evals and the report all talk
    /// about the same seed.
    pub fn random(width: usize, height: usize, seed: u64, density: f64) -> Grid {
        // splitmix-style seed scramble: raw `seed | 1` collapses adjacent
        // seeds (42 and 43 produced the same grid).
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut cells = vec![0u8; width * height];
        for c in cells.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            *c = (u < density) as u8;
        }
        Grid {
            width,
            height,
            cells,
        }
    }

    /// Place a glider with its top-left at `(x, y)`.
    pub fn place_glider(&mut self, x: usize, y: usize) {
        for (dx, dy) in [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)] {
            self.set((x + dx) % self.width, (y + dy) % self.height, 1);
        }
    }
}
