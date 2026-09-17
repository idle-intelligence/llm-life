//! wasm-bindgen wrapper: the same `Grid`/`Rule` the native evals use, exposed
//! to the demo page. No rule logic lives here.

use crate::{Grid, Rule};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct LifeApp {
    grid: Grid,
    rule: Rule,
    generation: usize,
}

#[wasm_bindgen]
impl LifeApp {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> LifeApp {
        LifeApp {
            grid: Grid::new(width, height),
            rule: Rule::life(),
            generation: 0,
        }
    }

    pub fn width(&self) -> usize {
        self.grid.width()
    }

    pub fn height(&self) -> usize {
        self.grid.height()
    }

    pub fn generation(&self) -> usize {
        self.generation
    }

    pub fn rulestring(&self) -> String {
        self.rule.to_rulestring()
    }

    /// Returns false if `s` isn't a valid B/S rulestring (the grid is left
    /// alone).
    pub fn set_rule(&mut self, s: &str) -> bool {
        match Rule::parse(s) {
            Some(r) => {
                self.rule = r;
                true
            }
            None => false,
        }
    }

    pub fn cells(&self) -> Vec<u8> {
        self.grid.cells().to_vec()
    }

    pub fn set_cells(&mut self, cells: &[u8]) {
        self.grid = Grid::from_cells(self.grid.width(), self.grid.height(), cells.to_vec());
    }

    pub fn toggle(&mut self, x: usize, y: usize) {
        self.grid.toggle(x, y);
    }

    pub fn clear(&mut self) {
        self.grid.clear();
        self.generation = 0;
    }

    pub fn live_count(&self) -> usize {
        self.grid.live_count()
    }

    pub fn randomize(&mut self, seed: u64, density: f64) {
        self.grid = Grid::random(self.grid.width(), self.grid.height(), seed, density);
        self.generation = 0;
    }

    pub fn place_glider(&mut self, x: usize, y: usize) {
        self.grid.place_glider(x, y);
    }

    pub fn step(&mut self, n: usize) {
        for _ in 0..n {
            self.grid = self.grid.step(&self.rule);
            self.generation += 1;
        }
    }
}
