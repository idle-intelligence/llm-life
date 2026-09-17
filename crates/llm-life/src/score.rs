//! Scoring the model against true Life (CONCEPT.md §8).

use life::Grid;

#[derive(Debug, Clone, Copy)]
pub struct GenScore {
    pub generation: usize,
    /// Fraction of cells where the model's argmax equals true Life.
    pub accuracy: f64,
    /// Same, over live cells of the *true* next grid only — the number that
    /// actually moves, since a mostly-dead grid scores ~97% by answering 0
    /// everywhere.
    pub live_recall: f64,
    /// Mean p(alive) assigned to cells true Life says are alive, minus the
    /// mean assigned to cells it says are dead. 0 = the model is not
    /// separating them at all.
    pub confidence_gap: f64,
    pub wrong_cells: usize,
    pub true_live: usize,
    pub model_live: usize,
}

pub fn score(truth: &Grid, model: &Grid, p_alive: &[f32], generation: usize) -> GenScore {
    let n = truth.cells().len();
    let wrong: usize = truth.diff(model).iter().map(|&d| d as usize).sum();
    let true_live = truth.live_count();

    let mut hit = 0usize;
    let mut sum_live = 0f64;
    let mut sum_dead = 0f64;
    for (i, &p) in p_alive.iter().enumerate().take(n) {
        if truth.cells()[i] != 0 {
            sum_live += p as f64;
            if model.cells()[i] != 0 {
                hit += 1;
            }
        } else {
            sum_dead += p as f64;
        }
    }
    let dead = n - true_live;
    GenScore {
        generation,
        accuracy: 1.0 - wrong as f64 / n as f64,
        live_recall: if true_live == 0 { 1.0 } else { hit as f64 / true_live as f64 },
        confidence_gap: (if true_live == 0 { 0.0 } else { sum_live / true_live as f64 })
            - (if dead == 0 { 0.0 } else { sum_dead / dead as f64 }),
        wrong_cells: wrong,
        true_live,
        model_live: model.live_count(),
    }
}
