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

/// Threshold p(alive) at the grid's **median** instead of 0.5.
///
/// The base/instruct models put every cell on the same side of 0.5 (they
/// answer `0` everywhere), so a fixed threshold reads out nothing even when
/// the model orders the cells correctly. The median is the cheapest
/// calibration that keeps the ordering and is equivalent to thresholding the
/// logit difference `1`-`0`, since the two-way softmax is monotone in it.
/// It hands the model the live fraction for free, so a live recall measured
/// this way is an upper bound on what the model knows, not an accuracy claim.
pub fn median_threshold_grid(p: &[f32], width: usize, height: usize) -> Grid {
    let mut sorted: Vec<f32> = p.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t = sorted[sorted.len() / 2];
    Grid::from_cells(width, height, p.iter().map(|&v| (v > t) as u8).collect())
}

/// Otsu's method (Otsu 1979, "A Threshold Selection Method from Gray-Level
/// Histograms"): the threshold that maximizes between-class variance on a
/// 256-bin histogram of `p`. Label-free — it looks only at the shape of the
/// p(alive) distribution, never at true Life.
///
/// A constant grid (max <= min) has no gap to find; the threshold is set to
/// the constant value itself, so `p > t` is false everywhere and the grid
/// comes back all-dead rather than an arbitrary all-alive split.
pub fn otsu_threshold(p: &[f32]) -> f32 {
    const BINS: usize = 256;
    let (min, max) = p
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    if max <= min || max.is_nan() {
        return max;
    }

    let mut hist = [0usize; BINS];
    for &v in p {
        let b = (((v - min) / (max - min)) * (BINS - 1) as f32).round() as usize;
        hist[b.min(BINS - 1)] += 1;
    }
    let total = p.len();
    let sum_all: f64 = hist.iter().enumerate().map(|(i, &c)| i as f64 * c as f64).sum();

    let mut sum_b = 0.0;
    let mut w_b = 0usize;
    let mut best_bin = 0usize;
    let mut max_var = -1.0f64;
    for (i, &c) in hist.iter().enumerate() {
        w_b += c;
        if w_b == 0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0 {
            break;
        }
        sum_b += i as f64 * c as f64;
        let m_b = sum_b / w_b as f64;
        let m_f = (sum_all - sum_b) / w_f as f64;
        let var_between = w_b as f64 * w_f as f64 * (m_b - m_f).powi(2);
        if var_between > max_var {
            max_var = var_between;
            best_bin = i;
        }
    }
    min + (best_bin as f32 / (BINS - 1) as f32) * (max - min)
}

pub fn otsu_threshold_grid(p: &[f32], width: usize, height: usize) -> Grid {
    let t = otsu_threshold(p);
    Grid::from_cells(width, height, p.iter().map(|&v| (v > t) as u8).collect())
}

/// mean + k*std of `p` — a label-free threshold that marks cells whose
/// p(alive) is an outlier on the high side. `k = 2.0` is the conventional
/// default (roughly the top ~2.3% of a normal distribution).
///
/// A constant grid has std = 0, so the threshold equals the constant value
/// and `p > t` is false everywhere: all-dead, same convention as
/// [`otsu_threshold`].
pub fn zscore_threshold(p: &[f32], k: f64) -> f32 {
    let n = p.len() as f64;
    let mean = p.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = p.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean + k * var.sqrt()) as f32
}

pub fn zscore_threshold_grid(p: &[f32], width: usize, height: usize, k: f64) -> Grid {
    let t = zscore_threshold(p, k);
    Grid::from_cells(width, height, p.iter().map(|&v| (v > t) as u8).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsu_finds_the_gap_in_a_bimodal_grid() {
        // 8 cells near 0.1, 8 cells near 0.9 (4x4 grid) — a clean bimodal
        // split. Otsu should separate them regardless of exactly where in
        // the gap it lands.
        let p: Vec<f32> = std::iter::repeat_n(0.1f32, 8)
            .chain(std::iter::repeat_n(0.9f32, 8))
            .collect();
        let grid = otsu_threshold_grid(&p, 4, 4);
        assert_eq!(grid.live_count(), 8);
        for (i, &v) in p.iter().enumerate() {
            assert_eq!(grid.cells()[i] != 0, v > 0.5, "cell {i} (p={v}) misclassified");
        }
    }

    #[test]
    fn otsu_leaves_an_all_equal_grid_all_dead() {
        let p = vec![0.42f32; 16];
        let grid = otsu_threshold_grid(&p, 4, 4);
        assert_eq!(grid.live_count(), 0);
    }

    #[test]
    fn zscore_marks_exactly_the_outliers() {
        // 14 cells at 0.0, 2 outliers at 10.0, in a 4x4 grid.
        // mean = 1.25, var = 10.9375, std ~= 3.307, threshold (k=2) ~= 7.86:
        // the two 10.0s clear it, nothing else does.
        let mut p = vec![0.0f32; 16];
        p[3] = 10.0;
        p[11] = 10.0;
        let grid = zscore_threshold_grid(&p, 4, 4, 2.0);
        assert_eq!(grid.live_count(), 2);
        assert_ne!(grid.cells()[3], 0);
        assert_ne!(grid.cells()[11], 0);
    }

    #[test]
    fn zscore_leaves_an_all_equal_grid_all_dead() {
        let p = vec![0.7f32; 16];
        let grid = zscore_threshold_grid(&p, 4, 4, 2.0);
        assert_eq!(grid.live_count(), 0);
    }
}
