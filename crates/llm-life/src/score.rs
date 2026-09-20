//! Scoring the model against true Life (CONCEPT.md §8).

use life::{Grid, Rule};

#[derive(Debug, Clone, Copy)]
pub struct GenScore {
    pub generation: usize,
    /// Fraction of cells where the model's argmax equals true Life.
    pub accuracy: f64,
    /// Same, over live cells of the *true* next grid only — the number that
    /// actually moves, since a mostly-dead grid scores ~97% by answering 0
    /// everywhere. Equal to `recall` below; kept under its original name
    /// since every existing table column is named `live_recall`.
    pub live_recall: f64,
    /// |pred alive ∩ true alive| / |model_live|. 1.0 when the model predicts
    /// no alive cells and truth has none either (no false positives to have).
    pub precision: f64,
    /// Alias of `live_recall`: |pred alive ∩ true alive| / |true_live|.
    pub recall: f64,
    /// Alias of `live_recall`/`recall`, spelled out so a table header can say
    /// "alive recall" next to "dead recall" without the reader having to know
    /// they're the same number under three names.
    pub alive_recall: f64,
    /// Fraction of true-*dead* cells the model also calls dead (specificity):
    /// |pred dead ∩ true dead| / |true_dead|. This is the number a dead-cell
    /// majority can inflate `accuracy` around while alive_recall looks fine —
    /// it catches the model over-predicting alive (TC's "13 instead of
    /// 5" case) even when every true-alive cell is still covered.
    pub dead_recall: f64,
    /// Alias of `precision`: |pred alive ∩ true alive| / |model_live|.
    pub alive_precision: f64,
    /// Harmonic mean of precision and recall. 0.0 when both are 0.
    pub f1: f64,
    /// IoU (Jaccard) of the alive sets: |pred ∧ true| / |pred ∨ true|.
    /// 1.0 when both the predicted and true alive sets are empty.
    pub iou: f64,
    /// Mean p(alive) assigned to cells true Life says are alive, minus the
    /// mean assigned to cells it says are dead. 0 = the model is not
    /// separating them at all.
    pub confidence_gap: f64,
    /// Hamming distance: number of cells where pred != true. The "mutation
    /// count per generation" TC asked accuracy alone to not hide.
    pub wrong_cells: usize,
    pub true_live: usize,
    pub model_live: usize,
    /// True positives: predicted alive, true alive.
    pub tp: usize,
    /// False positives: predicted alive, true dead.
    pub fp: usize,
    /// False negatives: predicted dead, true alive.
    pub fn_: usize,
    /// True negatives: predicted dead, true dead.
    pub tn: usize,
}

pub fn score(truth: &Grid, model: &Grid, p_alive: &[f32], generation: usize) -> GenScore {
    let n = truth.cells().len();
    let wrong: usize = truth.diff(model).iter().map(|&d| d as usize).sum();
    let true_live = truth.live_count();
    let model_live = model.live_count();

    let mut hit = 0usize; // |pred alive ∩ true alive|
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
    let union = true_live + model_live - hit;
    let recall = if true_live == 0 { 1.0 } else { hit as f64 / true_live as f64 };
    let precision = if model_live == 0 {
        if true_live == 0 {
            1.0
        } else {
            0.0
        }
    } else {
        hit as f64 / model_live as f64
    };
    let tp = hit;
    let fp = model_live - hit;
    let fn_ = true_live - hit;
    let tn = dead - fp;
    let dead_recall = if dead == 0 { 1.0 } else { tn as f64 / dead as f64 };
    GenScore {
        generation,
        accuracy: 1.0 - wrong as f64 / n as f64,
        live_recall: recall,
        precision,
        recall,
        alive_recall: recall,
        dead_recall,
        alive_precision: precision,
        f1: if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        },
        iou: if union == 0 { 1.0 } else { hit as f64 / union as f64 },
        confidence_gap: (if true_live == 0 { 0.0 } else { sum_live / true_live as f64 })
            - (if dead == 0 { 0.0 } else { sum_dead / dead as f64 }),
        wrong_cells: wrong,
        true_live,
        model_live,
        tp,
        fp,
        fn_,
        tn,
    }
}

/// The six exhaustive Life cases, from a cell's own state and its live
/// neighbor count. Every cell in a grid falls into exactly one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifeCase {
    /// Dead, exactly 3 neighbours: the rule says it is born.
    Birth,
    /// Alive, exactly 2 neighbours: survives.
    Survive2,
    /// Alive, exactly 3 neighbours: survives.
    Survive3,
    /// Alive, fewer than 2 neighbours: dies of loneliness.
    DeathLonely,
    /// Alive, more than 3 neighbours: dies of overcrowding.
    DeathCrowded,
    /// Dead, not exactly 3 neighbours: stays dead.
    StayDead,
}

impl LifeCase {
    pub const ALL: [LifeCase; 6] = [
        LifeCase::Birth,
        LifeCase::Survive2,
        LifeCase::Survive3,
        LifeCase::DeathLonely,
        LifeCase::DeathCrowded,
        LifeCase::StayDead,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            LifeCase::Birth => "birth (dead, =3)",
            LifeCase::Survive2 => "survive-2",
            LifeCase::Survive3 => "survive-3",
            LifeCase::DeathLonely => "death-lonely (<2)",
            LifeCase::DeathCrowded => "death-crowded (>3)",
            LifeCase::StayDead => "stay-dead",
        }
    }
}

/// Classify one cell by its own state and live-neighbor count. Assumes
/// Conway's B3/S23 case boundaries (2/3 to survive, 3 to be born) — the only
/// rule this repo scores against.
pub fn classify(alive: bool, neighbors: usize) -> LifeCase {
    if alive {
        match neighbors {
            0 | 1 => LifeCase::DeathLonely,
            2 => LifeCase::Survive2,
            3 => LifeCase::Survive3,
            _ => LifeCase::DeathCrowded,
        }
    } else if neighbors == 3 {
        LifeCase::Birth
    } else {
        LifeCase::StayDead
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CaseCount {
    pub count: usize,
    pub correct: usize,
}

impl CaseCount {
    /// Fraction of this case's cells the model got right. 1.0 on an empty
    /// class (nothing to get wrong), same convention as `iou`.
    pub fn fraction(&self) -> f64 {
        if self.count == 0 {
            1.0
        } else {
            self.correct as f64 / self.count as f64
        }
    }
}

/// Per-neighborhood-class recall, computed from `input` (the seed the model
/// saw) and `rule`: for every cell, classify it by `input`'s own state and
/// live-neighbor count, then check whether `model`'s cell matches the one
/// correct next state for that class.
pub fn per_case_recall(input: &Grid, model: &Grid, rule: &Rule) -> [(LifeCase, CaseCount); 6] {
    let mut counts = [CaseCount::default(); 6];
    for i in 0..input.cells().len() {
        let alive = input.cells()[i] != 0;
        let n = input.live_neighbors(i);
        let case = classify(alive, n);
        let idx = LifeCase::ALL.iter().position(|&c| c == case).unwrap();
        counts[idx].count += 1;
        if (model.cells()[i] != 0) == rule.next(alive, n) {
            counts[idx].correct += 1;
        }
    }
    std::array::from_fn(|i| (LifeCase::ALL[i], counts[i]))
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

    // The owner's complaint: "99.80% accuracy" with 13 alive predicted vs 5
    // true feels wrong. IoU, Hamming and F1 are the numbers that catch it.
    #[test]
    fn iou_and_f1_punish_overshooting_a_small_true_set() {
        // 4x4 grid, true alive = {0, 1, 2, 3, 4} (5 cells), model alive =
        // {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} (13 cells), all of
        // true's cells included (so accuracy alone hides the overshoot).
        let mut true_cells = vec![0u8; 16];
        for i in 0..5 {
            true_cells[i] = 1;
        }
        let truth = Grid::from_cells(4, 4, true_cells);
        let mut model_cells = vec![0u8; 16];
        for i in 0..13 {
            model_cells[i] = 1;
        }
        let model = Grid::from_cells(4, 4, model_cells);
        let p_alive = vec![0.9f32; 16];
        let s = score(&truth, &model, &p_alive, 1);
        assert_eq!(s.wrong_cells, 8); // cells 5..=12 wrong, 8 of them
        assert_eq!(s.true_live, 5);
        assert_eq!(s.model_live, 13);
        assert!((s.recall - 1.0).abs() < 1e-9); // every true cell is covered
        assert!((s.precision - 5.0 / 13.0).abs() < 1e-9);
        assert!((s.iou - 5.0 / 13.0).abs() < 1e-9); // union is just model_live here
        assert!(s.iou < 0.4, "13-vs-5 overshoot should read far below 'looks right'");
    }

    // The owner's exact case, restated for the new per-class fields: 5 true
    // alive, model predicts 13 alive (8 false positives, 0 false negatives).
    // alive_recall reads perfect while dead_recall and IoU expose the
    // overshoot that plain accuracy hides.
    #[test]
    fn per_class_fields_expose_the_13_vs_5_overshoot() {
        let mut true_cells = vec![0u8; 16];
        for i in 0..5 {
            true_cells[i] = 1;
        }
        let truth = Grid::from_cells(4, 4, true_cells);
        let mut model_cells = vec![0u8; 16];
        for i in 0..13 {
            model_cells[i] = 1;
        }
        let model = Grid::from_cells(4, 4, model_cells);
        let p_alive = vec![0.9f32; 16];
        let s = score(&truth, &model, &p_alive, 1);

        assert_eq!(s.tp, 5);
        assert_eq!(s.fp, 8);
        assert_eq!(s.fn_, 0);
        assert_eq!(s.tn, 3); // 11 true-dead cells, 8 of them false positives

        assert!((s.alive_recall - 1.0).abs() < 1e-9);
        assert!((s.alive_precision - 5.0 / 13.0).abs() < 1e-9);
        assert!(s.dead_recall < 1.0);
        assert!((s.dead_recall - 3.0 / 11.0).abs() < 1e-9);
        assert!((s.iou - 5.0 / 13.0).abs() < 1e-9);
    }

    #[test]
    fn iou_is_one_when_both_alive_sets_are_empty() {
        let truth = Grid::new(4, 4);
        let model = Grid::new(4, 4);
        let s = score(&truth, &model, &vec![0.1f32; 16], 1);
        assert_eq!(s.iou, 1.0);
        assert_eq!(s.f1, 1.0); // precision=recall=1 -> f1=1
    }

    #[test]
    fn classify_covers_every_neighbor_count() {
        // Alive side: 0,1 -> lonely; 2,3 -> survive; 4..=8 -> crowded.
        assert_eq!(classify(true, 0), LifeCase::DeathLonely);
        assert_eq!(classify(true, 1), LifeCase::DeathLonely);
        assert_eq!(classify(true, 2), LifeCase::Survive2);
        assert_eq!(classify(true, 3), LifeCase::Survive3);
        for n in 4..=8 {
            assert_eq!(classify(true, n), LifeCase::DeathCrowded);
        }
        // Dead side: only 3 is birth, everything else stays dead.
        assert_eq!(classify(false, 3), LifeCase::Birth);
        for n in [0, 1, 2, 4, 5, 6, 7, 8] {
            assert_eq!(classify(false, n), LifeCase::StayDead);
        }
    }

    #[test]
    fn per_case_recall_counts_and_scores_a_glider_seed() {
        let rule = Rule::life();
        let mut input = Grid::new(4, 4);
        input.place_glider(0, 0);
        // Model that always answers correctly (true Life itself) should get
        // 100% on every non-empty case.
        let model = input.step(&rule);
        let cases = per_case_recall(&input, &model, &rule);
        let total: usize = cases.iter().map(|(_, c)| c.count).sum();
        assert_eq!(total, 16); // every cell classified exactly once
        for (case, c) in cases {
            if c.count > 0 {
                assert_eq!(c.fraction(), 1.0, "{:?} should be perfect against true Life", case);
            }
        }

        // A model that answers dead everywhere gets birth cases wrong (it
        // fails to birth) and stay-dead cases right (it already answers 0),
        // matching TC's "birth first" hypothesis.
        let all_dead = Grid::new(4, 4);
        let cases = per_case_recall(&input, &all_dead, &rule);
        for (case, c) in cases {
            match case {
                LifeCase::Birth => assert_eq!(c.fraction(), 0.0),
                LifeCase::StayDead | LifeCase::DeathLonely | LifeCase::DeathCrowded => {
                    if c.count > 0 {
                        assert_eq!(c.fraction(), 1.0);
                    }
                }
                _ => {}
            }
        }
    }
}
