//! Each test guards one classical-Life behavior the LLM variants are scored
//! against: if the ground truth is wrong, every accuracy number in
//! `docs/pictures/` is meaningless.

use life::{Grid, Rule};

#[test]
fn rulestring_roundtrip() {
    assert_eq!(Rule::life().to_rulestring(), "B3/S23");
    assert_eq!(Rule::parse("B36/S23").unwrap().to_rulestring(), "B36/S23");
    assert!(Rule::parse("B3S23").is_none());
    assert!(Rule::parse("B9/S23").is_none());
}

#[test]
fn block_is_still() {
    let mut g = Grid::new(8, 8);
    for (x, y) in [(2, 2), (3, 2), (2, 3), (3, 3)] {
        g.set(x, y, 1);
    }
    assert_eq!(g.step(&Rule::life()), g);
}

#[test]
fn blinker_has_period_2() {
    let mut g = Grid::new(8, 8);
    for (x, y) in [(2, 3), (3, 3), (4, 3)] {
        g.set(x, y, 1);
    }
    let one = g.step(&Rule::life());
    assert_ne!(one, g, "a blinker must not be still");
    assert_eq!(one.live_count(), 3);
    assert_eq!(one.step(&Rule::life()), g);
}

#[test]
fn glider_translates_by_one_one_every_four_generations() {
    let mut g = Grid::new(16, 16);
    g.place_glider(1, 1);
    let mut moved = g.clone();
    for _ in 0..4 {
        moved = moved.step(&Rule::life());
    }
    let mut expected = Grid::new(16, 16);
    expected.place_glider(2, 2);
    assert_eq!(moved, expected);
}

#[test]
fn neighbors_wrap_at_the_edges() {
    // Corner cell 0 of a 4x4 torus sees the opposite corner as a neighbor.
    let g = Grid::new(4, 4);
    let n = g.neighbor_indices(0);
    assert!(n.contains(&15), "NW of (0,0) on a torus is (3,3) = index 15");
    assert_eq!(n.len(), 8);
    let mut sorted = n.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 8, "the 8 neighbors must be distinct");
}

#[test]
fn random_fill_is_deterministic() {
    let a = Grid::random(32, 32, 42, 0.3);
    let b = Grid::random(32, 32, 42, 0.3);
    assert_eq!(a, b);
    assert_ne!(a, Grid::random(32, 32, 43, 0.3));
    let live = a.live_count() as f64 / (32.0 * 32.0);
    assert!((0.2..0.4).contains(&live), "density drifted: {live}");
}

#[test]
fn diff_counts_changed_cells() {
    let mut a = Grid::new(4, 4);
    let mut b = Grid::new(4, 4);
    b.set(1, 1, 1);
    b.set(2, 2, 1);
    assert_eq!(a.diff(&b).iter().map(|&d| d as usize).sum::<usize>(), 2);
    a.set(1, 1, 1);
    assert_eq!(a.diff(&b).iter().map(|&d| d as usize).sum::<usize>(), 1);
}
