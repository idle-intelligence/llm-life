//! The packing and the stencil mask: if these are wrong, the model is being
//! asked a different question than the report claims.

use life::{Grid, Rule};
use llm_life::variant_b::{argmax_grid, p_alive, pack, rules_prefix};

const DEAD: u32 = 15;
const ALIVE: u32 = 16;

fn packed_4x4() -> (Grid, llm_life::variant_b::Packed, usize) {
    let mut g = Grid::new(4, 4);
    g.set(1, 1, 1);
    g.set(2, 2, 1);
    let prefix = vec![100, 101, 102];
    let p = pack(&g, &prefix, DEAD, ALIVE);
    let t = p.len();
    (g, p, t)
}

#[test]
fn tokens_are_prefix_then_one_token_per_cell() {
    let (g, p, t) = packed_4x4();
    assert_eq!(t, 3 + 16);
    assert_eq!(p.grid_start, 3);
    assert_eq!(&p.tokens[..3], &[100, 101, 102]);
    for c in 0..16 {
        let expected = if g.cells()[c] != 0 { ALIVE } else { DEAD };
        assert_eq!(p.tokens[3 + c], expected, "cell {c}");
    }
}

#[test]
fn grid_tokens_share_one_position_and_the_prefix_keeps_its_own() {
    let (_, p, _) = packed_4x4();
    assert_eq!(&p.positions[..3], &[0, 1, 2]);
    assert!(p.positions[3..].iter().all(|&x| x == 3), "bag mode: one position for every cell");
}

#[test]
fn a_cell_sees_the_prefix_itself_and_exactly_eight_neighbors() {
    let (g, p, t) = packed_4x4();
    // Cell 5 = (1,1), interior of a 4x4 torus.
    let row = &p.allowed[(3 + 5) * t..(3 + 6) * t];
    assert!(row[..3].iter().all(|&a| a), "every cell must see the whole prefix");
    assert!(row[3 + 5], "a cell must see itself");
    for j in g.neighbor_indices(5) {
        assert!(row[3 + j], "cell 5 must see neighbor {j}");
    }
    assert_eq!(row.iter().filter(|&&a| a).count(), 3 + 1 + 8);
}

#[test]
fn a_cell_does_not_see_a_non_neighbor_even_though_it_is_earlier() {
    let (g, p, t) = packed_4x4();
    // Cell 15 is the last token in the sequence; cell 5 precedes it but is
    // not one of its 8 neighbors on this torus.
    assert!(!g.neighbor_indices(15).contains(&5));
    assert!(!p.allowed[(3 + 15) * t + 3 + 5], "the stencil is not causal-plus, it is a stencil");
}

#[test]
fn the_prefix_stays_causal() {
    let (_, p, t) = packed_4x4();
    assert!(p.allowed[t], "prefix token 1 sees token 0");
    assert!(!p.allowed[1], "prefix token 0 must not see token 1");
    assert!(!p.allowed[3 + 5], "prefix token 0 must not see a cell");
}

#[test]
fn the_rules_prefix_quotes_the_rulestring() {
    assert!(rules_prefix(&Rule::life()).contains("B3/S23"));
    assert!(rules_prefix(&Rule::parse("B36/S23").unwrap()).contains("B36/S23"));
}

#[test]
fn p_alive_reads_the_two_answer_columns_at_the_cell_rows() {
    // [T, 2] logits: prefix rows are junk, cell rows alternate confident
    // dead / confident alive.
    let grid_start = 2;
    let n = 4;
    let mut logits = vec![0.0f32; (grid_start + n) * 2];
    for c in 0..n {
        let (d, a) = if c % 2 == 0 { (5.0, -5.0) } else { (-5.0, 5.0) };
        logits[(grid_start + c) * 2] = d;
        logits[(grid_start + c) * 2 + 1] = a;
    }
    let p = p_alive(&logits, grid_start, n);
    assert!(p[0] < 0.001 && p[2] < 0.001);
    assert!(p[1] > 0.999 && p[3] > 0.999);
    assert_eq!(argmax_grid(&p, 2, 2).cells(), &[0, 1, 0, 1]);
}

#[test]
fn fewshot_prefix_keeps_the_examples_before_the_grid_lead_in() {
    // The examples have to land *before* `Grid:`, or the first cell token no
    // longer follows the lead-in and the packing means something else.
    let p = llm_life::variant_b::fewshot_rules_prefix(&Rule::life());
    assert!(p.ends_with("Grid:\n"));
    assert_eq!(p.matches("Grid:\n").count(), 1);
    assert!(p.contains("Examples:\n"));
    assert_eq!(p.matches("Next: ").count(), 6);
    assert!(p.find("Examples:").unwrap() < p.find("Grid:").unwrap());
}
