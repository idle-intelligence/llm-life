//! The block-diagonal packing: if a cell can see another cell's tokens, or
//! its positions don't restart, variant A is not "4096 independent prompts"
//! and the picture means something else than the report claims.

use life::Grid;
use llm_life::variant_a::{cell_prompt, cell_prompts, pack_chunk, p_alive_chunk};

const P: usize = 5;

/// Three cells with prompts of 2, 3 and 2 tokens.
fn chunk() -> llm_life::variant_a::Chunk {
    let lens = [2usize, 3, 2];
    pack_chunk(&[7, 8, 9], |c| vec![c as u32; lens[c - 7]], P)
}

#[test]
fn positions_restart_at_the_prefix_for_every_cell() {
    let c = chunk();
    assert_eq!(c.positions, vec![5, 6, 5, 6, 7, 5, 6]);
}

#[test]
fn a_cell_sees_the_prefix_and_its_own_block_and_nothing_else() {
    let c = chunk();
    let t = c.len();
    let kv = P + t;
    // Row 3 = second token of cell 8's block (block starts at 2).
    let row = &c.allowed[3 * kv..4 * kv];
    assert!(row[..P].iter().all(|&a| a), "the resident prefix is visible");
    assert!(row[P + 2] && row[P + 3], "causal within the block");
    assert!(!row[P + 4], "not its own future token");
    assert!(!row[P] && !row[P + 1], "cell 7's tokens are invisible");
    assert!(!row[P + 5] && !row[P + 6], "cell 9's tokens are invisible");
    assert_eq!(row.iter().filter(|&&a| a).count(), P + 2);
}

#[test]
fn every_row_allows_at_least_one_key() {
    // A fully masked row softmaxes to NaN.
    let c = chunk();
    let kv = P + c.len();
    for i in 0..c.len() {
        assert!(c.allowed[i * kv..(i + 1) * kv].iter().any(|&a| a), "row {i}");
    }
}

#[test]
fn the_answer_row_is_the_last_token_of_each_block() {
    let c = chunk();
    assert_eq!(c.answer_rows(), vec![1, 4, 6]);
}

#[test]
fn p_alive_is_read_at_the_answer_rows_in_cell_order() {
    let c = chunk();
    let mut logits = vec![0.0f32; c.len() * 2];
    for (i, r) in c.answer_rows().iter().enumerate() {
        logits[r * 2] = if i == 1 { -6.0 } else { 6.0 };
        logits[r * 2 + 1] = if i == 1 { 6.0 } else { -6.0 };
    }
    let p = p_alive_chunk(&logits, &c);
    assert_eq!(p.iter().map(|&(c, _)| c).collect::<Vec<_>>(), vec![7, 8, 9]);
    assert!(p[0].1 < 0.01 && p[2].1 < 0.01);
    assert!(p[1].1 > 0.99);
}

#[test]
fn the_prompt_states_the_eight_neighbors_then_self() {
    assert_eq!(
        cell_prompt(&[1, 0, 1, 0, 0, 1, 1, 0], 1),
        "Neighbors: 1 0 1 0 0 1 1 0 / Self: 1 / Next:"
    );
}

#[test]
fn every_cell_gets_a_prompt_holding_its_own_neighborhood() {
    let mut g = Grid::new(4, 4);
    g.set(1, 1, 1);
    g.set(2, 1, 1);
    let prompts = cell_prompts(&g);
    assert_eq!(prompts.len(), 16);
    // Cell (1,1) is alive and has one live neighbor, (2,1).
    assert!(prompts[5].ends_with("/ Self: 1 / Next:"));
    assert_eq!(prompts[5].matches(" 1").count(), 2);
}
