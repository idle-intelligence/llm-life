//! Batched-grid forward (the `stepGrid` gather the compare page's BERT/MLP
//! rungs need, per docs/runs/2026-09-20-compare-batched.md) must label every
//! cell exactly the way the existing per-cell loop does. Same gather
//! (`bert::data::case_at`, toroidal via `Grid::neighbor_indices`), same
//! model, same argmax — only whether it's one `[n, 9]` forward or `n` `[1,
//! 9]` forwards should differ.

use burn::backend::Wgpu;
use burn::tensor::{Int, Tensor, TensorData};
use life::Grid;
use llm_life::bert::data::case_at;
use llm_life::bert::model::BertConfig;
use llm_life::vector::model::Mlp2Config;

type B = Wgpu;

fn device() -> <B as burn::tensor::backend::Backend>::Device {
    Default::default()
}

fn argmax2(v: &[f32], row: usize) -> usize {
    if v[2 * row] >= v[2 * row + 1] {
        0
    } else {
        1
    }
}

#[test]
fn bert_batched_matches_per_cell_labels() {
    let device = device();
    let cfg = BertConfig::small(8, 1, 1);
    let model = cfg.init::<B>(&device);

    for size in [16usize, 32] {
        let grid = Grid::random(size, size, 7, 0.3);
        let n = grid.cells().len();

        // Batched: one [n, 9] forward.
        let mut toks: Vec<i64> = Vec::with_capacity(n * 9);
        for i in 0..n {
            toks.extend(case_at(&grid, i).iter().map(|&b| b as i64));
        }
        let t: Tensor<B, 2, Int> = Tensor::from_data(TensorData::new(toks, [n, 9]), &device);
        let batched = model.forward(t).into_data().into_vec::<f32>().unwrap();

        // Per-cell: n independent [1, 9] forwards.
        for i in 0..n {
            let toks: Vec<i64> = case_at(&grid, i).iter().map(|&b| b as i64).collect();
            let t: Tensor<B, 2, Int> = Tensor::from_data(TensorData::new(toks, [1, 9]), &device);
            let per_cell = model.forward(t).into_data().into_vec::<f32>().unwrap();
            assert_eq!(
                argmax2(&batched, i),
                argmax2(&per_cell, 0),
                "bert: cell {i} disagrees at grid size {size}"
            );
        }
    }
}

#[test]
fn vec_mlp_batched_matches_per_cell_labels() {
    let device = device();
    let cfg = Mlp2Config::new(8);
    let model = cfg.init::<B>(&device);

    for size in [16usize, 32] {
        let grid = Grid::random(size, size, 11, 0.3);
        let n = grid.cells().len();

        let mut bits: Vec<f32> = Vec::with_capacity(n * 9);
        for i in 0..n {
            bits.extend(case_at(&grid, i).iter().map(|&b| b as f32));
        }
        let t: Tensor<B, 2> = Tensor::from_data(TensorData::new(bits, [n, 9]), &device);
        let batched = model.forward(t).into_data().into_vec::<f32>().unwrap();

        for i in 0..n {
            let bits: Vec<f32> = case_at(&grid, i).iter().map(|&b| b as f32).collect();
            let t: Tensor<B, 2> = Tensor::from_data(TensorData::new(bits, [1, 9]), &device);
            let per_cell = model.forward(t).into_data().into_vec::<f32>().unwrap();
            assert_eq!(
                argmax2(&batched, i),
                argmax2(&per_cell, 0),
                "vec-mlp: cell {i} disagrees at grid size {size}"
            );
        }
    }
}
