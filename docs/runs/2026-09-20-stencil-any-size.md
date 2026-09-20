# stencil grid-to-grid: O(n^2) mask -> O(9n) gather

Machine: M2 laptop (native, wgpu/Metal backend). Commit: this change (see git log).

## Bug

`vector::model::stencil_mask(width, height, device)` built a dense additive
`[n, n]` f32 mask, `n = width*height`: 67 MB at 64², ~1 GB at 128², ~17 GB at
256² — the browser tab OOMs before the grid gets that large. Every query
token only ever attends to its own 9 `life::Grid::neighbor_indices` taps
(self + 8 neighbours, all always in-bounds on the toroidal grid), so the
dense `[n, n]` score/mask tensor never needed to exist.

## Fix

`stencil_mask` -> `stencil_neighbors`: a flat `[n * 9]` int index tensor
(row `i`'s 9 taps at `[i*9..i*9+9]`, its 8 neighbours then itself) instead of
a dense mask. `StencilBlock::attention` gathers k/v at those 9 taps per
query (`Tensor::select` along the sequence dim) and computes a `[.., 9]`
softmax instead of a `[.., n, n]` one — O(9n) memory and compute, not
O(n^2). The checkpoint format is unaffected: the mask/table was always
rebuilt from `width, height` at load time, never stored in the checkpoint,
so `stencil-d16-L1.bin` / `stencil-d32-L2.bin` load unchanged.

Call sites updated: `crates/llm-life/src/web.rs` (`VecStencilEngine::step_grid`),
`crates/llm-life/src/bin/llm-life.rs` (`bench_vec_native`, `bench_ladder` —
the latter already named `stencil_neighbors` and capped at
`--stencil-max-size` "pending the in-progress fix" in the preceding
`bench-ladder` commit), `crates/llm-life/src/vector/train.rs` (`train_stencil`).

## Equivalence

`crates/llm-life/src/vector/model.rs`'s `tests` module reproduces the old
dense-mask attention verbatim (same weights, same softmax) and compares it
against the new gather path on random grids, NdArray backend:

| grid | max abs diff (logits) | tolerance |
|---|---|---|
| 16x16 | < 1e-6 | 1e-6 |
| 32x32 | < 1e-6 | 1e-6 |

`cargo test --features native,cpu -p llm-life --lib vector::model::tests`:
2 passed.

## 32² score before/after

`train-vec --seed 1` (Stencil d=16 L=1 H=1, 3329 params), stencil-steps
1200, lr 3e-3 — same command, same seed, before (dense mask,
docs/runs/2026-09-20-vector.md) and after (this fix) the rewrite:

| | steps to IoU>=0.999 | train IoU | IoU 16² (5 gens x 3 seeds) | IoU 32² (5 gens x 3 seeds) |
|---|---|---|---|---|
| before (dense mask) | 80 | 1.0000 | 1.000 (all) | 1.000 (all) |
| after (gather, this fix) | 80 | 1.0000 | 1.000 (all) | 1.000 (all) |

Identical steps-to-converge and identical exact IoU at both sizes — the
CONCEPT.md §12 generalisation claim (16²-trained weights, freshly-built
neighbour table, exact at 32²) is unchanged by the rewrite.

## Native forward timing, `bench-vec --kind stencil` (stencil-d16-L1.bin)

Before this fix, 128²/256² were not reachable at all (1 GB / 17 GB mask
allocation). After:

| grid | cells | median s/gen | ms/cell |
|---|---|---|---|
| 16x16 | 256 | 0.001708 | 0.0067 |
| 32x32 | 1024 | 0.001404 | 0.0014 |
| 64x64 | 4096 | 0.002165 | 0.0005 |
| 128x128 | 16384 | 0.005799 | 0.0004 |
| 256x256 | 65536 | 0.020782 | 0.0003 |

Command: `llm-life bench-vec --checkpoint artifacts/vector/stencil-d16-L1.bin
--kind stencil --d-model 16 --n-layers 1 --n-heads 1 --sizes
16,32,64,128,256 --reps 5`. s/gen grows roughly linearly with `n` (not
quadratically) once past small-size dispatch-overhead noise, as expected
for an O(9n) forward — milliseconds even at 256², where the old dense mask
could not run at all.

## Verification commands run

- `cargo check --features native -p llm-life`
- `cargo clippy --features native -p llm-life -- -D warnings`
- `cargo clippy --features web -p llm-life --target wasm32-unknown-unknown -- -D warnings`
- `cargo test --features native,cpu -p llm-life --lib vector::model::tests`
- `wasm-pack build crates/llm-life --target web --out-dir <scratch> --no-default-features --features web` (built to a scratch directory, not copied into `web/`, since another session owns `web/` in this run — the real command to publish it is `wasm-pack build crates/llm-life --target web --out-dir ../../web/pkg-llm --no-default-features --features web`, per `README.md`)
