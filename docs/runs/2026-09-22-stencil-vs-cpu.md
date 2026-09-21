# Stencil model on GPU vs exact Life on one CPU core

Question: at large grids, does the learned stencil model (`stencil-d16-L1`,
`StencilConfig::new(16,1,1)`, whole-grid O(9n) `stencil_neighbors` gather,
`crates/llm-life/src/vector/model.rs`) beat the exact B3/S23 rule
(`life::Grid::step`, `crates/life/src/grid.rs`) run on one CPU thread — and
where, if anywhere, is the crossover.

## Setup

- Checkpoint: `web/stencil-d16-L1.bin`.
- Grid: `Grid::random(size, size, 1, 0.28)`, toroidal boundary, sizes 256,
  512, 1024, 2048, 4096.
- Model timing: warm-up 3 forwards, then median of 10, with the
  host readback (`.into_data()`) inside the timed region on every rep.
- CPU timing: median of 10 calls to `life::Grid::step`, single thread, no
  warm-up.
- Correctness: model logits thresholded at 0 (alive if logit > 0), compared
  cell-by-cell against `life::Grid::step`'s output on the same grid
  (`wrong_cells` column).
- Scratch binary `crates/llm-life/src/bin/bench-sizes.rs`, written for this
  run and removed afterwards; the model, grid and rule code are the
  committed ones.
- Command: `cargo build --release --features native -p llm-life --bin
  bench-sizes` then `./target/release/bench-sizes`, run from the repo root.
- Box: a Linux desktop with an RTX 3080, wgpu on Vulkan. Run as
  `systemd-run --user` unit `bench-sizes-run`, one job on the GPU, nothing
  else running (`nvidia-smi` at 0% util / 519 MiB used before launch, no
  other `chain-`/`bench-`/`judge-`/`dl-` units active).
- Mac: Apple M2, wgpu on Metal. Run natively, nothing else running.

## Results — RTX 3080 (Vulkan)

| size | cells | model ms/gen | rule ms/gen | model Mcells/s | rule Mcells/s | ratio (rule/model) | wrong cells |
|---|---|---|---|---|---|---|---|
| 256 | 65536 | 17.2067 | 0.9129 | 3.8087 | 71.7901 | 18.85 | 0 |
| 512 | 262144 | 7.8826 | 7.2605 | 33.2560 | 36.1055 | 1.09 | 0 |
| 1024 | 1048576 | 28.2163 | 14.2662 | 37.1620 | 73.5006 | 1.98 | 0 |
| 2048 | 4194304 | FAILED | — | — | — | — | — |
| 4096 | 16777216 | FAILED | — | — | — | — | — |

Source: `/tmp`.

## Results — Apple M2 (Metal)

| size | cells | model ms/gen | rule ms/gen | model Mcells/s | rule Mcells/s | ratio (rule/model) | wrong cells |
|---|---|---|---|---|---|---|---|
| 256 | 65536 | 20.8307 | 1.2825 | 3.1461 | 51.1002 | 16.24 | 0 |
| 512 | 262144 | 82.9852 | 1.1381 | 3.1589 | 230.3297 | 72.92 | 0 |
| 1024 | 1048576 | 330.3761 | 4.5783 | 3.1739 | 229.0301 | 72.16 | 0 |
| 2048 | 4194304 | 2985.2156 | 18.5621 | 1.4050 | 225.9608 | 160.83 | 0 |
| 4096 | 16777216 | FAILED | — | — | — | — | — |

Source: `/tmp`.

## Observations

- No crossover on either machine: the exact rule on one CPU core is faster
  than the GPU stencil model at every size that ran to completion.
- The 3080 gets closest at 512x512 (ratio 1.09, `bench-sizes-3080.log` line
  2), where the model's 33.26 Mcells/s is within 9% of the rule's 36.11
  Mcells/s; at 1024x1024 the gap widens back out to 1.98x.
- On the M2 the model's throughput is flat around 3.1-3.2 Mcells/s from
  256 to 1024 and drops to 1.41 Mcells/s at 2048, while the rule holds
  roughly 225-230 Mcells/s from 512 up — the ratio only grows with size
  (16x at 256 to 161x at 2048).
- Every size that completed was exact (`wrong_cells` = 0 in both logs):
  the model's argmax-over-0 threshold matches `life::Grid::step` cell for
  cell, so the timing comparison is apples to apples.
- 2048 failed on the 3080 and 4096 failed on both machines with the same
  cubecl/wgpu error, `can't allocate buffer of size: N` — a single-buffer
  allocation, not overall VRAM exhaustion (the 3080 reported 519 MiB used
  out of 10240 MiB before the run). N matches the stencil attention block's
  k/v neighbour-gather tensor exactly: `n * 9 * d_model * 4` bytes with
  `d_model = 16` (`StencilBlock::attention`'s `k_nb`/`v_nb`, shape
  `[1, 1, n, 9, 16]` f32) — `4194304 * 9 * 16 * 4 = 2415919104` at 2048 (the
  3080's failure) and `16777216 * 9 * 16 * 4 = 9663676416` at 4096 (both
  machines' failure). The O(9n) gather is still O(9n) in cell count but
  each cell's gathered neighbourhood carries the full `d_model` channel
  width, so the constant is large enough to hit a single-allocation limit
  well before either machine's grid gets anywhere near where the model
  might have caught up on throughput.
- The scratch binary (`crates/llm-life/src/bin/bench-sizes.rs`) has been
  removed from both the Mac repo and the box checkout; it is not part of
  this commit.
