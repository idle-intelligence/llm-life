# PyTorch retrain, parity against the published Burn checkpoints

Machine: Mac (CPU only, no MPS/GPU used — this repo's inference GPU is
shared with another job). Base model: `qwen2.5-0.5b-instruct-q4_0.gguf`,
loaded through `transformers`' own GGUF loader. Data: `pytorch/life.py`'s
xorshift64 port of `crates/life`, checked bit-for-bit against the Rust RNG.

See `docs/pytorch-training.md` for the bridge (`tools export`) and how the
oracle files were produced.

## Parity (PyTorch forward vs the exact Burn `TrainModel`/tiny-model forward,
same published weights)

| model | comparison | max abs diff |
|---|---|---|
| BERT of Life | logits, 512 exhaustive cases | 1.85e-6 |
| 9-number MLP (2-layer) | logits, 512 exhaustive cases | 7.63e-6 |
| stencil (grid-to-grid) | logits, 16x16 held-out grid | 5.72e-6 |
| LoRA variant B (`lora-b-16-s3-300`) | p(alive), 16x16 grid | 1.43e-7 |
| LoRA variant B (`lora-b-16-s3-300`) | logits, all 324 positions | 1.72e-4 |
| LoRA variant A, rules (`lora-a-rules-300`) | logits, 64-case chunk | 3.47e-4 |
| LoRA variant A, no-rules (`lora-a-norules-300`) | logits, 64-case chunk | 3.05e-4 |

The `life.py` RNG port also reproduces the Rust `Grid::random`/`step` output
bit-for-bit on the fixed oracle grid (seed 1,000,000, density 0.28).

## Retrain from scratch: stencil-life (tiny models)

Same architecture, same parameter counts as the published checkpoints,
trained on the same exhaustive-512-case split (BERT, MLP) or density sweep
(stencil).

| model | parameters | steps to 512/512 | held-out acc | mean IoU (rollout) |
|---|---|---|---|---|
| BERT of Life | 3,490 | 140 | 1.0 | 1.0 |
| 9-number MLP | 1,442 | 355 | 1.0 | 1.0 |
| stencil (grid-to-grid) | 3,329 | n/a (BCE, not classification) | n/a | 1.0 |

Published (Burn, `docs/runs/2026-09-20-bert.md`, `2026-09-20-vector.md`):
all three reach 256/256 cells correct at 16x16 in the README's headline
table. This retrain matches: 100% held-out accuracy and IoU 1.0 on the
rollout for all three.

## Retrain from scratch: LoRA variant B (whole grid, 16x16)

Hyperparameters match the published run
(`docs/runs/2026-09-20-ft-b-16-s3-300.md`): rank 8, alpha 16 (q/k/v/o),
300 steps, lr 5e-5, seed 3.

| | loss | accuracy | IoU | alive recall | dead recall |
|---|---|---|---|---|---|
| base (step 0) | 0.486 | 0.765 | 0.000 | 0.000 | 1.000 |
| after 300 steps | 0.0004 | 1.000 | 1.000 | 1.000 | 1.000 |

Published (Burn, same hyperparameters): reaches 256/256 cells correct at
16x16 (README's "LLM whole grid (trained)" row). This retrain: 100% held-out
accuracy/IoU on 8 held-out grids after 300 steps — matches.

Wall clock: 927s / 300 steps on CPU (~3.1s/step, batched forward+backward
over a 324-token sequence). Rust's published run measured 92.4s/300 steps on
a shared 3080 GPU (`docs/runs/2026-09-20-ft-b-16-s3-300.md`), i.e. Burn on
GPU is roughly 10x faster than this port on CPU — expected, and not a
framework comparison (no GPU backend was used here; see "What is not done"
below).

Sanity check: this retrain's step-1 loss (0.2138) matches the published
Burn run's step-1 loss exactly (same value, same seed) — see
`docs/pytorch-training.md` for why that is expected rather than a
coincidence, and why it is strong evidence the two frameworks' forward math
agrees end to end.

## Retrain from scratch: LoRA variant A (per-cell, exhaustive 512)

Hyperparameters match the published runs
(`docs/runs/2026-09-20-ft-a-{rules,norules}-300.md`): rank 8, alpha 16
(q/k/v/o), batch 64, lr 3e-5, seed 1.

Note on the published "step 0 (base)" numbers: the Burn CLI's
`--eval-cases` default is 128, so the published run docs' step-0 print
covers only the first 128 of the 512 exhaustive cases (all "self dead" —
`case_table`'s `survive-2`/`survive-3`/`death-lonely`/`death-crowded` rows
read `n=0` in that print for exactly this reason). This port's `evaluate()`
always uses the full 512, so its step-0 numbers are not directly comparable
to the published step-0 line; the **final/best-checkpoint** numbers in both
are always over the full 512 and are the ones to compare.

<!-- filled in once the retrain finishes -->
