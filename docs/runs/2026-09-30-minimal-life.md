# How small can a network be and still learn Game of Life exactly?

Question: what is the smallest network, in parameters, that learns the
Conway B3/S23 update rule exactly (not "mostly right")? A quick "just see
what happens" sweep, not a release-gated result.

Reference: Springer & Kenyon, "It's Hard for Neural Networks To Learn the
Game of Life" (arXiv:2009.01398). Its abstract: a 2n+1-layer CNN is the
theoretical minimum to implement n steps of the rule, but minimal networks
rarely converge in practice, near-minimal networks are fragile (a single
sign flip on one weight can break training), and there is a training
density d0 that strongly affects convergence odds. The full text (concrete
parameter counts for a minimal n=1 architecture) was not fetched — WebFetch
on the abstract page did not surface architecture specifics, so the model
family below is defined here, not copied from the paper.

## Setup

- Data generator, torus handling, and scoring: this repo's PyTorch port
  (`pytorch/life.py`'s `Grid`/`Rule`, xorshift64-seeded, toroidal boundary;
  `pytorch/score.py`), the same code the PyTorch retrain of the published
  stencil-life models uses (`docs/pytorch-training.md`). New code for this
  run: `pytorch/minimal_life.py`.
- Compute: a Linux desktop with an RTX 3080 (used its CPU — these models
  and batches are small enough that per-step Python overhead, not matmul
  throughput, dominates; a single `systemd-run --user` unit under
  `flock box.lock`, GPU confirmed idle throughout).
- Model families:
  - **CNN**: `Conv3x3(1->c, circular padding, bias) -> ReLU -> Conv1x1(c->1, bias)`,
    logit -> sigmoid. Params = 11c + 1. c=2 is "minimal" (m=1x): the
    smallest channel count for which a 3x3-to-1x1 map can represent both a
    neighbour-count feature and a self-state feature. c in {4, 8, 16} are
    the x2/x4/x8 widths.
  - **MLP-9**: the 9 neighbourhood values (8 neighbours + self, `life.py`'s
    `neighbor_indices` order) -> `Linear(9,h)` -> ReLU -> `Linear(h,1)` ->
    sigmoid. h in {2, 4, 8}, for comparison against the repo's published
    1,442-param 2-layer/hidden-32 MLP (`stencil_models.Mlp2OfLife`).
  - **n=2 DeepCNN**: two stacked `Conv3x3(circular)` + ReLU blocks (5x5
    receptive field) at channel width c, then a `Conv1x1` readout, trained
    to predict the state two rule-applications ahead in one forward pass.
    c in {2, 4, 8} (x1/x2/x4 of the n=1 minimal width).
- Training: Adam (0.9, 0.999, eps 1e-8), lr 3e-3, batch 8 random torus
  boards per step, board size 16x16, up to 2,500 steps (2,000 for the n=2
  conditions). 10 seeds per condition. Default training density 0.375; a
  d0 sweep over {0.2, 0.38, 0.5} run only for the minimal (c=2) CNN.
- Convergence: exact on all 512 3x3 neighbourhoods (the full truth table
  for a 1-step rule) **and** 100% of cells correct on 100 fresh random
  16x16 boards, evaluated every 25 training steps. The 512-case table is
  not meaningful for the n=2 models (a 2-step transition depends on the
  5x5 neighbourhood, 2**25 cases), so n=2 convergence is fresh-board cell
  accuracy only.

## Results

n=1, default training density (0.375), 10 seeds, up to 2,500 steps:

| model | params | converged / 10 | median steps to converge | best cells-correct (non-converged seeds) |
|---|---:|---:|---:|---:|
| CNN minimal (c=2) | 23 | 6 | 750 | 0.784 |
| CNN x2 (c=4) | 45 | 7 | 825 | 0.954 |
| CNN x4 (c=8) | 89 | 10 | 662.5 | - |
| CNN x8 (c=16) | 177 | 10 | 550 | - |
| MLP-9 hidden=2 | 23 | 4 | 800 | 0.790 |
| MLP-9 hidden=4 | 45 | 7 | 1250 | 0.996 |
| MLP-9 hidden=8 | 89 | 10 | 812.5 | - |

Training-density sweep, CNN minimal (c=2) only:

| d0 | converged / 10 | median steps to converge | best cells-correct (non-converged) |
|---|---:|---:|---:|
| 0.20 | 1 | 2325 | 0.952 |
| 0.38 | 6 | 787.5 | 0.786 |
| 0.50 | 2 | 2100 | 0.922 |

n=2 (predict two rule steps in one forward pass), default density, 10
seeds, up to 2,000 steps:

| model | params | converged / 10 | median steps to converge | best cells-correct (non-converged) |
|---|---:|---:|---:|---:|
| DeepCNN x1 (c=2) | 61 | 0 | - | 0.721 |
| DeepCNN x2 (c=4) | 193 | 0 | - | 0.773 |
| DeepCNN x4 (c=8) | 673 | 0 | - | 0.846 |

Raw per-condition JSON (including per-seed step counts and best accuracy):
`docs/runs/2026-09-30-minimal-life-results.json`.

## Observations

- The smallest network that reliably (10/10 seeds) learned the exact n=1
  rule in this sweep is the CNN at c=8 (89 parameters), tied by the MLP-9
  at hidden=8 (also 89 parameters). The theoretical-minimum-sized CNN
  (c=2, 23 parameters) converges only 6/10 seeds; c=4 (45 parameters) 7/10.
  This matches Springer & Kenyon's headline claim in direction: the
  literal minimal architecture is not reliable, and reliability arrives
  with moderate overcompleteness, not at the theoretical floor.
- The repo's published 9-number MLP (`Mlp2OfLife`, hidden=32, 1,442
  parameters) is far larger than needed to hit 100% reliability here
  (hidden=8, 89 parameters, was already 10/10); the published model was
  presumably sized for training convenience or as a from-scratch NLP-style
  reference architecture, not for parameter-minimality.
- Training density visibly gates convergence for the minimal CNN: d0=0.38
  (close to the repo's `sample_grid` default range) converges more often
  and faster (6/10, median 787.5 steps) than d0=0.2 or d0=0.5 (1/10 and
  2/10, both with medians above 2,000 steps for the seeds that did
  converge). This is the same density effect the paper's abstract
  describes, though the specific optimum here (mid-range density) was not
  compared against the paper's own numbers (not fetched).
- No n=2 model converged to exact within 2,000 steps at any width tried
  (up to 673 parameters, x4 the n=1-minimal width) under this short
  budget; best-effort cell accuracy did increase with width (0.72 to 0.85
  from c=2 to c=8), suggesting more steps or more width would eventually
  close the gap, consistent with the paper's "substantially more
  parameters than the theoretical minimum" framing for multi-step
  prediction. Given the ≤60-minute compute budget, this was not pushed
  further; it would be the natural next iteration were this promoted from
  a "just see what happens" experiment to a tracked one.
- All training ran on CPU (a Linux desktop's Ryzen, not its GPU): with
  batch size 8 and boards this small, per-step Python-level grid
  generation and scoring, not matmul throughput, dominated wall time, so
  GPU launch overhead would have made it slower, not faster, for this
  sweep.
