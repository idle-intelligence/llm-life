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

## Follow-up: n = 2, 3, 10 in one pass (2026-09-30, later the same day)

Follow-up question: what about predicting 3 or 10 rule-applications ahead
in a single forward pass, instead of 1 or 2? Two model families, this
time trained on the Linux desktop's GPU (RTX 3080) with a vectorized numpy
rule-stepper for target generation (`minimal_life_nstep.vec_step`, checked
bit-for-bit against `life.py`'s `Grid.step` before trusting it, since the
per-cell Python loop that exactness check depends on is far too slow to
call n times per training sample at batch size 32):

- **A, untied**: `DeepCNN`, L = n+1 stacked `Conv3x3(circular)+ReLU` layers
  then a `Conv1x1` readout, one forward pass predicts the n-step-ahead
  state directly. Widths c in {8, 32, 64}, later {8, 32, 64, 128} once the
  budget was lifted (see below).
- **B, weight-tied recurrent**: the n=1 minimal-reliable `MinimalCNN`
  (channels=8, 89 params) applied n times with shared weights, trained
  end-to-end on the n-step target only (no 1-step supervision). Intermediate
  states go through a straight-through estimator: forward value is the hard
  threshold of the sigmoid output (so the recurrence is the actual binary
  board-to-board dynamics, not a continuous relaxation), backward pass uses
  the sigmoid's own gradient. After training, the tied core is checked
  against the exact rule on all 512 3x3 neighbourhoods
  (`minimal_life.cnn_512_eval`) to see whether it discovered the Game of
  Life update rule itself.

Data: same `Grid.random` (xorshift64) initial boards as the n=1 sweep,
density 0.38 (the best-converging density from that sweep's d0 sweep).

This section covers two runs against this same question, in order:

1. A first, budget-capped pass (`pytorch/minimal_life_nstep.py`, ≤90 min
   wall clock including everything else that session did): widths {8, 32,
   64}, 5 seeds, up to 20,000/2,000 steps (A/n=2,3 vs the rest), a
   hopeless-vs-trivial-baseline early stop, and a hard global wall-clock
   deadline. Cut short after n=2 (all widths) and most of n=3 to make room
   for the pattern-eval work below; its numbers are superseded by run 2
   for every condition run 2 repeats, but n=3 c=64 and all of n=10 for
   this exact protocol only exist in run 1's raw JSON
   (`docs/runs/2026-09-30-minimal-life-nstep-results.json`).
2. A second, budget-lifted pass (`pytorch/minimal_life_full.py`, once the
   wall-clock cap was lifted): the same two families, but stopping on a
   plateau criterion instead of a wall-clock deadline (100,000-step ceiling, stop
   when best-cells-correct hasn't improved by 0.5% over the last 10 evals
   of 500 steps each — "train long enough to see a trend, stop when
   flat"), plus a c=128 width for n in {3, 10}, plus a structured-pattern
   training variant (25% of training boards replaced by a known Game of
   Life pattern, see the next section) on the best-converging pure-random
   width per n. This is the run the tables below report unless marked
   otherwise.

### Results, n = 2 (pattern-mixed variant: best pure-random width was c=32)

| model | params | converged / 5 | median steps to converge | mean final cells-correct (non-conv.) |
|---|---:|---:|---:|---:|
| A untied DeepCNN c=8 | 1,257 | 5 | 3500 | - |
| A untied DeepCNN c=32 | 18,849 | 5 | 2000 | - |
| A untied DeepCNN c=64 | 74,561 | 5 | 2000 | - |
| B tied MinimalCNN c=8 | 89 | 1 | 1000 | 0.747 |
| A untied DeepCNN c=32, +25% pattern-mixed training | 18,849 | 5 | 1500 | - |

All-dead trivial baseline on the random-soup eval at n=2: 0.685 cells
correct. B's `rule_discovery_512_acc_per_seed`: [0.713, 1.0, 0.502, 0.549,
0.607] — one of the five seeds discovered the exact single-step rule
(1.0 on all 512 neighbourhoods) despite never being supervised on it
directly, only on the 2-step composite; the other four learned some other
2-step-consistent function that is not the Life rule applied twice.

Structured-pattern eval (exact boards / total; all converged A widths tied
at 1.0 on every family so only one row is shown; B tied's one converged
seed contributes exactly 1/5 of each total, matching its 0.2 fractions):

| pattern | A (any converged width) exact/total | B tied exact/total (masked cells-correct) |
|---|---:|---:|
| block | 20/20 | 4/20 (0.20) |
| beehive | 40/40 | 8/40 (0.20) |
| blinker | 40/40 | 8/40 (0.20) |
| toad | 80/80 | 16/80 (0.20) |
| beacon | 40/40 | 8/40 (0.20) |
| glider | 160/160 | 32/160 (0.20) |
| lwss | 160/160 | 32/160 (0.20) |
| r_pentomino | 160/160 | 32/160 (0.20) |

Observation: every converged untied n=2 model generalizes perfectly to
gliders, LWSS, R-pentomino and all four still lifes/oscillators, at 4
random positions and all dihedral orientations, on a 32x32 board it never
saw a board that size during training (16x16) — consistent with the
architecture being a pure local rule (circular conv), not a
memorized-training-distribution fit. The pattern-mixed training variant
converged in fewer median steps (1500 vs 2000) than the same width trained
on random soup alone, though both were already reliable; on this n the
extra data variety mainly speeds convergence rather than rescuing a model
that wouldn't otherwise converge.

