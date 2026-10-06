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
- Compute: an RTX 3080 desktop (used its CPU — these models
  and batches are small enough that per-step Python overhead, not matmul
  throughput, dominates; a single `systemd-run --user` unit, GPU confirmed idle throughout).
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
- All training ran on CPU (the RTX 3080 machine's Ryzen, not its GPU): with
  batch size 8 and boards this small, per-step Python-level grid
  generation and scoring, not matmul throughput, dominated wall time, so
  GPU launch overhead would have made it slower, not faster, for this
  sweep.

## Follow-up: n = 2, 3, 10 in one pass (2026-09-30, later the same day)

Follow-up question: what about predicting 3 or 10 rule-applications ahead
in a single forward pass, instead of 1 or 2? Two model families, this
time trained on the RTX 3080 machine's GPU with a vectorized numpy
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


### Results, n = 3 (pattern-mixed variant: best pure-random width was c=32)

| model | params | converged / 5 | median steps to converge | mean final cells-correct |
|---|---:|---:|---:|---:|
| A untied DeepCNN c=8 | 1,841 | 3 | 5000 | 0.903 |
| A untied DeepCNN c=32 | 28,097 | 4 | 7250 | 0.945 |
| A untied DeepCNN c=64 | 111,489 | 1 | 4000 | 0.771 |
| A untied DeepCNN c=128 | 444,161 | 0 | - | 0.698 (= baseline) |
| B tied MinimalCNN c=8 | 89 | 0 | - | 0.698 (= baseline) |
| A untied DeepCNN c=32, +25% pattern-mixed training | 28,097 | 3 | 7500 | 0.888 |

All-dead trivial baseline on the random-soup eval at n=3: 0.698. Width is
**not** monotonic here: c=32 (4/5) beats both c=8 (3/5) and c=64 (1/5), and
c=128 collapses entirely (0/5, sitting exactly on the trivial baseline —
every seed learned to predict all-dead and never escaped it, the same
"hopeless" mode a wide model hit in the earlier budget-capped n=2 run).
Same optimizer, same learning rate, same steps budget for every width;
this is optimization difficulty at fixed hyperparameters, not a statement
that c=128 categorically cannot learn 3-step Life. B tied (applying the
89-param n=1 core three times with a straight-through estimator) did not
converge in any seed either, staying at baseline, though
`rule_discovery_512_acc_per_seed` [0.689, 0.596, 0.727, 0.656, 0.734] shows
every seed learned something systematic, just never the exact rule and
never a 3-step-exact composite.

Structured-pattern eval (exact boards / total, mean masked cells-correct in
parentheses) for the architectures that learned something beyond baseline:

| pattern | c=8 (3/5 converged) | c=32 (4/5 converged) | c=64 (1/5 converged) | c=32 +25% patterns (3/5 converged) |
|---|---:|---:|---:|---:|
| block | 8/20 (0.58) | 16/20 (0.80) | 8/20 (0.40) | 20/20 (1.00) |
| beehive | 32/40 (0.80) | 32/40 (0.85) | 8/40 (0.20) | 32/40 (0.95) |
| blinker | 24/40 (0.70) | 32/40 (0.80) | 8/40 (0.20) | 32/40 (0.83) |
| toad | 48/80 (0.66) | 64/80 (0.80) | 16/80 (0.20) | 64/80 (0.80) |
| beacon | 20/40 (0.68) | 32/40 (0.80) | 8/40 (0.34) | 32/40 (0.89) |
| glider | 96/160 (0.67) | 128/160 (0.83) | 32/160 (0.32) | 96/160 (0.79) |
| lwss | 96/160 (0.68) | 128/160 (0.86) | 32/160 (0.30) | 96/160 (0.78) |
| r_pentomino | 96/160 (0.74) | 128/160 (0.89) | 32/160 (0.25) | 96/160 (0.87) |

(c=128 untied and B tied score exactly 0 across every pattern family, since
both predict all-dead everywhere, so they are omitted from the table.)

Observations:

- Unlike n=2, generalization to structured patterns at n=3 tracks the
  fraction of seeds that actually converged exactly, not an all-or-nothing
  property: c=32 (4/5 exact-converged seeds) is clearly better across every
  pattern family than c=8 (3/5) and much better than c=64 (1/5), and the
  two 0/5-converged conditions (c=128, B tied) score exactly 0 on every
  pattern.
- The pattern-mixed training variant helped some patterns and not others
  at n=3: block went from 16/20 to a perfect 20/20 and beacon's masked
  accuracy rose from 0.80 to 0.89, but the fraction of seeds reaching exact
  100%-cells-correct on random soup dropped slightly (3/5 vs 4/5) and the
  median convergence step rose (7500 vs 7250). At n=2 the same recipe
  helped unambiguously (faster convergence, no seeds lost); at n=3 it is a
  wash at best for overall exactness, though it does look like it shifts
  the model's errors away from small still lifes specifically. Not enough
  seeds here to call this conclusively either way.
- No model, converged or not, discovered a still-life-preserving or
  glider-translating rule from scratch when it failed to converge: the
  under-capacity or badly-optimized runs (c=8, c=64) get roughly the same
  masked accuracy on every pattern family regardless of how different
  those patterns are (a glider is nothing like a block), consistent with
  learning some other, wrong but roughly-uniform 3-step function rather
  than a partially-correct version of the real rule.

### Results, n = 10

| model | params | converged / 5 | mean final cells-correct | mean final baseline |
|---|---:|---:|---:|---:|
| A untied DeepCNN c=8 | 5,929 | 0 | 0.7727 (= baseline) | 0.7727 |
| A untied DeepCNN c=32 | 92,833 | 0 | 0.7727 (= baseline) | 0.7727 |
| A untied DeepCNN c=64 | 369,985 | 0 | 0.7727 (= baseline) | 0.7727 |
| A untied DeepCNN c=128 | 1,477,249 | 0 | 0.7727 (= baseline) | 0.7727 |
| B tied MinimalCNN c=8 | 89 | 0 | 0.7727 (= baseline) | 0.7727 |
| A untied DeepCNN c=8, +25% pattern-mixed training | 5,929 | 0 | 0.7727 (= baseline) | 0.7727 |

At n=10, every pure-random-trained width and both families plateau
exactly at the trivial all-dead baseline (0.7727), the same number to four
decimal places for all five conditions — every one of them, all 5 seeds
each, converges to predicting all-dead and stops improving, regardless of
architecture or 165x range in parameter count (5,929 to 1,477,249). B
tied's `rule_discovery_512_acc_per_seed` is `[0.7266, 0.7266, 0.7266,
0.7266, 0.7266]` — identical across all 5 seeds to the last digit, meaning
every seed's 89-param core converges to the exact same fixed point on the
512-case table, not just the same random-soup behaviour. This is the
strongest evidence in the whole sweep for the paper's "hard for neural
networks to learn" framing: at 10 rule-applications, gradient descent from
this training density and learning rate does not find a better basin than
"predict the majority class" for any width tried, tied or untied.

Structured-pattern eval for the pattern-mixed variant (the only n=10
condition that is not identically the trivial baseline on patterns; the
five pure-random conditions score 0/total on every pattern family, same
as B tied and c=128 at n=3):

| pattern | c=8, +25% pattern-mixed training: exact/total (masked cells-correct) |
|---|---:|
| block | 16/20 (0.80) |
| beehive | 28/40 (0.78) |
| blinker | 32/40 (0.80) |
| toad | 40/80 (0.73) |
| beacon | 32/40 (0.80) |
| glider | 8/160 (0.29) |
| lwss | 0/160 (0.19) |
| r_pentomino | 12/160 (0.50) |

Observation: this is the one clear positive result for pattern-mixed
training in the whole sweep. The n=10 pattern-mixed model still fails to
converge on random soup (mean final cells-correct is 0.77265625, a
hundredth of a percent below the pure-random condition's baseline, not a
meaningfully different number) — it has not learned the 10-step rule in
any general sense. But on patterns it saw examples of during training
(still lifes, oscillators), it does dramatically better than predicting
all-dead (0.73 to 0.80 masked accuracy vs 0 for every pure-random-trained
n=10 model). It does worse on the spaceships (glider, LWSS) and the
methuselah (R-pentomino), which travel or grow rather than staying in one
place, consistent with the model having partially memorized how to keep a
small set of fixed local patterns stable rather than discovering the
general rule. This is a data-composition effect, not a capacity or
convergence effect: the model is still stuck in the same "predict nothing
changes" local optimum for arbitrary inputs, but 25% of its training
distribution being exactly the test patterns taught it those patterns'
specific 10-step fixed points.

### Summary across n

| n | best converged / 5 (width) | smallest exact-converging model | n=10-scale finding |
|---|---|---|---|
| 2 | 5/5 (all widths 8/32/64; also 5/5 with 25% patterns) | DeepCNN c=8, 1,257 params | - |
| 3 | 4/5 (c=32) | DeepCNN c=32, 28,097 params | width non-monotonic; c=128 and the tied model collapse to baseline |
| 10 | 0/5 (every width, both families) | none converged | universal collapse to the trivial baseline; pattern-mixed training rescues performance on trained-on patterns only |

Raw per-condition JSON for every run in this document:
`docs/runs/2026-09-30-minimal-life-nstep-results.json` (first, budget-capped
pass), `docs/runs/2026-09-30-minimal-life-patterns-quick-results.json`
(single-seed representative check across n=1/2/3/10), and
`docs/runs/2026-09-30-minimal-life-full-results.json` (second,
budget-lifted pass — the source for every table above from "Results, n = 2"
onward). Compute for every run in this document: the RTX
3080 machine's GPU, one `systemd-run --user` unit at a time.
