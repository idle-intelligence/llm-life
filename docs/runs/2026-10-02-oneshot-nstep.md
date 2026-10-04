# Can a plain MLP learn 2-step Life in one shot, with only the light cone as prior?

Question: with no convolution, no recurrence/weight-tying, and depth not
tied to N, can a network learn Life N steps ahead in a single forward pass,
given only the light-cone prior (input is the flattened (2N+1)x(2N+1)
window around a cell; output is that cell's state N steps later)? This run
covers N=2 only, trained and verified exhaustively over the full window
space; N=3, N=10, and a transformer-encoder second family were scoped out
of this run before it started (see "Scope" below) and are not addressed
here.

References:
- Springer and Kenyon, "It's Hard for Neural Networks to Learn the Game of
  Life", arXiv:2009.01398 -- motivates treating "did it converge exactly"
  as the headline question rather than assuming it, and this run's
  non-monotonic-in-width, non-monotonic-in-seed results (below) are a
  sharper version of the same difficulty this paper documents for
  conv-prior networks on the easier n=1 problem.
- Berkovich and Buehler, "LifeGPT: Topology-Agnostic Generative Pretrained
  Transformer Model for Cellular Automata", arXiv:2409.12182 -- the
  comparison point for "transformer with no locality bias, trained on raw
  cell tokens, can it learn a CA update rule"; this run's optional
  transformer-encoder family (CLS readout over window-cell tokens) was
  scoped out before being started, so no transformer results are reported
  here.
- `docs/runs/2026-09-30-minimal-life.md` -- this repo's conv-prior n=1/n=2
  sweep (DeepCNN, MinimalCNN); its n=2 untied DeepCNN (conv prior, 5x5
  receptive field via two stacked 3x3 convolutions) converged 5/5 seeds at
  c=8 (1,257 params) and c=32 (18,849 params) in float.
- `docs/runs/2026-10-02-bitslice-2step.md` -- the same n=2 problem, same
  conv-prior DeepCNN architecture, ternary weights/binary activations: 2/5
  seeds converged to exact at c=32, and one of those two (seed 3) was
  confirmed exact on the full exhaustive 2**25-window space, 0 mismatches.
  **This run's result (below) is the direct no-conv-prior counterpart to
  that exact result: no configuration tried here reached 0 mismatches.**

## Scope

The original brief asked for N in {2, 3, 10} and an optional transformer
family. Partway into this run, scope was narrowed to **N=2 only** (full
2**25 training, exhaustive verification, and probes on the best models);
N=3, N=10, and the transformer family were dropped and never started. This
document and its data cover N=2 exclusively.

## Architecture

Plain MLP family, `pytorch/oneshot_model.py`, input is the flattened 5x5
window (25 bits, `bitslice_compiler.window_offsets(5)` order, i.e. the same
row-major (dy,dx) for dy,dx in -2..2 convention
`docs/runs/2026-10-02-bitslice-2step.md`'s exhaustive checker uses), output
a single logit for the centre cell's state after 2 Life steps. No
convolution, no recurrence, no weight sharing between layers, and depth is
a free hyperparameter, not derived from N.

- depth in {2, 4} (`PlainMLP`): `Linear(25, w) -> ReLU -> [Linear(w,w) ->
  ReLU] * (depth-1) -> Linear(w, 1)`. No normalization, no residual.
- depth in {8, 16} (`ResidualMLP`): `Linear(25, w)` input projection, then
  `depth` pre-LN residual blocks (`x = x + Linear(ReLU(Linear(LayerNorm(x))))`,
  each block's own parameters, no sharing), then a final `LayerNorm` and
  `Linear(w, 1)` readout.
- widths w in {128, 512}.

24 conditions (depth x width x 3 seeds).

## Data and ground truth

Trained on the **full set of 2**25 5x5-window patterns**, not boards or a
sample: `pytorch/oneshot_n2_exhaustive.py` reuses
`bitslice_life2_exhaustive.py`'s already-verified bit-sliced generator and
ground truth (`make_planes_chunk`, `ground_truth_2step`,
`OFFSETS5`) from branch `bitslice-2step` directly -- no new ground-truth
code was written for this run. `ground_truth_2step` is a from-scratch
bitwise popcount/Life-step-twice reference (not reusing
`bitslice_ops`' popcount primitives, so a shared bug cannot cancel out),
itself cross-checked against `life.py`/`vec_step` on 20,000 random windows
in the prior run before being trusted.

Class balance over the full space: 8,502,430 / 33,554,432 patterns (25.3%)
have centre-cell-alive-after-2-steps = 1. Training used a single global
`pos_weight = 2.9465` (neg/pos ratio) in `BCEWithLogitsLoss`, computed once
by one exhaustive pass, not re-estimated per batch.

Training: full-dataset epochs over all 1,048,576 32-pattern words, visited
in randomly shuffled chunks (2,048 words = 65,536 patterns/step, 512
steps/epoch), Adam lr=1e-3, up to 20 epochs. Exactness (0/33,554,432
mismatches) checked by the identical chunked-exhaustive procedure every 2
epochs; training stops the moment a model is exact. No condition reached
that stop in this run, so every condition ran the full 20 epochs.

## Compute

GPU box (RTX 3080), confirmed idle before launch (`nvidia-smi` 0% util),
one `systemd-run --user` unit at a time under `flock
/data/remote-worker/lean/box.lock`. The sweep unit (24 conditions) was
paused and resumed twice (SIGSTOP/SIGCONT) by other workers needing the GPU
for browser timing work; the per-condition `wall_seconds` reported below
include those pauses and are not a clean throughput measurement -- the
per-epoch mismatch counts (not wall-clock) are the numbers this run's
conclusions rest on. Probes and the failure-case dump ran afterward as a
separate, unpaused unit.

## Results, N=2, full-2**25 training and exhaustive verification

No configuration reached 0 mismatches. Best-to-worst by final mismatch
count:

| depth | width | seed | params | exact? | mismatches / 33,554,432 | cells correct |
|---:|---:|---:|---:|---|---:|---:|
| 8 | 128 | 1 | 269,953 | no | 179 | 99.999% |
| 8 | 128 | 2 | 269,953 | no | 643 | 99.998% |
| 8 | 512 | 2 | 4,225,537 | no | 4,852 | 99.986% |
| 4 | 512 | 2 | 801,793 | no | 41,618 | 99.876% |
| 16 | 512 | 2 | 8,436,225 | no | 60,338 | 99.820% |
| 16 | 128 | 2 | 536,193 | no | 117,128 | 99.651% |
| 4 | 512 | 1 | 801,793 | no | 221,138 | 99.341% |
| 16 | 128 | 1 | 536,193 | no | 265,698 | 99.208% |
| 4 | 128 | 1 | 52,993 | no | 310,994 | 99.073% |
| 8 | 512 | 0 | 4,225,537 | no | 396,926 | 98.817% |
| 8 | 128 | 0 | 269,953 | no | 675,569 | 97.987% |
| 4 | 128 | 2 | 52,993 | no | 685,313 | 97.958% |
| 16 | 512 | 0 | 8,436,225 | no | 686,187 | 97.955% |
| 16 | 128 | 0 | 536,193 | no | 742,651 | 97.787% |
| 4 | 512 | 0 | 801,793 | no | 1,501,441 | 95.525% |
| 4 | 128 | 0 | 52,993 | no | 1,726,607 | 94.854% |
| 16 | 512 | 1 | 8,436,225 | no | 1,982,642 | 94.091% |
| 2 | 512 | 2 | 276,481 | no | 2,279,165 | 93.208% |
| 2 | 512 | 0 | 276,481 | no | 2,775,681 | 91.728% |
| 2 | 512 | 1 | 276,481 | no | 3,103,808 | 90.750% |
| 2 | 128 | 1 | 19,969 | no | 3,801,538 | 88.671% |
| 2 | 128 | 0 | 19,969 | no | 4,013,256 | 88.040% |
| 2 | 128 | 2 | 19,969 | no | 4,580,244 | 86.350% |
| 8 | 512 | 1 | 4,225,537 | no | 5,864,649 | 82.522% |

Raw per-condition JSON: `docs/runs/2026-10-02-oneshot-nstep-n2-results.json`.
Trained model checkpoints for the four models probed below (small enough to
keep in the repo, 5.3 MB total): `pytorch/oneshot_models/`. The full set of
24 checkpoints (168 MB) was left on the GPU box at
`/data/remote-worker/oneshot-nstep/results/models/` and is not in this
repo.

## Observations

- **No configuration tried is exact.** The closest, depth=8/width=128/seed=1
  (269,953 params, a residual+LayerNorm MLP), gets 179 of 33,554,432
  windows wrong -- 99.99947% of the full space, closer than any other
  condition by two orders of magnitude, but not a closed-form rule the way
  the conv-prior ternary circuit in `docs/runs/2026-10-02-bitslice-2step.md`
  (seed 3, c=32, 0/33,554,432) is.
- **Depth and width are not monotonic.** depth=8/width=128 (two seeds in
  the 179-643 range) beats every depth=16 and every width=512 condition
  except one (depth=8/width=512/seed=2 at 4,852). depth=8/width=512/seed=1
  is the single worst non-depth-2 result in the whole sweep (5,864,649
  mismatches, 82.5% -- worse than several depth=2 conditions), on the exact
  same architecture and hyperparameters as its sibling seeds 0 and 2
  (396,926 and 4,852). This matches `docs/runs/2026-09-30-minimal-life.md`'s
  own observation that width is "not monotonic" for n=3 Life, and
  Springer and Kenyon (arXiv:2009.01398)'s general finding that gradient
  descent on Life-like rules is prone to getting stuck in a
  capacity-independent way; here the effect is visible seed-to-seed at
  fixed depth and width, not just condition-to-condition.
- **Depth=2 (plain MLP, no residual/LayerNorm) is uniformly the worst
  family**: all 6 depth=2 conditions (both widths, all 3 seeds) land between
  86% and 93% cells correct, never dropping below roughly the "guess
  structured-majority-class" range and never approaching the 99%+ results
  every deeper architecture reaches at least once. Residual connections and
  LayerNorm were only used for depth in {8, 16} per the brief; depth=4
  (also plain, no residual/LayerNorm) does substantially better than
  depth=2 (best: 99.876% at width=512/seed=2) despite having the same
  architecture family, so the depth=2-to-depth=4 jump is not attributable
  to the residual/LayerNorm change -- it appears before that architectural
  switch happens.
- The best result (179 errors) is a small enough model (269,953 params)
  that it is not simply "throw more capacity at it": depth=8/width=512 (15x
  the parameters) does worse on 2 of its 3 seeds (396,926 and 5,864,649
  vs. 179-643), and depth=16 at either width never beats depth=8/width=128
  at all.

## Probes

For four models -- the two closest-to-exact (depth=8/width=128, seeds 1 and
2), the best width=512 result (depth=4/width=512/seed=2), and one
deliberately weak model for contrast (depth=2/width=128/seed=0, the
sweep's 4th-worst result at 88.04%) -- a logistic-regression probe
(`pytorch/oneshot_probe.py`, LBFGS, 4,000 training words / 1,000 held-out
test words, i.e. 128,000 / 32,000 windows) was fit per hidden layer to
predict the **centre cell's true state at t+1** (one Life step, not two).
The label is not re-derived: it is `life_step_planes` applied to the
centre 3x3 sub-window, the exact intermediate value
`ground_truth_2step` computes internally on the way to its final answer.

| model | layer 0 | layer 1 | layer 2 | layer 3 | layer 4 | layer 5 | layer 6 | layer 7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| depth8/width128/seed1 (179 err) | 89.88% | 100.0% | 100.0% | 99.98% | 99.95% | 99.98% | 99.95% | 99.92% |
| depth8/width128/seed2 (643 err) | 96.23% | 99.98% | 99.97% | 99.94% | 99.95% | 99.98% | 99.96% | 99.96% |
| depth4/width512/seed2 (41,618 err) | 92.66% | 99.99% | 100.0% | 99.997% | -- | -- | -- | -- |
| depth2/width128/seed0 (4,013,256 err, contrast) | 89.21% | 99.97% | -- | -- | -- | -- | -- | -- |

(Test-set accuracy shown; depth4 has 4 hidden layers, depth2 has 2, hence
the shorter rows.) Raw per-layer train/test accuracy:
`docs/runs/2026-10-02-oneshot-nstep-probe_depth8_width128_seed1.json`,
`..._probe_depth8_width128_seed2.json`, `..._probe_depth4_width512_seed2.json`,
`..._probe_depth2_width128_seed0.json`.

Observation, not interpreted beyond the numbers: in every model probed,
including the weak depth=2 contrast model, a linear probe recovers the
true one-step-ahead centre state at **≥99.97% accuracy from the very first
hidden layer onward** (layer 1 in every case) -- the one-step intermediate
state is linearly decodable almost immediately, well before the network's
own final output is anywhere near that accurate (the depth=2 contrast
model's own output is only 88.04% correct on the full 2-step task, despite
its layer-1 activations linearly encoding the 1-step state at 99.97%).
Layer 0 (the raw input projection, before any nonlinearity is applied
through a full block) is consistently the weakest layer (89-96%), and the
jump to ≥99.9% happens at the first post-nonlinearity layer in every model,
independent of how accurate that model's final 2-step output is.

## The 179 failing windows (depth=8/width=128/seed=1)

`pytorch/oneshot_n2_failures.py` re-ran the full exhaustive check for this
one model and recorded every failing window's live-cell count (of the 25
cells in the 5x5 window) and error direction. Full list (pattern indices
and 5x5 grids): `docs/runs/2026-10-02-oneshot-nstep-failures_depth8_width128_seed1.json`.

131 of 179 errors are "model says alive, truth says dead"; 48 are the
reverse. Live-cell-count histograms (count of live cells in the 5x5 window,
not a Life neighbour count):

| live cells in window | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model=1, truth=0 | 2 | 7 | 3 | 14 | 13 | 10 | 12 | 7 | 10 | 15 | 16 | 13 | 5 | 4 |
| model=0, truth=1 | - | 1 | 4 | 4 | 11 | 8 | 10 | 5 | 1 | 1 | 2 | - | 1 | - |

Observation, not interpreted beyond the numbers: the errors are spread
across live-cell counts from 3 to 16 (of 25 possible), with no single count
concentrating more than 16 of the 179 failures -- there is no visible
single "this one density of input is where the model fails," consistent
with scattered near-boundary errors rather than one systematic blind spot.

## Limits

- No configuration reached exact; the conv-prior comparison
  (`docs/runs/2026-10-02-bitslice-2step.md`) did, at a smaller parameter
  count (18,849 for the float DeepCNN c=32, or 269,953 for this run's
  closest no-prior result) -- this run does not establish whether more
  epochs, a different optimizer/schedule, or more seeds at depth=8/width=128
  would eventually reach exact; 20 epochs (512 steps/epoch, full-dataset
  coverage per epoch) and 3 seeds per condition is what the time budget
  allowed, not a claim that the architecture cannot reach exact.
- `wall_seconds` in the raw results JSON include two SIGSTOP/SIGCONT pauses
  from other GPU-box workers during the sweep (noted above); they are not
  used for any conclusion in this document.
- The probe's label (centre cell at t+1) is only one of the two
  intermediate quantities the brief asked about; the 3x3-cells-at-t+1 and
  t+2 probes for N=3 were not run, since N=3 was dropped from scope before
  this run started.
- The 179/643-error models were not fine-tuned toward exactness (the prior
  run, `docs/runs/2026-10-02-bitslice-2step.md`, found gradient-based
  surgical fixes on an already-converged discretized circuit tend to
  overshoot a single targeted boundary into many new errors; the same risk
  was not tested here on a continuous-weight model, and nothing here claims
  it would behave the same way).

## Data

Model family: `pytorch/oneshot_model.py`. Training/exhaustive verification:
`pytorch/oneshot_n2_exhaustive.py` (depends on `bitslice_life2_exhaustive.py`
and `bitslice_compiler.py` from branch `bitslice-2step`, synced unchanged).
Probes: `pytorch/oneshot_probe.py`. Failure-case dump:
`pytorch/oneshot_n2_failures.py`. Raw results:
`docs/runs/2026-10-02-oneshot-nstep-n2-results.json`,
`docs/runs/2026-10-02-oneshot-nstep-probe_depth8_width128_seed1.json`,
`docs/runs/2026-10-02-oneshot-nstep-probe_depth8_width128_seed2.json`,
`docs/runs/2026-10-02-oneshot-nstep-probe_depth4_width512_seed2.json`,
`docs/runs/2026-10-02-oneshot-nstep-probe_depth2_width128_seed0.json`,
`docs/runs/2026-10-02-oneshot-nstep-failures_depth8_width128_seed1.json`.
Checkpoints for the four probed models: `pytorch/oneshot_models/`. The
remaining 20 checkpoints (168 MB total for all 24) were left on the GPU box
at `/data/remote-worker/oneshot-nstep/results/models/`, not committed.
