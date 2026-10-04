# One trained network, Life two steps ahead, compiled to a bit-sliced circuit

Question: `docs/runs/2026-10-02-bitslice-compiler.md`'s test (b) showed a
flat 25-input MLP does not converge on Life-two-steps-ahead, and sidestepped
that by *composing* the already-exact 1-step network with itself
symbolically (10 instances of the same trained net, 920 gates). This run
asks the harder question directly: can a *single* network, trained
end-to-end on the 2-step target, with the architecture that already
converges reliably for 2-step Life in float
(`docs/runs/2026-09-30-minimal-life.md`'s untied DeepCNN), be made fully
ternary-weight/binary-activation and still converge -- and does its
compiled circuit beat, match, or lose to the composed 920-gate circuit?

References:
- Ma, Wang, Ma, Wang, Wang, Huang, Wei, "The Era of 1-bit LLMs: All Large
  Language Models are in 1.58 Bits" (BitNet b1.58), arXiv:2402.17764 -- the
  absmean ternary weight scheme used for the ternary condition here, same
  as `bitslice_life.py` on this branch's parent.
- Rastegari, Ordonez, Redmon, Farhadi, "XNOR-Net: ImageNet Classification
  Using Binary Convolutional Neural Networks", arXiv:1603.05279 -- the
  sign(w)/mean(|w|) binary-weight scheme used for the optional binary
  condition (section "Binary weights" below).
- Miotti et al., "Differentiable Logic Cellular Automata", arXiv:2506.04912
  -- closest prior art in domain (CA) and spirit (compile a trained
  thresholded CA update rule to a circuit); this run differs by training a
  genuine two-channel-width *convolutional* network directly on the 2-step
  target (not a per-cell logic-gate graph) and compiling it with a generic
  weight-row-to-threshold compiler rather than hand-designed gates.
- Springer and Kenyon, "It's Hard for Neural Networks to Learn the Game of
  Life", arXiv:2009.01398 -- shows gradient descent on networks with
  standard (non-quantized) activations reliably gets stuck learning Life
  exactly, motivating why "did it converge at all" and "how many of 5 seeds
  converged" are reported here rather than assumed; this run's ternary
  condition converged 2/5, a milder version of the same difficulty, further
  sharpened by quantization (see Observations).
- `docs/runs/2026-09-30-minimal-life.md` -- the float untied-DeepCNN
  precedent this run's architecture and training recipe (data generation,
  pattern-mixed boards, plateau stopping) are copied from: `pytorch/
  minimal_life_full.py`'s `DeepCNN n=2` condition (section "Results, n = 2"
  of that doc) converged 5/5 at c=8 and c=32 in float, with pattern-mixed
  training converging in fewer median steps than pure-random-soup training.
- `docs/runs/2026-10-02-bitslice-compiler.md` -- the ternary-weight/
  binary-activation QAT recipe (soft-surrogate STE `1/(1+|y|)^2`, which
  fixed convergence over the textbook hard-box-window STE) this run extends
  to multi-channel convolution.

## Architecture and training

`pytorch/bitslice_life2.py`'s `BitsliceDeepCNN`: `Conv3x3(1->c, circular,
ternary)` -> step -> `Conv3x3(c->c, circular, ternary)` -> step ->
`Conv1x1(c->1, ternary)` -> step. This is exactly the 5x5 receptive field a
2-step Life prediction needs (proven necessary and sufficient by the
composed circuit in the parent run) -- two conv layers, not
`minimal_life_full.py`'s `DeepCNN(c, depth=n+1)` which uses 3 conv layers
(7x7 receptive field) for n=2; the fixed-depth-2 architecture here is
deliberately the minimal receptive field, not a copy of that file's exact
instantiation.

Copied from `pytorch/minimal_life_full.py`'s `DeepCNN n=2` condition:
`make_batch_mixed` (25% of training boards are a known Game of Life pattern
in a random dihedral orientation and position, from `minimal_life_patterns
.PATTERNS`, the rest random soup; `vec_step` applied twice for the 2-step
target, checked bit-for-bit against `life.py`'s `Grid.step` via
`verify_vec_step` before every run), `fresh_boards_eval` (100 fresh 16x16
boards per eval), Adam optimizer, and the plateau-based stopping rule
(train up to 20,000 steps, stop on exact 100% match or when
best-cells-correct hasn't improved by 0.002 over the last 20 evals of 500
steps each).

What is new relative to that float precedent: ternary weights (BitNet
b1.58 absmean, one scalar scale per conv layer) and binary step
activations between layers (soft-surrogate STE, copied verbatim from
`bitslice_life.py`), in place of float weights and ReLU. The final 1x1
layer returns its raw pre-activation (the step/threshold is implicit in
`sigmoid(y) >= 0.5`, same convention as `bitslice_life.py`'s `ThresholdNet`)
so `binary_cross_entropy_with_logits` has something to train against.

Hyperparameters: board size 16, density 0.38, batch 32, lr 5e-3, 20,000-step
cap, plateau window 20 (of 500-step evals), plateau eps 0.002,
pattern_frac 0.25, 5 seeds, channels c=32. Training ran on the Mac's CPU
(`pytorch/bitslice_life2.py`, `--channels 32 --seeds 0,1,2,3,4
--train-steps 20000`); wall time 896 s (15 min) for all 5 seeds.

## Results, c=32, ternary weights

| seed | converged at step | best cells-correct | final cells-correct (200 fresh boards) | stopped early |
|---:|---:|---:|---:|---|
| 0 | 14,500 | 1.0 | 1.0 | - |
| 1 | - | 0.7048 | 0.6824 | plateau |
| 2 | - | 0.7866 | 0.7552 | - |
| 3 | 12,500 | 1.0 | 1.0 | - |
| 4 | - | 0.8496 | 0.6646 | - |

2/5 seeds converged to exact (100% cells-correct on fresh boards). All-dead
trivial baseline on this eval: ~0.68 cells correct (matches
`docs/runs/2026-09-30-minimal-life.md`'s n=2 baseline). Both converged
seeds (0, 3) scored 1.0 masked-cells-correct and matched every board exactly
on the full structured-pattern dataset (block, beehive, blinker, toad,
beacon, glider, lwss, r_pentomino; 4 positions x all dihedral variants each,
32x32 boards -- bigger than the 16x16 training boards): 140/140 exact
boards for each converged seed. Model dumps and the full per-seed training
log: `pytorch/bitslice_models/life_2step_cnn_c32_train.json`.

## Compilation and gate counts

Compiled with tritsliced, a private compiler.

| program | and | or | xor | not | total |
|---|---:|---:|---:|---:|---:|
| this run, seed 0 (trained 1->32->32->1 CNN) | 5,570 | 3,006 | 4,012 | 1,344 | 13,932 |
| this run, seed 3 (trained 1->32->32->1 CNN) | 5,100 | 2,895 | 3,610 | 1,189 | 12,794 |
| parent run, composed 1-step net x2 (9->8->1, composed 10x) | - | - | - | - | 920 |
| parent run, two separate dispatches of the 1-step circuit (2 x 92) | - | - | - | - | 184 |

## Did it learn the same thing twice? (layer-1 channel analysis)

For each converged seed, every layer-1 (`1->c`) output channel is a
function of only its own 3x3 neighbourhood (same receptive field as a
1-step Life rule), so it can be checked against the true 1-step Life
update on all 512 3x3 neighbourhoods, same enumeration as the parent run's
`cnn_512_eval`.

**No single channel equals Life(t+1) or its negation** in either converged
seed (best single-channel match: ~0.73, roughly the "predict mostly dead"
baseline rate, not a real detection -- see per-channel table in
`pytorch/bitslice_models/life_2step_cnn_c32_seed{0,3}_verify.json`,
`layer1_channel_analysis`). Most channels (28-30 of 32) are not expressible
as a simple neighbour-count threshold or threshold-AND/OR-self; a handful
are exactly `n>=k` for some k (pure 8-neighbour-count thresholds, no self
term) and one or two are exactly `(self + neighbour count) >= k`.

**A pairwise check (AND/OR/XOR/A-AND-NOT-B of every pair of the 32
channels) found an exact match in both converged seeds**: XOR of two
specific channels reproduces the ground-truth 1-step Life rule exactly on
all 512 neighbourhoods.

| seed | channel A | channel A's function | channel B | channel B's function | A XOR B |
|---|---:|---|---:|---|---|
| 0 | 5 | `neighbour_count >= 4` | 9 | `(self + neighbour_count) >= 3` | = Life(t+1), 512/512 |
| 3 | 17 | `neighbour_count >= 4` | 21 | `(self + neighbour_count) >= 3` | = Life(t+1), 512/512 |

Both converged seeds found the *same pair* of threshold functions (just at
different channel indices), and `XOR(n>=4, (self+n)>=3) == Life(t+1)` is in
fact an algebraic identity, checkable by hand: writing `a = (self+n)>=3`
and `b = n>=4`, for `self=0` (`a = n>=3`): birth is `n==3`, where `a=1,b=0`
(XOR=1) and every other `n` gives `a=b` (XOR=0); for `self=1` (`a = n>=2`):
survival is `n in {2,3}`, where `a=1,b=0` (XOR=1), and every other `n`
(0, 1, or >=4) gives `a=b` (XOR=0). So **layer 1 does not store the Life
rule in any single channel, but two of its channels are simple overlapping
neighbour-count thresholds whose XOR reconstructs it exactly** -- a
genuinely different, and in this case independently rediscovered-twice,
decomposition of the same 1-step rule than the parent run's hand-built
`(n>=3) AND NOT(n>=4)) OR (self AND (n>=2) AND NOT(n>=3))` circuit (which
also uses `n>=2/3/4` and `self`, combined with AND/OR instead of XOR).
Whether layer 1 is *using* this decomposition for the final 2-step output,
versus it being a coincidental exact match, is not established here --
only that it exists and is exact; the brief's "if no channel matches, say
what layer 1 computes instead" is answered by "neighbour-count/self
thresholds, mostly not individually meaningful, with this one XOR-pair
exception," not interpreted further.

## Binary weights (optional, step 5): negative result

Same architecture and training recipe, `wq = sign(w)` (never 0),
`scale = mean(|w|)` (XNOR-Net, arXiv:1603.05279) in place of BitNet
ternary absmean -- no compiler change needed, since a binary-weight layer
is just the special case of the ternary positive/negative split where a
weight of 0 never happens. `pytorch/bitslice_life2.py --channels 32 --seeds
0,1,2,3,4 --train-steps 20000 --quant binary`.

| seed | best cells-correct | final cells-correct | stopped early | steps run |
|---:|---:|---:|---|---:|
| 0 | 0.6946 | 0.6819 | plateau | 16,000 |
| 1 | 0.6944 | 0.6870 | plateau | 10,500 |
| 2 | 0.7023 | 0.6846 | plateau | 10,500 |
| 3 | 0.7014 | 0.6885 | plateau | 18,000 |
| 4 | 0.7013 | 0.6897 | plateau | 16,500 |

**0/5 seeds converged**; every seed plateaued within 1-2 points of the
~0.68 all-dead baseline, never reaching the 0.76-0.98 climb the ternary
condition's eventual winners passed through on the way to exact (compare
seed 0's ternary trajectory above). No model was compiled or verified here
since none converged -- reported as a clean negative result, not
extrapolated. The likely mechanism (not verified further, stated as a
hypothesis): removing the zero weight removes every unit's ability to
*ignore* an input; `conv2`'s units each sum over 288 (32 channels x 9
positions) forced-nonzero terms instead of a trained sparse subset, which
plausibly removes exactly the inductive bias (local, mostly-irrelevant-
input-suppressing threshold units) BitNet-style ternary quantization
preserves and binary does not.

## Limits

- 2**25 5x5-neighbourhood cases were initially assumed computationally
  intractable to check exhaustively; this was wrong -- bit-slicing the
  full 2**25-pattern space into 1,048,576 32-cell words takes under 10
  seconds per seed to check in full. Board-level evidence (tens of boards
  up to 64x64 cells) initially missed a real 1-in-33.5M error in seed 0;
  see "Exactness" below.
- Only c=32 was trained to convergence within the time budget; the gate
  count (13-14k) is set by that width, not minimized -- a narrower c (8 was
  the other float-reliable width in the parent float study) was not
  attempted here given the budget, and would very plausibly cut the gate
  count substantially at layer 2 (quadratic in c) while needing more seeds
  to find an exact one, per the ternary condition's 2/5 hit rate.
- 3/5 seeds at c=32 did not converge within 20,000 steps (plateaued around
  0.70-0.85 cells-correct) -- consistent with Springer and Kenyon
  (arXiv:2009.01398)'s general finding that gradient descent on Life is
  prone to getting stuck, here further sharpened by ternary
  quantization: the float precedent at the same width converged 5/5.
- No hardware verification of the emitted WGSL: checked for syntactic
  well-formedness and that the NumPy and WGSL backends share one circuit
  graph (so NumPy's bit-for-bit agreement with the trained network implies
  WGSL agreement by construction), not run on an actual GPU/SwiftShader in
  this session -- same limit the parent run noted for its own kernels.
- The "did it learn the same thing twice" pairwise check only tried
  AND/OR/XOR/A-AND-NOT-B of *pairs* of channels, not triples or arbitrary
  Boolean combinations; a channel or larger combination doing something
  else entirely that happens not to be captured by this search is possible
  and was not ruled out.

## Exactness

A board-sampled counterexample was found for seed 0's compiled circuit
(roughly 1 error per ~1.5M random cells in single-pass sampling). An
exhaustive enumeration of all 33,554,432 possible 5x5 neighbourhoods found
exactly that one mismatch for seed 0, and zero mismatches for seed 3.

**Seed 3 is fully exact: 0 errors out of all 33,554,432 possible 5x5
neighbourhoods, a closed proof, not sampled evidence.** A fine-tune attempt
on seed 0 to repair its single failing pattern did not reach zero
mismatches without regressing elsewhere, and was abandoned as a negative
result (gradient-based surgical fixes on an already-converged, highly
discretized circuit don't constrain the size of the resulting population
shift at the flipped decision boundary).

**Resolution: seed 3, already exact, is the deliverable.** It is used
directly as the "exact" model, recompiled and re-verified:
`pytorch/bitslice_models/life_2step_cnn_c32_seed3_exact.json` (model dump,
identical weights to the seed-3 entry in the original training run's
output, just saved under its own name), re-checked exhaustively
(`pytorch/bitslice_models/life_2step_cnn_c32_seed3_exact_exhaustive.json`:
0 / 33,554,432, confirming the compiled, saved artifact itself, not just
the in-memory model object), and compiled to WGSL
(`web/compare/bitslice_life_2step_cnn_compiled_exact.wgsl`). Gate counts
match the table above's seed-3 row exactly, as expected since the weights
are unchanged.

Seed 0's uncorrected, 1-error model and its WGSL kernel
(`web/compare/bitslice_life_2step_cnn_compiled.wgsl`) are left in place for
the record (and because the single error is extremely rare in practice,
~1 in 1.5M random cells) but should not be presented as exact; seed 3's
`_exact` artifacts are the ones to use wherever an exact 2-step kernel is
needed.

## Data

Training: `pytorch/bitslice_life2.py`. Trained models (ternary, c=32, both
converged seeds): `pytorch/bitslice_models/life_2step_cnn_c32_train.json`.
Verification results: `pytorch/bitslice_models/
life_2step_cnn_c32_seed{0,3}_verify.json`. Compiled WGSL kernel (seed 0):
`web/compare/bitslice_life_2step_cnn_compiled.wgsl`. Binary-weight variant
training log (0/5 converged): `pytorch/bitslice_models/
life_2step_cnn_c32_binary_train.json`. Exhaustive-check records:
`pytorch/bitslice_models/life_2step_cnn_c32_seed{0,3}_exhaustive_before.json`.
Exact model and verification: `pytorch/bitslice_models/
life_2step_cnn_c32_seed3_exact.json`, `pytorch/bitslice_models/
life_2step_cnn_c32_seed3_exact_exhaustive.json`. Exact WGSL kernel:
`web/compare/bitslice_life_2step_cnn_compiled_exact.wgsl`.
