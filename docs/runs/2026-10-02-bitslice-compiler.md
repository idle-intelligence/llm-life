# A generic compiler from trained low-bit networks to bit-sliced code

Compiled with tritsliced, a private compiler: a trained network with
binary inputs, ternary weights and binary inter-layer activations turns
into a straight-line program of AND/OR/XOR/NOT/shift over 32-bit words --
cost scaling with parameters, not with 2^fan-in. Prototyped and proven on
Conway's Game of Life.

## Test (a): Conway Life, 1 step, 3x3

Trained `ThresholdNet(9, [8])` (9 inputs -> 8 hidden step units -> 1 output
step unit). Compiled network (`pytorch/bitslice_models/life_3x3_1step.json`
-> `web/compare/bitslice_life_compiled.wgsl`): 2 layers, 8 then 1 output
units.

Exactness, compiled program vs. the ground-truth rule:

| check | cases | mismatches |
|---|---:|---:|
| exhaustive 3x3 (all 512 neighbourhoods) | 512 | 0 |
| random 64x64 boards, 100 generations | 409,600 cells | 0 |

Gate count:

| program | and | or | xor | not | total |
|---|---:|---:|---:|---:|---:|
| this compiler, trained 9->8->1 net | 36 | 18 | 28 | 10 | 92 |

## Test (b): Life two steps in one pass, 5x5 / 25 inputs

The already-verified exact 1-step network composed with itself (10
invocations of the proven-exact 2-layer circuit) rather than training a
flat 25-input network, which did not converge. Inherits test (a)'s
exactness by construction.

Exactness (`web/compare/bitslice_life_2step_compiled.wgsl`):

| check | cases | mismatches |
|---|---:|---:|
| 20 random 64x64 boards, 2 generations each | 81,920 cells | 0 |
| glider (32x32, 2 generations) | 1,024 cells | 0 |
| blinker (32x32, 2 generations) | 1,024 cells | 0 |
| lightweight spaceship (32x32, 2 generations) | 1,024 cells | 0 |

Gate count: 10 x 92 = 920, vs. a 5x5/25-input lookup table's 2^25 =
33,554,432 entries.

## Test (c): HighLife (B36/S23) -- partial, time-boxed

Not compiled or verified exact -- training did not converge to exact
within the time budgeted for this optional test; reported as a
negative/partial result.

## Data

Trained model dump for test (a): `pytorch/bitslice_models/life_3x3_1step.json`.
Compiled kernels: `web/compare/bitslice_life_compiled.wgsl` (test a),
`web/compare/bitslice_life_2step_compiled.wgsl` (test b).
