# Variant B — stencil mask, native

model: ~/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 64x64 (torus)
mode: teacher-forced (each generation starts from true Life)
positions: bag (all grid tokens share one position id)
mask: prefix (causal) + self + 8 neighbors
head: sliced to the two answer tokens
prefix: 244 tokens (rules + variant A's 6 worked examples)
Scored at two thresholds: p(alive) >= 0.5, and the grid median of
p(alive) (equivalently, the logit difference `1`-`0` calibrated to the
grid — the median hands the model the live fraction, so its live recall
is an upper bound, not an accuracy claim).

Timings provisional: the Metal GPU is shared with another repo's training job.

## seed glider

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9988 | 0.0000 | 0.5137 | 1.0000 | +0.1940 | 5 | 0 | 52.47 |
| 2 | 0.9988 | 0.0000 | 0.5090 | 1.0000 | +0.1699 | 5 | 0 | 49.72 |
| 3 | 0.9988 | 0.0000 | 0.5156 | 1.0000 | +0.1940 | 5 | 0 | 50.44 |

## seed 1

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6768 | 0.0000 | 0.8162 | 1.0000 | +0.0701 | 1294 | 30 | 50.42 |
| 2 | 0.6687 | 0.0000 | 0.6670 | 0.8183 | +0.0400 | 1073 | 284 | 50.22 |
| 3 | 0.7075 | 0.0000 | 0.7659 | 1.0000 | +0.0777 | 1088 | 110 | 50.48 |

## seed 2

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6704 | 0.0000 | 0.8188 | 1.0000 | +0.0694 | 1305 | 45 | 56.14 |
| 2 | 0.6626 | 0.0000 | 0.6467 | 0.7679 | +0.0373 | 1120 | 262 | 50.64 |
| 3 | 0.6975 | 0.0000 | 0.7698 | 1.0000 | +0.0697 | 1104 | 135 | 50.40 |

