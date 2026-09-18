# Variant B — stencil mask, native

model: ~/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 64x64 (torus)
mode: teacher-forced (each generation starts from true Life)
positions: bag (all grid tokens share one position id)
mask: prefix (causal) + self + 8 neighbors
head: sliced to the two answer tokens
prefix: 68 tokens (rules only)
Scored at two thresholds: p(alive) >= 0.5, and the grid median of
p(alive) (equivalently, the logit difference `1`-`0` calibrated to the
grid — the median hands the model the live fraction, so its live recall
is an upper bound, not an accuracy claim).

Timings provisional: the Metal GPU is shared with another repo's training job.

## seed glider

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9988 | 0.0000 | 0.5125 | 1.0000 | +0.1266 | 5 | 0 | 53.49 |
| 2 | 0.9988 | 0.0000 | 0.5164 | 1.0000 | +0.1127 | 5 | 0 | 52.00 |
| 3 | 0.9988 | 0.0000 | 0.5181 | 1.0000 | +0.1266 | 5 | 0 | 51.74 |

## seed 1

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6841 | 0.0000 | 0.8162 | 1.0000 | +0.0455 | 1294 | 0 | 53.07 |
| 2 | 0.7368 | 0.0000 | 0.6670 | 0.8183 | +0.0234 | 1073 | 5 | 52.80 |
| 3 | 0.7344 | 0.0000 | 0.7659 | 1.0000 | +0.0490 | 1088 | 0 | 51.86 |

## seed 2

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6814 | 0.0000 | 0.8188 | 1.0000 | +0.0435 | 1305 | 0 | 53.13 |
| 2 | 0.7253 | 0.0000 | 0.6462 | 0.7670 | +0.0210 | 1120 | 5 | 52.20 |
| 3 | 0.7288 | 0.0000 | 0.7698 | 1.0000 | +0.0432 | 1104 | 7 | 52.89 |

