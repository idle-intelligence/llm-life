# Variant B — stencil mask, native

model: Qwen2.5-0.5B.Q4_0.gguf
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
| 1 | 0.9988 | 0.0000 | 0.6389 | 1.0000 | +0.1265 | 5 | 0 | 17.43 |

## seed 1

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6841 | 0.0000 | 0.7092 | 0.8308 | +0.0450 | 1294 | 0 | 16.16 |

## seed 2

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6814 | 0.0000 | 0.7026 | 0.8176 | +0.0473 | 1305 | 0 | 16.46 |

