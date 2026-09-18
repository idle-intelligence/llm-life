# Variant A — packed per-cell prompts, native

model: ~/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 64x64 (torus)
mode: teacher-forced (each generation starts from true Life)
positions: restart at the prefix for every cell
mask: block-diagonal — resident prefix + the cell's own prompt
head: sliced to the two answer tokens (the prompt ends on a space)
prefix: 255 tokens (rules + 6 worked examples), resident in the KV cache
chunk: 64 cells
Scored at two thresholds: p(alive) >= 0.5, and the grid median of
p(alive) (equivalently, the logit difference `1`-`0` calibrated to the
grid — the median hands the model the live fraction, so its live recall
is an upper bound, not an accuracy claim).

Timings provisional: the Metal GPU is shared with another repo's training job.

## seed glider

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9985 | 0.6000 | 0.5911 | 1.0000 | +0.1843 | 5 | 7 | 861.33 |

## seed 2

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.6172 | 0.5747 | 0.5986 | 0.6536 | +0.0380 | 1305 | 1763 | 846.86 |

