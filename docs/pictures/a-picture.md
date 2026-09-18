# Variant A — packed per-cell prompts, native

model: ~/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 64x64 (torus)
mode: teacher-forced (each generation starts from true Life)
positions: restart at the prefix for every cell
mask: block-diagonal — resident prefix + the cell's own prompt
head: sliced to the two answer tokens (the prompt ends on a space)
prefix: 79 tokens (rules only), resident in the KV cache
chunk: 64 cells
Scored at two thresholds: p(alive) >= 0.5, and the grid median of
p(alive) (equivalently, the logit difference `1`-`0` calibrated to the
grid — the median hands the model the live fraction, so its live recall
is an upper bound, not an accuracy claim).

Timings provisional: the Metal GPU is shared with another repo's training job.

## seed glider

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0010 | 0.4000 | 0.7170 | 0.2000 | -0.2013 | 5 | 4091 | 956.35 |
| 2 | 0.0010 | 0.4000 | 0.7173 | 0.4000 | -0.2273 | 5 | 4091 | 950.27 |
| 3 | 0.0010 | 0.4000 | 0.7170 | 0.2000 | -0.2240 | 5 | 4091 | 519.12 |

## seed 1

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2998 | 0.5595 | 0.4524 | 0.4227 | -0.0945 | 1294 | 3022 | 568.51 |
| 2 | 0.2888 | 0.4753 | 0.4263 | 0.3346 | -0.1193 | 1073 | 2860 | 828.40 |
| 3 | 0.2439 | 0.4835 | 0.4224 | 0.3346 | -0.1382 | 1088 | 3061 | 946.51 |

## seed 2

| gen | accuracy | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2864 | 0.5172 | 0.4285 | 0.3870 | -0.1136 | 1305 | 2968 | 946.91 |
| 2 | 0.2866 | 0.4705 | 0.4170 | 0.3330 | -0.1246 | 1120 | 2856 | 942.53 |
| 3 | 0.2490 | 0.4692 | 0.4175 | 0.3342 | -0.1409 | 1104 | 3008 | 942.31 |

