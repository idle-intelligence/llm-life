# Variant A — packed per-cell prompts, native

model: /box/models/qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
grid: 32x32 (torus)
mode: teacher-forced (each generation starts from true Life)
positions: restart at the prefix for every cell
mask: block-diagonal — prefix (causal) + the cell's own prompt
head: sliced to the two answer tokens
forward: f32 training model (pure Burn ops), no persistent KV cache — full prefix+chunk forward every chunk
adapter: artifacts/lora-a-rules-300.bin, prefix: 79 tokens (rules)
Scored at two thresholds: p(alive) >= 0.5, and the grid median of
p(alive) (equivalently, the logit difference `1`-`0` calibrated to the
grid — the median hands the model the live fraction, so its live recall
is an upper bound, not an accuracy claim).

Timings provisional: the Metal GPU is shared with another repo's training job.

## seed 1

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9902 | 0.9682 | 0.9902 | 0.8018 | 1.0000 | +0.8918 | 307 | 311 | 12.31 |
| 2 | 0.9922 | 0.9686 | 0.9841 | 0.7480 | 1.0000 | +0.9170 | 251 | 251 | 10.41 |
| 3 | 0.9912 | 0.9651 | 0.9803 | 0.7549 | 1.0000 | +0.9059 | 254 | 253 | 10.40 |
| 4 | 0.9961 | 0.9831 | 0.9957 | 0.7285 | 1.0000 | +0.9283 | 233 | 235 | 10.41 |
| 5 | 0.9883 | 0.9508 | 0.9707 | 0.7344 | 1.0000 | +0.9048 | 239 | 237 | 10.43 |

## seed 2

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9893 | 0.9706 | 0.9891 | 0.8594 | 1.0000 | +0.8927 | 367 | 370 | 10.41 |
| 2 | 0.9961 | 0.9854 | 0.9890 | 0.7676 | 1.0000 | +0.9390 | 273 | 271 | 10.43 |
| 3 | 0.9893 | 0.9609 | 0.9854 | 0.7695 | 1.0000 | +0.9026 | 274 | 277 | 10.42 |
| 4 | 0.9961 | 0.9843 | 0.9921 | 0.7520 | 1.0000 | +0.9150 | 253 | 253 | 10.42 |
| 5 | 0.9932 | 0.9724 | 0.9920 | 0.7510 | 1.0000 | +0.9138 | 249 | 252 | 10.41 |

## seed 3

| gen | accuracy | IoU | live recall | acc (median) | live recall (median) | confidence gap | true live | model live | s/gen |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9854 | 0.9519 | 0.9834 | 0.7959 | 1.0000 | +0.8949 | 302 | 307 | 10.45 |
| 2 | 0.9971 | 0.9877 | 0.9959 | 0.7363 | 1.0000 | +0.9333 | 241 | 242 | 10.43 |
| 3 | 0.9941 | 0.9771 | 0.9922 | 0.7549 | 1.0000 | +0.9221 | 258 | 260 | 10.45 |
| 4 | 0.9951 | 0.9804 | 0.9921 | 0.7500 | 1.0000 | +0.9232 | 252 | 253 | 10.45 |
| 5 | 0.9941 | 0.9765 | 0.9842 | 0.7480 | 1.0000 | +0.9231 | 253 | 251 | 10.45 |

