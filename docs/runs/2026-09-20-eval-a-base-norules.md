# Variant A LoRA fine-tune — a-norules (B3/S23)

machine: Darwin 25.3.0 arm64
model: qwen2.5-0.5b-instruct-q4_0.gguf
rule: B3/S23
adapter: a-norules (no rule text in the prefix)
LoRA: rank 8, alpha 16, q/k/v/o
steps: 0, batch 64, lr 0.00003, seed 1
trainable parameters: 1081344
wall clock: 1111.9s

## Base (step 0)

loss 0.8037, accuracy 0.3809 (195/512)

| case | n | correct | frac |
|---|---|---|---|
| birth (dead, =3) | 56 | 35 | 0.6250 |
| survive-2 | 28 | 1 | 0.0357 |
| survive-3 | 56 | 11 | 0.1964 |
| death-lonely (<2) | 9 | 9 | 1.0000 |
| death-crowded (>3) | 163 | 88 | 0.5399 |
| stay-dead | 200 | 51 | 0.2550 |

## Loss

| step | loss | s |
|---|---|---|

## Held-out (every eval-every steps)

## Best-by-accuracy checkpoint (saved to `/tmp/lora-a-base-norules-discard.bin`)

accuracy 0.3809 (195/512), IoU 0.1354, alive recall 0.2951, dead recall 0.8534

| case | n | correct | frac |
|---|---|---|---|
| birth (dead, =3) | 56 | 35 | 0.6250 |
| survive-2 | 28 | 1 | 0.0357 |
| survive-3 | 56 | 11 | 0.1964 |
| death-lonely (<2) | 9 | 9 | 1.0000 |
| death-crowded (>3) | 163 | 88 | 0.5399 |
| stay-dead | 200 | 51 | 0.2550 |

