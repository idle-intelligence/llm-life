# Variant A LoRA fine-tune — a-norules

machine: x86_64
model: qwen2.5-0.5b-instruct-q4_0.gguf
adapter: a-norules (no rule text in the prefix)
LoRA: rank 8, alpha 16, q/k/v/o
steps: 60 (held-out eval set 100% correct, early stop), batch 64, lr 0.00003, seed 1
trainable parameters: 1081344
wall clock: 2808.2s

## Base (step 0)

loss 0.8660, accuracy 0.3594 (46/128)

Note (fixed post-hoc, 2026-09-20): this run predates the fix to the doc
generator's denominator, which hardcoded `/512` even though the
before-training and periodic evals below use `--eval-cases 128` (default).
The base/step-20/40/60 lines here have been corrected by hand to the true
`/128` denominator and grid count, and so has the "(512/512, early stop)"
header above (it was the held-out 128-case set that hit 100%, not the full
512, which the final check below shows at 99.61%). The per-case tables and
the final "Best-by-accuracy checkpoint" section were always correct.

| case | n | correct | frac |
|---|---|---|---|
| birth (dead, =3) | 35 | 25 | 0.7143 |
| survive-2 | 0 | 0 | 1.0000 |
| survive-3 | 0 | 0 | 1.0000 |
| death-lonely (<2) | 0 | 0 | 1.0000 |
| death-crowded (>3) | 0 | 0 | 1.0000 |
| stay-dead | 93 | 21 | 0.2258 |

## Loss

| step | loss | s |
|---|---|---|
| 1 | 0.8159 | 213.1 |
| 2 | 0.5817 | 239.5 |
| 3 | 0.6437 | 266.3 |
| 4 | 0.5308 | 292.9 |
| 5 | 0.5762 | 319.5 |
| 6 | 0.6289 | 346.2 |
| 7 | 0.5248 | 372.8 |
| 8 | 0.5474 | 399.5 |
| 9 | 0.5230 | 426.0 |
| 10 | 0.5203 | 452.4 |
| 11 | 0.4843 | 479.0 |
| 12 | 0.5398 | 505.5 |
| 13 | 0.4350 | 531.7 |
| 14 | 0.4267 | 558.1 |
| 15 | 0.3240 | 584.9 |
| 16 | 0.4716 | 611.4 |
| 17 | 0.3083 | 637.9 |
| 18 | 0.3340 | 664.2 |
| 19 | 0.4016 | 690.5 |
| 20 | 0.3970 | 717.4 |
| 21 | 0.3454 | 901.6 |
| 22 | 0.2603 | 927.9 |
| 23 | 0.3330 | 954.5 |
| 24 | 0.2216 | 980.7 |
| 25 | 0.2614 | 1007.3 |
| 26 | 0.1638 | 1033.6 |
| 27 | 0.2017 | 1060.2 |
| 28 | 0.1746 | 1086.7 |
| 29 | 0.1007 | 1113.1 |
| 30 | 0.1830 | 1139.5 |
| 31 | 0.4662 | 1165.7 |
| 32 | 0.1105 | 1192.1 |
| 33 | 0.1174 | 1218.4 |
| 34 | 0.0767 | 1244.9 |
| 35 | 0.1998 | 1271.0 |
| 36 | 0.1077 | 1297.4 |
| 37 | 0.0903 | 1323.9 |
| 38 | 0.1012 | 1350.3 |
| 39 | 0.0731 | 1376.6 |
| 40 | 0.1274 | 1402.9 |
| 41 | 0.1368 | 1586.4 |
| 42 | 0.0552 | 1612.8 |
| 43 | 0.0367 | 1639.1 |
| 44 | 0.0094 | 1665.5 |
| 45 | 0.0665 | 1691.8 |
| 46 | 0.0271 | 1718.3 |
| 47 | 0.0350 | 1744.8 |
| 48 | 0.0114 | 1771.1 |
| 49 | 0.0060 | 1797.2 |
| 50 | 0.0208 | 1823.7 |
| 51 | 0.0018 | 1850.3 |
| 52 | 0.0044 | 1876.9 |
| 53 | 0.0012 | 1903.5 |
| 54 | 0.0090 | 1930.0 |
| 55 | 0.0853 | 1956.7 |
| 56 | 0.0006 | 1983.3 |
| 57 | 0.0450 | 2009.9 |
| 58 | 0.0039 | 2036.4 |
| 59 | 0.0588 | 2065.9 |
| 60 | 0.0121 | 2092.5 |

## Held-out (every eval-every steps)

### step 20

loss 0.4832, accuracy 0.6250 (80/128), IoU (1 real grid) 0.1765

| case | n | correct | frac |
|---|---|---|---|
| birth (dead, =3) | 35 | 5 | 0.1429 |
| survive-2 | 0 | 0 | 1.0000 |
| survive-3 | 0 | 0 | 1.0000 |
| death-lonely (<2) | 0 | 0 | 1.0000 |
| death-crowded (>3) | 0 | 0 | 1.0000 |
| stay-dead | 93 | 75 | 0.8065 |

### step 40

loss 0.1054, accuracy 0.9766 (125/128), IoU (1 real grid) 0.6667

| case | n | correct | frac |
|---|---|---|---|
| birth (dead, =3) | 35 | 32 | 0.9143 |
| survive-2 | 0 | 0 | 1.0000 |
| survive-3 | 0 | 0 | 1.0000 |
| death-lonely (<2) | 0 | 0 | 1.0000 |
| death-crowded (>3) | 0 | 0 | 1.0000 |
| stay-dead | 93 | 93 | 1.0000 |

### step 60

loss 0.0129, accuracy 1.0000 (128/128), IoU (1 real grid) 1.0000

| case | n | correct | frac |
|---|---|---|---|
| birth (dead, =3) | 35 | 35 | 1.0000 |
| survive-2 | 0 | 0 | 1.0000 |
| survive-3 | 0 | 0 | 1.0000 |
| death-lonely (<2) | 0 | 0 | 1.0000 |
| death-crowded (>3) | 0 | 0 | 1.0000 |
| stay-dead | 93 | 93 | 1.0000 |

## Best-by-accuracy checkpoint (saved to `artifacts/lora-a-norules.bin`)

accuracy 0.9961 (510/512), IoU 0.9822

| case | n | correct | frac |
|---|---|---|---|
| birth (dead, =3) | 56 | 56 | 1.0000 |
| survive-2 | 28 | 28 | 1.0000 |
| survive-3 | 56 | 56 | 1.0000 |
| death-lonely (<2) | 9 | 7 | 0.7778 |
| death-crowded (>3) | 163 | 163 | 1.0000 |
| stay-dead | 200 | 200 | 1.0000 |

