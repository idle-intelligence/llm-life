# Pictures

Output of `llm-life picture` (variant B, native). Four PGMs per seed at
generation 1, plus `first-picture.md` with the per-generation table.

| file | what it is |
|---|---|
| `b-<seed>-gen1-palive.pgm` | the model's p(alive) per cell, as gray — the interesting pixel |
| `b-<seed>-gen1-argmax.pgm` | p(alive) thresholded at 0.5 — what the model "says" |
| `b-<seed>-gen1-true.pgm` | true Life's next generation from the same input |
| `b-<seed>-gen1-diff.pgm` | cells where argmax and true Life disagree |

PGM (binary P5, 8-bit gray, black = alive) because it needs no dependency and
every viewer and every Python script reads it. `zoom` is baked in so a 64x64
grid is legible without a zooming viewer.

Accuracy alone is a bad headline number: a Life grid is mostly dead, so
answering `0` everywhere already scores ~90%. `first-picture.md` therefore
also reports live recall (accuracy restricted to cells true Life says are
alive) and the confidence gap (mean p(alive) on truly-live cells minus mean
p(alive) on truly-dead cells, i.e. whether the model separates them at all).
