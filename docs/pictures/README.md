# Pictures

Output of `llm-life picture` (variant B) and `llm-life picture-a` (variant A),
both native. Four PGMs per seed at generation 1, plus a `<tag>-picture.md`
with the per-generation table.

| file | what it is |
|---|---|
| `<tag>-<seed>-gen1-palive.pgm` | the model's p(alive) per cell, as gray — the interesting pixel |
| `<tag>-<seed>-gen1-argmax.pgm` | p(alive) thresholded — what the model "says" |
| `<tag>-<seed>-gen1-true.pgm` | true Life's next generation from the same input |
| `<tag>-<seed>-gen1-diff.pgm` | cells where the thresholded grid and true Life disagree |

`<tag>` is `b` for variant B, `a` for variant A, and `a-<attempt>` for the
attempts at making `1` reachable (below).

PGM (binary P5, 8-bit gray, black = alive) because it needs no dependency and
every viewer and every Python script reads it. `zoom` is baked in so a 64x64
grid is legible without a zooming viewer.

Accuracy alone is a bad headline number: a Life grid is mostly dead, so
answering `0` everywhere already scores ~90%. `first-picture.md` therefore
also reports live recall (accuracy restricted to cells true Life says are
alive) and the confidence gap (mean p(alive) on truly-live cells minus mean
p(alive) on truly-dead cells, i.e. whether the model separates them at all).
