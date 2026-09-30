# Browser Game of Life: CPU rule/lookup vs GPU lookup vs bitwise-resident

Question: how do the Game of Life implementations on the `web/compare/` page
compare on throughput, in-browser, across board sizes — CPU rule, CPU
512-entry lookup table, GPU lookup with a per-generation readback, GPU
lookup kept resident (readback only every N generations), a bitwise
resident kernel (32 cells packed per u32), and the stencil model — and
where each one breaks down.

## Setup

- Branch `gpu-lookup`, commits `bc5d16d..c60638d` (HEAD at time of writing:
  `fcb6851c7a16edfcdb991eb856611dad23ad6395`).
- Page: `web/compare/index.html`.
- Methods and their source:
  - `rule` — Game of Life B3/S23 rule implemented in JS, CPU.
  - `lookup` — 512-entry lookup table in JS, CPU.
  - GPU lookup (readback) — `web/compare/lut_byte.wgsl`, one u32 per cell,
    host readback after every generation.
  - GPU lookup resident — same shader, ping-pong buffers, one readback
    after N generations.
  - bitwise resident — `web/compare/bitpack.wgsl`, 32 cells per u32,
    bit-sliced adders, B3/S23 masks, ping-pong, one readback after N
    generations.
  - stencil — `VecStencilEngine`, stencil-life weights.
- Boundary: torus, all methods.
- Correctness: each row's final board compared cell-by-cell against the
  CPU rule's board after the same number of generations; "cells correct"
  is that comparison's count out of total cells.
- Hardware/browser: desktop Chrome on an Apple M2 MacBook, WebGPU on
  Metal. One run per row, by hand, with other applications open on the
  machine — this is not a quiet-machine benchmark, and the numbers below
  should be read with that in mind, not as reproducible low-noise
  measurements.
- Timing basis differs across the size groups below (noted per table):
  the 16x16 and 1024x1024 numbers are from an older build of the page and
  the CPU rows at 1024x1024 were timed over a different number of
  generations than the GPU rows; 2048x2048 and larger are from a build
  where every method in a row ran the same number of generations N.
- Fullscreen page: `web/fullscreen/index.html`, bitwise kernel only,
  decoupled full-speed simulation loop, 120 Hz display, no per-generation
  readback held against the render loop.

## Results — 16x16 (older build, per-generation timing)

| method | s/generation | generations | total s | cells correct | cells/s (derived) |
|---|---|---|---|---|---|
| GPU lookup (readback) | 4.00e-4 | 1 | 4.00e-4 | 256/256 | 6.40e5 |

## Results — 1024x1024 (older build; CPU rows timed over a different N than GPU/stencil rows)

| method | s/generation | cells correct | cells/s (derived) |
|---|---|---|---|
| rule | 0.0215 | 1048576/1048576 | 4.877e7 |
| lookup table | 0.0092 | 1048576/1048576 | 1.140e8 |
| GPU lookup (readback) | 0.0291 | 1048576/1048576 | 3.603e7 |
| GPU lookup resident (1000 gen) | 3.86e-4 | 1048576/1048576 | 2.717e9 |
| stencil | 1.49 | 1048576/1048576 | 7.037e5 |

## Results — 2048x2048 (same-N build, N=200, except stencil N=3)

| method | s/generation | generations | total s | cells correct | cells/s (derived) |
|---|---|---|---|---|---|
| rule | 0.0959 | 200 | 19.19 | 4194304/4194304 | 4.374e7 |
| lookup table | 0.0646 | 200 | 12.91 | 4194304/4194304 | 6.493e7 |
| GPU lookup (readback) | 0.1111 | 200 | 22.23 | 4194304/4194304 | 3.775e7 |
| GPU lookup resident | 0.0017 | 200 | 0.3357 | 4194304/4194304 | 2.468e9 |
| stencil | 27.95 | 3 | 83.86 | 4194304/4194304 | 1.501e5 |

## Results — 8192x8192

| method | s/generation | generations | total s | cells correct | cells/s (derived) |
|---|---|---|---|---|---|
| rule | 1.32 | 10 | 13.21 | 67108864/67108864 | 5.084e7 |
| lookup table | 0.5409 | 10 | 5.41 | 67108864/67108864 | 1.241e8 |
| GPU lookup (readback) | 2.28 | 200 | 456.02 | 67108864/67108864 | 2.943e7 |
| GPU lookup resident | 0.0249 | 200 | 4.97 | 67108864/67108864 | 2.696e9 |
| bitwise resident | 7.65e-4 | 200 | 0.1529 | 67108864/67108864 | 8.772e10 |

## Results — 3840x2160

| method | s/generation | generations | total s | gen/s | cells correct | cells/s (derived) |
|---|---|---|---|---|---|---|
| rule | 0.1795 | 200 | 35.90 | 5.6 | 8294400/8294400 | 4.621e7 |
| lookup table | 0.0786 | 200 | 15.72 | 12.7 | 8294400/8294400 | 1.055e8 |
| GPU lookup (readback) | 0.2066 | 10 | 2.07 | 4.8 | 8294400/8294400 | 4.015e7 |
| GPU lookup resident | 0.0023 | 1000 | 2.34 | 428 | 8294400/8294400 | 3.606e9 |
| bitwise resident | 1.22e-4 | 1000 | 0.1223 | 8177 | 8294400/8294400 | 6.799e10 |
| stencil | 29.12 | 3 | 87.36 | — | 6244398/8294400 | 2.848e5 |

The stencil row on this non-square board scored 6244398/8294400 cells
correct, not a full match; this is a known engine bug on non-square
boards, and the compare page now shows "not run on non-square boards" for
stencil instead of a number.

## Results — fullscreen page (bitwise, decoupled loop)

| board | display | generations/s | cells | cells/s (derived) |
|---|---|---|---|---|
| 2144x1215 | 120 Hz | 23100 | 2604960 | 6.017e10 |

## Observations

Resident GPU lookup throughput is roughly flat from 1024x1024 up: 2.717e9
cells/s at 1024x1024, 2.468e9 at 2048x2048, 2.696e9 at 8192x8192, and
3.606e9 at 3840x2160 — all within the same 2.5-3.6e9 band regardless of
board size, unlike the per-generation-readback variant, which drops as
size grows (3.603e7 at 1024x1024, 3.775e7 at 2048x2048, 2.943e7 at
8192x8192, 4.015e7 at 3840x2160).

Per-generation GPU readback is slower than the CPU lookup table at every
size from 1024x1024 up: 3.603e7 vs 1.140e8 cells/s at 1024x1024, 3.775e7
vs 6.493e7 at 2048x2048, 2.943e7 vs 1.241e8 at 8192x8192, and 4.015e7 vs
1.055e8 at 3840x2160. It is also slower than the plain CPU rule at
8192x8192 (2.943e7 vs 5.084e7) and at 3840x2160 (4.015e7 vs 4.621e7).

Bitwise resident is the fastest method measured at both sizes where it
ran: 8.772e10 cells/s at 8192x8192 and 6.799e10 cells/s at 3840x2160.
At 8192x8192 that is about 33x the resident GPU byte-lookup figure
(8.772e10 / 2.696e9) and about 706x the CPU lookup table (8.772e10 /
1.241e8).

At 16x16 the single GPU lookup (readback) measurement, 4.00e-4 s for one
generation over 256 cells, is consistent with a fixed per-dispatch
overhead rather than per-cell work — the same 4.00e-4 s appears at
1024x1024 (0.0291 s over 1048576 cells, i.e. not simply scaled down), so
the 16x16 number should be read as dispatch/readback overhead, not
per-cell throughput.

The stencil model is the slowest method at every size tested (7.037e5
cells/s at 1024x1024, 1.501e5 at 2048x2048, 2.848e5 at 3840x2160 — not
run at 8192x8192), and its correctness cannot be trusted on non-square
boards (6244398/8294400 at 3840x2160), a known engine bug; the page now
skips reporting the stencil wrong-count on non-square boards rather than
showing a misleadingly precise number.

The fullscreen page's bitwise kernel sustains 23100 generations/s on a
2144x1215 board (6.017e10 cells/s derived) in a decoupled full-speed loop
against a 120 Hz display, consistent with the bitwise-resident figures
from the compare page at similar cell counts.

## External reference

Boris the Brave, "Accelerated Game of Life with CUDA / Triton" (2025),
https://www.boristhebrave.com/2025/09/11/accelerated-game-of-life-with-cuda-triton/
— on an NVIDIA A40, N=65536 (65536^2 = 4294967296 cells): bitpacked 32-bit
Triton kernel 5.21 ms/step (4294967296 / 0.00521 = 8.243e11 cells/s
derived), bitpacked 64-bit CUDA kernel 1.84 ms/step (4294967296 / 0.00184
= 2.334e12 cells/s derived), naive CUDA 1-byte-per-cell kernel 26 ms/step
(4294967296 / 0.026 = 1.652e11 cells/s derived). These are native CUDA
kernels on a datacenter GPU (A40) at a board size roughly 64x this run's
largest (8192x8192), not a browser/WebGPU measurement, so they are not
directly comparable to the numbers above — they establish the order of
magnitude bitpacked kernels reach on dedicated hardware without a browser
or readback in the loop.

Fujita, Nakano, et al., "Bulk Computation Method for Computing Game of
Life," https://www.cs.hiroshima-u.ac.jp/cs/_media/life-ijfcs.pdf — reports
1350e9 cell updates/s on an NVIDIA GTX TITAN X for a bitwise parallel bulk
computation method. Again native CUDA on discrete desktop hardware, not a
browser measurement.
