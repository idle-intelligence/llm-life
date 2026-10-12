# Cells per second against grid size, compare page methods (2026-10-12)

Question: at what square grid size does each GPU method pass the best CPU method, and does the logic compiled from a ternary network keep up with the hand-written bitwise kernel at every size.

## Parameters

- Machine and browser: Apple M2 (16 GB), Playwright's bundled Chromium (Chrome for Testing 1243, new headless), real WebGPU through `--use-angle=metal` (adapter vendor apple, architecture metal-3, not a fallback adapter), page served with COOP/COEP headers.
- Page: `web/compare/index.html` built with `scripts/build.sh` (build tag dc5c126) plus a `window.__bench` hook added on this branch. The hook runs one method at one square size and returns the final board.
- Grids: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 (torus, random board, seed 1, density 0.28). Methods are stopped at a size where one generation would take over about 30 s.
- Generations per point: N found by calibration so that a timed run lasts at least 0.7 s, capped at 20000. Capped points (GPU lookup kept on the GPU up to 256x256, the bitwise-family kernels up to about 1024x1024) run 20000 generations for under 1 s, which is above the 20 generation minimum. Each point is the median of 3 timed runs of N generations (1 run for a point over 40 s). Slow points (stencil at 2048) run N=1.
- What is inside the timed region: CPU methods, the chain of N steps with no yield to the event loop. GPU lookup with read-back: N dispatches each followed by a full read-back and conversion to an array, after one warm-up. Resident GPU methods (lookup, bitwise, compiled gates, 45-parameter bit-packed): upload excluded, N chained dispatches with one submit each, then one read-back of the final board included. Stencil: N worker round trips of the whole grid.
- Correctness: the final board of a separate run of C generations is compared cell by cell against the CPU rule run for C generations from the same board. C = N up to 128x128, C = min(N, 20) above. The CPU rule row is its own reference at C = N.
- Methods: CPU rule and CPU lookup table are the page's JavaScript implementations. Compiled gates is `bitslice_life_compiled.wgsl`, the emitted gate list of the 89-parameter ternary network (92 logic operations on 32 cells per word). Bitwise is `bitpack.wgsl`. Bitwise, compiled gates and the bit-packed 45-parameter network need a width that is a multiple of 32, so they have no 16x16 point.
- The LLM, BERT of Life and 9-numbers-to-centre rows come from one "run every method" pass of the page per size (16 and 32 only), each value the page's own per-step figure (median of 3 when a step is under 2 s). Backend is the Burn engine on WebGPU.
- Raw per-run timings: `cells_per_s.csv` and `raw-main.log` in the job directory (not in the repo).

## Results, cells per second

| method | backend | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|
| CPU rule | CPU | 5.88e+07 | 5.87e+07 | 5.93e+07 | 5.84e+07 | 5.8e+07 | 5.66e+07 | 5.43e+07 | 5.18e+07 | 5.02e+07 |
| CPU lookup table | CPU | 1.09e+08 | 1.16e+08 | 1.17e+08 | 1.17e+08 | 1.21e+08 | 1.24e+08 | 1.25e+08 | 1.2e+08 | 1.25e+08 |
| GPU lookup, read back each generation | GPU | 7.4e+05 | 2.72e+06 | 8.89e+06 | 2.17e+07 | 3.63e+07 | 4.21e+07 | 4.36e+07 | 4.07e+07 | 3.77e+07 |
| GPU lookup, kept on GPU | GPU | 8.08e+06 | 2.98e+07 | 1.15e+08 | 4.09e+08 | 1.51e+09 | 2.26e+09 | 2.78e+09 | 3.3e+09 | 1.97e+09 |
| stencil model GPU | GPU | 6.21e+04 | 2.06e+05 | 5.91e+05 | 1.17e+06 | 1.22e+06 | 1.32e+06 | 1.68e+06 | 8.82e+05 | n/a |
| bitwise GPU (hand-written) | GPU | n/a | 3.05e+07 | 1.3e+08 | 4.85e+08 | 1.87e+09 | 6.81e+09 | 2.45e+10 | 6.78e+10 | 8.36e+10 |
| compiled gates GPU (89-parameter ternary network) | GPU | n/a | 3.05e+07 | 1.22e+08 | 4.95e+08 | 1.9e+09 | 6.83e+09 | 2.45e+10 | 6.35e+10 | 7.86e+10 |
| 45-parameter ternary network, bit-packed GPU | GPU | n/a | 3.27e+07 | 1.22e+08 | 4.75e+08 | 1.85e+09 | 6.76e+09 | 2.46e+10 | 6.75e+10 | 8.39e+10 |

## Results, seconds per generation (median)

| method | backend | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|
| CPU rule | CPU | 4.35e-06 | 1.75e-05 | 6.9e-05 | 0.000281 | 0.00113 | 0.00463 | 0.0193 | 0.081 | 0.334 |
| CPU lookup table | CPU | 2.35e-06 | 8.83e-06 | 3.49e-05 | 0.00014 | 0.000542 | 0.00212 | 0.00841 | 0.035 | 0.135 |
| GPU lookup, read back each generation | GPU | 0.000346 | 0.000376 | 0.000461 | 0.000755 | 0.0018 | 0.00623 | 0.024 | 0.103 | 0.445 |
| GPU lookup, kept on GPU | GPU | 3.17e-05 | 3.44e-05 | 3.56e-05 | 4.01e-05 | 4.33e-05 | 0.000116 | 0.000378 | 0.00127 | 0.00852 |
| stencil model GPU | GPU | 0.00412 | 0.00497 | 0.00693 | 0.014 | 0.0535 | 0.199 | 0.625 | 4.75 | n/a |
| bitwise GPU (hand-written) | GPU | n/a | 3.36e-05 | 3.16e-05 | 3.38e-05 | 3.5e-05 | 3.85e-05 | 4.28e-05 | 6.19e-05 | 0.000201 |
| compiled gates GPU (89-parameter ternary network) | GPU | n/a | 3.35e-05 | 3.36e-05 | 3.31e-05 | 3.45e-05 | 3.84e-05 | 4.29e-05 | 6.6e-05 | 0.000214 |
| 45-parameter ternary network, bit-packed GPU | GPU | n/a | 3.13e-05 | 3.36e-05 | 3.45e-05 | 3.54e-05 | 3.88e-05 | 4.26e-05 | 6.21e-05 | 0.0002 |

## Results, generations timed per run (N)

| method | backend | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|---|
| CPU rule | CPU | 20000 | 20000 | 11765 | 2740 | 715 | 182 | 48 | 12 | 3 |
| CPU lookup table | CPU | 20000 | 20000 | 20000 | 6667 | 1740 | 470 | 120 | 29 | 8 |
| GPU lookup, read back each generation | GPU | 2106 | 2688 | 2151 | 1220 | 414 | 136 | 35 | 9 | 3 |
| GPU lookup, kept on GPU | GPU | 20000 | 20000 | 20000 | 20000 | 20000 | 6072 | 2598 | 758 | 85 |
| stencil model GPU | GPU | 252 | 204 | 115 | 17 | 18 | 7 | 2 | 1 | n/a |
| bitwise GPU (hand-written) | GPU | n/a | 20000 | 20000 | 20000 | 20000 | 20000 | 20000 | 13663 | 4594 |
| compiled gates GPU (89-parameter ternary network) | GPU | n/a | 20000 | 20000 | 20000 | 20000 | 20000 | 19175 | 13626 | 4014 |
| 45-parameter ternary network, bit-packed GPU | GPU | n/a | 20000 | 20000 | 20000 | 20000 | 20000 | 18582 | 14376 | 4698 |

## Results, LLM and small models (16x16 and 32x32 only)

| method | 16x16 s/step | 16x16 cells/s | 16x16 correct | 32x32 s/step | 32x32 cells/s | 32x32 correct |
|---|---|---|---|---|---|---|
| LLM per cell, base | 42.8 | 5.98 | False | 174 | 5.9 | False |
| LLM per cell, trained | 16.6 | 15.4 | True | 66.6 | 15.4 | True |
| LLM per cell, trained, batched | 7.97 | 32.1 | True | 31.8 | 32.2 | True |
| LLM whole grid, base | 0.484 | 529 | False | 1.41 | 729 | False |
| LLM whole grid, trained | 0.499 | 513 | True | 1.45 | 705 | True |
| BERT of Life | 0.405 | 632 | True | 2.1 | 488 | True |
| BERT of Life, batched | 0.1 | 2.56e+03 | True | 0.113 | 9.1e+03 | True |
| 9 numbers to centre | 0.148 | 1.73e+03 | True | 0.626 | 1.64e+03 | True |
| 9 numbers to centre, batched | 0.0842 | 3.04e+03 | True | 0.104 | 9.82e+03 | True |

Every measured row matched the rule except the two base (untrained) LLM rows, which are not expected to.

## Observations

- Best CPU method: the lookup table, 1.1e8 to 1.25e8 cells per second at every size, flat. The rule is 5.0e7 to 5.9e7, also flat.
- GPU lookup with a read-back every generation never passes the CPU lookup table. It peaks at 4.4e7 cells per second at 1024x1024, below the CPU rule (5.4e7 at that size), and is far lower on small grids (7.4e5 at 16x16).
- GPU lookup kept on the GPU: 1.15e8 at 64x64 (equal to the CPU lookup table at 1.17e8) and 4.1e8 at 128x128, so the crossover with the best CPU method is at 64x64 to 128x128. It peaks at 3.3e9 at 2048x2048 and falls to 2.0e9 at 4096x4096 (that point runs only 85 generations).
- Hand-written bitwise kernel: 1.30e8 at 64x64 against 1.17e8 for the CPU lookup table, so it passes the best CPU method at 64x64 (at 32x32 it is at 3.1e7, below). It reaches 2.4e10 at 1024x1024, 6.8e10 at 2048x2048 and 8.4e10 at 4096x4096.
- Compiled gates: 1.22e8 at 64x64 against 1.17e8, so it also crosses at 64x64, by a margin that is within the run to run spread of the other points. From 32x32 to 4096x4096 its rate is between 0.93 and 1.02 times the bitwise kernel's at every size (64: 0.94, 1024: 1.00, 2048: 0.94, 4096: 0.94). The bit-packed 45-parameter network is within 0.97 to 1.07 of the bitwise kernel at every size.
- Below 128x128 the resident GPU methods are flat in time, not in rate: about 30 microseconds per generation at 16, 32 and 64 for the lookup, bitwise and compiled kernels alike, so cells per second grows in proportion to the number of cells. This is the cost of one submit per generation, not the kernel. The bitwise and compiled kernels leave that floor at about 512x512 (1.8e9 at 256, 6.8e9 at 512).
- The stencil model peaks at 1.7e6 cells per second (1024x1024) and never reaches the CPU rule. Its 2048x2048 point is noisy (2.8, 4.8 and 10.2 s for three runs of one generation); 4096x4096 was not run.
- The base LLM rows are wrong on both sizes (as an untrained model); the trained LLM rows are exact and run at 5 to 30 cells per second per-cell, 5e2 to 7e2 whole-grid.
