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


## Larger grids (8192, 16384), 2026-10-12

Question: do the rates of the earlier table hold at 8192x8192 and 16384x16384, and where does a method stop working.

### Parameters

- Same machine, browser build, page, board (torus, random, seed 1, density 0.28) and `window.__bench` hook as above. Playwright's bundled Chromium 1243 (`channel: 'chromium'`, new headless), `--use-angle=metal`; the adapter reported vendor apple, architecture metal-3, isFallbackAdapter false, maxBufferSize and maxStorageBufferBindingSize 4294967292; `crossOriginIsolated` true; page served by `web/serve.py` (COOP/COEP). The built `web/pkg` was used as it was.
- Sizes 4096 (re-run with these settings), 8192, 16384. Methods: the six listed in the tables. Not run: GPU lookup with read-back, stencil, LLM, BERT and 9-numbers rows.
- Harness: a Node script with Playwright (not in the repo) that opens the page, calls `__bench.setSize(n)`, then per method: one 1-generation warm-up call, one 1-generation call to estimate the time per generation, N = ceil(0.75 s / estimate) capped at 20000, then the run is repeated with a smaller or larger N until one run of N generations lasts at least 0.7 s (N is capped at 20000). Then 5 timed calls of `__bench.run(method, N)`, each reporting `totalSeconds` (the same timed region as above). The tables give the median of the 5 and the minimum and maximum in brackets. One browser, one page per size, one method at a time.
- Before each method the 1-minute load average was read with `uptime` and had to be at or below 2.5 (otherwise the script waits and retries); other processes were checked and memory free percentage read with `memory_pressure` (60 to 76 percent free throughout, no point stopped for pressure). A Chrome for Testing 1248 process tree, language servers and a local dev server that this run did not start were present (their load was not measured).
- Correctness: after the timed runs, a run of C generations compared cell by cell, in the page, against the CPU rule run for C generations from the same board. C = min(N, 20) at 4096 and 8192, which gives C = N for the CPU methods and C = 1 or 2 where N is 1 or 2. At 16384 C = min(N, 2) for every method, to keep the CPU reference short.
- N is below the 20 generation correctness minimum for the CPU methods at 8192 and 16384 (N = 1 or 2, each timed run lasting 1.1 to 5.6 s), and for GPU lookup kept on the GPU at 8192 (N = 1).
- At 16384 the page's `drawBoard` throws `RangeError: Invalid array length` (it converts the 268 million cell board to a JavaScript array), so for that size only, the harness served the page with `drawBoard` returning at once when the width is above 8192. The drawing is outside every timed region. No file in the repo was changed for this.
- The GPU lookup kept on the GPU run includes a final read-back that converts the board with `Array.from(new Uint32Array(...), ...)` inside its timed region.

### Results, cells per second, median (minimum to maximum of 5)

| method | 4096 | 8192 | 16384 |
|---|---|---|---|
| CPU rule | 5.02e+07 (4.99e+07 to 5.04e+07) | 4.9e+07 (4.88e+07 to 4.92e+07) | 4.86e+07 (2.97e+07 to 4.9e+07) |
| CPU lookup table | 1.25e+08 (1.24e+08 to 1.26e+08) | 1.26e+08 (1.25e+08 to 1.26e+08) | 1.23e+08 (9e+07 to 1.23e+08) |
| GPU lookup, kept on GPU | 1.44e+09 (1.42e+09 to 1.46e+09) | 4.38e+07 (4.36e+07 to 4.45e+07) | error |
| bitwise GPU (hand-written) | 8.37e+10 (8.3e+10 to 8.42e+10) | 9.49e+10 (9.46e+10 to 9.52e+10) | 9.67e+10 (9.66e+10 to 9.78e+10) |
| compiled gates GPU (89-parameter ternary network) | 7.83e+10 (7.8e+10 to 7.87e+10) | 8.81e+10 (8.8e+10 to 8.83e+10) | 9e+10 (8.97e+10 to 9.09e+10) |
| 45-parameter ternary network, bit-packed GPU | 8.34e+10 (8.31e+10 to 8.41e+10) | 9.49e+10 (9.48e+10 to 9.53e+10) | 9.7e+10 (9.66e+10 to 9.75e+10) |

### Results, seconds per generation (median)

| method | 4096 | 8192 | 16384 |
|---|---|---|---|
| CPU rule | 0.334 | 1.37 | 5.52 |
| CPU lookup table | 0.134 | 0.533 | 2.18 |
| GPU lookup, kept on GPU | 0.0116 | 1.53 | error |
| bitwise GPU (hand-written) | 0.0002 | 0.000707 | 0.00277 |
| compiled gates GPU (89-parameter ternary network) | 0.000214 | 0.000761 | 0.00298 |
| 45-parameter ternary network, bit-packed GPU | 0.000201 | 0.000707 | 0.00277 |

### Results, generations timed per run (N)

| method | 4096 | 8192 | 16384 |
|---|---|---|---|
| CPU rule | 3 | 1 | 1 |
| CPU lookup table | 6 | 2 | 1 |
| GPU lookup, kept on GPU | 47 | 1 | error |
| bitwise GPU (hand-written) | 3731 | 1000 | 269 |
| compiled gates GPU (89-parameter ternary network) | 3488 | 987 | 254 |
| 45-parameter ternary network, bit-packed GPU | 3719 | 1061 | 269 |

### Results, correctness (C generations compared, result)

| method | 4096 | 8192 | 16384 |
|---|---|---|---|
| CPU rule | 3, True | 1, True | 1, True |
| CPU lookup table | 6, True | 2, True | 1, True |
| GPU lookup, kept on GPU | 20, True | 1, True | error |
| bitwise GPU (hand-written) | 20, True | 20, True | 2, True |
| compiled gates GPU (89-parameter ternary network) | 20, True | 20, True | 2, True |
| 45-parameter ternary network, bit-packed GPU | 20, True | 20, True | 2, True |

### Load average before / after each point (1 minute)

| method | 4096 | 8192 | 16384 |
|---|---|---|---|
| CPU rule | 2.33 / 2.27 | 2.11 / 2.23 | 2 / 1.81 |
| CPU lookup table | 2.27 / 2.25 | 2.23 / 2.11 | 1.81 / 1.71 |
| GPU lookup, kept on GPU | 2.25 / 2.13 | 2.11 / 2.16 | 1.71 / n/a |
| bitwise GPU (hand-written) | 2.13 / 2.12 | 2.16 / 2.17 | 1.74 / 1.76 |
| compiled gates GPU (89-parameter ternary network) | 2.12 / 1.95 | 2.17 / 2.07 | 1.76 / 1.66 |
| 45-parameter ternary network, bit-packed GPU | 1.95 / 2.11 | 2.07 / 2.14 | 1.66 / 1.51 |

### Failures

- GPU lookup, kept on GPU, 16384x16384: `page.evaluate: RangeError: Invalid array length at Array.from (<anonymous>) at gpuLookupResidentRun (compare/index.html:729:27)`. This is the final read-back conversion of 268 million cells to an array. No rate was obtained.
- No device loss, out of memory or buffer limit error occurred at any other point.

### Observations

- The bitwise kernel, the compiled gates and the 45-parameter bit-packed network keep rising: bitwise 8.4e10 (4096), 9.5e10 (8192), 9.7e10 (16384) cells per second; compiled gates 7.8e10, 8.8e10, 9.0e10; 45-parameter network 8.3e10, 9.5e10, 9.7e10.
- Compiled gates against bitwise: 0.94 (4096), 0.93 (8192), 0.93 (16384). The 45-parameter network against bitwise: 1.00 at all three sizes.
- The CPU rule (4.9e7 to 5.0e7) and the CPU lookup table (1.23e8 to 1.26e8) stay flat from 4096 to 16384. At 16384 one run of each CPU method had a longer time than the other four (CPU rule 9.05 s against 5.5 s, CPU lookup 2.98 s against 2.18 s), which sets the minimum rates.
- At 8192 the bitwise kernel (9.49e10) is 753 times the CPU lookup table (1.26e8) and 1940 times the CPU rule (4.9e7).
- GPU lookup kept on the GPU is 1.4e9 at 4096 (N = 47, earlier table 1.97e9 with N = 85) and 4.4e7 at 8192 (N = 1, 1.53 s for one generation including the read-back conversion), below the CPU lookup table at 8192, and it does not run at 16384.


## GPU timing without the final read-back (2026-10-13)

Question: how much of the earlier resident GPU figures was the final read-back of the board, and what are the rates when the timer stops when the GPU has finished the last generation.

### What changed in the timed region

- In `gpuLookupResidentRun`, `bitwiseResidentRun` (used for the hand-written bitwise kernel and for the 45-parameter bit-packed network) and `compiledResidentRun`, the timer now stops after the N chained dispatches have completed on the GPU: `await device.queue.onSubmittedWorkDone()` after the last submit, then the elapsed time is taken. The copy to the staging buffer, `mapAsync` and the conversion of the board happen after the timer, still in the same function, and the board is still returned for the correctness check. Upload stays outside the timed region as before.
- Two changes that do not touch the timing: the lookup run keeps the read-back board in a `Uint8Array` instead of converting 268 million cells with `Array.from` (which threw `RangeError: Invalid array length` at 16384), and `drawBoard` returns at once above 8192 columns (it threw the same error). Nothing else in the page changed; the page's per-method figures in its own table for these methods now also exclude the final read-back.
- Other methods (CPU, GPU lookup with read-back each generation, stencil, the 2-step and quantized rows) are unchanged and were not re-run. Their rows in the earlier tables stand.

### Parameters

- Same machine, browser and settings as the 8192 and 16384 section: Apple M2 (16 GB), Playwright's bundled Chromium (`channel: 'chromium'`, new headless), `--use-angle=metal`, adapter vendor apple, architecture metal-3, isFallbackAdapter not set (checked at every size, `crossOriginIsolated` true), page served by `web/serve.py` (COOP/COEP), board torus, random, seed 1, density 0.28. The page is served from source (`web/compare/index.html`); the built `web/pkg` was used as it was and no rebuild was needed.
- Harness as in the 8192 and 16384 section, with the method list reduced to the four resident GPU methods and no patched `drawBoard`: one warm-up call, one 1-generation call, N = ceil(0.75 s / estimate) capped at 20000, recalibrated until a run of N lasts at least 0.7 s or N reaches the cap, then 5 timed calls. Median with minimum and maximum of the 5. One browser, one page per size, one method at a time. The 1-minute load average had to be at or below 2.5 before each point (recorded below); `memory_pressure` free percentage was 61 to 77 percent throughout.
- Capped points (N = 20000) last less than 0.7 s because the timer no longer includes the read-back: 0.59 to 0.63 s at 16 to 1024 (shortest run of the 5 below). The calibration only stops early when N reaches the cap.
- Correctness: after the timed runs, a run of C generations compared cell by cell in the page against the CPU rule run for C generations from the same board. C = min(N, 20), and C = min(N, 2) at 16384.
- 16x16: the bitwise, compiled and 45-parameter kernels need a width that is a multiple of 32, so they have no point (a run at 16 reports a mismatch with the rule and is not a valid result).
- Raw per-run timings: `cells_per_s.csv` (column `timing` marks the convention) in the job directory, not in the repo.

### Results, cells per second, median (minimum to maximum of 5)

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU lookup, kept on GPU | 8.56e+06 (8.43e+06 to 8.62e+06) | 3.4e+07 (3.38e+07 to 3.42e+07) | 1.36e+08 (1.36e+08 to 1.36e+08) | 5.35e+08 (5.34e+08 to 5.38e+08) | 1.96e+09 (1.96e+09 to 2e+09) | 2.7e+09 (2.7e+09 to 2.7e+09) | 3.38e+09 (3.37e+09 to 3.38e+09) | 3.7e+09 (3.68e+09 to 3.71e+09) | 3.63e+09 (3.6e+09 to 3.64e+09) | 3.55e+09 (3.54e+09 to 3.55e+09) | 2.29e+09 (2.23e+09 to 2.44e+09) |
| bitwise GPU (hand-written) | n/a | 3.36e+07 (3.07e+07 to 3.43e+07) | 1.36e+08 (1.2e+08 to 1.37e+08) | 5.44e+08 (4.74e+08 to 5.49e+08) | 2.15e+09 (1.97e+09 to 2.2e+09) | 8.74e+09 (8.06e+09 to 8.78e+09) | 3.32e+10 (3.12e+10 to 3.35e+10) | 6.89e+10 (6.78e+10 to 6.9e+10) | 8.4e+10 (8.37e+10 to 8.41e+10) | 9.49e+10 (9.48e+10 to 9.52e+10) | 9.76e+10 (9.73e+10 to 9.8e+10) |
| compiled gates GPU (89-parameter ternary network) | n/a | 3.35e+07 (3.31e+07 to 3.44e+07) | 1.33e+08 (1.33e+08 to 1.36e+08) | 5.36e+08 (5.35e+08 to 5.47e+08) | 2.15e+09 (2.13e+09 to 2.19e+09) | 8.57e+09 (8.54e+09 to 8.66e+09) | 3.27e+10 (3.24e+10 to 3.36e+10) | 6.52e+10 (6.42e+10 to 6.53e+10) | 7.85e+10 (7.83e+10 to 7.89e+10) | 8.82e+10 (8.8e+10 to 8.85e+10) | 9.05e+10 (9.01e+10 to 9.12e+10) |
| 45-parameter ternary network, bit-packed GPU | n/a | 3.36e+07 (3.33e+07 to 3.42e+07) | 1.34e+08 (1.34e+08 to 1.36e+08) | 5.35e+08 (5.32e+08 to 5.4e+08) | 2.14e+09 (2.14e+09 to 2.19e+09) | 8.57e+09 (8.53e+09 to 8.74e+09) | 3.29e+10 (3.23e+10 to 3.32e+10) | 6.87e+10 (6.81e+10 to 6.89e+10) | 8.39e+10 (8.38e+10 to 8.41e+10) | 9.44e+10 (9.42e+10 to 9.45e+10) | 9.81e+10 (9.79e+10 to 9.86e+10) |

### Results, seconds per generation (median)

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU lookup, kept on GPU | 2.99e-05 | 3.01e-05 | 3.02e-05 | 3.06e-05 | 3.34e-05 | 9.71e-05 | 0.00031 | 0.00113 | 0.00462 | 0.0189 | 0.117 |
| bitwise GPU (hand-written) | n/a | 3.05e-05 | 3.02e-05 | 3.01e-05 | 3.05e-05 | 3e-05 | 3.15e-05 | 6.08e-05 | 0.0002 | 0.000707 | 0.00275 |
| compiled gates GPU (89-parameter ternary network) | n/a | 3.06e-05 | 3.07e-05 | 3.06e-05 | 3.05e-05 | 3.06e-05 | 3.21e-05 | 6.43e-05 | 0.000214 | 0.000761 | 0.00297 |
| 45-parameter ternary network, bit-packed GPU | n/a | 3.05e-05 | 3.05e-05 | 3.06e-05 | 3.06e-05 | 3.06e-05 | 3.18e-05 | 6.1e-05 | 0.0002 | 0.000711 | 0.00274 |

### Results, generations timed per run (N)

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU lookup, kept on GPU | 20000 | 20000 | 20000 | 20000 | 20000 | 7732 | 2430 | 661 | 153 | 38 | 6 |
| bitwise GPU (hand-written) | n/a | 20000 | 20000 | 20000 | 20000 | 20000 | 20000 | 12327 | 3584 | 1025 | 258 |
| compiled gates GPU (89-parameter ternary network) | n/a | 20000 | 20000 | 20000 | 20000 | 20000 | 20000 | 11574 | 3442 | 929 | 241 |
| 45-parameter ternary network, bit-packed GPU | n/a | 20000 | 20000 | 20000 | 20000 | 20000 | 20000 | 12315 | 3646 | 995 | 262 |

### Results, shortest of the 5 timed runs (seconds)

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU lookup, kept on GPU | 0.59 | 0.60 | 0.60 | 0.61 | 0.66 | 0.75 | 0.75 | 0.75 | 0.71 | 0.72 | 0.66 |
| bitwise GPU (hand-written) | n/a | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | 0.63 | 0.75 | 0.71 | 0.72 | 0.71 |
| compiled gates GPU (89-parameter ternary network) | n/a | 0.60 | 0.60 | 0.60 | 0.60 | 0.61 | 0.62 | 0.74 | 0.73 | 0.70 | 0.71 |
| 45-parameter ternary network, bit-packed GPU | n/a | 0.60 | 0.60 | 0.61 | 0.60 | 0.60 | 0.63 | 0.75 | 0.73 | 0.71 | 0.71 |

### Results, correctness (result, C generations compared)

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU lookup, kept on GPU | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=2) |
| bitwise GPU (hand-written) | n/a | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=2) |
| compiled gates GPU (89-parameter ternary network) | n/a | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=2) |
| 45-parameter ternary network, bit-packed GPU | n/a | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=20) | True (C=2) |

### Load average before / after each point (1 minute)

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU lookup, kept on GPU | 1.79 / 2.04 | 2.12 / 2.11 | 2.48 / 2.48 | 2.24 / 2.14 | 2.35 / 2.65 | 2.28 / 2.28 | 2.38 / 2.35 | 2.35 / 2.40 | 2.32 / 2.12 | 1.94 / 1.63 | 2.04 / 2.30 |
| bitwise GPU (hand-written) | n/a | 2.11 / 2.11 | 2.48 / 2.44 | 2.14 / 2.14 | 2.32 / 2.32 | 2.28 / 2.41 | 2.35 / 2.48 | 2.40 / 2.29 | 2.12 / 2.11 | 1.63 / 1.77 | 2.30 / 2.41 |
| compiled gates GPU (89-parameter ternary network) | n/a | 2.11 / 2.26 | 2.44 / 2.24 | 2.14 / 2.21 | 2.32 / 2.29 | 2.41 / 2.38 | 2.48 / 2.52 | 2.29 / 2.26 | 2.11 / 1.94 | 1.77 / 1.78 | 2.41 / 2.32 |
| 45-parameter ternary network, bit-packed GPU | n/a | 2.26 / 2.48 | 2.24 / 2.24 | 2.21 / 2.35 | 2.29 / 2.51 | 2.38 / 2.38 | 2.29 / 2.29 | 2.26 / 2.56 | 1.94 / 1.94 | 1.78 / 2.04 | 2.32 / 2.19 |

### Ratio of seconds per generation, earlier tables over this run

The earlier figure includes the final read-back, from the first table of this document up to 2048 (3 runs, older session) and from the 8192 and 16384 section at 4096 and above. A ratio above 1 is the weight of the read-back in the earlier timed region (1.10 means the read-back was about 9 percent of it). The earlier and the new figures come from different sessions, so differences of a few percent at small sizes are within run to run variation.

| method | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU lookup, kept on GPU | 1.06 | 1.14 | 1.18 | 1.31 | 1.29 | 1.19 | 1.22 | 1.12 | 2.51 | 81.02 | n/a |
| bitwise GPU (hand-written) | n/a | 1.10 | 1.05 | 1.12 | 1.15 | 1.28 | 1.36 | 1.02 | 1.00 | 1.00 | 1.01 |
| compiled gates GPU (89-parameter ternary network) | n/a | 1.10 | 1.09 | 1.08 | 1.13 | 1.25 | 1.34 | 1.03 | 1.00 | 1.00 | 1.00 |
| 45-parameter ternary network, bit-packed GPU | n/a | 1.03 | 1.10 | 1.13 | 1.16 | 1.27 | 1.34 | 1.02 | 1.01 | 0.99 | 1.01 |

### Observations

- Bitwise kernel, compiled gates and the 45-parameter network: the removal changes the figure by 1 to 15 percent up to 256x256, by a larger share at 512 and 1024 (the earlier time per generation was 1.25 to 1.36 times the new one), and by at most 3 percent from 2048 up (ratios 0.99 to 1.03). At 4096, 8192 and 16384 the read-back of a bit-packed board (a few MB to 32 MB) is negligible against N generations of 0.2 to 3 ms each.
- Rates without the final read-back: bitwise 3.3e10 at 1024, 6.9e10 at 2048, 8.4e10 at 4096, 9.5e10 at 8192, 9.8e10 at 16384 cells per second. Compiled gates against bitwise: 0.93 to 1.00 at every size from 32 up (0.93 at 8192 and 16384). The 45-parameter network against bitwise: 0.99 to 1.01 from 128 up.
- GPU lookup kept on the GPU: the read-back converted the board to an array, and the weight shows. Ratios of 1.1 to 1.3 up to 2048, 2.5 at 4096 (N = 47 earlier, 0.0116 s per generation earlier against 0.00462 s now) and 81 at 8192 (one generation of 1.53 s earlier, 0.0189 s per generation now). Its rate is 3.4e9 at 1024, 3.7e9 at 2048, 3.6e9 at 4096, 3.6e9 at 8192 and 2.3e9 at 16384 (N = 6, shortest run 0.66 s).
- GPU lookup kept on the GPU now passes the CPU lookup table (1.17e8 at 64, 1.25e8 flat) at 64x64 (1.36e8) and is above it at every larger size; at 8192 it is 28 times the CPU lookup table (1.26e8). It no longer fails at 16384.
- The bitwise kernel at 8192 is 753 times the CPU lookup table and at 16384 it is 794 times it (9.76e10 against 1.23e8 from the earlier section).
- Below 256x256 every resident method is about 30 microseconds per generation, 3.0e-5 to 3.1e-5 s for the four methods at 32, 64 and 128. This floor is the same with and without the read-back.
- The lookup kernel falls from 3.6e9 (8192) to 2.3e9 (16384). Its two buffers and the staging buffer are 1 GiB each at 16384 and the run was done with free memory down to 72 percent; no cause was investigated.
