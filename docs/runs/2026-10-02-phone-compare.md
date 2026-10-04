# Compare page on a phone, 2026-10-02

Device: Android phone, Chrome, WebGPU. Page: web/compare/ (branch gpu-lookup, at 2301f7b/ef6d419), reached through a tunnel to a local server.
Board 1024x1024 (1,048,576 cells), random start, 1,000 chained generations per row. Every row below was exact: 1048576 / 1048576 cells correct.
One run per row, read off the page table. The first attempt failed on the GPU rows because the tab went to the background mid-run.

| method | s / generation | generations / s | cells / s |
|---|---:|---:|---:|
| Game of Life (rule) | 0.0357 | 28.0 | 2.9e7 |
| lookup table | 0.0228 | 43.9 | 4.6e7 |
| GPU lookup | 0.0584 | 17.1 | 1.8e7 |
| GPU lookup (resident) | 0.0010 | 965 | 1.0e9 |
| bitwise (GPU, resident) | 1.30e-4 | 7669 | 8.1e9 |
| ternary model, float (45 parameters) | 0.0031 | 319 | 3.4e8 |
| ternary model, integer adds (45 parameters) | 0.0031 | 324 | 3.4e8 |
| binary model, float (177 parameters) | 0.0102 | 98.4 | 1.0e8 |
| binary model, integer adds (177 parameters) | 0.0102 | 98.2 | 1.0e8 |
| ternary model, bit-packed (45 parameters) | 1.54e-4 | 6477 | 6.8e9 |
| ternary model, compiled (89 parameters) | 1.43e-4 | 6998 | 7.3e9 |
| ternary model, compiled, 2 steps per pass (9,601 parameters) | 0.0173 | 57.8 | 6.1e7 |

An earlier run the same day (before the compiled rows existed) gave bitwise 1.39e-4, bit-packed 1.58e-4, GPU lookup (resident) 9.88e-4 s per generation.
