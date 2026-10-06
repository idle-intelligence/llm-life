# Lookup-table Game of Life: GPU (WGSL) vs CPU, 16x16 to the memory limit

Benchmark of the `life-lut-bench` crate (`crates/life-lut-bench`): a 512-entry
outer-totalistic lookup table keyed by the 9-bit neighbourhood, and a
bit-packed bitwise full-adder neighbour-count kernel, each implemented as CPU
(single-thread and rayon-threaded) and GPU (wgpu/WGSL) code. Correctness is
bit-exact against `life::grid::Grid::step` (the README's classical reference
engine) across several random grids, generations, a non-word-aligned width,
and a non-Conway rule (HighLife, B36/S23) — see `correctness` in
`crates/life-lut-bench/src/bin/main.rs`; all checks pass on both machines.

## Parameters

- Boundary: toroidal, matching `life::grid::Grid` (every cell has exactly 8
  neighbours, no edge special case).
- Rule under benchmark: Conway's Life, `B3/S23`.
- LUT encoding: one `u32` per cell on the GPU (WGSL has no native 1-byte
  storage-buffer element; a true 4-cells/`u32` byte-packed GPU buffer would
  need atomic read-modify-write and was out of scope — noted, not measured).
  The CPU LUT baseline is a true `u8`-per-cell buffer.
- Bit-packed encoding: 32 cells/`u32` on the GPU, 64 cells/`u64` on the CPU.
  Requires `width` divisible by the word size (32 GPU, 64 CPU); 16x16 is
  below the GPU word size and is not exercised for bit-packed rows.
- Workgroup size 256 on both kernels (1D dispatch, folded into a 2D
  workgroup grid once the workgroup count exceeds WebGPU's 65535-per-axis
  cap — first hit at 4096x4096, 65536 workgroups).
- CPU parallelism: rayon over rows, `rayon::current_num_threads()` (8
  logical cores on the Mac, 24 on the RTX 3080 machine). No SIMD intrinsics; only
  whatever LLVM auto-vectorizes from the scalar/bitwise loops.
- Timing: `warmup` untimed generations, then `generations` timed generations
  with ping-pong buffers, one readback at the very end (verified against
  `life::Grid::step` separately in `correctness`, not timed). Generation
  counts per grid size (same table on every machine, not tuned per device):

  | grid width  | warmup | timed generations |
  |---|---|---|
  | <=64        | 200 | 20,000 |
  | <=256       | 200 | 10,000 |
  | <=1024      | 200 | 2,000 |
  | <=4096      | 20  | 200 |
  | <=16384     | 2   | 20 |
  | larger      | 2   | 5 |

- GB/s formula: `2 x bytes_per_cell x cells x generations / seconds`, where
  `bytes_per_cell` is 1 (CPU LUT), 4 (GPU LUT, u32/cell), or 0.25
  (bit-packed, either machine, 4 bytes shared by 32 cells). This is the
  minimum-traffic model (one grid read + one grid write per generation); it
  does not credit the ~9x redundant neighbour reads a naive kernel performs
  out of cache, so it is a lower bound on cache/DRAM traffic actually moved.
- Machines: an RTX 3080 (10 GB, Vulkan backend, 24-core host — CPU
  numbers on that machine are not reported, since its CPU runs
  an unrelated background job) and a Mac laptop (Metal backend, 8-core CPU,
  unified memory) for both GPU and CPU baselines.

## Results — RTX 3080 (Vulkan), CPU baseline vs GPU

`max_storage_buffer_binding_size` reported by the adapter: 2,147,483,647
bytes (2 GiB - 1), well below the card's 10 GiB of VRAM — see Observations.

### LUT encoding (u32/cell GPU, u8/cell CPU)

| grid | cpu-1thread cells/s | cpu-1thread GB/s | cpu-rayon cells/s | cpu-rayon GB/s | gpu cells/s | gpu GB/s |
|---|---|---|---|---|---|---|
| 16x16 | 3.31e8 | 0.66 | 1.60e7 | 0.03 | 1.34e7 | 0.11 |
| 64x64 | 3.38e8 | 0.68 | 1.99e8 | 0.40 | 2.18e8 | 1.74 |
| 256x256 | 3.35e8 | 0.67 | 1.50e9 | 3.01 | 1.96e9 | 15.65 |
| 1024x1024 | 3.46e8 | 0.69 | 3.39e9 | 6.77 | 1.24e10 | 99.44 |
| 4096x4096 | 3.08e8 | 0.62 | 3.58e9 | 7.16 | 2.47e10 | 197.92 |
| 16384x16384 | 3.23e8 | 0.65 | 3.67e9 | 7.35 | 6.97e10 | 557.78 |
| 23000x23000 (max) | 3.43e8 | 0.69 | 3.68e9 | 7.36 | 7.63e10 | 610.28 |

### Bit-packed encoding (32 cells/u32 GPU, 64 cells/u64 CPU)

| grid | cpu-1thread cells/s | cpu-1thread GB/s | cpu-rayon cells/s | cpu-rayon GB/s | gpu cells/s | gpu GB/s |
|---|---|---|---|---|---|---|
| 64x64 | 3.94e9 | 0.99 | 1.99e8 | 0.05 | 2.11e8 | 0.05 |
| 256x256 | 4.59e9 | 1.15 | 2.16e9 | 0.54 | 3.42e9 | 0.85 |
| 1024x1024 | 4.94e9 | 1.24 | 1.87e10 | 4.68 | 5.24e10 | 13.11 |
| 4096x4096 | 5.26e9 | 1.31 | 4.99e10 | 12.47 | 4.40e11 | 109.91 |
| 16384x16384 | 5.08e9 | 1.27 | 5.82e10 | 14.56 | 7.95e10 | 19.87 |
| 122880x122880 (max) | 5.29e9 | 1.32 | 5.99e10 | 14.98 | 6.98e11 | 174.53 |

(16x16 is N/A for bit-packed: below the 32-cell GPU word width.)

## Results — Mac (Metal), CPU + GPU on the same machine

### LUT encoding

| grid | cpu-1thread cells/s | cpu-1thread GB/s | cpu-rayon cells/s | cpu-rayon GB/s | gpu cells/s | gpu GB/s |
|---|---|---|---|---|---|---|
| 16x16 | 3.60e8 | 0.72 | 1.47e7 | 0.03 | 3.49e6 | 0.03 |
| 64x64 | 3.73e8 | 0.75 | 1.49e8 | 0.30 | 5.29e7 | 0.42 |
| 256x256 | 4.18e8 | 0.84 | 9.30e8 | 1.86 | 8.94e8 | 7.15 |
| 1024x1024 | 4.24e8 | 0.85 | 2.11e9 | 4.21 | 3.67e9 | 29.35 |
| 4096x4096 | 3.51e8 | 0.70 | 9.59e8 | 1.92 | 4.25e9 | 34.03 |
| 16384x16384 | 3.67e8 | 0.73 | 1.75e9 | 3.50 | 4.50e9 | 36.01 |
| 32000x32000 (max) | 4.19e8 | 0.84 | 1.98e9 | 3.95 | 4.48e9 | 35.83 |

### Bit-packed encoding

| grid | cpu-1thread cells/s | cpu-1thread GB/s | cpu-rayon cells/s | cpu-rayon GB/s | gpu cells/s | gpu GB/s |
|---|---|---|---|---|---|---|
| 64x64 | 6.13e9 | 1.53 | 2.05e8 | 0.05 | 6.17e7 | 0.02 |
| 256x256 | 7.72e9 | 1.93 | 1.95e9 | 0.49 | 9.72e8 | 0.24 |
| 1024x1024 | 8.07e9 | 2.02 | 1.46e10 | 3.66 | 1.49e10 | 3.72 |
| 4096x4096 | 7.95e9 | 1.99 | 3.23e10 | 8.07 | 5.32e10 | 13.30 |
| 16384x16384 | 8.25e9 | 2.06 | 3.91e10 | 9.76 | 6.30e10 | 15.76 |
| 65536x65536 (max) | 8.01e9 | 2.00 | 3.90e10 | 9.75 | 6.35e10 | 15.87 |

Raw JSON for both runs: a scratch file (3080), another (Mac)
— not committed (scratch output of this session).

## Max grid per encoding, and the limiting formula

- LUT (u32/cell GPU, 2 ping-pong buffers, 8 bytes/cell total): the binding
  size cap `max_storage_buffer_binding_size` bounds a single buffer at
  `cells <= limit / 4`. On the 3080 (limit = 2^31-1) that is 536,870,911
  cells, sqrt ~= 23,170 — measured at 23000x23000. A 10 GiB *total-VRAM*
  budget at 8 bytes/cell would allow ~36,600x36,600 (1.34e9 cells): the
  per-binding cap binds first, well before the card's 10 GB VRAM does (see
  Observations). Splitting the grid across multiple bindings would remove
  this cap but was out of scope here.
- Bit-packed (u32/32-cells, 2 buffers, 0.25 bytes/cell): same binding cap,
  `words <= limit / 4`, i.e. `width^2 / 32 <= limit / 4` -> `width <=
  sqrt(32 * limit / 4)` ~= 131,068 on the 3080. Measured at 122,880x122,880
  (~15.1 billion cells, 1.9 GB per buffer) to stay under that cap with
  margin. A 10 GiB total-VRAM budget at 0.25 bytes/cell would allow
  ~207,000x207,000 — again the per-binding cap binds first.
- On the Mac, `max_storage_buffer_binding_size` was reported as 2^32-1
  (4,294,967,295), so the LUT max tested (32000x32000, 4.10e9 bytes/buffer)
  is memory-choice-bound, not binding-bound; 32768x32768 (exactly 2^32/4
  bytes/buffer) failed validation by one byte over the limit, confirming
  the reported cap is exact. Bit-packed max tested on the Mac: 65536x65536
  (1.07 GB/buffer), well under any binding limit, chosen for wall-clock
  time rather than a memory ceiling.

## Observations

- **GPU beats CPU, and by how much, is encoding- and size-dependent.**
  For the LUT encoding on the 3080, GPU overtakes CPU single-thread between
  64x64 (CPU 3.38e8 cells/s > GPU 2.18e8) and 256x256 (CPU 3.35e8 < GPU
  1.96e9) — a crossover between roughly 100 and 250 cells per side, close
  to the back-of-envelope's "CPU wins below ~300x300." Against CPU-rayon,
  the LUT crossover is earlier: rayon still wins at 16x16 (1.60e7 vs GPU
  1.34e7) but GPU is already ahead at 64x64 (2.18e8 vs rayon 1.99e8). At
  the largest measured LUT grid (23000x23000) the GPU is 222x CPU-1thread
  (7.63e10 vs 3.43e8 cells/s) and 20.7x CPU-rayon (7.63e10 vs 3.68e9).
- **The bit-packed CPU baseline is a much stronger competitor than the
  byte-LUT one**, because it processes 64 cells per scalar instruction on
  the CPU. On the 3080 machine, CPU-1thread bit-packed beats GPU bit-packed only
  at 64x64 and 256x256 (3.94e9 and 4.59e9 cells/s vs GPU's 2.11e8 and
  3.42e9); GPU is already ahead by 1024x1024 (5.24e10 vs CPU-1thread's
  4.94e9). CPU-rayon bit-packed is overtaken by GPU at the same point
  (rayon 1.87e10 vs GPU 5.24e10 at 1024x1024). At the largest measured
  bit-packed grid (122880x122880), GPU is 132x CPU-1thread and 11.7x
  CPU-rayon in cells/s.
- **GPU fixed launch cost matches the back-of-envelope order of magnitude,
  and differs sharply by backend.** From the 16x16 LUT timing (20,000
  generations, submit-per-generation with a blocking poll every 50): the
  3080 (Vulkan) averages 19.1 us/generation; the Mac (Metal) averages 73.4
  us/generation — both in the same ballpark as the "~10-20 us" estimate for
  a clean native backend, with Metal's per-dispatch/per-submit overhead
  costing roughly 4x more here (consistent with the higher per-encoder-pass
  cost `lean::Engine::flush_encoder`'s doc comment already documents for
  Metal). CPU-rayon shows a comparable fixed cost at tiny sizes: on the
  3080 machine, rayon's 16x16 LUT step averages ~16 us/generation (0.3195s /
  20,000), i.e. its thread-pool dispatch overhead is in the same range as
  a GPU launch — which is why rayon does not help, and sometimes loses to
  single-thread, at 16x16/64x64.
- **Achieved bandwidth vs device peak.** The RTX 3080's advertised memory
  bandwidth is ~760 GB/s (GDDR6X). The LUT kernel reaches 610.28 GB/s at
  23000x23000 (80% of peak) and 557.78 GB/s at 16384x16384 (73%) by the
  minimum-traffic GB/s formula above — higher utilization than the
  back-of-envelope's ~350 GB/s (46%) estimate. The bit-packed kernel does
  *not* approach peak bandwidth despite moving far less data (174.53 GB/s
  at its largest grid, 23% of peak): its bitwise full-adder network (9
  three/two-input adders plus a 9-way rule mux per word) makes it
  compute-bound rather than bandwidth-bound, which is also why its cells/s
  advantage over the LUT kernel (6.98e11 vs 7.63e10 at their respective max
  grids, ~9x) is smaller than the 32x cells-per-word packing ratio would
  suggest if it were purely bandwidth-bound.
- **The 10 GB card's actual ceiling, for an un-chunked buffer, is the
  driver's per-binding cap, not its VRAM.** `max_storage_buffer_binding_size`
  reported 2 GiB - 1 bytes on the 3080 (Vulkan) — the LUT and bit-packed
  kernels each hit that cap around 23,000 and 131,000 cells per side
  respectively, using only ~4.3 GB and ~3.8 GB of the card's 10 GB VRAM at
  those sizes (two ping-pong buffers). Reaching a 10 GB-VRAM-bound grid
  (~36,600x36,600 LUT, ~207,000x207,000 bit-packed) would require chunking
  the grid across multiple buffer bindings, which this benchmark does not
  implement.
- **A 512-entry table is a genuine "memory lookup at GPU speed", but the
  GPU's advantage over the CPU-1thread LUT baseline is bandwidth, not the
  lookup itself**: both machines' CPU-1thread LUT cells/s is flat across
  every grid size (3.1-3.5e8 on the 3080, 3.5-4.2e8 on the Mac) — the L1/L2
  table lookup itself is cheap and the CPU's per-cell cost is dominated by
  the 9 scattered neighbour reads, not compute. The GPU wins once the grid
  is large enough to amortize its fixed per-generation cost, by exploiting
  far higher aggregate memory bandwidth for the same scattered-read
  pattern, not by any advantage in the 512-entry lookup mechanism itself.
