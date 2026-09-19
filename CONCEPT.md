# llm-life — concept note

*2026-09-17. Written from a design session; nothing below is built yet.*

**One line:** a language model as the update rule of a cellular automaton. One pixel is one cell, one cell's next state is one forward pass over its neighborhood plus a shared rules prefix. Read logits, don't sample. The picture is the model's confidence per cell; the drift from true Life is the model's character rendered as an image.

Same family as "ask the LLM what time it is" (Hive/SWARM): take something a for-loop does perfectly, do it with 64k forward passes, and show the residual.

Sibling repo: `~/Code/jacobi2000` (physics track, floats not tokens). They share Burn-on-wgpu, the web glue, and eventually the stencil attention kernel. Nothing else.

---

## 1. The core mechanism

- **State** per cell: one token (`0`/`1` for Life; any word for open-ended variants).
- **Rule** lives in a shared text prefix, not in the weights. E.g. Life as a B/S rulestring: `B3/S23`.
- **Update**: each cell sees its 8 neighbors + itself + the prefix, and we read the logits at the answer position.
- **Output is a distribution, not a token.**
  - argmax → next binary state → diff against true Life → mutation count per generation.
  - softmax over the answer vocabulary → p(alive) → grayscale confidence map. This is the more interesting pixel.
  - temperature is just how you threshold/sample that gray.
- **Sliced lm-head**: only project hidden states onto the handful of answer tokens, never the full 151936 vocab. Kills the all-position-logits cost problem.
- **Activation = dirty list.** Only cells whose neighborhood changed get re-called (standard Life optimization). Stable regions cost nothing. The activation map is a second visualization: where the model is thinking.

Prefix example (base model, no chat template):
```
You are a cell in a grid. 1 = alive, 0 = dead.
A live cell with 2 or 3 live neighbors survives, otherwise dies.
A dead cell with exactly 3 live neighbors becomes alive.
Answer with one digit.
```
Per-cell suffix (variant A): `Neighbors: 1 0 1 0 0 1 1 0 / Self: 1 / Next:`

## 2. Variants

### A — 4096 LLMs called in parallel (build)
- Each cell gets its own tiny templated prompt (~10 tokens) after the shared prefix.
- **Implemented as packing, not batching**: one long token sequence, block-diagonal attention mask, per-cell restarting position ids, prefix resident from a KV image. No batch axis needed anywhere.
- Dense mask is fine up to ~4096 tokens. 4096 cells × 10 tokens = 40k tokens → either chunk into ~16 prefills against the resident prefix (zero new kernels) or use the sparse mask kernel from B.
- Cost is ~10× B. Needs a ~135M-class model to hit the budget in a tab. "Smaller and smaller until it fits."
- Target: **< 3–5 s per generation at 4096 cells in the browser.**

### B — one LLM whose attention is restricted to a 2D stencil (build, headline)
- One token per cell, grid serialized row-major, the grid *is* the context.
- Attention mask: token i attends to itself, its 8 grid neighbors, and the rules prefix. One forward pass = one generation for every cell simultaneously.
- Dense Burn-tensor mask up to 4096 cells. **65536 cells needs a custom WGSL stencil attention kernel** (each query reads 9 neighbor keys + prefix; never materialize 65536²).
- No per-cell template; the model must infer layout from raw adjacent tokens. Expect worse rule-following than A on base models; fine-tuning is where B shines.
- Story: "you didn't run 64k models, you made one model hallucinate that it's a cellular automaton."

### D — bidirectional masked LM with the same stencil (stretch, after A and B)
- Causal LLMs are the wrong shape for a CA (right neighbor is in the future). An encoder (ModernBERT-class) with the stencil mask predicts every cell from both sides. Architecturally honest, less funny. Probably fine-tunes best.

### Dropped
- **C** — full attention, no mask, width hint in prefix. Quadratic; "sucks."
- **E** — autoregressive baseline (whole grid as prompt, generate next grid token by token, ~4096 decode steps). "Sucks a bit too."

## 3. Cost model

Batched/packed prefill is compute-bound; single-stream 30 tok/s is irrelevant here.

| variant | tokens/gen (256×256) | M2, 135M–0.5B | H100 |
|---|---|---|---|
| A | ~650k | 30–60 s | <1 s |
| B | ~65k + prefix | 3–5 s | ~50 ms |

At 64×64: A ≈ 1–2 s, B ≈ sub-second, with a 0.5B model. B at 256×256 in a tab is live, not time-lapse.

## 4. Positions

Life is outer-totalistic: only "my state" + "how many neighbors alive" matter, never which neighbor. So **default = no positions**: give every grid token the same position id; RoPE is relative, all pairwise rotations vanish, neighbors become a bag. Self-vs-neighbor is carried by the mask + the query token.

Escalations, each earned by a rule that needs it:
1. Directional rules (sand falls, flow goes right): learned bias per neighbor offset (9 scalars/head, T5-relative-bias / Graphormer style). Fine-tunable.
2. Continuous geometry: 2D RoPE (vision-transformer style). Fights the 1D prior more.
3. Meshes: position as edge feature mixed into keys. (Lives in `blanks`.)

## 5. The matrix and the fine-tune

Four cells: **{A, B} × {base, fine-tuned}**.

- **Data**: generated from true Life (and other B/S rules). Millions of (neighborhood, next state) pairs, free.
- **Method**: LoRA on a 0.5B (or full FT on 135M), Burn autodiff, on M2 for small / 3080 or cloud for bigger. Same crate, native build. In-browser fine-tuning is a stretch flex.
- **For A** the model learns a 512-entry truth table → 100% → it's a lookup table. Exhibit = before/after + accuracy-per-generation training curve.
- **For B** the fine-tune teaches something real: that offset −W in a 1D sequence means "up." Training uses the same stencil mask (dense mask fine at 4096 tokens; kernel only needed for 65536 inference).
- **The chess-from-transcripts test**: rulestring in the prefix, train on a random family of B/S rules, hold some out, test on unseen rules. Held-out rule accuracy proves "read the rule, apply it locally" vs memorization. This is the result to aim at.

## 6. Engine

**Rust + WASM, native and browser from the first commit. Burn (to match llm-web).** No Python prototype phase.

Builds on `../llm-web` (Burn 0.20 + custom WGSL, Qwen2 arch, GGUF, tokenizer, prefix KV images, OPFS, worker, demo template). Survey findings (2026-09-17):
- batch is asserted to 1 — irrelevant, we pack instead.
- prefill mask is a Burn tensor op at one call site (`model.rs` ~504–580) → swap in an arbitrary bool mask.
- all-position logits exist (`forward_hidden` → `lm_head`) but flagged "small T only" because of full-vocab materialization → sliced lm-head fixes it.
- prefix KV images fully built (`kvimg.rs`, `kv.rs`, `web.rs`) → the shared rules prefix.
- Cargo manifests are marked frozen; code is not.

**Engine features to add in llm-web, on a branch, as generic capabilities:**
1. per-token position ids (RoPE positions restart per cell in A; constant for B-bag mode)
2. pluggable attention mask (caller-supplied bool tensor; later a sparse/stencil kernel)
3. sliced lm-head (project onto K answer tokens only)
4. all-position logits readback via the sliced head
5. Llama architecture (for SmolLM2 135M/360M — needed by A's budget)
6. 65536-cell stencil attention WGSL kernel (shared with `blanks`)

`llm-life` depends on `llm-web` as an rlib and owns: CA logic, packing/mask construction, the autodiff-capable training forward (pure Burn ops, no custom kernels, LoRA adapters), the grid UI, evals.

**Models**: Qwen2.5-0.5B base first (already loads via Qwen2 path). SmolLM2-135M for A. 1.5B/3B on the 3080/cloud for the remote 65536 grid and as a "does a bigger base model follow Life better untrained" row.

**Remote**: the 65536 grid runs the *same crate's native binary* on a 3080 / rented GPU and streams frames to the browser. No second stack.

## 7. Demo

- Template: STATUS / INPUT / OUTPUT / PERFORMANCE, like the other `*-web` demos. No settings, no URL inputs.
- Paint cells, hit step, watch the model try to run Life. Every mistake visible.
- Panels: model grid (confidence gray), true-Life grid, diff, activation map, accuracy trace per generation.
- Knobs: model, variant (A/B), temperature. Same seed diverges differently per model.
- Hosting: repo + models under idle-intelligence (HF), demo mirrored at trucs.ai.

## 8. Measurables (convergence criteria)

- seconds per generation, per variant × model × grid size (tab and native)
- rule accuracy per generation vs true Life (base and fine-tuned)
- held-out rulestring accuracy after fine-tuning
- (optional) cells activated per generation vs Life's dirty count

## 9. Plan

**First step**
1. Branch llm-web; add features 1–4 (position ids, pluggable mask, sliced lm-head, all-position logits). Native CLI test: Qwen2.5-0.5B, B at 64×64, dense mask, read p(alive) grid, diff against true Life. First picture.

**Next**
2. A at 64×64 via packing + chunked prefill against the resident prefix. Time it. Shrink model (wire Llama/SmolLM2) until < 3–5 s.
3. Grid UI in the tab (headless Chromium first). Both variants.
4. Data generator + training forward + LoRA in Burn. Fine-tune A and B on Life. Before/after.
5. Rule family + held-out rules experiment.
6. 65536 stencil kernel. B at 256×256 in the tab; remote grid on the 3080.
7. Write-up (do the prior-art search only now, not before).

**Future / ideas parked** (same engine, different prefix + vocabulary)
- Physics, badly: diffusion ("average your neighbors"), falling sand, Ising/voting (which word wins = model's prior).
- Language as a field: rumor propagation (spatial telephone, color by embedding), story grid (rows read as sentences), translation gradient (English left edge, French right edge, where's the boundary), rhyme/alliteration poem CA.
- Image things: LLM-as-blur-kernel vs real Gaussian, "a cat" by consensus, reaction-diffusion in tokens.
- Computation: Wireworld / logic gates on unreliable cells (generations until an adder lies), odd-even transposition sort with an LLM comparator, BFS by peer pressure.
- Meta: left half one model / right half another (the boundary), checkerboard of temperatures, cells whose rule is itself a mutable neighbor-editable state, second logit read for explicit confidence.
- D (masked LM encoder) variant.
- In-browser fine-tuning.

## 10. Framing for the write-up

- A transformer is message passing on whatever graph the mask defines. Language = chain, ViT/Swin = grid/windows, ESM/AlphaFold = residue graph, TabPFN = row/column stencil, Neural CA = conv-as-rule. This is a transformer on a grid graph.
- Novelty: a model pretrained on a *different* graph (chain) is forced onto a grid at inference; the rule is in context (in-context learning on a graph); held-out rules test reading vs memorizing; it runs in a tab.
- Prior-art policy (decided 2026-09-17): don't search before building; search before writing up. A is almost certainly done somewhere as a blog post; B + fine-tune + browser is the specific combination.

## 11. Reframing (owner, 2026-09-19 evening) — three poles, one write-up

The project illustrates three fundamentally different ways to compute the same thing:
1. **Scaling laws / overpowered thinking** — ONE big LLM, the whole board as text, the four rules, unlimited thinking, writes the next board. Variant E, revived: run on the 3080 box's local Qwen3.6-35B (llama-server, reasoning budget) against the same seeds; measure wrong cells per generation and tokens per correct cell. The tab shows the remote output.
2. **Find the physics** — variant B: attention shaped like the problem (stencil), one small pass per generation; measured in s/generation and per-case rule recall after fine-tuning.
3. **Brute force with the right data** — variant A: one LLM call per pixel with the 512-entry rule table learned (fine-tune A FIRST: "neighbors are 1 0 1 0 1 1 0 0, I am 1 →" should reach 100%); two adapters, rules in the prompt vs rules learned; and **BERT of Life** (backlog): a generated dataset of every neighborhood, a small encoder classifying 0/1 and which rule fired (underpopulation / survival / overpopulation / reproduction). "At its core, it's a data problem."

Order: fine-tune A → A in the tab with the adapter (the naive one-pixel-per-LLM must run) → thinking-model run on the 3080 → B retrain with a sane schedule (lr 3e-5, best-by-IoU checkpoint; run 1's collapse at step 60 was training dynamics, not B's locality) → BERT of Life.

## 12. "One pass", precisely (owner, 2026-09-19 late)

Two readings, both to do:
- **Numerical**: every cell is the same computation on the same input shape (8 neighbour bits + self after a shared prefix). (a) Batch it: 4096 identical sequences against one resident prefix = one forward with the block-diagonal sparse mask (~40k tokens), or one GEMM per layer with a true batch dimension in the engine. (b) NOT memoization (owner: the point is the honest cost of the stupid mode, not the cheapest way up the ladder); the 512-row table is a separate rung ("lookup"), and the LLM's behaviour can still be printed as such a table for the write-up.
- **Vector space**: no tokens. Input = the 3×3 patch as 9 numbers → centre; or the whole grid → the whole grid. A learned CA = jacobi2000's stencil model with one binary channel and a sigmoid/cross-entropy head. Two scales of the same experiment: the smallest model that fits the 512 cases (BERT of Life without text) vs a deliberately oversized model that must discover locality from grid pairs. Then **3D Life** is a 3×3×3 stencil with nothing else changed — the cheapest 3D generalization test for jacobi2000. Life becomes jacobi2000's first discrete dataset.
- Also still worth doing: the whole board as text to one thinking LLM (§11 pole 1).

## 13. The compute ladder (owner, 2026-09-19 late) — measure, don't optimise

Same 64×64 board, one generation, native and wasm, seconds (or ns/cell) per rung:
| rung | what | expected |
|---|---|---|
| CPU | plain Rust Life step | ns/cell |
| lookup | 9-bit index into a 512-entry table | ns/cell, memory-bound — no faster than the loop |
| BERT of Life | small encoder per cell, batched | ms/board |
| LLM, stupid mode | one 0.5B call per cell, all 4096 as ONE batched pass (shared prefix KV + batch 4096 × 10 suffix tokens → one GEMM per layer, weights read once; attention ~80 keys per cell) | ~28 TFLOP/generation ≈ 10 s M2, ~1 s 3080 (today: 9–16 min in 64-cell chunks) |
| LLM, whole board as text, thinking | one big model, remote (3080) | minutes; tokens per correct cell |

Engine work for the batched pass = a true batch dimension in llm-web's prefill (batched attention kernel index + KV layout; the matmul is already a GEMM) — the same batch axis t0-web needs for 1000 signals.

Grid-size axis (owner): measure ns/cell at 64², 256², 1024², 4096² for the loop and the lookup, native and wasm. Prediction: equal in L1 (64²); the lookup pulls ahead at scale when it stops being per-cell — rolling 9-bit index (one read per cell), then 12-bit tables over 3×4 windows (two cells per hit), then 16-bit strip tables on bit-packed rows (several cells per load, 8× less traffic). The gap is memory bandwidth, not arithmetic. HashLife is the far end of the same axis (memoizing regions in space and time) and belongs in the write-up.

## 14. The demo narrates the computation (owner, 2026-09-19 late)

Each rung of the ladder is a mode in the tab, same board and seed, and the STATUS log shows HOW it computed, not just the result:
- **LLM, one call per pixel**: one line per cell — `cell (0,1): neighbours 0 1 0 0 1 0 0 0 · self 1 → p(alive) 0.62 → 1 · 48 ms` — with running count and total.
- **LLM, batched**: same questions, mechanism visible — `prefix 79 tokens, KV resident (once)` · `batch 4096 × 10 tokens` · per layer `GEMM 40960×896, weights read once` · `attention: 80 keys/cell` · `readback` · `total 9.4 s` — and identical answers to the per-pixel mode (assert it).
- **BERT of Life**: `4096 cells → one batch → 12 ms` (+ rule fired per cell if the second head exists).
- **lookup / CPU**: `4096 cells → 3 µs`, ns/cell.
PERFORMANCE keeps one row per rung; the ladder is readable as a table after running each mode. The per-pixel and batched logs ship first.
