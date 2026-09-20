# Runtime-LoRA IoU bisect: browser 0.307 vs native 1.000, 2026-09-20

Symptom: `lora-a-norules-300.bin` on the Q4_0 base gives 512/512 + IoU 1.000
natively (`llm-wasm`'s `lora-eval` bin, Vulkan/3080) but IoU 0.307 in the
tab (WebGPU/Metal, `LifeEngine::stepChunkA`), per
`docs/runs/2026-09-20-runtime-lora.md` and `docs/runs/2026-09-20-tab.md`.

## Step 1: chunked path natively

`crates/llm-wasm/src/bin/lora_eval.rs` (llm-web worktree) already *is* the
chunked path: `pack_chunk` builds the exact block-diagonal mask
`variant_a::pack_chunk` builds (prefix in full + causal within each cell's
own block, positions restarting at `prefix_len` per block), and
`ForwardSpec::with_allowed` + `KvCache::snapshot`/`restore` round-trip is
identical to `LifeEngine::step_chunk_a`. Ran it with the default
`--chunk-cells 64` (matching the tab's `CHUNK_CELLS`): **512/512, IoU
1.0000**, same as the per-cell/chunk-1 numbers in
`docs/runs/2026-09-20-runtime-lora.md`. So the sparse/dense mask + LoRA
interaction, positions, and snapshot/restore are all correct — the chunked
attention path itself is not the bug. Did not need a Metal-native rerun;
the divergence is upstream of attention (see below).

## Step 2: found before reaching Metal-specific debugging

`crates/llm-wasm/src/bin/lora_eval.rs` builds its prefix from
`norules_prefix()` / `rules_prefix()` **with no few-shot text** — same as
`crates/llm-life/src/train/run_a.rs::prompt_a`, which is what the adapter
was actually trained/evaluated against (`docs/runs/2026-09-20-a-rollout.md`:
a-norules IoU 1.0000 at 16²/32², every seed, using `prompt_a`'s norules,
no-few-shot prefix).

`crates/llm-life/src/web.rs::LifeEngine::load` (browser path) instead always
built variant A's resident prefix as:

```rust
let mut prefix_a_text = variant_a::rules_prefix(&rule);
prefix_a_text.push_str(&variant_a::fewshot_examples());
```

— i.e. **the actual rule text stated explicitly, plus six few-shot
examples**, regardless of which adapter (`a-norules` or `a-rules`) was then
loaded via `loadAdapter`. The `a-norules-300` adapter's LoRA deltas were
fit to a one-line "answer with one digit" prefix and never saw rule text or
worked examples; every per-cell chunk in the tab was conditioned on a
resident prefix the adapter had never been tuned against. `loadAdapter`
itself (q/k/v/o delta application) was unaffected — same code path,
independent of mask type — so this is a prompt-construction bug, not an
attention/LoRA/Metal bug. Never needed to reach the per-cell-logit-diff or
f16-denormal steps in the bisect plan.

File/line: `crates/llm-life/src/web.rs`, `LifeEngine::load` (prefix_a
construction) and the missing prefix rebuild in the old `LifeEngine::
load_adapter`.

## Fix

`crates/llm-life/src/web.rs`:
- `LifeEngine` now keeps `rule: Option<Rule>` and `tokenizer: Option<Tokenizer>`
  after `load()` (previously the tokenizer was dropped).
- `load()`'s rules+few-shot prefix is now documented as the base-model-only
  fallback (no adapter loaded).
- `loadAdapter(bytes, norules: bool)` takes a new `norules` flag and, when it
  differs from the prefix currently resident, rebuilds `prefix_a` from
  `norules_prefix()`/`rules_prefix(&rule)` (no few-shot, matching
  `train::run_a::prompt_a`) and re-prefills `cache_a` before applying the
  LoRA weights.

`web/worker.js`: both `load` (initial adapter) and `loadAdapter` (adapter
switch) now derive `norules` from the adapter filename
(`name.includes('norules')`) and pass it through.

## Verification

- Native 512-case lookup (`lora-eval --norules`, unaffected by this fix):
  **512/512 = 1.0000**, 16×16 generation **IoU 1.0000**.
- Chunked-native == per-cell-native: same binary/path for both (step 1
  above) — confirmed equal.
- Browser (headless Chromium, `node scripts/headless/llm-narrated.mjs`,
  16×16 seed 1, `a-norules`, WebGPU/Metal): **IoU 1.0000**, `acc 1.000`,
  `live 83 (game of life 83)`, `wrong 0` — matches
  `docs/runs/2026-09-20-a-rollout.md`'s native number exactly. Adapter
  switch to `a-rules` (`LoRA adapter applied (..., rules prefix)`) also
  round-trips cleanly, no reload of the base model.

`docs/runs/2026-09-20-tab.md` should be updated with the corrected IoU
1.0000 / timing (superseding its "Anomaly" section, which was correct at
the time but is now resolved).
