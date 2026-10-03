// All inference runs here: the main thread does UI only.
//
// The worker fetches the GGUF in chunks and pushes them into the engine
// (`appendModelShard`) rather than handing it one giant ArrayBuffer - the
// engine reads through a sharded cursor for exactly this reason. The dev
// server (web/serve.py, stdlib http.server) doesn't support Range requests,
// so this streams the single GET response and slices it into chunks itself.
// Version tag on the engine URLs: browsers keep a wasm module at a fixed path
// across rebuilds, even through a hard reload. Bump when either engine
// (pkg-llm, pkg-lean) changes.
const ENGINE_BUILD = '2026-10-03-cpu-01';
// A message posted to this worker before its top-level `await import` below
// finishes can be dropped rather than queued (observed in this browser: the
// page's first 'load' message, sent right after `new Worker(...)`, arrived
// while this module was still evaluating and was lost). Buffer every message
// that arrives before the real handler is attached, then replay it.
const pending = [];
self.onmessage = (e) => pending.push(e);

// The small from-scratch models run on the Burn engine (pkg-llm); the
// language model runs on lean (crates/llm-life-lean), imported only when the
// LLM is first loaded, on the backend pickLeanBackend picks by capability
// (lean-backend.js: WebGPU, else CPU threads, else single-thread CPU;
// `backend` in the load payload forces one).
const { default: init, BertEngine, VecMlpEngine, VecStencilEngine, initWgpuDevice } =
  await import(`./pkg-llm/llm_life.js?v=${ENGINE_BUILD}`);
const { pickLeanBackend, createLifeEngine } = await import(`./lean-backend.js?v=${ENGINE_BUILD}`);
let lean = null;
let leanPicked = null;
async function ensureLean(requested) {
  if (!leanPicked) {
    leanPicked = await pickLeanBackend(requested, ENGINE_BUILD);
    lean = leanPicked.mod;
  }
  return leanPicked;
}

let engine = null;
// Tracks the grid size last given to `engine.setGrid`, so a run of `step`
// messages at the same size doesn't call it redundantly.
let engineGrid = null;

// BERT of Life - a separate, much smaller wasm-bindgen engine (no GGUF, no
// tokenizer, no KV cache), loaded independently of `engine` above.
let bertEngine = null;
let bertGrid = null;

// Vector-space variants (CONCEPT.md §12), each its own tiny engine.
// (i) MLP on the 9 neighbourhood numbers, one forward per cell.
let vecMlpEngine = null;
let vecMlpGrid = null;
// (ii) whole grid in one forward pass, no per-cell loop.
let vecStencilEngine = null;
let vecStencilGrid = null;

const CHUNK = 64 * 1024 * 1024;

// The base GGUF and the tokenizer rarely change and are the two large,
// slow-to-fetch files (the LoRA adapters and tiny checkpoints are small
// enough that recaching them on every load is not worth the complexity).
// Bump the version if either file's contents change.
const MODEL_CACHE = 'llm-life-model-v1';

// Hugging Face only counts a model download when a request hits a query
// file - config.json for a repo with no known library (see
// huggingface.co/docs/hub/models-download-stats). Neither Hub repo here has
// one, so a real .bin fetch alone counts for nothing. Ping config.json once
// per repo per page load, result ignored, never through the Cache API so
// every load actually hits the Hub.
const HF_REPOS = [
  'https://huggingface.co/idle-intelligence/llm-of-life-lora',
  'https://huggingface.co/idle-intelligence/stencil-life',
];
const countedRepos = new Set();
function countDownload(url) {
  const repo = HF_REPOS.find((r) => url.startsWith(r + '/'));
  if (!repo || countedRepos.has(repo)) return;
  countedRepos.add(repo);
  fetch(`${repo}/resolve/main/config.json`, { cache: 'no-store' }).catch((err) =>
    console.error('[worker] download-count ping failed:', err));
}

function splitIntoChunks(buf) {
  const chunks = [];
  for (let off = 0; off < buf.length; off += CHUNK) {
    chunks.push(buf.subarray(off, off + CHUNK));
  }
  return chunks;
}

async function fetchChunks(url, onProgress, cacheable) {
  const cache = cacheable ? await caches.open(MODEL_CACHE) : null;
  if (cache) {
    const cached = await cache.match(url);
    if (cached) {
      onProgress(1);
      return splitIntoChunks(new Uint8Array(await cached.arrayBuffer()));
    }
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url}: ${r.status}`);
  const total = Number(r.headers.get('content-length'));
  const reader = r.body.getReader();
  const chunks = [];
  let buf = [], buffered = 0, loaded = 0;
  const flush = () => {
    const merged = new Uint8Array(buffered);
    let off = 0;
    for (const b of buf) { merged.set(b, off); off += b.length; }
    chunks.push(merged);
    buf = [];
    buffered = 0;
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.push(value);
    buffered += value.length;
    loaded += value.length;
    onProgress(total ? loaded / total : 0);
    if (buffered >= CHUNK) flush();
  }
  if (buffered > 0) flush();
  if (cache) {
    const total2 = chunks.reduce((n, c) => n + c.length, 0);
    const merged = new Uint8Array(total2);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.length; }
    try {
      await cache.put(url, new Response(merged.buffer, { headers: { 'Content-Type': 'application/octet-stream' } }));
    } catch (err) {
      console.warn('[worker] could not cache:', err);
    }
  }
  return chunks;
}

async function cachedFetchText(url) {
  const cache = await caches.open(MODEL_CACHE);
  const cached = await cache.match(url);
  if (cached) return cached.text();
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url}: ${r.status}`);
  const text = await r.text();
  try {
    await cache.put(url, new Response(text, { headers: { 'Content-Type': 'application/json' } }));
  } catch (err) {
    console.warn('[worker] could not cache:', err);
  }
  return text;
}

// Every `LifeEngine` method call (constructor aside) is chained through this
// single promise so two can never run concurrently - wasm-bindgen throws
// "recursive use of an object detected" if the page (or a stale in-flight
// call) lets two overlap. Each queued task catches its own errors and
// replies for its own message id, so one failure never breaks the chain for
// messages queued after it.
let queue = Promise.resolve();
function enqueue(task) {
  const run = queue.then(task, task);
  queue = run.then(() => {}, () => {});
  return run;
}

// Set by a 'stop' message and checked between cells/chunks by the per-cell
// loops below, so Clear (etc.) mid-generation actually stops issuing
// forwards instead of just having the page ignore whatever result eventually
// comes back. Handled outside `enqueue` so it lands immediately rather than
// waiting behind the in-flight step in the queue.
let stopRequested = false;

self.onmessage = (e) => {
  const { id, type, payload } = e.data;
  if (type === 'stop') { stopRequested = true; return; }
  const reply = (ok, result) => self.postMessage({ id, ok, result });
  enqueue(() => handle(id, type, payload, reply));
};
for (const e of pending) self.onmessage(e);
pending.length = 0;

// `init()` (the Burn wasm module) and `initWgpuDevice()` (the WebGPU device
// the three small engines share) are each idempotent to call once; guarded
// here so the small models load on their own. The LLM has its own module
// and device (`ensureLean` above).
let wasmInited = false;
async function ensureWasm() {
  if (wasmInited) return;
  await init({ module_or_path: new URL(`./pkg-llm/llm_life_bg.wasm?v=${ENGINE_BUILD}`, import.meta.url) });
  await initWgpuDevice();
  wasmInited = true;
}

async function handle(id, type, payload, reply) {
  try {
    if (type === 'load') {
      const picked = await ensureLean(payload.backend);
      // Free the previous engine's buffers now rather than whenever the JS
      // garbage collector gets to them: each one holds the whole model.
      if (engine) { engine.free(); engine = null; }
      engine = await createLifeEngine(picked, payload.width, payload.height);
      engineGrid = { width: payload.width, height: payload.height };
      const chunks = await fetchChunks(payload.ggufUrl, (f) =>
        self.postMessage({ type: 'progress', stage: 'download', fraction: f }), true);
      for (const c of chunks) engine.appendModelShard(c);
      const tokenizerJson = await cachedFetchText(payload.tokenizerUrl);
      await engine.load(tokenizerJson, payload.rulestring);
      // Runtime LoRA (lean's lora module - not the offline GGUF merge):
      // applies q/k/v/o deltas onto the already-loaded base Q4 model, no
      // reload.
      // Optional - if `adapterUrl` isn't set, the engine is byte-for-byte
      // the base model.
      let adapter = null;
      if (payload.adapterUrl) {
        countDownload(payload.adapterUrl);
        const bytes = new Uint8Array(await (await fetch(payload.adapterUrl)).arrayBuffer());
        const name = payload.adapterUrl.split('/').pop().replace(/\.bin$/, '');
        // Variant A's resident prefix must match how this adapter was
        // trained/evaluated (`norules_prefix()` vs `rules_prefix()`, no
        // few-shot) - `LifeEngine::loadAdapter` rebuilds it from this flag.
        await engine.loadAdapter(bytes, name.includes('norules'));
        adapter = { name, bytes: bytes.length };
      }
      reply(true, { packedTokens: engine.packedTokens(), cellTokens: engine.cellTokens(), adapter, backend: picked.backend, backendLabel: picked.label });
    } else if (type === 'loadBert') {
      await ensureWasm();
      bertEngine = new BertEngine(payload.width, payload.height);
      bertGrid = { width: payload.width, height: payload.height };
      countDownload(payload.checkpointUrl);
      const bytes = new Uint8Array(await (await fetch(payload.checkpointUrl)).arrayBuffer());
      bertEngine.loadCheckpoint(bytes, payload.dModel, payload.nLayers, payload.nHeads);
      reply(true, { dModel: bertEngine.dModel(), numLayers: bertEngine.numLayers() });
    } else if (type === 'loadVecMlp') {
      await ensureWasm();
      vecMlpEngine = new VecMlpEngine(payload.width, payload.height);
      vecMlpGrid = { width: payload.width, height: payload.height };
      countDownload(payload.checkpointUrl);
      const bytes = new Uint8Array(await (await fetch(payload.checkpointUrl)).arrayBuffer());
      vecMlpEngine.loadCheckpoint(bytes, payload.hidden);
      reply(true, { hidden: vecMlpEngine.hidden() });
    } else if (type === 'loadVecStencil') {
      await ensureWasm();
      vecStencilEngine = new VecStencilEngine(payload.width, payload.height);
      vecStencilGrid = { width: payload.width, height: payload.height };
      countDownload(payload.checkpointUrl);
      const bytes = new Uint8Array(await (await fetch(payload.checkpointUrl)).arrayBuffer());
      vecStencilEngine.loadCheckpoint(bytes, payload.dModel, payload.nLayers, payload.nHeads);
      reply(true, { dModel: vecStencilEngine.dModel(), numLayers: vecStencilEngine.numLayers() });
    } else if (type === 'loadAdapter') {
      // Swap the runtime LoRA adapter without a full model reload - replaces
      // whatever adapter is currently applied (LifeEngine::loadAdapter does
      // not stack).
      countDownload(payload.adapterUrl);
      const bytes = new Uint8Array(await (await fetch(payload.adapterUrl)).arrayBuffer());
      const name = payload.adapterUrl.split('/').pop().replace(/\.bin$/, '');
      await engine.loadAdapter(bytes, name.includes('norules'));
      reply(true, { name, bytes: bytes.length });
    } else if (type === 'step' && payload.variant === 'bert') {
      // BERT of Life: a separate, much smaller engine - one forward per
      // cell, no chunking (the model is tiny enough that per-cell round
      // trips are still fast), one 'bertCell' narration message per cell.
      const t0 = performance.now();
      if (!bertGrid || bertGrid.width !== payload.width || bertGrid.height !== payload.height) {
        bertEngine.setGrid(payload.width, payload.height);
        bertGrid = { width: payload.width, height: payload.height };
      }
      const cells = new Uint8Array(payload.cells);
      const n = cells.length;
      const p = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const cellT0 = performance.now();
        const v = await bertEngine.stepCell(cells, i);
        const ms = performance.now() - cellT0;
        p[i] = v;
        self.postMessage({ type: 'bertCell', index: i, p: v, ms });
      }
      self.postMessage({ type: 'progress', stage: 'threshold' });
      // BERT of Life is a calibrated 2-class classifier trained on the exact
      // rule, not a repurposed answer-token logit like the LLM variants -
      // 0.5 is the threshold it was trained and evaluated against
      // (`bert::train`'s `rollout_iou`), so this rung uses it directly
      // rather than the label-free Otsu/z-score thresholds the noisy LLM
      // rungs need. Matches the offline sweep's IoU 1.000
      // (docs/runs/2026-09-20-bert.md) instead of silently drifting from it.
      const thresholdValue = 0.5;
      const binarized = Array.from(p, (v) => (v >= thresholdValue ? 1 : 0));
      self.postMessage({ type: 'progress', stage: 'done' });
      reply(true, {
        pAlive: Array.from(p),
        binarized,
        thresholdValue,
        chunks: n,
        tokens: n * 9,
        seconds: (performance.now() - t0) / 1000,
      });
    } else if (type === 'step' && payload.variant === 'bert-batched') {
      // Same weights, same per-cell neighbourhoods as the row above, but
      // gathered into one [n, 9] batch and run as a single forward - this is
      // the number CONCEPT.md's "one batch" framing actually means, not the
      // per-cell loop's wall-clock total.
      const t0 = performance.now();
      if (!bertGrid || bertGrid.width !== payload.width || bertGrid.height !== payload.height) {
        bertEngine.setGrid(payload.width, payload.height);
        bertGrid = { width: payload.width, height: payload.height };
      }
      const cells = new Uint8Array(payload.cells);
      self.postMessage({ type: 'progress', stage: 'forward' });
      const p = await bertEngine.stepGrid(cells);
      const ms = performance.now() - t0;
      self.postMessage({ type: 'bertGrid', cells: cells.length, ms });
      self.postMessage({ type: 'progress', stage: 'threshold' });
      const thresholdValue = 0.5;
      const binarized = Array.from(p, (v) => (v >= thresholdValue ? 1 : 0));
      self.postMessage({ type: 'progress', stage: 'done' });
      reply(true, {
        pAlive: Array.from(p), binarized, thresholdValue,
        chunks: 1, tokens: cells.length * 9, seconds: (performance.now() - t0) / 1000,
      });
    } else if (type === 'step' && payload.variant === 'vec-mlp') {
      // (i) 9 numbers -> centre, no tokens: one forward per cell through the
      // MLP baseline, narrated the same way BERT of Life's per-cell mode is.
      const t0 = performance.now();
      if (!vecMlpGrid || vecMlpGrid.width !== payload.width || vecMlpGrid.height !== payload.height) {
        vecMlpEngine.setGrid(payload.width, payload.height);
        vecMlpGrid = { width: payload.width, height: payload.height };
      }
      const cells = new Uint8Array(payload.cells);
      const n = cells.length;
      const p = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const cellT0 = performance.now();
        const v = await vecMlpEngine.stepCell(cells, i);
        const ms = performance.now() - cellT0;
        p[i] = v;
        self.postMessage({ type: 'vecMlpCell', index: i, p: v, ms });
      }
      self.postMessage({ type: 'progress', stage: 'threshold' });
      const thresholdValue = 0.5;
      const binarized = Array.from(p, (v) => (v >= thresholdValue ? 1 : 0));
      self.postMessage({ type: 'progress', stage: 'done' });
      reply(true, {
        pAlive: Array.from(p), binarized, thresholdValue,
        chunks: n, tokens: n * 9, seconds: (performance.now() - t0) / 1000,
      });
    } else if (type === 'step' && payload.variant === 'vec-mlp-batched') {
      // Same weights as the per-cell row above, one [n, 9] batch instead.
      const t0 = performance.now();
      if (!vecMlpGrid || vecMlpGrid.width !== payload.width || vecMlpGrid.height !== payload.height) {
        vecMlpEngine.setGrid(payload.width, payload.height);
        vecMlpGrid = { width: payload.width, height: payload.height };
      }
      const cells = new Uint8Array(payload.cells);
      self.postMessage({ type: 'progress', stage: 'forward' });
      const p = await vecMlpEngine.stepGrid(cells);
      const ms = performance.now() - t0;
      self.postMessage({ type: 'vecMlpGrid', cells: cells.length, ms });
      self.postMessage({ type: 'progress', stage: 'threshold' });
      const thresholdValue = 0.5;
      const binarized = Array.from(p, (v) => (v >= thresholdValue ? 1 : 0));
      self.postMessage({ type: 'progress', stage: 'done' });
      reply(true, {
        pAlive: Array.from(p), binarized, thresholdValue,
        chunks: 1, tokens: cells.length * 9, seconds: (performance.now() - t0) / 1000,
      });
    } else if (type === 'step' && payload.variant === 'vec-stencil') {
      // (ii) whole grid -> whole grid, one channel: one forward pass, no
      // per-cell loop - the narration strip gets exactly one line.
      const t0 = performance.now();
      if (!vecStencilGrid || vecStencilGrid.width !== payload.width || vecStencilGrid.height !== payload.height) {
        vecStencilEngine.setGrid(payload.width, payload.height);
        vecStencilGrid = { width: payload.width, height: payload.height };
      }
      const cells = new Uint8Array(payload.cells);
      self.postMessage({ type: 'progress', stage: 'forward' });
      const p = await vecStencilEngine.stepGrid(cells);
      const ms = performance.now() - t0;
      self.postMessage({ type: 'vecStencilGrid', cells: cells.length, ms });
      self.postMessage({ type: 'progress', stage: 'threshold' });
      const thresholdValue = 0.5;
      const binarized = Array.from(p, (v) => (v >= thresholdValue ? 1 : 0));
      self.postMessage({ type: 'progress', stage: 'done' });
      reply(true, {
        pAlive: Array.from(p), binarized, thresholdValue,
        chunks: 1, tokens: cells.length, seconds: (performance.now() - t0) / 1000,
      });
    } else if (type === 'step') {
      const t0 = performance.now();
      // The engine packs for whatever grid it was last told about; variant B
      // runs at 64x64 and variant A at 16x16 off the same loaded weights.
      // Only call setGrid when the size actually changed - redundant calls
      // are dropped, and since this runs inside the serialized queue it
      // never overlaps a step already in flight.
      if (!engineGrid || engineGrid.width !== payload.width || engineGrid.height !== payload.height) {
        engine.setGrid(payload.width, payload.height);
        engineGrid = { width: payload.width, height: payload.height };
      }
      const cells = new Uint8Array(payload.cells);
      let p;
      if (payload.variant === 'a' && payload.calls === 'percell') {
        // One cell per call: a real forward per cell (LifeEngine::stepCellA,
        // a chunk of one against the same resident prefix stepChunkA uses),
        // not a 64-cell chunk narrated one line at a time.
        stopRequested = false;
        const n = cells.length;
        p = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          if (stopRequested) { stopRequested = false; throw new Error('stopped'); }
          const cellT0 = performance.now();
          const v = await engine.stepCellA(cells, i);
          const ms = performance.now() - cellT0;
          self.postMessage({
            type: 'chunk', chunkIndex: i, chunkCount: n, firstCell: i,
            pAlive: [v], ms,
          });
          p[i] = v;
        }
      } else if (payload.variant === 'a') {
        // 64 cells per call: one forward per chunk of cells against the
        // resident prefix.
        stopRequested = false;
        const n = engine.chunkCount();
        p = new Float32Array(cells.length);
        let at = 0;
        for (let i = 0; i < n; i++) {
          if (stopRequested) { stopRequested = false; throw new Error('stopped'); }
          const chunkT0 = performance.now();
          const q = await engine.stepChunkA(cells, i);
          const ms = performance.now() - chunkT0;
          // One message per chunk carrying everything the narration strip
          // needs: the per-cell p(alive) this chunk actually computed and
          // the chunk's own forward time.
          self.postMessage({
            type: 'chunk', chunkIndex: i, chunkCount: n, firstCell: at,
            pAlive: Array.from(q), ms,
          });
          p.set(q, at);
          at += q.length;
        }
      } else {
        // engine.step() (crates/llm-life-lean/src/web.rs) is a single async call that
        // packs the grid, tokenizes, runs the forward pass and reads the logits
        // back with no phase hooks - so 'forward' here covers all three; we
        // can't report them separately without restructuring the engine.
        self.postMessage({ type: 'progress', stage: 'forward' });
        p = await engine.step(cells);
      }
      self.postMessage({ type: 'progress', stage: 'threshold' });
      const pArr = Float32Array.from(p);
      // Label-free binarization (Otsu/z-score) is reported for the p(alive)
      // overlay display only. The grid the model actually paints (and feeds
      // forward) is thresholded at 0.5, matching how the native scorer reads
      // these answer-token logits (p_alive is already a softmax over just
      // dead/alive, so >= 0.5 is exactly the token argmax) and how the
      // adapters were evaluated during training. Using the label-free
      // threshold here instead made a handful of correctly-answered cells
      // paint wrong whenever the grid's own p(alive) distribution pushed
      // Otsu's cut point off of 0.5.
      const thresholdValue = payload.threshold === 'zscore'
        ? lean.zscoreThreshold(pArr, payload.k ?? 2.0)
        : lean.otsuThreshold(pArr);
      const binarized = Array.from(p, (v) => (v >= 0.5 ? 1 : 0));
      self.postMessage({ type: 'progress', stage: 'done' });
      reply(true, {
        pAlive: Array.from(p),
        binarized,
        thresholdValue,
        chunks: payload.variant === 'a'
          ? (payload.calls === 'percell' ? cells.length : engine.chunkCount())
          : 1,
        tokens: payload.variant === 'a'
          ? (payload.calls === 'percell'
            ? cells.length * (engine.prefixTokensA() + engine.cellTokens())
            : engine.tokensPerGenerationA())
          : engine.packedTokens(),
        seconds: (performance.now() - t0) / 1000,
      });
    } else if (type === 'gridInfo') {
      // Queried when the page changes grid size in LLM mode, to report the
      // new forward's token cost. Runs through the same serialized queue as
      // 'step', so it naturally waits for any in-flight generation.
      if (!engine) { reply(true, { tokens: null }); return; }
      if (!engineGrid || engineGrid.width !== payload.width || engineGrid.height !== payload.height) {
        engine.setGrid(payload.width, payload.height);
        engineGrid = { width: payload.width, height: payload.height };
      }
      reply(true, { tokens: engine.packedTokens() });
    } else {
      throw new Error(`unknown message ${type}`);
    }
  } catch (err) {
    // Failure here doesn't break the queue: `enqueue` chains through
    // regardless, and `engineGrid`/`engine` state is left as-is, so the next
    // queued message (a retried step, a fresh load) runs normally.
    reply(false, String(err && err.stack ? err.stack : err));
  }
}
