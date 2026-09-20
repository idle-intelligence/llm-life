// All inference runs here: the main thread does UI only.
//
// The worker fetches the GGUF in chunks and pushes them into the engine
// (`appendModelShard`) rather than handing it one giant ArrayBuffer — the
// engine reads through a sharded cursor for exactly this reason. The dev
// server (web/serve.py, stdlib http.server) doesn't support Range requests,
// so this streams the single GET response and slices it into chunks itself.
import init, { LifeEngine, initWgpuDevice, otsuThreshold, zscoreThreshold } from './pkg-llm/llm_life.js';

let engine = null;
// Tracks the grid size last given to `engine.setGrid`, so a run of `step`
// messages at the same size doesn't call it redundantly.
let engineGrid = null;

const CHUNK = 64 * 1024 * 1024;

async function fetchChunks(url, onProgress) {
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
  return chunks;
}

// Every `LifeEngine` method call (constructor aside) is chained through this
// single promise so two can never run concurrently — wasm-bindgen throws
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

self.onmessage = (e) => {
  const { id, type, payload } = e.data;
  const reply = (ok, result) => self.postMessage({ id, ok, result });
  enqueue(() => handle(id, type, payload, reply));
};

async function handle(id, type, payload, reply) {
  try {
    if (type === 'load') {
      await init();
      await initWgpuDevice();
      engine = new LifeEngine(payload.width, payload.height);
      engineGrid = { width: payload.width, height: payload.height };
      const chunks = await fetchChunks(payload.ggufUrl, (f) =>
        self.postMessage({ type: 'progress', stage: 'download', fraction: f }));
      for (const c of chunks) engine.appendModelShard(c);
      const tokenizerJson = await (await fetch(payload.tokenizerUrl)).text();
      await engine.load(tokenizerJson, payload.rulestring);
      // Runtime LoRA (llm_wasm::lora — not the offline GGUF merge): applies
      // q/k/v/o deltas onto the already-loaded base Q4 model, no reload.
      // Optional — if `adapterUrl` isn't set, the engine is byte-for-byte
      // the base model.
      let adapter = null;
      if (payload.adapterUrl) {
        const bytes = new Uint8Array(await (await fetch(payload.adapterUrl)).arrayBuffer());
        const name = payload.adapterUrl.split('/').pop().replace(/\.bin$/, '');
        // Variant A's resident prefix must match how this adapter was
        // trained/evaluated (`norules_prefix()` vs `rules_prefix()`, no
        // few-shot) — `LifeEngine::loadAdapter` rebuilds it from this flag.
        engine.loadAdapter(bytes, name.includes('norules'));
        adapter = { name, bytes: bytes.length };
      }
      reply(true, { packedTokens: engine.packedTokens(), cellTokens: engine.cellTokens(), adapter });
    } else if (type === 'loadAdapter') {
      // Swap the runtime LoRA adapter without a full model reload — replaces
      // whatever adapter is currently applied (LifeEngine::loadAdapter does
      // not stack).
      const bytes = new Uint8Array(await (await fetch(payload.adapterUrl)).arrayBuffer());
      const name = payload.adapterUrl.split('/').pop().replace(/\.bin$/, '');
      engine.loadAdapter(bytes, name.includes('norules'));
      reply(true, { name, bytes: bytes.length });
    } else if (type === 'step') {
      const t0 = performance.now();
      // The engine packs for whatever grid it was last told about; variant B
      // runs at 64x64 and variant A at 16x16 off the same loaded weights.
      // Only call setGrid when the size actually changed — redundant calls
      // are dropped, and since this runs inside the serialized queue it
      // never overlaps a step already in flight.
      if (!engineGrid || engineGrid.width !== payload.width || engineGrid.height !== payload.height) {
        engine.setGrid(payload.width, payload.height);
        engineGrid = { width: payload.width, height: payload.height };
      }
      const cells = new Uint8Array(payload.cells);
      let p;
      if (payload.variant === 'a') {
        // Variant A is one forward per chunk of cells against the resident
        // prefix, so unlike variant B it has a natural progress granularity.
        const n = engine.chunkCount();
        p = new Float32Array(cells.length);
        let at = 0;
        for (let i = 0; i < n; i++) {
          const chunkT0 = performance.now();
          const q = await engine.stepChunkA(cells, i);
          const ms = performance.now() - chunkT0;
          // One message per chunk carrying everything the narration strip
          // needs: the per-cell p(alive) this chunk actually computed and
          // the chunk's own forward time. The engine only exposes chunk-level
          // calls (one forward per 64-cell chunk), so "per cell" narration
          // and timing are derived from this: chunk time / chunk size.
          self.postMessage({
            type: 'chunk', chunkIndex: i, chunkCount: n, firstCell: at,
            pAlive: Array.from(q), ms,
          });
          p.set(q, at);
          at += q.length;
        }
      } else {
        // engine.step() (crates/llm-life/src/web.rs) is a single async call that
        // packs the grid, tokenizes, runs the forward pass and reads the logits
        // back with no phase hooks — so 'forward' here covers all three; we
        // can't report them separately without restructuring the engine.
        self.postMessage({ type: 'progress', stage: 'forward' });
        p = await engine.step(cells);
      }
      self.postMessage({ type: 'progress', stage: 'threshold' });
      const pArr = Float32Array.from(p);
      // Label-free binarization — same code (crate::score) as the native
      // `rescore` tool and docs/pictures/README.md, not a JS reimplementation.
      const thresholdValue = payload.threshold === 'zscore'
        ? zscoreThreshold(pArr, payload.k ?? 2.0)
        : otsuThreshold(pArr);
      const binarized = Array.from(p, (v) => (v > thresholdValue ? 1 : 0));
      self.postMessage({ type: 'progress', stage: 'done' });
      reply(true, {
        pAlive: Array.from(p),
        binarized,
        thresholdValue,
        chunks: payload.variant === 'a' ? engine.chunkCount() : 1,
        tokens: payload.variant === 'a' ? engine.tokensPerGenerationA() : engine.packedTokens(),
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
