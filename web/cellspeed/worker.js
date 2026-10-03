// Cell-speed measurement: all inference runs here, the page does UI only.
// Loads Qwen2.5-0.5B (lean, pkg-lean) and the a-norules-300 adapter, then
// times one forward per cell against the resident prefix ("reuse") and the
// same cell with the prefix forwarded again in the same call ("no reuse").
// The GGUF fetch (fetchChunks, splitIntoChunks, cachedFetchText) is the one
// in ../worker.js, with the same cache name so a load there is a hit here.
const ENGINE_BUILD = '2026-10-03';
const pending = [];
self.onmessage = (e) => pending.push(e);

const lean = await import(`../pkg-lean/llm_life_lean.js?v=${ENGINE_BUILD}`);
await lean.default({ module_or_path: new URL(`../pkg-lean/llm_life_lean_bg.wasm?v=${ENGINE_BUILD}`, import.meta.url) });

const CHUNK = 64 * 1024 * 1024;
const MODEL_CACHE = 'llm-life-model-v1';

function splitIntoChunks(buf) {
  const chunks = [];
  for (let off = 0; off < buf.length; off += CHUNK) chunks.push(buf.subarray(off, off + CHUNK));
  return chunks;
}

async function fetchChunks(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(MODEL_CACHE); } catch { cache = null; }
  if (cache) {
    const cached = await cache.match(url);
    if (cached) {
      onProgress(1);
      return { chunks: splitIntoChunks(new Uint8Array(await cached.arrayBuffer())), cached: true };
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
    const n = chunks.reduce((s, c) => s + c.length, 0);
    const merged = new Uint8Array(n);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.length; }
    try {
      await cache.put(url, new Response(merged.buffer, { headers: { 'Content-Type': 'application/octet-stream' } }));
    } catch (err) {
      console.warn('[cellspeed] could not cache:', err);
    }
  }
  return { chunks, cached: false };
}

async function cachedFetchText(url) {
  let cache = null;
  try { cache = await caches.open(MODEL_CACHE); } catch { cache = null; }
  const cached = cache && await cache.match(url);
  if (cached) return cached.text();
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url}: ${r.status}`);
  const text = await r.text();
  try { if (cache) await cache.put(url, new Response(text, { headers: { 'Content-Type': 'application/json' } })); } catch { /* not cached */ }
  return text;
}

let engine = null;

async function load({ ggufUrl, tokenizerUrl, adapterUrl, width, height }) {
  const t0 = performance.now();
  engine = await lean.LifeEngine.create(width, height);
  const tDevice = performance.now();
  const { chunks, cached } = await fetchChunks(ggufUrl, (f) => self.postMessage({ type: 'progress', fraction: f }));
  const tokenizerJson = await cachedFetchText(tokenizerUrl);
  const tFetch = performance.now();
  for (const c of chunks) engine.appendModelShard(c);
  await engine.load(tokenizerJson, 'B3/S23');
  const tModel = performance.now();
  const bytes = new Uint8Array(await (await fetch(adapterUrl)).arrayBuffer());
  await engine.loadAdapter(bytes, true);
  const tAdapter = performance.now();
  return {
    deviceMs: tDevice - t0,
    fetchMs: tFetch - tDevice,
    ggufCached: cached,
    modelMs: tModel - tFetch,
    adapterMs: tAdapter - tModel,
    totalMs: tAdapter - t0,
    prefixTokens: engine.prefixTokensA(),
    adapterBytes: bytes.length,
  };
}

// Rounds alternate the two variants over the same cells (reuse block, then
// no-reuse block), after one warm-up call of each.
async function run({ cells, order, rounds }) {
  const board = new Uint8Array(cells);
  const call = async (variant, i) => {
    const t = performance.now();
    const r = variant === 'reuse' ? await engine.logitsCellA(board, i) : await engine.logitsCellAFull(board, i);
    return { ms: performance.now() - t, dead: r[0], alive: r[1], kase: r[2], tokens: r[3] };
  };
  await call('reuse', order[0]);
  await call('full', order[0]);
  for (let round = 0; round < rounds; round++) {
    for (const variant of ['reuse', 'full']) {
      for (const i of order) {
        const r = await call(variant, i);
        self.postMessage({ type: 'cell', round, variant, index: i, ...r });
      }
    }
  }
  // For scale: the first 64 cells in one forward (stepChunkA), after one
  // warm-up call.
  await engine.stepChunkA(board, 0);
  const chunkMs = [];
  for (let k = 0; k < 3; k++) {
    const t = performance.now();
    await engine.stepChunkA(board, 0);
    chunkMs.push(performance.now() - t);
  }
  return { chunkMs };
}

async function handle({ id, type, payload }) {
  try {
    const result = type === 'load' ? await load(payload) : await run(payload);
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    self.postMessage({ id, ok: false, result: String(err && err.message || err) });
  }
}

let queue = Promise.resolve();
self.onmessage = (e) => { queue = queue.then(() => handle(e.data)); };
for (const e of pending) self.onmessage(e);
pending.length = 0;
