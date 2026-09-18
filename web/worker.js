// All inference runs here: the main thread does UI only.
//
// The worker fetches the GGUF in chunks and pushes them into the engine
// (`appendModelShard`) rather than handing it one giant ArrayBuffer — the
// engine reads through a sharded cursor for exactly this reason. The dev
// server (web/serve.py, stdlib http.server) doesn't support Range requests,
// so this streams the single GET response and slices it into chunks itself.
import init, { LifeEngine, initWgpuDevice, otsuThreshold, zscoreThreshold } from './pkg-llm/llm_life.js';

let engine = null;

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

self.onmessage = async (e) => {
  const { id, type, payload } = e.data;
  const reply = (ok, result) => self.postMessage({ id, ok, result });
  try {
    if (type === 'load') {
      await init();
      await initWgpuDevice();
      engine = new LifeEngine(payload.width, payload.height);
      const chunks = await fetchChunks(payload.ggufUrl, (f) =>
        self.postMessage({ type: 'progress', stage: 'download', fraction: f }));
      for (const c of chunks) engine.appendModelShard(c);
      self.postMessage({ type: 'progress', stage: 'gpu', fraction: 0 });
      const tokenizerJson = await (await fetch(payload.tokenizerUrl)).text();
      await engine.load(tokenizerJson, payload.rulestring);
      reply(true, { packedTokens: engine.packedTokens() });
    } else if (type === 'step') {
      const t0 = performance.now();
      const p = await engine.step(new Uint8Array(payload.cells));
      const pArr = Float32Array.from(p);
      // Label-free binarization — same code (crate::score) as the native
      // `rescore` tool and docs/pictures/README.md, not a JS reimplementation.
      const thresholdValue = payload.threshold === 'zscore'
        ? zscoreThreshold(pArr, payload.k ?? 2.0)
        : otsuThreshold(pArr);
      const binarized = Array.from(p, (v) => (v > thresholdValue ? 1 : 0));
      reply(true, {
        pAlive: Array.from(p),
        binarized,
        thresholdValue,
        seconds: (performance.now() - t0) / 1000,
      });
    } else {
      throw new Error(`unknown message ${type}`);
    }
  } catch (err) {
    reply(false, String(err && err.stack ? err.stack : err));
  }
};
