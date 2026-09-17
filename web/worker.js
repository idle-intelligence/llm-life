// All inference runs here: the main thread does UI only.
//
// The worker fetches the GGUF in chunks and pushes them into the engine
// (`appendModelShard`) rather than handing it one giant ArrayBuffer — the
// engine reads through a sharded cursor for exactly this reason.
import init, { LifeEngine, initWgpuDevice } from './pkg-llm/llm_life.js';

let engine = null;

const CHUNK = 64 * 1024 * 1024;

async function fetchChunks(url, onProgress) {
  const head = await fetch(url, { method: 'HEAD' });
  if (!head.ok) throw new Error(`HEAD ${url}: ${head.status}`);
  const total = Number(head.headers.get('content-length'));
  const chunks = [];
  for (let start = 0; start < total; start += CHUNK) {
    const end = Math.min(start + CHUNK, total) - 1;
    const r = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
    if (!r.ok && r.status !== 206) throw new Error(`GET range ${url}: ${r.status}`);
    chunks.push(new Uint8Array(await r.arrayBuffer()));
    onProgress((end + 1) / total);
  }
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
      reply(true, { pAlive: Array.from(p), seconds: (performance.now() - t0) / 1000 });
    } else {
      throw new Error(`unknown message ${type}`);
    }
  } catch (err) {
    reply(false, String(err && err.stack ? err.stack : err));
  }
};
