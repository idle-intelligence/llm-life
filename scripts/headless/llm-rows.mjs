// The five LLM rows of the compare page, driven through web/worker.js with
// the exact message sequence runEveryRung() sends (load base, per cell one
// cell per call, adapter, per cell one call per cell and batched, reload
// base, whole grid, whole-grid adapter), on a fixed 16x16 board. Prints
// cells correct per row, the binarized grids and every console error of the
// compare page itself (opened first, then used as the worker's origin).
//
// Needs the model files under web/ as in `?local=1` (web/models/...,
// web/lora-*.bin). Timings under SwiftShader are not real.
//
//   PLAYWRIGHT_MODULE=... CHROMIUM_PATH=... node scripts/headless/llm-rows.mjs \
//     --url http://127.0.0.1:8010/ --build <the pages' BUILD tag> [--out result.json] \
//     [--rows llm-grid-base,llm-grid-trained,...] [--percell-cells N]
//
// --rows runs a subset, still with the loads/adapters the page does before
// each. --percell-cells N stops the two one-cell-per-call rows after their
// first N cells, and the batched row after the 64-cell chunk that reaches N
// (the worker's own 'stop' message, values read from its 'chunk'
// messages): 256 forwards per row take about an hour each under
// SwiftShader.

import fs from 'node:fs';

const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, all) => (v.startsWith('--') ? [...a, [v.slice(2), all[i + 1]]] : a), []));
const url = args.url || 'http://127.0.0.1:8010/';
const build = args.build;
if (!build) throw new Error('--build <tag> is required (the BUILD constant of web/compare/index.html)');
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE);

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  args: ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--use-gl=swiftshader'],
});
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); else if (m.text().startsWith('[')) console.log(m.text()); });
page.on('pageerror', (e) => errors.push(String(e)));

await page.exposeFunction('progress', (msg) => console.log(`[${new Date().toISOString().slice(11, 19)}] ${msg}`));
const only = args.rows ? args.rows.split(',') : null;
const percellCells = args['percell-cells'] ? Number(args['percell-cells']) : null;

await page.goto(url + 'compare/?local=1', { waitUntil: 'load', timeout: 60000 });
await page.waitForTimeout(2000);

const result = await page.evaluate(async ({ build, only, percellCells }) => {
  const want = (key) => !only || only.includes(key);
  const W = 16, H = 16;
  // Fixed board: an LCG at density ~0.35, the same on every run.
  let s = 12345;
  const board = Array.from({ length: W * H }, () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648 < 0.35 ? 1 : 0; });
  const next = board.map((c, i) => {
    const x = i % W, y = (i / W) | 0;
    let n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (dx || dy) n += board[((y + dy + H) % H) * W + (x + dx + W) % W];
    }
    return (n === 3 || (c && n === 2)) ? 1 : 0;
  });

  const worker = new Worker('../worker.js?v=' + build, { type: 'module' });
  let nextId = 1;
  const pending = new Map();
  let collecting = null;
  worker.onmessage = (e) => {
    const { id, ok, result, type } = e.data;
    if (type === 'chunk' && collecting && collecting.p.length < collecting.n) {
      collecting.p.push(...e.data.pAlive);
      if (collecting.p.length >= collecting.n) worker.postMessage({ type: 'stop' });
    }
    if (type === 'chunk' && (e.data.chunkIndex + 1) % 32 === 0) window.progress(`  ${e.data.chunkIndex + 1}/${e.data.chunkCount}, ${e.data.ms.toFixed(0)} ms for the last call`);
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    ok ? p.resolve(result) : p.reject(new Error(result));
  };
  const ask = (type, payload) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, payload });
  });
  const base = {
    ggufUrl: './models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf',
    tokenizerUrl: './models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json',
    width: W, height: H, rulestring: 'B3/S23',
  };
  const step = (variant, calls) => ask('step', { cells: board, variant, width: W, height: H, threshold: 'otsu', k: 2.0, ...(calls ? { calls } : {}) });
  const rows = {};
  const record = (key, r) => {
    window.progress(`${key}: done`);
    rows[key] = { correct: r.binarized.filter((v, i) => v === next[i]).length, total: W * H, seconds: r.seconds, binarized: r.binarized.join('') };
  };
  // Per-cell rows, stopped after `percellCells` cells when set (the batched
  // row stops at the end of the chunk that reaches it).
  const stepPercell = async (key, calls) => {
    if (!percellCells) return record(key, await step('a', calls));
    collecting = { n: percellCells, p: [] };
    try {
      await step('a', calls);
    } catch (e) {
      if (!String(e.message).includes('stopped')) throw e;
    }
    const p = collecting.p.slice(0, percellCells);
    collecting = null;
    window.progress(`${key}: first ${p.length} cells done`);
    const binarized = p.map((v) => (v >= 0.5 ? 1 : 0));
    rows[key] = { correct: binarized.filter((v, i) => v === next[i]).length, total: p.length, partial: true, binarized: binarized.join(''), pAlive: p };
  };
  const perCell = ['llm-percell-base', 'llm-percell-trained', 'llm-percell-batched'].some(want);
  if (perCell) {
    await ask('load', base);
    window.progress('loaded, base model');
    if (want('llm-percell-base')) await stepPercell('llm-percell-base', 'percell');
    await ask('loadAdapter', { adapterUrl: './lora-a-norules-300.bin' });
    if (want('llm-percell-trained')) await stepPercell('llm-percell-trained', 'percell');
    if (want('llm-percell-batched')) await stepPercell('llm-percell-batched');
  }
  if (want('llm-grid-base') || want('llm-grid-trained')) {
    await ask('load', base);
    window.progress('loaded, base model');
    if (want('llm-grid-base')) record('llm-grid-base', await step('b'));
    await ask('loadAdapter', { adapterUrl: './lora-b-16-s3-300.bin' });
    if (want('llm-grid-trained')) record('llm-grid-trained', await step('b'));
  }
  worker.terminate();
  return { board: board.join(''), next: next.join(''), rows };
}, { build, only, percellCells });

result.consoleErrors = errors;
const text = JSON.stringify(result, null, 2);
if (args.out) fs.writeFileSync(args.out, text);
for (const [k, r] of Object.entries(result.rows)) {
  console.log(`${k}: ${r.correct}/${r.total} correct${r.partial ? ' (first cells only)' : ` (${r.seconds.toFixed(1)} s, SwiftShader)`}`);
}
console.log('console errors:', JSON.stringify(errors));
await browser.close();
