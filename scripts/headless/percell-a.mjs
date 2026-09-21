// Headless check of LLM per cell mode with "one cell per call": the status
// counter must advance one real forward at a time (LifeEngine::stepCellA),
// not narrate a 64-cell chunk as if it were per-cell. Also runs "64 cells
// per call" once to confirm the batched path still works.
//
// Run: node scripts/headless/percell-a.mjs (serve web/ on 8010 first)
const PLAYWRIGHT_MODULE =
  process.env.PLAYWRIGHT_MODULE ??
  '/path/to/playwright/index.mjs';
const { chromium } = await import(PLAYWRIGHT_MODULE);

const EXECUTABLE_PATH =
  process.env.CHROMIUM_PATH ??
  '/path/to/chromium Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing';

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  return i === -1 ? dflt : process.argv[i + 1];
}
const URL_ = arg('url', 'http://127.0.0.1:8010/');
const GGUF = arg('gguf', 'http://127.0.0.1:8010/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf');
const TOKENIZER = arg('tokenizer', 'http://127.0.0.1:8010/models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json');
const TIMEOUT = parseInt(arg('timeout', String(20 * 60 * 1000)), 10);

const browser = await chromium.launch({
  executablePath: EXECUTABLE_PATH,
  args: ['--use-gl=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
page.on('console', (m) => console.log(`[page:${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => console.log(`[pageerror] ${e}`));

await page.goto(URL_, { waitUntil: 'load' });
await page.waitForFunction(() => window.__app && window.__app.ready, null, { timeout: 60_000 });

console.log('loading model…');
const t0 = Date.now();
const info = await page.evaluate(
  ([gguf, tokenizer]) => window.__app.loadLlm({ ggufUrl: gguf, tokenizerUrl: tokenizer }),
  [GGUF, TOKENIZER],
);
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`, JSON.stringify(info));

await page.evaluate(() => {
  window.__app.setMode('llm-a');
  window.__app.setNarrationStyle('percell');
  window.__app.randomize(1, 0.28);
});
const grid = await page.evaluate(() => window.__app.grid().length);
console.log(`grid cells: ${grid}`);

// Sample the STATUS line every 200ms while one generation runs, to prove
// the counter advances cell by cell (real forwards), not chunk by chunk.
const seen = [];
let poll = setInterval(async () => {
  try {
    const text = await page.evaluate(() => {
      const el = document.getElementById('statusText');
      return el ? el.textContent : null;
    });
    if (text && (seen.length === 0 || seen[seen.length - 1] !== text)) seen.push(text);
  } catch (e) { /* page navigating/busy, ignore */ }
}, 200);

const stepT0 = Date.now();
const result = await page.evaluate(async () => {
  await window.__app.step(1);
  return {
    seconds: window.__app.secondsPerGeneration(),
    msPerCell: window.__app.msPerCell(),
    cellsPerSecond: window.__app.cellsPerSecond(),
    iou: window.__app.iou(),
    accuracy: window.__app.accuracy(),
    narrationLines: window.__app.narrationLines(),
    grid: window.__app.grid(),
  };
}, { timeout: TIMEOUT });
clearInterval(poll);
console.log(`percell generation: ${((Date.now() - stepT0) / 1000).toFixed(1)}s wall, ` +
  `ms/cell=${result.msPerCell?.toFixed(1)} cells/s=${result.cellsPerSecond?.toFixed(2)} ` +
  `IoU=${result.iou?.toFixed(4)} acc=${(result.accuracy * 100).toFixed(2)}%`);
console.log(`distinct status values seen (${seen.length}):`);
for (const s of seen) console.log('  ' + s);
console.log('narration lines (last, up to 6):');
for (const l of result.narrationLines) console.log('  ' + l);

const cellCounterValues = seen.filter((s) => /cell \d+ \/ \d+/.test(s));
if (cellCounterValues.length < 3) {
  throw new Error(`expected the status counter to advance per cell, saw only ${cellCounterValues.length} distinct "cell N / M" values`);
}

// --- batched sanity: confirm 64-per-call path still works -----------------
await page.evaluate(() => { window.__app.randomize(2, 0.28); window.__app.setNarrationStyle('batched'); });
const batched = await page.evaluate(async () => {
  await window.__app.step(1);
  return {
    seconds: window.__app.secondsPerGeneration(),
    iou: window.__app.iou(),
    grid: window.__app.grid(),
  };
}, { timeout: TIMEOUT });
console.log(`batched generation: ${batched.seconds.toFixed(1)}s IoU=${batched.iou.toFixed(4)}`);

const errors = await page.evaluate(() => window.__pageErrors || []);
console.log('page errors:', errors);

await browser.close();
console.log('OK');
