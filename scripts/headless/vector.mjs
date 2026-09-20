// Headless check of the vector-space rungs (CONCEPT.md §12/§14): (i) the 9
// numbers -> centre MLP, one forward per cell, and (ii) the whole-grid
// stencil model, one forward pass — both at 16x16, IoU vs true Life,
// narration strip stays bounded, ladder rows filled.
//
// Run: node scripts/headless/vector.mjs [--url http://127.0.0.1:8010/]
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
const MLP_CHECKPOINT = arg('mlp-checkpoint', 'http://127.0.0.1:8010/mlp2-32-lrfix.bin');
const STENCIL_CHECKPOINT = arg('stencil-checkpoint', 'http://127.0.0.1:8010/stencil-d16-L1.bin');
const SCREENSHOT = arg('screenshot', '/tmp');
const TIMEOUT = parseInt(arg('timeout', String(5 * 60 * 1000)), 10);

const browser = await chromium.launch({
  executablePath: EXECUTABLE_PATH,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
const consoleErrors = [];
page.on('console', (m) => {
  console.log(`[page:${m.type()}] ${m.text()}`);
  if (m.type() === 'error') consoleErrors.push(m.text());
});
page.on('pageerror', (e) => {
  console.log(`[pageerror] ${e}`);
  consoleErrors.push(String(e));
});

await page.goto(URL_, { waitUntil: 'load' });
await page.waitForFunction(() => window.__app && window.__app.ready, null, { timeout: 60_000 });

// --- (i) 9 numbers -> centre (MLP) ---
console.log('loading vector MLP checkpoint…');
let t0 = Date.now();
const mlpInfo = await page.evaluate(
  (checkpointUrl) => window.__app.loadVecMlp({ checkpointUrl, hidden: 32 }),
  MLP_CHECKPOINT,
);
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s — hidden=${mlpInfo.hidden}`);
if (mlpInfo.hidden !== 32) throw new Error('unexpected vector MLP config loaded');
if (!(await page.evaluate(() => window.__app.vecMlpReady()))) throw new Error('vecMlpReady() false after load');

await page.evaluate(() => {
  window.__app.setMode('vec-mlp');
  window.__app.randomize(1, 0.28);
});

const genMlp = await page.evaluate(async () => {
  await window.__app.step(1);
  return {
    iou: window.__app.iou(),
    seconds: window.__app.secondsPerGeneration(),
    narrationHeight: document.getElementById('narrationLog').getBoundingClientRect().height,
    narrationLines: window.__app.narrationLines(),
    narrationCounter: document.getElementById('narrationCounter').textContent,
  };
}, { timeout: TIMEOUT });

console.log(`(i) MLP generation: IoU=${genMlp.iou.toFixed(4)} ${genMlp.seconds.toFixed(3)}s`);
console.log('narration lines (last 6):');
for (const l of genMlp.narrationLines) console.log('  ' + l);
if (genMlp.narrationLines.length > 6) throw new Error('narration strip exceeded 6 lines');
if (!/9 numbers · MLP 32.32/.test(genMlp.narrationLines.join('\n'))) {
  throw new Error('narration does not name the 9-numbers MLP rung');
}
if (genMlp.iou !== 1) throw new Error(`expected IoU 1.000 for the 9-numbers MLP (exact rung now), got ${genMlp.iou}`);
console.log(`(i) IoU vs true Life: ${genMlp.iou.toFixed(4)} (exact)`);

// --- (ii) grid -> grid stencil model ---
console.log('loading stencil checkpoint…');
t0 = Date.now();
const stencilInfo = await page.evaluate(
  (checkpointUrl) => window.__app.loadVecStencil({ checkpointUrl, dModel: 16, nLayers: 1, nHeads: 1 }),
  STENCIL_CHECKPOINT,
);
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s — d_model=${stencilInfo.dModel} layers=${stencilInfo.numLayers}`);
if (stencilInfo.dModel !== 16 || stencilInfo.numLayers !== 1) throw new Error('unexpected stencil config loaded');
if (!(await page.evaluate(() => window.__app.vecStencilReady()))) throw new Error('vecStencilReady() false after load');

await page.evaluate(() => {
  window.__app.setMode('vec-stencil');
  window.__app.randomize(1, 0.28);
});

const genStencil = await page.evaluate(async () => {
  await window.__app.step(1);
  return {
    iou: window.__app.iou(),
    seconds: window.__app.secondsPerGeneration(),
    narrationHeight: document.getElementById('narrationLog').getBoundingClientRect().height,
    narrationLines: window.__app.narrationLines(),
  };
}, { timeout: TIMEOUT });

console.log(`(ii) stencil generation: IoU=${genStencil.iou.toFixed(4)} ${genStencil.seconds.toFixed(3)}s`);
console.log('narration lines:');
for (const l of genStencil.narrationLines) console.log('  ' + l);
if (!/whole grid · 256 cells · stencil attention · one pass/.test(genStencil.narrationLines.join('\n'))) {
  throw new Error('narration does not name the stencil rung');
}
if (genStencil.iou !== 1) throw new Error(`expected IoU 1.000 for the stencil model at 16x16, got ${genStencil.iou}`);
console.log(`(ii) IoU vs true Life: ${genStencil.iou.toFixed(4)} (exact)`);

const ladder = await page.evaluate(() => window.__app.runLadder());
const mlpRow = ladder.find((r) => r.name === '9 numbers -> centre');
const stencilRow = ladder.find((r) => r.name === 'stencil (grid -> grid)');
console.log('ladder 9-numbers row:', mlpRow);
console.log('ladder stencil row:', stencilRow);
if (mlpRow.s16 == null || !isFinite(mlpRow.s16)) throw new Error('ladder 9-numbers row s16 not finite');
if (stencilRow.s16 == null || !isFinite(stencilRow.s16)) throw new Error('ladder stencil row s16 not finite');

await page.screenshot({ path: SCREENSHOT, fullPage: true });
console.log('screenshot saved to', SCREENSHOT);

if (consoleErrors.length > 0) {
  throw new Error(`console errors during run:\n${consoleErrors.join('\n')}`);
}

await browser.close();
console.log('OK');
