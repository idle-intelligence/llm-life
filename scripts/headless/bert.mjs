// Headless check of the BERT of Life rung: checkpoint load, one 16x16
// generation, narration strip stays a fixed height, IoU 1.000 vs true Life,
// PERFORMANCE finite, ladder row filled.
//
// Run: node scripts/headless/bert.mjs [--url http://127.0.0.1:8010/]
function fail(msg) {
  console.error(msg);
  process.exit(1);
}

const PLAYWRIGHT_MODULE =
  process.env.PLAYWRIGHT_MODULE ?? fail('set PLAYWRIGHT_MODULE to a playwright index.mjs');
const { chromium } = await import(PLAYWRIGHT_MODULE);

const EXECUTABLE_PATH =
  process.env.CHROMIUM_PATH ?? fail('set CHROMIUM_PATH to a Chromium executable');

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  return i === -1 ? dflt : process.argv[i + 1];
}
const URL_ = arg('url', 'http://127.0.0.1:8010/');
const CHECKPOINT = arg('checkpoint', 'http://127.0.0.1:8010/bert-d16-L1.bin');
const SCREENSHOT = arg('screenshot', 'scripts/headless/out/llm-life-bert.png');
const TIMEOUT = parseInt(arg('timeout', String(5 * 60 * 1000)), 10);

const browser = await chromium.launch({
  executablePath: EXECUTABLE_PATH,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
page.on('console', (m) => console.log(`[page:${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => console.log(`[pageerror] ${e}`));

await page.goto(URL_, { waitUntil: 'load' });
await page.waitForFunction(() => window.__app && window.__app.ready, null, { timeout: 60_000 });

console.log('loading BERT of Life checkpoint…');
const t0 = Date.now();
const info = await page.evaluate(
  (checkpointUrl) => window.__app.loadBert({ checkpointUrl, dModel: 16, nLayers: 1, nHeads: 1 }),
  CHECKPOINT,
);
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s — d_model=${info.dModel} layers=${info.numLayers}`);
if (info.dModel !== 16 || info.numLayers !== 1) throw new Error('unexpected BERT config loaded');
if (!(await page.evaluate(() => window.__app.bertReady()))) throw new Error('bertReady() false after load');

await page.evaluate(() => {
  window.__app.setMode('bert');
  window.__app.randomize(1, 0.28);
});

const gen = await page.evaluate(async () => {
  await window.__app.step(1);
  return {
    grid: window.__app.grid(),
    truth: window.__app.truthGrid(),
    iou: window.__app.iou(),
    accuracy: window.__app.accuracy(),
    seconds: window.__app.secondsPerGeneration(),
    narrationHeight: document.getElementById('narrationLog').getBoundingClientRect().height,
    narrationLines: window.__app.narrationLines(),
    narrationCounter: document.getElementById('narrationCounter').textContent,
  };
}, { timeout: TIMEOUT });

console.log(`generation: IoU=${gen.iou.toFixed(4)} acc=${(gen.accuracy * 100).toFixed(2)}% ${gen.seconds.toFixed(3)}s`);
console.log('narration lines (last 6):');
for (const l of gen.narrationLines) console.log('  ' + l);
console.log('narration counter:', gen.narrationCounter);
if (gen.narrationLines.length > 6) throw new Error('narration strip exceeded 6 lines');
if (!/9 tokens · encoder 1 layers/.test(gen.narrationLines.join('\n'))) {
  throw new Error('narration does not name the encoder rung');
}
if (gen.iou !== 1) throw new Error(`expected IoU 1.000 (BERT reproduces Life exactly), got ${gen.iou}`);

const ladder = await page.evaluate(() => window.__app.runLadder());
const bertRow = ladder.find((r) => r.name === 'BERT of Life');
console.log('ladder BERT row:', bertRow);
if (bertRow.s16 == null || !isFinite(bertRow.s16)) throw new Error('ladder BERT row s16 not finite');

await page.screenshot({ path: SCREENSHOT, fullPage: true });
console.log('screenshot saved to', SCREENSHOT);

await browser.close();
console.log('OK');
