// Headless check of LLM-mode Clear/Random/Glider (Playwright's BUNDLED
// Chromium): every reset must produce one fresh seed state that is
// simultaneously the model grid, the true-Life grid and the engine's next
// input — even when a generation is in flight.
//
// Regression target: pressing Clear/Random during play left the true-Life
// panel reset but the model panel kept the previous (stale) generation's
// binarized output, because the in-flight `llmStepOnce()` painted its result
// over the reset after it landed. Also checks the load-progress log: one
// `download: N%` line replaced in place, not six, and one `model loaded`
// line.
//
// Run: node scripts/headless/llm-reset.mjs [--url http://127.0.0.1:8012/]
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
const URL_ = arg('url', 'http://127.0.0.1:8012/');
const GGUF = arg('gguf', 'http://127.0.0.1:8012/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf');
const TOKENIZER = arg('tokenizer', 'http://127.0.0.1:8012/models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json');

const browser = await chromium.launch({
  executablePath: EXECUTABLE_PATH,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
const errors = [];
page.on('console', (m) => { console.log(`[page:${m.type()}] ${m.text()}`); if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => { console.log(`[pageerror] ${e}`); errors.push(String(e)); });

const checks = [];
function check(name, ok, detail) {
  checks.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
}

await page.goto(URL_, { waitUntil: 'load' });
await page.waitForFunction(() => window.__app && window.__app.ready, null, { timeout: 60_000 });

console.log('loading model…');
const t0 = Date.now();
await page.evaluate(
  ([gguf, tokenizer]) => window.__app.loadLlm({ ggufUrl: gguf, tokenizerUrl: tokenizer }),
  [GGUF, TOKENIZER],
);
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

const logLines = await page.evaluate(() => document.getElementById('statusLog').textContent.split('\n'));
const downloadLines = logLines.filter((l) => /^download: \d+%$/.test(l));
const modelLoadedLines = logLines.filter((l) => /^model loaded:/.test(l));
check('exactly one download progress line after load', downloadLines.length === 1, JSON.stringify(downloadLines));
check('exactly one model-loaded line after load', modelLoadedLines.length === 1, JSON.stringify(modelLoadedLines));

await page.evaluate(() => {
  window.__app.setMode('llm');
  window.__app.setGrid(32);
});

// Random: model and true-Life panels must be the same fresh state, generation 0.
const afterRandom = await page.evaluate(() => {
  window.__app.randomize(42, 0.28);
  return { model: window.__app.grid(), truth: window.__app.truthGrid(), gen: window.__app.generation() };
});
check('random: model grid equals true grid', JSON.stringify(afterRandom.model) === JSON.stringify(afterRandom.truth));
check('random: generation is 0', afterRandom.gen === 0, String(afterRandom.gen));

// Step once: the two panels are expected to diverge (model advances via the
// LLM forward pass + threshold, truth via the real rule).
console.log('stepping once…');
const afterStep = await page.evaluate(async () => {
  await window.__app.step(1);
  return { model: window.__app.grid(), truth: window.__app.truthGrid(), gen: window.__app.generation() };
});
check('step: generation is 1', afterStep.gen === 1, String(afterStep.gen));
check('step: model and true grids differ', JSON.stringify(afterStep.model) !== JSON.stringify(afterStep.truth));

// Clear during play: the in-flight generation's stale result must be
// discarded, not painted — after it lands, all three panels are empty and
// generation is back to 0.
console.log('clear during play…');
errors.length = 0;
await page.evaluate(() => window.__app.play(true));
await new Promise((r) => setTimeout(r, 100)); // let the first generation start (generating=true)
await page.evaluate(() => window.__app.clear());
const clearResult = await page.evaluate(async () => {
  const t0 = performance.now();
  for (;;) {
    const model = window.__app.grid();
    const truthG = window.__app.truthGrid();
    const allEmpty = model.every((v) => v === 0) && truthG.every((v) => v === 0) && window.__app.generation() === 0;
    if (allEmpty) return { model, truth: truthG, gen: window.__app.generation(), playing: window.__app.isPlaying() };
    if (performance.now() - t0 > 120_000) throw new Error('timed out waiting for clear to apply');
    await new Promise((r) => setTimeout(r, 50));
  }
});
check('clear during play: model panel empty', clearResult.model.every((v) => v === 0));
check('clear during play: true panel empty', clearResult.truth.every((v) => v === 0));
check('clear during play: generation is 0', clearResult.gen === 0, String(clearResult.gen));
check('clear during play: playback stopped', clearResult.playing === false);
check('no page errors during clear-during-play', errors.length === 0, JSON.stringify(errors));

await browser.close();

const failed = checks.filter((c) => !c.ok);
console.log(`\n${checks.length - failed.length}/${checks.length} checks passed`);
process.exit(failed.length === 0 ? 0 : 1);
