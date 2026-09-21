// Headless verification for the narration/grid-size v3 changes:
// - LLM per cell, "one cell per call": status counter advances cell by
//   cell, narration lines are not clipped.
// - LLM per cell, "64 cells per call": the 64 lines reveal one by one,
//   not all at once.
// - Clear mid-generation actually stops the worker issuing forwards.
// - Grid buttons: 128/256 enabled (with a title) for LLM per cell, disabled
//   (with "no adapter trained at this size") for LLM whole grid.
// - Classical rule mode still exact at 64x64 over 5 generations.
//
// Run: node scripts/headless/verify-v3.mjs (serve web/ on 8010 first)
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
const SS_DIR = arg('screenshots', '/tmp');

const browser = await chromium.launch({
  executablePath: EXECUTABLE_PATH,
  args: ['--use-gl=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-webgpu', '--enable-features=WebGPU', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
const consoleErrors = [];
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', (e) => consoleErrors.push(String(e)));

await page.goto(URL_, { waitUntil: 'load' });
await page.waitForFunction(() => window.__app && window.__app.ready, null, { timeout: 60_000 });

console.log('loading model...');
const info = await page.evaluate(
  ([gguf, tokenizer]) => window.__app.loadLlm({ ggufUrl: gguf, tokenizerUrl: tokenizer }),
  [GGUF, TOKENIZER],
);
console.log('loaded', JSON.stringify(info));

// --- 1. one cell per call: sample status for 30s, check narration box fit ---
await page.evaluate(() => {
  window.__app.setMode('llm-a');
  window.__app.setGrid(16);
  window.__app.setNarrationStyle('percell');
  window.__app.randomize(1, 0.28);
});

const seen = [];
const lineCountsAtStart = [];
const poll = setInterval(async () => {
  try {
    const state = await page.evaluate(() => {
      const el = document.getElementById('statusText');
      const log = document.getElementById('narrationLog');
      return {
        text: el ? el.textContent : null,
        lines: window.__app.narrationLines().length,
        fits: log ? log.scrollWidth <= log.clientWidth : null,
      };
    });
    if (state.text && (seen.length === 0 || seen[seen.length - 1].text !== state.text)) seen.push(state);
  } catch (e) { /* page busy */ }
}, 200);

// Don't await this: the point is to click Clear mid-generation without
// waiting for it to finish (the brief says not to wait for a full
// generation). `stepPromise` should still resolve cleanly once the worker's
// abort throws and the page's catch handler treats it as a reset, not a
// failure.
const stepPromise = page.evaluate(() => window.__app.step(1)).catch((e) => ({ threw: String(e) }));
await new Promise((r) => setTimeout(r, 30_000));
clearInterval(poll);
console.log(`\n[1] one cell per call: distinct status values in 30s (${seen.length}):`);
for (const s of seen) console.log(`  ${s.text}  lines=${s.lines} fits=${s.fits}`);
const cellValues = seen.filter((s) => /cell \d+ \/ \d+/.test(s.text)).map((s) => parseInt(s.text.match(/cell (\d+)/)[1], 10));
console.log(`  parsed cell numbers: ${cellValues.join(',')}`);
const clipped = seen.filter((s) => s.fits === false);
console.log(`  lines clipped (scrollWidth > clientWidth): ${clipped.length}`);

// --- 2. Clear mid-generation must stop the worker from issuing forwards ---
const beforeClear = await page.evaluate(() => ({
  lines: window.__app.narrationLines().length,
  status: document.getElementById('statusText').textContent,
}));
await page.evaluate(() => window.__app.clear());
const afterClear1 = beforeClear;
await new Promise((r) => setTimeout(r, 10_000));
const afterClear2 = await page.evaluate(() => ({
  lines: window.__app.narrationLines().length,
  status: document.getElementById('statusText').textContent,
}));
console.log(`\n[2] Clear mid-generation:`);
console.log(`  lines/status right before Clear: ${afterClear1.lines} lines, "${afterClear1.status}"`);
console.log(`  lines/status 10s after Clear: ${afterClear2.lines} lines, "${afterClear2.status}"`);
const clearOk = afterClear2.lines === afterClear1.lines;
console.log(`  no further forwards arrived: ${clearOk}`);
const stepOutcome = await stepPromise;
console.log(`  step(1) promise settled: ${stepOutcome && stepOutcome.threw ? 'threw: ' + stepOutcome.threw : 'resolved cleanly'}`);

// --- 3. 64 cells per call: sample line count every 100ms, first 5 after first chunk ---
await page.evaluate(() => {
  window.__app.setGrid(16);
  window.__app.randomize(4, 0.28);
  window.__app.setNarrationStyle('batched');
});
const batchedSamples = [];
const bpoll = setInterval(async () => {
  try {
    const n = await page.evaluate(() => window.__app.narrationLines().length);
    batchedSamples.push(n);
  } catch (e) { /* busy */ }
}, 100);
const batchedResult = await page.evaluate(async () => {
  await window.__app.step(1);
  return { lines: window.__app.narrationLines().length };
});
clearInterval(bpoll);
console.log(`\n[3] 64 cells per call: total lines after generation = ${batchedResult.lines}`);
const firstNonZero = batchedSamples.findIndex((n) => n > 0);
console.log(`  first 5 samples after first chunk arrived (index ${firstNonZero}): ${batchedSamples.slice(firstNonZero, firstNonZero + 5).join(', ')}`);
console.log(`  full sample count: ${batchedSamples.length}, max observed mid-run: ${Math.max(...batchedSamples.slice(0, -1), 0)}`);

// --- 4. Grid buttons: LLM per cell vs LLM whole grid ---
const gridButtonsA = await page.evaluate(() => {
  window.__app.setMode('llm-a');
  return Array.from(document.querySelectorAll('.gridsize-btn')).map((b) => ({
    size: b.dataset.size, disabled: b.disabled, title: b.title,
  }));
});
console.log('\n[4] grid buttons, LLM per cell selected:');
for (const b of gridButtonsA) console.log(`  ${b.size}: disabled=${b.disabled} title="${b.title}"`);

const gridButtonsB = await page.evaluate(() => {
  window.__app.setMode('llm');
  return Array.from(document.querySelectorAll('.gridsize-btn')).map((b) => ({
    size: b.dataset.size, disabled: b.disabled, title: b.title,
  }));
});
console.log('[4] grid buttons, LLM whole grid selected:');
for (const b of gridButtonsB) console.log(`  ${b.size}: disabled=${b.disabled} title="${b.title}"`);

// --- 5. classical rule mode, 5 generations at 64x64, still exact ---
const ruleCheck = await page.evaluate(async () => {
  window.__app.setMode('classical');
  window.__app.setGrid(64);
  window.__app.randomize(5, 0.3);
  await window.__app.step(5);
  const a = window.__app.grid(), b = window.__app.truthGrid();
  return { exact: a.every((v, i) => v === b[i]), gen: window.__app.generation() };
});
console.log(`\n[5] classical rule, 5 gens @ 64x64: exact=${ruleCheck.exact} gen=${ruleCheck.gen}`);

// --- 6. screenshots at 1400 and 390, LLM per cell mid-run ---
await page.evaluate(() => {
  window.__app.setMode('llm-a');
  window.__app.setGrid(16);
  window.__app.setNarrationStyle('percell');
  window.__app.randomize(6, 0.28);
  window.__app.step(1); // fire and forget, leave it mid-run for the screenshot
});
await new Promise((r) => setTimeout(r, 1500));
for (const w of [1400, 390]) {
  await page.setViewportSize({ width: w, height: 900 });
  await page.screenshot({ path: `${SS_DIR}/life-v3-${w}.png`, fullPage: true });
  console.log(`\n[6] screenshot saved: ${SS_DIR}/life-v3-${w}.png`);
}

console.log(`\nconsole/page errors: ${consoleErrors.length}`);
for (const e of consoleErrors) console.log('  ' + e);

await browser.close();
console.log('\nDONE');
