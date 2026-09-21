// Headless check of the LLM-mode play loop (Playwright's BUNDLED Chromium):
// completion-driven play must run several sequential generations without
// overlapping worker calls, and a grid-size change requested mid-play must
// apply after the in-flight generation, not crash it.
//
// Regression target: the classical fixed-rate timer used to keep firing
// while an LLM generation was still in flight, so `worker.js` called
// `LifeEngine.setGrid`/`step` on top of a step already running and
// wasm-bindgen threw "recursive use of an object detected".
//
// Run: node scripts/headless/llm-play.mjs [--url http://127.0.0.1:8012/]
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
const URL_ = arg('url', 'http://127.0.0.1:8012/');
const GGUF = arg('gguf', 'http://127.0.0.1:8012/models/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf');
const TOKENIZER = arg('tokenizer', 'http://127.0.0.1:8012/models/hf/Qwen2.5-0.5B-Instruct/tokenizer.json');
const N_GENS = parseInt(arg('generations', '3'), 10);
const TIMEOUT = parseInt(arg('timeout', String(15 * 60 * 1000)), 10);

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

await page.evaluate(() => {
  window.__app.setMode('llm');
  window.__app.setGrid(32);
  window.__app.glider();
});

// Press play, watch for `N_GENS` sequential generations via
// `window.__app.generation()`, then pause. Each generation's wall-clock
// timestamp is recorded on the page side so we can check they're strictly
// increasing (i.e. run one after another, not overlapping).
const playResult = await page.evaluate(async (n) => {
  const timestamps = [];
  let lastGen = window.__app.generation();
  window.__app.play(true);
  const t0 = performance.now();
  while (timestamps.length < n) {
    if (performance.now() - t0 > 120_000) throw new Error('timed out waiting for generations');
    await new Promise((r) => setTimeout(r, 50));
    const g = window.__app.generation();
    if (g > lastGen) {
      timestamps.push({ gen: g, t: performance.now(), sec: window.__app.secondsPerGeneration() });
      lastGen = g;
    }
  }
  window.__app.play(false);
  return { timestamps, playingAfterPause: window.__app.isPlaying() };
}, N_GENS);

check('play produced 3 sequential generations', playResult.timestamps.length === N_GENS,
  JSON.stringify(playResult.timestamps));
let sequential = true;
for (let i = 1; i < playResult.timestamps.length; i++) {
  if (playResult.timestamps[i].t <= playResult.timestamps[i - 1].t) sequential = false;
}
check('generation timestamps strictly increasing (no overlap)', sequential,
  JSON.stringify(playResult.timestamps.map((x) => x.t)));
check('play stopped cleanly', playResult.playingAfterPause === false);
check('no page errors during play', errors.length === 0, JSON.stringify(errors));

for (const t of playResult.timestamps) {
  console.log(`gen ${t.gen}: ${t.sec.toFixed(1)} s/generation (provisional, shared GPU)`);
}

// Resume play and switch grid size mid-flight; the switch must apply after
// the in-flight generation, not crash it.
errors.length = 0;
const switchResult = await page.evaluate(async () => {
  window.__app.play(true);
  // Give the in-flight generation a moment to actually start before the
  // resize request lands mid-generation.
  await new Promise((r) => setTimeout(r, 100));
  window.__app.setGrid(64);
  const genBefore = window.__app.generation();
  const t0 = performance.now();
  while (window.__app.generation() === genBefore) {
    if (performance.now() - t0 > 120_000) throw new Error('timed out waiting for post-switch generation');
    await new Promise((r) => setTimeout(r, 50));
  }
  window.__app.play(false);
  return { width: window.__app.width, height: window.__app.height };
});

check('grid switch during play applied (64x64)', switchResult.width === 64 && switchResult.height === 64,
  JSON.stringify(switchResult));
check('no page errors during mid-play grid switch', errors.length === 0, JSON.stringify(errors));

await browser.close();

const failed = checks.filter((c) => !c.ok);
console.log(`\n${checks.length - failed.length}/${checks.length} checks passed`);
process.exit(failed.length === 0 ? 0 : 1);
