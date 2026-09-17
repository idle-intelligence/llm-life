// End-to-end check of LLM mode in Playwright's BUNDLED Chromium: load the
// real GGUF into the worker, run one variant-B generation, report seconds
// per generation and agreement with true Life on the same grid.
//
// Needs two servers:
//   python3 ~/Code/idle-intelligence/llm-web/scripts/serve_models.py --dir ~/Code/idle-intelligence/models
//   python3 web/serve.py
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
const GGUF = arg('gguf', 'http://127.0.0.1:8001/gguf/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf');
const TOKENIZER = arg('tokenizer', 'http://127.0.0.1:8001/hf/Qwen2.5-0.5B-Instruct/tokenizer.json');
const GENS = parseInt(arg('generations', '1'), 10);
const TIMEOUT_LOAD = parseInt(arg('timeout-load', String(15 * 60 * 1000)), 10);
const TIMEOUT_STEP = parseInt(arg('timeout-step', String(15 * 60 * 1000)), 10);

const browser = await chromium.launch({
  executablePath: EXECUTABLE_PATH,
  args: ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--ignore-gpu-blocklist'],
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
console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s, ${info.packedTokens} tokens/forward pass`);

await page.evaluate(() => { window.__app.setMode('llm'); window.__app.randomize(1, 0.28); });

for (let g = 1; g <= GENS; g++) {
  const r = await page.evaluate(async () => {
    await window.__app.step(1);
    return {
      seconds: window.__app.secondsPerGeneration(),
      accuracy: window.__app.accuracy(),
      live: window.__app.liveCount(),
      gen: window.__app.generation(),
    };
  }, { timeout: TIMEOUT_STEP });
  console.log(`gen ${r.gen}: ${r.seconds.toFixed(1)}s  agreement=${(r.accuracy * 100).toFixed(2)}%  model_live=${r.live}`);
}

await browser.close();
