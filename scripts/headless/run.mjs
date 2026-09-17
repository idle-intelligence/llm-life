// Headless verification of web/ in Playwright's BUNDLED Chromium only —
// never TC's browser (CLAUDE.md "Demo"). Drives the page through
// `window.__app`, not through synthetic DOM clicks.
//
// Run: node scripts/headless/run.mjs [--url http://127.0.0.1:8010/] [--mode classical]
// This repo has no node_modules of its own; the Playwright module is
// borrowed the same way llm-web's harness borrows it (see its
// scripts/headless/README.md). Override with PLAYWRIGHT_MODULE.
const PLAYWRIGHT_MODULE =
  process.env.PLAYWRIGHT_MODULE ??
  '/path/to/playwright/index.mjs';
const { chromium } = await import(PLAYWRIGHT_MODULE);

// Playwright's BUNDLED Chromium, pinned explicitly so a stale default
// channel can never fall back to TC's own Chrome.
const EXECUTABLE_PATH =
  process.env.CHROMIUM_PATH ??
  '/path/to/chromium Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing';
const LAUNCH_ARGS = ['--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-angle=metal', '--ignore-gpu-blocklist'];

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  return i === -1 ? dflt : process.argv[i + 1];
}

const URL_ = arg('url', 'http://127.0.0.1:8010/');
const MODE = arg('mode', 'classical');

const browser = await chromium.launch({ executablePath: EXECUTABLE_PATH, args: LAUNCH_ARGS });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

await page.goto(URL_, { waitUntil: 'load' });
await page.waitForFunction(() => window.__app && window.__app.ready, null, { timeout: 60_000 });

const checks = [];
function check(name, ok, detail) {
  checks.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`);
}

const info = await page.evaluate(() => ({
  w: window.__app.width, h: window.__app.height, rule: window.__app.rulestring(),
}));
check('page reports a grid and a rule', info.w > 0 && info.h > 0 && info.rule === 'B3/S23',
  `${info.w}x${info.h} ${info.rule}`);

// `step` is async in LLM mode, so the harness awaits it in both modes.
// A single toggle must flip exactly one cell.
const toggled = await page.evaluate(() => {
  window.__app.clear();
  const before = window.__app.liveCount();
  window.__app.toggle(5, 7);
  return { before, after: window.__app.liveCount(), cell: window.__app.grid()[7 * window.__app.width + 5] };
});
check('toggle flips exactly one cell', toggled.before === 0 && toggled.after === 1 && toggled.cell === 1,
  JSON.stringify(toggled));

// A glider must still be a glider after 4 generations, translated by (1,1).
const glider = await page.evaluate(async () => {
  window.__app.glider();
  const before = window.__app.grid().slice();
  await window.__app.step(4);
  const after = window.__app.grid().slice();
  const W = window.__app.width, H = window.__app.height;
  let moved = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const a = before[y * W + x];
    const b = after[((y + 1) % H) * W + ((x + 1) % W)];
    if (a !== b) moved++;
  }
  return { live: after.reduce((s, v) => s + v, 0), mismatch: moved, gen: window.__app.generation() };
});
check('glider translates by (1,1) in 4 generations', glider.live === 5 && glider.mismatch === 0,
  JSON.stringify(glider));

// The model panel and the truth panel must agree in classical mode.
const agree = await page.evaluate(async () => {
  window.__app.randomize(1234, 0.3);
  await window.__app.step(5);
  const a = window.__app.grid(), b = window.__app.truthGrid();
  return a.every((v, i) => v === b[i]);
});
check('classical mode agrees with true Life', agree === (MODE === 'classical'));

check('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));

await browser.close();
const failed = checks.filter((c) => !c.ok);
console.log(`\n${checks.length - failed.length}/${checks.length} checks passed`);
process.exit(failed.length ? 1 : 0);
