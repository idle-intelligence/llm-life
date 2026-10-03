// Drives web/cellspeed/ in Playwright's Chromium with the real GPU (Metal
// through ANGLE on a Mac) and prints the page's result JSON. `--gpu 0`
// launches Chromium without the WebGPU flags (no adapter: the page falls
// back to the CPU backends); `--browser firefox` uses Playwright's Firefox.
//
//   PLAYWRIGHT_MODULE=<node_modules>/playwright/index.mjs \
//     node scripts/headless/cellspeed.mjs --url 'http://127.0.0.1:8010/cellspeed/?local=1&cells=64&rounds=3'
const args = Object.fromEntries(process.argv.slice(2).reduce((a, v, i, all) => (v.startsWith('--') ? [...a, [v.slice(2), all[i + 1]]] : a), []));
const url = args.url || 'http://127.0.0.1:8010/cellspeed/?local=1';
const pw = await import(process.env.PLAYWRIGHT_MODULE);
const gpuArgs = ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal', '--ignore-gpu-blocklist'];
const browser = args.browser === 'firefox'
  ? await pw.firefox.launch({ headless: true })
  : await pw.chromium.launch({ headless: true, args: args.gpu === '0' ? [] : gpuArgs });
const page = await browser.newPage();
const errs = [];
page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
page.on('pageerror', (e) => errs.push(String(e)));
await page.goto(url);
await page.waitForFunction(() => /^(done|error)/.test(document.getElementById('statusText').textContent), null, { timeout: Number(args.timeout || 600000) });
console.log('status:', await page.textContent('#statusText'));
console.log(await page.textContent('#result-json'));
if (args.screenshot) await page.screenshot({ path: args.screenshot, fullPage: true });
console.log('console errors:', errs.length ? errs.slice(0, 5) : 'none');
await browser.close();
