# Headless scripts

Set `PLAYWRIGHT_MODULE` to a local `playwright/index.mjs` and `CHROMIUM_PATH`
to a Chromium executable (both required, no defaults). Start the local
server first with `python3 web/serve.py`. Screenshots land in
`scripts/headless/out/` (gitignored) unless `--screenshot` overrides it.

Example:

```
PLAYWRIGHT_MODULE=/path/to/node_modules/playwright/index.mjs \
CHROMIUM_PATH=/path/to/chrome \
node scripts/headless/llm.mjs --url http://127.0.0.1:8010/
```
