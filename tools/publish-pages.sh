#!/usr/bin/env bash
# Build both wasm engines and publish the committed HEAD's web/ demo to an
# orphan `gh-pages` branch, following the layout used by ../t0-web,
# ../tts-web and ../stt-web (repo root = the served tree, no crates/ or
# models/).
#
# index.html and compare/index.html load `crates/life` at `./pkg/life.js`
# (classical engine, no GGUF); the LLM/BERT/vector-space modes load
# `crates/llm-life` through web/worker.js at `./pkg-llm/llm_life.js`. Model
# weights come from Hugging Face at runtime (web/models.js), never shipped
# here.
#
# Never checks out gh-pages in the main working tree; never pushes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKTREE_DIR="$(mktemp -d)/gh-pages-worktree"
EXPORT_DIR="$(mktemp -d)/web-export"

cleanup() {
    git worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 || true
    rm -rf "$EXPORT_DIR"
}
trap cleanup EXIT

echo "==> Building crates/life (classical engine)"
wasm-pack build crates/life --target web --out-dir ../../web/pkg --features web
echo "==> Building crates/llm-life (LLM/BERT/vector-space engine)"
wasm-pack build crates/llm-life --target web --out-dir ../../web/pkg-llm --no-default-features --features web

LIFE_SRC="$REPO_ROOT/web/pkg"
LLM_SRC="$REPO_ROOT/web/pkg-llm"
if [ ! -f "$LIFE_SRC/life.js" ] || [ ! -f "$LIFE_SRC/life_bg.wasm" ]; then
    echo "error: expected build output not found in $LIFE_SRC" >&2
    exit 1
fi
if [ ! -f "$LLM_SRC/llm_life.js" ] || [ ! -f "$LLM_SRC/llm_life_bg.wasm" ]; then
    echo "error: expected build output not found in $LLM_SRC" >&2
    exit 1
fi

echo "==> Exporting committed HEAD's web/ into $EXPORT_DIR"
mkdir -p "$EXPORT_DIR"
git archive HEAD web | tar -x -C "$EXPORT_DIR"

# web/models is gitignored (weights are fetched from Hugging Face and cached
# by the browser, never shipped) and so are the *.bin adapters/checkpoints;
# git archive already excludes them.
if [ -d "$EXPORT_DIR/web/models" ]; then
    echo "error: unexpected web/models in export -- refusing to publish weights" >&2
    exit 1
fi
if compgen -G "$EXPORT_DIR/web/*.bin" > /dev/null; then
    echo "error: unexpected *.bin in export -- refusing to publish weights" >&2
    exit 1
fi

echo "==> Placing the built wasm at web/pkg and web/pkg-llm"
rm -rf "$EXPORT_DIR/web/pkg" "$EXPORT_DIR/web/pkg-llm"
mkdir -p "$EXPORT_DIR/web/pkg" "$EXPORT_DIR/web/pkg-llm"
cp "$LIFE_SRC"/life.js "$LIFE_SRC"/life_bg.wasm "$EXPORT_DIR/web/pkg/"
[ -f "$LIFE_SRC/package.json" ] && cp "$LIFE_SRC/package.json" "$EXPORT_DIR/web/pkg/"
cp "$LLM_SRC"/llm_life.js "$LLM_SRC"/llm_life_bg.wasm "$EXPORT_DIR/web/pkg-llm/"
[ -f "$LLM_SRC/package.json" ] && cp "$LLM_SRC/package.json" "$EXPORT_DIR/web/pkg-llm/"

echo "==> Preparing orphan gh-pages worktree at $WORKTREE_DIR"
mkdir -p "$(dirname "$WORKTREE_DIR")"
if git show-ref --verify --quiet refs/heads/gh-pages; then
    git worktree add "$WORKTREE_DIR" gh-pages
else
    git worktree add --detach "$WORKTREE_DIR" HEAD
    git -C "$WORKTREE_DIR" checkout --orphan gh-pages
    git -C "$WORKTREE_DIR" rm -rf . >/dev/null 2>&1 || true
fi

find "$WORKTREE_DIR" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -R "$EXPORT_DIR/web" "$WORKTREE_DIR/web"

cd "$WORKTREE_DIR"
git add -A
if git diff --cached --quiet; then
    echo "==> No changes; gh-pages already up to date"
else
    git commit -q -m "Publish web/ demo to GitHub Pages"
fi
cd "$REPO_ROOT"

echo "==> gh-pages tip: $(git rev-parse gh-pages)"
echo "==> Push with: git push --force-with-lease origin gh-pages (never run automatically)"
