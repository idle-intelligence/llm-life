#!/usr/bin/env bash
# Build both wasm engines and publish the committed HEAD's web/ demo to an
# orphan `gh-pages` branch, following the layout used by ../t0-web,
# ../tts-web and ../stt-web (repo root = the served tree, no crates/ or
# models/).
#
# index.html and compare/index.html load `crates/life` at `./pkg/life.js`
# (classical engine, no GGUF); the BERT/vector-space modes load
# `crates/llm-life` (Burn) through web/worker.js at `./pkg-llm/llm_life.js`,
# the LLM modes `crates/llm-life-lean` (lean) at `./pkg-lean/llm_life_lean.js`.
# LEAN_PATH=<llm-web checkout>/crates/lean builds lean from a local checkout
# instead of its git source (needed while that branch is unpublished). Model
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

# The workspace lists crates/llm-life-lean as a member, so `cargo metadata`
# (which wasm-pack runs even to build crates/life or crates/llm-life, which
# don't depend on lean at all) tries to resolve the whole workspace,
# including llm-life-lean's `lean` git dependency -- and fails outright while
# that branch is unpublished ("no matching package named `lean`"). Apply the
# same temporary LEAN_PATH patch tools/build-lean.sh uses for its own build,
# here too, before either plain wasm-pack call, and remove it once they're
# done so build-lean.sh (which writes and removes its own copy) doesn't find
# a stale one and refuse to run.
if [ -n "${LEAN_PATH:-}" ]; then
    if [ -e .cargo/config.toml ]; then
        echo "error: .cargo/config.toml exists; refusing to overwrite it" >&2
        exit 1
    fi
    mkdir -p .cargo
    printf '[patch."https://github.com/idle-intelligence/llm-web"]\nlean = { path = "%s" }\n' "$LEAN_PATH" > .cargo/config.toml
fi

echo "==> Building crates/life (classical engine)"
wasm-pack build crates/life --target web --out-dir ../../web/pkg --features web
echo "==> Building crates/llm-life (BERT/vector-space engine, Burn)"
wasm-pack build crates/llm-life --target web --out-dir ../../web/pkg-llm --no-default-features --features web

if [ -n "${LEAN_PATH:-}" ]; then
    rm -f "$REPO_ROOT/.cargo/config.toml"
    rmdir "$REPO_ROOT/.cargo" 2>/dev/null || true
fi

echo "==> Building crates/llm-life-lean (LLM engine, lean: pkg-lean and the threaded pkg-lean-mt)"
# Honours LEAN_PATH the same way (temporary .cargo/config.toml patch).
tools/build-lean.sh

LIFE_SRC="$REPO_ROOT/web/pkg"
LLM_SRC="$REPO_ROOT/web/pkg-llm"
LEAN_SRC="$REPO_ROOT/web/pkg-lean"
LEAN_MT_SRC="$REPO_ROOT/web/pkg-lean-mt"
if [ ! -f "$LIFE_SRC/life.js" ] || [ ! -f "$LIFE_SRC/life_bg.wasm" ]; then
    echo "error: expected build output not found in $LIFE_SRC" >&2
    exit 1
fi
if [ ! -f "$LLM_SRC/llm_life.js" ] || [ ! -f "$LLM_SRC/llm_life_bg.wasm" ]; then
    echo "error: expected build output not found in $LLM_SRC" >&2
    exit 1
fi
if [ ! -f "$LEAN_MT_SRC/llm_life_lean.js" ] || [ ! -f "$LEAN_MT_SRC/llm_life_lean_bg.wasm" ]; then
    echo "error: expected build output not found in $LEAN_MT_SRC" >&2
    exit 1
fi
if [ ! -f "$LEAN_SRC/llm_life_lean.js" ] || [ ! -f "$LEAN_SRC/llm_life_lean_bg.wasm" ]; then
    echo "error: expected build output not found in $LEAN_SRC" >&2
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

echo "==> Placing the built wasm at web/pkg, web/pkg-llm, web/pkg-lean and web/pkg-lean-mt"
rm -rf "$EXPORT_DIR/web/pkg" "$EXPORT_DIR/web/pkg-llm" "$EXPORT_DIR/web/pkg-lean" "$EXPORT_DIR/web/pkg-lean-mt"
mkdir -p "$EXPORT_DIR/web/pkg" "$EXPORT_DIR/web/pkg-llm" "$EXPORT_DIR/web/pkg-lean"
# The threaded module is only picked on a cross-origin-isolated host;
# GitHub Pages is not one, so there the CPU backend runs single-threaded.
cp -R "$LEAN_MT_SRC" "$EXPORT_DIR/web/pkg-lean-mt"
cp "$LIFE_SRC"/life.js "$LIFE_SRC"/life_bg.wasm "$EXPORT_DIR/web/pkg/"
[ -f "$LIFE_SRC/package.json" ] && cp "$LIFE_SRC/package.json" "$EXPORT_DIR/web/pkg/"
cp "$LLM_SRC"/llm_life.js "$LLM_SRC"/llm_life_bg.wasm "$EXPORT_DIR/web/pkg-llm/"
[ -f "$LLM_SRC/package.json" ] && cp "$LLM_SRC/package.json" "$EXPORT_DIR/web/pkg-llm/"
cp "$LEAN_SRC"/llm_life_lean.js "$LEAN_SRC"/llm_life_lean_bg.wasm "$EXPORT_DIR/web/pkg-lean/"
[ -f "$LEAN_SRC/package.json" ] && cp "$LEAN_SRC/package.json" "$EXPORT_DIR/web/pkg-lean/"

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
