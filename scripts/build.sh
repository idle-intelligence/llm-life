#!/usr/bin/env bash
# Builds the three wasm variants and assembles the deployed site into
# _site/, same as .github/workflows/pages.yml's inline steps (moved here
# so CI and local builds share one script, llm-web lean-release shape).
#
#   web/pkg       classical Game of Life engine (crates/life)
#   web/pkg-llm   LLM/BERT/vector-space engine, Burn (crates/llm-life)
#   web/pkg-lean  lean engine, WebGPU + single-thread CPU (crates/llm-life-lean)
#   web/pkg-lean-mt  lean engine, CPU threads -- opt-in (BUILD_THREADS=1),
#                    needs a nightly toolchain with rust-src; Pages serves
#                    no COOP/COEP headers so a page there can never select
#                    it (see web/lean-backend.js's capability check), and
#                    this script does not build it unless asked.
#
# ENGINE_BUILD, if the caller sets it (CI passes the commit sha, matching
# llm-web's build.sh invocation shape), is not used for the `?v=` tag below:
# that tag is a content hash of the three built wasm files, same as this
# repo's pages.yml has always computed it, kept unchanged by this move.
#
# Requires wasm-pack. BUILD_THREADS=1 also requires a nightly toolchain
# with rust-src and wasm-bindgen-cli matching Cargo.lock's wasm-bindgen
# version exactly.
set -euo pipefail

BUILD_THREADS="${BUILD_THREADS:-}"
LEAN_NIGHTLY="${LEAN_NIGHTLY:-nightly-2026-09-20}"
MAX_MEMORY="${MAX_MEMORY:-2684354560}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Keep the runner's paths out of the compiled wasm (dependency source paths
# end up in panic messages otherwise).
export RUSTFLAGS="${RUSTFLAGS:-} --remap-path-prefix=$HOME/.cargo=/cargo --remap-path-prefix=$REPO_ROOT=/src"

echo "==> Building life (classical engine)"
wasm-pack build crates/life --target web --out-dir ../../web/pkg --features web

echo "==> Building llm-life (LLM/BERT/vector-space engine)"
wasm-pack build crates/llm-life --target web --out-dir ../../web/pkg-llm --no-default-features --features web

echo "==> Building llm-life-lean (WebGPU + single-thread CPU)"
RUSTFLAGS="$RUSTFLAGS -C target-feature=+simd128" \
  wasm-pack build crates/llm-life-lean --target web --out-dir ../../web/pkg-lean --no-default-features --features web

if [ -n "$BUILD_THREADS" ]; then
  echo "==> Building llm-life-lean (CPU threads, nightly build-std, BUILD_THREADS=1)"
  OUT_MT="web/pkg-lean-mt"
  RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
-C link-arg=--shared-memory -C link-arg=--max-memory=$MAX_MEMORY \
-C link-arg=--import-memory \
-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base \
--remap-path-prefix=$HOME/.cargo=/cargo --remap-path-prefix=$REPO_ROOT=/src" \
    cargo +"$LEAN_NIGHTLY" build -p llm-life-lean --lib \
      --target wasm32-unknown-unknown --release \
      --no-default-features --features web-mt \
      -Z build-std=panic_abort,std
  WASM_IN="target/wasm32-unknown-unknown/release/llm_life_lean.wasm"
  if [ ! -f "$WASM_IN" ]; then
    echo "error: expected build output not found at $WASM_IN" >&2
    exit 1
  fi
  rm -rf "$OUT_MT"
  mkdir -p "$OUT_MT"
  wasm-bindgen --target web --out-dir "$OUT_MT" --out-name llm_life_lean "$WASM_IN"
  HELPER=$(find "$OUT_MT/snippets" -name workerHelpers.js 2>/dev/null | head -1)
  if [ -z "$HELPER" ]; then
    echo "error: workerHelpers.js not found under $OUT_MT/snippets" >&2
    exit 1
  fi
  sed -i.bak \
    -e "s#await import('\.\./\.\./\.\.')#await import('../../../llm_life_lean.js')#" \
    "$HELPER"
  rm -f "$HELPER.bak"
fi

# --- Assemble the deployed site into _site/ ---
echo "==> Assembling _site"
rm -rf _site
mkdir -p _site
cp -R web _site/web

if [ -d _site/web/models ]; then
  echo "error: unexpected web/models in export -- refusing to publish weights" >&2
  exit 1
fi
if compgen -G "_site/web/*.bin" > /dev/null; then
  echo "error: unexpected *.bin in export -- refusing to publish weights" >&2
  exit 1
fi

# wasm-pack also writes .d.ts and .gitignore into web/pkg, web/pkg-llm,
# web/pkg-lean and (BUILD_THREADS=1) web/pkg-lean-mt; only the .js/.wasm/
# package.json files are published.
PKG_DIRS="_site/web/pkg _site/web/pkg-llm _site/web/pkg-lean"
[ -n "$BUILD_THREADS" ] && PKG_DIRS="$PKG_DIRS _site/web/pkg-lean-mt"
for d in $PKG_DIRS; do
  find "$d" -mindepth 1 -maxdepth 1 ! -name '*.js' ! -name '*.wasm' ! -name 'package.json' ! -name 'snippets' -exec rm -rf {} +
done
if [ -z "$BUILD_THREADS" ] && [ -d _site/web/pkg-lean-mt ]; then
  rm -rf _site/web/pkg-lean-mt
fi

# --- Tag and rewrite the ?v= build tag ---
echo "==> Tagging and rewriting ?v= build tag"
WASM_FILES="web/pkg/life_bg.wasm web/pkg-llm/llm_life_bg.wasm web/pkg-lean/llm_life_lean_bg.wasm"
TAG="$(cat $WASM_FILES | shasum -a 256 | cut -c1-7)"
echo "build tag: $TAG"

# './pkg/life.js?v=...' is a static import specifier (must be a string
# literal, can't take a JS variable) in both pages.
sed -i.bak "s#pkg/life.js?v=[A-Za-z0-9_.-]*#pkg/life.js?v=$TAG#" \
  _site/web/index.html _site/web/compare/index.html
# BUILD (worker.js's own URL, bumped alongside pkg/pkg-llm) and
# ENGINE_BUILD (the pkg-llm wasm's URL) are JS constants read by a
# template literal, so the tag goes into the assignment.
sed -i.bak "s/const BUILD = '[^']*'/const BUILD = '$TAG'/" \
  _site/web/index.html _site/web/compare/index.html
sed -i.bak "s/const ENGINE_BUILD = '[^']*'/const ENGINE_BUILD = '$TAG'/" \
  _site/web/worker.js
rm -f _site/web/index.html.bak _site/web/compare/index.html.bak _site/web/worker.js.bak

STATIC_COUNT="$(grep -rEo "pkg/life\.js\?v=[0-9a-fA-F]{7}" _site/web/index.html _site/web/compare/index.html | wc -l)"
if [ "$STATIC_COUNT" -ne 2 ]; then
  echo "error: expected 2 'pkg/life.js?v=$TAG' static imports, found $STATIC_COUNT" >&2
  exit 1
fi
BUILD_COUNT="$(grep -rEo "BUILD = '[0-9a-fA-F]{7}'" _site/web/index.html _site/web/compare/index.html _site/web/worker.js | wc -l)"
if [ "$BUILD_COUNT" -ne 3 ]; then
  echo "error: expected 3 BUILD/ENGINE_BUILD assignments rewritten to $TAG, found $BUILD_COUNT" >&2
  exit 1
fi

BAD="$(grep -rEo '\?v=[0-9a-fA-F.-]+' _site/web | grep -v ":?v=$TAG\$" || true)"
if [ -n "$BAD" ]; then
  echo "error: found ?v= URLs not rewritten to $TAG:" >&2
  echo "$BAD" >&2
  exit 1
fi

# --- Local-path / user-name leak check on built wasm outputs ---
# Matches on extracted printable strings, not raw bytes (llm-web's
# scripts/build.sh): the old raw `grep -c -a '~/'` matched two-byte
# coincidences inside wasm opcode/table bytes, not actual leaked paths.
echo "==> Checking for local build paths"
FAILED=0
for f in _site/web/pkg/life_bg.wasm _site/web/pkg-llm/llm_life_bg.wasm _site/web/pkg-lean/llm_life_lean_bg.wasm; do
  LEAKS="$(strings "$f" | grep -F -e "$HOME" -e "Code/" -e ".claude/" -e "/Users/" -e "/home/runner/" || true)"
  USER_HITS="$(strings "$f" | grep -Fw -e "$(id -un)" || true)"
  if [ -n "$LEAKS$USER_HITS" ]; then
    echo "error: $f contains local paths or the user name:" >&2
    printf '%s\n%s\n' "$LEAKS" "$USER_HITS" | grep -v '^$' | head -20 >&2
    FAILED=1
  fi
done
if [ "$FAILED" -ne 0 ]; then
  exit 1
fi

echo "==> Wrote _site (tag $TAG)"
