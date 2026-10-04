#!/usr/bin/env bash
# Build crates/llm-life-lean for the browser, both CPU-capable variants:
#   web/pkg-lean     single-thread module (WebGPU backend + CPU backend, SIMD128)
#   web/pkg-lean-mt  threaded module (same, CPU matmuls on a wasm-bindgen-rayon
#                    pool); needs a cross-origin-isolated page (web/serve.py
#                    sends COOP/COEP) and nightly with rust-src.
# The threaded recipe is lean's own (llm-web crates/lean, scripts/build_lean_mt.sh):
# nightly -Z build-std with atomics, wasm-bindgen by hand, then the
# workerHelpers.js import fixed for a no-bundler `--target web` page.
#
# LEAN_PATH=<llm-web checkout>/crates/lean builds lean from a local checkout
# (the branch is unpublished); the patch goes in a temporary
# .cargo/config.toml because wasm-pack runs its own `cargo metadata`.
# RUSTFLAGS replaces any config rustflags, so the home-directory remaps are
# passed here too: the built wasm carries no local paths.
#
# ENGINE_BUILD is required and is the same `?v=` tag the pages put on their
# loading URLs; it tags pkg-lean-mt's worker chain the way llm-web's
# scripts/build_lean_mt.sh does, so a worker never runs a cached glue file
# from an older build. MAX_MEMORY (bytes) overrides pkg-lean-mt's shared
# memory maximum; the default matches build_lean_mt.sh's 2.5 GiB
# (SmolLM2-1.7B Q4_0 trapped at 1-2 GiB, see llm-web docs/runs/2026-10-04-lean-release.md).
set -euo pipefail

: "${ENGINE_BUILD:?set ENGINE_BUILD to the ?v= tag the pages load this build with}"
MAX_MEMORY="${MAX_MEMORY:-2684354560}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -n "${LEAN_PATH:-}" ]; then
    if [ -e .cargo/config.toml ]; then
        echo "error: .cargo/config.toml exists; refusing to overwrite it" >&2
        exit 1
    fi
    mkdir -p .cargo
    printf '[patch."https://github.com/idle-intelligence/llm-web"]\nlean = { path = "%s" }\n' "$LEAN_PATH" > .cargo/config.toml
    trap 'rm -f "$REPO_ROOT/.cargo/config.toml"; rmdir "$REPO_ROOT/.cargo" 2>/dev/null || true' EXIT
fi

# Order matters: when several --remap-path-prefix rules match the same
# path, rustc applies the LAST matching rule in the argument list (verified
# against rustc 1.93: a later rule overrides an earlier one, not the other
# way round), so the most specific (deepest) prefixes must come LAST or
# they'd be overridden by the more general $HOME rule. LEAN_PATH (when set)
# and REPO_ROOT are both under $HOME, so they go last, each remapped to a
# neutral crate name rather than ~/..., which would otherwise survive
# as a private workspace/worktree layout.
REMAP="--remap-path-prefix=$HOME=~ --remap-path-prefix=$HOME/.cargo=cargo --remap-path-prefix=$REPO_ROOT=llm-life"
if [ -n "${LEAN_PATH:-}" ]; then
    REMAP="$REMAP --remap-path-prefix=$LEAN_PATH=lean"
fi
OUT_ST="${OUT_ST:-web/pkg-lean}"
OUT_MT="${OUT_MT:-web/pkg-lean-mt}"

echo "==> $OUT_ST (single thread, SIMD128)"
RUSTFLAGS="-C target-feature=+simd128 $REMAP" \
    wasm-pack build crates/llm-life-lean --target web --out-dir "../../$OUT_ST" --no-default-features --features web

echo "==> $OUT_MT (threads, nightly build-std)"
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
-C link-arg=--shared-memory -C link-arg=--max-memory=$MAX_MEMORY \
-C link-arg=--import-memory \
-C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align -C link-arg=--export=__tls_base $REMAP" \
    cargo +nightly build -p llm-life-lean --lib \
        --target wasm32-unknown-unknown --release \
        --no-default-features --features web-mt \
        -Z build-std=panic_abort,std
WASM_IN="target/wasm32-unknown-unknown/release/llm_life_lean.wasm"
rm -rf "$OUT_MT"
mkdir -p "$OUT_MT"
wasm-bindgen --target web --out-dir "$OUT_MT" --out-name llm_life_lean "$WASM_IN"
HELPER=$(find "$OUT_MT/snippets" -name workerHelpers.js | head -1)
if [ -z "$HELPER" ]; then
    echo "error: workerHelpers.js not found under $OUT_MT/snippets" >&2
    exit 1
fi
sed -i.bak \
    -e "s#await import('\.\./\.\./\.\.')#await import('../../../llm_life_lean.js?v=$ENGINE_BUILD')#" \
    -e "s#new URL('\./workerHelpers\.js', import\.meta\.url)#new URL('./workerHelpers.js?v=$ENGINE_BUILD', import.meta.url)#" \
    "$HELPER"
if ! grep -q "llm_life_lean.js?v=$ENGINE_BUILD" "$HELPER" || ! grep -q "workerHelpers.js?v=$ENGINE_BUILD" "$HELPER"; then
    echo "error: workerHelpers.js patch did not apply" >&2
    exit 1
fi
rm -f "$HELPER.bak"

# Fail the build if any private path fragment survived remapping: the
# literal $HOME, any "Code/" or "local-tool-state/" workspace/worktree layout,
# "~/" (macOS home root), or the user name as a path component. The
# user name is anchored to slashes so it doesn't false-positive on ordinary
# words like "match" or "dispatch" that happen to contain the same letters.
LEAK_PATTERN="$HOME|Code/|\\local-tool-state/|~/|/$(id -un)/"
for f in "$OUT_ST/llm_life_lean_bg.wasm" "$OUT_MT/llm_life_lean_bg.wasm"; do
    leak_count=$(strings "$f" | grep -cE "$LEAK_PATTERN" || true)
    echo "==> $f: $(wc -c < "$f") bytes, $leak_count leaked-path strings, sha256 $(shasum -a 256 "$f" | cut -d' ' -f1)"
    if [ "$leak_count" -ne 0 ]; then
        echo "error: $f still contains private path strings:" >&2
        strings "$f" | grep -E "$LEAK_PATTERN" | sort -u >&2
        exit 1
    fi
done
