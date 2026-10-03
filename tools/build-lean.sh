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
set -euo pipefail

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

REMAP="--remap-path-prefix=$HOME=~ --remap-path-prefix=$HOME/.cargo=cargo"
OUT_ST="${OUT_ST:-web/pkg-lean}"
OUT_MT="${OUT_MT:-web/pkg-lean-mt}"

echo "==> $OUT_ST (single thread, SIMD128)"
RUSTFLAGS="-C target-feature=+simd128 $REMAP" \
    wasm-pack build crates/llm-life-lean --target web --out-dir "../../$OUT_ST" --no-default-features --features web

echo "==> $OUT_MT (threads, nightly build-std)"
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
-C link-arg=--shared-memory -C link-arg=--max-memory=2147483648 \
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
sed -i.bak "s#await import('\.\./\.\./\.\.')#await import('../../../llm_life_lean.js')#" "$HELPER"
rm -f "$HELPER.bak"

for f in "$OUT_ST/llm_life_lean_bg.wasm" "$OUT_MT/llm_life_lean_bg.wasm"; do
    echo "==> $f: $(wc -c < "$f") bytes, $(strings "$f" | grep -c "$HOME" || true) home-directory strings, sha256 $(shasum -a 256 "$f" | cut -d' ' -f1)"
done
