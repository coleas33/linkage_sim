#!/usr/bin/env bash
# Build the linkage simulator for WebAssembly deployment.
#
# Prerequisites:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli
#
# Output: web/linkage-web.js + web/linkage-web_bg.wasm
#
# After building, serve with: python -m http.server 8080 --directory web

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Building WASM binary (release)..."
cargo build --release \
    --target wasm32-unknown-unknown \
    --bin linkage-web \
    --no-default-features

echo "Generating JS bindings..."
wasm-bindgen \
    target/wasm32-unknown-unknown/release/linkage-web.wasm \
    --out-dir web \
    --target web \
    --no-typescript

WASM_SIZE=$(du -h web/linkage-web_bg.wasm | cut -f1)
echo ""
echo "Build complete!"
echo "  web/linkage-web.js       (JS glue)"
echo "  web/linkage-web_bg.wasm  ($WASM_SIZE)"
echo ""
echo "To serve locally:"
echo "  cd $PROJECT_DIR/web && python -m http.server 8080"
echo "  Then open http://localhost:8080"
