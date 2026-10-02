#!/usr/bin/env bash
# Build the linkage simulator for WebAssembly deployment, then the magnetic
# coupling calculator's bundle (build_magcoupling_web.sh).
#
# Prerequisites:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli@0.2.114   (the version in both Cargo.lock files)
#
# Output: web/linkage-web.js + web/linkage-web_bg.wasm, and
#         web/magcoupling/magcoupling-web.js + magcoupling-web_bg.wasm
#
# Both builds fail if cargo built the bundle with the test-only workbook-parity
# feature of magcoupling-rs (magcoupling_assert_shipped in magcoupling_shipped.sh):
# the linkage bundle carries the calculator's panel (Tools -> Magnetic coupling).
# deploy-web.yml runs this script.
#
# After building, serve with: scripts/serve_web.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=magcoupling_shipped.sh
source "$SCRIPT_DIR/magcoupling_shipped.sh"

cd "$PROJECT_DIR"

echo "Building WASM binary (release)..."
cargo build --release "${LINKAGE_WEB_ARGS[@]}" \
    --message-format=json-render-diagnostics \
    | magcoupling_assert_shipped linkage-web

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

bash "$SCRIPT_DIR/build_magcoupling_web.sh"

echo ""
echo "To serve locally:"
echo "  $SCRIPT_DIR/serve_web.sh"
echo "  Then open http://localhost:8080 (linkage) and http://localhost:8080/magcoupling/"
