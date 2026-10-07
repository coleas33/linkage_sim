#!/usr/bin/env bash
# Build the linkage simulator for WebAssembly deployment, then the magnetic
# coupling calculator's bundle (build_magcoupling_web.sh).
#
# Prerequisites:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli@0.2.114   (the version in both Cargo.lock files)
#
# Output: web/tools/linkage/linkage-web.js + linkage-web_bg.wasm, and
#         web/tools/magcoupler/magcoupling-web.js + magcoupling-web_bg.wasm
#         (served at /tools/linkage/ and /tools/magcoupler/; the hub page
#         web/tools/index.html and the root's redirect page are committed)
#
# The calculator's build fails if cargo built it with the test-only
# workbook-parity feature (magcoupling_assert_shipped in magcoupling_shipped.sh);
# the linkage bundle has no magcoupling-rs in it. deploy-web.yml runs this script.
#
# After building, serve with: scripts/serve_web.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=magcoupling_shipped.sh
source "$SCRIPT_DIR/magcoupling_shipped.sh"

cd "$PROJECT_DIR"

echo "Building WASM binary (release)..."
cargo build --release "${LINKAGE_WEB_ARGS[@]}"

echo "Generating JS bindings..."
wasm-bindgen \
    target/wasm32-unknown-unknown/release/linkage-web.wasm \
    --out-dir web/tools/linkage \
    --target web \
    --no-typescript

WASM_SIZE=$(du -h web/tools/linkage/linkage-web_bg.wasm | cut -f1)
echo ""
echo "Build complete!"
echo "  web/tools/linkage/linkage-web.js       (JS glue)"
echo "  web/tools/linkage/linkage-web_bg.wasm  ($WASM_SIZE)"
echo ""

bash "$SCRIPT_DIR/build_magcoupling_web.sh"

echo ""
echo "To serve locally:"
echo "  $SCRIPT_DIR/serve_web.sh"
echo "  Then open http://localhost:8080/tools/ (the hub), /tools/linkage/ and /tools/magcoupler/"
