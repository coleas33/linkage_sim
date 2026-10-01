#!/usr/bin/env bash
# Build the magnetic coupling calculator (magcoupling-rs, feature app) for
# WebAssembly: the second web bundle, served at /magcoupling/ next to the
# linkage app. Called by build_web.sh and by .github/workflows/deploy-web.yml.
#
# Prerequisites: as build_web.sh (rustup target add wasm32-unknown-unknown;
# wasm-bindgen-cli at the wasm-bindgen version in magcoupling-rs/Cargo.lock,
# which gate.sh keeps equal to linkage-sim-rs's and to deploy-web.yml's pin).
#
# Output: web/magcoupling/magcoupling-web.js + magcoupling-web_bg.wasm (both
# gitignored), next to the committed web/magcoupling/index.html.
#
# The build fails if cargo built the bundle with the test-only workbook-parity
# feature (magcoupling_assert_shipped in magcoupling_shipped.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MAGCOUPLING_DIR="$(dirname "$PROJECT_DIR")/magcoupling-rs"
OUT_DIR="$PROJECT_DIR/web/magcoupling"

# shellcheck source=magcoupling_shipped.sh
source "$SCRIPT_DIR/magcoupling_shipped.sh"

echo "Building magcoupling WASM binary (release)..."
cargo build --release \
    --manifest-path "$MAGCOUPLING_DIR/Cargo.toml" \
    "${MAGCOUPLING_WEB_ARGS[@]}" \
    --message-format=json-render-diagnostics \
    | magcoupling_assert_shipped magcoupling-web

echo "Generating JS bindings..."
wasm-bindgen \
    "$MAGCOUPLING_DIR/target/wasm32-unknown-unknown/release/magcoupling-web.wasm" \
    --out-dir "$OUT_DIR" \
    --target web \
    --no-typescript

WASM_SIZE=$(du -h "$OUT_DIR/magcoupling-web_bg.wasm" | cut -f1)
echo ""
echo "Magcoupling build complete!"
echo "  web/magcoupling/magcoupling-web.js       (JS glue)"
echo "  web/magcoupling/magcoupling-web_bg.wasm  ($WASM_SIZE)"
