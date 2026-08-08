#!/usr/bin/env bash
# Verification gate for the agentic fix loop.
# Usage: scripts/gate.sh [--full]   (--full also runs the release WASM build)
# Exit 0 + "GATE PASS" = all gates green. Any failure exits non-zero.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== gate 1/3: cargo test --all =="
cargo test --all

echo "== gate 2/3: cargo clippy --all-targets =="
cargo clippy --all-targets

echo "== gate 3/3: WASM check =="
cargo check --target wasm32-unknown-unknown --bin linkage-web --no-default-features --features raster

if [[ "${1:-}" == "--full" ]]; then
  echo "== gate 4 (--full): release WASM build =="
  ./scripts/build_web.sh
fi

echo "GATE PASS"
