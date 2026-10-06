#!/usr/bin/env bash
# Serve the web build locally as colesorkness.com lays it out: the hub page at
# http://localhost:PORT/, the linkage app at /linkage/ and the magnetic
# coupling calculator at /magcoupler/. (The redirects in web/vercel.json run
# on Vercel only.)
# Usage: scripts/serve_web.sh [PORT]   (default 8080)
# Run build_web.sh first if you haven't already.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_DIR="$(dirname "$SCRIPT_DIR")/web"
PORT="${1:-8080}"

if [ ! -f "$WEB_DIR/linkage/linkage-web_bg.wasm" ]; then
    echo "WASM not found. Run build_web.sh first."
    exit 1
fi
if [ ! -f "$WEB_DIR/magcoupler/magcoupling-web_bg.wasm" ]; then
    echo "Magcoupling WASM not found: /magcoupler/ will not load. Run build_web.sh (or build_magcoupling_web.sh)."
fi

echo "Serving at http://localhost:$PORT/ (hub), http://localhost:$PORT/linkage/ and http://localhost:$PORT/magcoupler/"
echo "Press Ctrl+C to stop."
cd "$WEB_DIR" && python -m http.server "$PORT"
