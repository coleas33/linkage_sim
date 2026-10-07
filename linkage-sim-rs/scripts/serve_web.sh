#!/usr/bin/env bash
# Serve the web build locally as colesorkness.com lays it out: the hub page at
# http://localhost:PORT/tools/, the linkage app at /tools/linkage/ and the
# magnetic coupling calculator at /tools/magcoupler/ (/ forwards to /tools/
# through web/index.html; the redirects in web/vercel.json run on Vercel only).
# Usage: scripts/serve_web.sh [PORT]   (default 8080)
# Run build_web.sh first if you haven't already.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_DIR="$(dirname "$SCRIPT_DIR")/web"
PORT="${1:-8080}"

if [ ! -f "$WEB_DIR/tools/linkage/linkage-web_bg.wasm" ]; then
    echo "WASM not found. Run build_web.sh first."
    exit 1
fi
if [ ! -f "$WEB_DIR/tools/magcoupler/magcoupling-web_bg.wasm" ]; then
    echo "Magcoupling WASM not found: /tools/magcoupler/ will not load. Run build_web.sh (or build_magcoupling_web.sh)."
fi

echo "Serving at http://localhost:$PORT/tools/ (the hub), http://localhost:$PORT/tools/linkage/ and http://localhost:$PORT/tools/magcoupler/"
echo "Press Ctrl+C to stop."
cd "$WEB_DIR" && python -m http.server "$PORT"
