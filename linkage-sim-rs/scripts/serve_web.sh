#!/usr/bin/env bash
# Serve the web build locally.
# Run build_web.sh first if you haven't already.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_DIR="$(dirname "$SCRIPT_DIR")/web"

if [ ! -f "$WEB_DIR/linkage-web_bg.wasm" ]; then
    echo "WASM not found. Run build_web.sh first."
    exit 1
fi

echo "Serving at http://localhost:8080"
echo "Press Ctrl+C to stop."
cd "$WEB_DIR" && python -m http.server 8080
