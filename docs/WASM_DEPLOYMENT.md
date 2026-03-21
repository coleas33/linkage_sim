# WASM Deployment Guide

This document covers building, testing, and deploying the linkage simulator as a WebAssembly application.

## Prerequisites

1. **Rust toolchain** with the WASM target:
   ```bash
   rustup target add wasm32-unknown-unknown
   ```

2. **wasm-bindgen-cli** (must match the version used in CI, currently 0.2.114):
   ```bash
   cargo install wasm-bindgen-cli@0.2.114
   ```

3. **Python 3** (for the local dev server)

4. **Vercel CLI** (for deployment only):
   ```bash
   npm install -g vercel
   ```

## Build Steps

Run the build script from the repository root:

```bash
linkage-sim-rs/scripts/build_web.sh
```

This script performs two steps:

1. **Compile to WASM** (release mode, no default features):
   ```bash
   cargo build --release \
       --target wasm32-unknown-unknown \
       --bin linkage-web \
       --no-default-features
   ```

2. **Generate JS bindings** via wasm-bindgen:
   ```bash
   wasm-bindgen \
       target/wasm32-unknown-unknown/release/linkage-web.wasm \
       --out-dir web \
       --target web \
       --no-typescript
   ```

Output artifacts land in `linkage-sim-rs/web/`:
- `linkage-web.js` -- JS glue code
- `linkage-web_bg.wasm` -- compiled WASM binary

## Local Testing

After building, serve the `web/` directory locally:

```bash
linkage-sim-rs/scripts/serve_web.sh
```

This starts a Python HTTP server at `http://localhost:8080`. The script will exit with an error if the WASM binary has not been built yet.

You can also serve manually:

```bash
cd linkage-sim-rs/web && python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

## Vercel Deployment

Production deployments are automated via the GitHub Actions workflow at `.github/workflows/deploy-web.yml`. A push to `main` triggers the full pipeline:

1. Check out the repo.
2. Install the stable Rust toolchain with the `wasm32-unknown-unknown` target.
3. Restore the Cargo cache (keyed on `Cargo.lock`).
4. Install `wasm-bindgen-cli@0.2.114`.
5. Build the WASM binary and JS bindings (same commands as `build_web.sh`).
6. Install the Vercel CLI.
7. Pull the Vercel environment configuration.
8. Build the Vercel output (`vercel build --prod`).
9. Deploy to Vercel (`vercel deploy --prebuilt --prod`).

### Required GitHub Secrets

| Secret              | Description                        |
|---------------------|------------------------------------|
| `VERCEL_TOKEN`      | Vercel personal access token       |
| `VERCEL_ORG_ID`     | Vercel organization / team ID      |
| `VERCEL_PROJECT_ID` | Vercel project ID for this app     |

### Vercel Configuration

The file `linkage-sim-rs/web/vercel.json` configures:

- `outputDirectory` set to `.` (the `web/` folder itself is the deploy root).
- A header rule serving `.wasm` files with `Content-Type: application/wasm` and an immutable cache policy (`max-age=31536000`).

## Known Limitations (WASM Build)

The following features are **not available** in the browser / WASM build:

- **No file dialogs** -- Save, Open, and Save As use native file dialogs (`rfd` crate) which are gated behind the `native` feature flag and excluded from the WASM build.
- **No PNG / SVG / GIF export** -- Export functions rely on native filesystem access.
- **No autosave** -- The browser build has no persistent local storage integration; work is lost on page reload.
- **No recent-files list** -- Depends on native filesystem paths.
