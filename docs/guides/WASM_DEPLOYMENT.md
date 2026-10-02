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

This script builds both web bundles:

1. **The linkage app** (`web/`): `cargo build --release` with the shipped cargo arguments
   `LINKAGE_WEB_ARGS` from `scripts/magcoupling_shipped.sh` (`--bin linkage-web --target
   wasm32-unknown-unknown --no-default-features --features raster`), piped through the
   workbook-parity guard (`magcoupling_assert_shipped`: the bundle carries the magnetic coupling
   calculator's panel, Tools > Magnetic coupling, and must never have magcoupling-rs's test-only
   `workbook-parity` feature), then the JS bindings:
   ```bash
   wasm-bindgen \
       target/wasm32-unknown-unknown/release/linkage-web.wasm \
       --out-dir web \
       --target web \
       --no-typescript
   ```

2. **The magnetic coupling calculator** (`web/magcoupling/`, served at `/magcoupling/`):
   `scripts/build_magcoupling_web.sh`, the same way with `MAGCOUPLING_WEB_ARGS`.

To build by hand, run the script rather than copying its commands: the cargo arguments live once,
in `scripts/magcoupling_shipped.sh`, and the guard runs only through the scripts.

Output artifacts land in `linkage-sim-rs/web/`:
- `linkage-web.js` -- JS glue code
- `linkage-web_bg.wasm` -- compiled WASM binary
- `magcoupling/magcoupling-web.js` and `magcoupling/magcoupling-web_bg.wasm` -- the calculator's bundle

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
5. Run `scripts/build_web.sh`: both bundles, each piped through the workbook-parity guard.
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

- **No file dialogs** -- Save, Open, and Save As use native file dialogs (`rfd` crate) gated behind the `native` feature flag. The web build carries rfd only for the magnetic coupling calculator window's Load design (the browser's file chooser).
- **No PNG / SVG / GIF export** -- Export functions rely on native filesystem access.
- **No autosave** -- The browser build has no persistent local storage integration; work is lost on page reload.
- **No recent-files list** -- Depends on native filesystem paths.
