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

1. **The linkage app** (`web/tools/linkage/`, served at `/tools/linkage/`): `cargo build --release` with the
   cargo arguments `LINKAGE_WEB_ARGS` from `scripts/magcoupling_shipped.sh` (`--bin linkage-web
   --target wasm32-unknown-unknown --no-default-features --features raster`), then the JS bindings:
   ```bash
   wasm-bindgen \
       target/wasm32-unknown-unknown/release/linkage-web.wasm \
       --out-dir web/tools/linkage \
       --target web \
       --no-typescript
   ```

2. **The magnetic coupling calculator** (`web/tools/magcoupler/`, served at `/tools/magcoupler/`):
   `scripts/build_magcoupling_web.sh`, the same way with `MAGCOUPLING_WEB_ARGS`, piped through the
   workbook-parity guard (`magcoupling_assert_shipped`: the shipped calculator must never have
   magcoupling-rs's test-only `workbook-parity` feature).

The hub page, `web/tools/index.html` (served at `/tools/`), links both; `web/index.html` forwards `/` to it for a server without the redirect. Both are committed as they are.

To build by hand, run the script rather than copying its commands: the cargo arguments live once,
in `scripts/magcoupling_shipped.sh`, and the guard runs only through the scripts.

Output artifacts land in `linkage-sim-rs/web/`:
- `linkage/linkage-web.js` -- JS glue code
- `linkage/linkage-web_bg.wasm` -- compiled WASM binary
- `magcoupler/magcoupling-web.js` and `magcoupler/magcoupling-web_bg.wasm` -- the calculator's bundle

## Local Testing

After building, serve the `web/` directory locally:

```bash
linkage-sim-rs/scripts/serve_web.sh
```

This starts a Python HTTP server at `http://localhost:8080`: the hub page at `/tools/` (`/` forwards there), the linkage app at `/tools/linkage/` and the calculator at `/tools/magcoupler/`. The script will exit with an error if the WASM binary has not been built yet. The redirects in `vercel.json` run on Vercel only.

You can also serve manually:

```bash
cd linkage-sim-rs/web && python -m http.server 8080
```

Then open `http://localhost:8080/tools/` in your browser.

## Vercel Deployment

Production deployments are automated via the GitHub Actions workflow at `.github/workflows/deploy-web.yml`. A push to `main` triggers the full pipeline:

1. Check out the repo.
2. Install the stable Rust toolchain with the `wasm32-unknown-unknown` target.
3. Restore the Cargo cache (keyed on both `Cargo.lock` files, `linkage-sim-rs` and `magcoupling-rs`).
4. Install `wasm-bindgen-cli@0.2.114`.
5. Run `scripts/build_web.sh`: both bundles, the calculator's piped through the workbook-parity guard.
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

- `outputDirectory` set to `.` (the `web/` folder itself is the deploy root), served on colesorkness.com.
- `trailingSlash: true`, so `/tools/linkage` becomes `/tools/linkage/` (the pages import their glue by absolute paths either way).
- Redirects, first match wins, permanent (308) except the root's, which stays temporary (307) so a page of its own can replace it later: on linkage.colesorkness.com, `/magcoupling/...` to colesorkness.com/tools/magcoupler/..., `/?tool=magcoupling` (without `m`) to the calculator, and every other path to colesorkness.com/tools/linkage/...; on any host, `/magcoupling/...` to `/tools/magcoupler/...` and `/` to `/tools/`. Share links keep their `m` through an explicit capture (`has` query `m`, `?m=:m`), and the sources use `:path(.*)`, not `:path*`, which Vercel compiles so strictly that it never matches a path ending in `/`. `tests/web_layout.rs` pins them.
- Header rules: `.wasm` files with `Content-Type: application/wasm`, and the wasm and both JS glue files with `Cache-Control: public, max-age=0, must-revalidate`.

## Known Limitations (WASM Build)

The following features are **not available** in the browser / WASM build:

- **No file dialogs** -- Save, Open, and Save As use native file dialogs (`rfd` crate) gated behind the `native` feature flag.
- **No PNG / SVG / GIF export** -- Export functions rely on native filesystem access.
- **No autosave** -- The browser build has no persistent local storage integration; work is lost on page reload.
- **No recent-files list** -- Depends on native filesystem paths.
