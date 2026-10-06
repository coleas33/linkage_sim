# colesorkness.com: the URL move and the standalone calculator Implementation Plan

> **For agentic workers:** this plan was executed natively (decision 0 below): the dry run on branch `linkage/url-move` is the implementation, in the commits listed under Tasks. A fresh reviewer on the session model checks the whole branch before the merge.

**Goal:** Serve the two apps from colesorkness.com (a hub page at `/`, the linkage simulator at `/linkage/`, the magnetic coupling calculator at `/magcoupler/`), keep every old linkage.colesorkness.com link working, and stop embedding the calculator in the linkage app: Tools -> Magnetic coupling calculator opens the calculator's own site.

**Architecture:** One Vercel project (linkage-simulator) keeps serving `linkage-sim-rs/web/`, now with colesorkness.com attached. The web folder gains a hub page and moves each app into its own folder (`web/linkage/`, `web/magcoupler/`); `vercel.json` adds `trailingSlash` and host-scoped redirects from the old subdomain. The linkage app drops `gui/calculator_window.rs` and its `magcoupling-rs` dependency; its Tools menu asks egui to open the calculator's URL in a new tab. The gate's checks that existed because the linkage app compiled the calculator in (the linkage builds in gate 10's guard; egui/eframe/xlsx lock parity in gate 11) go; wasm-bindgen parity against the one CLI that binds both bundles stays.

**Tech Stack:** Rust 2024, egui/eframe 0.32 (`Context::open_url`, `OutputCommand::OpenUrl`), Vercel static hosting with `vercel.json` redirects, Cloudflare DNS (the user's), Playwright for the browser checks.

**Spec:** the user's requests: 2026-10-01 "move linkage.colesorkness.com to colesorkness.com/linkage and /magcoupling/ to colesorkness.com/magcoupler" ("probably next week"); 2026-10-03 "i don't want the calculator to be a tab in the linkage tool. Can it be a standalone site that you can get to from the linkage tool?"; 2026-10-06 "1A: URL move + un-embed", then issues 1-4 answered 1A, 2A, 3A, 4A (below).

## Decisions

**Confirmed by the user (2026-10-06):**

| Id | Question | Chosen |
|---|---|---|
| 1 | How the paths are served | **A:** colesorkness.com attached to the existing Vercel project; the user switches the apex DNS record in Cloudflare (DNS only) |
| 2 | colesorkness.com/ | **A:** a small hub page with two cards (Linkage simulator, Magnetic coupler); the dark landing page later becomes the front of /magcoupler |
| 3 | The calculator's path | **A:** `/magcoupler` (old `/magcoupling/` paths redirect permanently, share links included) |
| 4 | Old links | **A:** linkage.colesorkness.com stays attached and redirects (308) every path to the new addresses, keeping the `?m=` share data; `?tool=magcoupling` goes to the calculator |

**Rulings made while building (to confirm):**

| Id | Ruling (implemented) | Alternative |
|---|---|---|
| 0 | Executed natively: the plan writer built and dry-ran it on the branch, then one session-model reviewer checks the whole branch | subagent-driven transcription of the same code |
| U-1 | The menu item reads **"Magnetic coupling calculator"**, its hover text says it opens the calculator's site in a new browser tab (no "↗": egui's default fonts may lack the glyph; ASCII rule) | "Magnetic coupling calculator ->" |
| U-2 | **On the web the link is the same-origin path `/magcoupler/`** (a local server opens its own copy); the desktop app opens `https://colesorkness.com/magcoupler/` | always the public URL |
| U-3 | **Redirects rely on Vercel passing the query through** (old share links keep `?m=`), verified live right after the push; a prepared fallback captures `m` explicitly (Task 5) | the explicit capture from the start |
| U-4 | **The hub page is one self-contained file** in the linkage app's dark palette, no fonts or scripts from elsewhere; title "Engineering Tools", one-line card texts (wording is the user's to change) | Tailwind from a CDN now (the landing page will bring it later) |
| U-5 | **Gate 10 guards only the calculator's two shipped builds; gate 11 checks only wasm-bindgen** in both lock files against the `wasm-bindgen-cli` pin in deploy-web.yml (the one CLI binds both bundles). egui, egui_plot, eframe and rust_xlsxwriter parity is dropped: the crates no longer share a build | keep egui/eframe parity so upgrades stay in lockstep |
| U-6 | **`trailingSlash: true`, and both app pages import their glue and the favicon by absolute paths** (`/linkage/linkage-web.js`, `/magcoupler/magcoupling-web.js`, `/favicon.svg`) | relative paths plus `trailingSlash` only |
| U-7 | **BL-038, BL-039 and BL-040 closed as fixed by removal** (the window, its picker copy and its keyboard edges are gone) | leave them open as historical |
| U-8 | **Rollout order:** (1) the user attaches colesorkness.com (and www, redirecting to the apex) in Vercel and points Cloudflare at it, DNS only; (2) the controller checks colesorkness.com serves the current site; (3) merge and push; (4) the live checks of Task 5. Rollback: revert the merge commit (the subdomain stays attached, so the old site comes back as it was) | push first, then switch DNS |

## Global Constraints

- Worktree `C:/Users/Cole/source/repos/lsim-urls`, branch `linkage/url-move`, from `main` at `7d6d323`, LF line endings, worktree-scoped `core.autocrlf=false`. The main checkout is not modified until the merge.
- The calculator's own app is unchanged apart from its address (`PUBLIC_BASE_URL`, its page's folder and glue path).
- Share links made before the move must keep opening their mechanism or design (the redirects), and links made after it point at colesorkness.com.
- Nothing is pushed until colesorkness.com serves the current site from Vercel (U-8) and the user says go.
- The repo is public: no credentials, DNS tokens or the user's models in it.
- Docs change with code: README, FEATURES, `docs/guides/WASM_DEPLOYMENT.md`, `docs/ai/01-05`, backlog, `magcoupling-rs/README.md`.
- Never run `cargo fmt`; `docs/chebyshev_lambda/*.png` never committed.

## Review Focus

1. **An old share link** (`linkage.colesorkness.com/?m=...`). Expected: lands on `colesorkness.com/linkage/?m=...` with the same mechanism. Pinned by `tests/web_layout.rs::old_links_redirect_to_the_new_addresses` (the rule) and Task 5's live check (the query).
2. **`/linkage` without the slash.** Expected: `/linkage/` with the app (a relative glue path would resolve against `/`). `folders_get_their_trailing_slash`, plus absolute imports (`each_app_page_imports_its_glue_where_its_build_script_writes_it`).
3. **The Tools item on the desktop and on the web.** Expected: one new-tab request to the right URL, the menu closed. `tools_opens_the_magnetic_coupling_calculator_s_site_in_a_new_tab`, `the_desktop_app_opens_the_public_calculator_site`.
4. **A stale reference to the old layout** in a script, a page, `.gitignore` or `vercel.json`. `tests/web_layout.rs` (six tests) ties them together; `the_canvas_id_matches_the_web_page` and `the_web_page_background_is_the_panel_colour` read the moved calculator page.
5. **The guard and the parity gate after the dependency is gone.** Expected: gate 10 still trips its negative controls on both calculator builds; gate 11 fails if either lock drifts from the CLI pin. Gate run (Task 4).

## Layout

| Path on colesorkness.com | Source | Built by |
|---|---|---|
| `/` | `web/index.html` (committed) | - |
| `/linkage/` | `web/linkage/index.html` + `linkage-web.js`, `linkage-web_bg.wasm` | `scripts/build_web.sh` (`--out-dir web/linkage`) |
| `/magcoupler/` | `web/magcoupler/index.html` + `magcoupling-web.js`, `magcoupling-web_bg.wasm` | `scripts/build_magcoupling_web.sh` (`OUT_DIR=web/magcoupler`) |
| `/favicon.svg` | `web/favicon.svg` | - |

Redirects (`web/vercel.json`, permanent, first match wins):

| Host | Source | Destination |
|---|---|---|
| linkage.colesorkness.com | `/magcoupling/:path*` | `https://colesorkness.com/magcoupler/:path*` |
| linkage.colesorkness.com | `/` with `?tool=magcoupling` | `https://colesorkness.com/magcoupler/` |
| linkage.colesorkness.com | `/:path*` | `https://colesorkness.com/linkage/:path*` |
| any | `/magcoupling/:path*` | `/magcoupler/:path*` |

## Tasks

### Task 1: The colesorkness.com layout and addresses — commit `670b184`

**Files:** `web/index.html` (the hub, rewritten), `web/linkage/index.html` (moved from `web/index.html`; absolute paths), `web/magcoupler/index.html` (moved from `web/magcoupling/`), `web/vercel.json`, `web/.gitignore`, `scripts/build_web.sh`, `scripts/build_magcoupling_web.sh`, `scripts/serve_web.sh`, `src/gui/state/file_io.rs` (`SHARE_URL_BASE` and a test), `tests/web_layout.rs` (new, six tests), `src/gui/calculator_window.rs` (the page path, kept consistent until Task 2), `.claude/workflows/gui-smoke.js` (URLs); `magcoupling-rs/src/gui/session.rs` (`PUBLIC_BASE_URL` and its test), `src/app.rs` and `src/app/theme.rs` (the page path their tests read), `src/bin/magcoupling_web.rs` (docs).

Checks: `cargo test --test web_layout` 6 passed; the share-link and calculator-window tests 44 passed; magcoupling-rs's share, page and smoke tests 13 passed.

### Task 2: The standalone calculator — commit `5ab68c3`

**Files:** `src/gui/calculator_window.rs` (deleted), `src/gui/mod.rs` (the field, its two methods, the update call, the Help entry and the window's tests removed), `src/gui/menu_bar.rs` (`MAGCOUPLER_ITEM`, `MAGCOUPLER_URL`, the Tools item, new tests), `src/gui/test_support.rs` (helpers only the window's tests used: `magcoupling_gap_design`, `key_tap`, `drag_events`, `text_clip_rect`), `src/bin/linkage_web.rs` (no `?tool` handling), `Cargo.toml` (no `magcoupling-rs`, no wasm-only `rfd`), `Cargo.lock`, `scripts/gate.sh` (gates 10 and 11), `scripts/magcoupling_shipped.sh`, `scripts/build_web.sh` (the linkage build no longer piped through the guard), `.claude/workflows/gui-smoke.js` (a hub check in place of the calculator-window step).

Checks: `cargo test --lib menu_bar` 2 passed; `cargo build` and the wasm32 check of `linkage-web` clean apart from warnings already on `main`.

### Task 3: Docs and this plan — this commit

README, `docs/FEATURES.md`, `docs/guides/WASM_DEPLOYMENT.md`, `docs/ai/01-meta.yaml` (deployment), `02-system.yaml` (the calculator's entries, the `colesorkness_com_layout` invariant in place of the calculator window's key invariant, the limitations that went with the window), `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`, `backlog.yaml` (BL-038 to BL-040 fixed), `magcoupling-rs/README.md`.

### Task 4: Gate, local web build and browser check (controller)

- [ ] Gate: `MAGCOUPLING_PYTHON=... bash linkage-sim-rs/scripts/gate.sh` ends `GATE PASS` (gate 10 now four lines of guard output, two negative controls; gate 11 two wasm-bindgen lines). Restore `docs/chebyshev_lambda`.
- [ ] `bash linkage-sim-rs/scripts/build_web.sh`, then `serve_web.sh 8765`; Playwright: `/` shows both cards; `/linkage/` and `/linkage` (redirected by the local server) load the app; a share link `/linkage/?m=...` opens its mechanism; `/magcoupler/?m=...` opens its design; Tools -> Magnetic coupling calculator opens a new tab at `/magcoupler/`; zero console errors.
- [ ] The whole-branch review (session model), one fix wave if needed.

### Task 5: Rollout (after the user's DNS step and go)

- [ ] Before the push: `curl -sI https://colesorkness.com/` answers from Vercel (`server: Vercel`) with the current site.
- [ ] Merge `linkage/url-move` into `main` with `--no-ff`; push; wait for "Deploy Web to Vercel".
- [ ] Live checks:
  - `curl -sI https://colesorkness.com/linkage` -> 308 to `/linkage/`; `/linkage/` and `/magcoupler/` 200; `/` the hub.
  - `curl -sI 'https://linkage.colesorkness.com/?m=TEST'` -> 308, `location: https://colesorkness.com/linkage/?m=TEST`.
  - `curl -sI 'https://linkage.colesorkness.com/?tool=magcoupling'` -> 308 to `https://colesorkness.com/magcoupler/` (a trailing `?tool=magcoupling` is harmless).
  - `curl -sI 'https://linkage.colesorkness.com/magcoupling/?m=TEST'` -> 308 to `https://colesorkness.com/magcoupler/?m=TEST`.
  - Playwright: the press share link from the session opens on the new address with its 21,506 N peak; the calculator's share link opens its design.
- [ ] If the query is not kept on a redirect, replace the old host's catch-all with these two rules (in this order) and push again:

```json
{
  "source": "/:path*",
  "has": [
    { "type": "host", "value": "linkage.colesorkness.com" },
    { "type": "query", "key": "m", "value": "(?<m>.*)" }
  ],
  "destination": "https://colesorkness.com/linkage/:path*?m=:m",
  "permanent": true
},
{
  "source": "/:path*",
  "has": [{ "type": "host", "value": "linkage.colesorkness.com" }],
  "destination": "https://colesorkness.com/linkage/:path*",
  "permanent": true
}
```

(and the same capture on the `/magcoupling/:path*` rule), with `tests/web_layout.rs` updated to match.

## Self-review record

- Spec coverage: the paths and the hub (Task 1), old links (Task 1's redirects, Task 5's checks), the standalone calculator reachable from the linkage tool (Task 2), docs (Task 3).
- The dry run is the implementation: every check above was run on the branch; the gate result and the review are recorded in the session's ledger and below.
