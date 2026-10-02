# Magcoupling M5: Embed the Calculator in the Linkage App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tools → Magnetic coupling in the linkage app (native and web) opens the calculator's existing panel (`magcoupling::gui::MagcouplingPanel`) in an `egui::Window` whose state is independent of the linkage model and survives closing and reopening, with the keyboard going to whichever part the user pressed last, the panel's file requests done by the linkage app, the workbook-parity guard extended to the linkage app's shipped builds, and backlog BL-037 (the window title sent every frame) fixed on the way.

**Architecture:** `linkage-sim-rs` depends on `magcoupling-rs` by path with feature `gui` (egui and egui_plot only; one egui, 0.32.3, in both lock files). One new module, `linkage-sim-rs/src/gui/calculator_window.rs`, owns everything the embedding needs: `CalculatorWindow` holds the panel (created on the first opening, then kept), draws the window, decides which part has the keyboard from the frame's last pointer press (the window's layer and the band just outside its frame where egui resizes it give it, a Foreground layer such as an open menu's items or a drop-down list leaves it, anything else, the menu bar's buttons included, takes it), turns the panel's undo keys on only while it has the keyboard and then removes the frame's key events but egui's own zoom keys so no linkage key reader runs on them, skips the window on a screen without room for its frame (a hidden browser canvas), and does the panel's `PanelRequest`s (saving through the linkage app's `export::download`, picking through rfd natively and on the web). `LinkageApp::update` calls `CalculatorWindow::show` before anything else reads the keyboard (the old inline shortcut block becomes `handle_keyboard_shortcuts`; a source test pins the order); the menu bar gets a Tools menu; the web entry opens the window for `?tool=magcoupling`. The scripts define the linkage app's shipped cargo arguments once (`magcoupling_shipped.sh`), gate 10 runs the guard on them with a negative control per shipped build, and `deploy-web.yml` builds through `build_web.sh`, which pipes the linkage bundle through the guard.

**Tech Stack:** Rust 2024 (rustc 1.89, the CI pin); eframe/egui 0.32.3, egui_plot 0.33.0, rfd 0.15.4 (now also in the linkage wasm build); headless egui tests (`egui::Context::run` with injected input, painted shapes inspected); bash scripts (Git Bash); Playwright MCP or the `gui-smoke` workflow for the web check.

**Spec:** `C:/Users/Cole/source/repos/lsim-mag-m5/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, sections Structure ("`linkage-sim-rs` depends on it by path with feature `gui` and opens the panel from a **Tools** menu"; "No Cargo workspace conversion") and M5 ("**Tools → Magnetic coupling** opens the same panel in an `egui::Window`, with state independent of the linkage model. The standalone page stays at `/magcoupling/`. `build_web.sh` and `deploy-web.yml` build and ship both bundles." The second half was done by the M4 infrastructure; this plan keeps it and routes the deploy through `build_web.sh`). Out of scope (spec): "Any coupling between the magcoupling panel and the linkage model's data."

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-m5`, branch `magcoupling/m5`, created by Task 0 from `main` at `4c95255` (the merge of M4-3) with LF line endings and a worktree-scoped `core.autocrlf=false`. Every command uses absolute paths into it. Nothing is pushed.

**Process (the user's lean process):** six tasks (0 to 5): Tasks 1 to 4 give exact code, so the implementers transcribe, and Task 5 is the user's check by hand; every implementation runs on `sonnet`, and so does every per-task review but Task 2's, which runs on the session model (the keyboard routing and focus semantics are judgment); no pre-flight scan, so every block below quotes `main` at `4c95255` exactly (Task 0 Step 2 checks the files) and was replayed literally from this file (Verification record); the whole-branch review after Task 4 runs on the session model, and Task 5 follows it and its fixes, before the merge.

## Decisions to confirm

**Confirmed (user, 2026-10-02): the recommended option on all of M5-1 to M5-9.** No task stops on a decision.

These UX and engineering choices arose while turning the spec into code; no approved decision settles them. The plan implements the recommended option of each (the code comments and docs cite the ids). Task 0 Step 9 records the user's answers; if the user picks another option, the task that implements it stops and escalates instead of improvising.

| Id | Question | Recommended (implemented here) | Alternatives | Tasks |
|---|---|---|---|---|
| M5-1 | Where does the menu item go? | **A new Tools menu, the last one (after Image), with one item: a "Magnetic coupling" checkbox** that shows whether the window is open, toggles it and closes the menu (the spec names Tools; the View menu's toggles are checkboxes too) | (b) an item under View; (c) a toolbar button | 2 |
| M5-2 | Window size and place | **Contents 1100 x 700 points at (80, 100)**, below the menu bar and both toolbars of the 1400 x 900 native window; movable, resizable, collapsible, closable with its X; kept on the screen (a smaller browser window shrinks it: egui keeps a window's contents on the screen and this plan also its frame) | (b) nearly full screen; (c) 900 x 600 (the panel's three columns get cramped below about 1000 points) | 2 |
| M5-3 | Open at start-up, and how long does its state live? | **Closed at start-up.** The first opening creates the panel (the equation registry, about 3 ms, is built then, not at every linkage start-up); the panel keeps its design, undo history and views, open or closed, until the app quits; nothing is saved across restarts | (b) open at start-up; (c) remember open/closed in the linkage prefs | 2 |
| M5-4 | URL parameter | **`?tool=magcoupling`** opens the window when the web app starts (with `?m=` too); gui-smoke's linkage step uses it, so the smoke opens the window without clicking by coordinates. Share links made in the window open `/magcoupling/` on the same server. Known edge, documented (backlog BL-039): with an autosave in the browser the linkage app's "Recover Unsaved Work?" prompt shows at start too, and the raised window covers it (natively the same when the window is opened while the prompt shows); the user moves or collapses the window to answer it | (b) no parameter (the smoke cannot open the window without a coordinate click); (c) `#magcoupling`; (d) as recommended, and keep the prompt in front (both prompts on Foreground, or the window opened only once the prompt is answered: a change inside `update`, which headless tests never drive) | 3 |
| M5-5 | Keyboard: which part gets the keys? | **The part the user pressed last.** The window has the keyboard from its opening or a press on it (its frame, and the band just outside it where egui lets the user grab its edge to resize it) until a press on the linkage app: its panels, its canvas, its menu bar's buttons (File, Edit, View, Image, Tools: a Background panel) or another window. A press on an open menu's items or a drop-down list (Foreground layers) moves nothing. While it has the keyboard the panel's Ctrl+Z, Ctrl+Shift+Z, Ctrl+Y are on and the window removes the frame's key events but egui's own zoom keys (natively Ctrl+Plus, Ctrl+Minus and Ctrl+0 still zoom the whole UI; on the web eframe leaves them to the browser, which zooms the page, as before), so no linkage shortcut (undo, redo, save, new, paste hint, delete) and no canvas key (arrow nudge, F, Escape, Enter) runs; otherwise (and while the window is collapsed to its title bar, when no panel is drawn) the panel's undo keys are off and the linkage app reads the keys as before. Known edge, documented (backlog BL-040): the keys follow presses, not egui's keyboard focus, so after Tab moves the focus from one part into the other they stay where they were until a click there | (b) only Ctrl+Z / Ctrl+Y (the M4-1 note's minimum): an arrow key in a focused calculator slider would still nudge a selected link (an undo step), Delete would delete it, Ctrl+N would clear the model; (c) keys follow the hovered part; (d) as recommended, and the keyboard also follows egui's focus (Tab included: a further rule, BL-040's sketch) | 2 |
| M5-6 | Who saves and loads the calculator's files? | **The linkage app, with its own helpers where it has them:** saving through `export::download::download_text` (a file dialog natively, a browser download on the web); loading through a small `DesignPicker` on rfd (the pattern of `magcoupling-rs/src/app/files.rs`), natively and on the web, which adds rfd 0.15.4 (already in the lock file) to the linkage wasm build | (b) move `app/files.rs` behind a new magcoupling feature and use it from both hosts (one implementation; changes magcoupling's "gui = egui only" contract); (c) on the web, no picker: Load design asks for drag and drop | 2 |
| M5-7 | Fix BL-037 here? | **Yes:** it is in the same `update`. The window title is sent only when it changes, and never on wasm32 (the web backend ignores it and logged one warning per frame) | (b) leave it to the fix campaign | 3 |
| M5-8 | How does the shipped linkage bundle get the guard? | **`deploy-web.yml` builds both bundles through `build_web.sh`,** which pipes the linkage release build through the workbook-parity guard (as `build_magcoupling_web.sh` already does), so the guard sees the artifact that ships and the build arguments live once (`magcoupling_shipped.sh`). The deploy path itself runs only on a push of `main` | (b) keep deploy's inline linkage build (gate 10 checks the same arguments, but nothing ties deploy's copy to them) | 1 |
| M5-9 | A magcoupling design file dropped on the linkage page | **Stays a mechanism import** (the status bar says the JSON import failed); Load design in the window loads it. Documented as a known limitation | (b) route a dropped `.json` to the calculator while its window has the keyboard (`update`'s drop handling is not headless-testable: see Global Constraints) | 4 |
| M5-10 | The linkage web bundle's size | **Accept it:** every load of the linkage web app downloads the panel and rfd, whether or not the user opens the calculator: `linkage-web_bg.wasm` grows from 10,529,415 to 11,991,219 bytes (+1,461,804, +13.9 %). One build, the same window on the desktop and the web | (b) feature-gate the panel out of the linkage web build and make Tools → Magnetic coupling there a link to `/magcoupling/` (the old bundle size; two behaviours to document and test, and `?tool=magcoupling` becomes a redirect) | 1, 4 |

## Global Constraints

Every task's requirements implicitly include this section.

- Dependency: `linkage-sim-rs/Cargo.toml` gains `magcoupling-rs = { path = "../magcoupling-rs", features = ["gui"] }` and, for wasm32 only, `rfd = "0.15"`. Never `workbook-parity` in a shipped build (gate 10 now proves it for `linkage-gui` and `linkage-web` too, with a negative control per shipped build). One egui: `cargo tree -i egui --target all -e normal` resolves to `egui v0.32.3`, and gate 11 (lock parity) passes. The linkage `Cargo.lock` gains only the `magcoupling-rs` package entry and its line in `linkage-sim-rs`'s dependency list (13 lines). No Cargo workspace (spec Structure).
- magcoupling-rs does not change except its README: no file under `magcoupling-rs/src/`, `magcoupling-rs/tests/`, `magcoupling-rs/Cargo.*` or `reference/` is edited. Its tests keep Task 0's counts; its `cargo fmt --check` stays clean.
- Feature unification: magcoupling's `serde_json` has `float_roundtrip`, which now applies to the linkage crate's JSON parsing too (exact float parsing). Every linkage test passes with it (checked in the replay).
- State independence: the calculator's state lives in `LinkageApp` (`calculator_window::CalculatorWindow`), never in `AppState`; nothing in `calculator_window.rs` reads or writes the linkage model (spec M5 and Out of scope).
- Keyboard order: `LinkageApp::update` calls `self.calculator.show(ctx)` before every reader of the keys (`handle_keyboard_shortcuts`, `menu_bar::draw_menu_bar`, `handle_delete_shortcut`, `canvas::draw_canvas`, any `key_pressed`); `gui::tests::update_shows_the_calculator_window_before_anything_reads_the_keyboard` checks it.
- Headless tests never drive `LinkageApp::update`: `canvas::draw_canvas` writes `grid.spacing_m` every frame and `AppState::tick_save_user_prefs` would then write `~/.linkage-sim/preferences.json` on the developer's machine. Tests drive the pieces `update` calls, in its order (`keys_frame`), and pin the order with the source test.
- Formatting: `linkage-sim-rs` is not rustfmt-clean (about 1,600 diffs at `4c95255`), so **never run `cargo fmt` in `linkage-sim-rs`**. The new file is in rustfmt's layout: `rustfmt --check --edition 2024 C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/src/gui/calculator_window.rs` prints nothing. Edits to existing linkage files follow each file's own style. `cargo fmt --check` for `magcoupling-rs` stays clean (unchanged crate).
- Lints: `cargo clippy --all-targets` in `linkage-sim-rs` (gate 2; warnings allowed) keeps `4c95255`'s counts exactly: `(lib) generated 279 warnings`, `(lib test) generated 295 warnings (268 duplicates)`, `(test "property_tests") generated 3 warnings`; its wasm32 clippy of `linkage-web` keeps `(lib) generated 268 warnings`. No new warning in any changed file.
- Gate runs (Task 0 and every task before its commit): `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate` line; then `git -C C:/Users/Cole/source/repos/lsim-mag-m5 checkout -- docs/chebyshev_lambda` (the linkage tests rewrite those PNGs; never commit them).
- Blocks: they quote the files with LF line endings, as git stores them (this machine's system gitconfig sets `core.autocrlf=true`; Task 0 makes the worktree LF). "Create `path`:" writes a new file with the block's text. "In `path`, replace: ... with: ..." is one exact replacement: the old block occurs exactly once in the file at that point, as whole lines. Apply a step's blocks in the order given. If an old block is not found, stop and escalate; never improvise a match.
- Tests that read source files normalize CRLF (`.replace("\r\n", "\n")`): the order test reads `gui/mod.rs` (CRLF in the main checkout); the smoke test reads `.claude/workflows/gui-smoke.js` (LF by `.gitattributes`; normalized anyway).
- Windows paths: the worktree path is short on purpose (a release build under a deep directory fails with `LNK1104`, MAX_PATH).
- Commits: subjects `feat(linkage-sim-rs): ...` (Tasks 1 to 3) and `docs(magcoupling): ...` (Tasks 4 and 5). Every commit message ends with exactly two lines: a `Co-Authored-By:` line naming the model that writes the commit, then `Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9`. The blocks write `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the plan's writer); a `sonnet` implementer writes its own model's name there. Nothing is pushed.
- Docs change with code (repo rule, `docs/ai/01-meta.yaml`; the user's rule): Tasks 1 to 3 are intermediate commits on `magcoupling/m5`; the branch is reviewed as a whole after Task 4, which updates `README.md`, `magcoupling-rs/README.md`, `docs/FEATURES.md`, `docs/guides/WASM_DEPLOYMENT.md`, `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`, `backlog.yaml` (new YAML scalars holding `: ` are quoted), and commits this plan; it is merged only after Task 5, the user's sign-off (the spec's M5 gate: "Tests + smoke + user check"). Read `docs/ai/*.yaml` first (Task 0).
- Deploy: `deploy-web.yml` on `main` builds both bundles through `build_web.sh` from the next push of `main` on (decision M5-8); this plan pushes nothing and cannot run that job; Task 4 runs the script locally.
- Model tiers (CLAUDE.md section 5): Tasks 1 to 4 give the exact code, so their implementations and per-task reviews run on `sonnet` (`model: 'sonnet'`), except Task 2's per-task review, which runs on the session model (omit `model`, comment `// session model: keyboard routing judgment`): the code is exact, but judging its keyboard routing and focus semantics is not transcription, and section 5 says "When unsure, use the session model". A `sonnet` attempt that ends `blocked`, leaves a check red or is rejected in review escalates every retry to the session model. The whole-branch review after Task 4 runs on the session model (omit `model`, comment `// session model: final whole-branch review`). Task 5 is the user's, run with the controller. No task is physics: the engine is untouched.

## Review Focus

The conditions the spec implies but does not name that are most likely to bite a user, in five lines, each pinned by tests in the task that owns the code (two more edges are documented instead of handled: Tab focus moving between the parts, decision M5-5, and the window covering the recovery prompt, decision M5-4):

1. **A browser window smaller than the calculator window, down to none** (an 800 x 600 page against 1100 x 700 contents; a hidden canvas reports 0 x 0). Expected: the window, frame included, stays on the screen and still shows the panel, and after a 0 x 0 frame it comes back in its place at its size. egui 0.32 keeps only a window's contents on screen and let the frame's 14-point margins hang off the right and bottom edges, so the window is constrained to the screen less those margins; and egui stores the size it squeezes a window to (one 0 x 0 frame left it at the screen's corner, 609 x 900 points, for good), so on a screen without room for its frame the window skips that frame. Tests: `a_small_screen_keeps_the_window_on_it`, `a_screen_without_room_skips_the_window_and_keeps_its_place_and_size` (Task 2).
2. **Starting with `?tool=magcoupling` on an empty workspace.** Expected: the window covers the linkage welcome screen, not the reverse. egui brings every area that first shows in a frame to the front, the welcome screen (drawn after the window) included, so the window brings itself in front once more on its second frame. Test: `an_opened_window_is_in_front_of_what_the_app_draws_after_it` (Task 2).
3. **Closing the window while one of its text fields has the keyboard focus.** Expected: the linkage app's keys work again at once (Delete deletes, Ctrl+Z undoes the model). Test: `closing_the_window_with_a_focused_field_gives_the_keys_back` (Task 2).
4. **Where a press lands:** an open menu's items or a drop-down list (the calculator's combo lists, the linkage menus once open), the menu bar's buttons (the Tools button itself), the band just outside the window's frame where egui resizes it, another window over that band, **and taps whose press and release fall in one frame** (touch, slow frames). Expected: a popup press moves no keyboard; a menu-bar press gives it to the linkage app (the menu bar is a Background panel); a press on the band gives it to the window unless another window covers that point; a one-frame tap counts. Tests (Task 2): `the_keyboard_follows_presses_and_popups_leave_it_alone` (a Foreground area stands in for a popup; its last part sends a press and its release in one frame), `a_press_on_the_band_where_egui_resizes_the_window_gives_it_the_keyboard` (5 points from an edge, 10 around a corner, not 8 from an edge), `another_window_over_the_band_takes_the_keyboard`, `a_press_on_the_menu_bar_gives_the_keyboard_to_the_linkage_app` (a real menu-bar button, Tools) and `tools_magnetic_coupling_opens_and_closes_the_calculator_window` (opening from the menu leaves the window with the keyboard).
5. **Undo across the two parts.** Expected: one Ctrl+Z or Ctrl+Y never acts on both; an arrow key, Delete or Escape in the calculator never edits the model. Tests: `ctrl_z_undoes_only_the_part_that_has_the_keyboard`, `arrow_keys_and_delete_leave_the_model_alone_while_the_calculator_has_the_keyboard` (Task 2, with positive controls that the same keys reach the model once it has the keyboard), `with_the_keyboard_the_panel_undoes_and_the_host_sees_no_key`, `without_the_keyboard_the_panel_leaves_the_keys_to_the_host`, `a_collapsed_window_leaves_the_keys_to_the_host` (Task 2: a window collapsed to its title bar draws no panel, so it must not swallow the keys), and `with_the_keyboard_egui_s_zoom_keys_still_zoom_the_ui` (Task 2: Ctrl+Plus, Ctrl+Equals, Ctrl+Minus and Ctrl+0 still reach egui's zoom of the whole UI while the window takes every other key, bare Minus included. egui zooms with them natively; eframe's web backend turns that off and the browser zooms the page).

## File Structure

Created (under `C:/Users/Cole/source/repos/lsim-mag-m5/`):

| File | Responsibility | Task |
|---|---|---|
| `linkage-sim-rs/src/gui/calculator_window.rs` | Tools → Magnetic coupling: `CalculatorWindow` (`set_open`, `is_open`, `has_keyboard`, `show`; Task 3: `set_share_base`), the keyboard rule (`on_resize_band`, `UI_ZOOM_KEYS`, `is_ui_zoom_key`), the panel's requests (`save_filter`, `report_of`, `open_picked_file`, `DesignPicker`), `TITLE`; test-only `panel`, `design`, `load_design`, `default_rect`, `beside_the_window`; Task 3: `TOOL_PARAM`, `TOOL_MAGCOUPLING`, `MAGCOUPLING_PAGE_PATH`, `magcoupling_share_base`; headless tests | 2, 3 |

Modified:

| File | Change | Task |
|---|---|---|
| `linkage-sim-rs/Cargo.toml`, `Cargo.lock` | the `magcoupling-rs` dependency (feature `gui`); rfd for wasm32 | 1 |
| `linkage-sim-rs/scripts/magcoupling_shipped.sh` | `LINKAGE_WEB_ARGS`, `LINKAGE_NATIVE_ARGS` | 1 |
| `linkage-sim-rs/scripts/gate.sh` | gate 3 from `LINKAGE_WEB_ARGS`; gate 10 on the linkage builds; `assert_guard_trips` (a negative control per shipped build) | 1 |
| `linkage-sim-rs/scripts/build_web.sh` | the linkage release build through the guard | 1 |
| `linkage-sim-rs/scripts/build_magcoupling_web.sh` | header: deploy-web.yml runs it through `build_web.sh` | 1 |
| `.github/workflows/deploy-web.yml` | one build step: `bash scripts/build_web.sh` | 1 |
| `linkage-sim-rs/src/gui/mod.rs` | `mod calculator_window`; `LinkageApp::calculator`; `show` before the keys; `handle_keyboard_shortcuts` (the old inline block); the menu call; a Keyboard Shortcuts row; tests (Task 2). Task 3: `pub use` of the constants, `open_magcoupling`, `set_magcoupling_share_base`, `sent_title`, `window_title`, `send_title` (BL-037) and its test | 2, 3 |
| `linkage-sim-rs/src/gui/menu_bar.rs` | the Tools menu; `draw_menu_bar` takes the window; its first tests (the menu toggles the window; a menu-bar press takes the keyboard) | 2 |
| `linkage-sim-rs/src/gui/test_support.rs` | `key_tap`, `click_events`, `NATIVE_SCREEN`, `screen_input`, `magcoupling_gap_design` | 2 |
| `linkage-sim-rs/src/bin/linkage_web.rs` | `?tool=magcoupling`, the share base, `url_param` | 3 |
| `.claude/workflows/gui-smoke.js` | the linkage step opens the window; `LINKAGE_TOOL_QUERY`; title warning check | 3 |
| `README.md`, `magcoupling-rs/README.md`, `docs/FEATURES.md`, `docs/guides/WASM_DEPLOYMENT.md`, `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`, `backlog.yaml` | docs; BL-037 fixed; BL-038 to BL-040 filed | 4 |
| `docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md` | this plan | 4 |
| `docs/ai/04-memory.yaml` | the user's sign-off | 5 |

Order: the dependency and its guard first (a build-level change a reviewer can judge alone), then the window with its keyboard rule and menu (the feature, test-first), then the web entry, the title fix and the smoke, then the docs and the shipped-bundle check, then (after the whole-branch review) the user's check.

## Verification record

This plan was replayed before it was handed over, twice: once as first written and once after the revision that answers the critic's review (the second self-review record). The blocks were developed in a scratch tree (an LF export of `4c95255`, `git -c core.autocrlf=false archive 4c95255`), kept in one specification from which this file's blocks are generated (each replace block a whole-line hunk whose old text is unique when it is applied), then **parsed back out of this file and applied literally, task by task, to a fresh LF export of `4c95255`** (a scratch git repository with one commit per task, reset to its base commit before the second replay), running each step's commands as written with the worktree path replaced by the scratch path (`V:/verify`, a `subst` drive, for MAX_PATH). The blocks replayed are identical to this file's, checked by parsing both; only prose (the counts it now states and this record) changed after the replay. The cargo target directories were carried over (only the two path crates rebuilt). At the end the replayed `calculator_window.rs` equals the specification's file byte for byte. The numbers below are the second replay's unless marked.

- **Task 0** (first replay; the revision changes no file of `4c95255`): the baseline counts of Step 5 (linkage library `926 passed`, every integration binary as listed; magcoupling with `app` `449 passed`); the gate passed (`1159 passed`, the differential data current, `GATE PASS`, no `SKIP gate`); `cargo fmt --check` clean for magcoupling-rs; Step 8: `10529415` and `4996155` bytes.
- **Red steps failed as stated:** Task 1 Step 2 (`cargo compiled no magcoupling-rs unit for linkage-gui`, `exit=1`); Task 2 Step 2 (`135 previous errors; 23 warnings emitted`, E0061 twice); Task 3 Step 2 (`7 previous errors; 22 warnings emitted`) and Step 4 (`20 passed; 1 failed`, the smoke query missing). **Green steps passed** with the counts in the table.
- **After every task** the gate passed (`GATE PASS`, no `SKIP gate`; about 3 minutes a run on the warm cache): gate 10 printed the four `builds without workbook-parity` lines and the four negative controls (`magcoupling-app`, `magcoupling-web`, `linkage-gui`, `linkage-web`) from Task 1 on, gate 11 the four lock pairs and the CLI pin; `cargo clippy --all-targets` kept `279` / `295 (268 duplicates)` / `3` warnings and the wasm32 clippy of `linkage-web` `268`, none in a changed file; `rustfmt --check` passed on `calculator_window.rs`; the linkage `Cargo.lock` gained 13 lines; `cargo tree -i egui` resolved to `egui v0.32.3`.
- **Task 4:** the YAML checks (`02-system entries ok: 12`, the other three files whole) and `git diff --check`; `build_web.sh` through both guards: `linkage-web_bg.wasm` **10,529,415 to 11,991,219 bytes (+1,461,804, +13.9 %)** (the first replay measured 11,986,020, before the revision's band, zoom and screen code), `magcoupling-web_bg.wasm` 4,996,155 bytes unchanged. Served with `serve_web.sh` and opened with Playwright at 1400 x 900 (a build of the same 11,991,219 bytes, made before the revision's last edits, which touched comments and prose only): `/` and `/?tool=magcoupling` logged zero errors and zero warnings (no `Unhandled egui viewport command: Title`, which `/` logged once a frame before); the second page logged `magcoupling explorer: 392 equations` and showed the window at (80, 100) in front of the welcome screen with the calculator inside (inputs, geometry view, dashboard). Ctrl+= after a click in the window did not zoom that page: eframe's web backend turns egui's keyboard zoom off and leaves the shortcut to the browser (a headless browser does not zoom), hence "natively" wherever the docs mention the zoom keys. By hand in the first replay's page: the window's X closed it and Tools → Magnetic coupling reopened it; its triangle collapsed and expanded it; "Load design" opened the browser's file chooser, and a design file with a 2 mm face gap loaded ("Design file loaded", 2.00 mm); after a click in the window Ctrl+Z undid the load and Ctrl+Y redid it, while after a click on the linkage side panel Ctrl+Y left the calculator alone. Seen there: after the file chooser closes, the page's canvas needs a click before keys reach egui (the browser's focus; Task 4 Step 4 says so). The server's Python process outlived TaskStop both times and was killed by PID (Task 4 Step 4 says so).
- **Measured for finding 4** (a 0 x 0 screen, one frame between normal ones): without a guard the window moved from `(80, 100)-(1251.9, 847.7)` to `(0, 0)-(609.1, 900)` for good; with the clamp `(screen.size() - margins).max(Vec2::ZERO)` exactly the same; with the guard it came back at `(80, 100)-(1251.9, 847.7)`. The window spans about 80..1252 x 100..848 at its default size, wider and taller than its 1100 x 700 contents.
- **Mutation checks** (the second replay's tree, each mutation restored after; every one failed the named tests): the zoom keys removed with the rest (`with_the_keyboard_egui_s_zoom_keys_still_zoom_the_ui`); zoom keys matched without their modifier (`with_the_keyboard_the_panel_undoes_and_the_host_sees_no_key`, bare Minus); no resize band, or no corner squares (`a_press_on_the_band_where_egui_resizes_the_window_gives_it_the_keyboard`); the band counted over another window (`another_window_over_the_band_takes_the_keyboard`); no screen guard, or the critic's clamp in its place (`a_screen_without_room_skips_the_window_and_keeps_its_place_and_size`); the menu bar's Background layer treated as a popup (`a_press_on_the_menu_bar_gives_the_keyboard_to_the_linkage_app`); gui-smoke's linkage sentence without the registry check (`gui_smoke_opens_the_window_with_the_tool_parameter`). The first replay's checks, re-run on the revised code: no key-event removal (`with_the_keyboard_the_panel_undoes_and_the_host_sees_no_key` and `arrow_keys_and_delete_leave_the_model_alone_while_the_calculator_has_the_keyboard`; `ctrl_z_undoes_only_the_part_that_has_the_keyboard` still passes, because the panel consumes Ctrl+Z and Ctrl+Y itself); no Foreground rule (`the_keyboard_follows_presses_and_popups_leave_it_alone`); the panel's shortcuts always on (`without_the_keyboard_the_panel_leaves_the_keys_to_the_host`); raised on the first frame (`an_opened_window_is_in_front_of_what_the_app_draws_after_it`); no screen constraint (`a_small_screen_keeps_the_window_on_it`); a collapsed window counted as drawn (`a_collapsed_window_leaves_the_keys_to_the_host`); the window shown after the readers in `keys_frame` (both keyboard tests in `gui/mod.rs`) and in `update` (the order test).

| After task | Linkage library (`cargo test --lib`) | Red step | Clippy (lib / lib test / property_tests; wasm32) |
|---|---|---|---|
| 0 (base) | 926 | — | 279 / 295 / 3; 268 |
| 1 | 926 | guard: no magcoupling-rs unit | 279 / 295 / 3; 268 |
| 2 | 949 (window 18, menu 2, `gui::tests` 12) | 135 errors | 279 / 295 / 3; 268 |
| 3 | 953 (window 21, `gui::tests` 13) | 7 errors; then 20 passed, 1 failed | 279 / 295 / 3; 268 |
| 4 | 953 | — | 279 / 295 / 3; 268 |
| 5 | 953 | — (the sign-off block applied; `04-memory ok`) | — |

Every linkage integration test binary keeps Task 0's counts throughout; magcoupling-rs keeps `449 passed` with `app` and every engine count (gate 4, gate 7).

Not exercised by the replay: the `deploy-web.yml` job itself (it runs only on a push of `main`; its new step runs the `build_web.sh` replayed above, on Ubuntu with the same toolchain pin), a native `linkage-gui` window by hand (Task 5 puts it in front of the user; the headless tests drive the same window code), the native file dialogs (rfd's synchronous dialogs; `save_filter`, `report_of`, `open_picked_file` and the picker's inbox are tested), the `gui-smoke` workflow as a workflow (its linkage step's checks were done by hand above; its script compiles and its query and registry check are pinned by a test), Task 5's checklist (the user's; its sign-off block was applied and the file parsed), and the user's answers to the Decisions to confirm.

---

## Tasks

### Task 0: Worktree, baseline and context

**Model:** `sonnet` (creating the worktree, running commands and reporting output; CLAUDE.md section 5). Step 9 is the controller's.

Verification only; nothing to write.

**Files:**
- none

**Interfaces:**
- Consumes: `main` at `4c95255`, or a later commit that leaves the files this plan's blocks touch unchanged (Step 2).
- Produces: the worktree `C:/Users/Cole/source/repos/lsim-mag-m5` on the new branch `magcoupling/m5` with LF line endings, a green baseline with its test counts and the linkage wasm size before the change, and the controller's record of the user's answers to the Decisions to confirm.

- [ ] **Step 1: Create the worktree**

Run:

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation log --oneline -1 main
git -C C:/Users/Cole/source/repos/linkage_simulation config extensions.worktreeConfig true
git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b magcoupling/m5 C:/Users/Cole/source/repos/lsim-mag-m5 main
git -C C:/Users/Cole/source/repos/lsim-mag-m5 config --worktree core.autocrlf false
git -C C:/Users/Cole/source/repos/lsim-mag-m5 rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-m5 status --short
```

Expected: the `log` line is `4c95255 merge: magcoupling M4-3 GUI (equation typesetter and explorer with coloured terms and drill-down, assumptions view and banner, teaching notes with diagrams, material/part/grade pickers, warnings linked to notes)` or a later commit; `worktree add` prints `Preparing worktree (new branch 'magcoupling/m5')`; then `magcoupling/m5`; no status lines (the main checkout's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md` is not in the worktree).

- [ ] **Step 2: Check that the blocks' files are still those of `4c95255`**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m5 diff --stat 4c95255 HEAD -- linkage-sim-rs magcoupling-rs README.md docs/ai docs/FEATURES.md docs/guides/WASM_DEPLOYMENT.md .claude/workflows/gui-smoke.js .github/workflows/deploy-web.yml
```

Expected: no output. If any file is listed, stop and escalate: the blocks that touch it must be re-derived first.

- [ ] **Step 3: Check the line endings**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m5 config --get core.autocrlf; head -c 3000 C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/src/gui/mod.rs | tr -cd '\r' | wc -c
```

Expected: `false` and `0` carriage returns. If the count is not 0: run `git -C C:/Users/Cole/source/repos/lsim-mag-m5 rm -rq --cached . && git -C C:/Users/Cole/source/repos/lsim-mag-m5 reset -q --hard` and check again.

- [ ] **Step 4: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-mag-m5/docs/ai/01-meta.yaml`, `02-system.yaml`, `03-structure.yaml`, `04-memory.yaml` (the item this plan resolves: "Open (M5, from the M4-1 review)"), `backlog.yaml` (BL-037); `README.md` ("Rust solver kernel"); `magcoupling-rs/README.md` ("Features and binaries", "The panel", "workbook-parity never ships"); the spec's Structure and M5 sections; the panel's host API in `magcoupling-rs/src/gui/panel.rs` (`MagcouplingPanel::new`, `ui`, `take_requests`, `PanelRequest`, `load_design_file`, `report`, `set_share_base`, `set_keyboard_shortcuts`, `shortcuts`, `focused_text_field`) and its test `a_host_can_keep_ctrl_z_for_itself`; `magcoupling-rs/src/app.rs` and `app/files.rs` (the standalone host); the linkage `src/gui/mod.rs` (`update`), `menu_bar.rs`, `export/download.rs`, `canvas/interaction.rs` (its unguarded arrow, F, Escape and Enter keys), `test_support.rs`; `linkage-sim-rs/scripts/gate.sh`, `magcoupling_shipped.sh`, `build_web.sh`, `build_magcoupling_web.sh`; `.github/workflows/deploy-web.yml`; `docs/FEATURES.md`, `docs/guides/WASM_DEPLOYMENT.md`; and this plan's Decisions to confirm.

- [ ] **Step 5: Run both crates' tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --all 2>&1 | grep -E "Running|test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
git -C C:/Users/Cole/source/repos/lsim-mag-m5 checkout -- docs/chebyshev_lambda
```

Expected: the linkage library `test result: ok. 926 passed`; the binaries `linkage_gui` and `linkage_web` 0; `tests\actuator_force_label.rs` 4; `braindump_repro` 1; `compound_actuator_rebuild` 5; `compound_force_integration` 9; `dxf_import_test` 1; `force_zone_tests` 8; `geometry_tests` 18; `golden_fixtures` 11; `gravity_breakdown_reference` 2; `mount_point_integration` 1; `parallelogram_actuator_sample` 4; `property_tests` 8; `singular_behavior` 18; doc-tests 0. Then magcoupling with `app`: `test result: ok. 449 passed`. If a count differs but every binary is `ok`, record the actual counts and read every later count as an offset from them.

- [ ] **Step 6: Check the oracle Python and the CLI**

Run:

```bash
ls C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe
wasm-bindgen --version
rustc --version
```

Expected: the path, then `wasm-bindgen 0.2.114`, then `rustc 1.89.0 (29483883e 2025-08-04)`. If the Python is missing, stop and escalate: every gate run uses it.

- [ ] **Step 7: Run the gate**

Run:

```bash
mkdir -p C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task0.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task0.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task0.log
git -C C:/Users/Cole/source/repos/lsim-mag-m5 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0`; `cargo fmt --check` prints nothing.

- [ ] **Step 8: Measure the linkage web bundle before the change**

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && bash scripts/build_web.sh 2>&1 | grep -E "parity guard|Build complete|complete!"; wc -c web/linkage-web_bg.wasm web/magcoupling/magcoupling-web_bg.wasm
```

Expected: `Build complete!`, `parity guard: magcoupling-web builds without workbook-parity; ...`, `Magcoupling build complete!`; then about `10529415 web/linkage-web_bg.wasm` and `4996155 web/magcoupling/magcoupling-web_bg.wasm` (record both exact counts in the execution notes; Task 4 reports the change against them). The outputs are gitignored: `git -C C:/Users/Cole/source/repos/lsim-mag-m5 status --short` still prints nothing.

- [ ] **Step 9: Decisions**

The controller asks the user the plan's **Decisions to confirm** (M5-1 to M5-10) before Task 1 and records the answers in the execution notes. Every task implements the recommended option and names the decision where it implements it; if the user picks another option, the task that implements it stops and escalates instead of improvising.

---

### Task 1: The dependency and the workbook-parity guard on the linkage builds

**Model:** `sonnet` (the plan gives the exact code: transcription plus running commands; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `linkage-sim-rs/scripts/magcoupling_shipped.sh` (header comment; after `MAGCOUPLING_NATIVE_ARGS`)
- Modify: `linkage-sim-rs/Cargo.toml` (`[dependencies]` after `uuid`; wasm32 dependencies after `js-sys`), `linkage-sim-rs/Cargo.lock` (by cargo)
- Modify: `linkage-sim-rs/scripts/gate.sh` (header comment; `assert_guard_trips`; gates 3 and 10)
- Modify: `linkage-sim-rs/scripts/build_web.sh` (sources `magcoupling_shipped.sh`; the guarded build)
- Modify: `linkage-sim-rs/scripts/build_magcoupling_web.sh` (header comment: who calls it)
- Modify: `.github/workflows/deploy-web.yml` (the two build steps become one)

**Interfaces:**
- Consumes: `magcoupling_assert_shipped BIN` (existing, `magcoupling_shipped.sh`: reads cargo's `--message-format=json` on stdin; fails unless `BIN` was compiled, a magcoupling-rs unit was, and none has `workbook-parity`).
- Produces: the bash arrays `LINKAGE_WEB_ARGS=(--bin linkage-web --target wasm32-unknown-unknown --no-default-features --features raster)` and `LINKAGE_NATIVE_ARGS=(--bin linkage-gui)` (sourced from `magcoupling_shipped.sh`; cargo runs from `linkage-sim-rs/`); the gate function `assert_guard_trips BIN FEATURE CARGO_ARGS...`; the crate `magcoupling` (lib name of `magcoupling-rs`, feature `gui`) available to `linkage-sim-rs` code as `magcoupling::gui::{MagcouplingPanel, PanelRequest, ...}`; `rfd` available on wasm32.

- [ ] **Step 1: Define the linkage app's shipped build arguments**

In `linkage-sim-rs/scripts/magcoupling_shipped.sh`, replace:

````bash
# Sourced, never run: the cargo arguments of the shipped magcoupling-rs builds,
# defined once, and the guard that no shipped build has the test-only
# workbook-parity feature. Used by build_magcoupling_web.sh (the web bundle
# that deploy-web.yml ships) and gate.sh.
````

with:

````bash
# Sourced, never run: the cargo arguments of the shipped builds that contain
# magcoupling-rs (the calculator's own app, and the linkage app with its panel
# in Tools -> Magnetic coupling), defined once, and the guard that no shipped
# build has the test-only workbook-parity feature. Used by build_web.sh and
# build_magcoupling_web.sh (the two web bundles; deploy-web.yml runs
# build_web.sh) and gate.sh.
````

In `linkage-sim-rs/scripts/magcoupling_shipped.sh`, replace:

````bash
# The native desktop app.
MAGCOUPLING_NATIVE_ARGS=(--features app --bin magcoupling-app)
````

with:

````bash
# The native desktop app.
MAGCOUPLING_NATIVE_ARGS=(--features app --bin magcoupling-app)
# The linkage app's builds (cargo run from linkage-sim-rs/), which depend on
# magcoupling-rs with its feature gui: the web bundle (linkage-sim-rs/web/,
# served at /) and the native desktop app.
LINKAGE_WEB_ARGS=(--bin linkage-web --target wasm32-unknown-unknown --no-default-features --features raster)
LINKAGE_NATIVE_ARGS=(--bin linkage-gui)
````

- [ ] **Step 2: Run the guard on the linkage native build to verify it fails**

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && source scripts/magcoupling_shipped.sh && cargo check "${LINKAGE_NATIVE_ARGS[@]}" --message-format=json-render-diagnostics 2>/dev/null | magcoupling_assert_shipped linkage-gui; echo "exit=$?"
```

Expected: `parity guard: cargo compiled no magcoupling-rs unit for linkage-gui; nothing was checked` and `exit=1` (the linkage app does not depend on the calculator yet; `$?` is the exit of the pipeline's last command, the guard, in Git Bash's default shell without `pipefail`).

- [ ] **Step 3: Add the dependency**

In `linkage-sim-rs/Cargo.toml`, replace:

````toml
dxf = "0.6"
uuid = { version = "1", features = ["js"] }
````

with:

````toml
dxf = "0.6"
uuid = { version = "1", features = ["js"] }
# The magnetic coupling calculator's panel, shown in Tools -> Magnetic coupling
# (gui::calculator_window): feature gui only (egui, egui_plot; no eframe, no
# I/O). Never its test-only workbook-parity feature: gate 10 checks every
# shipped build (scripts/magcoupling_shipped.sh).
magcoupling-rs = { path = "../magcoupling-rs", features = ["gui"] }
````

In `linkage-sim-rs/Cargo.toml`, replace:

````toml
js-sys = "0.3"

[dev-dependencies]
````

with:

````toml
js-sys = "0.3"
# The calculator window's "Load design" file picker on the web
# (gui::calculator_window::DesignPicker); natively rfd comes with feature native.
rfd = "0.15"

[dev-dependencies]
````

Then run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && cargo check 2>&1 | grep -E "^error|Finished"; git -C C:/Users/Cole/source/repos/lsim-mag-m5 diff --stat -- linkage-sim-rs/Cargo.lock
```

Expected: a `Finished` line and no `error`; ` linkage-sim-rs/Cargo.lock | 13 +++++++++++++` and ` 1 file changed, 13 insertions(+)` (cargo added the `magcoupling-rs` package, depending on `base64`, `egui`, `egui_plot`, `flate2`, `log`, `serde_json`, and its line in `linkage-sim-rs`'s list; rfd and its wasm dependencies were already locked).

- [ ] **Step 4: Run the guard to verify it passes, and check the negative control and the single egui**

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && source scripts/magcoupling_shipped.sh && cargo check "${LINKAGE_NATIVE_ARGS[@]}" --message-format=json-render-diagnostics 2>/dev/null | magcoupling_assert_shipped linkage-gui; cargo check "${LINKAGE_WEB_ARGS[@]}" --message-format=json-render-diagnostics 2>/dev/null | magcoupling_assert_shipped linkage-web
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && source scripts/magcoupling_shipped.sh && forced="$(cargo check "${LINKAGE_WEB_ARGS[@]}" --features magcoupling-rs/workbook-parity --message-format=json-render-diagnostics 2>/dev/null)"; magcoupling_assert_shipped linkage-web <<<"$forced"; echo "exit=$?"
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && cargo tree -i egui --target all -e normal 2>&1 | head -1
```

Expected: `parity guard: linkage-gui builds without workbook-parity; magcoupling-rs features: "features":["default","gui"]` and the same for `linkage-web`; then `parity guard: FAIL: linkage-web was built with the test-only workbook-parity feature`, `magcoupling-rs features: "features":["default","gui","workbook-parity"]` and `exit=1`; then `egui v0.32.3` (one egui: `cargo tree -i` stops with "multiple `egui` packages" if two versions were in the graph).

- [ ] **Step 5: Gate 3 and gate 10 on the linkage builds; build_web.sh and the deploy through the guard (decision M5-8)**

In `linkage-sim-rs/scripts/gate.sh`, replace:

````bash
# Gates 1-3: linkage-sim-rs. Gates 4-6: magcoupling-rs, a separate crate (not
# a workspace member), the engine alone (no features); its clippy runs with
# warnings as errors, as in every magcoupling-rs gate. Gates 7-11:
# magcoupling-rs with its gui and app features (the panel and the standalone
# app): their tests, clippy on native and wasm32, the guard that no shipped
# build has the test-only workbook-parity feature (with a negative control
# that proves the guard trips), and the check that both crates lock the same
# egui, egui_plot, eframe and wasm-bindgen (the CLI version deploy-web.yml
# installs).
````

with:

````bash
# Gates 1-3: linkage-sim-rs (which depends on magcoupling-rs with its feature
# gui: the calculator window, Tools -> Magnetic coupling). Gates 4-6:
# magcoupling-rs, a separate crate (not a workspace member), the engine alone
# (no features); its clippy runs with warnings as errors, as in every
# magcoupling-rs gate. Gates 7-11: magcoupling-rs with its gui and app
# features (the panel and the standalone app): their tests, clippy on native
# and wasm32, the guard that no shipped build (the calculator's own and the
# linkage app's, native and wasm32) has the test-only workbook-parity feature
# (with negative controls that prove the guard trips), and the check that both
# crates lock the same egui, egui_plot, eframe and wasm-bindgen (the CLI
# version deploy-web.yml installs).
````

In `linkage-sim-rs/scripts/gate.sh`, replace:

````bash
# The versions of package $2 in the Cargo.lock $1, one per line.
````

with:

````bash
# Fails the gate unless the workbook-parity guard trips on the shipped build of
# binary $1 with feature $2 forced on; the remaining arguments are the build's
# cargo check arguments. Cargo's output is captured first, so a build that
# fails to compile fails the gate here instead of passing as a trip.
assert_guard_trips() {
  local bin="$1" feature="$2" forced
  shift 2
  forced="$(cargo check "$@" --features "$feature" --message-format=json-render-diagnostics)"
  if magcoupling_assert_shipped "$bin" <<<"$forced" 2>/dev/null; then
    echo "FAIL gate 10/12: the guard did not trip on $bin with workbook-parity forced on"
    exit 1
  fi
  echo "negative control: the guard trips on $bin when workbook-parity is forced on"
}

# The versions of package $2 in the Cargo.lock $1, one per line.
````

In `linkage-sim-rs/scripts/gate.sh`, replace:

````bash
echo "== gate 3/12: WASM check (linkage-sim-rs) =="
cargo check --target wasm32-unknown-unknown --bin linkage-web --no-default-features --features raster
````

with:

````bash
echo "== gate 3/12: WASM check (linkage-sim-rs) =="
cargo check "${LINKAGE_WEB_ARGS[@]}"
````

In `linkage-sim-rs/scripts/gate.sh`, replace:

````bash
echo "== gate 10/12: workbook-parity guard (shipped native and wasm32 builds; negative control) =="
cargo check --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_NATIVE_ARGS[@]}" --message-format=json-render-diagnostics \
  | magcoupling_assert_shipped magcoupling-app
cargo check --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_WEB_ARGS[@]}" --message-format=json-render-diagnostics \
  | magcoupling_assert_shipped magcoupling-web
# Negative control: the same shipped build with the feature forced on must trip
# the guard, or the guard has stopped seeing what cargo builds. Cargo's output is
# captured first, so a build that fails to compile fails the gate here instead
# of passing as a trip.
FORCED_BUILD="$(cargo check --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_WEB_ARGS[@]}" --features workbook-parity \
  --message-format=json-render-diagnostics)"
if magcoupling_assert_shipped magcoupling-web <<<"$FORCED_BUILD" 2>/dev/null; then
  echo "FAIL gate 10/12: the guard did not trip on a shipped build with workbook-parity forced on"
  exit 1
fi
echo "negative control: the guard trips when workbook-parity is forced on"
````

with:

````bash
echo "== gate 10/12: workbook-parity guard (shipped native and wasm32 builds of both apps; negative controls) =="
cargo check --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_NATIVE_ARGS[@]}" --message-format=json-render-diagnostics \
  | magcoupling_assert_shipped magcoupling-app
cargo check --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_WEB_ARGS[@]}" --message-format=json-render-diagnostics \
  | magcoupling_assert_shipped magcoupling-web
# The linkage app's builds carry the calculator's panel (magcoupling-rs, feature gui).
cargo check "${LINKAGE_NATIVE_ARGS[@]}" --message-format=json-render-diagnostics \
  | magcoupling_assert_shipped linkage-gui
cargo check "${LINKAGE_WEB_ARGS[@]}" --message-format=json-render-diagnostics \
  | magcoupling_assert_shipped linkage-web
# Negative controls, one per shipped build: the same build with the feature forced
# on must trip the guard, or the guard has stopped seeing what cargo builds.
assert_guard_trips magcoupling-app workbook-parity --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_NATIVE_ARGS[@]}"
assert_guard_trips magcoupling-web workbook-parity --manifest-path "$MAGCOUPLING" "${MAGCOUPLING_WEB_ARGS[@]}"
assert_guard_trips linkage-gui magcoupling-rs/workbook-parity "${LINKAGE_NATIVE_ARGS[@]}"
assert_guard_trips linkage-web magcoupling-rs/workbook-parity "${LINKAGE_WEB_ARGS[@]}"
````

In `linkage-sim-rs/scripts/build_web.sh`, replace:

````bash
# Output: web/linkage-web.js + web/linkage-web_bg.wasm, and
#         web/magcoupling/magcoupling-web.js + magcoupling-web_bg.wasm
#
# After building, serve with: scripts/serve_web.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Building WASM binary (release)..."
cargo build --release \
    --target wasm32-unknown-unknown \
    --bin linkage-web \
    --no-default-features \
    --features raster
````

with:

````bash
# Output: web/linkage-web.js + web/linkage-web_bg.wasm, and
#         web/magcoupling/magcoupling-web.js + magcoupling-web_bg.wasm
#
# Both builds fail if cargo built the bundle with the test-only workbook-parity
# feature of magcoupling-rs (magcoupling_assert_shipped in magcoupling_shipped.sh):
# the linkage bundle carries the calculator's panel (Tools -> Magnetic coupling).
# deploy-web.yml runs this script.
#
# After building, serve with: scripts/serve_web.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=magcoupling_shipped.sh
source "$SCRIPT_DIR/magcoupling_shipped.sh"

cd "$PROJECT_DIR"

echo "Building WASM binary (release)..."
cargo build --release "${LINKAGE_WEB_ARGS[@]}" \
    --message-format=json-render-diagnostics \
    | magcoupling_assert_shipped linkage-web
````

In `.github/workflows/deploy-web.yml`, replace:

````yaml
      - name: Build WASM
        working-directory: linkage-sim-rs
        run: |
          cargo build --release \
            --target wasm32-unknown-unknown \
            --bin linkage-web \
            --no-default-features \
            --features raster
          wasm-bindgen \
            target/wasm32-unknown-unknown/release/linkage-web.wasm \
            --out-dir web \
            --target web \
            --no-typescript

      # The second bundle, the magnetic coupling calculator at /magcoupling/
      # (web/magcoupling/). The script holds the build arguments and fails the
      # job if cargo built it with the test-only workbook-parity feature.
      - name: Build magcoupling WASM
        working-directory: linkage-sim-rs
        run: bash scripts/build_magcoupling_web.sh
````

with:

````yaml
      # Both bundles: the linkage app (web/, with the calculator's panel in
      # Tools -> Magnetic coupling) and the magnetic coupling calculator at
      # /magcoupling/ (web/magcoupling/). The scripts hold the build arguments
      # (scripts/magcoupling_shipped.sh) and fail the job if cargo built either
      # bundle with the test-only workbook-parity feature.
      - name: Build WASM (both bundles)
        working-directory: linkage-sim-rs
        run: bash scripts/build_web.sh
````

In `linkage-sim-rs/scripts/build_magcoupling_web.sh`, replace:

````bash
# linkage app. Called by build_web.sh and by .github/workflows/deploy-web.yml.
````

with:

````bash
# linkage app. Called by build_web.sh, which .github/workflows/deploy-web.yml
# runs to build both bundles.
````

Then run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && bash -n scripts/gate.sh && bash -n scripts/build_web.sh && bash -n scripts/build_magcoupling_web.sh && bash -n scripts/magcoupling_shipped.sh && echo syntax-ok
grep -n "build_web.sh\|cargo build\|wasm-bindgen-cli" C:/Users/Cole/source/repos/lsim-mag-m5/.github/workflows/deploy-web.yml
```

Expected: `syntax-ok`; then three lines, `- name: Install wasm-bindgen-cli`, `run: cargo install wasm-bindgen-cli@0.2.114 --force` and `run: bash scripts/build_web.sh`, and no `cargo build` line (deploy-web.yml now builds only through the script).

- [ ] **Step 6: Run the linkage tests (the unified serde_json) and the gate**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --all 2>&1 | grep -E "test result" | head -1
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task1.log 2>&1; echo "exit=$?"; grep -E "parity guard|negative control|in both lock|installs" C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task1.log; tail -1 C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task1.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task1.log
git -C C:/Users/Cole/source/repos/lsim-mag-m5 checkout -- docs/chebyshev_lambda
```

Expected: `test result: ok. 926 passed` (magcoupling's `serde_json` feature `float_roundtrip` now applies to the linkage crate too, and every test passes with it); `exit=0`; gate 10's header `== gate 10/12: workbook-parity guard (shipped native and wasm32 builds of both apps; negative controls) ==` and lines `parity guard: magcoupling-app builds without workbook-parity; magcoupling-rs features: "features":["app","default","gui"]`, the same for `magcoupling-web`, `parity guard: linkage-gui builds without workbook-parity; magcoupling-rs features: "features":["default","gui"]`, the same for `linkage-web`, then one negative control per shipped build: `negative control: the guard trips on magcoupling-app when workbook-parity is forced on` and the same for `magcoupling-web`, `linkage-gui` and `linkage-web`; gate 11's `egui 0.32.3 in both lock files`, `egui_plot 0.33.0 in both lock files`, `eframe 0.32.3 in both lock files`, `wasm-bindgen 0.2.114 in both lock files`, `deploy-web.yml installs wasm-bindgen-cli 0.2.114`; `GATE PASS`; `0`.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m5 add linkage-sim-rs/Cargo.toml linkage-sim-rs/Cargo.lock linkage-sim-rs/scripts/magcoupling_shipped.sh linkage-sim-rs/scripts/gate.sh linkage-sim-rs/scripts/build_web.sh linkage-sim-rs/scripts/build_magcoupling_web.sh .github/workflows/deploy-web.yml
git -C C:/Users/Cole/source/repos/lsim-mag-m5 commit -F - <<'EOF'
feat(linkage-sim-rs): depend on magcoupling-rs (feature gui) and guard the linkage builds

linkage-sim-rs depends on magcoupling-rs by path with feature gui (one egui,
0.32.3; the lock gains only the magcoupling-rs entry), and on rfd for wasm32
(the calculator window's web file picker). magcoupling_shipped.sh defines the
linkage app's shipped cargo arguments once (LINKAGE_WEB_ARGS,
LINKAGE_NATIVE_ARGS); gate 3 uses them, gate 10 runs the workbook-parity
guard on linkage-gui and linkage-web and a negative control on every shipped
build (assert_guard_trips; the calculator's native app gains its own),
build_web.sh pipes the linkage release build through the guard, and
deploy-web.yml builds both bundles through build_web.sh (decision M5-8).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m5 status --short
```

Expected: one new commit on `magcoupling/m5`; `status --short` prints nothing. (A `sonnet` implementer writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 2: The calculator window: Tools menu, keyboard, files

**Model:** implementation on `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5); escalate a retry to the session model. The per-task review runs on the session model (omit `model`, comment `// session model: keyboard routing judgment`): judging the keyboard routing and focus semantics is judgment, and section 5 says "When unsure, use the session model".

**Files:**
- Create: `linkage-sim-rs/src/gui/calculator_window.rs` (Step 1: its tests; Step 3: the module above them)
- Modify: `linkage-sim-rs/src/gui/test_support.rs` (`key_tap` and `click_events` before `typed`; `NATIVE_SCREEN` and `screen_input` before `central_panel_frame`; `magcoupling_gap_design` at the end)
- Modify: `linkage-sim-rs/src/gui/mod.rs` (`mod calculator_window`; `LinkageApp::calculator`; the keyboard section of `update`; the menu call; the Keyboard Shortcuts window; `handle_keyboard_shortcuts` before the delete shortcut; tests)
- Modify: `linkage-sim-rs/src/gui/menu_bar.rs` (imports and `draw_menu_bar`'s signature; the Tools menu after Image; a new test module at the end)

**Interfaces:**
- Consumes: `magcoupling::gui::{MagcouplingPanel, PanelRequest}` (Task 1's dependency): `MagcouplingPanel::new() -> Self`, `ui(&mut self, &mut egui::Ui)`, `take_requests(&mut self) -> Vec<PanelRequest>`, `set_keyboard_shortcuts(&mut self, bool)`, `load_design_file(&mut self, &str) -> Result<(), LoadError>`, `report(&mut self, Result<String, String>)`, `design(&self) -> Design`; `PanelRequest::SaveFile { file_name: String, mime: &'static str, contents: String }` and `PanelRequest::OpenDesign`; `magcoupling::gui::session::{Design, design_to_json}`; `magcoupling::gui::results_table::{CSV_FILE_NAME, JSON_FILE_NAME}`; the linkage `export::download::{download_text, DownloadOutcome, FileFilter}`; egui's `gui_zoom::kb_shortcuts::{ZOOM_IN, ZOOM_IN_SECONDARY, ZOOM_OUT, ZOOM_RESET}` and the grab radii `ctx.style().interaction.resize_grab_radius_side` and `.resize_grab_radius_corner` (no import); `test_support::{drawn_texts, drew_text, primary_button, text_rect}`.
- Produces (crate-private module `gui::calculator_window`): `pub const TITLE: &str = "Magnetic coupling"`; `pub struct CalculatorWindow` (`Default`) with `pub fn is_open(&self) -> bool`, `pub fn set_open(&mut self, open: bool)`, `pub fn has_keyboard(&self) -> bool`, `pub fn show(&mut self, ctx: &egui::Context)`, and for tests `pub(crate) fn panel(&self) -> Option<&MagcouplingPanel>`, `pub(crate) fn design(&self) -> Design`, `pub(crate) fn load_design(&mut self, design: &Design)`; `pub(crate) fn window_id() -> egui::Id`; for tests `pub(crate) fn default_rect() -> egui::Rect` and `pub(crate) fn beside_the_window() -> egui::Pos2`. In `gui/mod.rs`: `fn handle_keyboard_shortcuts(ctx: &egui::Context, state: &mut AppState)`; `menu_bar::draw_menu_bar(ctx, state, sample_thumbnails, calculator: &mut CalculatorWindow)`. In `test_support`: `pub(crate) fn key_tap(key: egui::Key, modifiers: egui::Modifiers) -> Vec<egui::Event>`, `pub(crate) fn click_events(at: egui::Pos2) -> [Vec<egui::Event>; 3]`, `pub(crate) const NATIVE_SCREEN: egui::Vec2`, `pub(crate) fn screen_input(events: Vec<egui::Event>, size: egui::Vec2) -> egui::RawInput`, `pub(crate) fn magcoupling_gap_design(gap_mm: f64) -> Design`.

- [ ] **Step 1: Write the failing tests**

The window's tests go in the new file first (its module comes in Step 3, above them); the keyboard tests in `gui/mod.rs` drive the pieces `update` calls, in its order, never `update` itself (Global Constraints).

In `linkage-sim-rs/src/gui/test_support.rs`, replace:

````rust
/// Text typed into the focused widget.
````

with:

````rust
/// A key tapped with `modifiers` held: its press and its release in one frame. egui keeps a
/// pressed key in `InputState::keys_down` until its release, so a press alone holds it down for
/// good (and an egui panel waiting for the keys to settle waits forever).
pub(crate) fn key_tap(key: egui::Key, modifiers: egui::Modifiers) -> Vec<egui::Event> {
    [true, false]
        .map(|pressed| egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers })
        .to_vec()
}

/// A click at `at` as the input of three frames: the pointer moved there,
/// the press, the release.
pub(crate) fn click_events(at: egui::Pos2) -> [Vec<egui::Event>; 3] {
    [vec![egui::Event::PointerMoved(at)], vec![primary_button(at, true)], vec![primary_button(at, false)]]
}

/// Text typed into the focused widget.
````

In `linkage-sim-rs/src/gui/test_support.rs`, replace:

````rust
/// One headless frame of `draw` inside a central panel, with `events` as
````

with:

````rust
/// The native app's default window [points]: the screen of the headless
/// window and menu tests.
pub(crate) const NATIVE_SCREEN: egui::Vec2 = egui::vec2(1400.0, 900.0);

/// A frame's input: `events` on a screen of `size` [points] at the origin.
pub(crate) fn screen_input(events: Vec<egui::Event>, size: egui::Vec2) -> egui::RawInput {
    egui::RawInput { events, screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)), ..Default::default() }
}

/// One headless frame of `draw` inside a central panel, with `events` as
````

In `linkage-sim-rs/src/gui/test_support.rs`, replace:

````rust
    assert_eq!(state.current_sweep_index(), Some(sample_at(state, deg)));
}
````

with:

````rust
    assert_eq!(state.current_sweep_index(), Some(sample_at(state, deg)));
}

/// The magnetic coupling calculator's default design with its face gap at
/// `gap_mm`: a design file its panel loads, one undo step of the panel.
pub(crate) fn magcoupling_gap_design(gap_mm: f64) -> magcoupling::gui::session::Design {
    let mut design = magcoupling::gui::session::Design::default();
    design.inputs.metal.face_gap_mm = gap_mm;
    design
}
````

Create `linkage-sim-rs/src/gui/calculator_window.rs`:

````rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::test_support::{
        NATIVE_SCREEN, click_events, drew_text, key_tap, magcoupling_gap_design, primary_button,
        screen_input, text_rect,
    };
    use magcoupling::gui::results_table::{CSV_FILE_NAME, JSON_FILE_NAME};
    use magcoupling::gui::session::{Design, design_to_json};

    /// The panel's heading (magcoupling-rs `gui::panel::HEADING`), a label: a click on it does
    /// nothing but give the window the keyboard.
    const HEADING: &str = "Magnetic coupling calculator";

    /// Where the host draws a Foreground area, standing in for an open menu or a drop-down list.
    const POPUP_AT: egui::Pos2 = egui::pos2(1250.0, 40.0);

    /// One host frame with `events`: the window first, as in `LinkageApp::update`, then a
    /// Foreground area and a central panel for the linkage app. Returns what egui painted and
    /// the keys the host still saw pressed after the window.
    fn frame(
        ctx: &egui::Context,
        window: &mut CalculatorWindow,
        events: Vec<egui::Event>,
    ) -> (egui::FullOutput, Vec<egui::Key>) {
        let mut keys = Vec::new();
        let output = ctx.run(screen_input(events, NATIVE_SCREEN), |ctx| {
            window.show(ctx);
            keys = ctx.input(|i| {
                i.events
                    .iter()
                    .filter_map(|event| match event {
                        egui::Event::Key {
                            key, pressed: true, ..
                        } => Some(*key),
                        _ => None,
                    })
                    .collect()
            });
            egui::Area::new(egui::Id::new("test_popup"))
                .order(egui::Order::Foreground)
                .fixed_pos(POPUP_AT)
                .show(ctx, |ui| ui.label("popup"));
            egui::CentralPanel::default().show(ctx, |ui| ui.label("linkage"));
        });
        (output, keys)
    }

    /// One frame of the window alone on a screen of `size` [points].
    fn window_frame(
        ctx: &egui::Context,
        window: &mut CalculatorWindow,
        size: egui::Vec2,
    ) -> egui::FullOutput {
        ctx.run(screen_input(Vec::new(), size), |ctx| window.show(ctx))
    }

    /// One frame of the window and then an area of `size` at `at` [points], drawn after the window
    /// as the linkage app draws its welcome screen and its other windows, with `events`.
    fn frame_with_area(
        ctx: &egui::Context,
        window: &mut CalculatorWindow,
        at: egui::Pos2,
        size: egui::Vec2,
        events: Vec<egui::Event>,
    ) {
        let _ = ctx.run(screen_input(events, NATIVE_SCREEN), |ctx| {
            window.show(ctx);
            egui::Area::new(egui::Id::new("test_area"))
                .fixed_pos(at)
                .show(ctx, |ui| {
                    ui.allocate_space(size);
                });
        });
    }

    /// Opens the window and runs the two frames a window takes to size itself and paint.
    fn opened(ctx: &egui::Context) -> (CalculatorWindow, egui::FullOutput) {
        let mut window = CalculatorWindow::default();
        window.set_open(true);
        frame(ctx, &mut window, Vec::new());
        let (output, _) = frame(ctx, &mut window, Vec::new());
        (window, output)
    }

    /// A press at `at` (with the pointer moved there first), then its release.
    fn press(ctx: &egui::Context, window: &mut CalculatorWindow, at: egui::Pos2) {
        for events in click_events(at) {
            frame(ctx, window, events);
        }
    }

    /// Loads the design with the face gap at `gap_mm` and runs the frame that makes it an undo
    /// step of the panel.
    fn load_gap(ctx: &egui::Context, window: &mut CalculatorWindow, gap_mm: f64) {
        window.load_design(&magcoupling_gap_design(gap_mm));
        frame(ctx, window, Vec::new());
    }

    fn window_rect(ctx: &egui::Context) -> egui::Rect {
        ctx.memory(|m| m.area_rect(window_id()))
            .expect("the window is shown")
    }

    #[test]
    fn the_window_is_closed_until_opened_and_then_shows_the_panel() {
        let ctx = egui::Context::default();
        let mut window = CalculatorWindow::default();
        assert!(!window.is_open());
        assert!(
            window.panel().is_none(),
            "no panel before the first opening"
        );
        let (output, _) = frame(&ctx, &mut window, Vec::new());
        assert!(!drew_text(&output, TITLE));
        assert!(!window.has_keyboard());

        let (window, output) = opened(&ctx);
        assert!(window.is_open());
        assert!(drew_text(&output, TITLE), "the window's title");
        assert!(drew_text(&output, HEADING), "the panel inside it");
        assert!(window.has_keyboard(), "an opened window has the keyboard");
    }

    #[test]
    fn an_opened_window_is_in_front_of_what_the_app_draws_after_it() {
        // At start-up (?tool=magcoupling) the linkage app's welcome screen, an area drawn after
        // the window, would otherwise cover it.
        let ctx = egui::Context::default();
        let mut window = CalculatorWindow::default();
        window.set_open(true);
        let inside = default_rect().center();
        for _ in 0..3 {
            let welcome_at = inside - egui::vec2(160.0, 120.0);
            frame_with_area(
                &ctx,
                &mut window,
                welcome_at,
                egui::vec2(320.0, 240.0),
                Vec::new(),
            );
        }
        assert_eq!(ctx.layer_id_at(inside), Some(window_layer()));
    }

    #[test]
    fn a_small_screen_keeps_the_window_on_it() {
        // A browser window smaller than the window's starting size (1100 x 700 at 80, 100).
        let ctx = egui::Context::default();
        let mut window = CalculatorWindow::default();
        window.set_open(true);
        let size = egui::vec2(800.0, 600.0);
        let mut output = None;
        for _ in 0..3 {
            output = Some(window_frame(&ctx, &mut window, size));
        }
        let rect = window_rect(&ctx);
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        assert!(
            screen.contains_rect(rect),
            "{rect:?} is not inside {screen:?}"
        );
        assert!(drew_text(&output.expect("three frames"), HEADING));
    }

    #[test]
    fn a_screen_without_room_skips_the_window_and_keeps_its_place_and_size() {
        // On the web a hidden canvas reports a 0 x 0 screen; the window comes back as it was.
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        let before = window_rect(&ctx);
        let output = window_frame(&ctx, &mut window, egui::Vec2::ZERO);
        assert!(
            !drew_text(&output, HEADING),
            "nothing drawn on a 0 x 0 screen"
        );
        frame(&ctx, &mut window, Vec::new());
        let (output, _) = frame(&ctx, &mut window, Vec::new());
        assert_eq!(window_rect(&ctx), before, "back in its place, at its size");
        assert!(drew_text(&output, HEADING));
    }

    #[test]
    fn closing_the_window_with_a_focused_field_gives_the_keys_back() {
        let ctx = egui::Context::default();
        let (mut window, output) = opened(&ctx);
        // The inner magnet part's text field (the first text drawn with the part's name).
        let field = text_rect(&output, "B842SH")
            .expect("the part field")
            .center();
        press(&ctx, &mut window, field);
        assert!(
            ctx.wants_keyboard_input(),
            "the field has the keyboard focus"
        );

        window.set_open(false);
        frame(&ctx, &mut window, Vec::new());
        frame(&ctx, &mut window, Vec::new());
        assert!(
            !ctx.wants_keyboard_input(),
            "the closed window's field lost its focus"
        );
        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Delete, egui::Modifiers::NONE),
        );
        assert_eq!(keys, [egui::Key::Delete]);
    }

    #[test]
    fn closing_and_reopening_keeps_the_design_and_its_undo_history() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        load_gap(&ctx, &mut window, 2.0);

        window.set_open(false);
        let (output, _) = frame(&ctx, &mut window, Vec::new());
        assert!(
            !drew_text(&output, HEADING),
            "a closed window draws nothing"
        );
        assert!(!window.has_keyboard(), "closing gives the keyboard back");

        window.set_open(true);
        frame(&ctx, &mut window, Vec::new());
        let (output, _) = frame(&ctx, &mut window, Vec::new());
        assert!(drew_text(&output, HEADING));
        assert_eq!(
            window.design(),
            magcoupling_gap_design(2.0),
            "the design is kept"
        );
        // The undo history too: Ctrl+Z (the reopened window has the keyboard) undoes the load.
        frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Z, egui::Modifiers::COMMAND),
        );
        assert_eq!(window.design(), Design::default());
    }

    #[test]
    fn the_keyboard_follows_presses_and_popups_leave_it_alone() {
        let ctx = egui::Context::default();
        let (mut window, output) = opened(&ctx);
        let heading = text_rect(&output, HEADING).expect("the heading").center();
        let popup = ctx
            .memory(|m| m.area_rect(egui::Id::new("test_popup")))
            .expect("the stand-in popup")
            .center();

        press(&ctx, &mut window, beside_the_window());
        assert!(
            !window.has_keyboard(),
            "a press on the linkage app takes it"
        );
        press(&ctx, &mut window, popup);
        assert!(
            !window.has_keyboard(),
            "an open menu or a drop-down leaves it with the linkage app"
        );
        press(&ctx, &mut window, heading);
        assert!(window.has_keyboard(), "a press on the window gives it");
        press(&ctx, &mut window, popup);
        assert!(
            window.has_keyboard(),
            "an open menu or a drop-down leaves it with the window"
        );
        // A press released in the same frame counts too.
        let outside = beside_the_window();
        frame(
            &ctx,
            &mut window,
            vec![
                primary_button(outside, true),
                primary_button(outside, false),
            ],
        );
        assert!(!window.has_keyboard());
    }

    #[test]
    fn a_press_on_the_band_where_egui_resizes_the_window_gives_it_the_keyboard() {
        // egui lets the user grab the window's edge from up to 5 points outside its frame and a
        // corner from up to 10 points around it (its Interaction style's grab radii).
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        let right_of = |ctx: &egui::Context, points: f32| {
            let rect = window_rect(ctx);
            egui::pos2(rect.right() + points, rect.center().y)
        };

        press(&ctx, &mut window, beside_the_window());
        let before = window_rect(&ctx);
        press(&ctx, &mut window, right_of(&ctx, 3.0));
        assert!(window.has_keyboard(), "the right edge's band gives it");
        assert!(
            window_rect(&ctx).right() > before.right(),
            "egui took the press for a grab of the edge (the edge jumps to the pointer)"
        );
        press(&ctx, &mut window, beside_the_window());
        let corner = window_rect(&ctx).right_bottom() + egui::vec2(7.0, 7.0);
        press(&ctx, &mut window, corner);
        assert!(window.has_keyboard(), "the corner's band gives it");
        press(&ctx, &mut window, right_of(&ctx, 8.0));
        assert!(
            !window.has_keyboard(),
            "beyond the band, the linkage app takes it"
        );
    }

    #[test]
    fn another_window_over_the_band_takes_the_keyboard() {
        // egui gives a press there to the window on top: here an area just right of the
        // calculator's edge, as another linkage window could be.
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        let rect = window_rect(&ctx);
        let neighbour_at = egui::pos2(rect.right() + 1.0, rect.center().y - 20.0);
        let press_at = egui::pos2(rect.right() + 3.0, rect.center().y);
        for events in [Vec::new(), Vec::new()]
            .into_iter()
            .chain(click_events(press_at))
        {
            frame_with_area(
                &ctx,
                &mut window,
                neighbour_at,
                egui::vec2(40.0, 40.0),
                events,
            );
        }
        assert!(!window.has_keyboard(), "the window on top took it");
    }

    #[test]
    fn with_the_keyboard_the_panel_undoes_and_the_host_sees_no_key() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        load_gap(&ctx, &mut window, 2.0);

        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Z, egui::Modifiers::COMMAND),
        );
        assert_eq!(
            window.design(),
            Design::default(),
            "the panel undid the load"
        );
        assert!(keys.is_empty(), "the host saw {keys:?}");
        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Y, egui::Modifiers::COMMAND),
        );
        assert_eq!(window.design(), magcoupling_gap_design(2.0), "and redid it");
        assert!(keys.is_empty(), "the host saw {keys:?}");
        // Minus alone is no zoom key: only Ctrl+Minus is (the next test).
        for key in [
            egui::Key::ArrowRight,
            egui::Key::Delete,
            egui::Key::F,
            egui::Key::Escape,
            egui::Key::Minus,
        ] {
            let (_, keys) = frame(&ctx, &mut window, key_tap(key, egui::Modifiers::NONE));
            assert!(keys.is_empty(), "{key:?}: the host saw {keys:?}");
        }
    }

    #[test]
    fn with_the_keyboard_egui_s_zoom_keys_still_zoom_the_ui() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        assert!(window.has_keyboard());
        for (key, zoom) in [
            (egui::Key::Plus, 1.1),
            (egui::Key::Equals, 1.2),
            (egui::Key::Minus, 1.1),
            (egui::Key::Num0, 1.0),
        ] {
            let (_, keys) = frame(&ctx, &mut window, key_tap(key, egui::Modifiers::COMMAND));
            assert_eq!(keys, [key], "Ctrl+{key:?} is left for egui");
            // egui zooms at the end of the frame and applies the new factor at the next one.
            frame(&ctx, &mut window, Vec::new());
            assert_eq!(ctx.zoom_factor(), zoom, "Ctrl+{key:?}");
        }
    }

    #[test]
    fn without_the_keyboard_the_panel_leaves_the_keys_to_the_host() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        load_gap(&ctx, &mut window, 2.0);
        press(&ctx, &mut window, beside_the_window());

        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Z, egui::Modifiers::COMMAND),
        );
        assert_eq!(
            window.design(),
            magcoupling_gap_design(2.0),
            "the panel did not undo"
        );
        assert_eq!(keys, [egui::Key::Z], "the host saw Ctrl+Z");
        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::ArrowRight, egui::Modifiers::NONE),
        );
        assert_eq!(keys, [egui::Key::ArrowRight]);
    }

    #[test]
    fn a_collapsed_window_leaves_the_keys_to_the_host() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        load_gap(&ctx, &mut window, 2.0);
        // Collapsed with its title bar's triangle: only the title bar shows.
        let mut collapsing = egui::collapsing_header::CollapsingState::load_with_default_open(
            &ctx,
            window_id().with("collapsing"),
            true,
        );
        collapsing.set_open(false);
        collapsing.store(&ctx);
        // The collapse animates over a few frames (egui adds 1/60 s a frame).
        for _ in 0..30 {
            frame(&ctx, &mut window, Vec::new());
        }
        let (output, _) = frame(&ctx, &mut window, Vec::new());
        assert!(drew_text(&output, TITLE), "the title bar shows");
        assert!(!drew_text(&output, HEADING), "the panel does not");

        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Z, egui::Modifiers::COMMAND),
        );
        assert_eq!(keys, [egui::Key::Z], "the host saw Ctrl+Z");
        assert_eq!(
            window.design(),
            magcoupling_gap_design(2.0),
            "the panel did not undo"
        );
    }

    #[test]
    fn a_closed_window_leaves_every_key_to_the_host() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        window.set_open(false);
        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Z, egui::Modifiers::COMMAND),
        );
        assert_eq!(keys, [egui::Key::Z]);
    }

    #[test]
    fn saved_files_get_the_filter_of_their_type() {
        for (name, label, extension) in [
            (JSON_FILE_NAME, "JSON", "json"),
            ("magcoupling-design.json", "JSON", "json"),
            (CSV_FILE_NAME, "CSV", "csv"),
            ("notes.txt", "All files", "*"),
            ("no_extension", "All files", "*"),
        ] {
            let filter = save_filter(name);
            assert_eq!(
                (filter.label, filter.extensions),
                (label, &[extension][..]),
                "{name}"
            );
        }
    }

    #[test]
    fn a_save_outcome_becomes_the_panel_s_report() {
        assert_eq!(
            report_of(DownloadOutcome::Saved("Saved: a.json".to_owned())),
            Some(Ok("Saved: a.json".to_owned()))
        );
        assert_eq!(report_of(DownloadOutcome::Cancelled), None);
        assert_eq!(
            report_of(DownloadOutcome::Failed("Write failed: denied".to_owned())),
            Some(Err("Write failed: denied".to_owned()))
        );
    }

    #[test]
    fn a_picked_design_file_loads_and_a_refused_one_changes_nothing() {
        let mut panel = MagcouplingPanel::new();
        let design = magcoupling_gap_design(2.0);
        open_picked_file(&mut panel, Ok(design_to_json(&design)));
        assert_eq!(panel.design(), design);
        open_picked_file(&mut panel, Ok("not a design".to_owned()));
        assert_eq!(panel.design(), design);
        open_picked_file(&mut panel, Err("the file is not UTF-8 text".to_owned()));
        assert_eq!(panel.design(), design);
    }

    #[test]
    fn a_picked_file_reaches_the_panel_on_the_next_frame() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        let design = magcoupling_gap_design(2.0);
        *window.picker.inbox.borrow_mut() = Some(Ok(design_to_json(&design)));
        frame(&ctx, &mut window, Vec::new());
        assert_eq!(window.design(), design);
        assert!(window.picker.take().is_none(), "taken once");
    }
}
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
mod state;
mod canvas;
````

with:

````rust
mod state;
mod calculator_window;
mod canvas;
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
    use super::*;
    use crate::gui::test_support::{central_panel_frame, drawn_texts, drew_text, key_press, typed};
````

with:

````rust
    use super::*;
    use super::calculator_window::CalculatorWindow;
    use crate::gui::test_support::{central_panel_frame, click_events, drawn_texts, drew_text, key_press, key_tap, magcoupling_gap_design, screen_input, typed, NATIVE_SCREEN};
    use magcoupling::gui::session::Design;
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
        assert!(state.find_point_mass("coupler", "W1").is_some(), "the weight survives");
        assert_eq!(state.selected, Some(weight("W1")));
        assert_eq!(state.undo_history.undo_count(), depth);
    }
}
````

with:

````rust
        assert!(state.find_point_mass("coupler", "W1").is_some(), "the weight survives");
        assert_eq!(state.selected, Some(weight("W1")));
        assert_eq!(state.undo_history.undo_count(), depth);
    }

    // ── Tools → Magnetic coupling beside the model (M5) ─────────────────────

    /// One frame of the linkage app's keyboard readers with `events`, `modifiers` held, in
    /// `update`'s order (the order test below pins it): the calculator window, the shortcuts,
    /// the delete shortcut, then the canvas (its arrow nudge, F, Escape and Enter keys).
    fn keys_frame(
        ctx: &egui::Context,
        calculator: &mut CalculatorWindow,
        state: &mut AppState,
        modifiers: egui::Modifiers,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        let input = egui::RawInput { modifiers, ..screen_input(events, NATIVE_SCREEN) };
        ctx.run(input, |ctx| {
            calculator.show(ctx);
            handle_keyboard_shortcuts(ctx, state);
            handle_delete_shortcut(ctx, state);
            egui::CentralPanel::default().show(ctx, |ui| canvas::draw_canvas(ui, state));
        })
    }

    /// The four-bar with weights W1 and W2 (two undo steps) on the fitted canvas, beside the
    /// calculator window, open (so it has the keyboard) with its design at a 2 mm face gap, one
    /// undo step of the calculator.
    fn fourbar_beside_the_calculator() -> (egui::Context, CalculatorWindow, AppState) {
        let ctx = egui::Context::default();
        let mut state = fourbar_with_weights();
        let mut calculator = CalculatorWindow::default();
        calculator.set_open(true);
        for _ in 0..2 {
            keys_frame(&ctx, &mut calculator, &mut state, egui::Modifiers::NONE, Vec::new());
        }
        calculator.load_design(&magcoupling_gap_design(2.0));
        keys_frame(&ctx, &mut calculator, &mut state, egui::Modifiers::NONE, Vec::new());
        (ctx, calculator, state)
    }

    /// A click on the canvas beside the calculator window (clear of the fitted four-bar), which
    /// gives the keyboard back to the linkage app.
    fn click_the_canvas(ctx: &egui::Context, calculator: &mut CalculatorWindow, state: &mut AppState) {
        for events in click_events(calculator_window::beside_the_window()) {
            keys_frame(ctx, calculator, state, egui::Modifiers::NONE, events);
        }
        assert!(!calculator.has_keyboard(), "the click took the keyboard back");
    }

    #[test]
    fn ctrl_z_undoes_only_the_part_that_has_the_keyboard() {
        let (ctx, mut calculator, mut state) = fourbar_beside_the_calculator();
        let depth = state.undo_history.undo_count();
        let ctrl = egui::Modifiers::COMMAND;

        // The calculator has the keyboard: Ctrl+Z and Ctrl+Y undo and redo its design only.
        keys_frame(&ctx, &mut calculator, &mut state, ctrl, key_tap(egui::Key::Z, ctrl));
        assert_eq!(calculator.design(), Design::default(), "the calculator undid");
        assert_eq!(state.undo_history.undo_count(), depth, "the model was not undone");
        assert!(state.find_point_mass("coupler", "W2").is_some());
        keys_frame(&ctx, &mut calculator, &mut state, ctrl, key_tap(egui::Key::Y, ctrl));
        assert_eq!(calculator.design(), magcoupling_gap_design(2.0), "the calculator redid");
        assert_eq!(state.undo_history.undo_count(), depth);

        // A click on the canvas gives the keyboard back: Ctrl+Z undoes the model only.
        click_the_canvas(&ctx, &mut calculator, &mut state);
        keys_frame(&ctx, &mut calculator, &mut state, ctrl, key_tap(egui::Key::Z, ctrl));
        assert!(state.find_point_mass("coupler", "W2").is_none(), "the model's last edit was undone");
        assert_eq!(state.undo_history.undo_count(), depth - 1);
        assert_eq!(calculator.design(), magcoupling_gap_design(2.0), "the calculator was not undone");
    }

    #[test]
    fn arrow_keys_and_delete_leave_the_model_alone_while_the_calculator_has_the_keyboard() {
        let (ctx, mut calculator, mut state) = fourbar_beside_the_calculator();
        let none = egui::Modifiers::NONE;
        let coupler = SelectedEntity::Body("coupler".to_string());
        state.selected = Some(coupler.clone());
        let depth = state.undo_history.undo_count();

        for key in [egui::Key::ArrowRight, egui::Key::Delete, egui::Key::Escape] {
            keys_frame(&ctx, &mut calculator, &mut state, none, key_tap(key, none));
        }
        assert_eq!(state.undo_history.undo_count(), depth, "nothing was nudged or deleted");
        assert_eq!(state.selected, Some(coupler.clone()));

        // With the keyboard back, the same arrow key nudges the selected link: one undo step.
        click_the_canvas(&ctx, &mut calculator, &mut state);
        state.selected = Some(coupler);
        keys_frame(&ctx, &mut calculator, &mut state, none, key_tap(egui::Key::ArrowRight, none));
        assert_eq!(state.undo_history.undo_count(), depth + 1, "the arrow key nudged the link");
    }

    #[test]
    fn update_shows_the_calculator_window_before_anything_reads_the_keyboard() {
        // CalculatorWindow::show takes the frame's key events while the window has the
        // keyboard, so it must come before every reader of the keys in `update`.
        let source = include_str!("mod.rs").replace("\r\n", "\n");
        let start = source.find("fn update(&mut self, ctx: &egui::Context").expect("update");
        let end = start + source[start..].find("\n    }\n}\n").expect("the end of update");
        let update = &source[start..end];
        let show = update.find("self.calculator.show(ctx);").expect("update shows the window");
        for reader in [
            "handle_keyboard_shortcuts(ctx",
            "menu_bar::draw_menu_bar(",
            "handle_delete_shortcut(ctx",
            "canvas::draw_canvas(",
            "key_pressed(",
        ] {
            let at = update.find(reader).unwrap_or_else(|| panic!("update calls {reader}"));
            assert!(show < at, "{reader} comes before the calculator window");
        }
    }
}
````

In `linkage-sim-rs/src/gui/menu_bar.rs`, replace:

````rust
        format!("{} days ago", delta / 86_400)
    }
}
````

with:

````rust
        format!("{} days ago", delta / 86_400)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::test_support::{click_events, drawn_texts, screen_input, text_rect, NATIVE_SCREEN};

    /// One frame of the menu bar with `events`.
    fn frame(
        ctx: &egui::Context,
        state: &mut AppState,
        calculator: &mut CalculatorWindow,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        ctx.run(screen_input(events, NATIVE_SCREEN), |ctx| draw_menu_bar(ctx, state, &HashMap::new(), calculator))
    }

    /// One frame of the calculator window and then the menu bar with `events`, in
    /// `LinkageApp::update`'s order (the window follows the frame's presses).
    fn frame_with_the_window(
        ctx: &egui::Context,
        state: &mut AppState,
        calculator: &mut CalculatorWindow,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        ctx.run(screen_input(events, NATIVE_SCREEN), |ctx| {
            calculator.show(ctx);
            draw_menu_bar(ctx, state, &HashMap::new(), calculator);
        })
    }

    /// Clicks the text `label` drawn in `output` (the pointer moved there, a press, a release:
    /// one frame each) and returns the frame after.
    fn click(
        ctx: &egui::Context,
        state: &mut AppState,
        calculator: &mut CalculatorWindow,
        output: &egui::FullOutput,
        label: &str,
    ) -> egui::FullOutput {
        let at = text_rect(output, label).unwrap_or_else(|| panic!("no {label:?}")).center();
        for events in click_events(at) {
            frame(ctx, state, calculator, events);
        }
        frame(ctx, state, calculator, Vec::new())
    }

    #[test]
    fn tools_magnetic_coupling_opens_and_closes_the_calculator_window() {
        let ctx = egui::Context::default();
        let mut state = AppState::default();
        let mut calculator = CalculatorWindow::default();
        let output = frame(&ctx, &mut state, &mut calculator, Vec::new());

        let output = click(&ctx, &mut state, &mut calculator, &output, "Tools");
        let output = click(&ctx, &mut state, &mut calculator, &output, calculator_window::TITLE);
        assert!(calculator.is_open());
        assert!(calculator.has_keyboard(), "the opened window has the keyboard");
        assert!(text_rect(&output, calculator_window::TITLE).is_none(), "the menu closed");

        let output = click(&ctx, &mut state, &mut calculator, &output, "Tools");
        click(&ctx, &mut state, &mut calculator, &output, calculator_window::TITLE);
        assert!(!calculator.is_open());
        assert!(calculator.panel().is_some(), "the calculator is kept while its window is closed");
    }

    #[test]
    fn a_press_on_the_menu_bar_gives_the_keyboard_to_the_linkage_app() {
        // The menu bar's buttons sit on a Background panel of the linkage app; only an open
        // menu's items are a Foreground popup, which leaves the keyboard where it is.
        let ctx = egui::Context::default();
        let mut state = AppState::default();
        let mut calculator = CalculatorWindow::default();
        calculator.set_open(true);
        frame_with_the_window(&ctx, &mut state, &mut calculator, Vec::new());
        let output = frame_with_the_window(&ctx, &mut state, &mut calculator, Vec::new());
        assert!(calculator.has_keyboard(), "the opened window has the keyboard");

        let tools = text_rect(&output, "Tools").expect("the Tools button").center();
        for events in click_events(tools) {
            frame_with_the_window(&ctx, &mut state, &mut calculator, events);
        }
        assert!(!calculator.has_keyboard(), "the press on Tools gave the keyboard to the linkage app");
        let output = frame_with_the_window(&ctx, &mut state, &mut calculator, Vec::new());
        let titles = drawn_texts(&output).into_iter().filter(|text| text == calculator_window::TITLE).count();
        assert_eq!(titles, 2, "the Tools menu opened: its item and the window's title");
    }
}
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --lib 2>&1 | grep -E "^error: could not compile|cannot find function .handle_keyboard_shortcuts|takes 3 arguments"
```

Expected: ``error[E0061]: this function takes 3 arguments but 4 arguments were supplied`` (twice: the menu tests' two frames), ``error[E0425]: cannot find function `handle_keyboard_shortcuts` in this scope`` and ``error: could not compile `linkage-sim-rs` (lib test) due to 135 previous errors; 23 warnings emitted`` (most of the errors are the window module's names, which do not exist yet).

- [ ] **Step 3: Write the implementation**

The module goes above the tests in `calculator_window.rs`; `update`'s inline shortcut block moves, unchanged but for `self.state` becoming `state`, into `handle_keyboard_shortcuts`, which runs right after the window (decision M5-5).

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
#[cfg(test)]
mod tests {
    use super::*;
````

with:

````rust
//! Tools → Magnetic coupling: the magnetic coupling calculator (`magcoupling-rs`'s
//! `gui::MagcouplingPanel`) in an `egui::Window` of the linkage app. Spec:
//! `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, section M5.
//!
//! The calculator's state is its own: the panel is created the first time the window opens (its
//! equation registry is built then, not at the linkage app's start-up) and kept, open or closed,
//! until the app quits. Nothing here reads or writes the linkage model (decision M5-3).
//!
//! Keyboard (decision M5-5): the window has the keyboard from its opening or a press on it (or on
//! the band just outside its frame where egui lets the user grab its edge to resize it) until a
//! press on the linkage app: its panels, its canvas, its menu bar's buttons (File, Edit, View,
//! Image, Tools: a Background panel) or another window. A press on an open menu's items or a
//! drop-down list (a Foreground layer: the linkage app's menus and the calculator's drop-downs
//! alike) leaves the keyboard where it is. While the window has the keyboard, the panel acts on
//! Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y (`MagcouplingPanel::set_keyboard_shortcuts`) and, once the panel
//! has drawn, [`CalculatorWindow::show`] takes the frame's key events out but egui's own zoom keys
//! (`UI_ZOOM_KEYS`), so no linkage shortcut or canvas key (undo, save, new, delete, the arrow nudge,
//! F, Escape, Enter) runs on them while Ctrl+Plus, Ctrl+Minus and Ctrl+0 still zoom the whole UI
//! (natively; on the web eframe leaves those keys to the browser, which zooms the page).
//! `LinkageApp::update` therefore shows the window before anything else reads the keyboard (a test
//! in `gui/mod.rs` checks the order). Without the keyboard, or while the window is collapsed to its
//! title bar, the panel leaves the keys alone and the linkage app reads them as before.
//!
//! Known edges, documented rather than handled (backlog BL-039, BL-040): the keyboard follows
//! presses, not egui's keyboard focus, so after Tab moves the focus from one part into the other
//! the keys stay where they were until a click there; and the window, once raised, covers the
//! linkage app's "Recover Unsaved Work?" prompt if that shows at the same time (move or collapse
//! the window to answer it).
//!
//! Files (decision M5-6): the panel's requests are done here. Saving uses the linkage app's
//! `export::download` (a file dialog natively, a download on the web); "Load design" uses
//! [`DesignPicker`] (rfd's file dialog natively, rfd's file picker on the web).

use std::cell::RefCell;
use std::rc::Rc;

use eframe::egui;
use magcoupling::gui::{MagcouplingPanel, PanelRequest};

use super::export::download::{self, DownloadOutcome, FileFilter};

/// The window's title and the Tools menu item.
pub const TITLE: &str = "Magnetic coupling";

/// Where the window first opens [points], below the menu bar and the two toolbars, and its
/// starting size (decision M5-2). egui keeps the window inside the screen.
const DEFAULT_POS: [f32; 2] = [80.0, 100.0];
const DEFAULT_SIZE: [f32; 2] = [1100.0, 700.0];

/// egui's own keyboard zoom of the whole UI (`egui::gui_zoom`, run at the end of every frame): the
/// window leaves these key events in place while it has the keyboard. In effect natively only:
/// eframe's web backend turns egui's keyboard zoom off and leaves these keys to the browser.
const UI_ZOOM_KEYS: [egui::KeyboardShortcut; 4] = [
    egui::gui_zoom::kb_shortcuts::ZOOM_IN,
    egui::gui_zoom::kb_shortcuts::ZOOM_IN_SECONDARY,
    egui::gui_zoom::kb_shortcuts::ZOOM_OUT,
    egui::gui_zoom::kb_shortcuts::ZOOM_RESET,
];

/// The window's id (its area's and its layer's).
pub(crate) fn window_id() -> egui::Id {
    egui::Id::new("magcoupling_window")
}

/// The window's layer: a press there gives it the keyboard.
fn window_layer() -> egui::LayerId {
    egui::LayerId::new(egui::Order::Middle, window_id())
}

/// The rect of the window's contents when it first opens (`DEFAULT_POS`, `DEFAULT_SIZE`). The
/// window itself is a little larger: its frame, its title bar and the panel's own minimum width
/// reach about 72 points right of it and 48 below it.
#[cfg(test)]
pub(crate) fn default_rect() -> egui::Rect {
    egui::Rect::from_min_size(DEFAULT_POS.into(), DEFAULT_SIZE.into())
}

/// A point on the host right of and below the window as it first opens, clear of its frame and of
/// the band around it where egui resizes it.
#[cfg(test)]
pub(crate) fn beside_the_window() -> egui::Pos2 {
    default_rect().right_bottom() + egui::vec2(120.0, 50.0)
}

/// The calculator window: whether it is open, its panel, and whether it has the keyboard.
#[derive(Default)]
pub struct CalculatorWindow {
    open: bool,
    /// Created on the first opening, then kept, open or closed, until the app quits.
    panel: Option<MagcouplingPanel>,
    /// Whether the window has the keyboard (module docs); cleared when the window closes.
    keyboard: bool,
    /// Set by opening, until the window has been brought in front of the app's other windows and
    /// areas (the linkage app's welcome screen included).
    raise: bool,
    picker: DesignPicker,
}

impl CalculatorWindow {
    /// Whether the window is open.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Opens the window, which then has the keyboard (the first opening creates the panel), or
    /// closes it, which gives the keyboard back. The panel and its undo history are kept.
    pub fn set_open(&mut self, open: bool) {
        self.open = open;
        self.keyboard = open;
        self.raise = open;
        if open && self.panel.is_none() {
            self.panel = Some(MagcouplingPanel::new());
        }
    }

    /// Whether the window has the keyboard (module docs).
    pub fn has_keyboard(&self) -> bool {
        self.open && self.keyboard
    }

    /// Draws the window when it is open and does the panel's requests; while the window has the
    /// keyboard, takes this frame's key events out once the panel has drawn. Call it before
    /// anything else in the frame reads the keyboard (module docs).
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        self.follow_presses(ctx);
        let Some(panel) = self.panel.as_mut() else {
            return; // set_open(true) always creates the panel
        };
        if let Some(picked) = self.picker.take() {
            open_picked_file(panel, picked);
        }
        panel.set_keyboard_shortcuts(self.keyboard);
        // egui 0.32 keeps a window's contents within its constrain rect but not its frame: on a
        // screen smaller than the window the frame's margins (14 points) hang off the right and
        // bottom edges. Constrain the window to the screen less those margins.
        let screen = ctx.screen_rect();
        let margins = egui::Frame::window(&ctx.style()).total_margin().sum();
        let room = screen.size() - margins;
        // A screen with no room for the frame (on the web a hidden canvas reports 0 x 0; the native
        // backend never sends an empty screen): skip the window this frame. egui would otherwise
        // keep the size it squeezed the window to, and show it back on a real screen at its
        // smallest, at the screen's corner.
        if room.x <= 0.0 || room.y <= 0.0 {
            return;
        }
        let mut open = true;
        let shown = egui::Window::new(TITLE)
            .id(window_id())
            .open(&mut open)
            .default_pos(DEFAULT_POS)
            .default_size(DEFAULT_SIZE)
            .constrain_to(egui::Rect::from_min_size(screen.min, room))
            .show(ctx, |ui| panel.ui(ui));
        // A collapsed window draws no panel (`inner` is `None`), so it takes no keys either.
        let drawn = shown.is_some_and(|response| response.inner.is_some());
        // egui brings a window in front when it first shows, but with it every area that first
        // shows in the same frame (the linkage app's welcome screen at start-up, drawn after the
        // window): bring it in front once more on the next frame.
        if self.raise && ctx.memory(|m| m.areas().visible_last_frame(&window_layer())) {
            ctx.move_to_top(window_layer());
            self.raise = false;
        }
        for request in panel.take_requests() {
            match request {
                PanelRequest::SaveFile {
                    file_name,
                    mime,
                    contents,
                } => {
                    let outcome = download::download_text(
                        &file_name,
                        mime,
                        &contents,
                        save_filter(&file_name),
                    );
                    if let Some(report) = report_of(outcome) {
                        panel.report(report);
                    }
                }
                PanelRequest::OpenDesign => self.picker.pick(ctx),
            }
        }
        if !open {
            self.set_open(false);
        }
        if self.has_keyboard() && drawn {
            ctx.input_mut(|i| {
                i.events.retain(|event| {
                    !matches!(event, egui::Event::Key { .. }) || is_ui_zoom_key(event)
                })
            });
        }
    }

    /// Moves the keyboard by this frame's last pointer press (module docs). The press is read
    /// from the events, so a press released in the same frame still counts.
    fn follow_presses(&mut self, ctx: &egui::Context) {
        let pressed_at = ctx.input(|i| {
            i.events.iter().rev().find_map(|event| match event {
                egui::Event::PointerButton {
                    pos, pressed: true, ..
                } => Some(*pos),
                _ => None,
            })
        });
        let Some(pos) = pressed_at else {
            return;
        };
        let layer = ctx.layer_id_at(pos);
        match layer {
            Some(layer) if layer == window_layer() => self.keyboard = true,
            Some(layer)
                if matches!(
                    layer.order,
                    egui::Order::Foreground | egui::Order::Tooltip | egui::Order::Debug
                ) => {}
            // The linkage app's panels and canvas (a Background layer) take it, unless the press
            // grabs the window's edge, which egui lets the user do from just outside its frame
            // (another window there takes it).
            _ if layer.is_none_or(|layer| layer.order == egui::Order::Background)
                && on_resize_band(ctx, pos) =>
            {
                self.keyboard = true
            }
            _ => self.keyboard = false,
        }
    }

    /// The panel, once the window has been opened.
    #[cfg(test)]
    pub(crate) fn panel(&self) -> Option<&MagcouplingPanel> {
        self.panel.as_ref()
    }

    /// The panel's design, once the window has been opened.
    #[cfg(test)]
    pub(crate) fn design(&self) -> magcoupling::gui::session::Design {
        self.panel().expect("the window was opened").design()
    }

    /// Loads `design` into the panel as a design file, once the window has been opened (the next
    /// frame makes it an undo step of the panel).
    #[cfg(test)]
    pub(crate) fn load_design(&mut self, design: &magcoupling::gui::session::Design) {
        let json = magcoupling::gui::session::design_to_json(design);
        let panel = self.panel.as_mut().expect("the window was opened");
        panel.load_design_file(&json).expect("a valid design");
    }
}

/// Whether `event` is one of egui's own zoom keys (`UI_ZOOM_KEYS`), matched as egui matches them.
fn is_ui_zoom_key(event: &egui::Event) -> bool {
    let egui::Event::Key { key, modifiers, .. } = event else {
        return false;
    };
    UI_ZOOM_KEYS
        .iter()
        .any(|zoom| zoom.logical_key == *key && modifiers.matches_logically(zoom.modifiers))
}

/// Whether `pos` is on the band around the window where egui lets the user grab its edge
/// (`resize_grab_radius_side` out from it) or a corner (`resize_grab_radius_corner` around it) to
/// resize it. The band lies outside the window's area, so `Context::layer_id_at` does not see it.
fn on_resize_band(ctx: &egui::Context, pos: egui::Pos2) -> bool {
    let Some(rect) = ctx.memory(|m| m.area_rect(window_id())) else {
        return false;
    };
    let style = ctx.style();
    let grab = &style.interaction;
    let corner = egui::Vec2::splat(2.0 * grab.resize_grab_radius_corner);
    rect.expand(grab.resize_grab_radius_side).contains(pos)
        || [
            rect.left_top(),
            rect.right_top(),
            rect.left_bottom(),
            rect.right_bottom(),
        ]
        .into_iter()
        .any(|at| egui::Rect::from_center_size(at, corner).contains(pos))
}

/// The native save dialog's filter for a file the panel saves: its design file and results
/// exports are JSON or CSV.
fn save_filter(file_name: &str) -> FileFilter {
    match file_name.rsplit_once('.').map(|(_, extension)| extension) {
        Some("json") => FileFilter {
            label: "JSON",
            extensions: &["json"],
        },
        Some("csv") => FileFilter {
            label: "CSV",
            extensions: &["csv"],
        },
        _ => FileFilter {
            label: "All files",
            extensions: &["*"],
        },
    }
}

/// What the panel shows after a save: the message, the error, or nothing when the user cancelled
/// the dialog.
fn report_of(outcome: DownloadOutcome) -> Option<Result<String, String>> {
    match outcome {
        DownloadOutcome::Saved(message) => Some(Ok(message)),
        DownloadOutcome::Cancelled => None,
        DownloadOutcome::Failed(error) => Some(Err(error)),
    }
}

/// Hands a picked design file's text to the panel (a refused file shows why there and changes
/// nothing); a file that could not be read is reported in the panel.
fn open_picked_file(panel: &mut MagcouplingPanel, picked: Result<String, String>) {
    match picked {
        Ok(text) => {
            if let Err(error) = panel.load_design_file(&text) {
                log::warn!("magcoupling: {error}");
            }
        }
        Err(error) => panel.report(Err(error)),
    }
}

/// A design file the user is picking ("Load design"). Natively rfd's dialog answers at once; on
/// the web rfd's file picker answers later, so the text lands in an inbox the window reads each
/// frame. The pattern of `magcoupling-rs/src/app/files.rs` (the standalone app's picker) with
/// the build cases of `export/download.rs`.
#[derive(Default)]
struct DesignPicker {
    inbox: Rc<RefCell<Option<Result<String, String>>>>,
}

impl DesignPicker {
    /// Shows the file dialog (JSON files); the text arrives in [`DesignPicker::take`].
    #[cfg(feature = "native")]
    fn pick(&mut self, _ctx: &egui::Context) {
        let picked = rfd::FileDialog::new()
            .add_filter("Design", &["json"])
            .pick_file();
        if let Some(path) = picked {
            *self.inbox.borrow_mut() = Some(
                std::fs::read_to_string(&path)
                    .map_err(|error| format!("Could not read {}: {error}", path.display())),
            );
        }
    }

    /// Shows the browser's file picker (JSON files); the text arrives in [`DesignPicker::take`]
    /// once the browser hands the file over.
    #[cfg(target_arch = "wasm32")]
    fn pick(&mut self, ctx: &egui::Context) {
        let inbox = Rc::clone(&self.inbox);
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let dialog = rfd::AsyncFileDialog::new().add_filter("Design", &["json"]);
            if let Some(file) = dialog.pick_file().await {
                let text = String::from_utf8(file.read().await)
                    .map_err(|_| "the file is not UTF-8 text".to_owned());
                *inbox.borrow_mut() = Some(text);
                ctx.request_repaint();
            }
        });
    }

    /// Without a file dialog (a desktop build without the `native` feature): says so.
    #[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
    fn pick(&mut self, _ctx: &egui::Context) {
        *self.inbox.borrow_mut() = Some(Err(
            "Loading a design file needs the desktop build (feature native) or the web build"
                .to_owned(),
        ));
    }

    /// The picked file's text (or why it could not be read), once.
    fn take(&mut self) -> Option<Result<String, String>> {
        self.inbox.borrow_mut().take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
    /// Index into SampleMechanism::all() for the current demo sample.
    demo_sample_index: usize,
}
````

with:

````rust
    /// Index into SampleMechanism::all() for the current demo sample.
    demo_sample_index: usize,
    /// Tools → Magnetic coupling: the calculator window, its state independent of the model.
    calculator: calculator_window::CalculatorWindow,
}
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
            demo_timer: 0.0,
            demo_sample_index: 0,
        }
    }
````

with:

````rust
            demo_timer: 0.0,
            demo_sample_index: 0,
            calculator: calculator_window::CalculatorWindow::default(),
        }
    }
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
        // ── Keyboard shortcuts ────────────────────────────────────────
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z) && !i.modifiers.shift) {
            self.state.undo();
        }
        if ctx.input(|i| {
            i.modifiers.command
                && (i.key_pressed(egui::Key::Y)
                    || (i.key_pressed(egui::Key::Z) && i.modifiers.shift))
        }) {
            self.state.redo();
        }
        // Ctrl+S — quick save to last path, or Save As if no path yet.
        #[cfg(feature = "native")]
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S) && !i.modifiers.shift) {
            if let Some(path) = self.state.last_save_path.clone() {
                if let Err(e) = self.state.save_to_file(&path) {
                    log::error!("Quick save failed: {}", e);
                }
            } else if let Some(path) = rfd::FileDialog::new()
                .add_filter("JSON", &["json"])
                .set_file_name("mechanism.json")
                .save_file()
            {
                if let Err(e) = self.state.save_to_file(&path) {
                    log::error!("Save failed: {}", e);
                }
            }
        }
        // Ctrl+Shift+S — Save As (always shows file dialog).
        #[cfg(feature = "native")]
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S) && i.modifiers.shift) {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("JSON", &["json"])
                .set_file_name("mechanism.json")
                .save_file()
            {
                if let Err(e) = self.state.save_to_file(&path) {
                    log::error!("Save As failed: {}", e);
                }
            }
        }

        // Ctrl+N — New empty mechanism.
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::N)) {
            self.state.new_empty_mechanism();
        }

        // Ctrl+V — hint that image paste is not yet supported.
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::V)) {
            self.state.status_message =
                Some("Image paste not supported yet \u{2014} drag & drop an image onto the canvas instead.".to_string());
            self.state.status_message_time = 4.0;
        }

````

with:

````rust
        // ── Tools → Magnetic coupling ─────────────────────────────────
        // Before anything else reads the keyboard: while the calculator window has the
        // keyboard, it takes the frame's key events (calculator_window module docs).
        self.calculator.show(ctx);

        // ── Keyboard shortcuts ────────────────────────────────────────
        handle_keyboard_shortcuts(ctx, &mut self.state);

````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
        // --- Menu bar ---
        menu_bar::draw_menu_bar(ctx, &mut self.state, &self.sample_thumbnails);
````

with:

````rust
        // --- Menu bar ---
        menu_bar::draw_menu_bar(ctx, &mut self.state, &self.sample_thumbnails, &mut self.calculator);
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
                                ("Right-click canvas", "Add Ground Pivot / Body"),
                            ];
````

with:

````rust
                                ("Right-click canvas", "Add Ground Pivot / Body"),
                                ("Magnetic coupling window", "From its opening or a click in it, keys go to the calculator (Ctrl+Z / Ctrl+Y undo there; Ctrl+Plus / Ctrl+Minus still zoom) until you click the mechanism, its panels or the menu bar"),
                            ];
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
// ── Delete shortcut ─────────────────────────────────────────────────────────
````

with:

````rust
// ── Keyboard shortcuts ──────────────────────────────────────────────────────

/// Ctrl+Z undo, Ctrl+Y or Ctrl+Shift+Z redo, Ctrl+S save, Ctrl+Shift+S save as (native),
/// Ctrl+N new mechanism, Ctrl+V the paste hint. Runs after the calculator window, which takes
/// the key events while it has the keyboard.
fn handle_keyboard_shortcuts(ctx: &egui::Context, state: &mut AppState) {
    if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z) && !i.modifiers.shift) {
        state.undo();
    }
    if ctx.input(|i| {
        i.modifiers.command
            && (i.key_pressed(egui::Key::Y)
                || (i.key_pressed(egui::Key::Z) && i.modifiers.shift))
    }) {
        state.redo();
    }
    // Ctrl+S — quick save to last path, or Save As if no path yet.
    #[cfg(feature = "native")]
    if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S) && !i.modifiers.shift) {
        if let Some(path) = state.last_save_path.clone() {
            if let Err(e) = state.save_to_file(&path) {
                log::error!("Quick save failed: {}", e);
            }
        } else if let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .set_file_name("mechanism.json")
            .save_file()
        {
            if let Err(e) = state.save_to_file(&path) {
                log::error!("Save failed: {}", e);
            }
        }
    }
    // Ctrl+Shift+S — Save As (always shows file dialog).
    #[cfg(feature = "native")]
    if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S) && i.modifiers.shift) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .set_file_name("mechanism.json")
            .save_file()
        {
            if let Err(e) = state.save_to_file(&path) {
                log::error!("Save As failed: {}", e);
            }
        }
    }

    // Ctrl+N — New empty mechanism.
    if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::N)) {
        state.new_empty_mechanism();
    }

    // Ctrl+V — hint that image paste is not yet supported.
    if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::V)) {
        state.status_message =
            Some("Image paste not supported yet \u{2014} drag & drop an image onto the canvas instead.".to_string());
        state.status_message_time = 4.0;
    }
}

// ── Delete shortcut ─────────────────────────────────────────────────────────
````

In `linkage-sim-rs/src/gui/menu_bar.rs`, replace:

````rust
use super::state::{AppState, AngleUnit, LengthUnit};
use super::samples::SampleMechanism;
use super::{dxf_import, export, tutorial};

pub(crate) fn draw_menu_bar(
    ctx: &egui::Context,
    state: &mut AppState,
    sample_thumbnails: &HashMap<SampleMechanism, egui::TextureHandle>,
) {
````

with:

````rust
use super::calculator_window::{self, CalculatorWindow};
use super::state::{AppState, AngleUnit, LengthUnit};
use super::samples::SampleMechanism;
use super::{dxf_import, export, tutorial};

pub(crate) fn draw_menu_bar(
    ctx: &egui::Context,
    state: &mut AppState,
    sample_thumbnails: &HashMap<SampleMechanism, egui::TextureHandle>,
    calculator: &mut CalculatorWindow,
) {
````

In `linkage-sim-rs/src/gui/menu_bar.rs`, replace:

````rust
                image_resp.response.on_hover_text("Background image import and controls");
            });
        });
}
````

with:

````rust
                image_resp.response.on_hover_text("Background image import and controls");

                // ── Tools menu ──────────────────────────────────────────
                let tools_resp = ui.menu_button("Tools", |ui| {
                    let mut open = calculator.is_open();
                    if ui
                        .checkbox(&mut open, calculator_window::TITLE)
                        .on_hover_text("The magnetic slip coupling calculator in a window of its own. Its design is separate from the mechanism and is kept while the window is closed.")
                        .changed()
                    {
                        calculator.set_open(open);
                        ui.ctx().request_repaint();
                        ui.close();
                    }
                });
                tools_resp.response.on_hover_text("Calculators beside the mechanism");
            });
        });
}
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --lib gui::calculator_window 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --lib gui::menu_bar 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --lib gui::tests 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --lib 2>&1 | grep -E "test result"
git -C C:/Users/Cole/source/repos/lsim-mag-m5 checkout -- docs/chebyshev_lambda
```

Expected: `test result: ok. 18 passed` (`gui::calculator_window`), `test result: ok. 2 passed` (`gui::menu_bar`), `test result: ok. 12 passed` (`gui::tests`: the earlier tests and the 3 new), then `test result: ok. 949 passed` for the library (Task 0's 926 plus 23).

- [ ] **Step 5: Format, lint and the wasm check**

Run:

```bash
rustfmt --check --edition 2024 C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/src/gui/calculator_window.rs && echo fmt-ok
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --all-targets 2>&1 | grep -E "generated"
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --all-targets 2>&1 | grep -cE "^\s+--> src.gui.calculator_window"
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && source scripts/magcoupling_shipped.sh && cargo check "${LINKAGE_WEB_ARGS[@]}" 2>&1 | grep -E "^error|Finished"
```

Expected: `fmt-ok` (never run `cargo fmt` here: Global Constraints); the three counts of Global Constraints (`(lib) generated 279 warnings`, `(test "property_tests") generated 3 warnings`, `(lib test) generated 295 warnings (268 duplicates)`); `0` (no warning in the new file); a `Finished` line and no `error` (the window compiles for the web: `DesignPicker` on rfd's async picker).

- [ ] **Step 6: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task2.log 2>&1; echo "exit=$?"; tail -1 C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task2.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task2.log
git -C C:/Users/Cole/source/repos/lsim-mag-m5 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`, `GATE PASS`, `0`.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m5 add linkage-sim-rs/src/gui/calculator_window.rs linkage-sim-rs/src/gui/mod.rs linkage-sim-rs/src/gui/menu_bar.rs linkage-sim-rs/src/gui/test_support.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m5 commit -F - <<'EOF'
feat(linkage-sim-rs): Tools > Magnetic coupling opens the calculator in a window

gui::calculator_window::CalculatorWindow holds the magcoupling panel, created
on the first opening and kept, with its design and undo history, while the
window is closed (decision M5-3); nothing in it touches the linkage model. The
window has the keyboard from its opening or a press on it (its resize band
included) until a press on the linkage app, its menu-bar buttons included (an
open menu's items or a drop-down list move nothing, decision M5-5): only then
are the panel's undo keys on, and the window removes the frame's key events
but egui's zoom keys so no linkage shortcut or canvas key runs on them. On a
screen without room for its frame the window skips the frame, so egui keeps
its size. update shows the window before any key reader (the shortcut block
moved to handle_keyboard_shortcuts; a source test pins the order). The window
saves through export::download and loads a design through rfd (decision
M5-6), opens at (80, 100) with 1100 x 700 contents kept on the screen and in
front of the welcome screen (M5-2). The Tools menu's Magnetic coupling
checkbox toggles it (M5-1).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m5 status --short
```

Expected: one new commit; `status --short` prints nothing.

---

### Task 3: The web entry, the window title (BL-037) and the smoke

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `linkage-sim-rs/src/gui/calculator_window.rs` (the constants and `magcoupling_share_base` after `TITLE` and `UI_ZOOM_KEYS`; the `share_base` field; `set_open`; `set_share_base`; three tests)
- Modify: `linkage-sim-rs/src/gui/mod.rs` (the `pub use`; `sent_title`; `open_magcoupling`, `set_magcoupling_share_base`; the title in `update`; `window_title`, `send_title`; a test)
- Modify: `linkage-sim-rs/src/bin/linkage_web.rs` (`?tool=magcoupling`, the share base; `url_param`, used by `extract_url_mechanism_param` too)
- Modify: `.claude/workflows/gui-smoke.js` (the linkage step)

**Interfaces:**
- Consumes: Task 2's `CalculatorWindow` (`set_open`, `panel`), `LinkageApp::calculator`, the tests' helpers in `calculator_window.rs` (`opened`, `frame`) and `gui/mod.rs`; `magcoupling::gui::session::PUBLIC_BASE_URL` (`"https://linkage.colesorkness.com/magcoupling/"`); `MagcouplingPanel::set_share_base(&mut self, impl Into<String>)`, `share_link(&self) -> String`; `magcoupling::gui::readouts::REGISTRY_LOG_PREFIX` (`"magcoupling explorer: "`).
- Produces: `pub const TOOL_PARAM: &str = "tool"`, `pub const TOOL_MAGCOUPLING: &str = "magcoupling"`, `pub const MAGCOUPLING_PAGE_PATH: &str = "/magcoupling/"`, `pub fn magcoupling_share_base(origin: &str) -> String` (re-exported from `linkage_sim_rs::gui` with the two parameters); `CalculatorWindow::set_share_base(&mut self, impl Into<String>)`; `LinkageApp::open_magcoupling(&mut self)`, `LinkageApp::set_magcoupling_share_base(&mut self, impl Into<String>)`; in `gui/mod.rs` `fn window_title(state: &AppState) -> String`, `fn send_title(ctx: &egui::Context, sent: &mut Option<String>, title: String)`; in `gui-smoke.js` `const LINKAGE_TOOL_QUERY = '?tool=magcoupling'` and the linkage step's result fields `magcoupling_window`, `title_warning`.

- [ ] **Step 1: Write the failing tests**

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
    use magcoupling::gui::session::{Design, design_to_json};
````

with:

````rust
    use magcoupling::gui::session::{Design, PUBLIC_BASE_URL, design_to_json};
````

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
        assert_eq!(keys, [egui::Key::Z]);
    }

    #[test]
    fn saved_files_get_the_filter_of_their_type() {
````

with:

````rust
        assert_eq!(keys, [egui::Key::Z]);
    }

    #[test]
    fn share_links_point_at_the_base_set_before_or_after_the_first_opening() {
        let mut window = CalculatorWindow::default();
        window.set_share_base("http://localhost:8080/magcoupling/");
        window.set_open(true);
        let link = window.panel().expect("opened").share_link();
        assert!(
            link.starts_with("http://localhost:8080/magcoupling/?m="),
            "{link}"
        );
        window.set_share_base("http://127.0.0.1:9000/magcoupling/");
        let link = window.panel().expect("opened").share_link();
        assert!(
            link.starts_with("http://127.0.0.1:9000/magcoupling/?m="),
            "{link}"
        );
    }

    #[test]
    fn gui_smoke_opens_the_window_with_the_tool_parameter() {
        // .claude/workflows/gui-smoke.js's linkage step opens the linkage app with this query and
        // looks for the equation registry's log line, which the panel logs when it is created.
        let script = include_str!("../../../.claude/workflows/gui-smoke.js").replace("\r\n", "\n");
        let query = format!("const LINKAGE_TOOL_QUERY = '?{TOOL_PARAM}={TOOL_MAGCOUPLING}'");
        assert!(script.contains(&query), "gui-smoke.js has no {query}");
        // The linkage step's own check (the /magcoupling/ step names the prefix too).
        let check = format!(
            "magcoupling_window=true only if a console message contains \"{}\"",
            magcoupling::gui::readouts::REGISTRY_LOG_PREFIX
        );
        assert!(script.contains(&check), "gui-smoke.js has no {check}");
    }

    #[test]
    fn the_production_origin_gives_the_calculator_s_public_address() {
        assert_eq!(
            magcoupling_share_base("https://linkage.colesorkness.com"),
            PUBLIC_BASE_URL
        );
    }

    #[test]
    fn saved_files_get_the_filter_of_their_type() {
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
            let at = update.find(reader).unwrap_or_else(|| panic!("update calls {reader}"));
            assert!(show < at, "{reader} comes before the calculator window");
        }
    }
}
````

with:

````rust
            let at = update.find(reader).unwrap_or_else(|| panic!("update calls {reader}"));
            assert!(show < at, "{reader} comes before the calculator window");
        }
    }

    /// The window titles egui was asked to send in a frame.
    fn title_commands(output: &egui::FullOutput) -> Vec<String> {
        output.viewport_output.get(&egui::ViewportId::ROOT).map_or_else(Vec::new, |viewport| {
            viewport
                .commands
                .iter()
                .filter_map(|command| match command {
                    egui::ViewportCommand::Title(title) => Some(title.clone()),
                    _ => None,
                })
                .collect()
        })
    }

    #[test]
    fn the_window_title_is_sent_only_when_it_changes() {
        let ctx = egui::Context::default();
        let mut state = AppState::default();
        let mut sent = None;
        let frame = |state: &AppState, sent: &mut Option<String>| {
            let output = ctx.run(egui::RawInput::default(), |ctx| send_title(ctx, sent, window_title(state)));
            title_commands(&output)
        };
        assert_eq!(frame(&state, &mut sent), ["Linkage Simulator"]);
        for _ in 0..3 {
            assert!(frame(&state, &mut sent).is_empty(), "an idle frame sends no title");
        }
        state.dirty = true;
        assert_eq!(frame(&state, &mut sent), ["Linkage Simulator \u{2014} unsaved*"]);
        assert!(frame(&state, &mut sent).is_empty());
        state.last_save_path = Some(std::path::PathBuf::from("lift.json"));
        assert_eq!(frame(&state, &mut sent), ["Linkage Simulator \u{2014} lift.json*"]);
    }
}
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --lib 2>&1 | grep -E "^error"
```

Expected: ``error[E0599]: no method named `set_share_base` found for struct `gui::calculator_window::CalculatorWindow` in the current scope`` (twice), ``error[E0425]: cannot find value `TOOL_PARAM` in this scope``, the same for `TOOL_MAGCOUPLING`, ``cannot find function `window_title` in this scope``, the same for `send_title` and `magcoupling_share_base`, and ``error: could not compile `linkage-sim-rs` (lib test) due to 7 previous errors; 22 warnings emitted``.

- [ ] **Step 3: Write the implementation**

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
/// The window's title and the Tools menu item.
pub const TITLE: &str = "Magnetic coupling";
````

with:

````rust
/// The window's title and the Tools menu item.
pub const TITLE: &str = "Magnetic coupling";

/// The URL query parameter that opens a tool when the web app starts (decision M5-4):
/// `?tool=magcoupling`. gui-smoke's linkage step opens the window with it.
pub const TOOL_PARAM: &str = "tool";

/// [`TOOL_PARAM`]'s value that opens this window.
pub const TOOL_MAGCOUPLING: &str = "magcoupling";

/// The path of the calculator's own page next to the linkage app (`web/magcoupling/`).
pub const MAGCOUPLING_PAGE_PATH: &str = "/magcoupling/";
````

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
    egui::gui_zoom::kb_shortcuts::ZOOM_RESET,
];
````

with:

````rust
    egui::gui_zoom::kb_shortcuts::ZOOM_RESET,
];

/// The address the calculator's share links point at on the web app served from `origin`: the
/// calculator's own page on the same server.
pub fn magcoupling_share_base(origin: &str) -> String {
    format!("{origin}{MAGCOUPLING_PAGE_PATH}")
}
````

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
    raise: bool,
    picker: DesignPicker,
````

with:

````rust
    raise: bool,
    /// The address the panel's share links point at, applied when the panel is created.
    share_base: Option<String>,
    picker: DesignPicker,
````

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
        if open && self.panel.is_none() {
            self.panel = Some(MagcouplingPanel::new());
        }
````

with:

````rust
        if open && self.panel.is_none() {
            let mut panel = MagcouplingPanel::new();
            if let Some(base) = &self.share_base {
                panel.set_share_base(base.clone());
            }
            self.panel = Some(panel);
        }
````

In `linkage-sim-rs/src/gui/calculator_window.rs`, replace:

````rust
    pub fn has_keyboard(&self) -> bool {
        self.open && self.keyboard
    }
````

with:

````rust
    pub fn has_keyboard(&self) -> bool {
        self.open && self.keyboard
    }

    /// Sets the address the panel's share links point at (the web app: the calculator's own page
    /// on the same server, [`magcoupling_share_base`]).
    pub fn set_share_base(&mut self, base: impl Into<String>) {
        let base = base.into();
        if let Some(panel) = &mut self.panel {
            panel.set_share_base(base.clone());
        }
        self.share_base = Some(base);
    }
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
pub use state::file_io::{decode_mechanism_from_url, encode_mechanism_for_url};
````

with:

````rust
pub use state::file_io::{decode_mechanism_from_url, encode_mechanism_for_url};
pub use calculator_window::{TOOL_MAGCOUPLING, TOOL_PARAM, magcoupling_share_base};
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
    calculator: calculator_window::CalculatorWindow,
}
````

with:

````rust
    calculator: calculator_window::CalculatorWindow,
    /// The window title last sent to the native window (BL-037: sent only when it changes).
    sent_title: Option<String>,
}
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
            calculator: calculator_window::CalculatorWindow::default(),
        }
    }
````

with:

````rust
            calculator: calculator_window::CalculatorWindow::default(),
            sent_title: None,
        }
    }

    /// Opens Tools → Magnetic coupling (the web entry's `?tool=magcoupling`).
    pub fn open_magcoupling(&mut self) {
        self.calculator.set_open(true);
    }

    /// Sets the address the calculator's share links point at (the web entry: the calculator's
    /// own page on the same server, [`magcoupling_share_base`]).
    pub fn set_magcoupling_share_base(&mut self, base: impl Into<String>) {
        self.calculator.set_share_base(base);
    }
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
        // ── Update window title to show filename and dirty state ──────
        let title = if let Some(ref path) = self.state.last_save_path {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if self.state.dirty {
                format!("Linkage Simulator \u{2014} {}*", name)
            } else {
                format!("Linkage Simulator \u{2014} {}", name)
            }
        } else if self.state.dirty {
            "Linkage Simulator \u{2014} unsaved*".to_string()
        } else {
            "Linkage Simulator".to_string()
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
````

with:

````rust
        // ── Update window title to show filename and dirty state ──────
        send_title(ctx, &mut self.sent_title, window_title(&self.state));
````

In `linkage-sim-rs/src/gui/mod.rs`, replace:

````rust
// ── Keyboard shortcuts ──────────────────────────────────────────────────────

/// Ctrl+Z undo, Ctrl+Y or Ctrl+Shift+Z redo, Ctrl+S save, Ctrl+Shift+S save as (native),
````

with:

````rust
// ── Window title ────────────────────────────────────────────────────────────

/// The native window's title: the file name and whether there are unsaved changes.
fn window_title(state: &AppState) -> String {
    if let Some(ref path) = state.last_save_path {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if state.dirty {
            format!("Linkage Simulator \u{2014} {}*", name)
        } else {
            format!("Linkage Simulator \u{2014} {}", name)
        }
    } else if state.dirty {
        "Linkage Simulator \u{2014} unsaved*".to_string()
    } else {
        "Linkage Simulator".to_string()
    }
}

/// Sends `title` to the native window when it differs from the last title sent (`sent`), so an
/// idle frame sends nothing (BL-037). The web backend does not implement the command and logs a
/// warning for each one, so the web build never sends it (the page has its own `<title>`).
fn send_title(ctx: &egui::Context, sent: &mut Option<String>, title: String) {
    if cfg!(target_arch = "wasm32") || sent.as_deref() == Some(title.as_str()) {
        return;
    }
    ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
    *sent = Some(title);
}

// ── Keyboard shortcuts ──────────────────────────────────────────────────────

/// Ctrl+Z undo, Ctrl+Y or Ctrl+Shift+Z redo, Ctrl+S save, Ctrl+Shift+S save as (native),
````

In `linkage-sim-rs/src/bin/linkage_web.rs`, replace:

````rust
    // ── Check for ?m= URL parameter (shared mechanism) ───────────────
    let shared_mechanism_json = extract_url_mechanism_param();
````

with:

````rust
    // ── Check for ?m= URL parameter (shared mechanism) ───────────────
    let shared_mechanism_json = extract_url_mechanism_param();

    // ── ?tool=magcoupling opens Tools → Magnetic coupling at start ────
    let open_magcoupling =
        url_param(linkage_sim_rs::gui::TOOL_PARAM).as_deref() == Some(linkage_sim_rs::gui::TOOL_MAGCOUPLING);
    // Share links made in that window open the calculator's own page on this server.
    let magcoupling_share_base = web_sys::window()
        .and_then(|window| window.location().origin().ok())
        .map(|origin| linkage_sim_rs::gui::magcoupling_share_base(&origin));
````

In `linkage-sim-rs/src/bin/linkage_web.rs`, replace:

````rust
                // If a ?m= parameter was found, load the shared mechanism.
                if let Some(json_str) = shared_mechanism_json {
                    app.load_shared_mechanism(&json_str);
                }
                Ok(Box::new(app))
````

with:

````rust
                // If a ?m= parameter was found, load the shared mechanism.
                if let Some(json_str) = shared_mechanism_json {
                    app.load_shared_mechanism(&json_str);
                }
                if let Some(base) = magcoupling_share_base {
                    app.set_magcoupling_share_base(base);
                }
                if open_magcoupling {
                    app.open_magcoupling();
                }
                Ok(Box::new(app))
````

In `linkage-sim-rs/src/bin/linkage_web.rs`, replace:

````rust
    use linkage_sim_rs::gui::decode_mechanism_from_url;

    let window = web_sys::window()?;
    let location = window.location();
    let search = location.search().ok()?;
    if search.is_empty() {
        return None;
    }

    // Parse URL search params. web_sys::UrlSearchParams expects the raw
    // search string including the leading '?'.
    let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
    let encoded = params.get("m")?;
    if encoded.is_empty() {
        return None;
    }

    match decode_mechanism_from_url(&encoded) {
````

with:

````rust
    use linkage_sim_rs::gui::decode_mechanism_from_url;

    let encoded = url_param("m")?;

    match decode_mechanism_from_url(&encoded) {
````

In `linkage-sim-rs/src/bin/linkage_web.rs`, replace:

````rust
            log::error!("Failed to decode shared mechanism from URL: {}", e);
            None
        }
    }
}
````

with:

````rust
            log::error!("Failed to decode shared mechanism from URL: {}", e);
            None
        }
    }
}

/// The value of the URL query parameter `name`, if the page's address has it and it is not empty.
#[cfg(target_arch = "wasm32")]
fn url_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    if search.is_empty() {
        return None;
    }
    // web_sys::UrlSearchParams takes the raw search string, leading '?' included.
    let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
    params.get(name).filter(|value| !value.is_empty())
}
````

- [ ] **Step 4: Run the tests: the smoke test still fails**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --lib gui::calculator_window 2>&1 | grep -E "test result|FAILED|gui-smoke.js has no"
```

Expected: `test gui::calculator_window::tests::gui_smoke_opens_the_window_with_the_tool_parameter ... FAILED`, its message `gui-smoke.js has no const LINKAGE_TOOL_QUERY = '?tool=magcoupling'`, and `test result: FAILED. 20 passed; 1 failed` (the workflow does not open the window yet; every other test of the module passes).

- [ ] **Step 5: The smoke's linkage step opens the window**

In `.claude/workflows/gui-smoke.js`, replace:

````js
  description: 'Smoke-test the locally served WASM builds (the linkage app and /magcoupling/): loads, canvas present, console clean',
````

with:

````js
  description: 'Smoke-test the locally served WASM builds (the linkage app with its calculator window, and /magcoupling/): loads, canvas present, console clean',
````

In `.claude/workflows/gui-smoke.js`, replace:

````js
const SMOKE_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
````

with:

````js
const SMOKE_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'magcoupling_window', 'title_warning'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    magcoupling_window: { type: 'boolean' },
    title_warning: { type: 'boolean' },
````

In `.claude/workflows/gui-smoke.js`, replace:

````js
const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
````

with:

````js
// The query that opens the linkage app with Tools -> Magnetic coupling open (decision M5-4).
// linkage-sim-rs/src/gui/calculator_window.rs (gui_smoke_opens_the_window_with_the_tool_parameter)
// checks it against TOOL_PARAM and TOOL_MAGCOUPLING, so it cannot go stale unnoticed.
const LINKAGE_TOOL_QUERY = '?tool=magcoupling'

const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
````

In `.claude/workflows/gui-smoke.js`, replace:

````js
Steps: navigate to ${url}; wait 5 seconds for WASM init; snapshot the page and confirm a <canvas> element exists; collect console messages; take a screenshot and judge whether it shows a rendered app (menu bar / toolbar / panels visible, any theme) vs a blank page; close the browser.
passed=true only if: page loaded, canvas present, zero console messages of type error (warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (caller must have run scripts/serve_web.sh).`,
    { label: 'gui-smoke', phase: 'Smoke', schema: SMOKE_SCHEMA, model: 'sonnet' },
  ) || { passed: false, canvas_present: false, console_errors: [], notes: 'smoke agent returned no result (agent error)' }
````

with:

````js
Steps: navigate to ${url}; wait 5 seconds for WASM init; snapshot the page and confirm a <canvas> element exists; collect the console messages at level "info" (it includes errors and warnings); take a screenshot and judge whether it shows a rendered app (menu bar / toolbar / panels visible, any theme) vs a blank page.
Then the calculator window: navigate to ${url}/${LINKAGE_TOOL_QUERY} (the linkage app with Tools -> Magnetic coupling open at start); wait 5 seconds; collect the console messages at level "info" again; take a screenshot. magcoupling_window=true only if a console message contains "magcoupling explorer: " (logged when the window's calculator is created) and the screenshot shows a window titled "Magnetic coupling" over the linkage app, holding the calculator (inputs on the left, dashboard on the right, the geometry view in the middle). Close the browser.
title_warning=true if any console message on either page contains "Unhandled egui viewport command: Title" (backlog BL-037: there must be none).
passed=true only if: both pages loaded, canvas present, magcoupling_window, title_warning=false, and zero console messages of type error on either page (other warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (caller must have run scripts/serve_web.sh).`,
    { label: 'gui-smoke', phase: 'Smoke', schema: SMOKE_SCHEMA, model: 'sonnet' },
  ) || { passed: false, canvas_present: false, console_errors: [], magcoupling_window: false, title_warning: false, notes: 'smoke agent returned no result (agent error)' }
````

- [ ] **Step 6: Run the tests to verify they pass; lint and the wasm checks**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --lib gui::calculator_window 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --lib gui::tests::the_window_title 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --lib 2>&1 | grep -E "test result"
git -C C:/Users/Cole/source/repos/lsim-mag-m5 checkout -- docs/chebyshev_lambda
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/magcoupling-rs/Cargo.toml --features app --lib app::tests 2>&1 | grep -E "test result"
node -e "const s=require('fs').readFileSync('C:/Users/Cole/source/repos/lsim-mag-m5/.claude/workflows/gui-smoke.js','utf8'); new Function('args','agent','return (async () => {' + s.replace(/^export const meta/m, 'const meta') + '})()'); console.log('gui-smoke.js parses')"
rustfmt --check --edition 2024 C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/src/gui/calculator_window.rs && echo fmt-ok
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/Cargo.toml --all-targets 2>&1 | grep -E "generated"
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && source scripts/magcoupling_shipped.sh && cargo check "${LINKAGE_WEB_ARGS[@]}" 2>&1 | grep -E "^error|Finished"; cargo clippy "${LINKAGE_WEB_ARGS[@]}" 2>&1 | grep -E "generated|^error"; cargo clippy "${LINKAGE_WEB_ARGS[@]}" 2>&1 | grep -cE "^\s+--> src.(bin.linkage_web|gui.calculator_window)"
```

Expected: `test result: ok. 21 passed` (`gui::calculator_window`), `test result: ok. 1 passed` (the title test), `test result: ok. 953 passed` for the library; the magcoupling app tests `test result: ok. 5 passed` (their smoke test still reads its pinned link and design file from the edited `gui-smoke.js`); `gui-smoke.js parses` (the workflow script compiles as the Workflow tool wraps it); `fmt-ok`; the three clippy counts of Global Constraints; a `Finished` line and no `error` for the web build (`url_param`, `?tool=magcoupling` and the share base compile for wasm32); ``warning: `linkage-sim-rs` (lib) generated 268 warnings`` (followed by its `cargo clippy --fix` hint) and no `error` for its wasm32 clippy; `0`.

- [ ] **Step 7: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task3.log 2>&1; echo "exit=$?"; tail -1 C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task3.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task3.log
git -C C:/Users/Cole/source/repos/lsim-mag-m5 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`, `GATE PASS`, `0`.

- [ ] **Step 8: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m5 add linkage-sim-rs/src/gui/calculator_window.rs linkage-sim-rs/src/gui/mod.rs linkage-sim-rs/src/bin/linkage_web.rs .claude/workflows/gui-smoke.js
git -C C:/Users/Cole/source/repos/lsim-mag-m5 commit -F - <<'EOF'
feat(linkage-sim-rs): ?tool=magcoupling, share links to /magcoupling/, the title only on change

The web entry opens Tools > Magnetic coupling for ?tool=magcoupling (decision
M5-4; TOOL_PARAM, TOOL_MAGCOUPLING) and points the window's share links at
/magcoupling/ on the same server (magcoupling_share_base, which gives the
calculator's public address for the production origin). url_param reads both
query parameters. The window title is sent only when it changes and never on
wasm32 (BL-037, decision M5-7: window_title, send_title). gui-smoke's linkage
step opens /?tool=magcoupling and checks the window, the equation registry's
log line and that no Title warning is logged; a test pins its query to the
constants.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m5 status --short
```

Expected: one new commit; `status --short` prints nothing.

---

### Task 4: Docs, the shipped bundles and the plan

**Model:** `sonnet` (doc and YAML edits, running commands; CLAUDE.md section 5). Step 4 needs the Playwright MCP tools (or the controller runs the `gui-smoke` workflow instead). The whole-branch review after this task runs on the session model.

**Files:**
- Modify: `README.md`, `magcoupling-rs/README.md`, `docs/FEATURES.md` (a Magnetic Coupling Calculator section under Implemented, not under Planned / Future's Engineering Tools), `docs/guides/WASM_DEPLOYMENT.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`
- Create: `docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md` (a copy of this plan)

**Interfaces:**
- Consumes: Tasks 1 to 3 (the names the docs cite: `gui::calculator_window`, `CalculatorWindow`, `handle_keyboard_shortcuts`, `send_title`, `TOOL_PARAM`, `LINKAGE_WEB_ARGS`, `assert_guard_trips`, `LINKAGE_TOOL_QUERY`); Task 0 Step 8's wasm sizes.
- Produces: the docs; BL-037 `status: fixed`; BL-038 (one design file picker for both hosts), BL-039 (the recovery prompt) and BL-040 (Tab focus) `status: open`; the plan in the repository.

- [ ] **Step 1: Update the docs**

In `README.md`, replace:

````markdown
- **Export**: PNG, SVG, GIF (ping-pong loop), DXF, CSV, HTML report with interactive Plotly charts
````

with:

````markdown
- **Export**: PNG, SVG, GIF (ping-pong loop), DXF, CSV, HTML report with interactive Plotly charts
- **Magnetic coupling calculator** (Tools > Magnetic coupling): the [`magcoupling-rs`](magcoupling-rs/README.md) calculator in a window beside the mechanism, its design kept apart from the mechanism and kept while the window is closed; the keys go to the window or the mechanism, whichever was clicked last; `?tool=magcoupling` opens it when the web app starts. The same calculator runs on its own at [`/magcoupling/`](https://linkage.colesorkness.com/magcoupling/)
````

In `README.md`, replace:

````markdown
│   │                       #   force_toolbar, tutorial, undo, error_panel
````

with:

````markdown
│   │                       #   force_toolbar, tutorial, undo, error_panel,
│   │                       #   calculator_window (Tools > Magnetic coupling)
````

In `magcoupling-rs/README.md`, replace:

````markdown
Not a Cargo workspace member: `linkage-sim-rs` will depend on it by path
(feature `gui`, M5).
````

with:

````markdown
Not a Cargo workspace member: `linkage-sim-rs` depends on it by path
(feature `gui`) for its Tools → Magnetic coupling window ([In the linkage app](#in-the-linkage-app-m5)).
````

In `magcoupling-rs/README.md`, replace:

````markdown
`app::CANVAS_ID`, which a test checks. `deploy-web.yml` runs
`build_magcoupling_web.sh` after the linkage build, so both bundles ship
````

with:

````markdown
`app::CANVAS_ID`, which a test checks. `deploy-web.yml` runs
`build_web.sh`, which builds the linkage bundle and then runs `build_magcoupling_web.sh`, so both bundles ship
````

In `magcoupling-rs/README.md`, replace:

````markdown
picker and checks `magcoupling: loaded a design file`, with zero console errors (a test decodes
the pinned link and the design file).
````

with:

````markdown
picker and checks `magcoupling: loaded a design file`, with zero console errors (a test decodes
the pinned link and the design file). Its linkage step opens the linkage app at `/?tool=magcoupling`
and checks the calculator window and the `magcoupling explorer: ` line there.
````

In `magcoupling-rs/README.md`, replace:

````markdown
cargo arguments once (`MAGCOUPLING_WEB_ARGS`, `MAGCOUPLING_NATIVE_ARGS`), and
`magcoupling_assert_shipped` reads cargo's `--message-format=json` record of
the units it compiled. It fails unless the shipped binary was compiled and no
magcoupling-rs unit has the feature. `build_magcoupling_web.sh` pipes its
release build through it, so the shipped path refuses such a bundle. Gate 10
runs the guard on the native and wasm32 builds, plus a negative control that
must trip.
````

with:

````markdown
cargo arguments once (`MAGCOUPLING_WEB_ARGS`, `MAGCOUPLING_NATIVE_ARGS`, and the linkage
app's `LINKAGE_WEB_ARGS`, `LINKAGE_NATIVE_ARGS`: its builds carry the panel), and
`magcoupling_assert_shipped` reads cargo's `--message-format=json` record of
the units it compiled. It fails unless the shipped binary was compiled and no
magcoupling-rs unit has the feature. `build_web.sh` and `build_magcoupling_web.sh` pipe their
release builds through it, so the shipped path refuses such a bundle. Gate 10
runs the guard on the native and wasm32 builds of both apps, plus a negative control per
shipped build that must trip.

### In the linkage app (M5)

Plan `docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md`. `linkage-sim-rs` depends on
this crate by path with feature `gui` (never `workbook-parity`: gate 10) and shows the panel in
its Tools → Magnetic coupling window, `linkage-sim-rs/src/gui/calculator_window.rs`, on the
desktop and on the web:

- The window is closed at start-up; its first opening creates the panel (the equation registry is
  built then, not at the linkage app's start-up), which keeps its design, undo history and views,
  open or closed, until the app quits. Nothing in it reads or writes the linkage model.
- The keys go to the part the user pressed last: the window from its opening or a press on it
  (or on the band just outside its frame where egui resizes it), the linkage app from a press on
  its panels, its canvas or its menu bar's buttons (a press on an open menu's items or on a
  drop-down list moves nothing). While the window has them, the panel's Ctrl+Z, Ctrl+Shift+Z and
  Ctrl+Y are on and the linkage app sees no key event but egui's own zoom keys (natively
  Ctrl+Plus, Ctrl+Minus and Ctrl+0 still zoom the whole UI; on the web the browser zooms the
  page): not its undo, save, delete, arrow nudge, F or Escape. Otherwise the panel's shortcuts
  are off (`set_keyboard_shortcuts(false)`) and the keys go to the linkage app. A window
  collapsed to its title bar draws no panel and takes no keys.
- Known edges (backlog BL-039, BL-040): the keys follow presses, not egui's keyboard focus (after
  Tab moves the focus into the other part, click there); and opened while the linkage app's
  "Recover Unsaved Work?" prompt shows, the window covers it (move or collapse the window).
- The window does the panel's requests: saving with the linkage app's download helper (a file
  dialog natively, a browser download on the web), "Load design" through rfd (the linkage web
  build carries rfd for the picker). Share links made there open `/magcoupling/` on the same
  server.
- `?tool=magcoupling` opens the window when the linkage web app starts (gui-smoke's linkage step).
````

In `docs/ai/02-system.yaml`, replace:

````yaml
    payload spec's hands-on checklist warns about it.

risks:
````

with:

````yaml
    payload spec's hands-on checklist warns about it.
  - "A magcoupling design file dropped on the linkage page is read as a mechanism (the status bar says the JSON import failed); the calculator window's Load design button loads it (decision M5-9)."
  - "The calculator window's keys follow presses, not egui's keyboard focus: after Tab moves the focus from one part into the other (the window and the linkage app), the keys stay where they were until a click there (decision M5-5; backlog BL-040)."
  - "Opened while the linkage app's 'Recover Unsaved Work?' prompt shows (?tool=magcoupling with an autosave on the web, or Tools > Magnetic coupling natively), the calculator window covers the prompt; move or collapse the window to answer it (decision M5-4; backlog BL-039)."

risks:
````

In `docs/ai/02-system.yaml`, replace:

````yaml
      M4-3 adds the equation explorer (typesetter, hover on every value, the docked Equation panel),
      the assumptions view and banner, the teaching notes with their diagrams, the material, part
      and grade pickers and the material warnings.
````

with:

````yaml
      M4-3 adds the equation explorer (typesetter, hover on every value, the docked Equation panel),
      the assumptions view and banner, the teaching notes with their diagrams, the material, part
      and grade pickers and the material warnings.
      M5 embeds the panel in the linkage app (linkage-sim-rs depends on the crate with feature gui
      and shows the panel in its Tools -> Magnetic coupling window, gui::calculator_window).
````

In `docs/ai/02-system.yaml`, replace:

````yaml
      - "workbook-parity never reaches a shipped build. The shipped cargo arguments live once, in linkage-sim-rs/scripts/magcoupling_shipped.sh. build_magcoupling_web.sh and gate 10 read cargo's --message-format=json record of the compiled units, and they fail if a magcoupling-rs unit has the feature. Gate 10 also runs a negative control that must trip."
````

with:

````yaml
      - "workbook-parity never reaches a shipped build, the linkage app's included (its native and web builds carry the panel). The shipped cargo arguments live once, in linkage-sim-rs/scripts/magcoupling_shipped.sh (MAGCOUPLING_* and LINKAGE_*). build_web.sh, build_magcoupling_web.sh and gate 10 read cargo's --message-format=json record of the compiled units, and they fail if a magcoupling-rs unit has the feature; deploy-web.yml builds through build_web.sh. Gate 10 also runs a negative control per shipped build that must trip."
````

In `docs/ai/02-system.yaml`, replace:

````yaml
      - "The panel does no I/O: saving and picking files are PanelRequests its host does (app::files natively and on the web; the linkage app in M5)."
````

with:

````yaml
      - "The panel does no I/O: saving and picking files are PanelRequests its host does (the standalone app: app::files; the linkage app: gui::calculator_window, export::download to save and rfd to pick, natively and on the web)."
````

In `docs/ai/02-system.yaml`, replace:

````yaml
  - delete_shortcut_ignores_focused_widgets — gui/mod.rs
````

with:

````yaml
  - "calculator_window_owns_the_keys_it_has — Tools -> Magnetic coupling (gui/calculator_window.rs): the window has the keyboard from its opening or a press on its layer (or on the band just outside its frame where egui resizes it) until a press on another Background or Middle layer (the linkage app's panels, canvas and menu-bar buttons, another window); a Foreground press (an open menu's items, a drop-down list) moves nothing. CalculatorWindow::show turns the panel's undo keys on only then and, after the panel drew (not while the window is collapsed to its title bar, nor on a screen without room for its frame), removes the frame's Event::Key events but egui's own zoom keys (UI_ZOOM_KEYS), so no linkage key reader runs on them. LinkageApp::update must call self.calculator.show(ctx) before every key reader (handle_keyboard_shortcuts, draw_menu_bar, handle_delete_shortcut, canvas::draw_canvas, any key_pressed): gui::tests::update_shows_the_calculator_window_before_anything_reads_the_keyboard checks the source order. The calculator's state lives in LinkageApp, never in AppState (independent of the model)."
  - delete_shortcut_ignores_focused_widgets — gui/mod.rs
````

In `docs/ai/02-system.yaml`, replace:

````yaml
  - "Windows MAX_PATH: a release build under a deep directory fails with LNK1104 (cannot open a build-script exe). Build from a short path (the worktree, or subst a drive letter)."
````

with:

````yaml
  - "Windows MAX_PATH: a release build under a deep directory fails with LNK1104 (cannot open a build-script exe). Build from a short path (the worktree, or subst a drive letter)."
  - "egui: Context::input_mut can remove events mid-frame; what was drawn before reads them, everything after sees none. InputState::key_pressed and consume_key count InputState::events, so removing the Event::Key events silences every later key reader while modifiers and keys_down stay (linkage gui::calculator_window)."
  - "Never drive LinkageApp::update in a headless test: canvas::draw_canvas writes grid.spacing_m every frame and tick_save_user_prefs then writes ~/.linkage-sim/preferences.json on the developer's machine. Test the pieces update calls, in its order (gui::tests keys_frame), and pin the order with a source test."
  - "Linkage tests read RawInput::modifiers (i.modifiers.command) while egui's consume_shortcut matches each key event's own modifiers: a headless Ctrl+Z for both sets both (gui::tests keys_frame)."
  - "egui 0.32: panels (CentralPanel, SidePanel, TopBottomPanel such as the menu bar) are Background areas, so Context::layer_id_at returns a Background layer over them. A window's resize band (Interaction::resize_grab_radius_side, 5 points out from its edges; resize_grab_radius_corner, 10 points around its corners) lies outside its area, and a press there makes egui move the edge to the pointer (linkage gui::calculator_window on_resize_band)."
  - "egui 0.32's Resize stores the size it clamps a window to: one frame on a 0 x 0 screen (the web backend passes the canvas size as it is; egui-winit sends no screen for a zero-size window) squeezes a window to its minimum at the screen's corner for good, unless the host skips the window that frame (linkage gui::calculator_window)."
  - "eframe 0.32's web backend sets Options::zoom_with_keyboard = false (the browser zooms the page and eframe follows it) and never prevents the browser's Ctrl+Plus, Ctrl+Minus or Ctrl+0, so egui's keyboard zoom (gui_zoom::kb_shortcuts) runs natively only; a headless test sees it because egui::Context defaults it on."
````

In `docs/ai/02-system.yaml`, replace:

````yaml
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim. M4 infrastructure in place (gui panel, app binaries, /magcoupling/ bundle, workbook-parity guard, gates 7-11). Addendum A-3 complete: equation explorer engine side, A3 traceability, A4 notes. M4-1 complete (inputs, dashboard, results table, session, sizing mode, theme). M4-2 complete (geometry view, five plots, clamp drawing and table, narrow results table). M4-3 complete (equation explorer: typesetter, hover, Equation panel; assumptions view and banner; teaching notes and diagrams; material, part and grade pickers; material warnings); next: M3 (live 3D fields), then M5 (embed)"
````

with:

````yaml
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim. M4 infrastructure in place (gui panel, app binaries, /magcoupling/ bundle, workbook-parity guard, gates 7-11). Addendum A-3 complete: equation explorer engine side, A3 traceability, A4 notes. M4-1 complete (inputs, dashboard, results table, session, sizing mode, theme). M4-2 complete (geometry view, five plots, clamp drawing and table, narrow results table). M4-3 complete (equation explorer: typesetter, hover, Equation panel; assumptions view and banner; teaching notes and diagrams; material, part and grade pickers; material warnings). M5 complete (the panel in the linkage app's Tools -> Magnetic coupling window, native and web; the keys follow the last press; gate 10 guards the linkage builds; BL-037 fixed); next: M3 (live 3D fields)"
````

In `docs/ai/03-structure.yaml`, replace:

````yaml
  shipped_builds: "linkage-sim-rs/scripts/magcoupling_shipped.sh (sourced by build_magcoupling_web.sh and gate.sh): MAGCOUPLING_WEB_ARGS, MAGCOUPLING_NATIVE_ARGS, magcoupling_assert_shipped (reads cargo --message-format=json; fails unless the shipped bin was compiled and no magcoupling-rs unit has workbook-parity)"
````

with:

````yaml
  shipped_builds: "linkage-sim-rs/scripts/magcoupling_shipped.sh (sourced by build_web.sh, build_magcoupling_web.sh and gate.sh): MAGCOUPLING_WEB_ARGS, MAGCOUPLING_NATIVE_ARGS, LINKAGE_WEB_ARGS, LINKAGE_NATIVE_ARGS (the linkage app's builds carry the panel), magcoupling_assert_shipped (reads cargo --message-format=json; fails unless the shipped bin was compiled and no magcoupling-rs unit has workbook-parity)"
````

In `docs/ai/03-structure.yaml`, replace:

````yaml
  gate: "linkage-sim-rs/scripts/gate.sh gates 4-12: 4-6 the engine (test, clippy -D warnings, wasm32 check); 7 cargo test --features app (panel and app headless tests, bins built with workbook-parity unified); 8 clippy --all-targets --features app -D warnings; 9 wasm32 clippy -D warnings (gui lib, magcoupling-web); 10 workbook-parity guard (native and wasm32 shipped builds, negative control); 11 lock parity (egui, egui_plot, eframe, wasm-bindgen; deploy-web.yml CLI pin); 12 Python parity suite + gen_differential.py --check; --full adds 13, build_web.sh (both bundles)"
````

with:

````yaml
  gate: "linkage-sim-rs/scripts/gate.sh gates 4-12: 4-6 the engine (test, clippy -D warnings, wasm32 check); 7 cargo test --features app (panel and app headless tests, bins built with workbook-parity unified); 8 clippy --all-targets --features app -D warnings; 9 wasm32 clippy -D warnings (gui lib, magcoupling-web); 10 workbook-parity guard (native and wasm32 shipped builds of the calculator's app and of the linkage app, a negative control per shipped build); 11 lock parity (egui, egui_plot, eframe, wasm-bindgen; deploy-web.yml CLI pin); 12 Python parity suite + gen_differential.py --check; --full adds 13, build_web.sh (both bundles)"
````

In `docs/ai/03-structure.yaml`, replace:

````yaml
    - mod.rs (LinkageApp + update loop orchestration; handle_delete_shortcut; draw_place_mass_field)
    - menu_bar.rs (File/Edit/Help/View/Image menus + sample gallery)
````

with:

````yaml
    - mod.rs (LinkageApp + update loop orchestration; window_title, send_title (BL-037); handle_keyboard_shortcuts; handle_delete_shortcut; draw_place_mass_field)
    - menu_bar.rs (File/Edit/Help/View/Image/Tools menus + sample gallery)
    - "calculator_window.rs (Tools -> Magnetic coupling, M5: CalculatorWindow holds the magcoupling panel, created on the first opening and kept; show draws the egui::Window (skipped on a screen without room for its frame), moves the keyboard by the last press (the window's layer and its resize band give it, a Foreground layer leaves it, anything else takes it), removes the key events but egui's zoom keys (UI_ZOOM_KEYS) while it has it and does the panel's requests (export::download, DesignPicker over rfd); TITLE, TOOL_PARAM, TOOL_MAGCOUPLING, magcoupling_share_base)"
````

In `docs/ai/03-structure.yaml`, replace:

````yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force, input events primary_button, primary_button_with, key_press, typed, the headless frame central_panel_frame, painted-output inspection drawn_texts, drew_text, text_rect, drawn_line_colors; the robot-lift fixture swept_lift with sample_at and pose_at; extend it instead of re-implementing fixtures per module)
````

with:

````yaml
    - test_support.rs (#[cfg(test)] helpers shared by GUI module tests, e.g. sorted_link_ids, set_actuator_stored_force, input events primary_button, primary_button_with, key_press, key_tap, typed, click_events, the headless screen NATIVE_SCREEN and screen_input, the headless frame central_panel_frame, painted-output inspection drawn_texts, drew_text, text_rect, drawn_line_colors; the robot-lift fixture swept_lift with sample_at and pose_at; the magnetic coupling calculator's design fixture magcoupling_gap_design; extend it instead of re-implementing fixtures per module)
````

In `docs/ai/03-structure.yaml`, replace:

````yaml
  wasm_bin: src/bin/linkage_web.rs (WASM build)
````

with:

````yaml
  wasm_bin: "src/bin/linkage_web.rs (WASM build; ?m= opens a shared mechanism, ?tool=magcoupling opens Tools -> Magnetic coupling)"
````

In `docs/ai/03-structure.yaml`, replace:

````yaml
  web_scripts: "scripts/build_web.sh (linkage bundle, then build_magcoupling_web.sh), scripts/build_magcoupling_web.sh (web/magcoupling/), scripts/serve_web.sh [PORT] (both bundles; default 8080), scripts/magcoupling_shipped.sh (sourced: shipped magcoupling-rs cargo args and the workbook-parity guard)"
````

with:

````yaml
  web_scripts: "scripts/build_web.sh (linkage bundle through the workbook-parity guard, then build_magcoupling_web.sh; deploy-web.yml runs it), scripts/build_magcoupling_web.sh (web/magcoupling/), scripts/serve_web.sh [PORT] (both bundles; default 8080), scripts/magcoupling_shipped.sh (sourced: shipped magcoupling-rs cargo args and the workbook-parity guard)"
````

In `docs/ai/04-memory.yaml`, replace:

````yaml
  - "Open (M5, from the M4-1 review): linkage-sim-rs/src/gui/mod.rs reads Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y with a non-consuming key_pressed, which runs whatever the panel consumes. Hosting MagcouplingPanel in an egui::Window, M5 must skip its own undo while the calculator window has the user's attention and turn the panel's shortcuts on only then (MagcouplingPanel::set_keyboard_shortcuts), so one key press never undoes both"
````

with:

````yaml
  - "RESOLVED (M5): the calculator window has the keyboard from its opening or a press on it (its resize band included) until a press on the linkage app (its panels, canvas or menu-bar buttons; an open menu's items or a drop-down list move nothing); only then are the panel's undo keys on, and CalculatorWindow::show removes the frame's key events but egui's zoom keys, so no linkage shortcut or canvas key runs on them (decision M5-5; linkage-sim-rs gui/calculator_window.rs). Not handled (backlog BL-039, BL-040): Tab focus moving between the parts, and the window covering the recovery prompt"
  - "RESOLVED (M5): decisions M5-1 to M5-10 as recommended in docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md: a Tools menu after Image with a Magnetic coupling checkbox; the window at (80, 100), 1100 x 700, resizable, kept on the screen; closed at start-up, the panel created on the first opening and kept for the session; ?tool=magcoupling; the keyboard follows the last press; saving through export::download and loading through rfd (also on the web); BL-037 fixed here; deploy-web.yml builds through build_web.sh; a design file dropped on the linkage page stays a mechanism import; the linkage web bundle carries the panel (about 1.46 MB more, accepted)"
````

In `docs/ai/backlog.yaml`, replace:

````yaml
  acceptance: "the title command is sent only when the title changes (or not at all on wasm32); a headless test counts the viewport commands over several idle frames (0 or 1); gui-smoke shows no Title warning"
  priority: 3
  status: open
  notes: "Seen in every gui-smoke run; benign. The magcoupling page does not have it."
````

with:

````yaml
  acceptance: "the title command is sent only when the title changes (or not at all on wasm32); a headless test counts the viewport commands over several idle frames (0 or 1); gui-smoke shows no Title warning"
  priority: 3
  status: fixed
  notes: "Seen in every gui-smoke run; benign. The magcoupling page does not have it. Fixed 2026-10-02 on magcoupling/m5 (plan docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md, decision M5-7): gui/mod.rs send_title sends the title only when it differs from the last one sent, and never on wasm32; gui::tests::the_window_title_is_sent_only_when_it_changes counts the commands; gui-smoke's linkage step reports title_warning."

- id: BL-038
  title: "Two copies of the design file picker: linkage gui::calculator_window::DesignPicker repeats magcoupling-rs app::files::DesignPicker"
  dimension: quality
  risk: mechanical
  evidence: "M5 plan review 2026-10-02 (docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md, decision M5-6): linkage-sim-rs/src/gui/calculator_window.rs DesignPicker (native rfd dialog, wasm32 AsyncFileDialog into an inbox) and save_filter follow magcoupling-rs/src/app/files.rs (its DesignPicker and extension filter) almost line for line, because M5 leaves magcoupling-rs unchanged."
  acceptance: "one DesignPicker, behind a magcoupling-rs feature that both hosts enable (e.g. files = [gui, dep:rfd, dep:wasm-bindgen-futures]); the standalone app's and the linkage window's picker tests pass; gate 10 still shows workbook-parity absent from every shipped build"
  priority: 3
  status: open
  notes: "Decision M5-6 option (b), deferred: it changes magcoupling-rs's Cargo.toml and its gui-is-egui-only contract."

- id: BL-039
  title: "The calculator window covers the linkage app's 'Recover Unsaved Work?' prompt when both show at once"
  dimension: gui
  risk: mechanical
  evidence: "M5 plan review 2026-10-02: linkage-sim-rs/src/gui/mod.rs draws the native and the wasm32 recovery prompts (egui::Window, Middle order, anchored at the centre) inline in update, after the calculator window, which raises itself on its second frame (gui/calculator_window.rs, raise). With ?tool=magcoupling and an autosave on the web, or Tools > Magnetic coupling opened natively while the prompt shows, the window hides the prompt."
  acceptance: "with the calculator window open and a recovery pending, the prompt is in front: a headless test draws the window and the prompt (moved out of update into a function) and Context::layer_id_at at the prompt's centre is the prompt's layer"
  priority: 3
  status: open
  notes: "Documented in M5 (decision M5-4, the calculator_window module docs, 02-system limitations): the user can move or collapse the window. Fix sketches: both prompts on Order::Foreground (a press on them then leaves the keyboard where it is, decision M5-5), or open_magcoupling deferred while a recovery is pending."

- id: BL-040
  title: "The calculator window's keyboard does not follow Tab focus between it and the linkage app"
  dimension: gui
  risk: mechanical
  evidence: "M5 plan review 2026-10-02: CalculatorWindow::follow_presses (linkage-sim-rs/src/gui/calculator_window.rs) moves the keyboard by presses only, while egui's Tab traversal (Focus::begin_pass reads Tab before any widget draws) can move the focus between the parts. Into a calculator field: the keys stay with the linkage app, so its canvas keys the field does not consume still run (Left and Right arrows nudge the selected link). Into a linkage field: the window keeps removing the key events, so Backspace, Enter and the arrows do nothing there."
  acceptance: "decided with the user first (decision M5-5 alternative d: the keyboard also follows egui's focus); then headless tests for both directions: a focus moved into the window gives it the keyboard, a focus moved to a linkage widget takes it, a focus on a Foreground layer leaves it"
  priority: 3
  status: open
  notes: "Sketch: each frame compare ctx.memory(|m| m.focused()) with the last frame's, also while the window is closed; on a change to Some(id), apply the press rule to ctx.read_response(id)'s layer_id. Documented in M5 (decision M5-5, the calculator_window module docs, 02-system limitations)."
````

In `docs/ai/05-update-tracker.md`, replace:

````markdown
---

## 2026-10-02 — Magcoupling M4-3: final review fix wave (branch magcoupling/m4-3)
````

with:

````markdown
---

## 2026-10-02 — Magcoupling M5: the calculator in the linkage app (branch magcoupling/m5)
- `linkage-sim-rs` depends on `magcoupling-rs` by path with feature `gui` (one egui: 0.32.3 in
  both lock files; gate 11 passes). Tools > Magnetic coupling (`gui/calculator_window.rs`)
  toggles an `egui::Window` holding one `MagcouplingPanel`, created on the first opening and kept,
  with its design and undo history, while the window is closed; nothing in it touches the model.
- Keyboard (decision M5-5): the window has the keys from its opening or a press on it (its
  resize band included) until a press on the linkage app: its panels, canvas or menu-bar buttons
  (an open menu's items and drop-down lists move nothing). Only then are the panel's undo keys
  on, and the window removes the frame's key events but egui's zoom keys, so neither the linkage
  shortcuts (undo, save, new, delete) nor the canvas keys (arrow nudge, F, Escape, Enter) run on
  them. `update` shows the window before any key reader; the shortcut block moved to
  `handle_keyboard_shortcuts` and a source test pins the order. Documented, not handled: Tab
  focus moving between the parts (BL-040) and the window covering the recovery prompt (BL-039).
- On a screen without room for its frame (a hidden browser canvas reports 0 x 0) the window skips
  the frame, so egui does not squeeze its stored size; it comes back in its place at its size.
- Files: saving through `export::download`, "Load design" through rfd natively and on the web
  (the linkage wasm build now has rfd). On the web, `?tool=magcoupling` opens the window at start
  and share links made there open `/magcoupling/` on the same server.
- BL-037 fixed (decision M5-7): the window title is sent only when it changes, never on wasm32.
- Gate 10 guards the linkage app's native and wasm32 builds too (`LINKAGE_*_ARGS` in
  `magcoupling_shipped.sh`), with a negative control per shipped build (the calculator's native
  app gained one); `build_web.sh` pipes the linkage release build through the guard and
  `deploy-web.yml` now builds both bundles through it.
- gui-smoke's linkage step opens `/?tool=magcoupling` and checks the window, the registry's log
  line and that no Title warning is logged.
- The linkage wasm bundle grows from about 10.5 MB to about 12.0 MB with the panel (rfd and the
  panel), accepted (decision M5-10); the calculator's own bundle is unchanged (5.0 MB). The
  magcoupling crate, its engine and its tests are unchanged. Nothing pushed.
- Docs: both READMEs, `docs/FEATURES.md` (Magnetic Coupling Calculator) and
  `docs/guides/WASM_DEPLOYMENT.md` (`build_web.sh` builds both bundles through the guard; the web
  build carries rfd for the picker). Backlog: BL-038 (one design file picker for both hosts),
  BL-039 and BL-040 (the documented keyboard and prompt edges).

## 2026-10-02 — Magcoupling M4-3: final review fix wave (branch magcoupling/m4-3)
````

In `docs/FEATURES.md`, replace:

````markdown
### Additional Fixes & Polish

- **Active tool highlight** -- blue filled background with white text on selected tool button
````

with:

````markdown
### Magnetic Coupling Calculator

- **Tools > Magnetic coupling** -- the magnetic slip coupling calculator (`magcoupling-rs`) in a window beside the mechanism, on the desktop and on the web (`?tool=magcoupling` opens it when the web app starts). Its design is separate from the mechanism and kept while the window is closed; it saves and loads its own design files. The keys go to the window or the mechanism, whichever was clicked last. The calculator also runs on its own at `/magcoupling/`

### Additional Fixes & Polish

- **Active tool highlight** -- blue filled background with white text on selected tool button
````

In `docs/guides/WASM_DEPLOYMENT.md`, replace:

````markdown
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
````

with:

````markdown
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
````

In `docs/guides/WASM_DEPLOYMENT.md`, replace:

````markdown
5. Build the WASM binary and JS bindings (same commands as `build_web.sh`).
````

with:

````markdown
5. Run `scripts/build_web.sh`: both bundles, each piped through the workbook-parity guard.
````

In `docs/guides/WASM_DEPLOYMENT.md`, replace:

````markdown
- **No file dialogs** -- Save, Open, and Save As use native file dialogs (`rfd` crate) which are gated behind the `native` feature flag and excluded from the WASM build.
````

with:

````markdown
- **No file dialogs** -- Save, Open, and Save As use native file dialogs (`rfd` crate) gated behind the `native` feature flag. The web build carries rfd only for the magnetic coupling calculator window's Load design (the browser's file chooser).
````

- [ ] **Step 2: Check the YAML entries and the whitespace**

`docs/ai/02-system.yaml` does not parse as a whole at `4c95255` (an older unquoted `: ` in `architecture_invariants`), so this checks each added list entry on its own; the other three files parse whole.

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5 && python - <<'EOF'
import subprocess, yaml
diff = subprocess.run(['git', 'diff', '-U0', '--', 'docs/ai/02-system.yaml'], capture_output=True, text=True, encoding='utf-8').stdout
items = [line[1:] for line in diff.split('\n') if line.startswith('+  ') and line[1:].lstrip().startswith('- "')]
for item in items:
    value = yaml.safe_load('k:\n' + item)['k']
    assert isinstance(value, list) and isinstance(value[0], str), item
print('02-system entries ok:', len(items))
for name in ['03-structure', '04-memory', 'backlog']:
    yaml.safe_load(open(f'docs/ai/{name}.yaml', encoding='utf-8'))
    print(name, 'ok')
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m5 diff --check && echo whitespace-ok
```

Expected: `02-system entries ok: 12`, `03-structure ok`, `04-memory ok`, `backlog ok`, `whitespace-ok`.

- [ ] **Step 3: Build both web bundles through the guard and measure them**

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && bash scripts/build_web.sh 2>&1 | grep -E "parity guard|complete!"; wc -c web/linkage-web_bg.wasm web/magcoupling/magcoupling-web_bg.wasm
```

Expected: `parity guard: linkage-web builds without workbook-parity; magcoupling-rs features: "features":["default","gui"]`, `Build complete!`, `parity guard: magcoupling-web builds without workbook-parity; magcoupling-rs features: "features":["app","default","gui"]`, `Magcoupling build complete!`; then about `11991219 web/linkage-web_bg.wasm` (about 1.46 MB, 14 %, more than Task 0 Step 8's count: the panel and rfd; decision M5-10) and `4996155 web/magcoupling/magcoupling-web_bg.wasm` (unchanged). Record both counts and the change in the execution notes for the final report; if the linkage count is not within 1 % of 11,991,219, stop and escalate.

- [ ] **Step 4: Open the linkage app and its calculator window in a browser**

Serve the bundles and check them with the Playwright MCP tools (load them via ToolSearch, e.g. `select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_resize,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_close`), or have the controller run the `gui-smoke` workflow (its linkage step does the same checks).

Run in the background (`run_in_background`):

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && bash scripts/serve_web.sh 8080
```

Then: resize the browser to 1400 x 900; navigate to `http://localhost:8080/`, wait 6 seconds, read the console at level `debug`; navigate to `http://localhost:8080/?tool=magcoupling`, wait 6 seconds, read the console at level `info` and take a screenshot (save it under `.playwright-mcp/` of the main checkout); optionally click "Load design" in the window (the browser's file chooser opens: rfd's web picker; upload a design file such as `{"format": "magcoupling-design", "version": 1, "inputs": {"metal.face_gap_mm": 2.0}}` written under `.playwright-mcp/`, then the header says "Design file loaded" and the face gap reads 2.00 mm; click inside the window before pressing keys, since the page's canvas loses the browser's focus to the chooser); close the browser; stop the server: TaskStop on the background task, then `netstat -ano | grep ":8080 .*LISTENING"`; the Python server survives TaskStop, so if a line is printed, `taskkill //PID <the PID in its last column> //F` and check again until nothing is printed; delete the files this step wrote under `.playwright-mcp/` (the screenshot and the design file; leave other sessions' files).

Expected: both pages have zero console errors and zero warnings (no `Unhandled egui viewport command: Title`: BL-037); the second page logs `magcoupling explorer: 392 equations`; the screenshot shows the window "Magnetic coupling" at the top left, below the toolbars, in front of the linkage welcome screen, holding the calculator (the header "Magnetic coupling calculator" with Undo, Redo, Reset all, Save design, Load design, Copy share link; the inputs on the left; the geometry view in the middle; the dashboard on the right, "Pull-out torque at operating temperature 2.688 N·m" first).

- [ ] **Step 5: Copy the plan into the repository**

Run:

```bash
cp C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/m5-plan/plan.md C:/Users/Cole/source/repos/lsim-mag-m5/docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md
head -1 C:/Users/Cole/source/repos/lsim-mag-m5/docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md
```

Expected: `# Magcoupling M5: Embed the Calculator in the Linkage App Implementation Plan`. If the scratchpad copy is gone, the controller supplies the plan file it executed.

- [ ] **Step 6: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task4.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task4.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task4.log; sed -n '/== gate 1\/12/,/== gate 2\/12/p' C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs/target/gate-task4.log | grep -E "test result" | head -1
git -C C:/Users/Cole/source/repos/lsim-mag-m5 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m5/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines as in Task 0 Step 7 (`1159 passed in ...s`, `differential data is current (...)`, `GATE PASS`); `0`; `test result: ok. 953 passed` (gate 1's library run); `cargo fmt --check` prints nothing.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m5 add README.md magcoupling-rs/README.md docs/FEATURES.md docs/guides/WASM_DEPLOYMENT.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/backlog.yaml docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md
git -C C:/Users/Cole/source/repos/lsim-mag-m5 commit -F - <<'EOF'
docs(magcoupling): M5 docs, BL-037 fixed and the plan

Both READMEs describe the calculator window (Tools > Magnetic coupling, the
keys follow the last press, the file requests, ?tool=magcoupling) and the
guard on the linkage builds; FEATURES.md lists the calculator and
WASM_DEPLOYMENT.md builds both bundles through build_web.sh; docs/ai records
the window's module and invariant, six egui and test lessons, three known
limitations (the dropped design file, decision M5-9; Tab focus; the
recovery prompt), the resolved M5 keyboard item and decisions M5-1 to
M5-10, the update entry, BL-037 as fixed and BL-038 to BL-040 as open. The
plan is docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m5 status --short
git -C C:/Users/Cole/source/repos/lsim-mag-m5 log --oneline main..HEAD
```

Expected: one new commit; `status --short` prints nothing; the log lists four commits (Tasks 1 to 4).

---

### Task 5: The user's check, after the whole-branch review

**Model:** the controller with the user (the spec's M5 gate is "Tests + smoke + user check"); the commit step is `sonnet`. Run it after the whole-branch review of Tasks 1 to 4 and its fixes, on the branch that will merge. If any check fails, stop: record what the user saw, fix it through a reviewed commit (escalate to the session model), and run this checklist again.

**Files:**
- Modify: `docs/ai/04-memory.yaml` (the sign-off, after the M5 decisions line)

**Interfaces:**
- Consumes: Tasks 1 to 4 (the native `linkage-gui` binary; the two web bundles, `scripts/build_web.sh`, `scripts/serve_web.sh`).
- Produces: the user's sign-off in `docs/ai/04-memory.yaml`.

- [ ] **Step 1: The user checks the native app**

Run in the background (`run_in_background`), then hand the window to the user:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && cargo run --release --bin linkage-gui
```

The user checks, and the controller notes each answer:

1. Tools → Magnetic coupling opens the window at the top left, below the toolbars, and the same item (or the window's X) closes it; reopened, the window keeps its design (change the face gap first).
2. Ctrl+Z and Ctrl+Y act on whichever part was clicked last: after a click in the window they undo and redo the calculator only; after a click on the mechanism or its panels, the mechanism only.
3. After a click in the window, the arrow keys, Delete, F and Escape do nothing to the mechanism (select a link first); after a click on the canvas they work again.
4. With the window holding the keys, a click on a menu-bar button (File, View, Tools) gives them to the mechanism; pressing just outside the window's right edge and dragging resizes the window and gives it the keys back.
5. Ctrl+Plus, Ctrl+Minus and Ctrl+0 zoom the whole UI whichever part has the keys.
6. Collapsed to its title bar (its triangle), the window leaves the keys to the mechanism.
7. Save design writes a `.json` file through the save dialog, and Load design reads it back ("Design file loaded").

Then close the app.

- [ ] **Step 2: The user checks the web build**

Run in the background (`run_in_background`), after `bash scripts/build_web.sh` if the review changed code since Task 4 Step 3:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5/linkage-sim-rs && bash scripts/serve_web.sh 8080
```

The user opens `http://localhost:8080/?tool=magcoupling` in a browser: the window shows over the welcome screen; Load design opens the browser's file chooser and loads the design saved in Step 1 (click inside the window afterwards before pressing keys: the chooser takes the page's focus). Then stop the server as in Task 4 Step 4 (TaskStop, then `netstat -ano | grep ":8080 .*LISTENING"` and `taskkill //PID <pid> //F` until nothing is printed).

- [ ] **Step 3: Record the sign-off**

Only once the user has passed every check of Steps 1 and 2:

In `docs/ai/04-memory.yaml`, replace:

````yaml
  - "RESOLVED (M5): decisions M5-1 to M5-10 as recommended in docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md: a Tools menu after Image with a Magnetic coupling checkbox; the window at (80, 100), 1100 x 700, resizable, kept on the screen; closed at start-up, the panel created on the first opening and kept for the session; ?tool=magcoupling; the keyboard follows the last press; saving through export::download and loading through rfd (also on the web); BL-037 fixed here; deploy-web.yml builds through build_web.sh; a design file dropped on the linkage page stays a mechanism import; the linkage web bundle carries the panel (about 1.46 MB more, accepted)"
````

with:

````yaml
  - "RESOLVED (M5): decisions M5-1 to M5-10 as recommended in docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md: a Tools menu after Image with a Magnetic coupling checkbox; the window at (80, 100), 1100 x 700, resizable, kept on the screen; closed at start-up, the panel created on the first opening and kept for the session; ?tool=magcoupling; the keyboard follows the last press; saving through export::download and loading through rfd (also on the web); BL-037 fixed here; deploy-web.yml builds through build_web.sh; a design file dropped on the linkage page stays a mechanism import; the linkage web bundle carries the panel (about 1.46 MB more, accepted)"
  - "M5 user check: the user ran the plan's Task 5 checklist on the reviewed branch (native linkage-gui and the web build: open and close, the design kept, undo on the part clicked last, the keys after a click in the window, the menu bar, the resize edge, the zoom keys, a collapsed window, Save and Load design, ?tool=magcoupling) and signed it off before the merge"
````

Then run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-m5 && python -c "import yaml; yaml.safe_load(open('docs/ai/04-memory.yaml', encoding='utf-8')); print('04-memory ok')"
git -C C:/Users/Cole/source/repos/lsim-mag-m5 add docs/ai/04-memory.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m5 commit -F - <<'EOF'
docs(magcoupling): M5 user check signed off

The user ran the plan's Task 5 checklist on the reviewed branch, native and
web, and signed it off before the merge.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m5 status --short
```

Expected: `04-memory ok`; one new commit; `status --short` prints nothing. The branch can now merge.

---

## Self-review record

Checked against the spec and the task brief before handing over:

1. **Spec coverage.** Structure: "depends on it by path with feature `gui`" (Task 1), "opens the panel from a **Tools** menu" (Task 2), "No Cargo workspace" (Global Constraints). M5: the same panel in an `egui::Window` (Task 2), "state independent of the linkage model" (the panel lives in `LinkageApp`, never `AppState`; tests: closing and reopening keeps the design and history; linkage undo does not touch the calculator and the reverse), "the standalone page stays at `/magcoupling/`" (untouched; share links point there), "`build_web.sh` and `deploy-web.yml` build and ship both bundles" (kept, deploy routed through `build_web.sh`, Task 1). Brief: the parity guard on the linkage shipped builds with proof (gate 10 plus a negative control per shipped build), one egui (`cargo tree -i egui`, gate 11), host I/O (Task 2), shortcuts that never clash (decision M5-5, Task 2 tests), BL-037 (Task 3), headless tests for the menu, the window, close and reopen, and undo in both directions (Task 2), the wasm bundle with the panel and its size (Task 4), gui-smoke's linkage step without coordinate clicks (Task 3), both READMEs and docs/ai (Task 4), UX choices in the Decisions table.
2. **Placeholders.** None: every code step is a block generated from code that compiled and passed; every command has its expected output from the replay. The `Co-Authored-By:` line names the plan's writer and the Global Constraints tell an implementer to write its own model's name; Task 4 Step 4's `taskkill` takes the PID netstat prints.
3. **Names and types across tasks.** `CalculatorWindow` (`set_open`, `is_open`, `has_keyboard`, `show`, test-only `panel`, `design`, `load_design`), `TITLE`, `window_id`, test-only `default_rect`, `beside_the_window` (Task 2) and `set_share_base`, `TOOL_PARAM`, `TOOL_MAGCOUPLING`, `MAGCOUPLING_PAGE_PATH`, `magcoupling_share_base` (Task 3) are used with the same names in `gui/mod.rs`, `menu_bar.rs`, `linkage_web.rs`, the tests and the docs; `key_tap`, `click_events`, `screen_input`, `NATIVE_SCREEN` and `magcoupling_gap_design` (Task 2, `test_support`) are the input and fixture helpers every new test uses; `LINKAGE_WEB_ARGS`, `LINKAGE_NATIVE_ARGS`, `assert_guard_trips` (Task 1) are the names the docs cite.
4. **Review Focus.** Each of its five lines has tests in Task 2. Writing them found three defects, fixed in the window's code before this plan was written: the frame hanging off a small screen, the welcome screen covering the window at start-up, and a collapsed window swallowing the keys; the closed window's focused field already let go (egui drops the focus of a widget that is no longer drawn). The dropped design file (decision M5-9) is documented rather than tested: `update`'s drop handling cannot be driven headlessly (Global Constraints).

## Self-review record (revision after the critic's review)

The critic found no blocking defect and sixteen non-blocking ones; each is fixed below (where, and what verifies it). Deviations from the critic's suggested fixes are marked "Deviation", and what fixing them turned up is marked "Found".

1. **Menu-bar buttons take the keyboard** (docs said "menus" leave it). Reworded wherever the rule is stated: Architecture, decision M5-5, Review Focus 4, the module docs, the Keyboard Shortcuts row, `magcoupling-rs/README.md`, the 02-system invariant, 04-memory, the update tracker and Task 2's commit message (an open menu's items and drop-down lists leave it; a menu-bar button takes it). The popup test is renamed `the_keyboard_follows_presses_and_popups_leave_it_alone`, and `a_press_on_the_menu_bar_gives_the_keyboard_to_the_linkage_app` (menu_bar.rs) presses the real Tools button while the window holds the keys, after the window in `update`'s order, and checks that the menu opened. A mutation treating Background as a popup fails it.
2. **egui's zoom keys** were removed with the rest. `UI_ZOOM_KEYS` (egui's four `kb_shortcuts`) and `is_ui_zoom_key` (matched as egui's `consume_shortcut` matches) keep them; `with_the_keyboard_egui_s_zoom_keys_still_zoom_the_ui` drives Ctrl+Plus, Ctrl+Equals, Ctrl+Minus and Ctrl+0 (zoom 1.1, 1.2, 1.1, 1.0, one frame later) and bare Minus joined the keys the host must not see. No linkage reader reads Plus, Minus, Equals or Num0 (grep), so the kept events reach only egui's end-of-frame zoom. Found while checking it in the browser: eframe's web backend turns egui's keyboard zoom off and leaves the keys to the browser, so the fix is native only and the docs say so (a new 02-system lesson).
3. **The resize band.** `on_resize_band` uses egui's own radii (5 points out from an edge, 10 around a corner: the critic's side band only would miss the corners). Deviation: it is checked after the window and Foreground arms, not first, so a menu hanging over the band still leaves the keyboard, and only for a press on a Background layer or none, so another window over the band takes it. Found while testing it: egui 0.32 panels are Background areas (`layer_id_at` returns a Background layer over them, not `None`), and egui moves the edge to the pointer on the press, so the test asserts that instead of "nothing resized". Tests: `a_press_on_the_band_where_egui_resizes_the_window_gives_it_the_keyboard` (+3 from the edge and +7,+7 from the corner give it, +8 does not) and `another_window_over_the_band_takes_the_keyboard`; mutations dropping the band, the corners or the Background condition fail them.
4. **A 0 x 0 screen.** Deviation: the critic's `(screen.size() - margins).max(Vec2::ZERO)` does not help. egui caps a window's size at its constrain rect (`window.rs` L497-503, which with the title bar goes negative) and its `Resize` stores the capped size (`resize.rs` L263-266); measured, one such frame left the window at `(0, 0)-(609.1, 900)` with or without the clamp. The window now skips a frame whose screen has no room for its frame and comes back exactly. Only the web can send that screen (egui-winit sends none for a zero-size window). Test: `a_screen_without_room_skips_the_window_and_keeps_its_place_and_size`, which fails without the guard and with the clamp in its place.
5. **The recovery prompt under the raised window.** Documented, not fixed: decision M5-4 (with alternative (d), keep the prompt in front), the module docs, 02-system's limitations, `magcoupling-rs/README.md`, the update tracker, and backlog BL-039 with both fix sketches. The prompts are inline in `update`, which headless tests never drive, so a fix there would ship untested; BL-039's acceptance asks for the prompt to be drawn by a testable function first.
6. **Tab focus.** Documented, not handled: a new keyboard rule is the user's decision, so decision M5-5 gains the known edge and alternative (d), "the keyboard also follows egui's focus"; the module docs, 02-system and the README state it; backlog BL-040 carries the two-direction sketch. Found while checking: a focused DragValue consumes Up and Down itself, so it is Left and Right that also nudge the link; and the reverse direction matters too (focus tabbed into a linkage field while the window holds the keys loses Backspace, Enter and the arrows).
7. **The smoke test's second assertion** was already true at `4c95255`. It now asserts the linkage step's own sentence (`magcoupling_window=true only if a console message contains "magcoupling explorer: "`); a mutation dropping it from `gui-smoke.js` fails the test.
8. **Guard coverage.** A negative control per shipped build: `linkage-gui` as asked, and also `magcoupling-app` (deviation: so "per shipped build" is true rather than "per linkage and calculator build" with one missing). Gate 10 printed all four in every gate run of the replay; the README, 02-system, 03-structure's gate line, the update tracker, Task 1 Step 6 and its commit message say so.
9. **Repeated test code.** `test_support` gains `NATIVE_SCREEN`, `screen_input` (one `RawInput` builder for the window, menu and `gui::tests` frames), `click_events` and `magcoupling_gap_design`; `calculator_window` gains test-only `default_rect`, `beside_the_window`, `design` and `load_design` (replacing `panel_mut`, both design accessors and both design loads); the window tests share `window_frame`, `frame_with_area` and `window_rect`. The "80..1180 x 100..800" comments are gone (they were also wrong: the window spans about 80..1252 x 100..848). 03-structure's `test_support` line lists the helpers.
10. **DesignPicker repeats `app/files.rs`.** Option (a) kept (magcoupling-rs stays unchanged); backlog BL-038 (risk mechanical) moves one picker behind a magcoupling feature both hosts use, with its acceptance.
11. **The spec's user check.** Task 5, after the whole-branch review and its fixes: a native and a web checklist (the zoom keys, the menu bar and the resize edge included), then the sign-off line in 04-memory (an exact block, applied in the replay) and its commit. The Process, Global Constraints (merge only after Task 5), File Structure and the Verification table name it.
12. **The bundle size.** Decision M5-10 (accept; or gate the panel out of the web build and link to `/magcoupling/`), with the measured sizes; cited in the update tracker, 04-memory and Task 4 Step 3.
13. **Model tiers.** Task 2's per-task review runs on the session model (`// session model: keyboard routing judgment`): the Process, Global Constraints and Task 2's Model line.
14. **Docs with code.** `docs/guides/WASM_DEPLOYMENT.md`: the build section describes both bundles through the guard (and says to run the script rather than copy its commands), deploy step 5 runs `build_web.sh`, and its rfd line (false after Task 1; not in the critic's list) says the web build carries rfd for Load design. `docs/FEATURES.md`: deviation, a Magnetic Coupling Calculator section under Implemented, because the critic's "Engineering Tools" heading sits under Planned / Future. Both are in Task 4's Files, Step 7's `git add` and Task 0 Step 2's check. Left alone: WASM_DEPLOYMENT.md's "No autosave" line, stale before M5 (the web build autosaves to the browser).
15. **The CRLF note** now says the smoke test's file is LF by `.gitattributes`.
16. **The Keyboard Shortcuts row** reads "From its opening or a click in it, keys go to the calculator (Ctrl+Z / Ctrl+Y undo there; Ctrl+Plus / Ctrl+Minus still zoom) until you click the mechanism, its panels or the menu bar".

Found as well: `build_magcoupling_web.sh`'s header said deploy-web.yml calls it, which M5-8 makes false; Task 1 fixes it (and syntax-checks the script). The original record above has its names updated (`design`, `load_design`, the shared helpers; a negative control per shipped build).
