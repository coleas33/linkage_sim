# Magcoupling M4-1: GUI Inputs, Dashboard, Results and Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the M4 tracer panel into the calculator's working GUI: every input generated from the engine's metadata (the Key design group first), a dashboard with verdict badges, corrected-vs-workbook markers, the audit M9 greying, the stored-3D label and the space claim, a searchable results table with CSV and JSON export, undo and redo, design files and share links, the Addendum A1 sizing mode switch, and the linkage app's theme, in the native and the web app.

**Architecture:** The panel (`gui::MagcouplingPanel`, feature `gui`) stays egui-only and host-neutral: it holds the design (the inputs and the sizing state), recomputes `compute_all` of the design shown every frame after the inputs are drawn, and queues the platform work (saving a file, picking one) as `PanelRequest`s its host does (feature `app`: rfd dialogs natively, a download on the web; the linkage app in M5). Small modules with one job each: an input catalogue built once from the metadata and one row widget for every field type; a dashboard and a results table that read results by path through one hover hook (`result_tooltip`, which M4-3's equation tooltip joins); a correction index built once from the deviation registry and the compiled golden files; design files and share links as one schema-versioned JSON (deflate and URL-safe base64 for `?m=`, the linkage tool's scheme); a snapshot history committed per settled edit; a sizing runner that calls `sizing::solve` only after a debounce, never per frame. The centre region (`CentreView`) and the hover hook are where M4-2 and M4-3 slot in.

**Tech Stack:** Rust 2024 (rustc 1.89); egui and eframe 0.32.3 (locked to linkage-sim-rs's versions); serde_json 1 (`float_roundtrip`), base64 0.22, flate2 1 and log 0.4 behind feature `gui`; rfd 0.15, web-sys 0.3 and js-sys 0.3 behind feature `app`; wasm-bindgen 0.2.114; headless egui tests (`egui::Context::run` with injected input); Playwright MCP for the web smoke; Git Bash for every command.

**Spec:** `C:/Users/Cole/source/repos/lsim-mag-m41/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`: M4 "Layout" (left inputs with the Key design group, right dashboard with badges and corrected markers, the results table), "Sliders", "Session" (undo/redo, reset all, design JSON, a `?m=` share link, the linkage app's theme); Addendum A1 (the mode switch in the Key design group, inverse sizing, the space claim badge); "Testing and validation summary" (headless egui tests: slider change, reset, selector branch, undo; `gui-smoke` extended to `/magcoupling/`). The user's decisions of 2026-09-30 (`docs/ai/04-memory.yaml`): M3 comes after M4, so the temperature limits carry "3D values from the workbook" instead of a "3D updating" badge; audit M9, f_end <= 0 greys the numbers computed from the pull-out under "end-effect model out of range". Evidence: `docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md` (decisions 27 to 31; its section 6: the material and grade pickers are presets, plan M4-3). Scope: the first of three M4 plans; M4-2 is the geometry view, the plot tabs and the clamp drawing; M4-3 the equation explorer, the assumptions panel, the teaching notes and the material and grade pickers with their warnings.

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-m41`, branch `magcoupling/m4-1`, created by Task 0 from `main` (verified at `8bad022`: the merge of the M4 infrastructure, `167db09`, plus two recorded A-2 carry-overs in `docs/ai/04-memory.yaml`). Every command uses absolute paths into it. Nothing is pushed.

**Merge order:** this plan (M4-1) merges before plan A-3, whose branch `magcoupling/addendum-a3` (at `0ca6b2f` when this plan was written) edits `magcoupling-rs/README.md`, adds Rust-only results and a test binary (`tests/explain.rs`), and touches engine files. If A-3 lands on `main` first, Task 0 Step 2 lists `magcoupling-rs/README.md` and stops: re-derive the README blocks of Task 11, re-run Task 0 Step 5 for the new counts (every later count is then an offset), and re-check what moves with the results layout: `every_marked_cell_belongs_to_a_registry_entry_and_most_results_are_unmarked` bounds the marked rows by `marked > 100 && marked < rows.len() / 2` (`gui/corrections.rs`), and `the_smoke_test_share_link_opens_its_design` compares the pinned link with `Design::default()` (a changed default or a removed input path fails it; the README says how to regenerate the link). The "N of M results" line and the results-table tests read `table_entries()`, so they follow the layout.

## Decisions to confirm

**Confirmed (user, 2026-10-01): the recommended option on all of M41-1 to M41-15.** No task stops on a decision.

These choices arose while turning the spec into code; no approved decision settles them. The plan implements the recommended option of each (the code comments and the README cite the ids). Task 0 records the user's answers; if the user picks another option, the task that implements it stops and escalates instead of improvising.

| Id | Question | Recommended (implemented here) | Alternatives | Tasks |
|---|---|---|---|---|
| M41-1 | egui snaps a slider value to `min + k * step`, so face gap 1.41 is stored as 1.4100000000000001 and stepping back to a default lands beside it (the changed-from-default dot would stay lit) | **Round every slider value to the step's decimals** (egui's `max_decimals` with `inputs::step_decimals`): the slider stores the decimal the user reads, and stepping back lands on the default exactly. That holds for every default on its step grid: all but `coupling.mu0` and `calibration.mu0` (default 1.256637e-6, engine step 1e-11), which a test lists; the engine fix (a 1e-12 step) is recorded as an open item | (b) keep egui's float noise (results differ at about 1e-16 relative; the dot compares with a tolerance) | 3, 4 |
| M41-2 | May a typed value leave the slider range? The engine's `SliderRange` does not reject typed values, and only the range is covered by the differential tests | **Clamp edits to the range, typed values included** (`SliderClamping::Edits`); a value already outside the range (from a design file, a share link or another version) is kept until edited and the row says "outside the slider range" | (b) let typed values leave the range (`SliderClamping::Never`, which also lets arrow keys step past the ends); (c) clamp loaded values too (a file would not load as written) | 4 |
| M41-3 | Light or dark? The linkage app applies its CAD dark visuals at start-up; its web build still renders light in a light-preference browser (backlog BL-013); the tracer page followed the browser | **Always the linkage app's CAD dark theme**: the visuals copied from `linkage-sim-rs/src/gui/theme.rs` (a test compares the function bodies) and `ThemePreference::Dark`, whatever the system prefers; the page background is the same panel colour (`#1e2026`). The panel sets no theme, so in M5 the host's applies | (b) follow the system: CAD dark when dark, egui's light otherwise (the page background must then follow too); (c) a light/dark toggle | 9 |
| M41-4 | Do design files and share links record the sizing state (mode, free variable, target torque: GUI state per A-2 decision A2-9)? | **Yes**: a link reopens in Torque -> Magnets with the same target, and the solve runs on load | (b) no: files and links hold the inputs only and always open in the forward mode | 1, 7, 8 |
| M41-5 | What does a share link carry? | **The whole design file** (every input and the sizing state), compact JSON, deflated, URL-safe base64: one format and one parser for files and links, immune to a later change of a default; 2,462 characters for the default design | (b) only the inputs that differ from the defaults (short links, but an old link changes meaning if a default changes) | 1 |
| M41-6 | A design file or link with an unknown path, a wrong type, a code outside its choices or a newer version | **Refuse it whole and name every problem**, the inputs' and the sizing state's together: nothing changes. A path a later version renames or removes is not unknown: `session::PATH_MIGRATIONS` maps it (a rename adds an entry and bumps the format version), so older files and links keep opening | (b) load what fits and warn about the rest | 1, 7 |
| M41-7 | Leaving Torque -> Magnets: what becomes of the free variable? | **Keep the value the panel showed** (the solved or best value is written into the inputs as one undoable edit), so "size it, then tweak" works. The value is solved for the design at the moment of leaving: a change still waiting for its debounce ("Solving...") is solved first, once | (b) go back to the input's own value | 8 |
| M41-8 | What does the panel show when the target is out of reach? | **The design at the best valid value the search found**, with "Not reachable (best Y at X)" | (b) the inputs' own value of the free variable, with the message | 8 |
| M41-9 | The target torque of Torque -> Magnets | **Its own setting**, starting at 2.5 N·m (the workbook's hot minimum), on the slider of `metal.required_min_Nm` (unit, range and step) | (b) tied to `metal.required_min_Nm` itself | 1, 8 |
| M41-10 | Which values carry the "corrected vs workbook" marker? The test-only switch that turns corrections off never ships, so the panel cannot compare | **Every value whose workbook cell the registry ties to an applied engine correction**: changed at the default design (`changes_at_defaults` and the golden files E3, E4, E5, compiled in), named by the report (`cells`), or shown by a probe; the tooltip says which, with the at-defaults values | (b) only cells changed at the default design; (c) none until plan A-3's per-record upstream corrections exist (M4-3 may re-source the markers from them) | 5 |
| M41-11 | How far does the audit M9 greying reach? | **The dashboard rows computed from the pull-out** (`END_EFFECT_ROWS`, pinned by a probe of c_end) greyed without badges, and the banner over the dashboard and the results table | (b) also grey every results-table row that depends on f_end, through a dependency probe now (plan A-3's graph does it properly later) | 5, 6 |
| M41-12 | Where does an optional input start when the user enters a value? | **At the result it overrides, unrounded**: the axial length override enters at the inner ring's length in use, so nothing moves; the measured drag enters at the model's equivalent mean drag torque, so the drag in use does not move, but entering any measured drag switches the thermal summary from the not-measured high estimate to the measured branch (at the defaults the steady high case goes from 92.51 to 74.17 °C and the slip loss from "not measured" to 2.751 W; rounding the seed to the slider's step, 0.0131, would move ten more results) | (b) at the slider minimum (the axial length override at 2 mm would put f_end out of range at once) | 3, 4 |
| M41-13 | Do the Key design inputs also appear in their package groups? | **Yes**: each group lists all its inputs, as the package and the results table do; both rows edit the same value | (b) only in the Key design group | 3 |
| M41-14 | Undo granularity | **One step per settled edit** (a drag, a typed value, a part name typed letter by letter) and one per arrow-key nudge; a key held down is an edit in progress, so a held arrow key's auto-repeat run is one step (it cannot flood the levels); 100 levels; Ctrl+Z inside a text field is the field's own | (b) one step per frame that changed the design | 2, 7 |
| M41-15 | JSON has no infinity or NaN (E13 gives +inf, E20 NaN), and serde_json writes `null` | **Write `"+inf"`, `"-inf"`, `"NaN"`** (the panel's display text) in the JSON export and in CSV | (b) `null` (indistinguishable from "not entered"); (c) bare `Infinity` (not valid JSON) | 6 |

Settled by this plan's design without a new decision (each is stated where it is implemented): reset all restores the whole design (inputs and the forward mode) and can be undone; a share link the web app opens at start-up is the session's start, not an undo step; the solve waits 0.25 s after the last change and for any drag, held key or typing in an input row to end (typing in the results search is no edit); the JSON export holds the design that produced the results (in Torque -> Magnets the inputs with the free variable at the value shown); the panel acts on Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y unless its host keeps them (`set_keyboard_shortcuts`, for M5); a selector code outside its choices shows "N (not a choice)"; the results table builds its rows once (the result layout does not depend on the inputs) and reads only the values on screen; the panel never touches files: the app does (natively a dialog, on the web a download and rfd's file picker).

## Global Constraints

Every task's requirements implicitly include this section.

- The engine does not change: no file under `magcoupling-rs/src/engine/` or `magcoupling-rs/tests/` is edited. Workbook parity (1,149 checks), the differential data (`gen_differential.py --check`), every registry test and `all_corrections_together_give_the_reviewed_headline` pass unchanged. `reference/magcoupling-py/` is not edited.
- Feature boundaries: the engine stays pure std with no dependencies (gate 6). Feature `gui` adds only egui, serde_json, base64, flate2 and log, and does no I/O: no file system, no network, no eframe. Feature `app` adds rfd, web-sys and js-sys. `workbook-parity` never reaches a shipped build (gate 10).
- Versions: egui and eframe 0.32.3 and wasm-bindgen 0.2.114 stay locked as in linkage-sim-rs (gate 11). The new crates take linkage-sim-rs's major versions (base64 0.22, flate2 1, rfd 0.15). Tasks 1 and 9 resolve them against crates.io: never pass `--offline` (it rewrites dozens of locked versions to whatever the local cache holds). Expected `Cargo.lock` additions: Task 1, `base64 0.22.1` only (flate2 is already locked); Task 9, `rfd 0.15.4` and its transitive crates (`ashpd`, `async-fs`, `async-net`, `block2`, `futures-channel`, `pollster`, `ppv-lite86`, `rand`, `rand_chacha`, `rand_core`, `urlencoding`; newer patch versions of these are fine). Both tasks commit `magcoupling-rs/Cargo.lock`.
- "live recompute while dragging": `compute_all` runs every frame (17 us release). `sizing::solve` (1 to 3 ms release, longer on wasm) and the registry builds (the correction index, the input catalogue, the result index) never run per frame: the solve is debounced, the builds happen once (`OnceLock`).
- The panel's host API does not break: `MagcouplingPanel::new`, `ui(&mut self, &mut egui::Ui)`, `inputs`, `results`, `reset` keep their signatures (M5 hosts the panel in an `egui::Window`); everything else is added.
- UI text: egui's default fonts have no U+2192 (the spec's arrow) or U+2264, so the panel writes `Torque -> Magnets` and `<=`; Task 8's `every_text_the_panel_shows_has_glyphs_in_the_default_fonts` guards every drawn and hover text.
- Every gate run: `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate` line; then `git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda` (the linkage tests rewrite those PNGs; never commit them).
- `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml` before every commit, and `cargo fmt --check` clean. Every block below is already in rustfmt's layout.
- The blocks quote the files with LF line endings, as git stores them (Task 0 creates the worktree with a worktree-scoped `core.autocrlf=false`: this machine's system gitconfig sets it to true). "Create" writes a new file; "Replace the whole of" overwrites one; "replace ... with" is one exact replacement whose old text occurs once in the file at that point. If a block's old text is not found, stop and escalate; do not improvise a match (Task 0 checks that the files the blocks touch are still those of `8bad022`).
- Windows paths: the worktree path is short on purpose. A release build under a deep directory fails with `LNK1104` (MAX_PATH); if a build must run elsewhere, use a short path (for example `subst` a drive letter).
- Every commit message ends with exactly two lines: a `Co-Authored-By:` line naming the model that wrote the commit, then `Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9`. The blocks write `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; a `sonnet` executor writes its own model's name. Commit subjects: `feat(magcoupling-rs): ...` (Tasks 1 to 9), `test(magcoupling): ...` (Task 10), `docs(magcoupling): ...` (Task 11). Nothing is pushed.
- Docs change with code (repo rule, `docs/ai/01-meta.yaml`): Task 11 updates `magcoupling-rs/README.md` (status, layout, the panel, tests) and `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`, and commits this plan; the branch is not merged without it. Read `docs/ai/*.yaml` before starting (Task 0).
- Deploy: `deploy-web.yml` on `main` already builds and ships `/magcoupling/` on the next push of `main`. This plan pushes nothing; that push needs the user's explicit go.
- Model tiers (CLAUDE.md section 5): every task gives the exact code, so every task runs on `sonnet` (set `model: 'sonnet'`), Task 0 and the web smoke included. The per-task review runs on `sonnet` (`model: 'sonnet'`) for Tasks 0 to 4, 6, 7 and 9 to 11. The reviews of Task 5 (the registry semantics behind the corrected-vs-workbook markers, decision M41-10) and Task 8 (the sizing state machine: debounce, leaving the mode, the shown design) run on the session model: omit `model` with the comment `// session model: judgment (registry semantics / sizing UX)`, as CLAUDE.md's "when unsure, use the session model" asks. The whole-branch review after Task 11 runs on the session model (omit `model`, comment `// session model: final whole-branch review`). A `sonnet` attempt that ends `blocked`, leaves the gate red or is rejected in review escalates every retry to the session model.

## Review Focus

Six input classes the spec implies, each pinned by a test in the task that owns the code:

1. **A design file or share link holding a value outside its slider range** (hand-edited, or from another version). Expected: it loads as written (the engine's `set` checks type, finiteness and choices, not ranges), the results follow it, the row says "outside the slider range", and the first edit brings it back into the range. Tests: `a_value_outside_its_slider_range_loads_as_it_is` (Task 1), `an_input_outside_its_slider_range_is_kept_until_edited_and_flagged` (Task 4).
2. **Text that is no number typed into a value box** ("wide", a stray comma). Expected: nothing changes and no error shows; the box keeps the last value. Test: the last case of `a_typed_value_snaps_to_the_step_and_is_clamped_into_the_slider_range` (Task 4).
3. **Results that are not finite** (E13's measured drag of exactly 0 gives +inf; E20's positive beta without a rating gives NaN). Expected: the dashboard and the table show `+inf` and `NaN` (the existing `format_value`), and both exports keep them as text, never `null`. Test: `non_finite_results_export_as_text_in_csv_and_json` (Task 6).
4. **A broken or foreign share link** (truncated, padded with `=`, a linkage-tool mechanism link, a deflate bomb). Expected: padding and surrounding blanks are tolerated; anything else is refused with a message and the design is unchanged. Tests: `a_share_link_round_trips_and_is_url_safe`, `broken_share_links_are_refused`, `a_share_link_that_inflates_past_the_limit_is_refused` (Task 1), `a_broken_share_link_changes_nothing_and_says_why` (Task 7).
5. **A sizing target out of reach or invalid, and a design the solve refuses** (a target past the free variable's range, a target of 0 from a file, inputs with a code outside its choices). Expected: "Not reachable (best Y at X)" with the design at the best value, or "Sizing refused: ..." with the inputs as they are; never a panic, never a solve per frame. Tests: `an_unreachable_target_shows_the_best_value`, `a_refused_solve_shows_why_and_keeps_the_inputs` (Task 8's runner), `an_unreachable_target_shows_the_best_value_and_its_overshoot` (Task 8's panel), `a_malformed_sizing_state_is_refused`, `a_file_with_bad_inputs_and_a_bad_sizing_state_names_both` (Task 1).
6. **Timing and focus at the edges of an edit** (a held arrow key auto-repeating, typing in the results search while a solve waits, leaving Torque -> Magnets inside the debounce, Undo right after a start-up share link, idle frames over values off their step grid). Expected: a held key is one undo step; the search defers nothing; the value written on leaving is this design's solution; the first Undo keeps the shared design; an idle frame writes nothing. Tests: `idle_frames_change_no_input` (Tasks 4, 6, 7: every group open, off-grid values), `a_held_arrow_key_is_one_undo_step`, `a_focused_results_search_holds_back_no_undo_step`, `a_share_link_opened_at_start_up_is_not_an_undo_step` (Task 7), `typing_in_the_results_search_does_not_defer_the_solve`, `leaving_torque_to_magnets_while_solving_writes_the_current_solution` (Task 8).

## File Structure

Created (under `C:/Users/Cole/source/repos/lsim-mag-m41/`):

| File | Responsibility | Task |
|---|---|---|
| `magcoupling-rs/src/gui/session.rs` | `Design` (inputs and sizing state); the design file format (`design_to_json`, `design_from_json`, `LoadError`, `PATH_MIGRATIONS`); share links (`encode_share_payload`, `decode_share_payload`, `share_link`); `json_number`, `json_value` | 1 |
| `magcoupling-rs/src/gui/sizing.rs` | `SizingMode`, `SizingState`, the free variable's keys and labels (Task 1); `SizingRunner`, the debounced solve and its status line (Task 8) | 1, 8 |
| `magcoupling-rs/src/gui/history.rs` | `History<T>`: undo and redo of snapshots, one step per settled edit | 2 |
| `magcoupling-rs/src/gui/inputs.rs` | `InputCatalogue` (the Key design group and every input by package group and section, built once), `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `outside_range`, `text_hint`, `input_tooltip` | 3 |
| `magcoupling-rs/src/gui/input_ui.rs` | `input_row`: one input row of any field type (slider and value box, drop-down, optional, text), the changed dot, reset, tooltip; `slider` | 4 |
| `magcoupling-rs/src/gui/corrections.rs` | `CorrectionIndex`: the corrected-vs-workbook markers from the registry and the compiled golden files | 5 |
| `magcoupling-rs/src/gui/dashboard.rs` | `DASHBOARD`, `verdict_level`, `END_EFFECT_ROWS`, `STORED_3D_ROWS`, `result_info`, `result_tooltip` (the hover hook), `dashboard_lines`, `dashboard_ui`, `end_effect_banner` | 5, 6 |
| `magcoupling-rs/src/gui/results_table.rs` | `table_entries`, `search`, `exact_number`, `results_csv`, `results_json`, `ResultsTable`, `row_tooltip` | 6 |
| `magcoupling-rs/src/app/theme.rs` | The linkage app's CAD dark visuals and spacing, forced dark | 9 |
| `magcoupling-rs/src/app/files.rs` | `save` (a dialog natively, a download on the web), `DesignPicker` | 9 |

Modified: `magcoupling-rs/Cargo.toml` and `Cargo.lock` (Tasks 1, 8, 9), `magcoupling-rs/src/gui/mod.rs` (each new module), `magcoupling-rs/src/gui/format.rs` (Task 1: `non_finite_text`, the one text of a non-finite number), `magcoupling-rs/src/gui/panel.rs` (Tasks 4 to 10: the layout, the dashboard, the centre region, the session, the sizing controls), `magcoupling-rs/src/gui/test_support.rs` (Tasks 4, 6, 8: `select_all` and key taps, sized frames, `short_magnets`), `magcoupling-rs/src/app.rs` (Tasks 4, 9, 10), `magcoupling-rs/src/bin/magcoupling_web.rs` and `linkage-sim-rs/web/magcoupling/index.html` (Task 9), `.claude/workflows/gui-smoke.js` (Task 10), `magcoupling-rs/README.md` and `docs/ai/{02-system,03-structure,04-memory}.yaml`, `docs/ai/05-update-tracker.md` (Task 11). No engine file, test data file or Python file changes.

Order: the pure, testable cores first (session and sizing state, history, the input catalogue), then the panel region by region (inputs, dashboard, results table), then the wiring that uses the cores (session, sizing), the app (theme, files, the share-link entry), the web smoke, and the docs.

## Verification record

This plan was replayed before it was handed over: its blocks were applied in order to a fresh LF export of `8bad022` (`git -c core.autocrlf=false archive 8bad022`, under a short path; it differs from `167db09` only by the two `docs/ai/04-memory.yaml` lines Task 11 now resolves and records), with a cargo target directory shared with the development tree for the compiled dependencies, and every command the plan marks with an expected output was run and compared (the gate's linkage gates excepted, below). The replay parsed the blocks out of this file and applied them literally.

- **Red steps** failed as stated (compile errors naming the items each Step 3 adds); **green steps** passed.
- **After every task** the magcoupling gates passed: `cargo test` (engine), `cargo clippy --all-targets -- -D warnings`, the wasm32 check, `cargo test --features app`, `cargo clippy --all-targets --features app -- -D warnings`, the wasm32 clippy of the panel and the web entry, and `cargo fmt --check`; after Task 11 also gate 10 (the parity guard and its negative control), gate 11 (lock parity) and gate 12 (`1159 passed`, the differential data current). The linkage crate's gates 1 to 3 were not run: this plan does not touch linkage-sim-rs's code.
- **After every task the replayed tree equals, file for file, the tree in which that task was developed and proven** (`Cargo.lock` included: online resolution added exactly the crates listed in Global Constraints; after Task 11 the replayed tree also holds this plan, which that task commits).
- **The web bundle** was built once at the end (`build_magcoupling_web.sh`, 4.0 MB wasm, the parity guard passing) and served from `linkage-sim-rs/web/` with `python -m http.server`; Playwright opened `/magcoupling/?m=<the gui-smoke payload>`: canvas present, the console showed `magcoupling: loaded the design from the share link (sizing: Torque -> Magnets)` and `magcoupling sizing: Solved at 14.18 mm (hot-low torque 2.500 N·m)`, zero errors and zero warnings; the screenshot shows the dark theme, the inputs, the results table, the dashboard and a disabled Undo (the start-up link is no undo step). Then the web file picker, as Task 10's smoke step does it: a click at the "Load design" button's centre read from the screenshot (`page.mouse.click` through `browser_run_code_unsafe`) showed rfd's overlay and opened the browser's file chooser at once; `browser_file_upload` of the pinned design file (the Playwright MCP accepts files under the repository only, so it goes in the gitignored `.playwright-mcp/`) and the overlay's "Ok" logged `magcoupling: loaded a design file`, still zero errors; the next screenshot showed "Design file loaded", the face gap at 2.00 mm in Magnets -> Torque, and Undo enabled. Two defects the first smoke screenshots revealed were fixed in the tasks that own them before this replay: the arrow glyph drew as a box (Task 1's labels, Task 8's glyph test) and an idle page kept showing "Solving..." after a solve (Task 8's repaint request).

| After task | `cargo test --features app --lib` | Engine unit tests (`cargo test`) |
|---|---|---|
| 0 (base) | 162 | 139 |
| 1 | 182 | 139 |
| 2 | 189 | 139 |
| 3 | 200 | 139 |
| 4 | 205 | 139 |
| 5 | 222 | 139 |
| 6 | 231 | 139 |
| 7 | 249 | 139 |
| 8 | 268 | 139 |
| 9 | 274 | 139 |
| 10 | 275 | 139 |
| 11 | 275 | 139 |

Every integration test binary keeps Task 0's counts throughout (assumptions 8, deviations 54, differential 19, grades 11, material_library 4, material_links 11, parity 4, python_schema 7, robustness 12, schema 7, sizing 31, static_data 8; doc-tests 1 passed, 1 ignored).

Not exercised by the replay: the linkage crate's gates (untouched code), the native file dialogs (platform UI; `app/files.rs` is thin and compiled on both targets; the web picker is exercised by the smoke above), and the user's answers to the Decisions to confirm.

---

## Tasks

### Task 0: Worktree, baseline and context

**Model:** `sonnet` (creating the worktree, running commands and reporting output; CLAUDE.md section 5). Step 8 is the controller's.

Verification only; nothing to write.

**Files:**
- none

**Interfaces:**
- Consumes: `main` at `8bad022` or a later commit that leaves the files this plan's blocks touch unchanged (Step 2).
- Produces: the worktree `C:/Users/Cole/source/repos/lsim-mag-m41` on the new branch `magcoupling/m4-1` with LF line endings, a green baseline with its test counts, and the controller's record of the user's answers to the Decisions to confirm.

- [ ] **Step 1: Create the worktree**

This machine's system gitconfig sets `core.autocrlf=true`, which would check the files out with CRLF; every block below quotes them with LF, as git stores them. So the worktree is created with LF and keeps a worktree-scoped `core.autocrlf=false`.

Run:

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation log --oneline -1 main
git -C C:/Users/Cole/source/repos/linkage_simulation config extensions.worktreeConfig true
git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b magcoupling/m4-1 C:/Users/Cole/source/repos/lsim-mag-m41 main
git -C C:/Users/Cole/source/repos/lsim-mag-m41 config --worktree core.autocrlf false
git -C C:/Users/Cole/source/repos/lsim-mag-m41 rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: the `log` line is `8bad022 docs(ai): record two A-2 carry-overs (per-grade magnet specific heat; exact-equality modified flag vs slider rounding)` or a later commit; `worktree add` prints `Preparing worktree (new branch 'magcoupling/m4-1')`; then `magcoupling/m4-1`; no status lines.

- [ ] **Step 2: Check that the blocks' files are still those of `8bad022`**

The blocks quote these files as they are at `8bad022`. If `main` has moved on (for example plan A-3 landed first: see Merge order), the files the blocks touch must not have changed.

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 diff --stat 8bad022 HEAD -- magcoupling-rs/Cargo.toml magcoupling-rs/src/gui magcoupling-rs/src/app.rs magcoupling-rs/src/bin magcoupling-rs/README.md docs/ai linkage-sim-rs/web/magcoupling/index.html .claude/workflows/gui-smoke.js
```

Expected: no output. If any file is listed, stop and escalate: the blocks of the tasks that touch it must be re-derived first. (Other files, such as the engine's, may differ; then also expect Step 5's counts to differ.)

- [ ] **Step 3: Check the line endings**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 config --get core.autocrlf; head -c 2000 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs | tr -cd '\r' | wc -c
```

Expected: `false` and `0` carriage returns. If the count is not 0: run `git -C C:/Users/Cole/source/repos/lsim-mag-m41 rm -rq --cached . && git -C C:/Users/Cole/source/repos/lsim-mag-m41 reset -q --hard` and check again.

- [ ] **Step 4: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/01-meta.yaml`, `02-system.yaml`, `03-structure.yaml`, `04-memory.yaml` (the "Open (M4 plans)" items this plan resolves); `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/README.md` whole; the spec's M4 and Addendum A1 sections; the current panel `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs` and app `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs`; and this plan's Decisions to confirm.

- [ ] **Step 5: Run the crate's tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: the engine run: unit tests `139 passed`; `tests\assumptions.rs` 8; `tests\deviations.rs` 54; `tests\differential.rs` 19; `tests\grades.rs` 11; `tests\material_library.rs` 4; `tests\material_links.rs` 11; `tests\parity.rs` 4; `tests\python_schema.rs` 7; `tests\robustness.rs` 12; `tests\schema.rs` 7; `tests\sizing.rs` 31; `tests\static_data.rs` 8; doc-tests `1 passed; 0 failed; 1 ignored`. Then with `app`: `test result: ok. 162 passed` (the engine's 139 and the tracer's 23). If a count differs but every binary is `ok`, record the actual counts and read every later task's counts as offsets from them.

- [ ] **Step 6: Check the oracle Python**

Run:

```bash
ls C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe
```

Expected: the path. If it is missing, stop and escalate: every gate run uses it.

- [ ] **Step 7: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task0.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task0.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task0.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

(`magcoupling-rs/target/` may not exist before the first build; if the redirect fails, run `mkdir -p C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target` first.)

Expected: `exit=0`; the last three lines are `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0`; `cargo fmt --check` prints nothing.

- [ ] **Step 8: Decisions**

The controller asks the user the plan's **Decisions to confirm** (M41-1 to M41-15) before Task 1 and records the answers in the execution notes. Every task implements the recommended option and names the decision where it implements it; if the user picks another option, the task that implements it stops and escalates instead of improvising.

---

### Task 1: Design files, share links and the sizing state

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

The pure core every later task builds on (spec M4 "Session": "save/load design JSON; share link (as the linkage tool's `?m=`)"). `gui/sizing.rs` gets the sizing state of Addendum A1 (the mode, the free variable, the target torque: GUI state per A-2 decision A2-9), with stable keys for files and labels for the UI; the labels write `->` because egui's default fonts have no arrow glyph. `gui/session.rs` gets `Design` (the inputs and the sizing state) and its one JSON format: `{"format": "magcoupling-design", "version": 1, "inputs": {path: value}, "sizing": {...}}` with every input (decision M41-5) and the sizing state (M41-4); loading goes through `InputSet::set` and is all or nothing, the inputs and the sizing state both checked so the refusal names every problem of either (`LoadError::Refused`, M41-6); a path a later version renames or removes goes through `PATH_MIGRATIONS` (empty now; a rename adds an entry and bumps `DESIGN_VERSION`). A share link is the compact JSON deflated and URL-safe base64 encoded, as in `linkage-sim-rs/src/gui/state/file_io.rs`, read with or without `=` padding, and refused past 1 MB inflated. `json_number` writes a non-finite number as `"+inf"`, `"-inf"` or `"NaN"` (M41-15) through the new `format::non_finite_text`, the one text of a non-finite number that the display and Task 6's results export share. Feature `gui` gains serde_json (with `float_roundtrip`, so a file reads back bit for bit), base64 and flate2, linkage-sim-rs's major versions.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml` (feature `gui`: serde_json, base64, flate2)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.lock` (by cargo: adds `base64 0.22.1`)
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/sizing.rs`
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/session.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/format.rs` (`non_finite_text`; `format_number` uses it)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs` (`pub mod session; pub mod sizing;`)

**Interfaces:**
- Consumes: `engine::sizing::FreeVariable` (`ALL`, `path()`), `engine::meta::{FieldType, InputSet, Value, input_rows}`, `DesignInputs`.
- Produces (`magcoupling::gui::sizing`): `enum SizingMode { MagnetsToTorque, TorqueToMagnets }` with `ALL`, `const fn key(self) -> &'static str` (`"magnets_to_torque"`, `"torque_to_magnets"`), `const fn label(self) -> &'static str` (`"Magnets -> Torque"`, `"Torque -> Magnets"`), `fn from_key(&str) -> Option<SizingMode>`; `const fn variable_key(FreeVariable) -> &'static str` (`"axial_length"`, `"magnets_per_ring"`, `"ring_radius"`), `fn variable_from_key(&str) -> Option<FreeVariable>`, `const fn variable_label(FreeVariable) -> &'static str`; `const TARGET_RANGE_INPUT: &str = "metal.required_min_Nm"`; `struct SizingState { pub mode: SizingMode, pub variable: FreeVariable, pub target_Nm: f64 }` (`Copy`, `Default`: forward, axial length, 2.5); `const DEFAULT_TARGET_NM: f64 = 2.5`.
- Produces (`magcoupling::gui::session`): `struct Design { pub inputs: DesignInputs, pub sizing: SizingState }` (`Clone`, `Debug`, `Default`, `PartialEq`); `DESIGN_FORMAT`, `DESIGN_VERSION: u64 = 1`, `PUBLIC_BASE_URL = "https://linkage.colesorkness.com/magcoupling/"`, `SHARE_PARAM = "m"`, `MAX_DESIGN_BYTES: u64 = 1 << 20`; `type PathMigration = (u64, &'static str, Option<&'static str>)` (the last version that wrote the old path, the old path, the new path or `None` for removed), `const PATH_MIGRATIONS: &[PathMigration] = &[]`; `enum LoadError { NotJson(String), NotADesign(String), UnsupportedVersion(String), Refused { inputs: Vec<String>, sizing: Vec<String> }, Link(String) }` (`Display`, `Error`); `fn design_to_json(&Design) -> String`; `fn design_from_json(&str) -> Result<Design, LoadError>`; `fn encode_share_payload(&Design) -> String`; `fn decode_share_payload(&str) -> Result<Design, LoadError>`; `fn share_link(base: &str, &Design) -> String`; `pub(crate) fn json_number(f64) -> serde_json::Value`, `pub(crate) fn json_value(&Value) -> serde_json::Value`.
- Produces (`gui::format`): `pub(crate) fn non_finite_text(f64) -> Option<&'static str>` (`"+inf"`, `"-inf"`, `"NaN"`; `None` when finite).

- [ ] **Step 1: Write the failing tests**

Each new module starts as its module docs and its tests; Step 3 adds the code between them. The tests pin the round trip of every input type and of the sizing state, the format's shape, the share link's alphabet, padding and length (decision M41-5 quotes 2,462 characters for the default design), the all-or-nothing load with every problem named (the inputs' and the sizing state's together, several of the sizing state's at once), the path migrations (a made-up table: renamed twice, removed, an old and a new name in one file; and a guard on the real, empty table), the refusals (not JSON, not a design, a newer or malformed version, a malformed sizing state, broken and foreign links, a deflate bomb), a value outside its slider range loading as written (Review Focus 1), and the non-finite JSON text; `format.rs` gets the test of `non_finite_text`.

Create `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/session.rs`:

````rust
//! Design files and share links (spec M4 "Session": "save/load design JSON; share link (as
//! the linkage tool's `?m=`)").
//!
//! One format for both. A design file is JSON:
//!
//! ```json
//! {
//!   "format": "magcoupling-design",
//!   "version": 1,
//!   "inputs": { "coupling.npole": 10, "metal.measured_drag_Nm": null, ... },
//!   "sizing": { "mode": "magnets_to_torque", "free_variable": "axial_length", "target_torque_Nm": 2.5 }
//! }
//! ```
//!
//! `inputs` maps every input's dotted path (the Python `input_schema()` paths) to its value: a
//! number, an integer (counts and selector codes), a string, or `null` for an optional input
//! left blank. A file names every input, so it keeps its design if a later version changes a
//! default (decision M41-5); a reader takes a missing path at its default, and a path a later
//! version renamed or removed through [`PATH_MIGRATIONS`] (the file names the version that
//! wrote it). `sizing` is the sizing state (decision M41-4); a file without it is in the
//! forward mode.
//!
//! A share link carries the same JSON, compact, deflated and URL-safe base64 encoded: the
//! linkage tool's `?m=` scheme (`linkage-sim-rs/src/gui/state/file_io.rs`).
//!
//! Loading is all or nothing (decision M41-6): a file with an unknown path, a value of the
//! wrong type, a value [`InputSet::set`] refuses (a selector code outside its choices), a
//! newer version or a malformed sizing state changes nothing. The inputs and the sizing state
//! are both checked, so the refusal names every problem of either.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sizing::FreeVariable;

    /// A design with a value of every input type changed, and a non-default sizing state.
    fn edited() -> Design {
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 1.41; // f64
        inputs.coupling.npole = 12; // i64 count
        inputs.coupling.backiron = 0; // selector
        inputs.coupling.magnets.part_inner = "K&J BX0X04-N52".to_owned(); // text
        inputs.coupling.magnets.axial_length_mm = Some(13.5); // optional, set
        inputs.metal.measured_drag_Nm = Some(0.0123);
        inputs.temperature.slip_loss.cap_integral_T2m4 = 5.27e-10; // tiny
        inputs.metal.life_events = 2.5e7; // large
        Design {
            inputs,
            sizing: SizingState {
                mode: SizingMode::TorqueToMagnets,
                variable: FreeVariable::RingRadius,
                target_Nm: 3.25,
            },
        }
    }

    #[test]
    fn a_design_file_round_trips_bit_for_bit() {
        for design in [Design::default(), edited()] {
            let text = design_to_json(&design);
            assert_eq!(design_from_json(&text), Ok(design.clone()), "{text}");
        }
    }

    #[test]
    fn a_design_file_names_every_input_and_the_sizing_state() {
        let json: Json = serde_json::from_str(&design_to_json(&edited())).unwrap();
        let inputs = json["inputs"].as_object().unwrap();
        let rows = input_rows(&DesignInputs::default());
        assert_eq!(inputs.len(), rows.len());
        for row in &rows {
            assert!(inputs.contains_key(&row.path), "missing {}", row.path);
        }
        assert_eq!(json["format"], DESIGN_FORMAT);
        assert_eq!(json["version"], DESIGN_VERSION);
        assert_eq!(inputs["coupling.npole"], 12);
        assert_eq!(inputs["coupling.magnets.axial_length_mm"], 13.5);
        assert_eq!(inputs["coupling.magnets.grade_inner"], "");
        assert_eq!(json["sizing"]["mode"], "torque_to_magnets");
        assert_eq!(json["sizing"]["free_variable"], "ring_radius");
        assert_eq!(json["sizing"]["target_torque_Nm"], 3.25);
        // A blank optional input is null, not 0 or a missing key.
        let default: Json = serde_json::from_str(&design_to_json(&Design::default())).unwrap();
        assert_eq!(default["inputs"]["metal.measured_drag_Nm"], Json::Null);
    }

    #[test]
    fn a_share_link_round_trips_and_is_url_safe() {
        for design in [Design::default(), edited()] {
            let link = share_link(PUBLIC_BASE_URL, &design);
            let payload = link
                .strip_prefix("https://linkage.colesorkness.com/magcoupling/?m=")
                .expect("the link is the base, then ?m=");
            assert!(
                payload
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
                "{payload}"
            );
            assert_eq!(decode_share_payload(payload), Ok(design.clone()));
            // Pasted with padding or surrounding whitespace, it still loads.
            let padded = format!("  {payload}{}\n", "=".repeat((4 - payload.len() % 4) % 4));
            assert_eq!(decode_share_payload(&padded), Ok(design));
        }
    }

    #[test]
    fn a_share_link_of_the_default_design_stays_short() {
        // Decision M41-5 quotes this length (2,462 characters at the time of writing): the
        // whole design is in the link.
        let payload = encode_share_payload(&Design::default());
        assert!(payload.len() < 2_500, "{} characters", payload.len());
    }

    #[test]
    fn a_missing_path_or_sizing_state_takes_the_default() {
        let text =
            r#"{"format": "magcoupling-design", "version": 1, "inputs": {"coupling.npole": 14}}"#;
        let design = design_from_json(text).unwrap();
        let mut want = DesignInputs::default();
        want.coupling.npole = 14;
        assert_eq!(design.inputs, want);
        assert_eq!(design.sizing, SizingState::default());
        let bare = r#"{"format": "magcoupling-design", "version": 1}"#;
        assert_eq!(design_from_json(bare), Ok(Design::default()));
    }

    #[test]
    fn a_value_outside_its_slider_range_loads_as_it_is() {
        // A hand-edited file or one from another version (Review Focus 1): set() checks type,
        // finiteness and choices, not the slider range, so the value is kept; the panel flags it.
        let text = r#"{"format": "magcoupling-design", "version": 1, "inputs": {"metal.face_gap_mm": 7.5, "coupling.npole": 64}}"#;
        let design = design_from_json(text).unwrap();
        assert_eq!(design.inputs.metal.face_gap_mm, 7.5);
        assert_eq!(design.inputs.coupling.npole, 64);
        assert_eq!(
            decode_share_payload(&encode_share_payload(&design)),
            Ok(design)
        );
    }

    #[test]
    fn a_float_input_accepts_a_whole_number_and_a_count_refuses_a_fraction() {
        let text = r#"{"format": "magcoupling-design", "version": 1,
            "inputs": {"metal.face_gap_mm": 2, "coupling.npole": 12.0}}"#;
        let Err(LoadError::Refused { inputs, sizing }) = design_from_json(text) else {
            panic!("12.0 is not a whole number in JSON")
        };
        assert_eq!(
            inputs,
            vec!["coupling.npole: expected a whole number, got 12.0".to_owned()]
        );
        assert!(sizing.is_empty());
        let text =
            r#"{"format": "magcoupling-design", "version": 1, "inputs": {"metal.face_gap_mm": 2}}"#;
        assert_eq!(
            design_from_json(text).unwrap().inputs.metal.face_gap_mm,
            2.0
        );
    }

    #[test]
    fn every_problem_in_the_inputs_is_reported_and_nothing_loads() {
        let text = r#"{"format": "magcoupling-design", "version": 1, "inputs": {
            "coupling.backiron": 7,
            "coupling.no_such": 1,
            "metal.face_gap_mm": "wide",
            "coupling.magnets.part_inner": null,
            "metal.measured_drag_Nm": "+inf",
            "metal.web_mm": 3.0
        }}"#;
        let Err(LoadError::Refused { inputs, sizing }) = design_from_json(text) else {
            panic!("refused")
        };
        assert!(sizing.is_empty());
        // serde_json's map is ordered by key.
        assert_eq!(
            inputs,
            vec![
                "coupling.backiron: 7 is not one of the choices".to_owned(),
                "coupling.magnets.part_inner: expected a string, got null".to_owned(),
                "coupling.no_such: no such input".to_owned(),
                "metal.face_gap_mm: expected a number, got \"wide\"".to_owned(),
                "metal.measured_drag_Nm: expected a number or null, got \"+inf\"".to_owned(),
            ]
        );
    }

    #[test]
    fn files_that_are_not_designs_are_refused() {
        let refuse = |text: &str| design_from_json(text).unwrap_err();
        assert!(matches!(refuse("not json"), LoadError::NotJson(_)));
        assert!(matches!(refuse("[1, 2]"), LoadError::NotADesign(_)));
        assert!(matches!(
            refuse(r#"{"format": "mechanism", "version": 1}"#),
            LoadError::NotADesign(_)
        ));
        assert!(matches!(
            refuse(r#"{"format": "magcoupling-design", "version": 1, "extra": 0}"#),
            LoadError::NotADesign(_)
        ));
        assert_eq!(
            refuse(r#"{"format": "magcoupling-design", "version": 2}"#),
            LoadError::UnsupportedVersion("2".to_owned())
        );
        for version in ["0", "1.5", "\"1\"", "-1"] {
            let text = format!(r#"{{"format": "magcoupling-design", "version": {version}}}"#);
            assert!(
                matches!(refuse(&text), LoadError::UnsupportedVersion(_)),
                "{version}"
            );
        }
        assert_eq!(
            refuse(r#"{"format": "magcoupling-design"}"#),
            LoadError::UnsupportedVersion("missing".to_owned())
        );
        assert_eq!(
            refuse(r#"{"format": "magcoupling-design", "version": 1, "inputs": []}"#),
            LoadError::Refused {
                inputs: vec!["\"inputs\" is not an object".to_owned()],
                sizing: Vec::new(),
            }
        );
    }

    #[test]
    fn a_malformed_sizing_state_is_refused() {
        let with = |sizing: &str| {
            design_from_json(&format!(
                r#"{{"format": "magcoupling-design", "version": 1, "sizing": {sizing}}}"#
            ))
        };
        for bad in [
            r#"{"mode": "sideways"}"#,
            r#"{"free_variable": "colour"}"#,
            r#"{"target_torque_Nm": 0}"#,
            r#"{"target_torque_Nm": -1.5}"#,
            r#"{"target_torque_Nm": "2.5"}"#,
            r#"{"target_Nm": 2.5}"#,
            r#"[]"#,
        ] {
            assert!(
                matches!(with(bad), Err(LoadError::Refused { inputs, sizing })
                    if inputs.is_empty() && sizing.len() == 1),
                "{bad}"
            );
        }
        // Every problem of the sizing state is named, in key order.
        assert_eq!(
            with(r#"{"mode": "x", "target_torque_Nm": 0}"#),
            Err(LoadError::Refused {
                inputs: Vec::new(),
                sizing: vec![
                    "unknown mode \"x\"".to_owned(),
                    "target torque 0 is not a positive number".to_owned(),
                ],
            })
        );
        let partial = with(r#"{"mode": "torque_to_magnets"}"#).unwrap().sizing;
        assert_eq!(
            partial,
            SizingState {
                mode: SizingMode::TorqueToMagnets,
                ..SizingState::default()
            }
        );
    }

    #[test]
    fn a_file_with_bad_inputs_and_a_bad_sizing_state_names_both() {
        let text = r#"{"format": "magcoupling-design", "version": 1,
            "inputs": {"coupling.backiron": 7}, "sizing": {"mode": "sideways"}}"#;
        let error = design_from_json(text).unwrap_err();
        assert_eq!(
            error,
            LoadError::Refused {
                inputs: vec!["coupling.backiron: 7 is not one of the choices".to_owned()],
                sizing: vec!["unknown mode \"sideways\"".to_owned()],
            }
        );
        assert_eq!(
            error.to_string(),
            "inputs refused: coupling.backiron: 7 is not one of the choices; \
             sizing state refused: unknown mode \"sideways\""
        );
    }

    /// A made-up history of renames and a removal, for the migration tests (the real table,
    /// [`PATH_MIGRATIONS`], is empty while every path is version 1's).
    const RENAMES: [PathMigration; 3] = [
        (1, "metal.gap_mm", Some("metal.face_gap_v2_mm")),
        (1, "coupling.retired_flag", None),
        (2, "metal.face_gap_v2_mm", Some("metal.face_gap_mm")),
    ];

    #[test]
    fn an_older_file_s_renamed_and_removed_paths_are_migrated() {
        let json = |text: &str| -> Json { serde_json::from_str(text).unwrap() };
        // Version 1 wrote metal.gap_mm: renamed twice since; its removed input is dropped.
        let v1 = json(r#"{"metal.gap_mm": 2.0, "coupling.retired_flag": 1, "coupling.npole": 12}"#);
        let inputs = inputs_from(&v1, 1, &RENAMES).unwrap();
        assert_eq!(inputs.metal.face_gap_mm, 2.0);
        assert_eq!(inputs.coupling.npole, 12);
        // Version 2 wrote the middle name; its old name is no input of version 2.
        let v2 = json(r#"{"metal.face_gap_v2_mm": 1.5}"#);
        assert_eq!(
            inputs_from(&v2, 2, &RENAMES).unwrap().metal.face_gap_mm,
            1.5
        );
        assert_eq!(
            inputs_from(&json(r#"{"metal.gap_mm": 2.0}"#), 2, &RENAMES),
            Err(vec!["metal.gap_mm: no such input".to_owned()])
        );
        // A file of the version after the last rename reads the current paths only.
        assert_eq!(
            inputs_from(&v2, 3, &RENAMES),
            Err(vec!["metal.face_gap_v2_mm: no such input".to_owned()])
        );
        // An old and a new name of one input in one file: refused, not silently one of them.
        let both = json(r#"{"metal.face_gap_mm": 1.0, "metal.gap_mm": 2.0}"#);
        assert_eq!(
            inputs_from(&both, 1, &RENAMES),
            Err(vec![
                "metal.gap_mm: metal.face_gap_mm is given twice".to_owned()
            ])
        );
    }

    #[test]
    fn every_path_migration_leads_to_an_input_of_this_version() {
        let paths: Vec<String> = input_rows(&DesignInputs::default())
            .into_iter()
            .map(|row| row.path)
            .collect();
        let mut last_version = 1;
        for &(written_by, old, _) in PATH_MIGRATIONS {
            assert!(written_by >= last_version, "oldest first: {old}");
            last_version = written_by;
            assert!(
                written_by < DESIGN_VERSION,
                "{old}: bump DESIGN_VERSION with it"
            );
            assert!(!paths.iter().any(|p| p == old), "{old} is still an input");
            if let Some(now) = migrated_path(old, written_by, PATH_MIGRATIONS) {
                assert!(
                    paths.iter().any(|p| p == now),
                    "{old} leads to {now}, no input"
                );
            }
        }
    }

    #[test]
    fn broken_share_links_are_refused() {
        assert!(matches!(
            decode_share_payload("!!!"),
            Err(LoadError::Link(_))
        ));
        let not_deflate = LINK_BASE64.encode(b"plain text, not deflate");
        assert!(matches!(
            decode_share_payload(&not_deflate),
            Err(LoadError::Link(_))
        ));
        // A link to the linkage tool's mechanism decodes but is not a design.
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(br#"{"joints": []}"#).unwrap();
        let mechanism = LINK_BASE64.encode(encoder.finish().unwrap());
        assert!(matches!(
            decode_share_payload(&mechanism),
            Err(LoadError::NotADesign(_))
        ));
    }

    #[test]
    fn a_share_link_that_inflates_past_the_limit_is_refused() {
        let huge = vec![b' '; (MAX_DESIGN_BYTES + 10) as usize];
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&huge).unwrap();
        let payload = LINK_BASE64.encode(encoder.finish().unwrap());
        assert!(payload.len() < 10_000, "deflate packs the blanks");
        let Err(LoadError::Link(message)) = decode_share_payload(&payload) else {
            panic!("refused")
        };
        assert!(message.contains("inflates past"), "{message}");
    }

    #[test]
    fn non_finite_numbers_are_strings_in_json() {
        assert_eq!(json_number(f64::INFINITY), Json::from("+inf"));
        assert_eq!(json_number(f64::NEG_INFINITY), Json::from("-inf"));
        assert_eq!(json_number(f64::NAN), Json::from("NaN"));
        assert_eq!(json_number(-0.5), serde_json::json!(-0.5));
        assert_eq!(json_value(&Value::None), Json::Null);
        assert_eq!(json_value(&Value::Int(-3)), Json::from(-3));
    }

    #[test]
    fn load_errors_read_as_sentences() {
        assert_eq!(
            LoadError::UnsupportedVersion("2".to_owned()).to_string(),
            "design file version 2: this calculator reads version 1 and older"
        );
        let refused = |inputs: &[&str], sizing: &[&str]| {
            LoadError::Refused {
                inputs: inputs.iter().map(|s| s.to_string()).collect(),
                sizing: sizing.iter().map(|s| s.to_string()).collect(),
            }
            .to_string()
        };
        assert_eq!(
            refused(&["a: x", "b: y"], &[]),
            "inputs refused: a: x; b: y"
        );
        assert_eq!(refused(&[], &["m"]), "sizing state refused: m");
        assert_eq!(
            refused(&["a: x"], &["m", "n"]),
            "inputs refused: a: x; sizing state refused: m; n"
        );
    }
}
````

Create `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/sizing.rs`:

```rust
//! The sizing mode of the Key design group (spec Addendum A1): Magnets → Torque, the forward
//! calculation, or Torque → Magnets, inverse sizing by [`crate::engine::sizing::solve`].
//!
//! The mode, the free variable and the target torque are GUI state (A-2 decision A2-9): the
//! engine takes them as arguments. Design files and share links record them (decision M41-4).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DesignInputs;

    #[test]
    fn every_mode_and_free_variable_round_trips_through_its_key() {
        for mode in SizingMode::ALL {
            assert_eq!(SizingMode::from_key(mode.key()), Some(mode));
        }
        for variable in FreeVariable::ALL {
            assert_eq!(variable_from_key(variable_key(variable)), Some(variable));
        }
        assert_eq!(SizingMode::from_key("Torque"), None);
        assert_eq!(variable_from_key(""), None);
    }

    #[test]
    fn the_default_state_is_the_forward_calculation_at_the_hot_minimum() {
        let state = SizingState::default();
        assert_eq!(state.mode, SizingMode::MagnetsToTorque);
        assert_eq!(state.variable, FreeVariable::AxialLength);
        assert_eq!(
            state.target_Nm,
            DesignInputs::default().metal.required_min_Nm
        );
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/format.rs`, replace:

```rust
    }

    #[test]
    fn integers_text_and_none_show_as_they_are() {
        assert_eq!(format_value(&Value::Int(10)), "10");
        assert_eq!(format_value(&Value::Int(-3)), "-3");
```

with:

```rust
    }

    #[test]
    fn only_non_finite_numbers_have_a_non_finite_text() {
        assert_eq!(non_finite_text(f64::INFINITY), Some("+inf"));
        assert_eq!(non_finite_text(f64::NEG_INFINITY), Some("-inf"));
        assert_eq!(non_finite_text(f64::NAN), Some("NaN"));
        assert_eq!(non_finite_text(-f64::NAN), Some("NaN"));
        for finite in [0.0, -0.0, f64::MAX, f64::MIN, f64::MIN_POSITIVE, 2.5] {
            assert_eq!(non_finite_text(finite), None, "{finite}");
        }
    }

    #[test]
    fn integers_text_and_none_show_as_they_are() {
        assert_eq!(format_value(&Value::Int(10)), "10");
        assert_eq!(format_value(&Value::Int(-3)), "-3");
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs`, replace:

```rust

mod format;
mod panel;
#[cfg(test)]
pub(crate) mod test_support;

```

with:

```rust

mod format;
mod panel;
pub mod session;
pub mod sizing;
#[cfg(test)]
pub(crate) mod test_support;

```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | sort -u | head -8
```

Expected: the build fails with compile errors naming the items Step 3 adds, among them ``error[E0412]: cannot find type `Design` in this scope``, ``error[E0425]: cannot find function `design_from_json` in this scope``, ``error[E0425]: cannot find function `decode_share_payload` in this scope`` and ``error[E0433]: failed to resolve: use of undeclared type `SizingMode` `` (the exact order may vary).

- [ ] **Step 3: Add the dependencies and write the two modules**

The dependencies first (cargo resolves base64 against crates.io on the next build; never `--offline`), then the code above each module's tests.

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml`, replace:

```toml
default = []
# The egui panel `gui::MagcouplingPanel`, hostable by any egui app (the
# standalone app, M4, and a window in linkage-sim-rs, M5). egui only: no
# windowing, no eframe.
gui = ["dep:egui"]
# The standalone app: `app::MagcouplingApp` and the binaries `magcoupling-app`
# (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen).
app = [
```

with:

```toml
default = []
# The egui panel `gui::MagcouplingPanel`, hostable by any egui app (the
# standalone app, M4, and a window in linkage-sim-rs, M5). egui only: no
# windowing, no eframe, no file dialogs. serde_json, base64 and flate2 write
# and read design files, share links and the results export.
gui = ["dep:egui", "dep:serde_json", "dep:base64", "dep:flate2"]
# The standalone app: `app::MagcouplingApp` and the binaries `magcoupling-app`
# (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen).
app = [
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml`, replace:

```toml
egui = { version = "0.32", optional = true }
eframe = { version = "0.32", optional = true }
log = { version = "0.4", optional = true }

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
env_logger = { version = "0.11", optional = true }
```

with:

```toml
egui = { version = "0.32", optional = true }
eframe = { version = "0.32", optional = true }
log = { version = "0.4", optional = true }
# Design files, share links and the results export (feature gui). The same
# crates and major versions as linkage-sim-rs's `?m=` share links (deflate,
# then URL-safe base64). float_roundtrip: a design file reads back bit for bit.
serde_json = { version = "1", optional = true, features = ["float_roundtrip"] }
base64 = { version = "0.22", optional = true }
flate2 = { version = "1", optional = true }

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
env_logger = { version = "0.11", optional = true }
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/format.rs`, replace:

```rust
    }
}

/// A number to [`SIGNIFICANT_DIGITS`] significant digits: fixed point when it
/// rounds to at least 1e-3 and below 1e6, scientific outside; zero (either
/// sign) as `0`.
fn format_number(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "+inf" } else { "-inf" }.to_owned();
    }
    if x == 0.0 {
        return "0".to_owned();
```

with:

```rust
    }
}

/// The text of a number that is not finite: `+inf`, `-inf` or `NaN`; `None` for a finite
/// number. The display and both results exports share it, so a non-finite value reads the
/// same everywhere (JSON has no infinity or NaN: the exports write this text, decision M41-15).
pub(crate) fn non_finite_text(x: f64) -> Option<&'static str> {
    if x.is_nan() {
        Some("NaN")
    } else if x.is_infinite() {
        Some(if x > 0.0 { "+inf" } else { "-inf" })
    } else {
        None
    }
}

/// A number to [`SIGNIFICANT_DIGITS`] significant digits: fixed point when it
/// rounds to at least 1e-3 and below 1e6, scientific outside; zero (either
/// sign) as `0`.
fn format_number(x: f64) -> String {
    if let Some(text) = non_finite_text(x) {
        return text.to_owned();
    }
    if x == 0.0 {
        return "0".to_owned();
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/session.rs`, replace:

```rust
//! wrong type, a value [`InputSet::set`] refuses (a selector code outside its choices), a
//! newer version or a malformed sizing state changes nothing. The inputs and the sizing state
//! are both checked, so the refusal names every problem of either.

#[cfg(test)]
mod tests {
```

with:

```rust
//! wrong type, a value [`InputSet::set`] refuses (a selector code outside its choices), a
//! newer version or a malformed sizing state changes nothing. The inputs and the sizing state
//! are both checked, so the refusal names every problem of either.

use std::fmt;
use std::io::{Read, Write};

use base64::Engine;
use base64::alphabet::URL_SAFE;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use flate2::Compression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use serde_json::{Map, Value as Json};

use crate::DesignInputs;
use crate::engine::meta::{FieldType, InputSet, Value, input_rows};
use crate::gui::format::non_finite_text;
use crate::gui::sizing::{SizingMode, SizingState, variable_from_key, variable_key};

/// The `format` of a design file.
pub const DESIGN_FORMAT: &str = "magcoupling-design";

/// The version this build writes and the newest it reads. Bump it when a path is renamed or
/// removed (with its [`PATH_MIGRATIONS`] entry) or a value's meaning changes, never for an
/// added input (a reader takes a missing path at its default).
pub const DESIGN_VERSION: u64 = 1;

/// An input path a later version renamed or removed: the last version that wrote the old
/// path, the old path, and the new path (`None`: the input was removed and its value is
/// dropped).
pub type PathMigration = (u64, &'static str, Option<&'static str>);

/// Every rename and removal of an input path since version 1, oldest first. A rename or a
/// removal adds its entry here and bumps [`DESIGN_VERSION`] in the same change: a reader
/// refuses an unknown path (decision M41-6), so without the entry every older file and share
/// link would stop opening.
pub const PATH_MIGRATIONS: &[PathMigration] = &[];

/// The public page share links point at when the host does not know its own address (the
/// native app). The web app uses its own address instead.
pub const PUBLIC_BASE_URL: &str = "https://linkage.colesorkness.com/magcoupling/";

/// The query parameter of a share link: the linkage tool's `?m=`.
pub const SHARE_PARAM: &str = "m";

/// The most bytes a share link may inflate to: a design is about 6 kB, so anything larger is
/// not one (and a deflate bomb stops here).
pub const MAX_DESIGN_BYTES: u64 = 1 << 20;

/// A design: what a file or a share link holds.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Design {
    pub inputs: DesignInputs,
    pub sizing: SizingState,
}

/// Why a design file or a share link was refused. Nothing was changed.
#[derive(Clone, Debug, PartialEq)]
pub enum LoadError {
    /// Not JSON at all (the parser's message).
    NotJson(String),
    /// JSON, but not a design file: no `"format": "magcoupling-design"`, or the top level is
    /// not an object, or it has keys a design file does not.
    NotADesign(String),
    /// Written by a newer version (or a version that is not a whole number).
    UnsupportedVersion(String),
    /// Values a design cannot hold: one message per problem, each naming its path or key in
    /// path order, the inputs' and the sizing state's (either list may be empty, not both).
    Refused {
        inputs: Vec<String>,
        sizing: Vec<String>,
    },
    /// A share link that does not decode (base64 or deflate).
    Link(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::NotJson(message) => write!(f, "not a JSON file: {message}"),
            LoadError::NotADesign(message) => write!(f, "not a design file: {message}"),
            LoadError::UnsupportedVersion(version) => write!(
                f,
                "design file version {version}: this calculator reads version {DESIGN_VERSION} and older"
            ),
            LoadError::Refused { inputs, sizing } => {
                let mut parts = Vec::new();
                if !inputs.is_empty() {
                    parts.push(format!("inputs refused: {}", inputs.join("; ")));
                }
                if !sizing.is_empty() {
                    parts.push(format!("sizing state refused: {}", sizing.join("; ")));
                }
                write!(f, "{}", parts.join("; "))
            }
            LoadError::Link(message) => write!(f, "share link refused: {message}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// A number as JSON. JSON has no infinity or NaN (serde_json would write `null`, which reads
/// back as "not entered"), so a non-finite number is the string `"+inf"`, `"-inf"` or `"NaN"`,
/// the panel's display text (decision M41-15). The results export uses it too.
pub(crate) fn json_number(x: f64) -> Json {
    match non_finite_text(x) {
        Some(text) => Json::from(text),
        None => {
            Json::Number(serde_json::Number::from_f64(x).expect("a finite number is a JSON number"))
        }
    }
}

/// A field value as JSON (numbers by [`json_number`]).
pub(crate) fn json_value(value: &Value) -> Json {
    match value {
        Value::Num(x) => json_number(*x),
        Value::Int(i) => Json::from(*i),
        Value::Text(text) => Json::String(text.clone()),
        Value::None => Json::Null,
    }
}

/// The design as a JSON object.
fn design_json(design: &Design) -> Json {
    let mut inputs = Map::new();
    for row in input_rows(&design.inputs) {
        inputs.insert(row.path, json_value(&row.value));
    }
    let sizing = &design.sizing;
    let mut sizing_json = Map::new();
    sizing_json.insert("mode".to_owned(), Json::from(sizing.mode.key()));
    sizing_json.insert(
        "free_variable".to_owned(),
        Json::from(variable_key(sizing.variable)),
    );
    sizing_json.insert("target_torque_Nm".to_owned(), json_number(sizing.target_Nm));
    let mut top = Map::new();
    top.insert("format".to_owned(), Json::from(DESIGN_FORMAT));
    top.insert("version".to_owned(), Json::from(DESIGN_VERSION));
    top.insert("inputs".to_owned(), Json::Object(inputs));
    top.insert("sizing".to_owned(), Json::Object(sizing_json));
    Json::Object(top)
}

/// The design file text: every input and the sizing state, indented.
pub fn design_to_json(design: &Design) -> String {
    let mut text = serde_json::to_string_pretty(&design_json(design))
        .expect("a JSON value of maps, strings and numbers always serializes");
    text.push('\n');
    text
}

/// The input value a JSON value stands for, given the field's type; `None` when it does not
/// fit the type (a string for a number, a fraction for a count).
fn input_value(ty: FieldType, json: &Json) -> Option<Value> {
    match (ty, json) {
        (FieldType::F64, Json::Number(n)) => n.as_f64().map(Value::Num),
        (FieldType::I64, Json::Number(n)) => n.as_i64().map(Value::Int),
        (FieldType::OptF64, Json::Null) => Some(Value::None),
        (FieldType::OptF64, Json::Number(n)) => n.as_f64().map(Value::Num),
        (FieldType::Text, Json::String(text)) => Some(Value::Text(text.clone())),
        _ => None,
    }
}

/// What a field of type `ty` expects, for a refusal message.
fn expected(ty: FieldType) -> &'static str {
    match ty {
        FieldType::F64 => "a number",
        FieldType::I64 => "a whole number",
        FieldType::OptF64 => "a number or null",
        FieldType::Text => "a string",
        FieldType::NumOrText => "a number or a string",
    }
}

/// The path an input written as `path` by a file of `version` has now: every migration of a
/// version at or after `version` applied in order (a path renamed twice follows both); `None`
/// when a migration removed it.
fn migrated_path<'a>(path: &'a str, version: u64, migrations: &[PathMigration]) -> Option<&'a str> {
    let mut current = path;
    for &(written_by, old, new) in migrations {
        if version <= written_by && current == old {
            current = new?;
        }
    }
    Some(current)
}

/// The inputs a file's `inputs` object gives: the defaults with every listed path set, a path
/// an older `version` wrote taken through `migrations` first; or every problem.
fn inputs_from(
    json: &Json,
    version: u64,
    migrations: &[PathMigration],
) -> Result<DesignInputs, Vec<String>> {
    let Json::Object(map) = json else {
        return Err(vec!["\"inputs\" is not an object".to_owned()]);
    };
    let mut inputs = DesignInputs::default();
    let types: Vec<(String, FieldType)> = input_rows(&inputs)
        .into_iter()
        .map(|row| (row.path, row.meta.ty))
        .collect();
    let mut problems = Vec::new();
    let mut set_paths: Vec<&str> = Vec::new();
    for (written, json) in map {
        let Some(path) = migrated_path(written, version, migrations) else {
            continue; // removed by a later version: the value means nothing now
        };
        let Some(ty) = types.iter().find(|(p, _)| p == path).map(|(_, ty)| *ty) else {
            problems.push(format!("{written}: no such input"));
            continue;
        };
        if set_paths.contains(&path) {
            problems.push(format!("{written}: {path} is given twice"));
            continue;
        }
        set_paths.push(path);
        match input_value(ty, json) {
            Some(value) => {
                if let Err(error) = inputs.set(path, value) {
                    problems.push(error.to_string());
                }
            }
            None => problems.push(format!("{written}: expected {}, got {json}", expected(ty))),
        }
    }
    if problems.is_empty() {
        Ok(inputs)
    } else {
        Err(problems)
    }
}

/// The sizing state a file's `sizing` object gives; or every problem.
fn sizing_from(json: &Json) -> Result<SizingState, Vec<String>> {
    let Json::Object(map) = json else {
        return Err(vec!["\"sizing\" is not an object".to_owned()]);
    };
    let mut state = SizingState::default();
    let mut problems = Vec::new();
    for (key, value) in map {
        match (key.as_str(), value) {
            ("mode", Json::String(name)) => match SizingMode::from_key(name) {
                Some(mode) => state.mode = mode,
                None => problems.push(format!("unknown mode {name:?}")),
            },
            ("free_variable", Json::String(name)) => match variable_from_key(name) {
                Some(variable) => state.variable = variable,
                None => problems.push(format!("unknown free variable {name:?}")),
            },
            ("target_torque_Nm", Json::Number(n)) => match n.as_f64() {
                Some(target) if target.is_finite() && target > 0.0 => state.target_Nm = target,
                _ => problems.push(format!("target torque {n} is not a positive number")),
            },
            (key, value) => problems.push(format!("unexpected {key:?}: {value}")),
        }
    }
    if problems.is_empty() {
        Ok(state)
    } else {
        Err(problems)
    }
}

/// Reads a design file (the module docs give the format and the rules).
pub fn design_from_json(text: &str) -> Result<Design, LoadError> {
    let json: Json =
        serde_json::from_str(text).map_err(|error| LoadError::NotJson(error.to_string()))?;
    let Json::Object(top) = &json else {
        return Err(LoadError::NotADesign(
            "the top level is not an object".to_owned(),
        ));
    };
    if top.get("format") != Some(&Json::from(DESIGN_FORMAT)) {
        return Err(LoadError::NotADesign(format!(
            "no \"format\": \"{DESIGN_FORMAT}\""
        )));
    }
    let version = match top.get("version") {
        Some(json) => json
            .as_u64()
            .filter(|v| (1..=DESIGN_VERSION).contains(v))
            .ok_or_else(|| LoadError::UnsupportedVersion(json.to_string()))?,
        None => return Err(LoadError::UnsupportedVersion("missing".to_owned())),
    };
    if let Some(key) = top
        .keys()
        .find(|key| !matches!(key.as_str(), "format" | "version" | "inputs" | "sizing"))
    {
        return Err(LoadError::NotADesign(format!("unexpected key {key:?}")));
    }
    // Both are checked before either refuses, so the refusal names every problem.
    let inputs = match top.get("inputs") {
        Some(json) => inputs_from(json, version, PATH_MIGRATIONS),
        None => Ok(DesignInputs::default()),
    };
    let sizing = match top.get("sizing") {
        Some(json) => sizing_from(json),
        None => Ok(SizingState::default()),
    };
    match (inputs, sizing) {
        (Ok(inputs), Ok(sizing)) => Ok(Design { inputs, sizing }),
        (inputs, sizing) => Err(LoadError::Refused {
            inputs: inputs.err().unwrap_or_default(),
            sizing: sizing.err().unwrap_or_default(),
        }),
    }
}

/// URL-safe base64 that writes no padding and reads it either way (a link pasted with `=`
/// padding still loads).
const LINK_BASE64: GeneralPurpose = GeneralPurpose::new(
    &URL_SAFE,
    GeneralPurposeConfig::new()
        .with_encode_padding(false)
        .with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// The `?m=` value of a share link: the compact design JSON, deflated, URL-safe base64.
pub fn encode_share_payload(design: &Design) -> String {
    let json = design_json(design).to_string();
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(json.as_bytes())
        .expect("writing to a Vec cannot fail");
    let compressed = encoder.finish().expect("finishing into a Vec cannot fail");
    LINK_BASE64.encode(compressed)
}

/// The design a `?m=` value holds (surrounding whitespace ignored).
pub fn decode_share_payload(payload: &str) -> Result<Design, LoadError> {
    let bytes = LINK_BASE64
        .decode(payload.trim())
        .map_err(|error| LoadError::Link(format!("not URL-safe base64 ({error})")))?;
    let mut text = String::new();
    DeflateDecoder::new(&bytes[..])
        .take(MAX_DESIGN_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| LoadError::Link(format!("not a compressed design ({error})")))?;
    if text.len() as u64 > MAX_DESIGN_BYTES {
        return Err(LoadError::Link(format!(
            "it inflates past {MAX_DESIGN_BYTES} bytes"
        )));
    }
    design_from_json(&text)
}

/// A share link: `base` (the page's address, e.g. [`PUBLIC_BASE_URL`]) with the `?m=` payload.
pub fn share_link(base: &str, design: &Design) -> String {
    format!("{base}?{SHARE_PARAM}={}", encode_share_payload(design))
}

#[cfg(test)]
mod tests {
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/sizing.rs`, replace:

```rust
//!
//! The mode, the free variable and the target torque are GUI state (A-2 decision A2-9): the
//! engine takes them as arguments. Design files and share links record them (decision M41-4).

#[cfg(test)]
mod tests {
```

with:

```rust
//!
//! The mode, the free variable and the target torque are GUI state (A-2 decision A2-9): the
//! engine takes them as arguments. Design files and share links record them (decision M41-4).

use crate::engine::sizing::FreeVariable;

/// Which way the calculator runs (spec A1 "Mode switch").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizingMode {
    /// The forward calculation: the inputs give the torque.
    MagnetsToTorque,
    /// Inverse sizing: the target torque gives the free variable.
    TorqueToMagnets,
}

impl SizingMode {
    /// Both modes; the first is the default.
    pub const ALL: [SizingMode; 2] = [SizingMode::MagnetsToTorque, SizingMode::TorqueToMagnets];

    /// The name a design file and a share link record.
    pub const fn key(self) -> &'static str {
        match self {
            SizingMode::MagnetsToTorque => "magnets_to_torque",
            SizingMode::TorqueToMagnets => "torque_to_magnets",
        }
    }

    /// The text of the mode switch: ASCII arrows, since egui's default fonts have no U+2192.
    pub const fn label(self) -> &'static str {
        match self {
            SizingMode::MagnetsToTorque => "Magnets -> Torque",
            SizingMode::TorqueToMagnets => "Torque -> Magnets",
        }
    }

    /// The mode a design file names, if it names one.
    pub fn from_key(key: &str) -> Option<SizingMode> {
        SizingMode::ALL.into_iter().find(|mode| mode.key() == key)
    }
}

/// The name a design file and a share link record for a free variable.
pub const fn variable_key(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => "axial_length",
        FreeVariable::MagnetsPerRing => "magnets_per_ring",
        FreeVariable::RingRadius => "ring_radius",
    }
}

/// The free variable a design file names, if it names one.
pub fn variable_from_key(key: &str) -> Option<FreeVariable> {
    FreeVariable::ALL
        .into_iter()
        .find(|&variable| variable_key(variable) == key)
}

/// The text of the free-variable picker.
pub const fn variable_label(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => "Axial magnet length",
        FreeVariable::MagnetsPerRing => "Magnets per ring",
        FreeVariable::RingRadius => "Ring radius",
    }
}

/// The input whose metadata gives the target torque its unit, range and step: the hot
/// minimum requirement, the torque inverse sizing is usually asked to meet.
pub const TARGET_RANGE_INPUT: &str = "metal.required_min_Nm";

/// The sizing state: the mode, the free variable and the target torque.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub struct SizingState {
    pub mode: SizingMode,
    pub variable: FreeVariable,
    /// The hot-low torque with production variation that inverse sizing must reach [N·m].
    pub target_Nm: f64,
}

/// The default target torque [N·m]: the workbook's hot minimum (`metal.required_min_Nm`,
/// Metal design C7), decision M41-9.
pub const DEFAULT_TARGET_NM: f64 = 2.5;

impl Default for SizingState {
    fn default() -> Self {
        Self {
            mode: SizingMode::MagnetsToTorque,
            variable: FreeVariable::ALL[0],
            target_Nm: DEFAULT_TARGET_NM,
        }
    }
}

#[cfg(test)]
mod tests {
```

- [ ] **Step 4: Run the tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED|panicked"
```

Expected: `test result: ok. 182 passed; 0 failed` (Task 0's 162 and the 20 new tests); no `FAILED`, no `panicked`. Then `git -C C:/Users/Cole/source/repos/lsim-mag-m41 diff --stat magcoupling-rs/Cargo.lock` shows only the `base64` entry and its two dependents' lines (`7 +` or so).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task1.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task1.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task1.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task1.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/Cargo.lock
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/Cargo.toml
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/format.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/session.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/sizing.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
feat(magcoupling-rs): design files, share links and the sizing state

The panel's session core (spec M4 "Session"). One JSON format for design files
and share links: every input by path and the sizing state (decisions M41-4,
M41-5), loaded all or nothing through InputSet::set (M41-6); a link is that JSON
deflated and URL-safe base64 in ?m=, as the linkage tool's; a refusal names
every problem of the inputs and the sizing state; renamed or removed paths of
older files go through PATH_MIGRATIONS. Non-finite numbers are "+inf", "-inf"
or "NaN" in JSON (M41-15), one text with the display (format::non_finite_text). The sizing state of Addendum A1:
mode, free variable, target torque (2.5 N·m to start, M41-9).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 2: Undo and redo history

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

Spec M4 "Session": "undo/redo of input changes". `History<T>` keeps snapshots of the design (Task 7 instantiates it with `Design`, so the sizing state is undone too). The panel shows it the state once per frame with `observe(current, settled)`; a change becomes one undo step only once the edit has settled (the panel decides: no pointer button down, no key held down, no text field of an input row focused: Task 7), so a drag or a part name typed letter by letter is one step, each arrow nudge its own, and a held arrow key's auto-repeat run one (decision M41-14). Undo during an unsettled edit reverts that edit; a new change drops the redo steps; 100 levels are kept.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/history.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs` (`pub mod history;`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (`magcoupling::gui::history`): `const MAX_UNDO_LEVELS: usize = 100`; `struct History<T>` with, for `T: Clone + PartialEq`, `fn new(initial: T) -> Self`, `fn with_levels(initial: T, max_levels: usize) -> Self`, `fn observe(&mut self, current: &T, settled: bool) -> bool`, `fn undo(&mut self, current: &T) -> Option<T>`, `fn redo(&mut self, current: &T) -> Option<T>`, `fn can_undo(&self, current: &T) -> bool`, `fn can_redo(&self, current: &T) -> bool`, `fn undo_len(&self) -> usize`.

- [ ] **Step 1: Write the failing tests**

The module docs and the tests (plain integers stand for the design): settled steps, coalescing an unsettled edit, no step for an unchanged state or a drag that ends where it began, undo during an edit, redo dropped by a new change, redo after an unrecorded change, the level cap.

Create `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/history.rs`:

```rust
//! Undo and redo of input changes (spec M4 "Session": "undo/redo of input changes").
//!
//! [`History`] keeps snapshots of the state (the panel's [`crate::gui::session::Design`]: the
//! inputs and the sizing state). The panel shows it the state once per frame with
//! [`History::observe`]; a change becomes one undo step when the edit has settled (no pointer
//! button down, no key held down, no text field of the design focused), so a drag, a typed
//! value or a part name typed letter by letter is one step, each arrow-key nudge (a key
//! pressed and released) is its own, and a held arrow key's whole auto-repeat run is one
//! (decision M41-14). Reset all and loading a design or a share link are edits like any
//! other, so they can be undone; a design the host opens as the session's start (a share link
//! at start-up) is not an edit: the panel starts a new history there.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_settled_change_is_one_step_and_undo_and_redo_walk_it() {
        let mut history = History::new(0);
        assert!(!history.can_undo(&0));
        assert!(history.observe(&1, true));
        assert!(history.observe(&2, true));
        assert_eq!(history.undo_len(), 2);
        assert_eq!(history.undo(&2), Some(1));
        assert_eq!(history.undo(&1), Some(0));
        assert_eq!(history.undo(&0), None);
        assert!(history.can_redo(&0));
        assert_eq!(history.redo(&0), Some(1));
        assert_eq!(history.redo(&1), Some(2));
        assert_eq!(history.redo(&2), None);
    }

    #[test]
    fn an_unsettled_edit_is_coalesced_into_one_step() {
        // A drag: many frames with the pointer down, then the release.
        let mut history = History::new(0);
        for value in 1..=5 {
            assert!(!history.observe(&value, false));
        }
        assert_eq!(history.undo_len(), 0);
        assert!(history.observe(&5, true));
        assert_eq!(history.undo_len(), 1);
        assert_eq!(history.undo(&5), Some(0));
    }

    #[test]
    fn an_unchanged_state_records_nothing() {
        let mut history = History::new(7);
        for _ in 0..3 {
            assert!(!history.observe(&7, true));
        }
        // A drag that ends where it started is no step either.
        assert!(!history.observe(&8, false));
        assert!(!history.observe(&7, true));
        assert_eq!(history.undo_len(), 0);
    }

    #[test]
    fn undo_during_an_unsettled_edit_reverts_that_edit() {
        let mut history = History::new(0);
        history.observe(&1, true);
        assert!(!history.observe(&9, false));
        assert!(history.can_undo(&9));
        assert_eq!(history.undo(&9), Some(1));
        assert_eq!(history.redo(&1), Some(9));
    }

    #[test]
    fn a_new_change_drops_the_redo_steps() {
        let mut history = History::new(0);
        history.observe(&1, true);
        assert_eq!(history.undo(&1), Some(0));
        assert!(history.observe(&5, true));
        assert!(!history.can_redo(&5));
        assert_eq!(history.redo(&5), None);
        assert_eq!(history.undo(&5), Some(0));
    }

    #[test]
    fn redo_after_an_unrecorded_change_records_it_and_gives_nothing() {
        let mut history = History::new(0);
        history.observe(&1, true);
        assert_eq!(history.undo(&1), Some(0));
        assert!(!history.can_redo(&3));
        assert_eq!(history.redo(&3), None);
        assert_eq!(history.undo(&3), Some(0));
    }

    #[test]
    fn the_oldest_step_is_dropped_past_the_limit() {
        let mut history = History::with_levels(0, 3);
        for value in 1..=5 {
            history.observe(&value, true);
        }
        assert_eq!(history.undo_len(), 3);
        assert_eq!(history.undo(&5), Some(4));
        assert_eq!(history.undo(&4), Some(3));
        assert_eq!(history.undo(&3), Some(2));
        assert_eq!(history.undo(&2), None);
        assert_eq!(History::with_levels(0, 0).max_levels, 1);
        assert_eq!(History::new(0).max_levels, MAX_UNDO_LEVELS);
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs`, replace:

```rust
//! without the feature.

mod format;
mod panel;
pub mod session;
pub mod sizing;
```

with:

```rust
//! without the feature.

mod format;
pub mod history;
mod panel;
pub mod session;
pub mod sizing;
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | sort -u | head -8
```

Expected: the build fails with ``error[E0433]: failed to resolve: use of undeclared type `History` `` (nine times) and ``error[E0425]: cannot find value `MAX_UNDO_LEVELS` in this scope``.

- [ ] **Step 3: Write the history**



In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/history.rs`, replace:

```rust
//! (decision M41-14). Reset all and loading a design or a share link are edits like any
//! other, so they can be undone; a design the host opens as the session's start (a share link
//! at start-up) is not an edit: the panel starts a new history there.

#[cfg(test)]
mod tests {
```

with:

```rust
//! (decision M41-14). Reset all and loading a design or a share link are edits like any
//! other, so they can be undone; a design the host opens as the session's start (a share link
//! at start-up) is not an edit: the panel starts a new history there.

use std::collections::VecDeque;

/// Undo levels kept; the oldest is dropped past this.
pub const MAX_UNDO_LEVELS: usize = 100;

/// Undo and redo stacks of state snapshots.
#[derive(Clone, Debug)]
pub struct History<T> {
    /// The state as of the last settled change: what the next undo step starts from.
    committed: T,
    /// Earlier settled states, oldest first.
    undo: VecDeque<T>,
    /// Undone states, the next redo last.
    redo: Vec<T>,
    max_levels: usize,
}

impl<T: Clone + PartialEq> History<T> {
    /// A history starting at `initial`, with nothing to undo, keeping [`MAX_UNDO_LEVELS`].
    pub fn new(initial: T) -> Self {
        Self::with_levels(initial, MAX_UNDO_LEVELS)
    }

    /// A history keeping at most `max_levels` undo steps (at least one).
    pub fn with_levels(initial: T, max_levels: usize) -> Self {
        Self {
            committed: initial,
            undo: VecDeque::new(),
            redo: Vec::new(),
            max_levels: max_levels.max(1),
        }
    }

    /// Records `current` as one undo step if it differs from the last settled state and the
    /// edit has `settled`. Returns whether it recorded a step.
    pub fn observe(&mut self, current: &T, settled: bool) -> bool {
        if settled && *current != self.committed {
            self.commit(current);
            true
        } else {
            false
        }
    }

    /// The state before the last change: an edit not yet settled counts as the last change.
    /// `None` when there is nothing to undo.
    pub fn undo(&mut self, current: &T) -> Option<T> {
        if *current != self.committed {
            self.commit(current);
        }
        let previous = self.undo.pop_back()?;
        let undone = std::mem::replace(&mut self.committed, previous.clone());
        self.redo.push(undone);
        Some(previous)
    }

    /// The state the last undo left. `None` when nothing was undone, or a change was made since
    /// (that change is recorded and the redo steps are dropped, as after any edit).
    pub fn redo(&mut self, current: &T) -> Option<T> {
        if *current != self.committed {
            self.commit(current);
            return None;
        }
        let next = self.redo.pop()?;
        let left = std::mem::replace(&mut self.committed, next.clone());
        self.undo.push_back(left);
        Some(next)
    }

    /// Whether [`History::undo`] would give a state for `current`.
    pub fn can_undo(&self, current: &T) -> bool {
        !self.undo.is_empty() || *current != self.committed
    }

    /// Whether [`History::redo`] would give a state for `current`.
    pub fn can_redo(&self, current: &T) -> bool {
        !self.redo.is_empty() && *current == self.committed
    }

    /// The number of undo steps recorded (an unsettled edit not included).
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    fn commit(&mut self, current: &T) {
        let previous = std::mem::replace(&mut self.committed, current.clone());
        self.undo.push_back(previous);
        while self.undo.len() > self.max_levels {
            self.undo.pop_front();
        }
        self.redo.clear();
    }
}

#[cfg(test)]
mod tests {
```

- [ ] **Step 4: Run the tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED|panicked"
```

Expected: `test result: ok. 189 passed; 0 failed` (Task 1's 182 and 7 new tests).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task2.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task2.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task2.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task2.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/history.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
feat(magcoupling-rs): undo and redo history of the design

History<T>: snapshots committed once an edit has settled, so a drag or a typed
name is one undo step and each arrow nudge its own (decision M41-14); undo
during an unsettled edit reverts it, a new change drops the redo steps, 100
levels.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 3: The input catalogue

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

Spec M4 "Layout": "inputs, generated from metadata, grouped as the package groups them (`coupling`, `metal`, `calibration`, `materials`, `temperature`, `clamps`). A Key design group on top: face gap, pole count, magnet part, axial length, operating temperature, back iron, cup wall, conductance, measured drag". `InputCatalogue` is built once (`OnceLock`) from `input_rows`: the Key design entries (`KEY_DESIGN`; the axial length is the A-2 override `coupling.magnets.axial_length_mm`, blank by default, which moves the torque at the default library part, where the manual lengths do not: the open item of `04-memory.yaml`), then every input in its group and section, each section under a heading (`SECTION_LABELS`); Key design inputs stay in their groups too (decision M41-13). The pure helpers the row widget of Task 4 needs: `step_decimals` (decision M41-1; a test lists the two defaults off their step grid, the vacuum permeability's), `outside_range` (M41-2), `OPTIONAL_SEEDS` (M41-12: entering an optional input starts it at the result it overrides, unrounded: the axial length override moves nothing; the measured drag switches the thermal summary to its measured branch), `text_hint` (library part or not, grade or not), `input_tooltip` (help, path, cell or "Rust-only input", slider range, default, assumption).

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/inputs.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs` (`pub mod inputs;`)

**Interfaces:**
- Consumes: `engine::meta::{FieldType, InputMeta, InputRow, SliderRange, Value, input_rows}`, `engine::library::lookup`, `engine::grades::grade`, `gui::format::{format_value, with_unit}`.
- Produces (`magcoupling::gui::inputs`): `const KEY_DESIGN: [&str; 10]`; `const SECTION_LABELS: [(&str, &str); 18]`; `const OPTIONAL_SEEDS: [(&str, &str); 2]`; `struct InputEntry { pub path: String, pub meta: &'static InputMeta, pub default: Value }`; `struct InputSection { pub prefix: String, pub label: &'static str, pub entries: Vec<InputEntry> }`; `struct InputGroup { pub name: String, pub label: &'static str, pub sections: Vec<InputSection> }`; `struct InputCatalogue { pub key_design: Vec<InputEntry>, pub groups: Vec<InputGroup> }` with `fn new() -> Self`, `fn get() -> &'static InputCatalogue`, `fn all(&self) -> impl Iterator<Item = &InputEntry>`, `fn entry(&self, path: &str) -> Option<&InputEntry>`; `fn section_label(&str) -> Option<&'static str>`; `fn step_decimals(step: f64) -> usize`; `fn outside_range(Option<SliderRange>, &Value) -> bool`; `fn optional_seed(&str) -> Option<&'static str>`; `fn text_hint(path: &str, text: &str) -> Option<&'static str>`; `fn input_tooltip(&InputEntry) -> String`.

- [ ] **Step 1: Write the failing tests**

The module docs and the tests: every input in exactly one section in schema order, every heading used, the Key design list with the blank axial length override, every field type the panel draws present (and no number without a slider range), step decimals for every step in the schema, every slider default on its step grid except the two listed in `OFF_GRID_DEFAULTS` (`coupling.mu0`, `calibration.mu0`), the out-of-range flag at its edges, the optional seeds (entering the axial length at its seed changes no result, bit for bit), the text hints, the tooltips.

Create `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/inputs.rs`:

```rust
//! The input list of the panel's left side (spec M4 "Layout": "inputs, generated from
//! metadata, grouped as the package groups them ... A Key design group on top").
//!
//! [`InputCatalogue`] is built once from the engine's input metadata: the Key design group
//! ([`KEY_DESIGN`]), then every input in its package group (`coupling`, `metal`,
//! `calibration`, `materials`, `temperature`, `clamps`), each group split into sections by
//! its nested input groups (`coupling.magnets`, `temperature.demag`, ...). Every input is in
//! exactly one section; the Key design inputs are also in their group (decision M41-13), and
//! both rows edit the same value.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::meta::{InputSet, ResultSet};

    #[test]
    fn every_input_is_in_exactly_one_section_in_schema_order() {
        let catalogue = InputCatalogue::new();
        let rows = input_rows(&DesignInputs::default());
        let listed: Vec<&str> = catalogue.all().map(|e| e.path.as_str()).collect();
        let schema: Vec<&str> = rows.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(listed, schema);
        let names: Vec<&str> = catalogue.groups.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "coupling",
                "metal",
                "calibration",
                "materials",
                "temperature",
                "clamps"
            ]
        );
        for group in &catalogue.groups {
            for section in &group.sections {
                assert!(
                    section.prefix.starts_with(&group.name),
                    "{}",
                    section.prefix
                );
                assert!(!section.entries.is_empty());
                for entry in &section.entries {
                    assert_eq!(entry.path.rsplit_once('.').unwrap().0, section.prefix);
                }
            }
        }
    }

    #[test]
    fn every_section_has_a_heading_and_every_heading_a_section() {
        let catalogue = InputCatalogue::new();
        let mut prefixes: Vec<&str> = catalogue
            .groups
            .iter()
            .flat_map(|g| {
                std::iter::once(g.name.as_str()).chain(g.sections.iter().map(|s| s.prefix.as_str()))
            })
            .collect();
        prefixes.dedup();
        for prefix in &prefixes {
            assert!(section_label(prefix).is_some(), "no heading for {prefix}");
        }
        for (prefix, _) in SECTION_LABELS {
            assert!(prefixes.contains(&prefix), "unused heading {prefix}");
        }
        assert_eq!(catalogue.groups[0].sections[1].label, "Magnets");
        assert_eq!(InputCatalogue::get(), &catalogue);
    }

    #[test]
    fn the_key_design_group_is_the_spec_list_with_the_axial_length_override() {
        let catalogue = InputCatalogue::new();
        let paths: Vec<&str> = catalogue
            .key_design
            .iter()
            .map(|e| e.path.as_str())
            .collect();
        assert_eq!(paths, KEY_DESIGN);
        let axial = catalogue.entry("coupling.magnets.axial_length_mm").unwrap();
        assert_eq!(axial.meta.ty, FieldType::OptF64);
        assert_eq!(axial.default, Value::None, "blank by default");
        // Every Key design entry is the same entry as in its group.
        for key in &catalogue.key_design {
            assert_eq!(Some(key), catalogue.entry(&key.path));
        }
    }

    #[test]
    fn the_inputs_cover_every_field_type_the_panel_draws() {
        let catalogue = InputCatalogue::new();
        let has = |pred: &dyn Fn(&InputEntry) -> bool| catalogue.all().any(pred);
        assert!(has(
            &|e| e.meta.ty == FieldType::F64 && e.meta.range.is_some()
        ));
        assert!(has(
            &|e| e.meta.ty == FieldType::F64 && e.meta.range.is_some_and(|r| r.log)
        ));
        assert!(has(
            &|e| e.meta.ty == FieldType::I64 && e.meta.choices.is_empty()
        ));
        assert!(has(
            &|e| e.meta.ty == FieldType::I64 && !e.meta.choices.is_empty()
        ));
        assert!(has(&|e| e.meta.ty == FieldType::OptF64));
        assert!(has(&|e| e.meta.ty == FieldType::Text));
        assert!(has(&|e| e.meta.rust_only));
        // No input is of a type the panel has no widget for.
        assert!(!has(&|e| e.meta.ty == FieldType::NumOrText));
        // Every number without choices has a slider range.
        assert!(!has(&|e| matches!(
            e.meta.ty,
            FieldType::F64 | FieldType::I64 | FieldType::OptF64
        ) && e.meta.choices.is_empty()
            && e.meta.range.is_none()));
    }

    #[test]
    fn step_decimals_write_each_step_exactly() {
        for (step, decimals) in [
            (2.0, 0),
            (1000.0, 0),
            (0.5, 1),
            (0.1, 1),
            (0.05, 2),
            (0.01, 2),
            (0.005, 3),
            (0.0001, 4),
            (1e-5, 5),
            (1e-7, 7),
            (1e-8, 8),
            (1e-11, 11),
            (1e-13, 13),
        ] {
            assert_eq!(step_decimals(step), decimals, "{step}");
        }
        for entry in InputCatalogue::new().all() {
            if let Some(r) = entry.meta.range {
                let d = step_decimals(r.step);
                assert!(d < 15, "{}: step {}", entry.path, r.step);
            }
        }
    }

    /// The inputs whose default is off its slider's step grid: the engine's step for the
    /// vacuum permeability (1e-11) is coarser than its default's last digit (1.256637e-6), so
    /// once nudged it never returns to the default (an open item of `04-memory.yaml`: the
    /// engine step should be 1e-12).
    const OFF_GRID_DEFAULTS: [&str; 2] = ["coupling.mu0", "calibration.mu0"];

    #[test]
    fn every_slider_default_is_on_its_step_grid_except_the_listed() {
        // Decision M41-1: a slider stores min + k * step rounded to the step's decimals, so
        // stepping back lands on a default exactly only if the default is such a value.
        let on_grid = |r: SliderRange, x: f64| {
            let k = (x - r.min) / r.step;
            let decimals = step_decimals(r.step);
            (k - k.round()).abs() < 1e-6 && format!("{x:.decimals$}").parse() == Ok(x)
        };
        let catalogue = InputCatalogue::new();
        let mut off_grid = Vec::new();
        for entry in catalogue.all() {
            let x = match entry.default {
                Value::Num(x) => x,
                Value::Int(i) => i as f64,
                _ => continue,
            };
            if let Some(r) = entry.meta.range
                && !on_grid(r, x)
            {
                off_grid.push(entry.path.as_str());
            }
        }
        assert_eq!(off_grid, OFF_GRID_DEFAULTS);
    }

    #[test]
    fn values_outside_the_slider_range_are_flagged() {
        let face_gap = InputCatalogue::new()
            .entry("metal.face_gap_mm")
            .unwrap()
            .meta
            .range;
        assert!(!outside_range(face_gap, &Value::Num(0.3)));
        assert!(!outside_range(face_gap, &Value::Num(5.0)));
        assert!(outside_range(face_gap, &Value::Num(5.0000001)));
        assert!(outside_range(face_gap, &Value::Num(0.29)));
        assert!(outside_range(face_gap, &Value::Int(7)));
        assert!(!outside_range(face_gap, &Value::None));
        assert!(!outside_range(None, &Value::Num(1e9)));
    }

    #[test]
    fn every_optional_input_starts_from_the_result_it_overrides() {
        let catalogue = InputCatalogue::new();
        let results = compute_all(&DesignInputs::default());
        let optional: Vec<&str> = catalogue
            .all()
            .filter(|e| e.meta.ty == FieldType::OptF64)
            .map(|e| e.path.as_str())
            .collect();
        let seeded: Vec<&str> = OPTIONAL_SEEDS.iter().map(|(input, _)| *input).collect();
        assert_eq!(optional, seeded);
        for (input, result) in OPTIONAL_SEEDS {
            assert_eq!(optional_seed(input), Some(result));
            let Some(Value::Num(seed)) = results.get(result) else {
                panic!("{result} is not a number")
            };
            let range = catalogue.entry(input).unwrap().meta.range.unwrap();
            assert!((range.min..=range.max).contains(&seed), "{input}: {seed}");
        }
    }

    #[test]
    fn entering_the_axial_length_at_its_seed_moves_nothing() {
        // Decision M41-12: the override starts at the inner ring's length (both default rings
        // are 12.7 mm B842SH), so the design is unchanged until the slider moves.
        let base = DesignInputs::default();
        let results = compute_all(&base);
        let Some(Value::Num(seed)) = results.get("model.inner_length_mm") else {
            panic!("a number")
        };
        let mut seeded = base.clone();
        seeded
            .set("coupling.magnets.axial_length_mm", Value::Num(seed))
            .unwrap();
        assert_eq!(compute_all(&seeded), results);
    }

    #[test]
    fn text_hints_say_what_the_engine_makes_of_the_text() {
        assert_eq!(
            text_hint("coupling.magnets.part_inner", "B842SH"),
            Some("Library part")
        );
        assert_eq!(
            text_hint("coupling.magnets.part_outer", "b842sh"),
            Some("Not a library part: the manual dimensions are used")
        );
        assert_eq!(
            text_hint("coupling.magnets.grade_inner", ""),
            Some("Blank: the manual Br, no rating")
        );
        assert_eq!(
            text_hint("coupling.magnets.grade_outer", "Y30"),
            Some("Grade table entry (used with manual dimensions)")
        );
        assert_eq!(
            text_hint("coupling.magnets.grade_outer", "N99"),
            Some("Not in the grade table: the manual Br, no rating")
        );
        assert_eq!(text_hint("metal.face_gap_mm", "1"), None);
        // Every text input has a hint.
        for entry in InputCatalogue::new().all() {
            if entry.meta.ty == FieldType::Text {
                assert!(text_hint(&entry.path, "").is_some(), "{}", entry.path);
            }
        }
    }

    #[test]
    fn tooltips_carry_help_path_cell_range_and_default() {
        let catalogue = InputCatalogue::new();
        assert_eq!(
            input_tooltip(catalogue.entry("metal.face_gap_mm").unwrap()),
            "Same as the measured prototype.\nmetal.face_gap_mm\nMetal design!C119\n\
             Slider 0.3 to 5 mm, step 0.01\nDefault: 1.400"
        );
        let backiron = input_tooltip(catalogue.entry("coupling.backiron").unwrap());
        assert!(
            backiron.ends_with("Default: 1 = steel circuit"),
            "{backiron}"
        );
        let drag = input_tooltip(catalogue.entry("metal.measured_drag_Nm").unwrap());
        assert!(
            drag.contains("logarithmic") && drag.ends_with("Default: blank"),
            "{drag}"
        );
        let harmonic = input_tooltip(catalogue.entry("coupling.max_harmonic").unwrap());
        assert!(
            harmonic.contains("Rust-only input (no workbook cell)")
                && harmonic.ends_with("Model assumption (Addendum A3)"),
            "{harmonic}"
        );
        let npole = input_tooltip(catalogue.entry("coupling.npole").unwrap());
        assert!(npole.contains("Slider 4 to 40, step 2"), "{npole}");
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs`, replace:

```rust

mod format;
pub mod history;
mod panel;
pub mod session;
pub mod sizing;
```

with:

```rust

mod format;
pub mod history;
pub mod inputs;
mod panel;
pub mod session;
pub mod sizing;
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | sort -u | head -8
```

Expected: the build fails with compile errors naming the items Step 3 adds, among them ``error[E0433]: failed to resolve: use of undeclared type `InputCatalogue` ``, ``error[E0425]: cannot find function `text_hint` in this scope``, ``error[E0425]: cannot find function `outside_range` in this scope`` and ``error[E0425]: cannot find value `SECTION_LABELS` in this scope``.

- [ ] **Step 3: Write the catalogue**



In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/inputs.rs`, replace:

```rust
//! its nested input groups (`coupling.magnets`, `temperature.demag`, ...). Every input is in
//! exactly one section; the Key design inputs are also in their group (decision M41-13), and
//! both rows edit the same value.

#[cfg(test)]
mod tests {
```

with:

```rust
//! its nested input groups (`coupling.magnets`, `temperature.demag`, ...). Every input is in
//! exactly one section; the Key design inputs are also in their group (decision M41-13), and
//! both rows edit the same value.

use std::sync::OnceLock;

use crate::DesignInputs;
use crate::engine::grades;
use crate::engine::library;
use crate::engine::meta::{FieldType, InputMeta, SliderRange, Value, input_rows};

/// The Key design group, in order (spec M4 "Layout": face gap, pole count, magnet part, axial
/// length, operating temperature, back iron, cup wall, conductance, measured drag). The axial
/// length is the A-2 override of both rings' length, blank by default (it moves the torque at
/// the default library part; the manual lengths do not); the sizing mode switch sits above
/// the group.
pub const KEY_DESIGN: [&str; 10] = [
    "metal.face_gap_mm",
    "coupling.npole",
    "coupling.magnets.part_inner",
    "coupling.magnets.part_outer",
    "coupling.magnets.axial_length_mm",
    "coupling.op_temp_C",
    "coupling.backiron",
    "metal.cup_wall_corner_mm",
    "temperature.thermal.conductance_W_K",
    "metal.measured_drag_Nm",
];

/// The heading of every input group and nested group, by path prefix, in the package's order.
pub const SECTION_LABELS: [(&str, &str); 18] = [
    ("coupling", "Coupling"),
    ("coupling.magnets", "Magnets"),
    ("metal", "Metal design"),
    ("calibration", "Calibration"),
    ("materials", "Materials"),
    ("materials.steel", "Back-iron steel"),
    ("materials.nickel", "Nickel plating"),
    ("materials.screws", "Screw classes"),
    ("materials.parts", "Part materials"),
    ("temperature", "Temperature design"),
    ("temperature.duty", "Duty"),
    ("temperature.demag", "Demagnetization"),
    ("temperature.adhesive", "Adhesive"),
    ("temperature.mismatch", "Thermal mismatch"),
    ("temperature.slip_loss", "Slip loss"),
    ("temperature.thermal", "Thermal network"),
    ("temperature.adhesive_life", "Adhesive life"),
    ("clamps", "Shaft clamps"),
];

/// Where an optional input starts when the user enters a value (decision M41-12): the result
/// it overrides, unrounded. The axial length override enters at the inner ring's length in
/// use (both rings take it), so nothing moves. The measured drag enters at the model's
/// equivalent mean drag torque; entering any measured drag switches the thermal summary from
/// the not-measured high estimate to the measured branch, so the slip loss and the steady
/// temperatures move (the drag torque itself is the model's).
pub const OPTIONAL_SEEDS: [(&str, &str); 2] = [
    ("coupling.magnets.axial_length_mm", "model.inner_length_mm"),
    ("metal.measured_drag_Nm", "temperature.slip_loss.drag_Nm"),
];

/// One input: its path, metadata and default value.
#[derive(Clone, Debug, PartialEq)]
pub struct InputEntry {
    pub path: String,
    pub meta: &'static InputMeta,
    pub default: Value,
}

/// A run of inputs under one heading: a group's own fields, or one of its nested groups.
#[derive(Clone, Debug, PartialEq)]
pub struct InputSection {
    /// The path prefix (`coupling`, `coupling.magnets`).
    pub prefix: String,
    pub label: &'static str,
    pub entries: Vec<InputEntry>,
}

/// A package group (`coupling`, ...) and its sections, in schema order.
#[derive(Clone, Debug, PartialEq)]
pub struct InputGroup {
    pub name: String,
    pub label: &'static str,
    pub sections: Vec<InputSection>,
}

/// Every input, arranged for the left side of the panel.
#[derive(Clone, Debug, PartialEq)]
pub struct InputCatalogue {
    /// [`KEY_DESIGN`], in order.
    pub key_design: Vec<InputEntry>,
    /// Every input, by package group and section, in schema order.
    pub groups: Vec<InputGroup>,
}

/// The heading of a path prefix; `None` if [`SECTION_LABELS`] lacks it (a test checks none
/// does).
pub fn section_label(prefix: &str) -> Option<&'static str> {
    SECTION_LABELS
        .iter()
        .find(|(p, _)| *p == prefix)
        .map(|(_, label)| *label)
}

impl InputCatalogue {
    /// The catalogue of the engine's inputs, with the defaults of [`DesignInputs::default`].
    pub fn new() -> Self {
        let rows = input_rows(&DesignInputs::default());
        let entry = |row: &crate::engine::meta::InputRow| InputEntry {
            path: row.path.clone(),
            meta: row.meta,
            default: row.value.clone(),
        };
        let key_design = KEY_DESIGN
            .iter()
            .map(|&path| {
                let row = rows.iter().find(|row| row.path == path);
                entry(row.unwrap_or_else(|| panic!("KEY_DESIGN: no input {path}")))
            })
            .collect();
        let mut groups: Vec<InputGroup> = Vec::new();
        for row in &rows {
            let (prefix, _) = row
                .path
                .rsplit_once('.')
                .expect("every input sits in a group");
            let name = prefix.split('.').next().unwrap_or(prefix);
            if groups.last().is_none_or(|g| g.name != name) {
                groups.push(InputGroup {
                    name: name.to_owned(),
                    label: section_label(name).unwrap_or("Inputs"),
                    sections: Vec::new(),
                });
            }
            let group = groups.last_mut().expect("pushed above");
            if group.sections.last().is_none_or(|s| s.prefix != prefix) {
                group.sections.push(InputSection {
                    prefix: prefix.to_owned(),
                    label: section_label(prefix).unwrap_or("Inputs"),
                    entries: Vec::new(),
                });
            }
            let section = group.sections.last_mut().expect("pushed above");
            section.entries.push(entry(row));
        }
        Self { key_design, groups }
    }

    /// The catalogue, built once: it depends only on the engine's metadata.
    pub fn get() -> &'static InputCatalogue {
        static CATALOGUE: OnceLock<InputCatalogue> = OnceLock::new();
        CATALOGUE.get_or_init(InputCatalogue::new)
    }

    /// Every entry of the package groups, in schema order.
    pub fn all(&self) -> impl Iterator<Item = &InputEntry> {
        self.groups
            .iter()
            .flat_map(|g| g.sections.iter())
            .flat_map(|s| s.entries.iter())
    }

    /// The entry of an input path.
    pub fn entry(&self, path: &str) -> Option<&InputEntry> {
        self.all().find(|entry| entry.path == path)
    }
}

impl Default for InputCatalogue {
    fn default() -> Self {
        Self::new()
    }
}

/// Decimal places of a slider step: the fewest that write the step exactly (0.01 → 2,
/// 0.005 → 3, 2 → 0, 1e-13 → 13). Slider values are rounded to them (decision M41-1), so a
/// value the slider sets is the decimal the user reads (1.41, not 1.4100000000000001), and
/// stepping back to a default lands on it exactly, for every default on its step grid (all
/// but the vacuum permeability's two, a test lists them).
pub fn step_decimals(step: f64) -> usize {
    (0..=15)
        .find(|&decimals| {
            let scaled = step * 10f64.powi(decimals as i32);
            scaled.round() >= 1.0 && (scaled - scaled.round()).abs() <= 1e-9 * scaled
        })
        .unwrap_or(15)
}

/// Whether a number lies outside its slider range (a value from a design file, a share link
/// or the struct: the slider keeps it until edited, decision M41-2, and the row flags it).
pub fn outside_range(range: Option<SliderRange>, value: &Value) -> bool {
    let x = match value {
        Value::Num(x) => *x,
        Value::Int(i) => *i as f64,
        _ => return false,
    };
    range.is_some_and(|r| !(r.min..=r.max).contains(&x))
}

/// The result an optional input starts from ([`OPTIONAL_SEEDS`]).
pub fn optional_seed(path: &str) -> Option<&'static str> {
    OPTIONAL_SEEDS
        .iter()
        .find(|(input, _)| *input == path)
        .map(|(_, result)| *result)
}

/// A short note under a text input saying what the engine makes of the text: whether a part
/// name is a library part, whether a grade name is in the grade table.
pub fn text_hint(path: &str, text: &str) -> Option<&'static str> {
    match path {
        "coupling.magnets.part_inner" | "coupling.magnets.part_outer" => {
            Some(if library::lookup(text).is_some() {
                "Library part"
            } else {
                "Not a library part: the manual dimensions are used"
            })
        }
        "coupling.magnets.grade_inner" | "coupling.magnets.grade_outer" => {
            Some(if text.is_empty() {
                "Blank: the manual Br, no rating"
            } else if grades::grade(text).is_some() {
                "Grade table entry (used with manual dimensions)"
            } else {
                "Not in the grade table: the manual Br, no rating"
            })
        }
        _ => None,
    }
}

/// The hover text of an input: help, path, workbook cell, slider range, default, and whether
/// it is a model assumption.
pub fn input_tooltip(entry: &InputEntry) -> String {
    let meta = entry.meta;
    let mut lines = Vec::new();
    if !meta.help.is_empty() {
        lines.push(meta.help.to_owned());
    }
    lines.push(entry.path.clone());
    lines.push(meta.cell.map_or_else(
        || "Rust-only input (no workbook cell)".to_owned(),
        str::to_owned,
    ));
    if let Some(r) = meta.range {
        let unit = crate::gui::format::with_unit(String::new(), meta.unit);
        let scale = if r.log { ", logarithmic" } else { "" };
        lines.push(format!(
            "Slider {} to {}{unit}, step {}{scale}",
            r.min, r.max, r.step
        ));
    }
    let default = match (&entry.default, meta.ty) {
        (Value::None, FieldType::OptF64) => "blank".to_owned(),
        (Value::Text(text), _) if text.is_empty() => "blank".to_owned(),
        (Value::Int(code), _) if !meta.choices.is_empty() => {
            let choice = meta.choices.iter().find(|(c, _)| c == code);
            choice.map_or_else(|| code.to_string(), |(c, text)| format!("{c} = {text}"))
        }
        (value, _) => crate::gui::format::format_value(value),
    };
    lines.push(format!("Default: {default}"));
    if meta.assumption {
        lines.push("Model assumption (Addendum A3)".to_owned());
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
```

- [ ] **Step 4: Run the tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED|panicked"
```

Expected: `test result: ok. 200 passed; 0 failed` (Task 2's 189 and 11 new tests).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task3.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task3.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task3.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task3.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/inputs.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
feat(magcoupling-rs): the input catalogue of the panel's left side

Every input from the metadata, built once: the Key design group (face gap,
poles, both parts, the A-2 axial length override, temperature, back iron, cup
wall, conductance, measured drag), then each package group by section, Key
design inputs staying in their groups (decision M41-13). The helpers the rows
use: step decimals (M41-1, every default on its grid but mu0's two), range
checks (M41-2), optional seeds (M41-12), text hints and tooltips.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 4: Every input as a row, in the three-region layout

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

Spec M4 "Sliders": "live recompute while dragging; value box for typed entry; arrow-key nudges; step snapping (e.g. even-only pole counts); logarithmic scale for wide ranges; per-field reset-to-default; a dot when changed from default; tooltip with help text and workbook cell; selectors as drop-downs". `gui/input_ui.rs` draws one row of any field type from its `InputEntry` and returns the edit asked for; the panel applies it with `InputSet::set`. A number is an egui slider with its value box, `SliderClamping::Edits` (edits clamped, typed values included; a value already outside is kept and flagged: decision M41-2), values rounded to the step's decimals (`max_decimals`, M41-1), typed values applied on Enter; an integer count steps by its step (poles by two); a selector is a drop-down; an optional input a checkbox and a slider, entered at its seed, clamped into the range but not rounded (M41-12); text a text field with its hint. The panel takes the spec's layout: a header (heading, "Reset all", the last refusal), the inputs in a resizable left side (the Key design group open, every package group collapsed), the headline numbers on the right (Task 5 makes them the dashboard), the centre for Task 6. The tracer's `KEY_INPUTS` goes (`gui::KEY_DESIGN` replaces it); its tests are rewritten: the axial length now edits the override, not the manual inner length. The app drops its outer scroll area: the sides scroll on their own. The panel tests tap keys, Ctrl+A included (`key_tap`, `select_all`: pressed and released in one frame): egui keeps a pressed key down until its release and reads later presses as repeats, and Task 7 treats a key held down as an edit in progress.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/input_ui.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs` (the layout, the rows, the tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/test_support.rs` (`select_all`, tapped; `key_press` becomes `key_event` and `key_tap`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs` (`pub mod input_ui;`, `pub use inputs::KEY_DESIGN;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs` (the panel without an outer scroll area)

**Interfaces:**
- Consumes: Task 3's `InputCatalogue::get`, `InputEntry`, `input_tooltip`, `outside_range`, `step_decimals`, `text_hint`, `optional_seed`; `gui::format::{format_value, with_unit}`.
- Produces (`magcoupling::gui::input_ui`): `const CHANGED_DOT: &str = "\u{2022}"`, `RESET_LABEL = "reset"`, `OUTSIDE_RANGE_NOTE = "outside the slider range"`, `BLANK_TEXT = "blank"`; `enum RowEdit { Set(Value), Reset }`; `struct RowOutput { pub edit: Option<RowEdit>, pub widget: egui::Response }`; `fn slider<'a>(value: &'a mut f64, meta: &InputMeta, range: SliderRange) -> egui::Slider<'a>`; `fn input_row(ui: &mut egui::Ui, entry: &InputEntry, current: &Value, seed: Option<f64>) -> RowOutput`.
- Produces (`gui::panel`): `const HEADING`, `RESET_ALL`, `KEY_DESIGN_HEADING`; the panel's private `key_widgets: Vec<(&'static str, egui::Id)>` (tests focus the Key design rows by path), `input_row_ui`, `seed`, `inputs_ui`, `header_ui`. `gui::KEY_INPUTS` is removed.
- Produces (`gui::test_support`, tests only): `fn select_all() -> Vec<egui::Event>` (Ctrl+A tapped); `fn key_event(key, pressed: bool) -> egui::Event` and `fn key_tap(key) -> Vec<egui::Event>` (press and release), replacing `key_press`.

- [ ] **Step 1: Write the failing tests**

The panel's test module is rewritten for the new layout (the harness finds a Key design row's widget by path) and the test helpers grow Ctrl+A and key taps. The tests pin: every headline number and Key design input drawn; idle frames change nothing, with every group open (bottom up, so no click lands on a row an opening group moved) and values off their step grid (a face gap of 1.4123, a measured drag of 0.012345, the vacuum permeability's defaults); a value outside the range kept until edited and flagged, the first edit clamping it; an arrow key giving exactly 1.41 and the headline of that design in the same frame; seven steps up and down landing on the default exactly (M41-1); a typed value snapped to the step and clamped, text that is no number ignored (Review Focus 2); the rail click; poles stepping by two; range ends as hard stops; the changed dot and per-field reset; the axial length override starting blank, entering at 12.7 mm with no result moving, then moving the torque; the measured drag entering at the model's drag, unrounded (the drag in use stays, the steady high case goes to the measured branch, 74.17 °C); a selector switching the branch (no back iron); a part name typed and its hint following; every group opening onto all its inputs; reset all; the panel inside an `egui::Window` (M5).

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::test_support::{
        central_panel_frame, drawn_texts, key_press, primary_button, text_rect,
    };

    const FACE_GAP: usize = 0;
    const POLES: usize = 1;
    const AXIAL_LENGTH: usize = 2;

    /// A panel and the egui context it is drawn in, frame by frame.
    struct Harness {
        ctx: egui::Context,
        panel: MagcouplingPanel,
    }

    impl Harness {
        fn new() -> Self {
            let mut harness = Self {
                ctx: egui::Context::default(),
                panel: MagcouplingPanel::new(),
```

with:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::input_ui::{CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::test_support::{
        central_panel_frame, drawn_texts, key_tap, primary_button, select_all, text_rect,
    };

    const FACE_GAP: &str = "metal.face_gap_mm";
    const POLES: &str = "coupling.npole";
    const AXIAL_LENGTH: &str = "coupling.magnets.axial_length_mm";
    const PART_INNER: &str = "coupling.magnets.part_inner";
    const MEASURED_DRAG: &str = "metal.measured_drag_Nm";

    /// A panel and the egui context it is drawn in, frame by frame.
    pub(crate) struct Harness {
        pub(crate) ctx: egui::Context,
        pub(crate) panel: MagcouplingPanel,
    }

    impl Harness {
        pub(crate) fn new() -> Self {
            let mut harness = Self {
                ctx: egui::Context::default(),
                panel: MagcouplingPanel::new(),
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
            harness
        }

        fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
            let panel = &mut self.panel;
            central_panel_frame(&self.ctx, events, |ui| panel.ui(ui))
        }

        /// The key input's slider (its rail) as drawn in the last frame.
        fn slider(&self, key: usize) -> egui::Response {
            let id = self.panel.slider_ids[key].expect("the slider was drawn");
            self.ctx
                .read_response(id)
                .expect("the slider has a response")
        }

        /// Gives the key input's slider keyboard focus.
        fn focus(&mut self, key: usize) {
            let id = self.slider(key).id;
            self.ctx.memory_mut(|m| m.request_focus(id));
            self.frame(Vec::new());
            assert!(
                self.ctx.memory(|m| m.has_focus(id)),
                "slider {key} has focus"
            );
        }

        /// A click (move, press, release) at `at`.
```

with:

```rust
            harness
        }

        pub(crate) fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
            let panel = &mut self.panel;
            central_panel_frame(&self.ctx, events, |ui| panel.ui(ui))
        }

        /// The Key design row's main widget as drawn in the last frame.
        fn widget(&self, path: &str) -> egui::Response {
            let (_, id) = self
                .panel
                .key_widgets
                .iter()
                .find(|(p, _)| *p == path)
                .copied()
                .unwrap_or_else(|| panic!("no Key design row {path}"));
            self.ctx
                .read_response(id)
                .expect("the widget has a response")
        }

        /// Gives the Key design row's widget keyboard focus.
        fn focus(&mut self, path: &str) {
            let id = self.widget(path).id;
            self.ctx.memory_mut(|m| m.request_focus(id));
            self.frame(Vec::new());
            assert!(self.ctx.memory(|m| m.has_focus(id)), "{path} has focus");
        }

        /// A click (move, press, release) at `at`.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
            self.frame(vec![primary_button(at, false)])
        }

        fn number(&self, key: usize) -> f64 {
            match self.panel.inputs.get(KEY_INPUTS[key]) {
                Some(Value::Num(x)) => x,
                Some(Value::Int(i)) => i as f64,
                other => panic!("{}: {other:?}", KEY_INPUTS[key]),
            }
        }
    }
```

with:

```rust
            self.frame(vec![primary_button(at, false)])
        }

        /// Clicks the first drawn text equal to `text`.
        fn click_text(&mut self, text: &str) -> egui::FullOutput {
            let output = self.frame(Vec::new());
            let rect = text_rect(&output, text).unwrap_or_else(|| panic!("no text {text:?}"));
            self.click(rect.center())
        }

        fn number(&self, path: &str) -> f64 {
            match self.panel.inputs.get(path) {
                Some(Value::Num(x)) => x,
                Some(Value::Int(i)) => i as f64,
                other => panic!("{path}: {other:?}"),
            }
        }
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        displayed_headline(inputs).remove(0)
    }

    #[test]
    fn key_inputs_are_numeric_inputs_with_slider_ranges() {
        let panel = MagcouplingPanel::new();
        for (path, meta) in panel.key_inputs {
            assert!(
                matches!(meta.ty, FieldType::F64 | FieldType::I64),
                "{path}: {:?}",
                meta.ty
            );
            let range = meta.range.unwrap_or_else(|| panic!("{path}: no range"));
            assert!(
                range.min < range.max && range.step > 0.0,
                "{path}: {range:?}"
            );
            assert!(meta.choices.is_empty(), "{path} is a selector");
        }
        assert_eq!(panel.key_inputs.map(|(path, _)| path), KEY_INPUTS);
    }

    #[test]
```

with:

```rust
        displayed_headline(inputs).remove(0)
    }

    fn count(output: &egui::FullOutput, text: &str) -> usize {
        drawn_texts(output).iter().filter(|t| *t == text).count()
    }

    #[test]
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    }

    #[test]
    fn every_headline_number_is_drawn_with_its_label() {
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_drew_headline(&output, &DesignInputs::default());
```

with:

```rust
    }

    #[test]
    fn every_headline_number_and_key_design_input_is_drawn_with_its_label() {
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_drew_headline(&output, &DesignInputs::default());
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        for meta in harness.panel.headline_meta {
            assert!(
                texts.iter().any(|t| t == meta.label),
                "missing label {:?}",
                meta.label
            );
        }
        for (_, meta) in harness.panel.key_inputs {
            assert!(
                texts.iter().any(|t| t == meta.label),
                "missing slider {:?}",
                meta.label
            );
        }
        // The corrected headline (E2: M4 x 14), not the workbook's (M4 x 12).
        assert!(texts.iter().any(|t| t.contains("M4 x 14")), "{texts:?}");
    }

    #[test]
    fn idle_frames_change_no_input() {
        let mut harness = Harness::new();
        for _ in 0..5 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.last_error, None);
    }

    #[test]
    fn an_input_outside_its_slider_range_is_kept_until_edited() {
        let mut harness = Harness::new();
        harness.panel.inputs.metal.face_gap_mm = 7.5; // range 0.3 to 5.0
        for _ in 0..3 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs.metal.face_gap_mm, 7.5);
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 7.5;
        assert_eq!(harness.panel.results(), &compute_all(&inputs));
    }

    #[test]
    fn an_arrow_key_on_the_face_gap_slider_updates_the_headline() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        let output = harness.frame(vec![key_press(egui::Key::ArrowRight)]);

        // One step (0.01 mm) up from 1.4 mm, snapped from the range start.
        assert!(
            (harness.number(FACE_GAP) - 1.41).abs() < 1e-12,
            "{}",
            harness.number(FACE_GAP)
        );
        let mut expected = DesignInputs::default();
        expected.metal.face_gap_mm = harness.panel.inputs.metal.face_gap_mm;
        assert_eq!(
            harness.panel.inputs(),
            &expected,
```

with:

```rust
        for meta in harness.panel.headline_meta {
            assert!(
                texts.iter().any(|t| t == meta.label),
                "missing {:?}",
                meta.label
            );
        }
        for entry in &InputCatalogue::get().key_design {
            assert!(
                texts.iter().any(|t| t == entry.meta.label),
                "missing input {:?}",
                entry.meta.label
            );
        }
        // The corrected headline (E2: M4 x 14), not the workbook's (M4 x 12).
        assert!(texts.iter().any(|t| t.contains("M4 x 14")), "{texts:?}");
        // Every Key design row drew its widget.
        assert_eq!(harness.panel.key_widgets.len(), 10);
    }

    #[test]
    fn idle_frames_change_no_input() {
        // Decision M41-2 (`SliderClamping::Edits`): a slider writes only on an edit, so idle
        // frames keep every value as it is, values off their step grid included: a face gap
        // and a measured drag from a file, and the vacuum permeability's two defaults (every
        // group open, so every row is drawn).
        let mut harness = Harness::new();
        harness.panel.inputs.metal.face_gap_mm = 1.4123;
        harness.panel.inputs.metal.measured_drag_Nm = Some(0.012345);
        let design = harness.panel.inputs.clone();
        // Bottom up: opening a group moves only the groups below it, so no click lands on a
        // row that an opening group above has just moved there.
        for group in InputCatalogue::get().groups.iter().rev() {
            harness.click_text(group.label);
        }
        let mut output = harness.frame(Vec::new());
        for _ in 0..15 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(
            count(&output, "Vacuum permeability"),
            2,
            "both mu0 rows drawn"
        );
        assert_eq!(harness.panel.inputs(), &design);
        assert_eq!(harness.panel.last_error, None);
    }

    #[test]
    fn an_input_outside_its_slider_range_is_kept_until_edited_and_flagged() {
        let mut harness = Harness::new();
        harness.panel.inputs.metal.face_gap_mm = 7.5; // range 0.3 to 5.0
        let mut output = harness.frame(Vec::new());
        for _ in 0..2 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs.metal.face_gap_mm, 7.5);
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 7.5;
        assert_eq!(harness.panel.results(), &compute_all(&inputs));
        assert_eq!(count(&output, OUTSIDE_RANGE_NOTE), 1);
        // Decision M41-2: an edit brings it back into the range.
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowLeft));
        assert_eq!(harness.number(FACE_GAP), 5.0);
        assert_eq!(count(&harness.frame(Vec::new()), OUTSIDE_RANGE_NOTE), 0);
    }

    #[test]
    fn an_arrow_key_on_the_face_gap_slider_updates_the_headline() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        let output = harness.frame(key_tap(egui::Key::ArrowRight));

        // One step (0.01 mm) up from 1.4 mm, rounded to the step's decimals (decision M41-1).
        assert_eq!(harness.number(FACE_GAP), 1.41);
        let mut expected = DesignInputs::default();
        expected.metal.face_gap_mm = 1.41;
        assert_eq!(
            harness.panel.inputs(),
            &expected,
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    }

    #[test]
    fn clicking_the_end_of_the_face_gap_rail_sets_its_maximum() {
        let mut harness = Harness::new();
        let rail = harness.slider(FACE_GAP).rect;
        let output = harness.click(rail.right_center() - egui::vec2(1.0, 0.0));
        assert_eq!(harness.number(FACE_GAP), 5.0);
        assert_drew_headline(&output, harness.panel.inputs());
```

with:

```rust
    }

    #[test]
    fn stepping_back_to_the_default_lands_on_it_exactly() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        for _ in 0..7 {
            harness.frame(key_tap(egui::Key::ArrowRight));
        }
        assert_eq!(harness.number(FACE_GAP), 1.47);
        for _ in 0..7 {
            harness.frame(key_tap(egui::Key::ArrowLeft));
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(count(&harness.frame(Vec::new()), CHANGED_DOT), 0);
    }

    #[test]
    fn clicking_the_end_of_the_face_gap_rail_sets_its_maximum() {
        let mut harness = Harness::new();
        let rail = harness.widget(FACE_GAP).rect;
        let output = harness.click(rail.right_center() - egui::vec2(1.0, 0.0));
        assert_eq!(harness.number(FACE_GAP), 5.0);
        assert_drew_headline(&output, harness.panel.inputs());
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    }

    #[test]
    fn the_pole_count_steps_by_two_and_stays_even() {
        let mut harness = Harness::new();
        harness.focus(POLES);
        harness.frame(vec![key_press(egui::Key::ArrowRight)]);
        assert_eq!(harness.panel.inputs.coupling.npole, 12);
        harness.frame(vec![key_press(egui::Key::ArrowLeft)]);
        harness.frame(vec![key_press(egui::Key::ArrowLeft)]);
        assert_eq!(harness.panel.inputs.coupling.npole, 8);
        let rail = harness.slider(POLES).rect;
        harness.click(rail.center());
        let npole = harness.panel.inputs.coupling.npole;
        assert!(npole % 2 == 0 && (4..=40).contains(&npole), "{npole}");
```

with:

```rust
    }

    #[test]
    fn a_typed_value_snaps_to_the_step_and_is_clamped_into_the_slider_range() {
        let mut harness = Harness::new();
        // The value box beside the slider shows the value with its unit; a click edits it,
        // and the typed text applies on Enter (decision M41-2).
        let type_in = |harness: &mut Harness, shown: &str, text: &str| {
            harness.click_text(shown);
            harness.frame([select_all(), vec![egui::Event::Text(text.to_owned())]].concat());
            harness.frame(key_tap(egui::Key::Enter));
        };
        type_in(&mut harness, "1.40 mm", "2.344");
        assert_eq!(harness.number(FACE_GAP), 2.34);
        type_in(&mut harness, "2.34 mm", "9");
        assert_eq!(harness.number(FACE_GAP), 5.0);
        type_in(&mut harness, "5.00 mm", "-1");
        assert_eq!(harness.number(FACE_GAP), 0.3);
        // Text that is no number changes nothing (Review Focus 2).
        type_in(&mut harness, "0.30 mm", "wide");
        assert_eq!(harness.number(FACE_GAP), 0.3);
        assert_eq!(harness.panel.last_error, None);
    }

    #[test]
    fn the_pole_count_steps_by_two_and_stays_even() {
        let mut harness = Harness::new();
        harness.focus(POLES);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(harness.panel.inputs.coupling.npole, 12);
        harness.frame(key_tap(egui::Key::ArrowLeft));
        harness.frame(key_tap(egui::Key::ArrowLeft));
        assert_eq!(harness.panel.inputs.coupling.npole, 8);
        let rail = harness.widget(POLES).rect;
        harness.click(rail.center());
        let npole = harness.panel.inputs.coupling.npole;
        assert!(npole % 2 == 0 && (4..=40).contains(&npole), "{npole}");
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        harness.panel.inputs.metal.face_gap_mm = 0.3;
        harness.frame(Vec::new());
        harness.focus(POLES);
        harness.frame(vec![key_press(egui::Key::ArrowRight)]);
        assert_eq!(harness.panel.inputs.coupling.npole, 40);
        harness.focus(FACE_GAP);
        harness.frame(vec![key_press(egui::Key::ArrowLeft)]);
        assert_eq!(harness.panel.inputs.metal.face_gap_mm, 0.3);
    }

    #[test]
    fn the_axial_length_slider_edits_the_manual_inner_length() {
        let mut harness = Harness::new();
        harness.focus(AXIAL_LENGTH);
        harness.frame(vec![key_press(egui::Key::ArrowRight)]);
        let length = harness.panel.inputs.coupling.magnets.manual_inner_length_mm;
        assert!((length - 12.71).abs() < 1e-12, "{length}");
        // A library part (the default) ignores the manual length.
        assert_eq!(
            harness.panel.results(),
            &compute_all(&DesignInputs::default())
        );
    }

    #[test]
    fn reset_all_restores_the_default_design_and_headline() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(vec![key_press(egui::Key::ArrowRight)]);
        harness.focus(POLES);
        harness.frame(vec![key_press(egui::Key::ArrowRight)]);
        assert_ne!(harness.panel.inputs(), &DesignInputs::default());

        let output = harness.frame(Vec::new());
        let button = text_rect(&output, "Reset all").expect("the reset button is drawn");
        harness.click(button.center());
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(
            harness.panel.results(),
```

with:

```rust
        harness.panel.inputs.metal.face_gap_mm = 0.3;
        harness.frame(Vec::new());
        harness.focus(POLES);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(harness.panel.inputs.coupling.npole, 40);
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowLeft));
        assert_eq!(harness.panel.inputs.metal.face_gap_mm, 0.3);
    }

    #[test]
    fn a_changed_input_shows_the_dot_and_its_reset_restores_the_default() {
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CHANGED_DOT), 0);
        assert_eq!(count(&output, RESET_LABEL), 0);
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CHANGED_DOT), 1);
        assert_eq!(count(&output, RESET_LABEL), 1);
        harness.click_text(RESET_LABEL);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(count(&harness.frame(Vec::new()), CHANGED_DOT), 0);
    }

    #[test]
    fn the_axial_length_override_starts_blank_and_enters_at_the_ring_length() {
        let mut harness = Harness::new();
        assert_eq!(harness.panel.inputs.coupling.magnets.axial_length_mm, None);
        // The checkbox enters a value: the inner ring's length in use, so nothing moves.
        harness.focus(AXIAL_LENGTH);
        harness.frame(key_tap(egui::Key::Space));
        assert_eq!(
            harness.panel.inputs.coupling.magnets.axial_length_mm,
            Some(12.7)
        );
        assert_eq!(
            harness.panel.results(),
            &compute_all(&DesignInputs::default())
        );
        // Then the slider moves both rings' length, and the torque with it (at the default
        // library part, which the manual lengths never move).
        harness.frame(Vec::new());
        harness.focus(AXIAL_LENGTH);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(
            harness.panel.inputs.coupling.magnets.axial_length_mm,
            Some(12.71)
        );
        assert_ne!(
            displayed_pullout(harness.panel.inputs()),
            displayed_pullout(&DesignInputs::default())
        );
        // Reset leaves it blank again.
        harness.click_text(RESET_LABEL);
        assert_eq!(harness.panel.inputs.coupling.magnets.axial_length_mm, None);
    }

    #[test]
    fn entering_the_measured_drag_starts_at_the_model_s_drag() {
        // Decision M41-12: the measured drag enters at the model's equivalent mean drag torque,
        // unrounded (its step's 0.0131 would move the slip loss again), so the drag in use stays
        // put; the thermal summary leaves the not-measured high estimate for the measured branch.
        let num = |results: &DesignResults, path: &str| match results.get(path) {
            Some(Value::Num(x)) => x,
            other => panic!("{path}: {other:?}"),
        };
        let default = compute_all(&DesignInputs::default());
        let model_drag = num(&default, "temperature.slip_loss.drag_Nm");
        assert_eq!(
            default.get("metal.slip_loss_W"),
            Some(Value::Text("not measured".to_owned()))
        );
        let mut harness = Harness::new();
        harness.focus(MEASURED_DRAG);
        harness.frame(key_tap(egui::Key::Space));
        assert_eq!(
            harness.panel.inputs.metal.measured_drag_Nm,
            Some(model_drag),
            "the seed, not rounded to the slider's step"
        );
        let results = harness.panel.results();
        assert_eq!(num(results, "temperature.slip_loss.drag_Nm"), model_drag);
        let estimate = num(results, "temperature.summary.steady_estimate_C");
        let high = num(results, "temperature.summary.steady_high_C");
        assert_eq!(
            estimate,
            num(&default, "temperature.summary.steady_estimate_C")
        );
        assert_eq!(high, estimate, "measured: the high case is the estimate");
        assert_eq!(format_value(&Value::Num(high)), "74.17");
        let not_measured = num(&default, "temperature.summary.steady_high_C");
        assert_eq!(format_value(&Value::Num(not_measured)), "92.51");
        let loss = num(results, "metal.slip_loss_W");
        assert_eq!(format_value(&Value::Num(loss)), "2.751");
    }

    #[test]
    fn a_selector_switches_the_branch() {
        let mut harness = Harness::new();
        assert_eq!(
            harness.panel.results().get("model.cup_ring_check"),
            Some(Value::Text("Too thin".to_owned()))
        );
        harness.click_text("steel circuit");
        harness.click_text("no back iron");
        assert_eq!(harness.panel.inputs.coupling.backiron, 0);
        assert_eq!(
            harness.panel.results().get("model.cup_ring_check"),
            Some(Value::Text("No back iron".to_owned()))
        );
        let mut expected = DesignInputs::default();
        expected.coupling.backiron = 0;
        assert_eq!(harness.panel.results(), &compute_all(&expected));
    }

    #[test]
    fn a_text_input_edits_the_part_name_and_says_what_it_resolves_to() {
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, "Library part"),
            2,
            "both parts are library parts"
        );
        harness.focus(PART_INNER);
        harness.frame(vec![egui::Event::Text("X".to_owned())]);
        let part = harness.panel.inputs.coupling.magnets.part_inner.clone();
        assert!(part == "B842SHX" || part == "XB842SH", "{part}");
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Library part"), 1);
        assert_eq!(
            count(
                &output,
                "Not a library part: the manual dimensions are used"
            ),
            1
        );
        let mut expected = DesignInputs::default();
        expected.coupling.magnets.part_inner = part;
        assert_eq!(harness.panel.results(), &compute_all(&expected));
    }

    #[test]
    fn every_group_opens_and_draws_a_row_for_each_of_its_inputs() {
        let catalogue = InputCatalogue::get();
        for group in &catalogue.groups {
            let mut harness = Harness::new();
            harness.click_text(group.label);
            // The header opens over a few frames (its animation).
            let mut output = harness.frame(Vec::new());
            for _ in 0..10 {
                output = harness.frame(Vec::new());
            }
            let texts = drawn_texts(&output);
            for entry in group.sections.iter().flat_map(|s| s.entries.iter()) {
                assert!(
                    texts.iter().any(|t| t == entry.meta.label),
                    "{}: missing {:?}",
                    group.name,
                    entry.meta.label
                );
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
    }

    #[test]
    fn reset_all_restores_the_default_design_and_headline() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.focus(POLES);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_ne!(harness.panel.inputs(), &DesignInputs::default());

        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(
            harness.panel.results(),
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        for _ in 0..2 {
            // A window sizes itself on its first frame and paints on the next.
            output = Some(ctx.run(egui::RawInput::default(), |ctx| {
                egui::Window::new("Magnetic coupling").show(ctx, |ui| panel.ui(ui));
            }));
        }
        assert_drew_headline(&output.expect("two frames ran"), &DesignInputs::default());
    }

    #[test]
    fn input_tooltips_carry_help_path_and_cell() {
        let panel = MagcouplingPanel::new();
        let (path, meta) = panel.key_inputs[FACE_GAP];
        assert_eq!(
            input_tooltip(path, meta),
            "Same as the measured prototype.\nmetal.face_gap_mm\nMetal design!C119"
        );
        let (path, meta) = panel.key_inputs[AXIAL_LENGTH];
        assert!(
            input_tooltip(path, meta).starts_with("Used only if the part is not in the library.")
        );
    }
}
```

with:

```rust
        for _ in 0..2 {
            // A window sizes itself on its first frame and paints on the next.
            output = Some(ctx.run(egui::RawInput::default(), |ctx| {
                egui::Window::new("Magnetic coupling")
                    .default_size([1100.0, 700.0])
                    .show(ctx, |ui| panel.ui(ui));
            }));
        }
        assert_drew_headline(&output.expect("two frames ran"), &DesignInputs::default());
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/test_support.rs`, replace:

```rust
    })
}

/// A key press with no modifiers.
pub(crate) fn key_press(key: egui::Key) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }
}

/// A primary-button press (`pressed`) or release at `pos`.
```

with:

```rust
    })
}

/// A key event with no modifiers: a press, or (`pressed` false) its release.
pub(crate) fn key_event(key: egui::Key, pressed: bool) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }
}

/// A key tapped with no modifiers: pressed and released in one frame, as a user taps it. egui
/// keeps a pressed key down until its release and reads another press of it as a repeat.
pub(crate) fn key_tap(key: egui::Key) -> Vec<egui::Event> {
    vec![key_event(key, true), key_event(key, false)]
}

/// Ctrl+A (Cmd+A on a Mac) tapped: select all in the focused text field.
pub(crate) fn select_all() -> Vec<egui::Event> {
    [true, false]
        .map(|pressed| egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        })
        .to_vec()
}

/// A primary-button press (`pressed`) or release at `pos`.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs`, replace:

```rust
pub(crate) mod test_support;

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use panel::{KEY_INPUTS, MagcouplingPanel};
```

with:

```rust
pub(crate) mod test_support;

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
pub use panel::MagcouplingPanel;
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | sort -u | head -8
```

Expected: the build fails with compile errors naming the items Step 3 adds, among them ``error[E0432]: unresolved import `crate::gui::input_ui` ``, ``error[E0433]: failed to resolve: use of undeclared type `InputCatalogue` ``, ``error[E0609]: no field `key_widgets` on type `gui::panel::MagcouplingPanel` `` and ``error[E0425]: cannot find value `RESET_ALL` in this scope``.

- [ ] **Step 3: Write the row widget and the layout**



Create `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/input_ui.rs`:

```rust
//! One input row of the left side (spec M4 "Sliders"): live recompute while dragging, a value
//! box for typed entry, arrow-key nudges, step snapping, logarithmic scale, per-field reset, a
//! dot when changed from default, a tooltip with help text and workbook cell; selectors as
//! drop-downs, an optional input as a checkbox and a slider, a text input as a text field.
//!
//! [`input_row`] draws a row from its [`InputEntry`] and the current value and returns the
//! edit asked for; the panel applies it with `InputSet::set`, so every edit is checked the
//! same way.

use crate::engine::meta::{FieldType, InputMeta, SliderRange, Value};
use crate::gui::format::{format_value, with_unit};
use crate::gui::inputs::{InputEntry, input_tooltip, outside_range, step_decimals, text_hint};

/// The changed-from-default dot.
pub const CHANGED_DOT: &str = "\u{2022}";

/// The per-field reset button.
pub const RESET_LABEL: &str = "reset";

/// The note on a value outside its slider range.
pub const OUTSIDE_RANGE_NOTE: &str = "outside the slider range";

/// The text of a blank optional input.
pub const BLANK_TEXT: &str = "blank";

/// What the user asked of an input row this frame.
#[derive(Clone, Debug, PartialEq)]
pub enum RowEdit {
    /// Set the input to this value.
    Set(Value),
    /// Back to the default.
    Reset,
}

/// What an input row drew and asked for.
pub struct RowOutput {
    /// The edit, if any.
    pub edit: Option<RowEdit>,
    /// The row's main widget (the slider, drop-down, checkbox or text field).
    pub widget: egui::Response,
}

/// A slider over `value` set up from the input's metadata: its range (logarithmic when
/// flagged), its step (also the arrow-key nudge), values rounded to the step's decimals
/// (decision M41-1), edits clamped to the range while a value already outside it is kept until
/// edited (decision M41-2), typed values applied on Enter or when the box loses focus, the
/// unit as a suffix.
pub fn slider<'a>(value: &'a mut f64, meta: &InputMeta, range: SliderRange) -> egui::Slider<'a> {
    let mut slider = egui::Slider::new(value, range.min..=range.max)
        .logarithmic(range.log)
        .clamping(egui::SliderClamping::Edits)
        .update_while_editing(false);
    if meta.ty == FieldType::I64 {
        slider = slider.integer();
    } else {
        slider = slider.max_decimals(step_decimals(range.step));
    }
    // After integer(), which sets a step of 1: pole counts step by 2.
    slider = slider.step_by(range.step);
    let suffix = with_unit(String::new(), meta.unit);
    if !suffix.is_empty() {
        slider = slider.suffix(suffix);
    }
    slider
}

/// The number in a value, if it is one.
fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Num(x) => Some(*x),
        Value::Int(i) => Some(*i as f64),
        _ => None,
    }
}

/// The value a slider position stands for, in the input's type.
fn typed(meta: &InputMeta, x: f64) -> Value {
    match meta.ty {
        FieldType::I64 => Value::Int(x.round() as i64),
        _ => Value::Num(x),
    }
}

/// Draws one input row: a header line (the changed dot, the label, an out-of-range note, the
/// reset button) and the widget under it. `seed` is where an optional input starts when the
/// user enters a value (`inputs::OPTIONAL_SEEDS`).
pub fn input_row(
    ui: &mut egui::Ui,
    entry: &InputEntry,
    current: &Value,
    seed: Option<f64>,
) -> RowOutput {
    let meta = entry.meta;
    let tooltip = input_tooltip(entry);
    let changed = *current != entry.default;
    let mut edit = None;
    ui.push_id(&entry.path, |ui| {
        ui.horizontal(|ui| {
            let dot = if changed { CHANGED_DOT } else { " " };
            ui.colored_label(ui.visuals().selection.stroke.color, dot)
                .on_hover_text("Changed from the default");
            ui.label(meta.label).on_hover_text(&tooltip);
            if outside_range(meta.range, current) {
                ui.colored_label(ui.visuals().warn_fg_color, OUTSIDE_RANGE_NOTE)
                    .on_hover_text(
                        "Kept until edited. The differential tests cover the slider range only.",
                    );
            }
            if changed
                && ui
                    .small_button(RESET_LABEL)
                    .on_hover_text(format!(
                        "Back to the default: {}",
                        format_value(&entry.default)
                    ))
                    .clicked()
            {
                edit = Some(RowEdit::Reset);
            }
        });
        let widget = widget(ui, entry, current, seed, &mut edit).on_hover_text(&tooltip);
        // Text inputs only (`text_hint` knows no other path).
        if let Some(hint) = text_hint(&entry.path, current_text(current)) {
            ui.weak(hint);
        }
        RowOutput { edit, widget }
    })
    .inner
}

fn current_text(value: &Value) -> &str {
    match value {
        Value::Text(text) => text,
        _ => "",
    }
}

/// The row's widget; sets `edit` when the user changed the value. A new edit replaces a reset
/// asked in the same frame (it cannot happen: the reset button and the widget are two clicks).
fn widget(
    ui: &mut egui::Ui,
    entry: &InputEntry,
    current: &Value,
    seed: Option<f64>,
    edit: &mut Option<RowEdit>,
) -> egui::Response {
    let meta = entry.meta;
    match (meta.ty, current) {
        (FieldType::I64, Value::Int(code)) if !meta.choices.is_empty() => {
            let mut selected = *code;
            let text = meta.choices.iter().find(|(c, _)| c == code).map_or_else(
                || format!("{code} (not a choice)"),
                |(_, t)| (*t).to_owned(),
            );
            let response = egui::ComboBox::from_id_salt("choice")
                .selected_text(text)
                .show_ui(ui, |ui| {
                    for &(choice, label) in meta.choices {
                        ui.selectable_value(&mut selected, choice, label);
                    }
                })
                .response;
            if selected != *code {
                *edit = Some(RowEdit::Set(Value::Int(selected)));
            }
            response
        }
        (FieldType::OptF64, _) => {
            ui.horizontal(|ui| {
                let mut entered = !matches!(current, Value::None);
                let checkbox = ui
                    .checkbox(&mut entered, "")
                    .on_hover_text("Enter a value; clear to leave the input blank");
                if checkbox.changed() {
                    *edit = Some(RowEdit::Set(match (entered, seed) {
                        (true, Some(x)) => Value::Num(x),
                        (true, None) => Value::Num(meta.range.map_or(0.0, |r| r.min)),
                        (false, _) => Value::None,
                    }));
                }
                // The slider while a value is entered, else the checkbox.
                match (number(current), meta.range) {
                    (Some(x), Some(range)) => {
                        let mut value = x;
                        let response = ui.add(slider(&mut value, meta, range));
                        if response.changed() && value != x {
                            *edit = Some(RowEdit::Set(Value::Num(value)));
                        }
                        response
                    }
                    _ => {
                        ui.weak(BLANK_TEXT);
                        checkbox
                    }
                }
            })
            .inner
        }
        (FieldType::Text, _) => {
            let mut text = current_text(current).to_owned();
            let response =
                ui.add(egui::TextEdit::singleline(&mut text).desired_width(f32::INFINITY));
            if response.changed() {
                *edit = Some(RowEdit::Set(Value::Text(text)));
            }
            response
        }
        _ => match (number(current), meta.range) {
            (Some(x), Some(range)) => {
                let mut value = x;
                let response = ui.add(slider(&mut value, meta, range));
                if response.changed() && value != x {
                    *edit = Some(RowEdit::Set(typed(meta, value)));
                }
                response
            }
            (Some(x), None) => {
                let mut value = x;
                let response = ui.add(egui::DragValue::new(&mut value));
                if response.changed() && value != x {
                    *edit = Some(RowEdit::Set(typed(meta, value)));
                }
                response
            }
            _ => ui.label(format_value(current)),
        },
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs`, replace:

```rust
}

impl MagcouplingApp {
    /// Draws one frame: the panel in a scrolling central panel.
    pub fn ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.panel.ui(ui));
        });
    }
}

```

with:

```rust
}

impl MagcouplingApp {
    /// Draws one frame: the panel fills the window (its sides scroll on their own).
    pub fn ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| self.panel.ui(ui));
    }
}

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
//! [`MagcouplingPanel`]: the calculator's state and its egui UI.
//!
//! The M4 infrastructure tracer: a few key-input sliders generated from the
//! engine's input metadata and the headline numbers, recomputed every frame
//! with every approved correction on ([`compute_all`]). The M4 plans grow it
//! into the spec's layout (inputs, geometry view, dashboard, plots).

use crate::engine::api::HEADLINE;
use crate::engine::meta::{
    FieldType, InputMeta, InputSet, ResultMeta, Value, input_rows, result_rows,
};
use crate::gui::format::{format_value, with_unit};
use crate::{DesignInputs, DesignResults, compute_all, headline};

/// Input paths of the tracer's sliders: face gap, pole count and axial length,
/// the first entries of the spec's Key design group.
///
/// The axial length is the manual inner magnet length, which the engine uses
/// only when the inner part is not a library part (its help says so); at the
/// default part (B842SH) it moves no result.
pub const KEY_INPUTS: [&str; 3] = [
    "metal.face_gap_mm",
    "coupling.npole",
    "coupling.magnets.manual_inner_length_mm",
];

/// The calculator panel: design inputs, their results, and the UI that edits
/// the one and shows the other.
///
/// Hostable by any egui app: the standalone app shows it as a full page
/// (`app::MagcouplingApp`), the linkage app in an `egui::Window` (M5). Call
```

with:

```rust
//! [`MagcouplingPanel`]: the calculator's state and its egui UI.
//!
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the dashboard on the right and
//! the centre region between them. Every result is recomputed with every approved correction
//! on ([`compute_all`]) each frame after the inputs are drawn, so the readouts show this
//! frame's edits.

use crate::engine::api::HEADLINE;
use crate::engine::meta::{InputSet, ResultMeta, ResultSet, Value, result_rows};
use crate::gui::format::{format_value, with_unit};
use crate::gui::input_ui::{RowEdit, input_row};
use crate::gui::inputs::{InputCatalogue, InputEntry, optional_seed};
use crate::{DesignInputs, DesignResults, compute_all, headline};

/// The heading of the panel.
pub const HEADING: &str = "Magnetic coupling calculator";

/// The label of the button that restores the default design.
pub const RESET_ALL: &str = "Reset all";

/// The heading of the Key design group.
pub const KEY_DESIGN_HEADING: &str = "Key design";

/// Starting width of the inputs side [points].
const INPUTS_WIDTH: f32 = 320.0;

/// Starting width of the dashboard side [points].
const DASHBOARD_WIDTH: f32 = 300.0;

/// The calculator panel: design inputs, their results, and the UI that edits the one and
/// shows the other.
///
/// Hostable by any egui app: the standalone app shows it as a full page
/// (`app::MagcouplingApp`), the linkage app in an `egui::Window` (M5). Call
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    inputs: DesignInputs,
    /// The results of `inputs`, recomputed by every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// Metadata of each [`KEY_INPUTS`] path, in order.
    key_inputs: [(&'static str, &'static InputMeta); KEY_INPUTS.len()],
    /// Metadata of each [`HEADLINE`] result, in order.
    headline_meta: [&'static ResultMeta; HEADLINE.len()],
    /// The id of each key input's slider in the last frame (the slider rail).
    slider_ids: [Option<egui::Id>; KEY_INPUTS.len()],
    /// Why the last edit was refused, until the next accepted edit or reset.
    last_error: Option<String>,
}
```

with:

```rust
    inputs: DesignInputs,
    /// The results of `inputs`, recomputed by every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// Metadata of each [`HEADLINE`] result, in order.
    headline_meta: [&'static ResultMeta; HEADLINE.len()],
    /// The main widget of each Key design row in the last frame, by input path.
    key_widgets: Vec<(&'static str, egui::Id)>,
    /// Why the last edit was refused, until the next accepted edit or reset.
    last_error: Option<String>,
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    pub fn new() -> Self {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let input_meta = input_rows(&inputs);
        let result_meta = result_rows(&results);
        let key_inputs = KEY_INPUTS.map(|path| {
            let row = input_meta.iter().find(|row| row.path == path);
            (
                path,
                row.unwrap_or_else(|| panic!("KEY_INPUTS: no input {path}"))
                    .meta,
            )
        });
        let headline_meta = HEADLINE.map(|(key, path)| {
            let row = result_meta.iter().find(|row| row.path == path);
            row.unwrap_or_else(|| panic!("HEADLINE {key}: no result {path}"))
```

with:

```rust
    pub fn new() -> Self {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let result_meta = result_rows(&results);
        let headline_meta = HEADLINE.map(|(key, path)| {
            let row = result_meta.iter().find(|row| row.path == path);
            row.unwrap_or_else(|| panic!("HEADLINE {key}: no result {path}"))
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        Self {
            inputs,
            results,
            key_inputs,
            headline_meta,
            slider_ids: [None; KEY_INPUTS.len()],
            last_error: None,
        }
    }
```

with:

```rust
        Self {
            inputs,
            results,
            headline_meta,
            key_widgets: Vec::new(),
            last_error: None,
        }
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    }

    /// Draws the panel into `ui` and applies this frame's edits.
    ///
    /// Recomputes every result after the inputs are drawn, so the readout
    /// shows this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.push_id("magcoupling_panel", |ui| {
            ui.heading("Magnetic coupling calculator");
            for (index, (path, meta)) in self.key_inputs.into_iter().enumerate() {
                self.slider_ids[index] = self.slider(ui, path, meta);
            }
            if let Some(error) = &self.last_error {
                ui.colored_label(ui.visuals().error_fg_color, error);
            }
            if ui.button("Reset all").clicked() {
                self.reset();
            }
            self.results = compute_all(&self.inputs);
            ui.separator();
            self.headline_ui(ui);
        });
    }

    /// One key input's slider, set up from its metadata; returns its id, or
    /// `None` when the input cannot have a slider.
    fn slider(
        &mut self,
        ui: &mut egui::Ui,
        path: &'static str,
        meta: &'static InputMeta,
    ) -> Option<egui::Id> {
        let (range, current) = match (meta.range, self.inputs.get(path)) {
            (Some(range), Some(Value::Num(x))) => (range, x),
            (Some(range), Some(Value::Int(i))) => (range, i as f64),
            (range, value) => {
                ui.label(format!(
                    "{}: no slider (range {range:?}, value {value:?})",
                    meta.label
                ));
                return None;
            }
        };
        let mut value = current;
        let mut slider = egui::Slider::new(&mut value, range.min..=range.max)
            .text(meta.label)
            .logarithmic(range.log)
            // Edits: the slider, arrow keys and typed values stay in the range,
            // and a value already outside it is kept until edited (never
            // rewritten by an idle frame).
            .clamping(egui::SliderClamping::Edits);
        if meta.ty == FieldType::I64 {
            slider = slider.integer();
        }
        // After integer(), which sets a step of 1: pole counts step by 2.
        slider = slider.step_by(range.step);
        if meta.unit != "-" {
            slider = slider.suffix(format!(" {}", meta.unit));
        }
        let response = ui.add(slider).on_hover_text(input_tooltip(path, meta));
        if response.changed() && value != current {
            let new = match meta.ty {
                FieldType::I64 => Value::Int(value.round() as i64),
                _ => Value::Num(value),
            };
            self.last_error = self.inputs.set(path, new).err().map(|e| e.to_string());
        }
        Some(response.id)
    }

    /// The headline numbers: label, then value with unit; the hover shows the
```

with:

```rust
    }

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.push_id("magcoupling_panel", |ui| {
            egui::TopBottomPanel::top("magcoupling_header").show_inside(ui, |ui| {
                self.header_ui(ui);
            });
            egui::SidePanel::left("magcoupling_inputs")
                .resizable(true)
                .default_width(INPUTS_WIDTH)
                .show_inside(ui, |ui| self.inputs_ui(ui));
            // After the inputs: the readouts show this frame's edits.
            self.results = compute_all(&self.inputs);
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
                .show_inside(ui, |ui| self.headline_ui(ui));
            egui::CentralPanel::default().show_inside(ui, |_ui| {});
        });
    }

    /// The header line: heading, the session buttons, the last refusal.
    fn header_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(HEADING);
            ui.separator();
            if ui.button(RESET_ALL).clicked() {
                self.reset();
            }
        });
        if let Some(error) = &self.last_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
    }

    /// The left side: the Key design group, then every input by package group.
    fn inputs_ui(&mut self, ui: &mut egui::Ui) {
        let catalogue = InputCatalogue::get();
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::CollapsingHeader::new(KEY_DESIGN_HEADING)
                    .id_salt("key_design")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.key_widgets.clear();
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry);
                            self.key_widgets.push((entry.path.as_str(), widget));
                        }
                    });
                for group in &catalogue.groups {
                    egui::CollapsingHeader::new(group.label)
                        .id_salt(("group", &group.name))
                        .default_open(false)
                        .show(ui, |ui| {
                            for section in &group.sections {
                                if section.prefix != group.name {
                                    ui.add_space(4.0);
                                    ui.strong(section.label);
                                }
                                for entry in &section.entries {
                                    self.input_row_ui(ui, entry);
                                }
                            }
                        });
                }
            });
    }

    /// One input row; applies its edit. Returns the id of its main widget.
    fn input_row_ui(&mut self, ui: &mut egui::Ui, entry: &'static InputEntry) -> egui::Id {
        let current = self.inputs.get(&entry.path).unwrap_or(Value::None);
        let seed = self.seed(entry);
        let output = input_row(ui, entry, &current, seed);
        if let Some(edit) = output.edit {
            let value = match edit {
                RowEdit::Set(value) => value,
                RowEdit::Reset => entry.default.clone(),
            };
            self.last_error = self
                .inputs
                .set(&entry.path, value)
                .err()
                .map(|e| e.to_string());
        }
        output.widget.id
    }

    /// Where an optional input starts when a value is entered: the result it overrides,
    /// clamped into its slider range but not rounded to the step (decision M41-12: rounding
    /// would move the design it is meant to keep). Like a loaded value, an off-grid seed is
    /// kept until the first edit.
    fn seed(&self, entry: &InputEntry) -> Option<f64> {
        let range = entry.meta.range?;
        match self.results.get(optional_seed(&entry.path)?)? {
            Value::Num(x) if x.is_finite() => Some(x.clamp(range.min, range.max)),
            _ => None,
        }
    }

    /// The headline numbers: label, then value with unit; the hover shows the
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
                }
            });
    }
}

/// Hover text of an input: its help, path and workbook cell.
fn input_tooltip(path: &str, meta: &InputMeta) -> String {
    let mut text = String::new();
    if !meta.help.is_empty() {
        text.push_str(meta.help);
        text.push('\n');
    }
    text.push_str(path);
    text.push('\n');
    text.push_str(meta.cell.unwrap_or("no workbook cell"));
    text
}

#[cfg(test)]
```

with:

```rust
                }
            });
    }
}

#[cfg(test)]
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs`, replace:

```rust

mod format;
pub mod history;
pub mod inputs;
mod panel;
pub mod session;
```

with:

```rust

mod format;
pub mod history;
pub mod input_ui;
pub mod inputs;
mod panel;
pub mod session;
```

- [ ] **Step 4: Run the tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED|panicked"
```

Expected: `test result: ok. 205 passed; 0 failed` (Task 3's 200, with the tracer's panel tests rewritten for the new layout and new ones added).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task4.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task4.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task4.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task4.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/app.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/input_ui.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/test_support.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
feat(magcoupling-rs): every input as a row, in the three-region layout

One row widget for every field type, from the metadata: slider and value box
(live, arrow nudges, log scale, step decimals M41-1, edits clamped and an
out-of-range value kept and flagged M41-2), drop-down, optional input entered at
its seed, unrounded (M41-12), text with its hint; the changed dot, per-field reset and the
tooltip. The panel takes the spec's layout: header, inputs on the left with the
Key design group, the headline on the right, the centre for the results table.
The Key design axial length is the A-2 override.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 5: The dashboard

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

Spec M4 "Layout": "Right - dashboard: headline numbers with green/amber/red badges derived from check verdicts; corrected values carry a 'corrected vs workbook' marker with the deviation in its tooltip", and the user's decisions of 2026-09-30. `gui/corrections.rs` builds, once, the index of every workbook cell the registry ties to an applied engine correction (changed at defaults, named, or in a probe), with the reviewed golden files E3, E4 and E5 compiled in (`include_str!`; a test keeps the list equal to the registry's `changes_file` entries), so the headline pull-out carries E3 with the workbook's 2.647 (decision M41-10). `gui/dashboard.rs` lists the rows (`DASHBOARD`: the 15 headline numbers in Python's order, the space claim badge, the end-effect flag), classifies each check's verdict (`verdict_level`, every branch reached by a design in the tests), greys the pull-out rows without badges when f_end <= 0 under the banner "End-effect model out of range" (`END_EFFECT_ROWS`, pinned against a probe of c_end at back iron 1 and 0, c_end = 0 included because only it flips the hot-minimum verdict; M41-11), and labels the temperature rows "3D values from the workbook" (`STORED_3D_ROWS`, pinned against the 14 stored 3D inputs at their slider ends). `result_tooltip` is the hover hook every readout uses, keyed by result path (M4-3 adds the equation there). Each row draws on two lines (badge and label, then the value and its marker), so a narrow side still reads.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/corrections.rs`
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/dashboard.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs` (the dashboard on the right; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs` (`pub mod corrections; pub mod dashboard;`)

**Interfaces:**
- Consumes: `engine::deviations::{REGISTRY, Deviation, DeviationClass, DeviationId, DeviationStatus}` (`cells`, `changes_at_defaults`, `changes_file`, `probes`, `title`, `evidence()`), `engine::model::END_EFFECT_OUT_OF_RANGE`, `engine::housing::{INSIDE_THE_SPACE_CLAIM, SPACE_CLAIM_UNKNOWN}`, `engine::meta::{ResultMeta, ResultSet, Value, result_rows}`, Task 3's `InputCatalogue` (tests).
- Produces (`magcoupling::gui::corrections`): `const GOLDEN_FILES: [(DeviationId, &str, &str); 3]`; `enum Tie { AtDefaults { workbook: Value, corrected: Value }, Named }`; `struct Mark { pub id: DeviationId, pub tie: Tie }`; `struct CorrectionIndex` with `fn build() -> Self`, `fn get() -> &'static CorrectionIndex`, `fn marks(&self, cell: Option<&str>) -> &[Mark]`; `fn marker_text(&[Mark]) -> String` (`"E3 E7 E8"`); `fn marker_tooltip(&[Mark]) -> String`.
- Produces (`magcoupling::gui::dashboard`): `enum Level { Good, Caution, Bad }` with `word()` and `color(&egui::Visuals)`; `const DASHBOARD: [(&str, Option<&str>); 17]`; `const END_EFFECT_ROWS: [&str; 7]`; `const STORED_3D_ROWS: [&str; 3]`; `STORED_3D_LABEL`, `STORED_3D_NOTE`, `END_EFFECT_BANNER`; `fn verdict_level(path: &str, text: &str) -> Option<Level>`; `struct ResultInfo { pub meta: &'static ResultMeta, pub cell: Option<String> }`; `fn result_info(&str) -> Option<&'static ResultInfo>`; `struct ResultNotes<'a> { pub marks: &'a [Mark], pub greyed: bool, pub stored_3d: bool }`; `fn result_tooltip(path: &str, info: &ResultInfo, notes: ResultNotes<'_>) -> String`; `struct DashboardLine { path, label, value, level, greyed, stored_3d, marker, tooltip }`; `fn end_effect_out_of_range(&DesignResults) -> Option<f64>`; `fn dashboard_lines(&DesignResults) -> Vec<DashboardLine>`; `fn dashboard_ui(&mut egui::Ui, &DesignResults)`.

- [ ] **Step 1: Write the failing tests**

The two new modules' docs and tests, and the panel's dashboard tests: the dashboard starts with `HEADLINE`; every verdict of every badge check has a level (designs reach each branch: covers hot min, meets the clearance, a 6 mm wall, a hot-day CHECK, no clamp screw, a sized design past the claim, a NaN claim, short magnets); the two pinned row lists against their probes; the default dashboard's badges, markers and 3D rows; the greying and its banner text with f_end = -1.202; the space-claim overshoot; the result tooltip; the badge colours per theme; the golden files equal the registry's; E3 on the pull-out with 2.647 and the marker `E3 E7 E8`; E2 and E6 marks; no mark for unmarked cells, Rust-only results or E14. The window test gets room for the two-line rows.

Create `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/corrections.rs`:

```rust
//! The "corrected vs workbook" markers (spec M4 "Layout": "corrected values carry a
//! 'corrected vs workbook' marker with the deviation in its tooltip").
//!
//! The panel always computes with every approved correction on, and the test-only switch that
//! turns them off never ships, so a marker cannot come from comparing the two. It comes from
//! the deviation registry instead (decision M41-10): a workbook cell carries a correction's
//! marker when the registry ties the cell to it, in any of three ways:
//!
//! - the correction changes the cell at the default design (`changes_at_defaults`, or for the
//!   broad corrections E3, E4 and E5 their reviewed golden files, which hold most of what E3
//!   moves, the headline pull-out included);
//! - the report names the cell for the correction (`cells`);
//! - one of its probes shows the correction changing the cell off the default design.
//!
//! Only applied engine corrections mark (E14 rewords help only). [`CorrectionIndex`] is built
//! once, keyed by workbook cell; a value with no cell (a Rust-only result) has no marker. Plan
//! A-3's equation registry may later supply each record's upstream corrections instead.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::meta::result_rows;
    use crate::{DesignInputs, compute_all};

    #[test]
    fn the_golden_files_are_the_registry_s() {
        let named: Vec<(DeviationId, &str)> = REGISTRY
            .iter()
            .filter_map(|d| d.changes_file.map(|file| (d.id, file)))
            .collect();
        let compiled: Vec<(DeviationId, &str)> = GOLDEN_FILES
            .iter()
            .map(|(id, file, _)| (*id, *file))
            .collect();
        assert_eq!(compiled, named);
        for (id, _, text) in GOLDEN_FILES {
            let json: Json = serde_json::from_str(text).unwrap();
            assert_eq!(json["id"], id.to_string());
        }
    }

    #[test]
    fn the_headline_pull_out_carries_e3_with_its_workbook_value() {
        let index = CorrectionIndex::get();
        let marks = index.marks(Some("Calculator!C93"));
        let e3 = marks.iter().find(|m| m.id == DeviationId::E3).expect("E3");
        let Tie::AtDefaults {
            workbook,
            corrected,
        } = &e3.tie
        else {
            panic!("E3 changes C93 at the defaults")
        };
        assert_eq!(format_value(workbook), "2.647");
        let pullout = compute_all(&DesignInputs::default()).model.pullout_Nm;
        assert_eq!(corrected, &Value::Num(pullout));
        // E7 and E8 have probes on it; the ids are in report order.
        assert_eq!(marker_text(marks), "E3 E7 E8");
        let tooltip = marker_tooltip(marks);
        assert!(
            tooltip.starts_with("Corrected vs workbook:\nE3: "),
            "{tooltip}"
        );
        assert!(
            tooltip.contains("at the default design: workbook 2.647, corrected 2.688"),
            "{tooltip}"
        );
        assert!(tooltip.contains("docs/analyses/2026-09-29-magcoupling-math-audit.md, entry E3"));
    }

    #[test]
    fn a_hand_listed_change_and_a_named_cell_mark_too() {
        let index = CorrectionIndex::get();
        // E2 lists the clamp screw at the defaults.
        let screw = index.marks(Some("Shaft clamps!C48"));
        assert!(screw.iter().any(|m| m.id == DeviationId::E2
            && m.tie
                == Tie::AtDefaults {
                    workbook: Value::Text("ISO 4762 M4 x 12, class 12.9".to_owned()),
                    corrected: Value::Text("ISO 4762 M4 x 14, class 12.9".to_owned()),
                }));
        // E6 names Calculator!C63 (and changes it).
        assert!(
            index
                .marks(Some("Calculator!C63"))
                .iter()
                .any(|m| m.id == DeviationId::E6)
        );
    }

    #[test]
    fn unmarked_cells_rust_only_results_and_documentation_corrections_have_no_marker() {
        let index = CorrectionIndex::get();
        assert!(index.marks(None).is_empty());
        assert!(index.marks(Some("No sheet!Z99")).is_empty());
        // E14 rewords Shaft clamps!C35's help only.
        assert!(
            !index
                .marks(Some("Shaft clamps!C35"))
                .iter()
                .any(|m| m.id == DeviationId::E14)
        );
        assert_eq!(marker_text(&[]), "");
    }

    #[test]
    fn every_marked_cell_belongs_to_a_registry_entry_and_most_results_are_unmarked() {
        let index = CorrectionIndex::get();
        let results = compute_all(&DesignInputs::default());
        let rows = result_rows(&results);
        let marked = rows
            .iter()
            .filter(|row| !index.marks(row.cell.as_deref()).is_empty())
            .count();
        // A marker means something: well under half of the results carry one.
        assert!(
            marked > 100 && marked < rows.len() / 2,
            "{marked} of {}",
            rows.len()
        );
        for marks in index.by_cell.values() {
            for mark in marks {
                assert!(marks_values(&REGISTRY[mark.id.index()]));
            }
        }
    }
}
```

Create `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/dashboard.rs`:

```rust
//! The dashboard on the right (spec M4 "Layout": "headline numbers with green/amber/red
//! badges derived from check verdicts; corrected values carry a 'corrected vs workbook' marker
//! with the deviation in its tooltip").
//!
//! [`DASHBOARD`] lists its rows: the 15 headline numbers in Python's order, then the A1 space
//! claim badge and the audit M9 end-effect flag. A row's badge comes from a check verdict
//! ([`verdict_level`]); its marker from the deviation registry
//! ([`crate::gui::corrections`]). Two notes from the user's decisions of 2026-09-30:
//!
//! - **End effect (audit M9).** When f_end ≤ 0 the pull-out and the numbers computed from it
//!   ([`END_EFFECT_ROWS`]) are greyed, without a badge, under the banner "End-effect model out
//!   of range".
//! - **Stored 3D values.** M3 (the live 3D field model) comes after M4, so the temperature
//!   rows that read the workbook's stored 3D fields ([`STORED_3D_ROWS`]) carry the label
//!   "3D values from the workbook" instead of a "3D updating" badge.
//!
//! [`result_tooltip`] is the hover text of every displayed result, keyed by result path: the
//! one hook the dashboard and the results table (and later the geometry callouts and the
//! equation explorer) go through.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::api::HEADLINE;
    use crate::engine::meta::InputSet;
    use std::collections::BTreeSet;

    #[test]
    fn the_dashboard_starts_with_the_headline_in_order() {
        let paths: Vec<&str> = DASHBOARD.iter().map(|(path, _)| *path).collect();
        let headline: Vec<&str> = HEADLINE.iter().map(|(_, path)| *path).collect();
        assert_eq!(paths[..15], headline[..]);
        assert_eq!(
            paths[15..],
            ["housing.space_claim_check", "model.end_effect_check"]
        );
        for path in END_EFFECT_ROWS.iter().chain(STORED_3D_ROWS.iter()) {
            assert!(paths.contains(path), "{path}");
        }
    }

    /// The defaults with `edit` applied.
    fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        edit(&mut inputs);
        inputs
    }

    /// A design with f_end below 0 (the engine's short-magnet test): 2 mm manual blocks, c_end 0.5.
    pub(crate) fn short_magnets() -> DesignInputs {
        design(|i| {
            i.coupling.c_end = 0.5;
            i.coupling.magnets.part_inner.clear();
            i.coupling.magnets.part_outer.clear();
            i.coupling.magnets.manual_inner_length_mm = 2.0;
            i.coupling.magnets.manual_outer_length_mm = 2.0;
        })
    }

    #[test]
    fn every_verdict_each_check_gives_has_a_level() {
        use Level::{Bad, Caution, Good};
        let cases: Vec<(&str, DesignInputs, Level)> = vec![
            ("metal.hot_min_check", DesignInputs::default(), Bad),
            (
                "metal.hot_min_check",
                design(|i| i.metal.required_min_Nm = 0.1),
                Good,
            ),
            ("metal.clearance_check", DesignInputs::default(), Bad),
            (
                "metal.clearance_check",
                design(|i| i.metal.face_gap_mm = 5.0),
                Good,
            ),
            ("materials.cup_wall_check", DesignInputs::default(), Bad),
            (
                "materials.cup_wall_check",
                design(|i| i.metal.cup_wall_corner_mm = 6.0),
                Good,
            ),
            (TEMPERATURE_VERDICT, DesignInputs::default(), Good),
            (
                TEMPERATURE_VERDICT,
                design(|i| {
                    i.temperature.duty.hot_ambient_C = 80.0;
                    i.temperature.duty.driving_rise_C = 60.0;
                }),
                Caution,
            ),
            ("clamps.recommended", DesignInputs::default(), Good),
            (
                "clamps.recommended",
                design(|i| {
                    i.clamps.boss_od_mm = 12.0;
                    i.clamps.clamp_length_mm = 3.0;
                }),
                Bad,
            ),
            ("housing.space_claim_check", DesignInputs::default(), Good),
            (
                "housing.space_claim_check",
                design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)),
                Bad,
            ),
            (
                "housing.space_claim_check",
                design(|i| i.metal.max_diameter_mm = f64::NAN),
                Caution,
            ),
            ("model.end_effect_check", DesignInputs::default(), Good),
            ("model.end_effect_check", short_magnets(), Bad),
        ];
        for (check, inputs, want) in cases {
            let Some(Value::Text(text)) = compute_all(&inputs).get(check) else {
                panic!("{check} is text")
            };
            assert_eq!(verdict_level(check, &text), Some(want), "{check}: {text:?}");
        }
        assert_eq!(verdict_level("metal.hot_min_check", "OK"), None);
        assert_eq!(verdict_level("model.pullout_Nm", "OK"), None);
    }

    /// The headline paths whose value moves when `inputs` take any of `values`, from the
    /// defaults at back iron 1 and 0.
    fn headline_moved_by(
        inputs: &[String],
        values: impl Fn(&str) -> Vec<f64>,
    ) -> BTreeSet<&'static str> {
        let mut moved = BTreeSet::new();
        for backiron in [1, 0] {
            let base = design(|i| i.coupling.backiron = backiron);
            let before = compute_all(&base);
            for path in inputs {
                for x in values(path) {
                    let mut edited = base.clone();
                    edited.set(path, Value::Num(x)).unwrap();
                    let after = compute_all(&edited);
                    for (_, result) in HEADLINE {
                        if before.get(result) != after.get(result) {
                            moved.insert(result);
                        }
                    }
                }
            }
        }
        moved
    }

    #[test]
    fn the_end_effect_rows_are_the_headline_rows_that_move_with_the_end_effect_coefficient() {
        // c_end = 0 (f_end = 1) is the one value that flips the hot-minimum verdict.
        let moved = headline_moved_by(&["coupling.c_end".to_owned()], |_| {
            vec![0.0, 0.05, 0.3, 0.5]
        });
        assert_eq!(moved, END_EFFECT_ROWS.into_iter().collect());
    }

    #[test]
    fn the_stored_3d_rows_are_the_headline_rows_that_move_with_the_stored_3d_inputs() {
        // The stored 3D inputs: the four reverse fields of the demagnetization check and the
        // slip-loss fields (steel circuit, and E17's free-space ones), which M3 computes live.
        let inputs: Vec<String> = [
            "temperature.demag.h_rev_aligned_kA_m",
            "temperature.demag.h_rev_pullout_kA_m",
            "temperature.demag.h_rev_likepole_kA_m",
            "temperature.demag.h_rev_single_ring_kA_m",
            "temperature.slip_loss.b_hub_T",
            "temperature.slip_loss.b_cup_T",
            "temperature.slip_loss.b_sleeve_T",
            "temperature.slip_loss.b_liner_T",
            "temperature.slip_loss.cap_integral_T2m4",
            "temperature.slip_loss.web_integral_T2m2",
            "temperature.slip_loss.b_magnet_T",
            "temperature.slip_loss.b_hub_free_T",
            "temperature.slip_loss.b_cup_free_T",
            "temperature.slip_loss.web_integral_free_T2m2",
        ]
        .map(str::to_owned)
        .to_vec();
        let catalogue = crate::gui::inputs::InputCatalogue::get();
        let moved = headline_moved_by(&inputs, |path| {
            let range = catalogue.entry(path).unwrap().meta.range.unwrap();
            vec![range.min, range.max]
        });
        assert_eq!(moved, STORED_3D_ROWS.into_iter().collect());
    }

    #[test]
    fn the_default_dashboard_has_badges_markers_and_the_3d_rows() {
        let lines = dashboard_lines(&compute_all(&DesignInputs::default()));
        assert_eq!(lines.len(), DASHBOARD.len());
        let line = |path: &str| lines.iter().find(|l| l.path == path).unwrap();
        assert_eq!(line("model.pullout_Nm").value, "2.688 N·m");
        assert_eq!(line("model.pullout_Nm").level, None);
        assert_eq!(line("model.pullout_Nm").marker, "E3 E7 E8");
        assert!(
            line("model.pullout_Nm")
                .tooltip
                .contains("workbook 2.647, corrected 2.688")
        );
        assert_eq!(line("metal.torque_hot_low_Nm").level, Some(Level::Bad));
        assert_eq!(line("materials.cup_wall_check").level, Some(Level::Bad));
        assert_eq!(line(TEMPERATURE_VERDICT).level, Some(Level::Good));
        assert_eq!(
            line("housing.space_claim_check").value,
            "Inside the space claim"
        );
        assert_eq!(line("housing.space_claim_check").level, Some(Level::Good));
        assert_eq!(
            line("housing.space_claim_check").marker,
            "",
            "Rust-only: no cell"
        );
        assert_eq!(line("model.end_effect_check").level, Some(Level::Good));
        assert!(lines.iter().all(|l| !l.greyed));
        for path in STORED_3D_ROWS {
            assert!(line(path).stored_3d);
            assert!(line(path).tooltip.contains(STORED_3D_LABEL));
        }
        assert_eq!(lines.iter().filter(|l| l.stored_3d).count(), 3);
    }

    #[test]
    fn out_of_range_end_effect_greys_the_pull_out_rows_without_badges() {
        let results = compute_all(&short_magnets());
        assert!(end_effect_out_of_range(&results).is_some_and(|f| f < 0.0));
        let lines = dashboard_lines(&results);
        for line in &lines {
            let derived = END_EFFECT_ROWS.contains(&line.path);
            assert_eq!(line.greyed, derived, "{}", line.path);
            if derived {
                assert_eq!(line.level, None, "{}", line.path);
                assert!(
                    line.tooltip.contains(END_EFFECT_OUT_OF_RANGE),
                    "{}",
                    line.path
                );
            }
        }
        let flag = lines
            .iter()
            .find(|l| l.path == "model.end_effect_check")
            .unwrap();
        assert_eq!(flag.level, Some(Level::Bad));
        assert_eq!(
            end_effect_out_of_range(&compute_all(&DesignInputs::default())),
            None
        );
    }

    #[test]
    fn a_design_past_the_space_claim_shows_the_overshoot_in_red() {
        let results = compute_all(&design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)));
        let lines = dashboard_lines(&results);
        let claim = lines
            .iter()
            .find(|l| l.path == "housing.space_claim_check")
            .unwrap();
        assert!(
            claim
                .value
                .starts_with("Exceeds the space claim: overall length "),
            "{}",
            claim.value
        );
        assert_eq!(claim.level, Some(Level::Bad));
    }

    #[test]
    fn the_result_tooltip_names_the_path_cell_and_notes() {
        let info = result_info("model.pullout_Nm").unwrap();
        let tooltip = result_tooltip("model.pullout_Nm", info, ResultNotes::default());
        assert!(
            tooltip.starts_with("Pull-out torque at operating temperature\n"),
            "{tooltip}"
        );
        assert!(
            tooltip.contains("\nmodel.pullout_Nm\nCalculator!C93"),
            "{tooltip}"
        );
        let rust_only = result_info("housing.space_claim_check").unwrap();
        assert!(
            result_tooltip(
                "housing.space_claim_check",
                rust_only,
                ResultNotes::default()
            )
            .contains("Rust-only result (no workbook cell)")
        );
        assert!(result_info("no.such").is_none());
        assert_eq!(
            result_info("gap_sweep[0].f_end").unwrap().cell.as_deref(),
            Some("Gap sweep!W6")
        );
    }

    #[test]
    fn badge_colours_follow_the_theme() {
        let dark = egui::Visuals::dark();
        let light = egui::Visuals::light();
        assert_eq!(Level::Bad.color(&dark), dark.error_fg_color);
        assert_eq!(Level::Caution.color(&light), light.warn_fg_color);
        assert_ne!(Level::Good.color(&dark), Level::Good.color(&light));
        assert_eq!(Level::Good.word(), "OK");
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::input_ui::{CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::test_support::{
        central_panel_frame, drawn_texts, key_tap, primary_button, select_all, text_rect,
    };

    const FACE_GAP: &str = "metal.face_gap_mm";
    const POLES: &str = "coupling.npole";
```

with:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::api::HEADLINE;
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::input_ui::{CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::test_support::{
        central_panel_frame, drawn_texts, key_tap, primary_button, select_all, text_rect,
    };
    use crate::headline;

    const FACE_GAP: &str = "metal.face_gap_mm";
    const POLES: &str = "coupling.npole";
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust

    /// The headline of `inputs`, as the panel displays it (value with unit).
    fn displayed_headline(inputs: &DesignInputs) -> Vec<String> {
        let panel = MagcouplingPanel::new();
        headline(&compute_all(inputs))
            .into_iter()
            .zip(panel.headline_meta)
            .map(|((_, value), meta)| with_unit(format_value(&value), meta.unit))
            .collect()
    }

```

with:

```rust

    /// The headline of `inputs`, as the panel displays it (value with unit).
    fn displayed_headline(inputs: &DesignInputs) -> Vec<String> {
        headline(&compute_all(inputs))
            .into_iter()
            .zip(HEADLINE)
            .map(|((_, value), (_, path))| {
                with_unit(format_value(&value), result_info(path).unwrap().meta.unit)
            })
            .collect()
    }

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        let output = harness.frame(Vec::new());
        assert_drew_headline(&output, &DesignInputs::default());
        let texts = drawn_texts(&output);
        for meta in harness.panel.headline_meta {
            assert!(
                texts.iter().any(|t| t == meta.label),
                "missing {:?}",
                meta.label
            );
        }
        for entry in &InputCatalogue::get().key_design {
            assert!(
```

with:

```rust
        let output = harness.frame(Vec::new());
        assert_drew_headline(&output, &DesignInputs::default());
        let texts = drawn_texts(&output);
        for (_, path) in HEADLINE {
            let label = result_info(path).unwrap().meta.label;
            assert!(texts.iter().any(|t| t == label), "missing {label:?}");
        }
        for entry in &InputCatalogue::get().key_design {
            assert!(
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    }

    #[test]
    fn reset_all_restores_the_default_design_and_headline() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
```

with:

```rust
    }

    #[test]
    fn the_dashboard_shows_the_stored_3d_label_and_the_corrected_markers() {
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, STORED_3D_LABEL), 1);
        // The pull-out's marker: E3 changes it at the defaults, E7 and E8 have probes on it.
        assert!(count(&output, "E3 E7 E8") >= 1);
        assert_eq!(count(&output, "Inside the space claim"), 1);
        assert!(
            !drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with(END_EFFECT_BANNER))
        );
    }

    #[test]
    fn an_out_of_range_end_effect_shows_the_banner() {
        let mut harness = Harness::new();
        let inputs = &mut harness.panel.inputs;
        inputs.coupling.c_end = 0.5;
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        inputs.coupling.magnets.manual_inner_length_mm = 2.0;
        inputs.coupling.magnets.manual_outer_length_mm = 2.0;
        let output = harness.frame(Vec::new());
        let banner: Vec<String> = drawn_texts(&output)
            .into_iter()
            .filter(|t| t.starts_with(&format!("{END_EFFECT_BANNER} (")))
            .collect();
        assert_eq!(
            banner,
            [
                "End-effect model out of range (f_end = -1.202): the pull-out and the numbers computed from it are greyed."
            ]
        );
    }

    #[test]
    fn a_design_past_the_space_claim_shows_its_overshoot() {
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.magnets.axial_length_mm = Some(50.8);
        let output = harness.frame(Vec::new());
        let claim = drawn_texts(&output)
            .into_iter()
            .find(|t| t.starts_with("Exceeds the space claim:"))
            .expect("the badge text");
        assert!(claim.contains("overall length"), "{claim}");
    }

    #[test]
    fn reset_all_restores_the_default_design_and_headline() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
            // A window sizes itself on its first frame and paints on the next.
            output = Some(ctx.run(egui::RawInput::default(), |ctx| {
                egui::Window::new("Magnetic coupling")
                    .default_size([1100.0, 700.0])
                    .show(ctx, |ui| panel.ui(ui));
            }));
        }
```

with:

```rust
            // A window sizes itself on its first frame and paints on the next.
            output = Some(ctx.run(egui::RawInput::default(), |ctx| {
                egui::Window::new("Magnetic coupling")
                    .default_size([1100.0, 1200.0])
                    .show(ctx, |ui| panel.ui(ui));
            }));
        }
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs`, replace:

```rust
//! window and event loop. The engine stays pure std; nothing here is compiled
//! without the feature.

mod format;
pub mod history;
pub mod input_ui;
```

with:

```rust
//! window and event loop. The engine stays pure std; nothing here is compiled
//! without the feature.

pub mod corrections;
pub mod dashboard;
mod format;
pub mod history;
pub mod input_ui;
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | sort -u | head -8
```

Expected: the build fails with compile errors naming the items Step 3 adds, among them ``error[E0432]: unresolved imports `crate::gui::dashboard::END_EFFECT_BANNER`, `crate::gui::dashboard::STORED_3D_LABEL`, `crate::gui::dashboard::result_info` ``, ``error[E0433]: failed to resolve: use of undeclared type `Level` `` and ``error[E0433]: failed to resolve: use of undeclared type `CorrectionIndex` ``.

- [ ] **Step 3: Write the correction index and the dashboard**



In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/corrections.rs`, replace:

```rust
//! Only applied engine corrections mark (E14 rewords help only). [`CorrectionIndex`] is built
//! once, keyed by workbook cell; a value with no cell (a Rust-only result) has no marker. Plan
//! A-3's equation registry may later supply each record's upstream corrections instead.

#[cfg(test)]
mod tests {
```

with:

```rust
//! Only applied engine corrections mark (E14 rewords help only). [`CorrectionIndex`] is built
//! once, keyed by workbook cell; a value with no cell (a Rust-only result) has no marker. Plan
//! A-3's equation registry may later supply each record's upstream corrections instead.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde_json::Value as Json;

use crate::engine::deviations::{
    Deviation, DeviationClass, DeviationId, DeviationStatus, REGISTRY,
};
use crate::engine::meta::Value;
use crate::gui::format::format_value;

/// The golden files of the broad corrections (decision D4), compiled in: the registry names
/// each in its `changes_file` (a test checks the two lists agree).
pub const GOLDEN_FILES: [(DeviationId, &str, &str); 3] = [
    (
        DeviationId::E3,
        "tests/data/deviations/E3.json",
        include_str!("../../tests/data/deviations/E3.json"),
    ),
    (
        DeviationId::E4,
        "tests/data/deviations/E4.json",
        include_str!("../../tests/data/deviations/E4.json"),
    ),
    (
        DeviationId::E5,
        "tests/data/deviations/E5.json",
        include_str!("../../tests/data/deviations/E5.json"),
    ),
];

/// How the registry ties a cell to a correction.
#[derive(Clone, Debug, PartialEq)]
pub enum Tie {
    /// The correction changes the cell at the default design: the workbook's value and the
    /// corrected one.
    AtDefaults { workbook: Value, corrected: Value },
    /// The report names the cell for the correction, or a probe shows it changing off the
    /// default design.
    Named,
}

/// One correction's marker on a cell.
#[derive(Clone, Debug, PartialEq)]
pub struct Mark {
    pub id: DeviationId,
    pub tie: Tie,
}

/// The markers of every workbook cell the registry ties to an applied engine correction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CorrectionIndex {
    by_cell: HashMap<String, Vec<Mark>>,
}

/// A golden file's value as an engine value (the other way from `session::json_value`): a
/// number or a string; anything else as none.
fn value_from_json(json: &Json) -> Value {
    match json {
        Json::Number(n) => n.as_f64().map_or(Value::None, Value::Num),
        Json::String(text) => Value::Text(text.clone()),
        _ => Value::None,
    }
}

impl CorrectionIndex {
    /// The index of [`REGISTRY`] and [`GOLDEN_FILES`].
    pub fn build() -> Self {
        let mut index = Self::default();
        for deviation in REGISTRY.iter().filter(|d| marks_values(d)) {
            let id = deviation.id;
            for change in deviation.changes_at_defaults {
                index.add(
                    change.cell,
                    id,
                    Tie::AtDefaults {
                        workbook: change.workbook.to_value(),
                        corrected: change.corrected.to_value(),
                    },
                );
            }
            if let Some((_, _, text)) = GOLDEN_FILES.iter().find(|(g, _, _)| *g == id) {
                let json: Json = serde_json::from_str(text).expect("a reviewed golden file");
                let changes = json["changes"].as_object().expect("a changes map");
                for (cell, pair) in changes {
                    index.add(
                        cell,
                        id,
                        Tie::AtDefaults {
                            workbook: value_from_json(&pair[0]),
                            corrected: value_from_json(&pair[1]),
                        },
                    );
                }
            }
            for cell in deviation.cells {
                index.add(cell, id, Tie::Named);
            }
            for probe in deviation.probes {
                for change in probe.expect {
                    index.add(change.cell, id, Tie::Named);
                }
            }
        }
        for marks in index.by_cell.values_mut() {
            marks.sort_by_key(|mark| mark.id.index());
        }
        index
    }

    /// The index, built once.
    pub fn get() -> &'static CorrectionIndex {
        static INDEX: OnceLock<CorrectionIndex> = OnceLock::new();
        INDEX.get_or_init(CorrectionIndex::build)
    }

    /// The markers of a workbook cell, by correction in report order; empty for a cell no
    /// correction touches or for no cell.
    pub fn marks(&self, cell: Option<&str>) -> &[Mark] {
        cell.and_then(|cell| self.by_cell.get(cell))
            .map_or(&[], Vec::as_slice)
    }

    /// Records a tie, once per correction and cell: the at-defaults values win over a name.
    fn add(&mut self, cell: &str, id: DeviationId, tie: Tie) {
        let marks = self.by_cell.entry(cell.to_owned()).or_default();
        match marks.iter_mut().find(|mark| mark.id == id) {
            Some(mark) => {
                if mark.tie == Tie::Named {
                    mark.tie = tie;
                }
            }
            None => marks.push(Mark { id, tie }),
        }
    }
}

/// Whether a correction marks values: applied and changing what the engine computes.
fn marks_values(deviation: &Deviation) -> bool {
    deviation.status == DeviationStatus::Applied && deviation.class == DeviationClass::Engine
}

/// The marker text beside a value: the corrections' ids (`E3 E7`); empty for none.
pub fn marker_text(marks: &[Mark]) -> String {
    marks
        .iter()
        .map(|mark| mark.id.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The tooltip lines of the markers: per correction its id and title, the workbook and
/// corrected values where it changes the cell at the default design, and its evidence.
pub fn marker_tooltip(marks: &[Mark]) -> String {
    let mut lines = vec!["Corrected vs workbook:".to_owned()];
    for mark in marks {
        let deviation = &REGISTRY[mark.id.index()];
        lines.push(format!("{}: {}", mark.id, deviation.title));
        if let Tie::AtDefaults {
            workbook,
            corrected,
        } = &mark.tie
        {
            lines.push(format!(
                "  at the default design: workbook {}, corrected {}",
                format_value(workbook),
                format_value(corrected)
            ));
        }
        lines.push(format!("  {}", deviation.evidence()));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/dashboard.rs`, replace:

```rust
//! [`result_tooltip`] is the hover text of every displayed result, keyed by result path: the
//! one hook the dashboard and the results table (and later the geometry callouts and the
//! equation explorer) go through.

#[cfg(test)]
mod tests {
```

with:

```rust
//! [`result_tooltip`] is the hover text of every displayed result, keyed by result path: the
//! one hook the dashboard and the results table (and later the geometry callouts and the
//! equation explorer) go through.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::engine::meta::{ResultMeta, ResultSet, Value, result_rows};
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
use crate::gui::format::{format_value, with_unit};
use crate::{DesignInputs, DesignResults, compute_all};

/// A badge colour, from a check verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Green: the check passes.
    Good,
    /// Amber: the check asks for a look (a temperature "CHECK", an unknown space claim).
    Caution,
    /// Red: the check fails.
    Bad,
}

impl Level {
    /// The word of the badge's hover text.
    pub const fn word(self) -> &'static str {
        match self {
            Level::Good => "OK",
            Level::Caution => "Check",
            Level::Bad => "Fails",
        }
    }

    /// The badge colour: green, the theme's warning colour, the theme's error colour.
    pub fn color(self, visuals: &egui::Visuals) -> egui::Color32 {
        match self {
            Level::Good if visuals.dark_mode => egui::Color32::from_rgb(90, 200, 120),
            Level::Good => egui::Color32::from_rgb(20, 130, 60),
            Level::Caution => visuals.warn_fg_color,
            Level::Bad => visuals.error_fg_color,
        }
    }
}

/// The temperature summary's verdict, the badge of the three temperature rows.
const TEMPERATURE_VERDICT: &str = "temperature.summary.verdict";

/// The dashboard rows: (result path, the check whose verdict gives its badge). The first 15
/// are `HEADLINE`, in order (a test checks).
pub const DASHBOARD: [(&str, Option<&str>); 17] = [
    ("model.pullout_Nm", None),
    ("model.pullout_20C_Nm", None),
    ("metal.torque_hot_low_Nm", Some("metal.hot_min_check")),
    ("metal.hot_min_check", Some("metal.hot_min_check")),
    ("metal.torque_cold_high_Nm", None),
    ("model.gearbox_input_ripple_Nm", None),
    ("model.cup_od_mm", None),
    ("mass.total_g", None),
    (
        "metal.min_running_clearance_mm",
        Some("metal.clearance_check"),
    ),
    ("metal.clearance_check", Some("metal.clearance_check")),
    ("materials.cup_wall_check", Some("materials.cup_wall_check")),
    (
        "temperature.summary.governing_limit_C",
        Some(TEMPERATURE_VERDICT),
    ),
    (
        "temperature.summary.margin_hot_day_C",
        Some(TEMPERATURE_VERDICT),
    ),
    (TEMPERATURE_VERDICT, Some(TEMPERATURE_VERDICT)),
    ("clamps.recommended", Some("clamps.recommended")),
    (
        "housing.space_claim_check",
        Some("housing.space_claim_check"),
    ),
    ("model.end_effect_check", Some("model.end_effect_check")),
];

/// The dashboard rows computed from the pull-out, greyed when f_end ≤ 0 (audit M9): the
/// headline rows that move with the end-effect coefficient (a test probes it at back iron 1
/// and 0). Per-row greying of the results table waits for plan A-3's dependency graph; the
/// table shows the banner instead (decision M41-11).
pub const END_EFFECT_ROWS: [&str; 7] = [
    "model.pullout_Nm",
    "model.pullout_20C_Nm",
    "metal.torque_hot_low_Nm",
    "metal.hot_min_check",
    "metal.torque_cold_high_Nm",
    "model.gearbox_input_ripple_Nm",
    "clamps.recommended",
];

/// The dashboard rows that read the workbook's stored 3D fields (the reverse fields of the
/// demagnetization check and the slip-loss fields): the headline rows that move with them (a
/// test probes each at its slider ends).
pub const STORED_3D_ROWS: [&str; 3] = [
    "temperature.summary.governing_limit_C",
    "temperature.summary.margin_hot_day_C",
    TEMPERATURE_VERDICT,
];

/// The label of the stored-3D rows.
pub const STORED_3D_LABEL: &str = "3D values from the workbook";

/// Its hover text.
pub const STORED_3D_NOTE: &str = "The demagnetization reverse fields and the slip-loss fields are the workbook's stored 3D values: they do not follow geometry changes until M3 computes them live.";

/// The start of the banner shown when f_end ≤ 0.
pub const END_EFFECT_BANNER: &str = END_EFFECT_OUT_OF_RANGE;

/// The badge level of a check's verdict text; `None` for a path that is no check or a text the
/// check does not produce.
pub fn verdict_level(path: &str, text: &str) -> Option<Level> {
    use Level::{Bad, Caution, Good};
    match (path, text) {
        ("metal.hot_min_check", "Estimate covers hot min") => Some(Good),
        ("metal.hot_min_check", "Below hot minimum") => Some(Bad),
        ("metal.clearance_check", "Meets assumed target") => Some(Good),
        ("metal.clearance_check", "Below target") => Some(Bad),
        ("materials.cup_wall_check", "OK") => Some(Good),
        ("materials.cup_wall_check", t) if t.starts_with("Too thin: ") => Some(Bad),
        (TEMPERATURE_VERDICT, t) if t.starts_with("OK on temperature.") => Some(Good),
        (TEMPERATURE_VERDICT, t) if t.starts_with("CHECK:") => Some(Caution),
        ("clamps.recommended", t) if t.starts_with("ISO 4762 ") => Some(Good),
        ("clamps.recommended", t) if t.starts_with("None:") => Some(Bad),
        ("housing.space_claim_check", crate::engine::housing::INSIDE_THE_SPACE_CLAIM) => Some(Good),
        ("housing.space_claim_check", crate::engine::housing::SPACE_CLAIM_UNKNOWN) => Some(Caution),
        ("housing.space_claim_check", t) if t.starts_with("Exceeds the space claim:") => Some(Bad),
        ("model.end_effect_check", "OK") => Some(Good),
        ("model.end_effect_check", END_EFFECT_OUT_OF_RANGE) => Some(Bad),
        _ => None,
    }
}

/// A result's metadata and workbook cell.
#[derive(Clone, Debug)]
pub struct ResultInfo {
    pub meta: &'static ResultMeta,
    pub cell: Option<String>,
}

/// Every result's metadata and cell by path, built once (the result layout does not depend
/// on the inputs: every table has a fixed number of rows).
pub fn result_info(path: &str) -> Option<&'static ResultInfo> {
    static INDEX: OnceLock<HashMap<String, ResultInfo>> = OnceLock::new();
    INDEX
        .get_or_init(|| {
            result_rows(&compute_all(&DesignInputs::default()))
                .into_iter()
                .map(|row| {
                    let info = ResultInfo {
                        meta: row.meta,
                        cell: row.cell,
                    };
                    (row.path, info)
                })
                .collect()
        })
        .get(path)
}

/// What a readout knows about its value beyond the metadata.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResultNotes<'a> {
    /// The corrections the registry ties to its cell.
    pub marks: &'a [Mark],
    /// Greyed: computed from the pull-out while f_end ≤ 0.
    pub greyed: bool,
    /// Reads the workbook's stored 3D fields.
    pub stored_3d: bool,
}

/// The hover text of a displayed result: label, help, path, workbook cell, then the notes.
/// The hook every readout goes through (plan M4-3 adds the equation here).
pub fn result_tooltip(path: &str, info: &ResultInfo, notes: ResultNotes<'_>) -> String {
    let meta = info.meta;
    let mut lines = vec![meta.label.to_owned()];
    if !meta.help.is_empty() {
        lines.push(meta.help.to_owned());
    }
    lines.push(path.to_owned());
    lines.push(
        info.cell
            .clone()
            .unwrap_or_else(|| "Rust-only result (no workbook cell)".to_owned()),
    );
    if notes.greyed {
        lines.push(format!(
            "{END_EFFECT_OUT_OF_RANGE}: computed from the pull-out, which is not valid while f_end <= 0."
        ));
    }
    if notes.stored_3d {
        lines.push(format!("{STORED_3D_LABEL}: {STORED_3D_NOTE}"));
    }
    if !notes.marks.is_empty() {
        lines.push(marker_tooltip(notes.marks));
    }
    lines.join("\n")
}

/// One dashboard row, ready to draw.
#[derive(Clone, Debug, PartialEq)]
pub struct DashboardLine {
    pub path: &'static str,
    pub label: &'static str,
    /// The value with its unit.
    pub value: String,
    /// The badge; none when the row has no check or is greyed.
    pub level: Option<Level>,
    pub greyed: bool,
    pub stored_3d: bool,
    /// The corrections' marker text (`E3 E7 E8`), empty for none.
    pub marker: String,
    pub tooltip: String,
}

/// The f_end of the design when it is out of the end-effect model's range, else `None`.
pub fn end_effect_out_of_range(results: &DesignResults) -> Option<f64> {
    (results.model.end_effect_check == END_EFFECT_OUT_OF_RANGE).then_some(results.model.f_end)
}

/// The dashboard rows of `results`.
pub fn dashboard_lines(results: &DesignResults) -> Vec<DashboardLine> {
    let out_of_range = end_effect_out_of_range(results).is_some();
    DASHBOARD
        .iter()
        .map(|&(path, badge)| {
            let info = result_info(path).unwrap_or_else(|| panic!("DASHBOARD: no result {path}"));
            let value = results.get(path).unwrap_or(Value::None);
            let greyed = out_of_range && END_EFFECT_ROWS.contains(&path);
            let stored_3d = STORED_3D_ROWS.contains(&path);
            let level = match (
                greyed,
                badge.and_then(|check| results.get(check).map(|v| (check, v))),
            ) {
                (false, Some((check, Value::Text(text)))) => verdict_level(check, &text),
                _ => None,
            };
            let marks = CorrectionIndex::get().marks(info.cell.as_deref());
            DashboardLine {
                path,
                label: info.meta.label,
                value: with_unit(format_value(&value), info.meta.unit),
                level,
                greyed,
                stored_3d,
                marker: marker_text(marks),
                tooltip: result_tooltip(
                    path,
                    info,
                    ResultNotes {
                        marks,
                        greyed,
                        stored_3d,
                    },
                ),
            }
        })
        .collect()
}

/// Draws the dashboard: the end-effect banner when f_end <= 0, then a row per line (badge
/// and label, then the value and the marker under them, so a narrow side still reads), the
/// stored-3D label after the last temperature row.
pub fn dashboard_ui(ui: &mut egui::Ui, results: &DesignResults) {
    let lines = dashboard_lines(results);
    if let Some(f_end) = end_effect_out_of_range(results) {
        ui.colored_label(
            ui.visuals().error_fg_color,
            format!(
                "{END_EFFECT_BANNER} (f_end = {}): the pull-out and the numbers computed from it are greyed.",
                format_value(&Value::Num(f_end))
            ),
        );
        ui.separator();
    }
    let last_3d = lines.iter().rposition(|line| line.stored_3d);
    for (index, line) in lines.iter().enumerate() {
        let weak = ui.visuals().weak_text_color();
        let tint = |rich: egui::RichText| if line.greyed { rich.color(weak) } else { rich };
        ui.horizontal(|ui| {
            badge(ui, line.level);
            ui.add(egui::Label::new(tint(egui::RichText::new(line.label))).wrap())
                .on_hover_text(&line.tooltip);
        });
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.add(egui::Label::new(tint(egui::RichText::new(&line.value).strong())).wrap())
                .on_hover_text(&line.tooltip);
            if !line.marker.is_empty() {
                ui.small(&line.marker).on_hover_text(&line.tooltip);
            }
        });
        if Some(index) == last_3d {
            ui.horizontal(|ui| {
                ui.add_space(16.0);
                ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
            });
        }
        ui.add_space(4.0);
    }
}

/// A badge: a filled circle in the level's colour, or an empty cell.
fn badge(ui: &mut egui::Ui, level: Option<Level>) {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    if let Some(level) = level {
        ui.painter()
            .circle_filled(rect.center(), 5.0, level.color(ui.visuals()));
        response.on_hover_text(level.word());
    }
}

#[cfg(test)]
mod tests {
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
//! on ([`compute_all`]) each frame after the inputs are drawn, so the readouts show this
//! frame's edits.

use crate::engine::api::HEADLINE;
use crate::engine::meta::{InputSet, ResultMeta, ResultSet, Value, result_rows};
use crate::gui::format::{format_value, with_unit};
use crate::gui::input_ui::{RowEdit, input_row};
use crate::gui::inputs::{InputCatalogue, InputEntry, optional_seed};
use crate::{DesignInputs, DesignResults, compute_all, headline};

/// The heading of the panel.
pub const HEADING: &str = "Magnetic coupling calculator";
```

with:

```rust
//! on ([`compute_all`]) each frame after the inputs are drawn, so the readouts show this
//! frame's edits.

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::gui::dashboard::dashboard_ui;
use crate::gui::input_ui::{RowEdit, input_row};
use crate::gui::inputs::{InputCatalogue, InputEntry, optional_seed};
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
pub const HEADING: &str = "Magnetic coupling calculator";
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    inputs: DesignInputs,
    /// The results of `inputs`, recomputed by every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// Metadata of each [`HEADLINE`] result, in order.
    headline_meta: [&'static ResultMeta; HEADLINE.len()],
    /// The main widget of each Key design row in the last frame, by input path.
    key_widgets: Vec<(&'static str, egui::Id)>,
    /// Why the last edit was refused, until the next accepted edit or reset.
```

with:

```rust
    inputs: DesignInputs,
    /// The results of `inputs`, recomputed by every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// The main widget of each Key design row in the last frame, by input path.
    key_widgets: Vec<(&'static str, egui::Id)>,
    /// Why the last edit was refused, until the next accepted edit or reset.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    pub fn new() -> Self {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let result_meta = result_rows(&results);
        let headline_meta = HEADLINE.map(|(key, path)| {
            let row = result_meta.iter().find(|row| row.path == path);
            row.unwrap_or_else(|| panic!("HEADLINE {key}: no result {path}"))
                .meta
        });
        Self {
            inputs,
            results,
            headline_meta,
            key_widgets: Vec::new(),
            last_error: None,
        }
```

with:

```rust
    pub fn new() -> Self {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        Self {
            inputs,
            results,
            key_widgets: Vec::new(),
            last_error: None,
        }
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
                .show_inside(ui, |ui| self.headline_ui(ui));
            egui::CentralPanel::default().show_inside(ui, |_ui| {});
        });
    }
```

with:

```rust
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| dashboard_ui(ui, &self.results));
                });
            egui::CentralPanel::default().show_inside(ui, |_ui| {});
        });
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
            Value::Num(x) if x.is_finite() => Some(x.clamp(range.min, range.max)),
            _ => None,
        }
    }

    /// The headline numbers: label, then value with unit; the hover shows the
    /// Python key, the result path and the workbook cell.
    fn headline_ui(&self, ui: &mut egui::Ui) {
        egui::Grid::new("magcoupling_headline")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                let rows = headline(&self.results)
                    .into_iter()
                    .zip(HEADLINE)
                    .zip(self.headline_meta);
                for (((key, value), (_, path)), meta) in rows {
                    ui.label(meta.label).on_hover_text(format!(
                        "{key}\n{path}\n{}",
                        meta.cell.unwrap_or("no workbook cell")
                    ));
                    ui.label(with_unit(format_value(&value), meta.unit));
                    ui.end_row();
                }
            });
    }
}

```

with:

```rust
            Value::Num(x) if x.is_finite() => Some(x.clamp(range.min, range.max)),
            _ => None,
        }
    }
}

```

- [ ] **Step 4: Run the tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED|panicked"
```

Expected: `test result: ok. 222 passed; 0 failed` (Task 4's 205 and 17 new tests).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task5.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task5.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task5.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task5.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/corrections.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/dashboard.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
feat(magcoupling-rs): the dashboard: badges, corrected markers, end-effect greying, stored-3D label

The headline, the space claim and the end-effect flag with green/amber/red
badges from the check verdicts; the corrected-vs-workbook markers from the
deviation registry and the compiled golden files (decision M41-10); the audit
M9 greying of the pull-out rows under its banner (M41-11) and the "3D values
from the workbook" label (M3 after M4), each row list pinned by a probe of the
engine; result_tooltip, the one hover hook of every readout.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 6: The results table and its export

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

Spec M4 "Layout": "Results table: every computed value with label, unit, cell; searchable; CSV and JSON export". The table's rows are built once (the result layout does not depend on the inputs) and a frame reads only the values of the rows on screen (`show_rows`), each row's hover text built only while hovered. The search matches label, path or cell, ignoring case. The exports write every result at full precision in schema order: CSV `path,label,value,unit,cell` (RFC 4180 quoting) and JSON with the design that produced it; a non-finite result is `+inf`, `-inf` or `NaN` in both (decision M41-15; `exact_number` takes the text from Task 1's `format::non_finite_text`). The panel cannot save files (feature `gui` does no I/O), so the export buttons queue a `PanelRequest::SaveFile` that the host drains with `take_requests` (Task 9's app saves it). The centre region becomes `CentreView` (one tab now; M4-2 adds the geometry view and the plots). The end-effect banner shows over the table too (M41-11), from `dashboard::end_effect_banner`.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/results_table.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/dashboard.rs` (`end_effect_banner`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/session.rs` (`design_json` becomes `pub(crate)`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs` (`PanelRequest`, `CentreView`, the centre region, `take_requests`; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/test_support.rs` (`SCREEN`, `sized_frame` replacing `central_panel_frame`, `short_magnets`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs` (`pub mod results_table;`, the new re-exports)

**Interfaces:**
- Consumes: Task 1's `Design`, `json_value` (and `design_json`, now `pub(crate)`), `format::non_finite_text`; Task 5's `result_info`, `ResultInfo`, `ResultNotes`, `result_tooltip`, `CorrectionIndex`, `marker_text`.
- Produces (`magcoupling::gui::results_table`): `RESULTS_FORMAT`, `RESULTS_VERSION: u64 = 1`, `CSV_FILE_NAME = "magcoupling-results.csv"`, `JSON_FILE_NAME = "magcoupling-results.json"`, `CSV_HEADER`, `EXPORT_CSV`, `EXPORT_JSON`, `SEARCH_HINT`; `struct TableEntry { pub path, pub info, pub marker, .. }`; `fn table_entries() -> &'static [TableEntry]`; `fn search(&[TableEntry], &str) -> Vec<usize>`; `fn exact_number(f64) -> String`; `fn results_csv(&DesignResults) -> String`; `fn results_json(&Design, &DesignResults) -> String`; `struct ResultsTable` with `query()` and `fn ui(&mut self, &mut egui::Ui, &DesignResults) -> Option<TableAction>`; `enum TableAction { ExportCsv, ExportJson }`; `fn row_tooltip(&TableEntry, &Value) -> String`.
- Produces (`magcoupling::gui::dashboard`): `fn end_effect_banner(&DesignResults) -> Option<String>`.
- Produces (`magcoupling::gui`, re-exported from `panel`): `enum PanelRequest { SaveFile { file_name: String, mime: &'static str, contents: String } }` (Task 7 adds `OpenDesign`); `enum CentreView { Results }` with `ALL` and `label()`; `MagcouplingPanel::take_requests(&mut self) -> Vec<PanelRequest>`.
- Produces (`gui::test_support`, tests only): `const SCREEN: egui::Vec2` (1280 x 1024), `fn sized_frame(ctx, size, events, draw) -> egui::FullOutput` (replaces `central_panel_frame`), `fn short_magnets() -> DesignInputs` (f_end -1.202).

- [ ] **Step 1: Write the failing tests**

The table module's docs and tests (one row per result in schema order with its marker; the search by cell, path and label; exact numbers reading back bit for bit; the CSV lines and quoting; `+inf` and `NaN` in CSV and JSON, from E20's positive-beta design; the JSON export with the design; the row tooltip with the exact value), the panel's tests (the table's first and last rows and the search through the box; the export buttons queueing their files; the banner over the dashboard and the table), and the shared `short_magnets` design moving to the test helpers; the panel tests now run on a 1280 x 1024 screen (`sized_frame`), so the idle-frames test takes a tall one to draw every row.

Create `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/results_table.rs`:

```rust
//! The results table (spec M4 "Layout": "every computed value with label, unit, cell;
//! searchable; CSV and JSON export").
//!
//! Every result, scalars and table rows, in the Python schema order ([`result_rows`]). The
//! layout of the results does not depend on the inputs, so the rows (path, metadata, cell,
//! marker, search text) are built once; a frame reads only the values of the rows on screen.
//! The exports write every result at full precision: CSV for spreadsheets, JSON with the
//! design that produced it. JSON has no infinity or NaN, so both write a non-finite number as
//! `+inf`, `-inf` or `NaN` (decision M41-15).

#[cfg(test)]
mod tests {
    use super::*;

    /// A positive beta typed in for manual magnets with no grade (E20): no hot limit, +inf, and
    /// no torque at it, NaN (`tests/robustness.rs`).
    fn non_finite_design() -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner = String::new();
        inputs.coupling.magnets.part_outer = String::new();
        inputs.temperature.demag.coercivity_source = 0;
        inputs.temperature.demag.beta_hcj_per_C = 0.0035;
        inputs
    }

    #[test]
    fn the_table_lists_every_result_once_in_schema_order() {
        let entries = table_entries();
        let rows = result_rows(&compute_all(&DesignInputs::default()));
        assert_eq!(entries.len(), rows.len());
        for (entry, row) in entries.iter().zip(&rows) {
            assert_eq!(entry.path, row.path);
            assert_eq!(entry.info.cell, row.cell);
        }
        let pullout = entries
            .iter()
            .find(|e| e.path == "model.pullout_Nm")
            .unwrap();
        assert_eq!(pullout.marker, "E3 E7 E8");
    }

    #[test]
    fn the_search_matches_label_path_and_cell_ignoring_case() {
        let entries = table_entries();
        let paths = |query: &str| -> Vec<&str> {
            search(entries, query)
                .into_iter()
                .map(|i| entries[i].path.as_str())
                .collect()
        };
        assert_eq!(paths("calculator!c93"), ["model.pullout_Nm"]);
        assert_eq!(paths("  MODEL.PULLOUT_20C_NM "), ["model.pullout_20C_Nm"]);
        assert!(paths("pull-out torque").contains(&"model.pullout_Nm"));
        let row3: Vec<&str> = paths("gap_sweep[3].");
        let columns = result_rows(&compute_all(&DesignInputs::default()))
            .iter()
            .filter(|r| r.path.starts_with("gap_sweep[3]."))
            .count();
        assert_eq!(row3.len(), columns);
        assert_eq!(search(entries, "").len(), entries.len());
        assert_eq!(search(entries, "   ").len(), entries.len());
        assert!(search(entries, "no such result anywhere").is_empty());
    }

    #[test]
    fn a_row_tooltip_is_the_hover_hook_s_with_the_exact_value() {
        let entries = table_entries();
        let pullout = entries
            .iter()
            .find(|e| e.path == "model.pullout_Nm")
            .unwrap();
        let value = Value::Num(2.6884762950539796);
        let tooltip = row_tooltip(pullout, &value);
        assert!(
            tooltip.starts_with("Pull-out torque at operating temperature\n"),
            "{tooltip}"
        );
        assert!(tooltip.contains("Corrected vs workbook:"), "{tooltip}");
        assert!(
            tooltip.ends_with("\nExact value: 2.6884762950539796"),
            "{tooltip}"
        );
        let claim = entries
            .iter()
            .find(|e| e.path == "housing.space_claim_check")
            .unwrap();
        let text = row_tooltip(claim, &Value::Text("Inside the space claim".to_owned()));
        assert!(!text.contains("Exact value"), "{text}");
    }

    #[test]
    fn exact_numbers_read_back_bit_for_bit() {
        for x in [
            2.6884762950539796,
            5.27e-10,
            -0.1031743945660739,
            50.0,
            1e300,
            0.0,
        ] {
            assert_eq!(exact_number(x).parse::<f64>().unwrap(), x, "{x}");
        }
        assert_eq!(exact_number(5.27e-10), "5.27e-10");
        assert_eq!(exact_number(f64::INFINITY), "+inf");
        assert_eq!(exact_number(f64::NEG_INFINITY), "-inf");
        assert_eq!(exact_number(f64::NAN), "NaN");
    }

    #[test]
    fn the_csv_has_a_line_per_result_with_quoted_text() {
        let results = compute_all(&DesignInputs::default());
        let csv = results_csv(&results);
        let lines: Vec<&str> = csv.split("\r\n").collect();
        assert_eq!(lines[0], CSV_HEADER);
        assert_eq!(
            lines.len(),
            result_rows(&results).len() + 2,
            "header, rows, final break"
        );
        assert_eq!(lines.last(), Some(&""));
        let pullout = lines
            .iter()
            .find(|l| l.starts_with("model.pullout_Nm,"))
            .unwrap();
        assert_eq!(
            *pullout,
            format!(
                "model.pullout_Nm,Pull-out torque at operating temperature,{},N·m,Calculator!C93",
                exact_number(results.model.pullout_Nm)
            )
        );
        let mass = lines
            .iter()
            .find(|l| l.starts_with("mass.total_g,"))
            .unwrap();
        assert!(
            mass.starts_with("mass.total_g,\"Preliminary rotating mass, including retainers\","),
            "{mass}"
        );
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(csv_field("a\nb"), "\"a\nb\"");
        // A Rust-only result has an empty cell field.
        let claim = lines
            .iter()
            .find(|l| l.starts_with("housing.space_claim_check,"))
            .unwrap();
        assert!(claim.ends_with(",Inside the space claim,,"), "{claim}");
    }

    #[test]
    fn non_finite_results_export_as_text_in_csv_and_json() {
        let design = Design {
            inputs: non_finite_design(),
            ..Design::default()
        };
        let results = compute_all(&design.inputs);
        assert_eq!(results.temperature.demag.magnet_limit_C, f64::INFINITY);
        assert!(results.temperature.demag.torque_at_limit_Nm.is_nan());
        let csv = results_csv(&results);
        let line = |path: &str| {
            csv.split("\r\n")
                .find(|l| l.starts_with(&format!("{path},")))
                .unwrap()
                .to_owned()
        };
        assert!(line("temperature.demag.magnet_limit_C").contains(",+inf,"));
        assert!(line("temperature.demag.torque_at_limit_Nm").contains(",NaN,"));
        let json: Json = serde_json::from_str(&results_json(&design, &results)).unwrap();
        let value = |path: &str| {
            json["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["path"] == path)
                .unwrap()["value"]
                .clone()
        };
        assert_eq!(
            value("temperature.demag.magnet_limit_C"),
            Json::from("+inf")
        );
        assert_eq!(
            value("temperature.demag.torque_at_limit_Nm"),
            Json::from("NaN")
        );
    }

    #[test]
    fn the_json_export_holds_the_design_and_every_result() {
        let design = Design::default();
        let results = compute_all(&design.inputs);
        let json: Json = serde_json::from_str(&results_json(&design, &results)).unwrap();
        assert_eq!(json["format"], RESULTS_FORMAT);
        assert_eq!(json["version"], RESULTS_VERSION);
        assert_eq!(json["design"], design_json(&design));
        let rows = json["results"].as_array().unwrap();
        assert_eq!(rows.len(), result_rows(&results).len());
        let pullout = rows
            .iter()
            .find(|r| r["path"] == "model.pullout_Nm")
            .unwrap();
        assert_eq!(pullout["value"].as_f64(), Some(results.model.pullout_Nm));
        assert_eq!(pullout["unit"], "N·m");
        assert_eq!(pullout["cell"], "Calculator!C93");
        let claim = rows
            .iter()
            .find(|r| r["path"] == "housing.space_claim_check")
            .unwrap();
        assert_eq!(claim["cell"], Json::Null);
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/dashboard.rs`, replace:

```rust
    use super::*;
    use crate::engine::api::HEADLINE;
    use crate::engine::meta::InputSet;
    use std::collections::BTreeSet;

    #[test]
```

with:

```rust
    use super::*;
    use crate::engine::api::HEADLINE;
    use crate::engine::meta::InputSet;
    use crate::gui::test_support::short_magnets;
    use std::collections::BTreeSet;

    #[test]
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/dashboard.rs`, replace:

```rust
        let mut inputs = DesignInputs::default();
        edit(&mut inputs);
        inputs
    }

    /// A design with f_end below 0 (the engine's short-magnet test): 2 mm manual blocks, c_end 0.5.
    pub(crate) fn short_magnets() -> DesignInputs {
        design(|i| {
            i.coupling.c_end = 0.5;
            i.coupling.magnets.part_inner.clear();
            i.coupling.magnets.part_outer.clear();
            i.coupling.magnets.manual_inner_length_mm = 2.0;
            i.coupling.magnets.manual_outer_length_mm = 2.0;
        })
    }

    #[test]
```

with:

```rust
        let mut inputs = DesignInputs::default();
        edit(&mut inputs);
        inputs
    }

    #[test]
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::input_ui::{CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::test_support::{
        central_panel_frame, drawn_texts, key_tap, primary_button, select_all, text_rect,
    };
    use crate::headline;

```

with:

```rust
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::input_ui::{CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::test_support::{
        SCREEN, drawn_texts, key_tap, primary_button, select_all, short_magnets, sized_frame,
        text_rect,
    };
    use crate::headline;

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    pub(crate) struct Harness {
        pub(crate) ctx: egui::Context,
        pub(crate) panel: MagcouplingPanel,
    }

    impl Harness {
        pub(crate) fn new() -> Self {
            let mut harness = Self {
                ctx: egui::Context::default(),
                panel: MagcouplingPanel::new(),
            };
            harness.frame(Vec::new());
            harness
```

with:

```rust
    pub(crate) struct Harness {
        pub(crate) ctx: egui::Context,
        pub(crate) panel: MagcouplingPanel,
        screen: egui::Vec2,
    }

    impl Harness {
        pub(crate) fn new() -> Self {
            Self::on_screen(SCREEN)
        }

        /// A harness on a screen of `size`.
        pub(crate) fn on_screen(size: egui::Vec2) -> Self {
            let mut harness = Self {
                ctx: egui::Context::default(),
                panel: MagcouplingPanel::new(),
                screen: size,
            };
            harness.frame(Vec::new());
            harness
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust

        pub(crate) fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
            let panel = &mut self.panel;
            central_panel_frame(&self.ctx, events, |ui| panel.ui(ui))
        }

        /// The Key design row's main widget as drawn in the last frame.
```

with:

```rust

        pub(crate) fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
            let panel = &mut self.panel;
            sized_frame(&self.ctx, self.screen, events, |ui| panel.ui(ui))
        }

        /// The Key design row's main widget as drawn in the last frame.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        // Decision M41-2 (`SliderClamping::Edits`): a slider writes only on an edit, so idle
        // frames keep every value as it is, values off their step grid included: a face gap
        // and a measured drag from a file, and the vacuum permeability's two defaults (every
        // group open, so every row is drawn).
        let mut harness = Harness::new();
        harness.panel.inputs.metal.face_gap_mm = 1.4123;
        harness.panel.inputs.metal.measured_drag_Nm = Some(0.012345);
        let design = harness.panel.inputs.clone();
```

with:

```rust
        // Decision M41-2 (`SliderClamping::Edits`): a slider writes only on an edit, so idle
        // frames keep every value as it is, values off their step grid included: a face gap
        // and a measured drag from a file, and the vacuum permeability's two defaults (every
        // group open, on a screen tall enough to draw every row).
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 12000.0));
        harness.panel.inputs.metal.face_gap_mm = 1.4123;
        harness.panel.inputs.metal.measured_drag_Nm = Some(0.012345);
        let design = harness.panel.inputs.clone();
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    fn every_group_opens_and_draws_a_row_for_each_of_its_inputs() {
        let catalogue = InputCatalogue::get();
        for group in &catalogue.groups {
            let mut harness = Harness::new();
            harness.click_text(group.label);
            // The header opens over a few frames (its animation).
            let mut output = harness.frame(Vec::new());
```

with:

```rust
    fn every_group_opens_and_draws_a_row_for_each_of_its_inputs() {
        let catalogue = InputCatalogue::get();
        for group in &catalogue.groups {
            // Tall enough for the longest group (temperature) to fit without scrolling.
            let mut harness = Harness::on_screen(egui::vec2(1280.0, 6000.0));
            harness.click_text(group.label);
            // The header opens over a few frames (its animation).
            let mut output = harness.frame(Vec::new());
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    #[test]
    fn an_out_of_range_end_effect_shows_the_banner() {
        let mut harness = Harness::new();
        let inputs = &mut harness.panel.inputs;
        inputs.coupling.c_end = 0.5;
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        inputs.coupling.magnets.manual_inner_length_mm = 2.0;
        inputs.coupling.magnets.manual_outer_length_mm = 2.0;
        let output = harness.frame(Vec::new());
        let banner: Vec<String> = drawn_texts(&output)
            .into_iter()
            .filter(|t| t.starts_with(&format!("{END_EFFECT_BANNER} (")))
            .collect();
        assert_eq!(
            banner,
            [
                "End-effect model out of range (f_end = -1.202): the pull-out and the numbers computed from it are greyed."
            ]
        );
    }

    #[test]
```

with:

```rust
    #[test]
    fn an_out_of_range_end_effect_shows_the_banner() {
        let mut harness = Harness::new();
        harness.panel.inputs = short_magnets();
        let output = harness.frame(Vec::new());
        let banner: Vec<String> = drawn_texts(&output)
            .into_iter()
            .filter(|t| t.starts_with(&format!("{END_EFFECT_BANNER} (")))
            .collect();
        // Over the dashboard and over the results table.
        let text = "End-effect model out of range (f_end = -1.202): the pull-out and the numbers computed from it are greyed.";
        assert_eq!(banner, [text, text]);
    }

    #[test]
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
            .find(|t| t.starts_with("Exceeds the space claim:"))
            .expect("the badge text");
        assert!(claim.contains("overall length"), "{claim}");
    }

    #[test]
```

with:

```rust
            .find(|t| t.starts_with("Exceeds the space claim:"))
            .expect("the badge text");
        assert!(claim.contains("overall length"), "{claim}");
    }

    #[test]
    fn the_results_table_lists_results_and_filters_by_the_search() {
        let entries = crate::gui::results_table::table_entries();
        let cell = |index: usize| entries[index].info.cell.clone().unwrap();
        let (first, last) = (cell(0), cell(entries.len() - 1));
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        let total = entries.len();
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        // The first rows are on screen (they show their cells), the last is far below.
        assert_eq!(count(&output, &first), 1, "{first}");
        assert_eq!(count(&output, &last), 0, "{last}");
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text(last.to_lowercase())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("1 of {total} results")), 1);
        assert_eq!(count(&output, &last), 1);
        assert_eq!(count(&output, &first), 0);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_export_buttons_queue_the_files_for_the_host() {
        use crate::gui::results_table::{EXPORT_CSV, EXPORT_JSON};
        let mut harness = Harness::new();
        assert!(harness.panel.take_requests().is_empty());
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.click_text(EXPORT_CSV);
        harness.click_text(EXPORT_JSON);
        let results = harness.panel.results().clone();
        let requests = harness.panel.take_requests();
        assert_eq!(
            requests,
            vec![
                PanelRequest::SaveFile {
                    file_name: "magcoupling-results.csv".to_owned(),
                    mime: "text/csv",
                    contents: results_csv(&results),
                },
                PanelRequest::SaveFile {
                    file_name: "magcoupling-results.json".to_owned(),
                    mime: "application/json",
                    contents: results_json(&harness.panel.design(), &results),
                },
            ]
        );
        assert!(harness.panel.take_requests().is_empty(), "drained");
    }

    #[test]
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/test_support.rs`, replace:

```rust
//! pattern as `linkage-sim-rs/src/gui/test_support.rs` (a separate crate, so
//! the few helpers used here are repeated, not shared).

/// One headless frame of `draw` inside a central panel, with `events` as the
/// frame's input. Returns what egui painted.
pub(crate) fn central_panel_frame(
    ctx: &egui::Context,
    events: Vec<egui::Event>,
    mut draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    let input = egui::RawInput {
        events,
        ..Default::default()
    };
    ctx.run(input, |ctx| {
```

with:

```rust
//! pattern as `linkage-sim-rs/src/gui/test_support.rs` (a separate crate, so
//! the few helpers used here are repeated, not shared).

/// A design with f_end below 0 (the engine's short-magnet test, audit M9): 2 mm manual
/// blocks with c_end 0.5.
pub(crate) fn short_magnets() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.c_end = 0.5;
    inputs.coupling.magnets.part_inner.clear();
    inputs.coupling.magnets.part_outer.clear();
    inputs.coupling.magnets.manual_inner_length_mm = 2.0;
    inputs.coupling.magnets.manual_outer_length_mm = 2.0;
    inputs
}

/// The screen of the headless frames [points]: a laptop window.
pub(crate) const SCREEN: egui::Vec2 = egui::vec2(1280.0, 1024.0);

/// One headless frame of `draw` inside a central panel on a screen of `size`, with `events`
/// as the frame's input. Returns what egui painted.
pub(crate) fn sized_frame(
    ctx: &egui::Context,
    size: egui::Vec2,
    events: Vec<egui::Event>,
    mut draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    let input = egui::RawInput {
        events,
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
        ..Default::default()
    };
    ctx.run(input, |ctx| {
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs`, replace:

```rust
pub mod input_ui;
pub mod inputs;
mod panel;
pub mod session;
pub mod sizing;
#[cfg(test)]
```

with:

```rust
pub mod input_ui;
pub mod inputs;
mod panel;
pub mod results_table;
pub mod session;
pub mod sizing;
#[cfg(test)]
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs`, replace:

```rust

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
pub use panel::MagcouplingPanel;
```

with:

```rust

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
pub use panel::{CentreView, MagcouplingPanel, PanelRequest};
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | sort -u | head -8
```

Expected: the library itself fails to build: ``error[E0432]: unresolved imports `panel::CentreView`, `panel::PanelRequest` `` (the re-exports Step 3 backs).

- [ ] **Step 3: Write the table and the centre region**



In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/dashboard.rs`, replace:

```rust
    (results.model.end_effect_check == END_EFFECT_OUT_OF_RANGE).then_some(results.model.f_end)
}

/// The dashboard rows of `results`.
pub fn dashboard_lines(results: &DesignResults) -> Vec<DashboardLine> {
    let out_of_range = end_effect_out_of_range(results).is_some();
```

with:

```rust
    (results.model.end_effect_check == END_EFFECT_OUT_OF_RANGE).then_some(results.model.f_end)
}

/// The banner shown over the dashboard and the results table when f_end ≤ 0.
pub fn end_effect_banner(results: &DesignResults) -> Option<String> {
    end_effect_out_of_range(results).map(|f_end| {
        format!(
            "{END_EFFECT_BANNER} (f_end = {}): the pull-out and the numbers computed from it are greyed.",
            format_value(&Value::Num(f_end))
        )
    })
}

/// The dashboard rows of `results`.
pub fn dashboard_lines(results: &DesignResults) -> Vec<DashboardLine> {
    let out_of_range = end_effect_out_of_range(results).is_some();
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/dashboard.rs`, replace:

```rust
/// stored-3D label after the last temperature row.
pub fn dashboard_ui(ui: &mut egui::Ui, results: &DesignResults) {
    let lines = dashboard_lines(results);
    if let Some(f_end) = end_effect_out_of_range(results) {
        ui.colored_label(
            ui.visuals().error_fg_color,
            format!(
                "{END_EFFECT_BANNER} (f_end = {}): the pull-out and the numbers computed from it are greyed.",
                format_value(&Value::Num(f_end))
            ),
        );
        ui.separator();
    }
    let last_3d = lines.iter().rposition(|line| line.stored_3d);
```

with:

```rust
/// stored-3D label after the last temperature row.
pub fn dashboard_ui(ui: &mut egui::Ui, results: &DesignResults) {
    let lines = dashboard_lines(results);
    if let Some(banner) = end_effect_banner(results) {
        ui.colored_label(ui.visuals().error_fg_color, banner);
        ui.separator();
    }
    let last_3d = lines.iter().rposition(|line| line.stored_3d);
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
//!
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the dashboard on the right and
//! the centre region between them. Every result is recomputed with every approved correction
//! on ([`compute_all`]) each frame after the inputs are drawn, so the readouts show this
//! frame's edits.

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::gui::dashboard::dashboard_ui;
use crate::gui::input_ui::{RowEdit, input_row};
use crate::gui::inputs::{InputCatalogue, InputEntry, optional_seed};
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
```

with:

```rust
//!
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the dashboard on the right and
//! the centre region between them ([`CentreView`]: the results table; plan M4-2 adds the
//! geometry view and the plots). Every result is recomputed with every approved correction
//! on ([`compute_all`]) each frame after the inputs are drawn, so the readouts show this
//! frame's edits.
//!
//! The panel never touches files or the network: what needs the platform (saving a file,
//! picking one) it queues as a [`PanelRequest`] for the host, which drains them with
//! [`MagcouplingPanel::take_requests`] after each frame.

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::gui::dashboard::dashboard_ui;
use crate::gui::input_ui::{RowEdit, input_row};
use crate::gui::inputs::{InputCatalogue, InputEntry, optional_seed};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
use crate::gui::session::Design;
use crate::gui::sizing::SizingState;
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
/// Starting width of the dashboard side [points].
const DASHBOARD_WIDTH: f32 = 300.0;

/// The calculator panel: design inputs, their results, and the UI that edits the one and
/// shows the other.
///
```

with:

```rust
/// Starting width of the dashboard side [points].
const DASHBOARD_WIDTH: f32 = 300.0;

/// What the panel asks of its host: the platform work an egui panel cannot do itself.
#[derive(Clone, Debug, PartialEq)]
pub enum PanelRequest {
    /// Save `contents` to a file the user picks (native) or download it (web).
    SaveFile {
        /// The suggested file name.
        file_name: String,
        /// The media type (the web download's Blob type).
        mime: &'static str,
        contents: String,
    },
}

/// The views of the centre region. Plan M4-2 adds the geometry view and the plot tabs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CentreView {
    /// Every result: label, value, unit, cell; searchable; CSV and JSON export.
    Results,
}

impl CentreView {
    /// Every view, in tab order.
    pub const ALL: [CentreView; 1] = [CentreView::Results];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
            CentreView::Results => "Results table",
        }
    }
}

/// The calculator panel: design inputs, their results, and the UI that edits the one and
/// shows the other.
///
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    key_widgets: Vec<(&'static str, egui::Id)>,
    /// Why the last edit was refused, until the next accepted edit or reset.
    last_error: Option<String>,
}

impl Default for MagcouplingPanel {
```

with:

```rust
    key_widgets: Vec<(&'static str, egui::Id)>,
    /// Why the last edit was refused, until the next accepted edit or reset.
    last_error: Option<String>,
    /// The view the centre region shows.
    centre: CentreView,
    results_table: ResultsTable,
    /// Platform work for the host, oldest first.
    requests: Vec<PanelRequest>,
}

impl Default for MagcouplingPanel {
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
            results,
            key_widgets: Vec::new(),
            last_error: None,
        }
    }

```

with:

```rust
            results,
            key_widgets: Vec::new(),
            last_error: None,
            centre: CentreView::Results,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
        }
    }

    /// The platform work queued since the last call (the host saves files), oldest first.
    pub fn take_requests(&mut self) -> Vec<PanelRequest> {
        std::mem::take(&mut self.requests)
    }

    /// The design as a file holds it.
    fn design(&self) -> Design {
        Design {
            inputs: self.inputs.clone(),
            sizing: SizingState::default(),
        }
    }

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| dashboard_ui(ui, &self.results));
                });
            egui::CentralPanel::default().show_inside(ui, |_ui| {});
        });
    }

    /// The header line: heading, the session buttons, the last refusal.
```

with:

```rust
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| dashboard_ui(ui, &self.results));
                });
            egui::CentralPanel::default().show_inside(ui, |ui| self.centre_ui(ui));
        });
    }

    /// The centre region: the view tabs, then the view.
    fn centre_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for view in CentreView::ALL {
                ui.selectable_value(&mut self.centre, view, view.label());
            }
        });
        ui.separator();
        match self.centre {
            CentreView::Results => {
                let action = self.results_table.ui(ui, &self.results);
                match action {
                    Some(TableAction::ExportCsv) => self.requests.push(PanelRequest::SaveFile {
                        file_name: CSV_FILE_NAME.to_owned(),
                        mime: "text/csv",
                        contents: results_csv(&self.results),
                    }),
                    Some(TableAction::ExportJson) => self.requests.push(PanelRequest::SaveFile {
                        file_name: JSON_FILE_NAME.to_owned(),
                        mime: "application/json",
                        contents: results_json(&self.design(), &self.results),
                    }),
                    None => {}
                }
            }
        }
    }

    /// The header line: heading, the session buttons, the last refusal.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/results_table.rs`, replace:

```rust
//! The exports write every result at full precision: CSV for spreadsheets, JSON with the
//! design that produced it. JSON has no infinity or NaN, so both write a non-finite number as
//! `+inf`, `-inf` or `NaN` (decision M41-15).

#[cfg(test)]
mod tests {
```

with:

```rust
//! The exports write every result at full precision: CSV for spreadsheets, JSON with the
//! design that produced it. JSON has no infinity or NaN, so both write a non-finite number as
//! `+inf`, `-inf` or `NaN` (decision M41-15).

use std::sync::OnceLock;

use serde_json::{Map, Value as Json};

use crate::engine::meta::{ResultSet, Value, result_rows};
use crate::gui::corrections::{CorrectionIndex, marker_text};
use crate::gui::dashboard::{
    ResultInfo, ResultNotes, end_effect_banner, result_info, result_tooltip,
};
use crate::gui::format::{format_value, non_finite_text, with_unit};
use crate::gui::session::{Design, design_json, json_value};
use crate::{DesignInputs, DesignResults, compute_all};

/// The `format` of a results export.
pub const RESULTS_FORMAT: &str = "magcoupling-results";

/// The version of the results export.
pub const RESULTS_VERSION: u64 = 1;

/// The file names the exports suggest.
pub const CSV_FILE_NAME: &str = "magcoupling-results.csv";
pub const JSON_FILE_NAME: &str = "magcoupling-results.json";

/// The CSV header.
pub const CSV_HEADER: &str = "path,label,value,unit,cell";

/// The button labels.
pub const EXPORT_CSV: &str = "Export CSV";
pub const EXPORT_JSON: &str = "Export JSON";

/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";

/// One row of the table: what does not change with the inputs.
#[derive(Clone, Debug)]
pub struct TableEntry {
    pub path: String,
    pub info: &'static ResultInfo,
    /// The corrections' marker text, empty for none.
    pub marker: String,
    /// Label, path and cell, lowercase: what the search matches.
    haystack: String,
}

/// Every result's row, in schema order, built once.
pub fn table_entries() -> &'static [TableEntry] {
    static ENTRIES: OnceLock<Vec<TableEntry>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        result_rows(&compute_all(&DesignInputs::default()))
            .into_iter()
            .map(|row| {
                let info = result_info(&row.path).expect("every result has its info");
                let haystack = format!(
                    "{}\n{}\n{}",
                    info.meta.label,
                    row.path,
                    info.cell.as_deref().unwrap_or("")
                )
                .to_lowercase();
                TableEntry {
                    marker: marker_text(CorrectionIndex::get().marks(info.cell.as_deref())),
                    path: row.path,
                    info,
                    haystack,
                }
            })
            .collect()
    })
}

/// The indices of the rows whose label, path or cell contains `query`, ignoring case and the
/// surrounding blanks; every row for a blank query.
pub fn search(entries: &[TableEntry], query: &str) -> Vec<usize> {
    let needle = query.trim().to_lowercase();
    (0..entries.len())
        .filter(|&i| needle.is_empty() || entries[i].haystack.contains(&needle))
        .collect()
}

/// A number at full precision: the shortest text that reads back as the same number
/// (`2.6884762950539796`, `5.27e-10`); `+inf`, `-inf` or `NaN` when not finite.
pub fn exact_number(x: f64) -> String {
    match non_finite_text(x) {
        Some(text) => text.to_owned(),
        None => format!("{x:?}"),
    }
}

/// A value at full precision; text as it is, an empty string for none.
fn exact_value(value: &Value) -> String {
    match value {
        Value::Num(x) => exact_number(*x),
        Value::Int(i) => i.to_string(),
        Value::Text(text) => text.clone(),
        Value::None => String::new(),
    }
}

/// A CSV field (RFC 4180): quoted, with quotes doubled, when it holds a comma, a quote or a
/// line break.
fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_owned()
    }
}

/// The CSV export: [`CSV_HEADER`], then one line per result in schema order.
pub fn results_csv(results: &DesignResults) -> String {
    let mut csv = String::from(CSV_HEADER);
    csv.push_str("\r\n");
    for row in result_rows(results) {
        let fields = [
            row.path.as_str(),
            row.meta.label,
            &exact_value(&row.value),
            row.meta.unit,
            row.cell.as_deref().unwrap_or(""),
        ];
        let line: Vec<String> = fields.iter().map(|f| csv_field(f)).collect();
        csv.push_str(&line.join(","));
        csv.push_str("\r\n");
    }
    csv
}

/// The JSON export: the design and every result with its label, unit and cell.
pub fn results_json(design: &Design, results: &DesignResults) -> String {
    let rows: Vec<Json> = result_rows(results)
        .into_iter()
        .map(|row| {
            let mut entry = Map::new();
            entry.insert("path".to_owned(), Json::String(row.path));
            entry.insert("label".to_owned(), Json::from(row.meta.label));
            entry.insert("value".to_owned(), json_value(&row.value));
            entry.insert("unit".to_owned(), Json::from(row.meta.unit));
            entry.insert("cell".to_owned(), row.cell.map_or(Json::Null, Json::String));
            Json::Object(entry)
        })
        .collect();
    let mut top = Map::new();
    top.insert("format".to_owned(), Json::from(RESULTS_FORMAT));
    top.insert("version".to_owned(), Json::from(RESULTS_VERSION));
    top.insert("design".to_owned(), design_json(design));
    top.insert("results".to_owned(), Json::Array(rows));
    let mut text = serde_json::to_string_pretty(&Json::Object(top))
        .expect("a JSON value of maps, strings and numbers always serializes");
    text.push('\n');
    text
}

/// The table's state: the search text and the rows it matches.
#[derive(Clone, Debug, Default)]
pub struct ResultsTable {
    query: String,
    /// The rows matching `matched_query`; `None` until the first frame.
    matches: Option<Vec<usize>>,
    matched_query: String,
}

/// What the user asked of the table this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableAction {
    ExportCsv,
    ExportJson,
}

impl ResultsTable {
    /// The search text.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Draws the table: the end-effect banner when f_end ≤ 0, the search box and the export
    /// buttons, then the rows on screen. Returns an export asked for.
    pub fn ui(&mut self, ui: &mut egui::Ui, results: &DesignResults) -> Option<TableAction> {
        let entries = table_entries();
        let mut action = None;
        if let Some(banner) = end_effect_banner(results) {
            ui.colored_label(ui.visuals().error_fg_color, banner);
        }
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.query)
                    .hint_text(SEARCH_HINT)
                    .desired_width(260.0),
            );
            if self.matches.is_none() || self.query != self.matched_query {
                self.matches = Some(search(entries, &self.query));
                self.matched_query = self.query.clone();
            }
            let shown = self.matches.as_ref().map_or(0, Vec::len);
            ui.weak(format!("{shown} of {} results", entries.len()));
            if ui.button(EXPORT_CSV).clicked() {
                action = Some(TableAction::ExportCsv);
            }
            if ui.button(EXPORT_JSON).clicked() {
                action = Some(TableAction::ExportJson);
            }
        });
        ui.separator();
        let matches = self.matches.as_deref().unwrap_or(&[]);
        let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;
        egui::ScrollArea::both()
            .id_salt("magcoupling_results_scroll")
            .auto_shrink([false, false])
            .show_rows(ui, row_height, matches.len(), |ui, range| {
                for &index in &matches[range] {
                    row_ui(ui, &entries[index], results, row_height);
                }
            });
        action
    }
}

/// A row's hover text: the hover hook's ([`result_tooltip`]) with the exact value.
pub fn row_tooltip(entry: &TableEntry, value: &Value) -> String {
    let marks = CorrectionIndex::get().marks(entry.info.cell.as_deref());
    let mut tooltip = result_tooltip(
        &entry.path,
        entry.info,
        ResultNotes {
            marks,
            ..ResultNotes::default()
        },
    );
    if let Value::Num(x) = value {
        tooltip.push_str(&format!("\nExact value: {}", exact_number(*x)));
    }
    tooltip
}

/// One table row: label, value with unit, workbook cell, marker (the path is in the hover
/// text, which is built only while the row is hovered).
fn row_ui(ui: &mut egui::Ui, entry: &TableEntry, results: &DesignResults, height: f32) {
    let value = results.get(&entry.path).unwrap_or(Value::None);
    let row = ui.horizontal(|ui| {
        // Left-aligned columns of fixed width (add_sized would centre the text).
        let cell = |ui: &mut egui::Ui, width: f32, text: &str| {
            let layout = egui::Layout::left_to_right(egui::Align::Center);
            ui.allocate_ui_with_layout(egui::vec2(width, height), layout, |ui| {
                ui.set_min_width(width);
                ui.add(egui::Label::new(text).truncate());
            });
        };
        cell(ui, 260.0, entry.info.meta.label);
        cell(
            ui,
            130.0,
            &with_unit(format_value(&value), entry.info.meta.unit),
        );
        cell(ui, 130.0, entry.info.cell.as_deref().unwrap_or("Rust-only"));
        cell(ui, 60.0, &entry.marker);
    });
    row.response.on_hover_ui(|ui| {
        ui.label(row_tooltip(entry, &value));
    });
}

#[cfg(test)]
mod tests {
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/session.rs`, replace:

```rust
    }
}

/// The design as a JSON object.
fn design_json(design: &Design) -> Json {
    let mut inputs = Map::new();
    for row in input_rows(&design.inputs) {
        inputs.insert(row.path, json_value(&row.value));
```

with:

```rust
    }
}

/// The design as a JSON object (the results export embeds it).
pub(crate) fn design_json(design: &Design) -> Json {
    let mut inputs = Map::new();
    for row in input_rows(&design.inputs) {
        inputs.insert(row.path, json_value(&row.value));
```

- [ ] **Step 4: Run the tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED|panicked"
```

Expected: `test result: ok. 231 passed; 0 failed` (Task 5's 222 and 9 new tests).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task6.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task6.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task6.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task6.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/dashboard.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/results_table.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/session.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/test_support.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
feat(magcoupling-rs): the results table with search and CSV and JSON export

Every result with label, value, unit, cell and marker; rows built once, values
read only for the rows on screen; search by label, path or cell. CSV and JSON
(with the design) at full precision, +inf/-inf/NaN as text (decision M41-15),
queued as a PanelRequest for the host: the panel does no I/O. The centre
region becomes CentreView for M4-2's views.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 7: The session in the panel

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

Spec M4 "Session": "undo/redo of input changes; reset all; save/load design JSON; share link". The panel now holds the design (the inputs and the sizing state, still forward until Task 8) and a `History<Design>`: at the end of every frame it observes the design, settled unless an edit is in progress (`editing`): a pointer button down, a key held down (egui's `keys_down`, so a held arrow key's auto-repeat run is one step, decision M41-14), or a text field of an input row focused. egui's `wants_keyboard_input` is true for any focused widget, a slider rail too, so a focused text field is found through its `TextEditState` (`focused_text_field`), and it counts only if an input row drew it this frame (`design_widgets`: each row's widget id, which a Slider gives its value box while that has focus); the results search box is no edit. The header gets Undo and Redo (also Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y, left to a focused text field, and to the host when it keeps them: `set_keyboard_shortcuts(false)`, for M5, whose linkage window reads the same keys), Reset all (undoable), Save design (a `SaveFile` request), Load design (a new `PanelRequest::OpenDesign`; the host hands the text to `load_design_file`) and Copy share link (`ctx.copy_text`; the link's base is the host's, `set_share_base`, by default the public page). A refused file or link changes nothing and the header names every problem; `report` shows what the host's save did. `open_design` and `open_share_payload` open a design as the session's start (a new history: the first Undo keeps a share link opened at start-up). A status line one frame late is the egui panel's sizing, so the tests run two frames before reading it.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`

**Interfaces:**
- Consumes: Task 1's `Design`, `LoadError`, `PUBLIC_BASE_URL`, `design_to_json`, `design_from_json`, `decode_share_payload`, `share_link`, `SizingState`; Task 2's `History`; Task 6's `PanelRequest`.
- Produces (`MagcouplingPanel`): `fn design(&self) -> Design`; `fn set_share_base(&mut self, impl Into<String>)`; `fn share_link(&self) -> String`; `fn load_design_file(&mut self, &str) -> Result<(), LoadError>`; `fn load_share_payload(&mut self, &str) -> Result<(), LoadError>`; `fn open_design(&mut self, Design)`; `fn open_share_payload(&mut self, &str) -> Result<(), LoadError>`; `fn set_keyboard_shortcuts(&mut self, bool)`; `fn undo(&mut self)`; `fn redo(&mut self)`; `reset` now restores the whole default design and can be undone; `PanelRequest::OpenDesign`; constants `UNDO`, `REDO`, `SAVE_DESIGN`, `LOAD_DESIGN`, `COPY_SHARE_LINK`, `DESIGN_FILE_NAME = "magcoupling-design.json"`, `UNDO_SHORTCUT`, `REDO_SHORTCUTS`; private `focused_text_field(&egui::Context) -> Option<egui::Id>`, `editing(&self, &egui::Ui) -> bool`, the fields `design_widgets` and `keyboard_shortcuts`. Task 9 adds `report`.

- [ ] **Step 1: Write the failing tests**

The panel's session tests: undo reverses a change and redo brings it back (the spec's "undo reverses a change"), each nudge one step, a held arrow key one step, a drag one step, the shortcuts (tapped), a host keeping Ctrl+Z for itself inside an `egui::Window`, Ctrl+Z inside a text field left to the field, a slider's value box being typed in counting as an edit of the design (one step on Enter), a focused results search holding back no undo step, only the rows drawn this frame counting as design fields, a typed part name one step once the field loses focus, reset all undoable, the save and load requests, a loaded file replacing the design (undoable) and a refused one changing nothing with its message, the share link on the clipboard reopening the design in another panel, a broken link refused, a share link opened at start-up being no undo step; the idle-frames test opens its design with `open_design` and also checks that no undo step was recorded.

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    }
}

#[cfg(test)]
mod tests {
    use super::*;
```

with:

```rust
    }
}

/// The widget with keyboard focus if it is a text field: a text input, or a slider's value box
/// being typed in (egui's `wants_keyboard_input` is true for any focused widget, a slider rail
/// too).
fn focused_text_field(ctx: &egui::Context) -> Option<egui::Id> {
    ctx.memory(|m| m.focused())
        .filter(|&id| egui::text_edit::TextEditState::load(ctx, id).is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::input_ui::{CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::test_support::{
        SCREEN, drawn_texts, key_tap, primary_button, select_all, short_magnets, sized_frame,
        text_rect,
    };
    use crate::headline;

```

with:

```rust
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::input_ui::{CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::session::encode_share_payload;
    use crate::gui::test_support::{
        SCREEN, drawn_texts, key_event, key_tap, primary_button, select_all, short_magnets,
        sized_frame, text_rect,
    };
    use crate::headline;

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        // and a measured drag from a file, and the vacuum permeability's two defaults (every
        // group open, on a screen tall enough to draw every row).
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 12000.0));
        harness.panel.inputs.metal.face_gap_mm = 1.4123;
        harness.panel.inputs.metal.measured_drag_Nm = Some(0.012345);
        let design = harness.panel.inputs.clone();
        // Bottom up: opening a group moves only the groups below it, so no click lands on a
        // row that an opening group above has just moved there.
        for group in InputCatalogue::get().groups.iter().rev() {
```

with:

```rust
        // and a measured drag from a file, and the vacuum permeability's two defaults (every
        // group open, on a screen tall enough to draw every row).
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 12000.0));
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 1.4123;
        design.inputs.metal.measured_drag_Nm = Some(0.012345);
        harness.panel.open_design(design.clone());
        // Bottom up: opening a group moves only the groups below it, so no click lands on a
        // row that an opening group above has just moved there.
        for group in InputCatalogue::get().groups.iter().rev() {
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
            2,
            "both mu0 rows drawn"
        );
        assert_eq!(harness.panel.inputs(), &design);
        assert_eq!(harness.panel.last_error, None);
    }

    #[test]
```

with:

```rust
            2,
            "both mu0 rows drawn"
        );
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(
            harness.panel.history.undo_len(),
            0,
            "no row reported an edit"
        );
        assert!(!harness.panel.history.can_undo(&design));
    }

    #[test]
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        assert_eq!(panel.results(), &compute_all(&DesignInputs::default()));
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
```

with:

```rust
        assert_eq!(panel.results(), &compute_all(&DesignInputs::default()));
    }

    /// A keyboard shortcut tapped: pressed and released in one frame.
    fn shortcut_tap(shortcut: egui::KeyboardShortcut) -> Vec<egui::Event> {
        [true, false]
            .map(|pressed| egui::Event::Key {
                key: shortcut.logical_key,
                physical_key: None,
                pressed,
                repeat: false,
                modifiers: shortcut.modifiers,
            })
            .to_vec()
    }

    /// The design with the face gap at `gap`.
    fn gap_design(gap: f64) -> Design {
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = gap;
        design
    }

    #[test]
    fn undo_reverses_a_change_and_redo_brings_it_back() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(harness.number(FACE_GAP), 1.41);
        harness.click_text(UNDO);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(
            harness.panel.results(),
            &compute_all(&DesignInputs::default())
        );
        harness.click_text(REDO);
        assert_eq!(harness.panel.design(), gap_design(1.41));
        assert_eq!(
            harness.panel.results(),
            &compute_all(&gap_design(1.41).inputs)
        );
    }

    #[test]
    fn each_arrow_nudge_is_one_undo_step() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        for _ in 0..3 {
            harness.frame(key_tap(egui::Key::ArrowRight));
        }
        assert_eq!(harness.panel.history.undo_len(), 3);
        harness.panel.undo();
        assert_eq!(harness.number(FACE_GAP), 1.42);
    }

    #[test]
    fn a_held_arrow_key_is_one_undo_step() {
        // A held arrow key auto-repeats (about 30 presses a second; egui reads every press after
        // the first as a repeat). The run is one edit until the key is released (decision
        // M41-14), so holding a key cannot flood the undo levels.
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        for _ in 0..20 {
            harness.frame(vec![key_event(egui::Key::ArrowRight, true)]);
        }
        assert_eq!(harness.number(FACE_GAP), 1.6);
        assert_eq!(harness.panel.history.undo_len(), 0, "still held");
        harness.frame(vec![key_event(egui::Key::ArrowRight, false)]);
        assert_eq!(harness.panel.history.undo_len(), 1);
        harness.panel.undo();
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn a_drag_is_one_undo_step() {
        let mut harness = Harness::new();
        let rail = harness.widget(FACE_GAP).rect;
        let start = rail.left_center() + egui::vec2(20.0, 0.0);
        harness.frame(vec![egui::Event::PointerMoved(start)]);
        harness.frame(vec![primary_button(start, true)]);
        for step in 1..=5 {
            let at = start + egui::vec2(10.0 * step as f32, 0.0);
            harness.frame(vec![egui::Event::PointerMoved(at)]);
        }
        let end = start + egui::vec2(50.0, 0.0);
        harness.frame(vec![primary_button(end, false)]);
        harness.frame(Vec::new());
        let dragged = harness.number(FACE_GAP);
        assert!(dragged > 1.4, "{dragged}");
        assert_eq!(harness.panel.history.undo_len(), 1, "the whole drag");
        harness.panel.undo();
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn the_keyboard_shortcuts_undo_and_redo() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.frame(shortcut_tap(UNDO_SHORTCUT));
        assert_eq!(harness.panel.design(), Design::default());
        harness.frame(shortcut_tap(REDO_SHORTCUTS[0]));
        assert_eq!(harness.panel.design(), gap_design(1.41));
        harness.frame(shortcut_tap(UNDO_SHORTCUT));
        harness.frame(shortcut_tap(REDO_SHORTCUTS[1]));
        assert_eq!(harness.panel.design(), gap_design(1.41));
    }

    #[test]
    fn a_host_can_keep_ctrl_z_for_itself() {
        // M5 hosts the panel in an egui::Window of the linkage app, which undoes its own model
        // on Ctrl+Z. With the panel's shortcuts off the key is the host's alone: the panel does
        // not undo, and the event is still there for the host after the panel's frame.
        let ctx = egui::Context::default();
        let mut panel = MagcouplingPanel::new();
        panel
            .load_design_file(&design_to_json(&gap_design(2.0)))
            .unwrap();
        let frame = |panel: &mut MagcouplingPanel, events: Vec<egui::Event>| {
            let mut host_saw_undo = false;
            let input = egui::RawInput {
                events,
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::Window::new("Magnetic coupling")
                    .default_size([1100.0, 1200.0])
                    .show(ctx, |ui| panel.ui(ui));
                host_saw_undo = ctx.input_mut(|i| i.consume_shortcut(&UNDO_SHORTCUT));
            });
            host_saw_undo
        };
        frame(&mut panel, Vec::new());
        panel.set_keyboard_shortcuts(false);
        assert!(
            frame(&mut panel, shortcut_tap(UNDO_SHORTCUT)),
            "left to the host"
        );
        assert_eq!(panel.design(), gap_design(2.0), "the panel did not undo");
        // On (the default), the panel undoes and takes the event.
        panel.set_keyboard_shortcuts(true);
        assert!(!frame(&mut panel, shortcut_tap(UNDO_SHORTCUT)));
        assert_eq!(panel.design(), Design::default());
    }

    #[test]
    fn ctrl_z_in_a_text_field_is_left_to_the_field() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.focus(PART_INNER);
        harness.frame(vec![egui::Event::Text("X".to_owned())]);
        harness.frame(shortcut_tap(UNDO_SHORTCUT));
        // The design's undo did not run: the face gap step is still there, nothing to redo.
        assert_eq!(harness.number(FACE_GAP), 1.41);
        assert!(!harness.panel.history.can_redo(&harness.panel.design()));
    }

    #[test]
    fn typing_a_part_name_is_one_undo_step_once_the_field_loses_focus() {
        let mut harness = Harness::new();
        harness.focus(PART_INNER);
        for letter in ["A", "B", "C"] {
            harness.frame(vec![egui::Event::Text(letter.to_owned())]);
        }
        assert_eq!(harness.panel.history.undo_len(), 0, "still typing");
        harness.ctx.memory_mut(|m| m.stop_text_input());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.history.undo_len(), 1);
        harness.panel.undo();
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn a_slider_s_value_box_being_typed_in_is_an_edit_of_the_design() {
        // egui gives a Slider's response its value box's id while the box has focus, so the
        // row's widget id names the focused text field: typing a value is an edit in progress.
        let mut harness = Harness::new();
        harness.click_text("1.40 mm");
        let focused = focused_text_field(&harness.ctx).expect("the value box has focus");
        assert!(harness.panel.design_widgets.contains(&focused));
        harness.frame([select_all(), vec![egui::Event::Text("2".to_owned())]].concat());
        assert_eq!(harness.panel.history.undo_len(), 0, "still typing");
        harness.frame(key_tap(egui::Key::Enter));
        harness.frame(Vec::new());
        assert_eq!(harness.number(FACE_GAP), 2.0);
        assert_eq!(harness.panel.history.undo_len(), 1);
    }

    #[test]
    fn a_focused_results_search_holds_back_no_undo_step() {
        // The search box is a text field, but it edits no design: a change while it has focus
        // (a design file the host's picker delivers) is an undo step at once.
        let mut harness = Harness::new();
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("pull".to_owned())]);
        assert!(
            focused_text_field(&harness.ctx).is_some(),
            "typing in the search"
        );
        harness
            .panel
            .load_design_file(&design_to_json(&gap_design(2.5)))
            .unwrap();
        harness.frame(Vec::new());
        assert!(
            focused_text_field(&harness.ctx).is_some(),
            "still in the search"
        );
        assert_eq!(harness.panel.history.undo_len(), 1);
    }

    #[test]
    fn only_the_rows_drawn_this_frame_count_as_design_fields() {
        let mut harness = Harness::new();
        let key_rows = InputCatalogue::get().key_design.len();
        assert_eq!(harness.panel.design_widgets.len(), key_rows);
        // Every group closed: no row is drawn, so none counts (no stale ids, no growth).
        harness.click_text(KEY_DESIGN_HEADING);
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        assert!(harness.panel.design_widgets.is_empty());
    }

    #[test]
    fn reset_all_can_be_undone() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.design(), Design::default());
        harness.click_text(UNDO);
        assert_eq!(harness.panel.design(), gap_design(1.41));
    }

    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.click_text(SAVE_DESIGN);
        harness.click_text(LOAD_DESIGN);
        assert_eq!(
            harness.panel.take_requests(),
            vec![
                PanelRequest::SaveFile {
                    file_name: DESIGN_FILE_NAME.to_owned(),
                    mime: "application/json",
                    contents: design_to_json(&gap_design(1.41)),
                },
                PanelRequest::OpenDesign,
            ]
        );
    }

    #[test]
    fn a_loaded_design_file_replaces_the_design_and_can_be_undone() {
        let mut harness = Harness::new();
        let mut loaded = gap_design(2.5);
        loaded.sizing.target_Nm = 3.0;
        harness
            .panel
            .load_design_file(&design_to_json(&loaded))
            .unwrap();
        // The header grows to the status line on the next frame.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.design(), loaded);
        assert_eq!(harness.panel.results(), &compute_all(&loaded.inputs));
        assert_eq!(count(&output, "Design file loaded"), 1);
        harness.panel.undo();
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn a_refused_design_file_changes_nothing_and_says_why() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        let text =
            r#"{"format": "magcoupling-design", "version": 1, "inputs": {"coupling.backiron": 7}}"#;
        let error = harness.panel.load_design_file(text).unwrap_err();
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.design(), gap_design(1.41));
        assert_eq!(count(&output, &error.to_string()), 1);
        assert_eq!(
            error.to_string(),
            "inputs refused: coupling.backiron: 7 is not one of the choices"
        );
    }

    #[test]
    fn copy_share_link_puts_the_design_s_link_on_the_clipboard() {
        let mut harness = Harness::new();
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness
            .panel
            .set_share_base("http://localhost:8080/magcoupling/");
        let output = harness.click_text(COPY_SHARE_LINK);
        let copied: Vec<&String> = output
            .platform_output
            .commands
            .iter()
            .filter_map(|c| match c {
                egui::OutputCommand::CopyText(text) => Some(text),
                _ => None,
            })
            .collect();
        assert_eq!(copied, [&harness.panel.share_link()]);
        let payload = copied[0]
            .strip_prefix("http://localhost:8080/magcoupling/?m=")
            .expect("the page's own address");
        // The link opens the same design in another panel.
        let mut other = MagcouplingPanel::new();
        other.load_share_payload(payload).unwrap();
        assert_eq!(other.design(), gap_design(1.41));
    }

    #[test]
    fn a_broken_share_link_changes_nothing_and_says_why() {
        let mut panel = MagcouplingPanel::new();
        let error = panel.load_share_payload("not-a-design").unwrap_err();
        assert!(matches!(error, LoadError::Link(_)), "{error}");
        assert_eq!(panel.design(), Design::default());
        assert_eq!(panel.last_error, Some(error.to_string()));
    }

    #[test]
    fn a_share_link_opened_at_start_up_is_not_an_undo_step() {
        // The web app opens a ?m= link as the session's start: the first Undo must not throw
        // the shared design away.
        let shared = gap_design(2.0);
        let mut harness = Harness::new();
        harness
            .panel
            .open_share_payload(&encode_share_payload(&shared))
            .unwrap();
        harness.frame(Vec::new());
        assert_eq!(harness.panel.design(), shared);
        assert!(!harness.panel.history.can_undo(&shared));
        harness.frame(shortcut_tap(UNDO_SHORTCUT));
        assert_eq!(harness.panel.design(), shared);
        // An edit after it undoes back to the shared design, not to the defaults.
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(harness.panel.design(), gap_design(2.01));
        harness.panel.undo();
        assert_eq!(harness.panel.design(), shared);
        // A refused link changes neither the design nor the history.
        let error = harness
            .panel
            .open_share_payload("not-a-design")
            .unwrap_err();
        assert_eq!(harness.panel.last_error, Some(error.to_string()));
        assert_eq!(harness.panel.design(), shared);
        assert!(harness.panel.history.can_redo(&shared));
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | sort -u | head -8
```

Expected: the build fails with compile errors naming the items Step 3 adds, among them ``error[E0609]: no field `history` on type `gui::panel::MagcouplingPanel` ``, ``error[E0599]: no method named `undo` found ...``, ``error[E0599]: no method named `load_design_file` found ...`` and ``error[E0425]: cannot find value `UNDO_SHORTCUT` in this scope``.

- [ ] **Step 3: Wire the session into the panel**



In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
//!
//! The panel never touches files or the network: what needs the platform (saving a file,
//! picking one) it queues as a [`PanelRequest`] for the host, which drains them with
//! [`MagcouplingPanel::take_requests`] after each frame.

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::gui::dashboard::dashboard_ui;
use crate::gui::input_ui::{RowEdit, input_row};
use crate::gui::inputs::{InputCatalogue, InputEntry, optional_seed};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
use crate::gui::session::Design;
use crate::gui::sizing::SizingState;
use crate::{DesignInputs, DesignResults, compute_all};

```

with:

```rust
//!
//! The panel never touches files or the network: what needs the platform (saving a file,
//! picking one) it queues as a [`PanelRequest`] for the host, which drains them with
//! [`MagcouplingPanel::take_requests`] after each frame and hands a picked design file back
//! through [`MagcouplingPanel::load_design_file`].
//!
//! Session (spec M4 "Session"): undo and redo of every change to the design (the inputs and
//! the sizing state, [`crate::gui::history`]: one step per settled edit), reset all, save and
//! load a design file, and a share link copied to the clipboard ([`crate::gui::session`]). A
//! host may open a design as the session's start ([`MagcouplingPanel::open_share_payload`]:
//! no undo step) and may keep the undo keys for itself
//! ([`MagcouplingPanel::set_keyboard_shortcuts`]).

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::gui::dashboard::dashboard_ui;
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row};
use crate::gui::inputs::{InputCatalogue, InputEntry, optional_seed};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
use crate::gui::session::{
    Design, LoadError, PUBLIC_BASE_URL, decode_share_payload, design_from_json, design_to_json,
    share_link,
};
use crate::gui::sizing::SizingState;
use crate::{DesignInputs, DesignResults, compute_all};

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust

/// The label of the button that restores the default design.
pub const RESET_ALL: &str = "Reset all";

/// The heading of the Key design group.
pub const KEY_DESIGN_HEADING: &str = "Key design";
```

with:

```rust

/// The label of the button that restores the default design.
pub const RESET_ALL: &str = "Reset all";

/// The session buttons.
pub const UNDO: &str = "Undo";
pub const REDO: &str = "Redo";
pub const SAVE_DESIGN: &str = "Save design";
pub const LOAD_DESIGN: &str = "Load design";
pub const COPY_SHARE_LINK: &str = "Copy share link";

/// The file name a saved design suggests.
pub const DESIGN_FILE_NAME: &str = "magcoupling-design.json";

/// Undo: Ctrl+Z (Cmd+Z on a Mac).
pub const UNDO_SHORTCUT: egui::KeyboardShortcut =
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z);

/// Redo: Ctrl+Shift+Z (Cmd+Shift+Z on a Mac), or Ctrl+Y.
pub const REDO_SHORTCUTS: [egui::KeyboardShortcut; 2] = [
    egui::KeyboardShortcut::new(
        egui::Modifiers::COMMAND.plus(egui::Modifiers::SHIFT),
        egui::Key::Z,
    ),
    egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y),
];

/// The heading of the Key design group.
pub const KEY_DESIGN_HEADING: &str = "Key design";
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        mime: &'static str,
        contents: String,
    },
}

/// The views of the centre region. Plan M4-2 adds the geometry view and the plot tabs.
```

with:

```rust
        mime: &'static str,
        contents: String,
    },
    /// Let the user pick a design file, then hand its text to
    /// [`MagcouplingPanel::load_design_file`].
    OpenDesign,
}

/// The views of the centre region. Plan M4-2 adds the geometry view and the plot tabs.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
/// [`MagcouplingPanel::ui`] once per frame.
pub struct MagcouplingPanel {
    inputs: DesignInputs,
    /// The results of `inputs`, recomputed by every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// The main widget of each Key design row in the last frame, by input path.
    key_widgets: Vec<(&'static str, egui::Id)>,
    /// Why the last edit was refused, until the next accepted edit or reset.
    last_error: Option<String>,
    /// The view the centre region shows.
```

with:

```rust
/// [`MagcouplingPanel::ui`] once per frame.
pub struct MagcouplingPanel {
    inputs: DesignInputs,
    /// The sizing mode, free variable and target torque.
    sizing: SizingState,
    /// Undo and redo of the design (inputs and sizing state).
    history: History<Design>,
    /// The address share links point at.
    share_base: String,
    /// What the last session action did, until the next one.
    status: Option<String>,
    /// The results of `inputs`, recomputed by every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// The main widget of each Key design row in the last frame, by input path.
    key_widgets: Vec<(&'static str, egui::Id)>,
    /// The main widget of every input row drawn in the last frame (a slider's value box while
    /// it has focus): a text field among them edits the design.
    design_widgets: Vec<egui::Id>,
    /// Whether the panel acts on the undo and redo keys
    /// ([`MagcouplingPanel::set_keyboard_shortcuts`]).
    keyboard_shortcuts: bool,
    /// Why the last edit was refused, until the next accepted edit or reset.
    last_error: Option<String>,
    /// The view the centre region shows.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        let results = compute_all(&inputs);
        Self {
            inputs,
            results,
            key_widgets: Vec::new(),
            last_error: None,
            centre: CentreView::Results,
            results_table: ResultsTable::default(),
```

with:

```rust
        let results = compute_all(&inputs);
        Self {
            inputs,
            sizing: SizingState::default(),
            history: History::new(Design::default()),
            share_base: PUBLIC_BASE_URL.to_owned(),
            status: None,
            results,
            key_widgets: Vec::new(),
            design_widgets: Vec::new(),
            keyboard_shortcuts: true,
            last_error: None,
            centre: CentreView::Results,
            results_table: ResultsTable::default(),
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        std::mem::take(&mut self.requests)
    }

    /// The design as a file holds it.
    fn design(&self) -> Design {
        Design {
            inputs: self.inputs.clone(),
            sizing: SizingState::default(),
        }
    }

```

with:

```rust
        std::mem::take(&mut self.requests)
    }

    /// The design: the inputs and the sizing state, as a file holds them.
    pub fn design(&self) -> Design {
        Design {
            inputs: self.inputs.clone(),
            sizing: self.sizing,
        }
    }

    /// Replaces the design (an edit like any other: it can be undone).
    fn set_design(&mut self, design: Design) {
        self.inputs = design.inputs;
        self.sizing = design.sizing;
        self.results = compute_all(&self.inputs);
        self.last_error = None;
    }

    /// Sets the address share links point at: the page's own address on the web (so a link
    /// made on a local server opens there), [`PUBLIC_BASE_URL`] by default.
    pub fn set_share_base(&mut self, base: impl Into<String>) {
        self.share_base = base.into();
    }

    /// The share link of the design.
    pub fn share_link(&self) -> String {
        share_link(&self.share_base, &self.design())
    }

    /// Loads a design file's text; on refusal the design is unchanged and the panel shows why.
    pub fn load_design_file(&mut self, text: &str) -> Result<(), LoadError> {
        self.load(design_from_json(text), "Design file loaded")
    }

    /// Loads the design a share link's `?m=` value holds; on refusal the design is unchanged
    /// and the panel shows why.
    pub fn load_share_payload(&mut self, payload: &str) -> Result<(), LoadError> {
        self.load(
            decode_share_payload(payload),
            "Design loaded from the share link",
        )
    }

    /// Opens `design` as the session's start: the design is replaced and the undo history
    /// starts there, so the first Undo does not throw it away. Every other replacement of the
    /// design (a loaded file, reset all) is an edit and can be undone.
    pub fn open_design(&mut self, design: Design) {
        self.history = History::new(design.clone());
        self.set_design(design);
    }

    /// Opens the design a share link's `?m=` value holds as the session's start
    /// ([`MagcouplingPanel::open_design`]: the web app's link at start-up); on refusal the
    /// design and the history are unchanged and the panel shows why.
    pub fn open_share_payload(&mut self, payload: &str) -> Result<(), LoadError> {
        self.load_with(
            decode_share_payload(payload),
            "Design loaded from the share link",
            Self::open_design,
        )
    }

    /// Lets the panel act on Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y (`true`, the default) or leaves
    /// the key events to the host. A host with its own undo turns them off while its own part
    /// has the user's attention, so one key press never undoes both (M5: the linkage app reads
    /// the same keys without consuming them).
    pub fn set_keyboard_shortcuts(&mut self, enabled: bool) {
        self.keyboard_shortcuts = enabled;
    }

    /// Applies a loaded design as an edit that can be undone.
    fn load(&mut self, design: Result<Design, LoadError>, done: &str) -> Result<(), LoadError> {
        self.load_with(design, done, Self::set_design)
    }

    /// Applies a loaded design with `apply` (an edit, or a new start) and says `done`; on
    /// refusal changes nothing and shows why.
    fn load_with(
        &mut self,
        design: Result<Design, LoadError>,
        done: &str,
        apply: fn(&mut Self, Design),
    ) -> Result<(), LoadError> {
        match design {
            Ok(design) => {
                apply(self, design);
                self.status = Some(done.to_owned());
                Ok(())
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
                self.status = None;
                Err(error)
            }
        }
    }

    /// Undoes the last change to the design, if any.
    pub fn undo(&mut self) {
        if let Some(design) = self.history.undo(&self.design()) {
            self.set_design(design);
        }
    }

    /// Redoes the last undone change, if any.
    pub fn redo(&mut self) {
        if let Some(design) = self.history.redo(&self.design()) {
            self.set_design(design);
        }
    }

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        &self.results
    }

    /// Back to the default design.
    pub fn reset(&mut self) {
        self.inputs = DesignInputs::default();
        self.results = compute_all(&self.inputs);
        self.last_error = None;
    }

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.push_id("magcoupling_panel", |ui| {
            egui::TopBottomPanel::top("magcoupling_header").show_inside(ui, |ui| {
                self.header_ui(ui);
            });
```

with:

```rust
        &self.results
    }

    /// Back to the default design (inputs and sizing state); it can be undone.
    pub fn reset(&mut self) {
        self.set_design(Design::default());
    }

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.push_id("magcoupling_panel", |ui| {
            self.shortcuts(ui);
            egui::TopBottomPanel::top("magcoupling_header").show_inside(ui, |ui| {
                self.header_ui(ui);
            });
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
                });
            egui::CentralPanel::default().show_inside(ui, |ui| self.centre_ui(ui));
        });
    }

    /// The centre region: the view tabs, then the view.
```

with:

```rust
                });
            egui::CentralPanel::default().show_inside(ui, |ui| self.centre_ui(ui));
        });
        // One undo step per settled edit.
        let settled = !self.editing(ui);
        self.history.observe(&self.design(), settled);
    }

    /// Whether an edit of the design is in progress: a pointer button down (a drag), a key held
    /// down (an arrow key auto-repeating: the whole run is one step, decision M41-14), or a
    /// text field of an input row focused (a typed value, a part name). Undo steps wait for it
    /// to end. A focused text field elsewhere (the results search) edits no design, so it holds
    /// nothing back.
    fn editing(&self, ui: &egui::Ui) -> bool {
        let design_text =
            focused_text_field(ui.ctx()).is_some_and(|id| self.design_widgets.contains(&id));
        design_text || ui.input(|i| i.pointer.any_down() || !i.keys_down.is_empty())
    }

    /// Ctrl+Z and Ctrl+Shift+Z or Ctrl+Y, unless the host keeps them
    /// ([`MagcouplingPanel::set_keyboard_shortcuts`]) or a text field has focus (it undoes its
    /// own typing).
    fn shortcuts(&mut self, ui: &mut egui::Ui) {
        if !self.keyboard_shortcuts || focused_text_field(ui.ctx()).is_some() {
            return;
        }
        // The redo shortcuts first: Ctrl+Z would also match Ctrl+Shift+Z.
        let redo = ui.input_mut(|i| REDO_SHORTCUTS.iter().any(|s| i.consume_shortcut(s)));
        let undo = !redo && ui.input_mut(|i| i.consume_shortcut(&UNDO_SHORTCUT));
        if redo {
            self.redo();
        } else if undo {
            self.undo();
        }
    }

    /// The centre region: the view tabs, then the view.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        }
    }

    /// The header line: heading, the session buttons, the last refusal.
    fn header_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading(HEADING);
            ui.separator();
            if ui.button(RESET_ALL).clicked() {
                self.reset();
            }
        });
        if let Some(error) = &self.last_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
    }

    /// The left side: the Key design group, then every input by package group.
    fn inputs_ui(&mut self, ui: &mut egui::Ui) {
        let catalogue = InputCatalogue::get();
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
            .auto_shrink([false, false])
```

with:

```rust
        }
    }

    /// The header line: heading, the session buttons, then the last refusal or what the last
    /// session action did.
    fn header_ui(&mut self, ui: &mut egui::Ui) {
        let design = self.design();
        ui.horizontal_wrapped(|ui| {
            ui.heading(HEADING);
            ui.separator();
            let can_undo = self.history.can_undo(&design);
            if ui
                .add_enabled(can_undo, egui::Button::new(UNDO))
                .on_hover_text("Ctrl+Z")
                .clicked()
            {
                self.undo();
            }
            let can_redo = self.history.can_redo(&design);
            if ui
                .add_enabled(can_redo, egui::Button::new(REDO))
                .on_hover_text("Ctrl+Shift+Z or Ctrl+Y")
                .clicked()
            {
                self.redo();
            }
            if ui.button(RESET_ALL).clicked() {
                self.reset();
                self.status = Some("Default design restored".to_owned());
            }
            ui.separator();
            if ui.button(SAVE_DESIGN).clicked() {
                self.requests.push(PanelRequest::SaveFile {
                    file_name: DESIGN_FILE_NAME.to_owned(),
                    mime: "application/json",
                    contents: design_to_json(&design),
                });
            }
            if ui.button(LOAD_DESIGN).clicked() {
                self.requests.push(PanelRequest::OpenDesign);
            }
            if ui
                .button(COPY_SHARE_LINK)
                .on_hover_text("A link that opens this design, sizing state included")
                .clicked()
            {
                let link = self.share_link();
                self.status = Some(format!(
                    "Share link copied to the clipboard ({} characters)",
                    link.len()
                ));
                ui.ctx().copy_text(link);
            }
        });
        if let Some(error) = &self.last_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        } else if let Some(status) = &self.status {
            ui.weak(status);
        }
    }

    /// The left side: the Key design group, then every input by package group.
    fn inputs_ui(&mut self, ui: &mut egui::Ui) {
        let catalogue = InputCatalogue::get();
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
            .auto_shrink([false, false])
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry);
                            self.key_widgets.push((entry.path.as_str(), widget));
                        }
                    });
                for group in &catalogue.groups {
```

with:

```rust
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry);
                            self.key_widgets.push((entry.path.as_str(), widget));
                            self.design_widgets.push(widget);
                        }
                    });
                for group in &catalogue.groups {
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
                                    ui.strong(section.label);
                                }
                                for entry in &section.entries {
                                    self.input_row_ui(ui, entry);
                                }
                            }
                        });
```

with:

```rust
                                    ui.strong(section.label);
                                }
                                for entry in &section.entries {
                                    let widget = self.input_row_ui(ui, entry);
                                    self.design_widgets.push(widget);
                                }
                            }
                        });
```

- [ ] **Step 4: Run the tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED|panicked"
```

Expected: `test result: ok. 249 passed; 0 failed` (Task 6's 231 and 18 new tests).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task7.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task7.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task7.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task7.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
feat(magcoupling-rs): session: undo and redo, design files and share links in the panel

Undo and redo of the design (buttons and shortcuts; one step per settled edit:
no pointer button down, no key held down, no text field of an input row
focused, found through its TextEditState since egui's wants_keyboard_input is
true for any focused widget; the results search is no edit), reset all as an
undoable edit, a start-up share link as the session's start, the undo keys
left to a host that keeps them (M5), Save design and Load design as host requests, the share link on the
clipboard; refused files and links change nothing and name every problem.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 8: The sizing mode switch

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

Spec Addendum A1: "Mode switch in the Key design group: Magnets -> Torque (today's forward calculation) and Torque -> Magnets (inverse sizing)", the free variable picked by the user, the target torque, "solved at X" or "not reachable (best Y)". `SizingRunner` calls `sizing::solve` once the design has been unchanged for `DEBOUNCE_S` (0.25 s) and no edit is in progress, never per frame (the test counts the solves), and asks for a repaint while a solve is pending (an idle web page would never fire it) and after one (the inputs side, drawn before the solve, shows the outcome on the next frame). The solve's key leaves out the free variable's own value, which the solve sets. In Torque -> Magnets the panel shows the inputs with the free variable at the solved value, or at the best value when the target is out of reach (decision M41-8); the free variable's row shows that value, locked; leaving the mode writes it into the inputs as one undoable edit, solving first, once, a change still waiting for its debounce (`solve_now`, M41-7). When the Key design group does not list the free variable (the ring radius), its locked row is drawn under the status line. The target slider uses `metal.required_min_Nm`'s metadata (M41-9), and its value box counts as a text field of the design. The JSON export now holds the design shown (the free variable at its solved value), so its results follow from it. Each outcome is logged (`magcoupling sizing: ...`, the web smoke reads it); feature `gui` gains `log`. A glyph test checks every drawn and hover text against egui's default fonts (the arrow U+2192 drew as a box in the first smoke).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml` (feature `gui` gains `dep:log`; no lock change)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/sizing.rs` (`SizingRunner`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs` (the controls, the shown design, the locked row; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/test_support.rs` (`sized_frame_at`)

**Interfaces:**
- Consumes: `engine::sizing::{FreeVariable, SizingError, SizingOutcome, SizingPoint, solve}`; Task 1's `SizingMode`, `SizingState`, `TARGET_RANGE_INPUT`, `variable_label`; Task 3's `InputCatalogue`; Task 4's `slider`; Task 5's `Level`.
- Produces (`magcoupling::gui::sizing`): `const DEBOUNCE_S: f64 = 0.25`; `SOLVED_PREFIX = "Solved at "`, `NOT_REACHABLE_PREFIX = "Not reachable"`, `SOLVING = "Solving..."`; `struct SizingRunner` (`Default`) with `fn update(&mut self, &DesignInputs, &SizingState, now: f64, editing: bool) -> Option<f64>` (seconds to wait while a solve is pending), `fn is_current(&self, &DesignInputs, &SizingState) -> bool`, `fn outcome(&self, FreeVariable) -> Option<&Result<SizingOutcome, SizingError>>`, `fn value(&self, FreeVariable) -> Option<f64>`, `fn shown(&self, &DesignInputs, &SizingState) -> DesignInputs`, `fn status(&self, &DesignInputs, &SizingState) -> String`, `fn solve_now(&mut self, &DesignInputs, &SizingState)`; `pub(crate) solves: usize`.
- Produces (`MagcouplingPanel`): `fn shown_inputs(&self) -> DesignInputs`; `fn sizing(&self) -> &SizingState`; `results()` now reads the design shown; constants `TARGET_LABEL`, `FREE_VARIABLE_LABEL`, `SIZED_NOTE`; private `run_sizing`, `log_sizing`, `sizing_ui` (Task 7's `editing` also gates the solve).
- Produces (`gui::test_support`, tests only): `fn sized_frame_at(ctx, size, time: Option<f64>, events, draw)`.

- [ ] **Step 1: Write the failing tests**

The runner's tests (the debounce and one solve, an edit in progress or a new change restarting the wait, `solve_now` solving a pending change at once, the free variable's own value ignored, the solved, unreachable and refused outcomes with their status lines) and the panel's (the spec's "sizing result display": the solved design shown; no solve per frame and a repaint after one; a drag defers the solve; the locked free-variable row; leaving the mode keeps the value as one undo step, and leaving while "Solving..." writes the current solution; typing in the results search does not defer the solve, typing a part name does; the JSON export holds the inputs that produced the results; an unreachable target shows the best value and its space-claim overshoot; the free-variable picker, with the ring radius's locked row shown while Coupling is closed; a share link carries the sizing state; every text has glyphs), and the timed frame helper.

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
mod tests {
    use super::*;
    use crate::engine::api::HEADLINE;
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::input_ui::{CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::session::encode_share_payload;
    use crate::gui::test_support::{
        SCREEN, drawn_texts, key_event, key_tap, primary_button, select_all, short_magnets,
        sized_frame, text_rect,
    };
    use crate::headline;

```

with:

```rust
mod tests {
    use super::*;
    use crate::engine::api::HEADLINE;
    use crate::engine::sizing::SizingOutcome;
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::input_ui::{CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
    use crate::gui::session::encode_share_payload;
    use crate::gui::sizing::DEBOUNCE_S;
    use crate::gui::test_support::{
        SCREEN, drawn_texts, key_event, key_tap, primary_button, select_all, short_magnets,
        sized_frame, sized_frame_at, text_rect,
    };
    use crate::headline;

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        pub(crate) fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
            let panel = &mut self.panel;
            sized_frame(&self.ctx, self.screen, events, |ui| panel.ui(ui))
        }

        /// The Key design row's main widget as drawn in the last frame.
```

with:

```rust
        pub(crate) fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
            let panel = &mut self.panel;
            sized_frame(&self.ctx, self.screen, events, |ui| panel.ui(ui))
        }

        /// A frame `dt` seconds after the last one.
        fn frame_after(&mut self, dt: f64, events: Vec<egui::Event>) -> egui::FullOutput {
            let time = self.ctx.input(|i| i.time) + dt;
            let panel = &mut self.panel;
            sized_frame_at(&self.ctx, self.screen, Some(time), events, |ui| {
                panel.ui(ui)
            })
        }

        /// Switches to Torque \u{2192} Magnets and lets the solve run.
        fn size(&mut self) -> egui::FullOutput {
            self.click_text(SizingMode::TorqueToMagnets.label());
            self.frame_after(DEBOUNCE_S + 0.01, Vec::new());
            self.frame(Vec::new());
            self.frame(Vec::new())
        }

        /// The solved point of the last solve.
        fn solved(&self) -> crate::engine::sizing::SizingPoint {
            match self.panel.runner.outcome(self.panel.sizing.variable) {
                Some(Ok(SizingOutcome::Solved(point))) => point.clone(),
                other => panic!("not solved: {other:?}"),
            }
        }

        /// The Key design row's main widget as drawn in the last frame.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
        let ctx = egui::Context::default();
```

with:

```rust
    }

    #[test]
    fn every_text_the_panel_shows_has_glyphs_in_the_default_fonts() {
        // egui draws an empty box for a character its default fonts lack (U+2192, the
        // spec's arrow, is one): every drawn text, in each state, and every hover text.
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 12000.0));
        let mut texts = drawn_texts(&harness.frame(Vec::new()));
        texts.extend(drawn_texts(&harness.size()));
        for group in &InputCatalogue::get().groups {
            harness.click_text(group.label);
        }
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.panel.inputs = short_magnets();
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        let catalogue = InputCatalogue::get();
        texts.extend(catalogue.all().map(crate::gui::inputs::input_tooltip));
        let results = harness.panel.results().clone();
        texts.extend(
            crate::gui::dashboard::dashboard_lines(&results)
                .into_iter()
                .map(|line| line.tooltip),
        );
        for entry in crate::gui::results_table::table_entries() {
            let value = results.get(&entry.path).unwrap_or(Value::None);
            texts.push(crate::gui::results_table::row_tooltip(entry, &value));
        }
        let font = egui::FontId::proportional(14.0);
        for text in &texts {
            for c in text.chars().filter(|c| !c.is_whitespace()) {
                assert!(
                    harness.ctx.fonts(|f| f.has_glyph(&font, c)),
                    "no glyph for {c:?} (U+{:04X}) in {text:?}",
                    c as u32
                );
            }
        }
    }

    #[test]
    fn torque_to_magnets_shows_the_solved_design() {
        let mut harness = Harness::new();
        let output = harness.size();
        let point = harness.solved();
        // The axial length (the default free variable) meets the 2.5 N\u{b7}m target.
        assert!((point.torque_hot_low_Nm - 2.5).abs() < 1e-6);
        let status = format!(
            "Solved at {} mm (hot-low torque 2.500 N\u{b7}m)",
            format_value(&Value::Num(point.value))
        );
        assert_eq!(count(&output, &status), 1, "{:?}", drawn_texts(&output));
        // The panel shows the sized design; the inputs keep their own (blank) length.
        assert_eq!(harness.panel.shown_inputs(), point.inputs);
        assert_eq!(harness.panel.results(), &compute_all(&point.inputs));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_drew_headline(&output, &point.inputs);
    }

    #[test]
    fn the_solve_waits_for_the_debounce_and_never_runs_per_frame() {
        let mut harness = Harness::new();
        harness.click_text(SizingMode::TorqueToMagnets.label());
        for _ in 0..5 {
            harness.frame_after(0.02, Vec::new());
        }
        assert_eq!(harness.panel.runner.solves, 0, "still waiting");
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SOLVING), 1);
        let solved = harness.frame_after(DEBOUNCE_S, Vec::new());
        assert_eq!(harness.panel.runner.solves, 1);
        // The solve asks for one more frame, so the inputs side shows its outcome.
        let repaint = solved.viewport_output[&egui::ViewportId::ROOT].repaint_delay;
        assert_eq!(repaint, std::time::Duration::ZERO);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SOLVING), 0);
        for _ in 0..10 {
            harness.frame_after(0.1, Vec::new());
        }
        assert_eq!(harness.panel.runner.solves, 1, "idle frames never solve");
    }

    #[test]
    fn a_drag_in_progress_defers_the_solve() {
        let mut harness = Harness::new();
        harness.size();
        assert_eq!(harness.panel.runner.solves, 1);
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        // A pointer held down (a drag) over empty space of the centre region.
        let blank = egui::pos2(700.0, 1000.0);
        harness.frame(vec![egui::Event::PointerMoved(blank)]);
        harness.frame(vec![primary_button(blank, true)]);
        for _ in 0..5 {
            harness.frame_after(0.2, Vec::new());
        }
        assert_eq!(
            harness.panel.runner.solves, 1,
            "not while the button is down"
        );
        harness.frame_after(0.01, vec![primary_button(blank, false)]);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert_eq!(harness.panel.runner.solves, 2);
    }

    #[test]
    fn the_free_variable_s_row_shows_the_solved_value_locked() {
        let mut harness = Harness::new();
        let output = harness.size();
        let point = harness.solved();
        assert_eq!(count(&output, SIZED_NOTE), 1);
        assert!(!harness.widget(AXIAL_LENGTH).enabled(), "locked");
        // Its value box shows the solved length.
        let shown = format!("{:.2} mm", point.value);
        assert!(count(&output, &shown) >= 1, "{shown}");
        // Arrow keys on it change nothing.
        let id = harness.widget(AXIAL_LENGTH).id;
        harness.ctx.memory_mut(|m| m.request_focus(id));
        harness.frame(key_tap(egui::Key::ArrowRight));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn leaving_torque_to_magnets_while_solving_writes_the_current_solution() {
        // Decision M41-7: the value written into the inputs is this design's solution, even
        // when the user leaves before the debounce ran out ("Solving...").
        let mut harness = Harness::new();
        harness.size();
        harness.panel.sizing.target_Nm = 3.0;
        assert_eq!(count(&harness.frame(Vec::new()), SOLVING), 1);
        harness.click_text(SizingMode::MagnetsToTorque.label());
        assert_eq!(harness.panel.sizing().mode, SizingMode::MagnetsToTorque);
        assert_eq!(harness.panel.runner.solves, 2, "solved on leaving");
        let Ok(SizingOutcome::Solved(point)) =
            crate::engine::sizing::solve(&DesignInputs::default(), FreeVariable::AxialLength, 3.0)
        else {
            panic!("3.0 N·m is reachable by the axial length")
        };
        assert_eq!(harness.panel.inputs(), &point.inputs);
        assert!((point.torque_hot_low_Nm - 3.0).abs() < 1e-6);
    }

    #[test]
    fn typing_in_the_results_search_does_not_defer_the_solve() {
        let mut harness = Harness::new();
        harness.size();
        harness.panel.sizing.target_Nm = 3.0;
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("pull".to_owned())]);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert!(
            focused_text_field(&harness.ctx).is_some(),
            "typing in the search"
        );
        assert_eq!(harness.panel.runner.solves, 2, "the search edits no design");
        // A text field of the design does defer it.
        harness.panel.sizing.target_Nm = 3.5;
        harness.focus(PART_INNER);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert_eq!(
            harness.panel.runner.solves, 2,
            "not while a part name is typed"
        );
    }

    #[test]
    fn the_json_export_in_torque_to_magnets_holds_the_inputs_that_produced_the_results() {
        use crate::gui::results_table::EXPORT_JSON;
        let mut harness = Harness::new();
        harness.size();
        harness.click_text(EXPORT_JSON);
        let requests = harness.panel.take_requests();
        let [PanelRequest::SaveFile { contents, .. }] = &requests[..] else {
            panic!("one export: {requests:?}")
        };
        let json: serde_json::Value = serde_json::from_str(contents).unwrap();
        let design = design_from_json(&json["design"].to_string()).unwrap();
        assert_eq!(design.inputs, harness.panel.shown_inputs());
        assert_eq!(&compute_all(&design.inputs), harness.panel.results());
        assert_ne!(
            &design.inputs,
            harness.panel.inputs(),
            "the solved length, not the blank override"
        );
        assert_eq!(design.sizing, *harness.panel.sizing());
    }

    #[test]
    fn leaving_torque_to_magnets_keeps_the_sized_value_as_one_undo_step() {
        let mut harness = Harness::new();
        harness.size();
        let point = harness.solved();
        let steps = harness.panel.history.undo_len();
        harness.click_text(SizingMode::MagnetsToTorque.label());
        assert_eq!(harness.panel.sizing().mode, SizingMode::MagnetsToTorque);
        assert_eq!(harness.panel.inputs(), &point.inputs, "decision M41-7");
        assert_eq!(harness.panel.results(), &compute_all(&point.inputs));
        assert_eq!(harness.panel.history.undo_len(), steps + 1);
        harness.panel.undo();
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn an_unreachable_target_shows_the_best_value_and_its_overshoot() {
        let mut harness = Harness::new();
        harness.size();
        harness.panel.sizing.target_Nm = 50.0;
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, "Not reachable (best 9.937 N\u{b7}m at 50.80 mm)"),
            1
        );
        assert_eq!(
            harness
                .panel
                .shown_inputs()
                .coupling
                .magnets
                .axial_length_mm,
            Some(50.8),
            "decision M41-8: the best value"
        );
        assert!(
            drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with("Exceeds the space claim:"))
        );
    }

    #[test]
    fn the_free_variable_picker_switches_the_variable() {
        let mut harness = Harness::new();
        harness.size();
        harness.click_text(variable_label(FreeVariable::AxialLength));
        harness.click_text(variable_label(FreeVariable::RingRadius));
        assert_eq!(harness.panel.sizing().variable, FreeVariable::RingRadius);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let point = harness.solved();
        // The ring radius is not in the Key design group: its locked row shows there anyway,
        // with the Coupling group closed.
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SIZED_NOTE), 1);
        let label = InputCatalogue::get()
            .entry(FreeVariable::RingRadius.path())
            .unwrap()
            .meta
            .label;
        assert_eq!(count(&output, label), 1);
        assert_eq!(
            harness.panel.shown_inputs().coupling.inner_back_apothem_mm,
            point.value
        );
        assert_eq!(
            harness
                .panel
                .shown_inputs()
                .coupling
                .magnets
                .axial_length_mm,
            None,
            "the axial length is no longer sized"
        );
    }

    #[test]
    fn a_share_link_carries_the_sizing_state() {
        let mut harness = Harness::new();
        harness.size();
        harness.panel.sizing.target_Nm = 3.0;
        let link = harness.panel.share_link();
        let payload = link.split("?m=").nth(1).unwrap();
        let mut other = Harness::new();
        other.panel.load_share_payload(payload).unwrap();
        assert_eq!(other.panel.design(), harness.panel.design());
        other.frame(Vec::new());
        other.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert_eq!(other.panel.shown_inputs(), harness.panel.shown_inputs());
        assert_eq!(other.panel.results(), harness.panel.results());
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
        let ctx = egui::Context::default();
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/sizing.rs`, replace:

```rust
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DesignInputs;

    #[test]
    fn every_mode_and_free_variable_round_trips_through_its_key() {
```

with:

```rust
    }
}

/// How long the design must stay unchanged before a solve runs [s].
pub const DEBOUNCE_S: f64 = 0.25;

/// The start of the status line of a solved design.
pub const SOLVED_PREFIX: &str = "Solved at ";

/// The start of the status line when the target is out of reach.
pub const NOT_REACHABLE_PREFIX: &str = "Not reachable";

/// The status line while a solve waits for the design to settle.
pub const SOLVING: &str = "Solving...";

/// What a solve depends on: the inputs (with the free variable cleared, since the solve sets
/// it), the free variable and the target.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the engine's names
struct SolveKey {
    inputs: DesignInputs,
    variable: FreeVariable,
    target_Nm: f64,
}

impl SolveKey {
    fn new(inputs: &DesignInputs, state: &SizingState) -> Self {
        Self {
            inputs: state.variable.apply(inputs, 0.0),
            variable: state.variable,
            target_Nm: state.target_Nm,
        }
    }
}

/// Runs inverse sizing for the panel: debounced, never per frame.
#[derive(Clone, Debug, Default)]
pub struct SizingRunner {
    /// The last solve: what it was for, and its outcome.
    last: Option<(SolveKey, Result<SizingOutcome, SizingError>)>,
    /// A changed design waiting to settle, and when it was last seen changing [s, egui time].
    pending: Option<(SolveKey, f64)>,
    /// How many solves ran (the tests check that a solve never runs per frame).
    pub(crate) solves: usize,
}

impl SizingRunner {
    /// Called once per frame in Torque → Magnets mode with the frame's time and whether an
    /// edit is in progress. Solves when the design has been unchanged for [`DEBOUNCE_S`] and no
    /// edit is in progress. Returns how long to wait before the next frame must run for a
    /// pending solve [s], or `None` when nothing is pending.
    pub fn update(
        &mut self,
        inputs: &DesignInputs,
        state: &SizingState,
        now: f64,
        editing: bool,
    ) -> Option<f64> {
        let key = SolveKey::new(inputs, state);
        if self.last.as_ref().is_some_and(|(last, _)| *last == key) {
            self.pending = None;
            return None;
        }
        let since = match &self.pending {
            Some((pending, since)) if *pending == key && !editing => *since,
            _ => {
                self.pending = Some((key, now));
                return Some(DEBOUNCE_S);
            }
        };
        let waited = now - since;
        if waited < DEBOUNCE_S {
            return Some(DEBOUNCE_S - waited);
        }
        self.solve_now(inputs, state);
        None
    }

    /// Solves this design now, whatever the debounce: the panel calls it once when the user
    /// leaves Torque → Magnets before a change was solved, so the value written into the
    /// inputs is this design's solution, not the last one's (decision M41-7). Never per frame.
    pub fn solve_now(&mut self, inputs: &DesignInputs, state: &SizingState) {
        let outcome = solve(inputs, state.variable, state.target_Nm);
        self.solves += 1;
        self.pending = None;
        self.last = Some((SolveKey::new(inputs, state), outcome));
    }

    /// Whether the last solve is for this design (nothing pending).
    pub fn is_current(&self, inputs: &DesignInputs, state: &SizingState) -> bool {
        self.last
            .as_ref()
            .is_some_and(|(key, _)| *key == SolveKey::new(inputs, state))
    }

    /// The last outcome, if the last solve was for the same free variable (it may be stale:
    /// see [`SizingRunner::is_current`]).
    pub fn outcome(&self, variable: FreeVariable) -> Option<&Result<SizingOutcome, SizingError>> {
        match &self.last {
            Some((key, outcome)) if key.variable == variable => Some(outcome),
            _ => None,
        }
    }

    /// The free variable's value the panel shows: the solved value, or the best valid value
    /// when the target is out of reach; `None` before the first solve of this variable, when
    /// no value is valid, or when the solve refused.
    pub fn value(&self, variable: FreeVariable) -> Option<f64> {
        match self.outcome(variable)? {
            Ok(SizingOutcome::Solved(point)) => Some(point.value),
            Ok(SizingOutcome::NotReachable { best: Some(best) }) => Some(best.value),
            _ => None,
        }
    }

    /// The design the panel shows in Torque → Magnets: `inputs` with the free variable at
    /// [`SizingRunner::value`], else `inputs` as they are.
    pub fn shown(&self, inputs: &DesignInputs, state: &SizingState) -> DesignInputs {
        match self.value(state.variable) {
            Some(value) => state.variable.apply(inputs, value),
            None => inputs.clone(),
        }
    }

    /// The status line under the sizing controls.
    pub fn status(&self, inputs: &DesignInputs, state: &SizingState) -> String {
        if !self.is_current(inputs, state) {
            return SOLVING.to_owned();
        }
        match self.outcome(state.variable) {
            Some(Ok(SizingOutcome::Solved(point))) => format!(
                "{SOLVED_PREFIX}{} (hot-low torque {})",
                variable_text(state.variable, point.value),
                torque_text(point.torque_hot_low_Nm)
            ),
            Some(Ok(SizingOutcome::NotReachable { best: Some(best) })) => format!(
                "{NOT_REACHABLE_PREFIX} (best {} at {})",
                torque_text(best.torque_hot_low_Nm),
                variable_text(state.variable, best.value)
            ),
            Some(Ok(SizingOutcome::NotReachable { best: None })) => format!(
                "{NOT_REACHABLE_PREFIX}: no value of the free variable gives a valid design"
            ),
            Some(Err(SizingError::InvalidTarget(target))) => format!(
                "Sizing refused: the target {} is not a positive torque",
                torque_text(*target)
            ),
            Some(Err(SizingError::InvalidInputs(errors))) => {
                let paths: Vec<&str> = errors.iter().map(|e| e.path.as_str()).collect();
                format!("Sizing refused: invalid inputs ({})", paths.join(", "))
            }
            None => SOLVING.to_owned(),
        }
    }
}

/// A free variable's value as the status line shows it, with its input's unit.
fn variable_text(variable: FreeVariable, value: f64) -> String {
    let shown = match variable {
        FreeVariable::MagnetsPerRing => Value::Int(value as i64),
        _ => Value::Num(value),
    };
    let unit = crate::gui::inputs::InputCatalogue::get()
        .entry(variable.path())
        .map_or("", |entry| entry.meta.unit);
    with_unit(format_value(&shown), unit)
}

/// A torque as the status line shows it.
#[allow(non_snake_case)] // unit suffix
fn torque_text(torque_Nm: f64) -> String {
    with_unit(format_value(&Value::Num(torque_Nm)), "N·m")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mode_and_free_variable_round_trips_through_its_key() {
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/sizing.rs`, replace:

```rust
            DesignInputs::default().metal.required_min_Nm
        );
    }
}
```

with:

```rust
            DesignInputs::default().metal.required_min_Nm
        );
    }

    #[allow(non_snake_case)] // unit suffix
    fn inverse(variable: FreeVariable, target_Nm: f64) -> SizingState {
        SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable,
            target_Nm,
        }
    }

    #[test]
    fn a_solve_waits_for_the_debounce_and_runs_once() {
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength, 2.5);
        let mut runner = SizingRunner::default();
        assert_eq!(
            runner.update(&inputs, &state, 10.0, false),
            Some(DEBOUNCE_S)
        );
        assert_eq!(runner.status(&inputs, &state), SOLVING);
        let wait = runner.update(&inputs, &state, 10.1, false).unwrap();
        assert!((wait - 0.15).abs() < 1e-12, "{wait}");
        assert_eq!(runner.solves, 0);
        assert_eq!(runner.update(&inputs, &state, 10.25, false), None);
        assert_eq!(runner.solves, 1);
        for frame in 0..10 {
            assert_eq!(
                runner.update(&inputs, &state, 10.3 + frame as f64, false),
                None
            );
        }
        assert_eq!(runner.solves, 1, "never per frame");
        assert!(runner.is_current(&inputs, &state));
    }

    #[test]
    fn an_edit_in_progress_or_a_new_change_restarts_the_wait() {
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength, 2.5);
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, 0.0, false);
        // A drag: still editing past the debounce, so nothing runs.
        runner.update(&inputs, &state, 1.0, true);
        assert_eq!(runner.solves, 0);
        // Released: the wait starts again from the release.
        runner.update(&inputs, &state, 1.1, false);
        runner.update(&inputs, &state, 1.2, false);
        assert_eq!(runner.solves, 0);
        inputs.metal.face_gap_mm = 1.5;
        runner.update(&inputs, &state, 1.4, false);
        runner.update(&inputs, &state, 1.5, false);
        assert_eq!(runner.solves, 0, "a new change restarts the wait");
        runner.update(&inputs, &state, 1.65, false);
        assert_eq!(runner.solves, 1);
    }

    #[test]
    fn solve_now_solves_a_pending_change_at_once() {
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength, 2.5);
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, 0.0, false);
        runner.update(&inputs, &state, 1.0, false);
        inputs.metal.face_gap_mm = 1.5;
        assert_eq!(runner.update(&inputs, &state, 1.1, false), Some(DEBOUNCE_S));
        assert!(
            !runner.is_current(&inputs, &state),
            "waiting for the debounce"
        );
        runner.solve_now(&inputs, &state);
        assert_eq!(runner.solves, 2);
        assert!(runner.is_current(&inputs, &state));
        let Some(Ok(SizingOutcome::Solved(point))) = runner.outcome(state.variable) else {
            panic!("2.5 N·m is reachable at a 1.5 mm gap")
        };
        assert_eq!(
            point.inputs.metal.face_gap_mm, 1.5,
            "this design's solution"
        );
        // Nothing is left pending: the next frame does not solve again.
        assert_eq!(runner.update(&inputs, &state, 1.2, false), None);
        assert_eq!(runner.solves, 2);
    }

    #[test]
    fn the_free_variable_s_own_value_does_not_trigger_a_solve() {
        // The solve sets the free variable, so its value in the inputs is not part of the key.
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength, 2.5);
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, 0.0, false);
        runner.update(&inputs, &state, 1.0, false);
        inputs.coupling.magnets.axial_length_mm = Some(30.0);
        assert!(runner.is_current(&inputs, &state));
        assert_eq!(runner.update(&inputs, &state, 2.0, false), None);
        assert_eq!(runner.solves, 1);
    }

    /// The runner after a solve of `state` at the defaults.
    fn solved(state: &SizingState) -> SizingRunner {
        let mut runner = SizingRunner::default();
        let inputs = DesignInputs::default();
        runner.update(&inputs, state, 0.0, false);
        runner.update(&inputs, state, 1.0, false);
        runner
    }

    #[test]
    fn a_solved_design_shows_the_solved_value() {
        let state = inverse(FreeVariable::AxialLength, 2.5);
        let runner = solved(&state);
        let inputs = DesignInputs::default();
        let Some(Ok(SizingOutcome::Solved(point))) = runner.outcome(state.variable) else {
            panic!("2.5 N·m is reachable by the axial length")
        };
        assert_eq!(runner.value(state.variable), Some(point.value));
        assert_eq!(runner.shown(&inputs, &state), point.inputs);
        assert_eq!(
            runner.status(&inputs, &state),
            format!(
                "Solved at {} mm (hot-low torque 2.500 N·m)",
                format_value(&Value::Num(point.value))
            )
        );
        // Another free variable has no outcome yet.
        assert_eq!(runner.value(FreeVariable::RingRadius), None);
    }

    #[test]
    fn an_unreachable_target_shows_the_best_value() {
        let state = inverse(FreeVariable::AxialLength, 50.0);
        let runner = solved(&state);
        let inputs = DesignInputs::default();
        assert_eq!(runner.value(state.variable), Some(50.8));
        assert_eq!(
            runner
                .shown(&inputs, &state)
                .coupling
                .magnets
                .axial_length_mm,
            Some(50.8)
        );
        assert_eq!(
            runner.status(&inputs, &state),
            "Not reachable (best 9.937 N·m at 50.80 mm)"
        );
        let poles = inverse(FreeVariable::MagnetsPerRing, 2.5);
        assert_eq!(
            solved(&poles).status(&inputs, &poles),
            "Not reachable (best 2.285 N·m at 10)"
        );
    }

    #[test]
    fn a_refused_solve_shows_why_and_keeps_the_inputs() {
        let state = inverse(FreeVariable::RingRadius, -1.0);
        let runner = solved(&state);
        let inputs = DesignInputs::default();
        assert_eq!(runner.value(state.variable), None);
        assert_eq!(runner.shown(&inputs, &state), inputs);
        assert_eq!(
            runner.status(&inputs, &state),
            "Sizing refused: the target -1.000 N·m is not a positive torque"
        );
        let mut bad = DesignInputs::default();
        bad.coupling.backiron = 7;
        let state = inverse(FreeVariable::RingRadius, 2.5);
        let mut runner = SizingRunner::default();
        runner.update(&bad, &state, 0.0, false);
        runner.update(&bad, &state, 1.0, false);
        assert_eq!(
            runner.status(&bad, &state),
            "Sizing refused: invalid inputs (coupling.backiron)"
        );
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/test_support.rs`, replace:

```rust
    ctx: &egui::Context,
    size: egui::Vec2,
    events: Vec<egui::Event>,
    mut draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    let input = egui::RawInput {
        events,
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
        ..Default::default()
    };
    ctx.run(input, |ctx| {
```

with:

```rust
    ctx: &egui::Context,
    size: egui::Vec2,
    events: Vec<egui::Event>,
    draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    sized_frame_at(ctx, size, None, events, draw)
}

/// [`sized_frame`] at the input time `time` [s] (`None`: egui adds its predicted frame time,
/// 1/60 s, to the last frame's).
pub(crate) fn sized_frame_at(
    ctx: &egui::Context,
    size: egui::Vec2,
    time: Option<f64>,
    events: Vec<egui::Event>,
    mut draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    let input = egui::RawInput {
        events,
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
        time,
        ..Default::default()
    };
    ctx.run(input, |ctx| {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | sort -u | head -8
```

Expected: the build fails with compile errors naming the items Step 3 adds (the runner's tests use its imports), among them ``error[E0412]: cannot find type `DesignInputs` in this scope``, ``error[E0433]: failed to resolve: use of undeclared type `SizingOutcome` `` and ``error[E0425]: cannot find function `format_value` in this scope``.

- [ ] **Step 3: Write the runner and the sizing controls**



In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml`, replace:

```toml
# The egui panel `gui::MagcouplingPanel`, hostable by any egui app (the
# standalone app, M4, and a window in linkage-sim-rs, M5). egui only: no
# windowing, no eframe, no file dialogs. serde_json, base64 and flate2 write
# and read design files, share links and the results export.
gui = ["dep:egui", "dep:serde_json", "dep:base64", "dep:flate2"]
# The standalone app: `app::MagcouplingApp` and the binaries `magcoupling-app`
# (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen).
app = [
```

with:

```toml
# The egui panel `gui::MagcouplingPanel`, hostable by any egui app (the
# standalone app, M4, and a window in linkage-sim-rs, M5). egui only: no
# windowing, no eframe, no file dialogs. serde_json, base64 and flate2 write
# and read design files, share links and the results export; log reports the
# sizing outcome (the host chooses the logger: the browser console on the web).
gui = ["dep:egui", "dep:log", "dep:serde_json", "dep:base64", "dep:flate2"]
# The standalone app: `app::MagcouplingApp` and the binaries `magcoupling-app`
# (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen).
app = [
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
//! host may open a design as the session's start ([`MagcouplingPanel::open_share_payload`]:
//! no undo step) and may keep the undo keys for itself
//! ([`MagcouplingPanel::set_keyboard_shortcuts`]).

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::gui::dashboard::dashboard_ui;
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row};
use crate::gui::inputs::{InputCatalogue, InputEntry, optional_seed};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
```

with:

```rust
//! host may open a design as the session's start ([`MagcouplingPanel::open_share_payload`]:
//! no undo step) and may keep the undo keys for itself
//! ([`MagcouplingPanel::set_keyboard_shortcuts`]).
//!
//! Sizing (spec Addendum A1): the mode switch tops the Key design group. In Torque → Magnets
//! the panel shows the design with the free variable at the solved value
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame), and the free variable's
//! row shows that value, locked.

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::{Level, dashboard_ui};
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    Design, LoadError, PUBLIC_BASE_URL, decode_share_payload, design_from_json, design_to_json,
    share_link,
};
use crate::gui::sizing::SizingState;
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
```

with:

```rust
    Design, LoadError, PUBLIC_BASE_URL, decode_share_payload, design_from_json, design_to_json,
    share_link,
};
use crate::gui::sizing::{
    SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, TARGET_RANGE_INPUT,
    variable_label,
};
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust

/// The heading of the Key design group.
pub const KEY_DESIGN_HEADING: &str = "Key design";

/// Starting width of the inputs side [points].
const INPUTS_WIDTH: f32 = 320.0;
```

with:

```rust

/// The heading of the Key design group.
pub const KEY_DESIGN_HEADING: &str = "Key design";

/// The label of the target torque slider (Torque → Magnets).
pub const TARGET_LABEL: &str = "Target hot-low torque";

/// The label of the free-variable picker.
pub const FREE_VARIABLE_LABEL: &str = "Free variable";

/// The note under the free variable's row in Torque → Magnets.
pub const SIZED_NOTE: &str = "Set by Torque -> Magnets";

/// Starting width of the inputs side [points].
const INPUTS_WIDTH: f32 = 320.0;
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    inputs: DesignInputs,
    /// The sizing mode, free variable and target torque.
    sizing: SizingState,
    /// Undo and redo of the design (inputs and sizing state).
    history: History<Design>,
    /// The address share links point at.
    share_base: String,
    /// What the last session action did, until the next one.
    status: Option<String>,
    /// The results of `inputs`, recomputed by every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// The main widget of each Key design row in the last frame, by input path.
    key_widgets: Vec<(&'static str, egui::Id)>,
```

with:

```rust
    inputs: DesignInputs,
    /// The sizing mode, free variable and target torque.
    sizing: SizingState,
    /// Runs inverse sizing in Torque → Magnets.
    runner: SizingRunner,
    /// Undo and redo of the design (inputs and sizing state).
    history: History<Design>,
    /// The address share links point at.
    share_base: String,
    /// What the last session action did, until the next one.
    status: Option<String>,
    /// The results of the design shown ([`MagcouplingPanel::shown_inputs`]), recomputed by
    /// every [`MagcouplingPanel::ui`].
    results: DesignResults,
    /// The main widget of each Key design row in the last frame, by input path.
    key_widgets: Vec<(&'static str, egui::Id)>,
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        Self {
            inputs,
            sizing: SizingState::default(),
            history: History::new(Design::default()),
            share_base: PUBLIC_BASE_URL.to_owned(),
            status: None,
```

with:

```rust
        Self {
            inputs,
            sizing: SizingState::default(),
            runner: SizingRunner::default(),
            history: History::new(Design::default()),
            share_base: PUBLIC_BASE_URL.to_owned(),
            status: None,
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    fn set_design(&mut self, design: Design) {
        self.inputs = design.inputs;
        self.sizing = design.sizing;
        self.results = compute_all(&self.inputs);
        self.last_error = None;
    }

    /// Sets the address share links point at: the page's own address on the web (so a link
```

with:

```rust
    fn set_design(&mut self, design: Design) {
        self.inputs = design.inputs;
        self.sizing = design.sizing;
        self.results = compute_all(&self.shown_inputs());
        self.last_error = None;
    }

    /// The design the panel shows: the inputs, or in Torque → Magnets the inputs with the free
    /// variable at the solved value (the best value when the target is out of reach).
    pub fn shown_inputs(&self) -> DesignInputs {
        match self.sizing.mode {
            SizingMode::MagnetsToTorque => self.inputs.clone(),
            SizingMode::TorqueToMagnets => self.runner.shown(&self.inputs, &self.sizing),
        }
    }

    /// The sizing state.
    pub fn sizing(&self) -> &SizingState {
        &self.sizing
    }

    /// Sets the address share links point at: the page's own address on the web (so a link
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        &self.inputs
    }

    /// The results of [`MagcouplingPanel::inputs`], as of the last frame or reset.
    pub fn results(&self) -> &DesignResults {
        &self.results
    }
```

with:

```rust
        &self.inputs
    }

    /// The results of the design shown ([`MagcouplingPanel::shown_inputs`]), as of the last
    /// frame or edit.
    pub fn results(&self) -> &DesignResults {
        &self.results
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
                .resizable(true)
                .default_width(INPUTS_WIDTH)
                .show_inside(ui, |ui| self.inputs_ui(ui));
            // After the inputs: the readouts show this frame's edits.
            self.results = compute_all(&self.inputs);
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
```

with:

```rust
                .resizable(true)
                .default_width(INPUTS_WIDTH)
                .show_inside(ui, |ui| self.inputs_ui(ui));
            self.run_sizing(ui);
            // After the inputs: the readouts show this frame's edits.
            self.results = compute_all(&self.shown_inputs());
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
        let design_text =
            focused_text_field(ui.ctx()).is_some_and(|id| self.design_widgets.contains(&id));
        design_text || ui.input(|i| i.pointer.any_down() || !i.keys_down.is_empty())
    }

    /// Ctrl+Z and Ctrl+Shift+Z or Ctrl+Y, unless the host keeps them
```

with:

```rust
        let design_text =
            focused_text_field(ui.ctx()).is_some_and(|id| self.design_widgets.contains(&id));
        design_text || ui.input(|i| i.pointer.any_down() || !i.keys_down.is_empty())
    }

    /// In Torque → Magnets, lets the runner solve if the design has settled, asks for a frame
    /// when a solve is pending, and logs each outcome.
    fn run_sizing(&mut self, ui: &egui::Ui) {
        if self.sizing.mode != SizingMode::TorqueToMagnets {
            return;
        }
        let now = ui.input(|i| i.time);
        let solves = self.runner.solves;
        if let Some(wait) = self
            .runner
            .update(&self.inputs, &self.sizing, now, self.editing(ui))
        {
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_secs_f64(wait));
        }
        if self.runner.solves != solves {
            self.log_sizing();
            // The inputs side was drawn before the solve: one more frame shows its outcome
            // there (an idle page would otherwise keep showing "Solving...").
            ui.ctx().request_repaint();
        }
    }

    /// Logs the outcome of the last solve (the web smoke reads it).
    fn log_sizing(&self) {
        log::info!(
            "magcoupling sizing: {}",
            self.runner.status(&self.inputs, &self.sizing)
        );
    }

    /// The sizing controls at the top of the Key design group: the mode switch, then in
    /// Torque → Magnets the free variable, the target torque and the outcome.
    fn sizing_ui(&mut self, ui: &mut egui::Ui) {
        let mut mode = self.sizing.mode;
        ui.horizontal(|ui| {
            for choice in SizingMode::ALL {
                ui.selectable_value(&mut mode, choice, choice.label());
            }
        });
        if mode != self.sizing.mode {
            if mode == SizingMode::MagnetsToTorque {
                // Decision M41-7: leaving inverse sizing keeps the value it shows, solved for
                // this design: a change still waiting for its debounce is solved first.
                if !self.runner.is_current(&self.inputs, &self.sizing) {
                    self.runner.solve_now(&self.inputs, &self.sizing);
                    self.log_sizing();
                }
                self.inputs = self.runner.shown(&self.inputs, &self.sizing);
            }
            self.sizing.mode = mode;
        }
        if self.sizing.mode != SizingMode::TorqueToMagnets {
            return;
        }
        egui::ComboBox::from_label(FREE_VARIABLE_LABEL)
            .selected_text(variable_label(self.sizing.variable))
            .show_ui(ui, |ui| {
                for variable in FreeVariable::ALL {
                    ui.selectable_value(
                        &mut self.sizing.variable,
                        variable,
                        variable_label(variable),
                    );
                }
            });
        let entry = InputCatalogue::get()
            .entry(TARGET_RANGE_INPUT)
            .expect("the target's metadata input exists");
        let range = entry.meta.range.expect("the hot minimum has a slider");
        ui.horizontal(|ui| {
            ui.label(TARGET_LABEL).on_hover_text(
                "The hot-low torque with production variation (metal.torque_hot_low_Nm) the free variable must reach",
            );
            let mut target = self.sizing.target_Nm;
            let response = ui.add(slider(&mut target, entry.meta, range));
            // Its value box is a text field of the design, as an input row's is.
            self.design_widgets.push(response.id);
            if response.changed() && target != self.sizing.target_Nm {
                self.sizing.target_Nm = target;
            }
        });
        let status = self.runner.status(&self.inputs, &self.sizing);
        let color = if status.starts_with(SOLVED_PREFIX) {
            Level::Good.color(ui.visuals())
        } else if status == SOLVING {
            ui.visuals().weak_text_color()
        } else {
            Level::Bad.color(ui.visuals())
        };
        ui.colored_label(color, status);
        // The free variable's row, locked at the value shown, when the Key design group does
        // not list it (the ring radius is in the Coupling group, closed by default).
        let path = self.sizing.variable.path();
        if !KEY_DESIGN.contains(&path) {
            let entry = InputCatalogue::get()
                .entry(path)
                .expect("every free variable is an input");
            let widget = self.input_row_ui(ui, entry);
            self.design_widgets.push(widget);
        }
        ui.separator();
    }

    /// Ctrl+Z and Ctrl+Shift+Z or Ctrl+Y, unless the host keeps them
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
                    Some(TableAction::ExportJson) => self.requests.push(PanelRequest::SaveFile {
                        file_name: JSON_FILE_NAME.to_owned(),
                        mime: "application/json",
                        contents: results_json(&self.design(), &self.results),
                    }),
                    None => {}
                }
```

with:

```rust
                    Some(TableAction::ExportJson) => self.requests.push(PanelRequest::SaveFile {
                        file_name: JSON_FILE_NAME.to_owned(),
                        mime: "application/json",
                        // The design that produced the results: in Torque -> Magnets the
                        // inputs with the free variable at the value shown.
                        contents: results_json(
                            &Design {
                                inputs: self.shown_inputs(),
                                sizing: self.sizing,
                            },
                            &self.results,
                        ),
                    }),
                    None => {}
                }
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
                    .id_salt("key_design")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.key_widgets.clear();
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry);
```

with:

```rust
                    .id_salt("key_design")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.sizing_ui(ui);
                        self.key_widgets.clear();
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry);
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
            });
    }

    /// One input row; applies its edit. Returns the id of its main widget.
    fn input_row_ui(&mut self, ui: &mut egui::Ui, entry: &'static InputEntry) -> egui::Id {
        let current = self.inputs.get(&entry.path).unwrap_or(Value::None);
        let seed = self.seed(entry);
        let output = input_row(ui, entry, &current, seed);
        if let Some(edit) = output.edit {
            let value = match edit {
                RowEdit::Set(value) => value,
```

with:

```rust
            });
    }

    /// One input row; applies its edit. Returns the id of its main widget. In Torque →
    /// Magnets the free variable's row shows the value the panel shows, locked.
    fn input_row_ui(&mut self, ui: &mut egui::Ui, entry: &'static InputEntry) -> egui::Id {
        let locked = self.sizing.mode == SizingMode::TorqueToMagnets
            && entry.path == self.sizing.variable.path();
        // The locked row reads the design shown (one clone, for that row only).
        let current = if locked {
            self.shown_inputs().get(&entry.path)
        } else {
            self.inputs.get(&entry.path)
        }
        .unwrap_or(Value::None);
        let seed = self.seed(entry);
        let output = ui
            .add_enabled_ui(!locked, |ui| input_row(ui, entry, &current, seed))
            .inner;
        if locked {
            ui.weak(SIZED_NOTE);
            return output.widget.id;
        }
        if let Some(edit) = output.edit {
            let value = match edit {
                RowEdit::Set(value) => value,
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/sizing.rs`, replace:

```rust
//!
//! The mode, the free variable and the target torque are GUI state (A-2 decision A2-9): the
//! engine takes them as arguments. Design files and share links record them (decision M41-4).

use crate::engine::sizing::FreeVariable;

/// Which way the calculator runs (spec A1 "Mode switch").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

with:

```rust
//!
//! The mode, the free variable and the target torque are GUI state (A-2 decision A2-9): the
//! engine takes them as arguments. Design files and share links record them (decision M41-4).
//!
//! [`SizingRunner`] runs the solve: on a change, once the design has been still for
//! [`DEBOUNCE_S`] and no edit is in progress (no drag, no typing), never per frame (a solve
//! takes about 1 to 3 ms in a release build, longer on wasm, against compute_all's 17 us). Until
//! then the panel keeps showing the last solved value, marked as solving. The design the panel
//! shows in Torque → Magnets is the inputs with the free variable at the solved value, or at
//! the best value when the target is not reachable (decision M41-8); the inputs keep their own
//! value of the free variable until the user leaves the mode (decision M41-7), when a change
//! still waiting for its debounce is solved at once ([`SizingRunner::solve_now`]).

use crate::DesignInputs;
use crate::engine::meta::Value;
use crate::engine::sizing::{FreeVariable, SizingError, SizingOutcome, solve};
use crate::gui::format::{format_value, with_unit};

/// Which way the calculator runs (spec A1 "Mode switch").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

- [ ] **Step 4: Run the tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED|panicked"
```

Expected: `test result: ok. 268 passed; 0 failed` (Task 7's 249 and 19 new tests).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task8.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task8.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task8.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task8.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/Cargo.toml
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/sizing.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/test_support.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
feat(magcoupling-rs): the sizing mode switch with a debounced inverse sizing

Magnets -> Torque and Torque -> Magnets in the Key design group (Addendum A1):
free variable, target torque (M41-9), sizing::solve once the design is still
for 0.25 s and no edit is in progress, never per frame; the design shown at the
solved or best value (M41-8) with its row locked, kept on leaving, a pending
change solved first (M41-7); each outcome logged; the JSON export holds the
design shown. A glyph test guards every text against egui's default fonts.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 9: The app: theme, files and the share-link entry

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

Spec M4 "Session": "Theme matches the linkage app". `app/theme.rs` copies `cad_dark_visuals` from `linkage-sim-rs/src/gui/theme.rs` (a test compares the two function bodies, so they cannot drift) and applies it with the linkage app's spacing, forcing `ThemePreference::Dark` so the system's light preference changes nothing (decision M41-3; the linkage web app's own light rendering is backlog BL-013); the page's background becomes the same panel colour, `#1e2026`. `app/files.rs` does the panel's requests: saving through an rfd dialog natively and as a Blob download on the web (as `linkage-sim-rs/src/gui/export/download.rs`), picking a design file through rfd (asynchronously on the web, the text arriving in an inbox the app reads each frame). The web entry reads `?m=` and opens its design as the session's start (`open_share_payload`: no undo step), logging `magcoupling: loaded the design from the share link` (a refused link is logged as a warning and shown in the panel), and points share links at the page's own address. A picked design file is logged as `magcoupling: loaded a design file` (`DESIGN_FILE_LOADED`; Task 10's smoke reads it, since the panel's own status is drawn on the canvas only). `MagcouplingPanel::report` shows what a save did.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml` (feature `app`: rfd, js-sys; the web-sys features)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.lock` (by cargo: adds rfd 0.15.4 and its transitive crates)
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app/theme.rs`
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app/files.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/bin/magcoupling_web.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs` (`report`; a test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/web/magcoupling/index.html` (the background colour)

**Interfaces:**
- Consumes: Task 7's `MagcouplingPanel::{load_design_file, open_share_payload, set_share_base, design, take_requests}`, Task 6's `PanelRequest`, Task 1's `SHARE_PARAM`, `LoadError`, Task 8's `sizing()`.
- Produces (`magcoupling::app`): `pub mod theme` (`fn cad_dark_visuals() -> egui::Visuals`, `const ITEM_SPACING`, `const BUTTON_PADDING`, `fn apply(&egui::Context)`), `pub mod files` (`fn save(file_name: &str, mime: &str, contents: &str) -> Option<Result<String, String>>`, `struct DesignPicker` with `pick(&mut self, &egui::Context)` and `take(&mut self) -> Option<Result<String, String>>`); `MagcouplingApp::new(&egui::Context) -> Self`, `panel(&self) -> &MagcouplingPanel`, `set_share_base`, `open_share_payload(&mut self, &str) -> Result<(), LoadError>`, private `open_picked_file(&mut self, Result<String, String>)`; `const SHARE_LINK_LOADED: &str = "magcoupling: loaded the design from the share link"`, `const DESIGN_FILE_LOADED: &str = "magcoupling: loaded a design file"`.
- Produces (`MagcouplingPanel`): `fn report(&mut self, outcome: Result<String, String>)`.

- [ ] **Step 1: Write the failing tests**

The theme module's docs and tests (the visuals equal linkage-sim-rs's, body for body; dark whatever the system prefers; the page background), the app's tests (the full page in the CAD theme; a share link opening its design as the session's start, so the first Ctrl+Z keeps it, and a broken one changing nothing; a picked design file loading and a refused one changing nothing), and the panel's `report` test. `app.rs` declares the theme module now, so its tests compile and fail.

Create `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app/theme.rs`:

```rust
//! The standalone app's theme: the linkage app's (spec M4 "Session": "Theme matches the
//! linkage app"). The linkage app applies its CAD dark visuals at start-up whatever the
//! system's light or dark preference; so does this app (decision M41-3), and the web page's
//! background is the same panel colour.
//!
//! The visuals are a copy of `cad_dark_visuals` in `linkage-sim-rs/src/gui/theme.rs` (the
//! crates are separate: linkage-sim-rs will depend on this one in M5, not the other way
//! round). A test compares the two function bodies, so they cannot drift apart. The panel
//! itself sets no theme: in M5 the linkage app's own theme applies.

#[cfg(test)]
mod tests {
    use super::*;

    /// The body of `fn cad_dark_visuals`, from its signature to the closing brace.
    fn body(source: &str) -> &str {
        let start = source
            .find("fn cad_dark_visuals() -> egui::Visuals {")
            .expect("the function is there");
        let end = source[start..].find("\n}\n").expect("it ends") + start;
        &source[start..end]
    }

    #[test]
    fn the_visuals_are_the_linkage_app_s() {
        let linkage = include_str!("../../../linkage-sim-rs/src/gui/theme.rs");
        let ours = include_str!("theme.rs");
        assert_eq!(body(ours), body(linkage));
        // And the spacing of its apply_cad_theme.
        assert!(linkage.contains("style.spacing.item_spacing = egui::vec2(6.0, 4.0);"));
        assert!(linkage.contains("style.spacing.button_padding = egui::vec2(8.0, 4.0);"));
        assert_eq!(
            (ITEM_SPACING, BUTTON_PADDING),
            (egui::vec2(6.0, 4.0), egui::vec2(8.0, 4.0))
        );
    }

    #[test]
    fn the_theme_is_dark_whatever_the_system_prefers() {
        let ctx = egui::Context::default();
        ctx.options_mut(|o| o.fallback_theme = egui::Theme::Light);
        apply(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        assert_eq!(ctx.theme(), egui::Theme::Dark);
        assert_eq!(ctx.style().visuals, cad_dark_visuals());
        assert_eq!(ctx.style().spacing.item_spacing, ITEM_SPACING);
        // The system turning light changes nothing: the preference is dark.
        let light = egui::RawInput {
            system_theme: Some(egui::Theme::Light),
            ..Default::default()
        };
        let _ = ctx.run(light, |_| {});
        assert_eq!(ctx.theme(), egui::Theme::Dark);
    }

    #[test]
    fn the_web_page_background_is_the_panel_colour() {
        let page = include_str!("../../../linkage-sim-rs/web/magcoupling/index.html");
        let [r, g, b, _] = cad_dark_visuals().panel_fill.to_array();
        let colour = format!("background: #{r:02x}{g:02x}{b:02x};");
        assert!(page.contains(&colour), "index.html has no {colour}");
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs`, replace:

```rust
//! The standalone app (feature `app`): [`MagcouplingApp`] shows the panel as a
//! full page, for the native binary `magcoupling-app` and the web binary
//! `magcoupling-web` (served at `/magcoupling/`).

use crate::gui::MagcouplingPanel;

```

with:

```rust
//! The standalone app (feature `app`): [`MagcouplingApp`] shows the panel as a
//! full page, for the native binary `magcoupling-app` and the web binary
//! `magcoupling-web` (served at `/magcoupling/`).

pub mod theme;

use crate::gui::MagcouplingPanel;

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs`, replace:

```rust
mod tests {
    use super::*;
    use crate::engine::meta::Value;
    use crate::gui::test_support::drawn_texts;
    use crate::{DesignInputs, compute_all, headline};

    #[test]
    fn the_app_shows_the_panel_as_a_full_page() {
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::default();
        let texts = drawn_texts(&ctx.run(egui::RawInput::default(), |ctx| app.ui(ctx)));
        assert!(
            texts.iter().any(|t| t == "Magnetic coupling calculator"),
            "{texts:?}"
        );
        // The clamp screw is the last headline row: the whole panel was laid out.
        let (key, screw) = headline(&compute_all(&DesignInputs::default()))
            .pop()
            .expect("15 rows");
```

with:

```rust
mod tests {
    use super::*;
    use crate::engine::meta::Value;
    use crate::gui::session::{Design, encode_share_payload};
    use crate::gui::test_support::drawn_texts;
    use crate::{DesignInputs, compute_all, headline};

    #[test]
    fn the_app_shows_the_panel_as_a_full_page() {
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(input(), |ctx| app.ui(ctx));
        let texts = drawn_texts(&ctx.run(input(), |ctx| app.ui(ctx)));
        assert!(
            texts.iter().any(|t| t == "Magnetic coupling calculator"),
            "{texts:?}"
        );
        // The clamp screw is the last headline row: the dashboard was laid out.
        let (key, screw) = headline(&compute_all(&DesignInputs::default()))
            .pop()
            .expect("15 rows");
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs`, replace:

```rust
            panic!("{key}: {screw:?}")
        };
        assert!(texts.contains(&screw), "missing {screw:?} in {texts:?}");
    }

    #[test]
```

with:

```rust
            panic!("{key}: {screw:?}")
        };
        assert!(texts.contains(&screw), "missing {screw:?} in {texts:?}");
        assert_eq!(ctx.style().visuals, theme::cad_dark_visuals());
    }

    #[test]
    fn a_share_link_opens_its_design() {
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 2.0;
        app.open_share_payload(&encode_share_payload(&design))
            .unwrap();
        assert_eq!(app.panel().design(), design);
        // The session starts there: the first Ctrl+Z keeps the shared design.
        for pressed in [true, false] {
            let undo = egui::Event::Key {
                key: egui::Key::Z,
                physical_key: None,
                pressed,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            };
            let input = egui::RawInput {
                events: vec![undo],
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| app.ui(ctx));
        }
        assert_eq!(app.panel().design(), design);
        assert!(app.open_share_payload("broken").is_err());
        assert_eq!(
            app.panel().design(),
            design,
            "a broken link changes nothing"
        );
    }

    #[test]
    fn a_picked_design_file_loads_and_a_refused_one_changes_nothing() {
        use crate::gui::session::design_to_json;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 2.0;
        app.open_picked_file(Ok(design_to_json(&design)));
        assert_eq!(app.panel().design(), design);
        app.open_picked_file(Ok("not a design".to_owned()));
        assert_eq!(app.panel().design(), design);
        app.open_picked_file(Err("the file is not UTF-8 text".to_owned()));
        assert_eq!(app.panel().design(), design);
    }

    #[test]
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    }

    #[test]
    fn a_broken_share_link_changes_nothing_and_says_why() {
        let mut panel = MagcouplingPanel::new();
        let error = panel.load_share_payload("not-a-design").unwrap_err();
```

with:

```rust
    }

    #[test]
    fn the_host_reports_what_a_save_did() {
        let mut harness = Harness::new();
        harness
            .panel
            .report(Ok("Saved C:/designs/a.json".to_owned()));
        harness.frame(Vec::new());
        assert_eq!(
            count(&harness.frame(Vec::new()), "Saved C:/designs/a.json"),
            1
        );
        harness
            .panel
            .report(Err("Could not save: disk full".to_owned()));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Could not save: disk full"), 1);
        assert_eq!(count(&output, "Saved C:/designs/a.json"), 0);
    }

    #[test]
    fn a_broken_share_link_changes_nothing_and_says_why() {
        let mut panel = MagcouplingPanel::new();
        let error = panel.load_share_payload("not-a-design").unwrap_err();
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | sort -u | head -8
```

Expected: the build fails with compile errors naming the items Step 3 adds, among them ``error[E0425]: cannot find function `cad_dark_visuals` in this scope``, ``error[E0425]: cannot find value `ITEM_SPACING` in this scope``, ``error[E0599]: no function or associated item named `new` found for struct `app::MagcouplingApp` `` and ``error[E0599]: no method named `report` found ...``.

- [ ] **Step 3: Write the theme, the files and the web entry**

The dependencies first (cargo resolves rfd against crates.io on the next build; never `--offline`).

Create `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app/files.rs`:

```rust
//! The standalone app's file work, for the panel's requests (`gui::PanelRequest`): saving a
//! file (a design, a results export) and picking a design file to load. Natively a file dialog
//! (rfd); on the web a browser download, and rfd's file picker. The same pattern as
//! `linkage-sim-rs/src/gui/export/download.rs`.

/// Saves `contents`: natively to the file the user picks, on the web as a download. Returns
/// the message to show, or `None` when the user cancelled.
pub fn save(file_name: &str, mime: &str, contents: &str) -> Option<Result<String, String>> {
    save_impl(file_name, mime, contents)
}

#[cfg(not(target_arch = "wasm32"))]
fn save_impl(file_name: &str, _mime: &str, contents: &str) -> Option<Result<String, String>> {
    let extension = file_name.rsplit_once('.').map_or("", |(_, ext)| ext);
    let path = rfd::FileDialog::new()
        .set_file_name(file_name)
        .add_filter(extension, &[extension])
        .save_file()?;
    Some(
        std::fs::write(&path, contents)
            .map(|()| format!("Saved {}", path.display()))
            .map_err(|error| format!("Could not save {}: {error}", path.display())),
    )
}

#[cfg(target_arch = "wasm32")]
fn save_impl(file_name: &str, mime: &str, contents: &str) -> Option<Result<String, String>> {
    Some(download(file_name, mime, contents.as_bytes()).map(|()| format!("Downloaded {file_name}")))
}

/// A browser download: a Blob behind a transient `<a download>` that is clicked.
#[cfg(target_arch = "wasm32")]
fn download(file_name: &str, mime: &str, contents: &[u8]) -> Result<(), String> {
    use wasm_bindgen::JsCast;

    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let parts = js_sys::Array::new();
    parts.push(&js_sys::Uint8Array::from(contents).into());
    let bag = web_sys::BlobPropertyBag::new();
    bag.set_type(mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts.into(), &bag)
        .map_err(|e| format!("Blob failed: {e:?}"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| format!("object URL failed: {e:?}"))?;
    let anchor = document
        .create_element("a")
        .map_err(|e| format!("no <a>: {e:?}"))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| "not an <a>")?;
    anchor.set_href(&url);
    anchor.set_download(file_name);
    let _ = anchor.style().set_property("display", "none");
    if let Some(body) = document.body() {
        let _ = body.append_child(&anchor);
        anchor.click();
        let _ = body.remove_child(&anchor);
    } else {
        anchor.click();
    }
    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}

/// A design file the user is picking: natively the dialog returns at once; on the web the
/// picker answers later, so the text arrives in an inbox the app reads each frame.
#[derive(Default)]
pub struct DesignPicker {
    #[cfg(target_arch = "wasm32")]
    inbox: std::rc::Rc<std::cell::RefCell<Option<Result<String, String>>>>,
    #[cfg(not(target_arch = "wasm32"))]
    inbox: Option<Result<String, String>>,
}

impl DesignPicker {
    /// Shows the file picker (JSON files); the text arrives in [`DesignPicker::take`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pick(&mut self, _ctx: &egui::Context) {
        let picked = rfd::FileDialog::new()
            .add_filter("Design", &["json"])
            .pick_file();
        self.inbox = picked.map(|path| {
            std::fs::read_to_string(&path)
                .map_err(|error| format!("Could not read {}: {error}", path.display()))
        });
    }

    /// Shows the file picker (JSON files); the text arrives in [`DesignPicker::take`] once the
    /// browser hands the file over.
    #[cfg(target_arch = "wasm32")]
    pub fn pick(&mut self, ctx: &egui::Context) {
        let inbox = self.inbox.clone();
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

    /// The picked file's text (or why it could not be read), once.
    pub fn take(&mut self) -> Option<Result<String, String>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inbox.take()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.inbox.borrow_mut().take()
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml`, replace:

```toml
gui = ["dep:egui", "dep:log", "dep:serde_json", "dep:base64", "dep:flate2"]
# The standalone app: `app::MagcouplingApp` and the binaries `magcoupling-app`
# (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen).
app = [
    "gui",
    "dep:eframe",
    "dep:log",
    "dep:env_logger",
    "dep:wasm-bindgen",
    "dep:wasm-bindgen-futures",
    "dep:web-sys",
]
# TEST-ONLY. Exposes the switch that turns the approved workbook corrections
# (the deviation registry) off, so parity and differential tests compare
```

with:

```toml
gui = ["dep:egui", "dep:log", "dep:serde_json", "dep:base64", "dep:flate2"]
# The standalone app: `app::MagcouplingApp` and the binaries `magcoupling-app`
# (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen).
# rfd: the native save and open dialogs, and the web file picker.
app = [
    "gui",
    "dep:eframe",
    "dep:log",
    "dep:env_logger",
    "dep:rfd",
    "dep:wasm-bindgen",
    "dep:wasm-bindgen-futures",
    "dep:web-sys",
    "dep:js-sys",
]
# TEST-ONLY. Exposes the switch that turns the approved workbook corrections
# (the deviation registry) off, so parity and differential tests compare
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml`, replace:

```toml
serde_json = { version = "1", optional = true, features = ["float_roundtrip"] }
base64 = { version = "0.22", optional = true }
flate2 = { version = "1", optional = true }

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
env_logger = { version = "0.11", optional = true }

# The web entry. wasm-bindgen must equal the wasm-bindgen-cli version that
# deploy-web.yml installs (0.2.114), pinned in Cargo.lock; gate.sh checks it.
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = { version = "0.2", optional = true }
wasm-bindgen-futures = { version = "0.4", optional = true }
web-sys = { version = "0.3", optional = true, features = ["Document", "Element", "HtmlCanvasElement", "Window"] }

[dev-dependencies]
magcoupling-rs = { path = ".", features = ["workbook-parity"] }
```

with:

```toml
serde_json = { version = "1", optional = true, features = ["float_roundtrip"] }
base64 = { version = "0.22", optional = true }
flate2 = { version = "1", optional = true }
# File dialogs of the standalone app (feature app), the version linkage-sim-rs
# uses: native save and open, and the web file picker.
rfd = { version = "0.15", optional = true }

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
env_logger = { version = "0.11", optional = true }

# The web entry. wasm-bindgen must equal the wasm-bindgen-cli version that
# deploy-web.yml installs (0.2.114), pinned in Cargo.lock; gate.sh checks it.
# web-sys and js-sys: the canvas, the share link in the page's address
# (Location, UrlSearchParams) and file downloads (Blob, an <a download>).
[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = { version = "0.2", optional = true }
wasm-bindgen-futures = { version = "0.4", optional = true }
web-sys = { version = "0.3", optional = true, features = [
    "Blob",
    "BlobPropertyBag",
    "CssStyleDeclaration",
    "Document",
    "Element",
    "HtmlAnchorElement",
    "HtmlCanvasElement",
    "HtmlElement",
    "Location",
    "Url",
    "UrlSearchParams",
    "Window",
] }
js-sys = { version = "0.3", optional = true }

[dev-dependencies]
magcoupling-rs = { path = ".", features = ["workbook-parity"] }
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/web/magcoupling/index.html`, replace:

```html
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #1e1e23;
        }
        canvas {
            width: 100%;
```

with:

```html
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #1e2026;
        }
        canvas {
            width: 100%;
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs`, replace:

```rust
//! The standalone app (feature `app`): [`MagcouplingApp`] shows the panel as a
//! full page, for the native binary `magcoupling-app` and the web binary
//! `magcoupling-web` (served at `/magcoupling/`).

pub mod theme;

use crate::gui::MagcouplingPanel;

/// Window and page title.
pub const TITLE: &str = "Magnetic Coupling Calculator";
```

with:

```rust
//! The standalone app (feature `app`): [`MagcouplingApp`] shows the panel as a full page,
//! for the native binary `magcoupling-app` and the web binary `magcoupling-web` (served at
//! `/magcoupling/`). It applies the linkage app's theme ([`theme`]) and does the panel's
//! platform work ([`files`]): saving files and picking a design file. The web entry also
//! opens a share link's design (`?m=`).

pub mod files;
pub mod theme;

use crate::gui::session::LoadError;
use crate::gui::{MagcouplingPanel, PanelRequest};

/// Window and page title.
pub const TITLE: &str = "Magnetic Coupling Calculator";
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs`, replace:

```rust
/// must use the same.
pub const CANVAS_ID: &str = "magcoupling_canvas";

/// The standalone app: one [`MagcouplingPanel`], full page.
#[derive(Default)]
pub struct MagcouplingApp {
    panel: MagcouplingPanel,
}

impl MagcouplingApp {
    /// Draws one frame: the panel fills the window (its sides scroll on their own).
    pub fn ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| self.panel.ui(ui));
    }
}

```

with:

```rust
/// must use the same.
pub const CANVAS_ID: &str = "magcoupling_canvas";

/// The log line of a share link opened at start-up (the web smoke test looks for it).
pub const SHARE_LINK_LOADED: &str = "magcoupling: loaded the design from the share link";

/// The log line of a design file loaded through the file picker (the web smoke test looks for
/// it: the panel's own "Design file loaded" is drawn on the canvas only).
pub const DESIGN_FILE_LOADED: &str = "magcoupling: loaded a design file";

/// The standalone app: one [`MagcouplingPanel`], full page.
#[derive(Default)]
pub struct MagcouplingApp {
    panel: MagcouplingPanel,
    picker: files::DesignPicker,
}

impl MagcouplingApp {
    /// The app at the default design, with the linkage app's theme applied to `ctx`.
    pub fn new(ctx: &egui::Context) -> Self {
        theme::apply(ctx);
        Self::default()
    }

    /// The panel.
    pub fn panel(&self) -> &MagcouplingPanel {
        &self.panel
    }

    /// Sets the address share links point at (the web page's own address).
    pub fn set_share_base(&mut self, base: impl Into<String>) {
        self.panel.set_share_base(base);
    }

    /// Opens the design of a share link's `?m=` value as the session's start (no undo step:
    /// the first Undo keeps the shared design), and logs the outcome.
    pub fn open_share_payload(&mut self, payload: &str) -> Result<(), LoadError> {
        let outcome = self.panel.open_share_payload(payload);
        match &outcome {
            Ok(()) => log::info!(
                "{SHARE_LINK_LOADED} (sizing: {})",
                self.panel.sizing().mode.label()
            ),
            Err(error) => log::warn!("magcoupling: {error}"),
        }
        outcome
    }

    /// Draws one frame: the panel fills the window (its sides scroll on their own); then the
    /// panel's requests are done.
    pub fn ui(&mut self, ctx: &egui::Context) {
        if let Some(picked) = self.picker.take() {
            self.open_picked_file(picked);
        }
        egui::CentralPanel::default().show(ctx, |ui| self.panel.ui(ui));
        for request in self.panel.take_requests() {
            match request {
                PanelRequest::SaveFile {
                    file_name,
                    mime,
                    contents,
                } => {
                    if let Some(outcome) = files::save(&file_name, mime, &contents) {
                        self.panel.report(outcome);
                    }
                }
                PanelRequest::OpenDesign => self.picker.pick(ctx),
            }
        }
    }
}

impl MagcouplingApp {
    /// Hands a picked design file's text to the panel (a refusal shows there) and logs the
    /// outcome; a file that could not be read is reported.
    fn open_picked_file(&mut self, picked: Result<String, String>) {
        match picked {
            Ok(text) => match self.panel.load_design_file(&text) {
                Ok(()) => log::info!("{DESIGN_FILE_LOADED}"),
                Err(error) => log::warn!("magcoupling: {error}"),
            },
            Err(error) => self.panel.report(Err(error)),
        }
    }
}

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs`, replace:

```rust
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_title(TITLE),
        ..Default::default()
    };
    eframe::run_native(
        TITLE,
        options,
        Box::new(|_cc| Ok(Box::new(MagcouplingApp::default()))),
    )
}

```

with:

```rust
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title(TITLE),
        ..Default::default()
    };
    eframe::run_native(
        TITLE,
        options,
        Box::new(|cc| Ok(Box::new(MagcouplingApp::new(&cc.egui_ctx)))),
    )
}

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app/theme.rs`, replace:

```rust
//! crates are separate: linkage-sim-rs will depend on this one in M5, not the other way
//! round). A test compares the two function bodies, so they cannot drift apart. The panel
//! itself sets no theme: in M5 the linkage app's own theme applies.

#[cfg(test)]
mod tests {
```

with:

```rust
//! crates are separate: linkage-sim-rs will depend on this one in M5, not the other way
//! round). A test compares the two function bodies, so they cannot drift apart. The panel
//! itself sets no theme: in M5 the linkage app's own theme applies.

/// Build the professional dark visuals inspired by CAD tools (SolidWorks, ANSYS).
pub fn cad_dark_visuals() -> egui::Visuals {
    let mut v = egui::Visuals::dark();

    // Darker, more professional background tones
    v.panel_fill = egui::Color32::from_rgb(30, 32, 38);
    v.window_fill = egui::Color32::from_rgb(35, 37, 44);
    v.extreme_bg_color = egui::Color32::from_rgb(20, 22, 28);
    v.faint_bg_color = egui::Color32::from_rgb(38, 40, 48);

    // Accent color for selections and interactions
    v.selection.bg_fill = egui::Color32::from_rgb(40, 100, 200);
    v.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255));

    // Widget styling — more rounded, cleaner
    v.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(42, 44, 52);
    v.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 62, 72));

    v.widgets.inactive.bg_fill = egui::Color32::from_rgb(50, 52, 62);
    v.widgets.inactive.bg_stroke = egui::Stroke::new(0.5, egui::Color32::from_rgb(70, 72, 82));

    v.widgets.hovered.bg_fill = egui::Color32::from_rgb(60, 65, 80);
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 140, 220));

    v.widgets.active.bg_fill = egui::Color32::from_rgb(40, 100, 200);
    v.widgets.active.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255));

    // Separator and window stroke
    v.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(55, 58, 68));

    v
}

/// The linkage app's spacing (`apply_cad_theme`): item spacing and button padding [points].
pub const ITEM_SPACING: egui::Vec2 = egui::vec2(6.0, 4.0);
pub const BUTTON_PADDING: egui::Vec2 = egui::vec2(8.0, 4.0);

/// Applies the theme: dark whatever the system prefers, with the CAD visuals and the linkage
/// app's spacing.
pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.visuals = cad_dark_visuals();
        style.spacing.item_spacing = ITEM_SPACING;
        style.spacing.button_padding = BUTTON_PADDING;
    });
}

#[cfg(test)]
mod tests {
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/bin/magcoupling_web.rs`, replace:

```rust
}

/// Starts the app in the canvas `magcoupling::app::CANVAS_ID`; called by the
/// wasm-bindgen glue when the module loads.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub async fn start() {
    use magcoupling::app::{CANVAS_ID, MagcouplingApp};
    use wasm_bindgen::JsCast;

    // Route log macros to the browser console.
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let canvas = web_sys::window()
        .expect("no window")
        .document()
        .expect("no document")
        .get_element_by_id(CANVAS_ID)
```

with:

```rust
}

/// Starts the app in the canvas `magcoupling::app::CANVAS_ID`; called by the
/// wasm-bindgen glue when the module loads. A `?m=` share link in the page's
/// address opens its design; share links made here point at this page.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub async fn start() {
    use magcoupling::app::{CANVAS_ID, MagcouplingApp};
    use magcoupling::gui::session::SHARE_PARAM;
    use wasm_bindgen::JsCast;

    // Route log macros to the browser console.
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let window = web_sys::window().expect("no window");
    let location = window.location();
    let base = format!(
        "{}{}",
        location.origin().unwrap_or_default(),
        location.pathname().unwrap_or_default()
    );
    let payload = location
        .search()
        .ok()
        .and_then(|search| web_sys::UrlSearchParams::new_with_str(&search).ok())
        .and_then(|params| params.get(SHARE_PARAM))
        .filter(|payload| !payload.is_empty());

    let canvas = window
        .document()
        .expect("no document")
        .get_element_by_id(CANVAS_ID)
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/bin/magcoupling_web.rs`, replace:

```rust
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|_cc| Ok(Box::new(MagcouplingApp::default()))),
        )
        .await
        .expect("Failed to start eframe");
```

with:

```rust
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(move |cc| {
                let mut app = MagcouplingApp::new(&cc.egui_ctx);
                app.set_share_base(base);
                if let Some(payload) = payload {
                    // A refused link is logged and shown in the panel; the default design stays.
                    let _ = app.open_share_payload(&payload);
                }
                Ok(Box::new(app))
            }),
        )
        .await
        .expect("Failed to start eframe");
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
                self.status = None;
                Err(error)
            }
        }
    }

```

with:

```rust
                self.status = None;
                Err(error)
            }
        }
    }

    /// Shows what a host's work for a request did: a message in the header, or why it failed.
    pub fn report(&mut self, outcome: Result<String, String>) {
        match outcome {
            Ok(message) => {
                self.status = Some(message);
                self.last_error = None;
            }
            Err(error) => self.last_error = Some(error),
        }
    }

```

- [ ] **Step 4: Run the tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED|panicked"
```

Expected: `test result: ok. 274 passed; 0 failed` (Task 8's 268 and 6 new tests). Then `git -C C:/Users/Cole/source/repos/lsim-mag-m41 diff --stat magcoupling-rs/Cargo.lock` lists the lock's new entries (rfd 0.15.4, ashpd, async-fs, async-net, block2, futures-channel, pollster, ppv-lite86, rand, rand_chacha, rand_core, urlencoding) and no changed version of a crate that was already locked (the only other changed lines are dependency lists that now name `block2` with its version, since two versions are locked); egui, eframe and wasm-bindgen stay at 0.32.3, 0.32.3 and 0.2.114 (gate 11 checks).

- [ ] **Step 5: Check the web build compiles**

Run:

```bash
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
```

Expected: `Finished ...` with no warning (rfd's file picker, the Blob download and the `?m=` entry compile for wasm32; gate 9 runs the same).

- [ ] **Step 6: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task9.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task9.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task9.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task9.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add linkage-sim-rs/web/magcoupling/index.html
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/Cargo.lock
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/Cargo.toml
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/app.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/app/files.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/app/theme.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/bin/magcoupling_web.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
feat(magcoupling-rs): the app's theme, file dialogs and share-link entry

The linkage app's CAD dark visuals and spacing, forced dark (decision M41-3; a
test keeps the visuals equal to linkage-sim-rs's), and the page background to
match. The panel's requests done natively (rfd dialogs) and on the web (a Blob
download, rfd's picker; a picked file is logged); the web entry opens a ?m=
share link as the session's start and points new links at the page itself.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 10: The web smoke of /magcoupling/

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model. The browser smoke is a GUI smoke run (also `sonnet`).

Spec "Testing and validation summary": "`gui-smoke` extended to `/magcoupling/`". The page is one canvas, so the smoke loads deterministic state through a share link and checks the console: `.claude/workflows/gui-smoke.js` gains a second agent step that opens `/magcoupling/?m=<payload>` (the default design with the face gap at 1.5 mm, in Torque -> Magnets sizing the axial length to 2.5 N·m), then clicks the "Load design" button once (its centre read off a screenshot, the step's only canvas click) and uploads a pinned design file through rfd's web overlay. It passes only with the canvas present, the three log lines (`magcoupling: loaded the design from the share link`, `magcoupling sizing: Solved at`, `magcoupling: loaded a design file`) and zero console errors; if the button cannot be found or the overlay does not appear after two attempts, it reports why (the canvas click is the fragile part; the share-link checks do not depend on it). The payload and the design file live once, in the workflow; a Rust test reads both from there, decodes them, checks the designs and runs the app to the solve, so they cannot go stale unnoticed (the payload carries every input, so a changed default or a removed input path fails that test until the link is regenerated: the workflow comment and the README say how). The panel's log prefix becomes a constant the test shares.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/.claude/workflows/gui-smoke.js`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs` (a test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs` (`SIZING_LOG_PREFIX`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs` (re-export it)

**Interfaces:**
- Consumes: Task 9's `MagcouplingApp::{new, open_share_payload, ui, panel}`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; Task 8's `SOLVED_PREFIX`, `shown_inputs`; Task 1's `decode_share_payload`, `design_from_json`.
- Produces: `gui::SIZING_LOG_PREFIX: &str = "magcoupling sizing: "`; the workflow's `MAGCOUPLING_SMOKE_DESIGN_FILE` and `MAGCOUPLING_TOOLS`; the `gui-smoke` workflow's `magcoupling` step (its result adds `design_file_loaded`) (args: `{"magcoupling": false}` skips it, `{"linkage": false}` skips the linkage step; the result keeps the linkage step's fields at the top level, with `linkage` and `magcoupling` beside them and `passed` for both).

- [ ] **Step 1: Write the failing test**

The app test that reads the workflow's payload and design file (the workflow itself changes in Step 3; the test fails first on the missing constant).

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/app.rs`, replace:

```rust
    }

    #[test]
    fn the_canvas_id_matches_the_web_page() {
        let page = include_str!("../../linkage-sim-rs/web/magcoupling/index.html");
        assert!(
```

with:

```rust
    }

    #[test]
    fn the_smoke_test_share_link_opens_its_design() {
        use crate::engine::sizing::FreeVariable;
        use crate::gui::SIZING_LOG_PREFIX;
        use crate::gui::session::{decode_share_payload, design_from_json};
        use crate::gui::sizing::{SOLVED_PREFIX, SizingMode, SizingState};
        // .claude/workflows/gui-smoke.js opens /magcoupling/ with this link, loads the design
        // file through the picker and looks for the three log lines in the browser console.
        let script = include_str!("../../.claude/workflows/gui-smoke.js");
        let quoted = |name: &str| -> &str {
            let marker = format!("const {name} = '");
            let start = script.find(&marker).unwrap_or_else(|| panic!("no {name}")) + marker.len();
            let end = start + script[start..].find('\'').expect("its closing quote");
            &script[start..end]
        };
        let payload = quoted("MAGCOUPLING_SMOKE_PAYLOAD");
        let design = decode_share_payload(payload).expect("a valid share link");
        let mut want = Design::default();
        want.inputs.metal.face_gap_mm = 1.5;
        want.sizing = SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable: FreeVariable::AxialLength,
            target_Nm: 2.5,
        };
        assert_eq!(design, want);
        assert!(script.contains(SHARE_LINK_LOADED));
        let solved = format!("{SIZING_LOG_PREFIX}{}", SOLVED_PREFIX.trim_end());
        assert!(script.contains(&solved), "{solved}");
        let file =
            design_from_json(quoted("MAGCOUPLING_SMOKE_DESIGN_FILE")).expect("a design file");
        let mut want_file = Design::default();
        want_file.inputs.metal.face_gap_mm = 2.0;
        assert_eq!(file, want_file);
        assert!(script.contains(DESIGN_FILE_LOADED));
        // The app opens the link and the solve succeeds after its debounce.
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        app.open_share_payload(payload).unwrap();
        for time in [0.0, 0.5, 0.6] {
            let input = egui::RawInput {
                time: Some(time),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| app.ui(ctx));
        }
        let shown = app.panel().shown_inputs();
        let length = shown.coupling.magnets.axial_length_mm.expect("sized");
        assert!((14.0..14.5).contains(&length), "{length}");
    }

    #[test]
    fn the_canvas_id_matches_the_web_page() {
        let page = include_str!("../../linkage-sim-rs/web/magcoupling/index.html");
        assert!(
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/mod.rs`, replace:

```rust

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
pub use panel::{CentreView, MagcouplingPanel, PanelRequest};
```

with:

```rust

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
pub use panel::{CentreView, MagcouplingPanel, PanelRequest, SIZING_LOG_PREFIX};
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | sort -u | head -8
```

Expected: the library fails to build: ``error[E0432]: unresolved import `panel::SIZING_LOG_PREFIX` ``.

- [ ] **Step 3: Add the constant and the workflow step**



In `C:/Users/Cole/source/repos/lsim-mag-m41/.claude/workflows/gui-smoke.js`, replace:

```js
export const meta = {
  name: 'gui-smoke',
  description: 'Smoke-test the locally served WASM build: loads, canvas present, console clean',
  whenToUse: 'Before merging a batch that touched src/gui/, and during Phase 1 audit',
  phases: [{ title: 'Smoke' }],
}

```

with:

```js
export const meta = {
  name: 'gui-smoke',
  description: 'Smoke-test the locally served WASM builds (the linkage app and /magcoupling/): loads, canvas present, console clean',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
  phases: [{ title: 'Smoke' }],
}

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/.claude/workflows/gui-smoke.js`, replace:

```js
  },
}

const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
if (ARGS.selftest) { return { ok: true } }

const url = ARGS.url || 'http://localhost:8080'
const result = await agent(
  `Smoke-test the WASM linkage app at ${url} using Playwright MCP tools (load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close").
Steps: navigate to ${url}; wait 5 seconds for WASM init; snapshot the page and confirm a <canvas> element exists; collect console messages; take a screenshot and judge whether it shows a rendered app (menu bar / toolbar / panels visible, any theme) vs a blank page; close the browser.
passed=true only if: page loaded, canvas present, zero console messages of type error (warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (caller must have run scripts/serve_web.sh).`,
  { label: 'gui-smoke', phase: 'Smoke', schema: SMOKE_SCHEMA, model: 'sonnet' },
)
return result || { passed: false, canvas_present: false, console_errors: [], notes: 'smoke agent returned no result (agent error)' }
```

with:

```js
  },
}

const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'share_link_loaded', 'sizing_solved', 'design_file_loaded'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    share_link_loaded: { type: 'boolean' },
    sizing_solved: { type: 'boolean' },
    design_file_loaded: { type: 'boolean' },
    screenshot_note: { type: 'string' },
    notes: { type: 'string' },
  },
}

// The design the /magcoupling/ smoke opens through its share link (?m=): the default design
// with the face gap at 1.5 mm, in Torque -> Magnets sizing the axial length to 2.5 N.m.
// magcoupling-rs/src/app.rs (the_smoke_test_share_link_opens_its_design) decodes it and
// compares it with Design::default() plus those settings, so it cannot go stale unnoticed. The
// link carries every input (decision M41-5): regenerate it with MagcouplingPanel::share_link
// when the format changes, a default changes, or an input path is renamed or removed (that
// test fails until then).
const MAGCOUPLING_SMOKE_PAYLOAD = 'lVhLr5w2FP4vrHMRMMO8dm0WrVq1qpRIXVoeMINzARMb7txplP_ec2wDNo9pmkWu5pzP9nk_-BYUQta0Cy5BTW-Z6NuKN7eXnCl-a4IPAW_avlPB5VuQ0YpfJe24aEJatSUlV0laJsnH4PIShVEUJx98UCu6ktWkroPLOTylPhcOfw4ucbjzyRlhTR5cojCe4QsCv4iQ_MYbWiHiPEcwSjohv_YsJn_Cm0kYHbYQiUbE4dEH3GhLclbwhuNPAPhssFDDOlKx5taVWq84mV9hMV3Js9eGKaVhuzBeh915bm86hLuZPjWjqpcstyJbiU8zUB8hNUkPh92RvcwUVi3NwJ1G1HDvMzumQE5Wt-jBBDw4Y4sODG7kVAgAdkXrVoH3K_EwxrGEd66hEnxj35rQVwFGELmmJ6nD0H88Y0YLbvdomfsSgOmN1azpyDvJtY9HXkEzsBQRDSMtZxnDGDnNud1dDFxXyELyzLjcRp4h3yRvSW2V2jn4EqzKNPXkUL8IDoJdRdWRjMusMojksIBsPGeYKpPsDhbfjfRX9iCZaDpQwZo39XhTFO2dpyp-o9pSc480PUigzYHvJyO9lawSNAfp6CjdcXoJuJwVi9sULVj3GC90PaI1IfBDKdeJquJGKNc_qkNbq8LX7g6xRkTv6mBLVHil2SuXQ5aOVCGZG0wD3a8rAzWX_I2RmQamJvkQJwUjnfEDGw6xjuW-EDdGJWFFwTPOmuwxFiuPr9MsuKSemMi5ineiy66GQPaaVw8OjDcNFF60AHGrbDxTD2LjTh9Q0dohwzzZbW7b9HUSsemragV3kzRnRL8N7SLYRIC_NhE1bXp4zCrg9ICnwGXFfQpfLb5PTyzq8AZaq_YjchvgD8tt4P9HbnPiP-VuqexGl_182ieffg22YIPf1mDvpKSyFg3PIGRdzrL_DKymFRWWWTe-RTt2nFQnKIweTEL4QVUC3VkV-jYADEb0hEJBlc59Mib_nJtBFy9Fr-DBVb6qGIOcBoFQXQ9gai9UQiEKEkfkTP74i0KR382E9XHJgDsf13EPzqqc_LQnx8gA93PtVcdA-auinQmtdMmEDpD3UJjfOBSrT6TWt-C_tZuyjg3DGc5mOJ3tlqicNQpvu5Gs3gWXo57S5qBa5H3VK_ILCp5E6dpzdU8khBloix0H57NkXS7VsoxDYSQlA1V_I6-330GPo-6tNYOTIc1pq5NM2BgwldxnFhXFBpxzalJlG2K6YjrnQjTndwq94mZbls9ueSW68fr4tIWoa_94RSaT1mhSNH5yHPhXcKGtODa6U49lknrJAku4pcSxljddOZJgDtixbGi0E8Pi94lP7krJoP0PWu9j_zrLdkawmfZZ37q9Jp2sjhzdyjMhB-Xj6XEYuSumR46F7thfCY7l49xj6ZD8Hfb0toIIs8dGruvdwyRg2V-37FjxghGoCo2ZdaMpsSxb9yuW58M4Dcwk9dmGnAxEO-K3QnFHNYf9jqaGH_bo3hEHmRXM0ybEJ0cmkY8Rb0yiXSfELnUQwwaRS2rGCNPZLRfm2qEYv-ydi6UeXwbTlqJaZJGPGGezgfm15_gqPmAWMeeg4jm2rw6V6-Y2kfC_NiUsLnaTcF_tm2EUdENElbTAXFUgTTYFZoT7jkWYgr_pPstfSd3TDILL8lIAwzPkeKLCPKsjiqglQ7a1ibPJbrpArglxPI1PwagH2_qQ1NFuoL9RKLLuRmHId3a1SQok9DVESwcRAVWsBE-8sVAxzDw7wK4hCGZGmFNePYi6o_k-DvV2G16ALDeYlyFEekkbs4slz06UUExhBTDJacMgtzm9fUox-cahQDwgHtXQKVx4ziALwyvcR8rsi_e5Il1DZoLJzHRYJXqZsYVdDM58HxlW3o_bT6MybwTWaqgFOXn9iegU3T8DV_yV4ehk0afD7hm6hYzGrDDg4zl-BsaRCC6W6EZzID2cNw5kX5LIguL0nKyjYFJj3r54nqH67qFXKHxRcsU2bIWwgvawOJs10PaVBQbDhNZXjlmFM2S6itLFHCZAOdTyVdS9xHyTAj8LeOnoImuuYITJyinwVInFzxuKIOWi48a5Ji8gCZ1x7OXEXp6Dvbvjw7ZMkmWihoKXQ2jh_IBtaCpD7hFdcyqYFUKQBZrxZ52Q8f45qpDgXQ2NTsfDEzA2VnPl3A5zlHPl8bRLtsGmpWrkPj5v42yX1cBnOFuk7YWHLSAOOvARhsEmW5HPSb3HDT2BzSaOto6A_Z3PKZv6l_wGYweEOIdVCBePRVpPWCguNSW7-GDm_HiXrkXxHG-iR584HJ8fwMbgqpkMa8J-Oya8M8aP-uAhPNnVzz1nOxV8eAE1cNtbSDMhaNXDrMB7_FD8DKcVhEFkv40xOxK2HPI3-R212n3_ECj-j14GvwVabt0tr7icBu63D9h4IfWQaj_3dGL8_gnPmYnF-RIEXRXuhglMmY_F3_8F'

// The design file the /magcoupling/ smoke loads through the file picker (rfd's HTML overlay on
// the web): the default design with the face gap at 2 mm. app.rs's smoke test reads it too.
const MAGCOUPLING_SMOKE_DESIGN_FILE = '{"format": "magcoupling-design", "version": 1, "inputs": {"metal.face_gap_mm": 2.0}}'

const PLAYWRIGHT_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close"'

// The calculator's step also clicks the canvas once and uploads a file.
const MAGCOUPLING_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_run_code_unsafe,mcp__plugin_playwright_playwright__browser_click,mcp__plugin_playwright_playwright__browser_file_upload,mcp__plugin_playwright_playwright__browser_close"'

const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
if (ARGS.selftest) { return { ok: true } }

const url = ARGS.url || 'http://localhost:8080'

let linkage = null
if (ARGS.linkage !== false) {
  linkage = await agent(
    `Smoke-test the WASM linkage app at ${url} using Playwright MCP tools (${PLAYWRIGHT_TOOLS}).
Steps: navigate to ${url}; wait 5 seconds for WASM init; snapshot the page and confirm a <canvas> element exists; collect console messages; take a screenshot and judge whether it shows a rendered app (menu bar / toolbar / panels visible, any theme) vs a blank page; close the browser.
passed=true only if: page loaded, canvas present, zero console messages of type error (warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (caller must have run scripts/serve_web.sh).`,
    { label: 'gui-smoke', phase: 'Smoke', schema: SMOKE_SCHEMA, model: 'sonnet' },
  ) || { passed: false, canvas_present: false, console_errors: [], notes: 'smoke agent returned no result (agent error)' }
}

// The magnetic coupling calculator: deterministic state through its share link, checked in
// the console (the page is one canvas). One click on the canvas, on the "Load design" button
// found in a screenshot, checks the web file picker (rfd's HTML overlay) and the file load.
let magcoupling = null
if (ARGS.magcoupling !== false) {
  const page = `${url}/magcoupling/?m=${MAGCOUPLING_SMOKE_PAYLOAD}`
  magcoupling = await agent(
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (inputs on the left, dashboard on the right, results table in the middle) vs a blank page.
Then the design file picker, the only click on the canvas: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
In the screenshot, find the "Load design" button in the header row at the top of the page (between "Save design" and "Copy share link") and click its centre with browser_run_code_unsafe, code: async (page) => { await page.mouse.click(X, Y); } (X and Y in CSS pixels of the screenshot; the canvas fills the page). rfd shows its overlay (#rfd-overlay: a file input #rfd-input shown as a "Choose File" button, and the buttons "Ok" and "Cancel") and opens the browser's file chooser at once (the tool output reports a "File chooser" modal state); if a snapshot shows the overlay but no chooser opened, click the "Choose File" button. Upload the file with browser_file_upload, click the overlay's "Ok" button, wait 2 seconds and collect the console messages again; delete the file.
design_file_loaded=true only if a console message contains "magcoupling: loaded a design file". If the button cannot be found or the overlay does not appear after two attempts (each with a fresh screenshot), design_file_loaded=false and say why in notes: the canvas click is the fragile part of this step, and the share-link checks do not depend on it. Close the browser.
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at".
passed=true only if: page loaded, canvas present, share_link_loaded, sizing_solved, design_file_loaded, and zero console messages of type error (warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
  ) || { passed: false, canvas_present: false, console_errors: [], share_link_loaded: false, sizing_solved: false, design_file_loaded: false, notes: 'smoke agent returned no result (agent error)' }
}

// The linkage app's fields at the top level, as before, with the calculator's beside them.
const passed = (linkage ? linkage.passed : true) && (magcoupling ? magcoupling.passed : true)
return { ...(linkage || {}), passed, linkage, magcoupling }
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust

/// The label of the free-variable picker.
pub const FREE_VARIABLE_LABEL: &str = "Free variable";

/// The note under the free variable's row in Torque → Magnets.
pub const SIZED_NOTE: &str = "Set by Torque -> Magnets";
```

with:

```rust

/// The label of the free-variable picker.
pub const FREE_VARIABLE_LABEL: &str = "Free variable";

/// The start of the log line of each sizing outcome (the web smoke test looks for it).
pub const SIZING_LOG_PREFIX: &str = "magcoupling sizing: ";

/// The note under the free variable's row in Torque → Magnets.
pub const SIZED_NOTE: &str = "Set by Torque -> Magnets";
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    /// Logs the outcome of the last solve (the web smoke reads it).
    fn log_sizing(&self) {
        log::info!(
            "magcoupling sizing: {}",
            self.runner.status(&self.inputs, &self.sizing)
        );
    }
```

with:

```rust
    /// Logs the outcome of the last solve (the web smoke reads it).
    fn log_sizing(&self) {
        log::info!(
            "{SIZING_LOG_PREFIX}{}",
            self.runner.status(&self.inputs, &self.sizing)
        );
    }
```

- [ ] **Step 4: Run the tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED|panicked"
```

Expected: `test result: ok. 275 passed; 0 failed` (Task 9's 274 and the smoke-link test).

- [ ] **Step 5: Build the web bundle and run the smoke**

The smoke runs the shipped web bundle in a browser through the `gui-smoke` workflow's link. `serve_web.sh` refuses to start without the linkage bundle (`linkage-web_bg.wasm`), so serve `linkage-sim-rs/web/` directly, and stop the server when done.

Run:

```bash
bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/build_magcoupling_web.sh 2>&1 | tail -4
cd C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/web && (python -m http.server 8765 > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/smoke-server.log 2>&1 &) ; sleep 2; curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/magcoupling/
```

Expected: `parity guard: magcoupling-web builds without workbook-parity; ...`, `Magcoupling build complete!` and the wasm at about 4.0 MB; then `200`.

Then, with the Playwright MCP tools (load them via ToolSearch: `select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_run_code_unsafe,mcp__plugin_playwright_playwright__browser_click,mcp__plugin_playwright_playwright__browser_file_upload,mcp__plugin_playwright_playwright__browser_close`): navigate to `http://localhost:8765/magcoupling/?m=` followed by the `MAGCOUPLING_SMOKE_PAYLOAD` string of `C:/Users/Cole/source/repos/lsim-mag-m41/.claude/workflows/gui-smoke.js` (copied whole); wait 8 seconds; take a snapshot (a `canvas` element is present); read the console messages at level `debug`; take a screenshot.

Expected: the console has 0 errors and 0 warnings, and contains `magcoupling: loaded the design from the share link (sizing: Torque -> Magnets)` and `magcoupling sizing: Solved at 14.18 mm (hot-low torque 2.500 N·m)`; the screenshot shows the dark theme, the inputs on the left (the Key design group with `Torque -> Magnets` selected and the green "Solved at 14.18 mm" line), the results table in the middle, the dashboard on the right, and Undo disabled in the header (the start-up link is no undo step).

Then the web file picker, the one click on the canvas: write the `MAGCOUPLING_SMOKE_DESIGN_FILE` text of the workflow to `.playwright-mcp/magcoupling-smoke-design.json` under the checkout the Claude session runs from (gitignored in every checkout; the Playwright MCP uploads only files under its allowed roots, and its refusal names them); read the centre of the header's "Load design" button off the screenshot (CSS pixels) and click it with `browser_run_code_unsafe`, code `async (page) => { await page.mouse.click(X, Y); }`. rfd shows its overlay (`#rfd-overlay`: a "Choose File" input and the buttons "Ok" and "Cancel") and the tool reports a "File chooser" modal state. Upload the file with `browser_file_upload`, click the overlay's "Ok" (`browser_click` on its snapshot ref), wait 2 seconds, read the console messages at level `info`, take a screenshot, close the browser, and delete the file.

Expected: the console adds `magcoupling: loaded a design file`, still 0 errors and 0 warnings; the screenshot shows "Design file loaded" under the heading, `Magnets -> Torque` selected, the face gap at 2.00 mm and Undo enabled.

Stop the server:

```powershell
$p = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($id in $p) { Stop-Process -Id $id -Force -Confirm:$false; "stopped $id" }
```

Expected: `stopped <pid>`. The build outputs (`web/magcoupling/magcoupling-web.js`, `magcoupling-web_bg.wasm`) are gitignored; `git status --short` lists none of them.

- [ ] **Step 6: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task10.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task10.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task10.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task10.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add .claude/workflows/gui-smoke.js
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/app.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
test(magcoupling): the /magcoupling/ web smoke through a pinned share link

gui-smoke opens /magcoupling/ through a share link (Torque -> Magnets, face gap
1.5 mm), then loads a pinned design file through rfd's web picker (one click on
the "Load design" button), and passes on the canvas, the share-link, sizing and
design-file log lines and zero console errors. The payload and the file live
once, in the workflow; a Rust test decodes both and runs the app to the solve.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 11: Docs: status, the panel, invariants and open items

**Model:** `sonnet` (doc and YAML edits, running commands; CLAUDE.md section 5). Escalate a retry to the session model.

Repo rule: docs change with code (`docs/ai/01-meta.yaml`). The README's status, layout and test rows and a new "The panel" section (with the file formats and the decisions); `02-system.yaml`'s invariants (every edit through `set`, the debounced solve, no I/O in the panel, the markers' source) and six lessons (egui's `wants_keyboard_input`, egui's `keys_down` and headless key taps, opening collapsing headers bottom up in tests, the missing arrow glyph, a side panel's one-frame lag, Windows MAX_PATH); `03-structure.yaml`'s gui and app entries; `04-memory.yaml`: the open M4 items resolved, `8bad022`'s exact-equality item among them (M41-1; its one caveat, the vacuum permeability's off-grid defaults, recorded open), the follow-ups recorded (the wasm size, BL-013's likely fix, what waits for plan A-3, M5's undo keys, the deploy that needs the user's go); the README's "Regenerating test data" says when the smoke link must be regenerated; the tracker's entry. This plan is committed beside the earlier ones.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/README.md`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/02-system.yaml`, `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/03-structure.yaml`, `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/04-memory.yaml`, `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/05-update-tracker.md`
- Create: `C:/Users/Cole/source/repos/lsim-mag-m41/docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md` (this plan)

**Interfaces:**
- Consumes: everything Tasks 1 to 10 produce (the docs name it).
- Produces: nothing code reads.

- [ ] **Step 1: Update the README and docs/ai**

The update tracker's entry goes right under its header's `---` line.

In `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/02-system.yaml`, replace:

```yaml
      Addendum A-2 adds the harmonic set, the assumptions registry, inverse sizing and the space claim, with an axial housing that follows the length override.
      M4 infrastructure adds the egui panel (feature gui), the standalone app and its native and
      web binaries (feature app), and the second web bundle at /magcoupling/.
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs, src/gui/panel.rs, src/app.rs]
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
      - Every departure from the workbook is an approved, registered deviation (E1-E20 in deviations.rs; E15-E17 are probed on top of E9, decision 15). compute_all always applies Deviations::ALL; the switch exists only behind the test-only workbook-parity feature.
```

with:

```yaml
      Addendum A-2 adds the harmonic set, the assumptions registry, inverse sizing and the space claim, with an axial housing that follows the length override.
      M4 infrastructure adds the egui panel (feature gui), the standalone app and its native and
      web binaries (feature app), and the second web bundle at /magcoupling/.
      M4-1 grows the panel: every input from the metadata, the dashboard, the results table,
      undo/redo, design files and share links, the sizing mode, the linkage app's theme.
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs, src/gui/panel.rs, src/gui/session.rs, src/gui/sizing.rs, src/gui/dashboard.rs, src/app.rs]
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
      - Every departure from the workbook is an approved, registered deviation (E1-E20 in deviations.rs; E15-E17 are probed on top of E9, decision 15). compute_all always applies Deviations::ALL; the switch exists only behind the test-only workbook-parity feature.
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/02-system.yaml`, replace:

```yaml
      - "Library parts and gradeless manual magnets use Calibration C22 and the NdFeB density; only a grade-mode ring (manual dimensions with a grade) takes its grade's alpha(Br) and density (decision A2-7)."
      - "The engine stays pure std: egui and eframe are optional dependencies behind the gui and app features, and the no-feature wasm32 check (gate 6) proves it. egui, eframe and wasm-bindgen are locked at linkage-sim-rs's versions, and wasm-bindgen at deploy-web.yml's wasm-bindgen-cli pin (gate 11)."
      - "workbook-parity never reaches a shipped build. The shipped cargo arguments live once, in linkage-sim-rs/scripts/magcoupling_shipped.sh. build_magcoupling_web.sh and gate 10 read cargo's --message-format=json record of the compiled units, and they fail if a magcoupling-rs unit has the feature. Gate 10 also runs a negative control that must trip."
      - "The panel (gui::MagcouplingPanel) computes only through compute_all, with every correction on. It recomputes every frame after the inputs are drawn, so the readout shows the same frame's edits. Its sliders come from the input metadata and clamp edits only, so an idle frame never rewrites an input."

dataflow_on_user_change: |
  user edit -> state.blueprint mutation -> state.push_undo() -> state.rebuild()
```

with:

```yaml
      - "Library parts and gradeless manual magnets use Calibration C22 and the NdFeB density; only a grade-mode ring (manual dimensions with a grade) takes its grade's alpha(Br) and density (decision A2-7)."
      - "The engine stays pure std: egui and eframe are optional dependencies behind the gui and app features, and the no-feature wasm32 check (gate 6) proves it. egui, eframe and wasm-bindgen are locked at linkage-sim-rs's versions, and wasm-bindgen at deploy-web.yml's wasm-bindgen-cli pin (gate 11)."
      - "workbook-parity never reaches a shipped build. The shipped cargo arguments live once, in linkage-sim-rs/scripts/magcoupling_shipped.sh. build_magcoupling_web.sh and gate 10 read cargo's --message-format=json record of the compiled units, and they fail if a magcoupling-rs unit has the feature. Gate 10 also runs a negative control that must trip."
      - "The panel (gui::MagcouplingPanel) computes only through compute_all, with every correction on. It recomputes every frame after the inputs are drawn, so the readout shows the same frame's edits. Its sliders come from the input metadata and clamp edits only, so an idle frame never rewrites an input; slider values are rounded to the step's decimals (decision M41-1)."
      - "Every edit goes through InputSet::set (rows, design files, share links); loading a file or a link is all or nothing and the refusal names every problem of the inputs and the sizing state (gui::session, decision M41-6). Files and links are one schema-versioned JSON format naming every input and the sizing state; a renamed or removed input path gets a session::PATH_MIGRATIONS entry and a DESIGN_VERSION bump, so older files and links keep opening."
      - "An edit is in progress while a pointer button is down, a key is held down (a held arrow key's auto-repeat run is one undo step) or a text field of an input row (or the target's value box) has focus; a focused text field elsewhere (the results search) is no edit. Undo steps and solves wait for it to end. A share link opened at start-up is the session's start (open_share_payload: no undo step)."
      - "sizing::solve never runs per frame: gui::sizing::SizingRunner solves once the design has been still for DEBOUNCE_S (0.25 s) and no edit is in progress. In Torque -> Magnets the panel shows the inputs with the free variable at the solved (or best) value; the inputs keep their own value until the mode is left, when a change still waiting for its debounce is solved at once (solve_now, one solve per click)."
      - "The panel does no I/O: saving and picking files are PanelRequests its host does (app::files natively and on the web; the linkage app in M5)."
      - "The corrected-vs-workbook markers come from the deviation registry (cells, changes at defaults, probes) and the compiled golden files (E3, E4, E5), never from computing with corrections off (gui::corrections)."

dataflow_on_user_change: |
  user edit -> state.blueprint mutation -> state.push_undo() -> state.rebuild()
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/02-system.yaml`, replace:

```yaml
    shipped builds and the wasm check do not.
  - "Cargo unifies a dev-dependency's features into every unit of a test build, the binaries included (cargo test --features app builds magcoupling-app with workbook-parity), so a compile_error! on a feature pair in a bin breaks cargo test. To check what a shipped binary is built with, read the features field of cargo's --message-format=json compiler-artifact lines, which lists the features each unit was actually compiled with (magcoupling_shipped.sh)."
  - "egui's Slider writes the snapped value back on idle frames only under the default SliderClamping::Always, which calls set_value(old) every frame. SliderClamping::Edits never does (magcoupling panel test idle_frames_change_no_input). Under SliderClamping::Never, arrow keys step past the range ends (egui 0.32.3 source: set_value clamps unless the clamping is Never)."

current_statuses:
  solver: Rust port complete; 716 lib tests pass; trajectory-mode IK Stage 1
```

with:

```yaml
    shipped builds and the wasm check do not.
  - "Cargo unifies a dev-dependency's features into every unit of a test build, the binaries included (cargo test --features app builds magcoupling-app with workbook-parity), so a compile_error! on a feature pair in a bin breaks cargo test. To check what a shipped binary is built with, read the features field of cargo's --message-format=json compiler-artifact lines, which lists the features each unit was actually compiled with (magcoupling_shipped.sh)."
  - "egui's Slider writes the snapped value back on idle frames only under the default SliderClamping::Always, which calls set_value(old) every frame. SliderClamping::Edits never does (magcoupling panel test idle_frames_change_no_input). Under SliderClamping::Never, arrow keys step past the range ends (egui 0.32.3 source: set_value clamps unless the clamping is Never)."
  - "egui 0.32's Context::wants_keyboard_input is true for any focused widget, a slider rail included. To know whether a text field has focus, check egui::text_edit::TextEditState::load(ctx, focused id) (magcoupling panel focused_text_field). A Slider's response takes its value box's id while the box has focus, so a row's response id names the focused text field."
  - "egui keeps a pressed key in InputState::keys_down until its release event and marks every later press of it a repeat (begin_pass), so a headless test that sends presses only holds the key down for good; tap keys instead (press and release in one frame: magcoupling test_support::key_tap). keys_down is cleared when the window loses focus."
  - "A headless test that opens several CollapsingHeaders by clicking their labels must click bottom up: a header above that opens animates the rows below over the next frames, so a click aimed at a later label can land on a slider (magcoupling idle_frames_change_no_input)."
  - "egui's default fonts have no U+2192 (the arrow) or U+2264: it draws an empty box. The magcoupling panel writes 'Torque -> Magnets' and '<=', and a test checks every drawn and hover text for glyphs (every_text_the_panel_shows_has_glyphs_in_the_default_fonts)."
  - "A panel shown with show_inside sizes from the previous frame: text added below its first-frame height is painted from the next frame on. After work done late in a frame (the magcoupling sizing solve), request a repaint so the parts drawn earlier catch up on an idle page."
  - "Windows MAX_PATH: a release build under a deep directory fails with LNK1104 (cannot open a build-script exe). Build from a short path (the worktree, or subst a drive letter)."

current_statuses:
  solver: Rust port complete; 716 lib tests pass; trajectory-mode IK Stage 1
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/02-system.yaml`, replace:

```yaml
  dxf_import: works native + WASM via drag-drop; selection-based conversions
  actuator_workflow: two-pass solve gives correct reactions with actuator loads
  non_grashof_sweep: NaN-padded across unreachable angles; playback ping-pongs
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim. M4 infrastructure in place (gui panel tracer, app binaries, /magcoupling/ bundle, workbook-parity guard, gates 7-11); next: Addendum A-3 (explanations), then the M4 GUI plans"
```

with:

```yaml
  dxf_import: works native + WASM via drag-drop; selection-based conversions
  actuator_workflow: two-pass solve gives correct reactions with actuator loads
  non_grashof_sweep: NaN-padded across unreachable angles; playback ping-pongs
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim. M4 infrastructure in place (gui panel, app binaries, /magcoupling/ bundle, workbook-parity guard, gates 7-11). M4-1 complete (inputs, dashboard, results table, session, sizing mode, theme); next: Addendum A-3 (explanations), M4-2 (geometry view, plots) and M4-3 (equation explorer, assumptions panel, notes, pickers)"
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/03-structure.yaml`, replace:

```yaml

magcoupling_rs:
  path: magcoupling-rs/ (repo root, sibling of linkage-sim-rs/; separate crate, NOT a workspace member)
  role: Rust port of reference/magcoupling-py (spec docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md); M2 complete - every Python module except fields3d (M3) is ported, workbook parity covers all 1,149 checks, E1-E14 applied; Addendum A-1 (grades, materials, E15-E20) and A-2 (harmonic set, assumptions, inverse sizing, space claim) complete; M4 infrastructure adds the egui panel tracer, the standalone app binaries and the /magcoupling/ web bundle
  guide: magcoupling-rs/README.md (tests, regenerating data, porting a module, Python-to-Rust translation rules, applying deviations)
  lib: magcoupling (src/lib.rs re-exports compute_all, headline, DesignInputs, DesignResults; mod gui behind feature gui, mod app behind feature app)
  engine: src/engine/ (pure std, wasm32-clean; one module per Python module)
```

with:

```yaml

magcoupling_rs:
  path: magcoupling-rs/ (repo root, sibling of linkage-sim-rs/; separate crate, NOT a workspace member)
  role: Rust port of reference/magcoupling-py (spec docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md); M2 complete - every Python module except fields3d (M3) is ported, workbook parity covers all 1,149 checks, E1-E14 applied; Addendum A-1 (grades, materials, E15-E20) and A-2 (harmonic set, assumptions, inverse sizing, space claim) complete; M4 infrastructure adds the egui panel, the standalone app binaries and the /magcoupling/ web bundle; M4-1 adds the inputs, dashboard, results table, session (undo/redo, design files, share links), sizing mode and theme
  guide: magcoupling-rs/README.md (tests, regenerating data, porting a module, Python-to-Rust translation rules, applying deviations)
  lib: magcoupling (src/lib.rs re-exports compute_all, headline, DesignInputs, DesignResults; mod gui behind feature gui, mod app behind feature app)
  engine: src/engine/ (pure std, wasm32-clean; one module per Python module)
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/03-structure.yaml`, replace:

```yaml
    addendum_only: [assumptions.rs, sizing.rs, housing.rs]
    remaining: [fields3d (M3)]
  features:
    gui: "src/gui/ (egui 0.32 only, no eframe): panel.rs (MagcouplingPanel: inputs, results, fn ui(&mut self, ui: &mut egui::Ui) recomputing compute_all every frame after the inputs are drawn; KEY_INPUTS = face gap, pole count, axial length (manual inner length) as metadata-driven sliders, SliderClamping::Edits; Reset all; the HEADLINE numbers), format.rs (format_value: 4 significant digits, +inf/-inf/NaN; with_unit), test_support.rs (cfg(test) headless egui helpers: central_panel_frame, key_press, primary_button, drawn_texts, text_rect)"
    app: "src/app.rs (MagcouplingApp, the panel as a full page; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; native fallback = run_native), both required-features app"
    workbook_parity: "test-only, enabled by a self dev-dependency; never in a shipped build (linkage-sim-rs/scripts/magcoupling_shipped.sh, gate 10)"
    versions: "Cargo.toml caret ranges (egui 0.32, eframe 0.32, wasm-bindgen 0.2); Cargo.lock pins linkage-sim-rs/Cargo.lock's egui/eframe 0.32.3 and wasm-bindgen 0.2.114 (= the deploy-web.yml wasm-bindgen-cli pin); gate 11 checks"
  web_bundle: "linkage-sim-rs/web/magcoupling/ (index.html committed, canvas magcoupling_canvas, absolute /magcoupling/ glue path; magcoupling-web.js and magcoupling-web_bg.wasm gitignored in web/.gitignore), built by linkage-sim-rs/scripts/build_magcoupling_web.sh (called by build_web.sh and by deploy-web.yml), served by serve_web.sh [PORT] at /magcoupling/; vercel.json revalidates /magcoupling/magcoupling-web.js"
```

with:

```yaml
    addendum_only: [assumptions.rs, sizing.rs, housing.rs]
    remaining: [fields3d (M3)]
  features:
    gui: "src/gui/ (egui 0.32, serde_json, base64, flate2, log; no eframe, no I/O): panel.rs (MagcouplingPanel: header with Undo/Redo/Reset all/Save design/Load design/Copy share link, inputs SidePanel, dashboard SidePanel, CentreView (results table); fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; PanelRequest (SaveFile, OpenDesign) drained by take_requests; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; sizing controls and run_sizing), inputs.rs (InputCatalogue::get, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS, step_decimals, outside_range, text_hint, input_tooltip), input_ui.rs (input_row, slider, RowEdit), dashboard.rs (DASHBOARD, Level, verdict_level, END_EFFECT_ROWS, STORED_3D_ROWS, result_info, result_tooltip (the hover hook), dashboard_lines, end_effect_banner), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip), results_table.rs (table_entries, search, exact_number, results_csv, results_json, ResultsTable, row_tooltip), session.rs (Design, design_to_json/design_from_json, encode/decode_share_payload, share_link, LoadError, PATH_MIGRATIONS, json_number), history.rs (History, MAX_UNDO_LEVELS 100), sizing.rs (SizingMode, SizingState, variable_key/label, SizingRunner (update, solve_now), DEBOUNCE_S 0.25), format.rs (format_value, with_unit, non_finite_text), test_support.rs (cfg(test): sized_frame(_at), SCREEN, key_event, key_tap, select_all, primary_button, drawn_texts, text_rect, short_magnets)"
    app: "src/app.rs (MagcouplingApp::new applies the theme; ui drains the panel's requests; open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (cad_dark_visuals, a copy of linkage-sim-rs's checked body for body; apply forces ThemePreference::Dark), src/app/files.rs (save: rfd dialog natively, Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
    workbook_parity: "test-only, enabled by a self dev-dependency; never in a shipped build (linkage-sim-rs/scripts/magcoupling_shipped.sh, gate 10)"
    versions: "Cargo.toml caret ranges (egui 0.32, eframe 0.32, wasm-bindgen 0.2); Cargo.lock pins linkage-sim-rs/Cargo.lock's egui/eframe 0.32.3 and wasm-bindgen 0.2.114 (= the deploy-web.yml wasm-bindgen-cli pin); gate 11 checks"
  web_bundle: "linkage-sim-rs/web/magcoupling/ (index.html committed, canvas magcoupling_canvas, absolute /magcoupling/ glue path; magcoupling-web.js and magcoupling-web_bg.wasm gitignored in web/.gitignore), built by linkage-sim-rs/scripts/build_magcoupling_web.sh (called by build_web.sh and by deploy-web.yml), served by serve_web.sh [PORT] at /magcoupling/; vercel.json revalidates /magcoupling/magcoupling-web.js"
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/04-memory.yaml`, replace:

```yaml
  - "Open (Addendum A-1 final review; for A-2/M4, no change in A-1): E20 applies the workbook's rating calibration (C50 = model onset t_ref minus the rating, temperature.rs ring_demag) to every grade. Where a rating exceeds the grade's uncalibrated t_ref the offset is negative and raises every onset, so the check is less conservative than the uncalibrated model. Measured at 8958deb (manual dimensions, the grade on both rings, other inputs default, every correction on): N52 -9.7 C (approved as-is: report 6.3 row B842-N52, C12 30.73 -> 0.17 C), N50 -6.1 (80 C rating), N50M -7.5, N38UH -7.3, N35EH -3.6, Recoma 26 -18.6 (report R11: Arnold's 350 C 'may be considerably lower at low load line'); large positive offsets the other way: BCN-19 +89.9, Recoma 30 +46, Recoma 20 +255. Candidate: clamp the offset at >= 0 as a later registered correction (user decision)"
  - "Open (Addendum A-1 Task 13 review; not in A-2 scope): the CTE-mismatch warning and E18's bond screen compare the hub with the NdFeB bond-plane CTE (Temperature design C95) for every grade, so a ferrite or SmCo ring reads the NdFeB mismatch. The A6 grade table carries no CTE; a per-grade magnet CTE (sourced, like the rest of the table) would fix both. Needs sourcing and the user's approval as a correction"
  - "Open (Addendum A-2 Task 10 review; same family as the per-grade CTE item): the magnets' heat capacity prices each ring's own mass (decision A2-7) at the NdFeB specific heat for every grade, so a ferrite or SmCo ring's thermal mass is off; a sourced per-grade specific heat would fix it. Needs sourcing and the user's approval as a correction"
  - "Open (M4, from the A-2 Task 4 review): assumptions::modified uses exact value equality, so a slider value that round-trips one ulp off its workbook default reads as modified; the M4 GUI must round slider values to the step's decimals (or compare within a tolerance) so untouched assumptions never show the modified dot or banner"
  - "RESOLVED (Addendum A-2, Task 5): tests/robustness.rs compute_all_never_panics_on_extreme_inputs runs every extreme-value case with backiron 1 and with backiron 0, so E17's free-space inputs (b_hub_free_T, b_cup_free_T, web_integral_free_T2m2) are extremed with E17 active"
  - "M3: fields3d.run gives 1.034632e-5; times 4 (E5) and formatted .4g that is 4.139e-5, not the M2 default 4.14e-5 (C125 differs by about 0.00009 W). M3 should pin the M2 default or expect the difference"
  - "Tidy after M2: magcoupling-rs/src/engine/temperature.rs is about 1,350 lines, past the architecture's ~800-line split point; split mechanically (inputs, results, compute, tests) when no correction work is in flight"
  - "Open, not in Addendum A-2's scope (the A3 panel lists no bondline assumption): E4 as approved adds the inner bondline input to the pole-sweep keyed-wall term, while the block-fit term keeps the workbook literal + 0.05 (the same bondline), so at bond_inner != 0.05 the two terms of the max() disagree (sweeps.rs pole_sweep); E4's test and golden file probe only 0.05. Owner: plan A-3 or a deviation proposal (changing E4 is a registered deviation and needs the user's approval); trigger: making the pole-sweep block-fit bondline an input, or any change to E4"
  - "RESOLVED (Addendum A-2, audit M9): the engine flags f_end <= 0 (model.end_effect_check, calibration.end_effect_check, model::end_effect_in_range for sweep rows); M4 greys the numbers computed from the pull-out; inverse sizing never counts such a value"
  - "M4 design questions (decision 28): the M41 cap thread sits below the 41.33 mm cup body OD; the boss OD is 22 mm on Metal design and 25 mm on Shaft clamps. M4 draws the axial housing in effect (housing.hub_length_mm, cup_depth_mm, retainer_span_mm: A-2 decision A2-8)"
  - "M4: the sizing mode, target torque and free variable are GUI state (A-2 decision A2-9); sizing::solve takes them as arguments. Decide whether design files and share links record them"
  - "M4 performance (deferred from the A-2 final review, not a defect): model::shear_stress evaluates all six harmonics (amplitudes and sinh/exp geometry factors of 7, 9 and 11 too) on every call, in the Calculator and each of the 19 sweep rows, even when coupling.max_harmonic sums only 1, 3, 5; with the general root search it makes up compute_all's 8.5 -> 17 us release cost (measured in the A-2 plan), about 50x inside the spec's milliseconds. If M4's per-frame budget (drawing, A-3 equation records) needs it, compute only the harmonics summed, keeping the bits of 1, 3, 5 and a left-out harmonic's reported 0"
  - "Plan A-3: the spec's dependency-graph traceability test for assumptions (each assumption changes every dependent result and no independent one) needs the equation registry; its records must include the Rust-only tau7_Pa-tau11_Pa terms of the pull-out and the Calibration sums, and it must exempt the three documented overrides (assumptions.rs module doc): C45 under E20 with the coercivity source at 1 (a graded magnet uses its grade's beta), a 1018 back iron's own design flux density, and C22 for a grade-mode ring (decision A2-7: the grade's alpha)"
  - "M3: a grade-mode ring's reverse fields are scaled with its own alpha(Br) (decision A2-7); the stored 3D fields do not separate the opposing ring's share, which live fields per ring would"
  - "RESOLVED (M4 infrastructure, magcoupling/m4-infra): the workbook-parity guard checks cargo's --message-format=json record of the shipped builds (linkage-sim-rs/scripts/magcoupling_shipped.sh, magcoupling_assert_shipped). build_magcoupling_web.sh pipes its release build through it, and gate 10 runs it on the native and wasm32 shipped builds plus a negative control that must trip. It trips when the feature is forced on the command line or through app in Cargo.toml, and cargo test and clippy --all-targets with app stay green. M5 must also run it on linkage-sim-rs's shipped builds once linkage-sim-rs depends on magcoupling-rs (the function already accepts any binary; the magcoupling-rs units are matched by package id)"
  - "Open (M4 plans, from the infrastructure tracer): the spec's Key design 'axial length' has no engine input that moves the results at the default library part. The tracer slider edits coupling.magnets.manual_inner_length_mm, which only matters with a blank part. Decide the mapping: the A1 sizing free variable, both manual lengths, or a part filter"
  - "Open (M4 plans): egui snaps slider values to min + k*step (face gap 1.41 is stored as 1.4100000000000001). Round to the step's decimals before set(), or accept the float noise (results differ at about 1e-16 relative)"
  - "Open (M4 plans): the tracer clamps slider edits to the metadata range (SliderClamping::Edits, typed values included), while the SliderRange doc says the range does not reject typed values. Decide whether typed entry may leave the range; SliderClamping::Never also lets arrow keys step past the ends"
  - "Open (M4 plans): the gui-smoke skill should also load /magcoupling/ (spec testing summary); the web page follows the browser's light/dark preference (light in Playwright), while the spec wants the linkage app's theme; magcoupling-web_bg.wasm is 3.5 MB with the default release profile (linkage-web_bg.wasm is 10.5 MB); a size profile (opt-level, LTO) or wasm-opt is a later option"
```

with:

```yaml
  - "Open (Addendum A-1 final review; for A-2/M4, no change in A-1): E20 applies the workbook's rating calibration (C50 = model onset t_ref minus the rating, temperature.rs ring_demag) to every grade. Where a rating exceeds the grade's uncalibrated t_ref the offset is negative and raises every onset, so the check is less conservative than the uncalibrated model. Measured at 8958deb (manual dimensions, the grade on both rings, other inputs default, every correction on): N52 -9.7 C (approved as-is: report 6.3 row B842-N52, C12 30.73 -> 0.17 C), N50 -6.1 (80 C rating), N50M -7.5, N38UH -7.3, N35EH -3.6, Recoma 26 -18.6 (report R11: Arnold's 350 C 'may be considerably lower at low load line'); large positive offsets the other way: BCN-19 +89.9, Recoma 30 +46, Recoma 20 +255. Candidate: clamp the offset at >= 0 as a later registered correction (user decision)"
  - "Open (Addendum A-1 Task 13 review; not in A-2 scope): the CTE-mismatch warning and E18's bond screen compare the hub with the NdFeB bond-plane CTE (Temperature design C95) for every grade, so a ferrite or SmCo ring reads the NdFeB mismatch. The A6 grade table carries no CTE; a per-grade magnet CTE (sourced, like the rest of the table) would fix both. Needs sourcing and the user's approval as a correction"
  - "Open (Addendum A-2 Task 10 review; same family as the per-grade CTE item): the magnets' heat capacity prices each ring's own mass (decision A2-7) at the NdFeB specific heat for every grade, so a ferrite or SmCo ring's thermal mass is off; a sourced per-grade specific heat would fix it. Needs sourcing and the user's approval as a correction"
  - "RESOLVED (M4-1, decision M41-1): slider values are rounded to the step's decimals and SliderClamping::Edits sliders never rewrite a value on an idle frame (idle_frames_change_no_input, every group open, off-grid values included), so an untouched input keeps its exact default and assumptions::modified (exact equality) never reads it as modified; the one caveat is the next item"
  - "Open (M4-1 review; engine metadata, out of M4-1's scope): coupling.mu0 and calibration.mu0 default to 1.256637e-6, off their slider grid (min 1.2566e-6, step 1e-11: 3.7 steps), so once nudged they never return to the default exactly (the changed dot stays lit) and the value box shows 0.00000125664. Neither is a flagged assumption, so assumptions::modified is not affected today. Fix: an engine step of 1e-12 (regenerate input_schema.json and the differential data, README 'Regenerating test data'); gui/inputs.rs OFF_GRID_DEFAULTS lists the two and every_slider_default_is_on_its_step_grid_except_the_listed fails until the list follows the fix"
  - "RESOLVED (Addendum A-2, Task 5): tests/robustness.rs compute_all_never_panics_on_extreme_inputs runs every extreme-value case with backiron 1 and with backiron 0, so E17's free-space inputs (b_hub_free_T, b_cup_free_T, web_integral_free_T2m2) are extremed with E17 active"
  - "M3: fields3d.run gives 1.034632e-5; times 4 (E5) and formatted .4g that is 4.139e-5, not the M2 default 4.14e-5 (C125 differs by about 0.00009 W). M3 should pin the M2 default or expect the difference"
  - "Tidy after M2: magcoupling-rs/src/engine/temperature.rs is about 1,350 lines, past the architecture's ~800-line split point; split mechanically (inputs, results, compute, tests) when no correction work is in flight"
  - "Open, not in Addendum A-2's scope (the A3 panel lists no bondline assumption): E4 as approved adds the inner bondline input to the pole-sweep keyed-wall term, while the block-fit term keeps the workbook literal + 0.05 (the same bondline), so at bond_inner != 0.05 the two terms of the max() disagree (sweeps.rs pole_sweep); E4's test and golden file probe only 0.05. Owner: plan A-3 or a deviation proposal (changing E4 is a registered deviation and needs the user's approval); trigger: making the pole-sweep block-fit bondline an input, or any change to E4"
  - "RESOLVED (Addendum A-2, audit M9): the engine flags f_end <= 0 (model.end_effect_check, calibration.end_effect_check, model::end_effect_in_range for sweep rows); M4 greys the numbers computed from the pull-out; inverse sizing never counts such a value"
  - "M4 design questions (decision 28): the M41 cap thread sits below the 41.33 mm cup body OD; the boss OD is 22 mm on Metal design and 25 mm on Shaft clamps. M4 draws the axial housing in effect (housing.hub_length_mm, cup_depth_mm, retainer_span_mm: A-2 decision A2-8)"
  - "RESOLVED (M4-1, decision M41-4): design files and share links record the sizing mode, free variable and target torque (gui::session \"sizing\")"
  - "M4 performance (deferred from the A-2 final review, not a defect): model::shear_stress evaluates all six harmonics (amplitudes and sinh/exp geometry factors of 7, 9 and 11 too) on every call, in the Calculator and each of the 19 sweep rows, even when coupling.max_harmonic sums only 1, 3, 5; with the general root search it makes up compute_all's 8.5 -> 17 us release cost (measured in the A-2 plan), about 50x inside the spec's milliseconds. If M4's per-frame budget (drawing, A-3 equation records) needs it, compute only the harmonics summed, keeping the bits of 1, 3, 5 and a left-out harmonic's reported 0"
  - "Plan A-3: the spec's dependency-graph traceability test for assumptions (each assumption changes every dependent result and no independent one) needs the equation registry; its records must include the Rust-only tau7_Pa-tau11_Pa terms of the pull-out and the Calibration sums, and it must exempt the three documented overrides (assumptions.rs module doc): C45 under E20 with the coercivity source at 1 (a graded magnet uses its grade's beta), a 1018 back iron's own design flux density, and C22 for a grade-mode ring (decision A2-7: the grade's alpha)"
  - "M3: a grade-mode ring's reverse fields are scaled with its own alpha(Br) (decision A2-7); the stored 3D fields do not separate the opposing ring's share, which live fields per ring would"
  - "RESOLVED (M4 infrastructure, magcoupling/m4-infra): the workbook-parity guard checks cargo's --message-format=json record of the shipped builds (linkage-sim-rs/scripts/magcoupling_shipped.sh, magcoupling_assert_shipped). build_magcoupling_web.sh pipes its release build through it, and gate 10 runs it on the native and wasm32 shipped builds plus a negative control that must trip. It trips when the feature is forced on the command line or through app in Cargo.toml, and cargo test and clippy --all-targets with app stay green. M5 must also run it on linkage-sim-rs's shipped builds once linkage-sim-rs depends on magcoupling-rs (the function already accepts any binary; the magcoupling-rs units are matched by package id)"
  - "RESOLVED (M4-1): the Key design 'axial length' is the A-2 override coupling.magnets.axial_length_mm, blank by default, entered at the inner ring's length in use (decision M41-12); slider values are rounded to the step's decimals (M41-1); edits, typed values included, are clamped to the slider range and a value already outside it is kept and flagged (M41-2); gui-smoke opens /magcoupling/ through a pinned share link; the standalone app forces the linkage app's CAD dark theme (M41-3)"
  - "Open (M4-1 follow-ups): magcoupling-web_bg.wasm is 4.0 MB with the default release profile after M4-1 (3.5 MB before; rfd, serde_json, flate2); a size profile (opt-level, LTO) or wasm-opt is a later option. The linkage web app itself still renders light in a light-preference browser (backlog BL-013): ctx.set_theme(egui::ThemePreference::Dark), as magcoupling's app/theme.rs does, is the likely fix"
  - "Open (M4-2, M4-3): per-row end-effect greying of the results table waits for plan A-3's dependency graph (M4-1 shows the banner over the table and greys the pinned dashboard rows); M4-3 may re-source the corrected-vs-workbook markers from A-3's per-record upstream corrections; result_tooltip (gui/dashboard.rs) is the hover hook the equation tooltip joins; the centre region's CentreView takes M4-2's geometry view and plot tabs"
  - "Open (M5, from the M4-1 review): linkage-sim-rs/src/gui/mod.rs reads Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y with a non-consuming key_pressed, which runs whatever the panel consumes. Hosting MagcouplingPanel in an egui::Window, M5 must skip its own undo while the calculator window has the user's attention and turn the panel's shortcuts on only then (MagcouplingPanel::set_keyboard_shortcuts), so one key press never undoes both"
  - "Deploy (M4-1): nothing pushed. deploy-web.yml on main already builds and ships /magcoupling/ on the next push of main; that push needs the user's explicit go"
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai/05-update-tracker.md`, replace:

```markdown
Reverse chronological (newest at top).

---

## 2026-09-30 — Magcoupling Addendum A-2: final review fix wave
- E18's corrected_formula no longer quotes the library's 6061 at 68.3 GPa (decision A8): since A2-6 it is
```

with:

```markdown
Reverse chronological (newest at top).

---

## 2026-10-01 — Magcoupling M4-1: GUI inputs, dashboard, results table and session (branch magcoupling/m4-1)
- `magcoupling-rs` panel (`src/gui/`): every input generated from the metadata, grouped as the package
  groups them, with the Key design group on top (the A-2 axial length override, blank by default);
  sliders with value boxes, arrow nudges, log scales, values rounded to the step's decimals, edits
  clamped to the range, per-field reset, the changed dot, tooltips; selectors, optional inputs and text
  inputs. The dashboard: the headline with green/amber/red badges from the check verdicts, the
  corrected-vs-workbook markers from the registry and the golden files, the audit M9 greying with its
  banner, the "3D values from the workbook" label, the space claim. The results table: search, CSV and
  JSON export (`+inf`/`NaN` as text).
- Session: undo/redo (one step per settled edit), reset all, design files and share links in one
  schema-versioned JSON format (every input and the sizing state; `?m=` deflate + URL-safe base64),
  loading all or nothing. The sizing mode switch (Torque -> Magnets): `sizing::solve` debounced
  (0.25 s, never per frame), the solved or best value shown and its row locked, kept on leaving.
- Edits in progress: a drag, a held key (a held arrow key is one undo step) or a text field of an input
  row; the results search is no edit. A start-up share link is the session's start (no undo step);
  leaving Torque -> Magnets solves a change still waiting for its debounce first; the JSON export holds
  the design that produced the results; a host can keep the undo keys (M5). Old files' renamed paths
  migrate through `session::PATH_MIGRATIONS`; a refused file names every problem of the inputs and
  the sizing state.
- App: the linkage app's CAD dark theme, forced dark; native file dialogs and web downloads (rfd); the
  web entry opens `?m=` links. `gui-smoke` opens `/magcoupling/` through a pinned link and loads a
  design file through rfd's web picker (zero console errors, share link loaded, sizing solved, design
  file loaded).
- Decisions M41-1 to M41-15 as recommended in the plan
  (`docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md`). The engine,
  parity, differential data and registry are unchanged. Nothing pushed: the next push of `main` deploys
  `/magcoupling/` and needs the user's go.

## 2026-09-30 — Magcoupling Addendum A-2: final review fix wave
- E18's corrected_formula no longer quotes the library's 6061 at 68.3 GPa (decision A8): since A2-6 it is
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/README.md`, replace:

```markdown
override with the axial housing that follows it (hub length, cup cavity depth,
retainer span), inverse sizing (A1), the housing autofit suggestion and the space
claim, one aluminium modulus, and each grade-mode ring's own alpha and density
(decisions A2-1 to A2-9). M4 infrastructure in place: the `gui` and `app` features, a tracer panel
(`gui::MagcouplingPanel`), the native and web binaries, and the second web
bundle at `/magcoupling/` (see [Features and binaries](#features-and-binaries)).
Next: Addendum A-3 (explanations), then the M4 GUI plans.

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
```

with:

```markdown
override with the axial housing that follows it (hub length, cup cavity depth,
retainer span), inverse sizing (A1), the housing autofit suggestion and the space
claim, one aluminium modulus, and each grade-mode ring's own alpha and density
(decisions A2-1 to A2-9). M4 infrastructure in place: the `gui` and `app` features, the panel
(`gui::MagcouplingPanel`), the native and web binaries, and the second web
bundle at `/magcoupling/` (see [Features and binaries](#features-and-binaries)).
M4-1 complete: every input generated from the metadata (the Key design group first), the
dashboard (badges, corrected-vs-workbook markers, the end-effect greying, the stored-3D label,
the space claim), the searchable results table with CSV and JSON export, undo and redo, design
files and share links, the sizing mode switch and the linkage app's theme (see
[The panel](#the-panel); decisions M41-1 to M41-15). Next: Addendum A-3 (explanations), M4-2
(geometry view, plots, clamp drawing) and M4-3 (equation explorer, assumptions panel, teaching
notes, material and grade pickers).

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28; with the axial length override set, `axial_housing` makes the hub length, the cup cavity depth and the retainer span follow the rings they bound, each at least its ring's length: decision A2-8) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, `space_claim_check`, and the hub length, cup depth and retainer span in effect) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`, `KEY_INPUTS`), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included), `test_support.rs` (headless egui helpers, tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page), `run_native`, `TITLE`, `CANVAS_ID` |
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |

```

with:

```markdown
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28; with the axial length override set, `axial_housing` makes the hub length, the cup cavity depth and the retainer span follow the rings they bound, each at least its ring's length: decision A2-8) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, `space_claim_check`, and the hub length, cup depth and retainer span in effect) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, the sizing controls), `inputs.rs` (`InputCatalogue`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type), `dashboard.rs` (`DASHBOARD`, `verdict_level`, `END_EFFECT_ROWS`, `STORED_3D_ROWS`, `result_info`, `result_tooltip`, the hover hook), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `search`, `results_csv`, `results_json`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`), `history.rs` (`History`: undo and redo), `sizing.rs` (`SizingMode`, `SizingState`, `SizingRunner`: debounced inverse sizing), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included), `test_support.rs` (headless egui helpers, tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the linkage app's CAD dark visuals, forced dark), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/README.md`, replace:

```markdown
| Feature | Adds | Dependencies (all optional) |
|---|---|---|
| none (default) | The engine | none: pure std, builds for wasm32 as it is |
| `gui` | `gui::MagcouplingPanel`: the design inputs and results, and `fn ui(&mut self, ui: &mut egui::Ui)`. Any egui app can host it: the standalone app as a full page, the linkage app in an `egui::Window` (M5) | egui 0.32 |
| `app` | `app::MagcouplingApp` and the binaries below | `gui`, eframe 0.32, log; env_logger (native); wasm-bindgen, wasm-bindgen-futures, web-sys (wasm32) |
| `workbook-parity` | **Test-only**: the switch that turns corrections off (see [Differences from the workbook](#differences-from-the-workbook)) | none |

The panel is the M4 tracer: sliders for face gap, pole count and axial length
(`KEY_INPUTS`), set up from the input metadata (label, unit, range, step, log
scale, help and cell in the tooltip), "Reset all", and the headline numbers,
recomputed with `compute_all` (every correction on) each frame after the inputs
are drawn. Sliders clamp edits only (`SliderClamping::Edits`): the slider, arrow
keys and typed values stay in the range, and an idle frame never rewrites a
value. The axial length is the manual inner length, which only matters when
the inner part is not a library part.

**Versions.** egui and eframe use the same 0.32 line as `linkage-sim-rs`, so
M5 embeds the panel with one egui; `Cargo.toml` has caret ranges, and
```

with:

```markdown
| Feature | Adds | Dependencies (all optional) |
|---|---|---|
| none (default) | The engine | none: pure std, builds for wasm32 as it is |
| `gui` | `gui::MagcouplingPanel`: the design inputs and results, and `fn ui(&mut self, ui: &mut egui::Ui)`. Any egui app can host it: the standalone app as a full page, the linkage app in an `egui::Window` (M5) | egui 0.32, serde_json, base64, flate2, log |
| `app` | `app::MagcouplingApp` and the binaries below | `gui`, eframe 0.32, log, rfd 0.15; env_logger (native); wasm-bindgen, wasm-bindgen-futures, web-sys, js-sys (wasm32) |
| `workbook-parity` | **Test-only**: the switch that turns corrections off (see [Differences from the workbook](#differences-from-the-workbook)) | none |

### The panel

M4-1 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md`).
A header with the session buttons; the inputs on the left; the dashboard on the right; the
centre region between them (`CentreView`: the results table; M4-2 adds the geometry view and the
plots). Every result is recomputed with `compute_all` (every correction on) each frame after the
inputs are drawn, so the readouts show the same frame's edits.

- **Inputs** (`inputs.rs`, `input_ui.rs`): every input, generated from its metadata and grouped as
  the package groups them, each nested group under its heading; the Key design group on top
  (`KEY_DESIGN`: face gap, pole count, both magnet parts, the A-2 axial length override (blank
  by default), operating temperature, back iron, cup wall, conductance, measured drag), whose
  inputs also stay in their groups. A number is a slider with a value box (live while dragging,
  arrow-key nudges by the step, a logarithmic scale where flagged, values rounded to the step's
  decimals: decision M41-1; every default sits on its step grid except the vacuum permeability's
  two, an open item); edits are clamped to the slider range, typed values included, while a value
  already outside it (from a file or a link) is kept and flagged "outside the slider range"
  (M41-2). A selector is a drop-down; an optional input a checkbox and a slider, entered at the
  result it overrides, unrounded (`OPTIONAL_SEEDS`, M41-12: the axial length override at the
  ring's length in use moves nothing; the measured drag at the model's drag switches the thermal
  summary to the measured branch); a text input a text field with a note (library part or not,
  grade or not). Each row shows a dot when changed from the default, a
  reset button, and a tooltip with help, path, workbook cell, slider range and default.
- **Dashboard** (`dashboard.rs`): the 15 headline numbers in Python's order, the space claim
  badge and the end-effect flag (`DASHBOARD`). Green, amber or red badges come from the check
  verdicts (`verdict_level`). A value whose workbook cell the deviation registry ties to an applied
  correction carries the ids (`E3 E7 E8`) with the corrections, the at-defaults workbook and
  corrected values and the evidence in its tooltip (`corrections.rs`, M41-10). When f_end <= 0
  (audit M9) the rows computed from the pull-out (`END_EFFECT_ROWS`) are greyed without badges
  under the banner "End-effect model out of range"; the temperature rows that read the stored 3D
  fields carry "3D values from the workbook" (M3 comes after M4). `result_tooltip` is the one
  hover hook of every readout, keyed by result path.
- **Results table** (`results_table.rs`): every result with label, value, unit, workbook cell and
  marker; search by label, path or cell; CSV (`path,label,value,unit,cell`) and JSON (with the
  design that produced the results: in Torque -> Magnets the inputs with the free variable at the
  value shown) at full precision, a non-finite number written `+inf`, `-inf` or `NaN` (M41-15).
- **Session** (`session.rs`, `history.rs`): undo and redo (buttons, Ctrl+Z, Ctrl+Shift+Z, Ctrl+Y)
  of every change to the design, one step per settled edit (a drag, a typed value, a part name
  typed letter by letter), one per arrow nudge and one for a held arrow key's whole auto-repeat
  run (M41-14): an edit is in progress while a pointer button is down, a key is held down or a
  text field of an input row has focus (the results search is no edit); reset all; save and load
  a design file; a share link copied to the clipboard. A link the web app opens at start-up is
  the session's start (`open_share_payload`: the first Undo keeps it). A host with its own undo
  can keep the keys (`set_keyboard_shortcuts(false)`: M5's linkage window). One format for files and links: JSON
  `{"format": "magcoupling-design", "version": 1, "inputs": {path: value}, "sizing": {...}}` with
  every input (M41-5) and the sizing state (M41-4); a link is that JSON deflated and URL-safe
  base64 in `?m=` (the linkage tool's scheme; about 2.5 kB at the defaults). Loading is all or
  nothing (M41-6): an unknown path, a wrong type, a code outside its choices, a newer version or
  a malformed sizing state changes nothing and the panel names every problem, the inputs' and the
  sizing state's. A later version that renames or removes an input path adds a
  `session::PATH_MIGRATIONS` entry and bumps `DESIGN_VERSION`, so older files and links keep
  opening.
- **Sizing** (`sizing.rs`, Addendum A1): the mode switch tops the Key design group. In
  Torque -> Magnets the free variable (axial length, magnets per ring, ring radius) and the target
  hot-low torque (2.5 N·m to start, M41-9) feed `sizing::solve`, which runs once the design has
  been still for 0.25 s and no edit is in progress, never per frame. The panel then shows the
  design with the free variable at the solved value, "Solved at X", or at the best value it
  found, "Not reachable (best Y at X)" (M41-8); the free variable's row shows that value, locked
  (under the status line when the Key design group does not list it: the ring radius). Leaving the
  mode keeps the value, solving first a change still waiting for its debounce (M41-7).
- **No I/O in the panel.** Saving and picking a file are `PanelRequest`s the host does
  (`take_requests`, then `load_design_file` and `report`), so the M5 linkage window can host the
  panel with its own file handling; the share link base is the host's (`set_share_base`).
- **Theme** (`app/theme.rs`): the standalone app applies the linkage app's CAD dark visuals and
  spacing, dark whatever the system prefers (M41-3; a test keeps the visuals equal to
  `linkage-sim-rs/src/gui/theme.rs`), and the web page's background is the same panel colour.
  The panel sets no theme: in M5 the host's applies.

**Versions.** egui and eframe use the same 0.32 line as `linkage-sim-rs`, so
M5 embeds the panel with one egui; `Cargo.toml` has caret ranges, and
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/README.md`, replace:

```markdown
glue and the wasm are gitignored build outputs). Its canvas id is
`app::CANVAS_ID`, which a test checks. `deploy-web.yml` runs
`build_magcoupling_web.sh` after the linkage build, so both bundles ship
(`linkage.colesorkness.com/magcoupling/`).

**workbook-parity never ships.** The feature reaches `cargo test` and
`cargo clippy --all-targets` through the self dev-dependency, and cargo then
```

with:

```markdown
glue and the wasm are gitignored build outputs). Its canvas id is
`app::CANVAS_ID`, which a test checks. `deploy-web.yml` runs
`build_magcoupling_web.sh` after the linkage build, so both bundles ship
(`linkage.colesorkness.com/magcoupling/`) on the next push of `main`; pushing needs the user's
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it opens
`/magcoupling/` through a pinned share link and checks the canvas, the console lines
`magcoupling: loaded the design from the share link` and `magcoupling sizing: Solved at`, then
clicks "Load design" once (found in a screenshot), uploads a pinned design file through rfd's web
picker and checks `magcoupling: loaded a design file`, with zero console errors (a test decodes
the pinned link and the design file).

**workbook-parity never ships.** The feature reaches `cargo test` and
`cargo clippy --all-targets` through the self dev-dependency, and cargo then
```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/README.md`, replace:

```markdown
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300, with the default back iron and with none, where E17's free-space field inputs act (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a harmonic set outside its choices (NaN torques in the Calculator, the Calibration prototype and every sweep row, never another set, with the corrections off and on; `validate()` names it: `an_invalid_harmonic_set_is_nan_not_another_set`), a coercivity source outside its choices (any code but 1 uses the Hcj and beta inputs, and `validate()` names it: `an_invalid_coercivity_source_uses_the_inputs`), a positive beta typed in for magnets with no grade and no rating (no hot limit: C60 = +inf, C61 = NaN, the adhesive governs C12: `a_positive_beta_without_a_rating_has_no_hot_limit`), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), inverse sizing on designs no slider reaches (an outcome or an error for every free variable, never a panic: `sizing_never_panics_on_extreme_designs`), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells (hand-listed, or in the correction's golden file under `tests/data/deviations/`); a correction that changes more than 15 cells uses a golden file and one that changes 15 or fewer lists them (`broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`); each applied correction's figures match the report (`e1_adhesive_shear_modulus_matches_the_report`, `e2_clamp_screw_length_matches_the_report`, `e3_library_remanence_matches_the_report`), and help a correction rewords belongs to a real input and differs from the workbook's. A correction that changes nothing at defaults (E7 to E13, E15 to E20) carries registry probes, off-default inputs from the report whose cells are checked with the correction off and on (`each_probe_shows_its_correction`); `every_applied_engine_correction_is_visible_somewhere` requires every applied engine correction to show at defaults or in a probe. A probe's workbook value can be a workbook error (`Literal::Error`, e.g. `#DIV/0!` where Python raises): that side still runs but is not compared. E7, E8, E9, E10, E11, E12 and E13 must leave every default cell bit for bit (`e7_leaves_every_default_cell_bit_for_bit`, `e8_leaves_flat_blocks_bit_for_bit`, `e9_leaves_every_default_cell_bit_for_bit`, `e10_leaves_every_default_cell_bit_for_bit`, `e11_leaves_every_default_cell_bit_for_bit`, `e12_leaves_every_default_cell_bit_for_bit`, `e13_leaves_every_default_cell_bit_for_bit`). `e7_finds_the_peak_at_a_fill_of_exactly_0_4` pins E7 where the fifth harmonic vanishes (a ring at a fill of exactly 0.4). The Addendum A entries name every cell their probes change, downstream cells included (`addendum_entries_name_every_cell_their_probes_change`); E15 to E18 have rows in the Addendum A report and E19 and E20 cite decisions only (`e15_to_e18_have_audit_rows_and_e19_e20_decisions_only`); each reproduces the report: `e15_heat_capacity_matches_the_report`, `e16_removed_disc_matches_the_report`, `e17_aluminium_eddy_losses_match_the_report` and `e15_to_e17_together_match_the_reports_headline_table` (on top of E9, decision 15), `e18_aluminium_hub_mismatch_matches_the_report`, `e19_supermagnetman_arcs_follow_the_vendor_grid`, and for E20 `e20_each_part_uses_its_own_coercivity`, `e20_the_hcj_and_beta_inputs_override_the_grade_when_selected`, `e20_ferrite_is_limited_on_the_cold_side`, `e20_mixed_rings_use_the_weaker_grade` and `e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature`; `e15_leaves_every_default_cell_bit_for_bit` to `e20_leaves_every_default_cell_bit_for_bit` pin that none moves a default cell. `all_corrections_together_give_the_reviewed_headline` pins what users see: the headline with every correction on at the default design. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame (drawn text equals `compute_all` of the edited inputs and differs from the default's); pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept; "Reset all" restores the default design and headline; the panel draws inside an `egui::Window` (M5); each key input is a numeric input with a slider range; tooltips carry help, path and cell |
| `src/gui/format.rs` (feature `gui`) | Four significant digits, scientific outside 1e-3 to 1e6, carries (9.99996 shows as 10.00), signed zero, `+inf`, `-inf`, `NaN`, integers, text, None, units |
| `src/app.rs` (feature `app`) | The app draws the whole panel as a page; `web/magcoupling/index.html` has the canvas `CANVAS_ID` and the title `TITLE` |

### Regenerating test data

```

with:

```markdown
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300, with the default back iron and with none, where E17's free-space field inputs act (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a harmonic set outside its choices (NaN torques in the Calculator, the Calibration prototype and every sweep row, never another set, with the corrections off and on; `validate()` names it: `an_invalid_harmonic_set_is_nan_not_another_set`), a coercivity source outside its choices (any code but 1 uses the Hcj and beta inputs, and `validate()` names it: `an_invalid_coercivity_source_uses_the_inputs`), a positive beta typed in for magnets with no grade and no rating (no hot limit: C60 = +inf, C61 = NaN, the adhesive governs C12: `a_positive_beta_without_a_rating_has_no_hot_limit`), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), inverse sizing on designs no slider reaches (an outcome or an error for every free variable, never a panic: `sizing_never_panics_on_extreme_designs`), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells (hand-listed, or in the correction's golden file under `tests/data/deviations/`); a correction that changes more than 15 cells uses a golden file and one that changes 15 or fewer lists them (`broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`); each applied correction's figures match the report (`e1_adhesive_shear_modulus_matches_the_report`, `e2_clamp_screw_length_matches_the_report`, `e3_library_remanence_matches_the_report`), and help a correction rewords belongs to a real input and differs from the workbook's. A correction that changes nothing at defaults (E7 to E13, E15 to E20) carries registry probes, off-default inputs from the report whose cells are checked with the correction off and on (`each_probe_shows_its_correction`); `every_applied_engine_correction_is_visible_somewhere` requires every applied engine correction to show at defaults or in a probe. A probe's workbook value can be a workbook error (`Literal::Error`, e.g. `#DIV/0!` where Python raises): that side still runs but is not compared. E7, E8, E9, E10, E11, E12 and E13 must leave every default cell bit for bit (`e7_leaves_every_default_cell_bit_for_bit`, `e8_leaves_flat_blocks_bit_for_bit`, `e9_leaves_every_default_cell_bit_for_bit`, `e10_leaves_every_default_cell_bit_for_bit`, `e11_leaves_every_default_cell_bit_for_bit`, `e12_leaves_every_default_cell_bit_for_bit`, `e13_leaves_every_default_cell_bit_for_bit`). `e7_finds_the_peak_at_a_fill_of_exactly_0_4` pins E7 where the fifth harmonic vanishes (a ring at a fill of exactly 0.4). The Addendum A entries name every cell their probes change, downstream cells included (`addendum_entries_name_every_cell_their_probes_change`); E15 to E18 have rows in the Addendum A report and E19 and E20 cite decisions only (`e15_to_e18_have_audit_rows_and_e19_e20_decisions_only`); each reproduces the report: `e15_heat_capacity_matches_the_report`, `e16_removed_disc_matches_the_report`, `e17_aluminium_eddy_losses_match_the_report` and `e15_to_e17_together_match_the_reports_headline_table` (on top of E9, decision 15), `e18_aluminium_hub_mismatch_matches_the_report`, `e19_supermagnetman_arcs_follow_the_vendor_grid`, and for E20 `e20_each_part_uses_its_own_coercivity`, `e20_the_hcj_and_beta_inputs_override_the_grade_when_selected`, `e20_ferrite_is_limited_on_the_cold_side`, `e20_mixed_rings_use_the_weaker_grade` and `e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature`; `e15_leaves_every_default_cell_bit_for_bit` to `e20_leaves_every_default_cell_bit_for_bit` pin that none moves a default cell. `all_corrections_together_give_the_reviewed_headline` pins what users see: the headline with every correction on at the default design. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input, a 1280 x 1024 screen): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame; the value lands on the step's decimals (1.41) and stepping back lands on the default exactly; a typed value snaps to the step and is clamped into the range; pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept until edited and flagged; the changed dot and per-field reset; the axial length override starts blank and enters at the ring's length; a selector switches the branch (no back iron); a text input edits the part name and its note follows; every group opens and draws every input; the dashboard's stored-3D label, markers, end-effect banner (over the dashboard and the table) and space-claim overshoot; the results table's search and its export requests; undo and redo (buttons and shortcuts; one step per drag, per nudge, per typed name; Ctrl+Z in a text field left to the field; reset all undoable); save and load requests, a loaded and a refused design file, the share link on the clipboard and a broken link; the host's report; Torque -> Magnets: the solved design shown, the debounce (no solve per frame, none while a button is down, a repaint after a solve), the locked free-variable row (also under the status line for the ring radius), leaving the mode keeps the value as one undo step and solves a pending change first, an unreachable target shows the best value, the free-variable picker, a share link carries the sizing state, the JSON export holds the design shown, typing in the results search does not defer the solve; a held arrow key is one undo step, a slider's value box being typed in is an edit of the design (one step on Enter), the results search holds back no undo step, only this frame's rows count as design fields, a start-up share link is no undo step, a host can keep Ctrl+Z for itself; idle frames change nothing with every group open and off-grid values; the measured drag enters at the model's drag, unrounded; every drawn text and hover text has glyphs in egui's default fonts; the panel draws inside an `egui::Window` (M5) |
| `src/gui/inputs.rs`, `src/gui/dashboard.rs`, `src/gui/corrections.rs`, `src/gui/results_table.rs` (feature `gui`) | Every input in exactly one section in schema order, every heading used, the Key design list, every input type covered; step decimals; every slider default on its step grid except the listed two; the optional seeds (entering the axial length at its seed moves nothing); text hints; tooltips. The dashboard starts with `HEADLINE`; every verdict each check gives has a level (designs that reach each branch); `END_EFFECT_ROWS` are exactly the headline rows that move with c_end (0 included: it flips the hot-minimum verdict) at back iron 1 and 0, `STORED_3D_ROWS` exactly those that move with the 14 stored 3D inputs at their slider ends; greying drops badges. The compiled golden files are the registry's; the headline pull-out carries E3 with the workbook's 2.647. The table lists every result once; the search; exact numbers read back bit for bit; CSV quoting; `+inf` and `NaN` in CSV and JSON (E20's positive-beta design); the JSON export holds the design |
| `src/gui/session.rs`, `src/gui/history.rs`, `src/gui/sizing.rs` (feature `gui`) | A design file and a share link round-trip bit for bit (every input type, the sizing state); a file names every input; a missing path or sizing state takes the default; every problem reported, nothing loaded, the inputs' and the sizing state's together; an older file's renamed and removed paths migrate (a made-up table) and `PATH_MIGRATIONS` leads to inputs of this version; not a design, a newer or malformed version, a malformed sizing state, a broken link, a link that inflates past 1 MB are refused; the default link stays under 2,500 characters. Undo and redo walk settled steps, coalesce an unsettled edit, drop redo on a new change, cap at 100. The runner waits for the debounce and runs once, restarts on a new change or an edit in progress, ignores the free variable's own value, solves a pending change at once on request; solved, unreachable and refused outcomes and their status lines |
| `src/gui/format.rs` (feature `gui`) | Four significant digits, scientific outside 1e-3 to 1e6, carries (9.99996 shows as 10.00), signed zero, `+inf`, `-inf`, `NaN` (one text for the display and both exports), integers, text, None, units |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the CAD dark theme; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the visuals are `linkage-sim-rs`'s, function body for body; the theme stays dark when the system turns light |

### Regenerating test data

```

In `C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/README.md`, replace:

```markdown
| 3 | `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells` | `tests/data/deviations/E<k>.json`, the golden files of the broad corrections (E3, E4, E5); see [Deviations](#deviations). It writes what this engine computes with the correction alone, so review the diff |

Run 1, then 2, when an input, a range, a choice or a table layout changes; run 3
when a broad correction changes cells. A Rust-only input (declared with
`param_rust_only`, exported with `"rust_only": true`) is left out by generator 2:
the Python engine has no such input, so every case keeps its Rust default and no
data file changes when one is added.
```

with:

```markdown
| 3 | `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells` | `tests/data/deviations/E<k>.json`, the golden files of the broad corrections (E3, E4, E5); see [Deviations](#deviations). It writes what this engine computes with the correction alone, so review the diff |

Run 1, then 2, when an input, a range, a choice or a table layout changes; run 3
when a broad correction changes cells. The `gui-smoke` share link
(`MAGCOUPLING_SMOKE_PAYLOAD` in `.claude/workflows/gui-smoke.js`) carries every input (decision
M41-5): regenerate it with `MagcouplingPanel::share_link` (the default design, face gap 1.5 mm,
Torque -> Magnets on the axial length at 2.5 N·m) when the design format changes, a default
changes, or an input path is renamed or removed; `the_smoke_test_share_link_opens_its_design`
fails until then. A Rust-only input (declared with
`param_rust_only`, exported with `"rust_only": true`) is left out by generator 2:
the Python engine has no such input, so every case keeps its Rust default and no
data file changes when one is added.
```

- [ ] **Step 2: Commit this plan beside the earlier ones**

Run:

```bash
cp C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/m41-plan/plan.md C:/Users/Cole/source/repos/lsim-mag-m41/docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md
ls -la C:/Users/Cole/source/repos/lsim-mag-m41/docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md
```

Expected: the file is listed (the README and the tracker link it). If the scratch copy is gone, copy the plan from wherever the controller keeps it.

- [ ] **Step 3: Check the decision ids**

Every id the code and the docs cite must have its row in the Decisions to confirm, and every row must be cited.

Run:

```bash
grep -rhoE "M41-[0-9]+" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/src C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/README.md C:/Users/Cole/source/repos/lsim-mag-m41/docs/ai | sort -u | sort -t- -k2 -n | tr '\n' ' '; echo
grep -oE "^\| M41-[0-9]+" C:/Users/Cole/source/repos/lsim-mag-m41/docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md | sort -u | wc -l
```

Expected: `M41-1 M41-2 M41-3 M41-4 M41-5 M41-6 M41-7 M41-8 M41-9 M41-10 M41-11 M41-12 M41-13 M41-14 M41-15`, then `15`.

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml
```

Expected: no output (every block above is already in rustfmt's layout).

The gate runs the linkage crate's gates 1 to 3, then the magcoupling gates: 4 to 6 the engine (test, clippy, wasm32), 7 `cargo test --features app` (this plan's headless egui tests), 8 clippy with `app`, 9 the wasm32 clippy of the panel and the web entry, 10 the workbook-parity guard, 11 the lock parity, 12 the Python oracle.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m41/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task11.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task11.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task11.log
git -C C:/Users/Cole/source/repos/lsim-mag-m41 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing. If the gate fails, read the log (`C:/Users/Cole/source/repos/lsim-mag-m41/magcoupling-rs/target/gate-task11.log`), fix within this task's files, and run it again; do not commit red.

- [ ] **Step 5: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add docs/ai/02-system.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add docs/ai/04-memory.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add docs/ai/05-update-tracker.md
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-m41 add docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md
git -C C:/Users/Cole/source/repos/lsim-mag-m41 commit -F - <<'EOF'
docs(magcoupling): M4-1 status, the panel, invariants and open items

README: status, layout, "The panel" (inputs, dashboard, results table,
session, sizing, theme; the file formats; decisions M41-1 to M41-15) and the
test rows; when to regenerate the smoke link. docs/ai: the panel's invariants
and six lessons, the gui and app structure, the open M4 items resolved (8bad022's
included) and the follow-ups recorded, the tracker entry. The plan itself. Nothing pushed: the next push of main deploys
/magcoupling/ and needs the user's go.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m41 status --short
```

Expected: one new commit on `magcoupling/m4-1`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

## Self-review record

- **Spec coverage.** Layout: left inputs from the metadata, grouped as the package groups them, Key design on top (Tasks 3, 4); right dashboard with badges from the check verdicts and the corrected-vs-workbook markers (Task 5); the results table with label, unit, cell, search and CSV/JSON export (Task 6). The geometry view, the plot tabs and the clamp drawing are M4-2 (the centre's `CentreView` takes them). Sliders: live recompute, value box, arrow nudges, step snapping (even poles), log scale, per-field reset, the changed dot, tooltips with help and cell, selectors as drop-downs (Task 4). Session: undo/redo, reset all, design JSON, the `?m=` share link (Tasks 1, 2, 7); theme (Task 9). Addendum A1: the mode switch, the free variable, the target, "solved at X" and "not reachable (best Y)" (Tasks 1, 8); the space claim badge (Task 5). The user's 2026-09-30 decisions: the stored-3D label and the end-effect greying (Task 5). Testing summary: slider change updates results, reset restores the default, selector switches branch, undo reverses a change (Tasks 4, 7); the 3D badge lifecycle becomes the stored-3D label (Task 5, until M3); `gui-smoke` loads `/magcoupling/` (Task 10).
- **Open M4 items of `04-memory.yaml`.** The axial length mapping (the A-2 override, Task 3), slider float noise (M41-1), typed-value clamping (M41-2), the light/dark behaviour (M41-3), the sizing state in files and links (M41-4), `gui-smoke` for `/magcoupling/` (Task 10) and `8bad022`'s exact-equality item (M41-1 and the idle test) are resolved; Task 11 records them, with the vacuum permeability's off-grid defaults and M5's undo keys as open items.
- **Placeholders.** None: every step carries its code or command and its expected output.
- **Type consistency.** The names each task's Interfaces block produces are the ones later tasks consume (checked by the replay, which compiles every task's state).
- **Review Focus.** Each of the six classes has its test in the owning task (Tasks 1, 4, 6, 7, 8).

## Execution handoff

Plan complete. The controller asks the user to review it, records the answers to the Decisions to confirm (Task 0, Step 8), and runs it subagent-driven: a fresh `sonnet` subagent per task with a fresh reviewer per task (on `sonnet`, except the reviews of Tasks 5 and 8 on the session model: Global Constraints, Model tiers), then one whole-branch review on the session model. Nothing is pushed: `deploy-web.yml` ships `/magcoupling/` on the next push of `main`, which needs the user's explicit go.

## Self-review record (revision 2: the critic's findings)

The plan was revised against eighteen critic findings, one of them blocking. Each change sits in the task that owns the code; the task history was rebuilt on `8bad022`, the red states rebuilt from it, this plan regenerated from the proven commits, and the whole plan replayed again (Verification record). Per finding: the fix, what pins it, and how it was verified.

| # | Finding | Fix (task) | Pinned by | Verified |
|---|---|---|---|---|
| 1 | **Blocking**: Task 0 Step 2 diffed from `167db09`, so it would stop on `main` at `8bad022` (two lines of `docs/ai/04-memory.yaml`) | The base is `8bad022`: Step 1's expected log line, Step 2's diff, Task 0's Interfaces, Global Constraints, the Verification record. Task 11's `04-memory.yaml` block marks `8bad022`'s exact-equality item RESOLVED (M41-1) and records the one caveat (finding 3) as open | Task 0 Step 2; Task 11's block quotes `8bad022`'s lines as its old text | Replayed on an `8bad022` export; `git diff --stat 167db09 8bad022` lists only `docs/ai/04-memory.yaml`, 2 insertions |
| 2 | M41-12 said entering an optional input moves nothing; untrue for the measured drag | M41-12 and the `OPTIONAL_SEEDS` docs reworded; the seed is clamped into the range, not rounded (`panel::seed`, Task 4) | `entering_the_measured_drag_starts_at_the_model_s_drag` (Task 4): the input is the unrounded 0.013136543139723558, the drag in use stays, the steady high case 92.51 -> 74.17 °C, the slip loss 2.751 W | Measured in the tree: 25 results move at the unrounded seed, 35 at the rounded 0.0131 |
| 3 | `coupling.mu0` and `calibration.mu0` default off their step grid | `every_slider_default_is_on_its_step_grid_except_the_listed` with `OFF_GRID_DEFAULTS` (Task 3); M41-1 restated; the open item asks for a 1e-12 engine step (Task 11). Option (c), `max_decimals` from the default, is not taken: it changes only the value box's text, the engine step is the real fix, and `slider()` would need the default in its signature | The grid test fails if another default leaves its grid, or once the engine step is fixed | The test finds exactly the two; neither is a flagged assumption, so `assumptions::modified` is unaffected today |
| 4 | `idle_frames_change_no_input` never exercised a snap | Every group open, bottom up; face gap 1.4123, measured drag 0.012345; both mu0 rows drawn (Task 4; a tall screen from Task 6; opened through `open_design` with `undo_len() == 0` from Task 7) | The test | Its first draft (groups opened top down) failed: a click aimed at the next group's label landed on a slider of the group opening above it. Bottom up fixed it; `02-system.yaml` records the lesson |
| 5 | `editing()` counted the results search box as an edit | An edit in progress counts only the text fields of input rows (their widget ids, gathered each frame: a Slider's response takes its value box's id while that has focus) and the target's value box (Tasks 7, 8) | `a_slider_s_value_box_being_typed_in_is_an_edit_of_the_design` (the egui behaviour the ids rely on), `a_focused_results_search_holds_back_no_undo_step`, `only_the_rows_drawn_this_frame_count_as_design_fields` (Task 7); `typing_in_the_results_search_does_not_defer_the_solve` (Task 8) | The rows test caught a leak in the first draft: the id list was cleared inside the Key design group only |
| 6 | Leaving Torque -> Magnets during "Solving..." wrote the previous design's solution | `SizingRunner::solve_now`, run once on leaving when the outcome is stale, and logged (Task 8); M41-7 restated | `solve_now_solves_a_pending_change_at_once`, `leaving_torque_to_magnets_while_solving_writes_the_current_solution` | The panel test compares with a fresh `solve` at 3.0 N·m |
| 7 | A file with bad inputs and a bad sizing state named only the inputs | `LoadError::Refused { inputs, sizing }` replaces `Inputs` and `Sizing`; `sizing_from` collects every problem; both are checked before refusing; `Display` is unchanged when only the inputs fail (Task 1) | `a_file_with_bad_inputs_and_a_bad_sizing_state_names_both`; the two-message case in `a_malformed_sizing_state_is_refused`; `load_errors_read_as_sentences` | The panel tests that compare `inputs refused: ...` text pass unchanged |
| 8 | The JSON export in Torque -> Magnets embedded the inputs, not the design shown | The export embeds `Design { inputs: shown_inputs(), sizing }` (Task 8) | `the_json_export_in_torque_to_magnets_holds_the_inputs_that_produced_the_results` | `compute_all` of the exported design equals the panel's results |
| 9 | The panel consumed Ctrl+Z from the whole context (M5 would undo twice) | `set_keyboard_shortcuts(enabled)`, on by default (Task 7). A pointer gate was not taken: it would disable Ctrl+Z in the standalone app whenever the pointer leaves the canvas. M5's host must still gate its own non-consuming `key_pressed` (open item, Task 11) | `a_host_can_keep_ctrl_z_for_itself`: in an `egui::Window`, off leaves the event to the host and undoes nothing; on undoes and spends it | |
| 10 | DRY: `+inf`/`-inf`/`NaN` spelled three times; two `json_value`s converting opposite ways | `format::non_finite_text`, used by `format_number`, `json_number` (Task 1) and `exact_number` (Task 6); the corrections helper is `value_from_json` (Task 5) | `only_non_finite_numbers_have_a_non_finite_text`, and the display, JSON and CSV tests | |
| 11 | A renamed or removed input path would break every older file and link | `PathMigration` and `PATH_MIGRATIONS` (empty), applied in `inputs_from` by the file's version; chains followed; an old and a new name of one input in one file refused (Task 1). A rename needs an entry and the version bump (`DESIGN_VERSION` docs, README) | `an_older_file_s_renamed_and_removed_paths_are_migrated` (a made-up table); `every_path_migration_leads_to_an_input_of_this_version` guards future entries | |
| 12 | The web file picker was unproven | The app logs `DESIGN_FILE_LOADED` for a picked file (`open_picked_file`, Task 9). The smoke step clicks "Load design" once, uploads the pinned `MAGCOUPLING_SMOKE_DESIGN_FILE` through rfd's overlay and checks the log line; it says what to report when the canvas click fails (Task 10) | `a_picked_design_file_loads_and_a_refused_one_changes_nothing`; `the_smoke_test_share_link_opens_its_design` decodes the pinned file | Done by hand with Playwright on the built bundle, exactly as the step words it (Verification record) |
| 13 | Docs drift: "Metal design C5"; when to regenerate the smoke link | C7 (Task 1). The workflow comment and the README's "Regenerating test data" add a changed default and a renamed or removed path (Tasks 10, 11) | | |
| 14 | A held arrow key filled the 100 undo levels in about 3 s | A key held down is an edit in progress (egui's `keys_down`), so the auto-repeat run is one step (Task 7); M41-14 restated. The panel tests tap keys, Ctrl+A included (`key_tap`, `select_all`, Task 4): egui keeps a pressed key down until its release and reads later presses as repeats, so press-only events would never settle | `a_held_arrow_key_is_one_undo_step` (20 presses, one release, one step) | Every nudge test still gives one step per tap |
| 15 | The ring radius's locked row sat in the closed Coupling group | The free variable's row is drawn under the status line when `KEY_DESIGN` lacks it (Task 8) | `the_free_variable_picker_switches_the_variable` (the note and the label drawn once, Coupling closed) | |
| 16 | A share link opened at start-up was an undo step | `open_design` and `open_share_payload` start a new history (Task 7); the app's `open_share_payload` and the web entry use it (Task 9) | `a_share_link_opened_at_start_up_is_not_an_undo_step` (Task 7); `a_share_link_opens_its_design` (Task 9: the first Ctrl+Z keeps it) | The smoke's first screenshot shows Undo disabled |
| 17 | Sequencing with plan A-3 | Merge order (M4-1 first), what to re-derive if A-3 lands first, and which assertions move with the results layout (header) | | `git diff --stat main...magcoupling/addendum-a3` read for the files it touches |
| 18 | The per-task reviewer's model was unspecified | Global Constraints, Model tiers: `sonnet`, except the reviews of Tasks 5 and 8 on the session model, with the comment; the Execution handoff says the same | | |

Found while revising, beyond the findings: `inputs::into_range` lost its only caller when the seed stopped rounding, so Task 3 no longer adds it (the grid test rounds by itself); the Files and Interfaces of Task 4 named `SCREEN` and `sized_frame`, which Task 6 adds, and Task 6's omitted them (corrected); the value-box test caught `select_all` pressing Ctrl+A without releasing it (A stayed down, so nothing settled), so it taps too; the doc comment that refused inputs are named "in the file's order" was wrong (serde_json orders keys) and now says path order. `docs/ai/02-system.yaml` does not parse as YAML at its line 14, before this plan as after; Task 11's additions to it are plain strings.
