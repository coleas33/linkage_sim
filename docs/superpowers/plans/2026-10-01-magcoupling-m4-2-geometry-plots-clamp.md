# Magcoupling M4-2: Geometry View, Plots and Clamp Drawing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the panel's centre region with the views M4 still lacks: the geometry view to scale (the default tab), five plots, the clamp drawing and table, and a results table that fits a narrow window, in the native and the web app.

**Architecture:** Each view is a pure builder plus a thin egui painter. `gui/geometry.rs` turns the design shown (its inputs and the results `compute_all` already computed this frame) into the end view, the side view and the notes in millimetres, every dimension read from the results (the axial ones from the housing in effect, `housing.*`), with dimension callouts that carry the result path they show; `gui/geometry_view.rs` paints both views at one scale, lists the callouts and hovers each one's result text through `dashboard::hover_text` (the hook M4-3's equation tooltip joins). `gui/plots.rs` builds each plot's series from the same results with the engine's own closed forms (pinned by tests to the engine's torques) and draws them with egui_plot 0.33, locked to linkage-sim-rs's version. `gui/clamp_drawing.rs` ports `drawing.py` to marks in millimetres and paints them. The panel's `CentreView` gains the tabs, the end-effect banner moves over the whole centre region, and nothing expensive runs per frame: no extra `compute_all`, no solve, only the tab shown builds its series.

**Tech Stack:** Rust 2024 (rustc 1.89); egui and eframe 0.32.3 and egui_plot 0.33.0 (locked to linkage-sim-rs's versions); headless egui tests (`egui::Context::run` with injected input, shapes inspected); Playwright MCP for the web check; Git Bash for every command.

**Spec:** `C:/Users/Cole/source/repos/lsim-mag-m42/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`: M4 "Layout" ("Center — geometry view, to scale: end view of both rings with blocks on polygon faces, cup, sleeve, liner, shaft and key; side view of the axial stack against the 20 mm bay and 35 mm overall length. Redraws live during drags. Dimension callouts for face gap, corner gap, running clearance; violations (negative clearance, envelope exceeded) draw red"; "Bottom — plot tabs: torque vs temperature (hot/cold allowance band, requirement line); gap sweep; pole sweep; slip heating over time vs limit; torque vs rotation angle (pull-out point); clamp table plus the clamp drawing (end and top views, egui painter port of `drawing.py`)") and Addendum A1 ("Housing autofit ... the geometry view redraws them live"; "Space claim ... drawn as a dashed outline. Exceeding it shows a red callout on the view naming the overshoot in mm per axis"). The user's decisions recorded in `docs/ai/04-memory.yaml`: audit M9, f_end <= 0 greys the numbers computed from the pull-out (2026-09-30); M4 draws the axial housing in effect (A-2 decision A2-8, decision 28); the two workbook design inconsistencies, the M41 cap thread below the 41.33 mm cup body OD and the 22 mm vs 25 mm boss OD on Metal design vs Shaft clamps, are flagged as design-check notes with no engine change (2026-10-01). Python reference of the clamp drawing: `C:/Users/Cole/source/repos/lsim-mag-m42/reference/magcoupling-py/magcoupling/drawing.py`. Conventions: plan M4-1, `C:/Users/Cole/source/repos/lsim-mag-m42/docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md`. Scope: the second of three M4 plans; M4-3 is the equation explorer, the assumptions panel, the teaching notes and the material and grade pickers.

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-m42`, branch `magcoupling/m4-2`, created by Task 0 from `main` at `93120f5` (the merge of M4-1, `42ab370`, plus a test fix, `22630af`, and two docs commits). Every command uses absolute paths into it. Nothing is pushed.

**Found while planning (fixed by Task 6):** `docs/ai/02-system.yaml` on `main` holds three unresolved merge-conflict hunks that the M4-1 merge (`42ab370`) left in the magcoupling block (the responsibility text and `key_files`, the invariants) and in its `current_statuses` line (`<<<<<<< HEAD` ... `>>>>>>> magcoupling/m4-1`). Task 6 edits exactly those lines, so it resolves them by keeping both sides (the A-3 and the M4-1 text) and adding M4-2; it rewords the M4-1 sentence ("grows the panel: every input") so the block parses as YAML. The rest of `02-system.yaml` has plain scalars with `: ` that YAML rejects (line 14 onwards); Task 6 does not touch them.

## Decisions to confirm

**Confirmed (user, 2026-10-01): the recommended option on all of M42-1 to M42-9.** No task stops on a decision.

These choices arose while turning the spec into code; no approved decision settles them. The plan implements the recommended option of each (the code comments and the README cite the ids). Task 0 records the user's answers; if the user picks another option, the task that implements it stops and escalates instead of improvising.

| Id | Question | Recommended (implemented here) | Alternatives | Tasks |
|---|---|---|---|---|
| M42-1 | How do the new views share the centre region? | **One tab row of eight views** (Geometry, Torque vs temperature, Gap sweep, Pole sweep, Slip heating, Torque vs rotation, Clamp, Results table), wrapping when the region is narrow; the end-effect banner sits over every view (and over the dashboard, as before), so the results table no longer draws its own | (b) the spec's literal layout: the geometry view above, the plot tabs in a resizable bottom panel of the centre region (both get half the height of a ~660 px wide region); (c) one "Plots" tab with sub-tabs | 2, 3, 4 |
| M42-2 | Which view opens first? | **The geometry view** (the spec's centre view); the results table stays a tab | (b) the results table, as in M4-1 | 2 |
| M42-3 | How is the geometry drawn? | **Both views side by side at one scale**, the largest that fits; the end view seen from the free end, face 0 at the top (the face gap there; the corner gap along face 1's normal, from the dashed circle its inner block's corners sweep to its outer block's flat, so both ends sit on drawn features and the line is the corner gap long; the running clearance at the bottom; the key at 0°); the side view is the **upper half section through the flat centres**, the cap at the left, and the parts inside the cup cavity (outer and inner blocks, liner, sleeve, hub) **centred on the cavity**, since the workbook gives no axial positions; arcs (`coupling.faceted` other than 1, as the engine reads it) are sectors of the block width at their mid-radius | (b) each view at its own scale (a larger end view); (c) a full side section; the cavity parts against the cap | 1, 2 |
| M42-4 | How are callouts and violations marked? | **Tagged dimension lines on the drawing, the callout texts listed under the views** (tag, text, hover text). A callout is **red** when violated: a gap below zero, the running clearance below its target (the dashboard's clearance check, red there too), a space-claim axis exceeded (its dashed line red too); **amber** when its value is not a number; otherwise the text colour (lines in a light dimension blue). The space claim's three axes are callouts whether or not exceeded ("axial stack 31.80 mm of 35.00 mm"; "is 34.90 mm over the 35.00 mm claim") | (b) red only below zero (the spec's literal "negative clearance"), amber below the target; (c) texts beside each dimension on the drawing (crowded at the ~7 points per mm of a 1280 px window) | 1, 2 |
| M42-5 | Plot axes, units and series | **Torque vs temperature:** x magnet temperature [°C] from `metal.min_temp_C` to 10 °C past the governing limit (or the operating temperature), 121 samples; y torque [N·m]: the pull-out, the band of the variation allowance as two thin lines (egui_plot's `Polygon` must be convex), the required minimum `metal.required_min_Nm`, the operating temperature and the governing limit, the design's point. **Sweeps:** y the pull-out at the operating temperature (column X) against the swept variable, a marker per row by status (green nominal, amber below the hot minimum, red a fit failure), the required floor `model.required_floor_Nm` (what the status compares), the design shown as a hollow marker. **Slip heating:** x time slipping [s] over five thermal time constants (longer to show the time to the limit), the estimate and the high case against the governing limit, the time to the limit marked. **Torque vs rotation:** x the relative rotation in mechanical degrees over one pole pair (720/N), 361 samples; y the torque at the operating temperature; the pull-out point at the E7 angle | (b) electrical degrees; (c) minutes on the heating axis; (d) the band as a filled area (needs a mesh per segment) | 3 |
| M42-6 | How far does the end-effect greying reach in the plots? | **The torque plots computed from the pull-out** (temperature and rotation) draw in the weak colour when f_end <= 0; each sweep row is grey when its own f_end <= 0 (`model::end_effect_in_range`); slip heating does not read the pull-out and stays coloured | (b) hide the torque plots behind the banner | 3 |
| M42-7 | The clamp tab | **drawing.py's pens mapped to the dark theme** (ink the strong text colour, cut red `#e15f50`, hidden the weak text colour, dimensions light blue; white fills the drawing background, the light cut fill translucent red), both views at one scale, texts scaled with the drawing (8 to 12 points); drawing.py's `ValueError` text when no screw size fits; the two-piece clamp drawn as the one-piece with a note (drawing.py's only layout); the clamp table: the Shaft clamps summary (11 results), the machining steps, then the 'Clamp screw sizes' table (collapsed by default) with the recommended size's column in green | (b) a light "paper" background as matplotlib's; (c) the screw table open by default | 4 |
| M42-8 | The results table at narrow widths (M4-1 review: a ~930 px window showed only the label column) | **The label column flexes**: the value (130), cell (130) and marker (60) columns keep their M4-1 widths and the label takes the rest, at least 120 points; the order is unchanged; narrower still, the rows scroll sideways | (b) also move the marker before the cell; (c) drop the cell column below a width | 5 |
| M42-9 | The design checks and the autofit wall | **Amber notes under the geometry view while each inconsistency holds** (the cap thread diameter below the cup body OD; `metal.boss_od_mm` differing from `clamps.boss_od_mm`), the autofit wall a weak hint ("the wall rule asks for at least 2.000 mm at the pocket corners (input 1.800 mm)") whenever the wall rule applies (not without back iron) | (b) always shown; (c) dashboard badges | 1 |

Settled by this plan's design without a new decision (each is stated where it is implemented): hover texts are built only while hovered; a piece or dashed line holding a number that is not finite is not drawn (a struct literal can hold NaN; `validate()` refuses it everywhere else), a note counts them per view, and a view with no finite extent says so; each view's extent comes from its pieces, and a space-claim line farther than `CLAIM_REACH` (10) times the pieces' extent along its axis is left off with a note (a design file can hold a finite but huge claim; drawn, it would shrink the pieces to nothing), the diameter decided once for both views; a region with no room paints no view (`side_by_side` returns no scale); blocks are drawn for 2 to 200 poles (a design file can hold any count), screws for at most 50; the plots leave out points and reference lines that are not finite and show "Nothing to plot" when none is left; plot ids (`PlotKind::id`, from `pub const` names) and legend names are constants the tests read; the pole sweep's line is named "Pull-out torque (rows at their smallest fitting apothem)", since its rows are the engine's (`sweeps::pole_sweep`: each count at its smallest fitting apothem) and the design's own marker can sit off that line (2.688 N·m against the 10-pole row's 2.608 at the defaults); the clamp's top view draws drawing.py's `range(int(row.screws_needed))` screws (whatever fits), within that bound of 50; the egui_plot dependency lives in feature `gui` (the panel draws the plots; M5 embeds it with linkage-sim-rs's same egui_plot).

## Global Constraints

Every task's requirements implicitly include this section.

- The engine does not change: no file under `magcoupling-rs/src/engine/` or `magcoupling-rs/tests/` is edited. Workbook parity (1,149 checks), the differential data (`gen_differential.py --check`), every registry test and `all_corrections_together_give_the_reviewed_headline` pass unchanged. `reference/magcoupling-py/` is not edited.
- Feature boundaries: the engine stays pure std (gate 6). Feature `gui` adds egui_plot to its egui, serde_json, base64, flate2 and log, and does no I/O. `workbook-parity` never reaches a shipped build (gate 10).
- Versions: egui_plot `0.33` in `Cargo.toml`, locked at linkage-sim-rs's `0.33.0` (its dependencies `ahash`, `egui` and `emath` are already locked); gate 11 checks it beside egui, eframe and wasm-bindgen. Task 3 resolves it against crates.io: never pass `--offline`. Expected `Cargo.lock` change: `egui_plot 0.33.0` added, and only that.
- Per frame: the panel still calls `compute_all` once per frame (on the design shown, bound once and passed to the centre region); the geometry and the plot series are built from those results, only for the tab shown; nothing calls `sizing::solve` or builds a registry per frame (the clamp table's rows are built once, `OnceLock`).
- The panel's host API does not break: `MagcouplingPanel::new`, `ui`, `inputs`, `results`, `reset` keep their signatures; `CentreView` gains variants (no code outside this crate matches on it yet; M5 has not started).
- UI text: egui's default fonts have no U+2192, U+2264 or U+2212; the blocks write `->`, `<=` and `-`. `°`, `Ø` and `·` are in the default fonts; `every_text_the_panel_shows_has_glyphs_in_the_default_fonts` clicks every tab and checks every drawn and hover text.
- Gate runs (Task 0, Task 3 and Task 6; Tasks 1, 2, 4 and 5 run the magcoupling checks of their Step "Format and lint", which are gates 4 to 9's commands): `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m42/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate` line; then `git -C C:/Users/Cole/source/repos/lsim-mag-m42 checkout -- docs/chebyshev_lambda` (the linkage tests rewrite those PNGs; never commit them).
- `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml` before every commit, and `cargo fmt --check` clean. Every block below is already in rustfmt's layout.
- The blocks quote the files with LF line endings, as git stores them (Task 0 creates the worktree with a worktree-scoped `core.autocrlf=false`: this machine's system gitconfig sets it to true). "Create" writes a new file; "In ..., replace: ... with: ..." is one exact replacement whose old text occurs exactly once in the file at that point (each block is a whole-line hunk with its context). Apply a step's blocks in the order given. If a block's old text is not found, stop and escalate; do not improvise a match (Task 0 checks that the files the blocks touch are still those of `93120f5`).
- Windows paths: the worktree path is short on purpose. A release build under a deep directory fails with `LNK1104` (MAX_PATH); `build_magcoupling_web.sh` writes to `magcoupling-rs/target/` inside the worktree.
- Tests that read source files must normalize CRLF (the main checkout is CRLF). This plan adds none; the existing `the_smoke_test_share_link_opens_its_design` reads `gui-smoke.js` and is unaffected by Task 6's edit (it reads the two constants and three log lines, which Task 6 keeps).
- Every commit message ends with exactly two lines: a `Co-Authored-By:` line naming the model that wrote the commit, then `Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9`. The blocks write `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; a `sonnet` executor writes its own model's name. Subjects: `feat(magcoupling-rs): ...` (Tasks 1 to 5), `docs(magcoupling): ...` (Task 6). Nothing is pushed.
- Docs change with code (repo rule, `docs/ai/01-meta.yaml`; the user's rule that every code change ships with its docs): Tasks 1 to 5 are intermediate commits on `magcoupling/m4-2`, and the branch is reviewed and merged only as a whole, after Task 6, so no code change reaches `main` without its docs. Task 6 updates `magcoupling-rs/README.md` and `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md` (YAML scalars holding `: ` are quoted), the `gui-smoke` workflow, and commits this plan; the branch is not merged without it. Read `docs/ai/*.yaml` before starting (Task 0).
- Deploy: `deploy-web.yml` on `main` builds and ships `/magcoupling/` on the next push of `main`; this plan pushes nothing.
- Model tiers (CLAUDE.md section 5; the user's lean process of 2026-10-01): every task gives the exact code, so every task runs on `sonnet` (`model: 'sonnet'`), Task 0 included, and so does each per-task review but Task 3's. Task 3 restates two engine closed forms in GUI code (`pullout_20C_Nm` times `model::ring_pair_factor` for torque against temperature; Σ amp_n sin(n x) times the area-lever product, f_end and f_cal for torque against rotation); its tests pin them to the engine (the pull-out at the operating temperature and at the E7 angle to 1e-12, the band edges to the hot-low and cold-high torques). It is not a `risk: physics` item (no engine change), and its implementation (transcription) stays on `sonnet`; but it matches both lists of section 5, where the session model wins, so **Task 3's per-task review runs on the session model** (omit `model`, comment `// session model: restated engine closed forms`). The whole-branch review after Task 6 runs on the session model (omit `model`, comment `// session model: final whole-branch review`). A `sonnet` attempt that ends `blocked`, leaves a check red or is rejected in review escalates every retry to the session model.

## Review Focus

Six input classes the spec implies, each pinned by a test in the task that owns the code:

1. **A design file or share link holding a count or a dimension far outside its slider** (`InputSet::set` checks type, finiteness and choices, not ranges: a pole count of 0, -4, 202 or `i64::MAX`; a claim of 1e300, 1e39 or -1e39 mm). Expected: no freeze and no panic; the end view draws no blocks outside 2 to 200 poles and says so in a note; a claim line past `CLAIM_REACH` times the pieces' extent is left off with one note per axis and the pieces keep their scale. Tests: `odd_pole_counts_from_a_file_never_draw_unbounded_blocks`, `a_claim_far_past_the_pieces_is_left_off_the_drawing_and_says_so` (Task 1), `a_claim_far_past_the_pieces_keeps_the_drawing_to_scale` (Task 2).
2. **Values that are not numbers** (NaN written into the struct; a harmonic set outside its choices; a conductance of NaN). Expected: the geometry view drops the pieces and lines that hold them, counts them in a note per view and names the view it cannot draw; a plot leaves the points and reference lines out or shows "Nothing to plot: the values are not numbers"; never a panic. Tests: `a_number_that_is_not_finite_draws_nothing_wrong_and_says_so` (Task 1), `a_dimension_that_is_not_a_number_draws_without_panicking` (Task 2), `plots_of_values_that_are_not_numbers_draw_without_panicking` (Task 3).
3. **A narrow, tiny or empty window** (the M4-1 review's ~930 px window leaves the centre region about 294 points; a 120 x 90 region; a 0 x 0 or 1 x 1 screen). Expected: every tab draws without panicking (an `f32::clamp` whose bounds cross would), the geometry still fits at one scale, a region with no room paints no view (never a mirrored one), the results table shows each row's label and value without scrolling. Tests: `a_dimension_that_is_not_a_number_draws_without_panicking` (its tiny screen), `a_region_with_no_room_draws_no_view` (Task 2), `the_clamp_tab_paints_both_views_to_scale_and_the_table` (294 and 120 points, and no room, Task 4), `a_narrow_window_shows_each_row_s_label_and_value_without_scrolling` (Task 5).
4. **A design past its limits** (the default design sized to 50.8 mm: 34.90 mm over the length, 36.90 mm over the bay; with a 40 mm diameter claim too, every axis at once; a claim exactly at the rotating OD and the next number below it; f_end <= 0 with short magnets; a boss too small for any screw). Expected: the overshoot named per axis in red with the exceeded dashed lines red, equality inside; the torque plots and the out-of-range sweep rows grey; drawing.py's message instead of a drawing. Tests: `the_space_claim_callouts_name_each_overshoot_in_red` (Task 1), `a_design_past_the_space_claim_draws_the_exceeded_axes_red` (Task 2), `out_of_range_end_effect_greys_the_torque_plots` and `sweep_rows_are_sorted_by_status_and_end_effect_range` (Task 3), `no_fitting_screw_gives_drawing_py_s_message` and `the_clamp_tab_draws_the_recommended_clamp_and_follows_the_design` (Task 4).
5. **Live edits and Torque -> Magnets** (an arrow key on a slider; the sized design, whose free variable is not in the inputs). Expected: the geometry view and the plots show the design shown, in the same frame (the torque-temperature plot paints the design's marker at the edited pull-out, where that frame's plot transform puts it). Tests: `the_geometry_view_is_the_default_and_follows_the_design_shown` (Task 2), `each_plot_tab_draws_its_plot_from_this_frame_s_results` (Task 3).
6. **A clamp result outside the table** (an index past the five sizes or below 1, a screw count of `i64::MAX` or below zero, from a struct written by hand). Expected: drawing.py's message, or at most 50 screws drawn (none below zero). Tests: `no_fitting_screw_gives_drawing_py_s_message`, `a_screw_count_past_the_table_draws_a_bounded_number` (Task 4).

## File Structure

Created (under `C:/Users/Cole/source/repos/lsim-mag-m42/`):

| File | Responsibility | Task |
|---|---|---|
| `magcoupling-rs/src/gui/geometry.rs` | The geometry view's drawing in millimetres, pure: `geometry` -> `Geometry` (end and side `View`s of `Piece`s, `Dashed` lines and `Callout`s, and `Note`s); the callout levels; the claim lines kept within `CLAIM_REACH`; the notes for parts not drawn; the design checks and the autofit hint; `finite` | 1 |
| `magcoupling-rs/src/gui/geometry_view.rs` | `geometry_ui`: both views at one scale with egui's painter (`Transform`, `side_by_side`), the dimension lines (`arrowhead`) and tags, the callout list, the hover texts | 2 |
| `magcoupling-rs/src/gui/plots.rs` | `PlotKind` (with its ids); the series builders (`torque_temperature`, `sweep_points`, `slip_heating`, `torque_angle`) and `plot_ui` (egui_plot) | 3 |
| `magcoupling-rs/src/gui/clamp_drawing.rs` | `clamp_drawing` (drawing.py as marks in millimetres; `fmt_g`, `clip_to_disc`), `drawing_ui` (through `side_by_side` and `arrowhead`), `clamp_ui` with the clamp table | 4 |

Modified: `magcoupling-rs/src/gui/mod.rs` (each new module), `magcoupling-rs/src/gui/panel.rs` (Tasks 2 to 5: `CentreView`'s tabs, the banner over the centre region, one copy of the design shown per frame, tests), `magcoupling-rs/src/gui/dashboard.rs` (Task 2: `hover_text`), `magcoupling-rs/src/gui/results_table.rs` (Task 2: `row_tooltip` through `hover_text`, the banner removed; Task 5: `column_widths`), `magcoupling-rs/src/gui/test_support.rs` (Task 2: `flat_shapes`, which `drawn_texts`, `text_rects`, `text_rect` and `text_color` and the tests' shape filters walk), `magcoupling-rs/Cargo.toml` and `Cargo.lock` and `linkage-sim-rs/scripts/gate.sh` (Task 3: egui_plot and gate 11), `.claude/workflows/gui-smoke.js`, `magcoupling-rs/README.md`, `docs/ai/{02-system,03-structure,04-memory}.yaml`, `docs/ai/05-update-tracker.md` (Task 6). No engine file, test data file or Python file changes.

Order: the pure geometry first, then its painter with the centre region's tab row (the default view changes there, so the results-table tests open their tab), then the plots, the clamp, the results table's widths, and the docs.

## Verification record

This plan was replayed before it was handed over, and again after the critic's review (see the Self-review record at the end). Its blocks were developed task by task in a scratch tree (a clone of an LF export of `93120f5`), then parsed out of this file by a script and applied literally, task by task, to a fresh LF export of `93120f5` (`git -c core.autocrlf=false archive 93120f5`), with a fresh cargo target directory under a short path. Every expected output below comes from the second replay.

- **Red steps** failed as stated: each Step 2 stops the build with errors naming the items its Step 3 adds (Tasks 1 to 5). **Green steps** passed.
- **After every task** the magcoupling checks of its Step "Format and lint" passed (`cargo fmt --check`; `cargo clippy --all-targets -- -D warnings` with no feature and with `app`; the wasm32 clippy of the `gui` library and of `magcoupling-web`; the wasm32 check of the engine), all in a target directory that started empty, and **the replayed tree equals, file for file, the tree in which that task was developed** (`Cargo.lock` included: online resolution added exactly `egui_plot 0.33.0`; Task 3 Step 4's `cargo update --precise 0.33.0` then changed nothing).
- **After Task 6**, gates 4 to 12 of `gate.sh` passed on the replayed tree with the oracle Python, `GATE PASS` and no `SKIP gate` line (gate 4: engine unit tests `172 passed`, every integration binary at Task 0's counts; gate 7: `349 passed`; gate 10: the parity guard on the native and wasm32 shipped builds and its negative control; gate 11: `egui 0.32.3`, `egui_plot 0.33.0`, `eframe 0.32.3`, `wasm-bindgen 0.2.114` in both lock files and the CLI pin; gate 12: `1159 passed`, the differential data current). The linkage crate's gates 1 to 3 were not run in the replay: this plan does not touch linkage-sim-rs's code (Task 3 adds one package name to gate 11's loop in `gate.sh`); Task 0, Task 3 and Task 6 run the whole gate.
- **The web bundle** was built from the replayed tree, copied to a short path (`build_magcoupling_web.sh`: the parity guard passed, `magcoupling-web_bg.wasm` 4.4 MB) and served with `python -m http.server`; Playwright opened `/magcoupling/?m=<the gui-smoke payload>` at 1280 x 800: the console showed `magcoupling: loaded the design from the share link (sizing: Torque -> Magnets)` and `magcoupling sizing: Solved at 14.18 mm (hot-low torque 2.500 N·m)`, zero errors and zero warnings; the screenshot shows the geometry view as the default tab with the sized design (both rings of ten blocks, the corner gap's tag 2 at face 1, the half side view with the dashed claim and the bay line red, "4 Overall length: axial stack 33.28 mm of 35.00 mm", "5 Large-diameter bay: stack 20.28 mm is 0.2813 mm over the 20.00 mm claim" in red, the two design checks in amber). The Pole sweep tab showed the line's legend "Pull-out torque (rows at their smallest fitting apothem)" with the design's marker just off it; the Clamp tab showed drawing.py's message (the smoke design's boss fits no screw), the summary and the machining steps; the console stayed clean.

| After task | `cargo test --features app --lib` | Engine unit tests (`cargo test`) |
|---|---|---|
| 0 (base) | 309 | 172 |
| 1 | 320 | 172 |
| 2 | 330 | 172 |
| 3 | 339 | 172 |
| 4 | 347 | 172 |
| 5 | 349 | 172 |
| 6 | 349 | 172 |

Every integration test binary keeps Task 0's counts throughout (assumptions 8, deviations 54, differential 19, explain 12 passed and 2 ignored, grades 11, material_library 4, material_links 11, parity 4, python_schema 7, robustness 12, schema 7, sizing 31, static_data 8; doc-tests 1 passed, 1 ignored).

Not exercised by the replay: the linkage crate's gates (untouched code), the `gui-smoke` workflow as a workflow (its Playwright steps were done by hand above, the design-file click excepted), and the user's answers to the Decisions to confirm.

---

## Tasks

### Task 0: Worktree, baseline and context

**Model:** `sonnet` (creating the worktree, running commands and reporting output; CLAUDE.md section 5). Step 8 is the controller's.

Verification only; nothing to write.

**Files:**
- none

**Interfaces:**
- Consumes: `main` at `93120f5` or a later commit that leaves the files this plan's blocks touch unchanged (Step 2).
- Produces: the worktree `C:/Users/Cole/source/repos/lsim-mag-m42` on the new branch `magcoupling/m4-2` with LF line endings, a green baseline with its test counts, and the controller's record of the user's answers to the Decisions to confirm.

- [ ] **Step 1: Create the worktree**

This machine's system gitconfig sets `core.autocrlf=true`, which would check the files out with CRLF; every block below quotes them with LF, as git stores them. So the worktree is created with LF and keeps a worktree-scoped `core.autocrlf=false`.

Run:

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation log --oneline -1 main
git -C C:/Users/Cole/source/repos/linkage_simulation config extensions.worktreeConfig true
git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b magcoupling/m4-2 C:/Users/Cole/source/repos/lsim-mag-m42 main
git -C C:/Users/Cole/source/repos/lsim-mag-m42 config --worktree core.autocrlf false
git -C C:/Users/Cole/source/repos/lsim-mag-m42 rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-m42 status --short
```

Expected: the `log` line is `93120f5 docs(plans): magcoupling Addendum A-4 corrections plan (E21, E22, E23, E24; 8 tasks) — on hold, decisions A4-1..A4-4 to confirm` or a later commit; `worktree add` prints `Preparing worktree (new branch 'magcoupling/m4-2')`; then `magcoupling/m4-2`; no status lines.

- [ ] **Step 2: Check that the blocks' files are still those of `93120f5`**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m42 diff --stat 93120f5 HEAD -- magcoupling-rs/Cargo.toml magcoupling-rs/Cargo.lock magcoupling-rs/src/gui magcoupling-rs/README.md docs/ai linkage-sim-rs/scripts/gate.sh .claude/workflows/gui-smoke.js magcoupling-rs/src/engine magcoupling-rs/src/lib.rs magcoupling-rs/tests/data
```

Expected: no output. If any file is listed, stop and escalate: the blocks of the tasks that touch it, or the engine numbers the tests pin (such as "Corner gap 1.027 mm", "Running clearance -0.1032 mm", the sweep counts (4, 2, 7) and "Solved at 14.18 mm"), must be re-derived first. (Files outside this list, such as the linkage crate's or another plan's, may differ.)

- [ ] **Step 3: Check the line endings**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m42 config --get core.autocrlf; head -c 2000 C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs | tr -cd '\r' | wc -c
```

Expected: `false` and `0` carriage returns. If the count is not 0: run `git -C C:/Users/Cole/source/repos/lsim-mag-m42 rm -rq --cached . && git -C C:/Users/Cole/source/repos/lsim-mag-m42 reset -q --hard` and check again.

- [ ] **Step 4: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/01-meta.yaml`, `02-system.yaml`, `03-structure.yaml`, `04-memory.yaml` (the M4-2 items this plan resolves: the "Open (M4-2, M4-3)" item, the design checks of decision 28); `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/README.md` ("The panel", "Versions", the tests table); the spec's M4 and Addendum A1 sections; `C:/Users/Cole/source/repos/lsim-mag-m42/reference/magcoupling-py/magcoupling/drawing.py`; the panel `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, `dashboard.rs` and `results_table.rs`; and this plan's Decisions to confirm.

- [ ] **Step 5: Run the crate's tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: the engine run: unit tests `172 passed`; `tests\assumptions.rs` 8; `tests\deviations.rs` 54; `tests\differential.rs` 19; `tests\explain.rs` 12 passed, 2 ignored; `tests\grades.rs` 11; `tests\material_library.rs` 4; `tests\material_links.rs` 11; `tests\parity.rs` 4; `tests\python_schema.rs` 7; `tests\robustness.rs` 12; `tests\schema.rs` 7; `tests\sizing.rs` 31; `tests\static_data.rs` 8; doc-tests `1 passed; 0 failed; 1 ignored`. Then with `app`: `test result: ok. 309 passed`. If a count differs but every binary is `ok`, record the actual counts and read every later task's counts as offsets from them.

- [ ] **Step 6: Check the oracle Python and the CLI**

Run:

```bash
ls C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe
wasm-bindgen --version
```

Expected: the path, then `wasm-bindgen 0.2.114`. If the Python is missing, stop and escalate: every gate run uses it.

- [ ] **Step 7: Run the gate**

Run:

```bash
mkdir -p C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m42/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target/gate-task0.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target/gate-task0.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target/gate-task0.log
git -C C:/Users/Cole/source/repos/lsim-mag-m42 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0`; `cargo fmt --check` prints nothing.

- [ ] **Step 8: Decisions**

The controller asks the user the plan's **Decisions to confirm** (M42-1 to M42-9) before Task 1 and records the answers in the execution notes. Every task implements the recommended option and names the decision where it implements it; if the user picks another option, the task that implements it stops and escalates instead of improvising.

---

### Task 1: The geometry model

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

The pure core of the geometry view (spec M4 "Layout" centre; Addendum A1's space claim). `geometry(&inputs, &results)` turns the design shown into two views in millimetres and the notes under them. Every number is a result of this frame: the end view's radii are the Calculator's apothems and radii (`model.*`) and the retainers' diameters (`retainers.*`), the side view's lengths the housing in effect (`housing.hub_length_mm`, `cup_depth_mm`, `retainer_span_mm`, A-2 decision A2-8), so a length-sized design grows. The three end-view callouts (face gap; corner gap, along face 1's normal from the dashed circle the inner corners sweep to the outer flat; running clearance) and the three space-claim callouts (one per axis: the axial stack against the overall length, the large-diameter stack against the bay, the rotating OD against the diameter) carry the result path their hover text will show (Task 2) and their level (decision M42-4). The notes are the autofit wall (decision 27's `materials.cup_wall_suggested_mm`) and the two design checks of decision 28, each while it holds (M42-9). Each view's extent comes from its pieces: a claim line farther than `CLAIM_REACH` times the pieces' extent along its axis is left off with a note (the diameter decided once, for both views); pieces and lines that hold a number that is not finite are dropped and counted in a note per view; blocks are drawn for 2 to 200 poles only (Review Focus 1 and 2).

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/geometry.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/mod.rs` (`pub mod geometry;`)

**Interfaces:**
- Consumes: `gui::dashboard::{Level, verdict_level}` (the clearance check's verdict), `gui::format::{format_value, with_unit}`, `engine::meta::{NumOrText, Value}`, `DesignInputs`, `DesignResults` (`model`, `retainers`, `metal`, `materials`, `housing`).
- Produces (`magcoupling::gui::geometry`): `pub type Mm = [f64; 2]`; `pub const MAX_DRAWN_POLES: i64 = 200`; `pub const CLAIM_REACH: f64 = 10.0`; `pub(crate) fn finite(Mm) -> bool`; `enum Part { Body, Cavity, Magnet { north: bool }, Retainer, Cap, Shaft, Key }`; `enum Outline { Disc { r }, Ring { r_in, r_out }, Polygon(Vec<Mm>), Sector { r_in, r_out, from, to }, Rect { min: Mm, max: Mm } }` with `fn is_finite(&self) -> bool`; `struct Piece { pub part: Part, pub outline: Outline }`; `struct Dashed { pub points: Vec<Mm>, pub level: Option<Level> }`; `struct Callout { pub tag: usize, pub path: &'static str, pub text: String, pub level: Option<Level>, pub from: Mm, pub to: Mm }` (tags 1 to 3 the end view, 4 to 6 the side view); `struct View { pub pieces: Vec<Piece>, pub dashed: Vec<Dashed>, pub callouts: Vec<Callout>, pub min: Mm, pub max: Mm }` with `fn is_drawable(&self) -> bool`; `struct Note { pub text: String, pub level: Option<Level> }`; `struct Geometry { pub end: View, pub side: View, pub notes: Vec<Note> }` with `fn callouts(&self) -> impl Iterator<Item = &Callout>`; `const BLOCKS_NOT_DRAWN`, `NOT_DRAWN`, `DESIGN_CHECK`, `AUTOFIT: &str`; `fn mm(f64) -> String` (four significant digits and " mm"); `fn geometry(&DesignInputs, &DesignResults) -> Geometry`.

- [ ] **Step 1: Write the failing tests**

The module starts as its docs and its tests; Step 3 adds the code between them. The tests pin every part of the end view to the result it comes from (the pocket's corners at the pocket corner radius, the rings at the retainers' diameters, the key from where its sides cross the bore), faceted blocks on their flats and arcs as sectors, each callout's line length against the gap it names and its text and level (the default running clearance, -0.1032 mm, is red; the corner gap's line along face 1's normal, from the dashed circle the inner corners sweep to the outer flat), the side view's spans in effect under a 50.8 mm length override (the hub and the retainers), the overshoot texts per axis (every axis at once; exactly at the claim and the next number below it), a claim far past the pieces (1e300, 1e39, -1e39; at the reach and just past it), the design checks at the defaults, when fixed and at equality, the autofit hint, pole counts from a file, and NaN (the parts dropped counted per view).

Create `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/geometry.rs`:

````rust
//! The geometry view's drawing, in millimetres (spec M4 "Layout": "Center — geometry view, to
//! scale: end view of both rings with blocks on polygon faces, cup, sleeve, liner, shaft and key;
//! side view of the axial stack against the 20 mm bay and 35 mm overall length. ... Dimension
//! callouts for face gap, corner gap, running clearance; violations (negative clearance, envelope
//! exceeded) draw red"; Addendum A1: the space claim "drawn as a dashed outline", exceeding it
//! "shows a red callout on the view naming the overshoot in mm per axis").
//!
//! Pure: [`geometry`] turns the design shown (its inputs and results) into two [`View`]s, each a
//! list of filled pieces, dashed lines and dimension [`Callout`]s in millimetres, and the
//! [`Note`]s under them; `gui::geometry_view` paints them to scale. Every dimension is read
//! from the engine's results, the axial ones from the housing in effect (`housing.*`, decision
//! A2-8), so the drawing follows every edit and the length override.
//!
//! - **End view** (x right, y up, the shaft axis at the origin, seen from the free end): the cup
//!   with its pocket, the outer blocks, the liner, the sleeve, the inner blocks, the hub, the
//!   shaft and the key, and the diameter claim as a dashed circle. Faceted blocks sit on the
//!   polygon flats; arcs (any other `coupling.faceted` code, as the engine reads it) are sectors
//!   of the width at their mid-radius. Face `k`'s centre is at 90° + 360° k / N: face 0 at the
//!   top carries the face gap; face 1 the corner gap, along its normal from the circle its inner
//!   block's corners sweep (dashed) to its outer block's flat; the running clearance is at the
//!   bottom.
//! - **Side view** (decision M42-3: the upper half section through the flat centres, x along
//!   the axis from the cap's front face, y the radius): cap, cup wall, web and boss, and inside
//!   the cavity the outer and inner blocks, the liner, the sleeve and the hub, each centred on
//!   the cavity (the workbook gives no axial positions); the space claim's overall length, bay
//!   and diameter as dashed lines; the axial stack, the large-diameter stack and the rotating OD
//!   as dimensions.
//!
//! A callout is red ([`Level::Bad`]) when it is violated (decision M42-4): a gap below zero, the
//! running clearance below its target (the dashboard's clearance check), a space-claim axis
//! exceeded; amber ([`Level::Caution`]) when its value is not a number.
//!
//! A design file can hold any finite number (`InputSet::set` checks no range), so each view's
//! extent comes from its pieces: a claim line farther than [`CLAIM_REACH`] times the pieces'
//! extent along its axis is left off with a note, and a piece or line that holds a number that
//! is not finite is dropped and counted in a note.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;

    fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        edit(&mut inputs);
        inputs
    }

    fn of(inputs: &DesignInputs) -> Geometry {
        geometry(inputs, &compute_all(inputs))
    }

    fn callout(g: &Geometry, tag: usize) -> &Callout {
        g.callouts().find(|c| c.tag == tag).unwrap()
    }

    fn length(c: &Callout) -> f64 {
        ((c.to[0] - c.from[0]).powi(2) + (c.to[1] - c.from[1]).powi(2)).sqrt()
    }

    #[test]
    fn the_end_view_draws_every_part_from_the_results() {
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let g = geometry(&inputs, &r);
        let n = inputs.coupling.npole as usize;
        let magnets = g
            .end
            .pieces
            .iter()
            .filter(|p| matches!(p.part, Part::Magnet { .. }))
            .count();
        assert_eq!(magnets, 2 * n, "both rings");
        assert_eq!(
            g.end.pieces[0],
            Piece {
                part: Part::Body,
                outline: Outline::Disc {
                    r: r.model.cup_od_mm / 2.0
                }
            }
        );
        // The pocket is the N-gon whose corners sit at the pocket corner radius (C61).
        let Outline::Polygon(pocket) = &g.end.pieces[1].outline else {
            panic!("faceted pocket")
        };
        assert_eq!(pocket.len(), n);
        for p in pocket {
            let r_corner = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((r_corner - r.model.pocket_corner_radius_mm).abs() < 1e-9);
        }
        let rings: Vec<&Outline> = g
            .end
            .pieces
            .iter()
            .filter(|p| p.part == Part::Retainer)
            .map(|p| &p.outline)
            .collect();
        assert_eq!(
            rings,
            [
                &Outline::Ring {
                    r_in: r.retainers.liner_id_mm / 2.0,
                    r_out: r.retainers.liner_od_mm / 2.0
                },
                &Outline::Ring {
                    r_in: r.retainers.sleeve_id_mm / 2.0,
                    r_out: r.retainers.sleeve_od_mm / 2.0
                },
            ]
        );
        assert!(
            g.end
                .pieces
                .iter()
                .any(|p| p.part == Part::Shaft && p.outline == Outline::Disc { r: 5.0 })
        );
        let key = g.end.pieces.iter().find(|p| p.part == Part::Key).unwrap();
        let Outline::Polygon(key) = &key.outline else {
            panic!("the key is a rectangle")
        };
        // From where its 4 mm sides cross the 10 mm bore to 1.7 mm past the bore.
        assert_eq!(key[1], [6.7, -2.0]);
        assert!((key[0][0] - 21.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn faceted_blocks_sit_on_the_flats_and_arcs_are_sectors() {
        let r = compute_all(&DesignInputs::default());
        let g = of(&DesignInputs::default());
        // The first outer block is face 0 at the top: its face at the outer face apothem, its
        // back at the outer back apothem, 6.35 mm wide.
        let first = g
            .end
            .pieces
            .iter()
            .find(|p| matches!(p.part, Part::Magnet { north: false }))
            .unwrap();
        let Outline::Polygon(q) = &first.outline else {
            panic!("a flat block")
        };
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(q[0][1], r.model.outer_face_apothem_mm));
        assert!(close(q[2][1], r.model.outer_back_apothem_mm));
        assert!(close((q[1][0] - q[0][0]).abs(), r.model.outer_width_mm));
        let arcs = of(&design(|i| i.coupling.faceted = 0));
        let sector = arcs
            .end
            .pieces
            .iter()
            .find(|p| matches!(p.part, Part::Magnet { .. }))
            .unwrap();
        assert!(matches!(sector.outline, Outline::Sector { .. }));
        assert!(matches!(arcs.end.pieces[1].outline, Outline::Disc { .. }));
    }

    #[test]
    fn the_end_view_callouts_measure_the_gaps_they_name() {
        let r = compute_all(&DesignInputs::default());
        let g = of(&DesignInputs::default());
        let tags: Vec<usize> = g.callouts().map(|c| c.tag).collect();
        assert_eq!(tags, [1, 2, 3, 4, 5, 6]);
        let face = callout(&g, 1);
        assert_eq!(face.text, "Face gap 1.400 mm");
        assert_eq!(face.path, "model.face_gap_mm");
        assert!((length(face) - r.model.face_gap_mm).abs() < 1e-9);
        assert_eq!(face.level, None);
        let corner = callout(&g, 2);
        assert_eq!(corner.text, "Corner gap 1.027 mm");
        assert!((length(corner) - r.model.corner_gap_mm).abs() < 1e-9);
        // Along face 1's normal: from the circle the inner corners sweep, drawn dashed, to the
        // outer block's flat (both ends on drawn features).
        let normal = face_angle(1, DesignInputs::default().coupling.npole);
        let dot = |p: Mm, a: f64| p[0] * a.cos() + p[1] * a.sin();
        assert!((dot(corner.to, normal) - r.model.outer_face_apothem_mm).abs() < 1e-9);
        assert!(dot(corner.to, normal + PI / 2.0).abs() <= r.model.outer_width_mm / 2.0);
        let on_corner_circle =
            |p: &Mm| (p[0].hypot(p[1]) - r.model.inner_corner_radius_mm).abs() < 1e-9;
        assert!(on_corner_circle(&corner.from));
        assert!(
            g.end
                .dashed
                .iter()
                .any(|d| d.level.is_none() && d.points.iter().all(on_corner_circle))
        );
        // The default design's running clearance is below zero: red (decision M42-4).
        let run = callout(&g, 3);
        assert_eq!(
            run.text,
            "Running clearance -0.1032 mm (sleeve-to-liner gap 0.6768 mm less movement 0.7800 mm)"
        );
        assert_eq!(run.level, Some(Level::Bad));
        assert!((length(run) - r.metal.sleeve_liner_clearance_mm).abs() < 1e-9);
    }

    #[test]
    fn a_clearance_below_its_target_is_red_and_a_negative_corner_gap_too() {
        // Above zero but below the 0.2 mm target: the dashboard's check fails, so red.
        let small = design(|i| i.metal.magnet_position_mm = 0.0);
        let r = compute_all(&small);
        assert!(r.metal.min_running_clearance_mm > 0.0 && r.metal.min_running_clearance_mm < 0.2);
        assert_eq!(callout(&of(&small), 3).level, Some(Level::Bad));
        // Past the target: no colour.
        let clear = of(&design(|i| {
            i.metal.magnet_position_mm = 0.0;
            i.metal.shaft_displacement_mm = 0.0;
        }));
        assert_eq!(callout(&clear, 3).level, None);
        // Face gap 0.3 mm puts the inner corners past the outer faces.
        let tight = of(&design(|i| i.metal.face_gap_mm = 0.3));
        assert_eq!(callout(&tight, 2).text, "Corner gap -0.07317 mm");
        assert_eq!(callout(&tight, 2).level, Some(Level::Bad));
        assert_eq!(callout(&tight, 1).level, None);
    }

    #[test]
    fn the_side_view_uses_the_housing_in_effect() {
        let inputs = design(|i| i.coupling.magnets.axial_length_mm = Some(50.8));
        let r = compute_all(&inputs);
        let g = geometry(&inputs, &r);
        let rects: Vec<(Part, Mm, Mm)> = g
            .side
            .pieces
            .iter()
            .map(|p| match p.outline {
                Outline::Rect { min, max } => (p.part, min, max),
                ref other => panic!("{other:?}"),
            })
            .collect();
        let cap = inputs.metal.cap_axial_mm;
        // The cup wall spans the cavity in effect (53.6 mm, not the 15.5 mm input).
        assert_eq!(rects[0].1[0], cap);
        assert!((rects[0].2[0] - (cap + r.housing.cup_depth_mm)).abs() < 1e-9);
        assert!((r.housing.cup_depth_mm - 53.6).abs() < 1e-9);
        // The hub is the hub length in effect long, centred on the cavity.
        let hub = rects.last().unwrap();
        assert_eq!(hub.0, Part::Body);
        assert!((hub.2[0] - hub.1[0] - r.housing.hub_length_mm).abs() < 1e-9);
        let mid = cap + r.housing.cup_depth_mm / 2.0;
        assert!(((hub.1[0] + hub.2[0]) / 2.0 - mid).abs() < 1e-9);
        // The liner and the sleeve span the retainer span in effect, not the input's.
        assert_ne!(r.housing.retainer_span_mm, inputs.metal.retainer_span_mm);
        let retainers: Vec<&(Part, Mm, Mm)> = rects
            .iter()
            .filter(|(p, ..)| *p == Part::Retainer)
            .collect();
        assert_eq!(retainers.len(), 2);
        for (_, min, max) in retainers {
            assert!((max[0] - min[0] - r.housing.retainer_span_mm).abs() < 1e-9);
        }
    }

    #[test]
    fn the_space_claim_callouts_name_each_overshoot_in_red() {
        let inside = of(&DesignInputs::default());
        assert_eq!(
            callout(&inside, 4).text,
            "Overall length: axial stack 31.80 mm of 35.00 mm"
        );
        assert_eq!(callout(&inside, 4).path, "metal.axial_stack_mm");
        assert_eq!(callout(&inside, 4).level, None);
        assert_eq!(
            callout(&inside, 6).text,
            "Diameter: rotating OD 42.80 mm of 43.00 mm"
        );
        assert!(inside.side.dashed.iter().all(|d| d.level.is_none()));
        let long = of(&design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)));
        let length = callout(&long, 4);
        assert_eq!(
            length.text,
            "Overall length: axial stack 69.90 mm is 34.90 mm over the 35.00 mm claim"
        );
        assert_eq!(length.path, "housing.length_overshoot_mm");
        assert_eq!(length.level, Some(Level::Bad));
        assert!(
            callout(&long, 5)
                .text
                .contains("36.90 mm over the 20.00 mm claim")
        );
        assert_eq!(callout(&long, 6).level, None);
        // The exceeded axes' dashed lines are red: the bay line and the right edge.
        let red = long
            .side
            .dashed
            .iter()
            .filter(|d| d.level == Some(Level::Bad))
            .count();
        assert_eq!(red, 2);
        // A diameter past the claim reddens the end view's claim circle.
        let wide = of(&design(|i| i.metal.max_diameter_mm = 40.0));
        assert_eq!(callout(&wide, 6).level, Some(Level::Bad));
        assert_eq!(wide.end.dashed[0].level, Some(Level::Bad));
        // Every axis at once: three red callouts, three red side lines and the red circle.
        let all = of(&design(|i| {
            i.coupling.magnets.axial_length_mm = Some(50.8);
            i.metal.max_diameter_mm = 40.0;
        }));
        for tag in [4, 5, 6] {
            assert_eq!(callout(&all, tag).level, Some(Level::Bad), "{tag}");
        }
        let red = all
            .side
            .dashed
            .iter()
            .filter(|d| d.level == Some(Level::Bad))
            .count();
        assert_eq!(red, 3);
        assert_eq!(all.end.dashed[0].level, Some(Level::Bad));
        // Exactly at the claim is inside; the next number below it is over (per axis, as the
        // dashboard's space-claim check reads it).
        let od = compute_all(&DesignInputs::default()).metal.rotating_od_mm;
        let at = of(&design(|i| i.metal.max_diameter_mm = od));
        assert_eq!(callout(&at, 6).level, None);
        assert_eq!(at.end.dashed[0].level, None);
        let below = of(&design(|i| i.metal.max_diameter_mm = od.next_down()));
        assert_eq!(callout(&below, 6).level, Some(Level::Bad));
        assert_eq!(
            of(&design(|i| i.metal.max_diameter_mm = f64::NAN))
                .callouts()
                .nth(5)
                .unwrap()
                .level,
            Some(Level::Caution)
        );
    }

    #[test]
    fn the_design_checks_show_while_each_inconsistency_holds() {
        let checks = |inputs: &DesignInputs| -> Vec<String> {
            of(inputs)
                .notes
                .into_iter()
                .filter(|n| n.text.starts_with(DESIGN_CHECK))
                .map(|n| n.text)
                .collect()
        };
        // The workbook's defaults hold both (decision 28).
        assert_eq!(
            checks(&DesignInputs::default()),
            [
                "Design check: the cap thread diameter 41.00 mm (Metal design C169) is below the cup body OD 41.33 mm (C165)",
                "Design check: the boss OD is 22.00 mm in Metal design (C127) and 25.00 mm in Shaft clamps (C35)",
            ]
        );
        let fixed = design(|i| {
            i.metal.cap_thread_dia_mm = 42.0;
            i.metal.boss_od_mm = 25.0;
        });
        assert!(checks(&fixed).is_empty());
        // At equality the thread is not below the cup body.
        let cup = compute_all(&DesignInputs::default()).metal.cup_body_od_mm;
        let equal = design(|i| i.metal.cap_thread_dia_mm = cup);
        assert_eq!(checks(&equal).len(), 1);
    }

    #[test]
    fn the_autofit_hint_shows_the_wall_rule_unless_there_is_no_back_iron() {
        let hint = |inputs: &DesignInputs| {
            of(inputs)
                .notes
                .into_iter()
                .find(|n| n.text.starts_with(AUTOFIT))
        };
        let note = hint(&DesignInputs::default()).expect("a steel circuit has a wall rule");
        assert_eq!(
            note.text,
            "Autofit: the wall rule asks for at least 2.000 mm at the pocket corners (input 1.800 mm)"
        );
        assert_eq!(note.level, None);
        assert_eq!(hint(&design(|i| i.coupling.backiron = 0)), None);
    }

    #[test]
    fn odd_pole_counts_from_a_file_never_draw_unbounded_blocks() {
        for npole in [0, 1, -4, MAX_DRAWN_POLES + 2, i64::MAX] {
            let g = of(&design(|i| i.coupling.npole = npole));
            assert!(
                g.end
                    .pieces
                    .iter()
                    .all(|p| !matches!(p.part, Part::Magnet { .. })),
                "{npole}"
            );
            assert!(
                g.notes.iter().any(|n| n.text.starts_with(BLOCKS_NOT_DRAWN)),
                "{npole}"
            );
        }
        let most = of(&design(|i| i.coupling.npole = MAX_DRAWN_POLES));
        let magnets = most
            .end
            .pieces
            .iter()
            .filter(|p| matches!(p.part, Part::Magnet { .. }))
            .count();
        assert_eq!(magnets, 2 * MAX_DRAWN_POLES as usize);
    }

    #[test]
    fn a_number_that_is_not_finite_draws_nothing_wrong_and_says_so() {
        // validate() refuses these at every boundary; a struct literal can still hold them.
        let g = of(&design(|i| i.coupling.bore_mm = f64::NAN));
        assert!(g.end.pieces.iter().all(|p| p.outline.is_finite()));
        assert!(g.side.pieces.iter().all(|p| p.outline.is_finite()));
        // The dropped parts are counted: the shaft and the key; the web, the boss and the hub.
        let texts: Vec<&str> = g.notes.iter().map(|n| n.text.as_str()).collect();
        assert!(
            texts
                .contains(&"Not drawn: 2 parts of the end view hold a number that is not a number"),
            "{texts:?}"
        );
        assert!(
            texts.contains(
                &"Not drawn: 3 parts of the side view hold a number that is not a number"
            ),
            "{texts:?}"
        );
        // A claim that is not a number: its circle and its line, one part per view.
        let nan_claim = of(&design(|i| i.metal.max_diameter_mm = f64::NAN));
        let notes: Vec<&Note> = nan_claim
            .notes
            .iter()
            .filter(|n| n.text.starts_with(NOT_DRAWN))
            .collect();
        assert_eq!(
            notes,
            [
                &Note {
                    text: "Not drawn: 1 part of the end view holds a number that is not a number"
                        .to_owned(),
                    level: Some(Level::Caution)
                },
                &Note {
                    text: "Not drawn: 1 part of the side view holds a number that is not a number"
                        .to_owned(),
                    level: Some(Level::Caution)
                },
            ]
        );
        // The cup's OD and the claim both NaN: the end view has no extent.
        let nan_cup = of(&design(|i| {
            i.metal.cup_wall_corner_mm = f64::NAN;
            i.metal.max_diameter_mm = f64::NAN;
        }));
        assert!(!nan_cup.end.is_drawable());
        assert!(nan_cup.side.is_drawable());
        assert!(
            nan_cup
                .notes
                .iter()
                .any(|n| n.text == "Not drawn: a dimension of the end view is not a number")
        );
        // The default design draws everything: no such note.
        assert!(
            of(&DesignInputs::default())
                .notes
                .iter()
                .all(|n| !n.text.starts_with(NOT_DRAWN))
        );
    }

    #[test]
    fn a_claim_far_past_the_pieces_is_left_off_the_drawing_and_says_so() {
        // InputSet::set checks no range: a design file can hold a finite but huge claim.
        let default = of(&DesignInputs::default());
        let off = |g: &Geometry| -> Vec<String> {
            g.notes
                .iter()
                .filter(|n| n.text.ends_with("is off the drawing"))
                .map(|n| {
                    assert_eq!(n.level, Some(Level::Caution));
                    n.text.clone()
                })
                .collect()
        };
        assert!(off(&default).is_empty());
        let wide = of(&design(|i| i.metal.max_diameter_mm = 1e300));
        // One note for both views; neither draws the claim, and each keeps its pieces' extent.
        assert_eq!(
            off(&wide),
            ["Not drawn: the diameter claim 1.000e300 mm is off the drawing"]
        );
        let r = compute_all(&DesignInputs::default());
        assert_eq!(wide.end.max[0], r.model.cup_od_mm / 2.0 + 2.5);
        assert_eq!(wide.end.dashed.len(), default.end.dashed.len() - 1);
        assert_eq!(wide.side.dashed.len(), default.side.dashed.len() - 1);
        assert!(wide.end.is_drawable() && wide.side.is_drawable());
        assert!(wide.side.max[1] < 30.0, "{:?}", wide.side.max);
        assert_eq!(
            callout(&wide, 6).level,
            None,
            "the callout still reads the claim"
        );
        // The overall length and the bay, either sign.
        let long = of(&design(|i| i.metal.max_overall_axial_mm = 1e39));
        assert_eq!(
            off(&long),
            ["Not drawn: the overall length claim 1.000e39 mm is off the drawing"]
        );
        assert_eq!(long.side.max[0], r.metal.axial_stack_mm + 2.0);
        let bay = of(&design(|i| i.metal.max_large_dia_axial_mm = -1e39));
        assert_eq!(
            off(&bay),
            ["Not drawn: the large-diameter bay claim -1.000e39 mm is off the drawing"]
        );
        assert_eq!(bay.side.max, default.side.max);
        // At the reach the claim is drawn; just past it, it is not.
        let reach = CLAIM_REACH * body_od(&DesignInputs::default(), &r);
        assert!(off(&of(&design(|i| i.metal.max_diameter_mm = reach))).is_empty());
        assert_eq!(
            off(&of(&design(|i| i.metal.max_diameter_mm = reach.next_up()))).len(),
            1
        );
    }
}
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub mod corrections;
pub mod dashboard;
mod format;
pub mod history;
pub mod input_ui;
pub mod inputs;
````

with:

````rust
pub mod corrections;
pub mod dashboard;
mod format;
pub mod geometry;
pub mod history;
pub mod input_ui;
pub mod inputs;
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib gui::geometry 2>&1 | grep -E "^error" | sort | uniq -c
```

Expected: FAIL to compile. The build stops with `error: could not compile `magcoupling-rs` (lib test) due to 90 previous errors; 1 warning emitted`; the errors name the items Step 3 adds, among them `error[E0425]: cannot find function `geometry` in this scope`, `error[E0412]: cannot find type `Geometry` in this scope`, `error[E0412]: cannot find type `Callout` in this scope`, `error[E0425]: cannot find value `MAX_DRAWN_POLES` in this scope` and `error[E0425]: cannot find value `CLAIM_REACH` in this scope`.

- [ ] **Step 3: Write the implementation**

The module's imports and code go between its docs and its tests.

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/geometry.rs`, replace:

````rust

#[cfg(test)]
mod tests {
````

with:

````rust

use std::f64::consts::PI;

use crate::engine::meta::{NumOrText, Value};
use crate::gui::dashboard::{Level, verdict_level};
use crate::gui::format::{format_value, with_unit};
use crate::{DesignInputs, DesignResults};

/// A point [mm]: x right, y up.
pub type Mm = [f64; 2];

/// The most poles per ring the end view draws blocks for (a design file can hold any count;
/// the sliders stop at 40).
pub const MAX_DRAWN_POLES: i64 = 200;

/// How far a space-claim line may lie from the axis or the cap's front face: at most this many
/// times the pieces' extent along its axis (the body's largest OD, the axial stack). Farther, a
/// finite but huge claim from a design file would shrink the pieces to nothing; it is left off
/// the drawing with a note.
pub const CLAIM_REACH: f64 = 10.0;

/// What a piece is a section of (its fill colour in the painter).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The steel (or, without back iron, the body material's) hub, cup, web and boss.
    Body,
    /// The empty cup pocket: the drawing's background.
    Cavity,
    /// A magnet block; `north` alternates around a ring.
    Magnet { north: bool },
    /// The 316L sleeve over the inner blocks and the liner inside the outer blocks.
    Retainer,
    /// The aluminium front cap.
    Cap,
    /// The keyed shaft.
    Shaft,
    /// The key.
    Key,
}

/// A filled outline [mm].
#[derive(Clone, Debug, PartialEq)]
pub enum Outline {
    /// A disc of radius `r` centred on the axis.
    Disc { r: f64 },
    /// A ring between radii `r_in` and `r_out`, centred on the axis.
    Ring { r_in: f64, r_out: f64 },
    /// A convex polygon.
    Polygon(Vec<Mm>),
    /// An annular sector between radii `r_in` and `r_out`, from angle `from` to `to` [rad].
    Sector {
        r_in: f64,
        r_out: f64,
        from: f64,
        to: f64,
    },
    /// An axis-aligned rectangle.
    Rect { min: Mm, max: Mm },
}

impl Outline {
    /// Whether every number of the outline is finite (a design written straight into the
    /// struct can hold NaN; such a piece is not drawn).
    pub fn is_finite(&self) -> bool {
        match self {
            Outline::Disc { r } => r.is_finite(),
            Outline::Ring { r_in, r_out } => r_in.is_finite() && r_out.is_finite(),
            Outline::Polygon(points) => points.iter().all(|p| finite(*p)),
            Outline::Sector {
                r_in,
                r_out,
                from,
                to,
            } => [r_in, r_out, from, to].iter().all(|x| x.is_finite()),
            Outline::Rect { min, max } => finite(*min) && finite(*max),
        }
    }
}

/// One filled piece of a view.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub part: Part,
    pub outline: Outline,
}

/// A dashed line [mm]: the space claim and the axis.
#[derive(Clone, Debug, PartialEq)]
pub struct Dashed {
    pub points: Vec<Mm>,
    /// Red when the claim axis it draws is exceeded; `None` for the plain line.
    pub level: Option<Level>,
}

/// A dimension: a line in its view, a tag drawn beside it, and its text in the list under the
/// views. Hovering it shows the hover text of the result it shows (`path`: the M4-3 hook).
#[derive(Clone, Debug, PartialEq)]
pub struct Callout {
    /// The tag beside the dimension and before its text: 1, 2, ...
    pub tag: usize,
    /// The result the dimension shows.
    pub path: &'static str,
    pub text: String,
    /// `Some(Level::Bad)` violated, `Some(Level::Caution)` not a number, `None` fine.
    pub level: Option<Level>,
    /// The dimension line [mm].
    pub from: Mm,
    pub to: Mm,
}

/// One view: what it draws, in paint order, and its extent [mm].
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub pieces: Vec<Piece>,
    pub dashed: Vec<Dashed>,
    pub callouts: Vec<Callout>,
    /// The corners of the area the view needs [mm].
    pub min: Mm,
    pub max: Mm,
}

impl View {
    /// Whether the view can be drawn: a finite extent of positive size.
    pub fn is_drawable(&self) -> bool {
        finite(self.min)
            && finite(self.max)
            && self.max[0] > self.min[0]
            && self.max[1] > self.min[1]
    }

    /// Drops every piece and dashed line that holds a number that is not finite and returns
    /// how many it dropped (a callout stays: the list shows its text, and the painter skips a
    /// line that is not finite).
    fn retain_finite(&mut self) -> usize {
        let before = self.pieces.len() + self.dashed.len();
        self.pieces.retain(|p| p.outline.is_finite());
        self.dashed.retain(|d| d.points.iter().all(|p| finite(*p)));
        before - self.pieces.len() - self.dashed.len()
    }
}

/// A line of text under the views: the autofit hint, a design check, or why something is not
/// drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    pub text: String,
    /// `Some(Level::Caution)` for a design check or a part not drawn; `None` for a hint.
    pub level: Option<Level>,
}

/// The two views and the notes of a design.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub end: View,
    pub side: View,
    pub notes: Vec<Note>,
}

impl Geometry {
    /// Every callout of both views, in tag order.
    pub fn callouts(&self) -> impl Iterator<Item = &Callout> {
        self.end.callouts.iter().chain(self.side.callouts.iter())
    }
}

/// The start of the note shown when the blocks are not drawn.
pub const BLOCKS_NOT_DRAWN: &str = "Blocks not drawn";

/// The start of the note shown when a view holds a number that is not finite.
pub const NOT_DRAWN: &str = "Not drawn";

/// The start of each design-check note (the workbook's two inconsistencies, decision 28; the
/// user's decision of 2026-10-01: flagged in the GUI, no engine change).
pub const DESIGN_CHECK: &str = "Design check";

/// The start of the autofit hint.
pub const AUTOFIT: &str = "Autofit";

/// Whether both coordinates of `p` are finite.
pub(crate) fn finite(p: Mm) -> bool {
    p[0].is_finite() && p[1].is_finite()
}

/// A length as the callouts show it: four significant digits and the unit.
pub fn mm(x: f64) -> String {
    with_unit(format_value(&Value::Num(x)), "mm")
}

/// The unit vector at angle `a` [rad].
fn radial(a: f64) -> Mm {
    [a.cos(), a.sin()]
}

/// `r` along the unit vector at angle `a`, plus `t` along its tangent (90° further).
fn at(r: f64, a: f64, t: f64) -> Mm {
    [r * a.cos() - t * a.sin(), r * a.sin() + t * a.cos()]
}

/// The centre angle of face `k` of `n` [rad]: face 0 at the top.
fn face_angle(k: i64, n: i64) -> f64 {
    PI / 2.0 + 2.0 * PI * k as f64 / n as f64
}

/// A block on face `k` of `n`: radial from apothem `r0` to `r1`, `width` wide; faceted on the
/// flat, or (`faceted` false) an arc of `width` at its mid-radius.
fn block(k: i64, n: i64, r0: f64, r1: f64, width: f64, faceted: bool) -> Outline {
    let a = face_angle(k, n);
    if faceted {
        let w = width / 2.0;
        Outline::Polygon(vec![
            at(r0, a, -w),
            at(r0, a, w),
            at(r1, a, w),
            at(r1, a, -w),
        ])
    } else {
        let half = width / 2.0 / ((r0 + r1) / 2.0);
        Outline::Sector {
            r_in: r0,
            r_out: r1,
            from: a - half,
            to: a + half,
        }
    }
}

/// A regular `n`-gon with its flats at `apothem`, flat 0 at the top; a disc of that radius when
/// not `faceted`.
fn polygon_or_disc(n: i64, apothem: f64, faceted: bool) -> Outline {
    if !faceted {
        return Outline::Disc { r: apothem };
    }
    let corner = apothem / (PI / n as f64).cos();
    Outline::Polygon(
        (0..n)
            .map(|k| {
                let a = face_angle(k, n) + PI / n as f64;
                [corner * a.cos(), corner * a.sin()]
            })
            .collect(),
    )
}

/// The level of a gap: red below zero, amber when not a number.
fn gap_level(x: f64) -> Option<Level> {
    if x.is_nan() {
        Some(Level::Caution)
    } else if x < 0.0 {
        Some(Level::Bad)
    } else {
        None
    }
}

/// The body's largest outside diameter, the cup's, the cap's or the boss's [mm]: the pieces'
/// extent across the axis in both views.
fn body_od(inputs: &DesignInputs, results: &DesignResults) -> f64 {
    let md = &inputs.metal;
    results.model.cup_od_mm.max(md.cap_od_mm).max(md.boss_od_mm)
}

/// A space-claim axis as the views draw it: `Some(claim)`, or `None` with a note when the claim
/// is off the drawing, farther than [`CLAIM_REACH`] times `extent`, the pieces' extent along its
/// axis. A claim that is not a number stays: its line holds NaN and is dropped as not finite.
fn drawn_claim(axis: &str, claim: f64, extent: f64, notes: &mut Vec<Note>) -> Option<f64> {
    if claim.abs() > CLAIM_REACH * extent.abs() {
        notes.push(Note {
            text: format!(
                "{NOT_DRAWN}: the {axis} claim {} is off the drawing",
                mm(claim)
            ),
            level: Some(Level::Caution),
        });
        None
    } else {
        Some(claim)
    }
}

/// The end view: the section through the magnets, seen from the free end, with the diameter
/// claim when it is drawn (`diameter`, [`drawn_claim`]).
fn end_view(
    inputs: &DesignInputs,
    results: &DesignResults,
    diameter: Option<f64>,
    notes: &mut Vec<Note>,
) -> View {
    let (c, m, ret) = (&inputs.coupling, &results.model, &results.retainers);
    let md = &inputs.metal;
    let n = c.npole;
    let faceted = c.faceted == 1;
    let draw_blocks = (2..=MAX_DRAWN_POLES).contains(&n);
    // Without blocks there are no flats to draw: the pocket and the hub become discs.
    let flats = faceted && draw_blocks;
    let a_i = c.inner_back_apothem_mm;
    let pocket = m.outer_back_apothem_mm + md.bond_outer_mm;
    let mut pieces = vec![
        Piece {
            part: Part::Body,
            outline: Outline::Disc {
                r: m.cup_od_mm / 2.0,
            },
        },
        Piece {
            part: Part::Cavity,
            outline: polygon_or_disc(n, pocket, flats),
        },
    ];
    let ring = |k: i64, r0: f64, r1: f64, width: f64, north_first: bool| Piece {
        part: Part::Magnet {
            north: (k % 2 == 0) == north_first,
        },
        outline: block(k, n, r0, r1, width, faceted),
    };
    if draw_blocks {
        for k in 0..n {
            pieces.push(ring(
                k,
                m.outer_face_apothem_mm,
                m.outer_back_apothem_mm,
                m.outer_width_mm,
                false,
            ));
        }
    } else {
        notes.push(Note {
            text: format!(
                "{BLOCKS_NOT_DRAWN}: {n} poles per ring is outside 2 to {MAX_DRAWN_POLES}"
            ),
            level: Some(Level::Caution),
        });
    }
    pieces.push(Piece {
        part: Part::Retainer,
        outline: Outline::Ring {
            r_in: ret.liner_id_mm / 2.0,
            r_out: ret.liner_od_mm / 2.0,
        },
    });
    pieces.push(Piece {
        part: Part::Retainer,
        outline: Outline::Ring {
            r_in: ret.sleeve_id_mm / 2.0,
            r_out: ret.sleeve_od_mm / 2.0,
        },
    });
    if draw_blocks {
        for k in 0..n {
            pieces.push(ring(k, a_i, m.inner_face_radius_mm, m.inner_width_mm, true));
        }
    }
    pieces.push(Piece {
        part: Part::Body,
        outline: polygon_or_disc(n, a_i - md.bond_inner_mm, flats),
    });
    let bore = c.bore_mm / 2.0;
    pieces.push(Piece {
        part: Part::Shaft,
        outline: Outline::Disc { r: bore },
    });
    // The key in the hub's keyway, at 0°: from where its sides cross the bore to the keyway's
    // depth past the bore (Calculator C40), the key's width (Shaft clamps C58).
    let w = inputs.clamps.key_width_mm / 2.0;
    let seat = (bore * bore - w * w).max(0.0).sqrt();
    let tip = bore + c.keyway_depth_mm;
    pieces.push(Piece {
        part: Part::Key,
        outline: Outline::Polygon(vec![[seat, -w], [tip, -w], [tip, w], [seat, w]]),
    });

    // Dimensions: the face gap at face 0; the corner gap along face 1's normal, from the circle
    // its inner block's corners sweep to its outer block's flat (the corner gap is the outer
    // face apothem less the inner corner radius); the running clearance at the bottom.
    let top = face_angle(0, n.max(1));
    let face_1 = if n >= 1 { face_angle(1, n) } else { top };
    // The angle from face 1's centre to its inner block's corners.
    let corner_half = if faceted {
        (m.inner_width_mm / 2.0).atan2(m.inner_face_radius_mm)
    } else {
        m.inner_width_mm / 2.0 / ((a_i + m.inner_face_radius_mm) / 2.0)
    };
    let scale = |r: f64, a: f64| {
        let u = radial(a);
        [r * u[0], r * u[1]]
    };
    let mt = &results.metal;
    let clearance_level = if !mt.min_running_clearance_mm.is_finite() {
        Some(Level::Caution)
    } else {
        match verdict_level("metal.clearance_check", &mt.clearance_check) {
            Some(Level::Bad) => Some(Level::Bad),
            _ => None,
        }
    };
    let callouts = vec![
        Callout {
            tag: 1,
            path: "model.face_gap_mm",
            text: format!("Face gap {}", mm(m.face_gap_mm)),
            level: gap_level(m.face_gap_mm),
            from: scale(m.inner_face_radius_mm, top),
            to: scale(m.outer_face_apothem_mm, top),
        },
        Callout {
            tag: 2,
            path: "model.corner_gap_mm",
            text: format!("Corner gap {}", mm(m.corner_gap_mm)),
            level: gap_level(m.corner_gap_mm),
            from: scale(m.inner_corner_radius_mm, face_1),
            to: scale(m.outer_face_apothem_mm, face_1),
        },
        Callout {
            tag: 3,
            path: "metal.min_running_clearance_mm",
            text: format!(
                "Running clearance {} (sleeve-to-liner gap {} less movement {})",
                mm(mt.min_running_clearance_mm),
                mm(mt.sleeve_liner_clearance_mm),
                mm(mt.adverse_movement_mm)
            ),
            level: clearance_level,
            from: scale(ret.sleeve_od_mm / 2.0, -PI / 2.0),
            to: scale(ret.liner_id_mm / 2.0, -PI / 2.0),
        },
    ];

    let mut dashed = Vec::new();
    let mut r = m.cup_od_mm / 2.0;
    if let Some(d) = diameter {
        let claim = d / 2.0;
        let over = results.housing.diameter_overshoot_mm;
        dashed.push(Dashed {
            points: (0..=64)
                .map(|i| scale(claim, 2.0 * PI * f64::from(i) / 64.0))
                .collect(),
            level: overshoot_level(over).filter(|l| *l == Level::Bad),
        });
        r = r.max(claim.abs());
    }
    // The arc the inner corners of face 1 sweep, from one corner's angle to the other's.
    dashed.push(Dashed {
        points: (0..=16)
            .map(|i| {
                let a = face_1 - corner_half + 2.0 * corner_half * f64::from(i) / 16.0;
                scale(m.inner_corner_radius_mm, a)
            })
            .collect(),
        level: None,
    });
    let r = r + 2.5;
    View {
        pieces,
        dashed,
        callouts,
        min: [-r, -r],
        max: [r, r],
    }
}

/// The level of a space-claim axis from its overshoot: red past the claim, amber when not a
/// number.
fn overshoot_level(over: f64) -> Option<Level> {
    if over.is_nan() {
        Some(Level::Caution)
    } else if over > 0.0 {
        Some(Level::Bad)
    } else {
        None
    }
}

/// One space-claim axis as a callout: the dimension it measures against its claim, or by how
/// much it passes it (the overshoot's result: housing.*), or unknown.
#[allow(clippy::too_many_arguments)] // one axis: its words, its two results, its numbers
fn claim_callout(
    tag: usize,
    axis: &str,
    dimension: &str,
    dimension_path: &'static str,
    overshoot_path: &'static str,
    value: f64,
    claim: f64,
    over: f64,
    line: (Mm, Mm),
) -> Callout {
    let level = overshoot_level(over);
    let (path, text) = match level {
        Some(Level::Bad) => (
            overshoot_path,
            format!(
                "{axis}: {dimension} {} is {} over the {} claim",
                mm(value),
                mm(over),
                mm(claim)
            ),
        ),
        Some(_) => (
            overshoot_path,
            format!("{axis}: unknown (a dimension or the claim is not a number)"),
        ),
        None => (
            dimension_path,
            format!("{axis}: {dimension} {} of {}", mm(value), mm(claim)),
        ),
    };
    Callout {
        tag,
        path,
        text,
        level,
        from: line.0,
        to: line.1,
    }
}

/// The side view: the upper half section through the flat centres, the axis along x from the
/// cap's front face, with the claim's lines that are drawn (`diameter`, [`drawn_claim`]; the
/// overall length and the bay are decided here, against the axial stack).
fn side_view(
    inputs: &DesignInputs,
    results: &DesignResults,
    diameter: Option<f64>,
    notes: &mut Vec<Note>,
) -> View {
    let (c, md, m) = (&inputs.coupling, &inputs.metal, &results.model);
    let (ret, h, mt) = (&results.retainers, &results.housing, &results.metal);
    let bore = c.bore_mm / 2.0;
    let cap = md.cap_axial_mm;
    let depth = h.cup_depth_mm;
    let web_end = cap + depth + md.web_mm;
    let boss_end = web_end + md.boss_length_mm;
    let rect = |part: Part, x0: f64, x1: f64, y0: f64, y1: f64| Piece {
        part,
        outline: Outline::Rect {
            min: [x0, y0],
            max: [x1, y1],
        },
    };
    // Inside the cavity every part is centred on it (the workbook gives no axial positions).
    let mid = cap + depth / 2.0;
    let centred = |part: Part, length: f64, y0: f64, y1: f64| {
        rect(part, mid - length / 2.0, mid + length / 2.0, y0, y1)
    };
    let pieces = vec![
        rect(
            Part::Body,
            cap,
            cap + depth,
            m.outer_back_apothem_mm + md.bond_outer_mm,
            m.cup_od_mm / 2.0,
        ),
        rect(Part::Body, cap + depth, web_end, bore, m.cup_od_mm / 2.0),
        rect(Part::Body, web_end, boss_end, bore, md.boss_od_mm / 2.0),
        rect(
            Part::Cap,
            0.0,
            cap,
            ret.liner_id_mm / 2.0,
            md.cap_od_mm / 2.0,
        ),
        centred(
            Part::Magnet { north: true },
            m.outer_length_mm,
            m.outer_face_apothem_mm,
            m.outer_back_apothem_mm,
        ),
        centred(
            Part::Retainer,
            h.retainer_span_mm,
            ret.liner_id_mm / 2.0,
            ret.liner_od_mm / 2.0,
        ),
        centred(
            Part::Retainer,
            h.retainer_span_mm,
            ret.sleeve_id_mm / 2.0,
            ret.sleeve_od_mm / 2.0,
        ),
        centred(
            Part::Magnet { north: false },
            m.inner_length_mm,
            c.inner_back_apothem_mm,
            m.inner_face_radius_mm,
        ),
        centred(
            Part::Body,
            h.hub_length_mm,
            bore,
            c.inner_back_apothem_mm - md.bond_inner_mm,
        ),
    ];
    // The pieces' extent: the axial stack and the body's top.
    let axial = boss_end.max(mt.axial_stack_mm);
    let body_top = body_od(inputs, results) / 2.0;
    let length = drawn_claim("overall length", md.max_overall_axial_mm, axial, notes);
    let bay = drawn_claim(
        "large-diameter bay",
        md.max_large_dia_axial_mm,
        axial,
        notes,
    );
    let radius = diameter.map(|d| d / 2.0);
    // The claim's lines reach its radius and its length where they are drawn and finite, else
    // the body's top and the axial stack.
    let claim_top = radius.filter(|r| r.is_finite()).unwrap_or(body_top);
    let claim_end = length.filter(|l| l.is_finite()).unwrap_or(axial);
    let red = |over: f64| overshoot_level(over).filter(|l| *l == Level::Bad);
    let right = claim_end.max(bay.unwrap_or(axial)).max(axial) + 2.0;
    let mut dashed = vec![
        Dashed {
            points: vec![[-1.0, 0.0], [right, 0.0]],
            level: None,
        },
        Dashed {
            points: vec![[0.0, 0.0], [0.0, claim_top]],
            level: None,
        },
    ];
    if let Some(radius) = radius {
        dashed.push(Dashed {
            points: vec![[0.0, radius], [claim_end, radius]],
            level: red(h.diameter_overshoot_mm),
        });
    }
    if let Some(length) = length {
        dashed.push(Dashed {
            points: vec![[length, 0.0], [length, claim_top]],
            level: red(h.length_overshoot_mm),
        });
    }
    if let Some(bay) = bay {
        dashed.push(Dashed {
            points: vec![[bay, 0.0], [bay, claim_top]],
            level: red(h.bay_overshoot_mm),
        });
    }
    let callouts = vec![
        claim_callout(
            4,
            "Overall length",
            "axial stack",
            "metal.axial_stack_mm",
            "housing.length_overshoot_mm",
            mt.axial_stack_mm,
            md.max_overall_axial_mm,
            h.length_overshoot_mm,
            ([0.0, -2.0], [mt.axial_stack_mm, -2.0]),
        ),
        claim_callout(
            5,
            "Large-diameter bay",
            "stack",
            "metal.large_dia_stack_mm",
            "housing.bay_overshoot_mm",
            mt.large_dia_stack_mm,
            md.max_large_dia_axial_mm,
            h.bay_overshoot_mm,
            ([0.0, -4.0], [mt.large_dia_stack_mm, -4.0]),
        ),
        claim_callout(
            6,
            "Diameter",
            "rotating OD",
            "metal.rotating_od_mm",
            "housing.diameter_overshoot_mm",
            mt.rotating_od_mm,
            md.max_diameter_mm,
            h.diameter_overshoot_mm,
            ([-2.0, 0.0], [-2.0, mt.rotating_od_mm / 2.0]),
        ),
    ];
    let top = radius.map_or(body_top, |r| body_top.max(r.abs())) + 2.0;
    View {
        pieces,
        dashed,
        callouts,
        min: [-4.0, -6.0],
        max: [right, top],
    }
}

/// The notes under the views: the autofit wall (Addendum A1, decision 27) and the two design
/// checks (decision 28, flagged only), each while it applies (decision M42-9).
fn notes(inputs: &DesignInputs, results: &DesignResults) -> Vec<Note> {
    let md = &inputs.metal;
    let mut notes = Vec::new();
    if let NumOrText::Num(wall) = results.materials.cup_wall_suggested_mm {
        notes.push(Note {
            text: format!(
                "{AUTOFIT}: the wall rule asks for at least {} at the pocket corners (input {})",
                mm(wall),
                mm(md.cup_wall_corner_mm)
            ),
            level: None,
        });
    }
    let cup_body = results.metal.cup_body_od_mm;
    if md.cap_thread_dia_mm < cup_body {
        notes.push(Note {
            text: format!(
                "{DESIGN_CHECK}: the cap thread diameter {} (Metal design C169) is below the cup body OD {} (C165)",
                mm(md.cap_thread_dia_mm),
                mm(cup_body)
            ),
            level: Some(Level::Caution),
        });
    }
    if md.boss_od_mm != inputs.clamps.boss_od_mm {
        notes.push(Note {
            text: format!(
                "{DESIGN_CHECK}: the boss OD is {} in Metal design (C127) and {} in Shaft clamps (C35)",
                mm(md.boss_od_mm),
                mm(inputs.clamps.boss_od_mm)
            ),
            level: Some(Level::Caution),
        });
    }
    notes
}

/// The note for `dropped` parts of the `name` view that hold a number that is not finite.
fn parts_not_drawn(dropped: usize, name: &str) -> String {
    if dropped == 1 {
        format!("{NOT_DRAWN}: 1 part of the {name} view holds a number that is not a number")
    } else {
        format!(
            "{NOT_DRAWN}: {dropped} parts of the {name} view hold a number that is not a number"
        )
    }
}

/// The geometry of the design shown: its inputs and the results computed from them.
pub fn geometry(inputs: &DesignInputs, results: &DesignResults) -> Geometry {
    let mut all_notes = Vec::new();
    // The diameter claim is decided once, for both views, against the body's largest OD.
    let diameter = drawn_claim(
        "diameter",
        inputs.metal.max_diameter_mm,
        body_od(inputs, results),
        &mut all_notes,
    );
    let mut end = end_view(inputs, results, diameter, &mut all_notes);
    let mut side = side_view(inputs, results, diameter, &mut all_notes);
    for (view, name) in [(&mut end, "end"), (&mut side, "side")] {
        let dropped = view.retain_finite();
        if !view.is_drawable() {
            all_notes.push(Note {
                text: format!("{NOT_DRAWN}: a dimension of the {name} view is not a number"),
                level: Some(Level::Caution),
            });
        } else if dropped > 0 {
            all_notes.push(Note {
                text: parts_not_drawn(dropped, name),
                level: Some(Level::Caution),
            });
        }
    }
    all_notes.extend(notes(inputs, results));
    Geometry {
        end,
        side,
        notes: all_notes,
    }
}

#[cfg(test)]
mod tests {
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib gui::geometry 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 11 passed; 0 failed` for `gui::geometry`, then `test result: ok. 320 passed; 0 failed` for the whole library (Task 0's 309 plus this plan's tests so far).

- [ ] **Step 5: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/geometry.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 commit -F - <<'EOF'
feat(magcoupling-rs): the geometry view's drawing in millimetres

gui::geometry turns the design shown into the end view (cup and pocket, both
rings of blocks on their flats or as arcs, liner, sleeve, hub, shaft, key, the
diameter claim) and the upper half side view (cap, cup wall, web, boss and the
cavity's parts, centred; the overall length, bay and diameter claims), every
dimension a result of this frame and the axial ones the housing in effect.
Callouts carry their result path and level: face gap, corner gap, running
clearance, and the space claim per axis (decisions M42-3, M42-4); notes give
the autofit wall and the two design checks while they hold (M42-9). A claim
line past ten times the pieces' extent is left off with a note; NaN pieces
and lines are dropped and counted in a note; blocks are drawn for 2 to 200
poles.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m42 status --short
```

Expected: one new commit on `magcoupling/m4-2`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 2: The geometry view, the centre region's default tab

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

`geometry_ui` paints Task 1's views with egui's painter at one scale (`side_by_side`: the largest that fits both side by side, decision M42-3; no scale and nothing painted when there is no room), draws each dimension as a line with arrowheads and its tag, lists the callouts (tag and text in their level's colour) and the notes under the drawing, and shows a callout's result hover text while its line or its text is hovered. That text comes from the new `dashboard::hover_text` (the hover hook's text with the correction marks), which the results table's `row_tooltip` now uses too: one place, which M4-3's equation tooltip joins. The panel's `CentreView` gains `Geometry`, the default view (M42-2); the tab row wraps when the region is narrow; the end-effect banner moves from the results table to the top of the centre region, over every view (M42-1); `ui` binds the design shown once per frame and passes it to the centre region (no second clone). The default view changes, so the panel tests that use the results table open its tab first; the glyph test clicks every tab and adds the callouts' hover texts.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/geometry_view.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/mod.rs` (`pub mod geometry_view;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/dashboard.rs` (`hover_text` and its test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/results_table.rs` (`row_tooltip` through `hover_text`; the banner removed)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/test_support.rs` (`flat_shapes`; `drawn_texts`, `text_rect` through it; `text_rects`, `text_color`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs` (`CentreView::Geometry`, the banner, the shown design; tests)

**Interfaces:**
- Consumes: Task 1's `geometry`, `Geometry`, `View`, `Callout`, `Mm`, `Outline`, `Part`, `finite` (and in the tests `NOT_DRAWN`, `mm`); `dashboard::{Level, result_info, result_tooltip, ResultNotes, end_effect_banner}`, `corrections::CorrectionIndex`.
- Produces (`magcoupling::gui::geometry_view`): `pub const DIMENSION: Color32`; `struct Transform { pub origin: Pos2, pub scale: f32 }` with `fn to_px(&self, Mm) -> Pos2` (y up) and `fn len(&self, f64) -> f32`; `struct GeometryLayout { pub rect: Rect, pub scale: f32, pub end: Option<Transform>, pub side: Option<Transform>, pub dimensions: Vec<(usize, Rect)> }`; `fn part_color(Part, &egui::Visuals) -> Color32`; `pub(crate) fn side_by_side(Rect, &[(Mm, Mm)], f64) -> Option<(f32, Vec<Transform>)>` (the scale and each view's transform, `None` without room or for an extent that is not finite); `pub(crate) fn arrowhead(&egui::Painter, Pos2, Vec2, Stroke)`; `fn geometry_ui(&mut egui::Ui, &DesignInputs, &DesignResults) -> GeometryLayout`.
- Produces (`gui::dashboard`): `pub fn hover_text(path: &str) -> Option<String>`. (`gui::panel`): `CentreView::Geometry`, `CentreView::ALL: [CentreView; 2]`, the default `centre`. (`gui::test_support`, tests only): `flat_shapes(&egui::FullOutput) -> Vec<&egui::Shape>` (nested shapes flattened, in paint order), `text_rects(&egui::FullOutput, &str) -> Vec<egui::Rect>`, `text_color(&egui::FullOutput, &str) -> Option<egui::Color32>`; `drawn_texts` and `text_rect` keep their signatures and walk `flat_shapes`.

- [ ] **Step 1: Write the failing tests**

The new module's docs and tests; `hover_text`'s test; the test helpers (`flat_shapes`, which `drawn_texts`, `text_rects`, `text_rect`, `text_color` (a drawn text's colour) and the tests' own filters walk); the panel's new test (the default view, a live redraw on an arrow key, the sized design in Torque -> Magnets, switching tabs) and the edits that open the results table's tab in the tests that use it (the banner test keeps its two banners: the dashboard's and the centre region's). The painter's tests pin the shared scale (the largest that fits inside the margins), known dimensions at their pixel lengths (the cup OD, the shaft, the face gap; the cup depth and the cap as painted rectangles), the callout texts and their colours (the default running clearance red, its line too), an overshoot in red, the hover text of a dimension under the pointer (tooltips at once: `tooltip_delay` 0), NaN and a tiny screen, a 0 x 0 and a 1 x 1 screen (no view painted; `side_by_side`'s refusals and its placement), and a 1e300 mm claim (the pieces still fill the drawing; its note in amber).

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
    }

    #[test]
    fn badge_colours_follow_the_theme() {
        let dark = egui::Visuals::dark();
        let light = egui::Visuals::light();
````

with:

````rust
    }

    #[test]
    fn the_hover_text_carries_the_marks_and_needs_a_result() {
        let text = hover_text("model.pullout_Nm").unwrap();
        assert!(
            text.starts_with("Pull-out torque at operating temperature\n"),
            "{text}"
        );
        assert!(text.contains("Corrected vs workbook:"), "{text}");
        assert!(
            hover_text("housing.length_overshoot_mm")
                .unwrap()
                .contains("Rust-only result (no workbook cell)")
        );
        assert_eq!(hover_text("no.such"), None);
    }

    #[test]
    fn badge_colours_follow_the_theme() {
        let dark = egui::Visuals::dark();
        let light = egui::Visuals::light();
````

Create `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/geometry_view.rs`:

````rust
//! The geometry view (spec M4 "Layout": the centre region's default view, decision M42-2):
//! paints [`crate::gui::geometry`] to scale with egui's painter, both views at one scale
//! (decision M42-3), then lists the dimension callouts and the notes under them.
//!
//! Each dimension is a line with arrowheads and a tag; the list repeats the tag with the
//! callout's text. Hovering the line or its text shows the hover text of the result it shows
//! ([`crate::gui::dashboard::hover_text`], the hook M4-3's equation tooltip joins), built only
//! while hovered. A violated callout is red, one that is not a number amber (decision M42-4);
//! the space claim is dashed, an exceeded axis red. The view is rebuilt from the design shown on
//! every frame, so it follows a slider while it is dragged.
//!
//! [`side_by_side`] (views at one scale, nothing at all when there is no room) and [`arrowhead`]
//! are shared with the clamp drawing.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::geometry::{NOT_DRAWN, mm};
    use crate::gui::test_support::{
        drawn_texts, flat_shapes, sized_frame, sized_frame_at, text_color,
    };

    /// One frame of the geometry view of `inputs` on a `size` screen, with `events`.
    fn frame(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        size: Vec2,
        events: Vec<egui::Event>,
    ) -> (egui::FullOutput, GeometryLayout) {
        let results = compute_all(inputs);
        let mut layout = None;
        let output = sized_frame(ctx, size, events, |ui| {
            layout = Some(geometry_ui(ui, inputs, &results));
        });
        (output, layout.expect("drawn"))
    }

    /// Every filled circle's and every stroked circle's radius [points].
    fn circle_radii(output: &egui::FullOutput) -> Vec<f32> {
        flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Circle(c) => Some(c.radius),
                _ => None,
            })
            .collect()
    }

    /// Every painted rectangle's width [points].
    fn rect_widths(output: &egui::FullOutput) -> Vec<f32> {
        flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r) => Some(r.rect.width()),
                _ => None,
            })
            .collect()
    }

    fn has_radius(radii: &[f32], want: f32) -> bool {
        radii.iter().any(|r| (r - want).abs() < 1e-3)
    }

    #[test]
    fn the_views_share_one_scale_that_fits_the_screen() {
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let (_, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 700.0), Vec::new());
        let g = geometry(&inputs, &compute_all(&inputs));
        let end = layout.end.expect("the end view is drawn");
        let side = layout.side.expect("the side view is drawn");
        assert_eq!(end.scale, side.scale);
        // Both views side by side, at the largest scale that fits inside the margins.
        let width = (g.end.max[0] - g.end.min[0]) + VIEW_GAP_MM + (g.side.max[0] - g.side.min[0]);
        let height = (g.end.max[1] - g.end.min[1]).max(g.side.max[1] - g.side.min[1]);
        let inner = layout.rect.shrink(MARGIN);
        let want = (inner.width() / width as f32).min(inner.height() / height as f32);
        assert_eq!(layout.scale, want);
        // The end view's extent lands inside the drawing.
        assert!(inner.contains(end.to_px(g.end.min)) && inner.contains(end.to_px(g.end.max)));
        assert!(layout.rect.width() > 900.0 && layout.rect.height() >= MIN_DRAWING_HEIGHT);
    }

    #[test]
    fn known_dimensions_map_to_their_pixel_distances() {
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let (output, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 700.0), Vec::new());
        let s = layout.scale;
        let radii = circle_radii(&output);
        // The cup OD (41.33 mm) and the 10 mm shaft, to scale.
        assert!(
            has_radius(&radii, (r.model.cup_od_mm / 2.0) as f32 * s),
            "{radii:?}"
        );
        assert!(has_radius(&radii, 5.0 * s), "{radii:?}");
        // The face gap's dimension spans 1.4 mm at the scale.
        let end = layout.end.unwrap();
        let g = geometry(&inputs, &r);
        let face = g.callouts().find(|c| c.tag == 1).unwrap();
        let span = (end.to_px(face.to) - end.to_px(face.from)).length();
        assert!((span - 1.4 * s).abs() < 1e-2, "{span} vs {}", 1.4 * s);
        // The side view, painted at the same scale: the cup wall spans the cup depth in effect
        // and the cap its 0.8 mm.
        assert!(layout.side.is_some());
        let widths = rect_widths(&output);
        for mm in [r.housing.cup_depth_mm, inputs.metal.cap_axial_mm] {
            assert!(
                widths.iter().any(|w| (w - mm as f32 * s).abs() < 1e-2),
                "{mm} mm in {widths:?}"
            );
        }
    }

    #[test]
    fn the_callouts_and_notes_are_listed_with_their_colours() {
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let (output, _) = frame(&ctx, &inputs, egui::vec2(1000.0, 900.0), Vec::new());
        let texts = drawn_texts(&output);
        for want in [
            "Face gap 1.400 mm",
            "Corner gap 1.027 mm",
            "Overall length: axial stack 31.80 mm of 35.00 mm",
            "Large-diameter bay: stack 18.80 mm of 20.00 mm",
            "Diameter: rotating OD 42.80 mm of 43.00 mm",
        ] {
            assert!(
                texts.iter().any(|t| t == want),
                "missing {want:?} in {texts:?}"
            );
            assert_eq!(
                text_color(&output, want),
                Some(ctx.style().visuals.text_color())
            );
        }
        // The default running clearance is below zero: its text and its line are red.
        let run =
            "Running clearance -0.1032 mm (sleeve-to-liner gap 0.6768 mm less movement 0.7800 mm)";
        let red = Level::Bad.color(&ctx.style().visuals);
        assert_eq!(text_color(&output, run), Some(red));
        assert!(segment_colors(&output).contains(&red));
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("Design check: the boss OD"))
        );
        assert!(texts.iter().any(|t| t.starts_with("Autofit:")));
    }

    /// The stroke colour of every line segment.
    fn segment_colors(output: &egui::FullOutput) -> Vec<Color32> {
        flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::LineSegment { stroke, .. } => Some(stroke.color),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_design_past_the_space_claim_draws_the_exceeded_axes_red() {
        let ctx = egui::Context::default();
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.axial_length_mm = Some(50.8);
        let (output, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 900.0), Vec::new());
        let red = Level::Bad.color(&ctx.style().visuals);
        let text = "Overall length: axial stack 69.90 mm is 34.90 mm over the 35.00 mm claim";
        assert_eq!(text_color(&output, text), Some(red));
        // The views still fit: the stack's 69.9 mm sets the scale.
        assert!(layout.side.is_some() && layout.scale > 0.0);
        assert!(
            segment_colors(&output)
                .iter()
                .filter(|c| **c == red)
                .count()
                >= 2
        );
    }

    #[test]
    fn hovering_a_dimension_shows_its_result_s_hover_text() {
        let ctx = egui::Context::default();
        ctx.style_mut(|s| {
            s.interaction.tooltip_delay = 0.0;
            s.interaction.show_tooltips_only_when_still = false;
        });
        let inputs = DesignInputs::default();
        let size = egui::vec2(1000.0, 700.0);
        let (_, layout) = frame(&ctx, &inputs, size, Vec::new());
        let (_, rect) = layout
            .dimensions
            .iter()
            .find(|(tag, _)| *tag == 1)
            .copied()
            .expect("the face gap is drawn");
        let at = rect.center();
        let want = hover_text("model.face_gap_mm").unwrap();
        let (before, _) = frame(&ctx, &inputs, size, vec![egui::Event::PointerMoved(at)]);
        assert!(
            !drawn_texts(&before).contains(&want),
            "not before the pointer is there"
        );
        let results = compute_all(&inputs);
        let mut output = None;
        for dt in [0.1, 0.2] {
            output = Some(sized_frame_at(&ctx, size, Some(dt), Vec::new(), |ui| {
                geometry_ui(ui, &inputs, &results);
            }));
        }
        assert!(
            want.starts_with("Gap at the flat centres (effective gap)\n"),
            "{want}"
        );
        let texts = drawn_texts(&output.unwrap());
        assert!(texts.contains(&want), "no hover text in {texts:?}");
    }

    #[test]
    fn a_dimension_that_is_not_a_number_draws_without_panicking() {
        // validate() refuses NaN at every boundary; a struct literal can still hold it.
        let ctx = egui::Context::default();
        let mut inputs = DesignInputs::default();
        inputs.metal.cup_wall_corner_mm = f64::NAN;
        inputs.metal.max_diameter_mm = f64::NAN;
        let (output, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 700.0), Vec::new());
        assert!(layout.end.is_none());
        assert!(layout.side.is_some());
        let texts = drawn_texts(&output);
        assert!(
            texts
                .iter()
                .any(|t| t == "Not drawn: a dimension of the end view is not a number")
        );
        // A tiny screen still draws.
        frame(
            &ctx,
            &DesignInputs::default(),
            egui::vec2(120.0, 90.0),
            Vec::new(),
        );
    }

    #[test]
    fn a_region_with_no_room_draws_no_view() {
        // A window shrunk to nothing: no scale, so neither view is painted (nor mirrored).
        let ctx = egui::Context::default();
        for size in [egui::vec2(0.0, 0.0), egui::vec2(1.0, 1.0)] {
            let (_, layout) = frame(&ctx, &DesignInputs::default(), size, Vec::new());
            assert!(layout.scale >= 0.0, "{size:?}: {layout:?}");
            assert!(
                layout.end.is_none() && layout.side.is_none(),
                "{size:?}: {layout:?}"
            );
            assert!(layout.dimensions.is_empty());
        }
        // side_by_side itself: no room, no extent, or an extent that is not finite; else the
        // largest scale that fits, the views centred.
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(100.0, 100.0));
        let unit: (Mm, Mm) = ([0.0, 0.0], [1.0, 1.0]);
        let empty = Rect::from_min_size(Pos2::ZERO, Vec2::ZERO);
        assert!(side_by_side(empty, &[unit], 0.0).is_none());
        assert!(side_by_side(rect, &[], 0.0).is_none());
        assert!(side_by_side(rect, &[([0.0, 0.0], [f64::NAN, 1.0])], 0.0).is_none());
        let (scale, transforms) = side_by_side(rect, &[unit, unit], 2.0).unwrap();
        assert_eq!(scale, 25.0, "1 + 2 + 1 mm across 100 points");
        assert_eq!(transforms[0].to_px([0.0, 0.0]), Pos2::new(0.0, 62.5));
        assert_eq!(transforms[1].to_px([0.0, 0.0]), Pos2::new(75.0, 62.5));
    }

    #[test]
    fn a_claim_far_past_the_pieces_keeps_the_drawing_to_scale() {
        // A design file's finite but huge claim is left off: the pieces still fill the drawing.
        let ctx = egui::Context::default();
        let mut inputs = DesignInputs::default();
        inputs.metal.max_diameter_mm = 1e300;
        let (output, layout) = frame(&ctx, &inputs, egui::vec2(1000.0, 700.0), Vec::new());
        assert!(layout.end.is_some() && layout.side.is_some());
        let g = geometry(&inputs, &compute_all(&inputs));
        let tall = (g.end.max[1] - g.end.min[1]) as f32 * layout.scale;
        assert!(
            layout.scale > 0.0 && tall > 0.5 * layout.rect.height(),
            "{} points per mm",
            layout.scale
        );
        let note = format!(
            "{NOT_DRAWN}: the diameter claim {} is off the drawing",
            mm(1e300)
        );
        assert_eq!(
            text_color(&output, &note),
            Some(Level::Caution.color(&ctx.style().visuals))
        );
    }
}
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub mod dashboard;
mod format;
pub mod geometry;
pub mod history;
pub mod input_ui;
pub mod inputs;
````

with:

````rust
pub mod dashboard;
mod format;
pub mod geometry;
pub mod geometry_view;
pub mod history;
pub mod input_ui;
pub mod inputs;
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            label_rect.bottom() <= value_rect.top() && value_rect.bottom() <= note_rect.top(),
            "{value:?} at {value_rect:?} is not in the row between {label_rect:?} and {note_rect:?}"
        );
    }

    #[test]
````

with:

````rust
            label_rect.bottom() <= value_rect.top() && value_rect.bottom() <= note_rect.top(),
            "{value:?} at {value_rect:?} is not in the row between {label_rect:?} and {note_rect:?}"
        );
    }

    #[test]
    fn the_geometry_view_is_the_default_and_follows_the_design_shown() {
        let mut harness = Harness::new();
        assert_eq!(harness.panel.centre, CentreView::Geometry);
        let output = harness.frame(Vec::new());
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == "Face gap 1.400 mm"), "{texts:?}");
        // An arrow key on the face gap moves the callout in the same frame (a live redraw).
        harness.focus(FACE_GAP);
        let output = harness.frame(key_tap(egui::Key::ArrowRight));
        assert!(
            drawn_texts(&output)
                .iter()
                .any(|t| t == "Face gap 1.410 mm")
        );
        // In Torque -> Magnets it draws the sized design: the axial stack follows the solved
        // length.
        let output = harness.size();
        let stack = harness.panel.results().metal.axial_stack_mm;
        assert_ne!(
            stack,
            compute_all(&DesignInputs::default()).metal.axial_stack_mm
        );
        let want = format!(
            "Overall length: axial stack {} of 35.00 mm",
            crate::gui::geometry::mm(stack)
        );
        assert!(drawn_texts(&output).contains(&want), "{want}");
        // The Results tab shows the table; the Geometry tab brings the view back.
        harness.click_text(CentreView::Results.label());
        assert_eq!(harness.panel.centre, CentreView::Results);
        let total = crate::gui::results_table::table_entries().len();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        assert!(
            !drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with("Face gap"))
        );
        harness.click_text(CentreView::Geometry.label());
        assert_eq!(harness.panel.centre, CentreView::Geometry);
    }

    #[test]
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            .into_iter()
            .filter(|t| t.starts_with(&format!("{END_EFFECT_BANNER} (")))
            .collect();
        // Over the dashboard and over the results table.
        let text = "End-effect model out of range (f_end = -1.202): the pull-out and the numbers computed from it are greyed.";
        assert_eq!(banner, [text, text]);
    }
````

with:

````rust
            .into_iter()
            .filter(|t| t.starts_with(&format!("{END_EFFECT_BANNER} (")))
            .collect();
        // Over the dashboard and over the centre region, whichever view it shows (decision
        // M42-1).
        let text = "End-effect model out of range (f_end = -1.202): the pull-out and the numbers computed from it are greyed.";
        assert_eq!(banner, [text, text]);
    }
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        let cell = |index: usize| entries[index].info.cell.clone().unwrap();
        let (first, last) = (cell(0), cell(entries.len() - 1));
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        let total = entries.len();
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        // The first rows are on screen (they show their cells), the last is far below.
````

with:

````rust
        let cell = |index: usize| entries[index].info.cell.clone().unwrap();
        let (first, last) = (cell(0), cell(entries.len() - 1));
        let mut harness = Harness::new();
        let output = harness.click_text(CentreView::Results.label());
        let total = entries.len();
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        // The first rows are on screen (they show their cells), the last is far below.
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        use crate::gui::results_table::{EXPORT_CSV, EXPORT_JSON};
        let mut harness = Harness::new();
        assert!(harness.panel.take_requests().is_empty());
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.click_text(EXPORT_CSV);
````

with:

````rust
        use crate::gui::results_table::{EXPORT_CSV, EXPORT_JSON};
        let mut harness = Harness::new();
        assert!(harness.panel.take_requests().is_empty());
        harness.click_text(CentreView::Results.label());
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.click_text(EXPORT_CSV);
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        // The search box is a text field, but it edits no design: a change while it has focus
        // (a design file the host's picker delivers) is an undo step at once.
        let mut harness = Harness::new();
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("pull".to_owned())]);
        assert!(
````

with:

````rust
        // The search box is a text field, but it edits no design: a change while it has focus
        // (a design file the host's picker delivers) is an undo step at once.
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("pull".to_owned())]);
        assert!(
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 12000.0));
        let mut texts = drawn_texts(&harness.frame(Vec::new()));
        texts.extend(drawn_texts(&harness.size()));
        for group in &InputCatalogue::get().groups {
            harness.click_text(group.label);
        }
````

with:

````rust
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 12000.0));
        let mut texts = drawn_texts(&harness.frame(Vec::new()));
        texts.extend(drawn_texts(&harness.size()));
        // Every view of the centre region, and the geometry callouts' hover texts.
        for view in CentreView::ALL {
            texts.extend(drawn_texts(&harness.click_text(view.label())));
            texts.extend(drawn_texts(&harness.frame(Vec::new())));
        }
        harness.click_text(CentreView::Geometry.label());
        let shown = harness.panel.shown_inputs();
        let geometry = crate::gui::geometry::geometry(&shown, harness.panel.results());
        texts.extend(
            geometry
                .callouts()
                .filter_map(|c| crate::gui::dashboard::hover_text(c.path)),
        );
        for group in &InputCatalogue::get().groups {
            harness.click_text(group.label);
        }
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        let mut harness = Harness::new();
        harness.size();
        harness.panel.sizing.target_Nm = 3.0;
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("pull".to_owned())]);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
````

with:

````rust
        let mut harness = Harness::new();
        harness.size();
        harness.panel.sizing.target_Nm = 3.0;
        harness.click_text(CentreView::Results.label());
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("pull".to_owned())]);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        use crate::gui::results_table::EXPORT_JSON;
        let mut harness = Harness::new();
        harness.size();
        harness.click_text(EXPORT_JSON);
        let requests = harness.panel.take_requests();
        let [PanelRequest::SaveFile { contents, .. }] = &requests[..] else {
````

with:

````rust
        use crate::gui::results_table::EXPORT_JSON;
        let mut harness = Harness::new();
        harness.size();
        harness.click_text(CentreView::Results.label());
        harness.click_text(EXPORT_JSON);
        let requests = harness.panel.take_requests();
        let [PanelRequest::SaveFile { contents, .. }] = &requests[..] else {
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/test_support.rs`, replace:

````rust
    }
}

/// Every text egui drew in a frame (widgets and painter text), nested shapes
/// included, in paint order.
pub(crate) fn drawn_texts(output: &egui::FullOutput) -> Vec<String> {
    fn walk(shape: &egui::Shape, texts: &mut Vec<String>) {
        match shape {
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, texts)),
            egui::Shape::Text(text) => texts.push(text.galley.text().to_owned()),
            _ => {}
        }
    }
    let mut texts = Vec::new();
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut texts);
    }
    texts
}

/// Screen rect of the first drawn text equal to `needle`, e.g. a button
/// label: lets a test click a widget whose id it cannot know.
pub(crate) fn text_rect(output: &egui::FullOutput, needle: &str) -> Option<egui::Rect> {
    fn walk(shape: &egui::Shape, needle: &str) -> Option<egui::Rect> {
        match shape {
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|s| walk(s, needle)),
            egui::Shape::Text(text) if text.galley.text() == needle => {
                Some(text.galley.rect.translate(text.pos.to_vec2()))
            }
            _ => None,
        }
    }
    output.shapes.iter().find_map(|c| walk(&c.shape, needle))
}
````

with:

````rust
    }
}

/// Every shape egui painted in a frame, the shapes nested in a `Shape::Vec`
/// flattened, in paint order: what the helpers below and the tests' own
/// filters walk.
pub(crate) fn flat_shapes(output: &egui::FullOutput) -> Vec<&egui::Shape> {
    fn walk<'a>(shape: &'a egui::Shape, shapes: &mut Vec<&'a egui::Shape>) {
        match shape {
            egui::Shape::Vec(nested) => nested.iter().for_each(|s| walk(s, shapes)),
            other => shapes.push(other),
        }
    }
    let mut shapes = Vec::new();
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut shapes);
    }
    shapes
}

/// Every text egui drew in a frame (widgets and painter text), nested shapes
/// included, in paint order.
pub(crate) fn drawn_texts(output: &egui::FullOutput) -> Vec<String> {
    flat_shapes(output)
        .into_iter()
        .filter_map(|shape| match shape {
            egui::Shape::Text(text) => Some(text.galley.text().to_owned()),
            _ => None,
        })
        .collect()
}

/// Screen rects of every drawn text equal to `needle`, in paint order.
pub(crate) fn text_rects(output: &egui::FullOutput, needle: &str) -> Vec<egui::Rect> {
    flat_shapes(output)
        .into_iter()
        .filter_map(|shape| match shape {
            egui::Shape::Text(text) if text.galley.text() == needle => {
                Some(text.galley.rect.translate(text.pos.to_vec2()))
            }
            _ => None,
        })
        .collect()
}

/// Screen rect of the first drawn text equal to `needle`, e.g. a button
/// label: lets a test click a widget whose id it cannot know.
pub(crate) fn text_rect(output: &egui::FullOutput, needle: &str) -> Option<egui::Rect> {
    text_rects(output, needle).into_iter().next()
}

/// The colour of the first drawn text equal to `needle`: its override, else its first section's
/// colour (a `RichText` colour or a painter text's), else the fallback colour.
pub(crate) fn text_color(output: &egui::FullOutput, needle: &str) -> Option<egui::Color32> {
    flat_shapes(output)
        .into_iter()
        .find_map(|shape| match shape {
            egui::Shape::Text(text) if text.galley.text() == needle => {
                Some(text.override_text_color.unwrap_or_else(|| {
                    match text.galley.job.sections.first().map(|s| s.format.color) {
                        Some(color) if color != egui::Color32::PLACEHOLDER => color,
                        _ => text.fallback_color,
                    }
                }))
            }
            _ => None,
        })
}
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib gui:: 2>&1 | grep -E "^error" | sort | uniq -c
```

Expected: FAIL to compile. The build stops with `error: could not compile `magcoupling-rs` (lib test) due to 47 previous errors; 1 warning emitted`; the errors name the items Step 3 adds, among them `error[E0425]: cannot find function `geometry_ui` in this scope`, `error[E0412]: cannot find type `GeometryLayout` in this scope`, `error[E0425]: cannot find function `side_by_side` in this scope`, `error[E0425]: cannot find function `hover_text` in module `crate::gui::dashboard`` and `error[E0599]: no variant or associated item named `Geometry` found for enum `gui::panel::CentreView` in the current scope`.

- [ ] **Step 3: Write the implementation**

The painter between its module's docs and tests; `hover_text` after `result_tooltip`; the results table without its banner and with `row_tooltip` through `hover_text`; the panel's tabs, the banner over the centre region and the design shown passed to it.

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
        lines.push(marker_tooltip(notes.marks));
    }
    lines.join("\n")
}

/// One dashboard row, ready to draw.
````

with:

````rust
        lines.push(marker_tooltip(notes.marks));
    }
    lines.join("\n")
}

/// The hover text of the result at `path` with its corrections' marks: the hook's text for a
/// readout outside the dashboard (a results-table row, a geometry callout); `None` for a path
/// that is no result.
pub fn hover_text(path: &str) -> Option<String> {
    let info = result_info(path)?;
    let marks = CorrectionIndex::get().marks(info.cell.as_deref());
    Some(result_tooltip(
        path,
        info,
        ResultNotes {
            marks,
            ..ResultNotes::default()
        },
    ))
}

/// One dashboard row, ready to draw.
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust

#[cfg(test)]
mod tests {
````

with:

````rust

use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};

use crate::gui::dashboard::{Level, hover_text};
use crate::gui::geometry::{Callout, Geometry, Mm, Outline, Part, View, finite, geometry};
use crate::{DesignInputs, DesignResults};

/// The colour of a dimension that is fine: drawing.py's dimension blue, lightened for a dark
/// background.
pub const DIMENSION: Color32 = Color32::from_rgb(90, 170, 230);

/// The gap between the end view and the side view [mm at the drawing's scale].
const VIEW_GAP_MM: f64 = 3.0;

/// The smallest height of the drawing [points]; the list under it scrolls when space is short.
const MIN_DRAWING_HEIGHT: f32 = 160.0;

/// The margin inside the drawing's area [points].
const MARGIN: f32 = 8.0;

/// Maps millimetres (x right, y up) to screen points (y down) at one scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    /// The screen point of the millimetre origin.
    pub origin: Pos2,
    /// Points per millimetre.
    pub scale: f32,
}

impl Transform {
    /// The screen point of `p`.
    pub fn to_px(&self, p: Mm) -> Pos2 {
        Pos2::new(
            self.origin.x + p[0] as f32 * self.scale,
            self.origin.y - p[1] as f32 * self.scale,
        )
    }

    /// A length in millimetres as points.
    pub fn len(&self, mm: f64) -> f32 {
        mm as f32 * self.scale
    }
}

/// What the geometry view drew: the drawing's area, the shared scale, each view's transform
/// (`None` when it is not drawable) and each dimension's hover area by tag.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometryLayout {
    pub rect: Rect,
    pub scale: f32,
    pub end: Option<Transform>,
    pub side: Option<Transform>,
    pub dimensions: Vec<(usize, Rect)>,
}

/// The fill colour of a part.
pub fn part_color(part: Part, visuals: &egui::Visuals) -> Color32 {
    match part {
        Part::Body => Color32::from_rgb(128, 132, 140),
        Part::Cavity => visuals.extreme_bg_color,
        Part::Magnet { north: true } => Color32::from_rgb(190, 110, 90),
        Part::Magnet { north: false } => Color32::from_rgb(95, 125, 175),
        Part::Retainer => Color32::from_rgb(150, 175, 165),
        Part::Cap => Color32::from_rgb(175, 180, 190),
        Part::Shaft => Color32::from_rgb(100, 104, 112),
        Part::Key => Color32::from_rgb(205, 180, 95),
    }
}

/// The colour of a callout or dashed line: its level's, else `plain`.
fn level_color(level: Option<Level>, plain: Color32, visuals: &egui::Visuals) -> Color32 {
    level.map_or(plain, |l| l.color(visuals))
}

/// Places views of the extents `extents` (each its min and max corner [mm]) side by side in
/// `rect`, `gap_mm` apart, centred, at one scale, the largest that fits: that scale [points per
/// mm] and each view's transform. `None` when there is no room (a region of no size or an
/// extent that is not finite, where the scale would be zero, negative or not a number and the
/// views mirrored): then nothing is painted.
pub(crate) fn side_by_side(
    rect: Rect,
    extents: &[(Mm, Mm)],
    gap_mm: f64,
) -> Option<(f32, Vec<Transform>)> {
    // f32::min passes over NaN, so an extent that is not finite is refused here.
    if !extents
        .iter()
        .all(|(min, max)| finite(*min) && finite(*max))
    {
        return None;
    }
    let gaps = extents.len().saturating_sub(1);
    let width: f64 = extents
        .iter()
        .map(|(min, max)| max[0] - min[0])
        .sum::<f64>()
        + gap_mm * gaps as f64;
    let height = extents
        .iter()
        .map(|(min, max)| max[1] - min[1])
        .fold(0.0, f64::max);
    let scale = (rect.width() / width as f32).min(rect.height() / height as f32);
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let total: f32 = extents
        .iter()
        .map(|(min, max)| (max[0] - min[0]) as f32 * scale)
        .sum::<f32>()
        + (gap_mm as f32 * scale) * gaps as f32;
    let mut left = rect.center().x - total / 2.0;
    let mut transforms = Vec::with_capacity(extents.len());
    for (min, max) in extents {
        transforms.push(Transform {
            origin: Pos2::new(
                left - min[0] as f32 * scale,
                rect.center().y + ((min[1] + max[1]) / 2.0) as f32 * scale,
            ),
            scale,
        });
        left += ((max[0] - min[0]) + gap_mm) as f32 * scale;
    }
    Some((scale, transforms))
}

/// Draws an arrowhead at `tip` pointing along `dir` (a unit vector): two 6-point strokes back
/// from the tip, 0.45 rad either side of the line.
pub(crate) fn arrowhead(painter: &egui::Painter, tip: Pos2, dir: Vec2, stroke: Stroke) {
    for side in [-1.0, 1.0] {
        let back = egui::emath::Rot2::from_angle(side * 0.45) * (-dir) * 6.0;
        painter.line_segment([tip, tip + back], stroke);
    }
}

/// Draws the geometry of the design shown (`inputs` and the `results` computed from them) and
/// returns what it drew.
pub fn geometry_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> GeometryLayout {
    let g = geometry(inputs, results);
    let row = ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y;
    let rows = g.end.callouts.len() + g.side.callouts.len() + g.notes.len();
    let list = rows as f32 * row * 1.5;
    let height = (ui.available_height() - list).max(MIN_DRAWING_HEIGHT);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
    let views: Vec<&View> = [&g.end, &g.side]
        .into_iter()
        .filter(|v| v.is_drawable())
        .collect();
    let mut layout = GeometryLayout {
        rect,
        scale: 0.0,
        end: None,
        side: None,
        dimensions: Vec::new(),
    };
    let extents: Vec<(Mm, Mm)> = views.iter().map(|v| (v.min, v.max)).collect();
    if let Some((scale, transforms)) = side_by_side(rect.shrink(MARGIN), &extents, VIEW_GAP_MM) {
        layout.scale = scale;
        for (view, transform) in views.into_iter().zip(transforms) {
            paint_view(ui, &painter, view, transform, &mut layout.dimensions);
            if std::ptr::eq(view, &g.end) {
                layout.end = Some(transform);
            } else {
                layout.side = Some(transform);
            }
        }
    }
    list_ui(ui, &g);
    layout
}

/// Paints one view's pieces, dashed lines and dimensions, and registers each dimension's hover
/// area.
fn paint_view(
    ui: &egui::Ui,
    painter: &egui::Painter,
    view: &View,
    t: Transform,
    dimensions: &mut Vec<(usize, Rect)>,
) {
    let visuals = ui.visuals();
    let centre = t.to_px([0.0, 0.0]);
    for piece in &view.pieces {
        let color = part_color(piece.part, visuals);
        match &piece.outline {
            Outline::Disc { r } => {
                painter.circle_filled(centre, t.len(*r), color);
            }
            Outline::Ring { r_in, r_out } => {
                let width = t.len(r_out - r_in);
                painter.circle_stroke(
                    centre,
                    t.len((r_in + r_out) / 2.0),
                    Stroke::new(width, color),
                );
            }
            Outline::Polygon(points) => {
                let points = points.iter().map(|p| t.to_px(*p)).collect();
                painter.add(egui::Shape::convex_polygon(points, color, Stroke::NONE));
            }
            Outline::Sector {
                r_in,
                r_out,
                from,
                to,
            } => {
                let r = (r_in + r_out) / 2.0;
                let points: Vec<Pos2> = (0..=16)
                    .map(|i| {
                        let a = from + (to - from) * f64::from(i) / 16.0;
                        t.to_px([r * a.cos(), r * a.sin()])
                    })
                    .collect();
                painter.add(egui::Shape::line(
                    points,
                    Stroke::new(t.len(r_out - r_in), color),
                ));
            }
            Outline::Rect { min, max } => {
                painter.rect_filled(Rect::from_two_pos(t.to_px(*min), t.to_px(*max)), 0.0, color);
            }
        }
    }
    for dashed in &view.dashed {
        let color = level_color(dashed.level, visuals.weak_text_color(), visuals);
        let points: Vec<Pos2> = dashed.points.iter().map(|p| t.to_px(*p)).collect();
        painter.extend(egui::Shape::dashed_line(
            &points,
            Stroke::new(1.0, color),
            6.0,
            4.0,
        ));
    }
    for callout in &view.callouts {
        if !(finite(callout.from) && finite(callout.to)) {
            continue;
        }
        let rect = paint_dimension(painter, callout, t, visuals);
        let response = ui.interact(
            rect,
            ui.id().with(("geometry_dimension", callout.tag)),
            Sense::hover(),
        );
        hover(response, callout);
        dimensions.push((callout.tag, rect));
    }
}

/// Paints a dimension line with its arrowheads and tag; returns its hover area.
fn paint_dimension(
    painter: &egui::Painter,
    callout: &Callout,
    t: Transform,
    visuals: &egui::Visuals,
) -> Rect {
    let color = level_color(callout.level, DIMENSION, visuals);
    let stroke = Stroke::new(1.5, color);
    let (a, b) = (t.to_px(callout.from), t.to_px(callout.to));
    painter.line_segment([a, b], stroke);
    let along = (b - a).normalized();
    if (b - a).length() >= 10.0 {
        for (tip, dir) in [(a, -along), (b, along)] {
            arrowhead(painter, tip, dir, stroke);
        }
    }
    let away = if along == Vec2::ZERO { Vec2::X } else { along };
    painter.text(
        b + away * 9.0,
        egui::Align2::CENTER_CENTER,
        callout.tag.to_string(),
        egui::FontId::proportional(12.0),
        color,
    );
    Rect::from_two_pos(a, b).expand(6.0)
}

/// Shows the hover text of the callout's result while `response` is hovered.
fn hover(response: egui::Response, callout: &Callout) {
    response.on_hover_ui(|ui| {
        ui.label(hover_text(callout.path).unwrap_or_else(|| callout.path.to_owned()));
    });
}

/// The callouts (tag and text, in their colours, with their hover text) and the notes.
fn list_ui(ui: &mut egui::Ui, g: &Geometry) {
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_geometry_list")
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for callout in g.callouts() {
                let color = callout.level.map(|l| l.color(ui.visuals()));
                let rich = |text: String| match color {
                    Some(c) => egui::RichText::new(text).color(c),
                    None => egui::RichText::new(text),
                };
                let response = ui
                    .horizontal(|ui| {
                        ui.label(rich(callout.tag.to_string()).strong());
                        ui.add(egui::Label::new(rich(callout.text.clone())).wrap())
                    })
                    .inner;
                hover(response, callout);
            }
            for note in &g.notes {
                let text = egui::RichText::new(&note.text);
                let text = match note.level {
                    Some(level) => text.color(level.color(ui.visuals())),
                    None => text.color(ui.visuals().weak_text_color()),
                };
                ui.add(egui::Label::new(text).wrap());
            }
        });
}

#[cfg(test)]
mod tests {
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
//!
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the dashboard on the right and
//! the centre region between them ([`CentreView`]: the results table; plan M4-2 adds the
//! geometry view and the plots). Every result is recomputed with every approved correction
//! on ([`compute_all`]) each frame after the inputs are drawn, so the readouts show this
//! frame's edits.
//!
````

with:

````rust
//!
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the dashboard on the right and
//! the centre region between them ([`CentreView`]: the geometry view first, the results table
//! last; decisions M42-1 and M42-2). Every result is recomputed with every approved correction
//! on ([`compute_all`]) each frame after the inputs are drawn, so the readouts show this
//! frame's edits.
//!
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::{Level, dashboard_ui};
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
````

with:

````rust

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::{Level, dashboard_ui, end_effect_banner};
use crate::gui::geometry_view::geometry_ui;
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    OpenDesign,
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
````

with:

````rust
    OpenDesign,
}

/// The views of the centre region, one tab row (decision M42-1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CentreView {
    /// The end view and the side view to scale, with the dimension callouts (the default view:
    /// decision M42-2).
    Geometry,
    /// Every result: label, value, unit, cell; searchable; CSV and JSON export.
    Results,
}

impl CentreView {
    /// Every view, in tab order.
    pub const ALL: [CentreView; 2] = [CentreView::Geometry, CentreView::Results];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
            CentreView::Geometry => "Geometry",
            CentreView::Results => "Results table",
        }
    }
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            design_widgets: Vec::new(),
            keyboard_shortcuts: true,
            last_error: None,
            centre: CentreView::Results,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
        }
````

with:

````rust
            design_widgets: Vec::new(),
            keyboard_shortcuts: true,
            last_error: None,
            centre: CentreView::Geometry,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
        }
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                .default_width(INPUTS_WIDTH)
                .show_inside(ui, |ui| self.inputs_ui(ui));
            self.run_sizing(ui);
            // After the inputs: the readouts show this frame's edits.
            self.results = compute_all(&self.shown_inputs());
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
````

with:

````rust
                .default_width(INPUTS_WIDTH)
                .show_inside(ui, |ui| self.inputs_ui(ui));
            self.run_sizing(ui);
            // After the inputs: the readouts show this frame's edits. One copy of the design
            // shown per frame, for the results and the centre region's views.
            let shown = self.shown_inputs();
            self.results = compute_all(&shown);
            egui::SidePanel::right("magcoupling_dashboard")
                .resizable(true)
                .default_width(DASHBOARD_WIDTH)
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| dashboard_ui(ui, &self.results));
                });
            egui::CentralPanel::default().show_inside(ui, |ui| self.centre_ui(ui));
        });
        // One undo step per settled edit.
        let settled = !self.editing(ui);
````

with:

````rust
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| dashboard_ui(ui, &self.results));
                });
            egui::CentralPanel::default().show_inside(ui, |ui| self.centre_ui(ui, &shown));
        });
        // One undo step per settled edit.
        let settled = !self.editing(ui);
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        }
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
````

with:

````rust
        }
    }

    /// The centre region: the view tabs (wrapping when the region is narrow), the end-effect
    /// banner when f_end <= 0 (over every view, decision M42-1), then the view of `shown`, the
    /// design shown.
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs) {
        ui.horizontal_wrapped(|ui| {
            for view in CentreView::ALL {
                ui.selectable_value(&mut self.centre, view, view.label());
            }
        });
        ui.separator();
        if let Some(banner) = end_effect_banner(&self.results) {
            ui.colored_label(ui.visuals().error_fg_color, banner);
        }
        match self.centre {
            CentreView::Geometry => {
                geometry_ui(ui, shown, &self.results);
            }
            CentreView::Results => {
                let action = self.results_table.ui(ui, &self.results);
                match action {
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                        // inputs with the free variable at the value shown.
                        contents: results_json(
                            &Design {
                                inputs: self.shown_inputs(),
                                sizing: self.sizing,
                            },
                            &self.results,
````

with:

````rust
                        // inputs with the free variable at the value shown.
                        contents: results_json(
                            &Design {
                                inputs: shown.clone(),
                                sizing: self.sizing,
                            },
                            &self.results,
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust

use crate::engine::meta::{ResultSet, Value, result_rows};
use crate::gui::corrections::{CorrectionIndex, marker_text};
use crate::gui::dashboard::{
    ResultInfo, ResultNotes, end_effect_banner, result_info, result_tooltip,
};
use crate::gui::format::{format_value, non_finite_text, with_unit};
use crate::gui::session::{Design, design_json, json_value};
use crate::{DesignInputs, DesignResults, compute_all};
````

with:

````rust

use crate::engine::meta::{ResultSet, Value, result_rows};
use crate::gui::corrections::{CorrectionIndex, marker_text};
use crate::gui::dashboard::{ResultInfo, hover_text, result_info};
use crate::gui::format::{format_value, non_finite_text, with_unit};
use crate::gui::session::{Design, design_json, json_value};
use crate::{DesignInputs, DesignResults, compute_all};
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
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
````

with:

````rust
        &self.query
    }

    /// Draws the table: the search box and the export buttons, then the rows on screen (the
    /// end-effect banner is the centre region's, over every view: decision M42-1). Returns an
    /// export asked for.
    pub fn ui(&mut self, ui: &mut egui::Ui, results: &DesignResults) -> Option<TableAction> {
        let entries = table_entries();
        let mut action = None;
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.query)
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
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
````

with:

````rust
    }
}

/// A row's hover text: the hover hook's ([`hover_text`]) with the exact value.
pub fn row_tooltip(entry: &TableEntry, value: &Value) -> String {
    let mut tooltip = hover_text(&entry.path).expect("every table row is a result");
    if let Value::Num(x) = value {
        tooltip.push_str(&format!("\nExact value: {}", exact_number(*x)));
    }
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib gui::geometry 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 19 passed; 0 failed` for `gui::geometry`, then `test result: ok. 330 passed; 0 failed` for the whole library (Task 0's 309 plus this plan's tests so far).

- [ ] **Step 5: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/geometry_view.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/dashboard.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/results_table.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/test_support.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 commit -F - <<'EOF'
feat(magcoupling-rs): the geometry view, the centre region's default tab

gui::geometry_view paints the end and side views at one scale (decision
M42-3), the dimension lines with their tags, the callout list in the callouts'
colours and the notes; hovering a dimension or its text shows its result's
hover text through dashboard::hover_text, which the results table's rows now
use too (the hook M4-3 joins). CentreView gains Geometry, the default view
(M42-2); the tab row wraps; the end-effect banner sits over the whole centre
region (M42-1); the design shown is bound once per frame for the results and
the centre region. side_by_side places views at one scale (none without
room) and arrowhead draws an arrow's head, for the clamp drawing too; the test
helpers walk one flat_shapes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m42 status --short
```

Expected: one new commit on `magcoupling/m4-2`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 3: The plot tabs

**Model:** `sonnet` for the implementation (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5); escalate a retry to the session model. **The per-task review runs on the session model** (omit `model`; `// session model: restated engine closed forms`): the task restates two engine closed forms in GUI code, and section 5's precedence rule gives a task on both lists the session model (see Global Constraints).

Five plots with egui_plot 0.33, the version linkage-sim-rs uses, so M5 embeds the panel with one egui_plot (spec M4 "Layout"'s plot tabs; decisions M42-5, M42-6). Each series builder is a pure function of this frame's results and three inputs of the design shown, with a test that pins it to the engine: the pull-out at 20 °C times `model::ring_pair_factor` (each ring's own alpha, decision A2-7) meets the pull-out at the operating temperature and its band edges the hot-low and cold-high torques; the torque-angle sum Σ amp_n sin(n x) times the area-lever product, f_end and the calibration factor meets the pull-out at the E7 angle; the slip-heating curve is the thermal network's first-order rise; the sweeps' markers are the tables' rows by status and end-effect range (the pole sweep's line named for its rows' smallest fitting apothem). Each plot has a constant id (`PlotKind::id`), so a test reads its transform from `egui_plot::PlotMemory`; a reference line (the required minimum, the operating temperature, the limit, the floor) is added only when its value is finite (`reference_hline`, `reference_vline`). `CentreView` gains `Plot(PlotKind)`, one tab per plot. Feature `gui` gains egui_plot; `Cargo.lock` takes linkage-sim-rs's 0.33.0, and gate 11's lock-parity list gains egui_plot.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/plots.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/mod.rs` (`pub mod plots;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs` (`CentreView::Plot`; a test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml` (egui_plot in feature `gui`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.lock` (by cargo: adds `egui_plot 0.33.0`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/linkage-sim-rs/scripts/gate.sh` (gate 11: egui_plot)

**Interfaces:**
- Consumes: `engine::model::{end_effect_in_range, harmonic_count, ring_pair_factor}` (`ring_pair_factor` is `pub(crate)`), `engine::sweeps::SweepRow`, `engine::meta::NumOrText`, `dashboard::{Level, end_effect_out_of_range}`.
- Produces (`magcoupling::gui::plots`): `enum PlotKind { TorqueTemperature, GapSweep, PoleSweep, SlipHeating, TorqueAngle }` with `ALL`, `const fn label(self) -> &'static str` and `fn id(self) -> egui::Id`; the id names `TORQUE_TEMPERATURE_ID`, `GAP_SWEEP_ID`, `POLE_SWEEP_ID`, `SLIP_HEATING_ID`, `TORQUE_ANGLE_ID`; `TEMPERATURE_SAMPLES = 121`, `PAST_THE_LIMIT_C = 10.0`, `HEATING_SAMPLES = 121`, `HEATING_SPAN_TAUS = 5.0`, `ANGLE_SAMPLES = 361`, `POINT_RADIUS = 4.0`; the legend names (`PULL_OUT`, `PULL_OUT_SMALLEST_APOTHEM`, `LOW_BAND`, `HIGH_BAND`, `REQUIRED`, `OPERATING`, `LIMIT`, `THIS_DESIGN`, `NOMINAL`, `BELOW_MINIMUM`, `NO_FIT`, `OUT_OF_RANGE`, `REQUIRED_FLOOR`, `ESTIMATE`, `HIGH_CASE`, `TIME_TO_LIMIT`, `PULL_OUT_POINT`) and `NOTHING_TO_PLOT`; `fn torque_at_temperature(&DesignResults, f64) -> f64`; `struct TorqueTemperature` and `fn torque_temperature(&DesignInputs, &DesignResults) -> TorqueTemperature`; `struct SweepPoints` and `fn sweep_points(&[SweepRow]) -> SweepPoints`; `struct SlipHeating` and `fn slip_heating(&DesignResults) -> SlipHeating`; `fn torque_at_angle(&DesignInputs, &DesignResults, f64) -> f64`; `struct TorqueAngle` and `fn torque_angle(&DesignInputs, &DesignResults) -> TorqueAngle`; `fn plot_ui(&mut egui::Ui, PlotKind, &DesignInputs, &DesignResults)`.
- Produces (`gui::panel`): `CentreView::Plot(PlotKind)`, `CentreView::ALL: [CentreView; 7]`.

- [ ] **Step 1: Write the failing tests**

The module's docs and tests, and the panel's test of the plot tabs. Besides the engine pins above, the tests count what each plot paints in a bare frame (a solid egui_plot `Line` of n points is one `Shape::Path` of n points, a `Points` marker a `Shape::Circle` of its radius): three temperature curves of 121 points, the gap sweep's 13-point line and 13 markers (and with short magnets 13 grey markers and no line), the pole sweep's 6 and 6 and its line's name, two heating curves of 121, the 361-point rotation curve and its pull-out point; and they check the legends, the greying when f_end <= 0, and values that are not numbers. The panel's test finds the design's marker, in the frame of an edit, at the edited pull-out through that frame's plot transform (`PlotMemory::load` with the plot's id), away from the old one.

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub mod input_ui;
pub mod inputs;
mod panel;
pub mod results_table;
pub mod session;
pub mod sizing;
````

with:

````rust
pub mod input_ui;
pub mod inputs;
mod panel;
pub mod plots;
pub mod results_table;
pub mod session;
pub mod sizing;
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        );
        harness.click_text(CentreView::Geometry.label());
        assert_eq!(harness.panel.centre, CentreView::Geometry);
    }

    #[test]
````

with:

````rust
        );
        harness.click_text(CentreView::Geometry.label());
        assert_eq!(harness.panel.centre, CentreView::Geometry);
    }

    #[test]
    fn each_plot_tab_draws_its_plot_from_this_frame_s_results() {
        use crate::gui::plots::{
            HIGH_CASE, POINT_RADIUS, PULL_OUT, PULL_OUT_POINT, PULL_OUT_SMALLEST_APOTHEM,
        };
        use crate::gui::test_support::flat_shapes;
        let mut harness = Harness::new();
        for kind in PlotKind::ALL {
            let legend = match kind {
                PlotKind::SlipHeating => HIGH_CASE,
                PlotKind::TorqueAngle => PULL_OUT_POINT,
                PlotKind::PoleSweep => PULL_OUT_SMALLEST_APOTHEM,
                _ => PULL_OUT,
            };
            let output = harness.click_text(kind.label());
            assert_eq!(harness.panel.centre, CentreView::Plot(kind));
            let output = [output, harness.frame(Vec::new())];
            assert!(
                output
                    .iter()
                    .any(|o| drawn_texts(o).iter().any(|t| t == legend)),
                "{kind:?} draws its legend"
            );
        }
        // The series come from this frame's results: in the edit's frame the design's marker
        // is painted at the new pull-out, where that frame's plot transform puts it.
        harness.click_text(PlotKind::TorqueTemperature.label());
        harness.focus(FACE_GAP);
        let before = harness.panel.results().model.pullout_Nm;
        let output = harness.frame(key_tap(egui::Key::ArrowRight));
        let after = harness.panel.results().model.pullout_Nm;
        assert_ne!(after, before);
        let memory = egui_plot::PlotMemory::load(&harness.ctx, PlotKind::TorqueTemperature.id())
            .expect("the plot ran this frame");
        let op = harness.panel.inputs().coupling.op_temp_C;
        let at = |torque: f64| {
            memory
                .transform()
                .position_from_point(&egui_plot::PlotPoint::new(op, torque))
        };
        assert!((at(after) - at(before)).length() > 0.5, "the edit moves it");
        let markers: Vec<egui::Pos2> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Circle(c) if c.radius == POINT_RADIUS + 1.0 => Some(c.center),
                _ => None,
            })
            .collect();
        assert!(
            markers.iter().any(|c| (*c - at(after)).length() < 1e-3),
            "{markers:?} vs {:?}",
            at(after)
        );
        assert!(markers.iter().all(|c| (*c - at(before)).length() > 0.5));
    }

    #[test]
````

Create `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/plots.rs`:

````rust
//! The plot tabs (spec M4 "Layout": "torque vs temperature (hot/cold allowance band,
//! requirement line); gap sweep; pole sweep; slip heating over time vs limit; torque vs rotation
//! angle (pull-out point)"), drawn with egui_plot (the 0.33 line linkage-sim-rs uses, locked to
//! its version: gate 11).
//!
//! Every series comes from the results computed this frame, plus three inputs of the design
//! shown (the minimum temperature, the variation allowance and the required minimum torque,
//! which the results do not repeat): a few hundred evaluations of closed forms, and only for the
//! tab shown. The closed forms are the engine's own (decision M42-5):
//!
//! - **Torque against temperature**: the pull-out at 20 °C times each ring's remanence factor,
//!   `model::ring_pair_factor` (Br_i(T) Br_o(T) / (Br_i Br_o), decision A2-7), the expression
//!   the Metal design and Temperature design torques use, so the curve passes through the
//!   pull-out at the operating temperature and its band edges through the hot-low and cold-high
//!   torques; from the minimum temperature to 10 °C past the governing limit.
//! - **Gap and pole sweeps**: the sweep tables' pull-out at the operating temperature against
//!   the swept variable, each row coloured by its status (the dashboard's colours: nominal
//!   green, below the hot minimum amber, a fit failure red) and greyed when its own f_end <= 0
//!   (`model::end_effect_in_range`), with the design shown as a marker. The pole sweep's rows
//!   take the smallest apothem that fits each count (the engine's `sweeps::pole_sweep`), so its
//!   line's legend says so: the design, at its own apothem, can sit off that line.
//! - **Slip heating**: the thermal network's first-order rise, start + rise (1 - exp(-t / tau)),
//!   for the estimate and the high case, against the governing limit, over five time constants
//!   (longer when the high case reaches the limit later), with the time to the limit marked.
//! - **Torque against rotation**: area-lever product × f_end × calibration factor ×
//!   Σ amp_n sin(n x) over the harmonics summed (`model.amp*_Pa`, the E7 torque-angle
//!   amplitudes), against the relative rotation in mechanical degrees over one pole pair
//!   (x = N/2 × the mechanical angle), with the pull-out point at the E7 angle.
//!
//! When f_end <= 0 (audit M9) every torque computed from the pull-out (the temperature and the
//! rotation plots) is drawn grey, as the dashboard greys those rows (decision M42-6).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::test_support::{drawn_texts, flat_shapes, short_magnets, sized_frame};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    #[test]
    fn the_torque_temperature_curve_meets_the_engine_s_torques() {
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let md = &r.metal;
        assert!(close(torque_at_temperature(&r, 50.0), r.model.pullout_Nm));
        assert!(close(
            torque_at_temperature(&r, 20.0),
            r.model.pullout_20C_Nm
        ));
        // The band edges: the hot-low torque at the operating temperature and the cold-high
        // torque at the minimum temperature (bit for bit: the engine's own expression).
        assert!(close(
            torque_at_temperature(&r, 50.0) * (1.0 - inputs.metal.variation),
            md.torque_hot_low_Nm
        ));
        assert_eq!(
            torque_at_temperature(&r, -40.0) * (1.0 + inputs.metal.variation),
            md.torque_cold_high_Nm
        );
        let s = torque_temperature(&inputs, &r);
        assert_eq!(s.nominal.len(), TEMPERATURE_SAMPLES);
        assert_eq!(s.low.len(), TEMPERATURE_SAMPLES);
        assert_eq!(s.high.len(), TEMPERATURE_SAMPLES);
        assert_eq!(s.nominal[0][0], -40.0);
        let limit = r.temperature.summary.governing_limit_C;
        assert_eq!(s.limit_C, Some(limit));
        assert!(close(
            s.nominal[TEMPERATURE_SAMPLES - 1][0],
            limit + PAST_THE_LIMIT_C
        ));
        assert_eq!(s.required_Nm, 2.5);
        assert_eq!(s.operating, [50.0, r.model.pullout_Nm]);
        assert!(!s.greyed);
        // Torque falls as the magnets warm (NdFeB's negative alpha).
        assert!(s.nominal.windows(2).all(|w| w[1][1] < w[0][1]));
    }

    #[test]
    fn a_grade_ring_s_own_alpha_shapes_the_curve() {
        // Decision A2-7: a grade-mode ring takes its grade's alpha; the curve still meets the
        // engine at the operating temperature and the band at the cold-high torque.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.grade_inner = "Y30".to_owned();
        let r = compute_all(&inputs);
        assert_ne!(r.model.inner_alpha_br_per_C, r.model.outer_alpha_br_per_C);
        assert!(close(torque_at_temperature(&r, 50.0), r.model.pullout_Nm));
        assert_eq!(
            torque_at_temperature(&r, inputs.metal.min_temp_C) * (1.0 + inputs.metal.variation),
            r.metal.torque_cold_high_Nm
        );
    }

    #[test]
    fn sweep_rows_are_sorted_by_status_and_end_effect_range() {
        let r = compute_all(&DesignInputs::default());
        let gap = sweep_points(&r.gap_sweep);
        assert_eq!(
            (gap.nominal.len(), gap.below_minimum.len(), gap.no_fit.len()),
            (4, 2, 7)
        );
        assert!(gap.out_of_range.is_empty());
        assert_eq!(gap.line.len(), 13);
        assert_eq!(gap.line[0], [0.5, r.gap_sweep[0].pullout_op_Nm]);
        let pole = sweep_points(&r.pole_sweep);
        assert_eq!(
            (
                pole.nominal.len(),
                pole.below_minimum.len(),
                pole.no_fit.len()
            ),
            (1, 1, 4)
        );
        // Short magnets put every row outside the end-effect model: all greyed, no line.
        let short = compute_all(&short_magnets());
        let greyed = sweep_points(&short.gap_sweep);
        assert_eq!(greyed.out_of_range.len(), 13);
        assert!(greyed.line.is_empty() && greyed.nominal.is_empty());
    }

    #[test]
    fn slip_heating_rises_to_the_steady_temperatures_and_marks_the_limit() {
        let r = compute_all(&DesignInputs::default());
        let th = &r.temperature.thermal;
        let s = slip_heating(&r);
        assert_eq!(s.estimate.len(), HEATING_SAMPLES);
        assert_eq!(s.high.len(), HEATING_SAMPLES);
        assert_eq!(s.high[0], [0.0, th.start_C]);
        let end = s.high[HEATING_SAMPLES - 1];
        assert!(close(end[0], HEATING_SPAN_TAUS * th.time_constant_s));
        let rise = th.steady_rise_high_C * (1.0 - (-HEATING_SPAN_TAUS).exp());
        assert!((end[1] - (th.start_C + rise)).abs() < 1e-9);
        // The default high case stays below the limit ("never").
        assert_eq!(s.time_to_limit, None);
        // A hotter start reaches it: the marker sits on the limit, on the curve.
        let mut inputs = DesignInputs::default();
        inputs.temperature.duty.hot_ambient_C += 5.0;
        let hot = compute_all(&inputs);
        let s = slip_heating(&hot);
        let [t, limit] = s.time_to_limit.expect("the high case reaches the limit");
        assert_eq!(limit, hot.temperature.summary.governing_limit_C);
        let th = &hot.temperature.thermal;
        let at = th.start_C + th.steady_rise_high_C * (1.0 - (-t / th.time_constant_s).exp());
        assert!((at - limit).abs() < 1e-9, "{at} vs {limit}");
        assert!(
            s.high.last().unwrap()[0] >= t,
            "the axis reaches the marker"
        );
    }

    #[test]
    fn the_torque_rotation_curve_peaks_at_the_pull_out() {
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let x = r.model.pullout_angle_rad;
        assert!(close(torque_at_angle(&inputs, &r, x), r.model.pullout_Nm));
        let s = torque_angle(&inputs, &r);
        assert_eq!(s.curve.len(), ANGLE_SAMPLES);
        // One pole pair of 10 poles: 72 mechanical degrees; the pull-out at a quarter of it.
        assert_eq!(s.curve[ANGLE_SAMPLES - 1][0], 72.0);
        assert!(close(s.pull_out[0], 18.0));
        assert_eq!(s.pull_out[1], r.model.pullout_Nm);
        let max = s.curve.iter().map(|p| p[1]).fold(f64::MIN, f64::max);
        assert!(max <= r.model.pullout_Nm * (1.0 + 1e-12));
        // A peak off half a pitch (E7): every harmonic up to 11 at the default design.
        let mut eleven = DesignInputs::default();
        eleven.coupling.max_harmonic = 11;
        let r11 = compute_all(&eleven);
        assert!(close(
            torque_at_angle(&eleven, &r11, r11.model.pullout_angle_rad),
            r11.model.pullout_Nm
        ));
        // A harmonic set outside its choices: no curve, never another set.
        let mut bad = DesignInputs::default();
        bad.coupling.max_harmonic = 4;
        assert!(torque_angle(&bad, &compute_all(&bad)).curve.is_empty());
    }

    #[test]
    fn out_of_range_end_effect_greys_the_torque_plots() {
        let inputs = short_magnets();
        let r = compute_all(&inputs);
        assert!(torque_temperature(&inputs, &r).greyed);
        assert!(torque_angle(&inputs, &r).greyed);
        assert!(
            !torque_temperature(
                &DesignInputs::default(),
                &compute_all(&DesignInputs::default())
            )
            .greyed
        );
    }

    /// The point count of every path egui painted, and the radius of every circle.
    fn paths_and_circles(output: &egui::FullOutput) -> (Vec<usize>, Vec<f32>) {
        let shapes = flat_shapes(output);
        let paths = shapes
            .iter()
            .filter_map(|shape| match shape {
                egui::Shape::Path(path) => Some(path.points.len()),
                _ => None,
            })
            .collect();
        let circles = shapes
            .iter()
            .filter_map(|shape| match shape {
                egui::Shape::Circle(circle) => Some(circle.radius),
                _ => None,
            })
            .collect();
        (paths, circles)
    }

    /// Two frames of `kind` for `inputs` (a plot settles its bounds on the first).
    fn draw(kind: PlotKind, inputs: &DesignInputs) -> egui::FullOutput {
        let ctx = egui::Context::default();
        let results = compute_all(inputs);
        let mut output = None;
        for _ in 0..2 {
            output = Some(sized_frame(
                &ctx,
                egui::vec2(900.0, 600.0),
                Vec::new(),
                |ui| plot_ui(ui, kind, inputs, &results),
            ));
        }
        output.unwrap()
    }

    fn count(values: &[usize], want: usize) -> usize {
        values.iter().filter(|v| **v == want).count()
    }

    #[test]
    fn each_plot_draws_its_series_with_their_point_counts() {
        let inputs = DesignInputs::default();
        let at = |radius: f32, circles: &[f32]| circles.iter().filter(|r| **r == radius).count();
        let output = draw(PlotKind::TorqueTemperature, &inputs);
        let (paths, _) = paths_and_circles(&output);
        assert_eq!(
            count(&paths, TEMPERATURE_SAMPLES),
            3,
            "nominal and both band edges"
        );
        let texts = drawn_texts(&output);
        for name in [
            PULL_OUT,
            LOW_BAND,
            HIGH_BAND,
            REQUIRED,
            OPERATING,
            LIMIT,
            THIS_DESIGN,
        ] {
            assert!(
                texts.iter().any(|t| t == name),
                "legend {name:?} in {texts:?}"
            );
        }
        let (paths, circles) = paths_and_circles(&draw(PlotKind::GapSweep, &inputs));
        assert_eq!(count(&paths, 13), 1, "the line through the 13 rows");
        assert_eq!(at(POINT_RADIUS, &circles), 13, "a marker per row");
        // Short magnets put every row outside the end-effect model: a grey marker per row and
        // no line.
        let (paths, circles) = paths_and_circles(&draw(PlotKind::GapSweep, &short_magnets()));
        assert_eq!(at(POINT_RADIUS, &circles), 13);
        assert_eq!(count(&paths, 13), 0);
        let output = draw(PlotKind::PoleSweep, &inputs);
        let (paths, circles) = paths_and_circles(&output);
        assert_eq!(count(&paths, 6), 1);
        assert_eq!(at(POINT_RADIUS, &circles), 6);
        assert!(
            drawn_texts(&output)
                .iter()
                .any(|t| t == PULL_OUT_SMALLEST_APOTHEM)
        );
        let output = draw(PlotKind::SlipHeating, &inputs);
        let (paths, _) = paths_and_circles(&output);
        assert_eq!(count(&paths, HEATING_SAMPLES), 2, "estimate and high case");
        let output = draw(PlotKind::TorqueAngle, &inputs);
        let (paths, circles) = paths_and_circles(&output);
        assert_eq!(count(&paths, ANGLE_SAMPLES), 1);
        assert_eq!(at(POINT_RADIUS + 1.0, &circles), 1, "the pull-out point");
        assert!(drawn_texts(&output).iter().any(|t| t == PULL_OUT_POINT));
    }

    #[test]
    fn plots_of_values_that_are_not_numbers_draw_without_panicking() {
        // validate() refuses these at every boundary; a struct literal can still hold them.
        let mut inputs = DesignInputs::default();
        inputs.coupling.op_temp_C = f64::NAN;
        inputs.temperature.thermal.conductance_W_K = f64::NAN;
        inputs.coupling.max_harmonic = 4;
        for kind in PlotKind::ALL {
            draw(kind, &inputs);
        }
        let texts = drawn_texts(&draw(PlotKind::TorqueAngle, &inputs));
        assert!(texts.iter().any(|t| t == NOTHING_TO_PLOT));
    }
}
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib gui:: 2>&1 | grep -E "^error" | sort | uniq -c
```

Expected: FAIL to compile. The build stops with `error: could not compile `magcoupling-rs` (lib test) due to 83 previous errors; 1 warning emitted`; the errors name the items Step 3 adds, among them `error[E0412]: cannot find type `PlotKind` in this scope`, `error[E0425]: cannot find function `plot_ui` in this scope`, `error[E0425]: cannot find function `torque_at_temperature` in this scope`, `error[E0433]: failed to resolve: use of unresolved module or unlinked crate `egui_plot`` and `error[E0599]: no variant or associated item named `Plot` found for enum `gui::panel::CentreView` in the current scope`.

- [ ] **Step 3: Write the implementation**

The dependency, gate 11's list, the module's code between its docs and its tests, and the panel's plot tabs.

In `C:/Users/Cole/source/repos/lsim-mag-m42/linkage-sim-rs/scripts/gate.sh`, replace:

````bash
# app): their tests, clippy on native and wasm32, the guard that no shipped
# build has the test-only workbook-parity feature (with a negative control
# that proves the guard trips), and the check that both crates lock the same
# egui, eframe and wasm-bindgen (the CLI version deploy-web.yml installs).
# Gate 12: the vendored Python oracle, reference/magcoupling-py: its parity
# suite, and a check that the committed differential test data is current.
# Gate 12 needs a Python with the oracle's dependencies; see oracle_python
````

with:

````bash
# app): their tests, clippy on native and wasm32, the guard that no shipped
# build has the test-only workbook-parity feature (with a negative control
# that proves the guard trips), and the check that both crates lock the same
# egui, egui_plot, eframe and wasm-bindgen (the CLI version deploy-web.yml
# installs).
# Gate 12: the vendored Python oracle, reference/magcoupling-py: its parity
# suite, and a check that the committed differential test data is current.
# Gate 12 needs a Python with the oracle's dependencies; see oracle_python
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/linkage-sim-rs/scripts/gate.sh`, replace:

````bash
fi
echo "negative control: the guard trips when workbook-parity is forced on"

echo "== gate 11/12: lock parity (egui, eframe, wasm-bindgen; wasm-bindgen-cli pin in deploy-web.yml) =="
LINKAGE_LOCK="Cargo.lock"
MAGCOUPLING_LOCK="$REPO_ROOT/magcoupling-rs/Cargo.lock"
for pkg in egui eframe wasm-bindgen; do
  linkage_version="$(lock_versions "$LINKAGE_LOCK" "$pkg")"
  magcoupling_version="$(lock_versions "$MAGCOUPLING_LOCK" "$pkg")"
  if [[ -z "$linkage_version" || "$linkage_version" != "$magcoupling_version" ]]; then
````

with:

````bash
fi
echo "negative control: the guard trips when workbook-parity is forced on"

echo "== gate 11/12: lock parity (egui, egui_plot, eframe, wasm-bindgen; wasm-bindgen-cli pin in deploy-web.yml) =="
LINKAGE_LOCK="Cargo.lock"
MAGCOUPLING_LOCK="$REPO_ROOT/magcoupling-rs/Cargo.lock"
for pkg in egui egui_plot eframe wasm-bindgen; do
  linkage_version="$(lock_versions "$LINKAGE_LOCK" "$pkg")"
  magcoupling_version="$(lock_versions "$MAGCOUPLING_LOCK" "$pkg")"
  if [[ -z "$linkage_version" || "$linkage_version" != "$magcoupling_version" ]]; then
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml`, replace:

````toml
# standalone app, M4, and a window in linkage-sim-rs, M5). egui only: no
# windowing, no eframe, no file dialogs. serde_json, base64 and flate2 write
# and read design files, share links and the results export; log reports the
# sizing outcome (the host chooses the logger: the browser console on the web).
gui = ["dep:egui", "dep:log", "dep:serde_json", "dep:base64", "dep:flate2"]
# The standalone app: `app::MagcouplingApp` and the binaries `magcoupling-app`
# (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen).
# rfd: the native save and open dialogs, and the web file picker.
````

with:

````toml
# standalone app, M4, and a window in linkage-sim-rs, M5). egui only: no
# windowing, no eframe, no file dialogs. serde_json, base64 and flate2 write
# and read design files, share links and the results export; log reports the
# sizing outcome (the host chooses the logger: the browser console on the web);
# egui_plot draws the plot tabs.
gui = [
    "dep:egui",
    "dep:egui_plot",
    "dep:log",
    "dep:serde_json",
    "dep:base64",
    "dep:flate2",
]
# The standalone app: `app::MagcouplingApp` and the binaries `magcoupling-app`
# (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen).
# rfd: the native save and open dialogs, and the web file picker.
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml`, replace:

````toml
# the exact version (0.32.3, as in linkage-sim-rs/Cargo.lock) is pinned in
# Cargo.lock, and gate.sh checks that both lock files agree.
egui = { version = "0.32", optional = true }
eframe = { version = "0.32", optional = true }
log = { version = "0.4", optional = true }
# Design files, share links and the results export (feature gui). The same
````

with:

````toml
# the exact version (0.32.3, as in linkage-sim-rs/Cargo.lock) is pinned in
# Cargo.lock, and gate.sh checks that both lock files agree.
egui = { version = "0.32", optional = true }
# The plot tabs (feature gui): the 0.33 line linkage-sim-rs uses, built on egui
# 0.32; Cargo.lock pins linkage-sim-rs's 0.33.0 and gate 11 checks it.
egui_plot = { version = "0.33", optional = true }
eframe = { version = "0.32", optional = true }
log = { version = "0.4", optional = true }
# Design files, share links and the results export (feature gui). The same
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
````

with:

````rust
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    /// The end view and the side view to scale, with the dimension callouts (the default view:
    /// decision M42-2).
    Geometry,
    /// Every result: label, value, unit, cell; searchable; CSV and JSON export.
    Results,
}

impl CentreView {
    /// Every view, in tab order.
    pub const ALL: [CentreView; 2] = [CentreView::Geometry, CentreView::Results];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
            CentreView::Geometry => "Geometry",
            CentreView::Results => "Results table",
        }
    }
````

with:

````rust
    /// The end view and the side view to scale, with the dimension callouts (the default view:
    /// decision M42-2).
    Geometry,
    /// One of the plots (egui_plot).
    Plot(PlotKind),
    /// Every result: label, value, unit, cell; searchable; CSV and JSON export.
    Results,
}

impl CentreView {
    /// Every view, in tab order.
    pub const ALL: [CentreView; 7] = [
        CentreView::Geometry,
        CentreView::Plot(PlotKind::TorqueTemperature),
        CentreView::Plot(PlotKind::GapSweep),
        CentreView::Plot(PlotKind::PoleSweep),
        CentreView::Plot(PlotKind::SlipHeating),
        CentreView::Plot(PlotKind::TorqueAngle),
        CentreView::Results,
    ];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
            CentreView::Geometry => "Geometry",
            CentreView::Plot(kind) => kind.label(),
            CentreView::Results => "Results table",
        }
    }
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            CentreView::Geometry => {
                geometry_ui(ui, shown, &self.results);
            }
            CentreView::Results => {
                let action = self.results_table.ui(ui, &self.results);
                match action {
````

with:

````rust
            CentreView::Geometry => {
                geometry_ui(ui, shown, &self.results);
            }
            CentreView::Plot(kind) => plot_ui(ui, kind, shown, &self.results),
            CentreView::Results => {
                let action = self.results_table.ui(ui, &self.results);
                match action {
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/plots.rs`, replace:

````rust

#[cfg(test)]
mod tests {
````

with:

````rust

use egui::Color32;
use egui_plot::{Corner, HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, Points, VLine};

use crate::engine::meta::NumOrText;
use crate::engine::model::{end_effect_in_range, harmonic_count, ring_pair_factor};
use crate::engine::sweeps::SweepRow;
use crate::gui::dashboard::{Level, end_effect_out_of_range};
use crate::{DesignInputs, DesignResults};

/// Samples of the torque-temperature curve.
pub const TEMPERATURE_SAMPLES: usize = 121;

/// How far past the governing limit the temperature axis runs [°C].
pub const PAST_THE_LIMIT_C: f64 = 10.0;

/// Samples of each slip-heating curve.
pub const HEATING_SAMPLES: usize = 121;

/// The slip-heating time axis, in thermal time constants (at least).
pub const HEATING_SPAN_TAUS: f64 = 5.0;

/// Samples of the torque-rotation curve.
pub const ANGLE_SAMPLES: usize = 361;

/// The radius of a sweep row's marker [points].
pub const POINT_RADIUS: f32 = 4.0;

/// The plots, one tab each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlotKind {
    TorqueTemperature,
    GapSweep,
    PoleSweep,
    SlipHeating,
    TorqueAngle,
}

impl PlotKind {
    /// Every plot, in tab order.
    pub const ALL: [PlotKind; 5] = [
        PlotKind::TorqueTemperature,
        PlotKind::GapSweep,
        PlotKind::PoleSweep,
        PlotKind::SlipHeating,
        PlotKind::TorqueAngle,
    ];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
            PlotKind::TorqueTemperature => "Torque vs temperature",
            PlotKind::GapSweep => "Gap sweep",
            PlotKind::PoleSweep => "Pole sweep",
            PlotKind::SlipHeating => "Slip heating",
            PlotKind::TorqueAngle => "Torque vs rotation",
        }
    }

    /// The plot's id: egui_plot keeps its bounds and transform under it
    /// (`egui_plot::PlotMemory::load`).
    pub fn id(self) -> egui::Id {
        egui::Id::new(match self {
            PlotKind::TorqueTemperature => TORQUE_TEMPERATURE_ID,
            PlotKind::GapSweep => GAP_SWEEP_ID,
            PlotKind::PoleSweep => POLE_SWEEP_ID,
            PlotKind::SlipHeating => SLIP_HEATING_ID,
            PlotKind::TorqueAngle => TORQUE_ANGLE_ID,
        })
    }
}

/// The plots' id names ([`PlotKind::id`]).
pub const TORQUE_TEMPERATURE_ID: &str = "magcoupling_plot_torque_temperature";
pub const GAP_SWEEP_ID: &str = "magcoupling_plot_gap_sweep";
pub const POLE_SWEEP_ID: &str = "magcoupling_plot_pole_sweep";
pub const SLIP_HEATING_ID: &str = "magcoupling_plot_slip_heating";
pub const TORQUE_ANGLE_ID: &str = "magcoupling_plot_torque_angle";

/// The legend names.
pub const PULL_OUT: &str = "Pull-out torque";
/// The pole sweep's line: its rows are not at the design's apothem.
pub const PULL_OUT_SMALLEST_APOTHEM: &str =
    "Pull-out torque (rows at their smallest fitting apothem)";
pub const LOW_BAND: &str = "Low: minus the variation allowance";
pub const HIGH_BAND: &str = "High: plus the variation allowance";
pub const REQUIRED: &str = "Required minimum";
pub const OPERATING: &str = "Operating temperature";
pub const LIMIT: &str = "Governing limit";
pub const THIS_DESIGN: &str = "This design";
pub const NOMINAL: &str = "Nominal: test needed";
pub const BELOW_MINIMUM: &str = "Below hot minimum";
pub const NO_FIT: &str = "Does not fit";
pub const OUT_OF_RANGE: &str = "End-effect model out of range";
pub const REQUIRED_FLOOR: &str = "Required floor";
pub const ESTIMATE: &str = "Estimate";
pub const HIGH_CASE: &str = "High case";
pub const TIME_TO_LIMIT: &str = "Time to the limit (high case)";
pub const PULL_OUT_POINT: &str = "Pull-out point";

/// The text shown instead of a plot with nothing finite to draw.
pub const NOTHING_TO_PLOT: &str = "Nothing to plot: the values are not numbers";

/// `n` evenly spaced values from `lo` to `hi`, both included.
fn samples(lo: f64, hi: f64, n: usize) -> impl Iterator<Item = f64> {
    (0..n).map(move |i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
}

/// `points` without those that are not finite (egui_plot's bounds need finite values).
fn finite(points: impl IntoIterator<Item = [f64; 2]>) -> Vec<[f64; 2]> {
    points
        .into_iter()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect()
}

/// The pull-out torque at magnet temperature `temp_C` [N·m]: the pull-out at 20 °C times each
/// ring's remanence factor (`model::ring_pair_factor`).
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub fn torque_at_temperature(results: &DesignResults, temp_C: f64) -> f64 {
    let m = &results.model;
    m.pullout_20C_Nm * ring_pair_factor(m.inner_alpha_br_per_C, m.outer_alpha_br_per_C, temp_C)
}

/// The torque-temperature plot's series.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix
pub struct TorqueTemperature {
    pub nominal: Vec<[f64; 2]>,
    pub low: Vec<[f64; 2]>,
    pub high: Vec<[f64; 2]>,
    pub required_Nm: f64,
    pub operating_C: f64,
    /// The governing limit, when finite.
    pub limit_C: Option<f64>,
    /// The pull-out at the operating temperature.
    pub operating: [f64; 2],
    /// f_end <= 0: the torques are greyed.
    pub greyed: bool,
}

/// The torque-temperature series of the design shown.
pub fn torque_temperature(inputs: &DesignInputs, results: &DesignResults) -> TorqueTemperature {
    let md = &inputs.metal;
    let op = inputs.coupling.op_temp_C;
    let limit = results.temperature.summary.governing_limit_C;
    let finite_limit = limit.is_finite().then_some(limit);
    let top = op.max(finite_limit.unwrap_or(op)) + PAST_THE_LIMIT_C;
    let temps: Vec<f64> = if top > md.min_temp_C {
        samples(md.min_temp_C, top, TEMPERATURE_SAMPLES).collect()
    } else {
        Vec::new()
    };
    let curve = |factor: f64| {
        finite(
            temps
                .iter()
                .map(|&t| [t, torque_at_temperature(results, t) * factor]),
        )
    };
    TorqueTemperature {
        nominal: curve(1.0),
        low: curve(1.0 - md.variation),
        high: curve(1.0 + md.variation),
        required_Nm: md.required_min_Nm,
        operating_C: op,
        limit_C: finite_limit,
        operating: [op, results.model.pullout_Nm],
        greyed: end_effect_out_of_range(results).is_some(),
    }
}

/// A sweep's rows as markers, by status, and the line through the rows in the end-effect
/// model's range.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SweepPoints {
    /// "nominal: test needed".
    pub nominal: Vec<[f64; 2]>,
    /// "below hot minimum".
    pub below_minimum: Vec<[f64; 2]>,
    /// A flat too narrow or outside the OD envelope.
    pub no_fit: Vec<[f64; 2]>,
    /// f_end <= 0, whatever the status.
    pub out_of_range: Vec<[f64; 2]>,
    pub line: Vec<[f64; 2]>,
}

/// The markers of a sweep's rows: (swept variable, pull-out at the operating temperature).
pub fn sweep_points(rows: &[SweepRow]) -> SweepPoints {
    let mut points = SweepPoints::default();
    for row in rows {
        let p = [row.variable, row.pullout_op_Nm];
        if !(p[0].is_finite() && p[1].is_finite()) {
            continue;
        }
        if !end_effect_in_range(row.f_end) {
            points.out_of_range.push(p);
            continue;
        }
        points.line.push(p);
        match row.status.as_str() {
            "nominal: test needed" => points.nominal.push(p),
            "below hot minimum" => points.below_minimum.push(p),
            _ => points.no_fit.push(p),
        }
    }
    points
}

/// The slip-heating plot's series.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix
pub struct SlipHeating {
    pub estimate: Vec<[f64; 2]>,
    pub high: Vec<[f64; 2]>,
    pub limit_C: f64,
    /// (time to the limit, the limit) in the high case, when it reaches it.
    pub time_to_limit: Option<[f64; 2]>,
}

/// The slip-heating series: continuous slip from the starting temperature.
pub fn slip_heating(results: &DesignResults) -> SlipHeating {
    let th = &results.temperature.thermal;
    let limit = results.temperature.summary.governing_limit_C;
    let tau = th.time_constant_s;
    let reached = match th.time_to_limit_high {
        NumOrText::Num(t) if t.is_finite() => Some(t),
        _ => None,
    };
    let span = (HEATING_SPAN_TAUS * tau).max(reached.map_or(0.0, |t| 1.1 * t));
    let times: Vec<f64> = if tau.is_finite() && tau > 0.0 && span.is_finite() {
        samples(0.0, span, HEATING_SAMPLES).collect()
    } else {
        Vec::new()
    };
    let curve = |rise: f64| {
        finite(
            times
                .iter()
                .map(|&t| [t, th.start_C + rise * (1.0 - (-t / tau).exp())]),
        )
    };
    SlipHeating {
        estimate: curve(th.steady_rise_est_C),
        high: curve(th.steady_rise_high_C),
        limit_C: limit,
        time_to_limit: reached.map(|t| [t, limit]),
    }
}

/// The torque-rotation plot's series.
#[derive(Clone, Debug, PartialEq)]
pub struct TorqueAngle {
    /// (relative rotation [° mechanical], torque [N·m]) over one pole pair.
    pub curve: Vec<[f64; 2]>,
    /// The pull-out point: the E7 angle and the pull-out torque.
    pub pull_out: [f64; 2],
    /// f_end <= 0: the torques are greyed.
    pub greyed: bool,
}

/// The torque at electrical angle `x` [rad] of the design shown [N·m]: area-lever product ×
/// f_end × calibration factor × Σ amp_n sin(n x) over the harmonics summed; NaN for a harmonic
/// set outside its choices.
pub fn torque_at_angle(inputs: &DesignInputs, results: &DesignResults, x: f64) -> f64 {
    let m = &results.model;
    let amps = [
        m.amp1_Pa, m.amp3_Pa, m.amp5_Pa, m.amp7_Pa, m.amp9_Pa, m.amp11_Pa,
    ];
    let Some(count) = harmonic_count(inputs.coupling.max_harmonic) else {
        return f64::NAN;
    };
    let tau = amps[..count]
        .iter()
        .zip([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
        .fold(0.0, |acc, (amp, n)| acc + amp * (n * x).sin());
    tau * m.area_lever_m3 * m.f_end * m.f_cal
}

/// The torque-rotation series of the design shown.
pub fn torque_angle(inputs: &DesignInputs, results: &DesignResults) -> TorqueAngle {
    let pairs = inputs.coupling.npole as f64 / 2.0;
    let pole_pair_deg = 360.0 / pairs;
    let curve = if pole_pair_deg.is_finite() && pole_pair_deg > 0.0 {
        finite(samples(0.0, pole_pair_deg, ANGLE_SAMPLES).map(|deg| {
            let x = (deg * pairs).to_radians();
            [deg, torque_at_angle(inputs, results, x)]
        }))
    } else {
        Vec::new()
    };
    TorqueAngle {
        curve,
        pull_out: [
            (results.model.pullout_angle_rad / pairs).to_degrees(),
            results.model.pullout_Nm,
        ],
        greyed: end_effect_out_of_range(results).is_some(),
    }
}

/// The colour of a torque series: `color`, or grey when the torques are greyed.
fn tint(color: Color32, greyed: bool, visuals: &egui::Visuals) -> Color32 {
    if greyed {
        visuals.weak_text_color()
    } else {
        color
    }
}

const BLUE: Color32 = Color32::from_rgb(100, 180, 255);
const AMBER: Color32 = Color32::from_rgb(230, 170, 70);
const VIOLET: Color32 = Color32::from_rgb(180, 140, 230);

/// A plot with its id ([`PlotKind::id`]), the panel's legend and axis labels, filling the
/// space left.
fn plot(ui: &egui::Ui, kind: PlotKind, x: &str, y: &str) -> Plot<'static> {
    Plot::new(kind.id())
        .id(kind.id())
        .legend(Legend::default().position(Corner::RightTop))
        .x_axis_label(x.to_owned())
        .y_axis_label(y.to_owned())
        .height(ui.available_height().max(150.0))
}

/// Draws the plot `kind` of the design shown (`inputs`, and the `results` computed from them).
pub fn plot_ui(ui: &mut egui::Ui, kind: PlotKind, inputs: &DesignInputs, results: &DesignResults) {
    match kind {
        PlotKind::TorqueTemperature => torque_temperature_ui(ui, inputs, results),
        PlotKind::GapSweep => sweep_ui(
            ui,
            kind,
            "Corner gap [mm]",
            PULL_OUT,
            &results.gap_sweep,
            [results.model.corner_gap_mm, results.model.pullout_Nm],
            results.model.required_floor_Nm,
        ),
        PlotKind::PoleSweep => sweep_ui(
            ui,
            kind,
            "Poles per ring",
            PULL_OUT_SMALLEST_APOTHEM,
            &results.pole_sweep,
            [inputs.coupling.npole as f64, results.model.pullout_Nm],
            results.model.required_floor_Nm,
        ),
        PlotKind::SlipHeating => slip_heating_ui(ui, results),
        PlotKind::TorqueAngle => torque_angle_ui(ui, inputs, results),
    }
}

/// Adds a dashed horizontal reference line `name` at `y`, only when `y` is finite (egui_plot's
/// bounds need finite values; a value that is not a number has no place on the axis).
fn reference_hline(p: &mut PlotUi<'_>, name: &str, y: f64, color: Color32) {
    if y.is_finite() {
        p.hline(
            HLine::new(name, y)
                .color(color)
                .style(LineStyle::Dashed { length: 6.0 }),
        );
    }
}

/// Adds a dashed vertical reference line `name` at `x`, only when `x` is finite.
fn reference_vline(p: &mut PlotUi<'_>, name: &str, x: f64, color: Color32) {
    if x.is_finite() {
        p.vline(
            VLine::new(name, x)
                .color(color)
                .style(LineStyle::Dashed { length: 4.0 }),
        );
    }
}

fn torque_temperature_ui(ui: &mut egui::Ui, inputs: &DesignInputs, results: &DesignResults) {
    let s = torque_temperature(inputs, results);
    if s.nominal.is_empty() {
        ui.weak(NOTHING_TO_PLOT);
        return;
    }
    let visuals = ui.visuals().clone();
    let band = tint(AMBER, s.greyed, &visuals);
    plot(
        ui,
        PlotKind::TorqueTemperature,
        "Magnet temperature [°C]",
        "Torque [N·m]",
    )
    .show(ui, |p| {
        p.line(
            Line::new(PULL_OUT, s.nominal)
                .color(tint(BLUE, s.greyed, &visuals))
                .width(2.0),
        );
        p.line(Line::new(LOW_BAND, s.low).color(band).width(1.0));
        p.line(Line::new(HIGH_BAND, s.high).color(band).width(1.0));
        reference_hline(p, REQUIRED, s.required_Nm, Level::Bad.color(&visuals));
        reference_vline(p, OPERATING, s.operating_C, visuals.weak_text_color());
        if let Some(limit) = s.limit_C {
            reference_vline(p, LIMIT, limit, VIOLET);
        }
        let operating = finite([s.operating]);
        if !operating.is_empty() {
            p.points(
                Points::new(THIS_DESIGN, PlotPoints::from(operating))
                    .radius(POINT_RADIUS + 1.0)
                    .color(tint(BLUE, s.greyed, &visuals)),
            );
        }
    });
}

/// A sweep's plot: `line` names the line through the rows in the end-effect model's range.
#[allow(non_snake_case)] // unit suffix
fn sweep_ui(
    ui: &mut egui::Ui,
    kind: PlotKind,
    x: &str,
    line: &str,
    rows: &[SweepRow],
    design: [f64; 2],
    floor_Nm: f64,
) {
    let s = sweep_points(rows);
    let visuals = ui.visuals().clone();
    plot(ui, kind, x, "Pull-out at the operating temperature [N·m]").show(ui, |p| {
        p.line(Line::new(line, s.line).color(BLUE).width(1.5));
        for (name, points, color) in [
            (NOMINAL, s.nominal, Level::Good.color(&visuals)),
            (
                BELOW_MINIMUM,
                s.below_minimum,
                Level::Caution.color(&visuals),
            ),
            (NO_FIT, s.no_fit, Level::Bad.color(&visuals)),
            (OUT_OF_RANGE, s.out_of_range, visuals.weak_text_color()),
        ] {
            if !points.is_empty() {
                p.points(Points::new(name, points).radius(POINT_RADIUS).color(color));
            }
        }
        reference_hline(p, REQUIRED_FLOOR, floor_Nm, Level::Bad.color(&visuals));
        let design = finite([design]);
        if !design.is_empty() {
            p.points(
                Points::new(THIS_DESIGN, design)
                    .radius(POINT_RADIUS + 2.0)
                    .filled(false)
                    .color(visuals.strong_text_color()),
            );
        }
    });
}

fn slip_heating_ui(ui: &mut egui::Ui, results: &DesignResults) {
    let s = slip_heating(results);
    if s.high.is_empty() {
        ui.weak(NOTHING_TO_PLOT);
        return;
    }
    let visuals = ui.visuals().clone();
    plot(
        ui,
        PlotKind::SlipHeating,
        "Time slipping [s]",
        "Magnet temperature [°C]",
    )
    .show(ui, |p| {
        p.line(Line::new(ESTIMATE, s.estimate).color(BLUE).width(1.5));
        p.line(Line::new(HIGH_CASE, s.high).color(AMBER).width(2.0));
        reference_hline(p, LIMIT, s.limit_C, Level::Bad.color(&visuals));
        if let Some(point) = s.time_to_limit {
            p.points(
                Points::new(TIME_TO_LIMIT, finite([point]))
                    .radius(POINT_RADIUS + 1.0)
                    .color(Level::Bad.color(&visuals)),
            );
        }
    });
}

fn torque_angle_ui(ui: &mut egui::Ui, inputs: &DesignInputs, results: &DesignResults) {
    let s = torque_angle(inputs, results);
    if s.curve.is_empty() {
        ui.weak(NOTHING_TO_PLOT);
        return;
    }
    let visuals = ui.visuals().clone();
    let color = tint(BLUE, s.greyed, &visuals);
    plot(
        ui,
        PlotKind::TorqueAngle,
        "Relative rotation [° mechanical]",
        "Torque [N·m]",
    )
    .show(ui, |p| {
        p.line(Line::new(PULL_OUT, s.curve).color(color).width(2.0));
        let point = finite([s.pull_out]);
        if !point.is_empty() {
            p.points(
                Points::new(PULL_OUT_POINT, point)
                    .radius(POINT_RADIUS + 1.0)
                    .color(tint(Level::Bad.color(&visuals), s.greyed, &visuals)),
            );
        }
    });
}

#[cfg(test)]
mod tests {
````

- [ ] **Step 4: Resolve egui_plot into the lock file**

Run (online; never `--offline`):

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib gui::plots 2>&1 | grep -E "Adding|Locking|test result"
cargo update --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml -p egui_plot --precise 0.33.0 2>&1 | tail -2
git -C C:/Users/Cole/source/repos/lsim-mag-m42 diff --stat -- magcoupling-rs/Cargo.lock
grep -A1 'name = "egui_plot"' C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.lock C:/Users/Cole/source/repos/lsim-mag-m42/linkage-sim-rs/Cargo.lock
```

Expected: `Locking 1 package to latest Rust 1.89.0 compatible version` and a line starting `Adding egui_plot v0.33.0` (its `(available: v0.34.1)` names the newest egui_plot on crates.io, built on a newer egui, and may show a later version; the caret range keeps 0.33.0), then `test result: ok. 8 passed; 0 failed`; then `cargo update` prints `Updating crates.io index` and `note: pass `--verbose` to see 11 unchanged dependencies behind latest` (the count may differ) and changes nothing, the lock already holding 0.33.0 (a `Downgrading egui_plot v0.33.x -> v0.33.0` line instead means crates.io now has a newer 0.33 release, which the first line locked: the pin restores 0.33.0, and Step 5 rebuilds with it); `Cargo.lock` changes by `12 +` (the `egui_plot` package and its line in magcoupling-rs's dependencies); both lock files show `version = "0.33.0"`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib gui::plots 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 8 passed; 0 failed` for `gui::plots`, then `test result: ok. 339 passed; 0 failed` for the whole library (Task 0's 309 plus this plan's tests so far).

- [ ] **Step 6: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 7: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m42/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target/gate-task3.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target/gate-task3.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target/gate-task3.log; grep -E "in both lock files" C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target/gate-task3.log
git -C C:/Users/Cole/source/repos/lsim-mag-m42 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines as in Task 0 Step 7; the `grep -c` prints `0`; gate 11 prints `egui 0.32.3 in both lock files`, `egui_plot 0.33.0 in both lock files`, `eframe 0.32.3 in both lock files` and `wasm-bindgen 0.2.114 in both lock files`. If the gate fails, read the log, fix within this task's files and run it again; do not commit red.

- [ ] **Step 8: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/plots.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/Cargo.toml
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/Cargo.lock
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add linkage-sim-rs/scripts/gate.sh
git -C C:/Users/Cole/source/repos/lsim-mag-m42 commit -F - <<'EOF'
feat(magcoupling-rs): the plot tabs (egui_plot 0.33, gate 11 lock parity)

gui::plots draws torque against temperature (the variation band, the required
minimum, the operating temperature, the governing limit), the gap and pole
sweeps (rows by status, grey outside the end-effect model, the design shown),
slip heating against the limit and torque against relative rotation with the
pull-out point (decisions M42-5, M42-6). The series come from this frame's
results through the engine's closed forms, pinned by tests to the engine's
torques; reference lines only when finite; each plot has a constant id.
egui_plot 0.33.0 is locked at linkage-sim-rs's version and gate 11 checks it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m42 status --short
```

Expected: one new commit on `magcoupling/m4-2`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 4: The clamp tab

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

The spec's "clamp table plus the clamp drawing (end and top views, egui painter port of `drawing.py`)". `clamp_drawing` transcribes drawing.py's `clamp_layout` as marks in millimetres with drawing.py's coordinates, texts (`fmt_g` is Python's `:g`) and limits; matplotlib's `set_clip_path(boss)` becomes a Sutherland-Hodgman clip of each cut rectangle to the boss (`clip_to_disc`, a 96-gon); Python's `ValueError` becomes `Err(NO_SCREW_FITS)`, shown in its place. `drawing_ui` paints both views at one scale (Task 2's `side_by_side`; its arrows through `arrowhead`) with drawing.py's four pens in the dark theme (M42-7), its texts scaled with the drawing; the top view draws drawing.py's `range(int(row.screws_needed))` screws, at most 50; `clamp_ui` adds the clamp table: the Shaft clamps summary (each line's hover text through `hover_text`), the machining steps and the 'Clamp screw sizes' table (rows built once from the results table's entries, the recommended size's column green). `CentreView` gains `Clamp`.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/clamp_drawing.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/mod.rs` (`pub mod clamp_drawing;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs` (`CentreView::Clamp`; a test)

**Interfaces:**
- Consumes: Task 1's `geometry::{Mm, finite}`; Task 2's `geometry_view::{DIMENSION, Transform, arrowhead, side_by_side}`, `dashboard::hover_text` and `test_support::flat_shapes`; `dashboard::{Level, result_info}`; `results_table::table_entries`; `engine::clamps::MACHINING_STEPS`, `engine::compat::fmt_fixed`.
- Produces (`magcoupling::gui::clamp_drawing`): `NO_SCREW_FITS`, `ONE_PIECE_ONLY`, `SCREW_SIZES: &str`; `MAX_DRAWN_SCREWS: i64 = 50`; `enum Pen { Ink, Cut, Hidden, Dimension }`; `enum Fill { None, Background, Cut, LightCut }`; `enum Mark { Circle { centre, r, pen, width }, Area { points, fill, pen, dashed, width }, CentreLine { from, to }, Dimension { from, to, text, offset }, Note { text, tip, at, pen } }`; `struct DrawingView { pub title: &'static str, pub marks: Vec<Mark>, pub min: Mm, pub max: Mm }`; `struct ClampDrawing { pub title: String, pub end: DrawingView, pub top: DrawingView }`; `fn fmt_g(f64) -> String`; `fn clip_to_disc(&[Mm], f64) -> Vec<Mm>`; `fn clamp_drawing(&DesignInputs, &DesignResults) -> Result<ClampDrawing, &'static str>`; `fn drawing_ui(&mut egui::Ui, &ClampDrawing, egui::Vec2) -> f32` (the scale, 0 without room); `const SUMMARY: [&str; 11]`; `fn screw_rows() -> &'static [(&'static str, &'static str, String)]`; `fn clamp_ui(&mut egui::Ui, &DesignInputs, &DesignResults)`.
- Produces (`gui::panel`): `CentreView::Clamp`, `CentreView::ALL: [CentreView; 8]`.

- [ ] **Step 1: Write the failing tests**

The module's docs and tests, and the panel's test of the clamp tab. They pin `fmt_g` against Python's `:g`, the clip, every text of both default views (drawing.py's, word for word, with the defaults' numbers), the boss and bore circles and every cut inside the boss, one screw axis, the limits, drawing.py's message for no fit and for an index outside the table, a bounded screw count (the screws needed whatever fits, as drawing.py; none below zero), the screw table's 34 rows, both views painted at one scale with their titles, the tab's summary and steps, a narrow (294 points) and tiny (120 x 90) region, and no room (scale 0).

Create `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/clamp_drawing.rs`:

````rust
//! The clamp tab (spec M4 "Layout": "clamp table plus the clamp drawing (end and top views,
//! egui painter port of `drawing.py`)").
//!
//! [`clamp_drawing`] ports `reference/magcoupling-py/magcoupling/drawing.py` (`clamp_layout`):
//! the end view and the top view of the one-piece slotted clamp for the recommended screw, as
//! marks in millimetres with drawing.py's coordinates, texts and limits, its patches clipped to
//! the boss as `set_clip_path` clips them. Python raises `ValueError` when no screw size fits;
//! here the tab shows that message instead of a drawing. [`clamp_ui`] paints both views at one
//! scale with drawing.py's four pens mapped to the panel's dark theme (decision M42-7), then
//! the clamp table: the Shaft clamps summary, the machining steps and the 'Clamp screw sizes'
//! table with the recommended size's column marked; every value shows its result's hover text.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::test_support::{drawn_texts, flat_shapes, sized_frame};

    fn texts(view: &DrawingView) -> Vec<String> {
        view.marks
            .iter()
            .filter_map(|m| match m {
                Mark::Dimension { text, .. } | Mark::Note { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn fmt_g_matches_python() {
        let cases = [
            (25.0, "25"),
            (0.8, "0.8"),
            (2.9, "2.9"),
            (27.436000000000003, "27.436"),
            (1e-5, "1e-05"),
            (1234567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (123456.0, "123456"),
            (-2.5, "-2.5"),
            (0.0, "0"),
            (f64::NAN, "nan"),
        ];
        for (x, want) in cases {
            assert_eq!(fmt_g(x), want, "{x}");
        }
    }

    #[test]
    fn clipping_keeps_what_is_inside_the_disc() {
        let inside = rect(-1.0, -1.0, 2.0, 2.0);
        assert_eq!(clip_to_disc(&inside, 10.0).len(), 4);
        let straddling = clip_to_disc(&rect(5.0, -1.0, 10.0, 2.0), 10.0);
        assert!(straddling.len() > 4, "the arc adds corners");
        for p in &straddling {
            assert!(p[0].hypot(p[1]) <= 10.0 + 1e-9, "{p:?}");
        }
        assert!(clip_to_disc(&rect(20.0, 20.0, 1.0, 1.0), 10.0).is_empty());
    }

    #[test]
    fn the_default_clamp_is_drawn_with_drawing_py_s_texts() {
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let drawing = clamp_drawing(&inputs, &r).expect("M4 fits at the defaults");
        assert_eq!(
            drawing.title,
            "One-piece slotted clamp, Ø10 keyed shaft: ISO 4762 M4 x 14, class 12.9, 5.1 N·m, 3 mm key"
        );
        assert_eq!(
            texts(&drawing.end),
            [
                "8.25",
                "Ø25 boss",
                "Ø10 H7",
                "Slit 0.8 wide,\nbore to OD",
                "Counterbore Ø7.5 from the OD;\nhead seat 4.26 from the slit",
                "Clearance Ø4.5",
                "M4 tapped\n(drill Ø3.3)",
                "Keyway 4 wide,\n90° from slit",
            ]
        );
        assert_eq!(
            texts(&drawing.top),
            [
                "10 clamp",
                "5",
                "Relief cut 1 wide,\n17 deep from the slit side\n(leaves 8 hinge)",
                "Screw axis\n(square to the slit)",
                "Flange to the steel cup web\n(pilot + 3 x M3 + dowel)",
            ]
        );
        // The boss and the bore, unfilled; every cut clipped to the boss.
        assert_eq!(
            drawing.end.marks[0],
            Mark::Circle {
                centre: [0.0, 0.0],
                r: 12.5,
                pen: Pen::Ink,
                width: 2.0
            }
        );
        for mark in &drawing.end.marks {
            if let Mark::Area {
                points,
                pen: Pen::Cut,
                ..
            } = mark
            {
                assert!(points.iter().all(|p| p[0].hypot(p[1]) <= 12.5 + 1e-9));
            }
        }
        // One screw (M4 x 1): one screw axis across the top view, at the first position.
        let axes: Vec<&Mark> = drawing
            .top
            .marks
            .iter()
            .filter(|m| matches!(m, Mark::CentreLine { from, .. } if from[0] == 5.0))
            .collect();
        assert_eq!(axes.len(), 1);
        assert_eq!(drawing.end.min, [-24.5, -18.5]);
        assert_eq!(drawing.top.max, [33.5, 24.5]);
    }

    #[test]
    fn no_fitting_screw_gives_drawing_py_s_message() {
        let mut inputs = DesignInputs::default();
        inputs.clamps.boss_od_mm = 12.0;
        inputs.clamps.clamp_length_mm = 3.0;
        let r = compute_all(&inputs);
        assert_eq!(r.clamps.index, 0);
        assert_eq!(clamp_drawing(&inputs, &r), Err(NO_SCREW_FITS));
        // An index outside the table (a struct written by hand) is no fit either.
        let mut odd = compute_all(&DesignInputs::default());
        odd.clamps.index = 9;
        assert_eq!(
            clamp_drawing(&DesignInputs::default(), &odd),
            Err(NO_SCREW_FITS)
        );
        odd.clamps.index = -1;
        assert_eq!(
            clamp_drawing(&DesignInputs::default(), &odd),
            Err(NO_SCREW_FITS)
        );
    }

    #[test]
    fn a_screw_count_past_the_table_draws_a_bounded_number() {
        let inputs = DesignInputs::default();
        let mut r = compute_all(&inputs);
        let row = (r.clamps.index - 1) as usize;
        r.clamps.table[row].screws_needed = i64::MAX;
        r.clamps.table[row].screws_fit = i64::MAX;
        let axes = |r: &DesignResults| {
            clamp_drawing(&inputs, r)
                .unwrap()
                .top
                .marks
                .iter()
                .filter(|m| matches!(m, Mark::CentreLine { from, to } if from[0] == to[0]))
                .count()
        };
        assert_eq!(axes(&r), MAX_DRAWN_SCREWS as usize);
        // drawing.py draws the screws needed, whatever fits (range(int(row.screws_needed))).
        r.clamps.table[row].screws_needed = 3;
        r.clamps.table[row].screws_fit = 1;
        assert_eq!(axes(&r), 3);
        r.clamps.table[row].screws_needed = -2;
        assert_eq!(axes(&r), 0);
    }

    #[test]
    fn the_screw_table_rows_follow_the_sheet() {
        let rows = screw_rows();
        assert_eq!(rows.len(), 34, "the size and the 33 celled rows");
        assert_eq!(rows[0].2, "size");
        assert_eq!(rows[1], ("Nominal diameter", "mm", "d_mm".to_owned()));
        let r = compute_all(&DesignInputs::default());
        for (_, _, field) in rows {
            for i in 0..r.clamps.table.len() {
                assert!(
                    r.get(&format!("clamps.table[{i}].{field}")).is_some(),
                    "{field}"
                );
            }
        }
    }

    #[test]
    fn the_clamp_tab_paints_both_views_to_scale_and_the_table() {
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let r = compute_all(&inputs);
        let drawing = clamp_drawing(&inputs, &r).unwrap();
        let mut scale = 0.0;
        let output = sized_frame(&ctx, egui::vec2(1000.0, 700.0), Vec::new(), |ui| {
            scale = drawing_ui(ui, &drawing, egui::vec2(980.0, 420.0));
        });
        assert!(scale > 0.0);
        let radii: Vec<f32> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Circle(c) => Some(c.radius),
                _ => None,
            })
            .collect();
        // The 25 mm boss and the 10 mm bore at one scale.
        assert!(
            radii.iter().any(|r| (r - 12.5 * scale).abs() < 1e-3),
            "{radii:?}"
        );
        assert!(
            radii.iter().any(|r| (r - 5.0 * scale).abs() < 1e-3),
            "{radii:?}"
        );
        let texts = drawn_texts(&output);
        for want in [
            drawing.title.as_str(),
            drawing.end.title,
            drawing.top.title,
            "Ø25 boss",
            "M4 tapped\n(drill Ø3.3)",
        ] {
            assert!(
                texts.iter().any(|t| t == want),
                "missing {want:?} in {texts:?}"
            );
        }
        // The whole tab: the summary and the machining steps; no fit shows the message.
        let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
            clamp_ui(ui, &inputs, &r)
        });
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == "ISO 4762 M4 x 14, class 12.9"));
        assert!(texts.iter().any(|t| t == MACHINING_STEPS[0]));
        assert!(texts.iter().any(|t| t == SCREW_SIZES));
        let mut tight = DesignInputs::default();
        tight.clamps.boss_od_mm = 12.0;
        tight.clamps.clamp_length_mm = 3.0;
        tight.clamps.clamp_type = 2;
        let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
            clamp_ui(ui, &tight, &compute_all(&tight))
        });
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == NO_SCREW_FITS));
        assert!(texts.iter().any(|t| t == ONE_PIECE_ONLY));
        // A narrow, short region still draws the tab (a ~930 px window leaves about 294
        // points).
        for size in [egui::vec2(294.0, 900.0), egui::vec2(120.0, 90.0)] {
            let output = sized_frame(&ctx, size, Vec::new(), |ui| clamp_ui(ui, &inputs, &r));
            assert!(drawn_texts(&output).iter().any(|t| t == &drawing.title));
        }
        // No room at all: no scale, nothing painted.
        let mut scale = None;
        sized_frame(&ctx, egui::vec2(1000.0, 700.0), Vec::new(), |ui| {
            scale = Some(drawing_ui(ui, &drawing, egui::Vec2::ZERO));
        });
        assert_eq!(scale, Some(0.0));
    }
}
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/mod.rs`, replace:

````rust
//! window and event loop. The engine stays pure std; nothing here is compiled
//! without the feature.

pub mod corrections;
pub mod dashboard;
mod format;
````

with:

````rust
//! window and event loop. The engine stays pure std; nothing here is compiled
//! without the feature.

pub mod clamp_drawing;
pub mod corrections;
pub mod dashboard;
mod format;
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            at(after)
        );
        assert!(markers.iter().all(|c| (*c - at(before)).length() > 0.5));
    }

    #[test]
````

with:

````rust
            at(after)
        );
        assert!(markers.iter().all(|c| (*c - at(before)).length() > 0.5));
    }

    #[test]
    fn the_clamp_tab_draws_the_recommended_clamp_and_follows_the_design() {
        let mut harness = Harness::new();
        harness.click_text(CentreView::Clamp.label());
        assert_eq!(harness.panel.centre, CentreView::Clamp);
        let output = harness.frame(Vec::new());
        let title = "One-piece slotted clamp, \u{d8}10 keyed shaft: ISO 4762 M4 x 14, class 12.9, 5.1 N\u{b7}m, 3 mm key";
        assert_eq!(count(&output, title), 1);
        // A boss too small for any screw: drawing.py's message instead of the drawing.
        harness.panel.inputs.clamps.boss_od_mm = 12.0;
        harness.panel.inputs.clamps.clamp_length_mm = 3.0;
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, crate::gui::clamp_drawing::NO_SCREW_FITS), 1);
        assert_eq!(count(&output, title), 0);
    }

    #[test]
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib gui:: 2>&1 | grep -E "^error" | sort | uniq -c
```

Expected: FAIL to compile. The build stops with `error: could not compile `magcoupling-rs` (lib test) due to 51 previous errors; 1 warning emitted`; the errors name the items Step 3 adds, among them `error[E0425]: cannot find function `clamp_drawing` in this scope`, `error[E0425]: cannot find function `fmt_g` in this scope`, `error[E0412]: cannot find type `Mark` in this scope` and `error[E0599]: no variant or associated item named `Clamp` found for enum `gui::panel::CentreView` in the current scope`.

- [ ] **Step 3: Write the implementation**

The module's code between its docs and its tests, and the panel's clamp tab.

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/clamp_drawing.rs`, replace:

````rust

#[cfg(test)]
mod tests {
````

with:

````rust

use std::sync::OnceLock;

use egui::{Color32, Stroke};

use crate::engine::clamps::MACHINING_STEPS;
use crate::engine::compat::fmt_fixed;
use crate::engine::meta::{NumOrText, ResultSet, Value};
use crate::gui::dashboard::{Level, hover_text};
use crate::gui::format::{format_value, with_unit};
use crate::gui::geometry::{Mm, finite};
use crate::gui::geometry_view::{DIMENSION, Transform, arrowhead, side_by_side};
use crate::gui::results_table::table_entries;
use crate::{DesignInputs, DesignResults};

/// drawing.py's `ValueError` text, shown when no screw size fits.
pub const NO_SCREW_FITS: &str = "No screw size fits; enlarge the boss or the clamp length first.";

/// The note under the drawing when the two-piece clamp is selected: drawing.py draws only the
/// one-piece layout.
pub const ONE_PIECE_ONLY: &str =
    "The drawing shows the one-piece slotted clamp (drawing.py's only layout).";

/// The most screws the top view draws (a design file can hold any clamp).
pub const MAX_DRAWN_SCREWS: i64 = 50;

/// The heading of the screw sizes table.
pub const SCREW_SIZES: &str = "Clamp screw sizes";

/// drawing.py's pens: INK, CUT, HID and DIM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pen {
    Ink,
    Cut,
    Hidden,
    Dimension,
}

/// drawing.py's fills: none, white (the background), CUT and the light cut `#f4d9d5`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    None,
    Background,
    Cut,
    LightCut,
}

/// One mark of a view [mm].
#[derive(Clone, Debug, PartialEq)]
pub enum Mark {
    /// An unfilled circle.
    Circle {
        centre: Mm,
        r: f64,
        pen: Pen,
        width: f32,
    },
    /// A closed convex area: a rectangle, or a rectangle clipped to the boss.
    Area {
        points: Vec<Mm>,
        fill: Fill,
        pen: Pen,
        dashed: bool,
        width: f32,
    },
    /// A centre line (drawing.py's dash-dot `HID` lines).
    CentreLine { from: Mm, to: Mm },
    /// drawing.py's `_dim`: a double arrow with its text at the middle plus `offset`.
    Dimension {
        from: Mm,
        to: Mm,
        text: String,
        offset: Mm,
    },
    /// drawing.py's `annotate`: `text` at `at` with an arrow to `tip`.
    Note {
        text: String,
        tip: Mm,
        at: Mm,
        pen: Pen,
    },
}

/// One view: its title, marks and limits [mm].
#[derive(Clone, Debug, PartialEq)]
pub struct DrawingView {
    pub title: &'static str,
    pub marks: Vec<Mark>,
    pub min: Mm,
    pub max: Mm,
}

/// The clamp drawing: drawing.py's suptitle and its two views.
#[derive(Clone, Debug, PartialEq)]
pub struct ClampDrawing {
    pub title: String,
    pub end: DrawingView,
    pub top: DrawingView,
}

/// Python's `f"{x:g}"`: six significant digits, trailing zeros dropped, scientific below 1e-4
/// and from 1e6 (`25.0` gives `25`, `0.8` gives `0.8`, `1e-5` gives `1e-05`).
pub fn fmt_g(x: f64) -> String {
    if !x.is_finite() {
        return match x {
            x if x.is_nan() => "nan".to_owned(),
            x if x > 0.0 => "inf".to_owned(),
            _ => "-inf".to_owned(),
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    let scientific = format!("{x:.5e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("Rust's {:e} has an exponent");
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    let trim = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s.to_owned()
        }
    };
    if (-4..6).contains(&exponent) {
        let decimals = (5 - exponent).max(0) as usize;
        trim(&format!("{x:.decimals$}"))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exponent.abs())
    }
}

/// A rectangle from its corner `(x, y)` and its size, as matplotlib's `Rectangle` takes it.
fn rect(x: f64, y: f64, w: f64, h: f64) -> Vec<Mm> {
    vec![[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
}

/// The convex polygon `points` clipped to the disc of radius `r` at the origin (approximated by
/// the regular 96-gon inside it): Sutherland–Hodgman against each of its edges. Empty when
/// nothing is inside.
pub fn clip_to_disc(points: &[Mm], r: f64) -> Vec<Mm> {
    const SIDES: usize = 96;
    let corner = |i: usize| {
        let a = std::f64::consts::TAU * i as f64 / SIDES as f64;
        [r * a.cos(), r * a.sin()]
    };
    let mut out = points.to_vec();
    for i in 0..SIDES {
        let (a, b) = (corner(i), corner(i + 1));
        // Inside: to the left of the edge a -> b (the polygon runs anticlockwise).
        let side = |p: Mm| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        let input = std::mem::take(&mut out);
        for (j, &p) in input.iter().enumerate() {
            let q = input[(j + 1) % input.len()];
            let (sp, sq) = (side(p), side(q));
            if sp >= 0.0 {
                out.push(p);
            }
            if (sp >= 0.0) != (sq >= 0.0) {
                let t = sp / (sp - sq);
                out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
            }
        }
        if out.is_empty() {
            break;
        }
    }
    out
}

/// A number result that may be text: the number, or NaN.
fn num(value: &NumOrText) -> f64 {
    match value {
        NumOrText::Num(x) => *x,
        NumOrText::Text(_) => f64::NAN,
    }
}

/// drawing.py's `clamp_layout`, as marks; `Err(NO_SCREW_FITS)` where Python raises.
#[allow(non_snake_case)] // drawing.py's names (D, L, R)
pub fn clamp_drawing(
    inputs: &DesignInputs,
    results: &DesignResults,
) -> Result<ClampDrawing, &'static str> {
    let (c, cl, md) = (&inputs.clamps, &results.clamps, &inputs.metal);
    let row = usize::try_from(cl.index)
        .ok()
        .and_then(|i| i.checked_sub(1))
        .and_then(|i| cl.table.get(i))
        .ok_or(NO_SCREW_FITS)?;
    let (D, d, L) = (c.boss_od_mm, cl.shaft_mm, c.clamp_length_mm);
    let (R, r) = (D / 2.0, d / 2.0);
    let (e, hole, cb, tap, thr) = (
        row.offset_mm,
        row.hole_mm,
        row.cbore_dia_mm,
        row.tap_drill_mm,
        row.d_mm,
    );
    let x_out = (R.powi(2) - e.powi(2)).sqrt();
    let x_seat = row.grip_mm + c.slit_mm / 2.0;
    let (key_w, key_d) = (c.key_width_mm, inputs.coupling.keyway_depth_mm);
    let slit = c.slit_mm;
    let clipped = |points: Vec<Mm>, fill: Fill, dashed: bool, width: f32| Mark::Area {
        points: clip_to_disc(&points, R),
        fill,
        pen: Pen::Cut,
        dashed,
        width,
    };

    // End view.
    let mut end = vec![
        Mark::Circle {
            centre: [0.0, 0.0],
            r: R,
            pen: Pen::Ink,
            width: 2.0,
        },
        Mark::Circle {
            centre: [0.0, 0.0],
            r,
            pen: Pen::Ink,
            width: 2.0,
        },
        Mark::Area {
            points: rect(r - 0.2, -key_w / 2.0, key_d + 0.2, key_w),
            fill: Fill::Background,
            pen: Pen::Ink,
            dashed: false,
            width: 1.5,
        },
        clipped(
            rect(-slit / 2.0, r, slit, R - r + 1.0),
            Fill::Cut,
            false,
            1.0,
        ),
        clipped(
            rect(-R - 1.0, e - cb / 2.0, R + 1.0 - x_seat, cb),
            Fill::LightCut,
            true,
            1.2,
        ),
        clipped(
            rect(-x_seat, e - hole / 2.0, x_seat - slit / 2.0, hole),
            Fill::LightCut,
            true,
            1.2,
        ),
        clipped(
            rect(slit / 2.0, e - thr / 2.0, R + 1.0 - slit / 2.0, thr),
            Fill::LightCut,
            true,
            1.2,
        ),
    ];
    for y in [e, 0.0] {
        end.push(Mark::CentreLine {
            from: [-R - 2.0, y],
            to: [R + 2.0, y],
        });
    }
    end.push(Mark::CentreLine {
        from: [0.0, -R - 2.0],
        to: [0.0, R + 2.0],
    });
    end.extend([
        Mark::Dimension {
            from: [R + 3.5, 0.0],
            to: [R + 3.5, e],
            text: fmt_fixed(e, 2),
            offset: [1.6, 0.0],
        },
        Mark::Dimension {
            from: [-R, -R - 3.0],
            to: [R, -R - 3.0],
            text: format!("Ø{} boss", fmt_g(D)),
            offset: [0.0, -1.2],
        },
        Mark::Dimension {
            from: [-r, -2.2],
            to: [r, -2.2],
            text: format!("Ø{} H7", fmt_g(d)),
            offset: [0.0, -1.1],
        },
        Mark::Note {
            text: format!("Slit {} wide,\nbore to OD", fmt_g(slit)),
            tip: [0.0, R - 1.0],
            at: [4.5, R + 4.5],
            pen: Pen::Cut,
        },
        Mark::Note {
            text: format!(
                "Counterbore Ø{} from the OD;\nhead seat {} from the slit",
                fmt_g(cb),
                fmt_fixed(x_seat, 2)
            ),
            tip: [-x_out + 2.0, e + 1.0],
            at: [-R - 11.0, R + 3.0],
            pen: Pen::Cut,
        },
        Mark::Note {
            text: format!("Clearance Ø{}", fmt_g(hole)),
            tip: [-2.5, e - hole / 2.0],
            at: [-R - 9.0, 1.5],
            pen: Pen::Cut,
        },
        Mark::Note {
            text: format!("{} tapped\n(drill Ø{})", row.size, fmt_g(tap)),
            tip: [x_out - 2.0, e + thr / 2.0],
            at: [R + 1.0, R + 3.0],
            pen: Pen::Cut,
        },
        Mark::Note {
            text: format!("Keyway {} wide,\n90° from slit", fmt_g(key_w)),
            tip: [r + key_d, 0.0],
            at: [R + 1.5, -7.5],
            pen: Pen::Ink,
        },
    ]);

    // Top view onto the slit.
    let (fl_D, fl_t, pl_D, pl_t) = (
        md.adapter_flange_dia_mm,
        md.adapter_flange_mm,
        md.adapter_pilot_dia_mm,
        md.adapter_pilot_mm,
    );
    let relief = c.relief_mm;
    let area = |points: Vec<Mm>, fill: Fill, pen: Pen, dashed: bool, width: f32| Mark::Area {
        points,
        fill,
        pen,
        dashed,
        width,
    };
    let mut top = vec![
        area(rect(0.0, -R, L, D), Fill::Background, Pen::Ink, false, 2.0),
        area(rect(L, -R, relief, D), Fill::LightCut, Pen::Cut, false, 1.2),
        area(
            rect(L + relief, -fl_D / 2.0, fl_t, fl_D),
            Fill::Background,
            Pen::Ink,
            false,
            2.0,
        ),
        area(
            rect(L + relief + fl_t, -pl_D / 2.0, pl_t, pl_D),
            Fill::Background,
            Pen::Ink,
            false,
            2.0,
        ),
        area(
            rect(0.0, -slit / 2.0, L, slit),
            Fill::Cut,
            Pen::Cut,
            false,
            1.0,
        ),
        Mark::CentreLine {
            from: [-2.0, 0.0],
            to: [L + relief + fl_t + pl_t + 2.0, 0.0],
        },
    ];
    let first = num(&cl.layout_first_mm);
    let pitch = num(&cl.layout_pitch_mm);
    // drawing.py draws `range(int(row.screws_needed))`, whatever fits; at most
    // MAX_DRAWN_SCREWS here.
    let screws = row.screws_needed.clamp(0, MAX_DRAWN_SCREWS);
    for i in 0..screws {
        let zc = first + i as f64 * pitch;
        top.push(Mark::CentreLine {
            from: [zc, -R - 1.5],
            to: [zc, R + 1.5],
        });
        top.push(area(
            rect(zc - cb / 2.0, -x_out, cb, x_out - x_seat),
            Fill::None,
            Pen::Cut,
            true,
            1.2,
        ));
        top.push(area(
            rect(zc - hole / 2.0, -x_seat, hole, x_seat - slit / 2.0),
            Fill::None,
            Pen::Cut,
            true,
            1.2,
        ));
        top.push(area(
            rect(zc - thr / 2.0, slit / 2.0, thr, x_out - slit / 2.0),
            Fill::None,
            Pen::Cut,
            true,
            1.2,
        ));
    }
    top.extend([
        Mark::Dimension {
            from: [0.0, R + 3.0],
            to: [L, R + 3.0],
            text: format!("{} clamp", fmt_g(L)),
            offset: [0.0, 1.2],
        },
        Mark::Dimension {
            from: [0.0, -R - 3.0],
            to: [first, -R - 3.0],
            text: fmt_g(first),
            offset: [0.0, -1.2],
        },
        Mark::Note {
            text: format!(
                "Relief cut {} wide,\n{} deep from the slit side\n(leaves {} hinge)",
                fmt_g(relief),
                fmt_g(D - c.hinge_mm),
                fmt_g(c.hinge_mm)
            ),
            tip: [L + relief / 2.0, R - 2.0],
            at: [L + 7.0, R + 7.0],
            pen: Pen::Cut,
        },
        Mark::Note {
            text: "Screw axis\n(square to the slit)".to_owned(),
            tip: [first, -R + 3.0],
            at: [-9.0, -R - 5.0],
            pen: Pen::Hidden,
        },
        Mark::Note {
            text: "Flange to the steel cup web\n(pilot + 3 x M3 + dowel)".to_owned(),
            tip: [L + relief + fl_t / 2.0, -fl_D / 2.0 + 2.0],
            at: [L + 9.0, -R - 7.0],
            pen: Pen::Ink,
        },
    ]);

    Ok(ClampDrawing {
        title: format!(
            "One-piece slotted clamp, Ø{} keyed shaft: {}, {} N·m, {} mm key",
            fmt_g(d),
            cl.recommended,
            fmt_fixed(num(&cl.tightening_Nm), 1),
            fmt_g(num(&cl.hex_mm))
        ),
        end: DrawingView {
            title: "End view from the free end (screw hidden, dashed)",
            marks: end,
            min: [-R - 12.0, -R - 6.0],
            max: [R + 12.0, R + 8.0],
        },
        top: DrawingView {
            title: "Top view onto the slit",
            marks: top,
            min: [-12.0, -R - 10.0],
            max: [L + relief + fl_t + pl_t + 16.0, R + 12.0],
        },
    })
}

/// The colour of a pen in the panel's theme.
fn pen_color(pen: Pen, visuals: &egui::Visuals) -> Color32 {
    match pen {
        Pen::Ink => visuals.strong_text_color(),
        Pen::Cut => Color32::from_rgb(225, 95, 80),
        Pen::Hidden => visuals.weak_text_color(),
        Pen::Dimension => DIMENSION,
    }
}

/// The colour of a fill in the panel's theme.
fn fill_color(fill: Fill, visuals: &egui::Visuals) -> Color32 {
    match fill {
        Fill::None => Color32::TRANSPARENT,
        Fill::Background => visuals.extreme_bg_color,
        Fill::Cut => pen_color(Pen::Cut, visuals),
        Fill::LightCut => Color32::from_rgba_unmultiplied(225, 95, 80, 60),
    }
}

/// Paints one view with `t`. The texts scale with the drawing (drawing.py's 9-point text is
/// about 0.9 mm on its figure), between 8 and 12 points, so they do not crowd a small drawing.
fn paint_view(painter: &egui::Painter, view: &DrawingView, t: Transform, visuals: &egui::Visuals) {
    let font = egui::FontId::proportional((t.scale * 0.9).clamp(8.0, 12.0));
    let arrow = |from: egui::Pos2, to: egui::Pos2, color: Color32| {
        let stroke = Stroke::new(1.0, color);
        painter.line_segment([from, to], stroke);
        arrowhead(painter, to, (to - from).normalized(), stroke);
    };
    for mark in &view.marks {
        match mark {
            Mark::Circle {
                centre,
                r,
                pen,
                width,
            } if finite(*centre) && r.is_finite() => {
                painter.circle_stroke(
                    t.to_px(*centre),
                    t.len(*r),
                    Stroke::new(*width, pen_color(*pen, visuals)),
                );
            }
            Mark::Area {
                points,
                fill,
                pen,
                dashed,
                width,
            } if points.len() >= 3 && points.iter().all(|p| finite(*p)) => {
                let px: Vec<egui::Pos2> = points.iter().map(|p| t.to_px(*p)).collect();
                let stroke = Stroke::new(*width, pen_color(*pen, visuals));
                let fill = fill_color(*fill, visuals);
                if *dashed {
                    painter.add(egui::Shape::convex_polygon(px.clone(), fill, Stroke::NONE));
                    let mut closed = px;
                    closed.push(closed[0]);
                    painter.extend(egui::Shape::dashed_line(&closed, stroke, 4.0, 3.0));
                } else {
                    painter.add(egui::Shape::convex_polygon(px, fill, stroke));
                }
            }
            Mark::CentreLine { from, to } if finite(*from) && finite(*to) => {
                let points = [t.to_px(*from), t.to_px(*to)];
                let stroke = Stroke::new(0.8, pen_color(Pen::Hidden, visuals));
                painter.extend(egui::Shape::dashed_line(&points, stroke, 8.0, 3.0));
            }
            Mark::Dimension {
                from,
                to,
                text,
                offset,
            } if finite(*from) && finite(*to) => {
                let color = pen_color(Pen::Dimension, visuals);
                let (a, b) = (t.to_px(*from), t.to_px(*to));
                let middle = a + (b - a) / 2.0;
                arrow(middle, a, color);
                arrow(middle, b, color);
                let at = t.to_px([
                    (from[0] + to[0]) / 2.0 + offset[0],
                    (from[1] + to[1]) / 2.0 + offset[1],
                ]);
                let galley = painter.layout_no_wrap(text.clone(), font.clone(), color);
                let rect = egui::Align2::CENTER_CENTER.anchor_size(at, galley.size());
                painter.rect_filled(rect.expand(1.0), 0.0, visuals.extreme_bg_color);
                painter.galley(rect.min, galley, color);
            }
            Mark::Note { text, tip, at, pen } if finite(*tip) && finite(*at) => {
                let color = pen_color(*pen, visuals);
                let start = t.to_px(*at);
                arrow(start, t.to_px(*tip), color);
                painter.text(start, egui::Align2::LEFT_BOTTOM, text, font.clone(), color);
            }
            _ => {}
        }
    }
    let title_at = t.to_px([(view.min[0] + view.max[0]) / 2.0, view.max[1]]);
    painter.text(
        title_at,
        egui::Align2::CENTER_TOP,
        view.title,
        egui::FontId::proportional((t.scale * 1.1).clamp(9.0, 14.0)),
        pen_color(Pen::Ink, visuals),
    );
}

/// Draws the clamp drawing (its title, then both views side by side at one scale,
/// [`side_by_side`]) in `size` and returns the scale [points per mm], 0 when there is no room.
pub fn drawing_ui(ui: &mut egui::Ui, drawing: &ClampDrawing, size: egui::Vec2) -> f32 {
    let visuals = ui.visuals().clone();
    ui.label(egui::RichText::new(&drawing.title).strong());
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, visuals.extreme_bg_color);
    let views = [&drawing.end, &drawing.top];
    let extents: Vec<(Mm, Mm)> = views.iter().map(|v| (v.min, v.max)).collect();
    let Some((scale, transforms)) = side_by_side(rect, &extents, 0.0) else {
        return 0.0;
    };
    for (view, t) in views.into_iter().zip(transforms) {
        paint_view(&painter, view, t, &visuals);
    }
    scale
}

/// The Shaft clamps results the clamp table lists first.
pub const SUMMARY: [&str; 11] = [
    "clamps.recommended",
    "clamps.length_note",
    "clamps.screws",
    "clamps.tightening_Nm",
    "clamps.hex_mm",
    "clamps.capacity_Nm",
    "clamps.sf_coupling",
    "clamps.head_check",
    "clamps.vent_port",
    "clamps.key_sf",
    "clamps.joint_sf",
];

/// The screw sizes table's rows, built once: (label, unit, field), in the table's order.
pub fn screw_rows() -> &'static [(&'static str, &'static str, String)] {
    static ROWS: OnceLock<Vec<(&'static str, &'static str, String)>> = OnceLock::new();
    ROWS.get_or_init(|| {
        table_entries()
            .iter()
            .filter_map(|entry| {
                let field = entry.path.strip_prefix("clamps.table[0].")?;
                Some((
                    entry.info.meta.label,
                    entry.info.meta.unit,
                    field.to_owned(),
                ))
            })
            .collect()
    })
}

/// The clamp tab: the drawing (or why there is none), then the clamp table.
pub fn clamp_ui(ui: &mut egui::Ui, inputs: &DesignInputs, results: &DesignResults) {
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_clamp_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            match clamp_drawing(inputs, results) {
                Ok(drawing) => {
                    // The two views' proportions (about 2:1), at most 80 % of the height, at
                    // least 160 points (f32::clamp would panic where the width allows less).
                    let width = ui.available_width();
                    let height = (width * 0.5).min(ui.available_height() * 0.8).max(160.0);
                    drawing_ui(ui, &drawing, egui::vec2(width, height));
                }
                Err(message) => {
                    ui.colored_label(Level::Bad.color(ui.visuals()), message);
                }
            }
            if inputs.clamps.clamp_type != 1 {
                ui.weak(ONE_PIECE_ONLY);
            }
            ui.separator();
            // One line per result, label then value, wrapping at the region's width; the hover
            // text is built only while hovered.
            for path in SUMMARY {
                let value = results.get(path).unwrap_or(Value::None);
                if value == Value::Text(String::new()) {
                    continue; // an empty length note
                }
                let Some(info) = crate::gui::dashboard::result_info(path) else {
                    continue;
                };
                let response = ui
                    .horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new(info.meta.label).weak());
                        ui.label(with_unit(format_value(&value), info.meta.unit));
                    })
                    .response;
                response.on_hover_ui(|ui| {
                    ui.label(hover_text(path).unwrap_or_default());
                });
            }
            ui.add_space(4.0);
            for step in MACHINING_STEPS {
                ui.add(egui::Label::new(egui::RichText::new(step).weak()).wrap());
            }
            egui::CollapsingHeader::new(SCREW_SIZES)
                .id_salt("magcoupling_screw_sizes")
                .default_open(false)
                .show(ui, |ui| screw_table_ui(ui, results));
        });
}

/// The 'Clamp screw sizes' table: a row per field, a column per size, the recommended size's
/// column in green.
fn screw_table_ui(ui: &mut egui::Ui, results: &DesignResults) {
    let sizes = results.clamps.table.len();
    let pick = usize::try_from(results.clamps.index)
        .ok()
        .and_then(|i| i.checked_sub(1));
    let good = Level::Good.color(ui.visuals());
    egui::ScrollArea::horizontal()
        .id_salt("magcoupling_screw_sizes_scroll")
        .show(ui, |ui| {
            egui::Grid::new("magcoupling_screw_sizes_grid")
                .striped(true)
                .show(ui, |ui| {
                    for (label, unit, field) in screw_rows() {
                        ui.label(with_unit((*label).to_owned(), unit));
                        for i in 0..sizes {
                            let path = format!("clamps.table[{i}].{field}");
                            let text = format_value(&results.get(&path).unwrap_or(Value::None));
                            let rich = egui::RichText::new(text);
                            let rich = if Some(i) == pick {
                                rich.color(good).strong()
                            } else {
                                rich
                            };
                            ui.label(rich).on_hover_ui(|ui| {
                                ui.label(hover_text(&path).unwrap_or_else(|| path.clone()));
                            });
                        }
                        ui.end_row();
                    }
                });
        });
}

#[cfg(test)]
mod tests {
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::{Level, dashboard_ui, end_effect_banner};
use crate::gui::geometry_view::geometry_ui;
use crate::gui::history::History;
````

with:

````rust

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::clamp_drawing::clamp_ui;
use crate::gui::dashboard::{Level, dashboard_ui, end_effect_banner};
use crate::gui::geometry_view::geometry_ui;
use crate::gui::history::History;
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    Geometry,
    /// One of the plots (egui_plot).
    Plot(PlotKind),
    /// Every result: label, value, unit, cell; searchable; CSV and JSON export.
    Results,
}

impl CentreView {
    /// Every view, in tab order.
    pub const ALL: [CentreView; 7] = [
        CentreView::Geometry,
        CentreView::Plot(PlotKind::TorqueTemperature),
        CentreView::Plot(PlotKind::GapSweep),
        CentreView::Plot(PlotKind::PoleSweep),
        CentreView::Plot(PlotKind::SlipHeating),
        CentreView::Plot(PlotKind::TorqueAngle),
        CentreView::Results,
    ];

````

with:

````rust
    Geometry,
    /// One of the plots (egui_plot).
    Plot(PlotKind),
    /// The clamp drawing (drawing.py's end and top views) and the clamp table.
    Clamp,
    /// Every result: label, value, unit, cell; searchable; CSV and JSON export.
    Results,
}

impl CentreView {
    /// Every view, in tab order.
    pub const ALL: [CentreView; 8] = [
        CentreView::Geometry,
        CentreView::Plot(PlotKind::TorqueTemperature),
        CentreView::Plot(PlotKind::GapSweep),
        CentreView::Plot(PlotKind::PoleSweep),
        CentreView::Plot(PlotKind::SlipHeating),
        CentreView::Plot(PlotKind::TorqueAngle),
        CentreView::Clamp,
        CentreView::Results,
    ];

````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        match self {
            CentreView::Geometry => "Geometry",
            CentreView::Plot(kind) => kind.label(),
            CentreView::Results => "Results table",
        }
    }
````

with:

````rust
        match self {
            CentreView::Geometry => "Geometry",
            CentreView::Plot(kind) => kind.label(),
            CentreView::Clamp => "Clamp",
            CentreView::Results => "Results table",
        }
    }
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                geometry_ui(ui, shown, &self.results);
            }
            CentreView::Plot(kind) => plot_ui(ui, kind, shown, &self.results),
            CentreView::Results => {
                let action = self.results_table.ui(ui, &self.results);
                match action {
````

with:

````rust
                geometry_ui(ui, shown, &self.results);
            }
            CentreView::Plot(kind) => plot_ui(ui, kind, shown, &self.results),
            CentreView::Clamp => clamp_ui(ui, shown, &self.results),
            CentreView::Results => {
                let action = self.results_table.ui(ui, &self.results);
                match action {
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib gui::clamp_drawing 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 7 passed; 0 failed` for `gui::clamp_drawing`, then `test result: ok. 347 passed; 0 failed` for the whole library (Task 0's 309 plus this plan's tests so far).

- [ ] **Step 5: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/clamp_drawing.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 commit -F - <<'EOF'
feat(magcoupling-rs): the clamp tab, drawing.py's clamp drawing and the clamp table

gui::clamp_drawing ports drawing.py's clamp_layout to marks in millimetres
(its coordinates, texts and limits; the cuts clipped to the boss) and paints
the end and top views at one scale in the dark theme's pens (decision M42-7);
no fitting screw shows drawing.py's message. The clamp table lists the Shaft
clamps summary, the machining steps and the screw sizes table with the
recommended column, every value with its hover text. CentreView gains Clamp.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m42 status --short
```

Expected: one new commit on `magcoupling/m4-2`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 5: The results table at narrow widths

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

The M4-1 final review's open item (`docs/ai/04-memory.yaml`): the table's fixed columns (260, 130, 130 and 60 points beside the 320- and 300-point sides) left a ~930 px window only the label column without scrolling, and the M4-2 tabs share that width. `column_widths` keeps the value, cell and marker columns and lets the label column take the rest, at least 120 points (decision M42-8), so the label and the value show at 930 px; narrower still, the rows scroll sideways as before.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/results_table.rs` (`column_widths` and its constants; `row_ui` takes the widths)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs` (a test)

**Interfaces:**
- Consumes: Task 2's `test_support::text_rects`.
- Produces (`gui::results_table`): `VALUE_WIDTH = 130.0`, `CELL_WIDTH = 130.0`, `MARKER_WIDTH = 60.0` (M4-1's widths), `LABEL_MIN_WIDTH = 120.0`; `fn column_widths(available: f32, spacing: f32) -> [f32; 4]` (label, value, cell, marker).

- [ ] **Step 1: Write the failing tests**

The widths at the default window (644 points: the label widens to 300) and at ~930 px (294 points: the label at its minimum, the value column ending at 120 + 8 + 130 = 258 points), NaN; and in the panel, a 930 x 1024 screen whose results table, searched for the pull-out's cell, draws the row's label (filling its 120-point column) and then its value left of the dashboard (the tab row wraps there and settles on the second frame).

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    use crate::gui::sizing::DEBOUNCE_S;
    use crate::gui::test_support::{
        SCREEN, drawn_texts, key_event, key_tap, primary_button, select_all, short_magnets,
        sized_frame, sized_frame_at, text_rect,
    };
    use crate::headline;

````

with:

````rust
    use crate::gui::sizing::DEBOUNCE_S;
    use crate::gui::test_support::{
        SCREEN, drawn_texts, key_event, key_tap, primary_button, select_all, short_magnets,
        sized_frame, sized_frame_at, text_rect, text_rects,
    };
    use crate::headline;

````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        assert_eq!(count(&output, &last), 1);
        assert_eq!(count(&output, &first), 0);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
````

with:

````rust
        assert_eq!(count(&output, &last), 1);
        assert_eq!(count(&output, &first), 0);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn a_narrow_window_shows_each_row_s_label_and_value_without_scrolling() {
        // The M4-1 review's ~930 px window: the inputs (320) and the dashboard (300) leave the
        // table about 294 points. The row's value must end left of the dashboard.
        let mut harness = Harness::on_screen(egui::vec2(930.0, 1024.0));
        // The wrapped tab row settles on the second frame (egui wraps from the last frame's
        // widths).
        harness.frame(Vec::new());
        harness.click_text(CentreView::Results.label());
        assert_eq!(harness.panel.centre, CentreView::Results);
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("calculator!c93".to_owned())]);
        let output = harness.frame(Vec::new());
        let value = displayed_pullout(&DesignInputs::default());
        let in_table: Vec<egui::Rect> = text_rects(&output, &value)
            .into_iter()
            .filter(|r| r.right() <= 930.0 - 300.0)
            .collect();
        assert_eq!(in_table.len(), 1, "the row's value is on screen: {value}");
        // And the row's label (truncated to its column, but there).
        let label = crate::gui::results_table::table_entries()
            .iter()
            .find(|e| e.info.cell.as_deref() == Some("Calculator!C93"))
            .expect("the pull-out's row")
            .info
            .meta
            .label;
        let labels: Vec<egui::Rect> = text_rects(&output, label)
            .into_iter()
            .filter(|r| r.left() >= 0.0 && r.right() <= 930.0 - 300.0)
            .collect();
        assert_eq!(labels.len(), 1, "the row's label is on screen: {label}");
        // It fills its column, at least 120 points, and ends left of the value.
        let min = crate::gui::results_table::LABEL_MIN_WIDTH;
        assert!(labels[0].width() >= min - 1.0, "{:?}", labels[0]);
        assert!(labels[0].right() <= in_table[0].left());
    }

    #[test]
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
            .find(|e| e.path == "model.pullout_Nm")
            .unwrap();
        assert_eq!(pullout.marker, "E3 E7 E8");
    }

    #[test]
````

with:

````rust
            .find(|e| e.path == "model.pullout_Nm")
            .unwrap();
        assert_eq!(pullout.marker, "E3 E7 E8");
    }

    #[test]
    fn the_label_column_flexes_down_to_its_minimum() {
        // The 1280 x 800 default window: the label column widens past the M4-1 260 points.
        assert_eq!(column_widths(644.0, 8.0), [300.0, 130.0, 130.0, 60.0]);
        // A ~930 px window leaves about 294 points: the label shrinks to its minimum, so the
        // value column ends at 120 + 8 + 130 = 258 points, on screen.
        assert_eq!(column_widths(294.0, 8.0), [120.0, 130.0, 130.0, 60.0]);
        assert_eq!(column_widths(0.0, 8.0)[0], LABEL_MIN_WIDTH);
        assert_eq!(column_widths(f32::NAN, 8.0)[0], LABEL_MIN_WIDTH);
    }

    #[test]
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib gui:: 2>&1 | grep -E "^error" | sort | uniq -c
```

Expected: FAIL to compile. The build stops with `error: could not compile `magcoupling-rs` (lib test) due to 7 previous errors`; the errors are `error[E0425]: cannot find function `column_widths` in this scope`, `error[E0425]: cannot find value `LABEL_MIN_WIDTH` in this scope` and `error[E0425]: cannot find value `LABEL_MIN_WIDTH` in module `crate::gui::results_table``.

- [ ] **Step 3: Write the implementation**

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust

/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";

/// One row of the table: what does not change with the inputs.
#[derive(Clone, Debug)]
````

with:

````rust

/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";

/// The value, cell and marker columns' widths [points], M4-1's; the label column takes the rest.
pub const VALUE_WIDTH: f32 = 130.0;
pub const CELL_WIDTH: f32 = 130.0;
pub const MARKER_WIDTH: f32 = 60.0;

/// The narrowest label column [points]: below it the rows scroll sideways.
pub const LABEL_MIN_WIDTH: f32 = 120.0;

/// The widths of the label, value, cell and marker columns of a table `available` points wide
/// with `spacing` points between columns (decision M42-8): the label column flexes, at least
/// [`LABEL_MIN_WIDTH`], so a narrow centre region (a ~930 px window, where the M4-1 table showed
/// only its labels) still shows the label and the value without scrolling.
pub fn column_widths(available: f32, spacing: f32) -> [f32; 4] {
    let fixed = VALUE_WIDTH + CELL_WIDTH + MARKER_WIDTH + 3.0 * spacing;
    [
        (available - fixed).max(LABEL_MIN_WIDTH),
        VALUE_WIDTH,
        CELL_WIDTH,
        MARKER_WIDTH,
    ]
}

/// One row of the table: what does not change with the inputs.
#[derive(Clone, Debug)]
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
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
````

with:

````rust
        ui.separator();
        let matches = self.matches.as_deref().unwrap_or(&[]);
        let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;
        let widths = column_widths(ui.available_width(), ui.spacing().item_spacing.x);
        egui::ScrollArea::both()
            .id_salt("magcoupling_results_scroll")
            .auto_shrink([false, false])
            .show_rows(ui, row_height, matches.len(), |ui, range| {
                for &index in &matches[range] {
                    row_ui(ui, &entries[index], results, row_height, widths);
                }
            });
        action
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
    tooltip
}

/// One table row: label, value with unit, workbook cell, marker (the path is in the hover
/// text, which is built only while the row is hovered).
fn row_ui(ui: &mut egui::Ui, entry: &TableEntry, results: &DesignResults, height: f32) {
    let value = results.get(&entry.path).unwrap_or(Value::None);
    let row = ui.horizontal(|ui| {
        // Left-aligned columns of fixed width (add_sized would centre the text).
````

with:

````rust
    tooltip
}

/// One table row: label, value with unit, workbook cell, marker, in columns `widths` wide
/// ([`column_widths`]; the path is in the hover text, which is built only while the row is
/// hovered).
fn row_ui(
    ui: &mut egui::Ui,
    entry: &TableEntry,
    results: &DesignResults,
    height: f32,
    widths: [f32; 4],
) {
    let value = results.get(&entry.path).unwrap_or(Value::None);
    let row = ui.horizontal(|ui| {
        // Left-aligned columns of fixed width (add_sized would centre the text).
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
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
````

with:

````rust
                ui.add(egui::Label::new(text).truncate());
            });
        };
        let [label, number, workbook, marker] = widths;
        cell(ui, label, entry.info.meta.label);
        cell(
            ui,
            number,
            &with_unit(format_value(&value), entry.info.meta.unit),
        );
        cell(
            ui,
            workbook,
            entry.info.cell.as_deref().unwrap_or("Rust-only"),
        );
        cell(ui, marker, &entry.marker);
    });
    row.response.on_hover_ui(|ui| {
        ui.label(row_tooltip(entry, &value));
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib the_label_column_flexes_down_to_its_minimum 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 1 passed; 0 failed` for `the_label_column_flexes_down_to_its_minimum`, then `test result: ok. 349 passed; 0 failed` for the whole library (Task 0's 309 plus this plan's tests so far).

- [ ] **Step 5: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/results_table.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m42 commit -F - <<'EOF'
feat(magcoupling-rs): the results table's label column flexes for narrow windows

The value, cell and marker columns keep their M4-1 widths (130, 130, 60) and
the label takes the rest, at least 120 points (decision M42-8), so a ~930 px window shows each
row's label and value without scrolling (the M4-1 final review's item).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m42 status --short
```

Expected: one new commit on `magcoupling/m4-2`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 6: Docs, the web smoke and the plan

**Model:** `sonnet` (doc and workflow edits with the exact text, running the gate; CLAUDE.md section 5). Escalate a retry to the session model.

The repo rule: docs change with the code. The README's status, layout, features, the panel's views, versions, web smoke and tests; `docs/ai/02-system.yaml` (the three conflict hunks resolved, keeping both sides, plus M4-2's sentence, key files, two invariants, four lessons and the status line), `03-structure.yaml` (the gui modules, versions, gate 11), `04-memory.yaml` (the design checks and the M4-2 part of the open item resolved, M4-3's kept), `05-update-tracker.md`; the `gui-smoke` workflow judges the geometry view in its first screenshot (it is the default view, so no click is needed) and requires it. Then this plan is committed, the whole gate runs, and the web bundle is built and opened once. Step 1's `04-memory.yaml` entry states that the user confirmed the recommended options of M42-1 to M42-9 at Task 0 Step 8; if they did not, stop here and escalate (the tasks that implement another answer will have stopped already).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/.claude/workflows/gui-smoke.js`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/README.md`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/02-system.yaml`, `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/03-structure.yaml`, `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/04-memory.yaml`, `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/05-update-tracker.md`
- Create: `C:/Users/Cole/source/repos/lsim-mag-m42/docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md` (this plan)

**Interfaces:**
- Consumes: every earlier task's names, as the docs quote them.
- Produces: the docs; nothing in code.

- [ ] **Step 1: Edit the workflow and the docs**

In `C:/Users/Cole/source/repos/lsim-mag-m42/.claude/workflows/gui-smoke.js`, replace:

````js
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
````

with:

````js
}

const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'geometry_view'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    geometry_view: { type: 'boolean' },
    share_link_loaded: { type: 'boolean' },
    sizing_solved: { type: 'boolean' },
    design_file_loaded: { type: 'boolean' },
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/.claude/workflows/gui-smoke.js`, replace:

````js
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
````

with:

````js
}

// The magnetic coupling calculator: deterministic state through its share link, checked in
// the console (the page is one canvas). The geometry view is the centre region's default view,
// so the first screenshot shows it with no click. One click on the canvas, on the "Load design"
// button found in a screenshot, checks the web file picker (rfd's HTML overlay) and the file load.
let magcoupling = null
if (ARGS.magcoupling !== false) {
  const page = `${url}/magcoupling/?m=${MAGCOUPLING_SMOKE_PAYLOAD}`
  magcoupling = await agent(
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (inputs on the left, dashboard on the right, the geometry view in the middle) vs a blank page. geometry_view=true only if the middle region shows the geometry view: a row of tabs starting "Geometry", under it two drawings side by side (an end view of two rings of coloured magnet blocks inside a grey cup around a shaft, and a half side view with dashed space-claim lines), and a numbered list of dimension callouts under them ("1 Face gap ...", "4 Overall length: ...").
Then the design file picker, the only click on the canvas: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
In the screenshot, find the "Load design" button in the header row at the top of the page (between "Save design" and "Copy share link") and click its centre with browser_run_code_unsafe, code: async (page) => { await page.mouse.click(X, Y); } (X and Y in CSS pixels of the screenshot; the canvas fills the page). rfd shows its overlay (#rfd-overlay: a file input #rfd-input shown as a "Choose File" button, and the buttons "Ok" and "Cancel") and opens the browser's file chooser at once (the tool output reports a "File chooser" modal state); if a snapshot shows the overlay but no chooser opened, click the "Choose File" button. Upload the file with browser_file_upload, click the overlay's "Ok" button, wait 2 seconds and collect the console messages again; delete the file.
design_file_loaded=true only if a console message contains "magcoupling: loaded a design file". If the button cannot be found or the overlay does not appear after two attempts (each with a fresh screenshot), design_file_loaded=false and say why in notes: the canvas click is the fragile part of this step, and the share-link checks do not depend on it. Close the browser.
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at".
passed=true only if: page loaded, canvas present, geometry_view, share_link_loaded, sizing_solved, design_file_loaded, and zero console messages of type error (warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
  ) || { passed: false, canvas_present: false, console_errors: [], geometry_view: false, share_link_loaded: false, sizing_solved: false, design_file_loaded: false, notes: 'smoke agent returned no result (agent error)' }
}

// The linkage app's fields at the top level, as before, with the calculator's beside them.
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/02-system.yaml`, replace:

````yaml
      Addendum A-2 adds the harmonic set, the assumptions registry, inverse sizing and the space claim, with an axial housing that follows the length override.
      M4 infrastructure adds the egui panel (feature gui), the standalone app and its native and
      web binaries (feature app), and the second web bundle at /magcoupling/.
<<<<<<< HEAD
      Addendum A-3 adds the equation explorer's engine side (evaluable equation records proven by a drift guard, the dependency graph, the A3 traceability test and the A4 teaching notes).
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs, src/gui/panel.rs, src/app.rs]
=======
      M4-1 grows the panel: every input from the metadata, the dashboard, the results table,
      undo/redo, design files and share links, the sizing mode, the linkage app's theme.
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs, src/gui/panel.rs, src/gui/session.rs, src/gui/sizing.rs, src/gui/dashboard.rs, src/app.rs]
>>>>>>> magcoupling/m4-1
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
      - Every departure from the workbook is an approved, registered deviation (E1-E20 in deviations.rs; E15-E17 are probed on top of E9, decision 15). compute_all always applies Deviations::ALL; the switch exists only behind the test-only workbook-parity feature.
````

with:

````yaml
      Addendum A-2 adds the harmonic set, the assumptions registry, inverse sizing and the space claim, with an axial housing that follows the length override.
      M4 infrastructure adds the egui panel (feature gui), the standalone app and its native and
      web binaries (feature app), and the second web bundle at /magcoupling/.
      Addendum A-3 adds the equation explorer's engine side (evaluable equation records proven by a drift guard, the dependency graph, the A3 traceability test and the A4 teaching notes).
      M4-1 grows the panel with every input from the metadata, the dashboard, the results table,
      undo/redo, design files and share links, the sizing mode and the linkage app's theme.
      M4-2 adds the centre region's views, the geometry view to scale (the default), five plots
      (egui_plot) and the clamp drawing and table.
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs, src/gui/panel.rs, src/gui/session.rs, src/gui/sizing.rs, src/gui/dashboard.rs, src/gui/geometry.rs, src/gui/plots.rs, src/gui/clamp_drawing.rs, src/app.rs]
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
      - Every departure from the workbook is an approved, registered deviation (E1-E20 in deviations.rs; E15-E17 are probed on top of E9, decision 15). compute_all always applies Deviations::ALL; the switch exists only behind the test-only workbook-parity feature.
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/02-system.yaml`, replace:

````yaml
      - "The E7 peak search (model::peak_off_half_pitch) returns None exactly when half a pitch is the maximum (1e-12 relative), so every E7-neutral design keeps the workbook expression bit for bit; it finds every root of dT/dx in cos^2 x by recursion on derivatives (no closed form, no scan grid)."
      - "Inverse sizing (sizing::solve) evaluates compute_all only inside the free variable's slider range; a continuous variable's 64-cell scan is refined between samples at the first crossing, at each peak (golden section) and at each validity edge, so only a torque hump whose rise and fall both lie inside one cell is unseen (the documented limit). A value counts only if sizing::is_valid holds (the blocks fit, faceted blocks on their flats and arcs without overlapping, model::blocks_fit; the keyway leaves hub wall; f_end > 0, model::end_effect_in_range; a finite hot-low torque) and meets when its hot-low torque reaches the target; poles stay even."
      - "Library parts and gradeless manual magnets use Calibration C22 and the NdFeB density; only a grade-mode ring (manual dimensions with a grade) takes its grade's alpha(Br) and density (decision A2-7)."
      - "The engine stays pure std: egui and eframe are optional dependencies behind the gui and app features, and the no-feature wasm32 check (gate 6) proves it. egui, eframe and wasm-bindgen are locked at linkage-sim-rs's versions, and wasm-bindgen at deploy-web.yml's wasm-bindgen-cli pin (gate 11)."
      - "workbook-parity never reaches a shipped build. The shipped cargo arguments live once, in linkage-sim-rs/scripts/magcoupling_shipped.sh. build_magcoupling_web.sh and gate 10 read cargo's --message-format=json record of the compiled units, and they fail if a magcoupling-rs unit has the feature. Gate 10 also runs a negative control that must trip."
<<<<<<< HEAD
      - "The panel (gui::MagcouplingPanel) computes only through compute_all, with every correction on. It recomputes every frame after the inputs are drawn, so the readout shows the same frame's edits. Its sliders come from the input metadata and clamp edits only, so an idle frame never rewrites an input."
      - "Every equation record (src/engine/explain/records/) reproduces the engine's result by the parity rule at the defaults and at every differential case under every augmentation, corrections on, every value term of every record taking two values there (the drift guard, tests/explain.rs); the registry refuses a formula that reads a path in two units or would typeset two ways; the formula shown is the one evaluated (no custom evals); a record reads its terms' engine values and never recomputes upstream; a value the engine computed but did not expose becomes a Rust-only result (a read); every path of the explorer's scope (169) drills down to inputs."
      - "A3 traceability: nudging any input moves no explained result outside its dependency set (soundness; every input but the two no result reads changes some result at some design point), and each assumption moves every numeric result on its active path at some design point above rounding, with the algebraic cancellations listed with reasons (sensitivity)."
=======
      - "The panel (gui::MagcouplingPanel) computes only through compute_all, with every correction on. It recomputes every frame after the inputs are drawn, so the readout shows the same frame's edits. Its sliders come from the input metadata and clamp edits only, so an idle frame never rewrites an input; slider values are rounded to the step's decimals (decision M41-1)."
      - "Every edit goes through InputSet::set (rows, design files, share links); loading a file or a link is all or nothing and the refusal names every problem of the inputs and the sizing state (gui::session, decision M41-6). Files and links are one schema-versioned JSON format naming every input and the sizing state; a renamed or removed input path gets a session::PATH_MIGRATIONS entry and a DESIGN_VERSION bump, so older files and links keep opening."
      - "An edit is in progress while a pointer button is down, a key is held down (a held arrow key's auto-repeat run is one undo step) or a text field of an input row (or the target's value box) has focus; a focused text field elsewhere (the results search) is no edit. Undo steps and solves wait for it to end. A share link opened at start-up is the session's start (open_share_payload: no undo step)."
      - "sizing::solve never runs per frame: gui::sizing::SizingRunner solves once the design has been still for DEBOUNCE_S (0.25 s) and no edit is in progress. In Torque -> Magnets the panel shows the inputs with the free variable at the solved (or best) value; the inputs keep their own value until the mode is left, when a change still waiting for its debounce is solved at once (solve_now, one solve per click)."
      - "The panel does no I/O: saving and picking files are PanelRequests its host does (app::files natively and on the web; the linkage app in M5)."
      - "The corrected-vs-workbook markers come from the deviation registry (cells, changes at defaults, probes) and the compiled golden files (E3, E4, E5), never from computing with corrections off (gui::corrections)."
>>>>>>> magcoupling/m4-1

dataflow_on_user_change: |
  user edit -> state.blueprint mutation -> state.push_undo() -> state.rebuild()
````

with:

````yaml
      - "The E7 peak search (model::peak_off_half_pitch) returns None exactly when half a pitch is the maximum (1e-12 relative), so every E7-neutral design keeps the workbook expression bit for bit; it finds every root of dT/dx in cos^2 x by recursion on derivatives (no closed form, no scan grid)."
      - "Inverse sizing (sizing::solve) evaluates compute_all only inside the free variable's slider range; a continuous variable's 64-cell scan is refined between samples at the first crossing, at each peak (golden section) and at each validity edge, so only a torque hump whose rise and fall both lie inside one cell is unseen (the documented limit). A value counts only if sizing::is_valid holds (the blocks fit, faceted blocks on their flats and arcs without overlapping, model::blocks_fit; the keyway leaves hub wall; f_end > 0, model::end_effect_in_range; a finite hot-low torque) and meets when its hot-low torque reaches the target; poles stay even."
      - "Library parts and gradeless manual magnets use Calibration C22 and the NdFeB density; only a grade-mode ring (manual dimensions with a grade) takes its grade's alpha(Br) and density (decision A2-7)."
      - "The engine stays pure std: egui, egui_plot and eframe are optional dependencies behind the gui and app features, and the no-feature wasm32 check (gate 6) proves it. egui, egui_plot, eframe and wasm-bindgen are locked at linkage-sim-rs's versions, and wasm-bindgen at deploy-web.yml's wasm-bindgen-cli pin (gate 11)."
      - "workbook-parity never reaches a shipped build. The shipped cargo arguments live once, in linkage-sim-rs/scripts/magcoupling_shipped.sh. build_magcoupling_web.sh and gate 10 read cargo's --message-format=json record of the compiled units, and they fail if a magcoupling-rs unit has the feature. Gate 10 also runs a negative control that must trip."
      - "Every equation record (src/engine/explain/records/) reproduces the engine's result by the parity rule at the defaults and at every differential case under every augmentation, corrections on, every value term of every record taking two values there (the drift guard, tests/explain.rs); the registry refuses a formula that reads a path in two units or would typeset two ways; the formula shown is the one evaluated (no custom evals); a record reads its terms' engine values and never recomputes upstream; a value the engine computed but did not expose becomes a Rust-only result (a read); every path of the explorer's scope (169) drills down to inputs."
      - "A3 traceability: nudging any input moves no explained result outside its dependency set (soundness; every input but the two no result reads changes some result at some design point), and each assumption moves every numeric result on its active path at some design point above rounding, with the algebraic cancellations listed with reasons (sensitivity)."
      - "The panel (gui::MagcouplingPanel) computes only through compute_all, with every correction on. It recomputes every frame after the inputs are drawn, so the readout shows the same frame's edits. Its sliders come from the input metadata and clamp edits only, so an idle frame never rewrites an input; slider values are rounded to the step's decimals (decision M41-1)."
      - "Every edit goes through InputSet::set (rows, design files, share links); loading a file or a link is all or nothing and the refusal names every problem of the inputs and the sizing state (gui::session, decision M41-6). Files and links are one schema-versioned JSON format naming every input and the sizing state; a renamed or removed input path gets a session::PATH_MIGRATIONS entry and a DESIGN_VERSION bump, so older files and links keep opening."
      - "An edit is in progress while a pointer button is down, a key is held down (a held arrow key's auto-repeat run is one undo step) or a text field of an input row (or the target's value box) has focus; a focused text field elsewhere (the results search) is no edit. Undo steps and solves wait for it to end. A share link opened at start-up is the session's start (open_share_payload: no undo step)."
      - "sizing::solve never runs per frame: gui::sizing::SizingRunner solves once the design has been still for DEBOUNCE_S (0.25 s) and no edit is in progress. In Torque -> Magnets the panel shows the inputs with the free variable at the solved (or best) value; the inputs keep their own value until the mode is left, when a change still waiting for its debounce is solved at once (solve_now, one solve per click)."
      - "The panel does no I/O: saving and picking files are PanelRequests its host does (app::files natively and on the web; the linkage app in M5)."
      - "The corrected-vs-workbook markers come from the deviation registry (cells, changes at defaults, probes) and the compiled golden files (E3, E4, E5), never from computing with corrections off (gui::corrections)."
      - "The geometry view draws the design shown from its results only (gui::geometry): every dimension is a result, the axial ones the housing in effect (housing.*), so the drawing follows every edit, the length override and Torque -> Magnets; a piece holding a number that is not finite is not drawn, and blocks are drawn only for 2 to 200 poles. Every callout and clamp-table value hovers its result's text through dashboard::hover_text, the hook M4-3 joins."
      - "The plot tabs build their series from the same frame's results through the engine's own closed forms (model::ring_pair_factor for torque against temperature, the amp*_Pa amplitudes for torque against rotation), which tests pin to the engine's torques; only the tab shown builds its series, and a point that is not finite is left out."

dataflow_on_user_change: |
  user edit -> state.blueprint mutation -> state.push_undo() -> state.rebuild()
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/02-system.yaml`, replace:

````yaml
  - "A headless test that opens several CollapsingHeaders by clicking their labels must click bottom up: a header above that opens animates the rows below over the next frames, so a click aimed at a later label can land on a slider (magcoupling idle_frames_change_no_input)."
  - "egui's default fonts have no U+2192 (the arrow) or U+2264: it draws an empty box. The magcoupling panel writes 'Torque -> Magnets' and '<=', and a test checks every drawn and hover text for glyphs (every_text_the_panel_shows_has_glyphs_in_the_default_fonts)."
  - "A panel shown with show_inside sizes from the previous frame: text added below its first-frame height is painted from the next frame on. After work done late in a frame (the magcoupling sizing solve), request a repaint so the parts drawn earlier catch up on an idle page."
  - "Windows MAX_PATH: a release build under a deep directory fails with LNK1104 (cannot open a build-script exe). Build from a short path (the worktree, or subst a drive letter)."

current_statuses:
````

with:

````yaml
  - "A headless test that opens several CollapsingHeaders by clicking their labels must click bottom up: a header above that opens animates the rows below over the next frames, so a click aimed at a later label can land on a slider (magcoupling idle_frames_change_no_input)."
  - "egui's default fonts have no U+2192 (the arrow) or U+2264: it draws an empty box. The magcoupling panel writes 'Torque -> Magnets' and '<=', and a test checks every drawn and hover text for glyphs (every_text_the_panel_shows_has_glyphs_in_the_default_fonts)."
  - "A panel shown with show_inside sizes from the previous frame: text added below its first-frame height is painted from the next frame on. After work done late in a frame (the magcoupling sizing solve), request a repaint so the parts drawn earlier catch up on an idle page."
  - "egui's horizontal_wrapped lays a row out from the last frame's widths, so a wrapped row settles on the second frame; a headless test that clicks a wrapped tab runs one more frame first (magcoupling a_narrow_window_shows_each_row_s_label_and_value_without_scrolling)."
  - "f32::clamp(min, max) panics when min > max. A size bounded by two limits that cross at narrow widths uses min and max instead (magcoupling clamp_ui; its test draws the tab 294 and 120 points wide)."
  - "f32::min and f64::max pass over NaN (they return the other operand), so a scale taken with min over an extent that is not finite comes out finite; check the extents for finiteness first (magcoupling geometry_view::side_by_side)."
  - "egui_plot 0.33 paints a solid Line of n >= 2 points as one Shape::Path of n points and each Points marker as a Shape::Circle of its radius, so a headless test counts a plot's series by point count, in a bare frame where no other widget paints paths (magcoupling plots tests)."
  - "Windows MAX_PATH: a release build under a deep directory fails with LNK1104 (cannot open a build-script exe). Build from a short path (the worktree, or subst a drive letter)."

current_statuses:
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/02-system.yaml`, replace:

````yaml
  dxf_import: works native + WASM via drag-drop; selection-based conversions
  actuator_workflow: two-pass solve gives correct reactions with actuator loads
  non_grashof_sweep: NaN-padded across unreachable angles; playback ping-pongs
<<<<<<< HEAD
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim. M4 infrastructure in place (gui panel tracer, app binaries, /magcoupling/ bundle, workbook-parity guard, gates 7-11). Addendum A-3 complete: equation explorer engine side, A3 traceability, A4 notes; next: the M4 GUI plans"
=======
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim. M4 infrastructure in place (gui panel, app binaries, /magcoupling/ bundle, workbook-parity guard, gates 7-11). M4-1 complete (inputs, dashboard, results table, session, sizing mode, theme); next: Addendum A-3 (explanations), M4-2 (geometry view, plots) and M4-3 (equation explorer, assumptions panel, notes, pickers)"
>>>>>>> magcoupling/m4-1
````

with:

````yaml
  dxf_import: works native + WASM via drag-drop; selection-based conversions
  actuator_workflow: two-pass solve gives correct reactions with actuator loads
  non_grashof_sweep: NaN-padded across unreachable angles; playback ping-pongs
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim. M4 infrastructure in place (gui panel, app binaries, /magcoupling/ bundle, workbook-parity guard, gates 7-11). Addendum A-3 complete: equation explorer engine side, A3 traceability, A4 notes. M4-1 complete (inputs, dashboard, results table, session, sizing mode, theme). M4-2 complete (geometry view, five plots, clamp drawing and table, narrow results table); next: M4-3 (equation explorer, assumptions panel, notes, pickers), then M3 (live 3D fields)"
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/03-structure.yaml`, replace:

````yaml

magcoupling_rs:
  path: magcoupling-rs/ (repo root, sibling of linkage-sim-rs/; separate crate, NOT a workspace member)
  role: Rust port of reference/magcoupling-py (spec docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md); M2 complete - every Python module except fields3d (M3) is ported, workbook parity covers all 1,149 checks, E1-E14 applied; Addendum A-1 (grades, materials, E15-E20), A-2 (harmonic set, assumptions, inverse sizing, space claim) and A-3 (equation explorer engine side - evaluable records, drift guard, A3 traceability, A4 notes) complete; M4 infrastructure adds the egui panel, the standalone app binaries and the /magcoupling/ web bundle; M4-1 adds the inputs, dashboard, results table, session (undo/redo, design files, share links), sizing mode and theme
  guide: magcoupling-rs/README.md (tests, regenerating data, porting a module, Python-to-Rust translation rules, applying deviations)
  lib: magcoupling (src/lib.rs re-exports compute_all, headline, DesignInputs, DesignResults; mod gui behind feature gui, mod app behind feature app)
  engine: src/engine/ (pure std, wasm32-clean; one module per Python module)
````

with:

````yaml

magcoupling_rs:
  path: magcoupling-rs/ (repo root, sibling of linkage-sim-rs/; separate crate, NOT a workspace member)
  role: Rust port of reference/magcoupling-py (spec docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md); M2 complete - every Python module except fields3d (M3) is ported, workbook parity covers all 1,149 checks, E1-E14 applied; Addendum A-1 (grades, materials, E15-E20), A-2 (harmonic set, assumptions, inverse sizing, space claim) and A-3 (equation explorer engine side - evaluable records, drift guard, A3 traceability, A4 notes) complete; M4 infrastructure adds the egui panel, the standalone app binaries and the /magcoupling/ web bundle; M4-1 adds the inputs, dashboard, results table, session (undo/redo, design files, share links), sizing mode and theme; M4-2 adds the geometry view, five plots (egui_plot) and the clamp drawing and table
  guide: magcoupling-rs/README.md (tests, regenerating data, porting a module, Python-to-Rust translation rules, applying deviations)
  lib: magcoupling (src/lib.rs re-exports compute_all, headline, DesignInputs, DesignResults; mod gui behind feature gui, mod app behind feature app)
  engine: src/engine/ (pure std, wasm32-clean; one module per Python module)
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/03-structure.yaml`, replace:

````yaml
    addendum_only: [assumptions.rs, sizing.rs, housing.rs, explain/]
    remaining: [fields3d (M3)]
  features:
    gui: "src/gui/ (egui 0.32, serde_json, base64, flate2, log; no eframe, no I/O): panel.rs (MagcouplingPanel: header with Undo/Redo/Reset all/Save design/Load design/Copy share link, inputs SidePanel, dashboard SidePanel, CentreView (results table); fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; PanelRequest (SaveFile, OpenDesign) drained by take_requests; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; sizing controls and run_sizing), inputs.rs (InputCatalogue::get, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS, step_decimals, outside_range, text_hint, input_tooltip), input_ui.rs (input_row, slider, RowEdit), dashboard.rs (DASHBOARD, Level, verdict_level, END_EFFECT_ROWS, STORED_3D_ROWS, result_info, result_tooltip (the hover hook), dashboard_lines, end_effect_banner), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip), results_table.rs (table_entries, search, exact_number, results_csv, results_json, ResultsTable, row_tooltip), session.rs (Design, design_to_json/design_from_json, encode/decode_share_payload, share_link, LoadError, PATH_MIGRATIONS, json_number), history.rs (History, MAX_UNDO_LEVELS 100), sizing.rs (SizingMode, SizingState, variable_key/label, SizingRunner (update, solve_now), DEBOUNCE_S 0.25), format.rs (format_value, with_unit, non_finite_text), test_support.rs (cfg(test): sized_frame(_at), SCREEN, key_event, key_tap, select_all, primary_button, drawn_texts, text_rect, short_magnets)"
    app: "src/app.rs (MagcouplingApp::new applies the theme; ui drains the panel's requests; open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (cad_dark_visuals, a copy of linkage-sim-rs's checked body for body; apply forces ThemePreference::Dark), src/app/files.rs (save: rfd dialog natively, Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
    workbook_parity: "test-only, enabled by a self dev-dependency; never in a shipped build (linkage-sim-rs/scripts/magcoupling_shipped.sh, gate 10)"
    versions: "Cargo.toml caret ranges (egui 0.32, eframe 0.32, wasm-bindgen 0.2); Cargo.lock pins linkage-sim-rs/Cargo.lock's egui/eframe 0.32.3 and wasm-bindgen 0.2.114 (= the deploy-web.yml wasm-bindgen-cli pin); gate 11 checks"
  web_bundle: "linkage-sim-rs/web/magcoupling/ (index.html committed, canvas magcoupling_canvas, absolute /magcoupling/ glue path; magcoupling-web.js and magcoupling-web_bg.wasm gitignored in web/.gitignore), built by linkage-sim-rs/scripts/build_magcoupling_web.sh (called by build_web.sh and by deploy-web.yml), served by serve_web.sh [PORT] at /magcoupling/; vercel.json revalidates /magcoupling/magcoupling-web.js"
  shipped_builds: "linkage-sim-rs/scripts/magcoupling_shipped.sh (sourced by build_magcoupling_web.sh and gate.sh): MAGCOUPLING_WEB_ARGS, MAGCOUPLING_NATIVE_ARGS, magcoupling_assert_shipped (reads cargo --message-format=json; fails unless the shipped bin was compiled and no magcoupling-rs unit has workbook-parity)"
  tests: tests/ (the integration tests parity.rs, differential.rs, python_schema.rs, schema.rs, deviations.rs, static_data.rs, robustness.rs, grades.rs, material_library.rs, material_links.rs, assumptions.rs, sizing.rs, explain.rs (the drift guard, the registry, the scope, the A3 traceability and the A4 note links); common/mod.rs holds the shared differential Case loader and the PORTED_INPUTS / PORTED_RESULTS ratchets)
  test_data: tests/data/ (reference_values.json = copy of the vendored snapshot; input_schema.json via MAGCOUPLING_BLESS=1 cargo test --test schema; python_schema.json, static_data.json, differential/<group>.json (10 result groups plus helpers.json) and differential/full.json via tools/gen_differential.py; deviations/E3.json, E4.json and E5.json, the golden files of the broad corrections, via MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells)
  gate: "linkage-sim-rs/scripts/gate.sh gates 4-12: 4-6 the engine (test, clippy -D warnings, wasm32 check); 7 cargo test --features app (panel and app headless tests, bins built with workbook-parity unified); 8 clippy --all-targets --features app -D warnings; 9 wasm32 clippy -D warnings (gui lib, magcoupling-web); 10 workbook-parity guard (native and wasm32 shipped builds, negative control); 11 lock parity (egui, eframe, wasm-bindgen; deploy-web.yml CLI pin); 12 Python parity suite + gen_differential.py --check; --full adds 13, build_web.sh (both bundles)"

modules:
  core:
````

with:

````yaml
    addendum_only: [assumptions.rs, sizing.rs, housing.rs, explain/]
    remaining: [fields3d (M3)]
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log; no eframe, no I/O): panel.rs (MagcouplingPanel: header with Undo/Redo/Reset all/Save design/Load design/Copy share link, inputs SidePanel, dashboard SidePanel, CentreView (Geometry, the default; Plot(PlotKind) x5; Clamp; Results) under the end-effect banner; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; PanelRequest (SaveFile, OpenDesign) drained by take_requests; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; sizing controls and run_sizing), inputs.rs (InputCatalogue::get, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS, step_decimals, outside_range, text_hint, input_tooltip), input_ui.rs (input_row, slider, RowEdit), dashboard.rs (DASHBOARD, Level, verdict_level, END_EFFECT_ROWS, STORED_3D_ROWS, result_info, result_tooltip (the hover hook), hover_text, dashboard_lines, end_effect_banner), geometry.rs (geometry -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); DESIGN_CHECK, AUTOFIT, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout; Transform; side_by_side (views at one scale, None without room); arrowhead; part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names), clamp_drawing.rs (clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip), results_table.rs (table_entries, search, exact_number, results_csv, results_json, ResultsTable, row_tooltip, column_widths), session.rs (Design, design_to_json/design_from_json, encode/decode_share_payload, share_link, LoadError, PATH_MIGRATIONS, json_number), history.rs (History, MAX_UNDO_LEVELS 100), sizing.rs (SizingMode, SizingState, variable_key/label, SizingRunner (update, solve_now), DEBOUNCE_S 0.25), format.rs (format_value, with_unit, non_finite_text), test_support.rs (cfg(test): sized_frame(_at), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, text_rect, text_rects, text_color, short_magnets)"
    app: "src/app.rs (MagcouplingApp::new applies the theme; ui drains the panel's requests; open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (cad_dark_visuals, a copy of linkage-sim-rs's checked body for body; apply forces ThemePreference::Dark), src/app/files.rs (save: rfd dialog natively, Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
    workbook_parity: "test-only, enabled by a self dev-dependency; never in a shipped build (linkage-sim-rs/scripts/magcoupling_shipped.sh, gate 10)"
    versions: "Cargo.toml caret ranges (egui 0.32, egui_plot 0.33, eframe 0.32, wasm-bindgen 0.2); Cargo.lock pins linkage-sim-rs/Cargo.lock's egui/eframe 0.32.3, egui_plot 0.33.0 and wasm-bindgen 0.2.114 (= the deploy-web.yml wasm-bindgen-cli pin); gate 11 checks"
  web_bundle: "linkage-sim-rs/web/magcoupling/ (index.html committed, canvas magcoupling_canvas, absolute /magcoupling/ glue path; magcoupling-web.js and magcoupling-web_bg.wasm gitignored in web/.gitignore), built by linkage-sim-rs/scripts/build_magcoupling_web.sh (called by build_web.sh and by deploy-web.yml), served by serve_web.sh [PORT] at /magcoupling/; vercel.json revalidates /magcoupling/magcoupling-web.js"
  shipped_builds: "linkage-sim-rs/scripts/magcoupling_shipped.sh (sourced by build_magcoupling_web.sh and gate.sh): MAGCOUPLING_WEB_ARGS, MAGCOUPLING_NATIVE_ARGS, magcoupling_assert_shipped (reads cargo --message-format=json; fails unless the shipped bin was compiled and no magcoupling-rs unit has workbook-parity)"
  tests: tests/ (the integration tests parity.rs, differential.rs, python_schema.rs, schema.rs, deviations.rs, static_data.rs, robustness.rs, grades.rs, material_library.rs, material_links.rs, assumptions.rs, sizing.rs, explain.rs (the drift guard, the registry, the scope, the A3 traceability and the A4 note links); common/mod.rs holds the shared differential Case loader and the PORTED_INPUTS / PORTED_RESULTS ratchets)
  test_data: tests/data/ (reference_values.json = copy of the vendored snapshot; input_schema.json via MAGCOUPLING_BLESS=1 cargo test --test schema; python_schema.json, static_data.json, differential/<group>.json (10 result groups plus helpers.json) and differential/full.json via tools/gen_differential.py; deviations/E3.json, E4.json and E5.json, the golden files of the broad corrections, via MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells)
  gate: "linkage-sim-rs/scripts/gate.sh gates 4-12: 4-6 the engine (test, clippy -D warnings, wasm32 check); 7 cargo test --features app (panel and app headless tests, bins built with workbook-parity unified); 8 clippy --all-targets --features app -D warnings; 9 wasm32 clippy -D warnings (gui lib, magcoupling-web); 10 workbook-parity guard (native and wasm32 shipped builds, negative control); 11 lock parity (egui, egui_plot, eframe, wasm-bindgen; deploy-web.yml CLI pin); 12 Python parity suite + gen_differential.py --check; --full adds 13, build_web.sh (both bundles)"

modules:
  core:
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/04-memory.yaml`, replace:

````yaml
  - "RESOLVED 2026-10-01 (user): the E20 rating-calibration offset is clamped at 0 as a new registered correction E21 (never raise a grade's onsets above the model), verified M1-style and applied only after the user approves its numbers (after A-3 merges, since A-3 records explain the demag block)"
  - "RESOLVED 2026-10-01 (user): ferrite, SmCo and bonded grades get sourced per-grade magnet CTE and specific heat (cited, skeptic-checked), registered as corrections that move only non-NdFeB designs; same verification-then-approval path as E21"
  - "RESOLVED 2026-10-01 (user): all 10 A-4 verification decisions take option A (docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md section 5; data in ...-a4-data.json): E21 calibration clamp with the record fixes (C43, C47, probe 4); E24 cure margin = MIN over rings under E20, landed with E21; E22 inner-ring grade CTE in the bond plane and E23 each ring's grade specific heat as separate corrections; values Y30 10e-6 / 700, Recoma 20 14e-6 / 370, Recoma 26 and 30 13e-6 (Arnold) / 350, bonded keeps C95 with cp 420 flagged single-source; |shear| comparisons under E22; C95/C138 keep labels with help text and Rust-only in-effect fields; the bond screen's C12 swing documented in C101 help"
  - "RESOLVED 2026-10-01 (user): the two workbook design inconsistencies (the M41 cap thread below the 41.33 mm cup body OD; the 22 mm vs 25 mm boss OD on Metal design vs Shaft clamps) are flagged as design-check notes in the GUI (M4-2), with no engine change"
  - "M3 must apply E3 and E5 inside fields3d: rerun it with the corrected N42SH Br (E3) and multiply the web integral by 4 in fields3d.run (E5); see each entry's corrected_formula in magcoupling-rs/src/engine/deviations.rs"
  - "RESOLVED (Addendum A-2, decision 29): one general E7 peak search for any odd set up to 11 (every root of dT/dx in cos^2 x by recursion on derivatives), and the harmonic set is the Rust-only assumption coupling.max_harmonic (default 5 = 1, 3, 5) summed by the Calculator, both circuit sums, the sweeps and the Calibration prototype; the three cancellation tests and the fill-0.4 test pass unedited, and the closed form is the tests' reference (within 1e-12 rad)"
  - "RESOLVED (Addendum A-1): the per-grade demagnetization fix is E20 (decision 19): each ring is checked with its own Hcj and beta, Br and rating and the weaker ring governs (A-1 decision A13; the workbook read only the inner ring against the outer blocks' fields), C44/C45 as overrides through the Rust-only temperature.demag.coercivity_source; ferrite (positive beta) is limited on the cold side (hot onsets +inf, rating as hot limit, Rust-only cold onsets/limit/check in the verdict, both rings); tests e20_* incl. the ferrite cold case through the custom-dimension mode and the mixed rings"
````

with:

````yaml
  - "RESOLVED 2026-10-01 (user): the E20 rating-calibration offset is clamped at 0 as a new registered correction E21 (never raise a grade's onsets above the model), verified M1-style and applied only after the user approves its numbers (after A-3 merges, since A-3 records explain the demag block)"
  - "RESOLVED 2026-10-01 (user): ferrite, SmCo and bonded grades get sourced per-grade magnet CTE and specific heat (cited, skeptic-checked), registered as corrections that move only non-NdFeB designs; same verification-then-approval path as E21"
  - "RESOLVED 2026-10-01 (user): all 10 A-4 verification decisions take option A (docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md section 5; data in ...-a4-data.json): E21 calibration clamp with the record fixes (C43, C47, probe 4); E24 cure margin = MIN over rings under E20, landed with E21; E22 inner-ring grade CTE in the bond plane and E23 each ring's grade specific heat as separate corrections; values Y30 10e-6 / 700, Recoma 20 14e-6 / 370, Recoma 26 and 30 13e-6 (Arnold) / 350, bonded keeps C95 with cp 420 flagged single-source; |shear| comparisons under E22; C95/C138 keep labels with help text and Rust-only in-effect fields; the bond screen's C12 swing documented in C101 help"
  - "RESOLVED 2026-10-01 (user): the two workbook design inconsistencies (the M41 cap thread below the 41.33 mm cup body OD; the 22 mm vs 25 mm boss OD on Metal design vs Shaft clamps) are flagged as design-check notes in the GUI (M4-2), with no engine change (done in M4-2: gui::geometry's DESIGN_CHECK notes under the geometry view, each while its inconsistency holds)"
  - "M3 must apply E3 and E5 inside fields3d: rerun it with the corrected N42SH Br (E3) and multiply the web integral by 4 in fields3d.run (E5); see each entry's corrected_formula in magcoupling-rs/src/engine/deviations.rs"
  - "RESOLVED (Addendum A-2, decision 29): one general E7 peak search for any odd set up to 11 (every root of dT/dx in cos^2 x by recursion on derivatives), and the harmonic set is the Rust-only assumption coupling.max_harmonic (default 5 = 1, 3, 5) summed by the Calculator, both circuit sums, the sweeps and the Calibration prototype; the three cancellation tests and the fill-0.4 test pass unedited, and the closed form is the tests' reference (within 1e-12 rad)"
  - "RESOLVED (Addendum A-1): the per-grade demagnetization fix is E20 (decision 19): each ring is checked with its own Hcj and beta, Br and rating and the weaker ring governs (A-1 decision A13; the workbook read only the inner ring against the outer blocks' fields), C44/C45 as overrides through the Rust-only temperature.demag.coercivity_source; ferrite (positive beta) is limited on the cold side (hot onsets +inf, rating as hot limit, Rust-only cold onsets/limit/check in the verdict, both rings); tests e20_* incl. the ferrite cold case through the custom-dimension mode and the mixed rings"
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/04-memory.yaml`, replace:

````yaml
  - "Tidy after M2: magcoupling-rs/src/engine/temperature.rs is about 1,350 lines, past the architecture's ~800-line split point; split mechanically (inputs, results, compute, tests) when no correction work is in flight"
  - "Open, not in Addendum A-2's scope (the A3 panel lists no bondline assumption): E4 as approved adds the inner bondline input to the pole-sweep keyed-wall term, while the block-fit term keeps the workbook literal + 0.05 (the same bondline), so at bond_inner != 0.05 the two terms of the max() disagree (sweeps.rs pole_sweep); E4's test and golden file probe only 0.05. Owner: a deviation proposal (plan A-3 explains the engine as it is and did not change E4) (changing E4 is a registered deviation and needs the user's approval); trigger: making the pole-sweep block-fit bondline an input, or any change to E4"
  - "RESOLVED (Addendum A-2, audit M9): the engine flags f_end <= 0 (model.end_effect_check, calibration.end_effect_check, model::end_effect_in_range for sweep rows); M4 greys the numbers computed from the pull-out; inverse sizing never counts such a value"
  - "M4 design questions (decision 28): the M41 cap thread sits below the 41.33 mm cup body OD; the boss OD is 22 mm on Metal design and 25 mm on Shaft clamps. M4 draws the axial housing in effect (housing.hub_length_mm, cup_depth_mm, retainer_span_mm: A-2 decision A2-8)"
  - "RESOLVED (M4-1, decision M41-4): design files and share links record the sizing mode, free variable and target torque (gui::session \"sizing\")"
  - "M4 performance (deferred from the A-2 final review, not a defect): model::shear_stress evaluates all six harmonics (amplitudes and sinh/exp geometry factors of 7, 9 and 11 too) on every call, in the Calculator and each of the 19 sweep rows, even when coupling.max_harmonic sums only 1, 3, 5; with the general root search it makes up compute_all's 8.5 -> 17 us release cost (measured in the A-2 plan), about 50x inside the spec's milliseconds. If M4's per-frame budget (drawing, A-3 equation records) needs it, compute only the harmonics summed, keeping the bits of 1, 3, 5 and a left-out harmonic's reported 0"
  - "RESOLVED (Addendum A-3): the dependency-graph traceability test (tests/explain.rs: soundness for every input, sensitivity for the 15 assumption inputs, each reaching an explained result, with the cancellations listed with reasons); the records include tau7_Pa-tau11_Pa and the Calibration sums (model.tau#_Pa, calibration.tau#_Pa families)"
````

with:

````yaml
  - "Tidy after M2: magcoupling-rs/src/engine/temperature.rs is about 1,350 lines, past the architecture's ~800-line split point; split mechanically (inputs, results, compute, tests) when no correction work is in flight"
  - "Open, not in Addendum A-2's scope (the A3 panel lists no bondline assumption): E4 as approved adds the inner bondline input to the pole-sweep keyed-wall term, while the block-fit term keeps the workbook literal + 0.05 (the same bondline), so at bond_inner != 0.05 the two terms of the max() disagree (sweeps.rs pole_sweep); E4's test and golden file probe only 0.05. Owner: a deviation proposal (plan A-3 explains the engine as it is and did not change E4) (changing E4 is a registered deviation and needs the user's approval); trigger: making the pole-sweep block-fit bondline an input, or any change to E4"
  - "RESOLVED (Addendum A-2, audit M9): the engine flags f_end <= 0 (model.end_effect_check, calibration.end_effect_check, model::end_effect_in_range for sweep rows); M4 greys the numbers computed from the pull-out; inverse sizing never counts such a value"
  - "RESOLVED (M4-2, decision 28): the geometry view draws the axial housing in effect (housing.hub_length_mm, cup_depth_mm, retainer_span_mm: A-2 decision A2-8); the M41 cap thread below the 41.33 mm cup body OD and the 22 mm vs 25 mm boss OD show as design-check notes while they hold"
  - "RESOLVED (M4-1, decision M41-4): design files and share links record the sizing mode, free variable and target torque (gui::session \"sizing\")"
  - "M4 performance (deferred from the A-2 final review, not a defect): model::shear_stress evaluates all six harmonics (amplitudes and sinh/exp geometry factors of 7, 9 and 11 too) on every call, in the Calculator and each of the 19 sweep rows, even when coupling.max_harmonic sums only 1, 3, 5; with the general root search it makes up compute_all's 8.5 -> 17 us release cost (measured in the A-2 plan), about 50x inside the spec's milliseconds. If M4's per-frame budget (drawing, A-3 equation records) needs it, compute only the harmonics summed, keeping the bits of 1, 3, 5 and a left-out harmonic's reported 0"
  - "RESOLVED (Addendum A-3): the dependency-graph traceability test (tests/explain.rs: soundness for every input, sensitivity for the 15 assumption inputs, each reaching an explained result, with the cancellations listed with reasons); the records include tau7_Pa-tau11_Pa and the Calibration sums (model.tau#_Pa, calibration.tau#_Pa families)"
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/04-memory.yaml`, replace:

````yaml
  - "RESOLVED (M4 infrastructure, magcoupling/m4-infra): the workbook-parity guard checks cargo's --message-format=json record of the shipped builds (linkage-sim-rs/scripts/magcoupling_shipped.sh, magcoupling_assert_shipped). build_magcoupling_web.sh pipes its release build through it, and gate 10 runs it on the native and wasm32 shipped builds plus a negative control that must trip. It trips when the feature is forced on the command line or through app in Cargo.toml, and cargo test and clippy --all-targets with app stay green. M5 must also run it on linkage-sim-rs's shipped builds once linkage-sim-rs depends on magcoupling-rs (the function already accepts any binary; the magcoupling-rs units are matched by package id)"
  - "RESOLVED (M4-1): the Key design 'axial length' is the A-2 override coupling.magnets.axial_length_mm, blank by default, entered at the inner ring's length in use (decision M41-12); slider values are rounded to the step's decimals (M41-1); edits, typed values included, are clamped to the slider range and a value already outside it is kept and flagged (M41-2); gui-smoke opens /magcoupling/ through a pinned share link; the standalone app forces the linkage app's CAD dark theme (M41-3)"
  - "Open (M4-1 follow-ups): magcoupling-web_bg.wasm is 4.0 MB with the default release profile after M4-1 (3.5 MB before; rfd, serde_json, flate2); a size profile (opt-level, LTO) or wasm-opt is a later option. The linkage web app itself still renders light in a light-preference browser (backlog BL-013): ctx.set_theme(egui::ThemePreference::Dark), as magcoupling's app/theme.rs does, is the likely fix"
  - "Open (M4-2, M4-3): per-row end-effect greying of the results table waits for plan A-3's dependency graph (M4-1 shows the banner over the table and greys the pinned dashboard rows); M4-3 may re-source the corrected-vs-workbook markers from A-3's per-record upstream corrections; result_tooltip (gui/dashboard.rs) is the hover hook the equation tooltip joins; when it does, dashboard_ui must build its hover text lazily in on_hover_ui, as the results table's row_ui does (dashboard_lines builds all 17 rows' tooltips on every frame, negligible while they are a few lines: M4-1 final review); the centre region's CentreView takes M4-2's geometry view and plot tabs, and M4-2, whose tabs share that width, reworks the results table's fixed columns (row_ui: 260/130/130/60 points beside the 320- and 300-point sides, so a ~930 px window shows only the label column without scrolling; the 1280 x 800 native default fits: M4-1 final review)"
  - "Open (M5, from the M4-1 review): linkage-sim-rs/src/gui/mod.rs reads Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y with a non-consuming key_pressed, which runs whatever the panel consumes. Hosting MagcouplingPanel in an egui::Window, M5 must skip its own undo while the calculator window has the user's attention and turn the panel's shortcuts on only then (MagcouplingPanel::set_keyboard_shortcuts), so one key press never undoes both"
  - "Deploy (M4-1): nothing pushed. deploy-web.yml on main already builds and ships /magcoupling/ on the next push of main; that push needs the user's explicit go"
````

with:

````yaml
  - "RESOLVED (M4 infrastructure, magcoupling/m4-infra): the workbook-parity guard checks cargo's --message-format=json record of the shipped builds (linkage-sim-rs/scripts/magcoupling_shipped.sh, magcoupling_assert_shipped). build_magcoupling_web.sh pipes its release build through it, and gate 10 runs it on the native and wasm32 shipped builds plus a negative control that must trip. It trips when the feature is forced on the command line or through app in Cargo.toml, and cargo test and clippy --all-targets with app stay green. M5 must also run it on linkage-sim-rs's shipped builds once linkage-sim-rs depends on magcoupling-rs (the function already accepts any binary; the magcoupling-rs units are matched by package id)"
  - "RESOLVED (M4-1): the Key design 'axial length' is the A-2 override coupling.magnets.axial_length_mm, blank by default, entered at the inner ring's length in use (decision M41-12); slider values are rounded to the step's decimals (M41-1); edits, typed values included, are clamped to the slider range and a value already outside it is kept and flagged (M41-2); gui-smoke opens /magcoupling/ through a pinned share link; the standalone app forces the linkage app's CAD dark theme (M41-3)"
  - "Open (M4-1 follow-ups): magcoupling-web_bg.wasm is 4.0 MB with the default release profile after M4-1 (3.5 MB before; rfd, serde_json, flate2); a size profile (opt-level, LTO) or wasm-opt is a later option. The linkage web app itself still renders light in a light-preference browser (backlog BL-013): ctx.set_theme(egui::ThemePreference::Dark), as magcoupling's app/theme.rs does, is the likely fix"
  - "RESOLVED (M4-2): decisions M42-1 to M42-9 as recommended in docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md (confirmed by the user at its Task 0); the centre region's CentreView holds the geometry view (the default), the five plot tabs, the clamp tab and the results table, one tab row; the results table's label column flexes (results_table::column_widths, at least 120 points), so a ~930 px window shows each row's label and value; egui_plot is locked at linkage-sim-rs's 0.33.0 and gate 11 checks it; the end-effect banner sits over the dashboard and the centre region"
  - "Open (M4-3, carried from M4-1): per-row end-effect greying of the results table waits for plan A-3's dependency graph (M4-1 shows the banner over the table and greys the pinned dashboard rows); M4-3 may re-source the corrected-vs-workbook markers from A-3's per-record upstream corrections; result_tooltip (gui/dashboard.rs) is the hover hook the equation tooltip joins; when it does, dashboard_ui must build its hover text lazily in on_hover_ui, as the results table's row_ui and M4-2's geometry callouts and clamp table do (dashboard_lines builds all 17 rows' tooltips on every frame, negligible while they are a few lines: M4-1 final review); M4-2's readouts already go through dashboard::hover_text"
  - "Open (M5, from the M4-1 review): linkage-sim-rs/src/gui/mod.rs reads Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y with a non-consuming key_pressed, which runs whatever the panel consumes. Hosting MagcouplingPanel in an egui::Window, M5 must skip its own undo while the calculator window has the user's attention and turn the panel's shortcuts on only then (MagcouplingPanel::set_keyboard_shortcuts), so one key press never undoes both"
  - "Deploy (M4-1): nothing pushed. deploy-web.yml on main already builds and ships /magcoupling/ on the next push of main; that push needs the user's explicit go"
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/05-update-tracker.md`, replace:

````markdown
Reverse chronological (newest at top).

---

## 2026-10-01 — Magcoupling M4-1: final review fix wave
- The whole-branch review found no Critical or Important issue; no code changes.
````

with:

````markdown
Reverse chronological (newest at top).

---

## 2026-10-01 — Magcoupling M4-2: geometry view, plots and clamp drawing (branch magcoupling/m4-2)
- `magcoupling-rs` panel: the centre region is one tab row (`CentreView`): the geometry view (the
  default), five plots, the clamp and the results table, under the end-effect banner when
  f_end <= 0 (decisions M42-1, M42-2).
- Geometry view (`gui/geometry.rs`, `gui/geometry_view.rs`): the end view (cup and pocket, both rings
  of blocks on the flats or as arcs, liner, sleeve, hub, shaft, key) and the upper half side view
  (cap, cup wall, web, boss, and the parts in the cavity, centred), both at one scale, drawn from the
  results of the design shown with the housing in effect, so they follow every edit, the length
  override and Torque -> Magnets. Tagged dimension callouts listed under the views: face gap, corner
  gap (along face 1's normal, from the circle the inner corners sweep to the outer flat), running
  clearance, and the space claim per axis; red when violated (a gap below zero, the
  clearance below its target, an axis exceeded), amber when not a number (M42-3, M42-4). The space
  claim dashed; the autofit wall as a hint; the two workbook design checks (cap thread below the cup
  body OD, boss OD 22 vs 25 mm) as notes while they hold (flag only, no engine change; M42-9). Hovering a
  dimension shows its result's hover text (`dashboard::hover_text`, the M4-3 hook). A claim line
  farther than ten times the pieces' extent (a design file's finite but huge claim) is left off
  with a note, and parts holding a number that is not finite are counted in a note.
- Plots (`gui/plots.rs`, egui_plot 0.33.0, locked to linkage-sim-rs's and added to gate 11):
  torque vs temperature with the variation band, the requirement and the limit; the gap and pole
  sweeps by row status, greyed outside the end-effect model; slip heating vs the governing limit;
  torque vs rotation with the pull-out point. Series from the same frame's results through the
  engine's closed forms, pinned by tests to the engine's torques; the torque plots grey when
  f_end <= 0 (M42-5, M42-6).
- Clamp (`gui/clamp_drawing.rs`): a painter port of drawing.py's end and top views, its texts and
  limits, cuts clipped to the boss, drawing.py's message when no screw fits; the clamp table
  (summary, machining steps, the screw sizes table with the recommended column) (M42-7).
- Results table: the label column flexes (`column_widths`; the value, cell and marker columns keep
  M4-1's 130, 130 and 60 points), so a ~930 px window shows the label and the value (M42-8). `gui-smoke` judges the geometry view in its first screenshot (no click).
- `docs/ai/02-system.yaml`: the three unresolved merge-conflict hunks the M4-1 merge (42ab370) left
  in the magcoupling block and its status line are resolved, keeping both sides (A-3 and M4-1); the
  M4-1 sentence is reworded so the block parses as YAML.
- The engine, parity, differential data and registry are unchanged. Nothing pushed.

## 2026-10-01 — Magcoupling M4-1: final review fix wave
- The whole-branch review found no Critical or Important issue; no code changes.
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/README.md`, replace:

````markdown
dashboard (badges, corrected-vs-workbook markers, the end-effect greying, the stored-3D label,
the space claim), the searchable results table with CSV and JSON export, undo and redo, design
files and share links, the sizing mode switch and the linkage app's theme (see
[The panel](#the-panel); decisions M41-1 to M41-15). Next: M4-2
(geometry view, plots, clamp drawing) and M4-3 (equation explorer, assumptions panel, teaching
notes, material and grade pickers).

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
````

with:

````markdown
dashboard (badges, corrected-vs-workbook markers, the end-effect greying, the stored-3D label,
the space claim), the searchable results table with CSV and JSON export, undo and redo, design
files and share links, the sizing mode switch and the linkage app's theme (see
[The panel](#the-panel); decisions M41-1 to M41-15). M4-2 complete: the centre region's views,
the geometry view to scale (the default: both rings, the housing in effect, the dimension
callouts, the space claim and the design checks), five plots (egui_plot) and the clamp drawing
and table, and a results table that fits a narrow window (decisions M42-1 to M42-9). Next: M4-3
(equation explorer, assumptions panel, teaching notes, material and grade pickers).

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/README.md`, replace:

````markdown
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/engine/explain/` | Addendum A2 equation explorer, engine side (plan A-3): the formula markup (`markup.rs`: grammar, parser, parse tree), its evaluator with a trace of what a result depends on (`eval.rs`), the static tables a formula reads (`tables.rs`: magnets, grades, the part materials, aluminium alloys, adhesives, screw sizes), the authoring form (`record.rs`) and the records, one file per chain (`records/`), the registry (`registry.rs`: `Registry::build`, `equation_for`, `used_by`, the dependency graph, `term_rows`; each result's upstream corrections are precomputed at build, which a unit test checks against a walk of the graph), display symbols (`symbols.rs`), the v1 scope (`scope.rs`, decision 31) and a plain-text rendering (`render.rs`) |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, the sizing controls), `inputs.rs` (`InputCatalogue`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type), `dashboard.rs` (`DASHBOARD`, `verdict_level`, `END_EFFECT_ROWS`, `STORED_3D_ROWS`, `result_info`, `result_tooltip`, the hover hook), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `search`, `results_csv`, `results_json`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`), `history.rs` (`History`: undo and redo), `sizing.rs` (`SizingMode`, `SizingState`, `SizingRunner`: debounced inverse sizing), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included), `test_support.rs` (headless egui helpers, tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the linkage app's CAD dark visuals, forced dark), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
````

with:

````markdown
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/engine/explain/` | Addendum A2 equation explorer, engine side (plan A-3): the formula markup (`markup.rs`: grammar, parser, parse tree), its evaluator with a trace of what a result depends on (`eval.rs`), the static tables a formula reads (`tables.rs`: magnets, grades, the part materials, aluminium alloys, adhesives, screw sizes), the authoring form (`record.rs`) and the records, one file per chain (`records/`), the registry (`registry.rs`: `Registry::build`, `equation_for`, `used_by`, the dependency graph, `term_rows`; each result's upstream corrections are precomputed at build, which a unit test checks against a walk of the graph), display symbols (`symbols.rs`), the v1 scope (`scope.rs`, decision 31) and a plain-text rendering (`render.rs`) |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, the sizing controls), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list, `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type), `dashboard.rs` (`DASHBOARD`, `verdict_level`, `END_EFFECT_ROWS`, `STORED_3D_ROWS`, `result_info`, `result_tooltip`, the hover hook, and `hover_text`, its text with the correction marks), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `search`, `results_csv`, `results_json`, `column_widths`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`), `history.rs` (`History`: undo and redo), `sizing.rs` (`SizingMode`, `SizingState`, `SizingRunner`: debounced inverse sizing), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included), `test_support.rs` (headless egui helpers, tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the linkage app's CAD dark visuals, forced dark), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/README.md`, replace:

````markdown
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
````

with:

````markdown
| Feature | Adds | Dependencies (all optional) |
|---|---|---|
| none (default) | The engine | none: pure std, builds for wasm32 as it is |
| `gui` | `gui::MagcouplingPanel`: the design inputs and results, and `fn ui(&mut self, ui: &mut egui::Ui)`. Any egui app can host it: the standalone app as a full page, the linkage app in an `egui::Window` (M5) | egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log |
| `app` | `app::MagcouplingApp` and the binaries below | `gui`, eframe 0.32, log, rfd 0.15; env_logger (native); wasm-bindgen, wasm-bindgen-futures, web-sys, js-sys (wasm32) |
| `workbook-parity` | **Test-only**: the switch that turns corrections off (see [Differences from the workbook](#differences-from-the-workbook)) | none |

### The panel

M4-1 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md`)
and M4-2 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md`).
A header with the session buttons; the inputs on the left; the dashboard on the right; the
centre region between them (`CentreView`, one tab row: the geometry view, the five plots, the
clamp and the results table; decisions M42-1 and M42-2), under the end-effect banner when
f_end <= 0. Every result is recomputed with `compute_all` (every correction on) each frame after the
inputs are drawn, so the readouts show the same frame's edits.

- **Inputs** (`inputs.rs`, `input_ui.rs`): every input, generated from its metadata and grouped as
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/README.md`, replace:

````markdown
  marker; search by label, path or cell; CSV (`path,label,value,unit,cell`) and JSON (with the
  design that produced the results: in Torque -> Magnets the inputs with the free variable at the
  value shown) at full precision, a non-finite number written `+inf`, `-inf` or `NaN` (M41-15).
- **Session** (`session.rs`, `history.rs`): undo and redo (buttons, Ctrl+Z, Ctrl+Shift+Z, Ctrl+Y)
  of every change to the design, one step per settled edit (a drag, a typed value, a part name
  typed letter by letter), one per arrow nudge and one for a held arrow key's whole auto-repeat
````

with:

````markdown
  marker; search by label, path or cell; CSV (`path,label,value,unit,cell`) and JSON (with the
  design that produced the results: in Torque -> Magnets the inputs with the free variable at the
  value shown) at full precision, a non-finite number written `+inf`, `-inf` or `NaN` (M41-15).
  The label column flexes (`column_widths`, M42-8: whatever the value, cell and marker columns
  leave, at least 120 points), so a ~930 px window shows each row's label and value without
  scrolling.
- **Geometry view** (`geometry.rs`, `geometry_view.rs`; the default view, M42-2): the end view
  (the cup and its pocket, the outer blocks, the liner, the sleeve, the inner blocks on their
  flats or as arcs, the hub, the shaft and the key) and the upper half side view through the flat
  centres (cap, cup wall, web and boss; inside the cavity, each centred on it, the outer and inner
  blocks, the liner, the sleeve and the hub), both at one scale (M42-3), from the results of the
  design shown: the axial dimensions are the housing in effect (`housing.*`, A-2 decision A2-8),
  so a length-sized design grows. Tagged dimension callouts, listed under the views: the face
  gap, the corner gap (along face 1's normal, from the dashed circle its inner corners sweep to
  the outer block's flat), the running clearance (with the sleeve-to-liner gap and the movement), and
  per axis the space claim (the axial stack against the overall length, the large-diameter stack
  against the bay, the rotating OD against the diameter). A gap below zero, the clearance below
  its target and an exceeded axis draw red, a value that is not a number amber (M42-4). The space
  claim is dashed (the diameter as a circle in the end view; the overall length, the bay and the
  diameter in the side view), an exceeded axis red. Under the callouts: the autofit wall (the
  wall rule's `materials.cup_wall_suggested_mm`, unless there is no back iron) and the two
  design checks while they hold (the cap thread below the cup body OD; the boss OD differing
  between Metal design and Shaft clamps: flagged only, no engine change; M42-9). Hovering a dimension
  or its text shows its result's hover text (`dashboard::hover_text`, the hook M4-3 joins). A
  piece holding a number that is not finite is not drawn (a note counts them per view), a claim
  line farther than `CLAIM_REACH` (10) times the pieces' extent is left off with a note (a design
  file can hold any finite number), and blocks are drawn for 2 to 200 poles.
- **Plots** (`plots.rs`, egui_plot 0.33; M42-5, M42-6): torque against magnet temperature (the pull-out,
  the band of the variation allowance, the required minimum, the operating temperature and the
  governing limit), the gap and pole sweeps (each row by status: green nominal, amber below the
  hot minimum, red no fit, grey outside the end-effect model; the design shown as a marker; the
  required floor; the pole sweep's line is named for its rows' smallest fitting apothem, so the
  design's own marker can sit off it), slip heating over time (the estimate and the high case
  against the governing limit, the time to the limit marked) and torque against relative rotation
  over one pole pair (the pull-out point at the E7 angle). The series come from this frame's
  results through the engine's own closed forms (tests pin them to the engine's torques), only
  for the tab shown; when f_end <= 0 the torque plots are grey. Points and reference lines that
  are not finite are left out.
- **Clamp** (`clamp_drawing.rs`; M42-7): drawing.py's end and top views of the one-piece slotted
  clamp for the recommended screw (its texts, coordinates and limits; the cuts clipped to the
  boss; the screws needed, at most 50), both at one scale in the dark theme's pens, or
  drawing.py's message when no screw size
  fits; then the clamp table: the Shaft clamps summary, the machining steps and the 'Clamp screw
  sizes' table with the recommended size's column in green, each value with its hover text.
- **Session** (`session.rs`, `history.rs`): undo and redo (buttons, Ctrl+Z, Ctrl+Shift+Z, Ctrl+Y)
  of every change to the design, one step per settled edit (a drag, a typed value, a part name
  typed letter by letter), one per arrow nudge and one for a held arrow key's whole auto-repeat
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/README.md`, replace:

````markdown
**Versions.** egui and eframe use the same 0.32 line as `linkage-sim-rs`, so
M5 embeds the panel with one egui; `Cargo.toml` has caret ranges, and
`Cargo.lock` pins the versions of `linkage-sim-rs/Cargo.lock` (egui and eframe
0.32.3, wasm-bindgen 0.2.114, which is the `wasm-bindgen-cli` version
`deploy-web.yml` installs). Gate 11 fails when the two lock files or the CLI pin
disagree; bump all three together.

| Binary | Target | Run |
|---|---|---|
````

with:

````markdown
**Versions.** egui and eframe use the same 0.32 line as `linkage-sim-rs`, so
M5 embeds the panel with one egui; `Cargo.toml` has caret ranges, and
`Cargo.lock` pins the versions of `linkage-sim-rs/Cargo.lock` (egui and eframe
0.32.3, egui_plot 0.33.0, wasm-bindgen 0.2.114, which is the `wasm-bindgen-cli`
version `deploy-web.yml` installs). Gate 11 fails when the two lock files or the CLI
pin disagree; bump them together.

| Binary | Target | Run |
|---|---|---|
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/README.md`, replace:

````markdown
`build_magcoupling_web.sh` after the linkage build, so both bundles ship
(`linkage.colesorkness.com/magcoupling/`) on the next push of `main`; pushing needs the user's
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it opens
`/magcoupling/` through a pinned share link and checks the canvas, the console lines
`magcoupling: loaded the design from the share link` and `magcoupling sizing: Solved at`, then
clicks "Load design" once (found in a screenshot), uploads a pinned design file through rfd's web
picker and checks `magcoupling: loaded a design file`, with zero console errors (a test decodes
````

with:

````markdown
`build_magcoupling_web.sh` after the linkage build, so both bundles ship
(`linkage.colesorkness.com/magcoupling/`) on the next push of `main`; pushing needs the user's
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it opens
`/magcoupling/` through a pinned share link and checks the canvas, the geometry view in the first
screenshot (the default view: no click), the console lines
`magcoupling: loaded the design from the share link` and `magcoupling sizing: Solved at`, then
clicks "Load design" once (found in a screenshot), uploads a pinned design file through rfd's web
picker and checks `magcoupling: loaded a design file`, with zero console errors (a test decodes
````

In `C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/README.md`, replace:

````markdown
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300, with the default back iron and with none, where E17's free-space field inputs act (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a harmonic set outside its choices (NaN torques in the Calculator, the Calibration prototype and every sweep row, never another set, with the corrections off and on; `validate()` names it: `an_invalid_harmonic_set_is_nan_not_another_set`), a coercivity source outside its choices (any code but 1 uses the Hcj and beta inputs, and `validate()` names it: `an_invalid_coercivity_source_uses_the_inputs`), a positive beta typed in for magnets with no grade and no rating (no hot limit: C60 = +inf, C61 = NaN, the adhesive governs C12: `a_positive_beta_without_a_rating_has_no_hot_limit`), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), inverse sizing on designs no slider reaches (an outcome or an error for every free variable, never a panic: `sizing_never_panics_on_extreme_designs`), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells (hand-listed, or in the correction's golden file under `tests/data/deviations/`); a correction that changes more than 15 cells uses a golden file and one that changes 15 or fewer lists them (`broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`); each applied correction's figures match the report (`e1_adhesive_shear_modulus_matches_the_report`, `e2_clamp_screw_length_matches_the_report`, `e3_library_remanence_matches_the_report`), and help a correction rewords belongs to a real input and differs from the workbook's. A correction that changes nothing at defaults (E7 to E13, E15 to E20) carries registry probes, off-default inputs from the report whose cells are checked with the correction off and on (`each_probe_shows_its_correction`); `every_applied_engine_correction_is_visible_somewhere` requires every applied engine correction to show at defaults or in a probe. A probe's workbook value can be a workbook error (`Literal::Error`, e.g. `#DIV/0!` where Python raises): that side still runs but is not compared. E7, E8, E9, E10, E11, E12 and E13 must leave every default cell bit for bit (`e7_leaves_every_default_cell_bit_for_bit`, `e8_leaves_flat_blocks_bit_for_bit`, `e9_leaves_every_default_cell_bit_for_bit`, `e10_leaves_every_default_cell_bit_for_bit`, `e11_leaves_every_default_cell_bit_for_bit`, `e12_leaves_every_default_cell_bit_for_bit`, `e13_leaves_every_default_cell_bit_for_bit`). `e7_finds_the_peak_at_a_fill_of_exactly_0_4` pins E7 where the fifth harmonic vanishes (a ring at a fill of exactly 0.4). The Addendum A entries name every cell their probes change, downstream cells included (`addendum_entries_name_every_cell_their_probes_change`); E15 to E18 have rows in the Addendum A report and E19 and E20 cite decisions only (`e15_to_e18_have_audit_rows_and_e19_e20_decisions_only`); each reproduces the report: `e15_heat_capacity_matches_the_report`, `e16_removed_disc_matches_the_report`, `e17_aluminium_eddy_losses_match_the_report` and `e15_to_e17_together_match_the_reports_headline_table` (on top of E9, decision 15), `e18_aluminium_hub_mismatch_matches_the_report`, `e19_supermagnetman_arcs_follow_the_vendor_grid`, and for E20 `e20_each_part_uses_its_own_coercivity`, `e20_the_hcj_and_beta_inputs_override_the_grade_when_selected`, `e20_ferrite_is_limited_on_the_cold_side`, `e20_mixed_rings_use_the_weaker_grade` and `e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature`; `e15_leaves_every_default_cell_bit_for_bit` to `e20_leaves_every_default_cell_bit_for_bit` pin that none moves a default cell. `all_corrections_together_give_the_reviewed_headline` pins what users see: the headline with every correction on at the default design. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input, a 1280 x 1024 screen): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame; the value lands on the step's decimals (1.41) and stepping back lands on the default exactly; a typed value snaps to the step and is clamped into the range; pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept until edited and flagged; the changed dot and per-field reset; the axial length override starts blank and enters at the ring's length; a selector switches the branch (no back iron); a text input edits the part name and its note follows; every group opens and draws every input; the dashboard's stored-3D label, markers, end-effect banner (over the dashboard and the table) and space-claim overshoot; the results table's search and its export requests; undo and redo (buttons and shortcuts; one step per drag, per nudge, per typed name; Ctrl+Z in a text field left to the field; reset all undoable); save and load requests, a loaded and a refused design file, the share link on the clipboard and a broken link; the host's report; Torque -> Magnets: the solved design shown, the debounce (no solve per frame, none while a button is down, a repaint after a solve), the locked free-variable row (also under the status line for the ring radius), leaving the mode keeps the value as one undo step and solves a pending change first, an unreachable target shows the best value, the free-variable picker, a share link carries the sizing state, the JSON export holds the design shown, typing in the results search does not defer the solve; a held arrow key is one undo step, a slider's value box being typed in is an edit of the design (one step on Enter), the results search holds back no undo step, only this frame's rows count as design fields, a start-up share link is no undo step, a host can keep Ctrl+Z for itself; idle frames change nothing with every group open and off-grid values; the measured drag enters at the model's drag, unrounded; every drawn text and hover text has glyphs in egui's default fonts; the panel draws inside an `egui::Window` (M5) |
| `src/gui/inputs.rs`, `src/gui/dashboard.rs`, `src/gui/corrections.rs`, `src/gui/results_table.rs` (feature `gui`) | Every input in exactly one section in schema order, every heading used, the Key design list, every input type covered; step decimals; every slider default on its step grid except the listed two; the optional seeds (entering the axial length at its seed moves nothing); text hints; tooltips. The dashboard starts with `HEADLINE`; every verdict each check gives has a level (designs that reach each branch); `END_EFFECT_ROWS` are exactly the headline rows that move with c_end (0 included: it flips the hot-minimum verdict) at back iron 1 and 0, `STORED_3D_ROWS` exactly those that move with the 14 stored 3D inputs at their slider ends; greying drops badges. The compiled golden files are the registry's; the headline pull-out carries E3 with the workbook's 2.647. The table lists every result once; the search; exact numbers read back bit for bit; CSV quoting; `+inf` and `NaN` in CSV and JSON (E20's positive-beta design); the JSON export holds the design |
| `src/gui/session.rs`, `src/gui/history.rs`, `src/gui/sizing.rs` (feature `gui`) | A design file and a share link round-trip bit for bit (every input type, the sizing state); a file names every input; a missing path or sizing state takes the default; every problem reported, nothing loaded, the inputs' and the sizing state's together; an older file's renamed and removed paths migrate (a made-up table) and `PATH_MIGRATIONS` leads to inputs of this version; not a design, a newer or malformed version, a malformed sizing state, a broken link, a link that inflates past 1 MB are refused; the default link stays under 2,500 characters. Undo and redo walk settled steps, coalesce an unsettled edit, drop redo on a new change, cap at 100. The runner waits for the debounce and runs once, restarts on a new change or an edit in progress, ignores the free variable's own value, solves a pending change at once on request; solved, unreachable and refused outcomes and their status lines |
| `src/gui/format.rs` (feature `gui`) | Four significant digits, scientific outside 1e-3 to 1e6, carries (9.99996 shows as 10.00), signed zero, `+inf`, `-inf`, `NaN` (one text for the display and both exports), integers, text, None, units |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the CAD dark theme; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the visuals are `linkage-sim-rs`'s, function body for body; the theme stays dark when the system turns light |
````

with:

````markdown
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300, with the default back iron and with none, where E17's free-space field inputs act (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a harmonic set outside its choices (NaN torques in the Calculator, the Calibration prototype and every sweep row, never another set, with the corrections off and on; `validate()` names it: `an_invalid_harmonic_set_is_nan_not_another_set`), a coercivity source outside its choices (any code but 1 uses the Hcj and beta inputs, and `validate()` names it: `an_invalid_coercivity_source_uses_the_inputs`), a positive beta typed in for magnets with no grade and no rating (no hot limit: C60 = +inf, C61 = NaN, the adhesive governs C12: `a_positive_beta_without_a_rating_has_no_hot_limit`), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), inverse sizing on designs no slider reaches (an outcome or an error for every free variable, never a panic: `sizing_never_panics_on_extreme_designs`), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells (hand-listed, or in the correction's golden file under `tests/data/deviations/`); a correction that changes more than 15 cells uses a golden file and one that changes 15 or fewer lists them (`broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`); each applied correction's figures match the report (`e1_adhesive_shear_modulus_matches_the_report`, `e2_clamp_screw_length_matches_the_report`, `e3_library_remanence_matches_the_report`), and help a correction rewords belongs to a real input and differs from the workbook's. A correction that changes nothing at defaults (E7 to E13, E15 to E20) carries registry probes, off-default inputs from the report whose cells are checked with the correction off and on (`each_probe_shows_its_correction`); `every_applied_engine_correction_is_visible_somewhere` requires every applied engine correction to show at defaults or in a probe. A probe's workbook value can be a workbook error (`Literal::Error`, e.g. `#DIV/0!` where Python raises): that side still runs but is not compared. E7, E8, E9, E10, E11, E12 and E13 must leave every default cell bit for bit (`e7_leaves_every_default_cell_bit_for_bit`, `e8_leaves_flat_blocks_bit_for_bit`, `e9_leaves_every_default_cell_bit_for_bit`, `e10_leaves_every_default_cell_bit_for_bit`, `e11_leaves_every_default_cell_bit_for_bit`, `e12_leaves_every_default_cell_bit_for_bit`, `e13_leaves_every_default_cell_bit_for_bit`). `e7_finds_the_peak_at_a_fill_of_exactly_0_4` pins E7 where the fifth harmonic vanishes (a ring at a fill of exactly 0.4). The Addendum A entries name every cell their probes change, downstream cells included (`addendum_entries_name_every_cell_their_probes_change`); E15 to E18 have rows in the Addendum A report and E19 and E20 cite decisions only (`e15_to_e18_have_audit_rows_and_e19_e20_decisions_only`); each reproduces the report: `e15_heat_capacity_matches_the_report`, `e16_removed_disc_matches_the_report`, `e17_aluminium_eddy_losses_match_the_report` and `e15_to_e17_together_match_the_reports_headline_table` (on top of E9, decision 15), `e18_aluminium_hub_mismatch_matches_the_report`, `e19_supermagnetman_arcs_follow_the_vendor_grid`, and for E20 `e20_each_part_uses_its_own_coercivity`, `e20_the_hcj_and_beta_inputs_override_the_grade_when_selected`, `e20_ferrite_is_limited_on_the_cold_side`, `e20_mixed_rings_use_the_weaker_grade` and `e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature`; `e15_leaves_every_default_cell_bit_for_bit` to `e20_leaves_every_default_cell_bit_for_bit` pin that none moves a default cell. `all_corrections_together_give_the_reviewed_headline` pins what users see: the headline with every correction on at the default design. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input, a 1280 x 1024 screen): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame; the value lands on the step's decimals (1.41) and stepping back lands on the default exactly; a typed value snaps to the step and is clamped into the range; pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept until edited and flagged; the changed dot and per-field reset; the axial length override starts blank and enters at the ring's length; a selector switches the branch (no back iron); a text input edits the part name and its note follows; every group opens and draws every input; the dashboard's stored-3D label, markers, end-effect banner (over the dashboard and the centre region) and space-claim overshoot; the results table's search and its export requests; undo and redo (buttons and shortcuts; one step per drag, per nudge, per typed name; Ctrl+Z in a text field left to the field; reset all undoable); save and load requests, a loaded and a refused design file, the share link on the clipboard and a broken link; the host's report; Torque -> Magnets: the solved design shown, the debounce (no solve per frame, none while a button is down, a repaint after a solve), the locked free-variable row (also under the status line for the ring radius), leaving the mode keeps the value as one undo step and solves a pending change first, an unreachable target shows the best value, the free-variable picker, a share link carries the sizing state, the JSON export holds the design shown, typing in the results search does not defer the solve; a held arrow key is one undo step, a slider's value box being typed in is an edit of the design (one step on Enter), the results search holds back no undo step, only this frame's rows count as design fields, a start-up share link is no undo step, a host can keep Ctrl+Z for itself; idle frames change nothing with every group open and off-grid values; the measured drag enters at the model's drag, unrounded; every drawn text and hover text has glyphs in egui's default fonts (every tab of the centre region and the geometry callouts' hover texts); the panel draws inside an `egui::Window` (M5); the geometry view is the default and follows the design shown (a live redraw; the sized design in Torque -> Magnets); each plot tab draws its plot, the design's marker at the edited pull-out in the edit's frame; the clamp tab draws the recommended clamp or drawing.py's message; a ~930 px window shows a row's label (filling its 120-point column) and value |
| `src/gui/geometry.rs`, `src/gui/geometry_view.rs` (feature `gui`) | The end view draws every part from the results (the pocket's corners at the pocket corner radius, the rings at the retainers' diameters, the key from where it crosses the bore), faceted blocks on their flats and arcs as sectors; each end-view callout's line measures the gap it names (the corner gap along face 1's normal, from the dashed circle the inner corners sweep to the outer flat); red below zero and for a clearance below its target; the side view spans the housing in effect (a 50.8 mm override: the 53.6 mm cavity, the hub centred on it, the liner and sleeve the retainer span in effect); the space-claim callouts name each overshoot, in red, with the exceeded axes' dashed lines red (every axis at once too), exactly at the claim inside and the next number below over, and a NaN claim amber; a claim far past the pieces (1e300, 1e39, -1e39) is left off with one note, drawn at `CLAIM_REACH` times the extent and not past it; the design checks show while each inconsistency holds (equality is not below); the autofit hint unless there is no back iron; pole counts from a file draw at most 200 blocks; a NaN draws nothing wrong and says so (the parts dropped counted per view). Painted: both views at one scale, the largest that fits (`side_by_side`: none without room or for an extent that is not finite); the cup OD, the shaft and the face gap at their pixel lengths, the cup depth and the cap as painted rectangles; the callout texts in their colours (the default clearance red, its line too); an overshoot red; hovering a dimension shows its result's hover text; a NaN view and a tiny screen draw; a region of no size paints no view; a huge claim keeps the pieces to scale with its note |
| `src/gui/plots.rs` (feature `gui`) | The torque-temperature curve meets the engine's pull-out at the operating temperature and at 20 °C, its band edges the hot-low and cold-high torques (bit for bit), a grade ring's own alpha too; sweep rows by status and end-effect range (4, 2, 7 and 1, 1, 4 at the defaults; short magnets grey every row); slip heating rises to the steady temperatures and marks the time to the limit on the curve; the torque-rotation curve peaks at the pull-out (harmonics 1 to 11 too; an invalid set draws none); f_end <= 0 greys the torque plots; each plot paints its series with their point counts and legends (an all-greyed sweep: its markers and no line; the pole sweep's line named for its smallest fitting apothems); values that are not numbers draw without panicking |
| `src/gui/clamp_drawing.rs` (feature `gui`) | `fmt_g` matches Python's `:g`; clipping keeps what is inside the boss; the default clamp has drawing.py's texts, the boss and bore circles, every cut inside the boss and one screw axis; no fitting screw (and an index outside the table) gives drawing.py's message; drawing.py's screws needed are drawn whatever fits, at most 50; the screw table's rows follow the sheet; the tab paints both views at one scale, the summary, the machining steps and the screw table, drawing.py's message, the one-piece note, and a narrow or tiny region without panicking, and no room with no scale |
| `src/gui/inputs.rs`, `src/gui/dashboard.rs`, `src/gui/corrections.rs`, `src/gui/results_table.rs` (feature `gui`) | Every input in exactly one section in schema order, every heading used, the Key design list, every input type covered; step decimals; every slider default on its step grid except the listed two; the optional seeds (entering the axial length at its seed moves nothing); text hints; tooltips. The dashboard starts with `HEADLINE`; every verdict each check gives has a level (designs that reach each branch); `END_EFFECT_ROWS` are exactly the headline rows that move with c_end (0 included: it flips the hot-minimum verdict) at back iron 1 and 0, `STORED_3D_ROWS` exactly those that move with the 14 stored 3D inputs at their slider ends; greying drops badges. The compiled golden files are the registry's; the headline pull-out carries E3 with the workbook's 2.647. The table lists every result once; the search; exact numbers read back bit for bit; CSV quoting; `+inf` and `NaN` in CSV and JSON (E20's positive-beta design); the JSON export holds the design; the label column flexes down to its minimum; the hover text carries the correction marks |
| `src/gui/session.rs`, `src/gui/history.rs`, `src/gui/sizing.rs` (feature `gui`) | A design file and a share link round-trip bit for bit (every input type, the sizing state); a file names every input; a missing path or sizing state takes the default; every problem reported, nothing loaded, the inputs' and the sizing state's together; an older file's renamed and removed paths migrate (a made-up table) and `PATH_MIGRATIONS` leads to inputs of this version; not a design, a newer or malformed version, a malformed sizing state, a broken link, a link that inflates past 1 MB are refused; the default link stays under 2,500 characters. Undo and redo walk settled steps, coalesce an unsettled edit, drop redo on a new change, cap at 100. The runner waits for the debounce and runs once, restarts on a new change or an edit in progress, ignores the free variable's own value, solves a pending change at once on request; solved, unreachable and refused outcomes and their status lines |
| `src/gui/format.rs` (feature `gui`) | Four significant digits, scientific outside 1e-3 to 1e6, carries (9.99996 shows as 10.00), signed zero, `+inf`, `-inf`, `NaN` (one text for the display and both exports), integers, text, None, units |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the CAD dark theme; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the visuals are `linkage-sim-rs`'s, function body for body; the theme stays dark when the system turns light |
````

- [ ] **Step 2: Check the docs**

Run:

```bash
grep -c "^<<<<<<<\|^=======\|^>>>>>>>" C:/Users/Cole/source/repos/lsim-mag-m42/docs/ai/02-system.yaml
cd C:/Users/Cole/source/repos/lsim-mag-m42 && python - <<'PYEOF'
import yaml
for name in ("03-structure", "04-memory", "backlog"):
    yaml.safe_load(open(f"docs/ai/{name}.yaml", encoding="utf-8"))
    print(name, "parses")
s = open("docs/ai/02-system.yaml", encoding="utf-8").read()
block = s[s.index("  magcoupling:\n    path: magcoupling-rs/"):s.index("\ndataflow_on_user_change")]
yaml.safe_load("major_subsystems:\n" + block)
statuses = s[s.index("current_statuses:"):].split("\n\n")[0]
yaml.safe_load(statuses)
print("02-system magcoupling block and statuses parse")
PYEOF
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --features app --lib the_smoke_test_share_link_opens_its_design 2>&1 | grep -E "test result"
```

Expected: `0` (no conflict marker left); `03-structure parses`, `04-memory parses`, `backlog parses`, `02-system magcoupling block and statuses parse`; `test result: ok. 1 passed`.

- [ ] **Step 3: Copy this plan into the repository**

The controller has this plan's file (written to `C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/m42-plan/plan.md`, or wherever the controller saved this plan; if that file is gone, ask the controller for it rather than fail); copy it to `C:/Users/Cole/source/repos/lsim-mag-m42/docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md`.

Run:

```bash
cp C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/m42-plan/plan.md C:/Users/Cole/source/repos/lsim-mag-m42/docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md
head -1 C:/Users/Cole/source/repos/lsim-mag-m42/docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md
```

Expected: `# Magcoupling M4-2: Geometry View, Plots and Clamp Drawing Implementation Plan`.

- [ ] **Step 4: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m42/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target/gate-task6.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target/gate-task6.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/target/gate-task6.log
git -C C:/Users/Cole/source/repos/lsim-mag-m42 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m42/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines as in Task 0 Step 7; the `grep -c` prints `0`; `cargo fmt --check` prints nothing.

- [ ] **Step 5: Build the web bundle and open it**

Run:

```bash
bash C:/Users/Cole/source/repos/lsim-mag-m42/linkage-sim-rs/scripts/build_magcoupling_web.sh 2>&1 | tail -6
```

Expected: `parity guard: magcoupling-web builds without workbook-parity; magcoupling-rs features: "features":["app","default","gui"]`, `Generating JS bindings...`, a blank line, `Magcoupling build complete!`, `web/magcoupling/magcoupling-web.js       (JS glue)` and `web/magcoupling/magcoupling-web_bg.wasm  (4.4M)` (4.0 MB before M4-2; M4-2, egui_plot included, adds about 0.4 MB). Then serve `C:/Users/Cole/source/repos/lsim-mag-m42/linkage-sim-rs/web/` (`python -m http.server 8080` from that folder, in the background) and run the `gui-smoke` workflow with `{"magcoupling": true}` (or by hand with the Playwright MCP tools, as its prompt says, the design-file click included): the page `/magcoupling/?m=<MAGCOUPLING_SMOKE_PAYLOAD>` shows the dark panel with the geometry view in the middle (the sized design: "4 Overall length: axial stack 33.28 mm of 35.00 mm" in the callout list), the console shows the share-link and `Solved at 14.18 mm` lines and zero errors. Stop the server afterwards.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m42 add .claude/workflows/gui-smoke.js magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md
git -C C:/Users/Cole/source/repos/lsim-mag-m42 commit -F - <<'EOF'
docs(magcoupling): M4-2 status, the panel's views, invariants and open items

The README's status, layout, features, the geometry view, the plots, the clamp,
versions (egui_plot 0.33.0), the web smoke and the tests; docs/ai: the three
merge-conflict hunks main's 02-system.yaml kept from the M4-1 merge resolved
(both sides kept), M4-2's invariants and lessons, the structure, the design
checks and the M4-2 part of the open item resolved, the tracker. gui-smoke
judges the geometry view, the default tab, in its first screenshot. The plan.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m42 status --short
```

Expected: one new commit on `magcoupling/m4-2`; `status --short` prints nothing (the web bundle's js and wasm are gitignored).

---

## Self-review record

This revision answers the critic's review of the first version: no blocking defects (every anchor matched once, both views share one scale and read the housing in effect, `compute_all` still runs once per frame, egui_plot is locked at linkage-sim-rs's 0.33.0, the views cannot write back, no test reads a source file, every new test asserts), and 23 non-blocking findings on edge cases, DRY, test strength, the execution steps and wording. Every finding is fixed; none was declined. The code was changed in the scratch tree's task commits (each task rebuilt on the one before and its tests, clippy and fmt run before the next), the plan regenerated from those commits, and the whole plan replayed again on a fresh export with a fresh target directory (the Verification record above). The critic's line numbers refer to the first version; the last column names the places in this one.

| # | Finding | What changed | Where |
|---|---|---|---|
| 1 | M42-8 said the value, cell and marker columns "keep their widths", but `VALUE_WIDTH` was 120 (M4-1's value column was 130) | `VALUE_WIDTH = 130.0`, M4-1's width. The tests expect `[300, 130, 130, 60]` at 644 points and `[120, 130, 130, 60]` at 294 points. The comment reads 120 + 8 + 130 = 258. M42-8, Task 5's prose and commit message, the README and the tracker all say 130 | Decisions M42-8; Task 5 Steps 1 and 3 (`the_label_column_flexes_down_to_its_minimum`) |
| 2 | Task 0 Step 2 left out the engine, whose numbers the tests pin | The `diff --stat` list adds `magcoupling-rs/src/engine magcoupling-rs/src/lib.rs magcoupling-rs/tests/data`. The Expected names the pinned numbers and no longer says the engine may differ | Task 0 Step 2 |
| 3 | Task 3 Step 4 relied on 0.33.0 being the newest 0.33 release | `cargo update --manifest-path .../magcoupling-rs/Cargo.toml -p egui_plot --precise 0.33.0` runs right after the first `cargo test`, once the package is in the lock. Its Expected comes from the replay, including the `Downgrading` case | Task 3 Step 4 |
| 4 | Task 3's review ran on `sonnet` although the task restates two engine closed forms | The implementation stays on `sonnet`. The per-task review runs on the session model (omit `model`, `// session model: restated engine closed forms`), under section 5's precedence rule | Global Constraints (Model tiers); Task 3 **Model** |
| 5 | A 0 x 0 region gave a negative scale and mirrored transforms | New `geometry_view::side_by_side` returns `None` unless the scale is finite and above 0, and refuses an extent that is not finite. `geometry_ui` then paints no view and keeps `scale` at 0. New test `a_region_with_no_room_draws_no_view`: on 0 x 0 and 1 x 1 screens `scale >= 0`, no transform and no dimension; it also checks `side_by_side`'s refusals and its placement (two 1 mm views, 2 mm apart in 100 points: scale 25, origins at x 0 and 75) | Task 2 Steps 1 and 3 |
| 6 | A finite but huge claim (1e300, 1e39) shrank the drawing to nothing, unannounced | Each view's extent comes from its pieces. `drawn_claim` leaves off a claim line farther than `CLAIM_REACH` (10) times the pieces' extent along its axis (the body's largest OD, `body_od`; the axial stack) and pushes `Not drawn: the {axis} claim {mm} is off the drawing` in amber. The diameter is decided once for both views, so its note appears once. New tests: `a_claim_far_past_the_pieces_is_left_off_the_drawing_and_says_so` (Task 1: 1e300, 1e39 and -1e39; extents kept; one note per axis; drawn at the reach, not at the next number past it) and `a_claim_far_past_the_pieces_keeps_the_drawing_to_scale` (Task 2: scale above 0, the end view filling over half the drawing's height, the note drawn in amber) | Task 1 Steps 1 and 3; Task 2 Step 1 |
| 7 | `retain_finite` dropped pieces silently | `retain_finite` returns how many pieces and dashed lines it dropped. `geometry` then notes `Not drawn: {n} parts of the {view} view hold a number that is not a number` (singular for one) while the view is still drawable. `a_number_that_is_not_finite_draws_nothing_wrong_and_says_so` asserts 2 end-view parts and 3 side-view parts for a NaN bore, 1 and 1 for a NaN claim, and none at the defaults | Task 1 Steps 1 and 3 |
| 8 | Two cases were untested: every axis exceeded at once, and equality at the claim | `the_space_claim_callouts_name_each_overshoot_in_red` adds the 50.8 mm magnets with a 40 mm diameter claim: callouts 4, 5 and 6 red, `red == 3` side lines, the end circle red. It also adds `max_diameter_mm = rotating_od_mm` (callout 6 not coloured, circle plain) against `rotating_od_mm.next_down()` (red). The engine's `overshoot` and `rotating_od_mm = max(cup OD, cap OD)` were read first, to confirm the claim does not move the OD | Task 1 Step 1 |
| 9 | The corner-gap line stopped 0.41 mm short of the outer flat | The line runs along face 1's normal from `inner_corner_radius_mm` to `outer_face_apothem_mm`. With E8 on, `model.rs` has `corner_gap = A_o - r_corner_i`, so the line is still the corner gap long. A dashed arc at the inner corner radius spans face 1's inner corners. The test asserts `to` on the outer flat (its normal component the apothem, its tangential component within half the block width), `from` on that circle, and the arc drawn. M42-3, the module docs, the README and the tracker describe it | Decisions M42-3; Task 1 Steps 1 and 3 (`the_end_view_callouts_measure_the_gaps_they_name`) |
| 10 | The pole sweep's "This design" marker sat off its own curve | The pole sweep's line is named `PULL_OUT_SMALLEST_APOTHEM` ("Pull-out torque (rows at their smallest fitting apothem)"). The plots test and the panel test look for that legend | Task 3 Steps 1 and 3 |
| 11 | `torque_temperature_ui`'s reference lines had no finite guards | One helper pair, `reference_hline` and `reference_vline`, adds a dashed line only when its value is finite. All five reference lines go through it: the required minimum, the operating temperature, the governing limit (twice) and the floor | Task 3 Step 3 |
| 12 | `finite(Mm)` was written three times | `geometry::finite` is `pub(crate)`; `geometry_view` and `clamp_drawing` import it | Tasks 1, 2 and 4, Step 3 |
| 13 | The "views side by side at one scale" logic was written twice | `side_by_side` (finding 5) serves `geometry_ui` and the clamp's `drawing_ui`. `drawing_ui` now returns 0 without room, asserted in `the_clamp_tab_paints_both_views_to_scale_and_the_table` | Task 2 Step 3; Task 4 Steps 1 and 3 |
| 14 | The arrowhead code was duplicated | `geometry_view::arrowhead(painter, tip, dir, stroke)`, called by `paint_dimension` and by the clamp's `arrow` | Task 2 Step 3; Task 4 Step 3 |
| 15 | Seven recursive shape walkers, and a clamp loop that walked only top-level shapes | `test_support::flat_shapes` flattens `Shape::Vec`. `drawn_texts`, `text_rects`, `text_color`, `circle_radii`, `segment_colors`, `rect_widths`, `paths_and_circles` and the clamp test's circle list are filters over it, and `text_rect` is `text_rects(..).into_iter().next()`. `text_rects` moved from Task 5 to Task 2 with `flat_shapes`, so `text_rect` is written once and Task 5 no longer touches `test_support.rs` | Task 2 Step 1; Task 3 Step 1; Task 4 Step 1 |
| 16 | The plot-follows-the-edit assertion was true by construction | Each plot has a constant id (`PlotKind::id`). After the edit frame, the panel test loads the torque-temperature plot's `egui_plot::PlotMemory`, maps `[op, new pull-out]` through that frame's transform, and finds the painted marker (radius `POINT_RADIUS + 1`) there. The marker must also be at least 0.5 points from where the old pull-out maps. Of the critic's two options this is the painted-marker one: the bounds option would hold for the old pull-out too | Task 3 Steps 1 and 3 (`each_plot_tab_draws_its_plot_from_this_frame_s_results`) |
| 17 | The side-view test did not pin `retainer_span_mm` | Both `Part::Retainer` rects are `housing.retainer_span_mm` wide. The test also asserts that this span differs from the input's (52.6 mm against 14.5 mm) | Task 1 Step 1 (`the_side_view_uses_the_housing_in_effect`) |
| 18 | The side-view cap check only exercised `Transform` | The test collects the painted `Shape::Rect`s and finds the cup depth in effect and the 0.8 mm cap at the scale, to 1e-2 | Task 2 Step 1 (`known_dimensions_map_to_their_pixel_distances`) |
| 19 | The narrow-window test checked only the value | It also finds the Calculator!C93 row's label (looked up in `table_entries()` by cell) exactly once inside x 0 to 630 points, filling its 120-point column and ending left of the value | Task 5 Step 1 |
| 20 | No painted test of a sweep with an empty line | `each_plot_draws_its_series_with_their_point_counts` draws the gap sweep of `short_magnets()`: 13 markers of `POINT_RADIUS` and no 13-point path | Task 3 Step 1 |
| 21 | The screw count departed from drawing.py through `.min(screws_fit)` | `row.screws_needed.clamp(0, MAX_DRAWN_SCREWS)`, matching drawing.py's `range(int(row.screws_needed))`. The test adds 3 needed with 1 fitting (3 axes drawn) and -2 needed (none) | Task 4 Steps 1 and 3 |
| 22 | "plot ids ... are constants the tests read" was not true | The ids are `pub const` names behind `PlotKind::id`, and the panel test reads one (finding 16) | Decisions (the settled list); Task 3 |
| 23 | Tasks 1 to 5 commit code before Task 6's docs | Global Constraints now state that Tasks 1 to 5 are intermediate commits and that the branch is reviewed and merged only as a whole, after Task 6, so no code change reaches `main` without its docs | Global Constraints (Docs change with code) |

**Found while revising** (beyond the critic's list):

- `side_by_side`'s first test run failed on an extent with a NaN width. `f32::min` passes over NaN, so the scale came out finite. `side_by_side` now refuses extents that are not finite before computing the scale, and `02-system.yaml` records the lesson (Task 6).
- Galley text does not prove a label shows. `Galley::text()` returns the job's full text even when `Label::truncate` elides it, so finding 19's label check alone would also pass for a zero-width label. The test therefore also asserts the label's width (at least the 120-point minimum) and its order before the value.
- Task 2's 1e300 test cannot compare against the default design's scale. The extra note shortens the drawing, so the scale can be slightly smaller. The test instead asserts that the end view fills over half the drawing's height (before the fix, the scale was about 1e-298 points per mm).
- Five mutations were run in the scratch tree, each tripping exactly one test: the old corner line (`the_end_view_callouts_measure_the_gaps_they_name`), a dropped-part count forced to 0 (`a_number_that_is_not_finite_draws_nothing_wrong_and_says_so`), no reach check (`a_claim_far_past_the_pieces_is_left_off_the_drawing_and_says_so`), a diameter line never red (`the_space_claim_callouts_name_each_overshoot_in_red`) and a guard without `scale > 0` (`a_region_with_no_room_draws_no_view`). The other new assertions (findings 16 and 18 to 21) were not mutation-checked; each was seen to pass on the fixed code and reads the painted output or the marks directly.
- Rebasing the old Task 5 onto the new Task 2 re-applied `text_rects` a second time. The rebuilt Task 5 commit drops that hunk, and the replay confirms each task's tree equals the scratch tree.
- `body_od` holds the body's largest OD once, for the diameter decision, the side view's top and the test's reach.
- Task 6's docs cover the changes: `03-structure.yaml` (`CLAIM_REACH`, `NOT_DRAWN`, `finite`, `side_by_side`, `arrowhead`, the plot ids, `flat_shapes`), the README (the corner gap, the claim reach and the parts note, the pole sweep's legend, finite reference lines, the screws needed, the new tests), the tracker, and one more lesson in `02-system.yaml` (the four lessons in Task 6's prose).

**Re-verification** (all of it on the code in this plan's blocks):

- The scratch tree after each rebuilt task: `cargo test --features app --lib` (320, 330, 339, 347 and 349 passed), `cargo fmt --check`, and clippy `-D warnings` with no feature and with `app`. For Tasks 3 to 5 it also ran the wasm32 clippy of the `gui` library and of `magcoupling-web`, and for Task 3 the wasm32 engine check. The replay below is the full evidence: all six checks after every task.
- The replay parsed 73 blocks (69 replaces, 4 creates) from this file. It applied them task by task to a fresh LF export of `93120f5`, using a cargo target directory that started empty. Every anchor matched exactly once.
  - Each red step failed with the errors stated (90, 47, 83, 51 and 7).
  - Each green step passed with the counts above.
  - The six "Format and lint" checks passed after every task, the wasm32 clippy included, in a target directory that started empty before Task 1 (not one shared with the first version's builds).
  - Each task's tree equals the scratch tree file for file.
- Gates 4 to 12 passed on the replayed tree with `GATE PASS` and no `SKIP gate`. Task 3 Step 4's `cargo update --precise` printed the stated lines and left the lock unchanged.
- The web bundle was rebuilt and checked in the browser, as the Verification record says.
- The expected outputs, the Verification record, this record and a few prose touches (Task 3 Step 4's `tail -2`, the model-tier sentence, the settled list's wording) were regenerated into the plan after the replay. The final file's 73 blocks are identical, one by one, to those of the replayed file, so none of that changed code.
