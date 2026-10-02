# Magcoupling M4-3: Equation Explorer, Assumptions, Teaching Notes, Material and Grade Pickers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the calculator's panel the GUI side of spec Addendum A2 to A6: an equation explorer (every displayed value hovers its equation with coloured terms, which are then marked wherever their values appear, and a docked Equation panel that drills from a result down to its inputs), the assumptions view with its "assumptions modified" banner, the reviewed teaching notes with their diagrams and the "start here" order, and the material, magnet-part and grade pickers with the six material warnings linked to their notes, in the native and the web app.

**Architecture:** The engine side exists (plan A-3: `engine::explain`'s evaluable markup, `Registry`, `notes`; `engine::assumptions`; `engine::warnings`; the material, grade and part tables); this plan only draws it, and no engine file changes. `gui/typeset.rs` is a pure box-model typesetter (width, ascent, descent; text runs and strokes) for the markup's tree, with `TermColors` assigning each term a palette colour. `gui/readouts.rs` is the one readout hook: every view hands each value it displays to `Readouts::show`, which frames it in its term's colour when the equation in view reads it, shows its hover text and typeset equation only while hovered (`on_hover_ui`), and records a hover or a click; the panel turns the last frame's hover (or the open equation) into this frame's marks. `gui/explorer.rs` holds the Equation panel's state (trail, focus, Explain, start-here, a note opened alone) and draws it at the bottom of the centre region. The assumptions view is a tab of the inputs side drawing the inputs' own rows; `gui/diagrams.rs` paints the notes' six diagrams; `gui/pickers.rs` adds the library's properties to the material selectors and part and grade drop-downs beside the text fields; the dashboard lists the warnings that fire. The equation registry is built once per process.

**Tech Stack:** Rust 2024 (rustc 1.89); egui and eframe 0.32.3, egui_plot 0.33.0 (locked to linkage-sim-rs's versions, unchanged); headless egui tests (`egui::Context::run` with injected input, shapes inspected); Playwright MCP for the web check; Git Bash for every command.

**Spec:** `C:/Users/Cole/source/repos/lsim-mag-m43/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, Addendum A2 ("Hover. Any displayed value (dashboard, results table, geometry callouts) shows a tooltip with its equation. Each term is coloured, and the same colour marks that term wherever its value appears on screen"; "Equation panel (docked, toggleable). It shows the open equation large, its terms with values and units, a breadcrumb trail, and a 'used by' list. Clicking a term drills into that term's own equation. Leaf terms (inputs) highlight their slider. One equation at a time: never a page of every formula"; "Rendering. A small egui typesetter for the markup: inline fractions, sub/superscripts, Σ and √. No LaTeX dependency"), A3 ("Toggle panel that separates model assumptions from design inputs. Each assumption shows its value, unit, rationale and source"; "In the equation panel the term is styled as an assumption, with a changed-from-default dot. An 'assumptions modified' banner shows whenever any assumption differs from its workbook default, beside a 'reset to workbook defaults' button"), A4 ("An optional Explain section per equation, hidden by default and toggled in the equation panel. It always shows the note for the equation currently open"; "an optional small diagram"; "Start here: a short suggested order ... opens the matching equations in turn"; "Accuracy gate"), A5 ("Per-part material pickers backed by a small library"; the warnings "plain language, colour-coded, linked to their teaching note"), A6 ("Custom dimensions stay available: pick any grade with manual dimensions") and "Addendum testing / GUI (headless egui)". Engine side: plan A-3, `C:/Users/Cole/source/repos/lsim-mag-m43/docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`, and `magcoupling-rs/README.md` "Equation explorer". Conventions: plans M4-1 and M4-2 (`docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md`, `...-m4-2-geometry-plots-clamp.md`). Scope: the third and last of the M4 plans; next is M3 (live 3D fields).

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-m43`, branch `magcoupling/m4-3`, created by Task 0 from `main` at `513b169` (the merge of M4-2). Every command uses absolute paths into it. Nothing is pushed.

**Process (the user's lean process, 2026-10-01):** seven tasks of exact code; every task's implementation and per-task review run on `sonnet`, except the reviews of Task 1 (session model: its typesetter restates every equation) and Task 5 (session model: its diagrams restate physics); no pre-flight scan, so every block below quotes `main` at `513b169` exactly (Task 0 Step 2 checks the files); the whole-branch review after Task 7 runs on the session model.

## Decisions to confirm

**Confirmed (user, 2026-10-02): the recommended option on all of M43-1 to M43-15.** No task stops on a decision.

These choices arose while turning the spec into code; no approved decision settles them. The plan implements the recommended option of each (the code comments and the README cite the ids). Task 0 records the user's answers; if the user picks another option, the task that implements it stops and escalates instead of improvising.

| Id | Question | Recommended (implemented here) | Alternatives | Tasks |
|---|---|---|---|---|
| M43-1 | Where does the Equation panel dock? | **At the bottom of the centre region**, resizable, 320 points to start, over the full width between the inputs and the dashboard: long formulas fit on one line and the geometry view keeps its width (it gets shorter) | (b) a right side panel between the centre and the dashboard (a 1280 px window leaves the centre ~300 points); (c) a floating `egui::Window` (covers the views, M5 nests windows) | 3 |
| M43-2 | What is open when the page starts? | **The Equation panel closed** until a value is clicked or its header button "Equation panel" is pressed; Explain off (the spec: hidden by default); the inputs side on "Design inputs"; no start-here walk. The gui-smoke screenshot keeps M4-2's layout | (b) the panel open on the pull-out at start | 3, 4, 5 |
| M43-3 | Term colours and how a term is marked on screen | **Eight colours** (orange, sky blue, green, yellow, violet, vermillion, blue, pink: `typeset::TERM_PALETTE`), one per term in the formula tree's order (`Formula::visit`: a `cases` arm's condition before its value, though the row draws the value first), a ninth term taking the first again (13 records read more than eight terms; `temperature.thermal.heat_capacity_J_K` reads 14, so two of its values can share a frame colour); a Σ's term by its template, its members by the harmonics summed. A value or input row whose path is a term of the equation in view (the one hovered in the last frame, else the one open in the shown panel) gets a **2-point frame** in the term's colour; the target symbol is in the text colour. While a value of another equation is hovered, the marks are that equation's, so the Equation panel's term swatches fade (`explorer::SWATCH_DIM`) and a colour on screen is not read against the open equation's key | (b) tint the value's text (collides with the badges, the greying and the red callouts); (c) a coloured underline (thin at 13 points) | 1, 2 |
| M43-4 | How the pickers show properties | **Materials:** the selector stays the row's drop-down; each choice shows every library property on hover (sourced values, "not sourced" where the data file has none), and a small line under the row sums up the material picked. **Parts and grades:** the text fields stay (typed names and design files keep working); under each a "Pick a part" / "Pick a grade" drop-down lists "Custom dimensions (manual)" or "Blank" and the 15 library parts or 17 grades, each row's properties on hover, and a line under it sums up the part or grade in use | (b) a grid of every material and its properties in a collapsible table (wide for the 320-point side); (c) replace the part and grade text fields by drop-downs only (a design file's unknown name could not be shown) | 6 |
| M43-5 | Who owns the equation registry? | **A process-wide `OnceLock`** (`readouts::registry()`, as `InputCatalogue::get` and `CorrectionIndex::get`), built in `MagcouplingPanel::new` (about 3 ms in release, never per frame) and logged once ("magcoupling explorer: N equations") | (b) a panel field passed to every view (plan A-3's "own it in the app state"; the views run inside `&mut self` methods, so it needs an `Arc`, and every view's tests need a registry argument) | 2 |
| M43-6 | Characters egui's default fonts lack | **Display substitutions** (`typeset::glyph_safe`): ϑ (U+03D1, every temperature symbol) drawn as θ, the same letter; the superscript minus (U+207B, two notes) as ¯; ∝ (one note title) as ~; ∈ and ⌈⌉⌊⌋ drawn with strokes; `concat` as texts side by side. No record or note text changes (a changed note would return to Draft) | (b) bundle a font that has them (about 300 kB more wasm, a licence to check); (c) paint every missing glyph | 1, 5 |
| M43-7 | Where do the assumptions go? | **A tab row on the inputs side: "Design inputs" / "Assumptions"**; the Assumptions view draws each assumption's inputs with the inputs' own rows (so idle frames rewrite nothing and an edit is an edit), its rationale and source; the inputs also stay in their groups (as the Key design inputs do, decision M41-13); the banner sits in the header, visible from every view | (b) remove the assumption inputs from their groups; (c) a floating window | 4 |
| M43-8 | Where do the warnings show? | **At the top of the dashboard, under "Material warnings", only while one fires**: a badge and the text in the severity's colour (a warning red, a caution amber, the badge colours), then "Why: <note title>", a link that opens the reviewed note on its own in the Equation panel, with links to the equations it explains (five of the six warning notes explain no single equation) | (b) at the bottom of the dashboard (below the fold at 800 px); (c) a tab of the centre region | 6 |
| M43-9 | What do the clicks do? | **A readout clicked opens a new trail; a term or a "used by" link is one more step; a crumb goes back to its step**; the trail keeps its last 32 steps and the header shows the last 8 after "…" (32 crumbs would wrap over most of a short panel); a result without an equation record opens a view of its label, value and cell; a Σ's term opens its first harmonic summed | (b) a readout clicked appends to the trail | 3 |
| M43-10 | Where does Explain put the note? | **Under the open equation's value line, before the term list**, so the note is on screen when toggled (the term list scrolls under it) | (b) after the term list | 5 |
| M43-11 | "Plot readouts where a path exists" | **A line over each plot reading out the design's values it draws** (`PlotKind::readouts`: the pull-out, the hot-low and cold-high torques and the governing limit; the corner gap, the pull-out and the required floor; the time constant, the time to the limit and the limit; the pull-out angle), each a readout. The curves' points have no path: the plot keeps egui_plot's coordinate readout | (b) nothing over the plots (the plots show no value with a path) | 2 |
| M43-12 | How does a leaf term highlight its input? | **It shows the view holding the row** (the Assumptions view for an assumption, else the design inputs), **opens its group, scrolls the row into view once and frames it** (3 points, the selection colour) until another equation or term is opened | (b) a timed flash; (c) highlight without opening the group | 3, 4 |
| M43-13 | M4's typesetting pass: rename symbols? | **No renames in M4-3**; the notes are shown as reviewed, unchanged (the thermal time constant note writes tau and theta_start where the records show tau_th and theta_hot; recorded in 04-memory) | (b) rename now and send the affected notes back to Draft for a physics review | 7 |
| M43-14 | The corrected-vs-workbook markers | **Keep M4-1's markers** (`CorrectionIndex`, the "E3 E7 E8" pins); the Equation panel lists the open equation's `Registry::corrections_upstream` ("Embodies corrections: E7, E8, E3") | (b) re-source every marker from `corrections_upstream` (moves the dashboard and table markers, changes M4-1's tests) | 3 |
| M43-15 | Per-row end-effect greying of the results table (M4-1's carried item) | **Deferred: the banner stays.** The registry's graph covers only the explained results (about 400 of the table's rows), so greying by it would grey some rows computed from the pull-out and miss the others | (b) grey the rows downstream of `model.f_end` in the graph | 7 |

Settled by this plan's design without a new decision (each is stated where it is implemented): hover texts and equations are built only while hovered, the dashboard's too (`DashboardLine::tooltip`, the M4-1 review's item); a label stays a plain label and a click sensor goes over its rect (`Readouts::show_over`: a clickable label would change colour); a Σ's term list leaves out the harmonics past the set; a selector compared for equality (= or ≠) with a code shows its choice's label in conditions, while an ordering keeps its numbers (σ_n's `5 ≤ N_h` compares the harmonic index with the highest harmonic), and a selector's value shows its label in the term list; a screw-size row index shows the size's name; factors sit side by side as `render::plain` writes them (`n_slip 2 π/60`, × only between two numbers), so the panel and the plain text read alike; every note shown by id (a warning's, a start-here step's, one opened alone) goes through `explorer::reviewed_note`; the assumptions banner asks for one more frame when it changes (the header sizes from the last frame); rows of links set `TextWrapMode::Extend` so each moves whole.

## Global Constraints

Every task's requirements implicitly include this section.

- The engine does not change: no file under `magcoupling-rs/src/engine/` or `magcoupling-rs/tests/` is edited. Workbook parity (1,149 checks), the differential data (`gen_differential.py --check`), every registry test and `all_corrections_together_give_the_reviewed_headline` pass unchanged. `reference/magcoupling-py/` is not edited. No teaching note's text changes (decision M43-13: a changed note returns to Draft).
- Feature boundaries: the engine stays pure std (gate 6). Feature `gui` keeps its dependencies (egui, egui_plot, serde_json, base64, flate2, log); no new crate, so `Cargo.toml` and `Cargo.lock` do not change. `workbook-parity` never reaches a shipped build (gate 10).
- Per frame: the panel still calls `compute_all` once per frame; nothing builds the registry per frame (`readouts::registry()`, built once); a hover text or equation is built only inside `on_hover_ui`; the Equation panel typesets one equation a frame (one at a time: the spec).
- The panel's host API does not break: `MagcouplingPanel::new`, `ui`, `inputs`, `results`, `reset`, `open_share_payload`, `set_keyboard_shortcuts` keep their signatures. The views' public functions gain a `&mut Readouts` parameter (`dashboard_ui`, `ResultsTable::ui`, `geometry_ui`, `clamp_ui`, `plot_ui`); no code outside this crate calls them (M5 has not started).
- UI text: every text drawn has a glyph in egui's default fonts (`has_glyph` at proportional 14, the criterion of `every_text_the_panel_shows_has_glyphs_in_the_default_fonts`); the typesetter and the notes go through `typeset::glyph_safe` (decision M43-6). U+2192 stays `->`.
- Gate runs (Task 0 and Task 7; Tasks 1 to 6 run the magcoupling checks of their Step "Format and lint", which are gates 4 to 9's commands): `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m43/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate` line; then `git -C C:/Users/Cole/source/repos/lsim-mag-m43 checkout -- docs/chebyshev_lambda` (the linkage tests rewrite those PNGs; never commit them).
- `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml` before every commit, and `cargo fmt --check` clean. Every block below is already in rustfmt's layout.
- The blocks quote the files with LF line endings, as git stores them (Task 0 creates the worktree with a worktree-scoped `core.autocrlf=false`: this machine's system gitconfig sets it to true). "Create" writes a new file; "In ..., replace: ... with: ..." is one exact replacement whose old text occurs exactly once in the file at that point (each block is a whole-line hunk with its context). Apply a step's blocks in the order given. If a block's old text is not found, stop and escalate; do not improvise a match (Task 0 checks that the files the blocks touch are still those of `513b169`).
- Windows paths: the worktree path is short on purpose. A release build under a deep directory fails with `LNK1104` (MAX_PATH); `build_magcoupling_web.sh` writes to `magcoupling-rs/target/` inside the worktree.
- Tests that read source files must normalize CRLF (the main checkout is CRLF). This plan adds none; the existing `the_smoke_test_share_link_opens_its_design` reads `gui-smoke.js` and gains one `contains` check of a one-line constant (Task 7), unaffected by line endings.
- Every commit message ends with exactly two lines: a `Co-Authored-By:` line naming the model that wrote the commit, then `Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9`. The blocks write `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; a `sonnet` executor writes its own model's name. Subjects: `feat(magcoupling-rs): ...` (Tasks 1 to 6), `docs(magcoupling): ...` (Task 7). Nothing is pushed.
- Docs change with code (repo rule, `docs/ai/01-meta.yaml`; the user's rule that every code change ships with its docs): Tasks 1 to 6 are intermediate commits on `magcoupling/m4-3`, and the branch is reviewed and merged only as a whole, after Task 7, so no code change reaches `main` without its docs. Task 7 updates `magcoupling-rs/README.md` and `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md` (new YAML scalars holding `: ` are quoted), the `gui-smoke` workflow, and commits this plan. Read `docs/ai/*.yaml` before starting (Task 0).
- Deploy: `deploy-web.yml` on `main` builds and ships `/magcoupling/` on the next push of `main`; this plan pushes nothing.
- Model tiers (CLAUDE.md section 5; the user's lean process of 2026-10-01): every task gives the exact code, so every task's implementation runs on `sonnet` (`model: 'sonnet'`), Task 0 included, and so does each per-task review but Task 1's and Task 5's. Task 1's typesetter draws every physics equation of the registry, and a mis-render misstates one (as the swapped label of σ_n's `n ≤ N_h` did before this revision), so **Task 1's per-task review runs on the session model** (omit `model`, comment `// session model: typesetter restates every equation`). Task 5 paints diagrams that restate physics (the square wave's harmonic amplitudes 4/(nπ) sin(nπλ/2), first-order heating, the demagnetization knee and load lines, fringing, torque against angle); it is not a `risk: physics` item (no engine change), but it matches both lists of section 5, where the session model wins, so **Task 5's per-task review runs on the session model** (omit `model`, comment `// session model: diagrams restate physics`). The whole-branch review after Task 7 runs on the session model (omit `model`, comment `// session model: final whole-branch review`). A `sonnet` attempt that ends `blocked`, leaves a check red or is rejected in review escalates every retry to the session model.

## Review Focus

Eight input classes the spec implies, each pinned by a test in the task that owns the code:

1. **Characters egui's default fonts lack** (ϑ in every temperature symbol and two table fields, ∈ under every Σ, ⌈⌉⌊⌋ in the rounding records, the superscript minus and ∝ in three notes). Expected: never an empty box; the substitutions of decision M43-6, nothing else changed. Tests: `every_equation_typesets_with_glyphs_in_the_default_fonts` (every record at two sizes, Task 1), `every_teaching_note_shows_with_glyphs_in_the_default_fonts` (Task 5), `every_picker_text_has_glyphs_in_the_default_fonts` (Task 6), and `every_text_the_panel_shows_has_glyphs_in_the_default_fonts` extended with the Equation panel (Task 3), the Assumptions view and banner (Task 4) and the warnings (Task 6).
2. **A harmonic set other than the workbook's** (all six odd harmonics, or a code outside the choices from a design file). Expected: a Σ's term is marked on the harmonics summed and no other, the term list leaves the others out, nothing panics for a set that sums nothing. Tests: `a_family_term_is_coloured_by_template_and_marks_only_the_harmonics_summed` (Task 1), `the_term_list_leaves_out_the_harmonics_past_the_set` and `a_harmonic_set_outside_the_choices_lists_only_its_selector` (Task 3).
3. **A small or short window with the Equation panel open** (a 1 x 1, a 200 x 150 and a 640 x 240 screen, smaller than the panel's 320-point start). Expected: every frame draws without a panic and edits nothing. Tests: `the_equation_panel_draws_in_a_tiny_window` (Tasks 3 and 5), `a_long_trail_shows_its_last_crumbs_and_draws_in_a_short_window` (Task 3: a 32-step trail at 640 x 240, and its last crumbs drawn at 1280 x 240).
4. **Paths the explorer does not expect** (a result with no record, a Σ's family template clicked in the equation, a walk longer than the trail, a label shared by an input and an assumption heading). Expected: the result's label, value and cell; the first harmonic summed; the last 32 steps, the header showing the last 8 after "…"; the row frame on the input row. Tests: `a_result_without_an_equation_shows_its_value_and_cell`, `a_long_walk_keeps_the_last_steps`, `a_long_trail_shows_its_last_crumbs_and_draws_in_a_short_window`, `following_a_term_drills_into_a_result_and_focuses_an_input` (Task 3), `a_leaf_assumption_term_shows_its_row_in_the_assumptions_view` (Task 4).
5. **Assumption edits from every path** (a slider of the Key design, the Assumptions view's rows, a reset, an undo of the reset, idle frames). Expected: the banner appears and clears in step (one extra frame for the header), the reset leaves every design input as it was and can be undone, idle frames rewrite nothing. Tests: `the_assumptions_banner_appears_on_change_and_clears_on_reset`, `the_assumptions_view_lists_each_assumption_with_its_rationale_and_source` (Task 4).
6. **Material and magnet names from a design file that the tables lack** (a part name not in the library, a blank part, a grade not in the table, a material code outside the choices). Expected: the pickers show them as typed ("(not a library part: manual dimensions)", "Custom dimensions (manual)", "99 (not a choice)" with no material summary), pick from the tables, never panic, and idle frames write nothing back. Tests: `the_part_picker_sets_a_library_part_or_custom_dimensions`, `a_material_code_outside_the_choices_is_shown_and_kept` (Task 6) and M4-1's `a_text_input_edits_the_part_name_and_says_what_it_resolves_to` (unchanged, still green).
7. **A selector in a condition** (compared for equality with a code, or ordered against a harmonic index: σ_n's `n ≤ N_h`, where N_h's code 5 reads "1, 3, 5 (workbook)"). Expected: the choice's label only in an equality, the number in an ordering; every drawn run traceable to the record. Tests: `a_selector_code_shows_its_choice_and_a_screw_row_its_size` and `every_equation_draws_only_what_its_record_says` (Task 1).
8. **Values the engine leaves undefined** (NaN: every harmonic sum of a set outside the choices; the torque at a hot limit that does not exist, a positive Hcj coefficient without a rating). Expected: the panel opens the equation, draws the record's "undefined" arm, lists the selector alone, and edits nothing. Tests: `a_harmonic_set_outside_the_choices_lists_only_its_selector`, `an_undefined_value_shows_its_equation` (Task 3).

## File Structure

Created (under `C:/Users/Cole/source/repos/lsim-mag-m43/`):

| File | Responsibility | Task |
|---|---|---|
| `magcoupling-rs/src/gui/typeset.rs` | The equation typesetter, pure: `layout`, `layout_equation` (Task 3: `layout_symbol`) -> `Laid` (`Ink` text runs and strokes on a baseline); `TermColors`, `TERM_PALETTE`; `glyph_safe`; `equation_ui`, `laid_ui`, `term_at`, `paint` | 1 |
| `magcoupling-rs/src/gui/readouts.rs` | The readout hook: `registry()` (built once), `Readouts` (`show`, `show_over`, `mark`, `open_note`, `finish`), `ReadoutEvents` | 2 |
| `magcoupling-rs/src/gui/explorer.rs` | The Equation panel: `Explorer` (trail, hover, focus; Task 5: Explain, start here, a note alone) and `explorer_ui`; Task 4: `term_tag`; Task 5: `note_ui`, `reviewed_note` | 3 |
| `magcoupling-rs/src/gui/diagrams.rs` | The notes' six diagrams: `diagram_shapes` (pure), `diagram_ui` | 5 |
| `magcoupling-rs/src/gui/pickers.rs` | The material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts | 6 |

Modified: `magcoupling-rs/src/gui/mod.rs` (each new module), `magcoupling-rs/src/gui/test_support.rs` (Task 1: `assert_glyphs`, the glyph check every glyph test calls), `magcoupling-rs/src/gui/panel.rs` (Task 2: the readouts and their marks; Task 3: the explorer, its dock and header button, the leaf row; Task 4: the inputs side's tabs, the Assumptions view, the banner; tests in Tasks 2 to 6), `magcoupling-rs/src/gui/dashboard.rs` (Task 2: readouts, the lazy tooltip; Task 6: the warnings), `magcoupling-rs/src/gui/results_table.rs`, `geometry_view.rs`, `clamp_drawing.rs` (Task 2: readouts), `magcoupling-rs/src/gui/plots.rs` (Task 2: readouts and the readout line), `magcoupling-rs/src/gui/input_ui.rs` (Task 6: the pickers; `current_text` shared with them), `magcoupling-rs/src/app.rs` (Task 7: the smoke test checks the registry log line), `.claude/workflows/gui-smoke.js`, `magcoupling-rs/README.md`, `docs/ai/{02-system,03-structure,04-memory}.yaml`, `docs/ai/05-update-tracker.md` (Task 7). No engine file, test data file, `Cargo.toml`, `Cargo.lock` or Python file changes.

Order: the pure typesetter first; then the hook every view goes through (the hover and the marks), the Equation panel the clicks open, the assumptions (which style the panel's terms), the notes (which the panel shows), the pickers and warnings (whose links open notes), and the docs.

## Verification record

This plan was replayed before it was handed over, and replayed again after the critic revision (the Self-review record at the end lists what changed). The counts and results below come from that second replay: Task 1's typesetter block was amended after the first replay (a selector's choice label now reads only in an equality), so the first replay's record no longer applied. The blocks were developed task by task in a scratch tree (an LF export of `513b169`, `git -c core.autocrlf=false archive 513b169`, one commit per task, the revision folded into each task's commit), generated from those commits by a script (each replace block a whole-line hunk whose old text is unique at the moment it is applied), then parsed out of this file and applied literally, task by task, to a fresh LF export of `513b169`, with a cargo target directory that started empty.

- **Red steps** failed as stated: each Step 2 stops the build with the error count it names (Tasks 1 to 6). **Green steps** passed.
- **After every task** the magcoupling checks of its Step "Format and lint" passed (`cargo fmt --check`; `cargo clippy --all-targets -- -D warnings` with no feature and with `app`; the wasm32 clippy of the `gui` library and of `magcoupling-web`; the wasm32 check of the engine), and **the replayed tree equals, file for file, the tree in which that task was developed**. No `Cargo.toml` or `Cargo.lock` change.
- **After Task 7**, the whole gate passed on the replayed tree with the oracle Python: gates 1 to 12 of `gate.sh`, `GATE PASS` and no `SKIP gate` line (gate 1: the linkage crate's `926 passed` library run and its integration binaries; gate 4: engine unit tests `172 passed`, every integration binary at Task 0's counts; gate 7: `424 passed`; gate 10: the parity guard on the native and wasm32 shipped builds and its negative control; gate 11: `egui 0.32.3`, `egui_plot 0.33.0`, `eframe 0.32.3`, `wasm-bindgen 0.2.114` in both lock files and the CLI pin; gate 12: `1159 passed`, the differential data current).
- **The engine's ignored release gate**: `cargo test --test explain release_notes_are_reviewed -- --ignored` passes on the replayed tree (every start-here and warning note is Reviewed); it stays `#[ignore]` because this plan edits no engine test (Task 7 records it in 04-memory for the engine owner).
- **The web bundle** was built and opened before the critic revision, from the first replay's tree, through the shipped arguments and the parity guard (`magcoupling_shipped.sh`: "magcoupling-web builds without workbook-parity"; `magcoupling-web_bg.wasm` 4.8 MB, 4.4 MB after M4-2) and served with `python -m http.server`; Playwright opened `/magcoupling/` at 1280 x 800: the console showed `magcoupling explorer: 392 equations` with zero errors and zero warnings. Hovering the dashboard's "Pull-out torque at operating temperature" value showed its hover text and `T_pull = T_2D f_end f_cal` typeset with the three terms in orange, sky blue and green; a click opened the Equation panel at the bottom of the centre region (the crumb `T_pull`, the equation at 22 points, "T_pull = 2.688 N·m", "Embodies corrections: E7, E8, E3", the three terms with their values, "Used by T_pull,20 T_op T_svc T_ripple,in"). With "Pull-out torque at 20 °C" open (its stacked fraction B_r,i B_r,o over B_r,i,op B_r,o,op), hovering the cup OD showed `D_cup = 2 (r_pocket + t_wall)` and framed the Key design row "Minimum outer return ring wall" in t_wall's sky blue. "Start here" opened the harmonics step with its note and the square-wave diagram; picking "304 stainless (non-magnetic)" as the back iron showed its properties on hover and its summary, the two warnings at the top of the dashboard (red and amber), and "Why: Non-ferromagnetic back iron" opened the note with the flux-path diagram. The gui-smoke share link (`?m=`, Torque -> Magnets) logged `magcoupling explorer: 392 equations`, `magcoupling: loaded the design from the share link (sizing: Torque -> Magnets)` and `magcoupling sizing: Solved at 14.18 mm (hot-low torque 2.500 N·m)`, no error. The browser run was not repeated after the revision; its visible changes (σ_5's condition reading `5 ≤ N_h`, the number kept, the crumbs cut to the last 8, the swatches fading while another equation's value is hovered, the dashboard's "Why" link through `glyph_safe`, the same text for today's titles) are each pinned by a headless test of the replay above, and Task 7 Step 5 opens the bundle again.

| After task | `cargo test --features app --lib` | Step 2 (red) | Engine unit tests (`cargo test`) |
|---|---|---|---|
| 0 (base) | 367 | — | 172 |
| 1 | 377 | 83 errors | 172 |
| 2 | 384 | 47 errors | 172 |
| 3 | 401 | 73 errors | 172 |
| 4 | 406 | 24 errors | 172 |
| 5 | 415 | 65 errors | 172 |
| 6 | 424 | 41 errors | 172 |
| 7 | 424 | — | 172 |

Every integration test binary keeps Task 0's counts throughout (assumptions 8, deviations 54, differential 19, explain 12 passed and 2 ignored, grades 11, material_library 4, material_links 11, parity 4, python_schema 7, robustness 12, schema 7, sizing 31, static_data 8; doc-tests 1 passed, 1 ignored).

Not exercised by the replay: the `gui-smoke` workflow as a workflow (its console checks were done by hand above; its design-file click unchanged), a native window of `magcoupling-app` (the same panel; the headless tests and the web run cover it), and the user's answers to the Decisions to confirm.

---

## Tasks

### Task 0: Worktree, baseline and context

**Model:** `sonnet` (creating the worktree, running commands and reporting output; CLAUDE.md section 5). Step 8 is the controller's.

Verification only; nothing to write.

**Files:**
- none

**Interfaces:**
- Consumes: `main` at `513b169` or a later commit that leaves the files this plan's blocks touch unchanged (Step 2).
- Produces: the worktree `C:/Users/Cole/source/repos/lsim-mag-m43` on the new branch `magcoupling/m4-3` with LF line endings, a green baseline with its test counts, and the controller's record of the user's answers to the Decisions to confirm.

- [ ] **Step 1: Create the worktree**

This machine's system gitconfig sets `core.autocrlf=true`, which would check the files out with CRLF; every block below quotes them with LF, as git stores them. So the worktree is created with LF and keeps a worktree-scoped `core.autocrlf=false`.

Run:

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation log --oneline -1 main
git -C C:/Users/Cole/source/repos/linkage_simulation config extensions.worktreeConfig true
git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b magcoupling/m4-3 C:/Users/Cole/source/repos/lsim-mag-m43 main
git -C C:/Users/Cole/source/repos/lsim-mag-m43 config --worktree core.autocrlf false
git -C C:/Users/Cole/source/repos/lsim-mag-m43 rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-m43 status --short
```

Expected: the `log` line is `513b169 merge: magcoupling M4-2 GUI (to-scale geometry view with callouts, space claim and design checks; five plot tabs; clamp table and drawing; results table narrow-width fix)` or a later commit; `worktree add` prints `Preparing worktree (new branch 'magcoupling/m4-3')`; then `magcoupling/m4-3`; no status lines (the main checkout's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md` is not in the worktree).

- [ ] **Step 2: Check that the blocks' files are still those of `513b169`**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m43 diff --stat 513b169 HEAD -- magcoupling-rs/Cargo.toml magcoupling-rs/Cargo.lock magcoupling-rs/src magcoupling-rs/README.md docs/ai .claude/workflows/gui-smoke.js magcoupling-rs/tests/data
```

Expected: no output. If any file is listed, stop and escalate: the blocks of the tasks that touch it, or the engine values the tests pin (such as the 2.688 N·m pull-out, the 41.33 mm cup OD, the "E3 E7 E8" markers), must be re-derived first. (Files outside this list, such as the linkage crate's or another plan's, may differ.)

- [ ] **Step 3: Check the line endings**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m43 config --get core.autocrlf; head -c 2000 C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs | tr -cd '\r' | wc -c
```

Expected: `false` and `0` carriage returns. If the count is not 0: run `git -C C:/Users/Cole/source/repos/lsim-mag-m43 rm -rq --cached . && git -C C:/Users/Cole/source/repos/lsim-mag-m43 reset -q --hard` and check again.

- [ ] **Step 4: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/01-meta.yaml`, `02-system.yaml`, `03-structure.yaml`, `04-memory.yaml` (the M4-3 items this plan resolves: "Open (M4-3, carried from M4-1)", "M4 (from plan A-3)", "M4 typesetting pass"); `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md` ("The panel", "Equation explorer"); the spec's Addendum A2 to A6; `magcoupling-rs/src/engine/explain/` (`markup.rs`'s table of what each markup typesets as, `registry.rs`, `notes.rs`), `engine/assumptions.rs`, `engine/warnings.rs`; the panel `magcoupling-rs/src/gui/panel.rs`, `dashboard.rs`, `results_table.rs`, `geometry_view.rs`, `input_ui.rs`; and this plan's Decisions to confirm.

- [ ] **Step 5: Run the crate's tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: the engine run: unit tests `172 passed`; `tests\assumptions.rs` 8; `tests\deviations.rs` 54; `tests\differential.rs` 19; `tests\explain.rs` 12 passed, 2 ignored; `tests\grades.rs` 11; `tests\material_library.rs` 4; `tests\material_links.rs` 11; `tests\parity.rs` 4; `tests\python_schema.rs` 7; `tests\robustness.rs` 12; `tests\schema.rs` 7; `tests\sizing.rs` 31; `tests\static_data.rs` 8; doc-tests `1 passed; 0 failed; 1 ignored`. Then with `app`: `test result: ok. 367 passed`. If a count differs but every binary is `ok`, record the actual counts and read every later task's counts as offsets from them.

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
mkdir -p C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/target
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m43/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/target/gate-task0.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/target/gate-task0.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/target/gate-task0.log
git -C C:/Users/Cole/source/repos/lsim-mag-m43 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0`; `cargo fmt --check` prints nothing.

- [ ] **Step 8: Decisions**

The controller asks the user the plan's **Decisions to confirm** (M43-1 to M43-15) before Task 1 and records the answers in the execution notes. Every task implements the recommended option and names the decision where it implements it; if the user picks another option, the task that implements it stops and escalates instead of improvising.

---

### Task 1: The equation typesetter

**Model:** `sonnet` for the implementation (the plan gives the exact code). **The per-task review runs on the session model** (`// session model: typesetter restates every equation`; Global Constraints). Escalate a retry to the session model.

The pure core of the explorer (spec A2 "Rendering"). `layout` turns a parsed formula (`engine::explain::markup::Formula`, the tree the drift guard evaluates) into a `Laid` box of text runs and strokes around a baseline: `symbol = formula`, then one line per `where` binding. Each markup node is drawn as `engine::explain::markup`'s table says: a stacked fraction for `frac`, inline `a/b`, scripts (a power of a symbol stacks the exponent over its subscript), a radical of strokes, a large Σ with `n ∈ H` beneath (∈ in strokes), `arg max` for `peak`, delimiters scaled to their contents, brackets of strokes for `ceil` and `floor`, a brace with one row per `cases` arm, only the parentheses the markup writes, factors side by side as `render::plain` writes them (× only between two numbers). A selector compared for equality (= or ≠) with a code shows its choice's label; an ordering keeps its numbers (σ_n's `5 ≤ N_h` compares the harmonic index with the highest harmonic, whose code 5 reads "1, 3, 5 (workbook)"); a screw-size row index shows its size's name. Terms carry their path (for clicks) and their colour from `TermColors` (decision M43-3: in the formula tree's order, a ninth term taking the first again, a Σ's term by its template, the summed harmonics added by `with_members`). `glyph_safe` draws the characters the default fonts lack (decision M43-6). The tests pin a fraction, a subscript, a Σ and a root to their reading order and geometry, the term colours and the palette's wrap on the 14-term heat capacity, the choice label, σ_5's ordering left numeric and the screw size, a click on a drawn term, lay out every equation of the registry at two sizes checking each character against the fonts (Review Focus 1), trace every drawn text run to the record's plain rendering, a choice label of a selector it compares for equality, a screw size or the three words only the typesetter writes (Review Focus 7: a stray substitution is a run with no source), and the harmonic sets (Review Focus 2). `test_support::assert_glyphs` is the glyph check this and every later glyph test calls.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/typeset.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/mod.rs` (`pub mod typeset;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/test_support.rs` (`assert_glyphs`)

**Interfaces:**
- Consumes: `engine::explain::{Equation, Registry, TermSource, tables}` (`Registry::{symbol, family_symbol, choices, family_members}`, `tables::{field, lookup}`), `engine::explain::markup::{BinOp, Cond, Expr, Formula, Func, RelOp, Symbol}`, `engine::meta::Value`; in the tests `engine::explain::{Design, render}` (`render::plain`), `markup::parse`, `test_support::primary_button`.
- Produces (`magcoupling::gui::typeset`): `pub const TERM_PALETTE: [Color32; 8]`; `#[derive(Clone, Debug, Default, PartialEq)] pub struct TermColors` with `fn none() -> Self`, `fn of(&Equation) -> Self`, `fn with_members(self, &Registry, &dyn TermSource) -> Self`, `fn get(&self, &str) -> Option<Color32>`, `fn is_empty(&self) -> bool`; `pub enum Ink { Text { at: Vec2, galley: Arc<Galley>, term: Option<String> }, Line { from: Vec2, to: Vec2, width: f32 }, Path { points: Vec<Vec2>, width: f32 } }`; `#[derive(Clone, Debug, Default)] pub struct Laid { pub width: f32, pub ascent: f32, pub descent: f32, pub inks: Vec<Ink> }` with `fn size(&self) -> Vec2`, `fn texts(&self) -> Vec<(String, Rect)>`, `fn term_rects(&self) -> Vec<(String, Rect)>`; `pub fn glyph_safe(&str) -> String`; `pub fn layout(&Fonts, &Registry, symbol: &str, &Formula, &TermColors, size: f32, ink: Color32) -> Laid`; `pub fn layout_equation(&Fonts, &Registry, &Equation, &TermColors, size: f32, ink: Color32) -> Laid`; `pub fn paint(&egui::Painter, Pos2, &Laid, Color32)`; `pub fn equation_ui(&mut egui::Ui, &Registry, &Equation, &TermColors, size: f32) -> egui::Response` (takes no clicks: a tooltip's); `pub fn laid_ui(&mut egui::Ui, &Laid, Color32, egui::Sense) -> egui::Response`; `pub fn term_at(&Laid, origin: Pos2, pos: Pos2) -> Option<String>`. (`gui::test_support`, `cfg(test)`): `pub(crate) fn assert_glyphs(&egui::Context, text: &str, what: &str)`.

- [ ] **Step 1: Write the failing tests**

The module starts as its docs and its tests; Step 3 adds the code between them. The module is registered now, so Step 2 compiles the tests; `test_support` gains `assert_glyphs`, the glyph check the tests call.

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub mod sizing;
#[cfg(test)]
pub(crate) mod test_support;

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
````

with:

````rust
pub mod sizing;
#[cfg(test)]
pub(crate) mod test_support;
pub mod typeset;

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/test_support.rs`, replace:

````rust
    text_rects(output, needle).into_iter().next()
}

/// The colour of the first drawn text equal to `needle`: its override, else its first section's
/// colour (a `RichText` colour or a painter text's), else the fallback colour.
pub(crate) fn text_color(output: &egui::FullOutput, needle: &str) -> Option<egui::Color32> {
````

with:

````rust
    text_rects(output, needle).into_iter().next()
}

/// Asserts that every character of `text` but whitespace has a glyph in egui's default fonts
/// at proportional 14 (the UI's criterion): egui draws an empty box for one it lacks. `what`
/// names the text's source in the failure. The fonts load on `ctx`'s first frame.
pub(crate) fn assert_glyphs(ctx: &egui::Context, text: &str, what: &str) {
    let font = egui::FontId::proportional(14.0);
    for c in text.chars().filter(|c| !c.is_whitespace()) {
        assert!(
            ctx.fonts(|f| f.has_glyph(&font, c)),
            "{what}: no glyph for {c:?} (U+{:04X}) in {text:?}",
            c as u32
        );
    }
}

/// The colour of the first drawn text equal to `needle`: its override, else its first section's
/// colour (a `RichText` colour or a painter text's), else the fallback colour.
pub(crate) fn text_color(output: &egui::FullOutput, needle: &str) -> Option<egui::Color32> {
````

Create `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/typeset.rs`:

````rust
//! A small typesetter for the equation markup (spec Addendum A2 "Rendering": "a small egui
//! typesetter for the markup: inline fractions, sub/superscripts, Σ and √. No LaTeX
//! dependency").
//!
//! [`layout`] turns a parsed formula ([`crate::engine::explain::markup::Formula`], the tree the
//! drift guard evaluates) into a [`Laid`] box: text runs and strokes placed around a baseline,
//! the TeX box model in miniature (a width, the ascent above the baseline, the descent below
//! it). [`equation_ui`] paints it and says which term the user clicked. Every term is drawn
//! with its display symbol ([`Registry::symbol`]) in its colour ([`TermColors`]), so the
//! equation panel, the hover tooltips and the term list agree.
//!
//! What it draws, by markup (the table in `engine::explain::markup`): a stacked fraction for
//! `frac`, an inline `a/b` for `/`, scripts for symbols and powers (a power of a symbol
//! stacks the exponent over its subscript), a radical drawn with strokes for `sqrt`, a large Σ
//! with `n ∈ H` beneath for `sum`, `arg max` for `peak`, delimiters scaled to their contents,
//! a left brace with one row per arm for `cases`, the `where` bindings on lines of their own.
//! A selector compared for equality (= or ≠) with a code shows the choice's label
//! (`backiron = "steel circuit"`, [`Registry::choices`]); an ordering keeps its numbers (σ_n's
//! `5 ≤ N_h` compares the harmonic index with the highest harmonic, not a code). A screw-size
//! row index shows the size's name (`d_h(M4)`). Factors sit side by side as in
//! `render::plain` (`n_slip 2 π/60`), with × only between two numbers.
//!
//! egui's default fonts lack a few characters the markup and the teaching notes use (decision
//! M43-6): ϑ (U+03D1) is drawn as θ, the same letter, the superscript minus as ¯ and ∝ as ~
//! ([`glyph_safe`]); ∈ and the ceiling and floor brackets are drawn with strokes; `concat` sets
//! its texts side by side (no ⧺). A test lays out every equation of the registry and checks
//! every character drawn against the fonts.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DesignInputs;
    use crate::compute_all;
    use crate::engine::explain::markup::parse;
    use crate::engine::explain::{Design, render};
    use crate::gui::test_support::assert_glyphs;
    use std::sync::OnceLock;

    fn registry() -> &'static Registry {
        static REGISTRY: OnceLock<Registry> = OnceLock::new();
        REGISTRY.get_or_init(Registry::build)
    }

    /// Runs `f` with the default fonts loaded (they load on the first frame).
    fn with_fonts<R>(f: impl FnOnce(&Fonts) -> R) -> R {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        ctx.fonts(f)
    }

    fn rect_of<'a>(texts: &'a [(String, Rect)], text: &str) -> &'a Rect {
        &texts
            .iter()
            .find(|(t, _)| t == text)
            .unwrap_or_else(|| panic!("no {text:?} in {texts:?}"))
            .1
    }

    #[test]
    fn a_fraction_a_subscript_a_sum_and_a_root_are_laid_out_in_reading_order() {
        let formula = parse(
            "frac({coupling.c_end}, 2) + sqrt({model.pole_pitch_mm}) + sum(n in H: {model.tau#_Pa})",
            None,
        )
        .unwrap();
        let laid = with_fonts(|f| {
            layout(
                f,
                registry(),
                "x_{a}",
                &formula,
                &TermColors::none(),
                20.0,
                Color32::WHITE,
            )
        });
        let texts = laid.texts();
        let order: Vec<&str> = texts.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(
            order,
            [
                "x", "a", " = ", "c", "end", "2", " + ", "τ", "p", " + ", "Σ", "n", "H", "σ", "n"
            ]
        );
        let (x, a) = (rect_of(&texts, "x"), rect_of(&texts, "a"));
        assert!(
            a.left() >= x.right() - 0.5,
            "the subscript follows its base"
        );
        assert!(a.center().y > x.center().y, "and sits lower");
        assert!(a.height() < x.height(), "and is smaller");
        // The fraction: c_end over the bar over 2.
        let (c, two) = (rect_of(&texts, "c"), rect_of(&texts, "2"));
        assert!(
            c.bottom() <= two.top(),
            "the numerator is above the denominator"
        );
        let bar = laid
            .inks
            .iter()
            .find_map(|i| match i {
                Ink::Line { from, to, .. } if from.y == to.y => Some((*from, *to)),
                _ => None,
            })
            .expect("a fraction bar");
        assert!(c.bottom() <= bar.0.y && bar.0.y <= two.top());
        assert!(bar.0.x <= c.left() && c.right() <= bar.1.x);
        // The radical's overline runs over τ_p.
        let tau = rect_of(&texts, "τ");
        let radical = laid
            .inks
            .iter()
            .find_map(|i| match i {
                Ink::Path { points, .. } if points.len() == 5 => Some(points.clone()),
                _ => None,
            })
            .expect("a radical");
        let overline_y = radical[3].y;
        assert!(
            overline_y <= tau.top(),
            "the overline is above the radicand"
        );
        assert!(radical[3].x <= tau.left() && tau.right() <= radical[4].x);
        // Σ is larger than the text, with n and H beneath it.
        let sigma = rect_of(&texts, "Σ");
        assert!(sigma.height() > x.height());
        let n = rect_of(&texts, "n");
        assert!(n.top() >= sigma.bottom() - 2.0, "n ∈ H is beneath the Σ");
        assert!(laid.width > 0.0 && laid.ascent > 0.0 && laid.descent > 0.0);
    }

    #[test]
    fn terms_carry_their_colours_and_paths() {
        let eq = registry().equation_for("model.pullout_Nm").unwrap();
        let colors = TermColors::of(eq);
        // In the order the formula shows them: T_2D, f_end, f_cal.
        assert_eq!(colors.get("model.torque_2d_Nm"), Some(TERM_PALETTE[0]));
        assert_eq!(colors.get("model.f_end"), Some(TERM_PALETTE[1]));
        assert_eq!(colors.get("model.f_cal"), Some(TERM_PALETTE[2]));
        let laid =
            with_fonts(|f| layout_equation(f, registry(), eq, &colors, 16.0, Color32::WHITE));
        let paths: Vec<String> = laid.term_rects().into_iter().map(|(p, _)| p).collect();
        assert!(paths.contains(&"model.f_end".to_owned()), "{paths:?}");
        let f_end_color = laid.inks.iter().find_map(|i| match i {
            Ink::Text {
                galley,
                term: Some(t),
                ..
            } if t == "model.f_end" => galley.job.sections.first().map(|s| s.format.color),
            _ => None,
        });
        assert_eq!(f_end_color, Some(TERM_PALETTE[1]));
    }

    #[test]
    fn a_ninth_term_takes_the_first_colour_again() {
        // C_th reads 14 terms: eight colours, then the first again (decision M43-3), in the
        // formula tree's order, where a cases arm's condition (the circuit in effect) comes
        // before its value.
        let eq = registry()
            .equation_for("temperature.thermal.heat_capacity_J_K")
            .unwrap();
        let colors = TermColors::of(eq);
        assert_eq!(
            colors.get("materials.circuit_backiron"),
            Some(TERM_PALETTE[0])
        );
        assert_eq!(colors.get("mass.magnets_g"), Some(TERM_PALETTE[1]));
        assert_eq!(
            colors.get("temperature.thermal.steel_c"),
            Some(TERM_PALETTE[0]),
            "the ninth term"
        );
        assert_eq!(
            colors.get("retainers.retainers_g"),
            Some(TERM_PALETTE[1]),
            "the tenth"
        );
    }

    #[test]
    fn only_the_parentheses_the_markup_writes_are_drawn() {
        // T_pull = T_2D f_end f_cal: a product of products, no parentheses.
        let eq = registry().equation_for("model.pullout_Nm").unwrap();
        let laid = with_fonts(|f| {
            layout_equation(f, registry(), eq, &TermColors::none(), 14.0, Color32::WHITE)
        });
        let texts: Vec<String> = laid.texts().into_iter().map(|(t, _)| t).collect();
        assert!(!texts.iter().any(|t| t == "(" || t == ")"), "{texts:?}");
        // C_cup's markup writes them: 2 (r_corner + t_wall).
        let eq = registry().equation_for("model.cup_od_mm").unwrap();
        let laid = with_fonts(|f| {
            layout_equation(f, registry(), eq, &TermColors::none(), 14.0, Color32::WHITE)
        });
        let texts: Vec<String> = laid.texts().into_iter().map(|(t, _)| t).collect();
        assert_eq!(texts.iter().filter(|t| *t == "(").count(), 1, "{texts:?}");
        assert_eq!(texts.iter().filter(|t| *t == ")").count(), 1);
    }

    #[test]
    fn a_family_term_is_coloured_by_template_and_marks_only_the_harmonics_summed() {
        let eq = registry().equation_for("model.tau_Pa").unwrap();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let design = Design {
            inputs: &inputs,
            results: &results,
        };
        let colors = TermColors::of(eq).with_members(registry(), &design);
        let template = colors.get("model.tau#_Pa").expect("the template");
        for n in [1, 3, 5] {
            assert_eq!(colors.get(&format!("model.tau{n}_Pa")), Some(template));
        }
        // The workbook sums 1, 3, 5: 7 is not marked.
        assert_eq!(colors.get("model.tau7_Pa"), None);
        assert!(TermColors::none().is_empty());
        // Every odd harmonic up to 11: all six are marked.
        let mut all = DesignInputs::default();
        all.coupling.max_harmonic = 11;
        let results = compute_all(&all);
        let design = Design {
            inputs: &all,
            results: &results,
        };
        let colors = TermColors::of(eq).with_members(registry(), &design);
        for n in [1, 3, 5, 7, 9, 11] {
            assert_eq!(colors.get(&format!("model.tau{n}_Pa")), Some(template));
        }
        // A code outside the choices sums nothing: only the template keeps its colour.
        let mut odd = DesignInputs::default();
        odd.coupling.max_harmonic = 4;
        let results = compute_all(&odd);
        let design = Design {
            inputs: &odd,
            results: &results,
        };
        let colors = TermColors::of(eq).with_members(registry(), &design);
        assert_eq!(colors.get("model.tau1_Pa"), None);
        assert_eq!(colors.get("model.tau#_Pa"), Some(template));
    }

    #[test]
    fn a_selector_code_shows_its_choice_and_a_screw_row_its_size() {
        // A_1 compares the circuit in effect with code 1, the steel circuit.
        let eq = registry().equation_for("model.amp1_Pa").unwrap();
        let laid = with_fonts(|f| {
            layout_equation(f, registry(), eq, &TermColors::none(), 14.0, Color32::WHITE)
        });
        let texts: Vec<String> = laid.texts().into_iter().map(|(t, _)| t).collect();
        assert!(texts.iter().any(|t| t == "\"steel circuit\""), "{texts:?}");
        // σ_5 = A_5 · sin(5 φ_pull) if 5 ≤ N_h: an ordering keeps its number, though N_h is a
        // selector whose code 5 reads "1, 3, 5 (workbook)" (the index 5 is no code). The tau
        // records hold no text literal, so no run is quoted.
        let tau = registry().equation_for("model.tau5_Pa").unwrap();
        let laid = with_fonts(|f| {
            layout_equation(
                f,
                registry(),
                tau,
                &TermColors::none(),
                14.0,
                Color32::WHITE,
            )
        });
        let texts: Vec<String> = laid.texts().into_iter().map(|(t, _)| t).collect();
        assert!(!texts.iter().any(|t| t.starts_with('"')), "{texts:?}");
        let screw = registry()
            .equation_for("clamps.table[2].engagement_req_mm")
            .expect("a screw-size row");
        let laid = with_fonts(|f| {
            layout_equation(
                f,
                registry(),
                screw,
                &TermColors::none(),
                14.0,
                Color32::WHITE,
            )
        });
        let texts: Vec<String> = laid.texts().into_iter().map(|(t, _)| t).collect();
        assert!(texts.iter().any(|t| t == "M4"), "{texts:?}");
    }

    /// The quoted labels of the selectors `cond` compares for equality with a code, as the
    /// typesetter draws them.
    fn equality_labels(cond: &Cond, out: &mut Vec<String>) {
        match cond {
            Cond::And(a, b) | Cond::Or(a, b) => {
                equality_labels(a, out);
                equality_labels(b, out);
            }
            Cond::Rel(RelOp::Eq | RelOp::Ne, a, b) => {
                for side in [a, b] {
                    if let Expr::Term(r) = side {
                        out.extend(
                            registry()
                                .choices(&r.path)
                                .iter()
                                .map(|(_, label)| glyph_safe(&format!("\"{label}\""))),
                        );
                    }
                }
            }
            Cond::Rel(..) => {}
        }
    }

    #[test]
    fn every_equation_draws_only_what_its_record_says() {
        // The glyph test checks characters, not meaning. Here every text run the typesetter
        // draws must come from the record's plain rendering (`render::plain`, with θ for ϑ as
        // drawn), from a choice label of a selector the formula compares for equality, from
        // the size name of a `screw_sizes` row it reads, or from the words only the typesetter
        // writes. A stray substitution is a run with no source.
        const TYPESET_ONLY: [&str; 3] = ["and", "arg max", "0 ≤ φ ≤ π/2"];
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        for eq in registry().equations() {
            let mut sources = vec![glyph_safe(&render::plain(
                registry(),
                &eq.symbol,
                &eq.formula,
            ))];
            eq.formula.visit(&mut |e| match e {
                Expr::Cases { arms, .. } => {
                    for (cond, _) in arms {
                        equality_labels(cond, &mut sources);
                    }
                }
                Expr::Table { table, key, .. } if table == "screw_sizes" => {
                    if let Expr::Num { value, .. } = &**key
                        && let Ok(Some(Value::Text(name))) =
                            tables::lookup("screw_sizes", &Value::Num(*value), "name")
                    {
                        sources.push(name);
                    }
                }
                _ => {}
            });
            let laid = ctx.fonts(|f| {
                layout_equation(f, registry(), eq, &TermColors::none(), 14.0, Color32::WHITE)
            });
            for (text, _) in laid.texts() {
                let run = text.trim();
                assert!(
                    run.is_empty()
                        || TYPESET_ONLY.contains(&run)
                        || sources.iter().any(|s| s.contains(run)),
                    "{}: {run:?} is in none of {sources:?}",
                    eq.target
                );
            }
        }
    }

    #[test]
    fn every_equation_typesets_with_glyphs_in_the_default_fonts() {
        // Every Expr variant the records use, at a tooltip's size and the panel's: nothing
        // panics, every box is finite, and every character drawn has a glyph (egui draws an
        // empty box for one it lacks).
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let mut checked = 0;
        for eq in registry().equations() {
            for size in [14.0, 22.0] {
                let colors = TermColors::of(eq);
                let laid = ctx
                    .fonts(|f| layout_equation(f, registry(), eq, &colors, size, Color32::WHITE));
                assert!(
                    laid.width.is_finite() && laid.width > 0.0,
                    "{}: width {}",
                    eq.target,
                    laid.width
                );
                assert!(laid.ascent.is_finite() && laid.descent.is_finite());
                for (text, rect) in laid.texts() {
                    assert!(rect.is_finite(), "{}: {text:?} at {rect:?}", eq.target);
                    assert_glyphs(&ctx, &text, &eq.target);
                }
            }
            checked += 1;
        }
        assert_eq!(checked, registry().equations().len());
        assert!(checked > 300, "{checked}");
    }

    #[test]
    fn theta_is_drawn_for_the_vartheta_the_fonts_lack() {
        assert_eq!(glyph_safe("ϑ_{op}"), "θ_{op}");
        assert_eq!(glyph_safe("T_{pull}"), "T_{pull}");
        assert_eq!(glyph_safe("13 × 10⁻⁶ per °C"), "13 × 10¯⁶ per °C");
        assert_eq!(glyph_safe("torque ∝ Br²"), "torque ~ Br²");
    }

    #[test]
    fn clicking_a_term_of_a_drawn_equation_names_it() {
        let ctx = egui::Context::default();
        let eq = registry().equation_for("model.pullout_Nm").unwrap();
        let colors = TermColors::of(eq);
        // One frame drawing the equation: where it was drawn, its layout, the term clicked.
        let run = |events: Vec<egui::Event>| {
            let mut drawn = (Pos2::ZERO, Laid::default(), None);
            let input = egui::RawInput {
                events,
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800.0, 400.0))),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let origin = ui.cursor().min;
                    let laid = ui.fonts(|f| {
                        layout_equation(f, registry(), eq, &colors, 20.0, Color32::WHITE)
                    });
                    let response = laid_ui(ui, &laid, Color32::WHITE, Sense::click());
                    let clicked = response
                        .interact_pointer_pos()
                        .filter(|_| response.clicked())
                        .and_then(|at| term_at(&laid, response.rect.min, at));
                    drawn = (origin, laid, clicked);
                });
            });
            drawn
        };
        let (origin, laid, _) = run(Vec::new());
        let (_, rect) = laid
            .term_rects()
            .into_iter()
            .find(|(p, _)| p == "model.f_end")
            .unwrap();
        let at = rect.translate(origin.to_vec2()).center();
        run(vec![egui::Event::PointerMoved(at)]);
        run(vec![crate::gui::test_support::primary_button(at, true)]);
        let (_, _, clicked) = run(vec![crate::gui::test_support::primary_button(at, false)]);
        assert_eq!(clicked.as_deref(), Some("model.f_end"));
    }
}
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | tail -1
```

Expected: FAIL to compile. The last line is `` error: could not compile `magcoupling-rs` (lib test) due to 83 previous errors; 1 warning emitted ``; the errors above it come from the tests reaching for what Step 3 adds, among them `` error[E0412]: cannot find type `Registry` in this scope ``, `` error[E0433]: failed to resolve: use of undeclared type `Registry` ``, `` error[E0412]: cannot find type `Fonts` in this scope ``, `` error[E0412]: cannot find type `Rect` in this scope ``, `` error[E0433]: failed to resolve: use of undeclared type `Color32` ``.

- [ ] **Step 3: Write the implementation**

The typesetter between the module's docs and its tests.

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/typeset.rs`, replace:

````rust

#[cfg(test)]
mod tests {
````

with:

````rust

use std::collections::BTreeMap;
use std::sync::Arc;

use egui::text::{Fonts, Galley};
use egui::{Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2, vec2};

use crate::engine::explain::markup::{BinOp, Cond, Expr, Formula, Func, RelOp, Symbol};
use crate::engine::explain::{Equation, Registry, TermSource, tables};
use crate::engine::meta::Value;

/// The term colours, in the order the formula tree first reads the terms (decision M43-3; a
/// `cases` arm's condition before its value): distinct on the dark theme, none of them the
/// theme's text colour; a ninth term takes the first again.
pub const TERM_PALETTE: [Color32; 8] = [
    Color32::from_rgb(240, 160, 40),
    Color32::from_rgb(90, 180, 240),
    Color32::from_rgb(80, 200, 140),
    Color32::from_rgb(230, 210, 80),
    Color32::from_rgb(200, 130, 230),
    Color32::from_rgb(240, 110, 90),
    Color32::from_rgb(120, 150, 250),
    Color32::from_rgb(230, 120, 180),
];

/// How much smaller a script is than its base.
const SCRIPT_SCALE: f32 = 0.7;

/// The smallest text drawn [points].
const MIN_SIZE: f32 = 8.0;

/// The colour of each term an equation shows: a term by its path, a family term inside a Σ by
/// its template (`model.b_i#`) and, once [`TermColors::with_members`] has run, by the paths of
/// the harmonics summed, so a mark on screen finds the members' values too.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TermColors {
    colors: BTreeMap<String, Color32>,
}

impl TermColors {
    /// No colours: every term in the text colour.
    pub fn none() -> Self {
        Self::default()
    }

    /// One palette colour per term of `eq` in the formula tree's order ([`Formula::visit`]: the
    /// body, a `cases` arm's condition before its value, then each `where` binding): a path,
    /// or a family template inside a Σ.
    pub fn of(eq: &Equation) -> Self {
        let mut colors = BTreeMap::new();
        eq.formula.visit(&mut |e| {
            if let Expr::Term(r) | Expr::FamilyTerm(r) = e {
                let next = TERM_PALETTE[colors.len() % TERM_PALETTE.len()];
                colors.entry(r.path.clone()).or_insert(next);
            }
        });
        Self { colors }
    }

    /// Adds, for each family template, the paths of the harmonics the design sums
    /// ([`Registry::family_members`]) in the template's colour.
    pub fn with_members(mut self, registry: &Registry, src: &dyn TermSource) -> Self {
        let templates: Vec<(String, Color32)> = self
            .colors
            .iter()
            .filter(|(path, _)| path.contains('#'))
            .map(|(path, color)| (path.clone(), *color))
            .collect();
        for (template, color) in templates {
            for member in registry.family_members(&template, src) {
                self.colors.entry(member).or_insert(color);
            }
        }
        self
    }

    /// The colour of a path or template, if the equation shows it.
    pub fn get(&self, path: &str) -> Option<Color32> {
        self.colors.get(path).copied()
    }

    /// Whether no term has a colour.
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }
}

/// One thing the typesetter draws, placed relative to its box's top-left corner.
#[derive(Clone, Debug)]
pub enum Ink {
    /// A text run; `term` is the path (or template) it shows, for clicks.
    Text {
        at: Vec2,
        galley: Arc<Galley>,
        term: Option<String>,
    },
    /// A straight stroke `width` points wide (a fraction bar, a radical's overline).
    Line { from: Vec2, to: Vec2, width: f32 },
    /// A polyline (a radical sign, ∈, a bracket).
    Path { points: Vec<Vec2>, width: f32 },
}

impl Ink {
    fn shifted(self, by: Vec2) -> Ink {
        match self {
            Ink::Text { at, galley, term } => Ink::Text {
                at: at + by,
                galley,
                term,
            },
            Ink::Line { from, to, width } => Ink::Line {
                from: from + by,
                to: to + by,
                width,
            },
            Ink::Path { points, width } => Ink::Path {
                points: points.into_iter().map(|p| p + by).collect(),
                width,
            },
        }
    }
}

/// A laid-out box: `width`, `ascent` above the baseline and `descent` below it, and what it
/// draws (the baseline is at `y = ascent`).
#[derive(Clone, Debug, Default)]
pub struct Laid {
    pub width: f32,
    pub ascent: f32,
    pub descent: f32,
    pub inks: Vec<Ink>,
}

impl Laid {
    /// The box's size.
    pub fn size(&self) -> Vec2 {
        vec2(self.width, self.ascent + self.descent)
    }

    /// Every text run in drawing order, with its rect in the box.
    pub fn texts(&self) -> Vec<(String, Rect)> {
        self.inks
            .iter()
            .filter_map(|ink| match ink {
                Ink::Text { at, galley, .. } => Some((
                    galley.text().to_owned(),
                    Rect::from_min_size(Pos2::ZERO + *at, galley.size()),
                )),
                _ => None,
            })
            .collect()
    }

    /// The rect of each term's text runs in the box, by path (a symbol's base and scripts are
    /// separate runs of one term).
    pub fn term_rects(&self) -> Vec<(String, Rect)> {
        self.inks
            .iter()
            .filter_map(|ink| match ink {
                Ink::Text {
                    at,
                    galley,
                    term: Some(term),
                } => Some((
                    term.clone(),
                    Rect::from_min_size(Pos2::ZERO + *at, galley.size()),
                )),
                _ => None,
            })
            .collect()
    }

    fn gap(width: f32) -> Laid {
        Laid {
            width,
            ..Laid::default()
        }
    }

    /// The boxes side by side on one baseline.
    fn row(parts: Vec<Laid>) -> Laid {
        let ascent = parts.iter().map(|p| p.ascent).fold(0.0, f32::max);
        let descent = parts.iter().map(|p| p.descent).fold(0.0, f32::max);
        let mut x = 0.0;
        let mut inks = Vec::new();
        for p in parts {
            let by = vec2(x, ascent - p.ascent);
            inks.extend(p.inks.into_iter().map(|i| i.shifted(by)));
            x += p.width;
        }
        Laid {
            width: x,
            ascent,
            descent,
            inks,
        }
    }

    /// The boxes one under another, left-aligned, `gap` points apart; the baseline is the
    /// first box's.
    fn column(parts: Vec<Laid>, gap: f32) -> Laid {
        let mut y = 0.0;
        let mut inks = Vec::new();
        let mut width: f32 = 0.0;
        let ascent = parts.first().map_or(0.0, |p| p.ascent);
        let count = parts.len();
        for (i, p) in parts.into_iter().enumerate() {
            width = width.max(p.width);
            let h = p.ascent + p.descent;
            inks.extend(p.inks.into_iter().map(|ink| ink.shifted(vec2(0.0, y))));
            y += h;
            if i + 1 < count {
                y += gap;
            }
        }
        Laid {
            width,
            ascent,
            descent: y - ascent,
            inks,
        }
    }
}

/// The text of `s` as the default fonts can draw it (decision M43-6): ϑ (U+03D1) as θ, the same
/// letter; the superscript minus (U+207B) as ¯, the raised bar it looks like (10¯⁶); ∝
/// (U+221D) as ~.
pub fn glyph_safe(s: &str) -> String {
    s.replace('\u{3d1}', "\u{3b8}")
        .replace('\u{207b}', "\u{af}")
        .replace('\u{221d}', "~")
}

/// The typesetter's state for one formula.
struct Setter<'a> {
    fonts: &'a Fonts,
    registry: &'a Registry,
    formula: &'a Formula,
    colors: &'a TermColors,
    ink: Color32,
}

impl Setter<'_> {
    fn text(&self, s: &str, size: f32, color: Color32, term: Option<&str>) -> Laid {
        let galley = self.fonts.layout_no_wrap(
            glyph_safe(s),
            FontId::proportional(size.max(MIN_SIZE)),
            color,
        );
        let height = galley.size().y;
        let baseline = galley
            .rows
            .first()
            .and_then(|r| r.row.glyphs.first().map(|g| r.pos.y + g.pos.y))
            .unwrap_or(height * 0.8);
        Laid {
            width: galley.size().x,
            ascent: baseline,
            descent: height - baseline,
            inks: vec![Ink::Text {
                at: Vec2::ZERO,
                galley,
                term: term.map(str::to_owned),
            }],
        }
    }

    /// Plain text in the ink colour.
    fn plain(&self, s: &str, size: f32) -> Laid {
        self.text(s, size, self.ink, None)
    }

    fn script_size(size: f32) -> f32 {
        (size * SCRIPT_SCALE).max(MIN_SIZE)
    }

    /// `base` with a subscript and a superscript stacked after it.
    fn scripts(base: Laid, sub: Option<Laid>, sup: Option<Laid>, size: f32) -> Laid {
        let raise = 0.42 * size;
        let lower = 0.28 * size;
        let mut ascent = base.ascent;
        let mut descent = base.descent;
        if let Some(s) = &sup {
            ascent = ascent.max(raise + s.ascent);
        }
        if let Some(s) = &sub {
            descent = descent.max(lower + s.descent);
        }
        let x = base.width;
        let mut width = x;
        let mut inks: Vec<Ink> = base
            .inks
            .into_iter()
            .map(|i| i.shifted(vec2(0.0, ascent - base.ascent)))
            .collect();
        if let Some(s) = sub {
            width = width.max(x + s.width);
            let by = vec2(x, ascent + lower - s.ascent);
            inks.extend(s.inks.into_iter().map(|i| i.shifted(by)));
        }
        if let Some(s) = sup {
            width = width.max(x + s.width);
            let by = vec2(x, ascent - raise - s.ascent);
            inks.extend(s.inks.into_iter().map(|i| i.shifted(by)));
        }
        Laid {
            width,
            ascent,
            descent,
            inks,
        }
    }

    /// A display symbol (symbol markup) in `color`, with `power` stacked over its subscript.
    fn symbol(
        &self,
        markup: &str,
        size: f32,
        color: Color32,
        term: Option<&str>,
        power: Option<Laid>,
    ) -> Laid {
        let symbol = Symbol::parse(markup).unwrap_or(Symbol {
            base: markup.to_owned(),
            sub: None,
            sup: None,
        });
        let small = Self::script_size(size);
        let base = self.text(&symbol.base, size, color, term);
        let sub = symbol.sub.map(|s| self.text(&s, small, color, term));
        let own_sup = symbol.sup.map(|s| self.text(&s, small, color, term));
        // A symbol with its own superscript is never raised to a power (the registry refuses
        // it), so at most one of the two is set.
        let sup = power.or(own_sup);
        Self::scripts(base, sub, sup, size)
    }

    /// `content` between `open` and `close` (either may be empty), the delimiters scaled to
    /// its height and centred on it.
    fn fenced(&self, open: &str, content: Laid, close: &str, size: f32) -> Laid {
        let height = content.ascent + content.descent;
        let dsize = size.max(height * 0.85);
        let centre = (content.descent - content.ascent) / 2.0;
        let delim = |s: &str| {
            let mut d = self.plain(s, dsize);
            let h = d.ascent + d.descent;
            d.ascent = h / 2.0 - centre;
            d.descent = h / 2.0 + centre;
            d
        };
        let mut parts = Vec::new();
        if !open.is_empty() {
            parts.push(delim(open));
        }
        parts.push(content);
        if !close.is_empty() {
            parts.push(delim(close));
        }
        Laid::row(parts)
    }

    /// `content` between ceiling (`ceil`) or floor brackets, drawn with strokes.
    fn brackets(&self, content: Laid, ceil: bool, size: f32) -> Laid {
        let w = 0.3 * size;
        let pad = 0.1 * size;
        let top = 0.0;
        let bottom = content.ascent + content.descent;
        let stroke = (size / 14.0).max(1.0);
        let serif = if ceil { top } else { bottom };
        let width = w + pad + content.width + pad + w;
        let right = width - 0.1 * size;
        let left = 0.1 * size;
        let mut inks = vec![
            Ink::Path {
                points: vec![
                    vec2(left + w * 0.6, serif),
                    vec2(left, serif),
                    vec2(left, if ceil { bottom } else { top }),
                ],
                width: stroke,
            },
            Ink::Path {
                points: vec![
                    vec2(right - w * 0.6, serif),
                    vec2(right, serif),
                    vec2(right, if ceil { bottom } else { top }),
                ],
                width: stroke,
            },
        ];
        let ascent = content.ascent;
        let descent = content.descent;
        inks.extend(
            content
                .inks
                .into_iter()
                .map(|i| i.shifted(vec2(w + pad, 0.0))),
        );
        Laid {
            width,
            ascent,
            descent,
            inks,
        }
    }

    /// A stacked fraction, its bar on the math axis.
    fn frac(num: Laid, den: Laid, size: f32) -> Laid {
        let pad = 0.15 * size;
        let gap = 0.12 * size;
        let axis = 0.28 * size;
        let width = num.width.max(den.width) + 2.0 * pad;
        let num_h = num.ascent + num.descent;
        let den_h = den.ascent + den.descent;
        let ascent = axis + gap + num_h;
        let descent = (den_h + gap - axis).max(0.0);
        let bar = ascent - axis;
        let mut inks: Vec<Ink> = num
            .inks
            .into_iter()
            .map(|i| i.shifted(vec2((width - num.width) / 2.0, bar - gap - num_h)))
            .collect();
        inks.push(Ink::Line {
            from: vec2(0.0, bar),
            to: vec2(width, bar),
            width: (size / 14.0).max(1.0),
        });
        inks.extend(
            den.inks
                .into_iter()
                .map(|i| i.shifted(vec2((width - den.width) / 2.0, bar + gap))),
        );
        Laid {
            width,
            ascent,
            descent,
            inks,
        }
    }

    /// √ drawn with strokes: a tick, the sign down and up to the overline over `body`.
    fn sqrt(body: Laid, size: f32) -> Laid {
        let gap = 0.12 * size;
        let stroke = (size / 14.0).max(1.0);
        let sign = 0.55 * size;
        let over = body.ascent + gap;
        let ascent = over + stroke;
        let descent = body.descent;
        let base = ascent;
        let width = sign + body.width + 0.15 * size;
        let mut inks = vec![Ink::Path {
            points: vec![
                vec2(0.0, base - 0.3 * size),
                vec2(0.15 * size, base - 0.38 * size),
                vec2(0.3 * size, base + descent),
                vec2(sign, base - over),
                vec2(width, base - over),
            ],
            width: stroke,
        }];
        inks.extend(
            body.inks
                .into_iter()
                .map(|i| i.shifted(vec2(sign + 0.05 * size, ascent - body.ascent))),
        );
        Laid {
            width,
            ascent,
            descent,
            inks,
        }
    }

    /// ∈ drawn with strokes (the default fonts lack U+2208), the height of a small letter: an
    /// arc open to the right and a bar through its middle.
    fn element(size: f32) -> Laid {
        let r = 0.25 * size;
        let cx = 0.1 * size + r;
        let tip = cx + 0.15 * size;
        let stroke = (size / 14.0).max(1.0);
        let mut points = vec![vec2(tip, 0.0)];
        points.extend((0..=12).map(|i| {
            let a = std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / 12.0;
            vec2(cx + r * a.cos(), r - r * a.sin())
        }));
        points.push(vec2(tip, 2.0 * r));
        Laid {
            width: tip + 0.1 * size,
            ascent: 2.0 * r,
            descent: 0.0,
            inks: vec![
                Ink::Path {
                    points,
                    width: stroke,
                },
                Ink::Line {
                    from: vec2(cx - r, r),
                    to: vec2(tip, r),
                    width: stroke,
                },
            ],
        }
    }

    /// `op` with `under` centred beneath it; the baseline is `op`'s.
    fn under(op: Laid, under: Laid, size: f32) -> Laid {
        let width = op.width.max(under.width);
        let gap = 0.05 * size;
        let op_h = op.ascent + op.descent;
        let mut inks: Vec<Ink> = op
            .inks
            .into_iter()
            .map(|i| i.shifted(vec2((width - op.width) / 2.0, 0.0)))
            .collect();
        inks.extend(
            under
                .inks
                .into_iter()
                .map(|i| i.shifted(vec2((width - under.width) / 2.0, op_h + gap))),
        );
        Laid {
            width,
            ascent: op.ascent,
            descent: op.descent + gap + under.ascent + under.descent,
            inks,
        }
    }

    /// Σ with `n ∈ H` beneath, then the body.
    fn sum(&self, body: Laid, size: f32) -> Laid {
        let small = (size * 0.6).max(MIN_SIZE);
        let sigma = self.plain("Σ", size * 1.5);
        let set = Laid::row(vec![
            self.plain("n", small),
            Self::element(small),
            self.plain("H", small),
        ]);
        let op = Self::under(sigma, set, size);
        Laid::row(vec![op, Laid::gap(0.15 * size), body])
    }

    /// The amplitude of a `peak`, wrapped when compound so that `sin(nφ)` multiplies all of it
    /// (as `render::plain`). Elsewhere the typesetter draws only the parentheses the markup
    /// writes (the registry refuses markup that would read two ways).
    fn grouped(&self, e: &Expr, size: f32) -> Laid {
        let atomic = matches!(
            e,
            Expr::Num { .. }
                | Expr::Text(_)
                | Expr::NoneLit
                | Expr::Pi
                | Expr::Term(_)
                | Expr::FamilyTerm(_)
                | Expr::Index
                | Expr::Local(_)
                | Expr::Paren(_)
                | Expr::Call(..)
                | Expr::Table { .. }
                | Expr::Frac(..)
        );
        let laid = self.expr(e, size);
        if atomic {
            laid
        } else {
            self.fenced("(", laid, ")", size)
        }
    }

    fn term(&self, path: &str, family: bool, size: f32, power: Option<Laid>) -> Laid {
        let symbol = if family {
            self.registry.family_symbol(path)
        } else {
            self.registry.symbol(path).map(str::to_owned)
        };
        let color = self.colors.get(path).unwrap_or(self.ink);
        match symbol {
            Some(s) => self.symbol(&s, size, color, Some(path), power),
            None => {
                let laid = self.text(&format!("[{path}]"), size, color, Some(path));
                match power {
                    Some(p) => Self::scripts(laid, None, Some(p), size),
                    None => laid,
                }
            }
        }
    }

    fn call(&self, f: Func, args: &[Expr], size: f32) -> Laid {
        let arg = |i: usize| self.expr(&args[i], size);
        match f {
            Func::Sqrt => Self::sqrt(arg(0), size),
            Func::Exp => {
                let e = self.plain("e", size);
                Self::scripts(
                    e,
                    None,
                    Some(self.expr(&args[0], Self::script_size(size))),
                    size,
                )
            }
            Func::Abs => self.fenced("|", arg(0), "|", size),
            Func::Ceil => self.brackets(arg(0), true, size),
            Func::Floor => self.brackets(arg(0), false, size),
            Func::CeilTo | Func::FloorTo => {
                let b = self.brackets(arg(0), f == Func::CeilTo, size);
                let step = self.expr(&args[1], Self::script_size(size));
                Self::scripts(b, Some(step), None, size)
            }
            Func::Fmt => Laid::row(vec![
                arg(0),
                Laid::gap(0.2 * size),
                self.text(
                    &format!("(to {} decimals)", render_number(&args[1])),
                    Self::script_size(size),
                    self.ink.gamma_multiply(0.7),
                    None,
                ),
            ]),
            Func::FmtNum => arg(0),
            Func::Concat => {
                let mut parts = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        parts.push(Laid::gap(0.25 * size));
                    }
                    parts.push(self.expr(a, size));
                }
                Laid::row(parts)
            }
            _ => {
                let mut inner = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        inner.push(self.plain(", ", size));
                    }
                    inner.push(self.expr(a, size));
                }
                let name = self.plain(f.name(), size);
                let gap = Laid::gap(0.1 * size);
                Laid::row(vec![
                    name,
                    gap,
                    self.fenced("(", Laid::row(inner), ")", size),
                ])
            }
        }
    }

    /// A condition of a `cases` arm; a selector compared for equality with a code shows the
    /// choice label.
    fn cond(&self, c: &Cond, size: f32) -> Laid {
        match c {
            Cond::And(a, b) | Cond::Or(a, b) => {
                let word = if matches!(c, Cond::And(..)) {
                    " and "
                } else {
                    " or "
                };
                Laid::row(vec![
                    self.cond(a, size),
                    self.plain(word, size),
                    self.cond(b, size),
                ])
            }
            Cond::Rel(op, a, b) => {
                // A label reads only in an equality: σ_n's n ≤ N_h compares the harmonic index
                // with the highest harmonic, not a code.
                let labelled = matches!(op, RelOp::Eq | RelOp::Ne);
                let op = match op {
                    RelOp::Lt => " < ",
                    RelOp::Le => " ≤ ",
                    RelOp::Gt => " > ",
                    RelOp::Ge => " ≥ ",
                    RelOp::Eq => " = ",
                    RelOp::Ne => " ≠ ",
                };
                let left = if labelled {
                    self.choice_or_expr(a, b, size)
                } else {
                    self.expr(a, size)
                };
                let right = if labelled {
                    self.choice_or_expr(b, a, size)
                } else {
                    self.expr(b, size)
                };
                Laid::row(vec![left, self.plain(op, size), right])
            }
        }
    }

    /// `e`, or when `e` is a code compared for equality with a selector (`other`), the choice's
    /// label.
    fn choice_or_expr(&self, e: &Expr, other: &Expr, size: f32) -> Laid {
        if let (Expr::Num { value, .. }, Expr::Term(r)) = (e, other) {
            let choices = self.registry.choices(&r.path);
            if let Some((_, label)) = choices.iter().find(|(code, _)| *code as f64 == *value) {
                return self.plain(&format!("\"{label}\""), size);
            }
        }
        self.expr(e, size)
    }

    fn expr(&self, e: &Expr, size: f32) -> Laid {
        match e {
            Expr::Num { text, .. } => self.plain(text, size),
            Expr::Text(t) => self.plain(&format!("\"{t}\""), size),
            Expr::NoneLit => self.plain("none", size),
            Expr::Pi => self.plain("π", size),
            Expr::Term(r) => self.term(&r.path, false, size, None),
            Expr::FamilyTerm(r) => self.term(&r.path, true, size, None),
            Expr::Index => self.plain("n", size),
            Expr::Local(k) => {
                let symbol = &self.formula.bindings[*k].symbol;
                self.symbol(symbol, size, self.ink, None, None)
            }
            Expr::Neg(a) => Laid::row(vec![self.plain("−", size), self.expr(a, size)]),
            Expr::Paren(a) => self.fenced("(", self.expr(a, size), ")", size),
            Expr::Bin(op, a, b) => self.binary(*op, a, b, size),
            Expr::Frac(a, b) => {
                let inner = (size * 0.9).max(MIN_SIZE + 1.0).min(size);
                Self::frac(self.expr(a, inner), self.expr(b, inner), size)
            }
            Expr::Call(f, args) => self.call(*f, args, size),
            Expr::Sum(_, body) => self.sum(self.expr(body, size), size),
            Expr::Peak(_, body) => {
                let small = (size * 0.6).max(MIN_SIZE);
                let argmax = Self::under(
                    self.plain("arg max", size),
                    self.plain("0 ≤ φ ≤ π/2", small),
                    size,
                );
                let body = Laid::row(vec![
                    self.grouped(body, size),
                    Laid::gap(0.15 * size),
                    self.plain("sin(nφ)", size),
                ]);
                Laid::row(vec![argmax, Laid::gap(0.2 * size), self.sum(body, size)])
            }
            Expr::Table { table, key, field } => {
                let symbol = tables::field(table, field).map_or(field.as_str(), |f| f.symbol);
                let name = self.symbol(symbol, size, self.ink, None, None);
                let key = match (table.as_str(), &**key) {
                    ("screw_sizes", Expr::Num { value, .. }) => {
                        match tables::lookup("screw_sizes", &Value::Num(*value), "name") {
                            Ok(Some(Value::Text(size_name))) => self.plain(&size_name, size),
                            _ => self.expr(key, size),
                        }
                    }
                    _ => self.expr(key, size),
                };
                Laid::row(vec![name, self.fenced("(", key, ")", size)])
            }
            Expr::Cases {
                arms, otherwise, ..
            } => self.cases(arms, otherwise, size),
        }
    }

    fn binary(&self, op: BinOp, a: &Expr, b: &Expr, size: f32) -> Laid {
        let thin = Laid::gap(0.17 * size);
        match op {
            BinOp::Add => Laid::row(vec![
                self.expr(a, size),
                self.plain(" + ", size),
                self.expr(b, size),
            ]),
            BinOp::Sub => Laid::row(vec![
                self.expr(a, size),
                self.plain(" − ", size),
                self.expr(b, size),
            ]),
            BinOp::Mul => {
                let numbers = matches!(a, Expr::Num { .. }) && matches!(b, Expr::Num { .. });
                // A family member's index 1 as a factor is not drawn (k_1 = N/(2 R_g)).
                let unit_factor = matches!(a, Expr::Num { value, .. } if *value == 1.0);
                if unit_factor && !numbers {
                    self.expr(b, size)
                } else if numbers {
                    Laid::row(vec![
                        self.expr(a, size),
                        self.plain(" × ", size),
                        self.expr(b, size),
                    ])
                } else {
                    Laid::row(vec![self.expr(a, size), thin, self.expr(b, size)])
                }
            }
            BinOp::Dot => Laid::row(vec![
                self.expr(a, size),
                self.plain(" · ", size),
                self.expr(b, size),
            ]),
            BinOp::Div => Laid::row(vec![
                self.expr(a, size),
                self.plain("/", size),
                self.expr(b, size),
            ]),
            BinOp::Pow => {
                let exponent = match b {
                    Expr::Paren(inner) => &**inner,
                    other => other,
                };
                let sup = self.expr(exponent, Self::script_size(size));
                match a {
                    Expr::Term(r) => self.term(&r.path, false, size, Some(sup)),
                    Expr::FamilyTerm(r) => self.term(&r.path, true, size, Some(sup)),
                    _ => Self::scripts(self.expr(a, size), None, Some(sup), size),
                }
            }
        }
    }

    /// A left brace and one row per arm: the value, then `if` and the condition; the last row
    /// `otherwise`. The block is centred on the math axis.
    fn cases(&self, arms: &[(Cond, Expr)], otherwise: &Expr, size: f32) -> Laid {
        let mut values: Vec<Laid> = arms.iter().map(|(_, v)| self.expr(v, size)).collect();
        values.push(self.expr(otherwise, size));
        let column = values.iter().map(|v| v.width).fold(0.0, f32::max);
        let mut rows = Vec::new();
        let count = values.len();
        for (i, value) in values.into_iter().enumerate() {
            let pad = Laid::gap(column - value.width + 0.8 * size);
            let tail = if i + 1 < count {
                Laid::row(vec![self.plain("if ", size), self.cond(&arms[i].0, size)])
            } else {
                self.plain("otherwise", size)
            };
            rows.push(Laid::row(vec![value, pad, tail]));
        }
        let mut block = Laid::column(rows, 0.25 * size);
        let height = block.ascent + block.descent;
        let axis = 0.28 * size;
        block.ascent = height / 2.0 + axis;
        block.descent = height / 2.0 - axis;
        let brace = self.fenced("{", block, "", size);
        Laid::row(vec![brace, Laid::gap(0.1 * size)])
    }
}

/// The text of a number literal, for `fmt`'s decimals.
fn render_number(e: &Expr) -> String {
    match e {
        Expr::Num { text, .. } => text.clone(),
        _ => "d".to_owned(),
    }
}

/// Lays out `symbol = formula` (then one line per `where` binding) at text size `size`, every
/// term in its colour from `colors` (the text colour `ink` otherwise).
pub fn layout(
    fonts: &Fonts,
    registry: &Registry,
    symbol: &str,
    formula: &Formula,
    colors: &TermColors,
    size: f32,
    ink: Color32,
) -> Laid {
    let setter = Setter {
        fonts,
        registry,
        formula,
        colors,
        ink,
    };
    let target = setter.symbol(symbol, size, ink, None, None);
    let mut lines = vec![Laid::row(vec![
        target,
        setter.plain(" = ", size),
        setter.expr(&formula.body, size),
    ])];
    for (i, b) in formula.bindings.iter().enumerate() {
        let lead = if i == 0 { "where " } else { "and " };
        lines.push(Laid::row(vec![
            Laid::gap(size),
            setter.text(lead, size, ink.gamma_multiply(0.7), None),
            setter.symbol(&b.symbol, size, ink, None, None),
            setter.plain(" = ", size),
            setter.expr(&b.expr, size),
        ]));
    }
    Laid::column(lines, 0.35 * size)
}

/// Lays out the equation of `eq` ([`layout`] with its symbol and formula).
pub fn layout_equation(
    fonts: &Fonts,
    registry: &Registry,
    eq: &Equation,
    colors: &TermColors,
    size: f32,
    ink: Color32,
) -> Laid {
    layout(fonts, registry, &eq.symbol, &eq.formula, colors, size, ink)
}

/// Paints `laid` with its top-left corner at `origin`; strokes in `ink`.
pub fn paint(painter: &egui::Painter, origin: Pos2, laid: &Laid, ink: Color32) {
    for item in &laid.inks {
        match item {
            Ink::Text { at, galley, .. } => {
                painter.galley(origin + *at, galley.clone(), ink);
            }
            Ink::Line { from, to, width } => {
                painter.line_segment([origin + *from, origin + *to], Stroke::new(*width, ink));
            }
            Ink::Path { points, width } => {
                let points: Vec<Pos2> = points.iter().map(|p| origin + *p).collect();
                painter.add(egui::Shape::line(points, Stroke::new(*width, ink)));
            }
        }
    }
}

/// Draws `eq` at text size `size` with its terms in `colors` (a tooltip's equation: it takes
/// no clicks).
pub fn equation_ui(
    ui: &mut egui::Ui,
    registry: &Registry,
    eq: &Equation,
    colors: &TermColors,
    size: f32,
) -> egui::Response {
    let ink = ui.visuals().text_color();
    let laid = ui.fonts(|f| layout_equation(f, registry, eq, colors, size, ink));
    laid_ui(ui, &laid, ink, Sense::hover())
}

/// Allocates room for `laid` with `sense` and paints it there.
pub fn laid_ui(ui: &mut egui::Ui, laid: &Laid, ink: Color32, sense: Sense) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(laid.size(), sense);
    paint(ui.painter(), rect.min, laid, ink);
    response
}

/// The term drawn at `pos` by `laid` painted at `origin` (its path, or a family template).
pub fn term_at(laid: &Laid, origin: Pos2, pos: Pos2) -> Option<String> {
    laid.term_rects()
        .into_iter()
        .find(|(_, r)| r.translate(origin.to_vec2()).contains(pos))
        .map(|(path, _)| path)
}

#[cfg(test)]
mod tests {
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib gui::typeset 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 10 passed; 0 failed` for `gui::typeset`, then `test result: ok. 377 passed; 0 failed` for the whole library (Task 0's 367 plus this plan's tests so far).

- [ ] **Step 5: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m43 add magcoupling-rs/src/gui/typeset.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/gui/test_support.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m43 commit -F - <<'EOF'
feat(magcoupling-rs): an egui typesetter for the equation markup

gui::typeset lays out the A-3 markup's tree as boxes on a baseline (text runs
and strokes): stacked fractions, inline a/b, scripts (a power stacked over a
symbol's subscript), a stroked radical, a large Sigma over "n in H", arg max,
scaled delimiters, stroked ceiling and floor brackets, a brace per cases arm,
the where lines, and only the parentheses the markup writes. A selector
compared for equality with a code shows its choice's label (an ordering keeps
its numbers) and a screw row its size. Each term carries its path and its
palette colour (TermColors, decision M43-3); glyph_safe draws the characters
egui's default fonts lack (decision M43-6). Tests lay out every registry
equation at two sizes against the fonts and trace every drawn text run to the
record's plain rendering; test_support::assert_glyphs is the shared check.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m43 status --short
```

Expected: one new commit on `magcoupling/m4-3`; `status --short` prints nothing. (A `sonnet` executor writes its own model's name in the `Co-Authored-By:` line.)

---

### Task 2: Every displayed value hovers its equation and marks its terms

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

The readout hook (spec A2 "Hover"). `gui/readouts.rs` owns the registry, built once per process (decision M43-5), and `Readouts`, the per-frame context every view now takes: `show` (a response that already senses clicks: the geometry view's dimension lines) and `show_over` (a click sensor over a label's rect: the label keeps its look) frame the value in its term's colour when the equation in view reads its path, show its hover text and its typeset equation only inside `on_hover_ui` with the hint "Click to open it in the Equation panel", and record the hover and the click. The dashboard rows, the results-table rows, the geometry callouts (line and list), the clamp summary and screw table, and a new readout line over each plot (`PlotKind::readouts`, decision M43-11) all go through it; the dashboard's hover text is now built only while hovered (`DashboardLine::tooltip`, the M4-1 review's item: the field becomes a method). The panel keeps the path hovered in the last frame and marks its equation's terms this frame, the input rows included (`Readouts::mark`), and asks for a repaint when the hover changes. The views' tests pass a default `Readouts`. New tests: the hover text, equation and hint of a hovered value, a click recorded, the frame on a term's value only, the plot readouts, and in the panel the dashboard's tooltip and a cup-OD hover framing the Key design wall row in its term's colour, its tooltip drawing t_wall in that same colour, cleared when the pointer leaves.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/readouts.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/mod.rs` (`pub mod readouts;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs` (`dashboard_ui` takes `&mut Readouts`; `DashboardLine::tooltip` built lazily; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/results_table.rs` (`ResultsTable::ui` and `row_ui` take `&mut Readouts`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs` (`geometry_ui`, `paint_view`, `list_ui` take `&mut Readouts`; `hover` becomes `callout_text`; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/clamp_drawing.rs` (`clamp_ui`, `screw_table_ui` take `&mut Readouts`; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/plots.rs` (`PlotKind::readouts`, `readouts_ui`, `plot_ui` takes `&mut Readouts`; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs` (the registry at start-up, `hovered`, `marks`, `end_frame`, the readouts passed to every view and the input rows; tests)

**Interfaces:**
- Consumes: Task 1's `TermColors` (`of`, `with_members`, `get`, `none`) and `equation_ui`; `engine::explain::{Registry, Design}`; `dashboard::{hover_text, result_info, result_tooltip, ResultNotes}`, `corrections::Mark`.
- Produces (`magcoupling::gui::readouts`): `pub fn registry() -> &'static Registry`; `pub const OPEN_HINT: &str`, `TOOLTIP_SIZE: f32` (15), `MARK_WIDTH: f32` (2), `REGISTRY_LOG_PREFIX: &str` (`"magcoupling explorer: "`); `#[derive(Clone, Debug, Default, PartialEq, Eq)] pub struct ReadoutEvents { pub hovered: Option<String>, pub clicked: Option<String>, pub note: Option<&'static str> }`; `#[derive(Clone, Debug, Default)] pub struct Readouts` with `fn new(TermColors) -> Self`, `fn marks(&self) -> &TermColors`, `fn mark(&self, &egui::Ui, Rect, &str)`, `fn show(&mut self, &egui::Ui, egui::Response, &str, impl FnOnce() -> String)`, `fn show_over(&mut self, &egui::Ui, Rect, &str, impl FnOnce() -> String)`, `fn open_note(&mut self, &'static str)`, `fn events(&self) -> &ReadoutEvents`, `fn finish(self) -> ReadoutEvents`.
- Changes: `dashboard_ui(&mut egui::Ui, &DesignResults, &mut Readouts)`; `DashboardLine { .., pub marks: &'static [Mark] }` (the `tooltip` field removed) with `fn tooltip(&self) -> String`; `ResultsTable::ui(&mut self, &mut egui::Ui, &DesignResults, &mut Readouts) -> Option<TableAction>`; `geometry_ui(&mut egui::Ui, &DesignInputs, &DesignResults, &mut Readouts) -> GeometryLayout`; `clamp_ui(&mut egui::Ui, &DesignInputs, &DesignResults, &mut Readouts)`; `plot_ui(&mut egui::Ui, PlotKind, &DesignInputs, &DesignResults, &mut Readouts)`; `PlotKind::readouts(self) -> &'static [&'static str]`. In `panel.rs` (private): the field `hovered: Option<String>`, `fn marks(&self) -> TermColors`, `fn end_frame(&mut self, &egui::Ui, ReadoutEvents)` (Task 3 replaces the three with `Explorer`).

- [ ] **Step 1: Write the failing tests**

The new module's docs and tests; the views' test call sites pass `&mut Readouts::default()`; the dashboard tests read `tooltip()`; the plots' readout test; the panel's hover tests and helpers (`tooltips_at_once`, `mark_rects`).

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/clamp_drawing.rs`, replace:

````rust
        }
        // The whole tab: the summary and the machining steps; no fit shows the message.
        let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
            clamp_ui(ui, &inputs, &r)
        });
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == "ISO 4762 M4 x 14, class 12.9"));
````

with:

````rust
        }
        // The whole tab: the summary and the machining steps; no fit shows the message.
        let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
            clamp_ui(ui, &inputs, &r, &mut Readouts::default())
        });
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == "ISO 4762 M4 x 14, class 12.9"));
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/clamp_drawing.rs`, replace:

````rust
        tight.clamps.clamp_length_mm = 3.0;
        tight.clamps.clamp_type = 2;
        let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
            clamp_ui(ui, &tight, &compute_all(&tight))
        });
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == NO_SCREW_FITS));
````

with:

````rust
        tight.clamps.clamp_length_mm = 3.0;
        tight.clamps.clamp_type = 2;
        let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
            clamp_ui(ui, &tight, &compute_all(&tight), &mut Readouts::default())
        });
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == NO_SCREW_FITS));
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/clamp_drawing.rs`, replace:

````rust
        // A narrow, short region still draws the tab (a ~930 px window leaves about 294
        // points).
        for size in [egui::vec2(294.0, 900.0), egui::vec2(120.0, 90.0)] {
            let output = sized_frame(&ctx, size, Vec::new(), |ui| clamp_ui(ui, &inputs, &r));
            assert!(drawn_texts(&output).iter().any(|t| t == &drawing.title));
        }
        // No room at all: no scale, nothing painted.
````

with:

````rust
        // A narrow, short region still draws the tab (a ~930 px window leaves about 294
        // points).
        for size in [egui::vec2(294.0, 900.0), egui::vec2(120.0, 90.0)] {
            let output = sized_frame(&ctx, size, Vec::new(), |ui| {
                clamp_ui(ui, &inputs, &r, &mut Readouts::default())
            });
            assert!(drawn_texts(&output).iter().any(|t| t == &drawing.title));
        }
        // No room at all: no scale, nothing painted.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/clamp_drawing.rs`, replace:

````rust
        std::thread::spawn(move || {
            let ctx = egui::Context::default();
            let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
                clamp_ui(ui, &inputs, &results)
            });
            let shapes = flat_shapes(&output);
            let painted = Painted {
````

with:

````rust
        std::thread::spawn(move || {
            let ctx = egui::Context::default();
            let output = sized_frame(&ctx, egui::vec2(1000.0, 1400.0), Vec::new(), |ui| {
                clamp_ui(ui, &inputs, &results, &mut Readouts::default())
            });
            let shapes = flat_shapes(&output);
            let painted = Painted {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
        assert_eq!(line("model.pullout_Nm").marker, "E3 E7 E8");
        assert!(
            line("model.pullout_Nm")
                .tooltip
                .contains("with E3 alone: workbook 2.647, corrected 2.688")
        );
        assert_eq!(line("metal.torque_hot_low_Nm").level, Some(Level::Bad));
````

with:

````rust
        assert_eq!(line("model.pullout_Nm").marker, "E3 E7 E8");
        assert!(
            line("model.pullout_Nm")
                .tooltip()
                .contains("with E3 alone: workbook 2.647, corrected 2.688")
        );
        assert_eq!(line("metal.torque_hot_low_Nm").level, Some(Level::Bad));
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
        assert!(lines.iter().all(|l| !l.greyed));
        for path in STORED_3D_ROWS {
            assert!(line(path).stored_3d);
            assert!(line(path).tooltip.contains(STORED_3D_LABEL));
        }
        assert_eq!(lines.iter().filter(|l| l.stored_3d).count(), 3);
    }
````

with:

````rust
        assert!(lines.iter().all(|l| !l.greyed));
        for path in STORED_3D_ROWS {
            assert!(line(path).stored_3d);
            assert!(line(path).tooltip().contains(STORED_3D_LABEL));
        }
        assert_eq!(lines.iter().filter(|l| l.stored_3d).count(), 3);
    }
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
            if derived {
                assert_eq!(line.level, None, "{}", line.path);
                assert!(
                    line.tooltip.contains(END_EFFECT_OUT_OF_RANGE),
                    "{}",
                    line.path
                );
````

with:

````rust
            if derived {
                assert_eq!(line.level, None, "{}", line.path);
                assert!(
                    line.tooltip().contains(END_EFFECT_OUT_OF_RANGE),
                    "{}",
                    line.path
                );
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust
        let results = compute_all(inputs);
        let mut layout = None;
        let output = sized_frame(ctx, size, events, |ui| {
            layout = Some(geometry_ui(ui, inputs, &results));
        });
        (output, layout.expect("drawn"))
    }
````

with:

````rust
        let results = compute_all(inputs);
        let mut layout = None;
        let output = sized_frame(ctx, size, events, |ui| {
            layout = Some(geometry_ui(ui, inputs, &results, &mut Readouts::default()));
        });
        (output, layout.expect("drawn"))
    }
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust
        let mut output = None;
        for dt in [0.1, 0.2] {
            output = Some(sized_frame_at(&ctx, size, Some(dt), Vec::new(), |ui| {
                geometry_ui(ui, &inputs, &results);
            }));
        }
        assert!(
````

with:

````rust
        let mut output = None;
        for dt in [0.1, 0.2] {
            output = Some(sized_frame_at(&ctx, size, Some(dt), Vec::new(), |ui| {
                geometry_ui(ui, &inputs, &results, &mut Readouts::default());
            }));
        }
        assert!(
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub mod inputs;
mod panel;
pub mod plots;
pub mod results_table;
pub mod session;
pub mod sizing;
````

with:

````rust
pub mod inputs;
mod panel;
pub mod plots;
pub mod readouts;
pub mod results_table;
pub mod session;
pub mod sizing;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        texts.extend(
            crate::gui::dashboard::dashboard_lines(&results)
                .into_iter()
                .map(|line| line.tooltip),
        );
        for entry in crate::gui::results_table::table_entries() {
            let value = results.get(&entry.path).unwrap_or(Value::None);
````

with:

````rust
        texts.extend(
            crate::gui::dashboard::dashboard_lines(&results)
                .into_iter()
                .map(|line| line.tooltip()),
        );
        for entry in crate::gui::results_table::table_entries() {
            let value = results.get(&entry.path).unwrap_or(Value::None);
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        assert_eq!(other.panel.results(), harness.panel.results());
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
````

with:

````rust
        assert_eq!(other.panel.results(), harness.panel.results());
    }

    /// Shows tooltips at once (no delay, the pointer need not rest), as the geometry view's
    /// hover test does.
    fn tooltips_at_once(harness: &Harness) {
        harness.ctx.style_mut(|s| {
            s.interaction.tooltip_delay = 0.0;
            s.interaction.show_tooltips_only_when_still = false;
        });
    }

    /// The rects of the term marks painted in a frame, in `color` (any colour for `None`).
    fn mark_rects(output: &egui::FullOutput, color: Option<egui::Color32>) -> Vec<egui::Rect> {
        use crate::gui::readouts::MARK_WIDTH;
        crate::gui::test_support::flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r)
                    if r.stroke.width == MARK_WIDTH
                        && color.is_none_or(|c| r.stroke.color == c) =>
                {
                    Some(r.rect)
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn hovering_a_dashboard_value_shows_its_equation() {
        use crate::gui::readouts::OPEN_HINT;
        let mut harness = Harness::new();
        tooltips_at_once(&harness);
        let output = harness.frame(Vec::new());
        let pullout = displayed_pullout(&DesignInputs::default());
        let at = text_rect(&output, &pullout).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.hovered.as_deref(), Some("model.pullout_Nm"));
        let texts = drawn_texts(&output);
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("Pull-out torque at operating temperature\n")),
            "the hover text: {texts:?}"
        );
        // T_pull = T_2D f_end f_cal, typeset run by run, and the hint.
        for run in ["T", "pull", " = ", "2D", "end", "cal", OPEN_HINT] {
            assert!(texts.iter().any(|t| t == run), "no {run:?} in {texts:?}");
        }
    }

    #[test]
    fn hovering_a_value_marks_its_terms_where_their_values_are_shown() {
        // D_cup = 2 (r_corner + t_wall): hovering the cup OD on the dashboard frames the Key
        // design row of the wall in its term's colour, from the next frame, and its tooltip
        // draws t_wall in that colour; moving the pointer away clears every mark.
        let mut harness = Harness::new();
        tooltips_at_once(&harness);
        let output = harness.frame(Vec::new());
        let cup_od = harness.panel.results().model.cup_od_mm;
        let shown = with_unit(format_value(&Value::Num(cup_od)), "mm");
        let at = text_rect(&output, &shown).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        let eq = registry().equation_for("model.cup_od_mm").unwrap();
        let wall = TermColors::of(eq).get("metal.cup_wall_corner_mm").unwrap();
        let label = InputCatalogue::get()
            .entry("metal.cup_wall_corner_mm")
            .unwrap()
            .meta
            .label;
        let label_rect = text_rect(&output, label).unwrap();
        let marks = mark_rects(&output, Some(wall));
        assert!(
            marks.iter().any(|r| r.contains_rect(label_rect)),
            "{label_rect:?} not inside a mark: {marks:?}"
        );
        // The tooltip and the marks colour the term alike: t_wall's subscript run.
        assert_eq!(
            crate::gui::test_support::text_color(&output, "wall"),
            Some(wall),
            "the tooltip's t_wall: {:?}",
            drawn_texts(&output)
        );
        harness.frame(vec![egui::Event::PointerMoved(egui::pos2(
            5.0,
            SCREEN.y - 5.0,
        ))]);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.hovered, None);
        assert!(mark_rects(&output, None).is_empty());
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/plots.rs`, replace:

````rust
                &ctx,
                egui::vec2(900.0, 600.0),
                Vec::new(),
                |ui| plot_ui(ui, kind, inputs, &results),
            ));
        }
        output.unwrap()
````

with:

````rust
                &ctx,
                egui::vec2(900.0, 600.0),
                Vec::new(),
                |ui| plot_ui(ui, kind, inputs, &results, &mut Readouts::default()),
            ));
        }
        output.unwrap()
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/plots.rs`, replace:

````rust

    fn count(values: &[usize], want: usize) -> usize {
        values.iter().filter(|v| **v == want).count()
    }

    #[test]
````

with:

````rust

    fn count(values: &[usize], want: usize) -> usize {
        values.iter().filter(|v| **v == want).count()
    }

    #[test]
    fn each_plot_reads_out_the_design_values_it_draws() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        for kind in PlotKind::ALL {
            let texts = drawn_texts(&draw(kind, &inputs));
            for path in kind.readouts() {
                let info = result_info(path).unwrap_or_else(|| panic!("{path} is a result"));
                let value = results.get(path).unwrap();
                let want = format!(
                    "{}: {}",
                    info.meta.label,
                    with_unit(format_value(&value), info.meta.unit)
                );
                assert!(texts.contains(&want), "{kind:?}: missing {want:?}");
            }
            // The first readout of each plot has an equation to show.
            let first = kind.readouts()[0];
            assert!(
                crate::gui::readouts::registry()
                    .equation_for(first)
                    .is_some(),
                "{first}"
            );
        }
    }

    #[test]
````

Create `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/readouts.rs`:

````rust
//! The readout hook (spec Addendum A2 "Hover": "Any displayed value (dashboard, results
//! table, geometry callouts) shows a tooltip with its equation. Each term is coloured, and the
//! same colour marks that term wherever its value appears on screen").
//!
//! Every view hands each value it displays to [`Readouts::show`], with its result path and
//! its hover text: the dashboard rows, the results-table rows, the geometry callouts, the
//! clamp summary and screw table, the plot readouts. While the value is hovered the tooltip
//! shows that text, then the value's equation typeset with its terms in colour
//! ([`TermColors::of`]); a click asks the panel to open it in the Equation panel. Both the
//! text and the typesetting run only inside `on_hover_ui`, so a frame pays for the one value
//! hovered. A value whose path is a term of the equation in view (the one hovered in the last
//! frame, else the one open in the Equation panel) gets a frame in that term's colour, and so
//! do the input rows of the equation's leaf terms ([`Readouts::mark`]).
//!
//! The equation registry is built once per process ([`registry`], decision M43-5): the panel
//! builds it at start-up (about 3 ms in a release build), and every frame only looks paths up.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::test_support::{drawn_texts, flat_shapes, sized_frame, sized_frame_at};

    /// A label of `text` showing the result at `path`, drawn with `readouts`.
    fn draw(ui: &mut egui::Ui, readouts: &mut Readouts, path: &str, text: &str) -> Rect {
        let rect = ui.label(text).rect;
        readouts.show_over(ui, rect, path, || format!("hover text of {path}"));
        rect
    }

    #[test]
    fn the_registry_is_built_once() {
        assert!(std::ptr::eq(registry(), registry()));
        assert!(registry().equation_for("model.pullout_Nm").is_some());
    }

    #[test]
    fn a_hovered_value_shows_its_text_and_equation_and_a_click_asks_for_it() {
        let ctx = egui::Context::default();
        ctx.style_mut(|s| {
            s.interaction.tooltip_delay = 0.0;
            s.interaction.show_tooltips_only_when_still = false;
        });
        let size = egui::vec2(800.0, 600.0);
        let mut rect = Rect::NOTHING;
        sized_frame(&ctx, size, Vec::new(), |ui| {
            rect = draw(
                ui,
                &mut Readouts::default(),
                "model.pullout_Nm",
                "2.688 N·m",
            );
        });
        let at = rect.center();
        let mut events = ReadoutEvents::default();
        let mut output = None;
        for (time, input) in [
            (0.1, vec![egui::Event::PointerMoved(at)]),
            (0.2, Vec::new()),
        ] {
            output = Some(sized_frame_at(&ctx, size, Some(time), input, |ui| {
                let mut readouts = Readouts::default();
                draw(ui, &mut readouts, "model.pullout_Nm", "2.688 N·m");
                events = readouts.finish();
            }));
        }
        assert_eq!(events.hovered.as_deref(), Some("model.pullout_Nm"));
        assert_eq!(events.clicked, None);
        let texts = drawn_texts(&output.unwrap());
        assert!(texts.contains(&"hover text of model.pullout_Nm".to_owned()));
        // The equation T_pull = T_2D f_end f_cal, typeset run by run.
        for run in ["T", "pull", " = ", "2D", "end", "cal"] {
            assert!(texts.iter().any(|t| t == run), "no {run:?} in {texts:?}");
        }
        assert!(texts.contains(&OPEN_HINT.to_owned()));
        // A click.
        for pressed in [true, false] {
            sized_frame(
                &ctx,
                size,
                vec![crate::gui::test_support::primary_button(at, pressed)],
                |ui| {
                    let mut readouts = Readouts::default();
                    draw(ui, &mut readouts, "model.pullout_Nm", "2.688 N·m");
                    events = readouts.finish();
                },
            );
        }
        assert_eq!(events.clicked.as_deref(), Some("model.pullout_Nm"));
    }

    #[test]
    fn a_value_whose_path_is_a_term_in_view_is_framed_in_its_colour() {
        let eq = registry().equation_for("model.pullout_Nm").unwrap();
        let marks = TermColors::of(eq);
        let f_end = marks.get("model.f_end").unwrap();
        let ctx = egui::Context::default();
        let output = sized_frame(&ctx, egui::vec2(600.0, 400.0), Vec::new(), |ui| {
            let mut readouts = Readouts::new(marks.clone());
            draw(ui, &mut readouts, "model.f_end", "0.9046");
            draw(ui, &mut readouts, "mass.total_g", "48.15 g");
        });
        let frames: Vec<egui::Color32> = flat_shapes(&output)
            .into_iter()
            .filter_map(|s| match s {
                egui::Shape::Rect(r) if r.stroke.width == MARK_WIDTH => Some(r.stroke.color),
                _ => None,
            })
            .collect();
        assert_eq!(frames, [f_end], "only the term's value is framed");
    }

    #[test]
    fn a_warning_link_asks_for_its_note() {
        let mut readouts = Readouts::default();
        readouts.open_note("a5.low_saturation");
        assert_eq!(readouts.events().note, Some("a5.low_saturation"));
        assert_eq!(readouts.finish().note, Some("a5.low_saturation"));
    }
}
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | tail -1
```

Expected: FAIL to compile. The last line is `` error: could not compile `magcoupling-rs` (lib test) due to 47 previous errors; 1 warning emitted ``; the errors above it come from the tests reaching for what Step 3 adds, among them `` error[E0432]: unresolved import `crate::gui::readouts::MARK_WIDTH` ``, `` error[E0432]: unresolved import `crate::gui::readouts::OPEN_HINT` ``, `` error[E0425]: cannot find function `registry` in this scope ``, `` error[E0433]: failed to resolve: use of undeclared type `TermColors` ``, `` error[E0425]: cannot find function `result_info` in this scope ``.

- [ ] **Step 3: Write the implementation**

The hook between its docs and tests; each view's readouts; the panel's marks.

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/clamp_drawing.rs`, replace:

````rust
use crate::gui::format::{format_value, with_unit};
use crate::gui::geometry::{Mm, finite};
use crate::gui::geometry_view::{DIMENSION, Transform, arrowhead, plated_text, side_by_side};
use crate::gui::results_table::table_entries;
use crate::{DesignInputs, DesignResults};

````

with:

````rust
use crate::gui::format::{format_value, with_unit};
use crate::gui::geometry::{Mm, finite};
use crate::gui::geometry_view::{DIMENSION, Transform, arrowhead, plated_text, side_by_side};
use crate::gui::readouts::Readouts;
use crate::gui::results_table::table_entries;
use crate::{DesignInputs, DesignResults};

````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/clamp_drawing.rs`, replace:

````rust
    })
}

/// The clamp tab: the drawing (or why there is none), then the clamp table.
pub fn clamp_ui(ui: &mut egui::Ui, inputs: &DesignInputs, results: &DesignResults) {
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_clamp_scroll")
        .auto_shrink([false, false])
````

with:

````rust
    })
}

/// The clamp tab: the drawing (or why there is none), then the clamp table, each value a
/// readout (`readouts`).
pub fn clamp_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_clamp_scroll")
        .auto_shrink([false, false])
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/clamp_drawing.rs`, replace:

````rust
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
````

with:

````rust
                let Some(info) = crate::gui::dashboard::result_info(path) else {
                    continue;
                };
                let rect = ui
                    .horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new(info.meta.label).weak());
                        ui.label(with_unit(format_value(&value), info.meta.unit));
                    })
                    .response
                    .rect;
                readouts.show_over(ui, rect, path, || hover_text(path).unwrap_or_default());
            }
            ui.add_space(4.0);
            for step in MACHINING_STEPS {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/clamp_drawing.rs`, replace:

````rust
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
````

with:

````rust
            egui::CollapsingHeader::new(SCREW_SIZES)
                .id_salt("magcoupling_screw_sizes")
                .default_open(false)
                .show(ui, |ui| screw_table_ui(ui, results, readouts));
        });
}

/// The 'Clamp screw sizes' table: a row per field, a column per size, the recommended size's
/// column in green; each value a readout.
fn screw_table_ui(ui: &mut egui::Ui, results: &DesignResults, readouts: &mut Readouts) {
    let sizes = results.clamps.table.len();
    let pick = usize::try_from(results.clamps.index)
        .ok()
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/clamp_drawing.rs`, replace:

````rust
                            } else {
                                rich
                            };
                            ui.label(rich).on_hover_ui(|ui| {
                                ui.label(hover_text(&path).unwrap_or_else(|| path.clone()));
                            });
                        }
                        ui.end_row();
````

with:

````rust
                            } else {
                                rich
                            };
                            let rect = ui.label(rich).rect;
                            readouts.show_over(ui, rect, &path, || {
                                hover_text(&path).unwrap_or_else(|| path.clone())
                            });
                        }
                        ui.end_row();
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
//!   "3D values from the workbook" instead of a "3D updating" badge.
//!
//! [`result_tooltip`] is the hover text of every displayed result, keyed by result path: the
//! one hook the dashboard and the results table (and later the geometry callouts and the
//! equation explorer) go through.

use std::collections::HashMap;
use std::sync::OnceLock;
````

with:

````rust
//!   "3D values from the workbook" instead of a "3D updating" badge.
//!
//! [`result_tooltip`] is the hover text of every displayed result, keyed by result path: the
//! text every readout hands to [`crate::gui::readouts::Readouts::show`], which adds the
//! value's equation (plan M4-3). The dashboard builds a row's text only while it is hovered
//! ([`DashboardLine::tooltip`]).

use std::collections::HashMap;
use std::sync::OnceLock;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
use crate::gui::format::{format_value, with_unit};
use crate::{DesignInputs, DesignResults, compute_all};

/// A badge colour, from a check verdict.
````

with:

````rust
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
use crate::gui::format::{format_value, with_unit};
use crate::gui::readouts::Readouts;
use crate::{DesignInputs, DesignResults, compute_all};

/// A badge colour, from a check verdict.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
}

/// The hover text of a displayed result: label, help, path, workbook cell, then the notes.
/// The hook every readout goes through (plan M4-3 adds the equation here).
pub fn result_tooltip(path: &str, info: &ResultInfo, notes: ResultNotes<'_>) -> String {
    let meta = info.meta;
    let mut lines = vec![meta.label.to_owned()];
````

with:

````rust
}

/// The hover text of a displayed result: label, help, path, workbook cell, then the notes.
/// The text every readout shows (`Readouts::show` adds the equation under it).
pub fn result_tooltip(path: &str, info: &ResultInfo, notes: ResultNotes<'_>) -> String {
    let meta = info.meta;
    let mut lines = vec![meta.label.to_owned()];
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
    pub stored_3d: bool,
    /// The corrections' marker text (`E3 E7 E8`), empty for none.
    pub marker: String,
    pub tooltip: String,
}

/// The f_end of the design when it is out of the end-effect model's range, else `None`.
````

with:

````rust
    pub stored_3d: bool,
    /// The corrections' marker text (`E3 E7 E8`), empty for none.
    pub marker: String,
    /// The corrections the registry ties to its cell (the hover text's notes).
    pub marks: &'static [Mark],
}

impl DashboardLine {
    /// The row's hover text (built only while the row is hovered).
    pub fn tooltip(&self) -> String {
        let info = result_info(self.path).expect("a dashboard row is a result");
        result_tooltip(
            self.path,
            info,
            ResultNotes {
                marks: self.marks,
                greyed: self.greyed,
                stored_3d: self.stored_3d,
            },
        )
    }
}

/// The f_end of the design when it is out of the end-effect model's range, else `None`.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
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
````

with:

````rust
                greyed,
                stored_3d,
                marker: marker_text(marks),
                marks,
            }
        })
        .collect()
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust

/// Draws the dashboard: the end-effect banner when f_end <= 0, then a row per line (badge
/// and label, then the value and the marker under them, so a narrow side still reads), the
/// stored-3D label after the last temperature row.
pub fn dashboard_ui(ui: &mut egui::Ui, results: &DesignResults) {
    let lines = dashboard_lines(results);
    if let Some(banner) = end_effect_banner(results) {
        ui.colored_label(ui.visuals().error_fg_color, banner);
````

with:

````rust

/// Draws the dashboard: the end-effect banner when f_end <= 0, then a row per line (badge
/// and label, then the value and the marker under them, so a narrow side still reads), the
/// stored-3D label after the last temperature row. Each row is a readout (`readouts`): its
/// hover text and equation while hovered, a click opens it in the Equation panel.
pub fn dashboard_ui(ui: &mut egui::Ui, results: &DesignResults, readouts: &mut Readouts) {
    let lines = dashboard_lines(results);
    if let Some(banner) = end_effect_banner(results) {
        ui.colored_label(ui.visuals().error_fg_color, banner);
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
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
````

with:

````rust
    for (index, line) in lines.iter().enumerate() {
        let weak = ui.visuals().weak_text_color();
        let tint = |rich: egui::RichText| if line.greyed { rich.color(weak) } else { rich };
        let title = ui.horizontal(|ui| {
            badge(ui, line.level);
            ui.add(egui::Label::new(tint(egui::RichText::new(line.label))).wrap());
        });
        let value = ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.add(egui::Label::new(tint(egui::RichText::new(&line.value).strong())).wrap());
            if !line.marker.is_empty() {
                ui.small(&line.marker);
            }
        });
        let rect = title.response.rect.union(value.response.rect);
        readouts.show_over(ui, rect, line.path, || line.tooltip());
        if Some(index) == last_3d {
            ui.horizontal(|ui| {
                ui.add_space(16.0);
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust
//! (decision M42-3), then lists the dimension callouts and the notes under them.
//!
//! Each dimension is a line with arrowheads and a tag; the list repeats the tag with the
//! callout's text. Hovering the line or its text shows the hover text of the result it shows
//! ([`crate::gui::dashboard::hover_text`], the hook M4-3's equation tooltip joins), built only
//! while hovered. A violated callout is red, one that is not a number amber (decision M42-4);
//! the space claim is dashed, an exceeded axis red. The view is rebuilt from the design shown on
//! every frame, so it follows a slider while it is dragged.
//!
````

with:

````rust
//! (decision M42-3), then lists the dimension callouts and the notes under them.
//!
//! Each dimension is a line with arrowheads and a tag; the list repeats the tag with the
//! callout's text. The line and the text are readouts ([`crate::gui::readouts::Readouts`]):
//! hovering either shows the hover text of the result it shows
//! ([`crate::gui::dashboard::hover_text`]) and its equation, built only while hovered, and a
//! click opens it in the Equation panel. A violated callout is red, one that is not a number amber (decision M42-4);
//! the space claim is dashed, an exceeded axis red. The view is rebuilt from the design shown on
//! every frame, so it follows a slider while it is dragged.
//!
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust

use crate::gui::dashboard::{Level, hover_text};
use crate::gui::geometry::{Callout, Geometry, Mm, Outline, Part, View, finite, geometry};
use crate::{DesignInputs, DesignResults};

/// The colour of a dimension that is fine: drawing.py's dimension blue, lightened for a dark
````

with:

````rust

use crate::gui::dashboard::{Level, hover_text};
use crate::gui::geometry::{Callout, Geometry, Mm, Outline, Part, View, finite, geometry};
use crate::gui::readouts::Readouts;
use crate::{DesignInputs, DesignResults};

/// The colour of a dimension that is fine: drawing.py's dimension blue, lightened for a dark
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust
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
````

with:

````rust
    }
}

/// Draws the geometry of the design shown (`inputs` and the `results` computed from them),
/// each callout a readout (`readouts`), and returns what it drew.
pub fn geometry_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) -> GeometryLayout {
    let g = geometry(inputs, results);
    let row = ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust
    if let Some((scale, transforms)) = side_by_side(rect.shrink(MARGIN), &extents, VIEW_GAP_MM) {
        layout.scale = scale;
        for (view, transform) in views.into_iter().zip(transforms) {
            paint_view(ui, &painter, view, transform, &mut layout.dimensions);
            if std::ptr::eq(view, &g.end) {
                layout.end = Some(transform);
            } else {
````

with:

````rust
    if let Some((scale, transforms)) = side_by_side(rect.shrink(MARGIN), &extents, VIEW_GAP_MM) {
        layout.scale = scale;
        for (view, transform) in views.into_iter().zip(transforms) {
            paint_view(
                ui,
                &painter,
                view,
                transform,
                &mut layout.dimensions,
                readouts,
            );
            if std::ptr::eq(view, &g.end) {
                layout.end = Some(transform);
            } else {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust
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
````

with:

````rust
            }
        }
    }
    list_ui(ui, &g, readouts);
    layout
}

/// Paints one view's pieces, dashed lines and dimensions, and registers each dimension as a
/// readout.
fn paint_view(
    ui: &egui::Ui,
    painter: &egui::Painter,
    view: &View,
    t: Transform,
    dimensions: &mut Vec<(usize, Rect)>,
    readouts: &mut Readouts,
) {
    let visuals = ui.visuals();
    let centre = t.to_px([0.0, 0.0]);
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust
        let response = ui.interact(
            rect,
            ui.id().with(("geometry_dimension", callout.tag)),
            Sense::hover(),
        );
        hover(response, callout);
        dimensions.push((callout.tag, rect));
    }
}
````

with:

````rust
        let response = ui.interact(
            rect,
            ui.id().with(("geometry_dimension", callout.tag)),
            Sense::click(),
        );
        readouts.show(ui, response, callout.path, || callout_text(callout));
        dimensions.push((callout.tag, rect));
    }
}
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust
    painter.galley(rect.min, galley, color);
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
````

with:

````rust
    painter.galley(rect.min, galley, color);
}

/// The hover text of the callout's result.
fn callout_text(callout: &Callout) -> String {
    hover_text(callout.path).unwrap_or_else(|| callout.path.to_owned())
}

/// The callouts (tag and text, in their colours, each a readout) and the notes.
fn list_ui(ui: &mut egui::Ui, g: &Geometry, readouts: &mut Readouts) {
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_geometry_list")
        .auto_shrink([false, true])
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/geometry_view.rs`, replace:

````rust
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
````

with:

````rust
                    Some(c) => egui::RichText::new(text).color(c),
                    None => egui::RichText::new(text),
                };
                let rect = ui
                    .horizontal(|ui| {
                        ui.label(rich(callout.tag.to_string()).strong());
                        ui.add(egui::Label::new(rich(callout.text.clone())).wrap());
                    })
                    .response
                    .rect;
                readouts.show_over(ui, rect, callout.path, || callout_text(callout));
            }
            for note in &g.notes {
                let text = egui::RichText::new(&note.text);
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame), and the free variable's
//! row shows that value, locked.

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::clamp_drawing::clamp_ui;
````

with:

````rust
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame), and the free variable's
//! row shows that value, locked.

use crate::engine::explain::Design as Terms;
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::clamp_drawing::clamp_ui;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
````

with:

````rust
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{ReadoutEvents, Readouts, registry};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, TARGET_RANGE_INPUT,
    variable_label,
};
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
````

with:

````rust
    SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, TARGET_RANGE_INPUT,
    variable_label,
};
use crate::gui::typeset::TermColors;
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    results_table: ResultsTable,
    /// Platform work for the host, oldest first.
    requests: Vec<PanelRequest>,
}

impl Default for MagcouplingPanel {
````

with:

````rust
    results_table: ResultsTable,
    /// Platform work for the host, oldest first.
    requests: Vec<PanelRequest>,
    /// The path of the readout hovered in the last frame: this frame marks its equation's
    /// terms wherever their values are shown (spec Addendum A2 "Hover").
    hovered: Option<String>,
}

impl Default for MagcouplingPanel {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
impl MagcouplingPanel {
    /// A panel at the default design.
    pub fn new() -> Self {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        Self {
````

with:

````rust
impl MagcouplingPanel {
    /// A panel at the default design.
    pub fn new() -> Self {
        // The equation registry, built once per process, at start-up rather than on the first
        // hover.
        registry();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        Self {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            centre: CentreView::Geometry,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
        }
    }

````

with:

````rust
            centre: CentreView::Geometry,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
            hovered: None,
        }
    }

````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.push_id("magcoupling_panel", |ui| {
            self.shortcuts(ui);
            egui::TopBottomPanel::top("magcoupling_header").show_inside(ui, |ui| {
````

with:

````rust

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let mut readouts = Readouts::new(self.marks());
        ui.push_id("magcoupling_panel", |ui| {
            self.shortcuts(ui);
            egui::TopBottomPanel::top("magcoupling_header").show_inside(ui, |ui| {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            egui::SidePanel::left("magcoupling_inputs")
                .resizable(true)
                .default_width(INPUTS_WIDTH)
                .show_inside(ui, |ui| self.inputs_ui(ui));
            self.run_sizing(ui);
            // After the inputs: the readouts show this frame's edits. One copy of the design
            // shown per frame, for the results and the centre region's views.
````

with:

````rust
            egui::SidePanel::left("magcoupling_inputs")
                .resizable(true)
                .default_width(INPUTS_WIDTH)
                .show_inside(ui, |ui| self.inputs_ui(ui, &readouts));
            self.run_sizing(ui);
            // After the inputs: the readouts show this frame's edits. One copy of the design
            // shown per frame, for the results and the centre region's views.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| dashboard_ui(ui, &self.results));
                });
            egui::CentralPanel::default().show_inside(ui, |ui| self.centre_ui(ui, &shown));
        });
        // One undo step per settled edit.
        let settled = !self.editing(ui);
        self.history.observe(&self.design(), settled);
    }

    /// Whether an edit of the design is in progress: a pointer button down (a drag), a key held
````

with:

````rust
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| dashboard_ui(ui, &self.results, &mut readouts));
                });
            egui::CentralPanel::default()
                .show_inside(ui, |ui| self.centre_ui(ui, &shown, &mut readouts));
        });
        self.end_frame(ui, readouts.finish());
        // One undo step per settled edit.
        let settled = !self.editing(ui);
        self.history.observe(&self.design(), settled);
    }

    /// The term colours this frame marks: the terms of the equation of the readout hovered in
    /// the last frame, a family's by the harmonics the design sums; none without one.
    fn marks(&self) -> TermColors {
        let Some(eq) = self
            .hovered
            .as_deref()
            .and_then(|path| registry().equation_for(path))
        else {
            return TermColors::none();
        };
        let terms = Terms {
            inputs: &self.inputs,
            results: &self.results,
        };
        TermColors::of(eq).with_members(registry(), &terms)
    }

    /// Takes in what the user did with the readouts this frame: the value hovered marks its
    /// terms from the next frame, which is asked for at once.
    fn end_frame(&mut self, ui: &egui::Ui, events: ReadoutEvents) {
        if events.hovered != self.hovered {
            self.hovered = events.hovered;
            ui.ctx().request_repaint();
        }
    }

    /// Whether an edit of the design is in progress: a pointer button down (a drag), a key held
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust

    /// The sizing controls at the top of the Key design group: the mode switch, then in
    /// Torque → Magnets the free variable, the target torque and the outcome.
    fn sizing_ui(&mut self, ui: &mut egui::Ui) {
        let mut mode = self.sizing.mode;
        ui.horizontal(|ui| {
            for choice in SizingMode::ALL {
````

with:

````rust

    /// The sizing controls at the top of the Key design group: the mode switch, then in
    /// Torque → Magnets the free variable, the target torque and the outcome.
    fn sizing_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let mut mode = self.sizing.mode;
        ui.horizontal(|ui| {
            for choice in SizingMode::ALL {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            let entry = InputCatalogue::get()
                .entry(path)
                .expect("every free variable is an input");
            let widget = self.input_row_ui(ui, entry);
            self.design_widgets.push(widget);
        }
        ui.separator();
````

with:

````rust
            let entry = InputCatalogue::get()
                .entry(path)
                .expect("every free variable is an input");
            let widget = self.input_row_ui(ui, entry, readouts);
            self.design_widgets.push(widget);
        }
        ui.separator();
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust

    /// The centre region: the view tabs (wrapping when the region is narrow), the end-effect
    /// banner when f_end <= 0 (over every view, decision M42-1), then the view of `shown`, the
    /// design shown.
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs) {
        ui.horizontal_wrapped(|ui| {
            for view in CentreView::ALL {
                ui.selectable_value(&mut self.centre, view, view.label());
````

with:

````rust

    /// The centre region: the view tabs (wrapping when the region is narrow), the end-effect
    /// banner when f_end <= 0 (over every view, decision M42-1), then the view of `shown`, the
    /// design shown, its values readouts (`readouts`).
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        ui.horizontal_wrapped(|ui| {
            for view in CentreView::ALL {
                ui.selectable_value(&mut self.centre, view, view.label());
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        }
        match self.centre {
            CentreView::Geometry => {
                geometry_ui(ui, shown, &self.results);
            }
            CentreView::Plot(kind) => plot_ui(ui, kind, shown, &self.results),
            CentreView::Clamp => clamp_ui(ui, shown, &self.results),
            CentreView::Results => {
                let action = self.results_table.ui(ui, &self.results);
                match action {
                    Some(TableAction::ExportCsv) => self.requests.push(PanelRequest::SaveFile {
                        file_name: CSV_FILE_NAME.to_owned(),
````

with:

````rust
        }
        match self.centre {
            CentreView::Geometry => {
                geometry_ui(ui, shown, &self.results, readouts);
            }
            CentreView::Plot(kind) => plot_ui(ui, kind, shown, &self.results, readouts),
            CentreView::Clamp => clamp_ui(ui, shown, &self.results, readouts),
            CentreView::Results => {
                let action = self.results_table.ui(ui, &self.results, readouts);
                match action {
                    Some(TableAction::ExportCsv) => self.requests.push(PanelRequest::SaveFile {
                        file_name: CSV_FILE_NAME.to_owned(),
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    }

    /// The left side: the Key design group, then every input by package group.
    fn inputs_ui(&mut self, ui: &mut egui::Ui) {
        let catalogue = InputCatalogue::get();
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
````

with:

````rust
    }

    /// The left side: the Key design group, then every input by package group.
    fn inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let catalogue = InputCatalogue::get();
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                    .id_salt("key_design")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.sizing_ui(ui);
                        self.key_widgets.clear();
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry);
                            self.key_widgets.push((entry.path.as_str(), widget));
                            self.design_widgets.push(widget);
                        }
````

with:

````rust
                    .id_salt("key_design")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.sizing_ui(ui, readouts);
                        self.key_widgets.clear();
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry, readouts);
                            self.key_widgets.push((entry.path.as_str(), widget));
                            self.design_widgets.push(widget);
                        }
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                                    ui.strong(section.label);
                                }
                                for entry in &section.entries {
                                    let widget = self.input_row_ui(ui, entry);
                                    self.design_widgets.push(widget);
                                }
                            }
````

with:

````rust
                                    ui.strong(section.label);
                                }
                                for entry in &section.entries {
                                    let widget = self.input_row_ui(ui, entry, readouts);
                                    self.design_widgets.push(widget);
                                }
                            }
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    }

    /// One input row; applies its edit. Returns the id of its main widget. In Torque →
    /// Magnets the free variable's row shows the value the panel shows, locked.
    fn input_row_ui(&mut self, ui: &mut egui::Ui, entry: &'static InputEntry) -> egui::Id {
        let locked = self.sizing.mode == SizingMode::TorqueToMagnets
            && entry.path == self.sizing.variable.path();
        // The locked row reads the design shown (one clone, for that row only).
````

with:

````rust
    }

    /// One input row; applies its edit. Returns the id of its main widget. In Torque →
    /// Magnets the free variable's row shows the value the panel shows, locked. The row is
    /// framed in its term's colour while the equation in view reads the input (`readouts`).
    fn input_row_ui(
        &mut self,
        ui: &mut egui::Ui,
        entry: &'static InputEntry,
        readouts: &Readouts,
    ) -> egui::Id {
        let locked = self.sizing.mode == SizingMode::TorqueToMagnets
            && entry.path == self.sizing.variable.path();
        // The locked row reads the design shown (one clone, for that row only).
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        }
        .unwrap_or(Value::None);
        let seed = self.seed(entry);
        let output = ui
            .add_enabled_ui(!locked, |ui| input_row(ui, entry, &current, seed))
            .inner;
        if locked {
            ui.weak(SIZED_NOTE);
            return output.widget.id;
````

with:

````rust
        }
        .unwrap_or(Value::None);
        let seed = self.seed(entry);
        let row = ui.add_enabled_ui(!locked, |ui| input_row(ui, entry, &current, seed));
        readouts.mark(ui, row.response.rect, &entry.path);
        let output = row.inner;
        if locked {
            ui.weak(SIZED_NOTE);
            return output.widget.id;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/plots.rs`, replace:

````rust
//! [`NOTHING_TO_PLOT`] (the torque-temperature plot [`EMPTY_TEMPERATURE_AXIS`] when its axis is
//! empty although every value is a number), and a sweep that left rows out counts them in a note
//! over the plot ([`rows_not_plotted`]).

use egui::Color32;
use egui_plot::{Corner, HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, Points, VLine};

use crate::engine::meta::NumOrText;
use crate::engine::model::{end_effect_in_range, harmonic_count, ring_pair_factor};
use crate::engine::sweeps::SweepRow;
use crate::gui::dashboard::{Level, end_effect_out_of_range};
use crate::{DesignInputs, DesignResults};

/// Samples of the torque-temperature curve.
````

with:

````rust
//! [`NOTHING_TO_PLOT`] (the torque-temperature plot [`EMPTY_TEMPERATURE_AXIS`] when its axis is
//! empty although every value is a number), and a sweep that left rows out counts them in a note
//! over the plot ([`rows_not_plotted`]).
//!
//! Over each plot a line reads out the design's values the plot shows ([`PlotKind::readouts`],
//! decision M43-11), each a readout: hover it for its equation, click it to open it (spec
//! Addendum A2 "Hover"). The curves' points have no result path, so the plot itself keeps
//! egui_plot's coordinate readout.

use egui::Color32;
use egui_plot::{Corner, HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, Points, VLine};

use crate::engine::meta::{NumOrText, ResultSet, Value};
use crate::engine::model::{end_effect_in_range, harmonic_count, ring_pair_factor};
use crate::engine::sweeps::SweepRow;
use crate::gui::dashboard::{Level, end_effect_out_of_range, hover_text, result_info};
use crate::gui::format::{format_value, with_unit};
use crate::gui::readouts::Readouts;
use crate::{DesignInputs, DesignResults};

/// Samples of the torque-temperature curve.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/plots.rs`, replace:

````rust
            PlotKind::PoleSweep => "Pole sweep",
            PlotKind::SlipHeating => "Slip heating",
            PlotKind::TorqueAngle => "Torque vs rotation",
        }
    }

````

with:

````rust
            PlotKind::PoleSweep => "Pole sweep",
            PlotKind::SlipHeating => "Slip heating",
            PlotKind::TorqueAngle => "Torque vs rotation",
        }
    }

    /// The results read out over the plot (decision M43-11): the design's values it draws.
    pub const fn readouts(self) -> &'static [&'static str] {
        match self {
            PlotKind::TorqueTemperature => &[
                "model.pullout_Nm",
                "metal.torque_hot_low_Nm",
                "metal.torque_cold_high_Nm",
                "temperature.summary.governing_limit_C",
            ],
            PlotKind::GapSweep => &[
                "model.corner_gap_mm",
                "model.pullout_Nm",
                "model.required_floor_Nm",
            ],
            PlotKind::PoleSweep => &["model.pullout_Nm", "model.required_floor_Nm"],
            PlotKind::SlipHeating => &[
                "temperature.thermal.time_constant_s",
                "temperature.thermal.time_to_limit_high",
                "temperature.summary.governing_limit_C",
            ],
            PlotKind::TorqueAngle => &["model.pullout_Nm", "model.pullout_angle_rad"],
        }
    }

````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/plots.rs`, replace:

````rust
        .height(ui.available_height().max(150.0))
}

/// Draws the plot `kind` of the design shown (`inputs`, and the `results` computed from them).
pub fn plot_ui(ui: &mut egui::Ui, kind: PlotKind, inputs: &DesignInputs, results: &DesignResults) {
    match kind {
        PlotKind::TorqueTemperature => torque_temperature_ui(ui, inputs, results),
        PlotKind::GapSweep => sweep_ui(
````

with:

````rust
        .height(ui.available_height().max(150.0))
}

/// Draws the plot `kind` of the design shown (`inputs`, and the `results` computed from them),
/// under its readouts (`readouts`).
pub fn plot_ui(
    ui: &mut egui::Ui,
    kind: PlotKind,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    readouts_ui(ui, kind, results, readouts);
    match kind {
        PlotKind::TorqueTemperature => torque_temperature_ui(ui, inputs, results),
        PlotKind::GapSweep => sweep_ui(
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/plots.rs`, replace:

````rust
        PlotKind::SlipHeating => slip_heating_ui(ui, results),
        PlotKind::TorqueAngle => torque_angle_ui(ui, inputs, results),
    }
}

/// Adds a dashed horizontal reference line `name` at `y`, only when `y` is finite (egui_plot's
````

with:

````rust
        PlotKind::SlipHeating => slip_heating_ui(ui, results),
        PlotKind::TorqueAngle => torque_angle_ui(ui, inputs, results),
    }
}

/// The line over a plot: each of [`PlotKind::readouts`] as "label: value", a readout.
fn readouts_ui(
    ui: &mut egui::Ui,
    kind: PlotKind,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    ui.horizontal_wrapped(|ui| {
        // Whole readouts move to the next row (a wrapped text would start mid-row, and its
        // hover area with it).
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        for &path in kind.readouts() {
            let Some(info) = result_info(path) else {
                continue;
            };
            let value = results.get(path).unwrap_or(Value::None);
            let text = format!(
                "{}: {}",
                info.meta.label,
                with_unit(format_value(&value), info.meta.unit)
            );
            let rect = ui.small(text).rect;
            readouts.show_over(ui, rect, path, || hover_text(path).unwrap_or_default());
        }
    });
}

/// Adds a dashed horizontal reference line `name` at `y`, only when `y` is finite (egui_plot's
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/readouts.rs`, replace:

````rust

#[cfg(test)]
mod tests {
````

with:

````rust

use std::sync::OnceLock;

use egui::{Rect, Stroke, StrokeKind};

use crate::engine::explain::Registry;
use crate::gui::typeset::{TermColors, equation_ui};

/// The hint under a tooltip's equation.
pub const OPEN_HINT: &str = "Click to open it in the Equation panel";

/// The text size of a tooltip's equation [points].
pub const TOOLTIP_SIZE: f32 = 15.0;

/// The width of a term's mark [points].
pub const MARK_WIDTH: f32 = 2.0;

/// The start of the log line written when the registry is built (the web smoke looks for it).
pub const REGISTRY_LOG_PREFIX: &str = "magcoupling explorer: ";

/// The equation registry, built on first use and kept for the life of the process: every
/// record parsed and checked once (`Registry::build` panics only on a broken record, which
/// `tests/explain.rs` rules out).
pub fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let registry = Registry::build();
        log::info!(
            "{REGISTRY_LOG_PREFIX}{} equations",
            registry.equations().len()
        );
        registry
    })
}

/// What the user did with the readouts in one frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadoutEvents {
    /// The path of the value under the pointer.
    pub hovered: Option<String>,
    /// The path of the value clicked.
    pub clicked: Option<String>,
    /// The teaching note a warning's link asked for, by id.
    pub note: Option<&'static str>,
}

/// One frame's readouts: the term colours they mark and what the user did.
#[derive(Clone, Debug, Default)]
pub struct Readouts {
    marks: TermColors,
    events: ReadoutEvents,
}

impl Readouts {
    /// The readouts of a frame that marks the terms in `marks`.
    pub fn new(marks: TermColors) -> Self {
        Self {
            marks,
            events: ReadoutEvents::default(),
        }
    }

    /// The colours this frame marks.
    pub fn marks(&self) -> &TermColors {
        &self.marks
    }

    /// Frames `rect` in the colour of the term at `path`, if the equation in view shows it.
    pub fn mark(&self, ui: &egui::Ui, rect: Rect, path: &str) {
        if let Some(color) = self.marks.get(path) {
            ui.painter().rect_stroke(
                rect.expand(1.0),
                3.0,
                Stroke::new(MARK_WIDTH, color),
                StrokeKind::Outside,
            );
        }
    }

    /// Shows the value of the result at `path` that `response` displays: its mark, and while
    /// hovered `text()` and its equation; a click asks for it in the Equation panel. `text` is
    /// called only while the value is hovered.
    pub fn show(
        &mut self,
        ui: &egui::Ui,
        response: egui::Response,
        path: &str,
        text: impl FnOnce() -> String,
    ) {
        self.mark(ui, response.rect, path);
        if response.hovered() {
            self.events.hovered = Some(path.to_owned());
        }
        if response.clicked() {
            self.events.clicked = Some(path.to_owned());
        }
        response.on_hover_ui(|ui| {
            ui.label(text());
            if let Some(eq) = registry().equation_for(path) {
                ui.separator();
                equation_ui(ui, registry(), eq, &TermColors::of(eq), TOOLTIP_SIZE);
            }
            ui.weak(OPEN_HINT);
        });
    }

    /// [`Readouts::show`] for a value drawn in `rect` by widgets that take no clicks (labels
    /// keep their look): a click sensor over the rect, keyed by `path` in `ui`.
    pub fn show_over(
        &mut self,
        ui: &egui::Ui,
        rect: Rect,
        path: &str,
        text: impl FnOnce() -> String,
    ) {
        let response = ui.interact(rect, ui.id().with(("readout", path)), egui::Sense::click());
        self.show(ui, response, path, text);
    }

    /// Asks for the teaching note `id` (a warning's link).
    pub fn open_note(&mut self, id: &'static str) {
        self.events.note = Some(id);
    }

    /// What the user did this frame.
    pub fn events(&self) -> &ReadoutEvents {
        &self.events
    }

    /// Ends the frame: what the user did.
    pub fn finish(self) -> ReadoutEvents {
        self.events
    }
}

#[cfg(test)]
mod tests {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
use crate::gui::corrections::{CorrectionIndex, marker_text};
use crate::gui::dashboard::{ResultInfo, hover_text, result_info};
use crate::gui::format::{format_value, non_finite_text, with_unit};
use crate::gui::session::{Design, design_json, json_value};
use crate::{DesignInputs, DesignResults, compute_all};

````

with:

````rust
use crate::gui::corrections::{CorrectionIndex, marker_text};
use crate::gui::dashboard::{ResultInfo, hover_text, result_info};
use crate::gui::format::{format_value, non_finite_text, with_unit};
use crate::gui::readouts::Readouts;
use crate::gui::session::{Design, design_json, json_value};
use crate::{DesignInputs, DesignResults, compute_all};

````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
    }

    /// Draws the table: the search box and the export buttons, then the rows on screen (the
    /// end-effect banner is the centre region's, over every view: decision M42-1). Returns an
    /// export asked for.
    pub fn ui(&mut self, ui: &mut egui::Ui, results: &DesignResults) -> Option<TableAction> {
        let entries = table_entries();
        let mut action = None;
        ui.horizontal_wrapped(|ui| {
````

with:

````rust
    }

    /// Draws the table: the search box and the export buttons, then the rows on screen (the
    /// end-effect banner is the centre region's, over every view: decision M42-1), each row a
    /// readout (`readouts`). Returns an export asked for.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        results: &DesignResults,
        readouts: &mut Readouts,
    ) -> Option<TableAction> {
        let entries = table_entries();
        let mut action = None;
        ui.horizontal_wrapped(|ui| {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
            .auto_shrink([false, false])
            .show_rows(ui, row_height, matches.len(), |ui, range| {
                for &index in &matches[range] {
                    row_ui(ui, &entries[index], results, row_height, widths);
                }
            });
        action
````

with:

````rust
            .auto_shrink([false, false])
            .show_rows(ui, row_height, matches.len(), |ui, range| {
                for &index in &matches[range] {
                    row_ui(ui, &entries[index], results, row_height, widths, readouts);
                }
            });
        action
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust

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
````

with:

````rust

/// One table row: label, value with unit, workbook cell, marker, in columns `widths` wide
/// ([`column_widths`]; the path is in the hover text, which is built only while the row is
/// hovered). The whole row is a readout: hover it for its equation, click it to open it.
fn row_ui(
    ui: &mut egui::Ui,
    entry: &TableEntry,
    results: &DesignResults,
    height: f32,
    widths: [f32; 4],
    readouts: &mut Readouts,
) {
    let value = results.get(&entry.path).unwrap_or(Value::None);
    let row = ui.horizontal(|ui| {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
        );
        cell(ui, marker, &entry.marker);
    });
    row.response.on_hover_ui(|ui| {
        ui.label(row_tooltip(entry, &value));
    });
}

````

with:

````rust
        );
        cell(ui, marker, &entry.marker);
    });
    readouts.show_over(ui, row.response.rect, &entry.path, || {
        row_tooltip(entry, &value)
    });
}

````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib gui::readouts 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 4 passed; 0 failed` for `gui::readouts`, then `test result: ok. 384 passed; 0 failed` for the whole library.

- [ ] **Step 5: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m43 add magcoupling-rs/src/gui/readouts.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/results_table.rs magcoupling-rs/src/gui/geometry_view.rs magcoupling-rs/src/gui/clamp_drawing.rs magcoupling-rs/src/gui/plots.rs magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m43 commit -F - <<'EOF'
feat(magcoupling-rs): every displayed value hovers its equation and marks its terms

gui::readouts is the one readout hook: the dashboard, the results table, the
geometry callouts, the clamp tab and a new readout line over each plot
(decision M43-11) hand every value to Readouts::show, which shows its hover
text and its typeset equation only while hovered, records hovers and clicks,
and frames a value in its term's colour when the equation in view reads it.
The panel marks the terms of the equation hovered in the last frame, input
rows included. The equation registry is built once per process (decision
M43-5); the dashboard's hover text is now built lazily (DashboardLine::tooltip).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m43 status --short
```

Expected: one new commit on `magcoupling/m4-3`; `status --short` prints nothing.

---

### Task 3: The Equation panel

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

The docked, toggleable panel (spec A2 "Equation panel"). `Explorer` keeps the trail of paths opened, the readout hovered in the last frame and the input row a leaf term highlights; `explorer_ui` draws the header row (the heading, the breadcrumb of plain symbols, Close), then the path shown: its label and cell, the equation large (22 points, horizontally scrollable, a term clicked in it followed), its value, the corrections it embodies (`Registry::corrections_upstream`, decision M43-14), the term list (swatch, symbol typeset by the new `typeset::layout_symbol`, value in the unit the formula reads, label to click, what the term is) without the harmonics past the set, its swatches fading while a value of another equation is hovered (the marks on screen are then that equation's; M43-3), and "used by"; a long trail shows its last 8 crumbs after "…"; a result without a record shows its label, value and cell (decision M43-9). The panel docks at the bottom of the centre region (decision M43-1), closed until a value is clicked or its header button pressed (M43-2); the marks follow the hovered equation, else the open one. A leaf term focuses its input: the panel opens its group (`CollapsingHeader::open(Some(true))` for that frame: the header's own id lives in a child ui), scrolls the row into view once and frames it (decision M43-12). The panel's `hovered`, `marks` and `end_frame` move into `Explorer`. Rows of links set `TextWrapMode::Extend` (a wrapped label starts mid-row). The tests pin the trail (open, drill, back, a 32-step cap), following a term (an input, a result, a Σ's template), the marks' precedence, the swatches fading, the term list's hidden harmonics, the labels; in the panel the header toggle, click-to-open, drill and crumb, used by, the leaf row, a result without a record, a tiny window, a 32-step trail's last crumbs on a laptop and a short screen (Review Focus 3 and 4), a harmonic set outside the choices and an undefined torque opened in the panel (Review Focus 8); the glyph test opens the panel on a temperature chain and goes through `assert_glyphs`.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/mod.rs` (`pub mod explorer;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/typeset.rs` (`layout_symbol` and its test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs` (`explorer`, the dock, the header button, the leaf row; tests)

**Interfaces:**
- Consumes: Tasks 1 and 2 (`TermColors`, `glyph_safe`, `laid_ui`, `layout_equation`, `term_at`; `registry`, `Readouts`, `ReadoutEvents`); `engine::explain::{Design, Equation, TermKind}`, `markup::{Expr, Symbol}`, `notes::covers`; `dashboard::result_info`, `inputs::InputCatalogue`.
- Produces (`magcoupling::gui::explorer`): `pub const EQUATION_PANEL, CLOSE, EMPTY_TEXT, TERMS, USED_BY, NO_EQUATION, CORRECTIONS: &str`; `pub const PANEL_SIZE: f32` (22), `TERM_SIZE: f32` (15), `PANEL_HEIGHT: f32` (320), `MAX_TRAIL: usize` (32), `SHOWN_CRUMBS: usize` (8), `ELIDED: &str` ("…"), `FOCUS_WIDTH: f32` (3), `SWATCH_DIM: f32` (0.35); `#[derive(Clone, Debug, PartialEq, Eq)] pub struct Focus { pub path: String, pub scroll: bool }`; `#[derive(Clone, Debug, Default, PartialEq, Eq)] pub struct Explorer { pub open: bool, .. }` with `fn current(&self) -> Option<&str>`, `fn trail(&self) -> &[String]`, `fn hovered(&self) -> Option<&str>`, `fn open_path(&mut self, &str)`, `fn drill(&mut self, &str)`, `fn back_to(&mut self, usize)`, `fn focus_input(&mut self, &str)`, `fn focus(&self) -> Option<&Focus>`, `fn take_scroll(&mut self, &str) -> bool`, `fn follow(&mut self, &str, &Design<'_>)`, `fn marks(&self, &DesignInputs, &DesignResults) -> TermColors`, `fn end_frame(&mut self, &egui::Context, ReadoutEvents)`; `pub fn term_label(&str) -> &str`; `pub fn plain_symbol(&str) -> String`; `pub fn explorer_ui(&mut egui::Ui, &mut Explorer, &DesignInputs, &DesignResults)`. (`typeset`): `pub fn layout_symbol(&Fonts, &Registry, markup: &str, size: f32, color: Color32) -> Laid`.

- [ ] **Step 1: Write the failing tests**

Create `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`:

````rust
//! The Equation panel (spec Addendum A2 "Equation panel (docked, toggleable). It shows the
//! open equation large, its terms with values and units, a breadcrumb trail, and a 'used by'
//! list. Clicking a term drills into that term's own equation. Leaf terms (inputs) highlight
//! their slider. One equation at a time: never a page of every formula").
//!
//! [`Explorer`] is the panel's state: whether it is shown, the breadcrumb trail of the paths
//! opened (the last one shown), the readout hovered in the last frame and the input row to
//! highlight. A click on a readout opens its value here (a new trail); a click on a term of the
//! open equation, in the equation or in the term list, drills into it (a step on the trail) or,
//! for an input, highlights its row on the inputs side, opening its group and scrolling to it
//! (decision M43-12); a crumb goes back to its step; "used by" goes up the chain. A result
//! without an equation record shows its label, value and workbook cell. The panel docks at the
//! bottom of the centre region, closed until a value is clicked or its header button is
//! pressed (decisions M43-1, M43-2).
//!
//! The terms the panel or the hovered readout shows are marked on screen in their colours
//! ([`Explorer::marks`]): the hovered readout's equation wins, else the open one's. While the
//! marks are another equation's, the term list's swatches fade ([`SWATCH_DIM`]). A long trail
//! shows its last [`SHOWN_CRUMBS`] crumbs after one [`ELIDED`].

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::meta::ResultSet;
    use crate::gui::results_table::table_entries;

    fn design() -> (DesignInputs, DesignResults) {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        (inputs, results)
    }

    #[test]
    fn opening_drilling_and_going_back_walk_the_trail() {
        let mut explorer = Explorer::default();
        assert_eq!(explorer.current(), None);
        explorer.open_path("model.pullout_Nm");
        assert!(explorer.open);
        explorer.drill("model.f_end");
        explorer.drill("model.f_end");
        assert_eq!(explorer.trail(), ["model.pullout_Nm", "model.f_end"]);
        explorer.back_to(0);
        assert_eq!(explorer.current(), Some("model.pullout_Nm"));
        // A readout clicked starts a new trail.
        explorer.drill("model.f_cal");
        explorer.open_path("mass.total_g");
        assert_eq!(explorer.trail(), ["mass.total_g"]);
    }

    #[test]
    fn a_long_walk_keeps_the_last_steps() {
        let mut explorer = Explorer::default();
        explorer.open_path("p0");
        for i in 1..=MAX_TRAIL + 5 {
            explorer.drill(&format!("p{i}"));
        }
        assert_eq!(explorer.trail().len(), MAX_TRAIL);
        assert_eq!(
            explorer.current(),
            Some(format!("p{}", MAX_TRAIL + 5).as_str())
        );
    }

    #[test]
    fn following_a_term_drills_into_a_result_and_focuses_an_input() {
        let (inputs, results) = design();
        let terms = Design {
            inputs: &inputs,
            results: &results,
        };
        let mut explorer = Explorer::default();
        explorer.open_path("model.f_end");
        explorer.follow("coupling.c_end", &terms);
        assert_eq!(
            explorer.current(),
            Some("model.f_end"),
            "an input is a leaf"
        );
        assert_eq!(
            explorer.focus().map(|f| f.path.as_str()),
            Some("coupling.c_end")
        );
        assert!(explorer.take_scroll("coupling.c_end"));
        assert!(!explorer.take_scroll("coupling.c_end"), "once");
        explorer.follow("model.pole_pitch_mm", &terms);
        assert_eq!(explorer.current(), Some("model.pole_pitch_mm"));
        assert_eq!(explorer.focus(), None);
        // A Σ's family term drills into the first harmonic summed.
        explorer.open_path("model.tau_Pa");
        explorer.follow("model.tau#_Pa", &terms);
        assert_eq!(explorer.current(), Some("model.tau1_Pa"));
    }

    #[test]
    fn the_marks_follow_the_hovered_readout_else_the_open_equation() {
        let (inputs, results) = design();
        let mut explorer = Explorer::default();
        assert!(explorer.marks(&inputs, &results).is_empty());
        explorer.open_path("model.f_end");
        assert!(
            explorer
                .marks(&inputs, &results)
                .get("coupling.c_end")
                .is_some()
        );
        explorer.open = false;
        assert!(
            explorer.marks(&inputs, &results).is_empty(),
            "a closed panel marks nothing"
        );
        explorer.open = true;
        let ctx = egui::Context::default();
        explorer.end_frame(
            &ctx,
            ReadoutEvents {
                hovered: Some("model.pullout_Nm".to_owned()),
                ..ReadoutEvents::default()
            },
        );
        let marks = explorer.marks(&inputs, &results);
        assert!(
            marks.get("model.f_end").is_some(),
            "the hovered equation's terms"
        );
        assert!(marks.get("coupling.c_end").is_none());
        explorer.end_frame(
            &ctx,
            ReadoutEvents {
                clicked: Some("mass.total_g".to_owned()),
                ..ReadoutEvents::default()
            },
        );
        assert_eq!(explorer.trail(), ["mass.total_g"]);
        assert_eq!(explorer.hovered(), None);
    }

    #[test]
    fn the_swatches_fade_while_another_equation_s_value_is_hovered() {
        // The marks on screen follow the hovered value's equation (decision M43-3): while it
        // is not the open one, the term list's swatches fade, so a colour on screen is never
        // read against the open equation's key.
        use crate::gui::test_support::{SCREEN, flat_shapes, sized_frame};
        use crate::gui::typeset::TERM_PALETTE;
        let (inputs, results) = design();
        let ctx = egui::Context::default();
        let swatches = |explorer: &mut Explorer| -> Vec<Color32> {
            let output = sized_frame(&ctx, SCREEN, Vec::new(), |ui| {
                explorer_ui(ui, explorer, &inputs, &results);
            });
            flat_shapes(&output)
                .into_iter()
                .filter_map(|shape| match shape {
                    egui::Shape::Rect(r) if r.rect.size() == Vec2::splat(10.0) => Some(r.fill),
                    _ => None,
                })
                .collect()
        };
        let hover = |explorer: &mut Explorer, path: &str| {
            explorer.end_frame(
                &ctx,
                ReadoutEvents {
                    hovered: Some(path.to_owned()),
                    ..ReadoutEvents::default()
                },
            );
        };
        let mut explorer = Explorer::default();
        explorer.open_path("model.pullout_Nm");
        // T_pull = T_2D f_end f_cal: three swatches in the first three colours.
        let full = TERM_PALETTE[..3].to_vec();
        assert_eq!(swatches(&mut explorer), full);
        // The open equation's own value hovered: the marks are its colours.
        hover(&mut explorer, "model.pullout_Nm");
        assert_eq!(swatches(&mut explorer), full);
        // The cup OD hovered: the marks on screen are D_cup's.
        hover(&mut explorer, "model.cup_od_mm");
        let faded: Vec<Color32> = full.iter().map(|c| c.gamma_multiply(SWATCH_DIM)).collect();
        assert_eq!(swatches(&mut explorer), faded);
        explorer.end_frame(&ctx, ReadoutEvents::default());
        assert_eq!(swatches(&mut explorer), full, "the pointer left");
    }

    #[test]
    fn the_term_list_leaves_out_the_harmonics_past_the_set() {
        let (inputs, results) = design();
        let terms = Design {
            inputs: &inputs,
            results: &results,
        };
        let eq = registry().equation_for("model.tau_Pa").unwrap();
        let hidden = unsummed(eq, &terms);
        assert_eq!(hidden, ["model.tau7_Pa", "model.tau9_Pa", "model.tau11_Pa"]);
        assert!(unsummed(registry().equation_for("model.f_end").unwrap(), &terms).is_empty());
    }

    #[test]
    fn labels_symbols_and_choices_read_as_the_panel_shows_them() {
        assert_eq!(term_label("model.f_end"), "End-effect factor");
        assert_eq!(term_label("coupling.c_end"), "End-effect coefficient");
        assert_eq!(term_label("no.such"), "no.such");
        assert_eq!(plain_symbol("ϑ_{op}"), "θ_op");
        assert_eq!(plain_symbol("T_{pull}"), "T_pull");
        assert_eq!(plain_symbol("S_{3}^{iron}"), "S_3^iron");
        assert_eq!(
            value_text("coupling.backiron", &Value::Int(1), "-"),
            "1 (steel circuit)"
        );
        let (_, results) = design();
        let cell_only = table_entries()
            .iter()
            .find(|e| registry().term_kind(&e.path) == Some(TermKind::CellOnly))
            .expect("a result without a record");
        assert!(results.get(&cell_only.path).is_some());
    }
}
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub mod clamp_drawing;
pub mod corrections;
pub mod dashboard;
mod format;
pub mod geometry;
pub mod geometry_view;
````

with:

````rust
pub mod clamp_drawing;
pub mod corrections;
pub mod dashboard;
pub mod explorer;
mod format;
pub mod geometry;
pub mod geometry_view;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            texts.extend(drawn_texts(&harness.click_text(view.label())));
            texts.extend(drawn_texts(&harness.frame(Vec::new())));
        }
        harness.click_text(CentreView::Geometry.label());
        let shown = harness.panel.shown_inputs();
        let geometry = crate::gui::geometry::geometry(&shown, harness.panel.results());
````

with:

````rust
            texts.extend(drawn_texts(&harness.click_text(view.label())));
            texts.extend(drawn_texts(&harness.frame(Vec::new())));
        }
        // The Equation panel open on a chain of temperature symbols (ϑ is drawn as θ).
        harness
            .panel
            .explorer
            .open_path("temperature.summary.governing_limit_C");
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.panel.explorer.open = false;
        harness.click_text(CentreView::Geometry.label());
        let shown = harness.panel.shown_inputs();
        let geometry = crate::gui::geometry::geometry(&shown, harness.panel.results());
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
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

````

with:

````rust
            let value = results.get(&entry.path).unwrap_or(Value::None);
            texts.push(crate::gui::results_table::row_tooltip(entry, &value));
        }
        for text in &texts {
            crate::gui::test_support::assert_glyphs(&harness.ctx, text, "the panel");
        }
    }

````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        let at = text_rect(&output, &pullout).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.hovered.as_deref(), Some("model.pullout_Nm"));
        let texts = drawn_texts(&output);
        assert!(
            texts
````

with:

````rust
        let at = text_rect(&output, &pullout).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.explorer.hovered(), Some("model.pullout_Nm"));
        let texts = drawn_texts(&output);
        assert!(
            texts
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        let eq = registry().equation_for("model.cup_od_mm").unwrap();
        let wall = TermColors::of(eq).get("metal.cup_wall_corner_mm").unwrap();
        let label = InputCatalogue::get()
            .entry("metal.cup_wall_corner_mm")
            .unwrap()
````

with:

````rust
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        let eq = registry().equation_for("model.cup_od_mm").unwrap();
        let wall = crate::gui::typeset::TermColors::of(eq)
            .get("metal.cup_wall_corner_mm")
            .unwrap();
        let label = InputCatalogue::get()
            .entry("metal.cup_wall_corner_mm")
            .unwrap()
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            SCREEN.y - 5.0,
        ))]);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.hovered, None);
        assert!(mark_rects(&output, None).is_empty());
    }

    #[test]
````

with:

````rust
            SCREEN.y - 5.0,
        ))]);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.explorer.hovered(), None);
        assert!(mark_rects(&output, None).is_empty());
    }

    #[test]
    fn the_equation_panel_starts_closed_and_its_header_button_toggles_it() {
        use crate::gui::explorer::{CLOSE, EMPTY_TEXT};
        let mut harness = Harness::new();
        assert!(!harness.panel.explorer.open);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EMPTY_TEXT), 0);
        harness.click_text(EQUATION_PANEL);
        assert!(harness.panel.explorer.open);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EMPTY_TEXT), 1);
        harness.click_text(CLOSE);
        assert!(!harness.panel.explorer.open);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn clicking_a_value_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns() {
        use crate::gui::explorer::TERMS;
        let mut harness = Harness::new();
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.trail(), ["model.pullout_Nm"]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, TERMS), 1);
        // The term list: T_2D, f_end and f_cal by their labels.
        for label in ["2D pull-out torque (infinite length)", "End-effect factor"] {
            assert!(
                drawn_texts(&output).iter().any(|t| t == label),
                "no {label:?}"
            );
        }
        harness.click_text("End-effect factor");
        assert_eq!(
            harness.panel.explorer.trail(),
            ["model.pullout_Nm", "model.f_end"]
        );
        let output = harness.frame(Vec::new());
        assert!(
            drawn_texts(&output)
                .iter()
                .any(|t| t == "End-effect coefficient"),
            "f_end's terms"
        );
        // The first crumb goes back.
        harness.click_text("T_pull");
        assert_eq!(harness.panel.explorer.trail(), ["model.pullout_Nm"]);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_used_by_list_goes_up_the_chain() {
        let mut harness = Harness::new();
        harness.panel.explorer.open_path("model.f_end");
        // The "used by" row wraps: it settles on its second frame.
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        harness.click_text("T_pull");
        assert_eq!(
            harness.panel.explorer.trail(),
            ["model.f_end", "model.pullout_Nm"]
        );
    }

    #[test]
    fn a_leaf_term_opens_its_group_and_highlights_its_input_row() {
        // T_ripple,in = T_pull / (i_g η_g): the gearbox ratio is in the Coupling group, closed
        // by default.
        let mut harness = Harness::new();
        harness
            .panel
            .explorer
            .open_path("model.gearbox_input_ripple_Nm");
        harness.frame(Vec::new());
        harness.click_text("Gearbox ratio");
        assert_eq!(
            harness.panel.explorer.focus().map(|f| f.path.as_str()),
            Some("coupling.gear_ratio")
        );
        assert_eq!(
            harness.panel.explorer.trail(),
            ["model.gearbox_input_ripple_Nm"]
        );
        let mut output = harness.frame(Vec::new());
        for _ in 0..10 {
            output = harness.frame(Vec::new());
        }
        let focus: Vec<egui::Rect> = crate::gui::test_support::flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r) if r.stroke.width == FOCUS_WIDTH => Some(r.rect),
                _ => None,
            })
            .collect();
        // The inputs side paints first: the first "Gearbox ratio" is the row's.
        let label = text_rect(&output, "Gearbox ratio").unwrap();
        assert!(label.left() < INPUTS_WIDTH, "the row on the inputs side");
        assert!(
            focus.iter().any(|r| r.contains_rect(label)),
            "{label:?} not inside {focus:?}"
        );
        assert!(
            !harness.panel.explorer.focus().unwrap().scroll,
            "scrolled once"
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_equation_panel_draws_in_a_tiny_window() {
        // A window smaller than the panel's starting height: nothing panics, every frame.
        for size in [
            egui::vec2(1.0, 1.0),
            egui::vec2(200.0, 150.0),
            egui::vec2(640.0, 240.0),
        ] {
            let mut harness = Harness::on_screen(size);
            harness
                .panel
                .explorer
                .open_path("temperature.summary.governing_limit_C");
            for _ in 0..3 {
                harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
    }

    #[test]
    fn a_long_trail_shows_its_last_crumbs_and_draws_in_a_short_window() {
        use crate::gui::explorer::{ELIDED, MAX_TRAIL, SHOWN_CRUMBS, plain_symbol};
        // A walk longer than the trail, down distinct equations.
        let paths: Vec<&str> = registry()
            .equations()
            .iter()
            .map(|eq| eq.target.as_str())
            .take(MAX_TRAIL + 5)
            .collect();
        let last = plain_symbol(
            &registry()
                .equation_for(paths[MAX_TRAIL + 4])
                .unwrap()
                .symbol,
        );
        // A laptop screen, a short one, and a narrow one: at 640 x 240 the side panels (320 +
        // 300 points) leave the centre region about 20 points, so the panel draws nothing to
        // read there and must only not panic or edit.
        let narrow = egui::vec2(640.0, 240.0);
        for size in [SCREEN, egui::vec2(1280.0, 240.0), narrow] {
            let mut harness = Harness::on_screen(size);
            harness.panel.explorer.open_path(paths[0]);
            for path in &paths[1..] {
                harness.panel.explorer.drill(path);
            }
            assert_eq!(harness.panel.explorer.trail().len(), MAX_TRAIL);
            let mut output = harness.frame(Vec::new());
            for _ in 0..3 {
                output = harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
            if size != narrow {
                // The last crumbs after one ELIDED, each after its ">".
                assert!(drawn_texts(&output).contains(&last), "{size:?}");
                assert_eq!(count(&output, ELIDED), 1);
                assert_eq!(count(&output, ">"), SHOWN_CRUMBS);
                crate::gui::test_support::assert_glyphs(&harness.ctx, ELIDED, "the crumbs");
            }
        }
    }

    #[test]
    fn a_harmonic_set_outside_the_choices_lists_only_its_selector() {
        use crate::gui::explorer::term_label;
        // A design file's max_harmonic of 4 is no choice: every harmonic sum is NaN (decision
        // D3) and τ's term list keeps N_h, every harmonic left out.
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.max_harmonic = 4;
        let expected = harness.panel.inputs.clone();
        harness.panel.explorer.open_path("model.tau_Pa");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert!(matches!(
            harness.panel.results().get("model.tau_Pa"),
            Some(Value::Num(x)) if x.is_nan()
        ));
        let texts = drawn_texts(&output);
        assert!(
            texts
                .iter()
                .any(|t| t == term_label("coupling.max_harmonic"))
        );
        for n in [1, 3, 5, 7, 9, 11] {
            let path = format!("model.tau{n}_Pa");
            let label = term_label(&path);
            assert!(!texts.iter().any(|t| t == label), "{label}");
        }
        assert_eq!(harness.panel.inputs(), &expected);
    }

    #[test]
    fn an_undefined_value_shows_its_equation() {
        // A positive Hcj coefficient typed for manual magnets (no rating): the magnets reach no
        // hot limit, so the torque there takes the record's nan arm, drawn "undefined".
        let mut harness = Harness::new();
        let inputs = &mut harness.panel.inputs;
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        inputs.temperature.demag.coercivity_source = 0;
        inputs.temperature.demag.beta_hcj_per_C = 0.005;
        let expected = inputs.clone();
        harness
            .panel
            .explorer
            .open_path("temperature.demag.torque_at_limit_Nm");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert!(matches!(
            harness
                .panel
                .results()
                .get("temperature.demag.torque_at_limit_Nm"),
            Some(Value::Num(x)) if x.is_nan()
        ));
        assert_eq!(count(&output, "undefined"), 1);
        assert_eq!(harness.panel.inputs(), &expected);
    }

    #[test]
    fn a_result_without_an_equation_shows_its_value_and_cell() {
        use crate::engine::explain::TermKind;
        use crate::gui::explorer::NO_EQUATION;
        let entry = crate::gui::results_table::table_entries()
            .iter()
            .find(|e| registry().term_kind(&e.path) == Some(TermKind::CellOnly))
            .unwrap();
        let mut harness = Harness::new();
        harness.panel.explorer.open_path(&entry.path);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, NO_EQUATION), 1);
        if let Some(cell) = &entry.info.cell {
            assert!(count(&output, cell) >= 1, "{cell}");
        }
    }

    #[test]
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/typeset.rs`, replace:

````rust
    }

    #[test]
    fn theta_is_drawn_for_the_vartheta_the_fonts_lack() {
        assert_eq!(glyph_safe("ϑ_{op}"), "θ_{op}");
        assert_eq!(glyph_safe("T_{pull}"), "T_{pull}");
````

with:

````rust
    }

    #[test]
    fn a_symbol_alone_lays_out_its_base_and_scripts() {
        let laid = with_fonts(|f| layout_symbol(f, registry(), "S_{3}^{iron}", 15.0, Color32::RED));
        let texts = laid.texts();
        let order: Vec<&str> = texts.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(order, ["S", "3", "iron"]);
        let (s, iron) = (rect_of(&texts, "S"), rect_of(&texts, "iron"));
        assert!(iron.center().y < s.center().y, "the superscript is raised");
    }

    #[test]
    fn theta_is_drawn_for_the_vartheta_the_fonts_lack() {
        assert_eq!(glyph_safe("ϑ_{op}"), "θ_{op}");
        assert_eq!(glyph_safe("T_{pull}"), "T_{pull}");
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | tail -1
```

Expected: FAIL to compile. The last line is `` error: could not compile `magcoupling-rs` (lib test) due to 73 previous errors; 1 warning emitted ``; the errors above it come from the tests reaching for what Step 3 adds, among them `` error[E0432]: unresolved imports `crate::gui::explorer::CLOSE`, `crate::gui::explorer::EMPTY_TEXT` ``, `` error[E0432]: unresolved import `crate::gui::explorer::TERMS` ``, `` error[E0432]: unresolved imports `crate::gui::explorer::ELIDED`, `crate::gui::explorer::MAX_TRAIL`, `crate::gui::explorer::SHOWN_CRUMBS`, `crate::gui::explorer::plain_symbol` ``, `` error[E0432]: unresolved import `crate::gui::explorer::term_label` ``, `` error[E0432]: unresolved import `crate::gui::explorer::NO_EQUATION` ``.

- [ ] **Step 3: Write the implementation**

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust

#[cfg(test)]
mod tests {
````

with:

````rust

use egui::{Color32, Sense, Vec2};

use crate::engine::explain::markup::{Expr, Symbol};
use crate::engine::explain::notes::covers;
use crate::engine::explain::{Design, Equation, TermKind};
use crate::engine::meta::{ResultSet, Value};
use crate::gui::dashboard::result_info;
use crate::gui::format::{format_value, with_unit};
use crate::gui::inputs::InputCatalogue;
use crate::gui::readouts::{ReadoutEvents, registry};
use crate::gui::typeset::{
    TermColors, glyph_safe, laid_ui, layout_equation, layout_symbol, term_at,
};
use crate::{DesignInputs, DesignResults};

/// The header button that shows or hides the panel.
pub const EQUATION_PANEL: &str = "Equation panel";

/// The panel's close button.
pub const CLOSE: &str = "Close";

/// What the panel says before anything is opened.
pub const EMPTY_TEXT: &str =
    "Hover a value to see its equation; click it to open it here, then click a term to follow it.";

/// The heading of the term list.
pub const TERMS: &str = "Terms";

/// The heading of the "used by" list.
pub const USED_BY: &str = "Used by";

/// What a result without an equation record shows under its value.
pub const NO_EQUATION: &str =
    "No equation record: the value, its label and its workbook cell are what the explorer knows.";

/// The start of the line naming the corrections an equation embodies.
pub const CORRECTIONS: &str = "Embodies corrections";

/// The text size of the open equation [points].
pub const PANEL_SIZE: f32 = 22.0;

/// The text size of a term's symbol in the term list [points].
pub const TERM_SIZE: f32 = 15.0;

/// The panel's starting height [points].
pub const PANEL_HEIGHT: f32 = 320.0;

/// The longest breadcrumb trail; a longer walk drops its oldest steps.
pub const MAX_TRAIL: usize = 32;

/// The crumbs the header shows: a longer trail shows its last ones after [`ELIDED`] (32 crumbs
/// would wrap over most of a short panel).
pub const SHOWN_CRUMBS: usize = 8;

/// What stands for the crumbs a long trail does not show (its hover counts them).
pub const ELIDED: &str = "\u{2026}";

/// The width of the frame around the input row a leaf term highlights [points].
pub const FOCUS_WIDTH: f32 = 3.0;

/// How strongly the term list's swatches fade while a value of another equation is hovered:
/// the marks on screen are then that equation's colours, not these (decision M43-3).
pub const SWATCH_DIM: f32 = 0.35;

/// The input row a leaf term highlights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Focus {
    /// The input path.
    pub path: String,
    /// The row still has to be scrolled into view (once, when it is first drawn).
    pub scroll: bool,
}

/// The Equation panel's state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Explorer {
    /// Whether the panel is shown.
    pub open: bool,
    /// The paths opened, oldest first; the last one is shown.
    trail: Vec<String>,
    /// The readout hovered in the last frame.
    hovered: Option<String>,
    /// The input row a leaf term highlights.
    focus: Option<Focus>,
}

impl Explorer {
    /// The path shown, if any.
    pub fn current(&self) -> Option<&str> {
        self.trail.last().map(String::as_str)
    }

    /// The breadcrumb trail, oldest first.
    pub fn trail(&self) -> &[String] {
        &self.trail
    }

    /// The readout hovered in the last frame.
    pub fn hovered(&self) -> Option<&str> {
        self.hovered.as_deref()
    }

    /// Opens `path` (a readout clicked): a new trail, the panel shown.
    pub fn open_path(&mut self, path: &str) {
        self.trail = vec![path.to_owned()];
        self.open = true;
        self.focus = None;
    }

    /// Follows a term or a "used by" link to `path`: one more step on the trail.
    pub fn drill(&mut self, path: &str) {
        if self.current() == Some(path) {
            return;
        }
        self.trail.push(path.to_owned());
        if self.trail.len() > MAX_TRAIL {
            self.trail.remove(0);
        }
        self.focus = None;
    }

    /// Goes back to the crumb at `index` (dropping the steps after it).
    pub fn back_to(&mut self, index: usize) {
        self.trail.truncate(index + 1);
        self.focus = None;
    }

    /// Highlights the row of the input at `path` (a leaf term clicked).
    pub fn focus_input(&mut self, path: &str) {
        self.focus = Some(Focus {
            path: path.to_owned(),
            scroll: true,
        });
    }

    /// The input row highlighted, if any.
    pub fn focus(&self) -> Option<&Focus> {
        self.focus.as_ref()
    }

    /// Whether the highlighted row at `path` has to be scrolled into view now: true once.
    pub fn take_scroll(&mut self, path: &str) -> bool {
        match &mut self.focus {
            Some(focus) if focus.path == path && focus.scroll => {
                focus.scroll = false;
                true
            }
            _ => false,
        }
    }

    /// What a click on the term `path` of the open equation does: an input highlights its row;
    /// a result is drilled into; a family template (a Σ's term) drills into the first harmonic
    /// the design sums.
    pub fn follow(&mut self, path: &str, terms: &Design<'_>) {
        match registry().term_kind(path) {
            Some(TermKind::Input { .. }) => self.focus_input(path),
            Some(TermKind::Explained | TermKind::CellOnly) => self.drill(path),
            None => {
                if let Some(first) = registry().family_members(path, terms).first() {
                    self.drill(first);
                }
            }
        }
    }

    /// The term colours this frame marks: the equation of the readout hovered in the last
    /// frame, else the one open (when the panel is shown), a family's by the harmonics the
    /// design sums; none without either.
    pub fn marks(&self, inputs: &DesignInputs, results: &DesignResults) -> TermColors {
        let shown = self.current().filter(|_| self.open);
        let Some(eq) = self
            .hovered
            .as_deref()
            .or(shown)
            .and_then(|path| registry().equation_for(path))
        else {
            return TermColors::none();
        };
        TermColors::of(eq).with_members(registry(), &Design { inputs, results })
    }

    /// Takes in what the user did with the readouts this frame: the value hovered marks its
    /// terms from the next frame (asked for at once); a value clicked opens here.
    pub fn end_frame(&mut self, ctx: &egui::Context, events: ReadoutEvents) {
        if events.hovered != self.hovered {
            self.hovered = events.hovered;
            ctx.request_repaint();
        }
        if let Some(path) = events.clicked {
            self.open_path(&path);
            ctx.request_repaint();
        }
    }
}

/// The label of an input or result path (the path itself for neither).
pub fn term_label(path: &str) -> &str {
    result_info(path)
        .map(|info| info.meta.label)
        .or_else(|| InputCatalogue::get().entry(path).map(|e| e.meta.label))
        .unwrap_or(path)
}

/// A symbol as one line of plain text the default fonts draw (`T_pull`, `θ_op`, `S_3^iron`),
/// for the breadcrumb and the "used by" links.
pub fn plain_symbol(markup: &str) -> String {
    let text = match Symbol::parse(markup) {
        Ok(s) => {
            let mut text = s.base;
            if let Some(sub) = s.sub {
                text.push('_');
                text.push_str(&sub);
            }
            if let Some(sup) = s.sup {
                text.push('^');
                text.push_str(&sup);
            }
            text
        }
        Err(_) => markup.to_owned(),
    };
    glyph_safe(&text)
}

/// The plain symbol of `path`, its label for a path without one.
fn crumb(path: &str) -> String {
    registry()
        .symbol(path)
        .map_or_else(|| term_label(path).to_owned(), plain_symbol)
}

/// A value with its unit; a selector code with its choice's label (`1 (steel circuit)`).
fn value_text(path: &str, value: &Value, unit: &str) -> String {
    let text = with_unit(format_value(value), unit);
    match value {
        Value::Int(code) => registry()
            .choices(path)
            .iter()
            .find(|(c, _)| c == code)
            .map_or(text.clone(), |(_, label)| format!("{text} ({label})")),
        _ => text,
    }
}

/// The paths of family members the formula lists but the design does not sum (the harmonics
/// past the set): the term list leaves them out.
fn unsummed(eq: &Equation, terms: &Design<'_>) -> Vec<String> {
    let mut templates = Vec::new();
    eq.formula.visit(&mut |e| {
        if let Expr::FamilyTerm(r) = e
            && !templates.contains(&r.path)
        {
            templates.push(r.path.clone());
        }
    });
    let summed: Vec<String> = templates
        .iter()
        .flat_map(|t| registry().family_members(t, terms))
        .collect();
    eq.terms
        .iter()
        .filter(|path| templates.iter().any(|t| covers(t, path)) && !summed.contains(path))
        .cloned()
        .collect()
}

/// Draws the Equation panel: its header row (heading, breadcrumb, close), then the path shown.
pub fn explorer_ui(
    ui: &mut egui::Ui,
    explorer: &mut Explorer,
    inputs: &DesignInputs,
    results: &DesignResults,
) {
    let terms = Design { inputs, results };
    ui.horizontal_wrapped(|ui| {
        // Whole crumbs move to the next row (a wrapped text would start mid-row).
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        ui.strong(EQUATION_PANEL);
        ui.separator();
        let mut back = None;
        let last = explorer.trail.len().saturating_sub(1);
        let first = explorer.trail.len().saturating_sub(SHOWN_CRUMBS);
        if first > 0 {
            ui.weak(ELIDED)
                .on_hover_text(format!("{first} earlier steps"));
        }
        for (i, path) in explorer.trail.iter().enumerate().skip(first) {
            if i > 0 {
                ui.weak(">");
            }
            if ui
                .selectable_label(i == last, crumb(path))
                .on_hover_text(term_label(path))
                .clicked()
                && i != last
            {
                back = Some(i);
            }
        }
        if let Some(i) = back {
            explorer.back_to(i);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button(CLOSE).clicked() {
                explorer.open = false;
            }
        });
    });
    ui.separator();
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_equation_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let Some(path) = explorer.current().map(str::to_owned) else {
                ui.weak(EMPTY_TEXT);
                return;
            };
            // The marks on screen are another equation's while its value is hovered.
            let others_marked = explorer.hovered().is_some_and(|h| h != path);
            let follow = match registry().equation_for(&path) {
                Some(eq) => equation_body(ui, eq, &terms, others_marked),
                None => {
                    no_equation_body(ui, &path, results);
                    None
                }
            };
            ui.add_space(6.0);
            let users = registry().used_by(&path);
            let mut up = None;
            if !users.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    ui.strong(USED_BY);
                    for user in users {
                        if ui
                            .link(crumb(user))
                            .on_hover_text(term_label(user))
                            .clicked()
                        {
                            up = Some(user.clone());
                        }
                    }
                });
            }
            if let Some(term) = follow {
                explorer.follow(&term, &terms);
            } else if let Some(user) = up {
                explorer.drill(&user);
            }
        });
}

/// The open equation: its label and cell, the equation large (a term clicked is followed), the
/// value, the corrections it embodies, and the term list, its swatches faded while
/// `others_marked`. Returns the term clicked.
fn equation_body(
    ui: &mut egui::Ui,
    eq: &Equation,
    terms: &Design<'_>,
    others_marked: bool,
) -> Option<String> {
    let mut clicked = None;
    ui.weak(format!(
        "{} ({})",
        eq.label,
        eq.cell.as_deref().unwrap_or("Rust-only result")
    ));
    let colors = TermColors::of(eq);
    let ink = ui.visuals().text_color();
    let laid = ui.fonts(|f| layout_equation(f, registry(), eq, &colors, PANEL_SIZE, ink));
    egui::ScrollArea::horizontal()
        .id_salt("magcoupling_equation_formula")
        .show(ui, |ui| {
            let response = laid_ui(ui, &laid, ink, Sense::click());
            if response.clicked()
                && let Some(at) = response.interact_pointer_pos()
            {
                clicked = term_at(&laid, response.rect.min, at);
            }
        });
    let value = terms.results.get(&eq.target).unwrap_or(Value::None);
    ui.strong(format!(
        "{} = {}",
        plain_symbol(&eq.symbol),
        value_text(&eq.target, &value, eq.unit)
    ));
    let corrections = registry().corrections_upstream(&eq.target);
    if !corrections.is_empty() {
        let ids: Vec<String> = corrections.iter().map(ToString::to_string).collect();
        ui.weak(format!("{CORRECTIONS}: {}", ids.join(", ")));
    }
    ui.add_space(6.0);
    ui.strong(TERMS);
    let hidden = unsummed(eq, terms);
    egui::Grid::new("magcoupling_equation_terms")
        .striped(true)
        .show(ui, |ui| {
            for row in registry().term_rows(eq, terms) {
                if hidden.contains(&row.path) {
                    continue;
                }
                let color = colors
                    .get(&row.path)
                    .or_else(|| {
                        // A family member takes its template's colour.
                        colors_of_template(eq, &colors, &row.path)
                    })
                    .unwrap_or(ink);
                let key = if others_marked {
                    color.gamma_multiply(SWATCH_DIM)
                } else {
                    color
                };
                swatch(ui, key);
                let symbol =
                    ui.fonts(|f| layout_symbol(f, registry(), &row.symbol, TERM_SIZE, color));
                laid_ui(ui, &symbol, ink, Sense::hover());
                let value = row.value.clone().unwrap_or(Value::None);
                ui.label(value_text(&row.path, &value, &row.unit));
                if ui
                    .add(egui::Label::new(term_label(&row.path)).sense(Sense::click()))
                    .on_hover_text(&row.path)
                    .clicked()
                {
                    clicked = Some(row.path.clone());
                }
                ui.weak(kind_text(row.kind));
                ui.end_row();
            }
        });
    clicked
}

/// The colour of the family template of `eq` that covers `path`.
fn colors_of_template(eq: &Equation, colors: &TermColors, path: &str) -> Option<Color32> {
    let mut found = None;
    eq.formula.visit(&mut |e| {
        if let Expr::FamilyTerm(r) = e
            && found.is_none()
            && covers(&r.path, path)
        {
            found = colors.get(&r.path);
        }
    });
    found
}

/// What a term is, in the term list.
fn kind_text(kind: TermKind) -> &'static str {
    match kind {
        TermKind::Input { .. } => "input: click to find its row",
        TermKind::Explained => "click to open its equation",
        TermKind::CellOnly => "no equation record",
    }
}

/// A small filled square in a term's colour.
fn swatch(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
}

/// A result without an equation record: its label, value and cell.
fn no_equation_body(ui: &mut egui::Ui, path: &str, results: &DesignResults) {
    let label = term_label(path);
    let value = results.get(path).unwrap_or(Value::None);
    let unit = result_info(path).map_or("", |info| info.meta.unit);
    ui.strong(format!("{label} = {}", value_text(path, &value, unit)));
    let cell = result_info(path)
        .and_then(|info| info.cell.clone())
        .unwrap_or_else(|| "Rust-only result".to_owned());
    ui.weak(cell);
    ui.weak(NO_EQUATION);
}

#[cfg(test)]
mod tests {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame), and the free variable's
//! row shows that value, locked.

use crate::engine::explain::Design as Terms;
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::clamp_drawing::clamp_ui;
use crate::gui::dashboard::{Level, dashboard_ui, end_effect_banner};
use crate::gui::geometry_view::geometry_ui;
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{ReadoutEvents, Readouts, registry};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
````

with:

````rust
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame), and the free variable's
//! row shows that value, locked.

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::clamp_drawing::clamp_ui;
use crate::gui::dashboard::{Level, dashboard_ui, end_effect_banner};
use crate::gui::explorer::{EQUATION_PANEL, Explorer, FOCUS_WIDTH, PANEL_HEIGHT, explorer_ui};
use crate::gui::geometry_view::geometry_ui;
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{Readouts, registry};
use crate::gui::results_table::{
    CSV_FILE_NAME, JSON_FILE_NAME, ResultsTable, TableAction, results_csv, results_json,
};
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, TARGET_RANGE_INPUT,
    variable_label,
};
use crate::gui::typeset::TermColors;
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
````

with:

````rust
    SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, TARGET_RANGE_INPUT,
    variable_label,
};
use crate::{DesignInputs, DesignResults, compute_all};

/// The heading of the panel.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    results_table: ResultsTable,
    /// Platform work for the host, oldest first.
    requests: Vec<PanelRequest>,
    /// The path of the readout hovered in the last frame: this frame marks its equation's
    /// terms wherever their values are shown (spec Addendum A2 "Hover").
    hovered: Option<String>,
}

impl Default for MagcouplingPanel {
````

with:

````rust
    results_table: ResultsTable,
    /// Platform work for the host, oldest first.
    requests: Vec<PanelRequest>,
    /// The Equation panel: the equation open, its trail, the readout hovered in the last frame
    /// (this frame marks its equation's terms) and the input row a leaf term highlights.
    explorer: Explorer,
}

impl Default for MagcouplingPanel {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            centre: CentreView::Geometry,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
            hovered: None,
        }
    }

````

with:

````rust
            centre: CentreView::Geometry,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
            explorer: Explorer::default(),
        }
    }

````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let mut readouts = Readouts::new(self.marks());
        ui.push_id("magcoupling_panel", |ui| {
            self.shortcuts(ui);
            egui::TopBottomPanel::top("magcoupling_header").show_inside(ui, |ui| {
````

with:

````rust

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let mut readouts = Readouts::new(self.explorer.marks(&self.inputs, &self.results));
        ui.push_id("magcoupling_panel", |ui| {
            self.shortcuts(ui);
            egui::TopBottomPanel::top("magcoupling_header").show_inside(ui, |ui| {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            egui::CentralPanel::default()
                .show_inside(ui, |ui| self.centre_ui(ui, &shown, &mut readouts));
        });
        self.end_frame(ui, readouts.finish());
        // One undo step per settled edit.
        let settled = !self.editing(ui);
        self.history.observe(&self.design(), settled);
    }

    /// The term colours this frame marks: the terms of the equation of the readout hovered in
    /// the last frame, a family's by the harmonics the design sums; none without one.
    fn marks(&self) -> TermColors {
        let Some(eq) = self
            .hovered
            .as_deref()
            .and_then(|path| registry().equation_for(path))
        else {
            return TermColors::none();
        };
        let terms = Terms {
            inputs: &self.inputs,
            results: &self.results,
        };
        TermColors::of(eq).with_members(registry(), &terms)
    }

    /// Takes in what the user did with the readouts this frame: the value hovered marks its
    /// terms from the next frame, which is asked for at once.
    fn end_frame(&mut self, ui: &egui::Ui, events: ReadoutEvents) {
        if events.hovered != self.hovered {
            self.hovered = events.hovered;
            ui.ctx().request_repaint();
        }
    }

    /// Whether an edit of the design is in progress: a pointer button down (a drag), a key held
````

with:

````rust
            egui::CentralPanel::default()
                .show_inside(ui, |ui| self.centre_ui(ui, &shown, &mut readouts));
        });
        self.explorer.end_frame(ui.ctx(), readouts.finish());
        // One undo step per settled edit.
        let settled = !self.editing(ui);
        self.history.observe(&self.design(), settled);
    }

    /// Whether an edit of the design is in progress: a pointer button down (a drag), a key held
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    /// banner when f_end <= 0 (over every view, decision M42-1), then the view of `shown`, the
    /// design shown, its values readouts (`readouts`).
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        ui.horizontal_wrapped(|ui| {
            for view in CentreView::ALL {
                ui.selectable_value(&mut self.centre, view, view.label());
````

with:

````rust
    /// banner when f_end <= 0 (over every view, decision M42-1), then the view of `shown`, the
    /// design shown, its values readouts (`readouts`).
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        // The Equation panel docks at the bottom of the centre region (decision M43-1).
        if self.explorer.open {
            egui::TopBottomPanel::bottom("magcoupling_equation_panel")
                .resizable(true)
                .default_height(PANEL_HEIGHT)
                .show_inside(ui, |ui| {
                    explorer_ui(ui, &mut self.explorer, shown, &self.results);
                });
        }
        ui.horizontal_wrapped(|ui| {
            for view in CentreView::ALL {
                ui.selectable_value(&mut self.centre, view, view.label());
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                ));
                ui.ctx().copy_text(link);
            }
        });
        if let Some(error) = &self.last_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
````

with:

````rust
                ));
                ui.ctx().copy_text(link);
            }
            ui.separator();
            if ui
                .selectable_label(self.explorer.open, EQUATION_PANEL)
                .on_hover_text("Show or hide the equation of the value clicked")
                .clicked()
            {
                self.explorer.open = !self.explorer.open;
            }
        });
        if let Some(error) = &self.last_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        let catalogue = InputCatalogue::get();
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
            .auto_shrink([false, false])
````

with:

````rust
        let catalogue = InputCatalogue::get();
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
        // A leaf term clicked in the Equation panel: its group opens so its row can scroll
        // into view (the Key design group for a Key design input).
        let focus = self
            .explorer
            .focus()
            .filter(|f| f.scroll)
            .map(|f| f.path.clone());
        let open_key_design = focus
            .as_deref()
            .is_some_and(|path| KEY_DESIGN.contains(&path));
        let open_group = focus
            .as_deref()
            .filter(|_| !open_key_design)
            .and_then(|path| {
                catalogue
                    .groups
                    .iter()
                    .find(|g| {
                        g.sections
                            .iter()
                            .any(|s| s.entries.iter().any(|e| e.path == path))
                    })
                    .map(|g| g.name.as_str())
            });
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
            .auto_shrink([false, false])
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                egui::CollapsingHeader::new(KEY_DESIGN_HEADING)
                    .id_salt("key_design")
                    .default_open(true)
                    .show(ui, |ui| {
                        self.sizing_ui(ui, readouts);
                        self.key_widgets.clear();
````

with:

````rust
                egui::CollapsingHeader::new(KEY_DESIGN_HEADING)
                    .id_salt("key_design")
                    .default_open(true)
                    .open(open_key_design.then_some(true))
                    .show(ui, |ui| {
                        self.sizing_ui(ui, readouts);
                        self.key_widgets.clear();
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                    egui::CollapsingHeader::new(group.label)
                        .id_salt(("group", &group.name))
                        .default_open(false)
                        .show(ui, |ui| {
                            for section in &group.sections {
                                if section.prefix != group.name {
````

with:

````rust
                    egui::CollapsingHeader::new(group.label)
                        .id_salt(("group", &group.name))
                        .default_open(false)
                        .open((open_group == Some(group.name.as_str())).then_some(true))
                        .show(ui, |ui| {
                            for section in &group.sections {
                                if section.prefix != group.name {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        let seed = self.seed(entry);
        let row = ui.add_enabled_ui(!locked, |ui| input_row(ui, entry, &current, seed));
        readouts.mark(ui, row.response.rect, &entry.path);
        let output = row.inner;
        if locked {
            ui.weak(SIZED_NOTE);
````

with:

````rust
        let seed = self.seed(entry);
        let row = ui.add_enabled_ui(!locked, |ui| input_row(ui, entry, &current, seed));
        readouts.mark(ui, row.response.rect, &entry.path);
        if self
            .explorer
            .focus()
            .is_some_and(|focus| focus.path == entry.path)
        {
            ui.painter().rect_stroke(
                row.response.rect.expand(3.0),
                3.0,
                egui::Stroke::new(FOCUS_WIDTH, ui.visuals().selection.stroke.color),
                egui::StrokeKind::Outside,
            );
            if self.explorer.take_scroll(&entry.path) {
                row.response.scroll_to_me(Some(egui::Align::Center));
            }
        }
        let output = row.inner;
        if locked {
            ui.weak(SIZED_NOTE);
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/typeset.rs`, replace:

````rust
    layout(fonts, registry, &eq.symbol, &eq.formula, colors, size, ink)
}

/// Paints `laid` with its top-left corner at `origin`; strokes in `ink`.
pub fn paint(painter: &egui::Painter, origin: Pos2, laid: &Laid, ink: Color32) {
    for item in &laid.inks {
````

with:

````rust
    layout(fonts, registry, &eq.symbol, &eq.formula, colors, size, ink)
}

/// Lays out one display symbol (symbol markup) at text size `size` in `color`: a term of the
/// Equation panel's term list.
pub fn layout_symbol(
    fonts: &Fonts,
    registry: &Registry,
    markup: &str,
    size: f32,
    color: Color32,
) -> Laid {
    let formula = Formula {
        body: Expr::NoneLit,
        bindings: Vec::new(),
        cases_count: 0,
    };
    let colors = TermColors::none();
    let setter = Setter {
        fonts,
        registry,
        formula: &formula,
        colors: &colors,
        ink: color,
    };
    setter.symbol(markup, size, color, None, None)
}

/// Paints `laid` with its top-left corner at `origin`; strokes in `ink`.
pub fn paint(painter: &egui::Painter, origin: Pos2, laid: &Laid, ink: Color32) {
    for item in &laid.inks {
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib gui::explorer 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 7 passed; 0 failed` for `gui::explorer`, then `test result: ok. 401 passed; 0 failed` for the whole library.

- [ ] **Step 5: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m43 add magcoupling-rs/src/gui/explorer.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/gui/typeset.rs magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m43 commit -F - <<'EOF'
feat(magcoupling-rs): the docked Equation panel

gui::explorer draws the equation of the value clicked at the bottom of the
centre region (decisions M43-1, M43-2): the breadcrumb, the equation large,
its value and the corrections it embodies (decision M43-14), the term list
with values in the units the formula reads (the harmonics past the set left
out) and "used by". A term clicked drills in, a crumb goes back (decision
M43-9); an input term opens its group, scrolls its row into view once and
frames it (decision M43-12). The hover and the marks move into Explorer: the
hovered equation's terms, else the open one's; the term swatches fade while
another equation's are marked. A long trail shows its last 8 crumbs.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m43 status --short
```

Expected: one new commit on `magcoupling/m4-3`; `status --short` prints nothing.

---

### Task 4: The assumptions view, banner and term styling

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

Spec A3. The inputs side gains a tab row, "Design inputs" / "Assumptions" (`InputsView`, decision M43-7). The Assumptions view lists `engine::assumptions::states`: each assumption's changed-from-default dot and label, its inputs' own rows (`input_row_ui`, so an edit there is an edit, undoable, and an idle frame rewrites nothing), its rationale and its source. The header shows "Assumptions modified: <labels>" with "Reset to workbook defaults" (`assumptions::reset_to_workbook_defaults`: the design inputs stay; an edit like any other) while any assumption differs from its default; when the banner's text changes the panel asks for one more frame (the header sizes from the last frame), and an input edit asks for one too. A leaf term that is an assumption shows the Assumptions view, any other input the design inputs (M43-12). In the Equation panel a term's tag says what it is (`term_tag`: "assumption", the changed dot, "depends on a modified assumption"; `Registry::term_style`), assumption and affected tags in the warning colour, and the open equation names the modified assumptions it depends on (`modified_assumptions_upstream`). The tests pin the tags, the banner through a Key design slider, the reset (the face gap kept) and its undo, the view's every rationale and source with idle frames, the styled terms, and an assumption leaf's row (Review Focus 4 and 5); the glyph test adds the Assumptions view and the banner.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs` (`InputsView`, the constants, `inputs_ui` split into the tabs, `assumptions_ui` and `design_inputs_ui`, the banner; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs` (`term_tag`, the dependence line; tests)

**Interfaces:**
- Consumes: Task 3's `Explorer` (`focus`, `open_path`), `explorer_ui`'s term list; `engine::assumptions::{ASSUMPTIONS, states, modified, any_modified, reset_to_workbook_defaults}`; `Registry::{term_style, modified_assumptions_upstream}`, `engine::explain::TermStyle`; `input_ui::CHANGED_DOT`.
- Produces (`gui::panel`): `pub const ASSUMPTIONS_MODIFIED, RESET_ASSUMPTIONS, SOURCE, ASSUMPTIONS_NOTE: &str`; `#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum InputsView { Design, Assumptions }` with `const ALL: [InputsView; 2]`, `const fn label(self) -> &'static str`; private fields `inputs_view`, `banner`. (`gui::explorer`): `pub const DEPENDS_ON_MODIFIED, ASSUMPTION_TAG: &str`; `pub fn term_tag(TermStyle) -> String`.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
    }

    #[test]
    fn labels_symbols_and_choices_read_as_the_panel_shows_them() {
        assert_eq!(term_label("model.f_end"), "End-effect factor");
        assert_eq!(term_label("coupling.c_end"), "End-effect coefficient");
````

with:

````rust
    }

    #[test]
    fn a_term_s_tag_says_assumption_changed_and_affected() {
        let mut inputs = DesignInputs::default();
        let tag = |path: &str, inputs: &DesignInputs| {
            term_tag(registry().term_style(path, inputs).unwrap())
        };
        assert_eq!(tag("coupling.c_end", &inputs), ASSUMPTION_TAG);
        assert_eq!(
            tag("coupling.npole", &inputs),
            "input: click to find its row"
        );
        assert_eq!(tag("model.f_end", &inputs), "click to open its equation");
        inputs.coupling.c_end = 0.2;
        assert_eq!(
            tag("coupling.c_end", &inputs),
            format!("{CHANGED_DOT} changed; {ASSUMPTION_TAG}")
        );
        assert_eq!(
            tag("model.f_end", &inputs),
            "click to open its equation; depends on a modified assumption"
        );
        inputs.coupling.npole = 12;
        assert_eq!(
            tag("coupling.npole", &inputs),
            format!("{CHANGED_DOT} changed; input: click to find its row")
        );
    }

    #[test]
    fn labels_symbols_and_choices_read_as_the_panel_shows_them() {
        assert_eq!(term_label("model.f_end"), "End-effect factor");
        assert_eq!(term_label("coupling.c_end"), "End-effect coefficient");
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            harness.frame(Vec::new());
        }
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.panel.inputs = short_magnets();
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        let catalogue = InputCatalogue::get();
        texts.extend(catalogue.all().map(crate::gui::inputs::input_tooltip));
        let results = harness.panel.results().clone();
````

with:

````rust
            harness.frame(Vec::new());
        }
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        // The Assumptions view: every rationale and source.
        harness.click_text(InputsView::Assumptions.label());
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.click_text(InputsView::Design.label());
        harness.panel.inputs = short_magnets();
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        // Its c_end of 0.5 is a modified assumption: the banner (the header grows a frame later).
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        assert!(texts.iter().any(|t| t.starts_with(ASSUMPTIONS_MODIFIED)));
        let catalogue = InputCatalogue::get();
        texts.extend(catalogue.all().map(crate::gui::inputs::input_tooltip));
        let results = harness.panel.results().clone();
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
        let ctx = egui::Context::default();
````

with:

````rust
    }

    #[test]
    fn the_assumptions_banner_appears_on_change_and_clears_on_reset() {
        let mut harness = Harness::new();
        let banner = |output: &egui::FullOutput| {
            drawn_texts(output)
                .into_iter()
                .filter(|t| t.starts_with(ASSUMPTIONS_MODIFIED))
                .collect::<Vec<_>>()
        };
        assert!(banner(&harness.frame(Vec::new())).is_empty());
        // The thermal conductance is an assumption and a Key design row.
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.focus("temperature.thermal.conductance_W_K");
        harness.frame(key_tap(egui::Key::ArrowRight));
        // The header sizes from the last frame: the banner line shows from the next one.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            banner(&output),
            [format!("{ASSUMPTIONS_MODIFIED}: Thermal conductance")]
        );
        harness.click_text(RESET_ASSUMPTIONS);
        let output = harness.frame(Vec::new());
        assert!(banner(&output).is_empty());
        let defaults = DesignInputs::default();
        assert_eq!(
            harness.panel.inputs().temperature.thermal.conductance_W_K,
            defaults.temperature.thermal.conductance_W_K
        );
        // The design input stays as it was.
        assert_eq!(harness.number(FACE_GAP), 1.41);
        assert!(!assumptions::any_modified(harness.panel.inputs()));
        // The reset is an edit: undo brings the assumption back.
        harness.panel.undo();
        assert!(assumptions::any_modified(harness.panel.inputs()));
    }

    #[test]
    fn the_assumptions_view_lists_each_assumption_with_its_rationale_and_source() {
        // Tall enough for all fourteen without scrolling.
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 6000.0));
        harness.click_text(InputsView::Assumptions.label());
        assert_eq!(harness.panel.inputs_view, InputsView::Assumptions);
        let output = harness.frame(Vec::new());
        let texts = drawn_texts(&output);
        for assumption in ASSUMPTIONS {
            for want in [
                assumption.label.to_owned(),
                assumption.rationale.to_owned(),
                format!("{SOURCE}: {}", assumption.source),
            ] {
                assert!(texts.contains(&want), "missing {want:?}");
            }
        }
        // Idle frames rewrite no assumption (the rows are the inputs' own).
        for _ in 0..3 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        harness.click_text(InputsView::Design.label());
        assert_eq!(harness.panel.inputs_view, InputsView::Design);
    }

    #[test]
    fn an_assumption_term_is_styled_in_the_equation_panel() {
        use crate::gui::explorer::{ASSUMPTION_TAG, DEPENDS_ON_MODIFIED};
        let mut harness = Harness::new();
        harness.panel.explorer.open_path("model.f_end");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, ASSUMPTION_TAG), 1, "c_end is an assumption");
        harness.panel.inputs.coupling.c_end = 0.2;
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, &format!("{CHANGED_DOT} changed; {ASSUMPTION_TAG}")),
            1
        );
        assert_eq!(
            count(
                &output,
                &format!("{DEPENDS_ON_MODIFIED}: End-effect coefficient")
            ),
            1
        );
    }

    #[test]
    fn a_leaf_assumption_term_shows_its_row_in_the_assumptions_view() {
        let mut harness = Harness::new();
        harness.panel.explorer.open_path("model.f_end");
        harness.frame(Vec::new());
        harness.click_text("End-effect coefficient");
        let mut output = harness.frame(Vec::new());
        for _ in 0..5 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs_view, InputsView::Assumptions);
        let focus: Vec<egui::Rect> = crate::gui::test_support::flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r) if r.stroke.width == FOCUS_WIDTH => Some(r.rect),
                _ => None,
            })
            .collect();
        // The assumption's heading and its row's label are both "End-effect coefficient";
        // the row's label sits inside the frame.
        let labels = crate::gui::test_support::text_rects(&output, "End-effect coefficient");
        assert!(
            labels
                .iter()
                .any(|l| l.left() < INPUTS_WIDTH && focus.iter().any(|r| r.contains_rect(*l))),
            "{labels:?} {focus:?}"
        );
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
        let ctx = egui::Context::default();
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | tail -1
```

Expected: FAIL to compile. The last line is `` error: could not compile `magcoupling-rs` (lib test) due to 24 previous errors ``; the errors above it come from the tests reaching for what Step 3 adds, among them `` error[E0432]: unresolved imports `crate::gui::explorer::ASSUMPTION_TAG`, `crate::gui::explorer::DEPENDS_ON_MODIFIED` ``, `` error[E0425]: cannot find value `ASSUMPTION_TAG` in this scope ``, `` error[E0425]: cannot find value `CHANGED_DOT` in this scope ``, `` error[E0425]: cannot find value `ASSUMPTIONS_MODIFIED` in this scope ``, `` error[E0425]: cannot find value `RESET_ASSUMPTIONS` in this scope ``.

- [ ] **Step 3: Write the implementation**

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
//! ([`Explorer::marks`]): the hovered readout's equation wins, else the open one's. While the
//! marks are another equation's, the term list's swatches fade ([`SWATCH_DIM`]). A long trail
//! shows its last [`SHOWN_CRUMBS`] crumbs after one [`ELIDED`].

use egui::{Color32, Sense, Vec2};

use crate::engine::explain::markup::{Expr, Symbol};
use crate::engine::explain::notes::covers;
use crate::engine::explain::{Design, Equation, TermKind};
use crate::engine::meta::{ResultSet, Value};
use crate::gui::dashboard::result_info;
use crate::gui::format::{format_value, with_unit};
use crate::gui::inputs::InputCatalogue;
use crate::gui::readouts::{ReadoutEvents, registry};
use crate::gui::typeset::{
````

with:

````rust
//! ([`Explorer::marks`]): the hovered readout's equation wins, else the open one's. While the
//! marks are another equation's, the term list's swatches fade ([`SWATCH_DIM`]). A long trail
//! shows its last [`SHOWN_CRUMBS`] crumbs after one [`ELIDED`].
//!
//! Assumptions (spec Addendum A3 "Traceability"): an assumption term is tagged as one in the
//! term list, with the changed-from-default dot when it differs from its workbook default; a
//! result some modified assumption flows into is tagged too, and the open equation says which
//! modified assumptions it depends on ([`Registry::term_style`],
//! [`Registry::modified_assumptions_upstream`]).
//!
//! [`Registry::term_style`]: crate::engine::explain::Registry::term_style
//! [`Registry::modified_assumptions_upstream`]: crate::engine::explain::Registry::modified_assumptions_upstream

use egui::{Color32, Sense, Vec2};

use crate::engine::explain::markup::{Expr, Symbol};
use crate::engine::explain::notes::covers;
use crate::engine::explain::{Design, Equation, TermKind, TermStyle};
use crate::engine::meta::{ResultSet, Value};
use crate::gui::dashboard::result_info;
use crate::gui::format::{format_value, with_unit};
use crate::gui::input_ui::CHANGED_DOT;
use crate::gui::inputs::InputCatalogue;
use crate::gui::readouts::{ReadoutEvents, registry};
use crate::gui::typeset::{
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust

/// The start of the line naming the corrections an equation embodies.
pub const CORRECTIONS: &str = "Embodies corrections";

/// The text size of the open equation [points].
pub const PANEL_SIZE: f32 = 22.0;
````

with:

````rust

/// The start of the line naming the corrections an equation embodies.
pub const CORRECTIONS: &str = "Embodies corrections";

/// The start of the line naming the modified assumptions an equation depends on.
pub const DEPENDS_ON_MODIFIED: &str = "Depends on modified assumptions";

/// The term list's tag of an assumption.
pub const ASSUMPTION_TAG: &str = "assumption: click to find its row";

/// The text size of the open equation [points].
pub const PANEL_SIZE: f32 = 22.0;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
        plain_symbol(&eq.symbol),
        value_text(&eq.target, &value, eq.unit)
    ));
    let corrections = registry().corrections_upstream(&eq.target);
    if !corrections.is_empty() {
        let ids: Vec<String> = corrections.iter().map(ToString::to_string).collect();
````

with:

````rust
        plain_symbol(&eq.symbol),
        value_text(&eq.target, &value, eq.unit)
    ));
    let modified = registry().modified_assumptions_upstream(&eq.target, terms.inputs);
    if !modified.is_empty() {
        let names: Vec<&str> = modified.iter().map(|a| a.label).collect();
        ui.colored_label(
            ui.visuals().warn_fg_color,
            format!("{DEPENDS_ON_MODIFIED}: {}", names.join(", ")),
        );
    }
    let corrections = registry().corrections_upstream(&eq.target);
    if !corrections.is_empty() {
        let ids: Vec<String> = corrections.iter().map(ToString::to_string).collect();
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
                {
                    clicked = Some(row.path.clone());
                }
                ui.weak(kind_text(row.kind));
                ui.end_row();
            }
        });
````

with:

````rust
                {
                    clicked = Some(row.path.clone());
                }
                let style = registry()
                    .term_style(&row.path, terms.inputs)
                    .expect("every term is an input or a result (registry build)");
                let tag = term_tag(style);
                if style.affected_by_modified_assumption
                    || matches!(style.kind, TermKind::Input { assumption: true })
                {
                    ui.colored_label(ui.visuals().warn_fg_color, tag);
                } else {
                    ui.weak(tag);
                }
                ui.end_row();
            }
        });
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
    found
}

/// What a term is, in the term list.
fn kind_text(kind: TermKind) -> &'static str {
    match kind {
        TermKind::Input { .. } => "input: click to find its row",
        TermKind::Explained => "click to open its equation",
        TermKind::CellOnly => "no equation record",
    }
}

/// A small filled square in a term's colour.
````

with:

````rust
    found
}

/// What a term is, in the term list: an assumption, an input, a result with or without an
/// equation; the changed-from-default dot on an input that differs from its default; a result
/// a modified assumption flows into.
pub fn term_tag(style: TermStyle) -> String {
    let kind = match style.kind {
        TermKind::Input { assumption: true } => ASSUMPTION_TAG,
        TermKind::Input { assumption: false } => "input: click to find its row",
        TermKind::Explained => "click to open its equation",
        TermKind::CellOnly => "no equation record",
    };
    let mut tag = if style.changed_from_default {
        format!("{CHANGED_DOT} changed; {kind}")
    } else {
        kind.to_owned()
    };
    if style.affected_by_modified_assumption && !matches!(style.kind, TermKind::Input { .. }) {
        tag.push_str("; depends on a modified assumption");
    }
    tag
}

/// A small filled square in a term's colour.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame), and the free variable's
//! row shows that value, locked.

use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::clamp_drawing::clamp_ui;
````

with:

````rust
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame), and the free variable's
//! row shows that value, locked.

use crate::engine::assumptions::{self, ASSUMPTIONS};
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::sizing::FreeVariable;
use crate::gui::clamp_drawing::clamp_ui;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
use crate::gui::explorer::{EQUATION_PANEL, Explorer, FOCUS_WIDTH, PANEL_HEIGHT, explorer_ui};
use crate::gui::geometry_view::geometry_ui;
use crate::gui::history::History;
use crate::gui::input_ui::{RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{Readouts, registry};
````

with:

````rust
use crate::gui::explorer::{EQUATION_PANEL, Explorer, FOCUS_WIDTH, PANEL_HEIGHT, explorer_ui};
use crate::gui::geometry_view::geometry_ui;
use crate::gui::history::History;
use crate::gui::input_ui::{CHANGED_DOT, RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{Readouts, registry};
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust

/// The note under the free variable's row in Torque → Magnets.
pub const SIZED_NOTE: &str = "Set by Torque -> Magnets";

/// Starting width of the inputs side [points].
const INPUTS_WIDTH: f32 = 320.0;
````

with:

````rust

/// The note under the free variable's row in Torque → Magnets.
pub const SIZED_NOTE: &str = "Set by Torque -> Magnets";

/// The banner shown while an assumption differs from its workbook default (spec Addendum A3),
/// followed by the assumptions' names.
pub const ASSUMPTIONS_MODIFIED: &str = "Assumptions modified";

/// The button beside the banner.
pub const RESET_ASSUMPTIONS: &str = "Reset to workbook defaults";

/// The start of an assumption's source line.
pub const SOURCE: &str = "Source";

/// The line at the top of the Assumptions view.
pub const ASSUMPTIONS_NOTE: &str = "The model's assumptions, apart from the design inputs (each \
     also stays in its input group). The equation panel tags them and what they flow into.";

/// What the inputs side shows (decision M43-7): the design inputs, or the model assumptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputsView {
    Design,
    Assumptions,
}

impl InputsView {
    /// Both views, in tab order.
    pub const ALL: [InputsView; 2] = [InputsView::Design, InputsView::Assumptions];

    /// The tab text.
    pub const fn label(self) -> &'static str {
        match self {
            InputsView::Design => "Design inputs",
            InputsView::Assumptions => "Assumptions",
        }
    }
}

/// Starting width of the inputs side [points].
const INPUTS_WIDTH: f32 = 320.0;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    last_error: Option<String>,
    /// The view the centre region shows.
    centre: CentreView,
    results_table: ResultsTable,
    /// Platform work for the host, oldest first.
    requests: Vec<PanelRequest>,
````

with:

````rust
    last_error: Option<String>,
    /// The view the centre region shows.
    centre: CentreView,
    /// What the inputs side shows.
    inputs_view: InputsView,
    /// The assumptions banner drawn in the last frame: the header sizes from the last frame,
    /// so a change asks for one more frame.
    banner: Option<String>,
    results_table: ResultsTable,
    /// Platform work for the host, oldest first.
    requests: Vec<PanelRequest>,
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            keyboard_shortcuts: true,
            last_error: None,
            centre: CentreView::Geometry,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
            explorer: Explorer::default(),
````

with:

````rust
            keyboard_shortcuts: true,
            last_error: None,
            centre: CentreView::Geometry,
            inputs_view: InputsView::Design,
            banner: None,
            results_table: ResultsTable::default(),
            requests: Vec::new(),
            explorer: Explorer::default(),
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                self.explorer.open = !self.explorer.open;
            }
        });
        if let Some(error) = &self.last_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        } else if let Some(status) = &self.status {
````

with:

````rust
                self.explorer.open = !self.explorer.open;
            }
        });
        // Spec Addendum A3: the banner while any assumption differs from its workbook default.
        let modified = assumptions::modified(&self.inputs);
        let banner = (!modified.is_empty()).then(|| {
            let names: Vec<&str> = modified.iter().map(|a| a.label).collect();
            format!("{ASSUMPTIONS_MODIFIED}: {}", names.join(", "))
        });
        if banner != self.banner {
            self.banner.clone_from(&banner);
            ui.ctx().request_repaint();
        }
        if let Some(banner) = banner {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(ui.visuals().warn_fg_color, banner);
                if ui
                    .button(RESET_ASSUMPTIONS)
                    .on_hover_text(
                        "Every assumption back to its workbook default; the design inputs stay as they are",
                    )
                    .clicked()
                {
                    assumptions::reset_to_workbook_defaults(&mut self.inputs);
                    self.status = Some("Assumptions reset to the workbook defaults".to_owned());
                    self.last_error = None;
                }
            });
        }
        if let Some(error) = &self.last_error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        } else if let Some(status) = &self.status {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        }
    }

    /// The left side: the Key design group, then every input by package group.
    fn inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let catalogue = InputCatalogue::get();
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
        // A leaf term clicked in the Equation panel: its group opens so its row can scroll
        // into view (the Key design group for a Key design input).
        let focus = self
````

with:

````rust
        }
    }

    /// The left side: the tabs of the design inputs and the assumptions, then the view chosen.
    /// A leaf term clicked in the Equation panel shows the view that holds its row.
    fn inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
        if let Some(focus) = self.explorer.focus().filter(|f| f.scroll) {
            let assumption = ASSUMPTIONS
                .iter()
                .any(|a| a.paths.contains(&focus.path.as_str()));
            self.inputs_view = if assumption {
                InputsView::Assumptions
            } else {
                InputsView::Design
            };
        }
        ui.horizontal(|ui| {
            for view in InputsView::ALL {
                ui.selectable_value(&mut self.inputs_view, view, view.label());
            }
        });
        ui.separator();
        match self.inputs_view {
            InputsView::Design => self.design_inputs_ui(ui, readouts),
            InputsView::Assumptions => self.assumptions_ui(ui, readouts),
        }
    }

    /// The Assumptions view (spec Addendum A3 "Toggle panel"): each assumption with its
    /// changed-from-default dot, the rows of its inputs (value, unit, reset: the inputs' own
    /// rows, so an edit here is an edit like any other), its rationale and its source.
    fn assumptions_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let catalogue = InputCatalogue::get();
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_assumptions_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add(egui::Label::new(egui::RichText::new(ASSUMPTIONS_NOTE).weak()).wrap());
                for state in assumptions::states(&self.inputs) {
                    let assumption = state.assumption;
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        let dot = if state.modified { CHANGED_DOT } else { " " };
                        ui.colored_label(ui.visuals().selection.stroke.color, dot)
                            .on_hover_text("Changed from the workbook default");
                        ui.strong(assumption.label);
                    });
                    for path in assumption.paths {
                        let entry = catalogue
                            .entry(path)
                            .expect("an assumption path is an input (tests/assumptions.rs)");
                        let widget = self.input_row_ui(ui, entry, readouts);
                        self.design_widgets.push(widget);
                    }
                    ui.add(
                        egui::Label::new(egui::RichText::new(assumption.rationale).weak()).wrap(),
                    );
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!("{SOURCE}: {}", assumption.source))
                                .small()
                                .weak(),
                        )
                        .wrap(),
                    );
                }
            });
    }

    /// The design inputs: the Key design group, then every input by package group.
    fn design_inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let catalogue = InputCatalogue::get();
        // A leaf term clicked in the Equation panel: its group opens so its row can scroll
        // into view (the Key design group for a Key design input).
        let focus = self
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                .set(&entry.path, value)
                .err()
                .map(|e| e.to_string());
        }
        output.widget.id
    }
````

with:

````rust
                .set(&entry.path, value)
                .err()
                .map(|e| e.to_string());
            // The header (the assumptions banner) was drawn before this edit.
            ui.ctx().request_repaint();
        }
        output.widget.id
    }
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib gui::panel 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 74 passed; 0 failed` for `gui::panel`, then `test result: ok. 406 passed; 0 failed` for the whole library.

- [ ] **Step 5: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m43 add magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/explorer.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m43 commit -F - <<'EOF'
feat(magcoupling-rs): the assumptions view, banner and reset; assumption terms tagged

The inputs side's tab row switches to the Assumptions view (decision M43-7):
each assumption's dot, its inputs' own rows, its rationale and source. While
any assumption differs from its workbook default the header shows the
"Assumptions modified" banner beside "Reset to workbook defaults" (design
inputs kept, undoable). An assumption leaf shows its row there. In the
Equation panel assumption terms, changed inputs and results a modified
assumption flows into are tagged, and the equation names the modified
assumptions it depends on.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m43 status --short
```

Expected: one new commit on `magcoupling/m4-3`; `status --short` prints nothing.

---

### Task 5: Teaching notes: Explain, start here and the six diagrams

**Model:** `sonnet` for the implementation (the plan gives the exact code). **The per-task review runs on the session model** (`// session model: diagrams restate physics`; Global Constraints). Escalate a retry to the session model.

Spec A4. The Equation panel's header gains an Explain checkbox (off by default) and a "Start here" button. Every note the panel names or draws by id (a start-here step's title, a note opened alone) goes through the new `reviewed_note`, as `notes::note_for` gates an equation's, so a note sent back to Draft never shows. With Explain on, the open equation's reviewed note (`notes::note_for`: only notes past the accuracy gate) shows under its value line, before the term list (decision M43-10): title, sentences, watch-out line, its diagram and its sources, every text through `glyph_safe`; an equation without one says "No teaching note for this equation." "Start here" walks `notes::START_HERE`: each step opens its equation with Explain on, under a bar with Previous, Next and Stop; drilling keeps the walk, a readout clicked leaves it. A note can be opened on its own (`Explorer::open_note`, for Task 6's warnings): its title as the crumb, links to the equations it explains first, then the note. `gui/diagrams.rs` paints the six `notes::Diagram` kinds with egui's painter, schematics of the idea, not plots of the design: the square wave of fill 0.8 with its fundamental and the sum of harmonics 1, 3, 5 (amplitudes 4/(nπ) sin(nπλ/2)); flux loops closing through steel against loops bulging out in free space; torque against electrical angle with a third harmonic and its peak; field lines fringing at a block's ends; the intrinsic curve with its knee and three dashed load lines; first-order heating with τ at 63 % and the steady temperature. The tests pin every diagram inside its rect with finite points, labels with glyphs and six distinct label sets, the harmonic sum, the start-here walk and its bounds, every start-here step's equation and reviewed note, `reviewed_note` against every note's review, a note alone, the Explain toggle following the open equation (with a diagram painted), start here through the buttons, and every note's texts against the fonts (Review Focus 1; the diagrams' and the notes' glyph checks go through `assert_glyphs`); the tiny-window test runs with Explain on.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/diagrams.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/mod.rs` (`pub mod diagrams;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs` (Explain, start here, a note alone, `note_ui`; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs` (tests)

**Interfaces:**
- Consumes: Task 3's `Explorer`, `explorer_ui`, `equation_body`; Task 1's `glyph_safe`, `TERM_PALETTE`, `test_support::assert_glyphs`; `engine::explain::notes::{self, Diagram, Note, Review, START_HERE}`.
- Produces (`magcoupling::gui::diagrams`): `pub const DIAGRAM_SIZE: Vec2` (300 x 130); `pub fn diagram_shapes(&Fonts, Diagram, Rect, &egui::Visuals) -> Vec<Shape>`; `pub fn diagram_ui(&mut egui::Ui, Diagram) -> egui::Response`. (`gui::explorer`): `pub explain: bool` on `Explorer`; `fn open_note(&mut self, &'static str)`, `fn note(&self) -> Option<&'static str>`, `fn start(&mut self, usize)`, `fn start_here(&self) -> Option<usize>`, `fn stop(&mut self)`; `pub const EXPLAIN, START_HERE_BUTTON, PREVIOUS, NEXT, STOP, NO_NOTE, WATCH_OUT, SOURCES, EXPLAINS: &str`; `pub fn start_here_text(usize) -> String`; `pub fn note_ui(&mut egui::Ui, &Note)`; `pub fn reviewed_note(&str) -> Option<&'static Note>`.

- [ ] **Step 1: Write the failing tests**

Create `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/diagrams.rs`:

````rust
//! The teaching notes' small diagrams (spec Addendum A4: "an optional small diagram (for
//! example the square-wave magnetization and its harmonics, and the flux path with and without
//! back iron)"), drawn with egui's painter: no image files.
//!
//! [`diagram_shapes`] builds the shapes of one [`Diagram`] kind inside a rect (pure, testable);
//! [`diagram_ui`] allocates [`DIAGRAM_SIZE`] and paints them. Each diagram is a schematic of
//! the idea its note explains, not a plot of the design's numbers: the square wave of fill 0.8
//! and the sum of its harmonics 1, 3 and 5 (amplitudes 4/(nπ) sin(nπλ/2)); the flux closing
//! through steel against spreading behind the magnets; torque against electrical angle with
//! a third harmonic and its peak; the field fringing at a magnet's ends; the intrinsic
//! demagnetization curve with its knee and three load lines; first-order heating with the time
//! constant at 63 %.

#[cfg(test)]
mod tests {
    use super::*;

    const KINDS: [Diagram; 6] = [
        Diagram::SquareWaveHarmonics,
        Diagram::FluxPathBackIron,
        Diagram::TorqueAngle,
        Diagram::EndFringing,
        Diagram::DemagKnee,
        Diagram::HeatingCurve,
    ];

    /// Every point a shape draws (its path, its circle's centre, its rect's corners, its
    /// text's rect corners).
    fn points(shape: &Shape) -> Vec<Pos2> {
        match shape {
            Shape::Path(p) => p.points.clone(),
            Shape::LineSegment { points, .. } => points.to_vec(),
            Shape::Circle(c) => vec![c.center],
            Shape::Rect(r) => vec![r.rect.min, r.rect.max],
            Shape::Text(t) => {
                let rect = t.galley.rect.translate(t.pos.to_vec2());
                vec![rect.min, rect.max]
            }
            Shape::Vec(v) => v.iter().flat_map(points).collect(),
            _ => Vec::new(),
        }
    }

    fn shapes_of(kind: Diagram, rect: Rect) -> (Vec<Shape>, Vec<String>) {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let visuals = egui::Visuals::dark();
        let shapes = ctx.fonts(|f| diagram_shapes(f, kind, rect, &visuals));
        let texts: Vec<String> = shapes
            .iter()
            .filter_map(|s| match s {
                Shape::Text(t) => Some(t.galley.text().to_owned()),
                _ => None,
            })
            .collect();
        for text in &texts {
            crate::gui::test_support::assert_glyphs(&ctx, text, &format!("{kind:?}"));
        }
        (shapes, texts)
    }

    #[test]
    fn every_diagram_draws_inside_its_rect_with_finite_points_and_labels() {
        let rect = Rect::from_min_size(pos2(40.0, 30.0), DIAGRAM_SIZE);
        let mut signatures = Vec::new();
        for kind in KINDS {
            let (shapes, texts) = shapes_of(kind, rect);
            assert!(shapes.len() >= 5, "{kind:?}: {} shapes", shapes.len());
            assert!(!texts.is_empty(), "{kind:?} is labelled");
            for shape in &shapes {
                for p in points(shape) {
                    assert!(p.x.is_finite() && p.y.is_finite(), "{kind:?}: {p:?}");
                    assert!(
                        rect.expand(1.0).contains(p),
                        "{kind:?}: {p:?} outside {rect:?}"
                    );
                }
            }
            signatures.push(texts);
        }
        // Six diagrams, six sets of labels.
        for (i, a) in signatures.iter().enumerate() {
            for b in &signatures[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn the_harmonics_sum_follows_the_square_wave() {
        // At a north block's centre the wave is 1 and the sum of 1, 3, 5 is near it; its
        // fundamental alone is 4/π sin(0.4π) = 1.211.
        let fill = 0.8f32;
        let sum: f32 = [1.0f32, 3.0, 5.0]
            .iter()
            .map(|n| 4.0 / (n * PI) * (n * PI * fill / 2.0).sin())
            .sum();
        assert!((sum - 1.0).abs() < 0.25, "{sum}");
        let first = 4.0 / PI * (PI * fill / 2.0).sin();
        assert!((first - 1.211).abs() < 1e-3, "{first}");
    }
}
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
    }

    #[test]
    fn a_term_s_tag_says_assumption_changed_and_affected() {
        let mut inputs = DesignInputs::default();
        let tag = |path: &str, inputs: &DesignInputs| {
````

with:

````rust
    }

    #[test]
    fn start_here_walks_the_suggested_order_and_a_click_leaves_it() {
        let mut explorer = Explorer::default();
        assert!(!explorer.explain, "Explain is off by default");
        explorer.start(0);
        assert_eq!(explorer.start_here(), Some(0));
        assert!(explorer.explain && explorer.open);
        assert_eq!(explorer.current(), Some(START_HERE[0].1));
        let last = START_HERE.len() - 1;
        explorer.start(last);
        explorer.start(last + 1);
        assert_eq!(
            explorer.start_here(),
            Some(last),
            "past the last step: nothing"
        );
        assert_eq!(explorer.current(), Some(START_HERE[last].1));
        // Drilling keeps the walk; a readout clicked leaves it.
        explorer.drill("clamps.preload_N");
        assert_eq!(explorer.start_here(), Some(last));
        explorer.open_path("mass.total_g");
        assert_eq!(explorer.start_here(), None);
        explorer.start(2);
        explorer.stop();
        assert_eq!(explorer.start_here(), None);
        assert_eq!(explorer.current(), Some(START_HERE[2].1));
        assert_eq!(
            start_here_text(0),
            format!(
                "{START_HERE_BUTTON} 1 of {}: Harmonic decomposition",
                START_HERE.len()
            )
        );
    }

    #[test]
    fn every_start_here_step_opens_an_equation_with_its_reviewed_note() {
        for &(id, path) in START_HERE {
            assert!(registry().equation_for(path).is_some(), "{path}");
            assert_eq!(notes::note_for(path).map(|n| n.id), Some(id), "{path}");
        }
    }

    #[test]
    fn a_note_by_id_shows_only_once_reviewed() {
        use crate::engine::explain::notes::NOTES;
        for note in NOTES {
            let reviewed = matches!(note.review, Review::Reviewed { .. });
            assert_eq!(
                reviewed_note(note.id).map(|n| n.id),
                reviewed.then_some(note.id)
            );
        }
        assert_eq!(reviewed_note("no.such.note").map(|n| n.id), None);
    }

    #[test]
    fn a_note_opened_alone_replaces_the_trail_until_a_value_is_opened() {
        let mut explorer = Explorer::default();
        explorer.open_path("model.f_end");
        explorer.open_note("a5.low_saturation");
        assert_eq!(explorer.note(), Some("a5.low_saturation"));
        assert_eq!(explorer.current(), None);
        assert!(explorer.open);
        explorer.open_path("materials.circuit_backiron");
        assert_eq!(explorer.note(), None);
        explorer.open_note("a5.low_saturation");
        explorer.start(0);
        assert_eq!(explorer.note(), None);
    }

    #[test]
    fn a_term_s_tag_says_assumption_changed_and_affected() {
        let mut inputs = DesignInputs::default();
        let tag = |path: &str, inputs: &DesignInputs| {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub mod clamp_drawing;
pub mod corrections;
pub mod dashboard;
pub mod explorer;
mod format;
pub mod geometry;
````

with:

````rust
pub mod clamp_drawing;
pub mod corrections;
pub mod dashboard;
pub mod diagrams;
pub mod explorer;
mod format;
pub mod geometry;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                .panel
                .explorer
                .open_path("temperature.summary.governing_limit_C");
            for _ in 0..3 {
                harness.frame(Vec::new());
            }
````

with:

````rust
                .panel
                .explorer
                .open_path("temperature.summary.governing_limit_C");
            harness.panel.explorer.explain = true;
            for _ in 0..3 {
                harness.frame(Vec::new());
            }
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        );
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
````

with:

````rust
        );
    }

    /// Every painted shape that is a diagram's: the frames of the term marks and the panel's
    /// own shapes aside, a diagram paints filled circles of radius 3.5 (its marked points).
    fn diagram_points(output: &egui::FullOutput) -> usize {
        crate::gui::test_support::flat_shapes(output)
            .into_iter()
            .filter(|s| matches!(s, egui::Shape::Circle(c) if c.radius == 3.5))
            .count()
    }

    #[test]
    fn the_explain_toggle_shows_the_open_equation_s_note_and_follows_it() {
        use crate::engine::explain::notes::note;
        use crate::gui::explorer::{EXPLAIN, NO_NOTE};
        let mut harness = Harness::new();
        harness.panel.explorer.open_path("model.pullout_Nm");
        harness.frame(Vec::new());
        let pullout = note("pullout_angle").unwrap();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, pullout.title), 0, "hidden by default");
        harness.click_text(EXPLAIN);
        assert!(harness.panel.explorer.explain);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, pullout.title), 1);
        assert_eq!(count(&output, pullout.sentences[0]), 1);
        assert!(diagram_points(&output) >= 1, "its torque-angle diagram");
        // Drill to f_end (its term row is below the note now): the note follows the equation.
        harness.panel.explorer.drill("model.f_end");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let end = note("end_effect").unwrap();
        assert_eq!(count(&output, end.title), 1);
        assert_eq!(count(&output, pullout.title), 0);
        // An equation no reviewed note explains.
        harness.panel.explorer.open_path("mass.total_g");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, NO_NOTE), 1);
        // Off again: no note.
        harness.click_text(EXPLAIN);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, NO_NOTE), 0);
    }

    #[test]
    fn start_here_opens_the_matching_equations_in_turn() {
        use crate::engine::explain::notes::{START_HERE, note};
        use crate::gui::explorer::{NEXT, PREVIOUS, START_HERE_BUTTON, STOP, start_here_text};
        let mut harness = Harness::new();
        harness.click_text(EQUATION_PANEL);
        harness.click_text(START_HERE_BUTTON);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[0].1));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &start_here_text(0)), 1);
        assert_eq!(count(&output, note(START_HERE[0].0).unwrap().title), 1);
        harness.click_text(NEXT);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[1].1));
        harness.click_text(NEXT);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[2].1));
        harness.click_text(PREVIOUS);
        assert_eq!(harness.panel.explorer.start_here(), Some(1));
        harness.click_text(STOP);
        assert_eq!(harness.panel.explorer.start_here(), None);
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[1].1));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn every_teaching_note_shows_with_glyphs_in_the_default_fonts() {
        use crate::engine::explain::notes::NOTES;
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        for note in NOTES {
            harness.panel.explorer.open_note(note.id);
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            let texts = drawn_texts(&output);
            let title = crate::gui::typeset::glyph_safe(note.title);
            assert!(texts.contains(&title), "{}", note.id);
            for text in &texts {
                crate::gui::test_support::assert_glyphs(&harness.ctx, text, note.id);
            }
        }
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | tail -1
```

Expected: FAIL to compile. The last line is `` error: could not compile `magcoupling-rs` (lib test) due to 65 previous errors; 1 warning emitted ``; the errors above it come from the tests reaching for what Step 3 adds, among them `` error[E0432]: unresolved imports `crate::gui::explorer::EXPLAIN`, `crate::gui::explorer::NO_NOTE` ``, `` error[E0432]: unresolved imports `crate::gui::explorer::NEXT`, `crate::gui::explorer::PREVIOUS`, `crate::gui::explorer::START_HERE_BUTTON`, `crate::gui::explorer::STOP`, `crate::gui::explorer::start_here_text` ``, `` error[E0412]: cannot find type `Diagram` in this scope ``, `` error[E0433]: failed to resolve: use of undeclared type `Diagram` ``, `` error[E0412]: cannot find type `Shape` in this scope ``.

- [ ] **Step 3: Write the implementation**

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/diagrams.rs`, replace:

````rust

#[cfg(test)]
mod tests {
````

with:

````rust

use std::f32::consts::PI;

use egui::text::Fonts;
use egui::{Align2, Color32, FontId, Pos2, Rect, Shape, Stroke, Vec2, pos2, vec2};

use crate::engine::explain::notes::Diagram;
use crate::gui::typeset::TERM_PALETTE;

/// The size a diagram takes [points].
pub const DIAGRAM_SIZE: Vec2 = vec2(300.0, 130.0);

/// The text size of a diagram's labels [points].
const LABEL_SIZE: f32 = 11.0;

/// The colours a diagram draws with.
struct Pens {
    ink: Color32,
    weak: Color32,
    north: Color32,
    south: Color32,
    steel: Color32,
    first: Color32,
    second: Color32,
}

impl Pens {
    fn of(visuals: &egui::Visuals) -> Self {
        Pens {
            ink: visuals.text_color(),
            weak: visuals.weak_text_color(),
            north: Color32::from_rgb(220, 110, 90),
            south: Color32::from_rgb(90, 130, 210),
            steel: Color32::from_gray(120),
            first: TERM_PALETTE[0],
            second: TERM_PALETTE[1],
        }
    }
}

/// `n` points of `f` over `[0, 1]`, mapped into `rect` (y up, `f` from `lo` to `hi`).
fn curve(rect: Rect, lo: f32, hi: f32, n: usize, f: impl Fn(f32) -> f32) -> Vec<Pos2> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            let y = (f(t) - lo) / (hi - lo);
            pos2(
                rect.left() + t * rect.width(),
                rect.bottom() - y * rect.height(),
            )
        })
        .collect()
}

/// The points of an elliptic arc around `centre` with radii `r`, from angle `from` to `to`.
fn arc(centre: Pos2, r: Vec2, from: f32, to: f32) -> Vec<Pos2> {
    (0..=16)
        .map(|i| {
            let a = from + (to - from) * i as f32 / 16.0;
            centre + vec2(r.x * a.cos(), -r.y * a.sin())
        })
        .collect()
}

fn label(fonts: &Fonts, at: Pos2, anchor: Align2, text: &str, color: Color32) -> Shape {
    Shape::text(
        fonts,
        at,
        anchor,
        text,
        FontId::proportional(LABEL_SIZE),
        color,
    )
}

/// The shapes of the diagram `kind` inside `rect`.
pub fn diagram_shapes(
    fonts: &Fonts,
    kind: Diagram,
    rect: Rect,
    visuals: &egui::Visuals,
) -> Vec<Shape> {
    let pens = Pens::of(visuals);
    let inner = rect.shrink(12.0);
    match kind {
        Diagram::SquareWaveHarmonics => square_wave(fonts, inner, &pens),
        Diagram::FluxPathBackIron => flux_paths(fonts, inner, &pens),
        Diagram::TorqueAngle => torque_angle(fonts, inner, &pens),
        Diagram::EndFringing => end_fringing(fonts, inner, &pens),
        Diagram::DemagKnee => demag_knee(fonts, inner, &pens),
        Diagram::HeatingCurve => heating_curve(fonts, inner, &pens),
    }
}

/// The magnetization of two pole pairs at fill 0.8 (blocks and gaps), the fundamental and the
/// sum of harmonics 1, 3 and 5.
fn square_wave(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let fill = 0.8;
    let poles = 4.0;
    // +1 over the north blocks, -1 over the south ones, 0 in the gaps.
    let wave = |t: f32| {
        let x = t * poles;
        let pole = x.floor();
        let within = (x - pole - 0.5).abs() * 2.0;
        if within <= fill {
            if pole as i32 % 2 == 0 { 1.0 } else { -1.0 }
        } else {
            0.0
        }
    };
    let harmonic = |n: f32, t: f32| {
        let amp = 4.0 / (n * PI) * (n * PI * fill / 2.0).sin();
        // Pole centres at x = 0.5, 1.5, ... (in poles): cos(n π (x - 0.5)).
        amp * (n * PI * (t * poles - 0.5)).cos()
    };
    let plot = Rect::from_min_max(r.min, pos2(r.max.x, r.max.y - 14.0));
    let axis = plot.center().y;
    let mut shapes = vec![Shape::line_segment(
        [pos2(plot.left(), axis), pos2(plot.right(), axis)],
        Stroke::new(1.0, pens.weak),
    )];
    shapes.push(Shape::line(
        curve(plot, -1.4, 1.4, 241, wave),
        Stroke::new(1.5, pens.ink),
    ));
    shapes.push(Shape::line(
        curve(plot, -1.4, 1.4, 241, |t| harmonic(1.0, t)),
        Stroke::new(1.0, pens.second),
    ));
    shapes.push(Shape::line(
        curve(plot, -1.4, 1.4, 241, |t| {
            harmonic(1.0, t) + harmonic(3.0, t) + harmonic(5.0, t)
        }),
        Stroke::new(1.5, pens.first),
    ));
    shapes.push(label(
        fonts,
        pos2(r.left(), r.bottom()),
        Align2::LEFT_BOTTOM,
        "blocks and gaps (fill 0.8)",
        pens.ink,
    ));
    shapes.push(label(
        fonts,
        pos2(r.center().x + 10.0, r.bottom()),
        Align2::LEFT_BOTTOM,
        "n = 1",
        pens.second,
    ));
    shapes.push(label(
        fonts,
        pos2(r.right(), r.bottom()),
        Align2::RIGHT_BOTTOM,
        "1 + 3 + 5",
        pens.first,
    ));
    shapes
}

/// One panel of the flux-path diagram: a ring of four blocks above the gap and one below,
/// with steel behind both (`steel`) or not, and two flux loops.
fn flux_panel(fonts: &Fonts, r: Rect, steel: bool, pens: &Pens, title: &str) -> Vec<Shape> {
    let mut shapes = Vec::new();
    let block_w = r.width() / 4.0;
    let block_h = 12.0;
    let gap = 14.0;
    let mid = r.center().y + 4.0;
    let top = mid - gap / 2.0 - block_h;
    let bottom = mid + gap / 2.0;
    for i in 0..4 {
        let x = r.left() + i as f32 * block_w;
        let (upper, lower) = if i % 2 == 0 {
            (pens.north, pens.south)
        } else {
            (pens.south, pens.north)
        };
        shapes.push(Shape::rect_filled(
            Rect::from_min_size(pos2(x + 1.0, top), vec2(block_w - 2.0, block_h)),
            1.0,
            upper,
        ));
        shapes.push(Shape::rect_filled(
            Rect::from_min_size(pos2(x + 1.0, bottom), vec2(block_w - 2.0, block_h)),
            1.0,
            lower,
        ));
    }
    if steel {
        for y in [top - 8.0, bottom + block_h] {
            shapes.push(Shape::rect_filled(
                Rect::from_min_size(pos2(r.left(), y), vec2(r.width(), 8.0)),
                1.0,
                pens.steel,
            ));
        }
    }
    // Two loops, each across the gap between neighbouring blocks: through the steel they stay
    // inside it; in free space they bulge out behind the magnets.
    for i in [0.5f32, 2.5] {
        let cx = r.left() + (i + 0.5) * block_w;
        let (rx, ry) = if steel {
            (block_w * 0.55, gap / 2.0 + block_h + 4.0)
        } else {
            (block_w * 0.9, gap / 2.0 + block_h + 28.0)
        };
        shapes.push(Shape::closed_line(
            arc(pos2(cx, mid), vec2(rx, ry), 0.0, 2.0 * PI),
            Stroke::new(1.0, pens.first),
        ));
    }
    shapes.push(label(
        fonts,
        pos2(r.center().x, r.top()),
        Align2::CENTER_TOP,
        title,
        pens.ink,
    ));
    shapes
}

fn flux_paths(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let half = r.width() / 2.0 - 8.0;
    let left = Rect::from_min_size(r.min, vec2(half, r.height()));
    let right = Rect::from_min_size(pos2(r.right() - half, r.top()), vec2(half, r.height()));
    let mut shapes = flux_panel(fonts, left, true, pens, "with back iron");
    shapes.extend(flux_panel(fonts, right, false, pens, "free space"));
    shapes
}

/// Torque against electrical angle over 0 to π with a third harmonic, its peak marked.
fn torque_angle(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let plot = Rect::from_min_max(
        pos2(r.left() + 14.0, r.top()),
        pos2(r.right(), r.bottom() - 14.0),
    );
    let torque = |t: f32| (t * PI).sin() + 0.15 * (3.0 * t * PI).sin();
    let points = curve(plot, 0.0, 1.2, 181, torque);
    let peak = points
        .iter()
        .copied()
        .min_by(|a, b| a.y.total_cmp(&b.y))
        .unwrap_or(plot.center());
    vec![
        Shape::line_segment(
            [plot.left_bottom(), plot.right_bottom()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line_segment(
            [plot.left_bottom(), plot.left_top()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line(points, Stroke::new(1.5, pens.first)),
        Shape::circle_filled(peak, 3.5, pens.second),
        label(
            fonts,
            peak + vec2(6.0, 0.0),
            Align2::LEFT_CENTER,
            "pull-out",
            pens.second,
        ),
        label(
            fonts,
            plot.right_bottom(),
            Align2::RIGHT_TOP,
            "φ: 0 to 180°",
            pens.ink,
        ),
        label(fonts, plot.left_top(), Align2::RIGHT_TOP, "T", pens.ink),
    ]
}

/// A magnet block with straight field lines over its middle and lines bulging out at its ends.
fn end_fringing(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let block = Rect::from_center_size(
        pos2(r.center().x, r.bottom() - 22.0),
        vec2(r.width() * 0.6, 14.0),
    );
    let mut shapes = vec![Shape::rect_filled(block, 1.0, pens.north)];
    let lines = 7;
    for i in 0..lines {
        let t = i as f32 / (lines - 1) as f32;
        let x = block.left() + t * block.width();
        // The outer lines lean out: the more, the nearer the end.
        let lean = (t - 0.5) * 2.0;
        let bulge = lean * lean.abs() * 40.0;
        let points: Vec<Pos2> = (0..=10)
            .map(|k| {
                let s = k as f32 / 10.0;
                pos2(
                    x + bulge * s * s,
                    block.top() - s * (block.top() - r.top() - 4.0),
                )
            })
            .collect();
        shapes.push(Shape::line(points, Stroke::new(1.0, pens.first)));
    }
    shapes.push(label(
        fonts,
        pos2(block.center().x, block.bottom() + 2.0),
        Align2::CENTER_TOP,
        "magnet length L",
        pens.ink,
    ));
    shapes.push(label(
        fonts,
        pos2(r.right(), r.top()),
        Align2::RIGHT_TOP,
        "fringing at the ends",
        pens.second,
    ));
    shapes
}

/// The intrinsic curve J(H) in the second quadrant with its knee, and three load lines.
fn demag_knee(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let plot = Rect::from_min_max(
        pos2(r.left(), r.top() + 4.0),
        pos2(r.right() - 14.0, r.bottom() - 14.0),
    );
    // t = 0 at H = -Hcj (left), t = 1 at H = 0 (right); J flat near Br, falling past the knee.
    let knee_t = 0.1;
    let intrinsic = |t: f32| {
        if t >= knee_t {
            1.0 - 0.05 * (1.0 - t)
        } else {
            let s = t / knee_t;
            (1.0 - 0.05 * (1.0 - knee_t)) * s.sqrt()
        }
    };
    let points = curve(plot, 0.0, 1.1, 201, intrinsic);
    let knee = curve(plot, 0.0, 1.1, 2, |_| intrinsic(knee_t))[0];
    let knee = pos2(plot.left() + knee_t * plot.width(), knee.y);
    let origin = plot.right_bottom();
    let mut shapes = vec![
        Shape::line_segment(
            [plot.left_bottom(), plot.right_bottom()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line_segment(
            [plot.right_bottom(), plot.right_top()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line(points, Stroke::new(1.5, pens.first)),
        Shape::circle_filled(knee, 3.5, pens.second),
        label(
            fonts,
            knee + vec2(6.0, -4.0),
            Align2::LEFT_BOTTOM,
            "knee",
            pens.second,
        ),
    ];
    // Load lines from the origin: steeper is a higher permeance coefficient.
    for slope in [0.4f32, 0.8, 1.6] {
        let dx = (plot.height() / slope).min(plot.width());
        let end = pos2(origin.x - dx, origin.y - dx * slope);
        shapes.extend(Shape::dashed_line(
            &[origin, end],
            Stroke::new(1.0, pens.weak),
            4.0,
            3.0,
        ));
    }
    shapes.push(label(
        fonts,
        plot.left_bottom(),
        Align2::LEFT_TOP,
        "H (reverse field)",
        pens.ink,
    ));
    shapes.push(label(
        fonts,
        plot.right_top(),
        Align2::LEFT_TOP,
        "J",
        pens.ink,
    ));
    shapes.push(label(
        fonts,
        pos2(plot.center().x, plot.bottom() - 4.0),
        Align2::CENTER_BOTTOM,
        "load lines",
        pens.weak,
    ));
    shapes
}

/// First-order heating toward a steady temperature over five time constants, τ at 63 %.
fn heating_curve(fonts: &Fonts, r: Rect, pens: &Pens) -> Vec<Shape> {
    let plot = Rect::from_min_max(
        pos2(r.left() + 14.0, r.top() + 4.0),
        pos2(r.right(), r.bottom() - 14.0),
    );
    let span = 5.0;
    let rise = |t: f32| 1.0 - (-t * span).exp();
    let tau = pos2(
        plot.left() + plot.width() / span,
        plot.bottom() - rise(1.0 / span) / 1.1 * plot.height(),
    );
    let steady = plot.bottom() - plot.height() / 1.1;
    let mut shapes = vec![
        Shape::line_segment(
            [plot.left_bottom(), plot.right_bottom()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line_segment(
            [plot.left_bottom(), plot.left_top()],
            Stroke::new(1.0, pens.weak),
        ),
        Shape::line(
            curve(plot, 0.0, 1.1, 121, rise),
            Stroke::new(1.5, pens.first),
        ),
        Shape::circle_filled(tau, 3.5, pens.second),
        label(
            fonts,
            tau + vec2(6.0, 2.0),
            Align2::LEFT_TOP,
            "τ: 63 %",
            pens.second,
        ),
        label(
            fonts,
            plot.right_bottom(),
            Align2::RIGHT_TOP,
            "time slipping",
            pens.ink,
        ),
    ];
    shapes.extend(Shape::dashed_line(
        &[pos2(plot.left(), steady), pos2(plot.right(), steady)],
        Stroke::new(1.0, pens.weak),
        4.0,
        3.0,
    ));
    shapes.push(label(
        fonts,
        pos2(plot.right(), steady - 2.0),
        Align2::RIGHT_BOTTOM,
        "steady temperature",
        pens.weak,
    ));
    shapes
}

/// Allocates [`DIAGRAM_SIZE`] and paints the diagram `kind` there.
pub fn diagram_ui(ui: &mut egui::Ui, kind: Diagram) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(DIAGRAM_SIZE, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
    let shapes = ui.fonts(|f| diagram_shapes(f, kind, rect, ui.visuals()));
    painter.extend(shapes);
    response
}

#[cfg(test)]
mod tests {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
//! marks are another equation's, the term list's swatches fade ([`SWATCH_DIM`]). A long trail
//! shows its last [`SHOWN_CRUMBS`] crumbs after one [`ELIDED`].
//!
//! Assumptions (spec Addendum A3 "Traceability"): an assumption term is tagged as one in the
//! term list, with the changed-from-default dot when it differs from its workbook default; a
//! result some modified assumption flows into is tagged too, and the open equation says which
````

with:

````rust
//! marks are another equation's, the term list's swatches fade ([`SWATCH_DIM`]). A long trail
//! shows its last [`SHOWN_CRUMBS`] crumbs after one [`ELIDED`].
//!
//! Teaching notes (spec Addendum A4): the Explain toggle, off by default, shows under the open
//! equation the reviewed note that explains it ([`notes::note_for`]: only notes a physics
//! reviewer has checked), with its watch-out line and its diagram ([`crate::gui::diagrams`]),
//! so it follows what the user investigates. "Start here" walks the suggested order
//! ([`notes::START_HERE`]): each step opens its equation with Explain on. A warning's link
//! opens its note on its own ([`Explorer::open_note`]; five of the six warning notes explain
//! no single equation), with links to the equations it does explain. Every note the panel
//! names or draws by id goes through [`reviewed_note`], so a note sent back to Draft never
//! shows.
//!
//! Assumptions (spec Addendum A3 "Traceability"): an assumption term is tagged as one in the
//! term list, with the changed-from-default dot when it differs from its workbook default; a
//! result some modified assumption flows into is tagged too, and the open equation says which
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
use egui::{Color32, Sense, Vec2};

use crate::engine::explain::markup::{Expr, Symbol};
use crate::engine::explain::notes::covers;
use crate::engine::explain::{Design, Equation, TermKind, TermStyle};
use crate::engine::meta::{ResultSet, Value};
use crate::gui::dashboard::result_info;
use crate::gui::format::{format_value, with_unit};
use crate::gui::input_ui::CHANGED_DOT;
use crate::gui::inputs::InputCatalogue;
````

with:

````rust
use egui::{Color32, Sense, Vec2};

use crate::engine::explain::markup::{Expr, Symbol};
use crate::engine::explain::notes::{self, Note, Review, START_HERE, covers};
use crate::engine::explain::{Design, Equation, TermKind, TermStyle};
use crate::engine::meta::{ResultSet, Value};
use crate::gui::dashboard::result_info;
use crate::gui::diagrams::diagram_ui;
use crate::gui::format::{format_value, with_unit};
use crate::gui::input_ui::CHANGED_DOT;
use crate::gui::inputs::InputCatalogue;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust

/// The term list's tag of an assumption.
pub const ASSUMPTION_TAG: &str = "assumption: click to find its row";

/// The text size of the open equation [points].
pub const PANEL_SIZE: f32 = 22.0;
````

with:

````rust

/// The term list's tag of an assumption.
pub const ASSUMPTION_TAG: &str = "assumption: click to find its row";

/// The toggle of the teaching note under the open equation.
pub const EXPLAIN: &str = "Explain";

/// The button that starts the suggested reading order.
pub const START_HERE_BUTTON: &str = "Start here";

/// The start-here steps' buttons.
pub const PREVIOUS: &str = "Previous";
pub const NEXT: &str = "Next";
pub const STOP: &str = "Stop";

/// What Explain shows for an equation without a reviewed note.
pub const NO_NOTE: &str = "No teaching note for this equation.";

/// The start of a note's watch-out line.
pub const WATCH_OUT: &str = "Watch out";

/// The start of a note's sources line.
pub const SOURCES: &str = "Sources";

/// The start of the links under a note opened on its own.
pub const EXPLAINS: &str = "Explains";

/// The note `id` if a physics reviewer has checked it (spec A4 "Accuracy gate"), as
/// [`notes::note_for`] gives an equation's: every note the panel or the dashboard shows by id.
pub fn reviewed_note(id: &str) -> Option<&'static Note> {
    notes::note(id).filter(|n| matches!(n.review, Review::Reviewed { .. }))
}

/// The start of the start-here bar.
pub fn start_here_text(step: usize) -> String {
    let title = reviewed_note(START_HERE[step].0).map_or("", |n| n.title);
    format!(
        "{START_HERE_BUTTON} {} of {}: {title}",
        step + 1,
        START_HERE.len()
    )
}

/// The text size of the open equation [points].
pub const PANEL_SIZE: f32 = 22.0;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
    hovered: Option<String>,
    /// The input row a leaf term highlights.
    focus: Option<Focus>,
}

impl Explorer {
````

with:

````rust
    hovered: Option<String>,
    /// The input row a leaf term highlights.
    focus: Option<Focus>,
    /// Whether the teaching note of the open equation is shown (off by default: spec A4).
    pub explain: bool,
    /// The start-here step shown, if the user is walking the suggested order.
    start_here: Option<usize>,
    /// A teaching note opened on its own (a warning's link), by id.
    note: Option<&'static str>,
}

impl Explorer {
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
        self.trail = vec![path.to_owned()];
        self.open = true;
        self.focus = None;
    }

    /// Follows a term or a "used by" link to `path`: one more step on the trail.
````

with:

````rust
        self.trail = vec![path.to_owned()];
        self.open = true;
        self.focus = None;
        self.note = None;
        self.start_here = None;
    }

    /// Opens the teaching note `id` on its own (a warning's link).
    pub fn open_note(&mut self, id: &'static str) {
        self.note = Some(id);
        self.trail.clear();
        self.open = true;
        self.focus = None;
        self.start_here = None;
    }

    /// The note opened on its own, if any.
    pub fn note(&self) -> Option<&'static str> {
        self.note
    }

    /// Opens start-here step `step` (its equation, Explain on); a step past the last one
    /// changes nothing.
    pub fn start(&mut self, step: usize) {
        if let Some(&(_, path)) = START_HERE.get(step) {
            self.trail = vec![path.to_owned()];
            self.open = true;
            self.focus = None;
            self.note = None;
            self.explain = true;
            self.start_here = Some(step);
        }
    }

    /// The start-here step shown, if any.
    pub fn start_here(&self) -> Option<usize> {
        self.start_here
    }

    /// Leaves the start-here walk (the equation stays open).
    pub fn stop(&mut self) {
        self.start_here = None;
    }

    /// Follows a term or a "used by" link to `path`: one more step on the trail.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        ui.strong(EQUATION_PANEL);
        ui.separator();
        let mut back = None;
        let last = explorer.trail.len().saturating_sub(1);
        let first = explorer.trail.len().saturating_sub(SHOWN_CRUMBS);
````

with:

````rust
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        ui.strong(EQUATION_PANEL);
        ui.separator();
        if let Some(note) = explorer.note.and_then(reviewed_note) {
            let _ = ui.selectable_label(true, glyph_safe(note.title));
        }
        let mut back = None;
        let last = explorer.trail.len().saturating_sub(1);
        let first = explorer.trail.len().saturating_sub(SHOWN_CRUMBS);
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
            if ui.button(CLOSE).clicked() {
                explorer.open = false;
            }
        });
    });
    ui.separator();
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_equation_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let Some(path) = explorer.current().map(str::to_owned) else {
                ui.weak(EMPTY_TEXT);
                return;
````

with:

````rust
            if ui.button(CLOSE).clicked() {
                explorer.open = false;
            }
            if ui
                .button(START_HERE_BUTTON)
                .on_hover_text("A suggested order: torque, back iron, temperature, demagnetization, slip heating, clamps")
                .clicked()
            {
                explorer.start(0);
            }
            ui.checkbox(&mut explorer.explain, EXPLAIN)
                .on_hover_text("The teaching note of the open equation");
        });
    });
    if let Some(step) = explorer.start_here {
        ui.horizontal_wrapped(|ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.weak(start_here_text(step));
            if ui
                .add_enabled(step > 0, egui::Button::new(PREVIOUS))
                .clicked()
            {
                explorer.start(step - 1);
            }
            if ui
                .add_enabled(step + 1 < START_HERE.len(), egui::Button::new(NEXT))
                .clicked()
            {
                explorer.start(step + 1);
            }
            if ui.button(STOP).clicked() {
                explorer.stop();
            }
        });
    }
    ui.separator();
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_equation_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if let Some(note) = explorer.note.and_then(reviewed_note) {
                if let Some(path) = note_alone_ui(ui, note) {
                    explorer.open_path(&path);
                }
                return;
            }
            let Some(path) = explorer.current().map(str::to_owned) else {
                ui.weak(EMPTY_TEXT);
                return;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
            // The marks on screen are another equation's while its value is hovered.
            let others_marked = explorer.hovered().is_some_and(|h| h != path);
            let follow = match registry().equation_for(&path) {
                Some(eq) => equation_body(ui, eq, &terms, others_marked),
                None => {
                    no_equation_body(ui, &path, results);
                    None
                }
            };
````

with:

````rust
            // The marks on screen are another equation's while its value is hovered.
            let others_marked = explorer.hovered().is_some_and(|h| h != path);
            let follow = match registry().equation_for(&path) {
                Some(eq) => equation_body(ui, eq, &terms, others_marked, explorer.explain),
                None => {
                    no_equation_body(ui, &path, results);
                    if explorer.explain {
                        ui.weak(NO_NOTE);
                    }
                    None
                }
            };
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
        });
}

/// The open equation: its label and cell, the equation large (a term clicked is followed), the
/// value, the corrections it embodies, and the term list, its swatches faded while
/// `others_marked`. Returns the term clicked.
fn equation_body(
    ui: &mut egui::Ui,
    eq: &Equation,
    terms: &Design<'_>,
    others_marked: bool,
) -> Option<String> {
    let mut clicked = None;
    ui.weak(format!(
````

with:

````rust
        });
}

/// A teaching note: its title, its sentences, its watch-out line, its diagram and its sources
/// (every text drawn through `glyph_safe`).
pub fn note_ui(ui: &mut egui::Ui, note: &Note) {
    ui.strong(glyph_safe(note.title));
    for sentence in note.sentences {
        ui.add(egui::Label::new(glyph_safe(sentence)).wrap());
    }
    if let Some(watch_out) = note.watch_out {
        ui.add(
            egui::Label::new(
                egui::RichText::new(format!("{WATCH_OUT}: {}", glyph_safe(watch_out)))
                    .color(ui.visuals().warn_fg_color),
            )
            .wrap(),
        );
    }
    if let Some(diagram) = note.diagram {
        diagram_ui(ui, diagram);
    }
    ui.add(
        egui::Label::new(
            egui::RichText::new(format!(
                "{SOURCES}: {}",
                glyph_safe(&note.sources.join("; "))
            ))
            .small()
            .weak(),
        )
        .wrap(),
    );
}

/// A note opened on its own: links to the equations it explains (a family's template is left
/// out), then the note. Returns the equation clicked.
fn note_alone_ui(ui: &mut egui::Ui, note: &Note) -> Option<String> {
    let paths: Vec<&str> = note
        .equations
        .iter()
        .copied()
        .filter(|p| !p.contains('#') && registry().equation_for(p).is_some())
        .collect();
    let mut clicked = None;
    if !paths.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.strong(EXPLAINS);
            for path in paths {
                if ui
                    .link(crumb(path))
                    .on_hover_text(term_label(path))
                    .clicked()
                {
                    clicked = Some(path.to_owned());
                }
            }
        });
    }
    note_ui(ui, note);
    clicked
}

/// The open equation: its label and cell, the equation large (a term clicked is followed), the
/// value, the corrections it embodies, with `explain` its teaching note, then the term list,
/// its swatches faded while `others_marked`. Returns the term clicked.
fn equation_body(
    ui: &mut egui::Ui,
    eq: &Equation,
    terms: &Design<'_>,
    others_marked: bool,
    explain: bool,
) -> Option<String> {
    let mut clicked = None;
    ui.weak(format!(
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
    if !corrections.is_empty() {
        let ids: Vec<String> = corrections.iter().map(ToString::to_string).collect();
        ui.weak(format!("{CORRECTIONS}: {}", ids.join(", ")));
    }
    ui.add_space(6.0);
    ui.strong(TERMS);
````

with:

````rust
    if !corrections.is_empty() {
        let ids: Vec<String> = corrections.iter().map(ToString::to_string).collect();
        ui.weak(format!("{CORRECTIONS}: {}", ids.join(", ")));
    }
    if explain {
        ui.separator();
        match notes::note_for(&eq.target) {
            Some(note) => note_ui(ui, note),
            None => {
                ui.weak(NO_NOTE);
            }
        }
        ui.separator();
    }
    ui.add_space(6.0);
    ui.strong(TERMS);
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib gui::diagrams 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 2 passed; 0 failed` for `gui::diagrams`, then `test result: ok. 415 passed; 0 failed` for the whole library.

- [ ] **Step 5: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m43 add magcoupling-rs/src/gui/diagrams.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/gui/explorer.rs magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m43 commit -F - <<'EOF'
feat(magcoupling-rs): teaching notes: Explain, start here and the six diagrams

The Equation panel's Explain toggle (off by default) shows the open
equation's reviewed note under its value (decision M43-10): sentences,
watch-out line, diagram and sources. "Start here" walks notes::START_HERE,
each step opening its equation with Explain on. A note can be opened on its
own, with links to the equations it explains; every note shown by id goes
through explorer::reviewed_note. gui::diagrams paints the six
diagram kinds with egui's painter: the square wave and its harmonics, flux
with and without back iron, torque against angle, end fringing, the
demagnetization knee with load lines, first-order heating.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m43 status --short
```

Expected: one new commit on `magcoupling/m4-3`; `status --short` prints nothing.

---

### Task 6: Material, part and grade pickers; material warnings linked to their notes

**Model:** `sonnet` (the plan gives the exact code: transcription plus running tests; CLAUDE.md section 5). Escalate a retry to the session model.

Spec A5 and A6 (decisions M43-4, M43-8). `gui/pickers.rs` adds the library to the input rows without replacing them: each choice of the three part-material selectors shows its library row on hover (`choice_hover`: name, condition, ferromagnetic, μr, Bsat, conductivity, density, expansion, modulus, yield, specific heat, the wall check's design flux density, plating; "not sourced" where the data file has none) and a small line under the row sums up the pick; under each magnet part's text field a "Pick a part" drop-down lists "Custom dimensions (manual)" (a blank part) and the 15 library parts, each with vendor, shape, dimensions, grade, remanence, rating, coating and magnetization on hover with every correction on (E3's remanence, E19's rating and grade), and a line under it sums up the part in use; under each grade's text field a "Pick a grade" drop-down lists "Blank" and the 17 grades with their table rows. `input_row` calls `picker_ui` under every row (nothing for other inputs) and `choice_hover` per selector choice; the part and grade pickers read the row's text through `input_ui::current_text`, now `pub(crate)`. The dashboard lists the warnings that fire at its top under "Material warnings" (`warning_lines`): a badge and the text in the severity's colour (`severity_level`: a warning red, a caution amber), then "Why: <note title>" (the title through `glyph_safe`, as the panel draws it) for a reviewed note (`warning_note`, which calls Task 5's `reviewed_note`), which asks the panel to open the note alone (`Readouts::open_note`, handled in `Explorer::end_frame`). The tests pin the pickers' selectors and choices, the material texts, the parts and grades with the corrections, every picker text against the fonts, the warnings for 304 stainless and their notes; in the panel the part picker (custom, then a library part, the results recomputed), a material choice's summary and the warnings it fires, a material code outside the choices shown as "99 (not a choice)" with no summary and kept through idle frames, and a warning's colours and link to its note and on to the equation it explains (Review Focus 6); the pickers' glyph test goes through `assert_glyphs`.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/pickers.rs`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/mod.rs` (`pub mod pickers;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/input_ui.rs` (`picker_ui` under the row, `choice_hover` on the choices)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs` (the warnings; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs` (`end_frame` opens a note asked for)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs` (tests)

**Interfaces:**
- Consumes: Task 2's `Readouts::open_note`, `ReadoutEvents::note`; Task 5's `Explorer::open_note`, `note`, `reviewed_note`; Task 1's `glyph_safe`, `test_support::assert_glyphs`; `engine::material_library::{BACK_IRON_CHOICES, SLEEVE_LINER_CHOICES, CAP_HOUSING_CHOICES, Material, Sourced, chosen}`, `engine::library::{self, MAGNET_LIBRARY, MagnetSpec}`, `engine::grades::{GRADES, Grade, GradeFamily, grade}`, `engine::deviations::Deviations::ALL`, `engine::warnings::{Severity, WARNING_RULES, WarningRule}`, `engine::explain::notes`; `input_ui::{RowEdit, current_text}`, `format::format_value`.
- Produces (`magcoupling::gui::pickers`): `pub const MATERIAL_PICKERS: [(&str, &[(i64, &str)]); 3]`, `PART_PATHS: [&str; 2]`, `GRADE_PATHS: [&str; 2]`, `CUSTOM, BLANK_GRADE, PICK_PART, PICK_GRADE: &str`; `pub fn material_of(&str, i64) -> Option<&'static Material>`, `material_summary(&Material) -> String`, `material_properties(&Material) -> String`, `part_label(&MagnetSpec) -> String`, `part_properties(&MagnetSpec) -> String`, `grade_label(&Grade) -> String`, `grade_properties(&Grade) -> String`, `choice_hover(&str, i64) -> Option<String>`, `picker_ui(&mut egui::Ui, &str, &Value) -> Option<RowEdit>`. (`gui::dashboard`): `pub const WARNINGS_HEADING, WHY: &str`; `pub const fn severity_level(Severity) -> Level`; `pub fn warning_lines(&DesignResults) -> Vec<(&'static WarningRule, String)>`; `pub fn warning_note(&WarningRule) -> Option<&'static notes::Note>`. Changes: `input_ui::current_text(&Value) -> &str` becomes `pub(crate)` (the pickers' private copy of it is not written).

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

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
    fn the_warnings_that_fire_are_listed_with_their_reviewed_notes() {
        assert!(warning_lines(&compute_all(&DesignInputs::default())).is_empty());
        // 304 stainless back iron: an open circuit, and its expansion against the magnets'.
        let results = compute_all(&design(|i| i.materials.parts.back_iron = 7));
        let ids: Vec<&str> = warning_lines(&results)
            .iter()
            .map(|(rule, _)| rule.id)
            .collect();
        assert_eq!(
            ids,
            ["non_ferromagnetic_back_iron", "cte_mismatch_with_magnets"]
        );
        let (rule, text) = &warning_lines(&results)[0];
        assert_eq!(text, rule.text);
        assert_eq!(severity_level(rule.severity), Level::Bad);
        assert_eq!(severity_level(Severity::Caution), Level::Caution);
        // Every rule's note has passed the accuracy gate, so every warning links to it.
        for rule in &WARNING_RULES {
            assert_eq!(warning_note(rule).map(|n| n.id), Some(rule.note_id));
        }
    }

    #[test]
    fn badge_colours_follow_the_theme() {
        let dark = egui::Visuals::dark();
        let light = egui::Visuals::light();
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub mod input_ui;
pub mod inputs;
mod panel;
pub mod plots;
pub mod readouts;
pub mod results_table;
````

with:

````rust
pub mod input_ui;
pub mod inputs;
mod panel;
pub mod pickers;
pub mod plots;
pub mod readouts;
pub mod results_table;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        assert!(texts.iter().any(|t| t.starts_with(ASSUMPTIONS_MODIFIED)));
        let catalogue = InputCatalogue::get();
        texts.extend(catalogue.all().map(crate::gui::inputs::input_tooltip));
        let results = harness.panel.results().clone();
        texts.extend(
            crate::gui::dashboard::dashboard_lines(&results)
````

with:

````rust
        assert!(texts.iter().any(|t| t.starts_with(ASSUMPTIONS_MODIFIED)));
        let catalogue = InputCatalogue::get();
        texts.extend(catalogue.all().map(crate::gui::inputs::input_tooltip));
        // The material warnings the dashboard can show.
        texts.extend(
            crate::engine::warnings::WARNING_RULES
                .iter()
                .map(|rule| rule.text.to_owned()),
        );
        let results = harness.panel.results().clone();
        texts.extend(
            crate::gui::dashboard::dashboard_lines(&results)
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
        let ctx = egui::Context::default();
````

with:

````rust
    }

    #[test]
    fn the_part_picker_sets_a_library_part_or_custom_dimensions() {
        use crate::gui::pickers::{CUSTOM, part_label};
        let mut harness = Harness::new();
        let b842sh = part_label(crate::engine::library::lookup("B842SH").unwrap());
        // The inner part's picker (the Key design row, drawn first).
        harness.click_text(&b842sh);
        harness.click_text(CUSTOM);
        assert_eq!(harness.panel.inputs().coupling.magnets.part_inner, "");
        let output = harness.frame(Vec::new());
        assert!(count(&output, CUSTOM) >= 1);
        // Back to a library part, from the same picker.
        harness.click_text(CUSTOM);
        let b842 = part_label(crate::engine::library::lookup("B842").unwrap());
        harness.click_text(&b842);
        assert_eq!(harness.panel.inputs().coupling.magnets.part_inner, "B842");
        let mut expected = DesignInputs::default();
        expected.coupling.magnets.part_inner = "B842".to_owned();
        assert_eq!(harness.panel.results(), &compute_all(&expected));
    }

    #[test]
    fn a_material_choice_sums_up_its_properties_and_fires_its_warnings() {
        use crate::gui::dashboard::WARNINGS_HEADING;
        use crate::gui::pickers::{material_of, material_summary};
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text("Materials");
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        let steel = material_of("materials.parts.back_iron", 1).unwrap();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &material_summary(steel)), 1);
        assert_eq!(count(&output, WARNINGS_HEADING), 0);
        harness.click_text("4140 annealed");
        harness.click_text("304 stainless (non-magnetic)");
        assert_eq!(harness.panel.inputs().materials.parts.back_iron, 7);
        let output = harness.frame(Vec::new());
        let stainless = material_of("materials.parts.back_iron", 7).unwrap();
        assert_eq!(count(&output, &material_summary(stainless)), 1);
        assert_eq!(count(&output, WARNINGS_HEADING), 1);
    }

    #[test]
    fn a_material_code_outside_the_choices_is_shown_and_kept() {
        // A design file's back iron code 99 is no choice: the drop-down says so, the row sums
        // up no material, and idle frames write nothing back.
        use crate::gui::pickers::{MATERIAL_PICKERS, material_of, material_summary};
        let summaries: Vec<String> = MATERIAL_PICKERS
            .iter()
            .flat_map(|(path, choices)| {
                choices
                    .iter()
                    .filter_map(|&(code, _)| material_of(path, code))
            })
            .map(material_summary)
            .collect();
        let drawn_summaries = |output: &egui::FullOutput| {
            drawn_texts(output)
                .iter()
                .filter(|t| summaries.contains(t))
                .count()
        };
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text("Materials");
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        let output = harness.frame(Vec::new());
        assert_eq!(drawn_summaries(&output), MATERIAL_PICKERS.len());
        harness.panel.inputs.materials.parts.back_iron = 99;
        let mut expected = DesignInputs::default();
        expected.materials.parts.back_iron = 99;
        let mut output = harness.frame(Vec::new());
        for _ in 0..2 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(count(&output, "99 (not a choice)"), 1);
        assert_eq!(drawn_summaries(&output), MATERIAL_PICKERS.len() - 1);
        assert_eq!(harness.panel.inputs(), &expected);
    }

    #[test]
    fn a_warning_shows_in_its_colour_and_links_to_its_note() {
        use crate::engine::warnings::WARNING_RULES;
        use crate::gui::dashboard::WHY;
        let mut harness = Harness::new();
        harness.panel.inputs.materials.parts.back_iron = 7;
        let output = harness.frame(Vec::new());
        let rule = &WARNING_RULES[0];
        assert_eq!(
            crate::gui::test_support::text_color(&output, rule.text),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        let caution = &WARNING_RULES[5];
        assert_eq!(
            crate::gui::test_support::text_color(&output, caution.text),
            Some(Level::Caution.color(&harness.ctx.style().visuals))
        );
        let note = crate::engine::explain::notes::note(rule.note_id).unwrap();
        harness.click_text(&format!(
            "{WHY}: {}",
            crate::gui::typeset::glyph_safe(note.title)
        ));
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.note(), Some(rule.note_id));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, note.sentences[0]), 1);
        assert!(
            diagram_points(&output) == 0,
            "the flux-path diagram marks no point"
        );
        // Its one equation is a link: the circuit in effect, opened like a value.
        harness.click_text("circuit");
        assert_eq!(harness.panel.explorer.note(), None);
        assert_eq!(
            harness.panel.explorer.trail(),
            ["materials.circuit_backiron"]
        );
    }

    #[test]
    fn the_panel_works_inside_an_egui_window() {
        // M5 shows the panel in a window of the linkage app.
        let ctx = egui::Context::default();
````

Create `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/pickers.rs`:

````rust
//! The material, magnet-part and grade pickers (spec Addendum A5 "Per-part material pickers
//! backed by a small library", A6 "Two tables replace the single part list ... Custom
//! dimensions stay available: pick any grade with manual dimensions").
//!
//! The rows stay the inputs' own (M4-1's: a part or a grade is a text the engine looks up, a
//! material a selector code), so a design file or a typed name keeps working; the pickers add
//! what the library knows (decision M43-4):
//!
//! - **Materials** (`materials.parts.*`): each choice of the drop-down shows, on hover, every
//!   property the library holds for it ([`material_properties`]), sourced values only, with
//!   "not sourced" where the data file has none; a line under the row sums up the material
//!   picked ([`material_summary`]).
//! - **Magnet parts** (`coupling.magnets.part_*`): a "pick" drop-down under the text field lists
//!   "Custom dimensions (manual)" (a blank part: the manual dimensions) and the 15 library
//!   parts, each with vendor, shape, dimensions, grade, remanence, rating, coating and
//!   magnetization on hover ([`part_properties`], every approved correction on: E3's N42SH
//!   remanence, E19's vendor rating and grade); a line under it sums up the part in use.
//! - **Grades** (`coupling.magnets.grade_*`): a drop-down of "Blank" and the 17 grades, each
//!   with its table row on hover ([`grade_properties`]); the grade applies to a ring with manual
//!   dimensions, which "Custom dimensions" gives.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::inputs::InputCatalogue;

    #[test]
    fn the_material_pickers_are_the_three_selectors_and_their_library_choices() {
        for (path, choices) in MATERIAL_PICKERS {
            let entry = InputCatalogue::get()
                .entry(path)
                .expect("a material selector");
            let codes: Vec<i64> = entry.meta.choices.iter().map(|(c, _)| *c).collect();
            let library: Vec<i64> = choices.iter().map(|(c, _)| *c).collect();
            assert_eq!(codes, library, "{path}");
            for &(code, id) in choices.iter() {
                let m = material_of(path, code).unwrap();
                assert_eq!(m.id, id);
                assert!(choice_hover(path, code).unwrap().starts_with(m.name));
            }
            assert_eq!(material_of(path, 0), None);
        }
        assert_eq!(material_of("coupling.backiron", 1), None);
    }

    #[test]
    fn a_material_s_summary_and_properties_say_what_the_library_holds() {
        let steel = material_of("materials.parts.back_iron", 1).unwrap();
        assert_eq!(
            material_summary(steel),
            "ferromagnetic; conductivity 4.330 MS/m; density 7.850 g/cm³; expansion 12.20e-6/K; modulus 205.0 GPa"
        );
        let props = material_properties(steel);
        assert!(
            props.contains("Saturation flux density: not sourced"),
            "{props}"
        );
        assert!(
            props.contains("Wall check design flux density: 1.500 T"),
            "{props}"
        );
        assert!(props.contains("needs plating"), "{props}");
        let al = material_of("materials.parts.back_iron", 8).unwrap();
        assert!(material_summary(al).starts_with("non-ferromagnetic"));
    }

    #[test]
    fn parts_and_grades_read_with_every_correction_on() {
        let b842sh = library::lookup("B842SH").unwrap();
        assert_eq!(
            part_label(b842sh),
            "B842SH: K&J block 12.70 × 6.350 × 3.170 mm, N42SH"
        );
        // E3: the N42SH parts take the grade's 1.30 T, not the workbook row's 1.29 T.
        assert!(part_properties(b842sh).contains("Remanence at 20 °C: 1.300 T"));
        let y30 = grade("Y30").unwrap();
        assert!(grade_properties(y30).contains("demagnetization risk is at the cold end"));
        assert!(grade_label(grade("N42SH").unwrap()).starts_with("N42SH: Br 1.300 T"));
    }

    #[test]
    fn every_picker_text_has_glyphs_in_the_default_fonts() {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let mut texts = vec![CUSTOM.to_owned(), BLANK_GRADE.to_owned()];
        for (path, choices) in MATERIAL_PICKERS {
            for &(code, _) in choices {
                let m = material_of(path, code).unwrap();
                texts.push(material_summary(m));
                texts.push(material_properties(m));
            }
        }
        for spec in &MAGNET_LIBRARY {
            texts.push(part_properties(spec));
        }
        for g in &GRADES {
            texts.push(grade_label(g));
            texts.push(grade_properties(g));
        }
        for text in &texts {
            crate::gui::test_support::assert_glyphs(&ctx, text, "a picker");
        }
    }
}
````

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "^error" | tail -1
```

Expected: FAIL to compile. The last line is `` error: could not compile `magcoupling-rs` (lib test) due to 41 previous errors; 1 warning emitted ``; the errors above it come from the tests reaching for what Step 3 adds, among them `` error[E0432]: unresolved imports `crate::gui::pickers::CUSTOM`, `crate::gui::pickers::part_label` ``, `` error[E0432]: unresolved import `crate::gui::dashboard::WARNINGS_HEADING` ``, `` error[E0432]: unresolved imports `crate::gui::pickers::material_of`, `crate::gui::pickers::material_summary` ``, `` error[E0432]: unresolved imports `crate::gui::pickers::MATERIAL_PICKERS`, `crate::gui::pickers::material_of`, `crate::gui::pickers::material_summary` ``, `` error[E0432]: unresolved import `crate::gui::dashboard::WHY` ``.

- [ ] **Step 3: Write the implementation**

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
//!   rows that read the workbook's stored 3D fields ([`STORED_3D_ROWS`]) carry the label
//!   "3D values from the workbook" instead of a "3D updating" badge.
//!
//! [`result_tooltip`] is the hover text of every displayed result, keyed by result path: the
//! text every readout hands to [`crate::gui::readouts::Readouts::show`], which adds the
//! value's equation (plan M4-3). The dashboard builds a row's text only while it is hovered
````

with:

````rust
//!   rows that read the workbook's stored 3D fields ([`STORED_3D_ROWS`]) carry the label
//!   "3D values from the workbook" instead of a "3D updating" badge.
//!
//! Over the rows, the material warnings that fire (spec Addendum A5: "plain language,
//! colour-coded, linked to their teaching note"): each rule's text in its severity's colour (a
//! warning red, a caution amber) with a link that opens its reviewed note in the Equation panel
//! ([`warning_lines`], decision M43-8).
//!
//! [`result_tooltip`] is the hover text of every displayed result, keyed by result path: the
//! text every readout hands to [`crate::gui::readouts::Readouts::show`], which adds the
//! value's equation (plan M4-3). The dashboard builds a row's text only while it is hovered
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::engine::meta::{ResultMeta, ResultSet, Value, result_rows};
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
use crate::gui::format::{format_value, with_unit};
use crate::gui::readouts::Readouts;
use crate::{DesignInputs, DesignResults, compute_all};

/// A badge colour, from a check verdict.
````

with:

````rust
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::engine::explain::notes;
use crate::engine::meta::{ResultMeta, ResultSet, Value, result_rows};
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
use crate::engine::warnings::{Severity, WARNING_RULES, WarningRule};
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
use crate::gui::explorer::reviewed_note;
use crate::gui::format::{format_value, with_unit};
use crate::gui::readouts::Readouts;
use crate::gui::typeset::glyph_safe;
use crate::{DesignInputs, DesignResults, compute_all};

/// A badge colour, from a check verdict.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
            Level::Bad => visuals.error_fg_color,
        }
    }
}

/// The temperature summary's verdict, the badge of the three temperature rows.
````

with:

````rust
            Level::Bad => visuals.error_fg_color,
        }
    }
}

/// The heading of the material warnings.
pub const WARNINGS_HEADING: &str = "Material warnings";

/// The start of a warning's link to its teaching note.
pub const WHY: &str = "Why";

/// A warning's badge colour: a warning (the coupling does not work as designed) red, a caution
/// amber.
pub const fn severity_level(severity: Severity) -> Level {
    match severity {
        Severity::Warning => Level::Bad,
        Severity::Caution => Level::Caution,
    }
}

/// The material warnings that fire for `results`, in the rules' order: each rule and its text.
pub fn warning_lines(results: &DesignResults) -> Vec<(&'static WarningRule, String)> {
    WARNING_RULES
        .iter()
        .filter_map(|rule| match results.get(&format!("warnings.{}", rule.id)) {
            Some(Value::Text(text)) if !text.is_empty() => Some((rule, text)),
            _ => None,
        })
        .collect()
}

/// The reviewed teaching note of a warning rule, if its note has passed the accuracy gate
/// ([`reviewed_note`]).
pub fn warning_note(rule: &WarningRule) -> Option<&'static notes::Note> {
    reviewed_note(rule.note_id)
}

/// The temperature summary's verdict, the badge of the three temperature rows.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
        ui.colored_label(ui.visuals().error_fg_color, banner);
        ui.separator();
    }
    let last_3d = lines.iter().rposition(|line| line.stored_3d);
    for (index, line) in lines.iter().enumerate() {
        let weak = ui.visuals().weak_text_color();
````

with:

````rust
        ui.colored_label(ui.visuals().error_fg_color, banner);
        ui.separator();
    }
    warnings_ui(ui, results, readouts);
    let last_3d = lines.iter().rposition(|line| line.stored_3d);
    for (index, line) in lines.iter().enumerate() {
        let weak = ui.visuals().weak_text_color();
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
        }
        ui.add_space(4.0);
    }
}

/// A badge: a filled circle in the level's colour, or an empty cell.
````

with:

````rust
        }
        ui.add_space(4.0);
    }
}

/// The material warnings that fire: a badge and the text in the severity's colour, then a link
/// to the teaching note (it asks the panel to open the note).
fn warnings_ui(ui: &mut egui::Ui, results: &DesignResults, readouts: &mut Readouts) {
    let lines = warning_lines(results);
    if lines.is_empty() {
        return;
    }
    ui.strong(WARNINGS_HEADING);
    for (rule, text) in lines {
        let level = severity_level(rule.severity);
        ui.horizontal(|ui| {
            badge(ui, Some(level));
            ui.add(
                egui::Label::new(egui::RichText::new(text).color(level.color(ui.visuals()))).wrap(),
            );
        });
        if let Some(note) = warning_note(rule) {
            ui.horizontal(|ui| {
                ui.add_space(16.0);
                if ui
                    .link(format!("{WHY}: {}", glyph_safe(note.title)))
                    .clicked()
                {
                    readouts.open_note(rule.note_id);
                }
            });
        }
    }
    ui.separator();
}

/// A badge: a filled circle in the level's colour, or an empty cell.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/explorer.rs`, replace:

````rust
        }
        if let Some(path) = events.clicked {
            self.open_path(&path);
            ctx.request_repaint();
        }
    }
````

with:

````rust
        }
        if let Some(path) = events.clicked {
            self.open_path(&path);
            ctx.request_repaint();
        }
        if let Some(id) = events.note {
            self.open_note(id);
            ctx.request_repaint();
        }
    }
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/input_ui.rs`, replace:

````rust
//!
//! [`input_row`] draws a row from its [`InputEntry`] and the current value and returns the
//! edit asked for; the panel applies it with `InputSet::set`, so every edit is checked the
//! same way.

use crate::engine::meta::{FieldType, InputMeta, SliderRange, Value};
use crate::gui::format::{format_value, with_unit};
use crate::gui::inputs::{InputEntry, input_tooltip, outside_range, step_decimals, text_hint};

/// The changed-from-default dot.
pub const CHANGED_DOT: &str = "\u{2022}";
````

with:

````rust
//!
//! [`input_row`] draws a row from its [`InputEntry`] and the current value and returns the
//! edit asked for; the panel applies it with `InputSet::set`, so every edit is checked the
//! same way. The material, magnet-part and grade rows add their pickers
//! ([`crate::gui::pickers`]): a material choice's properties on hover, a part or grade picked
//! from the library tables.

use crate::engine::meta::{FieldType, InputMeta, SliderRange, Value};
use crate::gui::format::{format_value, with_unit};
use crate::gui::inputs::{InputEntry, input_tooltip, outside_range, step_decimals, text_hint};
use crate::gui::pickers::{choice_hover, picker_ui};

/// The changed-from-default dot.
pub const CHANGED_DOT: &str = "\u{2022}";
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/input_ui.rs`, replace:

````rust
            }
        });
        let widget = widget(ui, entry, current, seed, &mut edit).on_hover_text(&tooltip);
        // Text inputs only (`text_hint` knows no other path).
        if let Some(hint) = text_hint(&entry.path, current_text(current)) {
            ui.weak(hint);
````

with:

````rust
            }
        });
        let widget = widget(ui, entry, current, seed, &mut edit).on_hover_text(&tooltip);
        if let Some(picked) = picker_ui(ui, &entry.path, current) {
            edit = Some(picked);
        }
        // Text inputs only (`text_hint` knows no other path).
        if let Some(hint) = text_hint(&entry.path, current_text(current)) {
            ui.weak(hint);
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/input_ui.rs`, replace:

````rust
    .inner
}

fn current_text(value: &Value) -> &str {
    match value {
        Value::Text(text) => text,
        _ => "",
````

with:

````rust
    .inner
}

/// The text of a text input (empty for any other value): the row's hint and the part and
/// grade pickers read it.
pub(crate) fn current_text(value: &Value) -> &str {
    match value {
        Value::Text(text) => text,
        _ => "",
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/input_ui.rs`, replace:

````rust
                .selected_text(text)
                .show_ui(ui, |ui| {
                    for &(choice, label) in meta.choices {
                        ui.selectable_value(&mut selected, choice, label);
                    }
                })
                .response;
````

with:

````rust
                .selected_text(text)
                .show_ui(ui, |ui| {
                    for &(choice, label) in meta.choices {
                        let option = ui.selectable_value(&mut selected, choice, label);
                        if let Some(properties) = choice_hover(&entry.path, choice) {
                            option.on_hover_text(properties);
                        }
                    }
                })
                .response;
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/gui/pickers.rs`, replace:

````rust

#[cfg(test)]
mod tests {
````

with:

````rust

use egui::ComboBox;

use crate::engine::deviations::Deviations;
use crate::engine::grades::{GRADES, Grade, GradeFamily, grade};
use crate::engine::library::{self, MAGNET_LIBRARY, MagnetSpec};
use crate::engine::material_library::{
    BACK_IRON_CHOICES, CAP_HOUSING_CHOICES, Material, SLEEVE_LINER_CHOICES, Sourced, chosen,
};
use crate::engine::meta::Value;
use crate::gui::format::format_value;
use crate::gui::input_ui::{RowEdit, current_text};

/// Each material selector and the library choices behind its codes.
pub const MATERIAL_PICKERS: [(&str, &[(i64, &str)]); 3] = [
    ("materials.parts.back_iron", &BACK_IRON_CHOICES),
    ("materials.parts.sleeve_liner", &SLEEVE_LINER_CHOICES),
    ("materials.parts.cap_housing", &CAP_HOUSING_CHOICES),
];

/// The magnet part inputs.
pub const PART_PATHS: [&str; 2] = ["coupling.magnets.part_inner", "coupling.magnets.part_outer"];

/// The magnet grade inputs.
pub const GRADE_PATHS: [&str; 2] = [
    "coupling.magnets.grade_inner",
    "coupling.magnets.grade_outer",
];

/// The part picker's choice of a blank part (the manual dimensions).
pub const CUSTOM: &str = "Custom dimensions (manual)";

/// The grade picker's choice of a blank grade.
pub const BLANK_GRADE: &str = "Blank: the manual Br, no rating";

/// The part picker's label.
pub const PICK_PART: &str = "Pick a part";

/// The grade picker's label.
pub const PICK_GRADE: &str = "Pick a grade";

/// A number to four significant digits.
fn num(x: f64) -> String {
    format_value(&Value::Num(x))
}

/// A sourced property times `scale` with its unit (`unit` empty: none), or "not sourced".
fn sourced(value: &Sourced, scale: f64, unit: &str) -> String {
    value.value.map_or_else(
        || "not sourced".to_owned(),
        |x| format!("{}{unit}", num(x * scale)),
    )
}

/// The library material a selector `code` picks at the material input `path`.
pub fn material_of(path: &str, code: i64) -> Option<&'static Material> {
    MATERIAL_PICKERS
        .iter()
        .find(|(p, _)| *p == path)
        .and_then(|(_, choices)| chosen(choices, code))
}

/// The line under a material picker: what the material is and the properties the engine reads.
pub fn material_summary(m: &Material) -> String {
    format!(
        "{}; conductivity {}; density {}; expansion {}; modulus {}",
        if m.ferromagnetic {
            "ferromagnetic"
        } else {
            "non-ferromagnetic"
        },
        sourced(&m.sigma_S_m, 1e-6, " MS/m"),
        sourced(&m.density_g_cm3, 1.0, " g/cm³"),
        sourced(&m.cte_1e6_per_K, 1.0, "e-6/K"),
        sourced(&m.modulus_GPa, 1.0, " GPa"),
    )
}

/// Every library property of a material, one per line (a material choice's hover text).
pub fn material_properties(m: &Material) -> String {
    let mut lines = vec![
        m.name.to_owned(),
        m.condition.to_owned(),
        format!(
            "Ferromagnetic: {}",
            if m.ferromagnetic { "yes" } else { "no" }
        ),
        format!(
            "Relative permeability: {} (reference only)",
            sourced(&m.mu_r, 1.0, "")
        ),
        format!("Saturation flux density: {}", sourced(&m.bsat_T, 1.0, " T")),
        format!(
            "Electrical conductivity: {}",
            sourced(&m.sigma_S_m, 1e-6, " MS/m")
        ),
        format!("Density: {}", sourced(&m.density_g_cm3, 1.0, " g/cm³")),
        format!(
            "Expansion coefficient: {}",
            sourced(&m.cte_1e6_per_K, 1.0, "e-6/K")
        ),
        format!("Young's modulus: {}", sourced(&m.modulus_GPa, 1.0, " GPa")),
        format!("Yield strength: {}", sourced(&m.yield_MPa, 1.0, " MPa")),
        format!("Specific heat: {}", sourced(&m.cp_J_kgK, 1.0, " J/(kg·K)")),
    ];
    if let Some(b) = m.design_flux_density_T {
        lines.push(format!("Wall check design flux density: {} T", num(b)));
    }
    if m.needs_plating {
        lines.push("Plain or low-alloy steel: needs plating".to_owned());
    }
    lines.join("\n")
}

/// A part's picker label: `B842SH: K&J block 12.7 × 6.35 × 3.17 mm, N42SH`.
pub fn part_label(spec: &MagnetSpec) -> String {
    format!(
        "{}: {} {} {} × {} × {} mm, {}",
        spec.part,
        spec.vendor,
        spec.shape,
        num(spec.length_mm),
        num(spec.width_mm),
        num(spec.thickness_mm),
        library::grade_id(spec, Deviations::ALL)
    )
}

/// What the calculator uses of a library part, with every approved correction on (a part
/// choice's hover text and the line under the part picker).
pub fn part_properties(spec: &MagnetSpec) -> String {
    let mut lines = vec![
        part_label(spec),
        format!(
            "Remanence at 20 °C: {} T; maximum operating temperature: {} °C",
            num(library::br_T(spec, Deviations::ALL)),
            num(library::tmax_C(spec, Deviations::ALL))
        ),
    ];
    if !spec.coating.is_empty() {
        lines.push(format!("Coating: {}", spec.coating));
    }
    if !spec.magnetization.is_empty() {
        lines.push(format!("Magnetization: {}", spec.magnetization));
    }
    lines.join("\n")
}

fn family(f: GradeFamily) -> &'static str {
    match f {
        GradeFamily::NdFeB => "sintered NdFeB",
        GradeFamily::SmCo2_17 => "sintered Sm2Co17",
        GradeFamily::SmCo1_5 => "sintered SmCo5",
        GradeFamily::Ferrite => "hard ferrite",
        GradeFamily::BondedNdFeB => "bonded NdFeB",
    }
}

/// A grade's picker label: `N42SH: Br 1.300 T, Hcj 1592 kA/m, 150.0 °C`.
pub fn grade_label(g: &Grade) -> String {
    format!(
        "{}: Br {} T, Hcj {} kA/m, {} °C",
        g.name,
        num(g.br_T),
        num(g.hcj20_kA_m),
        num(g.tmax_C)
    )
}

/// A grade's table row, one property per line (a grade choice's hover text).
pub fn grade_properties(g: &Grade) -> String {
    let mut lines = vec![
        format!("{} ({})", g.name, family(g.family)),
        format!("Remanence Br at 20 °C: {} T", num(g.br_T)),
        format!(
            "Intrinsic coercivity Hcj at 20 °C: {} kA/m",
            num(g.hcj20_kA_m)
        ),
        format!("Normal coercivity Hcb: {} kA/m", num(g.hcb_kA_m)),
        format!("Maximum energy product: {} kJ/m³", num(g.bhmax_kJ_m3)),
        format!(
            "Br temperature coefficient: {} %/°C",
            num(g.alpha_br_per_C * 100.0)
        ),
        format!(
            "Hcj temperature coefficient: {} %/°C",
            num(g.beta_hcj_per_C * 100.0)
        ),
        format!("Maximum operating temperature: {} °C", num(g.tmax_C)),
        format!("Density: {} g/cm³", num(g.density_g_mm3 * 1000.0)),
    ];
    if let Some(mu) = g.mu_rec {
        lines.push(format!("Recoil permeability: {}", num(mu)));
    }
    if g.beta_hcj_per_C > 0.0 {
        lines.push(
            "Positive Hcj coefficient: the demagnetization risk is at the cold end".to_owned(),
        );
    }
    lines.join("\n")
}

/// The hover text of a selector choice that a picker describes (a material's properties).
pub fn choice_hover(path: &str, code: i64) -> Option<String> {
    material_of(path, code).map(material_properties)
}

/// Draws the picker of the input at `path` under its row (nothing for an input without one) and
/// returns the edit picked.
pub fn picker_ui(ui: &mut egui::Ui, path: &str, current: &Value) -> Option<RowEdit> {
    if MATERIAL_PICKERS.iter().any(|(p, _)| *p == path) {
        if let Value::Int(code) = current
            && let Some(m) = material_of(path, *code)
        {
            ui.add(
                egui::Label::new(egui::RichText::new(material_summary(m)).small().weak()).wrap(),
            );
        }
        return None;
    }
    if PART_PATHS.contains(&path) {
        return part_picker(ui, current_text(current));
    }
    if GRADE_PATHS.contains(&path) {
        return grade_picker(ui, current_text(current));
    }
    None
}

fn part_picker(ui: &mut egui::Ui, current: &str) -> Option<RowEdit> {
    let spec = library::lookup(current);
    let shown = match spec {
        Some(spec) => part_label(spec),
        None if current.is_empty() => CUSTOM.to_owned(),
        None => format!("{current} (not a library part: manual dimensions)"),
    };
    let mut picked = None;
    ui.horizontal(|ui| {
        ui.weak(PICK_PART);
        ComboBox::from_id_salt("part_picker")
            .selected_text(shown)
            .width(ui.available_width())
            .truncate()
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(current.is_empty(), CUSTOM)
                    .on_hover_text("The manual dimensions below, with a grade or a manual Br")
                    .clicked()
                {
                    picked = Some(String::new());
                }
                for spec in &MAGNET_LIBRARY {
                    if ui
                        .selectable_label(current == spec.part, part_label(spec))
                        .on_hover_text(part_properties(spec))
                        .clicked()
                    {
                        picked = Some(spec.part.to_owned());
                    }
                }
            });
    });
    if let Some(spec) = spec {
        ui.add(egui::Label::new(egui::RichText::new(part_properties(spec)).small().weak()).wrap());
    }
    picked
        .filter(|p| p != current)
        .map(|p| RowEdit::Set(Value::Text(p)))
}

fn grade_picker(ui: &mut egui::Ui, current: &str) -> Option<RowEdit> {
    let shown = match grade(current) {
        Some(g) => grade_label(g),
        None if current.is_empty() => BLANK_GRADE.to_owned(),
        None => format!("{current} (not in the grade table)"),
    };
    let mut picked = None;
    ui.horizontal(|ui| {
        ui.weak(PICK_GRADE);
        ComboBox::from_id_salt("grade_picker")
            .selected_text(shown)
            .width(ui.available_width())
            .truncate()
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(current.is_empty(), BLANK_GRADE)
                    .clicked()
                {
                    picked = Some(String::new());
                }
                for g in &GRADES {
                    if ui
                        .selectable_label(current == g.id, grade_label(g))
                        .on_hover_text(grade_properties(g))
                        .clicked()
                    {
                        picked = Some(g.id.to_owned());
                    }
                }
            });
    });
    if let Some(g) = grade(current) {
        ui.add(egui::Label::new(egui::RichText::new(grade_properties(g)).small().weak()).wrap());
    }
    picked
        .filter(|p| p != current)
        .map(|p| RowEdit::Set(Value::Text(p)))
}

#[cfg(test)]
mod tests {
````

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib gui::pickers 2>&1 | grep -E "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result"
```

Expected: `test result: ok. 4 passed; 0 failed` for `gui::pickers`, then `test result: ok. 424 passed; 0 failed` for the whole library.

- [ ] **Step 5: Format and lint**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --check
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
```

Expected: `cargo fmt` changes nothing (the blocks are in rustfmt's layout) and `--check` prints nothing; each of the five other commands prints its `Finished` line (with `-D warnings` any warning would end a clippy run with `error: could not compile` instead).

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m43 add magcoupling-rs/src/gui/pickers.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/gui/input_ui.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/explorer.rs magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-m43 commit -F - <<'EOF'
feat(magcoupling-rs): material, part and grade pickers; material warnings linked to notes

gui::pickers adds the libraries to the input rows (decision M43-4): each
material choice's properties on hover and a summary under the row; "Pick a
part" (custom dimensions or the 15 library parts, with every correction on)
and "Pick a grade" (blank or the 17 grades) under the text fields, which stay.
The dashboard lists the material warnings that fire at its top in their
severity's colour (decision M43-8), each with a "Why" link that opens its
reviewed note in the Equation panel (explorer::reviewed_note).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m43 status --short
```

Expected: one new commit on `magcoupling/m4-3`; `status --short` prints nothing.

---

### Task 7: Docs, the web smoke and the plan

**Model:** `sonnet` (doc and YAML edits, running commands; CLAUDE.md section 5). The whole-branch review after this task runs on the session model.

The docs ship with the code (repo rule): `magcoupling-rs/README.md` (status, layout, "The panel": the explorer, assumptions, notes, pickers and warnings, the smoke's new line), `docs/ai/02-system.yaml` (responsibility, key files, two invariants, three egui lessons, the status line), `03-structure.yaml` (the gui modules), `04-memory.yaml` (the M4-3 item resolved; plan A-3's M4 item and the M4 typesetting pass resolved, M43-5 and M43-13 superseding two of their points; the deferred per-row greying, M43-15; no renames, M43-13; `release_notes_are_reviewed`, which passes with `--ignored`, left for the engine owner to un-ignore), `05-update-tracker.md`. The `gui-smoke` workflow checks one more console line, `magcoupling explorer: ` (the registry built in the browser), with no click (`explorer_ready`); `the_smoke_test_share_link_opens_its_design` checks that the script holds `REGISTRY_LOG_PREFIX`. Then this plan is copied into the repository, the gate runs, and the web bundle is built and opened.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/.claude/workflows/gui-smoke.js`
- Modify: `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/app.rs` (the smoke test's one check)
- Create: `C:/Users/Cole/source/repos/lsim-mag-m43/docs/superpowers/plans/2026-10-01-magcoupling-m4-3-explorer-assumptions-notes-pickers.md` (this plan)

**Interfaces:**
- Consumes: every task's names, for the docs; `readouts::REGISTRY_LOG_PREFIX`.
- Produces: the docs; the smoke's `explorer_ready` field.

- [ ] **Step 1: Edit the workflow and the docs**

In `C:/Users/Cole/source/repos/lsim-mag-m43/.claude/workflows/gui-smoke.js`, replace:

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

with:

````js
}

const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'geometry_view', 'explorer_ready'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    geometry_view: { type: 'boolean' },
    explorer_ready: { type: 'boolean' },
    share_link_loaded: { type: 'boolean' },
    sizing_solved: { type: 'boolean' },
    design_file_loaded: { type: 'boolean' },
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/.claude/workflows/gui-smoke.js`, replace:

````js
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

with:

````js
Then the design file picker, the only click on the canvas: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
In the screenshot, find the "Load design" button in the header row at the top of the page (between "Save design" and "Copy share link") and click its centre with browser_run_code_unsafe, code: async (page) => { await page.mouse.click(X, Y); } (X and Y in CSS pixels of the screenshot; the canvas fills the page). rfd shows its overlay (#rfd-overlay: a file input #rfd-input shown as a "Choose File" button, and the buttons "Ok" and "Cancel") and opens the browser's file chooser at once (the tool output reports a "File chooser" modal state); if a snapshot shows the overlay but no chooser opened, click the "Choose File" button. Upload the file with browser_file_upload, click the overlay's "Ok" button, wait 2 seconds and collect the console messages again; delete the file.
design_file_loaded=true only if a console message contains "magcoupling: loaded a design file". If the button cannot be found or the overlay does not appear after two attempts (each with a fresh screenshot), design_file_loaded=false and say why in notes: the canvas click is the fragile part of this step, and the share-link checks do not depend on it. Close the browser.
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, geometry_view, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, and zero console messages of type error (warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
  ) || { passed: false, canvas_present: false, console_errors: [], geometry_view: false, explorer_ready: false, share_link_loaded: false, sizing_solved: false, design_file_loaded: false, notes: 'smoke agent returned no result (agent error)' }
}

// The linkage app's fields at the top level, as before, with the calculator's beside them.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/02-system.yaml`, replace:

````yaml
      undo/redo, design files and share links, the sizing mode and the linkage app's theme.
      M4-2 adds the centre region's views, the geometry view to scale (the default), five plots
      (egui_plot) and the clamp drawing and table.
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs, src/gui/panel.rs, src/gui/session.rs, src/gui/sizing.rs, src/gui/dashboard.rs, src/gui/geometry.rs, src/gui/geometry_view.rs, src/gui/plots.rs, src/gui/clamp_drawing.rs, src/app.rs]
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
      - Every departure from the workbook is an approved, registered deviation (E1-E20 in deviations.rs; E15-E17 are probed on top of E9, decision 15). compute_all always applies Deviations::ALL; the switch exists only behind the test-only workbook-parity feature.
````

with:

````yaml
      undo/redo, design files and share links, the sizing mode and the linkage app's theme.
      M4-2 adds the centre region's views, the geometry view to scale (the default), five plots
      (egui_plot) and the clamp drawing and table.
      M4-3 adds the equation explorer (typesetter, hover on every value, the docked Equation panel),
      the assumptions view and banner, the teaching notes with their diagrams, the material, part
      and grade pickers and the material warnings.
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs, src/gui/panel.rs, src/gui/session.rs, src/gui/sizing.rs, src/gui/dashboard.rs, src/gui/geometry.rs, src/gui/geometry_view.rs, src/gui/plots.rs, src/gui/clamp_drawing.rs, src/gui/typeset.rs, src/gui/readouts.rs, src/gui/explorer.rs, src/gui/pickers.rs, src/app.rs]
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
      - Every departure from the workbook is an approved, registered deviation (E1-E20 in deviations.rs; E15-E17 are probed on top of E9, decision 15). compute_all always applies Deviations::ALL; the switch exists only behind the test-only workbook-parity feature.
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/02-system.yaml`, replace:

````yaml
      - "sizing::solve never runs per frame: gui::sizing::SizingRunner solves once the design has been still for DEBOUNCE_S (0.25 s) and no edit is in progress. In Torque -> Magnets the panel shows the inputs with the free variable at the solved (or best) value; the inputs keep their own value until the mode is left, when a change still waiting for its debounce is solved at once (solve_now, one solve per click)."
      - "The panel does no I/O: saving and picking files are PanelRequests its host does (app::files natively and on the web; the linkage app in M5)."
      - "The corrected-vs-workbook markers come from the deviation registry (cells, changes at defaults, probes) and the compiled golden files (E3, E4, E5), never from computing with corrections off (gui::corrections)."
      - "The geometry view draws the design shown from its results only (gui::geometry): every dimension is a result, the axial ones the housing in effect (housing.*), so the drawing follows every edit, the length override and Torque -> Magnets; a piece holding a number that is not finite is not drawn, and blocks are drawn only for 2 to 200 poles (the pocket and the hub round below 3 poles: a 2-gon has no corners). Every callout and clamp-table value hovers its result's text through dashboard::hover_text, the hook M4-3 joins."
      - "The plot tabs build their series from the same frame's results through the engine's own closed forms (model::ring_pair_factor for torque against temperature, the amp*_Pa amplitudes for torque against rotation), which tests pin to the engine's torques; only the tab shown builds its series, and a point that is not finite is left out."

dataflow_on_user_change: |
````

with:

````yaml
      - "sizing::solve never runs per frame: gui::sizing::SizingRunner solves once the design has been still for DEBOUNCE_S (0.25 s) and no edit is in progress. In Torque -> Magnets the panel shows the inputs with the free variable at the solved (or best) value; the inputs keep their own value until the mode is left, when a change still waiting for its debounce is solved at once (solve_now, one solve per click)."
      - "The panel does no I/O: saving and picking files are PanelRequests its host does (app::files natively and on the web; the linkage app in M5)."
      - "The corrected-vs-workbook markers come from the deviation registry (cells, changes at defaults, probes) and the compiled golden files (E3, E4, E5), never from computing with corrections off (gui::corrections)."
      - "The geometry view draws the design shown from its results only (gui::geometry): every dimension is a result, the axial ones the housing in effect (housing.*), so the drawing follows every edit, the length override and Torque -> Magnets; a piece holding a number that is not finite is not drawn, and blocks are drawn only for 2 to 200 poles (the pocket and the hub round below 3 poles: a 2-gon has no corners)."
      - "Every displayed value goes through gui::readouts::Readouts::show (the dashboard, the results table, the geometry callouts, the clamp tab, the plot readouts): its hover text and equation are built only while it is hovered, a click opens it in the Equation panel, and a value whose path is a term of the equation in view (hovered last frame, else open) is framed in that term's colour. The equation registry is built once per process (readouts::registry, a OnceLock), never per frame; a frame only looks paths up."
      - "The typesetter draws only what the record says (a test checks every text run against render::plain; a selector compared for equality, = or ≠, with a code shows its choice's label, an ordering keeps its numbers) and only the parentheses the markup writes and only characters egui's default fonts have: gui::typeset::glyph_safe draws the missing ϑ, superscript minus and ∝ as θ, ¯ and ~ (display only; no record or note text changes), ∈ and the ceiling and floor brackets are strokes; a test lays out every equation and checks every character, another every teaching note."
      - "The plot tabs build their series from the same frame's results through the engine's own closed forms (model::ring_pair_factor for torque against temperature, the amp*_Pa amplitudes for torque against rotation), which tests pin to the engine's torques; only the tab shown builds its series, and a point that is not finite is left out."

dataflow_on_user_change: |
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/02-system.yaml`, replace:

````yaml
  - "f32::min and f64::max pass over NaN (they return the other operand), so a scale taken with min over an extent that is not finite comes out finite; check the extents for finiteness first (magcoupling geometry_view::side_by_side)."
  - "egui_plot 0.33 paints a solid Line of n >= 2 points as one Shape::Path of n points and each Points marker as a Shape::Circle of its radius, so a headless test counts a plot's series by point count, in a bare frame where no other widget paints paths (magcoupling plots tests)."
  - "Windows MAX_PATH: a release build under a deep directory fails with LNK1104 (cannot open a build-script exe). Build from a short path (the worktree, or subst a drive letter)."

current_statuses:
  solver: Rust port complete; 716 lib tests pass; trajectory-mode IK Stage 1
````

with:

````yaml
  - "f32::min and f64::max pass over NaN (they return the other operand), so a scale taken with min over an extent that is not finite comes out finite; check the extents for finiteness first (magcoupling geometry_view::side_by_side)."
  - "egui_plot 0.33 paints a solid Line of n >= 2 points as one Shape::Path of n points and each Points marker as a Shape::Circle of its radius, so a headless test counts a plot's series by point count, in a bare frame where no other widget paints paths (magcoupling plots tests)."
  - "Windows MAX_PATH: a release build under a deep directory fails with LNK1104 (cannot open a build-script exe). Build from a short path (the worktree, or subst a drive letter)."
  - "egui 0.32's CollapsingHeader computes its id inside a child ui (ui.vertical), so ui.make_persistent_id(salt) in the parent names another state: to open a header from code, pass CollapsingHeader::open(Some(true)) for that frame (magcoupling panel: a leaf term opens its input group)."
  - "egui 0.32: a Label in a horizontal_wrapped row wraps by laying its galley out from the row's start with leading space, so its text rect (and a test's click at its centre) starts mid-row; set ui.style_mut().wrap_mode = Some(TextWrapMode::Extend) in rows of links so each moves whole (magcoupling explorer crumbs, used-by, plot readouts)."
  - "egui 0.32: Label::sense(Sense::click()) draws the label in the interactive text colour (widgets.inactive), not the plain text colour; to make a value clickable without changing its look, put a click sensor over its rect with ui.interact (magcoupling Readouts::show_over)."

current_statuses:
  solver: Rust port complete; 716 lib tests pass; trajectory-mode IK Stage 1
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/02-system.yaml`, replace:

````yaml
  dxf_import: works native + WASM via drag-drop; selection-based conversions
  actuator_workflow: two-pass solve gives correct reactions with actuator loads
  non_grashof_sweep: NaN-padded across unreachable angles; playback ping-pongs
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim. M4 infrastructure in place (gui panel, app binaries, /magcoupling/ bundle, workbook-parity guard, gates 7-11). Addendum A-3 complete: equation explorer engine side, A3 traceability, A4 notes. M4-1 complete (inputs, dashboard, results table, session, sizing mode, theme). M4-2 complete (geometry view, five plots, clamp drawing and table, narrow results table); next: M4-3 (equation explorer, assumptions panel, notes, pickers), then M3 (live 3D fields)"
````

with:

````yaml
  dxf_import: works native + WASM via drag-drop; selection-based conversions
  actuator_workflow: two-pass solve gives correct reactions with actuator loads
  non_grashof_sweep: NaN-padded across unreachable angles; playback ping-pongs
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim. M4 infrastructure in place (gui panel, app binaries, /magcoupling/ bundle, workbook-parity guard, gates 7-11). Addendum A-3 complete: equation explorer engine side, A3 traceability, A4 notes. M4-1 complete (inputs, dashboard, results table, session, sizing mode, theme). M4-2 complete (geometry view, five plots, clamp drawing and table, narrow results table). M4-3 complete (equation explorer: typesetter, hover, Equation panel; assumptions view and banner; teaching notes and diagrams; material, part and grade pickers; material warnings); next: M3 (live 3D fields), then M5 (embed)"
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/03-structure.yaml`, replace:

````yaml

magcoupling_rs:
  path: magcoupling-rs/ (repo root, sibling of linkage-sim-rs/; separate crate, NOT a workspace member)
  role: Rust port of reference/magcoupling-py (spec docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md); M2 complete - every Python module except fields3d (M3) is ported, workbook parity covers all 1,149 checks, E1-E14 applied; Addendum A-1 (grades, materials, E15-E20), A-2 (harmonic set, assumptions, inverse sizing, space claim) and A-3 (equation explorer engine side - evaluable records, drift guard, A3 traceability, A4 notes) complete; M4 infrastructure adds the egui panel, the standalone app binaries and the /magcoupling/ web bundle; M4-1 adds the inputs, dashboard, results table, session (undo/redo, design files, share links), sizing mode and theme; M4-2 adds the geometry view, five plots (egui_plot) and the clamp drawing and table
  guide: magcoupling-rs/README.md (tests, regenerating data, porting a module, Python-to-Rust translation rules, applying deviations)
  lib: magcoupling (src/lib.rs re-exports compute_all, headline, DesignInputs, DesignResults; mod gui behind feature gui, mod app behind feature app)
  engine: src/engine/ (pure std, wasm32-clean; one module per Python module)
````

with:

````yaml

magcoupling_rs:
  path: magcoupling-rs/ (repo root, sibling of linkage-sim-rs/; separate crate, NOT a workspace member)
  role: Rust port of reference/magcoupling-py (spec docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md); M2 complete - every Python module except fields3d (M3) is ported, workbook parity covers all 1,149 checks, E1-E14 applied; Addendum A-1 (grades, materials, E15-E20), A-2 (harmonic set, assumptions, inverse sizing, space claim) and A-3 (equation explorer engine side - evaluable records, drift guard, A3 traceability, A4 notes) complete; M4 infrastructure adds the egui panel, the standalone app binaries and the /magcoupling/ web bundle; M4-1 adds the inputs, dashboard, results table, session (undo/redo, design files, share links), sizing mode and theme; M4-2 adds the geometry view, five plots (egui_plot) and the clamp drawing and table; M4-3 adds the equation explorer (typesetter, hover, Equation panel), the assumptions view and banner, the teaching notes and diagrams, the material, part and grade pickers and the material warnings
  guide: magcoupling-rs/README.md (tests, regenerating data, porting a module, Python-to-Rust translation rules, applying deviations)
  lib: magcoupling (src/lib.rs re-exports compute_all, headline, DesignInputs, DesignResults; mod gui behind feature gui, mod app behind feature app)
  engine: src/engine/ (pure std, wasm32-clean; one module per Python module)
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/03-structure.yaml`, replace:

````yaml
    addendum_only: [assumptions.rs, sizing.rs, housing.rs, explain/]
    remaining: [fields3d (M3)]
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log; no eframe, no I/O): panel.rs (MagcouplingPanel: header with Undo/Redo/Reset all/Save design/Load design/Copy share link, inputs SidePanel, dashboard SidePanel, CentreView (Geometry, the default; Plot(PlotKind) x5; Clamp; Results) under the end-effect banner; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; PanelRequest (SaveFile, OpenDesign) drained by take_requests; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; sizing controls and run_sizing), inputs.rs (InputCatalogue::get, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS, step_decimals, outside_range, text_hint, input_tooltip), input_ui.rs (input_row, slider, RowEdit), dashboard.rs (DASHBOARD, Level, verdict_level, END_EFFECT_ROWS, STORED_3D_ROWS, result_info, result_tooltip (the hover hook), hover_text, dashboard_lines, end_effect_banner), geometry.rs (geometry -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); DESIGN_CHECK, AUTOFIT, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout; Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip), results_table.rs (table_entries, search, exact_number, results_csv, results_json, ResultsTable, row_tooltip, column_widths), session.rs (Design, design_to_json/design_from_json, encode/decode_share_payload, share_link, LoadError, PATH_MIGRATIONS, json_number), history.rs (History, MAX_UNDO_LEVELS 100), sizing.rs (SizingMode, SizingState, variable_key/label, SizingRunner (update, solve_now), DEBOUNCE_S 0.25), format.rs (format_value, with_unit, non_finite_text), test_support.rs (cfg(test): sized_frame(_at), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, text_rect, text_rects, text_color, short_magnets)"
    app: "src/app.rs (MagcouplingApp::new applies the theme; ui drains the panel's requests; open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (cad_dark_visuals, a copy of linkage-sim-rs's checked body for body; apply forces ThemePreference::Dark), src/app/files.rs (save: rfd dialog natively, Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
    workbook_parity: "test-only, enabled by a self dev-dependency; never in a shipped build (linkage-sim-rs/scripts/magcoupling_shipped.sh, gate 10)"
    versions: "Cargo.toml caret ranges (egui 0.32, egui_plot 0.33, eframe 0.32, wasm-bindgen 0.2); Cargo.lock pins linkage-sim-rs/Cargo.lock's egui/eframe 0.32.3, egui_plot 0.33.0 and wasm-bindgen 0.2.114 (= the deploy-web.yml wasm-bindgen-cli pin); gate 11 checks"
````

with:

````yaml
    addendum_only: [assumptions.rs, sizing.rs, housing.rs, explain/]
    remaining: [fields3d (M3)]
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log; no eframe, no I/O): typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (show, show_over, mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, open_note, start, stop, marks, end_frame; explorer_ui, note_ui, term_tag, term_label, plain_symbol, reviewed_note (every note shown by id); EQUATION_PANEL, PANEL_HEIGHT, MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), panel.rs (MagcouplingPanel: header with Undo/Redo/Reset all/Save design/Load design/Copy share link, inputs SidePanel, dashboard SidePanel, CentreView (Geometry, the default; Plot(PlotKind) x5; Clamp; Results) under the end-effect banner; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; PanelRequest (SaveFile, OpenDesign) drained by take_requests; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; sizing controls and run_sizing), inputs.rs (InputCatalogue::get, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS, step_decimals, outside_range, text_hint, input_tooltip), input_ui.rs (input_row, slider, RowEdit), dashboard.rs (DASHBOARD, Level, verdict_level, END_EFFECT_ROWS, STORED_3D_ROWS, result_info, result_tooltip (the hover hook), hover_text, dashboard_lines, end_effect_banner), geometry.rs (geometry -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); DESIGN_CHECK, AUTOFIT, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout; Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip), results_table.rs (table_entries, search, exact_number, results_csv, results_json, ResultsTable, row_tooltip, column_widths), session.rs (Design, design_to_json/design_from_json, encode/decode_share_payload, share_link, LoadError, PATH_MIGRATIONS, json_number), history.rs (History, MAX_UNDO_LEVELS 100), sizing.rs (SizingMode, SizingState, variable_key/label, SizingRunner (update, solve_now), DEBOUNCE_S 0.25), format.rs (format_value, with_unit, non_finite_text), test_support.rs (cfg(test): sized_frame(_at), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, text_rect, text_rects, text_color, assert_glyphs, short_magnets)"
    app: "src/app.rs (MagcouplingApp::new applies the theme; ui drains the panel's requests; open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (cad_dark_visuals, a copy of linkage-sim-rs's checked body for body; apply forces ThemePreference::Dark), src/app/files.rs (save: rfd dialog natively, Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
    workbook_parity: "test-only, enabled by a self dev-dependency; never in a shipped build (linkage-sim-rs/scripts/magcoupling_shipped.sh, gate 10)"
    versions: "Cargo.toml caret ranges (egui 0.32, egui_plot 0.33, eframe 0.32, wasm-bindgen 0.2); Cargo.lock pins linkage-sim-rs/Cargo.lock's egui/eframe 0.32.3, egui_plot 0.33.0 and wasm-bindgen 0.2.114 (= the deploy-web.yml wasm-bindgen-cli pin); gate 11 checks"
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/04-memory.yaml`, replace:

````yaml
  - "M4 performance (deferred from the A-2 final review, not a defect): model::shear_stress evaluates all six harmonics (amplitudes and sinh/exp geometry factors of 7, 9 and 11 too) on every call, in the Calculator and each of the 19 sweep rows, even when coupling.max_harmonic sums only 1, 3, 5; with the general root search it makes up compute_all's 8.5 -> 17 us release cost (measured in the A-2 plan), about 50x inside the spec's milliseconds. If M4's per-frame budget (drawing, A-3 equation records) needs it, compute only the harmonics summed, keeping the bits of 1, 3, 5 and a left-out harmonic's reported 0"
  - "RESOLVED (Addendum A-3): the dependency-graph traceability test (tests/explain.rs: soundness for every input, sensitivity for the 15 assumption inputs, each reaching an explained result, with the cancellations listed with reasons); the records include tau7_Pa-tau11_Pa and the Calibration sums (model.tau#_Pa, calibration.tau#_Pa families)"
  - "RESOLVED (Addendum A-3 decisions G1-G7, as the user confirmed them before execution): G1 the geometry callouts are the face gap, corner gap, running clearance and its two parts, and the overshoot per axis with the dimension it measures (169 scope paths); G2 housing.space_claim_check stays unexplained (its text joins the exceeded axes); G3 the record batches follow the chain closure (demagnetization, slip heating, temperature, clamps, dashboard, geometry); G4 the outer ring's demagnetization block is computed with E20 off too and governs nothing; G5 17 notes (the spec's list, ferrite's cold side, the one-point calibration) and two more diagram kinds, DemagKnee and HeatingCurve; G6 the spec's A2 eval closure became the evaluable markup (a Rust closure only as a counted escape hatch, none used); G7 the spec's A3 test became two one-sided checks (soundness for every input; sensitivity on the active path at some design point, above rounding, the algebraic cancellations listed)"
  - "M4 (from plan A-3): the typesetter draws explain::markup::Expr; a selector compared with a code shows the choice label (explain::Registry::choices; materials.circuit_backiron borrows coupling.backiron's, explain::registry::RESULT_CHOICES); the corrected-vs-workbook marker reads Registry::corrections_upstream (a record's own corrections, then every upstream record's; precomputed at build, a lookup per shown value, while upstream and downstream walk the graph on every call, for the drill-down rather than every value of a frame); term_style styles the panel's terms only (a cell-only result is never marked affected); a screw_sizes table key is a row index, shown as the size name; paint the six notes::Diagram kinds; build explain::Registry once at start-up and own it in the app state; only reviewed notes show (notes::note_for); tests/explain.rs release_notes_are_reviewed (ignored) is on the hands-on checklist"
  - "M4 typesetting pass (from the A-3 final review): the symbol letters the chains reuse are listed in explain/symbols.rs's conventions header (C the heat capacity or a verdict text, t a time or a thickness, D_{slip} a duty among diameters, m_{hot} a margin ratio among masses, A an apothem, amplitude, area-lever product or area local); decide any renames with the panel in view. A symbol an A4 note quotes changes with the note, which sends it back to Draft and needs a new physics review; the thermal_time_constant note already writes tau and theta_start where the records show tau_th and theta_hot"
  - "M3: a grade-mode ring's reverse fields are scaled with its own alpha(Br) (decision A2-7); the stored 3D fields do not separate the opposing ring's share, which live fields per ring would"
  - "RESOLVED (M4 infrastructure, magcoupling/m4-infra): the workbook-parity guard checks cargo's --message-format=json record of the shipped builds (linkage-sim-rs/scripts/magcoupling_shipped.sh, magcoupling_assert_shipped). build_magcoupling_web.sh pipes its release build through it, and gate 10 runs it on the native and wasm32 shipped builds plus a negative control that must trip. It trips when the feature is forced on the command line or through app in Cargo.toml, and cargo test and clippy --all-targets with app stay green. M5 must also run it on linkage-sim-rs's shipped builds once linkage-sim-rs depends on magcoupling-rs (the function already accepts any binary; the magcoupling-rs units are matched by package id)"
  - "RESOLVED (M4-1): the Key design 'axial length' is the A-2 override coupling.magnets.axial_length_mm, blank by default, entered at the inner ring's length in use (decision M41-12); slider values are rounded to the step's decimals (M41-1); edits, typed values included, are clamped to the slider range and a value already outside it is kept and flagged (M41-2); gui-smoke opens /magcoupling/ through a pinned share link; the standalone app forces the linkage app's CAD dark theme (M41-3)"
  - "Open (M4-1 follow-ups): magcoupling-web_bg.wasm is 4.0 MB with the default release profile after M4-1 (3.5 MB before; rfd, serde_json, flate2); a size profile (opt-level, LTO) or wasm-opt is a later option. The linkage web app itself still renders light in a light-preference browser (backlog BL-013): ctx.set_theme(egui::ThemePreference::Dark), as magcoupling's app/theme.rs does, is the likely fix"
  - "RESOLVED (M4-2): decisions M42-1 to M42-9 as recommended in docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md (confirmed by the user at its Task 0); the centre region's CentreView holds the geometry view (the default), the five plot tabs, the clamp tab and the results table, one tab row; the results table's label column flexes (results_table::column_widths, at least 120 points), so a ~930 px window shows each row's label and value; egui_plot is locked at linkage-sim-rs's 0.33.0 and gate 11 checks it; the end-effect banner sits over the dashboard and the centre region"
  - "Open (M4-3, carried from M4-1): per-row end-effect greying of the results table waits for plan A-3's dependency graph (M4-1 shows the banner over the table and greys the pinned dashboard rows); M4-3 may re-source the corrected-vs-workbook markers from A-3's per-record upstream corrections; result_tooltip (gui/dashboard.rs) is the hover hook the equation tooltip joins; when it does, dashboard_ui must build its hover text lazily in on_hover_ui, as the results table's row_ui and M4-2's geometry callouts and clamp table do (dashboard_lines builds all 17 rows' tooltips on every frame, negligible while they are a few lines: M4-1 final review); M4-2's readouts already go through dashboard::hover_text"
  - "Open (M4-2 final review, deferred): the clamp summary shows 'Screws per clamp 1.000' because format_value prints clamps.screws, the workbook's number (NumOrText), to four significant digits, as the results table always has; an integer display needs a per-result rule in the metadata, not a string special case for one path. Below about 480 points wide the clamp drawing's texts stop shrinking at 8 points, so the three-line relief note reaches the band of the view titles. With 2 poles (a design file only) face 1 is at the bottom, so the corner-gap and running-clearance callouts share it. Not measured by the review: the per-frame cost on wasm (structurally one compute_all per frame, the geometry and plot series only for the tab shown, the clamp rows OnceLock) and a browser hover of a callout's text in the list (the dimension line's hover has a headless test)."
  - "Open (M5, from the M4-1 review): linkage-sim-rs/src/gui/mod.rs reads Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y with a non-consuming key_pressed, which runs whatever the panel consumes. Hosting MagcouplingPanel in an egui::Window, M5 must skip its own undo while the calculator window has the user's attention and turn the panel's shortcuts on only then (MagcouplingPanel::set_keyboard_shortcuts), so one key press never undoes both"
  - "Deploy (M4-1): nothing pushed. deploy-web.yml on main already builds and ships /magcoupling/ on the next push of main; that push needs the user's explicit go"
````

with:

````yaml
  - "M4 performance (deferred from the A-2 final review, not a defect): model::shear_stress evaluates all six harmonics (amplitudes and sinh/exp geometry factors of 7, 9 and 11 too) on every call, in the Calculator and each of the 19 sweep rows, even when coupling.max_harmonic sums only 1, 3, 5; with the general root search it makes up compute_all's 8.5 -> 17 us release cost (measured in the A-2 plan), about 50x inside the spec's milliseconds. If M4's per-frame budget (drawing, A-3 equation records) needs it, compute only the harmonics summed, keeping the bits of 1, 3, 5 and a left-out harmonic's reported 0"
  - "RESOLVED (Addendum A-3): the dependency-graph traceability test (tests/explain.rs: soundness for every input, sensitivity for the 15 assumption inputs, each reaching an explained result, with the cancellations listed with reasons); the records include tau7_Pa-tau11_Pa and the Calibration sums (model.tau#_Pa, calibration.tau#_Pa families)"
  - "RESOLVED (Addendum A-3 decisions G1-G7, as the user confirmed them before execution): G1 the geometry callouts are the face gap, corner gap, running clearance and its two parts, and the overshoot per axis with the dimension it measures (169 scope paths); G2 housing.space_claim_check stays unexplained (its text joins the exceeded axes); G3 the record batches follow the chain closure (demagnetization, slip heating, temperature, clamps, dashboard, geometry); G4 the outer ring's demagnetization block is computed with E20 off too and governs nothing; G5 17 notes (the spec's list, ferrite's cold side, the one-point calibration) and two more diagram kinds, DemagKnee and HeatingCurve; G6 the spec's A2 eval closure became the evaluable markup (a Rust closure only as a counted escape hatch, none used); G7 the spec's A3 test became two one-sided checks (soundness for every input; sensitivity on the active path at some design point, above rounding, the algebraic cancellations listed)"
  - "RESOLVED (M4-3; decisions M43-5 and M43-13 supersede two points, marked here): the typesetter draws explain::markup::Expr; a selector compared for equality (= or ≠) with a code shows the choice label (an ordering keeps its numbers: σ_n's n <= N_h) (explain::Registry::choices; materials.circuit_backiron borrows coupling.backiron's, explain::registry::RESULT_CHOICES); the corrected-vs-workbook marker reads Registry::corrections_upstream (a record's own corrections, then every upstream record's; precomputed at build, a lookup per shown value, while upstream and downstream walk the graph on every call, for the drill-down rather than every value of a frame); term_style styles the panel's terms only (a cell-only result is never marked affected); a screw_sizes table key is a row index, shown as the size name; paint the six notes::Diagram kinds; build explain::Registry once at start-up (superseded by M43-5: a process-wide OnceLock, gui::readouts::registry, not owned in the app state); only reviewed notes show (notes::note_for; a note shown by id through gui::explorer::reviewed_note); tests/explain.rs release_notes_are_reviewed (ignored) is on the hands-on checklist"
  - "RESOLVED (M4-3, decision M43-13: no renames; the open M43-13 item below carries the note's tau and theta_start): M4 typesetting pass (from the A-3 final review): the symbol letters the chains reuse are listed in explain/symbols.rs's conventions header (C the heat capacity or a verdict text, t a time or a thickness, D_{slip} a duty among diameters, m_{hot} a margin ratio among masses, A an apothem, amplitude, area-lever product or area local); decide any renames with the panel in view. A symbol an A4 note quotes changes with the note, which sends it back to Draft and needs a new physics review; the thermal_time_constant note already writes tau and theta_start where the records show tau_th and theta_hot"
  - "M3: a grade-mode ring's reverse fields are scaled with its own alpha(Br) (decision A2-7); the stored 3D fields do not separate the opposing ring's share, which live fields per ring would"
  - "RESOLVED (M4 infrastructure, magcoupling/m4-infra): the workbook-parity guard checks cargo's --message-format=json record of the shipped builds (linkage-sim-rs/scripts/magcoupling_shipped.sh, magcoupling_assert_shipped). build_magcoupling_web.sh pipes its release build through it, and gate 10 runs it on the native and wasm32 shipped builds plus a negative control that must trip. It trips when the feature is forced on the command line or through app in Cargo.toml, and cargo test and clippy --all-targets with app stay green. M5 must also run it on linkage-sim-rs's shipped builds once linkage-sim-rs depends on magcoupling-rs (the function already accepts any binary; the magcoupling-rs units are matched by package id)"
  - "RESOLVED (M4-1): the Key design 'axial length' is the A-2 override coupling.magnets.axial_length_mm, blank by default, entered at the inner ring's length in use (decision M41-12); slider values are rounded to the step's decimals (M41-1); edits, typed values included, are clamped to the slider range and a value already outside it is kept and flagged (M41-2); gui-smoke opens /magcoupling/ through a pinned share link; the standalone app forces the linkage app's CAD dark theme (M41-3)"
  - "Open (M4-1 follow-ups): magcoupling-web_bg.wasm is 4.0 MB with the default release profile after M4-1 (3.5 MB before; rfd, serde_json, flate2); a size profile (opt-level, LTO) or wasm-opt is a later option. The linkage web app itself still renders light in a light-preference browser (backlog BL-013): ctx.set_theme(egui::ThemePreference::Dark), as magcoupling's app/theme.rs does, is the likely fix"
  - "RESOLVED (M4-2): decisions M42-1 to M42-9 as recommended in docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md (confirmed by the user at its Task 0); the centre region's CentreView holds the geometry view (the default), the five plot tabs, the clamp tab and the results table, one tab row; the results table's label column flexes (results_table::column_widths, at least 120 points), so a ~930 px window shows each row's label and value; egui_plot is locked at linkage-sim-rs's 0.33.0 and gate 11 checks it; the end-effect banner sits over the dashboard and the centre region"
  - "RESOLVED (M4-3): decisions M43-1 to M43-15 as recommended in docs/superpowers/plans/2026-10-01-magcoupling-m4-3-explorer-assumptions-notes-pickers.md; every readout goes through gui::readouts::Readouts::show with its hover text built lazily (the dashboard's too: DashboardLine::tooltip); the corrected-vs-workbook markers stay M4-1's CorrectionIndex and the Equation panel lists Registry::corrections_upstream (M43-14); the registry is a process-wide OnceLock built in MagcouplingPanel::new (M43-5)"
  - "Open (M4-3, deferred, M43-15): per-row end-effect greying of the results table stays the banner: the registry's graph covers only the explained results (about 400 of the table's rows), so greying by it would grey some rows computed from the pull-out and miss the cell-only ones; revisit when every result has a record or the engine flags its own dependents"
  - "Open (M4-3, for the engine owner): tests/explain.rs release_notes_are_reviewed passes when run with --ignored (every START_HERE and warning note is Reviewed, checked at M4-3); it stays #[ignore] only because M4-3 edits no engine test, so un-ignoring it, which makes the release gate permanent, is the next engine change's step"
  - "Open (M4-3, M43-13): no symbol renames in M4-3. The A4 notes are shown as reviewed, unchanged; the thermal_time_constant note writes tau and theta_start where the records show tau_th and theta_hot (a reader sees both in the panel); a rename or a note edit sends the note back to Draft and needs a physics review"
  - "Open (M4-2 final review, deferred): the clamp summary shows 'Screws per clamp 1.000' because format_value prints clamps.screws, the workbook's number (NumOrText), to four significant digits, as the results table always has; an integer display needs a per-result rule in the metadata, not a string special case for one path. Below about 480 points wide the clamp drawing's texts stop shrinking at 8 points, so the three-line relief note reaches the band of the view titles. With 2 poles (a design file only) face 1 is at the bottom, so the corner-gap and running-clearance callouts share it. Not measured by the review: the per-frame cost on wasm (structurally one compute_all per frame, the geometry and plot series only for the tab shown, the clamp rows OnceLock) and a browser hover of a callout's text in the list (the dimension line's hover has a headless test)."
  - "Open (M5, from the M4-1 review): linkage-sim-rs/src/gui/mod.rs reads Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y with a non-consuming key_pressed, which runs whatever the panel consumes. Hosting MagcouplingPanel in an egui::Window, M5 must skip its own undo while the calculator window has the user's attention and turn the panel's shortcuts on only then (MagcouplingPanel::set_keyboard_shortcuts), so one key press never undoes both"
  - "Deploy (M4-1): nothing pushed. deploy-web.yml on main already builds and ships /magcoupling/ on the next push of main; that push needs the user's explicit go"
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/05-update-tracker.md`, replace:

````markdown
Reverse chronological (newest at top).

---

## 2026-10-01 — Magcoupling M4-2: final review fix wave (branch magcoupling/m4-2)
- The whole-branch review found no Critical or Important issue; five Minor findings, fixed with a
````

with:

````markdown
Reverse chronological (newest at top).

---

## 2026-10-01 — Magcoupling M4-3: equation explorer, assumptions, teaching notes, pickers (branch magcoupling/m4-3)
- Equation explorer (`gui/typeset.rs`, `gui/readouts.rs`, `gui/explorer.rs`): an egui typesetter for
  the A-3 markup (stacked fractions, scripts, strokes for √, ∈ and the ceiling and floor brackets,
  a large Σ, `arg max`, scaled delimiters, `cases` braces, `where` lines); every displayed value
  (dashboard, results table, geometry callouts, clamp tab, a readout line over each plot) shows its
  equation on hover with each term coloured, and the terms' values and input rows are framed in
  those colours on screen; a click opens the docked Equation panel (bottom of the centre region:
  breadcrumb, the equation large, value and corrections, term list, "used by"); clicking a term
  drills in, an input term opens its group, scrolls to and frames its row. The registry is built
  once per process. A selector compared for equality with a code shows its choice's label (an
  ordering keeps its numbers); a test checks every drawn run against the record's plain rendering.
  A long walk shows its last 8 crumbs; the term list's swatches fade while another equation's
  value is hovered.
- Assumptions: the inputs side's Assumptions view (each assumption's rows, rationale and source),
  the header banner with "Reset to workbook defaults", assumption and affected terms tagged in the
  Equation panel.
- Teaching notes: the Explain toggle shows the open equation's reviewed note with its painted
  diagram (`gui/diagrams.rs`: six kinds); "Start here" walks the suggested order.
- Pickers and warnings (`gui/pickers.rs`): material choices with their library properties; part
  and grade pickers from the A6 tables beside the text fields; the material warnings at the top of
  the dashboard in their severity's colour, each linked to its note.
- Decisions M43-1 to M43-15 (the plan's table); deferred: per-row end-effect greying (M43-15),
  symbol renames (M43-13). The engine, parity, differential data and registry are unchanged.
  Nothing pushed.

## 2026-10-01 — Magcoupling M4-2: final review fix wave (branch magcoupling/m4-2)
- The whole-branch review found no Critical or Important issue; five Minor findings, fixed with a
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md`, replace:

````markdown
[The panel](#the-panel); decisions M41-1 to M41-15). M4-2 complete: the centre region's views,
the geometry view to scale (the default: both rings, the housing in effect, the dimension
callouts, the space claim and the design checks), five plots (egui_plot) and the clamp drawing
and table, and a results table that fits a narrow window (decisions M42-1 to M42-9). Next: M4-3
(equation explorer, assumptions panel, teaching notes, material and grade pickers).

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
````

with:

````markdown
[The panel](#the-panel); decisions M41-1 to M41-15). M4-2 complete: the centre region's views,
the geometry view to scale (the default: both rings, the housing in effect, the dimension
callouts, the space claim and the design checks), five plots (egui_plot) and the clamp drawing
and table, and a results table that fits a narrow window (decisions M42-1 to M42-9). M4-3
complete: the equation explorer (an egui typesetter for the A-3 markup, every displayed value's
equation on hover with its terms coloured and marked on screen, the docked Equation panel with
its breadcrumb, term list, "used by" and leaf-to-slider links), the assumptions view and banner,
the teaching notes with their diagrams and the start-here order, the material, part and grade
pickers and the material warnings (decisions M43-1 to M43-15). Next: M3 (live 3D fields).

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md`, replace:

````markdown
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/engine/explain/` | Addendum A2 equation explorer, engine side (plan A-3): the formula markup (`markup.rs`: grammar, parser, parse tree), its evaluator with a trace of what a result depends on (`eval.rs`), the static tables a formula reads (`tables.rs`: magnets, grades, the part materials, aluminium alloys, adhesives, screw sizes), the authoring form (`record.rs`) and the records, one file per chain (`records/`), the registry (`registry.rs`: `Registry::build`, `equation_for`, `used_by`, the dependency graph, `term_rows`; each result's upstream corrections are precomputed at build, which a unit test checks against a walk of the graph), display symbols (`symbols.rs`), the v1 scope (`scope.rs`, decision 31) and a plain-text rendering (`render.rs`) |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, the sizing controls), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list, `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type), `dashboard.rs` (`DASHBOARD`, `verdict_level`, `END_EFFECT_ROWS`, `STORED_3D_ROWS`, `result_info`, `result_tooltip`, the hover hook, and `hover_text`, its text with the correction marks), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `search`, `results_csv`, `results_json`, `column_widths`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`), `history.rs` (`History`: undo and redo), `sizing.rs` (`SizingMode`, `SizingState`, `SizingRunner`: debounced inverse sizing), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included), `test_support.rs` (headless egui helpers, tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the linkage app's CAD dark visuals, forced dark), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
````

with:

````markdown
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/engine/explain/` | Addendum A2 equation explorer, engine side (plan A-3): the formula markup (`markup.rs`: grammar, parser, parse tree), its evaluator with a trace of what a result depends on (`eval.rs`), the static tables a formula reads (`tables.rs`: magnets, grades, the part materials, aluminium alloys, adhesives, screw sizes), the authoring form (`record.rs`) and the records, one file per chain (`records/`), the registry (`registry.rs`: `Registry::build`, `equation_for`, `used_by`, the dependency graph, `term_rows`; each result's upstream corrections are precomputed at build, which a unit test checks against a walk of the graph), display symbols (`symbols.rs`), the v1 scope (`scope.rs`, decision 31) and a plain-text rendering (`render.rs`) |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, `InputsView` (the design inputs or the assumptions), the assumptions banner, the sizing controls), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::show`, `show_over`, `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `note_ui`, `term_tag`, `reviewed_note`), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list, `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type), `dashboard.rs` (`DASHBOARD`, `verdict_level`, `END_EFFECT_ROWS`, `STORED_3D_ROWS`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `search`, `results_csv`, `results_json`, `column_widths`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`), `history.rs` (`History`: undo and redo), `sizing.rs` (`SizingMode`, `SizingState`, `SizingRunner`: debounced inverse sizing), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included), `test_support.rs` (headless egui helpers, tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the linkage app's CAD dark visuals, forced dark), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md`, replace:

````markdown

### The panel

M4-1 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md`)
and M4-2 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md`).
A header with the session buttons; the inputs on the left; the dashboard on the right; the
centre region between them (`CentreView`, one tab row: the geometry view, the five plots, the
clamp and the results table; decisions M42-1 and M42-2), under the end-effect banner when
````

with:

````markdown

### The panel

M4-1 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md`),
M4-2 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md`) and M4-3
(plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-3-explorer-assumptions-notes-pickers.md`).
A header with the session buttons; the inputs on the left; the dashboard on the right; the
centre region between them (`CentreView`, one tab row: the geometry view, the five plots, the
clamp and the results table; decisions M42-1 and M42-2), under the end-effect banner when
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md`, replace:

````markdown
  ring's length in use moves nothing; the measured drag at the model's drag switches the thermal
  summary to the measured branch); a text input a text field with a note (library part or not,
  grade or not). Each row shows a dot when changed from the default, a
  reset button, and a tooltip with help, path, workbook cell, slider range and default.
- **Dashboard** (`dashboard.rs`): the 15 headline numbers in Python's order, the space claim
  badge and the end-effect flag (`DASHBOARD`). Green, amber or red badges come from the check
  verdicts (`verdict_level`). A value whose workbook cell the deviation registry ties to an applied
````

with:

````markdown
  ring's length in use moves nothing; the measured drag at the model's drag switches the thermal
  summary to the measured branch); a text input a text field with a note (library part or not,
  grade or not). Each row shows a dot when changed from the default, a
  reset button, and a tooltip with help, path, workbook cell, slider range and default. A tab
  row above them switches to the **Assumptions** view (decision M43-7).
- **Dashboard** (`dashboard.rs`): the 15 headline numbers in Python's order, the space claim
  badge and the end-effect flag (`DASHBOARD`). Green, amber or red badges come from the check
  verdicts (`verdict_level`). A value whose workbook cell the deviation registry ties to an applied
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md`, replace:

````markdown
  corrected values and the evidence in its tooltip (`corrections.rs`, M41-10). When f_end <= 0
  (audit M9) the rows computed from the pull-out (`END_EFFECT_ROWS`) are greyed without badges
  under the banner "End-effect model out of range"; the temperature rows that read the stored 3D
  fields carry "3D values from the workbook" (M3 comes after M4). `result_tooltip` is the one
  hover hook of every readout, keyed by result path.
- **Results table** (`results_table.rs`): every result with label, value, unit, workbook cell and
  marker; search by label, path or cell; CSV (`path,label,value,unit,cell`) and JSON (with the
  design that produced the results: in Torque -> Magnets the inputs with the free variable at the
````

with:

````markdown
  corrected values and the evidence in its tooltip (`corrections.rs`, M41-10). When f_end <= 0
  (audit M9) the rows computed from the pull-out (`END_EFFECT_ROWS`) are greyed without badges
  under the banner "End-effect model out of range"; the temperature rows that read the stored 3D
  fields carry "3D values from the workbook" (M3 comes after M4). `result_tooltip` is the hover
  text of every readout, keyed by result path, built only while the row is hovered
  (`DashboardLine::tooltip`). Over the rows, the material warnings that fire (below).
- **Results table** (`results_table.rs`): every result with label, value, unit, workbook cell and
  marker; search by label, path or cell; CSV (`path,label,value,unit,cell`) and JSON (with the
  design that produced the results: in Torque -> Magnets the inputs with the free variable at the
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md`, replace:

````markdown
  diameter in the side view), an exceeded axis red. Under the callouts: the autofit wall (the
  wall rule's `materials.cup_wall_suggested_mm`, unless there is no back iron) and the two
  design checks while they hold (the cap thread below the cup body OD; the boss OD differing
  between Metal design and Shaft clamps: flagged only, no engine change; M42-9). Hovering a dimension
  or its text shows its result's hover text (`dashboard::hover_text`, the hook M4-3 joins). A
  piece holding a number that is not finite is not drawn (a note counts them per view), a claim
  line farther than `CLAIM_REACH` (10) times the pieces' extent is left off with a note (a design
  file can hold any finite number), and blocks are drawn for 2 to 200 poles (below 3 the pocket
````

with:

````markdown
  diameter in the side view), an exceeded axis red. Under the callouts: the autofit wall (the
  wall rule's `materials.cup_wall_suggested_mm`, unless there is no back iron) and the two
  design checks while they hold (the cap thread below the cup body OD; the boss OD differing
  between Metal design and Shaft clamps: flagged only, no engine change; M42-9). A dimension and
  its text are readouts: hovering shows the result's hover text and equation (below). A
  piece holding a number that is not finite is not drawn (a note counts them per view), a claim
  line farther than `CLAIM_REACH` (10) times the pieces' extent is left off with a note (a design
  file can hold any finite number), and blocks are drawn for 2 to 200 poles (below 3 the pocket
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md`, replace:

````markdown
  drawing.py's message when no screw size
  fits; then the clamp table: the Shaft clamps summary, the machining steps and the 'Clamp screw
  sizes' table with the recommended size's column in green, each value with its hover text.
- **Session** (`session.rs`, `history.rs`): undo and redo (buttons, Ctrl+Z, Ctrl+Shift+Z, Ctrl+Y)
  of every change to the design, one step per settled edit (a drag, a typed value, a part name
  typed letter by letter), one per arrow nudge and one for a held arrow key's whole auto-repeat
````

with:

````markdown
  drawing.py's message when no screw size
  fits; then the clamp table: the Shaft clamps summary, the machining steps and the 'Clamp screw
  sizes' table with the recommended size's column in green, each value with its hover text.
- **Equation explorer** (`typeset.rs`, `readouts.rs`, `explorer.rs`; spec Addendum A2): every
  displayed value (the dashboard rows, the results-table rows, the geometry callouts, the clamp
  summary and screw table, and a readout line over each plot, `PlotKind::readouts`, M43-11) goes
  through `Readouts::show`: hovering it shows its hover text, then its equation typeset (stacked
  fractions, scripts, √ and ∈ drawn with strokes, a large Σ, `arg max`, delimiters scaled to their
  contents, a brace per `cases`, the `where` lines; a selector compared for equality (= or ≠)
  with a code as its choice's label, an ordering with its numbers (`5 ≤ N_h`), a screw row as its
  size; factors side by side as in `render::plain`), each term in its own colour (`TERM_PALETTE`,
  in the formula tree's order, a ninth term taking the first again, M43-3); from the next frame
  every value and input row of a term of that equation is framed in the term's colour, a Σ's term
  by the harmonics the design sums. A click opens the value in the
  **Equation panel**, docked at the bottom of the centre region and closed until then (its header
  button toggles it; M43-1, M43-2): the equation large, its label, value, workbook cell and the
  corrections it embodies (`corrections_upstream`; the dashboard and table markers stay M4-1's,
  M43-14), the term list (colour, symbol, value in the unit the formula reads, label, and what the
  term is; the colour swatches fade while another equation's value is hovered, its terms being
  the ones marked on screen), the breadcrumb (a long walk shows its last 8 crumbs after "…") and
  "used by". Clicking a term, in the equation or the list, drills
  into it; an input term instead opens its group (or the Assumptions view), scrolls its row into
  view once and frames it (M43-12); a result without a record shows its label, value and cell
  (M43-9). The registry is built once per process (`readouts::registry`, M43-5) and logs
  `magcoupling explorer: N equations`. egui's default fonts lack ϑ, the superscript minus and ∝:
  they are drawn as θ, ¯ and ~ (`glyph_safe`, M43-6), the markup and the notes unchanged; a test
  lays out every equation and checks every character against the fonts, and another checks every
  text run against the record's plain rendering (a choice label only where the formula compares
  for equality).
- **Assumptions** (panel.rs; spec Addendum A3): the inputs side's Assumptions view lists the 14
  assumptions (`engine::assumptions::ASSUMPTIONS`), each with its changed-from-default dot, its
  input rows (the inputs' own rows, so an edit there is an edit like any other; the inputs also
  stay in their groups), its rationale and its source. While any differs from its workbook
  default, the header shows "Assumptions modified: ..." beside "Reset to workbook defaults"
  (`assumptions::reset_to_workbook_defaults`: the design inputs stay; the reset can be undone).
  In the Equation panel an assumption term is tagged, with the dot when changed, a result a
  modified assumption flows into is tagged too, and the open equation names the modified
  assumptions it depends on (`Registry::term_style`, `modified_assumptions_upstream`).
- **Teaching notes** (`explorer.rs`, `diagrams.rs`; spec Addendum A4): the Explain toggle (off by
  default) shows under the open equation's value its reviewed note (`notes::note_for`; a note
  shown by id, a warning's or a start-here step's, goes through `explorer::reviewed_note`), with its
  watch-out line, its diagram painted with egui (the square wave and its harmonics, the flux paths
  with and without back iron, torque against angle, the end fringing, the demagnetization knee
  with load lines, first-order heating) and its sources (M43-10). "Start here" walks
  `notes::START_HERE`, each step opening its equation with Explain on (Previous, Next, Stop).
- **Pickers and warnings** (`pickers.rs`, `dashboard.rs`; spec Addendum A5, A6; M43-4, M43-8): a
  part material's drop-down shows each choice's library row on hover and sums up the material
  picked under it; under each magnet part's text field a "Pick a part" drop-down lists "Custom
  dimensions (manual)" and the 15 library parts (with every correction on: E3's remanence, E19's
  rating and grade), and under each grade's a "Pick a grade" drop-down lists "Blank" and the 17
  grades, each with its row on hover; the text fields stay for typed names. The material
  warnings that fire show at the top of the dashboard in their severity's colour, each with a
  "Why: <note>" link that opens its reviewed note on its own in the Equation panel, with links to
  the equations it explains.
- **Session** (`session.rs`, `history.rs`): undo and redo (buttons, Ctrl+Z, Ctrl+Shift+Z, Ctrl+Y)
  of every change to the design, one step per settled edit (a drag, a typed value, a part name
  typed letter by letter), one per arrow nudge and one for a held arrow key's whole auto-repeat
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md`, replace:

````markdown
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it opens
`/magcoupling/` through a pinned share link and checks the canvas, the geometry view in the first
screenshot (the default view: no click), the console lines
`magcoupling: loaded the design from the share link` and `magcoupling sizing: Solved at`, then
clicks "Load design" once (found in a screenshot), uploads a pinned design file through rfd's web
picker and checks `magcoupling: loaded a design file`, with zero console errors (a test decodes
the pinned link and the design file).
````

with:

````markdown
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it opens
`/magcoupling/` through a pinned share link and checks the canvas, the geometry view in the first
screenshot (the default view: no click), the console lines
`magcoupling: loaded the design from the share link`, `magcoupling sizing: Solved at` and
`magcoupling explorer: ` (the equation registry built in the browser), then
clicks "Load design" once (found in a screenshot), uploads a pinned design file through rfd's web
picker and checks `magcoupling: loaded a design file`, with zero console errors (a test decodes
the pinned link and the design file).
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md`, replace:

````markdown
change; one exception, decision G4: with E20 off the outer ring's demagnetization block is
computed anew for the per-ring results, and it governs nothing and moves no workbook cell). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. Formatted verdicts are stated in the markup too (`concat`, `fmt`, `fmtnum`), with `ceilto` and `floorto` for Excel's CEILING and FLOOR and the literals `inf` and `nan`, so no record needs a Rust closure. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). The A4 teaching notes are static data (`notes.rs`), each listing the equations it explains; the panel shows a note only once a physics reviewer has signed it off (`note_for`); 17 of the 17 are signed off (plan A-3 Task 16; six of them revised after an independent physics review on 2026-10-01 and signed off again). There are 17: the spec's list (harmonics, the back-iron factor, pull-out against angle, end effect, Br(T), demagnetization, slip loss and skin depth, the thermal time constant, clamp preload, and the physics behind each of the six A5 warnings) plus ferrite's cold side and the one-point calibration; `START_HERE` opens them in the spec's order (torque chain, back iron, temperature, demagnetization, slip heating, clamps). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.

| Chain (decision 31) | Records | Status |
|---|---|---|
````

with:

````markdown
change; one exception, decision G4: with E20 off the outer ring's demagnetization block is
computed anew for the per-ring results, and it governs nothing and moves no workbook cell). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. Formatted verdicts are stated in the markup too (`concat`, `fmt`, `fmtnum`), with `ceilto` and `floorto` for Excel's CEILING and FLOOR and the literals `inf` and `nan`, so no record needs a Rust closure. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). The A4 teaching notes are static data (`notes.rs`), each listing the equations it explains; the panel shows a note only once a physics reviewer has signed it off (`note_for`); 17 of the 17 are signed off (plan A-3 Task 16; six of them revised after an independent physics review on 2026-10-01 and signed off again). There are 17: the spec's list (harmonics, the back-iron factor, pull-out against angle, end effect, Br(T), demagnetization, slip loss and skin depth, the thermal time constant, clamp preload, and the physics behind each of the six A5 warnings) plus ferrite's cold side and the one-point calibration; `START_HERE` opens them in the spec's order (torque chain, back iron, temperature, demagnetization, slip heating, clamps). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`. The GUI side
(plan M4-3) typesets the records, hovers and opens them, shows the notes and paints their
diagrams: see [The panel](#the-panel).

| Chain (decision 31) | Records | Status |
|---|---|---|
````

In `C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/src/app.rs`, replace:

````rust
        want_file.inputs.metal.face_gap_mm = 2.0;
        assert_eq!(file, want_file);
        assert!(script.contains(DESIGN_FILE_LOADED));
        // The app opens the link and the solve succeeds after its debounce.
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
````

with:

````rust
        want_file.inputs.metal.face_gap_mm = 2.0;
        assert_eq!(file, want_file);
        assert!(script.contains(DESIGN_FILE_LOADED));
        // The equation registry's log line, built when the panel starts.
        assert!(script.contains(crate::gui::readouts::REGISTRY_LOG_PREFIX));
        // The app opens the link and the solve succeeds after its debounce.
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
````

- [ ] **Step 2: Check the docs and the smoke test**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --features app --lib the_smoke_test_share_link_opens_its_design 2>&1 | grep -E "test result"
python -c "import yaml; [yaml.safe_load(open('C:/Users/Cole/source/repos/lsim-mag-m43/docs/ai/' + f, encoding='utf-8')) for f in ('03-structure.yaml', '04-memory.yaml')]; print('yaml ok')"
grep -c "M43-" C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/README.md
grep -c "explorer_ready" C:/Users/Cole/source/repos/lsim-mag-m43/.claude/workflows/gui-smoke.js
```

Expected: `test result: ok. 1 passed; 0 failed`; `yaml ok` (the system Python 3.12, which has PyYAML; the oracle venv has none. `02-system.yaml` has older plain scalars YAML rejects, recorded in plan M4-2; this task's lines there are quoted); `11` (the README's decision ids); `5` (`explorer_ready` in the schema's required list, its property, the prompt's rule, the `passed` rule and the fallback).

- [ ] **Step 3: Copy this plan into the repository**

Run:

```bash
mkdir -p C:/Users/Cole/source/repos/lsim-mag-m43/docs/superpowers/plans
cp C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/m43-plan/plan.md C:/Users/Cole/source/repos/lsim-mag-m43/docs/superpowers/plans/2026-10-01-magcoupling-m4-3-explorer-assumptions-notes-pickers.md
head -1 C:/Users/Cole/source/repos/lsim-mag-m43/docs/superpowers/plans/2026-10-01-magcoupling-m4-3-explorer-assumptions-notes-pickers.md
```

Expected: `# Magcoupling M4-3: Equation Explorer, Assumptions, Teaching Notes, Material and Grade Pickers Implementation Plan`. (If the scratchpad copy is gone, the controller supplies the plan file it executed from.)

- [ ] **Step 4: Run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/Cargo.toml --check
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m43/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/target/gate-task7.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/target/gate-task7.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/target/gate-task7.log
sed -n '/== gate 7\/12/,/== gate 8\/12/p' C:/Users/Cole/source/repos/lsim-mag-m43/magcoupling-rs/target/gate-task7.log | grep -E "test result: ok\. [0-9]+ passed" | head -1
git -C C:/Users/Cole/source/repos/lsim-mag-m43 checkout -- docs/chebyshev_lambda
```

Expected: `cargo fmt --check` prints nothing; `exit=0`; the last three lines end with `GATE PASS` (as Task 0's); the `grep -c` prints `0`; the `sed` line, gate 7's library run, reads `test result: ok. 424 passed` (gate 4's engine runs keep Task 0's counts).

- [ ] **Step 5: Build the web bundle and open it**

Run:

```bash
bash C:/Users/Cole/source/repos/lsim-mag-m43/linkage-sim-rs/scripts/build_magcoupling_web.sh 2>&1 | tail -4
```

Expected: `Magcoupling build complete!` and the two output files (the wasm about 4.8 MB). Then the controller serves `C:/Users/Cole/source/repos/lsim-mag-m43/linkage-sim-rs/web` in the background (`cd C:/Users/Cole/source/repos/lsim-mag-m43/linkage-sim-rs/web && python -m http.server 8080`: `serve_web.sh` refuses to start without the linkage bundle, which this plan does not build), and with Playwright (load the tools via ToolSearch) opens `http://localhost:8080/magcoupling/` at 1280 x 800: the console holds `magcoupling explorer: 392 equations` and no error; hovering the dashboard's pull-out value ("2.688 N·m") shows its hover text with `T_pull = T_2D f_end f_cal` typeset in colour; a click on it opens the Equation panel at the bottom of the centre region (the breadcrumb `T_pull`, the equation large, "Embodies corrections: E7, E8, E3", the three terms); Explain shows the "Pull-out torque against rotation angle" note with its torque-angle diagram. Stop the server and its process afterwards.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m43 add magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md .claude/workflows/gui-smoke.js magcoupling-rs/src/app.rs docs/superpowers/plans/2026-10-01-magcoupling-m4-3-explorer-assumptions-notes-pickers.md
git -C C:/Users/Cole/source/repos/lsim-mag-m43 commit -F - <<'EOF'
docs(magcoupling): M4-3 docs, the web smoke's explorer line and the plan

README (status, layout, the panel's explorer, assumptions, notes, pickers and
warnings), docs/ai (02-system responsibility, invariants and egui lessons;
03-structure gui modules; 04-memory: the M4-3 item and plan A-3's M4 and
typesetting items resolved, per-row greying deferred (M43-15), no renames
(M43-13), the ignored release-gate test passing; 05-update-tracker). gui-smoke checks
the "magcoupling explorer: " console line (no click), which the smoke test
ties to REGISTRY_LOG_PREFIX. The plan, decisions M43-1 to M43-15.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-m43 status --short
```

Expected: one new commit on `magcoupling/m4-3`; `status --short` prints nothing. The branch is then reviewed as a whole (session model) and merged by the controller with the user's go; nothing is pushed.

---

## Self-review record

Checked against the spec with the plan in hand (writing-plans "Self-Review"):

- **Spec coverage.** A2: the typesetter (Task 1: fractions, scripts, Σ, √, no LaTeX); hover on every displayed value with coloured terms (Task 2: dashboard, results table, geometry callouts, clamp tab, plot readouts) and the same colour marking each term's value and input row (Task 2); the docked, toggleable Equation panel with the equation large, terms with values and units, breadcrumb, "used by", drill-down by clicking a term, leaf terms highlighting their rows, one equation at a time (Task 3); the registry built once and hover texts built lazily (Tasks 2, 3). A3: the toggle panel separating assumptions from design inputs with value, unit, rationale and source, the changed-from-default dot, the banner with "reset to workbook defaults", assumption terms styled in the panel (Task 4). A4: the Explain toggle, hidden by default, following the open equation, reviewed notes only, the diagrams, the start-here order (Task 5). A5/A6: per-part material pickers with their properties, grade and part pickers with custom dimensions, the six warnings plain-language, colour-coded and linked to their notes (Task 6). "Addendum testing / GUI (headless egui)": hover shows the equation (`hovering_a_dashboard_value_shows_its_equation`, `a_hovered_value_shows_its_text_and_equation_and_a_click_asks_for_it`), drill-in and breadcrumb (`clicking_a_value_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns`), Explain shows the current equation's note (`the_explain_toggle_shows_the_open_equation_s_note_and_follows_it`), the banner on change and reset (`the_assumptions_banner_appears_on_change_and_clears_on_reset`), a warning linking to its note (`a_warning_shows_in_its_colour_and_links_to_its_note`), the typesetter's glyph order (`a_fraction_a_subscript_a_sum_and_a_root_are_laid_out_in_reading_order`). The gui-smoke still passes and gains a console line, no coordinate click (Task 7).
- **04-memory's M4-3 items.** The dashboard's hover text is built lazily (Task 2, `DashboardLine::tooltip`); the registry is built once (M43-5); the stale-symbol notes and the typesetting pass are decided as "no renames" (M43-13) and recorded; the per-row greying is deferred with its reason (M43-15); the markers stay M4-1's and the panel lists `corrections_upstream` (M43-14); plan A-3's M4 list (choice labels, screw-size names, term styles, the six diagrams, reviewed notes only) is implemented in Tasks 1, 3, 4, 5.
- **Placeholders.** None: every code step is a Create or a replace block generated from the developed tree and replayed literally; every command has its expected output from the replay.
- **Type consistency.** The Interfaces blocks were written from the final code; the replay compiles each task alone on top of the previous ones (`same tree as the commit` after every task), so a name a later task uses is defined by then.
- **Review Focus.** Eight input classes (six in the first draft, two added by the critic revision below), each pinned by a named test in its task.
- **Known limits, stated for the reviewer.** ∈ under Σ is a small stroked arc (it reads close to ε at 13 points); egui's ComboBox popups in the browser open on a press and release slower than Playwright's default click (a human click is fine; the smoke clicks no combo); the Equation panel's 320-point start leaves the geometry view shorter while it is open (resizable; M43-1).

## Self-review record: critic revision

The critic's review of the first draft found one blocking defect and fifteen further items. Each is fixed in the task that owns the code (folded into that task's commit in the scratch tree, then regenerated and replayed: the Verification record), or, where the critic offered "do nothing" or "optional", decided and recorded:

1. **Blocking: a selector's label in an ordering** (Task 1). `Setter::cond` swapped any number compared with a selector for that selector's choice label, so the 12 harmonic records `model.tau#_Pa` and `calibration.tau#_Pa` drew `if "1, 3, 5 (workbook)" ≤ N_h` where the record says `5 ≤ N_h`. Now only `=` and `≠` label (`let labelled = matches!(op, RelOp::Eq | RelOp::Ne)`, with the critic's comment); `a_selector_code_shows_its_choice_and_a_screw_row_its_size` lays out `model.tau5_Pa` and asserts no quoted run, amp1's `"steel circuit"` staying the positive control. The rule is qualified ("compared for equality (= or ≠)") in the module doc, both comments, the Settled list, Task 1's prose and commit message, the README, 02-system and plan A-3's item in 04-memory.
2. **Test hardening** (Task 1). `every_equation_draws_only_what_its_record_says` traces every drawn run of every equation to `render::plain` (through `glyph_safe`), to a choice label of a selector the formula compares for equality, to a `screw_sizes` row's size name (missing from the critic's list: the clamp records draw `M4` where the plain text has the row index), or to one of three typesetter-only words (`and`, `arg max`, `0 ≤ φ ≤ π/2`; the critic's other tokens already occur in the plain text, so the list stays as short as the records allow). It failed on the unfixed typesetter (`model.tau1_Pa: "\"1\""`) and passes after the fix.
3. **Tooltip and mark colours agree** (Task 2). `hovering_a_value_marks_its_terms_where_their_values_are_shown` shows tooltips at once before the hover and asserts the tooltip's `wall` run is drawn in t_wall's mark colour.
4. **Draft notes gated where they are drawn** (Tasks 5, 6). `explorer::reviewed_note` (the critic's helper) serves the header's note crumb, the note opened alone, `start_here_text` and `dashboard::warning_note`; `a_note_by_id_shows_only_once_reviewed` checks it against every note's review and an unknown id. `release_notes_are_reviewed` passes with `--ignored` on the replayed tree; 04-memory gets an open line asking the engine owner to un-ignore it (this plan edits no engine test).
5. **Palette wrap** (Task 1). Kept (M43-3). The docs now say "in the formula tree's order" (`TERM_PALETTE`, `TermColors::of`, M43-3, the README), naming the `cases` condition coming before its value, and `a_ninth_term_takes_the_first_colour_again` pins heat capacity's first term (the condition's `materials.circuit_backiron`) and ninth (`temperature.thermal.steel_c`) to `TERM_PALETTE[0]`, its second and tenth to `TERM_PALETTE[1]`.
6. **Deep breadcrumbs** (Task 3). The header shows the last `SHOWN_CRUMBS` (8) crumbs after `ELIDED` ("…", whose hover counts the steps left out). `a_long_trail_shows_its_last_crumbs_and_draws_in_a_short_window` drills 36 distinct equations: at 1280 x 1024 and at 1280 x 240 the last crumb, one "…" and eight ">" are drawn ("…" checked against the fonts); at 640 x 240 the side panels leave the centre region about 20 points and the panel's heading is not drawn at all (a probe showed it), so there the test asserts no panic and no edit.
7. **NaN values in the panel** (Task 3). `a_harmonic_set_outside_the_choices_lists_only_its_selector` (max_harmonic 4: τ is NaN, the term list keeps N_h and hides every σ_n, the design unchanged) and `an_undefined_value_shows_its_equation` (manual magnets, the coercivity typed with β = +0.005: the torque at the limit is asserted NaN and the record's "undefined" arm is drawn once).
8. **A material code from a file outside the choices** (Task 6). `a_material_code_outside_the_choices_is_shown_and_kept`: back iron 99 draws "99 (not a choice)", one material summary fewer than the default design draws, and the whole design is unchanged after three frames.
9. **DRY** (Tasks 1, 3, 5, 6). `pickers::text` is not written; `input_ui::current_text` is `pub(crate)` and the pickers call it. `test_support::assert_glyphs` (Task 1, generated into Step 1 with the tests that call it, and added to that step's `git add`) replaces the five glyph loops the tree held: typeset (Task 1), the panel's (Task 3), diagrams and notes (Task 5), pickers (Task 6); the critic counted six.
10. **The "Why" link's glyphs** (Task 6). It draws `format!("{WHY}: {}", glyph_safe(note.title))`, and the warning test clicks the same text; the first record's known limit about it is removed.
11. **One term in two colours** (Task 3). While a value of another equation is hovered (`others_marked`), the term list's swatches fade (`SWATCH_DIM`, 0.35), so the panel's key never contradicts the marks on screen; the open equation's coloured terms keep their colours (the critic's option names the swatches). `the_swatches_fade_while_another_equation_s_value_is_hovered` checks them full, with the same equation hovered, faded, and full again when the pointer leaves. M43-3 says so.
12. **× after a factor: not done, decided.** The critic marked it optional and noted it matches `render::plain`. Drawing × would make the typesetter and the plain text differ in 11 records (the engine cannot follow: no engine edits) and would put × on item 2's typesetter-only list, weakening the check that catches item 1's class of defect. The Settled list and the module doc state the rule: factors side by side as `render::plain` writes them, × only between two numbers.
13. **04-memory's A-3 items** (Task 7). Plan A-3's "M4" item and the "M4 typesetting pass" are marked RESOLVED by M4-3, naming M43-5 (a process-wide OnceLock, not the app state) and M43-13 (no renames) where they supersede the item.
14. **Model tiers.** Task 1's per-task review runs on the session model (`// session model: typesetter restates every equation`): the Process line, Global Constraints and Task 1's Model line.
15. **Verification record.** Re-run in full rather than amended by a sentence: library counts 377, 384, 401, 406, 415, 424, 424; Step 2's red counts 83, 47, 73, 24, 65, 41; the whole gate on the replayed tree; the browser run predates the revision, as the record says.

Checks after the revision: every block applies uniquely and the replayed tree equals the developed one after every task; `cargo fmt --check`, both clippys with `-D warnings`, the wasm32 clippys and the wasm32 check pass after every task; no `Cargo.toml`, `Cargo.lock`, engine, test-data or Python file changes; Task 7 Step 2's counts still hold (11 README lines naming a decision, 5 `explorer_ready`); no new test reads a source file; the hardening test and the σ_5 assertion were run red against the unfixed typesetter before the fix.
