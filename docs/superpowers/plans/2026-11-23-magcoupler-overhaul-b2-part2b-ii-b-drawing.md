# Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-b (the drawing's selections and the word budget): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ] `) syntax for tracking.

**Goal:** Every result is a selection (the design's 5.1 and 5.2). A click on a dimension tag of the drawing, on a row of the callout list or on a results-table row (or Enter or Space on it focused) selects its result, and the Explain panel shows its entry: its value and name, what it is, the check of its row, the stored-3D line where it applies, its lesson, its "Equation" link and, with spreadsheet details on, its cell (5.3). Such a click no longer opens the Equation panel; the entry's Equation link does (5.5). With the tags clickable, the callout list and the notes under the drawing leave the opening tab for the Geometry tab (4.4; part 1's decision D-19), and the three dashboard rows of part 1's D-17 get their home: one click on their results-table row, one more on their lesson. The first view draws 200 words at 1280 x 800, held to the design's 250 (goal G-1, test 8). The gui-smoke run clicks a dimension tag and reads its entry. Nothing that is on screen today is lost (the nothing-lost table).

**Architecture:**
- **A selecting readout.** `Readouts::select` and `select_over` are `show` and `show_over` for a value whose click selects: a click, or Enter or Space on the focused value (egui's click on a focusable sensor), sets `ReadoutEvents::selected` to the result's path; the keyboard's focus arriving sets nothing (BL-093's rule); the hover is the same text and equation without `OPEN_HINT`. The dimension tags (`geometry_view::paint_view`, their click area now the line and the tag's plate), the callout rows (`list_ui`) and the results-table rows (`results_table::row_ui`) draw through them. Every other readout keeps `show` (decision D-61).
- **The panel.** After the frame, `events.selected` sets `explain::result_selection(path)`: the check whose own result it is (a check's verdict row), else `Selection::Result(path)`. The click's trace stays as it was (decision D-62): a result selected traces the inputs it reads unless an input's trace marks it. `Explorer::end_frame` opens only `events.clicked`, so the Equation panel stays as it is.
- **Any result's entry.** `explain::result_ui` draws any result (decision D-64): a tile's number and label as the tile draws them, another's value and name as its results-table row draws them (`value_with_unit` of `shown_text`, `display_label`); "What it is" (its help); "What a change does" (the torque card, the Hot-low tile's result only); "Checks it touches" (its dashboard row's check); "Watch out" with `STORED_3D_LABEL` (and its hover) for `STORED_3D_ROWS`; the foot: "Full lesson:" where `ROW_LEARN` has its row, "Equation" where the registry has its equation, the cell with the details on.
- **The opening tab's drawing.** `geometry_view::drawing_ui` draws the drawing with its tags and, under it, only the notes that carry a level (a part not drawn, a design check's caution), drawn while true (decision D-66); `geometry_ui`, the Geometry tab's, keeps the callout list and every note. Both are `view_ui` with or without the list.
- **Unchanged:** the engine, the exports, the top bar, the Size column, the tiles, the check list, an input's entry, the note entries, the Equation panel, the inputs side and its trace banner, the clamp and plot readouts, the linkage app.

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32, egui_plot 0.33; the headless egui harness (`magcoupling-rs/src/gui/test_support.rs`, the panel tests' `Harness`, the explain tests' `frames` and `click`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. This part is its sections 4.4 (the callout list and the notes to the Geometry tab; a click on a tag selects it), 5.1 to 5.3 for results (a result as a selection; "a click on ... a dimension tag, a results-table row or a callout row selects that result or check"; the result column of 5.3), 5.5 (the Equation panel changes only through its own controls or an "Equation" link; a selection made elsewhere does not re-point it), goal G-1 and test 8 of section 8.2 (250 words), test 10 for these targets, and the docs and release rules of section 9. Executors read the spec's sections 4.4, 5, 8.2, 8.3 and 9 beside this plan. Part 2b-ii-a's plan (`docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, "What follows", part 2b-ii-b items 1 to 4) sets the scope. Backlog item: **BL-094** (`docs/ai/backlog.yaml`, filed with this plan).

**How this plan was made.** The part was built once as a prototype on a copy of `main` at `5645506` (BL-093 merged), one commit per task, then BL-094 filed and the docs swept. Every block below was generated from those commits' diffs (a code task's red step is its test hunks, its green step the rest; Task 3's red step is the prompt's guard in `app.rs` and the verdict test, its green step the prompt), then applied, in this plan's order, with the apply tool, to a fresh copy of `main` with BL-094 filed. Each block applied once; each red step failed as its "Expected" says; the result equals the prototype file for file (line endings aside) and passes `cargo test` with and without the app feature (1111 lib tests with it against 1104 on the base), both native clippy runs after each task, both wasm32 clippy runs, `cargo fmt --check` and the gui-smoke verdict test (`all 81 checks hold`). Each count after a task is the replay's. Not run for this plan: gates 1 to 3 (linkage-sim-rs, untouched), gate 12 (the Python oracle), the release WASM build, the browser and the pictures. Task 5 runs them.

**Where this part applies.** On `main` at `5645506` with this plan's commit (the plan and BL-094), and nothing else that touches `magcoupling-rs/`, `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` or Task 4's nine docs. If a block does not match, stop and report: the controller regenerates the blocks (the planner's tools are listed under "Before Task 1"), it does not adapt a block in place.

## Why one plan, and what follows

Part 2b-ii-a's plan listed four items for this part. The prototype holds them in three code tasks (about 1150, 360 and 230 plan lines), each smaller than part 2b-ii-a's first two, and one docs task: one reviewer can hold them, so this plan does not cut again (decision D-60). Task 1 is the selections and the entry, Task 2 the opening tab's drawing and the word budget, Task 3 the smoke step.

**What follows, each with its own plan and backlog item** (unchanged from part 2b-ii-a's list):

**Part 2b-ii-c (the left column and the nothing-lost test):**
1. The left column no longer changes with the centre tab (4.3): the Size column on every tab, the workflow groups only inside Detailed inputs; the trace's banner and Clear trace placed with it (part 2b-ii-a's D-52); the `InputsView` tabs go.
2. The full table-driven nothing-lost test of test 9 (every top-bar action, every `CentreView`, every shown note in Learn, every `DASHBOARD` row's home, every input in the Size column or Detailed inputs, the Assumptions view from the opening tab). This part's table-row selections give every `DASHBOARD` row an entry one click from its Headline row.

**Part 2b-iii (the Plots tab and the look)**, unchanged: the Plots tab with its picker (C-9), the plots' red reference lines (C-3), egui_plot's fading tick numbers (E-7), the controls' outlines (E-4, R-6), a SemiBold face (S-1), the page shell's loading line (S-4), BL-086, BL-088, BL-089. Whether the plots' readouts select their result (owner question 56) is that plan's to weigh. BL-083, BL-084 and BL-091 stay open.

## What the prototype showed

1. **The tag was clickable only near its line.** A dimension's click sensor was its line, expanded by 6 points; the tag sits 9 points past the line's end. egui's interact radius (5 points) let a click on the number's centre reach the line, but a click on the tag's far part hit nothing. The sensor is now the line's rect joined with the tag's plate (`plated_text` returns the plate). The geometry tests that hover a dimension's sensor still pass.
2. **Five tests opened the Equation panel by a click.** `clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group`, `clicking_a_value_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns`, `hovering_a_headline_value_shows_its_equation`, `reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows` and `the_panel_works_inside_an_egui_window` clicked a results-table value and expected the Equation panel open (or its hint in the hover). Each now selects, then opens the panel through the entry's "Equation" link, and keeps every assertion about the trace, the trail and the terms.
3. **A paused clamp's result.** The first result with no equation record in schema order is `clamps.recommended`, which the table hides while the clamp section is paused; the tests pick results the screen shows (`clamps::hides`).
4. **The starting design draws no amber note.** Its geometry notes are the wall rule (no level); the example design adds the cap thread caution ("Design check: Cap thread nominal diameter 41.00 mm is below Steel cup body OD 41.33 mm"), which stays under its opening tab's drawing (D-66). At 1024 x 700 it wraps to two rows, so the opening tab leaves each such note its height wrapped at the drawing's width, not a row and a half (the plan review's UXT-1: the second row was cut by 4.4 points).
5. **200 words.** With the callout list off the opening tab the first view draws 200 words at 1280 x 800 (272 before), so the design's 250 holds with room; no walk title is needed (D-67).
6. **Keyboard.** A selecting readout is egui's focusable click sensor, so Tab reaches it, the focus arriving selects nothing, and Enter or Space selects (a unit test drives all three). The rows were Tab stops before; the order does not change.
7. **`main` is not rustfmt-clean.** BL-093 left two hunks unformatted (`panel.rs`'s keyboard test and `size_controls::touched`); Task 1's `cargo fmt` reformats them and its blocks carry the result (decision D-70).

No pictures were drawn for this plan: Task 5's review renders a tag's entry, a table row's entry and the opening tab at the three window sizes.

## Global Constraints

- **No engine number moves** (the design's 2.3). No file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` changes.
- **The linkage app is not touched** (O-1). No file under `linkage-sim-rs/src/` or `linkage-sim-rs/web/` changes.
- **The panel sets no theme and no font** (2.2, 6.3). Code under `magcoupling-rs/src/gui/` reads `ui.visuals()`.
- **No new explanatory text** (5.3), and no new on-screen string at all: a result's entry draws its value and name as its table row does, its help, its check's mark and name, `STORED_3D_LABEL`, `FULL_LESSON` and its link, `EQUATION`, `SPREADSHEET_CELL` and the cell. One line leaves three hovers: `OPEN_HINT` ("Click to open it in the Equation panel") on a tag, a callout row and a table row, where it would be false.
- **Crimson is never text, never on a value, never in the check list** (C-5). An entry's link to a check draws the check's state mark and word.
- **The selection is view state** (5.1): a result selected is in no design file, share link or undo step, and no click on a tag, a callout row or a table row edits the design.
- **A hover selects nothing** (C-2), and neither does the keyboard's focus arriving (BL-093): only a click, or Enter or Space on the focused value.
- **The exports do not change** (12.1).
- **The blocks.** Each Replace block's old text occurs exactly once in its file at the moment its step applies it. Apply a step's blocks in the order given, top to bottom. Keep each file's line endings. If an old text is not found, or is found twice, stop and report: do not adapt the block. Line numbers are not given because earlier steps move them. `C:\Users\Cole\source\repos\linkage_simulation\.superpowers\sdd\2026-11-22-magcoupler-overhaul-b2\apply_plan.py` (gitignored, in the main checkout; stage 1's tool, unchanged) applies one step's blocks this way: `python <it> <this plan> <task> <step>`, from the worktree's root. The replay used it.
- **TDD.** Red, then green. A red step that calls an API the green step adds fails to compile; that is its red, and the plan says so.
- **Before each commit**, from the repository root:
  ```bash
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
  ```
  After `cargo fmt`, `git status --short` shows only the files the task names. If it shows another, stop and report.
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2biib/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Shell.** On this machine `bash` typed in PowerShell is WSL, which fails on the checkout's line endings. Shell scripts run through Git Bash: `& 'C:\Program Files\Git\bin\bash.exe' <script>`. Python scripts are written with the Write tool, not a here-document.
- **Never run `cargo fmt` inside `linkage-sim-rs`.** `magcoupling-rs` is fmt-clean after Task 1; keep it so.
- **Docs ship with the code** (repository rule): Task 4 sweeps the README and `docs/ai/`, the register, the owner-question list and the design's status line.

## Model per task

CLAUDE.md section 5. BL-094 is `risk: mechanical` (decision D-69): the part shows no new number, makes no new physical claim and adds no on-screen string; every line a result's entry draws is reviewed text the screen drew. So the tasks that transcribe the plan's exact blocks run on `sonnet`, and every review and the release run on `fable`. A `sonnet` task that ends blocked, leaves the gate red or is rejected in review is redone on `fable` (escalate, never downgrade).

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | Any result's entry; a tag, a callout row and a table row select their result (`Readouts::select`, `select_over`, `ReadoutEvents::selected`, `result_selection`, `result_ui` for any result, the tag's click area) | `sonnet` |
| 2 | The callout list and its notes to the Geometry tab; the first view at most 250 words (`drawing_ui`) | `sonnet` |
| 3 | gui-smoke reads a dimension tag's entry (`tag_entry`) | `sonnet` |
| 4 | Docs sweep | `sonnet` |
| 5 | Release: the full gate, gui-smoke, the text review, the independent whole-branch review, one fix wave, the merge and the push | `fable` controller; fresh-context `fable` reviewers; `haiku` for git; the smoke workflow's own agents on `sonnet` |

Each task's diff is reviewed by a fresh-context reviewer before the next task starts (`sonnet` for these mechanical diffs, CLAUDE.md section 5); it checks the diff against the task's blocks and nothing else. A task that ends blocked or leaves a test red stops the build and goes back to the controller.

## Review Focus

Ten inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named.

1. **A click on the tag's number.** The number sits past its line's end; the click area joins the two. Task 1: `a_click_on_a_tag_a_callout_or_a_table_row_selects_its_result_and_opens_no_equation` clicks the "1" on the drawing at its edge farthest from the line, past the line's sensor and egui's 5-point interact radius (a click on the number's centre reaches the line's sensor alone, so it would not catch a click area without the tag).
2. **The open Equation panel.** A selection made elsewhere leaves it where it is (5.5): with the panel open at the pull-out, a click on the cup OD's row selects the cup OD and the trail stays. Task 1: the same test.
3. **The keyboard.** Enter or Space on a focused tag, callout row or table row selects; the focus arriving does not (BL-093). Task 1: `a_selecting_value_asks_for_the_explain_panel_by_click_or_key_and_not_on_focus`.
4. **A check's own row.** A results-table row whose result is a guide's check (its verdict) selects the check, not a result entry that would repeat it thinly. Task 1: `a_result_that_is_a_check_s_own_selects_the_check_and_any_other_the_result`.
5. **A result with no equation record.** Its entry has no "Equation" link; its click keeps an input's trace. Task 1: `a_result_s_entry_is_its_value_label_help_check_watch_out_lesson_and_equation` (one such result), `reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows`.
6. **A tile's result from the table.** The pull-out's row selects the Pull-out tile's entry (one key, as a one-input Size row and its Detailed row are one entry); the Hot-low row's entry reads the consequences for its card. Task 1: the panel test above; `Selection::reads_consequences` unchanged.
7. **A stored-3D row.** Its entry says, under "Watch out", that it reads fixed 3D fields (the line and its hover the check entries draw). Task 1: the entry test.
8. **The three rows no check holds** (D-17). Their entry's lesson line leads where the dashboard's Why link did; the cup OD's says no note covers it yet. Task 1: `a_result_s_equation_link_asks_for_its_equation_and_its_lesson_for_its_note`.
9. **An amber note on the opening tab.** It stays under the drawing while true, drawn whole where it wraps (the example design's cap thread caution takes two rows at 1024 x 700); the plain wall-rule note and the callout list move. Task 2: `the_opening_tab_s_drawing_keeps_its_tags_and_only_the_notes_with_a_level`, `the_first_view_keeps_its_three_columns_its_tiles_and_the_fixed_3d_line_in_the_app_s_own_font`.
10. **The paused clamp.** No entry this part adds names the clamp: a clamp result is hidden from the table and the drawing while paused. Task 1: `a_result_s_entry_text_has_glyphs_and_names_no_paused_clamp`.

## Decisions this plan takes

Each is a reading of the design where it was open, numbered after part 2b-ii-a's D-49 to D-59 (one stage, one list). The owner may overrule any of them; Task 5 lists them in its report. The ones that need a row in the register have one (Task 4).

| Id | Decision | Why |
|---|---|---|
| D-60 | Part 2b-ii-b is one plan: part 2b-ii-a's four items (the selections and any result's entry, the callout list's move, the word budget, the smoke's tag entry). | The prototype's three code tasks are each smaller than part 2b-ii-a's first two; one reviewer can hold them. |
| D-61 | A click on a dimension tag, a callout row or a results-table row (or Enter or Space on it focused) selects its result and opens no equation; the entry's "Equation" link opens the Equation panel, and the selection stays. Those three hovers lose `OPEN_HINT`. Every other readout (the numbers behind a check in its entry, the plots' readouts, the end-effect banner, the paused clamp's) still opens the Equation panel and keeps its hint. Register row A-33; owner question 56. | 5.2 names the three; 5.5: "the Equation panel changes only through its own controls or an Equation link". Changing the others would reach surfaces this part does not otherwise touch (a check's entry; the Plots tab, part 2b-iii's). The line goes where it would be false. |
| D-62 | A result selected traces the inputs it reads, as the click did, unless an input's trace marks it or it has no equation record. Register row A-34; owner question 57. | Nothing reachable is lost and no text is new; the design gives a result's trace no link of its own. |
| D-63 | The row of a check's own result (its verdict) selects the check (`result_selection`). | 5.2: "selects that result or check". The check's entry holds the Problem, Why and fixes; a result entry for the verdict would show its word and less. |
| D-64 | Any result's entry: a tile's number and label as the tile draws them; another's value and name as its results-table row draws them; "What it is" (its help); the torque card for the Hot-low tile's result only; "Checks it touches" for its dashboard row's check; "Watch out" with the stored-3D line for `STORED_3D_ROWS`; "Full lesson:" only where `ROW_LEARN` has its row; "Equation" only where the registry has its equation; the cell with the details. | 5.3's result column with what the screen has: the help, the badge's check, "the stored-3D caution where it applies". 5.3 leaves out a heading an entry has nothing for: "no note yet" is the honest text of the dashboard rows without a note (D-42), not of the other thousand results, and an Equation link to a result with no record would open nothing. |
| D-65 | A dimension's click area is its line's and its tag's plate together. | 4.4: "a click on one [tag] selects it"; the tag's far part sat outside the line's sensor and egui's interact radius (the prototype, item 1). |
| D-66 | The callout list and the plain notes (the wall rule) move to the Geometry tab; a note with a level (a part not drawn, a design check's caution) stays under the opening tab's drawing while it is true, given its wrapped height. A callout with a level (a red gap, an axis over its claim) moves with the list: on the opening tab its tag stays red and the check list carries the failure. Register row A-35; owner question 58. | A drawing missing parts with no note saying so would mislead; the amber notes are conditions like the "Exploration only" lines and the wall cautions 4.4 keeps above the drawing. None shows on the starting design. |
| D-67 | The first view is held to the design's 250 words (200 at the starting design). Part 2a's D-32 left the three copies of a Start here step's title to this budget: all three stay. Register row A-36; owner question 59. | The budget counts the first view, where no walk runs, and is met without them; dropping a copy is a text change for the text review. |
| D-68 | The gui-smoke prompt gains `tag_entry` on its first page, after the input's entry: on the Geometry tab, a click on tag 1 of the drawing; the Explain panel shows a value in mm, the face gap's name (`TAG_LABEL`, typed once, held equal to the app's) and the "Equation" link, and no Equation panel opens. `first_view` adds "small numbers beside its dimension lines and no list of numbered lines under it". | 4.4 and 5.2 in a browser with no new navigation: the step reuses the Geometry tab the tooltip step opened. Part 2b-ii-a's D-58 left the tag's entry to this part. |
| D-69 | BL-094 is `risk: mechanical`; its build tasks run on `sonnet`, its reviews and release on `fable`. | No new number, physical claim or on-screen string (the design's header). |
| D-70 | Task 1's blocks carry rustfmt's reformatting of two hunks BL-093 left unformatted (`panel.rs`'s keyboard trace test, `size_controls::touched`). | "Before each commit" runs `cargo fmt`, which would rewrite them anyway; carrying them keeps the replay exact and `cargo fmt --check` green from Task 1 on. |

## Nothing lost

Every text and action this part moves, and where it is after. Anything not listed (every tile, check row, input, the Size column, the trace's banner and Clear trace, every other readout and its hover, the Geometry tab's list, every note entry) does not move. The "Test" column names the test that holds the new home.

| Today | After this part | Test |
|---|---|---|
| A click on a dimension tag's line, a callout row or a results-table row opens the Equation panel at its result | The click selects the result; its entry's "Equation" link opens the Equation panel there (one more click), the selection staying | `panel::tests::a_click_on_a_tag_a_callout_or_a_table_row_selects_its_result_and_opens_no_equation`, `a_value_s_equation_link_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns`, `the_panel_works_inside_an_egui_window`; `explain::tests::a_result_s_equation_link_asks_for_its_equation_and_its_lesson_for_its_note` |
| Those three hovers end "Click to open it in the Equation panel" | The line is gone there (it would be false; D-61); the text and the equation stay; every other readout keeps it | `hovering_a_headline_value_shows_its_equation`; `readouts::tests::a_selecting_value_asks_for_the_explain_panel_by_click_or_key_and_not_on_focus` |
| A click on a result traces the inputs it reads | The same, from the selection (D-62) | `clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group`, `reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows` |
| The callout list (each dimension's tag and text, with the space claim's values) under the opening tab's drawing | On the Geometry tab, unchanged; on the opening tab each tag selects its result, whose entry shows its value and name | `a_new_panel_opens_on_size_and_torque_with_the_tiles_over_the_drawing`; `geometry_view::tests::the_opening_tab_s_drawing_keeps_its_tags_and_only_the_notes_with_a_level` |
| The notes under the opening tab's drawing | A note with a level stays there while true; the wall-rule note is on the Geometry tab | `the_opening_tab_s_drawing_keeps_its_tags_and_only_the_notes_with_a_level` |
| Part 1's D-17: the notes of Pull-out torque at 20 °C, the slip swing at the gearbox input and the cup OD, three clicks or more away | The Results tab, the row, its entry's lesson link | `a_result_s_equation_link_asks_for_its_equation_and_its_lesson_for_its_note` |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/readouts.rs` | `ReadoutEvents::selected`; `Readouts::select`, `select_over`; the module comment; a test | 1 |
| `magcoupling-rs/src/gui/explain.rs` | `result_ui` for any result; `result_selection`; the `Selection::Result` and module comments; tests | 1 |
| `magcoupling-rs/src/gui/geometry_view.rs` | The tags and the callout rows select; a tag's click area joins its plate (`plated_text` returns it) (1); `drawing_ui`, `view_ui` (the notes' wrapped height), `list_ui` without the list; a test (2) | 1, 2 |
| `magcoupling-rs/src/gui/results_table.rs` | A row selects (`row_ui` takes the `'static` entry) | 1 |
| `magcoupling-rs/src/gui/panel.rs` | `events.selected` sets `result_selection` and traces; five tests through the Equation link; a new selection test; rustfmt's BL-093 hunk (1); the opening tab draws `drawing_ui`; the first view's list test (2) | 1, 2 |
| `magcoupling-rs/src/gui/size_controls.rs` | rustfmt's BL-093 hunk (D-70) | 1 |
| `magcoupling-rs/src/app.rs` | The first view at most 250 words, and each note with a level drawn whole on it (2); the smoke prompt's guard: `TAG_LABEL`, the tag step after the input's, collected before the reopen (3) | 2, 3 |
| `.claude/workflows/gui-smoke.js` | `tag_entry` (schema, prompt, passed list), `TAG_LABEL`, `first_view`'s words, the description and comments | 3 |
| `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` | `CHECKS` gains `tag_entry` | 3 |
| `magcoupling-rs/README.md`, `docs/ai/*`, the register, the owner questions, the design's header | The docs sweep | 4 |
| `docs/ai/backlog.yaml` | BL-094 in progress (4), fixed (5) | 4, 5 |

**Existing tests this plan edits, and how.** No assertion about behaviour that stays is weakened; where an action moved (the Equation panel from the click to the entry's link), the test takes the new path and keeps every assertion about the trace, the trail and the terms.
- `panel::tests`, Task 1: `clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group` expects the pull-out selected and the Equation panel closed where it expected it open (every trace, banner, frame and group-count assertion kept); `reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows` clicks the traced row (selected, the trace kept), then the entry's "Equation" link (the panel open, the trace and the table's rows kept); `hovering_a_headline_value_shows_its_equation` expects the hover's text and equation and no `OPEN_HINT`; `clicking_a_value_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns` becomes `a_value_s_equation_link_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns` (the click selects, the link opens; every trail and term assertion kept); `the_panel_works_inside_an_egui_window` opens the value through its entry's link. rustfmt reflows two asserts in `a_keyboard_user_traces_a_middle_input_with_tab_or_shift_tab_to_its_entry_s_link` (D-70).
- `panel::tests`, Task 2: `a_new_panel_opens_on_size_and_torque_with_the_tiles_over_the_drawing` expects no callout text on the opening tab, then finds the list under the drawing on the Geometry tab. `app::tests`, Task 2: `the_first_view_draws_at_most_two_hundred_and_eighty_words_at_1280_by_800` becomes `the_first_view_draws_at_most_two_hundred_and_fifty_words_at_1280_by_800`; `the_first_view_keeps_its_three_columns_its_tiles_and_the_fixed_3d_line_in_the_app_s_own_font` also finds each geometry note with a level drawn whole once, at each of its sizes and on both designs.
- `app::tests::the_smoke_test_share_link_opens_its_design`, Task 3: the first page's console messages are collected after "tag_entry=true was judged" (the step judged last on that page), where it read "input_entry=true was judged"; a new block holds `TAG_LABEL`, the tag step's place and its Equation link. `gui_smoke_verdict_test.cjs`, Task 3: `CHECKS` lists `tag_entry`, so its case "tag_entry false with the agent's passed true" runs.

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` after this plan has merged, create the branch `magcoupling/overhaul-b2-part2b-ii-b` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: `test result: ok. 1104 passed; 0 failed; 6 ignored` and `all 79 checks hold`. Another count means `main` is not the base the blocks were checked on: stop and report.

- [ ] **Step 3: Check the base.** `git log --oneline 5645506..main -- magcoupling-rs .claude/workflows/gui-smoke.js linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs docs/ai docs/analyses/2026-11-10-conservative-assumptions.md docs/analyses/2026-11-18-owner-questions.md docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` lists this plan's commit (it touches `docs/ai/backlog.yaml`) and nothing else. If it lists more, stop: the controller regenerates the blocks on that `main` with the planner's tools (gitignored, in the main checkout's `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2biib/`: `gen_blocks.py`, `frags.py`, `replay.py`, `edit_lib.py` with the edit scripts `e1*.py` to `e4.py` that built each task's state, `file_bl094.py`, and `seq.bundle`, the prototype's commits: `s0` the base, `s1` to `s3` the code tasks, `s3b` BL-094 filed, `s4` the docs) before Task 1.

## Tasks

### Task 1: Any result's entry; a tag, a callout row and a table row select their result

**Model:** `sonnet` (BL-094 is `risk: mechanical`; the task transcribes exact blocks).

`Readouts::select` and `select_over` record a click (or Enter or Space on the focused value) as `ReadoutEvents::selected`, with the hover's text and equation and no `OPEN_HINT`; the dimension tags (their click area now the line and the tag's plate), the callout rows and the results-table rows draw through them. The panel sets `explain::result_selection(path)` (a check's own result selects the check) and traces as a click on a result did; the Equation panel no longer opens on that click. `explain::result_ui` draws any result. Five panel tests that opened the Equation panel by a click open it through the entry's "Equation" link.

**Files:**
- Modify: `magcoupling-rs/src/gui/explain.rs`, `magcoupling-rs/src/gui/geometry_view.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/readouts.rs`, `magcoupling-rs/src/gui/results_table.rs`, `magcoupling-rs/src/gui/size_controls.rs` (rustfmt's hunk only, D-70)

**Interfaces:**
- Produces: `ReadoutEvents::selected: Option<&'static str>`; `Readouts::select(&mut self, &egui::Ui, egui::Response, &'static str, impl FnOnce(bool) -> String)` and `Readouts::select_over(&mut self, &egui::Ui, Rect, &'static str, impl FnOnce(bool) -> String)`; `explain::result_selection(&'static str) -> Selection` (`pub(crate)`).
- Changes: `explain::result_ui` draws any result (`None` only for a path with no `result_info`); `geometry_view::plated_text` returns its plate's `Rect`; `results_table::row_ui` takes `&'static TableEntry`.

- [ ] **Step 1 (red): the selections' tests.** The blocks add four tests in `explain::tests` with their helper `selected_results`, the readouts test `a_selecting_value_asks_for_the_explain_panel_by_click_or_key_and_not_on_focus`, the panel test `a_click_on_a_tag_a_callout_or_a_table_row_selects_its_result_and_opens_no_equation`, and take the five panel tests that opened the Equation panel by a click through the entry's link (rustfmt's reflow of the BL-093 keyboard test rides along, D-70).

**`magcoupling-rs/src/gui/explain.rs`**

Replace:

```rust
        assert!(pullout.is_some(), "the pull-out has a cell");
    }
}

```

with:

```rust
        assert!(pullout.is_some(), "the pull-out has a cell");
    }

    // ------------------------------------------------------------------ any result's entry

    /// The results the tests below select: every dashboard row (the four tiles among them),
    /// every dimension of the drawing of `inputs` and the first results-table row with no
    /// equation record, each one the screen shows while the shaft-clamp section is paused.
    fn selected_results(inputs: &DesignInputs) -> Vec<&'static str> {
        use crate::engine::clamps;
        let results = compute_all(inputs);
        let shown = |path: &&str| !clamps::hides(path, clamps::PAUSED);
        let recordless = crate::gui::results_table::table_entries()
            .iter()
            .map(|entry| entry.path.as_str())
            .filter(shown)
            .find(|path| registry().equation_for(path).is_none())
            .expect("a result with no equation record");
        let geometry = crate::gui::geometry::geometry(inputs, &results, clamps::PAUSED);
        DASHBOARD
            .iter()
            .map(|&(path, _)| path)
            .chain(geometry.callouts().map(|callout| callout.path))
            .chain(std::iter::once(recordless))
            .filter(shown)
            .collect()
    }

    #[test]
    fn a_result_s_entry_is_its_value_label_help_check_watch_out_lesson_and_equation() {
        // Spec 5.1 to 5.3 for any result (a dimension tag, a callout row, a results-table row):
        // its value as its results-table row shows it, leading its label; "What it is" (its
        // help); the check of its dashboard row; for a row that reads the stored 3D fields, the
        // line that says so under "Watch out"; its row's lesson where it has one; the Equation
        // link where it has an equation record. A tile's keeps the tile's number and label.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        for path in selected_results(&inputs) {
            let (output, _, _) = frames(&ctx, &inputs, Selection::Result(path), false, vec![]);
            let texts = drawn_texts(&output);
            let has = |text: &str| texts.iter().any(|t| t == text);
            let info = result_info(path).expect("a result");
            let label = match Selection::Result(path).tile() {
                Some(tile) => tile.label(&results),
                None => {
                    let value = results.get(path).unwrap_or(Value::None);
                    let shown = value_with_unit(&value, shown_text(path, &value), info.meta.unit);
                    assert!(has(&shown), "{path}: {shown:?} in {texts:?}");
                    display_label(path, info.meta.label).to_owned()
                }
            };
            assert!(has(&label), "{path}: {label:?} in {texts:?}");
            let help = help_for(path);
            assert_eq!(has(WHAT_IT_IS), !help.is_empty(), "{path}");
            assert!(help.is_empty() || has(help), "{path}: {texts:?}");
            assert_eq!(
                has(CHECKS_IT_TOUCHES),
                badging_check(path).is_some(),
                "{path}"
            );
            let stored_3d = STORED_3D_ROWS.contains(&path);
            assert_eq!(has(WATCH_OUT), stored_3d, "{path}");
            assert_eq!(has(STORED_3D_LABEL), stored_3d, "{path}");
            assert_eq!(has(FULL_LESSON), row_learn(path).is_some(), "{path}");
            let equation = registry().equation_for(path).is_some();
            assert_eq!(has(EQUATION), equation, "{path}");
            assert!(
                !has(WHAT_A_CHANGE_DOES),
                "{path}: no card without the consequences"
            );
        }
    }

    #[test]
    fn a_result_s_equation_link_asks_for_its_equation_and_its_lesson_for_its_note() {
        // Spec 5.5: an entry's Equation link opens the Equation panel at its result, and the
        // selection stays. Part 1's decision D-17: the three dashboard rows no check holds
        // (the pull-out at 20 °C, the slip swing at the gearbox input, the cup's outside
        // diameter) lead from their entry's lesson line where the dashboard's Why link did,
        // one click from their results-table row.
        let inputs = DesignInputs::default();
        const FACE_GAP_RESULT: &str = "model.face_gap_mm";
        let (asked, events) = click(&inputs, Selection::Result(FACE_GAP_RESULT), EQUATION);
        assert_eq!(asked, Some(Asked::Equation(FACE_GAP_RESULT)));
        assert_eq!(events, ReadoutEvents::default());
        for path in [
            "model.pullout_20C_Nm",
            "model.gearbox_input_ripple_Nm",
            "model.cup_od_mm",
        ] {
            assert_eq!(badging_check(path), None, "{path}");
            let learn = row_learn(path).expect("a dashboard row");
            let (asked, events) = click(&inputs, Selection::Result(path), &learn_link_text(learn));
            assert_eq!(asked, None, "{path}");
            let want = match learn {
                Learn::Note(id) => Some(id),
                Learn::Assumption(_) | Learn::None(_) => None,
            };
            assert_eq!(events.note, want, "{path}");
        }
    }

    #[test]
    fn a_result_that_is_a_check_s_own_selects_the_check_and_any_other_the_result() {
        // Spec 5.2: a click on a results-table row "selects that result or check". The row of
        // a check's own result (its verdict) selects the check, whose entry holds its Problem,
        // Why and fixes; any other row selects the result.
        let checks: Vec<&CheckGuide> = CHECK_GUIDES
            .iter()
            .filter(|guide| result_info(guide.check).is_some())
            .collect();
        assert!(!checks.is_empty());
        for guide in checks {
            assert_eq!(result_selection(guide.check), Selection::Check(guide.check));
        }
        for entry in crate::gui::results_table::table_entries() {
            let path = entry.path.as_str();
            if guidance::guide(path).is_none() {
                assert_eq!(result_selection(path), Selection::Result(path), "{path}");
            }
        }
    }

    #[test]
    fn a_result_s_entry_text_has_glyphs_and_names_no_paused_clamp() {
        // Spec 8.3: every text a result's entry draws, with the spreadsheet details on designs
        // where each check fails, has its glyphs in the app's font, and none names the paused
        // clamp.
        let ctx = glyph_context();
        for inputs in designs() {
            for path in selected_results(&inputs) {
                let (output, _, _) = frames(&ctx, &inputs, Selection::Result(path), true, vec![]);
                for text in drawn_texts(&output) {
                    assert_glyphs(&ctx, &text, path);
                    assert!(!text.to_lowercase().contains("clamp"), "{path}: {text}");
                }
            }
        }
    }
}

```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
            }
            assert!(harness.ctx.memory(|m| m.has_focus(target)), "Tab reaches it");
            harness.frame(key_tap(egui::Key::Enter));
```

with:

```rust
            }
            assert!(
                harness.ctx.memory(|m| m.has_focus(target)),
                "Tab reaches it"
            );
            harness.frame(key_tap(egui::Key::Enter));
```

Replace:

```rust
            }
            assert!(on_link, "the keyboard reaches the link (backwards: {backwards})");
            assert_eq!(
```

with:

```rust
            }
            assert!(
                on_link,
                "the keyboard reaches the link (backwards: {backwards})"
            );
            assert_eq!(
```

Replace:

```rust
    #[test]
    fn clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group() {
```

with:

```rust
    #[test]
    fn a_click_on_a_tag_a_callout_or_a_table_row_selects_its_result_and_opens_no_equation() {
        // UX overhaul spec 5.2, 5.5 and 8.2 test 10: a click on a dimension tag (on the opening
        // tab), a callout row (on the Geometry tab) or a results-table row selects its result,
        // and the Explain panel shows its entry; the Equation panel stays closed until the
        // entry's Equation link opens it, the selection staying; nothing edits the design or
        // makes an undo step.
        use crate::gui::explain::EQUATION;
        const FACE_GAP_RESULT: &str = "model.face_gap_mm";
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        let right = right_column_left(&output);
        // Tag 1, the face gap's, on the drawing: the topmost "1" in the centre region.
        let tag = text_rects(&output, "1")
            .into_iter()
            .filter(|r| r.left() > INPUTS_WIDTH && r.right() < right)
            .min_by(|a, b| a.top().total_cmp(&b.top()))
            .expect("tag 1 on the drawing");
        // The click lands on the number's edge farthest from its line, past the line's sensor
        // and egui's interact radius, so only the tag's own click area can take it (D-65).
        harness.click(tag.center_top() + egui::vec2(0.0, 1.0));
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Result(FACE_GAP_RESULT));
        explain_rect(&output, label_for(FACE_GAP_RESULT));
        assert!(!harness.panel.explorer.open, "a tag opens no equation");
        // The corner gap's row of the callout list, on the Geometry tab.
        harness.click_text(CentreView::Geometry.label());
        let shown = harness.panel.shown_inputs();
        let geometry = crate::gui::geometry::geometry(&shown, harness.panel.results(), true);
        let corner = geometry
            .callouts()
            .find(|callout| callout.tag == 2)
            .expect("the corner gap's callout");
        harness.click_text(&corner.text);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Result(corner.path));
        explain_rect(&output, label_for(corner.path));
        assert!(!harness.panel.explorer.open, "a callout opens no equation");
        // The pull-out's row of the results table, then its entry's Equation link.
        harness.click_text(CentreView::Results.label());
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
        assert_eq!(
            harness.panel.selection,
            Selection::Result("model.pullout_Nm")
        );
        assert!(!harness.panel.explorer.open, "a row opens no equation");
        harness.click_text(EQUATION);
        assert_eq!(harness.panel.explorer.trail(), ["model.pullout_Nm"]);
        assert_eq!(
            harness.panel.selection,
            Selection::Result("model.pullout_Nm")
        );
        // A selection made elsewhere leaves the open equation where it is (spec 5.5): the cup
        // OD's row selects the cup OD, and the Equation panel stays at the pull-out.
        let cup_od = harness.panel.results().model.cup_od_mm;
        harness.click_text(&with_unit(format_value(&Value::Num(cup_od)), "mm"));
        assert_eq!(
            harness.panel.selection,
            Selection::Result("model.cup_od_mm")
        );
        assert_eq!(harness.panel.explorer.trail(), ["model.pullout_Nm"]);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group() {
```

Replace:

```rust
        );
        assert!(harness.panel.explorer.open, "the click still opens it");
        let output = harness.frame(Vec::new());
```

with:

```rust
        );
        // The click selects the result (UX overhaul spec 5.2); its entry's Equation link
        // opens the Equation panel (spec 5.5).
        assert_eq!(
            harness.panel.selection,
            Selection::Result("model.pullout_Nm")
        );
        assert!(!harness.panel.explorer.open, "the click opens no equation");
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        // An input traced and "Traced only" ticked: a click on a traced row (the table's main
        // gesture, to read its equation) opens its equation and keeps the input's trace, so the
        // table keeps its rows; a click on a result without an equation record keeps it too.
        let entries = table_entries();
```

with:

```rust
        // An input traced and "Traced only" ticked: a click on a traced row (the table's main
        // gesture) selects it, and its entry's Equation link opens its equation; both keep the
        // input's trace, so the table keeps its rows; a click on a result without an equation
        // record keeps it too.
        let entries = table_entries();
```

Replace:

```rust
        harness.click(row.center());
        let output = harness.frame(Vec::new());
```

with:

```rust
        harness.click(row.center());
        assert_eq!(
            harness.panel.selection,
            Selection::Result("model.pullout_Nm")
        );
        assert_eq!(
            harness.panel.trace.as_ref(),
            Some(&trace),
            "the selection keeps it"
        );
        harness.click_text(crate::gui::explain::EQUATION);
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        assert!(harness.panel.explorer.open, "the click opens its equation");
        assert_eq!(
```

with:

```rust
        let output = harness.frame(Vec::new());
        assert!(harness.panel.explorer.open, "the link opens its equation");
        assert_eq!(
```

Replace:

```rust
        );
        // T_pull = T_2D f_end f_cal, typeset run by run, and the hint.
        for run in ["T", "pull", " = ", "2D", "end", "cal", OPEN_HINT] {
            assert!(texts.iter().any(|t| t == run), "no {run:?} in {texts:?}");
```

with:

```rust
        );
        // T_pull = T_2D f_end f_cal, typeset run by run. No hint that a click opens it: a
        // click on a results-table row selects its result (UX overhaul spec 5.2, 5.5).
        for run in ["T", "pull", " = ", "2D", "end", "cal"] {
            assert!(texts.iter().any(|t| t == run), "no {run:?} in {texts:?}");
```

Replace:

```rust
            assert!(texts.iter().any(|t| t == run), "no {run:?} in {texts:?}");
        }
    }

```

with:

```rust
            assert!(texts.iter().any(|t| t == run), "no {run:?} in {texts:?}");
        }
        assert!(!texts.iter().any(|t| t == OPEN_HINT), "{texts:?}");
    }

```

Replace:

```rust
    #[test]
    fn clicking_a_value_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns() {
        use crate::gui::explorer::TERMS;
```

with:

```rust
    #[test]
    fn a_value_s_equation_link_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns() {
        // A click on a results-table row selects its result (UX overhaul spec 5.2); its entry's
        // Equation link opens the Equation panel there, and the selection stays (spec 5.5).
        use crate::gui::explain::EQUATION;
        use crate::gui::explorer::TERMS;
```

Replace:

```rust
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
        assert!(harness.panel.explorer.open);
```

with:

```rust
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
        assert!(!harness.panel.explorer.open);
        harness.click_text(EQUATION);
        assert!(harness.panel.explorer.open);
```

Replace:

```rust
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.trail(), ["model.pullout_Nm"]);
```

with:

```rust
        assert!(harness.panel.explorer.open);
        assert_eq!(
            harness.panel.selection,
            Selection::Result("model.pullout_Nm")
        );
        assert_eq!(harness.panel.explorer.trail(), ["model.pullout_Nm"]);
```

Replace:

```rust
        // M5 shows the panel in a window of the linkage app. Past its first frames: a value
        // opened, a term followed, a leaf assumption term's row shown (the Assumptions view
        // and its frame), then idle frames. The window keeps its size, the Equation panel
```

with:

```rust
        // M5 shows the panel in a window of the linkage app. Past its first frames: a value
        // selected and opened from its entry, a term followed, a leaf assumption term's row
        // shown (the Assumptions view
        // and its frame), then idle frames. The window keeps its size, the Equation panel
```

Replace:

```rust
        );
        assert_eq!(panel.explorer.trail(), ["model.pullout_Nm"]);
```

with:

```rust
        );
        let output = click(&mut panel, &output, crate::gui::explain::EQUATION);
        assert_eq!(panel.explorer.trail(), ["model.pullout_Nm"]);
```

**`magcoupling-rs/src/gui/readouts.rs`**

Replace:

```rust
        assert_eq!(readouts.finish().focus, Some("metal.cup_wall_corner_mm"));
    }
}

```

with:

```rust
        assert_eq!(readouts.finish().focus, Some("metal.cup_wall_corner_mm"));
    }

    #[test]
    fn a_selecting_value_asks_for_the_explain_panel_by_click_or_key_and_not_on_focus() {
        // UX overhaul spec 5.2 and 5.5: a dimension tag, a callout row and a results-table row
        // select their result. The hover shows its text and its equation without the line
        // that says a click opens the Equation panel; a click asks for the result in the
        // Explain panel and opens no equation; the keyboard's focus arriving asks for nothing,
        // and Enter or Space on the focused value asks as a click does (BL-093).
        use crate::gui::test_support::{key_tap, primary_button};
        const PATH: &str = "model.pullout_Nm";
        let ctx = egui::Context::default();
        tooltips_at_once(&ctx);
        let size = egui::vec2(800.0, 600.0);
        let frame = |input: Vec<egui::Event>, time: Option<f64>| {
            let mut events = ReadoutEvents::default();
            let mut rect = Rect::NOTHING;
            let output = sized_frame_at(&ctx, size, time, input, |ui| {
                let mut readouts = Readouts::default();
                rect = ui.label("2.688 N·m").rect;
                readouts.select_over(ui, rect, PATH, |_| format!("hover text of {PATH}"));
                events = readouts.finish();
            });
            (output, events, rect)
        };
        let (_, _, rect) = frame(Vec::new(), Some(0.0));
        let at = rect.center();
        frame(vec![egui::Event::PointerMoved(at)], Some(0.1));
        let (output, events, _) = frame(Vec::new(), Some(0.2));
        assert_eq!(events.hovered.as_deref(), Some(PATH));
        let texts = drawn_texts(&output);
        assert!(
            texts.contains(&format!("hover text of {PATH}")),
            "{texts:?}"
        );
        for run in ["T", "pull", " = ", "2D", "end", "cal"] {
            assert!(texts.iter().any(|t| t == run), "no {run:?} in {texts:?}");
        }
        assert!(!texts.contains(&OPEN_HINT.to_owned()), "{texts:?}");
        frame(vec![primary_button(at, true)], Some(0.3));
        let (_, events, _) = frame(vec![primary_button(at, false)], Some(0.4));
        assert_eq!((events.selected, events.clicked), (Some(PATH), None));
        // The keyboard: the pointer gone, a fresh focus on the value, then Enter and Space.
        frame(vec![egui::Event::PointerGone], Some(0.5));
        let (_, events, _) = frame(key_tap(egui::Key::Tab), Some(0.6));
        assert_eq!(events.selected, None, "the focus arriving selects nothing");
        assert!(
            ctx.memory(|m| m.focused().is_some()),
            "the value takes the focus"
        );
        for key in [egui::Key::Enter, egui::Key::Space] {
            let (_, events, _) = frame(key_tap(key), Some(0.7));
            assert_eq!(events.selected, Some(PATH), "{key:?}");
        }
    }
}

```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 15
  ```

  Expected: the build fails with 10 errors, the first "cannot find function `shown_text` in this scope": the green step adds what the tests call.

- [ ] **Step 3 (green): the selecting readouts, the panel, any result's entry.**

**`magcoupling-rs/src/gui/explain.rs`**

Replace:

```rust
//! with the links it had; a tile's entry holds its result's help text and, for the hot-low
//! torque, the torque card (decision C-10).
//!
```

with:

```rust
//! with the links it had; a tile's entry holds its result's help text and, for the hot-low
//! torque, the torque card (decision C-10). A dimension tag, a callout row and a results-table
//! row select their result too (spec 5.2): its value and its name as the table shows them, its
//! help, the check of its row, the stored-3D line where its row reads those fields, its row's
//! lesson and its Equation link; a check's own result selects the check.
//!
```

Replace:

```rust
use crate::engine::explain::notes::{NOTES, shown_when};
use crate::engine::meta::{InputSet, Value};
use crate::engine::size_controls::dimensions;
```

with:

```rust
use crate::engine::explain::notes::{NOTES, shown_when};
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::size_controls::dimensions;
```

Replace:

```rust
use crate::gui::dashboard::{
    DOES_NOT_FIT, FITS, Level, MARKER_LEGEND, STORED_3D_LABEL, STORED_3D_NOTE, badging_check,
    learn_ui, line_ui, result_info, shown_lines,
};
```

with:

```rust
use crate::gui::dashboard::{
    DOES_NOT_FIT, FITS, Level, MARKER_LEGEND, STORED_3D_LABEL, STORED_3D_NOTE, STORED_3D_ROWS,
    badging_check, learn_ui, line_ui, result_info, shown_lines, shown_text,
};
```

Replace:

```rust
use crate::gui::plain_text::{
    DIAMETER_HINT_KEY, LENGTH_HINT_KEY, MINIMUM_OD_HINT_KEY, NO_SPREADSHEET_CELL, help_for,
    hint_for, label_for,
};
```

with:

```rust
use crate::gui::plain_text::{
    DIAMETER_HINT_KEY, LENGTH_HINT_KEY, MINIMUM_OD_HINT_KEY, NO_SPREADSHEET_CELL, display_label,
    help_for, hint_for, label_for,
};
```

Replace:

```rust
    Check(&'static str),
    /// A result, by its path: one of the four tiles ([`Tile::path`]).
    Result(&'static str),
```

with:

```rust
    Check(&'static str),
    /// A result, by its path: one of the four tiles ([`Tile::path`]), a dimension tag or a
    /// callout row of the drawing, a row of the results table.
    Result(&'static str),
```

Replace:

```rust

/// The entry of the result at `path`, one of the tiles' (spec 5.3): its number and its name;
/// what it is (the result's help text); for the hot-low torque, what a change does (the
/// torque card, [`card_ui`]); the check that badges its row, a link that selects it; and the
/// foot links, with the spreadsheet details its cell. Nothing for a path that is no tile's.
fn result_ui(
```

with:

```rust

/// The entry of the result at `path` (spec 5.3): a tile's, a dimension's, a callout's or a
/// results-table row's. Its number and its name (a tile's as the tile draws them, another's as
/// its results-table row does); what it is (the result's help text); for the hot-low torque,
/// what a change does (the torque card, [`card_ui`]); the check that badges its row, a link
/// that selects it; for a row that reads the stored 3D fields, the line that says so ("Watch
/// out"); and the foot links (its row's lesson where it has one, the Equation link where it
/// has an equation), with the spreadsheet details its cell. Nothing for a path that is no
/// result.
fn result_ui(
```

Replace:

```rust
) -> Option<Asked> {
    let tile = Selection::Result(path).tile()?;
    ui.horizontal_wrapped(|ui| {
```

with:

```rust
) -> Option<Asked> {
    let info = result_info(path)?;
    let tile = Selection::Result(path).tile();
    ui.horizontal_wrapped(|ui| {
```

Replace:

```rust
    ui.horizontal_wrapped(|ui| {
        match tile.number(shown.inputs, shown.results) {
            Ok(number) => ui.strong(number),
            Err(words) => ui.weak(words),
        };
        ui.colored_label(look::quiet_text(ui.visuals()), tile.label(shown.results));
    });
```

with:

```rust
    ui.horizontal_wrapped(|ui| {
        let quiet = look::quiet_text(ui.visuals());
        match tile {
            Some(tile) => {
                match tile.number(shown.inputs, shown.results) {
                    Ok(number) => ui.strong(number),
                    Err(words) => ui.weak(words),
                };
                ui.colored_label(quiet, tile.label(shown.results));
            }
            None => {
                let value = shown.results.get(path).unwrap_or(Value::None);
                let text = shown_text(path, &value);
                ui.strong(value_with_unit(&value, text, info.meta.unit));
                ui.colored_label(quiet, display_label(path, info.meta.label));
            }
        }
    });
```

Replace:

```rust
    }
    if tile == Tile::HotLow
        && let Some(consequences) = shown.consequences
```

with:

```rust
    }
    if tile == Some(Tile::HotLow)
        && let Some(consequences) = shown.consequences
```

Replace:

```rust
    }
    let learn = row_learn(path).unwrap_or(Learn::None(""));
    let cell = readouts
```

with:

```rust
    }
    if STORED_3D_ROWS.contains(&path) {
        heading(ui, WATCH_OUT);
        ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
    }
    let cell = readouts
```

Replace:

```rust
        .show_details()
        .then(|| cell_line(result_info(path).and_then(|info| info.cell.as_deref())));
    asked.or(foot_ui(ui, Some(learn), Some(path), cell, readouts))
}
```

with:

```rust
        .show_details()
        .then(|| cell_line(info.cell.as_deref()));
    let equation = registry().equation_for(path).map(|_| path);
    asked.or(foot_ui(ui, row_learn(path), equation, cell, readouts))
}

/// What a click on the result at `path` selects (spec 5.2): the check whose own result it is
/// ([`guidance::guide`]), else the result.
pub(crate) fn result_selection(path: &'static str) -> Selection {
    match guidance::guide(path) {
        Some(guide) => Selection::Check(guide.check),
        None => Selection::Result(path),
    }
}
```

**`magcoupling-rs/src/gui/geometry_view.rs`**

Replace:

```rust
//! ([`crate::gui::dashboard::hover_text`]) and its equation, built only while hovered, and a
//! click opens it in the Equation panel. A violated callout is red, one that is not a number amber (decision M42-4);
//! the space claim is dashed, an exceeded axis red. The view is rebuilt from the design shown on
```

with:

```rust
//! ([`crate::gui::dashboard::hover_text`]) and its equation, built only while hovered, and a
//! click selects its result for the Explain panel (UX overhaul spec 5.2). A violated callout is red, one that is not a number amber (decision M42-4);
//! the space claim is dashed, an exceeded axis red. The view is rebuilt from the design shown on
```

Replace:

```rust
        );
        readouts.show(ui, response, callout.path, |show_details| {
            callout_text(callout, show_details)
```

with:

```rust
        );
        readouts.select(ui, response, callout.path, |show_details| {
            callout_text(callout, show_details)
```

Replace:

```rust

/// Paints a dimension line with its arrowheads and tag; returns its hover area.
fn paint_dimension(
```

with:

```rust

/// Paints a dimension line with its arrowheads and tag; returns its hover area: the line's and
/// the tag's, so a click on the tag selects its result (UX overhaul spec 4.4).
fn paint_dimension(
```

Replace:

```rust
    // Past the outer end the tag lands on a block or the liner: on a plate, it stays readable.
    plated_text(
        painter,
```

with:

```rust
    // Past the outer end the tag lands on a block or the liner: on a plate, it stays readable.
    let tag = plated_text(
        painter,
```

Replace:

```rust
    );
    Rect::from_two_pos(a, b).expand(6.0)
}
```

with:

```rust
    );
    Rect::from_two_pos(a, b).expand(6.0).union(tag)
}
```

Replace:

```rust
/// Paints `text` centred at `at` on a plate of `plate` one point larger than the text, so it
/// reads over whatever is drawn beneath it.
pub(crate) fn plated_text(
```

with:

```rust
/// Paints `text` centred at `at` on a plate of `plate` one point larger than the text, so it
/// reads over whatever is drawn beneath it; returns the plate.
pub(crate) fn plated_text(
```

Replace:

```rust
    plate: Color32,
) {
    let galley = painter.layout_no_wrap(text, font, color);
```

with:

```rust
    plate: Color32,
) -> Rect {
    let galley = painter.layout_no_wrap(text, font, color);
```

Replace:

```rust
    painter.galley(rect.min, galley, color);
}
```

with:

```rust
    painter.galley(rect.min, galley, color);
    rect.expand(1.0)
}
```

Replace:

```rust
                    .rect;
                readouts.show_over(ui, rect, callout.path, |show_details| {
                    callout_text(callout, show_details)
```

with:

```rust
                    .rect;
                readouts.select_over(ui, rect, callout.path, |show_details| {
                    callout_text(callout, show_details)
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
use crate::gui::dashboard::{Level, end_effect_banner_ui};
use crate::gui::explain::{Asked, LEARN, Selection, explain_ui};
use crate::gui::explorer::{Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights};
```

with:

```rust
use crate::gui::dashboard::{Level, end_effect_banner_ui};
use crate::gui::explain::{Asked, LEARN, Selection, explain_ui, result_selection};
use crate::gui::explorer::{Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights};
```

Replace:

```rust
        }
        // A result clicked opens in the Equation panel and traces its inputs, unless an input's
        // trace marks it: reading the equations of the results an input drives keeps that trace
        // (and the table's "Traced only" rows). A result without an equation record keeps the
        // trace too: nothing is known of its inputs.
        if let Some(path) = &events.clicked {
            let inside = self
```

with:

```rust
        }
        // A result selected (a dimension tag, a callout row, a results-table row: UX overhaul
        // spec 5.2) shows in the Explain panel, drawn before this click: the next frame shows
        // it. It no longer opens in the Equation panel, whose entry link does (spec 5.5).
        if let Some(path) = events.selected {
            self.selection = result_selection(path);
            ui.ctx().request_repaint();
        }
        // A result clicked (it opens in the Equation panel) or selected traces its inputs,
        // unless an input's trace marks it: reading the results an input drives keeps that
        // trace (and the table's "Traced only" rows). A result without an equation record keeps
        // the trace too: nothing is known of its inputs.
        if let Some(path) = events.clicked.as_deref().or(events.selected) {
            let inside = self
```

**`magcoupling-rs/src/gui/readouts.rs`**

Replace:

```rust
//! ([`TermColors::of`]), no wider than the window; a click asks the panel to open it in the
//! Equation panel. Both the
//! text and the typesetting run only inside `on_hover_ui`, so a frame pays for the one value
```

with:

```rust
//! ([`TermColors::of`]), no wider than the window; a click asks the panel to open it in the
//! Equation panel, or, for a dimension tag, a callout row and a results-table row
//! ([`Readouts::select`]), to select its result for the Explain panel. Both the
//! text and the typesetting run only inside `on_hover_ui`, so a frame pays for the one value
```

Replace:

```rust
    pub hovered: Option<String>,
    /// The path of the value clicked.
    pub clicked: Option<String>,
```

with:

```rust
    pub hovered: Option<String>,
    /// The path of the value clicked, which opens in the Equation panel.
    pub clicked: Option<String>,
```

Replace:

```rust
    pub clicked: Option<String>,
    /// The teaching note a warning's or a check's Why link asked for, by id.
```

with:

```rust
    pub clicked: Option<String>,
    /// The path of the result a click selected for the Explain panel (a dimension tag, a
    /// callout row, a results-table row: UX overhaul spec 5.2).
    pub selected: Option<&'static str>,
    /// The teaching note a warning's or a check's Why link asked for, by id.
```

Replace:

```rust
        self.tip(ui, response, path, text, false);
    }
```

with:

```rust
        self.tip(ui, response, path, text, false);
    }

    /// [`Readouts::show`] for a result a click selects (a dimension tag, a callout row, a
    /// results-table row: UX overhaul spec 5.2) rather than opens in the Equation panel (spec
    /// 5.5): its mark, and while hovered `text(show_details)` and its equation; a click, or
    /// Enter or Space on it focused, asks for it in the Explain panel. The keyboard's focus
    /// arriving asks for nothing.
    pub fn select(
        &mut self,
        ui: &egui::Ui,
        response: egui::Response,
        path: &'static str,
        text: impl FnOnce(bool) -> String,
    ) {
        if response.clicked() {
            self.events.selected = Some(path);
        }
        self.tip(ui, response, path, text, false);
    }

    /// [`Readouts::select`] for a result drawn in `rect` by widgets that take no clicks: a
    /// click sensor over the rect, keyed by `path` in `ui`.
    pub fn select_over(
        &mut self,
        ui: &egui::Ui,
        rect: Rect,
        path: &'static str,
        text: impl FnOnce(bool) -> String,
    ) {
        let response = ui.interact(rect, ui.id().with(("readout", path)), egui::Sense::click());
        self.select(ui, response, path, text);
    }
```

**`magcoupling-rs/src/gui/results_table.rs`**

Replace:

```rust
/// ([`column_widths`]; the path is in the hover text, which is built only while the row is
/// hovered). The whole row is a readout: hover it for its equation, click it to open it.
fn row_ui(
```

with:

```rust
/// ([`column_widths`]; the path is in the hover text, which is built only while the row is
/// hovered). The whole row is a readout: hover it for its equation, click it to select its
/// result for the Explain panel (UX overhaul spec 5.2).
fn row_ui(
```

Replace:

```rust
    ui: &mut egui::Ui,
    entry: &TableEntry,
    results: &DesignResults,
```

with:

```rust
    ui: &mut egui::Ui,
    entry: &'static TableEntry,
    results: &DesignResults,
```

Replace:

```rust
    });
    readouts.show_over(ui, row.response.rect, &entry.path, |show_details| {
        row_tooltip(entry, &value, show_details)
```

with:

```rust
    });
    readouts.select_over(ui, row.response.rect, &entry.path, |show_details| {
        row_tooltip(entry, &value, show_details)
```

**`magcoupling-rs/src/gui/size_controls.rs`**

Replace:

```rust
        || (response.has_focus()
            && response.ctx.input(|i| {
                i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space)
            }))
}
```

with:

```rust
        || (response.has_focus()
            && response
                .ctx
                .input(|i| i.key_pressed(egui::Key::Enter) || i.key_pressed(egui::Key::Space)))
}
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1110 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** the six files: `feat(magcoupler): a dimension tag, a callout row or a results-table row selects its result, whose Explain entry opens its equation (BL-094 T1)`.

### Task 2: The callout list and its notes to the Geometry tab; the first view at most 250 words

**Model:** `sonnet`.

The opening tab draws `geometry_view::drawing_ui`: the drawing with its tags and, under it, only the notes that carry a level, while true, with the height they take wrapped; the Geometry tab's `geometry_ui` keeps the list. The first view's word ceiling drops from 280 to the design's 250 (it draws 200).

**Files:**
- Modify: `magcoupling-rs/src/gui/geometry_view.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/app.rs`

**Interfaces:**
- Produces: `geometry_view::drawing_ui(ui, inputs, results, clamps_paused, readouts) -> GeometryLayout` (the signature of `geometry_ui`).
- Changes: `geometry_ui` and `drawing_ui` are `view_ui(.., with_list)`; `list_ui` takes `with_list`.

- [ ] **Step 1 (red): the opening tab's tests.** The blocks add `the_opening_tab_s_drawing_keeps_its_tags_and_only_the_notes_with_a_level`, look for the callout list on the Geometry tab in `a_new_panel_opens_on_size_and_torque_with_the_tiles_over_the_drawing`, hold the first view to 250 words, and find each note with a level drawn whole on the first view at 1024 x 700, 1280 x 800 and 1920 x 1080 (the example design's caution wraps at 1024).

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
                    "{size:?}"
                );
            }
        }
```

with:

```rust
                    "{size:?}"
                );
                // Each note with a level, under the drawing (decision D-66), is drawn whole: the
                // example design's cap thread caution wraps to two rows at 1024 wide.
                let paused = crate::engine::clamps::PAUSED;
                let geometry = crate::gui::geometry::geometry(&inputs, &results, paused);
                for note in geometry.notes.iter().filter(|note| note.level.is_some()) {
                    whole(&note.text);
                }
            }
        }
```

Replace:

```rust
    #[test]
    fn the_first_view_draws_at_most_two_hundred_and_eighty_words_at_1280_by_800() {
        // UX overhaul spec 2.1 (goal G-1) and 8.2, test 8, measured as the inventory did: the
```

with:

```rust
    #[test]
    fn the_first_view_draws_at_most_two_hundred_and_fifty_words_at_1280_by_800() {
        // UX overhaul spec 2.1 (goal G-1) and 8.2, test 8, measured as the inventory did: the
```

Replace:

```rust
        // the drawing kept until its tags are selections (decision D-19); 427 after part 2a's
        // top bar; 272 after part 2b-i moved the Size column's lines into the Explain panel.
        // The design's 250 needs the callout list to leave the opening tab (part 2b-ii), which
        // lowers this ceiling to it.
        use crate::gui::test_support::words_shown;
```

with:

```rust
        // the drawing kept until its tags are selections (decision D-19); 427 after part 2a's
        // top bar; 272 after part 2b-i moved the Size column's lines into the Explain panel;
        // 200 after part 2b-ii-b moved the callout list and its notes to the Geometry tab.
        use crate::gui::test_support::words_shown;
```

Replace:

```rust
        let words = words_shown(&output, egui::Rect::from_min_size(egui::Pos2::ZERO, size));
        assert!(words <= 280, "{words} words on the first view");
    }
```

with:

```rust
        let words = words_shown(&output, egui::Rect::from_min_size(egui::Pos2::ZERO, size));
        assert!(words <= 250, "{words} words on the first view");
    }
```

**`magcoupling-rs/src/gui/geometry_view.rs`**

Replace:

```rust

    /// The stroke colour of every line segment.
```

with:

```rust

    #[test]
    fn the_opening_tab_s_drawing_keeps_its_tags_and_only_the_notes_with_a_level() {
        // UX overhaul spec 4.4: on the opening tab the numbered tags stay on the drawing (a
        // click on one selects its result) and the callout list and the notes go to the
        // Geometry tab, but a note with a level (a design check's caution, a part not drawn)
        // stays under the drawing while it is true, as the "Exploration only" lines do. The
        // drawing takes the height the list left.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let g = geometry(&inputs, &results, false);
        assert!(g.notes.iter().any(|note| note.level.is_some()));
        assert!(g.notes.iter().any(|note| note.level.is_none()));
        let size = egui::vec2(1000.0, 700.0);
        let draw = |with_list: bool| {
            let mut layout = None;
            let output = sized_frame(&ctx, size, Vec::new(), |ui| {
                let readouts = &mut Readouts::default();
                layout = Some(if with_list {
                    geometry_ui(ui, &inputs, &results, false, readouts)
                } else {
                    drawing_ui(ui, &inputs, &results, false, readouts)
                });
            });
            (output, layout.expect("drawn"))
        };
        let (_, full) = draw(true);
        let (output, drawing) = draw(false);
        let texts = drawn_texts(&output);
        for callout in g.callouts() {
            assert!(!texts.contains(&callout.text), "{}", callout.text);
            let tag = callout.tag.to_string();
            assert_eq!(text_rects(&output, &tag).len(), 1, "tag {tag} once");
        }
        for note in &g.notes {
            assert_eq!(
                texts.contains(&note.text),
                note.level.is_some(),
                "{}",
                note.text
            );
        }
        assert_eq!(drawing.dimensions.len(), full.dimensions.len());
        assert!(
            drawing.rect.height() > full.rect.height(),
            "{drawing:?} {full:?}"
        );
    }

    /// The stroke colour of every line segment.
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
        }
        // Under the drawing its callout list, as on the Geometry tab, until a click on a
        // dimension tag shows its callout in the Explain panel (decision D-19).
        let list = centre_rect(&output, "Face gap 1.400 mm");
```

with:

```rust
        }
        // The callout list is the Geometry tab's (spec 4.4): a click on a dimension tag shows
        // its result in the Explain panel. There it is under the drawing.
        assert_eq!(count(&output, "Face gap 1.400 mm"), 0);
        harness.click_text(CentreView::Geometry.label());
        let output = harness.frame(Vec::new());
        let drawing = drawing_rect(&output);
        let list = centre_rect(&output, "Face gap 1.400 mm");
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 15
  ```

  Expected: the build fails with 1 errors, the first "cannot find function `drawing_ui` in this scope": the green step adds what the tests call.

- [ ] **Step 3 (green): the opening tab's drawing.**

**`magcoupling-rs/src/gui/geometry_view.rs`**

Replace:

```rust
//! The geometry view (spec M4 "Layout": the centre region's Geometry tab; Size and torque
//! draws it too): paints [`crate::gui::geometry`] to scale with egui's painter, both views at
//! one scale (decision M42-3), then lists the dimension callouts and the notes under them.
//!
```

with:

```rust
//! The geometry view (spec M4 "Layout": the centre region's Geometry tab; Size and torque
//! draws its drawing, [`drawing_ui`]): paints [`crate::gui::geometry`] to scale with egui's
//! painter, both views at one scale (decision M42-3), then lists the dimension callouts and the
//! notes under them.
//!
```

Replace:

```rust
/// without the clamp-boss design check while `clamps_paused` ([`geometry`]), each
/// callout a readout (`readouts`), and returns what it drew.
pub fn geometry_ui(
```

with:

```rust
/// without the clamp-boss design check while `clamps_paused` ([`geometry`]), each
/// callout a readout (`readouts`), and returns what it drew: the Geometry tab's view, the
/// drawing with the callout list and the notes under it.
pub fn geometry_ui(
```

Replace:

```rust
) -> GeometryLayout {
    let g = geometry(inputs, results, clamps_paused);
```

with:

```rust
) -> GeometryLayout {
    view_ui(ui, inputs, results, clamps_paused, readouts, true)
}

/// [`geometry_ui`] for the opening tab (UX overhaul spec 4.4): the drawing with its tags, and
/// under it only the notes that carry a level (a part not drawn, a design check's caution),
/// drawn only while true, as the "Exploration only" lines are. The callout list and the plain
/// notes are the Geometry tab's.
pub fn drawing_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
    readouts: &mut Readouts,
) -> GeometryLayout {
    view_ui(ui, inputs, results, clamps_paused, readouts, false)
}

/// The drawing, then under it the callout list and every note `with_list`, else the notes
/// with a level alone.
fn view_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
    readouts: &mut Readouts,
    with_list: bool,
) -> GeometryLayout {
    let g = geometry(inputs, results, clamps_paused);
```

Replace:

```rust
    let row = list_row(ui);
    let rows = g.end.callouts.len() + g.side.callouts.len() + g.notes.len();
    let list = rows as f32 * row * 1.5;
    let height = drawing_height(ui.available_height(), list, row);
```

with:

```rust
    let row = list_row(ui);
    let list = if with_list {
        let rows = g.end.callouts.len() + g.side.callouts.len() + g.notes.len();
        rows as f32 * row * 1.5
    } else {
        // Each note's height wrapped at this width: one row and a half's spare does not cover a
        // caution that wraps to two rows (the example design's at 1024 wide).
        g.notes
            .iter()
            .filter(|note| note.level.is_some())
            .map(|note| {
                let text = egui::WidgetText::from(note.text.as_str());
                let wrap = Some(egui::TextWrapMode::Wrap);
                let body = egui::TextStyle::Body;
                let galley = text.into_galley(ui, wrap, ui.available_width(), body);
                galley.size().y + ui.spacing().item_spacing.y
            })
            .sum()
    };
    let height = drawing_height(ui.available_height(), list, row);
```

Replace:

```rust
    }
    list_ui(ui, &g, readouts);
    layout
```

with:

```rust
    }
    list_ui(ui, &g, readouts, with_list);
    layout
```

Replace:

```rust
/// The callouts (tag and text, in their colours, each a readout) and the notes, scrolling in
/// the height left (egui's 64-point floor for a scroll area lowered to none).
fn list_ui(ui: &mut egui::Ui, g: &Geometry, readouts: &mut Readouts) {
    egui::ScrollArea::vertical()
```

with:

```rust
/// The callouts (tag and text, in their colours, each a readout) and the notes, scrolling in
/// the height left (egui's 64-point floor for a scroll area lowered to none); without the
/// list (`with_list` false), the notes with a level alone.
fn list_ui(ui: &mut egui::Ui, g: &Geometry, readouts: &mut Readouts, with_list: bool) {
    egui::ScrollArea::vertical()
```

Replace:

```rust
        .show(ui, |ui| {
            for callout in g.callouts() {
                let color = callout.level.map(|l| l.color(ui.visuals()));
```

with:

```rust
        .show(ui, |ui| {
            for callout in g.callouts().filter(|_| with_list) {
                let color = callout.level.map(|l| l.color(ui.visuals()));
```

Replace:

```rust
            }
            for note in &g.notes {
                let text = egui::RichText::new(&note.text);
```

with:

```rust
            }
            for note in g.notes.iter().filter(|n| with_list || n.level.is_some()) {
                let text = egui::RichText::new(&note.text);
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
use crate::gui::geometry::wall_cautions;
use crate::gui::geometry_view::geometry_ui;
use crate::gui::guidance::{
```

with:

```rust
use crate::gui::geometry::wall_cautions;
use crate::gui::geometry_view::{drawing_ui, geometry_ui};
use crate::gui::guidance::{
```

Replace:

```rust
    /// panel; the "Exploration only" lines and the keyway's advisory, only when true; then the
    /// drawing and its callout list in all the height left (the plan's decision D-19).
    fn view_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
```

with:

```rust
    /// panel; the "Exploration only" lines and the keyway's advisory, only when true; then the
    /// drawing in all the height left, a click on a tag selecting its result, with the notes
    /// that carry a level under it; the callout list is the Geometry tab's (spec 4.4).
    fn view_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
```

Replace:

```rust
                }
                geometry_ui(ui, shown, &self.results, self.clamps_paused, readouts);
            }
```

with:

```rust
                }
                drawing_ui(ui, shown, &self.results, self.clamps_paused, readouts);
            }
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1111 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** the three files: `feat(magcoupler): the callout list and its notes move to the Geometry tab; the first view draws at most 250 words (BL-094 T2)`.

### Task 3: gui-smoke reads a dimension tag's entry

**Model:** `sonnet`.

After the input's entry, on the Geometry tab, the smoke prompt clicks tag 1 of the drawing and judges `tag_entry` in the Explain panel; `first_view` also looks for the tags and no list under the drawing. The red step is the prompt's guard in `app.rs` and the verdict test's list of checks; the green step is the prompt.

**Files:**
- Modify: `magcoupling-rs/src/app.rs`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`, `.claude/workflows/gui-smoke.js`

- [ ] **Step 1 (red): the guard and the verdict test.**

**`linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`**

Replace:

```js
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'check_entry',
  'lesson_link_opened', 'tooltip_plain', 'input_entry', 'torque_card', 'address_follows']
// A calculator agent's reading of a good run; each case changes one thing.
```

with:

```js
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'check_entry',
  'lesson_link_opened', 'tooltip_plain', 'input_entry', 'tag_entry', 'torque_card',
  'address_follows']
// A calculator agent's reading of a good run; each case changes one thing.
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        assert!(
            between("input_entry=true was judged", "page.goto(").contains(collect),
            "the first page's console messages are collected before the link is opened again"
```

with:

```rust
        assert!(
            between("tag_entry=true was judged", "page.goto(").contains(collect),
            "the first page's console messages are collected before the link is opened again"
```

Replace:

```rust
            );
        }
        assert!(
            between("drag_moved=true", "then close the browser").contains(collect),
```

with:

```rust
            );
        }
        // UX overhaul slice B, stage 2, part 2b-ii-b (spec 4.4, 5.2): after the input's entry
        // a click on tag 1 of the drawing (the face gap's dimension) selects its result, and the
        // step finds the result's entry in the Explain panel by the name typed once as
        // TAG_LABEL, with its Equation link and no Equation panel opened.
        {
            use crate::gui::explain::EQUATION;
            let label = crate::gui::plain_text::label_for("model.face_gap_mm");
            assert_eq!(quoted("TAG_LABEL"), label, "the script's TAG_LABEL");
            assert_eq!(
                script.matches(label).count(),
                1,
                "the smoke prompt types the face gap's name once, in TAG_LABEL"
            );
            let step = between("tag_entry=true only if", "tag_entry=true was judged");
            assert!(
                between("input_entry=true only if", "tag_entry=true only if")
                    .contains("click the number \"1\""),
                "the tag's entry is judged after the input's, from a click on tag 1"
            );
            assert!(
                step.contains("Explain panel") && step.contains(&format!("\"{EQUATION}\"")),
                "the tag's entry is judged in the Explain panel by its Equation link: {step}"
            );
        }
        assert!(
            between("drag_moved=true", "then close the browser").contains(collect),
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib app::tests::the_smoke_test_share_link_opens_its_design 2>&1 | tail -n 15
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: 1 failed: `the_smoke_test_share_link_opens_its_design`. The verdict test's last line `1 of 80 checks fail` (its case "tag_entry false with the agent's passed true" passes the run, since the script does not know the check yet).

- [ ] **Step 3 (green): the prompt.**

**`.claude/workflows/gui-smoke.js`**

Replace:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, an input tooltip shows no spreadsheet details and a click on its label shows the input\'s entry in the Explain panel, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

with:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, an input tooltip shows no spreadsheet details and a click on its label shows the input\'s entry in the Explain panel, a click on a dimension tag of the drawing shows its result\'s entry there, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

Replace:

```js
const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'check_entry', 'lesson_link_opened', 'tooltip_plain', 'input_entry', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
```

with:

```js
const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'check_entry', 'lesson_link_opened', 'tooltip_plain', 'input_entry', 'tag_entry', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
```

Replace:

```js
    input_entry: { type: 'boolean' },
    torque_card: { type: 'boolean' },
```

with:

```js
    input_entry: { type: 'boolean' },
    tag_entry: { type: 'boolean' },
    torque_card: { type: 'boolean' },
```

Replace:

```js

const PLAYWRIGHT_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close"'
```

with:

```js

// The name of the face gap's result, at the top of its Explain entry, which the tag step finds the
// entry by (UX overhaul spec 4.4 and 5.2: a click on a dimension tag selects its result): typed
// once here. app.rs's smoke test holds it equal to gui::plain_text::label_for of the face gap.
const TAG_LABEL = 'Gap at the flat centres (effective gap)'

const PLAYWRIGHT_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close"'
```

Replace:

```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Design menu and
// its Load design item, a check's row, a lesson link, the Geometry tab, an input's label, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

with:

```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Design menu and
// its Load design item, a check's row, a lesson link, the Geometry tab, an input's label, a dimension tag, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

Replace:

```js
// row's tooltip shows no spreadsheet details while the toggle is off, and (part 2b-ii, spec 5.2)
// a click on that row's label shows the input's entry in the Explain panel. Then (slice 3, spec 7.2 and 8.6) the link is opened again with a timer on every frame the
// app draws: on that fresh page a click on the Hot-low tile shows the torque card's heading,
```

with:

```js
// row's tooltip shows no spreadsheet details while the toggle is off, and (part 2b-ii, spec 5.2)
// a click on that row's label shows the input's entry in the Explain panel, and (part 2b-ii-b,
// spec 4.4 and 5.2) a click on a dimension tag of the drawing shows its result's entry there. Then (slice 3, spec 7.2 and 8.6) the link is opened again with a timer on every frame the
// app draws: on that fresh page a click on the Hot-low tile shows the torque card's heading,
```

Replace:

```js
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (the size controls on the left, the check list and the Explain panel on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the page shows the first view with no click. The middle region: at its top a row of tabs starting "Size and torque" (the selected tab); under the tabs four tiles side by side, each a large number with its unit over a smaller, quieter label, the second labelled "Hot-low torque" with a line starting "needs" under its label; then a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft. The right side: the heading "CHECKS", a status line, then one row per check, each a small shape, the check's name and a word at the row's right end (OK, Check or Fails); a quiet line under the rows about fixed 3D fields; then the heading "EXPLAIN" and a paragraph starting "Two rings of magnets" with three numbered lines under it. Then the Design menu: click "Design" in the header row (every click as below: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot), wait 1 second and take a screenshot. details_toggle=true only if a menu opened under "Design" listing, top to bottom, "Reset to the starting design", "Open the example design", "Save design", "Load design" and a checkbox labelled "Show spreadsheet details", unticked. Press Escape with code: async (page) => { await page.keyboard.press('Escape'); } and wait 1 second (the menu closes).
Then the design file picker: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
```

with:

```js
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (the size controls on the left, the check list and the Explain panel on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the page shows the first view with no click. The middle region: at its top a row of tabs starting "Size and torque" (the selected tab); under the tabs four tiles side by side, each a large number with its unit over a smaller, quieter label, the second labelled "Hot-low torque" with a line starting "needs" under its label; then a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft, small numbers beside its dimension lines and no list of numbered lines under it. The right side: the heading "CHECKS", a status line, then one row per check, each a small shape, the check's name and a word at the row's right end (OK, Check or Fails); a quiet line under the rows about fixed 3D fields; then the heading "EXPLAIN" and a paragraph starting "Two rings of magnets" with three numbered lines under it. Then the Design menu: click "Design" in the header row (every click as below: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot), wait 1 second and take a screenshot. details_toggle=true only if a menu opened under "Design" listing, top to bottom, "Reset to the starting design", "Open the example design", "Save design", "Load design" and a checkbox labelled "Show spreadsheet details", unticked. Press Escape with code: async (page) => { await page.keyboard.press('Escape'); } and wait 1 second (the menu closes).
Then the design file picker: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
```

Replace:

```js
Then click that label (as above, at the X, Y the mouse is at), move the mouse off the label onto the drawing in the middle region, wait 1 second and take a screenshot. input_entry=true only if the Explain panel (the right side, under the check list) now shows the input's entry: at its top the input's value and the label you clicked, under them the link "${TRACE_LINK}", then the heading "What it is" with a sentence about the input.
If a target of these last checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes; the check list's last row is "Magnets fit". input_entry=true was judged last on this page. Collect the console messages at level "info" (the check's entry, the lesson link, the tooltip and the input's entry ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; and the tooltip step left the Geometry tab selected, where no tile is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
```

with:

```js
Then click that label (as above, at the X, Y the mouse is at), move the mouse off the label onto the drawing in the middle region, wait 1 second and take a screenshot. input_entry=true only if the Explain panel (the right side, under the check list) now shows the input's entry: at its top the input's value and the label you clicked, under them the link "${TRACE_LINK}", then the heading "What it is" with a sentence about the input.
Then a dimension tag: the drawing in the middle region (the Geometry tab, still selected) has blue dimension lines, each with a small number on a dark plate at its end. In a fresh screenshot click the number "1" on the drawing: the face gap's, a small number on a dark plate at the top of the round view on the left, on the outer ring of magnets just past the outer end of the short blue line across the gap between the rings; not the "1" of the numbered list under the drawing. Move the mouse off it onto an empty part of the middle region, wait 1 second and take a screenshot. tag_entry=true only if the Explain panel (the right side, under the check list) now shows that dimension's entry: at its top a value in mm and the name "${TAG_LABEL}", and further down the link "Equation"; and no Equation panel opened at the foot of the middle region.
If a target of these last checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes; the check list's last row is "Magnets fit". tag_entry=true was judged last on this page. Collect the console messages at level "info" (the check's entry, the lesson link, the tooltip, the input's entry and the tag's entry ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; and the tooltip step left the Geometry tab selected, where no tile is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
```

Replace:

```js
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, check_entry, lesson_link_opened, tooltip_plain, input_entry, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
```

with:

```js
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, check_entry, lesson_link_opened, tooltip_plain, input_entry, tag_entry, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
```

- [ ] **Step 4: Run the checks**, then the whole crate both ways, wasm32 and the verdict test:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml 2>&1 | grep -E "^test result" | grep -vc " 0 failed"
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app 2>&1 | grep -E "^test result" | grep -vc " 0 failed"
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: `test result: ok. 1111 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both `grep -vc` lines print `0` (no test binary with a failure); both wasm32 clippy runs clean; `cargo fmt --check` prints nothing; the verdict test's last line `all 81 checks hold`.

- [ ] **Step 5: Commit** the three files: `test(magcoupler): gui-smoke clicks a dimension tag and reads its result's entry in the Explain panel (BL-094 T3)`.

### Task 4: Docs sweep

**Model:** `sonnet`.

The README's first-view, Explain panel, inputs-side, layout, module and smoke passages; `docs/ai/` (02 the system, 03 the modules, 04 the issue, 05 the tracker, the backlog item in progress); the register's rows A-33 to A-36 and its log; owner questions 56 to 59; the design's status line.

**Files:**
- Modify: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/analyses/2026-11-18-owner-questions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

- [ ] **Step 1: The docs.**

**`docs/ai/02-system.yaml`**

Replace:

```yaml
      - "Since BL-092 (UX overhaul slice B, stage 2, part 2b-ii-a; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md): every input is a selection. Touching a Detailed inputs or Assumptions row (input_ui::RowOutput::label_clicked, an edit or a reset, or size_controls::touched on its widget, as the Size column's touches) sets explain::Selection::Input(path); explain::input_ui draws any input (input_row: its value through inputs::value_text, its help, its hint, the checks whose fixes name it, input_trade_offs, its cell) and leads every input's entry with TRACE_LINK, whose click (Asked::Trace) toggles the trace (MagcouplingPanel::toggle_trace); the label no longer traces. A check's fix link selects its input (explain::fix_selection; the ring radius selects Outside diameter's row) and still focuses its row. With the details on, a tile's and a check's entry end with their cell (cell_line). Selection::reads_consequences holds for a Size column row only, so a Detailed input's entry asks the ConsequenceCache nothing."
      - "Since BL-085 (UX overhaul slice B, stage 2, part 2a; plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md): the top bar is Undo, Redo, Design (a menu: the reset, Open the example design, Save design, Load design, Show spreadsheet details), Share, Export and Learn, each a word with no box at rest (panel.rs bar_button, design_menu_ui, reset_label: Reset to the starting design where the baseline is not the spreadsheet's design). The Explain panel draws the teaching notes (decision C-11): explain::Selection gains Note and NoteList; every note link (ReadoutEvents::note, the Size column's Why links, explorer_ui's return: the lesson line under an open equation, a Start here step) sets the selection; the Equation panel has no Explain checkbox, no note view and no button on the bar. format::value_with_unit: a text value takes no unit (design 9.2)."
```

with:

```yaml
      - "Since BL-092 (UX overhaul slice B, stage 2, part 2b-ii-a; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md): every input is a selection. Touching a Detailed inputs or Assumptions row (input_ui::RowOutput::label_clicked, an edit or a reset, or size_controls::touched on its widget, as the Size column's touches) sets explain::Selection::Input(path); explain::input_ui draws any input (input_row: its value through inputs::value_text, its help, its hint, the checks whose fixes name it, input_trade_offs, its cell) and leads every input's entry with TRACE_LINK, whose click (Asked::Trace) toggles the trace (MagcouplingPanel::toggle_trace); the label no longer traces. A check's fix link selects its input (explain::fix_selection; the ring radius selects Outside diameter's row) and still focuses its row. With the details on, a tile's and a check's entry end with their cell (cell_line). Selection::reads_consequences holds for a Size column row only, so a Detailed input's entry asks the ConsequenceCache nothing."
      - "Since BL-094 (UX overhaul slice B, stage 2, part 2b-ii-b; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md): every result is a selection. A dimension tag (its click area the line and the tag's plate), a row of the callout list and a results-table row draw through Readouts::select or select_over: a click, or Enter or Space on the focused value, sets ReadoutEvents::selected (the focus arriving does not), the hover has no OPEN_HINT, and the panel sets explain::result_selection(path) (a check's own result selects the check, any other Selection::Result) and traces as a click on a result did; the Equation panel no longer opens on that click (spec 5.5), the entry's Equation link opens it. explain::result_ui draws any result: a tile's number and label, another's value and name as its table row draws them, its help, its dashboard row's check, the stored-3D line under Watch out for STORED_3D_ROWS, Full lesson where ROW_LEARN has its row, Equation where the registry has its equation, its cell with the details. The opening tab draws geometry_view::drawing_ui (the tags, and only the notes with a level); the Geometry tab keeps the callout list. The first view draws 200 words at 1280 x 800, held to 250 (goal G-1)."
      - "Since BL-085 (UX overhaul slice B, stage 2, part 2a; plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md): the top bar is Undo, Redo, Design (a menu: the reset, Open the example design, Save design, Load design, Show spreadsheet details), Share, Export and Learn, each a word with no box at rest (panel.rs bar_button, design_menu_ui, reset_label: Reset to the starting design where the baseline is not the spreadsheet's design). The Explain panel draws the teaching notes (decision C-11): explain::Selection gains Note and NoteList; every note link (ReadoutEvents::note, the Size column's Why links, explorer_ui's return: the lesson line under an open equation, a Start here step) sets the selection; the Equation panel has no Explain checkbox, no note view and no button on the bar. format::value_with_unit: a text value takes no unit (design 9.2)."
```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel: new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider, RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the inputs side; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

with:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), select and select_over (a click asks for the result in the Explain panel: ReadoutEvents::selected, since BL-094), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel: new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider, RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); drawing_ui (the opening tab's: the tags, and only the notes with a level, since BL-094); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the inputs side; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-a (BL-092, plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md, decisions D-49 to D-59): every input a selection, the trace as a link at the top of its entry, a fix link selects its input, the cell foot of a tile's and a check's entry, gui-smoke's input_entry. Part 2b-ii is cut in three (D-49, register row A-29): 2b-ii-b the dimension tags, table rows and callout rows as selections, the callout list's move, the word budget of 250 and gui-smoke reading a tag's entry; 2b-ii-c the left column no longer swapping (4.3) and the table-driven nothing-lost test. Touching a row (its label, a drag, Enter or Space on a focused control, an edit; since BL-093 not the focus arriving) selects its input, as in the Size column (D-50). The trace's banner stays at the inputs side's foot (D-52, A-30, owner question 55); the label's hover loses its trace line, with no line in its place (D-53, A-31, owner question 53); a note's change-entry links still only show the row (D-55, A-32, owner question 54). risk: mechanical."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
```

with:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-a (BL-092, plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md, decisions D-49 to D-59): every input a selection, the trace as a link at the top of its entry, a fix link selects its input, the cell foot of a tile's and a check's entry, gui-smoke's input_entry. Part 2b-ii is cut in three (D-49, register row A-29): 2b-ii-b the dimension tags, table rows and callout rows as selections, the callout list's move, the word budget of 250 and gui-smoke reading a tag's entry; 2b-ii-c the left column no longer swapping (4.3) and the table-driven nothing-lost test. Touching a row (its label, a drag, Enter or Space on a focused control, an edit; since BL-093 not the focus arriving) selects its input, as in the Size column (D-50). The trace's banner stays at the inputs side's foot (D-52, A-30, owner question 55); the label's hover loses its trace line, with no line in its place (D-53, A-31, owner question 53); a note's change-entry links still only show the row (D-55, A-32, owner question 54). risk: mechanical."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-b (BL-094, plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md, decisions D-60 to D-70): a click on a dimension tag, a callout row or a results-table row selects its result (a check's own result the check) and opens no equation; the entry's Equation link does (D-61, A-33, owner question 56: every other readout still opens the Equation panel); the selection keeps the click's trace (D-62, A-34, owner question 57); any result's entry from texts the screen drew (D-64); the callout list and the plain notes to the Geometry tab, a note with a level kept under the opening tab's drawing (D-66, A-35, owner question 58); the first view at most 250 words (200), the walk's three copies of a step's title kept (D-67, A-36, owner question 59); gui-smoke's tag_entry (D-68). risk: mechanical (D-69)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown

---

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-a: every input a selection (BL-092)
```

with:

```markdown

---

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-b: the drawing's selections and the word budget (BL-094)
- What changed: a click on a dimension tag (its line or its number), a row of the Geometry tab's callout list or a results-table row selects its result, and the Explain panel shows its entry (its value and name, what it is, the check of its row, the stored-3D line where it applies, its lesson, its Equation link, its cell with spreadsheet details on); a check's own row selects the check. Such a click no longer opens the Equation panel; the entry's Equation link does, and those hovers lost the line that said a click opens it. The opening tab draws the drawing with its tags and no callout list (a note with a level stays under it while true); the Geometry tab keeps the list. The first view draws 200 words at 1280 x 800, held to 250. The gui-smoke run clicks tag 1 on the Geometry tab and checks the face gap's entry (`tag_entry`).
- Tests: a selecting readout by click and by Enter or Space, not on focus (readouts); any result's entry, the Equation and lesson links, a check's own result (explain); a tag, a callout and a table row select and open no equation (panel); the opening tab's drawing keeps its tags and only the notes with a level (geometry_view); the first view at most 250 words (app); the trace and Equation panel tests go through the entry's link; the smoke prompt guard and the verdict test (81 checks).
- Not in this part: the left column no longer changing with the centre tab, the table-driven nothing-lost test (part 2b-ii-c); part 2b-iii as before.

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-a: every input a selection (BL-092)
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  priority: 2
  status: open
  notes: "Filed 2026-11-23 with its plan (the design's header: each plan files one item). Part 2b-ii-b of BL-092's list, items 1 to 4, in one plan (decision D-60): the prototype's diff is three code tasks no larger than part 2b-ii-a's. risk: mechanical (decision D-69): no new number and no new on-screen string; a result's entry draws texts the screen drew (the table row's value and name, the help, the check's mark and name, the stored-3D line, the lesson line, Equation, the cell). Part 2b-ii-c (the left column, the table-driven nothing-lost test) and part 2b-iii are not touched."

```

with:

```yaml
  priority: 2
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2b-ii-b (Tasks 1 to 3 of the plan); the full gate, the smoke run and the whole-branch review are its Task 5. Filed 2026-11-23 with its plan (the design's header: each plan files one item). Part 2b-ii-b of BL-092's list, items 1 to 4, in one plan (decision D-60): the prototype's diff is three code tasks no larger than part 2b-ii-a's. risk: mechanical (decision D-69): no new number and no new on-screen string; a result's entry draws texts the screen drew (the table row's value and name, the help, the check's mark and name, the stored-3D line, the lesson line, Equation, the cell). Part 2b-ii-c (the left column, the table-driven nothing-lost test) and part 2b-iii are not touched."

```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
| A-31 | A row's label hover ended "Click the label to highlight the results it drives that have an equation". A click on the label now selects the input. What does the hover's last line say? | Nothing: the line goes, and no line takes its place. The entry's link says what it does. | The old line would be false; a new line is new text for review. Owner question 53. | A line such as "Click the label to show its entry" ends the hover. | The plan of part 2b-ii-a (BL-092), decision D-53 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
```

with:

```markdown
| A-31 | A row's label hover ended "Click the label to highlight the results it drives that have an equation". A click on the label now selects the input. What does the hover's last line say? | Nothing: the line goes, and no line takes its place. The entry's link says what it does. | The old line would be false; a new line is new text for review. Owner question 53. | A line such as "Click the label to show its entry" ends the hover. | The plan of part 2b-ii-a (BL-092), decision D-53 |
| A-33 | A click on a dimension tag, a callout row or a results-table row now selects its result (5.2), and 5.5 says the Equation panel changes only through its own controls or an "Equation" link. What do the other readouts' clicks do (the numbers behind a check in its entry, the plots' readouts, the end-effect banner)? | They still open the Equation panel at their result, and their hover still ends "Click to open it in the Equation panel". The three 5.2 names select and open nothing; their entry's Equation link opens the panel. | 5.2 names the tag, the callout row and the table row; changing the others would change surfaces this part does not otherwise touch (a check's entry, the Plots tab, which part 2b-iii rebuilds). Owner question 56. | Every result readout selects its result, and only an Equation link opens the Equation panel. | The plan of part 2b-ii-b (BL-094), decision D-61 |
| A-34 | A click on a result traced the inputs it reads (decision O-8). The click now selects the result. Does it still trace? | Yes: the selection traces as the click did, unless an input's trace marks the result or the result has no equation record. | Nothing that was reachable is lost, and no new text: a result's trace has no link of its own in the design. Owner question 57. | The trace moves to a link in the result's entry, as an input's did. | The plan of part 2b-ii-b (BL-094), decision D-62 |
| A-35 | 4.4 moves the callout list and the notes under the drawing to the Geometry tab. Some notes carry a level: a part the drawing leaves out ("Not drawn: ...") and a design check's caution. Do those leave the opening tab too? | No: a note with a level stays under the opening tab's drawing while it is true, as the "Exploration only" lines and the wall cautions above the drawing do; the callout list and the plain notes (the wall rule) move. | A drawing with parts missing and no note saying so would mislead; the amber notes are conditions, like the cautions 4.4 keeps. None shows on the starting design. A callout with a level (a red gap, an axis over its claim) moves with the list: on the opening tab its tag stays red and the check list carries the failure. Owner question 58. | Every note moves, as 4.4 says. | The plan of part 2b-ii-b (BL-094), decision D-66 |
| A-36 | Part 2a's decision D-32 left the three copies of a Start here step's title (the note's heading, the lesson line, the walk bar's step line) for the word budget of 250 to decide. The budget measures the first view, where no walk runs (200 words with the callout list moved). Which copies go? | None: all three stay. | The budget is met without them; dropping one is a text change for the text review, and each copy is its own place's form. Owner question 59. | The lesson line under the equation drops the title while the walk runs (the walk bar and the note's heading name it). | The plan of part 2b-ii-b (BL-094), decision D-67 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
```

Replace:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-23: A-29, A-30, A-31 and A-32 added with the plan of slice B's stage 2, part 2b-ii-a (BL-092). Each is an assumption, not the owner's answer.
```

with:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-23: A-33, A-34, A-35 and A-36 added with the plan of slice B's stage 2, part 2b-ii-b (BL-094). Each is an assumption, not the owner's answer.
- 2026-11-23: A-29, A-30, A-31 and A-32 added with the plan of slice B's stage 2, part 2b-ii-a (BL-092). Each is an assumption, not the owner's answer.
```

**`docs/analyses/2026-11-18-owner-questions.md`**

Replace:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 35 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54 and 55 on 2026-11-23). Part D, five that matter only if the tools stop being generic.

```

with:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 39 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54, 55, 56, 57, 58 and 59 on 2026-11-23). Part D, five that matter only if the tools stop being generic.

```

Replace:

```markdown

### Linkage simulator
```

with:

```markdown

### 56. Should every number on the page select its result, and only an "Equation" link open the Equation panel?

Added 2026-11-23 (the next free number). Part 2b-ii-b makes a click on a dimension tag, a row of the drawing's callout list or a results-table row show the result's entry in the Explain panel; the entry's "Equation" link opens the Equation panel (the design's 5.2 and 5.5). Other numbers still open the Equation panel on a click: the numbers behind a check in its entry, the plots' readouts and the end-effect banner, and their hover still says so.

- **If you do not answer:** as described.
- (a) As the default.
- (b) Every number selects its result, and only an "Equation" link opens the Equation panel (a click on a check's number would replace the check's entry with the number's).

Source: UX design sections 5.2 and 5.5; A-33; BL-094 plan, decision D-61.

### 57. Should a click on a result still highlight the inputs it reads?

Added 2026-11-23 (the next free number). Until now a click on a result opened its equation and highlighted the inputs it reads. The click now shows the result's entry instead, and still highlights those inputs (unless you are highlighting what an input drives, which stays).

- **If you do not answer:** the click shows the entry and highlights the inputs.
- (a) As the default.
- (b) The click only shows the entry; a link in it ("Highlight the inputs this reads", new text) highlights them, as an input's link does.

Source: UX design section 5.2; A-34; BL-094 plan, decision D-62.

### 58. Should the drawing's amber notes stay on the opening tab?

Added 2026-11-23 (the next free number). The list of dimensions and the notes under the drawing move to the Geometry tab (the design's 4.4). Two kinds of note carry a colour: a part the drawing leaves out ("Not drawn: ...") and a design check's caution. They stay under the opening tab's drawing while they are true, as the amber lines above it do; the plain note on the cup wall's rule moves. None shows on the starting design. A dimension in red (a gap below zero, an axis over its space claim) moves with the list: on the opening tab its number stays red and the check list shows the failing check.

- **If you do not answer:** the coloured notes stay on the opening tab.
- (a) As the default.
- (b) Every note moves to the Geometry tab, as the design says.

Source: UX design section 4.4; A-35; BL-094 plan, decision D-66.

### 59. During the Start here walk, which copies of a step's title should go?

Added 2026-11-23 (the next free number). During the walk a step's title shows three times: the note's heading in the Explain panel, the lesson line under the equation and the walk bar's "Start here N of 14: title". Part 2a left it to the word budget of 250, which counts the first view, where no walk runs; with the dimension list moved, the first view draws 200 words. All three copies stay.

- **If you do not answer:** all three stay.
- (a) As the default.
- (b) The lesson line drops the title while the walk runs.
- (c) The walk bar's step line drops the title ("Start here N of 14").

Source: part 2a plan, decision D-32; A-36; BL-094 plan, decision D-67.

### Linkage simulator
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`; parts 2b-ii-b, 2b-ii-c and 2b-iii and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

with:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, its second (the drawing's selections and the word budget; BL-094) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md`; parts 2b-ii-c and 2b-iii and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

**`magcoupling-rs/README.md`**

Replace:

```markdown
| `src/engine/explain/records/shaft_joint.rs` | BL-064 shaft joint records: one per numeric `shaft_joint.*` result (23; the verdict, a text, has none), transcribed from `shaft_joint::compute` and `PartProperties::hub` with every correction on: the part material in effect as cases over the circuit and the back-iron pick (the Materials inputs for the default pick, a pick's library values, else the workbook's 6061-T6), each joint's Lamé rows (holding, bore stress, needed minimum, elastic edge) with a no-wall arm that gives NaN, the key alone (0 with no keyway, +inf with no bearing length or contact), each shaft's shrink as cases on 265 K with NIST's Fe-2.25Ni polynomial, the fitting temperature and the fittable edge, and the bonded inner hub's allowed fitting temperature as the lowest of your limit, the adhesive's design limit and the inner ring's own single-ring onset less 10 °C, NaN when any of the three is (owner R-15) |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, `InputsView` (the design inputs or the assumptions), the assumptions banner, the sizing controls, the right column (`checks_and_explain_ui`: the check list over the Explain panel, one `explain::Selection`, the Hot-low entry's consequences asked of the `ConsequenceCache` through `size_tab_consequences`, the Size tab's one ask, with the design's own results computed once per design (`ResultsCache`))), `tiles.rs` (since BL-081, the four tiles over the drawing on Size and torque: `Tile`, `tiles_ui`, `MIN_TILE_WIDTH`, the Hot-low tile's `needs_lines` and `gap_pieces`), `checks.rs` (since BL-081, the check list: `checks_ui`, `state_mark`, `mark_level`, `CHECKS_SHARE`), `explain.rs` (since BL-081, the Explain panel: `explain_ui`, `Selection`, `Asked`, the orientation and each entry; the torque card sits in the Hot-low tile's entry, `card_ui`, decision C-10; an entry's "Full lesson:" link is `guidance::row_learn`'s), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::new(marks, show_details)`, `show_details`, `Readouts::show` and `show_over` (the hover text built with the frame's `show_details`), `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `note_ui` (a note drawn against a `NoteView`, returning the `NoteLink` clicked), `held_reason`, `term_tag`, `reviewed_note`, `note_fallback` (Explain's next best read under an equation without a note, `NoteFallback`), `start_here_text` with `FERRITE_ONLY` on the ferrite step; its heights: `panel_heights`, `PANEL_SHARE`, `MIN_PANEL_HEIGHT`, `VIEW_STRIP`), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `plain_text.rs` (the display-name and help overlay over the engine's metadata, keyed by path: `label_for`, `help_for`, `original_help`, `hint_for`, `display_label`; `LABELS` holds spec 3.5's canonical names, three of them constants the Size tab and the plots draw too: `REQUIRED_TORQUE`, `PULL_OUT_NOMINAL`, `HOT_LOW_TORQUE`; `HELP` and `HINTS` the slice-2 help), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`; `wall_cautions`, the Size tab's amber wall lines: the wall past the keyway and, since BL-082, the cap skirt under `MIN_CAP_SKIRT_MM`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list, `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `slider_range` (an input's slider range from the catalogue built once, the shared spelling the torque card, the Size tab's controls, the guides and the tests read; the Equation panel and the panel's Required slider keep their own), `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type; a click on its label traces the input; `weak_line` and `hint_line`: a weak line wrapped to its region's width (under a control, and the torque card's lines), and the hint line `plain_text::hint_for` gives a Key design row or a Size-tab control), `dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`, `shown_lines` (the dashboard rows drawn while the shaft-clamp section is paused or not; `dashboard_lines` keeps every row for the exports), `END_EFFECT_ROWS`, `greyed_by_end_effect`, `STORED_3D_ROWS`, `STORED_3D_INPUTS`, `STORED_3D_SENTENCE`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`; `why_note_text` and `why_note_link`, a "Why: <title>" link outside the rows, and `end_effect_banner_ui`, the banner with its link), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), `result_groups.rs` (`result_groups`: the headline, the nine chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`, `target_mismatch_notice`), `history.rs` (`History`: undo and redo), `guidance.rs` (the guidance layer, pure functions: `CHECK_GUIDES`, `counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `check_terms` (`CheckTerm`, `Quoted` with its one constructor `Quoted::new`; slice 3: each counted check's terms in a fixed order, the one table the fix tests and the torque card read), `SPACE_CLAIM_AXIS_SIZES`, `all_fixes`, `FixCache`, `within_minimum_diameter`, `guide_notes` (`trade_offs`, `side_effects` and the unchecked costs), `design_status`, `Consequences` (the lines under the Size tab's controls and its torque card, `card`), `ConsequenceCache` (one key: the design shown, the Minimum OD and the shaft-clamp pause), `magnet_count_line`, `SIZE_TAB_NOTE`, `TORQUE_BAND_NOTE` and `BANNER_NOTE` (where the Size tab's and the banner's Why links lead), `plain_text_violations` (re-exported from `engine::explain::plain`)), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines (since slice 3 without the N-2 and N+2 lines, which are the torque card's count row) and Why links; `slider_minimum`, the Outside diameter slider's lowest value, one expression with the torque card; `SliderMapping`, each Size slider's own mapping of a value to a design, one mapping for the controls and the torque card's steps, which sets the count whole, refusing a value a cast would change, and checks no range on the gap or the count: its caller holds the value to the slider), `torque_card.rs` (the torque card, slice 3, BL-073; pure, no egui: `card` -> `TorqueCard` (`ShownCountBelowSlider`, `ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow`, or `Rows`, one `Row` per Size control (`Control`), read by `rows`), each control one fixed step each way through its slider's own mapping (`size_controls::SliderMapping`, the one the controls apply), in the order of tests of its spec's section 3.3: a side's `Words`, or its `Change` (the stepped design's hot-low, its change in N·m and as a fraction of the design shown's, the Outside diameter and Overall length changes, whether the calibration factor changes, the counted checks whose level flips (`Flip`, over `counted_guides` at the panel's shaft-clamp pause) and the failing checks' quantities the step moves (`MarginChange`, from `guidance::check_terms`, drawn where `shows`)); `lines(auto)` draws it, a row's two N·m at one precision and the row Auto holds as `held_line`, every text a named constant (`CARD_HEADING`, `CAPTION`, the three Auto lines, the side words, `JUMP` and `CUP_JUMP`, `CALIBRATION_CHANGE`, the four one-line forms and the flip words; `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` also end the Size tab's exploration lines); `side_words` (sizing's validity rule, every row's, the keyway left out on the count row: decision D-1) and `count_step` (two magnets each way: checked, even, inside the slider), both moved from the N-2 and N+2 lines that the count row replaced; `STEP_SLACK_MM`, the one slack of the range test and the jump rules: the overlapping magnet length on the Overall length row, the cup's diameter on the Outside diameter row since the owner's FO-2), `sizing.rs` (`SizingMode`, `held_by_auto`, `SizingState` (the mode and the free variable; the target is `metal.required_min_Nm`), `SizingRunner`: debounced inverse sizing over a search range; `sizing_range`, the range Auto searches, one rule for the panel and the torque card's test table, slice 3), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell; the Size tab's one number format, slice 3 decision 10B: `fixed` (a number to a set number of decimals, with no + added), and `signed_fixed`, `signed_change` and `change_decimals` (a change to the fewest decimals, from its own (`TORQUE_DECIMALS`, two, for N·m: the readout's gap and the torque card), that keep its true sign and read a digit that is not zero, at most `CHANGE_EXTRA_DECIMALS` more: `decimals_with_a_digit`, the one cap rule, which the torque card's `shows` reads too), `shared_change_decimals` (a torque card row's two N·m at one precision), the one-decimal `signed_percent`, the format's bound since the owner's FO-4 at the slice-3 STOP (a number whose fixed form would need more than six whole digits, `MOST_WHOLE_DIGITS`, reads four significant digits with its sign, as the readouts write values, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row: `TorquePrecision`, the one precision `shared_torque_precision` finds for a row, the readout's gap a row of one, which `signed_torque_change` draws for a torque card side and the readout's margin and `torque_shortfall` for its shortfall's size, which is drawn without a sign), and `fewest_decimals_apart`, the guidance's search for more decimals), `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries; `mark_ui`, the brand mark before the title; `tab_ui`, a tab of the centre's row, underlined in the accent when it is the active one; `quiet_text`, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes; `mark_color`, the colour of the changed dot and a trace's marks), `test_support.rs` (headless egui helpers and the shared test designs, the torque card's among them with their one table, `card_designs()`; tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests; on the web it writes the settled design into the page's address), `starting_design` (the design it opens on and hands the panel as its baseline), `address_of` (the page address that stands for a design), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the calculator's own theme, "Charcoal and crimson": the token table and the visuals built from it, forced dark), `app/fonts.rs` (the app's font: a cut of Inter Regular with tabular digits, ahead of egui's built-in fonts), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
```

with:

```markdown
| `src/engine/explain/records/shaft_joint.rs` | BL-064 shaft joint records: one per numeric `shaft_joint.*` result (23; the verdict, a text, has none), transcribed from `shaft_joint::compute` and `PartProperties::hub` with every correction on: the part material in effect as cases over the circuit and the back-iron pick (the Materials inputs for the default pick, a pick's library values, else the workbook's 6061-T6), each joint's Lamé rows (holding, bore stress, needed minimum, elastic edge) with a no-wall arm that gives NaN, the key alone (0 with no keyway, +inf with no bearing length or contact), each shaft's shrink as cases on 265 K with NIST's Fe-2.25Ni polynomial, the fitting temperature and the fittable edge, and the bonded inner hub's allowed fitting temperature as the lowest of your limit, the adhesive's design limit and the inner ring's own single-ring onset less 10 °C, NaN when any of the three is (owner R-15) |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, `InputsView` (the design inputs or the assumptions), the assumptions banner, the sizing controls, the right column (`checks_and_explain_ui`: the check list over the Explain panel, one `explain::Selection`, the Hot-low entry's consequences asked of the `ConsequenceCache` through `size_tab_consequences`, the Size tab's one ask, with the design's own results computed once per design (`ResultsCache`))), `tiles.rs` (since BL-081, the four tiles over the drawing on Size and torque: `Tile`, `tiles_ui`, `MIN_TILE_WIDTH`, the Hot-low tile's `needs_lines` and `gap_pieces`), `checks.rs` (since BL-081, the check list: `checks_ui`, `state_mark`, `mark_level`, `CHECKS_SHARE`), `explain.rs` (since BL-081, the Explain panel: `explain_ui`, `Selection`, `Asked`, the orientation and each entry; the torque card sits in the Hot-low tile's entry, `card_ui`, decision C-10; an entry's "Full lesson:" link is `guidance::row_learn`'s), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::new(marks, show_details)`, `show_details`, `Readouts::show` and `show_over` (the hover text built with the frame's `show_details`), `select` and `select_over` (a click selects the result, since BL-094), `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `note_ui` (a note drawn against a `NoteView`, returning the `NoteLink` clicked), `held_reason`, `term_tag`, `reviewed_note`, `note_fallback` (Explain's next best read under an equation without a note, `NoteFallback`), `start_here_text` with `FERRITE_ONLY` on the ferrite step; its heights: `panel_heights`, `PANEL_SHARE`, `MIN_PANEL_HEIGHT`, `VIEW_STRIP`), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `plain_text.rs` (the display-name and help overlay over the engine's metadata, keyed by path: `label_for`, `help_for`, `original_help`, `hint_for`, `display_label`; `LABELS` holds spec 3.5's canonical names, three of them constants the Size tab and the plots draw too: `REQUIRED_TORQUE`, `PULL_OUT_NOMINAL`, `HOT_LOW_TORQUE`; `HELP` and `HINTS` the slice-2 help), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`; `wall_cautions`, the Size tab's amber wall lines: the wall past the keyway and, since BL-082, the cap skirt under `MIN_CAP_SKIRT_MM`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list; `drawing_ui`, the opening tab's, without the list; `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `slider_range` (an input's slider range from the catalogue built once, the shared spelling the torque card, the Size tab's controls, the guides and the tests read; the Equation panel and the panel's Required slider keep their own), `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type; a click on its label traces the input; `weak_line` and `hint_line`: a weak line wrapped to its region's width (under a control, and the torque card's lines), and the hint line `plain_text::hint_for` gives a Key design row or a Size-tab control), `dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`, `shown_lines` (the dashboard rows drawn while the shaft-clamp section is paused or not; `dashboard_lines` keeps every row for the exports), `END_EFFECT_ROWS`, `greyed_by_end_effect`, `STORED_3D_ROWS`, `STORED_3D_INPUTS`, `STORED_3D_SENTENCE`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`; `why_note_text` and `why_note_link`, a "Why: <title>" link outside the rows, and `end_effect_banner_ui`, the banner with its link), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), `result_groups.rs` (`result_groups`: the headline, the nine chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`, `target_mismatch_notice`), `history.rs` (`History`: undo and redo), `guidance.rs` (the guidance layer, pure functions: `CHECK_GUIDES`, `counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `check_terms` (`CheckTerm`, `Quoted` with its one constructor `Quoted::new`; slice 3: each counted check's terms in a fixed order, the one table the fix tests and the torque card read), `SPACE_CLAIM_AXIS_SIZES`, `all_fixes`, `FixCache`, `within_minimum_diameter`, `guide_notes` (`trade_offs`, `side_effects` and the unchecked costs), `design_status`, `Consequences` (the lines under the Size tab's controls and its torque card, `card`), `ConsequenceCache` (one key: the design shown, the Minimum OD and the shaft-clamp pause), `magnet_count_line`, `SIZE_TAB_NOTE`, `TORQUE_BAND_NOTE` and `BANNER_NOTE` (where the Size tab's and the banner's Why links lead), `plain_text_violations` (re-exported from `engine::explain::plain`)), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines (since slice 3 without the N-2 and N+2 lines, which are the torque card's count row) and Why links; `slider_minimum`, the Outside diameter slider's lowest value, one expression with the torque card; `SliderMapping`, each Size slider's own mapping of a value to a design, one mapping for the controls and the torque card's steps, which sets the count whole, refusing a value a cast would change, and checks no range on the gap or the count: its caller holds the value to the slider), `torque_card.rs` (the torque card, slice 3, BL-073; pure, no egui: `card` -> `TorqueCard` (`ShownCountBelowSlider`, `ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow`, or `Rows`, one `Row` per Size control (`Control`), read by `rows`), each control one fixed step each way through its slider's own mapping (`size_controls::SliderMapping`, the one the controls apply), in the order of tests of its spec's section 3.3: a side's `Words`, or its `Change` (the stepped design's hot-low, its change in N·m and as a fraction of the design shown's, the Outside diameter and Overall length changes, whether the calibration factor changes, the counted checks whose level flips (`Flip`, over `counted_guides` at the panel's shaft-clamp pause) and the failing checks' quantities the step moves (`MarginChange`, from `guidance::check_terms`, drawn where `shows`)); `lines(auto)` draws it, a row's two N·m at one precision and the row Auto holds as `held_line`, every text a named constant (`CARD_HEADING`, `CAPTION`, the three Auto lines, the side words, `JUMP` and `CUP_JUMP`, `CALIBRATION_CHANGE`, the four one-line forms and the flip words; `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` also end the Size tab's exploration lines); `side_words` (sizing's validity rule, every row's, the keyway left out on the count row: decision D-1) and `count_step` (two magnets each way: checked, even, inside the slider), both moved from the N-2 and N+2 lines that the count row replaced; `STEP_SLACK_MM`, the one slack of the range test and the jump rules: the overlapping magnet length on the Overall length row, the cup's diameter on the Outside diameter row since the owner's FO-2), `sizing.rs` (`SizingMode`, `held_by_auto`, `SizingState` (the mode and the free variable; the target is `metal.required_min_Nm`), `SizingRunner`: debounced inverse sizing over a search range; `sizing_range`, the range Auto searches, one rule for the panel and the torque card's test table, slice 3), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell; the Size tab's one number format, slice 3 decision 10B: `fixed` (a number to a set number of decimals, with no + added), and `signed_fixed`, `signed_change` and `change_decimals` (a change to the fewest decimals, from its own (`TORQUE_DECIMALS`, two, for N·m: the readout's gap and the torque card), that keep its true sign and read a digit that is not zero, at most `CHANGE_EXTRA_DECIMALS` more: `decimals_with_a_digit`, the one cap rule, which the torque card's `shows` reads too), `shared_change_decimals` (a torque card row's two N·m at one precision), the one-decimal `signed_percent`, the format's bound since the owner's FO-4 at the slice-3 STOP (a number whose fixed form would need more than six whole digits, `MOST_WHOLE_DIGITS`, reads four significant digits with its sign, as the readouts write values, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row: `TorquePrecision`, the one precision `shared_torque_precision` finds for a row, the readout's gap a row of one, which `signed_torque_change` draws for a torque card side and the readout's margin and `torque_shortfall` for its shortfall's size, which is drawn without a sign), and `fewest_decimals_apart`, the guidance's search for more decimals), `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries; `mark_ui`, the brand mark before the title; `tab_ui`, a tab of the centre's row, underlined in the accent when it is the active one; `quiet_text`, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes; `mark_color`, the colour of the changed dot and a trace's marks), `test_support.rs` (headless egui helpers and the shared test designs, the torque card's among them with their one table, `card_designs()`; tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests; on the web it writes the settled design into the page's address), `starting_design` (the design it opens on and hands the panel as its baseline), `address_of` (the page address that stands for a design), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the calculator's own theme, "Charcoal and crimson": the token table and the visuals built from it, forced dark), `app/fonts.rs` (the app's font: a cut of Inter Regular with tabular digits, ahead of egui's built-in fonts), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
```

Replace:

```markdown
"needs" and the requirement and its gap under its label, **Cold-high torque** and the rotating
mass; `gui::tiles`, two rows of two where four would be narrower than 130 points), and under the
drawing its callout list, as on the Geometry tab, until part 2 makes the dimension tags
selections (the plan's decision D-19); on the right the check list over the Explain panel
(`gui::checks`, `gui::explain`). The check list is the design status line, then one row per
```

with:

```markdown
"needs" and the requirement and its gap under its label, **Cold-high torque** and the rotating
mass; `gui::tiles`, two rows of two where four would be narrower than 130 points), then the
drawing with its numbered dimension tags (since BL-094 the callout list and its plain notes are
the Geometry tab's, `geometry_view::drawing_ui`: a note with a level, a part not drawn or a design
check's caution, stays under the drawing while it is true); on the right the check list over the Explain panel
(`gui::checks`, `gui::explain`). The check list is the design status line, then one row per
```

Replace:

```markdown

Each of the five controls (Minimum OD, Outside diameter, Overall length, Magnet face gap and
```

with:

```markdown

Since BL-094 every result is a selection too: a click on a dimension tag of the drawing (its line
or its number), on a row of the Geometry tab's callout list or on a results-table row, or Enter or
Space on it focused (the focus arriving selects nothing), selects its result (`Readouts::select`,
`explain::result_selection`; the row of a check's own result selects the check). The entry leads
with the value and the name as the results table shows them (a tile's as the tile does), then
"What it is" (the result's help), "Checks it touches" (the check of its dashboard row), "Watch
out" (the stored-3D line, for a row that reads those fields), "Full lesson:" where its dashboard
row has a lesson, the "Equation" link where it has an equation, and with Show spreadsheet details
on its cell. Such a click no longer opens the Equation panel (the design's 5.5: the entry's
Equation link does), and those three hovers no longer end "Click to open it in the Equation
panel"; every other readout's click still opens it. The three dashboard rows no check holds
(the pull-out at 20 °C, the slip swing at the gearbox input, the cup's outside diameter) reach
their lesson from their results-table row's entry.

Each of the five controls (Minimum OD, Outside diameter, Overall length, Magnet face gap and
```

Replace:

```markdown
geometry drawing keeps 160 points while the callout list keeps three rows under it, then shrinks
(to scale), and the list scrolls in the rest; each plot takes the height its readouts line leaves
(no 150-point floor); the clamp view and the results table scroll in what is left (egui's 64-point
```

with:

```markdown
geometry drawing keeps 160 points while the callout list keeps three rows under it, then shrinks
(to scale), and the list scrolls in the rest (on the opening tab, with no list, the drawing takes
the height the tiles leave); each plot takes the height its readouts line leaves
(no 150-point floor); the clamp view and the results table scroll in what is left (egui's 64-point
```

Replace:

```markdown
  trace moves no row above it); a second click on the link ends it, and the locked free
  variable's label selects it too; a click on a result (it still opens the Equation panel) traces
  the inputs it reads (`upstream_inputs`), unless an input's trace marks it or it has no equation
```

with:

```markdown
  trace moves no row above it); a second click on the link ends it, and the locked free
  variable's label selects it too; a click on a result (a dimension tag, a callout row or a
  results-table row selects it since BL-094; any other still opens the Equation panel) traces
  the inputs it reads (`upstream_inputs`), unless an input's trace marks it or it has no equation
```

Replace:

```markdown
checks) and clicks its label (`input_entry`, since BL-092: the Explain panel shows the input's entry, its title, the link
"Highlight the results this drives" and "What it is"); then (slice 3) it opens the link again with a timer on every frame the app draws (eframe asks for an animation frame at every display refresh; the timer keeps those that make a WebGL draw call), clicks the Hot-low tile and checks the torque card's heading with its Why link, caption, Auto line and four rows in its Explain entry (the Overall length row held by Auto among them), and drags the Magnet face gap slider's handle, moving the pointer 60 times, 50 ms apart, alternately 3 px right and 3 px left of where it pressed: the drag's drawn-frame count, median, p95 and slowest frame are recorded on every run, and with `frame_check: true` (the release run) a drag that moved the gap, at least 30 frames and a p95 within one 60 fps frame (16.7 ms) are required; last (slice A, `address_follows`) it reads the address, which the app rewrote after the drag's settled edit, reloads the page and checks that the address is the same and the edited design is back (the one check of the `history.replaceState` call); `frame_check` is `true`, `false` or absent (any other value stops the run before its first step), and the calculator's result records it, so a run shows whether its check applied; the script checks the calculator agent's `passed` again from the agent's own fields (every check its schema requires true, the drag's two judged by the frame rule, no console error listed), and a step whose agent returns nothing reads every field its schema requires as false or empty; all with zero console
errors (a test decodes the pinned link and the design file, and checks that the steps name the right column's
```

with:

```markdown
checks) and clicks its label (`input_entry`, since BL-092: the Explain panel shows the input's entry, its title, the link
"Highlight the results this drives" and "What it is"), then clicks tag 1 of the drawing (`tag_entry`, since BL-094: the
Explain panel shows the face gap's entry, its name and its "Equation" link, and no Equation panel opens); then (slice 3) it opens the link again with a timer on every frame the app draws (eframe asks for an animation frame at every display refresh; the timer keeps those that make a WebGL draw call), clicks the Hot-low tile and checks the torque card's heading with its Why link, caption, Auto line and four rows in its Explain entry (the Overall length row held by Auto among them), and drags the Magnet face gap slider's handle, moving the pointer 60 times, 50 ms apart, alternately 3 px right and 3 px left of where it pressed: the drag's drawn-frame count, median, p95 and slowest frame are recorded on every run, and with `frame_check: true` (the release run) a drag that moved the gap, at least 30 frames and a p95 within one 60 fps frame (16.7 ms) are required; last (slice A, `address_follows`) it reads the address, which the app rewrote after the drag's settled edit, reloads the page and checks that the address is the same and the edited design is back (the one check of the `history.replaceState` call); `frame_check` is `true`, `false` or absent (any other value stops the run before its first step), and the calculator's result records it, so a run shows whether its check applied; the script checks the calculator agent's `passed` again from the agent's own fields (every check its schema requires true, the drag's two judged by the frame rule, no console error listed), and a step whose agent returns nothing reads every field its schema requires as false or empty; all with zero console
errors (a test decodes the pinned link and the design file, and checks that the steps name the right column's
```

Replace:

```markdown
headings, the orientation, the entry's headings and its lesson link, and no dashboard, the
toggle, the note's parts, the Geometry tab, the trace link, the last row and the card's heading, caption, Auto line, held row and row starts as the page draws them). Its other step checks the hub page at `/tools/` and the linkage
app at `/tools/linkage/`. Gate 13 of `gate.sh` runs the script's own rules under node with a stub agent and no browser
```

with:

```markdown
headings, the orientation, the entry's headings and its lesson link, and no dashboard, the
toggle, the note's parts, the Geometry tab, the trace link, the face gap's name, the last row and the card's heading, caption, Auto line, held row and row starts as the page draws them). Its other step checks the hub page at `/tools/` and the linkage
app at `/tools/linkage/`. Gate 13 of `gate.sh` runs the script's own rules under node with a stub agent and no browser
```

- [ ] **Step 2: Check.** `git diff --stat` lists the nine files and no other; `python -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ['docs/ai/backlog.yaml', 'docs/ai/02-system.yaml', 'docs/ai/03-structure.yaml', 'docs/ai/04-memory.yaml']]"` prints nothing; `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3` still reads `1111 passed`.

- [ ] **Step 3: Commit** the nine files: `docs(magcoupler): slice B stage 2 part 2b-ii-b docs sweep: README, docs/ai, the register, owner questions 56 to 59, the design's header, BL-094 in progress (BL-094 T4)`.

### Task 5: Release: the full gate, gui-smoke, the reviews, one fix wave, the merge and the push

**Model:** the `fable` controller; the text reviewer and the whole-branch reviewers are fresh-context `fable` agents; git steps run on `haiku` at low effort; the gui-smoke workflow runs its own agents on `sonnet`.

**Files:**
- Modify: `docs/ai/backlog.yaml` (Step 7)

- [ ] **Step 1: Preconditions.**

  ```bash
  git status --short
  git diff --stat main...HEAD -- magcoupling-rs/src/engine magcoupling-rs/tests reference linkage-sim-rs/src linkage-sim-rs/web
  ```

  Expected: a clean tree (the owner's untracked press analysis aside); the second command prints nothing.

- [ ] **Step 2: The full gate.** From PowerShell at the worktree's root:

  ```powershell
  $env:MAGCOUPLING_PYTHON = "C:\Users\Cole\source\repos\linkage_simulation\reference\magcoupling-py\.venv\Scripts\python.exe"
  & 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/gate.sh --full
  git checkout -- docs/chebyshev_lambda
  ```

  Expected: the last line is `GATE PASS`; gate 12 printed `oracle Python:` and not a SKIP line; gate 13 printed `all 81 checks hold`; gate 14 built both web bundles. If a gate fails, fix the cause on the branch, red test first, and run this step again. Never edit `gate.sh` to pass.

- [ ] **Step 3: gui-smoke, from the branch's own script.**
  1. Check that port 8080 is free (`netstat -ano | findstr :8080` prints nothing): another branch's release uses it on this machine. Then, in the background: `& 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/serve_web.sh 8080`.
  2. Run the Workflow tool with `scriptPath` set to this worktree's `.claude/workflows/gui-smoke.js` and the arguments `{"frame_check": true}`.
  3. Expected: `passed: true`; in `magcoupling`: `starting_design`, `look`, `first_view`, `details_toggle`, `design_file_loaded`, `check_entry`, `lesson_link_opened`, `tooltip_plain`, `input_entry`, `tag_entry`, `torque_card`, `address_follows`, `drag_frames_within_60fps` all `true`, `console_errors: []`. Keep the result's JSON in `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2biib/smoke-local.json` and the run's first screenshot beside it.
  4. Stop the server: `TaskStop` on the background task that runs it, or `Stop-Process -Id <pid>` for the PID `netstat -ano | findstr :8080` shows. Check again that the port is free.

  A `tag_entry` false is a finding about the build until shown otherwise: the step clicks the "1" on the Geometry tab's drawing (the face gap's dimension) and looks for the face gap's name and the "Equation" link in the Explain panel, and for no Equation panel. A `first_view` false that names the list under the drawing is Task 2's.

- [ ] **Step 4: The text review** (fresh context, `fable`; the design's 8.3). There is no new on-screen string; it checks that every line a result's entry draws is a text the screen drew at `main` (the value and name as the results table drew them, the help as the hover showed it, the check's mark and name, `STORED_3D_LABEL` and its hover, the lesson line, "Equation", the cell), that the only text gone is `OPEN_HINT` on the three selecting readouts, and that no engine text changed (`git diff main...HEAD -- magcoupling-rs/src/engine` is empty). It rules on two things and records what it decides: whether a hover with no line about the click still tells a person that a tag, a callout row or a table row is clickable (decision D-61, owner question 56), and whether an amber note left under the opening tab's drawing reads well without the list it closed (decision D-66, owner question 58); a change the owner should weigh goes to `docs/analyses/2026-11-18-owner-questions.md` as an open question with its conservative answer, not shipped silently. Its findings join Step 5's fix wave. Record it in `docs/analyses/2026-11-23-magcoupler-slice-b2p2bii-b-review.md` (committed in Step 7).

- [ ] **Step 5: The independent whole-branch review** (fresh context, `fable`, three lenses in parallel, each with a skeptic on every finding that is not a suggestion). Give each `git diff main...HEAD`, the design, this plan and Step 3's screenshot, and none of the builders' notes.
  1. **Conformance:** the design's 4.4, 5.1 to 5.3 for results and 5.5, goal G-1 and tests 8 and 10 of 8.2 for these targets, item by item against the diff; the decisions D-60 to D-70, each what the code does and none hiding a choice the owner should see; the nothing-lost table, row by row against the code.
  2. **Regressions:** no engine number moves; nothing reachable today is unreachable (every result's equation, through its entry or its other readouts; every callout's text, on the Geometry tab; every note; every trace a click made); no existing assertion about behaviour that stays was weakened (the list "Existing tests this plan edits"); an idle frame computes nothing new (a result's entry asks no consequences but the Hot-low tile's, as before); the keyboard's focus arriving selects nothing; the anti-goals of 2.3.
  3. **What a person sees:** a fresh render at 1024 x 700, 1280 x 800 and 1920 x 1080 of the opening tab (the drawing taller, its tags, no list; the example design's amber note under it), a tag's entry, a results-table row's entry (a number, a verdict row that selects its check, a stored-3D row, a result with no equation), the Geometry tab with its list, and the Equation panel opened from an entry: nothing cut or overlapping; the entries read in the design's order; crimson only on the brand mark and the active tab.

  One fix wave on `fable` for what Steps 4 and 5 find, each fix with its red test first; the reviewers re-read the fixes. Record the review in `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2biib/final-review.json` (gitignored).

- [ ] **Step 6: If Step 4 or 5 changed a file, run Steps 2 and 3 again.** The merge needs a gate and a smoke run of the tree that is merged.

- [ ] **Step 7: Close the item and merge.** In `docs/ai/backlog.yaml` (the day's date; say what the reviews found):

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2b-ii-b (Tasks 1 to 3 of the plan); the full gate, the smoke run and the whole-branch review are its Task 5. Filed
```

with:

```yaml
  status: fixed
  notes: "FIXED 2026-11-2X: built on branch magcoupling/overhaul-b2-part2b-ii-b; GATE PASS with --full, the gui-smoke run passed with frame_check true, the text review and the independent whole-branch review approved it. Filed
```

  (Write the day's date in place of `2026-11-2X`.) Commit it with the review record (`docs(magcoupler): BL-094 fixed: the gate, the smoke run and the reviews passed`), then merge the branch into `main` with a merge commit whose message lists the tasks' commit subjects. Remove the worktree with `git worktree remove`, never `git worktree prune`.

- [ ] **Step 8: Push, then report** (the owner's ruling of 2026-11-22: each stage goes live once the gate, the smoke run and the reviews pass, with a report afterwards; no stop). Push `main` (it deploys the site). When the `deploy-web.yml` run is green, run the smoke workflow from `main`'s script with `{"url": "https://colesorkness.com"}` and report the result; if it fails, say so plainly: the site is live with the change. The report to the owner holds: what changed on screen, in three sentences; two pictures (the opening tab with a tag's entry, a results-table row's entry); the decisions D-60 to D-70, one line each; owner questions 56 to 59; parts 2b-ii-c and 2b-iii; what was not verified (Safari and Firefox; in a browser, a callout row's and a table row's click and the keyboard's Enter on a tag: the smoke clicks one tag only).

## What this plan covers of the design

| Design item | Where |
|---|---|
| 4.4 The callout list and the notes under the drawing move to the Geometry tab; the tags stay, a click on one selects it | Tasks 1 (D-65), 2 (D-66: a note with a level stays) |
| 5.1 A result as a selection (a tile, a dimension, a table row); Esc returns the orientation | Task 1 (Esc as for every selection) |
| 5.2 A click on a dimension tag, a results-table row or a callout row selects that result or check | Task 1 (D-61, D-63) |
| 5.2 Hover selects nothing; the keyboard's focus arriving selects nothing (BL-093) | Task 1 |
| 5.3 A result's entry: title and value, What it is, Checks it touches, Watch out (the stored-3D caution), foot (Full lesson, Equation, Spreadsheet cell) | Task 1 (D-64) |
| 5.5 The Equation panel changes only through its own controls or an Equation link; a selection made elsewhere does not re-point it | Task 1 (D-61) for the three targets; the other readouts keep their click (owner question 56) |
| 2.1 G-1 and 8.2 test 8: at most 250 words at 1280 x 800 | Task 2 (D-67; 200) |
| Part 1's D-17: the three rows no check holds get their one-click home | Task 1 |
| Part 2a's D-32: which copies of a walk step's title go | Task 2's decision D-67: none (owner question 59) |
| 8.2 Test 10 (selection) for a tag, a callout row and a table row; a selection leaves the open equation where it is | Task 1 |
| 8.3 Text rules | Task 1 (glyphs, no paused clamp), Task 5 Step 4 |
| 9.1 Docs with the slice, the `gui-smoke` checklist (a dimension tag's entry) | Tasks 3 (D-68), 4 |
| 9.4 Release | Task 5 (no stop: the owner's ruling of 2026-11-22) |
| 4.3 The left column no longer changes with the centre tab; 8.2 test 9 table-driven | Part 2b-ii-c |
