# Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-a (every input a selection): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ] `) syntax for tracking.

**Goal:** Every input is a selection (the design's 5.1 and 5.2). Touching a row of Detailed inputs or Assumptions (a click on its label, a drag of its slider, focus in its field, an edit or a reset, as a Size control's touch since part 2b-i) selects its input, and the Explain panel shows its entry: its value, what it is, its hint, the checks whose fixes name it, the "pulls two ways" sentence where it applies and, with spreadsheet details on, its cell. The trace a label click started becomes the link "Highlight the results this drives" at the top of every input's entry. A check's fix link selects its input as well as showing its row. With spreadsheet details on, a tile's and a check's entry end with their spreadsheet cell (5.3). The gui-smoke run clicks an input's label and reads its entry. Nothing that is on screen today is lost (the nothing-lost table).

**Architecture:**
- **Any input's entry.** `explain::input_ui` draws a Size column row as part 2b-i built it, or any other input by its path: `input_row` builds a `SizeRow` of its own (its label as the inputs side draws it, `plain_text::label_for`; its input; no torque-card row; no lesson), so the same headings, the same "Checks it touches" and "Watch out" and the same cell line serve both. A row's value reads as its tooltip's Default line does for a blank, a selector or a text (`inputs::value_text`, factored out of `input_tooltip`). `Selection::reads_consequences` holds for a Size column row only, so a Detailed input's entry asks the `ConsequenceCache` nothing.
- **The trace as a link.** Every input's entry leads, under its title, with `explain::TRACE_LINK` ("Highlight the results this drives", the design's own words); a click returns `Asked::Trace(path)`, and the panel's `toggle_trace` (factored out of `input_row_ui`) traces the input or, on a second click, ends the trace (decision O-8). The label's click selects the input (`Selection::Input(path)`) instead of tracing, and so does any other touch of the row: an edit or a reset, or `size_controls::touched` on its widget (a click, a drag start, a change, the keyboard's focus arriving), the Size column's own rule. The label's hover loses the line that said it traced.
- **The fix link selects.** A check's fix link still asks for its input's row (`Readouts::focus_input`) and now also returns `Asked::Select(fix_selection(path))`: the input, or for the ring radius (`SIZE_TAB_ALIASES`) Outside diameter's row, whose slider the link shows.
- **The cell foot.** `cell_line` writes "Spreadsheet cell: <cell>" (or `NO_SPREADSHEET_CELL`); a tile's entry and a check's entry that is a result draw it with spreadsheet details on.
- **Unchanged:** the engine, the exports, the top bar, the Size column, the tiles, the check list, the note entries (their input links still only show the row: decision D-55), the centre tabs, the inputs side's layout and its trace banner, the Equation panel, the linkage app.

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32, egui_plot 0.33; the headless egui harness (`magcoupling-rs/src/gui/test_support.rs`, the panel tests' `Harness`, the explain tests' `frames` and `input_frames`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. This part is its sections 5.1 and 5.2 for inputs (an input as a selection; touching an input selects it; "tracing stays, as a link in the input's Explain entry"; "a fix link selects and focuses its input"), the "Spreadsheet cell" foot of 5.3 for a result and a check, test 10 of section 8.2 (a click on an input row's label puts it in the Explain panel; a hover changes nothing) and the docs and release rules of section 9. Executors read the spec's sections 5, 8.2, 8.3 and 9 beside this plan. Part 2b-i's plan (`docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`, "What follows") and BL-090's notes set the scope of part 2b-ii; this plan cuts it again (decision D-49). Backlog item: **BL-092** (`docs/ai/backlog.yaml`, filed with this plan).

**How this plan was made.** The part was built once as a prototype on a copy of `main` at `fe2bb2d` (BL-090 merged and closed), one commit per task, then BL-092 filed and the docs swept. Every block below was generated from those commits' diffs (a code task's red step is its test hunks, its green step the rest; Task 3's red step is the prompt's guard in `app.rs` and the verdict test, its green step the prompt), then applied, in this plan's order, with the apply tool, to a fresh copy of the same base with BL-092 filed. Each block applied once; each red step failed as its "Expected" says; the result equals the prototype file for file (line endings aside) and passes `cargo test` with and without the app feature (1103 lib tests with it against 1096 on the base), both native clippy runs after each task, both wasm32 clippy runs, `cargo fmt --check` and the gui-smoke verdict test (`all 79 checks hold`). Each count after a task is the replay's. Not run for this plan: gates 1 to 3 (linkage-sim-rs, untouched), gate 12 (the Python oracle), the release WASM build, the browser and the pictures. Task 5 runs them.

**Where this part applies.** On `main` at `fe2bb2d` with this plan's commit (the plan and BL-092), and nothing else that touches `magcoupling-rs/`, `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` or Task 4's nine docs. If a block does not match, stop and report: the controller regenerates the blocks (the planner's tools are listed under "Before Task 1"), it does not adapt a block in place.

## Why a part, and what follows

BL-090's part 2b-ii holds the Detailed inputs' rows, the dimension tags, the results-table rows and the callout rows as selections, the trace as an entry link, the callout list's move to the Geometry tab, the word budget of 250 with part 2a's D-32, the table-driven nothing-lost test, gui-smoke reading an input's entry, the design's 4.3 (the left column no longer changes with the centre tab), 5.2's fix link and the cell foot of a result's and a check's entry. The left column's change alone rewrites the dozens of panel tests that drive the inputs side from the Geometry and Results tabs; the tags and the callout list's move touch the drawing, its tests and the first view's word count. Together they are again more than a reviewer can hold. This plan cuts part 2b-ii in three (decision D-49) and is the first. It changes the Explain panel's input entry, the input row's label and the smoke prompt, ships on its own and loses nothing: the trace is one more click away, in the input's entry, and every other text stays where it is.

**What follows, each with its own plan and backlog item:**

**Part 2b-ii-b (the drawing's selections and the word budget):**
1. Dimension tags, results-table rows and callout rows as selections (5.1, 5.2): a result's entry for any result (today a tile's only), with the "Spreadsheet cell" foot this part gives a tile's. A click on a table row or a callout opens its equation today and traces its inputs; the design keeps the Equation panel's own controls and the "Equation" link (5.5), so that plan decides what such a click still opens.
2. With the tags clickable, the callout list (and the notes under the drawing) leaves the opening tab for the Geometry tab (4.4; part 1's decision D-19), and the three dashboard rows of part 1's D-17 get their one-click home.
3. The word budget of 250 at 1280 x 800 (goal G-1, test 8; 272 on `main`), and which copies of a Start here step's title go during the walk (part 2a's D-32).
4. gui-smoke reads a dimension tag's entry (part 2b-i's D-45 deferred it with the input's; this plan's D-58 takes the input's only).

**Part 2b-ii-c (the left column and the nothing-lost test):**
1. The left column no longer changes with the centre tab (4.3): the Size column on every tab, the workflow groups only inside Detailed inputs; the trace's banner and Clear trace placed with it (decision D-52 keeps them where they are until then); the `InputsView` tabs go.
2. The full table-driven nothing-lost test of test 9 (every top-bar action, every `CentreView`, every shown note in Learn, every `DASHBOARD` row's home, every input in the Size column or Detailed inputs, the Assumptions view from the opening tab).

**Part 2b-iii (the Plots tab and the look)**, unchanged from part 2b-i's list: the Plots tab with its picker (C-9), the plots' red reference lines (C-3), egui_plot's fading tick numbers (E-7), the controls' outlines (E-4, R-6), a SemiBold face (S-1), the page shell's loading line (S-4), BL-086, BL-088, BL-089. BL-083, BL-084 and BL-091 stay open (BL-091, the Detailed inputs fold's nested scroll area, is 2b-ii-c's to weigh with the column).

## What the prototype showed

1. **The face gap is one entry.** The Key design row of the face gap, a one-input Size row (part 2b-i's D-35), selects `Selection::Input("metal.face_gap_mm")`: the Size column's gap entry, with its card row. Magnets per ring and Required torque do the same from their Detailed inputs rows. The ring radius and the magnet length, which Outside diameter and Overall length write, are inputs of their own: their rows select their own entries.
2. **The link must lead.** At 1280 x 800 a Size row's entry (its help, its lines, its card) is taller than the Explain panel under the check list; a link at the entry's foot would be out of view, and egui paints no text scrolled out of view, so neither a person nor the harness would see it. Under the title it is on screen whatever the entry holds.
3. **Every input traces.** Every input is a leaf of the equation registry (`trace::tests` already holds it), so every input's entry draws the link; Minimum OD, which writes no input, is the one row without it. An input whose results have no equation record (the drive torque) traces none, and the banner says so, as before.
4. **A selector's value.** `format_value` writes a selector's code ("1"); the entry writes its choice as the tooltip's Default line does ("1 = ..."), and "blank" for a blank optional input such as the measured drag.
5. **The label no longer traces, and no test lost its subject.** Ten panel tests traced by a label click. Each now traces through the entry's link (`Harness::trace_input`: a click on the label, then on the link) and keeps every assertion about the trace; the two that held "a second click on the label ends the trace" hold it of the link, and that the label's second click keeps the trace.
6. **A fix on the ring radius.** At 12 poles the fit's first fix raises Outside diameter: its link opens the opening tab (contract C-4) and now selects Outside diameter's row, whose slider it shows.
7. **The smoke step.** On the Geometry tab after the design file's load (no sizing), the first Key design row is the face gap: its label's tooltip step, then a click on it, shows the gap's entry with the link and "What it is". The prompt types the link once (`TRACE_LINK`), held equal to the app's.

No pictures were drawn for this plan: Task 5's review renders a Detailed input's entry and the Size column's gap entry at the three window sizes.

## Global Constraints

- **No engine number moves** (the design's 2.3). No file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` changes.
- **The linkage app is not touched** (O-1). No file under `linkage-sim-rs/src/` or `linkage-sim-rs/web/` changes.
- **The panel sets no theme and no font** (2.2, 6.3). Code under `magcoupling-rs/src/gui/` reads `ui.visuals()`.
- **No new explanatory text** (5.3). The entries draw texts the screen drew before: an input's label, its help (its row's hover), its hint (its Key design row's line), the guides' labels and state words, the "pulls two ways" sentence, `SPREADSHEET_CELL` and the cells. The one new on-screen string is `TRACE_LINK` ("Highlight the results this drives", the design's own words in 5.2); it passes the plain-words guard and the glyph guard (Task 1). One string leaves the screen: the hover's last line `TRACE_HINT` (decision D-53).
- **Crimson is never text, never on a value, never in the check list** (C-5). An entry's link to a check draws the check's state mark and word, as part 2b-i's does.
- **The selection is view state** (5.1): an input selected is in no design file, share link or undo step. Neither a label's click nor the trace link edits the design or makes an undo step.
- **A hover selects nothing** (C-2): a row's label keeps its tooltip and selects only on a click.
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
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2bii/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Shell.** On this machine `bash` typed in PowerShell is WSL, which fails on the checkout's line endings. Shell scripts run through Git Bash: `& 'C:\Program Files\Git\bin\bash.exe' <script>`. Python scripts are written with the Write tool, not a here-document.
- **Never run `cargo fmt` inside `linkage-sim-rs`.** `magcoupling-rs` is fmt-clean; keep it so.
- **Docs ship with the code** (repository rule): Task 4 sweeps the README and `docs/ai/`, the register, the owner-question list and the design's status line.

## Model per task

CLAUDE.md section 5. BL-092 is `risk: mechanical` (decision D-57): the part shows no new number and makes no new physical claim; its one new string is the design's own "Highlight the results this drives", and every other line its entries draw is reviewed text the screen drew. So the tasks that transcribe the plan's exact blocks run on `sonnet`, and every review and the release run on `fable`. A `sonnet` task that ends blocked, leaves the gate red or is rejected in review is redone on `fable` (escalate, never downgrade).

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | Any input's entry, the trace link, the fix link's selection, the cell feet (`input_row`, `TRACE_LINK`, `Asked::Trace`, `fix_selection`, `cell_line`, `inputs::value_text`, `toggle_trace`) | `sonnet` |
| 2 | A row's label selects its input; the trace goes through the entry's link | `sonnet` |
| 3 | gui-smoke reads an input's entry (`input_entry`) | `sonnet` |
| 4 | Docs sweep | `sonnet` |
| 5 | Release: the full gate, gui-smoke, the text review, the independent whole-branch review, one fix wave, the merge and the push | `fable` controller; fresh-context `fable` reviewers; `haiku` for git; the smoke workflow's own agents on `sonnet` |

Each task's diff is reviewed by a fresh-context reviewer before the next task starts (`sonnet` for these mechanical diffs, CLAUDE.md section 5); it checks the diff against the task's blocks and nothing else. A task that ends blocked or leaves a test red stops the build and goes back to the controller.

## Review Focus

Eight inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named.

1. **The face gap from two places.** Its Key design row and its Size column row select one entry (D-35), with the card row; a Detailed input that a size writes (the ring radius, the magnet length) selects its own. Task 2: `a_click_on_an_input_row_s_label_selects_its_input_and_a_hover_selects_nothing`; Task 1: `an_input_s_entry_leads_with_the_link_that_traces_it`.
2. **A hover over a label.** It shows the tooltip (without the old trace line) and selects nothing; a click selects, in the same frame. Task 2: the test above, `an_input_row_s_hover_names_its_path_and_cell_only_with_the_spreadsheet_details`.
3. **The locked free variable** (Auto on the magnet length). Its row is disabled but its label selects it, and its entry traces it. Task 2: `the_locked_free_variable_s_label_still_selects_it_and_its_entry_traces_it`.
4. **A fix on the ring radius.** It opens the opening tab and selects Outside diameter's row, not a Detailed row. Task 1: `a_fix_link_selects_its_input_and_asks_for_its_row`, `an_outside_diameter_fix_opens_the_size_tab_and_edits_nothing`.
5. **An input that reaches no result** (the drive torque). The link still shows; the trace reaches none and its banner says so; "Traced only" says the trace marks nothing. Task 2: `an_input_whose_results_have_no_equation_record_traces_none_and_says_so`.
6. **A selector, a blank optional input, an input with no hint or no help.** The value reads in words; a heading with nothing under it is left out. Task 1: `a_detailed_input_s_entry_is_its_value_label_help_hint_checks_and_cell`.
7. **A check that is no result** (Magnets fit). With the details on it draws no cell line, as it draws no Equation link. Task 1: `a_result_s_and_a_check_s_entry_show_their_cell_only_with_the_spreadsheet_details`.
8. **A touch that is not the label's click.** The keyboard's focus arriving on a row's slider (no edit), a drag of it and a click on its reset each select the input; typing in a text field selects it, and the field's Esc keeps that selection. Task 2: `touching_a_detailed_row_s_widget_selects_its_input`, `esc_in_a_text_field_is_the_field_s_and_keeps_the_selection`, `the_selection_is_a_view_choice_in_no_file_link_or_undo_step`.

## Decisions this plan takes

Each is a reading of the design where it was open, numbered after part 2b-i's D-34 to D-48 (one stage, one list). The owner may overrule any of them; Task 5 lists them in its report. The ones that need a row in the register have one (Task 4).

| Id | Decision | Why |
|---|---|---|
| D-49 | Part 2b-ii is built in three; this plan is part 2b-ii-a (every input a selection: the rows' labels, the trace link, the fix link, the cell feet, gui-smoke's input entry). Parts 2b-ii-b and 2b-ii-c are listed under "What follows". Register row A-29. | Part 2b-ii entire is again more than one reviewer can hold (D-34's reasoning); the left column's change alone rewrites the inputs side's tests. This part changes the input entry and the label's click, and loses nothing. The conservative answer to Q-4 is smaller releases. |
| D-50 | Touching a row of Detailed inputs or Assumptions (its label's click; its widget's click, drag start, change or focus, `size_controls::touched`; an edit or a reset) selects `Selection::Input(path)`; a locked row's widget is disabled, so only its label selects it. Its entry is the input column of 5.3 with what the screen has for it: the value and the label, "What it is" (its help, the row's hover), "What sets it" (its hint, where the Key design group draws one), "Checks it touches", "Watch out" and, with the details, its cell. No torque-card row, no "Full lesson:" and no "Equation" link. | 5.3 leaves out a heading an entry has nothing for: only the four size controls have a card row, no input has a note of its own (D-42) and an input has no equation (D-41). Keying by path makes a one-input Size row and its Detailed row one entry (D-35). The touch is the Size column's (part 2b-i's D-36), so an input selects alike in both places, as 5.2 lists: "a click on its label, a drag of its slider, focus in its field". |
| D-51 | Every input's entry leads, under its title, with the link "Highlight the results this drives"; a click traces the input and a second click ends the trace (O-8). Outside diameter and Overall length trace the input their slider writes; Minimum OD has no link. | 5.2's words; under the title it is in view at 1280 x 800 whatever the entry holds (the prototype, item 2). The O-8 toggle is the label's behaviour moved to the link. |
| D-52 | The trace's banner and Clear trace stay at the foot of the inputs side on the other tabs; on the opening tab a trace draws its frames and no banner, as since part 2b-i. Register row A-30; owner question 55. | No new text and no second copy of the banner. Part 2b-ii-c, which stops the inputs side swapping, must place the banner anyway. The cost, until then: on the opening tab, once the selection moves on, no off switch is in view (select the input again and click its link, or use Clear trace on another tab), and a trace that reaches no result (the drive torque) shows nothing there. Owner question 55. |
| D-53 | The label's hover loses its last line `TRACE_HINT` ("Click the label to highlight the results it drives that have an equation"); no line takes its place. Register row A-31; owner question 53. | The line would be false; a replacement is new text for review. The entry's link says what it does. |
| D-54 | A check's fix link selects its input (`fix_selection`; the ring radius selects Outside diameter's row) and still asks for its row, which opens the fold or the tab that holds it; the check's entry gives way to the input's. | 5.2: "a fix link selects and focuses its input". The input's entry links back to the check under "Checks it touches". |
| D-55 | A teaching note's "What to change" input links keep showing the row only; the note stays in the Explain panel. Register row A-32; owner question 54. | 5.2 names the fix link; a note is a lesson being read (the walk selects one per step), and a click that replaced it would lose the reader's place. |
| D-56 | With the details on, a tile's entry ends with "Spreadsheet cell:" and its result's cell (or `NO_SPREADSHEET_CELL`), a check's entry likewise where the check is a result (its Equation link's rule); the Magnets fit check, which is none, draws no cell line. | 5.3's foot. A cell line for a check no result holds would claim a provenance the tool does not know. |
| D-57 | BL-092 is `risk: mechanical`; its build tasks run on `sonnet`, its reviews and release on `fable`. | No new number or physical claim (the design's header). |
| D-58 | The gui-smoke prompt gains `input_entry`: after the tooltip step on the Geometry tab, a click on the same label; the Explain panel shows the input's title, the link and "What it is". The link's text is typed once in the script (`TRACE_LINK`), held equal to the app's. | 5.2 in a browser, with no new navigation: the step reuses the tooltip's target. Part 2b-i's D-45 deferred it to this part. |
| D-59 | `Selection::reads_consequences` holds for a Size column row only: a Detailed input's entry asks the `ConsequenceCache` nothing. | An input that is no Size row draws no what-moves-what line and no card row; asking would compute for nothing (test 14 of 8.2). |

## Nothing lost

Every text and action this part moves, and where it is after. Anything not listed (every row, label, value, slider, tooltip line other than the trace line, the Size column, the trace's banner and Clear trace, the tiles, the check list, the note entries) does not move. The "Test" column names the test that holds the new home.

| Today | After this part | Test |
|---|---|---|
| A click on a Detailed inputs or Assumptions row's label traces its input; a second click ends the trace | The link "Highlight the results this drives" at the top of the input's entry traces it; a second click on the link ends it. The label's click selects the input | `panel::tests::tracing_an_input_frames_the_results_it_drives_until_a_second_click`, `a_trace_moves_no_input_row_and_a_second_click_on_the_label_traces_nothing`, `an_input_s_entry_link_traces_it_until_a_second_click_and_edits_nothing`; `explain::tests::an_input_s_entry_leads_with_the_link_that_traces_it` |
| The locked free variable's label traces it | Its label selects it; its entry traces it | `the_locked_free_variable_s_label_still_selects_it_and_its_entry_traces_it` |
| The hover's last line "Click the label to highlight the results it drives that have an equation" | Gone (it would be false; D-53); the link's own words say what it does | `an_input_row_s_hover_names_its_path_and_cell_only_with_the_spreadsheet_details`, `the_changed_dot_a_row_s_reset_and_the_default_line_follow_the_baseline` |
| A check's fix link shows its input's row (and opens the opening tab for the ring radius) | The same, and it selects the input (Outside diameter's row for the ring radius) | `a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing`, `an_outside_diameter_fix_opens_the_size_tab_and_edits_nothing`; `explain::tests::a_fix_link_selects_its_input_and_asks_for_its_row` |
| A check's entry stays in the Explain panel after a click on one of its fix links | The input's entry takes its place (D-54); its "Checks it touches" lists the check, one click back | `a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing` (Task 2: the link back, clicked, selects the check) |
| Every trace test's assertion (frames, banner, counts, "Traced only", the table's filter) | The same assertions, the trace started through the entry's link | the ten tests of "Existing tests this plan edits", Task 2 |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/explain.rs` | `TRACE_LINK`; `Asked::Trace`; `Selection::Input` for any input, `reads_consequences` for a Size row only; `input_ui` for any input with the trace link; `input_row`, `fix_selection`, `cell_line`; `row_value` through `value_text`; a check's fix link selects; a tile's and a check's cell foot; tests | 1 |
| `magcoupling-rs/src/gui/inputs.rs` | `value_text`, factored out of `input_tooltip` | 1 |
| `magcoupling-rs/src/gui/panel.rs` | `Asked::Trace` handled, `toggle_trace` (1); `input_row_ui` selects on a touch instead of tracing; the trace tests through `Harness::trace_input`; two new selection tests (2) | 1, 2 |
| `magcoupling-rs/src/gui/input_ui.rs` | `TRACE_HINT` gone; the label's hover is the tooltip; doc comments | 2 |
| `magcoupling-rs/src/gui/trace.rs` | Two comments | 2 |
| `magcoupling-rs/src/app.rs` | The smoke prompt's guard: `TRACE_LINK` typed once, the input step after the tooltip, collected before the reopen | 3 |
| `.claude/workflows/gui-smoke.js` | `input_entry` (schema, prompt, passed list), `TRACE_LINK`, the description and comments | 3 |
| `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` | `CHECKS` gains `input_entry` | 3 |
| `magcoupling-rs/README.md`, `docs/ai/*`, the register, the owner questions, the design's header | The docs sweep | 4 |
| `docs/ai/backlog.yaml` | BL-092 in progress (4), fixed (5) | 4, 5 |

**Existing tests this plan edits, and how.** No assertion about behaviour that stays is weakened; where an action moved (the trace from the label to the link), the test takes the new path and keeps every assertion about the trace.
- `panel::tests`, Task 1: `a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing` also holds that the link selects its input; `an_outside_diameter_fix_opens_the_size_tab_and_edits_nothing` also holds that it selects Outside diameter's row.
- `explain::tests`, Task 1: `the_panel_s_own_strings_keep_the_text_rules` checks `TRACE_LINK` too.
- `panel::tests`, Task 2: `clicking_an_input_s_label_frames_the_results_it_drives_until_a_second_click` becomes `tracing_an_input_frames_the_results_it_drives_until_a_second_click` (the label's click selects and traces nothing; the link traces; a second click on the link ends it; every frame and count assertion kept); `a_trace_moves_no_input_row_so_a_second_click_on_the_label_ends_it` becomes `a_trace_moves_no_input_row_and_a_second_click_on_the_label_traces_nothing` (the row, the toggle and the banner's place held as before; the label's second click keeps the trace, the link's ends it); `the_locked_free_variable_s_label_still_traces_it` becomes `the_locked_free_variable_s_label_still_selects_it_and_its_entry_traces_it`; `the_panel_s_marks_are_the_host_s_link_colour`, `the_traced_filter_shows_the_results_the_trace_marks`, `an_input_whose_results_have_no_equation_record_traces_none_and_says_so`, `a_trace_filter_left_ticked_off_the_table_does_not_filter_a_later_trace`, `a_result_traced_off_the_table_also_turns_the_trace_filter_off`, `reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows` and `a_search_that_hides_the_traced_rows_says_so_and_the_headings_count_what_it_shows` trace through `Harness::trace_input` (a click on the label, then on the link); `an_input_row_s_hover_names_its_path_and_cell_only_with_the_spreadsheet_details` and `the_changed_dot_a_row_s_reset_and_the_default_line_follow_the_baseline` expect the tooltip without the trace line; `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` and `the_panel_s_own_strings_read_in_plain_words` check `TRACE_LINK` where they checked `TRACE_HINT`; `esc_in_a_text_field_is_the_field_s_and_keeps_the_selection` and `the_selection_is_a_view_choice_in_no_file_link_or_undo_step` expect the edited input selected (typing or a nudge touches it) and still hold that Esc in the field, the undo, reset all and the load keep the selection; `a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing` also clicks the input entry's link back to the check; `the_check_list_and_the_explain_panel_compute_nothing_on_an_idle_frame` also idles with a Detailed input's entry (D-59). `trace::tests`: one comment.
- `app::tests::the_smoke_test_share_link_opens_its_design`, Task 3: the first page's console messages are collected after "input_entry=true was judged" (the step judged last on that page), where it read "tooltip_plain=true was judged"; a new block holds `TRACE_LINK` and the step's place. `gui_smoke_verdict_test.cjs`, Task 3: `CHECKS` lists `input_entry`, so its case "input_entry false with the agent's passed true" runs.

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` after this plan has merged, create the branch `magcoupling/overhaul-b2-part2b-ii-a` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: `test result: ok. 1096 passed; 0 failed; 6 ignored` and `all 77 checks hold`. Another count means `main` is not the base the blocks were checked on: stop and report.

- [ ] **Step 3: Check the base.** `git log --oneline fe2bb2d..main -- magcoupling-rs .claude/workflows/gui-smoke.js linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs docs/ai docs/analyses/2026-11-10-conservative-assumptions.md docs/analyses/2026-11-18-owner-questions.md docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` lists this plan's commit (it touches `docs/ai/backlog.yaml`) and nothing else. If it lists more, stop: the controller regenerates the blocks on that `main` with the planner's tools (gitignored, in the main checkout's `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2bii/`: `gen_blocks.py`, `frags.py`, `replay.py`, the scripts that built each task's state, and `seq.bundle`, the prototype's commits: `s0` the base, `s1` to `s3` the code tasks, `s3b` BL-092 filed, `s4` the docs) before Task 1.

## Tasks

### Task 1: Any input's entry, the trace link, the fix link's selection, the cell feet

**Model:** `sonnet` (BL-092 is `risk: mechanical`; the task transcribes exact blocks).

`explain::input_ui` draws any input by its path (`input_row`), leads every input's entry with `TRACE_LINK` (`Asked::Trace`, which the panel's new `toggle_trace` answers), and writes a row's value through `inputs::value_text` (factored out of `input_tooltip`). A check's fix link also selects its input (`fix_selection`); a tile's and a check's entry draw their cell with the details on (`cell_line`). Nothing selects a Detailed input yet (Task 2 wires the label), but the Size column's rows and the fix links reach the new entry and the link at once.

**Files:**
- Modify: `magcoupling-rs/src/gui/explain.rs`, `magcoupling-rs/src/gui/inputs.rs`, `magcoupling-rs/src/gui/panel.rs`

**Interfaces:**
- Produces: `explain::TRACE_LINK`; `Asked::Trace(&'static str)`; `explain::{input_row, fix_selection, cell_line}` (private); `inputs::value_text(&Value, &InputMeta) -> String`; `MagcouplingPanel::toggle_trace(&mut self, &str)` (private).
- Changes: `Selection::reads_consequences` is true for the Hot-low tile and a Size column row only; `entry_ui` returns a fix link's `Asked::Select`.

- [ ] **Step 1 (red): the entries' tests.** The blocks add four tests in `explain::tests` with their helper `detailed_inputs`, add `TRACE_LINK` to the text rules' list, add the panel test `an_input_s_entry_link_traces_it_until_a_second_click_and_edits_nothing`, and add the selection to the two fix-link panel tests.

**`magcoupling-rs/src/gui/explain.rs`**

Replace:

```rust
            LEARN,
            crate::gui::checks::HEADING,
```

with:

```rust
            LEARN,
            TRACE_LINK,
            crate::gui::checks::HEADING,
```

Replace:

```rust
            }
        }
    }
}

```

with:

```rust
            }
        }
    }

    // ------------------------------------------------------------------ any input's entry

    /// The inputs of the detailed inputs that the tests below select: every Key design input
    /// that is no Size column row's key, and each assumption's first input.
    fn detailed_inputs() -> Vec<&'static str> {
        let catalogue = crate::gui::inputs::InputCatalogue::get();
        catalogue
            .key_design
            .iter()
            .map(|entry| entry.path.as_str())
            .filter(|path| size_row(path).is_none())
            .chain(ASSUMPTIONS.iter().map(|a| a.paths[0]))
            .collect()
    }

    #[test]
    fn a_detailed_input_s_entry_is_its_value_label_help_hint_checks_and_cell() {
        // Spec 5.1 to 5.3 for a row of the detailed inputs or the assumptions: the input's
        // value leading its label, "What it is" (its help, as its row's hover shows it), "What
        // sets it" (its hint, as the Key design row draws it), the checks whose fixes name it
        // and, with the spreadsheet details, its cell. It has no torque-card row and no lesson,
        // and its selection reads no consequences. A selector's value is its choice in words.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let fixes = all_fixes(&inputs, &results, false);
        for path in detailed_inputs() {
            let meta = crate::gui::inputs::InputCatalogue::get()
                .entry(path)
                .expect("an input")
                .meta;
            let texts = input_texts(&inputs, path, false);
            let has = |text: &str| texts.iter().any(|t| t == text);
            assert!(has(label_for(path)), "{path}: {texts:?}");
            let help = help_for(path);
            assert_eq!(has(WHAT_IT_IS), !help.is_empty(), "{path}");
            assert!(help.is_empty() || has(help), "{path}: {texts:?}");
            let hint = hint_for(path);
            assert_eq!(has(WHAT_SETS_IT), hint.is_some(), "{path}");
            assert!(hint.is_none_or(has), "{path}: {texts:?}");
            for absent in [WHAT_A_CHANGE_DOES, CARD_HEADING, FULL_LESSON] {
                assert!(!has(absent), "{path}: {absent}");
            }
            let named = CHECK_GUIDES
                .iter()
                .zip(&fixes)
                .filter(|(_, list)| list.iter().any(|&(p, _)| p == path))
                .count();
            assert_eq!(has(CHECKS_IT_TOUCHES), named > 0, "{path}");
            let cells: Vec<String> = input_texts(&inputs, path, true)
                .into_iter()
                .filter(|t| t.starts_with(SPREADSHEET_CELL))
                .collect();
            assert_eq!(cells, [cell_line(meta.cell)], "{path}");
            assert!(!Selection::Input(path).reads_consequences(), "{path}");
        }
        let op_temp = "coupling.op_temp_C";
        let value = inputs.get(op_temp).unwrap();
        let unit = crate::gui::inputs::InputCatalogue::get()
            .entry(op_temp)
            .unwrap()
            .meta
            .unit;
        let want = value_with_unit(&value, format_value(&value), unit);
        assert!(
            input_texts(&inputs, op_temp, false).contains(&want),
            "{want}"
        );
        let backiron = "coupling.backiron";
        let meta = crate::gui::inputs::InputCatalogue::get()
            .entry(backiron)
            .unwrap()
            .meta;
        let choice = value_text(&inputs.get(backiron).unwrap(), meta);
        assert!(choice.contains(" = "), "{choice}");
        assert!(input_texts(&inputs, backiron, false).contains(&choice));
        // A blank optional input's value is the word the tooltip's Default line uses.
        let drag = "metal.measured_drag_Nm";
        assert_eq!(inputs.get(drag), Some(Value::None));
        assert!(input_texts(&inputs, drag, false).contains(&"blank".to_owned()));
        // A path that is no input draws the heading and the way back alone.
        assert_eq!(
            input_texts(&inputs, "no.such.input", false),
            [HEADING, ABOUT]
        );
    }

    #[test]
    fn an_input_s_entry_leads_with_the_link_that_traces_it() {
        // Spec 5.2: tracing stays, as a link in the input's entry, right under its title. A
        // click asks the panel to trace the input (a second click on it ends the trace, decision
        // O-8); Outside diameter and Overall length trace the input their slider writes, and
        // Minimum OD, which writes none, has no link.
        let inputs = DesignInputs::default();
        let mut keys: Vec<&'static str> = SIZE_ROWS.iter().map(|row| row.key).collect();
        keys.extend(detailed_inputs());
        for key in keys {
            let input = size_row(key).map_or(Some(key), |row| row.input);
            let ctx = egui::Context::default();
            let (output, asked) = input_frames(&ctx, &inputs, key, false, None, None, vec![]);
            assert_eq!(asked, None);
            let texts = drawn_texts(&output);
            let links = texts.iter().filter(|t| *t == TRACE_LINK).count();
            assert_eq!(links, usize::from(input.is_some()), "{key}: {texts:?}");
            let Some(input) = input else { continue };
            let link = text_rect(&output, TRACE_LINK).unwrap();
            let label = size_row(key).map_or_else(|| label_for(key), |row| row.label);
            let title = crate::gui::test_support::last_characters(&output, label)[0];
            assert!(title.y < link.top(), "{key}: the title leads");
            if let Some(what) = text_rect(&output, WHAT_IT_IS) {
                assert!(
                    link.bottom() <= what.top(),
                    "{key}: the link before What it is"
                );
            }
            let (_, asked) = input_frames(
                &ctx,
                &inputs,
                key,
                false,
                None,
                None,
                click_at(link.center()),
            );
            assert_eq!(asked, Some(Asked::Trace(input)), "{key}");
        }
    }

    #[test]
    fn a_fix_link_selects_its_input_and_asks_for_its_row() {
        // Spec 5.2: a fix link selects and focuses its input. The input the Size column shows
        // under its own name (the ring radius, under Outside diameter) selects that row.
        let none: &dyn Fn() -> Vec<String> = &Vec::new;
        let guide = guide_with(Learn::None("A test reason."));
        let red = entry_at(&guide, Level::Bad, &CUP_WALL_FIX, none);
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let (asked, events) = click_in_entry(&red, &fix);
        assert_eq!(
            asked,
            Some(Asked::Select(Selection::Input("metal.cup_wall_corner_mm")))
        );
        assert_eq!(events.focus, Some("metal.cup_wall_corner_mm"));
        let (ring, _) = SIZE_TAB_ALIASES[0];
        let ring_fix = [(ring, Direction::Raise)];
        let red = entry_at(&guide, Level::Bad, &ring_fix, none);
        let (asked, events) = click_in_entry(&red, &fix_text(ring, Direction::Raise));
        assert_eq!(
            asked,
            Some(Asked::Select(Selection::Input(DIAMETER_HINT_KEY)))
        );
        assert_eq!(events.focus, Some(ring));
        assert_eq!(fix_selection(FACE_GAP), Selection::Input(FACE_GAP));
    }

    #[test]
    fn a_result_s_and_a_check_s_entry_show_their_cell_only_with_the_spreadsheet_details() {
        // Spec 5.3's "Spreadsheet cell" foot for a result and a check, shown only with the
        // details on: the result's cell, or the line that says the spreadsheet has none; a
        // check that is no result (the magnets fit) has no cell line, as it has no Equation
        // link.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let cells = |selection: Selection, details: bool| -> Vec<String> {
            let (output, _, _) = frames(&ctx, &inputs, selection, details, vec![]);
            drawn_texts(&output)
                .into_iter()
                .filter(|t| t.starts_with(SPREADSHEET_CELL))
                .collect()
        };
        for tile in Tile::ALL {
            let selection = Selection::Result(tile.path());
            assert!(cells(selection, false).is_empty(), "{tile:?}");
            let info = result_info(tile.path()).expect("a result");
            assert_eq!(
                cells(selection, true),
                [cell_line(info.cell.as_deref())],
                "{tile:?}"
            );
        }
        for guide in counted_guides(false) {
            let selection = Selection::Check(guide.check);
            assert!(cells(selection, false).is_empty(), "{}", guide.check);
            let want: Vec<String> = result_info(guide.check)
                .map(|info| cell_line(info.cell.as_deref()))
                .into_iter()
                .collect();
            assert_eq!(cells(selection, true), want, "{}", guide.check);
        }
        assert!(result_info(FIT_CHECK).is_none());
        let pullout = result_info(Tile::Pullout.path()).and_then(|info| info.cell.clone());
        assert!(pullout.is_some(), "the pull-out has a cell");
    }
}

```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
        assert!(trace_marks(&harness, &output).is_empty());
        assert_eq!(harness.panel.design(), Design::default());
```

with:

```rust
        assert!(trace_marks(&harness, &output).is_empty());
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn an_input_s_entry_link_traces_it_until_a_second_click_and_edits_nothing() {
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::trace::TraceKind;
        // Spec 5.2: the trace is a link in the input's Explain entry. A click traces the input,
        // a second click ends it (decision O-8); neither edits the design or makes an undo step,
        // and the selection stays.
        let mut harness = Harness::new();
        harness.panel.selection = Selection::Input(FACE_GAP);
        harness.click_text(TRACE_LINK);
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (trace.kind, trace.source.as_str()),
            (TraceKind::Input, FACE_GAP)
        );
        let output = harness.frame(Vec::new());
        assert!(
            !trace_marks(&harness, &output).is_empty(),
            "its frames show"
        );
        harness.click_text(TRACE_LINK);
        assert_eq!(harness.panel.trace, None);
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        assert_eq!(harness.panel.design(), Design::default());
```

Replace:

```rust
                    "{view:?}"
                );
                assert!(
                    !harness.panel.explorer.open,
```

with:

```rust
                    "{view:?}"
                );
                // Spec 5.2: the link selects its input too.
                assert_eq!(harness.panel.selection, Selection::Input(path), "{view:?}");
                assert!(
                    !harness.panel.explorer.open,
```

Replace:

```rust
        assert_eq!(harness.panel.explorer.focus(), None, "no row is asked for");
        assert!(!harness.panel.explorer.open);
```

with:

```rust
        assert_eq!(harness.panel.explorer.focus(), None, "no row is asked for");
        // Spec 5.2: the link selects the control it shows.
        assert_eq!(harness.panel.selection, Selection::Input(DIAMETER_HINT_KEY));
        assert!(!harness.panel.explorer.open);
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 15
  ```

  Expected: the build fails with 11 errors, the first "unresolved import `crate::gui::explain::TRACE_LINK`": the green step adds what the tests call.

- [ ] **Step 3 (green): the entry, the link, the fix link, the feet.**

**`magcoupling-rs/src/gui/explain.rs`**

Replace:

```rust
//! it, where two of them pull it opposite ways the sentence that says so, its Why link's note
//! and, with the spreadsheet details, its input's cell.
//!
```

with:

```rust
//! it, where two of them pull it opposite ways the sentence that says so, its Why link's note
//! and, with the spreadsheet details, its input's cell. A row of the detailed inputs or the
//! assumptions selects the same form, keyed by its input's path: its value and label, its help,
//! its hint, its checks, its "Watch out" and its cell. An input's entry leads with the link that
//! traces it ([`TRACE_LINK`]), and a check's fix link selects its input (spec 5.2). With the
//! spreadsheet details a result's and a check's entry end with their cell too.
//!
```

Replace:

```rust
    self, CHECK_GUIDES, CheckGuide, Consequences, Direction, FIT_CHECK, Learn, NO_FIX,
    SIZE_TAB_NOTE, fix_text, guide_notes, input_trade_offs, row_learn,
};
```

with:

```rust
    self, CHECK_GUIDES, CheckGuide, Consequences, Direction, FIT_CHECK, Learn, NO_FIX,
    SIZE_TAB_ALIASES, SIZE_TAB_NOTE, fix_text, guide_notes, input_trade_offs, row_learn,
};
```

Replace:

```rust
use crate::gui::input_ui::weak_line;
use crate::gui::inputs::InputCatalogue;
use crate::gui::look;
```

with:

```rust
use crate::gui::input_ui::weak_line;
use crate::gui::inputs::{InputCatalogue, value_text};
use crate::gui::look;
```

Replace:

```rust
    DIAMETER_HINT_KEY, LENGTH_HINT_KEY, MINIMUM_OD_HINT_KEY, NO_SPREADSHEET_CELL, help_for,
    hint_for,
};
```

with:

```rust
    DIAMETER_HINT_KEY, LENGTH_HINT_KEY, MINIMUM_OD_HINT_KEY, NO_SPREADSHEET_CELL, help_for,
    hint_for, label_for,
};
```

Replace:

```rust
};
use crate::gui::readouts::Readouts;
use crate::gui::size_controls::{
```

with:

```rust
};
use crate::gui::readouts::{Readouts, registry};
use crate::gui::size_controls::{
```

Replace:

```rust
use crate::gui::size_controls::{
    LINKED_HOUSING, SizeRow, floor_line, row_help, size_row, slider_minimum,
};
```

with:

```rust
use crate::gui::size_controls::{
    LINKED_HOUSING, SIZE_ROWS, SizeRow, floor_line, row_help, size_row, slider_minimum,
};
```

Replace:

```rust

/// The head of an input's foot line with the spreadsheet details on (spec 5.3): its workbook
/// cell, or the line that says the spreadsheet has none.
```

with:

```rust

/// The head of an entry's foot line with the spreadsheet details on (spec 5.3): its workbook
/// cell, or the line that says the spreadsheet has none.
```

Replace:

```rust

/// The top bar's button that lists the teaching notes here, and the list's heading (UX
```

with:

```rust

/// An input's link that traces it (spec 5.2): it frames the results the input drives, and a
/// second click ends the trace (decision O-8).
pub const TRACE_LINK: &str = "Highlight the results this drives";

/// The top bar's button that lists the teaching notes here, and the list's heading (UX
```

Replace:

```rust
    NoteList,
    /// A row of the Size column, by its key ([`SizeRow::key`]).
    Input(&'static str),
```

with:

```rust
    NoteList,
    /// An input: a row of the Size column, by its key ([`SizeRow::key`]), or a row of the
    /// detailed inputs or the assumptions, by its input's path.
    Input(&'static str),
```

Replace:

```rust

    /// The key of the Size column's row selected, if one is.
    pub fn input(self) -> Option<&'static str> {
```

with:

```rust

    /// The key of the input selected, if one is.
    pub fn input(self) -> Option<&'static str> {
```

Replace:

```rust
    /// panel asks its cache for them only then. A Size column row's entry draws its
    /// what-moves-what lines and its torque-card row from them.
    pub fn reads_consequences(self) -> bool {
```

with:

```rust
    /// panel asks its cache for them only then. A Size column row's entry draws its
    /// what-moves-what lines and its torque-card row from them; another input's entry reads
    /// none.
    pub fn reads_consequences(self) -> bool {
```

Replace:

```rust
    pub fn reads_consequences(self) -> bool {
        self.tile() == Some(Tile::HotLow) || self.input().is_some()
    }
```

with:

```rust
    pub fn reads_consequences(self) -> bool {
        self.tile() == Some(Tile::HotLow) || self.input().and_then(size_row).is_some()
    }
```

Replace:

```rust
    StartHere,
}
```

with:

```rust
    StartHere,
    /// Trace the input at this path, or end its trace if it is the one traced (spec 5.2,
    /// decision O-8).
    Trace(&'static str),
}
```

Replace:

```rust

/// The entry of the Size column's row `key` (spec 5.3's input column): its value and its
/// label; what it is (its label's hover); what sets it (its hint, its what-moves-what lines,
/// for Minimum OD the floor line, for Outside diameter and Overall length the footnote on what
/// the controls move); what a change does (its torque-card row); the checks whose fixes name
/// its input; where two of them move it opposite ways, the sentence that says so ("Watch
/// out"); and the foot (its Why link's note, and with the spreadsheet details the cell of a
/// row that is one input). Nothing for a key that is no row's.
fn input_ui(
```

with:

```rust

/// The entry of the input `key` (spec 5.3's input column): its value and its label; for an
/// input the trace knows, the link that traces it ([`TRACE_LINK`]); what it is (its label's
/// hover); what sets it (its hint, its what-moves-what lines, for Minimum OD the floor line,
/// for Outside diameter and Overall length the footnote on what the controls move); what a
/// change does (its torque-card row); the checks whose fixes name its input; where two of them
/// move it opposite ways, the sentence that says so ("Watch out"); and the foot (its Why
/// link's note, and with the spreadsheet details the cell of a row that is one input). `key`
/// is a Size column row's ([`size_row`]) or an input's path ([`input_row`]); nothing for
/// another.
fn input_ui(
```

Replace:

```rust
) -> Option<Asked> {
    let row = size_row(key)?;
    ui.horizontal_wrapped(|ui| {
```

with:

```rust
) -> Option<Asked> {
    let row = size_row(key).copied().or_else(|| input_row(key))?;
    ui.horizontal_wrapped(|ui| {
```

Replace:

```rust
    ui.horizontal_wrapped(|ui| {
        if let Some(value) = row_value(row, shown) {
            ui.strong(value);
```

with:

```rust
    ui.horizontal_wrapped(|ui| {
        if let Some(value) = row_value(&row, shown) {
            ui.strong(value);
```

Replace:

```rust
    });
    heading(ui, WHAT_IT_IS);
    ui.add(egui::Label::new(row_help(key)).wrap());
    let lines = sets_it_lines(row, shown);
    if !lines.is_empty() {
```

with:

```rust
    });
    let mut asked = None;
    if let Some(input) = row.input.filter(|input| registry().is_leaf_input(input))
        && ui.link(TRACE_LINK).clicked()
    {
        asked = Some(Asked::Trace(input));
    }
    let help = row_help(key);
    if !help.is_empty() {
        heading(ui, WHAT_IT_IS);
        ui.add(egui::Label::new(help).wrap());
    }
    let lines = sets_it_lines(&row, shown);
    if !lines.is_empty() {
```

Replace:

```rust
    }
    let mut asked = None;
    let touched: Vec<&CheckGuide> = row.input.map_or_else(Vec::new, |input| {
```

with:

```rust
    }
    let touched: Vec<&CheckGuide> = row.input.map_or_else(Vec::new, |input| {
```

Replace:

```rust
    let cell = (readouts.show_details() && row.input == Some(row.key)).then(|| {
        let cell = InputCatalogue::get()
            .entry(row.key)
            .and_then(|entry| entry.meta.cell)
            .unwrap_or(NO_SPREADSHEET_CELL);
        format!("{SPREADSHEET_CELL} {cell}")
    });
```

with:

```rust
    let cell = (readouts.show_details() && row.input == Some(row.key)).then(|| {
        cell_line(
            InputCatalogue::get()
                .entry(row.key)
                .and_then(|entry| entry.meta.cell),
        )
    });
```

Replace:

```rust

/// The row's value as its slider shows it, with its unit: the input's value, or for the
/// controls that are no one input the size the slider shows (`dimensions`); none where the
/// design has no size controls.
fn row_value(row: &SizeRow, shown: &Shown<'_>) -> Option<String> {
```

with:

```rust

/// The entry of a row of the detailed inputs or the assumptions, drawn as a Size column row of
/// its own: keyed by its input's path, its label the inputs side's ([`label_for`]), no
/// torque-card row and no lesson. `None` for a path that is no input.
fn input_row(path: &str) -> Option<SizeRow> {
    let path = InputCatalogue::get().entry(path)?.path.as_str();
    Some(SizeRow {
        key: path,
        label: label_for(path),
        input: Some(path),
        card: None,
        lesson: None,
    })
}

/// What an input's fix link selects (spec 5.2): the input, by its path, or for an input the
/// Size column shows under its own name ([`SIZE_TAB_ALIASES`]: the ring radius) that control's
/// row, whose slider the link shows.
fn fix_selection(path: &'static str) -> Selection {
    let key = SIZE_TAB_ALIASES
        .iter()
        .find(|&&(aliased, _)| aliased == path)
        .and_then(|&(_, label)| SIZE_ROWS.iter().find(|row| row.label == label))
        .map_or(path, |row| row.key);
    Selection::Input(key)
}

/// An entry's foot line with the spreadsheet details on: [`SPREADSHEET_CELL`] and `cell`, or
/// the line that says the spreadsheet has none.
fn cell_line(cell: Option<&str>) -> String {
    format!("{SPREADSHEET_CELL} {}", cell.unwrap_or(NO_SPREADSHEET_CELL))
}

/// The row's value as its slider shows it, with its unit: the input's value (a blank, a
/// selector's choice or a text in words: [`value_text`]), or for the controls that are no one
/// input the size the slider shows (`dimensions`); none where the design has no size controls.
fn row_value(row: &SizeRow, shown: &Shown<'_>) -> Option<String> {
```

Replace:

```rust
            let value = shown.inputs.get(path)?;
            let unit = InputCatalogue::get().entry(path)?.meta.unit;
            Some(value_with_unit(&value, format_value(&value), unit))
        }
```

with:

```rust
            let value = shown.inputs.get(path)?;
            let meta = InputCatalogue::get().entry(path)?.meta;
            let text = value_text(&value, meta);
            Some(match value {
                Value::Num(_) | Value::Int(_) if meta.choices.is_empty() => {
                    value_with_unit(&value, text, meta.unit)
                }
                _ => text,
            })
        }
```

Replace:

```rust
/// A check's entry: its state and name; why it matters; then, where it does not pass, what is
/// wrong in this design (in the level's colour), the fixes, each a link that shows its input's
/// row, or the line saying no listed change helps, and the notes on those fixes; the numbers
/// the check reads, each a readout; and the foot links.
fn entry_ui(
```

with:

```rust
/// A check's entry: its state and name; why it matters; then, where it does not pass, what is
/// wrong in this design (in the level's colour), the fixes, each a link that selects its input
/// and shows its row ([`fix_selection`]), or the line saying no listed change helps, and the
/// notes on those fixes; the numbers the check reads, each a readout; and the foot links, with
/// the spreadsheet details the cell of a check that is a result.
fn entry_ui(
```

Replace:

```rust
    let guide = entry.guide;
    ui.horizontal_wrapped(|ui| {
```

with:

```rust
    let guide = entry.guide;
    let mut asked = None;
    ui.horizontal_wrapped(|ui| {
```

Replace:

```rust
                readouts.focus_input(path);
            }
```

with:

```rust
                readouts.focus_input(path);
                asked = Some(Asked::Select(fix_selection(path)));
            }
```

Replace:

```rust
    numbers_ui(ui, guide, entry.level, shown, readouts);
    let equation = result_info(guide.check).map(|_| guide.check);
    foot_ui(ui, Some(guide.learn), equation, None, readouts)
}
```

with:

```rust
    numbers_ui(ui, guide, entry.level, shown, readouts);
    let info = result_info(guide.check);
    let equation = info.map(|_| guide.check);
    let cell = info
        .filter(|_| readouts.show_details())
        .map(|info| cell_line(info.cell.as_deref()));
    asked.or(foot_ui(ui, Some(guide.learn), equation, cell, readouts))
}
```

Replace:

```rust
/// torque card, [`card_ui`]); the check that badges its row, a link that selects it; and the
/// foot links. Nothing for a path that is no tile's.
fn result_ui(
```

with:

```rust
/// torque card, [`card_ui`]); the check that badges its row, a link that selects it; and the
/// foot links, with the spreadsheet details its cell. Nothing for a path that is no tile's.
fn result_ui(
```

Replace:

```rust
    let learn = row_learn(path).unwrap_or(Learn::None(""));
    asked.or(foot_ui(ui, Some(learn), Some(path), None, readouts))
}
```

with:

```rust
    let learn = row_learn(path).unwrap_or(Learn::None(""));
    let cell = readouts
        .show_details()
        .then(|| cell_line(result_info(path).and_then(|info| info.cell.as_deref())));
    asked.or(foot_ui(ui, Some(learn), Some(path), cell, readouts))
}
```

**`magcoupling-rs/src/gui/inputs.rs`**

Replace:

```rust
    }
    let default = match (default, meta.ty) {
        (Value::None, FieldType::OptF64) => "blank".to_owned(),
```

with:

```rust
    }
    lines.push(format!("Default: {}", value_text(default, meta)));
    if meta.assumption {
        lines.push(ASSUMPTION_LINE.to_owned());
    }
    lines.join("\n")
}

/// An input's value as the tooltip's Default line and an input's Explain entry write it:
/// "blank" for a blank optional or text input, a selector's code with its choice ("1 = ..."),
/// else the value as the panel formats it, without its unit.
pub fn value_text(value: &Value, meta: &InputMeta) -> String {
    match (value, meta.ty) {
        (Value::None, FieldType::OptF64) => "blank".to_owned(),
```

Replace:

```rust
        (value, _) => crate::gui::format::format_value(value),
    };
    lines.push(format!("Default: {default}"));
    if meta.assumption {
        lines.push(ASSUMPTION_LINE.to_owned());
    }
```

with:

```rust
        (value, _) => crate::gui::format::format_value(value),
    }
```

Replace:

```rust
        (value, _) => crate::gui::format::format_value(value),
    }
    lines.join("\n")
}

```

with:

```rust
        (value, _) => crate::gui::format::format_value(value),
    }
}

```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
            }
            None => {}
```

with:

```rust
            }
            Some(Asked::Trace(path)) => self.toggle_trace(path),
            None => {}
```

Replace:

```rust
        if output.label_clicked {
            // A second click on the source's label ends the trace.
            let same = self.trace.as_ref().is_some_and(|t| t.source == entry.path);
            self.trace = if same { None } else { Trace::of(&entry.path) };
            ui.ctx().request_repaint();
```

with:

```rust
        if output.label_clicked {
            self.toggle_trace(&entry.path);
            ui.ctx().request_repaint();
```

Replace:

```rust
        output.widget.id
    }
```

with:

```rust
        output.widget.id
    }

    /// Traces the input at `path`, or ends the trace if `path` is its source: a second click
    /// ends it (decision O-8).
    fn toggle_trace(&mut self, path: &str) {
        let same = self.trace.as_ref().is_some_and(|t| t.source == path);
        self.trace = if same { None } else { Trace::of(path) };
    }
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1101 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** the three files: `feat(magcoupler): any input's Explain entry, led by the link that traces it; a fix link selects its input; a tile's and a check's cell foot (BL-092 T1)`.

### Task 2: Touching a row selects its input; the trace goes through the entry's link

**Model:** `sonnet`.

A click on a Detailed inputs or Assumptions row's label selects its input instead of tracing it, and so does any other touch of the row (an edit, a reset, `size_controls::touched` on its widget); the label's hover loses the trace line. The ten panel tests that traced by a label click trace through the entry's link (`Harness::trace_input`); two new tests hold the selection, the hover, the face gap's one entry and the touches.

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/input_ui.rs`, `magcoupling-rs/src/gui/trace.rs` (two comments)

**Interfaces:**
- Removes: `input_ui::TRACE_HINT`.
- Changes: `RowOutput::label_clicked`, an edit or a touch of `RowOutput::widget` selects the input (`MagcouplingPanel::input_row_ui`).

- [ ] **Step 1 (red): the label's tests.** The blocks add the harness helper `trace_input`, `a_click_on_an_input_row_s_label_selects_its_input_and_a_hover_selects_nothing` and `touching_a_detailed_row_s_widget_selects_its_input`, rewrite the trace tests to trace through the link, expect the hover without `TRACE_HINT`, expect an edited input selected in the Esc and view-choice tests, add the fix link's way back to the check and idle with a Detailed input's entry.

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust

        /// Opens the top bar's Design menu and clicks its item `item` (UX overhaul spec 4.2).
```

with:

```rust

        /// Traces the input labelled `label` as a person does (UX overhaul spec 5.2): a click on
        /// its label selects it, then a click on its Explain entry's trace link.
        fn trace_input(&mut self, label: &str) -> egui::FullOutput {
            self.click_text(label);
            self.click_text(crate::gui::explain::TRACE_LINK)
        }

        /// Opens the top bar's Design menu and clicks its item `item` (UX overhaul spec 4.2).
```

Replace:

```rust
        harness.frame(vec![egui::Event::Text("X".to_owned())]);
        harness.frame(key_tap(egui::Key::Escape));
```

with:

```rust
        harness.frame(vec![egui::Event::Text("X".to_owned())]);
        // Spec 5.2: typing in the field touches its input, which it selects.
        assert_eq!(harness.panel.selection, Selection::Input(PART_INNER));
        harness.frame(key_tap(egui::Key::Escape));
```

Replace:

```rust
        harness.frame(key_tap(egui::Key::Escape));
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
        // The field let go of the keyboard: the next Esc is the panel's.
```

with:

```rust
        harness.frame(key_tap(egui::Key::Escape));
        assert_eq!(harness.panel.selection, Selection::Input(PART_INNER));
        // The field let go of the keyboard: the next Esc is the panel's.
```

Replace:

```rust
        harness.frame(Vec::new());
        harness.panel.undo();
```

with:

```rust
        harness.frame(Vec::new());
        // Spec 5.2: the nudge touches the gap, which it selects; the undo, reset all and the
        // load keep that selection.
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        harness.panel.undo();
```

Replace:

```rust
        harness.panel.undo();
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
        harness.panel.reset();
```

with:

```rust
        harness.panel.undo();
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        harness.panel.reset();
```

Replace:

```rust
        harness.panel.load_design_file(&file).unwrap();
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
    }
```

with:

```rust
        harness.panel.load_design_file(&file).unwrap();
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
    }
```

Replace:

```rust
        // the Hot-low tile's (the torque card: its consequences are 30 compute_all runs), with
        // a note's entry (its diagram) and with the note list (plan review COV-1), a frame in
        // which nothing changed runs no fix search and no results of its own.
        use crate::engine::explain::notes::START_HERE;
```

with:

```rust
        // the Hot-low tile's (the torque card: its consequences are 30 compute_all runs), with
        // a note's entry (its diagram), with the note list (plan review COV-1) and with a
        // Detailed input's entry (decision D-59), a frame in which nothing changed runs no fix
        // search and no results of its own.
        use crate::engine::explain::notes::START_HERE;
```

Replace:

```rust
            Some(Selection::NoteList),
        ] {
```

with:

```rust
            Some(Selection::NoteList),
            Some(Selection::Input("coupling.op_temp_C")),
        ] {
```

Replace:

```rust
    fn an_input_row_s_hover_names_its_path_and_cell_only_with_the_spreadsheet_details() {
        use crate::gui::input_ui::TRACE_HINT;
        use crate::gui::inputs::input_tooltip;
```

with:

```rust
    fn an_input_row_s_hover_names_its_path_and_cell_only_with_the_spreadsheet_details() {
        use crate::gui::inputs::input_tooltip;
```

Replace:

```rust
            let output = harness.frame(Vec::new());
            let want = format!(
                "{}\n{TRACE_HINT}",
                input_tooltip(entry, &entry.default, show_details)
            );
            assert!(drawn_texts(&output).contains(&want), "{want:?}");
```

with:

```rust
            let output = harness.frame(Vec::new());
            let want = input_tooltip(entry, &entry.default, show_details);
            assert!(drawn_texts(&output).contains(&want), "{want:?}");
```

Replace:

```rust
        // UX overhaul spec 7.1 (C-22): on screen "the default" is the baseline's value.
        use crate::gui::input_ui::TRACE_HINT;
        use crate::gui::inputs::input_tooltip;
```

with:

```rust
        // UX overhaul spec 7.1 (C-22): on screen "the default" is the baseline's value.
        use crate::gui::inputs::input_tooltip;
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        let want = format!("{tooltip}\n{TRACE_HINT}");
        assert!(drawn_texts(&output).contains(&want), "{want:?}");
        // One step off the baseline: the dot and the reset, whose hover names the baseline's
```

with:

```rust
        let output = harness.frame(Vec::new());
        assert!(drawn_texts(&output).contains(&tooltip), "{tooltip:?}");
        // One step off the baseline: the dot and the reset, whose hover names the baseline's
```

Replace:

```rust
        assert_eq!(text_color(&output, CHANGED_DOT), Some(blue));
        harness.click_text(label_for(FACE_GAP));
        let banner = harness.panel.trace.clone().expect("a trace").banner(false);
```

with:

```rust
        assert_eq!(text_color(&output, CHANGED_DOT), Some(blue));
        harness.trace_input(label_for(FACE_GAP));
        let banner = harness.panel.trace.clone().expect("a trace").banner(false);
```

Replace:

```rust
    #[test]
    fn clicking_an_input_s_label_frames_the_results_it_drives_until_a_second_click() {
        use crate::gui::guidance::counted_guides;
```

with:

```rust
    #[test]
    fn tracing_an_input_frames_the_results_it_drives_until_a_second_click() {
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::guidance::counted_guides;
```

Replace:

```rust
        use crate::gui::trace::TraceKind;
        let mut harness = Harness::new();
```

with:

```rust
        use crate::gui::trace::TraceKind;
        // Spec 5.2: a click on the label selects the input and traces nothing; its entry's link
        // traces it.
        let mut harness = Harness::new();
```

Replace:

```rust
        harness.click_text(gap);
        let trace = harness.panel.trace.clone().expect("a trace");
```

with:

```rust
        harness.click_text(gap);
        assert_eq!(harness.panel.trace, None);
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        harness.click_text(TRACE_LINK);
        let trace = harness.panel.trace.clone().expect("a trace");
```

Replace:

```rust
        assert!(on_inputs[0].contains_rect(row));
        // A second click on the label ends the trace and its frames.
        harness.click_text(gap);
        assert_eq!(harness.panel.trace, None);
```

with:

```rust
        assert!(on_inputs[0].contains_rect(row));
        // A second click on the link ends the trace and its frames.
        harness.click_text(TRACE_LINK);
        assert_eq!(harness.panel.trace, None);
```

Replace:

```rust
    #[test]
    fn an_input_s_entry_link_traces_it_until_a_second_click_and_edits_nothing() {
```

with:

```rust
    #[test]
    fn a_click_on_an_input_row_s_label_selects_its_input_and_a_hover_selects_nothing() {
        use crate::gui::explain::TRACE_LINK;
        // Spec 5.1 and 5.2: a click on the label of a row of the detailed inputs or the
        // assumptions selects its input, whose entry the Explain panel shows in the same frame;
        // a hover shows the row's tooltip and selects nothing (C-2). The face gap's Key design
        // row selects the entry the Size column's gap row selects (decision D-35). No click
        // traces, edits the design or makes an undo step.
        let op_temp = "coupling.op_temp_C";
        let label = label_for(op_temp);
        let mut harness = Harness::new();
        tooltips_at_once(&harness.ctx);
        let output = harness.frame(Vec::new());
        let at = text_rect(&output, label).expect("its row").center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Nothing, "a hover");
        let output = harness.click(at);
        assert_eq!(harness.panel.selection, Selection::Input(op_temp));
        let checks_left = right_column_left(&output);
        let in_explain = |text: &str| {
            text_rects(&output, text)
                .into_iter()
                .any(|r| r.left() > checks_left)
        };
        assert!(in_explain(label), "its entry, in the click's frame");
        assert!(in_explain(TRACE_LINK));
        harness.click_text(label_for(FACE_GAP));
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        // An assumption's row, in the Assumptions view.
        let assumption = ASSUMPTIONS[0].paths[0];
        harness.click_text(InputsView::Assumptions.label());
        harness.click_text(label_for(assumption));
        assert_eq!(harness.panel.selection, Selection::Input(assumption));
        // On the opening tab, in the Detailed inputs fold (a tall window: its row in view).
        let mut first = Harness::of(MagcouplingPanel::new(), TALL);
        first.click_text(DETAILED_INPUTS);
        first.click_text(label);
        assert_eq!(first.panel.selection, Selection::Input(op_temp));
        for harness in [&harness, &first] {
            assert_eq!(harness.panel.trace, None);
            assert_eq!(
                harness.panel.design(),
                harness.panel.settled_design().clone()
            );
            assert_eq!(harness.panel.history.undo_len(), 0);
        }
    }

    #[test]
    fn touching_a_detailed_row_s_widget_selects_its_input() {
        // Spec 5.2: touching an input selects it, as part 2b-i's touches do for the Size
        // column (size_controls::touched): the keyboard's focus arriving on a Key design row's
        // slider (a focus that edits nothing), a drag of it, and a click on its reset. A hover
        // over the slider selects nothing (C-2).
        let op_temp = "coupling.op_temp_C";
        let before = KEY_DESIGN[KEY_DESIGN.iter().position(|p| *p == op_temp).unwrap() - 1];
        let mut harness = Harness::new();
        let target = harness.widget(op_temp).id;
        let at = harness.widget(op_temp).rect.left_center() + egui::vec2(20.0, 0.0);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Nothing, "a hover");
        // Tab from the row before (a focus requested between frames is not one that arrives).
        harness.focus(before);
        harness.panel.selection = Selection::Nothing;
        for _ in 0..6 {
            harness.frame(key_tap(egui::Key::Tab));
            if harness.ctx.memory(|m| m.has_focus(target)) {
                break;
            }
        }
        assert!(
            harness.ctx.memory(|m| m.has_focus(target)),
            "Tab reaches it"
        );
        harness.frame(Vec::new());
        assert_eq!(
            harness.panel.selection,
            Selection::Input(op_temp),
            "the focus"
        );
        assert_eq!(
            harness.panel.history.undo_len(),
            0,
            "the focus edits nothing"
        );
        harness.ctx.memory_mut(|m| m.surrender_focus(target));
        harness.panel.selection = Selection::Nothing;
        harness.frame(Vec::new());
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(vec![primary_button(at, true)]);
        harness.frame(vec![egui::Event::PointerMoved(at + egui::vec2(12.0, 0.0))]);
        harness.frame(vec![primary_button(at + egui::vec2(12.0, 0.0), false)]);
        assert_eq!(harness.panel.selection, Selection::Input(op_temp), "a drag");
        assert_ne!(harness.panel.design(), Design::default(), "the drag edits");
        harness.panel.selection = Selection::Nothing;
        harness.click_text(RESET_LABEL);
        assert_eq!(
            harness.panel.selection,
            Selection::Input(op_temp),
            "a reset"
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn an_input_s_entry_link_traces_it_until_a_second_click_and_edits_nothing() {
```

Replace:

```rust
        // Trace the face gap and show only the results it drives.
        harness.click_text(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
```

with:

```rust
        // Trace the face gap and show only the results it drives.
        harness.trace_input(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
```

Replace:

```rust
        let drive = "coupling.drive_torque_Nm";
        harness.click_text(InputCatalogue::get().entry(drive).unwrap().meta.label);
        let trace = harness.panel.trace.clone().expect("a trace");
```

with:

```rust
        let drive = "coupling.drive_torque_Nm";
        harness.trace_input(InputCatalogue::get().entry(drive).unwrap().meta.label);
        let trace = harness.panel.trace.clone().expect("a trace");
```

Replace:

```rust
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        harness.click_text(label_for(FACE_GAP));
        let output = harness.frame(Vec::new());
```

with:

```rust
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        harness.trace_input(label_for(FACE_GAP));
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        // "Traced only" goes off with the trace even when the table is not on screen: Clear
        // trace and a click on another input's label, both with the Geometry view showing, must
        // not leave the box ticked for the new trace (it would filter the table unasked).
        let entries = table_entries();
```

with:

```rust
        // "Traced only" goes off with the trace even when the table is not on screen: Clear
        // trace and a trace of another input, both with the Geometry view showing, must not
        // leave the box ticked for the new trace (it would filter the table unasked).
        let entries = table_entries();
```

Replace:

```rust
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        harness.click_text(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
        let gap = harness.panel.trace.clone().expect("a trace");
```

with:

```rust
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        harness.trace_input(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
        let gap = harness.panel.trace.clone().expect("a trace");
```

Replace:

```rust
        assert_eq!(harness.panel.trace, None);
        harness.click_text(label_for(POLES));
        let poles = harness.panel.trace.clone().expect("a trace");
```

with:

```rust
        assert_eq!(harness.panel.trace, None);
        harness.trace_input(label_for(POLES));
        let poles = harness.panel.trace.clone().expect("a trace");
```

Replace:

```rust
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(CentreView::Results.label());
        harness.click_text(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
        let gap = harness.panel.trace.clone().expect("a trace");
```

with:

```rust
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(CentreView::Results.label());
        harness.trace_input(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
        let gap = harness.panel.trace.clone().expect("a trace");
```

Replace:

```rust
        );
        harness.click_text(label_for(POLES));
        let poles = harness.panel.trace.clone().expect("a trace");
```

with:

```rust
        );
        harness.trace_input(label_for(POLES));
        let poles = harness.panel.trace.clone().expect("a trace");
```

Replace:

```rust
        harness.click_text(CentreView::Results.label());
        harness.click_text(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
```

with:

```rust
        harness.click_text(CentreView::Results.label());
        harness.trace_input(label_for(FACE_GAP));
        harness.click_text(TRACED_ONLY);
```

Replace:

```rust
        harness.click_text(CentreView::Results.label());
        harness.click_text(label_for(FACE_GAP));
        let trace = harness.panel.trace.clone().expect("a trace");
```

with:

```rust
        harness.click_text(CentreView::Results.label());
        harness.trace_input(label_for(FACE_GAP));
        let trace = harness.panel.trace.clone().expect("a trace");
```

Replace:

```rust
    #[test]
    fn a_trace_moves_no_input_row_so_a_second_click_on_the_label_ends_it() {
        use crate::gui::trace::CLEAR_TRACE;
```

with:

```rust
    #[test]
    fn a_trace_moves_no_input_row_and_a_second_click_on_the_label_traces_nothing() {
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::trace::CLEAR_TRACE;
```

Replace:

```rust
        use crate::gui::trace::CLEAR_TRACE;
        // The banner sits at the foot of the inputs side: a trace moves no row above it, so the
        // second click of "a second click on the label ends the trace" lands on the label again,
        // not on whatever a banner on top would have pushed under the pointer.
        let mut harness = Harness::new();
```

with:

```rust
        use crate::gui::trace::CLEAR_TRACE;
        // The banner sits at the foot of the inputs side: a trace moves no row above it (decision
        // O-8). The label selects its input (spec 5.2): a second click on it lands on it again
        // and keeps the trace, which the entry's link ends.
        let mut harness = Harness::new();
```

Replace:

```rust
        harness.click(label.center());
        let trace = harness.panel.trace.clone().expect("a trace");
```

with:

```rust
        harness.click(label.center());
        assert_eq!(harness.panel.trace, None, "the label traces nothing");
        harness.click_text(TRACE_LINK);
        let trace = harness.panel.trace.clone().expect("a trace");
```

Replace:

```rust
        assert!(on_inputs_side(&output, CLEAR_TRACE).top() > label.bottom());
        // The second click, at the same point.
        harness.click(label.center());
```

with:

```rust
        assert!(on_inputs_side(&output, CLEAR_TRACE).top() > label.bottom());
        // The second click, at the same point: the label again, the trace kept.
        harness.click(label.center());
```

Replace:

```rust
        harness.click(label.center());
        assert_eq!(harness.panel.trace, None);
```

with:

```rust
        harness.click(label.center());
        assert_eq!(harness.panel.trace.as_ref(), Some(&trace));
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        harness.click_text(TRACE_LINK);
        assert_eq!(harness.panel.trace, None);
```

Replace:

```rust
    #[test]
    fn the_locked_free_variable_s_label_still_traces_it() {
        use crate::gui::trace::TraceKind;
```

with:

```rust
    #[test]
    fn the_locked_free_variable_s_label_still_selects_it_and_its_entry_traces_it() {
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::trace::TraceKind;
```

Replace:

```rust
        // In Torque -> Magnets the free variable's row is locked (its value widgets disabled);
        // its label still traces the input, as every row's does.
        let mut harness = Harness::new();
```

with:

```rust
        // In Torque -> Magnets the free variable's row is locked (its value widgets disabled);
        // its label still selects the input and its entry traces it, as every row's does.
        let mut harness = Harness::new();
```

Replace:

```rust
        harness.click_text(label);
        let trace = harness.panel.trace.clone().expect("a trace");
```

with:

```rust
        harness.click_text(label);
        assert_eq!(harness.panel.selection, Selection::Input(AXIAL_LENGTH));
        harness.click_text(TRACE_LINK);
        let trace = harness.panel.trace.clone().expect("a trace");
```

Replace:

```rust
        );
        // A second click ends it; neither click edits the design or makes an undo step.
        harness.click_text(label);
        assert_eq!(harness.panel.trace, None);
```

with:

```rust
        );
        // A second click ends it; no click edits the design or makes an undo step.
        harness.click_text(TRACE_LINK);
        assert_eq!(harness.panel.trace, None);
```

Replace:

```rust
        texts.push(crate::gui::results_table::TRACED_ONLY.to_owned());
        texts.push(crate::gui::input_ui::TRACE_HINT.to_owned());
        texts.push(crate::gui::trace::CLEAR_TRACE.to_owned());
```

with:

```rust
        texts.push(crate::gui::results_table::TRACED_ONLY.to_owned());
        texts.push(crate::gui::explain::TRACE_LINK.to_owned());
        texts.push(crate::gui::trace::CLEAR_TRACE.to_owned());
```

Replace:

```rust
        use crate::gui::dashboard::END_EFFECT_GREYED;
        use crate::gui::explorer::{NO_EQUATION, term_tag};
```

with:

```rust
        use crate::gui::dashboard::END_EFFECT_GREYED;
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::explorer::{NO_EQUATION, term_tag};
```

Replace:

```rust
        use crate::gui::explorer::{NO_EQUATION, term_tag};
        use crate::gui::input_ui::{CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, TRACE_HINT};
        use crate::gui::inputs::ASSUMPTION_LINE;
```

with:

```rust
        use crate::gui::explorer::{NO_EQUATION, term_tag};
        use crate::gui::input_ui::{CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT};
        use crate::gui::inputs::ASSUMPTION_LINE;
```

Replace:

```rust
            NO_EQUATION,
            TRACE_HINT,
            OUTSIDE_RANGE_HINT,
```

with:

```rust
            NO_EQUATION,
            TRACE_LINK,
            OUTSIDE_RANGE_HINT,
```

Replace:

```rust
    fn a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing() {
        use crate::gui::guidance::{CHECK_GUIDES, all_fixes, fix_text};
        // Every fix of the workbook example's red checks, each in its check's Explain entry,
```

with:

```rust
    fn a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing() {
        use crate::gui::explain::CHECKS_IT_TOUCHES;
        use crate::gui::guidance::{CHECK_GUIDES, all_fixes, fix_text, guide};
        // Every fix of the workbook example's red checks, each in its check's Explain entry,
```

Replace:

```rust
                    "{view:?} {path}: {rows:?} {focus:?}"
                );
            }
        }
```

with:

```rust
                    "{view:?} {path}: {rows:?} {focus:?}"
                );
                // Decision D-54: the input's entry links back to the check under "Checks it
                // touches"; a click on it selects the check again.
                let heading = text_rect(&output, CHECKS_IT_TOUCHES)
                    .unwrap_or_else(|| panic!("{view:?} {path}: no {CHECKS_IT_TOUCHES:?}"));
                let back = text_rects(&output, guide(check).expect("a guide").label)
                    .into_iter()
                    .find(|r| r.top() >= heading.bottom())
                    .unwrap_or_else(|| panic!("{view:?} {path}: no link back to {check}"));
                harness.click(back.center());
                assert_eq!(
                    harness.panel.selection,
                    Selection::Check(check),
                    "{view:?} {path}"
                );
            }
        }
```

**`magcoupling-rs/src/gui/trace.rs`**

Replace:

```rust
        assert_eq!((explained, table_entries().len()), (421, 1120));
        // Every input is a leaf of the registry: a click on any input's label traces it.
        for entry in crate::gui::inputs::InputCatalogue::get().all() {
```

with:

```rust
        assert_eq!((explained, table_entries().len()), (421, 1120));
        // Every input is a leaf of the registry: any input's entry links to its trace.
        for entry in crate::gui::inputs::InputCatalogue::get().all() {
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 15
  ```

  Expected: 16 failed: `a_click_on_an_input_row_s_label_selects_its_input_and_a_hover_selects_nothing`, `a_result_traced_off_the_table_also_turns_the_trace_filter_off`, `a_search_that_hides_the_traced_rows_says_so_and_the_headings_count_what_it_shows`, `a_trace_moves_no_input_row_and_a_second_click_on_the_label_traces_nothing`, `a_trace_filter_left_ticked_off_the_table_does_not_filter_a_later_trace`, `an_input_row_s_hover_names_its_path_and_cell_only_with_the_spreadsheet_details`, `an_input_whose_results_have_no_equation_record_traces_none_and_says_so`, `esc_in_a_text_field_is_the_field_s_and_keeps_the_selection`, `reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows`, `the_changed_dot_a_row_s_reset_and_the_default_line_follow_the_baseline`, `the_locked_free_variable_s_label_still_selects_it_and_its_entry_traces_it`, `the_panel_s_marks_are_the_host_s_link_colour`, `the_selection_is_a_view_choice_in_no_file_link_or_undo_step`, `tracing_an_input_frames_the_results_it_drives_until_a_second_click`, `touching_a_detailed_row_s_widget_selects_its_input`, `the_traced_filter_shows_the_results_the_trace_marks`.

- [ ] **Step 3 (green): a touch selects.**

**`magcoupling-rs/src/gui/input_ui.rs`**

Replace:

```rust

/// The line the label's hover text ends with: a click on the label traces the input
/// (decision O-8).
pub const TRACE_HINT: &str =
    "Click the label to highlight the results it drives that have an equation";

/// The hover text of the changed-from-default dot, on an input row and an assumption alike.
```

with:

```rust

/// The hover text of the changed-from-default dot, on an input row and an assumption alike.
```

Replace:

```rust
    pub widget: egui::Response,
    /// The label was clicked: trace the input (decision O-8).
    pub label_clicked: bool,
```

with:

```rust
    pub widget: egui::Response,
    /// The label was clicked: select the input, whose Explain entry traces it (UX overhaul spec
    /// 5.2).
    pub label_clicked: bool,
```

Replace:

```rust
/// user enters a value (`inputs::OPTIONAL_SEEDS`). A `locked` row (Torque -> Magnets' free
/// variable) disables everything but its dot and label: the label still traces the input.
/// `show_details` (the header's Show spreadsheet details) adds the path, the cell and the
```

with:

```rust
/// user enters a value (`inputs::OPTIONAL_SEEDS`). A `locked` row (Torque -> Magnets' free
/// variable) disables everything but its dot and label: the label still selects the input.
/// `show_details` (the header's Show spreadsheet details) adds the path, the cell and the
```

Replace:

```rust
                .on_hover_ui(|ui| {
                    ui.add(egui::Label::new(format!("{}\n{TRACE_HINT}", tooltip())));
                })
```

with:

```rust
                .on_hover_ui(|ui| {
                    ui.add(egui::Label::new(tooltip()));
                })
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
        }
        // After every change of the trace this frame (Clear trace, a label click, a result
        // click), whichever view the centre region shows: the results table's trace filter goes
        // off with an input's trace even while the table is not drawn.
```

with:

```rust
        }
        // After every change of the trace this frame (Clear trace, an entry's trace link, a
        // result click), whichever view the centre region shows: the results table's trace filter goes
        // off with an input's trace even while the table is not drawn.
```

Replace:

```rust
        // The trace's banner at the foot of the side, laid out bottom up first so the view takes
        // the room above it in the same frame: a trace moves no row above it, so a second click
        // on the label lands on the label again (decision O-8).
        let clamps_paused = self.clamps_paused;
```

with:

```rust
        // The trace's banner at the foot of the side, laid out bottom up first so the view takes
        // the room above it in the same frame: a trace moves no row above it (decision O-8).
        let clamps_paused = self.clamps_paused;
```

Replace:

```rust
    /// ring size the cap OD and thread that follow it) show the value the panel shows, locked
    /// (the label still traces the input). The row is framed in its term's colour while the equation in view
    /// reads the input (`readouts`).
```

with:

```rust
    /// ring size the cap OD and thread that follow it) show the value the panel shows, locked
    /// (the label still selects the input). A click on a row's label selects its input (spec
    /// 5.2). The row is framed in its term's colour while the equation in view
    /// reads the input (`readouts`).
```

Replace:

```rust
        let output = row.inner;
        if output.label_clicked {
            self.toggle_trace(&entry.path);
            ui.ctx().request_repaint();
```

with:

```rust
        let output = row.inner;
        // Spec 5.2: touching an input (its label's click, its widget's click, drag or focus, an
        // edit or a reset) selects it, as a Size control's touch does; its entry holds the
        // trace's link. A locked row's widget is disabled: only its label touches it.
        if output.label_clicked || output.edit.is_some() || size_controls::touched(&output.widget) {
            self.selection = Selection::Input(entry.path.as_str());
            ui.ctx().request_repaint();
```

**`magcoupling-rs/src/gui/trace.rs`**

Replace:

```rust
//! Tracing between the inputs and the results (decision O-8): a click on an input's label
//! traces the results its value flows into (the registry's `downstream`), a click on a result
//! (any readout: a dashboard row, a table row, a callout) traces the inputs it reads (its
```

with:

```rust
//! Tracing between the inputs and the results (decision O-8): the link in an input's Explain
//! entry (UX overhaul spec 5.2; a click on its label selects it) traces the results its value
//! flows into (the registry's `downstream`), a click on a result
//! (any readout: a dashboard row, a table row, a callout) traces the inputs it reads (its
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1103 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** the three files: `feat(magcoupler): touching an input row selects its input; the trace is its entry's link (BL-092 T2)`.

### Task 3: gui-smoke reads an input's entry

**Model:** `sonnet`.

After the tooltip step the smoke prompt clicks the same label and judges `input_entry` in the Explain panel. The red step is the prompt's guard in `app.rs` and the verdict test's list of checks; the green step is the prompt.

**Files:**
- Modify: `magcoupling-rs/src/app.rs`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`, `.claude/workflows/gui-smoke.js`

- [ ] **Step 1 (red): the guard and the verdict test.**

**`linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`**

Replace:

```js
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'check_entry',
  'lesson_link_opened', 'tooltip_plain', 'torque_card', 'address_follows']
// A calculator agent's reading of a good run; each case changes one thing.
```

with:

```js
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'check_entry',
  'lesson_link_opened', 'tooltip_plain', 'input_entry', 'torque_card', 'address_follows']
// A calculator agent's reading of a good run; each case changes one thing.
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        assert!(
            between("tooltip_plain=true was judged", "page.goto(").contains(collect),
            "the first page's console messages are collected before the link is opened again"
```

with:

```rust
        assert!(
            between("input_entry=true was judged", "page.goto(").contains(collect),
            "the first page's console messages are collected before the link is opened again"
```

Replace:

```rust
            "the first page's console messages are collected before the link is opened again"
        );
        assert!(
            between("drag_moved=true", "then close the browser").contains(collect),
```

with:

```rust
            "the first page's console messages are collected before the link is opened again"
        );
        // UX overhaul slice B, stage 2, part 2b-ii (spec 5.2): after the tooltip a click on the
        // label it showed beside selects that input, and the step finds the input's entry in the
        // Explain panel by its trace link, typed once as TRACE_LINK.
        {
            use crate::gui::explain::TRACE_LINK;
            assert_eq!(quoted("TRACE_LINK"), TRACE_LINK, "the script's TRACE_LINK");
            assert_eq!(
                script.matches(TRACE_LINK).count(),
                1,
                "the smoke prompt types the trace link once, in TRACE_LINK"
            );
            assert!(
                between("tooltip_plain=true only if", "input_entry=true only if")
                    .contains("click that label"),
                "the input's entry is judged after the tooltip, from a click on its label"
            );
            assert!(
                between("input_entry=true only if", "page.goto(").contains("Explain panel"),
                "the input's entry is judged in the Explain panel"
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

  Expected: 1 failed: `the_smoke_test_share_link_opens_its_design`. The verdict test's last line `1 of 78 checks fail` (its case "input_entry false with the agent's passed true" passes the run, since the script does not know the check yet).

- [ ] **Step 3 (green): the prompt.**

**`.claude/workflows/gui-smoke.js`**

Replace:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, an input tooltip shows no spreadsheet details, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

with:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, an input tooltip shows no spreadsheet details and a click on its label shows the input\'s entry in the Explain panel, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

Replace:

```js
const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'check_entry', 'lesson_link_opened', 'tooltip_plain', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
```

with:

```js
const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'check_entry', 'lesson_link_opened', 'tooltip_plain', 'input_entry', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
```

Replace:

```js
    tooltip_plain: { type: 'boolean' },
    torque_card: { type: 'boolean' },
```

with:

```js
    tooltip_plain: { type: 'boolean' },
    input_entry: { type: 'boolean' },
    torque_card: { type: 'boolean' },
```

Replace:

```js

const PLAYWRIGHT_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close"'
```

with:

```js

// The link at the top of an input's Explain entry, which the input step finds the entry by (UX
// overhaul spec 5.2): typed once here. app.rs's smoke test holds it equal to
// gui::explain::TRACE_LINK.
const TRACE_LINK = 'Highlight the results this drives'

const PLAYWRIGHT_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close"'
```

Replace:

```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Design menu and
// its Load design item, a check's row, a lesson link, the Geometry tab, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

with:

```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Design menu and
// its Load design item, a check's row, a lesson link, the Geometry tab, an input's label, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

Replace:

```js
// and 5.3) a click on a check's row shows its entry in the Explain panel, which scrolls on its
// own, and the entry's lesson link shows a teaching note there (decision C-11); and an input
// row's tooltip shows no spreadsheet details while the toggle is off. Then (slice 3, spec 7.2 and 8.6) the link is opened again with a timer on every frame the
// app draws: on that fresh page a click on the Hot-low tile shows the torque card's heading,
```

with:

```js
// and 5.3) a click on a check's row shows its entry in the Explain panel, which scrolls on its
// own, and the entry's lesson link shows a teaching note there (decision C-11); an input
// row's tooltip shows no spreadsheet details while the toggle is off, and (part 2b-ii, spec 5.2)
// a click on that row's label shows the input's entry in the Explain panel. Then (slice 3, spec 7.2 and 8.6) the link is opened again with a timer on every frame the
// app draws: on that fresh page a click on the Hot-low tile shows the torque card's heading,
```

Replace:

```js
Then an input tooltip: click the "Geometry" tab in the row of tabs at the top of the middle region and wait 1 second. On the left the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
If a target of these last checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes; the check list's last row is "Magnets fit". tooltip_plain=true was judged last on this page. Collect the console messages at level "info" (the check's entry, the lesson link and the tooltip ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; and the tooltip step left the Geometry tab selected, where no tile is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
```

with:

```js
Then an input tooltip: click the "Geometry" tab in the row of tabs at the top of the middle region and wait 1 second. On the left the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
Then click that label (as above, at the X, Y the mouse is at), move the mouse off the label onto the drawing in the middle region, wait 1 second and take a screenshot. input_entry=true only if the Explain panel (the right side, under the check list) now shows the input's entry: at its top the input's value and the label you clicked, under them the link "${TRACE_LINK}", then the heading "What it is" with a sentence about the input.
If a target of these last checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes; the check list's last row is "Magnets fit". input_entry=true was judged last on this page. Collect the console messages at level "info" (the check's entry, the lesson link, the tooltip and the input's entry ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; and the tooltip step left the Geometry tab selected, where no tile is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
```

Replace:

```js
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, check_entry, lesson_link_opened, tooltip_plain, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
```

with:

```js
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, check_entry, lesson_link_opened, tooltip_plain, input_entry, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
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

  Expected: `test result: ok. 1103 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both `grep -vc` lines print `0` (no test binary with a failure); both wasm32 clippy runs clean; `cargo fmt --check` prints nothing; the verdict test's last line `all 79 checks hold`.

- [ ] **Step 5: Commit** the three files: `test(magcoupler): gui-smoke clicks an input's label and reads its entry in the Explain panel (BL-092 T3)`.

### Task 4: Docs sweep

**Model:** `sonnet`.

The README's Explain panel, inputs side and smoke passages; `docs/ai/` (02 the system, 03 the modules, 04 the issue, 05 the tracker, the backlog item in progress); the register's rows A-29 to A-32 and its log; owner questions 53 to 55; the design's status line.

**Files:**
- Modify: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/analyses/2026-11-18-owner-questions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

- [ ] **Step 1: The docs.**

**`docs/ai/02-system.yaml`**

Replace:

```yaml
      - "Since BL-090 (UX overhaul slice B, stage 2, part 2b-i; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md): the Size column draws each control's label and slider and nothing about them, then the folds Detailed inputs and Assumptions (closed; an assumption's row asked for opens the second). Touching a control (size_controls::touched: a click, a drag, an edit, the focus arriving; never a hover) or clicking its label sets explain::Selection::Input(key), key from size_controls::SIZE_ROWS; the entry (explain::input_ui) draws the lines the column drew (the hint, the ConsequenceCache's what-moves-what lines, the floor line, LINKED_HOUSING in the Outside diameter and Overall length entries unless Auto sets the magnet count), the control's torque-card row (TorqueCard::control_lines), the checks whose fixes name its input, guidance::input_trade_offs, its Why note and, with the details, its cell. Only the Explain panel asks the ConsequenceCache now, for the Hot-low tile and a Size row; the Size column asks nothing (size_tab_consequences is gone)."
      - "Since BL-085 (UX overhaul slice B, stage 2, part 2a; plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md): the top bar is Undo, Redo, Design (a menu: the reset, Open the example design, Save design, Load design, Show spreadsheet details), Share, Export and Learn, each a word with no box at rest (panel.rs bar_button, design_menu_ui, reset_label: Reset to the starting design where the baseline is not the spreadsheet's design). The Explain panel draws the teaching notes (decision C-11): explain::Selection gains Note and NoteList; every note link (ReadoutEvents::note, the Size column's Why links, explorer_ui's return: the lesson line under an open equation, a Start here step) sets the selection; the Equation panel has no Explain checkbox, no note view and no button on the bar. format::value_with_unit: a text value takes no unit (design 9.2)."
```

with:

```yaml
      - "Since BL-090 (UX overhaul slice B, stage 2, part 2b-i; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md): the Size column draws each control's label and slider and nothing about them, then the folds Detailed inputs and Assumptions (closed; an assumption's row asked for opens the second). Touching a control (size_controls::touched: a click, a drag, an edit, the focus arriving; never a hover) or clicking its label sets explain::Selection::Input(key), key from size_controls::SIZE_ROWS; the entry (explain::input_ui) draws the lines the column drew (the hint, the ConsequenceCache's what-moves-what lines, the floor line, LINKED_HOUSING in the Outside diameter and Overall length entries unless Auto sets the magnet count), the control's torque-card row (TorqueCard::control_lines), the checks whose fixes name its input, guidance::input_trade_offs, its Why note and, with the details, its cell. Only the Explain panel asks the ConsequenceCache now, for the Hot-low tile and a Size row; the Size column asks nothing (size_tab_consequences is gone)."
      - "Since BL-092 (UX overhaul slice B, stage 2, part 2b-ii-a; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md): every input is a selection. Touching a Detailed inputs or Assumptions row (input_ui::RowOutput::label_clicked, an edit or a reset, or size_controls::touched on its widget, as the Size column's touches) sets explain::Selection::Input(path); explain::input_ui draws any input (input_row: its value through inputs::value_text, its help, its hint, the checks whose fixes name it, input_trade_offs, its cell) and leads every input's entry with TRACE_LINK, whose click (Asked::Trace) toggles the trace (MagcouplingPanel::toggle_trace); the label no longer traces. A check's fix link selects its input (explain::fix_selection; the ring radius selects Outside diameter's row) and still focuses its row. With the details on, a tile's and a check's entry end with their cell (cell_line). Selection::reads_consequences holds for a Size column row only, so a Detailed input's entry asks the ConsequenceCache nothing."
      - "Since BL-085 (UX overhaul slice B, stage 2, part 2a; plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md): the top bar is Undo, Redo, Design (a menu: the reset, Open the example design, Save design, Load design, Show spreadsheet details), Share, Export and Learn, each a word with no box at rest (panel.rs bar_button, design_menu_ui, reset_label: Reset to the starting design where the baseline is not the spreadsheet's design). The Explain panel draws the teaching notes (decision C-11): explain::Selection gains Note and NoteList; every note link (ReadoutEvents::note, the Size column's Why links, explorer_ui's return: the lesson line under an open equation, a Start here step) sets the selection; the Equation panel has no Explain checkbox, no note view and no button on the bar. format::value_with_unit: a text value takes no unit (design 9.2)."
```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel: new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption)), input_ui.rs (input_row (a locked row keeps its label live, to trace) -> RowOutput {edit, widget, label_clicked}, slider, RowEdit, TRACE_HINT, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the inputs side; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

with:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel: new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider, RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the inputs side; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-i (BL-090, plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md, decisions D-34 to D-48): the Size column's lines in the Explain panel, its six rows as selections, the Assumptions fold. Part 2b is cut in three (D-34, register row A-25); 2b-ii is the other selections, the callout list's move and the word budget of 250; 2b-iii the Plots tab, the outlines, the face, the page shell and BL-086, BL-088, BL-089. 'Set by Auto' stays under the locked slider (D-37, A-26, owner question 51); 'Checks it touches' reads the fixes on the design shown (D-39, A-27). risk: mechanical."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
```

with:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-i (BL-090, plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md, decisions D-34 to D-48): the Size column's lines in the Explain panel, its six rows as selections, the Assumptions fold. Part 2b is cut in three (D-34, register row A-25); 2b-ii is the other selections, the callout list's move and the word budget of 250; 2b-iii the Plots tab, the outlines, the face, the page shell and BL-086, BL-088, BL-089. 'Set by Auto' stays under the locked slider (D-37, A-26, owner question 51); 'Checks it touches' reads the fixes on the design shown (D-39, A-27). risk: mechanical."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-a (BL-092, plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md, decisions D-49 to D-59): every input a selection, the trace as a link at the top of its entry, a fix link selects its input, the cell foot of a tile's and a check's entry, gui-smoke's input_entry. Part 2b-ii is cut in three (D-49, register row A-29): 2b-ii-b the dimension tags, table rows and callout rows as selections, the callout list's move, the word budget of 250 and gui-smoke reading a tag's entry; 2b-ii-c the left column no longer swapping (4.3) and the table-driven nothing-lost test. Touching a row (its label, a drag, focus, an edit) selects its input, as in the Size column (D-50). The trace's banner stays at the inputs side's foot (D-52, A-30, owner question 55); the label's hover loses its trace line, with no line in its place (D-53, A-31, owner question 53); a note's change-entry links still only show the row (D-55, A-32, owner question 54). risk: mechanical."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown

---

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-i: the Size column's lines in the Explain panel (BL-090)
```

with:

```markdown

---

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-a: every input a selection (BL-092)
- What changed: touching a Detailed inputs or Assumptions row (a click on its label, a drag of its slider, focus in its field, an edit or a reset) selects its input, and the Explain panel shows its entry (its value, what it is, its hint, the checks whose fixes name it, the "pulls two ways" sentence, its cell with spreadsheet details on). Every input's entry starts with the link "Highlight the results this drives", which traces it (a second click ends the trace); the label no longer traces, and its hover lost the line that said it did. A check's fix link selects its input as well as showing its row. With spreadsheet details on, a tile's and a check's entry end with their cell. The gui-smoke run clicks the hovered Key design label and checks the input's entry (`input_entry`).
- Tests: any input's entry, the trace link, the fix link's selection, the cell feet (explain); a label's click and a touch of the row's widget select and a hover does not, the entry's link traces (panel); the trace tests go through the link; the smoke prompt guard and the verdict test (79 checks).
- Not in this part (plan decision D-49): dimension tags, results-table rows and callout rows as selections, the callout list to the Geometry tab, the word budget of 250, gui-smoke reading a dimension tag's entry (part 2b-ii-b); the left column no longer changing with the centre tab, the table-driven nothing-lost test (part 2b-ii-c); part 2b-iii as before.

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-i: the Size column's lines in the Explain panel (BL-090)
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  priority: 2
  status: open
  notes: "Filed 2026-11-23 with its plan (the design's header: each plan files one item). BL-090's part 2b-ii is cut in three (the plan's decision D-49, register row A-29): this part, 2b-ii-a, makes every input a selection; 2b-ii-b (dimension tags, results-table rows and callout rows as selections; the callout list to the Geometry tab; the word budget of 250 at 1280 x 800, goal G-1 and test 8, with part 2a's D-32, which copies of a Start here step's title go; gui-smoke reads a dimension tag's entry, part 2b-i's D-45, the plan's D-58) and 2b-ii-c (the left column no longer changing with the centre tab, 4.3, the trace's banner placed with it; the table-driven nothing-lost test of test 9) each get their own plan and item, as 2b-iii does. risk: mechanical (decision D-57): the entries draw texts the screen drew before; the one new on-screen string is 'Highlight the results this drives', the design's own words (5.2); no number is new."
- id: BL-055
```

with:

```yaml
  priority: 2
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2b-ii-a (Tasks 1 to 3 of the plan); the full gate, the smoke run and the whole-branch review are its Task 5. Filed 2026-11-23 with its plan (the design's header: each plan files one item). BL-090's part 2b-ii is cut in three (the plan's decision D-49, register row A-29): this part, 2b-ii-a, makes every input a selection; 2b-ii-b (dimension tags, results-table rows and callout rows as selections; the callout list to the Geometry tab; the word budget of 250 at 1280 x 800, goal G-1 and test 8, with part 2a's D-32, which copies of a Start here step's title go; gui-smoke reads a dimension tag's entry, part 2b-i's D-45, the plan's D-58) and 2b-ii-c (the left column no longer changing with the centre tab, 4.3, the trace's banner placed with it; the table-driven nothing-lost test of test 9) each get their own plan and item, as 2b-iii does. risk: mechanical (decision D-57): the entries draw texts the screen drew before; the one new on-screen string is 'Highlight the results this drives', the design's own words (5.2); no number is new."
- id: BL-055
```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
| A-28 | The footnote on what the size controls move ("Outside diameter moves the ring radius... Gap and magnet count stay as you set them.") moved into the controls' Explain entries. Its first two clauses repeat the Outside diameter and Overall length hovers word for word, and its last is false while Auto sets the magnet count. Which entries draw it, and is it cut? | Outside diameter's and Overall length's entries draw the whole sentence, except while Auto sets the magnet count; the gap's and the count's entries never draw it. No text is cut, and the hint lines that repeat a hover stay. | It removes the one false statement on screen and the lines that say nothing for the entry they sit in, and adds no new text to review. Owner question 52. | Cut it to "Gap and magnet count stay as you set them." in those two entries, and drop the hint lines of Minimum OD, Outside diameter, Magnet face gap and Magnets per ring, whose hovers say the same. | BL-090 review of 2026-11-23, findings PT-1 and PT-2 |
| A-22 | BL-082: below what skirt does the cap draw a caution? The owner's answer 4a (2026-11-22) took an amber caution below a minimum and the controller's offered 0.5 mm; no source gives a making limit for a thread-cut skirt of this size. | 0.5 mm, a guide value (`gui::geometry::MIN_CAP_SKIRT_MM`), against the nominal skirt (a tapped thread's root sits a few hundredths of a mm past its nominal diameter, so the real wall is that much thinner: about 0.52 to 0.53 mm, still over 0.5, for the starting design's nominal 0.55 mm on a fine thread). A skirt under it, compared as quoted at four significant digits, draws one amber line on the Size tab naming the skirt and the minimum; it counts as no failed check. The starting design's 0.55 mm draws none. | A caution, not a fail: it warns without moving the approved starting design or any verdict, and it says only that the wall is thin against a guide, not that it cannot be made. 0.5 mm keeps the starting design clear; a minimum between 0.55 and 0.70 mm would force a new starting design (owner question 4). | A larger minimum moves the starting design; a counted check would change the status line's count; no caution reverts BL-082. | `magcoupling-rs/src/gui/geometry.rs`, BL-082 |
```

with:

```markdown
| A-28 | The footnote on what the size controls move ("Outside diameter moves the ring radius... Gap and magnet count stay as you set them.") moved into the controls' Explain entries. Its first two clauses repeat the Outside diameter and Overall length hovers word for word, and its last is false while Auto sets the magnet count. Which entries draw it, and is it cut? | Outside diameter's and Overall length's entries draw the whole sentence, except while Auto sets the magnet count; the gap's and the count's entries never draw it. No text is cut, and the hint lines that repeat a hover stay. | It removes the one false statement on screen and the lines that say nothing for the entry they sit in, and adds no new text to review. Owner question 52. | Cut it to "Gap and magnet count stay as you set them." in those two entries, and drop the hint lines of Minimum OD, Outside diameter, Magnet face gap and Magnets per ring, whose hovers say the same. | BL-090 review of 2026-11-23, findings PT-1 and PT-2 |
| A-29 | BL-090's part 2b-ii (the Detailed inputs' rows, dimension tags, results-table rows and callout rows as selections, the trace as an entry link, the callout list's move, the word budget of 250, the left column that stops swapping, the table-driven nothing-lost test, the cell foot, the fix link's selection, gui-smoke's input entry) is again more than one plan a reviewer can hold. Built at once or in parts? | Three parts. 2b-ii-a: every input a selection (the rows' labels, the trace link, the fix link, the cell feet, gui-smoke's input entry). 2b-ii-b: dimension tags, results-table rows and callout rows as selections, the callout list to the Geometry tab, the word budget of 250 and which copies of a Start here step's title go. 2b-ii-c: the left column no longer changing with the centre tab and the table-driven nothing-lost test. | Smaller releases (the conservative answer to Q-4, as A-21, A-23 and A-25); each part loses nothing on screen. | One plan for all of 2b-ii. | The plan of part 2b-ii-a (BL-092), decision D-49 |
| A-30 | The trace becomes a link in an input's entry (5.2). Where do the trace's banner ("Tracing ...: drives N results with an equation") and its Clear trace button go? | They stay at the foot of the inputs side on the tabs other than the opening one, as now; on the opening tab, as since part 2b-i, a trace draws its frames and no banner. A second click on the entry's link ends the trace. | No new text, and no second copy of the banner on screen; part 2b-ii-c, which stops the inputs side swapping, has to place the banner anyway. The cost, until then: on the opening tab, once the selection moves on, no off switch is in view (select the input again and click its link, or use Clear trace on another tab), and a trace that reaches no result (the drive torque) shows nothing there. Owner question 55. | The banner and Clear trace move under the link in the entry. | The plan of part 2b-ii-a (BL-092), decision D-52 |
| A-31 | A row's label hover ended "Click the label to highlight the results it drives that have an equation". A click on the label now selects the input. What does the hover's last line say? | Nothing: the line goes, and no line takes its place. The entry's link says what it does. | The old line would be false; a new line is new text for review. Owner question 53. | A line such as "Click the label to show its entry" ends the hover. | The plan of part 2b-ii-a (BL-092), decision D-53 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
| A-22 | BL-082: below what skirt does the cap draw a caution? The owner's answer 4a (2026-11-22) took an amber caution below a minimum and the controller's offered 0.5 mm; no source gives a making limit for a thread-cut skirt of this size. | 0.5 mm, a guide value (`gui::geometry::MIN_CAP_SKIRT_MM`), against the nominal skirt (a tapped thread's root sits a few hundredths of a mm past its nominal diameter, so the real wall is that much thinner: about 0.52 to 0.53 mm, still over 0.5, for the starting design's nominal 0.55 mm on a fine thread). A skirt under it, compared as quoted at four significant digits, draws one amber line on the Size tab naming the skirt and the minimum; it counts as no failed check. The starting design's 0.55 mm draws none. | A caution, not a fail: it warns without moving the approved starting design or any verdict, and it says only that the wall is thin against a guide, not that it cannot be made. 0.5 mm keeps the starting design clear; a minimum between 0.55 and 0.70 mm would force a new starting design (owner question 4). | A larger minimum moves the starting design; a counted check would change the status line's count; no caution reverts BL-082. | `magcoupling-rs/src/gui/geometry.rs`, BL-082 |
```

Replace:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-23: A-28 added by the fix wave of the review of part 2b-i (BL-090; findings PT-1, PT-2). It is an assumption, not the owner's answer.
```

with:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-23: A-29, A-30, A-31 and A-32 added with the plan of slice B's stage 2, part 2b-ii-a (BL-092). Each is an assumption, not the owner's answer.
- 2026-11-23: A-28 added by the fix wave of the review of part 2b-i (BL-090; findings PT-1, PT-2). It is an assumption, not the owner's answer.
```

**`docs/analyses/2026-11-18-owner-questions.md`**

Replace:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 32 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, question 51 on 2026-11-23). Part D, five that matter only if the tools stop being generic.

```

with:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 35 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54 and 55 on 2026-11-23). Part D, five that matter only if the tools stop being generic.

```

Replace:

```markdown

### Linkage simulator
```

with:

```markdown

### 53. What should a row's label hover say now that a click on the label shows the input's entry?

Added 2026-11-23 (the next free number). The hover over a Detailed inputs row's label ended "Click the label to highlight the results it drives that have an equation". Part 2b-ii-a makes the click select the input (the design's 5.2); the highlighting is the link "Highlight the results this drives" at the top of the input's entry. The old line would be false, so it goes.

- **If you do not answer:** the hover ends at its Default line (and its spreadsheet lines), with no line about the click.
- (a) As the default.
- (b) A last line such as "Click the label to show its entry".

Source: UX design section 5.2; A-31; BL-092 plan, decision D-53.

### 54. Should a teaching note's input links select the input, as a check's fix links now do?

Added 2026-11-23 (the next free number). In a check's entry a fix link ("Raise Magnet face gap") now selects the input, so the Explain panel shows the input's entry, and shows its row. A teaching note's "What to change" lines link to inputs too; part 2b-ii-a leaves them as they were: the row shows and the note stays in the Explain panel.

- **If you do not answer:** a note's input links show the row and keep the note.
- (a) As the default.
- (b) They select the input, as a fix link does (the note gives way to the input's entry; "Checks it touches" and Learn lead back).

Source: UX design section 5.2; A-32; BL-092 plan, decision D-55.

### 55. Should a trace started on the opening tab keep an off switch in view?

Added 2026-11-23 (the next free number). Part 2b-ii-a moves the trace from a row's label to the link "Highlight the results this drives" in the input's Explain entry; a second click on the link ends it. On the opening tab the trace draws its frames and no banner or Clear trace (as since part 2b-i). Once you select something else there, the trace stays on with no off switch in view: select the input again and click its link, or open another tab for Clear trace. An input whose results have no equation record (the drive torque) traces none, and on the opening tab nothing says so. Part 2b-ii-c, which keeps the left column the same on every tab, places the banner anyway.

- **If you do not answer:** as described, until part 2b-ii-c.
- (a) As the default.
- (b) While the selected input is the one traced, its entry also shows the banner and Clear trace under the link (no new text; a second copy of the banner).

Source: UX design section 5.2; A-30; BL-092 plan, decision D-52; plan review UXP-3.

### Linkage simulator
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; parts 2b-ii and 2b-iii and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

with:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`; parts 2b-ii-b, 2b-ii-c and 2b-iii and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

**`magcoupling-rs/README.md`**

Replace:

```markdown

Each of the five controls (Minimum OD, Outside diameter, Overall length, Magnet face gap and
```

with:

```markdown

Since BL-092 every input is a selection: touching a row of Detailed inputs or Assumptions (a
click on its label, a drag of its slider, focus in its field, an edit or a reset, on the opening
tab and on the other tabs) selects its input, keyed by its path, so
the face gap's Key design row selects the Size column's gap entry. The entry leads with the value
(a blank, a selector's choice or a text in words: `inputs::value_text`) and the label, then "What
it is" (the row's help), "What sets it" (its hint, where it has one), "Checks it touches",
"Watch out" and, with Show spreadsheet details on, its cell; it has no torque-card row and no
lesson. Every input's entry starts with the link "Highlight the results this drives"
(`explain::TRACE_LINK`), which traces the input (see the inputs side below). A check's fix link
selects its input as well as showing its row (the ring radius's selects Outside diameter's row),
and with Show spreadsheet details on a tile's and a check's entry end with "Spreadsheet cell: ..."
(the Magnets fit check, which is no result, has none).

Each of the five controls (Minimum OD, Outside diameter, Overall length, Magnet face gap and
```

Replace:

```markdown
  row above them switches to the **Assumptions** view (decision M43-7). A click on a row's label
  traces the input (decision O-8, `trace.rs`): every explained result its value flows into (the
  registry's `downstream`; the registry explains 416 of the 1112 results, so an input such as the
```

with:

```markdown
  row above them switches to the **Assumptions** view (decision M43-7). A click on a row's label
  selects its input (since BL-092), and the link "Highlight the results this drives" at the top
  of its Explain entry traces it (decision O-8, `trace.rs`): every explained result its value flows into (the
  registry's `downstream`; the registry explains 416 of the 1112 results, so an input such as the
```

Replace:

```markdown
  panel's mark colour (`look::mark_color`, the host's link colour), with a banner and a Clear trace button at the foot of the inputs side (so a
  trace moves no row above it, and a second click on the label ends it; the locked free
  variable's label traces too); a click on a result (it still opens the Equation panel) traces
  the inputs it reads (`upstream_inputs`), unless an input's trace marks it or it has no equation
```

with:

```markdown
  panel's mark colour (`look::mark_color`, the host's link colour), with a banner and a Clear trace button at the foot of the inputs side (so a
  trace moves no row above it); a second click on the link ends it, and the locked free
  variable's label selects it too; a click on a result (it still opens the Equation panel) traces
  the inputs it reads (`upstream_inputs`), unless an input's trace marks it or it has no equation
```

Replace:

```markdown
and no path or spreadsheet cell, the Design menu's "Show spreadsheet details" toggle being off, as `details_toggle`
checks); then (slice 3) it opens the link again with a timer on every frame the app draws (eframe asks for an animation frame at every display refresh; the timer keeps those that make a WebGL draw call), clicks the Hot-low tile and checks the torque card's heading with its Why link, caption, Auto line and four rows in its Explain entry (the Overall length row held by Auto among them), and drags the Magnet face gap slider's handle, moving the pointer 60 times, 50 ms apart, alternately 3 px right and 3 px left of where it pressed: the drag's drawn-frame count, median, p95 and slowest frame are recorded on every run, and with `frame_check: true` (the release run) a drag that moved the gap, at least 30 frames and a p95 within one 60 fps frame (16.7 ms) are required; last (slice A, `address_follows`) it reads the address, which the app rewrote after the drag's settled edit, reloads the page and checks that the address is the same and the edited design is back (the one check of the `history.replaceState` call); `frame_check` is `true`, `false` or absent (any other value stops the run before its first step), and the calculator's result records it, so a run shows whether its check applied; the script checks the calculator agent's `passed` again from the agent's own fields (every check its schema requires true, the drag's two judged by the frame rule, no console error listed), and a step whose agent returns nothing reads every field its schema requires as false or empty; all with zero console
errors (a test decodes the pinned link and the design file, and checks that the steps name the right column's
```

with:

```markdown
and no path or spreadsheet cell, the Design menu's "Show spreadsheet details" toggle being off, as `details_toggle`
checks) and clicks its label (`input_entry`, since BL-092: the Explain panel shows the input's entry, its title, the link
"Highlight the results this drives" and "What it is"); then (slice 3) it opens the link again with a timer on every frame the app draws (eframe asks for an animation frame at every display refresh; the timer keeps those that make a WebGL draw call), clicks the Hot-low tile and checks the torque card's heading with its Why link, caption, Auto line and four rows in its Explain entry (the Overall length row held by Auto among them), and drags the Magnet face gap slider's handle, moving the pointer 60 times, 50 ms apart, alternately 3 px right and 3 px left of where it pressed: the drag's drawn-frame count, median, p95 and slowest frame are recorded on every run, and with `frame_check: true` (the release run) a drag that moved the gap, at least 30 frames and a p95 within one 60 fps frame (16.7 ms) are required; last (slice A, `address_follows`) it reads the address, which the app rewrote after the drag's settled edit, reloads the page and checks that the address is the same and the edited design is back (the one check of the `history.replaceState` call); `frame_check` is `true`, `false` or absent (any other value stops the run before its first step), and the calculator's result records it, so a run shows whether its check applied; the script checks the calculator agent's `passed` again from the agent's own fields (every check its schema requires true, the drag's two judged by the frame rule, no console error listed), and a step whose agent returns nothing reads every field its schema requires as false or empty; all with zero console
errors (a test decodes the pinned link and the design file, and checks that the steps name the right column's
```

Replace:

```markdown
headings, the orientation, the entry's headings and its lesson link, and no dashboard, the
toggle, the note's parts, the Geometry tab, the last row and the card's heading, caption, Auto line, held row and row starts as the page draws them). Its other step checks the hub page at `/tools/` and the linkage
app at `/tools/linkage/`. Gate 13 of `gate.sh` runs the script's own rules under node with a stub agent and no browser
```

with:

```markdown
headings, the orientation, the entry's headings and its lesson link, and no dashboard, the
toggle, the note's parts, the Geometry tab, the trace link, the last row and the card's heading, caption, Auto line, held row and row starts as the page draws them). Its other step checks the hub page at `/tools/` and the linkage
app at `/tools/linkage/`. Gate 13 of `gate.sh` runs the script's own rules under node with a stub agent and no browser
```

- [ ] **Step 2: Check.** `git diff --stat` lists the nine files and no other; `python -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ['docs/ai/backlog.yaml', 'docs/ai/02-system.yaml', 'docs/ai/03-structure.yaml', 'docs/ai/04-memory.yaml']]"` prints nothing; `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3` still reads `1103 passed`.

- [ ] **Step 3: Commit** the nine files: `docs(magcoupler): slice B stage 2 part 2b-ii-a docs sweep: README, docs/ai, the register, owner questions 53 to 55, the design's header, BL-092 in progress (BL-092 T4)`.

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

  Expected: the last line is `GATE PASS`; gate 12 printed `oracle Python:` and not a SKIP line; gate 13 printed `all 79 checks hold`; gate 14 built both web bundles. If a gate fails, fix the cause on the branch, red test first, and run this step again. Never edit `gate.sh` to pass.

- [ ] **Step 3: gui-smoke, from the branch's own script.**
  1. Check that port 8080 is free (`netstat -ano | findstr :8080` prints nothing): another branch's release uses it on this machine. Then, in the background: `& 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/serve_web.sh 8080`.
  2. Run the Workflow tool with `scriptPath` set to this worktree's `.claude/workflows/gui-smoke.js` and the arguments `{"frame_check": true}`.
  3. Expected: `passed: true`; in `magcoupling`: `starting_design`, `look`, `first_view`, `details_toggle`, `design_file_loaded`, `check_entry`, `lesson_link_opened`, `tooltip_plain`, `input_entry`, `torque_card`, `address_follows`, `drag_frames_within_60fps` all `true`, `console_errors: []`. Keep the result's JSON in `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2bii/smoke-local.json` and the run's first screenshot beside it.
  4. Stop the server: `TaskStop` on the background task that runs it, or `Stop-Process -Id <pid>` for the PID `netstat -ano | findstr :8080` shows. Check again that the port is free.

  An `input_entry` false is a finding about the build until shown otherwise: the step clicks the label it hovered (the face gap's Key design row on the Geometry tab) and looks for the entry's title, the link and "What it is" in the Explain panel.

- [ ] **Step 4: The text review** (fresh context, `fable`; the design's 8.3). It reads the one new string (`TRACE_LINK`) against the plain-words guard and the design's 5.2, checks that every line a Detailed input's entry draws is a text the screen drew at `main` (its label, its help as its hover showed it, its hint as its Key design row draws it, the guides' labels and words, the "pulls two ways" sentence, the cell) and that no engine text changed (`git diff main...HEAD -- magcoupling-rs/src/engine` is empty). It rules on two things and records what it decides: whether the hover, now without its last line, still tells a person that the label is clickable (decision D-53, owner question 53), and whether an entry that shows a Key design input's help and then its hint says one fact twice (as part 2b-i's review found for the Size rows, owner question 52); a change the owner should weigh goes to `docs/analyses/2026-11-18-owner-questions.md` as an open question with its conservative answer, not shipped silently. Its findings join Step 5's fix wave. Record it in `docs/analyses/2026-11-23-magcoupler-slice-b2p2bii-a-review.md` (committed in Step 7).

- [ ] **Step 5: The independent whole-branch review** (fresh context, `fable`, three lenses in parallel, each with a skeptic on every finding that is not a suggestion). Give each `git diff main...HEAD`, the design, this plan and Step 3's screenshot, and none of the builders' notes.
  1. **Conformance:** the design's 5.1 to 5.3 for inputs, results and checks where this part touches them, test 10 of 8.2 for an input, item by item against the diff; the decisions D-49 to D-59, each what the code does and none hiding a choice the owner should see; the nothing-lost table, row by row against the code.
  2. **Regressions:** no engine number moves; nothing reachable today is unreachable (the trace from every input, every fix link's row, the Assumptions view, every tooltip line but the trace line); no existing assertion about behaviour that stays was weakened (the list "Existing tests this plan edits"); an idle frame computes nothing new (a Detailed input's entry asks no consequences); the anti-goals of 2.3.
  3. **What a person sees:** a fresh render at 1024 x 700, 1280 x 800 and 1920 x 1080 of a Detailed input's entry (a number, a selector, a blank optional input), the Size column's gap entry reached from its Key design row, a check's entry with the details on, and a traced input: nothing cut or overlapping; the link under the title; the entries read in the design's order; crimson only on the brand mark and the active tab.

  One fix wave on `fable` for what Steps 4 and 5 find, each fix with its red test first; the reviewers re-read the fixes. Record the review in `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2bii/final-review.json` (gitignored).

- [ ] **Step 6: If Step 4 or 5 changed a file, run Steps 2 and 3 again.** The merge needs a gate and a smoke run of the tree that is merged.

- [ ] **Step 7: Close the item and merge.** In `docs/ai/backlog.yaml` (the day's date; say what the reviews found):

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2b-ii-a (Tasks 1 to 3 of the plan); the full gate, the smoke run and the whole-branch review are its Task 5. Filed
```

with:

```yaml
  status: fixed
  notes: "FIXED 2026-11-2X: built on branch magcoupling/overhaul-b2-part2b-ii-a; GATE PASS with --full, the gui-smoke run passed with frame_check true, the text review and the independent whole-branch review approved it. Filed
```

  (Write the day's date in place of `2026-11-2X`.) Commit it with the review record (`docs(magcoupler): BL-092 fixed: the gate, the smoke run and the reviews passed`), then merge the branch into `main` with a merge commit whose message lists the tasks' commit subjects. Remove the worktree with `git worktree remove`, never `git worktree prune`.

- [ ] **Step 8: Push, then report** (the owner's ruling of 2026-11-22: each stage goes live once the gate, the smoke run and the reviews pass, with a report afterwards; no stop). Push `main` (it deploys the site). When the `deploy-web.yml` run is green, run the smoke workflow from `main`'s script with `{"url": "https://colesorkness.com"}` and report the result; if it fails, say so plainly: the site is live with the change. The report to the owner holds: what changed on screen, in three sentences; two pictures (a Detailed input's entry, a check's entry with the details on); the decisions D-49 to D-59, one line each; owner questions 53 to 55; parts 2b-ii-b, 2b-ii-c and 2b-iii; what was not verified (Safari and Firefox; in a browser, an assumption's row and a fix link's selection: the smoke clicks one Key design label only).

## What this plan covers of the design

| Design item | Where |
|---|---|
| 5.1 An input as a selection (any input, not only the Size column's six rows); Esc returns the orientation | Tasks 1, 2 (Esc as for every selection) |
| 5.2 Touching an input selects it: a click on a Detailed inputs or Assumptions row's label, a drag of its slider, focus in its field (and an edit or a reset) | Task 2 (D-50); the Size column's touches since part 2b-i |
| 5.2 Tracing stays, as a link in the input's entry ("Highlight the results this drives") | Tasks 1, 2 (D-51, D-52) |
| 5.2 A fix link selects and focuses its input, opening the fold that holds it | Task 1 (D-54); the fold since part 2b-i (D-48) |
| 5.2 A link inside the Explain panel selects what it names (a note's input links) | Not in this part: they keep showing the row (D-55, owner question 54) |
| 5.2 A hover selects nothing | Task 2 |
| 5.3 An input's entry for any input: title and value, What it is, What sets it, Checks it touches, Watch out, foot | Task 1 (D-50) |
| 5.3 "Spreadsheet cell" foot of a result's and a check's entry | Task 1 for a tile and a check (D-56); any result's in part 2b-ii-b |
| 5.1, 5.2 Dimension tags, results-table rows, callout rows as selections; 4.4 the callout list to the Geometry tab | Part 2b-ii-b |
| 8.2 Test 8 (word budget of 250) | Part 2b-ii-b (the ceiling of 280 since part 2b-i holds) |
| 9.1 The `gui-smoke` checklist: a dimension tag's entry | Part 2b-ii-b (this part adds the input's, D-58) |
| 4.3 The left column no longer changes with the centre tab; 8.2 test 9 table-driven | Part 2b-ii-c |
| 8.2 Test 10 (selection): a click on an input row's label; a hover changes nothing | Task 2 |
| 8.3 Text rules | Task 1 (plain words, glyphs), Task 5 Step 4 |
| 9.1 Docs with the slice, the `gui-smoke` checklist | Tasks 3, 4 |
| 9.4 Release | Task 5 (no stop: the owner's ruling of 2026-11-22) |
