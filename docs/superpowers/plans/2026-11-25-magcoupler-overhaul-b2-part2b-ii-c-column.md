# Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-c (the left column and the nothing-lost test): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ] `) syntax for tracking.

**Goal:** The left column no longer changes with the centre tab (the design's 4.3). On every tab it is the Size column: the six size controls, then two folds, both shut at first, Detailed inputs (the order toggle, the filter box, the Key design group and every input group, now their only home) and Assumptions. The tabs "Design inputs" and "Assumptions" that swapped the side go. A trace's banner and its Clear trace button sit at the column's foot on every tab, the opening one too (part 2b-ii-a's decision D-52 left them for this part). The folds scroll with the column (BL-091) and are drawn with no indent, so an opened fold is no wider than the inputs side was (320 points in egui's default font; in the app's own font the column still grows by 10 to 12 points when a fold first opens, as the inputs side did: BL-099). Tab through the column selects nothing while every size is on its slider's step (BL-093's rule; BL-097 holds the rest). The design's nothing-lost table (8.2 test 9) becomes one table-driven test. The gui-smoke run judges the left column on the Geometry tab. Nothing that is on screen today is lost (the nothing-lost table).

**Architecture:**
- **One column.** `MagcouplingPanel::inputs_ui` draws `size_column_ui` in the column's one scroll area whatever `CentreView` shows, over the trace's banner and Clear trace, laid out bottom up at the column's foot as they were at the foot of the inputs side. `InputsView` and the field `inputs_view` go; `ASSUMPTIONS_FOLD` is its own text. A row asked for (a leaf term, a fix link) opens its fold, as it did on the opening tab.
- **The folds.** `design_inputs_ui` and `filtered_inputs_ui` draw no scroll area of their own (BL-091, as `assumptions_ui` since BL-090), and both folds are drawn with `show_unindented`: an indented fold pushed its rows past the column's edge and widened the column from 320 to 333.2 points when it opened. Those are the numbers of egui's default font, the panel tests' font. In the app's own font some rows are wider than the column either way (the magnet part pickers, and labels that do not wrap): the column is 330.1 points once Detailed inputs has opened, as the inputs side was at `main`, against 348.1 with the indent. That growth is BL-099, filed and not fixed here.
- **The Outside diameter's fix.** Its link still opens the Size and torque tab; `column_to_top` then scrolls the column back to its top for one frame, where the slider is: inside the open Detailed inputs the column can be scrolled far below it.
- **The Minimum OD's box.** `size_controls::controls_ui` hands the Minimum OD's slider the value its box shows (`emath::round_to_decimals` at the step's decimals). egui writes a value box's text back when the keyboard's focus leaves it, and the floor of most designs is off the box's one decimal, so a Tab that only passed counted as a change: it selected the row and set the Minimum OD.
- **The tests' harness.** `Harness::on_screen` (the Geometry tab) opens the Detailed inputs fold and brings its header to the column's top, so the tests that drive the Key design group, the groups and the filter see them where the inputs side drew them; `Harness::size_and_torque` clicks the centre's tab and closes that fold, for the tests of the size controls; `Harness::fold` shows either fold. No assertion about behaviour that stays is weakened.
- **Unchanged:** the engine, the exports, the top bar, the Size column's own rows and their entries, the tiles, the check list, the Explain panel, the Equation panel, the centre views, every input row, the Assumptions view's contents, the trace itself and its frames, the linkage app.

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32, egui_plot 0.33; the headless egui harness (`magcoupling-rs/src/gui/test_support.rs`, the panel tests' `Harness`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. This part is its sections 4.1 (the layout's left column), 4.3 ("The left column no longer changes with the centre tab. Today it swaps between the size controls and the workflow groups; after, the workflow groups live only inside Detailed inputs"), 5.2 as BL-093 amended it (the keyboard's focus arriving selects nothing), goal G-4 and test 9 of section 8.2 (nothing lost, table-driven), and the docs and release rules of section 9. Executors read the spec's sections 4.1 to 4.3, 5.2, 8.2, 8.3 and 9 beside this plan. Part 2b-ii-a's plan (`docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, "What follows", part 2b-ii-c items 1 and 2) sets the scope. Backlog item: **BL-096** (`docs/ai/backlog.yaml`, filed with this plan); **BL-091** is taken with it (decision D-72). The planner's two findings on `main`, filed with this plan and not fixed by it: **BL-097** and **BL-098**. The plan's review filed two more, on `main` too and not fixed by it: **BL-099** and **BL-100**.

**How this plan was made.** The part was built once as a prototype on a copy of `main` at `f1f1b90` (BL-094 merged), one commit per task and one for the column task's red state, then the backlog items filed (BL-096 to BL-098) and the docs swept. The plan was then reviewed through two lenses (the replay and its coverage; the UX and the on-screen text). One fix wave followed: the review's two findings on `main` were filed (BL-099, BL-100) and the prototype's states rebuilt with its fixes, which are Task 1's second red test, the text guard and six tests of Task 3, and the lines Task 5 had left false. Every block below was generated from those commits' diffs (Tasks 1 and 2: the red step is the test hunks, the green step the rest; Task 3: the red step is the new tests with the harness's new moves, the green step the column and the tests it moves; Task 4: the red step is the prompt's guard in `app.rs` and the verdict test, the green step the prompt), then applied, in this plan's order, with the apply tool, to a fresh copy of `main` with BL-096 to BL-100 filed, in a target directory of its own. Each block applied once; each red step failed as its "Expected" says; the result equals the prototype file for file (line endings aside) and passes `cargo test` with and without the app feature (1119 lib tests with it against 1113 on the base), both native clippy runs after each task, both wasm32 clippy runs, `cargo fmt --check` and the gui-smoke verdict test (`all 83 checks hold`). Each count after a task is the replay's. Not run for this plan: gates 1 to 3 (linkage-sim-rs, untouched), gate 12 (the Python oracle), the release WASM build, the browser and the pictures. Task 6 runs them.

**Where this part applies.** On `main` at `f1f1b90` with this plan's commit (the plan and BL-096 to BL-100), and nothing else that touches `magcoupling-rs/`, `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` or Task 5's ten files. If a block does not match, stop and report: the controller regenerates the blocks (the planner's tools are listed under "Before Task 1"), it does not adapt a block in place.

## Why one plan, and what follows

Part 2b-ii-a's plan listed two items for this part. The prototype holds them in four code tasks and one docs task. Tasks 1, 2 and 4 are small (about 430, 170 and 270 plan lines). Task 3 is the column, and the long one: its red step is three new tests with the harness's moves (about 530 plan lines), its green step 107 blocks (about 1,990 plan lines). Of those, the production change is about 70 lines of `panel.rs`; the rest move the tests to what they read, 23 of them by one line each (a click on the Size and torque tab becomes `size_and_torque()`), each block that line with its neighbours. The harness, the 20 tests that read the column and the column change together or the suite is red (four more are edited so that none leans on the harness's scroll or misses the opening tab: the plan's review, RC-1 and RC-3); the 23 one-line edits could go first in a task of their own, which would have no red step and would spare a reviewer only the blocks that are easiest to check. One reviewer can hold each task with the list under "Files" beside it, so this plan does not cut again (decision D-71).

**What follows, with its own plan and backlog item** (unchanged from part 2b-ii-a's list):

**Part 2b-iii (the Plots tab and the look):** the Plots tab with its picker (C-9), the plots' red reference lines (C-3), egui_plot's fading tick numbers (E-7), the controls' outlines (E-4, R-6), a SemiBold face (S-1), the page shell's loading line (S-4), BL-086, BL-088, BL-089. Whether the plots' readouts select their result (owner question 56) is that plan's to weigh. BL-083, BL-084 and BL-095 stay open, and so do this plan's four findings: BL-097 (Tab through a value box rewrites a value off its step), BL-098 (a teaching note widens the right column by 16 points), and from its review BL-099 (in the app's own font an opened fold or a filter widens the left column) and BL-100 (a row that takes the keyboard's focus is not scrolled into view).

## What the prototype showed

1. **The Minimum OD took the selection from the keyboard.** With the column on every tab, Shift+Tab from a Detailed input passes the size controls. On the spreadsheet's design the Minimum OD's floor is off the one decimal its box shows; egui writes a value box's text back when the focus leaves it, so the slider reported a change, the row took the selection and the Minimum OD was set. BL-093's own test (`a_keyboard_user_traces_a_middle_input_with_tab_or_shift_tab_to_its_entry_s_link`) failed backwards for it. On `main` the same happens on the opening tab with Tab forwards (Task 2's red step shows it). Task 2 hands that slider the value its box shows. The wider case, any input whose value is off its step (a loaded file's), is on `main` too and is filed as BL-097: the probe, a face gap of 1.4123, became 1.41 after 17 presses of Tab, with an undo step. The plan's review found the way there that needs no file: Auto on, Auto off (the inputs adopt the solved magnet length, 11.942 mm on the starting design), then Tab through the Size column: the length becomes 11.9 mm, hot-low falls to 1.3445 N·m under the 1.35 N·m requirement and a check fails, with nothing typed. On `main` too; it is BL-097's red test, and BL-097 is priority 2.
2. **An opened fold widened the column.** `CollapsingHeader::show` indents its body by 18 points; the input rows then ran past the column's edge and the left side panel grew from 320 to 333.2 points (334 with the Assumptions fold), moving the centre region. `show_unindented` gives the rows the width they had on the inputs side (decision D-75). The existing test `hint_lines_stay_inside_their_sides_in_a_narrow_window` caught it once the harness opened the fold. In the app's own font the column still grows when a fold opens, to 330.1 points (348.1 with the indent), as the inputs side did at `main`: the review's finding, filed as BL-099.
3. **The folds leave room.** The size controls end at 333 points down the window at 1280 x 800 and at 1024 x 700; the two folds' headers follow at 333 and 354. A banner at the column's foot is whole inside the window at 1024 x 700.
4. **A long scroll takes time.** A row asked for far down the open Detailed inputs scrolls into view over up to a third of a second (egui's scroll animation, 18 frames); the tests that asked five frames of it ask thirty.
5. **A row in both folds.** With the Detailed inputs open, an assumption's row asked for is framed in its input group, which opens, and in the Assumptions fold, which opens too; the column scrolls to the first drawn (decision D-78). With the Detailed inputs shut, as on a new panel, only the Assumptions fold's row shows.
6. **"Size and torque" twice.** The column's heading and the centre's tab bear the same words on every tab now, so a test's click on the first such text hit the heading. `Harness::tab` clicks the one in the centre region. The text guard's helper, `every_shown_text`, walked the tabs by the first such text and so stopped visiting the opening tab's centre, with the guard still green (the review's finding RC-1): it clicks through `Harness::tab` too, and the guard asserts that a tile's word is collected.
7. **A fix on the Outside diameter could show nothing.** Its link opened the Size and torque tab, whose left column was the size controls. With one column, scrolled down inside the open Detailed inputs, the slider stayed out of view. `column_to_top` scrolls it back (decision D-76); the test fails without it.
8. **A teaching note widens the right column.** The nothing-lost test reads the right column's edge off each frame because the note "harmonics", and every note after it, leaves that column 16 points wider. On `main`; filed as BL-098.
9. **rustfmt and line endings.** `cargo fmt` rewrites a file it changes with LF line endings on this machine; git's `autocrlf` takes them back at the commit, and the apply tool reads each file's endings as it finds them.

No pictures were drawn for this plan: Task 6's review renders the Geometry and Results tabs with the folds shut and open, and a trace's banner on the opening tab, at the three window sizes.

## Global Constraints

- **No engine number moves** (the design's 2.3). No file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` changes.
- **The linkage app is not touched** (O-1). No file under `linkage-sim-rs/src/` or `linkage-sim-rs/web/` changes.
- **The panel sets no theme and no font** (2.2, 6.3). Code under `magcoupling-rs/src/gui/` reads `ui.visuals()`.
- **No new on-screen string.** The column draws on every tab what the opening tab drew; the banner and Clear trace are the texts they were. One text leaves the screen: the tab "Design inputs" (the tab "Assumptions" is the fold's header).
- **Crimson is never text, never on a value, never in the check list** (C-5).
- **View state stays view state** (5.1): the folds' open state, the column's scroll and `column_to_top` are in no design file, share link or undo step.
- **A hover selects nothing** (C-2), and neither does the keyboard's focus arriving or leaving (BL-093): only a click, a drag, an edit, or Enter or Space on the focused control.
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
  `cargo fmt` changes no text of a task's files (the blocks are formatted); it may rewrite a file it reads with other line endings, which git's `autocrlf` undoes at the commit. After it, `git status --short` shows only the files the task names. If it shows another, stop and report.
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-25-magcoupler-overhaul-b2p2biic/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Shell.** On this machine `bash` typed in PowerShell is WSL, which fails on the checkout's line endings. Shell scripts run through Git Bash: `& 'C:\Program Files\Git\bin\bash.exe' <script>`. Python scripts are written with the Write tool, not a here-document. Do not edit a source file with `sed -i` from Git Bash: it rewrites the file's line endings.
- **Never run `cargo fmt` inside `linkage-sim-rs`.** `magcoupling-rs` is fmt-clean; keep it so.
- **Docs ship with the code** (repository rule): Task 5 sweeps the README and `docs/ai/`, the register, the owner-question list and the design's status line.

## Model per task

CLAUDE.md section 5. BL-096 is `risk: mechanical` (decision D-83): the part shows no new number, makes no new physical claim and adds no on-screen string; the column draws on every tab what the opening tab drew. So the tasks that transcribe the plan's exact blocks run on `sonnet`, and every review and the release run on `fable`. A `sonnet` task that ends blocked, leaves the gate red or is rejected in review is redone on `fable` (escalate, never downgrade).

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | The Detailed inputs fold scrolls with the Size column (BL-091: `design_inputs_ui`, `filtered_inputs_ui`) | `sonnet` |
| 2 | Tab through the Size column selects nothing and sets no Minimum OD (`size_controls::controls_ui`) | `sonnet` |
| 3 | The left column is the Size column on every tab (`inputs_ui`, `InputsView` gone, the banner, the folds' width, `column_to_top`); the tests' harness; the nothing-lost test | `sonnet` |
| 4 | gui-smoke judges the left column on the Geometry tab (`same_column`) | `sonnet` |
| 5 | Docs sweep | `sonnet` |
| 6 | Release: the full gate, gui-smoke, the text review, the independent whole-branch review, one fix wave, the merge and the push | `fable` controller; fresh-context `fable` reviewers; `haiku` for git; the smoke workflow's own agents on `sonnet` |

Each task's diff is reviewed by a fresh-context reviewer before the next task starts (`sonnet` for these mechanical diffs, CLAUDE.md section 5); it checks the diff against the task's blocks and nothing else. A task that ends blocked or leaves a test red stops the build and goes back to the controller.

## Review Focus

Ten inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named.

1. **A fold opened on one tab, read on another.** The column, its rows and a fold's open state are the same on every tab. Task 3: `the_left_column_is_the_size_column_on_every_tab` (each row's place on all nine tabs; the Detailed inputs opened on the Results tab and read on the Geometry tab).
2. **The banner on the opening tab at 1024 x 700.** It was drawn on no first view before. Task 3: `a_trace_s_banner_and_clear_trace_sit_at_the_left_column_s_foot_on_every_tab` (whole inside the window, under the folds, no row moved, still there with another thing selected).
3. **Shift+Tab from a Detailed input.** The keyboard's way back to the entry's trace link now passes the size controls. Task 2: `tab_through_the_size_column_selects_nothing_and_sets_no_minimum_od`; Task 3 keeps BL-093's `a_keyboard_user_traces_a_middle_input_with_tab_or_shift_tab_to_its_entry_s_link` unedited.
4. **The column's width with a fold open.** Task 3: `the_left_column_is_the_size_column_on_every_tab` (320 points with either fold open), and `hint_lines_stay_inside_their_sides_in_a_narrow_window`, unedited. Both in egui's default font; the app's own font is BL-099's.
5. **A row asked for far down the column.** A leaf term or a fix link opens the fold and scrolls the one column. Task 1: `the_detailed_inputs_fold_scrolls_with_the_size_column_to_its_last_group` (at 1024 x 700, from the column's foot back to the face gap) and `the_filtered_inputs_scroll_with_the_size_column_to_their_last_row` (the filter's list, to its last row); Task 3: `a_leaf_assumption_term_shows_its_row_in_the_assumptions_view` (both with the Detailed inputs shut and open).
6. **A fix on the Outside diameter from a scrolled column.** Task 3: `an_outside_diameter_fix_opens_the_size_tab_and_edits_nothing` (at 1280 x 800, the slider out of view before the link and in view after).
7. **Auto on with the fold open.** The solver's status shows in the Size column and in the Key design group (decision D-77). Task 3: `an_unreachable_target_shows_the_best_value_and_its_overshoot` counts both.
8. **A window too short for the folds.** At 240 or 300 points no fold's header is in view, but a fix link or a leaf term opens the Detailed inputs without it; the panel must only not panic and edit nothing. Task 3: the three small-window tests (`the_equation_panel_draws_in_a_tiny_window` and its two neighbours), the fold opened that way (`explorer.focus_input`) so its rows are drawn in each window.
9. **The paused clamp.** Its tab, its inputs and its two assumptions stay out of the column on every tab. Task 3: the nothing-lost test runs with the section shown and paused.
10. **The Minimum OD still moves.** Handing its slider the shown value must not stop an arrow key or a typed value. Task 2: the same test's arrow key; `manual_size_has_an_adjustable_minimum_od` and `manual_minimum_rejects_nonfinite_typed_values_without_changing_design`, their assertions unedited.

## Decisions this plan takes

Each is a reading of the design where it was open, numbered after part 2b-ii-b's D-60 to D-70 (one stage, one list). The owner may overrule any of them; Task 6 lists them in its report. The ones that need a row in the register have one (Task 5).

| Id | Decision | Why |
|---|---|---|
| D-71 | Part 2b-ii-c is one plan: part 2b-ii-a's two items, in four code tasks and a docs task. | Three tasks are small. The column task is long in plan lines (107 blocks) and short in change: about 70 lines of production code, and test edits that are mostly one line each. The column, the harness and the tests that read the column change together; a task of the one-line edits alone would have no red step. |
| D-72 | BL-091 is taken into this part, as Task 1, and closed with BL-096. | The fold becomes the only home of the input groups on every tab; the short nested pane BL-091 describes would be the only way to them. The harness of Task 3 scrolls the one column. |
| D-73 | The Minimum OD's slider is handed the value its box shows, so the keyboard's focus leaving its box changes nothing: no selection, no Minimum OD set. The wider case (any value off its step, which Auto turned off leaves behind) is filed as BL-097 and not fixed here. | BL-093's rule, and the brief's: with the column on every tab, Shift+Tab from any Detailed input passes this box. The fix is four lines in one control; BL-097 reaches every input row's edit path. |
| D-74 | The trace's banner and Clear trace sit at the foot of the left column on every tab, the opening one too, under the column's scroll area. Register row A-37; owner question 60. | Part 2b-ii-a's D-52 (A-30) left them for this part. No new text and one copy; where they sat, on a column that no longer swaps. The opening tab gains the off switch owner question 55 asked about. |
| D-75 | The two folds are drawn at the column's full width, with no indent and no new mark between them. Register row A-41; owner question 64. | An indented fold widened the column from 320 to 333.2 points when opened (the prototype, item 2). The rows keep the width they had on the inputs side, so the layout tests keep their numbers (egui's default font; in the app's own font the column is 330.1 points with a fold open, as the inputs side was: BL-099). The cost: with Detailed inputs open, the Assumptions header starts at the same edge as the eight group headings and reads as a ninth group. |
| D-76 | A fix link on the Outside diameter still opens the Size and torque tab, and the left column scrolls back to its top, where the slider is. Register row A-38; owner question 61. | The tab switch takes nothing a user had. The scroll is new and needed: the link's job is to show the slider (5.2), and one column can be scrolled far below it. |
| D-77 | The Key design group keeps Auto's line ("... set by the solver (Size tab)"), its "Open the Size tab" button and the solver's status. Register row A-39; owner question 62. | On-screen wording is kept unless the scope needs new text; nothing reads false (the button opens the Size and torque tab). The cost: with Auto on and the fold open, the status shows twice in the column, as on the opening tab since part 2b-i. |
| D-78 | A row asked for is framed in each open fold that holds it and scrolled to where it is drawn first: in its input group when the Detailed inputs are open, else in the Assumptions fold for an assumption. | The opening tab's behaviour since part 2b-i, now every tab's; no new rule. Before, the other tabs switched the side to the Assumptions view. The cost: with the Detailed inputs open, the Assumptions fold opens out of view and the assumption's rationale is not brought into view. |
| D-79 | A trace frames input rows only inside the folds: the size controls take no frame and the Detailed inputs' header shows no count. Register row A-40; owner question 63. | No new mark and no new text. The cost: with a result traced and the folds shut, the banner says "reads N inputs" and no input row is framed in view; one click on the fold shows them. |
| D-80 | The nothing-lost test looks for every input and every assumption from every tab, and is part of Task 3's red step. | Test 9 with 4.3. On `main` it fails on the Geometry tab, where the left column has no Size control: the red is this part's own change. |
| D-81 | The existing tests keep their subject through three moves of their harness: `on_screen` opens the Detailed inputs and brings them to the column's top; `size_and_torque()` clicks the centre's tab and shuts the fold; `fold()` shows either fold. The three small-window tests build a new panel, whose windows show no fold's header to click, and open the Detailed inputs as a fix link or a leaf term does (`explorer.focus_input`). | Each test then reads the rows it read, where it read them. The first-view tests (`first_view`, `size_tab`) open nothing, as a new panel does. |
| D-82 | The gui-smoke prompt gains `same_column`, judged on the Geometry tab before the tooltip step, which now opens the Detailed inputs fold to reach its Key design row. | 4.3 in a browser with no new navigation; without the fold's click the tooltip step would find no row. |
| D-83 | BL-096 is `risk: mechanical`; its build tasks run on `sonnet`, its reviews and release on `fable`. | No new number, physical claim or on-screen string (the design's header). |
| D-84 | Four findings on `main` are filed and not fixed: the planner's BL-097 (Tab through a value box rewrites a value off its step) and BL-098 (a teaching note widens the right column by 16 points), and the plan's review's BL-099 (in the app's own font an opened fold or a filter widens the left column) and BL-100 (a row that takes the keyboard's focus is not scrolled into view). | The design's 9.1: new findings are filed, never fixed in passing. Each has its probe in its backlog item. |
| D-85 | The left column keeps its heading "Size and torque" on every tab. Register row A-42; owner question 65. | No new on-screen string (Global Constraints): the heading is the text the opening tab's column had. The cost: on the other tabs it repeats the words of a tab that is not the one shown, and the design's sketch (4.1) heads the column "Size". Task 6's text review rules on it. |

## Nothing lost

Every text and action this part moves, and where it is after. Anything not listed (the top bar, every tile, check row and entry, the Size column's rows, every input row, every centre view) does not move. The "Test" column names the test that holds the new home. The design's own table (8.2 test 9) is the test `nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home`.

| Today | After this part | Test |
|---|---|---|
| On every tab but the opening one, the left side is the tabs "Design inputs" and "Assumptions" over the input groups or the Assumptions view | The Size column, as on the opening tab; the input groups one click away, in its Detailed inputs fold; the assumptions one click away, in its Assumptions fold | `the_left_column_is_the_size_column_on_every_tab`; the nothing-lost test (every input and every assumption, from every tab) |
| The order toggle, the filter box and the Key design group, in view at the top of that side | Inside the Detailed inputs fold, in the same order. A cost: the order toggle, the filter box and the filter's "N of M inputs" line stayed in view above the side's own scroll area while the rows scrolled; they now scroll away with the column, so refining a filter far down a long list means scrolling back up (BL-091: the fold has no pane of its own) | `the_left_column_is_the_size_column_on_every_tab`; `the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design`, `the_order_toggle_switches_the_view_and_changes_no_input` |
| The size controls, on the opening tab only | On every tab | `the_left_column_is_the_size_column_on_every_tab` |
| A trace's banner and Clear trace at the foot of the inputs side; neither on the opening tab | At the foot of the left column on every tab (D-74) | `a_trace_s_banner_and_clear_trace_sit_at_the_left_column_s_foot_on_every_tab`; `a_trace_moves_no_input_row_and_a_second_click_on_the_label_traces_nothing` |
| A traced result frames the Key design rows it reads, in view on the other tabs | The same rows, framed, inside the Detailed inputs fold (D-79) | `clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group` |
| A leaf term or a fix link of an assumption switches the side to the Assumptions view and frames its row | It opens the Assumptions fold and frames its row there; with the Detailed inputs open, in its input group too (D-78). A cost: with the Detailed inputs open the column scrolls to the row in its input group and the Assumptions fold opens out of view, so the rationale the other tabs brought into view is not shown until the user scrolls to it | `a_leaf_assumption_term_shows_its_row_in_the_assumptions_view`, `a_sum_s_harmonic_set_marks_its_row_and_its_h_finds_it`, `the_panel_works_inside_an_egui_window` |
| A fix link on the Outside diameter opens the Size and torque tab, whose left side shows the slider | It opens that tab and scrolls the column back to the slider (D-76) | `an_outside_diameter_fix_opens_the_size_tab_and_edits_nothing` |
| The Detailed inputs fold scrolls in a pane of its own | It scrolls with the column (BL-091), the filtered view too | `the_detailed_inputs_fold_scrolls_with_the_size_column_to_its_last_group`, `the_filtered_inputs_scroll_with_the_size_column_to_their_last_row` |
| The keyboard's way to the assumptions: on every tab but the first, the Assumptions tab is the sixth press of Tab, in view | The Assumptions fold's header, after the size controls: about the twentieth press with the folds shut. A cost: with Detailed inputs open it comes after every row of that fold (about the sixtieth press) and below the window's foot, where the column does not follow the focus (BL-100; on `main` too for the inputs side's lower rows). Nothing is unreachable and nothing traps the focus | No test holds the count of presses; `tab_through_the_size_column_selects_nothing_and_sets_no_minimum_od` holds that the walk selects nothing |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/panel.rs` | `design_inputs_ui` and `filtered_inputs_ui` without a scroll area, the inputs side wraps the design inputs; two tests (1). A test (2). `inputs_ui` draws the Size column on every tab over the banner; `InputsView` and `inputs_view` go; `ASSUMPTIONS_FOLD`; the folds unindented; `column_to_top`; the module's and four functions' comments; the harness (`fold`, `size_and_torque`, `tab`, `click_fold`, `in_left_column`, `column_rows`); three new tests; 47 tests and two helpers edited, the helper `count_in_fold` added (3) | 1, 2, 3 |
| `magcoupling-rs/src/gui/size_controls.rs` | The Minimum OD's slider is handed the value its box shows | 2 |
| `magcoupling-rs/src/gui/mod.rs` | `SIZE_ROWS` re-exported for the app's test of the smoke prompt | 4 |
| `magcoupling-rs/src/app.rs` | The smoke prompt's guard: the left column's words on the Geometry tab, the fold opened before the tooltip | 4 |
| `.claude/workflows/gui-smoke.js` | `same_column` (schema, prompt, passed list), the fold's click before the tooltip, the description and comments | 4 |
| `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` | `CHECKS` gains `same_column` | 4 |
| `magcoupling-rs/README.md`, `docs/ai/*`, the register, the owner questions, the design's header, `magcoupling-rs/src/gui/trace.rs` (two comments) | The docs sweep | 5 |
| `docs/ai/backlog.yaml` | BL-096 in progress (5); BL-096 and BL-091 fixed (6) | 5, 6 |

**Existing tests this plan edits, and how.** No assertion about behaviour that stays is weakened. All of these are `panel::tests`, in Task 3's green step, beside the change that moves what they read.
- **The harness.** `Harness::on_screen` and `with_baseline` open the Detailed inputs fold and bring its header to the column's top (`fold`); `size_tab` builds on a new panel, which opens on that tab, and no longer clicks it. The 130 or so tests that only build `Harness::new()` and drive the Key design rows, the groups or the filter are not edited: they find those rows where the inputs side drew them. Seven tests need the harness's scroll to reach their row: `a_leaf_term_opens_its_group_and_highlights_its_input_row`, `an_input_row_s_hover_names_its_path_and_cell_only_with_the_spreadsheet_details`, `an_outside_diameter_fix_opens_the_size_tab_and_edits_nothing`, `hovering_a_value_marks_its_terms_where_their_values_are_shown`, `open_the_example_design_is_one_undo_step_and_keeps_the_baseline`, `the_changed_dot_a_row_s_reset_and_the_default_line_follow_the_baseline` and `tracing_an_input_frames_the_results_it_drives_until_a_second_click`. The size controls above the fold bear the same labels and values, so the fold's row must be the first match of its text (the Outside diameter's test starts from the scrolled column by design). That is their path to the row, not a weakened assertion: with the harness's wheel removed these seven fail and no other test does.
- **23 tests of the size controls** change one line each: their click on the first "Size and torque" text becomes `harness.size_and_torque()` (the centre's tab, and the fold shut): `idle_frames_change_no_input`, the `manual_size_*`, `manual_minimum_*`, `loaded_smaller_designs_*`, `smaller_exploration_*` and `the_skirt_caution_*` tests, `size_tab_idle_frames_keep_an_off_step_or_out_of_range_requirement`, `leaving_torque_to_magnets_keeps_the_sized_value_as_one_undo_step`, `idle_frames_with_auto_on_change_nothing_and_solve_once`, the seven `auto_*` tests, `a_size_tab_edit_under_ring_size_auto_keeps_the_inputs_cap`, `a_diameter_edit_under_magnet_count_auto_maps_under_the_count_shown`, `an_inverse_mode_design_file_opens_with_auto_on` and `the_size_tab_keeps_auto_on_and_auto_off_adopts_the_solution`.
- **Three tests return to the Detailed inputs after the size controls** and add `harness.fold(DETAILED_INPUTS)`: `the_free_variable_picker_switches_the_variable`, `ring_size_auto_locks_the_cap_rows_at_the_design_shown` and `the_auto_texts_have_glyphs_and_wrap_on_the_inputs_side`. The helper that collects every text the panel shows (`every_shown_text`) opens the Assumptions fold and returns the same way, and clicks each tab in the centre region (`Harness::tab`): the first "Size and torque" on the page is the column's heading now, and a click on it changes no tab. The guard that reads the helper, `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`, asserts that a tile's word (`tiles::NEEDS`) is collected; without the helper's edit that assertion fails.
- **Seven tests named the side's two views.** `a_click_on_an_input_row_s_label_selects_its_input_and_a_hover_selects_nothing`, `the_paused_section_s_assumptions_leave_the_view_and_the_banner`, `the_assumptions_view_lists_each_assumption_with_its_rationale_and_source` and `a_sum_s_harmonic_set_marks_its_row_and_its_h_finds_it` open the Assumptions fold where they clicked its tab; where they asserted `inputs_view`, they assert the fold open or shut by its note (`ASSUMPTIONS_NOTE`) and the Key design heading. `a_focused_advanced_input_opens_its_group_and_heading_and_clears_the_filter` asserts the Assumptions fold stays shut. `a_leaf_assumption_term_shows_its_row_in_the_assumptions_view` starts with the folds shut (the fold opens, its rationale is drawn, the framed row is whole in the window after thirty frames) and then repeats with the Detailed inputs open (two framed rows, one in the window). `the_panel_works_inside_an_egui_window` asserts the Detailed inputs stay shut, so its one framed row is the Assumptions fold's.
- **Five tests read the column by position or count.** `clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group` takes each label's lowest copy in the column (the Key design row; the size controls above bear two of the labels). `an_unreachable_target_shows_the_best_value_and_its_overshoot` counts the line twice (D-77). `only_the_rows_drawn_this_frame_count_as_design_fields` counts the six size controls with the Key design rows, and the six alone once the group is shut. `the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design` takes the first row after the six. `the_key_design_group_shows_auto_s_status_and_a_button_to_the_size_tab` looks for the group's own line by `auto_status` (the column's Auto row starts with the same words).
- **Three tests counted a text the size controls draw too**, and would have held only because the harness had scrolled those controls out of view (the plan's review, RC-3). `size_and_torque_has_direct_dimension_and_gap_controls` counts the centre's tab, the copy right of the column, before it clicks it. `the_solve_waits_for_the_debounce_and_never_runs_per_frame` and `leaving_torque_to_magnets_while_solving_writes_the_current_solution` count the Key design group's "Solving..." with `count_in_fold`, the copies under the fold's header (decision D-77: the size controls draw Auto's outcome too). With the harness's wheel removed all three pass.
- **Three small-window tests** (`the_equation_panel_draws_in_a_tiny_window`, `no_centre_view_paints_or_takes_clicks_in_the_equation_panel`, `a_long_trail_shows_its_last_crumbs_and_draws_in_a_short_window`) build a new panel, whose small windows show no fold's header to click, and open the Detailed inputs as a fix link or a leaf term does (`explorer.focus_input(FACE_GAP)`), so the Key design rows are drawn in each window as the inputs side drew them at `main`; each asserts the fold's rows were laid out (`key_widgets`). The third sets the Geometry tab itself.
- **Two tests of fix links.** `an_outside_diameter_fix_opens_the_size_tab_and_edits_nothing` runs at 1280 x 800 with the column scrolled into the Detailed inputs, and adds that the slider is out of view before the link and in view after; every other assertion stays. `a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing` changes its comment only.

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` after this plan has merged, create the branch `magcoupling/overhaul-b2-part2b-ii-c` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: `test result: ok. 1113 passed; 0 failed; 6 ignored` and `all 81 checks hold`. Another count means `main` is not the base the blocks were checked on: stop and report.

- [ ] **Step 3: Check the base.** `git log --oneline f1f1b90..main -- magcoupling-rs .claude/workflows/gui-smoke.js linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs docs/ai docs/analyses/2026-11-10-conservative-assumptions.md docs/analyses/2026-11-18-owner-questions.md docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` lists this plan's commit (it touches `docs/ai/backlog.yaml`) and nothing else. If it lists more, stop: the controller regenerates the blocks on that `main` with the planner's tools (gitignored, in the main checkout's `.superpowers/sdd/2026-11-25-magcoupler-overhaul-b2p2biic/`: `gen_blocks.py`, `frags.py`, `replay.py`, `edit_lib.py` with the edit scripts `e1*.py` to `e5.py` (a record of most of what built each task's state: a few fixes were made by hand between them, so the bundle's commits are the states), `build_s3r.py` (Task 3's red state), `file_bl096.py`, and `seq.bundle`, the prototype's commits: `s0` the base, `s1` and `s2` the first two tasks, `s3r` and `s3` Task 3's red and green states, `s4` the smoke, `s4b` the backlog items filed, `s5` the docs; and the fix wave after the plan's review: `fw.py` rebuilds every state from the bundle's branch `before-fixwave`, the states the plan was first made from, with the edit scripts `f1.py`, `f3.py`, `f4b.py` and `f5.py`, and `plan_fix.py` holds its edits to this text) before Task 1.

## Tasks

### Task 1: The Detailed inputs fold scrolls with the Size column (BL-091)

**Model:** `sonnet` (BL-091 and BL-096 are `risk: mechanical`; the task transcribes exact blocks).

`design_inputs_ui` and `filtered_inputs_ui` draw no scroll area of their own, as `assumptions_ui` since BL-090: inside the Size column's fold they scroll with the column, and a row asked for (a leaf term, a fix link) scrolls into view through the column's one scroll area. Until Task 3 the inputs side of the other tabs still draws the design inputs, so it wraps them in the scroll area they lost (the order toggle and the filter box scroll with the rows there for two tasks).

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs`

**Interfaces:**
- Changes: `design_inputs_ui` and `filtered_inputs_ui` (private) draw no `ScrollArea`; `inputs_ui` wraps the design inputs in `magcoupling_inputs_scroll`.

- [ ] **Step 1 (red): the fold's two tests.** The block adds `the_detailed_inputs_fold_scrolls_with_the_size_column_to_its_last_group`: at 1024 x 700, the fold opened, a wheel over the column brings the last group's heading into the window, and from there a row asked for (the face gap) scrolls back into view. And `the_filtered_inputs_scroll_with_the_size_column_to_their_last_row`, for the filtered view's own scroll area: the same fold with a filter that lists every input (`.`), where a wheel over the column brings the list's last row into the window (two inputs bear its label, both at the list's end).

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    #[test]
    fn size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter() {
```

with:

```rust
    #[test]
    fn the_detailed_inputs_fold_scrolls_with_the_size_column_to_its_last_group() {
        // BL-091 (UXQ-2 of the 2026-11-23 review): the fold drew the design inputs' own scroll
        // area inside the column's, a short pane the column could not scroll past. Opened at
        // 1024x700, a wheel over the column brings the last group's heading into the window,
        // and a row asked for (a leaf term, a fix link) still scrolls into view, through the
        // column's own scroll area.
        let size = egui::vec2(1024.0, 700.0);
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        let mut harness = size_tab(size, DesignInputs::default());
        let output = harness.frame(Vec::new());
        let header = text_rects(&output, DETAILED_INPUTS)
            .into_iter()
            .find(|rect| rect.right() <= INPUTS_WIDTH)
            .expect("the fold's header in the Size column");
        harness.click(header.center());
        let last = InputCatalogue::get()
            .workflow
            .last()
            .expect("a workflow group")
            .label;
        let output = harness.frame(Vec::new());
        assert!(
            shown_whole(&output, last, screen).is_empty(),
            "the last group is in view before a scroll: the test would prove nothing"
        );
        harness.frame(vec![egui::Event::PointerMoved(header.center())]);
        harness.frame(vec![egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, -20000.0),
            modifiers: egui::Modifiers::NONE,
        }]);
        // egui spreads a wheel step over several frames.
        let mut output = harness.frame(Vec::new());
        for _ in 0..20 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(
            shown_whole(&output, last, screen).len(),
            1,
            "the last group's heading is in the window"
        );
        // From there a row asked for near the fold's top (the face gap, a Key design row)
        // scrolls back into view.
        let gap = label_for(FACE_GAP);
        assert!(
            shown_whole(&output, gap, screen)
                .iter()
                .all(|rect| rect.left() > INPUTS_WIDTH),
            "the face gap's rows are scrolled out of the column"
        );
        // The pointer leaves: what is left of the wheel step would end the scroll to the row.
        harness.frame(vec![egui::Event::PointerGone]);
        harness.panel.explorer.focus_input(FACE_GAP);
        let mut output = harness.frame(Vec::new());
        for _ in 0..30 {
            output = harness.frame(Vec::new());
        }
        let focus = focus_rects(&output);
        assert!(
            shown_whole(&output, gap, screen)
                .iter()
                .any(|row| focus.iter().any(|f| f.contains_rect(*row))),
            "the face gap's framed row is in the window: {focus:?}"
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_filtered_inputs_scroll_with_the_size_column_to_their_last_row() {
        // BL-091's other view: the filter's matches drew in a scroll area of their own too, the
        // same short pane inside the column's. With the fold open at 1024x700 and a filter that
        // lists every input, a wheel over the column brings the last row into the window.
        let size = egui::vec2(1024.0, 700.0);
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        let mut harness = size_tab(size, DesignInputs::default());
        let output = harness.frame(Vec::new());
        let header = text_rects(&output, DETAILED_INPUTS)
            .into_iter()
            .find(|rect| rect.right() <= INPUTS_WIDTH)
            .expect("the fold's header in the Size column");
        harness.click(header.center());
        // Every input's path holds a dot: the filter lists them all.
        harness.panel.input_filter = ".".to_owned();
        let groups = InputCatalogue::get().groups_in(harness.panel.input_order);
        let matches = filter_inputs(groups, ".");
        assert_eq!(
            matches.iter().map(|run| run.entries.len()).sum::<usize>(),
            InputCatalogue::get().all().count()
        );
        let last = matches
            .last()
            .and_then(|run| run.entries.last())
            .expect("a match");
        // The last row's label, and how many rows bear it (two inputs do, both at the list's end).
        let last = label_for(&last.path);
        let copies = matches
            .iter()
            .flat_map(|run| &run.entries)
            .filter(|entry| label_for(&entry.path) == last)
            .count();
        let in_column = |output: &egui::FullOutput| {
            shown_whole(output, last, screen)
                .into_iter()
                .filter(|rect| rect.right() <= INPUTS_WIDTH)
                .count()
        };
        let output = harness.frame(Vec::new());
        assert_eq!(
            in_column(&output),
            0,
            "the last row is in view before a scroll: the test would prove nothing"
        );
        harness.frame(vec![egui::Event::PointerMoved(header.center())]);
        harness.frame(vec![egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, -200000.0),
            modifiers: egui::Modifiers::NONE,
        }]);
        // egui spreads a wheel step over several frames.
        let mut output = harness.frame(Vec::new());
        for _ in 0..60 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(
            in_column(&output),
            copies,
            "the filter's last row is in the window"
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter() {
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 12
  ```

  Expected: 2 failed: `the_detailed_inputs_fold_scrolls_with_the_size_column_to_its_last_group`, `the_filtered_inputs_scroll_with_the_size_column_to_their_last_row`. `the_detailed_inputs_fold_scrolls_with_the_size_column_to_its_last_group` fails at "the last group's heading is in the window", and `the_filtered_inputs_scroll_with_the_size_column_to_their_last_row` at "the filter's last row is in the window" (`left: 0`): the wheel moves the column to the end of its short content, and each view's nested pane keeps its rows.

- [ ] **Step 3 (green): no scroll area in the fold.**

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
                ui.separator();
                match self.inputs_view {
```

with:

```rust
                ui.separator();
                // Each view in a scroll area of the side's: neither draws its own (BL-091).
                match self.inputs_view {
```

Replace:

```rust
                match self.inputs_view {
                    InputsView::Design => self.design_inputs_ui(ui, readouts),
                    InputsView::Assumptions => {
```

with:

```rust
                match self.inputs_view {
                    InputsView::Design => {
                        egui::ScrollArea::vertical()
                            .id_salt("magcoupling_inputs_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| self.design_inputs_ui(ui, readouts));
                    }
                    InputsView::Assumptions => {
```

Replace:

```rust
    /// (decision O-1), each workflow group's advanced sections under its closed Advanced heading;
    /// while the filter holds more than blanks, the matching inputs alone.
    fn design_inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

with:

```rust
    /// (decision O-1), each workflow group's advanced sections under its closed Advanced heading;
    /// while the filter holds more than blanks, the matching inputs alone. No scroll area of its
    /// own (BL-091): the Size column's fold scrolls with the column, the inputs side wraps it in
    /// one.
    fn design_inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

Replace:

```rust
        };
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
```

with:

```rust
        };
        egui::CollapsingHeader::new(KEY_DESIGN_HEADING)
            .id_salt("key_design")
            .default_open(true)
            .open(open_key_design.then_some(true))
            .show(ui, |ui| {
```

Replace:

```rust
            .show(ui, |ui| {
                egui::CollapsingHeader::new(KEY_DESIGN_HEADING)
                    .id_salt("key_design")
                    .default_open(true)
                    .open(open_key_design.then_some(true))
                    .show(ui, |ui| {
                        self.sizing_ui(ui, readouts);
                        self.key_widgets.clear();
                        for entry in &catalogue.key_design {
                            let widget = self.input_row_ui(ui, entry, readouts);
                            self.key_widgets.push((entry.path.as_str(), widget));
                            self.design_widgets.push(widget);
                            hint_line(ui, &entry.path);
                        }
                    });
                for group in groups {
                    let (open_group, open_advanced) = match focused {
                        Some((name, advanced)) if name == group.name => (true, advanced),
                        _ => (false, false),
                    };
                    // A trace counts the rows it frames in each group, closed or open.
                    let traced = self.trace.as_ref().map_or(0, |trace| {
                        trace.count_in(
                            group
                                .sections
                                .iter()
                                .flat_map(|s| s.entries.iter())
                                .map(|e| e.path.as_str()),
                        )
                    });
                    let title = if traced > 0 {
                        format!("{} ({traced} traced)", group.label)
                    } else {
                        group.label.to_owned()
                    };
                    egui::CollapsingHeader::new(title)
                        .id_salt((salt, &group.name))
                        .default_open(false)
                        .open(open_group.then_some(true))
                        .show(ui, |ui| {
                            for section in group.plain_sections() {
                                self.section_ui(ui, group, section, readouts);
                            }
                            if group.has_advanced() {
                                egui::CollapsingHeader::new(ADVANCED_HEADING)
                                    .id_salt(("advanced", &group.name))
                                    .default_open(false)
                                    .open(open_advanced.then_some(true))
                                    .show(ui, |ui| {
                                        for section in group.advanced_sections() {
                                            self.section_ui(ui, group, section, readouts);
                                        }
                                    });
                            }
                        });
                }
```

with:

```rust
            .show(ui, |ui| {
                self.sizing_ui(ui, readouts);
                self.key_widgets.clear();
                for entry in &catalogue.key_design {
                    let widget = self.input_row_ui(ui, entry, readouts);
                    self.key_widgets.push((entry.path.as_str(), widget));
                    self.design_widgets.push(widget);
                    hint_line(ui, &entry.path);
                }
```

Replace:

```rust
                    hint_line(ui, &entry.path);
                }
            });
    }

    /// One section of a group: its heading ([`InputSection::heading_in`]: none for the group's
```

with:

```rust
                    hint_line(ui, &entry.path);
                }
            });
        for group in groups {
            let (open_group, open_advanced) = match focused {
                Some((name, advanced)) if name == group.name => (true, advanced),
                _ => (false, false),
            };
            // A trace counts the rows it frames in each group, closed or open.
            let traced = self.trace.as_ref().map_or(0, |trace| {
                trace.count_in(
                    group
                        .sections
                        .iter()
                        .flat_map(|s| s.entries.iter())
                        .map(|e| e.path.as_str()),
                )
            });
            let title = if traced > 0 {
                format!("{} ({traced} traced)", group.label)
            } else {
                group.label.to_owned()
            };
            egui::CollapsingHeader::new(title)
                .id_salt((salt, &group.name))
                .default_open(false)
                .open(open_group.then_some(true))
                .show(ui, |ui| {
                    for section in group.plain_sections() {
                        self.section_ui(ui, group, section, readouts);
                    }
                    if group.has_advanced() {
                        egui::CollapsingHeader::new(ADVANCED_HEADING)
                            .id_salt(("advanced", &group.name))
                            .default_open(false)
                            .open(open_advanced.then_some(true))
                            .show(ui, |ui| {
                                for section in group.advanced_sections() {
                                    self.section_ui(ui, group, section, readouts);
                                }
                            });
                    }
                });
        }
    }

    /// One section of a group: its heading ([`InputSection::heading_in`]: none for the group's
```

Replace:

```rust
        ui.weak(format!("{shown} of {total} inputs"));
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_filter_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for run in &matches {
                    ui.add_space(4.0);
                    ui.strong(run.heading());
                    for &entry in &run.entries {
                        let widget = self.input_row_ui(ui, entry, readouts);
                        self.design_widgets.push(widget);
                    }
                }
            });
    }
```

with:

```rust
        ui.weak(format!("{shown} of {total} inputs"));
        for run in &matches {
            ui.add_space(4.0);
            ui.strong(run.heading());
            for &entry in &run.entries {
                let widget = self.input_row_ui(ui, entry, readouts);
                self.design_widgets.push(widget);
            }
        }
    }
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1115 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text.

- [ ] **Step 5: Commit** the one file: `fix(magcoupler): the Detailed inputs fold scrolls with the Size column (BL-091, BL-096 T1)`.

### Task 2: Tab through the Size column selects nothing and sets no Minimum OD

**Model:** `sonnet`.

`size_controls::controls_ui` hands the Minimum OD's slider the value its box shows (the floor rounded to the step's decimals, as egui rounds it for the box), and writes the Minimum OD only when that slider reports a change. The keyboard's focus leaving the box then writes back the value the slider already holds: no change, no touch (decision D-73). The text the box shows does not change, and an idle frame still leaves the floor exact.

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs` (the test), `magcoupling-rs/src/gui/size_controls.rs`

**Interfaces:**
- Changes: none outside `controls_ui`.

- [ ] **Step 1 (red): the keyboard's test.** The block adds `tab_through_the_size_column_selects_nothing_and_sets_no_minimum_od`: on a new panel (the spreadsheet's design, whose floor is off its box's decimal) Tab passes all six sliders and their boxes with nothing selected, no Minimum OD set and no edit; then an arrow key on the Minimum OD's slider moves it one step from the value shown and selects its row.

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    #[test]
    fn the_size_column_draws_its_controls_and_two_folds_and_no_line_about_them() {
```

with:

```rust
    #[test]
    fn tab_through_the_size_column_selects_nothing_and_sets_no_minimum_od() {
        // BL-093 for every row of the Size column, the left column of every tab (UX overhaul
        // spec 4.3): the keyboard's focus passes each slider and its value box and selects
        // nothing. The Minimum OD's floor at the spreadsheet's design is off the one decimal
        // its box shows, and egui writes a box's text back when the focus leaves it: that
        // write is no touch and sets no Minimum OD. An arrow key on the slider still does.
        use crate::gui::size_controls::slider_minimum;
        let mut harness = Harness::first_view();
        let floor = slider_minimum(
            &dimensions(harness.panel.inputs()).expect("size controls"),
            None,
        );
        assert_ne!(
            floor,
            (floor * 10.0).round() / 10.0,
            "the floor is on its box's decimals: the test would prove nothing"
        );
        // The six sliders: the four controls, Minimum OD, then Required (size_column_ui).
        let sliders: Vec<egui::Id> = harness.panel.design_widgets[..6].to_vec();
        let mut passed = vec![false; sliders.len()];
        let mut presses = 0;
        // Until the focus has been on every slider, and two presses more: past the last
        // slider's value box.
        let mut after = 0;
        while after < 2 {
            harness.frame(key_tap(egui::Key::Tab));
            harness.frame(Vec::new());
            presses += 1;
            assert!(presses < 60, "Tab passes the Size column: {passed:?}");
            for (slider, passed) in sliders.iter().zip(&mut passed) {
                *passed |= harness.ctx.memory(|m| m.has_focus(*slider));
            }
            after += usize::from(passed.iter().all(|&passed| passed));
            assert_eq!(
                harness.panel.selection,
                Selection::Nothing,
                "press {presses}"
            );
        }
        assert_eq!(harness.panel.minimum_diameter_mm, None);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
        // The Minimum OD's slider, focused: an arrow key moves it from the value its box
        // shows, and that is a touch.
        let minimum = sliders[4];
        harness.ctx.memory_mut(|m| m.request_focus(minimum));
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowLeft));
        harness.frame(Vec::new());
        let shown = (floor * 10.0).round() / 10.0;
        let set = harness.panel.minimum_diameter_mm.expect("a Minimum OD");
        assert!((set - (shown - 0.1)).abs() < 1e-9, "{set} from {shown}");
        assert_eq!(
            harness.panel.selection,
            Selection::Input(crate::gui::plain_text::MINIMUM_OD_HINT_KEY)
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_size_column_draws_its_controls_and_two_folds_and_no_line_about_them() {
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 12
  ```

  Expected: 1 failed: `tab_through_the_size_column_selects_nothing_and_sets_no_minimum_od`. It fails at "press 11" with `Input("size_tab.minimum_od")` selected: the focus has just left the Minimum OD's box.

- [ ] **Step 3 (green): the slider is handed the value its box shows.**

**`magcoupling-rs/src/gui/size_controls.rs`**

Replace:

```rust
use crate::gui::input_ui::slider;
use crate::gui::inputs::{InputCatalogue, slider_range};
use crate::gui::plain_text::{
```

with:

```rust
use crate::gui::input_ui::slider;
use crate::gui::inputs::{InputCatalogue, slider_range, step_decimals};
use crate::gui::plain_text::{
```

Replace:

```rust
            let clicked = row_label(ui, row(MINIMUM_OD_HINT_KEY), None);
            let response = ui
```

with:

```rust
            let clicked = row_label(ui, row(MINIMUM_OD_HINT_KEY), None);
            // The slider is handed the value its box shows. egui writes a box's text back when
            // the keyboard's focus leaves it, so a floor off the box's decimals would count as
            // moved, and its row as touched, by a Tab that only passed (BL-093).
            let mut shown = egui::emath::round_to_decimals(minimum, step_decimals(range.step));
            let response = ui
```

Replace:

```rust
                .push_id("minimum_diameter_mm", |ui| {
                    ui.add(slider(&mut minimum, entry.meta, range))
                })
```

with:

```rust
                .push_id("minimum_diameter_mm", |ui| {
                    ui.add(slider(&mut shown, entry.meta, range))
                })
```

Replace:

```rust
            if response.changed() {
                output.minimum_diameter_mm = Some(minimum);
```

with:

```rust
            if response.changed() {
                minimum = shown;
                output.minimum_diameter_mm = Some(minimum);
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1116 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text.

- [ ] **Step 5: Commit** the two files: `fix(magcoupler): Tab through the Size column selects nothing and sets no Minimum OD (BL-096 T2)`.

### Task 3: The left column is the Size column on every tab; the nothing-lost test

**Model:** `sonnet`.

`inputs_ui` draws the Size column whatever the centre shows, over the trace's banner and Clear trace at the column's foot; `InputsView`, its tabs and the field `inputs_view` go; the two folds are drawn at the column's full width; a fix on the Outside diameter scrolls the column back to its top (`column_to_top`). The panel tests' harness opens the Detailed inputs fold where a test drives its rows (`Harness::on_screen`), shows the size controls again for the tests of them (`size_and_torque`), and 47 tests take the new path to what they read (the list is under "Files"). The design's nothing-lost table becomes one test.

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs`

**Interfaces:**
- Removes: `gui::panel::InputsView` (public in a private module, re-exported nowhere), the field `inputs_view`.
- Changes: `ASSUMPTIONS_FOLD` is the literal `"Assumptions"` (the text it had); `inputs_ui`, `size_column_ui` as above.
- Produces (tests): `Harness::fold(&'static str)`, `Harness::size_and_torque()`, `Harness::tab(CentreView)`, `in_left_column`, `column_rows`.

- [ ] **Step 1 (red): the column's tests and the harness's moves.** The blocks add the harness's new moves (`fold`, `size_and_torque`, `tab`, `click_fold`, `in_left_column`), not yet used by `Harness::on_screen`, and three tests: `the_left_column_is_the_size_column_on_every_tab`, `a_trace_s_banner_and_clear_trace_sit_at_the_left_column_s_foot_on_every_tab` and `nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home` with its table `TOP_BAR_ACTIONS`. No existing test changes in this step.

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
        screen: egui::Vec2,
    }
```

with:

```rust
        screen: egui::Vec2,
        /// The fold of the Size column [`Harness::fold`] opened last, if any.
        fold: Option<&'static str>,
    }
```

Replace:

```rust
                screen: size,
            };
```

with:

```rust
                screen: size,
                fold: None,
            };
```

Replace:

```rust

        pub(crate) fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
```

with:

```rust

        /// Shows the Size column's fold headed `fold` ([`DETAILED_INPUTS`] or
        /// [`ASSUMPTIONS_FOLD`]) as a person does: a click on the header of the fold this
        /// harness opened before closes that one, a click on `fold`'s header opens it, and a
        /// wheel over the column brings the header to just under the column's top edge, so the
        /// fold's rows are in view down to the window's foot (in a window taller than the
        /// column nothing scrolls). The pointer then leaves the panel. Returns the frame after.
        fn fold(&mut self, fold: &'static str) -> egui::FullOutput {
            match self.fold.replace(fold) {
                Some(open) if open == fold => return self.frame(Vec::new()),
                Some(open) => self.click_fold(open),
                None => {}
            }
            self.click_fold(fold);
            let output = self.frame(Vec::new());
            let header = in_left_column(&output, fold);
            let side = egui::Id::new("magcoupling_inputs");
            let column = egui::containers::panel::PanelState::load(&self.ctx, side)
                .expect("the left column is drawn")
                .rect;
            // A wheel step under 8 points is a trackpad's, which egui applies in its own frame
            // (a larger one is spread over many): the whole distance in one frame of them.
            let distance = (header.top() - column.top() - 8.0).max(0.0);
            let steps = (distance / 7.0).ceil().max(1.0);
            let step = egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, -distance / steps),
                modifiers: egui::Modifiers::NONE,
            };
            self.frame(vec![egui::Event::PointerMoved(header.center())]);
            self.frame(vec![step; steps as usize]);
            self.frame(vec![egui::Event::PointerGone]);
            self.frame(Vec::new())
        }

        /// Shows the Size and torque tab as a person arrives on it: a click on its tab (the one
        /// in the centre region: the left column's heading bears the same words), and the fold
        /// this harness opened closed again by a click on its header, so the column, short
        /// again, shows its controls from its top. Returns the frame after.
        fn size_and_torque(&mut self) -> egui::FullOutput {
            self.tab(CentreView::SizeAndTorque);
            if let Some(open) = self.fold.take() {
                self.click_fold(open);
            }
            self.frame(Vec::new())
        }

        /// A click on the tab of `view` in the centre region's tab row.
        fn tab(&mut self, view: CentreView) -> egui::FullOutput {
            let output = self.frame(Vec::new());
            let tab = text_rects(&output, view.label())
                .into_iter()
                .find(|rect| rect.left() > INPUTS_WIDTH)
                .unwrap_or_else(|| panic!("no tab {:?}", view.label()));
            self.click(tab.center())
        }

        /// A click on the header of the Size column's fold headed `fold`, then the frames egui
        /// takes to open or close it.
        fn click_fold(&mut self, fold: &str) {
            let output = self.frame(Vec::new());
            self.click(in_left_column(&output, fold).center());
            for _ in 0..8 {
                self.frame(Vec::new());
            }
        }

        pub(crate) fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
```

Replace:

```rust
                other => panic!("{path}: {other:?}"),
            }
        }
    }

    /// The left edge of the right column in `output`, less its margin: where the check list's
```

with:

```rust
                other => panic!("{path}: {other:?}"),
            }
        }
    }

    /// The first `text` drawn in the left column of `output`.
    fn in_left_column(output: &egui::FullOutput, text: &str) -> egui::Rect {
        text_rects(output, text)
            .into_iter()
            .find(|rect| rect.right() <= INPUTS_WIDTH)
            .unwrap_or_else(|| panic!("no {text:?} in the left column"))
    }

    /// The left edge of the right column in `output`, less its margin: where the check list's
```

Replace:

```rust
    }

    #[test]
    fn the_assumptions_fold_scrolls_with_the_size_column_to_its_last_assumption() {
```

with:

```rust
    }

    /// The left column's fixed rows in `output`, top to bottom: its heading, each Size row's
    /// label and the two folds' headers.
    fn column_rows(output: &egui::FullOutput) -> Vec<egui::Rect> {
        [size_controls::HEADING]
            .into_iter()
            .chain(size_controls::SIZE_ROWS.iter().map(|row| row.label))
            .chain([DETAILED_INPUTS, ASSUMPTIONS_FOLD])
            .map(|text| in_left_column(output, text))
            .collect()
    }

    #[test]
    fn the_left_column_is_the_size_column_on_every_tab() {
        // UX overhaul spec 4.3: "The left column no longer changes with the centre tab". On
        // every tab it is the Size column, its rows where the opening tab draws them: the
        // heading, each control's label, then the folds Detailed inputs and Assumptions, both
        // shut. The two tabs that swapped the side ("Design inputs", "Assumptions") are gone:
        // the input groups show only inside the Detailed inputs, the assumptions only inside
        // their fold, and a fold opened on one tab is open on the next.
        let side = |harness: &Harness| {
            egui::containers::panel::PanelState::load(
                &harness.ctx,
                egui::Id::new("magcoupling_inputs"),
            )
            .expect("the left column is drawn")
            .rect
        };
        // A tall window: every group of the open fold is in view.
        let mut harness = Harness::of(MagcouplingPanel::new(), TALL);
        let first = column_rows(&harness.frame(Vec::new()));
        assert!(
            first
                .windows(2)
                .all(|pair| pair[0].bottom() <= pair[1].top()),
            "top to bottom: {first:?}"
        );
        let width = side(&harness).width();
        assert_eq!(width, INPUTS_WIDTH);
        for view in CentreView::tabs(false) {
            let output = harness.tab(view);
            assert_eq!(harness.panel.centre, view);
            assert_eq!(column_rows(&output), first, "{}", view.label());
            for gone in ["Design inputs", KEY_DESIGN_HEADING, ASSUMPTIONS_NOTE] {
                assert_eq!(count(&output, gone), 0, "{gone:?} on {}", view.label());
            }
        }
        // The Detailed inputs, opened on the Results tab: the order toggle, the filter box, the
        // Key design group and every workflow group, under the fold's header; still open on
        // the Geometry tab; and the column as wide as it was (the fold has no indent).
        harness.tab(CentreView::Results);
        let output = harness.fold(DETAILED_INPUTS);
        let header = in_left_column(&output, DETAILED_INPUTS);
        let mut inside = vec![InputOrder::Workbook.label(), KEY_DESIGN_HEADING];
        inside.extend(
            InputCatalogue::get()
                .workflow
                .iter()
                .map(|group| group.label),
        );
        for view in [CentreView::Results, CentreView::Geometry] {
            let output = harness.tab(view);
            for text in &inside {
                let at = in_left_column(&output, text);
                assert!(at.top() >= header.bottom(), "{text:?} on {}", view.label());
            }
            assert_eq!(count(&output, ASSUMPTIONS_NOTE), 0, "{}", view.label());
            assert_eq!(side(&harness).width(), width, "{}", view.label());
        }
        // The Assumptions fold in its place: the note that heads it, and the same width.
        let output = harness.fold(ASSUMPTIONS_FOLD);
        assert_eq!(count(&output, ASSUMPTIONS_NOTE), 1);
        assert_eq!(count(&output, KEY_DESIGN_HEADING), 0);
        assert_eq!(side(&harness).width(), width);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn a_trace_s_banner_and_clear_trace_sit_at_the_left_column_s_foot_on_every_tab() {
        // Part 2b-ii-a's decision D-52 left the trace's banner and Clear trace at the foot of
        // the inputs side of the other tabs, and the opening tab with neither. With the column
        // the same on every tab (UX overhaul spec 4.3) both sit at its foot on every tab, the
        // opening one too: under the folds, whole inside the window at its smallest, moving no
        // row of the column (decision O-8), and still there once the selection moves on.
        use crate::gui::trace::{CLEAR_TRACE, TRACING};
        for size in [egui::vec2(1024.0, 700.0), SCREEN] {
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let mut harness = Harness::of(MagcouplingPanel::new(), size);
            let output = harness.frame(Vec::new());
            let rows = column_rows(&output);
            let tracing = |output: &egui::FullOutput| {
                drawn_texts(output)
                    .iter()
                    .any(|text| text.starts_with(TRACING))
            };
            assert!(!tracing(&output), "no trace, no banner");
            assert_eq!(count(&output, CLEAR_TRACE), 0);
            // The gap's row of the Size column, then its entry's trace link.
            harness.trace_input(size_controls::GAP);
            let banner = harness.panel.trace.clone().expect("a trace").banner(false);
            for view in CentreView::tabs(false) {
                harness.tab(view);
                let output = harness.frame(Vec::new());
                let what = format!("{size:?}, {}", view.label());
                let at = shown_whole(&output, &banner, screen);
                let clear = shown_whole(&output, CLEAR_TRACE, screen);
                assert_eq!((at.len(), clear.len()), (1, 1), "{what}");
                assert!(at[0].center().x < INPUTS_WIDTH, "{what}: {at:?}");
                let folds = rows.last().expect("the Assumptions fold's header");
                assert!(
                    folds.bottom() <= at[0].top() && at[0].bottom() <= clear[0].top(),
                    "{what}: {folds:?}, {at:?}, {clear:?}"
                );
                assert_eq!(column_rows(&output), rows, "{what}: a trace moves no row");
            }
            // On the opening tab, with another thing selected: the banner and its button stay.
            harness.size_and_torque();
            let output = harness.select(GEARBOX_CHECK);
            assert_eq!(count(&output, &banner), 1, "{size:?}");
            harness.click_text(CLEAR_TRACE);
            assert_eq!(harness.panel.trace, None, "{size:?}");
            let output = harness.frame(Vec::new());
            assert!(!tracing(&output), "{size:?}");
            assert_eq!(count(&output, CLEAR_TRACE), 0, "{size:?}");
            assert_eq!(harness.panel.design(), Design::default());
            assert_eq!(harness.panel.history.undo_len(), 0);
        }
    }

    /// Where an action of the top bar before the overhaul lives now (UX overhaul spec 4.2).
    #[derive(Clone, Copy, Debug)]
    enum Home {
        /// A word of the top bar.
        Bar(&'static str),
        /// An item of the Design menu.
        Menu(&'static str),
        /// The "Equation" link of an Explain entry (decision C-11).
        EquationLink,
        /// The row under the bar while an assumption differs from its default.
        AssumptionsRow(&'static str),
    }

    /// Every action of the top bar before the overhaul, by its words then (`header_ui` at
    /// 1048962, the design's 4.2 table; "Open the example design" joined it in slice A), and
    /// its home.
    const TOP_BAR_ACTIONS: [(&str, Home); 11] = [
        ("Undo", Home::Bar(UNDO)),
        ("Redo", Home::Bar(REDO)),
        ("Reset all", Home::Menu(RESET_TO_START)),
        ("Open the example design", Home::Menu(OPEN_EXAMPLE)),
        ("Save design", Home::Menu(SAVE_DESIGN)),
        ("Load design", Home::Menu(LOAD_DESIGN)),
        ("Copy share link", Home::Bar(SHARE)),
        ("Export spreadsheet", Home::Bar(EXPORT)),
        ("Equation panel", Home::EquationLink),
        ("Show spreadsheet details", Home::Menu(SHOW_DETAILS)),
        (
            "Reset assumptions to defaults",
            Home::AssumptionsRow(RESET_ASSUMPTIONS),
        ),
    ];

    #[test]
    fn nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home() {
        // UX overhaul spec 8.2, test 9 (goal G-4), table by table, with the shaft-clamp
        // section shown and paused: every action of the old top bar is on the bar, in the
        // Design menu, behind an entry's Equation link or on the assumptions row; every centre
        // view has its tab; every shown note is in the Learn list; every dashboard row is a
        // Headline row of the results table, and a tile or a number of its check's entry where
        // it had a check; every input is a row of the Detailed inputs, under the six Size
        // controls; every assumption is in the Assumptions fold. The inputs and the assumptions
        // are looked for from every tab: the left column is the same on each (spec 4.3). The
        // panel is the app's (its baseline is not the spreadsheet's design, so the menu holds
        // the reset and the example design both), in a window that draws every input's row.
        use crate::gui::dashboard::{DASHBOARD, dashboard_lines};
        use crate::gui::explain::{EQUATION, shown_notes};
        use crate::gui::guidance::counted_guides;
        use crate::gui::plain_text::display_label;
        use crate::gui::result_groups::result_groups;
        use crate::gui::results_table::table_entries;
        use crate::gui::tiles::Tile;
        use crate::gui::typeset::glyph_safe;
        let window = egui::vec2(1280.0, 20000.0);
        // The texts `output` draws in the left column.
        let left_texts = |output: &egui::FullOutput| -> std::collections::HashSet<String> {
            flat_shapes(output)
                .into_iter()
                .filter_map(|shape| match shape {
                    egui::Shape::Text(text)
                        if text.visual_bounding_rect().center().x < INPUTS_WIDTH =>
                    {
                        Some(text.galley.text().to_owned())
                    }
                    _ => None,
                })
                .collect()
        };
        for clamps_paused in [false, true] {
            let panel = MagcouplingPanel::with_baseline(passing_design());
            let mut harness = Harness::of(panel, window);
            harness.panel.clamps_paused = clamps_paused;
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            // In the right column (a wide note widens it, so its edge is read off each frame).
            let in_explain = |output: &egui::FullOutput, text: &str| {
                let right = right_column_left(output);
                text_rects(output, text)
                    .iter()
                    .any(|rect| rect.left() > right)
            };

            // 1. The top bar's actions.
            let bar = text_rect(&output, HEADING).expect("the title");
            harness.click_text(DESIGN_MENU);
            let menu = harness.frame(Vec::new());
            harness.frame(key_tap(egui::Key::Escape));
            for (was, home) in TOP_BAR_ACTIONS {
                match home {
                    Home::Bar(text) => {
                        let at = text_rect(&output, text)
                            .unwrap_or_else(|| panic!("{was}: no {text:?}"));
                        assert!(
                            (at.center().y - bar.center().y).abs() < 20.0,
                            "{was}: {text:?} is on the bar"
                        );
                    }
                    Home::Menu(text) => {
                        assert_eq!(count(&output, text), 0, "{was}: shut in the menu");
                        assert_eq!(count(&menu, text), 1, "{was}: {text:?} in the menu");
                    }
                    Home::EquationLink => {
                        harness.select(HOT_MIN_CHECK);
                        harness.click_text(EQUATION);
                        assert!(harness.panel.explorer.open, "{was}");
                        harness.panel.explorer.close();
                        harness.panel.selection = Selection::Nothing;
                    }
                    Home::AssumptionsRow(text) => {
                        assert_eq!(count(&output, text), 0, "{was}: no assumption differs");
                        harness.panel.inputs.coupling.c_end = 0.2;
                        harness.frame(Vec::new());
                        let differs = harness.frame(Vec::new());
                        assert_eq!(count(&differs, text), 1, "{was}: {text:?} under the bar");
                        harness.panel.inputs = passing_design();
                        harness.frame(Vec::new());
                    }
                }
            }

            // 2. Every centre view has its tab; the paused section's has none.
            for view in CentreView::ALL {
                if clamps_paused && view == CentreView::Clamp {
                    let output = harness.frame(Vec::new());
                    assert_eq!(count(&output, view.label()), 0, "the paused tab");
                    continue;
                }
                harness.tab(view);
                assert_eq!(harness.panel.centre, view, "{}", view.label());
            }

            // 3. Every shown note is in the Learn list, and its title shows it.
            let notes: Vec<_> = shown_notes(clamps_paused).collect();
            assert!(notes.len() > 10, "{}", notes.len());
            for note in notes {
                harness.click_text(LEARN);
                assert_eq!(harness.panel.selection, Selection::NoteList);
                let output = harness.frame(Vec::new());
                let title = glyph_safe(note.title);
                let right = right_column_left(&output);
                let at = text_rects(&output, &title)
                    .into_iter()
                    .find(|rect| rect.left() > right)
                    .unwrap_or_else(|| panic!("{title:?} in the Learn list"));
                harness.click(at.center());
                assert_eq!(harness.panel.selection, Selection::Note(note.id));
            }

            // 4. Every dashboard row: a Headline row of the results table, drawn under its
            // name; its tile where it has one; a number of its check's entry where it has a
            // check, and that check a row of the check list.
            let headline: Vec<&str> = result_groups()[0]
                .rows
                .iter()
                .map(|&row| table_entries()[row].path.as_str())
                .collect();
            harness.tab(CentreView::Results);
            let table = harness.frame(Vec::new());
            harness.tab(CentreView::SizeAndTorque);
            let tiles = harness.frame(Vec::new());
            let centre = |output: &egui::FullOutput, text: &str| {
                let right = right_column_left(output);
                text_rects(output, text)
                    .iter()
                    .any(|rect| rect.left() > INPUTS_WIDTH && rect.right() < right)
            };
            let lines = dashboard_lines(harness.panel.results());
            let mut homes = [0; 3];
            for (line, (path, check)) in lines.iter().zip(DASHBOARD) {
                if clamps::hides(path, clamps_paused) {
                    continue;
                }
                assert!(headline.contains(&path), "{path}: a Headline row");
                let name = display_label(path, result_info(path).unwrap().meta.label);
                assert!(centre(&table, name), "{path}: {name:?} on the Results tab");
                homes[0] += 1;
                if let Some(tile) = Tile::ALL.into_iter().find(|tile| tile.path() == path) {
                    let label = tile.label(harness.panel.results());
                    assert!(centre(&tiles, &label), "{path}: its tile, {label:?}");
                    homes[1] += 1;
                }
                if let Some(check) = check {
                    let guide = counted_guides(clamps_paused)
                        .find(|guide| guide.check == check)
                        .unwrap_or_else(|| panic!("{path}: {check} is no row of the check list"));
                    let entry = harness.select(check);
                    assert!(in_explain(&entry, guide.label), "{path}: {check}'s entry");
                    if path != check {
                        let value = line.shown_value();
                        assert!(in_explain(&entry, &value), "{path}: {value:?} in the entry");
                    }
                    homes[2] += 1;
                }
            }
            // The paused section takes its one row; four rows are tiles; most have a check.
            let rows = DASHBOARD.len() - usize::from(clamps_paused);
            assert_eq!(homes[..2], [rows, Tile::ALL.len()], "{homes:?}");
            assert!(homes[2] >= 10, "{homes:?}");
            harness.panel.selection = Selection::Nothing;

            // 5. Every input: a row of the Detailed inputs (the filter "." matches every
            // path), under the six Size controls, from every tab.
            let catalogue = InputCatalogue::shown(clamps_paused);
            let total = catalogue.all().count();
            harness.fold(DETAILED_INPUTS);
            harness.panel.input_filter = ".".to_owned();
            for view in CentreView::tabs(clamps_paused) {
                harness.tab(view);
                let column = left_texts(&harness.frame(Vec::new()));
                let tab = view.label();
                for row in size_controls::SIZE_ROWS {
                    assert!(column.contains(row.label), "{:?} on {tab}", row.label);
                }
                let shown = format!("{total} of {total} inputs");
                assert!(column.contains(&shown), "{shown:?} on {tab}");
                for entry in catalogue.all() {
                    let label = label_for(&entry.path);
                    assert!(column.contains(label), "{}: {label:?} on {tab}", entry.path);
                }
            }
            harness.panel.input_filter.clear();

            // 6. Every assumption shown, with its rationale, in the Assumptions fold, from
            // every tab (the opening one among them).
            harness.fold(ASSUMPTIONS_FOLD);
            for view in CentreView::tabs(clamps_paused) {
                harness.tab(view);
                let column = left_texts(&harness.frame(Vec::new()));
                let tab = view.label();
                assert!(column.contains(ASSUMPTIONS_NOTE), "the fold on {tab}");
                for assumption in ASSUMPTIONS {
                    let shown = assumption.shown(clamps_paused);
                    for text in [assumption.label, assumption.rationale] {
                        assert_eq!(column.contains(text), shown, "{text:?} on {tab}");
                    }
                }
            }
            assert_eq!(harness.panel.inputs(), &passing_design());
        }
    }

    #[test]
    fn the_assumptions_fold_scrolls_with_the_size_column_to_its_last_assumption() {
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 14
  ```

  Expected: 3 failed: `a_trace_s_banner_and_clear_trace_sit_at_the_left_column_s_foot_on_every_tab`, `nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home`, `the_left_column_is_the_size_column_on_every_tab`. Each fails on the left column of another tab or on the opening tab's missing banner. `a_trace_s_banner_and_clear_trace_sit_at_the_left_column_s_foot_on_every_tab` finds no banner on the opening tab: `[1024.0 700.0], Size and torque`, `left: (0, 0)`. `nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home` finds no Size control on the Geometry tab: `"Minimum OD" on Geometry`. `the_left_column_is_the_size_column_on_every_tab` finds no heading there: `no "Size and torque" in the left column`. Every other test passes.

- [ ] **Step 3 (green): one column, and the tests that read it.** The blocks change the column (`InputsView` out, `inputs_ui`, `column_to_top`, the folds unindented, the comments), make `Harness::on_screen` open the Detailed inputs fold, and move the 47 tests listed under "Files" to the path that reaches what they read. Apply them all before the next step: between blocks the file does not build.

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
//!
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the check list over the Explain
//! panel on the right ([`crate::gui::checks`], [`crate::gui::explain`]: UX overhaul spec 4.5) and
```

with:

```rust
//!
//! Layout (spec M4 "Layout"): a header line, the Size column on the left, the same on every tab
//! (UX overhaul spec 4.3: the size controls over two folds, the detailed inputs, which are the
//! Key design group, then every input by group, [`crate::gui::inputs`], and the assumptions),
//! the check list over the Explain
//! panel on the right ([`crate::gui::checks`], [`crate::gui::explain`]: UX overhaul spec 4.5) and
```

Replace:

```rust
/// The Size column's fold of every design input (UX overhaul spec 4.3), over its fold of the
/// model's assumptions, which bears the Assumptions view's name.
pub const DETAILED_INPUTS: &str = "Detailed inputs";
```

with:

```rust
/// The Size column's fold of every design input (UX overhaul spec 4.3), over its fold of the
/// model's assumptions (the Assumptions view of decision M43-7, whose name it keeps). The two
/// folds are the only home of the input groups and of the assumptions, on every tab.
pub const DETAILED_INPUTS: &str = "Detailed inputs";
```

Replace:

```rust
pub const DETAILED_INPUTS: &str = "Detailed inputs";
pub const ASSUMPTIONS_FOLD: &str = InputsView::Assumptions.label();

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

```

with:

```rust
pub const DETAILED_INPUTS: &str = "Detailed inputs";
pub const ASSUMPTIONS_FOLD: &str = "Assumptions";

```

Replace:

```rust
    show_details: bool,
    /// What the inputs side shows.
    inputs_view: InputsView,
    /// How the design inputs are ordered (decision O-1): a view of this session only, never in
```

with:

```rust
    show_details: bool,
    /// How the design inputs are ordered (decision O-1): a view of this session only, never in
```

Replace:

```rust
    typing: bool,
}
```

with:

```rust
    typing: bool,
    /// Whether the left column scrolls back to its top in the next frame: a fix on a control
    /// of the Size column (the Outside diameter) shows its slider, wherever the column was
    /// scrolled to. View state.
    column_to_top: bool,
}
```

Replace:

```rust
            show_details: false,
            inputs_view: InputsView::Design,
            input_order: InputOrder::default(),
```

with:

```rust
            show_details: false,
            input_order: InputOrder::default(),
```

Replace:

```rust
            typing: false,
        }
```

with:

```rust
            typing: false,
            column_to_top: false,
        }
```

Replace:

```rust
        }
        // A fix on an input the Size tab names (the Outside diameter, contract C-4) opens that
        // tab, where its slider is, instead of the input's row.
        if events
```

with:

```rust
        }
        // A fix on an input the Size column names (the Outside diameter, contract C-4) shows
        // that control instead of the input's row: the column, the same on every tab, scrolls
        // back to its top, where the slider is. It still opens the Size and torque tab, as it
        // did while only that tab had the slider.
        if events
```

Replace:

```rust
            self.centre = CentreView::SizeAndTorque;
        }
```

with:

```rust
            self.centre = CentreView::SizeAndTorque;
            self.column_to_top = true;
        }
```

Replace:

```rust

    /// The left side: the tabs of the design inputs and the assumptions, then the view chosen,
    /// with the trace's banner at its foot. A leaf term clicked in the Equation panel shows the
    /// view that holds its row.
    fn inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

with:

```rust

    /// The left side (UX overhaul spec 4.3), the same on every tab of the centre region: the
    /// Size column, with the trace's banner and its Clear trace at its foot while a trace is on.
    fn inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

Replace:

```rust
    fn inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        // This frame's rows only, whichever groups are open.
        self.design_widgets.clear();
```

with:

```rust
    fn inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        // This frame's rows only, whichever folds and groups are open.
        self.design_widgets.clear();
```

Replace:

```rust
        self.design_widgets.clear();
        if self.centre == CentreView::SizeAndTorque {
            // The column scrolls in a short window: its overflow would stretch the panel past
            // the window's foot, and the centre region and its drawing with it.
            egui::ScrollArea::vertical()
                .id_salt("magcoupling_size_column")
                .auto_shrink([false, true])
                .show(ui, |ui| self.size_column_ui(ui, readouts));
            return;
        }
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
        // The trace's banner at the foot of the side, laid out bottom up first so the view takes
        // the room above it in the same frame: a trace moves no row above it (decision O-8).
        let clamps_paused = self.clamps_paused;
```

with:

```rust
        self.design_widgets.clear();
        // The trace's banner at the foot of the side, laid out bottom up first so the column
        // takes the room above it in the same frame: a trace moves no row above it (decision O-8).
        let clamps_paused = self.clamps_paused;
```

Replace:

```rust
            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                ui.separator();
                // Each view in a scroll area of the side's: neither draws its own (BL-091).
                match self.inputs_view {
                    InputsView::Design => {
                        egui::ScrollArea::vertical()
                            .id_salt("magcoupling_inputs_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| self.design_inputs_ui(ui, readouts));
                    }
                    InputsView::Assumptions => {
                        egui::ScrollArea::vertical()
                            .id_salt("magcoupling_assumptions_scroll")
                            .auto_shrink([false, false])
                            .show(ui, |ui| self.assumptions_ui(ui, readouts));
                    }
                }
```

with:

```rust
            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                // The column scrolls in a short window: its overflow would stretch the panel
                // past the window's foot, and the centre region and its drawing with it.
                let mut column = egui::ScrollArea::vertical()
                    .id_salt("magcoupling_size_column")
                    .auto_shrink([false, true]);
                if std::mem::take(&mut self.column_to_top) {
                    column = column.vertical_scroll_offset(0.0);
                }
```

Replace:

```rust
                    column = column.vertical_scroll_offset(0.0);
                }
            });
        });
```

with:

```rust
                    column = column.vertical_scroll_offset(0.0);
                }
                column.show(ui, |ui| self.size_column_ui(ui, readouts));
            });
        });
```

Replace:

```rust

    /// The Size column of the opening tab (UX overhaul spec 4.3): its heading, the Required
    /// torque row, Auto and its outcome, the size controls, then two folds, both closed until a
    /// fix link or a term asks for a row in one: the detailed inputs and the assumptions. A
    /// control touched selects its row, whose lines show in the Explain panel.
    fn size_column_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

with:

```rust

    /// The Size column, the left column of every tab (UX overhaul spec 4.3): its heading, the
    /// Required torque row, Auto and its outcome, the size controls, then two folds, both closed
    /// until a fix link or a term asks for a row in one: the detailed inputs and the
    /// assumptions, each drawn at the column's full width (a fold's indent would widen the
    /// column by the rows' overhang when it opens). A control touched selects its row, whose
    /// lines show in the Explain panel.
    fn size_column_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

Replace:

```rust
            .open((assumption == Some(false)).then_some(true))
            .show(ui, |ui| self.design_inputs_ui(ui, readouts));
        egui::CollapsingHeader::new(ASSUMPTIONS_FOLD)
```

with:

```rust
            .open((assumption == Some(false)).then_some(true))
            .show_unindented(ui, |ui| self.design_inputs_ui(ui, readouts));
        egui::CollapsingHeader::new(ASSUMPTIONS_FOLD)
```

Replace:

```rust
            .open((assumption == Some(true)).then_some(true))
            .show(ui, |ui| self.assumptions_ui(ui, readouts));
    }
```

with:

```rust
            .open((assumption == Some(true)).then_some(true))
            .show_unindented(ui, |ui| self.assumptions_ui(ui, readouts));
    }
```

Replace:

```rust
    /// spreadsheet details (spec S2-5), its source. No scroll area of its own: the Size
    /// column's fold scrolls with the column, the inputs side wraps it in one.
    fn assumptions_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

with:

```rust
    /// spreadsheet details (spec S2-5), its source. No scroll area of its own: the Size
    /// column's fold scrolls with the column.
    fn assumptions_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

Replace:

```rust
    /// while the filter holds more than blanks, the matching inputs alone. No scroll area of its
    /// own (BL-091): the Size column's fold scrolls with the column, the inputs side wraps it in
    /// one.
    fn design_inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

with:

```rust
    /// while the filter holds more than blanks, the matching inputs alone. No scroll area of its
    /// own (BL-091): the Size column's fold scrolls with the column.
    fn design_inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

Replace:

```rust

        /// A harness on a screen of `size`, on the Geometry tab: most tests drive the whole
        /// inputs side (the Key design group, the groups, the filter, the Assumptions view),
        /// which the Size tab a new panel opens on folds under "Detailed inputs". The first
        /// view's own tests use [`Harness::first_view`].
        pub(crate) fn on_screen(size: egui::Vec2) -> Self {
```

with:

```rust

        /// A harness on a screen of `size`, on the Geometry tab, the Size column's Detailed
        /// inputs fold open at the column's top ([`Harness::fold`]): most tests drive the
        /// detailed inputs (the Key design group, the groups, the filter), which the left
        /// column, the same on every tab (UX overhaul spec 4.3), keeps under that closed fold.
        /// The first view's own tests use [`Harness::first_view`].
        pub(crate) fn on_screen(size: egui::Vec2) -> Self {
```

Replace:

```rust
            panel.centre = CentreView::Geometry;
            Self::of(panel, size)
        }
```

with:

```rust
            panel.centre = CentreView::Geometry;
            let mut harness = Self::of(panel, size);
            harness.fold(DETAILED_INPUTS);
            harness
        }
```

Replace:

```rust

        /// The Size tab's slider under `label` as drawn in the last frame (the Size tab must be
        /// shown): `size_controls::controls_ui` pushes its sliders first, in this order.
        fn size_widget(&self, label: &str) -> egui::Response {
```

with:

```rust

        /// The Size column's slider under `label` as drawn in the last frame (on any tab):
        /// `size_controls::controls_ui` pushes its sliders first, in this order.
        fn size_widget(&self, label: &str) -> egui::Response {
```

Replace:

```rust

    /// Asserts the locked free-variable row draws `value` in its value box: on the line
```

with:

```rust

    /// How many times `text` is drawn inside the Detailed inputs fold of `output`: in the left
    /// column, under the fold's header. The size controls above it draw Auto's outcome too
    /// (decision D-77), in view or scrolled out of it.
    fn count_in_fold(output: &egui::FullOutput, text: &str) -> usize {
        let header = in_left_column(output, DETAILED_INPUTS);
        text_rects(output, text)
            .into_iter()
            .filter(|rect| rect.right() <= INPUTS_WIDTH && rect.top() >= header.bottom())
            .count()
    }

    /// Asserts the locked free-variable row draws `value` in its value box: on the line
```

Replace:

```rust
        harness.panel.clamps_paused = true;
        harness.click_text(CentreView::SizeAndTorque.label());
        for open in [true, false] {
```

with:

```rust
        harness.panel.clamps_paused = true;
        harness.size_and_torque();
        for open in [true, false] {
```

Replace:

```rust
        let before = harness.number("coupling.gear_ratio");
        let slider = harness.panel.design_widgets[0];
        harness.ctx.memory_mut(|m| m.request_focus(slider));
```

with:

```rust
        let before = harness.number("coupling.gear_ratio");
        // The first row of the fold: the Size column's controls come before it.
        let slider = harness.panel.design_widgets[size_controls::SIZE_ROWS.len()];
        harness.ctx.memory_mut(|m| m.request_focus(slider));
```

Replace:

```rust
        assert_eq!(harness.panel.input_filter, "");
        assert_eq!(harness.panel.inputs_view, InputsView::Design);
        let label = InputCatalogue::get()
```

with:

```rust
        assert_eq!(harness.panel.input_filter, "");
        assert_eq!(
            count(&output, ASSUMPTIONS_NOTE),
            0,
            "no assumption: the Assumptions fold stays closed"
        );
        let label = InputCatalogue::get()
```

Replace:

```rust
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        // An assumption's row, in the Assumptions view.
        let assumption = ASSUMPTIONS[0].paths[0];
```

with:

```rust
        assert_eq!(harness.panel.selection, Selection::Input(FACE_GAP));
        // An assumption's row, in the Assumptions fold.
        let assumption = ASSUMPTIONS[0].paths[0];
```

Replace:

```rust
        let assumption = ASSUMPTIONS[0].paths[0];
        harness.click_text(InputsView::Assumptions.label());
        harness.click_text(label_for(assumption));
```

with:

```rust
        let assumption = ASSUMPTIONS[0].paths[0];
        harness.fold(ASSUMPTIONS_FOLD);
        harness.click_text(label_for(assumption));
```

Replace:

```rust
        assert_eq!(count(&output, &trace.banner(false)), 1);
        // Each Key design row is framed exactly when the pull-out reads its input.
        let marks = trace_marks(&harness, &output);
```

with:

```rust
        assert_eq!(count(&output, &trace.banner(false)), 1);
        // Each Key design row is framed exactly when the pull-out reads its input (the last
        // of its label down the column: the Size column's rows of the gap and of the magnet
        // count, above the fold, bear the same labels and take no frame).
        let marks = trace_marks(&harness, &output);
```

Replace:

```rust
                .into_iter()
                .find(|r| r.left() < INPUTS_WIDTH)
                .unwrap();
```

with:

```rust
                .into_iter()
                .filter(|r| r.left() < INPUTS_WIDTH)
                .max_by(|a, b| a.top().total_cmp(&b.top()))
                .unwrap();
```

Replace:

```rust
        let mut harness = Harness::new();
        let key_rows = InputCatalogue::get().key_design.len();
```

with:

```rust
        let mut harness = Harness::new();
        // The Size column's controls are drawn on every tab; then the Key design rows.
        let size_rows = size_controls::SIZE_ROWS.len();
        let key_rows = InputCatalogue::get().key_design.len();
```

Replace:

```rust
        let key_rows = InputCatalogue::get().key_design.len();
        assert_eq!(harness.panel.design_widgets.len(), key_rows);
        // Every group closed: no row is drawn, so none counts (no stale ids, no growth).
        harness.click_text(KEY_DESIGN_HEADING);
```

with:

```rust
        let key_rows = InputCatalogue::get().key_design.len();
        assert_eq!(harness.panel.design_widgets.len(), size_rows + key_rows);
        // Every group closed: no row of the fold is drawn, so none counts (no stale ids, no
        // growth).
        harness.click_text(KEY_DESIGN_HEADING);
```

Replace:

```rust
        }
        assert!(harness.panel.design_widgets.is_empty());
    }
```

with:

```rust
        }
        assert_eq!(harness.panel.design_widgets.len(), size_rows);
    }
```

Replace:

```rust
            let mut texts = every_shown_text(&mut harness);
            for text in &texts {
```

with:

```rust
            let mut texts = every_shown_text(&mut harness);
            // Not vacuous for the opening tab's centre: its tiles' word is collected. The left
            // column's heading bears that tab's words, so the guard clicks the centre's tab.
            assert!(
                texts.iter().any(|t| t == crate::gui::tiles::NEEDS),
                "the Size and torque tab's tiles are not collected"
            );
            for text in &texts {
```

Replace:

```rust
        for view in CentreView::tabs(clamps_paused) {
            texts.extend(drawn_texts(&harness.click_text(view.label())));
            texts.extend(drawn_texts(&harness.frame(Vec::new())));
```

with:

```rust
        for view in CentreView::tabs(clamps_paused) {
            texts.extend(drawn_texts(&harness.tab(view)));
            texts.extend(drawn_texts(&harness.frame(Vec::new())));
```

Replace:

```rust
        // The Assumptions view: every rationale and source.
        harness.click_text(InputsView::Assumptions.label());
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
```

with:

```rust
        // The Assumptions view: every rationale and source.
        harness.fold(ASSUMPTIONS_FOLD);
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
```

Replace:

```rust
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.click_text(InputsView::Design.label());
        harness.panel.inputs = short_magnets();
```

with:

```rust
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.fold(DETAILED_INPUTS);
        harness.panel.inputs = short_magnets();
```

Replace:

```rust
            harness.panel.clamps_paused = clamps_paused;
            harness.click_text(InputsView::Assumptions.label());
            let texts = drawn_texts(&harness.frame(Vec::new()));
```

with:

```rust
            harness.panel.clamps_paused = clamps_paused;
            harness.fold(ASSUMPTIONS_FOLD);
            let texts = drawn_texts(&harness.frame(Vec::new()));
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Size and torque"), 1);
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
```

with:

```rust
        let output = harness.frame(Vec::new());
        // The centre's tab: the left column's heading bears the same words on every tab.
        let tabs = text_rects(&output, "Size and torque")
            .into_iter()
            .filter(|rect| rect.left() > INPUTS_WIDTH)
            .count();
        assert_eq!(tabs, 1);
        harness.size_and_torque();
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        harness.frame(Vec::new());
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
```

with:

```rust
        harness.frame(Vec::new());
        harness.size_and_torque();
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        let mut harness = Harness::new();
        harness.click_text("Size and torque");
        let before = harness.panel.design();
```

with:

```rust
        let mut harness = Harness::new();
        harness.size_and_torque();
        let before = harness.panel.design();
```

Replace:

```rust
        harness.panel.load_design_file(&text).unwrap();
        harness.click_text("Size and torque");
        for _ in 0..3 {
```

with:

```rust
        harness.panel.load_design_file(&text).unwrap();
        harness.size_and_torque();
        for _ in 0..3 {
```

Replace:

```rust
        .unwrap();
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
```

with:

```rust
        .unwrap();
        harness.size_and_torque();
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        harness.panel.inputs = with_cap(42.8, 41.82);
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
```

with:

```rust
        harness.panel.inputs = with_cap(42.8, 41.82);
        harness.size_and_torque();
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        harness.frame(Vec::new());
        harness.click_text("Size and torque");
        harness.frame(Vec::new());
```

with:

```rust
        harness.frame(Vec::new());
        harness.size_and_torque();
        harness.frame(Vec::new());
```

Replace:

```rust
        let mut harness = Harness::new();
        harness.click_text("Size and torque");
        harness.frame(Vec::new());
```

with:

```rust
        let mut harness = Harness::new();
        harness.size_and_torque();
        harness.frame(Vec::new());
```

Replace:

```rust
        assert!(end_effect_banner(harness.panel.results()).is_some());
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
```

with:

```rust
        assert!(end_effect_banner(harness.panel.results()).is_some());
        harness.size_and_torque();
        let output = harness.frame(Vec::new());
```

Replace:

```rust

    /// A harness on a screen of `screen` showing the Size tab for `inputs`, set as the panel's
    /// inputs: the tab clicked, then one more frame, so the inputs side draws its controls. The
    /// Hot-low tile is selected, so the Explain panel draws the torque card (decision C-10).
    fn size_tab(screen: egui::Vec2, inputs: DesignInputs) -> Harness {
```

with:

```rust

    /// A harness on a screen of `screen` showing the Size tab, the one a panel opens on, for
    /// `inputs`, set as the panel's inputs, then two more frames, so the Size column draws its
    /// controls for them. The Hot-low tile is selected, so the Explain panel draws the torque
    /// card (decision C-10).
    fn size_tab(screen: egui::Vec2, inputs: DesignInputs) -> Harness {
```

Replace:

```rust
    fn size_tab(screen: egui::Vec2, inputs: DesignInputs) -> Harness {
        let mut harness = Harness::on_screen(screen);
        harness.panel.selection = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
```

with:

```rust
    fn size_tab(screen: egui::Vec2, inputs: DesignInputs) -> Harness {
        let mut harness = Harness::of(MagcouplingPanel::new(), screen);
        harness.panel.selection = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
```

Replace:

```rust
        harness.frame(Vec::new());
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.frame(Vec::new());
```

with:

```rust
        harness.frame(Vec::new());
        harness.frame(Vec::new());
```

Replace:

```rust
            harness.panel.open_design(design.clone());
            harness.click_text(CentreView::SizeAndTorque.label());
            for _ in 0..10 {
```

with:

```rust
            harness.panel.open_design(design.clone());
            harness.size_and_torque();
            for _ in 0..10 {
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SOLVING), 1);
        asks_within_the_debounce(&output);
```

with:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count_in_fold(&output, SOLVING), 1);
        asks_within_the_debounce(&output);
```

Replace:

```rust
        harness.panel.inputs.metal.required_min_Nm = 3.0;
        assert_eq!(count(&harness.frame(Vec::new()), SOLVING), 1);
        harness.panel.set_sizing_mode(SizingMode::MagnetsToTorque);
```

with:

```rust
        harness.panel.inputs.metal.required_min_Nm = 3.0;
        assert_eq!(count_in_fold(&harness.frame(Vec::new()), SOLVING), 1);
        harness.panel.set_sizing_mode(SizingMode::MagnetsToTorque);
```

Replace:

```rust
        // Auto off on the Size tab; opening the tab is no edit.
        harness.click_text(CentreView::SizeAndTorque.label());
        assert_eq!(harness.panel.history.undo_len(), steps);
```

with:

```rust
        // Auto off on the Size tab; opening the tab is no edit.
        harness.size_and_torque();
        assert_eq!(harness.panel.history.undo_len(), steps);
```

Replace:

```rust
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, "Not reachable: best 9.937 N\u{b7}m at 50.80 mm"),
```

with:

```rust
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        // Twice in this tall window: Auto's outcome in the Size column, and the solver's
        // status in the Key design group under it.
        assert_eq!(
            count(&output, "Not reachable: best 9.937 N\u{b7}m at 50.80 mm"),
```

Replace:

```rust
            count(&output, "Not reachable: best 9.937 N\u{b7}m at 50.80 mm"),
            1
        );
```

with:

```rust
            count(&output, "Not reachable: best 9.937 N\u{b7}m at 50.80 mm"),
            2
        );
```

Replace:

```rust
        // The picker is the Size tab's Auto row; the Key design group then shows the result.
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
```

with:

```rust
        // The picker is the Size tab's Auto row; the Key design group then shows the result.
        harness.size_and_torque();
        harness.click_text(auto_choice(FreeVariable::AxialLength));
```

Replace:

```rust
        harness.click_text(CentreView::Geometry.label());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
```

with:

```rust
        harness.click_text(CentreView::Geometry.label());
        harness.fold(DETAILED_INPUTS);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
```

Replace:

```rust
        tooltips_at_once(&harness.ctx);
        harness.click_text(CentreView::SizeAndTorque.label());
        // Off: every size slider is the user's.
```

with:

```rust
        tooltips_at_once(&harness.ctx);
        harness.size_and_torque();
        // Off: every size slider is the user's.
```

Replace:

```rust
        use crate::gui::size_controls::{AUTO, GAP, change_text};
        let mut harness = Harness::new();
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
```

with:

```rust
        use crate::gui::size_controls::{AUTO, GAP, change_text};
        let mut harness = Harness::new();
        harness.size_and_torque();
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
```

Replace:

```rust
        let mut harness = Harness::new();
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
```

with:

```rust
        let mut harness = Harness::new();
        harness.size_and_torque();
        harness.click_text(AUTO);
```

Replace:

```rust
        harness.panel.inputs.metal.required_min_Nm = 50.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
```

with:

```rust
        harness.panel.inputs.metal.required_min_Nm = 50.0;
        harness.size_and_torque();
        harness.click_text(AUTO);
```

Replace:

```rust
        harness.panel.sizing.variable = FreeVariable::RingRadius;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
```

with:

```rust
        harness.panel.sizing.variable = FreeVariable::RingRadius;
        harness.size_and_torque();
        harness.click_text(AUTO);
```

Replace:

```rust
        harness.panel.inputs.metal.required_min_Nm = 0.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
```

with:

```rust
        harness.panel.inputs.metal.required_min_Nm = 0.0;
        harness.size_and_torque();
        harness.click_text(AUTO);
```

Replace:

```rust
        harness.panel.inputs.coupling.npole = 8;
        harness.click_text(CentreView::SizeAndTorque.label());
        // The requirement: the hot-low torque of an 8.5 mm ring of 8 magnets, a valid layout.
```

with:

```rust
        harness.panel.inputs.coupling.npole = 8;
        harness.size_and_torque();
        // The requirement: the hot-low torque of an 8.5 mm ring of 8 magnets, a valid layout.
```

Replace:

```rust
        harness.panel.inputs.metal.required_min_Nm = 50.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
```

with:

```rust
        harness.panel.inputs.metal.required_min_Nm = 50.0;
        harness.size_and_torque();
        harness.click_text(auto_choice(FreeVariable::AxialLength));
```

Replace:

```rust
        harness.panel.inputs.metal.required_min_Nm = 1.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
```

with:

```rust
        harness.panel.inputs.metal.required_min_Nm = 1.0;
        harness.size_and_torque();
        harness.click_text(auto_choice(FreeVariable::AxialLength));
```

Replace:

```rust
        let shown = harness.panel.shown_inputs();
        // The cap's rows, and two cap rows Auto does not set, on the inputs side of another tab.
        harness.click_text(CentreView::Geometry.label());
```

with:

```rust
        let shown = harness.panel.shown_inputs();
        // The cap's rows, and two cap rows Auto does not set, in the Detailed inputs, on
        // another tab.
        harness.click_text(CentreView::Geometry.label());
```

Replace:

```rust
        harness.click_text(CentreView::Geometry.label());
        harness.panel.input_filter = "metal.cap_".to_owned();
```

with:

```rust
        harness.click_text(CentreView::Geometry.label());
        harness.fold(DETAILED_INPUTS);
        harness.panel.input_filter = "metal.cap_".to_owned();
```

Replace:

```rust
        harness.panel.inputs.metal.required_min_Nm = 2.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
```

with:

```rust
        harness.panel.inputs.metal.required_min_Nm = 2.0;
        harness.size_and_torque();
        harness.click_text(auto_choice(FreeVariable::AxialLength));
```

Replace:

```rust
        harness.panel.inputs.metal.required_min_Nm = required;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
```

with:

```rust
        harness.panel.inputs.metal.required_min_Nm = required;
        harness.size_and_torque();
        harness.click_text(auto_choice(FreeVariable::AxialLength));
```

Replace:

```rust
        // Opening the Size tab switches no mode.
        harness.click_text(CentreView::SizeAndTorque.label());
        assert_eq!(harness.panel.design(), design);
```

with:

```rust
        // Opening the Size tab switches no mode.
        harness.size_and_torque();
        assert_eq!(harness.panel.design(), design);
```

Replace:

```rust
            .unwrap();
        linked.click_text(CentreView::SizeAndTorque.label());
        assert_eq!(linked.panel.design(), design);
```

with:

```rust
            .unwrap();
        linked.size_and_torque();
        assert_eq!(linked.panel.design(), design);
```

Replace:

```rust
    fn the_key_design_group_shows_auto_s_status_and_a_button_to_the_size_tab() {
        use crate::gui::size_controls::{AUTO_STATUS_START, OPEN_SIZE_TAB, auto_status};
        let mut harness = Harness::new();
```

with:

```rust
    fn the_key_design_group_shows_auto_s_status_and_a_button_to_the_size_tab() {
        use crate::gui::size_controls::{OPEN_SIZE_TAB, auto_status};
        let mut harness = Harness::new();
```

Replace:

```rust
            assert_eq!(count(&output, text), 0, "{text}");
        }
        assert!(
            !drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with(AUTO_STATUS_START)),
            "Auto is off"
        );
```

with:

```rust
            assert_eq!(count(&output, text), 0, "{text}");
        }
        // The group's own line, whichever size Auto sets (the Size column's Auto row starts
        // with the same words).
        assert!(
            FreeVariable::ALL
                .iter()
                .all(|&variable| count(&output, &auto_status(variable)) == 0),
            "Auto is off"
        );
```

Replace:

```rust
        // Opening the Size tab switches nothing.
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.frame(Vec::new());
```

with:

```rust
        // Opening the Size tab switches nothing.
        harness.size_and_torque();
        harness.frame(Vec::new());
```

Replace:

```rust
        harness.frame(Vec::new());
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
```

with:

```rust
        harness.frame(Vec::new());
        harness.size_and_torque();
        harness.click_text(AUTO);
```

Replace:

```rust
        all.extend(drawn_texts(&harness.frame(Vec::new())));
        // The Key design group's line, on another tab.
        harness.click_text(CentreView::Geometry.label());
```

with:

```rust
        all.extend(drawn_texts(&harness.frame(Vec::new())));
        // The Key design group's line, in the Detailed inputs, on another tab.
        harness.click_text(CentreView::Geometry.label());
```

Replace:

```rust
        harness.click_text(CentreView::Geometry.label());
        all.extend(drawn_texts(&harness.frame(Vec::new())));
        assert!(all.contains(&auto_status(FreeVariable::AxialLength)));
```

with:

```rust
        harness.click_text(CentreView::Geometry.label());
        all.extend(drawn_texts(&harness.fold(DETAILED_INPUTS)));
        assert!(all.contains(&auto_status(FreeVariable::AxialLength)));
```

Replace:

```rust
            egui::vec2(1280.0, 300.0),
        ] {
            let mut harness = Harness::on_screen(size);
            harness
                .panel
```

with:

```rust
            egui::vec2(1280.0, 300.0),
        ] {
            // These windows show no fold's header. A fix link or a leaf term opens the Detailed
            // inputs without it (here the face gap's), so their rows are drawn in each window.
            let mut harness = Harness::of(MagcouplingPanel::new(), size);
            harness
                .panel
```

Replace:

```rust
                .open_path("temperature.summary.governing_limit_C");
            harness.panel.selection = Selection::Note("br_temperature");
```

with:

```rust
                .open_path("temperature.summary.governing_limit_C");
            harness.panel.explorer.focus_input(FACE_GAP);
            harness.panel.selection = Selection::Note("br_temperature");
```

Replace:

```rust
                    harness.frame(Vec::new());
                }
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
    }
```

with:

```rust
                    harness.frame(Vec::new());
                }
            }
            assert!(!harness.panel.key_widgets.is_empty(), "the fold is open");
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
    }
```

Replace:

```rust
        ] {
            let mut harness = Harness::on_screen(size);
            harness.panel.explorer.open_path("model.pullout_Nm");
```

with:

```rust
        ] {
            // The shortest window shows no fold's header. A fix link or a leaf term opens the
            // Detailed inputs without it (here the face gap's), so their rows are drawn.
            let mut harness = Harness::of(MagcouplingPanel::new(), size);
            harness.panel.explorer.open_path("model.pullout_Nm");
```

Replace:

```rust
            let mut harness = Harness::of(MagcouplingPanel::new(), size);
            harness.panel.explorer.open_path("model.pullout_Nm");
            harness.frame(Vec::new());
            if dragged {
```

with:

```rust
            let mut harness = Harness::of(MagcouplingPanel::new(), size);
            harness.panel.explorer.open_path("model.pullout_Nm");
            harness.panel.explorer.focus_input(FACE_GAP);
            harness.frame(Vec::new());
            if dragged {
```

Replace:

```rust
                }
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
```

with:

```rust
                }
            }
            assert!(!harness.panel.key_widgets.is_empty(), "the fold is open");
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
```

Replace:

```rust
        for size in [SCREEN, egui::vec2(1280.0, 240.0), narrow] {
            let mut harness = Harness::on_screen(size);
            harness.panel.explorer.open_path(paths[0]);
```

with:

```rust
        for size in [SCREEN, egui::vec2(1280.0, 240.0), narrow] {
            // The short windows show no fold's header. A fix link or a leaf term opens the
            // Detailed inputs without it (here the face gap's), so their rows are drawn.
            let mut harness = Harness::of(MagcouplingPanel::new(), size);
            harness.panel.centre = CentreView::Geometry;
            harness.panel.explorer.open_path(paths[0]);
```

Replace:

```rust
            }
            assert_eq!(harness.panel.explorer.trail().len(), MAX_TRAIL);
```

with:

```rust
            }
            harness.panel.explorer.focus_input(FACE_GAP);
            assert_eq!(harness.panel.explorer.trail().len(), MAX_TRAIL);
```

Replace:

```rust
                output = harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
            if size != narrow {
```

with:

```rust
                output = harness.frame(Vec::new());
            }
            assert!(!harness.panel.key_widgets.is_empty(), "the fold is open");
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
            if size != narrow {
```

Replace:

```rust
            harness.panel.show_details = show_details;
            harness.click_text(InputsView::Assumptions.label());
            assert_eq!(harness.panel.inputs_view, InputsView::Assumptions);
            let output = harness.frame(Vec::new());
            let texts = drawn_texts(&output);
```

with:

```rust
            harness.panel.show_details = show_details;
            let output = harness.fold(ASSUMPTIONS_FOLD);
            assert_eq!(count(&output, ASSUMPTIONS_NOTE), 1, "the fold is open");
            assert_eq!(count(&output, KEY_DESIGN_HEADING), 0, "and the other shut");
            let texts = drawn_texts(&output);
```

Replace:

```rust
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
            harness.click_text(InputsView::Design.label());
            assert_eq!(harness.panel.inputs_view, InputsView::Design);
        }
```

with:

```rust
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
            let output = harness.fold(DETAILED_INPUTS);
            assert_eq!(count(&output, ASSUMPTIONS_NOTE), 0, "the fold is shut");
            assert_eq!(count(&output, KEY_DESIGN_HEADING), 1, "and the other open");
        }
```

Replace:

```rust
    fn a_leaf_assumption_term_shows_its_row_in_the_assumptions_view() {
        let mut harness = Harness::new();
```

with:

```rust
    fn a_leaf_assumption_term_shows_its_row_in_the_assumptions_view() {
        // The view is the Size column's Assumptions fold (UX overhaul spec 4.3), shut until
        // the term opens it: on the Geometry tab, as on every tab. The column then scrolls to
        // the row, over up to a third of a second.
        let rationale = ASSUMPTIONS
            .iter()
            .find(|a| a.id == "end_effect")
            .expect("the end-effect coefficient")
            .rationale;
        let mut harness = Harness::first_view();
        harness.click_text(CentreView::Geometry.label());
        harness.panel.explorer.open_path("model.f_end");
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, rationale), 0, "the fold starts shut");
        harness.click_text("End-effect coefficient");
        let mut output = harness.frame(Vec::new());
        for _ in 0..30 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(count(&output, rationale), 1, "the Assumptions fold is open");
        assert_eq!(
            count(&output, KEY_DESIGN_HEADING),
            0,
            "the Detailed inputs stay shut"
        );
        let focus = focus_rects(&output);
        // The assumption's heading and its row's label are both "End-effect coefficient";
        // the row's label sits inside the frame, in the window.
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN);
        let labels = shown_whole(&output, "End-effect coefficient", screen);
        assert!(
            labels
                .iter()
                .any(|l| l.left() < INPUTS_WIDTH && focus.iter().any(|r| r.contains_rect(*l))),
            "{labels:?} {focus:?}"
        );
        // With the Detailed inputs open (this harness's), the row shows in its input group
        // too, which opens: a framed row is in the window all the same.
        let mut harness = Harness::new();
```

Replace:

```rust
        harness.click_text("End-effect coefficient");
        let mut output = harness.frame(Vec::new());
        for _ in 0..5 {
            output = harness.frame(Vec::new());
        }
```

with:

```rust
        harness.click_text("End-effect coefficient");
        let mut output = harness.frame(Vec::new());
        for _ in 0..30 {
            output = harness.frame(Vec::new());
        }
```

Replace:

```rust
        }
        assert_eq!(harness.panel.inputs_view, InputsView::Assumptions);
        let focus = focus_rects(&output);
```

with:

```rust
        }
        let focus = focus_rects(&output);
```

Replace:

```rust
        let focus = focus_rects(&output);
        // The assumption's heading and its row's label are both "End-effect coefficient";
        // the row's label sits inside the frame.
        let labels = crate::gui::test_support::text_rects(&output, "End-effect coefficient");
        assert!(
```

with:

```rust
        let focus = focus_rects(&output);
        assert_eq!(
            focus.len(),
            2,
            "in its input group and in the Assumptions fold"
        );
        let labels = shown_whole(&output, "End-effect coefficient", screen);
        assert!(
```

Replace:

```rust
        harness.panel.explorer.open_path("model.tau_Pa");
        harness.click_text(InputsView::Assumptions.label());
        let output = harness.frame(Vec::new());
```

with:

```rust
        harness.panel.explorer.open_path("model.tau_Pa");
        harness.fold(ASSUMPTIONS_FOLD);
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        );
        // Back on the design inputs, the H under the Σ in the Equation panel.
        harness.click_text(InputsView::Design.label());
        assert_eq!(harness.panel.inputs_view, InputsView::Design);
        let output = harness.frame(Vec::new());
        let h = text_rects(&output, "H")
```

with:

```rust
        );
        // Back on the detailed inputs (the Assumptions fold shut again), the H under the Σ in
        // the Equation panel.
        let output = harness.fold(DETAILED_INPUTS);
        assert_eq!(count(&output, ASSUMPTIONS_NOTE), 0, "the fold is shut");
        let h = text_rects(&output, "H")
```

Replace:

```rust
        }
        assert_eq!(harness.panel.inputs_view, InputsView::Assumptions);
        let row = text_rects(&output, label)
```

with:

```rust
        }
        assert_eq!(count(&output, ASSUMPTIONS_NOTE), 1, "the H opens the fold");
        // The set's row in the Assumptions fold: the last one down the column (with the
        // Detailed inputs open its input group shows the row too, framed as well).
        let row = text_rects(&output, label)
```

Replace:

```rust
        let row = text_rects(&output, label)
            .into_iter()
            .find(|r| r.left() < INPUTS_WIDTH)
            .expect("the set's row on the inputs side");
        let focus = focus_rects(&output);
```

with:

```rust
        let row = text_rects(&output, label)
            .into_iter()
            .filter(|r| r.left() < INPUTS_WIDTH)
            .max_by(|a, b| a.top().total_cmp(&b.top()))
            .expect("the set's row on the inputs side");
        let focus = focus_rects(&output);
```

Replace:

```rust
        // selected and opened from its entry, a term followed, a leaf assumption term's row
        // shown (the Assumptions view
        // and its frame), then idle frames. The window keeps its size, the Equation panel
```

with:

```rust
        // selected and opened from its entry, a term followed, a leaf assumption term's row
        // shown (the Assumptions fold
        // and its frame), then idle frames. The window keeps its size, the Equation panel
```

Replace:

```rust
        let mut panel = MagcouplingPanel::new();
        // The Assumptions view is the inputs side of every view but Size and torque (which
        // folds the inputs under "Detailed inputs"; a new panel opens there). The Results tab's
        // Headline group lists the headline values.
        panel.centre = CentreView::Results;
```

with:

```rust
        let mut panel = MagcouplingPanel::new();
        // The Results tab's Headline group lists the headline values. The left column is the
        // Size column there as on every tab, its Assumptions fold shut until the term opens it.
        panel.centre = CentreView::Results;
```

Replace:

```rust
        }
        assert_eq!(panel.inputs_view, InputsView::Assumptions);
        assert_eq!(focus_rects(&output).len(), 1, "the row is framed");
```

with:

```rust
        }
        // The Assumptions fold's row: the Detailed inputs, shut, draw no other.
        assert_eq!(count(&output, KEY_DESIGN_HEADING), 0);
        assert_eq!(focus_rects(&output).len(), 1, "the row is framed");
```

Replace:

```rust
        // Every fix of the workbook example's red checks, each in its check's Explain entry,
        // from the Geometry view (the input's group or the Assumptions view opens) and from
        // Size and torque (its Detailed inputs open).
        let mut harness = Harness::on_screen(TALL);
```

with:

```rust
        // Every fix of the workbook example's red checks, each in its check's Explain entry,
        // from the Geometry tab and from Size and torque: the left column is the same on both,
        // and the input's group opens in its Detailed inputs (an assumption's fold too).
        let mut harness = Harness::on_screen(TALL);
```

Replace:

```rust
        use crate::gui::size_controls::DIAMETER;
        // Contract C-4: the ring radius's fix bears the Size tab's name, and its link opens
        // that tab, where the Outside diameter slider is, instead of a row. At 12 poles the
        // blocks do not fit, and the fit's first fix raises the Outside diameter.
        let inputs = with_poles(12);
```

with:

```rust
        use crate::gui::size_controls::DIAMETER;
        // Contract C-4: the ring radius's fix bears the Size column's name, and its link shows
        // the Outside diameter slider instead of a row: it opens the Size and torque tab, as
        // it did while only that tab had the slider, and the left column, scrolled down in its
        // Detailed inputs, scrolls back to the slider. At 12 poles the blocks do not fit, and
        // the fit's first fix raises the Outside diameter.
        let inputs = with_poles(12);
```

Replace:

```rust
        assert_eq!(text, format!("Raise {DIAMETER}"));
        let mut harness = Harness::on_screen(TALL);
        harness.panel.inputs = inputs.clone();
```

with:

```rust
        assert_eq!(text, format!("Raise {DIAMETER}"));
        let size = egui::vec2(1280.0, 800.0);
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        let mut harness = Harness::on_screen(size);
        harness.panel.inputs = inputs.clone();
```

Replace:

```rust
        harness.frame(Vec::new());
        harness.select(crate::gui::guidance::FIT_CHECK);
        let undo = harness.panel.history.undo_len();
```

with:

```rust
        harness.frame(Vec::new());
        let output = harness.select(crate::gui::guidance::FIT_CHECK);
        let slider = |output: &egui::FullOutput| {
            shown_whole(output, DIAMETER, screen)
                .into_iter()
                .any(|rect| rect.right() <= INPUTS_WIDTH)
        };
        assert!(
            !slider(&output),
            "the slider is in view before the link: the test would prove nothing"
        );
        let undo = harness.panel.history.undo_len();
```

Replace:

```rust
        assert!(!harness.panel.explorer.open);
        for _ in 0..5 {
```

with:

```rust
        assert!(!harness.panel.explorer.open);
        let mut output = harness.frame(Vec::new());
        for _ in 0..5 {
```

Replace:

```rust
        let mut output = harness.frame(Vec::new());
        for _ in 0..5 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.inputs(), &inputs);
        assert_eq!(harness.panel.history.undo_len(), undo);
```

with:

```rust
        let mut output = harness.frame(Vec::new());
        for _ in 0..5 {
            output = harness.frame(Vec::new());
        }
        assert!(slider(&output), "the column is back at the slider");
        assert_eq!(harness.panel.inputs(), &inputs);
        assert_eq!(harness.panel.history.undo_len(), undo);
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1119 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text.

- [ ] **Step 5: Commit** the one file: `feat(magcoupler): the left column is the Size column on every tab, a trace's banner at its foot; the nothing-lost test (BL-096 T3)`.

### Task 4: gui-smoke judges the left column on the Geometry tab

**Model:** `sonnet`.

On the Geometry tab the smoke prompt judges `same_column` (the left side shows the size controls and the two shut folds, and no "Design inputs" tab), then opens the Detailed inputs fold before its tooltip step, whose Key design row is inside it now. `app.rs`'s test holds the step's words to the app's (`SIZE_ROWS`, `DETAILED_INPUTS`, `ASSUMPTIONS_FOLD`); the verdict test's `CHECKS` gains `same_column`.

**Files:**
- Modify: `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`, `magcoupling-rs/src/app.rs`, `magcoupling-rs/src/gui/mod.rs`

**Interfaces:**
- Produces: `gui::SIZE_ROWS`, re-exported under `cfg(all(test, feature = "app"))` (the module `size_controls` is private to `gui`).
- Changes: the smoke agent's schema gains the required boolean `same_column`, which `CALCULATOR_CHECKS` then counts.

- [ ] **Step 1 (red): the guard and the verdict test.**

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
pub use size_controls::torque_gap_text;

```

with:

```rust
pub use size_controls::torque_gap_text;
// The Size column's rows, for the app's check of the smoke prompt's words.
#[cfg(all(test, feature = "app"))]
pub(crate) use size_controls::SIZE_ROWS;

```

**`linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`**

Replace:

```js
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'check_entry',
  'lesson_link_opened', 'tooltip_plain', 'input_entry', 'tag_entry', 'torque_card',
  'address_follows']
```

with:

```js
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'check_entry',
  'lesson_link_opened', 'same_column', 'tooltip_plain', 'input_entry', 'tag_entry', 'torque_card',
  'address_follows']
```

**`magcoupling-rs/src/app.rs`**

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
        // UX overhaul slice B, stage 2, part 2b-ii-c (spec 4.3): on the Geometry tab the step
        // judges the left side as the Size column, by the words the page draws there (each
        // control's label and the two folds, with no "Design inputs" tab), before it opens the
        // Detailed inputs fold for the row the tooltip step hovers.
        {
            use crate::gui::{ASSUMPTIONS_FOLD, DETAILED_INPUTS};
            let tab = format!(
                "click the \"{}\" tab",
                crate::gui::CentreView::Geometry.label()
            );
            let step = between("same_column=true only if", "tooltip_plain=true only if");
            assert!(
                between(&tab, "tooltip_plain=true only if").contains("same_column=true only if"),
                "the left side is judged on the Geometry tab, before the tooltip"
            );
            for row in crate::gui::SIZE_ROWS {
                assert!(
                    step.contains(&format!("\"{}\"", row.label)),
                    "the step names the control {:?}",
                    row.label
                );
            }
            for fold in [DETAILED_INPUTS, ASSUMPTIONS_FOLD] {
                assert!(
                    step.contains(&format!("\"{fold}\"")),
                    "the step names the fold {fold:?}"
                );
            }
            assert!(
                step.contains("no tab named \"Design inputs\""),
                "the step looks for the tabs that swapped the side"
            );
            assert!(
                step.contains(&format!("click the fold \"{DETAILED_INPUTS}\"")),
                "the tooltip's row is in the Detailed inputs, opened first"
            );
        }
        assert!(
            between("drag_moved=true", "then close the browser").contains(collect),
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib app::tests::the_smoke_test_share_link_opens_its_design 2>&1 | tail -n 8
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: 1 failed: `the_smoke_test_share_link_opens_its_design`. It panics with `no same_column=true only if` (the prompt has no such step yet). The verdict test ends `1 of 82 checks fail` (the script's schema does not have the check the test names).

- [ ] **Step 3 (green): the prompt.**

**`.claude/workflows/gui-smoke.js`**

Replace:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, an input tooltip shows no spreadsheet details and a click on its label shows the input\'s entry in the Explain panel, a click on a dimension tag of the drawing shows its result\'s entry there, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

with:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, the left side on the Geometry tab is still the size controls over the folds Detailed inputs and Assumptions, an input tooltip shows no spreadsheet details and a click on its label shows the input\'s entry in the Explain panel, a click on a dimension tag of the drawing shows its result\'s entry there, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

Replace:

```js
const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'check_entry', 'lesson_link_opened', 'tooltip_plain', 'input_entry', 'tag_entry', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
```

with:

```js
const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'check_entry', 'lesson_link_opened', 'same_column', 'tooltip_plain', 'input_entry', 'tag_entry', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
```

Replace:

```js
    lesson_link_opened: { type: 'boolean' },
    tooltip_plain: { type: 'boolean' },
```

with:

```js
    lesson_link_opened: { type: 'boolean' },
    same_column: { type: 'boolean' },
    tooltip_plain: { type: 'boolean' },
```

Replace:

```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Design menu and
// its Load design item, a check's row, a lesson link, the Geometry tab, an input's label, a dimension tag, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

with:

```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Design menu and
// its Load design item, a check's row, a lesson link, the Geometry tab, the Detailed inputs fold, an input's label, a dimension tag, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

Replace:

```js
// and 5.3) a click on a check's row shows its entry in the Explain panel, which scrolls on its
// own, and the entry's lesson link shows a teaching note there (decision C-11); an input
// row's tooltip shows no spreadsheet details while the toggle is off, and (part 2b-ii, spec 5.2)
```

with:

```js
// and 5.3) a click on a check's row shows its entry in the Explain panel, which scrolls on its
// own, and the entry's lesson link shows a teaching note there (decision C-11); on the Geometry
// tab the left side is still the Size column (part 2b-ii-c, spec 4.3), whose Detailed inputs fold
// the step opens; an input
// row's tooltip shows no spreadsheet details while the toggle is off, and (part 2b-ii, spec 5.2)
```

Replace:

```js
Then a check's entry and its lesson link (every click and mouse move below as above: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot): in the check list (the right side) click the name of the row "Hot-low torque". Wait 1 second and take a screenshot. Under the check list the Explain panel now shows the check's entry: a small shape, "Hot-low torque" and its word, then the headings "What it is" and "Numbers behind this check" with the check's numbers under the second. If the line "Full lesson:" at the entry's foot is not in view, move the mouse over the Explain panel and scroll down with code: async (page) => { await page.mouse.move(X, Y); await page.mouse.wheel(0, 1500); await page.waitForTimeout(1000); }, then take a screenshot. check_entry=true only if the entry shows as described, the line "Full lesson:" followed by a link is in view, and the check list's rows above the Explain panel are still in view (the Explain panel scrolls on its own). Then click the link right of "Full lesson:". Wait 1 second and take a screenshot. lesson_link_opened=true only if the Explain panel (the right side, under the check list) now shows a teaching note in place of the check's entry: a bold title, a few sentences, the heading "What to change" with lines that start with Raise, Lower, Choose or Enter, and a closed fold "Where this comes from" (scroll the Explain panel down as above if the fold is not in view).
Then an input tooltip: click the "Geometry" tab in the row of tabs at the top of the middle region and wait 1 second. On the left the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
Then click that label (as above, at the X, Y the mouse is at), move the mouse off the label onto the drawing in the middle region, wait 1 second and take a screenshot. input_entry=true only if the Explain panel (the right side, under the check list) now shows the input's entry: at its top the input's value and the label you clicked, under them the link "${TRACE_LINK}", then the heading "What it is" with a sentence about the input.
```

with:

```js
Then a check's entry and its lesson link (every click and mouse move below as above: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot): in the check list (the right side) click the name of the row "Hot-low torque". Wait 1 second and take a screenshot. Under the check list the Explain panel now shows the check's entry: a small shape, "Hot-low torque" and its word, then the headings "What it is" and "Numbers behind this check" with the check's numbers under the second. If the line "Full lesson:" at the entry's foot is not in view, move the mouse over the Explain panel and scroll down with code: async (page) => { await page.mouse.move(X, Y); await page.mouse.wheel(0, 1500); await page.waitForTimeout(1000); }, then take a screenshot. check_entry=true only if the entry shows as described, the line "Full lesson:" followed by a link is in view, and the check list's rows above the Explain panel are still in view (the Explain panel scrolls on its own). Then click the link right of "Full lesson:". Wait 1 second and take a screenshot. lesson_link_opened=true only if the Explain panel (the right side, under the check list) now shows a teaching note in place of the check's entry: a bold title, a few sentences, the heading "What to change" with lines that start with Raise, Lower, Choose or Enter, and a closed fold "Where this comes from" (scroll the Explain panel down as above if the fold is not in view).
Then the left side on another tab: click the "Geometry" tab in the row of tabs at the top of the middle region, wait 1 second and take a screenshot. same_column=true only if the left side shows what it showed on the first view: under the heading "Size and torque" the controls "Required torque", "Minimum OD", "Outside diameter", "Overall length", "Magnet face gap" and "Magnets per ring", each a label over a slider, then two closed folds, "Detailed inputs" and "Assumptions"; and no tab named "Design inputs" at its top. Then an input tooltip: click the fold "Detailed inputs" on the left and wait 1 second. Under it the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row (the row under the heading "Key design": the size control of the same name is above the folds) with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
Then click that label (as above, at the X, Y the mouse is at), move the mouse off the label onto the drawing in the middle region, wait 1 second and take a screenshot. input_entry=true only if the Explain panel (the right side, under the check list) now shows the input's entry: at its top the input's value and the label you clicked, under them the link "${TRACE_LINK}", then the heading "What it is" with a sentence about the input.
```

Replace:

```js
Then a dimension tag: the drawing in the middle region (the Geometry tab, still selected) has blue dimension lines, each with a small number on a dark plate at its end. In a fresh screenshot click the number "1" on the drawing: the face gap's, a small number on a dark plate at the top of the round view on the left, on the outer ring of magnets just past the outer end of the short blue line across the gap between the rings; not the "1" of the numbered list under the drawing. Move the mouse off it onto an empty part of the middle region, wait 1 second and take a screenshot. tag_entry=true only if the Explain panel (the right side, under the check list) now shows that dimension's entry: at its top a value in mm and the name "${TAG_LABEL}", and further down the link "Equation"; and no Equation panel opened at the foot of the middle region.
If a target of these last checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes; the check list's last row is "Magnets fit". tag_entry=true was judged last on this page. Collect the console messages at level "info" (the check's entry, the lesson link, the tooltip, the input's entry and the tag's entry ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; and the tooltip step left the Geometry tab selected, where no tile is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
```

with:

```js
Then a dimension tag: the drawing in the middle region (the Geometry tab, still selected) has blue dimension lines, each with a small number on a dark plate at its end. In a fresh screenshot click the number "1" on the drawing: the face gap's, a small number on a dark plate at the top of the round view on the left, on the outer ring of magnets just past the outer end of the short blue line across the gap between the rings; not the "1" of the numbered list under the drawing. Move the mouse off it onto an empty part of the middle region, wait 1 second and take a screenshot. tag_entry=true only if the Explain panel (the right side, under the check list) now shows that dimension's entry: at its top a value in mm and the name "${TAG_LABEL}", and further down the link "Equation"; and no Equation panel opened at the foot of the middle region.
If a target of these last checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes; the check list's last row is "Magnets fit". tag_entry=true was judged last on this page. Collect the console messages at level "info" (the check's entry, the lesson link, the left side, the tooltip, the input's entry and the tag's entry ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; and the tooltip step left the Geometry tab selected, where no tile is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
```

Replace:

```js
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, check_entry, lesson_link_opened, tooltip_plain, input_entry, tag_entry, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
```

with:

```js
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, check_entry, lesson_link_opened, same_column, tooltip_plain, input_entry, tag_entry, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
```

- [ ] **Step 4: Run the checks**, then the whole crate both ways, wasm32 and the verdict test:

  ```bash
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
  cargo test --manifest-path magcoupling-rs/Cargo.toml
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: `test result: ok. 1119 passed; 0 failed; 6 ignored` for the lib tests with the app feature (1530 tests in all with it, 656 without); every clippy run clean; `all 83 checks hold`.

- [ ] **Step 5: Commit** the four files: `test(magcoupler): gui-smoke judges the left column on the Geometry tab and opens the Detailed inputs fold (BL-096 T4)`.

### Task 5: Docs sweep

**Model:** `sonnet`.

The README's module, left-column and smoke passages; `docs/ai/` (02 the system, 03 the modules, 04 the issue, 05 the tracker, the backlog item in progress); the register's rows A-37 to A-42 and its log; owner questions 60 to 65, question 55's default and the list's count of part C; the design's status line; the lines that still placed the banner on the inputs side or named its Assumptions view (the README, `docs/ai/03-structure.yaml`, two comments in `trace.rs`).

**Files:**
- Modify: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/analyses/2026-11-18-owner-questions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, `magcoupling-rs/src/gui/trace.rs` (two comments)

- [ ] **Step 1: The docs.**

**`docs/ai/02-system.yaml`**

Replace:

```yaml
      - "Since BL-092 (UX overhaul slice B, stage 2, part 2b-ii-a; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md): every input is a selection. Touching a Detailed inputs or Assumptions row (input_ui::RowOutput::label_clicked, an edit or a reset, or size_controls::touched on its widget, as the Size column's touches) sets explain::Selection::Input(path); explain::input_ui draws any input (input_row: its value through inputs::value_text, its help, its hint, the checks whose fixes name it, input_trade_offs, its cell) and leads every input's entry with TRACE_LINK, whose click (Asked::Trace) toggles the trace (MagcouplingPanel::toggle_trace); the label no longer traces. A check's fix link selects its input (explain::fix_selection; the ring radius selects Outside diameter's row) and still focuses its row. With the details on, a tile's and a check's entry end with their cell (cell_line). Selection::reads_consequences holds for a Size column row only, so a Detailed input's entry asks the ConsequenceCache nothing."
      - "Since BL-094 (UX overhaul slice B, stage 2, part 2b-ii-b; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md): every result is a selection. A dimension tag (its click area the line and the tag's plate), a row of the callout list and a results-table row draw through Readouts::select or select_over: a click, or Enter or Space on the focused value, sets ReadoutEvents::selected (the focus arriving does not), the value is outlined while it has the focus or its result is the selection (Readouts::with_selection, as a tile and a check row are), the hover has no OPEN_HINT, and the panel sets explain::result_selection(path) (a check's own result selects the check, any other Selection::Result) and traces as a click on a result did; the Equation panel no longer opens on that click (spec 5.5), the entry's Equation link opens it. explain::result_ui draws any result: a tile's number and label, another's value and name as its table row draws them, its help, its dashboard row's check, the stored-3D line under Watch out for STORED_3D_ROWS, Full lesson where ROW_LEARN has its row, Equation where the registry has its equation, its cell with the details. The opening tab draws geometry_view::drawing_ui (the tags, and only the notes with a level); the Geometry tab keeps the callout list. The first view draws 200 words at 1280 x 800, held to 250 (goal G-1)."
```

with:

```yaml
      - "Since BL-092 (UX overhaul slice B, stage 2, part 2b-ii-a; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md): every input is a selection. Touching a Detailed inputs or Assumptions row (input_ui::RowOutput::label_clicked, an edit or a reset, or size_controls::touched on its widget, as the Size column's touches) sets explain::Selection::Input(path); explain::input_ui draws any input (input_row: its value through inputs::value_text, its help, its hint, the checks whose fixes name it, input_trade_offs, its cell) and leads every input's entry with TRACE_LINK, whose click (Asked::Trace) toggles the trace (MagcouplingPanel::toggle_trace); the label no longer traces. A check's fix link selects its input (explain::fix_selection; the ring radius selects Outside diameter's row) and still focuses its row. With the details on, a tile's and a check's entry end with their cell (cell_line). Selection::reads_consequences holds for a Size column row only, so a Detailed input's entry asks the ConsequenceCache nothing."
      - "Since BL-096 (UX overhaul slice B, stage 2, part 2b-ii-c; plan docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md): the left column is the Size column on every tab (design 4.3). MagcouplingPanel::inputs_ui draws size_column_ui in one scroll area whatever CentreView shows: the six controls, then the folds DETAILED_INPUTS (design_inputs_ui: the order toggle, the filter, the Key design group, every input group) and ASSUMPTIONS_FOLD (assumptions_ui), each at the column's full width (show_unindented: no wider than the inputs side was, 320 points in egui's default font; in the app's own font the column still grows to 330.1 points when a fold first opens, BL-099) and neither with a scroll area of its own (BL-091). InputsView and the panel's inputs_view are gone. The trace's banner and Clear trace sit at the column's foot on every tab. A fix on the Outside diameter (SIZE_TAB_ALIASES) sets column_to_top, which scrolls the column back to its slider, and still opens the Size and torque tab. size_controls::controls_ui hands the Minimum OD's slider the value its box shows, so Tab through the Size column selects nothing while every size is on its slider's step (BL-093; a size off its step, as Auto leaves one when it is turned off, is still rewritten when the focus leaves its box: BL-097). panel::tests::nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home is the design's nothing-lost table (8.2 test 9)."
      - "Since BL-094 (UX overhaul slice B, stage 2, part 2b-ii-b; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md): every result is a selection. A dimension tag (its click area the line and the tag's plate), a row of the callout list and a results-table row draw through Readouts::select or select_over: a click, or Enter or Space on the focused value, sets ReadoutEvents::selected (the focus arriving does not), the value is outlined while it has the focus or its result is the selection (Readouts::with_selection, as a tile and a check row are), the hover has no OPEN_HINT, and the panel sets explain::result_selection(path) (a check's own result selects the check, any other Selection::Result) and traces as a click on a result did; the Equation panel no longer opens on that click (spec 5.5), the entry's Equation link opens it. explain::result_ui draws any result: a tile's number and label, another's value and name as its table row draws them, its help, its dashboard row's check, the stored-3D line under Watch out for STORED_3D_ROWS, Full lesson where ROW_LEARN has its row, Equation where the registry has its equation, its cell with the details. The opening tab draws geometry_view::drawing_ui (the tags, and only the notes with a level); the Geometry tab keeps the callout list. The first view draws 200 words at 1280 x 800, held to 250 (goal G-1)."
```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), select and select_over (a click asks for the result in the Explain panel: ReadoutEvents::selected, since BL-094), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel: new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider, RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); drawing_ui (the opening tab's: the tags, and only the notes with a level, since BL-094); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the inputs side; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

with:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), select and select_over (a click asks for the result in the Explain panel: ReadoutEvents::selected, since BL-094), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel, whose left column is the Size column on every tab since BL-096 (inputs_ui, size_column_ui: the size controls over the folds DETAILED_INPUTS and ASSUMPTIONS_FOLD, the trace's banner at its foot; no InputsView): new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider, RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); drawing_ui (the opening tab's: the tags, and only the notes with a level, since BL-094); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the left column, on every tab since BL-096; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

Replace:

```yaml
  web_bundle: "linkage-sim-rs/web/tools/magcoupler/ (index.html committed, canvas magcoupling_canvas, absolute /tools/magcoupler/ glue path; magcoupling-web.js and magcoupling-web_bg.wasm gitignored in web/.gitignore), built by linkage-sim-rs/scripts/build_magcoupling_web.sh (called by build_web.sh, which deploy-web.yml runs), served at colesorkness.com/tools/magcoupler/ (serve_web.sh [PORT] locally); vercel.json revalidates /tools/magcoupler/magcoupling-web.js and redirects the old /magcoupling/ paths there"
  manual_size_gui: "src/gui/size_controls.rs (controls_ui, the direct size/count/gap sliders and a local Minimum OD range preference); panel.rs CentreView::SizeAndTorque replaces the inputs side with these controls and a closed Detailed inputs section, keeps live engine torque and geometry in the centre and all dashboard checks, and joins edits to existing history; since slice 1 it is the first view and the solver's home: the Required torque slider (metal.required_min_Nm), Hot-low torque and its gap (torque_gap_text), the Outside diameter and Overall length hovers (DIAMETER_HOVER, LENGTH_HOVER; since BL-074 the length hover also says that the first move from a blank axial magnet length sets both rings to one length), pullout_label, the Auto toggle (AUTO, AUTO_SUFFIX, variable_slider, change_text, SET_BY_AUTO: the matching slider disabled) and the consequence lines under each control (Consequences, one ConsequenceCache held by the panel); since slice 2 a hint line (plain_text::HINTS) under each control, above its consequence lines; since slice 3 the torque card (gui::torque_card, held in the same ConsequenceCache; since BL-081 drawn in the Hot-low tile's Explain entry, explain::card_ui, not between the readouts and the drawing), and under Magnets per ring the ring-size line and the fit line only (the lines for two magnets fewer and two more are the card's count row)"
  shipped_builds: "linkage-sim-rs/scripts/magcoupling_shipped.sh (sourced by build_web.sh, build_magcoupling_web.sh and gate.sh): MAGCOUPLING_WEB_ARGS, MAGCOUPLING_NATIVE_ARGS, LINKAGE_WEB_ARGS (the linkage web bundle; no magcoupling-rs in it), magcoupling_assert_shipped (reads cargo --message-format=json; fails unless the shipped bin was compiled and no magcoupling-rs unit has workbook-parity)"
```

with:

```yaml
  web_bundle: "linkage-sim-rs/web/tools/magcoupler/ (index.html committed, canvas magcoupling_canvas, absolute /tools/magcoupler/ glue path; magcoupling-web.js and magcoupling-web_bg.wasm gitignored in web/.gitignore), built by linkage-sim-rs/scripts/build_magcoupling_web.sh (called by build_web.sh, which deploy-web.yml runs), served at colesorkness.com/tools/magcoupler/ (serve_web.sh [PORT] locally); vercel.json revalidates /tools/magcoupler/magcoupling-web.js and redirects the old /magcoupling/ paths there"
  manual_size_gui: "src/gui/size_controls.rs (controls_ui, the direct size/count/gap sliders and a local Minimum OD range preference); panel.rs draws these controls in the left column of every tab, over the closed Detailed inputs and Assumptions folds (the Size column, since BL-096; until then only CentreView::SizeAndTorque's side had them); CentreView::SizeAndTorque keeps live engine torque and geometry in the centre and all dashboard checks, and joins edits to existing history; since slice 1 it is the first view and the solver's home: the Required torque slider (metal.required_min_Nm), Hot-low torque and its gap (torque_gap_text), the Outside diameter and Overall length hovers (DIAMETER_HOVER, LENGTH_HOVER; since BL-074 the length hover also says that the first move from a blank axial magnet length sets both rings to one length), pullout_label, the Auto toggle (AUTO, AUTO_SUFFIX, variable_slider, change_text, SET_BY_AUTO: the matching slider disabled) and the consequence lines under each control (Consequences, one ConsequenceCache held by the panel); since slice 2 a hint line (plain_text::HINTS) under each control, above its consequence lines; since slice 3 the torque card (gui::torque_card, held in the same ConsequenceCache; since BL-081 drawn in the Hot-low tile's Explain entry, explain::card_ui, not between the readouts and the drawing), and under Magnets per ring the ring-size line and the fit line only (the lines for two magnets fewer and two more are the card's count row)"
  shipped_builds: "linkage-sim-rs/scripts/magcoupling_shipped.sh (sourced by build_web.sh, build_magcoupling_web.sh and gate.sh): MAGCOUPLING_WEB_ARGS, MAGCOUPLING_NATIVE_ARGS, LINKAGE_WEB_ARGS (the linkage web bundle; no magcoupling-rs in it), magcoupling_assert_shipped (reads cargo --message-format=json; fails unless the shipped bin was compiled and no magcoupling-rs unit has workbook-parity)"
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-b (BL-094, plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md, decisions D-60 to D-70): a click on a dimension tag, a callout row or a results-table row selects its result (a check's own result the check) and opens no equation; the entry's Equation link does (D-61, A-33, owner question 56: every other readout still opens the Equation panel); the selection keeps the click's trace (D-62, A-34, owner question 57); any result's entry from texts the screen drew (D-64); the callout list and the plain notes to the Geometry tab, a note with a level kept under the opening tab's drawing (D-66, A-35, owner question 58); the first view at most 250 words (200), the walk's three copies of a step's title kept (D-67, A-36, owner question 59); gui-smoke's tag_entry (D-68). risk: mechanical (D-69)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
```

with:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-b (BL-094, plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md, decisions D-60 to D-70): a click on a dimension tag, a callout row or a results-table row selects its result (a check's own result the check) and opens no equation; the entry's Equation link does (D-61, A-33, owner question 56: every other readout still opens the Equation panel); the selection keeps the click's trace (D-62, A-34, owner question 57); any result's entry from texts the screen drew (D-64); the callout list and the plain notes to the Geometry tab, a note with a level kept under the opening tab's drawing (D-66, A-35, owner question 58); the first view at most 250 words (200), the walk's three copies of a step's title kept (D-67, A-36, owner question 59); gui-smoke's tag_entry (D-68). risk: mechanical (D-69)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-c (BL-096, plan docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md, decisions D-71 to D-85): the left column is the Size column on every tab, the input groups only in its Detailed inputs fold and the assumptions only in its Assumptions fold; the InputsView tabs are gone; the trace's banner and Clear trace at the column's foot on every tab (D-74, register row A-37, owner question 60); BL-091 taken with it (D-72); the Minimum OD's slider is handed the value its box shows, so Tab through the Size column selects nothing while every size is on its slider's step (D-73, BL-093's rule; BL-097 holds a size off its step); the folds have no indent (D-75: an indented fold widened the column by 13 points; nothing marks where one fold ends, A-41, question 64; in the app's own font the column still grows to 330.1 points when a fold first opens, BL-099); the column's heading stays Size and torque on every tab (D-85, A-42, question 65); the Outside diameter's fix link still opens the Size and torque tab and scrolls the column to the slider (D-76, A-38, question 61); the Key design group keeps Auto's line and its Open the Size tab button (D-77, A-39, question 62); a trace frames rows only inside the folds (D-79, A-40, question 63); the table-driven nothing-lost test (D-80); gui-smoke's same_column (D-82). Findings filed, not fixed: BL-097 (Tab through a value box rewrites a value off its step; Auto on, Auto off, Tab is the short way there), BL-098 (a teaching note widens the right column by 16 points), BL-099 (in the app's font an opened fold or a filter widens the left column) and BL-100 (a row that takes the keyboard's focus is not scrolled into view). risk: mechanical."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown

---

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-b: the drawing's selections and the word budget (BL-094)
```

with:

```markdown

---

## 2026-11-25 - Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-c: the left column and the nothing-lost test (BL-096)
- What changed: the left column is the Size column on every tab (design 4.3): the size controls, then the folds Detailed inputs and Assumptions, which are now the only home of the input groups and of the assumptions; the tabs "Design inputs" and "Assumptions" are gone (`InputsView`). The folds are drawn with no indent, so an opened fold is no wider than the inputs side was, and scroll with the column (BL-091). A trace's banner and Clear trace sit at the column's foot on every tab, the opening one too. A fix link on the Outside diameter scrolls the column back to its slider. Tab through the Size column selects nothing and sets no Minimum OD while every size is on its slider's step. gui-smoke judges the left side on the Geometry tab (`same_column`) and opens the Detailed inputs fold for its tooltip step.
- Tests: the fold and the filtered view scroll with the column and a row asked for still scrolls into view; Tab through the Size column; the column's rows in the same place on every tab, the folds' state kept across tabs, the column 320 points wide with a fold open (egui's default font); the banner and Clear trace at the column's foot on every tab at 1024 x 700 and 1280 x 1024; the Outside diameter's fix link from a scrolled column; the table-driven nothing-lost test (the old top bar's actions, every CentreView, every shown note in Learn, every DASHBOARD row's home, every input and every assumption from every tab); the smoke prompt's words (app) and the verdict test's list. The panel tests' harness opens the Detailed inputs fold where a test drives its rows.
- Not in this part: part 2b-iii (the Plots tab and the look). Filed, not fixed: BL-097 (Tab through a value box rewrites a value off its step: Auto on, Auto off, Tab), BL-098 (a teaching note widens the right column), BL-099 (in the app's own font an opened fold or a filter widens the left column), BL-100 (a row that takes the keyboard's focus is not scrolled into view).

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-b: the drawing's selections and the word budget (BL-094)
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  priority: 2
  status: open
  notes: "Filed 2026-11-25 with its plan (the design's header: each plan files one item). Part 2b-ii-c of BL-092's list, items 1 and 2, in one plan (decision D-71). It takes BL-091 with it (decision D-72): the fold becomes the only home of the input groups. risk: mechanical (decision D-83): no new number, no new physical claim and no new on-screen string; the column draws on every tab what the opening tab drew. Part 2b-iii is not touched. The planner's findings on main, not fixed here: BL-097 and BL-098. The plan's review (2026-11-25, two lenses) filed two more, BL-099 and BL-100, and its fix wave changed the plan before any build: Task 1's second red test (the filtered view), the text guard's click on the centre's tab, three tests that held only by the harness's scroll, the small-window tests with the fold open, and the docs sweep's stale lines."
- id: BL-097
```

with:

```yaml
  priority: 2
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2b-ii-c (Tasks 1 to 4 of the plan); the full gate, the smoke run and the whole-branch review are its Task 6. Filed 2026-11-25 with its plan (the design's header: each plan files one item). Part 2b-ii-c of BL-092's list, items 1 and 2, in one plan (decision D-71). It takes BL-091 with it (decision D-72): the fold becomes the only home of the input groups. risk: mechanical (decision D-83): no new number, no new physical claim and no new on-screen string; the column draws on every tab what the opening tab drew. Part 2b-iii is not touched. The planner's findings on main, not fixed here: BL-097 and BL-098. The plan's review (2026-11-25, two lenses) filed two more, BL-099 and BL-100, and its fix wave changed the plan before any build: Task 1's second red test (the filtered view), the text guard's click on the centre's tab, three tests that held only by the harness's scroll, the small-window tests with the fold open, and the docs sweep's stale lines."
- id: BL-097
```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
| A-36 | Part 2a's decision D-32 left the three copies of a Start here step's title (the note's heading, the lesson line, the walk bar's step line) for the word budget of 250 to decide. The budget measures the first view, where no walk runs (200 words with the callout list moved). Which copies go? | None: all three stay. | The budget is met without them; dropping one is a text change for the text review, and each copy is its own place's form. Owner question 59. | The lesson line under the equation drops the title while the walk runs (the walk bar and the note's heading name it). | The plan of part 2b-ii-b (BL-094), decision D-67 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
```

with:

```markdown
| A-36 | Part 2a's decision D-32 left the three copies of a Start here step's title (the note's heading, the lesson line, the walk bar's step line) for the word budget of 250 to decide. The budget measures the first view, where no walk runs (200 words with the callout list moved). Which copies go? | None: all three stay. | The budget is met without them; dropping one is a text change for the text review, and each copy is its own place's form. Owner question 59. | The lesson line under the equation drops the title while the walk runs (the walk bar and the note's heading name it). | The plan of part 2b-ii-b (BL-094), decision D-67 |
| A-37 | Part 2b-ii-a's decision D-52 (row A-30) left the trace's banner and its Clear trace button at the foot of the inputs side of the other tabs until the left column stopped swapping. Where do they go now? | At the foot of the left column on every tab, the opening one too, under the column's scroll area, as they sat at the foot of the inputs side. | No new text and one copy of the banner. The opening tab gains the off switch question 55 asked about. Owner question 60. | The banner and Clear trace under the trace link in the traced input's entry (question 55's answer b), or in both places. | The plan of part 2b-ii-c (BL-096), decision D-74 |
| A-38 | A fix link on the Outside diameter opened the Size and torque tab because only that tab had the slider. The slider is now on every tab. Does the link still switch the tab? | Yes, as before; and the left column scrolls back to its top, where the slider is. | Keeping the switch takes nothing a user had. The scroll is needed either way: inside the open Detailed inputs the slider can be scrolled out of view. Owner question 61. | The link selects the control and scrolls to it, and the centre tab stays where it is. | The plan of part 2b-ii-c (BL-096), decision D-76 |
| A-39 | The Key design group shows Auto's line while Auto is on ("Auto: ... set by the solver (Size tab)"), a button "Open the Size tab" and the solver's status. The Size column, with Auto's row and outcome, is now above it on every tab. Do the line and the button stay? | They stay, word for word. | The on-screen wording is kept unless the scope needs new text, and nothing reads false: the button still opens the Size and torque tab. The cost: with Auto on and the fold open the solver's status shows twice in the column, as on the opening tab since part 2b-i. Owner question 62. | Drop the button and the words "(Size tab)", or the whole line (the locked row stays). | The plan of part 2b-ii-c (BL-096), decision D-77 |
| A-40 | A trace frames the input rows it reaches. Those rows are now inside the Detailed inputs fold, shut at first. Should the Size controls take the frame, or the fold's header count the traced rows as each group's does? | Neither: the frames stay on the rows inside the folds, the banner names the trace and what it reaches, and Clear trace ends it. | No new mark and no new text. The cost: with a result traced and the folds shut, the banner says "reads N inputs" and no input row is framed in view; before, the other tabs showed the Key design rows framed. One click on the fold shows them. Owner question 63. | The fold's header reads "Detailed inputs (N traced)", or the Size controls whose input is traced are framed too. | The plan of part 2b-ii-c (BL-096), decision D-79 |
| A-41 | The two folds of the left column are drawn with no indent (decision D-75). With Detailed inputs open, its eight group headings and the Assumptions fold's header then start at the same edge with the same triangle: Assumptions reads as a ninth input group, and nothing marks where the Detailed inputs end. Should a line or a heavier header mark the folds? | No: the two headers stay as they are. | No new mark and no new on-screen string, and an indent is what widened the column. The cost: the end of the open fold is not marked. Owner question 64. | A separator line above the Assumptions fold's header, or the two fold headers in a heavier face than the group headings. | The plan of part 2b-ii-c (BL-096), decision D-75; its review, UXT-4 |
| A-42 | The left column is headed "Size and torque" on every tab. On every tab but the first those are the words of a centre tab that is not the one shown. The design's sketch (4.1) and its section 4.3 head the column "Size". Which words head the column? | "Size and torque", the heading the opening tab's column has today. | The part adds no on-screen string; the heading is the text it was. The cost: the same words twice on the screen, one of them a tab's. Owner question 65. | The heading reads "Size". | The plan of part 2b-ii-c (BL-096), decision D-85; its review, UXT-4 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
```

Replace:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-23: A-33, A-34, A-35 and A-36 added with the plan of slice B's stage 2, part 2b-ii-b (BL-094). Each is an assumption, not the owner's answer.
```

with:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-25: A-37, A-38, A-39 and A-40 added with the plan of slice B's stage 2, part 2b-ii-c (BL-096). Each is an assumption, not the owner's answer. A-30's wait is over: the banner and Clear trace are placed by A-37.
- 2026-11-25: A-41 and A-42 added by the fix wave of that plan's review (UXT-4): the folds' boundary and the left column's heading. Each is an assumption, not the owner's answer.
- 2026-11-23: A-33, A-34, A-35 and A-36 added with the plan of slice B's stage 2, part 2b-ii-b (BL-094). Each is an assumption, not the owner's answer.
```

**`docs/analyses/2026-11-18-owner-questions.md`**

Replace:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 39 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54, 55, 56, 57, 58 and 59 on 2026-11-23). Part D, five that matter only if the tools stop being generic.

```

with:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 45 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54, 55, 56, 57, 58 and 59 on 2026-11-23, questions 60 to 65 on 2026-11-25). Part D, five that matter only if the tools stop being generic.

```

Replace:

```markdown

- **If you do not answer:** as described, until part 2b-ii-c.
- (a) As the default.
```

with:

```markdown

- **If you do not answer:** as described, until part 2b-ii-c. Since part 2b-ii-c (BL-096) the banner and Clear trace sit at the foot of the left column on every tab, the opening one too: question 60.
- (a) As the default.
```

Replace:

```markdown

### Linkage simulator
```

with:

```markdown

### 60. Where should a trace's banner and its Clear trace button sit?

Added 2026-11-25 (the next free number). Since part 2b-ii-c the left column is the same on every tab, so the banner ("Tracing ...: drives N results with an equation") and Clear trace sit at its foot on every tab. The opening tab, which had neither (question 55), now shows both while a trace is on.

- **If you do not answer:** at the foot of the left column, on every tab.
- (a) As the default.
- (b) Under the trace link in the traced input's entry instead (question 55's answer b).
- (c) In both places.

Source: UX design sections 4.3 and 5.2; A-37; BL-096 plan, decision D-74.

### 61. Should a fix link on the Outside diameter still switch to the Size and torque tab?

Added 2026-11-25 (the next free number). The link switched the tab because only that tab had the Outside diameter slider. The slider is now on the left of every tab. The link still switches the tab, and now also scrolls the left column back to the slider.

- **If you do not answer:** it switches the tab and scrolls to the slider.
- (a) As the default.
- (b) It scrolls to the slider and leaves the centre tab where it is.

Source: UX design sections 4.3 and 5.2; A-38; BL-096 plan, decision D-76.

### 62. Should the Key design group keep Auto's line and its "Open the Size tab" button?

Added 2026-11-25 (the next free number). While Auto is on, the Key design group shows "Auto: ... set by the solver (Size tab)", a button "Open the Size tab" and the solver's status. The size controls, with Auto's row and its outcome, are now above that group on every tab, so with the Detailed inputs open the status shows twice.

- **If you do not answer:** the line and the button stay, word for word.
- (a) As the default.
- (b) Drop the button and the words "(Size tab)".
- (c) Drop the whole line; the locked row of the size Auto sets stays.

Source: UX design section 4.3; A-39; BL-096 plan, decision D-77.

### 63. With a result traced and the folds shut, should something on the left show what is traced?

Added 2026-11-25 (the next free number). A trace frames the input rows it reaches. Those rows are inside the Detailed inputs fold, shut at first. The banner says "Tracing ...: reads N inputs", and no input row is framed until the fold is opened. Before, the other tabs showed the Key design rows framed.

- **If you do not answer:** nothing more: the banner, and the frames once the fold is open.
- (a) As the default.
- (b) The fold's header counts the rows, "Detailed inputs (N traced)", as each group's does.
- (c) The size controls whose input is traced are framed too.

Source: UX design section 4.3; decision O-8; A-40; BL-096 plan, decision D-79.

### 64. With Detailed inputs open, should something mark where that fold ends and Assumptions begins?

Added 2026-11-25 (the next free number). The two folds of the left column are drawn with no indent, so that an opened fold does not widen the column. With Detailed inputs open, its eight group headings and the Assumptions header start at the same edge with the same triangle, and Assumptions reads as a ninth input group.

- **If you do not answer:** nothing more: the two headers as they are.
- (a) As the default.
- (b) A separator line above the Assumptions header.
- (c) The two fold headers in a heavier face than the group headings.

Source: UX design sections 4.1 and 4.3; A-41; BL-096 plan, decision D-75.

### 65. Should the left column's heading read "Size" instead of "Size and torque"?

Added 2026-11-25 (the next free number). The left column is now the same on every tab and is headed "Size and torque", the words of the first centre tab. On the other tabs those words stand over the column while that tab is not the one shown. The design's sketch heads the column "Size".

- **If you do not answer:** "Size and torque", the heading it has today.
- (a) As the default.
- (b) "Size", as the design's sketch.

Source: UX design sections 4.1 and 4.3; A-42; BL-096 plan, decision D-85.

### Linkage simulator
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, its second (the drawing's selections and the word budget; BL-094) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md`; parts 2b-ii-c and 2b-iii and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

with:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, its second (the drawing's selections and the word budget; BL-094) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md`, its third (the left column and the nothing-lost test; BL-096) by the plan `docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md`; part 2b-iii and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

**`magcoupling-rs/README.md`**

Replace:

```markdown
| `src/engine/explain/records/shaft_joint.rs` | BL-064 shaft joint records: one per numeric `shaft_joint.*` result (23; the verdict, a text, has none), transcribed from `shaft_joint::compute` and `PartProperties::hub` with every correction on: the part material in effect as cases over the circuit and the back-iron pick (the Materials inputs for the default pick, a pick's library values, else the workbook's 6061-T6), each joint's Lamé rows (holding, bore stress, needed minimum, elastic edge) with a no-wall arm that gives NaN, the key alone (0 with no keyway, +inf with no bearing length or contact), each shaft's shrink as cases on 265 K with NIST's Fe-2.25Ni polynomial, the fitting temperature and the fittable edge, and the bonded inner hub's allowed fitting temperature as the lowest of your limit, the adhesive's design limit and the inner ring's own single-ring onset less 10 °C, NaN when any of the three is (owner R-15) |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, `InputsView` (the design inputs or the assumptions), the assumptions banner, the sizing controls, the right column (`checks_and_explain_ui`: the check list over the Explain panel, one `explain::Selection`, the Hot-low entry's consequences asked of the `ConsequenceCache` through `size_tab_consequences`, the Size tab's one ask, with the design's own results computed once per design (`ResultsCache`))), `tiles.rs` (since BL-081, the four tiles over the drawing on Size and torque: `Tile`, `tiles_ui`, `MIN_TILE_WIDTH`, the Hot-low tile's `needs_lines` and `gap_pieces`), `checks.rs` (since BL-081, the check list: `checks_ui`, `state_mark`, `mark_level`, `CHECKS_SHARE`), `explain.rs` (since BL-081, the Explain panel: `explain_ui`, `Selection`, `Asked`, the orientation and each entry; the torque card sits in the Hot-low tile's entry, `card_ui`, decision C-10; an entry's "Full lesson:" link is `guidance::row_learn`'s), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::new(marks, show_details)`, `show_details`, `Readouts::show` and `show_over` (the hover text built with the frame's `show_details`), `select` and `select_over` (a click selects the result, since BL-094; outlined while focused or selected, `with_selection`), `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `note_ui` (a note drawn against a `NoteView`, returning the `NoteLink` clicked), `held_reason`, `term_tag`, `reviewed_note`, `note_fallback` (Explain's next best read under an equation without a note, `NoteFallback`), `start_here_text` with `FERRITE_ONLY` on the ferrite step; its heights: `panel_heights`, `PANEL_SHARE`, `MIN_PANEL_HEIGHT`, `VIEW_STRIP`), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `plain_text.rs` (the display-name and help overlay over the engine's metadata, keyed by path: `label_for`, `help_for`, `original_help`, `hint_for`, `display_label`; `LABELS` holds spec 3.5's canonical names, three of them constants the Size tab and the plots draw too: `REQUIRED_TORQUE`, `PULL_OUT_NOMINAL`, `HOT_LOW_TORQUE`; `HELP` and `HINTS` the slice-2 help), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`; `wall_cautions`, the Size tab's amber wall lines: the wall past the keyway and, since BL-082, the cap skirt under `MIN_CAP_SKIRT_MM`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list; `drawing_ui`, the opening tab's, without the list; `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `slider_range` (an input's slider range from the catalogue built once, the shared spelling the torque card, the Size tab's controls, the guides and the tests read; the Equation panel and the panel's Required slider keep their own), `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type; a click on its label traces the input; `weak_line` and `hint_line`: a weak line wrapped to its region's width (under a control, and the torque card's lines), and the hint line `plain_text::hint_for` gives a Key design row or a Size-tab control), `dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`, `shown_lines` (the dashboard rows drawn while the shaft-clamp section is paused or not; `dashboard_lines` keeps every row for the exports), `END_EFFECT_ROWS`, `greyed_by_end_effect`, `STORED_3D_ROWS`, `STORED_3D_INPUTS`, `STORED_3D_SENTENCE`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`; `why_note_text` and `why_note_link`, a "Why: <title>" link outside the rows, and `end_effect_banner_ui`, the banner with its link), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), `result_groups.rs` (`result_groups`: the headline, the nine chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`, `target_mismatch_notice`), `history.rs` (`History`: undo and redo), `guidance.rs` (the guidance layer, pure functions: `CHECK_GUIDES`, `counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `check_terms` (`CheckTerm`, `Quoted` with its one constructor `Quoted::new`; slice 3: each counted check's terms in a fixed order, the one table the fix tests and the torque card read), `SPACE_CLAIM_AXIS_SIZES`, `all_fixes`, `FixCache`, `within_minimum_diameter`, `guide_notes` (`trade_offs`, `side_effects` and the unchecked costs), `design_status`, `Consequences` (the lines under the Size tab's controls and its torque card, `card`), `ConsequenceCache` (one key: the design shown, the Minimum OD and the shaft-clamp pause), `magnet_count_line`, `SIZE_TAB_NOTE`, `TORQUE_BAND_NOTE` and `BANNER_NOTE` (where the Size tab's and the banner's Why links lead), `plain_text_violations` (re-exported from `engine::explain::plain`)), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines (since slice 3 without the N-2 and N+2 lines, which are the torque card's count row) and Why links; `slider_minimum`, the Outside diameter slider's lowest value, one expression with the torque card; `SliderMapping`, each Size slider's own mapping of a value to a design, one mapping for the controls and the torque card's steps, which sets the count whole, refusing a value a cast would change, and checks no range on the gap or the count: its caller holds the value to the slider), `torque_card.rs` (the torque card, slice 3, BL-073; pure, no egui: `card` -> `TorqueCard` (`ShownCountBelowSlider`, `ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow`, or `Rows`, one `Row` per Size control (`Control`), read by `rows`), each control one fixed step each way through its slider's own mapping (`size_controls::SliderMapping`, the one the controls apply), in the order of tests of its spec's section 3.3: a side's `Words`, or its `Change` (the stepped design's hot-low, its change in N·m and as a fraction of the design shown's, the Outside diameter and Overall length changes, whether the calibration factor changes, the counted checks whose level flips (`Flip`, over `counted_guides` at the panel's shaft-clamp pause) and the failing checks' quantities the step moves (`MarginChange`, from `guidance::check_terms`, drawn where `shows`)); `lines(auto)` draws it, a row's two N·m at one precision and the row Auto holds as `held_line`, every text a named constant (`CARD_HEADING`, `CAPTION`, the three Auto lines, the side words, `JUMP` and `CUP_JUMP`, `CALIBRATION_CHANGE`, the four one-line forms and the flip words; `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` also end the Size tab's exploration lines); `side_words` (sizing's validity rule, every row's, the keyway left out on the count row: decision D-1) and `count_step` (two magnets each way: checked, even, inside the slider), both moved from the N-2 and N+2 lines that the count row replaced; `STEP_SLACK_MM`, the one slack of the range test and the jump rules: the overlapping magnet length on the Overall length row, the cup's diameter on the Outside diameter row since the owner's FO-2), `sizing.rs` (`SizingMode`, `held_by_auto`, `SizingState` (the mode and the free variable; the target is `metal.required_min_Nm`), `SizingRunner`: debounced inverse sizing over a search range; `sizing_range`, the range Auto searches, one rule for the panel and the torque card's test table, slice 3), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell; the Size tab's one number format, slice 3 decision 10B: `fixed` (a number to a set number of decimals, with no + added), and `signed_fixed`, `signed_change` and `change_decimals` (a change to the fewest decimals, from its own (`TORQUE_DECIMALS`, two, for N·m: the readout's gap and the torque card), that keep its true sign and read a digit that is not zero, at most `CHANGE_EXTRA_DECIMALS` more: `decimals_with_a_digit`, the one cap rule, which the torque card's `shows` reads too), `shared_change_decimals` (a torque card row's two N·m at one precision), the one-decimal `signed_percent`, the format's bound since the owner's FO-4 at the slice-3 STOP (a number whose fixed form would need more than six whole digits, `MOST_WHOLE_DIGITS`, reads four significant digits with its sign, as the readouts write values, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row: `TorquePrecision`, the one precision `shared_torque_precision` finds for a row, the readout's gap a row of one, which `signed_torque_change` draws for a torque card side and the readout's margin and `torque_shortfall` for its shortfall's size, which is drawn without a sign), and `fewest_decimals_apart`, the guidance's search for more decimals), `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries; `mark_ui`, the brand mark before the title; `tab_ui`, a tab of the centre's row, underlined in the accent when it is the active one; `quiet_text`, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes; `mark_color`, the colour of the changed dot and a trace's marks), `test_support.rs` (headless egui helpers and the shared test designs, the torque card's among them with their one table, `card_designs()`; tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests; on the web it writes the settled design into the page's address), `starting_design` (the design it opens on and hands the panel as its baseline), `address_of` (the page address that stands for a design), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the calculator's own theme, "Charcoal and crimson": the token table and the visuals built from it, forced dark), `app/fonts.rs` (the app's font: a cut of Inter Regular with tabular digits, ahead of egui's built-in fonts), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
```

with:

```markdown
| `src/engine/explain/records/shaft_joint.rs` | BL-064 shaft joint records: one per numeric `shaft_joint.*` result (23; the verdict, a text, has none), transcribed from `shaft_joint::compute` and `PartProperties::hub` with every correction on: the part material in effect as cases over the circuit and the back-iron pick (the Materials inputs for the default pick, a pick's library values, else the workbook's 6061-T6), each joint's Lamé rows (holding, bore stress, needed minimum, elastic edge) with a no-wall arm that gives NaN, the key alone (0 with no keyway, +inf with no bearing length or contact), each shaft's shrink as cases on 265 K with NIST's Fe-2.25Ni polynomial, the fitting temperature and the fittable edge, and the bonded inner hub's allowed fitting temperature as the lowest of your limit, the adhesive's design limit and the inner ring's own single-ring onset less 10 °C, NaN when any of the three is (owner R-15) |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, the left column (since BL-096 the Size column on every tab, with its folds `DETAILED_INPUTS` and `ASSUMPTIONS_FOLD`), the assumptions banner, the sizing controls, the right column (`checks_and_explain_ui`: the check list over the Explain panel, one `explain::Selection`, the Hot-low entry's consequences asked of the `ConsequenceCache` through `size_tab_consequences`, the Size tab's one ask, with the design's own results computed once per design (`ResultsCache`))), `tiles.rs` (since BL-081, the four tiles over the drawing on Size and torque: `Tile`, `tiles_ui`, `MIN_TILE_WIDTH`, the Hot-low tile's `needs_lines` and `gap_pieces`), `checks.rs` (since BL-081, the check list: `checks_ui`, `state_mark`, `mark_level`, `CHECKS_SHARE`), `explain.rs` (since BL-081, the Explain panel: `explain_ui`, `Selection`, `Asked`, the orientation and each entry; the torque card sits in the Hot-low tile's entry, `card_ui`, decision C-10; an entry's "Full lesson:" link is `guidance::row_learn`'s), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::new(marks, show_details)`, `show_details`, `Readouts::show` and `show_over` (the hover text built with the frame's `show_details`), `select` and `select_over` (a click selects the result, since BL-094; outlined while focused or selected, `with_selection`), `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `note_ui` (a note drawn against a `NoteView`, returning the `NoteLink` clicked), `held_reason`, `term_tag`, `reviewed_note`, `note_fallback` (Explain's next best read under an equation without a note, `NoteFallback`), `start_here_text` with `FERRITE_ONLY` on the ferrite step; its heights: `panel_heights`, `PANEL_SHARE`, `MIN_PANEL_HEIGHT`, `VIEW_STRIP`), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `plain_text.rs` (the display-name and help overlay over the engine's metadata, keyed by path: `label_for`, `help_for`, `original_help`, `hint_for`, `display_label`; `LABELS` holds spec 3.5's canonical names, three of them constants the Size tab and the plots draw too: `REQUIRED_TORQUE`, `PULL_OUT_NOMINAL`, `HOT_LOW_TORQUE`; `HELP` and `HINTS` the slice-2 help), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`; `wall_cautions`, the Size tab's amber wall lines: the wall past the keyway and, since BL-082, the cap skirt under `MIN_CAP_SKIRT_MM`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list; `drawing_ui`, the opening tab's, without the list; `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `slider_range` (an input's slider range from the catalogue built once, the shared spelling the torque card, the Size tab's controls, the guides and the tests read; the Equation panel and the panel's Required slider keep their own), `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type; a click on its label traces the input; `weak_line` and `hint_line`: a weak line wrapped to its region's width (under a control, and the torque card's lines), and the hint line `plain_text::hint_for` gives a Key design row or a Size-tab control), `dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`, `shown_lines` (the dashboard rows drawn while the shaft-clamp section is paused or not; `dashboard_lines` keeps every row for the exports), `END_EFFECT_ROWS`, `greyed_by_end_effect`, `STORED_3D_ROWS`, `STORED_3D_INPUTS`, `STORED_3D_SENTENCE`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`; `why_note_text` and `why_note_link`, a "Why: <title>" link outside the rows, and `end_effect_banner_ui`, the banner with its link), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), `result_groups.rs` (`result_groups`: the headline, the nine chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`, `target_mismatch_notice`), `history.rs` (`History`: undo and redo), `guidance.rs` (the guidance layer, pure functions: `CHECK_GUIDES`, `counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `check_terms` (`CheckTerm`, `Quoted` with its one constructor `Quoted::new`; slice 3: each counted check's terms in a fixed order, the one table the fix tests and the torque card read), `SPACE_CLAIM_AXIS_SIZES`, `all_fixes`, `FixCache`, `within_minimum_diameter`, `guide_notes` (`trade_offs`, `side_effects` and the unchecked costs), `design_status`, `Consequences` (the lines under the Size tab's controls and its torque card, `card`), `ConsequenceCache` (one key: the design shown, the Minimum OD and the shaft-clamp pause), `magnet_count_line`, `SIZE_TAB_NOTE`, `TORQUE_BAND_NOTE` and `BANNER_NOTE` (where the Size tab's and the banner's Why links lead), `plain_text_violations` (re-exported from `engine::explain::plain`)), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines (since slice 3 without the N-2 and N+2 lines, which are the torque card's count row) and Why links; `slider_minimum`, the Outside diameter slider's lowest value, one expression with the torque card; `SliderMapping`, each Size slider's own mapping of a value to a design, one mapping for the controls and the torque card's steps, which sets the count whole, refusing a value a cast would change, and checks no range on the gap or the count: its caller holds the value to the slider), `torque_card.rs` (the torque card, slice 3, BL-073; pure, no egui: `card` -> `TorqueCard` (`ShownCountBelowSlider`, `ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow`, or `Rows`, one `Row` per Size control (`Control`), read by `rows`), each control one fixed step each way through its slider's own mapping (`size_controls::SliderMapping`, the one the controls apply), in the order of tests of its spec's section 3.3: a side's `Words`, or its `Change` (the stepped design's hot-low, its change in N·m and as a fraction of the design shown's, the Outside diameter and Overall length changes, whether the calibration factor changes, the counted checks whose level flips (`Flip`, over `counted_guides` at the panel's shaft-clamp pause) and the failing checks' quantities the step moves (`MarginChange`, from `guidance::check_terms`, drawn where `shows`)); `lines(auto)` draws it, a row's two N·m at one precision and the row Auto holds as `held_line`, every text a named constant (`CARD_HEADING`, `CAPTION`, the three Auto lines, the side words, `JUMP` and `CUP_JUMP`, `CALIBRATION_CHANGE`, the four one-line forms and the flip words; `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` also end the Size tab's exploration lines); `side_words` (sizing's validity rule, every row's, the keyway left out on the count row: decision D-1) and `count_step` (two magnets each way: checked, even, inside the slider), both moved from the N-2 and N+2 lines that the count row replaced; `STEP_SLACK_MM`, the one slack of the range test and the jump rules: the overlapping magnet length on the Overall length row, the cup's diameter on the Outside diameter row since the owner's FO-2), `sizing.rs` (`SizingMode`, `held_by_auto`, `SizingState` (the mode and the free variable; the target is `metal.required_min_Nm`), `SizingRunner`: debounced inverse sizing over a search range; `sizing_range`, the range Auto searches, one rule for the panel and the torque card's test table, slice 3), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell; the Size tab's one number format, slice 3 decision 10B: `fixed` (a number to a set number of decimals, with no + added), and `signed_fixed`, `signed_change` and `change_decimals` (a change to the fewest decimals, from its own (`TORQUE_DECIMALS`, two, for N·m: the readout's gap and the torque card), that keep its true sign and read a digit that is not zero, at most `CHANGE_EXTRA_DECIMALS` more: `decimals_with_a_digit`, the one cap rule, which the torque card's `shows` reads too), `shared_change_decimals` (a torque card row's two N·m at one precision), the one-decimal `signed_percent`, the format's bound since the owner's FO-4 at the slice-3 STOP (a number whose fixed form would need more than six whole digits, `MOST_WHOLE_DIGITS`, reads four significant digits with its sign, as the readouts write values, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row: `TorquePrecision`, the one precision `shared_torque_precision` finds for a row, the readout's gap a row of one, which `signed_torque_change` draws for a torque card side and the readout's margin and `torque_shortfall` for its shortfall's size, which is drawn without a sign), and `fewest_decimals_apart`, the guidance's search for more decimals), `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries; `mark_ui`, the brand mark before the title; `tab_ui`, a tab of the centre's row, underlined in the accent when it is the active one; `quiet_text`, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes; `mark_color`, the colour of the changed dot and a trace's marks), `test_support.rs` (headless egui helpers and the shared test designs, the torque card's among them with their one table, `card_designs()`; tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests; on the web it writes the settled design into the page's address), `starting_design` (the design it opens on and hands the panel as its baseline), `address_of` (the page address that stands for a design), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the calculator's own theme, "Charcoal and crimson": the token table and the visuals built from it, forced dark), `app/fonts.rs` (the app's font: a cut of Inter Regular with tabular digits, ahead of egui's built-in fonts), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
```

Replace:

```markdown

Since BL-094 every result is a selection too: a click on a dimension tag of the drawing (its line
```

with:

```markdown

Since BL-096 the left column no longer changes with the centre tab (design 4.3). On every tab it
is the Size column: the six size controls, then two folds, **Detailed inputs** (the order toggle,
the filter box, the Key design group and every input group, now their only home) and
**Assumptions**. Both are shut at first, drawn with no indent and scroll with the
column (BL-091); a fold opened on one tab is open on the next. An opened fold is no wider than the
inputs side was: 320 points in egui's default font; in the app's own font the column still grows by
10 to 12 points when a fold first opens, and more with a filter (BL-099). The tabs "Design inputs" and
"Assumptions" that swapped the side are gone. While a trace is on, its banner and its **Clear
trace** button sit at the column's foot on every tab, the opening one too. A fix link on the
Outside diameter still opens the Size and torque tab and scrolls the column back to its slider.
Tab through the size controls selects nothing while every size is on its slider's step: the Minimum
OD's slider is handed the value its box shows, so the focus leaving its box changes nothing. A size
off its step is still rewritten when the focus leaves its box (Auto turned off leaves one: BL-097),
and a row that takes the focus below the window's foot is not scrolled into view (BL-100). The design's nothing-lost table (8.2, test 9)
is one test, `nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home`:
the old top bar's actions, every centre view, every shown note in Learn, every dashboard row's
home, and every input and assumption from every tab.

Since BL-094 every result is a selection too: a click on a dimension tag of the drawing (its line
```

Replace:

```markdown
  grade or not). Each row shows a dot when changed from the default, a
  reset button, and a tooltip with help (the overlay's where it has a row, `plain_text::help_for`), slider range and default (with Show spreadsheet details on, also the original spreadsheet note where the help replaces it, the path and the workbook cell). A row's label is its on-screen name (`plain_text::label_for`, spec 3.5: Ring radius, Magnets per ring, Magnet face gap, Required torque, Production variation allowance); the exports keep the original labels. A tab
  row above them switches to the **Assumptions** view (decision M43-7). A click on a row's label
  selects its input (since BL-092), and the link "Highlight the results this drives" at the top
```

with:

```markdown
  grade or not). Each row shows a dot when changed from the default, a
  reset button, and a tooltip with help (the overlay's where it has a row, `plain_text::help_for`), slider range and default (with Show spreadsheet details on, also the original spreadsheet note where the help replaces it, the path and the workbook cell). A row's label is its on-screen name (`plain_text::label_for`, spec 3.5: Ring radius, Magnets per ring, Magnet face gap, Required torque, Production variation allowance); the exports keep the original labels. The
  **Assumptions** view (decision M43-7) is the fold under the Detailed inputs; until BL-096 a tab row above the inputs switched to it. A click on a row's label
  selects its input (since BL-092), and the link "Highlight the results this drives" at the top
```

Replace:

```markdown
  drive torque reaches none, and its banner says so) is framed wherever it is drawn, in the
  panel's mark colour (`look::mark_color`, the host's link colour), with a banner and a Clear trace button at the foot of the inputs side (so a
  trace moves no row above it); a second click on the link ends it, and the locked free
```

with:

```markdown
  drive torque reaches none, and its banner says so) is framed wherever it is drawn, in the
  panel's mark colour (`look::mark_color`, the host's link colour), with a banner and a Clear trace button at the foot of the left column, on every tab (so a
  trace moves no row above it); a second click on the link ends it, and the locked free
```

Replace:

```markdown
  term, in the equation (the H under a Σ included) or the list, drills
  into it; an input term instead opens its group (or the Assumptions view), scrolls its row into
  view once and frames it until another equation or term is opened or the panel is closed
```

with:

```markdown
  term, in the equation (the H under a Σ included) or the list, drills
  into it; an input term instead opens the fold of the left column that holds its row (Detailed inputs and its group, or Assumptions), scrolls its row into
  view once and frames it until another equation or term is opened or the panel is closed
```

Replace:

```markdown
  for equality).
- **Assumptions** (panel.rs; spec Addendum A3): the inputs side's Assumptions view lists the 14
  assumptions (`engine::assumptions::ASSUMPTIONS`), each with its changed-from-default dot, its
```

with:

```markdown
  for equality).
- **Assumptions** (panel.rs; spec Addendum A3): the left column's Assumptions fold lists the 14
  assumptions (`engine::assumptions::ASSUMPTIONS`), each with its changed-from-default dot, its
```

Replace:

```markdown
through that lesson link (`lesson_link_opened`: the Explain panel shows "What to change" and the closed "Where this
comes from" fold), hovers the first Key design row on the Geometry tab (the tooltip shows the help and its default
and no path or spreadsheet cell, the Design menu's "Show spreadsheet details" toggle being off, as `details_toggle`
```

with:

```markdown
through that lesson link (`lesson_link_opened`: the Explain panel shows "What to change" and the closed "Where this
comes from" fold), judges the left side on the Geometry tab (`same_column`, since BL-096: the size controls over the two shut folds), opens the Detailed inputs fold, hovers the first Key design row (the tooltip shows the help and its default
and no path or spreadsheet cell, the Design menu's "Show spreadsheet details" toggle being off, as `details_toggle`
```

**`magcoupling-rs/src/gui/trace.rs`**

Replace:

```rust

/// The start of the trace's banner, at the foot of the inputs side.
pub const TRACING: &str = "Tracing";
```

with:

```rust

/// The start of the trace's banner, at the foot of the left column.
pub const TRACING: &str = "Tracing";
```

Replace:

```rust

    /// The banner at the foot of the inputs side: what is traced and how far it reaches,
    /// counting only the paths shown while the shaft-clamp section is paused (`clamps_paused`,
```

with:

```rust

    /// The banner at the foot of the left column: what is traced and how far it reaches,
    /// counting only the paths shown while the shaft-clamp section is paused (`clamps_paused`,
```

- [ ] **Step 2: Check.** `git diff --stat` lists the ten files and no other; `cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check` prints nothing (`trace.rs`'s two comments are the step's only lines of code); `python -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ['docs/ai/backlog.yaml', 'docs/ai/02-system.yaml', 'docs/ai/03-structure.yaml', 'docs/ai/04-memory.yaml']]"` prints nothing; `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3` still reads `1119 passed`.

- [ ] **Step 3: Commit** the ten files: `docs(magcoupler): slice B stage 2 part 2b-ii-c docs sweep: README, docs/ai, the register, owner questions 60 to 65, the design's header, BL-096 in progress (BL-096 T5)`.

### Task 6: Release: the full gate, gui-smoke, the reviews, one fix wave, the merge and the push

**Model:** `fable` controller; fresh-context `fable` reviewers; `haiku` at low effort for git; the smoke workflow's own agents on `sonnet`.

**Files:**
- Modify: `docs/ai/backlog.yaml` (Step 7); create `docs/analyses/2026-11-25-magcoupler-slice-b2p2bii-c-review.md` (Steps 4, 5 and 7).

- [ ] **Step 1: Preconditions.** Tasks 1 to 5 are committed, each reviewed; `git status --short` shows only the owner's untracked press analysis; `git diff main...HEAD --stat -- magcoupling-rs/src/engine magcoupling-rs/tests reference linkage-sim-rs/src linkage-sim-rs/web` is empty (no engine file, no linkage file).

- [ ] **Step 2: The full gate.** From PowerShell at the worktree's root:

  ```powershell
  & 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/gate.sh --full
  git checkout -- docs/chebyshev_lambda
  ```

  Expected: `GATE PASS`. A red gate goes back to the controller with the gate's output; nothing is merged.

- [ ] **Step 3: gui-smoke, from the branch's own script.** Build the web calculator from the branch (`linkage-sim-rs/scripts/build_magcoupling_web.sh` through Git Bash), serve `linkage-sim-rs/web/`, and run the `gui-smoke` workflow from this branch's `.claude/workflows/gui-smoke.js` with `{"frame_check": true}`. Expected: `passed: true` for the calculator with `same_column: true`, `tooltip_plain: true` and `input_entry: true` (the fold's click is the step these three rest on), and a p95 under one 60 fps frame. Keep the screenshot of the Geometry tab with the fold open.

- [ ] **Step 4: The text review** (fresh context, `fable`; the design's 8.3). There is no new on-screen string. It checks that every text the left column draws on a tab other than the opening one is a text the opening tab's column or the old inputs side drew at `main` (the size controls' labels, the folds' headers, the order toggle, the filter's hint, the groups' headings, the banner, Clear trace), that the one text gone is the tab "Design inputs", and that no engine text changed (`git diff main...HEAD -- magcoupling-rs/src/engine` is empty). It rules on four things and records what it decides: whether the column's heading "Size and torque" reads well over the column on a tab that is not Size and torque, the design's sketch heading it "Size" (decision D-85, owner question 65); whether "Auto: ... set by the solver (Size tab)" and "Open the Size tab" still read true under the size controls (decision D-77, owner question 62); whether "Set by Auto (Size tab)" under a locked row does; and whether the banner at the foot of the opening tab reads well with the folds shut and no row framed (decisions D-74 and D-79, owner questions 60 and 63). A change the owner should weigh goes to `docs/analyses/2026-11-18-owner-questions.md` as an open question with its conservative answer, not shipped silently. Its findings join Step 5's fix wave. Record it in `docs/analyses/2026-11-25-magcoupler-slice-b2p2bii-c-review.md` (committed in Step 7).

- [ ] **Step 5: The independent whole-branch review** (fresh context, `fable`, three lenses in parallel, each with a skeptic on every finding that is not a suggestion). Give each `git diff main...HEAD`, the design, this plan and Step 3's screenshot, and none of the builders' notes.
  - **Does every rewritten test keep its subject?** For each of the 47 tests under "Files": the assertions it had are still made, on the rows it read; none passes only because the size controls are scrolled out of view or a fold is shut. The seven tests listed there as needing the harness's scroll to reach the fold's row are not findings.
  - **Is anything lost or moved without a test?** The left column on every tab against the inputs side at `main`: every row, the trace's frames and banner, the focus of a row asked for, the keyboard's Tab order through the column (Review Focus 3), the layout at 1024 x 700, 1280 x 800 and 1920 x 1080 with the folds shut and open (render them). Three things are known and filed, on `main` too, and are not findings of this part: in the app's own font the left column is 330.1 points wide, not 320, once a fold has opened, and wider with a filter (BL-099); with Detailed inputs open about half of the column's Tab stops land below the window's foot and the column does not follow (BL-100); Auto on, Auto off, then Tab rewrites the size Auto left (BL-097). Weigh whether, with Detailed inputs open, the Assumptions header reads as a ninth input group (decision D-75, owner question 64).
  - **The code.** `inputs_ui`'s bottom-up layout, `column_to_top`, `show_unindented`, the Minimum OD's shown value (an arrow key, a typed value, a drag, an idle frame), dead code left by `InputsView`.
  One fix wave follows, on `fable`: each accepted finding gets its red test first. A finding that is not a defect of this part is filed in `docs/ai/backlog.yaml` with the next free id.

- [ ] **Step 6: If Step 4 or 5 changed a file, run Steps 2 and 3 again.** The merge needs a gate and a smoke run of the tree that is merged.

- [ ] **Step 7: Close the items and merge.** In `docs/ai/backlog.yaml` (the day's date; say what the reviews found):

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2b-ii-c (Tasks 1 to 4 of the plan);
```

with:

```yaml
  status: fixed
  notes: "FIXED <date>: merged (<the gate, the reviews and the smoke run in a few words>). Built on branch magcoupling/overhaul-b2-part2b-ii-c (Tasks 1 to 4 of the plan);
```

Replace:

```yaml
  status: open
  notes: "Filed 2026-11-23 from the BL-090 review (UXQ-2); not taken in BL-090's fix wave
```

with:

```yaml
  status: fixed
  notes: "FIXED <date> with BL-096 (its plan's Task 1: design_inputs_ui and filtered_inputs_ui draw no scroll area; red tests the_detailed_inputs_fold_scrolls_with_the_size_column_to_its_last_group and the_filtered_inputs_scroll_with_the_size_column_to_their_last_row). Filed 2026-11-23 from the BL-090 review (UXQ-2); not taken in BL-090's fix wave
```

  Commit it with the review record: `docs(magcoupler): BL-096 and BL-091 fixed: the reviews, the gate and the smoke run passed`. Then merge the branch into `main` with a merge commit (`git merge --no-ff`), its message one line of what the screen now does.

- [ ] **Step 8: Push, then report** (the owner's ruling of 2026-11-22: each stage goes live once the gate, the smoke run and the reviews pass, with a report afterwards; no stop). Push `main` (it deploys the site). When the `deploy-web.yml` run is green, run the smoke workflow from `main`'s script with `{"url": "https://colesorkness.com"}` and report the result; if it fails, say so plainly: the site is live with the change. The report to the owner holds: what changed on screen, in three sentences; two pictures (the Geometry tab with the Detailed inputs open, the opening tab with a trace's banner); the decisions D-71 to D-85, one line each; owner questions 60 to 65; the four findings filed (BL-097 to BL-100); part 2b-iii; what was not verified (Safari and Firefox; in a browser, the banner on the opening tab, the Assumptions fold, the Outside diameter's fix link and the keyboard's Tab through the column: the smoke opens one fold and reads one row). Say with the keyboard's Tab what is known of it: with Detailed inputs open, 27 of the left column's 56 Tab stops at 1280 x 800 land below the window's foot and the column does not follow them (BL-100), and Auto on, Auto off, then Tab still rewrites the size Auto left (BL-097).

## What this plan covers of the design

| The design | Where |
|---|---|
| 4.1, 4.3 The left column: the size controls, then the folds Detailed inputs and Assumptions, on every tab | Task 3 |
| 4.3 "The left column no longer changes with the centre tab ... the workflow groups live only inside Detailed inputs" | Task 3 (`the_left_column_is_the_size_column_on_every_tab`) |
| 4.3 "Detailed inputs (today's fold, unchanged inside)" | Tasks 1 and 3 (the fold's contents are `design_inputs_ui`'s; it loses its own scroll area and its indent, D-72 and D-75) |
| 5.2 The keyboard's focus arriving selects nothing (BL-093) | Task 2 (the Minimum OD's box; a size off its step is BL-097) |
| 5.2 A fix link selects and focuses its input | Task 3 (`column_to_top`, D-76) |
| 5.2 Tracing stays, as a link in the input's entry; the banner's place (part 2b-ii-a's D-52) | Task 3 (D-74) |
| 8.2 test 9, goal G-4: nothing lost, table-driven | Task 3 (`nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home`) |
| 8.2 test 15: the layout at 1024 x 700 | Tasks 1 and 3 (the fold's and the banner's tests run at 1024 x 700) |
| 9.1 The docs, the backlog item and the gui-smoke checklist with the slice | Tasks 4 and 5 |
| 9.4 Release | Task 6 |
| Not in this part | Part 2b-iii: the Plots tab and the look. BL-097 to BL-100: filed, not fixed. |
