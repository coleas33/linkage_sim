# Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-a (the Plots tab): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ] `) syntax for tracking.

**Goal:** The five plots sit under one tab, "Plots", with a picker (the design's decision C-9). The tab row becomes Size and torque, Geometry, Plots and Results table (the Clamp tab before the table while that section shows), one row from 1024 x 700 up, where nine tabs took three rows at 1024 x 700 and two at 1280 x 800 in the app's own font. Under the Plots tab a row of the five plots' names, in the order their tabs had, marks the plot shown; a click on a name shows that plot, and the tab opens on the plot shown last. Red in a plot means a failing check and nothing else (decision C-5): the requirement lines, the governing limit and two marks leave the failing red they took whatever the design's verdict (stage 1's review, finding C-3), and red stays on a sweep row that does not fit. Each axis draws a tick number at full strength or not at all (finding E-7). The gui-smoke run judges the tab row and the picker. Nothing that is on screen today is lost (the nothing-lost table).

**Architecture:**
- **One tab, five views.** `CentreView` keeps a view per plot: `CentreView::ALL` and `CentreView::tabs` still list every view, so every test that walks the views still reaches each plot. `CentreView::tab` is the text of the tab that shows a view: `plots::PLOTS` for each plot, the view's own name for the others. `centre_ui`'s tab row draws the Plots tab once, at the first plot's place; a click on it opens `MagcouplingPanel::plot`, the plot shown last (view state, set from the view shown at the top of each frame).
- **The picker.** Over a plot, `view_ui` draws the five `PlotKind` names as `selectable_value` in a wrapping row (the widget of the inputs' order toggle and of the results table's order), the plot shown picked; a click shows its plot in the same frame. No new name: each is the tab text it was.
- **Red.** `plots.rs` draws the required torque and the sweeps' required floor in the strong text colour, slip heating's governing limit in the violet the torque plot already gave its own, the time to the limit in that violet and the pull-out point in the strong text colour. Red is left to a sweep row that does not fit (a row whose design fails the fit check) and its legend entry.
- **Tick numbers.** `plot` builds each axis with `AxisHints::label_spacing` set to one point, egui_plot's own threshold (60 points apart on the horizontal axis, 20 on the vertical): every number egui_plot drew is drawn at full strength, none faded on the way.
- **The tests' harness.** `Harness::tab` reaches a plot through the Plots tab and its picker; the app tests do the same (`show_view`). No assertion about behaviour that stays is weakened.
- **Unchanged:** the engine, the exports, every plot's series, readouts, legends and axis labels, each plot's own zoom (egui_plot keeps it under `PlotKind::id`), the left column, the right column, the opening tab, the Geometry, Clamp and Results views, the Equation panel, the linkage app.

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32, egui_plot 0.33; the headless egui harness (`magcoupling-rs/src/gui/test_support.rs`, the panel tests' `Harness`, the app tests' `frame_on`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. This part is its decision C-9 ("The five plots sit under one 'Plots' tab with a picker. The tab row becomes Size and torque, Geometry, Plots, Results"), section 4.4 ("Plots. A picker of the five plots (same names, same order), then the plot with its readout line, as today"), decision C-5 ("Red means a failing check and nothing else"), goal G-4 and test 9 of section 8.2 (every `CentreView` reachable), the text rules of section 8.3 and the docs and release rules of section 9. Executors read the spec's sections 1.3 (C-5, C-9), 4.4, 8.2, 8.3 and 9 beside this plan. Part 2b-i's plan (`docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`, "What follows", part 2b-iii item 1) sets the scope, and slice B stage 1's plan (`docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`, "What follows" items 5 and 11, and its table "Declined in this stage") holds the two review findings, C-3 and E-7. Backlog item: **BL-104** (`docs/ai/backlog.yaml`, filed with this plan).

**How this plan was made.** The part was built once as a prototype on a copy of `main` at `65d9fe7`, one commit per task and one for Task 1's red state, then BL-104 filed and the docs swept. Every block below was generated from those commits' diffs (Task 1: the red step is its red state, the `PLOTS` name with the new test, and the green step the rest; Tasks 2 and 3: the red step is the test hunks, the green step the rest; Task 4: the red step is the prompt's guard in `app.rs` and the verdict test, the green step the prompt), then applied, in this plan's order, with the apply tool, to a fresh copy of `main` with BL-104 filed, in a target directory of its own. The fix wave of the plan's review (findings RC-1 and UX-1 to UX-5) re-made the prototype's states with three more edit scripts, generated the blocks again and ran this replay again, in a new target directory; the counts below are that run's. Each block applied once; each red step failed as its "Expected" says; the result equals the prototype file for file (line endings aside) and passes `cargo test` with and without the app feature (1141 lib tests with it against 1137 on the base), both native clippy runs after each task, both wasm32 clippy runs, `cargo fmt --check` and the gui-smoke verdict test (`all 85 checks hold`). Each count after a task is the replay's. Not run for this plan: gates 1 to 3 (linkage-sim-rs, untouched), gate 12 (the Python oracle), the release WASM build, the browser and the pictures. Task 6 runs them.

**Where this part applies.** On `main` at `65d9fe7` with this plan's commit (the plan and BL-104), and nothing else that touches `magcoupling-rs/`, `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` or Task 5's nine files. If a block does not match, stop and report: the controller regenerates the blocks (the planner's tools are listed under "Before Task 1"), it does not adapt a block in place.

## Why part 2b-iii is cut, and what follows

Part 2b-i's plan listed three items for part 2b-iii: the Plots tab with its picker and the plots' two review findings; the controls' outlines, a SemiBold face and the page shell's loading line; and three Equation panel items, BL-086, BL-088 and BL-089. They are three subjects a reviewer holds apart, and none needs another: the Plots tab is a view's navigation and the plots' drawing; the look is a new theme colour, a second font file and the page shell; the Equation panel is its texts' wrapping, a pure move out of `explorer.rs` (3,110 lines) and a text no user can see. Two of the look's items cannot be checked the way this plan's blocks are: the SemiBold face is a binary font file made by the committed font script, which a text block cannot carry, and the loading line is a new on-screen string in the page shell that only a browser shows. So part 2b-iii is cut in three (decision D-86, register row A-47), and this plan is the first, in full. Its four code tasks are short, Task 1 the longest (about 1,020 plan lines: the tab row, the picker, the tests that walk the views and the keyboard's test), and one reviewer holds each with its list under "Files".

**What follows, each with its own plan and backlog item:**

**Part 2b-iii-b (the look):** the controls' outlines (stage 1 review E-4 and R-6: an unticked checkbox's box, a slider's rail and a value box at rest are 1.1:1 to 1.2:1 on the panel; no token gives a 3:1 outline without other damage, so one more colour, about `#626973`, if the owner agrees), a SemiBold face for the title and the headings (S-1: Inter SemiBold, cut by the committed font script, about 30 KB; the plan's task runs the script and commits its output), the page shell (S-4: `lang`, `theme-color`, `color-scheme: dark` and one muted loading line while the wasm file loads, in `linkage-sim-rs/web/tools/magcoupler/index.html`; a new on-screen string, a text review and a browser run).

**Part 2b-iii-c (the Equation panel):** BL-086 (the term rows, their tags and the walk bar's step line cut at the panel's edge during the Start here walk; the app test widened to every text of the panel at the three window sizes), BL-088 (the note drawing out of `explorer.rs` into its own module, a pure move), BL-089 (`EMPTY_TEXT` can no longer be seen: the text, its test and its guard rows go, or a way to open the panel empty comes back). BL-101's Equation panel part (its nested sideways scroll areas do not follow the keyboard's focus) is the same panel: that plan takes it if it holds it, else leaves it open.

Not touched by any of the three: BL-083, BL-084 and BL-087 (physics: they wait for the fable decider, back 2026-12-01), BL-095, BL-102 and BL-103, and BL-105 (filed by this plan's Task 5: at 1024 x 700 and 1280 x 800 four plots' horizontal axes carry one or two numbers, or none, as on `main`; decision D-91 keeps egui_plot's threshold, and no one constant fixes it).

Declined in part 2b-iii: stage 1's "What follows" item 11 holds a second thing beside E-7, the second part of suggestion S-5 (the real arrow and minus signs for "->" and "-1 mm" in the welcome line and the torque card). The text rule the panel's guard checks bans the real arrow (`plain_text.rs`: "write -> and <="), so the arrow needs that rule changed first, a decision of its own, and the minus sign alone is a change of on-screen words that no part of 2b-iii needs. The ASCII signs stay; no plan holds it.

## What the prototype showed

1. **The tab row fits one row.** In the app's own theme and font (`MagcouplingApp::new`, which applies `theme::apply` and `fonts::install`), at a pixel ratio of 1, `main`'s nine tabs (eight with the shaft-clamp section paused) took three rows at 1024 x 700, two at 1280 x 800 and one at 1920 x 1080. After this part the row is Size and torque, Geometry, Plots and Results table on one row at all three, ending at x 665 of a centre region that ends at 716 at 1024 x 700; every view but a plot gains two rows of height at 1024 x 700 and one at 1280 x 800. In egui's default font, the panel tests' font, the row with the Clamp tab shown is one row at 1024 x 700 too (Task 1's test).
2. **The picker.** In the app's font the five names are one row at 1280 x 800 and 1920 x 1080 (ending at x 880 of 972 at 1280) and two at 1024 x 700 (three names, then two), so a plot's view starts as low as it did at 1024 x 700 and 1280 x 800. No text is cut on any tab or any plot at the three sizes (the app test `no_text_is_cut_in_any_tab_in_the_app_s_own_font` covers the first two; the prototype measured the third).
3. **A picked plot shows in the frame of the click.** `selectable_value` on a copy of the view's `PlotKind` writes the pick before the plot is drawn; the picked mark, drawn before the click is read, follows in the next frame, so the test reads the fill from the frame after the click. Task 1's test holds the first: a plot's egui_plot memory, absent until the plot first draws, is there after the frame of its click.
4. **Two tests leaned on the wrapped tab row.** With the row on one line the Geometry tab's callout list held every row at 1024 x 700 under the open Equation panel at its starting height, and `the_centre_s_scrolling_views_follow_the_keyboard_s_focus` failed its own guard ("no stop was out of view"); the Equation panel dragged 40 points taller, as a person can, gives it rows past the list's foot again. `the_active_tab_is_underlined_in_the_accent_and_no_other` checked the underline of a tab on the row's second line at 1280 x 800; the row wraps now only in a window narrower than the app supports, such as a host's (the panel can be drawn in an egui window the user sizes, as `the_panel_works_inside_an_egui_window` draws it), so that check runs at 860 x 800.
5. **Red.** On the passing design the only red left in the plots is the pole sweep's rows that do not fit and that legend entry. egui_plot's legend in the plot's right corner sets each name left of its icon, so the test finds an icon's name on either side.
6. **A one-point label spacing.** egui_plot skips a number at or under the spacing's lower end and fades one between its ends with `remap_clamp`, which returns full strength for a value at or past the upper end before its zero-width check (emath 0.32.3), so a one-point range never meets that check's debug assertion. The numbers drawn are those egui_plot drew before; on `main`, Task 3's test finds 800 of them faded over its 25 window sizes, five plots and two visuals.
7. **rustfmt and line endings.** `cargo fmt` may rewrite a file it changes with LF line endings on this machine; git's `autocrlf` takes them back at the commit, and the apply tool reads each file's endings as it finds them.

No pictures were drawn for this plan: Task 6's review renders the Plots tab with each plot picked at the three window sizes, and a failing design's plots.

## Global Constraints

- **No engine number moves** (the design's 2.3). No file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` changes.
- **The linkage app is not touched** (O-1). No file under `linkage-sim-rs/src/` or `linkage-sim-rs/web/` changes.
- **The panel sets no theme and no font** (2.2, 6.3). Code under `magcoupling-rs/src/gui/` reads `ui.visuals()`.
- **One new on-screen string:** the tab's word "Plots" (`plots::PLOTS`), the design's own (C-9). It passes the plain-words guard (Task 1's test) and the glyph guard (`every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` collects it). The picker's five names are the tab texts they were. No text leaves the screen.
- **Crimson is never text, never on a value, never in the check list** (C-5), and **red means a failing check and nothing else** (C-5): after Task 2 no plot draws a reference line or a mark in the failing red.
- **View state stays view state** (5.1): the plot shown and the plot the Plots tab opens are in no design file, share link or undo step.
- **A hover selects nothing** (C-2), and neither does the keyboard's focus arriving (BL-093); a pick in the Plots tab selects nothing either: it changes the view, not the Explain panel.
- **The exports do not change** (12.1).
- **The blocks.** Each Replace block's old text occurs exactly once in its file at the moment its step applies it. Apply a step's blocks in the order given, top to bottom. Keep each file's line endings. If an old text is not found, or is found twice, stop and report: do not adapt the block. Line numbers are not given because earlier steps move them. `C:\Users\Cole\source\repos\linkage_simulation\.superpowers\sdd\2026-11-22-magcoupler-overhaul-b2\apply_plan.py` (gitignored, in the main checkout; stage 1's tool, unchanged) applies one step's blocks this way: `python <it> <this plan> <task> <step>`, from the worktree's root. The replay used it.
- **TDD.** Red, then green. Every red step here fails at run time with the message its "Expected" names.
- **Before each commit**, from the repository root:
  ```bash
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
  ```
  `cargo fmt` changes no text of a task's files (the blocks are formatted); it may rewrite a file it reads with other line endings, which git's `autocrlf` undoes at the commit. After it, `git status --short` shows only the files the task names. If it shows another, stop and report.
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-27-magcoupler-overhaul-b2p2biii/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Shell.** On this machine `bash` typed in PowerShell is WSL, which fails on the checkout's line endings. Shell scripts run through Git Bash: `& 'C:\Program Files\Git\bin\bash.exe' <script>`. Python scripts are written with the Write tool, not a here-document. Do not edit a source file with `sed -i` from Git Bash: it rewrites the file's line endings.
- **Never run `cargo fmt` inside `linkage-sim-rs`.** `magcoupling-rs` is fmt-clean; keep it so.
- **Docs ship with the code** (repository rule): Task 5 sweeps the README and `docs/ai/`, the register, the owner-question list and the design's status line.

## Model per task

CLAUDE.md section 5: `fable` is the controller and the decider. BL-104 is `risk: mechanical` (decision D-94): the part shows no new number and makes no new physical claim, and its one new on-screen string is the design's own word. So the tasks that transcribe the plan's exact blocks run on `sonnet`, and every review and the release run on `fable`; while `fable` is out (back 2026-12-01) the decider's steps inherit the session model, and the report says which model ran them. A `sonnet` task that ends blocked, leaves the gate red or is rejected in review is redone on `fable` (escalate, never downgrade).

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | The five plots share one Plots tab with a picker (`CentreView::tab`, the tab row, the picker, the plot shown last); the harness and the app tests reach each plot through it | `sonnet` |
| 2 | Red in a plot only on a sweep row that does not fit | `sonnet` |
| 3 | Each tick number at full strength or not drawn | `sonnet` |
| 4 | gui-smoke judges the tab row and the Plots picker (`plots_picker`) | `sonnet` |
| 5 | Docs sweep | `sonnet` |
| 6 | Release: the full gate, gui-smoke, the text review, the independent whole-branch review, one fix wave, the merge and the push | `fable` controller (the session model while `fable` is out); fresh-context `fable` reviewers (likewise); `haiku` for git; the smoke workflow's own agents on `sonnet` |

Each task's diff is reviewed by a fresh-context reviewer before the next task starts (`sonnet` for these mechanical diffs, CLAUDE.md section 5); it checks the diff against the task's blocks and nothing else. A task that ends blocked or leaves a test red stops the build and goes back to the controller.

## Review Focus

Eleven inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named.

1. **The tab row at the smallest window with every tab.** Task 1: `the_five_plots_share_one_tab_whose_picker_shows_each_plot` (at 1024 x 700 with the Clamp tab shown, in egui's default font: the five tab texts on the opening tab's row, in order).
2. **The picker on two rows.** At 1024 x 700 its names wrap; each, on either row, takes its click. Task 1: the same test clicks all five at 1024 x 700 and reads them in reading order.
3. **The Plots tab clicked while a plot shows.** It keeps the plot, and from another view it opens the plot shown last. Task 1: the same test.
4. **A pick is view state.** No edit, no selection, no undo step. Task 1: the same test.
5. **Each plot still reached by the walks over the views.** Task 1: `Harness::tab` and the app's `show_view` reach a plot through the picker, so `every_shown_text` (the glyph guard), the nothing-lost test, the left column's tests and the app's `no_text_is_cut_in_any_tab_in_the_app_s_own_font` and column-width tests still visit all five.
6. **Red on a failing design and on a greyed one.** Task 2: `red_marks_a_sweep_row_that_does_not_fit_and_nothing_else_in_any_plot` (the passing design, the example design with four failing checks, a design whose high case reaches the limit, a hot day, a design the end-effect model greys).
7. **Where the red lines went.** Task 2: the same test finds the requirement lines in the strong text colour in three plots and slip heating's limit in violet.
8. **Tick numbers under a theme with a weak colour** (the app's). Task 3: `every_tick_number_is_drawn_at_full_strength_or_not_at_all` (egui's visuals and a theme's weak colour, 25 window sizes, every plot).
9. **The smoke prompt's tab row as the app draws it.** Task 4: the guard builds the list from `CentreView::tabs` with the section paused, and the five names from `PlotKind::ALL`.
10. **The tab's word in plain words.** Task 1: the test runs the plain-words guard and the gearbox guard on `PLOTS`.
11. **The keyboard through the tab and the picker.** Focus arriving picks nothing, Enter on the tab, Space or Enter on a name, the picker's two rows at 1024 x 700. Task 1: `the_keyboard_reaches_the_plots_tab_and_picks_a_plot_by_space_or_enter`.

## Decisions this plan takes

Each is a reading of the design where it was open, numbered after part 2b-ii-c's D-71 to D-85 (one stage, one list). The owner may overrule any of them; Task 6 lists them in its report. The ones that need a row in the register have one (Task 5).

| Id | Decision | Why |
|---|---|---|
| D-86 | Part 2b-iii is cut in three: 2b-iii-a, the Plots tab (this plan: C-9, the stage 1 review's C-3 and E-7, gui-smoke's Plots check); 2b-iii-b, the look (E-4 and R-6, S-1, S-4); 2b-iii-c, the Equation panel (BL-086, BL-088, BL-089, and BL-101's Equation panel part if that plan holds it). Register row A-47. | Three subjects that do not need each other (see "Why part 2b-iii is cut"); smaller releases are the conservative answer to Q-4 (A-21, A-23, A-25, A-29). The face is a binary file a plan's blocks cannot carry, and the loading line needs a browser. |
| D-87 | The picker is a row of the five names under the tab, the plot shown marked (`selectable_value`, the widget of the inputs' order toggle and of the results table's order), wrapping at 1024 x 700; not a drop-down. Register row A-48; owner question 70. | Each plot stays one click from the Plots tab with its name in view, as its tab was; a drop-down hides the names behind a click. No new text. |
| D-88 | The Plots tab opens on the plot shown last, and a click on it while a plot shows keeps that plot. View state. Register row A-51 (the builder's call, no owner question). | Returning to the plot one was reading stays one click, as its tab was; a click on the tab of the view shown changes nothing, as every other tab's. |
| D-89 | The last tab keeps its text, "Results table"; the design's C-9 writes "Results". Register row A-49; owner question 71. | No on-screen word changes that the scope does not need. |
| D-90 | In a plot red marks only a sweep row that does not fit and its legend entry. The required torque and the sweeps' required floor are drawn in the strong text colour, the governing limit in violet in both plots that draw it, the time to the limit in violet, the pull-out point in the strong text colour. Register row A-50; owner question 72. | The approved rule C-5 stands (the stage 1 review named two ways: the lines leave the red, or C-5 is narrowed). No colour is new to the plots. The pull-out point and the time to the limit were red on every design too; finding C-3 did not name them, and they follow the same rule. The cost: a requirement line no longer looks like a warning. |
| D-91 | Each axis keeps egui_plot's own threshold (60 points apart on the horizontal axis, 20 on the vertical) and draws every number past it at full strength (`X_TICK_SPACING`, `Y_TICK_SPACING`). | The least change: the same numbers as before, none faded; the upper ends (80 and 30) would drop the numbers between, and no number appears that was absent. |
| D-92 | Owner question 56 keeps its default: a plot's readout still opens the Equation panel on a click (part 2b-ii-c's plan left it to this plan to weigh). | Nothing in the design's 4.4 asks for it ("the plot with its readout line, as today"), and the question stands open with that default. |
| D-93 | The gui-smoke prompt gains `plots_picker`, judged after the lesson link and before the Geometry tab on the same page: a click on the Plots tab, the tab row and the five names judged, then a click on Slip heating and its legend's "High case". | C-9 in a browser with two clicks and no new page; the Geometry step that follows starts from the Plots tab as it started from the Size and torque tab. |
| D-94 | BL-104 is `risk: mechanical`; its build tasks run on `sonnet`, its reviews and release on `fable` (the session model while `fable` is out). | No new number or physical claim; the one new on-screen string is the design's own word (the design's header). |

## Nothing lost

Every text and action this part moves, and where it is after. Anything not listed (the top bar, the left column, the right column, the opening tab, the Geometry, Clamp and Results views, every plot's series, readouts, legends and axis labels) does not move. The "Test" column names the test that holds the new home. The design's own table (8.2 test 9) is the test `nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home`, whose "every centre view has its tab" now reaches each plot through the picker.

| Today | After this part | Test |
|---|---|---|
| Five tabs, one per plot, in the tab row ("Torque vs temperature", "Gap sweep", "Pole sweep", "Slip heating", "Torque vs rotation") | One tab, "Plots"; the five names, the same words in the same order, in its picker | `the_five_plots_share_one_tab_whose_picker_shows_each_plot`; the nothing-lost test (every `CentreView`) |
| Each plot one click from any view | The plot shown last one click from any view (the Plots tab); another plot one more click (its name). A cost: from the Geometry tab to a plot that was not the last one shown is two clicks where it was one | `the_five_plots_share_one_tab_whose_picker_shows_each_plot` |
| The tab row on three rows at 1024 x 700 and two at 1280 x 800 (the app's font) | One row at 1024 x 700, 1280 x 800 and 1920 x 1080 | `the_five_plots_share_one_tab_whose_picker_shows_each_plot` (egui's font); the app's `no_text_is_cut_in_any_tab_in_the_app_s_own_font` |
| The tab shown marked by the accent's underline | The Plots tab marked so on every plot; the plot shown marked by the selection's fill in the picker | `the_active_tab_is_underlined_in_the_accent_and_no_other`; `the_five_plots_share_one_tab_whose_picker_shows_each_plot` |
| The required torque, the required floors and slip heating's limit as red dashed lines, the pull-out point and the time to the limit as red dots, with red legend dots | The same lines and dots, with their legend names, in the strong text colour (the requirement lines, the pull-out point) and violet (the limit, the time to it) | `red_marks_a_sweep_row_that_does_not_fit_and_nothing_else_in_any_plot` |
| Tick numbers fading in as the ticks spread (down to 1.3:1) | The same numbers at full strength | `every_tick_number_is_drawn_at_full_strength_or_not_at_all` |
| The keyboard: Tab passed nine tabs (eight with the section paused) | Tab passes four (five), and on a plot the picker's five names; Space or Enter on a focused name picks it (egui 0.32.3 takes either key on a focused widget for a click, `Context` source). Nothing is unreachable | `the_keyboard_reaches_the_plots_tab_and_picks_a_plot_by_space_or_enter` (Tab to the tab and through the five names with no view changed, Enter on the tab, Space and Enter on a name, at 1024 x 700 and 1280 x 1024; the count of presses is not held) |
| The gui-smoke run visits no plot | It opens the Plots tab, judges the row and the picker and picks Slip heating | the app's `the_smoke_test_share_link_opens_its_design`; the verdict test |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/panel.rs` | One test (1, red). `CentreView::tab`, the field `plot`, the tab row, the picker in `view_ui`, the module's and `CentreView`'s comments; `Harness::tab` reaches a plot through the picker; three tests and two comments of a fourth edited; one test of the keyboard added (1, green) | 1 |
| `magcoupling-rs/src/gui/plots.rs` | `PLOTS` (1, red); the module's and `PlotKind`'s comments (1, green); the colours of three lines and two marks, the module's comment, one test (2); `X_TICK_SPACING`, `Y_TICK_SPACING`, `plot` with its two axes, one test (3) | 1, 2, 3 |
| `magcoupling-rs/src/gui/look.rs` | `ACCENT`'s comment | 2 |
| `magcoupling-rs/src/app.rs` | The app tests reach each plot through the picker (`show_view`) and name the Plots tab as a quiet tab (1); the smoke prompt's guard (4) | 1, 4 |
| `.claude/workflows/gui-smoke.js` | `plots_picker` (schema, prompt, passed list), the description and comments | 4 |
| `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` | `CHECKS` gains `plots_picker` | 4 |
| `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`, the register, the owner questions, the design's header | The docs sweep | 5 |
| `docs/ai/backlog.yaml` | BL-104 in progress, BL-105 filed (5); BL-104 fixed (6) | 5, 6 |

**Existing tests this plan edits, and how.** No assertion about behaviour that stays is weakened. All are in Task 1's green step, beside the change that moves what they read.
- **The harness.** `Harness::tab(view)` clicks the tab of `view` in the centre region (`CentreView::tab`), then, for a plot, its name in the picker. The tests that walk `CentreView::tabs` through it (`every_shown_text`, the guard `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` reads; `the_left_column_is_the_size_column_on_every_tab`; `a_trace_s_banner_and_clear_trace_sit_at_the_left_column_s_foot_on_every_tab`; the nothing-lost test) are not edited and still visit every plot.
- **`each_plot_tab_draws_its_plot_from_this_frame_s_results`.** Its two clicks on a plot's tab become `harness.tab(CentreView::Plot(kind))`; every assertion stays.
- **`the_active_tab_is_underlined_in_the_accent_and_no_other`.** The tab that is not the active one is "Plots" (it was "Gap sweep"); the check of a tab on the row's second line draws a new panel at 860 x 800, a host's window, where the row still wraps (from 1024 x 700 up it is one row). Every assertion stays.
- **`the_centre_s_scrolling_views_follow_the_keyboard_s_focus`.** In its Geometry case the Equation panel is dragged 40 points taller after it opens (`drag_equation_panel`): with the tab row on one line the callout list held every row and the test's own guard, "no stop was out of view", failed. The walk and its assertions stay.
- **The nothing-lost test.** Two comments name the picker; its code is unchanged.
- **App tests.** `no_text_is_cut_in_any_tab_in_the_app_s_own_font` and `assert_the_column_holds` (the left and the right column's width tests) show each view through `show_view`: its tab in the centre region, then, for a plot, its name in the picker, each a click followed by the pointer leaving and two frames, as before. `a_label_is_muted_and_its_value_is_in_the_text_colour` reads the quiet tab "Plots" (it read "Gap sweep").

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` after this plan has merged, create the branch `magcoupling/overhaul-b2-part2b-iii-a` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: `test result: ok. 1137 passed; 0 failed; 6 ignored` and `all 83 checks hold`. Another count means `main` is not the base the blocks were checked on: stop and report.

- [ ] **Step 3: Check the base.** `git log --oneline 65d9fe7..main -- magcoupling-rs .claude/workflows/gui-smoke.js linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs docs/ai docs/analyses/2026-11-10-conservative-assumptions.md docs/analyses/2026-11-18-owner-questions.md docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` lists this plan's commit (it touches `docs/ai/backlog.yaml`) and nothing else. If it lists more, stop: the controller regenerates the blocks on that `main` with the planner's tools (gitignored, in the main checkout's `.superpowers/sdd/2026-11-27-magcoupler-overhaul-b2p2biii/`: `gen_blocks.py` (part 2b-ii-c's, unchanged), `frags.py`, `replay.py` and `finalize.py`, which read the prototype as a clone of `seq.bundle` named `proto` beside them (branch `proto`); `seq.bundle`, the prototype's commits on `main` at `65d9fe7`: `s1r` and `s1` Task 1's red and green states, `s2`, `s3` and `s4` the next three tasks, `s4b` BL-104 filed, `s5` the docs; the edit scripts that made most of them, `ed_lib.py`, `fmt.py`, `e1_app.py`, `e1_plots.py`, `e2_red.py`, `e2_green.py`, `e3_red.py`, `e3_green.py`, `e4_red.py`, `e4_green.py`, `e5.py`; the planner's own fixes before the replay, `f1.py`, `f1b.py`, `f1c.py`, `f23.py`, `f2doc.py` and `f5.py`, applied to the chain by `restate.py` (with `fix1.json` to `fix4.json`) and `rebuild_s1r.py`; the review's fix wave, `fw_rc1.py`, `fw_ux3.py` and `fw_docs.py`, applied by `restate.py` with `fix5.json`; and `file_bl104.py`; Task 1's green state was partly made by hand, so the bundle's commits are the states) before Task 1.

## Tasks

### Task 1: The five plots share one Plots tab with a picker

**Model:** `sonnet` (BL-104 is `risk: mechanical`; the task transcribes exact blocks).

The tab row draws one Plots tab where the five plot tabs were; over a plot the picker names the five; the Plots tab opens on the plot shown last (decisions D-87, D-88). The panel tests' `Harness::tab` and the app tests' new `show_view` reach a plot through the picker, so every test that walks the views still visits all five (the list is under "Files").

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/plots.rs`, `magcoupling-rs/src/app.rs`

**Interfaces:**
- Produces: `gui::plots::PLOTS` (the tab's word); `CentreView::tab(self) -> &'static str`; the panel's private field `plot: PlotKind`; in the app tests, `show_view(ctx, app, size, view, settle) -> egui::FullOutput`.
- Changes: `CentreView::label` is the view's name (a plot's name in the picker), as its text was; `CentreView::ALL` and `CentreView::tabs` are unchanged; `Harness::tab` clicks `view.tab()`, then a plot's name.

- [ ] **Step 1 (red): the Plots tab's test.** The blocks add `PLOTS` to `plots.rs` (a name no code reads yet) and `the_five_plots_share_one_tab_whose_picker_shows_each_plot`: at 1024 x 700 with the Clamp tab shown and at 1280 x 800 with it paused, the opening tab's row holds the tab texts in order and nothing else; a click on "Plots" opens the first plot; the picker's five names are under the row in their order; a click on each draws its plot in the frame of the click, marked (from the next frame) by the selection's fill and no other name; the tab keeps the plot on a click and, from the Geometry tab, opens the plot picked last; nothing is edited, selected or recorded; and the tab's word passes the plain-words and gearbox guards.

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
        assert_eq!(assumptions, 1, "one of the paths is an assumption");
    }
```

with:

```rust
        assert_eq!(assumptions, 1, "one of the paths is an assumption");
    }

    #[test]
    fn the_five_plots_share_one_tab_whose_picker_shows_each_plot() {
        // UX overhaul spec 4.4 and decision C-9: the five plots sit under one tab, "Plots", with
        // a picker of the five (their names, in the order their tabs had), and the tab row
        // becomes Size and torque, Geometry, Plots and the results table (the Clamp tab before
        // the table while the section shows). A click on a plot's name shows that plot, marked
        // as picked, with the accent's line under the Plots tab; the tab, clicked again from
        // another view, shows the plot picked last, and
        // clicked while a plot shows, keeps it. Picking edits nothing, selects nothing and is
        // no undo step. Nine tabs wrapped onto a second row at 1280 points; the tab row is one
        // row at 1024x700. The tab's word is the one new text, and it is in plain words.
        use crate::engine::explain::plain::mentions_gearbox_protection;
        use crate::gui::guidance::plain_text_violations;
        use crate::gui::plots::PLOTS;
        assert_eq!(plain_text_violations(PLOTS), Vec::<&str>::new());
        assert!(!mentions_gearbox_protection(PLOTS));
        for (size, clamps_paused) in [(egui::vec2(1024.0, 700.0), false), (SCREEN, true)] {
            let mut harness = Harness::of(MagcouplingPanel::new(), size);
            harness.panel.clamps_paused = clamps_paused;
            let output = harness.frame(Vec::new());
            let right = right_column_left(&output);
            // The texts drawn in the centre region in reading order: by row from the top (a
            // text within 4 points of a row's first is on that row), each row from the left.
            let centre = |output: &egui::FullOutput| -> Vec<(String, egui::Rect)> {
                let mut texts: Vec<(String, egui::Rect)> = flat_shapes(output)
                    .into_iter()
                    .filter_map(|shape| match shape {
                        egui::Shape::Text(text) => {
                            let rect = text.visual_bounding_rect();
                            (rect.left() > INPUTS_WIDTH && rect.right() < right)
                                .then(|| (text.galley.text().to_owned(), rect))
                        }
                        _ => None,
                    })
                    .collect();
                texts.sort_by(|a, b| a.1.center().y.total_cmp(&b.1.center().y));
                let mut rows: Vec<Vec<(String, egui::Rect)>> = Vec::new();
                for text in texts {
                    match rows.last_mut() {
                        Some(row) if (text.1.center().y - row[0].1.center().y).abs() < 4.0 => {
                            row.push(text)
                        }
                        _ => rows.push(vec![text]),
                    }
                }
                rows.into_iter()
                    .flat_map(|mut row| {
                        row.sort_by(|a, b| a.1.left().total_cmp(&b.1.left()));
                        row
                    })
                    .collect()
            };
            let what = format!("{size:?}, clamp section paused {clamps_paused}");
            let opening = CentreView::SizeAndTorque.label();
            let texts = centre(&output);
            let (_, first) = texts
                .iter()
                .find(|(text, _)| text == opening)
                .unwrap_or_else(|| panic!("{what}: the opening tab"));
            let row: Vec<&str> = texts
                .iter()
                .filter(|(_, rect)| (rect.center().y - first.center().y).abs() < 4.0)
                .map(|(text, _)| text.as_str())
                .collect();
            let mut want = vec![opening, CentreView::Geometry.label(), PLOTS];
            if !clamps_paused {
                want.push(CentreView::Clamp.label());
            }
            want.push(CentreView::Results.label());
            assert_eq!(row, want, "{what}: the tab row, on one row");
            let inputs = harness.panel.inputs().clone();
            let undo = harness.panel.history.undo_len();
            // A click on a text of the centre region under the tab row.
            let click = |harness: &mut Harness, text: &str| {
                let output = harness.frame(Vec::new());
                let (_, at) = centre(&output)
                    .into_iter()
                    .find(|(drawn, rect)| drawn == text && rect.top() >= first.top() - 1.0)
                    .unwrap_or_else(|| panic!("{what}: no {text:?} in the centre region"));
                harness.click(at.center())
            };
            click(&mut harness, PLOTS);
            assert_eq!(
                harness.panel.centre,
                CentreView::Plot(PlotKind::TorqueTemperature),
                "{what}: the Plots tab opens on the first plot"
            );
            // The picker: the five names, once each, in their order, under the tab row.
            let output = harness.frame(Vec::new());
            let names: Vec<&str> = PlotKind::ALL.iter().map(|kind| kind.label()).collect();
            let picker: Vec<String> = centre(&output)
                .into_iter()
                .filter(|(text, rect)| {
                    names.contains(&text.as_str()) && rect.top() > first.bottom()
                })
                .map(|(text, _)| text)
                .collect();
            assert_eq!(picker, names, "{what}: the picker");
            let selection = harness.ctx.style().visuals.selection.bg_fill;
            for kind in PlotKind::ALL {
                // The pick draws its plot in the frame of the click: a plot's memory is there
                // from the first frame it draws, and only the first plot drew before its click.
                let drawn = |harness: &Harness| {
                    egui_plot::PlotMemory::load(&harness.ctx, kind.id()).is_some()
                };
                assert_eq!(
                    drawn(&harness),
                    kind == PlotKind::TorqueTemperature,
                    "{what}: {kind:?} drawn before its click"
                );
                click(&mut harness, kind.label());
                assert!(
                    drawn(&harness),
                    "{what}: {kind:?} drawn in its click's frame"
                );
                assert_eq!(harness.panel.centre, CentreView::Plot(kind), "{what}");
                // The picked name sits on the selection's fill; no other name does.
                let output = harness.frame(Vec::new());
                let fills: Vec<egui::Rect> = flat_shapes(&output)
                    .into_iter()
                    .filter_map(|shape| match shape {
                        egui::Shape::Rect(rect) if rect.fill == selection => Some(rect.rect),
                        _ => None,
                    })
                    .collect();
                for other in PlotKind::ALL {
                    let (_, at) = centre(&output)
                        .into_iter()
                        .find(|(text, rect)| text == other.label() && rect.top() > first.bottom())
                        .unwrap_or_else(|| panic!("{what}: {other:?} in the picker"));
                    assert_eq!(
                        fills.iter().any(|fill| fill.contains_rect(at)),
                        other == kind,
                        "{what}: {other:?} picked with {kind:?} shown"
                    );
                }
                // The tab row marks the Plots tab, and no other, as the tab shown.
                let (_, tab) = centre(&output)
                    .into_iter()
                    .find(|(text, rect)| text == PLOTS && rect.top() < first.bottom())
                    .unwrap_or_else(|| panic!("{what}: the Plots tab"));
                let lines = accent_lines(&output);
                assert_eq!(lines.len(), 1, "{what}, {kind:?}: {lines:?}");
                let (left, right, y) = lines[0];
                assert!(
                    left <= tab.left() && tab.right() <= right && tab.bottom() <= y,
                    "{what}, {kind:?}: {lines:?} under {tab:?}"
                );
            }
            // The tab clicked while a plot shows keeps it; back from another view, the tab
            // shows the plot picked last.
            click(&mut harness, PLOTS);
            assert_eq!(
                harness.panel.centre,
                CentreView::Plot(PlotKind::TorqueAngle),
                "{what}: the tab keeps the plot shown"
            );
            click(&mut harness, CentreView::Geometry.label());
            assert_eq!(harness.panel.centre, CentreView::Geometry, "{what}");
            click(&mut harness, PLOTS);
            assert_eq!(
                harness.panel.centre,
                CentreView::Plot(PlotKind::TorqueAngle),
                "{what}: the plot picked last"
            );
            assert_eq!(harness.panel.inputs(), &inputs, "{what}");
            assert_eq!(harness.panel.selection, Selection::Nothing, "{what}");
            assert_eq!(harness.panel.history.undo_len(), undo, "{what}");
        }
    }
```

**`magcoupling-rs/src/gui/plots.rs`**

Replace:

```rust
pub const Y_LABEL_OVERHANG: f32 = 0.25;

```

with:

```rust
pub const Y_LABEL_OVERHANG: f32 = 0.25;

/// The centre region's tab of the five plots (UX overhaul spec 4.4, decision C-9): one tab, with
/// a picker of the plots under it.
pub const PLOTS: &str = "Plots";

```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 12
  ```

  Expected: 1 failed: `the_five_plots_share_one_tab_whose_picker_shows_each_plot`. It fails at "the tab row, on one row" with `left: ["Size and torque", "Geometry", "Torque vs temperature"]`: at 1024 x 700 the opening tab's row holds the first three of nine tabs.

- [ ] **Step 3 (green): the tab, the picker and the tests that walk the views.** The blocks give `CentreView` its `tab`, the panel its `plot`, the tab row its one Plots tab and `view_ui` the picker; `plots.rs`'s comments name the picker; `Harness::tab` and the app's `show_view` reach a plot through it, and the tests listed under "Files" take their new path. A new test, `the_keyboard_reaches_the_plots_tab_and_picks_a_plot_by_space_or_enter`, holds the keyboard's way through the tab and the picker (it passes with this step's code; before it there is no Plots tab to reach). Apply them all before the next step.

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        // left side's heading as well) is the text colour.
        all(&output, &["Gap sweep"], theme::MUTED);
        all(&output, &["Size and torque"], theme::TEXT);
```

with:

```rust
        // left side's heading as well) is the text colour.
        all(&output, &[crate::gui::plots::PLOTS], theme::MUTED);
        all(&output, &["Size and torque"], theme::TEXT);
```

Replace:

```rust
        use crate::gui::CentreView;
        use crate::gui::test_support::{cut_texts, last_characters, primary_button};
        for size in WINDOWS {
```

with:

```rust
        use crate::gui::CentreView;
        use crate::gui::test_support::cut_texts;
        for size in WINDOWS {
```

Replace:

```rust
            for view in CentreView::tabs(crate::engine::clamps::PAUSED) {
                let output = frame_on(&ctx, &mut app, size, Vec::new());
                let tab = last_characters(&output, view.label())[0];
                for pressed in [true, false] {
                    frame_on(&ctx, &mut app, size, vec![primary_button(tab, pressed)]);
                }
                // The pointer leaves, so no tooltip is in the picture; then the view settles.
                frame_on(&ctx, &mut app, size, vec![egui::Event::PointerGone]);
                frame_on(&ctx, &mut app, size, Vec::new());
                let output = frame_on(&ctx, &mut app, size, Vec::new());
                let cut = cut_texts(&output, screen);
```

with:

```rust
            for view in CentreView::tabs(crate::engine::clamps::PAUSED) {
                // The pointer leaves after each click, so no tooltip is in the picture; then
                // the view settles.
                let output = show_view(&ctx, &mut app, size, view, 2);
                let cut = cut_texts(&output, screen);
```

Replace:

```rust

    /// A click at `at` on a screen of `size`, the pointer gone after it, then `settle` frames
```

with:

```rust

    /// Shows `view` as a person does on a screen of `size`: a click on its tab in the centre
    /// region (the left column's heading bears the first tab's words), then, for a plot, a
    /// click on its name in the Plots tab's picker (decision C-9), each click as [`click_at`]
    /// with `settle` frames. Returns the last frame.
    fn show_view(
        ctx: &egui::Context,
        app: &mut MagcouplingApp,
        size: egui::Vec2,
        view: crate::gui::CentreView,
        settle: usize,
    ) -> egui::FullOutput {
        use crate::gui::test_support::last_characters;
        let mut output = frame_on(ctx, app, size, Vec::new());
        let mut clicks = vec![view.tab()];
        if matches!(view, crate::gui::CentreView::Plot(_)) {
            clicks.push(view.label());
        }
        for text in clicks {
            let (left, right) = (
                side_rect(ctx, "magcoupling_inputs").right(),
                side_rect(ctx, "magcoupling_dashboard").left(),
            );
            let at = last_characters(&output, text)
                .into_iter()
                .find(|at| left < at.x && at.x < right)
                .unwrap_or_else(|| panic!("{size:?}: no {text:?} in the centre region"));
            output = click_at(ctx, app, size, at, settle);
        }
        output
    }

    /// A click at `at` on a screen of `size`, the pointer gone after it, then `settle` frames
```

Replace:

```rust
        use crate::gui::CentreView;
        use crate::gui::test_support::{cut_texts, last_characters};
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
```

with:

```rust
        use crate::gui::CentreView;
        use crate::gui::test_support::cut_texts;
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
```

Replace:

```rust
        }
        let mut output = frame_on(ctx, app, size, Vec::new());
        // Last tab first: the page is left on its first view, the lightest to draw.
```

with:

```rust
        }
        // Last tab first: the page is left on its first view, the lightest to draw.
```

Replace:

```rust
        for view in views {
            // The tab in the centre region: the left column's heading bears the first tab's words.
            let left = side_rect(ctx, "magcoupling_inputs").right();
            let tab = last_characters(&output, view.label())
                .into_iter()
                .find(|at| at.x > left)
                .unwrap_or_else(|| panic!("{size:?}, {what}: no tab {}", view.label()));
            output = click_at(ctx, app, size, tab, 2);
            let state = format!("{size:?}, {what}, on {}", view.label());
```

with:

```rust
        for view in views {
            let output = show_view(ctx, app, size, view, 2);
            let state = format!("{size:?}, {what}, on {}", view.label());
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
//! the centre region between them ([`CentreView`]: one tab row, the results table last;
//! decision M42-1), opening on Size and torque with the four tiles over the drawing (decision
//! G-5, which reverses M42-2). Every result is recomputed with every approved correction
```

with:

```rust
//! the centre region between them ([`CentreView`]: one tab row, the results table last;
//! decision M42-1; the five plots under one tab, with a picker, decision C-9), opening on Size
//! and torque with the four tiles over the drawing (decision
//! G-5, which reverses M42-2). Every result is recomputed with every approved correction
```

Replace:

```rust
use crate::gui::look;
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{Readouts, registry};
```

with:

```rust
use crate::gui::look;
use crate::gui::plots::{PLOTS, PlotKind, plot_ui};
use crate::gui::readouts::{Readouts, registry};
```

Replace:

```rust

/// The views of the centre region, one tab row (decision M42-1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

with:

```rust

/// The views of the centre region, one tab row (decision M42-1): a tab per view, but the five
/// plots, which share the Plots tab and its picker (decision C-9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

Replace:

```rust
    Geometry,
    /// One of the plots (egui_plot).
    Plot(PlotKind),
```

with:

```rust
    Geometry,
    /// One of the plots (egui_plot), under the Plots tab with the picker of the five.
    Plot(PlotKind),
```

Replace:

```rust
impl CentreView {
    /// Every view, in tab order.
    pub const ALL: [CentreView; 9] = [
```

with:

```rust
impl CentreView {
    /// Every view, in tab order (the plots in the picker's order, at the Plots tab's place).
    pub const ALL: [CentreView; 9] = [
```

Replace:

```rust

    /// The views the tab row offers, in tab order: every view but the clamp drawing while the
    /// shaft-clamp section is paused (`clamps_paused`).
    pub fn tabs(clamps_paused: bool) -> impl Iterator<Item = CentreView> {
```

with:

```rust

    /// The views the tab row offers, in tab order (each plot through the Plots tab's picker):
    /// every view but the clamp drawing while the shaft-clamp section is paused
    /// (`clamps_paused`).
    pub fn tabs(clamps_paused: bool) -> impl Iterator<Item = CentreView> {
```

Replace:

```rust

    /// The tab text.
    pub const fn label(self) -> &'static str {
```

with:

```rust

    /// The view's name: its tab's text, or a plot's name in the Plots tab's picker.
    pub const fn label(self) -> &'static str {
```

Replace:

```rust
            CentreView::Results => "Results table",
        }
    }
}

/// The calculator panel: design inputs, their results, and the UI that edits the one and
```

with:

```rust
            CentreView::Results => "Results table",
        }
    }

    /// The text of the tab that shows the view: [`PLOTS`] for each plot (decision C-9), the
    /// view's own name for the others.
    pub const fn tab(self) -> &'static str {
        match self {
            CentreView::Plot(_) => PLOTS,
            view => view.label(),
        }
    }
}

/// The calculator panel: design inputs, their results, and the UI that edits the one and
```

Replace:

```rust
    centre: CentreView,
    /// Whether the header's Show spreadsheet details is on (spec S2-5): a view choice, never in
```

with:

```rust
    centre: CentreView,
    /// The plot the Plots tab shows when it is clicked: the plot shown last (decision C-9).
    /// View state, in no design file, link or undo step.
    plot: PlotKind,
    /// Whether the header's Show spreadsheet details is on (spec S2-5): a view choice, never in
```

Replace:

```rust
            centre: CentreView::SizeAndTorque,
            show_details: false,
```

with:

```rust
            centre: CentreView::SizeAndTorque,
            plot: PlotKind::TorqueTemperature,
            show_details: false,
```

Replace:

```rust
        confined(ui, "magcoupling_above_equation_panel", |ui| {
            ui.horizontal_wrapped(|ui| {
```

with:

```rust
        confined(ui, "magcoupling_above_equation_panel", |ui| {
            if let CentreView::Plot(kind) = self.centre {
                self.plot = kind;
            }
            ui.horizontal_wrapped(|ui| {
```

Replace:

```rust
                for view in CentreView::tabs(self.clamps_paused) {
                    if look::tab_ui(ui, self.centre == view, view.label()).clicked() {
                        self.centre = view;
```

with:

```rust
                for view in CentreView::tabs(self.clamps_paused) {
                    // The five plots share one tab, at the first plot's place: it shows the
                    // plot shown last (decision C-9).
                    let view = match view {
                        CentreView::Plot(PlotKind::TorqueTemperature) => {
                            CentreView::Plot(self.plot)
                        }
                        CentreView::Plot(_) => continue,
                        view => view,
                    };
                    let selected = self.centre.tab() == view.tab();
                    if look::tab_ui(ui, selected, view.tab()).clicked() {
                        self.centre = view;
```

Replace:

```rust
            }
            CentreView::Plot(kind) => plot_ui(ui, kind, shown, &self.results, readouts),
            CentreView::Clamp => clamp_ui(ui, shown, &self.results, readouts),
```

with:

```rust
            }
            CentreView::Plot(mut kind) => {
                // The Plots tab's picker (decision C-9): the five plots by name, in the order
                // their tabs had. A click shows its plot in this frame.
                ui.horizontal_wrapped(|ui| {
                    for plot in PlotKind::ALL {
                        ui.selectable_value(&mut kind, plot, plot.label());
                    }
                });
                self.centre = CentreView::Plot(kind);
                plot_ui(ui, kind, shown, &self.results, readouts);
            }
            CentreView::Clamp => clamp_ui(ui, shown, &self.results, readouts),
```

Replace:

```rust

        /// A click on the tab of `view` in the centre region's tab row.
        fn tab(&mut self, view: CentreView) -> egui::FullOutput {
```

with:

```rust

        /// A click on the tab of `view` in the centre region's tab row, then, for a plot, on its
        /// name in the Plots tab's picker under the row (decision C-9).
        fn tab(&mut self, view: CentreView) -> egui::FullOutput {
```

Replace:

```rust
            let output = self.frame(Vec::new());
            let tab = text_rects(&output, view.label())
                .into_iter()
```

with:

```rust
            let output = self.frame(Vec::new());
            let tab = text_rects(&output, view.tab())
                .into_iter()
```

Replace:

```rust
                .find(|rect| rect.left() > INPUTS_WIDTH)
                .unwrap_or_else(|| panic!("no tab {:?}", view.label()));
            self.click(tab.center())
        }
```

with:

```rust
                .find(|rect| rect.left() > INPUTS_WIDTH)
                .unwrap_or_else(|| panic!("no tab {:?}", view.tab()));
            let output = self.click(tab.center());
            if !matches!(view, CentreView::Plot(_)) {
                return output;
            }
            let output = self.frame(Vec::new());
            let right = right_column_left(&output);
            let name = text_rects(&output, view.label())
                .into_iter()
                .find(|rect| rect.left() > INPUTS_WIDTH && rect.right() < right)
                .unwrap_or_else(|| panic!("no {:?} in the picker", view.label()));
            self.click(name.center())
        }
```

Replace:

```rust
    #[test]
    fn each_plot_tab_draws_its_plot_from_this_frame_s_results() {
```

with:

```rust
    #[test]
    fn the_keyboard_reaches_the_plots_tab_and_picks_a_plot_by_space_or_enter() {
        // The Plots tab and its picker by the keyboard (decisions C-9, D-87): Tab reaches the
        // Plots tab, and the focus arriving on it or on a name changes no view (BL-093); Enter
        // on the tab opens the plot shown last; Tab walks the picker's five names in their
        // order, across both of its rows at 1024x700; Space or Enter on a focused name shows
        // its plot (egui takes either key on a focused widget for a click) and the focus stays
        // on the name. Nothing is edited, selected or recorded.
        use crate::gui::plots::PLOTS;
        for size in [egui::vec2(1024.0, 700.0), SCREEN] {
            let mut harness = Harness::of(MagcouplingPanel::new(), size);
            harness.panel.clamps_paused = true;
            let output = harness.frame(Vec::new());
            let tab = text_rects(&output, PLOTS)
                .into_iter()
                .find(|rect| rect.left() > INPUTS_WIDTH)
                .unwrap_or_else(|| panic!("{size:?}: the Plots tab"));
            let inputs = harness.panel.inputs().clone();
            let undo = harness.panel.history.undo_len();
            let focused = |harness: &Harness| {
                let id = harness.ctx.memory(|m| m.focused())?;
                harness.ctx.read_response(id).map(|response| response.rect)
            };
            // A key's tap, then a frame for the view to follow.
            let tap = |harness: &mut Harness, key: egui::Key| {
                harness.frame(key_tap(key));
                harness.frame(Vec::new());
            };
            let mut presses = 0;
            while !focused(&harness).is_some_and(|at| at.contains_rect(tab)) {
                assert!(presses < 200, "{size:?}: Tab never reaches the Plots tab");
                tap(&mut harness, egui::Key::Tab);
                presses += 1;
                assert_eq!(
                    harness.panel.centre,
                    CentreView::SizeAndTorque,
                    "{size:?}: Tab {presses} changes no view"
                );
            }
            tap(&mut harness, egui::Key::Enter);
            assert_eq!(
                harness.panel.centre,
                CentreView::Plot(PlotKind::TorqueTemperature),
                "{size:?}: Enter on the Plots tab"
            );
            let output = harness.frame(Vec::new());
            let right = right_column_left(&output);
            let names: Vec<(PlotKind, egui::Rect)> = PlotKind::ALL
                .into_iter()
                .map(|kind| {
                    let at = text_rects(&output, kind.label())
                        .into_iter()
                        .find(|rect| rect.left() > INPUTS_WIDTH && rect.right() < right)
                        .unwrap_or_else(|| panic!("{size:?}: {kind:?} in the picker"));
                    (kind, at)
                })
                .collect();
            let on_name = |harness: &Harness| {
                let at = focused(harness)?;
                names
                    .iter()
                    .find(|(_, name)| at.contains_rect(*name))
                    .map(|(kind, _)| *kind)
            };
            presses = 0;
            while on_name(&harness).is_none() {
                assert!(presses < 10, "{size:?}: Tab never reaches the picker");
                tap(&mut harness, egui::Key::Tab);
                presses += 1;
            }
            // Through the five names: Space on Slip heating, Enter on Torque vs rotation.
            let mut shown = PlotKind::TorqueTemperature;
            for (i, kind) in PlotKind::ALL.into_iter().enumerate() {
                if i > 0 {
                    tap(&mut harness, egui::Key::Tab);
                }
                assert_eq!(on_name(&harness), Some(kind), "{size:?}: name {i} focused");
                assert_eq!(
                    harness.panel.centre,
                    CentreView::Plot(shown),
                    "{size:?}: the focus on {kind:?} picks nothing"
                );
                let key = match kind {
                    PlotKind::SlipHeating => egui::Key::Space,
                    PlotKind::TorqueAngle => egui::Key::Enter,
                    _ => continue,
                };
                tap(&mut harness, key);
                shown = kind;
                assert_eq!(
                    harness.panel.centre,
                    CentreView::Plot(kind),
                    "{size:?}: {key:?} on {kind:?}"
                );
                assert_eq!(on_name(&harness), Some(kind), "{size:?}: the focus stays");
            }
            assert_eq!(harness.panel.inputs(), &inputs, "{size:?}");
            assert_eq!(harness.panel.selection, Selection::Nothing, "{size:?}");
            assert_eq!(harness.panel.history.undo_len(), undo, "{size:?}");
        }
    }

    #[test]
    fn each_plot_tab_draws_its_plot_from_this_frame_s_results() {
```

Replace:

```rust
            };
            let output = harness.click_text(kind.label());
            assert_eq!(harness.panel.centre, CentreView::Plot(kind));
```

with:

```rust
            };
            let output = harness.tab(CentreView::Plot(kind));
            assert_eq!(harness.panel.centre, CentreView::Plot(kind));
```

Replace:

```rust
        // is painted at the new pull-out, where that frame's plot transform puts it.
        harness.click_text(PlotKind::TorqueTemperature.label());
        harness.focus(FACE_GAP);
```

with:

```rust
        // is painted at the new pull-out, where that frame's plot transform puts it.
        harness.tab(CentreView::Plot(PlotKind::TorqueTemperature));
        harness.focus(FACE_GAP);
```

Replace:

```rust
        );
        let other = PlotKind::GapSweep.label();
        assert_eq!(text_color(&output, other), Some(visuals.text_color()));
```

with:

```rust
        );
        let other = PLOTS;
        assert_eq!(text_color(&output, other), Some(visuals.text_color()));
```

Replace:

```rust
            .style_mut(|style| style.visuals.weak_text_color = None);
        // The row wraps at this width: a tab on its second line is underlined where it is.
        let last = CentreView::Results.label();
```

with:

```rust
            .style_mut(|style| style.visuals.weak_text_color = None);
        // In a window narrower than the row (a host's: the panel can be drawn in an egui
        // window the user sizes, as the_panel_works_inside_an_egui_window draws it), the row
        // wraps: a tab on its second line is underlined where it is. Since the plots share one
        // tab (decision C-9) the row is one row from the smallest window the app supports
        // (1024 x 700) up.
        let mut harness = Harness::of(MagcouplingPanel::new(), egui::vec2(860.0, 800.0));
        let output = harness.frame(Vec::new());
        let size_tab = *text_rects(&output, CentreView::SizeAndTorque.label())
            .last()
            .expect("the tab");
        let last = CentreView::Results.label();
```

Replace:

```rust
                harness.panel.explorer.open_path("model.pullout_Nm");
            }
```

with:

```rust
                harness.panel.explorer.open_path("model.pullout_Nm");
                harness.frame(Vec::new());
                // Dragged 40 points taller, as a person can: since the plots share one tab
                // (decision C-9) the tab row is one row here, and the list had the room for
                // every row at the panel's starting height.
                let ctx = harness.ctx.clone();
                let top = equation_panel_rect(&ctx).top();
                drag_equation_panel(&ctx, &mut |events| harness.frame(events), top - 40.0);
            }
```

Replace:

```rust
        // Design menu, behind an entry's Equation link or on the assumptions row; every centre
        // view has its tab; every shown note is in the Learn list; every dashboard row is a
        // Headline row of the results table, and a tile or a number of its check's entry where
```

with:

```rust
        // Design menu, behind an entry's Equation link or on the assumptions row; every centre
        // view has its tab, each plot its name in the Plots tab's picker (decision C-9); every
        // shown note is in the Learn list; every dashboard row is a
        // Headline row of the results table, and a tile or a number of its check's entry where
```

Replace:

```rust

            // 2. Every centre view has its tab; the paused section's has none.
            for view in CentreView::ALL {
```

with:

```rust

            // 2. Every centre view has its tab (a plot, its name in the Plots tab's picker); the
            // paused section's has none.
            for view in CentreView::ALL {
```

**`magcoupling-rs/src/gui/plots.rs`**

Replace:

```rust
//! The plot tabs (spec M4 "Layout": "torque vs temperature (hot/cold allowance band,
//! requirement line); gap sweep; pole sweep; slip heating over time vs limit; torque vs rotation
//! angle (pull-out point)"), drawn with egui_plot (the 0.33 line linkage-sim-rs uses, locked to
//! its version: gate 11).
//!
```

with:

```rust
//! The plots (spec M4 "Layout": "torque vs temperature (hot/cold allowance band, requirement
//! line); gap sweep; pole sweep; slip heating over time vs limit; torque vs rotation angle
//! (pull-out point)"), drawn with egui_plot (the 0.33 line linkage-sim-rs uses, locked to its
//! version: gate 11). They share the centre region's one Plots tab ([`PLOTS`]), whose picker
//! names the five (UX overhaul spec 4.4, decision C-9).
//!
```

Replace:

```rust

/// The plots, one tab each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

with:

```rust

/// The plots, each one choice of the Plots tab's picker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

Replace:

```rust
impl PlotKind {
    /// Every plot, in tab order.
    pub const ALL: [PlotKind; 5] = [
```

with:

```rust
impl PlotKind {
    /// Every plot, in the picker's order (the order of the tabs they had).
    pub const ALL: [PlotKind; 5] = [
```

Replace:

```rust

    /// The tab text.
    pub const fn label(self) -> &'static str {
```

with:

```rust

    /// The plot's name in the Plots tab's picker.
    pub const fn label(self) -> &'static str {
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1139 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text.

- [ ] **Step 5: Commit** the three files: `feat(magcoupler): the five plots share one Plots tab with a picker; the tab row is one row (BL-104 T1)`.

### Task 2: Red in a plot only on a sweep row that does not fit

**Model:** `sonnet`.

The required torque (Torque vs temperature) and the required floor (both sweeps) are drawn in the strong text colour, slip heating's governing limit and its time to the limit in `VIOLET`, the torque plot's limit colour, and the pull-out point in the strong text colour (greyed with the curve when the end-effect model is out of range). Red is left to a sweep row that does not fit (decision D-90). `plots.rs`'s module comment says so, and `look::ACCENT`'s comment no longer says the plots' lines take the red.

**Files:**
- Modify: `magcoupling-rs/src/gui/plots.rs`, `magcoupling-rs/src/gui/look.rs`

**Interfaces:**
- Changes: none outside `plots.rs`'s drawing (`slip_heating_ui` no longer reads the visuals).

- [ ] **Step 1 (red): the rule's test.** The block adds `red_marks_a_sweep_row_that_does_not_fit_and_nothing_else_in_any_plot`: every plot on five designs (the passing design, the example design, one whose high case reaches the limit, a hot day, one the end-effect model greys), every red circle, line, path or rect is a sweep row of `POINT_RADIUS` or the "Does not fit" legend entry's icon; the three plots with a requirement line draw dashes in the strong text colour and slip heating in `VIOLET`.

**`magcoupling-rs/src/gui/plots.rs`**

Replace:

```rust
        }
    }
}

```

with:

```rust
        }
    }

    #[test]
    fn red_marks_a_sweep_row_that_does_not_fit_and_nothing_else_in_any_plot() {
        // UX overhaul decision C-5: red means a failing check and nothing else. The stage 1
        // review (C-3) found the Required torque line, the sweeps' Required floor and Slip
        // heating's Governing limit drawn in the failing red on a design that passes them; the
        // pull-out point and the time to the limit were red too. On a passing design, on the
        // example design (which fails four checks), on one whose high case reaches the limit,
        // on a hot day and on one the end-effect model greys, every red mark of every plot is
        // a sweep row that does not fit (a row whose design fails the fit check) or that row's
        // legend entry. The requirement lines take the strong text colour, and Slip heating's
        // limit the violet of the torque plot's.
        use crate::gui::test_support::{hot_day, passing_design};
        let visuals = egui::Context::default().style().visuals.clone();
        let red = visuals.error_fg_color;
        let mut reaches = DesignInputs::default();
        reaches.temperature.duty.hot_ambient_C += 5.0;
        let mut hot = passing_design();
        hot_day(&mut hot);
        assert!(slip_heating(&compute_all(&reaches)).time_to_limit.is_some());
        // The colours of the dashes `output` paints.
        let dashes = |output: &egui::FullOutput| -> Vec<Color32> {
            flat_shapes(output)
                .into_iter()
                .filter_map(|shape| match shape {
                    egui::Shape::LineSegment { stroke, .. } => Some(stroke.color),
                    _ => None,
                })
                .collect()
        };
        for (name, inputs) in [
            ("passing", passing_design()),
            ("example", DesignInputs::default()),
            ("reaches the limit", reaches),
            ("hot day", hot),
            ("greyed", short_magnets()),
        ] {
            for kind in PlotKind::ALL {
                let output = draw(kind, &inputs);
                let shapes = flat_shapes(&output);
                // A legend entry's name, beside its icon on the icon's row (the legend in the
                // plot's right corner sets each name left of its icon).
                let entry = |at: egui::Pos2| {
                    shapes.iter().find_map(|shape| match shape {
                        egui::Shape::Text(text) => {
                            let rect = text.visual_bounding_rect();
                            let beside =
                                (rect.left() - at.x).abs().min((at.x - rect.right()).abs());
                            ((rect.center().y - at.y).abs() < 4.0 && beside < 16.0)
                                .then(|| text.galley.text().to_owned())
                        }
                        _ => None,
                    })
                };
                let mut wrong = Vec::new();
                for shape in &shapes {
                    match shape {
                        egui::Shape::Circle(c) if c.fill == red || c.stroke.color == red => {
                            let legend = entry(c.center);
                            let row = c.radius == POINT_RADIUS && legend.is_none();
                            if !(row || legend.as_deref() == Some(NO_FIT)) {
                                wrong.push(format!("a circle of radius {} {legend:?}", c.radius));
                            }
                        }
                        egui::Shape::LineSegment { stroke, .. } if stroke.color == red => {
                            wrong.push("a line".to_owned())
                        }
                        egui::Shape::Path(path)
                            if path.fill == red
                                || path.stroke.color == egui::epaint::ColorMode::Solid(red) =>
                        {
                            wrong.push("a path".to_owned())
                        }
                        egui::Shape::Rect(rect) if rect.fill == red => {
                            wrong.push("a rect".to_owned())
                        }
                        _ => {}
                    }
                }
                wrong.sort();
                wrong.dedup();
                assert!(wrong.is_empty(), "{name}, {kind:?}: red {wrong:?}");
                let drawn = dashes(&output);
                match kind {
                    PlotKind::TorqueTemperature | PlotKind::GapSweep | PlotKind::PoleSweep => {
                        assert!(
                            drawn.contains(&visuals.strong_text_color()),
                            "{name}, {kind:?}: the requirement line in {drawn:?}"
                        );
                    }
                    PlotKind::SlipHeating => {
                        assert!(drawn.contains(&VIOLET), "{name}: the limit in {drawn:?}");
                    }
                    PlotKind::TorqueAngle => {}
                }
            }
        }
    }
}

```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 12
  ```

  Expected: 1 failed: `red_marks_a_sweep_row_that_does_not_fit_and_nothing_else_in_any_plot`. It fails on the passing design's first plot: `passing, TorqueTemperature: red ["a circle of radius 3.5 Some(\"Required torque\")", "a line"]` (the Required torque line and its legend dot).

- [ ] **Step 3 (green): the colours.**

**`magcoupling-rs/src/gui/look.rs`**

Replace:

```rust
/// The brand accent, crimson (decision C-5). Never a text colour, and never on a value or a
/// check: a failing check is the host's own red (`Visuals::error_fg_color`), which the plots'
/// reference lines also take (`plots::reference_hline`; stage 2 of slice B settles that).
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0xc8, 0x10, 0x2e);
```

with:

```rust
/// The brand accent, crimson (decision C-5). Never a text colour, and never on a value or a
/// check: a failing check is the host's own red (`Visuals::error_fg_color`), which a plot gives
/// only a sweep row that does not fit (`crate::gui::plots`).
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0xc8, 0x10, 0x2e);
```

**`magcoupling-rs/src/gui/plots.rs`**

Replace:

```rust
//! rotation plots) is drawn grey, as the dashboard greys those rows (decision M42-6).
//!
```

with:

```rust
//! rotation plots) is drawn grey, as the dashboard greys those rows (decision M42-6).
//!
//! Red means a failing check and nothing else (UX overhaul decision C-5): in a plot it marks a
//! sweep row that does not fit, a row whose design fails the fit check. The required torque,
//! the sweeps' required floor and the pull-out point are drawn in the strong text colour, the
//! governing limit in violet in both plots that draw it, and the time to the limit in that
//! violet, where the high case meets the limit (the stage 1 review, C-3).
//!
```

Replace:

```rust
        p.line(Line::new(HIGH_BAND, s.high).color(band).width(1.0));
        reference_hline(p, REQUIRED, s.required_Nm, Level::Bad.color(&visuals));
        reference_vline(p, OPERATING, s.operating_C, visuals.weak_text_color());
```

with:

```rust
        p.line(Line::new(HIGH_BAND, s.high).color(band).width(1.0));
        reference_hline(p, REQUIRED, s.required_Nm, visuals.strong_text_color());
        reference_vline(p, OPERATING, s.operating_C, visuals.weak_text_color());
```

Replace:

```rust
        }
        reference_hline(p, REQUIRED_FLOOR, floor_Nm, Level::Bad.color(&visuals));
        let design = finite([design]);
```

with:

```rust
        }
        reference_hline(p, REQUIRED_FLOOR, floor_Nm, visuals.strong_text_color());
        let design = finite([design]);
```

Replace:

```rust
    }
    let visuals = ui.visuals().clone();
    plot(
```

with:

```rust
    }
    plot(
```

Replace:

```rust
        p.line(Line::new(HIGH_CASE, s.high).color(AMBER).width(2.0));
        reference_hline(p, LIMIT, s.limit_C, Level::Bad.color(&visuals));
        if let Some(point) = s.time_to_limit {
```

with:

```rust
        p.line(Line::new(HIGH_CASE, s.high).color(AMBER).width(2.0));
        reference_hline(p, LIMIT, s.limit_C, VIOLET);
        if let Some(point) = s.time_to_limit {
```

Replace:

```rust
                    .radius(POINT_RADIUS + 1.0)
                    .color(Level::Bad.color(&visuals)),
            );
```

with:

```rust
                    .radius(POINT_RADIUS + 1.0)
                    .color(VIOLET),
            );
```

Replace:

```rust
                    .radius(POINT_RADIUS + 1.0)
                    .color(tint(Level::Bad.color(&visuals), s.greyed, &visuals)),
            );
```

with:

```rust
                    .radius(POINT_RADIUS + 1.0)
                    .color(tint(visuals.strong_text_color(), s.greyed, &visuals)),
            );
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1140 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text.

- [ ] **Step 5: Commit** the two files: `fix(magcoupler): red in a plot only on a sweep row that does not fit (BL-104 T2)`.

### Task 3: Each tick number at full strength or not drawn

**Model:** `sonnet`.

`plot` builds each axis with `AxisHints`: its label as before, and `label_spacing` a one-point range at egui_plot's own threshold (`X_TICK_SPACING` 60 points, `Y_TICK_SPACING` 20), so a number past it is drawn at full strength and none is faded (decision D-91).

**Files:**
- Modify: `magcoupling-rs/src/gui/plots.rs`

**Interfaces:**
- Produces: `gui::plots::X_TICK_SPACING`, `gui::plots::Y_TICK_SPACING`.

- [ ] **Step 1 (red): the numbers' test.** The block adds `every_tick_number_is_drawn_at_full_strength_or_not_at_all`: every plot at 25 window sizes (300 to 900 points wide, 220 to 700 tall), in egui's visuals and in a theme's that sets a weak colour, each text that is a number is drawn in the axes' colour (the weak colour, or the text colour where none is set); more than 500 numbers are read.

**`magcoupling-rs/src/gui/plots.rs`**

Replace:

```rust
    #[test]
    fn red_marks_a_sweep_row_that_does_not_fit_and_nothing_else_in_any_plot() {
```

with:

```rust
    #[test]
    fn every_tick_number_is_drawn_at_full_strength_or_not_at_all() {
        // The stage 1 review (E-7): egui_plot fades a tick number in as the ticks spread apart,
        // from nothing at 60 points apart (20 on the vertical axis) to full strength at 80
        // (30), and a number on the way is as faint as 1.3:1 against the plot. At every
        // spacing a window's size gives the ticks, each number is drawn in the axes' colour,
        // the plot's quiet text colour, or not at all: a theme's weak colour (the app's, the
        // muted token), and in egui's own visuals, which set none, the text colour.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let (mut numbers, mut faded) = (0, Vec::new());
        let theme = Color32::from_rgb(101, 102, 103);
        let cases = PlotKind::ALL
            .into_iter()
            .flat_map(|kind| [(kind, None), (kind, Some(theme))]);
        for (kind, weak) in cases {
            for width in [300.0, 420.0, 560.0, 700.0, 900.0] {
                for height in [220.0, 300.0, 400.0, 520.0, 700.0] {
                    let ctx = egui::Context::default();
                    ctx.style_mut(|style| style.visuals.weak_text_color = weak);
                    let axes = weak.unwrap_or(ctx.style().visuals.text_color());
                    let size = egui::vec2(width, height);
                    let mut output = None;
                    // A plot settles its bounds on the first frame.
                    for _ in 0..2 {
                        output = Some(sized_frame(&ctx, size, Vec::new(), |ui| {
                            plot_ui(ui, kind, &inputs, &results, &mut Readouts::default());
                        }));
                    }
                    for shape in flat_shapes(&output.expect("two frames")) {
                        let egui::Shape::Text(text) = shape else {
                            continue;
                        };
                        if text.galley.text().parse::<f64>().is_err() {
                            continue;
                        }
                        numbers += 1;
                        let mut colours = vec![text.fallback_color];
                        colours.extend(text.galley.job.sections.iter().map(|s| s.format.color));
                        if colours.iter().any(|&c| c != axes) {
                            faded.push(format!(
                                "{kind:?}, weak {weak:?}, at {size:?}: {:?} in {colours:?}",
                                text.galley.text()
                            ));
                        }
                    }
                }
            }
        }
        assert!(numbers > 500, "{numbers} tick numbers drawn");
        assert!(faded.is_empty(), "{} faded: {faded:#?}", faded.len());
    }

    #[test]
    fn red_marks_a_sweep_row_that_does_not_fit_and_nothing_else_in_any_plot() {
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 12
  ```

  Expected: 1 failed: `every_tick_number_is_drawn_at_full_strength_or_not_at_all`. It fails at "faded": 800 tick numbers are drawn faded, the first `TorqueTemperature, weak None, at [300.0 520.0]: "2.1" in [#34_34_34_5F, #34_34_34_5F]`.

- [ ] **Step 3 (green): the axes.**

**`magcoupling-rs/src/gui/plots.rs`**

Replace:

```rust
use egui::Color32;
use egui_plot::{Corner, HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, Points, VLine};

```

with:

```rust
use egui::Color32;
use egui_plot::{
    AxisHints, Corner, HLine, Legend, Line, LineStyle, Plot, PlotPoints, PlotUi, Points, VLine,
};

```

Replace:

```rust
pub const Y_LABEL_OVERHANG: f32 = 0.25;

```

with:

```rust
pub const Y_LABEL_OVERHANG: f32 = 0.25;

/// How far apart two ticks of the horizontal and of the vertical axis must be for their numbers
/// to be drawn [points]: egui_plot's own threshold. Past it egui_plot fades a number in, to full
/// strength at 80 and 30 points, and a number on the way is as faint as 1.3:1 against the plot
/// (the stage 1 review, E-7); each axis here draws the same numbers at full strength.
pub const X_TICK_SPACING: f32 = 60.0;
pub const Y_TICK_SPACING: f32 = 20.0;

```

Replace:

```rust

/// A plot with its id ([`PlotKind::id`]), the panel's legend and axis labels, filling the
/// height left and no more (`height` is egui_plot's smallest height too; it keeps at least 1
/// point).
fn plot(ui: &egui::Ui, kind: PlotKind, x: &str, y: &str) -> Plot<'static> {
```

with:

```rust

/// A plot with its id ([`PlotKind::id`]), the panel's legend and axis labels, its tick numbers
/// at full strength or not drawn ([`X_TICK_SPACING`], [`Y_TICK_SPACING`]), filling the height
/// left and no more (`height` is egui_plot's smallest height too; it keeps at least 1 point).
fn plot(ui: &egui::Ui, kind: PlotKind, x: &str, y: &str) -> Plot<'static> {
```

Replace:

```rust
fn plot(ui: &egui::Ui, kind: PlotKind, x: &str, y: &str) -> Plot<'static> {
    Plot::new(kind.id())
```

with:

```rust
fn plot(ui: &egui::Ui, kind: PlotKind, x: &str, y: &str) -> Plot<'static> {
    let x_axis = AxisHints::new_x()
        .label(x.to_owned())
        .label_spacing(egui::Rangef::point(X_TICK_SPACING));
    let y_axis = AxisHints::new_y()
        .label(y.to_owned())
        .label_spacing(egui::Rangef::point(Y_TICK_SPACING));
    Plot::new(kind.id())
```

Replace:

```rust
        .legend(Legend::default().position(Corner::RightTop))
        .x_axis_label(x.to_owned())
        .y_axis_label(y.to_owned())
        .height(ui.available_height().max(0.0))
```

with:

```rust
        .legend(Legend::default().position(Corner::RightTop))
        .custom_x_axes(vec![x_axis])
        .custom_y_axes(vec![y_axis])
        .height(ui.available_height().max(0.0))
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1141 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text.

- [ ] **Step 5: Commit** the one file: `fix(magcoupler): each plot axis draws a tick number at full strength or not at all (BL-104 T3)`.

### Task 4: gui-smoke judges the tab row and the Plots picker

**Model:** `sonnet`.

After the lesson link and before the Geometry tab, the smoke prompt clicks "Plots", judges the tab row (Size and torque, Geometry, Plots and Results table, on one row) and the five names under it, then clicks "Slip heating" and judges `plots_picker` by the name marked and the legend's "High case" (decision D-93). `app.rs`'s test holds the step's words to the app's (the tab row from `CentreView::tabs` with the section paused, the names from `PlotKind::ALL`, `HIGH_CASE`); the verdict test's `CHECKS` gains `plots_picker`.

**Files:**
- Modify: `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`, `magcoupling-rs/src/app.rs`

**Interfaces:**
- Changes: the smoke agent's schema gains the required boolean `plots_picker`, which `CALCULATOR_CHECKS` then counts.

- [ ] **Step 1 (red): the guard and the verdict test.**

**`linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`**

Replace:

```js
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'check_entry',
  'lesson_link_opened', 'same_column', 'tooltip_plain', 'input_entry', 'tag_entry', 'torque_card',
  'address_follows']
// A calculator agent's reading of a good run; each case changes one thing.
```

with:

```js
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'check_entry',
  'lesson_link_opened', 'plots_picker', 'same_column', 'tooltip_plain', 'input_entry', 'tag_entry',
  'torque_card', 'address_follows']
// A calculator agent's reading of a good run; each case changes one thing.
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
        // UX overhaul slice B, stage 2, part 2b-iii-a (spec 4.4, decision C-9): after the lesson
        // link and before the Geometry tab the step opens the Plots tab, judges its tab row and
        // its picker by the words the page draws (the tab row as the app draws it with the
        // shaft-clamp section paused, the five plots' names in their order), then picks Slip
        // heating and finds its plot by a name of its legend.
        {
            use crate::gui::CentreView;
            use crate::gui::plots::{HIGH_CASE, PLOTS, PlotKind};
            let open = format!("click \"{PLOTS}\" in the row of tabs");
            let geometry = format!("click the \"{}\" tab", CentreView::Geometry.label());
            assert!(
                between("lesson_link_opened=true only if", &geometry).contains(&open),
                "the Plots tab is opened after the lesson link, before the Geometry tab"
            );
            let step = between(&open, "same_column=true only if");
            // A list as the prompt writes it: "a", "b" and "c".
            let listed = |names: &[&str]| {
                let quoted: Vec<String> = names.iter().map(|name| format!("\"{name}\"")).collect();
                let (last, rest) = quoted.split_last().expect("a name");
                format!("{} and {last}", rest.join(", "))
            };
            let mut tabs: Vec<&str> = Vec::new();
            for view in CentreView::tabs(crate::engine::clamps::PAUSED) {
                if tabs.last() != Some(&view.tab()) {
                    tabs.push(view.tab());
                }
            }
            let plots: Vec<&str> = PlotKind::ALL.iter().map(|kind| kind.label()).collect();
            for (what, list) in [
                ("the tab row", listed(&tabs)),
                ("the picker", listed(&plots)),
            ] {
                assert!(
                    step.contains(&list),
                    "the step names {what} in order: {list}"
                );
            }
            let pick = PlotKind::SlipHeating.label();
            for (what, text) in [
                ("the pick", format!("click \"{pick}\"")),
                ("the plot's legend", format!("\"{HIGH_CASE}\"")),
                ("the verdict", "plots_picker=true only if".to_owned()),
            ] {
                assert!(step.contains(&text), "the step names {what}: {text}");
            }
        }
        assert!(
            between("drag_moved=true", "then close the browser").contains(collect),
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib app::tests::the_smoke_test_share_link_opens_its_design 2>&1 | tail -n 8
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: 1 failed: `the_smoke_test_share_link_opens_its_design`. It panics with `the Plots tab is opened after the lesson link, before the Geometry tab` (the prompt has no such step yet). The verdict test ends `1 of 84 checks fail` (the script's schema does not have the check the test names).

- [ ] **Step 3 (green): the prompt.**

**`.claude/workflows/gui-smoke.js`**

Replace:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, the left side on the Geometry tab is still the size controls over the folds Detailed inputs and Assumptions, an input tooltip shows no spreadsheet details and a click on its label shows the input\'s entry in the Explain panel, a click on a dimension tag of the drawing shows its result\'s entry there, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

with:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, the tab row is Size and torque, Geometry, Plots and Results table and the Plots tab\'s picker shows each plot, the left side on the Geometry tab is still the size controls over the folds Detailed inputs and Assumptions, an input tooltip shows no spreadsheet details and a click on its label shows the input\'s entry in the Explain panel, a click on a dimension tag of the drawing shows its result\'s entry there, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

Replace:

```js
const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'check_entry', 'lesson_link_opened', 'same_column', 'tooltip_plain', 'input_entry', 'tag_entry', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
```

with:

```js
const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'check_entry', 'lesson_link_opened', 'plots_picker', 'same_column', 'tooltip_plain', 'input_entry', 'tag_entry', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
```

Replace:

```js
    lesson_link_opened: { type: 'boolean' },
    same_column: { type: 'boolean' },
```

with:

```js
    lesson_link_opened: { type: 'boolean' },
    plots_picker: { type: 'boolean' },
    same_column: { type: 'boolean' },
```

Replace:

```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Design menu and
// its Load design item, a check's row, a lesson link, the Geometry tab, the Detailed inputs fold, an input's label, a dimension tag, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

with:

```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Design menu and
// its Load design item, a check's row, a lesson link, the Plots tab and a plot's name in its picker, the Geometry tab, the Detailed inputs fold, an input's label, a dimension tag, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

Replace:

```js
// and 5.3) a click on a check's row shows its entry in the Explain panel, which scrolls on its
// own, and the entry's lesson link shows a teaching note there (decision C-11); on the Geometry
// tab the left side is still the Size column (part 2b-ii-c, spec 4.3), whose Detailed inputs fold
```

with:

```js
// and 5.3) a click on a check's row shows its entry in the Explain panel, which scrolls on its
// own, and the entry's lesson link shows a teaching note there (decision C-11); the Plots tab
// (part 2b-iii-a, spec 4.4, decision C-9) sits in a tab row of four, and its picker names the five
// plots and shows the one picked; on the Geometry
// tab the left side is still the Size column (part 2b-ii-c, spec 4.3), whose Detailed inputs fold
```

Replace:

```js
// navigation or the close. src/app.rs checks that the prompt names the right column's headings,
// the orientation, the entry's headings, the toggle, the note's parts, the tab, the row and the
// card's words as the page does, that the card step
```

with:

```js
// navigation or the close. src/app.rs checks that the prompt names the right column's headings,
// the orientation, the entry's headings, the toggle, the note's parts, the tab row and the Plots
// tab's picker, the tab, the row and the
// card's words as the page does, that the card step
```

Replace:

```js
Then a check's entry and its lesson link (every click and mouse move below as above: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot): in the check list (the right side) click the name of the row "Hot-low torque". Wait 1 second and take a screenshot. Under the check list the Explain panel now shows the check's entry: a small shape, "Hot-low torque" and its word, then the headings "What it is" and "Numbers behind this check" with the check's numbers under the second. If the line "Full lesson:" at the entry's foot is not in view, move the mouse over the Explain panel and scroll down with code: async (page) => { await page.mouse.move(X, Y); await page.mouse.wheel(0, 1500); await page.waitForTimeout(1000); }, then take a screenshot. check_entry=true only if the entry shows as described, the line "Full lesson:" followed by a link is in view, and the check list's rows above the Explain panel are still in view (the Explain panel scrolls on its own). Then click the link right of "Full lesson:". Wait 1 second and take a screenshot. lesson_link_opened=true only if the Explain panel (the right side, under the check list) now shows a teaching note in place of the check's entry: a bold title, a few sentences, the heading "What to change" with lines that start with Raise, Lower, Choose or Enter, and a closed fold "Where this comes from" (scroll the Explain panel down as above if the fold is not in view).
Then the left side on another tab: click the "Geometry" tab in the row of tabs at the top of the middle region, wait 1 second and take a screenshot. same_column=true only if the left side shows what it showed on the first view: under the heading "Size and torque" the controls "Required torque", "Minimum OD", "Outside diameter", "Overall length", "Magnet face gap" and "Magnets per ring", each a label over a slider, then two closed folds, "Detailed inputs" and "Assumptions"; and no tab named "Design inputs" at its top. Then an input tooltip: click the fold "Detailed inputs" on the left and wait 1 second. Under it the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row (the row under the heading "Key design": the size control of the same name is above the folds) with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
```

with:

```js
Then a check's entry and its lesson link (every click and mouse move below as above: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot): in the check list (the right side) click the name of the row "Hot-low torque". Wait 1 second and take a screenshot. Under the check list the Explain panel now shows the check's entry: a small shape, "Hot-low torque" and its word, then the headings "What it is" and "Numbers behind this check" with the check's numbers under the second. If the line "Full lesson:" at the entry's foot is not in view, move the mouse over the Explain panel and scroll down with code: async (page) => { await page.mouse.move(X, Y); await page.mouse.wheel(0, 1500); await page.waitForTimeout(1000); }, then take a screenshot. check_entry=true only if the entry shows as described, the line "Full lesson:" followed by a link is in view, and the check list's rows above the Explain panel are still in view (the Explain panel scrolls on its own). Then click the link right of "Full lesson:". Wait 1 second and take a screenshot. lesson_link_opened=true only if the Explain panel (the right side, under the check list) now shows a teaching note in place of the check's entry: a bold title, a few sentences, the heading "What to change" with lines that start with Raise, Lower, Choose or Enter, and a closed fold "Where this comes from" (scroll the Explain panel down as above if the fold is not in view).
Then the Plots tab: click "Plots" in the row of tabs at the top of the middle region, wait 1 second and take a screenshot. The row of tabs now reads, left to right on one row, "Size and torque", "Geometry", "Plots" and "Results table", with no plot's name among them; under it a row of the five plots' names, "Torque vs temperature", "Gap sweep", "Pole sweep", "Slip heating" and "Torque vs rotation", the first marked as picked (on a filled background), over a plot. Then click "Slip heating" in that row of names, wait 1 second and take a screenshot. plots_picker=true only if the tab row and the row of names show as described, and after the click "Slip heating" is the name marked as picked and the plot under it has a legend that lists "High case".
Then the left side on another tab: click the "Geometry" tab in the row of tabs at the top of the middle region, wait 1 second and take a screenshot. same_column=true only if the left side shows what it showed on the first view: under the heading "Size and torque" the controls "Required torque", "Minimum OD", "Outside diameter", "Overall length", "Magnet face gap" and "Magnets per ring", each a label over a slider, then two closed folds, "Detailed inputs" and "Assumptions"; and no tab named "Design inputs" at its top. Then an input tooltip: click the fold "Detailed inputs" on the left and wait 1 second. Under it the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row (the row under the heading "Key design": the size control of the same name is above the folds) with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
```

Replace:

```js
Then a dimension tag: the drawing in the middle region (the Geometry tab, still selected) has blue dimension lines, each with a small number on a dark plate at its end. In a fresh screenshot click the number "1" on the drawing: the face gap's, a small number on a dark plate at the top of the round view on the left, on the outer ring of magnets just past the outer end of the short blue line across the gap between the rings; not the "1" of the numbered list under the drawing. Move the mouse off it onto an empty part of the middle region, wait 1 second and take a screenshot. tag_entry=true only if the Explain panel (the right side, under the check list) now shows that dimension's entry: at its top a value in mm and the name "${TAG_LABEL}", and further down the link "Equation"; and no Equation panel opened at the foot of the middle region.
If a target of these last checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes; the check list's last row is "Magnets fit". tag_entry=true was judged last on this page. Collect the console messages at level "info" (the check's entry, the lesson link, the left side, the tooltip, the input's entry and the tag's entry ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; and the tooltip step left the Geometry tab selected, where no tile is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
```

with:

```js
Then a dimension tag: the drawing in the middle region (the Geometry tab, still selected) has blue dimension lines, each with a small number on a dark plate at its end. In a fresh screenshot click the number "1" on the drawing: the face gap's, a small number on a dark plate at the top of the round view on the left, on the outer ring of magnets just past the outer end of the short blue line across the gap between the rings; not the "1" of the numbered list under the drawing. Move the mouse off it onto an empty part of the middle region, wait 1 second and take a screenshot. tag_entry=true only if the Explain panel (the right side, under the check list) now shows that dimension's entry: at its top a value in mm and the name "${TAG_LABEL}", and further down the link "Equation"; and no Equation panel opened at the foot of the middle region.
If a target of these last checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes; the check list's last row is "Magnets fit". tag_entry=true was judged last on this page. Collect the console messages at level "info" (the check's entry, the lesson link, the Plots tab, the left side, the tooltip, the input's entry and the tag's entry ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; and the tooltip step left the Geometry tab selected, where no tile is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
```

Replace:

```js
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, check_entry, lesson_link_opened, same_column, tooltip_plain, input_entry, tag_entry, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
```

with:

```js
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, check_entry, lesson_link_opened, plots_picker, same_column, tooltip_plain, input_entry, tag_entry, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
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

  Expected: `test result: ok. 1141 passed; 0 failed; 6 ignored` for the lib tests with the app feature (1552 tests in all with it, 656 without); every clippy run clean; `all 85 checks hold`.

- [ ] **Step 5: Commit** the three files: `test(magcoupler): gui-smoke judges the tab row and the Plots tab's picker (BL-104 T4)`.

### Task 5: Docs sweep

**Model:** `sonnet`.

The README's layout, Plots, look, smoke and tests passages; `docs/ai/` (02 the system, 03 the modules, 04 the issue, 05 the tracker, the backlog item in progress, and BL-105 filed: the sparse horizontal axes at 1024 x 700 and 1280 x 800, finding UX-2 of the plan's review); the register's rows A-47 to A-51 and its log; owner questions 70 to 72 and the list's count of part C (53: it read 47, three under the 50 it held); the design's status line.

**Files:**
- Modify: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/analyses/2026-11-18-owner-questions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

- [ ] **Step 1: The docs.**

**`docs/ai/02-system.yaml`**

Replace:

```yaml
      - "Since BL-096 (UX overhaul slice B, stage 2, part 2b-ii-c; plan docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md): the left column is the Size column on every tab (design 4.3). MagcouplingPanel::inputs_ui draws size_column_ui in one scroll area whatever CentreView shows: the six controls, then the folds DETAILED_INPUTS (design_inputs_ui: the order toggle, the filter, the Key design group, every input group) and ASSUMPTIONS_FOLD (assumptions_ui), each at the column's full width (show_unindented; since BL-099 the column keeps its 320 points in either font whatever is opened) and neither with a scroll area of its own (BL-091). An unindented body has no id of its own, so size_column_ui draws the Assumptions fold's body in push_id(ASSUMPTIONS_FOLD): a row the filter keeps in the Detailed inputs and the same row under Assumptions (the harmonics dropdown, coupling.max_harmonic) are then two widgets, and either dropdown opens. InputsView and the panel's inputs_view are gone. The trace's banner and Clear trace sit at the column's foot on every tab. A fix on the Outside diameter (SIZE_TAB_ALIASES) sets column_to_top, which scrolls the column back to its slider, and still opens the Size and torque tab; the Key design group's Open the Size tab button (sizing_ui, drawn while Auto is on) sets it too, so the column scrolls back to Auto's toggle. Since BL-097 a value box that takes and loses the keyboard's focus with nothing typed leaves its value bit for bit: input_ui::slider gives every slider (the Size column's, the Required torque's, an input row's) one custom_parser, input_ui::typed_number, which returns no number for the box's own text (egui writes a box's text back when the focus leaves it, typed or not), so Tab through the column edits nothing, makes no undo step, selects nothing (BL-093) and moves no check, for a size off its slider's step (Auto turned off leaves one) and for a value outside its slider's range alike. The number the box already shows, typed again, is no edit either (register row A-43, owner question 66); any other typed number, an arrow key or a drag still edits. The Minimum OD's slider is handed its own value again (decision D-73's rounded copy is gone). panel::tests::nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home is the design's nothing-lost table (8.2 test 9)."
      - "Since BL-100: a scroll area follows the keyboard's focus. gui::follow_focus(ui), called last in a scroll area's content, scrolls that area by the least that shows whole (ui.scroll_to_rect(rect, None)) the widget the focus has arrived on, in the first frame the area is drawn with the focus there and never again (it keeps the focused id it saw last in egui's temporary memory, by the content ui's id), so a wheel or a drag of the scroll bar afterwards is not fought, nothing is selected or edited (BL-093) and an idle frame asks for no repaint. Tab, Shift+Tab, an Up or Down arrow key, a click and a request_focus are all arrivals. The widget is this area's when its layer is the area's and its interact_rect is its rect cut by the area's clip rect; a widget of another panel or of a list open over the area is not. Called by the left column and the Explain panel (panel.rs), the Geometry tab's callout list (geometry_view::list_ui) and the Clamp tab (clamp_drawing::clamp_ui). Not by an area nested in another (egui's ScrollArea takes both axes' scroll targets, so the outer area never hears of a widget of the inner one): the Equation panel's formula and term list, the Clamp tab's screw-size table; nor by the Results table, which draws only the rows in view (BL-101). egui's own Tab stops that show no focus are BL-102."
```

with:

```yaml
      - "Since BL-096 (UX overhaul slice B, stage 2, part 2b-ii-c; plan docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md): the left column is the Size column on every tab (design 4.3). MagcouplingPanel::inputs_ui draws size_column_ui in one scroll area whatever CentreView shows: the six controls, then the folds DETAILED_INPUTS (design_inputs_ui: the order toggle, the filter, the Key design group, every input group) and ASSUMPTIONS_FOLD (assumptions_ui), each at the column's full width (show_unindented; since BL-099 the column keeps its 320 points in either font whatever is opened) and neither with a scroll area of its own (BL-091). An unindented body has no id of its own, so size_column_ui draws the Assumptions fold's body in push_id(ASSUMPTIONS_FOLD): a row the filter keeps in the Detailed inputs and the same row under Assumptions (the harmonics dropdown, coupling.max_harmonic) are then two widgets, and either dropdown opens. InputsView and the panel's inputs_view are gone. The trace's banner and Clear trace sit at the column's foot on every tab. A fix on the Outside diameter (SIZE_TAB_ALIASES) sets column_to_top, which scrolls the column back to its slider, and still opens the Size and torque tab; the Key design group's Open the Size tab button (sizing_ui, drawn while Auto is on) sets it too, so the column scrolls back to Auto's toggle. Since BL-097 a value box that takes and loses the keyboard's focus with nothing typed leaves its value bit for bit: input_ui::slider gives every slider (the Size column's, the Required torque's, an input row's) one custom_parser, input_ui::typed_number, which returns no number for the box's own text (egui writes a box's text back when the focus leaves it, typed or not), so Tab through the column edits nothing, makes no undo step, selects nothing (BL-093) and moves no check, for a size off its slider's step (Auto turned off leaves one) and for a value outside its slider's range alike. The number the box already shows, typed again, is no edit either (register row A-43, owner question 66); any other typed number, an arrow key or a drag still edits. The Minimum OD's slider is handed its own value again (decision D-73's rounded copy is gone). panel::tests::nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home is the design's nothing-lost table (8.2 test 9)."
      - "Since BL-104 (UX overhaul slice B, stage 2, part 2b-iii-a; plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md): the five plots share one tab (design C-9). CentreView keeps a view per plot (ALL and tabs list every view, in the order the tab row reaches them); CentreView::tab is the text of the tab that shows a view (plots::PLOTS for each plot), and centre_ui draws the Plots tab once, at the first plot's place, opening MagcouplingPanel::plot, the plot shown last (view state). Over a plot, view_ui draws the picker: the five PlotKind labels as selectable_value in a wrapping row, the plot shown picked. plots.rs draws no red but a sweep row that does not fit (decision C-5): the requirement lines and the pull-out point in visuals.strong_text_color(), the governing limit and the time to it in VIOLET. Each axis's AxisHints::label_spacing is one point (X_TICK_SPACING 60, Y_TICK_SPACING 20, egui_plot's own lower ends), so a tick number is drawn at full strength or not at all (the stage 1 review's E-7)."
      - "Since BL-100: a scroll area follows the keyboard's focus. gui::follow_focus(ui), called last in a scroll area's content, scrolls that area by the least that shows whole (ui.scroll_to_rect(rect, None)) the widget the focus has arrived on, in the first frame the area is drawn with the focus there and never again (it keeps the focused id it saw last in egui's temporary memory, by the content ui's id), so a wheel or a drag of the scroll bar afterwards is not fought, nothing is selected or edited (BL-093) and an idle frame asks for no repaint. Tab, Shift+Tab, an Up or Down arrow key, a click and a request_focus are all arrivals. The widget is this area's when its layer is the area's and its interact_rect is its rect cut by the area's clip rect; a widget of another panel or of a list open over the area is not. Called by the left column and the Explain panel (panel.rs), the Geometry tab's callout list (geometry_view::list_ui) and the Clamp tab (clamp_drawing::clamp_ui). Not by an area nested in another (egui's ScrollArea takes both axes' scroll targets, so the outer area never hears of a widget of the inner one): the Equation panel's formula and term list, the Clamp tab's screw-size table; nor by the Results table, which draws only the rows in view (BL-101). egui's own Tab stops that show no focus are BL-102."
```

Replace:

```yaml
      - "The typesetter draws only what the record says (a test checks every text run against render::plain; a selector compared for equality, = or ≠, with a code shows its choice's label, an ordering keeps its numbers) and only the parentheses the markup writes and only characters egui's default fonts have: gui::typeset::glyph_safe draws the missing ϑ, superscript minus and ∝ as θ, ¯ and ~ (display only; no record or note text changes), ∈ and the ceiling and floor brackets are strokes; a test lays out every equation and checks every character, another every teaching note."
      - "The plot tabs build their series from the same frame's results through the engine's own closed forms (model::ring_pair_factor for torque against temperature, the amp*_Pa amplitudes for torque against rotation), which tests pin to the engine's torques; only the tab shown builds its series, and a point that is not finite is left out."

```

with:

```yaml
      - "The typesetter draws only what the record says (a test checks every text run against render::plain; a selector compared for equality, = or ≠, with a code shows its choice's label, an ordering keeps its numbers) and only the parentheses the markup writes and only characters egui's default fonts have: gui::typeset::glyph_safe draws the missing ϑ, superscript minus and ∝ as θ, ¯ and ~ (display only; no record or note text changes), ∈ and the ceiling and floor brackets are strokes; a test lays out every equation and checks every character, another every teaching note."
      - "The plots (one Plots tab with a picker since BL-104) build their series from the same frame's results through the engine's own closed forms (model::ring_pair_factor for torque against temperature, the amp*_Pa amplitudes for torque against rotation), which tests pin to the engine's torques; only the plot shown builds its series, and a point that is not finite is left out."

```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): mod.rs (follow_focus: a scroll area scrolls, by the least that shows it whole, to the widget the keyboard's focus arrives on; called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, BL-100; wrapping_line: ui.horizontal_wrapped with one point of the width kept free, for every wrapping line of the left column, of the Explain panel and of the check list's warning links, BL-099 and BL-098), checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), select and select_over (a click asks for the result in the Explain panel: ReadoutEvents::selected, since BL-094), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds; diagram_ui scales a diagram as one picture to a place narrower than DIAGRAM_SIZE, BL-098), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel, whose left column is the Size column on every tab since BL-096 (inputs_ui, size_column_ui: the size controls over the folds DETAILED_INPUTS and ASSUMPTIONS_FOLD, the trace's banner at its foot; no InputsView): new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input; the label is one block in the head line, so a wrapped label is one Tab stop) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider (every slider's set-up; its value box's own text, the text egui's formatter wrote in it, is no typed value: typed_number, BL-097), head_line (a row's head line, the changed dot and then the label on a line that wraps at the row's width: an input row's and an assumption's, BL-099), RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); drawing_ui (the opening tab's: the tags, and only the notes with a level, since BL-094); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the left column, on every tab since BL-096; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), side_rect (a side panel's rect as egui keeps it: the panel's and the app's width tests), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

with:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): mod.rs (follow_focus: a scroll area scrolls, by the least that shows it whole, to the widget the keyboard's focus arrives on; called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, BL-100; wrapping_line: ui.horizontal_wrapped with one point of the width kept free, for every wrapping line of the left column, of the Explain panel and of the check list's warning links, BL-099 and BL-098), checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), select and select_over (a click asks for the result in the Explain panel: ReadoutEvents::selected, since BL-094), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds; diagram_ui scales a diagram as one picture to a place narrower than DIAGRAM_SIZE, BL-098), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel, whose left column is the Size column on every tab since BL-096 (inputs_ui, size_column_ui: the size controls over the folds DETAILED_INPUTS and ASSUMPTIONS_FOLD, the trace's banner at its foot; no InputsView): new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5, which share one tab since BL-104 (CentreView::tab: plots::PLOTS for each plot; the tab opens the panel's plot field, the plot shown last; view_ui draws the picker of the five over a plot); Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input; the label is one block in the head line, so a wrapped label is one Tab stop) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider (every slider's set-up; its value box's own text, the text egui's formatter wrote in it, is no typed value: typed_number, BL-097), head_line (a row's head line, the changed dot and then the label on a line that wraps at the row's width: an input row's and an assumption's, BL-099), RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); drawing_ui (the opening tab's: the tags, and only the notes with a level, since BL-094); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PLOTS, the centre's one tab of the five plots (decision C-9, BL-104); PlotKind, the choices of its picker, with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; X_TICK_SPACING and Y_TICK_SPACING, each axis's AxisHints label_spacing as one point, so a tick number is at full strength or not drawn (E-7, BL-104); no red but a sweep row that does not fit: the requirement lines and the pull-out point in the strong text colour, the governing limit and the time to it in VIOLET (C-5, BL-104); NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the left column, on every tab since BL-096; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), side_rect (a side panel's rect as egui keeps it: the panel's and the app's width tests), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
  - "Magcoupler, the columns hold still (branch magcoupling/columns-hold-still, cut from main 739a732 on 2026-11-26; one commit per item, in_progress until the controller's reviews, gate and merge). BL-097: a value box left with nothing typed changes nothing. input_ui::slider hands egui's Slider one custom_parser (input_ui::typed_number): the box's own text is no typed value, so Tab, Shift+Tab, Enter or a click elsewhere with nothing typed leaves a size off its step (Auto turned off leaves 11.942 mm shown as 11.9) and a value outside its slider's range bit for bit, with no undo step, no selection and no check moved; a typed number, an arrow key and a drag edit as before. Typing the number the box already shows is no edit either (register row A-43, owner question 66: the conservative answer, not the owner's). Decision D-73's special case for the Minimum OD's slider is removed: its tests hold on the one rule and fail without it. BL-100: a scroll area follows the keyboard's focus. gui::follow_focus, called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, scrolls the area by the least that shows whole the widget the focus has arrived on (Tab, Shift+Tab, an Up or Down arrow key), once, so a wheel is not fought, nothing is selected and an idle frame asks for no repaint. Filed from it, not fixed: BL-101 (the Equation panel, the Clamp tab's screw-size table and the Results table do not follow the focus: nested scroll areas and rows drawn only in view; register row A-44, owner question 67 on whether Tab should walk every row of the Results table) and BL-102 (egui's own Tab stops that show no focus: each scroll area's drag surface and scroll bar, each column's resize edge). BL-099: the left column keeps its 320 points whatever it shows. The head line of an input row and of an assumption wraps (input_ui::head_line), so a long label takes a second line where it ran past the column's edge and widened it for good (330.1 points with the Detailed inputs open in the app's font, 443.1 with the last group open, 418.8 in egui's default font); a group's heading wraps where its traced count makes it longer than the column (panel::wrapped_title; found by the build: 336.6 points with the Required torque traced). The item's evidence blamed the part pickers; they fitted their width, and still cut their choice with an ellipsis (register row A-45, owner question 68: the conservative answer, not the owner's). BL-098: the right column keeps its 300 points whatever the Explain panel shows. A note's diagram was 300 points wide in a column that holds 284 (316 points from the first note with a diagram on); it is now the same picture scaled to the width it is given (diagrams::diagram_ui; register row A-46, owner question 69: about 5 % smaller there). The sweep of every entry (notes, checks, tiles, inputs, all results; both fonts, both designs, details off and on) found one more cause, a wrapping line's label a fraction of a point past the line (0.06 in two notes, 0.375 in one result's entry): every wrapping line of the two side columns is now gui::wrapping_line, which keeps a point free (input_ui::head_line calls it). The sweep found nothing else, so no item was filed from it. risk: mechanical. Fix wave of the controller's review (2026-11-27): BL-097's rule compares the typed number with the text egui's formatter wrote in the box (recorded through Slider::custom_formatter), not with every coarser rounding of the value, which for a tiny value (the vacuum permeability) threw a typed number away; the check list's material warning's Why link wraps (gui::wrapping_line), which widened the right column with no back iron, on main too; tests added for follow_focus's clip check (a wheeled column holds still while the focus is elsewhere) and for the left column's width with Auto on; the grade pickers' blank choice is newly cut in the app's font (A-45 and owner question 68 corrected); BL-103 filed (the widths at pixel ratios 0.9 and 1.35). Fix wave of the re-review (2026-11-27): an input's label is drawn as one block in its head line, so a label that wraps is one Tab stop that shows its focus (SK-1: egui lays a label out in a wrapping line row by row, each row a Tab stop, and only the first showed the focus; 16 input labels wrap at the narrowest width the column gives a row in egui's default font; new with BL-099's wrapping head line); after a wrapped label the note and the reset button take a line of their own; the hold-still test's comment reads its own 24.5 points (RF-1)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
```

with:

```yaml
  - "Magcoupler, the columns hold still (branch magcoupling/columns-hold-still, cut from main 739a732 on 2026-11-26; one commit per item, in_progress until the controller's reviews, gate and merge). BL-097: a value box left with nothing typed changes nothing. input_ui::slider hands egui's Slider one custom_parser (input_ui::typed_number): the box's own text is no typed value, so Tab, Shift+Tab, Enter or a click elsewhere with nothing typed leaves a size off its step (Auto turned off leaves 11.942 mm shown as 11.9) and a value outside its slider's range bit for bit, with no undo step, no selection and no check moved; a typed number, an arrow key and a drag edit as before. Typing the number the box already shows is no edit either (register row A-43, owner question 66: the conservative answer, not the owner's). Decision D-73's special case for the Minimum OD's slider is removed: its tests hold on the one rule and fail without it. BL-100: a scroll area follows the keyboard's focus. gui::follow_focus, called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, scrolls the area by the least that shows whole the widget the focus has arrived on (Tab, Shift+Tab, an Up or Down arrow key), once, so a wheel is not fought, nothing is selected and an idle frame asks for no repaint. Filed from it, not fixed: BL-101 (the Equation panel, the Clamp tab's screw-size table and the Results table do not follow the focus: nested scroll areas and rows drawn only in view; register row A-44, owner question 67 on whether Tab should walk every row of the Results table) and BL-102 (egui's own Tab stops that show no focus: each scroll area's drag surface and scroll bar, each column's resize edge). BL-099: the left column keeps its 320 points whatever it shows. The head line of an input row and of an assumption wraps (input_ui::head_line), so a long label takes a second line where it ran past the column's edge and widened it for good (330.1 points with the Detailed inputs open in the app's font, 443.1 with the last group open, 418.8 in egui's default font); a group's heading wraps where its traced count makes it longer than the column (panel::wrapped_title; found by the build: 336.6 points with the Required torque traced). The item's evidence blamed the part pickers; they fitted their width, and still cut their choice with an ellipsis (register row A-45, owner question 68: the conservative answer, not the owner's). BL-098: the right column keeps its 300 points whatever the Explain panel shows. A note's diagram was 300 points wide in a column that holds 284 (316 points from the first note with a diagram on); it is now the same picture scaled to the width it is given (diagrams::diagram_ui; register row A-46, owner question 69: about 5 % smaller there). The sweep of every entry (notes, checks, tiles, inputs, all results; both fonts, both designs, details off and on) found one more cause, a wrapping line's label a fraction of a point past the line (0.06 in two notes, 0.375 in one result's entry): every wrapping line of the two side columns is now gui::wrapping_line, which keeps a point free (input_ui::head_line calls it). The sweep found nothing else, so no item was filed from it. risk: mechanical. Fix wave of the controller's review (2026-11-27): BL-097's rule compares the typed number with the text egui's formatter wrote in the box (recorded through Slider::custom_formatter), not with every coarser rounding of the value, which for a tiny value (the vacuum permeability) threw a typed number away; the check list's material warning's Why link wraps (gui::wrapping_line), which widened the right column with no back iron, on main too; tests added for follow_focus's clip check (a wheeled column holds still while the focus is elsewhere) and for the left column's width with Auto on; the grade pickers' blank choice is newly cut in the app's font (A-45 and owner question 68 corrected); BL-103 filed (the widths at pixel ratios 0.9 and 1.35). Fix wave of the re-review (2026-11-27): an input's label is drawn as one block in its head line, so a label that wraps is one Tab stop that shows its focus (SK-1: egui lays a label out in a wrapping line row by row, each row a Tab stop, and only the first showed the focus; 16 input labels wrap at the narrowest width the column gives a row in egui's default font; new with BL-099's wrapping head line); after a wrapped label the note and the reset button take a line of their own; the hold-still test's comment reads its own 24.5 points (RF-1)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-a (BL-104, plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md, decisions D-86 to D-94): the five plots under one Plots tab with a picker (C-9): the picker a row of the five names (D-87, register row A-48, owner question 70), the tab opens on the plot shown last (D-88, register row A-51, the builder's call), the tab row keeps 'Results table' (D-89, A-49, question 71); in a plot red only on a sweep row that does not fit (D-90, A-50, question 72: the requirement lines and the pull-out point in the strong text colour, the limit and the time to it violet); each tick number at full strength or not drawn (D-91, E-7); owner question 56 keeps its default (D-92); gui-smoke's plots_picker (D-93). Part 2b-iii is cut in three (D-86, A-47): 2b-iii-b the look (E-4, R-6, S-1, S-4), 2b-iii-c the Equation panel (BL-086, BL-088, BL-089; BL-101's Equation panel part weighed there). BL-105 filed: at 1024 x 700 and 1280 x 800 four plots' horizontal axes carry one or two numbers or none (egui_plot's 60-point threshold, kept by D-91; on main too). risk: mechanical (D-94)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown

---

## 2026-11-27 - Magcoupler: fix wave of the re-review of BL-097 to BL-100 (branch magcoupling/columns-hold-still)
```

with:

```markdown

---

## 2026-11-27 - Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-a: the Plots tab (BL-104)
- What changed: the five plots share one Plots tab (design C-9): the tab row is Size and torque, Geometry, Plots and Results table (the Clamp tab before the table while that section shows), one row from 1024 x 700 up where it wrapped at 1280 x 800. Under the Plots tab a picker names the five plots in the order their tabs had and marks the one shown; the tab opens on the plot shown last. Red in a plot marks only a sweep row that does not fit (decision C-5, the stage 1 review's C-3): the required torque and the sweeps' required floor are drawn in the strong text colour, the governing limit and the time to it in violet, the pull-out point in the strong text colour. Each axis draws a tick number at full strength or not at all (the stage 1 review's E-7). gui-smoke judges the tab row and the picker (`plots_picker`).
- Tests: the tab row, the picker, its picks (each drawn in its click's frame) and the tab's memory, and the keyboard through the Plots tab and the picker (Tab changes no view, Enter on the tab, Space or Enter on a name) (panel); red only on a sweep row that does not fit in every plot on five designs, every tick number in the axes' colour at 25 window sizes in two visuals (plots); the app's text-cut and column-width tests reach each plot through the picker; the smoke prompt's words (app) and the verdict test's list (85 checks). Two tests take a new setting because the tab row is now one row: the Geometry list's focus test drags the Equation panel 40 points taller, and the underline test's wrapped row is drawn in an 860 x 800 window.
- Not in this part: part 2b-iii-b (the look: the controls' outlines, a SemiBold face, the page shell's loading line) and part 2b-iii-c (the Equation panel: BL-086, BL-088, BL-089, BL-101's Equation panel part); the sparse horizontal axes at 1024 x 700 and 1280 x 800 (BL-105, filed: egui_plot's 60-point threshold, kept by decision D-91, as on main).
- Docs: README, docs/ai 02, 03, 04, backlog BL-104 in_progress and BL-105 filed, register rows A-47 to A-51, owner questions 70 to 72, the design's header.

## 2026-11-27 - Magcoupler: fix wave of the re-review of BL-097 to BL-100 (branch magcoupling/columns-hold-still)
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  acceptance: "Plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md: the tab row draws one Plots tab where the five plot tabs were (CentreView::tab), one row at 1024 x 700 with the Clamp tab shown; under it a picker of the five plots, their names in their old order, the plot shown marked as picked; a click on a name shows its plot; the Plots tab reopens the plot shown last and keeps it on a click; picking edits nothing, selects nothing and is no undo step; every plot keeps its readouts, legend and axes; no reference line or mark of a plot is red but a sweep row that does not fit and its legend entry (the requirement lines in the strong text colour, the governing limit and the time to it in violet, the pull-out point in the strong text colour); every tick number is drawn in the axes' colour or not at all; the app tests walk each plot through the picker; the gui-smoke prompt judges plots_picker after the lesson link, before the Geometry tab; each with its red test first; no engine file changes; README and docs/ai updated; GATE PASS with --full; the gui-smoke run from the branch's own script with frame_check: true; an independent whole-branch review on fable (while fable is out, on the session model); then merge and push."
  priority: 2
  status: open
  notes: "Filed 2026-11-27 with its plan (the design's header: each plan files one item). Part 2b-iii is cut in three (decision D-86, register row A-47): this part, 2b-iii-a, the Plots tab (C-9, the stage 1 review's C-3 and E-7); 2b-iii-b, the look (the controls' outlines E-4 and R-6, a SemiBold face S-1, the page shell's loading line S-4), and 2b-iii-c, the Equation panel (BL-086, BL-088, BL-089, with BL-101's Equation panel part weighed there), each with its own plan and item. risk: mechanical (decision D-94): no new number and no new physical claim; one new on-screen string, the tab's word 'Plots', the design's own (C-9). Owner question 56 (whether a plot's readout selects its result) keeps its default (decision D-92). BL-083, BL-084, BL-087, BL-095, BL-102 and BL-103 are not touched."

```

with:

```yaml
  acceptance: "Plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md: the tab row draws one Plots tab where the five plot tabs were (CentreView::tab), one row at 1024 x 700 with the Clamp tab shown; under it a picker of the five plots, their names in their old order, the plot shown marked as picked; a click on a name shows its plot; the Plots tab reopens the plot shown last and keeps it on a click; picking edits nothing, selects nothing and is no undo step; every plot keeps its readouts, legend and axes; no reference line or mark of a plot is red but a sweep row that does not fit and its legend entry (the requirement lines in the strong text colour, the governing limit and the time to it in violet, the pull-out point in the strong text colour); every tick number is drawn in the axes' colour or not at all; the app tests walk each plot through the picker; the gui-smoke prompt judges plots_picker after the lesson link, before the Geometry tab; each with its red test first; no engine file changes; README and docs/ai updated; GATE PASS with --full; the gui-smoke run from the branch's own script with frame_check: true; an independent whole-branch review on fable (while fable is out, on the session model); then merge and push."
  priority: 2
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2b-iii-a (Tasks 1 to 4 of the plan); the full gate, the smoke run and the whole-branch review are its Task 6. Filed 2026-11-27 with its plan (the design's header: each plan files one item). Part 2b-iii is cut in three (decision D-86, register row A-47): this part, 2b-iii-a, the Plots tab (C-9, the stage 1 review's C-3 and E-7); 2b-iii-b, the look (the controls' outlines E-4 and R-6, a SemiBold face S-1, the page shell's loading line S-4), and 2b-iii-c, the Equation panel (BL-086, BL-088, BL-089, with BL-101's Equation panel part weighed there), each with its own plan and item. risk: mechanical (decision D-94): no new number and no new physical claim; one new on-screen string, the tab's word 'Plots', the design's own (C-9). Owner question 56 (whether a plot's readout selects its result) keeps its default (decision D-92). BL-083, BL-084, BL-087, BL-095, BL-102 and BL-103 are not touched."
- id: BL-105
  title: "Magcoupler: at 1024 x 700 and 1280 x 800 four plots' horizontal axes carry one or two numbers, or none"
  dimension: gui
  risk: mechanical
  evidence: "The review of the plan of part 2b-iii-a (BL-104; 2026-11-27, finding UX-2), in the app's own theme and font on the starting design with the plan's Tasks 1 to 4 replayed, rerun by its fix wave: at 1024 x 700 and at 1280 x 800 the horizontal axis of Torque vs temperature carries '0' and '100' (of -40 to 110), Slip heating's '0' and '1000' (of 0 to 1300), Pole sweep's one number, '10' (of 6 to 16 magnets), and Torque vs rotation's one at most, '0' (of 0 to 90); Gap sweep's carries all four, 1 to 4. At 1920 x 1080 those four axes carry 10 to 16 numbers. egui_plot's spacer steps in tens, and the 60-point threshold the axes keep (X_TICK_SPACING, decision D-91: egui_plot's own lower end, so the numbers drawn are main's) skips the next finer step, 53 to 59 points apart at 1280 x 800. A lower threshold is not one constant: at 32 points the probe gives 6 to 16 and 0 to 90 at 1280 x 800 but cuts '110' by 4 points on Torque vs temperature and leaves 9-point gaps between Slip heating's numbers."
  acceptance: "A red test first (the app's theme and font, the starting design, 1024 x 700 and 1280 x 800): every plot's horizontal axis carries at least three numbers, none cut and none nearer its neighbour than a stated gap; the vertical axes and the 1920 x 1080 layout keep their numbers; every number still drawn at full strength or not at all (BL-104's test); README and docs/ai updated; GATE PASS."
  priority: 4
  status: open
  notes: "Filed 2026-11-27 with part 2b-iii-a's docs (the fix wave of its plan's review, UX-2), not taken there: D-91 keeps egui_plot's threshold on purpose (the numbers main drew, none faded), and no single constant fixes it. On main too."

```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
| A-46 | BL-098: a teaching note's diagram is 300 points wide and the Explain panel's column holds 284, so the first note with a diagram made the column 316 points wide for the rest of the session. The column now keeps its 300 points. How does the diagram fit? | It is the same picture scaled to the column: 284 x 123 points where it was 300 x 130, its labels and lines about 5 % smaller. | Nothing in the picture moves against anything else, so no label meets another, and no word or layout of the note changes. The cost: the labels are drawn at about 10.4 points where they were 11, and a scaled label can look a little softer than one set at its own size. Owner question 69. | Each diagram is laid out again for a 284-point width with its labels at full size (six layouts to redo), or the right column starts 316 points wide. | `magcoupling-rs/src/gui/diagrams.rs` (`diagram_ui`), BL-098 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
```

with:

```markdown
| A-46 | BL-098: a teaching note's diagram is 300 points wide and the Explain panel's column holds 284, so the first note with a diagram made the column 316 points wide for the rest of the session. The column now keeps its 300 points. How does the diagram fit? | It is the same picture scaled to the column: 284 x 123 points where it was 300 x 130, its labels and lines about 5 % smaller. | Nothing in the picture moves against anything else, so no label meets another, and no word or layout of the note changes. The cost: the labels are drawn at about 10.4 points where they were 11, and a scaled label can look a little softer than one set at its own size. Owner question 69. | Each diagram is laid out again for a 284-point width with its labels at full size (six layouts to redo), or the right column starts 316 points wide. | `magcoupling-rs/src/gui/diagrams.rs` (`diagram_ui`), BL-098 |
| A-47 | Part 2b-iii (the Plots tab with its picker, the plots' red lines and fading tick numbers, the controls' outlines, a SemiBold face, the page shell's loading line, BL-086, BL-088, BL-089) is again more than one plan a reviewer can hold. Built at once or in parts? | Three parts. 2b-iii-a: the Plots tab and its picker, the plots' red and their tick numbers, gui-smoke's Plots check. 2b-iii-b: the controls' outlines, the SemiBold face, the page shell's loading line. 2b-iii-c: the Equation panel (BL-086, BL-088, BL-089, and BL-101's Equation panel part if that plan holds it). | Smaller releases (the conservative answer to Q-4, as A-21, A-23, A-25 and A-29); 2b-iii-a ships alone and loses nothing. | One plan for all of part 2b-iii. | The plan of part 2b-iii-a (BL-104), decision D-86 |
| A-48 | The design's C-9 puts the five plots under one Plots tab "with a picker" and does not say which kind. Which? | A row of the five names under the tab, the one shown marked (the widget of the inputs' order toggle and of the results table's order), wrapping onto a second row at 1024 x 700. | Each plot stays one click from the Plots tab and its name stays in view, as its tab was; no new text. A drop-down would hide the names behind a click. Owner question 70. | A drop-down list of the five. | The plan of part 2b-iii-a (BL-104), decision D-87 |
| A-49 | C-9 names the last tab "Results"; it reads "Results table". Which word? | "Results table", as today. | No on-screen word changes that the scope does not need. Owner question 71. | The tab reads "Results". | The plan of part 2b-iii-a (BL-104), decision D-89 |
| A-50 | Stage 1's whole-branch review (C-3) found the plots' reference lines in the failing red on a passing design, against C-5 ("red means a failing check and nothing else"), and named two ways: the lines leave the red, or C-5 is narrowed. Which? | The lines leave the red: the requirement lines and the pull-out point in the strong text colour, the governing limit and the time to it in violet (the torque plot's limit colour). Red stays only on a sweep row that does not fit, and its legend entry. | The approved design's rule stands, and no colour is new to the plots (violet and the text colour were drawn there); the two marks the finding did not name break the same rule and follow it. The cost: a requirement line no longer looks like a warning. Owner question 72. | Red again for the lines and marks, and C-5 narrowed; or red only while the line's check fails. | The plan of part 2b-iii-a (BL-104), decision D-90 |
| A-51 | C-9 and 4.4 do not say which plot the Plots tab opens on, or what a click on the tab does while a plot shows. Which? | The plot shown last (Torque vs temperature until another is picked), and a click on the tab while a plot shows keeps that plot. View state: in no design file, share link or undo step. | Going back to the plot one was reading stays one click, as its own tab was; a click on the tab of the view shown changes nothing, as on every other tab. No owner question: the builder's call, not the owner's answer. | The tab always opens on Torque vs temperature. | The plan of part 2b-iii-a (BL-104), decision D-88 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
```

Replace:

```markdown
- 2026-11-26: A-46 added with BL-098 (branch `magcoupling/columns-hold-still`): a note's diagram is scaled to the Explain panel's column. It is the builder's call, not the owner's answer.
- 2026-11-23: A-33, A-34, A-35 and A-36 added with the plan of slice B's stage 2, part 2b-ii-b (BL-094). Each is an assumption, not the owner's answer.
```

with:

```markdown
- 2026-11-26: A-46 added with BL-098 (branch `magcoupling/columns-hold-still`): a note's diagram is scaled to the Explain panel's column. It is the builder's call, not the owner's answer.
- 2026-11-27: A-47, A-48, A-49 and A-50 added with the plan of slice B's stage 2, part 2b-iii-a (BL-104). Each is an assumption, not the owner's answer.
- 2026-11-27: A-51 added by the fix wave of that plan's review (UX-4): the Plots tab opens on the plot shown last (decision D-88). It is the builder's call, not the owner's answer.
- 2026-11-23: A-33, A-34, A-35 and A-36 added with the plan of slice B's stage 2, part 2b-ii-b (BL-094). Each is an assumption, not the owner's answer.
```

**`docs/analyses/2026-11-18-owner-questions.md`**

Replace:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 47 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54, 55, 56, 57, 58 and 59 on 2026-11-23, questions 60 to 65 on 2026-11-25, questions 66 to 69 on 2026-11-26). Part D, five that matter only if the tools stop being generic.

```

with:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 53 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54, 55, 56, 57, 58 and 59 on 2026-11-23, questions 60 to 65 on 2026-11-25, questions 66 to 69 on 2026-11-26, questions 70, 71 and 72 on 2026-11-27). Part D, five that matter only if the tools stop being generic.

```

Replace:

```markdown

### Linkage simulator
```

with:

```markdown

### 70. Should the Plots tab pick its plot from a row of names or from a drop-down?

Added 2026-11-27 (the next free number). The five plots now share one tab, "Plots" (the design's decision C-9). Under the tab a row shows the five plots' names, the one shown marked; a click on a name shows that plot. The design calls it a picker and does not say which kind. A drop-down would take one line and hide the names behind a click.

- **If you do not answer:** the row of names.
- (a) As the default.
- (b) A drop-down list.

Source: UX design decision C-9 and section 4.4; A-48; BL-104 plan, decision D-87.

### 71. Should the last tab read "Results" instead of "Results table"?

Added 2026-11-27 (the next free number). The tab row is now Size and torque, Geometry, Plots and Results table. The design's decision C-9 names the last tab "Results"; it has read "Results table" since it was made, and no word changed that did not have to.

- **If you do not answer:** "Results table".
- (a) As the default.
- (b) "Results", as the design writes it.

Source: UX design decision C-9 and section 4.4; A-49; BL-104 plan, decision D-89.

### 72. In the plots, should the required torque and the limits be drawn in red?

Added 2026-11-27 (the next free number). The design says red means a failing check and nothing else (decision C-5). The plots drew the required torque, the sweeps' required floor and slip heating's limit as red lines, and the pull-out point and the time to the limit as red dots, whether the design passed or not. They are now drawn in the bright text colour (the requirement lines and the pull-out point) and in violet (the limit and the time to it, the violet the torque plot already gave its limit line). Red is left to a sweep row whose design does not fit.

One cost in the two sweeps: their legend draws every entry as a dot in its colour, whatever its shape, so "Required floor" (a dashed line) and "This design" (a ring) now have the same bright dot there, where "Required floor" and "Does not fit" shared the red one before. On the plot itself the dashed line and the ring still look different. If that should change, the smallest change is the sweeps' "This design" ring in blue, as the torque-temperature plot draws its own.

- **If you do not answer:** as described.
- (a) As the default.
- (b) Red again for those lines and dots ("red is also a limit line"), and the design's C-5 narrowed to say so.
- (c) Red only while the check a line stands for fails.

Source: UX design decision C-5; the slice B stage 1 review, finding C-3; A-50; BL-104 plan, decision D-90.

### Linkage simulator
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, its second (the drawing's selections and the word budget; BL-094) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md`, its third (the left column and the nothing-lost test; BL-096) by the plan `docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md`; part 2b-iii and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

with:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, its second (the drawing's selections and the word budget; BL-094) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md`, its third (the left column and the nothing-lost test; BL-096) by the plan `docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md`; part 2b-iii is built in three, and its first (the Plots tab; BL-104) by the plan `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md`; parts 2b-iii-b and 2b-iii-c and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

**`magcoupling-rs/README.md`**

Replace:

```markdown
2026-10-13 (G-5), the geometry view, the five plots, the clamp and the results table;
decisions M42-1 and M42-2), under the end-effect banner when
f_end <= 0. Every result is recomputed with `compute_all` (every correction on) each frame after the
```

with:

```markdown
2026-10-13 (G-5), the geometry view, the five plots, the clamp and the results table;
decisions M42-1 and M42-2; since BL-104 the five plots share one Plots tab, whose picker names
them, and the tab opens on the plot shown last: UX overhaul decision C-9), under the end-effect banner when
f_end <= 0. Every result is recomputed with `compute_all` (every correction on) each frame after the
```

Replace:

```markdown
  and the hub are drawn round: a 2-gon has no corners).
- **Plots** (`plots.rs`, egui_plot 0.33; M42-5, M42-6): torque against magnet temperature (the
  nominal pull-out, the band of the production variation allowance, the required torque, the
```

with:

```markdown
  and the hub are drawn round: a 2-gon has no corners).
- **Plots** (`plots.rs`, egui_plot 0.33; M42-5, M42-6; since BL-104 under one Plots tab, whose
  picker names the five plots in the order their tabs had and marks the one shown, UX overhaul
  decision C-9): torque against magnet temperature (the
  nominal pull-out, the band of the production variation allowance, the required torque, the
```

Replace:

```markdown
  Magnets per ring, Required torque). The series come from this frame's results through the
  engine's own closed forms (tests pin them to the engine's torques), only for the tab shown; when
  f_end <= 0 the torque plots are grey. Points and reference lines that are not finite are left
```

with:

```markdown
  Magnets per ring, Required torque). The series come from this frame's results through the
  engine's own closed forms (tests pin them to the engine's torques), only for the plot shown; when
  f_end <= 0 the torque plots are grey. Points and reference lines that are not finite are left
```

Replace:

```markdown
  "Torque" lost its bar, on `main` too); the plot may now paint that far into the region's
  margin (`Y_LABEL_OVERHANG`).
- **Clamp** (`clamp_drawing.rs`; M42-7; not offered while the shaft-clamp section is paused, below): drawing.py's end and top views of the one-piece slotted
```

with:

```markdown
  "Torque" lost its bar, on `main` too); the plot may now paint that far into the region's
  margin (`Y_LABEL_OVERHANG`). Red marks a failing check and nothing else (decision C-5, since
  BL-104): in a plot only a sweep row that does not fit, and its legend entry, is red; the
  required torque and the sweeps' required floor are drawn in the strong text colour, the
  governing limit in violet in both plots that draw it, the time to the limit in violet and the
  pull-out point in the strong text colour (until BL-104 all five were red whatever the
  design's verdict: the slice B stage 1 review, C-3). Each axis draws a tick number at full strength or not at all
  (`X_TICK_SPACING`, `Y_TICK_SPACING`: egui_plot's own thresholds, 60 and 20 points apart; past
  them it faded a number in, to full strength at 80 and 30, as faint as 1.3:1 on the way: the
  stage 1 review, E-7). At 1024 x 700 and 1280 x 800 that threshold leaves the horizontal axes
  of four plots one or two numbers, or none, as on `main` (BL-105).
- **Clamp** (`clamp_drawing.rs`; M42-7; not offered while the shaft-clamp section is paused, below): drawing.py's end and top views of the one-piece slotted
```

Replace:

```markdown
  muted (`look::mark_ui`, `look::tab_ui`, `look::quiet_text`). It is never a text colour and never on a value or a check;
  a failing check is the theme's own red, `#ff4d57`, a lighter one. That red is not yet a
  failing check and nothing else, as decision C-5 asks: the plots draw three reference lines in
  it whatever the design's verdict (the required torque, the sweeps' required floor and slip
  heating's governing limit, each with its legend dot), as they did before this stage, so a
  design that passes every check shows red on those tabs. Stage 2, which rebuilds the Plots
  tab, settles it (the plan's "What follows"). In the plots the grid, the ticks, the axis labels and the value
  label egui_plot draws at the pointer take the quiet colour too (egui_plot draws all four from
```

with:

```markdown
  muted (`look::mark_ui`, `look::tab_ui`, `look::quiet_text`). It is never a text colour and never on a value or a check;
  a failing check is the theme's own red, `#ff4d57`, a lighter one. Since BL-104 that red is a
  failing check and nothing else in the plots too, as decision C-5 asks: until then they drew
  three reference lines and two marks in it whatever the design's verdict (the Plots bullet
  above says where those went). In the plots the grid, the ticks, the axis labels and the value
  label egui_plot draws at the pointer take the quiet colour too (egui_plot draws all four from
```

Replace:

```markdown
through that lesson link (`lesson_link_opened`: the Explain panel shows "What to change" and the closed "Where this
comes from" fold), judges the left side on the Geometry tab (`same_column`, since BL-096: the size controls over the two shut folds), opens the Detailed inputs fold, hovers the first Key design row (the tooltip shows the help and its default
and no path or spreadsheet cell, the Design menu's "Show spreadsheet details" toggle being off, as `details_toggle`
```

with:

```markdown
through that lesson link (`lesson_link_opened`: the Explain panel shows "What to change" and the closed "Where this
comes from" fold), opens the Plots tab and judges the tab row (Size and torque, Geometry, Plots and Results table) and the picker of the five plots under it, then picks Slip heating and finds its legend (`plots_picker`, since BL-104), judges the left side on the Geometry tab (`same_column`, since BL-096: the size controls over the two shut folds), opens the Detailed inputs fold, hovers the first Key design row (the tooltip shows the help and its default
and no path or spreadsheet cell, the Design menu's "Show spreadsheet details" toggle being off, as `details_toggle`
```

Replace:

```markdown
headings, the orientation, the entry's headings and its lesson link, and no dashboard, the
toggle, the note's parts, the Geometry tab, the trace link, the face gap's name, the last row and the card's heading, caption, Auto line, held row and row starts as the page draws them). Its other step checks the hub page at `/tools/` and the linkage
app at `/tools/linkage/`. Gate 13 of `gate.sh` runs the script's own rules under node with a stub agent and no browser
```

with:

```markdown
headings, the orientation, the entry's headings and its lesson link, and no dashboard, the
toggle, the note's parts, the tab row and the Plots tab's picker, the Geometry tab, the trace link, the face gap's name, the last row and the card's heading, caption, Auto line, held row and row starts as the page draws them). Its other step checks the hub page at `/tools/` and the linkage
app at `/tools/linkage/`. Gate 13 of `gate.sh` runs the script's own rules under node with a stub agent and no browser
```

Replace:

```markdown
| `src/engine/clamps.rs` | The paused section hides exactly the `clamps.` and `materials.screws.` paths and only while paused; each of its 28 inputs (25 clamp inputs, 3 screw classes), at every probe design, at both slider ends or every choice, moves no result outside it. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input, a 1280 x 1024 screen): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame; the value lands on the step's decimals (1.41) and stepping back lands on the default exactly; a typed value snaps to the step and is clamped into the range; pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept until edited and flagged; the changed dot and per-field reset; the session baseline (UX overhaul slice A: a panel built `with_baseline` opens on it and Reset all returns to it, `new()` keeps the spreadsheet's values, nothing moves it; the changed dot, a row's reset and its hover, the tooltip's Default line and the Equation panel's changed tag follow it; the status line reads "Starting design" at it; Open the example design is one undo step and is not drawn on a panel whose baseline is the spreadsheet's design; the new texts keep the text rules of the design's section 8.3; the settled design follows settled edits, undo, redo and an opened link); the axial length override starts blank and enters at the ring's length; a selector switches the branch (no back iron); a text input edits the part name and its note follows; every group opens and draws every input; the dashboard's stored-3D label, markers, end-effect banner (over the dashboard and the centre region) and space-claim overshoot; the results table's search and its export requests; undo and redo (buttons and shortcuts; one step per drag, per nudge, per typed name; Ctrl+Z in a text field left to the field; reset all undoable); save and load requests, a loaded and a refused design file, the share link on the clipboard and a broken link; the host's report; Torque -> Magnets: the solved design shown, the debounce (no solve per frame, none while a button is down, a repaint after a solve), the Size tab's locked Auto slider (Overall length, Outside diameter or Magnets per ring, at the solved value; with ring-size Auto the cap OD and thread rows are locked too, following the cup), leaving the mode keeps the value as one undo step and solves a pending change first, an unreachable target shows the best value, the free-variable picker, a share link carries the sizing state, the JSON export holds the design shown, typing in the results search does not defer the solve; a held arrow key is one undo step, a slider's value box being typed in is an edit of the design (one step on Enter), the results search holds back no undo step, only this frame's rows count as design fields, a start-up share link is no undo step, a host can keep Ctrl+Z for itself; idle frames change nothing with every group open and off-grid values; the measured drag enters at the model's drag, unrounded; every drawn text and hover text has glyphs in egui's default fonts (every tab of the centre region and the geometry callouts' hover texts); the panel draws inside an `egui::Window` (M5); the Geometry tab follows the design shown (a live redraw; the sized design in Torque -> Magnets); each plot tab draws its plot, the design's marker at the edited pull-out in the edit's frame; the clamp tab draws the recommended clamp or drawing.py's message; a ~930 px window shows a row's label (filling its 120-point column) and value; the ordering (O-1 to O-8): every group of both orders opens and draws its rows and headings, the Advanced sections closed until their heading is clicked; the order toggle changes no input, design, share link or undo history; the filter box shows the matches alone (count, run headings; a matched row nudges; no undo step for typing; the workbook order's headings; none; blank); a focused advanced input opens its group and heading past a filter; the results by group (every heading drawn, a heading opens and closes its rows, the engine order without headings); the failing filter shows exactly the failing checks with five red and three amber badges; a click on an input's label frames exactly the dashboard rows it drives and its own row until a second click; a click on a result frames exactly the Key design rows it reads and counts the traced rows per group; the traced filter (disabled without a trace, the traced count, the headline's traced heading); a heading shows the worst level of its group's checks (the headline and the closed coupling model red, the mass none); a heading click while searching leaves its group as it was, its hover text saying why; the failing filter with a search no failing check matches says no result matches; an input whose results have no equation record traces none, says so, and the traced filter then says the trace marks nothing (Clear trace turns the filter off); reading a traced result keeps the input's trace and its traced rows, a result without a record keeps it, a result outside it replaces it; idle frames in the workflow order's filtered view change nothing; the final review's fixes: out of the end-effect range the table's headline rows carry exactly the dashboard's badges (none on the hot minimum, the screw and the gearbox check) and the failing filter leaves the hot minimum out (seven red, four amber); a search that hides the traced rows says no result matches, and a heading counts the traced rows it shows; a trace moves no input row, so a second click at the same point ends it; the locked free variable's label traces it with no edit or undo step; views stay in their own area: on a 1280 x 620 page and the laptop screen, with the Equation panel at its starting height and dragged as far up as it goes, on a 1280 x 300 page and inside an `egui::Window` (the linkage Tools window's 1100 x 700), no view (geometry, the five plots, clamp, results) paints into the panel or takes a click inside it (shapes split from the panel's by its separator line, widgets by its resize handle), the scrolling views fill the space down to it, and the window keeps its size; the panel opens at about 45 % of the centre region and its drag stops `VIEW_STRIP` short of the region's top, then follows the pointer back down; every view draws beside the panel in a tiny window; both spreadsheet buttons (the header's and the results table's) queue `magcoupling-results.xlsx`, which reads back with the four sheets, the edited face gap as a number and the share link, follows the input order shown and, in Torque -> Magnets, holds the solved length (noted) and the solve's outcome; slice 1 (BL-055): the panel opens on Size and torque under the welcome strip, which closes for the session and leaves no undo step; the dashboard's status line counts the failing checks at the default design and after a fix; the Required slider edits `metal.required_min_Nm` (one undo restores it) and the Hot-low gap follows it; the consequence lines follow an input edit; Auto locks the matching slider, re-solves after the debounce, leaves idle frames alone and reports an unreachable requirement; opening the Size tab no longer changes the sizing mode; every new string has glyphs in egui's default fonts; slice 2 (BL-056): the Show spreadsheet details toggle starts off, a click turns it on and off, its hover says what it shows, and it is in no design file, share link or undo step and survives a loaded file, undo, reset all and a share link opened as the session's start; off, the face-gap row's hover names no path or cell and the pull-out's hover on the dashboard and on the Size tab no path, cell or correction, on they do; the correction ids and their legend, the results table's cells (pinned tests switch the details on), the Equation panel's cell and corrections lines and a result-without-record's cell, and the assumptions' sources show only with the details; at 930 px the table's label and value stay on screen with the cell and marker columns and without them; the toggle's label and hover, the legend, the marker heading and scope, the original-note prefix and the example design's status start and hover pass `plain_text_violations`; every text, the details' included, has glyphs in the default fonts; one name per quantity (`size_controls.rs` tests too): the Size tab's readout labels equal the input and dashboard rows' names (Required torque, Hot-low torque; one Hot-low torque readout in the centre beside the dashboard row of that name), the Auto row and its hover name the required torque, the footnote and the Outside diameter and Overall length hovers say what they move (the ring radius; both rings' magnet length) within the help caps (the length hover within its own cap, 4 sentences and 80 words, since BL-074: it also says that the first move from a blank axial magnet length sets both rings to one length, a test seen red first), each label shows its hover, the open equation's header is the display label, and Auto's no-valid-value line names the ring radius; the Outside diameter slider's lowest value is one expression, `slider_minimum` (the user's Minimum OD, else the range's start or a loaded smaller design's own diameter, held to the Minimum OD's range), which the controls pin on a loaded smaller design's first frame (`the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown`, slice 3); each Size slider makes its design through its own mapping, `SliderMapping`, which the torque card's steps share, the Outside diameter's at the slider's own minimum (at a Minimum OD below the default floor it reaches a diameter that floor refuses), the count's refusing a value a cast would change (not finite, not whole or past what an i64 holds) and neither the gap's nor the count's checking a range (`each_size_slider_makes_its_design_through_its_own_mapping`, slice 3); slice 2, help batch A (T7): each Key design row draws its hint line under it and above the next row (once: its own group open adds none), each of the five Size-tab controls its hint line under its label and above the next control's, and at a 930-point window every hint stays inside the inputs side, at least one wrapped; the panel's own strings (the assumption reset, the order toggle, the workflow headings, the trace banners and hint, the explorer's no-equation text and tag, the end-effect banner and greyed hover, the geometry notes, the welcome strip) pass the plain-words guard and name none of the engine's internal words; with Show spreadsheet details on, the dashboard keeps its width frame after frame (each value and its correction ids flow as one wrapped text); with Auto on (Torque -> Magnets, the axial magnet length free), Start here's step-1 note draws its magnet-length entry as plain text " (held by Auto)" once, a click on that label opens no row and keeps the note open, and its live values are the design shown's: the pull-out watch-out, which moves with the magnet length, reads the solved length's numbers and not the stored inputs' (Review Focus RF-1); the shaft-clamp pause (ruling 11): the Clamp tab and the clamp inputs (the workflow order's clamp sections, the original-sheet order's Shaft clamps group, the filter's matches) show only while the section is not paused, its two assumptions leave the Assumptions view and the banner, the shipped panel starts from the switch and passes its pause to the fixes (at the clamp's limit the hot-low torque's fix is drawn only while paused), the header's banner and the trace banner (`the_panel_passes_its_pause_to_the_fixes_the_banner_and_the_trace`), a design file written before the pause and a share link keep their clamp values in a paused panel (the engine still computes the section); one test, `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`, walks `every_shown_text` (every drawn text, and the hover texts of inputs, dashboard rows, table rows, callouts and the named static hovers; the rows through the panel's own filters) once per state: every such text has glyphs paused or not, and while paused none of them, nor a shown note's fields and drawn change entries, the Start here walk and hover, a counting guide's name and Why, the shaft joints' Problem as drawn on three designs that fail them (BL-064: the inner hub slipping, over its stress limit, on an uncooled wheel shaft), a Why link's reason or a shown assumption's name, rationale and source, names the clamp (the guard; no allow-list entry; no shown note's change entry or equation leads into the section; with the section shown the same enumeration names the clamp and reaches the Clamp tab, the clamp dashboard row, a clamp input row, the Shaft clamps group and the clamp friction assumption; while paused it reaches the label and help of every shaft-joint input and result and, each whole as drawn, their Shrink fit and Fitting temperatures sections and the Shaft joints headings of the original-sheet inputs and of the results table, BL-064); panel tests draw the whole section (`Harness::of` clears the pause) unless a test pauses it; slice 2 (T15): the gearbox ripple row's Why link opens the slip note, its first sentence drawn once and no input changed; slice 2 (T19, reachability, spec 3.7): the Size tab's Why links under Magnets per ring and Outside diameter, and on the torque card's heading (slice 3), open the torque note, and those at the end of the Required and Hot-low rows the torque-band note, each on its own in the Equation panel (clicked on the text's last character, `test_support::last_characters`, since a wrapped row's text rect also covers the slider before it); with an odd magnet count the Outside diameter control and its link are not drawn and the Magnets per ring link and the torque card's are; the link under each copy of the end-effect banner (over the dashboard and the centre) opens the end-effect note, and without the banner there is none; every dashboard row opened with Explain on shows its note, the closest note, an assumption or its own reason, never `NO_NOTE` (`explain_never_shows_no_teaching_note_on_a_dashboard_row`); the closest-note link, the one copy of the title on its line left of the dashboard, opens that note (`click_the_closest_note_link`, shared with the gearbox row's Explain test); with Explain on, a value with nothing upstream to read draws `NO_NOTE` and `START_HERE_HINT` once each, and Explain off draws neither; BL-059: the gearbox row's own Why link, the copy of the torque-band title under the row's label (the hot-low, hot-minimum and cold-high rows draw it first), opens that note and changes no input; BL-064: the shaft-joint row's own Why link, the one copy of the shaft-joint note's title on the dashboard, under the row's label, opens that note and changes no input (`the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note`, sharing `row_s_why_links` and `click_opens_note` with the gearbox row's test); with Explain on the gearbox row, which has no equation or note of its own, offers its Why link's note as the closest note, and the link beside the lead opens it; at the defaults the failing filter lists five red checks and one amber (the daily-cycle screen at the entered bondline, BL-063), the gearbox check's row once with a red badge (`table_badges_on`, shared with the end-effect test); slice 3 (T5): the Size tab draws the torque card in place of the label under the readouts, its heading and each of its lines once, one under the next, a click on the heading closing and opening it as one on its arrow does (`the_size_tab_draws_the_torque_card_in_place_of_the_label`); a typed Required value reaches the card in the frame that applies it (`a_required_edit_reaches_the_card_in_its_own_frame`); under Magnets per ring two lines, the ring-size line with Auto off or on the magnet length and the fit line, the card drawing the Auto line of each Auto state, the count's steps (at 4 magnets, and at 40 where 40 blocks fit, the side past the slider reading "outside the slider's range") and following each edit; idle frames compute the card once, a gap edit, a clamp-pause toggle and a new Minimum OD once each (the drawn Outside diameter row's -1 mm side then reading "outside the slider's range"), and turning Auto off at a solved length or Auto on the count at its best computes nothing, while the design's results the cache is handed are computed once per design, none for idle frames, the pause or the Minimum OD (`idle_frames_compute_the_card_once_and_an_edit_again`); a drag of the gap slider with Auto on the magnet length computes the card, and its design's results, once for each design it draws (`a_drag_computes_the_card_once_for_each_design_it_draws`); idle frames change no input with the card open and closed, at the clamp pause; slice 3 (T6, headless, each test pausing the clamp section before the card's first frame, but for the Equation panel's check on the screen, which draws every view unpaused): at 1280 x 800 the card sits under the Pull-out readout and above the drawing, open, every line drawn and the drawing at least 160 points, at the defaults and the smoke design; at 930 x 800 it starts closed (the region under 400 points), at short magnets too, closed no taller than the label it replaced, and its Why link wraps under the heading inside the region and opens the torque note; at 1080 x 745 no back iron starts closed (its open card would take the drawing under its floor) and short magnets open; each rule opens it at its boundary, a region exactly 400 points wide or exactly as tall as the open card, the item spacing and the floor, while 399.99 points or 0.01 point less, or the Equation panel open, closes it (`the_card_s_default_opens_at_each_rule_s_boundary`); with no user choice a window resize decides it again (open at 1280 x 800, closed again at 930 x 800), a change of its height alone too (at 1080 points wide open at 800 points tall, closed at 700, open again at 800); the Equation panel closes it and its closing opens it, and no view, the card open or closed, paints into the panel, on the screen or inside an `egui::Window`; a click on the heading or the arrow wins for the session, through a design edit, a resize, the Equation panel (the drawing keeping its 160 points beside it), Reset all and a load; with no user choice Reset all and a load keep the default decided for the view (at 1080 x 800 no back iron's closed card stays closed through Reset all and a load of 8 magnets, which would open it, and the defaults' open card stays open through a load of no back iron) (`reset_all_and_a_load_keep_the_card_s_default_with_no_user_choice`); drags of the gap, the Outside diameter across the fit edge and Required across hot-low (checked on every drag frame), the assumptions banner, and a side panel narrowing the region at no back iron leave it as decided; opened at 930 x 800, and at 1280 x 800 at a design file's calibration coefficient of 1e300, whose 887-character lines rule 3 starts closed, its body scrolls (the mouse wheel brings its last row into view) and the drawing keeps 160 points; its measured height is its drawn height within a point at 1280 x 800 and 930 x 1024, and at 1280 x 1024 at that calibration coefficient, and at 930 x 800 its body is the cap (`the_card_s_measured_height_is_the_drawn_height`); the look: one line in the accent, as wide as the active tab and just under its name, which follows a click to another tab, the active tab's name in the strong text colour and another's in the quiet one (a theme's weak colour; the text colour in egui's own visuals, which set none), and one accent square of 8 points before the title's first glyph, at the title's height (`the_active_tab_is_underlined_in_the_accent_and_no_other`, `the_brand_mark_is_one_accent_square_before_the_title`); the changed dot and a trace's banner and frames are the host's link colour, not the colour it gives a selected choice's name (`the_panel_s_marks_are_the_host_s_link_colour`); a readout's equation wider than the window is set smaller so its tooltip stays inside the window, and one that fits keeps the tooltip's size (`src/gui/readouts.rs`, `an_equation_wider_than_the_window_is_set_smaller_and_another_keeps_its_size`); BL-097: on the app's starting design, Auto on until solved and Auto off, Tab from the top of the page through the Size column changes no input, no result and no Minimum OD, makes no undo step and selects nothing (`tab_through_a_length_auto_left_off_its_step_changes_nothing`, with the `app` feature); with a face gap of 1.4123 mm and a requirement past its slider's range loaded, Tab past every value box of the Size column and of the open Detailed inputs does the same (`tab_past_every_value_box_keeps_a_value_off_its_step_or_outside_its_range`); Enter on a value box with nothing typed, or with the number it shows typed again, keeps the 1.4123, while a typed value edits on Tab (one undo step, the row selected), on Enter and on a click elsewhere, and an arrow key steps it onto the step (`a_typed_value_and_an_arrow_key_edit_a_value_off_its_step_and_its_own_text_does_not`); the rule itself (`src/gui/input_ui.rs`, `a_value_box_s_own_text_is_no_typed_number`); a number typed into a tiny value's box edits, one undo step, clamped to the range as egui clamps it: the vacuum permeability, the NdFeB expansion and the cap's field integral (`a_number_typed_into_a_tiny_value_s_box_edits`); BL-100: at 1280 x 800 and 1024 x 700 with the Detailed inputs open, after each press of Tab and of Shift+Tab every focused widget of the left column is whole in its view once the scroll has settled, scrolled no further than it was cut plus egui's gap at the edge; nothing is selected or edited; a wheel then takes the focused row out of view and it stays out; an idle frame asks for no repaint; a Down arrow key's move is followed too (`the_left_column_follows_the_keyboard_s_focus_and_fights_no_wheel`); the same walk through a check's entry in the Explain panel at 1024 x 700 (`the_explain_panel_follows_the_keyboard_s_focus`) and through the Geometry tab's callout list (under the open Equation panel) and the Clamp tab (`the_centre_s_scrolling_views_follow_the_keyboard_s_focus`); each fails with its area's call of `follow_focus` taken out; the left column wheeled 300 points down stays where the wheel left it while Tab, then Shift+Tab, pass the top bar, the centre and the right column (`a_wheeled_column_holds_still_while_the_focus_is_on_no_widget_of_its_own`; it fails with `follow_focus`'s clip check taken out); BL-098: at 1024 x 700 and 1280 x 800 every shown note in turn, then the note list, a check's entry, a tile's and the orientation, leaves the right column 300 points wide (egui's `PanelState`) with its right edge where it was (`the_right_column_keeps_its_width_whichever_note_is_shown`); so does the check list's material warning on a design with no back iron and on one whose sleeve conducts better than 316L, and its Why link opens its note (`the_right_column_keeps_its_width_with_a_material_warning_shown`); in the app's theme and font, on the starting design and on the failing example design with the spreadsheet details on, so does every entry the Explain panel can show: the orientation, the note list, every note, check, tile, Size column row and input, and every result of the results table (`the_right_column_keeps_its_width_whatever_entry_is_shown_in_the_app_s_own_font`, with the `app` feature); a diagram in a place 284 or 150 points wide is the picture scaled to that width with every shape and label inside it, and is 300 x 130 where it has room (`src/gui/diagrams.rs`, `a_diagram_takes_the_width_it_is_given_scaled_as_one_picture`) |
| `src/gui/geometry.rs`, `src/gui/geometry_view.rs` (feature `gui`) | The end view draws every part from the results (the pocket's corners at the pocket corner radius, the rings at the retainers' diameters, the key, the shaft joints' 3 mm, from where it crosses the bore), faceted blocks on their flats and arcs as sectors; each end-view callout's line measures the gap it names (the corner gap along face 1's normal, from the dashed circle the inner corners sweep to the outer flat); red below zero and for a clearance below its target; the side view spans the housing in effect (a 50.8 mm override: the 53.6 mm cavity, the hub centred on it, the liner and sleeve the retainer span in effect); the space-claim callouts name each overshoot, in red, with the exceeded axes' dashed lines red (every axis at once too), exactly at the claim inside and the next number below over, and a NaN claim amber; a claim far past the pieces (1e300, 1e39, -1e39) is left off with one note, drawn at `CLAIM_REACH` times the extent and not past it; the design checks show while each inconsistency holds (equality is not below); the wall-rule line unless there is no back iron (the cup wall input named by its label), asking for the wall the cup wall check compares with, so the first slider step that passes is not asked for the verdict's rounded-up suggestion; that wall rounded up at its last digit, so a wall shown equal to it passes (12 poles, a 1.23 T design flux density and a 1.95 mm wall fail the check, and the line asks for 1.951 mm); a minimum length rounds up at its last shown digit (past a power of ten too, and a length already on a digit stays), and one not finite or not positive shows as it is; pole counts from a file draw at most 200 blocks; two poles draw their four blocks in a round pocket and hub (a 2-gon's corners would fly off), three a triangular pocket; a NaN draws nothing wrong and says so (the parts dropped counted per view). Painted: both views at one scale, the largest that fits (`side_by_side`: none without room or for an extent that is not finite); the cup OD, the shaft and the face gap at their pixel lengths, the cup depth and the cap as painted rectangles; the callout texts in their colours (the default clearance red, its line too); an overshoot red; hovering a dimension shows its result's hover text; each dimension tag sits on its own plate of the drawing's background; a NaN view and a tiny screen draw; a region of no size paints no view; a huge claim keeps the pieces to scale with its note; in a short region (400 points down to none) the view ends at its foot, the drawing keeping 160 points while the list keeps three rows, then shrinking to scale (`drawing_height`); in a region of exactly `drawing_floor` (slice 3: the room the torque card leaves, pinned at 212.125 points in egui's default style, 160 over three 17.375-point list rows) the drawing keeps its 160 points, and a point less it shrinks below them; a dimension's hover and a callout's in the list under the drawing name their result's path only with the spreadsheet details; the design checks name their inputs by their labels and pass the plain-words guard; the key in both states of the shaft-clamp pause, at the shaft joints' key width (BL-064: each end of its slider; one typed wider than the bore from the axis; a width or a keyway depth that is not a number not drawn and counted; none at a keyway depth of 0 or less, nor at a key width of 0 or less), never at the paused clamp's, and the clamp-boss design check only while the shaft-clamp section is not paused |
```

with:

```markdown
| `src/engine/clamps.rs` | The paused section hides exactly the `clamps.` and `materials.screws.` paths and only while paused; each of its 28 inputs (25 clamp inputs, 3 screw classes), at every probe design, at both slider ends or every choice, moves no result outside it. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input, a 1280 x 1024 screen): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame; the value lands on the step's decimals (1.41) and stepping back lands on the default exactly; a typed value snaps to the step and is clamped into the range; pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept until edited and flagged; the changed dot and per-field reset; the session baseline (UX overhaul slice A: a panel built `with_baseline` opens on it and Reset all returns to it, `new()` keeps the spreadsheet's values, nothing moves it; the changed dot, a row's reset and its hover, the tooltip's Default line and the Equation panel's changed tag follow it; the status line reads "Starting design" at it; Open the example design is one undo step and is not drawn on a panel whose baseline is the spreadsheet's design; the new texts keep the text rules of the design's section 8.3; the settled design follows settled edits, undo, redo and an opened link); the axial length override starts blank and enters at the ring's length; a selector switches the branch (no back iron); a text input edits the part name and its note follows; every group opens and draws every input; the dashboard's stored-3D label, markers, end-effect banner (over the dashboard and the centre region) and space-claim overshoot; the results table's search and its export requests; undo and redo (buttons and shortcuts; one step per drag, per nudge, per typed name; Ctrl+Z in a text field left to the field; reset all undoable); save and load requests, a loaded and a refused design file, the share link on the clipboard and a broken link; the host's report; Torque -> Magnets: the solved design shown, the debounce (no solve per frame, none while a button is down, a repaint after a solve), the Size tab's locked Auto slider (Overall length, Outside diameter or Magnets per ring, at the solved value; with ring-size Auto the cap OD and thread rows are locked too, following the cup), leaving the mode keeps the value as one undo step and solves a pending change first, an unreachable target shows the best value, the free-variable picker, a share link carries the sizing state, the JSON export holds the design shown, typing in the results search does not defer the solve; a held arrow key is one undo step, a slider's value box being typed in is an edit of the design (one step on Enter), the results search holds back no undo step, only this frame's rows count as design fields, a start-up share link is no undo step, a host can keep Ctrl+Z for itself; idle frames change nothing with every group open and off-grid values; the measured drag enters at the model's drag, unrounded; every drawn text and hover text has glyphs in egui's default fonts (every tab of the centre region and the geometry callouts' hover texts); the panel draws inside an `egui::Window` (M5); the Geometry tab follows the design shown (a live redraw; the sized design in Torque -> Magnets); each plot draws its plot, the design's marker at the edited pull-out in the edit's frame; the clamp tab draws the recommended clamp or drawing.py's message; a ~930 px window shows a row's label (filling its 120-point column) and value; the ordering (O-1 to O-8): every group of both orders opens and draws its rows and headings, the Advanced sections closed until their heading is clicked; the order toggle changes no input, design, share link or undo history; the filter box shows the matches alone (count, run headings; a matched row nudges; no undo step for typing; the workbook order's headings; none; blank); a focused advanced input opens its group and heading past a filter; the results by group (every heading drawn, a heading opens and closes its rows, the engine order without headings); the failing filter shows exactly the failing checks with five red and three amber badges; a click on an input's label frames exactly the dashboard rows it drives and its own row until a second click; a click on a result frames exactly the Key design rows it reads and counts the traced rows per group; the traced filter (disabled without a trace, the traced count, the headline's traced heading); a heading shows the worst level of its group's checks (the headline and the closed coupling model red, the mass none); a heading click while searching leaves its group as it was, its hover text saying why; the failing filter with a search no failing check matches says no result matches; an input whose results have no equation record traces none, says so, and the traced filter then says the trace marks nothing (Clear trace turns the filter off); reading a traced result keeps the input's trace and its traced rows, a result without a record keeps it, a result outside it replaces it; idle frames in the workflow order's filtered view change nothing; the final review's fixes: out of the end-effect range the table's headline rows carry exactly the dashboard's badges (none on the hot minimum, the screw and the gearbox check) and the failing filter leaves the hot minimum out (seven red, four amber); a search that hides the traced rows says no result matches, and a heading counts the traced rows it shows; a trace moves no input row, so a second click at the same point ends it; the locked free variable's label traces it with no edit or undo step; views stay in their own area: on a 1280 x 620 page and the laptop screen, with the Equation panel at its starting height and dragged as far up as it goes, on a 1280 x 300 page and inside an `egui::Window` (the linkage Tools window's 1100 x 700), no view (geometry, the five plots, clamp, results) paints into the panel or takes a click inside it (shapes split from the panel's by its separator line, widgets by its resize handle), the scrolling views fill the space down to it, and the window keeps its size; the panel opens at about 45 % of the centre region and its drag stops `VIEW_STRIP` short of the region's top, then follows the pointer back down; every view draws beside the panel in a tiny window; both spreadsheet buttons (the header's and the results table's) queue `magcoupling-results.xlsx`, which reads back with the four sheets, the edited face gap as a number and the share link, follows the input order shown and, in Torque -> Magnets, holds the solved length (noted) and the solve's outcome; slice 1 (BL-055): the panel opens on Size and torque under the welcome strip, which closes for the session and leaves no undo step; the dashboard's status line counts the failing checks at the default design and after a fix; the Required slider edits `metal.required_min_Nm` (one undo restores it) and the Hot-low gap follows it; the consequence lines follow an input edit; Auto locks the matching slider, re-solves after the debounce, leaves idle frames alone and reports an unreachable requirement; opening the Size tab no longer changes the sizing mode; every new string has glyphs in egui's default fonts; slice 2 (BL-056): the Show spreadsheet details toggle starts off, a click turns it on and off, its hover says what it shows, and it is in no design file, share link or undo step and survives a loaded file, undo, reset all and a share link opened as the session's start; off, the face-gap row's hover names no path or cell and the pull-out's hover on the dashboard and on the Size tab no path, cell or correction, on they do; the correction ids and their legend, the results table's cells (pinned tests switch the details on), the Equation panel's cell and corrections lines and a result-without-record's cell, and the assumptions' sources show only with the details; at 930 px the table's label and value stay on screen with the cell and marker columns and without them; the toggle's label and hover, the legend, the marker heading and scope, the original-note prefix and the example design's status start and hover pass `plain_text_violations`; every text, the details' included, has glyphs in the default fonts; one name per quantity (`size_controls.rs` tests too): the Size tab's readout labels equal the input and dashboard rows' names (Required torque, Hot-low torque; one Hot-low torque readout in the centre beside the dashboard row of that name), the Auto row and its hover name the required torque, the footnote and the Outside diameter and Overall length hovers say what they move (the ring radius; both rings' magnet length) within the help caps (the length hover within its own cap, 4 sentences and 80 words, since BL-074: it also says that the first move from a blank axial magnet length sets both rings to one length, a test seen red first), each label shows its hover, the open equation's header is the display label, and Auto's no-valid-value line names the ring radius; the Outside diameter slider's lowest value is one expression, `slider_minimum` (the user's Minimum OD, else the range's start or a loaded smaller design's own diameter, held to the Minimum OD's range), which the controls pin on a loaded smaller design's first frame (`the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown`, slice 3); each Size slider makes its design through its own mapping, `SliderMapping`, which the torque card's steps share, the Outside diameter's at the slider's own minimum (at a Minimum OD below the default floor it reaches a diameter that floor refuses), the count's refusing a value a cast would change (not finite, not whole or past what an i64 holds) and neither the gap's nor the count's checking a range (`each_size_slider_makes_its_design_through_its_own_mapping`, slice 3); slice 2, help batch A (T7): each Key design row draws its hint line under it and above the next row (once: its own group open adds none), each of the five Size-tab controls its hint line under its label and above the next control's, and at a 930-point window every hint stays inside the inputs side, at least one wrapped; the panel's own strings (the assumption reset, the order toggle, the workflow headings, the trace banners and hint, the explorer's no-equation text and tag, the end-effect banner and greyed hover, the geometry notes, the welcome strip) pass the plain-words guard and name none of the engine's internal words; with Show spreadsheet details on, the dashboard keeps its width frame after frame (each value and its correction ids flow as one wrapped text); with Auto on (Torque -> Magnets, the axial magnet length free), Start here's step-1 note draws its magnet-length entry as plain text " (held by Auto)" once, a click on that label opens no row and keeps the note open, and its live values are the design shown's: the pull-out watch-out, which moves with the magnet length, reads the solved length's numbers and not the stored inputs' (Review Focus RF-1); the shaft-clamp pause (ruling 11): the Clamp tab and the clamp inputs (the workflow order's clamp sections, the original-sheet order's Shaft clamps group, the filter's matches) show only while the section is not paused, its two assumptions leave the Assumptions view and the banner, the shipped panel starts from the switch and passes its pause to the fixes (at the clamp's limit the hot-low torque's fix is drawn only while paused), the header's banner and the trace banner (`the_panel_passes_its_pause_to_the_fixes_the_banner_and_the_trace`), a design file written before the pause and a share link keep their clamp values in a paused panel (the engine still computes the section); one test, `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`, walks `every_shown_text` (every drawn text, and the hover texts of inputs, dashboard rows, table rows, callouts and the named static hovers; the rows through the panel's own filters) once per state: every such text has glyphs paused or not, and while paused none of them, nor a shown note's fields and drawn change entries, the Start here walk and hover, a counting guide's name and Why, the shaft joints' Problem as drawn on three designs that fail them (BL-064: the inner hub slipping, over its stress limit, on an uncooled wheel shaft), a Why link's reason or a shown assumption's name, rationale and source, names the clamp (the guard; no allow-list entry; no shown note's change entry or equation leads into the section; with the section shown the same enumeration names the clamp and reaches the Clamp tab, the clamp dashboard row, a clamp input row, the Shaft clamps group and the clamp friction assumption; while paused it reaches the label and help of every shaft-joint input and result and, each whole as drawn, their Shrink fit and Fitting temperatures sections and the Shaft joints headings of the original-sheet inputs and of the results table, BL-064); panel tests draw the whole section (`Harness::of` clears the pause) unless a test pauses it; slice 2 (T15): the gearbox ripple row's Why link opens the slip note, its first sentence drawn once and no input changed; slice 2 (T19, reachability, spec 3.7): the Size tab's Why links under Magnets per ring and Outside diameter, and on the torque card's heading (slice 3), open the torque note, and those at the end of the Required and Hot-low rows the torque-band note, each on its own in the Equation panel (clicked on the text's last character, `test_support::last_characters`, since a wrapped row's text rect also covers the slider before it); with an odd magnet count the Outside diameter control and its link are not drawn and the Magnets per ring link and the torque card's are; the link under each copy of the end-effect banner (over the dashboard and the centre) opens the end-effect note, and without the banner there is none; every dashboard row opened with Explain on shows its note, the closest note, an assumption or its own reason, never `NO_NOTE` (`explain_never_shows_no_teaching_note_on_a_dashboard_row`); the closest-note link, the one copy of the title on its line left of the dashboard, opens that note (`click_the_closest_note_link`, shared with the gearbox row's Explain test); with Explain on, a value with nothing upstream to read draws `NO_NOTE` and `START_HERE_HINT` once each, and Explain off draws neither; BL-059: the gearbox row's own Why link, the copy of the torque-band title under the row's label (the hot-low, hot-minimum and cold-high rows draw it first), opens that note and changes no input; BL-064: the shaft-joint row's own Why link, the one copy of the shaft-joint note's title on the dashboard, under the row's label, opens that note and changes no input (`the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note`, sharing `row_s_why_links` and `click_opens_note` with the gearbox row's test); with Explain on the gearbox row, which has no equation or note of its own, offers its Why link's note as the closest note, and the link beside the lead opens it; at the defaults the failing filter lists five red checks and one amber (the daily-cycle screen at the entered bondline, BL-063), the gearbox check's row once with a red badge (`table_badges_on`, shared with the end-effect test); slice 3 (T5): the Size tab draws the torque card in place of the label under the readouts, its heading and each of its lines once, one under the next, a click on the heading closing and opening it as one on its arrow does (`the_size_tab_draws_the_torque_card_in_place_of_the_label`); a typed Required value reaches the card in the frame that applies it (`a_required_edit_reaches_the_card_in_its_own_frame`); under Magnets per ring two lines, the ring-size line with Auto off or on the magnet length and the fit line, the card drawing the Auto line of each Auto state, the count's steps (at 4 magnets, and at 40 where 40 blocks fit, the side past the slider reading "outside the slider's range") and following each edit; idle frames compute the card once, a gap edit, a clamp-pause toggle and a new Minimum OD once each (the drawn Outside diameter row's -1 mm side then reading "outside the slider's range"), and turning Auto off at a solved length or Auto on the count at its best computes nothing, while the design's results the cache is handed are computed once per design, none for idle frames, the pause or the Minimum OD (`idle_frames_compute_the_card_once_and_an_edit_again`); a drag of the gap slider with Auto on the magnet length computes the card, and its design's results, once for each design it draws (`a_drag_computes_the_card_once_for_each_design_it_draws`); idle frames change no input with the card open and closed, at the clamp pause; slice 3 (T6, headless, each test pausing the clamp section before the card's first frame, but for the Equation panel's check on the screen, which draws every view unpaused): at 1280 x 800 the card sits under the Pull-out readout and above the drawing, open, every line drawn and the drawing at least 160 points, at the defaults and the smoke design; at 930 x 800 it starts closed (the region under 400 points), at short magnets too, closed no taller than the label it replaced, and its Why link wraps under the heading inside the region and opens the torque note; at 1080 x 745 no back iron starts closed (its open card would take the drawing under its floor) and short magnets open; each rule opens it at its boundary, a region exactly 400 points wide or exactly as tall as the open card, the item spacing and the floor, while 399.99 points or 0.01 point less, or the Equation panel open, closes it (`the_card_s_default_opens_at_each_rule_s_boundary`); with no user choice a window resize decides it again (open at 1280 x 800, closed again at 930 x 800), a change of its height alone too (at 1080 points wide open at 800 points tall, closed at 700, open again at 800); the Equation panel closes it and its closing opens it, and no view, the card open or closed, paints into the panel, on the screen or inside an `egui::Window`; a click on the heading or the arrow wins for the session, through a design edit, a resize, the Equation panel (the drawing keeping its 160 points beside it), Reset all and a load; with no user choice Reset all and a load keep the default decided for the view (at 1080 x 800 no back iron's closed card stays closed through Reset all and a load of 8 magnets, which would open it, and the defaults' open card stays open through a load of no back iron) (`reset_all_and_a_load_keep_the_card_s_default_with_no_user_choice`); drags of the gap, the Outside diameter across the fit edge and Required across hot-low (checked on every drag frame), the assumptions banner, and a side panel narrowing the region at no back iron leave it as decided; opened at 930 x 800, and at 1280 x 800 at a design file's calibration coefficient of 1e300, whose 887-character lines rule 3 starts closed, its body scrolls (the mouse wheel brings its last row into view) and the drawing keeps 160 points; its measured height is its drawn height within a point at 1280 x 800 and 930 x 1024, and at 1280 x 1024 at that calibration coefficient, and at 930 x 800 its body is the cap (`the_card_s_measured_height_is_the_drawn_height`); the look: one line in the accent, as wide as the active tab and just under its name, which follows a click to another tab, the active tab's name in the strong text colour and another's in the quiet one (a theme's weak colour; the text colour in egui's own visuals, which set none), and one accent square of 8 points before the title's first glyph, at the title's height (`the_active_tab_is_underlined_in_the_accent_and_no_other`, `the_brand_mark_is_one_accent_square_before_the_title`); the changed dot and a trace's banner and frames are the host's link colour, not the colour it gives a selected choice's name (`the_panel_s_marks_are_the_host_s_link_colour`); a readout's equation wider than the window is set smaller so its tooltip stays inside the window, and one that fits keeps the tooltip's size (`src/gui/readouts.rs`, `an_equation_wider_than_the_window_is_set_smaller_and_another_keeps_its_size`); BL-097: on the app's starting design, Auto on until solved and Auto off, Tab from the top of the page through the Size column changes no input, no result and no Minimum OD, makes no undo step and selects nothing (`tab_through_a_length_auto_left_off_its_step_changes_nothing`, with the `app` feature); with a face gap of 1.4123 mm and a requirement past its slider's range loaded, Tab past every value box of the Size column and of the open Detailed inputs does the same (`tab_past_every_value_box_keeps_a_value_off_its_step_or_outside_its_range`); Enter on a value box with nothing typed, or with the number it shows typed again, keeps the 1.4123, while a typed value edits on Tab (one undo step, the row selected), on Enter and on a click elsewhere, and an arrow key steps it onto the step (`a_typed_value_and_an_arrow_key_edit_a_value_off_its_step_and_its_own_text_does_not`); the rule itself (`src/gui/input_ui.rs`, `a_value_box_s_own_text_is_no_typed_number`); a number typed into a tiny value's box edits, one undo step, clamped to the range as egui clamps it: the vacuum permeability, the NdFeB expansion and the cap's field integral (`a_number_typed_into_a_tiny_value_s_box_edits`); BL-100: at 1280 x 800 and 1024 x 700 with the Detailed inputs open, after each press of Tab and of Shift+Tab every focused widget of the left column is whole in its view once the scroll has settled, scrolled no further than it was cut plus egui's gap at the edge; nothing is selected or edited; a wheel then takes the focused row out of view and it stays out; an idle frame asks for no repaint; a Down arrow key's move is followed too (`the_left_column_follows_the_keyboard_s_focus_and_fights_no_wheel`); the same walk through a check's entry in the Explain panel at 1024 x 700 (`the_explain_panel_follows_the_keyboard_s_focus`) and through the Geometry tab's callout list (under the open Equation panel) and the Clamp tab (`the_centre_s_scrolling_views_follow_the_keyboard_s_focus`); each fails with its area's call of `follow_focus` taken out; the left column wheeled 300 points down stays where the wheel left it while Tab, then Shift+Tab, pass the top bar, the centre and the right column (`a_wheeled_column_holds_still_while_the_focus_is_on_no_widget_of_its_own`; it fails with `follow_focus`'s clip check taken out); BL-098: at 1024 x 700 and 1280 x 800 every shown note in turn, then the note list, a check's entry, a tile's and the orientation, leaves the right column 300 points wide (egui's `PanelState`) with its right edge where it was (`the_right_column_keeps_its_width_whichever_note_is_shown`); so does the check list's material warning on a design with no back iron and on one whose sleeve conducts better than 316L, and its Why link opens its note (`the_right_column_keeps_its_width_with_a_material_warning_shown`); in the app's theme and font, on the starting design and on the failing example design with the spreadsheet details on, so does every entry the Explain panel can show: the orientation, the note list, every note, check, tile, Size column row and input, and every result of the results table (`the_right_column_keeps_its_width_whatever_entry_is_shown_in_the_app_s_own_font`, with the `app` feature); a diagram in a place 284 or 150 points wide is the picture scaled to that width with every shape and label inside it, and is 300 x 130 where it has room (`src/gui/diagrams.rs`, `a_diagram_takes_the_width_it_is_given_scaled_as_one_picture`); the five plots share one Plots tab (BL-104, decision C-9): the tab row is Size and torque, Geometry, Plots, Clamp and Results table, on one row at 1024 x 700, the picker under the Plots tab names the five plots in their order and marks the one shown, a click on a name shows its plot, the tab keeps the plot shown on a click and reopens the plot picked last from another view, picking edits nothing, selects nothing and is no undo step, a picked plot is drawn in the frame of its click, and the tab's word passes the plain-words guard (`the_five_plots_share_one_tab_whose_picker_shows_each_plot`); by the keyboard, Tab reaches the Plots tab and walks the picker's five names in their order with no view changed, Enter on the tab opens the plot shown last, and Space or Enter on a focused name shows its plot (`the_keyboard_reaches_the_plots_tab_and_picks_a_plot_by_space_or_enter`) |
| `src/gui/geometry.rs`, `src/gui/geometry_view.rs` (feature `gui`) | The end view draws every part from the results (the pocket's corners at the pocket corner radius, the rings at the retainers' diameters, the key, the shaft joints' 3 mm, from where it crosses the bore), faceted blocks on their flats and arcs as sectors; each end-view callout's line measures the gap it names (the corner gap along face 1's normal, from the dashed circle the inner corners sweep to the outer flat); red below zero and for a clearance below its target; the side view spans the housing in effect (a 50.8 mm override: the 53.6 mm cavity, the hub centred on it, the liner and sleeve the retainer span in effect); the space-claim callouts name each overshoot, in red, with the exceeded axes' dashed lines red (every axis at once too), exactly at the claim inside and the next number below over, and a NaN claim amber; a claim far past the pieces (1e300, 1e39, -1e39) is left off with one note, drawn at `CLAIM_REACH` times the extent and not past it; the design checks show while each inconsistency holds (equality is not below); the wall-rule line unless there is no back iron (the cup wall input named by its label), asking for the wall the cup wall check compares with, so the first slider step that passes is not asked for the verdict's rounded-up suggestion; that wall rounded up at its last digit, so a wall shown equal to it passes (12 poles, a 1.23 T design flux density and a 1.95 mm wall fail the check, and the line asks for 1.951 mm); a minimum length rounds up at its last shown digit (past a power of ten too, and a length already on a digit stays), and one not finite or not positive shows as it is; pole counts from a file draw at most 200 blocks; two poles draw their four blocks in a round pocket and hub (a 2-gon's corners would fly off), three a triangular pocket; a NaN draws nothing wrong and says so (the parts dropped counted per view). Painted: both views at one scale, the largest that fits (`side_by_side`: none without room or for an extent that is not finite); the cup OD, the shaft and the face gap at their pixel lengths, the cup depth and the cap as painted rectangles; the callout texts in their colours (the default clearance red, its line too); an overshoot red; hovering a dimension shows its result's hover text; each dimension tag sits on its own plate of the drawing's background; a NaN view and a tiny screen draw; a region of no size paints no view; a huge claim keeps the pieces to scale with its note; in a short region (400 points down to none) the view ends at its foot, the drawing keeping 160 points while the list keeps three rows, then shrinking to scale (`drawing_height`); in a region of exactly `drawing_floor` (slice 3: the room the torque card leaves, pinned at 212.125 points in egui's default style, 160 over three 17.375-point list rows) the drawing keeps its 160 points, and a point less it shrinks below them; a dimension's hover and a callout's in the list under the drawing name their result's path only with the spreadsheet details; the design checks name their inputs by their labels and pass the plain-words guard; the key in both states of the shaft-clamp pause, at the shaft joints' key width (BL-064: each end of its slider; one typed wider than the bore from the axis; a width or a keyway depth that is not a number not drawn and counted; none at a keyway depth of 0 or less, nor at a key width of 0 or less), never at the paused clamp's, and the clamp-boss design check only while the shaft-clamp section is not paused |
```

Replace:

```markdown
| `src/gui/geometry.rs`, `src/gui/geometry_view.rs` (feature `gui`) | The end view draws every part from the results (the pocket's corners at the pocket corner radius, the rings at the retainers' diameters, the key, the shaft joints' 3 mm, from where it crosses the bore), faceted blocks on their flats and arcs as sectors; each end-view callout's line measures the gap it names (the corner gap along face 1's normal, from the dashed circle the inner corners sweep to the outer flat); red below zero and for a clearance below its target; the side view spans the housing in effect (a 50.8 mm override: the 53.6 mm cavity, the hub centred on it, the liner and sleeve the retainer span in effect); the space-claim callouts name each overshoot, in red, with the exceeded axes' dashed lines red (every axis at once too), exactly at the claim inside and the next number below over, and a NaN claim amber; a claim far past the pieces (1e300, 1e39, -1e39) is left off with one note, drawn at `CLAIM_REACH` times the extent and not past it; the design checks show while each inconsistency holds (equality is not below); the wall-rule line unless there is no back iron (the cup wall input named by its label), asking for the wall the cup wall check compares with, so the first slider step that passes is not asked for the verdict's rounded-up suggestion; that wall rounded up at its last digit, so a wall shown equal to it passes (12 poles, a 1.23 T design flux density and a 1.95 mm wall fail the check, and the line asks for 1.951 mm); a minimum length rounds up at its last shown digit (past a power of ten too, and a length already on a digit stays), and one not finite or not positive shows as it is; pole counts from a file draw at most 200 blocks; two poles draw their four blocks in a round pocket and hub (a 2-gon's corners would fly off), three a triangular pocket; a NaN draws nothing wrong and says so (the parts dropped counted per view). Painted: both views at one scale, the largest that fits (`side_by_side`: none without room or for an extent that is not finite); the cup OD, the shaft and the face gap at their pixel lengths, the cup depth and the cap as painted rectangles; the callout texts in their colours (the default clearance red, its line too); an overshoot red; hovering a dimension shows its result's hover text; each dimension tag sits on its own plate of the drawing's background; a NaN view and a tiny screen draw; a region of no size paints no view; a huge claim keeps the pieces to scale with its note; in a short region (400 points down to none) the view ends at its foot, the drawing keeping 160 points while the list keeps three rows, then shrinking to scale (`drawing_height`); in a region of exactly `drawing_floor` (slice 3: the room the torque card leaves, pinned at 212.125 points in egui's default style, 160 over three 17.375-point list rows) the drawing keeps its 160 points, and a point less it shrinks below them; a dimension's hover and a callout's in the list under the drawing name their result's path only with the spreadsheet details; the design checks name their inputs by their labels and pass the plain-words guard; the key in both states of the shaft-clamp pause, at the shaft joints' key width (BL-064: each end of its slider; one typed wider than the bore from the axis; a width or a keyway depth that is not a number not drawn and counted; none at a keyway depth of 0 or less, nor at a key width of 0 or less), never at the paused clamp's, and the clamp-boss design check only while the shaft-clamp section is not paused |
| `src/gui/plots.rs` (feature `gui`) | The torque-temperature curve meets the engine's pull-out at the operating temperature and at 20 °C, its band edges the hot-low and cold-high torques (bit for bit), a grade ring's own alpha too; sweep rows by status and end-effect range (4, 2, 7 and 1, 1, 4 at the defaults; short magnets grey every row); slip heating rises to the steady temperatures and marks the time to the limit on the curve; the torque-rotation curve peaks at the pull-out (harmonics 1 to 11 too; an invalid set draws none); f_end <= 0 greys the torque plots; each plot paints its series with their point counts and legends (an all-greyed sweep: its markers and no line; the pole sweep's line named for its rows' smallest ring radius the inner magnets and the keyed bore allow; the legends and axes use the on-screen names (the pull-out and required torque as the dashboard and input rows, the torque-rotation curve "Torque (nominal)" since only its peak is the pull-out, the band as the production variation allowance, the pole sweep's axis Magnets per ring, the sweeps' amber status "Below the required floor"), each plain words with glyphs); values that are not numbers draw without panicking; a sweep with no finite row says there is nothing to plot, one with some counts them in a note; an empty temperature axis (a minimum temperature at or past its end) is named as such, apart from values that are not numbers; in a short region each plot ends at its foot, and a region a few points tall or none draws; a readout's hover names its path only with the spreadsheet details; the axis labels take a theme's weak text colour and the readouts over the plot keep the text colour, and in egui's own visuals, which set no weak colour, the axis labels keep the text colour and the operating temperature's line is the host's weak colour, not that colour faded again (`the_axes_take_the_weak_text_colour_and_the_readouts_do_not`); in a region narrower than its sentence the pole sweep's line is named once, over the plot, on more than one row, after one stroke in the line's colour and width, with no text cut, and the gap sweep's short name stays in the legend on one row (`the_pole_sweep_s_line_is_named_over_the_plot_on_as_many_rows_as_it_needs`); a readout longer than a row starts a row at the readouts' left edge and wraps, with no text cut (`a_readout_longer_than_a_row_wraps_on_rows_of_its_own`) |
| `src/gui/clamp_drawing.rs` (feature `gui`) | `fmt_g` matches Python's `:g`; clipping keeps what is inside the boss; the default clamp has drawing.py's texts, the boss and bore circles, every cut inside the boss and one screw axis; no fitting screw (and an index outside the table) gives drawing.py's message; drawing.py's screws needed are drawn whatever fits, at most 50; the screw table's rows follow the sheet; the tab paints both views at one scale, the summary, the machining steps and the screw table, drawing.py's message, the one-piece note, and a narrow or tiny region without panicking, and no room with no scale; at 540, 660 and 980 points wide every note and dimension text stays inside the drawing and clear of the view titles (the flange note moved back from the right edge); in a short region (300 points down to none) the view scrolls in exactly that space; a summary line's hover and a screw table cell's name their path only with the spreadsheet details |
```

with:

```markdown
| `src/gui/geometry.rs`, `src/gui/geometry_view.rs` (feature `gui`) | The end view draws every part from the results (the pocket's corners at the pocket corner radius, the rings at the retainers' diameters, the key, the shaft joints' 3 mm, from where it crosses the bore), faceted blocks on their flats and arcs as sectors; each end-view callout's line measures the gap it names (the corner gap along face 1's normal, from the dashed circle the inner corners sweep to the outer flat); red below zero and for a clearance below its target; the side view spans the housing in effect (a 50.8 mm override: the 53.6 mm cavity, the hub centred on it, the liner and sleeve the retainer span in effect); the space-claim callouts name each overshoot, in red, with the exceeded axes' dashed lines red (every axis at once too), exactly at the claim inside and the next number below over, and a NaN claim amber; a claim far past the pieces (1e300, 1e39, -1e39) is left off with one note, drawn at `CLAIM_REACH` times the extent and not past it; the design checks show while each inconsistency holds (equality is not below); the wall-rule line unless there is no back iron (the cup wall input named by its label), asking for the wall the cup wall check compares with, so the first slider step that passes is not asked for the verdict's rounded-up suggestion; that wall rounded up at its last digit, so a wall shown equal to it passes (12 poles, a 1.23 T design flux density and a 1.95 mm wall fail the check, and the line asks for 1.951 mm); a minimum length rounds up at its last shown digit (past a power of ten too, and a length already on a digit stays), and one not finite or not positive shows as it is; pole counts from a file draw at most 200 blocks; two poles draw their four blocks in a round pocket and hub (a 2-gon's corners would fly off), three a triangular pocket; a NaN draws nothing wrong and says so (the parts dropped counted per view). Painted: both views at one scale, the largest that fits (`side_by_side`: none without room or for an extent that is not finite); the cup OD, the shaft and the face gap at their pixel lengths, the cup depth and the cap as painted rectangles; the callout texts in their colours (the default clearance red, its line too); an overshoot red; hovering a dimension shows its result's hover text; each dimension tag sits on its own plate of the drawing's background; a NaN view and a tiny screen draw; a region of no size paints no view; a huge claim keeps the pieces to scale with its note; in a short region (400 points down to none) the view ends at its foot, the drawing keeping 160 points while the list keeps three rows, then shrinking to scale (`drawing_height`); in a region of exactly `drawing_floor` (slice 3: the room the torque card leaves, pinned at 212.125 points in egui's default style, 160 over three 17.375-point list rows) the drawing keeps its 160 points, and a point less it shrinks below them; a dimension's hover and a callout's in the list under the drawing name their result's path only with the spreadsheet details; the design checks name their inputs by their labels and pass the plain-words guard; the key in both states of the shaft-clamp pause, at the shaft joints' key width (BL-064: each end of its slider; one typed wider than the bore from the axis; a width or a keyway depth that is not a number not drawn and counted; none at a keyway depth of 0 or less, nor at a key width of 0 or less), never at the paused clamp's, and the clamp-boss design check only while the shaft-clamp section is not paused |
| `src/gui/plots.rs` (feature `gui`) | The torque-temperature curve meets the engine's pull-out at the operating temperature and at 20 °C, its band edges the hot-low and cold-high torques (bit for bit), a grade ring's own alpha too; sweep rows by status and end-effect range (4, 2, 7 and 1, 1, 4 at the defaults; short magnets grey every row); slip heating rises to the steady temperatures and marks the time to the limit on the curve; the torque-rotation curve peaks at the pull-out (harmonics 1 to 11 too; an invalid set draws none); f_end <= 0 greys the torque plots; each plot paints its series with their point counts and legends (an all-greyed sweep: its markers and no line; the pole sweep's line named for its rows' smallest ring radius the inner magnets and the keyed bore allow; the legends and axes use the on-screen names (the pull-out and required torque as the dashboard and input rows, the torque-rotation curve "Torque (nominal)" since only its peak is the pull-out, the band as the production variation allowance, the pole sweep's axis Magnets per ring, the sweeps' amber status "Below the required floor"), each plain words with glyphs); values that are not numbers draw without panicking; a sweep with no finite row says there is nothing to plot, one with some counts them in a note; an empty temperature axis (a minimum temperature at or past its end) is named as such, apart from values that are not numbers; in a short region each plot ends at its foot, and a region a few points tall or none draws; a readout's hover names its path only with the spreadsheet details; the axis labels take a theme's weak text colour and the readouts over the plot keep the text colour, and in egui's own visuals, which set no weak colour, the axis labels keep the text colour and the operating temperature's line is the host's weak colour, not that colour faded again (`the_axes_take_the_weak_text_colour_and_the_readouts_do_not`); in a region narrower than its sentence the pole sweep's line is named once, over the plot, on more than one row, after one stroke in the line's colour and width, with no text cut, and the gap sweep's short name stays in the legend on one row (`the_pole_sweep_s_line_is_named_over_the_plot_on_as_many_rows_as_it_needs`); a readout longer than a row starts a row at the readouts' left edge and wraps, with no text cut (`a_readout_longer_than_a_row_wraps_on_rows_of_its_own`); in every plot, on a passing design, the failing example design, a design whose high case reaches the limit, a hot day and a greyed design, every red mark is a sweep row that does not fit or its legend entry, the requirement lines are in the strong text colour and slip heating's limit is violet (`red_marks_a_sweep_row_that_does_not_fit_and_nothing_else_in_any_plot`, BL-104); every tick number of every plot, at 25 window sizes, in egui's visuals and in a theme's, is drawn in the axes' colour or not at all (`every_tick_number_is_drawn_at_full_strength_or_not_at_all`, BL-104) |
| `src/gui/clamp_drawing.rs` (feature `gui`) | `fmt_g` matches Python's `:g`; clipping keeps what is inside the boss; the default clamp has drawing.py's texts, the boss and bore circles, every cut inside the boss and one screw axis; no fitting screw (and an index outside the table) gives drawing.py's message; drawing.py's screws needed are drawn whatever fits, at most 50; the screw table's rows follow the sheet; the tab paints both views at one scale, the summary, the machining steps and the screw table, drawing.py's message, the one-piece note, and a narrow or tiny region without panicking, and no room with no scale; at 540, 660 and 980 points wide every note and dimension text stays inside the drawing and clear of the view titles (the flange note moved back from the right edge); in a short region (300 points down to none) the view scrolls in exactly that space; a summary line's hover and a screw table cell's name their path only with the spreadsheet details |
```

Replace:

```markdown
| `src/gui/plain_text.rs` (feature `gui`) | Every overlay key is an input or a result (a hint's an input or one of the Size tab's own hint keys), once, with text; no help row repeats the engine's; the nine canonical names of spec 3.5 and the gearbox ripple row's (BL-059: Nominal slip swing at the gearbox input), and no other row, the Size tab's gap and count names equal to the input rows'; `label_for` falls back to the metadata label and never to the path, and never lengthens a label; the help mechanism on a stand-in table (the overlay, else the engine; the original is the spreadsheet's own note, kept only where it is not empty: the deviation registry's workbook text over a help a correction reworded, else the engine's help) and the public accessors over every input's own engine help; every label, help and hint passes the plain-words guard and the glyph test, calls no torque guaranteed, holds no brace and keeps its caps (help <= 30 words in 1-2 sentences, hint <= 12 words in one); no display label is another input's or result's name; no overlay row, and no engine label or help shown where the overlay has none, says the coupling protects the gearbox (`no_label_help_or_hint_says_the_coupling_protects_the_gearbox`, BL-059). Every input has help from the engine or the overlay (`every_input_has_help_from_the_engine_or_the_overlay`); `overlay_strings_for_review` prints the overlay for the physics review. Help batch A (T7, `help_tests`): every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint groups has help; hints exist exactly for the Key design inputs and the three `SIZE_TAB_HINT_KEYS` (none of them an input path); batch A help and every hint carry no digit; no hint types an input label (catalogue or display); the review sheet lists every input of its groups and every hint; `help_for_review` (ignored) writes the G-3 sheet (`HELP_REVIEW_GROUPS`, `HELP_REVIEW_OUT`). Help batch B (T8, `batch_b_tests`): the 52 empty helps of the Calibration and model, Materials, Housing and retainers, and Thermal and demagnetization groups each have a row (the set is computed from `WORKFLOW`), and 31 non-empty helps are replaced (the grade-mode density, rating and coefficient helps, the stored 3D fields, placeholder and assumed values, and helps with a cell or correction id or a stale fact), none of them Rust-only; the toggle's original note is the spreadsheet's own text (the deviation registry's for a help a correction reworded, none where that is empty); 175 direction claims hold, each a result that rises, falls or stays when a batch-B input rises one slider step on a named design (the example, no back iron, a 1018 back iron, a 304 back iron, a titanium sleeve, a 7075 cap, screw classes 10.9 and A4-70, the coercivity inputs, a measured drag, manual magnets with and without Y30 grades, library N42 parts in both rings), and so do the outside-diameter (the boss or the cap sets it only as the largest part), cap-resize, cup-wall, measured-drag, magnet-mass and temperature-check claims; a help that names the shaft joint its input feeds has its claim among them, the way its words say (`batch_b_helps_name_the_shaft_joint_they_feed`: the boss length lets the cup's fit hold more, and the single-ring reverse field can lower the inner hub's allowed fitting temperature, which falls with library N42 parts in both rings, where the onset binds, and stays at the defaults); each stored-3D, prototype, unmeasured-default and adapter cluster is exactly the help rows that end with its caveat (each caveat is typed once, as a macro the rows are built from); the stored-3D cluster is the dashboard's `STORED_3D_INPUTS` less the Rust-only ones, and its caveat is the engine's sentence (`temperature::stored_3d_sentence!`), the one literal `STORED_3D_SENTENCE` and the engine's free-space helps read too; no help repeats its input's default; no overlay row covers a Rust-only input, whose help the engine words (R1).; the joint-neutral texts (ruling 11) claim what the engine does: a stronger manual block and longer magnets raise the pull-out, the cold-high torque and its gearbox-input value (longer magnets the overall length too), and a larger bore, both shafts' diameter (both joints' holding torques move with it), thins the hub wall under the magnets and past the keyway: its help opens with both shafts and names the hub check (by its label), which the bore's slider top fails on the 8-magnet probe design, keyed or keyless, with the wall past the keyway still standing, and that wall breaks through at the slider top on the smallest ring; the keyway's help names both bores and says it caps the key's contact height: one step deeper thins the hub wall past the key and, with the contact height under the keyway, leaves both joints' key pressures bit for bit, while on a keyway shallower than the contact height (1.0 mm, the "shallow keyway" design) it lowers both; the gearbox texts claim what the engine does (BL-059, `the_gearbox_texts_claim_what_the_engine_does`): the rating's help names the gearbox check and the cold-high torque at the gearbox input, the ripple row's label and help the slip swing, the pull-out at the gearbox output at the operating temperature without the production allowance, through the ratio and efficiency, the check row's help (the engine's) both sides of the gearbox and that fatigue from repeated slips is not checked, the gear ratio's help the slip swing as the ripple row does (one name on screen), and no text says peak, warm or no check (the rating's type is open; the operating temperature can sit below the minimum); at the five probe designs the swing is the nominal pull-out divided by the ratio and efficiency, and moves when the operating temperature rises one slider step but not when the minimum temperature or the variation does; and with the rating halfway between the swing and cold-high at the gearbox input the check follows cold-high, above the swing at the defaults and below it with the operating temperature under the minimum (-40 against 20 °C, variation 0.1) |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the calculator's theme and opens the starting design (its status line, no changed dot, the example design's button, the requirement and its margin on the readouts); the starting design passes every counted check with no caution and no material warning, pinned to `compute_all` with the numbers the README quotes, changes seven inputs and its fourteen stored 3D fields on their slider steps and no assumption, stores the 3D fields of a recorded run of the reference routine for its geometry, and draws no caution on its opening tab; a link and a design file saved from the first starting design (main 2ca65ee) open that design as it was; the example design opens as one undo step and Reset all returns the starting design; the spreadsheet export still flags against the workbook (the seven inputs and fourteen 3D fields at the starting design, none at the example design); the page address is clean at the starting design and the share link elsewhere, follows settled edits, undo and redo and waits for an edit in progress; a Ctrl+Z or Ctrl+Y held for 60 frames writes the address once, at the key's release, and a write the page refuses is asked again on the next frame (the tests' page records each write and can refuse one); a link opened at start-up stays the address until the first edit, and one that cannot be read changes nothing; a link cut at 2,000 characters is refused, changes nothing and the header says why; a link stays under 4,000 characters; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link, the smoke prompt names the header's Show spreadsheet details toggle (`SHOW_DETAILS`), and its R14 steps name `WELCOME`, `explorer::WHAT_TO_CHANGE`, `explorer::WHERE_FROM`, the Geometry tab and the Magnets fit row as the page draws them, and its card step quotes `torque_card::CARD_HEADING` (the script's own `CARD_HEADING`, typed once for the first view and the card step), `CAPTION`, `AUTO_HOLDS_LENGTH` and `held_line(Control::OverallLength)` whole and names the other three rows by their starts (slice 3, test 26); the frame timer `FRAME_TIMER` is one line that holds the WebGL draw-call test (`drawElements`, `if (window.__drew)`), and the card step installs it with `page.addInitScript` (plan review 4.1); the drag step finds the gap slider by the value its box shows, the script's `SMOKE_GAP_SHOWN` (the link's 1.5 mm at the slider's step's two decimals, with its unit, typed once), and counts the sizing lines by `SIZING_LOG_PREFIX` in its code and its words; each page's console messages are collected after its last step, since the Playwright MCP returns only those since the last navigation; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the tokens are the UX overhaul spec's table value for value, the visuals are that table field for field and egui's dark visuals in every other field, and every text token is at least 4.5:1 on every surface it is drawn on, with the spec's own figures to one decimal (`the_tokens_are_the_spec_s_table`, `the_visuals_are_the_token_table`, `every_text_token_reads_on_every_surface_it_is_drawn_on`); the selection fill is at least 3:1 against a field and the panel and 2.8:1 on a tooltip, the text colour on it is at least 4.5:1, no lighter fill could give a tooltip 3:1 and keep that, it is a blue, and no other text colour reads on it (`selected_text_shows_on_every_surface_and_reads_on_its_fill`); the app's font: Inter leads the proportional family and the built-in fonts follow, the ten digits have one advance width at seven sizes and are the cut's own, the cut holds printable ASCII and the other characters the app's texts use and not a character Inter lacks, a row of the installed font is as tall as the built-in font's at eight sizes, to egui's rounding (`a_row_is_as_tall_as_the_built_in_font_s`), a Cyrillic letter and an emoji, both outside the cut, have a glyph with the font installed and none with the cut alone (`a_character_outside_the_cut_is_drawn_by_a_built_in_font`), and the app installs it while a panel in a plain context keeps egui's fonts (`src/app/fonts.rs`; `the_app_draws_its_text_in_inter`); on the app's page, at the starting design and at the failing example design, the accent is on two shapes (the brand mark and the active tab's line), on no text and on no badge (`the_accent_marks_the_brand_and_the_active_tab_and_nothing_else`); on the app's page a dashboard row's title, the heads of the Why, Problem and Fix lines and the Size tab's readout labels are drawn in the muted token and their values in the text token, at the starting design and at the failing example design, and a tab that is not the active one is muted while the active tab's name is the text token (`a_label_is_muted_and_its_value_is_in_the_text_colour`); in the app's own theme and font, at 1280 x 800 and at 1024 x 700, no text of any tab is cut sideways by the rect egui clips it to or by the window, a plot's rotated axis label included (`no_text_is_cut_in_any_tab_in_the_app_s_own_font`: the panel's layout tests draw in egui's built-in font, which is about 13 % narrower, so only this test sees a text that no longer fits), and the widest tooltip, the Verdict row's, stays inside the window (`the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font`); BL-099: at 1024 x 700, 1280 x 800 and 1920 x 1080 the left column is 320 points wide (egui's `PanelState`) and no text is cut, on any tab or on any row of the column, with the folds shut, each fold opened and shut again, every input group and its Advanced heading opened and shut again, a filter that lists every input set and cleared, an input traced, and Auto on with the Detailed inputs open (`the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font`); every input row, as it is, changed, changed and outside its slider's range, and locked by Auto, takes no more than the width it is given, at the three widths the column gives a row, in egui's default font and in the app's (`every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font`); every input row whose label wraps at the narrowest of those widths, changed, has as many Tab stops as at a width where nothing wraps, and its reset button sits close under the label, in either font (`an_input_label_that_wraps_is_one_tab_stop_in_either_font`); BL-098: at 1280 x 800 and 1024 x 700, with a note that has a diagram shown through Learn and its title, the right column is 300 points wide and no text is cut on any tab (`the_right_column_keeps_its_width_with_a_note_shown_in_the_app_s_own_font`); the hub page's card for the calculator does not name the paused clamp section (`the_hub_page_does_not_name_the_paused_clamp_section`); the smoke script's `PAGE_BACKGROUND`, typed once, is the theme's panel colour as a browser reports it, and its clean-page step judges the look, reads the page's background and names the title and the active tab; the theme stays dark when the system turns light |

```

with:

```markdown
| `src/gui/plain_text.rs` (feature `gui`) | Every overlay key is an input or a result (a hint's an input or one of the Size tab's own hint keys), once, with text; no help row repeats the engine's; the nine canonical names of spec 3.5 and the gearbox ripple row's (BL-059: Nominal slip swing at the gearbox input), and no other row, the Size tab's gap and count names equal to the input rows'; `label_for` falls back to the metadata label and never to the path, and never lengthens a label; the help mechanism on a stand-in table (the overlay, else the engine; the original is the spreadsheet's own note, kept only where it is not empty: the deviation registry's workbook text over a help a correction reworded, else the engine's help) and the public accessors over every input's own engine help; every label, help and hint passes the plain-words guard and the glyph test, calls no torque guaranteed, holds no brace and keeps its caps (help <= 30 words in 1-2 sentences, hint <= 12 words in one); no display label is another input's or result's name; no overlay row, and no engine label or help shown where the overlay has none, says the coupling protects the gearbox (`no_label_help_or_hint_says_the_coupling_protects_the_gearbox`, BL-059). Every input has help from the engine or the overlay (`every_input_has_help_from_the_engine_or_the_overlay`); `overlay_strings_for_review` prints the overlay for the physics review. Help batch A (T7, `help_tests`): every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint groups has help; hints exist exactly for the Key design inputs and the three `SIZE_TAB_HINT_KEYS` (none of them an input path); batch A help and every hint carry no digit; no hint types an input label (catalogue or display); the review sheet lists every input of its groups and every hint; `help_for_review` (ignored) writes the G-3 sheet (`HELP_REVIEW_GROUPS`, `HELP_REVIEW_OUT`). Help batch B (T8, `batch_b_tests`): the 52 empty helps of the Calibration and model, Materials, Housing and retainers, and Thermal and demagnetization groups each have a row (the set is computed from `WORKFLOW`), and 31 non-empty helps are replaced (the grade-mode density, rating and coefficient helps, the stored 3D fields, placeholder and assumed values, and helps with a cell or correction id or a stale fact), none of them Rust-only; the toggle's original note is the spreadsheet's own text (the deviation registry's for a help a correction reworded, none where that is empty); 175 direction claims hold, each a result that rises, falls or stays when a batch-B input rises one slider step on a named design (the example, no back iron, a 1018 back iron, a 304 back iron, a titanium sleeve, a 7075 cap, screw classes 10.9 and A4-70, the coercivity inputs, a measured drag, manual magnets with and without Y30 grades, library N42 parts in both rings), and so do the outside-diameter (the boss or the cap sets it only as the largest part), cap-resize, cup-wall, measured-drag, magnet-mass and temperature-check claims; a help that names the shaft joint its input feeds has its claim among them, the way its words say (`batch_b_helps_name_the_shaft_joint_they_feed`: the boss length lets the cup's fit hold more, and the single-ring reverse field can lower the inner hub's allowed fitting temperature, which falls with library N42 parts in both rings, where the onset binds, and stays at the defaults); each stored-3D, prototype, unmeasured-default and adapter cluster is exactly the help rows that end with its caveat (each caveat is typed once, as a macro the rows are built from); the stored-3D cluster is the dashboard's `STORED_3D_INPUTS` less the Rust-only ones, and its caveat is the engine's sentence (`temperature::stored_3d_sentence!`), the one literal `STORED_3D_SENTENCE` and the engine's free-space helps read too; no help repeats its input's default; no overlay row covers a Rust-only input, whose help the engine words (R1).; the joint-neutral texts (ruling 11) claim what the engine does: a stronger manual block and longer magnets raise the pull-out, the cold-high torque and its gearbox-input value (longer magnets the overall length too), and a larger bore, both shafts' diameter (both joints' holding torques move with it), thins the hub wall under the magnets and past the keyway: its help opens with both shafts and names the hub check (by its label), which the bore's slider top fails on the 8-magnet probe design, keyed or keyless, with the wall past the keyway still standing, and that wall breaks through at the slider top on the smallest ring; the keyway's help names both bores and says it caps the key's contact height: one step deeper thins the hub wall past the key and, with the contact height under the keyway, leaves both joints' key pressures bit for bit, while on a keyway shallower than the contact height (1.0 mm, the "shallow keyway" design) it lowers both; the gearbox texts claim what the engine does (BL-059, `the_gearbox_texts_claim_what_the_engine_does`): the rating's help names the gearbox check and the cold-high torque at the gearbox input, the ripple row's label and help the slip swing, the pull-out at the gearbox output at the operating temperature without the production allowance, through the ratio and efficiency, the check row's help (the engine's) both sides of the gearbox and that fatigue from repeated slips is not checked, the gear ratio's help the slip swing as the ripple row does (one name on screen), and no text says peak, warm or no check (the rating's type is open; the operating temperature can sit below the minimum); at the five probe designs the swing is the nominal pull-out divided by the ratio and efficiency, and moves when the operating temperature rises one slider step but not when the minimum temperature or the variation does; and with the rating halfway between the swing and cold-high at the gearbox input the check follows cold-high, above the swing at the defaults and below it with the operating temperature under the minimum (-40 against 20 °C, variation 0.1) |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the calculator's theme and opens the starting design (its status line, no changed dot, the example design's button, the requirement and its margin on the readouts); the starting design passes every counted check with no caution and no material warning, pinned to `compute_all` with the numbers the README quotes, changes seven inputs and its fourteen stored 3D fields on their slider steps and no assumption, stores the 3D fields of a recorded run of the reference routine for its geometry, and draws no caution on its opening tab; a link and a design file saved from the first starting design (main 2ca65ee) open that design as it was; the example design opens as one undo step and Reset all returns the starting design; the spreadsheet export still flags against the workbook (the seven inputs and fourteen 3D fields at the starting design, none at the example design); the page address is clean at the starting design and the share link elsewhere, follows settled edits, undo and redo and waits for an edit in progress; a Ctrl+Z or Ctrl+Y held for 60 frames writes the address once, at the key's release, and a write the page refuses is asked again on the next frame (the tests' page records each write and can refuse one); a link opened at start-up stays the address until the first edit, and one that cannot be read changes nothing; a link cut at 2,000 characters is refused, changes nothing and the header says why; a link stays under 4,000 characters; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link, the smoke prompt names the header's Show spreadsheet details toggle (`SHOW_DETAILS`), and its R14 steps name `WELCOME`, `explorer::WHAT_TO_CHANGE`, `explorer::WHERE_FROM`, the Geometry tab and the Magnets fit row as the page draws them, and its card step quotes `torque_card::CARD_HEADING` (the script's own `CARD_HEADING`, typed once for the first view and the card step), `CAPTION`, `AUTO_HOLDS_LENGTH` and `held_line(Control::OverallLength)` whole and names the other three rows by their starts (slice 3, test 26); the frame timer `FRAME_TIMER` is one line that holds the WebGL draw-call test (`drawElements`, `if (window.__drew)`), and the card step installs it with `page.addInitScript` (plan review 4.1); the drag step finds the gap slider by the value its box shows, the script's `SMOKE_GAP_SHOWN` (the link's 1.5 mm at the slider's step's two decimals, with its unit, typed once), and counts the sizing lines by `SIZING_LOG_PREFIX` in its code and its words; each page's console messages are collected after its last step, since the Playwright MCP returns only those since the last navigation; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the tokens are the UX overhaul spec's table value for value, the visuals are that table field for field and egui's dark visuals in every other field, and every text token is at least 4.5:1 on every surface it is drawn on, with the spec's own figures to one decimal (`the_tokens_are_the_spec_s_table`, `the_visuals_are_the_token_table`, `every_text_token_reads_on_every_surface_it_is_drawn_on`); the selection fill is at least 3:1 against a field and the panel and 2.8:1 on a tooltip, the text colour on it is at least 4.5:1, no lighter fill could give a tooltip 3:1 and keep that, it is a blue, and no other text colour reads on it (`selected_text_shows_on_every_surface_and_reads_on_its_fill`); the app's font: Inter leads the proportional family and the built-in fonts follow, the ten digits have one advance width at seven sizes and are the cut's own, the cut holds printable ASCII and the other characters the app's texts use and not a character Inter lacks, a row of the installed font is as tall as the built-in font's at eight sizes, to egui's rounding (`a_row_is_as_tall_as_the_built_in_font_s`), a Cyrillic letter and an emoji, both outside the cut, have a glyph with the font installed and none with the cut alone (`a_character_outside_the_cut_is_drawn_by_a_built_in_font`), and the app installs it while a panel in a plain context keeps egui's fonts (`src/app/fonts.rs`; `the_app_draws_its_text_in_inter`); on the app's page, at the starting design and at the failing example design, the accent is on two shapes (the brand mark and the active tab's line), on no text and on no badge (`the_accent_marks_the_brand_and_the_active_tab_and_nothing_else`); on the app's page a dashboard row's title, the heads of the Why, Problem and Fix lines and the Size tab's readout labels are drawn in the muted token and their values in the text token, at the starting design and at the failing example design, and a tab that is not the active one is muted while the active tab's name is the text token (`a_label_is_muted_and_its_value_is_in_the_text_colour`); in the app's own theme and font, at 1280 x 800 and at 1024 x 700, no text of any tab (each plot through the Plots tab's picker, since BL-104) is cut sideways by the rect egui clips it to or by the window, a plot's rotated axis label included (`no_text_is_cut_in_any_tab_in_the_app_s_own_font`: the panel's layout tests draw in egui's built-in font, which is about 13 % narrower, so only this test sees a text that no longer fits), and the widest tooltip, the Verdict row's, stays inside the window (`the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font`); BL-099: at 1024 x 700, 1280 x 800 and 1920 x 1080 the left column is 320 points wide (egui's `PanelState`) and no text is cut, on any tab or on any row of the column, with the folds shut, each fold opened and shut again, every input group and its Advanced heading opened and shut again, a filter that lists every input set and cleared, an input traced, and Auto on with the Detailed inputs open (`the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font`); every input row, as it is, changed, changed and outside its slider's range, and locked by Auto, takes no more than the width it is given, at the three widths the column gives a row, in egui's default font and in the app's (`every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font`); every input row whose label wraps at the narrowest of those widths, changed, has as many Tab stops as at a width where nothing wraps, and its reset button sits close under the label, in either font (`an_input_label_that_wraps_is_one_tab_stop_in_either_font`); BL-098: at 1280 x 800 and 1024 x 700, with a note that has a diagram shown through Learn and its title, the right column is 300 points wide and no text is cut on any tab (`the_right_column_keeps_its_width_with_a_note_shown_in_the_app_s_own_font`); the hub page's card for the calculator does not name the paused clamp section (`the_hub_page_does_not_name_the_paused_clamp_section`); the smoke script's `PAGE_BACKGROUND`, typed once, is the theme's panel colour as a browser reports it, and its clean-page step judges the look, reads the page's background and names the title and the active tab; the theme stays dark when the system turns light |

```

- [ ] **Step 2: Check.** `git diff --stat` lists the nine files and no other; `python -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ['docs/ai/backlog.yaml', 'docs/ai/02-system.yaml', 'docs/ai/03-structure.yaml', 'docs/ai/04-memory.yaml']]"` prints nothing; `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3` still reads `1141 passed`.

- [ ] **Step 3: Commit** the nine files: `docs(magcoupler): slice B stage 2 part 2b-iii-a docs sweep: README, docs/ai, the register, owner questions 70 to 72, the design's header, BL-104 in progress, BL-105 filed (BL-104 T5)`.

### Task 6: Release: the full gate, gui-smoke, the reviews, one fix wave, the merge and the push

**Model:** `fable` controller; fresh-context `fable` reviewers (while `fable` is out, back 2026-12-01, both on the session model, and the report says so); `haiku` at low effort for git; the smoke workflow's own agents on `sonnet`.

**Files:**
- Modify: `docs/ai/backlog.yaml` (Step 7); create `docs/analyses/2026-11-27-magcoupler-slice-b2p2biii-a-review.md` (Steps 4, 5 and 7).

- [ ] **Step 1: Preconditions.** Tasks 1 to 5 are committed, each reviewed; `git status --short` shows only the owner's untracked press analysis; `git diff main...HEAD --stat -- magcoupling-rs/src/engine magcoupling-rs/tests reference linkage-sim-rs/src linkage-sim-rs/web` is empty (no engine file, no linkage file).

- [ ] **Step 2: The full gate.** From PowerShell at the worktree's root:

  ```powershell
  & 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/gate.sh --full
  git checkout -- docs/chebyshev_lambda
  ```

  Expected: `GATE PASS`. A red gate goes back to the controller with the gate's output; nothing is merged.

- [ ] **Step 3: gui-smoke, from the branch's own script.** Build the web calculator from the branch (`linkage-sim-rs/scripts/build_magcoupling_web.sh` through Git Bash), serve `linkage-sim-rs/web/`, and run the `gui-smoke` workflow from this branch's `.claude/workflows/gui-smoke.js` with `{"frame_check": true}`. Expected: `passed: true` for the calculator with `plots_picker: true` and `same_column: true` (the Geometry step now starts from the Plots tab), and a p95 under one 60 fps frame. Keep the screenshots of the Plots tab before and after the pick.

- [ ] **Step 4: The text review** (fresh context; the design's 8.3). One new on-screen string: the tab's word "Plots", the design's own (C-9). It checks that every other text of the tab row and the picker is a text the tab row drew at `main` (the five plots' names, "Results table"), that no text left the screen, and that no engine text changed (`git diff main...HEAD -- magcoupling-rs/src/engine` is empty). It rules on three things and records what it decides: whether "Results table" reads well as the fourth tab where the design writes "Results" (decision D-89, owner question 71); whether a row of five names reads as a picker under a tab and not as a second tab row, at 1024 x 700 where it wraps (decision D-87, owner question 70); and whether, with the red gone, a requirement line in the strong text colour and a limit in violet still read as a requirement and a limit beside their legend names (decision D-90, owner question 72; render each plot on the passing design and on the example design, at 1280 x 800, in the app's theme). It rules too on the sweeps' legend: egui_plot draws every legend entry as a filled dot in its colour, so "Required floor" and "This design" now have the same dot (on `main` "Required floor" and "Does not fit" shared the red one), while on the plot the dashed line and the ring still differ; if it rules for a change, the smallest is the sweeps' "This design" ring in `BLUE`, as the torque-temperature plot draws its own. A change the owner should weigh goes to `docs/analyses/2026-11-18-owner-questions.md` as an open question with its conservative answer, not shipped silently. Its findings join Step 5's fix wave. Record it in `docs/analyses/2026-11-27-magcoupler-slice-b2p2biii-a-review.md` (committed in Step 7).

- [ ] **Step 5: The independent whole-branch review** (fresh context, three lenses in parallel, each with a skeptic on every finding that is not a suggestion). Give each `git diff main...HEAD`, the design, this plan and Step 3's screenshots, and none of the builders' notes.
  - **Does every walk over the views still reach every plot, and does every edited test keep its subject?** For each test under "Files": the assertions it had are still made, on the views it read; `Harness::tab` and `show_view` reach each plot through the picker; the focus test's drag gives the callout list rows past its foot and changes nothing else it reads; the underline's wrapped row is a real wrap.
  - **Is anything lost or moved without a test?** The tab row and the picker against the tab row at `main`: each plot reached, its readouts, legend, axes and zoom kept, the plot shown last, the keyboard's Tab through the row and the picker (`the_keyboard_reaches_the_plots_tab_and_picks_a_plot_by_space_or_enter`), the layout at 1024 x 700, 1280 x 800 and 1920 x 1080 with each plot picked (render them), the colours of every plot on a passing, a failing and a greyed design.
  - **The code.** `CentreView::tab` and the tab row's one Plots tab (a click on it on every view, the plot field's update at the top of the frame), the picker's frame of a click, `plot`'s two axes (the axis labels and placement as before), dead code or names left by the five tabs.
  One fix wave follows, on the decider's model: each accepted finding gets its red test first. A finding that is not a defect of this part is filed in `docs/ai/backlog.yaml` with the next free id.

- [ ] **Step 6: If Step 4 or 5 changed a file, run Steps 2 and 3 again.** The merge needs a gate and a smoke run of the tree that is merged.

- [ ] **Step 7: Close the item and merge.** In `docs/ai/backlog.yaml` (the day's date; say what the reviews found):

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii (Tasks 1 to 4 of the plan);
```

with:

```yaml
  status: fixed
  notes: "FIXED <date>: merged (<the gate, the reviews and the smoke run in a few words>). Built on branch magcoupling/overhaul-b2p2b-iii (Tasks 1 to 4 of the plan);
```

  Commit it with the review record: `docs(magcoupler): BL-104 fixed: the reviews, the gate and the smoke run passed`. Then merge the branch into `main` with a merge commit (`git merge --no-ff`), its message one line of what the screen now does.

- [ ] **Step 8: Push, then report** (the owner's ruling of 2026-11-22: each stage goes live once the gate, the smoke run and the reviews pass, with a report afterwards; no stop). Push `main` (it deploys the site). When the `deploy-web.yml` run is green, run the smoke workflow from `main`'s script with `{"url": "https://colesorkness.com"}` and report the result; if it fails, say so plainly: the site is live with the change. The report to the owner holds: what changed on screen, in three sentences; two pictures (the Plots tab with the picker at 1280 x 800, a sweep on the passing design); the decisions D-86 to D-94, one line each; owner questions 70 to 72; parts 2b-iii-b and 2b-iii-c with their items; which model ran the decider's steps; what was not done (BL-105, the sparse horizontal axes at 1024 x 700 and 1280 x 800, filed and open); what was not verified (Safari and Firefox; in a browser, every plot but Slip heating: the smoke picks one).

## What this plan covers of the design

| The design | Where |
|---|---|
| C-9 The five plots under one "Plots" tab with a picker; the tab row Size and torque, Geometry, Plots, Results | Task 1 (the fourth tab keeps "Results table", D-89) |
| 4.4 "Plots. A picker of the five plots (same names, same order), then the plot with its readout line, as today" | Task 1 (`the_five_plots_share_one_tab_whose_picker_shows_each_plot`; the readouts unchanged) |
| C-5 Red means a failing check and nothing else (the plots, stage 1 review C-3) | Task 2 |
| 6.1 and G-7 Text contrast of at least 4.5:1 (egui_plot's faded tick numbers, stage 1 review E-7) | Task 3 |
| 8.2 test 9, goal G-4: every `CentreView` reachable | Task 1 (`Harness::tab` through the picker; the nothing-lost test unchanged) |
| 8.2 test 15: the layout at 1024 x 700 | Task 1 (the tab row on one row at 1024 x 700; the app's text-cut test on every plot) |
| 8.3 Text rules: the new string through the plain-words guard | Task 1 |
| 9.1 The docs, the backlog item and the gui-smoke checklist with the slice | Tasks 4 and 5 |
| 9.4 Release | Task 6 |
| Not in this part | Part 2b-iii-b (the look: E-4, R-6, S-1, S-4) and part 2b-iii-c (the Equation panel: BL-086, BL-088, BL-089, BL-101's Equation panel part). |
