# Magcoupler UX overhaul, slice B, stage 2, part 2a (the top bar and the notes): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ] `) syntax for tracking.

**Goal:** The top bar becomes the design's (4.2): Undo, Redo, **Design** (a menu), **Share**, **Export** and **Learn**, each a word with no box until the pointer is on it. The teaching notes move into the Explain panel: Learn lists them by title, and every note link on the page selects its note there, each step of the Start here walk included. The Equation panel loses its Explain checkbox and its button on the bar (C-11). The slip-heating readout loses its stray unit (9.2). Nothing that is on screen today is lost (the nothing-lost table).

**Architecture:**
- **The top bar** stays `MagcouplingPanel::header_ui`. The menu is egui's (`egui::containers::menu::MenuButton`), its items `design_menu_ui`; a bar button is `bar_button` (`Button::frame_when_inactive(false)`). The reset's name follows the baseline (`reset_label`).
- **The notes are the Explain panel's.** `explain::Selection` gains `Note(id)` and `NoteList`. A note's entry calls the note renderer the Equation panel used (`explorer::note_ui`) after the links to its equations (`explorer::explains_ui`, from the old note-alone view). Every way to a note sets the panel's selection: the readouts' note event, the Size column's Why links, and `explorer_ui`'s new return value (the lesson line under an open equation, a Start here step). The Equation panel keeps its equations, its walk bar and its Close button, and loses its Explain checkbox, its `explain` and `note` state and its note view.
- **One fix for the unit.** `format::value_with_unit` leaves a text value without its unit; the five readouts that pair a value with its unit call it.
- **Unchanged:** the engine, the exports, the columns, the Size column, the tiles, the check list, the check and tile entries, the centre tabs, the linkage app.

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32, egui_plot 0.33; the headless egui harness (`magcoupling-rs/src/gui/test_support.rs`, the panel tests' `Harness`, the app tests' `frame_on`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. This part is its section 4.2 (the top bar), the notes of 5.1 and 5.2 (a note and the note list as selections; "a note link selects the note"), 5.5's walk (each step selects its note), C-11 and C-12, tests 9 (the top bar's actions; every shown note in the Learn list) and 10 (the walk) of section 8.2 where they fall in this part, section 9.2's stray unit, stage 1's review suggestion S-3, and the docs and release rules of section 9. Executors read the spec's sections 2.3, 4.2, 5, 8.2, 8.3 and 9 beside this plan. Part 1's plan (`docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`, "Why a part, and what follows") and BL-081's notes set the scope of part 2; this plan cuts it again (decision D-20). Backlog item: **BL-085** (`docs/ai/backlog.yaml`, filed with this plan).

**How this plan was made.** The part was built once as a prototype on a copy of `main` at `290c99b` (BL-081 merged and closed) with BL-085 filed, one commit per task. Every block below was generated from those commits' diffs (a task's red step is its test hunks, its green step the rest), then applied, in this plan's order, with the apply tool, to a fresh copy of the same base. Each block applied once; the result equals the prototype file for file (line endings aside) and passes `cargo test` with and without the app feature (1077 lib tests with it against 1070 on the base), both native clippy runs, both wasm32 clippy runs of Task 4's step 4, `cargo fmt --check` and the gui-smoke verdict test (all 77 checks hold). Each red step's "Expected" below is what that replay printed, and each count after a task is the replay's. Not run for this plan: gates 1 to 3 (linkage-sim-rs, untouched), gate 12 (the Python oracle), the release WASM build and the browser. Task 6 runs them.

**The plan's review fix wave (2026-11-22).** Edited by hand after the generation above, and replayed the same way on a fresh copy of the same base: Task 3 draws the lesson line's title on its own row, so a long title wraps inside the Equation panel (finding UXP-1: at 1024 x 700 three walk steps' titles ran up to 58 points past the panel's edge), with an app test that walks every Start here step at three window sizes (red on the generated Task 3: "Back iron: why steel behind the magnets adds torque" cut by 40 points at step 3), and drops a stray doc comment the generator left on `lesson_ui`; Task 3's idle-frame test also loops over a note's entry and the note list (COV-1); Task 4 brings the gui-smoke description up to the new prompt (COV-3); decisions D-24 (a departure from 5.3's order, COV-2), D-32 (UXP-3) and D-33 (UXP-2, BL-086) are recorded. The counts after Task 3 and Task 4 are this replay's (1078 lib tests with the app feature).

**Where this part applies.** On `main` at `290c99b` with this plan's commit (the plan and BL-085), and nothing else that touches `magcoupling-rs/`, `.claude/workflows/gui-smoke.js` or Task 5's nine docs. If a block does not match, stop and report: the controller regenerates the blocks (the planner's tools are listed under "Before Task 1"), it does not adapt a block in place.

## Why a part, and what follows

BL-081's part 2 holds the top bar, the Size column's move into the Explain panel with inputs as selections, three more kinds of selection, the Plots tab, the word budget, the controls' outlines, a second font face and the page shell. Each touches `panel.rs` (11,000 lines) and its tests; together they are again more than one reviewer can hold. This plan cuts part 2 in two (decision D-20). Part 2a, this plan, changes the top bar and where a note is read, and nothing in the three columns. It ships on its own and loses nothing: every top-bar action is in the bar or the Design menu, and every note is one click from Learn.

**What follows, as part 2b, with its own plan and backlog item:**

1. The Size column (4.3): the folds Detailed inputs and Assumptions; every hint, "Set by", floor and Why line under a control and the `LINKED_HOUSING` footnote moving into the Explain panel when the control is selected; inputs as a `Selection` (touching an input selects it; "Highlight the results this drives" keeps the trace); the torque card's row of each size control in its entry (the second half of C-10); an entry's "Spreadsheet cell" foot link with the details on (5.3).
2. More selections (5.1, 5.2): dimension tags, results-table rows and callout rows. With the tags as selections the callout list leaves the opening tab for the Geometry tab (4.4; part 1's decision D-19), and the three dashboard rows of part 1's D-17 get their one-click home.
3. The Plots tab with its picker (C-9), the plots' red reference lines on a passing design (stage 1 review C-3) and egui_plot's fading tick numbers (E-7).
4. The word budget of 250 at 1280 x 800 (goal G-1, test 8), which needs items 1 and 2; this part lowers the ceiling to 430 (decision D-29).
5. The controls' outlines (stage 1 review E-4, R-6), a SemiBold face for the title and the headings (S-1), the page shell's loading line (S-4).
6. The full table-driven nothing-lost test of test 9 (every `CentreView`, every input, every `DASHBOARD` row's home), once the Size column has moved.

BL-083 (the Hot-low tile's gap at a hair) and BL-084 (the band torques' help rows) stay open: the design gives neither text (decision D-30).

## What the prototype showed

1. **Words.** At 1280 x 800 the first view draws 443 words on `main` and 427 after this part (the app test's count of every text shape drawn in view, part 1's measure): the bar's ten labels become six.
2. **The menu.** egui's menu opens under its button on the click's release and closes on any click inside it, the details checkbox included, or outside it. A hover on the checkbox shows its hint only once the pointer has moved after the click that opened the menu: egui holds a tooltip back while the last click is newer than the last move (`Tooltip::should_show_tooltip`), so the panel test moves the pointer a second later.
3. **A bar button's box.** `frame_when_inactive(false)` draws no fill at rest and the button's hovered fill under the pointer; the bar's own fill behind the words is the top panel's (a rect as wide as the window), which the test tells apart by its width.
4. **The notes in a 300-point column.** Every note draws in the Explain panel at the right column's width: its diagram, its change entries (wrapped, the input label still a link that shows its row) and its sources fold. `every_teaching_note_shows_with_glyphs_in_the_default_fonts` draws every note there and opens each fold.
5. **The walk.** The orientation's Start here link starts the walk; the Equation panel opens at the step's equation and the Explain panel shows the step's note. The step's title then shows twice (the note's heading and the lesson line under the equation); the Equation panel's walk bar names it a third time inside its step line (kept in this part: decision D-32).
6. **The stray unit was wider than the plot.** "never: steady state stays below the limit s" came from `with_unit` after any value. The same pairing is in the dashboard rows (drawn in a check's entry), the results table, the Equation panel's value and the clamp tab: each now calls `value_with_unit`. At the example design the slip-heating readout's time to the limit ends "never: steady state stays below the limit".

## Global Constraints

- **No engine number moves** (the design's 2.3). No file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` changes.
- **The linkage app is not touched** (O-1). No file under `linkage-sim-rs/src/` or `linkage-sim-rs/web/` changes.
- **The panel sets no theme and no font** (2.2, 6.3). Code under `magcoupling-rs/src/gui/` reads `ui.visuals()`.
- **No new explanatory text** (5.3). The new on-screen strings are the bar's and the menu's words: `DESIGN_MENU` ("Design"), `SHARE` ("Share"), `EXPORT` ("Export"), `LEARN` ("Learn", also the note list's heading) and `RESET_TO_START` ("Reset to the starting design", the design's own words). Each passes the plain-words guard and the glyph guard (Tasks 2, 3). The note entries are the reviewed notes, drawn by the renderer the Equation panel used. `EMPTY_TEXT` loses its last clause ("Explain shows the note under the open equation."), which names the checkbox that goes.
- **Crimson is never text, never on a value, never in the check list** (C-5). The bar's buttons and the menu use egui's widget colours from the theme.
- **The selection is view state** (5.1): a note or the note list is in no design file, share link or undo step, as a check or a tile is.
- **The exports do not change** (12.1). `value_with_unit` changes what the screen draws, not what an export writes (the exports format their own values).
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
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-22-magcoupler-overhaul-b2p2a/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Shell.** On this machine `bash` typed in PowerShell is WSL, which fails on the checkout's line endings. Shell scripts run through Git Bash: `& 'C:\Program Files\Git\bin\bash.exe' <script>`. Python scripts are written with the Write tool, not a here-document.
- **Never run `cargo fmt` inside `linkage-sim-rs`.** `magcoupling-rs` is fmt-clean; keep it so.
- **Docs ship with the code** (repository rule): Task 5 sweeps the README and `docs/ai/`, the register, the owner-question list and the design's status line.

## Model per task

CLAUDE.md section 5. BL-085 is `risk: mechanical` (decision D-31): the part shows no new number and makes no new physical claim; its new strings are the bar's and the menu's words, and the notes it moves are reviewed text. So the tasks that transcribe the plan's exact blocks run on `sonnet`, and every review and the release run on `fable`. A `sonnet` task that ends blocked, leaves the gate red or is rejected in review is redone on `fable` (escalate, never downgrade).

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | A text value takes no unit (`format::value_with_unit`) | `sonnet` |
| 2 | The top bar and its Design menu | `sonnet` |
| 3 | The notes in the Explain panel; the Equation panel's checkbox goes; the walk selects its notes | `sonnet` |
| 4 | gui-smoke reads the top bar and the note in the Explain panel | `sonnet` |
| 5 | Docs sweep | `sonnet` |
| 6 | Release: the full gate, gui-smoke, the text review, the independent whole-branch review, one fix wave, the merge and the push | `fable` controller; fresh-context `fable` reviewers; `haiku` for git; the smoke workflow's own agents on `sonnet` |

Each task's diff is reviewed by a fresh-context reviewer before the next task starts (`sonnet` for these mechanical diffs, CLAUDE.md section 5); it checks the diff against the task's blocks and nothing else. A task that ends blocked or leaves a test red stops the build and goes back to the controller.

## Review Focus

Five inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named.

1. **A panel whose baseline is the spreadsheet's design** (a host's, or a test's, built with `MagcouplingPanel::new`). "Reset to the starting design" would name a design that is not there, and "Open the example design" would do what the reset does. Task 2: `a_panel_whose_baseline_is_the_spreadsheet_s_design_has_no_example_button` (the menu holds "Reset all" and no example item there, and "Reset to the starting design" and the example item on the app's panel).
2. **The checkbox inside a menu.** A click on "Show spreadsheet details" closes the menu; a person who wants it off again opens the menu again. Its hover must still show. Task 2: `the_details_toggle_starts_off_and_a_click_turns_it_on_and_off`.
3. **A note link while the Equation panel shows an equation.** A person reading an equation clicks its lesson line: the note shows in the Explain panel and the equation stays (C-25). Task 3: `the_open_equation_s_lesson_line_shows_its_note_in_the_explain_panel`, `a_note_link_is_the_explain_panel_s_and_leaves_the_open_equation`.
4. **A held input in a note** (Auto on the magnet length). The note's change entry for it is plain text with its reason and no link, now in the Explain panel. Task 3: `with_auto_on_the_step_1_note_names_the_length_auto_holds_and_draws_the_design_shown`.
5. **Esc on a note.** Esc returns the orientation from a note or the note list as from a check (5.1). Task 3: `learn_lists_the_notes_in_the_explain_panel_and_a_title_shows_its_note`.

## Decisions this plan takes

Each is a reading of the design where it was open, numbered after part 1's D-1 to D-19 (one stage, one list). The owner may overrule any of them; Task 6 lists them in its report. The ones that need a row in the register have one (Task 5).

| Id | Decision | Why |
|---|---|---|
| D-20 | Part 2 is built in two parts; this plan is part 2a (the top bar with its Design menu, Share, Export and Learn; the notes and the note list in the Explain panel; the Equation panel's checkbox and bar button gone; each Start here step selecting its note; the stray unit; S-3). Part 2b is listed under "What follows". Register row A-23. | Part 2 entire is again more than one reviewer can hold (part 1's D-1 reasoning). This part changes no column, and loses nothing. The conservative answer to Q-4 is smaller releases. |
| D-21 | The Design menu's reset reads "Reset to the starting design" where the baseline is not the spreadsheet's design, and keeps "Reset all" where it is (a panel built with `MagcouplingPanel::new`); "Open the example design" shows only in the first case, as its button did. | The design's words name the starting design, which a panel at the spreadsheet's baseline does not have (C-21: such a panel "sees today's behaviour"). |
| D-22 | The status line of a file or a link stays under the bar, where it is today; the design's 4.2 puts it at the bar's right end. Register row A-24; owner question 50. | Tests and the smoke run read it there; a long status at the bar's end wraps the bar at 1280 points. The conservative answer moves nothing the owner has not seen. |
| D-23 | Under the open equation the lesson line names its reviewed note: "Full lesson:", and on the row under it the note's title, a link that selects the note (on its own row so a long title wraps inside the panel at 1024 x 700; plan review UXP-1); for an equation without a note, the next best read the Explain checkbox drew (`note_fallback_ui`), now always drawn. The note itself is not drawn in the Equation panel. | C-11: "the note for the open equation shows in the Explain panel". Two words and a title keep the equation one click from its note; no new string ("Full lesson:" is an Explain entry's). |
| D-24 | Every note link on the page selects its note (5.2): the Size column's Why links, the warnings' links, the entries' lesson links, the end-effect banner's link, the lesson line and each Start here step. The Equation panel's note-alone view goes; its "Explains" links (`explains_ui`) head the note's entry, and one opens the Equation panel at that equation and leaves the selection (C-25). Heading the entry departs from 5.3's one form (the title leads, links at the foot; plan review COV-2). | One place for words (C-11). The note-alone view would be a second place showing a note. The links stay where the note-alone view had them so they are in view without a scroll: at the foot, a long note's links fall below the Explain panel's fold (tried in the fix wave: the flux-path note's "circuit" link went out of view in `a_warning_shows_in_its_colour_and_links_to_its_note`). The owner may move them to the foot. |
| D-25 | Learn lists every reviewed note shown, in the notes' order (`explain::shown_notes`), by title, under the heading "Learn"; while the shaft-clamp section is paused its note is left out. The bar's Learn button has no hover text. | The design's 4.2 ("lists the shown notes by title"); `NOTES`' order is the reading order the notes were written in. A hover would be one more new string, and the word says it. |
| D-26 | Opening the walk from the orientation, Next, Previous and Stop keep working as today; Stop leaves the step's note selected. | 5.5 says each step selects its note and says nothing of Stop; leaving the note keeps what the reader was reading. |
| D-27 | A text value takes no unit, fixed once in `format::value_with_unit` for the five readouts that pair a value with its unit (the plots' readouts, the dashboard rows drawn in an entry, the results table, the Equation panel's value, the clamp tab), not only the slip-heating plot's. | The design's 9.2 names one readout; the cause is shared (`with_unit` after any value), and the same words show in the results table. |
| D-28 | The bar's buttons (Undo, Redo, Design, Share, Export, Learn) are words at rest and boxed under the pointer (`frame_when_inactive(false)`). The menu's items are egui's menu buttons; a click on any item, the checkbox included, closes the menu (egui's default). | Stage 1 review S-3; C-6 ("borders are hairlines; no gradients"). egui's default close keeps one rule for every item. |
| D-29 | The first view at 1280 x 800 is held to at most 430 words (427 now, 443 before); the design's 250 is part 2b's test. | Part 1's D-9 ceiling, lowered by what this part removed, so it cannot grow back. |
| D-30 | BL-083 and BL-084 are not folded in. | The task allows them only as text the design already gives. BL-083 needs a new quoting of the gap and BL-084 new help rows; both need a physics and text review. They stay open. |
| D-31 | BL-085 is `risk: mechanical`; its build tasks run on `sonnet`, its reviews and release on `fable`. | No new number or physical claim (the design's header: `risk: physics` "wherever a slice shows a new number or a new physical claim"). |
| D-32 | During the walk the step's title shows three times on one screen (the note's heading in the Explain panel, the lesson line, the walk bar's "Start here N of 14: title"). Kept in this part (plan review UXP-3). | Each copy is its own place's form (an entry's heading, the equation's link to its note, the walk's step line) and part 2a adds no text and drops none from the walk; part 2b's word budget of 250 (test 8) decides which copies go. |
| D-33 | The Equation panel's term rows and tags, and the walk bar's step line, are cut at the panel's edge on some walk steps (at 1024 x 700, and some term rows at 1280 x 800), now that the note no longer pushes the terms below the fold. Filed as BL-086, not fixed here (plan review UXP-2). | The term grid and the walk bar are today's code, unchanged by this part; making them wrap is a layout change with its own test (BL-086's acceptance: the walk test of UXP-1 widened to the whole Equation panel). The line this part adds, the lesson line, is drawn whole and tested (D-23). |

## Nothing lost

Every text and control of today's top bar and of the notes' places (at `290c99b`) that this part moves, and where it is after. Anything not listed (the three columns, the tiles, the check list, the check and tile entries, the centre tabs, the Equation panel's equations, breadcrumb, terms, Used by, walk bar and Close) does not move. The "Test" column names the test that holds the new home.

| Today | After this part | Test |
|---|---|---|
| Top bar: Undo, Redo (boxed buttons) | Same place and action, words with no box at rest | `panel::tests::the_top_bar_is_undo_redo_design_share_export_and_learn_and_the_menu_holds_the_rest`, `the_top_bar_s_buttons_show_their_box_only_under_the_pointer`, `the_keyboard_shortcuts_undo_and_redo` |
| Top bar: Reset all | Design menu: "Reset to the starting design" (the app), "Reset all" (a panel at the spreadsheet's baseline); same action and status line (D-21) | `a_panel_opens_on_its_baseline_and_reset_all_returns_to_it`, `a_panel_whose_baseline_is_the_spreadsheet_s_design_has_no_example_button` |
| Top bar: Open the example design (its hover) | Design menu, same words, hover and action, same condition | `open_the_example_design_is_one_undo_step_and_keeps_the_baseline` (and the test above) |
| Top bar: Save design, Load design | Design menu, same words and requests | `save_design_queues_the_design_file_and_load_design_asks_for_one` |
| Top bar: Copy share link (its hover) | **Share**, same action, hover and status line | `copy_share_link_puts_the_design_s_link_on_the_clipboard` |
| Top bar: Export spreadsheet (its hover) | **Export**, same action and hover; the Results tab keeps Export XLSX | `both_spreadsheet_buttons_queue_the_workbook_of_the_design_shown`, `the_spreadsheet_follows_the_input_order_shown_and_torque_to_magnets` |
| Top bar: Equation panel toggle | Gone (C-11): a value's click or an entry's Equation link opens the panel, its Close button closes it | `the_equation_panel_starts_closed_with_no_button_on_the_bar_and_a_value_opens_it`, `closing_the_equation_panel_clears_the_input_row_s_frame` |
| Top bar: Show spreadsheet details checkbox (its hint) | Design menu, same words, hint and setting (C-12) | `the_details_toggle_starts_off_and_a_click_turns_it_on_and_off`, `the_details_toggle_is_a_view_choice_in_no_file_link_or_undo_step` |
| (none) | **Learn**: the notes shown, by title, in the Explain panel | `learn_lists_the_notes_in_the_explain_panel_and_a_title_shows_its_note`; `explain::tests::learn_lists_every_shown_note_by_title_and_a_title_selects_it` |
| Equation panel: the Explain checkbox and the note under the open equation | Under the equation the lesson line ("Full lesson:" and the note's title on the row under it, or the next best read); a click shows the note in the Explain panel (D-23) | `the_open_equation_s_lesson_line_shows_its_note_in_the_explain_panel`; `explorer::tests::the_lesson_line_names_the_open_equation_s_note_and_asks_for_it`; `app::tests::every_start_here_step_s_lesson_line_is_drawn_whole_in_the_app_s_own_font` |
| Equation panel: a note opened on its own (a warning's link, a lesson link), with its "Explains" links and its title in the header row | The note's entry in the Explain panel: "Explains" links (one opens the equation in the Equation panel), then the note (D-24) | `a_warning_shows_in_its_colour_and_links_to_its_note`, `a_check_s_lesson_link_shows_its_note_in_the_explain_panel`, `explain::tests::a_note_s_entry_is_the_note_with_the_links_to_its_equations` |
| Size column: the Why links under Magnets per ring, Outside diameter, beside Required torque; the card's Why link | Same links; the note shows in the Explain panel | `the_size_tab_s_why_links_open_their_notes` |
| Centre: the end-effect banner's Why link | Same link; the note shows in the Explain panel | `the_end_effect_banner_links_to_its_note_over_the_centre` |
| Start here: each step's equation with the Explain checkbox switched on | Each step's equation in the Equation panel and its note in the Explain panel (5.5) | `start_here_opens_the_matching_equations_in_turn`; `explorer::tests::start_here_walks_the_suggested_order_and_a_click_leaves_it` |
| The empty Equation panel's line naming Start here and Explain | Names Start here (the checkbox is gone) | `explorer::tests::the_empty_text_and_explain_s_own_lines_name_the_buttons_in_plain_words` |
| A text value's readout with the result's unit after it ("... the limit s") | The words alone (D-27) | `plots::tests::the_slip_heating_readout_puts_no_unit_after_never`, `each_plot_reads_out_the_design_values_it_draws` |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/format.rs` | `value_with_unit` | 1 |
| `magcoupling-rs/src/gui/plots.rs`, `dashboard.rs`, `results_table.rs`, `explorer.rs`, `clamp_drawing.rs` | A value's unit through `value_with_unit` (1); `dashboard.rs` two doc lines (3) | 1, 3 |
| `magcoupling-rs/src/gui/panel.rs` | `DESIGN_MENU`, `RESET_TO_START`, `SHARE`, `EXPORT`; `header_ui` without the toggle; `design_menu_ui`, `reset_label`, `bar_button`; tests (2). Learn on the bar; a note asked for anywhere selects it; the walk's note; `explorer_ui` without `held`; tests (3) | 2, 3 |
| `magcoupling-rs/src/gui/mod.rs` | `DESIGN_MENU` (2), `EXPORT`, `LOAD_DESIGN`, `RESET_TO_START`, `SHARE` (4) re-exported | 2, 4 |
| `magcoupling-rs/src/app.rs` | The first view's example design is in the menu (2); the word ceiling at 430 and the walk's lesson line drawn whole (3); the smoke prompt's guard (4) | 2, 3, 4 |
| `magcoupling-rs/src/gui/explain.rs` | `Selection::Note`, `Selection::NoteList`, `LEARN`, `shown_notes`, a note's entry, the note list; `Shown::held`; tests | 3 |
| `magcoupling-rs/src/gui/explorer.rs` | `explain` and `note` state, `open_note`, `note`, `note_alone_ui` and `EXPLAIN` go; `start` returns the step's note; `follow_note_link` returns a note; `explorer_ui` returns the note asked for; `lesson_ui`; `explains_ui`; `crumb` visible to the crate; `EMPTY_TEXT` shorter; tests | 1, 3 |
| `magcoupling-rs/src/gui/readouts.rs` | One doc line | 3 |
| `magcoupling-rs/src/gui/test_support.rs` | `check_entry_texts` gives `Shown` its `held` | 3 |
| `.claude/workflows/gui-smoke.js` | The prompt reads the top bar, the Design menu and the note in the Explain panel; the description says so | 4 |
| `magcoupling-rs/README.md`, `docs/ai/*`, the register, the owner questions, the design's header | The docs sweep | 5 |
| `docs/ai/backlog.yaml` | BL-085 in progress (5), fixed (6) | 5, 6 |

**Existing tests this plan edits, and how.** No assertion about behaviour that stays is weakened; where an action moved, the test reaches it in its new place and asserts what it asserted.
- `panel::tests`, Task 2: every click on Reset all, Open the example design, Save design, Load design and Show spreadsheet details goes through the Design menu (`Harness::menu`); a panel with the starting design as its baseline clicks "Reset to the starting design"; Copy share link and Export spreadsheet are clicked as Share and Export. `the_details_toggle_starts_off_and_a_click_turns_it_on_and_off` also holds that the toggle is not on the bar and that a click closes the menu. `a_panel_whose_baseline_is_the_spreadsheet_s_design_has_no_example_button` reads the open menu and also holds the reset's two names. The text lists of the plain-words and glyph tests add the four new strings and the menu's items. `the_equation_panel_starts_closed_and_its_header_button_toggles_it` becomes `the_equation_panel_starts_closed_with_no_button_on_the_bar_and_a_value_opens_it`; `closing_the_equation_panel_clears_the_input_row_s_frame` closes by Close alone and shows the panel again by its state; `start_here_opens_the_matching_equations_in_turn` starts from the orientation's link.
- `panel::tests`, Task 3: `the_explain_toggle_shows_the_open_equation_s_note_and_follows_it` becomes `the_open_equation_s_lesson_line_shows_its_note_in_the_explain_panel` (same notes, same diagram count, now in the Explain panel; the next-best-read lines the same); every assertion that a link "opened note X" (`explorer.note() == Some(X)`) reads the selection (`Selection::Note(X)`), with the same link and the same note; `a_check_s_why_link_opens_its_note_in_the_equation_panel` is renamed `a_check_s_lesson_link_shows_its_note_in_the_explain_panel`; the tests that set `explorer.explain = true` drop the line (the lesson line is always drawn); `start_here_opens_the_matching_equations_in_turn` also holds each step's note as the selection; `the_check_list_and_the_explain_panel_compute_nothing_on_an_idle_frame` also loops over a note's entry and the note list (plan review COV-1; spec 8.2, test 14).
- `explorer::tests`, Task 3: the four `explorer_ui` calls drop `held`; `start_here_walks_the_suggested_order_and_a_click_leaves_it` holds the note each step returns in place of the checkbox's state; `a_note_opened_alone_replaces_the_trail_until_a_value_is_opened` becomes `a_note_link_is_the_explain_panel_s_and_leaves_the_open_equation`; `a_note_s_link_highlights_its_input_s_row_and_keeps_the_note_open` becomes `a_note_s_link_highlights_its_input_s_row_and_asks_for_no_note` (same focus assertion); the empty text's test holds that it no longer names Explain; `a_wrapped_change_entry_stays_inside_a_narrow_panel_and_its_label_links` draws the note with `note_ui`.
- `explain::tests`, `test_support.rs`, Task 3: each `Shown` gives `held: &|_| None`.
- `plots::tests::each_plot_reads_out_the_design_values_it_draws`, Task 1: a text value's expected readout has no unit.
- `app::tests`, Tasks 2 to 4: `the_app_opens_the_starting_design` finds Design on the first view and the example item not; the word test is renamed `the_first_view_draws_at_most_four_hundred_and_thirty_words_at_1280_by_800`; `the_smoke_test_share_link_opens_its_design` holds the new prompt.

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` after this plan has merged, create the branch `magcoupling/overhaul-b2-part2a` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  ```

  Expected: `test result: ok. 1070 passed; 0 failed; 6 ignored`. Another count means `main` is not the base the blocks were checked on: stop and report.

- [ ] **Step 3: Check the base.** `git log --oneline 290c99b..main -- magcoupling-rs .claude/workflows/gui-smoke.js linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs docs/ai docs/analyses/2026-11-10-conservative-assumptions.md docs/analyses/2026-11-18-owner-questions.md docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` lists this plan's commit (it touches `docs/ai/backlog.yaml`) and nothing else. If it lists more, stop: the controller regenerates the blocks on that `main` with the planner's tools (gitignored, in the main checkout's `.superpowers/sdd/2026-11-22-magcoupler-overhaul-b2p2a/`: `gen_blocks.py`, the scripts that built each task's state, `frags.py` and `replay.py`) before Task 1.

## Tasks

### Task 1: A text value takes no unit

**Model:** `sonnet` (BL-085 is `risk: mechanical`; the task transcribes exact blocks).

The design's 9.2: the slip-heating readout's "never: steady state stays below the limit s". `format::value_with_unit` writes a value's unit after it unless the value is a text, and the five readouts that pair a value with its unit call it (D-27).

**Files:**
- Modify: `magcoupling-rs/src/gui/format.rs`, `magcoupling-rs/src/gui/plots.rs`, `magcoupling-rs/src/gui/dashboard.rs`, `magcoupling-rs/src/gui/results_table.rs`, `magcoupling-rs/src/gui/explorer.rs`, `magcoupling-rs/src/gui/clamp_drawing.rs`

**Interfaces:**
- Produces: `format::value_with_unit(value: &Value, text: String, unit: &str) -> String`.

- [ ] **Step 1 (red): the readout's test.** The blocks also move `with_unit` into the test modules that still call it (the green step takes it out of the files' own imports).

**`magcoupling-rs/src/gui/explorer.rs`**

Replace:

```rust
    use crate::gui::format::format_value;
    use crate::gui::results_table::table_entries;
```

with:

```rust
    use crate::gui::format::format_value;
    use crate::gui::format::with_unit;
    use crate::gui::results_table::table_entries;
```

**`magcoupling-rs/src/gui/plots.rs`**

Replace:

```rust
    use crate::compute_all;
    use crate::gui::test_support::{drawn_texts, flat_shapes, short_magnets, sized_frame};
```

with:

```rust
    use crate::compute_all;
    use crate::gui::format::with_unit;
    use crate::gui::test_support::{drawn_texts, flat_shapes, short_magnets, sized_frame};
```

Replace:

```rust

    /// The point count of every path egui painted, and the radius of every circle.
```

with:

```rust

    #[test]
    fn the_slip_heating_readout_puts_no_unit_after_never() {
        // UX overhaul spec 9.2: with no drag the steady state stays below the limit, and the
        // time to the limit is the engine's words, which read "never: ... the limit", not
        // "the limit s".
        let mut inputs = DesignInputs::default();
        inputs.metal.measured_drag_Nm = Some(0.0);
        let path = "temperature.thermal.time_to_limit_high";
        assert_eq!(
            compute_all(&inputs).get(path),
            Some(Value::Text(crate::engine::temperature::NEVER.to_owned()))
        );
        let texts = drawn_texts(&draw(PlotKind::SlipHeating, &inputs));
        let want = format!(
            "{}: {}",
            crate::gui::plain_text::label_for(path),
            crate::engine::temperature::NEVER
        );
        assert!(texts.contains(&want), "missing {want:?} in {texts:?}");
        assert!(texts.iter().all(|t| !t.ends_with(" limit s")), "{texts:?}");
    }

    /// The point count of every path egui painted, and the radius of every circle.
```

Replace:

```rust
                let value = results.get(path).unwrap();
                let want = format!(
                    "{}: {}",
                    crate::gui::plain_text::label_for(path),
                    with_unit(format_value(&value), info.meta.unit)
                );
                assert!(texts.contains(&want), "{kind:?}: missing {want:?}");
```

with:

```rust
                let value = results.get(path).unwrap();
                // A text value is words and takes no unit (UX overhaul spec 9.2).
                let shown = match &value {
                    Value::Text(text) => text.clone(),
                    _ => with_unit(format_value(&value), info.meta.unit),
                };
                let want = format!("{}: {shown}", crate::gui::plain_text::label_for(path));
                assert!(texts.contains(&want), "{kind:?}: missing {want:?}");
```

**`magcoupling-rs/src/gui/results_table.rs`**

Replace:

```rust
    use super::*;
    use crate::gui::format::format_value;

```

with:

```rust
    use super::*;
    use crate::gui::format::{format_value, with_unit};

```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib plots:: 2>&1 | tail -n 15
  ```

  Expected: 2 failed: `the_slip_heating_readout_puts_no_unit_after_never`, `each_plot_reads_out_the_design_values_it_draws`.

- [ ] **Step 3 (green): `value_with_unit` and its five callers.**

**`magcoupling-rs/src/gui/clamp_drawing.rs`**

Replace:

```rust
use crate::gui::dashboard::{Level, hover_text};
use crate::gui::format::{format_value, with_unit};
use crate::gui::geometry::{Mm, finite};
```

with:

```rust
use crate::gui::dashboard::{Level, hover_text};
use crate::gui::format::{format_value, value_with_unit, with_unit};
use crate::gui::geometry::{Mm, finite};
```

Replace:

```rust
                        ui.label(egui::RichText::new(info.meta.label).weak());
                        ui.label(with_unit(format_value(&value), info.meta.unit));
                    })
```

with:

```rust
                        ui.label(egui::RichText::new(info.meta.label).weak());
                        ui.label(value_with_unit(
                            &value,
                            format_value(&value),
                            info.meta.unit,
                        ));
                    })
```

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
use crate::gui::explorer::reviewed_note;
use crate::gui::format::{format_value, with_unit};
use crate::gui::guidance::{BANNER_NOTE, Learn, design_status, status_line};
```

with:

```rust
use crate::gui::explorer::reviewed_note;
use crate::gui::format::{format_value, value_with_unit};
use crate::gui::guidance::{BANNER_NOTE, Learn, design_status, status_line};
```

Replace:

```rust
                label: info.meta.label,
                value: with_unit(format_value(&value), info.meta.unit),
                level,
```

with:

```rust
                label: info.meta.label,
                value: value_with_unit(&value, format_value(&value), info.meta.unit),
                level,
```

**`magcoupling-rs/src/gui/explorer.rs`**

Replace:

```rust
use crate::gui::diagrams::diagram_ui;
use crate::gui::format::with_unit;
use crate::gui::guidance::{Learn, result_text, result_unit, row_learn};
```

with:

```rust
use crate::gui::diagrams::diagram_ui;
use crate::gui::format::value_with_unit;
use crate::gui::guidance::{Learn, result_text, result_unit, row_learn};
```

Replace:

```rust
fn value_text(path: &str, value: &Value, unit: &str) -> String {
    let text = with_unit(shown_text(path, value), unit);
    match value {
```

with:

```rust
fn value_text(path: &str, value: &Value, unit: &str) -> String {
    let text = value_with_unit(value, shown_text(path, value), unit);
    match value {
```

**`magcoupling-rs/src/gui/format.rs`**

Replace:

```rust
        format!("{text} {unit}")
    }
```

with:

```rust
        format!("{text} {unit}")
    }
}

/// A value's text (`text`, drawn from `value`) followed by its unit as [`with_unit`] writes it,
/// unless the value is a text: words take no unit (UX overhaul spec 9.2: the slip-heating
/// readout read "never: steady state stays below the limit s").
pub fn value_with_unit(value: &Value, text: String, unit: &str) -> String {
    match value {
        Value::Text(_) => text,
        _ => with_unit(text, unit),
    }
```

**`magcoupling-rs/src/gui/plots.rs`**

Replace:

```rust
use crate::gui::dashboard::{Level, end_effect_out_of_range, hover_text, result_info};
use crate::gui::format::{format_value, with_unit};
use crate::gui::look;
```

with:

```rust
use crate::gui::dashboard::{Level, end_effect_out_of_range, hover_text, result_info};
use crate::gui::format::{format_value, value_with_unit};
use crate::gui::look;
```

Replace:

```rust
                label_for(path),
                with_unit(format_value(&value), info.meta.unit)
            );
```

with:

```rust
                label_for(path),
                value_with_unit(&value, format_value(&value), info.meta.unit)
            );
```

**`magcoupling-rs/src/gui/results_table.rs`**

Replace:

```rust
};
use crate::gui::format::{non_finite_text, search_haystack, search_needle, with_unit};
use crate::gui::plain_text::display_label;
```

with:

```rust
};
use crate::gui::format::{non_finite_text, search_haystack, search_needle, value_with_unit};
use crate::gui::plain_text::display_label;
```

Replace:

```rust
            number,
            &with_unit(shown_text(&entry.path, &value), entry.info.meta.unit),
            level,
```

with:

```rust
            number,
            &value_with_unit(
                &value,
                shown_text(&entry.path, &value),
                entry.info.meta.unit,
            ),
            level,
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1071 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** the six files: `fix(magcoupler): a text value takes no unit; the slip-heating readout reads "never: ... the limit" (BL-085 T1)`.

### Task 2: The top bar and its Design menu

**Model:** `sonnet`.

The bar becomes Undo, Redo, Design, Share and Export (Learn joins in Task 3), each a word with no box at rest. The Design menu holds the reset, the example design, the file's save and load and the details toggle. The Equation panel button leaves the bar.

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/mod.rs`, `magcoupling-rs/src/app.rs`

**Interfaces:**
- Produces: `panel::{DESIGN_MENU, RESET_TO_START, SHARE, EXPORT}` (re-exported: `DESIGN_MENU`); `MagcouplingPanel::design_menu_ui`, `reset_label` (private); `bar_button(text) -> egui::Button` (private).
- Removes: `COPY_SHARE_LINK`, `EXPORT_SPREADSHEET` (renamed `SHARE`, `EXPORT`), the bar's Equation panel toggle.

- [ ] **Step 1 (red): the top bar's tests.** The blocks add `Harness::menu`, the two new tests, and move every test click on a menu item through the menu.

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
pub use panel::{
    CentreView, MagcouplingPanel, OPEN_EXAMPLE, PanelRequest, SHOW_DETAILS, SIZING_LOG_PREFIX,
};
```

with:

```rust
pub use panel::{
    CentreView, DESIGN_MENU, MagcouplingPanel, OPEN_EXAMPLE, PanelRequest, SHOW_DETAILS,
    SIZING_LOG_PREFIX,
};
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        assert_eq!(count(CHANGED_DOT), 0, "no input is changed at the baseline");
        assert_eq!(
            count(crate::gui::OPEN_EXAMPLE),
            1,
            "the example design's button"
        );
        // The readouts show the requirement and its margin.
```

with:

```rust
        assert_eq!(count(CHANGED_DOT), 0, "no input is changed at the baseline");
        // The example design is an item of the Design menu, not on the first view.
        assert_eq!(count(crate::gui::OPEN_EXAMPLE), 0, "{texts:?}");
        assert_eq!(count(crate::gui::DESIGN_MENU), 1, "{texts:?}");
        // The readouts show the requirement and its margin.
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust

        fn number(&self, path: &str) -> f64 {
```

with:

```rust

        /// Opens the top bar's Design menu and clicks its item `item` (UX overhaul spec 4.2).
        fn menu(&mut self, item: &str) -> egui::FullOutput {
            self.click_text(DESIGN_MENU);
            self.click_text(item)
        }

        fn number(&self, path: &str) -> f64 {
```

Replace:

```rust
    #[test]
    fn the_details_toggle_starts_off_and_a_click_turns_it_on_and_off() {
```

with:

```rust
    #[test]
    fn the_top_bar_is_undo_redo_design_share_and_export_and_the_menu_holds_the_rest() {
        // UX overhaul spec 4.2: the reset, the example design, the file's save and load and
        // the details toggle in the Design menu; Share and Export on the bar; no Equation panel
        // button (decision C-11).
        use crate::gui::explorer::EQUATION_PANEL;
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
        let output = harness.frame(Vec::new());
        let bar = text_rect(&output, HEADING).expect("the title");
        let in_bar = |output: &egui::FullOutput, text: &str| {
            text_rect(output, text).is_some_and(|r| (r.center().y - bar.center().y).abs() < 20.0)
        };
        for text in [UNDO, REDO, DESIGN_MENU, SHARE, EXPORT] {
            assert!(in_bar(&output, text), "{text} on the bar");
        }
        let menu = [
            RESET_TO_START,
            OPEN_EXAMPLE,
            SAVE_DESIGN,
            LOAD_DESIGN,
            SHOW_DETAILS,
        ];
        for text in menu.iter().chain(&[EQUATION_PANEL, RESET_ALL]) {
            assert_eq!(count(&output, text), 0, "{text} not on the first view");
        }
        // Left to right: Undo, Redo, Design, Share, Export.
        let lefts: Vec<f32> = [UNDO, REDO, DESIGN_MENU, SHARE, EXPORT]
            .iter()
            .map(|t| text_rect(&output, t).unwrap().left())
            .collect();
        assert!(lefts.windows(2).all(|w| w[0] < w[1]), "{lefts:?}");
        // The menu, under its button, top to bottom.
        harness.click_text(DESIGN_MENU);
        let output = harness.frame(Vec::new());
        let tops: Vec<f32> = menu
            .iter()
            .map(|t| text_rect(&output, t).unwrap_or_else(|| panic!("{t}")).top())
            .collect();
        assert!(tops.windows(2).all(|w| w[0] < w[1]), "{tops:?}");
        assert!(tops[0] > bar.bottom(), "under the bar");
        // Opening the menu edits nothing.
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn the_top_bar_s_buttons_show_their_box_only_under_the_pointer() {
        // Stage 1 review S-3: a word at rest, a box while hovered.
        use crate::gui::test_support::flat_shapes;
        let boxed = |output: &egui::FullOutput, text: &str| {
            let at = text_rect(output, text).unwrap();
            flat_shapes(output).iter().any(|shape| match shape {
                // A button's box, not the bar's own fill behind it.
                egui::Shape::Rect(r) => {
                    r.rect.contains_rect(at) && r.rect.width() < 200.0 && r.fill.a() > 0
                }
                _ => false,
            })
        };
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        for text in [DESIGN_MENU, SHARE, EXPORT] {
            assert!(!boxed(&output, text), "{text} at rest");
        }
        let at = text_rect(&output, SHARE).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        assert!(boxed(&output, SHARE), "Share hovered");
        assert!(!boxed(&output, EXPORT), "Export not hovered");
    }

    #[test]
    fn the_details_toggle_starts_off_and_a_click_turns_it_on_and_off() {
```

Replace:

```rust
        assert!(!MagcouplingPanel::default().show_details);
        tooltips_at_once(&harness.ctx);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SHOW_DETAILS), 1);
```

with:

```rust
        assert!(!MagcouplingPanel::default().show_details);
        tooltips_at_once(&harness.ctx);
        // In the Design menu (decision C-12), not on the bar.
        assert_eq!(count(&harness.frame(Vec::new()), SHOW_DETAILS), 0);
        harness.click_text(DESIGN_MENU);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SHOW_DETAILS), 1);
```

Replace:

```rust
        // Its hover says what it shows.
        let at = text_rect(&output, SHOW_DETAILS).unwrap().center();
```

with:

```rust
        // Its hover says what it shows.
        // egui holds a tooltip back while the click that opened the menu is newer than the
        // pointer's last move: the move comes a second later.
        let at = text_rect(&output, SHOW_DETAILS).unwrap().center();
```

Replace:

```rust
        let at = text_rect(&output, SHOW_DETAILS).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
```

with:

```rust
        let at = text_rect(&output, SHOW_DETAILS).unwrap().center();
        harness.frame_after(1.0, vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        assert!(harness.panel.show_details);
        harness.click_text(SHOW_DETAILS);
        assert!(!harness.panel.show_details);
```

with:

```rust
        assert!(harness.panel.show_details);
        // The click closed the menu.
        assert_eq!(count(&harness.frame(Vec::new()), SHOW_DETAILS), 0);
        harness.menu(SHOW_DETAILS);
        assert!(!harness.panel.show_details);
```

Replace:

```rust
        let link = harness.panel.share_link();
        harness.click_text(SHOW_DETAILS);
        assert!(harness.panel.show_details);
```

with:

```rust
        let link = harness.panel.share_link();
        harness.menu(SHOW_DETAILS);
        assert!(harness.panel.show_details);
```

Replace:

```rust
        kept(&mut harness, "undo");
        harness.click_text(RESET_ALL);
        kept(&mut harness, "reset all");
```

with:

```rust
        kept(&mut harness, "undo");
        harness.menu(RESET_ALL);
        kept(&mut harness, "reset all");
```

Replace:

```rust
        harness.frame(key_tap(egui::Key::ArrowRight));
        // The header's button, then the results table's.
        harness.click_text(EXPORT_SPREADSHEET);
        harness.click_text(CentreView::Results.label());
```

with:

```rust
        harness.frame(key_tap(egui::Key::ArrowRight));
        // The top bar's button, then the results table's.
        harness.click_text(EXPORT);
        harness.click_text(CentreView::Results.label());
```

Replace:

```rust
        harness.click_text(InputOrder::Workbook.label());
        harness.click_text(EXPORT_SPREADSHEET);
        let books = queued_spreadsheets(&mut harness);
```

with:

```rust
        harness.click_text(InputOrder::Workbook.label());
        harness.click_text(EXPORT);
        let books = queued_spreadsheets(&mut harness);
```

Replace:

```rust
        harness.size();
        harness.click_text(EXPORT_SPREADSHEET);
        let books = queued_spreadsheets(&mut harness);
```

with:

```rust
        harness.size();
        harness.click_text(EXPORT);
        let books = queued_spreadsheets(&mut harness);
```

Replace:

```rust

        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
```

with:

```rust

        harness.menu(RESET_ALL);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
```

Replace:

```rust
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.design(), Design::default());
```

with:

```rust
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.menu(RESET_ALL);
        assert_eq!(harness.panel.design(), Design::default());
```

Replace:

```rust
        assert_ne!(harness.panel.design(), start);
        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.design(), start);
```

with:

```rust
        assert_ne!(harness.panel.design(), start);
        harness.menu(RESET_TO_START);
        assert_eq!(harness.panel.design(), start);
```

Replace:

```rust
        assert_eq!(count(&output, "Your design: passes all 10 checks"), 1);
        harness.click_text(RESET_ALL);
        let output = harness.frame(Vec::new());
```

with:

```rust
        assert_eq!(count(&output, "Your design: passes all 10 checks"), 1);
        harness.menu(RESET_TO_START);
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        let mut harness = Harness::with_baseline(baseline.clone(), SCREEN);
        harness.click_text(OPEN_EXAMPLE);
        assert_eq!(harness.panel.design(), Design::default());
```

with:

```rust
        let mut harness = Harness::with_baseline(baseline.clone(), SCREEN);
        harness.menu(OPEN_EXAMPLE);
        assert_eq!(harness.panel.design(), Design::default());
```

Replace:

```rust
        assert_eq!(harness.panel.design(), Design::default());
        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.design(), design_at(&baseline));
```

with:

```rust
        assert_eq!(harness.panel.design(), Design::default());
        harness.menu(RESET_TO_START);
        assert_eq!(harness.panel.design(), design_at(&baseline));
```

Replace:

```rust
            EXAMPLE_OPENED,
        ] {
```

with:

```rust
            EXAMPLE_OPENED,
            DESIGN_MENU,
            RESET_TO_START,
            SHARE,
            EXPORT,
        ] {
```

Replace:

```rust
    fn a_panel_whose_baseline_is_the_spreadsheet_s_design_has_no_example_button() {
        // Reset all already opens it there, and the header stays as it was (spec 8.1, test 2).
        let mut harness = Harness::new();
```

with:

```rust
    fn a_panel_whose_baseline_is_the_spreadsheet_s_design_has_no_example_button() {
        // Reset all already opens it there, and the Design menu keeps the reset's old name
        // (spec 8.1, test 2).
        let mut harness = Harness::new();
```

Replace:

```rust
        let mut harness = Harness::new();
        assert_eq!(count(&harness.frame(Vec::new()), OPEN_EXAMPLE), 0);
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
```

with:

```rust
        let mut harness = Harness::new();
        harness.click_text(DESIGN_MENU);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, OPEN_EXAMPLE), 0);
        assert_eq!(
            (count(&output, RESET_ALL), count(&output, RESET_TO_START)),
            (1, 0)
        );
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
```

Replace:

```rust
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
        assert_eq!(count(&harness.frame(Vec::new()), OPEN_EXAMPLE), 1);
    }
```

with:

```rust
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
        harness.click_text(DESIGN_MENU);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, OPEN_EXAMPLE), 1);
        assert_eq!(
            (count(&output, RESET_ALL), count(&output, RESET_TO_START)),
            (0, 1)
        );
    }
```

Replace:

```rust
        harness.click_text(REDO);
        harness.click_text(RESET_ALL);
        harness
```

with:

```rust
        harness.click_text(REDO);
        harness.menu(RESET_TO_START);
        harness
```

Replace:

```rust
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.click_text(SAVE_DESIGN);
        harness.click_text(LOAD_DESIGN);
        assert_eq!(
```

with:

```rust
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.menu(SAVE_DESIGN);
        harness.menu(LOAD_DESIGN);
        assert_eq!(
```

Replace:

```rust
            .set_share_base("http://localhost:8080/tools/magcoupler/");
        let output = harness.click_text(COPY_SHARE_LINK);
        let copied: Vec<&String> = output
```

with:

```rust
            .set_share_base("http://localhost:8080/tools/magcoupler/");
        let output = harness.click_text(SHARE);
        let copied: Vec<&String> = output
```

Replace:

```rust
        texts.extend([SHOW_DETAILS, SHOW_DETAILS_HINT].map(str::to_owned));
        texts.push(crate::gui::dashboard::MARKER_LEGEND.to_owned());
```

with:

```rust
        texts.extend([SHOW_DETAILS, SHOW_DETAILS_HINT].map(str::to_owned));
        texts.extend([RESET_ALL, RESET_TO_START, SAVE_DESIGN, LOAD_DESIGN].map(str::to_owned));
        texts.push(crate::gui::dashboard::MARKER_LEGEND.to_owned());
```

Replace:

```rust
    #[test]
    fn the_equation_panel_starts_closed_and_its_header_button_toggles_it() {
        use crate::gui::explorer::{CLOSE, EMPTY_TEXT};
        let mut harness = Harness::new();
```

with:

```rust
    #[test]
    fn the_equation_panel_starts_closed_with_no_button_on_the_bar_and_a_value_opens_it() {
        // Decision C-11: the Equation panel opens from a value (a click on a readout, an Explain
        // entry's Equation link), not from the top bar; its Close button closes it.
        use crate::gui::explorer::{CLOSE, EQUATION_PANEL};
        let mut harness = Harness::new();
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EMPTY_TEXT), 0);
        harness.click_text(EQUATION_PANEL);
        assert!(harness.panel.explorer.open);
        let output = harness.frame(Vec::new());
```

with:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EQUATION_PANEL), 0);
        harness.panel.explorer.open_path("model.pullout_Nm");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EMPTY_TEXT), 1);
        harness.click_text(CLOSE);
```

with:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EQUATION_PANEL), 1);
        harness.click_text(CLOSE);
```

Replace:

```rust
        // Decision M43-12: a leaf term frames its row while its equation is shown; closing
        // the panel, by its Close button or the header's toggle, takes the frame away.
        use crate::gui::explorer::CLOSE;
```

with:

```rust
        // Decision M43-12: a leaf term frames its row while its equation is shown; closing
        // the panel, by its Close button (the only one: decision C-11), takes the frame away.
        use crate::gui::explorer::CLOSE;
```

Replace:

```rust
        use crate::gui::explorer::CLOSE;
        for close in [CLOSE, EQUATION_PANEL] {
            let mut harness = Harness::new();
```

with:

```rust
        use crate::gui::explorer::CLOSE;
        {
            let close = CLOSE;
            let mut harness = Harness::new();
```

Replace:

```rust
            // Shown again, the panel keeps its equation but frames no row.
            harness.click_text(EQUATION_PANEL);
            assert!(harness.panel.explorer.open);
            assert_eq!(
```

with:

```rust
            // Shown again, the panel keeps its equation but frames no row.
            harness.panel.explorer.open = true;
            assert_eq!(
```

Replace:

```rust
        use crate::gui::explorer::{NEXT, PREVIOUS, START_HERE_BUTTON, STOP, start_here_text};
        let mut harness = Harness::new();
```

with:

```rust
        use crate::gui::explorer::{NEXT, PREVIOUS, START_HERE_BUTTON, STOP, start_here_text};
        // The Explain panel's orientation starts the walk.
        let mut harness = Harness::new();
```

Replace:

```rust
        let mut harness = Harness::new();
        harness.click_text(EQUATION_PANEL);
        harness.click_text(START_HERE_BUTTON);
```

with:

```rust
        let mut harness = Harness::new();
        harness.click_text(START_HERE_BUTTON);
```

Replace:

```rust
        // Reset all: the example again.
        harness.click_text(RESET_ALL);
        let output = harness.frame(Vec::new());
```

with:

```rust
        // Reset all: the example again.
        harness.menu(RESET_ALL);
        let output = harness.frame(Vec::new());
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib panel:: 2>&1 | tail -n 15
  ```

  Expected: the build fails with 1 errors, the first "unresolved import `panel::DESIGN_MENU`": the green step adds what the tests call.

- [ ] **Step 3 (green): the bar and the menu.**

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
use crate::gui::explain::{Asked, Selection, explain_ui};
use crate::gui::explorer::{
    EQUATION_PANEL, Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights,
};
use crate::gui::format::search_needle;
```

with:

```rust
use crate::gui::explain::{Asked, Selection, explain_ui};
use crate::gui::explorer::{Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights};
use crate::gui::format::search_needle;
```

Replace:

```rust

/// The label of the button that restores the default design.
pub const RESET_ALL: &str = "Reset all";
```

with:

```rust

/// The top bar's menu of the actions on the whole design (UX overhaul spec 4.2): the reset,
/// the example design, the file's save and load, and the details toggle (decision C-12).
pub const DESIGN_MENU: &str = "Design";

/// The Design menu's item that restores the baseline on a panel whose baseline is the
/// spreadsheet's design (a host's, or a test's, panel built with `MagcouplingPanel::new`).
pub const RESET_ALL: &str = "Reset all";
```

Replace:

```rust

/// The header button that opens the spreadsheet's design (UX overhaul spec 7.1), shown on a
/// panel whose baseline is another design: there Reset all no longer reaches it.
pub const OPEN_EXAMPLE: &str = "Open the example design";
```

with:

```rust

/// The same item on a panel whose baseline is another design: the app's, which opens on the
/// starting design (UX overhaul spec 4.2 and 7.1).
pub const RESET_TO_START: &str = "Reset to the starting design";

/// The Design menu's item that opens the spreadsheet's design (UX overhaul spec 7.1), shown on
/// a panel whose baseline is another design: there the reset no longer reaches it.
pub const OPEN_EXAMPLE: &str = "Open the example design";
```

Replace:

```rust
pub const LOAD_DESIGN: &str = "Load design";
pub const COPY_SHARE_LINK: &str = "Copy share link";

```

with:

```rust
pub const LOAD_DESIGN: &str = "Load design";

/// The top bar's button that copies the link of the design shown (UX overhaul spec 4.2: the
/// action of the button it replaces, "Copy share link").
pub const SHARE: &str = "Share";

```

Replace:

```rust

/// The header's spreadsheet button (decision X-8; the results table has its own, beside the CSV
/// and JSON exports).
pub const EXPORT_SPREADSHEET: &str = "Export spreadsheet";

```

with:

```rust

/// The top bar's spreadsheet button (decision X-8; UX overhaul spec 4.2 shortens its name from
/// "Export spreadsheet"; the results table has its own, beside the CSV and JSON exports).
pub const EXPORT: &str = "Export";

```

Replace:

```rust
            if ui
                .add_enabled(can_undo, egui::Button::new(UNDO))
                .on_hover_text("Ctrl+Z")
```

with:

```rust
            if ui
                .add_enabled(can_undo, bar_button(UNDO))
                .on_hover_text("Ctrl+Z")
```

Replace:

```rust
            if ui
                .add_enabled(can_redo, egui::Button::new(REDO))
                .on_hover_text("Ctrl+Shift+Z or Ctrl+Y")
```

with:

```rust
            if ui
                .add_enabled(can_redo, bar_button(REDO))
                .on_hover_text("Ctrl+Shift+Z or Ctrl+Y")
```

Replace:

```rust
            }
            if ui.button(RESET_ALL).clicked() {
                self.reset();
                self.status = Some("Default design restored".to_owned());
            }
            // Where the baseline is the spreadsheet's design, Reset all already opens it.
            if self.baseline != *workbook_defaults()
                && ui
                    .button(OPEN_EXAMPLE)
                    .on_hover_text(EXAMPLE_DESIGN_HOVER)
                    .clicked()
            {
                self.open_example_design();
                self.status = Some(EXAMPLE_OPENED.to_owned());
            }
            ui.separator();
```

with:

```rust
            }
            ui.separator();
```

Replace:

```rust
            ui.separator();
            if ui.button(SAVE_DESIGN).clicked() {
                self.requests.push(PanelRequest::SaveFile {
                    file_name: DESIGN_FILE_NAME.to_owned(),
                    mime: "application/json",
                    contents: design_to_json(&design).into_bytes(),
                });
            }
            if ui.button(LOAD_DESIGN).clicked() {
                self.requests.push(PanelRequest::OpenDesign);
            }
            if ui
```

with:

```rust
            ui.separator();
            egui::containers::menu::MenuButton::from_button(bar_button(DESIGN_MENU))
                .ui(ui, |ui| self.design_menu_ui(ui, &design));
            if ui
```

Replace:

```rust
            if ui
                .button(COPY_SHARE_LINK)
                .on_hover_text("A link that opens this design, sizing state included")
```

with:

```rust
            if ui
                .add(bar_button(SHARE))
                .on_hover_text("A link that opens this design, sizing state included")
```

Replace:

```rust
            if ui
                .button(EXPORT_SPREADSHEET)
                .on_hover_text(
```

with:

```rust
            if ui
                .add(bar_button(EXPORT))
                .on_hover_text(
```

Replace:

```rust
            }
            ui.separator();
            if ui
                .selectable_label(self.explorer.open, EQUATION_PANEL)
                .on_hover_text("Show or hide the equation of the value clicked")
                .clicked()
            {
                if self.explorer.open {
                    self.explorer.close();
                } else {
                    self.explorer.open = true;
                }
            }
            ui.separator();
            // The frame's readouts were set up before the header: the next frame draws every
            // view with the new setting.
            if ui
                .checkbox(&mut self.show_details, SHOW_DETAILS)
                .on_hover_text(SHOW_DETAILS_HINT)
                .changed()
            {
                ui.ctx().request_repaint();
            }
        });
```

with:

```rust
            }
        });
```

Replace:

```rust

    /// The left side: the tabs of the design inputs and the assumptions, then the view chosen,
```

with:

```rust

    /// The Design menu (UX overhaul spec 4.2): the reset to the baseline, the example design
    /// where the baseline is another, the file's save and load of `design`, and the details
    /// toggle (decision C-12). A click on an item closes the menu.
    fn design_menu_ui(&mut self, ui: &mut egui::Ui, design: &Design) {
        if ui.button(self.reset_label()).clicked() {
            self.reset();
            self.status = Some("Default design restored".to_owned());
        }
        // Where the baseline is the spreadsheet's design, the reset already opens it.
        if self.baseline != *workbook_defaults()
            && ui
                .button(OPEN_EXAMPLE)
                .on_hover_text(EXAMPLE_DESIGN_HOVER)
                .clicked()
        {
            self.open_example_design();
            self.status = Some(EXAMPLE_OPENED.to_owned());
        }
        ui.separator();
        if ui.button(SAVE_DESIGN).clicked() {
            self.requests.push(PanelRequest::SaveFile {
                file_name: DESIGN_FILE_NAME.to_owned(),
                mime: "application/json",
                contents: design_to_json(design).into_bytes(),
            });
        }
        if ui.button(LOAD_DESIGN).clicked() {
            self.requests.push(PanelRequest::OpenDesign);
        }
        ui.separator();
        // The frame's readouts were set up before the header: the next frame draws every
        // view with the new setting.
        if ui
            .checkbox(&mut self.show_details, SHOW_DETAILS)
            .on_hover_text(SHOW_DETAILS_HINT)
            .changed()
        {
            ui.ctx().request_repaint();
        }
    }

    /// The Design menu's reset: [`RESET_TO_START`] where the baseline is not the spreadsheet's
    /// design, [`RESET_ALL`] where it is (there the starting design and the example design are
    /// one).
    fn reset_label(&self) -> &'static str {
        if self.baseline == *workbook_defaults() {
            RESET_ALL
        } else {
            RESET_TO_START
        }
    }

    /// The left side: the tabs of the design inputs and the assumptions, then the view chosen,
```

Replace:

```rust
    })
}
```

with:

```rust
    })
}

/// A button of the top bar: its words alone at rest, its box while the pointer is on it (stage 1
/// review S-3; UX overhaul spec 4.1's bar of words).
fn bar_button(text: &str) -> egui::Button<'_> {
    egui::Button::new(text).frame_when_inactive(false)
}
```

- [ ] **Step 4: Run the checks.** Expected: `test result: ok. 1073 passed; 0 failed; 6 ignored`; clippy clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** the three files: `feat(magcoupler): the top bar is Undo, Redo, Design, Share and Export; the Design menu holds the reset, the example design, the file and the details toggle (BL-085 T2)`.

### Task 3: The notes in the Explain panel

**Model:** `sonnet`.

`Selection::Note` and `Selection::NoteList`; Learn on the bar; a note's entry; every note link selects its note; the lesson line under an open equation replaces the Explain checkbox; the Equation panel's note view goes; each Start here step selects its note; the first view's word ceiling falls to 430; the lesson line is drawn whole at every walk step.

**Files:**
- Modify: `magcoupling-rs/src/gui/explain.rs`, `magcoupling-rs/src/gui/explorer.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/dashboard.rs` (two doc lines), `magcoupling-rs/src/gui/readouts.rs` (one doc line), `magcoupling-rs/src/gui/test_support.rs`, `magcoupling-rs/src/app.rs`

**Interfaces:**
- Produces: `explain::{LEARN, shown_notes}`, `Selection::Note(&'static str)`, `Selection::NoteList`, `Shown::held`; `explorer::{explains_ui, lesson_ui (private)}`; `explorer::crumb` visible to the crate; `Explorer::start(step) -> Option<&'static str>`; `Explorer::follow_note_link(link) -> Option<&'static str>`; `explorer_ui(..) -> Option<&'static str>` without its `held` argument.
- Removes: `explorer::EXPLAIN`, `Explorer::explain`, `Explorer::open_note`, `Explorer::note`, `explorer::note_alone_ui`.

- [ ] **Step 1 (red): the notes' tests.**

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
    #[test]
    fn the_first_view_draws_at_most_four_hundred_and_fifty_words_at_1280_by_800() {
        // UX overhaul spec 2.1 (goal G-1) and 8.2, test 8, measured as the inventory did: the
```

with:

```rust
    #[test]
    fn the_first_view_draws_at_most_four_hundred_and_thirty_words_at_1280_by_800() {
        // UX overhaul spec 2.1 (goal G-1) and 8.2, test 8, measured as the inventory did: the
```

Replace:

```rust
        // words the opening screen shows. 608 before this stage (the dashboard's rows, the
        // torque readouts and the welcome strip); 443 after it, the callout list under the
        // drawing kept until its tags are selections (decision D-19). The design's 250 needs
        // the top bar, the Size column and the callout list of the stage's second part, which
        // lowers this ceiling.
        use crate::gui::test_support::words_shown;
```

with:

```rust
        // words the opening screen shows. 608 before this stage (the dashboard's rows, the
        // torque readouts and the welcome strip); 443 after its part 1, the callout list under
        // the drawing kept until its tags are selections (decision D-19); 427 after part 2a's
        // top bar. The design's 250 needs the Size column and the callout list of part 2b,
        // which lowers this ceiling.
        use crate::gui::test_support::words_shown;
```

Replace:

```rust
        let words = words_shown(&output, egui::Rect::from_min_size(egui::Pos2::ZERO, size));
        assert!(words <= 450, "{words} words on the first view");
    }
```

with:

```rust
        let words = words_shown(&output, egui::Rect::from_min_size(egui::Pos2::ZERO, size));
        assert!(words <= 430, "{words} words on the first view");
    }
```

Replace:

```rust
    #[test]
    fn the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font() {
```

with:

```rust
    #[test]
    fn every_start_here_step_s_lesson_line_is_drawn_whole_in_the_app_s_own_font() {
        // Plan review UXP-1 (BL-085): the lesson line under the open equation ("Full lesson:"
        // and the note's title, a link) ran past the Equation panel's edge at 1024 x 700 on
        // the long titles (steps 3, 7 and 12). Each step of the walk, at the three window
        // sizes, in the app's theme and font: no text of the line is cut.
        use crate::engine::explain::notes::START_HERE;
        use crate::gui::explain::FULL_LESSON;
        use crate::gui::explorer::{NEXT, START_HERE_BUTTON, reviewed_note};
        use crate::gui::test_support::{cut_texts, primary_button, text_center};
        use crate::gui::typeset::glyph_safe;
        for size in [WINDOWS[1], WINDOWS[0], egui::vec2(1920.0, 1080.0)] {
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            let click = |app: &mut MagcouplingApp, text: &str| {
                let output = frame_on(&ctx, app, size, Vec::new());
                let at = text_center(&output, text);
                frame_on(&ctx, app, size, vec![egui::Event::PointerMoved(at)]);
                for pressed in [true, false] {
                    frame_on(&ctx, app, size, vec![primary_button(at, pressed)]);
                }
                frame_on(&ctx, app, size, vec![egui::Event::PointerGone]);
                frame_on(&ctx, app, size, Vec::new());
                frame_on(&ctx, app, size, Vec::new())
            };
            frame_on(&ctx, &mut app, size, Vec::new());
            let mut output = click(&mut app, START_HERE_BUTTON);
            for (step, &(id, _)) in START_HERE.iter().enumerate() {
                if step > 0 {
                    output = click(&mut app, NEXT);
                }
                let title = glyph_safe(reviewed_note(id).expect("a reviewed note").title);
                let line = [FULL_LESSON, title.as_str()];
                let cut: Vec<String> = cut_texts(&output, screen)
                    .into_iter()
                    .filter(|cut| {
                        line.iter()
                            .any(|text| cut.starts_with(&format!("{text:?} by")))
                    })
                    .collect();
                assert!(cut.is_empty(), "{size:?}, step {}: cut: {cut:#?}", step + 1);
            }
        }
    }

    #[test]
    fn the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font() {
```

**`magcoupling-rs/src/gui/explain.rs`**

Replace:

```rust
                    consequences: None,
                    auto: None,
                };
                asked = explain_ui(ui, selection, &shown, &mut readouts);
```

with:

```rust
                    consequences: None,
                    auto: None,
                    held: &|_| None,
                };
                asked = explain_ui(ui, selection, &shown, &mut readouts);
```

Replace:

```rust
            EQUATION,
            crate::gui::checks::HEADING,
```

with:

```rust
            EQUATION,
            LEARN,
            crate::gui::checks::HEADING,
```

Replace:

```rust
    }

    #[test]
    fn about_this_tool_returns_to_the_orientation() {
```

with:

```rust
    }

    // ------------------------------------------------------------------ the notes

    #[test]
    fn learn_lists_every_shown_note_by_title_and_a_title_selects_it() {
        // UX overhaul spec 4.2: Learn lists the shown notes by title here; a title opens the
        // note here. The paused shaft-clamp section takes its note with it.
        use crate::engine::explain::notes::PAUSED_NOTE;
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let (output, asked, _) = frames(&ctx, &inputs, Selection::NoteList, false, vec![]);
        assert_eq!(asked, None);
        let mut want = vec![HEADING.to_owned(), ABOUT.to_owned(), LEARN.to_owned()];
        want.extend(shown_notes(false).map(|note| glyph_safe(note.title)));
        assert_eq!(drawn_texts(&output), want);
        // Every reviewed note, in the notes' order; while paused, all but the clamp's.
        let reviewed: Vec<&str> = NOTES
            .iter()
            .filter(|n| reviewed_note(n.id).is_some())
            .map(|n| n.id)
            .collect();
        assert_eq!(
            shown_notes(false).map(|n| n.id).collect::<Vec<_>>(),
            reviewed
        );
        let paused: Vec<&str> = shown_notes(true).map(|n| n.id).collect();
        assert!(!paused.contains(&PAUSED_NOTE));
        assert_eq!(paused.len(), reviewed.len() - 1);
        // A title selects its note.
        let first = shown_notes(false).next().unwrap();
        let (asked, events) = click(&inputs, Selection::NoteList, &glyph_safe(first.title));
        assert_eq!(asked, Some(Asked::Select(Selection::Note(first.id))));
        assert_eq!(events, ReadoutEvents::default());
    }

    #[test]
    fn a_note_s_entry_is_the_note_with_the_links_to_its_equations() {
        // Decision C-11: the note as the Equation panel drew it, after the links to the
        // equations it explains; an equation link opens it in the Equation panel and leaves
        // the selection; a change entry's input shows its row, as a check's fix does.
        use crate::gui::explorer::{EXPLAINS, WHERE_FROM, crumb};
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let note = reviewed_note("pullout_angle").unwrap();
        let selection = Selection::Note(note.id);
        let (output, asked, _) = frames(&ctx, &inputs, selection, false, vec![]);
        assert_eq!(asked, None);
        let texts = drawn_texts(&output);
        for text in [EXPLAINS, WHAT_TO_CHANGE, WHERE_FROM] {
            assert_eq!(count(&texts, text), 1, "{text} in {texts:?}");
        }
        assert_eq!(count(&texts, &glyph_safe(note.title)), 1, "{texts:?}");
        // The first equation it explains that has a record (a family's template has none).
        let equation = *note
            .equations
            .iter()
            .find(|p| crate::gui::readouts::registry().equation_for(p).is_some())
            .unwrap();
        let (asked, events) = click(&inputs, selection, &crumb(equation));
        assert_eq!(asked, Some(Asked::Equation(equation)));
        assert_eq!(events, ReadoutEvents::default());
        let input = note.change[0].input;
        let (asked, events) = click(&inputs, selection, &glyph_safe(label_for(input)));
        assert_eq!(asked, None);
        assert_eq!(events.focus, Some(input));
        // A note that is not reviewed, or no note: the heading and the way back alone.
        let (output, _, _) = frames(
            &ctx,
            &inputs,
            Selection::Note("no.such.note"),
            false,
            vec![],
        );
        assert_eq!(drawn_texts(&output), [HEADING, ABOUT]);
    }

    #[test]
    fn about_this_tool_returns_to_the_orientation() {
```

Replace:

```rust
                    auto: None,
                };
```

with:

```rust
                    auto: None,
                    held: &|_| None,
                };
```

Replace:

```rust
                        auto: None,
                    };
```

with:

```rust
                        auto: None,
                        held: &|_| None,
                    };
```

**`magcoupling-rs/src/gui/explorer.rs`**

Replace:

```rust
                false,
                &nothing_held,
                clamps::PAUSED,
```

with:

```rust
                false,
                clamps::PAUSED,
```

Replace:

```rust
    fn start_here_walks_the_suggested_order_and_a_click_leaves_it() {
        let mut explorer = Explorer::default();
```

with:

```rust
    fn start_here_walks_the_suggested_order_and_a_click_leaves_it() {
        // Each step opens its equation and names its note, which the panel shows in the
        // Explain panel (UX overhaul spec 5.5).
        let mut explorer = Explorer::default();
```

Replace:

```rust
        let mut explorer = Explorer::default();
        assert!(!explorer.explain, "Explain is off by default");
        explorer.start(0);
        assert_eq!(explorer.start_here(), Some(0));
```

with:

```rust
        let mut explorer = Explorer::default();
        assert_eq!(explorer.start(0), Some(START_HERE[0].0));
        assert_eq!(explorer.start_here(), Some(0));
```

Replace:

```rust
        assert_eq!(explorer.start_here(), Some(0));
        assert!(explorer.explain && explorer.open);
        assert_eq!(explorer.current(), Some(START_HERE[0].1));
```

with:

```rust
        assert_eq!(explorer.start_here(), Some(0));
        assert!(explorer.open);
        assert_eq!(explorer.current(), Some(START_HERE[0].1));
```

Replace:

```rust
        let last = START_HERE.len() - 1;
        explorer.start(last);
        explorer.start(last + 1);
        assert_eq!(
```

with:

```rust
        let last = START_HERE.len() - 1;
        assert_eq!(explorer.start(last), Some(START_HERE[last].0));
        assert_eq!(explorer.start(last + 1), None);
        assert_eq!(
```

Replace:

```rust
                    false,
                    &nothing_held,
                    paused,
```

with:

```rust
                    false,
                    paused,
```

Replace:

```rust
    #[test]
    fn a_note_opened_alone_replaces_the_trail_until_a_value_is_opened() {
        let mut explorer = Explorer::default();
```

with:

```rust
    #[test]
    fn the_lesson_line_names_the_open_equation_s_note_and_asks_for_it() {
        // Decision C-11: the note itself is the Explain panel's; under the equation the lesson
        // line names it after "Full lesson:", a link that returns it, and nothing of the note is
        // drawn here.
        use crate::engine::explain::notes::note_for;
        use crate::gui::test_support::{
            SCREEN, drawn_texts, primary_button, sized_frame, text_rect,
        };
        let path = "model.pullout_Nm";
        let note = note_for(path).expect("the pull-out's note");
        let (inputs, results) = design();
        let ctx = egui::Context::default();
        let mut explorer = Explorer::default();
        explorer.open_path(path);
        let frame = |events: Vec<egui::Event>, explorer: &mut Explorer| {
            let mut asked = None;
            let output = sized_frame(&ctx, SCREEN, events, |ui| {
                asked = explorer_ui(
                    ui,
                    explorer,
                    &inputs,
                    &results,
                    &DesignInputs::default(),
                    false,
                    clamps::PAUSED,
                );
            });
            (output, asked)
        };
        frame(Vec::new(), &mut explorer);
        let (output, asked) = frame(Vec::new(), &mut explorer);
        assert_eq!(asked, None);
        let texts = drawn_texts(&output);
        let title = glyph_safe(note.title);
        assert_eq!(
            texts.iter().filter(|t| **t == title).count(),
            1,
            "{texts:?}"
        );
        assert!(texts.iter().any(|t| t == FULL_LESSON), "{texts:?}");
        assert!(
            !texts.contains(&note_text_on(
                note.sentences[0],
                &NoteView {
                    inputs: &inputs,
                    results: &results,
                    held: &nothing_held,
                }
            )),
            "the note's sentences are the Explain panel's"
        );
        let at = text_rect(&output, &title).unwrap().center();
        frame(vec![egui::Event::PointerMoved(at)], &mut explorer);
        frame(vec![primary_button(at, true)], &mut explorer);
        let (_, asked) = frame(vec![primary_button(at, false)], &mut explorer);
        assert_eq!(asked, Some(note.id));
        assert_eq!(explorer.current(), Some(path), "the equation stays open");
    }

    #[test]
    fn a_note_link_is_the_explain_panel_s_and_leaves_the_open_equation() {
        // Decision C-11: a note asked for under an equation is returned for the Explain panel;
        // the equation stays open (decision C-25).
        let mut explorer = Explorer::default();
```

Replace:

```rust
        explorer.open_path("model.f_end");
        explorer.open_note("a5.low_saturation");
        assert_eq!(explorer.note(), Some("a5.low_saturation"));
        assert_eq!(explorer.current(), None);
        assert!(explorer.open);
```

with:

```rust
        explorer.open_path("model.f_end");
        assert_eq!(
            explorer.follow_note_link(NoteLink::Note("a5.low_saturation")),
            Some("a5.low_saturation")
        );
        assert_eq!(explorer.current(), Some("model.f_end"));
        assert!(explorer.open);
```

Replace:

```rust
        assert!(explorer.open);
        explorer.open_path("materials.circuit_backiron");
        assert_eq!(explorer.note(), None);
        explorer.open_note("a5.low_saturation");
        explorer.start(0);
        assert_eq!(explorer.note(), None);
    }
```

with:

```rust
        assert!(explorer.open);
    }
```

Replace:

```rust
                    show_details,
                    &nothing_held,
                    clamps::PAUSED,
```

with:

```rust
                    show_details,
                    clamps::PAUSED,
```

Replace:

```rust
                    clamps::PAUSED,
                )
            },
```

with:

```rust
                    clamps::PAUSED,
                );
            },
```

Replace:

```rust
                    false,
                    &nothing_held,
                    clamps::PAUSED,
```

with:

```rust
                    false,
                    clamps::PAUSED,
```

Replace:

```rust
    #[test]
    fn a_note_s_link_highlights_its_input_s_row_and_keeps_the_note_open() {
        let mut explorer = Explorer::default();
```

with:

```rust
    #[test]
    fn a_note_s_link_highlights_its_input_s_row_and_asks_for_no_note() {
        let mut explorer = Explorer::default();
```

Replace:

```rust
        let mut explorer = Explorer::default();
        explorer.open_note("a5.low_saturation");
        explorer.follow_note_link(NoteLink::Input("metal.cup_wall_corner_mm"));
        assert_eq!(
```

with:

```rust
        let mut explorer = Explorer::default();
        assert_eq!(
            explorer.follow_note_link(NoteLink::Input("metal.cup_wall_corner_mm")),
            None
        );
        assert_eq!(
```

Replace:

```rust
        );
        assert_eq!(explorer.note(), Some("a5.low_saturation"));
        assert!(explorer.open);
    }
```

with:

```rust
        );
    }
```

Replace:

```rust
    fn the_empty_text_and_explain_s_own_lines_name_the_buttons_in_plain_words() {
        // TC-29b: the empty panel names both ways to the notes by the buttons' own names; the
        // fallback lines read as plain words in egui's default fonts.
        assert!(EMPTY_TEXT.contains(START_HERE_BUTTON) && EMPTY_TEXT.contains(EXPLAIN));
        assert!(START_HERE_HINT.contains(START_HERE_BUTTON));
```

with:

```rust
    fn the_empty_text_and_explain_s_own_lines_name_the_buttons_in_plain_words() {
        // TC-29b: the empty panel names the walk through the notes by its button's own name
        // (the Explain toggle it also named went with decision C-11); the fallback lines read
        // as plain words in egui's default fonts.
        assert!(EMPTY_TEXT.contains(START_HERE_BUTTON));
        assert!(!EMPTY_TEXT.contains("Explain"), "{EMPTY_TEXT}");
        assert!(START_HERE_HINT.contains(START_HERE_BUTTON));
```

Replace:

```rust
    fn a_wrapped_change_entry_stays_inside_a_narrow_panel_and_its_label_links() {
        // Amendment A-25 T-6 (spec section 5, a narrow window): the pull-out note opened on its
        // own in a panel 400 points wide. Every change entry's input label is drawn inside the
        // panel, and a click on the last character of the inner magnet part's label, laid out in
```

with:

```rust
    fn a_wrapped_change_entry_stays_inside_a_narrow_panel_and_its_label_links() {
        // Amendment A-25 T-6 (spec section 5, a narrow window): the pull-out note drawn alone in
        // a panel 400 points wide. Every change entry's input label is drawn inside the
        // panel, and a click on the last character of the inner magnet part's label, laid out in
```

Replace:

```rust
            let output = sized_frame(&ctx, size, events, |ui| {
                link = note_alone_ui(ui, note, &view).1;
            });
```

with:

```rust
            let output = sized_frame(&ctx, size, events, |ui| {
                link = note_ui(ui, note, &view);
            });
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
        // Spec 8.2, test 14: with nothing selected, with a failing check's entry open and with
        // the Hot-low tile's (the torque card: its consequences are 30 compute_all runs), a
        // frame in which nothing changed runs no fix search and no results of its own.
        let mut harness = Harness::first_view();
        let hot_low = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
        let checks = [HOT_MIN_CHECK, GEARBOX_CHECK].map(|check| Some(Selection::Check(check)));
        for check in [None, checks[0], checks[1], Some(hot_low)] {
```

with:

```rust
        // Spec 8.2, test 14: with nothing selected, with a failing check's entry open, with
        // the Hot-low tile's (the torque card: its consequences are 30 compute_all runs), with
        // a note's entry (its diagram) and with the note list (plan review COV-1), a frame in
        // which nothing changed runs no fix search and no results of its own.
        use crate::engine::explain::notes::START_HERE;
        let mut harness = Harness::first_view();
        let hot_low = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
        let checks = [HOT_MIN_CHECK, GEARBOX_CHECK].map(|check| Some(Selection::Check(check)));
        let note = Some(Selection::Note(START_HERE[0].0));
        for check in [
            None,
            checks[0],
            checks[1],
            Some(hot_low),
            note,
            Some(Selection::NoteList),
        ] {
```

Replace:

```rust
        // Decision G-5 (spec 3.1, reversing M42-2) and UX overhaul spec 4.4: the first view is
        // the Size tab, its four tiles over the drawing; showing them edits nothing, and
        // Explain stays off (M43-2). The welcome strip is gone: the Explain panel's orientation
        // says what to do first.
        use crate::gui::guidance::{design_status, design_status_text};
```

with:

```rust
        // Decision G-5 (spec 3.1, reversing M42-2) and UX overhaul spec 4.4: the first view is
        // the Size tab, its four tiles over the drawing; showing them edits nothing, and the
        // Explain panel shows no note (M43-2). The welcome strip is gone: the Explain panel's
        // orientation says what to do first.
        use crate::gui::guidance::{design_status, design_status_text};
```

Replace:

```rust
        assert_eq!(panel.centre, CentreView::SizeAndTorque);
        assert!(!panel.explorer.explain);
        assert_eq!(
```

with:

```rust
        assert_eq!(panel.centre, CentreView::SizeAndTorque);
        assert_eq!(panel.selection, Selection::Nothing);
        assert_eq!(
```

Replace:

```rust
    #[test]
    fn the_top_bar_is_undo_redo_design_share_and_export_and_the_menu_holds_the_rest() {
        // UX overhaul spec 4.2: the reset, the example design, the file's save and load and
```

with:

```rust
    #[test]
    fn the_top_bar_is_undo_redo_design_share_export_and_learn_and_the_menu_holds_the_rest() {
        // UX overhaul spec 4.2: the reset, the example design, the file's save and load and
```

Replace:

```rust
        };
        for text in [UNDO, REDO, DESIGN_MENU, SHARE, EXPORT] {
            assert!(in_bar(&output, text), "{text} on the bar");
```

with:

```rust
        };
        for text in [UNDO, REDO, DESIGN_MENU, SHARE, EXPORT, LEARN] {
            assert!(in_bar(&output, text), "{text} on the bar");
```

Replace:

```rust
        }
        // Left to right: Undo, Redo, Design, Share, Export.
        let lefts: Vec<f32> = [UNDO, REDO, DESIGN_MENU, SHARE, EXPORT]
            .iter()
```

with:

```rust
        }
        // Left to right: Undo, Redo, Design, Share, Export, Learn.
        let lefts: Vec<f32> = [UNDO, REDO, DESIGN_MENU, SHARE, EXPORT, LEARN]
            .iter()
```

Replace:

```rust
        assert!(!boxed(&output, EXPORT), "Export not hovered");
    }
```

with:

```rust
        assert!(!boxed(&output, EXPORT), "Export not hovered");
    }

    #[test]
    fn learn_lists_the_notes_in_the_explain_panel_and_a_title_shows_its_note() {
        // UX overhaul spec 4.2: Learn, on the top bar, lists the shown notes by title in the
        // Explain panel; a title shows the note there; Esc returns the orientation (5.1).
        use crate::gui::explain::{ORIENTATION, shown_notes};
        use crate::gui::typeset::glyph_safe;
        let mut harness = Harness::on_screen(TALL);
        harness.click_text(LEARN);
        assert_eq!(harness.panel.selection, Selection::NoteList);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let left = right_column_left(&output);
        for note in shown_notes(harness.panel.clamps_paused) {
            let title = glyph_safe(note.title);
            assert!(
                text_rects(&output, &title).iter().any(|r| r.left() > left),
                "{title} in the Explain panel"
            );
        }
        let note = shown_notes(harness.panel.clamps_paused).nth(3).unwrap();
        harness.click_text(&glyph_safe(note.title));
        assert_eq!(harness.panel.selection, Selection::Note(note.id));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, ORIENTATION), 0);
        harness.frame(key_tap(egui::Key::Escape));
        assert_eq!(harness.panel.selection, Selection::Nothing);
        assert!(!harness.panel.explorer.open, "Learn opens no equation");
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }
```

Replace:

```rust
                .open_path("temperature.summary.governing_limit_C");
            harness.panel.explorer.explain = true;
            for view in CentreView::ALL {
```

with:

```rust
                .open_path("temperature.summary.governing_limit_C");
            harness.panel.selection = Selection::Note("br_temperature");
            for view in CentreView::ALL {
```

Replace:

```rust
    #[test]
    fn the_explain_toggle_shows_the_open_equation_s_note_and_follows_it() {
        use crate::engine::explain::notes::{note, note_for};
```

with:

```rust
    #[test]
    fn the_open_equation_s_lesson_line_shows_its_note_in_the_explain_panel() {
        // Decision C-11: the Equation panel's Explain toggle is gone. Under the open equation
        // its note's title follows "Full lesson:", and a click shows the note in the Explain
        // panel; the equation stays open (C-25). The line follows the equation.
        use crate::engine::explain::notes::{note, note_for};
```

Replace:

```rust
        use crate::engine::explain::notes::{note, note_for};
        use crate::gui::explorer::{
            EXPLAIN, NO_NOTE, NoteFallback, START_HERE_HINT, note_fallback,
        };
        let mut harness = Harness::on_screen(TALL);
```

with:

```rust
        use crate::engine::explain::notes::{note, note_for};
        use crate::gui::explain::FULL_LESSON;
        use crate::gui::explorer::{NO_NOTE, NoteFallback, START_HERE_HINT, note_fallback};
        let mut harness = Harness::on_screen(TALL);
```

Replace:

```rust
        let mut harness = Harness::on_screen(TALL);
        // A check's Explain entry names its note too: the Equation panel's note is one more.
        let pullout = note("pullout_angle").unwrap();
```

with:

```rust
        let mut harness = Harness::on_screen(TALL);
        let pullout = note("pullout_angle").unwrap();
```

Replace:

```rust
        let end = note("end_effect").unwrap();
        let output = harness.select("model.end_effect_check");
        let (pullout_links, end_links) = (count(&output, pullout.title), count(&output, end.title));
        assert_eq!(
            (pullout_links, end_links),
            (0, 1),
            "the end-effect check's entry links its note"
        );
        harness.panel.explorer.open_path("model.pullout_Nm");
```

with:

```rust
        let end = note("end_effect").unwrap();
        harness.panel.explorer.open_path("model.pullout_Nm");
```

Replace:

```rust
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, pullout.title),
            pullout_links,
            "hidden by default"
        );
        harness.click_text(EXPLAIN);
        assert!(harness.panel.explorer.explain);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, pullout.title), pullout_links + 1);
        assert_eq!(count(&output, pullout.sentences[0]), 1);
        assert!(diagram_points(&output) >= 1, "its torque-angle diagram");
        // Drill to f_end (its term row is below the note now): the note follows the equation.
        harness.panel.explorer.drill("model.f_end");
        harness.frame(Vec::new());
```

with:

```rust
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Explain"), 0, "no toggle");
        assert_eq!(count(&output, FULL_LESSON), 1);
        assert_eq!(count(&output, pullout.title), 1, "the lesson line alone");
        assert_eq!(count(&output, pullout.sentences[0]), 0);
        assert_eq!(
            diagram_points(&output),
            0,
            "no diagram in the Equation panel"
        );
        harness.click_text(pullout.title);
        assert_eq!(harness.panel.selection, Selection::Note(pullout.id));
        assert_eq!(harness.panel.explorer.current(), Some("model.pullout_Nm"));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, pullout.sentences[0]),
            1,
            "in the Explain panel"
        );
        assert!(diagram_points(&output) >= 1, "its torque-angle diagram");
        // Drill to f_end: the line names its note; the Explain panel keeps the pull-out's.
        harness.panel.explorer.drill("model.f_end");
        harness.frame(Vec::new());
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, end.title), end_links + 1);
        assert_eq!(count(&output, pullout.title), pullout_links);
        // A value no reviewed note explains, with nothing upstream that has a note or an
```

with:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, end.title), 1);
        assert_eq!(harness.panel.selection, Selection::Note(pullout.id));
        // A value no reviewed note explains, with nothing upstream that has a note or an
```

Replace:

```rust
        assert_eq!(count(&output, START_HERE_HINT), 1);
        // Off again: no note.
        harness.click_text(EXPLAIN);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, NO_NOTE), 0);
        assert_eq!(count(&output, START_HERE_HINT), 0);
    }
```

with:

```rust
        assert_eq!(count(&output, START_HERE_HINT), 1);
    }
```

Replace:

```rust
        use crate::gui::explorer::{NEXT, PREVIOUS, START_HERE_BUTTON, STOP, start_here_text};
        // The Explain panel's orientation starts the walk.
        let mut harness = Harness::new();
```

with:

```rust
        use crate::gui::explorer::{NEXT, PREVIOUS, START_HERE_BUTTON, STOP, start_here_text};
        // The Explain panel's orientation starts the walk; each step opens its equation and
        // selects its note, which the Explain panel shows (UX overhaul spec 5.5).
        let mut harness = Harness::new();
```

Replace:

```rust
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[0].1));
        let output = harness.frame(Vec::new());
```

with:

```rust
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[0].1));
        assert_eq!(harness.panel.selection, Selection::Note(START_HERE[0].0));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        assert_eq!(count(&output, &start_here_text(0)), 1);
        assert_eq!(count(&output, note(START_HERE[0].0).unwrap().title), 1);
        harness.click_text(NEXT);
```

with:

```rust
        assert_eq!(count(&output, &start_here_text(0)), 1);
        let first = note(START_HERE[0].0).unwrap();
        // The note's title heads its entry, and the lesson line under the equation names it.
        assert_eq!(count(&output, first.title), 2);
        assert_eq!(
            count(&output, first.sentences[0]),
            1,
            "the note in the Explain panel"
        );
        harness.click_text(NEXT);
```

Replace:

```rust
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[1].1));
        harness.click_text(NEXT);
```

with:

```rust
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[1].1));
        assert_eq!(harness.panel.selection, Selection::Note(START_HERE[1].0));
        harness.click_text(NEXT);
```

Replace:

```rust
        assert_eq!(harness.panel.explorer.start_here(), Some(1));
        harness.click_text(STOP);
```

with:

```rust
        assert_eq!(harness.panel.explorer.start_here(), Some(1));
        assert_eq!(harness.panel.selection, Selection::Note(START_HERE[1].0));
        harness.click_text(STOP);
```

Replace:

```rust
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[1].1));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
```

with:

```rust
        assert_eq!(harness.panel.explorer.current(), Some(START_HERE[1].1));
        assert_eq!(harness.panel.selection, Selection::Note(START_HERE[1].0));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
```

Replace:

```rust
        );
        harness.panel.explorer.open_note(id);
        harness.frame(Vec::new());
```

with:

```rust
        );
        harness.panel.selection = Selection::Note(id);
        harness.frame(Vec::new());
```

Replace:

```rust
        assert_eq!(
            harness.panel.explorer.note(),
            Some(id),
            "the note stays open"
```

with:

```rust
        assert_eq!(
            harness.panel.selection,
            Selection::Note(id),
            "the note stays open"
```

Replace:

```rust
    fn every_teaching_note_shows_with_glyphs_in_the_default_fonts() {
        // Each note opened alone, then its sources fold opened (it starts closed): every drawn
        // text, sources included, has glyphs.
        use crate::engine::explain::notes::NOTES;
```

with:

```rust
    fn every_teaching_note_shows_with_glyphs_in_the_default_fonts() {
        // Each note selected (the Explain panel draws it), then its sources fold opened (it
        // starts closed): every drawn text, sources included, has glyphs.
        use crate::engine::explain::notes::NOTES;
```

Replace:

```rust
        for note in NOTES {
            harness.panel.explorer.open_note(note.id);
            harness.frame(Vec::new());
```

with:

```rust
        for note in NOTES {
            harness.panel.selection = Selection::Note(note.id);
            harness.frame(Vec::new());
```

Replace:

```rust
        ));
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.note(), Some(rule.note_id));
        harness.frame(Vec::new());
```

with:

```rust
        ));
        assert_eq!(harness.panel.selection, Selection::Note(rule.note_id));
        harness.frame(Vec::new());
```

Replace:

```rust
        );
        // Its one equation is a link: the circuit in effect, opened like a value.
        harness.click_text("circuit");
```

with:

```rust
        );
        // Its one equation is a link: the circuit in effect, opened like a value in the
        // Equation panel; the note stays selected (decision C-25).
        harness.click_text("circuit");
```

Replace:

```rust
        harness.click_text("circuit");
        assert_eq!(harness.panel.explorer.note(), None);
        assert_eq!(
```

with:

```rust
        harness.click_text("circuit");
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.selection, Selection::Note(rule.note_id));
        assert_eq!(
```

Replace:

```rust
    #[test]
    fn a_check_s_why_link_opens_its_note_in_the_equation_panel() {
        use crate::gui::guidance::{CHECK_GUIDES, Learn};
```

with:

```rust
    #[test]
    fn a_check_s_lesson_link_shows_its_note_in_the_explain_panel() {
        use crate::gui::guidance::{CHECK_GUIDES, Learn};
```

Replace:

```rust
            harness.click_text(&crate::gui::typeset::glyph_safe(note.title));
            assert!(harness.panel.explorer.open, "{id}");
            assert_eq!(harness.panel.explorer.note(), Some(id));
            harness.frame(Vec::new());
```

with:

```rust
            harness.click_text(&crate::gui::typeset::glyph_safe(note.title));
            assert_eq!(harness.panel.selection, Selection::Note(id));
            harness.frame(Vec::new());
```

Replace:

```rust
        // link to the hot minimum check's note (the Hot-low tile's is its entry's lesson link).
        // Each opens its note on its own in the Equation panel.
        use crate::gui::dashboard::why_note_text;
```

with:

```rust
        // link to the hot minimum check's note (the Hot-low tile's is its entry's lesson link).
        // Each shows its note in the Explain panel (spec 5.2).
        use crate::gui::dashboard::why_note_text;
```

Replace:

```rust
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        for (id, links) in [(SIZE_TAB_NOTE, 3), (TORQUE_BAND_NOTE, 1)] {
```

with:

```rust
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let tile = harness.panel.selection;
        for (id, links) in [(SIZE_TAB_NOTE, 3), (TORQUE_BAND_NOTE, 1)] {
```

Replace:

```rust
            for index in 0..links {
                harness.panel.explorer = Explorer::default();
                let output = harness.frame(Vec::new());
```

with:

```rust
            for index in 0..links {
                harness.panel.selection = tile;
                let output = harness.frame(Vec::new());
```

Replace:

```rust
                harness.click(at[index]);
                assert!(harness.panel.explorer.open, "{text} #{index}");
                assert_eq!(harness.panel.explorer.note(), Some(id), "{text} #{index}");
            }
```

with:

```rust
                harness.click(at[index]);
                assert_eq!(
                    harness.panel.selection,
                    Selection::Note(id),
                    "{text} #{index}"
                );
            }
```

Replace:

```rust
        harness.click(links[0]);
        assert_eq!(harness.panel.explorer.note(), Some(BANNER_NOTE));
        assert_eq!(harness.panel.inputs(), &short_magnets());
```

with:

```rust
        harness.click(links[0]);
        assert_eq!(harness.panel.selection, Selection::Note(BANNER_NOTE));
        assert_eq!(harness.panel.inputs(), &short_magnets());
```

Replace:

```rust

    /// Opens `path` with Explain on and clicks its closest-note link: the one copy of `note`'s
    /// title on the lead's line, left of the dashboard (whose rows draw note titles in their Why
    /// lines), clicked on its text (it ends a wrapped row). A second copy there would make the
    /// click ambiguous: the helper says so rather than pass. The click opens `note` on its own
    /// and edits nothing.
    fn click_the_closest_note_link(path: &str, note: &crate::engine::explain::notes::Note) {
```

with:

```rust

    /// Opens `path` and clicks its closest-note link: the one copy of `note`'s title on the
    /// lead's line, left of the right column (whose entries draw note titles in their lesson
    /// links), clicked on its text (it ends a wrapped row). A second copy there would make the
    /// click ambiguous: the helper says so rather than pass. The click shows `note` in the
    /// Explain panel and edits nothing.
    fn click_the_closest_note_link(path: &str, note: &crate::engine::explain::notes::Note) {
```

Replace:

```rust
        let mut harness = Harness::on_screen(TALL);
        harness.panel.explorer.explain = true;
        harness.panel.explorer.open_path(path);
```

with:

```rust
        let mut harness = Harness::on_screen(TALL);
        harness.panel.explorer.open_path(path);
```

Replace:

```rust
        harness.click(beside[0]);
        assert_eq!(harness.panel.explorer.note(), Some(note.id), "{path}");
        assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{path}");
```

with:

```rust
        harness.click(beside[0]);
        assert_eq!(harness.panel.selection, Selection::Note(note.id), "{path}");
        assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{path}");
```

Replace:

```rust
    fn explain_offers_the_gearbox_row_its_why_note_and_the_link_opens_it() {
        // BL-059: the gearbox check has no equation and no note of its own. With Explain on,
        // its row offers the note its Why link leads to, the torque band (a row's own link comes
        // first), and the link beside the lead opens it on its own. Its "no note" and lead counts
        // are the dashboard loop's (`explain_never_shows_no_teaching_note_on_a_dashboard_row`).
```

with:

```rust
    fn explain_offers_the_gearbox_row_its_why_note_and_the_link_opens_it() {
        // BL-059: the gearbox check has no equation and no note of its own. Its row offers
        // the note its Why link leads to, the torque band (a row's own link comes
        // first), and the link beside the lead shows it in the Explain panel. Its "no note" and lead counts
        // are the dashboard loop's (`explain_never_shows_no_teaching_note_on_a_dashboard_row`).
```

Replace:

```rust
    fn explain_never_shows_no_teaching_note_on_a_dashboard_row() {
        // Spec section 6 test 11: each dashboard row opened with Explain on shows its note, the
        // closest note, an assumption's rationale, or its own reason with where the notes are.
```

with:

```rust
    fn explain_never_shows_no_teaching_note_on_a_dashboard_row() {
        // Spec section 6 test 11: each dashboard row opened shows its note's lesson line, the
        // closest note, an assumption's rationale, or its own reason with where the notes are.
```

Replace:

```rust
        let mut harness = Harness::on_screen(TALL);
        harness.panel.explorer.explain = true;
        for &(path, _) in &DASHBOARD {
```

with:

```rust
        let mut harness = Harness::on_screen(TALL);
        for &(path, _) in &DASHBOARD {
```

Replace:

```rust
        // TC-29a: a dashboard row whose equation has no note and whose Why link leads to none
        // offers the closest note upstream; its link opens it on its own.
        use crate::engine::explain::notes::note_for;
```

with:

```rust
        // TC-29a: a dashboard row whose equation has no note and whose Why link leads to none
        // offers the closest note upstream; its link shows it in the Explain panel.
        use crate::engine::explain::notes::note_for;
```

Replace:

```rust

    /// Clicks the note link at `at` and checks that the click opened note `id` in the explorer
    /// and edited no input. The harness must hold the default design.
    fn click_opens_note(harness: &mut Harness, at: egui::Pos2, id: &str) {
        harness.click(at);
```

with:

```rust

    /// Clicks the note link at `at` and checks that the click selected note `id` (the Explain
    /// panel shows it) and edited no input. The harness must hold the default design.
    fn click_opens_note(harness: &mut Harness, at: egui::Pos2, id: &'static str) {
        harness.click(at);
```

Replace:

```rust
        harness.click(at);
        assert!(harness.panel.explorer.open, "{id}");
        assert_eq!(harness.panel.explorer.note(), Some(id));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{id}");
```

with:

```rust
        harness.click(at);
        assert_eq!(harness.panel.selection, Selection::Note(id));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{id}");
```

**`magcoupling-rs/src/gui/test_support.rs`**

Replace:

```rust
            auto: None,
        };
```

with:

```rust
            auto: None,
            held: &|_| None,
        };
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 15
  ```

  Expected: the build fails with 54 errors, the first "unresolved import `crate::gui::explain::shown_notes`": the green step adds what the tests call. (The fix wave's walk test, `every_start_here_step_s_lesson_line_is_drawn_whole_in_the_app_s_own_font`, was red on the generated green: "Back iron: why steel behind the magnets adds torque" cut by 40 points at 1024 x 700, step 3.)

- [ ] **Step 3 (green): the notes, Learn, the lesson line, the walk.**

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
//! language, colour-coded, linked to their teaching note"): each rule's text in its severity's
//! colour (a warning red, a caution amber) with a link that opens its reviewed note in the
//! Equation panel ([`warning_lines`], decision M43-8).
//!
```

with:

```rust
//! language, colour-coded, linked to their teaching note"): each rule's text in its severity's
//! colour (a warning red, a caution amber) with a link that shows its reviewed note in the
//! Explain panel ([`warning_lines`], decision M43-8; UX overhaul decision C-11).
//!
```

Replace:

```rust

/// The lesson link of an Explain entry: a reviewed note's title (it asks for the note in the
/// Equation panel; a note not reviewed never shows), an assumption's label (it asks for its
/// input's row; the rationale is its hover text), or [`NO_NOTE_YET`] with the reason as its
```

with:

```rust

/// The lesson link of an Explain entry: a reviewed note's title (it asks for the note, which the
/// Explain panel then shows; a note not reviewed never shows), an assumption's label (it asks for its
/// input's row; the rationale is its hover text), or [`NO_NOTE_YET`] with the reason as its
```

**`magcoupling-rs/src/gui/explain.rs`**

Replace:

```rust
//!
//! The selection is view state: in no design file, share link or undo step.
```

with:

```rust
//!
//! A teaching note shows here too (decision C-11): the top bar's [`LEARN`] lists every note
//! shown by its title, and a title, a lesson link anywhere on the page or a step of the Start
//! here walk (spec 5.5) selects the note. Its entry is the links to the equations it explains,
//! then the note as the Equation panel drew it ([`note_ui`]).
//!
//! The selection is view state: in no design file, share link or undo step.
```

Replace:

```rust

use crate::engine::sizing::FreeVariable;
```

with:

```rust

use crate::engine::explain::notes::Note;
use crate::engine::explain::notes::{NOTES, shown_when};
use crate::engine::sizing::FreeVariable;
```

Replace:

```rust
};
use crate::gui::explorer::{START_HERE_BUTTON, WATCH_OUT, WHAT_TO_CHANGE, start_here_hover};
use crate::gui::guidance::{
```

with:

```rust
};
use crate::gui::explorer::{
    NoteLink, NoteView, START_HERE_BUTTON, WATCH_OUT, WHAT_TO_CHANGE, explains_ui, note_ui,
    reviewed_note, start_here_hover,
};
use crate::gui::guidance::{
```

Replace:

```rust
use crate::gui::torque_card::CARD_HEADING;
use crate::{DesignInputs, DesignResults};
```

with:

```rust
use crate::gui::torque_card::CARD_HEADING;
use crate::gui::typeset::glyph_safe;
use crate::{DesignInputs, DesignResults};
```

Replace:

```rust

/// What the Explain panel shows (spec 5.1).
```

with:

```rust

/// The top bar's button that lists the teaching notes here, and the list's heading (UX
/// overhaul spec 4.2).
pub const LEARN: &str = "Learn";

/// What the Explain panel shows (spec 5.1).
```

Replace:

```rust
    Result(&'static str),
}
```

with:

```rust
    Result(&'static str),
    /// A teaching note, by its id.
    Note(&'static str),
    /// The list of the teaching notes ([`LEARN`]).
    NoteList,
}
```

Replace:

```rust
            Selection::Check(check) => Some(check),
            Selection::Nothing | Selection::Result(_) => None,
        }
```

with:

```rust
            Selection::Check(check) => Some(check),
            _ => None,
        }
```

Replace:

```rust
            Selection::Result(path) => Tile::ALL.into_iter().find(|tile| tile.path() == path),
            Selection::Nothing | Selection::Check(_) => None,
        }
```

with:

```rust
            Selection::Result(path) => Tile::ALL.into_iter().find(|tile| tile.path() == path),
            _ => None,
        }
```

Replace:

```rust
/// ([`Selection::reads_consequences`]), the Size tab's consequences of the design, with the
/// size Auto sets while it is on.
pub(crate) struct Shown<'a> {
```

with:

```rust
/// ([`Selection::reads_consequences`]), the Size tab's consequences of the design, with the
/// size Auto sets while it is on; and why an input is not free on the design shown, for a
/// note's change entries ([`NoteView::held`]).
pub(crate) struct Shown<'a> {
```

Replace:

```rust
    pub auto: Option<FreeVariable>,
}
```

with:

```rust
    pub auto: Option<FreeVariable>,
    pub held: &'a dyn Fn(&str) -> Option<String>,
}
```

Replace:

```rust
        Selection::Result(path) => result_ui(ui, path, shown, readouts),
    };
```

with:

```rust
        Selection::Result(path) => result_ui(ui, path, shown, readouts),
        Selection::Note(id) => note_entry_ui(ui, id, shown, readouts),
        Selection::NoteList => note_list_ui(ui, shown.clamps_paused),
    };
```

Replace:

```rust
    asked.or(entry)
}
```

with:

```rust
    asked.or(entry)
}

/// The teaching notes shown, in their order ([`NOTES`]): every reviewed note but the one the
/// paused shaft-clamp section takes with it.
pub fn shown_notes(clamps_paused: bool) -> impl Iterator<Item = &'static Note> {
    NOTES
        .iter()
        .filter(move |note| shown_when(note.id, clamps_paused) && reviewed_note(note.id).is_some())
}

/// The note list ([`LEARN`]): each note shown, by its title, a link that selects it.
fn note_list_ui(ui: &mut egui::Ui, clamps_paused: bool) -> Option<Asked> {
    heading(ui, LEARN);
    let mut asked = None;
    for note in shown_notes(clamps_paused) {
        if ui.link(glyph_safe(note.title)).clicked() {
            asked = Some(Asked::Select(Selection::Note(note.id)));
        }
    }
    asked
}

/// A teaching note's entry: the links to the equations it explains ([`explains_ui`]), then the
/// note ([`note_ui`]) on the design shown. A change entry's input link shows its row, as a
/// check's fix does. Nothing for a note that is not reviewed.
fn note_entry_ui(
    ui: &mut egui::Ui,
    id: &'static str,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let note = reviewed_note(id)?;
    let equation = explains_ui(ui, note);
    let view = NoteView {
        inputs: shown.inputs,
        results: shown.results,
        held: shown.held,
    };
    match note_ui(ui, note, &view) {
        Some(NoteLink::Input(path)) => readouts.focus_input(path),
        Some(NoteLink::Note(id)) => return Some(Asked::Select(Selection::Note(id))),
        None => {}
    }
    equation.map(Asked::Equation)
}
```

**`magcoupling-rs/src/gui/explorer.rs`**

Replace:

```rust
//! workbook cell. The panel docks at the
//! bottom of the centre region, closed until a value is clicked or its header button is
//! pressed (decisions M43-1, M43-2).
//!
```

with:

```rust
//! workbook cell. The panel docks at the
//! bottom of the centre region, closed until a value is clicked or an Explain entry's Equation
//! link is followed (decisions M43-1, M43-2; UX overhaul decision C-11 took its button off the
//! top bar).
//!
```

Replace:

```rust
//!
//! Teaching notes (spec Addendum A4): the Explain toggle, off by default, shows under the open
//! equation the reviewed note that explains it ([`notes::note_for`]: only notes a physics
//! reviewer has checked), with its diagram ([`crate::gui::diagrams`]), its "What to change"
//! entries, its "Watch out" lines and its sources under a "Where this comes from" fold that
//! starts closed ([`note_ui`]), so it follows what the user investigates. Under an equation
//! without one it offers the next best read ([`note_fallback`], slice-2 spec 3.7): the closest
//! reviewed note upstream as a link, an assumption's rationale, or why there is none and where
//! the notes are; never "No teaching note" on a dashboard row. A `{result.path}` in a
//! note's text is drawn as that result's number on the design shown ([`live_value`]), or as
//! "no prediction" where it has none. An entry's input is a link to its row, or plain text with
//! the reason where the input is not free on the design shown ([`held_reason`]: Auto holds it
//! or a part pick sets it; or the slider is at its end that way). "Start here" walks the
//! suggested order ([`notes::START_HERE`]): each step opens its equation with Explain on. A warning's link
//! opens its note on its own ([`Explorer::open_note`]; five of the six warning notes explain
//! no single equation), with links to the equations it does explain. Every note the panel
//! names or draws by id goes through [`reviewed_note`], so a note sent back to Draft never
//! shows.
//!
```

with:

```rust
//!
//! Teaching notes (spec Addendum A4; UX overhaul decision C-11): the Explain panel draws them,
//! not this panel. Under the open equation the lesson line names the reviewed note explaining
//! it ([`notes::note_for`]: only notes a physics reviewer has checked) after "Full lesson:", a link
//! that shows the note in the Explain panel ([`lesson_ui`]). Under an equation without one it
//! offers the next best read ([`note_fallback`], slice-2 spec 3.7): the closest reviewed note
//! upstream as a link, an assumption's rationale, or why there is none and where the notes are;
//! never "No teaching note" on a dashboard row. A note itself ([`note_ui`], which the Explain
//! panel calls) has its diagram ([`crate::gui::diagrams`]), its "What to change" entries, its
//! "Watch out" lines and its sources under a "Where this comes from" fold that starts closed. A
//! `{result.path}` in a note's text is drawn as that result's number on the design shown
//! ([`live_value`]), or as "no prediction" where it has none. An entry's input is a link to its
//! row, or plain text with the reason where the input is not free on the design shown
//! ([`held_reason`]: Auto holds it or a part pick sets it; or the slider is at its end that
//! way); the links to the equations a note explains are [`explains_ui`]'s. "Start here" walks
//! the suggested order ([`notes::START_HERE`]): each step opens its equation here and its note
//! in the Explain panel (UX overhaul spec 5.5). Every note named or drawn by id goes through
//! [`reviewed_note`], so a note sent back to Draft never shows.
//!
```

Replace:

```rust
use crate::gui::diagrams::diagram_ui;
use crate::gui::format::value_with_unit;
```

with:

```rust
use crate::gui::diagrams::diagram_ui;
use crate::gui::explain::FULL_LESSON;
use crate::gui::format::value_with_unit;
```

Replace:

```rust

/// The header button that shows or hides the panel.
pub const EQUATION_PANEL: &str = "Equation panel";
```

with:

```rust

/// The panel's heading.
pub const EQUATION_PANEL: &str = "Equation panel";
```

Replace:

```rust
/// What the panel says before anything is opened (spec 3.7, TC-29b): how to open an equation,
/// and the two ways to the teaching notes, by their buttons' names (a test checks).
pub const EMPTY_TEXT: &str = "Hover a value to see its equation; click it to open it here, then click a term to follow it. Start here walks the main teaching notes in a suggested order; Explain shows the note under the open equation.";

```

with:

```rust
/// What the panel says before anything is opened (spec 3.7, TC-29b): how to open an equation,
/// and the walk through the teaching notes, by its button's name (a test checks).
pub const EMPTY_TEXT: &str = "Hover a value to see its equation; click it to open it here, then click a term to follow it. Start here walks the main teaching notes in a suggested order.";

```

Replace:

```rust

/// The toggle of the teaching note under the open equation.
pub const EXPLAIN: &str = "Explain";

/// The button that starts the suggested reading order.
```

with:

```rust

/// The button that starts the suggested reading order.
```

Replace:

```rust

/// The start of the links under a note opened on its own.
pub const EXPLAINS: &str = "Explains";
```

with:

```rust

/// The start of the links at the head of a note to the equations it explains.
pub const EXPLAINS: &str = "Explains";
```

Replace:

```rust
    focus: Option<Focus>,
    /// Whether the teaching note of the open equation is shown (off by default: spec A4).
    pub explain: bool,
    /// The start-here step shown, if the user is walking the suggested order.
```

with:

```rust
    focus: Option<Focus>,
    /// The start-here step shown, if the user is walking the suggested order.
```

Replace:

```rust
    start_here: Option<usize>,
    /// A teaching note opened on its own (a warning's link), by id.
    note: Option<&'static str>,
}
```

with:

```rust
    start_here: Option<usize>,
}
```

Replace:

```rust
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
```

with:

```rust
        self.focus = None;
        self.start_here = None;
```

Replace:

```rust

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
```

with:

```rust

    /// Opens start-here step `step`'s equation and returns its note, which the panel shows in
    /// the Explain panel (UX overhaul spec 5.5); a step past the last one changes nothing and
    /// returns none.
    pub fn start(&mut self, step: usize) -> Option<&'static str> {
        let &(id, path) = START_HERE.get(step)?;
        self.trail = vec![path.to_owned()];
        self.open = true;
        self.focus = None;
        self.start_here = Some(step);
        Some(id)
    }
```

Replace:

```rust

    /// Follows a link clicked in a teaching note or in Explain's line under an equation without
    /// one: an input's row is highlighted and scrolled to, as a leaf term's is, and the note
    /// stays open; a note is opened on its own ([`Explorer::open_note`]).
    pub fn follow_note_link(&mut self, link: NoteLink) {
        match link {
```

with:

```rust

    /// Follows a link clicked in a teaching note or in the lesson line under an equation: an
    /// input's row is highlighted and scrolled to, as a leaf term's is; a note is returned, for
    /// the panel to show in the Explain panel (decision C-11).
    pub fn follow_note_link(&mut self, link: NoteLink) -> Option<&'static str> {
        match link {
```

Replace:

```rust
        match link {
            NoteLink::Input(path) => self.focus_input(path),
            NoteLink::Note(id) => self.open_note(id),
        }
```

with:

```rust
        match link {
            NoteLink::Input(path) => {
                self.focus_input(path);
                None
            }
            NoteLink::Note(id) => Some(id),
        }
```

Replace:

```rust
    /// terms from the next frame (asked for at once); a value clicked opens here; a check
    /// guide's link highlights its input's row, as a leaf term does.
    pub fn end_frame(&mut self, ctx: &egui::Context, events: ReadoutEvents) {
```

with:

```rust
    /// terms from the next frame (asked for at once); a value clicked opens here; a check
    /// guide's link highlights its input's row, as a leaf term does. A note asked for is the
    /// Explain panel's: the panel takes it before this.
    pub fn end_frame(&mut self, ctx: &egui::Context, events: ReadoutEvents) {
```

Replace:

```rust
        }
        if let Some(id) = events.note {
            self.open_note(id);
            ctx.request_repaint();
        }
        if let Some(path) = events.focus {
```

with:

```rust
        }
        if let Some(path) = events.focus {
```

Replace:

```rust
/// The plain symbol of `path`, its label for a path without one.
fn crumb(path: &str) -> String {
    registry()
```

with:

```rust
/// The plain symbol of `path`, its label for a path without one.
pub(crate) fn crumb(path: &str) -> String {
    registry()
```

Replace:

```rust

/// Draws the Equation panel: its header row (heading, breadcrumb, close), then the path shown;
/// with `show_details` (the header's Show spreadsheet details, spec S2-5) also the value's
/// workbook cell, the corrections its equation embodies and each term's internal name on hover.
/// `baseline` is the session's baseline (UX overhaul spec 7.1, C-22): an input term carries the
/// changed tag while it differs from the baseline's value.
/// `held` says why an input is not free on the design shown ([`held_reason`]), for the notes'
/// change entries. `clamps_paused` is the panel's pause of the shaft-clamp section: the Start here
/// hover and the Used by links leave the clamp out while it holds.
#[allow(clippy::too_many_arguments)] // the design shown, its results and its baseline, then the view's switches
pub fn explorer_ui(
```

with:

```rust

/// Draws the Equation panel: its header row (heading, breadcrumb, Start here, close), then the
/// path shown; with `show_details` (the Design menu's Show spreadsheet details, spec S2-5) also
/// the value's workbook cell, the corrections its equation embodies and each term's internal
/// name on hover. `baseline` is the session's baseline (UX overhaul spec 7.1, C-22): an input
/// term carries the changed tag while it differs from the baseline's value. `clamps_paused` is
/// the panel's pause of the shaft-clamp section: the Start here hover and the Used by links
/// leave the clamp out while it holds. Returns the note asked for, which the Explain panel
/// shows (decision C-11): a lesson link's, or a Start here step's (spec 5.5).
pub fn explorer_ui(
```

Replace:

```rust
    show_details: bool,
    held: &dyn Fn(&str) -> Option<String>,
    clamps_paused: bool,
```

with:

```rust
    show_details: bool,
    clamps_paused: bool,
```

Replace:

```rust
    clamps_paused: bool,
) {
    let terms = Design { inputs, results };
```

with:

```rust
    clamps_paused: bool,
) -> Option<&'static str> {
    let terms = Design { inputs, results };
```

Replace:

```rust
    let terms = Design { inputs, results };
    let view = NoteView {
        inputs,
        results,
        held,
    };
    ui.horizontal_wrapped(|ui| {
```

with:

```rust
    let terms = Design { inputs, results };
    let mut asked = None;
    ui.horizontal_wrapped(|ui| {
```

Replace:

```rust
        ui.separator();
        if let Some(note) = explorer.note.and_then(reviewed_note) {
            let _ = ui.selectable_label(true, glyph_safe(note.title));
        }
        let mut back = None;
```

with:

```rust
        ui.separator();
        let mut back = None;
```

Replace:

```rust
            {
                explorer.start(0);
            }
```

with:

```rust
            {
                asked = explorer.start(0);
            }
```

Replace:

```rust
            }
            ui.checkbox(&mut explorer.explain, EXPLAIN)
                .on_hover_text("The teaching note of the open equation");
        });
```

with:

```rust
            }
        });
```

Replace:

```rust
            {
                explorer.start(step - 1);
            }
```

with:

```rust
            {
                asked = explorer.start(step - 1);
            }
```

Replace:

```rust
            {
                explorer.start(step + 1);
            }
```

with:

```rust
            {
                asked = explorer.start(step + 1);
            }
```

Replace:

```rust
        .show(ui, |ui| {
            if let Some(note) = explorer.note.and_then(reviewed_note) {
                let (equation, link) = note_alone_ui(ui, note, &view);
                if let Some(link) = link {
                    explorer.follow_note_link(link);
                }
                if let Some(path) = equation {
                    explorer.open_path(&path);
                }
                return;
            }
            let Some(path) = explorer.current().map(str::to_owned) else {
```

with:

```rust
        .show(ui, |ui| {
            let Some(path) = explorer.current().map(str::to_owned) else {
```

Replace:

```rust
            let (follow, link) = match registry().equation_for(&path) {
                Some(eq) => equation_body(
                    ui,
                    eq,
                    &terms,
                    baseline,
                    others_marked,
                    explorer.explain,
                    show_details,
                    &view,
                ),
                None => {
```

with:

```rust
            let (follow, link) = match registry().equation_for(&path) {
                Some(eq) => equation_body(ui, eq, &terms, baseline, others_marked, show_details),
                None => {
```

Replace:

```rust
                    no_equation_body(ui, &path, results, show_details);
                    let link = if explorer.explain {
                        note_fallback_ui(ui, &path)
                    } else {
                        None
                    };
                    (None, link)
                }
```

with:

```rust
                    no_equation_body(ui, &path, results, show_details);
                    (None, note_fallback_ui(ui, &path))
                }
```

Replace:

```rust
            if let Some(link) = link {
                explorer.follow_note_link(link);
            }
```

with:

```rust
            if let Some(link) = link {
                asked = asked.or(explorer.follow_note_link(link));
            }
```

Replace:

```rust
        });
}
```

with:

```rust
        });
    asked
}
```

Replace:

```rust

/// What a click in a teaching note, or in Explain's line under an equation without one, asks
/// for ([`Explorer::follow_note_link`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

with:

```rust

/// What a click in a teaching note, or in the lesson line under an equation, asks for
/// ([`Explorer::follow_note_link`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

Replace:

```rust
    Input(&'static str),
    /// Open this reviewed note on its own (Explain's closest note, [`note_fallback`]).
    Note(&'static str),
```

with:

```rust
    Input(&'static str),
    /// Show this reviewed note in the Explain panel (an equation's lesson, [`lesson_ui`], or
    /// its closest note, [`note_fallback`]).
    Note(&'static str),
```

Replace:

```rust

/// A note opened on its own: links to the equations it explains (a family's template is left
/// out), then the note. Returns the equation clicked and the note's link clicked.
fn note_alone_ui(
    ui: &mut egui::Ui,
    note: &Note,
    view: &NoteView<'_>,
) -> (Option<String>, Option<NoteLink>) {
    let paths: Vec<&str> = note
        .equations
```

with:

```rust

/// The links at the head of a note in the Explain panel to the equations it explains (a
/// family's template is left out), after [`EXPLAINS`]; nothing for a note that explains none.
/// Returns the equation clicked, which the panel opens here.
pub fn explains_ui(ui: &mut egui::Ui, note: &Note) -> Option<&'static str> {
    let paths: Vec<&'static str> = note
        .equations
```

Replace:

```rust
                {
                    clicked = Some(path.to_owned());
                }
```

with:

```rust
                {
                    clicked = Some(path);
                }
```

Replace:

```rust
    }
    let link = note_ui(ui, note, view);
    (clicked, link)
}
```

with:

```rust
    }
    clicked
}
```

Replace:

```rust
/// clicked is followed), the value, with `show_details` the corrections it embodies (their
/// titles on hover), with `explain` its teaching note (drawn against `view`) or, without one,
/// Explain's next best read ([`note_fallback_ui`]), then the term list, its swatches faded while
/// `others_marked`, a term's internal name on hover with `show_details`. An input term carries
/// the changed tag while it differs from `baseline`'s value. Returns the term clicked and the
/// link clicked in the note or in the next best read.
#[allow(clippy::too_many_arguments)] // the equation, its design and baseline, then the view's switches
fn equation_body(
```

with:

```rust
/// clicked is followed), the value, with `show_details` the corrections it embodies (their
/// titles on hover), its lesson line ([`lesson_ui`]), then the term list, its swatches faded
/// while `others_marked`, a term's internal name on hover with `show_details`. An input term
/// carries the changed tag while it differs from `baseline`'s value. Returns the term clicked
/// and the link clicked in the lesson line.
fn equation_body(
```

Replace:

```rust
    others_marked: bool,
    explain: bool,
    show_details: bool,
```

with:

```rust
    others_marked: bool,
    show_details: bool,
```

Replace:

```rust
    show_details: bool,
    view: &NoteView<'_>,
) -> (Option<String>, Option<NoteLink>) {
```

with:

```rust
    show_details: bool,
) -> (Option<String>, Option<NoteLink>) {
```

Replace:

```rust
    let mut clicked = None;
    let mut link = None;
    if show_details {
```

with:

```rust
    let mut clicked = None;
    if show_details {
```

Replace:

```rust
    }
    if explain {
        ui.separator();
        match notes::note_for(&eq.target) {
            Some(note) => link = note_ui(ui, note, view),
            None => {
                link = note_fallback_ui(ui, &eq.target);
            }
        }
        ui.separator();
    }
    ui.add_space(6.0);
```

with:

```rust
    }
    ui.separator();
    let link = lesson_ui(ui, &eq.target);
    ui.separator();
    ui.add_space(6.0);
```

Replace:

```rust
/// Explain's lines under an equation no reviewed note explains ([`note_fallback`]): the closest
/// note as a link, an assumption's label and rationale, or the reason and where the notes are.
/// Returns the note asked for ([`NoteLink::Note`]).
fn note_fallback_ui(ui: &mut egui::Ui, path: &str) -> Option<NoteLink> {
```

with:

```rust
/// The lesson line under the equation of `path` (decision C-11): [`FULL_LESSON`], then on the
/// row under it the title of the reviewed note that explains it, a link that shows the note in
/// the Explain panel (its own row, so a long title wraps inside the panel: plan review UXP-1);
/// for an equation no note explains, the next best read ([`note_fallback_ui`]). Returns the
/// link clicked.
fn lesson_ui(ui: &mut egui::Ui, path: &str) -> Option<NoteLink> {
    let Some(note) = notes::note_for(path) else {
        return note_fallback_ui(ui, path);
    };
    ui.label(FULL_LESSON);
    ui.link(glyph_safe(note.title))
        .clicked()
        .then_some(NoteLink::Note(note.id))
}

/// The next best read under an equation no reviewed note explains ([`note_fallback`]): the
/// closest note, a link; an assumption's rationale; or why there is none.
fn note_fallback_ui(ui: &mut egui::Ui, path: &str) -> Option<NoteLink> {
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
use crate::gui::dashboard::{Level, end_effect_banner_ui, why_note_link};
use crate::gui::explain::{Asked, Selection, explain_ui};
use crate::gui::explorer::{Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights};
```

with:

```rust
use crate::gui::dashboard::{Level, end_effect_banner_ui, why_note_link};
use crate::gui::explain::{Asked, LEARN, Selection, explain_ui};
use crate::gui::explorer::{Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights};
```

Replace:

```rust
        let mut events = readouts.finish();
        // A fix on an input the Size tab names (the Outside diameter, contract C-4) opens that
```

with:

```rust
        let mut events = readouts.finish();
        // A note asked for anywhere (a Why link, a warning's link, an entry's lesson link)
        // shows in the Explain panel (UX overhaul spec 5.2, decision C-11).
        if let Some(id) = events.note.take() {
            self.selection = Selection::Note(id);
            ui.ctx().request_repaint();
        }
        // A fix on an input the Size tab names (the Outside diameter, contract C-4) opens that
```

Replace:

```rust
            .show(ui, |ui| {
                let shown = crate::gui::explain::Shown {
```

with:

```rust
            .show(ui, |ui| {
                // A note's change entry names an input Auto or a part pick holds as plain
                // text, on the design shown.
                let sizing = self.sizing;
                let held = |path: &str| held_reason(&sizing, shown, path);
                let shown = crate::gui::explain::Shown {
```

Replace:

```rust
                    auto,
                };
```

with:

```rust
                    auto,
                    held: &held,
                };
```

Replace:

```rust
            Some(Asked::Equation(path)) => self.explorer.open_path(path),
            Some(Asked::StartHere) => self.explorer.start(0),
            None => {}
```

with:

```rust
            Some(Asked::Equation(path)) => self.explorer.open_path(path),
            Some(Asked::StartHere) => {
                if let Some(id) = self.explorer.start(0) {
                    self.selection = Selection::Note(id);
                }
            }
            None => {}
```

Replace:

```rust
            let (range, start) = panel_heights(ui.available_rect_before_wrap().height());
            egui::TopBottomPanel::bottom(EQUATION_PANEL_ID)
                .resizable(true)
```

with:

```rust
            let (range, start) = panel_heights(ui.available_rect_before_wrap().height());
            let note = egui::TopBottomPanel::bottom(EQUATION_PANEL_ID)
                .resizable(true)
```

Replace:

```rust
                .show_inside(ui, |ui| {
                    // A note's change entry names an input Auto or a part pick holds as plain
                    // text, on the design shown.
                    let sizing = self.sizing;
                    let held = |path: &str| held_reason(&sizing, shown, path);
                    explorer_ui(
```

with:

```rust
                .show_inside(ui, |ui| {
                    explorer_ui(
```

Replace:

```rust
                        show_details,
                        &held,
                        self.clamps_paused,
```

with:

```rust
                        show_details,
                        self.clamps_paused,
```

Replace:

```rust
                        self.clamps_paused,
                    );
                });
        }
```

with:

```rust
                        self.clamps_paused,
                    )
                })
                .inner;
            // A lesson link or a Start here step: its note in the Explain panel (spec 5.5).
            if let Some(id) = note {
                self.selection = Selection::Note(id);
                ui.ctx().request_repaint();
            }
        }
```

Replace:

```rust
                self.export_spreadsheet();
            }
        });
        // Spec Addendum A3: the banner while any assumption differs from its workbook default.
```

with:

```rust
                self.export_spreadsheet();
            }
            ui.separator();
            if ui.add(bar_button(LEARN)).clicked() {
                self.selection = Selection::NoteList;
            }
        });
        // Spec Addendum A3: the banner while any assumption differs from its workbook default.
```

Replace:

```rust
        if let Some(id) = output.note {
            self.explorer.open_note(id);
            ui.ctx().request_repaint();
```

with:

```rust
        if let Some(id) = output.note {
            self.selection = Selection::Note(id);
            ui.ctx().request_repaint();
```

Replace:

```rust
            if why_note_link(ui, TORQUE_BAND_NOTE) {
                self.explorer.open_note(TORQUE_BAND_NOTE);
                ui.ctx().request_repaint();
```

with:

```rust
            if why_note_link(ui, TORQUE_BAND_NOTE) {
                self.selection = Selection::Note(TORQUE_BAND_NOTE);
                ui.ctx().request_repaint();
```

**`magcoupling-rs/src/gui/readouts.rs`**

Replace:

```rust

    /// Asks for the teaching note `id` (a warning's link).
    pub fn open_note(&mut self, id: &'static str) {
```

with:

```rust

    /// Asks for the teaching note `id` (a warning's link, a lesson link), which the panel shows
    /// in the Explain panel.
    pub fn open_note(&mut self, id: &'static str) {
```

- [ ] **Step 4: Run the checks.** Expected: `test result: ok. 1078 passed; 0 failed; 6 ignored`; clippy clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** the seven files: `feat(magcoupler): the teaching notes in the Explain panel: Learn lists them, every note link and each Start here step selects its note, the Equation panel's Explain checkbox goes (BL-085 T3)`.

### Task 4: gui-smoke reads the top bar and the note in the Explain panel

**Model:** `sonnet`.

The prompt's first view names the bar's six words and no Equation panel button; `details_toggle` opens the Design menu; the file step opens the menu for Load design; `lesson_link_opened` finds the note in the Explain panel. The workflow's description, the gui-smoke skill's listing, says so (plan review COV-3). The check keys do not change, so the verdict test does not either. The Rust guard comes first.

**Files:**
- Modify: `magcoupling-rs/src/app.rs`, `magcoupling-rs/src/gui/mod.rs`, `.claude/workflows/gui-smoke.js`

- [ ] **Step 1 (red): the prompt's guard.**

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
pub use panel::{
    CentreView, DESIGN_MENU, MagcouplingPanel, OPEN_EXAMPLE, PanelRequest, SHOW_DETAILS,
    SIZING_LOG_PREFIX,
};
```

with:

```rust
pub use panel::{
    CentreView, DESIGN_MENU, EXPORT, LOAD_DESIGN, MagcouplingPanel, OPEN_EXAMPLE, PanelRequest,
    RESET_TO_START, SHARE, SHOW_DETAILS, SIZING_LOG_PREFIX,
};
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        assert!(script.contains(crate::gui::readouts::REGISTRY_LOG_PREFIX));
        // The header's Show spreadsheet details toggle (spec S2-5), which the first view shows.
        assert!(
```

with:

```rust
        assert!(script.contains(crate::gui::readouts::REGISTRY_LOG_PREFIX));
        // The Show spreadsheet details toggle (spec S2-5), in the Design menu (UX overhaul
        // decision C-12).
        assert!(
```

Replace:

```rust
            script.contains(crate::gui::SHOW_DETAILS),
            "the smoke prompt names the header's toggle"
        );
```

with:

```rust
            script.contains(crate::gui::SHOW_DETAILS),
            "the smoke prompt names the details toggle"
        );
```

Replace:

```rust
        );
        // Spec R14 (guidance slice 2): the steps name a note's What to change and its sources
```

with:

```rust
        );
        // UX overhaul spec 4.2: the prompt names the top bar's buttons by the words the page
        // draws, finds the menu's items under Design, and no longer looks for the buttons that
        // left the bar.
        {
            use crate::gui::explain::LEARN;
            use crate::gui::{EXPORT, LOAD_DESIGN, RESET_TO_START, SHARE};
            for (what, text) in [
                ("the Design menu", crate::gui::DESIGN_MENU),
                ("the reset", RESET_TO_START),
                ("the file's load", LOAD_DESIGN),
                ("Share", SHARE),
                ("Export", EXPORT),
                ("Learn", LEARN),
            ] {
                assert!(
                    script.contains(&format!("\"{text}\"")),
                    "the smoke prompt names {what}: {text:?}"
                );
            }
            for gone in [
                "\"Reset all\"",
                "\"Copy share link\"",
                "\"Equation panel\" button",
                "after \"Reset all\"",
            ] {
                assert!(
                    !script.contains(gone),
                    "the smoke prompt still names {gone}"
                );
            }
            // The lesson link shows its note in the Explain panel (decision C-11), where the
            // step looks for it.
            let link = script
                .find("lesson_link_opened=true")
                .expect("the link's verdict");
            let verdict = &script[link..];
            let verdict = &verdict[..verdict.find('.').unwrap_or(verdict.len())];
            assert!(
                verdict.contains("Explain panel") && !verdict.contains("Equation panel"),
                "the note is judged in the Explain panel: {verdict}"
            );
        }
        // Spec R14 (guidance slice 2): the steps name a note's What to change and its sources
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_smoke_test_share_link_opens_its_design 2>&1 | tail -n 15
  ```

  Expected: 1 failed: `the_smoke_test_share_link_opens_its_design`.

- [ ] **Step 3 (green): the prompt.**

**`.claude/workflows/gui-smoke.js`**

Replace:

```js
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows four tiles over the drawing and the check list over the Explain panel, a click on a check shows its entry and its lesson link opens a note, an input tooltip
```

with:

```js
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, an input tooltip
```

Replace:

```js

// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Load design
// button, a check's row, a lesson link, the Geometry tab, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

with:

```js

// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Design menu and
// its Load design item, a check's row, a lesson link, the Geometry tab, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

Replace:

```js
// slice B, stage 2 (spec 4.4 and 4.5), the first screenshot shows the four tiles over the drawing
// and the check list over the Explain panel's orientation with no click. One click on the
// canvas, on the "Load design" button found in a screenshot, checks the web file picker (rfd's
// HTML overlay) and the file load. Then (spec 5.2 and 5.3) a click on a check's row shows its
// entry in the Explain panel, which scrolls on its own, and the entry's lesson link opens a
// teaching note; and an input row's tooltip shows no spreadsheet details while the toggle is
// off. Then (slice 3, spec 7.2 and 8.6) the link is opened again with a timer on every frame the
// app draws: on that fresh page a click on the Hot-low tile shows the torque card's heading,
```

with:

```js
// slice B, stage 2 (spec 4.4 and 4.5), the first screenshot shows the four tiles over the drawing
// and the check list over the Explain panel's orientation with no click; since part 2a (spec 4.2)
// the top bar is Undo, Redo, Design, Share, Export and Learn, and the Design menu holds the details
// toggle. Two clicks on the canvas, on the Design menu and its "Load design" item found in
// screenshots, check the web file picker (rfd's HTML overlay) and the file load. Then (spec 5.2
// and 5.3) a click on a check's row shows its entry in the Explain panel, which scrolls on its
// own, and the entry's lesson link shows a teaching note there (decision C-11); and an input
// row's tooltip shows no spreadsheet details while the toggle is off. Then (slice 3, spec 7.2 and 8.6) the link is opened again with a timer on every frame the
// app draws: on that fresh page a click on the Hot-low tile shows the torque card's heading,
```

Replace:

```js
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: first set the window to 1280 x 800 CSS pixels with browser_resize (the laptop size of the first view). Then the starting design: navigate to ${url}/tools/magcoupler/ (the clean address, with no share link) and wait 8 seconds (WASM init). Take a screenshot and read the address's query with browser_run_code_unsafe, code: async (page) => page.evaluate(() => location.search). starting_design=true only if the check list (the right side), under its heading "CHECKS", is topped by the line "${STARTING_STATUS}", the header row at the top of the page (it may wrap to a second line) shows the button "Open the example design" right after "Reset all", and the code returned an empty string (at the starting design the address stays clean). Then the look: read the page's background with browser_run_code_unsafe, code: async (page) => page.evaluate(() => getComputedStyle(document.body).backgroundColor). look=true only if the code returned "${PAGE_BACKGROUND}" and the screenshot shows the calculator's own look: a charcoal page; at the top left a small crimson square right before the title "Magnetic coupling calculator"; in the row of tabs the selected tab, "Size and torque", marked by a crimson line under its name and not by a filled button; and every text drawn in one clean sans-serif face, with no empty box where a character should be. Collect the console messages at level "info" (this page's: the next step navigates away).
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
```

with:

```js
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: first set the window to 1280 x 800 CSS pixels with browser_resize (the laptop size of the first view). Then the starting design: navigate to ${url}/tools/magcoupler/ (the clean address, with no share link) and wait 8 seconds (WASM init). Take a screenshot and read the address's query with browser_run_code_unsafe, code: async (page) => page.evaluate(() => location.search). starting_design=true only if the check list (the right side), under its heading "CHECKS", is topped by the line "${STARTING_STATUS}", the header row at the top of the page (it may wrap to a second line) shows the words "Undo", "Redo", "Design", "Share", "Export" and "Learn" in this order, each without a box around it, and no "Equation panel" or "Open the example design" button, and the code returned an empty string (at the starting design the address stays clean). Then the look: read the page's background with browser_run_code_unsafe, code: async (page) => page.evaluate(() => getComputedStyle(document.body).backgroundColor). look=true only if the code returned "${PAGE_BACKGROUND}" and the screenshot shows the calculator's own look: a charcoal page; at the top left a small crimson square right before the title "Magnetic coupling calculator"; in the row of tabs the selected tab, "Size and torque", marked by a crimson line under its name and not by a filled button; and every text drawn in one clean sans-serif face, with no empty box where a character should be. Collect the console messages at level "info" (this page's: the next step navigates away).
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
```

Replace:

```js
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (the size controls on the left, the check list and the Explain panel on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the page shows the first view with no click. The middle region: at its top a row of tabs starting "Size and torque" (the selected tab); under the tabs four tiles side by side, each a large number with its unit over a smaller, quieter label, the second labelled "Hot-low torque" with a line starting "needs" under its label; then a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft. The right side: the heading "CHECKS", a status line, then one row per check, each a small shape, the check's name and a word at the row's right end (OK, Check or Fails); a quiet line under the rows about fixed 3D fields; then the heading "EXPLAIN" and a paragraph starting "Two rings of magnets" with three numbered lines under it. details_toggle=true only if the header row at the top of the page (it may wrap to a second line) shows, after the "Equation panel" button, a checkbox labelled "Show spreadsheet details", unticked.
Then the design file picker: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
```

with:

```js
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (the size controls on the left, the check list and the Explain panel on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the page shows the first view with no click. The middle region: at its top a row of tabs starting "Size and torque" (the selected tab); under the tabs four tiles side by side, each a large number with its unit over a smaller, quieter label, the second labelled "Hot-low torque" with a line starting "needs" under its label; then a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft. The right side: the heading "CHECKS", a status line, then one row per check, each a small shape, the check's name and a word at the row's right end (OK, Check or Fails); a quiet line under the rows about fixed 3D fields; then the heading "EXPLAIN" and a paragraph starting "Two rings of magnets" with three numbered lines under it. Then the Design menu: click "Design" in the header row (every click as below: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot), wait 1 second and take a screenshot. details_toggle=true only if a menu opened under "Design" listing, top to bottom, "Reset to the starting design", "Open the example design", "Save design", "Load design" and a checkbox labelled "Show spreadsheet details", unticked. Press Escape with code: async (page) => { await page.keyboard.press('Escape'); } and wait 1 second (the menu closes).
Then the design file picker: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
```

Replace:

```js
Then the design file picker: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
In the screenshot, find the "Load design" button in the header row at the top of the page (between "Save design" and "Copy share link") and click its centre with browser_run_code_unsafe, code: async (page) => { await page.mouse.click(X, Y); } (X and Y in CSS pixels of the screenshot; the canvas fills the page). rfd shows its overlay (#rfd-overlay: a file input #rfd-input shown as a "Choose File" button, and the buttons "Ok" and "Cancel") and opens the browser's file chooser at once (the tool output reports a "File chooser" modal state); if a snapshot shows the overlay but no chooser opened, click the "Choose File" button. Upload the file with browser_file_upload, click the overlay's "Ok" button, wait 2 seconds and collect the console messages again; delete the file.
design_file_loaded=true only if a console message contains "magcoupling: loaded a design file". If the button cannot be found or the overlay does not appear after two attempts (each with a fresh screenshot), design_file_loaded=false and say why in notes: the canvas click is the fragile part of this step, and the share-link checks do not depend on it.
Then a check's entry and its lesson link (every click and mouse move below as above: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot): in the check list (the right side) click the name of the row "Hot-low torque". Wait 1 second and take a screenshot. Under the check list the Explain panel now shows the check's entry: a small shape, "Hot-low torque" and its word, then the headings "What it is" and "Numbers behind this check" with the check's numbers under the second. If the line "Full lesson:" at the entry's foot is not in view, move the mouse over the Explain panel and scroll down with code: async (page) => { await page.mouse.move(X, Y); await page.mouse.wheel(0, 1500); await page.waitForTimeout(1000); }, then take a screenshot. check_entry=true only if the entry shows as described, the line "Full lesson:" followed by a link is in view, and the check list's rows above the Explain panel are still in view (the Explain panel scrolls on its own). Then click the link right of "Full lesson:". Wait 1 second and take a screenshot. lesson_link_opened=true only if the Equation panel is now open, docked at the bottom of the middle region, and shows a teaching note: a bold title, a few sentences, the heading "What to change" with lines that start with Raise, Lower, Choose or Enter, and a closed fold "Where this comes from".
Then an input tooltip: click the "Geometry" tab in the row of tabs at the top of the middle region and wait 1 second. On the left the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
```

with:

```js
Then the design file picker: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
In a fresh screenshot click "Design" in the header row at the top of the page with browser_run_code_unsafe, code: async (page) => { await page.mouse.click(X, Y); } (X and Y in CSS pixels of the screenshot; the canvas fills the page), wait 1 second, take a screenshot, and click the menu's "Load design" item the same way. rfd shows its overlay (#rfd-overlay: a file input #rfd-input shown as a "Choose File" button, and the buttons "Ok" and "Cancel") and opens the browser's file chooser at once (the tool output reports a "File chooser" modal state); if a snapshot shows the overlay but no chooser opened, click the "Choose File" button. Upload the file with browser_file_upload, click the overlay's "Ok" button, wait 2 seconds and collect the console messages again; delete the file.
design_file_loaded=true only if a console message contains "magcoupling: loaded a design file". If the menu or its item cannot be found or the overlay does not appear after two attempts (each with a fresh screenshot), design_file_loaded=false and say why in notes: the canvas click is the fragile part of this step, and the share-link checks do not depend on it.
Then a check's entry and its lesson link (every click and mouse move below as above: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot): in the check list (the right side) click the name of the row "Hot-low torque". Wait 1 second and take a screenshot. Under the check list the Explain panel now shows the check's entry: a small shape, "Hot-low torque" and its word, then the headings "What it is" and "Numbers behind this check" with the check's numbers under the second. If the line "Full lesson:" at the entry's foot is not in view, move the mouse over the Explain panel and scroll down with code: async (page) => { await page.mouse.move(X, Y); await page.mouse.wheel(0, 1500); await page.waitForTimeout(1000); }, then take a screenshot. check_entry=true only if the entry shows as described, the line "Full lesson:" followed by a link is in view, and the check list's rows above the Explain panel are still in view (the Explain panel scrolls on its own). Then click the link right of "Full lesson:". Wait 1 second and take a screenshot. lesson_link_opened=true only if the Explain panel (the right side, under the check list) now shows a teaching note in place of the check's entry: a bold title, a few sentences, the heading "What to change" with lines that start with Raise, Lower, Choose or Enter, and a closed fold "Where this comes from" (scroll the Explain panel down as above if the fold is not in view).
Then an input tooltip: click the "Geometry" tab in the row of tabs at the top of the middle region and wait 1 second. On the left the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
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

  Expected: `test result: ok. 1078 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both `grep -vc` lines print `0` (no test binary with a failure); both wasm32 clippy runs clean; `cargo fmt --check` prints nothing; the verdict test's last line `all 77 checks hold`.

- [ ] **Step 5: Commit** the three files: `test(magcoupler): gui-smoke opens the Design menu and reads the note in the Explain panel (BL-085 T4)`.

### Task 5: Docs sweep

**Model:** `sonnet`.

The README's top bar, Equation panel, notes, warnings, session, baseline and smoke-run passages; `docs/ai/` (02 the layout, 03 the modules, 04 the issue, 05 the tracker, the backlog item in progress); the register's rows A-23 and A-24 and its log; owner question 50; the design's status line.

**Files:**
- Modify: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/analyses/2026-11-18-owner-questions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

- [ ] **Step 1: The docs.**

**`docs/ai/02-system.yaml`**

Replace:

```yaml
      - "The session baseline and the two named designs (UX overhaul slice A, BL-077; design docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md sections 7.1 and 7.2): DesignInputs::default() is the example design, the spreadsheet's values, and the overhaul never moves it. The standalone app opens on app::starting_design() (seven inputs off the spreadsheet's, among them the standard 1.4 mm hub keyway, and its own fourteen stored 3D fields; Required torque 1.35 N·m; its test pins it to compute_all: every counted check Good, no caution, no material warning, no assumption changed) and hands it to the panel as its baseline (MagcouplingPanel::with_baseline; new() keeps the spreadsheet's values). On screen 'the default' is the baseline: the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's 'Starting design'. The spreadsheet export's changed flag, the assumptions banner and the Assumptions view's dots stay against the workbook. Nothing moves the baseline (no undo, file, link or opened design), and it is in no file, link or undo step. Design files and links do not depend on it: they name every input, and a missing input still reads the spreadsheet's value. On the web the app writes app::address_of(the settled design) into the page's address with history.replaceState: the clean address at the starting design, the Share link elsewhere; Share always copies the full link. It writes only while no key is held (a held Ctrl+Z or Ctrl+Y settles a step at every key repeat; the run gives one write at the release), and address_design advances only when the browser took the write (Safari refuses more than about 100 in 10 seconds); a refused write is asked again on the next frame, with no back-off. If a later physics fix makes the starting design fail, it gets a new starting design, never a looser test. Owner rulings of 2026-11-22 (BL-080; docs/analyses/2026-11-10-conservative-assumptions.md rows A-17 and A-18): its hub keyway is the standard 1.4 mm (DIN 6885-1 t2 for a 3 x 3 mm key), so the wall past it is 2.55 mm and its opening tab draws no caution; its temperature check and the inner hub's fitting temperature read 3D fields computed for its own geometry (STARTING_3D_FIELDS, a run of reference/magcoupling-py/tools/starting_design_fields.py pinned by a test), which stay fixed under an edit like the example design's."
      - "Since BL-081 (UX overhaul slice B, stage 2, part 1; plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md): the right column is gui::checks over gui::explain, the opening tab is gui::tiles over geometry_view::geometry_ui (the drawing and its callout list, which leaves for the Geometry tab with part 2's tag selections, decision D-19), and the Size column starts with the Required torque row and scrolls in a short window. The panel owns one explain::Selection (Nothing, a check, a tile's result): view state, set by a click on a check's row, a tile or a tile entry's check, cleared by Esc (unless a text field or a popup has the key) or About this tool; a hover never sets it. An entry's Equation link opens the Equation panel and leaves the selection (C-25); the orientation's Start here link starts the walk. The dashboard's list (dashboard_ui), its open Problem, Why and Fix blocks, the welcome strip (WELCOME) and the centre's torque readouts and collapsing torque card are gone; their texts are the entries' (the plan's nothing-lost table). Every state's mark is checks::state_mark's shape (C-4). The check list and the Explain panel add no compute_all to an idle frame: the consequences are asked of the ConsequenceCache only while the Hot-low tile is selected."
```

with:

```yaml
      - "The session baseline and the two named designs (UX overhaul slice A, BL-077; design docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md sections 7.1 and 7.2): DesignInputs::default() is the example design, the spreadsheet's values, and the overhaul never moves it. The standalone app opens on app::starting_design() (seven inputs off the spreadsheet's, among them the standard 1.4 mm hub keyway, and its own fourteen stored 3D fields; Required torque 1.35 N·m; its test pins it to compute_all: every counted check Good, no caution, no material warning, no assumption changed) and hands it to the panel as its baseline (MagcouplingPanel::with_baseline; new() keeps the spreadsheet's values). On screen 'the default' is the baseline: the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's 'Starting design'. The spreadsheet export's changed flag, the assumptions banner and the Assumptions view's dots stay against the workbook. Nothing moves the baseline (no undo, file, link or opened design), and it is in no file, link or undo step. Design files and links do not depend on it: they name every input, and a missing input still reads the spreadsheet's value. On the web the app writes app::address_of(the settled design) into the page's address with history.replaceState: the clean address at the starting design, the Share link elsewhere; Share always copies the full link. It writes only while no key is held (a held Ctrl+Z or Ctrl+Y settles a step at every key repeat; the run gives one write at the release), and address_design advances only when the browser took the write (Safari refuses more than about 100 in 10 seconds); a refused write is asked again on the next frame, with no back-off. If a later physics fix makes the starting design fail, it gets a new starting design, never a looser test. Owner rulings of 2026-11-22 (BL-080; docs/analyses/2026-11-10-conservative-assumptions.md rows A-17 and A-18): its hub keyway is the standard 1.4 mm (DIN 6885-1 t2 for a 3 x 3 mm key), so the wall past it is 2.55 mm and its opening tab draws no caution; its temperature check and the inner hub's fitting temperature read 3D fields computed for its own geometry (STARTING_3D_FIELDS, a run of reference/magcoupling-py/tools/starting_design_fields.py pinned by a test), which stay fixed under an edit like the example design's."
      - "Since BL-085 (UX overhaul slice B, stage 2, part 2a; plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md): the top bar is Undo, Redo, Design (a menu: the reset, Open the example design, Save design, Load design, Show spreadsheet details), Share, Export and Learn, each a word with no box at rest (panel.rs bar_button, design_menu_ui, reset_label: Reset to the starting design where the baseline is not the spreadsheet's design). The Explain panel draws the teaching notes (decision C-11): explain::Selection gains Note and NoteList; every note link (ReadoutEvents::note, the Size column's Why links, explorer_ui's return: the lesson line under an open equation, a Start here step) sets the selection; the Equation panel has no Explain checkbox, no note view and no button on the bar. format::value_with_unit: a text value takes no unit (design 9.2)."
      - "Since BL-081 (UX overhaul slice B, stage 2, part 1; plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md): the right column is gui::checks over gui::explain, the opening tab is gui::tiles over geometry_view::geometry_ui (the drawing and its callout list, which leaves for the Geometry tab with part 2's tag selections, decision D-19), and the Size column starts with the Required torque row and scrolls in a short window. The panel owns one explain::Selection (Nothing, a check, a tile's result): view state, set by a click on a check's row, a tile or a tile entry's check, cleared by Esc (unless a text field or a popup has the key) or About this tool; a hover never sets it. An entry's Equation link opens the Equation panel and leaves the selection (C-25); the orientation's Start here link starts the walk. The dashboard's list (dashboard_ui), its open Problem, Why and Fix blocks, the welcome strip (WELCOME) and the centre's torque readouts and collapsing torque card are gone; their texts are the entries' (the plan's nothing-lost table). Every state's mark is checks::state_mark's shape (C-4). The check list and the Explain panel add no compute_all to an idle frame: the consequences are asked of the ConsequenceCache only while the Hot-low tile is selected."
```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel: new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption)), input_ui.rs (input_row (a locked row keeps its label live, to trace) -> RowOutput {edit, widget, label_clicked}, slider, RowEdit, TRACE_HINT, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the inputs side; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

with:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel: new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption)), input_ui.rs (input_row (a locked row keeps its label live, to trace) -> RowOutput {edit, widget, label_clicked}, slider, RowEdit, TRACE_HINT, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the inputs side; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 1 (BL-081, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md, decisions D-1 to D-19): the check list over the Explain panel and four tiles over the drawing. Stage 2 is cut in two (D-1): part 2 is the top bar and Design menu, Learn, the Size column's folds and its hint lines moving into the Explain panel with inputs as selections, dimension tags as selections with the callout list leaving the opening tab (D-19), the Plots tab with its picker, the word budget of 250 and the slip-heating unit. A check's row shows a shape and a word, not a short form of its quantity (D-2, register row A-20, owner question 49). risk: physics for the orientation and the tile's needs line, so every task runs on fable."
  - "Payload weights Track 2: before merge the user runs the hands-on checklist at the end of docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md (its section 0 is the gate and gui-smoke)."
```

with:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 1 (BL-081, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md, decisions D-1 to D-19): the check list over the Explain panel and four tiles over the drawing. Stage 2 is cut in two (D-1): part 2 is the top bar and Design menu, Learn, the Size column's folds and its hint lines moving into the Explain panel with inputs as selections, dimension tags as selections with the callout list leaving the opening tab (D-19), the Plots tab with its picker, the word budget of 250 and the slip-heating unit. A check's row shows a shape and a word, not a short form of its quantity (D-2, register row A-20, owner question 49). risk: physics for the orientation and the tile's needs line, so every task runs on fable."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
  - "Payload weights Track 2: before merge the user runs the hands-on checklist at the end of docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md (its section 0 is the gate and gui-smoke)."
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown

---

## 2026-11-22 - Magcoupler UX overhaul, slice B, stage 2, part 1: the instrument's columns (BL-081)
```

with:

```markdown

---

## 2026-11-22 - Magcoupler UX overhaul, slice B, stage 2, part 2a: the top bar and the notes (BL-085)
- What changed: the top bar is Undo, Redo, Design, Share, Export and Learn, words with no box until hovered; the Design menu holds the reset (Reset to the starting design where the baseline is the starting design), Open the example design, Save design, Load design and Show spreadsheet details; the Equation panel button left the bar. The Explain panel draws the teaching notes: Learn lists them by title, and every note link on the page (Why links, warnings' links, lesson links, the lesson line under an open equation, each Start here step) selects its note there; the Equation panel's Explain checkbox and its note view are gone. A text value takes no unit (the slip-heating readout read "never: steady state stays below the limit s").
- Tests: the top bar's and the menu's order and places, the buttons' box only under the pointer, Learn's list and a note's entry, the lesson line (drawn whole at every walk step in the app's font), the walk selecting its notes, the readout's words; the panel's note-link tests read the selection; the first view holds at most 430 words (427).
- Not in this part (plan decision D-20): the Size column's folds and lines in the Explain panel, inputs, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell (part 2b); BL-083 and BL-084 (D-30).

## 2026-11-22 - Magcoupler UX overhaul, slice B, stage 2, part 1: the instrument's columns (BL-081)
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  priority: 2
  status: open
  notes: "Filed 2026-11-22 with its plan (the design's header: each plan files one item). BL-081's part 2 is cut in two (the plan's decision D-20, register row A-23): this part, 2a, is the top bar and the notes; part 2b, with its own plan and item, is the Size column (its folds, its hint, Set by, floor and Why lines and the LINKED_HOUSING footnote moving into the Explain panel with inputs as selections, the torque card's row of each size control, an entry's 'Spreadsheet cell' foot link), dimension tags, results-table rows and callout rows as selections with the callout list leaving the opening tab, the Plots tab with its picker (C-9) and the plots' red reference lines (stage 1 review C-3), the word budget of 250 (goal G-1, test 8), the controls' outlines (E-4), a SemiBold face (S-1) and the page shell (S-4). BL-083 and BL-084 are not folded in (decision D-30): the design gives neither text. risk: mechanical (decision D-31): the new strings are the bar's and the menu's words ('Design', 'Share', 'Export', 'Learn'; 'Reset to the starting design' is the design's own), the notes are reviewed text moved from the Equation panel to the Explain panel, and the stray unit's removal states nothing new. So the build tasks transcribe the plan's exact blocks on sonnet and the reviews and the release run on fable (CLAUDE.md section 5). Open question 50 (the status line's place) takes its conservative answer: it stays under the bar (decision D-22, register row A-24)."
- id: BL-055
```

with:

```yaml
  priority: 2
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2a (Tasks 1 to 4 of the plan); the full gate, the smoke run and the whole-branch review are its Task 6. Filed 2026-11-22 with its plan (the design's header: each plan files one item). BL-081's part 2 is cut in two (the plan's decision D-20, register row A-23): this part, 2a, is the top bar and the notes; part 2b, with its own plan and item, is the Size column (its folds, its hint, Set by, floor and Why lines and the LINKED_HOUSING footnote moving into the Explain panel with inputs as selections, the torque card's row of each size control, an entry's 'Spreadsheet cell' foot link), dimension tags, results-table rows and callout rows as selections with the callout list leaving the opening tab, the Plots tab with its picker (C-9) and the plots' red reference lines (stage 1 review C-3), the word budget of 250 (goal G-1, test 8), the controls' outlines (E-4), a SemiBold face (S-1) and the page shell (S-4). BL-083 and BL-084 are not folded in (decision D-30): the design gives neither text. risk: mechanical (decision D-31): the new strings are the bar's and the menu's words ('Design', 'Share', 'Export', 'Learn'; 'Reset to the starting design' is the design's own), the notes are reviewed text moved from the Equation panel to the Explain panel, and the stray unit's removal states nothing new. So the build tasks transcribe the plan's exact blocks on sonnet and the reviews and the release run on fable (CLAUDE.md section 5). Open question 50 (the status line's place) takes its conservative answer: it stays under the bar (decision D-22, register row A-24)."
- id: BL-055
```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
| A-21 | Slice B's stage 2 is the whole of sections 4 and 5. Built at once or in parts? | Two parts. The first is the right column (the check list, the Explain panel with its orientation, check and tile entries), the four tiles, the Required torque row and the fixed-3D line on the first view; the opening tab keeps the callout list under the drawing. The second is the top bar and its Design menu, Learn, the Size column's folds and hint lines in the Explain panel, inputs and dimension tags as selections (the callout list then moves to the Geometry tab), the Plots tab and the word budget of 250. | Smaller releases (Q-4's answer); each part ships with nothing visible today lost, and the first changes the first view the owner called outdated. | The second part is planned and built with the first's branch before either merges. | BL-081; the plan's decisions D-1 and D-19 |

```

with:

```markdown
| A-21 | Slice B's stage 2 is the whole of sections 4 and 5. Built at once or in parts? | Two parts. The first is the right column (the check list, the Explain panel with its orientation, check and tile entries), the four tiles, the Required torque row and the fixed-3D line on the first view; the opening tab keeps the callout list under the drawing. The second is the top bar and its Design menu, Learn, the Size column's folds and hint lines in the Explain panel, inputs and dimension tags as selections (the callout list then moves to the Geometry tab), the Plots tab and the word budget of 250. | Smaller releases (Q-4's answer); each part ships with nothing visible today lost, and the first changes the first view the owner called outdated. | The second part is planned and built with the first's branch before either merges. | BL-081; the plan's decisions D-1 and D-19 |
| A-23 | BL-081's part 2 (the top bar, the Size column, more selections, the Plots tab, the word budget, the controls' outlines, the second face, the page shell) is still too large for one plan a reviewer can hold. Built at once or in parts? | Two parts. Part 2a: the top bar with its Design menu, Share, Export and Learn, the notes in the Explain panel, the Equation panel's Explain checkbox and bar button gone, each Start here step selecting its note, the slip-heating readout's stray unit. Part 2b: the rest. | Smaller releases (the conservative answer to Q-4, as A-21); 2a changes no column and loses nothing. | One plan for all of part 2. | The plan of part 2a (BL-085), decision D-20 |
| A-24 | The design (4.2) puts the status line for file and link results at the right end of the top bar. Today it is a line under the bar. Move it? | It stays under the bar. | Its place is read by tests and by the smoke run; a long status ("Share link copied to the clipboard (2,690 characters)") at the bar's end would wrap the bar at 1280 points. Owner question 50. | It moves to the bar's right end. | The plan of part 2a (BL-085), decision D-22 |

```

Replace:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-22: A-20 and A-21 added with the plan of slice B's stage 2, part 1 (BL-081). Each is an assumption, not the owner's answer. A-18's line is met there: the check list closes with the stored-3D line, on the first view at 1280 x 800, and an app test holds it.
```

with:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-22: A-23 and A-24 added with the plan of slice B's stage 2, part 2a (BL-085). Each is an assumption, not the owner's answer.
- 2026-11-22: A-20 and A-21 added with the plan of slice B's stage 2, part 1 (BL-081). Each is an assumption, not the owner's answer. A-18's line is met there: the check list closes with the stored-3D line, on the first view at 1280 x 800, and an app test holds it.
```

**`docs/analyses/2026-11-18-owner-questions.md`**

Replace:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 30 calls to make when convenient (questions 48 and 49 were added there on 2026-11-22). Part D, five that matter only if the tools stop being generic.

```

with:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 31 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22). Part D, five that matter only if the tools stop being generic.

```

Replace:

```markdown

### Linkage simulator
```

with:

```markdown

### 50. Should the status line of a file or a link sit at the right end of the top bar?

Added 2026-11-22 (the next free number). The design puts the line that reports a saved or loaded file or a copied link ("Share link copied to the clipboard (2,690 characters)") at the right end of the top bar. Today it is a line under the bar, and part 2a keeps it there.

- **If you do not answer:** it stays under the bar.
- (a) Under the bar, as now.
- (b) At the right end of the bar (a long line wraps the bar onto a second row at 1280 points).

Source: UX design section 4.2; A-24; BL-085 plan, decision D-22.

### Linkage simulator
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

with:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

**`magcoupling-rs/README.md`**

Replace:

```markdown
(plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-3-explorer-assumptions-notes-pickers.md`).
A header with the session buttons and the **Show spreadsheet details** toggle (slice 2, S2-5:
off by default; a view choice, in no design file, share link or undo step, kept by loads, undo
```

with:

```markdown
(plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-3-explorer-assumptions-notes-pickers.md`).
A top bar (since slice B, stage 2, part 2a of the UX overhaul, BL-085, plan
`docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`, design section 4.2): the brand mark
and the title, then **Undo**, **Redo**, **Design**, **Share** (the full link of the design shown,
to the clipboard, as Copy share link did), **Export** (the spreadsheet) and **Learn** (the
teaching notes by title in the Explain panel), each a word with no box until the pointer is on it
(stage 1 review S-3). The **Design** menu holds the reset (**Reset to the starting design** on a
panel whose baseline is not the spreadsheet's design, Reset all on one whose baseline is),
**Open the example design** (where the baseline is not the spreadsheet's), **Save design**,
**Load design** and the **Show spreadsheet details** toggle (decision C-12); a click on an item
closes the menu. The bar has no Equation panel button (decision C-11). The status line of a file
or a link stays under the bar (the plan's decision D-22). The toggle (slice 2, S2-5:
off by default; a view choice, in no design file, share link or undo step, kept by loads, undo
```

Replace:

```markdown
  `docs/superpowers/plans/2026-10-03-magcoupling-xlsx-export.md`, decisions X-1 to X-12): "Export
  spreadsheet" in the header and "Export XLSX" beside the CSV and JSON exports save the design
  shown as `magcoupling-results.xlsx`, laid out as the page. Sheets: **Summary** (the title, the
```

with:

```markdown
  `docs/superpowers/plans/2026-10-03-magcoupling-xlsx-export.md`, decisions X-1 to X-12): "Export
  spreadsheet" in the header (since BL-085 **Export** on the top bar) and "Export XLSX" beside the CSV and JSON exports save the design
  shown as `magcoupling-results.xlsx`, laid out as the page. Sheets: **Summary** (the title, the
```

Replace:

```markdown
  design sums. A click opens the value in the
  **Equation panel**, docked at the bottom of the centre region and closed until then (its header
  button toggles it; M43-1, M43-2; it opens at 45 % of the region's height, at least
  `MIN_PANEL_HEIGHT` 120 points, and its drag stops `VIEW_STRIP` 160 points below the region's
```

with:

```markdown
  design sums. A click opens the value in the
  **Equation panel**, docked at the bottom of the centre region and closed until then (since
  BL-085 no button on the bar opens it: a value's click or an Explain entry's Equation link does,
  and its Close button closes it, decision C-11; M43-1, M43-2; it opens at 45 % of the region's height, at least
  `MIN_PANEL_HEIGHT` 120 points, and its drag stops `VIEW_STRIP` 160 points below the region's
```

Replace:

```markdown
  `explorer::live_value`), "no prediction" where there is none (a value that is not a finite number, or one computed
  from an out-of-range end effect), never in its first sentence or its title. The Explain toggle (off by default)
  shows under the open equation's value its reviewed note (`notes::note_for`; a note shown by id, a warning's or a
  start-here step's, goes through `explorer::reviewed_note`) with its diagram painted with egui (the square wave and
  the sum of its harmonics, the flux paths with and without back iron with each block lettered N or S, torque
```

with:

```markdown
  `explorer::live_value`), "no prediction" where there is none (a value that is not a finite number, or one computed
  from an out-of-range end effect), never in its first sentence or its title. Since BL-085 (decision C-11) the
  Explain panel draws the notes and the Equation panel's Explain toggle is gone: under the open equation the lesson
  line names its reviewed note after "Full lesson:" (`notes::note_for`), on its own row so a long title wraps, a link
  that selects the note
  (`explain::Selection::Note`), and every other note link on the page (a Why link, a warning's link, an Explain
  entry's lesson link) selects its note too; **Learn** on the top bar selects the list of the notes shown
  (`Selection::NoteList`, `explain::shown_notes`: every reviewed note, the paused clamp section's left out), each
  title a link to its note. A note's entry is the links to the equations it explains ("Explains",
  `explorer::explains_ui`: one opens the Equation panel and leaves the selection, C-25) and the note
  (`explorer::note_ui`; a note shown by id goes through `explorer::reviewed_note`) with its diagram painted with egui (the square wave and
  the sum of its harmonics, the flux paths with and without back iron with each block lettered N or S, torque
```

Replace:

```markdown
  the shaft joints' keyed shrink fits,
  each step opening its equation with Explain on (Previous, Next, Stop); the ferrite step's bar adds "(ferrite rings
  only)" (`explorer::FERRITE_ONLY`): its check reads n/a on NdFeB rings. The empty Equation panel names Start here
```

with:

```markdown
  the shaft joints' keyed shrink fits,
  each step opening its equation and selecting its note in the Explain panel (Previous, Next, Stop; spec 5.5,
  `Explorer::start` returns the step's note); the ferrite step's bar adds "(ferrite rings
  only)" (`explorer::FERRITE_ONLY`): its check reads n/a on NdFeB rings. The empty Equation panel names Start here
```

Replace:

```markdown
  only)" (`explorer::FERRITE_ONLY`): its check reads n/a on NdFeB rings. The empty Equation panel names Start here
  and Explain (TC-29b). Under an equation no reviewed note explains, or a value with no equation, Explain offers the
  next best read (`explorer::note_fallback`, TC-29a, TC-30): the dashboard row's own Why note or assumption, else
```

with:

```markdown
  only)" (`explorer::FERRITE_ONLY`): its check reads n/a on NdFeB rings. The empty Equation panel names Start here
  (TC-29b). Under an equation no reviewed note explains, or a value with no equation, the lesson line offers the
  next best read (`explorer::note_fallback`, TC-29a, TC-30): the dashboard row's own Why note or assumption, else
```

Replace:

```markdown
  warnings that fire show at the top of the dashboard in their severity's colour, each with a
  "Why: <note>" link that opens its reviewed note on its own in the Equation panel, with links to
  the equations it explains.
- **Session** (`session.rs`, `history.rs`): undo and redo (buttons, Ctrl+Z, Ctrl+Shift+Z, Ctrl+Y)
```

with:

```markdown
  warnings that fire show at the top of the dashboard in their severity's colour, each with a
  "Why: <note>" link that shows its reviewed note in the Explain panel (since BL-085), with links
  to the equations it explains.
- **Session** (`session.rs`, `history.rs`): undo and redo (buttons, Ctrl+Z, Ctrl+Shift+Z, Ctrl+Y)
```

Replace:

```markdown
  a design file; a share link copied to the clipboard. A link the web app opens at start-up is
  the session's start (`open_share_payload`: the first Undo keeps it). The web app also writes the design into the page's address (UX overhaul slice A, design section 7.2): after every settled edit, undo, redo, load or opened example design, `MagcouplingApp::ui` rewrites the address with `history.replaceState` to `app::address_of` the settled design (`MagcouplingPanel::settled_design`, the undo history's current state). It waits while a key is held: a held Ctrl+Z or Ctrl+Y undoes or redoes a step at every key repeat, and the whole run gives one write at the key's release. A write the browser refuses (Safari refuses more than about 100 in 10 seconds) is asked again on the next frame. The address written is the clean address at the starting design and the link Copy share link copies for any other design, so a reload or a bookmark brings the design back, and the browser's Back still leaves the tool (Undo is the tool's). Copy share link copies the full link at the starting design too. A link that cannot be read changes nothing, says why in the header, and leaves the address as it was until the first edit. The native app writes nothing. The Size tab's Minimum OD is a view preference, not a design input: a reload does not bring it back. A host with its own undo
  can keep the keys (`set_keyboard_shortcuts(false)`: M5's linkage window). One format for files and links: JSON
```

with:

```markdown
  a design file; a share link copied to the clipboard. A link the web app opens at start-up is
  the session's start (`open_share_payload`: the first Undo keeps it). The web app also writes the design into the page's address (UX overhaul slice A, design section 7.2): after every settled edit, undo, redo, load or opened example design, `MagcouplingApp::ui` rewrites the address with `history.replaceState` to `app::address_of` the settled design (`MagcouplingPanel::settled_design`, the undo history's current state). It waits while a key is held: a held Ctrl+Z or Ctrl+Y undoes or redoes a step at every key repeat, and the whole run gives one write at the key's release. A write the browser refuses (Safari refuses more than about 100 in 10 seconds) is asked again on the next frame. The address written is the clean address at the starting design and the link Share copies for any other design, so a reload or a bookmark brings the design back, and the browser's Back still leaves the tool (Undo is the tool's). Share copies the full link at the starting design too. A link that cannot be read changes nothing, says why in the header, and leaves the address as it was until the first edit. The native app writes nothing. The Size tab's Minimum OD is a view preference, not a design input: a reload does not bring it back. A host with its own undo
  can keep the keys (`set_keyboard_shortcuts(false)`: M5's linkage window). One format for files and links: JSON
```

Replace:

```markdown
  opening.
- **Baseline** (UX overhaul slice A, BL-077; design `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, section 7.1): a panel holds one baseline design, set when it is built and fixed for the session. `MagcouplingPanel::with_baseline(inputs)` opens on it; `MagcouplingPanel::new()` keeps the spreadsheet's values as its baseline. On screen "the default" is the baseline's value: the changed dot, a row's reset and its hover, the "Default:" line of an input's tooltip, Reset all and the Equation panel's "changed" tag compare with it. The spreadsheet export's "Changed from default" column, the assumptions banner and the dots of the Assumptions view still compare with the spreadsheet's values. No undo, redo, file, link or opened design moves the baseline, and it is in no design file, link or undo step. On a panel whose baseline is not the spreadsheet's design the header shows **Open the example design** after Reset all: it opens the spreadsheet's values as one undo step (its hover: "The values from the original spreadsheet."), and Reset all still returns to the baseline. A panel built with `new()` has no such button: there Reset all already opens the example design.
- **Sizing** (`sizing.rs`, Addendum A1; the Size tab's Auto since slice 1): Auto on the Size tab is Torque -> Magnets (the Key design group shows
```

with:

```markdown
  opening.
- **Baseline** (UX overhaul slice A, BL-077; design `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, section 7.1): a panel holds one baseline design, set when it is built and fixed for the session. `MagcouplingPanel::with_baseline(inputs)` opens on it; `MagcouplingPanel::new()` keeps the spreadsheet's values as its baseline. On screen "the default" is the baseline's value: the changed dot, a row's reset and its hover, the "Default:" line of an input's tooltip, Reset all and the Equation panel's "changed" tag compare with it. The spreadsheet export's "Changed from default" column, the assumptions banner and the dots of the Assumptions view still compare with the spreadsheet's values. No undo, redo, file, link or opened design moves the baseline, and it is in no design file, link or undo step. On a panel whose baseline is not the spreadsheet's design the Design menu shows **Open the example design** under the reset, which it names **Reset to the starting design**: it opens the spreadsheet's values as one undo step (its hover: "The values from the original spreadsheet."), and the reset still returns to the baseline. A panel built with `new()` has no such item and its reset keeps the name Reset all: there it already opens the example design.
- **Sizing** (`sizing.rs`, Addendum A1; the Size tab's Auto since slice 1): Auto on the Size tab is Torque -> Magnets (the Key design group shows
```

Replace:

```markdown
`magcoupling explorer: ` (the equation registry built in the browser), then
clicks "Load design" once (found in a screenshot), uploads a pinned design file through rfd's web
picker and checks `magcoupling: loaded a design file`; then it clicks the check row "Hot-low torque" and
```

with:

```markdown
`magcoupling explorer: ` (the equation registry built in the browser), then
opens the Design menu and clicks its "Load design" (found in screenshots; since BL-085 the first screenshot also checks the top bar's Undo, Redo, Design, Share, Export and Learn, and `details_toggle` the Design menu's items), uploads a pinned design file through rfd's web
picker and checks `magcoupling: loaded a design file`; then it clicks the check row "Hot-low torque" and
```

Replace:

```markdown
reads its Explain entry down to "Full lesson:" (`check_entry`), and opens a teaching note
through that lesson link (`lesson_link_opened`: the Equation panel shows "What to change" and the closed "Where this
comes from" fold), hovers the first Key design row on the Geometry tab (the tooltip shows the help and its default
```

with:

```markdown
reads its Explain entry down to "Full lesson:" (`check_entry`), and opens a teaching note
through that lesson link (`lesson_link_opened`: the Explain panel shows "What to change" and the closed "Where this
comes from" fold), hovers the first Key design row on the Geometry tab (the tooltip shows the help and its default
```

Replace:

```markdown
comes from" fold), hovers the first Key design row on the Geometry tab (the tooltip shows the help and its default
and no path or spreadsheet cell, the header's "Show spreadsheet details" toggle being off, as the first screenshot
checks); then (slice 3) it opens the link again with a timer on every frame the app draws (eframe asks for an animation frame at every display refresh; the timer keeps those that make a WebGL draw call), clicks the Hot-low tile and checks the torque card's heading with its Why link, caption, Auto line and four rows in its Explain entry (the Overall length row held by Auto among them), and drags the Magnet face gap slider's handle, moving the pointer 60 times, 50 ms apart, alternately 3 px right and 3 px left of where it pressed: the drag's drawn-frame count, median, p95 and slowest frame are recorded on every run, and with `frame_check: true` (the release run) a drag that moved the gap, at least 30 frames and a p95 within one 60 fps frame (16.7 ms) are required; last (slice A, `address_follows`) it reads the address, which the app rewrote after the drag's settled edit, reloads the page and checks that the address is the same and the edited design is back (the one check of the `history.replaceState` call); `frame_check` is `true`, `false` or absent (any other value stops the run before its first step), and the calculator's result records it, so a run shows whether its check applied; the script checks the calculator agent's `passed` again from the agent's own fields (every check its schema requires true, the drag's two judged by the frame rule, no console error listed), and a step whose agent returns nothing reads every field its schema requires as false or empty; all with zero console
```

with:

```markdown
comes from" fold), hovers the first Key design row on the Geometry tab (the tooltip shows the help and its default
and no path or spreadsheet cell, the Design menu's "Show spreadsheet details" toggle being off, as `details_toggle`
checks); then (slice 3) it opens the link again with a timer on every frame the app draws (eframe asks for an animation frame at every display refresh; the timer keeps those that make a WebGL draw call), clicks the Hot-low tile and checks the torque card's heading with its Why link, caption, Auto line and four rows in its Explain entry (the Overall length row held by Auto among them), and drags the Magnet face gap slider's handle, moving the pointer 60 times, 50 ms apart, alternately 3 px right and 3 px left of where it pressed: the drag's drawn-frame count, median, p95 and slowest frame are recorded on every run, and with `frame_check: true` (the release run) a drag that moved the gap, at least 30 frames and a p95 within one 60 fps frame (16.7 ms) are required; last (slice A, `address_follows`) it reads the address, which the app rewrote after the drag's settled edit, reloads the page and checks that the address is the same and the edited design is back (the one check of the `history.replaceState` call); `frame_check` is `true`, `false` or absent (any other value stops the run before its first step), and the calculator's result records it, so a run shows whether its check applied; the script checks the calculator agent's `passed` again from the agent's own fields (every check its schema requires true, the drag's two judged by the frame rule, no console error listed), and a step whose agent returns nothing reads every field its schema requires as false or empty; all with zero console
```

- [ ] **Step 2: Check.** `git diff --stat` lists the nine files and no other; `python -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ['docs/ai/backlog.yaml', 'docs/ai/02-system.yaml', 'docs/ai/03-structure.yaml', 'docs/ai/04-memory.yaml']]"` prints nothing; `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3` still reads `1078 passed`.

- [ ] **Step 3: Commit** the nine files: `docs(magcoupler): slice B stage 2 part 2a docs sweep: README, docs/ai, the register, owner question 50, the design's header, BL-085 in progress (BL-085 T5)`.

### Task 6: Release: the full gate, gui-smoke, the reviews, one fix wave, the merge and the push

**Model:** the `fable` controller; the text reviewer and the whole-branch reviewers are fresh-context `fable` agents; git steps run on `haiku` at low effort; the gui-smoke workflow runs its own agents on `sonnet`.

**Files:**
- Modify: `docs/ai/backlog.yaml` (Step 7)

- [ ] **Step 1: Preconditions.**

  ```bash
  git status --short
  git diff --stat main...HEAD -- magcoupling-rs/src/engine magcoupling-rs/tests reference linkage-sim-rs/src linkage-sim-rs/web
  ```

  Expected: a clean tree; the second command prints nothing.

- [ ] **Step 2: The full gate.** From PowerShell at the worktree's root:

  ```powershell
  $env:MAGCOUPLING_PYTHON = "C:\Users\Cole\source\repos\linkage_simulation\reference\magcoupling-py\.venv\Scripts\python.exe"
  & 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/gate.sh --full
  git checkout -- docs/chebyshev_lambda
  ```

  Expected: the last line is `GATE PASS`; gate 12 printed `oracle Python:` and not a SKIP line; gate 13 printed `all 77 checks hold`; gate 14 built both web bundles. If a gate fails, fix the cause on the branch, red test first, and run this step again. Never edit `gate.sh` to pass.

- [ ] **Step 3: gui-smoke, from the branch's own script.**
  1. Check that port 8080 is free (`netstat -ano | findstr :8080` prints nothing): another branch's release uses it on this machine. Then, in the background: `& 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/serve_web.sh 8080`.
  2. Run the Workflow tool with `scriptPath` set to this worktree's `.claude/workflows/gui-smoke.js` and the arguments `{"frame_check": true}`. The session's `gui-smoke` skill runs the main checkout's script, which reads the old top bar: do not use it for this run.
  3. Expected: `passed: true`; in `magcoupling`: `starting_design`, `look`, `first_view`, `details_toggle`, `design_file_loaded`, `check_entry`, `lesson_link_opened`, `tooltip_plain`, `torque_card`, `address_follows`, `drag_frames_within_60fps` all `true`, `console_errors: []`. Keep the result's JSON in `.superpowers/sdd/2026-11-22-magcoupler-overhaul-b2p2a/smoke-local.json` and the run's first screenshot beside it.
  4. Stop the server: `TaskStop` on the background task that runs it, or `Stop-Process -Id <pid>` for the PID `netstat -ano | findstr :8080` shows. Check again that the port is free.

  A failure of `starting_design`, `details_toggle`, `design_file_loaded` or `lesson_link_opened` is a finding about the build until shown otherwise: open the page by hand and look at the bar, the menu and the note. A check that fails because the agent could not read a label is reported, not waved through.

- [ ] **Step 4: The text review** (fresh context, `fable`; the design's 8.3). It reads the new strings (`DESIGN_MENU`, `SHARE`, `EXPORT`, `LEARN`, `RESET_TO_START`) and the shortened `EMPTY_TEXT` against the plain-words guard and the design's 4.2, and checks that no note's text changed (`git diff main...HEAD -- magcoupling-rs/src/engine` is empty) and that every way to a note on the page lands in the Explain panel. Its findings join Step 5's fix wave. Record it in `docs/analyses/2026-11-22-magcoupler-slice-b2p2a-review.md` (committed in Step 7).

- [ ] **Step 5: The independent whole-branch review** (fresh context, `fable`, three lenses in parallel, each with a skeptic on every finding that is not a suggestion). Give each `git diff main...HEAD`, the design, this plan and Step 3's screenshot, and none of the builders' notes.
  1. **Conformance:** the design's 4.2, 5.1, 5.2 and 5.5 for notes, C-11, C-12, tests 9 and 10 of 8.2 where they fall here, item by item against the diff; the decisions D-20 to D-33, each what the code does and none hiding a choice the owner should see; the nothing-lost table, row by row against the code.
  2. **Regressions:** no engine number moves; nothing reachable today is unreachable (every action of the old bar, every note, the next-best-read lines, the walk); no existing assertion about behaviour that stays was weakened (the list "Existing tests this plan edits"); the anti-goals of 2.3.
  3. **What a person sees:** a fresh render at 1024 x 700, 1280 x 800 and 1920 x 1080 with the menu open, Learn's list, a note with its diagram, the walk's first step, and at 1024 x 700 every step of the walk (plan review UXP-1): nothing cut or overlapping beyond BL-086's known cuts (D-33); the bar reads as six words; crimson only on the brand mark and the active tab.

  One fix wave on `fable` for what Steps 4 and 5 find, each fix with its red test first; the reviewers re-read the fixes. Record the review in `.superpowers/sdd/2026-11-22-magcoupler-overhaul-b2p2a/final-review.json` (gitignored).

- [ ] **Step 6: If Step 4 or 5 changed a file, run Steps 2 and 3 again.** The merge needs a gate and a smoke run of the tree that is merged.

- [ ] **Step 7: Close the item and merge.** In `docs/ai/backlog.yaml` (the day's date; say what the reviews found):

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2a (Tasks 1 to 4 of the plan); the full gate, the smoke run and the whole-branch review are its Task 6. Filed
```

with:

```yaml
  status: fixed
  notes: "FIXED 2026-11-2X: built on branch magcoupling/overhaul-b2-part2a; GATE PASS with --full, the gui-smoke run passed with starting_design, details_toggle and lesson_link_opened true, the text review and the independent whole-branch review approved it. Filed
```

  (Write the day's date in place of `2026-11-2X`.) Commit it with the review record (`docs(magcoupler): BL-085 fixed: the gate, the smoke run and the reviews passed`), then merge the branch into `main` with a merge commit whose message lists the tasks' commit subjects. Remove the worktree with `git worktree remove`, never `git worktree prune`.

- [ ] **Step 8: Push, then report** (the owner's ruling of 2026-11-22: each stage goes live once the gate, the smoke run and the reviews pass, with a report afterwards; no stop). Push `main` (it deploys the site). When the `deploy-web.yml` run is green, run the smoke workflow from `main`'s script with `{"url": "https://colesorkness.com"}` and report the result; if it fails, say so plainly: the site is live with the change. The report to the owner holds: what changed on screen, in three sentences; two pictures (the first view with the menu open, a note in the Explain panel); the decisions D-20 to D-33, one line each; owner question 50; part 2b's list; what was not verified (Safari and Firefox).

## What this plan covers of the design

| Design item | Where |
|---|---|
| 4.2 The top bar: the mark and the title; Undo, Redo; Design; Share; Export; Learn | Tasks 2, 3 |
| 4.2 Reset all becomes "Reset to the starting design" in the Design menu | Task 2 (D-21 for a panel at the spreadsheet's baseline) |
| 4.2 Save design, Load design, Open the example design in the Design menu | Task 2 |
| 4.2 Copy share link becomes Share; Export spreadsheet becomes Export; the Results tab keeps its exports | Task 2 |
| 4.2 The Equation panel toggle leaves the bar (C-11) | Task 2 |
| 4.2 Show spreadsheet details in the Design menu (C-12) | Task 2 |
| 4.2 The assumptions banner's row under the bar | Unchanged |
| 4.2 The status line at the bar's right end | Kept under the bar (D-22, owner question 50) |
| 4.2 Learn lists the shown notes by title in the Explain panel; a title opens the note there | Task 3 (D-25) |
| 4.3 The Size column's folds and lines in the Explain panel | Part 2b |
| 4.4 The callout list to the Geometry tab; the Plots tab with its picker | Part 2b |
| 5.1 A note and the note list as selections; Esc returns the orientation | Task 3 |
| 5.2 A note link selects the note | Task 3 (D-24) |
| 5.2 Inputs, dimension tags, table rows, callout rows as selections | Part 2b |
| 5.3 A note's entry: the note's own form; "Where this comes from" | Task 3 (the note as reviewed, its "Explains" links at its head: a departure from 5.3's order, D-24) |
| 5.3 "Spreadsheet cell" foot link | Part 2b |
| 5.5 Each step of the walk selects its note; the Equation panel changes only through its own controls or an Equation link | Task 3 (D-23, D-26) |
| 8.2 Test 8 (word budget) | Task 3 as a ceiling of 430 (D-29); 250 in part 2b |
| 8.2 Test 9 (nothing lost): the top bar's actions, the Learn list | Tasks 2, 3 and the nothing-lost table; the full table-driven test in part 2b |
| 8.2 Test 10 (selection): a note, the walk | Task 3 |
| 8.3 Text rules | Tasks 2, 3 (plain words, glyphs), Task 6 Step 4 |
| 9.1 Docs with the slice | Task 5 |
| 9.2 The slip-heating readout's unit | Task 1 (D-27) |
| 9.4 Release | Task 6 (no stop: the owner's ruling of 2026-11-22) |
| Stage 1's review suggestions S-3 (buttons without a box) | Task 2 (D-28); E-4, S-1, S-4 and C-3 in part 2b |
