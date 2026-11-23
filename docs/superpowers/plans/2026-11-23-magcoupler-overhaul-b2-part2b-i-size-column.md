# Magcoupler UX overhaul, slice B, stage 2, part 2b-i (the Size column): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ] `) syntax for tracking.

**Goal:** The Size column becomes the design's (4.3): each control is its label and its slider, and under the controls two folds, **Detailed inputs** and **Assumptions**, both closed. What the column drew under its controls (every hint line, the what-moves-what lines such as "Set by: cap 42.8, cup 41.3, boss 22.0 mm", the range floor line, the Why links and the footnote on what the controls move) moves into the Explain panel: touching a control selects it, and its entry shows those lines with the control's row of the torque card, the checks whose fixes name it and its lesson (5.2, 5.3). The first view falls from 427 words to 272. Nothing that is on screen today is lost (the nothing-lost table).

**Architecture:**
- **One more kind of selection.** `explain::Selection::Input(key)`: a row of the Size column, by its key. `size_controls::SIZE_ROWS` lists the six rows (Required torque, Minimum OD, Outside diameter, Overall length, Magnet face gap, Magnets per ring) with their label, the input the slider writes, their torque-card control and their lesson. A row that is one input is keyed by its input path; the other three by the hint key the column already used (`plain_text::SIZE_TAB_HINT_KEYS`), so part 2b-ii's Detailed inputs rows can select the same entry for the same input (decision D-35).
- **The entry reads what the column drew.** `explain::input_ui` draws, under the design's headings: the value and the label; "What it is" (`size_controls::row_help`, the label's hover); "What sets it" (`hint_for`, the `Consequences` lines, `size_controls::floor_line`, `LINKED_HOUSING`); "What a change does" (`TorqueCard::control_lines`, the card's caption, its Auto line while Auto is on, and the control's row); "Checks it touches" (the guides whose fixes on the design shown name the row's input, each a link); "Watch out" (`guidance::input_trade_offs`, the "pulls two ways" sentence of the checks' own entries, factored out of `trade_offs`); the foot ("Full lesson:" and the note, and with the spreadsheet details "Spreadsheet cell:" and the cell).
- **The column touches, the panel selects.** `size_controls::controls_ui` draws labels (`row_label`, clickable) and sliders and reports the row touched (`touched`: a click, a drag's start, a change, the focus arriving). `requirement_ui` does the same for Required torque. The panel sets the selection; the Size column no longer asks the `ConsequenceCache` (the Explain panel does, for the Hot-low tile and a Size row), so `size_tab_consequences` goes.
- **Unchanged:** the engine, the exports, the top bar, the tiles, the check list, the check, tile and note entries, the centre tabs, the Detailed inputs inside, the Equation panel, the gui-smoke prompt, the linkage app.

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32, egui_plot 0.33; the headless egui harness (`magcoupling-rs/src/gui/test_support.rs`, the panel tests' `Harness`, the app tests' `frame_on`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. This part is its section 4.3 (the left column), the input column of 5.3's table and the inputs of 5.1 and 5.2, tests 8 (as a ceiling, decision D-44), 9 (the Assumptions view reachable from the opening tab), 10 (a click on a Size control selects it; a hover changes nothing) and 11 (each size control's entry holds the lines the left column shows today and its torque-card row) of section 8.2, and the docs and release rules of section 9. Executors read the spec's sections 2.3, 4.3, 5, 8.2, 8.3 and 9 beside this plan. Part 2a's plan (`docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`, "Why a part, and what follows") and BL-085's notes set the scope of part 2b; this plan cuts it again (decision D-34). Backlog item: **BL-090** (`docs/ai/backlog.yaml`, filed with this plan).

**How this plan was made.** The part was built once as a prototype on a copy of `main` at `dea5683` (BL-085 merged and closed), one commit per task, then BL-090 filed and the docs swept. Every block below was generated from those commits' diffs (a code task's red step is its test hunks, its green step the rest), then applied, in this plan's order, with the apply tool, to a fresh copy of the same base with BL-090 filed. Each block applied once; each red step failed as its "Expected" says; the result equals the prototype file for file (line endings aside) and passes `cargo test` with and without the app feature (1094 lib tests with it against 1084 on the base), both native clippy runs after each task, both wasm32 clippy runs of Task 2's step 4, `cargo fmt --check` and the gui-smoke verdict test (all 77 checks hold). Each count after a task is the replay's. Not run for this plan: gates 1 to 3 (linkage-sim-rs, untouched), gate 12 (the Python oracle), the release WASM build, the browser and the pictures. Task 4 runs them.

**Where this part applies.** On `main` at `dea5683` with this plan's commit (the plan and BL-090), and nothing else that touches `magcoupling-rs/` or Task 3's nine docs. If a block does not match, stop and report: the controller regenerates the blocks (the planner's tools are listed under "Before Task 1"), it does not adapt a block in place.

## Why a part, and what follows

BL-085's part 2b holds the Size column's move into the Explain panel, three more kinds of selection with the callout list's move, the Plots tab, the word budget of 250, the controls' outlines, a second face, the page shell, and BL-086, BL-088 and BL-089 where they fit. Each touches `panel.rs` (10,600 lines) and its tests; together they are again more than a reviewer can hold. This plan cuts part 2b in three (decision D-34) and is the first. It changes the Size column and the Explain panel and nothing else, ships on its own and loses nothing: every line the column drew is one click away, in its control's entry.

**What follows, each with its own plan and backlog item:**

**Part 2b-ii (more selections and the word budget):**
1. Dimension tags, results-table rows and callout rows as selections (5.1, 5.2); with the tags clickable the callout list leaves the opening tab for the Geometry tab (4.4; part 1's decision D-19), and the three dashboard rows of part 1's D-17 get their one-click home.
2. The Detailed inputs' rows as selections, keyed as this part keys the Size column's (D-35): a click on a row's label selects its input, and the trace it starts today becomes the entry's link "Highlight the results this drives" (5.2). This part leaves the label's trace as it is (D-36).
3. The "Spreadsheet cell" foot of a result's and a check's entry (this part gives an input's, D-41).
4. The word budget of 250 at 1280 x 800 (goal G-1, test 8): the callout list's move takes the first view from 272 under it. Which copies of a Start here step's title go during the walk (part 2a's D-32).
5. The full table-driven nothing-lost test of test 9 (every `CentreView`, every input, every `DASHBOARD` row's home).
6. gui-smoke reads an input's entry and a tag's (this part leaves the prompt as it is, D-45).
7. The left column no longer changes with the centre tab (4.3): after this part the inputs side still swaps on the other tabs.
8. A fix link selects and focuses its input (5.2); this part opens the fold that holds it (D-48) and selects nothing.

**Part 2b-iii (the Plots tab and the look):**
1. The Plots tab with its picker (C-9), the plots' red reference lines on a passing design (stage 1 review C-3) and egui_plot's fading tick numbers (E-7).
2. The controls' outlines (stage 1 review E-4, R-6), a SemiBold face for the title and the headings (S-1), the page shell's loading line (S-4).
3. BL-086 (the Equation panel's term rows during the walk), BL-088 (the note drawing out of `explorer.rs`) and BL-089 (the Equation panel's empty text).

BL-083 (the Hot-low tile's gap at a hair) and BL-084 (the band torques' help rows) stay open: the design gives neither text (decision D-47, as part 2a's D-30).

## What the prototype showed

1. **Words.** At 1280 x 800 the first view draws 427 words on `main` and 272 after this part (the app test's count of every text shape drawn in view, part 1's measure): the column's hint, what-moves-what and floor lines, its Why links and its footnote are most of the cut. The callout list under the drawing is most of what stands between 272 and the design's 250 (part 2b-ii).
2. **Both folds on the first view.** With the lines gone the column is short: at 1024 x 700, 1280 x 800 and 1920 x 1080 both fold headings, Detailed inputs over Assumptions, are drawn whole on the first view (part 1 held Detailed inputs there only from 800 points tall). The app test now holds both at every size.
3. **A label in a wrapping row starts at the row's start.** An entry's title is its value then its label in one wrapping row; egui lays the label's galley out from the row's left edge with its first line indented, so the label's rect begins where the value's does. The test finds the label's place by its last character.
4. **Typing in Required torque while the Hot-low tile is selected.** The edit is a touch, so the Explain panel shows Required torque's entry and the torque card leaves (spec 5.2: touching an input selects it). The card still follows the edit: it shows again when the tile is clicked, and the cache keys on the design.
5. **"Pulls two ways" names more than one pair.** Under a 30 mm overall-length claim the magnet length is raised by the hot-low check and lowered by both the space claim and the gearbox input check: Overall length's entry draws both sentences, each once, as the checks' entries do.
6. **The title's number format.** The title uses the panel's number format (`format_value`): the starting design's Outside diameter reads "42.80 mm" in the entry where its slider's box reads to its step's decimals (D-43).
7. **No extra compute on a drag.** The Size column no longer asks the `ConsequenceCache`; the Explain panel asks it for the selected Size row as it asked for the Hot-low tile. A drag of the gap slider (which now selects the gap) still computes the consequences once per design drawn (`a_drag_computes_the_card_once_for_each_design_it_draws`, unchanged), and an idle frame none.

No pictures were drawn for this plan: Task 4's review renders the first view and a selected row at the three window sizes.

## Global Constraints

- **No engine number moves** (the design's 2.3). No file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` changes.
- **The linkage app is not touched** (O-1). No file under `linkage-sim-rs/src/` or `linkage-sim-rs/web/` changes.
- **The panel sets no theme and no font** (2.2, 6.3). Code under `magcoupling-rs/src/gui/` reads `ui.visuals()`.
- **No new explanatory text** (5.3). The entries draw texts the screen drew before (the hints, the what-moves-what lines, the floor line, `LINKED_HOUSING`, the controls' hovers, the torque card's heading, caption and rows, the "pulls two ways" sentence, the notes' titles). The one new on-screen string is `SPREADSHEET_CELL` ("Spreadsheet cell:", the design's own words), shown only with the spreadsheet details on. `MINIMUM_HOVER` is the Minimum OD label's hover, unchanged, now a constant. Each passes the plain-words guard and the glyph guard (Task 1).
- **Crimson is never text, never on a value, never in the check list** (C-5). An entry's link to a check draws the check's state mark and word, as a tile's entry does.
- **The selection is view state** (5.1): a Size row selected is in no design file, share link or undo step, as a check or a note is.
- **A hover selects nothing** (C-2): a control's label keeps its hover text and selects only on a click.
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
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2b/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Shell.** On this machine `bash` typed in PowerShell is WSL, which fails on the checkout's line endings. Shell scripts run through Git Bash: `& 'C:\Program Files\Git\bin\bash.exe' <script>`. Python scripts are written with the Write tool, not a here-document.
- **Never run `cargo fmt` inside `linkage-sim-rs`.** `magcoupling-rs` is fmt-clean; keep it so.
- **Docs ship with the code** (repository rule): Task 3 sweeps the README and `docs/ai/`, the register, the owner-question list and the design's status line.

## Model per task

CLAUDE.md section 5. BL-090 is `risk: mechanical` (decision D-46): the part shows no new number and makes no new physical claim; its one new string is "Spreadsheet cell:", and every line its entries draw is reviewed text the column drew. So the tasks that transcribe the plan's exact blocks run on `sonnet`, and every review and the release run on `fable`. A `sonnet` task that ends blocked, leaves the gate red or is rejected in review is redone on `fable` (escalate, never downgrade).

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | Each Size column row's Explain entry (`Selection::Input`, `input_ui`, `SIZE_ROWS`, `control_lines`, `input_trade_offs`) | `sonnet` |
| 2 | The Size column draws its controls and two folds; touching a control selects it | `sonnet` |
| 3 | Docs sweep | `sonnet` |
| 4 | Release: the full gate, gui-smoke, the text review, the independent whole-branch review, one fix wave, the merge and the push | `fable` controller; fresh-context `fable` reviewers; `haiku` for git; the smoke workflow's own agents on `sonnet` |

Each task's diff is reviewed by a fresh-context reviewer before the next task starts (`sonnet` for these mechanical diffs, CLAUDE.md section 5); it checks the diff against the task's blocks and nothing else. A task that ends blocked or leaves a test red stops the build and goes back to the controller.

## Review Focus

Six inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named.

1. **Typing in Required torque while the Hot-low tile's card is shown.** The edit selects Required torque (5.2) and the card leaves the Explain panel. The frame that applies the typed value draws the entry with the typed value, and an arrow key on a control draws the edited design's lines and card row in its own frame. Task 2: `a_size_control_s_edit_reaches_its_entry_in_its_own_frame`.
2. **A hover over a label.** A label shows its hover and selects nothing; a click selects; the keyboard's focus arriving on a slider (a Tab) selects; a drag of a slider selects. Task 2: `touching_a_size_control_selects_its_row_and_a_hover_selects_nothing`.
3. **A design with no size controls** (an odd magnet count) while Outside diameter's entry is selected. The entry draws no value and no card row, and keeps its help and its lines. Tasks 1 and 2: `a_size_row_s_title_leads_with_its_value`, `without_the_diameter_control_the_size_tab_still_links_from_magnets_per_ring`.
4. **A link to an assumption's row on the opening tab.** The Assumptions fold opens, not Detailed inputs, and frames the row. Task 2: `a_focused_input_opens_the_detailed_inputs_on_the_first_view`.
5. **A passing design.** No fix names any input: no entry draws "Checks it touches" or "Watch out". Task 1: `a_size_row_s_entry_names_the_checks_whose_fixes_name_its_input_and_a_click_selects_one`.
6. **One input that two failing checks pull opposite ways.** Each "pulls two ways" sentence once. Task 1: `a_size_row_s_entry_warns_when_two_failing_checks_pull_its_input_opposite_ways`.

## Decisions this plan takes

Each is a reading of the design where it was open, numbered after part 2a's D-20 to D-33 (one stage, one list). The owner may overrule any of them; Task 4 lists them in its report. The ones that need a row in the register have one (Task 3).

| Id | Decision | Why |
|---|---|---|
| D-34 | Part 2b is built in three; this plan is part 2b-i (the Size column's lines in the Explain panel, its six rows as selections, the Assumptions fold). Parts 2b-ii and 2b-iii are listed under "What follows". Register row A-25. | Part 2b entire is again more than one reviewer can hold (D-1's and D-20's reasoning). This part changes one column and the Explain panel, and loses nothing. The conservative answer to Q-4 is smaller releases. |
| D-35 | A Size row's selection key is its input path where the row is one input (Required torque, Magnet face gap, Magnets per ring) and its hint key otherwise (Minimum OD, Outside diameter, Overall length: `SIZE_TAB_HINT_KEYS`). | The design's selection is "an input"; Outside diameter and Overall length are sizes that write an input (the ring radius, the magnet length) without being it, and Minimum OD is a view preference. Keying a one-input row by its path lets part 2b-ii's Detailed inputs row of the same input select the same entry. |
| D-36 | Touching a control is a click on its label, a click or the start of a drag on its slider, a change (an arrow key, a typed value) or the keyboard's focus arriving on its slider or value box; a hover is not (C-2). A label is a Tab stop (a clickable egui label is focusable): the focus arriving on it selects nothing, and Space or Enter on it selects, as a click does. In the Detailed inputs fold a label's click still traces its input, as today. | The design's 5.2 list ("a click on its label, a drag of its slider, focus in its field"). The trace's move into an entry link is part 2b-ii, with the Detailed inputs' rows as selections. |
| D-37 | The line "Set by Auto; turn Auto off to edit" under the slider Auto locks stays under the slider. Register row A-26; owner question 51. | It says, at the slider, why a drag does nothing; the "Set by" lines the design moves are the what-moves-what lines ("Set by: cap ..."). |
| D-38 | `LINKED_HOUSING` ("Outside diameter moves the ring radius; ...") shows under "What sets it" in each of the four size controls' entries, not in Required torque's or Minimum OD's. | The design's 5.3 cell for an input's "What sets it" names the linked housing; the sentence speaks of those four controls. |
| D-39 | An input's "Checks it touches" lists the guides whose fixes on the design shown name its input (none on a passing design, which leaves the heading out); each is a link that selects the check, with its state's mark and word. Register row A-27. | The fix lists are per design and empty for a passing check (`all_fixes`); a list of every fix a guide could offer would need a new table and would name checks that cannot move. |
| D-40 | An input's "Watch out" is each "pulls two ways" sentence that names its input (`guidance::input_trade_offs`, the sentence the two checks' entries draw, factored out of `trade_offs`), each once. | The design's 5.3 cell; the sentence is reviewed text the check entries draw. |
| D-41 | With the spreadsheet details on, an input's foot draws "Spreadsheet cell:" and the cell (or `NO_SPREADSHEET_CELL`) for a row that is one input; Outside diameter, Overall length and Minimum OD draw none. It is a weak line, not a link. A result's and a check's entry get theirs in part 2b-ii. An input's foot draws no "Equation" link, although 5.3's input column lists one: an input has no equation of its own (the Equation panel opens at a result), and the trace becomes the input's link in part 2b-ii. | The two sizes have no cell of their own (the cell of the input their slider writes would name another quantity); a cell opens nothing, so it is a line. |
| D-42 | An input's "Full lesson:" shows for the rows whose Why link the column drew (Required torque, Outside diameter, Magnets per ring) with that note; the other rows leave the heading out. | 5.3: an entry leaves out a heading it has nothing for; "no note yet" is the honest text of the five result rows (5.3, slice F), not of an input. |
| D-43 | An entry's title leads with the row's value in the panel's number format (`format_value` with its unit): "42.80 mm" where the slider's box reads "42.8 mm". Minimum OD's value is the slider's lowest value at the person's preference. A design with no size controls draws no value for the three sizes. | 5.3: "Title, with the live value leading"; the panel's format is the one the Results tab and the Equation panel use. |
| D-44 | The first view at 1280 x 800 is held to at most 280 words (272 now, 427 before); the design's 250 is part 2b-ii's test. | Part 1's D-9 ceiling, lowered by what this part removed, so it cannot grow back. |
| D-45 | The gui-smoke prompt does not change. Its drag of the gap slider now also selects the gap (its entry shows in the Explain panel); no step reads a line this part moves. | The smoke's checks still hold; an input entry check joins with part 2b-ii's tag and row selections. |
| D-46 | BL-090 is `risk: mechanical`; its build tasks run on `sonnet`, its reviews and release on `fable`. | No new number or physical claim (the design's header). |
| D-47 | BL-083 and BL-084 are not folded in; BL-086, BL-088 and BL-089 are part 2b-iii's. | The design gives neither BL-083's nor BL-084's text (part 2a's D-30). The other three touch the Equation panel, which this part does not. |
| D-48 | A link that asks for an assumption's row on the opening tab opens the Assumptions fold; any other row's opens Detailed inputs. | The inputs side's own rule since M43-7 (an assumption's row is the Assumptions view's). |

## Nothing lost

Every text and control of the Size column (at `dea5683`) that this part moves, and where it is after. Anything not listed (the heading, the labels and sliders, Auto and its outcome line, the "Set by Auto" line, the "Size controls unavailable" line, Detailed inputs inside, every other column and tab) does not move. The "Test" column names the test that holds the new home.

| Today | After this part | Test |
|---|---|---|
| A hint line under each of the five controls (Minimum OD, Outside diameter, Overall length, Magnet face gap, Magnets per ring) | The control's entry, the first line under "What sets it" | `panel::tests::each_size_control_s_entry_draws_its_hint_under_what_sets_it`, `hint_lines_stay_inside_their_sides_in_a_narrow_window`; `explain::tests::each_size_row_s_entry_is_its_help_its_lines_its_card_row_and_its_lesson` |
| Under Outside diameter: "Set by: cap A, cup B, boss C mm" | Outside diameter's entry, under "What sets it" | `each_size_control_s_entry_draws_its_what_moves_what_lines_under_what_sets_it`, `size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter` |
| Under Overall length: "Magnets L mm + housing H mm" | Overall length's entry, under "What sets it" | `each_size_control_s_entry_draws_its_what_moves_what_lines_under_what_sets_it`, `size_tab_what_moves_what_lines_follow_each_edit` |
| Under Magnets per ring: the ring-size line (by the Auto state) and the fit line | Magnets per ring's entry, under "What sets it" | `size_tab_first_magnet_line_follows_the_auto_state`, `size_tab_smallest_diameter_line_searches_with_the_minimum_od`, `size_tab_magnet_slider_ends_show_only_the_neighbour_in_range` |
| "Shaft and housing range floor: X mm" under the size sliders | Minimum OD's entry, under "What sets it" | `explain::tests::each_size_row_s_entry_is_its_help_its_lines_its_card_row_and_its_lesson`; `size_tab_new_texts_have_glyphs_and_plain_words` |
| The Why links under Outside diameter and Magnets per ring ("Why: How the coupling makes torque") | Their entries' "Full lesson:" link to the same note; the torque card's heading keeps its own | `the_size_tab_s_why_links_open_their_notes`, `without_the_diameter_control_the_size_tab_still_links_from_magnets_per_ring` |
| The Why link beside Required torque | Required torque's entry's "Full lesson:" link to the same note | `the_size_tab_s_why_links_open_their_notes` |
| The footnote under the Detailed inputs fold (`LINKED_HOUSING`) | The four size controls' entries, under "What sets it" (D-38) | `explain::tests::each_size_row_s_entry_is_its_help_its_lines_its_card_row_and_its_lesson`, `size_tab_new_texts_have_glyphs_and_plain_words` |
| Each control's label hover (Minimum OD's typed in place) | The same hover (Minimum OD's now `MINIMUM_HOVER`), and the entry's "What it is" | `the_size_tab_s_control_labels_show_their_hovers` (unchanged); `explain::tests::the_size_rows_new_texts_have_glyphs_and_plain_words` |
| The torque card's row of each size control, in the Hot-low tile's card, under the card's Auto line while Auto is on (owner 5A) | Also in the control's entry, under "What a change does" (C-10's second half), with the caption and the same Auto line | `size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter`, `size_tab_what_moves_what_lines_follow_each_edit`, `size_tab_first_magnet_line_follows_the_auto_state`; `explain::tests::a_size_row_s_card_row_keeps_the_auto_line_while_auto_is_on` |
| The Assumptions view (the inputs side on the other tabs) | Also the Assumptions fold under Detailed inputs on the opening tab | `the_size_column_draws_its_controls_and_two_folds_and_no_line_about_them`, `a_focused_input_opens_the_detailed_inputs_on_the_first_view`; `app::tests::the_first_view_keeps_its_three_columns_its_tiles_and_the_fixed_3d_line_in_the_app_s_own_font` |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/explain.rs` | `Selection::Input`, `Selection::input`, `reads_consequences` for a Size row; `Shown::minimum_diameter_mm`; `SPREADSHEET_CELL`; `input_ui`, `row_value`, `sets_it_lines`, `check_link_ui` (from `result_ui`); `foot_ui` takes an optional lesson and a cell line; tests | 1 |
| `magcoupling-rs/src/gui/size_controls.rs` | `MINIMUM_HOVER`, `SizeRow`, `SIZE_ROWS`, `size_row`, `row_help`, `floor_line` (1); `touched`, `row_label`, `ControlsOutput::touched`; `controls_ui` draws labels and sliders only and takes no consequences (2) | 1, 2 |
| `magcoupling-rs/src/gui/torque_card.rs` | `TorqueCard::control_lines` (the caption, the Auto line, the row), `Row::text` | 1 |
| `magcoupling-rs/src/gui/guidance.rs` | `input_trade_offs`, `trade_off_text`, `trade_off_pairs`; the text review's dump lists the two strings (1); two doc comments (2) | 1, 2 |
| `magcoupling-rs/src/gui/test_support.rs` | `check_entry_texts` gives `Shown` its `minimum_diameter_mm` | 1 |
| `magcoupling-rs/src/gui/panel.rs` | `Shown`'s new field (1); `DETAILED_INPUTS`, `ASSUMPTIONS_FOLD`; `size_column_ui` with the touch and the two folds; `requirement_ui` without its Why link; `size_tab_consequences` gone; tests (2) | 1, 2 |
| `magcoupling-rs/src/gui/mod.rs` | `DETAILED_INPUTS`, `ASSUMPTIONS_FOLD` re-exported | 2 |
| `magcoupling-rs/src/app.rs` | The word ceiling at 280; both folds on the first view at every size | 2 |
| `magcoupling-rs/README.md`, `docs/ai/*`, the register, the owner questions, the design's header | The docs sweep | 3 |
| `docs/ai/backlog.yaml` | BL-090 in progress (3), fixed (4) | 3, 4 |

**Existing tests this plan edits, and how.** No assertion about behaviour that stays is weakened; where a line moved, the test selects its control and asserts the same line, once, in the entry.
- `panel::tests`, Task 2: `size_tab_draws_each_what_moves_what_line_under_its_control` becomes `each_size_control_s_entry_draws_its_what_moves_what_lines_under_what_sets_it` and `size_tab_draws_each_hint_line_under_its_control` becomes `each_size_control_s_entry_draws_its_hint_under_what_sets_it` (each line once, in order under "What sets it", and none in the column); `hint_lines_stay_inside_the_inputs_side_in_a_narrow_window` becomes `hint_lines_stay_inside_their_sides_in_a_narrow_window` (the Key design hints in the inputs side as before, the Size controls' in the right column); `size_tab_owner_case_at_8_magnets_...`, `size_tab_first_magnet_line_follows_the_auto_state`, `size_tab_what_moves_what_lines_follow_each_edit`, `size_tab_smallest_diameter_line_searches_with_the_minimum_od`, `size_tab_magnet_slider_ends_show_only_the_neighbour_in_range` and `size_tab_narrow_window_wraps_the_new_lines_inside_their_sides` read the lines in the controls' entries and the card lines in the Hot-low tile's (the same lines, the same counts, the same order; `size_tab_first_magnet_line_follows_the_auto_state` holds the Auto line of the state in Magnets per ring's entry as well as in the card); `size_tab_new_texts_have_glyphs_and_plain_words` draws the Hot-low tile's entry and each row's and checks the same texts plus the floor line and `LINKED_HOUSING`; `idle_frames_compute_the_card_once_and_an_edit_again` selects Outside diameter before it reads that row (the gap edit selected the gap); `the_size_tab_s_why_links_open_their_notes` clicks the entries' lesson links and the card's Why link and holds that the column draws none; `without_the_diameter_control_...` also holds the odd count's Outside diameter entry; `a_focused_input_opens_the_detailed_inputs_on_the_first_view` also holds that an assumption opens the Assumptions fold; `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` also collects each Size row's entry on the designs it loops over while the clamp section is paused (test 13 of 8.2). `a_required_edit_reaches_the_card_in_its_own_frame` becomes `a_size_control_s_edit_reaches_its_entry_in_its_own_frame`: a Required edit now selects Required torque, so the frame of the edit draws its entry (with the typed value), not the Hot-low tile's card; the same-frame property is held on Magnets per ring's lines and card row instead.
- `size_controls::tests::the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown`, Task 2: `controls_ui` is called without consequences.
- `explain::tests`, `test_support.rs`, Task 1: each `Shown` gives `minimum_diameter_mm: None`; the new helper `input_frames` takes the Auto state.
- `app::tests`, Task 2: the word test is renamed `the_first_view_draws_at_most_two_hundred_and_eighty_words_at_1280_by_800`; the first-view test holds both folds at every window size (it held Detailed inputs from 800 points tall).

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` after this plan has merged, create the branch `magcoupling/overhaul-b2-part2b-i` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  ```

  Expected: `test result: ok. 1084 passed; 0 failed; 6 ignored`. Another count means `main` is not the base the blocks were checked on: stop and report.

- [ ] **Step 3: Check the base.** `git log --oneline dea5683..main -- magcoupling-rs .claude/workflows/gui-smoke.js linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs docs/ai docs/analyses/2026-11-10-conservative-assumptions.md docs/analyses/2026-11-18-owner-questions.md docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` lists this plan's commit (it touches `docs/ai/backlog.yaml`) and nothing else. If it lists more, stop: the controller regenerates the blocks on that `main` with the planner's tools (gitignored, in the main checkout's `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2b/`: `gen_blocks.py`, the scripts that built each task's state, `frags.py` and `replay.py`) before Task 1.

## Tasks

### Task 1: Each Size column row's Explain entry

**Model:** `sonnet` (BL-090 is `risk: mechanical`; the task transcribes exact blocks).

`Selection::Input(key)` and its entry, drawn from what the column draws today; nothing selects it yet (Task 2 wires the touch), so the column is unchanged after this task. The torque card gives one control's row (`control_lines`); the "pulls two ways" sentence is factored out of `trade_offs` so an input's entry can name its own (`input_trade_offs`); `result_ui`'s check link becomes `check_link_ui`, which both entries call; `foot_ui` takes an optional lesson and a cell line.

**Files:**
- Modify: `magcoupling-rs/src/gui/explain.rs`, `magcoupling-rs/src/gui/guidance.rs`, `magcoupling-rs/src/gui/panel.rs` (one line), `magcoupling-rs/src/gui/size_controls.rs`, `magcoupling-rs/src/gui/test_support.rs`, `magcoupling-rs/src/gui/torque_card.rs`

**Interfaces:**
- Produces: `explain::{SPREADSHEET_CELL}`, `Selection::Input(&'static str)`, `Selection::input(self) -> Option<&'static str>`, `Shown::minimum_diameter_mm: Option<f64>`; `size_controls::{MINIMUM_HOVER, SizeRow, SIZE_ROWS, size_row, row_help, floor_line}`; `TorqueCard::control_lines(&self, Control, Option<FreeVariable>) -> Vec<String>` (the caption, the Auto line while Auto is on, the row, as `lines` orders them); `guidance::input_trade_offs(all, path) -> Vec<String>`.

- [ ] **Step 1 (red): the entries' tests.** The blocks add eight tests in `explain::tests` with their helpers `input_frames` and `input_texts`, give each test's `Shown` its new field, and add the two strings to the text review's dump.

**`magcoupling-rs/src/gui/explain.rs`**

Replace:

```rust
    use crate::gui::explorer::reviewed_note;
    use crate::gui::guidance::{Fixes, all_fixes, counted_guides, plain_text_violations};
```

with:

```rust
    use crate::gui::explorer::reviewed_note;
    use crate::gui::guidance::FACE_GAP;
    use crate::gui::guidance::{Fixes, all_fixes, counted_guides, plain_text_violations};
```

Replace:

```rust
    use crate::gui::readouts::ReadoutEvents;
    use crate::gui::test_support::{
```

with:

```rust
    use crate::gui::readouts::ReadoutEvents;
    use crate::gui::size_controls::{MINIMUM_HOVER, SIZE_ROWS};
    use crate::gui::test_support::{
```

Replace:

```rust
                    auto: None,
                    held: &|_| None,
                };
                asked = explain_ui(ui, selection, &shown, &mut readouts);
```

with:

```rust
                    auto: None,
                    held: &|_| None,
                    minimum_diameter_mm: None,
                };
                asked = explain_ui(ui, selection, &shown, &mut readouts);
```

Replace:

```rust

    /// What the entry of `check` draws on `inputs`.
```

with:

```rust

    /// Frames of the entry of the Size column's row `key` on `inputs` in `ctx`, with the Size
    /// tab's consequences at the Minimum OD `minimum`, Auto on `auto` (`None`: off) and the
    /// spreadsheet details `show_details`: a first one with no input, then one per entry of
    /// `events`. What the last drew and what it asked for.
    fn input_frames(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        key: &'static str,
        show_details: bool,
        minimum: Option<f64>,
        auto: Option<FreeVariable>,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, Option<Asked>) {
        let results = compute_all(inputs);
        let fixes = all_fixes(inputs, &results, false);
        let consequences = Consequences::of(inputs, &results, minimum, false);
        let mut last = None;
        for input in std::iter::once(Vec::new()).chain(events) {
            let mut readouts = Readouts::new(TermColors::default(), show_details);
            let mut asked = None;
            let output = sized_frame(ctx, egui::vec2(320.0, 4000.0), input, |ui| {
                let shown = Shown {
                    inputs,
                    results: &results,
                    fixes: &fixes,
                    clamps_paused: false,
                    consequences: Some(&consequences),
                    auto,
                    held: &|_| None,
                    minimum_diameter_mm: minimum,
                };
                asked = explain_ui(ui, Selection::Input(key), &shown, &mut readouts);
            });
            last = Some((output, asked));
        }
        last.expect("one frame at least")
    }

    /// What the entry of the Size column's row `key` draws on `inputs` ([`input_frames`]).
    fn input_texts(inputs: &DesignInputs, key: &'static str, show_details: bool) -> Vec<String> {
        let ctx = egui::Context::default();
        drawn_texts(&input_frames(&ctx, inputs, key, show_details, None, None, vec![]).0)
    }

    /// What the entry of `check` draws on `inputs`.
```

Replace:

```rust
                    held: &|_| None,
                };
```

with:

```rust
                    held: &|_| None,
                    minimum_diameter_mm: None,
                };
```

Replace:

```rust
                        held: &|_| None,
                    };
```

with:

```rust
                        held: &|_| None,
                        minimum_diameter_mm: None,
                    };
```

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
    fn a_size_row_s_selection_reads_the_consequences_and_is_no_check_or_tile() {
        // UX overhaul spec 5.1: an input is one more kind of selection. Its entry draws its
        // what-moves-what lines and its torque-card row, so the panel asks its cache for them.
        for row in SIZE_ROWS {
            let selection = Selection::Input(row.key);
            assert_eq!(selection.input(), Some(row.key));
            assert!(selection.reads_consequences(), "{}", row.label);
            assert_eq!((selection.check(), selection.tile()), (None, None));
        }
        assert_eq!(Selection::Nothing.input(), None);
    }

    #[test]
    fn each_size_row_s_entry_is_its_help_its_lines_its_card_row_and_its_lesson() {
        // UX overhaul spec 4.3 and 5.3, test 11 of 8.2 for the size controls: a row's entry is
        // its title, "What it is" (its label's hover), "What sets it" (the lines the column
        // drew under it: its hint, its what-moves-what lines, Minimum OD's floor, the footnote
        // on what the four size controls move), "What a change does" (its torque-card row,
        // under the card's heading and caption), then "Full lesson:" and the note its Why link
        // opened, in that order; a heading with nothing under it is left out ("Checks it
        // touches" and "Watch out" have their own test). At the failing example design and at
        // a passing one.
        use crate::gui::torque_card::CAPTION;
        for inputs in [DesignInputs::default(), passing_design()] {
            let results = compute_all(&inputs);
            let consequences = Consequences::of(&inputs, &results, None, false);
            let size = dimensions(&inputs).expect("size controls");
            for row in SIZE_ROWS {
                let ctx = egui::Context::default();
                let (output, _) = input_frames(&ctx, &inputs, row.key, false, None, None, vec![]);
                let texts = drawn_texts(&output);
                let has = |text: &str| texts.iter().any(|t| t == text);
                let label = row.label;
                assert!(has(label), "{label}: {texts:?}");
                assert!(has(row_help(row.key)), "{label}: {texts:?}");
                let mut lines: Vec<String> =
                    hint_for(row.key).map(str::to_owned).into_iter().collect();
                match row.key {
                    DIAMETER_HINT_KEY => lines.extend(consequences.diameter_line()),
                    LENGTH_HINT_KEY => lines.extend(consequences.length_line()),
                    MINIMUM_OD_HINT_KEY => lines.push(floor_line(size.minimum_diameter_range.min)),
                    key if key == FreeVariable::MagnetsPerRing.path() => {
                        lines.extend(consequences.magnet_lines(inputs.coupling.npole, None))
                    }
                    _ => {}
                }
                if row.card.is_some() {
                    lines.push(LINKED_HOUSING.to_owned());
                }
                assert_eq!(has(WHAT_SETS_IT), !lines.is_empty(), "{label}");
                for line in &lines {
                    assert!(has(line), "{label}: no {line:?} in {texts:?}");
                }
                let card = row
                    .card
                    .map(|control| consequences.card.control_lines(control, None))
                    .unwrap_or_default();
                assert_eq!(has(WHAT_A_CHANGE_DOES), !card.is_empty(), "{label}");
                assert_eq!(has(CARD_HEADING), !card.is_empty(), "{label}");
                if row.card.is_some() {
                    assert_eq!(card.len(), 2, "{label}: the caption and the row: {card:?}");
                    assert_eq!(card[0], CAPTION);
                }
                for line in &card {
                    assert!(has(line), "{label}: no {line:?}");
                }
                assert_eq!(has(FULL_LESSON), row.lesson.is_some(), "{label}");
                if let Some(id) = row.lesson {
                    let note = reviewed_note(id).expect("a Why link's note is reviewed");
                    assert!(has(&glyph_safe(note.title)), "{label}: {texts:?}");
                }
                let tops: Vec<f32> = [WHAT_IT_IS, WHAT_SETS_IT, WHAT_A_CHANGE_DOES, FULL_LESSON]
                    .into_iter()
                    .filter_map(|heading| text_rect(&output, heading))
                    .map(|rect| rect.top())
                    .collect();
                assert!(
                    tops.is_sorted(),
                    "{label}: the headings out of order: {tops:?}"
                );
                assert!(
                    text_rect(&output, label).expect("its title").top() < tops[0],
                    "{label}: the title leads"
                );
            }
        }
    }

    #[test]
    fn a_size_row_s_title_leads_with_its_value() {
        // Spec 5.3: "Title, with the live value leading", in the panel's number format: an
        // input's value with its unit; Outside diameter, Overall length and Minimum OD the sizes
        // their sliders set, Minimum OD at the person's preference where one is set. An odd magnet count has no size
        // controls: Outside diameter's entry has no value and no card row, and keeps its help
        // and lines.
        let inputs = DesignInputs::default();
        let size = dimensions(&inputs).expect("size controls");
        let mm = |x: f64| value_with_unit(&Value::Num(x), format_value(&Value::Num(x)), "mm");
        let gap = Value::Num(inputs.metal.face_gap_mm);
        for (key, minimum, want) in [
            (DIAMETER_HINT_KEY, None, mm(size.diameter_mm)),
            (LENGTH_HINT_KEY, None, mm(size.overall_length_mm)),
            (MINIMUM_OD_HINT_KEY, None, mm(slider_minimum(&size, None))),
            (MINIMUM_OD_HINT_KEY, Some(38.0), mm(38.0)),
            (
                FACE_GAP,
                None,
                value_with_unit(&gap, format_value(&gap), "mm"),
            ),
        ] {
            let ctx = egui::Context::default();
            let (output, _) = input_frames(&ctx, &inputs, key, false, minimum, None, vec![]);
            let label = size_row(key).expect("a row").label;
            let value = text_rect(&output, &want)
                .unwrap_or_else(|| panic!("{key}: no {want:?} in {:?}", drawn_texts(&output)));
            // The label follows the value on its row: a label in a wrapping row is laid out
            // from the row's start, so its last character marks where it is.
            let end = crate::gui::test_support::last_characters(&output, label)[0];
            assert!(value.right() < end.x, "{key}: the value leads");
            assert!(value.y_range().contains(end.y), "{key}: one line");
        }
        let odd = with_poles(7);
        assert!(dimensions(&odd).is_none());
        let texts = input_texts(&odd, DIAMETER_HINT_KEY, false);
        let value = |t: &String| t.ends_with(" mm") && t.starts_with(|c: char| c.is_ascii_digit());
        assert!(!texts.iter().any(value), "no value: {texts:?}");
        assert!(!texts.contains(&WHAT_A_CHANGE_DOES.to_owned()), "{texts:?}");
        assert!(
            texts.contains(&row_help(DIAMETER_HINT_KEY).to_owned()),
            "{texts:?}"
        );
        assert!(texts.contains(&LINKED_HOUSING.to_owned()), "{texts:?}");
    }

    #[test]
    fn a_size_row_s_entry_names_the_checks_whose_fixes_name_its_input_and_a_click_selects_one() {
        // Spec 5.3, "Checks it touches" for an input: the guides whose fixes name it, on the
        // design shown, each with its state's mark and word, a link that selects the check. At
        // the example design the running clearance's and the gearbox input's fixes name the face
        // gap; a passing design has no fixes, so no entry draws the heading. Minimum OD writes no input and names none.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let fixes = all_fixes(&inputs, &results, false);
        let named = |input: Option<&str>| -> Vec<&'static CheckGuide> {
            CHECK_GUIDES
                .iter()
                .zip(&fixes)
                .filter(|(_, list)| input.is_some_and(|i| list.iter().any(|&(p, _)| p == i)))
                .map(|(guide, _)| guide)
                .collect()
        };
        for row in SIZE_ROWS {
            let texts = input_texts(&inputs, row.key, false);
            let guides = named(row.input);
            assert_eq!(
                texts.contains(&CHECKS_IT_TOUCHES.to_owned()),
                !guides.is_empty(),
                "{}",
                row.label
            );
            for guide in guides {
                assert!(
                    texts.contains(&guide.label.to_owned()),
                    "{}: {}",
                    row.label,
                    guide.label
                );
            }
        }
        assert!(named(None).is_empty());
        let gap = named(Some(FACE_GAP));
        let first = gap
            .first()
            .expect("a failing check's fix names the face gap");
        let ctx = egui::Context::default();
        let (output, _) = input_frames(&ctx, &inputs, FACE_GAP, false, None, None, vec![]);
        let at = text_rect(&output, first.label)
            .expect("the check's link")
            .center();
        let (_, asked) = input_frames(&ctx, &inputs, FACE_GAP, false, None, None, click_at(at));
        assert_eq!(asked, Some(Asked::Select(Selection::Check(first.check))));
        for row in SIZE_ROWS {
            let texts = input_texts(&passing_design(), row.key, false);
            assert!(
                !texts.contains(&CHECKS_IT_TOUCHES.to_owned()),
                "{}",
                row.label
            );
        }
    }

    #[test]
    fn a_size_row_s_entry_warns_when_two_failing_checks_pull_its_input_opposite_ways() {
        // Spec 5.3, "Watch out" for an input: each "pulls two ways" sentence the two checks'
        // blocks draw, once. Under a 30 mm overall-length claim the hot-low torque needs longer
        // magnets and the space claim (and the gearbox input) shorter ones: Overall length's
        // entry says so, after its checks; the face gap, which no two checks pull apart, has no
        // "Watch out".
        let inputs = length_conflict();
        let results = compute_all(&inputs);
        let fixes = all_fixes(&inputs, &results, false);
        let length = size_row(LENGTH_HINT_KEY).expect("a row");
        let notes = input_trade_offs(&fixes, length.input.expect("an input"));
        assert!(!notes.is_empty());
        let ctx = egui::Context::default();
        let (output, _) = input_frames(&ctx, &inputs, LENGTH_HINT_KEY, false, None, None, vec![]);
        let texts = drawn_texts(&output);
        for note in &notes {
            assert_eq!(
                texts.iter().filter(|t| *t == note).count(),
                1,
                "{note}: {texts:?}"
            );
        }
        let rect = |text: &str| text_rect(&output, text).expect(text);
        assert!(rect(CHECKS_IT_TOUCHES).top() < rect(WATCH_OUT).top());
        assert!(rect(WATCH_OUT).bottom() <= rect(&notes[0]).top() + 0.5);
        assert!(input_trade_offs(&fixes, FACE_GAP).is_empty());
        assert!(!input_texts(&inputs, FACE_GAP, false).contains(&WATCH_OUT.to_owned()));
    }

    #[test]
    fn a_size_row_s_entry_shows_its_input_s_cell_only_with_the_spreadsheet_details() {
        // Spec 5.3's "Spreadsheet cell" foot, shown only with the details on: the cell of a row
        // that is one input (Required torque, Magnet face gap, Magnets per ring). Minimum OD,
        // Outside diameter and Overall length are sizes the spreadsheet has no cell for.
        for row in SIZE_ROWS {
            let cell = |texts: &[String]| -> Vec<String> {
                texts
                    .iter()
                    .filter(|t| t.starts_with(SPREADSHEET_CELL))
                    .cloned()
                    .collect()
            };
            assert!(cell(&input_texts(&DesignInputs::default(), row.key, false)).is_empty());
            let on = cell(&input_texts(&DesignInputs::default(), row.key, true));
            if row.input == Some(row.key) {
                let meta = crate::gui::inputs::InputCatalogue::get()
                    .entry(row.key)
                    .expect("an input")
                    .meta;
                let want = format!(
                    "{SPREADSHEET_CELL} {}",
                    meta.cell.unwrap_or(NO_SPREADSHEET_CELL)
                );
                assert_eq!(on, [want], "{}", row.label);
            } else {
                assert!(on.is_empty(), "{}: {on:?}", row.label);
            }
        }
    }

    #[test]
    fn a_size_row_s_card_row_keeps_the_auto_line_while_auto_is_on() {
        // Owner ruling 5A, as the Hot-low tile's card draws it: while Auto is on, the Auto line
        // stands between the caption and a row, since an edit of the row makes Auto re-solve.
        // With Auto on the magnet length, Outside diameter's entry draws that line once,
        // between the caption and its row; with Auto off, none.
        use crate::gui::torque_card::{AUTO_HOLDS_LENGTH, CAPTION, Control};
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let card = Consequences::of(&inputs, &results, None, false).card;
        let auto = Some(FreeVariable::AxialLength);
        let lines = card.control_lines(Control::OutsideDiameter, auto);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert_eq!(lines[..2], [CAPTION, AUTO_HOLDS_LENGTH]);
        let ctx = egui::Context::default();
        let (output, _) = input_frames(&ctx, &inputs, DIAMETER_HINT_KEY, false, None, auto, vec![]);
        let texts = drawn_texts(&output);
        let tops: Vec<f32> = lines
            .iter()
            .map(|line| {
                let n = texts.iter().filter(|t| *t == line).count();
                assert_eq!(n, 1, "{line:?} in {texts:?}");
                text_rect(&output, line).expect("drawn").top()
            })
            .collect();
        assert!(tops.is_sorted(), "out of order: {tops:?}");
        let off = input_texts(&inputs, DIAMETER_HINT_KEY, false);
        assert!(!off.contains(&AUTO_HOLDS_LENGTH.to_owned()), "{off:?}");
    }

    #[test]
    fn the_size_rows_new_texts_have_glyphs_and_plain_words() {
        // Spec 8.3: the entry's new words, the foot line's head and Minimum OD's hover (a
        // string the column drew before, now a constant), pass the plain-words guard, say
        // nothing of protecting the gearbox and have glyphs in the default fonts; so does every
        // text each row's entry draws, with the details on.
        let ctx = glyph_context();
        let floor = floor_line(
            dimensions(&DesignInputs::default())
                .unwrap()
                .minimum_diameter_range
                .min,
        );
        for text in [SPREADSHEET_CELL, MINIMUM_HOVER, floor.as_str()] {
            assert!(plain_text_violations(text).is_empty(), "{text:?}");
            assert!(!mentions_gearbox_protection(text), "{text:?}");
            assert_glyphs(&ctx, text, "a Size row's entry");
        }
        for row in SIZE_ROWS {
            for text in input_texts(&DesignInputs::default(), row.key, true) {
                assert_glyphs(&ctx, &text, row.label);
            }
        }
    }
}

```

**`magcoupling-rs/src/gui/guidance.rs`**

Replace:

```rust
        emit("size tab/Hot-low label", sc::HOT_LOW);
        emit("size tab/Outside diameter hover", sc::DIAMETER_HOVER);
```

with:

```rust
        emit("size tab/Hot-low label", sc::HOT_LOW);
        emit("size tab/Minimum OD hover", sc::MINIMUM_HOVER);
        emit("size tab/Outside diameter hover", sc::DIAMETER_HOVER);
```

Replace:

```rust
                ("equation", ex::EQUATION),
            ] {
```

with:

```rust
                ("equation", ex::EQUATION),
                ("spreadsheet cell", ex::SPREADSHEET_CELL),
            ] {
```

**`magcoupling-rs/src/gui/test_support.rs`**

Replace:

```rust
            held: &|_| None,
        };
```

with:

```rust
            held: &|_| None,
            minimum_diameter_mm: None,
        };
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib explain:: 2>&1 | tail -n 15
  ```

  Expected: the build fails with 54 errors, the first "unresolved imports `crate::gui::size_controls::MINIMUM_HOVER`, `crate::gui::size_controls::SIZE_ROWS`": the green step adds what the tests call.

- [ ] **Step 3 (green): the rows, the entry, the card's row, the trade-offs.**

**`magcoupling-rs/src/gui/explain.rs`**

Replace:

```rust
//!
//! A teaching note shows here too (decision C-11): the top bar's [`LEARN`] lists every note
```

with:

```rust
//!
//! A row of the Size column selects its entry when touched (spec 4.3, 5.2): what the control is
//! (its label's hover), the lines the column drew under it (its hint, what sets it, the floor,
//! the footnote on what the controls move), its torque-card row, the checks whose fixes name
//! it, where two of them pull it opposite ways the sentence that says so, its Why link's note
//! and, with the spreadsheet details, its input's cell.
//!
//! A teaching note shows here too (decision C-11): the top bar's [`LEARN`] lists every note
```

Replace:

```rust
use crate::engine::explain::notes::{NOTES, shown_when};
use crate::engine::sizing::FreeVariable;
```

with:

```rust
use crate::engine::explain::notes::{NOTES, shown_when};
use crate::engine::meta::{InputSet, Value};
use crate::engine::size_controls::dimensions;
use crate::engine::sizing::FreeVariable;
```

Replace:

```rust
};
use crate::gui::guidance::{
```

with:

```rust
};
use crate::gui::format::{format_value, value_with_unit};
use crate::gui::guidance::{
```

Replace:

```rust
    self, CHECK_GUIDES, CheckGuide, Consequences, Direction, FIT_CHECK, Learn, NO_FIX,
    SIZE_TAB_NOTE, fix_text, guide_notes, row_learn,
};
```

with:

```rust
    self, CHECK_GUIDES, CheckGuide, Consequences, Direction, FIT_CHECK, Learn, NO_FIX,
    SIZE_TAB_NOTE, fix_text, guide_notes, input_trade_offs, row_learn,
};
```

Replace:

```rust
use crate::gui::input_ui::weak_line;
use crate::gui::look;
```

with:

```rust
use crate::gui::input_ui::weak_line;
use crate::gui::inputs::InputCatalogue;
use crate::gui::look;
```

Replace:

```rust
use crate::gui::look;
use crate::gui::plain_text::help_for;
use crate::gui::readouts::Readouts;
```

with:

```rust
use crate::gui::look;
use crate::gui::plain_text::{
    DIAMETER_HINT_KEY, LENGTH_HINT_KEY, MINIMUM_OD_HINT_KEY, NO_SPREADSHEET_CELL, help_for,
    hint_for,
};
use crate::gui::readouts::Readouts;
```

Replace:

```rust
use crate::gui::readouts::Readouts;
use crate::gui::tiles::Tile;
```

with:

```rust
use crate::gui::readouts::Readouts;
use crate::gui::size_controls::{
    LINKED_HOUSING, SizeRow, floor_line, row_help, size_row, slider_minimum,
};
use crate::gui::tiles::Tile;
```

Replace:

```rust

/// The top bar's button that lists the teaching notes here, and the list's heading (UX
```

with:

```rust

/// The head of an input's foot line with the spreadsheet details on (spec 5.3): its workbook
/// cell, or the line that says the spreadsheet has none.
pub const SPREADSHEET_CELL: &str = "Spreadsheet cell:";

/// The top bar's button that lists the teaching notes here, and the list's heading (UX
```

Replace:

```rust
    NoteList,
}
```

with:

```rust
    NoteList,
    /// A row of the Size column, by its key ([`SizeRow::key`]).
    Input(&'static str),
}
```

Replace:

```rust
            | Selection::Result(_)
            | Selection::Note(_)
            | Selection::NoteList => None,
        }
    }
```

with:

```rust
            | Selection::Result(_)
            | Selection::Note(_)
            | Selection::NoteList
            | Selection::Input(_) => None,
        }
    }
```

Replace:

```rust
            | Selection::Note(_)
            | Selection::NoteList => None,
```

with:

```rust
            | Selection::Note(_)
            | Selection::NoteList
            | Selection::Input(_) => None,
        }
    }

    /// The key of the Size column's row selected, if one is.
    pub fn input(self) -> Option<&'static str> {
        match self {
            Selection::Input(key) => Some(key),
            Selection::Nothing
            | Selection::Check(_)
            | Selection::Result(_)
            | Selection::Note(_)
            | Selection::NoteList => None,
```

Replace:

```rust
    /// Whether the entry reads the Size tab's consequences ([`Shown::consequences`]): the
    /// panel asks its cache for them only then.
    pub fn reads_consequences(self) -> bool {
```

with:

```rust
    /// Whether the entry reads the Size tab's consequences ([`Shown::consequences`]): the
    /// panel asks its cache for them only then. A Size column row's entry draws its
    /// what-moves-what lines and its torque-card row from them.
    pub fn reads_consequences(self) -> bool {
```

Replace:

```rust
    pub fn reads_consequences(self) -> bool {
        self.tile() == Some(Tile::HotLow)
    }
```

with:

```rust
    pub fn reads_consequences(self) -> bool {
        self.tile() == Some(Tile::HotLow) || self.input().is_some()
    }
```

Replace:

```rust
/// ([`Selection::reads_consequences`]), the Size tab's consequences of the design, with the
/// size Auto sets while it is on; and why an input is not free on the design shown, for a
/// note's change entries ([`NoteView::held`]).
pub(crate) struct Shown<'a> {
```

with:

```rust
/// ([`Selection::reads_consequences`]), the Size tab's consequences of the design, with the
/// size Auto sets while it is on; why an input is not free on the design shown, for a note's
/// change entries ([`NoteView::held`]); and the Size column's Minimum OD preference.
pub(crate) struct Shown<'a> {
```

Replace:

```rust
    pub held: &'a dyn Fn(&str) -> Option<String>,
}
```

with:

```rust
    pub held: &'a dyn Fn(&str) -> Option<String>,
    pub minimum_diameter_mm: Option<f64>,
}
```

Replace:

```rust
        Selection::NoteList => note_list_ui(ui, shown.clamps_paused),
    };
```

with:

```rust
        Selection::NoteList => note_list_ui(ui, shown.clamps_paused),
        Selection::Input(key) => input_ui(ui, key, shown, readouts),
    };
```

Replace:

```rust

/// The teaching notes shown, in their order ([`NOTES`]): every reviewed note but the one the
```

with:

```rust

/// The entry of the Size column's row `key` (spec 5.3's input column): its value and its
/// label; what it is (its label's hover); what sets it (its hint, its what-moves-what lines,
/// for Minimum OD the floor line, for the four size controls the footnote on what they move);
/// what a change does (its torque-card row); the checks whose fixes name its input; where two
/// of them move it opposite ways, the sentence that says so ("Watch out"); and the foot (its Why link's note, and with the spreadsheet details the cell of a row that is one
/// input). Nothing for a key that is no row's.
fn input_ui(
    ui: &mut egui::Ui,
    key: &'static str,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let row = size_row(key)?;
    ui.horizontal_wrapped(|ui| {
        if let Some(value) = row_value(row, shown) {
            ui.strong(value);
        }
        ui.colored_label(look::quiet_text(ui.visuals()), row.label);
    });
    heading(ui, WHAT_IT_IS);
    ui.add(egui::Label::new(row_help(key)).wrap());
    let lines = sets_it_lines(row, shown);
    if !lines.is_empty() {
        heading(ui, WHAT_SETS_IT);
        for line in &lines {
            weak_line(ui, line);
        }
    }
    let card = match (row.card, shown.consequences) {
        (Some(control), Some(consequences)) => consequences.card.control_lines(control, shown.auto),
        _ => Vec::new(),
    };
    if !card.is_empty() {
        heading(ui, WHAT_A_CHANGE_DOES);
        ui.label(CARD_HEADING);
        for line in &card {
            weak_line(ui, line);
        }
    }
    let mut asked = None;
    let touched: Vec<&CheckGuide> = row.input.map_or_else(Vec::new, |input| {
        CHECK_GUIDES
            .iter()
            .zip(shown.fixes)
            .filter(|(_, fixes)| fixes.iter().any(|&(path, _)| path == input))
            .map(|(guide, _)| guide)
            .collect()
    });
    if !touched.is_empty() {
        heading(ui, CHECKS_IT_TOUCHES);
        for guide in touched {
            asked = asked.or(check_link_ui(ui, guide, shown));
        }
    }
    let pulls = row
        .input
        .map_or_else(Vec::new, |input| input_trade_offs(shown.fixes, input));
    if !pulls.is_empty() {
        heading(ui, WATCH_OUT);
        for note in pulls {
            ui.add(egui::Label::new(note).wrap());
        }
    }
    // The cell of a row that is one input; Outside diameter and Overall length are sizes the
    // spreadsheet has no cell for, though their sliders write an input.
    let cell = (readouts.show_details() && row.input == Some(row.key)).then(|| {
        let cell = InputCatalogue::get()
            .entry(row.key)
            .and_then(|entry| entry.meta.cell)
            .unwrap_or(NO_SPREADSHEET_CELL);
        format!("{SPREADSHEET_CELL} {cell}")
    });
    asked.or(foot_ui(
        ui,
        row.lesson.map(Learn::Note),
        None,
        cell,
        readouts,
    ))
}

/// The row's value as its slider shows it, with its unit: the input's value, or for the
/// controls that are no one input the size the slider shows (`dimensions`); none where the
/// design has no size controls.
fn row_value(row: &SizeRow, shown: &Shown<'_>) -> Option<String> {
    let mm = |x: f64| value_with_unit(&Value::Num(x), format_value(&Value::Num(x)), "mm");
    match row.key {
        MINIMUM_OD_HINT_KEY | DIAMETER_HINT_KEY | LENGTH_HINT_KEY => {
            let size = dimensions(shown.inputs)?;
            Some(mm(match row.key {
                MINIMUM_OD_HINT_KEY => slider_minimum(&size, shown.minimum_diameter_mm),
                DIAMETER_HINT_KEY => size.diameter_mm,
                _ => size.overall_length_mm,
            }))
        }
        path => {
            let value = shown.inputs.get(path)?;
            let unit = InputCatalogue::get().entry(path)?.meta.unit;
            Some(value_with_unit(&value, format_value(&value), unit))
        }
    }
}

/// The lines the Size column drew under the row (spec 4.3): its hint; its what-moves-what lines
/// ([`Consequences`]); for Minimum OD the diameter range's floor; for the four controls the
/// torque card steps, [`LINKED_HOUSING`].
fn sets_it_lines(row: &SizeRow, shown: &Shown<'_>) -> Vec<String> {
    let mut lines: Vec<String> = hint_for(row.key).map(str::to_owned).into_iter().collect();
    if let Some(consequences) = shown.consequences {
        match row.key {
            DIAMETER_HINT_KEY => lines.extend(consequences.diameter_line()),
            LENGTH_HINT_KEY => lines.extend(consequences.length_line()),
            key if key == FreeVariable::MagnetsPerRing.path() => {
                lines.extend(consequences.magnet_lines(shown.inputs.coupling.npole, shown.auto))
            }
            _ => {}
        }
    }
    if row.key == MINIMUM_OD_HINT_KEY
        && let Some(size) = dimensions(shown.inputs)
    {
        lines.push(floor_line(size.minimum_diameter_range.min));
    }
    if row.card.is_some() {
        lines.push(LINKED_HOUSING.to_owned());
    }
    lines
}

/// A check named in an entry: its state's mark, its label, a link that selects it, and its
/// state's word. Returns the selection a click asked for.
fn check_link_ui(
    ui: &mut egui::Ui,
    guide: &'static CheckGuide,
    shown: &Shown<'_>,
) -> Option<Asked> {
    let mut asked = None;
    ui.horizontal_wrapped(|ui| {
        let mark = mark_level(guide, shown.inputs, shown.results);
        state_mark(ui, mark);
        if ui.link(guide.label).clicked() {
            asked = Some(Asked::Select(Selection::Check(guide.check)));
        }
        if let Some(mark) = mark {
            ui.colored_label(mark.color(ui.visuals()), mark.word());
        }
    });
    asked
}

/// The teaching notes shown, in their order ([`NOTES`]): every reviewed note but the one the
```

Replace:

```rust
    let equation = result_info(guide.check).map(|_| guide.check);
    foot_ui(ui, guide.learn, equation, readouts)
}
```

with:

```rust
    let equation = result_info(guide.check).map(|_| guide.check);
    foot_ui(ui, Some(guide.learn), equation, None, readouts)
}
```

Replace:

```rust
        heading(ui, CHECKS_IT_TOUCHES);
        ui.horizontal_wrapped(|ui| {
            let mark = mark_level(guide, shown.inputs, shown.results);
            state_mark(ui, mark);
            if ui.link(guide.label).clicked() {
                asked = Some(Asked::Select(Selection::Check(guide.check)));
            }
            if let Some(mark) = mark {
                ui.colored_label(mark.color(ui.visuals()), mark.word());
            }
        });
    }
```

with:

```rust
        heading(ui, CHECKS_IT_TOUCHES);
        asked = check_link_ui(ui, guide, shown);
    }
```

Replace:

```rust
    let learn = row_learn(path).unwrap_or(Learn::None(""));
    asked.or(foot_ui(ui, learn, Some(path), readouts))
}
```

with:

```rust
    let learn = row_learn(path).unwrap_or(Learn::None(""));
    asked.or(foot_ui(ui, Some(learn), Some(path), None, readouts))
}
```

Replace:

```rust

/// An entry's foot: [`FULL_LESSON`] and where `learn` leads ([`learn_ui`]), then, for an entry
/// with a result (`equation`), the [`EQUATION`] link. Returns what the Equation link asked.
fn foot_ui(
```

with:

```rust

/// An entry's foot: [`FULL_LESSON`] and where `learn` leads ([`learn_ui`]), for an entry
/// that has a lesson; for an entry with a result (`equation`), the [`EQUATION`] link; then its
/// `cell` line, weak. Returns what the Equation link asked.
fn foot_ui(
```

Replace:

```rust
    ui: &mut egui::Ui,
    learn: Learn,
    equation: Option<&'static str>,
```

with:

```rust
    ui: &mut egui::Ui,
    learn: Option<Learn>,
    equation: Option<&'static str>,
```

Replace:

```rust
    equation: Option<&'static str>,
    readouts: &mut Readouts,
```

with:

```rust
    equation: Option<&'static str>,
    cell: Option<String>,
    readouts: &mut Readouts,
```

Replace:

```rust
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label(FULL_LESSON);
        learn_ui(ui, learn, readouts);
    });
    let path = equation?;
    ui.link(EQUATION).clicked().then_some(Asked::Equation(path))
}
```

with:

```rust
    ui.separator();
    if let Some(learn) = learn {
        ui.horizontal_wrapped(|ui| {
            ui.label(FULL_LESSON);
            learn_ui(ui, learn, readouts);
        });
    }
    let asked =
        equation.and_then(|path| ui.link(EQUATION).clicked().then_some(Asked::Equation(path)));
    if let Some(cell) = cell {
        ui.add(egui::Label::new(egui::RichText::new(cell).small().weak()).wrap());
    }
    asked
}
```

**`magcoupling-rs/src/gui/guidance.rs`**

Replace:

```rust
pub fn trade_offs(all: &[Vec<(&'static str, Direction)>], index: usize) -> Vec<String> {
    let (Some(own), Some(guide)) = (all.get(index), CHECK_GUIDES.get(index)) else {
```

with:

```rust
pub fn trade_offs(all: &[Vec<(&'static str, Direction)>], index: usize) -> Vec<String> {
    trade_off_pairs(all, index)
        .into_iter()
        .map(|(path, raising, lowering)| trade_off_text(path, raising, lowering))
        .collect()
}

/// The [`trade_offs`] that name the input `path`, each once (UX overhaul spec 5.3: an input's
/// Explain entry's "Watch out"): the same sentence the two checks' blocks draw.
pub fn input_trade_offs(all: &[Vec<(&'static str, Direction)>], path: &str) -> Vec<String> {
    let mut notes: Vec<String> = Vec::new();
    for index in 0..CHECK_GUIDES.len() {
        for (input, raising, lowering) in trade_off_pairs(all, index) {
            let note = trade_off_text(input, raising, lowering);
            if input == path && !notes.contains(&note) {
                notes.push(note);
            }
        }
    }
    notes
}

/// The sentence of one trade-off: `raising`'s check needs the input at `path` raised and
/// `lowering`'s lowered.
fn trade_off_text(path: &str, raising: &CheckGuide, lowering: &CheckGuide) -> String {
    format!(
        "One input pulls two ways ({}): {} needs it raised and {} needs it lowered, so change another input to settle one of them.",
        fix_label(path),
        raising.label,
        lowering.label
    )
}

/// The trade-offs of the guide at `index` ([`trade_offs`]): each input its fixes move that
/// another failing check's fix moves the other way, with the guide that raises it and the one
/// that lowers it.
fn trade_off_pairs(
    all: &[Vec<(&'static str, Direction)>],
    index: usize,
) -> Vec<(&'static str, &'static CheckGuide, &'static CheckGuide)> {
    let (Some(own), Some(guide)) = (all.get(index), CHECK_GUIDES.get(index)) else {
```

Replace:

```rust
    };
    let mut notes = Vec::new();
    for &(path, direction) in own {
```

with:

```rust
    };
    let mut pairs = Vec::new();
    for &(path, direction) in own {
```

Replace:

```rust
            };
            notes.push(format!(
                "One input pulls two ways ({}): {} needs it raised and {} needs it lowered, so change another input to settle one of them.",
                fix_label(path),
                raising.label,
                lowering.label
            ));
        }
```

with:

```rust
            };
            pairs.push((path, raising, lowering));
        }
```

Replace:

```rust
            pairs.push((path, raising, lowering));
        }
    }
    notes
}

/// A fix that also sets back another guide's check, by physics the fix's own check does not
```

with:

```rust
            pairs.push((path, raising, lowering));
        }
    }
    pairs
}

/// A fix that also sets back another guide's check, by physics the fix's own check does not
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
                    held: &held,
                };
```

with:

```rust
                    held: &held,
                    minimum_diameter_mm: self.minimum_diameter_mm,
                };
```

**`magcoupling-rs/src/gui/size_controls.rs`**

Replace:

```rust
};
use crate::gui::guidance::{Consequences, FACE_GAP, SIZE_TAB_NOTE};
use crate::gui::input_ui::{hint_line, slider, weak_line};
```

with:

```rust
};
use crate::gui::guidance::{Consequences, FACE_GAP, SIZE_TAB_NOTE, TORQUE_BAND_NOTE};
use crate::gui::input_ui::{hint_line, slider, weak_line};
```

Replace:

```rust
};
use crate::gui::sizing::{SizingMode, SizingState, auto_choice, variable_label};
use crate::{DesignInputs, DesignResults, compute_all};
```

with:

```rust
};
use crate::gui::sizing::{REQUIREMENT_INPUT, SizingMode, SizingState, auto_choice, variable_label};
use crate::gui::torque_card::Control;
use crate::{DesignInputs, DesignResults, compute_all};
```

Replace:

```rust

/// The hover text of the Outside diameter control (R-3): what it is (the largest of the cup,
```

with:

```rust

/// The hover text of the Minimum OD control: what it does.
pub const MINIMUM_HOVER: &str =
    "Sets the lower end of the diameter slider without changing your current size.";

/// The hover text of the Outside diameter control (R-3): what it is (the largest of the cup,
```

Replace:

```rust
pub const OPEN_SIZE_TAB: &str = "Open the Size tab";

```

with:

```rust
pub const OPEN_SIZE_TAB: &str = "Open the Size tab";

/// A row of the Size column (UX overhaul spec 4.3), the key of its Explain entry
/// (`explain::Selection::Input`): the input its slider writes, or for a control that is no one
/// input its hint key (`plain_text::SIZE_TAB_HINT_KEYS`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SizeRow {
    /// The row's key: its hint key, and the input path of a row that is one input.
    pub key: &'static str,
    /// The label drawn above its slider.
    pub label: &'static str,
    /// The input the slider writes, whose spreadsheet cell and fixes are the row's; none for
    /// Minimum OD, a view preference.
    pub input: Option<&'static str>,
    /// The torque card's row of the control, for the four controls the card steps.
    pub card: Option<Control>,
    /// The note its Why link opened in the column (spec 3.7).
    pub lesson: Option<&'static str>,
}

/// The rows of the Size column, top to bottom.
pub(crate) const SIZE_ROWS: [SizeRow; 6] = [
    SizeRow {
        key: REQUIREMENT_INPUT,
        label: REQUIRED,
        input: Some(REQUIREMENT_INPUT),
        card: None,
        lesson: Some(TORQUE_BAND_NOTE),
    },
    SizeRow {
        key: MINIMUM_OD_HINT_KEY,
        label: MINIMUM,
        input: None,
        card: None,
        lesson: None,
    },
    SizeRow {
        key: DIAMETER_HINT_KEY,
        label: DIAMETER,
        input: Some("coupling.inner_back_apothem_mm"),
        card: Some(Control::OutsideDiameter),
        lesson: Some(SIZE_TAB_NOTE),
    },
    SizeRow {
        key: LENGTH_HINT_KEY,
        label: LENGTH,
        input: Some("coupling.magnets.axial_length_mm"),
        card: Some(Control::OverallLength),
        lesson: None,
    },
    SizeRow {
        key: FACE_GAP,
        label: GAP,
        input: Some(FACE_GAP),
        card: Some(Control::FaceGap),
        lesson: None,
    },
    SizeRow {
        key: FreeVariable::MagnetsPerRing.path(),
        label: MAGNETS,
        input: Some(FreeVariable::MagnetsPerRing.path()),
        card: Some(Control::MagnetsPerRing),
        lesson: Some(SIZE_TAB_NOTE),
    },
];

/// The Size column's row of `key`.
pub(crate) fn size_row(key: &str) -> Option<&'static SizeRow> {
    SIZE_ROWS.iter().find(|row| row.key == key)
}

/// What the row of `key` is: the hover text its label shows.
pub(crate) fn row_help(key: &str) -> &'static str {
    match key {
        MINIMUM_OD_HINT_KEY => MINIMUM_HOVER,
        DIAMETER_HINT_KEY => DIAMETER_HOVER,
        LENGTH_HINT_KEY => LENGTH_HOVER,
        path => help_for(path),
    }
}

/// The line on the diameter range's floor, in Minimum OD's entry.
pub(crate) fn floor_line(floor_mm: f64) -> String {
    format!("Shaft and housing range floor: {floor_mm:.1} mm")
}

```

Replace:

```rust
            let mut minimum = slider_minimum(&legacy, minimum_diameter_mm);
            ui.label(MINIMUM).on_hover_text(
                "Sets the lower end of the diameter slider without changing your current size.",
            );
            let response = ui
```

with:

```rust
            let mut minimum = slider_minimum(&legacy, minimum_diameter_mm);
            ui.label(MINIMUM).on_hover_text(MINIMUM_HOVER);
            let response = ui
```

**`magcoupling-rs/src/gui/torque_card.rs`**

Replace:

```rust

    /// "{label}: {fewer}; {more}" (spec 4.1), the two sides' N·m at the row's one precision.
```

with:

```rust

    /// The row's line while Auto is on `auto`: [`held_line`] for the row Auto holds (owner 5A),
    /// else [`Row::line`].
    fn text(&self, auto: Option<FreeVariable>) -> String {
        if auto.is_some() && self.control.held_by() == auto {
            held_line(self.control)
        } else {
            self.line()
        }
    }

    /// "{label}: {fewer}; {more}" (spec 4.1), the two sides' N·m at the row's one precision.
```

Replace:

```rust
                lines.extend(auto.map(|variable| auto_line(variable).to_owned()));
                lines.extend(rows.iter().map(|row| {
                    if auto.is_some() && row.control.held_by() == auto {
                        held_line(row.control)
                    } else {
                        row.line()
                    }
                }));
                lines
```

with:

```rust
                lines.extend(auto.map(|variable| auto_line(variable).to_owned()));
                lines.extend(rows.iter().map(|row| row.text(auto)));
                lines
```

Replace:

```rust
                lines
            }
        }
    }
```

with:

```rust
                lines
            }
        }
    }

    /// The lines of `control`'s row alone, as a Size control's Explain entry draws them (UX
    /// overhaul spec 5.3, decision C-10): [`CAPTION`], the Auto line while Auto is on (owner
    /// 5A: it says which edit Auto answers by re-solving), then the row's line, as
    /// [`lines`](Self::lines) draws them; none where the card has no row for the control (an odd
    /// magnet count has no Outside diameter or Overall length row); for the four one-line forms,
    /// their line alone.
    pub fn control_lines(&self, control: Control, auto: Option<FreeVariable>) -> Vec<String> {
        match self {
            TorqueCard::Rows(rows) => {
                rows.iter()
                    .find(|row| row.control == control)
                    .map_or_else(Vec::new, |row| {
                        let mut lines = vec![CAPTION.to_owned()];
                        lines.extend(auto.map(|variable| auto_line(variable).to_owned()));
                        lines.push(row.text(auto));
                        lines
                    })
            }
            _ => self.lines(auto),
        }
    }
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1092 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** the six files: `feat(magcoupler): each Size column row has its Explain entry: what it is, what sets it, its torque-card row, the checks it touches, its lesson and cell (BL-090 T1)`.

### Task 2: The Size column draws its controls and two folds; touching a control selects it

**Model:** `sonnet`.

The column draws each control's label (clickable) and slider and nothing about it; under them the folds Detailed inputs and Assumptions. Touching a control selects its row; the Required torque row loses its Why link (its entry's lesson). The Size column no longer asks the `ConsequenceCache`. The size tab's line tests read the entries; two new tests hold the touch and the column; the first view's word ceiling falls to 280 and both folds are on it at every size.

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/size_controls.rs`, `magcoupling-rs/src/gui/mod.rs`, `magcoupling-rs/src/app.rs`, `magcoupling-rs/src/gui/guidance.rs` (two doc comments)

**Interfaces:**
- Produces: `panel::{DETAILED_INPUTS, ASSUMPTIONS_FOLD}` (re-exported from `gui`); `size_controls::{touched, row_label}`; `ControlsOutput::touched: Option<&'static str>`; `controls_ui(ui, inputs, minimum_diameter_mm, auto)`; `requirement_ui(..) -> (egui::Id, bool)` (private).
- Removes: `ControlsOutput::note`, `controls_ui`'s `consequences` argument, `MagcouplingPanel::size_tab_consequences`, the column's hint, what-moves-what and floor lines, its Why links and its `LINKED_HOUSING` footnote.

- [ ] **Step 1 (red): the column's tests.** The blocks add the panel tests' helpers `entry` and `explain_rect`, rewrite the size tab's line tests to read the entries, add `touching_a_size_control_selects_its_row_and_a_hover_selects_nothing` (a label's click, a Tab onto a slider, a drag) and `the_size_column_draws_its_controls_and_two_folds_and_no_line_about_them`, and change the app's first-view and word tests and the re-exports they read.

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
pub use panel::{
    CentreView, DESIGN_MENU, EXPORT, LOAD_DESIGN, MagcouplingPanel, OPEN_EXAMPLE, PanelRequest,
    RESET_TO_START, SHARE, SHOW_DETAILS, SIZING_LOG_PREFIX,
};
```

with:

```rust
pub use panel::{
    ASSUMPTIONS_FOLD, CentreView, DESIGN_MENU, DETAILED_INPUTS, EXPORT, LOAD_DESIGN,
    MagcouplingPanel, OPEN_EXAMPLE, PanelRequest, RESET_TO_START, SHARE, SHOW_DETAILS,
    SIZING_LOG_PREFIX,
};
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
                let size_column = whole("Required torque");
                // The fold of the detailed inputs, the way to the other inputs, on the first view
                // where the window is at least the laptop's height (plan review E-4).
                if size.y >= WINDOWS[0].y {
                    whole("Detailed inputs");
                }
                let checks = whole(crate::gui::checks::HEADING);
```

with:

```rust
                let size_column = whole("Required torque");
                // The folds of the detailed inputs and of the assumptions, the ways to the other
                // inputs, on the first view at every size (plan review E-4; UX overhaul spec 4.3
                // and 8.2, test 9: the Assumptions view is reachable from the opening tab).
                whole(crate::gui::DETAILED_INPUTS);
                whole(crate::gui::ASSUMPTIONS_FOLD);
                let checks = whole(crate::gui::checks::HEADING);
```

Replace:

```rust
    #[test]
    fn the_first_view_draws_at_most_four_hundred_and_thirty_words_at_1280_by_800() {
        // UX overhaul spec 2.1 (goal G-1) and 8.2, test 8, measured as the inventory did: the
```

with:

```rust
    #[test]
    fn the_first_view_draws_at_most_two_hundred_and_eighty_words_at_1280_by_800() {
        // UX overhaul spec 2.1 (goal G-1) and 8.2, test 8, measured as the inventory did: the
```

Replace:

```rust
        // the drawing kept until its tags are selections (decision D-19); 427 after part 2a's
        // top bar. The design's 250 needs the Size column and the callout list of part 2b,
        // which lowers this ceiling.
        use crate::gui::test_support::words_shown;
```

with:

```rust
        // the drawing kept until its tags are selections (decision D-19); 427 after part 2a's
        // top bar; 272 after part 2b-i moved the Size column's lines into the Explain panel.
        // The design's 250 needs the callout list to leave the opening tab (part 2b-ii), which
        // lowers this ceiling to it.
        use crate::gui::test_support::words_shown;
```

Replace:

```rust
        let words = words_shown(&output, egui::Rect::from_min_size(egui::Pos2::ZERO, size));
        assert!(words <= 430, "{words} words on the first view");
    }
```

with:

```rust
        let words = words_shown(&output, egui::Rect::from_min_size(egui::Pos2::ZERO, size));
        assert!(words <= 280, "{words} words on the first view");
    }
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    use crate::gui::guidance::{
        Consequences, FACE_GAP, RING_SIZE_STAYS, SIZE_TAB_NOTE, plain_text_violations,
    };
```

with:

```rust
    use crate::gui::guidance::{
        Consequences, FACE_GAP, RING_SIZE_STAYS, SIZE_TAB_NOTE, TORQUE_BAND_NOTE,
        plain_text_violations,
    };
```

Replace:

```rust
        // screen draws every row.
        for path in [FACE_GAP, "coupling.c_end", "coupling.mu0"] {
```

with:

```rust
        // screen draws every row.
        let mut assumptions = 0;
        for path in [FACE_GAP, "coupling.c_end", "coupling.mu0"] {
```

Replace:

```rust
            );
            assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{path}");
```

with:

```rust
            );
            // An assumption's row is the Assumptions fold's (UX overhaul spec 4.3), which
            // opens with its rationale; any other row's fold is the detailed inputs.
            let assumption = ASSUMPTIONS.iter().find(|a| a.paths.contains(&path));
            assert_eq!(
                assumption.map(|a| count(&output, a.rationale)),
                assumption.map(|_| 1),
                "{path}"
            );
            assumptions += usize::from(assumption.is_some());
            assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{path}");
```

Replace:

```rust
            assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{path}");
        }
    }

```

with:

```rust
            assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{path}");
        }
        assert_eq!(assumptions, 1, "one of the paths is an assumption");
    }

```

Replace:

```rust
            if clamps_paused {
                // The check list and each check's Explain entry on designs failing every check,
                // the clamp's too.
                let mut hot = DesignInputs::default();
```

with:

```rust
            if clamps_paused {
                // The check list, each check's Explain entry and each Size row's (BL-090) on
                // designs failing every check, the clamp's too.
                let mut hot = DesignInputs::default();
```

Replace:

```rust
                        texts.extend(drawn_texts(&harness.frame(Vec::new())));
                    }
                }
                // The shaft joints' guide (BL-064) counts while the section is paused: its
```

with:

```rust
                        texts.extend(drawn_texts(&harness.frame(Vec::new())));
                    }
                    for row in size_controls::SIZE_ROWS {
                        harness.panel.selection = Selection::Input(row.key);
                        texts.extend(drawn_texts(&harness.frame(Vec::new())));
                    }
                }
                // The shaft joints' guide (BL-064) counts while the section is paused: its
```

Replace:

```rust

    /// The what-moves-what lines and the torque card of the design the panel shows, as the Size
```

with:

```rust

    /// Shows the entry of the Size column's row `key` (UX overhaul spec 4.3), as touching its
    /// control does: the Explain panel draws it in the frame this returns.
    fn entry(harness: &mut Harness, key: &'static str) -> egui::FullOutput {
        harness.panel.selection = Selection::Input(key);
        harness.frame(Vec::new())
    }

    /// The rect of `text` drawn in the right column (the check list over the Explain panel):
    /// an entry's copy, where the Size column draws the same words.
    fn explain_rect(output: &egui::FullOutput, text: &str) -> egui::Rect {
        let left = right_column_left(output);
        text_rects(output, text)
            .into_iter()
            .find(|rect| rect.left() >= left - 1.0)
            .unwrap_or_else(|| panic!("no {text:?} in the right column"))
    }

    /// The what-moves-what lines and the torque card of the design the panel shows, as the Size
```

Replace:

```rust
    #[test]
    fn a_required_edit_reaches_the_card_in_its_own_frame() {
        // Slice 3 spec 3.5: the card draws the design shown after the frame's edits, the Required
        // slider's included, which the readouts above it make. At the defaults (hot-low
        // 2.29 N·m), Required 2.4 N·m lies under the Overall length +1 mm side's hot-low
        // (2.49 N·m), so that side's Hot-low torque flips: the frame that applies the typed
        // value draws the edited design's card, not the one before the edit.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
```

with:

```rust
    #[test]
    fn a_size_control_s_edit_reaches_its_entry_in_its_own_frame() {
        // Slice 3 spec 3.5 and UX overhaul spec 5.2: an edit of a Size control selects it, and
        // the frame that applies the edit draws the edited design in the control's entry, not
        // the design before it. The Required box typed into: its entry's title reads the typed
        // 2.4 N·m in the frame that applies it. An arrow key on Magnets per ring: its lines and
        // its torque-card row are the edited design's in the frame of the key.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        let before = card_lines(&harness);
        let value_box = required_box(&output);
```

with:

```rust
        let output = harness.frame(Vec::new());
        let value_box = required_box(&output);
```

Replace:

```rust
        harness.click(value_box.center());
        harness.frame([select_all(), vec![egui::Event::Text("2.4".to_owned())]].concat());
```

with:

```rust
        harness.click(value_box.center());
        assert_eq!(harness.panel.selection, Selection::Input(REQUIREMENT_INPUT));
        harness.frame([select_all(), vec![egui::Event::Text("2.4".to_owned())]].concat());
```

Replace:

```rust
        assert_eq!(harness.panel.inputs().metal.required_min_Nm, 2.4);
        let after = card_lines(&harness);
        assert_ne!(
```

with:

```rust
        assert_eq!(harness.panel.inputs().metal.required_min_Nm, 2.4);
        let unit = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .expect("the requirement has input metadata")
            .meta
            .unit;
        let typed = Value::Num(2.4);
        let title = crate::gui::format::value_with_unit(&typed, format_value(&typed), unit);
        explain_rect(&output, &title);
        let before = shown_consequences(&harness);
        let magnets = harness.panel.design_widgets[3];
        harness.ctx.memory_mut(|m| m.request_focus(magnets));
        harness.frame(Vec::new());
        let output = harness.frame(key_tap(egui::Key::ArrowLeft));
        assert_eq!(harness.panel.inputs().coupling.npole, 8);
        assert_eq!(harness.panel.selection, Selection::Input(POLES));
        let after = shown_consequences(&harness);
        let row = |c: &Consequences| c.card.control_lines(Control::MagnetsPerRing, None);
        assert_ne!(
```

Replace:

```rust
        assert_ne!(
            after, before,
            "Required 2.4 N·m moves a Hot-low torque flip"
        );
```

with:

```rust
        assert_ne!(
            row(&after),
            row(&before),
            "two magnets fewer move the count row"
        );
```

Replace:

```rust
        );
        for line in &after {
            assert_eq!(count(&output, line), 1, "{line}");
```

with:

```rust
        );
        for line in row(&after).iter().chain(&after.magnet_lines(8, None)) {
            assert_eq!(count(&output, line), 1, "{line}");
```

Replace:

```rust
            assert_eq!(count(&output, line), 1, "{line}");
        }
    }

```

with:

```rust
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in row(&before)
            .iter()
            .filter(|line| !row(&after).contains(line))
        {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
    }

```

Replace:

```rust
    #[test]
    fn size_tab_draws_each_what_moves_what_line_under_its_control() {
        // Spec 3.3 at the default design: the Set-by line under Outside diameter, the length
        // split under Overall length, and the magnet count's lines under Magnets per ring, the
        // ring-size line first, each under the one before.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
```

with:

```rust
    #[test]
    fn each_size_control_s_entry_draws_its_what_moves_what_lines_under_what_sets_it() {
        // Spec 3.3 at the default design, in the Explain panel since UX overhaul spec 4.3: the
        // Set-by line in Outside diameter's entry, the length split in Overall length's, and the
        // magnet count's lines in Magnets per ring's, the ring-size line first, each under the
        // one before and under the entry's "What sets it". The Size column draws none of them.
        use crate::gui::explain::WHAT_SETS_IT;
        let mut harness = size_tab(SCREEN, DesignInputs::default());
```

Replace:

```rust
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let output = harness.frame(Vec::new());
        let consequences = shown_consequences(&harness);
```

with:

```rust
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let consequences = shown_consequences(&harness);
```

Replace:

```rust
        let consequences = shown_consequences(&harness);
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        let diameter = consequences
```

with:

```rust
        let consequences = shown_consequences(&harness);
        let diameter = consequences
```

Replace:

```rust
        let length = consequences.length_line().expect("and a length split");
        assert_eq!(count(&output, &diameter), 1);
        assert_eq!(count(&output, &length), 1);
        assert!(rect(DIAMETER).bottom() <= rect(diameter.as_str()).top());
        assert!(rect(diameter.as_str()).bottom() <= rect(LENGTH).top());
        assert!(rect(LENGTH).bottom() <= rect(length.as_str()).top());
        assert!(rect(length.as_str()).bottom() <= rect(GAP).top());
        let lines = consequences.magnet_lines(10, None);
        assert_eq!(lines.first().map(String::as_str), Some(RING_SIZE_STAYS));
        // The ring-size line and the fit line (slice 3, decision 7A: the lines for two magnets
```

with:

```rust
        let length = consequences.length_line().expect("and a length split");
        let magnets = consequences.magnet_lines(10, None);
        assert_eq!(magnets.first().map(String::as_str), Some(RING_SIZE_STAYS));
        // The ring-size line and the fit line (slice 3, decision 7A: the lines for two magnets
```

Replace:

```rust
        // fewer and two more are the torque card's count row).
        assert_eq!(lines.len(), 2, "{lines:?}");
        let mut above = rect(MAGNETS);
        for line in &lines {
            assert_eq!(count(&output, line), 1, "{line}");
            let at = rect(line.as_str());
            assert!(
                above.bottom() <= at.top() + 0.5,
                "{line:?} is not under the line above it"
            );
            above = at;
        }
```

with:

```rust
        // fewer and two more are the torque card's count row).
        assert_eq!(magnets.len(), 2, "{magnets:?}");
        let column = harness.frame(Vec::new());
        for (key, lines) in [
            (DIAMETER_HINT_KEY, vec![diameter]),
            (LENGTH_HINT_KEY, vec![length]),
            (POLES, magnets),
        ] {
            for line in &lines {
                assert_eq!(count(&column, line), 0, "the column draws no {line:?}");
            }
            let output = entry(&mut harness, key);
            let mut above = explain_rect(&output, WHAT_SETS_IT);
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{key}: {line}");
                let at = explain_rect(&output, line);
                assert!(
                    above.bottom() <= at.top() + 0.5,
                    "{key}: {line:?} is not under the line above it"
                );
                above = at;
            }
        }
```

Replace:

```rust

    /// The Size tab's five controls in drawing order, each with its hint key (spec 3.5): the gap
    /// and the magnet count draw their Key design rows' hints.
    const SIZE_TAB_HINTS: [(&str, &str); 5] = [
```

with:

```rust

    /// The Size column's five controls in drawing order, each with its hint key (spec 3.5), which
    /// is its row's key: the gap and the magnet count's hints are their Key design rows'.
    const SIZE_TAB_HINTS: [(&str, &str); 5] = [
```

Replace:

```rust
    #[test]
    fn size_tab_draws_each_hint_line_under_its_control() {
        // Slice 2 spec 3.5 (S2-7): one weak line under each of the five Size-tab controls,
        // under its label and above the next control's.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
```

with:

```rust
    #[test]
    fn each_size_control_s_entry_draws_its_hint_under_what_sets_it() {
        // Slice 2 spec 3.5 (S2-7), moved by UX overhaul spec 4.3: each of the five controls'
        // hints shows once, in its entry, under the entry's title and its "What sets it"; the
        // Size column draws none of them.
        use crate::gui::explain::WHAT_SETS_IT;
        let mut harness = size_tab(SCREEN, DesignInputs::default());
```

Replace:

```rust
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let output = harness.frame(Vec::new());
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        for (i, &(label, key)) in SIZE_TAB_HINTS.iter().enumerate() {
            let hint = hint_for(key).unwrap_or_else(|| panic!("{key} has no hint"));
```

with:

```rust
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let column = harness.frame(Vec::new());
        for (label, key) in SIZE_TAB_HINTS {
            let hint = hint_for(key).unwrap_or_else(|| panic!("{key} has no hint"));
```

Replace:

```rust
            let hint = hint_for(key).unwrap_or_else(|| panic!("{key} has no hint"));
            assert_eq!(count(&output, hint), 1, "{label}: {hint}");
```

with:

```rust
            let hint = hint_for(key).unwrap_or_else(|| panic!("{key} has no hint"));
            assert_eq!(count(&column, hint), 0, "{label}: the column draws no hint");
            let output = entry(&mut harness, key);
            assert_eq!(count(&output, hint), 1, "{label}: {hint}");
```

Replace:

```rust
            assert_eq!(count(&output, hint), 1, "{label}: {hint}");
            let at = rect(hint);
            assert!(
```

with:

```rust
            assert_eq!(count(&output, hint), 1, "{label}: {hint}");
            let heading = explain_rect(&output, WHAT_SETS_IT);
            assert!(
```

Replace:

```rust
            assert!(
                rect(label).bottom() <= at.top() + 0.5,
                "{label}: the hint is not under its control"
            );
```

with:

```rust
            assert!(
                explain_rect(&output, label).bottom() <= heading.top() + 0.5,
                "{label}: the entry's title leads"
            );
```

Replace:

```rust
            );
            if let Some(&(next, _)) = SIZE_TAB_HINTS.get(i + 1) {
                assert!(
                    at.bottom() <= rect(next).top() + 0.5,
                    "{label}: the hint runs past {next}"
                );
            }
        }
```

with:

```rust
            );
            assert!(
                heading.bottom() <= explain_rect(&output, hint).top() + 0.5,
                "{label}: the hint is not under its heading"
            );
        }
```

Replace:

```rust
    #[test]
    fn hint_lines_stay_inside_the_inputs_side_in_a_narrow_window() {
        // Spec 5 "A narrow window: hint lines wrap": at the M4-1 review's 930-point window every
```

with:

```rust
    #[test]
    fn hint_lines_stay_inside_their_sides_in_a_narrow_window() {
        // Spec 5 "A narrow window: hint lines wrap": at the M4-1 review's 930-point window every
```

Replace:

```rust
        // Spec 5 "A narrow window: hint lines wrap": at the M4-1 review's 930-point window every
        // hint, on the Size tab and in the Key design group, wraps inside the inputs side.
        let narrow = egui::vec2(930.0, 3000.0);
```

with:

```rust
        // Spec 5 "A narrow window: hint lines wrap": at the M4-1 review's 930-point window every
        // hint wraps inside its side: the Key design group's in the inputs side, each Size
        // control's in its entry in the right column (UX overhaul spec 4.3).
        let narrow = egui::vec2(930.0, 3000.0);
```

Replace:

```rust
        let narrow = egui::vec2(930.0, 3000.0);
        let size_keys: Vec<&str> = SIZE_TAB_HINTS.iter().map(|&(_, key)| key).collect();
        let views = [
            (size_tab(narrow, DesignInputs::default()), size_keys),
            (Harness::on_screen(narrow), KEY_DESIGN.to_vec()),
        ];
        let mut wrapped = false;
```

with:

```rust
        let narrow = egui::vec2(930.0, 3000.0);
        let side = |harness: &Harness, id: &str| {
            egui::containers::panel::PanelState::load(&harness.ctx, egui::Id::new(id))
                .unwrap_or_else(|| panic!("{id} is drawn"))
                .rect
        };
        let mut wrapped = false;
```

Replace:

```rust
        let mut wrapped = false;
        for (mut harness, keys) in views {
            let output = harness.frame(Vec::new());
            let left = egui::containers::panel::PanelState::load(
                &harness.ctx,
                egui::Id::new("magcoupling_inputs"),
            )
            .expect("the inputs side is drawn")
            .rect
            .left();
            let right = left + INPUTS_WIDTH;
            for key in keys {
                let hint = hint_for(key).unwrap_or_else(|| panic!("{key} has no hint"));
                let at = text_rect(&output, hint).unwrap_or_else(|| panic!("no {hint:?}"));
                assert!(
                    at.left() >= left - 1.0 && at.right() <= right + 1.0,
                    "{hint:?} at {at:?} runs out of the inputs side ({left} to {right})"
                );
                wrapped |= rows(&output, hint) > 1;
            }
        }
```

with:

```rust
        let mut wrapped = false;
        let mut inside = |output: &egui::FullOutput, hint: &str, left: f32, right: f32| {
            let at = text_rect(output, hint).unwrap_or_else(|| panic!("no {hint:?}"));
            assert!(
                at.left() >= left - 1.0 && at.right() <= right + 1.0,
                "{hint:?} at {at:?} runs out of its side ({left} to {right})"
            );
            wrapped |= rows(output, hint) > 1;
        };
        let mut harness = Harness::on_screen(narrow);
        let output = harness.frame(Vec::new());
        let left = side(&harness, "magcoupling_inputs").left();
        for path in KEY_DESIGN {
            let hint = hint_for(path).unwrap_or_else(|| panic!("{path} has no hint"));
            inside(&output, hint, left, left + INPUTS_WIDTH);
        }
        let mut harness = size_tab(narrow, DesignInputs::default());
        for (_, key) in SIZE_TAB_HINTS {
            let output = entry(&mut harness, key);
            let right = side(&harness, "magcoupling_dashboard");
            let hint = hint_for(key).unwrap_or_else(|| panic!("{key} has no hint"));
            inside(&output, hint, right.left(), right.right());
        }
```

Replace:

```rust
    #[test]
    fn size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter() {
```

with:

```rust
    #[test]
    fn touching_a_size_control_selects_its_row_and_a_hover_selects_nothing() {
        // UX overhaul spec 5.2: a click on a Size control's label, the keyboard's focus
        // arriving on its slider, or a drag of its slider, selects its row; a hover over a label
        // shows its hover text and selects nothing (C-2). A label's click and the focus edit
        // nothing.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let tile = harness.panel.selection;
        let label_at = |output: &egui::FullOutput, label: &str| {
            text_rects(output, label)
                .into_iter()
                .find(|rect| rect.right() <= INPUTS_WIDTH)
                .unwrap_or_else(|| panic!("no {label:?} in the Size column"))
                .center()
        };
        for row in size_controls::SIZE_ROWS {
            let output = harness.frame(Vec::new());
            harness.frame(vec![egui::Event::PointerMoved(label_at(
                &output, row.label,
            ))]);
            harness.frame(Vec::new());
            assert_eq!(harness.panel.selection, tile, "a hover on {:?}", row.label);
        }
        for row in size_controls::SIZE_ROWS {
            harness.panel.selection = Selection::Nothing;
            let output = harness.frame(Vec::new());
            harness.click(label_at(&output, row.label));
            assert_eq!(
                harness.panel.selection,
                Selection::Input(row.key),
                "{:?}",
                row.label
            );
        }
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "a click edits nothing"
        );
        harness.panel.selection = Selection::Nothing;
        harness.frame(Vec::new());
        // Tab from the slider before it (a focus requested between frames is not one that
        // arrives) onto the gap's slider.
        let gap = harness.size_widget(GAP).id;
        let length = harness.size_widget(LENGTH).id;
        harness.ctx.memory_mut(|m| m.request_focus(length));
        harness.frame(Vec::new());
        harness.panel.selection = Selection::Nothing;
        for _ in 0..6 {
            if harness.ctx.memory(|m| m.has_focus(gap)) {
                break;
            }
            harness.frame(key_tap(egui::Key::Tab));
        }
        assert!(
            harness.ctx.memory(|m| m.has_focus(gap)),
            "Tab reaches the gap"
        );
        harness.frame(Vec::new());
        assert_eq!(
            harness.panel.selection,
            Selection::Input(FACE_GAP),
            "the keyboard's focus"
        );
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "the focus edits nothing"
        );
        harness.panel.selection = Selection::Nothing;
        harness.frame(Vec::new());
        let at = harness.size_widget(GAP).rect.left_center() + egui::vec2(20.0, 0.0);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(vec![primary_button(at, true)]);
        harness.frame(vec![egui::Event::PointerMoved(at + egui::vec2(12.0, 0.0))]);
        harness.frame(vec![primary_button(at + egui::vec2(12.0, 0.0), false)]);
        assert_eq!(
            harness.panel.selection,
            Selection::Input(FACE_GAP),
            "a drag"
        );
    }

    #[test]
    fn the_size_column_draws_its_controls_and_two_folds_and_no_line_about_them() {
        // UX overhaul spec 4.3: each control is its label and its slider; what left the column
        // (every hint, Set-by, floor and what-moves-what line, the Why links and the footnote
        // on what the controls move) is drawn nowhere until a row is selected. Under the
        // controls, the folds Detailed inputs and Assumptions, both closed.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explorer::reviewed_note;
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        let shown = harness.panel.shown_inputs();
        let consequences = shown_consequences(&harness);
        let size = dimensions(&shown).expect("the starting design has size controls");
        let mut gone: Vec<String> = SIZE_TAB_HINTS
            .iter()
            .map(|&(_, key)| hint_for(key).expect("a hint").to_owned())
            .collect();
        gone.extend(consequences.magnet_lines(shown.coupling.npole, harness.panel.auto()));
        gone.extend(consequences.diameter_line());
        gone.extend(consequences.length_line());
        gone.push(size_controls::floor_line(size.minimum_diameter_range.min));
        gone.push(size_controls::LINKED_HOUSING.to_owned());
        for id in [SIZE_TAB_NOTE, TORQUE_BAND_NOTE] {
            gone.push(why_note_text(reviewed_note(id).expect("a reviewed note")));
        }
        for text in &gone {
            assert_eq!(count(&output, text), 0, "{text:?} is drawn");
        }
        let column = |text: &str| {
            text_rects(&output, text)
                .into_iter()
                .find(|rect| rect.right() <= INPUTS_WIDTH)
                .unwrap_or_else(|| panic!("no {text:?} in the Size column"))
        };
        for row in size_controls::SIZE_ROWS {
            column(row.label);
        }
        let detailed = column(DETAILED_INPUTS);
        let assumptions = column(ASSUMPTIONS_FOLD);
        assert!(
            detailed.bottom() <= assumptions.top(),
            "{detailed:?} {assumptions:?}"
        );
        assert_eq!(
            count(&output, ASSUMPTIONS_NOTE),
            0,
            "the Assumptions fold starts closed"
        );
    }

    #[test]
    fn size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter() {
```

Replace:

```rust
        // Spec 6 test 10, the owner's case of 2026-10-13: from 10 to 8 magnets the coupling did
        // not shrink and nothing said why. At 8 magnets the Size tab says the ring keeps its
        // size, and its Set-by line shows the cap as the largest part: the outside diameter is
        // the cap's, so fewer magnets cannot lower it.
        let mut harness = size_tab(SCREEN, with_poles(8));
```

with:

```rust
        // Spec 6 test 10, the owner's case of 2026-10-13: from 10 to 8 magnets the coupling did
        // not shrink and nothing said why. At 8 magnets Magnets per ring's entry says the ring
        // keeps its size, and Outside diameter's Set-by line shows the cap as the largest part:
        // the outside diameter is the cap's, so fewer magnets cannot lower it (in the Explain
        // panel since UX overhaul spec 4.3).
        use crate::gui::explain::WHAT_SETS_IT;
        let mut harness = size_tab(SCREEN, with_poles(8));
```

Replace:

```rust
        let mut harness = size_tab(SCREEN, with_poles(8));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, RING_SIZE_STAYS), 1);
        let shown = harness.panel.shown_inputs();
```

with:

```rust
        let mut harness = size_tab(SCREEN, with_poles(8));
        let shown = harness.panel.shown_inputs();
```

Replace:

```rust
        assert!(set_by.starts_with("Set by: cap "), "{set_by}");
        assert_eq!(count(&output, &set_by), 1);
```

with:

```rust
        assert!(set_by.starts_with("Set by: cap "), "{set_by}");
        let output = entry(&mut harness, DIAMETER_HINT_KEY);
        assert_eq!(count(&output, &set_by), 1);
```

Replace:

```rust
        assert_eq!(count(&output, &set_by), 1);
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        assert!(
```

with:

```rust
        assert_eq!(count(&output, &set_by), 1);
        assert!(
```

Replace:

```rust
        assert!(
            rect(DIAMETER).bottom() <= rect(set_by.as_str()).top(),
            "under Outside diameter"
        );
```

with:

```rust
        assert!(
            explain_rect(&output, WHAT_SETS_IT).bottom() <= explain_rect(&output, &set_by).top(),
            "in Outside diameter's entry, under What sets it"
        );
```

Replace:

```rust
        );
        assert!(
            rect(set_by.as_str()).bottom() <= rect(LENGTH).top(),
            "above Overall length"
        );
        assert!(
            rect(MAGNETS).bottom() <= rect(RING_SIZE_STAYS).top(),
            "under Magnets per ring"
        );
        // T5/T6 amendment: with Minimum OD unset the slider floor, not the coupling, stops the
```

with:

```rust
        );
        let output = entry(&mut harness, POLES);
        assert_eq!(count(&output, RING_SIZE_STAYS), 1);
        // T5/T6 amendment: with Minimum OD unset the slider floor, not the coupling, stops the
```

Replace:

```rust
        assert_eq!(count(&output, &fit), 1, "{fit}");
        // Slice 3: two lines under Magnets per ring, and the torque card above the drawing
        // answers the owner's question in numbers (decision 7A moved the neighbour lines there):
        // every one of its lines is drawn.
        assert_eq!(consequences.magnet_lines(8, None).len(), 2);
```

with:

```rust
        assert_eq!(count(&output, &fit), 1, "{fit}");
        // Slice 3: two lines for Magnets per ring, and the torque card answers the owner's
        // question in numbers (decision 7A moved the neighbour lines there): the entry draws
        // the card's count row, and the Hot-low tile's entry every one of its lines.
        assert_eq!(consequences.magnet_lines(8, None).len(), 2);
```

Replace:

```rust
        assert_eq!(consequences.magnet_lines(8, None).len(), 2);
        for line in consequences.card.lines(None) {
```

with:

```rust
        assert_eq!(consequences.magnet_lines(8, None).len(), 2);
        for line in consequences
            .card
            .control_lines(Control::MagnetsPerRing, None)
        {
            assert_eq!(count(&output, &line), 1, "{line}");
        }
        harness.panel.selection = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
        let output = harness.frame(Vec::new());
        for line in consequences.card.lines(None) {
```

Replace:

```rust
    fn size_tab_first_magnet_line_follows_the_auto_state() {
        // Slice 3 spec 4.3 (physics review 3, N4, as the card leaves it): the line under
        // Magnets per ring says what a new count does to the ring: it stays with Auto off and
        // with Auto on the magnet length; with Auto on the ring size the torque card's Auto line
        // says Auto sizes the ring again (decision 5A), and with Auto on the count (its slider
        // locked) no line speaks of changing it. The lines under the control are drawn one under
        // the next, and the card draws the Auto line of the state and no other.
        use crate::gui::torque_card::{AUTO_HOLDS_COUNT, AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING};
```

with:

```rust
    fn size_tab_first_magnet_line_follows_the_auto_state() {
        // Slice 3 spec 4.3 (physics review 3, N4, as the card leaves it): Magnets per ring's
        // first line says what a new count does to the ring: it stays with Auto off and with
        // Auto on the magnet length; with Auto on the ring size the torque card's Auto line says
        // Auto sizes the ring again (decision 5A), and with Auto on the count (its slider
        // locked) no line speaks of changing it. The control's lines are drawn one under the
        // next in its entry, and the card draws the Auto line of the state and no other.
        use crate::gui::explain::WHAT_SETS_IT;
        use crate::gui::torque_card::{AUTO_HOLDS_COUNT, AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING};
```

Replace:

```rust
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(
                count(&output, RING_SIZE_STAYS),
```

with:

```rust
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            for line in [AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING, AUTO_HOLDS_COUNT] {
                let wanted = usize::from(Some(line) == auto_line);
                assert_eq!(count(&output, line), wanted, "{auto:?}: {line}");
            }
            let output = entry(&mut harness, POLES);
            // The entry's card row keeps the same Auto line (owner 5A), between the caption
            // and the row.
            for line in [AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING, AUTO_HOLDS_COUNT] {
                let wanted = usize::from(Some(line) == auto_line);
                assert_eq!(count(&output, line), wanted, "{auto:?}: the entry's {line}");
            }
            assert_eq!(
                count(&output, RING_SIZE_STAYS),
```

Replace:

```rust
            );
            for line in [AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING, AUTO_HOLDS_COUNT] {
                let wanted = usize::from(Some(line) == auto_line);
                assert_eq!(count(&output, line), wanted, "{auto:?}: {line}");
            }
            let shown = harness.panel.shown_inputs();
```

with:

```rust
            );
            let shown = harness.panel.shown_inputs();
```

Replace:

```rust
            );
            let rect =
                |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
            let mut above = rect(MAGNETS);
            for line in &lines {
```

with:

```rust
            );
            let mut above = explain_rect(&output, WHAT_SETS_IT);
            for line in &lines {
```

Replace:

```rust
                assert_eq!(count(&output, line), 1, "{auto:?}: {line}");
                let at = rect(line.as_str());
                assert!(above.bottom() <= at.top() + 0.5, "{auto:?}: {line:?}");
```

with:

```rust
                assert_eq!(count(&output, line), 1, "{auto:?}: {line}");
                let at = explain_rect(&output, line);
                assert!(above.bottom() <= at.top() + 0.5, "{auto:?}: {line:?}");
```

Replace:

```rust
        // it), from the magnet count's own slider and from any other input; so is the torque
        // card that took over the neighbour lines (slice 3, decision 7A).
        let mut harness = size_tab(SCREEN, DesignInputs::default());
```

with:

```rust
        // it), from the magnet count's own slider and from any other input; so is the torque
        // card that took over the neighbour lines (slice 3, decision 7A). The arrow key selects
        // the control, whose entry draws its lines and its card row (UX overhaul spec 4.3); the
        // other controls' entries and the Hot-low tile's whole card follow too.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
```

Replace:

```rust
        let at_10 = shown_consequences(&harness).magnet_lines(10, None);
        let card_at_10 = card_lines(&harness);
        // The controls' widgets in drawing order: diameter, length, gap, magnets, Minimum OD.
```

with:

```rust
        let at_10 = shown_consequences(&harness).magnet_lines(10, None);
        let row = |c: &Consequences| c.card.control_lines(Control::MagnetsPerRing, None);
        let row_at_10 = row(&shown_consequences(&harness));
        // The controls' widgets in drawing order: diameter, length, gap, magnets, Minimum OD.
```

Replace:

```rust
        assert_eq!(harness.panel.inputs().coupling.npole, 8);
        let at_8 = shown_consequences(&harness).magnet_lines(8, None);
```

with:

```rust
        assert_eq!(harness.panel.inputs().coupling.npole, 8);
        assert_eq!(harness.panel.selection, Selection::Input(POLES));
        let at_8 = shown_consequences(&harness).magnet_lines(8, None);
```

Replace:

```rust
        assert_ne!(at_8, at_10);
        for line in &at_8 {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in at_10.iter().filter(|line| !at_8.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        let card_at_8 = card_lines(&harness);
        assert_ne!(card_at_8, card_at_10);
        for line in &card_at_8 {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in card_at_10.iter().filter(|line| !card_at_8.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
```

with:

```rust
        assert_ne!(at_8, at_10);
        let row_at_8 = row(&shown_consequences(&harness));
        assert_ne!(row_at_8, row_at_10);
        for (now, then) in [(&at_8, &at_10), (&row_at_8, &row_at_10)] {
            for line in now {
                assert_eq!(count(&output, line), 1, "{line}");
            }
            for line in then.iter().filter(|line| !now.contains(line)) {
                assert_eq!(count(&output, line), 0, "stale: {line}");
            }
        }
```

Replace:

```rust
        assert_ne!(lines, at_8);
        for line in &lines {
            assert_eq!(count(&output, line), 1, "{line}");
        }
```

with:

```rust
        assert_ne!(lines, at_8);
        let row_wider = row(&wider);
        assert_ne!(row_wider, row_at_8);
        for (now, then) in [(&lines, &at_8), (&row_wider, &row_at_8)] {
            for line in now {
                assert_eq!(count(&output, line), 1, "{line}");
            }
            for line in then.iter().filter(|line| !now.contains(line)) {
                assert_eq!(count(&output, line), 0, "stale: {line}");
            }
        }
```

Replace:

```rust
        }
        for line in at_8.iter().filter(|line| !lines.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        let card_wider = wider.card.lines(None);
        assert_ne!(card_wider, card_at_8);
        for line in &card_wider {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in card_at_8.iter().filter(|line| !card_wider.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        for line in wider.diameter_line().into_iter().chain(wider.length_line()) {
            assert_eq!(count(&output, &line), 1, "{line}");
```

with:

```rust
        }
        let output = entry(&mut harness, DIAMETER_HINT_KEY);
        let diameter = wider.diameter_line().expect("a Set-by line");
        assert_eq!(count(&output, &diameter), 1, "{diameter}");
        let output = entry(&mut harness, LENGTH_HINT_KEY);
        let length = wider.length_line().expect("a length split");
        assert_eq!(count(&output, &length), 1, "{length}");
        harness.panel.selection = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
        let output = harness.frame(Vec::new());
        for line in wider.card.lines(None) {
            assert_eq!(count(&output, &line), 1, "{line}");
```

Replace:

```rust
        let mut harness = size_tab(SCREEN, with_poles(8));
        assert_eq!(harness.panel.minimum_diameter_mm, None);
```

with:

```rust
        let mut harness = size_tab(SCREEN, with_poles(8));
        harness.panel.selection = Selection::Input(POLES);
        assert_eq!(harness.panel.minimum_diameter_mm, None);
```

Replace:

```rust
        // Spec 5, as the torque card's count row has it (slice 3, decision 7A): at 4 or 40
        // magnets per ring the side past the slider reads END_OF_RANGE, drawn on the card, the
        // side inside it is a step, and no text names 2 or 42 magnets. The default blocks do
        // not fit 40 magnets (the card reads its one line there); forty narrow blocks do.
```

with:

```rust
        // Spec 5, as the torque card's count row has it (slice 3, decision 7A): at 4 or 40
        // magnets per ring the side past the slider reads END_OF_RANGE, drawn in the control's
        // entry as its card row, the side inside it is a step, and no text names 2 or 42 magnets. The default blocks do
        // not fit 40 magnets (the card reads its one line there); forty narrow blocks do.
```

Replace:

```rust
            let mut harness = size_tab(SCREEN, inputs);
            let output = harness.frame(Vec::new());
            let consequences = shown_consequences(&harness);
```

with:

```rust
            let mut harness = size_tab(SCREEN, inputs);
            let output = entry(&mut harness, POLES);
            let consequences = shown_consequences(&harness);
```

Replace:

```rust
            assert_eq!(count(&output, &count_line), 1, "{count_line}");
            // Two lines under Magnets per ring at a slider end (three before the card): the
            // ring-size line and the fit line.
```

with:

```rust
            assert_eq!(count(&output, &count_line), 1, "{count_line}");
            // Two lines for Magnets per ring at a slider end (three before the card): the
            // ring-size line and the fit line.
```

Replace:

```rust
        // Spec 5 "Narrow windows", at the M4-1 review's ~930 px window and the owner's 8 magnets:
        // the what-moves-what lines wrap inside the inputs side (its 320 points), one under the
        // next, the Required row is that side's too, and the tiles stay in the centre region,
        // left of the right column.
        let mut harness = size_tab(egui::vec2(930.0, 1024.0), with_poles(8));
```

with:

```rust
        // Spec 5 "Narrow windows", at the M4-1 review's ~930 px window and the owner's 8 magnets:
        // the what-moves-what lines wrap inside the right column, in their controls' entries
        // (UX overhaul spec 4.3), one under the next; the Required row is the inputs side's
        // (its 320 points), and the tiles stay in the centre region, left of the right column.
        use crate::gui::explain::WHAT_SETS_IT;
        let mut harness = size_tab(egui::vec2(930.0, 1024.0), with_poles(8));
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        let side = |id: &str| {
            egui::containers::panel::PanelState::load(&harness.ctx, egui::Id::new(id))
```

with:

```rust
        let output = harness.frame(Vec::new());
        let side = |harness: &Harness, id: &str| {
            egui::containers::panel::PanelState::load(&harness.ctx, egui::Id::new(id))
```

Replace:

```rust
        };
        let inputs_left = side("magcoupling_inputs").left();
        let inputs_right = inputs_left + INPUTS_WIDTH;
```

with:

```rust
        };
        let inputs_left = side(&harness, "magcoupling_inputs").left();
        let inputs_right = inputs_left + INPUTS_WIDTH;
```

Replace:

```rust
        let inputs_right = inputs_left + INPUTS_WIDTH;
        let dashboard_left = side("magcoupling_dashboard").left();
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
```

with:

```rust
        let inputs_right = inputs_left + INPUTS_WIDTH;
        let dashboard_left = side(&harness, "magcoupling_dashboard").left();
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
```

Replace:

```rust
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        let consequences = shown_consequences(&harness);
        let magnet_lines = consequences.magnet_lines(8, None);
        let mut above = rect(MAGNETS);
        for line in &magnet_lines {
            let at = rect(line.as_str());
            assert!(
                above.bottom() <= at.top() + 0.5,
                "{line:?} overlaps the line above it"
            );
            above = at;
        }
        let mut lines = magnet_lines.clone();
        lines.extend(consequences.diameter_line());
        lines.extend(consequences.length_line());
        for line in &lines {
            let at = rect(line.as_str());
            assert!(
                at.left() >= inputs_left - 1.0 && at.right() <= inputs_right + 1.0,
                "{line:?} at {at:?} runs out of the inputs side ({inputs_left} to {inputs_right})"
            );
        }
        assert!(
            lines.iter().any(|line| rows(&output, line) > 1),
            "no line wrapped: the test no longer exercises wrapping"
        );
        let results = harness.panel.results().clone();
```

with:

```rust
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        let results = harness.panel.results().clone();
```

Replace:

```rust
            "{required:?}"
        );
    }

```

with:

```rust
            "{required:?}"
        );
        let consequences = shown_consequences(&harness);
        let mut wrapped = false;
        for (key, lines) in [
            (POLES, consequences.magnet_lines(8, None)),
            (
                DIAMETER_HINT_KEY,
                consequences.diameter_line().into_iter().collect(),
            ),
            (
                LENGTH_HINT_KEY,
                consequences.length_line().into_iter().collect(),
            ),
        ] {
            assert!(!lines.is_empty(), "{key}");
            let output = entry(&mut harness, key);
            let right = side(&harness, "magcoupling_dashboard");
            let mut above = explain_rect(&output, WHAT_SETS_IT);
            for line in &lines {
                let at = explain_rect(&output, line);
                assert!(
                    above.bottom() <= at.top() + 0.5,
                    "{line:?} overlaps the line above it"
                );
                assert!(
                    at.left() >= right.left() - 1.0 && at.right() <= right.right() + 1.0,
                    "{line:?} at {at:?} runs out of the right column ({right:?})"
                );
                wrapped |= rows(&output, line) > 1;
                above = at;
            }
        }
        assert!(
            wrapped,
            "no line wrapped: the test no longer exercises wrapping"
        );
    }

```

Replace:

```rust
        // default, a met requirement, both ends of the magnet slider, the owner's 8 magnets, no
        // prediction (short magnets) and a ring too small for its magnets. Every drawn text has
        // glyphs in the default fonts; the new ones name no workbook cell, correction or sheet.
        let mut met = DesignInputs::default();
```

with:

```rust
        // default, a met requirement, both ends of the magnet slider, the owner's 8 magnets, no
        // prediction (short magnets) and a ring too small for its magnets; the controls' lines
        // drawn in their entries (UX overhaul spec 4.3). Every drawn text has glyphs in the
        // default fonts; the new ones name no workbook cell, correction or sheet.
        let mut met = DesignInputs::default();
```

Replace:

```rust
            CARD_HEADING.to_owned(),
        ];
```

with:

```rust
            CARD_HEADING.to_owned(),
            size_controls::MINIMUM_HOVER.to_owned(),
            crate::gui::explain::SPREADSHEET_CELL.to_owned(),
        ];
```

Replace:

```rust
        ];
        for (inputs, minimum) in states {
```

with:

```rust
        ];
        // The Hot-low tile's entry (the whole card) and each Size control's (UX overhaul spec
        // 4.3: the lines that were drawn under the controls).
        let selections = std::iter::once(Selection::Result(crate::gui::tiles::Tile::HotLow.path()))
            .chain(size_controls::SIZE_ROWS.map(|row| Selection::Input(row.key)));
        let selections: Vec<Selection> = selections.collect();
        for (inputs, minimum) in states {
```

Replace:

```rust
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            let shown = harness.panel.shown_inputs();
```

with:

```rust
            harness.frame(Vec::new());
            let mut all = Vec::new();
            for &selection in &selections {
                harness.panel.selection = selection;
                all.extend(drawn_texts(&harness.frame(Vec::new())));
            }
            let shown = harness.panel.shown_inputs();
```

Replace:

```rust
            texts.extend(consequences.card.lines(None));
            if dimensions(&shown).is_some() {
                texts.extend(consequences.diameter_line());
```

with:

```rust
            texts.extend(consequences.card.lines(None));
            if let Some(size) = dimensions(&shown) {
                texts.extend(consequences.diameter_line());
```

Replace:

```rust
                texts.extend(consequences.length_line());
            }
```

with:

```rust
                texts.extend(consequences.length_line());
                texts.push(size_controls::floor_line(size.minimum_diameter_range.min));
            }
```

Replace:

```rust
            }
            texts.push(pullout_label(results.metal.op_temp_C));
```

with:

```rust
            }
            texts.push(size_controls::LINKED_HOUSING.to_owned());
            texts.push(pullout_label(results.metal.op_temp_C));
```

Replace:

```rust
            }
            let all = drawn_texts(&output);
            for text in &texts {
```

with:

```rust
            }
            for text in &texts {
```

Replace:

```rust
    fn the_size_tab_s_why_links_open_their_notes() {
        // Spec 3.7 (S2-8): under Magnets per ring and Outside diameter, and on the torque card's
        // heading (slice 3 spec 6.1; in the Hot-low tile's Explain entry, which `size_tab`
        // shows), a link to how the coupling makes torque; beside the Required row's label a
        // link to the hot minimum check's note (the Hot-low tile's is its entry's lesson link).
        // Each shows its note in the Explain panel (spec 5.2).
        use crate::gui::dashboard::why_note_text;
```

with:

```rust
    fn the_size_tab_s_why_links_open_their_notes() {
        // Spec 3.7 (S2-8), moved by UX overhaul spec 4.3: the Why links the Size column drew are
        // its rows' entries' lessons: Outside diameter's and Magnets per ring's how the coupling
        // makes torque, Required torque's the hot minimum check's note; the torque card's
        // heading keeps its link (slice 3 spec 6.1; in the Hot-low tile's entry). Each shows its
        // note in the Explain panel (spec 5.2). The column draws no Why link.
        use crate::gui::dashboard::why_note_text;
```

Replace:

```rust
        // note in the Explain panel (spec 5.2). The column draws no Why link.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explorer::reviewed_note;
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let tile = harness.panel.selection;
```

with:

```rust
        // note in the Explain panel (spec 5.2). The column draws no Why link.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explain::FULL_LESSON;
        use crate::gui::explorer::reviewed_note;
        use crate::gui::typeset::glyph_safe;
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let tile = harness.panel.selection;
```

Replace:

```rust
        let tile = harness.panel.selection;
        for (id, links) in [(SIZE_TAB_NOTE, 3), (TORQUE_BAND_NOTE, 1)] {
            let text =
                why_note_text(reviewed_note(id).expect("a Size-tab link's note is reviewed"));
            for index in 0..links {
                harness.panel.selection = tile;
                let output = harness.frame(Vec::new());
                // The Required and Hot-low links end wrapped rows: click their text.
                let at = last_characters(&output, &text);
                assert_eq!(at.len(), links, "{text}");
                harness.click(at[index]);
                assert_eq!(
                    harness.panel.selection,
                    Selection::Note(id),
                    "{text} #{index}"
                );
            }
        }
```

with:

```rust
        let tile = harness.panel.selection;
        let note = |id| reviewed_note(id).expect("a Size-tab link's note is reviewed");
        harness.panel.selection = Selection::Nothing;
        let column = harness.frame(Vec::new());
        for id in [SIZE_TAB_NOTE, TORQUE_BAND_NOTE] {
            assert_eq!(count(&column, &why_note_text(note(id))), 0, "{id}");
        }
```

Replace:

```rust
            assert_eq!(count(&column, &why_note_text(note(id))), 0, "{id}");
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }
```

with:

```rust
            assert_eq!(count(&column, &why_note_text(note(id))), 0, "{id}");
        }
        for (key, id) in [
            (DIAMETER_HINT_KEY, SIZE_TAB_NOTE),
            (POLES, SIZE_TAB_NOTE),
            (REQUIREMENT_INPUT, TORQUE_BAND_NOTE),
        ] {
            let output = entry(&mut harness, key);
            explain_rect(&output, FULL_LESSON);
            harness.click(explain_rect(&output, &glyph_safe(note(id).title)).center());
            assert_eq!(harness.panel.selection, Selection::Note(id), "{key}");
        }
        harness.panel.selection = tile;
        let output = harness.frame(Vec::new());
        let at = last_characters(&output, &why_note_text(note(SIZE_TAB_NOTE)));
        assert_eq!(at.len(), 1, "the card's heading's link");
        harness.click(at[0]);
        assert_eq!(harness.panel.selection, Selection::Note(SIZE_TAB_NOTE));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }
```

Replace:

```rust
    fn without_the_diameter_control_the_size_tab_still_links_from_magnets_per_ring() {
        // An odd magnet count has no diameter range: the Outside diameter control and its link
        // are not drawn; the link under Magnets per ring is, and the torque card's heading's (the
        // card keeps its gap and count rows, slice 3 T-5).
        use crate::gui::dashboard::why_note_text;
```

with:

```rust
    fn without_the_diameter_control_the_size_tab_still_links_from_magnets_per_ring() {
        // An odd magnet count has no diameter range: the Outside diameter control is not drawn;
        // Magnets per ring's entry still links its note, and the torque card's heading its (the
        // card keeps its gap and count rows, slice 3 T-5). Outside diameter's entry, if it was
        // selected before the count went odd, draws no value and no card row.
        use crate::gui::dashboard::why_note_text;
```

Replace:

```rust
        // selected before the count went odd, draws no value and no card row.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explorer::reviewed_note;
        use crate::gui::guidance::SIZE_TAB_NOTE;
```

with:

```rust
        // selected before the count went odd, draws no value and no card row.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explain::WHAT_A_CHANGE_DOES;
        use crate::gui::explorer::reviewed_note;
        use crate::gui::guidance::SIZE_TAB_NOTE;
```

Replace:

```rust
        use crate::gui::guidance::SIZE_TAB_NOTE;
        let odd = with_poles(7);
```

with:

```rust
        use crate::gui::guidance::SIZE_TAB_NOTE;
        use crate::gui::typeset::glyph_safe;
        let odd = with_poles(7);
```

Replace:

```rust
        let mut harness = size_tab(SCREEN, odd);
        let output = harness.frame(Vec::new());
```

with:

```rust
        let mut harness = size_tab(SCREEN, odd);
        let note = reviewed_note(SIZE_TAB_NOTE).unwrap();
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        let text = why_note_text(reviewed_note(SIZE_TAB_NOTE).unwrap());
        assert_eq!(text_rects(&output, &text).len(), 2);
    }
```

with:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(text_rects(&output, &why_note_text(note)).len(), 1);
        assert_eq!(count(&output, DIAMETER), 0, "no Outside diameter control");
        let output = entry(&mut harness, POLES);
        assert_eq!(count(&output, &glyph_safe(note.title)), 1);
        let output = entry(&mut harness, DIAMETER_HINT_KEY);
        assert_eq!(count(&output, WHAT_A_CHANGE_DOES), 0);
        assert_eq!(count(&output, size_controls::DIAMETER_HOVER), 1);
    }
```

Replace:

```rust
        // outside it (spec 3.4). Idle frames compute nothing; a gap edit computes once (the frame
        // that draws it asks with the edited design, the next frame's controls hit); a toggle of
        // the clamp pause computes once; so does a new Minimum OD, which the drawn Outside
```

with:

```rust
        // outside it (spec 3.4). Idle frames compute nothing; a gap edit computes once (the frame
        // that draws it asks with the edited design, the next frames hit); a toggle of
        // the clamp pause computes once; so does a new Minimum OD, which the drawn Outside
```

Replace:

```rust
        // Outside diameter row reads END_OF_RANGE on its -1 mm side, where with none set it read
        // that the magnets do not fit.
        use crate::gui::torque_card::END_OF_RANGE;
```

with:

```rust
        // Outside diameter row reads END_OF_RANGE on its -1 mm side, where with none set it read
        // that the magnets do not fit. Outside diameter's entry draws the row (UX overhaul spec
        // 4.3); the gap edit above selected the gap's.
        use crate::gui::torque_card::END_OF_RANGE;
```

Replace:

```rust
        use crate::gui::torque_card::END_OF_RANGE;
        let diameter_row = |harness: &Harness| {
```

with:

```rust
        use crate::gui::torque_card::END_OF_RANGE;
        harness.panel.selection = Selection::Input(DIAMETER_HINT_KEY);
        let diameter_row = |harness: &Harness| {
```

**`magcoupling-rs/src/gui/size_controls.rs`**

Replace:

```rust
        let pinned = |inputs: &DesignInputs| {
            let consequences = Consequences::of(
                inputs,
                &compute_all(inputs),
                None,
                crate::engine::clamps::PAUSED,
            );
            let mut minimum = None;
```

with:

```rust
        let pinned = |inputs: &DesignInputs| {
            let mut minimum = None;
```

Replace:

```rust
            sized_frame(&ctx, SCREEN, Vec::new(), |ui| {
                minimum = controls_ui(ui, inputs, None, &consequences, None).minimum_diameter_mm;
            });
```

with:

```rust
            sized_frame(&ctx, SCREEN, Vec::new(), |ui| {
                minimum = controls_ui(ui, inputs, None, None).minimum_diameter_mm;
            });
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 15
  ```

  Expected: the build fails with 1 errors, the first "unresolved imports `panel::ASSUMPTIONS_FOLD`, `panel::DETAILED_INPUTS`": the green step adds what the tests call.

- [ ] **Step 3 (green): the column, the touch, the folds.**

**`magcoupling-rs/src/gui/guidance.rs`**

Replace:

```rust

/// Where the Size tab's Why links under Magnets per ring and Outside diameter (spec 3.7) and on
/// the torque card's heading (slice 3 spec 6.1) lead: how the coupling makes torque, Start
/// here's first step.
pub const SIZE_TAB_NOTE: &str = "torque_from_shear";
```

with:

```rust

/// Where the Why links of Magnets per ring and Outside diameter (spec 3.7; their Explain
/// entries' lessons since UX overhaul spec 4.3) and the torque card's heading's (slice 3 spec
/// 6.1) lead: how the coupling makes torque, Start here's first step.
pub const SIZE_TAB_NOTE: &str = "torque_from_shear";
```

Replace:

```rust

/// Where the Why links at the end of the Size tab's Required and Hot-low rows lead: the hot
/// minimum check's note (a test keeps it [`HOT_MIN_GUIDE`]'s).
pub const TORQUE_BAND_NOTE: &str = "hot_low_cold_high";
```

with:

```rust

/// Where the Why links of the Required torque row (its Explain entry's lesson since UX overhaul
/// spec 4.3) and of the hot-low torque lead: the hot minimum check's note (a test keeps it
/// [`HOT_MIN_GUIDE`]'s).
pub const TORQUE_BAND_NOTE: &str = "hot_low_cold_high";
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
use crate::gui::clamp_drawing::clamp_ui;
use crate::gui::dashboard::{Level, end_effect_banner_ui, why_note_link};
use crate::gui::explain::{Asked, LEARN, Selection, explain_ui};
```

with:

```rust
use crate::gui::clamp_drawing::clamp_ui;
use crate::gui::dashboard::{Level, end_effect_banner_ui};
use crate::gui::explain::{Asked, LEARN, Selection, explain_ui};
```

Replace:

```rust
use crate::gui::guidance::{
    ConsequenceCache, Consequences, Direction, EXAMPLE_DESIGN_HOVER, FixCache, SIZE_TAB_ALIASES,
    TORQUE_BAND_NOTE, within_minimum_diameter,
};
```

with:

```rust
use crate::gui::guidance::{
    ConsequenceCache, Direction, EXAMPLE_DESIGN_HOVER, FixCache, SIZE_TAB_ALIASES,
    within_minimum_diameter,
};
```

Replace:

```rust

/// What the inputs side shows (decision M43-7): the design inputs, or the model assumptions.
```

with:

```rust

/// The Size column's fold of every design input (UX overhaul spec 4.3), over its fold of the
/// model's assumptions, which bears the Assumptions view's name.
pub const DETAILED_INPUTS: &str = "Detailed inputs";
pub const ASSUMPTIONS_FOLD: &str = InputsView::Assumptions.label();

/// What the inputs side shows (decision M43-7): the design inputs, or the model assumptions.
```

Replace:

```rust

/// The results of the design the Size tab last asked its [`ConsequenceCache`] about, kept with
/// that design (slice 3 spec 3.5): the lines under the controls and the torque card each ask
/// once a frame, mostly about one design, so its `compute_all` runs once per design, not once
/// per ask.
#[derive(Default)]
```

with:

```rust

/// The results of the design the Explain panel last asked its [`ConsequenceCache`] about, kept
/// with that design (slice 3 spec 3.5): a Size control's entry and the Hot-low tile's ask once
/// a frame, mostly about one design, so its `compute_all` runs once per design, not once per
/// ask.
#[derive(Default)]
```

Replace:

```rust
    /// design shown, the minimum OD and the shaft-clamp pause, recomputed only when one of the
    /// three changes: view state, in no design file, link or undo step. Asked only through
    /// [`MagcouplingPanel::size_tab_consequences`].
    consequences: ConsequenceCache,
```

with:

```rust
    /// design shown, the minimum OD and the shaft-clamp pause, recomputed only when one of the
    /// three changes: view state, in no design file, link or undo step. Asked only by the
    /// Explain panel, for an entry that reads them (`checks_and_explain_ui`).
    consequences: ConsequenceCache,
```

Replace:

```rust
    consequences: ConsequenceCache,
    /// The results [`MagcouplingPanel::size_tab_consequences`] hands `consequences`, kept with
    /// the design they were computed for: a cache like it, in no design file, link or undo step.
    size_tab_results: ResultsCache,
```

with:

```rust
    consequences: ConsequenceCache,
    /// The results the Explain panel hands `consequences`, kept with the design they were
    /// computed for: a cache like it, in no design file, link or undo step.
    size_tab_results: ResultsCache,
```

Replace:

```rust

    /// The Size tab's what-moves-what lines and torque card for `shown`, the design shown (slice
    /// 3 spec 3.5): the one cache ask of the lines under the controls (the design before this
    /// frame's edits) and of the card (the design after them), so both ask with one key, the
    /// design, the panel's Minimum OD and its shaft-clamp pause (spec 3.2). The cache is handed
    /// that design's own results, not `self.results`, which can lag the inputs by a frame (the
    /// header's assumptions reset edits them after the last compute): a lagging number would
    /// stick in the lines until the next edit. `size_tab_results` keeps them with their design,
    /// so they are computed once per design, not once per ask: none on an idle frame, one on an
    /// edit's. The torque card's stepped runs and the fit search rerun only when the design, the
    /// minimum or the pause changes.
    fn size_tab_consequences(&mut self, shown: &DesignInputs) -> &Consequences {
        let results = self.size_tab_results.get(shown);
        self.consequences
            .get(shown, results, self.minimum_diameter_mm, self.clamps_paused)
    }

    /// The Size tab's Auto outcome line and its colour: what the solve changed, from the
```

with:

```rust

    /// The Size tab's Auto outcome line and its colour: what the solve changed, from the
```

Replace:

```rust
        let auto = self.auto();
        // Asked as the Size column asks (`size_tab_consequences`), and only for an entry that
        // reads them: the cache computes once per design.
        let consequences = self.selection.reads_consequences().then(|| {
```

with:

```rust
        let auto = self.auto();
        // The Size tab's what-moves-what lines and torque card for the design shown (slice 3
        // spec 3.5), asked only for an entry that reads them, with one key: the design, the
        // panel's Minimum OD and its shaft-clamp pause (spec 3.2). The cache is handed that
        // design's own results, not `self.results`, which can lag the inputs by a frame (the
        // header's assumptions reset edits them after the last compute). `size_tab_results`
        // keeps them with their design, so they are computed once per design, not once per
        // ask: none on an idle frame, one on an edit's.
        let consequences = self.selection.reads_consequences().then(|| {
```

Replace:

```rust

    /// The Size column of the opening tab: its heading, the Required torque row, Auto and its
    /// outcome, the size controls with their lines, the detailed inputs, folded, then the
    /// footnote on what the size controls move.
    fn size_column_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

with:

```rust

    /// The Size column of the opening tab (UX overhaul spec 4.3): its heading, the Required
    /// torque row, Auto and its outcome, the size controls, then two folds, both closed until a
    /// fix link or a term asks for a row in one: the detailed inputs and the assumptions. A
    /// control touched selects its row, whose lines show in the Explain panel.
    fn size_column_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
```

Replace:

```rust
        ui.heading(crate::gui::size_controls::HEADING);
        let requirement = self.requirement_ui(ui);
        // Auto (spec guidance 3.5) is the solver's one home. Turning it off adopts the value
```

with:

```rust
        ui.heading(crate::gui::size_controls::HEADING);
        let (requirement, required_touched) = self.requirement_ui(ui);
        // Auto (spec guidance 3.5) is the solver's one home. Turning it off adopts the value
```

Replace:

```rust
        let minimum = self.minimum_diameter_mm;
        // Asked before the controls write the minimum back, with the design before this
        // frame's edits.
        let consequences = self.size_tab_consequences(&shown);
        let output = size_controls::controls_ui(ui, &shown, minimum, consequences, auto);
        // A Why link under Magnets per ring or Outside diameter (spec 3.7) opens its note.
        if let Some(id) = output.note {
            self.show_note(ui.ctx(), id);
        }
```

with:

```rust
        let minimum = self.minimum_diameter_mm;
        let output = size_controls::controls_ui(ui, &shown, minimum, auto);
        let touched = output
            .touched
            .or(required_touched.then_some(REQUIREMENT_INPUT));
        if let Some(key) = touched {
            self.selection = Selection::Input(key);
        }
```

Replace:

```rust
        }
        // The lines under the controls were drawn from the design before this frame's edit:
        // one more frame shows the edited design's.
        if output.inputs != shown || output.minimum_diameter_mm != minimum {
```

with:

```rust
        }
        // The top bar was drawn before this frame's edit: one more frame shows its state.
        if output.inputs != shown || output.minimum_diameter_mm != minimum {
```

Replace:

```rust
        ui.separator();
        egui::CollapsingHeader::new("Detailed inputs")
            .default_open(false)
```

with:

```rust
        ui.separator();
        // A row asked for (a fix link, a term) opens the fold that holds it: an assumption's
        // row is in the Assumptions fold, every other in the detailed inputs.
        let assumption = self
            .explorer
            .focus()
            .filter(|focus| focus.scroll)
            .map(|focus| {
                ASSUMPTIONS
                    .iter()
                    .any(|a| a.paths.contains(&focus.path.as_str()))
            });
        egui::CollapsingHeader::new(DETAILED_INPUTS)
            .default_open(false)
```

Replace:

```rust
            .default_open(false)
            .open(
                self.explorer
                    .focus()
                    .filter(|focus| focus.scroll)
                    .map(|_| true),
            )
            .show(ui, |ui| self.design_inputs_ui(ui, readouts));
```

with:

```rust
            .default_open(false)
            .open((assumption == Some(false)).then_some(true))
            .show(ui, |ui| self.design_inputs_ui(ui, readouts));
```

Replace:

```rust
            .show(ui, |ui| self.design_inputs_ui(ui, readouts));
        // Under the fold, so the fold is on the first view (plan review E-4).
        ui.label(size_controls::LINKED_HOUSING);
    }
```

with:

```rust
            .show(ui, |ui| self.design_inputs_ui(ui, readouts));
        egui::CollapsingHeader::new(ASSUMPTIONS_FOLD)
            .default_open(false)
            .open((assumption == Some(true)).then_some(true))
            .show(ui, |ui| self.assumptions_ui(ui, readouts));
    }
```

Replace:

```rust
    /// first of the Size tab's readouts): its label, quieter than the value in the slider's box
    /// ([`look::quiet_text`]), with the Why link to the hot minimum check's note (spec 3.7),
    /// then the requirement's slider, an ordinary design edit (undo, files, links). Returns the
    /// slider's widget: its value box is a text field of the design, as an input row's is.
    fn requirement_ui(&mut self, ui: &mut egui::Ui) -> egui::Id {
        let entry = InputCatalogue::get()
```

with:

```rust
    /// first of the Size tab's readouts): its label, quieter than the value in the slider's box
    /// ([`look::quiet_text`]), then the requirement's slider, an ordinary design edit (undo,
    /// files, links). Its Why link is its Explain entry's lesson. Returns the slider's widget
    /// (its value box is a text field of the design, as an input row's is) and whether a person
    /// touched the row ([`size_controls::touched`]).
    fn requirement_ui(&mut self, ui: &mut egui::Ui) -> (egui::Id, bool) {
        let entry = InputCatalogue::get()
```

Replace:

```rust
        let required = self.inputs.metal.required_min_Nm;
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(look::quiet_text(ui.visuals()), size_controls::REQUIRED);
            if why_note_link(ui, TORQUE_BAND_NOTE) {
                self.show_note(ui.ctx(), TORQUE_BAND_NOTE);
            }
        });
        let mut value = required;
```

with:

```rust
        let required = self.inputs.metal.required_min_Nm;
        let clicked = ui
            .add(
                egui::Label::new(
                    egui::RichText::new(size_controls::REQUIRED)
                        .color(look::quiet_text(ui.visuals())),
                )
                .sense(egui::Sense::click()),
            )
            .clicked();
        let mut value = required;
```

Replace:

```rust
        }
        response.id
    }
```

with:

```rust
        }
        (response.id, clicked || size_controls::touched(&response))
    }
```

**`magcoupling-rs/src/gui/size_controls.rs`**

Replace:

```rust
//!
//! Under each control, its hint line ([`hint_line`], slice 2 spec 3.5) and its what-moves-what
//! lines (spec 3.3: [`Consequences`], computed by the guidance layer); the lines for two magnets
//! fewer and two more are the torque card's count row since slice 3 (decision 7A). The panel's
//! centre view draws the torque readouts (spec 3.2) with the texts here: [`REQUIRED`],
//! [`HOT_LOW`] with [`torque_gap_text`], [`pullout_label`], and [`NO_PREDICTION`] or
//! [`NO_TORQUE`] where the model makes none ([`predicts`]), then the torque card.

```

with:

```rust
//!
//! Each control is its label and its slider (UX overhaul spec 4.3); touching one ([`touched`])
//! selects it, and its hint line, its what-moves-what lines (spec 3.3, computed by the guidance
//! layer), its floor line, [`LINKED_HOUSING`] and its Why link show in the Explain panel's entry
//! of its row ([`SIZE_ROWS`]). The lines for two magnets fewer and two more are the torque
//! card's count row since slice 3 (decision 7A). The tiles draw the torques with the texts
//! here: [`REQUIRED`], [`HOT_LOW`] with [`torque_gap_text`], [`pullout_label`], and
//! [`NO_PREDICTION`] or [`NO_TORQUE`] where the model makes none ([`predicts`]).

```

Replace:

```rust
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::{end_effect_out_of_range, why_note_link};
use crate::gui::format::{
```

with:

```rust
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::end_effect_out_of_range;
use crate::gui::format::{
```

Replace:

```rust
};
use crate::gui::guidance::{Consequences, FACE_GAP, SIZE_TAB_NOTE, TORQUE_BAND_NOTE};
use crate::gui::input_ui::{hint_line, slider, weak_line};
use crate::gui::inputs::{InputCatalogue, slider_range};
```

with:

```rust
};
use crate::gui::guidance::{FACE_GAP, SIZE_TAB_NOTE, TORQUE_BAND_NOTE};
use crate::gui::input_ui::slider;
use crate::gui::inputs::{InputCatalogue, slider_range};
```

Replace:

```rust
pub const MAGNETS: &str = "Magnets per ring";
/// The Size tab's footnote (O-7): what the diameter and length sliders move
/// ([`crate::engine::size_controls`]).
pub const LINKED_HOUSING: &str = "Outside diameter moves the ring radius; the cap and its thread follow the cup. Overall length sets both rings' magnet length, and the housing follows. Gap and magnet count stay as you set them.";
```

with:

```rust
pub const MAGNETS: &str = "Magnets per ring";
/// What the size controls move (O-7; [`crate::engine::size_controls`]): the footnote under the
/// Size column until UX overhaul spec 4.3, a line of each size control's Explain entry since.
pub const LINKED_HOUSING: &str = "Outside diameter moves the ring radius; the cap and its thread follow the cup. Overall length sets both rings' magnet length, and the housing follows. Gap and magnet count stay as you set them.";
```

Replace:

```rust
    format!("Shaft and housing range floor: {floor_mm:.1} mm")
}
```

with:

```rust
    format!("Shaft and housing range floor: {floor_mm:.1} mm")
}

/// Whether a person touched a control this frame (UX overhaul spec 5.2): a click, the start of
/// a drag, a change (an arrow key, a typed value) or the keyboard's focus arriving.
pub(crate) fn touched(response: &egui::Response) -> bool {
    response.clicked() || response.drag_started() || response.changed() || response.gained_focus()
}

/// A control's label, which shows `hover` and selects its row on a click.
pub(crate) fn row_label(ui: &mut egui::Ui, row: &SizeRow, hover: &str) -> bool {
    ui.add(egui::Label::new(row.label).sense(egui::Sense::click()))
        .on_hover_text(hover)
        .clicked()
}
```

Replace:

```rust
    pub error: Option<String>,
    /// The note a Why link under Magnets per ring or Outside diameter asked for this frame.
    pub note: Option<&'static str>,
}
```

with:

```rust
    pub error: Option<String>,
    /// The key of the row a person touched this frame ([`touched`], [`SizeRow::key`]).
    pub touched: Option<&'static str>,
}
```

Replace:

```rust
/// Draw before computing the panel's results, so the readouts reflect this frame's edits.
/// `inputs` is the design shown. `consequences` are its what-moves-what lines (spec 3.3), each
/// drawn under its control: they show the design before this frame's edit, and the panel asks
/// for the next frame. With `auto`, the slider of the size Auto sets ([`variable_slider`]) is
/// locked at its value there: disabled, saying why.
pub(crate) fn controls_ui(
```

with:

```rust
/// Draw before computing the panel's results, so the readouts reflect this frame's edits.
/// `inputs` is the design shown. Each control is its label and its slider; the lines about it
/// are its row's Explain entry. With `auto`, the slider of the size Auto sets
/// ([`variable_slider`]) is locked at its value there: disabled, saying why.
pub(crate) fn controls_ui(
```

Replace:

```rust
    minimum_diameter_mm: Option<f64>,
    consequences: &Consequences,
    auto: Option<FreeVariable>,
```

with:

```rust
    minimum_diameter_mm: Option<f64>,
    auto: Option<FreeVariable>,
```

Replace:

```rust
        error: None,
        note: None,
    };
```

with:

```rust
        error: None,
        touched: None,
    };
```

Replace:

```rust
    };
    let catalogue = InputCatalogue::get();
```

with:

```rust
    };
    let row = |key: &str| size_row(key).expect("every control has its row");
    let catalogue = InputCatalogue::get();
```

Replace:

```rust
            let mut minimum = slider_minimum(&legacy, minimum_diameter_mm);
            ui.label(MINIMUM).on_hover_text(MINIMUM_HOVER);
            let response = ui
```

with:

```rust
            let mut minimum = slider_minimum(&legacy, minimum_diameter_mm);
            let clicked = row_label(ui, row(MINIMUM_OD_HINT_KEY), MINIMUM_HOVER);
            let response = ui
```

Replace:

```rust
            minimum_widget = Some(response.id);
            hint_line(ui, MINIMUM_OD_HINT_KEY);
            if response.changed() {
```

with:

```rust
            minimum_widget = Some(response.id);
            if clicked || touched(&response) {
                output.touched = Some(MINIMUM_OD_HINT_KEY);
            }
            if response.changed() {
```

Replace:

```rust
                .expect("a bounded minimum defines a diameter range");
            for (label, hover, path, hint, mut value, range, mapping, line) in [
                (
```

with:

```rust
                .expect("a bounded minimum defines a diameter range");
            for (key, hover, mut value, range, mapping) in [
                (
```

Replace:

```rust
                (
                    DIAMETER,
                    DIAMETER_HOVER,
```

with:

```rust
                (
                    DIAMETER_HINT_KEY,
                    DIAMETER_HOVER,
```

Replace:

```rust
                    DIAMETER_HOVER,
                    "coupling.inner_back_apothem_mm",
                    DIAMETER_HINT_KEY,
                    size.diameter_mm,
```

with:

```rust
                    DIAMETER_HOVER,
                    size.diameter_mm,
```

Replace:

```rust
                    SliderMapping::Diameter { minimum },
                    consequences.diameter_line(),
                ),
```

with:

```rust
                    SliderMapping::Diameter { minimum },
                ),
```

Replace:

```rust
                (
                    LENGTH,
                    LENGTH_HOVER,
```

with:

```rust
                (
                    LENGTH_HINT_KEY,
                    LENGTH_HOVER,
```

Replace:

```rust
                    LENGTH_HOVER,
                    "coupling.magnets.axial_length_mm",
                    LENGTH_HINT_KEY,
                    size.overall_length_mm,
```

with:

```rust
                    LENGTH_HOVER,
                    size.overall_length_mm,
```

Replace:

```rust
                    SliderMapping::Length,
                    consequences.length_line(),
                ),
```

with:

```rust
                    SliderMapping::Length,
                ),
```

Replace:

```rust
            ] {
                ui.label(label).on_hover_text(hover);
                let entry = catalogue
```

with:

```rust
            ] {
                let row = row(key);
                let path = row.input.expect("a size control writes an input");
                let clicked = row_label(ui, row, hover);
                let entry = catalogue
```

Replace:

```rust
                    .expect("a size control has input metadata");
                let response =
                    size_slider(ui, path, &mut value, entry.meta, range, is_locked(label));
                output.widgets.push(response.id);
```

with:

```rust
                    .expect("a size control has input metadata");
                let response = size_slider(
                    ui,
                    path,
                    &mut value,
                    entry.meta,
                    range,
                    is_locked(row.label),
                );
                output.widgets.push(response.id);
```

Replace:

```rust
                output.widgets.push(response.id);
                hint_line(ui, hint);
                if response.changed() {
```

with:

```rust
                output.widgets.push(response.id);
                if clicked || touched(&response) {
                    output.touched = Some(row.key);
                }
                if response.changed() {
```

Replace:

```rust
                }
                if let Some(line) = line {
                    weak_line(ui, &line);
                }
                if label == DIAMETER && why_note_link(ui, SIZE_TAB_NOTE) {
                    output.note = Some(SIZE_TAB_NOTE);
                }
            }
```

with:

```rust
                }
            }
```

Replace:

```rust
            }
            ui.weak(format!(
                "Shaft and housing range floor: {:.1} mm",
                range.min
            ));
        } else {
```

with:

```rust
            }
        } else {
```

Replace:

```rust
        }
        for (label, path, lines) in [
            (GAP, FACE_GAP, Vec::new()),
            (
                MAGNETS,
                FreeVariable::MagnetsPerRing.path(),
                consequences.magnet_lines(inputs.coupling.npole, auto),
            ),
        ] {
            let entry = catalogue
```

with:

```rust
        }
        for path in [FACE_GAP, FreeVariable::MagnetsPerRing.path()] {
            let row = row(path);
            let entry = catalogue
```

Replace:

```rust
            };
            ui.label(label).on_hover_text(help_for(path));
            let range = slider_range(path).expect("a size control has a slider");
```

with:

```rust
            };
            let clicked = row_label(ui, row, help_for(path));
            let range = slider_range(path).expect("a size control has a slider");
```

Replace:

```rust
            let range = slider_range(path).expect("a size control has a slider");
            let response = size_slider(ui, path, &mut value, entry.meta, range, is_locked(label));
            output.widgets.push(response.id);
```

with:

```rust
            let range = slider_range(path).expect("a size control has a slider");
            let response = size_slider(
                ui,
                path,
                &mut value,
                entry.meta,
                range,
                is_locked(row.label),
            );
            output.widgets.push(response.id);
```

Replace:

```rust
            output.widgets.push(response.id);
            hint_line(ui, path);
            if response.changed() {
```

with:

```rust
            output.widgets.push(response.id);
            if clicked || touched(&response) {
                output.touched = Some(row.key);
            }
            if response.changed() {
```

Replace:

```rust
                }
            }
            for line in &lines {
                weak_line(ui, line);
            }
            if label == MAGNETS && why_note_link(ui, SIZE_TAB_NOTE) {
                output.note = Some(SIZE_TAB_NOTE);
            }
```

with:

```rust
                }
            }
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

  Expected: `test result: ok. 1094 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both `grep -vc` lines print `0` (no test binary with a failure); both wasm32 clippy runs clean; `cargo fmt --check` prints nothing; the verdict test's last line `all 77 checks hold`.

- [ ] **Step 5: Commit** the five files: `feat(magcoupler): the Size column draws its controls and two folds; touching a control selects its entry, where its lines now are (BL-090 T2)`.

### Task 3: Docs sweep

**Model:** `sonnet`.

The README's Size column passages; `docs/ai/` (02 the layout, 03 the modules, 04 the issue, 05 the tracker, the backlog item in progress); the register's rows A-25 to A-27 and its log; owner question 51; the design's status line.

**Files:**
- Modify: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/analyses/2026-11-18-owner-questions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

- [ ] **Step 1: The docs.**

**`docs/ai/02-system.yaml`**

Replace:

```yaml
      - "The session baseline and the two named designs (UX overhaul slice A, BL-077; design docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md sections 7.1 and 7.2): DesignInputs::default() is the example design, the spreadsheet's values, and the overhaul never moves it. The standalone app opens on app::starting_design() (seven inputs off the spreadsheet's, among them the standard 1.4 mm hub keyway, and its own fourteen stored 3D fields; Required torque 1.35 N·m; its test pins it to compute_all: every counted check Good, no caution, no material warning, no assumption changed) and hands it to the panel as its baseline (MagcouplingPanel::with_baseline; new() keeps the spreadsheet's values). On screen 'the default' is the baseline: the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's 'Starting design'. The spreadsheet export's changed flag, the assumptions banner and the Assumptions view's dots stay against the workbook. Nothing moves the baseline (no undo, file, link or opened design), and it is in no file, link or undo step. Design files and links do not depend on it: they name every input, and a missing input still reads the spreadsheet's value. On the web the app writes app::address_of(the settled design) into the page's address with history.replaceState: the clean address at the starting design, the Share link elsewhere; Share always copies the full link. It writes only while no key is held (a held Ctrl+Z or Ctrl+Y settles a step at every key repeat; the run gives one write at the release), and address_design advances only when the browser took the write (Safari refuses more than about 100 in 10 seconds); a refused write is asked again on the next frame, with no back-off. If a later physics fix makes the starting design fail, it gets a new starting design, never a looser test. Owner rulings of 2026-11-22 (BL-080; docs/analyses/2026-11-10-conservative-assumptions.md rows A-17 and A-18): its hub keyway is the standard 1.4 mm (DIN 6885-1 t2 for a 3 x 3 mm key), so the wall past it is 2.55 mm and its opening tab draws no caution; its temperature check and the inner hub's fitting temperature read 3D fields computed for its own geometry (STARTING_3D_FIELDS, a run of reference/magcoupling-py/tools/starting_design_fields.py pinned by a test), which stay fixed under an edit like the example design's."
      - "Since BL-085 (UX overhaul slice B, stage 2, part 2a; plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md): the top bar is Undo, Redo, Design (a menu: the reset, Open the example design, Save design, Load design, Show spreadsheet details), Share, Export and Learn, each a word with no box at rest (panel.rs bar_button, design_menu_ui, reset_label: Reset to the starting design where the baseline is not the spreadsheet's design). The Explain panel draws the teaching notes (decision C-11): explain::Selection gains Note and NoteList; every note link (ReadoutEvents::note, the Size column's Why links, explorer_ui's return: the lesson line under an open equation, a Start here step) sets the selection; the Equation panel has no Explain checkbox, no note view and no button on the bar. format::value_with_unit: a text value takes no unit (design 9.2)."
```

with:

```yaml
      - "The session baseline and the two named designs (UX overhaul slice A, BL-077; design docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md sections 7.1 and 7.2): DesignInputs::default() is the example design, the spreadsheet's values, and the overhaul never moves it. The standalone app opens on app::starting_design() (seven inputs off the spreadsheet's, among them the standard 1.4 mm hub keyway, and its own fourteen stored 3D fields; Required torque 1.35 N·m; its test pins it to compute_all: every counted check Good, no caution, no material warning, no assumption changed) and hands it to the panel as its baseline (MagcouplingPanel::with_baseline; new() keeps the spreadsheet's values). On screen 'the default' is the baseline: the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's 'Starting design'. The spreadsheet export's changed flag, the assumptions banner and the Assumptions view's dots stay against the workbook. Nothing moves the baseline (no undo, file, link or opened design), and it is in no file, link or undo step. Design files and links do not depend on it: they name every input, and a missing input still reads the spreadsheet's value. On the web the app writes app::address_of(the settled design) into the page's address with history.replaceState: the clean address at the starting design, the Share link elsewhere; Share always copies the full link. It writes only while no key is held (a held Ctrl+Z or Ctrl+Y settles a step at every key repeat; the run gives one write at the release), and address_design advances only when the browser took the write (Safari refuses more than about 100 in 10 seconds); a refused write is asked again on the next frame, with no back-off. If a later physics fix makes the starting design fail, it gets a new starting design, never a looser test. Owner rulings of 2026-11-22 (BL-080; docs/analyses/2026-11-10-conservative-assumptions.md rows A-17 and A-18): its hub keyway is the standard 1.4 mm (DIN 6885-1 t2 for a 3 x 3 mm key), so the wall past it is 2.55 mm and its opening tab draws no caution; its temperature check and the inner hub's fitting temperature read 3D fields computed for its own geometry (STARTING_3D_FIELDS, a run of reference/magcoupling-py/tools/starting_design_fields.py pinned by a test), which stay fixed under an edit like the example design's."
      - "Since BL-090 (UX overhaul slice B, stage 2, part 2b-i; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md): the Size column draws each control's label and slider and nothing about them, then the folds Detailed inputs and Assumptions (closed; an assumption's row asked for opens the second). Touching a control (size_controls::touched: a click, a drag, an edit, the focus arriving; never a hover) or clicking its label sets explain::Selection::Input(key), key from size_controls::SIZE_ROWS; the entry (explain::input_ui) draws the lines the column drew (the hint, the ConsequenceCache's what-moves-what lines, the floor line, LINKED_HOUSING), the control's torque-card row (TorqueCard::control_lines), the checks whose fixes name its input, guidance::input_trade_offs, its Why note and, with the details, its cell. Only the Explain panel asks the ConsequenceCache now, for the Hot-low tile and a Size row; the Size column asks nothing (size_tab_consequences is gone)."
      - "Since BL-085 (UX overhaul slice B, stage 2, part 2a; plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md): the top bar is Undo, Redo, Design (a menu: the reset, Open the example design, Save design, Load design, Show spreadsheet details), Share, Export and Learn, each a word with no box at rest (panel.rs bar_button, design_menu_ui, reset_label: Reset to the starting design where the baseline is not the spreadsheet's design). The Explain panel draws the teaching notes (decision C-11): explain::Selection gains Note and NoteList; every note link (ReadoutEvents::note, the Size column's Why links, explorer_ui's return: the lesson line under an open equation, a Start here step) sets the selection; the Equation panel has no Explain checkbox, no note view and no button on the bar. format::value_with_unit: a text value takes no unit (design 9.2)."
```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel: new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption)), input_ui.rs (input_row (a locked row keeps its label live, to trace) -> RowOutput {edit, widget, label_clicked}, slider, RowEdit, TRACE_HINT, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the inputs side; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

with:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel: new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption)), input_ui.rs (input_row (a locked row keeps its label live, to trace) -> RowOutput {edit, widget, label_clicked}, slider, RowEdit, TRACE_HINT, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PlotKind with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the inputs side; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 1 (BL-081, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md, decisions D-1 to D-19): the check list over the Explain panel and four tiles over the drawing. Stage 2 is cut in two (D-1): part 2 is the top bar and Design menu, Learn, the Size column's folds and its hint lines moving into the Explain panel with inputs as selections, dimension tags as selections with the callout list leaving the opening tab (D-19), the Plots tab with its picker, the word budget of 250 and the slip-heating unit. A check's row shows a shape and a word, not a short form of its quantity (D-2, register row A-20, owner question 49). risk: physics for the orientation and the tile's needs line, so every task runs on fable."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
```

with:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 1 (BL-081, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md, decisions D-1 to D-19): the check list over the Explain panel and four tiles over the drawing. Stage 2 is cut in two (D-1): part 2 is the top bar and Design menu, Learn, the Size column's folds and its hint lines moving into the Explain panel with inputs as selections, dimension tags as selections with the callout list leaving the opening tab (D-19), the Plots tab with its picker, the word budget of 250 and the slip-heating unit. A check's row shows a shape and a word, not a short form of its quantity (D-2, register row A-20, owner question 49). risk: physics for the orientation and the tile's needs line, so every task runs on fable."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-i (BL-090, plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md, decisions D-34 to D-48): the Size column's lines in the Explain panel, its six rows as selections, the Assumptions fold. Part 2b is cut in three (D-34, register row A-25); 2b-ii is the other selections, the callout list's move and the word budget of 250; 2b-iii the Plots tab, the outlines, the face, the page shell and BL-086, BL-088, BL-089. 'Set by Auto' stays under the locked slider (D-37, A-26, owner question 51); 'Checks it touches' reads the fixes on the design shown (D-39, A-27). risk: mechanical."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2a (BL-085, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md, decisions D-20 to D-33): the top bar (Undo, Redo, Design, Share, Export, Learn) and the teaching notes in the Explain panel; the Equation panel's Explain checkbox and bar button go; each Start here step selects its note; a text value takes no unit. Part 2b, with its own plan and item: the Size column's folds and lines in the Explain panel with inputs as selections, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell. Open: owner question 50 (the status line's place; register row A-24)."
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown

---

## 2026-11-22 - Magcoupler UX overhaul, slice B, stage 2, part 2a: the top bar and the notes (BL-085)
```

with:

```markdown

---

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-i: the Size column's lines in the Explain panel (BL-090)
- What changed: the Size column draws each control's label and slider, then the folds Detailed inputs and Assumptions (the Assumptions view is now one click from the opening tab). Touching a control selects it, and the Explain panel shows its entry: its value, what it is, what sets it (its hint, the what-moves-what lines, Minimum OD's floor, the line on what the controls move), its torque-card row, the checks whose fixes name it, the "pulls two ways" sentence where it applies, its Why note, and with spreadsheet details on its cell. The hint lines, the what-moves-what lines, the floor line, the Why links and the footnote no longer show under the controls.
- Tests: each row's entry (explain), touching selects and a hover does not, the column draws no moved line, the Assumptions fold opens on an assumption's row; the size tab's line tests read the entries; the first view holds at most 280 words (272).
- Not in this part (plan decision D-34): dimension tags, results-table rows, callout rows and Detailed inputs rows as selections, the callout list to the Geometry tab, the word budget of 250 (part 2b-ii); the Plots tab, the outlines, the SemiBold face, the page shell, BL-086, BL-088, BL-089 (part 2b-iii).

## 2026-11-22 - Magcoupler UX overhaul, slice B, stage 2, part 2a: the top bar and the notes (BL-085)
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  priority: 2
  status: open
  notes: "Filed 2026-11-23 with its plan (the design's header: each plan files one item). BL-085's part 2b is cut in three (the plan's decision D-34, register row A-25): this part, 2b-i, is the Size column; 2b-ii (dimension tags, results-table rows, callout rows and the Detailed inputs' rows as selections, with the trace as a link in an input's entry; the callout list to the Geometry tab; the word budget of 250 and which copies of a Start here step's title go, D-32; the table-driven nothing-lost test; gui-smoke reading an input's entry; the 'Spreadsheet cell' foot of a result's and a check's entry; the left column no longer changing with the centre tab, 4.3; a fix link selecting and focusing its input, 5.2) and 2b-iii (the Plots tab's picker C-9, the plots' red reference lines C-3, egui_plot's fading tick numbers E-7, the controls' outlines E-4, a SemiBold face S-1, the page shell S-4, BL-086, BL-088 and BL-089) each get their own plan and item. BL-083 and BL-084 stay open (the design gives neither text). risk: mechanical (decision D-46): the entries draw texts the column drew, each reviewed; the one new on-screen string is 'Spreadsheet cell:', the design's own words; no number is new."
- id: BL-055
```

with:

```yaml
  priority: 2
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2b-i (Tasks 1 and 2 of the plan); the full gate, the smoke run and the whole-branch review are its Task 4. Filed 2026-11-23 with its plan (the design's header: each plan files one item). BL-085's part 2b is cut in three (the plan's decision D-34, register row A-25): this part, 2b-i, is the Size column; 2b-ii (dimension tags, results-table rows, callout rows and the Detailed inputs' rows as selections, with the trace as a link in an input's entry; the callout list to the Geometry tab; the word budget of 250 and which copies of a Start here step's title go, D-32; the table-driven nothing-lost test; gui-smoke reading an input's entry; the 'Spreadsheet cell' foot of a result's and a check's entry; the left column no longer changing with the centre tab, 4.3; a fix link selecting and focusing its input, 5.2) and 2b-iii (the Plots tab's picker C-9, the plots' red reference lines C-3, egui_plot's fading tick numbers E-7, the controls' outlines E-4, a SemiBold face S-1, the page shell S-4, BL-086, BL-088 and BL-089) each get their own plan and item. BL-083 and BL-084 stay open (the design gives neither text). risk: mechanical (decision D-46): the entries draw texts the column drew, each reviewed; the one new on-screen string is 'Spreadsheet cell:', the design's own words; no number is new."
- id: BL-055
```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
| A-24 | The design (4.2) puts the status line for file and link results at the right end of the top bar. Today it is a line under the bar. Move it? | It stays under the bar. | Its place is read by tests and by the smoke run; a long status ("Share link copied to the clipboard (2,690 characters)") at the bar's end would wrap the bar at 1280 points. Owner question 50. | It moves to the bar's right end. | The plan of part 2a (BL-085), decision D-22 |
| A-22 | BL-082: below what skirt does the cap draw a caution? The owner's answer 4a (2026-11-22) took an amber caution below a minimum and the controller's offered 0.5 mm; no source gives a making limit for a thread-cut skirt of this size. | 0.5 mm, a guide value (`gui::geometry::MIN_CAP_SKIRT_MM`), against the nominal skirt (a tapped thread's root sits a few hundredths of a mm past its nominal diameter, so the real wall is that much thinner: about 0.52 to 0.53 mm, still over 0.5, for the starting design's nominal 0.55 mm on a fine thread). A skirt under it, compared as quoted at four significant digits, draws one amber line on the Size tab naming the skirt and the minimum; it counts as no failed check. The starting design's 0.55 mm draws none. | A caution, not a fail: it warns without moving the approved starting design or any verdict, and it says only that the wall is thin against a guide, not that it cannot be made. 0.5 mm keeps the starting design clear; a minimum between 0.55 and 0.70 mm would force a new starting design (owner question 4). | A larger minimum moves the starting design; a counted check would change the status line's count; no caution reverts BL-082. | `magcoupling-rs/src/gui/geometry.rs`, BL-082 |
```

with:

```markdown
| A-24 | The design (4.2) puts the status line for file and link results at the right end of the top bar. Today it is a line under the bar. Move it? | It stays under the bar. | Its place is read by tests and by the smoke run; a long status ("Share link copied to the clipboard (2,690 characters)") at the bar's end would wrap the bar at 1280 points. Owner question 50. | It moves to the bar's right end. | The plan of part 2a (BL-085), decision D-22 |
| A-25 | BL-085's part 2b (the Size column, more selections, the Plots tab, the word budget, the controls' outlines, the second face, the page shell, BL-086, BL-088, BL-089) is again more than one plan a reviewer can hold. Built at once or in parts? | Three parts. 2b-i: the Size column's lines in the Explain panel, its six rows as selections, the Assumptions fold. 2b-ii: dimension tags, results-table rows, callout rows and the Detailed inputs' rows as selections, the callout list to the Geometry tab, the word budget of 250, the table-driven nothing-lost test. 2b-iii: the Plots tab, the outlines, the face, the page shell, BL-086, BL-088, BL-089. | Smaller releases (the conservative answer to Q-4, as A-21 and A-23); 2b-i ships alone and loses nothing. | One plan for all of part 2b. | The plan of part 2b-i (BL-090), decision D-34 |
| A-26 | The design (4.3) moves every "Set by" line under a Size control into the Explain panel. Does the line "Set by Auto; turn Auto off to edit" under the slider Auto locks move too? | It stays under the slider. The "Set by" lines that move are the what-moves-what lines ("Set by: cap 42.8, cup 41.3, boss 22.0 mm"). | It says, at the slider, why a drag does nothing; in the Explain panel it would show only after a click on a slider that does not move. Owner question 51. | It moves into the control's entry, under "What sets it". | The plan of part 2b-i (BL-090), decision D-37 |
| A-27 | The design's 5.3 gives an input's entry "Checks it touches: the guides whose fixes name it". The fixes on the design shown, or every fix a guide could offer? | The guides whose fixes on the design shown name its input; on a passing design there are none and the heading is left out. Each is a link that selects the check. | The fix lists (`all_fixes`) are per design and empty for a passing check; a list of every fix a guide could ever offer needs a new table, a new text and would name checks that cannot move. | A table of each guide's possible fixes, shown on every design. | The plan of part 2b-i (BL-090), decision D-39 |
| A-22 | BL-082: below what skirt does the cap draw a caution? The owner's answer 4a (2026-11-22) took an amber caution below a minimum and the controller's offered 0.5 mm; no source gives a making limit for a thread-cut skirt of this size. | 0.5 mm, a guide value (`gui::geometry::MIN_CAP_SKIRT_MM`), against the nominal skirt (a tapped thread's root sits a few hundredths of a mm past its nominal diameter, so the real wall is that much thinner: about 0.52 to 0.53 mm, still over 0.5, for the starting design's nominal 0.55 mm on a fine thread). A skirt under it, compared as quoted at four significant digits, draws one amber line on the Size tab naming the skirt and the minimum; it counts as no failed check. The starting design's 0.55 mm draws none. | A caution, not a fail: it warns without moving the approved starting design or any verdict, and it says only that the wall is thin against a guide, not that it cannot be made. 0.5 mm keeps the starting design clear; a minimum between 0.55 and 0.70 mm would force a new starting design (owner question 4). | A larger minimum moves the starting design; a counted check would change the status line's count; no caution reverts BL-082. | `magcoupling-rs/src/gui/geometry.rs`, BL-082 |
```

Replace:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-22: A-23 and A-24 added with the plan of slice B's stage 2, part 2a (BL-085). Each is an assumption, not the owner's answer.
```

with:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-23: A-25, A-26 and A-27 added with the plan of slice B's stage 2, part 2b-i (BL-090). Each is an assumption, not the owner's answer.
- 2026-11-22: A-23 and A-24 added with the plan of slice B's stage 2, part 2a (BL-085). Each is an assumption, not the owner's answer.
```

**`docs/analyses/2026-11-18-owner-questions.md`**

Replace:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 31 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22). Part D, five that matter only if the tools stop being generic.

```

with:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 32 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, question 51 on 2026-11-23). Part D, five that matter only if the tools stop being generic.

```

Replace:

```markdown

### Linkage simulator
```

with:

```markdown

### 51. Should "Set by Auto; turn Auto off to edit" stay under the slider Auto locks?

Added 2026-11-23 (the next free number). With Auto on, the slider of the size Auto sets cannot be dragged, and a line under it says so. Part 2b-i moves every other line under the Size controls into the Explain panel (the design's 4.3) and keeps this one under the slider.

- **If you do not answer:** it stays under the slider.
- (a) Under the slider, as now.
- (b) In the control's Explain entry only (a click on the locked slider shows it).

Source: UX design section 4.3; A-26; BL-090 plan, decision D-37.

### Linkage simulator
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

with:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; parts 2b-ii and 2b-iii and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

**`magcoupling-rs/README.md`**

Replace:

```markdown
build checks stay in the check list. Torque is unavailable outside the end-effect model's
range. **Detailed inputs** keeps every original input, and all results, equations and
exports remain available. **Auto: smallest [magnet length | ring radius | magnet count] that
meets the required torque** (the hot-low torque; it does not look at the other checks)
```

with:

```markdown
build checks stay in the check list. Torque is unavailable outside the end-effect model's
range. Under the controls two folds, both closed at first: **Detailed inputs** keeps every
original input, and **Assumptions** is the Assumptions view, which the opening tab could not
reach before part 2b-i of the UX overhaul (BL-090, plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`); a fix
link or a term that asks for an assumption's row opens the second, any other row's the first. All
results, equations and exports remain available. **Auto: smallest [magnet length | ring radius | magnet count] that
meets the required torque** (the hot-low torque; it does not look at the other checks)
```

Replace:

```markdown

Under each of the five controls (Minimum OD, Outside diameter, Overall length, Magnet face gap
and Magnets per ring) a weak hint line of at most 12 words says what it does
(`gui::plain_text::HINTS`, slice 2; the gap's and the magnet count's are their Key design rows'
hints, the other three are keyed `SIZE_TAB_HINT_KEYS`, which are not input paths). The Minimum
OD hint replaced the line under the controls that said the same.

```

with:

```markdown

Since BL-090 each control is its label and its slider, and touching it (a click on the label, a
click or drag of the slider, an edit, or the keyboard's focus arriving; a hover selects nothing)
selects its row: the Explain panel shows the row's entry (`explain::Selection::Input`, keyed by
`size_controls::SIZE_ROWS`: the input path where the row is one input, the hint key otherwise).
The entry leads with the value and the label, then "What it is" (the label's hover text), "What
sets it" (the lines below, which the column drew under the control until then, and for the four
size controls the line "Outside diameter moves the ring radius; ...", once the column's
footnote), "What a change does" (the control's row of the torque card, under the card's heading
and caption: `TorqueCard::control_lines`), "Checks it touches" (the checks whose fixes on the
design shown name its input, each a link that selects the check), "Watch out" (where two of those
checks move it opposite ways, the "pulls two ways" sentence of their entries:
`guidance::input_trade_offs`), "Full lesson:" with the note its Why link opened (Required torque,
Outside diameter, Magnets per ring), and with Show spreadsheet details on a weak line
"Spreadsheet cell: ..." for a row that is one input. The line "Set by Auto; turn Auto off to edit"
stays under the slider Auto locks (decision D-37, register row A-26).

```

Replace:

```markdown

Under the controls, short lines say what moves what, each read from the engine (the diameters
and lengths from `engine::size_controls`), never typed. For **Magnets per ring**: a first line
that follows the Auto state (`guidance::magnet_count_line`, slice 3 spec 4.3): with Auto off and
```

with:

```markdown

Each of the five controls (Minimum OD, Outside diameter, Overall length, Magnet face gap and
Magnets per ring) has a hint of at most 12 words that says what it does, the first line under its
entry's "What sets it" (`gui::plain_text::HINTS`, slice 2; the gap's and the magnet count's are
their Key design rows' hints, the other three are keyed `SIZE_TAB_HINT_KEYS`, which are not input
paths). Minimum OD's entry adds the floor of the diameter range ("Shaft and housing range floor:
... mm").

Short lines say what moves what, each read from the engine (the diameters and lengths from
`engine::size_controls`), never typed. For **Magnets per ring**: a first line
that follows the Auto state (`guidance::magnet_count_line`, slice 3 spec 4.3): with Auto off and
```

Replace:

```markdown
after the frame's edits, from the `ConsequenceCache` (one key: the design, the Minimum OD and the
shaft-clamp pause; the Auto marking is applied outside the cache), and asks it only while the Hot-low tile is selected. Until BL-081 it started open, but closed while the Equation panel was open (a click on its own Why link opens the panel, so the card then closes), in a centre region under 400 points wide (`CARD_OPEN_MIN_WIDTH`; a 930 x 800 window's), or where the open card, measured before it is drawn, would take the drawing under its floor of 160 points over three list rows; that default is decided again only when the Equation panel opens or closes or the window is resized, never on a design change, and a click on the heading or the arrow is the user's choice for the session (kept by Reset all and every load). Its body scrolls above the drawing's floor.

```

with:

```markdown
after the frame's edits, from the `ConsequenceCache` (one key: the design, the Minimum OD and the
shaft-clamp pause; the Auto marking is applied outside the cache), and asks it only while the Hot-low tile or a Size column row is selected. Until BL-081 it started open, but closed while the Equation panel was open (a click on its own Why link opens the panel, so the card then closes), in a centre region under 400 points wide (`CARD_OPEN_MIN_WIDTH`; a 930 x 800 window's), or where the open card, measured before it is drawn, would take the drawing under its floor of 160 points over three list rows; that default is decided again only when the Equation panel opens or closes or the window is resized, never on a design change, and a click on the heading or the arrow is the user's choice for the session (kept by Reset all and every load). Its body scrolls above the drawing's floor.

```

- [ ] **Step 2: Check.** `git diff --stat` lists the nine files and no other; `python -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ['docs/ai/backlog.yaml', 'docs/ai/02-system.yaml', 'docs/ai/03-structure.yaml', 'docs/ai/04-memory.yaml']]"` prints nothing; `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3` still reads `1094 passed`.

- [ ] **Step 3: Commit** the nine files: `docs(magcoupler): slice B stage 2 part 2b-i docs sweep: README, docs/ai, the register, owner question 51, the design's header, BL-090 in progress (BL-090 T3)`.

### Task 4: Release: the full gate, gui-smoke, the reviews, one fix wave, the merge and the push

**Model:** the `fable` controller; the text reviewer and the whole-branch reviewers are fresh-context `fable` agents; git steps run on `haiku` at low effort; the gui-smoke workflow runs its own agents on `sonnet`.

**Files:**
- Modify: `docs/ai/backlog.yaml` (Step 7)

- [ ] **Step 1: Preconditions.**

  ```bash
  git status --short
  git diff --stat main...HEAD -- magcoupling-rs/src/engine magcoupling-rs/tests reference linkage-sim-rs/src linkage-sim-rs/web .claude/workflows/gui-smoke.js
  ```

  Expected: a clean tree (the owner's untracked press analysis aside); the second command prints nothing.

- [ ] **Step 2: The full gate.** From PowerShell at the worktree's root:

  ```powershell
  $env:MAGCOUPLING_PYTHON = "C:\Users\Cole\source\repos\linkage_simulation\reference\magcoupling-py\.venv\Scripts\python.exe"
  & 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/gate.sh --full
  git checkout -- docs/chebyshev_lambda
  ```

  Expected: the last line is `GATE PASS`; gate 12 printed `oracle Python:` and not a SKIP line; gate 13 printed `all 77 checks hold`; gate 14 built both web bundles. If a gate fails, fix the cause on the branch, red test first, and run this step again. Never edit `gate.sh` to pass.

- [ ] **Step 3: gui-smoke, from the branch's own script.**
  1. Check that port 8080 is free (`netstat -ano | findstr :8080` prints nothing): another branch's release uses it on this machine. Then, in the background: `& 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/serve_web.sh 8080`.
  2. Run the Workflow tool with `scriptPath` set to this worktree's `.claude/workflows/gui-smoke.js` and the arguments `{"frame_check": true}`.
  3. Expected: `passed: true`; in `magcoupling`: `starting_design`, `look`, `first_view`, `details_toggle`, `design_file_loaded`, `check_entry`, `lesson_link_opened`, `tooltip_plain`, `torque_card`, `address_follows`, `drag_frames_within_60fps` all `true`, `console_errors: []`. Keep the result's JSON in `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2b/smoke-local.json` and the run's first screenshot beside it.
  4. Stop the server: `TaskStop` on the background task that runs it, or `Stop-Process -Id <pid>` for the PID `netstat -ano | findstr :8080` shows. Check again that the port is free.

  The drag of the gap slider now selects the gap, so the Explain panel shows its entry during the drag: a `drag_frames_within_60fps` failure is a finding about the build until shown otherwise (the entry asks the `ConsequenceCache` once per design, as the Hot-low tile's card did, and the Size column no longer asks it at all).

- [ ] **Step 4: The text review** (fresh context, `fable`; the design's 8.3). It reads the one new string (`SPREADSHEET_CELL`) and `MINIMUM_HOVER` against the plain-words guard and the design's 5.3, checks that every line an entry draws is a text the column drew at `main` (the nothing-lost table, row by row) and that no engine text changed (`git diff main...HEAD -- magcoupling-rs/src/engine` is empty). It also rules on repetition inside an entry and records what it decides: stacked, the hover, the hint, the what-moves-what line and `LINKED_HOUSING` say one fact two or three times (Outside diameter's "the cap and its thread follow the cup", Overall length's "the housing follows"), and each size control's entry carries all of `LINKED_HOUSING`, two thirds of it about other controls. It keeps the stack or trims it (for example only the control's own clause of `LINKED_HOUSING`, or no hint where the hover says it), with its reason; a trim the owner should weigh goes to `docs/analyses/2026-11-18-owner-questions.md` as an open question with its conservative answer, not shipped silently. Its findings join Step 5's fix wave. Record it in `docs/analyses/2026-11-23-magcoupler-slice-b2p2b-i-review.md` (committed in Step 7).

- [ ] **Step 5: The independent whole-branch review** (fresh context, `fable`, three lenses in parallel, each with a skeptic on every finding that is not a suggestion). Give each `git diff main...HEAD`, the design, this plan and Step 3's screenshot, and none of the builders' notes.
  1. **Conformance:** the design's 4.3, 5.1 to 5.3 for inputs, tests 8 to 11 of 8.2 where they fall here, item by item against the diff; the decisions D-34 to D-48, each what the code does and none hiding a choice the owner should see; the nothing-lost table, row by row against the code.
  2. **Regressions:** no engine number moves; nothing reachable today is unreachable (every line of the column, every Why link's note, the Assumptions view, Detailed inputs, the torque card); no existing assertion about behaviour that stays was weakened (the list "Existing tests this plan edits"); an idle frame computes nothing new; the anti-goals of 2.3.
  3. **What a person sees:** a fresh render at 1024 x 700, 1280 x 800 and 1920 x 1080 of the first view, each Size row selected (the starting design and the example design), the Assumptions fold open and Auto on: nothing cut or overlapping; the entries read in the design's order; crimson only on the brand mark and the active tab.

  One fix wave on `fable` for what Steps 4 and 5 find, each fix with its red test first; the reviewers re-read the fixes. Record the review in `.superpowers/sdd/2026-11-23-magcoupler-overhaul-b2p2b/final-review.json` (gitignored).

- [ ] **Step 6: If Step 4 or 5 changed a file, run Steps 2 and 3 again.** The merge needs a gate and a smoke run of the tree that is merged.

- [ ] **Step 7: Close the item and merge.** In `docs/ai/backlog.yaml` (the day's date; say what the reviews found):

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-part2b-i (Tasks 1 and 2 of the plan); the full gate, the smoke run and the whole-branch review are its Task 4. Filed
```

with:

```yaml
  status: fixed
  notes: "FIXED 2026-11-2X: built on branch magcoupling/overhaul-b2-part2b-i; GATE PASS with --full, the gui-smoke run passed with frame_check true, the text review and the independent whole-branch review approved it. Filed
```

  (Write the day's date in place of `2026-11-2X`.) Commit it with the review record (`docs(magcoupler): BL-090 fixed: the gate, the smoke run and the reviews passed`), then merge the branch into `main` with a merge commit whose message lists the tasks' commit subjects. Remove the worktree with `git worktree remove`, never `git worktree prune`.

- [ ] **Step 8: Push, then report** (the owner's ruling of 2026-11-22: each stage goes live once the gate, the smoke run and the reviews pass, with a report afterwards; no stop). Push `main` (it deploys the site). When the `deploy-web.yml` run is green, run the smoke workflow from `main`'s script with `{"url": "https://colesorkness.com"}` and report the result; if it fails, say so plainly: the site is live with the change. The report to the owner holds: what changed on screen, in three sentences; two pictures (the first view, a Size row's entry); the decisions D-34 to D-48, one line each; owner question 51; parts 2b-ii and 2b-iii; what was not verified (Safari and Firefox; in a browser, that a touch selects a Size row and shows its entry: the smoke's drag selects the gap, but its prompt reads no input entry, D-45).

## What this plan covers of the design

| Design item | Where |
|---|---|
| 4.3 The size controls, one row each: label, value with unit, slider; Required torque first | Unchanged (part 1); Task 2 makes each label and slider a touch |
| 4.3 Two folds, closed at first paint: Detailed inputs and Assumptions | Task 2 (D-48 for which opens on a link) |
| 4.3 Every hint, "Set by", floor line and Why link under a control, and the `LINKED_HOUSING` footnote, shown in the Explain panel when the control is selected | Tasks 1, 2 (D-37 keeps "Set by Auto" under its slider; D-38) |
| 4.3 The left column no longer changes with the centre tab | Not in this part: the inputs side still swaps on the other tabs (part 2b-ii, with the Detailed inputs' rows as selections) |
| 5.1 An input as a selection; Esc returns the orientation | Tasks 1, 2 (Esc as for every selection) |
| 5.2 Touching an input selects it; tracing stays as an entry link | Task 2 for the Size column's six rows (D-36); the Detailed inputs' rows and the trace link in part 2b-ii |
| 5.2 A fix link selects and focuses its input, opening the fold that holds it | Task 2 opens the fold (D-48); the selection on a fix link in part 2b-ii |
| 5.3 An input's entry: title and value, What it is, What sets it, What a change does, Checks it touches, Watch out, foot | Task 1 (D-39 to D-43); the foot has no Equation link (D-41) |
| 5.3 "Spreadsheet cell" foot link | Task 1 for an input (D-41); a result's and a check's in part 2b-ii |
| 8.2 Test 8 (word budget) | Task 2 as a ceiling of 280 (D-44); 250 in part 2b-ii |
| 8.2 Test 9 (nothing lost): the Assumptions view reachable from the opening tab | Task 2 and the nothing-lost table; the full table-driven test in part 2b-ii |
| 8.2 Test 10 (selection): a click on a Size control; a hover changes nothing | Task 2 |
| 8.2 Test 11 (entries): each size control's entry holds the column's lines and its torque-card row | Tasks 1, 2 |
| 8.3 Text rules | Task 1 (plain words, glyphs), Task 4 Step 4 |
| 9.1 Docs with the slice | Task 3 |
| 9.4 Release | Task 4 (no stop: the owner's ruling of 2026-11-22) |
