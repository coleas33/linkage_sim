# Magcoupler UX overhaul, slice B, stage 2, part 1 (the instrument's columns): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ] `) syntax for tracking.

**Goal:** The first view becomes the instrument the owner chose (option B): four tiles over the drawing in the centre, the check list over the Explain panel on the right, the Required torque at the head of the Size column. The line that says two checks read fixed 3D fields is on the first view. Nothing that is on screen today is lost: every text and control the dashboard, the welcome strip and the centre's readouts and torque card held has a home (the nothing-lost table).

**Architecture:**
- **Three new modules, not more lines in `panel.rs`** (the design's section 11). `gui/checks.rs` draws the check list; `gui/tiles.rs` the four tiles; `gui/explain.rs` the Explain panel and its one `Selection`. Each returns what was clicked; the panel owns the selection.
- **The texts are today's.** A check's Explain entry draws its guide's Problem, Why sentence, fixes and notes, and the dashboard rows its verdict badges (`dashboard::line_ui`, the dashboard's row drawing, factored out). A tile's entry draws its result's help, for hot-low the torque card (`TorqueCard::lines`, unchanged) and the check of its row. The new strings are the headings, the orientation, `NEEDS` and the right column's two headings.
- **One mark per state.** `checks::state_mark` (a filled disc fails, a triangle asks for a look, a ring passes, decision C-4) draws every state's mark on the page: the check list's rows, an entry's numbers, the material warnings and the Results tab's check rows. The dashboard's `badge`, a filled disc for every level, goes in Task 1.
- **The panel** draws `checks::checks_ui` and `explain::explain_ui` in the right `SidePanel`, `tiles::tiles_ui` over `geometry_view::geometry_ui` (the drawing and, until part 2, its callout list: decision D-19) on the opening tab, and the Required torque row at the head of the Size column, which now scrolls in a short window. `dashboard_ui`, the welcome strip and the centre's torque readouts and collapsing card go.
- **Unchanged:** the engine, the exports, the top bar, the Size column's controls and their lines, the Geometry, plot and Results tabs, the Equation panel, the linkage app.

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32, egui_plot 0.33; the headless egui harness (`magcoupling-rs/src/gui/test_support.rs`, the panel tests' `Harness`, the app tests' `frame_on`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. This part is its sections 4.4 (the opening tab and the tiles), 4.5 (the check list), 5 (the Explain panel, for checks and tiles), the Required torque row of 4.3, tests 8 (as a ceiling, decision D-9) and 10 to 15 of section 8.2 where they fall in this part, register row A-18 and the docs and release rules of section 9. Executors read the spec's sections 2.3, 4, 5, 8.2, 8.3 and 9 beside this plan. Stage 1's plan (`docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`) and its section "Stage 2, to plan next" ("What follows", items 1 to 11) set the scope; its whole-branch review's suggestions are ruled on below (decision D-10). Inventory: `docs/analyses/2026-11-09-magcoupler-ux-inventory.md`. The owner's chosen mockup: `.superpowers/brainstorm/522-1794188235/content/first-arrival.html`, option B, and `teaching-reveal.html` (gitignored, in the main checkout). Backlog item: **BL-081** (`docs/ai/backlog.yaml`, filed with this plan).

**How this plan was made.** The part was built once as a prototype on a copy of the stage 1 branch's tip `272d086`, reviewed by three independent reviewers (coverage, a replay, the pictures), then fixed and moved onto the base it will be built on: `main` at `a98835a` (stage 1 merged) with BL-080's branch `magcoupling/starting-design-own-fields` at `d8eac16` merged into it, its conflicts resolved as "Where this part applies" says (the trial base). Every block below was then applied, in this plan's order, with the apply tool, to a fresh copy of the trial base with BL-081 filed. The result equals the prototype file for file (line endings aside) and passes `cargo test` with and without the app feature (1069 lib tests with it against 1060 on the base: the new modules', the panel's and the app's new tests in, the dashboard list's, the welcome strip's and the torque card's placement tests out), both native clippy runs, both wasm32 clippy runs, `cargo fmt --check` and the gui-smoke verdict test (77 checks). Each red step's "Expected" below is what that replay printed, and each count after a task is the replay's. The pictures of the prototype were drawn by stage 1's headless renderer (the real `MagcouplingApp`, tessellated and rasterized). Not run for this plan: gates 1 to 3 (linkage-sim-rs, untouched), gate 12 (the Python oracle), the release WASM build and the browser. Task 7 runs them.

**Where this part applies.** On `main` after stage 1 (BL-079, merged at `a94acc5`, `main` at `a98835a`) and BL-080 (`magcoupling/starting-design-own-fields` at `d8eac16`: `f63fea4`, `6642608`, `5d75bcb`, `d8eac16`) have both merged, and nothing else. The blocks were checked against exactly that pair, merged as a trial; no other commit was seen. The merge of BL-080 into `a98835a` conflicts in eight files, and the blocks assume this resolution: in `magcoupling-rs/src/app.rs`'s tests, stage 1's `frame_output` and `frame_with` stay and BL-080's `const LAPTOP` is added (its own copy of `frame_on` is dropped: stage 1 defines `frame_on` further down); in `magcoupling-rs/src/gui/test_support.rs`, stage 1's `text_color` and `text_colors` stay and BL-080's `shape_color` is added; in the docs both sides' entries are kept (in `docs/ai/02-system.yaml` stage 1's look line and BL-080's session-baseline line; in `docs/ai/backlog.yaml` BL-079 fixed, then BL-080 and BL-082; in the register BL-080's rows A-17 and A-18 and stage 1's A-19, and both log lines). No block quotes a line inside those conflicts: the code blocks of Tasks 1 to 5 touch none of them (the new app tests call `frame_on`, which both sides define), and each block of Task 6 is anchored on a line both sides leave alone (a heading, a neighbouring entry, a table row from `main`); the replay also applied every block on a second resolution of the same merge (BL-080's side of each docs conflict, its entry above stage 1's in the tracker), with the same result. Task 6's owner-question block anchors on BL-080's question 48, and its count line is BL-080's: the question this plan adds is 49. No block quotes the stored-3D line's words or the starting design's values, and every test reaches the stored-3D line by its constant (`dashboard::STORED_3D_LABEL`). No test of this plan pins the amber keyway line on the starting design. If a block does not match on that `main`, stop and report: the controller repairs the block, it is not adapted in place.

## Why a part, and what follows

Stage 2 is the design's sections 4 and 5 entire. Built at once it is one change to every column, the top bar and the selection model, which no reviewer can hold in their head (`panel.rs` alone is 10,000 lines). This plan cuts it in two (decision D-1). Part 1, this plan, changes the first view visibly on its own: the right column and the centre. It loses nothing: the top bar, the Size column's controls and lines, and the other tabs stay as they are, and every text and control that leaves the screen has a home in this part (the nothing-lost table).

**What follows, as part 2, with its own plan and backlog item:**

1. The top bar (4.2): the Design menu (Reset to the starting design, Save design, Load design, Open the example design, Show spreadsheet details), Share, Export and Learn with its note list; the Equation panel button leaves the bar and the panel's Explain checkbox goes (C-11), so the Equation link of an Explain entry becomes its only door besides a readout's click, and each step of the Start here walk selects its note in the Explain panel (5.5, test 10). Suggestions S-1 (a SemiBold face for the title and the headings) and S-3 (buttons without a box at rest) of stage 1's review are ruled there.
2. The Size column (4.3): the folds Detailed inputs and Assumptions; every hint, "Set by", floor and Why line under a control and the `LINKED_HOUSING` footnote moving into the Explain panel when the control is selected; inputs as a `Selection` (touching an input selects it; "Highlight the results this drives" keeps the trace); the torque card's row of each size control in its entry (the second half of C-10); an entry's "Spreadsheet cell" foot link with the details on (5.3).
3. More selections (5.1, 5.2): dimension tags, results-table rows, callout rows, a note, the note list. With the tags as selections the callout list leaves the opening tab for the Geometry tab (4.4; this part keeps it under the drawing, decision D-19), and the three rows of D-17 get their one-click home.
4. The Plots tab with its picker (C-9) and the plots' red reference lines (stage 1 review C-3), the slip-heating readout's stray unit (9.2), egui_plot's fading tick numbers (E-7).
5. The word budget of 250 at 1280 x 800 (goal G-1, test 8), which needs items 1 and 2; this part holds the first view to 450 (decision D-9).
6. The controls' outlines (stage 1 review E-4, R-6) and the page shell (S-4).
7. The cap's skirt-wall caution (owner answer 4a of 2026-11-22: amber below a minimum, 0.5 mm offered) is BL-082 on its own branch, not part 2; the check list leaves it room (decision D-13).

## What the prototype showed

1. **Words.** At 1280 x 800 the first view draws 608 words before this part and 443 after it (the renderer's count of every text shape drawn in view, the inventory's measure); without the callout list under the drawing it would be 382 (decision D-19 keeps it). The dashboard's 19 rows with their Why lines, the welcome strip and the torque card are most of the cut. The design's 250 needs part 2.
2. **The fixed-3D line.** At 1280 x 800, 1024 x 700 and 1920 x 1080 the check list closes with it inside the first view, under the nine rows; the Explain panel's orientation is under it.
3. **A window of 1024 x 700 overflowed before this part.** The Size tab's left column has no scroll area, and its content (about 795 points tall at the starting design) stretched the panel past the window's foot: the centre region and its drawing ran under the window's edge. The tiles and the Required torque row made it worse. The Size column now scrolls (decision D-7), and the app test holds the drawing inside the window.
4. **Tiles of equal height.** A tile's frame sits in a left-to-right row, so a spacer added under its content sat beside it; the row's height is set as each frame's least height instead (`tiles::tile_ui`).
5. **The no-prediction words.** A tile reads `NO_PREDICTION` or `NO_TORQUE` per torque, as the Size tab's readout did: with a design file's variation of -1e308 the pull-out is a number and both hot-low and cold-high read "no prediction (the torque is not finite for this design)"; an unmagnetized inner ring is a prediction of zero and reads "0 N·m".
6. **Contrast of the new pairings** (the token table's values, WCAG formula): a tile's number (text `#e6e8eb`) on its fill (panel2 `#1e2227`) 13.0:1; its label and the word "needs" (muted `#8a919a`) on panel2 5.0:1; the requirement after it (text) on panel2 13.0:1; the right column's headings (muted) on panel 5.4:1; the pass green `#4cc38a` on panel 7.8:1 (a row's word) and on panel2 7.2:1 (the tile's margin line); the failing red `#ff4d57` on panel 5.3:1 and on panel2 4.9:1 (the Hot-low tile's "needs" line); the warn amber `#e0a100` on panel 7.6:1. A selected check row and a selected tile use panel2 as their fill and the text colour as the tile's outline. The accent is on nothing new.
7. **What the review found in the pictures, and what changed.** The Hot-low tile's gap wrapped inside its brackets ("short 0.21 N·m (-8.6" over "%)"): it is drawn as two pieces that a line breaks only between (`tiles::gap_pieces`). On a passing design the whole "needs" line was muted: the word stays muted and the requirement after it is a value, in the text colour. An entry's numbers and the material warnings drew a filled disc for every state, the fail shape on a passing number: every mark is now `checks::state_mark`'s. At 1280 x 800 the Detailed inputs fold sat half cut at the window's foot: the footnote on what the size controls move now follows the fold. The opening tab had dropped the callout list while its tags could not yet be clicked: it stays (D-19).
8. **A fix link in a short window.** The Size column scrolls, and Detailed inputs keeps its own scroll area inside it. A fix link followed at 1280 x 800 or 1024 x 700 brings its input's row inside the window: the column scrolls there in an animation of about a third of a second (the inner area, unbounded inside the column, passes the scroll on). A test holds it (Task 5).

Pictures, in `.superpowers/sdd/2026-11-22-magcoupler-overhaul-b2/spike/` of the main checkout (gitignored), redrawn after the review's fixes: `before-first-view-1280.png` (the stage 1 tip, before BL-080), `after-first-view-1280.png`, `after-first-view-1024.png`, `after-first-view-1920.png`, `after-example-design-1280.png` (the failing example design), `after-check-selected-1280.png` (the example design with Running clearance selected), `after-hot-low-tile-selected-1280.png` (the Hot-low tile outlined, the torque card and "Checks it touches" in the Explain panel).

## Global Constraints

- **No engine number moves** (the design's 2.3). No file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` changes.
- **The linkage app is not touched** (O-1). No file under `linkage-sim-rs/src/` or `linkage-sim-rs/web/` changes.
- **The panel sets no theme and no font** (2.2, 6.3). Code under `magcoupling-rs/src/gui/` reads `ui.visuals()`; the colours it draws are the host's, `look::quiet_text`, `look::mark_color` and the levels' (`Level::color`).
- **A label is muted and its value keeps the text colour** in every new piece: a tile's label, the word "needs" on the Hot-low tile, an entry's headings and the label of each row under "Numbers behind this check" are `look::quiet_text`; a tile's number, the requirement after "needs" while it is met, a check's name and a row's value are the text colour. A test holds each (Tasks 2, 5).
- **Crimson is never text, never on a value, never in the check list** (C-5). Red means a failing check: the fail disc, a failing check's word, a failing Problem and the Hot-low tile's "needs" line when it fails, never the tile's number. State is a shape and a word as well as a colour (C-4).
- **The check list keeps one fixed order** (C-3): `CHECK_GUIDES`' order, failing checks not sorted.
- **No new explanatory text** (5.3): every text under an entry's headings ships today. The new strings are the right column's two headings, the entry headings ("What it is", "What sets it", "What a change does", "Checks it touches", "Numbers behind this check"), "About this tool", "Full lesson:", "Equation", `NEEDS`, and the orientation; each passes the plain-words guard and the glyph guard, and the orientation and the "needs" line get the physics review of Task 7 (8.3).
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
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-22-magcoupler-overhaul-b2/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Shell.** On this machine `bash` typed in PowerShell is WSL, which fails on the checkout's line endings. Shell scripts run through Git Bash: `& 'C:\Program Files\Git\bin\bash.exe' <script>`. Python scripts are written with the Write tool, not a here-document.
- **Never run `cargo fmt` inside `linkage-sim-rs`.** `magcoupling-rs` is fmt-clean; keep it so.
- **Docs ship with the code** (repository rule): Task 6 sweeps the README and `docs/ai/`, and the register and the owner-question list.

## Model per task

CLAUDE.md section 5. BL-081 is `risk: physics` (decision D-14): the Explain panel's orientation states how the coupling works in new words, and the Hot-low tile restates the requirement in a new line ("needs 1.350 N·m"). Every other number and claim on screen is today's text moved. By section 5's precedence every task of a `risk: physics` item runs on `fable`, though Tasks 1 to 6 transcribe exact blocks. Git housekeeping runs on `haiku` at low effort.

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | The check list (`gui::checks`) | `fable` |
| 2 | The tiles (`gui::tiles`, `guidance::hot_low_quotes`) | `fable` |
| 3 | The Explain panel (`gui::explain`, `dashboard::line_ui`) | `fable` |
| 4 | gui-smoke reads the new layout | `fable` |
| 5 | The panel draws the new columns; the layout tests in the app's own font | `fable` |
| 6 | Docs sweep | `fable` |
| 7 | Release: the full gate, gui-smoke, the physics review, the independent whole-branch review, one fix wave, the merge and the push | `fable` controller; fresh-context `fable` reviewers; `haiku` for git; the smoke workflow's own agents on `sonnet` |

Each task's diff is reviewed by a fresh-context `fable` reviewer before the next task starts; it checks the diff against the task's blocks and nothing else. A task that ends blocked or leaves a test red stops the build and goes back to the controller.

## Review Focus

Six inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named.

1. **A short window.** At 1024 x 700 the Size column is taller than the window. A person expects the drawing and the tiles inside the window. Task 5: `the_first_view_keeps_its_three_columns_its_tiles_and_the_fixed_3d_line_in_the_app_s_own_font` (it failed on the prototype before the column scrolled).
2. **A design the model cannot predict.** Short magnets, a calibration coefficient of 0, a variation of -1e308. A person expects words, not "inf N·m" or "NaN", and no "needs" line on a tile with no number. Task 2: `a_torque_the_model_does_not_predict_reads_so_in_place_of_its_number`; Task 1: `a_check_that_gives_no_badge_has_no_mark_and_no_word`.
3. **A hot-low a hair under the requirement.** Both round to the same four digits. A person must never read "1.448 N·m, needs 1.448 N·m" on a failing tile. Task 2: `a_failing_hot_low_tile_never_shows_two_equal_numbers` (BL-068's rule, shared with the Problem).
4. **Esc while typing.** A person editing a value box presses Esc to drop the edit; the Explain panel must keep what it shows. Task 5: `esc_in_a_text_field_is_the_field_s_and_keeps_the_selection`.
5. **An idle frame with the Hot-low tile selected.** The torque card is the costliest entry (its consequences are 30 `compute_all` runs). A person expects a still screen to cost nothing. Task 5: `the_check_list_and_the_explain_panel_compute_nothing_on_an_idle_frame` (nothing selected, two checks' entries and the Hot-low tile's), and `idle_frames_compute_the_card_once_and_an_edit_again`.
6. **A fix link in a short window.** The Size column scrolls and Detailed inputs scrolls inside it. A person who follows a fix link at 1024 x 700 expects to see the row it names. Task 5: `a_fix_link_on_the_opening_tab_shows_its_row_inside_a_short_window`.

## Decisions this plan takes

Each is a reading of the design where it was open. The owner may overrule any of them; Task 7 lists them in its report. The ones that need a row in the register have one (Task 6).

| Id | Decision | Why |
|---|---|---|
| D-1 | Stage 2 is built in two parts; this plan is part 1 (the right column, the tiles, the Required torque row, the fixed-3D line on the first view). Part 2 is listed under "What follows". Register row A-21. | One plan for sections 4 and 5 entire cannot be held by a reviewer. This part changes the first view the owner called outdated and loses nothing. The conservative answer to Q-4 is smaller releases. |
| D-2 | A check's row shows its state as a shape and a word (OK, Check, Fails) beside the check's name, and no short form of its quantity. A row whose check gives no badge (it does not apply, or it is computed from a pull-out the end-effect model cannot predict) has no shape and no word, and its name is weak while greyed. Register row A-20; owner question 49. | The design's 4.5 asks for "the check's own quantity in short form where it has one" and leaves the forms to the builder (11). Nine short forms are nine new ways of stating a number, each needing a physics review, and would compete with the four tiles. The word is the badge's existing hover word; the numbers are one click away. |
| D-3 | In this part the `Selection` is nothing, a check or a tile's result. Inputs, dimension tags, table rows, callout rows, notes and the note list are part 2. Touching an input selects nothing yet: its lines stay under it in the Size column, as today. | Inputs as selections come with the Size column's move of its lines into the Explain panel (4.3); doing one without the other would show each hint twice. |
| D-4 | A check's entry: its shape, name and word; "What it is" (the guide's Why sentence); while the check does not pass, "What sets it" (the Problem, in the level's colour), "What to change" (each fix a link that shows its input's row, as `fix_text` links do today, or `NO_FIX`) and "Watch out" (the guide's notes); then "Numbers behind this check" (the dashboard rows its verdict badges, drawn as the dashboard drew them, each a readout, with the stored-3D line under the rows that read the stored fields; for Magnets fit, whether the blocks fit); then "Full lesson:" (the row's note, assumption or "no note yet") and "Equation". | The design's 5.3 table, with today's texts. Today a red block is open on arrival and an amber one opens at one click on its header; after, both open at one click on the row (G-8 and the anti-goal accept exactly one click for a red check). |
| D-5 | The Required torque row moves to the head of the Size column (4.3): its muted label, its Why link to the torque band note and its slider, an ordinary design edit. | The centre readouts that held it leave with the tiles. |
| D-6 | The torque card is the "What a change does" of the Hot-low tile's entry (C-10). The "one row under each size control" half of C-10 waits for inputs as selections (part 2). | The whole card is one click from the first view; nothing of it is lost meanwhile. |
| D-7 | The Size column scrolls (a vertical scroll area, full width) on the opening tab. | Found by the prototype: at 1024 x 700 the column's overflow stretched the panel and the centre's drawing past the window's foot (the stage 1 tip does it too). |
| D-8 | A tile: its number at 22 points (the design's 20 to 24), Regular, the text colour; its label muted, wrapping, under it; panel2 fill (`faint_bg_color`), a hairline, corner radius 4; the selected tile outlined in the text colour. Four in a row while each is at least 130 points wide, else two rows of two; one height per row. | C-6 and the token table; test 15's "the four tiles wrap to two rows instead of clipping". No SemiBold face (D-10). |
| D-9 | The first view at 1280 x 800 is held to at most 450 words (443 now, 608 before); the design's 250 is part 2's test. | Test 8's 250 needs the top bar's, the Size column's and the callout list's words (the Size column's lines alone are about 145, the list about 60). A ceiling now keeps this part from growing back. |
| D-10 | Stage 1's review suggestions: S-1 (SemiBold) waits for part 2's top bar and headings; S-2 (the brand mark as a bar) is declined, the mark is the owner's mockup's and owner question 9 asks; S-3 (buttons without a box) and S-4 (the page shell) are part 2's; S-5 and E-3 (the drawing's height) are met: the opening tab's drawing takes the height under the tiles that its callout list leaves (at 1280 x 800 about 330 points, from 197; about 520 once part 2 moves the list, D-19); C-3 (the plots' red reference lines) is part 2's, with the Plots tab; C-6 (layout tests in the app's font) is met by Task 5's two app tests; E-4 (the controls' outlines) is part 2's. | Each where the widget it concerns is rebuilt. |
| D-11 | The check list takes its natural height up to 45 % of the right column and scrolls beyond; the Explain panel takes the rest and scrolls on its own (`checks::CHECKS_SHARE`). | The design's 4.5 ("about 45 %"). |
| D-12 | The stored-3D line closes the check list, weak, with its hover (`STORED_3D_NOTE`), and an entry that shows a stored-3D row repeats it under that row. Tests find it by its constant. | Register row A-18 and the design's 4.5; BL-080 rewords the line, and these tests pass through. |
| D-13 | The cap's skirt-wall caution (BL-082) is not built here. Its natural place is the check list's foot, among the material warnings under the rows (`dashboard::warnings_ui`): BL-082 asks for a caution that is not counted as a failed check, so it is no guide. This part reserves nothing in code for it. | Owner answer 4a; the item is on another branch. |
| D-14 | BL-081 is `risk: physics`. | The orientation (5.4) is a physical claim in new words; the "needs" line restates the requirement. |
| D-15 | A check's state on its row: Magnets fit from its guide's level (it has no engine result), every other check from its dashboard badge (`check_level`). The entry's text depends on the guide's level, as the dashboard's block did. | A row must never show a state its check does not give. |
| D-16 | gui-smoke: `dashboard_scrolls` becomes `check_entry` (a click on the Hot-low torque row shows its entry down to "Full lesson:", and the Explain panel scrolls on its own) and `why_link_opened` becomes `lesson_link_opened` (that entry's lesson link opens the note); `first_view`, `starting_design` and `torque_card` read the new layout; the other checks keep their words. | The dashboard and its Why links are gone; the names say what each check sees. |
| D-17 | Three dashboard rows belong to no check: Pull-out torque at 20 °C, Nominal slip swing at the gearbox input and Cup outer diameter. Their values are in the Results tab's Headline group (as every row's is), and their notes open in the Equation panel (`explorer::note_fallback` reads the same `ROW_LEARN`) once its Explain checkbox is on: the Results tab, the row, then the checkbox, three clicks or more, where the dashboard's Why link was one. | The design's 4.5: "every row's value is in the Results tab's Headline group". Part 2 ("What follows", items 1 and 3) removes the checkbox (C-11) and gives these rows one click through result selections; the release report names the trade. |
| D-18 | An entry's form departs from 5.3 in two ways. A check's entry names its rows "Numbers behind this check" (4.5's own words for them) where 5.3 says "Checks it touches", and draws them after "Watch out", which 5.3 puts last: the check's notes are about its fixes, so they stay beside "What to change", as the dashboard drew them. A tile's entry keeps 5.3's "Checks it touches": the check of its row, its shape, name and word, the name a link that selects it (only the Hot-low tile's row has a check). | Today's texts in today's order for a check; 5.3 for a tile, which needs no new text. |
| D-19 | The opening tab keeps the callout list and its notes under the drawing in this part; 4.4 moves them to the Geometry tab. | 4.4 moves the list because a click on a dimension tag will show its text in the Explain panel, and that click is part 2's (D-3). Without it the first view would show six numbered tags nothing explains, and the space-claim values ("42.80 mm of 43.00 mm") would leave it. The drawing is shorter for it (the list takes about 165 points at 1280 x 800); the tiles took less room than the readouts and the card did. |


## Nothing lost

Every text and control on today's screen (the inventory's sections 1 to 7, at the stage 1 tip) that this part moves, and where it is after. Anything not listed (the top bar and its second row, the Size column's controls, Auto and every line under a control, Detailed inputs, the tab row, the end-effect banner, the Geometry, plot and Results tabs, the Equation panel) does not move. The "Test" column names the test that holds the new home.

| Today | After this part | Test |
|---|---|---|
| Welcome strip "Set the size -> read the torque -> open any red check to see why and what to change." and its Hide button | Gone; the Explain panel's orientation replaces it (5.4): what the coupling is, the three moves, the Start here link. Nothing to hide: the orientation is the panel's empty state | `explain::tests::nothing_selected_shows_the_orientation_in_at_most_sixty_words`; `panel::tests::the_right_column_is_the_check_list_over_the_explain_panel_on_its_orientation` |
| Centre: Required torque slider, its Why link | Size column, first row, same slider, label and Why link (D-5) | `panel::tests::a_new_panel_opens_on_size_and_torque_with_the_tiles_over_the_drawing`, `the_tiles_read_hot_low_what_it_needs_its_gap_and_the_nominal_pull_out` |
| Centre: Hot-low torque readout, its gap ("margin +0.10 N·m (+7.3 %)"), its Why link | The Hot-low tile: its number, "needs 1.350 N·m" (the word muted, the requirement a value), the same gap text in the same colours, breaking only between its torque and its percent; the Why link's note is the tile's entry's "Full lesson:" link | `tiles::tests::the_hot_low_tile_says_what_it_needs_and_its_gap_and_fails_by_shape_and_red_on_that_line`; `explain::tests` (the tile's foot) |
| Centre: Pull-out torque (nominal, at 50 °C) readout | The first tile, same label and number | `tiles::tests::the_four_tiles_show_the_results_own_numbers_over_their_labels` |
| Centre: the "no prediction" words of a readout | The tile's number's place | `tiles::tests::a_torque_the_model_does_not_predict_reads_so_in_place_of_its_number` |
| Centre: the torque card (heading, Why link, caption, Auto line, four rows, or its one line) | The Hot-low tile's Explain entry, "What a change does" (D-6) | `panel::tests::the_hot_low_tile_s_entry_draws_the_torque_card` |
| Centre: "Wall past keyway" amber line, "Magnets do not fit" red line, "Exploration only" lines | Same place: under the tiles, over the drawing, unchanged words and conditions | `panel::tests` (the exploration lines' tests keep their assertions) |
| Centre: the callout list and notes under the drawing on the opening tab | Same place, under the drawing (D-19), and on the Geometry tab as before; part 2 moves them with the tags' selection | `panel::tests::a_new_panel_opens_on_size_and_torque_with_the_tiles_over_the_drawing`; `geometry_view::tests` |
| Size column: the footnote "Outside diameter moves the ring radius; ..." under the size controls | Under the Detailed inputs fold, same text, so the fold is on the first view | `app::tests::the_first_view_keeps_its_three_columns_its_tiles_and_the_fixed_3d_line_in_the_app_s_own_font` (the fold whole at 1280 x 800 and 1920 x 1080) |
| Dashboard: status line "Starting design: passes all 9 checks" and its hover | The check list's first line under "CHECKS", same text, colour and hover | `checks::tests::the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour`, `the_starting_design_s_status_line_says_on_hover_what_it_is` |
| Dashboard: 19 rows (badge, label, value, correction ids with the details on, hover with equation, click opens the Equation panel) | The four tile rows are tiles. The rows a check badges are under "Numbers behind this check" in that check's entry, drawn by the same `line_ui` (its state's shape, muted label, value, correction ids, hover, click). Every row's value is in the Results tab's Headline group. Three rows belong to no check (D-17) | `explain::tests::the_numbers_behind_a_check_are_the_dashboard_rows_its_verdict_badges`, `the_numbers_of_a_passing_check_carry_the_pass_shape_and_never_the_fail_shape`; `panel::tests::hovering_a_headline_value_shows_its_equation`; `the_right_column_shows_the_stored_3d_line_and_the_correction_ids` |
| Dashboard: a guided check's badge | Its check-list row: shape, name, word (D-2, D-15) | `checks::tests::each_state_has_its_own_shape_and_word_as_well_as_its_colour` |
| Dashboard: red block's "Problem:", "Why:" sentence, "Fix:" links, notes; amber block's "Problem and fixes" header | The check's entry: "What sets it", "What it is", "What to change" (the same links, asking for the same input rows), "Watch out" (the same notes). One click on the row for red and amber alike (D-4) | `guidance::tests` (`check_entry_texts` replaces `dashboard_ui` in the guides' tests, same strings); `explain::tests`; `panel::tests::an_amber_check_s_entry_is_open_at_its_one_click` |
| Dashboard: every row's "Why:" link (note, assumption or "no note yet") | The entry's "Full lesson:" link, same target (a check's guide's, a tile's row's); the three rows of D-17 through the Results tab and the Equation panel's Explain checkbox (D-17) | `panel::tests::the_gearbox_check_s_lesson_link_opens_the_torque_band_note`, `the_shaft_joint_check_s_lesson_link_opens_the_shaft_joint_note` |
| Dashboard: Magnets fit row ("Fit around the ring" or not) | Its check-list row and its entry's "Numbers behind this check" | `checks::tests`, `explain::tests` |
| Dashboard: "Material warnings" heading and its rows with their note links | The check list, under the rows, same drawing (`warnings_ui`), each mark its severity's shape | `checks::tests::the_warnings_that_fire_follow_the_rows_and_the_stored_3d_line_closes_the_list` |
| Dashboard: "Fixed 3D fields from the example design" under the shaft joints row, with its hover; below the first screen at 1280 x 800 | The check list's last line, with its hover, on the first view (A-18); repeated under a stored-3D row in an entry | `app::tests::the_first_view_keeps_its_three_columns_its_tiles_and_the_fixed_3d_line_in_the_app_s_own_font`; `checks::tests` |
| Dashboard: the correction legend (with the details on) | Over the numbers in an entry, with the details on | `panel::tests::the_right_column_shows_the_stored_3d_line_and_the_correction_ids` |
| Dashboard: the end-effect banner over the rows | Over the centre, as it was there too; the greyed rows are greyed in the entries, and their checks show no shape | `panel::tests::the_end_effect_banner_links_to_its_note_over_the_centre`; `checks::tests::a_check_that_gives_no_badge_has_no_mark_and_no_word` |
| Dashboard: the clamp's row and guide while the section shows | Its check-list row while it shows; none while paused | `checks::tests::the_list_is_the_status_line_a_row_per_counted_check_in_the_guides_order_and_the_3d_line` |
| Dashboard: the ripple row's Why link to the slip note | The Nominal slip swing row in the Results tab's Headline group, its note in the Equation panel with the Explain checkbox on (D-17) | `explorer::tests` (note fallback, unchanged) |
| Results tab: each check row's badge, a filled disc for every level | The state's shape (C-4) | `panel::tests::the_failing_filter_shows_exactly_the_failing_checks_with_their_badges`, `out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too` |
| (new) The Hot-low tile's entry: the check of its row | "Checks it touches": the Hot-low torque check's shape, name and word; the name selects it (D-18) | `explain::tests::a_tile_s_entry_names_the_check_of_its_row_and_a_click_selects_it` |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/checks.rs` | New: `HEADING`, `CHECKS_SHARE`, `mark_level`, `state_mark`, `checks_ui`; tests | 1 |
| `magcoupling-rs/src/gui/results_table.rs` | A check row's mark is `state_mark`'s | 1 |
| `magcoupling-rs/src/gui/readouts.rs` | `Readouts::hover` (a readout's mark and tooltip without its click) | 1 |
| `magcoupling-rs/src/gui/dashboard.rs` | `status_ui`, `warnings_ui` and every mark `state_mark`'s, `badge` gone (1); `learn_ui` and `line_ui` (3) for the new modules; `dashboard_ui` and its guide drawing go, its tests move (5) | 1, 3, 5 |
| `magcoupling-rs/src/gui/tiles.rs` | New: `Tile`, `NEEDS`, `NUMBER_SIZE`, `MIN_TILE_WIDTH`, `needs_lines`, `gap_pieces`, `tiles_ui`; tests | 2 |
| `magcoupling-rs/src/gui/guidance.rs` | `hot_low_quotes` (2); the text review's list names the new strings (3); the guides' tests read the entry, not the dashboard (5) | 2, 3, 5 |
| `magcoupling-rs/src/gui/explain.rs` | New: `Selection`, `Asked`, `Shown`, `explain_ui`, the orientation and the headings (`CHECKS_IT_TOUCHES` among them); tests | 3 |
| `magcoupling-rs/src/gui/mod.rs` | The three modules, each with an `allow(dead_code)` until Task 5 calls it | 1, 2, 3, 5 |
| `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` | The prompt and the verdict test read the new layout (D-16) | 4 |
| `magcoupling-rs/src/app.rs` | The smoke prompt's guard (4); the label colours, the opening, the layout and word tests (5) | 4, 5 |
| `magcoupling-rs/src/gui/panel.rs` | Two Results-tab tests read the marks by their shapes (1); the right column, the opening tab, the Required torque row, the scrolling Size column with the footnote under the fold, Esc; the welcome strip, the readouts and the card go; tests (5) | 1, 5 |
| `magcoupling-rs/src/gui/geometry_view.rs` | `drawing_floor` goes with the card | 5 |
| `magcoupling-rs/src/gui/size_controls.rs` | The footnote leaves `controls_ui` (the panel draws it under the fold); one doc comment | 5 |
| `magcoupling-rs/src/gui/test_support.rs` | `check_entry_texts`, `shown_whole`, `words_shown` | 5 |
| `magcoupling-rs/README.md`, `docs/ai/*`, the register, the owner questions, the design's header | The docs sweep | 6 |
| `docs/ai/backlog.yaml` | BL-081 in progress (6), fixed (7) | 6, 7 |

**Existing tests this plan edits, and how.** No assertion about behaviour that stays is weakened; the tests of behaviour that leaves the screen go with it and the table above names what holds each new home.
- `dashboard::tests`: the tests of the dashboard's list, its guide blocks, its Why lines, its status line and its Magnets fit row go with `dashboard_ui` (Task 5); the status line's three move to `checks::tests` (Task 1) with the same assertions, the guide blocks' to `explain::tests` (Task 3), the row-label test is renamed `each_row_is_exported_under_its_catalogue_label_and_four_have_a_display_label` and keeps its assertions on the export.
- `panel::tests`: four tests of the welcome strip and twelve of the torque card's place over the drawing go (the strip and that place are gone); `a_new_panel_opens_on_size_and_torque_under_the_welcome_strip`, `size_tab_reads_the_requirement_the_hot_low_gap_and_the_nominal_pull_out`, `hovering_a_dashboard_value_shows_its_equation`, `a_narrow_window_wraps_the_check_guides_inside_the_dashboard`, `a_raised_minimum_od_takes_lower_outside_diameter_off_the_dashboard`, `the_dashboard_shows_the_stored_3d_label_and_the_corrected_markers`, `the_end_effect_banner_links_to_its_note_over_the_dashboard_and_the_centre` and the three row Why-link tests are rewritten for the new homes under the names in the table above, with the same targets and texts; the size controls' tests find the Required row through `required_box`.
- `panel::tests::the_failing_filter_shows_exactly_the_failing_checks_with_their_badges` and `out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too` count the Results tab's marks by their shapes (`mark_at`) in place of filled discs, with the same counts (Task 1).
- `guidance::tests::assert_open_block_draws_no_fix` reads the check's entry (`check_entry_texts`) in place of the dashboard; its assertion (`NO_FIX` right after the fix heading) stays.
- `geometry_view::tests::the_drawing_floor_is_the_room_that_keeps_the_drawing_over_three_list_rows` goes with `drawing_floor` (the card it served is gone).
- `app::tests::a_label_is_muted_and_its_value_is_in_the_text_colour` holds the new labels (the right column's headings, a check's entry's headings and a number row's label) in place of the dashboard's; `the_app_opens_the_starting_design` reads the tile's "needs" word and requirement, the gap as its two pieces and the orientation in place of the card and the one-piece gap; `the_smoke_test_share_link_opens_its_design` loses the welcome strip and gains the right column's words.

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` after stage 1 (BL-079), BL-080 and this plan have merged, create the branch `magcoupling/overhaul-b2-the-instrument` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  ```

  Expected: `test result: ok. 1060 passed; 0 failed; 6 ignored` (`a98835a`'s 1056 and BL-080's four). Another count means `main` is not the base the blocks were checked on: stop and report.

- [ ] **Step 3: Check the base.** `git log --oneline a98835a..main` lists BL-080's four commits (`f63fea4`, `6642608`, `5d75bcb`, `d8eac16`), the merge that brought them in, and this plan's commits, and nothing else that touches `magcoupling-rs/`, `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` or Task 6's nine docs (`git log --oneline a98835a..main -- <those paths>`). Then check the merge's resolution: `grep -n "fn frame_output\|const LAPTOP\|fn frame_on" magcoupling-rs/src/app.rs` shows one `frame_output`, one `LAPTOP` and one `frame_on`, and `grep -n "fn text_colors\|fn shape_color" magcoupling-rs/src/gui/test_support.rs` shows both. If any of this differs, stop: the controller regenerates the blocks on that `main` (the planner's tools in `.superpowers/sdd/2026-11-22-magcoupler-overhaul-b2/`) before Task 1.

## Tasks

### Task 1: The check list

**Model:** `fable` (BL-081 is `risk: physics`; the task transcribes exact blocks).

The check list as a module the panel does not call yet: the heading, the status line, one row per counted check in `CHECK_GUIDES` order with its state's shape and word, the material warnings and the stored-3D line; a click returns the check. Its module carries `allow(dead_code)` until Task 5 draws it.

**Files:**
- Create: `magcoupling-rs/src/gui/checks.rs`
- Modify: `magcoupling-rs/src/gui/mod.rs`, `magcoupling-rs/src/gui/dashboard.rs` (two functions visible to the crate; `badge` goes, every mark is `state_mark`'s), `magcoupling-rs/src/gui/readouts.rs`, `magcoupling-rs/src/gui/results_table.rs` (its marks), `magcoupling-rs/src/gui/panel.rs` (two Results-tab tests count marks by shape)

**Interfaces:**
- Produces: `checks::HEADING`, `checks::CHECKS_SHARE`, `checks::mark_level(guide, inputs, results) -> Option<Level>`, `checks::state_mark(ui, Option<Level>)`, `checks::checks_ui(ui, inputs, baseline, results, clamps_paused, selected, readouts) -> Option<&'static str>`; `Readouts::hover(ui, response, path, text)`.

- [ ] **Step 1 (red): the check list's tests.**

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust

pub mod clamp_drawing;
```

with:

```rust

// Drawn by the panel from Task 5 of this stage's plan, which removes this allow.
#[allow(dead_code)]
pub mod checks;
pub mod clamp_drawing;
```

**`magcoupling-rs/src/gui/checks.rs`**

Create the file with:

```rust
//! The check list (UX overhaul spec 4.5): the top of the right column. The design status line,
//! then one row per check that counts, in [`CHECK_GUIDES`]' order, which never changes (decision
//! C-3): the check's state as a shape and a word as well as a colour (decision C-4), and its
//! name. A click on a row selects the check, and the Explain panel under the list
//! ([`crate::gui::explain`]) shows its problem, why it matters and what to change. Under the
//! rows, the material warnings that fire, and the line that says two of the checks read stored
//! 3D fields ([`STORED_3D_LABEL`]): at the foot of the list, it is on the first view.
//!
//! [`CHECK_GUIDES`]: crate::gui::guidance::CHECK_GUIDES

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::dashboard::{WARNINGS_HEADING, severity_level, warning_lines};
    use crate::gui::guidance::{
        CHECK_GUIDES, EXAMPLE_DESIGN, EXAMPLE_DESIGN_HOVER, STARTING_DESIGN, STARTING_DESIGN_HOVER,
        YOUR_DESIGN, design_status, design_status_text,
    };
    use crate::gui::test_support::{
        drawn_texts, flat_shapes, hover_frames, only_the_clamp_fails, only_the_gearbox_fails,
        only_the_joint_fails, passing_design, primary_button, reversed_hub_band, short_magnets,
        sized_frame, text_center, text_color, text_rect, with_poles,
    };

    /// The defaults with `edit` applied.
    fn design(edit: impl FnOnce(&mut DesignInputs)) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        edit(&mut inputs);
        inputs
    }

    /// The defaults with a hot day past the temperature limits: the verdict asks for a look.
    fn hot_day() -> DesignInputs {
        design(crate::gui::test_support::hot_day)
    }

    /// Frames of the check list of `inputs` against the spreadsheet's baseline in `ctx`, with
    /// the shaft-clamp section paused (`clamps_paused`) or not and `selected` selected: a
    /// first one with no input, then one per entry of `events`. What the last drew and the
    /// check it returned.
    fn frames(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        clamps_paused: bool,
        selected: Option<&'static str>,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, Option<&'static str>) {
        let results = compute_all(inputs);
        let mut last = None;
        for input in std::iter::once(Vec::new()).chain(events) {
            let mut clicked = None;
            let output = sized_frame(ctx, egui::vec2(320.0, 1200.0), input, |ui| {
                clicked = checks_ui(
                    ui,
                    inputs,
                    &DesignInputs::default(),
                    &results,
                    clamps_paused,
                    selected,
                    &mut Readouts::default(),
                );
            });
            last = Some((output, clicked));
        }
        last.expect("one frame at least")
    }

    /// What the check list of `inputs` draws.
    fn list_texts(inputs: &DesignInputs, clamps_paused: bool) -> Vec<String> {
        let ctx = egui::Context::default();
        drawn_texts(&frames(&ctx, inputs, clamps_paused, None, Vec::new()).0)
    }

    #[test]
    fn the_list_is_the_status_line_a_row_per_counted_check_in_the_guides_order_and_the_3d_line() {
        // Spec 4.5 and decision C-3: one fixed order, failing checks not sorted to the top. At
        // the example design four checks fail and five pass.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        for paused in [false, true] {
            let mut want = vec![
                HEADING.to_owned(),
                design_status_text(&design_status(&inputs, &results, paused)),
            ];
            for guide in counted_guides(paused) {
                want.push(guide.label.to_owned());
                want.push(guidance::level(guide, &inputs, &results).word().to_owned());
            }
            want.push(STORED_3D_LABEL.to_owned());
            assert_eq!(list_texts(&inputs, paused), want, "paused {paused}");
        }
        let labels: Vec<&str> = counted_guides(true).map(|guide| guide.label).collect();
        assert_eq!(
            labels,
            [
                "Hot-low torque",
                "Running clearance",
                "Cup wall",
                "Temperature limits",
                "Shaft joints",
                "Gearbox input torque",
                "Space claim",
                "End-effect model",
                "Magnets fit"
            ]
        );
        // Ruling 11: a design that fails only the clamp reads as passing while it is paused,
        // and its row is drawn only while the section shows.
        let inputs = only_the_clamp_fails();
        let clamp = guidance::guide("clamps.recommended").expect("the clamp's guide");
        for paused in [false, true] {
            let texts = list_texts(&inputs, paused);
            let rows = texts.iter().filter(|t| *t == clamp.label).count();
            assert_eq!(rows, usize::from(!paused), "paused {paused}");
            let fails = texts.iter().filter(|t| *t == Level::Bad.word()).count();
            assert_eq!(fails, usize::from(!paused), "paused {paused}");
        }
        assert_eq!(
            list_texts(&inputs, true)[1],
            "Your design: passes all 9 checks"
        );
    }

    /// The marks the rows of the check list of `inputs` draw, top to bottom: the level each
    /// one's shape stands for. (A material warning under the rows has a badge of its own.)
    fn marks(ctx: &egui::Context, inputs: &DesignInputs) -> Vec<Level> {
        let visuals = ctx.style().visuals.clone();
        let (output, _) = frames(ctx, inputs, false, None, Vec::new());
        let foot = text_rect(&output, CHECK_GUIDES.last().unwrap().label)
            .expect("the last row")
            .bottom();
        flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                // A filled disc: fails.
                egui::Shape::Circle(c)
                    if c.center.y < foot && c.fill == Level::Bad.color(&visuals) =>
                {
                    Some(Level::Bad)
                }
                // A ring: passes.
                egui::Shape::Circle(c)
                    if c.fill == egui::Color32::TRANSPARENT
                        && c.stroke.color == Level::Good.color(&visuals) =>
                {
                    Some(Level::Good)
                }
                // A triangle: asks for a look.
                egui::Shape::Path(p)
                    if p.closed
                        && p.points.len() == 3
                        && p.points[0].y < foot
                        && p.fill == Level::Caution.color(&visuals) =>
                {
                    Some(Level::Caution)
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn each_state_has_its_own_shape_and_word_as_well_as_its_colour() {
        // Spec 8.2, test 12 (decision C-4): a filled disc where a check fails, a triangle where
        // it asks for a look, a ring where it passes, each with its word beside the name.
        let ctx = egui::Context::default();
        let mut reached = std::collections::BTreeSet::new();
        for inputs in [
            DesignInputs::default(),
            hot_day(),
            passing_design(),
            with_poles(40),
        ] {
            let results = compute_all(&inputs);
            let want: Vec<Level> = CHECK_GUIDES
                .iter()
                .map(|guide| guidance::level(guide, &inputs, &results))
                .collect();
            assert_eq!(marks(&ctx, &inputs), want);
            let (output, _) = frames(&ctx, &inputs, false, None, Vec::new());
            let visuals = ctx.style().visuals.clone();
            for level in &want {
                assert_eq!(
                    text_color(&output, level.word()),
                    Some(level.color(&visuals)),
                    "{level:?}"
                );
            }
            reached.extend(want);
        }
        assert_eq!(reached.len(), 3, "the designs reach every state");
        let words = [Level::Good, Level::Caution, Level::Bad].map(Level::word);
        assert_eq!(words, ["OK", "Check", "Fails"]);
    }

    #[test]
    fn a_check_that_gives_no_badge_has_no_mark_and_no_word() {
        // Short magnets: the end-effect model is out of range, so the checks computed from the
        // pull-out are greyed and judged by the End-effect model row alone. Without back iron
        // the cup wall check does not apply.
        let ctx = egui::Context::default();
        for (inputs, unjudged) in [
            (
                short_magnets(),
                vec![
                    "metal.hot_min_check",
                    "clamps.recommended",
                    "shaft_joint.check",
                    "gearbox.cold_high_check",
                ],
            ),
            (
                design(|i| i.coupling.backiron = 0),
                vec!["materials.cup_wall_check"],
            ),
        ] {
            let results = compute_all(&inputs);
            for guide in &CHECK_GUIDES {
                let mark = mark_level(guide, &inputs, &results);
                assert_eq!(
                    mark.is_none(),
                    unjudged.contains(&guide.check),
                    "{}",
                    guide.check
                );
            }
            let judged = CHECK_GUIDES.len() - unjudged.len();
            assert_eq!(marks(&ctx, &inputs).len(), judged);
            let words = [Level::Good, Level::Caution, Level::Bad].map(Level::word);
            let texts = list_texts(&inputs, false);
            let drawn = texts.iter().filter(|t| words.contains(&t.as_str())).count();
            assert_eq!(drawn, judged);
        }
    }

    #[test]
    fn a_click_on_a_row_returns_its_check_and_the_selected_row_is_raised() {
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let gearbox = guidance::guide("gearbox.cold_high_check").unwrap();
        let (output, clicked) = frames(&ctx, &inputs, false, None, Vec::new());
        assert_eq!(clicked, None);
        let at = text_rect(&output, gearbox.label).unwrap().center();
        // A hover returns nothing; the click returns the check on its release.
        let (_, hovered) = frames(
            &ctx,
            &inputs,
            false,
            None,
            vec![vec![egui::Event::PointerMoved(at)]],
        );
        assert_eq!(hovered, None);
        let click = vec![
            vec![primary_button(at, true)],
            vec![primary_button(at, false)],
        ];
        let (_, clicked) = frames(&ctx, &inputs, false, None, click);
        assert_eq!(clicked, Some(gearbox.check));
        // The selected row sits on the raised panel colour, and no other does.
        let raised = |selected| {
            let ctx = egui::Context::default();
            let fill = ctx.style().visuals.faint_bg_color;
            let (output, _) = frames(&ctx, &inputs, false, selected, Vec::new());
            let label = text_rect(&output, gearbox.label).unwrap();
            flat_shapes(&output)
                .into_iter()
                .filter(|shape| {
                    matches!(shape, egui::Shape::Rect(r) if r.fill == fill && r.rect.contains_rect(label))
                })
                .count()
        };
        assert_eq!((raised(None), raised(Some(gearbox.check))), (0, 1));
        assert_eq!(raised(Some("metal.hot_min_check")), 0);
    }

    #[test]
    fn the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour() {
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let mut passing_hot_day = passing_design();
        passing_hot_day.temperature.duty = hot_day().temperature.duty;
        let mut passing_reversed = passing_design();
        reversed_hub_band(&mut passing_reversed);
        let cases = [
            // The workbook example: four red checks.
            (DesignInputs::default(), 4, 0, Level::Bad),
            // One fixed: the cup wall at its rule's 2.0 mm and up.
            (
                design(|i| i.metal.cup_wall_corner_mm = 2.4),
                3,
                0,
                Level::Bad,
            ),
            (passing_design(), 0, 0, Level::Good),
            // The passing design with the example's gearbox rating: the gearbox check alone.
            (only_the_gearbox_fails(), 1, 0, Level::Bad),
            // The passing design with the inner hub's minimum at 8 µm: the shaft joints alone.
            (only_the_joint_fails(), 1, 0, Level::Bad),
            // The passing design with the inner hub's band reversed: the shaft joints ask for a
            // look.
            (passing_reversed, 0, 1, Level::Caution),
            // The hot day's temperature verdict is amber, counted apart from the red.
            (hot_day(), 4, 1, Level::Bad),
            // The passing design on the hot day: amber alone.
            (passing_hot_day, 0, 1, Level::Caution),
        ];
        for (inputs, bad, caution, colour) in cases {
            let results = compute_all(&inputs);
            let status = design_status(&inputs, &results, false);
            assert_eq!((status.bad, status.caution), (bad, caution));
            assert_eq!(status.total, CHECK_GUIDES.len());
            assert_eq!(status.is_default, inputs == DesignInputs::default());
            let (output, _) = frames(&ctx, &inputs, false, None, Vec::new());
            let text = design_status_text(&status);
            assert_eq!(
                drawn_texts(&output).get(1),
                Some(&text),
                "the status line tops the list, under its heading"
            );
            assert_eq!(text_color(&output, &text), Some(colour.color(&visuals)));
        }
    }

    /// The texts drawn while the status line `text` of `inputs` against `baseline` is hovered.
    fn status_hover(inputs: &DesignInputs, baseline: &DesignInputs, text: &str) -> Vec<String> {
        let results = compute_all(inputs);
        let (_, output) = hover_frames(
            &egui::Context::default(),
            egui::vec2(320.0, 1200.0),
            |output| text_center(output, text),
            |ui| {
                checks_ui(
                    ui,
                    inputs,
                    baseline,
                    &results,
                    false,
                    None,
                    &mut Readouts::default(),
                );
            },
        );
        drawn_texts(&output)
    }

    #[test]
    fn the_example_design_s_status_line_says_on_hover_where_its_values_come_from() {
        // Spec S2-5: "Example design" at the defaults, its hover the original spreadsheet;
        // "Your design" has no hover.
        for (inputs, example) in [(DesignInputs::default(), true), (passing_design(), false)] {
            let results = compute_all(&inputs);
            let text = design_status_text(&design_status(&inputs, &results, false));
            assert_eq!(text.starts_with(EXAMPLE_DESIGN), example, "{text}");
            let texts = status_hover(&inputs, &DesignInputs::default(), &text);
            let hover = EXAMPLE_DESIGN_HOVER.to_owned();
            assert_eq!(texts.contains(&hover), example, "{text}");
        }
    }

    #[test]
    fn the_starting_design_s_status_line_says_on_hover_what_it_is() {
        // UX overhaul spec 7.1: at the session's baseline, when that is not the spreadsheet's
        // design, the line reads "Starting design" and its hover says what the design is; the
        // same design on a panel whose baseline is the spreadsheet's is "Your design", with no
        // hover.
        let inputs = passing_design();
        for (baseline, name, starting) in [
            (passing_design(), STARTING_DESIGN, true),
            (DesignInputs::default(), YOUR_DESIGN, false),
        ] {
            let text = format!("{name}: passes all 10 checks");
            let texts = status_hover(&inputs, &baseline, &text);
            let hover = STARTING_DESIGN_HOVER.to_owned();
            assert_eq!(texts.contains(&hover), starting, "{text}");
            let example = EXAMPLE_DESIGN_HOVER.to_owned();
            assert!(!texts.contains(&example), "{text}");
        }
    }

    #[test]
    fn the_warnings_that_fire_follow_the_rows_and_the_stored_3d_line_closes_the_list() {
        // An aluminium hub behind the inner magnets fires a material warning.
        let inputs = design(|i| i.coupling.backiron = 0);
        let results = compute_all(&inputs);
        let warnings = warning_lines(&results);
        assert!(!warnings.is_empty(), "a design that fires a warning");
        let ctx = egui::Context::default();
        let (output, _) = frames(&ctx, &inputs, false, None, Vec::new());
        let y = |text: &str| {
            text_rect(&output, text)
                .unwrap_or_else(|| panic!("no {text:?}"))
                .top()
        };
        let last_row = CHECK_GUIDES.last().unwrap().label;
        assert!(y(last_row) < y(WARNINGS_HEADING));
        for (_, text) in &warnings {
            assert!(y(WARNINGS_HEADING) < y(text) && y(text) < y(STORED_3D_LABEL));
        }
        assert_eq!(drawn_texts(&output).last().unwrap(), STORED_3D_LABEL);
        // Each warning's mark is its level's shape (decision C-4): the expansion mismatch's
        // caution a triangle, never the fail disc in the caution colour.
        let caution = Level::Caution.color(&ctx.style().visuals);
        let cautions = warnings
            .iter()
            .filter(|(rule, _)| severity_level(rule.severity) == Level::Caution);
        assert!(cautions.count() > 0, "a design that fires a caution");
        let shapes = flat_shapes(&output);
        let disc = |s: &&egui::Shape| matches!(s, egui::Shape::Circle(c) if c.fill == caution);
        let triangle =
            |s: &&egui::Shape| matches!(s, egui::Shape::Path(p) if p.closed && p.fill == caution);
        assert!(!shapes.iter().any(disc));
        assert!(shapes.iter().any(triangle));
        // Without a warning the heading is not drawn.
        let texts = list_texts(&DesignInputs::default(), false);
        assert!(!texts.iter().any(|t| t == WARNINGS_HEADING));
    }
}
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
        }
        // Five red badges and three amber in the table, in the dashboard's colours: every badge
        // drawn but the dashboard's own.
        let visuals = harness.ctx.style().visuals.clone();
```

with:

```rust
        }
        // Five red badges and three amber in the table, in the dashboard's shapes (decision
        // C-4): every mark drawn but the dashboard's own.
        let visuals = harness.ctx.style().visuals.clone();
```

Replace:

```rust
                .into_iter()
                .filter(|shape| matches!(shape, egui::Shape::Circle(c) if c.fill == level.color(&visuals)))
                .count();
```

with:

```rust
                .into_iter()
                .filter(|shape| mark_at(shape, &visuals).is_some_and(|(_, l)| l == level))
                .count();
```

Replace:

```rust
                shapes.iter().any(|shape| {
                    matches!(shape, egui::Shape::Circle(c)
                        if c.fill == level.color(visuals)
                            && in_table(c.center.x)
                            && (c.center.y - row.center().y).abs() < row.height())
                })
```

with:

```rust
                shapes.iter().any(|shape| {
                    mark_at(shape, visuals).is_some_and(|(at, l)| {
                        l == *level
                            && in_table(at.x)
                            && (at.y - row.center().y).abs() < row.height()
                    })
                })
```

Replace:

```rust
    }

    #[test]
    fn out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too() {
```

with:

```rust
    }

    /// Where shape sits and the level it stands for, if it is a state's mark
    /// (checks::state_mark, decision C-4): a filled disc fails, a triangle asks for a look,
    /// a ring passes.
    fn mark_at(shape: &egui::Shape, visuals: &egui::Visuals) -> Option<(egui::Pos2, Level)> {
        match shape {
            egui::Shape::Circle(c) if c.fill == Level::Bad.color(visuals) => {
                Some((c.center, Level::Bad))
            }
            egui::Shape::Circle(c)
                if c.fill == egui::Color32::TRANSPARENT
                    && c.stroke.color == Level::Good.color(visuals) =>
            {
                Some((c.center, Level::Good))
            }
            egui::Shape::Path(p)
                if p.closed && p.points.len() == 3 && p.fill == Level::Caution.color(visuals) =>
            {
                let sum = p
                    .points
                    .iter()
                    .fold(egui::Vec2::ZERO, |s, q| s + q.to_vec2());
                Some(((sum / 3.0).to_pos2(), Level::Caution))
            }
            _ => None,
        }
    }

    #[test]
    fn out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too() {
```

Replace:

```rust
        assert_eq!(drawn("model.end_effect_check"), 1);
        // Seven red badges and four amber in the table.
        let in_table = |level: Level| {
```

with:

```rust
        assert_eq!(drawn("model.end_effect_check"), 1);
        // Seven red badges and four amber in the table, in their shapes.
        let in_table = |level: Level| {
```

Replace:

```rust
                .filter(|shape| {
                    matches!(shape, egui::Shape::Circle(c)
                        if c.fill == level.color(&visuals) && centre(c.center.x))
                })
```

with:

```rust
                .filter(|shape| {
                    mark_at(shape, &visuals).is_some_and(|(at, l)| l == level && centre(at.x))
                })
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib checks:: 2>&1 | tail -n 15
  ```

  Expected: the build fails with 61 errors, the first "cannot find type `DesignInputs` in this scope": the green step adds what the tests call.

- [ ] **Step 3 (green): the check list.**

**`magcoupling-rs/src/gui/checks.rs`**

Replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;
```

with:

```rust
use crate::gui::dashboard::{
    Level, STORED_3D_LABEL, STORED_3D_NOTE, check_level, greyed_by_end_effect, shown_lines,
    status_ui, warnings_ui,
};
use crate::gui::guidance::{self, CheckGuide, FIT_CHECK, counted_guides};
use crate::gui::look;
use crate::gui::readouts::Readouts;
use crate::{DesignInputs, DesignResults};

/// The heading of the check list.
pub const HEADING: &str = "CHECKS";

/// The most of the right column's height the check list takes before it scrolls (spec 4.5:
/// "about 45 %"); the Explain panel has the rest.
pub const CHECKS_SHARE: f32 = 0.45;

/// The side of a state mark's cell [points].
const MARK_SIDE: f32 = 12.0;

/// The state a check's row shows: its guide's level, or none where the check gives no badge (it
/// does not apply, as the cup wall without back iron, or it is computed from a pull-out the
/// end-effect model cannot predict). The Magnets fit row always has one.
pub(crate) fn mark_level(
    guide: &CheckGuide,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> Option<Level> {
    if guide.check == FIT_CHECK {
        Some(guidance::level(guide, inputs, results))
    } else {
        check_level(results, guide.check)
    }
}

/// A check's state as a shape in its level's colour (decision C-4): a filled disc where it
/// fails, a triangle where it asks for a look, a ring where it passes; an empty cell for no
/// state. The level's word is its hover text.
pub(crate) fn state_mark(ui: &mut egui::Ui, level: Option<Level>) {
    let (rect, response) =
        ui.allocate_exact_size(egui::Vec2::splat(MARK_SIDE), egui::Sense::hover());
    let Some(level) = level else {
        return;
    };
    let color = level.color(ui.visuals());
    let centre = rect.center();
    match level {
        Level::Bad => {
            ui.painter().circle_filled(centre, 5.0, color);
        }
        Level::Caution => {
            let corner = |x: f32, y: f32| centre + egui::vec2(x, y);
            ui.painter().add(egui::Shape::convex_polygon(
                vec![corner(0.0, -5.0), corner(5.0, 4.0), corner(-5.0, 4.0)],
                color,
                egui::Stroke::NONE,
            ));
        }
        Level::Good => {
            ui.painter()
                .circle_stroke(centre, 4.0, egui::Stroke::new(1.5, color));
        }
    }
    response.on_hover_text(level.word());
}

/// Draws the check list of `results`, computed from `inputs`: [`HEADING`], the design status
/// line (named against the session's `baseline`), a row per guide that counts while the
/// shaft-clamp section is paused (`clamps_paused`) or not, the material warnings that fire and
/// the stored-3D line. The row of the check `selected` is drawn on the raised panel colour. A
/// row with a result is a readout (`readouts`): its hover text and equation while hovered.
/// Returns the check whose row was clicked.
pub(crate) fn checks_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    baseline: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
    selected: Option<&'static str>,
    readouts: &mut Readouts,
) -> Option<&'static str> {
    ui.colored_label(look::quiet_text(ui.visuals()), HEADING);
    status_ui(ui, inputs, baseline, results, clamps_paused);
    let lines = shown_lines(results, clamps_paused);
    let mut clicked = None;
    for guide in counted_guides(clamps_paused) {
        let level = mark_level(guide, inputs, results);
        let background = ui.painter().add(egui::Shape::Noop);
        let row = ui
            .horizontal(|ui| {
                state_mark(ui, level);
                let label = egui::RichText::new(guide.label);
                let label = if greyed_by_end_effect(results, guide.check) {
                    label.weak()
                } else {
                    label
                };
                ui.label(label);
                if let Some(level) = level {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.colored_label(level.color(ui.visuals()), level.word());
                    });
                }
            })
            .response
            .rect;
        let rect = row.with_max_x(ui.max_rect().right());
        let response = ui.interact(
            rect,
            ui.id().with(("check_row", guide.check)),
            egui::Sense::click(),
        );
        if selected == Some(guide.check) || response.hovered() {
            ui.painter().set(
                background,
                egui::Shape::rect_filled(
                    rect.expand2(egui::vec2(2.0, 1.0)),
                    4.0,
                    ui.visuals().faint_bg_color,
                ),
            );
        }
        if response.clicked() {
            clicked = Some(guide.check);
        }
        if let Some(line) = lines.iter().find(|line| line.path == guide.check) {
            readouts.hover(ui, response, line.path, |show_details| {
                line.tooltip(show_details)
            });
        }
    }
    ui.separator();
    warnings_ui(ui, results, readouts);
    ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;
```

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
use crate::engine::warnings::{Severity, WARNING_RULES, WarningRule};
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
```

with:

```rust
use crate::engine::warnings::{Severity, WARNING_RULES, WarningRule};
use crate::gui::checks::state_mark;
use crate::gui::corrections::{CorrectionIndex, Mark, marker_text, marker_tooltip};
```

Replace:

```rust
        let title = ui.horizontal(|ui| {
            badge(ui, line.level);
            let label = egui::RichText::new(label_for(line.path)).color(quiet);
```

with:

```rust
        let title = ui.horizontal(|ui| {
            state_mark(ui, line.level);
            let label = egui::RichText::new(label_for(line.path)).color(quiet);
```

Replace:

```rust
/// spec 7.1).
fn status_ui(
    ui: &mut egui::Ui,
```

with:

```rust
/// spec 7.1).
pub(crate) fn status_ui(
    ui: &mut egui::Ui,
```

Replace:

```rust
    ui.horizontal(|ui| {
        badge(ui, Some(level));
        let label = egui::RichText::new(guide.label).color(look::quiet_text(ui.visuals()));
```

with:

```rust
    ui.horizontal(|ui| {
        state_mark(ui, Some(level));
        let label = egui::RichText::new(guide.label).color(look::quiet_text(ui.visuals()));
```

Replace:

```rust

/// The material warnings that fire: a badge and the text in the severity's colour, then a link
/// to the teaching note (it asks the panel to open the note).
```

with:

```rust

/// The material warnings that fire: the severity's shape ([`state_mark`]) and the text in its
/// colour, then a link
/// to the teaching note (it asks the panel to open the note).
```

Replace:

```rust
/// to the teaching note (it asks the panel to open the note).
fn warnings_ui(ui: &mut egui::Ui, results: &DesignResults, readouts: &mut Readouts) {
    let lines = warning_lines(results);
```

with:

```rust
/// to the teaching note (it asks the panel to open the note).
pub(crate) fn warnings_ui(ui: &mut egui::Ui, results: &DesignResults, readouts: &mut Readouts) {
    let lines = warning_lines(results);
```

Replace:

```rust
        ui.horizontal(|ui| {
            badge(ui, Some(level));
            ui.add(
```

with:

```rust
        ui.horizontal(|ui| {
            state_mark(ui, Some(level));
            ui.add(
```

Replace:

```rust
    ui.separator();
}

/// A badge: a filled circle in the level's colour, or an empty cell (the results table's check
/// rows draw it too).
pub(crate) fn badge(ui: &mut egui::Ui, level: Option<Level>) {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    if let Some(level) = level {
        ui.painter()
            .circle_filled(rect.center(), 5.0, level.color(ui.visuals()));
        response.on_hover_text(level.word());
    }
}
```

with:

```rust
    ui.separator();
}
```

**`magcoupling-rs/src/gui/readouts.rs`**

Replace:

```rust
    ) {
        self.mark(ui, response.rect, path);
```

with:

```rust
    ) {
        if response.clicked() {
            self.events.clicked = Some(path.to_owned());
        }
        self.tip(ui, response, path, text, true);
    }

    /// [`Readouts::show`] without the click, for a value whose click is its owner's (a check's
    /// row and a tile select it, UX overhaul spec 5.2): its mark, and while hovered
    /// `text(show_details)` and its equation.
    pub fn hover(
        &mut self,
        ui: &egui::Ui,
        response: egui::Response,
        path: &str,
        text: impl FnOnce(bool) -> String,
    ) {
        self.tip(ui, response, path, text, false);
    }

    /// The mark of the value at `path` and, while `response` is hovered, its tooltip:
    /// `text(show_details)`, its equation and, with `open_hint`, [`OPEN_HINT`].
    fn tip(
        &mut self,
        ui: &egui::Ui,
        response: egui::Response,
        path: &str,
        text: impl FnOnce(bool) -> String,
        open_hint: bool,
    ) {
        self.mark(ui, response.rect, path);
```

Replace:

```rust
        }
        if response.clicked() {
            self.events.clicked = Some(path.to_owned());
        }
        let show_details = self.show_details;
```

with:

```rust
        }
        let show_details = self.show_details;
```

Replace:

```rust
            }
            ui.weak(OPEN_HINT);
        });
```

with:

```rust
            }
            if open_hint {
                ui.weak(OPEN_HINT);
            }
        });
```

**`magcoupling-rs/src/gui/results_table.rs`**

Replace:

```rust
use crate::engine::meta::{ResultSet, Value, result_rows};
use crate::gui::corrections::{CorrectionIndex, marker_text};
```

with:

```rust
use crate::engine::meta::{ResultSet, Value, result_rows};
use crate::gui::checks::state_mark;
use crate::gui::corrections::{CorrectionIndex, marker_text};
```

Replace:

```rust
use crate::gui::dashboard::{
    CHECKS, Level, ResultInfo, badge, check_level, failing_checks, hover_text, result_info,
    shown_text,
};
```

with:

```rust
use crate::gui::dashboard::{
    CHECKS, Level, ResultInfo, check_level, failing_checks, hover_text, result_info, shown_text,
};
```

Replace:

```rust
        );
        badge(&mut badge_ui, line.level);
    }
```

with:

```rust
        );
        state_mark(&mut badge_ui, line.level);
    }
```

Replace:

```rust
                if level.is_some() {
                    badge(ui, level);
                }
```

with:

```rust
                if level.is_some() {
                    state_mark(ui, level);
                }
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1068 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** `magcoupling-rs/src/gui/checks.rs`, `mod.rs`, `dashboard.rs`, `readouts.rs`, `results_table.rs`, `panel.rs`: `feat(magcoupler): the check list, a row per check with its state's shape and word, and every state's mark a shape (BL-081 T1)`.

### Task 2: The tiles

**Model:** `fable`.

The four tiles as a module: pull-out (nominal), hot-low with "needs" and its gap, cold-high, the rotating mass; the number leads, large; hot-low and the requirement quoted by BL-068's rule (`guidance::hot_low_quotes`); two rows of two in a narrow region; a click returns the tile, a hover shows the result's tooltip.

**Files:**
- Create: `magcoupling-rs/src/gui/tiles.rs`
- Modify: `magcoupling-rs/src/gui/mod.rs`, `magcoupling-rs/src/gui/guidance.rs`

**Interfaces:**
- Consumes: `checks::state_mark` (Task 1), `Readouts::hover` (Task 1).
- Produces: `tiles::Tile` (`ALL`, `path`, `label`, `number`), `tiles::NEEDS`, `NUMBER_SIZE`, `MIN_TILE_WIDTH`, `tiles::needs_lines` (the requirement as quoted, the gap, whether it is met), `tiles::gap_pieces`, `tiles::tiles_ui(ui, inputs, results, selected, readouts) -> Option<Tile>`; `guidance::hot_low_quotes(inputs, results) -> (String, String)`.

- [ ] **Step 1 (red): the tiles' tests.**

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
pub(crate) mod test_support;
pub mod torque_card;
```

with:

```rust
pub(crate) mod test_support;
// Drawn by the panel from Task 5 of this stage's plan, which removes this allow.
#[allow(dead_code)]
pub mod tiles;
pub mod torque_card;
```

**`magcoupling-rs/src/gui/tiles.rs`**

Create the file with:

```rust
//! The four tiles of the opening tab (UX overhaul spec 4.4): the nominal pull-out, hot-low with
//! what it must reach, cold-high and the rotating mass. On each the number leads, large, and
//! its label follows, small (decision C-6). A tile is a readout: its hover is the result's hover
//! text and equation, and a click selects it, so the Explain panel shows what it is.
//!
//! The numbers are the results' own, through the formatters every readout uses. Hot-low and
//! the requirement are quoted as a pair ([`hot_low_quotes`], BL-068), so a failing tile never
//! shows two equal numbers. Where the model makes no prediction a torque tile says so in place
//! of its number, in the words the Size tab's readouts had.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::dashboard::END_EFFECT_BANNER;
    use crate::gui::guidance::plain_text_violations;
    use crate::gui::test_support::{
        assert_glyphs, drawn_texts, flat_shapes, glyph_context, hover_frames, no_back_iron,
        passing_design, primary_button, short_magnets, sized_frame, text_color, text_rect,
        zero_torque,
    };

    /// Frames of the tiles of `inputs` in a region `width` points wide in `ctx`, with
    /// `selected` selected: a first one with no input, a second so each row has its height,
    /// then one per entry of `events`. What the last drew and the tile it returned.
    fn frames(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        width: f32,
        selected: Option<Tile>,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, Option<Tile>) {
        let results = compute_all(inputs);
        let mut last = None;
        for input in [Vec::new(), Vec::new()].into_iter().chain(events) {
            let mut clicked = None;
            let output = sized_frame(ctx, egui::vec2(width, 600.0), input, |ui| {
                clicked = tiles_ui(ui, inputs, &results, selected, &mut Readouts::default());
            });
            last = Some((output, clicked));
        }
        last.expect("two frames at least")
    }

    /// The rect of `text` in `output`.
    fn rect(output: &egui::FullOutput, text: &str) -> egui::Rect {
        text_rect(output, text).unwrap_or_else(|| panic!("no {text:?} drawn"))
    }

    /// The frames of the tiles, each a filled rect with a hairline, as drawn left to right and
    /// top to bottom.
    fn tile_rects(ctx: &egui::Context, output: &egui::FullOutput) -> Vec<egui::Rect> {
        let fill = ctx.style().visuals.faint_bg_color;
        flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r) if r.fill == fill => Some(r.rect),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_four_tiles_show_the_results_own_numbers_over_their_labels() {
        // UX overhaul spec 4.4 and decision C-6: the nominal pull-out, hot-low, cold-high and
        // the rotating mass, each number as every readout formats it, leading its label.
        let ctx = egui::Context::default();
        for inputs in [DesignInputs::default(), passing_design()] {
            let results = compute_all(&inputs);
            let (output, _) = frames(&ctx, &inputs, 700.0, None, Vec::new());
            // One row of body text: a label may wrap onto two.
            let body = egui::TextStyle::Body.resolve(&ctx.style());
            let row = ctx.fonts(|fonts| fonts.row_height(&body));
            let mut left = f32::NEG_INFINITY;
            for tile in Tile::ALL {
                let number = result_text(&results, tile.path());
                assert_eq!(tile.number(&inputs, &results), Ok(number.clone()));
                let (number, label) =
                    (rect(&output, &number), rect(&output, &tile.label(&results)));
                assert!(number.bottom() <= label.top() + 0.5, "{tile:?}");
                assert!(number.height() > 1.5 * row, "{tile:?}: the number is large");
                assert!(number.left() > left, "{tile:?}: in order from the left");
                left = number.left();
            }
            // Four tiles in one row, equally wide and equally tall.
            let tiles = tile_rects(&ctx, &output);
            assert_eq!(tiles.len(), 4);
            for tile in &tiles {
                assert!((tile.top() - tiles[0].top()).abs() < 0.5, "{tiles:?}");
                assert!((tile.width() - tiles[0].width()).abs() < 0.5, "{tiles:?}");
                assert!((tile.height() - tiles[0].height()).abs() < 0.5, "{tiles:?}");
            }
        }
        assert_eq!(
            Tile::ALL.map(Tile::path),
            [
                "model.pullout_Nm",
                "metal.torque_hot_low_Nm",
                "metal.torque_cold_high_Nm",
                "mass.total_g"
            ]
        );
    }

    #[test]
    fn a_label_is_quiet_and_its_number_keeps_the_text_colour() {
        // UX overhaul spec 6.1: a reader scanning for a number finds it as the bright line of
        // its tile. In a theme that sets a weak colour the labels take it.
        let ctx = egui::Context::default();
        let weak = egui::Color32::from_rgb(1, 2, 3);
        ctx.style_mut(|style| style.visuals.weak_text_color = Some(weak));
        let strong = ctx.style().visuals.strong_text_color();
        let inputs = passing_design();
        let results = compute_all(&inputs);
        let (output, _) = frames(&ctx, &inputs, 700.0, None, Vec::new());
        for tile in Tile::ALL {
            let number = result_text(&results, tile.path());
            assert_eq!(text_color(&output, &number), Some(strong), "{tile:?}");
            let label = tile.label(&results);
            assert_eq!(text_color(&output, &label), Some(weak), "{tile:?}");
        }
        // A tile that meets its requirement says what it needs: the word as a caption, the
        // requirement as a value.
        let (required, _, meets) = needs_lines(&inputs, &results);
        assert!(meets);
        assert_eq!(text_color(&output, NEEDS), Some(weak));
        let text = ctx.style().visuals.text_color();
        assert_eq!(text_color(&output, &required), Some(text));
    }

    #[test]
    fn the_hot_low_tile_says_what_it_needs_and_its_gap_and_fails_by_shape_and_red_on_that_line() {
        // Spec 4.4: "needs <required>" under the label, then the existing gap text. A failing
        // tile is marked by the fail shape and red on its "needs" line, never on the number.
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let discs = |output: &egui::FullOutput| {
            flat_shapes(output)
                .into_iter()
                .filter(|shape| {
                    matches!(shape, egui::Shape::Circle(c) if c.fill == Level::Bad.color(&visuals))
                })
                .count()
        };
        // The example design fails its hot minimum.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let (required, gap, meets) = needs_lines(&inputs, &results);
        assert_eq!((required.as_str(), meets), ("2.500 N·m", false));
        assert_eq!(gap, "short 0.21 N·m (-8.6 %)");
        assert_eq!(gap_pieces(&gap), ["short 0.21 N·m", "(-8.6 %)"]);
        let (output, _) = frames(&ctx, &inputs, 700.0, None, Vec::new());
        let bad = Level::Bad.color(&visuals);
        for text in [NEEDS, required.as_str(), "short 0.21 N·m", "(-8.6 %)"] {
            assert_eq!(text_color(&output, text), Some(bad), "{text}");
        }
        assert_eq!(
            text_color(&output, "2.285 N·m"),
            Some(visuals.strong_text_color())
        );
        assert_eq!(discs(&output), 1);
        let label = rect(&output, HOT_LOW);
        assert!(label.bottom() <= rect(&output, NEEDS).top() + 0.5);
        assert!(rect(&output, NEEDS).bottom() <= rect(&output, "short 0.21 N·m").top() + 0.5);
        // A passing design: the margin in the pass colour, and no fail shape.
        let inputs = passing_design();
        let results = compute_all(&inputs);
        let (required, gap, meets) = needs_lines(&inputs, &results);
        assert!(meets && gap.starts_with("margin +"), "{gap}");
        let (output, _) = frames(&ctx, &inputs, 700.0, None, Vec::new());
        for piece in gap_pieces(&gap) {
            let good = Level::Good.color(&visuals);
            assert_eq!(text_color(&output, &piece), Some(good), "{piece}");
        }
        assert_eq!(discs(&output), 0);
        assert!(text_rect(&output, NEEDS).is_some() && text_rect(&output, &required).is_some());
    }

    #[test]
    fn the_gap_line_breaks_only_between_its_torque_and_its_percent() {
        // UX overhaul plan review E-3: in a narrow tile the gap wrapped inside its brackets,
        // "short 0.21 N·m (-8.6" over "%)". Each piece is drawn whole, on one row, inside its
        // tile, at every width.
        let ctx = egui::Context::default();
        frames(&ctx, &DesignInputs::default(), 700.0, None, Vec::new());
        let body = egui::TextStyle::Body.resolve(&ctx.style());
        let row = ctx.fonts(|fonts| fonts.row_height(&body));
        for inputs in [DesignInputs::default(), passing_design()] {
            let results = compute_all(&inputs);
            let (_, gap, _) = needs_lines(&inputs, &results);
            assert_eq!(gap_pieces(&gap).concat().len(), gap.len() - 1, "{gap}");
            for width in [700.0, 560.0, 300.0] {
                let (output, _) = frames(&ctx, &inputs, width, None, Vec::new());
                let tile = tile_rects(&ctx, &output)[1];
                for piece in gap_pieces(&gap) {
                    let at = rect(&output, &piece);
                    assert!(at.height() < 1.5 * row, "{width}: {piece:?} wrapped");
                    assert!(tile.contains_rect(at), "{width}: {piece:?} at {at:?}");
                }
            }
        }
        assert_eq!(gap_pieces("margin +0 N·m"), ["margin +0 N·m"]);
    }

    #[test]
    fn a_failing_hot_low_tile_never_shows_two_equal_numbers() {
        // Spec 4.4 (BL-068): where hot-low is a hair under the requirement, both would read
        // the same at four significant digits. The tile quotes the pair as the hot-minimum
        // Problem does: hot-low one last digit below.
        let mut inputs = DesignInputs::default();
        let hot_low = compute_all(&inputs).metal.torque_hot_low_Nm;
        inputs.metal.required_min_Nm = hot_low + 1e-7;
        let results = compute_all(&inputs);
        assert!(results.metal.torque_hot_low_Nm < inputs.metal.required_min_Nm);
        let number = Tile::HotLow.number(&inputs, &results).unwrap();
        let (required, _, meets) = needs_lines(&inputs, &results);
        assert!(!meets);
        assert_eq!(required, result_text(&results, Tile::HotLow.path()));
        assert_ne!(required, number);
        // The Problem quotes the same two numbers.
        let problem = (crate::gui::guidance::guide("metal.hot_min_check")
            .unwrap()
            .problem)(&inputs, &results)
        .expect("the check fails");
        assert!(
            problem.contains(&number) && problem.contains(&required),
            "{problem}"
        );
        // A passing tile reads as its readout does, with no digit moved.
        inputs.metal.required_min_Nm = hot_low;
        let results = compute_all(&inputs);
        assert_eq!(
            Tile::HotLow.number(&inputs, &results),
            Ok(result_text(&results, Tile::HotLow.path()))
        );
    }

    #[test]
    fn a_torque_the_model_does_not_predict_reads_so_in_place_of_its_number() {
        // Spec 4.4: the no-prediction texts replace the number as on the Size tab's readouts,
        // and the Hot-low tile shows neither what it needs nor a gap. The mass is still a
        // number.
        let ctx = egui::Context::default();
        // Without back iron, a design file's calibration coefficient of 0: no torque is finite
        // while the end effect is in range. A variation of -1e308: the pull-out stays 2.69 N·m
        // and the two torques of the variation band are not finite.
        let mut zero_coefficient = no_back_iron();
        zero_coefficient.calibration.f_cal_original = 0.0;
        let mut huge_variation = DesignInputs::default();
        huge_variation.metal.variation = -1e308;
        let all = [Tile::Pullout, Tile::HotLow, Tile::ColdHigh];
        for (inputs, words, unpredicted) in [
            (short_magnets(), NO_PREDICTION, &all[..]),
            (zero_coefficient, NO_TORQUE, &all[..]),
            (
                huge_variation,
                NO_TORQUE,
                &[Tile::HotLow, Tile::ColdHigh][..],
            ),
        ] {
            let results = compute_all(&inputs);
            assert_eq!(
                end_effect_banner(&results).is_some(),
                words == NO_PREDICTION,
                "{END_EFFECT_BANNER}"
            );
            for tile in all {
                let number = tile.number(&inputs, &results);
                if unpredicted.contains(&tile) {
                    assert_eq!(number, Err(words), "{tile:?}");
                } else {
                    assert_eq!(number, Ok(result_text(&results, tile.path())), "{tile:?}");
                }
            }
            assert!(Tile::Mass.number(&inputs, &results).is_ok());
            let (output, _) = frames(&ctx, &inputs, 700.0, None, Vec::new());
            let texts = drawn_texts(&output);
            let drawn = texts.iter().filter(|t| *t == words).count();
            assert_eq!(drawn, unpredicted.len(), "{texts:?}");
            assert!(!texts.iter().any(|t| t.starts_with(NEEDS)), "{texts:?}");
        }
        // An unmagnetized inner ring is a prediction of zero, drawn as a number.
        let zero = zero_torque();
        let results = compute_all(&zero);
        assert_eq!(
            Tile::Pullout.number(&zero, &results),
            Ok(result_text(&results, Tile::Pullout.path()))
        );
    }

    #[test]
    fn in_a_narrow_region_the_tiles_wrap_to_two_rows_and_nothing_is_cut() {
        // Spec 8.2, test 15: four in a row where each keeps MIN_TILE_WIDTH, else two rows of
        // two. No tile runs past the region, and every text stays inside its tile.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        for (width, per_row) in [(700.0, 4), (560.0, 4), (520.0, 2), (300.0, 2)] {
            let (output, _) = frames(&ctx, &inputs, width, None, Vec::new());
            let tiles = tile_rects(&ctx, &output);
            assert_eq!(tiles.len(), 4, "{width}");
            let first_row = tiles
                .iter()
                .filter(|t| (t.top() - tiles[0].top()).abs() < 0.5);
            assert_eq!(first_row.count(), per_row, "{width}: {tiles:?}");
            for (tile, rect_of) in Tile::ALL.iter().zip(&tiles) {
                assert!(rect_of.right() <= width + 0.5, "{width}: {rect_of:?}");
                for text in [result_text(&results, tile.path()), tile.label(&results)] {
                    let at = rect(&output, &text);
                    assert!(rect_of.contains_rect(at), "{width}: {text:?} at {at:?}");
                }
            }
        }
    }

    #[test]
    fn a_click_returns_the_tile_a_hover_shows_its_readout_and_the_selected_tile_is_outlined() {
        // Spec 5.2 and decision C-2: a click selects, a hover never does. The hover is the
        // result's own text; the selected tile's hairline is the text colour.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let (output, clicked) = frames(&ctx, &inputs, 700.0, None, Vec::new());
        assert_eq!(clicked, None);
        let at = rect(&output, &Tile::ColdHigh.label(&results)).center();
        let (_, hovered) = frames(
            &ctx,
            &inputs,
            700.0,
            None,
            vec![vec![egui::Event::PointerMoved(at)]],
        );
        assert_eq!(hovered, None);
        let click = vec![
            vec![primary_button(at, true)],
            vec![primary_button(at, false)],
        ];
        let (_, clicked) = frames(&ctx, &inputs, 700.0, None, click);
        assert_eq!(clicked, Some(Tile::ColdHigh));
        // The hover text.
        let label = Tile::Mass.label(&results);
        let (_, hover) = hover_frames(
            &egui::Context::default(),
            egui::vec2(700.0, 600.0),
            |output| rect(output, &label).center(),
            |ui| {
                tiles_ui(ui, &inputs, &results, None, &mut Readouts::default());
            },
        );
        let help = hover_text(Tile::Mass.path(), false).unwrap();
        assert!(drawn_texts(&hover).contains(&help), "{help}");
        // The outline.
        let outlined = |selected| {
            let ctx = egui::Context::default();
            let stroke = ctx.style().visuals.strong_text_color();
            let (output, _) = frames(&ctx, &inputs, 700.0, selected, Vec::new());
            flat_shapes(&output)
                .into_iter()
                .filter(|shape| matches!(shape, egui::Shape::Rect(r) if r.stroke.color == stroke))
                .count()
        };
        assert_eq!((outlined(None), outlined(Some(Tile::Pullout))), (0, 1));
    }

    #[test]
    fn the_tiles_own_words_have_glyphs_and_plain_words() {
        // Spec 8.3: the one new string of the tiles is NEEDS; the rest are the readouts'.
        let ctx = glyph_context();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let (required, gap, _) = needs_lines(&inputs, &results);
        let mut texts = vec![required, gap, NEEDS.to_owned()];
        texts.extend(Tile::ALL.map(|tile| tile.label(&results)));
        for text in &texts {
            assert_glyphs(&ctx, text, "a tile");
            let violations = plain_text_violations(text);
            assert!(violations.is_empty(), "{text:?}: {violations:?}");
        }
    }
}
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib tiles:: 2>&1 | tail -n 15
  ```

  Expected: the build fails with 86 errors, the first "cannot find type `DesignInputs` in this scope": the green step adds what the tests call.

- [ ] **Step 3 (green): the tiles and the pair's quoting.**

**`magcoupling-rs/src/gui/guidance.rs`**

Replace:

```rust
        (false, false) => number_text(quoted_limit(x, bound), path),
    }
}

```

with:

```rust
        (false, false) => number_text(quoted_limit(x, bound), path),
    }
}

/// Hot-low and the requirement as the Hot-low tile quotes the pair (UX overhaul spec 4.4): the
/// rule the hot-minimum Problem quotes them by (BL-068, [`apart_from`]). Each reads as its
/// readout shows it; where hot-low is under the requirement and both would read the same at
/// four significant digits, hot-low is quoted one last digit below. So a failing tile never
/// shows two equal numbers.
pub(crate) fn hot_low_quotes(inputs: &DesignInputs, results: &DesignResults) -> (String, String) {
    const HOT_LOW: &str = "metal.torque_hot_low_Nm";
    let required = inputs.metal.required_min_Nm;
    let hot_low = if result_number(results, HOT_LOW) < required {
        apart_from(results, HOT_LOW, required, false)
    } else {
        result_text(results, HOT_LOW)
    };
    (hot_low, input_text(inputs, "metal.required_min_Nm"))
}

```

**`magcoupling-rs/src/gui/tiles.rs`**

Replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;
```

with:

```rust
use crate::gui::checks::state_mark;
use crate::gui::dashboard::{Level, end_effect_banner, hover_text};
use crate::gui::guidance::{hot_low_quotes, result_number, result_text};
use crate::gui::look;
use crate::gui::plain_text::label_for;
use crate::gui::readouts::Readouts;
use crate::gui::size_controls::{
    HOT_LOW, NO_PREDICTION, NO_TORQUE, predicts, pullout_label, torque_gap_text,
};
use crate::{DesignInputs, DesignResults};

/// The start of the line under the Hot-low tile's label: what hot-low must reach, the
/// requirement ("needs 1.350 N·m").
pub const NEEDS: &str = "needs";

/// The size of a tile's number [points].
pub const NUMBER_SIZE: f32 = 22.0;

/// The narrowest a tile gets in a row of four [points]: in a narrower region the tiles wrap to
/// two rows of two instead of clipping.
pub const MIN_TILE_WIDTH: f32 = 130.0;

/// The gap between tiles [points].
const TILE_GAP: f32 = 6.0;

/// A tile, by the result it shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    /// The nominal pull-out torque at the operating temperature.
    Pullout,
    /// The hot-low torque, with the requirement it must reach.
    HotLow,
    /// The cold-high torque.
    ColdHigh,
    /// The rotating mass.
    Mass,
}

impl Tile {
    /// Every tile, in the order drawn.
    pub const ALL: [Tile; 4] = [Tile::Pullout, Tile::HotLow, Tile::ColdHigh, Tile::Mass];

    /// The result the tile shows.
    pub const fn path(self) -> &'static str {
        match self {
            Tile::Pullout => "model.pullout_Nm",
            Tile::HotLow => "metal.torque_hot_low_Nm",
            Tile::ColdHigh => "metal.torque_cold_high_Nm",
            Tile::Mass => "mass.total_g",
        }
    }

    /// The tile's label: the result's on-screen name; the pull-out's with its operating
    /// temperature, as the Size tab's readout named it.
    pub fn label(self, results: &DesignResults) -> String {
        match self {
            Tile::Pullout => pullout_label(results.metal.op_temp_C),
            Tile::HotLow => HOT_LOW.to_owned(),
            Tile::ColdHigh | Tile::Mass => label_for(self.path()).to_owned(),
        }
    }

    /// The tile's number with its unit, or, for a torque the model does not predict
    /// ([`predicts`]), the words that say so.
    pub fn number(
        self,
        inputs: &DesignInputs,
        results: &DesignResults,
    ) -> Result<String, &'static str> {
        if self != Tile::Mass && !predicts(results, result_number(results, self.path())) {
            return Err(if end_effect_banner(results).is_some() {
                NO_PREDICTION
            } else {
                NO_TORQUE
            });
        }
        Ok(match self {
            Tile::HotLow => hot_low_quotes(inputs, results).0,
            Tile::Pullout | Tile::ColdHigh | Tile::Mass => result_text(results, self.path()),
        })
    }
}

/// The lines under the Hot-low tile's label while the model predicts hot-low: the requirement
/// it must reach, as [`hot_low_quotes`] quotes it (drawn after [`NEEDS`]), its gap to the
/// requirement ([`torque_gap_text`]) and whether it meets it.
pub(crate) fn needs_lines(
    inputs: &DesignInputs,
    results: &DesignResults,
) -> (String, String, bool) {
    let (_, required) = hot_low_quotes(inputs, results);
    let (gap, meets) = torque_gap_text(
        result_number(results, Tile::HotLow.path()),
        inputs.metal.required_min_Nm,
    );
    (required, gap, meets)
}

/// The gap text in the pieces a line may break between: the torque, then its percent in
/// brackets, so a number never leaves its unit ("short 0.21 N·m", "(-8.6 %)").
pub(crate) fn gap_pieces(gap: &str) -> Vec<String> {
    match gap.split_once(" (") {
        Some((torque, percent)) => vec![torque.to_owned(), format!("({percent}")],
        None => vec![gap.to_owned()],
    }
}

/// Draws `pieces` as one line that wraps only between them, a space apart, after the shape of
/// `mark` where there is one.
fn unbroken_line(ui: &mut egui::Ui, mark: Option<Level>, pieces: Vec<egui::RichText>) {
    ui.horizontal_wrapped(|ui| {
        if mark.is_some() {
            state_mark(ui, mark);
        }
        let body = egui::TextStyle::Body.resolve(ui.style());
        ui.spacing_mut().item_spacing.x = ui.fonts(|fonts| fonts.glyph_width(&body, ' '));
        for piece in pieces {
            ui.add(egui::Label::new(piece).extend());
        }
    });
}

/// Draws the tiles of `results`, computed from `inputs`, across the width `ui` has: four in a
/// row, or two rows of two where a row of four would make a tile narrower than
/// [`MIN_TILE_WIDTH`]. The tile `selected` is outlined in the text colour. Each is a readout
/// (`readouts`: its hover). Returns the tile clicked.
pub(crate) fn tiles_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    selected: Option<Tile>,
    readouts: &mut Readouts,
) -> Option<Tile> {
    let available = ui.available_width();
    let columns = if available >= 4.0 * MIN_TILE_WIDTH + 3.0 * TILE_GAP {
        4
    } else {
        2
    };
    let width = ((available - (columns - 1) as f32 * TILE_GAP) / columns as f32).max(0.0);
    let mut clicked = None;
    for row in Tile::ALL.chunks(columns) {
        // One height per row: the tallest tile's content as the last frame drew it.
        let id = ui.id().with(("magcoupling_tile_row", row[0].path()));
        let height: f32 = ui.data(|data| data.get_temp(id)).unwrap_or(0.0);
        let mut tallest: f32 = 0.0;
        ui.horizontal_top(|ui| {
            ui.spacing_mut().item_spacing.x = TILE_GAP;
            for &tile in row {
                let is_selected = selected == Some(tile);
                let (content, response) =
                    tile_ui(ui, tile, (width, height), is_selected, inputs, results);
                tallest = tallest.max(content);
                if response.clicked() {
                    clicked = Some(tile);
                }
                readouts.hover(ui, response, tile.path(), |show_details| {
                    hover_text(tile.path(), show_details).unwrap_or_default()
                });
            }
        });
        if tallest != height {
            ui.data_mut(|data| data.insert_temp(id, tallest));
            ui.ctx().request_repaint();
        }
    }
    clicked
}

/// One tile, `size.0` points wide with its content at least `size.1` points tall: its number
/// (or the words for no prediction), its label and, on the Hot-low tile, what it must reach.
/// Returns its content's own height and its click sensor.
fn tile_ui(
    ui: &mut egui::Ui,
    tile: Tile,
    size: (f32, f32),
    selected: bool,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> (f32, egui::Response) {
    let visuals = ui.visuals();
    let stroke = if selected {
        egui::Stroke::new(1.0, visuals.strong_text_color())
    } else {
        visuals.widgets.noninteractive.bg_stroke
    };
    let frame = egui::Frame::new()
        .fill(visuals.faint_bg_color)
        .stroke(stroke)
        .corner_radius(4.0)
        .inner_margin(8.0);
    let inner = frame.show(ui, |ui| {
        ui.set_width((size.0 - frame.total_margin().sum().x).max(0.0));
        let content = ui
            .vertical(|ui| {
                let number = tile.number(inputs, results);
                match &number {
                    Ok(number) => {
                        ui.label(egui::RichText::new(number).size(NUMBER_SIZE).strong());
                    }
                    Err(words) => {
                        ui.add(egui::Label::new(egui::RichText::new(*words).weak()).wrap());
                    }
                }
                let quiet = look::quiet_text(ui.visuals());
                let label = egui::RichText::new(tile.label(results)).color(quiet);
                ui.add(egui::Label::new(label).wrap());
                if tile == Tile::HotLow && number.is_ok() {
                    let (required, gap, meets) = needs_lines(inputs, results);
                    let level = if meets { Level::Good } else { Level::Bad };
                    let color = level.color(ui.visuals());
                    // A failing tile is marked by the fail shape and red on this line, never on
                    // the number alone (spec 4.4). A passing one: the word muted, the value in
                    // the text colour.
                    let needs = egui::RichText::new(NEEDS);
                    let required = egui::RichText::new(required);
                    if meets {
                        unbroken_line(ui, None, vec![needs.color(quiet), required]);
                    } else {
                        let line = vec![needs.color(color), required.color(color)];
                        unbroken_line(ui, Some(level), line);
                    }
                    let pieces = gap_pieces(&gap).into_iter();
                    let pieces = pieces
                        .map(|p| egui::RichText::new(p).color(color))
                        .collect();
                    unbroken_line(ui, None, pieces);
                }
            })
            .response
            .rect
            .height();
        // The frame's own layout is its row's, left to right: a spacer would sit beside the
        // content, so the row's height is set as the frame's least height.
        ui.set_min_height(size.1);
        content
    });
    let id = ui.id().with(("magcoupling_tile", tile.path()));
    let response = ui.interact(inner.response.rect, id, egui::Sense::click());
    (inner.inner, response)
}

#[cfg(test)]
mod tests {
    use super::*;
```

- [ ] **Step 4: Run the checks.** Expected: `test result: ok. 1077 passed; 0 failed; 6 ignored`; clippy clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** `tiles.rs`, `mod.rs`, `guidance.rs`: `feat(magcoupler): the four tiles, the number leading its muted label (BL-081 T2)`.

### Task 3: The Explain panel

**Model:** `fable`.

The Explain panel as a module: the one `Selection`; the orientation; a check's entry (its guide's Why sentence, and while it does not pass its Problem, fixes and notes, then the rows its verdict badges and the foot links); a tile's entry (its help, the torque card for hot-low, the check of its row, the foot links). The dashboard's row drawing is factored into `dashboard::line_ui`, which the dashboard's own list calls until Task 5.

**Files:**
- Create: `magcoupling-rs/src/gui/explain.rs`
- Modify: `magcoupling-rs/src/gui/mod.rs`, `magcoupling-rs/src/gui/dashboard.rs`, `magcoupling-rs/src/gui/guidance.rs` (the text review's list of strings)

**Interfaces:**
- Consumes: `checks::{mark_level, state_mark}`, `tiles::Tile`.
- Produces: `explain::Selection` (`Nothing`, `Check`, `Result`; `check`, `tile`, `reads_consequences`), `explain::Asked` (`Select`, `Equation`, `StartHere`), `explain::Shown`, `explain::explain_ui(ui, selection, shown, readouts) -> Option<Asked>`, the constants `HEADING`, `ORIENTATION`, `STEPS`, `ABOUT`, `WHAT_IT_IS`, `WHAT_SETS_IT`, `WHAT_A_CHANGE_DOES`, `NUMBERS`, `CHECKS_IT_TOUCHES`, `FULL_LESSON`, `EQUATION`; `dashboard::line_ui`, `dashboard::learn_ui`.

- [ ] **Step 1 (red): the Explain panel's tests.**

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
pub mod diagrams;
pub mod explorer;
```

with:

```rust
pub mod diagrams;
// Drawn by the panel from Task 5 of this stage's plan, which removes this allow.
#[allow(dead_code)]
pub mod explain;
pub mod explorer;
```

**`magcoupling-rs/src/gui/explain.rs`**

Create the file with:

```rust
//! The Explain panel (UX overhaul spec 5): the right column under the check list. It shows one
//! thing, the [`Selection`]: what was clicked last, or the orientation while nothing is. A
//! click selects; a hover never changes the panel (decision C-2).
//!
//! Every entry has the same headings in the same order and leaves out a heading it has nothing
//! for (spec 5.3): its title, [`WHAT_IT_IS`], [`WHAT_SETS_IT`], "What to change",
//! [`WHAT_A_CHANGE_DOES`], [`CHECKS_IT_TOUCHES`], "Watch out", then its foot links
//! ([`FULL_LESSON`], [`EQUATION`]). A check's entry names its rows [`NUMBERS`] and draws them
//! after "Watch out", the notes on its fixes staying beside the fixes (the plan's decision
//! D-18). The texts under the headings are ones the screen drew before this panel: a
//! check's entry holds its guide's Problem, its Why sentence, its fixes and their notes, each
//! with the links it had; a tile's entry holds its result's help text and, for the hot-low
//! torque, the torque card (decision C-10).
//!
//! The selection is view state: in no design file, share link or undo step.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::assumptions::ASSUMPTIONS;
    use crate::engine::explain::plain::mentions_gearbox_protection;
    use crate::gui::dashboard::{NO_NOTE_YET, NOTE_NOT_REVIEWED, dashboard_lines, verdict_display};
    use crate::gui::explorer::reviewed_note;
    use crate::gui::guidance::{Fixes, all_fixes, counted_guides, plain_text_violations};
    use crate::gui::plain_text::label_for;
    use crate::gui::readouts::ReadoutEvents;
    use crate::gui::test_support::{
        assert_glyphs, drawn_texts, flat_shapes, glyph_context, passing_design, primary_button,
        short_magnets, sized_frame, text_color, text_rect, tooltips_at_once, with_poles,
    };
    use crate::gui::typeset::{TermColors, glyph_safe};

    /// The defaults with `edit` applied.
    fn design(edit: impl FnOnce(&mut DesignInputs)) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        edit(&mut inputs);
        inputs
    }

    /// The defaults with a hot day past the temperature limits: the verdict asks for a look.
    fn hot_day() -> DesignInputs {
        design(crate::gui::test_support::hot_day)
    }

    /// The defaults under a 30 mm overall-length claim: the hot-low torque needs longer magnets
    /// and the space claim shorter ones, both failing (a note on each).
    fn length_conflict() -> DesignInputs {
        design(|i| i.metal.max_overall_axial_mm = 30.0)
    }

    /// Designs that between them fail or caution every check.
    fn designs() -> Vec<DesignInputs> {
        vec![
            DesignInputs::default(),
            short_magnets(),
            passing_design(),
            hot_day(),
            with_poles(40),
            length_conflict(),
            design(|i| i.metal.max_diameter_mm = f64::NAN),
        ]
    }

    /// Frames of the Explain panel showing `selection` on `inputs` in `ctx`, a first one with
    /// no input, then one per entry of `events`: what the last drew, what it asked for and
    /// what the readouts heard in it.
    fn frames(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        selection: Selection,
        show_details: bool,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, Option<Asked>, ReadoutEvents) {
        let results = compute_all(inputs);
        let fixes = all_fixes(inputs, &results, false);
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
                    consequences: None,
                    auto: None,
                };
                asked = explain_ui(ui, selection, &shown, &mut readouts);
            });
            last = Some((output, asked, readouts.finish()));
        }
        last.expect("one frame at least")
    }

    /// What the entry of `check` draws on `inputs`.
    fn entry_texts(inputs: &DesignInputs, check: &'static str, show_details: bool) -> Vec<String> {
        let ctx = egui::Context::default();
        let (output, _, _) = frames(&ctx, inputs, Selection::Check(check), show_details, vec![]);
        drawn_texts(&output)
    }

    /// The frames of a click at `at` (move, press, release).
    fn click_at(at: egui::Pos2) -> Vec<Vec<egui::Event>> {
        vec![
            vec![egui::Event::PointerMoved(at)],
            vec![primary_button(at, true)],
            vec![primary_button(at, false)],
        ]
    }

    /// A click on `text` in the Explain panel showing `selection` on `inputs`: what it asked
    /// for and what the readouts heard.
    fn click(
        inputs: &DesignInputs,
        selection: Selection,
        text: &str,
    ) -> (Option<Asked>, ReadoutEvents) {
        let ctx = egui::Context::default();
        let (output, _, _) = frames(&ctx, inputs, selection, false, vec![]);
        let at = text_rect(&output, text)
            .unwrap_or_else(|| panic!("no {text:?} in {:?}", drawn_texts(&output)))
            .center();
        let (_, asked, events) = frames(&ctx, inputs, selection, false, click_at(at));
        (asked, events)
    }

    fn count(texts: &[String], text: &str) -> usize {
        texts.iter().filter(|t| *t == text).count()
    }

    // ------------------------------------------------------------------ the orientation

    #[test]
    fn nothing_selected_shows_the_orientation_in_at_most_sixty_words() {
        // Spec 5.4: what the thing is, then the three moves, then the link to Start here.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let (output, asked, _) = frames(&ctx, &inputs, Selection::Nothing, false, vec![]);
        assert_eq!(asked, None);
        let mut want = vec![HEADING, ORIENTATION];
        want.extend(STEPS);
        want.push(START_HERE_BUTTON);
        assert_eq!(drawn_texts(&output), want);
        let words: usize = std::iter::once(ORIENTATION)
            .chain(STEPS)
            .map(|text| text.split_whitespace().count())
            .sum();
        assert!(words <= 60, "{words} words");
        // The link starts the walk; no About link shows while nothing is selected.
        let (asked, events) = click(&inputs, Selection::Nothing, START_HERE_BUTTON);
        assert_eq!(asked, Some(Asked::StartHere));
        assert_eq!(events, ReadoutEvents::default());
    }

    #[test]
    fn the_panel_s_own_strings_keep_the_text_rules() {
        // Spec 8.3: plain words, glyphs in the app's font, nothing about protecting the
        // gearbox, no pressure-boundary word (the tool models none) and no ring named by a
        // colour.
        let ctx = glyph_context();
        let mut texts = vec![
            HEADING,
            ORIENTATION,
            ABOUT,
            WHAT_IT_IS,
            WHAT_SETS_IT,
            WHAT_A_CHANGE_DOES,
            CHECKS_IT_TOUCHES,
            NUMBERS,
            FULL_LESSON,
            EQUATION,
            crate::gui::checks::HEADING,
        ];
        texts.extend(STEPS);
        for text in texts {
            assert_glyphs(&ctx, text, "the Explain panel");
            assert!(plain_text_violations(text).is_empty(), "{text:?}");
            assert!(!mentions_gearbox_protection(text), "{text:?}");
            let lower = text.to_lowercase();
            for banned in ["sealed", "containment", "leak", "red ring", "blue ring"] {
                assert!(!lower.contains(banned), "{text:?} says {banned:?}");
            }
        }
        // The orientation says what the notes say: torque across a gap without touching, and
        // slip past the pull-out torque.
        for words in ["without touching", "pull-out torque", "slip"] {
            assert!(ORIENTATION.contains(words), "{words}");
        }
    }

    #[test]
    fn about_this_tool_returns_to_the_orientation() {
        let inputs = DesignInputs::default();
        let selection = Selection::Check(CHECK_GUIDES[0].check);
        let (asked, events) = click(&inputs, selection, ABOUT);
        assert_eq!(asked, Some(Asked::Select(Selection::Nothing)));
        assert_eq!(events, ReadoutEvents::default());
        // A path that is no guide's check draws the heading and the link alone.
        let texts = entry_texts(&inputs, "model.pullout_Nm", false);
        assert_eq!(texts, [HEADING, ABOUT]);
    }

    // ------------------------------------------------------------------ a check's entry

    /// The cup wall fix the test guide offers.
    const CUP_WALL_FIX: [(&str, Direction); 1] = [("metal.cup_wall_corner_mm", Direction::Raise)];

    /// A guide of the cup wall check, the test's own, linking to `learn`.
    fn guide_with(learn: Learn) -> CheckGuide {
        CheckGuide {
            check: "materials.cup_wall_check",
            label: "Cup wall",
            problem: |_, results| Some(results.materials.cup_wall_check.clone()),
            why: "A test sentence.",
            fixes: Fixes::Static(&CUP_WALL_FIX),
            learn,
        }
    }

    /// The test guide's entry at `level`, with `fixes` and `notes`.
    fn entry_at<'a>(
        guide: &'a CheckGuide,
        level: Level,
        fixes: &'a [(&'static str, Direction)],
        notes: &'a dyn Fn() -> Vec<String>,
    ) -> Entry<'a> {
        Entry {
            guide,
            mark: Some(level),
            level,
            fixes,
            notes,
        }
    }

    /// Frames of `entry` on the defaults in `ctx`, a first one with no input, then one per
    /// entry of `events`: what the last drew, what it asked for and what the readouts heard.
    fn entry_frames(
        ctx: &egui::Context,
        entry: &Entry<'_>,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, Option<Asked>, ReadoutEvents) {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let mut last = None;
        for input in std::iter::once(Vec::new()).chain(events) {
            let mut readouts = Readouts::default();
            let mut asked = None;
            let output = sized_frame(ctx, egui::vec2(320.0, 1200.0), input, |ui| {
                let shown = Shown {
                    inputs: &inputs,
                    results: &results,
                    fixes: &[],
                    clamps_paused: false,
                    consequences: None,
                    auto: None,
                };
                asked = entry_ui(ui, entry, &shown, &mut readouts);
            });
            last = Some((output, asked, readouts.finish()));
        }
        last.expect("one frame at least")
    }

    /// Where `entry` draws `text`.
    fn entry_text_at(ctx: &egui::Context, entry: &Entry<'_>, text: &str) -> egui::Pos2 {
        let (output, _, _) = entry_frames(ctx, entry, Vec::new());
        text_rect(&output, text)
            .unwrap_or_else(|| panic!("no {text:?} in {:?}", drawn_texts(&output)))
            .center()
    }

    /// A click on `text` drawn by `entry`: what it asked for and what the readouts heard.
    fn click_in_entry(entry: &Entry<'_>, text: &str) -> (Option<Asked>, ReadoutEvents) {
        let ctx = egui::Context::default();
        let at = entry_text_at(&ctx, entry, text);
        let (_, asked, events) = entry_frames(&ctx, entry, click_at(at));
        (asked, events)
    }

    /// The texts drawn while `text` of `entry` is hovered, tooltips shown at once.
    fn hover_in_entry(entry: &Entry<'_>, text: &str) -> Vec<String> {
        let ctx = egui::Context::default();
        tooltips_at_once(&ctx);
        let at = entry_text_at(&ctx, entry, text);
        let hover = vec![vec![egui::Event::PointerMoved(at)], Vec::new()];
        drawn_texts(&entry_frames(&ctx, entry, hover).0)
    }

    #[test]
    fn a_lesson_link_asks_for_its_note_or_its_assumption_s_row_and_a_fix_for_its_input_s_row() {
        let none: &dyn Fn() -> Vec<String> = &Vec::new;
        // A reviewed note: its title (glyph-safe) asks for the note.
        let note = reviewed_note("end_effect").expect("a reviewed note");
        let guide = guide_with(Learn::Note(note.id));
        let green = entry_at(&guide, Level::Good, &CUP_WALL_FIX, none);
        let (asked, events) = click_in_entry(&green, &glyph_safe(note.title));
        assert_eq!(
            (asked, events.note, events.focus),
            (None, Some(note.id), None)
        );
        // An assumption: its label asks for its first input's row; its hover text is the
        // rationale.
        let assumption = ASSUMPTIONS
            .iter()
            .find(|a| a.id == "production_variation")
            .unwrap();
        let guide = guide_with(Learn::Assumption(assumption.id));
        let green = entry_at(&guide, Level::Good, &CUP_WALL_FIX, none);
        let (_, events) = click_in_entry(&green, assumption.label);
        assert_eq!(
            (events.note, events.focus),
            (None, Some(assumption.paths[0]))
        );
        let texts = hover_in_entry(&green, assumption.label);
        assert!(texts.iter().any(|t| t == assumption.rationale));
        // No note, or a note that has not passed review (no note has this id): "no note yet",
        // which asks for nothing; its hover text says why.
        for (learn, reason) in [
            (Learn::None("A test reason."), "A test reason."),
            (Learn::Note("no_such_note"), NOTE_NOT_REVIEWED),
        ] {
            let guide = guide_with(learn);
            let green = entry_at(&guide, Level::Good, &CUP_WALL_FIX, none);
            let (asked, events) = click_in_entry(&green, NO_NOTE_YET);
            assert_eq!((asked, events), (None, ReadoutEvents::default()));
            assert!(hover_in_entry(&green, NO_NOTE_YET).contains(&reason.to_owned()));
        }
        // A failing check's fix asks for its input's row; a two-pick magnet fix for its part's.
        let guide = guide_with(Learn::None("A test reason."));
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let red = entry_at(&guide, Level::Bad, &CUP_WALL_FIX, none);
        let (_, events) = click_in_entry(&red, &fix);
        assert_eq!(
            (events.note, events.focus),
            (None, Some("metal.cup_wall_corner_mm"))
        );
        let part = [("coupling.magnets.part_inner", Direction::CustomHotterGrade)];
        let two_picks = entry_at(&guide, Level::Bad, &part, none);
        let (_, events) = click_in_entry(&two_picks, &fix_text(part[0].0, part[0].1));
        assert_eq!(events.focus, Some("coupling.magnets.part_inner"));
        // The Equation link asks for the check's result, and nothing of the readouts.
        let (asked, events) = click_in_entry(&red, EQUATION);
        assert_eq!(asked, Some(Asked::Equation("materials.cup_wall_check")));
        assert_eq!(events, ReadoutEvents::default());
    }

    #[test]
    fn an_entry_shows_its_problem_and_fixes_where_the_check_does_not_pass() {
        let guide = guide_with(Learn::None("A test reason."));
        let problem = compute_all(&DesignInputs::default())
            .materials
            .cup_wall_check;
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let none: &dyn Fn() -> Vec<String> = &Vec::new;
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let y = |output: &egui::FullOutput, text: &str| {
            text_rect(output, text)
                .unwrap_or_else(|| panic!("no {text:?}"))
                .top()
        };
        // Red and amber alike (an amber check's entry is open: its row was clicked): the
        // headings in the plan's order (decision D-18), the problem in the level's colour, the
        // state word.
        for level in [Level::Bad, Level::Caution] {
            let entry = entry_at(&guide, level, &CUP_WALL_FIX, none);
            let (output, _, _) = entry_frames(&ctx, &entry, Vec::new());
            let order = [
                guide.label,
                WHAT_IT_IS,
                guide.why,
                WHAT_SETS_IT,
                problem.as_str(),
                WHAT_TO_CHANGE,
                fix.as_str(),
                NUMBERS,
                FULL_LESSON,
                EQUATION,
            ];
            for pair in order.windows(2) {
                assert!(
                    y(&output, pair[0]) < y(&output, pair[1]),
                    "{level:?}: {:?} is not above {:?}",
                    pair[0],
                    pair[1]
                );
            }
            assert_eq!(text_color(&output, &problem), Some(level.color(&visuals)));
            assert_eq!(
                text_color(&output, level.word()),
                Some(level.color(&visuals))
            );
            assert_eq!(count(&drawn_texts(&output), WATCH_OUT), 0);
        }
        // Green: why it matters, its numbers and its foot; no problem and no fix.
        let entry = entry_at(&guide, Level::Good, &CUP_WALL_FIX, none);
        let (output, _, _) = entry_frames(&ctx, &entry, Vec::new());
        let texts = drawn_texts(&output);
        for want in [
            guide.label,
            Level::Good.word(),
            WHAT_IT_IS,
            guide.why,
            NUMBERS,
        ] {
            assert_eq!(count(&texts, want), 1, "{want:?}");
        }
        for absent in [WHAT_SETS_IT, WHAT_TO_CHANGE, WATCH_OUT, fix.as_str()] {
            assert_eq!(count(&texts, absent), 0, "{absent:?}");
        }
    }

    #[test]
    fn a_failing_entry_without_a_fix_says_so_and_its_notes_follow_under_watch_out() {
        let guide = guide_with(Learn::None("A test reason."));
        let ctx = egui::Context::default();
        let none: &dyn Fn() -> Vec<String> = &Vec::new;
        let (output, _, _) = entry_frames(&ctx, &entry_at(&guide, Level::Bad, &[], none), vec![]);
        let texts = drawn_texts(&output);
        assert_eq!(count(&texts, WHAT_TO_CHANGE), 1);
        assert_eq!(count(&texts, NO_FIX), 1);
        let note = "A test trade-off.".to_owned();
        let notes = || vec![note.clone()];
        let noted = entry_at(&guide, Level::Bad, &CUP_WALL_FIX, &notes);
        let (output, _, _) = entry_frames(&ctx, &noted, Vec::new());
        assert_eq!(count(&drawn_texts(&output), NO_FIX), 0);
        let fix = fix_text(CUP_WALL_FIX[0].0, CUP_WALL_FIX[0].1);
        let y = |text: &str| text_rect(&output, text).unwrap().top();
        assert!(y(&fix) < y(WATCH_OUT) && y(WATCH_OUT) < y(&note));
    }

    #[test]
    fn an_entry_reads_its_notes_only_where_its_check_does_not_pass() {
        let guide = guide_with(Learn::None("A test reason."));
        let reads = std::cell::Cell::new(0);
        let counting = || {
            reads.set(reads.get() + 1);
            vec!["A test trade-off.".to_owned()]
        };
        for (level, want) in [(Level::Good, 0), (Level::Caution, 1), (Level::Bad, 1)] {
            reads.set(0);
            let entry = entry_at(&guide, level, &CUP_WALL_FIX, &counting);
            let ctx = egui::Context::default();
            let (output, _, _) = entry_frames(&ctx, &entry, Vec::new());
            assert_eq!(reads.get(), want, "{level:?}");
            let shown = count(&drawn_texts(&output), "A test trade-off.");
            assert_eq!(shown, want, "{level:?}");
        }
    }

    /// The text of the link where `learn` leads.
    fn learn_link_text(learn: Learn) -> String {
        match learn {
            Learn::Note(id) => {
                reviewed_note(id).map_or(NO_NOTE_YET.to_owned(), |n| glyph_safe(n.title))
            }
            Learn::Assumption(id) => ASSUMPTIONS
                .iter()
                .find(|a| a.id == id)
                .unwrap()
                .label
                .to_owned(),
            Learn::None(_) => NO_NOTE_YET.to_owned(),
        }
    }

    #[test]
    fn each_check_s_entry_holds_the_problem_why_and_fixes_its_dashboard_block_held() {
        // Spec 8.2, test 11: the same strings and the same links, for every guide on designs
        // that between them fail or caution every check. An amber check's entry is open too:
        // its row was clicked.
        for inputs in designs() {
            let results = compute_all(&inputs);
            let fixes = all_fixes(&inputs, &results, false);
            for (index, guide) in CHECK_GUIDES.iter().enumerate() {
                let texts = entry_texts(&inputs, guide.check, false);
                let what = guide.check;
                let level = guidance::level(guide, &inputs, &results);
                let failing = usize::from(level != Level::Good);
                // (The hot minimum's label is also its torque row's.)
                assert!(count(&texts, guide.label) >= 1, "{what}");
                assert_eq!(count(&texts, guide.why), 1, "{what}");
                let problem = (guide.problem)(&inputs, &results);
                assert_eq!(problem.is_some(), level != Level::Good, "{what}");
                assert_eq!(count(&texts, WHAT_SETS_IT), failing, "{what}");
                if let Some(problem) = &problem {
                    assert_eq!(count(&texts, problem), 1, "{what}: {problem:?}");
                }
                assert_eq!(count(&texts, WHAT_TO_CHANGE), failing, "{what}");
                for &(path, direction) in &fixes[index] {
                    assert_eq!(count(&texts, &fix_text(path, direction)), 1, "{what}");
                }
                let no_fix = usize::from(level != Level::Good && fixes[index].is_empty());
                assert_eq!(count(&texts, NO_FIX), no_fix, "{what}");
                let notes = if level == Level::Good {
                    Vec::new()
                } else {
                    guide_notes(&fixes, index, &inputs, &results)
                };
                assert_eq!(count(&texts, WATCH_OUT), usize::from(!notes.is_empty()));
                for note in &notes {
                    let wanted = notes.iter().filter(|n| *n == note).count();
                    assert_eq!(count(&texts, note), wanted, "{what}: {note:?}");
                }
                assert_eq!(count(&texts, FULL_LESSON), 1, "{what}");
                assert_eq!(count(&texts, &learn_link_text(guide.learn)), 1, "{what}");
                let equation = usize::from(guide.check != FIT_CHECK);
                assert_eq!(count(&texts, EQUATION), equation, "{what}");
            }
        }
        // The designs above reach a note and an amber check.
        let conflict = length_conflict();
        let results = compute_all(&conflict);
        let fixes = all_fixes(&conflict, &results, false);
        assert!(
            (0..CHECK_GUIDES.len())
                .any(|i| !guide_notes(&fixes, i, &conflict, &results).is_empty())
        );
        let hot = hot_day();
        let results = compute_all(&hot);
        assert!(
            CHECK_GUIDES
                .iter()
                .any(|g| guidance::level(g, &hot, &results) == Level::Caution)
        );
    }

    #[test]
    fn the_numbers_behind_a_check_are_the_dashboard_rows_its_verdict_badges() {
        // Every dashboard row with a check is in that check's entry, under its display label
        // with its value as the dashboard drew it; a row no check badges is in none.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let lines = dashboard_lines(&results);
        for &(path, check) in &DASHBOARD {
            let line = lines.iter().find(|line| line.path == path).unwrap();
            for guide in &CHECK_GUIDES {
                let texts = entry_texts(&inputs, guide.check, false);
                let drawn = count(&texts, &line.shown_value());
                if check == Some(guide.check) {
                    assert!(drawn >= 1, "{path} in {}", guide.check);
                    assert!(count(&texts, label_for(path)) >= 1, "{path}");
                } else if check.is_none() {
                    assert_eq!(drawn, 0, "{path} in {}", guide.check);
                }
            }
        }
        // The cup wall's failing verdict reads in plain words (ruling R9), not the engine's.
        let shown = verdict_display(
            "materials.cup_wall_check",
            &results.materials.cup_wall_check,
        )
        .expect("the default cup wall fails");
        let texts = entry_texts(&inputs, "materials.cup_wall_check", false);
        assert_eq!(count(&texts, &shown), 1, "{shown}");
        let engine_words = crate::engine::materials::CUP_WALL_TOO_THIN.trim_end();
        assert!(!texts.iter().any(|t| t.contains(engine_words)));
        // The stored-3D line closes the numbers of the two checks that read the stored fields
        // (owner R-15), and of no other.
        for guide in &CHECK_GUIDES {
            let stored = ["temperature.summary.verdict", "shaft_joint.check"];
            let texts = entry_texts(&inputs, guide.check, false);
            assert_eq!(
                count(&texts, STORED_3D_LABEL),
                usize::from(stored.contains(&guide.check)),
                "{}",
                guide.check
            );
        }
        // The correction ids and their legend show only with the spreadsheet details.
        let hot_min = "metal.hot_min_check";
        let markers: Vec<String> = lines
            .iter()
            .filter(|line| !line.marker.is_empty())
            .filter(|line| DASHBOARD.contains(&(line.path, Some(hot_min))))
            .map(|line| line.marker.clone())
            .collect();
        assert!(!markers.is_empty(), "the hot minimum's rows carry ids");
        for show_details in [false, true] {
            let texts = entry_texts(&inputs, hot_min, show_details);
            for marker in &markers {
                assert_eq!(texts.contains(marker), show_details, "{marker}");
            }
            assert_eq!(count(&texts, MARKER_LEGEND), usize::from(show_details));
        }
    }

    #[test]
    fn the_fit_entry_says_whether_the_blocks_fit_and_guides_a_misfit() {
        let fit = guidance::guide(FIT_CHECK).unwrap();
        let texts = entry_texts(&DesignInputs::default(), FIT_CHECK, false);
        assert_eq!(count(&texts, fit.label), 1);
        assert_eq!((count(&texts, FITS), count(&texts, DOES_NOT_FIT)), (1, 0));
        assert_eq!(count(&texts, Level::Good.word()), 1);
        // 40 poles: the blocks do not fit; the fail mark, the problem and the guide's fixes.
        let inputs = with_poles(40);
        let results = compute_all(&inputs);
        assert_eq!(guidance::level(fit, &inputs, &results), Level::Bad);
        let ctx = egui::Context::default();
        let (output, _, _) = frames(&ctx, &inputs, Selection::Check(FIT_CHECK), false, vec![]);
        let texts = drawn_texts(&output);
        assert_eq!((count(&texts, FITS), count(&texts, DOES_NOT_FIT)), (0, 1));
        let problem = (fit.problem)(&inputs, &results).expect("a misfit has a Problem");
        assert_eq!(count(&texts, &problem), 1);
        // The entry's mark is the fail shape: a filled disc in the failing colour.
        let visuals = ctx.style().visuals.clone();
        let disc = flat_shapes(&output)
            .into_iter()
            .find_map(|shape| match shape {
                egui::Shape::Circle(c) if c.radius == 5.0 => Some(c.fill),
                _ => None,
            });
        assert_eq!(disc, Some(Level::Bad.color(&visuals)));
    }

    #[test]
    fn the_numbers_of_a_passing_check_carry_the_pass_shape_and_never_the_fail_shape() {
        // Decision C-4 (plan review E-1): a filled disc means fails, a ring passes. A passing
        // check's entry draws a ring by its name and one by each row it badges, and no disc.
        let ctx = egui::Context::default();
        let good = Level::Good.color(&ctx.style().visuals);
        let inputs = passing_design();
        let results = compute_all(&inputs);
        let check = "metal.hot_min_check";
        let rows = dashboard_lines(&results)
            .into_iter()
            .filter(|line| DASHBOARD.contains(&(line.path, Some(check))))
            .inspect(|line| assert_eq!(line.level, Some(Level::Good), "{}", line.path))
            .count();
        assert_eq!(rows, 2, "the hot-low torque and its check");
        let (output, _, _) = frames(&ctx, &inputs, Selection::Check(check), false, vec![]);
        let shapes = flat_shapes(&output);
        let circles = |filled: bool| {
            shapes
                .iter()
                .filter(|shape| match shape {
                    egui::Shape::Circle(c) if filled => c.fill == good,
                    egui::Shape::Circle(c) => {
                        c.fill == egui::Color32::TRANSPARENT && c.stroke.color == good
                    }
                    _ => false,
                })
                .count()
        };
        assert_eq!((circles(true), circles(false)), (0, 1 + rows));
    }

    #[test]
    fn a_tile_s_entry_names_the_check_of_its_row_and_a_click_selects_it() {
        // Spec 5.3, "Checks it touches" for a result: the check of its row (plan review C-3,
        // E-5). The Hot-low tile's entry shows the hot minimum check's state and name; a click
        // on the name selects the check. A tile no check badges has no such heading.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let check = "metal.hot_min_check";
        let guide = guidance::guide(check).unwrap();
        let hot_low = Selection::Result(Tile::HotLow.path());
        let (output, _, _) = frames(&ctx, &inputs, hot_low, false, vec![]);
        let texts = drawn_texts(&output);
        assert_eq!(count(&texts, CHECKS_IT_TOUCHES), 1, "{texts:?}");
        assert_eq!(count(&texts, Level::Bad.word()), 1, "{texts:?}");
        let heading = text_rect(&output, CHECKS_IT_TOUCHES).unwrap();
        // The tile's own label is the check's name too: the link is the one under the heading.
        let link = crate::gui::test_support::text_rects(&output, guide.label)
            .into_iter()
            .find(|r| r.top() >= heading.bottom())
            .expect("the check's name under the heading");
        let (_, asked, events) = frames(&ctx, &inputs, hot_low, false, click_at(link.center()));
        assert_eq!(asked, Some(Asked::Select(Selection::Check(check))));
        assert_eq!(events, ReadoutEvents::default());
        for tile in [Tile::Pullout, Tile::ColdHigh, Tile::Mass] {
            let (output, _, _) =
                frames(&ctx, &inputs, Selection::Result(tile.path()), false, vec![]);
            assert_eq!(
                count(&drawn_texts(&output), CHECKS_IT_TOUCHES),
                0,
                "{tile:?}"
            );
        }
    }

    #[test]
    fn every_entry_text_has_glyphs_in_the_app_s_font_and_plain_words() {
        // On designs where each check fails: every guide's entry, and each guide at red
        // whatever the design with its static fixes; the fixed texts, and the hover texts.
        let ctx = egui::Context::default();
        let mut plain: Vec<String> = Vec::new();
        let mut all = designs();
        all.push(design(|i| {
            i.clamps.boss_od_mm = 12.0;
            i.clamps.clamp_length_mm = 3.0;
        }));
        all.push(design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)));
        for inputs in all {
            let results = compute_all(&inputs);
            let fixes = all_fixes(&inputs, &results, false);
            for (index, guide) in CHECK_GUIDES.iter().enumerate() {
                plain.extend(entry_texts(&inputs, guide.check, false));
                let notes = || guide_notes(&fixes, index, &inputs, &results);
                let listed = match guide.fixes {
                    Fixes::Static(list) => list,
                    Fixes::PerTerm(_) => &fixes[index][..],
                };
                let red = Entry {
                    guide,
                    mark: Some(Level::Bad),
                    level: Level::Bad,
                    fixes: listed,
                    notes: &notes,
                };
                let output = sized_frame(&ctx, egui::vec2(320.0, 4000.0), vec![], |ui| {
                    let shown = Shown {
                        inputs: &inputs,
                        results: &results,
                        fixes: &fixes,
                        clamps_paused: false,
                        consequences: None,
                        auto: None,
                    };
                    entry_ui(ui, &red, &shown, &mut Readouts::default());
                });
                plain.extend(drawn_texts(&output));
            }
        }
        plain.extend([NOTE_NOT_REVIEWED, FITS, DOES_NOT_FIT, NO_FIX].map(str::to_owned));
        // An assumption's rationale is the hover text of its lesson link: plain words too.
        for guide in counted_guides(false) {
            match guide.learn {
                Learn::None(reason) => plain.push(reason.to_owned()),
                Learn::Assumption(id) => plain.push(
                    ASSUMPTIONS
                        .iter()
                        .find(|a| a.id == id)
                        .unwrap()
                        .rationale
                        .to_owned(),
                ),
                Learn::Note(_) => {}
            }
        }
        for text in &plain {
            assert_glyphs(&ctx, text, "the Explain panel");
            assert!(
                plain_text_violations(text).is_empty(),
                "{text:?}: {:?}",
                plain_text_violations(text)
            );
        }
    }
}
```

**`magcoupling-rs/src/gui/guidance.rs`**

Replace:

```rust
        emit("panel/welcome strip", crate::gui::panel::WELCOME);

```

with:

```rust
        emit("panel/welcome strip", crate::gui::panel::WELCOME);
        {
            use crate::gui::explain as ex;
            emit("explain/orientation", ex::ORIENTATION);
            for (index, step) in ex::STEPS.iter().enumerate() {
                emit(&format!("explain/orientation step {}", index + 1), step);
            }
            for (tag, text) in [
                ("heading", ex::HEADING),
                ("about", ex::ABOUT),
                ("what it is", ex::WHAT_IT_IS),
                ("what sets it", ex::WHAT_SETS_IT),
                ("what a change does", ex::WHAT_A_CHANGE_DOES),
                ("checks it touches", ex::CHECKS_IT_TOUCHES),
                ("numbers", ex::NUMBERS),
                ("full lesson", ex::FULL_LESSON),
                ("equation", ex::EQUATION),
            ] {
                emit(&format!("explain/{tag}"), text);
            }
            emit("checks/heading", crate::gui::checks::HEADING);
            emit("tiles/needs", crate::gui::tiles::NEEDS);
        }

```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib explain:: 2>&1 | tail -n 15
  ```

  Expected: the build fails with 206 errors, the first "cannot find type `DesignInputs` in this scope": the green step adds what the tests call.

- [ ] **Step 3 (green): the Explain panel and the row drawing it shares.**

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
    for (index, line) in lines.iter().enumerate() {
        let weak = ui.visuals().weak_text_color();
        let quiet = look::quiet_text(ui.visuals());
        let tint = |rich: egui::RichText| if line.greyed { rich.color(weak) } else { rich };
        let title = ui.horizontal(|ui| {
            state_mark(ui, line.level);
            let label = egui::RichText::new(label_for(line.path)).color(quiet);
            ui.add(egui::Label::new(tint(label)).wrap());
        });
        let value = ui.horizontal(|ui| {
            ui.add_space(16.0);
            // One wrapped flow, indented: the correction ids follow the value on its last line
            // or the next. Beside a value wrapped to the full width they would run past the row,
            // and the resizable dashboard would widen every frame.
            ui.horizontal_wrapped(|ui| {
                let value = egui::RichText::new(line.shown_value()).strong();
                ui.add(egui::Label::new(tint(value)).wrap());
                if show_details && !line.marker.is_empty() {
                    ui.add(egui::Label::new(egui::RichText::new(&line.marker).small()).extend());
                }
            });
        });
        let rect = title.response.rect.union(value.response.rect);
        readouts.show_over(ui, rect, line.path, |show_details| {
            line.tooltip(show_details)
        });
        // A check's block under its own row (the rows its verdict only badges have none), then
```

with:

```rust
    for (index, line) in lines.iter().enumerate() {
        line_ui(ui, line, readouts);
        // A check's block under its own row (the rows its verdict only badges have none), then
```

Replace:

```rust

/// The design status line (guidance spec 3.1): whose design it is against the session's
```

with:

```rust

/// Draws one dashboard row: its badge and label, the label quieter than the value
/// ([`look::quiet_text`]), then its value and, with the spreadsheet details, its correction
/// ids, so a narrow side still reads. The row is a readout
/// (`readouts`): its hover text and equation while hovered, a click opens it in the Equation
/// panel. The Explain panel draws a check's rows with it (UX overhaul spec 4.5: "Numbers behind
/// this check").
pub(crate) fn line_ui(ui: &mut egui::Ui, line: &DashboardLine, readouts: &mut Readouts) {
    let show_details = readouts.show_details();
    let weak = ui.visuals().weak_text_color();
    let tint = |rich: egui::RichText| if line.greyed { rich.color(weak) } else { rich };
    let title = ui.horizontal(|ui| {
        state_mark(ui, line.level);
        let label = egui::RichText::new(label_for(line.path)).color(look::quiet_text(ui.visuals()));
        ui.add(egui::Label::new(tint(label)).wrap());
    });
    let value = ui.horizontal(|ui| {
        ui.add_space(16.0);
        // One wrapped flow, indented: the correction ids follow the value on its last line
        // or the next. Beside a value wrapped to the full width they would run past the row,
        // and the resizable side would widen every frame.
        ui.horizontal_wrapped(|ui| {
            let value = egui::RichText::new(line.shown_value()).strong();
            ui.add(egui::Label::new(tint(value)).wrap());
            if show_details && !line.marker.is_empty() {
                ui.add(egui::Label::new(egui::RichText::new(&line.marker).small()).extend());
            }
        });
    });
    let rect = title.response.rect.union(value.response.rect);
    readouts.show_over(ui, rect, line.path, |show_details| {
        line.tooltip(show_details)
    });
}

/// The design status line (guidance spec 3.1): whose design it is against the session's
```

Replace:

```rust
/// hover text.
fn learn_ui(ui: &mut egui::Ui, learn: Learn, readouts: &mut Readouts) {
    match learn {
```

with:

```rust
/// hover text.
pub(crate) fn learn_ui(ui: &mut egui::Ui, learn: Learn, readouts: &mut Readouts) {
    match learn {
```

**`magcoupling-rs/src/gui/explain.rs`**

Replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;
```

with:

```rust
use crate::engine::sizing::FreeVariable;
use crate::gui::checks::{mark_level, state_mark};
use crate::gui::dashboard::why_note_link;
use crate::gui::dashboard::{
    DASHBOARD, DOES_NOT_FIT, FITS, Level, MARKER_LEGEND, STORED_3D_LABEL, STORED_3D_NOTE, learn_ui,
    line_ui, result_info, shown_lines,
};
use crate::gui::explorer::{START_HERE_BUTTON, WATCH_OUT, WHAT_TO_CHANGE, start_here_hover};
use crate::gui::guidance::{
    self, CHECK_GUIDES, CheckGuide, Consequences, Direction, FIT_CHECK, Learn, NO_FIX,
    SIZE_TAB_NOTE, fix_text, guide_notes, row_learn,
};
use crate::gui::input_ui::weak_line;
use crate::gui::look;
use crate::gui::plain_text::help_for;
use crate::gui::readouts::Readouts;
use crate::gui::tiles::Tile;
use crate::gui::torque_card::CARD_HEADING;
use crate::{DesignInputs, DesignResults};

/// The heading of the Explain panel.
pub const HEADING: &str = "EXPLAIN";

/// The orientation's first paragraph (spec 5.4): what the thing is.
pub const ORIENTATION: &str = "Two rings of magnets, one inside the other, pass torque across the gap between them without touching. Up to the pull-out torque the rings turn together; past it they slip.";

/// The orientation's three moves (spec 5.4).
pub const STEPS: [&str; 3] = [
    "1. Set the size on the left.",
    "2. Read the torque above the drawing.",
    "3. Open any check that needs attention.",
];

/// The link at the head of an entry that returns to the orientation (spec 5.1).
pub const ABOUT: &str = "About this tool";

/// An entry's headings (spec 5.3), with [`WHAT_TO_CHANGE`] and [`WATCH_OUT`], which the
/// teaching notes use too.
pub const WHAT_IT_IS: &str = "What it is";
pub const WHAT_SETS_IT: &str = "What sets it";
pub const WHAT_A_CHANGE_DOES: &str = "What a change does";
pub const NUMBERS: &str = "Numbers behind this check";
pub const CHECKS_IT_TOUCHES: &str = "Checks it touches";

/// The head of an entry's lesson link: its teaching note, its assumption, or the line that
/// says no note covers it yet.
pub const FULL_LESSON: &str = "Full lesson:";

/// The foot link that opens the entry's result in the Equation panel (decision C-11).
pub const EQUATION: &str = "Equation";

/// What the Explain panel shows (spec 5.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Selection {
    /// Nothing is selected: the orientation.
    #[default]
    Nothing,
    /// A check, by its guide's check path ([`CheckGuide::check`]).
    Check(&'static str),
    /// A result, by its path: one of the four tiles ([`Tile::path`]).
    Result(&'static str),
}

impl Selection {
    /// The check selected, if one is.
    pub fn check(self) -> Option<&'static str> {
        match self {
            Selection::Check(check) => Some(check),
            Selection::Nothing | Selection::Result(_) => None,
        }
    }

    /// The tile selected, if one is.
    pub fn tile(self) -> Option<Tile> {
        match self {
            Selection::Result(path) => Tile::ALL.into_iter().find(|tile| tile.path() == path),
            Selection::Nothing | Selection::Check(_) => None,
        }
    }

    /// Whether the entry reads the Size tab's consequences ([`Shown::consequences`]): the
    /// panel asks its cache for them only then.
    pub fn reads_consequences(self) -> bool {
        self.tile() == Some(Tile::HotLow)
    }
}

/// What a click in the Explain panel asks of the panel that owns it. A note and an input's row
/// are asked for through the readouts, as every other link asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Asked {
    /// Show this instead.
    Select(Selection),
    /// Open this result in the Equation panel. The selection stays (decision C-25).
    Equation(&'static str),
    /// Start the Start here walk (spec 5.4).
    StartHere,
}

/// What the entries are drawn from: the design shown, its results, every guide's fixes in
/// [`CHECK_GUIDES`]' order and the shaft-clamp section's pause; for an entry that reads them
/// ([`Selection::reads_consequences`]), the Size tab's consequences of the design, with the
/// size Auto sets while it is on.
pub(crate) struct Shown<'a> {
    pub inputs: &'a DesignInputs,
    pub results: &'a DesignResults,
    pub fixes: &'a [Vec<(&'static str, Direction)>],
    pub clamps_paused: bool,
    pub consequences: Option<&'a Consequences>,
    pub auto: Option<FreeVariable>,
}

/// Draws the Explain panel: [`HEADING`], with [`ABOUT`] beside it while something is selected,
/// then the entry of `selection`. Returns what a click asked for.
pub(crate) fn explain_ui(
    ui: &mut egui::Ui,
    selection: Selection,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let mut asked = None;
    ui.horizontal(|ui| {
        ui.colored_label(look::quiet_text(ui.visuals()), HEADING);
        if selection != Selection::Nothing {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.link(ABOUT).clicked() {
                    asked = Some(Asked::Select(Selection::Nothing));
                }
            });
        }
    });
    let entry = match selection {
        Selection::Nothing => orientation_ui(ui, shown.clamps_paused),
        Selection::Check(check) => check_ui(ui, check, shown, readouts),
        Selection::Result(path) => result_ui(ui, path, shown, readouts),
    };
    asked.or(entry)
}

/// The orientation (spec 5.4): what the thing is, the three moves, and the link to the Start
/// here walk.
fn orientation_ui(ui: &mut egui::Ui, clamps_paused: bool) -> Option<Asked> {
    ui.add(egui::Label::new(ORIENTATION).wrap());
    ui.add_space(4.0);
    for step in STEPS {
        ui.add(egui::Label::new(step).wrap());
    }
    ui.add_space(4.0);
    ui.link(START_HERE_BUTTON)
        .on_hover_text(start_here_hover(clamps_paused))
        .clicked()
        .then_some(Asked::StartHere)
}

/// One of an entry's headings.
fn heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(4.0);
    ui.colored_label(look::quiet_text(ui.visuals()), text);
}

/// What a check's entry is drawn with: its guide, the state its row shows
/// ([`mark_level`]), its level, the fixes to show (empty: the no-fix line) and the notes on
/// them ([`guide_notes`]), read only where the check does not pass.
struct Entry<'a> {
    guide: &'a CheckGuide,
    mark: Option<Level>,
    level: Level,
    fixes: &'a [(&'static str, Direction)],
    notes: &'a dyn Fn() -> Vec<String>,
}

/// The entry of the check `check` in the design shown ([`entry_ui`]); nothing for a path that
/// is no guide's check.
fn check_ui(
    ui: &mut egui::Ui,
    check: &'static str,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let index = CHECK_GUIDES.iter().position(|g| g.check == check)?;
    let guide = &CHECK_GUIDES[index];
    let notes = || guide_notes(shown.fixes, index, shown.inputs, shown.results);
    let entry = Entry {
        guide,
        mark: mark_level(guide, shown.inputs, shown.results),
        level: guidance::level(guide, shown.inputs, shown.results),
        fixes: shown.fixes.get(index).map_or(&[], Vec::as_slice),
        notes: &notes,
    };
    entry_ui(ui, &entry, shown, readouts)
}

/// A check's entry: its state and name; why it matters; then, where it does not pass, what is
/// wrong in this design (in the level's colour), the fixes, each a link that shows its input's
/// row, or the line saying no listed change helps, and the notes on those fixes; the numbers
/// the check reads, each a readout; and the foot links.
fn entry_ui(
    ui: &mut egui::Ui,
    entry: &Entry<'_>,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let guide = entry.guide;
    ui.horizontal_wrapped(|ui| {
        state_mark(ui, entry.mark);
        ui.strong(guide.label);
        if let Some(mark) = entry.mark {
            ui.colored_label(mark.color(ui.visuals()), mark.word());
        }
    });
    heading(ui, WHAT_IT_IS);
    ui.add(egui::Label::new(guide.why).wrap());
    if entry.level != Level::Good {
        if let Some(problem) = (guide.problem)(shown.inputs, shown.results) {
            heading(ui, WHAT_SETS_IT);
            let text = egui::RichText::new(problem).color(entry.level.color(ui.visuals()));
            ui.add(egui::Label::new(text).wrap());
        }
        heading(ui, WHAT_TO_CHANGE);
        if entry.fixes.is_empty() {
            ui.add(egui::Label::new(NO_FIX).wrap());
        }
        for &(path, direction) in entry.fixes {
            if ui.link(fix_text(path, direction)).clicked() {
                readouts.focus_input(path);
            }
        }
        let notes = (entry.notes)();
        if !notes.is_empty() {
            heading(ui, WATCH_OUT);
            for note in notes {
                ui.add(egui::Label::new(note).wrap());
            }
        }
    }
    heading(ui, NUMBERS);
    numbers_ui(ui, guide, entry.level, shown, readouts);
    let equation = result_info(guide.check).map(|_| guide.check);
    foot_ui(ui, guide.learn, equation, readouts)
}

/// The numbers behind `guide`'s check: the dashboard rows its verdict badges, drawn as the
/// dashboard drew them (each a readout), with the stored-3D line under the rows that read the
/// stored fields; for the Magnets fit check, which no result holds, whether the blocks fit.
fn numbers_ui(
    ui: &mut egui::Ui,
    guide: &CheckGuide,
    level: Level,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) {
    if guide.check == FIT_CHECK {
        let value = if level == Level::Bad {
            DOES_NOT_FIT
        } else {
            FITS
        };
        ui.add(egui::Label::new(egui::RichText::new(value).strong()).wrap());
        return;
    }
    if readouts.show_details() {
        ui.add(egui::Label::new(egui::RichText::new(MARKER_LEGEND).small().weak()).wrap());
    }
    let mut stored_3d = false;
    for line in shown_lines(shown.results, shown.clamps_paused) {
        let badged_by = DASHBOARD
            .iter()
            .find(|&&(path, _)| path == line.path)
            .and_then(|&(_, check)| check);
        if badged_by == Some(guide.check) {
            line_ui(ui, &line, readouts);
            stored_3d |= line.stored_3d;
        }
    }
    if stored_3d {
        ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
    }
}

/// The entry of the result at `path`, one of the tiles' (spec 5.3): its number and its name;
/// what it is (the result's help text); for the hot-low torque, what a change does (the
/// torque card, [`card_ui`]); the check that badges its row, a link that selects it; and the
/// foot links. Nothing for a path that is no tile's.
fn result_ui(
    ui: &mut egui::Ui,
    path: &'static str,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let tile = Selection::Result(path).tile()?;
    ui.horizontal_wrapped(|ui| {
        match tile.number(shown.inputs, shown.results) {
            Ok(number) => ui.strong(number),
            Err(words) => ui.weak(words),
        };
        ui.colored_label(look::quiet_text(ui.visuals()), tile.label(shown.results));
    });
    let help = help_for(path);
    if !help.is_empty() {
        heading(ui, WHAT_IT_IS);
        ui.add(egui::Label::new(help).wrap());
    }
    if tile == Tile::HotLow
        && let Some(consequences) = shown.consequences
    {
        heading(ui, WHAT_A_CHANGE_DOES);
        card_ui(ui, consequences, shown.auto, readouts);
    }
    let mut asked = None;
    let check = DASHBOARD
        .iter()
        .find(|&&(row, _)| row == path)
        .and_then(|&(_, check)| check);
    if let Some(guide) = check.and_then(guidance::guide) {
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
    let learn = row_learn(path).unwrap_or(Learn::None(""));
    asked.or(foot_ui(ui, learn, Some(path), readouts))
}

/// The torque card (slice 3 spec 6.1; decision C-10 puts it here): its heading with the Why
/// link to how the coupling makes torque, then its lines for the design shown, weak, as the
/// Size tab drew them.
fn card_ui(
    ui: &mut egui::Ui,
    consequences: &Consequences,
    auto: Option<FreeVariable>,
    readouts: &mut Readouts,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label(CARD_HEADING);
        if why_note_link(ui, SIZE_TAB_NOTE) {
            readouts.open_note(SIZE_TAB_NOTE);
        }
    });
    for line in consequences.card.lines(auto) {
        weak_line(ui, &line);
    }
}

/// An entry's foot: [`FULL_LESSON`] and where `learn` leads ([`learn_ui`]), then, for an entry
/// with a result (`equation`), the [`EQUATION`] link. Returns what the Equation link asked.
fn foot_ui(
    ui: &mut egui::Ui,
    learn: Learn,
    equation: Option<&'static str>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    ui.separator();
    ui.horizontal_wrapped(|ui| {
        ui.label(FULL_LESSON);
        learn_ui(ui, learn, readouts);
    });
    let path = equation?;
    ui.link(EQUATION).clicked().then_some(Asked::Equation(path))
}

#[cfg(test)]
mod tests {
    use super::*;
```

- [ ] **Step 4: Run the checks.** Expected: `test result: ok. 1090 passed; 0 failed; 6 ignored`; clippy clean; `cargo fmt` changes nothing.

- [ ] **Step 5: Commit** `explain.rs`, `mod.rs`, `dashboard.rs`, `guidance.rs`: `feat(magcoupler): the Explain panel, its orientation and the entries of a check and a tile (BL-081 T3)`.

### Task 4: gui-smoke reads the new layout

**Model:** `fable`.

The browser smoke run's prompt describes the first view, a check's entry and its lesson link, and the torque card in the Hot-low tile's entry (decision D-16); its verdict test lists the renamed checks. The prompt's Rust guard (`app::tests::the_smoke_test_share_link_opens_its_design`) names the new words as the page draws them, and no longer the welcome strip. The run itself is Task 7's.

**Files:**
- Modify: `magcoupling-rs/src/app.rs`, `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`

- [ ] **Step 1 (red): the prompt's guard.**

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        );
        // Spec R14 (guidance slice 2): the steps name the welcome strip, a note's What to change
        // and its sources fold, the Geometry tab and the dashboard's last row as the page draws
        // them, so the prompt cannot go stale unnoticed.
        assert!(script.contains(crate::gui::WELCOME), "the welcome strip");
        assert!(
```

with:

```rust
        );
        // Spec R14 (guidance slice 2): the steps name a note's What to change and its sources
        // fold, the Geometry tab and the check list's last row as the page draws them, so the
        // prompt cannot go stale unnoticed.
        assert!(
```

Replace:

```rust
            crate::gui::guidance::guide(crate::gui::guidance::FIT_CHECK).expect("the fit guide");
        assert!(script.contains(fit.label), "the dashboard's last row");
        // Slice 3 (spec 8.6, test 26): the card step quotes the torque card whole as the page
```

with:

```rust
            crate::gui::guidance::guide(crate::gui::guidance::FIT_CHECK).expect("the fit guide");
        assert!(script.contains(fit.label), "the check list's last row");
        // UX overhaul slice B, stage 2 (spec 4.4, 4.5 and 5): the first view and the check's
        // entry are named by the words the page draws: the right column's two headings, the
        // orientation's opening words, the Hot-low tile's line, the entry's headings and its
        // lesson link; and the card step looks for the card under its heading in the entry.
        {
            use crate::gui::explain::{
                FULL_LESSON, NUMBERS, ORIENTATION, WHAT_A_CHANGE_DOES, WHAT_IT_IS,
            };
            let opening: String = ORIENTATION
                .split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ");
            for (what, text) in [
                ("the check list's heading", crate::gui::checks::HEADING),
                ("the Explain panel's heading", crate::gui::explain::HEADING),
                ("the orientation's opening words", opening.as_str()),
                ("the Hot-low tile's line", crate::gui::tiles::NEEDS),
                ("the entry's first heading", WHAT_IT_IS),
                ("the entry's numbers", NUMBERS),
                ("the entry's lesson link", FULL_LESSON),
                ("the tile's entry's card heading", WHAT_A_CHANGE_DOES),
            ] {
                assert!(
                    script.contains(text),
                    "the smoke prompt names {what}: {text}"
                );
            }
            let hot_low = crate::gui::guidance::guide("metal.hot_min_check").expect("its guide");
            assert!(script.contains(hot_low.label), "the check the step opens");
            let entry = script
                .find("check_entry=true")
                .expect("the entry's verdict");
            let link = script
                .find("lesson_link_opened=true")
                .expect("the link's verdict");
            assert!(
                entry < link && script[entry..link].contains(FULL_LESSON),
                "the check's entry is judged before its lesson link is clicked"
            );
            // The dashboard is gone: the prompt does not send the agent looking for it.
            assert!(!script.to_lowercase().contains("dashboard"));
        }
        // Slice 3 (spec 8.6, test 26): the card step quotes the torque card whole as the page
```

Replace:

```rust
        assert!(
            between("dashboard_scrolls=true", "page.goto(").contains(collect),
            "the first page's console messages are collected before the link is opened again"
```

with:

```rust
        assert!(
            between("tooltip_plain=true was judged", "page.goto(").contains(collect),
            "the first page's console messages are collected before the link is opened again"
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib smoke 2>&1 | tail -n 15
  ```

  Expected: 1 failed: `the_smoke_test_share_link_opens_its_design`.

- [ ] **Step 3 (green): the prompt and the verdict test.**

**`.claude/workflows/gui-smoke.js`**

Replace:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), a Why link opens a note, an input tooltip shows no spreadsheet details, the dashboard scrolls at 1280x800, the torque card shows its rows between the readouts and the drawing, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

with:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows four tiles over the drawing and the check list over the Explain panel, a click on a check shows its entry and its lesson link opens a note, an input tooltip shows no spreadsheet details, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

Replace:

```js
const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'why_link_opened', 'tooltip_plain', 'dashboard_scrolls', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
```

with:

```js
const MAGCOUPLING_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'check_entry', 'lesson_link_opened', 'tooltip_plain', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
```

Replace:

```js
    details_toggle: { type: 'boolean' },
    why_link_opened: { type: 'boolean' },
    tooltip_plain: { type: 'boolean' },
```

with:

```js
    details_toggle: { type: 'boolean' },
    check_entry: { type: 'boolean' },
    lesson_link_opened: { type: 'boolean' },
    tooltip_plain: { type: 'boolean' },
```

Replace:

```js
    tooltip_plain: { type: 'boolean' },
    dashboard_scrolls: { type: 'boolean' },
    torque_card: { type: 'boolean' },
```

with:

```js
    tooltip_plain: { type: 'boolean' },
    torque_card: { type: 'boolean' },
```

Replace:

```js

// The torque card's heading, which the calculator's first view and its card step both look for:
// typed once here. app.rs's smoke test holds it equal to gui::torque_card::CARD_HEADING.
const CARD_HEADING = 'One step of each size control'
```

with:

```js

// The torque card's heading, which the card step looks for in the Explain panel: typed once here. app.rs's smoke test holds it equal to gui::torque_card::CARD_HEADING.
const CARD_HEADING = 'One step of each size control'
```

Replace:

```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Load design
// button, a Why link, the Geometry tab), uploads a file, moves the mouse (a tooltip, a scroll)
// and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
const MAGCOUPLING_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_resize,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_run_code_unsafe,mcp__plugin_playwright_playwright__browser_click,mcp__plugin_playwright_playwright__browser_file_upload,mcp__plugin_playwright_playwright__browser_close"'
```

with:

```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Load design
// button, a check's row, a lesson link, the Geometry tab, a tile), uploads a file, moves the mouse
// (a tooltip, a scroll) and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
const MAGCOUPLING_TOOLS = 'load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_resize,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_run_code_unsafe,mcp__plugin_playwright_playwright__browser_click,mcp__plugin_playwright_playwright__browser_file_upload,mcp__plugin_playwright_playwright__browser_close"'
```

Replace:

```js
// deterministic state through its share link, checked in
// the console (the page is one canvas). A new page opens on Size and torque under the welcome
// strip (decision G-5), so the first screenshot shows both with no click. One click on the
// canvas, on the "Load design" button found in a screenshot, checks the web file picker (rfd's
```

with:

```js
// deterministic state through its share link, checked in
// the console (the page is one canvas). A new page opens on Size and torque (decision G-5): since
// slice B, stage 2 (spec 4.4 and 4.5), the first screenshot shows the four tiles over the drawing
// and the check list over the Explain panel's orientation with no click. One click on the
// canvas, on the "Load design" button found in a screenshot, checks the web file picker (rfd's
```

Replace:

```js
// canvas, on the "Load design" button found in a screenshot, checks the web file picker (rfd's
// HTML overlay) and the file load. Then (spec R14, guidance slice 2) a dashboard Why link opens
// a teaching note, an input row's tooltip shows no spreadsheet details while the toggle is
// off, and the dashboard scrolls to its last row at 1280 x 800. Then (slice 3, spec 7.2 and
// 8.6) the link is opened again with a timer on every frame the app draws: on that fresh page the
// torque card shows its heading, caption, Auto line and four rows between the readouts and the
// drawing, and one drag of the gap slider is timed frame by frame. Last (slice A, spec 7.2 and
// 8.1 test 5) the address: the drag was a settled edit, so the app rewrote the address with
```

with:

```js
// canvas, on the "Load design" button found in a screenshot, checks the web file picker (rfd's
// HTML overlay) and the file load. Then (spec 5.2 and 5.3) a click on a check's row shows its
// entry in the Explain panel, which scrolls on its own, and the entry's lesson link opens a
// teaching note; and an input row's tooltip shows no spreadsheet details while the toggle is
// off. Then (slice 3, spec 7.2 and 8.6) the link is opened again with a timer on every frame the
// app draws: on that fresh page a click on the Hot-low tile shows the torque card's heading,
// caption, Auto line and four rows in the Explain panel (decision C-10), and one drag of the gap
// slider is timed frame by frame. Last (slice A, spec 7.2 and
// 8.1 test 5) the address: the drag was a settled edit, so the app rewrote the address with
```

Replace:

```js
// navigation, so each page's messages are collected after its last step, before the next
// navigation or the close. src/app.rs checks that the prompt names the strip, the toggle, the
// note's parts, the tab, the row and the card's words as the page does, that the card step
// installs FRAME_TIMER, that the drag step reads SMOKE_GAP_SHOWN and the sizing lines' prefix as
```

with:

```js
// navigation, so each page's messages are collected after its last step, before the next
// navigation or the close. src/app.rs checks that the prompt names the right column's headings,
// the orientation, the entry's headings, the toggle, the note's parts, the tab, the row and the
// card's words as the page does, that the card step
// installs FRAME_TIMER, that the drag step reads SMOKE_GAP_SHOWN and the sizing lines' prefix as
```

Replace:

```js
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: first set the window to 1280 x 800 CSS pixels with browser_resize (the laptop size the dashboard must scroll at). Then the starting design: navigate to ${url}/tools/magcoupler/ (the clean address, with no share link) and wait 8 seconds (WASM init). Take a screenshot and read the address's query with browser_run_code_unsafe, code: async (page) => page.evaluate(() => location.search). starting_design=true only if the dashboard (the right side) is topped by the line "${STARTING_STATUS}", the header row at the top of the page (it may wrap to a second line) shows the button "Open the example design" right after "Reset all", and the code returned an empty string (at the starting design the address stays clean). Then the look: read the page's background with browser_run_code_unsafe, code: async (page) => page.evaluate(() => getComputedStyle(document.body).backgroundColor). look=true only if the code returned "${PAGE_BACKGROUND}" and the screenshot shows the calculator's own look: a charcoal page; at the top left a small crimson square right before the title "Magnetic coupling calculator"; in the row of tabs the selected tab, "Size and torque", marked by a crimson line under its name and not by a filled button; and every text drawn in one clean sans-serif face, with no empty box where a character should be. Collect the console messages at level "info" (this page's: the next step navigates away).
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
```

with:

```js
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: first set the window to 1280 x 800 CSS pixels with browser_resize (the laptop size of the first view). Then the starting design: navigate to ${url}/tools/magcoupler/ (the clean address, with no share link) and wait 8 seconds (WASM init). Take a screenshot and read the address's query with browser_run_code_unsafe, code: async (page) => page.evaluate(() => location.search). starting_design=true only if the check list (the right side), under its heading "CHECKS", is topped by the line "${STARTING_STATUS}", the header row at the top of the page (it may wrap to a second line) shows the button "Open the example design" right after "Reset all", and the code returned an empty string (at the starting design the address stays clean). Then the look: read the page's background with browser_run_code_unsafe, code: async (page) => page.evaluate(() => getComputedStyle(document.body).backgroundColor). look=true only if the code returned "${PAGE_BACKGROUND}" and the screenshot shows the calculator's own look: a charcoal page; at the top left a small crimson square right before the title "Magnetic coupling calculator"; in the row of tabs the selected tab, "Size and torque", marked by a crimson line under its name and not by a filled button; and every text drawn in one clean sans-serif face, with no empty box where a character should be. Collect the console messages at level "info" (this page's: the next step navigates away).
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
```

Replace:

```js
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (the size controls on the left, dashboard on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the middle region shows the first view with no click: at its top the line "Set the size -> read the torque -> open any red check to see why and what to change." with a "Hide" button, under it a row of tabs starting "Size and torque" (the selected tab), and under the tabs the torque readouts, then the heading "${CARD_HEADING}" with a link starting "Why:" beside it, open, with lines of text under it, then a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft. details_toggle=true only if the header row at the top of the page (it may wrap to a second line) shows, after the "Equation panel" button, a checkbox labelled "Show spreadsheet details", unticked.
Then the design file picker: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
```

with:

```js
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (the size controls on the left, the check list and the Explain panel on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the page shows the first view with no click. The middle region: at its top a row of tabs starting "Size and torque" (the selected tab); under the tabs four tiles side by side, each a large number with its unit over a smaller, quieter label, the second labelled "Hot-low torque" with a line starting "needs" under its label; then a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft. The right side: the heading "CHECKS", a status line, then one row per check, each a small shape, the check's name and a word at the row's right end (OK, Check or Fails); a quiet line under the rows about fixed 3D fields; then the heading "EXPLAIN" and a paragraph starting "Two rings of magnets" with three numbered lines under it. details_toggle=true only if the header row at the top of the page (it may wrap to a second line) shows, after the "Equation panel" button, a checkbox labelled "Show spreadsheet details", unticked.
Then the design file picker: write this text, exactly, to the file .playwright-mcp/magcoupling-smoke-design.json under the repository root (gitignored; the Playwright MCP uploads only files under the repository) and note its absolute path: ${MAGCOUPLING_SMOKE_DESIGN_FILE}
```

Replace:

```js
design_file_loaded=true only if a console message contains "magcoupling: loaded a design file". If the button cannot be found or the overlay does not appear after two attempts (each with a fresh screenshot), design_file_loaded=false and say why in notes: the canvas click is the fragile part of this step, and the share-link checks do not depend on it.
Then a teaching note through a Why link (every click and mouse move below as above: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot): on the dashboard (the right side), under its first row (the nominal pull-out torque), find the line that starts "Why:" and click the link text right of "Why:". Wait 1 second and take a screenshot. why_link_opened=true only if the Equation panel is now open, docked at the bottom of the middle region, and shows a teaching note: a bold title, a few sentences, the heading "What to change" with lines that start with Raise, Lower, Choose or Enter, and a closed fold "Where this comes from".
Then an input tooltip: click the "Geometry" tab in the row of tabs at the top of the middle region and wait 1 second. On the left the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
```

with:

```js
design_file_loaded=true only if a console message contains "magcoupling: loaded a design file". If the button cannot be found or the overlay does not appear after two attempts (each with a fresh screenshot), design_file_loaded=false and say why in notes: the canvas click is the fragile part of this step, and the share-link checks do not depend on it.
Then a check's entry and its lesson link (every click and mouse move below as above: browser_run_code_unsafe with page.mouse at X, Y read off a fresh screenshot): in the check list (the right side) click the name of the row "Hot-low torque". Wait 1 second and take a screenshot. Under the check list the Explain panel now shows the check's entry: a small shape, "Hot-low torque" and its word, then the headings "What it is" and "Numbers behind this check" with the check's numbers under the second. If the line "Full lesson:" at the entry's foot is not in view, move the mouse over the Explain panel and scroll down with code: async (page) => { await page.mouse.move(X, Y); await page.mouse.wheel(0, 1500); await page.waitForTimeout(1000); }, then take a screenshot. check_entry=true only if the entry shows as described, the line "Full lesson:" followed by a link is in view, and the check list's rows above the Explain panel are still in view (the Explain panel scrolls on its own). Then click the link right of "Full lesson:". Wait 1 second and take a screenshot. lesson_link_opened=true only if the Equation panel is now open, docked at the bottom of the middle region, and shows a teaching note: a bold title, a few sentences, the heading "What to change" with lines that start with Raise, Lower, Choose or Enter, and a closed fold "Where this comes from".
Then an input tooltip: click the "Geometry" tab in the row of tabs at the top of the middle region and wait 1 second. On the left the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
```

Replace:

```js
Then an input tooltip: click the "Geometry" tab in the row of tabs at the top of the middle region and wait 1 second. On the left the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
Then the dashboard's scrolling: in a fresh screenshot move the mouse to the middle of the dashboard and scroll down with code: async (page) => { await page.mouse.move(X, Y); await page.mouse.wheel(0, 4000); await page.waitForTimeout(1000); }, then take a screenshot. dashboard_scrolls=true only if the dashboard's last row, "Magnets fit", and the "Why:" line under it are in view and its first row has scrolled out of view.
If a target of these last three checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes. Collect the console messages at level "info" (the Why link, the tooltip and the scroll ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; the dashboard's Why link opened the Equation panel, which closes the card; and the tooltip step left the Geometry tab selected, where no card is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
Wait 8 seconds (WASM init, then the sizing solve) and take a screenshot. torque_card=true only if the middle region shows, between the torque readouts and the drawing, the heading "${CARD_HEADING}" with a link starting "Why:" beside it, open, and under it, in this order: the line "The change in the model's hot-low torque after one step of one control alone.", the line "Auto sets the magnet length again after a change; each row keeps the length shown.", then four rows: one starting "Outside diameter:", the row "Overall length: Set by Auto; turn Auto off to edit", one starting "Magnet face gap:" and one starting "Magnets per ring:"; and the drawing still shows below the card. A row may wrap onto more lines.
Then the drag: in a fresh screenshot find the handle of the "Magnet face gap" slider among the size controls on the left (it reads ${SMOKE_GAP_SHOWN}; Auto on the magnet length leaves this slider free). Drag it and read the frame times in one call of browser_run_code_unsafe, code: async (page) => { const x = X, y = Y; let sizing = 0; const onConsole = m => { if (m.text().includes('magcoupling sizing: ')) sizing += 1; }; page.on('console', onConsole); await page.mouse.move(x, y); await page.evaluate(() => { window.__frameMs = []; }); await page.mouse.down(); for (let i = 0; i < 60; i++) { await page.mouse.move(i % 2 === 0 ? x + 3 : x - 3, y); await page.waitForTimeout(50); } await page.mouse.up(); const ms = await page.evaluate(() => window.__frameMs.slice()); page.off('console', onConsole); const s = [...ms].sort((a, b) => a - b); const rank = q => (s.length ? s[Math.ceil(q * s.length) - 1] : null); return { frames: s.length, median: rank(0.5), p95: rank(0.95), max: s.length ? s[s.length - 1] : null, sizing_lines: sizing }; } (X and Y the handle's centre in CSS pixels of the screenshot; the timer is emptied just before the press, so only the drag's frames count, not the load, the solve or the screenshots). Then take a screenshot. drag_moved=true only if the Magnet face gap now reads a value other than ${SMOKE_GAP_SHOWN} (the drag held the handle). From the returned object: drag_frames=frames, drag_frame_ms_median=median, drag_frame_ms_p95=p95 and drag_frame_ms_max=max (milliseconds per frame, nearest rank). drag_frames_within_60fps=true only if drag_moved, drag_frames >= ${MIN_DRAG_FRAMES} and drag_frame_ms_p95 <= ${FRAME_BUDGET_MS} (this script checks it again from your numbers); under ${MIN_DRAG_FRAMES} frames the timer saw the app draw too few frames: say so in notes. In notes give sizing_lines, the number of "magcoupling sizing: " console lines logged during the drag (expected 0: the moves come 50 ms apart, so the 0.25 s debounce never runs a solve mid-drag). If the handle cannot be found after two attempts (each with a fresh screenshot), set drag_moved=false, leave the four numbers out and say why in notes.
```

with:

```js
Then an input tooltip: click the "Geometry" tab in the row of tabs at the top of the middle region and wait 1 second. On the left the first group, Key design, lists inputs; in a fresh screenshot move the mouse onto the label of its first row with code: async (page) => { await page.mouse.move(X, Y); await page.waitForTimeout(1500); }, then take a screenshot. tooltip_plain=true only if a tooltip shows beside that label with at least one sentence about the input and a line that starts "Default:", and no line that is a dotted name such as "metal.face_gap_mm" and no spreadsheet cell such as "Metal design!C119" (the "Show spreadsheet details" toggle is off).
If a target of these last checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes; the check list's last row is "Magnets fit". tooltip_plain=true was judged last on this page. Collect the console messages at level "info" (the check's entry, the lesson link and the tooltip ran on this page).
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; and the tooltip step left the Geometry tab selected, where no tile is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
Wait 8 seconds (WASM init, then the sizing solve), take a screenshot and click the second tile, "Hot-low torque", in the middle region. Wait 1 second and take a screenshot; if the heading "${CARD_HEADING}" is not in view in the Explain panel (the right side, under the check list), scroll it down as above and take another. torque_card=true only if the Explain panel shows the tile's entry: its number and "Hot-low torque", the heading "What a change does", then the heading "${CARD_HEADING}" with a link starting "Why:" beside it, and under it, in this order: the line "The change in the model's hot-low torque after one step of one control alone.", the line "Auto sets the magnet length again after a change; each row keeps the length shown.", then four rows: one starting "Outside diameter:", the row "Overall length: Set by Auto; turn Auto off to edit", one starting "Magnet face gap:" and one starting "Magnets per ring:"; and the middle region still shows the four tiles over the drawing. A row may wrap onto more lines.
Then the drag: in a fresh screenshot find the handle of the "Magnet face gap" slider among the size controls on the left (it reads ${SMOKE_GAP_SHOWN}; Auto on the magnet length leaves this slider free). Drag it and read the frame times in one call of browser_run_code_unsafe, code: async (page) => { const x = X, y = Y; let sizing = 0; const onConsole = m => { if (m.text().includes('magcoupling sizing: ')) sizing += 1; }; page.on('console', onConsole); await page.mouse.move(x, y); await page.evaluate(() => { window.__frameMs = []; }); await page.mouse.down(); for (let i = 0; i < 60; i++) { await page.mouse.move(i % 2 === 0 ? x + 3 : x - 3, y); await page.waitForTimeout(50); } await page.mouse.up(); const ms = await page.evaluate(() => window.__frameMs.slice()); page.off('console', onConsole); const s = [...ms].sort((a, b) => a - b); const rank = q => (s.length ? s[Math.ceil(q * s.length) - 1] : null); return { frames: s.length, median: rank(0.5), p95: rank(0.95), max: s.length ? s[s.length - 1] : null, sizing_lines: sizing }; } (X and Y the handle's centre in CSS pixels of the screenshot; the timer is emptied just before the press, so only the drag's frames count, not the load, the solve or the screenshots). Then take a screenshot. drag_moved=true only if the Magnet face gap now reads a value other than ${SMOKE_GAP_SHOWN} (the drag held the handle). From the returned object: drag_frames=frames, drag_frame_ms_median=median, drag_frame_ms_p95=p95 and drag_frame_ms_max=max (milliseconds per frame, nearest rank). drag_frames_within_60fps=true only if drag_moved, drag_frames >= ${MIN_DRAG_FRAMES} and drag_frame_ms_p95 <= ${FRAME_BUDGET_MS} (this script checks it again from your numbers); under ${MIN_DRAG_FRAMES} frames the timer saw the app draw too few frames: say so in notes. In notes give sizing_lines, the number of "magcoupling sizing: " console lines logged during the drag (expected 0: the moves come 50 ms apart, so the 0.25 s debounce never runs a solve mid-drag). If the handle cannot be found after two attempts (each with a fresh screenshot), set drag_moved=false, leave the four numbers out and say why in notes.
```

Replace:

```js
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, why_link_opened, tooltip_plain, dashboard_scrolls, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
```

with:

```js
share_link_loaded=true only if a console message contains "magcoupling: loaded the design from the share link". sizing_solved=true only if a console message contains "magcoupling sizing: Solved at". explorer_ready=true only if a console message contains "magcoupling explorer: " (the equation registry, built when the page starts).
passed=true only if: page loaded, canvas present, starting_design, look, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, check_entry, lesson_link_opened, tooltip_plain, torque_card, address_follows, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
    { label: 'gui-smoke-magcoupling', phase: 'Smoke', schema: MAGCOUPLING_SCHEMA, model: 'sonnet' },
```

**`linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`**

Replace:

```js
const CHECKS = ['canvas_present', 'starting_design', 'look', 'first_view', 'details_toggle',
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'why_link_opened',
  'tooltip_plain', 'dashboard_scrolls', 'torque_card', 'address_follows']
// A calculator agent's reading of a good run; each case changes one thing.
```

with:

```js
const CHECKS = ['canvas_present', 'starting_design', 'look', 'first_view', 'details_toggle',
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'check_entry',
  'lesson_link_opened', 'tooltip_plain', 'torque_card', 'address_follows']
// A calculator agent's reading of a good run; each case changes one thing.
```

- [ ] **Step 4: Run the checks**, and the verdict test: `node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1`. Expected: `test result: ok. 1090 passed; 0 failed; 6 ignored`; clippy clean; the verdict test's last line `all 77 checks hold`.

- [ ] **Step 5: Commit** the three files: `test(magcoupler): gui-smoke reads the check list, a check's entry and the torque card in the Explain panel (BL-081 T4)`.

### Task 5: The panel draws the new columns

**Model:** `fable`.

The right `SidePanel` draws the check list over the Explain panel; the opening tab draws the tiles, the exploration lines and the drawing with its callout list in all the height left (D-19); the Size column opens with the Required torque row, scrolls, and draws the footnote under the Detailed inputs fold; Esc returns the orientation unless a text field or a popup has the key. `dashboard_ui` and its guide blocks, the welcome strip, the torque readouts and the collapsing card go, and the three modules lose their `allow`. The app's own tests draw the first view in its theme and font.

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/dashboard.rs`, `magcoupling-rs/src/gui/geometry_view.rs`, `magcoupling-rs/src/gui/guidance.rs`, `magcoupling-rs/src/gui/mod.rs`, `magcoupling-rs/src/gui/size_controls.rs`, `magcoupling-rs/src/gui/test_support.rs`, `magcoupling-rs/src/app.rs`

**Interfaces:**
- Consumes: Tasks 1 to 3.
- Produces: `test_support::{check_entry_texts, shown_whole, words_shown}`. Gone: `dashboard::dashboard_ui` and `PROBLEM`, `FIX`, `WHY_LINK`, `CAUTION_DETAILS`; `gui::WELCOME`; `geometry_view::drawing_floor`.

- [ ] **Step 1 (red): the tests of the new columns.** The blocks rewrite the panel's tests of the right column and the opening tab, move the dashboard's guide tests to the entry, and add the app's layout and word tests.

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust

// Drawn by the panel from Task 5 of this stage's plan, which removes this allow.
#[allow(dead_code)]
pub mod checks;
```

with:

```rust

pub mod checks;
```

Replace:

```rust
pub mod diagrams;
// Drawn by the panel from Task 5 of this stage's plan, which removes this allow.
#[allow(dead_code)]
pub mod explain;
```

with:

```rust
pub mod diagrams;
pub mod explain;
```

Replace:

```rust
pub(crate) mod test_support;
// Drawn by the panel from Task 5 of this stage's plan, which removes this allow.
#[allow(dead_code)]
pub mod tiles;
```

with:

```rust
pub(crate) mod test_support;
pub mod tiles;
```

Replace:

```rust
    CentreView, MagcouplingPanel, OPEN_EXAMPLE, PanelRequest, SHOW_DETAILS, SIZING_LOG_PREFIX,
    WELCOME,
};
```

with:

```rust
    CentreView, MagcouplingPanel, OPEN_EXAMPLE, PanelRequest, SHOW_DETAILS, SIZING_LOG_PREFIX,
};
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        };
        // The starting design: a dashboard row's title and the head of its Why line; the Size
        // tab's three readout labels ("Hot-low torque" is drawn as both). Then their values.
        let output = frame_output(&ctx, &mut app, Vec::new());
```

with:

```rust
        };
        // The starting design: the right column's two headings and two of the Size tab's
        // readout labels. Then their values.
        let output = frame_output(&ctx, &mut app, Vec::new());
```

Replace:

```rust
        let labels = [
            "Pull-out torque (nominal)",
            "Hot-low torque",
            "Why:",
            "Required torque",
```

with:

```rust
        let labels = [
            crate::gui::checks::HEADING,
            crate::gui::explain::HEADING,
            "Required torque",
```

Replace:

```rust
        all(&output, &["1.704 N·m", "1.448 N·m"], theme::TEXT);
        // A tab that is not the active one is quiet too; the active tab's name (drawn as the
```

with:

```rust
        all(&output, &["1.704 N·m", "1.448 N·m"], theme::TEXT);
        // A check's name is what its row says: the text colour.
        all(&output, &["Running clearance"], theme::TEXT);
        // A tab that is not the active one is quiet too; the active tab's name (drawn as the
```

Replace:

```rust
        all(&output, &["Size and torque"], theme::TEXT);
        // The failing example design: the heads of an open block are captions too, and the
        // Magnets fit row's title is a label (the dashboard scrolled to its foot).
        app.panel.open_example_design();
```

with:

```rust
        all(&output, &["Size and torque"], theme::TEXT);
        // The failing example design with its Running clearance check selected: the entry's
        // headings are captions, and under the last a row's label is muted and its value is
        // not.
        use crate::gui::explain::{NUMBERS, WHAT_IT_IS, WHAT_SETS_IT};
        use crate::gui::explorer::WHAT_TO_CHANGE;
        use crate::gui::test_support::{primary_button, text_center};
        app.panel.open_example_design();
```

Replace:

```rust
        frame_texts(&ctx, &mut app);
        let output = frame_output(&ctx, &mut app, Vec::new());
        all(&output, &["Problem:", "Fix:", "Why:"], theme::MUTED);
        all(&output, &["2.688 N·m"], theme::TEXT);
    }
```

with:

```rust
        frame_texts(&ctx, &mut app);
        let laptop = WINDOWS[0];
        let output = frame_on(&ctx, &mut app, laptop, Vec::new());
        let row = text_center(&output, "Running clearance");
        frame_on(&ctx, &mut app, laptop, vec![egui::Event::PointerMoved(row)]);
        for pressed in [true, false] {
            frame_on(&ctx, &mut app, laptop, vec![primary_button(row, pressed)]);
        }
        frame_on(&ctx, &mut app, laptop, vec![egui::Event::PointerGone]);
        let output = frame_on(&ctx, &mut app, laptop, Vec::new());
        let headings = [WHAT_IT_IS, WHAT_SETS_IT, WHAT_TO_CHANGE, NUMBERS];
        all(&output, &headings, theme::MUTED);
        const CLEARANCE: &str = "metal.min_running_clearance_mm";
        let label = crate::gui::plain_text::label_for(CLEARANCE);
        all(&output, &[label], theme::MUTED);
        let results = compute_all(&DesignInputs::default());
        let lines = crate::gui::dashboard::shown_lines(&results, crate::engine::clamps::PAUSED);
        let line = lines.iter().find(|line| line.path == CLEARANCE);
        let value = line.expect("the clearance has a row").shown_value();
        all(&output, &[value.as_str()], theme::TEXT);
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
    fn the_first_view_keeps_its_three_columns_its_tiles_and_the_fixed_3d_line_in_the_app_s_own_font()
     {
        // UX overhaul spec 8.2, test 15, and the stage 1 review's C-6: in the app's theme and
        // font, at the smallest window supported, the laptop screen and a desktop one. The four
        // tile numbers are drawn whole, between the Size column and the check list; the drawing
        // keeps its least height; the check list shows the line that says two checks read
        // stored 3D fields (register row A-18; spec 2.3: it stays visible), and the Explain
        // panel its orientation. The line is found by its constant, whatever its words.
        use crate::gui::dashboard::STORED_3D_LABEL;
        use crate::gui::explain::ORIENTATION;
        use crate::gui::geometry_view::MIN_DRAWING_HEIGHT;
        use crate::gui::test_support::{flat_shapes, shown_whole};
        use crate::gui::tiles::Tile;
        // The starting design, then the failing example design (its longer entries and
        // lines).
        for example in [false, true] {
            let inputs = if example {
                DesignInputs::default()
            } else {
                starting_design()
            };
            let results = compute_all(&inputs);
            for size in [WINDOWS[1], WINDOWS[0], egui::vec2(1920.0, 1080.0)] {
                let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
                let ctx = egui::Context::default();
                let mut app = MagcouplingApp::new(&ctx);
                if example {
                    app.panel.open_example_design();
                }
                for _ in 0..3 {
                    frame_on(&ctx, &mut app, size, Vec::new());
                }
                let output = frame_on(&ctx, &mut app, size, Vec::new());
                let whole = |text: &str| {
                    let rects = shown_whole(&output, text, screen);
                    assert_eq!(
                        rects.len(),
                        1,
                        "{size:?}: {text:?} drawn whole once: {rects:?}"
                    );
                    rects[0]
                };
                whole(STORED_3D_LABEL);
                whole(ORIENTATION);
                let size_column = whole("Required torque");
                // The fold of the detailed inputs, the way to the other inputs, on the first view
                // where the window is at least the laptop's height (plan review E-4).
                if size.y >= WINDOWS[0].y {
                    whole("Detailed inputs");
                }
                let checks = whole(crate::gui::checks::HEADING);
                let numbers: Vec<egui::Rect> = Tile::ALL
                    .iter()
                    .map(|tile| whole(&tile.number(&inputs, &results).unwrap()))
                    .collect();
                for number in &numbers {
                    assert!(
                        size_column.right() < number.left() && number.right() < checks.left(),
                        "{size:?}: a tile's number {number:?} between {size_column:?} and {checks:?}"
                    );
                }
                // The four tiles take one row where the centre holds four of the narrowest, else
                // two rows of two.
                let tops: std::collections::BTreeSet<i32> =
                    numbers.iter().map(|n| n.top().round() as i32).collect();
                assert!(tops.len() <= 2, "{size:?}: {numbers:?}");
                // The drawing: the largest plate in the background colour.
                let drawing = flat_shapes(&output)
                    .into_iter()
                    .filter_map(|shape| match shape {
                        egui::Shape::Rect(r) if r.fill == theme::BG => Some(r.rect),
                        _ => None,
                    })
                    .max_by(|a, b| a.area().total_cmp(&b.area()))
                    .expect("the drawing's plate");
                assert!(
                    drawing.height() >= MIN_DRAWING_HEIGHT && drawing.bottom() <= size.y,
                    "{size:?}: {drawing:?}"
                );
                assert!(
                    numbers.iter().all(|n| n.bottom() < drawing.top()),
                    "{size:?}"
                );
            }
        }
    }

    #[test]
    fn the_first_view_draws_at_most_four_hundred_and_fifty_words_at_1280_by_800() {
        // UX overhaul spec 2.1 (goal G-1) and 8.2, test 8, measured as the inventory did: the
        // words the opening screen shows. 608 before this stage (the dashboard's rows, the
        // torque readouts and the welcome strip); 443 after it, the callout list under the
        // drawing kept until its tags are selections (decision D-19). The design's 250 needs
        // the top bar, the Size column and the callout list of the stage's second part, which
        // lowers this ceiling.
        use crate::gui::test_support::words_shown;
        let size = WINDOWS[0];
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        for _ in 0..3 {
            frame_on(&ctx, &mut app, size, Vec::new());
        }
        let output = frame_on(&ctx, &mut app, size, Vec::new());
        let words = words_shown(&output, egui::Rect::from_min_size(egui::Pos2::ZERO, size));
        assert!(words <= 450, "{words} words on the first view");
    }

    #[test]
    fn the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font() {
```

Replace:

```rust
        assert_eq!(count("1.35 N·m"), 1, "{texts:?}");
        assert_eq!(count("margin +0.10 N·m (+7.3 %)"), 1, "{texts:?}");
        // The header takes a second row at this width; the torque card still opens.
        assert_eq!(count(crate::gui::torque_card::CAPTION), 1, "{texts:?}");
        // An edit, then Reset all: the starting design again, not the spreadsheet's.
```

with:

```rust
        assert_eq!(count("1.35 N·m"), 1, "{texts:?}");
        // The gap breaks only between its torque and its percent.
        assert_eq!(count("margin +0.10 N·m"), 1, "{texts:?}");
        assert_eq!(count("(+7.3 %)"), 1, "{texts:?}");
        // The Hot-low tile says what it must reach, and the Explain panel is on its
        // orientation: the torque card is one click away, on that tile.
        assert_eq!(count(crate::gui::tiles::NEEDS), 1, "{texts:?}");
        assert_eq!(count("1.350 N·m"), 1, "{texts:?}");
        assert_eq!(count(crate::gui::explain::ORIENTATION), 1, "{texts:?}");
        assert_eq!(count(crate::gui::torque_card::CAPTION), 0, "{texts:?}");
        // An edit, then Reset all: the starting design again, not the spreadsheet's.
```

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
    use crate::gui::corrections::MARKER_HEADING;
    use crate::gui::guidance::{
        CHECK_GUIDES, Direction, Fixes, ROW_LEARN, all_fixes, design_status_text, trade_offs,
    };
    use crate::gui::readouts::ReadoutEvents;
    use crate::gui::test_support::{
```

with:

```rust
    use crate::gui::corrections::MARKER_HEADING;
    use crate::gui::guidance::{self, CHECK_GUIDES, FIT_CHECK};
    use crate::gui::test_support::{
```

Replace:

```rust
    use crate::gui::test_support::{
        assert_glyphs, drawn_texts, hover_frames, hub_overstressed, only_the_clamp_fails,
        only_the_gearbox_fails, only_the_joint_fails, passing_design, primary_button,
        reversed_hub_band, short_magnets, sized_frame, text_center, text_color, text_rect,
        tooltips_at_once, with_poles,
    };
```

with:

```rust
    use crate::gui::test_support::{
        SCREEN, hub_overstressed, only_the_clamp_fails, only_the_joint_fails, passing_design,
        reversed_hub_band, short_magnets, sized_frame, text_color, with_poles,
    };
```

Replace:

```rust
    #[test]
    fn the_stored_3d_label_is_drawn_once_under_the_shaft_joints_row() {
        // Owner R-15: one label, after the last stored-3D row, now the shaft joints row: below
        // its label and above the gearbox row's, whether the clamp section shows or not.
        let ctx = egui::Context::default();
        for paused in [true, false] {
            let output = dashboard_frame_with(&ctx, &DesignInputs::default(), paused);
            let texts = drawn_texts(&output);
            let count = texts.iter().filter(|t| *t == STORED_3D_LABEL).count();
            assert_eq!(count, 1, "paused {paused}");
            let y = |text: &str| {
                text_rect(&output, text)
                    .unwrap_or_else(|| panic!("{text:?} drawn"))
                    .center()
                    .y
            };
            let label = y(STORED_3D_LABEL);
            assert!(label > y(label_for("shaft_joint.check")), "paused {paused}");
            assert!(
                label < y(label_for("gearbox.cold_high_check")),
                "paused {paused}"
            );
        }
    }

    #[test]
    fn out_of_range_end_effect_greys_the_pull_out_rows_without_badges() {
```

with:

```rust
    #[test]
    fn out_of_range_end_effect_greys_the_pull_out_rows_without_badges() {
```

Replace:

```rust
    #[test]
    fn the_correction_ids_and_their_legend_show_only_with_the_spreadsheet_details() {
        use crate::gui::typeset::TermColors;
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let fixes = all_fixes(&inputs, &results, false);
        let markers: Vec<String> = dashboard_lines(&results)
            .into_iter()
            .map(|line| line.marker)
            .filter(|marker| !marker.is_empty())
            .collect();
        assert!(!markers.is_empty(), "the defaults carry correction ids");
        let status = design_status_text(&design_status(&inputs, &results, false));
        for show_details in [false, true] {
            let ctx = egui::Context::default();
            let output = sized_frame(&ctx, egui::vec2(1280.0, 4000.0), Vec::new(), |ui| {
                let mut readouts = Readouts::new(TermColors::default(), show_details);
                dashboard_ui(
                    ui,
                    &inputs,
                    &DesignInputs::default(),
                    &results,
                    &fixes,
                    false,
                    &mut readouts,
                );
            });
            let texts = drawn_texts(&output);
            for marker in &markers {
                assert_eq!(texts.contains(marker), show_details, "{marker}");
            }
            let legends = texts.iter().filter(|t| *t == MARKER_LEGEND).count();
            assert_eq!(legends, usize::from(show_details));
            // The status line still tops the dashboard.
            assert_eq!(texts.first(), Some(&status));
        }
    }

    #[test]
    fn the_result_tooltip_names_the_path_cell_and_notes() {
```

with:

```rust
    #[test]
    fn the_result_tooltip_names_the_path_cell_and_notes() {
```

Replace:

```rust
    #[test]
    fn each_row_is_drawn_under_its_display_label_and_exported_under_its_catalogue_label() {
        let ctx = egui::Context::default();
        let texts = drawn_texts(&dashboard_frame(&ctx, &DesignInputs::default()));
        let lines = dashboard_lines(&compute_all(&DesignInputs::default()));
```

with:

```rust
    #[test]
    fn each_row_is_exported_under_its_catalogue_label_and_four_have_a_display_label() {
        // The Explain panel draws a check's rows under their display labels
        // (explain.rs: the_numbers_behind_a_check_are_the_dashboard_rows_its_verdict_badges).
        let lines = dashboard_lines(&compute_all(&DesignInputs::default()));
```

Replace:

```rust
        for line in &lines {
            assert!(
                texts.iter().any(|t| t == label_for(line.path)),
                "{}",
                line.path
            );
            // The line keeps the catalogue label: the spreadsheet's Summary writes it.
```

with:

```rust
        for line in &lines {
            // The line keeps the catalogue label: the spreadsheet's Summary writes it.
```

Replace:

```rust
        }
        // The overlay renames four rows (spec 3.5's three and BL-059's ripple row); their
        // catalogue labels are not drawn.
        let renamed: Vec<&str> = lines
```

with:

```rust
        }
        // The overlay renames four rows (spec 3.5's three and BL-059's ripple row).
        let renamed: Vec<&str> = lines
```

Replace:

```rust
        );
        for line in lines.iter().filter(|line| renamed.contains(&line.path)) {
            assert!(!texts.iter().any(|t| t == line.label), "{}", line.path);
        }
    }
```

with:

```rust
        );
    }
```

Replace:

```rust
        // UX overhaul spec 6.1: labels and captions are quieter than values
        // (`look::quiet_text`). The label, the value and the head of the Why line of the first
        // row, at the example design.
        let colours = |ctx: &egui::Context| {
```

with:

```rust
        // UX overhaul spec 6.1: labels and captions are quieter than values
        // (`look::quiet_text`). The label and the value of the pull-out's row, as the Explain
        // panel draws it under "Numbers behind this check", at the example design.
        let results = compute_all(&DesignInputs::default());
        let lines = shown_lines(&results, false);
        let line = lines
            .iter()
            .find(|line| line.path == "model.pullout_Nm")
            .expect("the pull-out has a row");
        let colours = |ctx: &egui::Context| {
```

Replace:

```rust
        let colours = |ctx: &egui::Context| {
            let output = dashboard_frame(ctx, &DesignInputs::default());
            [label_for("model.pullout_Nm"), "2.688 N·m", WHY_LINK].map(|text| {
                text_color(&output, text).unwrap_or_else(|| panic!("no {text:?} drawn"))
```

with:

```rust
        let colours = |ctx: &egui::Context| {
            let output = sized_frame(ctx, SCREEN, Vec::new(), |ui| {
                line_ui(ui, line, &mut Readouts::default());
            });
            [label_for("model.pullout_Nm"), "2.688 N·m"].map(|text| {
                text_color(&output, text).unwrap_or_else(|| panic!("no {text:?} drawn"))
```

Replace:

```rust
        };
        // A theme that sets a weak colour (the standalone app's): the label and the head take
        // it, the value keeps the strong colour.
        let themed = egui::Context::default();
```

with:

```rust
        };
        // A theme that sets a weak colour (the standalone app's): the label takes it, the value
        // keeps the strong colour.
        let themed = egui::Context::default();
```

Replace:

```rust
        let strong = themed.style().visuals.strong_text_color();
        assert_eq!(colours(&themed), [weak, strong, weak]);
        // egui's own visuals set none: their weak colour is too faint for a name (the text
```

with:

```rust
        let strong = themed.style().visuals.strong_text_color();
        assert_eq!(colours(&themed), [weak, strong]);
        // egui's own visuals set none: their weak colour is too faint for a name (the text
```

Replace:

```rust
            colours(&plain),
            [
                visuals.text_color(),
                visuals.strong_text_color(),
                visuals.text_color()
            ]
        );
```

with:

```rust
            colours(&plain),
            [visuals.text_color(), visuals.strong_text_color()]
        );
```

Replace:

```rust

    /// The defaults under a 30 mm overall-length claim: the hot-low torque needs longer
    /// magnets and the space claim shorter ones, both failing (a trade-off note in each).
    fn length_conflict() -> DesignInputs {
        design(|i| i.metal.max_overall_axial_mm = 30.0)
    }

    /// One frame of the dashboard of `inputs` at full width on a screen tall enough for every
    /// row: what it drew.
    fn dashboard_frame(ctx: &egui::Context, inputs: &DesignInputs) -> egui::FullOutput {
        dashboard_frame_with(ctx, inputs, false)
    }

    /// [`dashboard_frame`] with the shaft-clamp section paused (`clamps_paused`) or not.
    fn dashboard_frame_with(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        clamps_paused: bool,
    ) -> egui::FullOutput {
        let results = compute_all(inputs);
        let fixes = all_fixes(inputs, &results, clamps_paused);
        sized_frame(ctx, egui::vec2(1280.0, 4000.0), Vec::new(), |ui| {
            dashboard_ui(
                ui,
                inputs,
                &DesignInputs::default(),
                &results,
                &fixes,
                clamps_paused,
                &mut Readouts::default(),
            );
        })
    }

    #[test]
    fn while_the_clamp_section_is_paused_its_row_and_guide_leave_the_dashboard() {
        // Ruling 11: a design that fails only the clamp reads as passing while it is paused;
        // with the section shown its row, its Problem and its Why sentence are drawn.
        let ctx = egui::Context::default();
        let inputs = only_the_clamp_fails();
        let results = compute_all(&inputs);
        let clamp = guidance::guide("clamps.recommended").expect("the clamp's guide");
        let problem = (clamp.problem)(&inputs, &results).expect("the clamp fails");
        for paused in [false, true] {
            let output = dashboard_frame_with(&ctx, &inputs, paused);
            let texts = drawn_texts(&output);
            let count = |text: &str| texts.iter().filter(|t| *t == text).count();
            let status = design_status_text(&design_status(&inputs, &results, paused));
            assert_eq!(texts.first(), Some(&status), "paused {paused}");
            let shown = usize::from(!paused);
            assert_eq!(
                count(label_for("clamps.recommended")),
                shown,
                "paused {paused}"
            );
            assert_eq!(count(&problem), shown, "paused {paused}");
            assert_eq!(count(clamp.why), shown, "paused {paused}");
        }
        assert_eq!(
            design_status_text(&design_status(&inputs, &results, true)),
            "Your design: passes all 9 checks"
        );
    }

    #[test]
```

with:

```rust

    #[test]
```

Replace:

```rust

    /// The text of the Why link to `learn`, as a reader would name it: a reviewed note's title,
    /// an assumption's label, else "no note yet".
    fn learn_link_text(learn: Learn) -> String {
        match learn {
            Learn::Note(id) => {
                reviewed_note(id).map_or(NO_NOTE_YET.to_owned(), |n| glyph_safe(n.title))
            }
            Learn::Assumption(id) => ASSUMPTIONS
                .iter()
                .find(|a| a.id == id)
                .unwrap()
                .label
                .to_owned(),
            Learn::None(_) => NO_NOTE_YET.to_owned(),
        }
    }

    #[test]
```

with:

```rust

    #[test]
```

Replace:

```rust
    #[test]
    fn a_red_check_shows_its_block_an_amber_one_its_header_and_every_row_its_why_link() {
        let count = |texts: &[String], text: &str| texts.iter().filter(|t| *t == text).count();
        // Every fix text any guide can show: the static lists, and the fixes of the designs.
        let mut fix_texts: BTreeSet<String> = CHECK_GUIDES
            .iter()
            .filter_map(|g| match g.fixes {
                Fixes::Static(list) => Some(list),
                Fixes::PerTerm(_) => None,
            })
            .flatten()
            .map(|&(path, dir)| fix_text(path, dir))
            .collect();
        let ctx = egui::Context::default();
        for inputs in [
            DesignInputs::default(),
            short_magnets(),
            passing_design(),
            hot_day(),
            too_many_poles(),
            length_conflict(),
            design(|i| i.metal.max_diameter_mm = f64::NAN),
        ] {
            let results = compute_all(&inputs);
            let fixes = all_fixes(&inputs, &results, false);
            fix_texts.extend(
                fixes
                    .iter()
                    .flatten()
                    .map(|&(path, dir)| fix_text(path, dir)),
            );
            let texts = drawn_texts(&dashboard_frame(&ctx, &inputs));
            let level = |g: &CheckGuide| guidance::level(g, &inputs, &results);
            // Red blocks are open; amber ones are shut under their header.
            let open: Vec<usize> = (0..CHECK_GUIDES.len())
                .filter(|&i| level(&CHECK_GUIDES[i]) == Level::Bad)
                .collect();
            let amber = CHECK_GUIDES
                .iter()
                .filter(|g| level(g) == Level::Caution)
                .count();
            assert_eq!(count(&texts, CAUTION_DETAILS), amber);
            let problems: Vec<String> = open
                .iter()
                .filter_map(|&i| (CHECK_GUIDES[i].problem)(&inputs, &results))
                .collect();
            assert_eq!(count(&texts, PROBLEM), problems.len());
            for problem in &problems {
                assert_eq!(count(&texts, problem), 1, "{problem:?}");
            }
            for (i, guide) in CHECK_GUIDES.iter().enumerate() {
                assert_eq!(
                    count(&texts, guide.why),
                    usize::from(open.contains(&i)),
                    "{}",
                    guide.check
                );
            }
            // Each open block has its Fix line: its fixes, each once, or the no-fix line; then
            // its notes (trade-offs and side effects).
            assert_eq!(count(&texts, FIX), open.len());
            let mut want: Vec<String> = open
                .iter()
                .flat_map(|&i| fixes[i].iter().map(|&(path, dir)| fix_text(path, dir)))
                .collect();
            let mut drawn: Vec<String> = texts
                .iter()
                .filter(|t| fix_texts.contains(*t))
                .cloned()
                .collect();
            want.sort();
            drawn.sort();
            assert_eq!(drawn, want);
            let no_fix = open.iter().filter(|&&i| fixes[i].is_empty()).count();
            assert_eq!(count(&texts, guidance::NO_FIX), no_fix);
            let notes: Vec<String> = open
                .iter()
                .flat_map(|&i| guide_notes(&fixes, i, &inputs, &results))
                .collect();
            for note in &notes {
                assert_eq!(
                    count(&texts, note),
                    notes.iter().filter(|n| *n == note).count()
                );
            }
            // Every row has its Why line, the fit row's included, each leading where
            // `row_learn` (the fit row: its guide) says.
            assert_eq!(count(&texts, WHY_LINK), DASHBOARD.len() + 1);
            let fit = guidance::guide(FIT_CHECK).unwrap().learn;
            let links: Vec<String> = ROW_LEARN
                .iter()
                .map(|&(_, learn)| learn)
                .chain([fit])
                .map(learn_link_text)
                .collect();
            for link in &links {
                assert_eq!(
                    count(&texts, link),
                    links.iter().filter(|l| *l == link).count(),
                    "{link:?}"
                );
            }
        }
        // The workbook example's four red checks are the ones with blocks.
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        let red: BTreeSet<&str> = CHECK_GUIDES
            .iter()
            .filter(|g| guidance::level(g, &defaults, &results) == Level::Bad)
            .map(|g| g.check)
            .collect();
        let want = [
            "metal.hot_min_check",
            "metal.clearance_check",
            "materials.cup_wall_check",
            "gearbox.cold_high_check",
        ];
        assert_eq!(red, BTreeSet::from(want));
        // The designs above reach a trade-off note and an amber header.
        let conflict = length_conflict();
        let fixes = all_fixes(&conflict, &compute_all(&conflict), false);
        assert!((0..CHECK_GUIDES.len()).any(|i| !trade_offs(&fixes, i).is_empty()));
        let texts = drawn_texts(&dashboard_frame(&ctx, &hot_day()));
        assert_eq!(count(&texts, CAUTION_DETAILS), 1);
    }

    #[test]
    fn the_example_design_s_status_line_says_on_hover_where_its_values_come_from() {
        // Spec S2-5: "Example design" at the defaults, its hover the original spreadsheet;
        // "Your design" has no hover.
        for (inputs, example) in [(DesignInputs::default(), true), (passing_design(), false)] {
            let results = compute_all(&inputs);
            let fixes = all_fixes(&inputs, &results, false);
            let text = design_status_text(&design_status(&inputs, &results, false));
            assert_eq!(
                text.starts_with(guidance::EXAMPLE_DESIGN),
                example,
                "{text}"
            );
            let (_, output) = hover_frames(
                &egui::Context::default(),
                egui::vec2(1280.0, 4000.0),
                |output| text_center(output, &text),
                |ui| {
                    dashboard_ui(
                        ui,
                        &inputs,
                        &DesignInputs::default(),
                        &results,
                        &fixes,
                        false,
                        &mut Readouts::default(),
                    )
                },
            );
            let hover = guidance::EXAMPLE_DESIGN_HOVER.to_owned();
            assert_eq!(drawn_texts(&output).contains(&hover), example, "{text}");
        }
    }

    #[test]
    fn the_starting_design_s_status_line_says_on_hover_what_it_is() {
        // UX overhaul spec 7.1: at the session's baseline, when that is not the spreadsheet's
        // design, the line reads "Starting design" and its hover says what the design is; the
        // same design on a panel whose baseline is the spreadsheet's is "Your design", with no
        // hover.
        let inputs = passing_design();
        let results = compute_all(&inputs);
        let fixes = all_fixes(&inputs, &results, false);
        for (baseline, name, starting) in [
            (passing_design(), guidance::STARTING_DESIGN, true),
            (DesignInputs::default(), guidance::YOUR_DESIGN, false),
        ] {
            let text = format!("{name}: passes all 10 checks");
            let (_, output) = hover_frames(
                &egui::Context::default(),
                egui::vec2(1280.0, 4000.0),
                |output| text_center(output, &text),
                |ui| {
                    dashboard_ui(
                        ui,
                        &inputs,
                        &baseline,
                        &results,
                        &fixes,
                        false,
                        &mut Readouts::default(),
                    )
                },
            );
            let texts = drawn_texts(&output);
            let hover = guidance::STARTING_DESIGN_HOVER.to_owned();
            assert_eq!(texts.contains(&hover), starting, "{text}");
            let example = guidance::EXAMPLE_DESIGN_HOVER.to_owned();
            assert!(!texts.contains(&example), "{text}");
        }
    }

    #[test]
    fn the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour() {
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let mut passing_hot_day = passing_design();
        passing_hot_day.temperature.duty = hot_day().temperature.duty;
        let mut passing_reversed = passing_design();
        reversed_hub_band(&mut passing_reversed);
        let cases = [
            // The workbook example: four red checks.
            (DesignInputs::default(), 4, 0, Level::Bad),
            // One fixed: the cup wall at its rule's 2.0 mm and up.
            (
                design(|i| i.metal.cup_wall_corner_mm = 2.4),
                3,
                0,
                Level::Bad,
            ),
            (passing_design(), 0, 0, Level::Good),
            // The passing design with the example's gearbox rating: the gearbox check alone.
            (only_the_gearbox_fails(), 1, 0, Level::Bad),
            // The passing design with the inner hub's minimum at 8 µm: the shaft joints alone.
            (only_the_joint_fails(), 1, 0, Level::Bad),
            // The passing design with the inner hub's band reversed: the shaft joints ask for a
            // look.
            (passing_reversed, 0, 1, Level::Caution),
            // The hot day's temperature verdict is amber, counted apart from the red.
            (hot_day(), 4, 1, Level::Bad),
            // The passing design on the hot day: amber alone.
            (passing_hot_day, 0, 1, Level::Caution),
        ];
        for (inputs, bad, caution, colour) in cases {
            let results = compute_all(&inputs);
            let status = design_status(&inputs, &results, false);
            assert_eq!((status.bad, status.caution), (bad, caution));
            assert_eq!(status.total, CHECK_GUIDES.len());
            assert_eq!(status.is_default, inputs == DesignInputs::default());
            let output = dashboard_frame(&ctx, &inputs);
            let text = design_status_text(&status);
            assert_eq!(
                drawn_texts(&output).first(),
                Some(&text),
                "the status line tops the dashboard"
            );
            assert_eq!(text_color(&output, &text), Some(colour.color(&visuals)));
        }
    }

    #[test]
    fn the_fit_row_says_whether_the_blocks_fit_and_guides_a_misfit() {
        let ctx = egui::Context::default();
        let fit = guidance::guide(FIT_CHECK).unwrap();
        let count = |output: &egui::FullOutput, text: &str| {
            drawn_texts(output).iter().filter(|t| *t == text).count()
        };
        let output = dashboard_frame(&ctx, &DesignInputs::default());
        assert_eq!(count(&output, fit.label), 1);
        assert_eq!((count(&output, FITS), count(&output, DOES_NOT_FIT)), (1, 0));
        assert_eq!(count(&output, fit.why), 0);
        // 40 poles: the blocks do not fit; a red badge, the problem and the guide's fixes.
        let inputs = too_many_poles();
        let results = compute_all(&inputs);
        assert!(!crate::engine::model::blocks_fit(
            &inputs.coupling,
            &results.model
        ));
        assert_eq!(guidance::level(fit, &inputs, &results), Level::Bad);
        let output = dashboard_frame(&ctx, &inputs);
        assert_eq!((count(&output, FITS), count(&output, DOES_NOT_FIT)), (0, 1));
        assert_eq!(count(&output, fit.why), 1);
        let problem = (fit.problem)(&inputs, &results).expect("a misfit has a Problem");
        assert_eq!(count(&output, &problem), 1);
        // The last badge drawn is the fit row's, red.
        let visuals = ctx.style().visuals.clone();
        let last_badge = crate::gui::test_support::flat_shapes(&output)
            .into_iter()
            .rev()
            .find_map(|shape| match shape {
                egui::Shape::Circle(c) if c.radius == 5.0 => Some(c.fill),
                _ => None,
            });
        assert_eq!(last_badge, Some(Level::Bad.color(&visuals)));
    }

    /// The cup wall fix the test guide offers.
    const CUP_WALL_FIX: [(&str, Direction); 1] = [("metal.cup_wall_corner_mm", Direction::Raise)];

    /// A guide of the cup wall check, the test's own, linking to `learn`.
    fn guide_with(learn: Learn) -> CheckGuide {
        CheckGuide {
            check: "materials.cup_wall_check",
            label: "Cup wall",
            problem: |_, results| Some(results.materials.cup_wall_check.clone()),
            why: "A test sentence.",
            fixes: Fixes::Static(&CUP_WALL_FIX),
            learn,
        }
    }

    /// A test guide's lines at `level`, with the cup wall fix and no trade-off note.
    fn shown_at(level: Level) -> GuideShown<'static> {
        GuideShown {
            level,
            fixes: &CUP_WALL_FIX,
            notes: &Vec::new,
        }
    }

    /// Frames of `guide` drawn as `block` on the defaults in `ctx`, a first one with no input,
    /// then one per entry of `events`: what the last drew and what the readouts heard in it.
    fn guide_frames(
        ctx: &egui::Context,
        guide: &CheckGuide,
        block: &GuideShown,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, ReadoutEvents) {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let mut last = None;
        for input in std::iter::once(Vec::new()).chain(events) {
            let mut readouts = Readouts::default();
            let output = sized_frame(ctx, egui::vec2(600.0, 400.0), input, |ui| {
                guide_ui(ui, guide, block, &inputs, &results, &mut readouts);
            });
            last = Some((output, readouts.finish()));
        }
        last.expect("one frame at least")
    }

    /// Where `guide` drawn as `block` draws `text` on the defaults in `ctx`.
    fn guide_text_at(
        ctx: &egui::Context,
        guide: &CheckGuide,
        block: &GuideShown,
        text: &str,
    ) -> egui::Pos2 {
        let (output, _) = guide_frames(ctx, guide, block, Vec::new());
        text_rect(&output, text)
            .unwrap_or_else(|| panic!("no {text:?} in {:?}", drawn_texts(&output)))
            .center()
    }

    /// The frames of a click at `at` (move, press, release).
    fn click_at(at: egui::Pos2) -> Vec<Vec<egui::Event>> {
        vec![
            vec![egui::Event::PointerMoved(at)],
            vec![primary_button(at, true)],
            vec![primary_button(at, false)],
        ]
    }

    /// A click on `text` drawn by `guide` as `block`: what the readouts heard.
    fn click_in_guide(guide: &CheckGuide, block: &GuideShown, text: &str) -> ReadoutEvents {
        let ctx = egui::Context::default();
        let at = guide_text_at(&ctx, guide, block, text);
        guide_frames(&ctx, guide, block, click_at(at)).1
    }

    /// The texts drawn while `text` of `guide` as `block` is hovered, tooltips shown at once
    /// (no delay, the pointer need not rest): its hover text among them.
    fn hover_in_guide(guide: &CheckGuide, block: &GuideShown, text: &str) -> Vec<String> {
        let ctx = egui::Context::default();
        tooltips_at_once(&ctx);
        let at = guide_text_at(&ctx, guide, block, text);
        let hover = vec![vec![egui::Event::PointerMoved(at)], Vec::new()];
        drawn_texts(&guide_frames(&ctx, guide, block, hover).0)
    }

    #[test]
    fn a_why_link_asks_for_its_note_or_its_assumption_s_row_and_a_fix_for_its_input_s_row() {
        let green = shown_at(Level::Good);
        // A reviewed note: its title (glyph-safe) asks for the note.
        let note = reviewed_note("end_effect").expect("a reviewed note");
        let events = click_in_guide(
            &guide_with(Learn::Note(note.id)),
            &green,
            &glyph_safe(note.title),
        );
        assert_eq!((events.note, events.focus), (Some(note.id), None));
        // An assumption: its label asks for its first input's row; its hover text is the
        // rationale.
        let assumption = ASSUMPTIONS
            .iter()
            .find(|a| a.id == "production_variation")
            .unwrap();
        let guide = guide_with(Learn::Assumption(assumption.id));
        let events = click_in_guide(&guide, &green, assumption.label);
        assert_eq!(
            (events.note, events.focus),
            (None, Some(assumption.paths[0]))
        );
        let texts = hover_in_guide(&guide, &green, assumption.label);
        assert!(texts.iter().any(|t| t == assumption.rationale));
        // No note, or a note that has not passed review (no note has this id): "no note yet",
        // which asks for nothing; its hover text says why.
        for (learn, reason) in [
            (Learn::None("A test reason."), "A test reason."),
            (Learn::Note("no_such_note"), NOTE_NOT_REVIEWED),
        ] {
            let guide = guide_with(learn);
            assert_eq!(
                click_in_guide(&guide, &green, NO_NOTE_YET),
                ReadoutEvents::default()
            );
            assert!(hover_in_guide(&guide, &green, NO_NOTE_YET).contains(&reason.to_owned()));
        }
        // A red row's fix asks for its input's row; a two-pick magnet fix for its part's row.
        let guide = guide_with(Learn::None("A test reason."));
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let events = click_in_guide(&guide, &shown_at(Level::Bad), &fix);
        assert_eq!(
            (events.note, events.focus),
            (None, Some("metal.cup_wall_corner_mm"))
        );
        let part = [("coupling.magnets.part_inner", Direction::CustomHotterGrade)];
        let two_picks = GuideShown {
            level: Level::Bad,
            fixes: &part,
            notes: &Vec::new,
        };
        let events = click_in_guide(&guide, &two_picks, &fix_text(part[0].0, part[0].1));
        assert_eq!(events.focus, Some("coupling.magnets.part_inner"));
    }

    #[test]
    fn the_block_is_open_under_red_shut_under_amber_until_asked_and_absent_under_green() {
        let guide = guide_with(Learn::None("A test reason."));
        let problem = compute_all(&DesignInputs::default())
            .materials
            .cup_wall_check;
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let block_texts = [PROBLEM, problem.as_str(), guide.why, FIX, fix.as_str()];
        let y = |output: &egui::FullOutput, text: &str| text_rect(output, text).unwrap().top();
        // Red: open, in order (the problem in red, the sentence then its link, the fixes).
        let (output, _) = guide_frames(&ctx, &guide, &shown_at(Level::Bad), Vec::new());
        let texts = drawn_texts(&output);
        for want in block_texts.iter().chain(&[WHY_LINK, NO_NOTE_YET]) {
            assert!(texts.iter().any(|t| t == want), "no {want:?}");
        }
        assert!(!texts.iter().any(|t| t == CAUTION_DETAILS));
        assert_eq!(
            text_color(&output, &problem),
            Some(Level::Bad.color(&visuals))
        );
        assert!(y(&output, &problem) < y(&output, guide.why));
        assert!(y(&output, guide.why) < y(&output, NO_NOTE_YET));
        assert!(y(&output, NO_NOTE_YET) < y(&output, &fix));
        // Amber: a header and the Why link, the block shut; a click on the header opens it,
        // the problem in amber, and the Why link stays.
        let ctx = egui::Context::default();
        let amber = shown_at(Level::Caution);
        let (output, _) = guide_frames(&ctx, &guide, &amber, Vec::new());
        assert_eq!(
            drawn_texts(&output),
            [CAUTION_DETAILS, WHY_LINK, NO_NOTE_YET]
        );
        let header = guide_text_at(&ctx, &guide, &amber, CAUTION_DETAILS);
        let mut frames = click_at(header);
        frames.push(Vec::new());
        let (output, events) = guide_frames(&ctx, &guide, &amber, frames);
        assert_eq!(events, ReadoutEvents::default(), "opening asks for nothing");
        let texts = drawn_texts(&output);
        for want in block_texts
            .iter()
            .chain(&[CAUTION_DETAILS, WHY_LINK, NO_NOTE_YET])
        {
            assert!(
                texts.iter().any(|t| t == want),
                "amber, opened: no {want:?}"
            );
        }
        assert_eq!(
            text_color(&output, &problem),
            Some(Level::Caution.color(&visuals))
        );
        // Green: the Why line alone.
        let (output, _) = guide_frames(&ctx, &guide, &shown_at(Level::Good), Vec::new());
        assert_eq!(drawn_texts(&output), [WHY_LINK, NO_NOTE_YET]);
    }

    #[test]
    fn a_red_block_without_a_fix_says_so_and_its_trade_off_notes_follow_the_fixes() {
        let guide = guide_with(Learn::None("A test reason."));
        let ctx = egui::Context::default();
        let none = GuideShown {
            level: Level::Bad,
            fixes: &[],
            notes: &Vec::new,
        };
        let (output, _) = guide_frames(&ctx, &guide, &none, Vec::new());
        let texts = drawn_texts(&output);
        assert!(texts.iter().any(|t| t == FIX));
        assert!(texts.iter().any(|t| t == guidance::NO_FIX));
        let note = "A test trade-off.".to_owned();
        let notes = [note.clone()];
        let noted = GuideShown {
            level: Level::Bad,
            fixes: &CUP_WALL_FIX,
            notes: &|| notes.to_vec(),
        };
        let (output, _) = guide_frames(&ctx, &guide, &noted, Vec::new());
        assert!(!drawn_texts(&output).iter().any(|t| t == guidance::NO_FIX));
        let fix = fix_text(CUP_WALL_FIX[0].0, CUP_WALL_FIX[0].1);
        let y = |text: &str| text_rect(&output, text).unwrap().top();
        assert!(y(&fix) < y(&note));
    }

    #[test]
    fn a_guide_reads_its_notes_only_when_its_block_is_open() {
        let guide = guide_with(Learn::None("A test reason."));
        let reads = std::cell::Cell::new(0);
        let counting = || {
            reads.set(reads.get() + 1);
            vec!["A test trade-off.".to_owned()]
        };
        for (level, want) in [(Level::Good, 0), (Level::Caution, 0), (Level::Bad, 1)] {
            reads.set(0);
            let block = GuideShown {
                level,
                fixes: &CUP_WALL_FIX,
                notes: &counting,
            };
            let ctx = egui::Context::default();
            let (output, _) = guide_frames(&ctx, &guide, &block, Vec::new());
            assert_eq!(reads.get(), want, "{level:?}");
            let shown = drawn_texts(&output)
                .iter()
                .any(|t| t == "A test trade-off.");
            assert_eq!(shown, want > 0, "{level:?}");
        }
    }

    #[test]
    fn every_guide_line_has_glyphs_in_the_default_fonts_and_plain_words() {
        // On designs where each check fails: the status line, the fit row, every row's Why
        // line, and each guide at red whatever the design (its problem, its sentence, its
        // fixes for the design and its static ones, its notes); the fixed texts, and
        // the hover texts.
        let ctx = egui::Context::default();
        let mut plain: Vec<String> = Vec::new();
        for inputs in [
            DesignInputs::default(),
            short_magnets(),
            passing_design(),
            hot_day(),
            too_many_poles(),
            length_conflict(),
            design(|i| {
                i.clamps.boss_od_mm = 12.0;
                i.clamps.clamp_length_mm = 3.0;
            }),
            design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)),
        ] {
            let results = compute_all(&inputs);
            let fixes = all_fixes(&inputs, &results, false);
            let output = sized_frame(&ctx, egui::vec2(600.0, 8000.0), Vec::new(), |ui| {
                let mut readouts = Readouts::default();
                status_ui(ui, &inputs, &DesignInputs::default(), &results, false);
                fit_ui(ui, &inputs, &results, &fixes, &mut readouts);
                for &(_, learn) in &ROW_LEARN {
                    why_line(ui, None, learn, &mut readouts);
                }
                for (i, guide) in CHECK_GUIDES.iter().enumerate() {
                    let notes = guide_notes(&fixes, i, &inputs, &results);
                    let listed = match guide.fixes {
                        Fixes::Static(list) => list,
                        Fixes::PerTerm(_) => &fixes[i][..],
                    };
                    for fixes in [&fixes[i][..], listed] {
                        let shown = GuideShown {
                            level: Level::Bad,
                            fixes,
                            notes: &|| notes.clone(),
                        };
                        guide_ui(ui, guide, &shown, &inputs, &results, &mut readouts);
                    }
                }
            });
            plain.extend(drawn_texts(&output));
        }
        plain.extend(
            [
                NOTE_NOT_REVIEWED,
                FITS,
                DOES_NOT_FIT,
                CAUTION_DETAILS,
                guidance::NO_FIX,
            ]
            .map(str::to_owned),
        );
        // An assumption's rationale is the hover text of its Why link: plain words too (spec
        // 3.4, ruling R10; the slice-1 exemption for cells is gone). The branch stays even while
        // no row links an assumption: it covers a row linked to one again.
        let fit = guidance::guide(FIT_CHECK).unwrap().learn;
        for learn in ROW_LEARN.iter().map(|&(_, learn)| learn).chain([fit]) {
            match learn {
                Learn::None(reason) => plain.push(reason.to_owned()),
                Learn::Assumption(id) => plain.push(
                    ASSUMPTIONS
                        .iter()
                        .find(|a| a.id == id)
                        .unwrap()
                        .rationale
                        .to_owned(),
                ),
                Learn::Note(_) => {}
            }
        }
        for text in &plain {
            assert_glyphs(&ctx, text, "the check guides");
        }
        for text in &plain {
            assert!(
                guidance::plain_text_violations(text).is_empty(),
                "{text:?}: {:?}",
                guidance::plain_text_violations(text)
            );
        }
    }

    #[test]
    fn the_production_variation_assumption_is_named_as_its_input() {
```

with:

```rust
    #[test]
    fn the_production_variation_assumption_is_named_as_its_input() {
```

Replace:

```rust
    #[test]
    fn the_dashboard_draws_the_cup_wall_verdict_in_plain_words() {
        let ctx = egui::Context::default();
        let results = compute_all(&DesignInputs::default());
        let shown = verdict_display(
            "materials.cup_wall_check",
            &results.materials.cup_wall_check,
        )
        .expect("the default cup wall fails");
        let texts = drawn_texts(&dashboard_frame(&ctx, &DesignInputs::default()));
        assert_eq!(texts.iter().filter(|t| **t == shown).count(), 1, "{shown}");
        let engine_words = CUP_WALL_TOO_THIN.trim_end();
        assert!(
            !texts.iter().any(|t| t.contains(engine_words)),
            "the engine's cell-named verdict is drawn"
        );
    }

    #[test]
    fn a_why_link_outside_the_rows_is_one_link_naming_its_note() {
```

with:

```rust
    #[test]
    fn a_why_link_outside_the_rows_is_one_link_naming_its_note() {
```

**`magcoupling-rs/src/gui/geometry_view.rs`**

Replace:

```rust
    #[test]
    fn the_drawing_floor_is_the_room_that_keeps_the_drawing_over_three_list_rows() {
        // Slice 3 spec 6.2: drawing_floor is MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows of the
        // callout list (list_row, the row geometry_ui reserves). In a region of exactly that
        // height the drawing of the defaults (a list of more than three rows) keeps its 160
        // points; a point less and it shrinks below them.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let size = Vec2::new(1000.0, 900.0);
        let mut floor = f32::NAN;
        // The figures by hand, not the formulas again: in egui's default style (this context's)
        // a body text row is 14.375 points and the item spacing 3, so a list row is 17.375
        // points and the floor 160 + 3 x 17.375 = 212.125 (spec 6.2's 160-point drawing over
        // three list rows).
        sized_frame(&ctx, size, Vec::new(), |ui| {
            assert_eq!(
                (
                    ui.text_style_height(&egui::TextStyle::Body),
                    ui.spacing().item_spacing.y
                ),
                (14.375, 3.0),
                "egui's default style"
            );
            assert_eq!(list_row(ui), 17.375);
            floor = drawing_floor(ui);
        });
        assert_eq!(floor, 212.125);
        for (height, keeps) in [(floor, true), (floor - 1.0, false)] {
            let region = Rect::from_min_size(Pos2::new(20.0, 30.0), Vec2::new(700.0, height));
            let mut drawing = f32::NAN;
            for _ in 0..2 {
                crate::gui::test_support::region_frame(&ctx, size, region, |ui| {
                    let layout =
                        geometry_ui(ui, &inputs, &results, false, &mut Readouts::default());
                    drawing = layout.rect.height();
                });
            }
            assert_eq!(
                drawing >= MIN_DRAWING_HEIGHT - 1e-3,
                keeps,
                "a region of {height}: a drawing of {drawing}"
            );
        }
    }

    #[test]
    fn the_view_stays_in_a_short_region_the_drawing_shrinking_and_the_list_scrolling() {
```

with:

```rust
    #[test]
    fn the_view_stays_in_a_short_region_the_drawing_shrinking_and_the_list_scrolling() {
```

**`magcoupling-rs/src/gui/guidance.rs`**

Replace:

```rust
    use crate::engine::sizing::{SizingOutcome, solve};
    use crate::gui::dashboard::{CHECKS, DASHBOARD, FIX, dashboard_ui};
    use crate::gui::explorer::reviewed_note;
```

with:

```rust
    use crate::engine::sizing::{SizingOutcome, solve};
    use crate::gui::dashboard::{CHECKS, DASHBOARD};
    use crate::gui::explorer::reviewed_note;
```

Replace:

```rust
    use crate::gui::plain_text::label_for;
    use crate::gui::readouts::Readouts;
    use crate::gui::test_support::{
```

with:

```rust
    use crate::gui::plain_text::label_for;
    use crate::gui::test_support::{
```

Replace:

```rust
    use crate::gui::test_support::{
        QUOTED_UNITS, assert_glyphs, auto_states, clamp_at_its_limit, drawn_texts, end_effect_edge,
        glyph_context, hot_day, hub_overstressed, is_shaft_joint, keyway_through, mixed_rings,
        n42_parts, only_the_clamp_fails, only_the_gearbox_fails, only_the_joint_fails,
        passing_design, reversed_hub_band, shaft_joint_input_rows, shaft_joint_result_rows,
```

with:

```rust
    use crate::gui::test_support::{
        QUOTED_UNITS, assert_glyphs, auto_states, check_entry_texts, clamp_at_its_limit,
        end_effect_edge, glyph_context, hot_day, hub_overstressed, is_shaft_joint, keyway_through,
        mixed_rings, n42_parts, only_the_clamp_fails, only_the_gearbox_fails, only_the_joint_fails,
        passing_design, reversed_hub_band, shaft_joint_input_rows, shaft_joint_result_rows,
```

Replace:

```rust
        passing_design, reversed_hub_band, shaft_joint_input_rows, shaft_joint_result_rows,
        short_magnets, sized_frame, slider_range, small_boss, with_magnets, with_poles,
    };
```

with:

```rust
        passing_design, reversed_hub_band, shaft_joint_input_rows, shaft_joint_result_rows,
        short_magnets, slider_range, small_boss, with_magnets, with_poles,
    };
```

Replace:

```rust
    /// design (`inputs`, whose results are `results`) with the shaft-clamp section paused or
    /// shown (`paused`), and that the dashboard, fed the fixes as the panel hands them to it (no
    /// Minimum OD set), draws [`NO_FIX`] right after the Fix label of the guide's open block,
    /// found by its `problem`: `dashboard::fix_line` draws it for an open block whose fixes are
    /// empty (ruling B-4).
    fn assert_open_block_draws_no_fix(
```

with:

```rust
    /// design (`inputs`, whose results are `results`) with the shaft-clamp section paused or
    /// shown (`paused`), and that the check's Explain entry, fed the fixes as the panel hands
    /// them to it (no Minimum OD set), draws [`NO_FIX`] right under its What to change heading,
    /// after its `problem`: the entry draws it for a failing check whose fixes are empty
    /// (ruling B-4).
    fn assert_open_block_draws_no_fix(
```

Replace:

```rust
    ) {
        let all = within_minimum_diameter(&all_fixes(inputs, results, paused), inputs, None);
```

with:

```rust
    ) {
        use crate::gui::explorer::WHAT_TO_CHANGE;
        let all = within_minimum_diameter(&all_fixes(inputs, results, paused), inputs, None);
```

Replace:

```rust
        assert_eq!(all[guide_index], [], "paused {paused}");
        let output = sized_frame(
            &egui::Context::default(),
            egui::vec2(1280.0, 4000.0),
            Vec::new(),
            |ui| {
                dashboard_ui(
                    ui,
                    inputs,
                    &DesignInputs::default(),
                    results,
                    &all,
                    paused,
                    &mut Readouts::default(),
                )
            },
        );
        let texts = drawn_texts(&output);
        let check = CHECK_GUIDES[guide_index].check;
```

with:

```rust
        assert_eq!(all[guide_index], [], "paused {paused}");
        let check = CHECK_GUIDES[guide_index].check;
```

Replace:

```rust
        let check = CHECK_GUIDES[guide_index].check;
        let open = texts
```

with:

```rust
        let check = CHECK_GUIDES[guide_index].check;
        let texts = check_entry_texts(inputs, results, &all, paused, check);
        let open = texts
```

Replace:

```rust
            .position(|t| t == problem)
            .unwrap_or_else(|| panic!("the block of {check} is open, paused {paused}"));
        let fix = open + texts[open..].iter().position(|t| t == FIX).unwrap();
        assert_eq!(texts[fix + 1], NO_FIX, "paused {paused}");
```

with:

```rust
            .position(|t| t == problem)
            .unwrap_or_else(|| panic!("the entry of {check} has its problem, paused {paused}"));
        let fix = open
            + texts[open..]
                .iter()
                .position(|t| t == WHAT_TO_CHANGE)
                .unwrap();
        assert_eq!(texts[fix + 1], NO_FIX, "paused {paused}");
```

Replace:

```rust
        );
        emit("panel/welcome strip", crate::gui::panel::WELCOME);
        {
```

with:

```rust
        );
        {
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    use crate::engine::sizing::SizingOutcome;
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
    use crate::gui::format::{format_value, with_unit};
```

with:

```rust
    use crate::engine::sizing::SizingOutcome;
    use crate::gui::dashboard::{
        END_EFFECT_BANNER, STORED_3D_LABEL, end_effect_banner, result_info,
    };
    use crate::gui::format::{format_value, with_unit};
```

Replace:

```rust
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::geometry_view::MIN_DRAWING_HEIGHT;
    use crate::gui::guidance::{Consequences, FACE_GAP, RING_SIZE_STAYS, plain_text_violations};
    use crate::gui::input_ui::{BLANK_TEXT, CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
```

with:

```rust
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::guidance::{
        Consequences, FACE_GAP, RING_SIZE_STAYS, SIZE_TAB_NOTE, plain_text_violations,
    };
    use crate::gui::input_ui::{BLANK_TEXT, CHANGED_DOT, OUTSIDE_RANGE_NOTE, RESET_LABEL};
```

Replace:

```rust
        SCREEN, clamp_at_its_limit, drawing_rect, drawn_texts, flat_shapes, hot_day,
        hub_overstressed, key_event, key_tap, last_characters, no_back_iron, only_the_clamp_fails,
        only_the_joint_fails, passing_design, primary_button, row_line, select_all, short_magnets,
```

with:

```rust
        SCREEN, clamp_at_its_limit, drawing_rect, drawn_texts, flat_shapes, hot_day,
        hub_overstressed, key_event, key_tap, last_characters, only_the_clamp_fails,
        only_the_joint_fails, passing_design, primary_button, row_line, select_all, short_magnets,
```

Replace:

```rust
    };
    use crate::gui::torque_card::{CAPTION, Control, held_line};
    use crate::headline;
```

with:

```rust
    };
    use crate::gui::torque_card::{CAPTION, CARD_HEADING, Control, held_line};
    use crate::headline;
```

Replace:

```rust
    const GEARBOX_CHECK: &str = "gearbox.cold_high_check";

```

with:

```rust
    const GEARBOX_CHECK: &str = "gearbox.cold_high_check";
    const HOT_MIN_CHECK: &str = "metal.hot_min_check";
    const SPACE_CLAIM_CHECK: &str = "housing.space_claim_check";

```

Replace:

```rust

        /// A harness on a new panel as it opens (decision G-5): the Size tab under the welcome
        /// strip, on the laptop screen.
        pub(crate) fn first_view() -> Self {
```

with:

```rust

        /// A harness on a new panel as it opens (decision G-5): the Size tab, the Explain panel
        /// on its orientation, on the laptop screen.
        pub(crate) fn first_view() -> Self {
```

Replace:

```rust

        /// A click (move, press, release) at `at`.
```

with:

```rust

        /// Shows the Results tab, one frame in: its Headline group lists every headline number,
        /// each a readout (its hover, its click), as the right column did before it became the
        /// check list (UX overhaul spec 4.5).
        fn on_results(mut self) -> Self {
            self.panel.centre = CentreView::Results;
            self.frame(Vec::new());
            self
        }

        /// Selects `check`, as a click on its row of the check list does: the Explain panel
        /// shows its entry in the frame this returns.
        fn select(&mut self, check: &'static str) -> egui::FullOutput {
            self.panel.selection = Selection::Check(check);
            self.frame(Vec::new())
        }

        /// A click (move, press, release) at `at`.
```

Replace:

```rust

    /// The headline of `inputs`, as the panel displays it (value with unit; the cup wall's
    /// failing verdict with its input's label, ruling R9).
    fn displayed_headline(inputs: &DesignInputs) -> Vec<String> {
```

with:

```rust

    /// The left edge of the right column in `output`, less its margin: where the check list's
    /// heading starts.
    fn right_column_left(output: &egui::FullOutput) -> f32 {
        text_rect(output, crate::gui::checks::HEADING)
            .expect("the check list's heading")
            .left()
            - 10.0
    }

    /// The headline of `inputs`, as the Results tab's Headline group displays it (value with
    /// unit; the cup wall's failing verdict with its input's label, ruling R9).
    fn displayed_headline(inputs: &DesignInputs) -> Vec<String> {
```

Replace:

```rust
    }

    #[test]
    fn a_new_panel_opens_on_size_and_torque_under_the_welcome_strip() {
        // Decision G-5 (spec 3.1, reversing M42-2): the first view is the Size tab with the
        // welcome strip above the centre tabs; showing them edits nothing, and Explain stays
        // off (M43-2).
        use crate::gui::guidance::{design_status, design_status_text};
        use crate::gui::size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
```

with:

```rust
    }

    // ------------------------------------------------------------------ the right column

    /// Whether the one copy of `text` drawn in `output` is in view on `screen`: inside its
    /// clip rect and inside the screen.
    fn in_view(output: &egui::FullOutput, text: &str, screen: egui::Vec2) -> bool {
        let rects = text_rects(output, text);
        assert_eq!(rects.len(), 1, "{text:?}: {rects:?}");
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, screen);
        text_clip_rect(output, text).contains_rect(rects[0]) && screen.contains_rect(rects[0])
    }

    #[test]
    fn the_right_column_is_the_check_list_over_the_explain_panel_on_its_orientation() {
        // UX overhaul spec 4.5, 5.4 and 8.2 test 10: at 1280 x 800 a new panel draws the check
        // list (its status line and a row per check, in the guides' order) and under it the
        // Explain panel, which shows the orientation on the first frame. Register row A-18: the
        // line that says two of the checks read fixed 3D fields is in view with no scroll.
        use crate::gui::explorer::START_HERE_BUTTON;
        use crate::gui::guidance::counted_guides;
        use crate::gui::{checks, explain};
        let mut harness = Harness::first_view();
        let first = harness.frame(Vec::new());
        assert_eq!(count(&first, explain::ORIENTATION), 1, "on the first frame");
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Nothing);
        let mut column = vec![checks::HEADING];
        column.extend(counted_guides(false).map(|guide| guide.label));
        column.extend([STORED_3D_LABEL, explain::HEADING, explain::ORIENTATION]);
        column.extend(explain::STEPS);
        column.push(START_HERE_BUTTON);
        let left = right_column_left(&output);
        let mut last = f32::NEG_INFINITY;
        for text in column {
            // The Hot-low torque row's label is the Size tab's readout label too.
            let rect = text_rects(&output, text)
                .into_iter()
                .find(|rect| rect.left() > left)
                .unwrap_or_else(|| panic!("{text:?} in the right column"));
            assert!(
                text_clip_rect(&output, STORED_3D_LABEL).contains_rect(rect),
                "{text:?} is in view at {rect:?}"
            );
            assert!(rect.top() > last, "{text:?} is under the text before it");
            last = rect.top();
        }
        assert!(in_view(&output, STORED_3D_LABEL, SCREEN), "row A-18");
        // The list never takes more than its share of the column: the Explain panel's heading
        // starts above the middle of the screen.
        let explain = text_rect(&output, explain::HEADING).unwrap();
        assert!(explain.top() < SCREEN.y / 2.0, "{explain:?}");
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn a_click_on_a_check_shows_its_entry_and_esc_or_about_returns_the_orientation() {
        // Spec 5.1, 5.2 and 8.2 test 10, and goal G-8: one click on a failing check shows its
        // problem, its reason and its fixes. A hover changes nothing (decision C-2).
        use crate::gui::explain;
        use crate::gui::guidance::guide;
        let mut harness = Harness::first_view();
        let gearbox = guide(GEARBOX_CHECK).unwrap();
        let cup_wall = guide("materials.cup_wall_check").unwrap();
        harness.click_text(gearbox.label);
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
        let output = harness.frame(Vec::new());
        let panel = &harness.panel;
        let problem = (gearbox.problem)(panel.inputs(), panel.results()).expect("it fails");
        for text in [gearbox.why, problem.as_str(), explain::ABOUT] {
            assert_eq!(count(&output, text), 1, "{text:?}");
        }
        assert_eq!(count(&output, explain::ORIENTATION), 0);
        // Hovering another check's row leaves the entry as it is.
        let at = text_rect(&output, cup_wall.label).unwrap().center();
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
        assert_eq!(count(&output, gearbox.why), 1);
        // The last click owns the panel.
        harness.click(at);
        assert_eq!(
            harness.panel.selection,
            Selection::Check("materials.cup_wall_check")
        );
        // Esc returns the orientation; so does the link at the entry's head.
        let output = harness.frame(key_tap(egui::Key::Escape));
        assert_eq!(harness.panel.selection, Selection::Nothing);
        assert_eq!(count(&output, explain::ORIENTATION), 1);
        harness.click_text(gearbox.label);
        harness.click_text(explain::ABOUT);
        assert_eq!(harness.panel.selection, Selection::Nothing);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn esc_in_a_text_field_is_the_field_s_and_keeps_the_selection() {
        let mut harness = Harness::new();
        harness.select(GEARBOX_CHECK);
        harness.focus(PART_INNER);
        harness.frame(vec![egui::Event::Text("X".to_owned())]);
        harness.frame(key_tap(egui::Key::Escape));
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
        // The field let go of the keyboard: the next Esc is the panel's.
        harness.frame(key_tap(egui::Key::Escape));
        assert_eq!(harness.panel.selection, Selection::Nothing);
    }

    #[test]
    fn an_amber_check_s_entry_is_open_at_its_one_click() {
        // Before the Explain panel an amber check's block was shut under a header: two clicks.
        use crate::gui::guidance::guide;
        let mut hot = DesignInputs::default();
        hot_day(&mut hot);
        let temperature = guide("temperature.summary.verdict").unwrap();
        let mut harness = Harness::first_view();
        harness.panel.inputs = hot;
        harness.frame(Vec::new());
        harness.click_text(temperature.label);
        let output = harness.frame(Vec::new());
        let panel = &harness.panel;
        let problem = (temperature.problem)(panel.inputs(), panel.results()).expect("amber");
        assert_eq!(count(&output, &problem), 1);
        let visuals = harness.ctx.style().visuals.clone();
        assert_eq!(
            text_color(&output, &problem),
            Some(Level::Caution.color(&visuals))
        );
    }

    #[test]
    fn the_selection_is_a_view_choice_in_no_file_link_or_undo_step() {
        // Spec 5.1: selecting edits nothing, and reset all, a load and an undo keep it.
        let mut harness = Harness::new();
        let (file, link) = (
            design_to_json(&harness.panel.design()),
            harness.panel.share_link(),
        );
        harness.select(GEARBOX_CHECK);
        harness.frame(Vec::new());
        assert_eq!(design_to_json(&harness.panel.design()), file);
        assert_eq!(harness.panel.share_link(), link);
        assert_eq!(harness.panel.history.undo_len(), 0);
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.frame(Vec::new());
        harness.panel.undo();
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
        harness.panel.reset();
        harness.panel.load_design_file(&file).unwrap();
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
    }

    #[test]
    fn an_entry_s_equation_link_opens_the_equation_panel_and_the_selection_stays() {
        // Decisions C-11 and C-25: the Equation link opens the docked panel at the entry's
        // result, with no trace; a selection made afterwards leaves the open equation where it
        // is. The orientation's link starts the Start here walk.
        use crate::gui::explain;
        use crate::gui::explorer::START_HERE_BUTTON;
        use crate::gui::guidance::guide;
        let mut harness = Harness::first_view();
        assert!(!harness.panel.explorer.open);
        harness.click_text(guide(GEARBOX_CHECK).unwrap().label);
        harness.click_text(explain::EQUATION);
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.trail(), [GEARBOX_CHECK]);
        assert_eq!(harness.panel.selection, Selection::Check(GEARBOX_CHECK));
        assert_eq!(harness.panel.trace, None);
        harness.click_text(guide(SPACE_CLAIM_CHECK).unwrap().label);
        assert_eq!(harness.panel.selection, Selection::Check(SPACE_CLAIM_CHECK));
        assert_eq!(harness.panel.explorer.trail(), [GEARBOX_CHECK]);
        // Closing the Equation panel leaves the selection as it is.
        harness.panel.explorer.close();
        harness.frame(Vec::new());
        assert_eq!(harness.panel.selection, Selection::Check(SPACE_CLAIM_CHECK));
        harness.frame(key_tap(egui::Key::Escape));
        harness.click_text(START_HERE_BUTTON);
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.start_here(), Some(0));
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn the_check_list_and_the_explain_panel_compute_nothing_on_an_idle_frame() {
        // Spec 8.2, test 14: with nothing selected, with a failing check's entry open and with
        // the Hot-low tile's (the torque card: its consequences are 30 compute_all runs), a
        // frame in which nothing changed runs no fix search and no results of its own.
        let mut harness = Harness::first_view();
        let hot_low = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
        let checks = [HOT_MIN_CHECK, GEARBOX_CHECK].map(|check| Some(Selection::Check(check)));
        for check in [None, checks[0], checks[1], Some(hot_low)] {
            match check {
                Some(Selection::Check(path)) => {
                    harness.select(path);
                }
                Some(selection) => harness.panel.selection = selection,
                None => {}
            }
            harness.frame(Vec::new());
            let fixes = harness.panel.fixes.computed;
            let results = harness.panel.size_tab_results.computed;
            let consequences = harness.panel.consequences.computed;
            for _ in 0..5 {
                harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.fixes.computed, fixes, "{check:?}");
            assert_eq!(
                harness.panel.size_tab_results.computed, results,
                "{check:?}"
            );
            assert_eq!(
                harness.panel.consequences.computed, consequences,
                "{check:?}"
            );
        }
    }

    #[test]
    fn a_new_panel_opens_on_size_and_torque_with_the_tiles_over_the_drawing() {
        // Decision G-5 (spec 3.1, reversing M42-2) and UX overhaul spec 4.4: the first view is
        // the Size tab, its four tiles over the drawing; showing them edits nothing, and
        // Explain stays off (M43-2). The welcome strip is gone: the Explain panel's orientation
        // says what to do first.
        use crate::gui::guidance::{design_status, design_status_text};
        use crate::gui::size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
```

Replace:

```rust
        use crate::gui::size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
        let panel = MagcouplingPanel::new();
```

with:

```rust
        use crate::gui::size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
        use crate::gui::tiles::Tile;
        let panel = MagcouplingPanel::new();
```

Replace:

```rust
        assert_eq!(panel.centre, CentreView::SizeAndTorque);
        assert!(!panel.welcome_dismissed);
        assert!(!panel.explorer.explain);
```

with:

```rust
        assert_eq!(panel.centre, CentreView::SizeAndTorque);
        assert!(!panel.explorer.explain);
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, WELCOME), 1);
        assert_eq!(count(&output, HIDE_WELCOME), 1);
        // The inputs side is the Size tab's: its controls, the Key design group folded away
        // under the detailed inputs.
        for label in [DIAMETER, LENGTH, GAP, MAGNETS] {
            assert_eq!(count(&output, label), 1, "{label}");
```

with:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, crate::gui::explain::ORIENTATION), 1);
        // The inputs side is the Size tab's: the Required torque, then its controls, the Key
        // design group folded away under the detailed inputs.
        for label in [REQUIRED, DIAMETER, LENGTH, GAP, MAGNETS] {
            assert_eq!(count(&output, label), 1, "{label}");
```

Replace:

```rust
        assert_eq!(count(&output, KEY_DESIGN_HEADING), 0);
        // The dashboard's design status line (spec 3.1) is on the first view, counted live.
        let status = design_status(harness.panel.inputs(), harness.panel.results(), false);
```

with:

```rust
        assert_eq!(count(&output, KEY_DESIGN_HEADING), 0);
        // The check list's design status line (spec 3.1) is on the first view, counted live.
        let status = design_status(harness.panel.inputs(), harness.panel.results(), false);
```

Replace:

```rust
        assert_eq!(count(&output, &design_status_text(&status)), 1);
        // The strip is in the centre region, above the tab row.
        let strip = text_rect(&output, WELCOME).expect("the strip");
        let tabs = text_rect(&output, CentreView::Geometry.label()).expect("the tab row");
```

with:

```rust
        assert_eq!(count(&output, &design_status_text(&status)), 1);
        // The tiles are in the centre region, under the tab row and over the drawing, in their
        // order from the left.
        let tabs = text_rect(&output, CentreView::Geometry.label()).expect("the tab row");
```

Replace:

```rust
        let tabs = text_rect(&output, CentreView::Geometry.label()).expect("the tab row");
        assert!(strip.bottom() <= tabs.top(), "{strip:?} above {tabs:?}");
        assert!(
            strip.left() >= INPUTS_WIDTH && strip.right() <= SCREEN.x - DASHBOARD_WIDTH,
            "{strip:?}"
        );
        assert_eq!(harness.panel.design(), Design::default());
```

with:

```rust
        let tabs = text_rect(&output, CentreView::Geometry.label()).expect("the tab row");
        let drawing = drawing_rect(&output);
        let mut left = INPUTS_WIDTH;
        for tile in Tile::ALL {
            let label = tile.label(harness.panel.results());
            let at = centre_rect(&output, &label);
            assert!(
                tabs.bottom() <= at.top() && at.bottom() <= drawing.top(),
                "{label}: {at:?} between {tabs:?} and {drawing:?}"
            );
            assert!(at.left() >= left, "{label}: {at:?} right of {left}");
            left = at.left();
        }
        // Under the drawing its callout list, as on the Geometry tab, until a click on a
        // dimension tag shows its callout in the Explain panel (decision D-19).
        let list = centre_rect(&output, "Face gap 1.400 mm");
        assert!(drawing.bottom() <= list.top(), "{drawing:?} over {list:?}");
        assert_eq!(harness.panel.design(), Design::default());
```

Replace:

```rust
    #[test]
    fn hiding_the_welcome_strip_frees_its_row_and_edits_no_design() {
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        let tabs_under_strip = text_rect(&output, CentreView::Geometry.label()).unwrap();
        let file = design_to_json(&harness.panel.design());
        let link = harness.panel.share_link();
        harness.click_text(HIDE_WELCOME);
        assert!(harness.panel.welcome_dismissed);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, WELCOME), 0);
        assert_eq!(count(&output, HIDE_WELCOME), 0);
        let tabs = text_rect(&output, CentreView::Geometry.label()).unwrap();
        assert!(
            tabs.top() < tabs_under_strip.top(),
            "the tab row moves up into the strip's room: {tabs:?} vs {tabs_under_strip:?}"
        );
        // A view choice of this session: the view, the design, its file and its link stay as
        // they were, and it is no undo step.
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(design_to_json(&harness.panel.design()), file);
        assert_eq!(harness.panel.share_link(), link);
        assert_eq!(harness.panel.history.undo_len(), 0);
        assert!(!harness.panel.history.can_undo(&Design::default()));
        assert_eq!(harness.panel.last_error, None);
    }

    #[test]
    fn the_hidden_welcome_strip_stays_hidden_for_the_session() {
        // A loaded file, undo and redo, reset all, a refused file, a share link (as an edit and
        // as the session's start) and every view keep it hidden; only a new panel (the page
        // opened again) shows it.
        let hidden = |harness: &mut Harness, after: &str| {
            let output = harness.frame(Vec::new());
            assert!(harness.panel.welcome_dismissed, "{after}");
            assert_eq!(count(&output, WELCOME), 0, "{after}");
        };
        let mut harness = Harness::first_view();
        harness.click_text(HIDE_WELCOME);
        harness
            .panel
            .load_design_file(&design_to_json(&gap_design(2.0)))
            .unwrap();
        hidden(&mut harness, "a loaded design file");
        harness.click_text(UNDO);
        assert_eq!(harness.panel.design(), Design::default());
        hidden(&mut harness, "undo");
        harness.click_text(REDO);
        assert_eq!(harness.panel.design(), gap_design(2.0));
        hidden(&mut harness, "redo");
        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.design(), Design::default());
        hidden(&mut harness, "reset all");
        harness.panel.load_design_file("not a design").unwrap_err();
        hidden(&mut harness, "a refused design file");
        harness
            .panel
            .load_share_payload(&encode_share_payload(&gap_design(1.5)))
            .unwrap();
        hidden(&mut harness, "a share link");
        harness
            .panel
            .open_share_payload(&encode_share_payload(&gap_design(1.6)))
            .unwrap();
        hidden(&mut harness, "a share link opened as the session's start");
        for view in CentreView::ALL {
            harness.panel.centre = view;
            hidden(&mut harness, view.label());
        }
        assert!(!MagcouplingPanel::new().welcome_dismissed, "a new session");
    }

    #[test]
    fn the_details_toggle_starts_off_and_a_click_turns_it_on_and_off() {
```

with:

```rust
    #[test]
    fn the_details_toggle_starts_off_and_a_click_turns_it_on_and_off() {
```

Replace:

```rust
        use crate::gui::dashboard::{dashboard_lines, hover_text};
        // The pull-out on the dashboard and on the Size tab: one result, two readouts.
        let path = "model.pullout_Nm";
```

with:

```rust
        use crate::gui::dashboard::{dashboard_lines, hover_text};
        // The pull-out on the Size tab: its readout's hover, and its dashboard row's text (the
        // Results tab and the spreadsheet's Summary read the row).
        let path = "model.pullout_Nm";
```

Replace:

```rust
            }
            // The Size tab's readout in the centre, the dashboard's row on the right.
            let mut rects = text_rects(&output, &shown);
            rects.sort_by(|a, b| a.left().total_cmp(&b.left()));
            assert_eq!(rects.len(), 2, "{rects:?}");
            let (on_size_tab, on_dashboard) = (rects[0], rects[1]);
            // At the text's right end (a wrapped label's rect starts at the row's left), with
            // a frame for the last tooltip to close.
            for (rect, want) in [(on_size_tab, &readout), (on_dashboard, &row)] {
                let at = egui::pos2(rect.right() - 4.0, rect.center().y);
                harness.frame(vec![egui::Event::PointerMoved(at)]);
                harness.frame(Vec::new());
                let output = harness.frame(Vec::new());
                assert!(drawn_texts(&output).contains(want), "{want:?}");
            }
        }
```

with:

```rust
            }
            // The Size tab's readout in the centre.
            let rects = text_rects(&output, &shown);
            assert_eq!(rects.len(), 1, "{rects:?}");
            // At the text's right end (a wrapped label's rect starts at the row's left).
            let at = egui::pos2(rects[0].right() - 4.0, rects[0].center().y);
            harness.frame(vec![egui::Event::PointerMoved(at)]);
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert!(drawn_texts(&output).contains(&readout), "{readout:?}");
        }
```

Replace:

```rust
        // The web app opens a link's design on a new panel (MagcouplingApp::open_share_payload):
        // the Size tab and the welcome strip, as for the default design.
        let mut panel = MagcouplingPanel::new();
```

with:

```rust
        // The web app opens a link's design on a new panel (MagcouplingApp::open_share_payload):
        // the Size tab and the Explain panel's orientation, as for the default design.
        let mut panel = MagcouplingPanel::new();
```

Replace:

```rust
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(count(&output, WELCOME), 1);
        assert_eq!(harness.panel.design(), gap_design(2.0));
```

with:

```rust
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(count(&output, crate::gui::explain::ORIENTATION), 1);
        assert_eq!(harness.panel.design(), gap_design(2.0));
```

Replace:

```rust
        // only on an edit (decision M41-2): idle frames on the first view keep values off the
        // sliders' steps as they are. Slice 3 test 11: so do they under the torque card, open and
        // closed by a click on its heading, at the shipped clamp pause (the first case is the
        // harness's own, unpaused).
        let mut design = Design::default();
```

with:

```rust
        // only on an edit (decision M41-2): idle frames on the first view keep values off the
        // sliders' steps as they are. Slice 3 test 11: so do they while the Explain panel
        // draws the torque card (the Hot-low tile's entry), at the shipped clamp pause and
        // without it.
        use crate::gui::tiles::Tile;
        let mut design = Design::default();
```

Replace:

```rust
        design.inputs.metal.required_min_Nm = 2.2345;
        for card in [None, Some(true), Some(false)] {
            let mut harness = Harness::first_view();
```

with:

```rust
        design.inputs.metal.required_min_Nm = 2.2345;
        for (paused, card) in [(false, false), (true, false), (true, true)] {
            let mut harness = Harness::first_view();
```

Replace:

```rust
            let mut harness = Harness::first_view();
            harness.panel.clamps_paused = card.is_some();
            harness.panel.open_design(design.clone());
```

with:

```rust
            let mut harness = Harness::first_view();
            harness.panel.clamps_paused = paused;
            harness.panel.open_design(design.clone());
```

Replace:

```rust
            harness.panel.open_design(design.clone());
            let mut output = harness.frame(Vec::new());
```

with:

```rust
            harness.panel.open_design(design.clone());
            if card {
                harness.panel.selection = Selection::Result(Tile::HotLow.path());
            }
            let mut output = harness.frame(Vec::new());
```

Replace:

```rust
            let mut output = harness.frame(Vec::new());
            if card == Some(false) {
                harness.click(text_center(&output, CARD_HEADING));
            }
            for _ in 0..15 {
```

with:

```rust
            let mut output = harness.frame(Vec::new());
            for _ in 0..15 {
```

Replace:

```rust
            }
            assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
```

with:

```rust
            }
            let case = format!("paused {paused}, the card {card}");
            assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
```

Replace:

```rust
            assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
            assert_eq!(count(&output, WELCOME), 1);
            assert_eq!(
```

with:

```rust
            assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
            assert_eq!(
```

Replace:

```rust
                count(&output, CAPTION),
                usize::from(card != Some(false)),
                "{card:?}: the card's caption shows exactly while it is open"
            );
```

with:

```rust
                count(&output, CAPTION),
                usize::from(card),
                "{case}: the card's caption shows exactly in the Hot-low tile's entry"
            );
```

Replace:

```rust
            );
            assert_eq!(harness.panel.design(), design, "{card:?}");
            assert_eq!(harness.panel.last_error, None, "{card:?}");
            assert_eq!(harness.panel.history.undo_len(), 0, "{card:?}");
        }
```

with:

```rust
            );
            assert_eq!(harness.panel.design(), design, "{case}");
            assert_eq!(harness.panel.last_error, None, "{case}");
            assert_eq!(harness.panel.history.undo_len(), 0, "{case}");
        }
```

Replace:

```rust
    #[test]
    fn the_welcome_strip_wraps_in_a_narrow_window() {
        // The M4-1 review's ~930 px window leaves the centre region about 300 points: the strip
        // wraps inside it, left of the dashboard, above the tab row.
        let width = 930.0;
        let mut harness = Harness::of(MagcouplingPanel::new(), egui::vec2(width, 1024.0));
        // The wrapped rows settle on the second frame (egui wraps from the last frame's widths).
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let strip = text_rect(&output, WELCOME).expect("the strip");
        let hide = text_rect(&output, HIDE_WELCOME).expect("its button");
        let tabs = text_rect(&output, CentreView::Geometry.label()).expect("the tab row");
        assert!(
            strip.left() >= INPUTS_WIDTH && strip.right() <= width - DASHBOARD_WIDTH,
            "{strip:?}"
        );
        assert!(hide.right() <= width - DASHBOARD_WIDTH, "{hide:?}");
        assert!(
            strip.height() >= 2.0 * hide.height(),
            "wrapped: {strip:?} {hide:?}"
        );
        assert!(strip.bottom() <= tabs.top(), "{strip:?} above {tabs:?}");
    }

    #[test]
    fn the_welcome_strip_s_texts_are_plain_language() {
        // Spec section 6, test 3: no cell references, correction ids or workbook jargon.
        for text in [WELCOME, HIDE_WELCOME, HIDE_WELCOME_HINT] {
            assert_eq!(
                crate::gui::guidance::plain_text_violations(text),
                Vec::<&str>::new(),
                "{text}"
            );
        }
    }

    #[test]
    fn each_plot_tab_draws_its_plot_from_this_frame_s_results() {
```

with:

```rust
    #[test]
    fn each_plot_tab_draws_its_plot_from_this_frame_s_results() {
```

Replace:

```rust
    fn every_headline_number_and_key_design_input_is_drawn_with_its_label() {
        let mut harness = Harness::on_screen(TALL);
        let output = harness.frame(Vec::new());
```

with:

```rust
    fn every_headline_number_and_key_design_input_is_drawn_with_its_label() {
        let mut harness = Harness::on_screen(TALL).on_results();
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        let texts = drawn_texts(&output);
        for (_, path) in HEADLINE {
```

with:

```rust
        let texts = drawn_texts(&output);
        // The Results tab's Headline group draws each under its catalogue label.
        for (_, path) in HEADLINE {
```

Replace:

```rust
        for (_, path) in HEADLINE {
            let label = label_for(path);
            assert!(texts.iter().any(|t| t == label), "missing {label:?}");
```

with:

```rust
        for (_, path) in HEADLINE {
            let label = result_info(path).unwrap().meta.label;
            assert!(texts.iter().any(|t| t == label), "missing {label:?}");
```

Replace:

```rust
        assert_eq!(harness.panel.history.undo_len(), 0);
        // Slice 3 test 11: the Size tab under the torque card, open and then closed by a click
        // on its heading, at the shipped clamp pause: idle frames still change no input.
        harness.panel.clamps_paused = true;
```

with:

```rust
        assert_eq!(harness.panel.history.undo_len(), 0);
        // Slice 3 test 11: the Size tab with the torque card drawn (the Hot-low tile's Explain
        // entry) and then without it, at the shipped clamp pause: idle frames still change no
        // input.
        harness.panel.clamps_paused = true;
```

Replace:

```rust
        for open in [true, false] {
            let mut output = harness.frame(Vec::new());
```

with:

```rust
        for open in [true, false] {
            harness.panel.selection = if open {
                Selection::Result(crate::gui::tiles::Tile::HotLow.path())
            } else {
                Selection::Nothing
            };
            let mut output = harness.frame(Vec::new());
```

Replace:

```rust
            let mut output = harness.frame(Vec::new());
            if !open {
                harness.click(text_center(&output, CARD_HEADING));
            }
            for _ in 0..15 {
```

with:

```rust
            let mut output = harness.frame(Vec::new());
            for _ in 0..15 {
```

Replace:

```rust
    fn an_arrow_key_on_the_face_gap_slider_updates_the_headline() {
        let mut harness = Harness::on_screen(TALL);
        harness.focus(FACE_GAP);
```

with:

```rust
    fn an_arrow_key_on_the_face_gap_slider_updates_the_headline() {
        let mut harness = Harness::on_screen(TALL).on_results();
        harness.focus(FACE_GAP);
```

Replace:

```rust
    fn clicking_the_end_of_the_face_gap_rail_sets_its_maximum() {
        let mut harness = Harness::on_screen(TALL);
        let rail = harness.widget(FACE_GAP).rect;
```

with:

```rust
    fn clicking_the_end_of_the_face_gap_rail_sets_its_maximum() {
        let mut harness = Harness::on_screen(TALL).on_results();
        let rail = harness.widget(FACE_GAP).rect;
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Coupling"), 1);
        assert_eq!(count(&output, "Shaft joints"), 1);
        assert_eq!(count(&output, &format!("4 of {total} inputs")), 1);
```

with:

```rust
        let output = harness.frame(Vec::new());
        // On the inputs side: the check list has a "Shaft joints" row of its own.
        let headings = |text: &str| {
            text_rects(&output, text)
                .into_iter()
                .filter(|rect| rect.left() < INPUTS_WIDTH)
                .count()
        };
        assert_eq!(headings("Coupling"), 1);
        assert_eq!(headings("Shaft joints"), 1);
        assert_eq!(count(&output, &format!("4 of {total} inputs")), 1);
```

Replace:

```rust
    #[test]
    fn the_dashboard_shows_the_stored_3d_label_and_the_corrected_markers() {
        use crate::gui::dashboard::MARKER_LEGEND;
```

with:

```rust
    #[test]
    fn the_right_column_shows_the_stored_3d_line_and_the_correction_ids() {
        use crate::gui::dashboard::MARKER_LEGEND;
```

Replace:

```rust
        let mut harness = Harness::on_screen(TALL);
        // The correction ids and their legend only with the spreadsheet details (spec S2-5).
        harness.panel.show_details = true;
        let output = harness.frame(Vec::new());
```

with:

```rust
        let mut harness = Harness::on_screen(TALL);
        // The check list closes with the stored-3D line, whatever is selected; the entry of a
        // check that reads the stored fields repeats it under its numbers.
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        assert_eq!(count(&output, STORED_3D_LABEL), 1);
        // The pull-out's marker: E3 changes it at the defaults, E7 and E8 have probes on it.
        assert!(count(&output, "E3 E7 E8") >= 1);
        assert_eq!(count(&output, MARKER_LEGEND), 1);
        assert_eq!(count(&output, "Inside the space claim"), 1);
```

with:

```rust
        assert_eq!(count(&output, STORED_3D_LABEL), 1);
        let output = harness.select("temperature.summary.verdict");
        assert_eq!(count(&output, STORED_3D_LABEL), 2);
        let output = harness.select(SPACE_CLAIM_CHECK);
        assert_eq!(count(&output, STORED_3D_LABEL), 1);
        assert_eq!(count(&output, "Inside the space claim"), 1);
```

Replace:

```rust
        );
        harness.panel.show_details = false;
```

with:

```rust
        );
        // The correction ids and their legend only with the spreadsheet details (spec S2-5):
        // the hot-low torque's, among the numbers behind the hot minimum.
        let marker = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
            .find(|line| line.path == "metal.torque_hot_low_Nm")
            .expect("the hot-low torque's row")
            .marker;
        assert!(
            !marker.is_empty(),
            "the hot-low torque carries correction ids"
        );
        harness.panel.show_details = true;
        let output = harness.select(HOT_MIN_CHECK);
        assert!(count(&output, &marker) >= 1);
        assert_eq!(count(&output, MARKER_LEGEND), 1);
        harness.panel.show_details = false;
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "E3 E7 E8"), 0);
        assert_eq!(count(&output, MARKER_LEGEND), 0);
```

with:

```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &marker), 0);
        assert_eq!(count(&output, MARKER_LEGEND), 0);
```

Replace:

```rust
            .collect();
        // Over the dashboard and over the centre region, whichever view it shows (decision
        // M42-1).
        let text = "End-effect model out of range (end-effect factor -1.202): the pull-out and the numbers computed from it are greyed.";
```

with:

```rust
            .collect();
        // Over the centre region, whichever view it shows (decision M42-1). The check list
        // marks the End-effect model check as failing; it does not repeat the banner.
        let text = "End-effect model out of range (end-effect factor -1.202): the pull-out and the numbers computed from it are greyed.";
```

Replace:

```rust
        let text = "End-effect model out of range (end-effect factor -1.202): the pull-out and the numbers computed from it are greyed.";
        assert_eq!(banner, [text, text]);
    }
```

with:

```rust
        let text = "End-effect model out of range (end-effect factor -1.202): the pull-out and the numbers computed from it are greyed.";
        assert_eq!(banner, [text]);
    }
```

Replace:

```rust
        harness.panel.inputs.coupling.magnets.axial_length_mm = Some(50.8);
        let output = harness.frame(Vec::new());
        let claim = drawn_texts(&output)
```

with:

```rust
        harness.panel.inputs.coupling.magnets.axial_length_mm = Some(50.8);
        let output = harness.select(SPACE_CLAIM_CHECK);
        let claim = drawn_texts(&output)
```

Replace:

```rust
            .find(|t| t.starts_with("Exceeds the space claim:"))
            .expect("the badge text");
        assert!(claim.contains("overall length"), "{claim}");
```

with:

```rust
            .find(|t| t.starts_with("Exceeds the space claim:"))
            .expect("the verdict, among the numbers of the check's entry");
        assert!(claim.contains("overall length"), "{claim}");
```

Replace:

```rust
        }
        // Five red badges and three amber in the table, in the dashboard's shapes (decision
        // C-4): every mark drawn but the dashboard's own.
        let visuals = harness.ctx.style().visuals.clone();
```

with:

```rust
        }
        // Five red badges and three amber in the table, in the check list's shapes (decision
        // C-4): every disc and triangle drawn but the check list's own, one per failing check.
        let visuals = harness.ctx.style().visuals.clone();
```

Replace:

```rust
        let visuals = harness.ctx.style().visuals.clone();
        let lines = crate::gui::dashboard::dashboard_lines(harness.panel.results());
        let table_badges = |level: Level| {
```

with:

```rust
        let visuals = harness.ctx.style().visuals.clone();
        let panel = &harness.panel;
        let fail_marks = crate::gui::guidance::counted_guides(false)
            .filter(|guide| {
                crate::gui::guidance::level(guide, panel.inputs(), panel.results()) == Level::Bad
            })
            .count();
        let caution_marks = crate::gui::guidance::counted_guides(false)
            .filter(|guide| {
                crate::gui::guidance::level(guide, panel.inputs(), panel.results())
                    == Level::Caution
            })
            .count();
        let table_badges = |level: Level| {
```

Replace:

```rust
            drawn
                - lines
                    .iter()
                    .filter(|line| line.level == Some(level))
                    .count()
        };
```

with:

```rust
            drawn
                - match level {
                    Level::Bad => fail_marks,
                    Level::Caution => caution_marks,
                    Level::Good => 0,
                }
        };
```

Replace:

```rust
        use crate::gui::test_support::flat_shapes;
        // Short magnets (f_end < 0): the dashboard greys the rows computed from the pull-out,
        // without badges. In the table's headline (open) each check row carries exactly the
        // dashboard's badge, so the hot minimum (red by its text), the screw sized from the
        // invalid pull-out, and the shaft joints and gearbox checks of a cold-high below zero
```

with:

```rust
        use crate::gui::test_support::flat_shapes;
        // Short magnets (f_end < 0): the rows computed from the pull-out are greyed, without
        // badges. In the table's headline (open) each check row carries exactly its dashboard
        // line's badge, so the hot minimum (red by its text), the screw sized from the
        // invalid pull-out, and the shaft joints and gearbox checks of a cold-high below zero
```

Replace:

```rust
        let visuals = harness.ctx.style().visuals.clone();
        // Between the inputs and the dashboard's badges (left of its first label).
        let pullout = label_for("model.pullout_Nm");
        let dashboard_left = text_rects(&output, pullout)
            .into_iter()
            .map(|r| r.left())
            .fold(f32::MIN, f32::max)
            - 30.0;
        let centre = |x: f32| x > INPUTS_WIDTH && x < dashboard_left;
        let lines = dashboard_lines(harness.panel.results());
```

with:

```rust
        let visuals = harness.ctx.style().visuals.clone();
        // Between the inputs and the check list.
        let checks_left = right_column_left(&output);
        let centre = |x: f32| x > INPUTS_WIDTH && x < checks_left;
        let lines = dashboard_lines(harness.panel.results());
```

Replace:

```rust
    fn clicking_an_input_s_label_frames_the_results_it_drives_until_a_second_click() {
        use crate::gui::dashboard::DASHBOARD;
        use crate::gui::trace::TraceKind;
```

with:

```rust
    fn clicking_an_input_s_label_frames_the_results_it_drives_until_a_second_click() {
        use crate::gui::guidance::counted_guides;
        use crate::gui::trace::TraceKind;
```

Replace:

```rust
        let marks = trace_marks(&harness, &output);
        // On the dashboard (right of its first label's badge): one frame per row the gap
        // drives, none on the others.
        let pullout_label = label_for("model.pullout_Nm");
        let dashboard_left = text_rect(&output, pullout_label).unwrap().left() - 30.0;
        let on_dashboard = marks
            .iter()
            .filter(|m| m.center().x > dashboard_left)
            .count();
        let driven = DASHBOARD
            .iter()
            .filter(|(path, _)| trace.marks(path))
            .count();
```

with:

```rust
        let marks = trace_marks(&harness, &output);
        // In the check list: one frame per check the gap drives, none on the others.
        let checks_left = right_column_left(&output);
        let on_checks = marks.iter().filter(|m| m.center().x > checks_left).count();
        let driven = counted_guides(false)
            .filter(|guide| trace.marks(guide.check))
            .count();
```

Replace:

```rust
        assert!(driven > 0);
        assert_eq!(on_dashboard, driven);
        // On the inputs side: the gap's own Key design row alone.
```

with:

```rust
        assert!(driven > 0);
        assert_eq!(on_checks, driven);
        // On the inputs side: the gap's own Key design row alone.
```

Replace:

```rust
        use crate::gui::trace::{CLEAR_TRACE, TraceKind};
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
```

with:

```rust
        use crate::gui::trace::{CLEAR_TRACE, TraceKind};
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0)).on_results();
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
```

Replace:

```rust
        use crate::gui::trace::TraceKind;
        // A result clicked on the dashboard replaces an input's trace with its own (which marks
        // inputs, and cannot filter the table): the box goes off with it, so an input traced
```

with:

```rust
        use crate::gui::trace::TraceKind;
        // A result clicked in the Explain panel replaces an input's trace with its own (which marks
        // inputs, and cannot filter the table): the box goes off with it, so an input traced
```

Replace:

```rust
        harness.click_text(CentreView::Geometry.label());
        let outside = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
```

with:

```rust
        harness.click_text(CentreView::Geometry.label());
        // One of the numbers behind a check, in that check's Explain entry.
        let (outside, check) = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
```

Replace:

```rust
            .into_iter()
            .find(|line| registry().equation_for(line.path).is_some() && !gap.marks(line.path))
            .expect("a dashboard result the face gap does not drive");
        let output = harness.frame(Vec::new());
        // The rightmost: the dashboard's.
        let value = text_rects(&output, &outside.shown_value())
```

with:

```rust
            .into_iter()
            .filter_map(|line| {
                let (_, check) = crate::gui::dashboard::DASHBOARD
                    .iter()
                    .find(|&&(path, _)| path == line.path)?;
                Some((line, (*check)?))
            })
            .find(|(line, _)| registry().equation_for(line.path).is_some() && !gap.marks(line.path))
            .expect("a number behind a check that the face gap does not drive");
        let output = harness.select(check);
        // The rightmost: the Explain panel's.
        let value = text_rects(&output, &outside.shown_value())
```

Replace:

```rust
        assert_eq!(harness.panel.trace.as_ref(), Some(&trace));
        // A result outside the trace, with a record (on the dashboard): its trace replaces the
        // input's.
        let outside = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
```

with:

```rust
        assert_eq!(harness.panel.trace.as_ref(), Some(&trace));
        // A result outside the trace, with a record (in the Explain panel): its trace replaces
        // the input's.
        // One of the numbers behind a check, in that check's Explain entry.
        let (outside, check) = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
```

Replace:

```rust
            .into_iter()
            .find(|line| registry().equation_for(line.path).is_some() && !trace.marks(line.path))
            .expect("a dashboard result the face gap does not drive");
        let output = harness.frame(Vec::new());
        // The rightmost: the dashboard's.
        let value = text_rects(&output, &outside.shown_value())
```

with:

```rust
            .into_iter()
            .filter_map(|line| {
                let (_, check) = crate::gui::dashboard::DASHBOARD
                    .iter()
                    .find(|&&(path, _)| path == line.path)?;
                Some((line, (*check)?))
            })
            .find(|(line, _)| {
                registry().equation_for(line.path).is_some() && !trace.marks(line.path)
            })
            .expect("a number behind a check that the face gap does not drive");
        let output = harness.select(check);
        // The rightmost: the Explain panel's.
        let value = text_rects(&output, &outside.shown_value())
```

Replace:

```rust
    fn reset_all_restores_the_default_design_and_headline() {
        let mut harness = Harness::on_screen(TALL);
        harness.focus(FACE_GAP);
```

with:

```rust
    fn reset_all_restores_the_default_design_and_headline() {
        let mut harness = Harness::on_screen(TALL).on_results();
        harness.focus(FACE_GAP);
```

Replace:

```rust
            if clamps_paused {
                // The dashboard and its guides on designs failing every check, the clamp's too.
                let mut hot = DesignInputs::default();
```

with:

```rust
            if clamps_paused {
                // The check list and each check's Explain entry on designs failing every check,
                // the clamp's too.
                let mut hot = DesignInputs::default();
```

Replace:

```rust
                    harness.panel.inputs = inputs;
                    harness.frame(Vec::new());
                    texts.extend(drawn_texts(&harness.frame(Vec::new())));
                }
```

with:

```rust
                    harness.panel.inputs = inputs;
                    for guide in crate::gui::guidance::counted_guides(true) {
                        harness.select(guide.check);
                        texts.extend(drawn_texts(&harness.frame(Vec::new())));
                    }
                }
```

Replace:

```rust
                // The shaft joints' guide (BL-064) counts while the section is paused: its
                // Problem drawn, open under its red row, where the inner hub slips, where it is
                // over its stress limit and cannot be fitted either, and on an uncooled wheel
```

with:

```rust
                // The shaft joints' guide (BL-064) counts while the section is paused: its
                // Problem drawn in its Explain entry, where the inner hub slips, where it is
                // over its stress limit and cannot be fitted either, and on an uncooled wheel
```

Replace:

```rust
                    harness.panel.inputs = inputs;
                    harness.frame(Vec::new());
                    let drawn = drawn_texts(&harness.frame(Vec::new()));
```

with:

```rust
                    harness.panel.inputs = inputs;
                    harness.select(joint.check);
                    let drawn = drawn_texts(&harness.frame(Vec::new()));
```

Replace:

```rust
        texts.push(crate::gui::results_table::CLEAR_TO_CLOSE.to_owned());
        texts.extend([WELCOME, HIDE_WELCOME, HIDE_WELCOME_HINT].map(str::to_owned));
        texts.extend([SHOW_DETAILS, SHOW_DETAILS_HINT].map(str::to_owned));
```

with:

```rust
        texts.push(crate::gui::results_table::CLEAR_TO_CLOSE.to_owned());
        texts.extend([SHOW_DETAILS, SHOW_DETAILS_HINT].map(str::to_owned));
```

Replace:

```rust
            harness.panel.inputs = at_limit.clone();
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
```

with:

```rust
            harness.panel.inputs = at_limit.clone();
            harness.select(HOT_MIN_CHECK);
            let output = harness.frame(Vec::new());
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        // Hot-low and the pull-out read no prediction, and the gap is not shown (spec 3.2, 5).
        assert_eq!(count(&output, NO_PREDICTION), 2);
        assert_eq!(beside(&output, HOT_LOW)[0].0, NO_PREDICTION);
        let (gap, _) = torque_gap_text(
```

with:

```rust
        let output = harness.frame(Vec::new());
        // The three torque tiles read no prediction, and the Hot-low tile shows neither what it
        // needs nor the gap (spec 3.2, 5; UX overhaul spec 4.4).
        assert_eq!(count(&output, NO_PREDICTION), 3);
        let (gap, _) = torque_gap_text(
```

Replace:

```rust
        assert_eq!(count(&output, &gap), 0, "{gap}");
    }
```

with:

```rust
        assert_eq!(count(&output, &gap), 0, "{gap}");
        let texts = drawn_texts(&output);
        assert!(
            !texts
                .iter()
                .any(|t| t.starts_with(crate::gui::tiles::NEEDS)),
            "{texts:?}"
        );
    }
```

Replace:

```rust
    /// A harness on a screen of `screen` showing the Size tab for `inputs`, set as the panel's
    /// inputs: the tab clicked, then one more frame, so the inputs side draws its controls.
    fn size_tab(screen: egui::Vec2, inputs: DesignInputs) -> Harness {
```

with:

```rust
    /// A harness on a screen of `screen` showing the Size tab for `inputs`, set as the panel's
    /// inputs: the tab clicked, then one more frame, so the inputs side draws its controls. The
    /// Hot-low tile is selected, so the Explain panel draws the torque card (decision C-10).
    fn size_tab(screen: egui::Vec2, inputs: DesignInputs) -> Harness {
```

Replace:

```rust
        let mut harness = Harness::on_screen(screen);
        harness.panel.inputs = inputs;
```

with:

```rust
        let mut harness = Harness::on_screen(screen);
        harness.panel.selection = Selection::Result(crate::gui::tiles::Tile::HotLow.path());
        harness.panel.inputs = inputs;
```

Replace:

```rust

    /// The texts drawn on the row of the label `label` and right of it, nearest first, with
    /// their rects: what a readout row shows beside its label (texts further right on the same
    /// row, the dashboard's, come after).
    fn beside(output: &egui::FullOutput, label: &str) -> Vec<(String, egui::Rect)> {
        let at = centre_rect(output, label);
        let mut row: Vec<(String, egui::Rect)> = crate::gui::test_support::flat_shapes(output)
            .into_iter()
```

with:

```rust

    /// The Size column's Required slider's value box in `output`: the first text under the
    /// row's label in the inputs side, the requirement with its unit.
    fn required_box(output: &egui::FullOutput) -> egui::Rect {
        let label = text_rect(output, REQUIRED).expect("the Required row");
        assert!(label.right() <= INPUTS_WIDTH, "{label:?}");
        let mut under: Vec<(String, egui::Rect)> = crate::gui::test_support::flat_shapes(output)
            .into_iter()
```

Replace:

```rust
                egui::Shape::Text(text) => {
                    let mut rect = text.galley.rect.translate(text.pos.to_vec2());
                    // In a wrapped row a label's galley starts at the row's left edge and is
                    // indented to where the text goes: its first glyph marks the text's left.
                    if let Some(glyph) = text.galley.rows.first().and_then(|r| r.glyphs.first()) {
                        rect.min.x = rect.min.x.max(text.pos.x + glyph.pos.x);
                    }
                    let on_row = (rect.center().y - at.center().y).abs() < at.height() / 2.0;
                    (on_row && rect.left() >= at.right() - 0.5)
                        .then(|| (text.galley.text().to_owned(), rect))
```

with:

```rust
                egui::Shape::Text(text) => {
                    let rect = text.galley.rect.translate(text.pos.to_vec2());
                    (rect.right() <= INPUTS_WIDTH && rect.top() >= label.bottom() - 0.5)
                        .then(|| (text.galley.text().to_owned(), rect))
```

Replace:

```rust
            .collect();
        row.sort_by(|a, b| a.1.left().total_cmp(&b.1.left()));
        row
    }
```

with:

```rust
            .collect();
        under.sort_by(|a, b| a.1.top().total_cmp(&b.1.top()));
        let (text, rect) = under.remove(0);
        assert!(text.ends_with(" N\u{b7}m"), "{text}");
        rect
    }
```

Replace:

```rust
    #[test]
    fn size_tab_reads_the_requirement_the_hot_low_gap_and_the_nominal_pull_out() {
        // Spec 3.2 at the default design, whose hot-low torque is below the requirement: the gap
        // reads short, in red, as the dashboard's hot minimum check fails.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
```

with:

```rust
    #[test]
    fn the_tiles_read_hot_low_what_it_needs_its_gap_and_the_nominal_pull_out() {
        // Spec 3.2 and UX overhaul spec 4.4 at the default design, whose hot-low torque is below
        // the requirement: the gap reads short, in red, as the hot minimum check fails.
        use crate::gui::test_support::text_color;
        let mut harness = size_tab(SCREEN, DesignInputs::default());
```

Replace:

```rust
        }
        // Hot-low torque is the dashboard row's name too (spec 3.5): one readout in the centre.
        let in_centre =
```

with:

```rust
        }
        // Hot-low torque is a check's name too (spec 3.5): one tile in the centre.
        let in_centre =
```

Replace:

```rust
        assert!(!meets && gap.starts_with("short "), "{gap}");
        let hot_low_row: Vec<String> = beside(&output, HOT_LOW)
            .into_iter()
```

with:

```rust
        assert!(!meets && gap.starts_with("short "), "{gap}");
        // The Hot-low tile from its top: its number, its label, what it needs, the gap.
        use crate::gui::tiles::{NEEDS, gap_pieces};
        let needs = with_unit(format_value(&Value::Num(required)), "N\u{b7}m");
        let pieces = gap_pieces(&gap);
        let hot_low = shown("metal.torque_hot_low_Nm");
        let tops: Vec<f32> = [hot_low.as_str(), HOT_LOW, NEEDS, pieces[0].as_str()]
            .into_iter()
```

Replace:

```rust
            .into_iter()
            .take(2)
            .map(|(text, _)| text)
            .collect();
```

with:

```rust
            .into_iter()
            .map(|text| centre_rect(&output, text).top())
            .collect();
```

Replace:

```rust
            .collect();
        assert_eq!(hot_low_row, [shown("metal.torque_hot_low_Nm"), gap.clone()]);
        assert_eq!(beside(&output, &label)[0].0, shown("model.pullout_Nm"));
        assert!(
            beside(&output, REQUIRED)[0].0.ends_with(" N\u{b7}m"),
            "the Required slider's value box"
        );
        assert_eq!(
```

with:

```rust
            .collect();
        assert!(tops.is_sorted(), "{tops:?}");
        // The pull-out's tile: its number over its label.
        let pullout = centre_rect(&output, &shown("model.pullout_Nm"));
        assert!(pullout.bottom() <= centre_rect(&output, &label).top() + 0.5);
        // Red is on what the tile needs and on the gap, never on the number alone.
        let visuals = harness.ctx.style().visuals.clone();
        let bad = Level::Bad.color(&visuals);
        let reds = [NEEDS, needs.as_str()].into_iter();
        for text in reds.chain(pieces.iter().map(String::as_str)) {
            assert_eq!(text_color(&output, text), Some(bad), "{text}");
        }
        assert_eq!(
```

Replace:

```rust
        assert_eq!(
            crate::gui::test_support::text_color(&output, &gap),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
```

with:

```rust
        assert_eq!(
            text_color(&output, &hot_low),
            Some(visuals.strong_text_color())
        );
```

Replace:

```rust
            let output = harness.frame(Vec::new());
            let (_, value_box) = beside(&output, REQUIRED).remove(0);
            harness.click(value_box.center());
```

with:

```rust
            let output = harness.frame(Vec::new());
            let value_box = required_box(&output);
            harness.click(value_box.center());
```

Replace:

```rust
        assert!(meets && gap.starts_with("margin +"), "{gap}");
        assert_eq!(
            crate::gui::test_support::text_color(&output, &gap),
            Some(Level::Good.color(&harness.ctx.style().visuals))
        );
        // Files and links carry it.
```

with:

```rust
        assert!(meets && gap.starts_with("margin +"), "{gap}");
        for piece in crate::gui::tiles::gap_pieces(&gap) {
            assert_eq!(
                crate::gui::test_support::text_color(&output, &piece),
                Some(Level::Good.color(&harness.ctx.style().visuals)),
                "{piece}"
            );
        }
        // Files and links carry it.
```

Replace:

```rust
        let output = harness.frame(Vec::new());
        let (_, value_box) = beside(&output, REQUIRED).remove(0);
        harness.click(value_box.center());
```

with:

```rust
        let output = harness.frame(Vec::new());
        let value_box = required_box(&output);
        harness.click(value_box.center());
```

Replace:

```rust
    #[test]
    fn the_size_tab_draws_the_torque_card_in_place_of_the_label() {
        // Slice 3 spec 6.1: the card's heading, then the card's lines for the design shown,
        // each drawn once, one under the next; the label it replaced is gone. A click on the
        // heading closes the card, as one on its arrow does, and another opens it; neither edits
        // the design. Its place under the readouts is pinned by
        // the_card_sits_between_the_readouts_and_the_drawing, and its Why link by
        // the_size_tab_s_why_links_open_their_notes.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let output = harness.frame(Vec::new());
```

with:

```rust
    #[test]
    fn the_hot_low_tile_s_entry_draws_the_torque_card() {
        // Slice 3 spec 6.1 and decision C-10: the card is the Hot-low tile's Explain entry.
        // Until the tile is clicked no line of it is on screen; then its heading and its lines
        // for the design shown are drawn once each, one under the next, in the right column.
        // Another tile's entry has no card. No click edits the design.
        use crate::gui::explain::WHAT_A_CHANGE_DOES;
        use crate::gui::tiles::{NEEDS, Tile};
        let mut harness = Harness::first_view();
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
```

Replace:

```rust
        assert_eq!(count(&output, REPLACED_LABEL), 0);
        assert_eq!(count(&output, CARD_HEADING), 1);
        let lines = card_lines(&harness);
```

with:

```rust
        assert_eq!(count(&output, REPLACED_LABEL), 0);
        assert_eq!(count(&output, CARD_HEADING), 0);
        let lines = card_lines(&harness);
```

Replace:

```rust
        assert_eq!(lines.first().map(String::as_str), Some(CAPTION));
        let rect = |output: &egui::FullOutput, text: &str| {
```

with:

```rust
        assert_eq!(lines.first().map(String::as_str), Some(CAPTION));
        for line in &lines {
            assert_eq!(count(&output, line), 0, "{line}");
        }
        let needs = drawn_texts(&output)
            .into_iter()
            .find(|text| text.starts_with(NEEDS))
            .expect("the Hot-low tile says what it needs");
        harness.click(text_center(&output, &needs));
        assert_eq!(
            harness.panel.selection,
            Selection::Result(Tile::HotLow.path())
        );
        let output = harness.frame(Vec::new());
        let rect = |output: &egui::FullOutput, text: &str| {
```

Replace:

```rust
        };
        let mut above = rect(&output, CARD_HEADING);
```

with:

```rust
        };
        assert_eq!(count(&output, WHAT_A_CHANGE_DOES), 1);
        assert_eq!(count(&output, CARD_HEADING), 1);
        let mut above = rect(&output, CARD_HEADING);
```

Replace:

```rust
        let mut above = rect(&output, CARD_HEADING);
        for line in &lines {
```

with:

```rust
        let mut above = rect(&output, CARD_HEADING);
        assert!(rect(&output, WHAT_A_CHANGE_DOES).bottom() <= above.top() + 0.5);
        for line in &lines {
```

Replace:

```rust
                above.bottom() <= at.top() + 0.5,
                "{line:?} is not under the line above it"
            );
            above = at;
        }
        for open in [false, true] {
            let output = harness.frame(Vec::new());
            harness.click(text_center(&output, CARD_HEADING));
            // Past the collapse animation.
            let output = harness.frame_after(1.0, Vec::new());
            assert_eq!(count(&output, CARD_HEADING), 1, "open {open}");
            for line in &lines {
                assert_eq!(
                    count(&output, line),
                    usize::from(open),
                    "open {open}: {line}"
                );
            }
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }
```

with:

```rust
                above.bottom() <= at.top() + 0.5,
                "{line:?} is not under the line above it"
            );
            assert!(
                at.left() >= SCREEN.x - DASHBOARD_WIDTH,
                "{line:?} at {at:?}"
            );
            above = at;
        }
        // The pull-out's entry: what it is, and no card.
        let pullout = pullout_label(harness.panel.results().metal.op_temp_C);
        harness.click(centre_rect(&output, &pullout).center());
        assert_eq!(
            harness.panel.selection,
            Selection::Result(Tile::Pullout.path())
        );
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CARD_HEADING), 0);
        assert_eq!(
            count(
                &output,
                crate::gui::plain_text::help_for(Tile::Pullout.path())
            ),
            1
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }
```

Replace:

```rust
        let before = card_lines(&harness);
        let (_, value_box) = beside(&output, REQUIRED).remove(0);
        harness.click(value_box.center());
```

with:

```rust
        let before = card_lines(&harness);
        let value_box = required_box(&output);
        harness.click(value_box.center());
```

Replace:

```rust

    /// `inputs` as a design with Auto off.
    fn design_of(inputs: DesignInputs) -> Design {
        Design {
            inputs,
            sizing: SizingState::default(),
        }
    }

    /// The defaults with a design file's calibration coefficient of 1e300 (Review Focus 5): of
    /// the degenerate values the card's robustness test carries, the one whose card lines run
    /// longest, 887 characters (each number within the owner's FO-4 bound), three times the
    /// defaults' longest.
    fn huge_calibration() -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.calibration.f_cal_original = 1e300;
        inputs
    }

    /// The gui-smoke share link's design (slice 3 spec 2.1, (b)): the defaults with a 1.5 mm face
    /// gap, Auto on the magnet length.
    fn smoke_design() -> Design {
        Design {
            inputs: smoke_inputs(),
            sizing: SizingState {
                mode: SizingMode::TorqueToMagnets,
                variable: FreeVariable::AxialLength,
            },
        }
    }

    /// A harness on a screen of `size` showing the Size tab for `design` (an Auto one solved),
    /// with the shaft-clamp section paused and the Equation panel open or not, and the frame
    /// that drew the card first, the one that decided its default (slice 3 spec 8.5). The
    /// harness starts on the Geometry tab, which draws no card, and the pause, the design and
    /// the Equation panel are set there, so the default is decided for them, at the shipped
    /// pause; the last frames step past the collapse animation.
    fn card_tab(
        size: egui::Vec2,
        design: Design,
        equation_open: bool,
    ) -> (Harness, egui::FullOutput) {
        let mut harness = Harness::on_screen(size);
        harness.panel.clamps_paused = true;
        harness.panel.open_design(design);
        if equation_open {
            harness.panel.explorer.open_note(SIZE_TAB_NOTE);
        }
        harness.frame(Vec::new());
        // An Auto design: the solve runs once the design has been still for the debounce.
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.card_start, None, "no card drawn yet");
        let deciding = harness.click_text(CentreView::SizeAndTorque.label());
        assert!(
            harness.panel.card_start.is_some(),
            "the click's frame drew the card"
        );
        harness.frame_after(1.0, Vec::new());
        harness.frame(Vec::new());
        (harness, deciding)
    }

    /// Whether the torque card is open, as egui keeps it.
    fn card_open(harness: &Harness) -> bool {
        card_open_in(&harness.ctx)
    }

    /// Whether the torque card drawn in `ctx` is open, as egui keeps it: a panel drawn without
    /// a [`Harness`], as inside an `egui::Window`, reads it here.
    fn card_open_in(ctx: &egui::Context) -> bool {
        egui::collapsing_header::CollapsingState::load(ctx, egui::Id::new(TORQUE_CARD_ID))
            .expect("the card has been drawn")
            .is_open()
    }

    /// The card's Why link in a frame: its text rect and where a click lands on its last
    /// character. Of the Size tab's links to how the coupling makes torque, the card's is the
    /// one at or right of the drawing's left edge (the other two are on the inputs side).
    fn card_link(output: &egui::FullOutput) -> (egui::Rect, egui::Pos2) {
        let text = why_note_text(reviewed_note(SIZE_TAB_NOTE).expect("the note is reviewed"));
        let left = drawing_rect(output).left();
        let rects: Vec<egui::Rect> = text_rects(output, &text)
            .into_iter()
            .filter(|rect| rect.left() >= left)
            .collect();
        let at: Vec<egui::Pos2> = last_characters(output, &text)
            .into_iter()
            .filter(|at| at.x >= left)
            .collect();
        assert_eq!((rects.len(), at.len()), (1, 1), "{rects:?}");
        (rects[0], at[0])
    }

    /// The vertical item spacing in the harness's style (egui's default) [points]: the gap
    /// between the card's header and its body, and between the card and the drawing.
    fn item_spacing(harness: &Harness) -> f32 {
        harness.ctx.style().spacing.item_spacing.y
    }

    /// Whether `drawing` keeps the drawing's floor, [`MIN_DRAWING_HEIGHT`] (slice 3 spec 6.2), to
    /// within 0.01 point: a card body capped to the room over the floor leaves the drawing that
    /// tall up to rounding in the cap's arithmetic. The card tests' one check of the floor.
    fn keeps_its_floor(drawing: egui::Rect) -> bool {
        drawing.height() >= MIN_DRAWING_HEIGHT - 0.01
    }

    /// The room at the torque card in `output` [points]: from its heading's top to the bottom
    /// of the heading's clip, the centre view's region (`confined` clips to it). It is the
    /// height that rule 3 reads at the card (`ui.available_height()`).
    fn room_at_card(output: &egui::FullOutput) -> f32 {
        let heading = text_rect(output, CARD_HEADING).expect("the heading");
        text_clip_rect(output, CARD_HEADING).bottom() - heading.top()
    }

    /// The torque card's drawn height in `output` [points]: from its heading's top to the
    /// drawing's top, less the item spacing between the card and the drawing.
    fn drawn_card_height(harness: &Harness, output: &egui::FullOutput) -> f32 {
        let heading = text_rect(output, CARD_HEADING).expect("the heading");
        drawing_rect(output).top() - heading.top() - item_spacing(harness)
    }

    /// The card's pre-measure of `lines` ([`card_open_height`]) and the drawing's floor
    /// ([`drawing_floor`]), measured as the panel measures them on a deciding frame, in a scratch
    /// frame of a region `width` points wide in the harness's style (egui's default).
    fn scratch_measure(width: f32, lines: &[String]) -> (f32, f32) {
        let mut measured = (f32::NAN, f32::NAN);
        let region = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, 4000.0));
        crate::gui::test_support::region_frame(
            &egui::Context::default(),
            egui::vec2(width + 100.0, 4000.0),
            region,
            |ui| measured = (card_open_height(ui, lines), drawing_floor(ui)),
        );
        measured
    }

    /// What rule 3 leaves for the torque card in `output` (slice 3 spec 6.2) [points], as the
    /// panel measures it on a deciding frame: the room at the card ([`room_at_card`]), less the
    /// open card's pre-measure of the design shown's lines in the centre region's width (the
    /// drawing's; [`scratch_measure`]), the item spacing before the drawing and the drawing's
    /// floor. Rule 3 alone opens the card where it is not negative. This is the tests' one copy
    /// of the rule in [`MagcouplingPanel::card_default`].
    fn rule_three_margin(harness: &Harness, output: &egui::FullOutput) -> f32 {
        let lines = card_lines(harness);
        let (open_height, floor) = scratch_measure(drawing_rect(output).width(), &lines);
        room_at_card(output) - open_height - item_spacing(harness) - floor
    }

    /// The height of the label the card replaced, laid out as it was under the readouts (a
    /// wrapped label at the region's width `width`) in the harness's style [points].
    fn replaced_label_height(harness: &Harness, width: f32) -> f32 {
        let font = harness.ctx.style().text_styles[&egui::TextStyle::Body].clone();
        harness
            .ctx
            .fonts(|f| f.layout(REPLACED_LABEL.to_owned(), font, egui::Color32::WHITE, width))
            .size()
            .y
    }

    #[test]
    fn the_card_sits_between_the_readouts_and_the_drawing() {
        // Slice 3 spec 8.5 test 19: at 1280 x 800, at the defaults and at the smoke design
        // (Auto on the magnet length), the card's heading is under the Pull-out readout and
        // above the drawing; the card starts open, every one of its lines is drawn, the drawing
        // keeps its 160 points and the label the card replaced is gone.
        for design in [design_of(DesignInputs::default()), smoke_design()] {
            let (mut harness, _) = card_tab(egui::vec2(1280.0, 800.0), design, false);
            let output = harness.frame(Vec::new());
            let case = format!("{:?}", harness.panel.sizing);
            assert!(card_open(&harness), "{case}");
            let heading = text_rect(&output, CARD_HEADING).expect("the heading");
            let label = pullout_label(harness.panel.results().metal.op_temp_C);
            let pull_out = centre_rect(&output, &label);
            let drawing = drawing_rect(&output);
            assert!(
                pull_out.bottom() <= heading.top(),
                "{case}: {pull_out:?} {heading:?}"
            );
            assert!(
                heading.bottom() <= drawing.top(),
                "{case}: {heading:?} {drawing:?}"
            );
            let lines = card_lines(&harness);
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{case}: {line}");
            }
            assert!(keeps_its_floor(drawing), "{case}: {drawing:?}");
            assert_eq!(count(&output, REPLACED_LABEL), 0, "{case}");
        }
    }

    #[test]
    fn the_card_starts_closed_in_a_narrow_window_and_the_drawing_gains() {
        // Slice 3 spec 8.5 test 20 (rule 2, T-10): at 930 x 800, with the Equation panel closed,
        // the card starts closed at the defaults and at short magnets, whose one-line form rule
        // 3 alone would open; closed, it is no taller than the label it replaced, laid out at the
        // region's width, so the drawing is at least as tall as before the card. At 1280 x 800
        // short magnets start open, and the card closed by its heading is again no taller than
        // the label there.
        let no_taller_than_the_label =
            |harness: &Harness, output: &egui::FullOutput, what: &str| {
                let closed = drawn_card_height(harness, output);
                let label = replaced_label_height(harness, drawing_rect(output).width());
                assert!(
                    closed <= label,
                    "{what}: closed {closed} against the label's {label}"
                );
            };
        for inputs in [DesignInputs::default(), short_magnets()] {
            let one_line = inputs == short_magnets();
            let (mut harness, _) = card_tab(egui::vec2(930.0, 800.0), design_of(inputs), false);
            let output = harness.frame(Vec::new());
            let region = drawing_rect(&output).width();
            assert!(!card_open(&harness), "one-line form {one_line}");
            assert!(region < CARD_OPEN_MIN_WIDTH, "{region}");
            no_taller_than_the_label(&harness, &output, &format!("one-line form {one_line}"));
            if one_line {
                // Rule 3 alone would open the one-line form here: the width rule closes it.
                let margin = rule_three_margin(&harness, &output);
                assert!(margin >= 0.0, "rule 3 leaves {margin}");
            }
        }
        let (harness, _) = card_tab(egui::vec2(1280.0, 800.0), design_of(short_magnets()), false);
        assert!(card_open(&harness), "the one-line form in a wide window");
        let (mut harness, _) = card_tab(
            egui::vec2(1280.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        let output = harness.frame(Vec::new());
        harness.click(text_center(&output, CARD_HEADING));
        let output = harness.frame_after(1.0, Vec::new());
        assert!(!card_open(&harness));
        no_taller_than_the_label(&harness, &output, "closed by its heading at 1280 x 800");
        // At 930 x 800 the header row wraps (spec 6.1): the card's Why link continues under the
        // heading inside the centre region, and a click on its last character opens how the
        // coupling makes torque (test 21 clicks it in a wide window).
        let (mut harness, _) = card_tab(
            egui::vec2(930.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        let output = harness.frame(Vec::new());
        let (link, at) = card_link(&output);
        let heading = text_rect(&output, CARD_HEADING).expect("the heading");
        assert!(
            link.right() <= drawing_rect(&output).right(),
            "{link:?} past {:?}",
            drawing_rect(&output)
        );
        assert!(
            link.bottom() > heading.bottom(),
            "the link wraps under the heading"
        );
        harness.click(at);
        assert_eq!(harness.panel.explorer.note(), Some(SIZE_TAB_NOTE));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_card_starts_closed_when_it_would_take_the_drawing_under_its_floor() {
        // Slice 3 spec 8.5 test 20a (rule 3 deciding alone): at 1080 x 745, with the Equation
        // panel closed and the welcome strip shown. On the deciding frame the view has the
        // Equation panel closed and the centre region (the drawing's width: the drawing fills
        // it) is at least CARD_OPEN_MIN_WIDTH wide, so rules 1 and 2 pass. No back iron's open
        // card would take the drawing under its floor: it starts closed; the one-line form at
        // short magnets leaves room: it starts open.
        for (inputs, open) in [(no_back_iron(), false), (short_magnets(), true)] {
            let (harness, deciding) = card_tab(egui::vec2(1080.0, 745.0), design_of(inputs), false);
            let (view, default) = harness.panel.card_start.expect("decided");
            assert!(!view.equation_open);
            let region = drawing_rect(&deciding).width();
            assert!(region >= CARD_OPEN_MIN_WIDTH, "{region}");
            assert_eq!(count(&deciding, WELCOME), 1);
            assert_eq!((default, card_open(&harness)), (open, open));
            assert_eq!(harness.panel.card_choice, None);
        }
    }

    #[test]
    fn the_card_s_default_opens_at_each_rule_s_boundary() {
        // T6 M-2 (the fold-in): spec 6.2's rules say "at least", so each opens the card at its
        // boundary. Rule 2: a centre region 400 points wide (T-10) opens it, one 399.99 wide
        // closes it. Rule 3: a region exactly as tall as the open card's pre-measure, the item
        // spacing before the drawing and the drawing's floor opens it, one 0.01 point shorter
        // closes it. Rule 1: the Equation panel open closes it where rules 2 and 3 pass.
        // card_default decides in a scratch frame of the case's width and height (a region at
        // the origin, so its available height is that height exactly), each case on a new
        // panel, so no stored default answers.
        let defaults = DesignInputs::default();
        let lines = Consequences::of(&defaults, &compute_all(&defaults), None, true)
            .card
            .lines(None);
        let decide = |equation_open: bool, width: f32, height: f32| {
            let mut panel = MagcouplingPanel::new();
            let card = CardFrame {
                view: CardView {
                    equation_open,
                    screen_width: 1280,
                    screen_height: 800,
                },
                region_width: width,
            };
            let region = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, height));
            let mut open = None;
            crate::gui::test_support::region_frame(
                &egui::Context::default(),
                egui::vec2(width + 100.0, 4000.0),
                region,
                |ui| open = Some(panel.card_default(ui, card, &lines)),
            );
            open.expect("decided")
        };
        // Rule 2, with room to spare for rule 3.
        assert!(decide(false, 400.0, 4000.0), "400 points wide");
        assert!(!decide(false, 399.99, 4000.0), "399.99 points wide");
        // Rule 3, in a region wide enough for rule 2.
        let width = 600.0;
        let (open_height, floor) = scratch_measure(width, &lines);
        let spacing = egui::Context::default().style().spacing.item_spacing.y;
        let exact = open_height + spacing + floor;
        // The room the panel reads in a region `exact` tall is the open height to the bit, so
        // the case below sits on the boundary.
        assert_eq!(exact - spacing - floor, open_height, "{exact}");
        assert!(decide(false, width, exact), "exactly the room: {exact}");
        assert!(
            !decide(false, width, exact - 0.01),
            "0.01 point short of {exact}"
        );
        // Rule 1 closes it where rules 2 and 3 pass.
        assert!(!decide(true, width, 4000.0), "the Equation panel open");
    }

    #[test]
    fn the_card_starts_closed_while_the_equation_panel_is_open() {
        // Slice 3 spec 8.5 test 21 (rule 1): at 1280 x 800 the card starts closed while the
        // Equation panel is open, and so it does on a screen tall enough for the open card above
        // the panel. Started open, a click on the card's own Why link opens the panel, and with
        // no user choice the card closes (decision 4A); the panel closing opens it again.
        let wide = egui::vec2(1280.0, 800.0);
        let (harness, _) = card_tab(wide, design_of(DesignInputs::default()), true);
        assert!(!card_open(&harness));
        // On a tall screen the open card fits above the Equation panel, so rules 2 and 3 alone
        // would open it there: rule 1 closes it on its own.
        let (mut harness, _) = card_tab(
            egui::vec2(1280.0, 2000.0),
            design_of(DesignInputs::default()),
            true,
        );
        assert!(!card_open(&harness));
        let output = harness.frame(Vec::new());
        let region = drawing_rect(&output).width();
        assert!(region >= CARD_OPEN_MIN_WIDTH, "{region}");
        let margin = rule_three_margin(&harness, &output);
        assert!(margin >= 0.0, "rule 3 leaves {margin}");
        let (mut harness, _) = card_tab(wide, design_of(DesignInputs::default()), false);
        assert!(card_open(&harness));
        let output = harness.frame(Vec::new());
        let (_, at) = card_link(&output);
        harness.click(at);
        assert_eq!(harness.panel.explorer.note(), Some(SIZE_TAB_NOTE));
        harness.frame_after(1.0, Vec::new());
        assert!(
            !card_open(&harness),
            "the Equation panel's opening closes it"
        );
        assert_eq!(harness.panel.card_choice, None);
        harness.panel.explorer.close();
        harness.frame_after(1.0, Vec::new());
        assert!(card_open(&harness), "and its closing opens it again");
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_user_s_open_or_close_wins_for_the_session() {
        // Slice 3 spec 8.5 test 22: at 930 x 800 the card starts closed; a click on its heading
        // opens it, and it stays open after a design edit, a resize to 1280 x 800, the Equation
        // panel opening, Reset all and a load. A click on its arrow then closes it and one on
        // its heading opens it again: each is the user's choice.
        let (mut harness, _) = card_tab(
            egui::vec2(930.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        assert!(!card_open(&harness));
        let output = harness.frame(Vec::new());
        harness.click(text_center(&output, CARD_HEADING));
        harness.frame_after(1.0, Vec::new());
        assert_eq!(harness.panel.card_choice, Some(true));
        // Open by the user's choice, and its body capped so the drawing keeps its 160 points,
        // the Equation panel's opening at 1280 x 800 included.
        let still_open = |harness: &mut Harness, what: &str| {
            harness.frame_after(1.0, Vec::new());
            let output = harness.frame(Vec::new());
            assert!(card_open(harness), "{what}");
            assert_eq!(harness.panel.card_choice, Some(true), "{what}");
            let drawing = drawing_rect(&output);
            assert!(keeps_its_floor(drawing), "{what}: {drawing:?}");
        };
        still_open(&mut harness, "the click");
        harness.panel.inputs.metal.face_gap_mm = 2.0;
        still_open(&mut harness, "a design edit");
        harness.screen = egui::vec2(1280.0, 800.0);
        still_open(&mut harness, "a resize");
        harness.panel.explorer.open_note(SIZE_TAB_NOTE);
        still_open(&mut harness, "the Equation panel opening");
        harness.panel.explorer.close();
        harness.click_text(RESET_ALL);
        still_open(&mut harness, "Reset all");
        harness
            .panel
            .load_design_file(&design_to_json(&design_of(with_poles(8))))
            .expect("the file loads");
        still_open(&mut harness, "a load");
        // The arrow, half an indent left of the heading's text, closes it; the heading opens it.
        let output = harness.frame(Vec::new());
        let heading = text_rect(&output, CARD_HEADING).expect("the heading");
        let indent = harness.ctx.style().spacing.indent;
        harness.click(egui::pos2(
            heading.left() - indent / 2.0,
            heading.center().y,
        ));
        harness.frame_after(1.0, Vec::new());
        assert_eq!(harness.panel.card_choice, Some(false));
        assert!(!card_open(&harness));
        let output = harness.frame(Vec::new());
        harness.click(text_center(&output, CARD_HEADING));
        harness.frame_after(1.0, Vec::new());
        assert_eq!(harness.panel.card_choice, Some(true));
        assert!(card_open(&harness));
    }

    #[test]
    fn the_card_is_decided_again_when_the_window_is_resized_with_no_user_choice() {
        // Slice 3 spec 6.2: with no user choice, the default is decided again when the view
        // changes, and a resize of the window is one of its triggers (test 22 resizes only after
        // the user's click, which wins). Started closed at 930 x 800 (rule 2), the card opens
        // when the window widens to 1280 x 800 and closes when it narrows again. Then the height
        // alone changes at 1080 points wide, where rules 1 and 2 pass and rule 3 decides: open at
        // 1080 x 800, closed at 1080 x 700, where the defaults' open card would take the drawing
        // under its floor, and open again at 800. Each time the view it was decided for is the
        // new screen, and no user choice is recorded.
        let (mut harness, _) = card_tab(
            egui::vec2(930.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        assert!(!card_open(&harness));
        for (width, height, open) in [
            (1280.0, 800.0, true),
            (930.0, 800.0, false),
            (1080.0, 800.0, true),
            (1080.0, 700.0, false),
            (1080.0, 800.0, true),
        ] {
            harness.screen = egui::vec2(width, height);
            harness.frame(Vec::new());
            harness.frame_after(1.0, Vec::new());
            let output = harness.frame(Vec::new());
            let case = format!("{width} x {height}");
            let (view, default) = harness.panel.card_start.expect("decided");
            assert_eq!(
                (view.screen_width, view.screen_height, default),
                (width as u32, height as u32, open),
                "{case}"
            );
            assert_eq!(card_open(&harness), open, "{case}");
            assert_eq!(harness.panel.card_choice, None, "{case}");
            // The rule that decides (the Equation panel stays closed, so rule 1 passes): rule 2
            // in a region under 400 points, else rule 3.
            let region = drawing_rect(&output).width();
            if region < CARD_OPEN_MIN_WIDTH {
                assert!(!open, "{case}: rule 2 closes it in {region} points");
            } else {
                let margin = rule_three_margin(&harness, &output);
                assert_eq!(margin >= 0.0, open, "{case}: rule 3 leaves {margin}");
            }
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_card_never_opens_or_closes_under_a_drag() {
        // Slice 3 spec 8.5 test 23: the card's default is decided for the view (the Equation
        // panel and the screen), never on a design change. At 1280 x 800, from the defaults,
        // open: twenty drag frames on the gap slider; the Outside diameter dragged down across
        // the fit edge (the exploration line appears above the card); Required dragged across
        // hot-low; an assumption changed, so the assumptions banner takes a line of the header
        // and the region loses that height. Through each, on every frame of a drag and after it,
        // the card stays open and the view it was decided for does not change.
        let (mut harness, _) = card_tab(
            egui::vec2(1280.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        assert!(card_open(&harness));
        let start = harness.panel.card_start;
        // The card as it was decided: open, for the view it was decided for, no user choice.
        let as_decided = |harness: &Harness, what: &str| {
            assert!(card_open(harness), "{what}");
            assert_eq!(harness.panel.card_start, start, "{what}");
            assert_eq!(harness.panel.card_choice, None, "{what}");
        };
        let holds = |harness: &mut Harness, what: &str| {
            harness.frame_after(1.0, Vec::new());
            let output = harness.frame(Vec::new());
            as_decided(harness, what);
            output
        };
        // A drag on a slider's rail: press, moves, release, a frame each, the card as decided on
        // every one of them (spec 8.5: through the drag frames), so a flip that a later frame
        // undoes fails here, not only one that outlasts the drag.
        let drag = |harness: &mut Harness, from: egui::Pos2, moves: &[egui::Vec2], what: &str| {
            let mut at = from;
            let mut events = vec![egui::Event::PointerMoved(from), primary_button(from, true)];
            for &step in moves {
                at += step;
                events.push(egui::Event::PointerMoved(at));
            }
            events.push(primary_button(at, false));
            for (frame, event) in events.into_iter().enumerate() {
                harness.frame(vec![event]);
                as_decided(harness, &format!("{what}, drag frame {frame}"));
            }
        };
        let gap = harness.size_widget(GAP).rect;
        let to_and_fro: Vec<egui::Vec2> = (0..20)
            .map(|i| egui::vec2(if i % 2 == 0 { 3.0 } else { -3.0 }, 0.0))
            .collect();
        let before = harness.panel.inputs().metal.face_gap_mm;
        drag(
            &mut harness,
            gap.left_center() + egui::vec2(20.0, 0.0),
            &to_and_fro,
            "the gap",
        );
        assert_ne!(
            harness.panel.inputs().metal.face_gap_mm,
            before,
            "the drag moved the gap"
        );
        holds(&mut harness, "twenty gap drag frames");
        let diameter = harness.size_widget(DIAMETER).rect;
        drag(
            &mut harness,
            diameter.left_center() + egui::vec2(2.0, 0.0),
            &[egui::vec2(1.0, 0.0), egui::vec2(-1.0, 0.0)],
            "the diameter",
        );
        let output = holds(&mut harness, "the diameter across the fit edge");
        let fit = format!("{EXPLORATION_ONLY}{SIDE_DOES_NOT_FIT} at this diameter.");
        assert_eq!(count(&output, &fit), 1, "the exploration line shows");
        let required = harness
            .ctx
            .read_response(
                harness
                    .panel
                    .requirement_widget
                    .expect("the Required slider"),
            )
            .expect("drawn")
            .rect;
        let hot_low = harness.panel.results().metal.torque_hot_low_Nm;
        assert!(harness.panel.inputs().metal.required_min_Nm > hot_low);
        drag(
            &mut harness,
            required.left_center() + egui::vec2(2.0, 0.0),
            &[egui::vec2(1.0, 0.0), egui::vec2(-1.0, 0.0)],
            "Required",
        );
        assert!(
            harness.panel.inputs().metal.required_min_Nm < hot_low,
            "across hot-low"
        );
        holds(&mut harness, "Required across hot-low");
        harness.panel.inputs.metal.variation = 0.2;
        let output = holds(&mut harness, "the assumptions banner");
        let banner = assumptions_banner(harness.panel.inputs(), true).expect("a banner");
        assert_eq!(count(&output, &banner), 1);
        // At 1080 x 800, from the defaults, open with no user choice: no back iron widens the
        // dashboard to its content and the centre region narrows (428.0 to 425.7 points in the
        // harness's style), where rule 3 would close the card; it stays open, its view unchanged.
        let (mut harness, _) = card_tab(
            egui::vec2(1080.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        assert!(card_open(&harness));
        let start = harness.panel.card_start;
        let wide = drawing_rect(&harness.frame(Vec::new())).width();
        harness.panel.inputs.coupling.backiron = 0;
        for _ in 0..6 {
            harness.frame(Vec::new());
        }
        let output = harness.frame_after(1.0, Vec::new());
        let narrow = drawing_rect(&output).width();
        assert!(narrow < wide, "the region narrows: {wide} -> {narrow}");
        assert!(card_open(&harness));
        assert_eq!(harness.panel.card_start, start);
        let margin = rule_three_margin(&harness, &output);
        assert!(
            margin < 0.0,
            "rule 3 at no back iron would close it: it leaves {margin}"
        );
    }

    #[test]
    fn reset_all_and_a_load_keep_the_card_s_default_with_no_user_choice() {
        // Slice 3 spec 6.2: the default decided for the view is kept by Reset all and every load,
        // as the user's choice is (test 22). At 1080 x 800 no back iron's open card would take
        // the drawing under its floor, so the card starts closed; Reset all (the defaults) and
        // then a load of 8 magnets, whose open cards would leave the floor there, keep it
        // closed. Started open at the defaults, a load of no back iron keeps it open. Each time
        // the view it was decided for is unchanged and no user choice is recorded.
        let screen = egui::vec2(1080.0, 800.0);
        // The card as `start` decided it after `what`, past the collapse animation, and what
        // rule 3 would leave it at the design now shown (`rule_three_margin`).
        let kept = |harness: &mut Harness, start: Option<(CardView, bool)>, what: &str| {
            harness.frame_after(1.0, Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(harness.panel.card_start, start, "{what}");
            assert_eq!(
                Some(card_open(harness)),
                start.map(|(_, open)| open),
                "{what}"
            );
            assert_eq!(harness.panel.card_choice, None, "{what}");
            rule_three_margin(harness, &output)
        };
        let (mut harness, _) = card_tab(screen, design_of(no_back_iron()), false);
        let start = harness.panel.card_start;
        assert_eq!(start.map(|(_, open)| open), Some(false), "no back iron");
        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        let margin = kept(&mut harness, start, "Reset all");
        assert!(margin >= 0.0, "rule 3 alone would open it: {margin}");
        harness
            .panel
            .load_design_file(&design_to_json(&design_of(with_poles(8))))
            .expect("the file loads");
        let margin = kept(&mut harness, start, "a load of 8 magnets");
        assert!(margin >= 0.0, "rule 3 alone would open it: {margin}");
        let (mut harness, _) = card_tab(screen, design_of(DesignInputs::default()), false);
        let start = harness.panel.card_start;
        assert_eq!(start.map(|(_, open)| open), Some(true), "the defaults");
        harness
            .panel
            .load_design_file(&design_to_json(&design_of(no_back_iron())))
            .expect("the file loads");
        let margin = kept(&mut harness, start, "a load of no back iron");
        assert!(margin < 0.0, "rule 3 alone would close it: {margin}");
    }

    #[test]
    fn the_card_body_scrolls_above_the_drawing_s_floor() {
        // Slice 3 spec 8.5 test 24: opened by a click on its heading at 930 x 800, where rule 2
        // starts it closed and its body is taller than the room, the card's body scrolls and the
        // drawing keeps its 160 points, at the defaults and at no back iron (a taller card). So
        // it does at 1280 x 800 at a design file's calibration coefficient of 1e300 (Review Focus
        // 5), whose lines are the longest of the card's robustness values (`huge_calibration`):
        // there rules 1 and 2 pass, and rule 3, reading the open card's height before it is
        // drawn, starts it closed.
        // Opened by a click on its heading, the body is capped, so its last row starts out of
        // view (not painted, or cut by the body's clip); the mouse wheel over the body brings it
        // into view, whole, and the drawing keeps its floor throughout.
        let opened_scrolls = |harness: &mut Harness, case: &str| {
            let output = harness.frame(Vec::new());
            assert!(!card_open(harness), "{case}: starts closed");
            harness.click(text_center(&output, CARD_HEADING));
            let output = harness.frame_after(1.0, Vec::new());
            assert!(card_open(harness), "{case}");
            let drawing = drawing_rect(&output);
            assert!(keeps_its_floor(drawing), "{case}: {drawing:?}");
            let last = card_lines(harness).pop().expect("a row");
            assert!(
                text_rect(&output, &last)
                    .is_none_or(|row| row.bottom() > text_clip_rect(&output, &last).bottom() + 1.0),
                "{case}: nothing to scroll: {last}"
            );
            let body = text_center(&output, CAPTION);
            harness.frame(vec![egui::Event::PointerMoved(body)]);
            harness.frame(vec![egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, -5000.0),
                modifiers: egui::Modifiers::NONE,
            }]);
            // egui spreads a wheel step over several frames.
            let mut output = harness.frame(Vec::new());
            for _ in 0..20 {
                output = harness.frame(Vec::new());
            }
            let row = text_rect(&output, &last).expect("the last row in view");
            let clip = text_clip_rect(&output, &last);
            assert!(clip.contains_rect(row), "{case}: {row:?} cut by {clip:?}");
            let drawing = drawing_rect(&output);
            assert!(keeps_its_floor(drawing), "{case}: {drawing:?}");
            assert!(card_open(harness), "{case}");
        };
        let narrow = egui::vec2(930.0, 800.0);
        let (mut harness, _) = card_tab(narrow, design_of(DesignInputs::default()), false);
        opened_scrolls(&mut harness, "the defaults at 930 x 800");
        let (mut harness, _) = card_tab(narrow, design_of(no_back_iron()), false);
        opened_scrolls(&mut harness, "no back iron at 930 x 800");
        let (mut harness, deciding) = card_tab(
            egui::vec2(1280.0, 800.0),
            design_of(huge_calibration()),
            false,
        );
        let longest = card_lines(&harness)
            .iter()
            .map(|line| line.chars().count())
            .max()
            .expect("lines");
        assert!(longest > 800, "{longest}");
        let region = drawing_rect(&deciding).width();
        assert!(region >= CARD_OPEN_MIN_WIDTH, "{region}");
        let output = harness.frame(Vec::new());
        let margin = rule_three_margin(&harness, &output);
        assert!(margin < 0.0, "rule 3 leaves {margin}");
        opened_scrolls(
            &mut harness,
            "a calibration coefficient of 1e300 at 1280 x 800",
        );
    }

    #[test]
    fn the_card_s_measured_height_is_the_drawn_height() {
        // Slice 3 spec 8.5 test 25: the card's pre-measure, card_open_height in a scratch frame
        // of the centre region's width, equals its drawn open height (the heading's top to the
        // drawing's, less an item spacing) within a point, at the defaults and the smoke design,
        // at 1280 x 800, where its header is one row, and at 930 x 1024, where the row wraps and
        // a click on the heading opens the card (rule 2 starts it closed); the bodies are under
        // their caps there. So it does at 1280 x 1024 at a design file's calibration coefficient
        // of 1e300, whose lines are the longest of the card's robustness values (Review Focus 5,
        // `huge_calibration`). At 930 x 800 the bodies are over the cap: the drawn body is the
        // cap, the room at the card less the header, two item spacings and the drawing's floor.
        let body_row = |harness: &Harness| {
            let font = harness.ctx.style().text_styles[&egui::TextStyle::Body].clone();
            harness.ctx.fonts(|f| f.row_height(&font))
        };
        let defaults = || design_of(DesignInputs::default());
        // Each case: the screen, the design, and whether the body is over its cap there.
        for (size, design, capped) in [
            (egui::vec2(1280.0, 800.0), defaults(), false),
            (egui::vec2(1280.0, 800.0), smoke_design(), false),
            (egui::vec2(930.0, 1024.0), defaults(), false),
            (egui::vec2(930.0, 1024.0), smoke_design(), false),
            (
                egui::vec2(1280.0, 1024.0),
                design_of(huge_calibration()),
                false,
            ),
            (egui::vec2(930.0, 800.0), defaults(), true),
            (egui::vec2(930.0, 800.0), smoke_design(), true),
        ] {
            let (mut harness, _) = card_tab(size, design, false);
            if !card_open(&harness) {
                let output = harness.frame(Vec::new());
                harness.click(text_center(&output, CARD_HEADING));
                harness.frame_after(1.0, Vec::new());
            }
            let output = harness.frame(Vec::new());
            let case = format!("{size:?} {:?}", harness.panel.sizing.mode);
            let spacing = item_spacing(&harness);
            let heading = text_rect(&output, CARD_HEADING).expect("the heading");
            let drawing = drawing_rect(&output);
            let drawn = drawn_card_height(&harness, &output);
            let header = heading.bottom().max(card_link(&output).0.bottom()) - heading.top();
            let body = drawn - header - spacing;
            let lines = card_lines(&harness);
            let (measured, floor) = scratch_measure(drawing.width(), &lines);
            let room = room_at_card(&output);
            let cap = (room - header - 2.0 * spacing - floor).max(body_row(&harness));
            if capped {
                assert!(measured > drawn + 1.0, "{case}: the body is capped");
                assert!((body - cap).abs() <= 1.0, "{case}: body {body}, cap {cap}");
            } else {
                assert!(body < cap, "{case}: body {body} under the cap {cap}");
                assert!(
                    (measured - drawn).abs() <= 1.0,
                    "{case}: measured {measured}, drawn {drawn}"
                );
            }
        }
    }

    #[test]
```

with:

```rust

    #[test]
```

Replace:

```rust
        // the what-moves-what lines wrap inside the inputs side (its 320 points), one under the
        // next, and the torque readouts stay in the centre region, left of the dashboard.
        let mut harness = size_tab(egui::vec2(930.0, 1024.0), with_poles(8));
```

with:

```rust
        // the what-moves-what lines wrap inside the inputs side (its 320 points), one under the
        // next, the Required row is that side's too, and the tiles stay in the centre region,
        // left of the right column.
        let mut harness = size_tab(egui::vec2(930.0, 1024.0), with_poles(8));
```

Replace:

```rust
        let label = pullout_label(results.metal.op_temp_C);
        for text in [REQUIRED, HOT_LOW, gap.as_str(), label.as_str()] {
            let at = centre_rect(&output, text);
```

with:

```rust
        let label = pullout_label(results.metal.op_temp_C);
        let pieces = crate::gui::tiles::gap_pieces(&gap);
        let texts = [HOT_LOW, label.as_str()].into_iter();
        for text in texts.chain(pieces.iter().map(String::as_str)) {
            let at = centre_rect(&output, text);
```

Replace:

```rust
                "{text:?} at {at:?} runs out of the centre ({inputs_right} to {dashboard_left})"
            );
        }
    }

    #[test]
```

with:

```rust
                "{text:?} at {at:?} runs out of the centre ({inputs_right} to {dashboard_left})"
            );
        }
        let required = rect(REQUIRED);
        assert!(
            required.left() >= inputs_left - 1.0 && required.right() <= inputs_right + 1.0,
            "{required:?}"
        );
    }

    #[test]
```

Replace:

```rust
            let hot_low = results.metal.torque_hot_low_Nm;
            texts.push(if predicts(&results, hot_low) {
                torque_gap_text(hot_low, shown.metal.required_min_Nm).0
            } else {
```

with:

```rust
            let hot_low = results.metal.torque_hot_low_Nm;
            if predicts(&results, hot_low) {
                let gap = torque_gap_text(hot_low, shown.metal.required_min_Nm).0;
                texts.extend(crate::gui::tiles::gap_pieces(&gap));
            } else {
```

Replace:

```rust
            } else {
                NO_PREDICTION.to_owned()
            });
            let all = drawn_texts(&output);
```

with:

```rust
            } else {
                texts.push(NO_PREDICTION.to_owned());
            }
            let all = drawn_texts(&output);
```

Replace:

```rust
    fn torque_to_magnets_shows_the_solved_design() {
        let mut harness = Harness::on_screen(TALL);
        let output = harness.size();
```

with:

```rust
    fn torque_to_magnets_shows_the_solved_design() {
        let mut harness = Harness::on_screen(TALL).on_results();
        let output = harness.size();
```

Replace:

```rust
    fn an_unreachable_target_shows_the_best_value_and_its_overshoot() {
        let mut harness = Harness::on_screen(TALL);
        harness.size();
        harness.panel.inputs.metal.required_min_Nm = 50.0;
```

with:

```rust
    fn an_unreachable_target_shows_the_best_value_and_its_overshoot() {
        let mut harness = Harness::on_screen(TALL);
        harness.select(SPACE_CLAIM_CHECK);
        harness.size();
        harness.panel.inputs.metal.required_min_Nm = 50.0;
```

Replace:

```rust
    #[test]
    fn hovering_a_dashboard_value_shows_its_equation() {
        use crate::gui::readouts::OPEN_HINT;
```

with:

```rust
    #[test]
    fn hovering_a_headline_value_shows_its_equation() {
        use crate::gui::readouts::OPEN_HINT;
```

Replace:

```rust
        use crate::gui::readouts::OPEN_HINT;
        let mut harness = Harness::new();
        tooltips_at_once(&harness.ctx);
```

with:

```rust
        use crate::gui::readouts::OPEN_HINT;
        let mut harness = Harness::new().on_results();
        tooltips_at_once(&harness.ctx);
```

Replace:

```rust
    fn hovering_a_value_marks_its_terms_where_their_values_are_shown() {
        // D_cup = 2 (r_corner + t_wall): hovering the cup OD on the dashboard frames the Key
        // design row of the wall in its term's colour, from the next frame, and its tooltip
        // draws t_wall in that colour; moving the pointer away clears every mark.
        let mut harness = Harness::new();
        tooltips_at_once(&harness.ctx);
```

with:

```rust
    fn hovering_a_value_marks_its_terms_where_their_values_are_shown() {
        // D_cup = 2 (r_corner + t_wall): hovering the cup OD in the Results tab's Headline group
        // frames the Key design row of the wall in its term's colour, from the next frame, and
        // its tooltip draws t_wall in that colour; moving the pointer away clears every mark.
        let mut harness = Harness::new().on_results();
        tooltips_at_once(&harness.ctx);
```

Replace:

```rust
        tooltips_at_once(&harness.ctx);
        // The dashboard settles its width on the second frame, and the check guides above the
        // cup OD wrap to it: measure where the row rests.
        harness.frame(Vec::new());
```

with:

```rust
        tooltips_at_once(&harness.ctx);
        // The right column settles its width on the second frame: measure where the row rests.
        harness.frame(Vec::new());
```

Replace:

```rust
        use crate::gui::explorer::TERMS;
        let mut harness = Harness::new();
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
```

with:

```rust
        use crate::gui::explorer::TERMS;
        let mut harness = Harness::new().on_results();
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
```

Replace:

```rust
            }
            // Slice 3 test 21: the Size tab's torque card closed, as the Equation panel's opening
            // leaves it, and opened by the user; neither paints into the panel.
            for choice in [None, Some(true)] {
                harness.panel.card_choice = choice;
                harness.panel.centre = CentreView::SizeAndTorque;
                harness.frame_after(1.0, Vec::new());
                let output = harness.frame(Vec::new());
                let case = format!("{size:?}, dragged up {dragged}, the card's choice {choice:?}");
                if let Some(overlap) = equation_panel_overlap(&harness.ctx, &output) {
                    failures.push(format!("{case}: {overlap}"));
                }
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
```

with:

```rust
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
```

Replace:

```rust
            }
            // The torque card opened by the user's choice, past the collapse animation (each
            // frame here is egui's predicted 1/60 s later); the views above drew it closed.
            panel.card_choice = Some(true);
            panel.centre = CentreView::SizeAndTorque;
            let mut output = frame(&mut panel, Vec::new());
            for _ in 0..10 {
                output = frame(&mut panel, Vec::new());
            }
            let case = format!("dragged up {dragged}, the card opened by the user");
            assert!(card_open_in(&ctx), "{case}");
            if let Some(overlap) = equation_panel_overlap(&ctx, &output) {
                failures.push(format!("{case}: {overlap}"));
            }
            assert_eq!(window().unwrap().size(), size, "{case}");
            panel.card_choice = None;
        }
```

with:

```rust
            }
        }
```

Replace:

```rust
        // Each value and its correction ids flow as one wrapped text: a long value (the cup
        // wall's verdict in plain words, ruling R9) with its ids never runs past the row, which
        // would widen the resizable dashboard a little more every frame.
        for size in [egui::vec2(930.0, 1024.0), SCREEN] {
```

with:

```rust
        // Each value and its correction ids flow as one wrapped text: a long value (the cup
        // wall's verdict in plain words, ruling R9, among the numbers of its check's entry)
        // with its ids never runs past the row, which would widen the resizable right column a
        // little more every frame.
        for size in [egui::vec2(930.0, 1024.0), SCREEN] {
```

Replace:

```rust
            harness.panel.show_details = true;
            let lefts: Vec<f32> = (0..4)
```

with:

```rust
            harness.panel.show_details = true;
            harness.select("materials.cup_wall_check");
            let lefts: Vec<f32> = (0..4)
```

Replace:

```rust
            let lefts: Vec<f32> = (0..4)
                .map(|_| {
                    let output = harness.frame(Vec::new());
                    // The rightmost: the dashboard's row.
                    text_rects(&output, label_for("model.pullout_Nm"))
                        .into_iter()
                        .map(|rect| rect.left())
                        .fold(f32::NEG_INFINITY, f32::max)
                })
                .collect();
```

with:

```rust
            let lefts: Vec<f32> = (0..4)
                .map(|_| right_column_left(&harness.frame(Vec::new())))
                .collect();
```

Replace:

```rust
            assert!(
                lefts[0].is_finite(),
                "{size:?}: the pull-out's row is drawn"
            );
            assert!(
                lefts.iter().all(|&left| left == lefts[0]),
```

with:

```rust
            assert!(
                lefts.iter().all(|&left| left == lefts[0]),
```

Replace:

```rust
                lefts.iter().all(|&left| left == lefts[0]),
                "{size:?}: the dashboard moved {lefts:?}"
            );
```

with:

```rust
                lefts.iter().all(|&left| left == lefts[0]),
                "{size:?}: the right column moved {lefts:?}"
            );
```

Replace:

```rust
        let mut texts: Vec<String> = [
            WELCOME,
            RESET_ASSUMPTIONS,
```

with:

```rust
        let mut texts: Vec<String> = [
            RESET_ASSUMPTIONS,
```

Replace:

```rust
        let mut harness = Harness::on_screen(TALL);
        // The dashboard's Why links name notes too: the Equation panel's note is one more.
        let pullout = note("pullout_angle").unwrap();
```

with:

```rust
        let mut harness = Harness::on_screen(TALL);
        // A check's Explain entry names its note too: the Equation panel's note is one more.
        let pullout = note("pullout_angle").unwrap();
```

Replace:

```rust
        let end = note("end_effect").unwrap();
        let output = harness.frame(Vec::new());
        let (pullout_links, end_links) = (count(&output, pullout.title), count(&output, end.title));
```

with:

```rust
        let end = note("end_effect").unwrap();
        let output = harness.select("model.end_effect_check");
        let (pullout_links, end_links) = (count(&output, pullout.title), count(&output, end.title));
```

Replace:

```rust
        let (pullout_links, end_links) = (count(&output, pullout.title), count(&output, end.title));
        assert!(
            pullout_links >= 1 && end_links >= 1,
            "the dashboard links both"
        );
```

with:

```rust
        let (pullout_links, end_links) = (count(&output, pullout.title), count(&output, end.title));
        assert_eq!(
            (pullout_links, end_links),
            (0, 1),
            "the end-effect check's entry links its note"
        );
```

Replace:

```rust
        // The Assumptions view is the inputs side of every view but Size and torque (which
        // folds the inputs under "Detailed inputs"; a new panel opens there).
        panel.centre = CentreView::Geometry;
        let frame = |panel: &mut MagcouplingPanel, events: Vec<egui::Event>| {
```

with:

```rust
        // The Assumptions view is the inputs side of every view but Size and torque (which
        // folds the inputs under "Detailed inputs"; a new panel opens there). The Results tab's
        // Headline group lists the headline values.
        panel.centre = CentreView::Results;
        let frame = |panel: &mut MagcouplingPanel, events: Vec<egui::Event>| {
```

Replace:

```rust
        };
        // A window sizes itself on its first frame and paints on the next; the check guides
        // wrap to the dashboard's width as it settles, which a third frame takes in.
        frame(&mut panel, Vec::new());
```

with:

```rust
        };
        // A window sizes itself on its first frame and paints on the next; the right column
        // settles its width, which a third frame takes in.
        frame(&mut panel, Vec::new());
```

Replace:

```rust
        let output = frame(&mut panel, Vec::new());
        // The dashboard's top in view (its lower rows scroll, under the check guides): the
        // status line and the first headline value.
        let defaults = DesignInputs::default();
```

with:

```rust
        let output = frame(&mut panel, Vec::new());
        // The check list's status line, and the table's first headline value.
        let defaults = DesignInputs::default();
```

Replace:

```rust
    fn a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing() {
        use crate::gui::guidance::{all_fixes, fix_text};
        // Every fix of the workbook example's red checks, from the Geometry view (the input's
        // group or the Assumptions view opens) and from Size and torque (its Detailed inputs
        // open). Tall enough for every dashboard row.
        let mut harness = Harness::on_screen(TALL);
```

with:

```rust
    fn a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing() {
        use crate::gui::guidance::{CHECK_GUIDES, all_fixes, fix_text};
        // Every fix of the workbook example's red checks, each in its check's Explain entry,
        // from the Geometry view (the input's group or the Assumptions view opens) and from
        // Size and torque (its Detailed inputs open).
        let mut harness = Harness::on_screen(TALL);
```

Replace:

```rust
        let mut harness = Harness::on_screen(TALL);
        // The dashboard settles its width on the second frame, and the guides above a link wrap
        // to it: click where the link rests.
        harness.frame(Vec::new());
        let defaults = DesignInputs::default();
```

with:

```rust
        let mut harness = Harness::on_screen(TALL);
        let defaults = DesignInputs::default();
```

Replace:

```rust
        let defaults = DesignInputs::default();
        let fixes: std::collections::BTreeSet<(&str, String)> =
            all_fixes(&defaults, &compute_all(&defaults), false)
                .into_iter()
                .flatten()
                .map(|(path, dir)| (path, fix_text(path, dir)))
                .collect();
        assert!(!fixes.is_empty(), "the red checks have fixes");
```

with:

```rust
        let defaults = DesignInputs::default();
        let fixes: Vec<(&str, &str, String)> = CHECK_GUIDES
            .iter()
            .zip(all_fixes(&defaults, &compute_all(&defaults), false))
            .flat_map(|(guide, fixes)| {
                fixes
                    .into_iter()
                    .map(|(path, dir)| (guide.check, path, fix_text(path, dir)))
            })
            .collect();
        assert!(!fixes.is_empty(), "the red checks have fixes");
```

Replace:

```rust
        for view in [CentreView::Geometry, CentreView::SizeAndTorque] {
            for (path, text) in &fixes {
                // No frame left from the last fix.
```

with:

```rust
        for view in [CentreView::Geometry, CentreView::SizeAndTorque] {
            for (check, path, text) in &fixes {
                // No frame left from the last fix.
```

Replace:

```rust
                harness.panel.centre = view;
                harness.click_text(text);
```

with:

```rust
                harness.panel.centre = view;
                // The right column settles its width on the second frame: click where the link
                // rests.
                harness.select(check);
                harness.click_text(text);
```

Replace:

```rust
    #[test]
    fn an_outside_diameter_fix_opens_the_size_tab_and_edits_nothing() {
```

with:

```rust
    #[test]
    fn a_fix_link_on_the_opening_tab_shows_its_row_inside_a_short_window() {
        use crate::gui::guidance::{CHECK_GUIDES, all_fixes, fix_text};
        // Plan review C-9: the Size column scrolls (decision D-7) and Detailed inputs keeps its
        // own scroll area inside it. In a laptop window and in the smallest one, each fix link
        // of the example's red checks that the Explain panel shows opens Detailed inputs and
        // brings its input's row inside the window, not under its foot.
        let defaults = DesignInputs::default();
        let fixes: Vec<(&str, &str, String)> = CHECK_GUIDES
            .iter()
            .zip(all_fixes(&defaults, &compute_all(&defaults), false))
            .flat_map(|(guide, fixes)| {
                fixes
                    .into_iter()
                    .map(|(path, dir)| (guide.check, path, fix_text(path, dir)))
            })
            .collect();
        for size in [egui::vec2(1280.0, 800.0), egui::vec2(1024.0, 700.0)] {
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let mut followed = 0;
            for (check, path, text) in &fixes {
                let mut harness = Harness::on_screen(size);
                harness.select(check);
                let output = harness.frame(Vec::new());
                let link = text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
                if !screen.contains_rect(link) {
                    continue;
                }
                harness.click(link.center());
                assert_eq!(
                    harness.panel.explorer.focus().map(|f| f.path.as_str()),
                    Some(*path)
                );
                // The column scrolls there in an animation of up to half a second.
                let mut output = harness.frame(Vec::new());
                for _ in 0..40 {
                    output = harness.frame(Vec::new());
                }
                let rows: Vec<egui::Rect> = text_rects(&output, label_for(path))
                    .into_iter()
                    .filter(|r| r.left() < INPUTS_WIDTH)
                    .collect();
                let focus = focus_rects(&output);
                assert!(
                    rows.iter()
                        .any(|r| screen.contains_rect(*r)
                            && focus.iter().any(|f| f.contains_rect(*r))),
                    "{size:?} {path}: {rows:?} {focus:?}"
                );
                followed += 1;
            }
            assert!(followed > 0, "{size:?}: no fix link on screen");
        }
    }

    #[test]
    fn an_outside_diameter_fix_opens_the_size_tab_and_edits_nothing() {
```

Replace:

```rust
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        let undo = harness.panel.history.undo_len();
```

with:

```rust
        harness.frame(Vec::new());
        harness.select(crate::gui::guidance::FIT_CHECK);
        let undo = harness.panel.history.undo_len();
```

Replace:

```rust
    #[test]
    fn a_raised_minimum_od_takes_lower_outside_diameter_off_the_dashboard() {
        use crate::engine::size_controls::dimensions;
```

with:

```rust
    #[test]
    fn a_raised_minimum_od_takes_lower_outside_diameter_off_the_space_claim_s_entry() {
        use crate::engine::size_controls::dimensions;
```

Replace:

```rust
        use crate::engine::size_controls::dimensions;
        use crate::gui::guidance::{Direction, all_fixes, fix_text};
        // The space claim of a 12 mm ring (the cup sets the diameter, past its claim) offers
```

with:

```rust
        use crate::engine::size_controls::dimensions;
        use crate::gui::guidance::{CHECK_GUIDES, Direction, all_fixes, fix_text};
        // The space claim of a 12 mm ring (the cup sets the diameter, past its claim) offers
```

Replace:

```rust
        // "Lower Outside diameter"; with the Minimum OD at the diameter the slider cannot go
        // down, so the link goes and the claim's other fixes stay.
        let mut inputs = crate::gui::test_support::passing_design();
```

with:

```rust
        // "Lower Outside diameter"; with the Minimum OD at the diameter the slider cannot go
        // down, so the link goes and every other fix stays.
        let mut inputs = crate::gui::test_support::passing_design();
```

Replace:

```rust
        let ring = "coupling.inner_back_apothem_mm";
        let all = all_fixes(&inputs, &compute_all(&inputs), false);
```

with:

```rust
        let ring = "coupling.inner_back_apothem_mm";
        let claim = CHECK_GUIDES
            .iter()
            .position(|guide| guide.check == SPACE_CLAIM_CHECK)
            .expect("the space claim's guide");
        let all = all_fixes(&inputs, &compute_all(&inputs), false);
```

Replace:

```rust
        let lower = fix_text(ring, Direction::Lower);
        let others: Vec<String> = all
            .iter()
```

with:

```rust
        let lower = fix_text(ring, Direction::Lower);
        let others: Vec<(&str, String)> = CHECK_GUIDES
            .iter()
```

Replace:

```rust
            .iter()
            .flatten()
            .filter(|&&fix| fix != (ring, Direction::Lower))
            .map(|&(path, dir)| fix_text(path, dir))
            .collect();
```

with:

```rust
            .iter()
            .zip(&all)
            .flat_map(|(guide, fixes)| fixes.iter().map(move |&fix| (guide.check, fix)))
            .filter(|&(_, fix)| fix != (ring, Direction::Lower))
            .map(|(check, (path, dir))| (check, fix_text(path, dir)))
            .collect();
```

Replace:

```rust
            .collect();
        assert!(all.iter().flatten().any(|&f| f == (ring, Direction::Lower)));
        assert!(!others.is_empty(), "the claim has another fix");
        let mut harness = Harness::on_screen(TALL);
```

with:

```rust
            .collect();
        assert_eq!(all[claim], [(ring, Direction::Lower)]);
        assert!(!others.is_empty(), "another check has a fix");
        let mut harness = Harness::on_screen(TALL);
```

Replace:

```rust
        harness.panel.inputs = inputs.clone();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &lower), 1);
```

with:

```rust
        harness.panel.inputs = inputs.clone();
        let output = harness.select(SPACE_CLAIM_CHECK);
        assert_eq!(count(&output, &lower), 1);
```

Replace:

```rust
        assert_eq!(count(&output, &lower), 1);
        harness.panel.minimum_diameter_mm = Some(dimensions(&inputs).unwrap().diameter_mm);
```

with:

```rust
        assert_eq!(count(&output, &lower), 1);
        assert_eq!(count(&output, crate::gui::guidance::NO_FIX), 0);
        harness.panel.minimum_diameter_mm = Some(dimensions(&inputs).unwrap().diameter_mm);
```

Replace:

```rust
        assert_eq!(count(&output, &lower), 0);
        for other in &others {
            assert_eq!(count(&output, other), 1, "{other}");
```

with:

```rust
        assert_eq!(count(&output, &lower), 0);
        // Its one fix gone, the entry says no listed change helps; every other check keeps
        // its fixes.
        assert_eq!(count(&output, crate::gui::guidance::NO_FIX), 1);
        for (check, other) in &others {
            let output = harness.select(check);
            assert_eq!(count(&output, other), 1, "{other}");
```

Replace:

```rust
        use crate::gui::guidance::{CHECK_GUIDES, Learn};
        let notes: Vec<&'static str> = CHECK_GUIDES
            .iter()
```

with:

```rust
        use crate::gui::guidance::{CHECK_GUIDES, Learn};
        let notes: Vec<(&'static str, &'static str)> = CHECK_GUIDES
            .iter()
```

Replace:

```rust
            .filter_map(|g| match g.learn {
                Learn::Note(id) => Some(id),
                _ => None,
```

with:

```rust
            .filter_map(|g| match g.learn {
                Learn::Note(id) => Some((g.check, id)),
                _ => None,
```

Replace:

```rust
        let mut harness = Harness::on_screen(TALL);
        // Click where the links rest (the dashboard settles its width on the second frame).
        harness.frame(Vec::new());
        for id in notes {
            let note = crate::gui::explorer::reviewed_note(id).expect("a guide's note is reviewed");
```

with:

```rust
        let mut harness = Harness::on_screen(TALL);
        for (check, id) in notes {
            let note = crate::gui::explorer::reviewed_note(id).expect("a guide's note is reviewed");
```

Replace:

```rust
            harness.panel.explorer.close();
            harness.click_text(&crate::gui::typeset::glyph_safe(note.title));
```

with:

```rust
            harness.panel.explorer.close();
            // The lesson link of the check's Explain entry (the right column settles its width
            // on the second frame: click where the link rests).
            harness.select(check);
            harness.click_text(&crate::gui::typeset::glyph_safe(note.title));
```

Replace:

```rust
        // Spec 3.7 (S2-8): under Magnets per ring and Outside diameter, and on the torque card's
        // heading (slice 3 spec 6.1, drawn open or closed), a link to how the coupling makes
        // torque; at the end of the Required and Hot-low rows a link to the hot minimum check's
        // note. Each opens its note on its own in the Equation panel.
        use crate::gui::dashboard::why_note_text;
```

with:

```rust
        // Spec 3.7 (S2-8): under Magnets per ring and Outside diameter, and on the torque card's
        // heading (slice 3 spec 6.1; in the Hot-low tile's Explain entry, which `size_tab`
        // shows), a link to how the coupling makes torque; beside the Required row's label a
        // link to the hot minimum check's note (the Hot-low tile's is its entry's lesson link).
        // Each opens its note on its own in the Equation panel.
        use crate::gui::dashboard::why_note_text;
```

Replace:

```rust
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        for (id, links) in [(SIZE_TAB_NOTE, 3), (TORQUE_BAND_NOTE, 2)] {
            let text =
```

with:

```rust
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        for (id, links) in [(SIZE_TAB_NOTE, 3), (TORQUE_BAND_NOTE, 1)] {
            let text =
```

Replace:

```rust
    #[test]
    fn the_end_effect_banner_links_to_its_note_over_the_dashboard_and_the_centre() {
        // Spec 3.7 (CG-10): the link under each copy of the banner opens the end-effect check's
        // note; without the banner there is no link.
        use crate::gui::dashboard::why_note_text;
```

with:

```rust
    #[test]
    fn the_end_effect_banner_links_to_its_note_over_the_centre() {
        // Spec 3.7 (CG-10): the link under the banner opens the end-effect check's note;
        // without the banner there is no link.
        use crate::gui::dashboard::why_note_text;
```

Replace:

```rust
        harness.panel.inputs = short_magnets();
        for index in 0..2 {
            harness.panel.explorer = Explorer::default();
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            let links = last_characters(&output, &text);
            assert_eq!(
                links.len(),
                2,
                "under the dashboard's banner and the centre region's"
            );
            harness.click(links[index]);
            assert_eq!(
                harness.panel.explorer.note(),
                Some(BANNER_NOTE),
                "link {index}"
            );
        }
        assert_eq!(harness.panel.inputs(), &short_magnets());
```

with:

```rust
        harness.panel.inputs = short_magnets();
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let links = last_characters(&output, &text);
        assert_eq!(links.len(), 1, "under the centre region's banner");
        harness.click(links[0]);
        assert_eq!(harness.panel.explorer.note(), Some(BANNER_NOTE));
        assert_eq!(harness.panel.inputs(), &short_magnets());
```

Replace:

```rust

    #[test]
    fn the_ripple_row_s_why_link_opens_the_slip_note() {
        // F10: the note's first sentence is drawn as one label, once; the click edits nothing.
        let mut harness = Harness::on_screen(TALL);
        harness.frame(Vec::new());
        let note = crate::gui::explorer::reviewed_note("when_slipping")
            .expect("the slip note is reviewed");
        harness.click_text(&crate::gui::typeset::glyph_safe(note.title));
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.note(), Some("when_slipping"));
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, note.sentences[0]), 1);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    /// A dashboard row's Why links, for the row-link tests (BL-059, BL-064): every copy of the
    /// reviewed note `id`'s title drawn on the dashboard, in paint order (top first), and the
    /// row's own copies, those under `check`'s label and above the next row's label. Each copy is
    /// where a click lands on it, its last character (`last_characters`). A row draws its Why
    /// line whether or not its check fails.
    fn row_s_why_links(
        output: &egui::FullOutput,
        check: &str,
        id: &str,
    ) -> (Vec<egui::Pos2>, Vec<egui::Pos2>) {
        use crate::gui::dashboard::DASHBOARD;
        use crate::gui::typeset::glyph_safe;
        let note = crate::gui::explorer::reviewed_note(id)
```

with:

```rust

    /// Where a click lands on each copy of the reviewed note `id`'s title drawn in the right
    /// column of `output`: its last character (`last_characters`). A check's Explain entry
    /// draws its note's title as its lesson link.
    fn lesson_links(output: &egui::FullOutput, id: &str) -> Vec<egui::Pos2> {
        let note = crate::gui::explorer::reviewed_note(id)
```

Replace:

```rust
            .unwrap_or_else(|| panic!("the {id} note is reviewed"));
        let on_dashboard = |x: f32| x > TALL.x - DASHBOARD_WIDTH;
        let label_top = |path: &str| {
            text_rects(output, label_for(path))
                .into_iter()
                .find(|r| on_dashboard(r.center().x))
                .unwrap_or_else(|| panic!("{path} on the dashboard"))
                .top()
        };
        let index = DASHBOARD
            .iter()
            .position(|&(path, _)| path == check)
            .unwrap_or_else(|| panic!("{check} is a dashboard row"));
        let (next_path, _) = DASHBOARD
            .get(index + 1)
            .unwrap_or_else(|| panic!("a dashboard row after {check}"));
        let (row, next) = (label_top(check), label_top(next_path));
        let copies: Vec<egui::Pos2> = last_characters(output, &glyph_safe(note.title))
            .into_iter()
```

with:

```rust
            .unwrap_or_else(|| panic!("the {id} note is reviewed"));
        let left = right_column_left(output);
        last_characters(output, &crate::gui::typeset::glyph_safe(note.title))
            .into_iter()
```

Replace:

```rust
            .into_iter()
            .filter(|at| on_dashboard(at.x))
            .collect();
        let own = copies
            .iter()
            .copied()
            .filter(|at| at.y > row && at.y < next)
            .collect();
        (copies, own)
    }
```

with:

```rust
            .into_iter()
            .filter(|at| at.x > left)
            .collect()
    }
```

Replace:

```rust
    #[test]
    fn the_gearbox_row_s_own_why_link_opens_the_torque_band_note() {
        // BL-059: the gearbox check's Why link leads to the torque-band note, as the hot-low,
        // hot-minimum and cold-high rows above it do, so those rows draw the note's title first.
        // The copy clicked is the gearbox row's own: on the dashboard, below the row's label and
        // above the next row's. The click edits nothing.
        use crate::gui::guidance::TORQUE_BAND_NOTE;
```

with:

```rust
    #[test]
    fn the_gearbox_check_s_lesson_link_opens_the_torque_band_note() {
        // BL-059: the gearbox check's lesson link leads to the torque-band note, as the hot
        // minimum's does. Its Explain entry draws the note's title once; the click edits
        // nothing.
        use crate::gui::guidance::TORQUE_BAND_NOTE;
```

Replace:

```rust
        let mut harness = Harness::on_screen(TALL);
        let output = harness.frame(Vec::new());
        let (copies, own) = row_s_why_links(&output, GEARBOX_CHECK, TORQUE_BAND_NOTE);
        assert_eq!(own.len(), 1, "one link under the gearbox row: {copies:?}");
        assert!(
            copies[0].y < own[0].y,
            "the rows above draw it first: {copies:?}"
        );
        click_opens_note(&mut harness, own[0], TORQUE_BAND_NOTE);
    }
```

with:

```rust
        let mut harness = Harness::on_screen(TALL);
        let output = harness.select(GEARBOX_CHECK);
        let links = lesson_links(&output, TORQUE_BAND_NOTE);
        assert_eq!(links.len(), 1, "one link in the gearbox check's entry");
        click_opens_note(&mut harness, links[0], TORQUE_BAND_NOTE);
    }
```

Replace:

```rust
    #[test]
    fn the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note() {
        // BL-064: the shaft-joint row's Why link leads to the shaft-joint note, which no other
        // row links, so its title is drawn once on the dashboard, under the row's label and
        // above the next row's. The click edits nothing.
        const SHAFT_JOINT_CHECK: &str = "shaft_joint.check";
```

with:

```rust
    #[test]
    fn the_shaft_joint_check_s_lesson_link_opens_the_shaft_joint_note() {
        // BL-064: the shaft joints check's lesson link leads to the shaft-joint note. Its
        // Explain entry draws the note's title once; the click edits nothing.
        const SHAFT_JOINT_CHECK: &str = "shaft_joint.check";
```

Replace:

```rust
        let mut harness = Harness::on_screen(TALL);
        let output = harness.frame(Vec::new());
        let (copies, own) = row_s_why_links(&output, SHAFT_JOINT_CHECK, "shaft_joint");
        assert_eq!(copies.len(), 1, "one link to the note: {copies:?}");
        assert_eq!(own, copies, "under the shaft-joint row");
        click_opens_note(&mut harness, own[0], "shaft_joint");
    }
```

with:

```rust
        let mut harness = Harness::on_screen(TALL);
        let output = harness.select(SHAFT_JOINT_CHECK);
        let links = lesson_links(&output, "shaft_joint");
        assert_eq!(links.len(), 1, "one link to the note");
        click_opens_note(&mut harness, links[0], "shaft_joint");
    }
```

Replace:

```rust
    #[test]
    fn a_narrow_window_wraps_the_check_guides_inside_the_dashboard() {
        // The M4-1 review's ~930 px window, tall enough for every row: every guide line of the
```

with:

```rust
    #[test]
    fn a_narrow_window_wraps_each_check_s_entry_inside_the_right_column() {
        // The M4-1 review's ~930 px window, tall enough for every row: every guide line of the
```

Replace:

```rust
        let mut harness = Harness::on_screen(egui::vec2(width, 3000.0));
        // The dashboard settles its width on the second frame.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let inputs = DesignInputs::default();
```

with:

```rust
        let mut harness = Harness::on_screen(egui::vec2(width, 3000.0));
        let inputs = DesignInputs::default();
```

Replace:

```rust
        let fixes = all_fixes(&inputs, &results, false);
        let mut texts = vec![design_status_text(&design_status(&inputs, &results, false))];
        for (guide, fixes) in CHECK_GUIDES.iter().zip(&fixes) {
```

with:

```rust
        let fixes = all_fixes(&inputs, &results, false);
        let status = design_status_text(&design_status(&inputs, &results, false));
        let mut entries = 0;
        for (guide, fixes) in CHECK_GUIDES.iter().zip(&fixes) {
```

Replace:

```rust
        for (guide, fixes) in CHECK_GUIDES.iter().zip(&fixes) {
            if level(guide, &inputs, &results) == Level::Bad {
                texts.extend((guide.problem)(&inputs, &results));
                texts.push(guide.why.to_owned());
                texts.extend(fixes.iter().map(|&(path, dir)| fix_text(path, dir)));
            }
```

with:

```rust
        for (guide, fixes) in CHECK_GUIDES.iter().zip(&fixes) {
            if level(guide, &inputs, &results) != Level::Bad {
                continue;
            }
            entries += 1;
            // The right column settles its width on the second frame.
            harness.select(guide.check);
            let output = harness.frame(Vec::new());
            let mut texts = vec![status.clone(), guide.why.to_owned()];
            texts.extend((guide.problem)(&inputs, &results));
            texts.extend(fixes.iter().map(|&(path, dir)| fix_text(path, dir)));
            // Right, the side panel's margin inside the frame's (8 points each), give or take
            // egui's rounding of a widget's edges (a line that wraps to the full width can end
            // one rounding step past it).
            let column = right_column_left(&output)..=width - 16.0 + egui::emath::GUI_ROUNDING;
            for text in &texts {
                let rects = text_rects(&output, text);
                assert!(!rects.is_empty(), "{text:?} is drawn");
                for rect in rects {
                    assert!(
                        column.contains(&rect.left()) && column.contains(&rect.right()),
                        "{text:?} at {rect:?}, the right column {column:?}"
                    );
                }
            }
```

Replace:

```rust
        }
        // Left of its first label's badge; right, the side panel's margin inside the frame's
        // (8 points each), give or take egui's rounding of a widget's edges (a line that wraps
        // to the full width can end one rounding step past it).
        let pullout = label_for("model.pullout_Nm");
        let dashboard = text_rect(&output, pullout).unwrap().left() - 30.0
            ..=width - 16.0 + egui::emath::GUI_ROUNDING;
        for text in &texts {
            let rects = text_rects(&output, text);
            assert!(!rects.is_empty(), "{text:?} is drawn");
            for rect in rects {
                assert!(
                    dashboard.contains(&rect.left()) && dashboard.contains(&rect.right()),
                    "{text:?} at {rect:?}, the dashboard {dashboard:?}"
                );
            }
        }
    }
```

with:

```rust
        }
        assert_eq!(entries, 4, "the example design's four red checks");
    }
```

**`magcoupling-rs/src/gui/test_support.rs`**

Replace:

```rust

/// A key event with no modifiers: a press, or (`pressed` false) its release.
```

with:

```rust

/// What the Explain panel draws for the check `check` of `inputs` (whose results are `results`),
/// handed every guide's `fixes` and the shaft-clamp section's pause, with the spreadsheet
/// details off: one frame, tall enough for the whole entry.
pub(crate) fn check_entry_texts(
    inputs: &crate::DesignInputs,
    results: &crate::DesignResults,
    fixes: &[Vec<(&'static str, crate::gui::guidance::Direction)>],
    clamps_paused: bool,
    check: &'static str,
) -> Vec<String> {
    use crate::gui::explain::{Selection, Shown, explain_ui};
    let ctx = egui::Context::default();
    let output = sized_frame(&ctx, egui::vec2(320.0, 4000.0), Vec::new(), |ui| {
        let shown = Shown {
            inputs,
            results,
            fixes,
            clamps_paused,
            consequences: None,
            auto: None,
        };
        let mut readouts = crate::gui::readouts::Readouts::default();
        explain_ui(ui, Selection::Check(check), &shown, &mut readouts);
    });
    drawn_texts(&output)
}

/// A key event with no modifiers: a press, or (`pressed` false) its release.
```

Replace:

```rust

/// A context whose fonts are loaded, for [`assert_glyphs`]: egui loads them on a context's first
```

with:

```rust

/// The rects of `text` where `output` draws it whole: inside its clip rect and inside `screen`.
/// A text in a list scrolled past, or under a panel's edge, is left out.
pub(crate) fn shown_whole(
    output: &egui::FullOutput,
    text: &str,
    screen: egui::Rect,
) -> Vec<egui::Rect> {
    clipped_flat_shapes(output)
        .into_iter()
        .filter_map(|(clip, shape)| match shape {
            egui::Shape::Text(shape) if shape.galley.text() == text => {
                let rect = shape.visual_bounding_rect();
                (clip.intersect(screen).expand(0.5).contains_rect(rect)).then_some(rect)
            }
            _ => None,
        })
        .collect()
}

/// The words `output` shows on `screen` (UX overhaul spec 2.1, goal G-1): every text that is
/// drawn at least in part inside its clip rect and the screen, counted by whitespace.
pub(crate) fn words_shown(output: &egui::FullOutput, screen: egui::Rect) -> usize {
    clipped_flat_shapes(output)
        .into_iter()
        .filter_map(|(clip, shape)| match shape {
            egui::Shape::Text(shape)
                if clip
                    .intersect(screen)
                    .intersects(shape.visual_bounding_rect()) =>
            {
                Some(shape.galley.text().split_whitespace().count())
            }
            _ => None,
        })
        .sum()
}

/// A context whose fonts are loaded, for [`assert_glyphs`]: egui loads them on a context's first
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 15
  ```

  Expected: the build fails with 43 errors, the first "failed to resolve: use of undeclared type `Selection`": the green step adds what the tests call.

- [ ] **Step 3 (green): the panel.**

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
//! The dashboard on the right (spec M4 "Layout": "headline numbers with green/amber/red
//! badges derived from check verdicts; corrected values carry a 'corrected vs workbook' marker
//! with the deviation in its tooltip").
//!
```

with:

```rust
//! The dashboard's rows and their badges (spec M4 "Layout": "headline numbers with
//! green/amber/red badges derived from check verdicts; corrected values carry a 'corrected vs
//! workbook' marker with the deviation in its tooltip"). Since the UX overhaul (spec 4.5) no
//! one list draws them all: the right column is the check list ([`crate::gui::checks`]) over
//! the Explain panel ([`crate::gui::explain`]), which draws the rows a check's verdict badges
//! as that check's numbers ([`line_ui`]); the Results tab's Headline group lists every row, and
//! the spreadsheet's Summary writes them.
//!
```

Replace:

```rust
//!
//! [`DASHBOARD`] lists its rows: the 15 headline numbers in Python's order, then the shaft
//! joints check (BL-064), the gearbox check (BL-059), the A1 space claim badge and the audit M9
```

with:

```rust
//!
//! [`DASHBOARD`] lists the rows: the 15 headline numbers in Python's order, then the shaft
//! joints check (BL-064), the gearbox check (BL-059), the A1 space claim badge and the audit M9
```

Replace:

```rust
//!   of range", which carries a "Why: <note>" link to the end-effect check's note (slice-2 spec
//!   3.7; `end_effect_banner_ui` draws it here and over the centre region).
//! - **Stored 3D values.** The temperature rows and the shaft joints row that read the stored
```

with:

```rust
//!   of range", which carries a "Why: <note>" link to the end-effect check's note (slice-2 spec
//!   3.7; `end_effect_banner_ui` draws it over the centre region).
//! - **Stored 3D values.** The temperature rows and the shaft joints row that read the stored
```

Replace:

```rust
//!   design or for the starting design ([`STARTING_3D_FIELDS`]) and stay fixed when the design
//!   changes; no live 3D model runs.
//!
```

with:

```rust
//!   design or for the starting design ([`STARTING_3D_FIELDS`]) and stay fixed when the design
//!   changes; no live 3D model runs. The check list closes with the label, and the entries of
//!   the two checks that read the fields repeat it.
//!
```

Replace:

```rust
//!
//! Over the rows, the material warnings that fire (spec Addendum A5: "plain language,
//! colour-coded, linked to their teaching note"): each rule's text in its severity's colour (a
//! warning red, a caution amber) with a link that opens its reviewed note in the Equation panel
//! ([`warning_lines`], decision M43-8).
//!
```

with:

```rust
//!
//! Under the check list's rows, the material warnings that fire (spec Addendum A5: "plain
//! language, colour-coded, linked to their teaching note"): each rule's text in its severity's
//! colour (a warning red, a caution amber) with a link that opens its reviewed note in the
//! Equation panel ([`warning_lines`], decision M43-8).
//!
```

Replace:

```rust
//! text every readout hands to [`crate::gui::readouts::Readouts::show`], which adds the
//! value's equation (plan M4-3). The dashboard builds a row's text only while it is hovered
//! ([`DashboardLine::tooltip`]).
```

with:

```rust
//! text every readout hands to [`crate::gui::readouts::Readouts::show`], which adds the
//! value's equation (plan M4-3). A row's text is built only while it is hovered
//! ([`DashboardLine::tooltip`]).
```

Replace:

```rust
//! Check guides (guidance spec 3.1 and 3.4, [`crate::gui::guidance`]): the design status line
//! tops the dashboard, and the Magnets fit row follows the end-effect flag (a GUI check of
//! `model::blocks_fit`, so it is drawn apart from [`DASHBOARD`], whose paths are results). Under
//! a guided check's own row, a red level opens its block: the problem from this design's
//! results, the guide's sentence and its fixes (each a link that shows its input's row, the
//! Outside diameter's the Size tab), or a line saying no listed change helps, then any
//! trade-off and side-effect notes. An amber level keeps the block shut under a one-line header, so the
//! dashboard stays short. Every row, at any level, has its Why line: a link to its reviewed
//! note (opened in the Equation panel), a link to its assumption (its row shown, the rationale
//! in the hover text) or "no note yet" ([`crate::gui::guidance::row_learn`]).

```

with:

```rust
//! Check guides (guidance spec 3.1 and 3.4, [`crate::gui::guidance`]): the design status line
//! tops the check list ([`status_ui`]), and the Magnets fit check closes its rows (a GUI check
//! of `model::blocks_fit`, so it is apart from [`DASHBOARD`], whose paths are results). A
//! check's problem, its guide's sentence, its fixes and their notes are its Explain entry. An
//! entry's lesson link ([`learn_ui`]) is a link to its reviewed note (opened in the Equation
//! panel), a link to its assumption (its row shown, the rationale in the hover text) or "no
//! note yet".

```

Replace:

```rust
use crate::gui::format::{format_value, with_unit};
use crate::gui::guidance::{
    self, BANNER_NOTE, CHECK_GUIDES, CheckGuide, Direction, FIT_CHECK, Learn, NO_FIX,
    design_status, fix_text, guide_notes, row_learn, status_line,
};
use crate::gui::look;
```

with:

```rust
use crate::gui::format::{format_value, with_unit};
use crate::gui::guidance::{BANNER_NOTE, Learn, design_status, status_line};
use crate::gui::look;
```

Replace:

```rust

/// The head of a red or opened amber check's problem line (guidance spec 3.4).
pub const PROBLEM: &str = "Problem:";

/// The head of a red or opened amber check's fix line: each fix a link that shows its input.
pub const FIX: &str = "Fix:";

/// The head of every row's Why line: under an open block the guide's sentence, then the link
/// to read more.
pub const WHY_LINK: &str = "Why:";

/// What a Why line says when no reviewed note or assumption explains the row.
pub const NO_NOTE_YET: &str = "no note yet";
```

with:

```rust

/// What a lesson link says when no reviewed note or assumption explains the row.
pub const NO_NOTE_YET: &str = "no note yet";
```

Replace:

```rust

/// The hover text of a Why line whose note is not reviewed (sent back to Draft): it never shows.
pub const NOTE_NOT_REVIEWED: &str = "Its teaching note has not passed review.";
```

with:

```rust

/// The hover text of a lesson link whose note is not reviewed (sent back to Draft): it never
/// shows.
pub const NOTE_NOT_REVIEWED: &str = "Its teaching note has not passed review.";
```

Replace:

```rust

/// The one-line header an amber check's block is shut under: a click opens it.
pub const CAUTION_DETAILS: &str = "Problem and fixes";

/// The Magnets fit row's value: the blocks fit around the ring, or not.
pub const FITS: &str = "Fit around the ring";
```

with:

```rust

/// The Magnets fit check's value in its Explain entry: the blocks fit around the ring, or not.
pub const FITS: &str = "Fit around the ring";
```

Replace:

```rust

/// Draws the dashboard of `results`, computed from `inputs`: the design status line, the
/// end-effect banner when f_end <= 0, then a row per line (badge and label, the label quieter
/// than the value, [`look::quiet_text`]; then the value and,
/// with the spreadsheet details, its correction ids under a one-line legend, so a narrow side
/// still reads) with its Why line, a guided check's
/// block over its Why line, the stored-3D label after the last stored-3D row, and the
/// Magnets fit row last. `fixes` holds every guide's fixes in [`CHECK_GUIDES`]' order
/// (`guidance::FixCache`). Each result row is a readout (`readouts`): its hover text and
/// equation while hovered, a click opens it in the Equation panel. While the shaft-clamp
/// section is paused (`clamps_paused`) its rows, their guide and their Why links are left out
/// and the status line does not count its check. `baseline` is the session's baseline: the
/// status line names the design against it ([`guidance::status_line`]).
pub fn dashboard_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    baseline: &DesignInputs,
    results: &DesignResults,
    fixes: &[Vec<(&'static str, Direction)>],
    clamps_paused: bool,
    readouts: &mut Readouts,
) {
    status_ui(ui, inputs, baseline, results, clamps_paused);
    let lines = shown_lines(results, clamps_paused);
    if end_effect_banner_ui(ui, results, readouts) {
        ui.separator();
    }
    warnings_ui(ui, results, readouts);
    // The correction ids beside the values, and what they are, only with the spreadsheet
    // details (spec S2-5).
    let show_details = readouts.show_details();
    if show_details {
        ui.add(egui::Label::new(egui::RichText::new(MARKER_LEGEND).small().weak()).wrap());
    }
    let last_3d = lines.iter().rposition(|line| line.stored_3d);
    for (index, line) in lines.iter().enumerate() {
        line_ui(ui, line, readouts);
        // A check's block under its own row (the rows its verdict only badges have none), then
        // every row's Why line.
        match CHECK_GUIDES.iter().position(|g| g.check == line.path) {
            Some(guide) => indexed_guide_ui(ui, guide, inputs, results, fixes, readouts),
            None => {
                let learn = row_learn(line.path).expect("every row has a Why link (guidance)");
                why_line(ui, None, learn, readouts);
            }
        }
        if Some(index) == last_3d {
            ui.horizontal(|ui| {
                ui.add_space(16.0);
                ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
            });
        }
        ui.add_space(4.0);
    }
    fit_ui(ui, inputs, results, fixes, readouts);
}

/// Draws one dashboard row: its badge and label, the label quieter than the value
```

with:

```rust

/// Draws one dashboard row: its badge and label, the label quieter than the value
```

Replace:

```rust

/// The Magnets fit row: its badge and label, whether the blocks fit (from the guide's level: red
/// when they do not), then its guide.
fn fit_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    fixes: &[Vec<(&'static str, Direction)>],
    readouts: &mut Readouts,
) {
    let index = CHECK_GUIDES
        .iter()
        .position(|g| g.check == FIT_CHECK)
        .expect("the Magnets fit row has a guide");
    let guide = &CHECK_GUIDES[index];
    let level = guidance::level(guide, inputs, results);
    ui.horizontal(|ui| {
        state_mark(ui, Some(level));
        let label = egui::RichText::new(guide.label).color(look::quiet_text(ui.visuals()));
        ui.add(egui::Label::new(label).wrap());
    });
    ui.horizontal(|ui| {
        ui.add_space(16.0);
        let value = if level == Level::Bad {
            DOES_NOT_FIT
        } else {
            FITS
        };
        ui.add(egui::Label::new(egui::RichText::new(value).strong()).wrap());
    });
    indexed_guide_ui(ui, index, inputs, results, fixes, readouts);
}

/// What a guide's lines are drawn with: its check's level, the fixes to show (empty: the
/// no-fix line) and the notes under them ([`guide_notes`]).
struct GuideShown<'a> {
    level: Level,
    fixes: &'a [(&'static str, Direction)],
    /// Read only when the block is open (they are drawn under its fixes).
    notes: &'a dyn Fn() -> Vec<String>,
}

/// The lines of the guide at `index` of [`CHECK_GUIDES`] in this design ([`guide_ui`]): its
/// level, its fixes from `fixes`, and its notes ([`guide_notes`]).
fn indexed_guide_ui(
    ui: &mut egui::Ui,
    index: usize,
    inputs: &DesignInputs,
    results: &DesignResults,
    fixes: &[Vec<(&'static str, Direction)>],
    readouts: &mut Readouts,
) {
    let guide = &CHECK_GUIDES[index];
    let notes = || guide_notes(fixes, index, inputs, results);
    let shown = GuideShown {
        level: guidance::level(guide, inputs, results),
        fixes: fixes.get(index).map_or(&[], Vec::as_slice),
        notes: &notes,
    };
    guide_ui(ui, guide, &shown, inputs, results, readouts);
}

/// A guided check's lines under its row. Its block, open under red and under amber once its
/// header is clicked: the problem (from `inputs` and `results`, in the level's colour), the
/// Why line with the guide's sentence, and the Fix line ([`fix_line`]). Otherwise the Why
/// line's link alone ([`why_line`]).
fn guide_ui(
    ui: &mut egui::Ui,
    guide: &CheckGuide,
    shown: &GuideShown,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    let open = match shown.level {
        Level::Bad => true,
        Level::Caution => caution_header_ui(ui, guide.check),
        Level::Good => false,
    };
    if open && let Some(problem) = (guide.problem)(inputs, results) {
        guide_line(ui, PROBLEM, |ui| {
            let text = egui::RichText::new(problem).color(shown.level.color(ui.visuals()));
            ui.add(egui::Label::new(text).wrap());
        });
    }
    why_line(ui, open.then_some(guide.why), guide.learn, readouts);
    if open {
        fix_line(ui, shown, readouts);
    }
}

/// The header an amber check's block is shut under ([`CAUTION_DETAILS`]), shut at first: a
/// click on it or on its arrow opens or shuts the block. Whether the block is open.
fn caution_header_ui(ui: &mut egui::Ui, check: &str) -> bool {
    let id = ui.make_persistent_id(("check_guide", check));
    let mut state =
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
    ui.horizontal(|ui| {
        ui.add_space(16.0);
        state.show_toggle_button(ui, egui::collapsing_header::paint_default_icon);
        let header = egui::Label::new(CAUTION_DETAILS).sense(egui::Sense::click());
        if ui.add(header).clicked() {
            state.toggle(ui);
        }
    });
    state.store(ui.ctx());
    state.is_open()
}

/// The Fix line of an open block: a link per fix, which asks for its input's row (a two-pick
/// magnet fix: its part's row), or [`NO_FIX`] when no listed change helps; then the notes
/// ([`guide_notes`]).
fn fix_line(ui: &mut egui::Ui, shown: &GuideShown, readouts: &mut Readouts) {
    guide_line(ui, FIX, |ui| {
        if shown.fixes.is_empty() {
            ui.add(egui::Label::new(NO_FIX).wrap());
        }
        for &(path, direction) in shown.fixes {
            if ui.link(fix_text(path, direction)).clicked() {
                readouts.focus_input(path);
            }
        }
        for note in (shown.notes)() {
            ui.add(egui::Label::new(note).wrap());
        }
    });
}

/// A row's Why line: under an open block the guide's `sentence`, then the link ([`learn_ui`]).
fn why_line(ui: &mut egui::Ui, sentence: Option<&str>, learn: Learn, readouts: &mut Readouts) {
    guide_line(ui, WHY_LINK, |ui| {
        if let Some(sentence) = sentence {
            ui.add(egui::Label::new(sentence).wrap());
        }
        learn_ui(ui, learn, readouts);
    });
}

/// The link at the end of a Why line: a reviewed note's title (it asks for the note in the
/// Equation panel; a note not reviewed never shows), an assumption's label (it asks for its
```

with:

```rust

/// The lesson link of an Explain entry: a reviewed note's title (it asks for the note in the
/// Equation panel; a note not reviewed never shows), an assumption's label (it asks for its
```

Replace:

```rust
    }
}

/// One line of a row's guide, 16 points in under its row: `head`, a caption
/// ([`look::quiet_text`]), then `body` in a column beside it that wraps within the width left.
fn guide_line(ui: &mut egui::Ui, head: &str, body: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal_top(|ui| {
        ui.add_space(16.0);
        ui.colored_label(look::quiet_text(ui.visuals()), head);
        ui.vertical(body);
    });
}
```

with:

```rust
    }
}
```

**`magcoupling-rs/src/gui/geometry_view.rs`**

Replace:

```rust
/// The height of a row of the callout list under the drawing [points]: a body text row and the
/// item spacing, as [`geometry_ui`] reserves it (one expression for it and [`drawing_floor`]).
pub(crate) fn list_row(ui: &egui::Ui) -> f32 {
```

with:

```rust
/// The height of a row of the callout list under the drawing [points]: a body text row and the
/// item spacing, as [`geometry_ui`] reserves it.
pub(crate) fn list_row(ui: &egui::Ui) -> f32 {
```

Replace:

```rust
    ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y
}

/// The room the drawing needs to keep [`MIN_DRAWING_HEIGHT`] with [`MIN_LIST_ROWS`] rows of the
/// callout list under it [points] (slice 3 spec 6.2): the room the Size tab's torque card leaves
/// the drawing below it, starting closed rather than take it and capping its body above it.
pub(crate) fn drawing_floor(ui: &egui::Ui) -> f32 {
    MIN_DRAWING_HEIGHT + MIN_LIST_ROWS * list_row(ui)
}
```

with:

```rust
    ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y
}
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the dashboard on the right and
//! the centre region between them ([`CentreView`]: one tab row, the results table last;
```

with:

```rust
//! Layout (spec M4 "Layout"): a header line, the inputs on the left (the Key design group,
//! then every input by package group, [`crate::gui::inputs`]), the check list over the Explain
//! panel on the right ([`crate::gui::checks`], [`crate::gui::explain`]: UX overhaul spec 4.5) and
//! the centre region between them ([`CentreView`]: one tab row, the results table last;
```

Replace:

```rust
use crate::engine::sizing::{FreeVariable, SizingOutcome};
use crate::gui::clamp_drawing::clamp_ui;
```

with:

```rust
use crate::engine::sizing::{FreeVariable, SizingOutcome};
use crate::gui::checks::{CHECKS_SHARE, checks_ui};
use crate::gui::clamp_drawing::clamp_ui;
```

Replace:

```rust
use crate::gui::clamp_drawing::clamp_ui;
use crate::gui::dashboard::{
    Level, dashboard_ui, end_effect_banner, end_effect_banner_ui, why_note_link, why_note_text,
};
```

with:

```rust
use crate::gui::clamp_drawing::clamp_ui;
use crate::gui::dashboard::{Level, end_effect_banner_ui, why_note_link};
use crate::gui::explain::{Asked, Selection, explain_ui};
use crate::gui::explorer::{
    EQUATION_PANEL, Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights,
};
```

Replace:

```rust
};
use crate::gui::explorer::{
    EQUATION_PANEL, Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights, reviewed_note,
};
use crate::gui::format::{format_value, search_needle, with_unit};
use crate::gui::geometry_view::{drawing_floor, geometry_ui};
use crate::gui::guidance::{
```

with:

```rust
};
use crate::gui::format::search_needle;
use crate::gui::geometry_view::geometry_ui;
use crate::gui::guidance::{
```

Replace:

```rust
use crate::gui::guidance::{
    ConsequenceCache, Consequences, EXAMPLE_DESIGN_HOVER, FixCache, SIZE_TAB_ALIASES,
    SIZE_TAB_NOTE, TORQUE_BAND_NOTE, within_minimum_diameter,
};
```

with:

```rust
use crate::gui::guidance::{
    ConsequenceCache, Consequences, Direction, EXAMPLE_DESIGN_HOVER, FixCache, SIZE_TAB_ALIASES,
    TORQUE_BAND_NOTE, within_minimum_diameter,
};
```

Replace:

```rust
use crate::gui::history::History;
use crate::gui::input_ui::{CHANGED_DOT, RowEdit, hint_line, input_row, slider, weak_line};
use crate::gui::inputs::{
```

with:

```rust
use crate::gui::history::History;
use crate::gui::input_ui::{CHANGED_DOT, RowEdit, hint_line, input_row, slider};
use crate::gui::inputs::{
```

Replace:

```rust
use crate::gui::spreadsheet::Snapshot;
use crate::gui::torque_card::{CARD_HEADING, KEYWAY_BREAKS_THROUGH, SIDE_DOES_NOT_FIT};
use crate::gui::trace::{CLEAR_TRACE, Trace, TraceKind};
```

with:

```rust
use crate::gui::spreadsheet::Snapshot;
use crate::gui::tiles::tiles_ui;
use crate::gui::torque_card::{KEYWAY_BREAKS_THROUGH, SIDE_DOES_NOT_FIT};
use crate::gui::trace::{CLEAR_TRACE, Trace, TraceKind};
```

Replace:

```rust
pub const HEADING: &str = "Magnetic coupling calculator";

/// The welcome strip above the centre tabs (spec 3.1), until the session hides it.
pub const WELCOME: &str =
    "Set the size -> read the torque -> open any red check to see why and what to change.";

/// The button at the end of the welcome strip that hides it for the rest of the session.
pub const HIDE_WELCOME: &str = "Hide";

/// The hover text of [`HIDE_WELCOME`].
pub const HIDE_WELCOME_HINT: &str =
    "Hide this line until the page is opened again; the design is not changed";

```

with:

```rust
pub const HEADING: &str = "Magnetic coupling calculator";

```

Replace:

```rust

/// The id of the torque card's collapsing state on the Size tab (slice 3 spec 6.1): egui keeps
/// the card open or closed under it. The card's body salts its scroll area with it too.
const TORQUE_CARD_ID: &str = "magcoupling_torque_card";

/// The narrowest centre region the torque card starts open in [points] (slice 3 spec T-10,
/// decision 4A's "the window is narrow" as a width rule): between the 278 points of a
/// 930 x 800 window's region and the 628 of a 1280 x 800 window's.
const CARD_OPEN_MIN_WIDTH: f32 = 400.0;

/// The view the torque card's default is decided for (slice 3 spec 6.2): whether the Equation
/// panel is open, and the screen's size in whole points. A design change never changes it, so
/// the card never opens or closes under a drag; nor does the centre region's width, which a side
/// panel widening to fit a design's content moves (code review 3 MF-2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CardView {
    equation_open: bool,
    screen_width: u32,
    screen_height: u32,
}

impl CardView {
    /// This frame's view: whether the Equation panel is open (`equation_open`), and `ctx`'s
    /// screen rounded to whole points.
    fn of(ctx: &egui::Context, equation_open: bool) -> Self {
        let screen = ctx.screen_rect();
        Self {
            equation_open,
            screen_width: screen.width().round() as u32,
            screen_height: screen.height().round() as u32,
        }
    }
}

/// What the torque card's open-or-closed rule reads of a frame (slice 3 spec 6.2), taken at the
/// top of the centre region before the Equation panel takes its part: the view, and the region's
/// width [points] (rule 2, read on a deciding frame only).
#[derive(Clone, Copy, Debug)]
struct CardFrame {
    view: CardView,
    region_width: f32,
}

/// The results of the design the Size tab last asked its [`ConsequenceCache`] about, kept with
```

with:

```rust

/// The results of the design the Size tab last asked its [`ConsequenceCache`] about, kept with
```

Replace:

```rust
    design_widgets: Vec<egui::Id>,
    /// The Size tab's Required slider's widget as drawn in the last frame. The centre view is
    /// drawn after the solve checks [`MagcouplingPanel::editing`], so the inputs side lists it
    /// in `design_widgets` ahead of that check: typing in its value box holds the solve.
    requirement_widget: Option<egui::Id>,
    /// Whether the panel acts on the undo and redo keys
```

with:

```rust
    design_widgets: Vec<egui::Id>,
    /// Whether the panel acts on the undo and redo keys
```

Replace:

```rust
    centre: CentreView,
    /// Whether the session has hidden the welcome strip (spec 3.1): a view choice, never in a
    /// design file, a share link or the undo history, and kept by reset all and every load.
    welcome_dismissed: bool,
    /// The user's own open or close of the Size tab's torque card (slice 3 spec 6.2), set by a
    /// click on its arrow or its heading: it wins for the session. A view choice, as
    /// `welcome_dismissed` is.
    card_choice: Option<bool>,
    /// The torque card's default and the view it was decided for (slice 3 spec 6.2): decided
    /// again only when the view differs, while the user has made no choice. A view choice too.
    card_start: Option<(CardView, bool)>,
    /// Whether the header's Show spreadsheet details is on (spec S2-5): a view choice, never in
```

with:

```rust
    centre: CentreView,
    /// Whether the header's Show spreadsheet details is on (spec S2-5): a view choice, never in
```

Replace:

```rust
    clamps_paused: bool,
}
```

with:

```rust
    clamps_paused: bool,
    /// What the Explain panel shows (UX overhaul spec 5.1): the thing clicked last, or nothing
    /// (the orientation). View state: in no design file, link or undo step, and kept by reset
    /// all and every load.
    selection: Selection,
    /// Whether a text field had the keyboard as the last frame ended: Esc is then the field's,
    /// not the Explain panel's.
    typing: bool,
}
```

Replace:

```rust
            design_widgets: Vec::new(),
            requirement_widget: None,
            keyboard_shortcuts: true,
```

with:

```rust
            design_widgets: Vec::new(),
            keyboard_shortcuts: true,
```

Replace:

```rust
            centre: CentreView::SizeAndTorque,
            welcome_dismissed: false,
            card_choice: None,
            card_start: None,
            show_details: false,
```

with:

```rust
            centre: CentreView::SizeAndTorque,
            show_details: false,
```

Replace:

```rust
            clamps_paused: clamps::PAUSED,
        }
```

with:

```rust
            clamps_paused: clamps::PAUSED,
            selection: Selection::default(),
            typing: false,
        }
```

Replace:

```rust
        let mut readouts = Readouts::new(marks, self.show_details);
        ui.push_id("magcoupling_panel", |ui| {
```

with:

```rust
        let mut readouts = Readouts::new(marks, self.show_details);
        // Esc returns the Explain panel to its orientation (UX overhaul spec 5.1), unless the
        // key is another's: a text field that had the keyboard as the last frame ended (egui
        // takes the keyboard from it at this key before the frame is drawn), or an open list.
        if !self.typing
            && !egui::Popup::is_any_open(ui.ctx())
            && ui.input(|i| i.key_pressed(egui::Key::Escape))
        {
            self.selection = Selection::Nothing;
        }
        ui.push_id("magcoupling_panel", |ui| {
```

Replace:

```rust
                .show_inside(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("magcoupling_dashboard_scroll")
                        .show(ui, |ui| {
                            dashboard_ui(
                                ui,
                                &shown,
                                &self.baseline,
                                &self.results,
                                &fixes,
                                self.clamps_paused,
                                &mut readouts,
                            )
                        });
                });
```

with:

```rust
                .show_inside(ui, |ui| {
                    self.checks_and_explain_ui(ui, &shown, &fixes, &mut readouts)
                });
```

Replace:

```rust
        self.history.observe(&self.design(), settled);
    }
```

with:

```rust
        self.history.observe(&self.design(), settled);
        self.typing = focused_text_field(ui.ctx()).is_some();
    }

    /// The right column (UX overhaul spec 4.5): the check list, at its own height up to
    /// [`CHECKS_SHARE`] of the column and scrolling beyond it, then the Explain panel in the
    /// rest, scrolling on its own. A click on a check's row selects it, and the entry under the
    /// list shows it in the same frame.
    fn checks_and_explain_ui(
        &mut self,
        ui: &mut egui::Ui,
        shown: &DesignInputs,
        fixes: &[Vec<(&'static str, Direction)>],
        readouts: &mut Readouts,
    ) {
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_checks_scroll")
            .max_height(CHECKS_SHARE * ui.available_height())
            .show(ui, |ui| {
                let clicked = checks_ui(
                    ui,
                    shown,
                    &self.baseline,
                    &self.results,
                    self.clamps_paused,
                    self.selection.check(),
                    readouts,
                );
                if let Some(check) = clicked {
                    self.selection = Selection::Check(check);
                }
            });
        ui.separator();
        let auto = self.auto();
        // Asked as the Size column asks (`size_tab_consequences`), and only for an entry that
        // reads them: the cache computes once per design.
        let consequences = self.selection.reads_consequences().then(|| {
            let results = self.size_tab_results.get(shown);
            self.consequences
                .get(shown, results, self.minimum_diameter_mm, self.clamps_paused)
        });
        let asked = egui::ScrollArea::vertical()
            .id_salt("magcoupling_explain_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let shown = crate::gui::explain::Shown {
                    inputs: shown,
                    results: &self.results,
                    fixes,
                    clamps_paused: self.clamps_paused,
                    consequences,
                    auto,
                };
                explain_ui(ui, self.selection, &shown, readouts)
            })
            .inner;
        match asked {
            Some(Asked::Select(selection)) => self.selection = selection,
            Some(Asked::Equation(path)) => self.explorer.open_path(path),
            Some(Asked::StartHere) => self.explorer.start(0),
            None => {}
        }
        if asked.is_some() {
            ui.ctx().request_repaint();
        }
    }
```

Replace:

```rust
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        // What the torque card's open-or-closed rule reads (slice 3 spec 6.2), once, before the
        // Equation panel takes its part of the region.
        let card = CardFrame {
            view: CardView::of(ui.ctx(), self.explorer.open),
            region_width: ui.available_rect_before_wrap().width(),
        };
        // The Equation panel docks at the bottom of the centre region (decision M43-1). It
```

with:

```rust
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        // The Equation panel docks at the bottom of the centre region (decision M43-1). It
```

Replace:

```rust
        confined(ui, "magcoupling_above_equation_panel", |ui| {
            if !self.welcome_dismissed {
                self.welcome_ui(ui);
            }
            ui.horizontal_wrapped(|ui| {
```

with:

```rust
        confined(ui, "magcoupling_above_equation_panel", |ui| {
            ui.horizontal_wrapped(|ui| {
```

Replace:

```rust
            confined(ui, "magcoupling_centre_view", |ui| {
                self.view_ui(ui, shown, readouts, card);
            });
```

with:

```rust
            confined(ui, "magcoupling_centre_view", |ui| {
                self.view_ui(ui, shown, readouts);
            });
```

Replace:

```rust

    /// The Size tab's torque readouts (spec 3.2), each label quieter than its value
    /// ([`look::quiet_text`]): the requirement's slider, an ordinary design
    /// edit (undo, files, links); the hot-low torque with its gap to the requirement, red short
    /// or green margin; the nominal pull-out at the operating temperature. The Required and
    /// Hot-low rows end with the Why link to the hot minimum check's note (spec 3.7). Where the
    /// model makes no prediction a torque reads [`size_controls::NO_PREDICTION`] (or
    /// `NO_TORQUE`) and the gap is left out.
    fn torque_readouts_ui(&mut self, ui: &mut egui::Ui, readouts: &mut Readouts) {
        let entry = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .expect("the requirement has input metadata");
        let range = entry.meta.range.expect("the requirement has a slider");
        let required = self.inputs.metal.required_min_Nm;
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(look::quiet_text(ui.visuals()), size_controls::REQUIRED);
            let mut value = required;
            let response = ui
                .push_id(REQUIREMENT_INPUT, |ui| {
                    ui.add(slider(&mut value, entry.meta, range))
                })
                .inner;
            // Its value box is a text field of the design, as an input row's is; the next
            // frame's inputs side lists it before the solve looks (`requirement_widget`).
            if !self.design_widgets.contains(&response.id) {
                self.design_widgets.push(response.id);
            }
            self.requirement_widget = Some(response.id);
            if response.changed() && value != required {
                self.last_error = self
                    .inputs
                    .set(REQUIREMENT_INPUT, Value::Num(value))
                    .err()
                    .map(|e| e.to_string());
                // The dashboard was drawn before this edit: the readouts below show it now, the
                // dashboard on the next frame.
                self.results = compute_all(&self.shown_inputs());
                ui.ctx().request_repaint();
            }
            if why_note_link(ui, TORQUE_BAND_NOTE) {
                readouts.open_note(TORQUE_BAND_NOTE);
            }
        });
        let required = self.inputs.metal.required_min_Nm;
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(look::quiet_text(ui.visuals()), size_controls::HOT_LOW);
            if let Some(hot_low) =
                torque_readout(ui, &self.results, "metal.torque_hot_low_Nm", readouts)
            {
                let (gap, meets) = size_controls::torque_gap_text(hot_low, required);
                let level = if meets { Level::Good } else { Level::Bad };
                ui.colored_label(level.color(ui.visuals()), gap);
            }
            if why_note_link(ui, TORQUE_BAND_NOTE) {
                readouts.open_note(TORQUE_BAND_NOTE);
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                look::quiet_text(ui.visuals()),
                size_controls::pullout_label(self.results.metal.op_temp_C),
            );
            torque_readout(ui, &self.results, "model.pullout_Nm", readouts);
        });
    }

    /// The welcome strip (spec 3.1): what to do first, wrapping in a narrow region, and the
    /// button that hides it for the rest of the session.
    fn welcome_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(WELCOME);
            if ui
                .small_button(HIDE_WELCOME)
                .on_hover_text(HIDE_WELCOME_HINT)
                .clicked()
            {
                self.welcome_dismissed = true;
            }
        });
    }

    /// The torque card (slice 3 spec 6.1), between the Size tab's readouts and its drawing: a
    /// collapsing header whose one wrapping row holds the heading, which opens and closes the
    /// card as its arrow does, and the Why link to how the coupling makes torque; under it the
    /// card's lines for `shown`, the design shown after this frame's edits (spec 3.5), weak like
    /// the lines under the controls, in a body that scrolls above the drawing's floor (spec 6.2).
    /// Open or closed: the user's own choice for the session, else the default for `card`'s view
    /// ([`MagcouplingPanel::card_default`]). It asks the cache as the lines under the controls
    /// do ([`MagcouplingPanel::size_tab_consequences`]), so the cache's key and the results it is
    /// handed come from one design whatever a later change puts between the panel's results and
    /// this view.
    fn torque_card_ui(
        &mut self,
        ui: &mut egui::Ui,
        readouts: &mut Readouts,
        card: CardFrame,
        shown: &DesignInputs,
    ) {
        let auto = self.auto();
        let lines = self.size_tab_consequences(shown).card.lines(auto);
        let open = match self.card_choice {
            Some(choice) => choice,
            None => self.card_default(ui, card, &lines),
        };
        // `open` decides on every frame (`set_open`): egui keeps a card it already holds as it
        // left it, so without it a default decided again for a new view would not reach the
        // card. The default argument only builds the state of a card egui does not hold yet
        // (egui has no other way to build one), where it is `open` too.
        let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            egui::Id::new(TORQUE_CARD_ID),
            open,
        );
        state.set_open(open);
        // The heading toggles the card as the arrow does. `HeaderResponse` keeps the closure's
        // return private until `body`, so the click is recorded here.
        let mut heading_clicked = false;
        let mut header = state.show_header(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                heading_clicked = ui
                    .add(egui::Label::new(CARD_HEADING).sense(egui::Sense::click()))
                    .clicked();
                if why_note_link(ui, SIZE_TAB_NOTE) {
                    readouts.open_note(SIZE_TAB_NOTE);
                }
            });
        });
        if heading_clicked {
            header.toggle();
        }
        // A click on the arrow (inside `show_header`) or on the heading is the user's choice.
        if header.is_open() != open {
            self.card_choice = Some(header.is_open());
        }
        header.body(|ui| {
            // The body scrolls inside a cap that keeps the drawing's floor under the card: the
            // room over the floor left under the header (`room_over_the_floor`, the room rule 3
            // reads at the card), and at least one row. The cap is also the body's least height
            // while it scrolls: egui's own (64 points) would hold it taller and push the drawing
            // under its floor beside the Equation panel.
            let cap = room_over_the_floor(ui).max(ui.text_style_height(&egui::TextStyle::Body));
            egui::ScrollArea::vertical()
                .id_salt((TORQUE_CARD_ID, "body"))
                .max_height(cap)
                .min_scrolled_height(cap)
                .show(ui, |ui| {
                    for line in &lines {
                        weak_line(ui, line);
                    }
                });
        });
    }

    /// The torque card's default for `card`'s view (slice 3 spec 6.2): the one decided for that
    /// view, else decided now and kept with it, so a design change never decides it again. Open
    /// iff the Equation panel is closed (rule 1), the centre region is at least
    /// [`CARD_OPEN_MIN_WIDTH`] wide (rule 2) and the open card leaves the drawing its floor
    /// (rule 3: the card's height measured before it is drawn, [`card_open_height`] of `lines`,
    /// is at most the room `ui` has at the card over the drawing's floor,
    /// [`room_over_the_floor`], which the card's body is capped at too). Rule 3 is read last:
    /// its pre-measure lays every line out, so it runs only where rules 1 and 2 open the card.
    fn card_default(&mut self, ui: &egui::Ui, card: CardFrame, lines: &[String]) -> bool {
        if let Some((view, open)) = self.card_start
            && view == card.view
        {
            return open;
        }
        let open = !card.view.equation_open
            && card.region_width >= CARD_OPEN_MIN_WIDTH
            && card_open_height(ui, lines) <= room_over_the_floor(ui);
        self.card_start = Some((card.view, open));
        open
    }

    /// The view of the centre region the tabs select, of `shown`, the design shown, its values
```

with:

```rust

    /// The view of the centre region the tabs select, of `shown`, the design shown, its values
```

Replace:

```rust
    /// The view of the centre region the tabs select, of `shown`, the design shown, its values
    /// readouts (`readouts`), with what the torque card's open-or-closed rule reads of the frame
    /// (`card`). Each view fits itself to `ui`'s available height.
    fn view_ui(
        &mut self,
        ui: &mut egui::Ui,
        shown: &DesignInputs,
        readouts: &mut Readouts,
        card: CardFrame,
    ) {
        match self.centre {
```

with:

```rust
    /// The view of the centre region the tabs select, of `shown`, the design shown, its values
    /// readouts (`readouts`). Each view fits itself to `ui`'s available height. The opening
    /// tab (UX overhaul spec 4.4): the four tiles, a click on one selecting it for the Explain
    /// panel; the "Exploration only" lines and the keyway's advisory, only when true; then the
    /// drawing and its callout list in all the height left (the plan's decision D-19).
    fn view_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        match self.centre {
```

Replace:

```rust
            CentreView::SizeAndTorque => {
                self.torque_readouts_ui(ui, readouts);
                if !crate::engine::model::blocks_fit(&shown.coupling, &self.results.model) {
```

with:

```rust
            CentreView::SizeAndTorque => {
                let selected = self.selection.tile();
                if let Some(tile) = tiles_ui(ui, shown, &self.results, selected, readouts) {
                    // The Explain panel was drawn before this click: the next frame shows it.
                    self.selection = Selection::Result(tile.path());
                    ui.ctx().request_repaint();
                }
                if !crate::engine::model::blocks_fit(&shown.coupling, &self.results.model) {
```

Replace:

```rust
                }
                // The design shown after this view's one edit, the Required slider's above (slice
                // 3 spec 3.5): the card and the drawing below read this one copy, built once a
                // frame (with Auto on the ring size each build runs two geometries).
                let shown_after_edits = self.shown_inputs();
                self.torque_card_ui(ui, readouts, card, &shown_after_edits);
                geometry_ui(
                    ui,
                    &shown_after_edits,
                    &self.results,
                    self.clamps_paused,
                    readouts,
                );
            }
```

with:

```rust
                }
                geometry_ui(ui, shown, &self.results, self.clamps_paused, readouts);
            }
```

Replace:

```rust
        if self.centre == CentreView::SizeAndTorque {
            ui.heading(crate::gui::size_controls::HEADING);
            // Auto (spec guidance 3.5) is the solver's one home. Turning it off adopts the value
            // shown (decision M41-7) through the mode switch.
            let asked = size_controls::auto_ui(ui, self.sizing);
            self.sizing.variable = asked.variable;
            self.set_sizing_mode(asked.mode);
            let auto = self.auto();
            let shown = self.shown_inputs();
            if auto.is_some() {
                let (line, color) = self.auto_outcome(&shown, ui.visuals());
                ui.add(egui::Label::new(egui::RichText::new(line).color(color)).wrap());
            }
            let minimum = self.minimum_diameter_mm;
            // Asked before the controls write the minimum back, with the design before this
            // frame's edits.
            let consequences = self.size_tab_consequences(&shown);
            let output = size_controls::controls_ui(ui, &shown, minimum, consequences, auto);
            // A Why link under Magnets per ring or Outside diameter (spec 3.7) opens its note.
            if let Some(id) = output.note {
                self.explorer.open_note(id);
                ui.ctx().request_repaint();
            }
            // The lines under the controls were drawn from the design before this frame's edit:
            // one more frame shows the edited design's.
            if output.inputs != shown || output.minimum_diameter_mm != minimum {
                ui.ctx().request_repaint();
            }
            self.design_widgets.extend(output.widgets);
            // The Required slider is drawn after `run_sizing`: last frame's id stands in.
            self.design_widgets.extend(self.requirement_widget);
            self.minimum_diameter_mm = output.minimum_diameter_mm;
            if output.inputs != shown {
                // The controls edit the design shown: the inputs keep their own value of the size
                // Auto sets until Auto turns off (decision M41-7).
                self.inputs = match auto {
                    Some(variable) => with_own_value(variable, output.inputs, &self.inputs),
                    None => output.inputs,
                };
                self.last_error = output.error;
            } else if output.error.is_some() {
                self.last_error = output.error;
            }
            ui.separator();
            egui::CollapsingHeader::new("Detailed inputs")
                .default_open(false)
                .open(
                    self.explorer
                        .focus()
                        .filter(|focus| focus.scroll)
                        .map(|_| true),
                )
                .show(ui, |ui| self.design_inputs_ui(ui, readouts));
            return;
```

with:

```rust
        if self.centre == CentreView::SizeAndTorque {
            // The column scrolls in a short window: its overflow would stretch the panel past
            // the window's foot, and the centre region and its drawing with it.
            egui::ScrollArea::vertical()
                .id_salt("magcoupling_size_column")
                .auto_shrink([false, true])
                .show(ui, |ui| self.size_column_ui(ui, readouts));
            return;
```

Replace:

```rust

    /// The Assumptions view (spec Addendum A3 "Toggle panel"): each assumption with its
```

with:

```rust

    /// The Size column of the opening tab: its heading, the Required torque row, Auto and its
    /// outcome, the size controls with their lines, the detailed inputs, folded, then the
    /// footnote on what the size controls move.
    fn size_column_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        ui.heading(crate::gui::size_controls::HEADING);
        let requirement = self.requirement_ui(ui);
        // Auto (spec guidance 3.5) is the solver's one home. Turning it off adopts the value
        // shown (decision M41-7) through the mode switch.
        let asked = size_controls::auto_ui(ui, self.sizing);
        self.sizing.variable = asked.variable;
        self.set_sizing_mode(asked.mode);
        let auto = self.auto();
        let shown = self.shown_inputs();
        if auto.is_some() {
            let (line, color) = self.auto_outcome(&shown, ui.visuals());
            ui.add(egui::Label::new(egui::RichText::new(line).color(color)).wrap());
        }
        let minimum = self.minimum_diameter_mm;
        // Asked before the controls write the minimum back, with the design before this
        // frame's edits.
        let consequences = self.size_tab_consequences(&shown);
        let output = size_controls::controls_ui(ui, &shown, minimum, consequences, auto);
        // A Why link under Magnets per ring or Outside diameter (spec 3.7) opens its note.
        if let Some(id) = output.note {
            self.explorer.open_note(id);
            ui.ctx().request_repaint();
        }
        // The lines under the controls were drawn from the design before this frame's edit:
        // one more frame shows the edited design's.
        if output.inputs != shown || output.minimum_diameter_mm != minimum {
            ui.ctx().request_repaint();
        }
        self.design_widgets.extend(output.widgets);
        // After the controls' own: the tests find those by their place in the list.
        self.design_widgets.push(requirement);
        self.minimum_diameter_mm = output.minimum_diameter_mm;
        if output.inputs != shown {
            // The controls edit the design shown: the inputs keep their own value of the size
            // Auto sets until Auto turns off (decision M41-7).
            self.inputs = match auto {
                Some(variable) => with_own_value(variable, output.inputs, &self.inputs),
                None => output.inputs,
            };
            self.last_error = output.error;
        } else if output.error.is_some() {
            self.last_error = output.error;
        }
        ui.separator();
        egui::CollapsingHeader::new("Detailed inputs")
            .default_open(false)
            .open(
                self.explorer
                    .focus()
                    .filter(|focus| focus.scroll)
                    .map(|_| true),
            )
            .show(ui, |ui| self.design_inputs_ui(ui, readouts));
        // Under the fold, so the fold is on the first view (plan review E-4).
        ui.label(size_controls::LINKED_HOUSING);
    }

    /// The Required torque row that leads the Size column (UX overhaul spec 4.3; it was the
    /// first of the Size tab's readouts): its label, quieter than the value in the slider's box
    /// ([`look::quiet_text`]), with the Why link to the hot minimum check's note (spec 3.7),
    /// then the requirement's slider, an ordinary design edit (undo, files, links). Returns the
    /// slider's widget: its value box is a text field of the design, as an input row's is.
    fn requirement_ui(&mut self, ui: &mut egui::Ui) -> egui::Id {
        let entry = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .expect("the requirement has input metadata");
        let range = entry.meta.range.expect("the requirement has a slider");
        let required = self.inputs.metal.required_min_Nm;
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(look::quiet_text(ui.visuals()), size_controls::REQUIRED);
            if why_note_link(ui, TORQUE_BAND_NOTE) {
                self.explorer.open_note(TORQUE_BAND_NOTE);
                ui.ctx().request_repaint();
            }
        });
        let mut value = required;
        let response = ui
            .push_id(REQUIREMENT_INPUT, |ui| {
                ui.add(slider(&mut value, entry.meta, range))
            })
            .inner;
        if response.changed() && value != required {
            self.last_error = self
                .inputs
                .set(REQUIREMENT_INPUT, Value::Num(value))
                .err()
                .map(|e| e.to_string());
            ui.ctx().request_repaint();
        }
        response.id
    }

    /// The Assumptions view (spec Addendum A3 "Toggle panel"): each assumption with its
```

Replace:

```rust
    }
}

/// One torque readout of the Size tab: the value of the result at `path` with its unit, a
/// readout (`readouts`: hover text and equation; a click opens it), or
/// [`size_controls::NO_PREDICTION`] / [`size_controls::NO_TORQUE`] where the model makes none
/// ([`size_controls::predicts`]). Returns the torque shown [N·m], if any.
fn torque_readout(
    ui: &mut egui::Ui,
    results: &DesignResults,
    path: &str,
    readouts: &mut Readouts,
) -> Option<f64> {
    let Some(Value::Num(torque)) = results.get(path) else {
        unreachable!("{path} is a torque result")
    };
    if !size_controls::predicts(results, torque) {
        ui.weak(if end_effect_banner(results).is_some() {
            size_controls::NO_PREDICTION
        } else {
            size_controls::NO_TORQUE
        });
        return None;
    }
    let text = with_unit(format_value(&Value::Num(torque)), "N\u{b7}m");
    let response = ui.add(egui::Label::new(text).sense(egui::Sense::click()));
    readouts.show(ui, response, path, |show_details| {
        crate::gui::dashboard::hover_text(path, show_details).unwrap_or_default()
    });
    Some(torque)
}

/// The torque card's height when open [points], measured before it is drawn (slice 3 spec 6.2):
/// its header row laid out as [`MagcouplingPanel::torque_card_ui`] draws it (beside the arrow,
/// an indent wide, the heading, then the Why link continuing after it and wrapping under it in a
/// narrow region, as egui lays a label out in a wrapping row, the row at least an interactive
/// widget's height), then the item spacing and each of `lines` as a galley wrapped at the body's
/// width (the row's, less the body's indent), with the item spacing between them. It repeats
/// egui 0.32.3's label layout (`widgets/label.rs`), so an egui upgrade can move it:
/// `the_card_s_measured_height_is_the_drawn_height` pins it to the drawn height.
fn card_open_height(ui: &egui::Ui, lines: &[String]) -> f32 {
    let spacing = ui.spacing();
    let width = ui.available_width() - spacing.indent;
    // A label's galley as egui lays it out: wrapped at `width`, its first row starting
    // `indentation` in and at least `first_row` tall.
    let galley = |text: &str, indentation: f32, first_row: f32| {
        let mut job =
            std::sync::Arc::unwrap_or_clone(egui::WidgetText::from(text).into_layout_job(
                ui.style(),
                egui::FontSelection::Default,
                ui.text_valign(),
            ));
        job.wrap.max_width = width;
        job.first_row_min_height = first_row;
        job.halign = egui::Align::Min;
        job.justify = false;
        if let Some(section) = job.sections.first_mut() {
            section.leading_space = indentation;
        }
        ui.fonts(|fonts| fonts.layout_job(job))
    };
    let row = spacing.interact_size.y;
    let heading = galley(CARD_HEADING, 0.0, row).size();
    let header = match reviewed_note(SIZE_TAB_NOTE) {
        Some(note) => galley(
            &why_note_text(note),
            heading.x + spacing.item_spacing.x,
            row,
        )
        .size()
        .y
        .max(heading.y),
        // A note not reviewed has no link (`why_note_link` draws none). The release gate keeps
        // this note reviewed, so only a later change to the note reaches this arm.
        None => heading.y,
    };
    let body: f32 = lines
        .iter()
        .map(|line| galley(line, 0.0, 0.0).size().y)
        .sum::<f32>()
        + spacing.item_spacing.y * lines.len().saturating_sub(1) as f32;
    header + spacing.item_spacing.y + body
}

/// The room `ui` has over the drawing's floor [points] (slice 3 spec 6.2): its available height
/// less the item spacing before the drawing and [`drawing_floor`]. Rule 3 reads it at the
/// torque card, where the open card must fit in it ([`MagcouplingPanel::card_default`]), and
/// the card's body reads it under the header as its cap
/// ([`MagcouplingPanel::torque_card_ui`]): one expression, so the rule and the cap agree.
fn room_over_the_floor(ui: &egui::Ui) -> f32 {
    ui.available_height() - ui.spacing().item_spacing.y - drawing_floor(ui)
}
```

with:

```rust
    }
}
```

**`magcoupling-rs/src/gui/size_controls.rs`**

Replace:

```rust

/// The label of the hot-low torque (`metal.torque_hot_low_Nm`), its gap to the requirement
/// beside it ([`torque_gap_text`]): the dashboard row's display label.
pub const HOT_LOW: &str = HOT_LOW_TORQUE;
```

with:

```rust

/// The label of the hot-low torque's tile (`metal.torque_hot_low_Nm`), what it must reach and
/// its gap to the requirement under it ([`torque_gap_text`]): the result's display label.
pub const HOT_LOW: &str = HOT_LOW_TORQUE;
```

Replace:

```rust
        }
        ui.label(LINKED_HOUSING);
    });
```

with:

```rust
        }
    });
```

- [ ] **Step 4: Run the checks**, then the whole crate both ways and wasm32:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings
  ```

  Expected: `test result: ok. 1069 passed; 0 failed; 6 ignored` for the lib tests with the app feature; every other test result ok; all four clippy runs clean; `cargo fmt` changes nothing. A red here that the blocks did not foresee is a stop: report it.

- [ ] **Step 5: Commit** the eight files: `feat(magcoupler): the instrument's columns: four tiles over the drawing, the check list over the Explain panel, the fixed-3D line on the first view (BL-081 T5)`.

### Task 6: Docs sweep

**Model:** `fable`.

The README's first view, Required torque, torque card, dashboard, label colour and smoke-run passages and its test table; `docs/ai/` (02 the layout, 03 the modules, 04 the decisions, 05 the tracker, the backlog item in progress); the register's rows A-20 and A-21 and its log; owner question 49; the design's status line.

**Files:**
- Modify: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/analyses/2026-11-18-owner-questions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

- [ ] **Step 1: The docs.**

**`magcoupling-rs/README.md`**

Replace:

```markdown
A one-line strip over the tabs, "Set the size -> read the torque -> open any red check to see
why and what to change.", closes for the session (it is in no file, link or undo step). Adjust
```

with:

```markdown
Since slice B, stage 2, part 1 of the UX overhaul (BL-081, plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`)
the first view is the instrument of the owner's option B: the Size column on the left, led by
**Required torque**; in the centre four tiles over the drawing, each a readout whose number leads
and whose label is muted under it (**Pull-out torque (nominal, at T)**, **Hot-low torque** with
"needs" and the requirement and its gap under its label, **Cold-high torque** and the rotating
mass; `gui::tiles`, two rows of two where four would be narrower than 130 points), and under the
drawing its callout list, as on the Geometry tab, until part 2 makes the dimension tags
selections (the plan's decision D-19); on the right the check list over the Explain panel
(`gui::checks`, `gui::explain`). The check list is the design status line, then one row per
counted check in one fixed order, each a shape and a word as well as a colour (a filled disc and
"Fails", a triangle and "Check", a ring and "OK"; every mark of a state on the page is one of
these shapes, `checks::state_mark`), then the material warnings that fire and the line that says
two checks read stored 3D fields: at 1280 x 800 that line is on the first view. A click on a row
or a tile selects it, and the Explain panel shows its entry under the headings "What it is",
"What sets it", "What to change", "Watch out" and "Numbers behind this check" for a check, "What
it is", "What a change does" and "Checks it touches" for a tile, then "Full lesson:" and, for a
result, "Equation"; a check's entry holds exactly the Problem, Why sentence, fixes and notes the
dashboard drew under its row. With nothing selected the panel shows its orientation (what the
coupling is and the three moves) and the link to Start here; Esc or "About this tool" goes back
to it, and a hover selects nothing. The selection is view state, in no file, link or undo step.
The welcome strip is gone (the orientation replaces it), and the Size column scrolls in a short
window. Adjust
```

Replace:

```markdown
magnet length of both rings; the readouts are **Required torque**, **Hot-low torque** and the
nominal pull-out.
```

with:

```markdown
magnet length of both rings; the Required torque row leads the column and the tiles read the
torques.
```

Replace:

```markdown
build checks stay in the dashboard.
```

with:

```markdown
build checks stay in the check list.
```

Replace:

```markdown
**Required torque** is a slider on `metal.required_min_Nm`, the requirement the dashboard's checks
compare against
```

with:

```markdown
**Required torque** is the Size column's first row, a slider on `metal.required_min_Nm`, the
requirement the checks compare against
```

Replace:

```markdown
links carry it. **Hot-low torque** (`metal.torque_hot_low_Nm`) is shown with its gap to
```

with:

```markdown
links carry it. **Hot-low torque** (`metal.torque_hot_low_Nm`, the second tile, its gap breaking
only between the torque and the percent) is shown with its gap to
```

Replace:

```markdown
`docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md`) sits between the readouts
and the drawing, in place of a label that pointed at the dashboard: a collapsing header, "One
step of each size control", whose heading opens and closes it as its arrow does, with the
```

with:

```markdown
`docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md`) is the "What a change
does" part of the Hot-low tile's Explain entry (UX overhaul decision C-10, since BL-081; until
then a collapsing header between the readouts and the drawing): the heading "One
step of each size control", with the
```

Replace:

```markdown
shaft-clamp pause; the Auto marking is applied outside the cache). It starts open, but closed while the Equation panel is open
```

with:

```markdown
shaft-clamp pause; the Auto marking is applied outside the cache), and asks it only while the Hot-low tile is selected. Until BL-081 it started open, but closed while the Equation panel was open
```

Replace:

```markdown
- **Dashboard** (`dashboard.rs`): the 15 headline numbers
```

with:

```markdown
- **Dashboard rows** (`dashboard.rs`; since BL-081 no one list draws them all: the check list
  draws a row per check, a check's Explain entry draws the rows its verdict badges under
  "Numbers behind this check" with `line_ui`, and the Results tab's Headline group lists every
  row): the 15 headline numbers
```

Replace:

```markdown
(`look::quiet_text`; the spec's 6.1 gives muted to labels and captions): a dashboard row's
  title, the heads "Why:", "Problem:" and "Fix:" and the Size tab's three readout labels, so a
```

with:

```markdown
(`look::quiet_text`; the spec's 6.1 gives muted to labels and captions): a row's label
  under an entry's "Numbers behind this check", a tile's label and the word "needs" on the
  Hot-low tile (the requirement after it is a value, in the text colour while it is met), the
  right column's headings and an entry's headings, and the Required torque row's label, so a
```

Replace:

```markdown
`/tools/magcoupler/` through a pinned share link and checks the canvas, the Size and torque tab with its
torque card (open, between the readouts and the drawing) and geometry drawing in the first screenshot (the first view: no click), the console lines
```

with:

```markdown
`/tools/magcoupler/` through a pinned share link and checks the canvas, the Size and torque tab with its
four tiles over the drawing and the check list over the Explain panel's orientation in the first screenshot (the first view: no click; since BL-081), the console lines
```

Replace:

```markdown
picker and checks `magcoupling: loaded a design file`; then (spec R14, guidance slice 2) it opens a teaching note
through the first dashboard row's Why link (the Equation panel shows "What to change" and the closed "Where this
comes from" fold), hovers
```

with:

```markdown
picker and checks `magcoupling: loaded a design file`; then it clicks the check row "Hot-low torque" and
reads its Explain entry down to "Full lesson:" (`check_entry`), and opens a teaching note
through that lesson link (`lesson_link_opened`: the Equation panel shows "What to change" and the closed "Where this
comes from" fold), hovers
```

Replace:

```markdown
checks) and scrolls the dashboard to its last row, "Magnets fit", in a 1280 x 800 window; then (slice 3) it opens the link again
```

with:

```markdown
checks); then (slice 3) it opens the link again
```

Replace:

```markdown
checks the torque card's heading with its Why link, caption, Auto line and four rows
```

with:

```markdown
clicks the Hot-low tile and checks the torque card's heading with its Why link, caption, Auto line and four rows in its Explain entry
```

Replace:

```markdown
errors (a test decodes the pinned link and the design file, and checks that the steps name the welcome strip, the
toggle,
```

with:

```markdown
errors (a test decodes the pinned link and the design file, and checks that the steps name the right column's
headings, the orientation, the entry's headings and its lesson link, and no dashboard, the
toggle,
```

Replace:

```markdown
| `src/gui/guidance.rs` (feature `gui`) | 
```

with:

```markdown
| `src/gui/checks.rs`, `src/gui/tiles.rs`, `src/gui/explain.rs` (feature `gui`) | The check list is the status line, a row per counted check in `CHECK_GUIDES` order (the clamp's only while it shows) and the stored-3D line last; each state has its own shape and word as well as its colour, the material warnings' and the Results tab's marks too, and a check with no badge has neither; a click returns the check, a hover does not; the selected row is raised. The four tiles show the results' own numbers over their muted labels, equally tall; the Hot-low tile says what it needs (the word muted, the requirement a value) and its gap, which breaks only between its torque and its percent, and fails by shape and red on that line, never on the number; hot-low and the requirement never read as two equal numbers on a failing tile; a torque the model does not predict reads so; the tiles wrap to two rows of two in a narrow region with every text inside its tile. The Explain panel opens on the orientation (at most 60 words); a check's entry holds exactly the guide's Problem, Why, fixes (each a link that asks for its input's row) and notes, and the rows its verdict badges, each with its state's shape; an amber check's entry is open at its one click; the Hot-low tile's entry holds the torque card and names the check of its row, a link that selects it; the Equation link opens the Equation panel and About this tool returns the orientation; every new string has its glyphs and plain words. |
| `src/gui/guidance.rs` (feature `gui`) | 
```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
look.rs (the panel's own part of the look, UX overhaul slice B stage 1:
```

with:

```yaml
checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links), look.rs (the panel's own part of the look, UX overhaul slice B stage 1:
```

**`docs/ai/02-system.yaml`**

Replace:

```yaml
      - "The first view is Size and torque (decision G-5; reverses M42-2):
```

with:

```yaml
      - "Since BL-081 (UX overhaul slice B, stage 2, part 1; plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md): the right column is gui::checks over gui::explain, the opening tab is gui::tiles over geometry_view::geometry_ui (the drawing and its callout list, which leaves for the Geometry tab with part 2's tag selections, decision D-19), and the Size column starts with the Required torque row and scrolls in a short window. The panel owns one explain::Selection (Nothing, a check, a tile's result): view state, set by a click on a check's row, a tile or a tile entry's check, cleared by Esc (unless a text field or a popup has the key) or About this tool; a hover never sets it. An entry's Equation link opens the Equation panel and leaves the selection (C-25); the orientation's Start here link starts the walk. The dashboard's list (dashboard_ui), its open Problem, Why and Fix blocks, the welcome strip (WELCOME) and the centre's torque readouts and collapsing torque card are gone; their texts are the entries' (the plan's nothing-lost table). Every state's mark is checks::state_mark's shape (C-4). The check list and the Explain panel add no compute_all to an idle frame: the consequences are asked of the ConsequenceCache only while the Hot-low tile is selected."
      - "The first view is Size and torque (decision G-5; reverses M42-2):
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
  - "Payload weights Track 2: before merge the user runs
```

with:

```yaml
  - "Magcoupler UX overhaul, slice B, stage 2, part 1 (BL-081, plan docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md, decisions D-1 to D-19): the check list over the Explain panel and four tiles over the drawing. Stage 2 is cut in two (D-1): part 2 is the top bar and Design menu, Learn, the Size column's folds and its hint lines moving into the Explain panel with inputs as selections, dimension tags as selections with the callout list leaving the opening tab (D-19), the Plots tab with its picker, the word budget of 250 and the slip-heating unit. A check's row shows a shape and a word, not a short form of its quantity (D-2, register row A-20, owner question 49). risk: physics for the orientation and the tile's needs line, so every task runs on fable."
  - "Payload weights Track 2: before merge the user runs
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown
Reverse chronological (newest at top).

---

```

with:

```markdown
Reverse chronological (newest at top).

---

## 2026-11-22 - Magcoupler UX overhaul, slice B, stage 2, part 1: the instrument's columns (BL-081)
- What changed: the right column is the check list (`gui::checks`: the status line, a row per counted check in one fixed order with a shape and a word for its state, the material warnings and the stored-3D line, which is now on the first view at 1280 x 800) over the Explain panel (`gui::explain`: the orientation on the first frame; a click on a check shows its Problem, Why sentence, fixes, notes and numbers; a click on a tile shows its help and, for hot-low, the torque card and the check of its row). The opening tab is four tiles over the drawing and its callout list (`gui::tiles`); Required torque leads the Size column, which scrolls in a short window, and the footnote on what the size controls move sits under the Detailed inputs fold, which is on the first view. Every state's mark on the page is a shape (`checks::state_mark`): the dashboard's filled-disc badge is gone from the entries, the material warnings and the Results tab. The dashboard's list, the welcome strip and the centre's readouts and card are gone, each text to the home the plan's nothing-lost table names.
- Tests: each module's own, red first; `the_first_view_keeps_its_three_columns_its_tiles_and_the_fixed_3d_line_in_the_app_s_own_font` (1024 x 700, 1280 x 800, 1920 x 1080 in the app's theme and font, the starting and the example design) and `the_first_view_draws_at_most_four_hundred_and_fifty_words_at_1280_by_800` (608 words before, 443 after); `a_fix_link_on_the_opening_tab_shows_its_row_inside_a_short_window`; the gui-smoke prompt reads the new layout (`check_entry` and `lesson_link_opened` replace `dashboard_scrolls` and `why_link_opened`).
- Not in this part (plan decision D-1): the top bar and Design menu, Learn, the Size column's folds and hint lines in the Explain panel, inputs and dimension tags as selections (the callout list then leaves the opening tab), the Plots tab's picker, the word budget of 250, the slip-heating readout's stray unit.

```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
Stage 2 draws its own selected row and may take the choices off this fill. | BL-079; plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`, decision L-3 |

```

with:

```markdown
Stage 2 draws its own selected row and may take the choices off this fill. | BL-079; plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`, decision L-3 |
| A-20 | The design's check list (4.5) gives each row "the check's own quantity in short form where it has one" and leaves the short forms to the builder (section 11). What does a row show? | Its state as a shape and a word (OK, Check, Fails) beside the check's name; no quantity. The check's numbers are in its Explain entry, one click away, under "Numbers behind this check". | A short form is a new way of stating a number, which needs its own text and physics review, and nine of them would compete with the four tiles. The word is the badge's existing hover word. | Each guide gains a short-form function from the numbers its Problem already quotes, and the row draws it right-aligned before the word; a physics review of the nine forms. | BL-081; plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`, decision D-2; owner question 49 |
| A-21 | Slice B's stage 2 is the whole of sections 4 and 5. Built at once or in parts? | Two parts. The first is the right column (the check list, the Explain panel with its orientation, check and tile entries), the four tiles, the Required torque row and the fixed-3D line on the first view; the opening tab keeps the callout list under the drawing. The second is the top bar and its Design menu, Learn, the Size column's folds and hint lines in the Explain panel, inputs and dimension tags as selections (the callout list then moves to the Geometry tab), the Plots tab and the word budget of 250. | Smaller releases (Q-4's answer); each part ships with nothing visible today lost, and the first changes the first view the owner called outdated. | The second part is planned and built with the first's branch before either merges. | BL-081; the plan's decisions D-1 and D-19 |

```

Replace:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.

```

with:

```markdown
- 2026-11-22: A-19 added by the fix wave of stage 1's whole-branch review (BL-079; findings C-1, R-3, E-5): the selection fill, the one colour outside the design's token table. It is the builder's call, not the owner's answer.
- 2026-11-22: A-20 and A-21 added with the plan of slice B's stage 2, part 1 (BL-081). Each is an assumption, not the owner's answer. A-18's line is met there: the check list closes with the stored-3D line, on the first view at 1280 x 800, and an app test holds it.

```

**`docs/analyses/2026-11-18-owner-questions.md`**

Replace:

```markdown
Part C, 29 calls to make when convenient (question 48 was added there on 2026-11-22).
```

with:

```markdown
Part C, 30 calls to make when convenient (questions 48 and 49 were added there on 2026-11-22).
```

Replace:

```markdown
Source: BL-080 (a by-product of the 3D run for A-18); BL-065; question 15.

```

with:

```markdown
Source: BL-080 (a by-product of the 3D run for A-18); BL-065; question 15.

### 49. Should a check's row show a short form of its number, or only its state?

Added 2026-11-22 (the next free number). Since the layout stage each check is one row in the right column: a shape, its name and a word (OK, Check or Fails). The design asked for the check's own number in short form on the row too, such as "0.26 / 0.20 mm" on Running clearance; the numbers are one click away, in the check's Explain entry.

- **If you do not answer:** the row shows the shape, the name and the word.
- (a) Shape, name and word, as now.
- (b) Add a short form of the check's number to each row (nine new short texts, each with a physics review).

Source: UX design section 4.5; A-20; BL-081 plan, decision D-2.

```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown
is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the rest of slice B (sections 4 and 5) and slices C to F are not built.**
```

with:

```markdown
is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part and slices C to F are not built.**
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: open
  notes: "Filed with its plan on 2026-11-22 (the design's header: each plan files one item).
```

with:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-the-instrument (Tasks 1 to 6 of the plan); the full gate, the smoke run, the physics review and the whole-branch review are its Task 7. Filed with its plan on 2026-11-22 (the design's header: each plan files one item).
```

- [ ] **Step 2: Check.** `git diff --stat` lists the nine files and no other; `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3` still reads `1069 passed`.

- [ ] **Step 3: Commit** the nine files: `docs(magcoupler): slice B stage 2 part 1 docs sweep: README, docs/ai, the register, the owner questions, the design's header, BL-081 in progress (BL-081 T6)`.

### Task 7: Release: the full gate, gui-smoke, the reviews, one fix wave, the merge and the push

**Model:** the `fable` controller; the physics reviewer and the whole-branch reviewers are fresh-context `fable` agents; git steps run on `haiku` at low effort; the gui-smoke workflow runs its own agents on `sonnet`.

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
  1. In the background: `& 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/serve_web.sh 8080`.
  2. Run the Workflow tool with `scriptPath` set to this worktree's `.claude/workflows/gui-smoke.js` and the arguments `{"frame_check": true}`. The session's `gui-smoke` skill runs the main checkout's script, which reads the old layout: do not use it for this run.
  3. Expected: `passed: true`; in `magcoupling`: `starting_design`, `look`, `first_view`, `check_entry`, `lesson_link_opened`, `tooltip_plain`, `torque_card`, `address_follows`, `drag_frames_within_60fps` all `true`, `console_errors: []`. Keep the result's JSON in `.superpowers/sdd/2026-11-22-magcoupler-overhaul-b2/smoke-local.json` and the run's first screenshot beside it.
  Before 1, check that port 8080 is free (`netstat -ano | findstr :8080` prints nothing): another branch's release uses it on this machine.
  4. Stop the server: `TaskStop` on the background task that runs it, or `Stop-Process -Id <pid>` for the PID `netstat -ano | findstr :8080` shows. Check again that the port is free.

  A failure of `first_view`, `check_entry` or `torque_card` is a finding about the build until shown otherwise: open the page by hand and compare it with `spike/after-first-view-1280.png` and `spike/after-hot-low-tile-selected-1280.png`. A check that fails because the agent could not read a label is reported, not waved through.

- [ ] **Step 4: The physics and text review** (fresh context, `fable`; the design's 8.3). It reads the orientation (`explain::ORIENTATION`, `STEPS`) against the notes (two rings pass torque across a gap without touching and slip past the pull-out torque; nothing about a sealed wall, protecting the gearbox or containment) and the Hot-low tile's "needs" line and its quoting rule against BL-068's Problem, and every new string against the plain-words guard. Its findings join Step 5's fix wave. Record it in `docs/analyses/2026-11-22-magcoupler-slice-b2-review.md` (committed in Step 7).

- [ ] **Step 5: The independent whole-branch review** (fresh context, `fable`, three lenses in parallel, each with a skeptic on every finding that is not a suggestion). Give each `git diff main...HEAD`, the design, this plan, the spike's pictures and Step 3's screenshot, and none of the builders' notes.
  1. **Conformance:** the design's 4.3 (the Required row), 4.4, 4.5 and 5 for checks and tiles, tests 10 to 15 of 8.2 where they fall here, item by item against the diff; the decisions D-1 to D-19, each what the code does and none hiding a choice the owner should see; the nothing-lost table, row by row against the code.
  2. **Regressions:** no engine number moves; nothing visible today is unreachable; no existing assertion about behaviour that stays was weakened (the list "Existing tests this plan edits"); the anti-goals of 2.3 (the fixed-3D line, the exploration lines, the end-effect banner and the assumptions banner are drawn when they apply; no hidden check).
  3. **What a person sees:** the pictures and a fresh render at 1024 x 700, 1280 x 800 and 1920 x 1080: values stand out from labels; crimson only on the brand mark and the active tab; a failing design reads at a glance (what fails, what to do next); nothing cut or overlapping.

  One fix wave on `fable` for what Steps 4 and 5 find, each fix with its red test first; the reviewers re-read the fixes. Record the review in `.superpowers/sdd/2026-11-22-magcoupler-overhaul-b2/final-review.json` (gitignored).

- [ ] **Step 6: If Step 4 or 5 changed a file, run Steps 2 and 3 again.** The merge needs a gate and a smoke run of the tree that is merged.

- [ ] **Step 7: Close the item and merge.** In `docs/ai/backlog.yaml` (the day's date; say what the reviews found):

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2-the-instrument (Tasks 1 to 6 of the plan); the full gate, the smoke run, the physics review and the whole-branch review are its Task 7. Filed
```

with:

```yaml
  status: fixed
  notes: "FIXED 2026-11-2X: built on branch magcoupling/overhaul-b2-the-instrument; GATE PASS with --full, the gui-smoke run passed with first_view, check_entry and torque_card true, the physics review and the independent whole-branch review approved it. Filed
```

  (Write the day's date in place of `2026-11-2X`.) Commit it with the review record (`docs(magcoupler): BL-081 fixed: the gate, the smoke run and the reviews passed`), then merge the branch into `main` with a merge commit whose message lists the tasks' commit subjects. Remove the worktree with `git worktree remove`, never `git worktree prune`.

- [ ] **Step 8: Push, then report** (the owner's ruling of 2026-11-22: each stage goes live once the gate, the smoke run and the reviews pass, with a report afterwards; no stop). Push `main` (it deploys the site). When the `deploy-web.yml` run is green, run the smoke workflow from `main`'s script with `{"url": "https://colesorkness.com"}` and report the result; if it fails, say so plainly: the site is live with the change. The report to the owner holds: what changed on screen, in three sentences; three pictures (the first view, the failing example design, a check's entry); the decisions D-1 to D-19, one line each; the values one click or one tab further away than today (D-2: a check's numbers in its entry; D-17: three rows' notes through the Results tab), next to owner question 49; part 2's list; what was not verified (Safari and Firefox).

## What this plan covers of the design

| Design item | Where |
|---|---|
| 4.1 Three columns; the frame of the panel kept | Task 5 |
| 4.2 The top bar | Part 2 |
| 4.3 Required torque joins the size controls as their first row | Task 5 (D-5) |
| 4.3 The folds, the hint lines into the Explain panel | Part 2 (D-3) |
| 4.4 The opening tab: tiles, exploration lines, the drawing at the full remaining height; callouts on the Geometry tab | Task 5; the callout list stays under the drawing until part 2 (D-19) |
| 4.4 The tiles: pull-out, hot-low with "needs" and its gap, cold-high, mass; number large and leading, label small and muted; a readout's hover; a click selects; failing hot-low by shape and red on its "needs" line; BL-068's quoting; the no-prediction texts | Task 2 |
| 4.4 Plots tab with a picker; the welcome strip removed | Part 2; Task 5 (the strip) |
| 4.5 The check list: the status line word for word, one row per counted guide in order, the state's shape, the label; a readout's hover; a click selects; material warnings; the fixed-3D line closes the list | Task 1 (D-2 for the quantity), Task 5 |
| 4.5 The list up to 45 % of the column, the Explain panel the rest, each scrolling | Task 5 (D-11) |
| 5.1 One selection, view state; Esc and About return the orientation | Tasks 3, 5 (D-3: checks and tiles in this part) |
| 5.2 A click on a tile or a check row selects; a hover selects nothing; a fix link focuses its input | Tasks 1 to 3, 5 |
| 5.3 One form, headings in order, a heading left out when empty; no new explanatory text | Task 3 (D-4; D-18 for a check's two departures; the tile's "Checks it touches"); "Spreadsheet cell" in part 2 |
| 5.4 The orientation, at most 60 words, with the Start-here link | Task 3; Task 7's physics review |
| 5.5 The Equation link opens the Equation panel; a selection elsewhere leaves the open equation | Task 3, Task 5 (`an_entry_s_equation_link_opens_the_equation_panel_and_the_selection_stays`) |
| 8.2 Test 8 (word budget) | Task 5 as a ceiling of 450 (D-9); 250 in part 2 |
| 8.2 Test 9 (nothing lost) | The nothing-lost table and the tests it names; the full table-driven test with part 2, when the top bar and the Size column move |
| 8.2 Test 10 (selection) | Tasks 1 to 3, 5 (checks and tiles) |
| 8.2 Test 11 (entries) | Task 3, Task 5 (`guidance::tests` through `check_entry_texts`) |
| 8.2 Test 12 (state without colour) | Task 1 |
| 8.2 Test 13 (still honest) | Tasks 1, 5 (the fixed-3D line, the banner, the exploration lines; the paused clamp's row) |
| 8.2 Test 14 (idle frames) | Task 5 |
| 8.2 Test 15 (layout) | Task 2 (the wrap), Task 5 (three window sizes in the app's font) |
| 8.3 Text rules | Tasks 2, 3 (plain words, glyphs), Task 7 Step 4 |
| 9.1 Docs with the slice | Task 6 |
| 9.2 The slip-heating readout's unit | Part 2 |
| 9.4 Release | Task 7 (no stop: the owner's ruling of 2026-11-22) |
| Register row A-18 on the first view | Task 5 |
| Stage 1's review suggestions | Decision D-10 |
