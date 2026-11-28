# Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-c (the Equation panel): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ] `) syntax for tracking.

**Goal:** The Equation panel's last three items, and the keyboard's reach into the scroll areas BL-100 left out; with them slice B is built. Walking Start here in the smallest window, no term name, tag or step line runs past the panel's edge, and the panel no longer grows over the right column, nor on any other equation (BL-086): a term's name and what it is share the term grid's last column and wrap in the panel's width, a term's value wraps past a third of that width, and the walk bar's step line wraps. The equation itself, one line of typeset maths, keeps scrolling sideways where it is wider than the panel. The teaching-note drawing leaves the Equation panel's module for its own, `gui/notes_view.rs`, unchanged (BL-088). The panel's empty text, which no state of the app showed, goes (BL-089). The keyboard's focus is scrolled into view in the Equation panel and its formula, in the Clamp tab's screw-size table and in the check list, and a Results table line takes the focus only while the table shows it whole (BL-101, all four parts). No on-screen string is added or changed. Nothing that is on screen today is lost (the nothing-lost table).

**Architecture:**
- **A pure move.** `gui/notes_view.rs` holds what draws a note (`note_ui`, `change_ui`, `explains_ui`, `NoteView`, `NoteLink`, `held_reason`, `live_value`, the seven note strings) and its tests, moved verbatim; `explorer.rs` keeps the panel, its lesson line (`lesson_ui`) and its next best read (`note_fallback_ui`), and the sign-off sheet, whose text names it. Callers import from the new module; nothing is re-exported.
- **Wrapping in the panel.** The term grid takes four columns, `Grid::num_columns(4)`: egui gives the last the width the panel has left, and the term's name and tag sit in one wrapping line there, so a row is never wider than the panel and the terms' sideways scroll area goes. A term's value takes at most a third of the list's width (`TERM_VALUE_SHARE`) and wraps past it, laid out as a galley at that width: egui lays a grid cell before the last out in the width its column had in the frame before, so a wrapping label there would never widen, and a long text value on one line ("never: steady state stays below the limit") left the last column about 20 points. The walk bar's step line is a wrapping label; Previous, Next and Stop follow it. The formula keeps its own sideways scroll area.
- **The focus in nested areas.** `gui::follow_focus` returns the rect it scrolled to and takes as its own only a widget that also lies in its area's content (`ui.min_rect()`): egui keeps the parent's clip on the axis an area does not scroll, so a sideways area inside an up-and-down one is clipped up and down by the outer view, and without the check it took the outer area's widgets as its own. `gui::sideways_scroll_area`, used by the formula and the screw-size table, calls `follow_focus` last in its content and, since egui hands both axes' scroll targets to the innermost area, asks the outer area's content again for the rect once it has ended. The Equation panel's own area calls `follow_focus`.
- **The Results table and the check list.** A table line's click sensor takes the keyboard's focus only while the table's view shows the line whole up and down (`results_table::line_sense`, through `readouts::sensor`'s new sense argument): Tab passes the lines on screen and leaves the table, as owner question 67's default has it, and never stops on a line half seen. The check list's area calls `follow_focus`, which keeps the focus it saw last under its content's unique id: egui gives a scroll area's content its parent's id, which the check list and the Explain panel under it share.
- **Unchanged:** the engine, the exports, every on-screen word but the empty text that no one saw, the formula and its sideways scroll, the left and right columns' widths (320 and 300 points), part 2b-iii-b's look, the linkage app, the page shell, the smoke workflow.

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32; the headless egui harness (the panel tests' `Harness` and its walk `follow`, the app tests' `frame_on`; `gui/test_support.rs`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. This part is its test 15 of section 8.2 (the layout at 1024 x 700, 1280 x 800 and 1920 x 1080), 5.5 (Start here), 5.2 with BL-093 (the keyboard's focus arriving selects nothing), the text rules of section 8.3 and the docs and release rules of section 9. Executors read the spec's sections 5.2, 5.5, 8.2, 8.3 and 9 beside this plan. Part 2b-iii-a's plan (`docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md`, "Why part 2b-iii is cut, and what follows", decision D-86) sets the scope; the items are BL-086, BL-088, BL-089 and BL-101 in `docs/ai/backlog.yaml`, each with its evidence and acceptance. Backlog item: **BL-112** (`docs/ai/backlog.yaml`, filed with this plan).

**How this plan was made.** The part was built once as a prototype on a copy of `main` at `c569d03`, one commit per step (Task 1's move; Tasks 2 to 5, each its red and its green), then BL-112 filed and the docs swept; each state was made by an edit script (Task 1's by exact spans, so the moved text is the old text). Every block below was generated from those commits' diffs. Then the plan was replayed with the apply tool, in this plan's order, on a fresh copy of `main` with BL-112 filed, in a target directory of its own, built fresh by a first run that took every step to the docs; the decisions were then renumbered (a test comment and the docs name them) and the replay ran again in full, in the same directory. The plan review's fix wave then changed the prototype's Tasks 3, 5 and 6 and BL-112's acceptance (findings RC-1, RC-2, UXT-1 and UXT-2 taken; RC-3 answered under "Review Focus", item 14), the blocks were generated again from its commits, and the whole plan was replayed once more on a fresh copy of `main` with BL-112 filed, in a new target directory of its own; the counts below are that run's. Each block applied once; each red step failed as its "Expected" says; the result equals the prototype file for file (line endings aside) and passes `cargo test` with and without the app feature (1568 tests with it, 656 without; 1157 lib tests with it against 1151 on the base), both native clippy runs after each task, both wasm32 clippy runs, `cargo fmt --check` and the gui-smoke verdict test (`all 85 checks hold`); Task 7 Step 7's five blocks applied to the docs swept and set the five items fixed. Each count after a task is the replay's. Each new test was also run with its fix's piece taken out ("What the prototype showed" item 8). Not run for this plan: the full gate (`gate.sh --full`: gate 12, the Python oracle, and gates 1 to 3 on `linkage-sim-rs`, untouched), the release WASM build, the browser and the pictures. Task 7 runs them.

**Where this part applies.** On `main` at `c569d03` with this plan's commits (the plan and BL-112, then the plan review's fix wave), and nothing else that touches `magcoupling-rs/` or Task 6's nine files. If a block does not match, stop and report: the controller regenerates the blocks (the planner's tools are listed under "Before Task 1"), it does not adapt a block in place.

## Why part 2b-iii-c is one plan, and what follows

Part 2b-iii-a's plan cut part 2b-iii in three (decision D-86, register row A-47) and left this part the Equation panel's items, BL-086, BL-088 and BL-089, and BL-101's Equation panel part "if that plan holds it". It holds all four parts of BL-101: they are one rule (`follow_focus`) reaching four scroll areas, and the Equation panel's part cannot be built without the change the screw-size table needs (the same nesting). The owner wants the site fully released by 2026-11-29, which favours one release. The plan's size is Task 1: a pure move of about 1,060 lines, so its blocks carry the moved text twice (out of `explorer.rs`, into `notes_view.rs`), about 2,600 plan lines that a reviewer checks by diffing the two, not by reading; the other code tasks run from about 190 to 550 plan lines each, and one reviewer holds each with its list under "Files". So the part is not cut again (decision D-106).

**What follows:** nothing of slice B. Slices C to F have their own plans. Not touched here: BL-083, BL-084, BL-087 and BL-067 (physics: they wait for the fable decider, back 2026-12-01, a later part), BL-095, BL-102 (egui's Tab stops that show no focus; this part adds none), BL-103, BL-105, BL-106, BL-110 and BL-111.

## What the prototype showed

In the app's own theme and font (`MagcouplingApp::new`, which applies `theme::apply` and `fonts::install`), at a pixel ratio of 1, unless the panel tests' harness (egui's built-in font) is named.

1. **The walk on `main`.** At 1024 x 700, over the 14 Start here steps: a term's name or tag ran past the Equation panel's edge on steps 1, 2, 5, 8, 9 and 11 ("Total magnetic shear stress at pull-out" by 46 points, "assumption: click to find its row" by 159 on step 5, "Thermal conductance to the housing and shafts" by 89 on step 11); the walk bar's step line on steps 2, 3, 4, 6, 7, 12 and 13 (by 185 points on step 13, the ferrite step), and that line, laid out on one row, made the panel wider than the centre region: its right edge at x 732.6 on step 2 and x 901.7 on step 13, the region ending at 716. At 1280 x 800 a tag ran past on steps 8 and 11 (by up to 84 points); at 1920 x 1080 nothing did. The formula ran past the edge on steps 6, 10, 12, 13 and 14 at 1024 x 700 and on 6, 13 and 14 at 1280 x 800, inside its own sideways scroll area; the widest, the ferrite step's cold check, is about 1,040 points at 22 points, so set smaller to fit the 362-point view it would need about 7.6 points.
2. **The walk after Task 3.** At the three sizes, on every step, the panel stays between the two columns and no text in it is cut but the formula's. On σ in a 1000-point window (egui's font) the assumption's tag takes two rows inside the panel. The same over every equation the registry holds, opened in the app's font in a window 1024 points wide on the starting design and on the example design with the spreadsheet details on (Task 3's test), and at 1280 and 1920 points wide on the starting design (a run of the test at those sizes, not kept). With a term's value laid out on one line, as this plan first had it, `temperature.thermal.rotations_to_limit_high` ("never: steady state stays below the limit") pushed the panel's right edge to x 728.1, past the region's 716; the plan review (UXT-1) measured that row and the Verdict row (`temperature.summary.verdict`) leaving the name and tag column 19 to 27 points, a word to a row.
3. **The panel's keyboard walk on `main`** (the harness at 1024 x 700). Its own area did not follow the focus: on the hot-low torque the stop under the panel's foot (y 687.8 to 701.8, the view ending at 684) stayed cut; the term rows' links, in their sideways area, stayed cut sideways. On the hot minimum check the formula is 493.6 points wide in a view of 362: it can never be whole, and egui's least scroll shows it from its start. With the panel dragged to its least height (120 points), Shift+Tab from the foot reaches the formula above the view: only the second ask brings it down.
4. **Three egui facts the fix rests on** (egui 0.32.3, `containers/scroll_area.rs` and `grid.rs`). A scroll area's content takes the parent's clip on the axis it does not scroll, so a sideways area's content is clipped up and down by the outer view, not by its own rows: `follow_focus` in the formula's area took the lesson link under it as its own, read its rect from the frame before, and the outer area scrolled twice (69.3 points to show a link cut by 31.7). The content check stops it. And the content's `Ui` gets its parent's id (`new_child` with no salt), so sibling areas share it: with the check list's call added, its `follow_focus` and the Explain panel's shared one slot, and the Explain panel's walk failed (a link 111 points above its view left there). Keyed by the content's unique id, each area has its own. And a grid cell before the last is laid out in the width its column had in the frame before (`GridLayout::available_rect`), so a wrapping label there never widens; a value given its width in a child ui (`ui.scope`, `set_max_width`) made each term row about 146 points tall at 1280 x 800, and two panel tests lost the Used by line under the view. Laid out as a galley at its width (`WidgetText::into_galley`), the value takes its own size.
5. **The screw-size table.** At 1024 wide its cells past x 708 are cut sideways, and the table's own drag surface, one of egui's Tab stops (BL-102), is as tall as the table, 711 points, more than the 610-point view at 1024 x 700: it can never be whole there, so Task 4's walk runs in a window 1000 points tall.
6. **The Results table on `main`.** At 1024 x 700 the line drawn past the view's foot, a group heading (y 699.9 to 718.2, the view ending at 684), took the focus on the second Shift+Tab, unseen; BL-101 found the row half seen above it (y 677.8 to 696.1) taking it too. Wheeled 300 points down at 1280 x 800, a data row the view's top cuts (y 103.0 to 121.4, the view starting at 119.8) takes it (seen with the rows' held-back focus alone taken out; the walks as the table opens meet no data row cut). After Task 5 the walk passes only whole lines (more than ten stops each way, as the table opens and wheeled) and no text of the table moves.
7. **The move.** `explorer.rs` goes from 3,110 lines to 2,065 after Task 1 (2,100 after the part); `notes_view.rs` is 1,061. The lib tests with the app feature stay at 1151: 14 tests and one ignored review tool run as `gui::notes_view::` that ran as `gui::explorer::`.
8. **Each test fails on what it names.** Taken out one at a time on the last state: `sideways_scroll_area`'s second ask (the Equation panel's walk and the screw-size table's fail), `follow_focus`'s content check (the Equation panel's walk fails: the outer area scrolled twice), the Equation panel's own call (its walk fails), the table lines' held-back focus (the Results table's walk fails), and so does each of `line_sense`'s two callers alone: the group headings' (the walk fails as the table opens: press 2, the heading drawn past the foot) and the data rows' (it fails only wheeled: press 32 at 1280 x 800, a row cut at the top); the check list's call (its walk fails); a term value's wrap, its galley laid out on one line (the every-equation test fails on `temperature.thermal.rotations_to_limit_high`, the panel's right edge at x 728.1). The every-equation test passes on the old layout: there the term list scrolled sideways and egui painted nothing past its view, which no test of drawn text sees (σ's test stops at its `expect` for that reason).
9. **Line endings and rustfmt.** `cargo fmt` may rewrite a file it changes with LF line endings on this machine; git's `autocrlf` takes them back at the commit, and the apply tool reads each file's endings as it finds them. Every file this part touches is CRLF.

No pictures are kept with this plan: Task 7's text review renders the walk at the three window sizes, the formula past the edge and the keyboard's walks.

## Global Constraints

- **No engine number moves** (the design's 2.3). No file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` changes.
- **The linkage app, the hub and the page shell are not touched** (O-1). No file under `linkage-sim-rs/` changes, nor `.claude/workflows/gui-smoke.js`.
- **No on-screen string is added or changed.** One goes: `EMPTY_TEXT`, which no state of the app drew. The strings that move with the note drawing move word for word. The formula, the term names, the tags and the step line keep their words; only where they wrap changes.
- **The columns keep their widths:** the left 320 points, the right 300, in the app's font (the BL-099 and BL-098 tests, unchanged and passing). Part 2b-iii-b's look stays (no file under `magcoupling-rs/src/app/` changes).
- **A hover selects nothing** (C-2), **and neither does the keyboard's focus arriving** (BL-093): every walk in Tasks 4 and 5 asserts the selection and the design unchanged. **No Tab stop is added** (BL-102): Task 5 only holds the focus back from table lines half seen.
- **View state and the exports do not change** (5.1, 12.1).
- **The blocks.** Each Replace block's old text occurs exactly once in its file at the moment its step applies it. Apply a step's blocks in the order given, top to bottom. Keep each file's line endings. If an old text is not found, or is found twice, stop and report: do not adapt the block. Line numbers are not given because earlier steps move them. `C:\Users\Cole\source\repos\linkage_simulation\.superpowers\sdd\2026-11-22-magcoupler-overhaul-b2\apply_plan.py` (gitignored, in the main checkout; stage 1's tool, unchanged) applies one step's blocks this way: `python <it> <this plan> <task> <step>`, from the worktree's root. The replay used it.
- **TDD.** Red, then green. Every red step here fails at run time with the message its "Expected" names. Task 1 is a pure move: no test can fail first, and its first step records the counts its last step checks.
- **Before each commit**, from the repository root:
  ```bash
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
  ```
  `cargo fmt` changes no text of a task's files (the blocks are formatted); it may rewrite a file it reads with other line endings, which git's `autocrlf` undoes at the commit. After it, `git status --short` shows only the files the task names. If it shows another, stop and report.
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-27-magcoupler-overhaul-b2p2biiic/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Shell.** On this machine `bash` typed in PowerShell is WSL, which fails on the checkout's line endings. Shell commands and scripts run through Git Bash: `& 'C:\Program Files\Git\bin\bash.exe' <script>`, or Git Bash itself. Python scripts are written with the Write tool, not a here-document. Do not edit a source file with `sed -i` from Git Bash: it rewrites the file's line endings.
- **Never run `cargo fmt` inside `linkage-sim-rs`.** `magcoupling-rs` is fmt-clean; keep it so.
- **Docs ship with the code** (repository rule): Task 6 sweeps the README and `docs/ai/`, the register, the owner-question list and the design's header.

## Model per task

CLAUDE.md section 5: `fable` is the controller and the decider. BL-112 is `risk: mechanical` (decision D-115): the part shows no new number, makes no physical claim and adds no on-screen string. So the tasks that transcribe the plan's exact blocks and run its exact commands run on `sonnet`, and every review and the release run on `fable`; while `fable` is out (back 2026-12-01) the decider's steps inherit the session model, and the report says which model ran them. A `sonnet` task that ends blocked, leaves the gate red or is rejected in review is redone on `fable` (escalate, never downgrade).

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | The note drawing in its own module (BL-088, a pure move) | `sonnet` |
| 2 | The Equation panel's empty text goes (BL-089) | `sonnet` |
| 3 | Every text of the Equation panel but its formula wraps inside it (BL-086) | `sonnet` |
| 4 | The keyboard's focus followed into the nested sideways areas (BL-101 parts 1 and 2) | `sonnet` |
| 5 | The Results table's Tab stops are its whole lines; the check list follows the focus (BL-101 parts 3 and 4) | `sonnet` |
| 6 | Docs sweep | `sonnet` |
| 7 | Release: the full gate, gui-smoke, the text review with its pictures, the independent whole-branch review, one fix wave, the merge and the push | `fable` controller (the session model while `fable` is out); fresh-context `fable` reviewers (likewise); `haiku` for git; the smoke workflow's own agents on `sonnet` |

Each task's diff is reviewed by a fresh-context reviewer before the next task starts (`sonnet` for these mechanical diffs, CLAUDE.md section 5); it checks the diff against the task's blocks and nothing else, and for Task 1 that what leaves `explorer.rs` arrives in `notes_view.rs` and `test_support.rs` unchanged but for the three changes the task names. A task that ends blocked or leaves a test red stops the build and goes back to the controller.

## Review Focus

Fourteen inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named, or says where it is checked.

1. **The formula wider than the panel.** It cannot wrap and cannot be whole; the focus shows it from its start. Task 4: the walk on the hot minimum check (the helper's whole check lets a widget wider than its region show from its start); Task 3: the walk's text test lets the formula's own area alone run past the edge (decision D-109).
2. **The formula out of view up or down.** Only the second ask scrolls the outer area to it. Task 4: the walk with the panel at its least height (fails with the second ask taken out).
3. **A widget of the outer area under a nested one.** The nested area must not take it. Task 4: the same walk (fails with the content check taken out: the outer area scrolled twice).
4. **Two sibling scroll areas.** The check list and the Explain panel share a content id. Task 5: `the_explain_panel_follows_the_keyboard_s_focus`, unchanged, fails with the slot under `ui.id()`.
5. **A table line half seen at the foot or the top, and the one drawn past the foot.** Task 5: the Results table's walk (no stop out of view on arrival), as the table opens (a group heading drawn past the foot) and wheeled 300 points down (at 1280 x 800 a data row cut at the top); it fails with either of `line_sense`'s two callers given the plain click back.
6. **The table moving under Tab.** It must not scroll (owner question 67's default). Task 5: no text of the centre region moves over the two walks.
7. **Every Start here step at every size.** Task 3: the 14 steps at 1024 x 700, 1280 x 800 and 1920 x 1080.
8. **The panel pushed wider by a long line.** Task 3: the panel's edges inside the centre region on every step.
9. **A narrow region with a long tag.** Task 3: σ at 1000 x 800, the tag on two rows and whole.
10. **A long text value in the term list.** "never: steady state stays below the limit" or "n/a (coercivity rises as the magnet cools)", laid out on one line, left a term's name and tag about 20 points. Task 3: every equation the registry holds, in the app's font at 1024 points wide, on the starting design and on the example design with the details on (fails with the value laid out on one line).
11. **The screw-size table's cells past its right edge and under the foot.** Task 4: its walk (the cells cut sideways are stops; the window holds the table's drag surface).
12. **Focus arriving selects or edits.** Tasks 4 and 5: each walk asserts the selection and the design unchanged.
13. **The move loses a test.** Task 1: the lib count unchanged, the moved tests counted under the new module and out of the old.
14. **A wheel after the focus arrived, in the four areas this part makes follow it.** BL-101's acceptance asks that a wheel after the arrival is not fought. No walk of the Equation panel, its formula, the screw-size table or the check list wheels after an arrival, and none is added: `follow_focus` scrolls only in the frame the focus changes, in every area alike, and `sideways_scroll_area` asks the outer area again only with the rect `follow_focus` returned that frame. The rule is held by `the_left_column_follows_the_keyboard_s_focus_and_fights_no_wheel` and `a_wheeled_column_holds_still_while_the_focus_is_on_no_widget_of_its_own`, unchanged and passing after Task 5 keys the slot by the content's unique id; decision D-114 records when an area may follow its widget once more.

## Decisions this plan takes

Each is a reading of the design or of a backlog item where it was open, numbered after part 2b-iii-b's D-95 to D-105 (one stage, one list). The owner may overrule any of them; Task 7 lists them in its report. The ones that need a row in the register have one (Task 6).

| Id | Decision | Why |
|---|---|---|
| D-106 | Part 2b-iii-c is one plan: BL-088, BL-089, BL-086 and all four parts of BL-101. | One panel and one rule; the Equation panel's part of BL-101 needs the screw-size table's change; one release by 2026-11-29. See "Why part 2b-iii-c is one plan". |
| D-107 | BL-088 moves the note drawing, its seven strings, its tests and the review tool `notes_for_review` to `gui/notes_view.rs` verbatim; `design` and `nothing_held`, called by both test modules, move to `test_support.rs`; `note_text_on` becomes `pub(crate)` for the lesson line's test; nothing is re-exported. The sign-off sheet stays in `explorer.rs`. | The backlog's own name for the module, and a pure move. The sheet the sign-off test writes says it is generated by `explorer.rs`; moving it would make the committed owner record name the wrong file. A copied helper would be the repetition the codebase avoids. |
| D-108 | BL-089: `EMPTY_TEXT` goes, with its test's lines and its row in the strings sheet; a panel open with no trail draws its header row only. No way to open the panel empty comes back. | No state of the app showed it; the conservative answer adds no on-screen text (the backlog's own rule). Register row A-56. |
| D-109 | The formula keeps its own sideways scroll area at its 22 points; it is the one text of the panel that may run past its edge. Register row A-55; owner question 76. | An equation does not wrap; set smaller to fit, the widest walk step's would need about 7.6 points, under the tooltips' 8-point floor. BL-086 names the term rows, the tags and the step line, not the formula. The cost: on five walk steps at 1024 x 700 the equation's end is out of view until scrolled. |
| D-110 | The term grid has four columns; a term's name and tag share the last, one wrapping line, and the terms' sideways scroll area goes; a term's value takes at most a third of the list's width (`TERM_VALUE_SHARE`) and wraps past it. | egui gives a grid's last column the width left, and a wrapping line there keeps every row inside the panel; a long text value on one line left that column about 20 points (the plan review's UXT-1), and a third keeps a number on one line in the smallest window. The cost: the tags no longer line up in a column of their own, and a long text value takes two or three rows. |
| D-111 | The walk bar's step line wraps; Previous, Next and Stop follow it, on its last row or the next. | The step line is the bar's one long text, and the buttons stay whole. |
| D-112 | `follow_focus` returns the rect it scrolled to and checks the widget lies in its area's content; `gui::sideways_scroll_area` serves the formula and the screw-size table; the Equation panel's own area calls `follow_focus`. | BL-101's own way in ("follow_focus returns the rect it scrolled to, and the outer area's content scrolls to it again"); one helper for the two nested areas, so the rule is reused, not copied; the content check is what egui's clip on the unscrolled axis requires. |
| D-113 | The Results table keeps owner question 67's default (Tab passes the lines on screen; the table does not scroll for the focus), and a line takes the focus only while its view shows it whole. Register row A-44 (its cost updated); question 67's text updated. | BL-101's acceptance: "the default: as today, and then no focused row is left cut at the view's foot". Following the focus would make Tab walk several hundred rows (the question's option b). The cost: a focused line a wheel then cuts loses the focus. |
| D-114 | The check list follows the focus (BL-101's fourth part), and `follow_focus` keeps its last focus under the content's unique id. | BL-101's acceptance names it; its call made the shared id a real collision. The unique id holds while the widgets before the area in its parent stay as they are; when they change, an area may follow its focused widget once more. |
| D-115 | BL-112 is `risk: mechanical`; its build tasks run on `sonnet`, its reviews and release on `fable` (the session model while `fable` is out). | No new number, physical claim or on-screen string. |

## Nothing lost

Every text, look and action this part changes, and where it is after. Anything not listed (every other text, every position outside the Equation panel, every view, the columns, the look, the linkage app) does not change. The "Test" column names the test that holds the new state.

| Today | After this part | Test |
|---|---|---|
| A term row: swatch, symbol, value, name, tag in five columns, the list scrolling sideways in a narrow region | The same five, the name and the tag wrapping together in the last column, a long value wrapping in its own; no sideways scroll for the list | `every_text_of_the_equation_panel_is_whole_on_each_start_here_step_in_the_app_s_own_font`; `every_text_of_the_equation_panel_is_whole_on_every_equation_in_the_app_s_own_font`; `the_term_list_wraps_inside_a_narrow_centre_region`; the panel's term tests (`a_sum_s_harmonic_set_has_its_swatch_in_the_term_list`, `the_term_list_leaves_out_the_harmonics_past_the_set`) unchanged |
| A term's name clickable (drills, or frames an input's row), its internal name on hover with the details | The same, on its wrapped label | the panel's drill and hover tests, unchanged |
| The walk bar: "Start here N of 14: title" on one row, then Previous, Next, Stop | The same words and buttons, the line wrapping | `the_start_here_bar_names_every_step_it_shows`, unchanged; the app's walk test |
| The formula large, clickable term by term, scrolling sideways | The same; Tab shows it from its start | the panel's formula tests, unchanged; Task 4's walk |
| The empty panel's text, drawn by no state the app reaches | Gone; a panel open with no trail draws its header row | `a_panel_open_with_no_trail_draws_its_header_row_and_nothing_else` |
| The note drawing in `explorer.rs` | The same code and tests in `notes_view.rs` | the moved tests, unchanged, under `gui::notes_view::` |
| Tab in the Equation panel, the screw-size table and the check list: stops left cut | Each stop scrolled into view | Tasks 4 and 5's walks |
| Tab in the Results table: the lines on screen, a line half seen at the foot among them | The lines the view shows whole; the table still does not scroll for Tab | `the_results_table_s_tab_stops_are_the_lines_it_shows_whole` |
| A click on any table line, half seen or whole | The same | the table's click tests, unchanged |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/notes_view.rs` | New: the note drawing and its tests, moved | 1 |
| `magcoupling-rs/src/gui/explorer.rs` | The note drawing out, its imports, its module comment (1); one test (2, red); `EMPTY_TEXT`, its draw and its test's lines (2, green); the walk bar's line and the term grid (3); the formula through the helper and the area's `follow_focus` (4) | 1, 2, 3, 4 |
| `magcoupling-rs/src/gui/mod.rs` | `pub mod notes_view` (1); `follow_focus`'s return and content check, `sideways_scroll_area` (4); `follow_focus`'s slot (5) | 1, 4, 5 |
| `magcoupling-rs/src/gui/test_support.rs` | `design`, `nothing_held` (1); `cut_texts_in` (3) | 1, 3 |
| `magcoupling-rs/src/gui/explain.rs`, `guidance.rs`, `app.rs` | Imports of the moved items (1); the strings sheet's row (2, `guidance.rs`); the app's walk test (3, `app.rs`) | 1, 2, 3 |
| `magcoupling-rs/src/gui/panel.rs` | Imports (1); the term list test and the every-equation test (3); the walk helper and two tests (4); two tests (5, red); the check list's `follow_focus` (5, green) | 1, 3, 4, 5 |
| `magcoupling-rs/src/gui/clamp_drawing.rs` | The screw-size table through the helper | 4 |
| `magcoupling-rs/src/gui/readouts.rs`, `results_table.rs` | `sensor`'s sense; `line_sense` for the table's headings and rows | 5 |
| `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`, the register, the owner questions, the design's header | The docs sweep | 6 |
| `docs/ai/backlog.yaml` | BL-112 filed with this plan; BL-086, BL-088, BL-089, BL-101 and BL-112 in progress (6); all five fixed (7) | 6, 7 |

**Existing tests this plan edits, and how.** No assertion about behaviour that stays is weakened.
- **The moved tests** (Task 1): unchanged, in `notes_view.rs`; `design()` and `nothing_held` are called from `test_support`.
- **`the_lesson_line_names_the_open_equation_s_note_and_asks_for_it`** (`explorer.rs`, Task 1): unchanged; it reads `note_text_on`, `NoteView` and `nothing_held` from their new homes.
- **`the_empty_text_and_explain_s_own_lines_name_the_buttons_in_plain_words`** (Task 2) becomes `explain_s_own_lines_name_the_buttons_in_plain_words`: `EMPTY_TEXT`'s two asserts and its row go with it; the fallback lines' plain-words and glyph checks stay.
- **`every_start_here_step_s_lesson_line_is_drawn_whole_in_the_app_s_own_font`** (`app.rs`, Task 3) becomes `every_text_of_the_equation_panel_is_whole_on_each_start_here_step_in_the_app_s_own_font`: the same walk at the same sizes, its filter widened from the lesson line's two texts to every text clipped inside the panel but the formula's area, and the panel's edges checked against the centre region.
- **`the_term_list_scrolls_sideways_in_a_narrow_centre_region`** (`panel.rs`, Task 3) becomes `the_term_list_wraps_inside_a_narrow_centre_region`: the same window and equation; where it wheeled the list sideways and found the tag whole, it finds the tag whole with no wheel, and its guard (the test would prove nothing) now asks that the tag wraps.
- **The walk helper `follow` and `in_region`** (`panel.rs`, Task 4): a widget wider than the region passes when whole up and down and shown from its start (no widget of the three existing walks is wider than its region); a widget its view cuts sideways belongs to the region where its view ends (for a widget cut up or down, the same region as before). `the_left_column_follows_the_keyboard_s_focus_and_fights_no_wheel`, `the_explain_panel_follows_the_keyboard_s_focus`, `a_wheeled_column_holds_still_while_the_focus_is_on_no_widget_of_its_own` and `the_centre_s_scrolling_views_follow_the_keyboard_s_focus` pass unchanged.
- **Unchanged and still passing:** the BL-099 and BL-098 width tests, `no_text_is_cut_in_any_tab_in_the_app_s_own_font`, `the_first_view_draws_at_most_two_hundred_and_fifty_words_at_1280_by_800`, the nothing-lost test, the glyph guard and the smoke prompt's guard.

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` after this plan has merged, create the branch `magcoupling/overhaul-b2p2b-iii-c` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: `test result: ok. 1151 passed; 0 failed; 6 ignored` and `all 85 checks hold`. Another count means `main` is not the base the blocks were checked on: stop and report.

- [ ] **Step 3: Check the base.** `git log --oneline c569d03..main -- magcoupling-rs docs/ai docs/analyses/2026-11-10-conservative-assumptions.md docs/analyses/2026-11-18-owner-questions.md docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` lists the two commits of this plan that touch `docs/ai/backlog.yaml` (the plan with BL-112, and the review's fix wave), the merge that brings them if there is one, and nothing else. If it lists more, stop: the controller regenerates the blocks on that `main` with the planner's tools (gitignored, in the main checkout's `.superpowers/sdd/2026-11-27-magcoupler-overhaul-b2p2biiic/`, the scripts in its `tools/` folder: `gen_blocks.py` (part 2b-iii-b's, unchanged), `frags.py`, `replay.py` and `finalize.py`, which read the prototype as a clone of `seq.bundle` named `proto` in the folder above them (branch `proto`), and `untemplate.py` (with `finalize_paths.py`), which takes this plan's text back to the parts `replay.py` assembles (`plan_head.md`, `plan_tasks.md`) from the blocks `frags.py` writes, copied to `frag_old/`; `seq.bundle`, the prototype's commits on `main` at `c569d03`: `s1` Task 1, `s2r` and `s2` Task 2's red and green, `s3r`, `s3`, `s4r`, `s4`, `s5r` and `s5` Tasks 3 to 5's, `s5b` BL-112 filed, `s6` the docs; and the edit scripts that made them, run in order by `run_steps.py` from `main`: `ed_lib.py`, `fmt.py`, `e1_move.py`, `e1_refs.py`, `e2_red.py`, `e2_green.py`, `e3_red.py`, `e3_green.py`, `e4_red.py`, `e4_green.py`, `e5_red.py`, `e5_green.py`, `file_bl112.py` and `e6_docs.py`; `mutate.py` takes each fix's piece out in turn) before Task 1.

## Tasks

### Task 1: The note drawing in its own module (BL-088)

**Model:** `sonnet` (BL-112 is `risk: mechanical`; the task transcribes exact blocks).

A pure move (decision D-107). Out of `gui/explorer.rs`, into the new `gui/notes_view.rs`, verbatim: the seven note strings (`WATCH_OUT` to `EXPLAINS`), then `NoteView` to `explains_ui` in their order (`NoteView`, `NoteLink`, `held_reason`, `unlinked_reason`, `live_value`, `after_the_end_factor`, `note_text_on`, `note_ui`, `change_ui`, `explains_ui`); from the test module, the note drawing's helpers and tests from `RENDER_NOTE` to `notes_for_review`, and `a_wrapped_change_entry_stays_inside_a_narrow_panel_and_its_label_links`. Two test helpers both modules call, `design` and `nothing_held`, move to `gui/test_support.rs`. The only text that changes in what moves: `note_text_on` becomes `pub(crate)` (an Equation panel test reads it), `NoteLink`'s two doc links name the Equation panel's items by their paths, and the module comments split their one paragraph between the two modules. Every file that named a moved item through `explorer` names it through `notes_view`. The sign-off sheet (`note_now`, `signoff_sheet`, `notes_signoff_sheet` and its test) stays in `explorer.rs`: the sheet it writes names that file as its maker.

**Files:**
- Create: `magcoupling-rs/src/gui/notes_view.rs`
- Modify: `magcoupling-rs/src/gui/mod.rs`, `magcoupling-rs/src/gui/explorer.rs`, `magcoupling-rs/src/gui/test_support.rs`, `magcoupling-rs/src/gui/explain.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/guidance.rs`, `magcoupling-rs/src/app.rs`

**Interfaces:**
- Produces: `pub mod gui::notes_view` with, moved unchanged, `NoteView`, `NoteLink`, `held_reason`, `live_value`, `note_ui`, `explains_ui`, `WATCH_OUT`, `WHAT_TO_CHANGE`, `WHERE_FROM`, `HELD_BY_AUTO`, `AT_TOP`, `AT_BOTTOM`, `EXPLAINS`; `pub(crate) fn note_text_on`; `#[cfg(test)] pub(crate) fn gui::test_support::design() -> (DesignInputs, DesignResults)` and `nothing_held(&str) -> Option<String>`.
- Removes: the same items from `gui::explorer` (no re-export).

- [ ] **Step 1: The tests before the move.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::notes_view:: 2>&1 | tail -n 2
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::explorer:: 2>&1 | tail -n 2
  ```

  Expected: `test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 1157 filtered out` (no such module yet), then `test result: ok. 44 passed; 0 failed; 2 ignored; 0 measured; 1111 filtered out`. A move changes no behaviour, so no test can fail first; these counts are what Step 3 checks against.

- [ ] **Step 2: The move.**

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
pub mod look;
mod panel;
```

with:

```rust
pub mod look;
pub mod notes_view;
mod panel;
```

**`magcoupling-rs/src/gui/notes_view.rs`**

Create the file with:

```rust
//! A teaching note as the Explain panel draws it (spec Addendum A4; UX overhaul decision C-11).
//! A note ([`note_ui`]) has its diagram ([`crate::gui::diagrams`]), its "What to change"
//! entries, its "Watch out" lines and its sources under a "Where this comes from" fold that
//! starts closed. A `{result.path}` in a note's text is drawn as that result's number on the
//! design shown ([`live_value`]), or as "no prediction" where it has none. An entry's input is a
//! link to its row, or plain text with the reason where the input is not free on the design
//! shown ([`held_reason`]: Auto holds it or a part pick sets it; or the slider is at its end
//! that way); the links to the equations a note explains are [`explains_ui`]'s. The Equation
//! panel (`crate::gui::explorer`) shows a note's title alone, as its lesson link; this module
//! was part of it until BL-088.

use crate::engine::explain::note_text;
use crate::engine::explain::notes::{Change, Note};
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::picks::set_by_pick;
use crate::gui::dashboard::{end_effect_out_of_range, greyed_by_end_effect};
use crate::gui::diagrams::diagram_ui;
use crate::gui::explorer::{crumb, term_label};
use crate::gui::guidance::result_text;
use crate::gui::inputs::InputCatalogue;
use crate::gui::plain_text::label_for;
use crate::gui::readouts::registry;
use crate::gui::sizing::{SizingState, held_by_auto};
use crate::gui::typeset::glyph_safe;
use crate::gui::wrapping_line;
use crate::{DesignInputs, DesignResults};

/// The heading of a note's watch-out lines.
pub const WATCH_OUT: &str = "Watch out";

/// The heading of a note's "What to change" entries (slice-2 spec 3.1).
pub const WHAT_TO_CHANGE: &str = "What to change";

/// The fold under a note that holds its sources, closed until clicked (slice-2 spec 3.1).
pub const WHERE_FROM: &str = "Where this comes from";

/// Why a change entry's input is plain text, not a link: Auto sets it (Torque -> Magnets).
pub const HELD_BY_AUTO: &str = "held by Auto";

/// Why a Raise entry's input is plain text: it is at the top of its slider.
pub const AT_TOP: &str = "at the top of its range";

/// Why a Lower entry's input is plain text: it is at the bottom of its slider.
pub const AT_BOTTOM: &str = "at the bottom of its range";

/// The start of the links at the head of a note to the equations it explains.
pub const EXPLAINS: &str = "Explains";

/// What a note's change entries are drawn against: the design shown, its results, and why an
/// input is not free on it (`held`: Auto holds it or a part pick sets it, [`held_reason`];
/// `None` for a free input).
pub struct NoteView<'a> {
    pub inputs: &'a DesignInputs,
    pub results: &'a DesignResults,
    pub held: &'a dyn Fn(&str) -> Option<String>,
}

/// What a click in a teaching note, or in the lesson line under an equation, asks for
/// ([`Explorer::follow_note_link`](crate::gui::explorer::Explorer::follow_note_link)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoteLink {
    /// Highlight this input's row, as a leaf term does.
    Input(&'static str),
    /// Show this reviewed note in the Explain panel (an equation's lesson, `lesson_ui`, or
    /// its closest note, [`note_fallback`](crate::gui::explorer::note_fallback)).
    Note(&'static str),
}

/// Why the input at `path` is not free on the design `inputs` shown with `sizing`, for a note's
/// change entry ([`NoteView::held`]): Auto sets it in Torque -> Magnets ([`HELD_BY_AUTO`]), or
/// a part pick sets it in place of its own value ("set by the `<pick's label>` pick",
/// [`set_by_pick`]); `None` when it is free.
pub fn held_reason(sizing: &SizingState, inputs: &DesignInputs, path: &str) -> Option<String> {
    if held_by_auto(sizing, path) {
        return Some(HELD_BY_AUTO.to_owned());
    }
    set_by_pick(inputs, path).map(|pick| format!("set by the {} pick", label_for(pick)))
}

/// Why a change entry draws its input as plain text on `view`'s design: the input is not free
/// there (`view.held`), or one more slider step its way is past its slider ([`AT_TOP`],
/// [`AT_BOTTOM`]; the step a check guide's fix takes, `SliderRange::step_from`). `None` when the
/// input is a link: a choice, an entered number and an optional number left blank (it enters at
/// a result) are never at an end.
fn unlinked_reason(change: &Change, view: &NoteView<'_>) -> Option<String> {
    if let Some(reason) = (view.held)(change.input) {
        return Some(reason);
    }
    let up = change.step.up()?;
    let range = InputCatalogue::get().entry(change.input)?.meta.range?;
    let x = match view.inputs.get(change.input)? {
        Value::Num(x) if x.is_finite() => x,
        Value::Int(n) => n as f64,
        _ => return None,
    };
    if range.step_from(x, up).is_some() {
        return None;
    }
    Some(if up { AT_TOP } else { AT_BOTTOM }.to_owned())
}

/// What a `{path}` in a note shows on the design whose results are `results` (slice-2 spec
/// 3.2): the result's number with its unit, as the readouts show it
/// (`guidance::result_text`); `None`, drawn as [`note_text::NO_VALUE`], for a
/// path no result has, a number that is not finite, a text (a result that reads "n/a"
/// where it has no number, or a verdict: a live value is a number), and a result computed
/// from an out-of-range end effect (the pull-out and everything after it: the model gives no
/// prediction there, spec 5).
pub fn live_value(results: &DesignResults, path: &str) -> Option<String> {
    if end_effect_out_of_range(results).is_some()
        && (after_the_end_factor().contains(path) || greyed_by_end_effect(results, path))
    {
        return None;
    }
    match results.get(path)? {
        Value::Num(x) if x.is_finite() => Some(result_text(results, path)),
        Value::Int(_) => Some(result_text(results, path)),
        Value::Num(_) | Value::Text(_) | Value::None => None,
    }
}

/// Every result computed from the end-effect factor (the registry is fixed: computed once).
fn after_the_end_factor() -> &'static std::collections::BTreeSet<String> {
    static AFTER: std::sync::OnceLock<std::collections::BTreeSet<String>> =
        std::sync::OnceLock::new();
    AFTER.get_or_init(|| registry().downstream("model.f_end"))
}

/// A note's text as drawn on `view`'s design: each `{path}` replaced by its live value
/// ([`live_value`]), then through `glyph_safe`.
pub(crate) fn note_text_on(text: &str, view: &NoteView<'_>) -> String {
    glyph_safe(&note_text::render(text, |path| {
        live_value(view.results, path)
    }))
}

/// A teaching note (slice-2 spec 3.1): its title; what it is (its sentences) and its diagram;
/// "What to change", each entry "`<Step> <input's label>: <text>`", the label a link to the
/// input's row unless `unlinked_reason` gives a reason (then plain text with the reason);
/// "Watch out" and its lines; and the "Where this comes from" fold, closed until clicked,
/// holding its sources. The sentences, change texts and watch-outs show live values
/// (`note_text_on`); every text is drawn through `glyph_safe`. Returns the link clicked.
pub fn note_ui(ui: &mut egui::Ui, note: &Note, view: &NoteView<'_>) -> Option<NoteLink> {
    let mut clicked = None;
    ui.strong(glyph_safe(note.title));
    for sentence in note.sentences {
        ui.add(egui::Label::new(note_text_on(sentence, view)).wrap());
    }
    if let Some(diagram) = note.diagram {
        diagram_ui(ui, diagram);
    }
    if !note.change.is_empty() {
        ui.strong(WHAT_TO_CHANGE);
        for change in note.change {
            if change_ui(ui, change, view) {
                clicked = Some(NoteLink::Input(change.input));
            }
        }
    }
    if !note.watch_out.is_empty() {
        let warn = ui.visuals().warn_fg_color;
        ui.label(egui::RichText::new(WATCH_OUT).strong().color(warn));
        for line in note.watch_out {
            let text = egui::RichText::new(note_text_on(line, view)).color(warn);
            ui.add(egui::Label::new(text).wrap());
        }
    }
    egui::CollapsingHeader::new(WHERE_FROM)
        .id_salt(("magcoupling_note_sources", note.id))
        .default_open(false)
        .show(ui, |ui| {
            for source in note.sources {
                ui.add(
                    egui::Label::new(egui::RichText::new(glyph_safe(source)).small().weak()).wrap(),
                );
            }
        });
    clicked
}

/// One change entry on one wrapped line: its step word, its input's display label (a link, or
/// plain text and the reason), then ": " and its text. True when the link was clicked.
fn change_ui(ui: &mut egui::Ui, change: &Change, view: &NoteView<'_>) -> bool {
    let label = glyph_safe(label_for(change.input));
    let mut clicked = false;
    wrapping_line(ui, |ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label(format!("{} ", change.step.word()));
        match unlinked_reason(change, view) {
            Some(reason) => {
                ui.label(label);
                ui.weak(format!(" ({reason})"));
            }
            None => clicked = ui.link(label).clicked(),
        }
        ui.add(egui::Label::new(format!(": {}", note_text_on(change.text, view))).wrap());
    });
    clicked
}

/// The links at the head of a note in the Explain panel to the equations it explains (a
/// family's template is left out), after [`EXPLAINS`]; nothing for a note that explains none.
/// Returns the equation clicked, which the panel opens here.
pub fn explains_ui(ui: &mut egui::Ui, note: &Note) -> Option<&'static str> {
    let paths: Vec<&'static str> = note
        .equations
        .iter()
        .copied()
        .filter(|p| !p.contains('#') && registry().equation_for(p).is_some())
        .collect();
    let mut clicked = None;
    if !paths.is_empty() {
        wrapping_line(ui, |ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.strong(EXPLAINS);
            for path in paths {
                if ui
                    .link(crumb(path))
                    .on_hover_text(term_label(path))
                    .clicked()
                {
                    clicked = Some(path);
                }
            }
        });
    }
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::explain::notes::{self, Review};
    use crate::gui::dashboard::result_info;
    use crate::gui::explorer::reviewed_note;
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::test_support::{design, nothing_held};

    /// A note with a change entry of each kind, for the renderer's own tests (no shipped note
    /// has change entries before the note tasks rewrite them).
    const RENDER_NOTE: Note = Note {
        id: "test.render",
        title: "A note to draw",
        equations: &[],
        sentences: &["The first sentence of the note.", "The second sentence."],
        change: &[
            Change {
                input: "metal.face_gap_mm",
                step: notes::Step::Lower,
                aim: Some(("model.pullout_Nm", notes::Sense::Up)),
                text: "the entry on the face gap.",
            },
            Change {
                input: "coupling.op_temp_C",
                step: notes::Step::Raise,
                aim: Some(("model.pullout_Nm", notes::Sense::Down)),
                text: "the entry on the operating temperature.",
            },
            Change {
                input: "materials.parts.back_iron",
                step: notes::Step::Choose,
                aim: None,
                text: "the entry on the back iron.",
            },
        ],
        watch_out: &["The first watch-out.", "The second watch-out."],
        diagram: None,
        sources: &["The first source.", "The second source."],
        review: Review::Draft,
    };

    /// One frame of `note` drawn against `view` with `events`: what egui painted and the link a
    /// click asked for.
    fn note_frame(
        ctx: &egui::Context,
        note: &Note,
        view: &NoteView<'_>,
        events: Vec<egui::Event>,
    ) -> (egui::FullOutput, Option<NoteLink>) {
        use crate::gui::test_support::{SCREEN, sized_frame};
        let mut link = None;
        let output = sized_frame(ctx, SCREEN, events, |ui| {
            if let Some(clicked) = note_ui(ui, note, view) {
                link = Some(clicked);
            }
        });
        (output, link)
    }

    /// Clicks the first drawn text equal to `text` (move, press, release): the link the click
    /// asked for, and a frame drawn once a fold the click opened has settled.
    fn click_in_note(
        ctx: &egui::Context,
        note: &Note,
        view: &NoteView<'_>,
        text: &str,
    ) -> (Option<NoteLink>, egui::FullOutput) {
        use crate::gui::test_support::{primary_button, text_rect};
        let (output, _) = note_frame(ctx, note, view, Vec::new());
        // A text in a wrapped row is laid out from the row's start, so the left part of its rect
        // covers the widgets before it on that row: click its last character.
        let rect = text_rect(&output, text).unwrap_or_else(|| panic!("no text {text:?}"));
        let at = egui::pos2(rect.max.x - 2.0, rect.center().y);
        note_frame(ctx, note, view, vec![egui::Event::PointerMoved(at)]);
        note_frame(ctx, note, view, vec![primary_button(at, true)]);
        let (_, link) = note_frame(ctx, note, view, vec![primary_button(at, false)]);
        let mut output = note_frame(ctx, note, view, Vec::new()).0;
        for _ in 0..15 {
            output = note_frame(ctx, note, view, Vec::new()).0;
        }
        (link, output)
    }

    /// Asserts that `note`'s entry on `input` draws its label as plain text: a click on it asks
    /// for no link, and " (<reason>)" is drawn once. The entry on `still_linked` stays a link.
    fn assert_entry_plain_with_reason(
        ctx: &egui::Context,
        note: &Note,
        view: &NoteView<'_>,
        input: &str,
        reason: &str,
        still_linked: &'static str,
    ) {
        use crate::gui::test_support::drawn_count;
        let (link, output) = click_in_note(ctx, note, view, label_for(input));
        assert_eq!(link, None, "{input}: an unlinked input is no link");
        assert_eq!(drawn_count(&output, &format!(" ({reason})")), 1, "{input}");
        let (link, _) = click_in_note(ctx, note, view, label_for(still_linked));
        assert_eq!(
            link,
            Some(NoteLink::Input(still_linked)),
            "{still_linked} stays a link"
        );
    }

    /// The slider of the input at `path`.
    fn slider(path: &str) -> crate::engine::meta::SliderRange {
        InputCatalogue::get()
            .entry(path)
            .unwrap()
            .meta
            .range
            .unwrap()
    }

    /// The panel's `held` with Auto off: only a pick holds an input of `inputs`.
    fn auto_off(inputs: &DesignInputs) -> impl Fn(&str) -> Option<String> + '_ {
        move |path| held_reason(&SizingState::default(), inputs, path)
    }

    /// Why `change` draws its input as plain text on `inputs`, with `held` naming held inputs.
    fn unlinked_with(
        inputs: &DesignInputs,
        change: &Change,
        held: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        let results = compute_all(inputs);
        let view = NoteView {
            inputs,
            results: &results,
            held,
        };
        unlinked_reason(change, &view)
    }

    /// Why `change` draws its input as plain text on `inputs` with Auto off ([`auto_off`]).
    fn unlinked_on(inputs: &DesignInputs, change: &Change) -> Option<String> {
        unlinked_with(inputs, change, &auto_off(inputs))
    }

    #[test]
    fn a_note_draws_its_parts_in_order_with_its_sources_folded_until_clicked() {
        use crate::gui::test_support::{drawn_texts, text_color, text_rect};
        let ctx = egui::Context::default();
        let (inputs, results) = design();
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &nothing_held,
        };
        note_frame(&ctx, &RENDER_NOTE, &view, Vec::new());
        let (output, _) = note_frame(&ctx, &RENDER_NOTE, &view, Vec::new());
        let texts = drawn_texts(&output);
        let count = |text: &str| texts.iter().filter(|t| *t == text).count();
        let once = [RENDER_NOTE.title, WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM]
            .into_iter()
            .chain(RENDER_NOTE.sentences.iter().copied())
            .chain(RENDER_NOTE.watch_out.iter().copied());
        for text in once {
            assert_eq!(count(text), 1, "{text}");
        }
        for change in RENDER_NOTE.change {
            assert_eq!(
                count(&format!("{} ", change.step.word())),
                1,
                "{}",
                change.input
            );
            assert_eq!(count(label_for(change.input)), 1, "{}", change.input);
            assert_eq!(count(&format!(": {}", change.text)), 1, "{}", change.input);
        }
        for source in RENDER_NOTE.sources {
            assert_eq!(count(source), 0, "the fold starts closed: {source}");
        }
        // Top to bottom: what it is, what to change, watch out, where this comes from.
        let top = |text: &str| text_rect(&output, text).unwrap().min.y;
        let order = [
            RENDER_NOTE.sentences[1],
            WHAT_TO_CHANGE,
            label_for(RENDER_NOTE.change[2].input),
            WATCH_OUT,
            RENDER_NOTE.watch_out[1],
            WHERE_FROM,
        ];
        for pair in order.windows(2) {
            assert!(top(pair[0]) < top(pair[1]), "{pair:?}");
        }
        let warn = ctx.style().visuals.warn_fg_color;
        for line in RENDER_NOTE.watch_out {
            assert_eq!(text_color(&output, line), Some(warn), "{line}");
        }
        let (link, output) = click_in_note(&ctx, &RENDER_NOTE, &view, WHERE_FROM);
        assert_eq!(link, None, "the fold is no link to an input");
        let texts = drawn_texts(&output);
        for source in RENDER_NOTE.sources {
            assert!(texts.iter().any(|t| t == source), "{source}");
        }
    }

    #[test]
    fn a_free_input_s_label_is_a_link_to_its_row_and_the_text_is_not() {
        let ctx = egui::Context::default();
        let (inputs, results) = design();
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &nothing_held,
        };
        for change in RENDER_NOTE.change {
            let (link, _) = click_in_note(&ctx, &RENDER_NOTE, &view, label_for(change.input));
            assert_eq!(
                link,
                Some(NoteLink::Input(change.input)),
                "{}",
                change.input
            );
            let text = format!(": {}", change.text);
            let (link, _) = click_in_note(&ctx, &RENDER_NOTE, &view, &text);
            assert_eq!(link, None, "{text}");
        }
    }

    #[test]
    fn an_input_that_is_not_free_is_plain_text_with_the_reason() {
        let ctx = egui::Context::default();
        let (inputs, results) = design();
        let held_entry = &RENDER_NOTE.change[0];
        let held = |path: &str| (path == held_entry.input).then(|| HELD_BY_AUTO.to_owned());
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &held,
        };
        let free = RENDER_NOTE.change[1].input;
        assert_entry_plain_with_reason(
            &ctx,
            &RENDER_NOTE,
            &view,
            held_entry.input,
            HELD_BY_AUTO,
            free,
        );
    }

    #[test]
    fn an_entry_whose_input_cannot_move_its_way_names_why() {
        use crate::engine::explain::notes::Step;
        let entry = |input, step| Change {
            input,
            step,
            aim: None,
            text: "",
        };
        let reason =
            |inputs: &DesignInputs, change: &Change| unlinked_with(inputs, change, &nothing_held);
        let gap = "metal.face_gap_mm";
        let (lowest, highest) = (slider(gap).min, slider(gap).max);
        let mut inputs = DesignInputs::default();
        assert_eq!(reason(&inputs, &entry(gap, Step::Raise)), None);
        assert_eq!(reason(&inputs, &entry(gap, Step::Lower)), None);
        inputs.set(gap, Value::Num(highest)).unwrap();
        assert_eq!(
            reason(&inputs, &entry(gap, Step::Raise)).as_deref(),
            Some(AT_TOP)
        );
        assert_eq!(reason(&inputs, &entry(gap, Step::Lower)), None);
        inputs.set(gap, Value::Num(lowest)).unwrap();
        assert_eq!(
            reason(&inputs, &entry(gap, Step::Lower)).as_deref(),
            Some(AT_BOTTOM)
        );
        assert_eq!(reason(&inputs, &entry(gap, Step::Raise)), None);
        // A count at its end; a choice and a blank optional number are never at an end.
        let mut poles = DesignInputs::default();
        poles.coupling.npole = slider("coupling.npole").max as i64;
        assert_eq!(
            reason(&poles, &entry("coupling.npole", Step::Raise)).as_deref(),
            Some(AT_TOP)
        );
        let defaults = DesignInputs::default();
        assert_eq!(
            reason(&defaults, &entry("materials.parts.back_iron", Step::Choose)),
            None
        );
        assert_eq!(
            reason(
                &defaults,
                &entry("coupling.magnets.axial_length_mm", Step::Raise)
            ),
            None
        );
        // A held input names why it is held, whatever its slider says.
        let held = |_: &str| Some(HELD_BY_AUTO.to_owned());
        assert_eq!(
            unlinked_with(&inputs, &entry(gap, Step::Lower), &held).as_deref(),
            Some(HELD_BY_AUTO)
        );
    }

    #[test]
    fn held_reason_names_auto_and_the_pick_that_sets_an_input() {
        use crate::engine::explain::plain::plain_text_violations;
        use crate::engine::picks::BACK_IRON_PICK;
        use crate::engine::sizing::FreeVariable;
        use crate::gui::sizing::SizingMode;
        let inputs = DesignInputs::default();
        let off = SizingState::default();
        let auto = SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable: FreeVariable::AxialLength,
        };
        let length = FreeVariable::AxialLength.path();
        assert_eq!(held_reason(&off, &inputs, length), None);
        assert_eq!(
            held_reason(&auto, &inputs, length).as_deref(),
            Some(HELD_BY_AUTO)
        );
        assert_eq!(held_reason(&auto, &inputs, "metal.face_gap_mm"), None);
        let part_pick = format!(
            "set by the {} pick",
            label_for("coupling.magnets.part_inner")
        );
        assert_eq!(
            held_reason(&off, &inputs, "coupling.magnets.manual_inner_br_T"),
            Some(part_pick.clone())
        );
        let steel = "materials.steel.cte_per_C";
        assert_eq!(held_reason(&off, &inputs, steel), None);
        let mut picked = inputs.clone();
        picked.materials.parts.back_iron = 2;
        let steel_pick = format!("set by the {} pick", label_for(BACK_IRON_PICK));
        assert_eq!(held_reason(&off, &picked, steel), Some(steel_pick.clone()));
        for reason in [
            HELD_BY_AUTO.to_owned(),
            part_pick,
            steel_pick,
            AT_TOP.into(),
            AT_BOTTOM.into(),
        ] {
            assert_eq!(
                plain_text_violations(&reason),
                Vec::<&str>::new(),
                "{reason}"
            );
        }
    }

    #[test]
    fn the_outer_block_entries_name_the_part_pick_that_sets_them() {
        // Review Focus RF-4 (amendment A-25 T-4, the harmonics part): at the defaults the outer
        // library part sets the block's width and thickness, so the harmonics and back-iron
        // entries on them are plain text naming the pick; with custom dimensions (the Y30 probe
        // design, both parts blank) the same entries are links.
        use crate::engine::explain::probes::probe_designs;
        let entry = |id: &str, input: &str| {
            *notes::note(id)
                .and_then(|n| n.change.iter().find(|c| c.input == input))
                .unwrap_or_else(|| panic!("{id}: no entry on {input}"))
        };
        let width = entry("harmonics", "coupling.magnets.manual_outer_width_mm");
        let thickness = entry(
            "back_iron_factor",
            "coupling.magnets.manual_outer_thickness_mm",
        );
        let pick = format!(
            "set by the {} pick",
            label_for("coupling.magnets.part_outer")
        );
        let defaults = DesignInputs::default();
        assert_eq!(unlinked_on(&defaults, &width), Some(pick.clone()));
        assert_eq!(unlinked_on(&defaults, &thickness), Some(pick));
        let (_, custom) = probe_designs()
            .into_iter()
            .find(|(name, _)| *name == "Y30 ferrite, custom dimensions")
            .expect("the custom-dimensions probe design");
        assert_eq!(unlinked_on(&custom, &width), None);
        assert_eq!(unlinked_on(&custom, &thickness), None);
    }

    #[test]
    fn the_wall_note_s_design_flux_entry_names_the_pick_that_sets_it() {
        // Review Focus RF-4 (amendment A-25 T-4): 1018 brings its own design flux density, so the
        // wall note's entry on that input is plain text naming the back-iron pick; 12L14 has none
        // of its own and leaves the input in effect, so the same entry is a link.
        use crate::engine::material_library::BACK_IRON_CHOICES;
        use crate::engine::picks::BACK_IRON_PICK;
        assert_eq!(BACK_IRON_CHOICES[1], (2, "1018_hot_rolled"));
        assert_eq!(BACK_IRON_CHOICES[2], (3, "12L14_cold_drawn"));
        let entry = *notes::note("a5.low_saturation")
            .and_then(|n| {
                n.change
                    .iter()
                    .find(|c| c.input == "materials.steel.bsat_T")
            })
            .expect("the wall note has an entry on the design flux density");
        let reason = |back_iron: i64| {
            let mut inputs = DesignInputs::default();
            inputs.materials.parts.back_iron = back_iron;
            unlinked_on(&inputs, &entry)
        };
        let set = reason(2).expect("1018 sets the design flux density");
        assert!(set.starts_with("set by the"), "{set}");
        assert_eq!(
            set,
            format!("set by the {} pick", label_for(BACK_IRON_PICK))
        );
        assert_eq!(reason(3), None);
    }

    #[test]
    fn the_sleeve_entry_at_the_bottom_of_its_slider_is_plain_text_with_the_reason() {
        // Amendment A-25 T-7: at the sleeve's slider minimum one step thinner is past the slider,
        // so the sleeve note's entry draws its label as plain text with the reason and no link;
        // the liner entry beside it stays a link.
        let sleeve = "metal.sleeve_mm";
        let note = notes::note("a5.high_conductivity_sleeve_or_liner").unwrap();
        let entry = note
            .change
            .iter()
            .find(|c| c.input == sleeve)
            .expect("the sleeve note has an entry on the sleeve thickness");
        assert_eq!(entry.step, notes::Step::Lower);
        let mut inputs = DesignInputs::default();
        inputs.set(sleeve, Value::Num(slider(sleeve).min)).unwrap();
        let results = compute_all(&inputs);
        let held = auto_off(&inputs);
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &held,
        };
        let ctx = egui::Context::default();
        let liner = "metal.liner_mm";
        assert_entry_plain_with_reason(&ctx, note, &view, sleeve, AT_BOTTOM, liner);
    }

    #[test]
    fn a_live_value_reads_as_the_readouts_and_none_where_there_is_no_prediction() {
        use crate::engine::explain::probes::probe_designs;
        let designs = probe_designs();
        let default = compute_all(&designs[0].1);
        let pullout = "model.pullout_Nm";
        let unit = result_info(pullout).unwrap().meta.unit;
        assert_eq!(
            live_value(&default, pullout),
            Some(with_unit(
                format_value(&default.get(pullout).unwrap()),
                unit
            ))
        );
        // A text where a number belongs: the cold limit of NdFeB rings reads "n/a".
        let cold = "temperature.demag.cold_limit_C";
        assert!(matches!(default.get(cold), Some(Value::Text(_))));
        assert_eq!(live_value(&default, cold), None);
        // On the ferrite probe the cold limit is a number and the hot onsets are not finite.
        let ferrite = compute_all(&designs[3].1);
        let cold_unit = result_info(cold).unwrap().meta.unit;
        assert_eq!(
            live_value(&ferrite, cold),
            Some(with_unit(
                format_value(&ferrite.get(cold).unwrap()),
                cold_unit
            ))
        );
        let onset = "temperature.demag.onset_skipping_C";
        assert!(matches!(ferrite.get(onset), Some(Value::Num(x)) if !x.is_finite()));
        assert_eq!(live_value(&ferrite, onset), None);
        // A verdict, an input and a path no result has: none.
        assert_eq!(live_value(&default, "materials.cup_wall_check"), None);
        assert_eq!(live_value(&default, "metal.face_gap_mm"), None);
        assert_eq!(live_value(&default, "no.such.result"), None);
    }

    #[test]
    fn a_value_computed_from_an_out_of_range_pull_out_reads_no_prediction() {
        let results = compute_all(&crate::gui::test_support::short_magnets());
        assert!(crate::gui::dashboard::end_effect_out_of_range(&results).is_some());
        for path in [
            "model.pullout_Nm",
            "metal.torque_hot_low_Nm",
            "metal.cold_high_input_Nm",
            "model.gearbox_input_ripple_Nm",
        ] {
            assert_eq!(live_value(&results, path), None, "{path}");
        }
        for path in [
            "model.f_end",
            "model.pole_pitch_mm",
            "model.active_length_mm",
        ] {
            assert!(live_value(&results, path).is_some(), "{path}");
        }
    }

    /// A note with live values in a sentence, a change entry and a watch-out.
    const LIVE_NOTE: Note = Note {
        id: "test.live",
        title: "A note with live values",
        equations: &[],
        sentences: &[
            "The first sentence has none.",
            "Here the pull-out is {model.pullout_Nm}.",
        ],
        change: &[Change {
            input: "metal.face_gap_mm",
            step: notes::Step::Lower,
            aim: Some(("model.pullout_Nm", notes::Sense::Up)),
            text: "the skipping onset is {temperature.demag.onset_skipping_C} here.",
        }],
        watch_out: &["The end factor is {model.f_end} on this design."],
        diagram: None,
        sources: &["A source."],
        review: Review::Draft,
    };

    #[test]
    fn a_note_draws_its_live_values_from_the_design_shown() {
        use crate::engine::explain::note_text::NO_VALUE;
        use crate::engine::explain::probes::probe_designs;
        use crate::gui::test_support::drawn_texts;
        let ctx = egui::Context::default();
        let designs = probe_designs();
        for (name, inputs) in [&designs[0], &designs[3]] {
            let results = compute_all(inputs);
            let view = NoteView {
                inputs,
                results: &results,
                held: &nothing_held,
            };
            note_frame(&ctx, &LIVE_NOTE, &view, Vec::new());
            let (output, _) = note_frame(&ctx, &LIVE_NOTE, &view, Vec::new());
            let texts = drawn_texts(&output);
            let value = |path: &str| live_value(&results, path).unwrap_or(NO_VALUE.to_owned());
            for want in [
                LIVE_NOTE.sentences[0].to_owned(),
                format!("Here the pull-out is {}.", value("model.pullout_Nm")),
                format!(
                    ": the skipping onset is {} here.",
                    value("temperature.demag.onset_skipping_C")
                ),
                format!("The end factor is {} on this design.", value("model.f_end")),
            ] {
                assert!(texts.contains(&want), "{name}: {want:?} in {texts:?}");
            }
        }
        // The ferrite rings' hot onset has no prediction; the NdFeB default's has one.
        let onset = "temperature.demag.onset_skipping_C";
        assert!(live_value(&compute_all(&designs[0].1), onset).is_some());
        assert_eq!(live_value(&compute_all(&designs[3].1), onset), None);
    }

    /// BL-064 (spec 6.1 and 7.11): the shaft-joint note's sentence 4 as the panel draws it: the
    /// temperature the inner hub's fit would need, beside the wheel shaft's live shrink, against
    /// the allowed temperature. At the defaults; at library N42 parts in both rings, where the
    /// inner ring's single-ring onset less the cure margin binds (owner R-15; the guide tests'
    /// `breaking(n42_parts)`); with mixed rings (inner N42M, outer N35; the guide tests'
    /// `breaking(mixed_rings)`), where the outer ring governs and the cap is still the inner
    /// ring's own onset less the margin (the owner's ring ruling); on an uncooled wheel shaft,
    /// whose shrink reads 0; and on a 40 °C one, its slider's top, where the shaft grows (the
    /// render the physics review decides on).
    #[test]
    fn the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink() {
        use crate::engine::explain::note_text::render;
        use crate::engine::temperature::RING_OUTER;
        use crate::gui::test_support::{mixed_rings, n42_parts, passing_design};
        let sentence = notes::note("shaft_joint")
            .expect("the shaft-joint note")
            .sentences[4];
        let drawn = |inputs: &DesignInputs| {
            let results = compute_all(inputs);
            render(sentence, |path: &str| live_value(&results, path))
        };
        let wheel_shaft_at = |celsius: f64| {
            let mut inputs = DesignInputs::default();
            inputs.shaft_joint.wheel_shaft_fit_C = celsius;
            inputs
        };
        let mut n42 = passing_design();
        n42_parts(&mut n42);
        // Mixed rings: the outer ring governs the magnets' limit, so its single-ring onset is not
        // the inner ring's own, which the bonded inner hub's cap reads.
        let mut mixed = passing_design();
        mixed_rings(&mut mixed);
        let demag = compute_all(&mixed).temperature.demag;
        assert_eq!(demag.demag_ring, RING_OUTER);
        assert_ne!(demag.onset_single_ring_C, demag.inner_onset_single_ring_C);
        for (inputs, words) in [
            (
                DesignInputs::default(),
                "would need 93.27 °C on the wheel shaft shrunk 18.99 µm; allowed: 100.0 °C,",
            ),
            (n42, "allowed: 59.83 °C,"),
            (mixed, "allowed: 81.35 °C,"),
            (
                wheel_shaft_at(20.0),
                "would need 247.6 °C on the wheel shaft shrunk 0 µm;",
            ),
            (
                wheel_shaft_at(40.0),
                "would need 266.3 °C on the wheel shaft shrunk -2.300 µm;",
            ),
        ] {
            let text = drawn(&inputs);
            assert!(text.contains(words), "{words:?} in {text:?}");
        }
    }

    #[test]
    fn every_note_draws_at_every_probe_design_with_glyphs_and_no_braces() {
        use crate::engine::explain::note_text::rendered_texts;
        use crate::engine::explain::probes::probe_designs;
        use crate::gui::test_support::{assert_glyphs, glyph_context};
        let ctx = glyph_context();
        for (name, inputs) in probe_designs() {
            let results = compute_all(&inputs);
            let view = NoteView {
                inputs: &inputs,
                results: &results,
                held: &nothing_held,
            };
            for note in notes::NOTES {
                for text in rendered_texts(note) {
                    let drawn = note_text_on(text, &view);
                    assert!(
                        !drawn.contains(['{', '}']),
                        "{} at {name}: {drawn}",
                        note.id
                    );
                    assert_glyphs(&ctx, &drawn, note.id);
                }
            }
        }
    }

    /// The per-note physics review's sheet (slice-2 spec section 4): each note, or each named
    /// in `NOTES_FOR_REVIEW` (ids, comma-separated), its title and sources once, then at each
    /// probe design its sentences, change entries and watch-outs as the panel draws them; a
    /// change entry with its input's display label, the reason it is not a link there, and
    /// its aim's value before and after one step. A text the parser refuses or the plain-words
    /// guard flags is marked "<<<". Written to `NOTES_REVIEW_OUT` (a path from `magcoupling-rs/`;
    /// default `target/notes-for-review.txt` there).
    #[test]
    #[ignore = "a review tool, not a check: NOTES_FOR_REVIEW=id1,id2 NOTES_REVIEW_OUT=<file> cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture"]
    fn notes_for_review() {
        use std::fmt::Write as _;

        use crate::engine::explain::note_text::segments;
        use crate::engine::explain::plain::plain_text_violations;
        use crate::engine::explain::probes::{probe_designs, step_input};
        let wanted: Vec<String> = std::env::var("NOTES_FOR_REVIEW")
            .map(|ids| {
                ids.split(',')
                    .map(|id| id.trim().to_owned())
                    .filter(|id| !id.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        let unknown: Vec<&String> = wanted
            .iter()
            .filter(|id| notes::note(id).is_none())
            .collect();
        assert!(
            unknown.is_empty(),
            "NOTES_FOR_REVIEW names no note: {unknown:?}"
        );
        let chosen: Vec<&Note> = notes::NOTES
            .iter()
            .filter(|n| wanted.is_empty() || wanted.iter().any(|id| id == n.id))
            .collect();
        let designs: Vec<(&str, DesignInputs, DesignResults)> = probe_designs()
            .into_iter()
            .map(|(name, inputs)| {
                let results = compute_all(&inputs);
                (name, inputs, results)
            })
            .collect();
        // A raw text the parser refuses, or a drawn text the guard flags, is marked.
        let marks = |raw: &str, drawn: &str| {
            let mut found = Vec::new();
            if let Err(fault) = segments(raw) {
                found.push(fault);
            }
            let banned = plain_text_violations(drawn);
            if !banned.is_empty() {
                found.push(format!("plain words: {banned:?}"));
            }
            if found.is_empty() {
                String::new()
            } else {
                format!("   <<< {}", found.join("; "))
            }
        };
        let sizing = SizingState::default();
        let mut sheet = String::new();
        for note in &chosen {
            let status = match note.review {
                Review::Draft => "Draft",
                Review::Reviewed { .. } => "Reviewed",
            };
            writeln!(sheet, "==== {} ({status})", note.id).unwrap();
            writeln!(
                sheet,
                "Title: {}{}",
                note.title,
                marks(note.title, note.title)
            )
            .unwrap();
            writeln!(sheet, "{WHERE_FROM}:").unwrap();
            for source in note.sources {
                writeln!(sheet, "  - {source}{}", marks(source, source)).unwrap();
            }
            for (name, inputs, results) in &designs {
                let held = |path: &str| held_reason(&sizing, inputs, path);
                let view = NoteView {
                    inputs,
                    results,
                    held: &held,
                };
                writeln!(sheet, "\n-- at the {name} design").unwrap();
                writeln!(sheet, "What it is:").unwrap();
                for (i, sentence) in note.sentences.iter().enumerate() {
                    let drawn = note_text_on(sentence, &view);
                    writeln!(sheet, "  {}. {drawn}{}", i + 1, marks(sentence, &drawn)).unwrap();
                }
                if !note.change.is_empty() {
                    writeln!(sheet, "{WHAT_TO_CHANGE}:").unwrap();
                }
                for change in note.change {
                    let reason =
                        unlinked_reason(change, &view).map_or(String::new(), |r| format!(" ({r})"));
                    let drawn = note_text_on(change.text, &view);
                    writeln!(
                        sheet,
                        "  - {} {}{reason}: {drawn}{}",
                        change.step.word(),
                        label_for(change.input),
                        marks(change.text, &drawn)
                    )
                    .unwrap();
                    let Some((aim, sense)) = change.aim else {
                        continue;
                    };
                    let outcome = match step_input(inputs, change.input, change.step) {
                        Err(why) => format!("not stepped here: {why:?}"),
                        Ok(stepped) => match (results.get(aim), compute_all(&stepped).get(aim)) {
                            (Some(Value::Num(before)), Some(Value::Num(after)))
                                if before.is_finite() && after.is_finite() =>
                            {
                                let verdict = if sense.holds(before, after) {
                                    "moves as it says"
                                } else {
                                    "DOES NOT MOVE AS IT SAYS"
                                };
                                format!("{before} -> {after}: {verdict}")
                            }
                            (before, after) => {
                                format!("no numbers to compare: {before:?} -> {after:?}")
                            }
                        },
                    };
                    writeln!(sheet, "      aim {aim} {sense:?}, one step: {outcome}").unwrap();
                }
                if !note.watch_out.is_empty() {
                    writeln!(sheet, "{WATCH_OUT}:").unwrap();
                }
                for line in note.watch_out {
                    let drawn = note_text_on(line, &view);
                    writeln!(sheet, "  - {drawn}{}", marks(line, &drawn)).unwrap();
                }
            }
            writeln!(sheet).unwrap();
        }
        let out = std::env::var("NOTES_REVIEW_OUT").map_or_else(
            |_| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/notes-for-review.txt")
            },
            std::path::PathBuf::from,
        );
        if let Some(dir) = out.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).expect("the sheet's folder is made");
        }
        std::fs::write(&out, &sheet).expect("the sheet is written");
        println!(
            "{} notes at {} designs: {}",
            chosen.len(),
            designs.len(),
            out.display()
        );
    }

    #[test]
    fn a_wrapped_change_entry_stays_inside_a_narrow_panel_and_its_label_links() {
        // Amendment A-25 T-6 (spec section 5, a narrow window): the pull-out note drawn alone in
        // a panel 400 points wide. Every change entry's input label is drawn inside the
        // panel, and a click on the last character of the inner magnet part's label, laid out in
        // a wrapped row, asks for that input's row.
        use crate::gui::test_support::{last_characters, primary_button, sized_frame, text_rects};
        let size = egui::vec2(400.0, 800.0);
        let note = reviewed_note("pullout_angle").expect("the pull-out note is reviewed");
        let (inputs, results) = design();
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &nothing_held,
        };
        let ctx = egui::Context::default();
        let frame = |events: Vec<egui::Event>| {
            let mut link = None;
            let output = sized_frame(&ctx, size, events, |ui| {
                link = note_ui(ui, note, &view);
            });
            (output, link)
        };
        frame(Vec::new());
        let (output, _) = frame(Vec::new());
        let panel = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        for change in note.change {
            let label = glyph_safe(label_for(change.input));
            let rects = text_rects(&output, &label);
            assert_eq!(rects.len(), 1, "{label:?} is drawn once");
            assert!(panel.contains_rect(rects[0]), "{label:?} at {:?}", rects[0]);
        }
        let part = "coupling.magnets.part_inner";
        let at = last_characters(&output, &glyph_safe(label_for(part)))[0];
        frame(vec![egui::Event::PointerMoved(at)]);
        frame(vec![primary_button(at, true)]);
        let (_, link) = frame(vec![primary_button(at, false)]);
        assert_eq!(link, Some(NoteLink::Input(part)));
    }
}
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        use crate::gui::explain::{NUMBERS, WHAT_IT_IS, WHAT_SETS_IT};
        use crate::gui::explorer::WHAT_TO_CHANGE;
        use crate::gui::test_support::{primary_button, text_center};
```

with:

```rust
        use crate::gui::explain::{NUMBERS, WHAT_IT_IS, WHAT_SETS_IT};
        use crate::gui::notes_view::WHAT_TO_CHANGE;
        use crate::gui::test_support::{primary_button, text_center};
```

Replace:

```rust
        assert!(
            script.contains(crate::gui::explorer::WHAT_TO_CHANGE),
            "a note's What to change"
```

with:

```rust
        assert!(
            script.contains(crate::gui::notes_view::WHAT_TO_CHANGE),
            "a note's What to change"
```

Replace:

```rust
        assert!(
            script.contains(crate::gui::explorer::WHERE_FROM),
            "a note's sources fold"
```

with:

```rust
        assert!(
            script.contains(crate::gui::notes_view::WHERE_FROM),
            "a note's sources fold"
```

**`magcoupling-rs/src/gui/explain.rs`**

Replace:

```rust
};
use crate::gui::explorer::{
    NoteLink, NoteView, START_HERE_BUTTON, WATCH_OUT, WHAT_TO_CHANGE, explains_ui, note_ui,
    reviewed_note, start_here_hover,
};
use crate::gui::format::{format_value, value_with_unit};
```

with:

```rust
};
use crate::gui::explorer::{START_HERE_BUTTON, reviewed_note, start_here_hover};
use crate::gui::format::{format_value, value_with_unit};
```

Replace:

```rust
use crate::gui::look;
use crate::gui::plain_text::{
```

with:

```rust
use crate::gui::look;
use crate::gui::notes_view::{NoteLink, NoteView, WATCH_OUT, WHAT_TO_CHANGE, explains_ui, note_ui};
use crate::gui::plain_text::{
```

Replace:

```rust
        // the selection; a change entry's input shows its row, as a check's fix does.
        use crate::gui::explorer::{EXPLAINS, WHERE_FROM, crumb};
        let ctx = egui::Context::default();
```

with:

```rust
        // the selection; a change entry's input shows its row, as a check's fix does.
        use crate::gui::explorer::crumb;
        use crate::gui::notes_view::{EXPLAINS, WHERE_FROM};
        let ctx = egui::Context::default();
```

**`magcoupling-rs/src/gui/explorer.rs`**

Replace:

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

with:

```rust
//!
//! Teaching notes (spec Addendum A4; UX overhaul decision C-11): the Explain panel draws them
//! ([`crate::gui::notes_view`]), not this panel. Under the open equation the lesson line names
//! the reviewed note explaining it ([`notes::note_for`]: only notes a physics reviewer has
//! checked) after "Full lesson:", a link that shows the note in the Explain panel
//! ([`lesson_ui`]). Under an equation without one it offers the next best read
//! ([`note_fallback`], slice-2 spec 3.7): the closest reviewed note upstream as a link, an
//! assumption's rationale, or why there is none and where the notes are; never "No teaching
//! note" on a dashboard row. "Start here" walks the suggested order ([`notes::START_HERE`]):
//! each step opens its equation here and its note in the Explain panel (UX overhaul spec 5.5).
//! Every note named or drawn by id goes through [`reviewed_note`], so a note sent back to
//! Draft never shows.
//!
```

Replace:

```rust
use crate::engine::explain::markup::{Expr, Symbol};
use crate::engine::explain::note_text;
use crate::engine::explain::notes::{self, Change, Note, Review, START_HERE, covers};
use crate::engine::explain::{Design, Equation, TermKind, TermStyle};
```

with:

```rust
use crate::engine::explain::markup::{Expr, Symbol};
use crate::engine::explain::notes::{self, Note, Review, START_HERE, covers};
use crate::engine::explain::{Design, Equation, TermKind, TermStyle};
```

Replace:

```rust
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::picks::set_by_pick;
use crate::gui::corrections::correction_title;
```

with:

```rust
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::gui::corrections::correction_title;
```

Replace:

```rust
use crate::gui::corrections::correction_title;
use crate::gui::dashboard::{
    end_effect_out_of_range, greyed_by_end_effect, result_info, shown_text,
};
use crate::gui::diagrams::diagram_ui;
use crate::gui::explain::FULL_LESSON;
```

with:

```rust
use crate::gui::corrections::correction_title;
use crate::gui::dashboard::{result_info, shown_text};
use crate::gui::explain::FULL_LESSON;
```

Replace:

```rust
use crate::gui::format::value_with_unit;
use crate::gui::guidance::{Learn, result_text, result_unit, row_learn};
use crate::gui::input_ui::CHANGED_DOT;
```

with:

```rust
use crate::gui::format::value_with_unit;
use crate::gui::guidance::{Learn, result_unit, row_learn};
use crate::gui::input_ui::CHANGED_DOT;
```

Replace:

```rust
use crate::gui::input_ui::CHANGED_DOT;
use crate::gui::inputs::InputCatalogue;
use crate::gui::plain_text::label_for;
```

with:

```rust
use crate::gui::input_ui::CHANGED_DOT;
use crate::gui::notes_view::NoteLink;
use crate::gui::plain_text::label_for;
```

Replace:

```rust
use crate::gui::readouts::{ReadoutEvents, registry};
use crate::gui::sizing::{SizingState, held_by_auto};
use crate::gui::typeset::{
```

with:

```rust
use crate::gui::readouts::{ReadoutEvents, registry};
use crate::gui::typeset::{
```

Replace:

```rust
};
use crate::gui::wrapping_line;
use crate::{DesignInputs, DesignResults};
```

with:

```rust
};
use crate::{DesignInputs, DesignResults};
```

Replace:

```rust
pub const NEAREST_ASSUMPTION: &str = "Assumption behind this value:";

/// The heading of a note's watch-out lines.
pub const WATCH_OUT: &str = "Watch out";

/// The heading of a note's "What to change" entries (slice-2 spec 3.1).
pub const WHAT_TO_CHANGE: &str = "What to change";

/// The fold under a note that holds its sources, closed until clicked (slice-2 spec 3.1).
pub const WHERE_FROM: &str = "Where this comes from";

/// Why a change entry's input is plain text, not a link: Auto sets it (Torque -> Magnets).
pub const HELD_BY_AUTO: &str = "held by Auto";

/// Why a Raise entry's input is plain text: it is at the top of its slider.
pub const AT_TOP: &str = "at the top of its range";

/// Why a Lower entry's input is plain text: it is at the bottom of its slider.
pub const AT_BOTTOM: &str = "at the bottom of its range";

/// The start of the links at the head of a note to the equations it explains.
pub const EXPLAINS: &str = "Explains";

```

with:

```rust
pub const NEAREST_ASSUMPTION: &str = "Assumption behind this value:";

```

Replace:

```rust

/// What a note's change entries are drawn against: the design shown, its results, and why an
/// input is not free on it (`held`: Auto holds it or a part pick sets it, [`held_reason`];
/// `None` for a free input).
pub struct NoteView<'a> {
    pub inputs: &'a DesignInputs,
    pub results: &'a DesignResults,
    pub held: &'a dyn Fn(&str) -> Option<String>,
}

/// What a click in a teaching note, or in the lesson line under an equation, asks for
/// ([`Explorer::follow_note_link`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoteLink {
    /// Highlight this input's row, as a leaf term does.
    Input(&'static str),
    /// Show this reviewed note in the Explain panel (an equation's lesson, [`lesson_ui`], or
    /// its closest note, [`note_fallback`]).
    Note(&'static str),
}

/// Why the input at `path` is not free on the design `inputs` shown with `sizing`, for a note's
/// change entry ([`NoteView::held`]): Auto sets it in Torque -> Magnets ([`HELD_BY_AUTO`]), or
/// a part pick sets it in place of its own value ("set by the `<pick's label>` pick",
/// [`set_by_pick`]); `None` when it is free.
pub fn held_reason(sizing: &SizingState, inputs: &DesignInputs, path: &str) -> Option<String> {
    if held_by_auto(sizing, path) {
        return Some(HELD_BY_AUTO.to_owned());
    }
    set_by_pick(inputs, path).map(|pick| format!("set by the {} pick", label_for(pick)))
}

/// Why a change entry draws its input as plain text on `view`'s design: the input is not free
/// there (`view.held`), or one more slider step its way is past its slider ([`AT_TOP`],
/// [`AT_BOTTOM`]; the step a check guide's fix takes, `SliderRange::step_from`). `None` when the
/// input is a link: a choice, an entered number and an optional number left blank (it enters at
/// a result) are never at an end.
fn unlinked_reason(change: &Change, view: &NoteView<'_>) -> Option<String> {
    if let Some(reason) = (view.held)(change.input) {
        return Some(reason);
    }
    let up = change.step.up()?;
    let range = InputCatalogue::get().entry(change.input)?.meta.range?;
    let x = match view.inputs.get(change.input)? {
        Value::Num(x) if x.is_finite() => x,
        Value::Int(n) => n as f64,
        _ => return None,
    };
    if range.step_from(x, up).is_some() {
        return None;
    }
    Some(if up { AT_TOP } else { AT_BOTTOM }.to_owned())
}

/// What a `{path}` in a note shows on the design whose results are `results` (slice-2 spec
/// 3.2): the result's number with its unit, as the readouts show it
/// (`guidance::result_text`); `None`, drawn as [`note_text::NO_VALUE`], for a
/// path no result has, a number that is not finite, a text (a result that reads "n/a"
/// where it has no number, or a verdict: a live value is a number), and a result computed
/// from an out-of-range end effect (the pull-out and everything after it: the model gives no
/// prediction there, spec 5).
pub fn live_value(results: &DesignResults, path: &str) -> Option<String> {
    if end_effect_out_of_range(results).is_some()
        && (after_the_end_factor().contains(path) || greyed_by_end_effect(results, path))
    {
        return None;
    }
    match results.get(path)? {
        Value::Num(x) if x.is_finite() => Some(result_text(results, path)),
        Value::Int(_) => Some(result_text(results, path)),
        Value::Num(_) | Value::Text(_) | Value::None => None,
    }
}

/// Every result computed from the end-effect factor (the registry is fixed: computed once).
fn after_the_end_factor() -> &'static std::collections::BTreeSet<String> {
    static AFTER: std::sync::OnceLock<std::collections::BTreeSet<String>> =
        std::sync::OnceLock::new();
    AFTER.get_or_init(|| registry().downstream("model.f_end"))
}

/// A note's text as drawn on `view`'s design: each `{path}` replaced by its live value
/// ([`live_value`]), then through `glyph_safe`.
fn note_text_on(text: &str, view: &NoteView<'_>) -> String {
    glyph_safe(&note_text::render(text, |path| {
        live_value(view.results, path)
    }))
}

/// A teaching note (slice-2 spec 3.1): its title; what it is (its sentences) and its diagram;
/// "What to change", each entry "`<Step> <input's label>: <text>`", the label a link to the
/// input's row unless `unlinked_reason` gives a reason (then plain text with the reason);
/// "Watch out" and its lines; and the "Where this comes from" fold, closed until clicked,
/// holding its sources. The sentences, change texts and watch-outs show live values
/// (`note_text_on`); every text is drawn through `glyph_safe`. Returns the link clicked.
pub fn note_ui(ui: &mut egui::Ui, note: &Note, view: &NoteView<'_>) -> Option<NoteLink> {
    let mut clicked = None;
    ui.strong(glyph_safe(note.title));
    for sentence in note.sentences {
        ui.add(egui::Label::new(note_text_on(sentence, view)).wrap());
    }
    if let Some(diagram) = note.diagram {
        diagram_ui(ui, diagram);
    }
    if !note.change.is_empty() {
        ui.strong(WHAT_TO_CHANGE);
        for change in note.change {
            if change_ui(ui, change, view) {
                clicked = Some(NoteLink::Input(change.input));
            }
        }
    }
    if !note.watch_out.is_empty() {
        let warn = ui.visuals().warn_fg_color;
        ui.label(egui::RichText::new(WATCH_OUT).strong().color(warn));
        for line in note.watch_out {
            let text = egui::RichText::new(note_text_on(line, view)).color(warn);
            ui.add(egui::Label::new(text).wrap());
        }
    }
    egui::CollapsingHeader::new(WHERE_FROM)
        .id_salt(("magcoupling_note_sources", note.id))
        .default_open(false)
        .show(ui, |ui| {
            for source in note.sources {
                ui.add(
                    egui::Label::new(egui::RichText::new(glyph_safe(source)).small().weak()).wrap(),
                );
            }
        });
    clicked
}

/// One change entry on one wrapped line: its step word, its input's display label (a link, or
/// plain text and the reason), then ": " and its text. True when the link was clicked.
fn change_ui(ui: &mut egui::Ui, change: &Change, view: &NoteView<'_>) -> bool {
    let label = glyph_safe(label_for(change.input));
    let mut clicked = false;
    wrapping_line(ui, |ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label(format!("{} ", change.step.word()));
        match unlinked_reason(change, view) {
            Some(reason) => {
                ui.label(label);
                ui.weak(format!(" ({reason})"));
            }
            None => clicked = ui.link(label).clicked(),
        }
        ui.add(egui::Label::new(format!(": {}", note_text_on(change.text, view))).wrap());
    });
    clicked
}

/// The links at the head of a note in the Explain panel to the equations it explains (a
/// family's template is left out), after [`EXPLAINS`]; nothing for a note that explains none.
/// Returns the equation clicked, which the panel opens here.
pub fn explains_ui(ui: &mut egui::Ui, note: &Note) -> Option<&'static str> {
    let paths: Vec<&'static str> = note
        .equations
        .iter()
        .copied()
        .filter(|p| !p.contains('#') && registry().equation_for(p).is_some())
        .collect();
    let mut clicked = None;
    if !paths.is_empty() {
        wrapping_line(ui, |ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.strong(EXPLAINS);
            for path in paths {
                if ui
                    .link(crumb(path))
                    .on_hover_text(term_label(path))
                    .clicked()
                {
                    clicked = Some(path);
                }
            }
        });
    }
    clicked
}

/// The open equation: its label (and with `show_details` its cell), the equation large (a term
```

with:

```rust

/// The open equation: its label (and with `show_details` its cell), the equation large (a term
```

Replace:

```rust
    use crate::compute_all;
    use crate::engine::meta::ResultSet;
```

with:

```rust
    use crate::compute_all;
    use crate::engine::explain::note_text;
    use crate::engine::meta::ResultSet;
```

Replace:

```rust
    use crate::engine::meta::ResultSet;
    use crate::gui::format::format_value;
    use crate::gui::format::with_unit;
    use crate::gui::results_table::table_entries;
```

with:

```rust
    use crate::engine::meta::ResultSet;
    use crate::gui::notes_view::{
        NoteView, WATCH_OUT, WHAT_TO_CHANGE, WHERE_FROM, live_value, note_text_on,
    };
    use crate::gui::results_table::table_entries;
```

Replace:

```rust
    use crate::gui::results_table::table_entries;

    fn design() -> (DesignInputs, DesignResults) {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        (inputs, results)
    }

```

with:

```rust
    use crate::gui::results_table::table_entries;
    use crate::gui::test_support::{design, nothing_held};

```

Replace:

```rust

    /// The view of a design on which no input is held.
    fn nothing_held(_: &str) -> Option<String> {
        None
    }

    /// A note with a change entry of each kind, for the renderer's own tests (no shipped note
    /// has change entries before the note tasks rewrite them).
    const RENDER_NOTE: Note = Note {
        id: "test.render",
        title: "A note to draw",
        equations: &[],
        sentences: &["The first sentence of the note.", "The second sentence."],
        change: &[
            Change {
                input: "metal.face_gap_mm",
                step: notes::Step::Lower,
                aim: Some(("model.pullout_Nm", notes::Sense::Up)),
                text: "the entry on the face gap.",
            },
            Change {
                input: "coupling.op_temp_C",
                step: notes::Step::Raise,
                aim: Some(("model.pullout_Nm", notes::Sense::Down)),
                text: "the entry on the operating temperature.",
            },
            Change {
                input: "materials.parts.back_iron",
                step: notes::Step::Choose,
                aim: None,
                text: "the entry on the back iron.",
            },
        ],
        watch_out: &["The first watch-out.", "The second watch-out."],
        diagram: None,
        sources: &["The first source.", "The second source."],
        review: Review::Draft,
    };

    /// One frame of `note` drawn against `view` with `events`: what egui painted and the link a
    /// click asked for.
    fn note_frame(
        ctx: &egui::Context,
        note: &Note,
        view: &NoteView<'_>,
        events: Vec<egui::Event>,
    ) -> (egui::FullOutput, Option<NoteLink>) {
        use crate::gui::test_support::{SCREEN, sized_frame};
        let mut link = None;
        let output = sized_frame(ctx, SCREEN, events, |ui| {
            if let Some(clicked) = note_ui(ui, note, view) {
                link = Some(clicked);
            }
        });
        (output, link)
    }

    /// Clicks the first drawn text equal to `text` (move, press, release): the link the click
    /// asked for, and a frame drawn once a fold the click opened has settled.
    fn click_in_note(
        ctx: &egui::Context,
        note: &Note,
        view: &NoteView<'_>,
        text: &str,
    ) -> (Option<NoteLink>, egui::FullOutput) {
        use crate::gui::test_support::{primary_button, text_rect};
        let (output, _) = note_frame(ctx, note, view, Vec::new());
        // A text in a wrapped row is laid out from the row's start, so the left part of its rect
        // covers the widgets before it on that row: click its last character.
        let rect = text_rect(&output, text).unwrap_or_else(|| panic!("no text {text:?}"));
        let at = egui::pos2(rect.max.x - 2.0, rect.center().y);
        note_frame(ctx, note, view, vec![egui::Event::PointerMoved(at)]);
        note_frame(ctx, note, view, vec![primary_button(at, true)]);
        let (_, link) = note_frame(ctx, note, view, vec![primary_button(at, false)]);
        let mut output = note_frame(ctx, note, view, Vec::new()).0;
        for _ in 0..15 {
            output = note_frame(ctx, note, view, Vec::new()).0;
        }
        (link, output)
    }

    /// Asserts that `note`'s entry on `input` draws its label as plain text: a click on it asks
    /// for no link, and " (<reason>)" is drawn once. The entry on `still_linked` stays a link.
    fn assert_entry_plain_with_reason(
        ctx: &egui::Context,
        note: &Note,
        view: &NoteView<'_>,
        input: &str,
        reason: &str,
        still_linked: &'static str,
    ) {
        use crate::gui::test_support::drawn_count;
        let (link, output) = click_in_note(ctx, note, view, label_for(input));
        assert_eq!(link, None, "{input}: an unlinked input is no link");
        assert_eq!(drawn_count(&output, &format!(" ({reason})")), 1, "{input}");
        let (link, _) = click_in_note(ctx, note, view, label_for(still_linked));
        assert_eq!(
            link,
            Some(NoteLink::Input(still_linked)),
            "{still_linked} stays a link"
        );
    }

    /// The slider of the input at `path`.
    fn slider(path: &str) -> crate::engine::meta::SliderRange {
        InputCatalogue::get()
            .entry(path)
            .unwrap()
            .meta
            .range
            .unwrap()
    }

    /// The panel's `held` with Auto off: only a pick holds an input of `inputs`.
    fn auto_off(inputs: &DesignInputs) -> impl Fn(&str) -> Option<String> + '_ {
        move |path| held_reason(&SizingState::default(), inputs, path)
    }

    /// Why `change` draws its input as plain text on `inputs`, with `held` naming held inputs.
    fn unlinked_with(
        inputs: &DesignInputs,
        change: &Change,
        held: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        let results = compute_all(inputs);
        let view = NoteView {
            inputs,
            results: &results,
            held,
        };
        unlinked_reason(change, &view)
    }

    /// Why `change` draws its input as plain text on `inputs` with Auto off ([`auto_off`]).
    fn unlinked_on(inputs: &DesignInputs, change: &Change) -> Option<String> {
        unlinked_with(inputs, change, &auto_off(inputs))
    }

    #[test]
    fn a_note_draws_its_parts_in_order_with_its_sources_folded_until_clicked() {
        use crate::gui::test_support::{drawn_texts, text_color, text_rect};
        let ctx = egui::Context::default();
        let (inputs, results) = design();
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &nothing_held,
        };
        note_frame(&ctx, &RENDER_NOTE, &view, Vec::new());
        let (output, _) = note_frame(&ctx, &RENDER_NOTE, &view, Vec::new());
        let texts = drawn_texts(&output);
        let count = |text: &str| texts.iter().filter(|t| *t == text).count();
        let once = [RENDER_NOTE.title, WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM]
            .into_iter()
            .chain(RENDER_NOTE.sentences.iter().copied())
            .chain(RENDER_NOTE.watch_out.iter().copied());
        for text in once {
            assert_eq!(count(text), 1, "{text}");
        }
        for change in RENDER_NOTE.change {
            assert_eq!(
                count(&format!("{} ", change.step.word())),
                1,
                "{}",
                change.input
            );
            assert_eq!(count(label_for(change.input)), 1, "{}", change.input);
            assert_eq!(count(&format!(": {}", change.text)), 1, "{}", change.input);
        }
        for source in RENDER_NOTE.sources {
            assert_eq!(count(source), 0, "the fold starts closed: {source}");
        }
        // Top to bottom: what it is, what to change, watch out, where this comes from.
        let top = |text: &str| text_rect(&output, text).unwrap().min.y;
        let order = [
            RENDER_NOTE.sentences[1],
            WHAT_TO_CHANGE,
            label_for(RENDER_NOTE.change[2].input),
            WATCH_OUT,
            RENDER_NOTE.watch_out[1],
            WHERE_FROM,
        ];
        for pair in order.windows(2) {
            assert!(top(pair[0]) < top(pair[1]), "{pair:?}");
        }
        let warn = ctx.style().visuals.warn_fg_color;
        for line in RENDER_NOTE.watch_out {
            assert_eq!(text_color(&output, line), Some(warn), "{line}");
        }
        let (link, output) = click_in_note(&ctx, &RENDER_NOTE, &view, WHERE_FROM);
        assert_eq!(link, None, "the fold is no link to an input");
        let texts = drawn_texts(&output);
        for source in RENDER_NOTE.sources {
            assert!(texts.iter().any(|t| t == source), "{source}");
        }
    }

    #[test]
    fn a_free_input_s_label_is_a_link_to_its_row_and_the_text_is_not() {
        let ctx = egui::Context::default();
        let (inputs, results) = design();
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &nothing_held,
        };
        for change in RENDER_NOTE.change {
            let (link, _) = click_in_note(&ctx, &RENDER_NOTE, &view, label_for(change.input));
            assert_eq!(
                link,
                Some(NoteLink::Input(change.input)),
                "{}",
                change.input
            );
            let text = format!(": {}", change.text);
            let (link, _) = click_in_note(&ctx, &RENDER_NOTE, &view, &text);
            assert_eq!(link, None, "{text}");
        }
    }

    #[test]
    fn an_input_that_is_not_free_is_plain_text_with_the_reason() {
        let ctx = egui::Context::default();
        let (inputs, results) = design();
        let held_entry = &RENDER_NOTE.change[0];
        let held = |path: &str| (path == held_entry.input).then(|| HELD_BY_AUTO.to_owned());
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &held,
        };
        let free = RENDER_NOTE.change[1].input;
        assert_entry_plain_with_reason(
            &ctx,
            &RENDER_NOTE,
            &view,
            held_entry.input,
            HELD_BY_AUTO,
            free,
        );
    }

    #[test]
    fn an_entry_whose_input_cannot_move_its_way_names_why() {
        use crate::engine::explain::notes::Step;
        let entry = |input, step| Change {
            input,
            step,
            aim: None,
            text: "",
        };
        let reason =
            |inputs: &DesignInputs, change: &Change| unlinked_with(inputs, change, &nothing_held);
        let gap = "metal.face_gap_mm";
        let (lowest, highest) = (slider(gap).min, slider(gap).max);
        let mut inputs = DesignInputs::default();
        assert_eq!(reason(&inputs, &entry(gap, Step::Raise)), None);
        assert_eq!(reason(&inputs, &entry(gap, Step::Lower)), None);
        inputs.set(gap, Value::Num(highest)).unwrap();
        assert_eq!(
            reason(&inputs, &entry(gap, Step::Raise)).as_deref(),
            Some(AT_TOP)
        );
        assert_eq!(reason(&inputs, &entry(gap, Step::Lower)), None);
        inputs.set(gap, Value::Num(lowest)).unwrap();
        assert_eq!(
            reason(&inputs, &entry(gap, Step::Lower)).as_deref(),
            Some(AT_BOTTOM)
        );
        assert_eq!(reason(&inputs, &entry(gap, Step::Raise)), None);
        // A count at its end; a choice and a blank optional number are never at an end.
        let mut poles = DesignInputs::default();
        poles.coupling.npole = slider("coupling.npole").max as i64;
        assert_eq!(
            reason(&poles, &entry("coupling.npole", Step::Raise)).as_deref(),
            Some(AT_TOP)
        );
        let defaults = DesignInputs::default();
        assert_eq!(
            reason(&defaults, &entry("materials.parts.back_iron", Step::Choose)),
            None
        );
        assert_eq!(
            reason(
                &defaults,
                &entry("coupling.magnets.axial_length_mm", Step::Raise)
            ),
            None
        );
        // A held input names why it is held, whatever its slider says.
        let held = |_: &str| Some(HELD_BY_AUTO.to_owned());
        assert_eq!(
            unlinked_with(&inputs, &entry(gap, Step::Lower), &held).as_deref(),
            Some(HELD_BY_AUTO)
        );
    }

    #[test]
    fn held_reason_names_auto_and_the_pick_that_sets_an_input() {
        use crate::engine::explain::plain::plain_text_violations;
        use crate::engine::picks::BACK_IRON_PICK;
        use crate::engine::sizing::FreeVariable;
        use crate::gui::sizing::SizingMode;
        let inputs = DesignInputs::default();
        let off = SizingState::default();
        let auto = SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable: FreeVariable::AxialLength,
        };
        let length = FreeVariable::AxialLength.path();
        assert_eq!(held_reason(&off, &inputs, length), None);
        assert_eq!(
            held_reason(&auto, &inputs, length).as_deref(),
            Some(HELD_BY_AUTO)
        );
        assert_eq!(held_reason(&auto, &inputs, "metal.face_gap_mm"), None);
        let part_pick = format!(
            "set by the {} pick",
            label_for("coupling.magnets.part_inner")
        );
        assert_eq!(
            held_reason(&off, &inputs, "coupling.magnets.manual_inner_br_T"),
            Some(part_pick.clone())
        );
        let steel = "materials.steel.cte_per_C";
        assert_eq!(held_reason(&off, &inputs, steel), None);
        let mut picked = inputs.clone();
        picked.materials.parts.back_iron = 2;
        let steel_pick = format!("set by the {} pick", label_for(BACK_IRON_PICK));
        assert_eq!(held_reason(&off, &picked, steel), Some(steel_pick.clone()));
        for reason in [
            HELD_BY_AUTO.to_owned(),
            part_pick,
            steel_pick,
            AT_TOP.into(),
            AT_BOTTOM.into(),
        ] {
            assert_eq!(
                plain_text_violations(&reason),
                Vec::<&str>::new(),
                "{reason}"
            );
        }
    }

    #[test]
    fn the_outer_block_entries_name_the_part_pick_that_sets_them() {
        // Review Focus RF-4 (amendment A-25 T-4, the harmonics part): at the defaults the outer
        // library part sets the block's width and thickness, so the harmonics and back-iron
        // entries on them are plain text naming the pick; with custom dimensions (the Y30 probe
        // design, both parts blank) the same entries are links.
        use crate::engine::explain::probes::probe_designs;
        let entry = |id: &str, input: &str| {
            *notes::note(id)
                .and_then(|n| n.change.iter().find(|c| c.input == input))
                .unwrap_or_else(|| panic!("{id}: no entry on {input}"))
        };
        let width = entry("harmonics", "coupling.magnets.manual_outer_width_mm");
        let thickness = entry(
            "back_iron_factor",
            "coupling.magnets.manual_outer_thickness_mm",
        );
        let pick = format!(
            "set by the {} pick",
            label_for("coupling.magnets.part_outer")
        );
        let defaults = DesignInputs::default();
        assert_eq!(unlinked_on(&defaults, &width), Some(pick.clone()));
        assert_eq!(unlinked_on(&defaults, &thickness), Some(pick));
        let (_, custom) = probe_designs()
            .into_iter()
            .find(|(name, _)| *name == "Y30 ferrite, custom dimensions")
            .expect("the custom-dimensions probe design");
        assert_eq!(unlinked_on(&custom, &width), None);
        assert_eq!(unlinked_on(&custom, &thickness), None);
    }

    #[test]
    fn the_wall_note_s_design_flux_entry_names_the_pick_that_sets_it() {
        // Review Focus RF-4 (amendment A-25 T-4): 1018 brings its own design flux density, so the
        // wall note's entry on that input is plain text naming the back-iron pick; 12L14 has none
        // of its own and leaves the input in effect, so the same entry is a link.
        use crate::engine::material_library::BACK_IRON_CHOICES;
        use crate::engine::picks::BACK_IRON_PICK;
        assert_eq!(BACK_IRON_CHOICES[1], (2, "1018_hot_rolled"));
        assert_eq!(BACK_IRON_CHOICES[2], (3, "12L14_cold_drawn"));
        let entry = *notes::note("a5.low_saturation")
            .and_then(|n| {
                n.change
                    .iter()
                    .find(|c| c.input == "materials.steel.bsat_T")
            })
            .expect("the wall note has an entry on the design flux density");
        let reason = |back_iron: i64| {
            let mut inputs = DesignInputs::default();
            inputs.materials.parts.back_iron = back_iron;
            unlinked_on(&inputs, &entry)
        };
        let set = reason(2).expect("1018 sets the design flux density");
        assert!(set.starts_with("set by the"), "{set}");
        assert_eq!(
            set,
            format!("set by the {} pick", label_for(BACK_IRON_PICK))
        );
        assert_eq!(reason(3), None);
    }

    #[test]
    fn the_sleeve_entry_at_the_bottom_of_its_slider_is_plain_text_with_the_reason() {
        // Amendment A-25 T-7: at the sleeve's slider minimum one step thinner is past the slider,
        // so the sleeve note's entry draws its label as plain text with the reason and no link;
        // the liner entry beside it stays a link.
        let sleeve = "metal.sleeve_mm";
        let note = notes::note("a5.high_conductivity_sleeve_or_liner").unwrap();
        let entry = note
            .change
            .iter()
            .find(|c| c.input == sleeve)
            .expect("the sleeve note has an entry on the sleeve thickness");
        assert_eq!(entry.step, notes::Step::Lower);
        let mut inputs = DesignInputs::default();
        inputs.set(sleeve, Value::Num(slider(sleeve).min)).unwrap();
        let results = compute_all(&inputs);
        let held = auto_off(&inputs);
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &held,
        };
        let ctx = egui::Context::default();
        let liner = "metal.liner_mm";
        assert_entry_plain_with_reason(&ctx, note, &view, sleeve, AT_BOTTOM, liner);
    }

    #[test]
    fn a_live_value_reads_as_the_readouts_and_none_where_there_is_no_prediction() {
        use crate::engine::explain::probes::probe_designs;
        let designs = probe_designs();
        let default = compute_all(&designs[0].1);
        let pullout = "model.pullout_Nm";
        let unit = result_info(pullout).unwrap().meta.unit;
        assert_eq!(
            live_value(&default, pullout),
            Some(with_unit(
                format_value(&default.get(pullout).unwrap()),
                unit
            ))
        );
        // A text where a number belongs: the cold limit of NdFeB rings reads "n/a".
        let cold = "temperature.demag.cold_limit_C";
        assert!(matches!(default.get(cold), Some(Value::Text(_))));
        assert_eq!(live_value(&default, cold), None);
        // On the ferrite probe the cold limit is a number and the hot onsets are not finite.
        let ferrite = compute_all(&designs[3].1);
        let cold_unit = result_info(cold).unwrap().meta.unit;
        assert_eq!(
            live_value(&ferrite, cold),
            Some(with_unit(
                format_value(&ferrite.get(cold).unwrap()),
                cold_unit
            ))
        );
        let onset = "temperature.demag.onset_skipping_C";
        assert!(matches!(ferrite.get(onset), Some(Value::Num(x)) if !x.is_finite()));
        assert_eq!(live_value(&ferrite, onset), None);
        // A verdict, an input and a path no result has: none.
        assert_eq!(live_value(&default, "materials.cup_wall_check"), None);
        assert_eq!(live_value(&default, "metal.face_gap_mm"), None);
        assert_eq!(live_value(&default, "no.such.result"), None);
    }

    #[test]
    fn a_value_computed_from_an_out_of_range_pull_out_reads_no_prediction() {
        let results = compute_all(&crate::gui::test_support::short_magnets());
        assert!(crate::gui::dashboard::end_effect_out_of_range(&results).is_some());
        for path in [
            "model.pullout_Nm",
            "metal.torque_hot_low_Nm",
            "metal.cold_high_input_Nm",
            "model.gearbox_input_ripple_Nm",
        ] {
            assert_eq!(live_value(&results, path), None, "{path}");
        }
        for path in [
            "model.f_end",
            "model.pole_pitch_mm",
            "model.active_length_mm",
        ] {
            assert!(live_value(&results, path).is_some(), "{path}");
        }
    }

    /// A note with live values in a sentence, a change entry and a watch-out.
    const LIVE_NOTE: Note = Note {
        id: "test.live",
        title: "A note with live values",
        equations: &[],
        sentences: &[
            "The first sentence has none.",
            "Here the pull-out is {model.pullout_Nm}.",
        ],
        change: &[Change {
            input: "metal.face_gap_mm",
            step: notes::Step::Lower,
            aim: Some(("model.pullout_Nm", notes::Sense::Up)),
            text: "the skipping onset is {temperature.demag.onset_skipping_C} here.",
        }],
        watch_out: &["The end factor is {model.f_end} on this design."],
        diagram: None,
        sources: &["A source."],
        review: Review::Draft,
    };

    #[test]
    fn a_note_draws_its_live_values_from_the_design_shown() {
        use crate::engine::explain::note_text::NO_VALUE;
        use crate::engine::explain::probes::probe_designs;
        use crate::gui::test_support::drawn_texts;
        let ctx = egui::Context::default();
        let designs = probe_designs();
        for (name, inputs) in [&designs[0], &designs[3]] {
            let results = compute_all(inputs);
            let view = NoteView {
                inputs,
                results: &results,
                held: &nothing_held,
            };
            note_frame(&ctx, &LIVE_NOTE, &view, Vec::new());
            let (output, _) = note_frame(&ctx, &LIVE_NOTE, &view, Vec::new());
            let texts = drawn_texts(&output);
            let value = |path: &str| live_value(&results, path).unwrap_or(NO_VALUE.to_owned());
            for want in [
                LIVE_NOTE.sentences[0].to_owned(),
                format!("Here the pull-out is {}.", value("model.pullout_Nm")),
                format!(
                    ": the skipping onset is {} here.",
                    value("temperature.demag.onset_skipping_C")
                ),
                format!("The end factor is {} on this design.", value("model.f_end")),
            ] {
                assert!(texts.contains(&want), "{name}: {want:?} in {texts:?}");
            }
        }
        // The ferrite rings' hot onset has no prediction; the NdFeB default's has one.
        let onset = "temperature.demag.onset_skipping_C";
        assert!(live_value(&compute_all(&designs[0].1), onset).is_some());
        assert_eq!(live_value(&compute_all(&designs[3].1), onset), None);
    }

    /// BL-064 (spec 6.1 and 7.11): the shaft-joint note's sentence 4 as the panel draws it: the
    /// temperature the inner hub's fit would need, beside the wheel shaft's live shrink, against
    /// the allowed temperature. At the defaults; at library N42 parts in both rings, where the
    /// inner ring's single-ring onset less the cure margin binds (owner R-15; the guide tests'
    /// `breaking(n42_parts)`); with mixed rings (inner N42M, outer N35; the guide tests'
    /// `breaking(mixed_rings)`), where the outer ring governs and the cap is still the inner
    /// ring's own onset less the margin (the owner's ring ruling); on an uncooled wheel shaft,
    /// whose shrink reads 0; and on a 40 °C one, its slider's top, where the shaft grows (the
    /// render the physics review decides on).
    #[test]
    fn the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink() {
        use crate::engine::explain::note_text::render;
        use crate::engine::temperature::RING_OUTER;
        use crate::gui::test_support::{mixed_rings, n42_parts, passing_design};
        let sentence = notes::note("shaft_joint")
            .expect("the shaft-joint note")
            .sentences[4];
        let drawn = |inputs: &DesignInputs| {
            let results = compute_all(inputs);
            render(sentence, |path: &str| live_value(&results, path))
        };
        let wheel_shaft_at = |celsius: f64| {
            let mut inputs = DesignInputs::default();
            inputs.shaft_joint.wheel_shaft_fit_C = celsius;
            inputs
        };
        let mut n42 = passing_design();
        n42_parts(&mut n42);
        // Mixed rings: the outer ring governs the magnets' limit, so its single-ring onset is not
        // the inner ring's own, which the bonded inner hub's cap reads.
        let mut mixed = passing_design();
        mixed_rings(&mut mixed);
        let demag = compute_all(&mixed).temperature.demag;
        assert_eq!(demag.demag_ring, RING_OUTER);
        assert_ne!(demag.onset_single_ring_C, demag.inner_onset_single_ring_C);
        for (inputs, words) in [
            (
                DesignInputs::default(),
                "would need 93.27 °C on the wheel shaft shrunk 18.99 µm; allowed: 100.0 °C,",
            ),
            (n42, "allowed: 59.83 °C,"),
            (mixed, "allowed: 81.35 °C,"),
            (
                wheel_shaft_at(20.0),
                "would need 247.6 °C on the wheel shaft shrunk 0 µm;",
            ),
            (
                wheel_shaft_at(40.0),
                "would need 266.3 °C on the wheel shaft shrunk -2.300 µm;",
            ),
        ] {
            let text = drawn(&inputs);
            assert!(text.contains(words), "{words:?} in {text:?}");
        }
    }

    #[test]
    fn every_note_draws_at_every_probe_design_with_glyphs_and_no_braces() {
        use crate::engine::explain::note_text::rendered_texts;
        use crate::engine::explain::probes::probe_designs;
        use crate::gui::test_support::{assert_glyphs, glyph_context};
        let ctx = glyph_context();
        for (name, inputs) in probe_designs() {
            let results = compute_all(&inputs);
            let view = NoteView {
                inputs: &inputs,
                results: &results,
                held: &nothing_held,
            };
            for note in notes::NOTES {
                for text in rendered_texts(note) {
                    let drawn = note_text_on(text, &view);
                    assert!(
                        !drawn.contains(['{', '}']),
                        "{} at {name}: {drawn}",
                        note.id
                    );
                    assert_glyphs(&ctx, &drawn, note.id);
                }
            }
        }
    }

    /// The per-note physics review's sheet (slice-2 spec section 4): each note, or each named
    /// in `NOTES_FOR_REVIEW` (ids, comma-separated), its title and sources once, then at each
    /// probe design its sentences, change entries and watch-outs as the panel draws them; a
    /// change entry with its input's display label, the reason it is not a link there, and
    /// its aim's value before and after one step. A text the parser refuses or the plain-words
    /// guard flags is marked "<<<". Written to `NOTES_REVIEW_OUT` (a path from `magcoupling-rs/`;
    /// default `target/notes-for-review.txt` there).
    #[test]
    #[ignore = "a review tool, not a check: NOTES_FOR_REVIEW=id1,id2 NOTES_REVIEW_OUT=<file> cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture"]
    fn notes_for_review() {
        use std::fmt::Write as _;

        use crate::engine::explain::note_text::segments;
        use crate::engine::explain::plain::plain_text_violations;
        use crate::engine::explain::probes::{probe_designs, step_input};
        let wanted: Vec<String> = std::env::var("NOTES_FOR_REVIEW")
            .map(|ids| {
                ids.split(',')
                    .map(|id| id.trim().to_owned())
                    .filter(|id| !id.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        let unknown: Vec<&String> = wanted
            .iter()
            .filter(|id| notes::note(id).is_none())
            .collect();
        assert!(
            unknown.is_empty(),
            "NOTES_FOR_REVIEW names no note: {unknown:?}"
        );
        let chosen: Vec<&Note> = notes::NOTES
            .iter()
            .filter(|n| wanted.is_empty() || wanted.iter().any(|id| id == n.id))
            .collect();
        let designs: Vec<(&str, DesignInputs, DesignResults)> = probe_designs()
            .into_iter()
            .map(|(name, inputs)| {
                let results = compute_all(&inputs);
                (name, inputs, results)
            })
            .collect();
        // A raw text the parser refuses, or a drawn text the guard flags, is marked.
        let marks = |raw: &str, drawn: &str| {
            let mut found = Vec::new();
            if let Err(fault) = segments(raw) {
                found.push(fault);
            }
            let banned = plain_text_violations(drawn);
            if !banned.is_empty() {
                found.push(format!("plain words: {banned:?}"));
            }
            if found.is_empty() {
                String::new()
            } else {
                format!("   <<< {}", found.join("; "))
            }
        };
        let sizing = SizingState::default();
        let mut sheet = String::new();
        for note in &chosen {
            let status = match note.review {
                Review::Draft => "Draft",
                Review::Reviewed { .. } => "Reviewed",
            };
            writeln!(sheet, "==== {} ({status})", note.id).unwrap();
            writeln!(
                sheet,
                "Title: {}{}",
                note.title,
                marks(note.title, note.title)
            )
            .unwrap();
            writeln!(sheet, "{WHERE_FROM}:").unwrap();
            for source in note.sources {
                writeln!(sheet, "  - {source}{}", marks(source, source)).unwrap();
            }
            for (name, inputs, results) in &designs {
                let held = |path: &str| held_reason(&sizing, inputs, path);
                let view = NoteView {
                    inputs,
                    results,
                    held: &held,
                };
                writeln!(sheet, "\n-- at the {name} design").unwrap();
                writeln!(sheet, "What it is:").unwrap();
                for (i, sentence) in note.sentences.iter().enumerate() {
                    let drawn = note_text_on(sentence, &view);
                    writeln!(sheet, "  {}. {drawn}{}", i + 1, marks(sentence, &drawn)).unwrap();
                }
                if !note.change.is_empty() {
                    writeln!(sheet, "{WHAT_TO_CHANGE}:").unwrap();
                }
                for change in note.change {
                    let reason =
                        unlinked_reason(change, &view).map_or(String::new(), |r| format!(" ({r})"));
                    let drawn = note_text_on(change.text, &view);
                    writeln!(
                        sheet,
                        "  - {} {}{reason}: {drawn}{}",
                        change.step.word(),
                        label_for(change.input),
                        marks(change.text, &drawn)
                    )
                    .unwrap();
                    let Some((aim, sense)) = change.aim else {
                        continue;
                    };
                    let outcome = match step_input(inputs, change.input, change.step) {
                        Err(why) => format!("not stepped here: {why:?}"),
                        Ok(stepped) => match (results.get(aim), compute_all(&stepped).get(aim)) {
                            (Some(Value::Num(before)), Some(Value::Num(after)))
                                if before.is_finite() && after.is_finite() =>
                            {
                                let verdict = if sense.holds(before, after) {
                                    "moves as it says"
                                } else {
                                    "DOES NOT MOVE AS IT SAYS"
                                };
                                format!("{before} -> {after}: {verdict}")
                            }
                            (before, after) => {
                                format!("no numbers to compare: {before:?} -> {after:?}")
                            }
                        },
                    };
                    writeln!(sheet, "      aim {aim} {sense:?}, one step: {outcome}").unwrap();
                }
                if !note.watch_out.is_empty() {
                    writeln!(sheet, "{WATCH_OUT}:").unwrap();
                }
                for line in note.watch_out {
                    let drawn = note_text_on(line, &view);
                    writeln!(sheet, "  - {drawn}{}", marks(line, &drawn)).unwrap();
                }
            }
            writeln!(sheet).unwrap();
        }
        let out = std::env::var("NOTES_REVIEW_OUT").map_or_else(
            |_| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/notes-for-review.txt")
            },
            std::path::PathBuf::from,
        );
        if let Some(dir) = out.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).expect("the sheet's folder is made");
        }
        std::fs::write(&out, &sheet).expect("the sheet is written");
        println!(
            "{} notes at {} designs: {}",
            chosen.len(),
            designs.len(),
            out.display()
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

    #[test]
    fn a_wrapped_change_entry_stays_inside_a_narrow_panel_and_its_label_links() {
        // Amendment A-25 T-6 (spec section 5, a narrow window): the pull-out note drawn alone in
        // a panel 400 points wide. Every change entry's input label is drawn inside the
        // panel, and a click on the last character of the inner magnet part's label, laid out in
        // a wrapped row, asks for that input's row.
        use crate::gui::test_support::{last_characters, primary_button, sized_frame, text_rects};
        let size = egui::vec2(400.0, 800.0);
        let note = reviewed_note("pullout_angle").expect("the pull-out note is reviewed");
        let (inputs, results) = design();
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &nothing_held,
        };
        let ctx = egui::Context::default();
        let frame = |events: Vec<egui::Event>| {
            let mut link = None;
            let output = sized_frame(&ctx, size, events, |ui| {
                link = note_ui(ui, note, &view);
            });
            (output, link)
        };
        frame(Vec::new());
        let (output, _) = frame(Vec::new());
        let panel = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        for change in note.change {
            let label = glyph_safe(label_for(change.input));
            let rects = text_rects(&output, &label);
            assert_eq!(rects.len(), 1, "{label:?} is drawn once");
            assert!(panel.contains_rect(rects[0]), "{label:?} at {:?}", rects[0]);
        }
        let part = "coupling.magnets.part_inner";
        let at = last_characters(&output, &glyph_safe(label_for(part)))[0];
        frame(vec![egui::Event::PointerMoved(at)]);
        frame(vec![primary_button(at, true)]);
        let (_, link) = frame(vec![primary_button(at, false)]);
        assert_eq!(link, Some(NoteLink::Input(part)));
    }

    // The sign-off sheet's text before slice 2: T4's snapshot and its reader.
```

with:

```rust

    // The sign-off sheet's text before slice 2: T4's snapshot and its reader.
```

**`magcoupling-rs/src/gui/guidance.rs`**

Replace:

```rust
    ) {
        use crate::gui::explorer::WHAT_TO_CHANGE;
        let all = within_minimum_diameter(&all_fixes(inputs, results, paused), inputs, None);
```

with:

```rust
    ) {
        use crate::gui::notes_view::WHAT_TO_CHANGE;
        let all = within_minimum_diameter(&all_fixes(inputs, results, paused), inputs, None);
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
use crate::gui::explain::{Asked, LEARN, Selection, explain_ui, result_selection};
use crate::gui::explorer::{Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights};
use crate::gui::follow_focus;
```

with:

```rust
use crate::gui::explain::{Asked, LEARN, Selection, explain_ui, result_selection};
use crate::gui::explorer::{Explorer, FOCUS_WIDTH, explorer_ui, panel_heights};
use crate::gui::follow_focus;
```

Replace:

```rust
use crate::gui::look;
use crate::gui::plots::{PLOTS, PlotKind, plot_ui};
```

with:

```rust
use crate::gui::look;
use crate::gui::notes_view::held_reason;
use crate::gui::plots::{PLOTS, PlotKind, plot_ui};
```

Replace:

```rust
        use crate::engine::explain::notes::note;
        use crate::gui::explorer::{HELD_BY_AUTO, live_value};
        let id = "torque_from_shear";
```

with:

```rust
        use crate::engine::explain::notes::note;
        use crate::gui::notes_view::{HELD_BY_AUTO, live_value};
        let id = "torque_from_shear";
```

Replace:

```rust
        use crate::engine::explain::notes::NOTES;
        use crate::gui::explorer::WHERE_FROM;
        use crate::gui::typeset::glyph_safe;
```

with:

```rust
        use crate::engine::explain::notes::NOTES;
        use crate::gui::notes_view::WHERE_FROM;
        use crate::gui::typeset::glyph_safe;
```

**`magcoupling-rs/src/gui/test_support.rs`**

Replace:

```rust
        .unwrap_or_else(|| panic!("no workflow group {id}"))
}
```

with:

```rust
        .unwrap_or_else(|| panic!("no workflow group {id}"))
}

/// The default design and its results.
pub(crate) fn design() -> (crate::DesignInputs, crate::DesignResults) {
    let inputs = crate::DesignInputs::default();
    let results = crate::compute_all(&inputs);
    (inputs, results)
}

/// The view of a design on which no input is held.
pub(crate) fn nothing_held(_: &str) -> Option<String> {
    None
}
```

- [ ] **Step 3: Run the checks** of "Before each commit", and the two runs of Step 1 again. Expected: `test result: ok. 1151 passed; 0 failed; 6 ignored` for the lib tests with the app feature (as many as on the base: nothing lost, nothing added), `test result: ok. 14 passed; 0 failed; 1 ignored; 0 measured; 1142 filtered out` and `test result: ok. 30 passed; 0 failed; 1 ignored; 0 measured; 1126 filtered out` (the 14 tests and the one review tool that moved, out of the explorer's count); both clippy runs clean; `cargo fmt` changes no text. Then `git add -N magcoupling-rs/src/gui/notes_view.rs` and `git diff --color-moved=plain --stat`: eight files, the explorer's lines out and the new module's in.

- [ ] **Step 4: Commit** the eight files: `refactor(magcoupler): the teaching-note drawing moves from the Equation panel's module to gui/notes_view.rs, unchanged (BL-088, BL-112 T1)`.

### Task 2: The Equation panel's empty text goes (BL-089)

**Model:** `sonnet`.

Every way the app opens the panel sets a trail first (a value clicked, an Equation link followed, a Start here step), so `EMPTY_TEXT` is drawn in no state the app reaches. It goes, with its test's lines and its row in the strings sheet; a panel open with no trail draws its header row and nothing under it (decision D-108). No text replaces it.

**Files:**
- Modify: `magcoupling-rs/src/gui/explorer.rs`, `magcoupling-rs/src/gui/guidance.rs`

**Interfaces:**
- Removes: `gui::explorer::EMPTY_TEXT`.

- [ ] **Step 1 (red): the empty panel's test.** The blocks add `a_panel_open_with_no_trail_draws_its_header_row_and_nothing_else` (an `Explorer` open with no trail, drawn by `explorer_ui`: the texts drawn are the heading, Close and Start here, in that order).

**`magcoupling-rs/src/gui/explorer.rs`**

Replace:

```rust
    #[test]
    fn the_empty_text_and_explain_s_own_lines_name_the_buttons_in_plain_words() {
```

with:

```rust
    #[test]
    fn a_panel_open_with_no_trail_draws_its_header_row_and_nothing_else() {
        // BL-089: every way the app opens the panel sets a trail first (a value clicked, an
        // Equation link followed, a Start here step), so a panel open with none is no state a
        // person reaches. It draws its header row, and no text under it.
        use crate::gui::test_support::{SCREEN, drawn_texts, sized_frame};
        let (inputs, results) = design();
        let mut explorer = Explorer {
            open: true,
            ..Explorer::default()
        };
        let ctx = egui::Context::default();
        let output = sized_frame(&ctx, SCREEN, Vec::new(), |ui| {
            explorer_ui(
                ui,
                &mut explorer,
                &inputs,
                &results,
                &inputs,
                false,
                clamps::PAUSED,
            );
        });
        assert_eq!(
            drawn_texts(&output),
            [EQUATION_PANEL, CLOSE, START_HERE_BUTTON]
        );
    }

    #[test]
    fn the_empty_text_and_explain_s_own_lines_name_the_buttons_in_plain_words() {
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::explorer:: 2>&1 | tail -n 12
  ```

  Expected: 1 failed: `a_panel_open_with_no_trail_draws_its_header_row_and_nothing_else`. `a_panel_open_with_no_trail_draws_its_header_row_and_nothing_else` panics with `` assertion `left == right` failed ``.

- [ ] **Step 3 (green): the text goes.** The blocks take out `EMPTY_TEXT` and its draw (the scroll area returns with nothing drawn), take its two asserts and its row out of `the_empty_text_and_explain_s_own_lines_name_the_buttons_in_plain_words`, renamed `explain_s_own_lines_name_the_buttons_in_plain_words` (every other line and assert stays), and take its row out of `guidance_strings_for_review` (an ignored test that prints the strings for the physics review).

**`magcoupling-rs/src/gui/explorer.rs`**

Replace:

```rust
pub const CLOSE: &str = "Close";

/// What the panel says before anything is opened (spec 3.7, TC-29b): how to open an equation,
/// and the walk through the teaching notes, by its button's name (a test checks). Since the
/// panel's toggle went (part 2a), every way the app opens the panel sets a trail first, so this
/// is only a fallback for a panel opened with none; BL-089 decides whether it goes.
pub const EMPTY_TEXT: &str = "Hover a value to see its equation; the Equation link in its entry opens it here, then click a term to follow it. Start here walks the main teaching notes in a suggested order.";

```

with:

```rust
pub const CLOSE: &str = "Close";

```

Replace:

```rust
        .show(ui, |ui| {
            let Some(path) = explorer.current().map(str::to_owned) else {
```

with:

```rust
        .show(ui, |ui| {
            // Every way the app opens the panel sets a trail first: with none there is nothing
            // to draw (BL-089).
            let Some(path) = explorer.current().map(str::to_owned) else {
```

Replace:

```rust
            let Some(path) = explorer.current().map(str::to_owned) else {
                ui.weak(EMPTY_TEXT);
                return;
```

with:

```rust
            let Some(path) = explorer.current().map(str::to_owned) else {
                return;
```

Replace:

```rust
    #[test]
    fn the_empty_text_and_explain_s_own_lines_name_the_buttons_in_plain_words() {
        // TC-29b: the empty panel names the walk through the notes by its button's own name
        // (the Explain toggle it also named went with decision C-11); the fallback lines read
        // as plain words in egui's default fonts.
        assert!(EMPTY_TEXT.contains(START_HERE_BUTTON));
        assert!(!EMPTY_TEXT.contains("Explain"), "{EMPTY_TEXT}");
        assert!(START_HERE_HINT.contains(START_HERE_BUTTON));
```

with:

```rust
    #[test]
    fn explain_s_own_lines_name_the_buttons_in_plain_words() {
        // TC-29b: the fallback lines name the walk through the notes by its button's own name
        // and read as plain words in egui's default fonts. (The empty panel's text, which named
        // it too, went with BL-089: no way of opening the panel leaves it empty.)
        assert!(START_HERE_HINT.contains(START_HERE_BUTTON));
```

Replace:

```rust
        );
        for text in [
            EMPTY_TEXT,
            NO_NOTE,
            START_HERE_HINT,
            NEAREST_NOTE,
            NEAREST_ASSUMPTION,
        ] {
            assert_eq!(
```

with:

```rust
        );
        for text in [NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION] {
            assert_eq!(
```

**`magcoupling-rs/src/gui/guidance.rs`**

Replace:

```rust
        for (tag, text) in [
            ("explorer/empty text", ex::EMPTY_TEXT),
            ("explorer/no note", ex::NO_NOTE),
```

with:

```rust
        for (tag, text) in [
            ("explorer/no note", ex::NO_NOTE),
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1152 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text; `git grep -n EMPTY_TEXT -- magcoupling-rs` prints nothing.

- [ ] **Step 5: Commit** the two files: `fix(magcoupler): the Equation panel's empty text goes; a panel open with no trail draws its header row only (BL-089, BL-112 T2)`.

### Task 3: Every text of the Equation panel but its formula wraps inside it (BL-086)

**Model:** `sonnet`.

A term's name and what it is share the term grid's last column, which egui gives the width the panel has left, and wrap in it; the terms' sideways scroll area goes; a term's value wraps past a third of the list's width, laid out as a galley at that width, since egui lays a grid cell before the last out in the width its column had in the frame before (decision D-110). The walk bar's step line wraps, the buttons after it (decision D-111). The formula keeps its sideways scroll area (decision D-109). The app test of the walk is widened from the lesson line to every text of the panel but the formula, and holds the panel inside the centre region; the panel test that held the terms' sideways scroll now holds the wrap; a new panel test opens every equation the registry holds in the app's font (the plan review's UXT-1).

**Files:**
- Modify: `magcoupling-rs/src/app.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/test_support.rs` (red); `magcoupling-rs/src/gui/explorer.rs` (green)

**Interfaces:**
- Produces: `#[cfg(test)] pub(crate) fn gui::test_support::cut_texts_in(&FullOutput, Rect, impl Fn(Rect) -> bool) -> Vec<String>`, `cut_texts` with a filter on the clip rect; `cut_texts` is now `cut_texts_in(.., |_| true)`.
- Produces: `pub const gui::explorer::TERM_VALUE_SHARE: f32` (1/3, the share of the term list's width a term's value takes at most).
- Changes: `equation_body`'s term grid is `Grid::num_columns(4)` with no scroll area around it; a term's value is a label of a galley laid out at `TERM_VALUE_SHARE` of the list's width, wrapping past it; the walk bar's step line is a wrapping label.

- [ ] **Step 1 (red): the tests.** The blocks add `cut_texts_in` to `test_support.rs`; rename `every_start_here_step_s_lesson_line_is_drawn_whole_in_the_app_s_own_font` to `every_text_of_the_equation_panel_is_whole_on_each_start_here_step_in_the_app_s_own_font` and widen it (each of the 14 steps at 1024 x 700, 1280 x 800 and 1920 x 1080, in the app's theme and font: the panel's left and right edges inside the centre region, and no text clipped inside the panel cut but those of the formula's area, found by the formula's " = "); and rewrite `the_term_list_scrolls_sideways_in_a_narrow_centre_region` as `the_term_list_wraps_inside_a_narrow_centre_region` (at 1000 x 800 on σ, the assumption's tag is more than one row tall and whole inside its clip, with no scroll; the design unchanged); and add `every_text_of_the_equation_panel_is_whole_on_every_equation_in_the_app_s_own_font` (`#[cfg(feature = "app")]`, in `panel.rs`: every equation the registry holds, opened in the app's theme and font in a window 1024 x 3000, as narrow as the smallest and tall enough for the panel's foot, on the starting design and on the example design with the details on: the panel's edges inside the centre region, its Used by line drawn wherever it has one, and no text of it cut but the formula's).

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
    #[test]
    fn every_start_here_step_s_lesson_line_is_drawn_whole_in_the_app_s_own_font() {
        // Plan review UXP-1 (BL-085): the lesson line under the open equation ("Full lesson:"
        // and the note's title, a link) ran past the Equation panel's edge at 1024 x 700 on
        // the long titles (steps 3, 7 and 12). Each step of the walk, at the three window
        // sizes, in the app's theme and font: no text of the line is cut.
        use crate::engine::explain::notes::START_HERE;
```

with:

```rust
    #[test]
    fn every_text_of_the_equation_panel_is_whole_on_each_start_here_step_in_the_app_s_own_font() {
        // Plan review UXP-1 (BL-085): the lesson line under the open equation ran past the
        // Equation panel's edge at 1024 x 700 on the long titles. BL-086: so did a term's name
        // and its tag (at 1280 x 800 too) and the walk bar's step line, which also pushed the
        // panel over the right column. Each step of the walk, at the three window sizes, in the
        // app's theme and font: the panel stays inside the centre region, and no text of it is
        // cut but the formula's, which scrolls sideways in an area of its own (decision D-109:
        // an equation does not wrap).
        use crate::engine::explain::notes::START_HERE;
```

Replace:

```rust
        use crate::engine::explain::notes::START_HERE;
        use crate::gui::explain::FULL_LESSON;
        use crate::gui::explorer::{NEXT, START_HERE_BUTTON, reviewed_note};
        use crate::gui::test_support::{cut_texts, primary_button, text_center};
        use crate::gui::typeset::glyph_safe;
        for size in [WINDOWS[1], WINDOWS[0], egui::vec2(1920.0, 1080.0)] {
```

with:

```rust
        use crate::engine::explain::notes::START_HERE;
        use crate::gui::explorer::{NEXT, START_HERE_BUTTON};
        use crate::gui::test_support::{cut_texts_in, primary_button, text_center, text_shapes};
        for size in [WINDOWS[1], WINDOWS[0], egui::vec2(1920.0, 1080.0)] {
```

Replace:

```rust
            let mut output = click(&mut app, START_HERE_BUTTON);
            for (step, &(id, _)) in START_HERE.iter().enumerate() {
                if step > 0 {
```

with:

```rust
            let mut output = click(&mut app, START_HERE_BUTTON);
            for step in 0..START_HERE.len() {
                if step > 0 {
```

Replace:

```rust
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
```

with:

```rust
                }
                let at = format!("{size:?}, step {}", step + 1);
                let panel = side_rect(&ctx, "magcoupling_equation_panel");
                let centre = side_rect(&ctx, "magcoupling_inputs").right()
                    ..=side_rect(&ctx, "magcoupling_dashboard").left();
                assert!(
                    centre.contains(&panel.left()) && centre.contains(&panel.right()),
                    "{at}: the panel {panel:?} runs out of the centre region {centre:?}"
                );
                // The formula's own area: where the panel draws its " = ".
                let inside = |clip: egui::Rect| panel.expand(1.0).contains_rect(clip);
                let formula = text_shapes(&output, " = ")
                    .map(|(clip, _)| clip)
                    .find(|&clip| inside(clip))
                    .unwrap_or_else(|| panic!("{at}: no formula"));
                let cut = cut_texts_in(&output, screen, |clip| inside(clip) && clip != formula);
                assert!(cut.is_empty(), "{at}: cut: {cut:#?}");
            }
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    #[test]
    fn the_term_list_scrolls_sideways_in_a_narrow_centre_region() {
        // At 1000 points the centre region is about 380 wide: σ's term list (swatch, symbol,
        // value, label, tag) is wider, so its last column, the tag, starts cut off or out of
        // view. The list scrolls sideways (as the equation above it does), so the tag can be
        // read in full.
        use crate::gui::explorer::ASSUMPTION_TAG;
```

with:

```rust
    #[test]
    fn the_term_list_wraps_inside_a_narrow_centre_region() {
        // At 1000 points the centre region is about 380 wide: a row of σ's term list (swatch,
        // symbol, value, name, tag) is wider. The name and the tag share the row's last column
        // and wrap in it (BL-086), so the tag is read in full where it is, with no sideways
        // scroll: here it takes two rows.
        use crate::gui::explorer::ASSUMPTION_TAG;
```

Replace:

```rust
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert!(
            text_rect(&output, ASSUMPTION_TAG)
                .is_none_or(|tag| tag.right() > text_clip_rect(&output, ASSUMPTION_TAG).right()),
            "the tag fits without scrolling: the test would prove nothing"
        );
        // The pointer on the start of the set's value, in view at the left of the list.
        let at = text_rect(&output, "5: 1, 3, 5 (default)")
            .expect("the set's value in the term list")
            .left_center()
            + egui::vec2(4.0, 0.0);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(vec![egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(-5000.0, 0.0),
            modifiers: egui::Modifiers::NONE,
        }]);
        // egui spreads a wheel step over several frames.
        let mut output = harness.frame(Vec::new());
        for _ in 0..20 {
            output = harness.frame(Vec::new());
        }
        let tag = text_rect(&output, ASSUMPTION_TAG).expect("the tag is drawn");
        let clip = text_clip_rect(&output, ASSUMPTION_TAG);
        assert!(clip.contains_rect(tag), "{tag:?} cut by {clip:?}");
```

with:

```rust
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let value = text_rect(&output, "5: 1, 3, 5 (default)").expect("the set's value");
        let tag = text_rect(&output, ASSUMPTION_TAG).expect("the tag is drawn");
        assert!(
            tag.height() > 1.5 * value.height(),
            "the tag fits on one row: the test would prove nothing"
        );
        let clip = text_clip_rect(&output, ASSUMPTION_TAG);
        assert!(clip.contains_rect(tag), "{tag:?} cut by {clip:?}");
```

Replace:

```rust
    }

    #[test]
    fn closing_the_equation_panel_clears_the_input_row_s_frame() {
```

with:

```rust
    }

    #[cfg(feature = "app")]
    #[test]
    fn every_text_of_the_equation_panel_is_whole_on_every_equation_in_the_app_s_own_font() {
        // BL-086, the plan review's UXT-1: a term's value has a column of its own, and a long
        // text value ("never: steady state stays below the limit") laid out on one line left
        // the name and tag about 20 points, where their words stood one to a row and ran past
        // the panel's edge, the panel past the centre region. Every equation the registry
        // holds, opened in the app's theme and font in a window as narrow as the smallest and
        // tall enough for the panel to show it to its foot, on the starting design and on the
        // example design with the spreadsheet details on: the panel stays inside the centre
        // region, and no text of it is cut but the formula's (decision D-109). (Before the
        // wrap the term list scrolled sideways and egui painted nothing past its view, which
        // no test of drawn text sees: there the narrow region's test holds the tag.)
        use crate::gui::explorer::{USED_BY, used_by_shown};
        use crate::gui::test_support::cut_texts_in;
        let size = egui::vec2(SMALLEST.x, 3000.0);
        let panel = MagcouplingPanel::with_baseline(crate::app::starting_design());
        let mut harness = Harness::of(panel, size);
        crate::app::theme::apply(&harness.ctx);
        crate::app::fonts::install(&harness.ctx);
        harness.frame(Vec::new());
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        let shown = [
            ("the starting design", false),
            ("the example design, details on", true),
        ];
        for (design, show_details) in shown {
            if design.starts_with("the example") {
                harness.panel.open_example_design();
            }
            harness.panel.show_details = show_details;
            for eq in registry().equations() {
                let at = format!("{}, {design}", eq.target);
                harness.panel.explorer.open_path(&eq.target);
                harness.frame(Vec::new());
                let output = harness.frame(Vec::new());
                let centre = centre_region(&harness.ctx);
                let centre = centre.left()..=centre.right();
                let panel = equation_panel_rect(&harness.ctx);
                assert!(
                    centre.contains(&panel.left()) && centre.contains(&panel.right()),
                    "{at}: the panel {panel:?} runs out of the centre region {centre:?}"
                );
                let inside = |clip: egui::Rect| panel.expand(1.0).contains_rect(clip);
                // Its last line, Used by, is drawn: every row above it is in view and checked.
                let users = used_by_shown(&eq.target, harness.panel.clamps_paused);
                assert!(
                    users.is_empty() || text_shapes(&output, USED_BY).any(|(clip, _)| inside(clip)),
                    "{at}: the panel's foot is out of view: the test would prove nothing"
                );
                // The formula's own area: where the panel draws its " = ".
                let formula = text_shapes(&output, " = ")
                    .map(|(clip, _)| clip)
                    .find(|&clip| inside(clip));
                let cut = cut_texts_in(&output, screen, |clip| {
                    inside(clip) && Some(clip) != formula
                });
                assert!(cut.is_empty(), "{at}: cut: {cut:?}");
            }
        }
    }

    #[test]
    fn closing_the_equation_panel_clears_the_input_row_s_frame() {
```

**`magcoupling-rs/src/gui/test_support.rs`**

Replace:

```rust
pub(crate) fn cut_texts(output: &egui::FullOutput, screen: egui::Rect) -> Vec<String> {
    clipped_flat_shapes(output)
```

with:

```rust
pub(crate) fn cut_texts(output: &egui::FullOutput, screen: egui::Rect) -> Vec<String> {
    cut_texts_in(output, screen, |_| true)
}

/// [`cut_texts`] of the texts whose clip rect `within` takes: those of one panel, or all but
/// those of an area that scrolls sideways by design.
pub(crate) fn cut_texts_in(
    output: &egui::FullOutput,
    screen: egui::Rect,
    within: impl Fn(egui::Rect) -> bool,
) -> Vec<String> {
    clipped_flat_shapes(output)
```

Replace:

```rust
            };
            let ink = text.visual_bounding_rect();
```

with:

```rust
            };
            if !within(clip) {
                return None;
            }
            let ink = text.visual_bounding_rect();
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 12
  ```

  Expected: 2 failed: `every_text_of_the_equation_panel_is_whole_on_each_start_here_step_in_the_app_s_own_font`, `the_term_list_wraps_inside_a_narrow_centre_region`. `every_text_of_the_equation_panel_is_whole_on_each_start_here_step_in_the_app_s_own_font` panics with `[1024.0 700.0], step 2: the panel [[336.0 395.1] - [732.6 684.0]] runs out of the centre region 328.0..=716.0`; `the_term_list_wraps_inside_a_narrow_centre_region` panics with `the tag is drawn`. On the old layout σ's tag lies past the term list's sideways view, where egui paints nothing, so the second test stops at its `expect`. The every-equation test passes on the old layout for the same reason (nothing past the term list's view is drawn); it holds Step 3's value wrap: with the value laid out on one line it fails on `temperature.thermal.rotations_to_limit_high` ("What the prototype showed" item 8).

- [ ] **Step 3 (green): the wraps.** The blocks make the walk bar's step line a wrapping label and the term grid four columns, the name and the tag in one wrapping line in the last, and lay a term's value out as a galley at a third of the list's width (`TERM_VALUE_SHARE`), wrapping past it.

**`magcoupling-rs/src/gui/explorer.rs`**

Replace:

```rust
pub const TERM_SIZE: f32 = 15.0;

```

with:

```rust
pub const TERM_SIZE: f32 = 15.0;

/// The share of the term list's width a term's value takes at most: past it the value wraps, so
/// the term's name and tag keep the rest. A number fits on one line; a text value ("never:
/// steady state stays below the limit") wraps in the smallest window (BL-086).
pub const TERM_VALUE_SHARE: f32 = 1.0 / 3.0;

```

Replace:

```rust
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.weak(start_here_text(step));
            if ui
```

with:

```rust
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            // The step line wraps inside the panel (BL-086: a long title ran past its edge and
            // made it wider than the centre region); the buttons stay whole.
            let line = egui::RichText::new(start_here_text(step)).weak();
            ui.add(egui::Label::new(line).wrap());
            if ui
```

Replace:

```rust
    let hidden = unsummed(eq, terms);
    // The list scrolls sideways when the region is narrower than its rows (the tag, its last
    // column, would be cut off), as the equation above it does.
    egui::ScrollArea::horizontal()
        .id_salt("magcoupling_equation_terms_scroll")
        .show(ui, |ui| {
```

with:

```rust
    let hidden = unsummed(eq, terms);
    // A term's name and its tag share the last column, which takes the panel's width left and
    // wraps them in it (BL-086: as two columns they ran past the panel's edge, in a sideways
    // scroll area of their own).
    let value_width = TERM_VALUE_SHARE * ui.available_width();
    egui::Grid::new("magcoupling_equation_terms")
        .num_columns(4)
        .striped(true)
        .show(ui, |ui| {
```

Replace:

```rust
        .show(ui, |ui| {
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
                        let symbol = ui
                            .fonts(|f| layout_symbol(f, registry(), &row.symbol, TERM_SIZE, color));
                        laid_ui(ui, &symbol, ink, Sense::hover());
                        let value = row.value.clone().unwrap_or(Value::None);
                        ui.label(value_text(&row.path, &value, &row.unit));
                        let term =
                            ui.add(egui::Label::new(term_label(&row.path)).sense(Sense::click()));
                        // Its internal name only with the spreadsheet details.
                        let term = if show_details {
                            term.on_hover_text(&row.path)
                        } else {
                            term
                        };
                        if term.clicked() {
                            clicked = Some(row.path.clone());
                        }
                        let mut style = registry()
                            .term_style(&row.path, terms.inputs)
                            .expect("every term is an input or a result (registry build)");
                        // The registry compares with the spreadsheet's values; on screen
                        // "changed" is against the session's baseline.
                        if matches!(style.kind, TermKind::Input { .. }) {
                            style.changed_from_default =
                                terms.inputs.get(&row.path) != baseline.get(&row.path);
                        }
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
```

with:

```rust
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
                // A long text value wraps at its share of the width, so the name and the tag
                // keep the rest. It is laid out here: a grid cell is only as wide as its column
                // was in the frame before, so a wrapping label there would never widen.
                let value = egui::WidgetText::from(value_text(&row.path, &value, &row.unit))
                    .into_galley(
                        ui,
                        Some(egui::TextWrapMode::Wrap),
                        value_width,
                        egui::TextStyle::Body,
                    );
                ui.label(value);
                ui.horizontal_wrapped(|ui| {
                    let term =
                        ui.add(egui::Label::new(term_label(&row.path)).sense(Sense::click()));
                    // Its internal name only with the spreadsheet details.
                    let term = if show_details {
                        term.on_hover_text(&row.path)
                    } else {
                        term
                    };
                    if term.clicked() {
                        clicked = Some(row.path.clone());
                    }
                    let mut style = registry()
                        .term_style(&row.path, terms.inputs)
                        .expect("every term is an input or a result (registry build)");
                    // The registry compares with the spreadsheet's values; on screen "changed"
                    // is against the session's baseline.
                    if matches!(style.kind, TermKind::Input { .. }) {
                        style.changed_from_default =
                            terms.inputs.get(&row.path) != baseline.get(&row.path);
                    }
                    let tag = term_tag(style);
                    if style.affected_by_modified_assumption
                        || matches!(style.kind, TermKind::Input { assumption: true })
                    {
                        ui.colored_label(ui.visuals().warn_fg_color, tag);
                    } else {
                        ui.weak(tag);
                    }
```

Replace:

```rust
                });
        });
```

with:

```rust
                });
                ui.end_row();
            }
        });
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1153 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text.

- [ ] **Step 5: Commit** the four files: `fix(magcoupler): a term's name and tag, a long value and the walk bar's step line wrap inside the Equation panel; the panel stays in the centre region on every equation (BL-086, BL-112 T3)`.

### Task 4: The keyboard's focus followed into the nested sideways areas (BL-101, parts 1 and 2)

**Model:** `sonnet`.

`follow_focus` returns the rect it scrolled to and takes as its own only a widget that lies in its area's content; one helper, `gui::sideways_scroll_area`, draws a sideways scroll area inside an up-and-down one that follows the focus both ways, by calling `follow_focus` last in its content and asking the outer area again for the rect once it has ended; the Equation panel's formula and the Clamp tab's screw-size table use it, and the Equation panel's own area calls `follow_focus` (decision D-112). Focus arriving selects nothing (BL-093); no Tab stop is added (BL-102).

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs` (red); `magcoupling-rs/src/gui/mod.rs`, `magcoupling-rs/src/gui/explorer.rs`, `magcoupling-rs/src/gui/clamp_drawing.rs` (green)

**Interfaces:**
- Changes: `pub(crate) fn gui::follow_focus(&egui::Ui) -> Option<egui::Rect>` (it returned nothing; its callers ignore the value); the widget it takes must also lie in `ui.min_rect()`.
- Produces: `pub(crate) fn gui::sideways_scroll_area<R>(&mut egui::Ui, id_salt: &str, add: impl FnOnce(&mut egui::Ui) -> R) -> R`.

- [ ] **Step 1 (red): the walks.** The blocks widen the walk helper `follow`'s whole check (a widget wider than its region, the formula, is whole up and down and shows from its start) and `in_region` (a widget its view cuts sideways belongs where its view ends), and add `the_equation_panel_follows_the_keyboard_s_focus_into_its_terms_and_formula` (1024 x 700, the panel open on the hot-low torque and on the hot minimum check, at its starting height and dragged to its least: after each Tab and Shift+Tab every stop of the panel is whole once settled, at least one stop out of view on arrival, nothing selected or edited) and `the_screw_size_table_follows_the_keyboard_s_focus_up_down_and_sideways` (the Clamp tab's table opened in a tall window, then a window 1024 wide and 1000 tall that holds the table's own drag surface: its cells cut sideways, and every stop of the centre region whole once settled).

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    /// stop of the top bar is above the region and whole, while a row scrolled out of the
    /// region's view, above it or below it, is cut.
    fn in_region(r: &egui::Response, region: egui::Rect) -> bool {
```

with:

```rust
    /// stop of the top bar is above the region and whole, while a row scrolled out of the
    /// region's view, above it or below it, is cut. A widget its view cuts sideways (a cell of
    /// the screw-size table past the table's right edge) is the view's, which ends in the
    /// region.
    fn in_region(r: &egui::Response, region: egui::Rect) -> bool {
```

Replace:

```rust
    fn in_region(r: &egui::Response, region: egui::Rect) -> bool {
        region.x_range().contains(r.rect.center().x)
            && (r.rect.top() >= region.top() || r.interact_rect != r.rect)
    }
```

with:

```rust
    fn in_region(r: &egui::Response, region: egui::Rect) -> bool {
        let cut = r.interact_rect != r.rect;
        let x = if cut {
            r.interact_rect.max.x
        } else {
            r.rect.center().x
        };
        region.x_range().contains(x) && (r.rect.top() >= region.top() || cut)
    }
```

Replace:

```rust
    /// scroll has settled, the focused widget of the region is whole in its scroll area's view
    /// (BL-100). Returns how many stops the region held, and how many of them were out of view
    /// when the focus arrived.
    fn follow(harness: &mut Harness, keys: &[egui::Event], region: egui::Rect) -> (usize, usize) {
```

with:

```rust
    /// scroll has settled, the focused widget of the region is whole in its scroll area's view
    /// (BL-100); one wider than the region (the Equation panel's formula, in a sideways area of
    /// its own) is whole up and down and shows from its start (BL-101). Returns how many stops
    /// the region held, and how many of them were out of view when the focus arrived.
    fn follow(harness: &mut Harness, keys: &[egui::Event], region: egui::Rect) -> (usize, usize) {
```

Replace:

```rust
        };
        let (mut stops, mut out_of_view) = (0, 0);
```

with:

```rust
        };
        let shown = |r: &egui::Response| {
            r.interact_rect == r.rect
                || (r.rect.width() > region.width()
                    && r.interact_rect.left() == r.rect.left()
                    && r.interact_rect.y_range() == r.rect.y_range())
        };
        let (mut stops, mut out_of_view) = (0, 0);
```

Replace:

```rust
            for _ in 0..8 {
                let whole = settled.interact_rect == settled.rect;
                harness.frame_after(1.0, Vec::new());
```

with:

```rust
            for _ in 0..8 {
                let whole = shown(&settled);
                harness.frame_after(1.0, Vec::new());
```

Replace:

```rust
            }
            assert_eq!(
                settled.interact_rect, settled.rect,
                "press {press}: the focused widget is cut by its area's view"
            );
```

with:

```rust
            }
            assert!(
                shown(&settled),
                "press {press}: the focused widget {:?} is cut by its area's view to {:?}",
                settled.rect,
                settled.interact_rect
            );
```

Replace:

```rust
    #[test]
    fn the_left_column_is_the_size_column_on_every_tab() {
```

with:

```rust
    #[test]
    fn the_equation_panel_follows_the_keyboard_s_focus_into_its_terms_and_formula() {
        // BL-101 (1): the Equation panel's own scroll area did not follow the focus, and its
        // formula scrolls sideways in an area nested in it, which egui gives every scroll
        // target. In the smallest window, the panel open on the hot-low torque and on the hot
        // minimum check, at its starting height and dragged to its least height: after each
        // Tab and Shift+Tab every stop of the panel is whole once the scroll has settled, the
        // hot minimum check's formula, wider than the panel, from its start. Nothing is
        // selected or edited.
        for (path, least) in [
            ("metal.torque_hot_low_Nm", false),
            ("metal.hot_min_check", false),
            ("metal.torque_hot_low_Nm", true),
            ("metal.hot_min_check", true),
        ] {
            let mut harness = Harness::of(MagcouplingPanel::new(), SMALLEST);
            harness.panel.explorer.open_path(path);
            harness.frame(Vec::new());
            if least {
                let ctx = harness.ctx.clone();
                let foot = equation_panel_rect(&ctx).bottom();
                drag_equation_panel(&ctx, &mut |events| harness.frame(events), foot);
            }
            harness.frame(Vec::new());
            let selection = harness.panel.selection;
            let panel = equation_panel_rect(&harness.ctx);
            let out_of_view: usize = [shift_tab(), key_tap(egui::Key::Tab)]
                .iter()
                .map(|keys| follow(&mut harness, keys, panel).1)
                .sum();
            assert!(
                out_of_view > 0,
                "{path}, least {least}: no stop was out of view"
            );
            assert_eq!(harness.panel.selection, selection, "{path}");
            assert_eq!(harness.panel.design(), Design::default(), "{path}");
        }
    }

    #[test]
    fn the_screw_size_table_follows_the_keyboard_s_focus_up_down_and_sideways() {
        // BL-101 (2): the Clamp tab's screw-size table scrolls sideways in an area nested in
        // the tab's own. Opened in a tall window, then in one of the smallest width and a
        // height that holds the whole table (its own drag surface, a Tab stop of egui's, is as
        // tall as the table: BL-102): after each Tab and Shift+Tab every stop of the centre
        // region is whole once the scroll has settled, the table's cells past its right edge
        // and under the view's foot among them.
        use crate::gui::clamp_drawing::SCREW_SIZES;
        use crate::gui::test_support::cut_texts;
        let mut harness = Harness::of(MagcouplingPanel::new(), TALL);
        harness.tab(CentreView::Clamp);
        harness.click_text(SCREW_SIZES);
        for _ in 0..8 {
            harness.frame(Vec::new());
        }
        harness.screen = egui::vec2(SMALLEST.x, 1000.0);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let centre = centre_region(&harness.ctx);
        let cells = format!("{SCREW_SIZES} is open with its cells cut sideways");
        assert!(!cut_texts(&output, centre).is_empty(), "{cells}");
        let out_of_view: usize = [shift_tab(), key_tap(egui::Key::Tab)]
            .iter()
            .map(|keys| follow(&mut harness, keys, centre).1)
            .sum();
        assert!(out_of_view > 0, "no stop was out of view");
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn the_left_column_is_the_size_column_on_every_tab() {
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib keyboard_s_focus 2>&1 | tail -n 12
  ```

  Expected: 2 failed: `the_equation_panel_follows_the_keyboard_s_focus_into_its_terms_and_formula`, `the_screw_size_table_follows_the_keyboard_s_focus_up_down_and_sideways`. `the_equation_panel_follows_the_keyboard_s_focus_into_its_terms_and_formula` panics with `press 20: the focused widget [[394.9 687.8] - [427.8 701.8]] is cut by its area's view to [[394.9 687.8] - [427.8 684.0]]`; `the_screw_size_table_follows_the_keyboard_s_focus_up_down_and_sideways` panics with `press 3: the focused widget [[791.8 968.0] - [798.8 982.0]] is cut by its area's view to [[791.8 968.0] - [708.0 982.0]]`.

- [ ] **Step 3 (green): the nested areas.** The blocks give `follow_focus` its return and its content check, add `sideways_scroll_area`, and use it for the formula and the screw-size table; the Equation panel's area calls `follow_focus` last.

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
/// drawn with the focus there, so a wheel or a drag of the scroll bar afterwards is not fought.
///
/// Not for an area nested in another: egui hands a scroll to the innermost area alone.
pub(crate) fn follow_focus(ui: &egui::Ui) {
    let focused = ui.memory(|m| m.focused());
```

with:

```rust
/// drawn with the focus there, so a wheel or a drag of the scroll bar afterwards is not fought.
/// Returns the widget's rect when it scrolls to it: egui hands both axes' scroll targets to the
/// innermost area alone, so an area nested in another passes the rect on to the outer one
/// ([`sideways_scroll_area`], BL-101).
pub(crate) fn follow_focus(ui: &egui::Ui) -> Option<egui::Rect> {
    let focused = ui.memory(|m| m.focused());
```

Replace:

```rust
    // adds little, as a list's widget passes the clip check only whole in the view, where no
    // scroll follows.
    if focused != seen
```

with:

```rust
    // adds little, as a list's widget passes the clip check only whole in the view, where no
    // scroll follows. A sideways area nested in an up-and-down one is clipped up and down by
    // the outer area's view (egui keeps the parent's clip on the axis an area does not scroll),
    // so its widgets are also those in its content: else it would take the outer area's,
    // read from the frame before, and scroll the outer area a second time (BL-101).
    if focused != seen
```

Replace:

```rust
        && widget.interact_rect == widget.rect.intersect(ui.clip_rect())
    {
```

with:

```rust
        && widget.interact_rect == widget.rect.intersect(ui.clip_rect())
        && ui.min_rect().contains_rect(widget.rect)
    {
```

Replace:

```rust
        ui.scroll_to_rect(widget.rect, None);
    }
```

with:

```rust
        ui.scroll_to_rect(widget.rect, None);
        return Some(widget.rect);
    }
```

Replace:

```rust
    }
}
```

with:

```rust
    }
    None
}
```

Replace:

```rust
    None
}

```

with:

```rust
    None
}

/// A sideways scroll area inside an up-and-down one (the Equation panel's formula, the Clamp
/// tab's screw-size table) that follows the keyboard's focus both ways (BL-101). It calls
/// [`follow_focus`] last in its content, so it scrolls sideways to the widget; egui gives it
/// both axes' targets, so the outer area's content asks again for the rect once it has ended,
/// and the outer area scrolls up or down to the widget. A widget wider than the view shows from
/// its start (egui's least scroll).
pub(crate) fn sideways_scroll_area<R>(
    ui: &mut egui::Ui,
    id_salt: &str,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let (inner, followed) = egui::ScrollArea::horizontal()
        .id_salt(id_salt)
        .show(ui, |ui| {
            let inner = add(ui);
            (inner, follow_focus(ui))
        })
        .inner;
    if let Some(rect) = followed {
        ui.scroll_to_rect(rect, None);
    }
    inner
}

```

**`magcoupling-rs/src/gui/clamp_drawing.rs`**

Replace:

```rust
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
                            let rect = ui.label(rich).rect;
                            readouts.show_over(ui, rect, &path, |show_details| {
                                hover_text(&path, show_details).unwrap_or_else(|| path.clone())
                            });
                        }
                        ui.end_row();
                    }
```

with:

```rust
    let good = Level::Good.color(ui.visuals());
    crate::gui::sideways_scroll_area(ui, "magcoupling_screw_sizes_scroll", |ui| {
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
                        let rect = ui.label(rich).rect;
                        readouts.show_over(ui, rect, &path, |show_details| {
                            hover_text(&path, show_details).unwrap_or_else(|| path.clone())
                        });
                    }
```

Replace:

```rust
                    }
                });
        });
}
```

with:

```rust
                    }
                    ui.end_row();
                }
            });
    });
}
```

**`magcoupling-rs/src/gui/explorer.rs`**

Replace:

```rust
};
use crate::{DesignInputs, DesignResults};
```

with:

```rust
};
use crate::gui::{follow_focus, sideways_scroll_area};
use crate::{DesignInputs, DesignResults};
```

Replace:

```rust
                explorer.drill(&user);
            }
        });
    asked
```

with:

```rust
                explorer.drill(&user);
            }
            follow_focus(ui);
        });
    asked
```

Replace:

```rust
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
```

with:

```rust
    let laid = ui.fonts(|f| layout_equation(f, registry(), eq, &colors, PANEL_SIZE, ink));
    sideways_scroll_area(ui, "magcoupling_equation_formula", |ui| {
        let response = laid_ui(ui, &laid, ink, Sense::click());
        if response.clicked()
            && let Some(at) = response.interact_pointer_pos()
        {
            clicked = term_at(&laid, response.rect.min, at);
        }
    });
    let value = terms.results.get(&eq.target).unwrap_or(Value::None);
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1155 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text.

- [ ] **Step 5: Commit** the four files: `fix(magcoupler): the Equation panel, its formula and the screw-size table scroll to the keyboard's focus (BL-101, BL-112 T4)`.

### Task 5: The Results table's Tab stops are its whole lines; the check list follows the focus (BL-101, parts 3 and 4)

**Model:** `sonnet`.

The Results table keeps owner question 67's default (Tab passes the lines on screen and does not scroll the table) and takes the focus only on a line its view shows whole up and down, so the focus never sits on a line half seen (decision D-113). The check list's scroll area calls `follow_focus`; `follow_focus` keeps the focus it saw last under its content's unique id, since egui gives a scroll area's content its parent's id and the check list and the Explain panel under it share one (decision D-114).

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs` (red and green), `magcoupling-rs/src/gui/readouts.rs`, `magcoupling-rs/src/gui/results_table.rs`, `magcoupling-rs/src/gui/mod.rs` (green)

**Interfaces:**
- Changes: `pub(crate) fn gui::readouts::sensor(&egui::Ui, Rect, &str, egui::Sense) -> egui::Response` (it was private with `Sense::click()` built in; `show_over` and `select_over` pass `Sense::click()`).
- Produces: `fn results_table::line_sense(&egui::Ui, egui::Rect) -> egui::Sense` (a click, and the focus only on a whole line).

- [ ] **Step 1 (red): the walks.** The blocks add `the_results_table_s_tab_stops_are_the_lines_it_shows_whole` (1024 x 700 and 1280 x 800, the Results tab, as it opens and wheeled 300 points down, where the wheel must move the table: after each Tab and Shift+Tab more than ten stops, none out of view as the focus arrives, no text of the centre region moved, nothing selected or edited) and `the_check_list_follows_the_keyboard_s_focus_where_it_scrolls` (a window 1024 x 460, where the list scrolls: after each Tab and Shift+Tab every stop of the right column whole once settled, at least one out of view on arrival, nothing selected or edited).

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    #[test]
    fn the_left_column_is_the_size_column_on_every_tab() {
```

with:

```rust
    #[test]
    fn the_results_table_s_tab_stops_are_the_lines_it_shows_whole() {
        // BL-101 (3), owner question 67's default (register row A-44): the table draws only the
        // lines in view and does not follow the focus, so Tab passes the lines on screen and
        // leaves it; the line its foot cut, and the one drawn past the foot, took the focus
        // unseen. In the smallest window and at 1280 x 800, as the table opens (a group heading
        // drawn past the foot) and wheeled 300 points down (at 1280 x 800 a row then cut at the
        // top), after each Tab and Shift+Tab every stop of the centre region is whole as it
        // arrives, and the table does not scroll. Nothing is selected or edited.
        for screen in [SMALLEST, egui::vec2(1280.0, 800.0)] {
            for wheel in [0.0, 300.0] {
                let at = format!("{screen:?}, wheeled {wheel}");
                let mut harness = Harness::of(MagcouplingPanel::new(), screen);
                harness.tab(CentreView::Results);
                harness.frame(Vec::new());
                let selection = harness.panel.selection;
                let centre = centre_region(&harness.ctx);
                // Every text of the centre region where it is drawn: while none moves, the
                // table has not scrolled.
                let texts = |harness: &mut Harness| -> Vec<(String, egui::Rect)> {
                    let output = harness.frame(Vec::new());
                    crate::gui::test_support::flat_shapes(&output)
                        .into_iter()
                        .filter_map(|shape| match shape {
                            egui::Shape::Text(text) => {
                                Some((text.galley.text().to_owned(), text.visual_bounding_rect()))
                            }
                            _ => None,
                        })
                        .filter(|(_, rect)| centre.contains(rect.center()))
                        .collect()
                };
                if wheel > 0.0 {
                    let opened = texts(&mut harness);
                    harness.wheel_column(centre.center(), wheel, 30);
                    harness.frame(vec![egui::Event::PointerGone]);
                    harness.frame_after(1.0, Vec::new());
                    assert!(
                        texts(&mut harness) != opened,
                        "{at}: the wheel moved nothing"
                    );
                }
                let before = texts(&mut harness);
                for keys in [shift_tab(), key_tap(egui::Key::Tab)] {
                    let (stops, out_of_view) = follow(&mut harness, &keys, centre);
                    assert!(stops > 10, "{at}: {stops} stops");
                    assert_eq!(out_of_view, 0, "{at}: of {stops} stops");
                }
                assert!(texts(&mut harness) == before, "{at}: the table scrolled");
                assert_eq!(harness.panel.selection, selection, "{at}");
                assert_eq!(harness.panel.design(), Design::default(), "{at}");
            }
        }
    }

    #[test]
    fn the_check_list_follows_the_keyboard_s_focus_where_it_scrolls() {
        // BL-101 (4): the check list scrolls past its share of the right column, and did not
        // follow the focus. In a window short enough for it to scroll (a host's window: the
        // page's smallest is taller), after each Tab and Shift+Tab every stop of the right
        // column is whole once the scroll has settled. Nothing is selected or edited.
        let mut harness = Harness::of(MagcouplingPanel::new(), egui::vec2(1024.0, 460.0));
        let selection = harness.panel.selection;
        let column = side_rect(&harness.ctx, "magcoupling_dashboard");
        let out_of_view: usize = [key_tap(egui::Key::Tab), shift_tab()]
            .iter()
            .map(|keys| follow(&mut harness, keys, column).1)
            .sum();
        assert!(out_of_view > 0, "no stop was out of view");
        assert_eq!(harness.panel.selection, selection);
        assert_eq!(harness.panel.design(), Design::default());
    }

    #[test]
    fn the_left_column_is_the_size_column_on_every_tab() {
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_results_table_s_tab_stops the_check_list_follows 2>&1 | tail -n 12
  ```

  Expected: 2 failed: `the_check_list_follows_the_keyboard_s_focus_where_it_scrolls`, `the_results_table_s_tab_stops_are_the_lines_it_shows_whole`. `the_check_list_follows_the_keyboard_s_focus_where_it_scrolls` panics with `press 30: the focused widget [[724.0 225.0] - [1008.0 243.0]] is cut by its area's view to [[724.0 225.0] - [1008.0 224.8]]`; `the_results_table_s_tab_stops_are_the_lines_it_shows_whole` panics with `press 2: the focused widget [[336.0 699.9] - [708.0 718.2]] is cut by its area's view to [[336.0 699.9] - [708.0 684.0]]`. The Results table's walk stops at its first case, 1024 x 700 as the table opens, on the group heading drawn past the foot; its wheeled case holds the data rows ("What the prototype showed" items 6 and 8).

- [ ] **Step 3 (green): the lines and the list.** The blocks give `readouts::sensor` its sense, the table's headings and rows `line_sense`, the check list's area its `follow_focus`, and `follow_focus` its slot under the unique id. `the_explain_panel_follows_the_keyboard_s_focus`, unchanged, holds the slot: with the check list's call and the slot under `ui.id()` it fails.

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
    // Where the focus was when this area was last drawn. egui's own `gained_focus` misses a
    // move made between two frames, as an Up or Down arrow key's is.
    let seen = ui.data_mut(|d| {
```

with:

```rust
    // Where the focus was when this area was last drawn. egui's own `gained_focus` misses a
    // move made between two frames, as an Up or Down arrow key's is. Kept under the content's
    // unique id: egui gives a scroll area's content its parent's id, which sibling areas share
    // (the check list and the Explain panel under it, BL-101); the unique id holds while the
    // widgets before the area in its parent stay as they are.
    let seen = ui.data_mut(|d| {
```

Replace:

```rust
    let seen = ui.data_mut(|d| {
        let seen = d.get_temp_mut_or_default::<Option<egui::Id>>(ui.id().with("follow_focus"));
        std::mem::replace(seen, focused)
```

with:

```rust
    let seen = ui.data_mut(|d| {
        let key = ui.unique_id().with("follow_focus");
        let seen = d.get_temp_mut_or_default::<Option<egui::Id>>(key);
        std::mem::replace(seen, focused)
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
                    self.selection = Selection::Check(check);
                }
            });
        ui.separator();
```

with:

```rust
                    self.selection = Selection::Check(check);
                }
                follow_focus(ui);
            });
        ui.separator();
```

**`magcoupling-rs/src/gui/readouts.rs`**

Replace:

```rust
    ) {
        self.select(ui, sensor(ui, rect, path), path, text);
    }
```

with:

```rust
    ) {
        self.select(ui, sensor(ui, rect, path, egui::Sense::click()), path, text);
    }
```

Replace:

```rust
    ) {
        self.show(ui, sensor(ui, rect, path), path, text);
    }
```

with:

```rust
    ) {
        self.show(ui, sensor(ui, rect, path, egui::Sense::click()), path, text);
    }
```

Replace:

```rust

/// The click sensor over `rect` of the readout at `path`, keyed by `path` in `ui`.
fn sensor(ui: &egui::Ui, rect: Rect, path: &str) -> egui::Response {
    ui.interact(rect, ui.id().with(("readout", path)), egui::Sense::click())
}
```

with:

```rust

/// The click sensor over `rect` of the readout at `path`, keyed by `path` in `ui`: `sense` is a
/// click, with the keyboard's focus or (a results table line the view cuts) without it.
pub(crate) fn sensor(ui: &egui::Ui, rect: Rect, path: &str, sense: egui::Sense) -> egui::Response {
    ui.interact(rect, ui.id().with(("readout", path)), sense)
}
```

**`magcoupling-rs/src/gui/results_table.rs`**

Replace:

```rust
fn group_heading_ui(ui: &mut egui::Ui, line: &HeadingLine, size: egui::Vec2) -> bool {
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    // The triangle where egui's collapsing header puts it: its inner icon square, centred in
```

with:

```rust
fn group_heading_ui(ui: &mut egui::Ui, line: &HeadingLine, size: egui::Vec2) -> bool {
    let at = egui::Rect::from_min_size(ui.cursor().min, size);
    let (rect, response) = ui.allocate_exact_size(size, line_sense(ui, at));
    // The triangle where egui's collapsing header puts it: its inner icon square, centred in
```

Replace:

```rust

/// One table row: label, value with unit (after the badge of a check's `level`), and with the
```

with:

```rust

/// What a line of the table at `rect` senses: a click, and the keyboard's focus only while the
/// table's view shows the line whole up and down (BL-101; owner question 67's default, register
/// row A-44). Tab passes the lines on screen and leaves the table after the last whole one: it
/// never stops on a line the view's top or foot cuts, nor on one drawn past the foot, and the
/// table does not scroll for it. A focused line a wheel then cuts loses the focus.
fn line_sense(ui: &egui::Ui, rect: egui::Rect) -> egui::Sense {
    let view = ui.clip_rect();
    if view.top() <= rect.top() && rect.bottom() <= view.bottom() {
        egui::Sense::click()
    } else {
        egui::Sense::CLICK
    }
}

/// One table row: label, value with unit (after the badge of a check's `level`), and with the
```

Replace:

```rust
    });
    readouts.select_over(ui, row.response.rect, &entry.path, |show_details| {
        row_tooltip(entry, &value, show_details)
```

with:

```rust
    });
    let rect = row.response.rect;
    let sensor = crate::gui::readouts::sensor(ui, rect, &entry.path, line_sense(ui, rect));
    readouts.select(ui, sensor, &entry.path, |show_details| {
        row_tooltip(entry, &value, show_details)
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

  Expected: `test result: ok. 1157 passed; 0 failed; 6 ignored` for the lib tests with the app feature (1568 tests in all with it, 656 without); every clippy run clean; `all 85 checks hold`.

- [ ] **Step 5: Commit** the four files: `fix(magcoupler): a Results table line takes the keyboard's focus only while whole; the check list follows the focus (BL-101, BL-112 T5)`.

### Task 6: Docs sweep

**Model:** `sonnet`.

The README's module table, keyboard, Equation panel, teaching notes and tests passages, among them the note drawing's tests row and the smoke prompt's R14 steps, which name `notes_view` now; `docs/ai/` (02 the system, with `notes_view::live_value`, `change_ui` and `explains_ui` and egui's grid-cell width among its gotchas; 03 the modules, 04 the issue, 05 the tracker, BL-086, BL-088, BL-089, BL-101 and BL-112 in progress); the register's row A-44 (its cost), rows A-55 and A-56 (BL-089's open choice, decision D-108) and their log; owner question 67's text, question 76 and the list's count of part C (57); the design's status line.

**Files:**
- Modify: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/analyses/2026-11-18-owner-questions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

- [ ] **Step 1: The docs.**

**`docs/ai/02-system.yaml`**

Replace:

```yaml
      - "Teaching notes (guidance slice 2, engine::explain::notes; spec docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md 3.1): a note's What to change entries name an input path, a Step (Raise, Lower, Choose or Enter) and, for a step, the result it moves and which way (aim; a model assumption's step may name none, an entered number or a choice never does); the label on screen is always gui::plain_text::label_for(path), never typed. An entry's input is a link to its row unless it is not free on the design shown: held by Auto (gui::sizing::held_by_auto, the rule that locks the row), set by a part pick (engine::picks::set_by_pick, proven against the engine by nudging every input on every pick), or at the end of its slider the entry's way (SliderRange::step_from, the step a check guide's fix takes). A note's sources sit under a fold that starts closed."
      - "Live values in teaching notes (guidance slice 2, engine::explain::note_text): a {result.path} in a sentence, a change text or a watch-out is drawn as that result's number with its unit on the design shown (gui::explorer::live_value, the readouts' format); a result with no finite number there, or computed from an out-of-range end effect, reads no prediction (NO_VALUE). None stands in a title, a sentence 0 or a source; each names a number result with no unit typed after it and stands in a clause with the whole word here or the words on this design (where and there do not count; tests/explain.rs). The five probe designs (engine::explain::probes::probe_designs: the default, 8 magnets per ring, a 304 back iron, Y30 on custom dimensions, a 20 mm ring radius set through the Size tab's mapping) are where the note tests and each note's physics review read every note; every change entry's aim is checked there by one slider step of its input (probes::step_input) against notes::Sense::holds, the one rule the review sheet prints as well. No note text or source holds an audit id (engine::explain::plain::audit_ids: TC-, PA-, CG-), checked by the notes' one plain-words rule (notes::plain_words_problems), which the all-notes plain-words test runs over every note shown at the five probe designs."
      - "Help batch A and the hint lines (slice 2, T7; spec docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md 3.5): gui::plain_text::HELP gives every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint (Shaft, key and clamps until T12b) workflow groups on-screen help: the 35 empty helps there, and 15 terse, stale or jargon workbook helps replaced on screen (the pole count, face gap, requirement, ring radius, operating temperature, gearbox rating, the three space-claim limits, the two unconfirmed allowances, slip speed, hot-day ambient, service life, daily swing); engine metadata keeps the workbook text. Each help is 1-2 sentences of at most 30 words with no number, cell or audit id. Every direction claim was probed with compute_all (the five probe designs; the G-3 review up to 24 designs, manual-block designs included), and the text follows the non-obvious results: the outer magnet thickness adds torque only up to a point and the inner one usually (it dips at 8 manual magnets per ring); fewer magnets usually lose torque and, if flat, enlarge the cup; a larger ring radius can lower the pull-out (at 8 magnets), so its help claims no torque direction; a larger bore moves the clamp screws outward, so fewer sizes fit; the hot-day ambient lowers the hot-day pull-out but not hot-low. The batch passed G-3 review in round 2 after round 1's eight must-fix items (docs/analyses/2026-10-14-magcoupler-help-review.md). gui::plain_text::HINTS holds one weak line of at most 12 words, drawn by gui::input_ui::hint_line, under each Key design row and each Size-tab control; the Size tab's Minimum OD, Outside diameter and Overall length are keyed by SIZE_TAB_HINT_KEYS (size_tab.*, not input paths: the Overall length edits the axial magnet length override, whose Key design row means the magnet length itself), the gap and the magnet count by their input paths; the Minimum OD hint replaced the line under the Size-tab controls that said the same. A hint never types an input label (a test). The two length hints name the clamp demand (longer magnets raise the cold-high torque, and at the example design the clamp fails before the space claim), and the back-iron and cup-wall hints name magnetic steel and the wall check, not saturation."
```

with:

```yaml
      - "Teaching notes (guidance slice 2, engine::explain::notes; spec docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md 3.1): a note's What to change entries name an input path, a Step (Raise, Lower, Choose or Enter) and, for a step, the result it moves and which way (aim; a model assumption's step may name none, an entered number or a choice never does); the label on screen is always gui::plain_text::label_for(path), never typed. An entry's input is a link to its row unless it is not free on the design shown: held by Auto (gui::sizing::held_by_auto, the rule that locks the row), set by a part pick (engine::picks::set_by_pick, proven against the engine by nudging every input on every pick), or at the end of its slider the entry's way (SliderRange::step_from, the step a check guide's fix takes). A note's sources sit under a fold that starts closed."
      - "Live values in teaching notes (guidance slice 2, engine::explain::note_text): a {result.path} in a sentence, a change text or a watch-out is drawn as that result's number with its unit on the design shown (gui::notes_view::live_value, the readouts' format); a result with no finite number there, or computed from an out-of-range end effect, reads no prediction (NO_VALUE). None stands in a title, a sentence 0 or a source; each names a number result with no unit typed after it and stands in a clause with the whole word here or the words on this design (where and there do not count; tests/explain.rs). The five probe designs (engine::explain::probes::probe_designs: the default, 8 magnets per ring, a 304 back iron, Y30 on custom dimensions, a 20 mm ring radius set through the Size tab's mapping) are where the note tests and each note's physics review read every note; every change entry's aim is checked there by one slider step of its input (probes::step_input) against notes::Sense::holds, the one rule the review sheet prints as well. No note text or source holds an audit id (engine::explain::plain::audit_ids: TC-, PA-, CG-), checked by the notes' one plain-words rule (notes::plain_words_problems), which the all-notes plain-words test runs over every note shown at the five probe designs."
      - "Help batch A and the hint lines (slice 2, T7; spec docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md 3.5): gui::plain_text::HELP gives every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint (Shaft, key and clamps until T12b) workflow groups on-screen help: the 35 empty helps there, and 15 terse, stale or jargon workbook helps replaced on screen (the pole count, face gap, requirement, ring radius, operating temperature, gearbox rating, the three space-claim limits, the two unconfirmed allowances, slip speed, hot-day ambient, service life, daily swing); engine metadata keeps the workbook text. Each help is 1-2 sentences of at most 30 words with no number, cell or audit id. Every direction claim was probed with compute_all (the five probe designs; the G-3 review up to 24 designs, manual-block designs included), and the text follows the non-obvious results: the outer magnet thickness adds torque only up to a point and the inner one usually (it dips at 8 manual magnets per ring); fewer magnets usually lose torque and, if flat, enlarge the cup; a larger ring radius can lower the pull-out (at 8 magnets), so its help claims no torque direction; a larger bore moves the clamp screws outward, so fewer sizes fit; the hot-day ambient lowers the hot-day pull-out but not hot-low. The batch passed G-3 review in round 2 after round 1's eight must-fix items (docs/analyses/2026-10-14-magcoupler-help-review.md). gui::plain_text::HINTS holds one weak line of at most 12 words, drawn by gui::input_ui::hint_line, under each Key design row and each Size-tab control; the Size tab's Minimum OD, Outside diameter and Overall length are keyed by SIZE_TAB_HINT_KEYS (size_tab.*, not input paths: the Overall length edits the axial magnet length override, whose Key design row means the magnet length itself), the gap and the magnet count by their input paths; the Minimum OD hint replaced the line under the Size-tab controls that said the same. A hint never types an input label (a test). The two length hints name the clamp demand (longer magnets raise the cold-high torque, and at the example design the clamp fails before the space claim), and the back-iron and cup-wall hints name magnetic steel and the wall check, not saturation."
```

Replace:

```yaml
      - "Help batch A and the hint lines (slice 2, T7; spec docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md 3.5): gui::plain_text::HELP gives every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint (Shaft, key and clamps until T12b) workflow groups on-screen help: the 35 empty helps there, and 15 terse, stale or jargon workbook helps replaced on screen (the pole count, face gap, requirement, ring radius, operating temperature, gearbox rating, the three space-claim limits, the two unconfirmed allowances, slip speed, hot-day ambient, service life, daily swing); engine metadata keeps the workbook text. Each help is 1-2 sentences of at most 30 words with no number, cell or audit id. Every direction claim was probed with compute_all (the five probe designs; the G-3 review up to 24 designs, manual-block designs included), and the text follows the non-obvious results: the outer magnet thickness adds torque only up to a point and the inner one usually (it dips at 8 manual magnets per ring); fewer magnets usually lose torque and, if flat, enlarge the cup; a larger ring radius can lower the pull-out (at 8 magnets), so its help claims no torque direction; a larger bore moves the clamp screws outward, so fewer sizes fit; the hot-day ambient lowers the hot-day pull-out but not hot-low. The batch passed G-3 review in round 2 after round 1's eight must-fix items (docs/analyses/2026-10-14-magcoupler-help-review.md). gui::plain_text::HINTS holds one weak line of at most 12 words, drawn by gui::input_ui::hint_line, under each Key design row and each Size-tab control; the Size tab's Minimum OD, Outside diameter and Overall length are keyed by SIZE_TAB_HINT_KEYS (size_tab.*, not input paths: the Overall length edits the axial magnet length override, whose Key design row means the magnet length itself), the gap and the magnet count by their input paths; the Minimum OD hint replaced the line under the Size-tab controls that said the same. A hint never types an input label (a test). The two length hints name the clamp demand (longer magnets raise the cold-high torque, and at the example design the clamp fails before the space claim), and the back-iron and cup-wall hints name magnetic steel and the wall check, not saturation."
      - "Reachability (guidance slice 2, spec 3.7, S2-8): more ways to the notes without turning Explain on first. Start here opens with the torque note; the empty Equation panel names Start here and Explain; Why links (gui::dashboard::why_note_link, one 'Why: <title>' link, never the bare title, which Start here draws once) sit under Magnets per ring and Outside diameter and on the torque card's heading (guidance::SIZE_TAB_NOTE; slice 3), at the end of the Required and Hot-low rows (TORQUE_BAND_NOTE, the hot minimum check's note) and under each copy of the end-effect banner (BANNER_NOTE, the end-effect check's note; dashboard::end_effect_banner_ui); a note not reviewed draws no link, and release_notes_are_reviewed keeps the three reviewed. Under an equation without a reviewed note, or a value with no equation record, Explain shows explorer::note_fallback: the dashboard row's own Why note or assumption, else the nearest reviewed note upstream (breadth first over the registry's terms; a NoteLink::Note link that opens it), else the nearest assumption's rationale, else the row's reason or NO_NOTE, with START_HERE_HINT; NO_NOTE never shows for a dashboard row. The walk runs on each frame Explain shows it (a few hundred paths at most), no cache. G-3 physics review of the strings: PASS in round 3 after round 1's three must-fix items (START_HERE_HINT and EMPTY_TEXT say Start here walks the main notes in a suggested order; NO_NOTE) and round 2's one (NO_NOTE claims only what the panel shows: under a value with no record nothing is known upstream) (docs/analyses/2026-10-14-magcoupler-help-review.md, Reachability strings (T19))."
      - "The look (UX overhaul slice B, stage 1, BL-079; design sections 6 and 8.1 tests 6 and 7): the standalone app owns its theme and its font, and the panel still sets neither. app::theme holds the one token table (BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK, LINK) and visuals(), egui's dark visuals with the tokens in place of its colours: a test holds every field to the table and every other field to egui's, and another holds every text token to 4.5:1 on every surface it is drawn on (muted and the level colours are under it on the border colour, the fill of a hovered button and of a selected chip, whose own text is TEXT and LINK; a caption whose text is selected with the pointer is not counted). Body text, values and headings share TEXT (egui has no bold: strong() no longer sets a value apart), weak() is MUTED. The accent is never a text colour and never on a value or a check: it marks the brand mark and the active tab only (gui::look::mark_ui and tab_ui; a test counts its shapes on the app's page). ACCENT and OK are the panel's own constants (gui::look), since no egui theme has a field for them; the spreadsheet export's fills keep their own palette (xlsx::fill). app::fonts installs a cut of Inter Regular first in the proportional family (egui's built-in fonts stay behind it and stay monospace); the cut is built once by assets/fonts/subset_inter.py, its ten digits are Inter's tabular ones, and with the app feature every glyph guard also holds its texts to the cut (test_support::assert_glyphs through app::fonts::cut_holds). The spacing is unchanged, and decision M41-3 (theme matches the linkage app) is reversed for the calculator alone. Since the fix wave of the stage's whole-branch review (2026-11-22): a label beside its value, a caption's head, a tab that is not active and a plot's axes are drawn in gui::look::quiet_text, a theme's own weak colour (the app's MUTED) and the host's text colour where a host sets none (egui's own weak text is about 2.6:1); the fill behind selected text and of a selected choice is app::theme::SELECTION, a quiet blue outside the token table (with CONTROL since BL-107) (3:1 against a field and the panel, the text colour on it 4.5:1; register row A-19), with a selected choice's name in the text colour; the panel's own marks (the changed dot, a trace's banner and frames) read gui::look::mark_color, the host's link colour; the font cut carries the ascender, descender and line gap of egui's built-in font, so a row is as tall as the layout's own font made it; and no text is cut sideways in any tab at 1280 x 800 or 1024 x 700 in the app's font, which the app test no_text_is_cut_in_any_tab_in_the_app_s_own_font holds (the pole sweep's line is named in a wrapping caption over its plot, a plot readout longer than a row wraps, a tooltip's equation wider than the window is set smaller, and a plot may paint its axis label a quarter row into the centre region's margin)."
```

with:

```yaml
      - "Help batch A and the hint lines (slice 2, T7; spec docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md 3.5): gui::plain_text::HELP gives every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint (Shaft, key and clamps until T12b) workflow groups on-screen help: the 35 empty helps there, and 15 terse, stale or jargon workbook helps replaced on screen (the pole count, face gap, requirement, ring radius, operating temperature, gearbox rating, the three space-claim limits, the two unconfirmed allowances, slip speed, hot-day ambient, service life, daily swing); engine metadata keeps the workbook text. Each help is 1-2 sentences of at most 30 words with no number, cell or audit id. Every direction claim was probed with compute_all (the five probe designs; the G-3 review up to 24 designs, manual-block designs included), and the text follows the non-obvious results: the outer magnet thickness adds torque only up to a point and the inner one usually (it dips at 8 manual magnets per ring); fewer magnets usually lose torque and, if flat, enlarge the cup; a larger ring radius can lower the pull-out (at 8 magnets), so its help claims no torque direction; a larger bore moves the clamp screws outward, so fewer sizes fit; the hot-day ambient lowers the hot-day pull-out but not hot-low. The batch passed G-3 review in round 2 after round 1's eight must-fix items (docs/analyses/2026-10-14-magcoupler-help-review.md). gui::plain_text::HINTS holds one weak line of at most 12 words, drawn by gui::input_ui::hint_line, under each Key design row and each Size-tab control; the Size tab's Minimum OD, Outside diameter and Overall length are keyed by SIZE_TAB_HINT_KEYS (size_tab.*, not input paths: the Overall length edits the axial magnet length override, whose Key design row means the magnet length itself), the gap and the magnet count by their input paths; the Minimum OD hint replaced the line under the Size-tab controls that said the same. A hint never types an input label (a test). The two length hints name the clamp demand (longer magnets raise the cold-high torque, and at the example design the clamp fails before the space claim), and the back-iron and cup-wall hints name magnetic steel and the wall check, not saturation."
      - "Reachability (guidance slice 2, spec 3.7, S2-8): more ways to the notes without turning Explain on first. Start here opens with the torque note; the empty Equation panel named Start here and Explain until BL-089 took its text away (every way of opening the panel sets a trail first); Why links (gui::dashboard::why_note_link, one 'Why: <title>' link, never the bare title, which Start here draws once) sit under Magnets per ring and Outside diameter and on the torque card's heading (guidance::SIZE_TAB_NOTE; slice 3), at the end of the Required and Hot-low rows (TORQUE_BAND_NOTE, the hot minimum check's note) and under each copy of the end-effect banner (BANNER_NOTE, the end-effect check's note; dashboard::end_effect_banner_ui); a note not reviewed draws no link, and release_notes_are_reviewed keeps the three reviewed. Under an equation without a reviewed note, or a value with no equation record, Explain shows explorer::note_fallback: the dashboard row's own Why note or assumption, else the nearest reviewed note upstream (breadth first over the registry's terms; a NoteLink::Note link that opens it), else the nearest assumption's rationale, else the row's reason or NO_NOTE, with START_HERE_HINT; NO_NOTE never shows for a dashboard row. The walk runs on each frame Explain shows it (a few hundred paths at most), no cache. G-3 physics review of the strings: PASS in round 3 after round 1's three must-fix items (START_HERE_HINT and EMPTY_TEXT say Start here walks the main notes in a suggested order; NO_NOTE) and round 2's one (NO_NOTE claims only what the panel shows: under a value with no record nothing is known upstream) (docs/analyses/2026-10-14-magcoupler-help-review.md, Reachability strings (T19))."
      - "The look (UX overhaul slice B, stage 1, BL-079; design sections 6 and 8.1 tests 6 and 7): the standalone app owns its theme and its font, and the panel still sets neither. app::theme holds the one token table (BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK, LINK) and visuals(), egui's dark visuals with the tokens in place of its colours: a test holds every field to the table and every other field to egui's, and another holds every text token to 4.5:1 on every surface it is drawn on (muted and the level colours are under it on the border colour, the fill of a hovered button and of a selected chip, whose own text is TEXT and LINK; a caption whose text is selected with the pointer is not counted). Body text, values and headings share TEXT (egui has no bold: strong() no longer sets a value apart), weak() is MUTED. The accent is never a text colour and never on a value or a check: it marks the brand mark and the active tab only (gui::look::mark_ui and tab_ui; a test counts its shapes on the app's page). ACCENT and OK are the panel's own constants (gui::look), since no egui theme has a field for them; the spreadsheet export's fills keep their own palette (xlsx::fill). app::fonts installs a cut of Inter Regular first in the proportional family (egui's built-in fonts stay behind it and stay monospace); the cut is built once by assets/fonts/subset_inter.py, its ten digits are Inter's tabular ones, and with the app feature every glyph guard also holds its texts to the cut (test_support::assert_glyphs through app::fonts::cut_holds). The spacing is unchanged, and decision M41-3 (theme matches the linkage app) is reversed for the calculator alone. Since the fix wave of the stage's whole-branch review (2026-11-22): a label beside its value, a caption's head, a tab that is not active and a plot's axes are drawn in gui::look::quiet_text, a theme's own weak colour (the app's MUTED) and the host's text colour where a host sets none (egui's own weak text is about 2.6:1); the fill behind selected text and of a selected choice is app::theme::SELECTION, a quiet blue outside the token table (with CONTROL since BL-107) (3:1 against a field and the panel, the text colour on it 4.5:1; register row A-19), with a selected choice's name in the text colour; the panel's own marks (the changed dot, a trace's banner and frames) read gui::look::mark_color, the host's link colour; the font cut carries the ascender, descender and line gap of egui's built-in font, so a row is as tall as the layout's own font made it; and no text is cut sideways in any tab at 1280 x 800 or 1024 x 700 in the app's font, which the app test no_text_is_cut_in_any_tab_in_the_app_s_own_font holds (the pole sweep's line is named in a wrapping caption over its plot, a plot readout longer than a row wraps, a tooltip's equation wider than the window is set smaller, and a plot may paint its axis label a quarter row into the centre region's margin)."
```

Replace:

```yaml
      - "Since BL-107 (UX overhaul slice B, stage 2, part 2b-iii-b; plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-b-the-look.md): the look's last three items (design section 6; the stage 1 review's E-4, R-6, S-1 and S-4). A control at rest shows its edge: app::theme::CONTROL (#656c76, the second colour outside the token table, register row A-52) is the resting outline (widgets.inactive.bg_stroke: a value box, a framed button, a picker, a text box) and the resting fill (inactive.bg_fill: a checkbox's box, a radio button, a slider's rail and handle), 3.3:1 on PANEL, 3.0:1 on PANEL2 (egui's menu_style drops the outline, so the Design menu's checkbox shows by its fill alone) and 3.5:1 on BG; hovered.bg_fill and active.bg_fill are CONTROL too, so a box keeps its fill and its outline brightens (MUTED, then TEXT); a slider's handle keeps its TEXT ring (fg_stroke) in every state and shows hover, drag and focus by the ring's width and its size alone (decision D-98). widgets.open.bg_stroke is CONTROL as well, so an open picker (egui's ComboBox draws its box in the open state) keeps its outline; menus drop it (menu_style). egui_plot's legend draws each entry's icon as a disc in the interact state's bg_fill and bg_stroke with the series dot on it while the series is shown, so the dot alone shows a line shown or hidden; on a CONTROL disc every coloured dot was 1.6 to 2.6:1 (BL-108), so gui::plots::plot_ui, in the scope where it sets the plot's visuals, sets widgets.inactive, hovered and active bg_fill to extreme_bg_color (the legend's own background): each dot sits on a BG disc inside the CONTROL ring (3.5:1 on BG), 5.7:1 (BAD's 'Does not fit') to 15:1 (TEXT), a hidden entry is the empty ring, and no control outside the plots changes. Frameless widgets (bar_button, tab_ui, menu items, an unselected selectable value) stay frameless; a selected one has SELECTION with the CONTROL edge. The tiles (tiles.rs) frame themselves on faint_bg_color with a 1-point outline in widgets.inactive.bg_stroke's colour at rest (CONTROL, 3.3:1 on PANEL; BL-109) and strong_text_color selected or focused, one width in both states so selecting a tile moves no row. The hairlines between the columns are no control and stay BORDER. app::fonts installs the whole FontDefinitions (set_fonts): Inter Regular first in the proportional family and Inter SemiBold first in FontFamily::Name(SEMIBOLD_NAME) (fonts::semibold()), egui's built-in fonts behind each, and points every style's TextStyle::Heading at the SemiBold family at its size, so the panel's two ui.heading texts (the title, the Size column's heading) are SemiBold and no other text is (the panel names no font; register row A-53). The SemiBold cut is made by assets/fonts/subset_inter.py from Inter 4.1's extras/ttf/Inter-SemiBold.ttf (hashes in its docstring) and holds the Regular cut's characters, digits and rows, so cut_holds stays the glyph guard's question. The page (linkage-sim-rs/web/tools/magcoupler/index.html) has lang en, theme-color #181b1f and color-scheme dark in its head and the line LOADING_LINE in MUTED (14 px system font: Inter is inside the wasm file), which app::remove_loading_line (eframe::App::update, wasm32 only) takes off the page at the end of each frame by LOADING_ID; eframe paints in the same task, so the line goes in the paint of the app's first frame (register row A-54). A wasm file that fails to load leaves the line on the page."
      - "Since BL-100: a scroll area follows the keyboard's focus. gui::follow_focus(ui), called last in a scroll area's content, scrolls that area by the least that shows whole (ui.scroll_to_rect(rect, None)) the widget the focus has arrived on, in the first frame the area is drawn with the focus there and never again (it keeps the focused id it saw last in egui's temporary memory, by the content ui's id), so a wheel or a drag of the scroll bar afterwards is not fought, nothing is selected or edited (BL-093) and an idle frame asks for no repaint. Tab, Shift+Tab, an Up or Down arrow key, a click and a request_focus are all arrivals. The widget is this area's when its layer is the area's and its interact_rect is its rect cut by the area's clip rect; a widget of another panel or of a list open over the area is not. Called by the left column and the Explain panel (panel.rs), the Geometry tab's callout list (geometry_view::list_ui) and the Clamp tab (clamp_drawing::clamp_ui). Not by an area nested in another (egui's ScrollArea takes both axes' scroll targets, so the outer area never hears of a widget of the inner one): the Equation panel's formula and term list, the Clamp tab's screw-size table; nor by the Results table, which draws only the rows in view (BL-101). egui's own Tab stops that show no focus are BL-102."
      - "Since BL-099: no row sets the left column's width. The column is 320 points (INPUTS_WIDTH) until its edge is dragged, in egui's default font and in the app's, with a fold, an input group or its Advanced heading opened, a filter set or an input traced. The head line of an input row and of an assumption is input_ui::head_line: the changed dot, then what the row adds (the label; for an input the note 'outside the slider range' and the reset button, each added to the line itself with ui.add_enabled, not in a child ui), on a wrapping line (gui::wrapping_line, one point narrower than the row: see BL-098 below), so a long label wraps onto a second line and the note and the button follow it. An input's label is drawn in that line as one block (input_ui::input_row: a top-down child ui as wide as the rest of the line and one text row high), so a wrapped label is one Tab stop that shows its focus on both lines, its second line starts under its first, and after it the note and the button start a line of their own at a widget's height (ui.end_row, ui.set_row_height); the app test an_input_label_that_wraps_is_one_tab_stop_in_either_font holds it (SK-1 of the fix wave of the re-review). A group's heading is drawn through panel::wrapped_title, a galley already wrapped at the header's width (a title with its '(N traced)' count is longer than the column in the app's font). The app tests the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font (PanelState and cut_texts on every tab, at 1024 x 700, 1280 x 800 and 1920 x 1080) and every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font hold it; a new kind of row must take no more than the width it is given (ui.min_rect() against the width, as the second test measures). The part and grade pickers fit their width by cutting their choice with an ellipsis (ComboBox::truncate), as before: register row A-45, owner question 68."
```

with:

```yaml
      - "Since BL-107 (UX overhaul slice B, stage 2, part 2b-iii-b; plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-b-the-look.md): the look's last three items (design section 6; the stage 1 review's E-4, R-6, S-1 and S-4). A control at rest shows its edge: app::theme::CONTROL (#656c76, the second colour outside the token table, register row A-52) is the resting outline (widgets.inactive.bg_stroke: a value box, a framed button, a picker, a text box) and the resting fill (inactive.bg_fill: a checkbox's box, a radio button, a slider's rail and handle), 3.3:1 on PANEL, 3.0:1 on PANEL2 (egui's menu_style drops the outline, so the Design menu's checkbox shows by its fill alone) and 3.5:1 on BG; hovered.bg_fill and active.bg_fill are CONTROL too, so a box keeps its fill and its outline brightens (MUTED, then TEXT); a slider's handle keeps its TEXT ring (fg_stroke) in every state and shows hover, drag and focus by the ring's width and its size alone (decision D-98). widgets.open.bg_stroke is CONTROL as well, so an open picker (egui's ComboBox draws its box in the open state) keeps its outline; menus drop it (menu_style). egui_plot's legend draws each entry's icon as a disc in the interact state's bg_fill and bg_stroke with the series dot on it while the series is shown, so the dot alone shows a line shown or hidden; on a CONTROL disc every coloured dot was 1.6 to 2.6:1 (BL-108), so gui::plots::plot_ui, in the scope where it sets the plot's visuals, sets widgets.inactive, hovered and active bg_fill to extreme_bg_color (the legend's own background): each dot sits on a BG disc inside the CONTROL ring (3.5:1 on BG), 5.7:1 (BAD's 'Does not fit') to 15:1 (TEXT), a hidden entry is the empty ring, and no control outside the plots changes. Frameless widgets (bar_button, tab_ui, menu items, an unselected selectable value) stay frameless; a selected one has SELECTION with the CONTROL edge. The tiles (tiles.rs) frame themselves on faint_bg_color with a 1-point outline in widgets.inactive.bg_stroke's colour at rest (CONTROL, 3.3:1 on PANEL; BL-109) and strong_text_color selected or focused, one width in both states so selecting a tile moves no row. The hairlines between the columns are no control and stay BORDER. app::fonts installs the whole FontDefinitions (set_fonts): Inter Regular first in the proportional family and Inter SemiBold first in FontFamily::Name(SEMIBOLD_NAME) (fonts::semibold()), egui's built-in fonts behind each, and points every style's TextStyle::Heading at the SemiBold family at its size, so the panel's two ui.heading texts (the title, the Size column's heading) are SemiBold and no other text is (the panel names no font; register row A-53). The SemiBold cut is made by assets/fonts/subset_inter.py from Inter 4.1's extras/ttf/Inter-SemiBold.ttf (hashes in its docstring) and holds the Regular cut's characters, digits and rows, so cut_holds stays the glyph guard's question. The page (linkage-sim-rs/web/tools/magcoupler/index.html) has lang en, theme-color #181b1f and color-scheme dark in its head and the line LOADING_LINE in MUTED (14 px system font: Inter is inside the wasm file), which app::remove_loading_line (eframe::App::update, wasm32 only) takes off the page at the end of each frame by LOADING_ID; eframe paints in the same task, so the line goes in the paint of the app's first frame (register row A-54). A wasm file that fails to load leaves the line on the page."
      - "Since BL-100: a scroll area follows the keyboard's focus. gui::follow_focus(ui), called last in a scroll area's content, scrolls that area by the least that shows whole (ui.scroll_to_rect(rect, None)) the widget the focus has arrived on, in the first frame the area is drawn with the focus there and never again (it keeps the focused id it saw last in egui's temporary memory, by the content ui's unique id: egui gives a scroll area's content its parent's id, which the check list and the Explain panel under it share, BL-101), so a wheel or a drag of the scroll bar afterwards is not fought, nothing is selected or edited (BL-093) and an idle frame asks for no repaint. Tab, Shift+Tab, an Up or Down arrow key, a click and a request_focus are all arrivals. The widget is this area's when its layer is the area's, its interact_rect is its rect cut by the area's clip rect and its rect lies in the area's content (ui.min_rect()); a widget of another panel or of a list open over the area is not. The content check is BL-101's: egui keeps the parent's clip on the axis an area does not scroll, so a sideways area nested in an up-and-down one is clipped up and down by the outer view, and without the check it took the outer area's widgets, read from the frame before, and scrolled the outer area a second time. It returns the rect it scrolled to. Called by the left column, the check list and the Explain panel (panel.rs), the Geometry tab's callout list (geometry_view::list_ui), the Clamp tab (clamp_drawing::clamp_ui) and the Equation panel (explorer::explorer_ui). An area nested in another (egui's ScrollArea takes both axes' scroll targets, so the outer area never hears of a widget of the inner one) is gui::sideways_scroll_area: it calls follow_focus last in its content and asks the outer area's content again for the rect once it has ended (the Equation panel's formula, the Clamp tab's screw-size table; the term list no longer scrolls sideways, BL-086). The Results table draws only the lines in view and is not followed (owner question 67's default): a line is a Tab stop only while the view shows it whole (results_table::line_sense, readouts::sensor's sense), so Tab passes the lines on screen and never stops on one half seen. egui's own Tab stops that show no focus are BL-102."
      - "Since BL-099: no row sets the left column's width. The column is 320 points (INPUTS_WIDTH) until its edge is dragged, in egui's default font and in the app's, with a fold, an input group or its Advanced heading opened, a filter set or an input traced. The head line of an input row and of an assumption is input_ui::head_line: the changed dot, then what the row adds (the label; for an input the note 'outside the slider range' and the reset button, each added to the line itself with ui.add_enabled, not in a child ui), on a wrapping line (gui::wrapping_line, one point narrower than the row: see BL-098 below), so a long label wraps onto a second line and the note and the button follow it. An input's label is drawn in that line as one block (input_ui::input_row: a top-down child ui as wide as the rest of the line and one text row high), so a wrapped label is one Tab stop that shows its focus on both lines, its second line starts under its first, and after it the note and the button start a line of their own at a widget's height (ui.end_row, ui.set_row_height); the app test an_input_label_that_wraps_is_one_tab_stop_in_either_font holds it (SK-1 of the fix wave of the re-review). A group's heading is drawn through panel::wrapped_title, a galley already wrapped at the header's width (a title with its '(N traced)' count is longer than the column in the app's font). The app tests the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font (PanelState and cut_texts on every tab, at 1024 x 700, 1280 x 800 and 1920 x 1080) and every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font hold it; a new kind of row must take no more than the width it is given (ui.min_rect() against the width, as the second test measures). The part and grade pickers fit their width by cutting their choice with an ellipsis (ComboBox::truncate), as before: register row A-45, owner question 68."
```

Replace:

```yaml
      - "Since BL-099: no row sets the left column's width. The column is 320 points (INPUTS_WIDTH) until its edge is dragged, in egui's default font and in the app's, with a fold, an input group or its Advanced heading opened, a filter set or an input traced. The head line of an input row and of an assumption is input_ui::head_line: the changed dot, then what the row adds (the label; for an input the note 'outside the slider range' and the reset button, each added to the line itself with ui.add_enabled, not in a child ui), on a wrapping line (gui::wrapping_line, one point narrower than the row: see BL-098 below), so a long label wraps onto a second line and the note and the button follow it. An input's label is drawn in that line as one block (input_ui::input_row: a top-down child ui as wide as the rest of the line and one text row high), so a wrapped label is one Tab stop that shows its focus on both lines, its second line starts under its first, and after it the note and the button start a line of their own at a widget's height (ui.end_row, ui.set_row_height); the app test an_input_label_that_wraps_is_one_tab_stop_in_either_font holds it (SK-1 of the fix wave of the re-review). A group's heading is drawn through panel::wrapped_title, a galley already wrapped at the header's width (a title with its '(N traced)' count is longer than the column in the app's font). The app tests the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font (PanelState and cut_texts on every tab, at 1024 x 700, 1280 x 800 and 1920 x 1080) and every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font hold it; a new kind of row must take no more than the width it is given (ui.min_rect() against the width, as the second test measures). The part and grade pickers fit their width by cutting their choice with an ellipsis (ComboBox::truncate), as before: register row A-45, owner question 68."
      - "Since BL-098: nothing the Explain panel shows sets the right column's width. The column is 300 points (DASHBOARD_WIDTH) until its edge is dragged, whichever entry is shown: the orientation, the note list, a note, a check, a tile, an input, a result. A note's diagram is DIAGRAM_SIZE (300 x 130) where its place is that wide and otherwise the same picture scaled to the width it is given (diagrams::diagram_ui draws the shapes at full size and scales them as one, labels and strokes too): 284 x 123 in the Explain panel. Every wrapping line of the two side columns is gui::wrapping_line, ui.horizontal_wrapped with one point of the width kept free: input_ui::head_line, size_controls::auto_ui and the panel's sizing_ui on the left; on the right the six of explain.rs (an entry's title line, a check link, the torque card's heading, the lesson line), explorer::change_ui and explains_ui (a note's change entries and its equation links) and dashboard::line_ui (a check's number row). A new wrapping line in either column goes through it too. The panel tests the_right_column_keeps_its_width_whichever_note_is_shown (both window sizes, egui's default font) and the_right_column_keeps_its_width_whatever_entry_is_shown_in_the_app_s_own_font (the app's font: every entry, on the starting design and on the example design with the details on), the app test the_right_column_keeps_its_width_with_a_note_shown_in_the_app_s_own_font (cut_texts on every tab) and diagrams' a_diagram_takes_the_width_it_is_given_scaled_as_one_picture hold it. The diagrams are drawn about 5 % smaller than before in that column: register row A-46, owner question 69."
      - "Since BL-094 (UX overhaul slice B, stage 2, part 2b-ii-b; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md): every result is a selection. A dimension tag (its click area the line and the tag's plate), a row of the callout list and a results-table row draw through Readouts::select or select_over: a click, or Enter or Space on the focused value, sets ReadoutEvents::selected (the focus arriving does not), the value is outlined while it has the focus or its result is the selection (Readouts::with_selection, as a tile and a check row are), the hover has no OPEN_HINT, and the panel sets explain::result_selection(path) (a check's own result selects the check, any other Selection::Result) and traces as a click on a result did; the Equation panel no longer opens on that click (spec 5.5), the entry's Equation link opens it. explain::result_ui draws any result: a tile's number and label, another's value and name as its table row draws them, its help, its dashboard row's check, the stored-3D line under Watch out for STORED_3D_ROWS, Full lesson where ROW_LEARN has its row, Equation where the registry has its equation, its cell with the details. The opening tab draws geometry_view::drawing_ui (the tags, and only the notes with a level); the Geometry tab keeps the callout list. The first view draws 200 words at 1280 x 800, held to 250 (goal G-1)."
```

with:

```yaml
      - "Since BL-099: no row sets the left column's width. The column is 320 points (INPUTS_WIDTH) until its edge is dragged, in egui's default font and in the app's, with a fold, an input group or its Advanced heading opened, a filter set or an input traced. The head line of an input row and of an assumption is input_ui::head_line: the changed dot, then what the row adds (the label; for an input the note 'outside the slider range' and the reset button, each added to the line itself with ui.add_enabled, not in a child ui), on a wrapping line (gui::wrapping_line, one point narrower than the row: see BL-098 below), so a long label wraps onto a second line and the note and the button follow it. An input's label is drawn in that line as one block (input_ui::input_row: a top-down child ui as wide as the rest of the line and one text row high), so a wrapped label is one Tab stop that shows its focus on both lines, its second line starts under its first, and after it the note and the button start a line of their own at a widget's height (ui.end_row, ui.set_row_height); the app test an_input_label_that_wraps_is_one_tab_stop_in_either_font holds it (SK-1 of the fix wave of the re-review). A group's heading is drawn through panel::wrapped_title, a galley already wrapped at the header's width (a title with its '(N traced)' count is longer than the column in the app's font). The app tests the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font (PanelState and cut_texts on every tab, at 1024 x 700, 1280 x 800 and 1920 x 1080) and every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font hold it; a new kind of row must take no more than the width it is given (ui.min_rect() against the width, as the second test measures). The part and grade pickers fit their width by cutting their choice with an ellipsis (ComboBox::truncate), as before: register row A-45, owner question 68."
      - "Since BL-098: nothing the Explain panel shows sets the right column's width. The column is 300 points (DASHBOARD_WIDTH) until its edge is dragged, whichever entry is shown: the orientation, the note list, a note, a check, a tile, an input, a result. A note's diagram is DIAGRAM_SIZE (300 x 130) where its place is that wide and otherwise the same picture scaled to the width it is given (diagrams::diagram_ui draws the shapes at full size and scales them as one, labels and strokes too): 284 x 123 in the Explain panel. Every wrapping line of the two side columns is gui::wrapping_line, ui.horizontal_wrapped with one point of the width kept free: input_ui::head_line, size_controls::auto_ui and the panel's sizing_ui on the left; on the right the six of explain.rs (an entry's title line, a check link, the torque card's heading, the lesson line), notes_view::change_ui and explains_ui (a note's change entries and its equation links) and dashboard::line_ui (a check's number row). A new wrapping line in either column goes through it too. The panel tests the_right_column_keeps_its_width_whichever_note_is_shown (both window sizes, egui's default font) and the_right_column_keeps_its_width_whatever_entry_is_shown_in_the_app_s_own_font (the app's font: every entry, on the starting design and on the example design with the details on), the app test the_right_column_keeps_its_width_with_a_note_shown_in_the_app_s_own_font (cut_texts on every tab) and diagrams' a_diagram_takes_the_width_it_is_given_scaled_as_one_picture hold it. The diagrams are drawn about 5 % smaller than before in that column: register row A-46, owner question 69."
      - "Since BL-094 (UX overhaul slice B, stage 2, part 2b-ii-b; plan docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md): every result is a selection. A dimension tag (its click area the line and the tag's plate), a row of the callout list and a results-table row draw through Readouts::select or select_over: a click, or Enter or Space on the focused value, sets ReadoutEvents::selected (the focus arriving does not), the value is outlined while it has the focus or its result is the selection (Readouts::with_selection, as a tile and a check row are), the hover has no OPEN_HINT, and the panel sets explain::result_selection(path) (a check's own result selects the check, any other Selection::Result) and traces as a click on a result did; the Equation panel no longer opens on that click (spec 5.5), the entry's Equation link opens it. explain::result_ui draws any result: a tile's number and label, another's value and name as its table row draws them, its help, its dashboard row's check, the stored-3D line under Watch out for STORED_3D_ROWS, Full lesson where ROW_LEARN has its row, Equation where the registry has its equation, its cell with the details. The opening tab draws geometry_view::drawing_ui (the tags, and only the notes with a level); the Geometry tab keeps the callout list. The first view draws 200 words at 1280 x 800, held to 250 (goal G-1)."
```

Replace:

```yaml
  - "egui 0.32: a ScrollArea that scrolls sideways still gives its content the viewport's width (content_max_size is the inner size), so wrapped labels inside keep wrapping and only content that does not wrap (a Grid row, a typeset formula) scrolls. A headless wheel-scroll test must put the pointer on a visible part of the area (a point past its clip rect hovers the next panel) and run several frames, since egui spreads a wheel step over frames (magcoupling explorer term list)."
  - "egui 0.32: a TopBottomPanel shown with show_inside only shrinks the parent's cursor. A widget laid out after it that forces a height still runs past the panel's top (a ScrollArea keeps at least min_scrolled_height, 64 points, unless lowered; egui_plot's Plot::height is also its min_size; an allocate_exact_size with a floor), and it paints over the panel with the parent's clip rect, which covers the panel, and takes its clicks there. Lay what follows the panel out in a child ui (new_child) whose max_rect and clip_rect (set_clip_rect, intersected with the parent's) are the space left, advance the parent by that space and never by the child's min_rect (an overflow the clip hides would grow a host egui::Window), and make each view honour available_height (magcoupling panel confined, the Equation panel). A headless test splits the frame's shapes at the panel's separator line (painted with the parent's painter after the panel's contents) and its widgets at the panel's resize handle (the panel id .with(\"__resize\"), registered after the contents); Frame::end registers a hover-only widget over its contents, so count only widgets whose sense is interactive (equation_panel_overlap)."
```

with:

```yaml
  - "egui 0.32: a ScrollArea that scrolls sideways still gives its content the viewport's width (content_max_size is the inner size), so wrapped labels inside keep wrapping and only content that does not wrap (a Grid row, a typeset formula) scrolls. A headless wheel-scroll test must put the pointer on a visible part of the area (a point past its clip rect hovers the next panel) and run several frames, since egui spreads a wheel step over frames (magcoupling explorer term list)."
  - "egui 0.32: a Grid cell is only as wide as its column was in the frame before (grid.rs available_rect; the last column alone takes the width left), so a wrapping label in an earlier column never widens past its first width; one in a child ui given a width (ui.scope, set_max_width) made each row of the magcoupling term list about 146 points tall in a test. To wrap a cell at a chosen width, lay its galley out at that width (WidgetText::into_galley) and add it as a Label (magcoupling explorer: a term's value, at most TERM_VALUE_SHARE of the term list's width, BL-086)."
  - "egui 0.32: a TopBottomPanel shown with show_inside only shrinks the parent's cursor. A widget laid out after it that forces a height still runs past the panel's top (a ScrollArea keeps at least min_scrolled_height, 64 points, unless lowered; egui_plot's Plot::height is also its min_size; an allocate_exact_size with a floor), and it paints over the panel with the parent's clip rect, which covers the panel, and takes its clicks there. Lay what follows the panel out in a child ui (new_child) whose max_rect and clip_rect (set_clip_rect, intersected with the parent's) are the space left, advance the parent by that space and never by the child's min_rect (an overflow the clip hides would grow a host egui::Window), and make each view honour available_height (magcoupling panel confined, the Equation panel). A headless test splits the frame's shapes at the panel's separator line (painted with the parent's painter after the panel's contents) and its widgets at the panel's resize handle (the panel id .with(\"__resize\"), registered after the contents); Frame::end registers a hover-only widget over its contents, so count only widgets whose sense is interactive (equation_panel_overlap)."
```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): mod.rs (follow_focus: a scroll area scrolls, by the least that shows it whole, to the widget the keyboard's focus arrives on; called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, BL-100; wrapping_line: ui.horizontal_wrapped with one point of the width kept free, for every wrapping line of the left column, of the Explain panel and of the check list's warning links, BL-099 and BL-098), checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), select and select_over (a click asks for the result in the Explain panel: ReadoutEvents::selected, since BL-094), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds; diagram_ui scales a diagram as one picture to a place narrower than DIAGRAM_SIZE, BL-098), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel, whose left column is the Size column on every tab since BL-096 (inputs_ui, size_column_ui: the size controls over the folds DETAILED_INPUTS and ASSUMPTIONS_FOLD, the trace's banner at its foot; no InputsView): new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5, which share one tab since BL-104 (CentreView::tab: plots::PLOTS for each plot; the tab opens the panel's plot field, the plot shown last; view_ui draws the picker of the five over a plot); Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused; CentreView::tab_row(clamps_paused): those views with each run under one tab's text kept once, the row centre_ui draws and the smoke prompt's guard in app.rs reads) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input; the label is one block in the head line, so a wrapped label is one Tab stop) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider (every slider's set-up; its value box's own text, the text egui's formatter wrote in it, is no typed value: typed_number, BL-097), head_line (a row's head line, the changed dot and then the label on a line that wraps at the row's width: an input row's and an assumption's, BL-099), RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); drawing_ui (the opening tab's: the tags, and only the notes with a level, since BL-094); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PLOTS, the centre's one tab of the five plots (decision C-9, BL-104); PlotKind, the choices of its picker, with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; X_TICK_SPACING and Y_TICK_SPACING, each axis's AxisHints label_spacing as one point, so a tick number is at full strength or not drawn (E-7, BL-104); no red but a sweep row that does not fit: the requirement lines and the pull-out point in the strong text colour, the governing limit and the time to it in VIOLET (C-5, BL-104); NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the left column, on every tab since BL-096; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), side_rect (a side panel's rect as egui keeps it: the panel's and the app's width tests), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas; LOADING_ID and LOADING_LINE, the page's loading line, which remove_loading_line (wasm32, called at the end of eframe::App::update) takes off the page, BL-107), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and the two colours outside it, SELECTION, the fill behind what is selected, and CONTROL, the edge of a control at rest (BL-107); visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark; luminance and contrast, the WCAG formulas, and hex, a colour as #rrggbb, tests only), src/app/fonts.rs (INTER and INTER_SEMIBOLD, the cuts of Inter Regular and SemiBold embedded from assets/fonts/; install sets the font definitions, Inter first in the proportional family and the SemiBold first in semibold(), FontFamily::Name(SEMIBOLD_NAME), egui's built-in fonts behind each, and points TextStyle::Heading at semibold(); cut_holds, tests only), assets/fonts/ (the two cuts, subset_inter.py that built them with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

with:

```yaml
  features:
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): mod.rs (follow_focus: a scroll area scrolls, by the least that shows it whole, to the widget the keyboard's focus arrives on, and returns its rect; called last in the content of the left column, the check list, the Explain panel, the Geometry tab's callout list, the Clamp tab and the Equation panel, BL-100 and BL-101; sideways_scroll_area: a sideways scroll area nested in an up-and-down one that follows the focus both ways, the Equation panel's formula and the screw-size table, BL-101; wrapping_line: ui.horizontal_wrapped with one point of the width kept free, for every wrapping line of the left column, of the Explain panel and of the check list's warning links, BL-099 and BL-098), checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), select and select_over (a click asks for the result in the Explain panel: ReadoutEvents::selected, since BL-094), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), notes_view.rs (BL-088, moved out of explorer.rs: note_ui and change_ui (a note drawn against a NoteView, returning the NoteLink clicked), explains_ui, held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), note_text_on; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM, EXPLAINS; its tests hold notes_for_review), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds; diagram_ui scales a diagram as one picture to a place narrower than DIAGRAM_SIZE, BL-098), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel, whose left column is the Size column on every tab since BL-096 (inputs_ui, size_column_ui: the size controls over the folds DETAILED_INPUTS and ASSUMPTIONS_FOLD, the trace's banner at its foot; no InputsView): new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5, which share one tab since BL-104 (CentreView::tab: plots::PLOTS for each plot; the tab opens the panel's plot field, the plot shown last; view_ui draws the picker of the five over a plot); Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused; CentreView::tab_row(clamps_paused): those views with each run under one tab's text kept once, the row centre_ui draws and the smoke prompt's guard in app.rs reads) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input; the label is one block in the head line, so a wrapped label is one Tab stop) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider (every slider's set-up; its value box's own text, the text egui's formatter wrote in it, is no typed value: typed_number, BL-097), head_line (a row's head line, the changed dot and then the label on a line that wraps at the row's width: an input row's and an assumption's, BL-099), RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); drawing_ui (the opening tab's: the tags, and only the notes with a level, since BL-094); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PLOTS, the centre's one tab of the five plots (decision C-9, BL-104); PlotKind, the choices of its picker, with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; X_TICK_SPACING and Y_TICK_SPACING, each axis's AxisHints label_spacing as one point, so a tick number is at full strength or not drawn (E-7, BL-104); no red but a sweep row that does not fit: the requirement lines and the pull-out point in the strong text colour, the governing limit and the time to it in VIOLET (C-5, BL-104); NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the left column, on every tab since BL-096; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), side_rect (a side panel's rect as egui keeps it: the panel's and the app's width tests), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas; LOADING_ID and LOADING_LINE, the page's loading line, which remove_loading_line (wasm32, called at the end of eframe::App::update) takes off the page, BL-107), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and the two colours outside it, SELECTION, the fill behind what is selected, and CONTROL, the edge of a control at rest (BL-107); visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark; luminance and contrast, the WCAG formulas, and hex, a colour as #rrggbb, tests only), src/app/fonts.rs (INTER and INTER_SEMIBOLD, the cuts of Inter Regular and SemiBold embedded from assets/fonts/; install sets the font definitions, Inter first in the proportional family and the SemiBold first in semibold(), FontFamily::Name(SEMIBOLD_NAME), egui's built-in fonts behind each, and points TextStyle::Heading at semibold(); cut_holds, tests only), assets/fonts/ (the two cuts, subset_inter.py that built them with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
  - "Magcoupler, the columns hold still (branch magcoupling/columns-hold-still, cut from main 739a732 on 2026-11-26; one commit per item, in_progress until the controller's reviews, gate and merge). BL-097: a value box left with nothing typed changes nothing. input_ui::slider hands egui's Slider one custom_parser (input_ui::typed_number): the box's own text is no typed value, so Tab, Shift+Tab, Enter or a click elsewhere with nothing typed leaves a size off its step (Auto turned off leaves 11.942 mm shown as 11.9) and a value outside its slider's range bit for bit, with no undo step, no selection and no check moved; a typed number, an arrow key and a drag edit as before. Typing the number the box already shows is no edit either (register row A-43, owner question 66: the conservative answer, not the owner's). Decision D-73's special case for the Minimum OD's slider is removed: its tests hold on the one rule and fail without it. BL-100: a scroll area follows the keyboard's focus. gui::follow_focus, called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, scrolls the area by the least that shows whole the widget the focus has arrived on (Tab, Shift+Tab, an Up or Down arrow key), once, so a wheel is not fought, nothing is selected and an idle frame asks for no repaint. Filed from it, not fixed: BL-101 (the Equation panel, the Clamp tab's screw-size table and the Results table do not follow the focus: nested scroll areas and rows drawn only in view; register row A-44, owner question 67 on whether Tab should walk every row of the Results table) and BL-102 (egui's own Tab stops that show no focus: each scroll area's drag surface and scroll bar, each column's resize edge). BL-099: the left column keeps its 320 points whatever it shows. The head line of an input row and of an assumption wraps (input_ui::head_line), so a long label takes a second line where it ran past the column's edge and widened it for good (330.1 points with the Detailed inputs open in the app's font, 443.1 with the last group open, 418.8 in egui's default font); a group's heading wraps where its traced count makes it longer than the column (panel::wrapped_title; found by the build: 336.6 points with the Required torque traced). The item's evidence blamed the part pickers; they fitted their width, and still cut their choice with an ellipsis (register row A-45, owner question 68: the conservative answer, not the owner's). BL-098: the right column keeps its 300 points whatever the Explain panel shows. A note's diagram was 300 points wide in a column that holds 284 (316 points from the first note with a diagram on); it is now the same picture scaled to the width it is given (diagrams::diagram_ui; register row A-46, owner question 69: about 5 % smaller there). The sweep of every entry (notes, checks, tiles, inputs, all results; both fonts, both designs, details off and on) found one more cause, a wrapping line's label a fraction of a point past the line (0.06 in two notes, 0.375 in one result's entry): every wrapping line of the two side columns is now gui::wrapping_line, which keeps a point free (input_ui::head_line calls it). The sweep found nothing else, so no item was filed from it. risk: mechanical. Fix wave of the controller's review (2026-11-27): BL-097's rule compares the typed number with the text egui's formatter wrote in the box (recorded through Slider::custom_formatter), not with every coarser rounding of the value, which for a tiny value (the vacuum permeability) threw a typed number away; the check list's material warning's Why link wraps (gui::wrapping_line), which widened the right column with no back iron, on main too; tests added for follow_focus's clip check (a wheeled column holds still while the focus is elsewhere) and for the left column's width with Auto on; the grade pickers' blank choice is newly cut in the app's font (A-45 and owner question 68 corrected); BL-103 filed (the widths at pixel ratios 0.9 and 1.35). Fix wave of the re-review (2026-11-27): an input's label is drawn as one block in its head line, so a label that wraps is one Tab stop that shows its focus (SK-1: egui lays a label out in a wrapping line row by row, each row a Tab stop, and only the first showed the focus; 16 input labels wrap at the narrowest width the column gives a row in egui's default font; new with BL-099's wrapping head line); after a wrapped label the note and the reset button take a line of their own; the hold-still test's comment reads its own 24.5 points (RF-1)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-b (BL-107, plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-b-the-look.md, decisions D-95 to D-105): the look's last three items, in one plan (D-95). The controls' edges: one more colour outside the token table, app::theme::CONTROL #656c76 (D-96, register row A-52, owner question 73), the darkest of the tokens' hue at 3:1 on a menu (the 2b-iii-a plan's 'about #626973' is 2.9:1 there); framed controls take it, an open picker keeps it, egui_plot draws every legend entry in a ring of it, the dot on a disc in the field colour (BL-108, fixed on the branch before its release: on a disc of the grey every coloured dot was 1.6 to 2.6:1, the review's LOOK-1), and the four tiles over the drawing take it as their resting outline (BL-109, LOOK-2, fixed on the branch too); frameless ones and the hairlines between the columns stay as they were (D-97); a box and a handle keep their fill under the pointer and pressed, and a handle still shows its state by its ring's width alone (D-98). A SemiBold face for egui's Heading style: the title and the Size column's heading, no other text, tile numbers Regular (D-99, A-53, question 74), installed with the whole font definitions so its family keeps the built-in fonts behind it (D-100). The page shell: lang, theme-color, color-scheme and the line 'Loading the calculator' in the muted colour (D-101, A-54, question 75), removed by the app at the end of its first frame, where the page's script removing it on init() left a blank frame in Chrome (D-102); a wasm file that fails to load leaves the line (D-103). gui-smoke's look check reads the line gone and judges the heavier title and the controls' edges, no new check (D-104). risk: mechanical (D-105). Next: part 2b-iii-c (the Equation panel: BL-086, BL-088, BL-089, BL-101's Equation panel part)."
```

with:

```yaml
  - "Magcoupler, the columns hold still (branch magcoupling/columns-hold-still, cut from main 739a732 on 2026-11-26; one commit per item, in_progress until the controller's reviews, gate and merge). BL-097: a value box left with nothing typed changes nothing. input_ui::slider hands egui's Slider one custom_parser (input_ui::typed_number): the box's own text is no typed value, so Tab, Shift+Tab, Enter or a click elsewhere with nothing typed leaves a size off its step (Auto turned off leaves 11.942 mm shown as 11.9) and a value outside its slider's range bit for bit, with no undo step, no selection and no check moved; a typed number, an arrow key and a drag edit as before. Typing the number the box already shows is no edit either (register row A-43, owner question 66: the conservative answer, not the owner's). Decision D-73's special case for the Minimum OD's slider is removed: its tests hold on the one rule and fail without it. BL-100: a scroll area follows the keyboard's focus. gui::follow_focus, called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, scrolls the area by the least that shows whole the widget the focus has arrived on (Tab, Shift+Tab, an Up or Down arrow key), once, so a wheel is not fought, nothing is selected and an idle frame asks for no repaint. Filed from it, not fixed: BL-101 (the Equation panel, the Clamp tab's screw-size table and the Results table do not follow the focus: nested scroll areas and rows drawn only in view; register row A-44, owner question 67 on whether Tab should walk every row of the Results table) and BL-102 (egui's own Tab stops that show no focus: each scroll area's drag surface and scroll bar, each column's resize edge). BL-099: the left column keeps its 320 points whatever it shows. The head line of an input row and of an assumption wraps (input_ui::head_line), so a long label takes a second line where it ran past the column's edge and widened it for good (330.1 points with the Detailed inputs open in the app's font, 443.1 with the last group open, 418.8 in egui's default font); a group's heading wraps where its traced count makes it longer than the column (panel::wrapped_title; found by the build: 336.6 points with the Required torque traced). The item's evidence blamed the part pickers; they fitted their width, and still cut their choice with an ellipsis (register row A-45, owner question 68: the conservative answer, not the owner's). BL-098: the right column keeps its 300 points whatever the Explain panel shows. A note's diagram was 300 points wide in a column that holds 284 (316 points from the first note with a diagram on); it is now the same picture scaled to the width it is given (diagrams::diagram_ui; register row A-46, owner question 69: about 5 % smaller there). The sweep of every entry (notes, checks, tiles, inputs, all results; both fonts, both designs, details off and on) found one more cause, a wrapping line's label a fraction of a point past the line (0.06 in two notes, 0.375 in one result's entry): every wrapping line of the two side columns is now gui::wrapping_line, which keeps a point free (input_ui::head_line calls it). The sweep found nothing else, so no item was filed from it. risk: mechanical. Fix wave of the controller's review (2026-11-27): BL-097's rule compares the typed number with the text egui's formatter wrote in the box (recorded through Slider::custom_formatter), not with every coarser rounding of the value, which for a tiny value (the vacuum permeability) threw a typed number away; the check list's material warning's Why link wraps (gui::wrapping_line), which widened the right column with no back iron, on main too; tests added for follow_focus's clip check (a wheeled column holds still while the focus is elsewhere) and for the left column's width with Auto on; the grade pickers' blank choice is newly cut in the app's font (A-45 and owner question 68 corrected); BL-103 filed (the widths at pixel ratios 0.9 and 1.35). Fix wave of the re-review (2026-11-27): an input's label is drawn as one block in its head line, so a label that wraps is one Tab stop that shows its focus (SK-1: egui lays a label out in a wrapping line row by row, each row a Tab stop, and only the first showed the focus; 16 input labels wrap at the narrowest width the column gives a row in egui's default font; new with BL-099's wrapping head line); after a wrapped label the note and the reset button take a line of their own; the hold-still test's comment reads its own 24.5 points (RF-1)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-c (BL-112, plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, decisions D-106 to D-115): the Equation panel, and slice B's last part, in one plan (D-106). BL-088: the note drawing, its strings and tests in gui/notes_view.rs, a pure move; the sign-off sheet stays in explorer.rs, whose name its text gives (D-107). BL-089: EMPTY_TEXT gone, a panel open with no trail draws its header row only (D-108, register row A-56). BL-086: a term's name and tag wrap in the term grid's last column, the terms' sideways scroll area gone (D-110), a term's value wraps past a third of the list's width (explorer::TERM_VALUE_SHARE, laid out as a galley: a grid cell is only as wide as its column was the frame before), the walk bar's step line wraps (D-111); the formula alone still scrolls sideways (D-109, register row A-55, owner question 76). BL-101: follow_focus returns its rect and checks the widget lies in the area's content; gui::sideways_scroll_area for the formula and the screw-size table; the Equation panel's own area follows (D-112); a Results table line takes the focus only while whole (D-113, A-44's default kept, question 67); the check list follows, follow_focus's slot keyed by the content's unique id (D-114). risk: mechanical (D-115). Slice B is complete with it."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-b (BL-107, plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-b-the-look.md, decisions D-95 to D-105): the look's last three items, in one plan (D-95). The controls' edges: one more colour outside the token table, app::theme::CONTROL #656c76 (D-96, register row A-52, owner question 73), the darkest of the tokens' hue at 3:1 on a menu (the 2b-iii-a plan's 'about #626973' is 2.9:1 there); framed controls take it, an open picker keeps it, egui_plot draws every legend entry in a ring of it, the dot on a disc in the field colour (BL-108, fixed on the branch before its release: on a disc of the grey every coloured dot was 1.6 to 2.6:1, the review's LOOK-1), and the four tiles over the drawing take it as their resting outline (BL-109, LOOK-2, fixed on the branch too); frameless ones and the hairlines between the columns stay as they were (D-97); a box and a handle keep their fill under the pointer and pressed, and a handle still shows its state by its ring's width alone (D-98). A SemiBold face for egui's Heading style: the title and the Size column's heading, no other text, tile numbers Regular (D-99, A-53, question 74), installed with the whole font definitions so its family keeps the built-in fonts behind it (D-100). The page shell: lang, theme-color, color-scheme and the line 'Loading the calculator' in the muted colour (D-101, A-54, question 75), removed by the app at the end of its first frame, where the page's script removing it on init() left a blank frame in Chrome (D-102); a wasm file that fails to load leaves the line (D-103). gui-smoke's look check reads the line gone and judges the heavier title and the controls' edges, no new check (D-104). risk: mechanical (D-105). Next: part 2b-iii-c (the Equation panel: BL-086, BL-088, BL-089, BL-101's Equation panel part)."
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown

---

## 2026-11-27 - Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-b: the look (BL-107)
```

with:

```markdown

---

## 2026-11-28 - Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-c: the Equation panel (BL-112)
- What changed: walking Start here at 1024 x 700 no term name, tag or step line runs past the Equation panel's edge (BL-086): a term's name and what it is share the term grid's last column and wrap in the panel's width, a long text value wraps in its own column, and the walk bar's step line wraps; the panel no longer grows past the centre region, on any equation. The equation itself still scrolls sideways where it is wider than the panel (decision D-109, owner question 76). The note drawing is `gui/notes_view.rs`, moved whole out of `gui/explorer.rs` (BL-088). The panel's empty text is gone (BL-089). The keyboard's focus is followed in the Equation panel, its formula and the Clamp tab's screw-size table (`gui::sideways_scroll_area`), and in the check list; a Results table line is a Tab stop only while its view shows it whole (BL-101, owner question 67's default).
- Tests: every text of the Equation panel but its formula whole on each Start here step at the three window sizes, the panel inside the centre region (app); the same on every equation the registry holds, in the app's font at 1024 points wide (panel); a panel open with no trail draws its header row only; σ's tag wraps inside a 1000-point window; the keyboard walks through the Equation panel (two equations, the panel at its starting and its least height), the open screw-size table, the Results tab (as it opens and wheeled) and the check list (panel).
- Docs: README, docs/ai 02, 03, 04, backlog BL-086, BL-088, BL-089, BL-101 and BL-112 in progress, register rows A-44 (its cost), A-55 and A-56, owner questions 67 (its text) and 76, the design's header.

## 2026-11-27 - Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-b: the look (BL-107)
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  priority: 4
  status: open
  notes: "Filed 2026-11-23 from the BL-085 fix wave; the doc comment on EMPTY_TEXT now says it is only a fallback."
- id: BL-088
```

with:

```yaml
  priority: 4
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Task 2): EMPTY_TEXT, its test's lines and its row in the strings sheet go; a panel open with no trail draws its header row only, and no way to open it empty comes back (decision D-108, register row A-56). Filed 2026-11-23 from the BL-085 fix wave; the doc comment on EMPTY_TEXT now says it is only a fallback."
- id: BL-088
```

Replace:

```yaml
  priority: 4
  status: open
  notes: "Filed 2026-11-23 from the BL-085 fix wave for part 2b; nothing is broken, the module docs say where the code lives."
- id: BL-087
```

with:

```yaml
  priority: 4
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Task 1): the note drawing, its strings and its tests moved whole to gui/notes_view.rs; the sign-off sheet stays in explorer.rs (decision D-107). Filed 2026-11-23 from the BL-085 fix wave for part 2b; nothing is broken, the module docs say where the code lives."
- id: BL-087
```

Replace:

```yaml
  priority: 3
  status: open
  notes: "Filed 2026-11-22 from the BL-085 plan review (UXP-2); not built with BL-085, whose own new line (the lesson line) is drawn whole and tested (UXP-1)."
- id: BL-081
```

with:

```yaml
  priority: 3
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Task 3): a term's name and tag wrap in the term grid's last column, a term's value wraps past a third of the list's width, and the walk bar's step line wraps (decisions D-110, D-111); the formula alone still scrolls sideways (D-109, owner question 76); the app test widened to every text of the panel but the formula, renamed every_text_of_the_equation_panel_is_whole_on_each_start_here_step_in_the_app_s_own_font, and every_text_of_the_equation_panel_is_whole_on_every_equation_in_the_app_s_own_font opens every equation in a window 1024 points wide (the plan review's UXT-1: a long text value laid out on one line squeezed the name and tag column to about 20 points). Filed 2026-11-22 from the BL-085 plan review (UXP-2); not built with BL-085, whose own new line (the lesson line) is drawn whole and tested (UXP-1)."
- id: BL-081
```

Replace:

```yaml
  priority: 3
  status: open
  notes: "Filed 2026-11-26 by the BL-100 build. On main before BL-100 (no scroll area followed the focus). A way in for the nested areas: follow_focus returns the rect it scrolled to, and the outer area's content scrolls to it again after the inner area has ended. Register row A-44 and owner question 67 hold the Results table's choice. An area added later needs its own call of follow_focus: nothing fails when one is forgotten."
- id: BL-102
```

with:

```yaml
  priority: 3
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Tasks 4 and 5): all four parts; the Results table keeps owner question 67's default and takes the focus only on a whole line (decisions D-112 to D-114). Filed 2026-11-26 by the BL-100 build. On main before BL-100 (no scroll area followed the focus). A way in for the nested areas: follow_focus returns the rect it scrolled to, and the outer area's content scrolls to it again after the inner area has ended. Register row A-44 and owner question 67 hold the Results table's choice. An area added later needs its own call of follow_focus: nothing fails when one is forgotten."
- id: BL-102
```

Replace:

```yaml
  priority: 2
  status: open
  notes: "Filed 2026-11-28 with its plan (the design's header: each plan files one item). Part 2b-iii's third and last part (decision D-86, register row A-47); slice B is built with it. It takes BL-086, BL-088, BL-089 and BL-101 whole (BL-101's four parts: the Equation panel, the screw-size table, the Results table and the check list). Not cut again (decision D-106). risk: mechanical (decision D-115): no new number, no physical claim and no new on-screen string. BL-083, BL-084, BL-087, BL-067, BL-095, BL-102, BL-103, BL-105, BL-106, BL-110 and BL-111 are not touched."

```

with:

```yaml
  priority: 2
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-c (Tasks 1 to 5 of the plan); docs swept (Task 6). Filed 2026-11-28 with its plan (the design's header: each plan files one item). Part 2b-iii's third and last part (decision D-86, register row A-47); slice B is built with it. It takes BL-086, BL-088, BL-089 and BL-101 whole (BL-101's four parts: the Equation panel, the screw-size table, the Results table and the check list). Not cut again (decision D-106). risk: mechanical (decision D-115): no new number, no physical claim and no new on-screen string. BL-083, BL-084, BL-087, BL-067, BL-095, BL-102, BL-103, BL-105, BL-106, BL-110 and BL-111 are not touched."

```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
| A-43 | BL-097: a value box left with nothing typed no longer rewrites its value. The rule reads the box's text when the focus leaves it, and the box's own text and the same number typed over it are the same text. Does typing the number a box already shows round a value off its slider's step to it (a magnet length of 11.942 mm shown as 11.9)? | No: it changes nothing, as leaving the box alone does. Any other number typed, an arrow key or a drag sets the value, on the step. | No number changes without an entry that differs from what is on screen, and the rule keeps no state between frames (a flag for a typed key would lose a value typed and entered inside one frame). The cost: a person who wants exactly the number shown must nudge the slider away and back. Owner question 66. | The box records whether a key was typed in it, and the shown number typed again rounds the value to it. | `magcoupling-rs/src/gui/input_ui.rs` (`typed_number`), BL-097 |
| A-44 | BL-100: a scroll area now follows the keyboard's focus, but not the Results table. The table draws only the rows in view, so only they take the focus: Tab passes the rows on screen (a row half cut at the foot among them) and moves on, and the rows below are reached by the wheel or the search alone. If the table followed the focus, each Tab would bring the next row into view, and Tab would walk every row, several hundred, before it left the table. Which? | The table is left as it is: Tab passes the rows in view. | It changes nothing a person relies on today, and Tab stays a short way across the page. The cost: a row cut at the foot can hold the focus half seen, and the lower rows cannot be reached by keyboard alone. Owner question 67. | The table follows the focus (Tab walks its rows), or its rows leave the Tab order and the arrow keys move through them. | `docs/ai/backlog.yaml`, BL-101 |
| A-45 | BL-099: the left column now keeps its 320 points, so the part and grade pickers always show their choice on one line, cut with an ellipsis where it does not fit ("B842SH: K&J block 12.70 × 6.350 × 3.170 mm, N42SH" is 245 points wide and its box holds about 180). They did so at 320 points before; only a column that had grown showed the choice whole. Should the whole choice show? | No change: one line, cut as before. | No new on-screen text and no changed layout. A part's whole label is the first of the lines under its picker, and a grade's values are in the lines under its own; the column's edge can be dragged wider. The cost: the grade's one-line label is not shown whole anywhere in a 320-point column, and in the app's font one text that was whole is now cut: a grade picker's blank choice, "Blank: the manual Br, no rating" (180.7 points in a 168-point box), which on main showed whole because the column had grown to 330.1 points before that row could be drawn; the hint line above the picker reads the same words whole. Owner question 68. | A hover on the picker shows its whole choice, or the picker wraps it onto two lines. | `magcoupling-rs/src/gui/pickers.rs`, BL-099 |
```

with:

```markdown
| A-43 | BL-097: a value box left with nothing typed no longer rewrites its value. The rule reads the box's text when the focus leaves it, and the box's own text and the same number typed over it are the same text. Does typing the number a box already shows round a value off its slider's step to it (a magnet length of 11.942 mm shown as 11.9)? | No: it changes nothing, as leaving the box alone does. Any other number typed, an arrow key or a drag sets the value, on the step. | No number changes without an entry that differs from what is on screen, and the rule keeps no state between frames (a flag for a typed key would lose a value typed and entered inside one frame). The cost: a person who wants exactly the number shown must nudge the slider away and back. Owner question 66. | The box records whether a key was typed in it, and the shown number typed again rounds the value to it. | `magcoupling-rs/src/gui/input_ui.rs` (`typed_number`), BL-097 |
| A-44 | BL-100: a scroll area now follows the keyboard's focus, but not the Results table. The table draws only the rows in view, so only they take the focus: Tab passes the rows on screen (a row half cut at the foot among them) and moves on, and the rows below are reached by the wheel or the search alone. If the table followed the focus, each Tab would bring the next row into view, and Tab would walk every row, several hundred, before it left the table. Which? | The table is left as it is: Tab passes the rows in view. | It changes nothing a person relies on today, and Tab stays a short way across the page. The cost: the lower rows cannot be reached by keyboard alone. Since BL-101 a line the view cuts, or one drawn past its foot, is no Tab stop, so the focus never sits on a row half seen; a focused line a wheel then cuts loses the focus (it lost it before once the wheel took it out of view). Owner question 67. | The table follows the focus (Tab walks its rows), or its rows leave the Tab order and the arrow keys move through them. | `docs/ai/backlog.yaml`, BL-101 |
| A-45 | BL-099: the left column now keeps its 320 points, so the part and grade pickers always show their choice on one line, cut with an ellipsis where it does not fit ("B842SH: K&J block 12.70 × 6.350 × 3.170 mm, N42SH" is 245 points wide and its box holds about 180). They did so at 320 points before; only a column that had grown showed the choice whole. Should the whole choice show? | No change: one line, cut as before. | No new on-screen text and no changed layout. A part's whole label is the first of the lines under its picker, and a grade's values are in the lines under its own; the column's edge can be dragged wider. The cost: the grade's one-line label is not shown whole anywhere in a 320-point column, and in the app's font one text that was whole is now cut: a grade picker's blank choice, "Blank: the manual Br, no rating" (180.7 points in a 168-point box), which on main showed whole because the column had grown to 330.1 points before that row could be drawn; the hint line above the picker reads the same words whole. Owner question 68. | A hover on the picker shows its whole choice, or the picker wraps it onto two lines. | `magcoupling-rs/src/gui/pickers.rs`, BL-099 |
```

Replace:

```markdown
| A-54 | S-4 asks for one loading line while the wasm file loads. Which words, and how does it go? | "Loading the calculator", the review's own words, in the muted colour in a 14 px system font across the middle of the charcoal page; the app takes it off the page at the end of its first frame, so it goes in the paint that shows the app. A wasm file that fails to load leaves the line. | The fewest new words that say the page is working; no colour new to the page; the line is gone exactly when the app is there (a browser run measured a blank frame when the page's script removed it). No error text: a new string no one asked for. Owner question 75. | Other words ("Loading…"), a spinner, or no line; an error line when the wasm file fails. | The plan of part 2b-iii-b (BL-107), decisions D-101 to D-103 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
```

with:

```markdown
| A-54 | S-4 asks for one loading line while the wasm file loads. Which words, and how does it go? | "Loading the calculator", the review's own words, in the muted colour in a 14 px system font across the middle of the charcoal page; the app takes it off the page at the end of its first frame, so it goes in the paint that shows the app. A wasm file that fails to load leaves the line. | The fewest new words that say the page is working; no colour new to the page; the line is gone exactly when the app is there (a browser run measured a blank frame when the page's script removed it). No error text: a new string no one asked for. Owner question 75. | Other words ("Loading…"), a spinner, or no line; an error line when the wasm file fails. | The plan of part 2b-iii-b (BL-107), decisions D-101 to D-103 |
| A-55 | BL-086 asks for every text of the Equation panel inside its width. An equation does not wrap, and on the Start here walk the formula runs past the panel's edge on five of the 14 steps at 1024 x 700 (up to 680 points past it) and three at 1280 x 800, inside its own sideways scroll area. Set it smaller to fit, or keep it scrolling? | It keeps scrolling sideways, at its 22 points, as before; every other text of the panel wraps. A keyboard user's Tab shows it from its start (BL-101). | The equation is as it was; set smaller to fit, the widest walk step's equation (the ferrite check at 1024 x 700) would need about 7.6 points, below the 8 the tooltips' equations stop at. The cost: on those steps the end of the equation is out of view until the panel is scrolled sideways (a scroll bar shows under the pointer) or widened. Owner question 76. | The equation set smaller to fit the panel, down to a floor (the tooltips' fitting, `typeset::equation_ui`), scrolling only below it. | The plan of part 2b-iii-c (BL-112), decision D-109 |
| A-56 | BL-089: since the panel's toggle went, every way of opening the Equation panel sets a trail first, so its empty text (how to open an equation, and Start here) is drawn in no state the app reaches. The item leaves the choice open: the text, its test and its guard rows go, or a way to open the panel empty comes back. Which? | The text goes, with its test's lines and its row in the strings sheet; a panel open with no trail draws its header row (the heading, Close and Start here) and nothing under it. No way to open the panel empty comes back. | No on-screen text is added, and none a reader sees is lost: no state of the app drew it, and the Explain panel's orientation and its Start here link carry its guidance. Undone by putting the string and its draw back. No owner question: the builder's call, not the owner's answer. | A way to open the panel empty comes back (a header toggle, as before part 2a), and the empty text with it, worded again and checked by the text review. | The plan of part 2b-iii-c (BL-112), decision D-108 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
```

Replace:

```markdown
- 2026-11-27: A-52's answer updated by BL-108 and BL-109, fixed on the branch of BL-107 before its release: a plot legend's dot sits on a disc in the field colour inside the grey ring (the dots 5.7:1 to 15:1, the ring 3.5:1 on the legend), and the four tiles over the drawing take the grey as their resting outline (3.3:1 on the panel). It is the builder's call, not the owner's answer.
- 2026-11-23: A-33, A-34, A-35 and A-36 added with the plan of slice B's stage 2, part 2b-ii-b (BL-094). Each is an assumption, not the owner's answer.
```

with:

```markdown
- 2026-11-27: A-52's answer updated by BL-108 and BL-109, fixed on the branch of BL-107 before its release: a plot legend's dot sits on a disc in the field colour inside the grey ring (the dots 5.7:1 to 15:1, the ring 3.5:1 on the legend), and the four tiles over the drawing take the grey as their resting outline (3.3:1 on the panel). It is the builder's call, not the owner's answer.
- 2026-11-28: A-55 added with the plan of slice B's stage 2, part 2b-iii-c (BL-112). It is an assumption, not the owner's answer.
- 2026-11-28: A-44's cost updated by the same plan (BL-101): a Results table line the view cuts is no Tab stop. The answer is unchanged.
- 2026-11-28: A-56 added by the fix wave of the same plan's review (UXT-2): BL-089's open choice, the Equation panel's empty text goes and nothing replaces it (decision D-108). It is the builder's call, not the owner's answer.
- 2026-11-23: A-33, A-34, A-35 and A-36 added with the plan of slice B's stage 2, part 2b-ii-b (BL-094). Each is an assumption, not the owner's answer.
```

**`docs/analyses/2026-11-18-owner-questions.md`**

Replace:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 56 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54, 55, 56, 57, 58 and 59 on 2026-11-23, questions 60 to 65 on 2026-11-25, questions 66 to 69 on 2026-11-26, questions 70 to 75 on 2026-11-27). Part D, five that matter only if the tools stop being generic.

```

with:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 57 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54, 55, 56, 57, 58 and 59 on 2026-11-23, questions 60 to 65 on 2026-11-25, questions 66 to 69 on 2026-11-26, questions 70 to 75 on 2026-11-27, question 76 on 2026-11-28). Part D, five that matter only if the tools stop being generic.

```

Replace:

```markdown

Added 2026-11-26 (the next free number). The columns now scroll to whatever the Tab key lands on, so the keyboard's focus is never out of sight there. The Results table is left out. It draws only the rows on screen, so Tab passes those and moves on; the rows below are reached with the wheel or the search box. If the table scrolled to the focus too, each Tab would bring up the next row, and Tab would go through all of its rows, several hundred, before it left the table.

```

with:

```markdown

Added 2026-11-26 (the next free number). The columns now scroll to whatever the Tab key lands on, so the keyboard's focus is never out of sight there. The Results table is left out. It draws only the rows on screen, so Tab passes those and moves on; the rows below are reached with the wheel or the search box. A row the table's edge cuts is no longer a Tab stop, so the focus never sits on a row half seen. If the table scrolled to the focus too, each Tab would bring up the next row, and Tab would go through all of its rows, several hundred, before it left the table.

```

Replace:

```markdown

### Linkage simulator
```

with:

```markdown

### 76. Should an equation wider than the Equation panel be set smaller to fit?

Added 2026-11-28 (the next free number). Every text of the Equation panel now wraps inside it: a term's name and what it is, and the Start here bar's step line, which ran past its edge in the smallest window. The equation itself is one line of typeset maths and does not wrap. On the Start here walk it is wider than the panel on five of the 14 steps in the smallest window, 1024 x 700, and on three at 1280 x 800; it scrolls sideways, as it always did, with a scroll bar under the pointer. Set smaller to fit, the widest (the ferrite magnets' cold check) would need about 7.6 points in the smallest window, below the 8 points the hover's equations stop at.

- **If you do not answer:** the equation keeps its size and scrolls sideways.
- (a) As the default.
- (b) An equation too wide is set smaller to fit, down to a smallest size, and scrolls only past it.

Source: BL-086; A-55; BL-112 plan, decision D-109.

### Linkage simulator
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, its second (the drawing's selections and the word budget; BL-094) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md`, its third (the left column and the nothing-lost test; BL-096) by the plan `docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md`; part 2b-iii is built in three, and its first (the Plots tab; BL-104) by the plan `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md`, its second (the look; BL-107) by the plan `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-b-the-look.md`; part 2b-iii-c and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

with:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, its second (the drawing's selections and the word budget; BL-094) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md`, its third (the left column and the nothing-lost test; BL-096) by the plan `docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md`; part 2b-iii is built in three, and its first (the Plots tab; BL-104) by the plan `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md`, its second (the look; BL-107) by the plan `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-b-the-look.md`, its third (the Equation panel; BL-112) by the plan `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md`, which completes slice B; slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

**`magcoupling-rs/README.md`**

Replace:

```markdown
| `src/engine/explain/records/shaft_joint.rs` | BL-064 shaft joint records: one per numeric `shaft_joint.*` result (23; the verdict, a text, has none), transcribed from `shaft_joint::compute` and `PartProperties::hub` with every correction on: the part material in effect as cases over the circuit and the back-iron pick (the Materials inputs for the default pick, a pick's library values, else the workbook's 6061-T6), each joint's Lamé rows (holding, bore stress, needed minimum, elastic edge) with a no-wall arm that gives NaN, the key alone (0 with no keyway, +inf with no bearing length or contact), each shaft's shrink as cases on 265 K with NIST's Fe-2.25Ni polynomial, the fitting temperature and the fittable edge, and the bonded inner hub's allowed fitting temperature as the lowest of your limit, the adhesive's design limit and the inner ring's own single-ring onset less 10 °C, NaN when any of the three is (owner R-15) |
| `src/gui/` | Feature `gui`: `mod.rs` (`follow_focus`: a scroll area scrolls, by the least that shows it whole, to the widget the keyboard's focus arrives on, BL-100; `wrapping_line`: a line that wraps with one point of its width kept free, for every wrapping line of the left column, of the Explain panel and of the check list's warning links, BL-099 and BL-098), `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, the left column (since BL-096 the Size column on every tab, with its folds `DETAILED_INPUTS` and `ASSUMPTIONS_FOLD`), the assumptions banner, the sizing controls, the right column (`checks_and_explain_ui`: the check list over the Explain panel, one `explain::Selection`, the Hot-low entry's consequences asked of the `ConsequenceCache` through `size_tab_consequences`, the Size tab's one ask, with the design's own results computed once per design (`ResultsCache`))), `tiles.rs` (since BL-081, the four tiles over the drawing on Size and torque: `Tile`, `tiles_ui`, `MIN_TILE_WIDTH`, the Hot-low tile's `needs_lines` and `gap_pieces`), `checks.rs` (since BL-081, the check list: `checks_ui`, `state_mark`, `mark_level`, `CHECKS_SHARE`), `explain.rs` (since BL-081, the Explain panel: `explain_ui`, `Selection`, `Asked`, the orientation and each entry; the torque card sits in the Hot-low tile's entry, `card_ui`, decision C-10; an entry's "Full lesson:" link is `guidance::row_learn`'s), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::new(marks, show_details)`, `show_details`, `Readouts::show` and `show_over` (the hover text built with the frame's `show_details`), `select` and `select_over` (a click selects the result, since BL-094; outlined while focused or selected, `with_selection`), `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `note_ui` (a note drawn against a `NoteView`, returning the `NoteLink` clicked), `held_reason`, `term_tag`, `reviewed_note`, `note_fallback` (Explain's next best read under an equation without a note, `NoteFallback`), `start_here_text` with `FERRITE_ONLY` on the ferrite step; its heights: `panel_heights`, `PANEL_SHARE`, `MIN_PANEL_HEIGHT`, `VIEW_STRIP`), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`, which scales a diagram to a place narrower than its 300 points, BL-098), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `plain_text.rs` (the display-name and help overlay over the engine's metadata, keyed by path: `label_for`, `help_for`, `original_help`, `hint_for`, `display_label`; `LABELS` holds spec 3.5's canonical names, three of them constants the Size tab and the plots draw too: `REQUIRED_TORQUE`, `PULL_OUT_NOMINAL`, `HOT_LOW_TORQUE`; `HELP` and `HINTS` the slice-2 help), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`; `wall_cautions`, the Size tab's amber wall lines: the wall past the keyway and, since BL-082, the cap skirt under `MIN_CAP_SKIRT_MM`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list; `drawing_ui`, the opening tab's, without the list; `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `slider_range` (an input's slider range from the catalogue built once, the shared spelling the torque card, the Size tab's controls, the guides and the tests read; the Equation panel and the panel's Required slider keep their own), `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider` (with `typed_number`: a value box's own text is no typed value, BL-097), `head_line` (a row's head line, the changed dot and then the label, on a line that wraps at the width the row is given: an input row's and an assumption's, BL-099): one input row of any type; a click on its label traces the input; `weak_line` and `hint_line`: a weak line wrapped to its region's width (under a control, and the torque card's lines), and the hint line `plain_text::hint_for` gives a Key design row or a Size-tab control), `dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`, `shown_lines` (the dashboard rows drawn while the shaft-clamp section is paused or not; `dashboard_lines` keeps every row for the exports), `END_EFFECT_ROWS`, `greyed_by_end_effect`, `STORED_3D_ROWS`, `STORED_3D_INPUTS`, `STORED_3D_SENTENCE`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`; `why_note_text` and `why_note_link`, a "Why: <title>" link outside the rows, and `end_effect_banner_ui`, the banner with its link), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), `result_groups.rs` (`result_groups`: the headline, the nine chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`, `target_mismatch_notice`), `history.rs` (`History`: undo and redo), `guidance.rs` (the guidance layer, pure functions: `CHECK_GUIDES`, `counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `check_terms` (`CheckTerm`, `Quoted` with its one constructor `Quoted::new`; slice 3: each counted check's terms in a fixed order, the one table the fix tests and the torque card read), `SPACE_CLAIM_AXIS_SIZES`, `all_fixes`, `FixCache`, `within_minimum_diameter`, `guide_notes` (`trade_offs`, `side_effects` and the unchecked costs), `design_status`, `Consequences` (the lines under the Size tab's controls and its torque card, `card`), `ConsequenceCache` (one key: the design shown, the Minimum OD and the shaft-clamp pause), `magnet_count_line`, `SIZE_TAB_NOTE`, `TORQUE_BAND_NOTE` and `BANNER_NOTE` (where the Size tab's and the banner's Why links lead), `plain_text_violations` (re-exported from `engine::explain::plain`)), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines (since slice 3 without the N-2 and N+2 lines, which are the torque card's count row) and Why links; `slider_minimum`, the Outside diameter slider's lowest value, one expression with the torque card; `SliderMapping`, each Size slider's own mapping of a value to a design, one mapping for the controls and the torque card's steps, which sets the count whole, refusing a value a cast would change, and checks no range on the gap or the count: its caller holds the value to the slider), `torque_card.rs` (the torque card, slice 3, BL-073; pure, no egui: `card` -> `TorqueCard` (`ShownCountBelowSlider`, `ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow`, or `Rows`, one `Row` per Size control (`Control`), read by `rows`), each control one fixed step each way through its slider's own mapping (`size_controls::SliderMapping`, the one the controls apply), in the order of tests of its spec's section 3.3: a side's `Words`, or its `Change` (the stepped design's hot-low, its change in N·m and as a fraction of the design shown's, the Outside diameter and Overall length changes, whether the calibration factor changes, the counted checks whose level flips (`Flip`, over `counted_guides` at the panel's shaft-clamp pause) and the failing checks' quantities the step moves (`MarginChange`, from `guidance::check_terms`, drawn where `shows`)); `lines(auto)` draws it, a row's two N·m at one precision and the row Auto holds as `held_line`, every text a named constant (`CARD_HEADING`, `CAPTION`, the three Auto lines, the side words, `JUMP` and `CUP_JUMP`, `CALIBRATION_CHANGE`, the four one-line forms and the flip words; `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` also end the Size tab's exploration lines); `side_words` (sizing's validity rule, every row's, the keyway left out on the count row: decision D-1) and `count_step` (two magnets each way: checked, even, inside the slider), both moved from the N-2 and N+2 lines that the count row replaced; `STEP_SLACK_MM`, the one slack of the range test and the jump rules: the overlapping magnet length on the Overall length row, the cup's diameter on the Outside diameter row since the owner's FO-2), `sizing.rs` (`SizingMode`, `held_by_auto`, `SizingState` (the mode and the free variable; the target is `metal.required_min_Nm`), `SizingRunner`: debounced inverse sizing over a search range; `sizing_range`, the range Auto searches, one rule for the panel and the torque card's test table, slice 3), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell; the Size tab's one number format, slice 3 decision 10B: `fixed` (a number to a set number of decimals, with no + added), and `signed_fixed`, `signed_change` and `change_decimals` (a change to the fewest decimals, from its own (`TORQUE_DECIMALS`, two, for N·m: the readout's gap and the torque card), that keep its true sign and read a digit that is not zero, at most `CHANGE_EXTRA_DECIMALS` more: `decimals_with_a_digit`, the one cap rule, which the torque card's `shows` reads too), `shared_change_decimals` (a torque card row's two N·m at one precision), the one-decimal `signed_percent`, the format's bound since the owner's FO-4 at the slice-3 STOP (a number whose fixed form would need more than six whole digits, `MOST_WHOLE_DIGITS`, reads four significant digits with its sign, as the readouts write values, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row: `TorquePrecision`, the one precision `shared_torque_precision` finds for a row, the readout's gap a row of one, which `signed_torque_change` draws for a torque card side and the readout's margin and `torque_shortfall` for its shortfall's size, which is drawn without a sign), and `fewest_decimals_apart`, the guidance's search for more decimals), `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries; `mark_ui`, the brand mark before the title; `tab_ui`, a tab of the centre's row, underlined in the accent when it is the active one; `quiet_text`, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes; `mark_color`, the colour of the changed dot and a trace's marks), `test_support.rs` (headless egui helpers and the shared test designs, the torque card's among them with their one table, `card_designs()`; tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests; on the web it writes the settled design into the page's address and, at the end of each frame, takes the page's loading line away), `starting_design` (the design it opens on and hands the panel as its baseline), `address_of` (the page address that stands for a design), `run_native`, `TITLE`, `CANVAS_ID`, `LOADING_ID`, `LOADING_LINE`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the calculator's own theme, "Charcoal and crimson": the token table and the visuals built from it, forced dark), `app/fonts.rs` (the app's font: cuts of Inter Regular and SemiBold with tabular digits, ahead of egui's built-in fonts; the SemiBold draws the Heading style), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
```

with:

```markdown
| `src/engine/explain/records/shaft_joint.rs` | BL-064 shaft joint records: one per numeric `shaft_joint.*` result (23; the verdict, a text, has none), transcribed from `shaft_joint::compute` and `PartProperties::hub` with every correction on: the part material in effect as cases over the circuit and the back-iron pick (the Materials inputs for the default pick, a pick's library values, else the workbook's 6061-T6), each joint's Lamé rows (holding, bore stress, needed minimum, elastic edge) with a no-wall arm that gives NaN, the key alone (0 with no keyway, +inf with no bearing length or contact), each shaft's shrink as cases on 265 K with NIST's Fe-2.25Ni polynomial, the fitting temperature and the fittable edge, and the bonded inner hub's allowed fitting temperature as the lowest of your limit, the adhesive's design limit and the inner ring's own single-ring onset less 10 °C, NaN when any of the three is (owner R-15) |
| `src/gui/` | Feature `gui`: `mod.rs` (`follow_focus`: a scroll area scrolls, by the least that shows it whole, to the widget the keyboard's focus arrives on, BL-100, and returns its rect; `sideways_scroll_area`: a sideways scroll area nested in an up-and-down one that follows the focus both ways, the Equation panel's formula and the Clamp tab's screw-size table, BL-101; `wrapping_line`: a line that wraps with one point of its width kept free, for every wrapping line of the left column, of the Explain panel and of the check list's warning links, BL-099 and BL-098), `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, the left column (since BL-096 the Size column on every tab, with its folds `DETAILED_INPUTS` and `ASSUMPTIONS_FOLD`), the assumptions banner, the sizing controls, the right column (`checks_and_explain_ui`: the check list over the Explain panel, one `explain::Selection`, the Hot-low entry's consequences asked of the `ConsequenceCache` through `size_tab_consequences`, the Size tab's one ask, with the design's own results computed once per design (`ResultsCache`))), `tiles.rs` (since BL-081, the four tiles over the drawing on Size and torque: `Tile`, `tiles_ui`, `MIN_TILE_WIDTH`, the Hot-low tile's `needs_lines` and `gap_pieces`), `checks.rs` (since BL-081, the check list: `checks_ui`, `state_mark`, `mark_level`, `CHECKS_SHARE`), `explain.rs` (since BL-081, the Explain panel: `explain_ui`, `Selection`, `Asked`, the orientation and each entry; the torque card sits in the Hot-low tile's entry, `card_ui`, decision C-10; an entry's "Full lesson:" link is `guidance::row_learn`'s), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::new(marks, show_details)`, `show_details`, `Readouts::show` and `show_over` (the hover text built with the frame's `show_details`), `select` and `select_over` (a click selects the result, since BL-094; outlined while focused or selected, `with_selection`), `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `term_tag`, `reviewed_note`, `note_fallback` (Explain's next best read under an equation without a note, `NoteFallback`), `start_here_text` with `FERRITE_ONLY` on the ferrite step; its heights: `panel_heights`, `PANEL_SHARE`, `MIN_PANEL_HEIGHT`, `VIEW_STRIP`), `notes_view.rs` (since BL-088, a teaching note as the Explain panel draws it: `note_ui` (a note drawn against a `NoteView`, returning the `NoteLink` clicked), `held_reason`, `live_value`, `explains_ui` and the note's strings), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`, which scales a diagram to a place narrower than its 300 points, BL-098), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `plain_text.rs` (the display-name and help overlay over the engine's metadata, keyed by path: `label_for`, `help_for`, `original_help`, `hint_for`, `display_label`; `LABELS` holds spec 3.5's canonical names, three of them constants the Size tab and the plots draw too: `REQUIRED_TORQUE`, `PULL_OUT_NOMINAL`, `HOT_LOW_TORQUE`; `HELP` and `HINTS` the slice-2 help), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`; `wall_cautions`, the Size tab's amber wall lines: the wall past the keyway and, since BL-082, the cap skirt under `MIN_CAP_SKIRT_MM`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list; `drawing_ui`, the opening tab's, without the list; `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `slider_range` (an input's slider range from the catalogue built once, the shared spelling the torque card, the Size tab's controls, the guides and the tests read; the Equation panel and the panel's Required slider keep their own), `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider` (with `typed_number`: a value box's own text is no typed value, BL-097), `head_line` (a row's head line, the changed dot and then the label, on a line that wraps at the width the row is given: an input row's and an assumption's, BL-099): one input row of any type; a click on its label traces the input; `weak_line` and `hint_line`: a weak line wrapped to its region's width (under a control, and the torque card's lines), and the hint line `plain_text::hint_for` gives a Key design row or a Size-tab control), `dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`, `shown_lines` (the dashboard rows drawn while the shaft-clamp section is paused or not; `dashboard_lines` keeps every row for the exports), `END_EFFECT_ROWS`, `greyed_by_end_effect`, `STORED_3D_ROWS`, `STORED_3D_INPUTS`, `STORED_3D_SENTENCE`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`; `why_note_text` and `why_note_link`, a "Why: <title>" link outside the rows, and `end_effect_banner_ui`, the banner with its link), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), `result_groups.rs` (`result_groups`: the headline, the nine chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`, `target_mismatch_notice`), `history.rs` (`History`: undo and redo), `guidance.rs` (the guidance layer, pure functions: `CHECK_GUIDES`, `counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `check_terms` (`CheckTerm`, `Quoted` with its one constructor `Quoted::new`; slice 3: each counted check's terms in a fixed order, the one table the fix tests and the torque card read), `SPACE_CLAIM_AXIS_SIZES`, `all_fixes`, `FixCache`, `within_minimum_diameter`, `guide_notes` (`trade_offs`, `side_effects` and the unchecked costs), `design_status`, `Consequences` (the lines under the Size tab's controls and its torque card, `card`), `ConsequenceCache` (one key: the design shown, the Minimum OD and the shaft-clamp pause), `magnet_count_line`, `SIZE_TAB_NOTE`, `TORQUE_BAND_NOTE` and `BANNER_NOTE` (where the Size tab's and the banner's Why links lead), `plain_text_violations` (re-exported from `engine::explain::plain`)), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines (since slice 3 without the N-2 and N+2 lines, which are the torque card's count row) and Why links; `slider_minimum`, the Outside diameter slider's lowest value, one expression with the torque card; `SliderMapping`, each Size slider's own mapping of a value to a design, one mapping for the controls and the torque card's steps, which sets the count whole, refusing a value a cast would change, and checks no range on the gap or the count: its caller holds the value to the slider), `torque_card.rs` (the torque card, slice 3, BL-073; pure, no egui: `card` -> `TorqueCard` (`ShownCountBelowSlider`, `ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow`, or `Rows`, one `Row` per Size control (`Control`), read by `rows`), each control one fixed step each way through its slider's own mapping (`size_controls::SliderMapping`, the one the controls apply), in the order of tests of its spec's section 3.3: a side's `Words`, or its `Change` (the stepped design's hot-low, its change in N·m and as a fraction of the design shown's, the Outside diameter and Overall length changes, whether the calibration factor changes, the counted checks whose level flips (`Flip`, over `counted_guides` at the panel's shaft-clamp pause) and the failing checks' quantities the step moves (`MarginChange`, from `guidance::check_terms`, drawn where `shows`)); `lines(auto)` draws it, a row's two N·m at one precision and the row Auto holds as `held_line`, every text a named constant (`CARD_HEADING`, `CAPTION`, the three Auto lines, the side words, `JUMP` and `CUP_JUMP`, `CALIBRATION_CHANGE`, the four one-line forms and the flip words; `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` also end the Size tab's exploration lines); `side_words` (sizing's validity rule, every row's, the keyway left out on the count row: decision D-1) and `count_step` (two magnets each way: checked, even, inside the slider), both moved from the N-2 and N+2 lines that the count row replaced; `STEP_SLACK_MM`, the one slack of the range test and the jump rules: the overlapping magnet length on the Overall length row, the cup's diameter on the Outside diameter row since the owner's FO-2), `sizing.rs` (`SizingMode`, `held_by_auto`, `SizingState` (the mode and the free variable; the target is `metal.required_min_Nm`), `SizingRunner`: debounced inverse sizing over a search range; `sizing_range`, the range Auto searches, one rule for the panel and the torque card's test table, slice 3), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell; the Size tab's one number format, slice 3 decision 10B: `fixed` (a number to a set number of decimals, with no + added), and `signed_fixed`, `signed_change` and `change_decimals` (a change to the fewest decimals, from its own (`TORQUE_DECIMALS`, two, for N·m: the readout's gap and the torque card), that keep its true sign and read a digit that is not zero, at most `CHANGE_EXTRA_DECIMALS` more: `decimals_with_a_digit`, the one cap rule, which the torque card's `shows` reads too), `shared_change_decimals` (a torque card row's two N·m at one precision), the one-decimal `signed_percent`, the format's bound since the owner's FO-4 at the slice-3 STOP (a number whose fixed form would need more than six whole digits, `MOST_WHOLE_DIGITS`, reads four significant digits with its sign, as the readouts write values, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row: `TorquePrecision`, the one precision `shared_torque_precision` finds for a row, the readout's gap a row of one, which `signed_torque_change` draws for a torque card side and the readout's margin and `torque_shortfall` for its shortfall's size, which is drawn without a sign), and `fewest_decimals_apart`, the guidance's search for more decimals), `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries; `mark_ui`, the brand mark before the title; `tab_ui`, a tab of the centre's row, underlined in the accent when it is the active one; `quiet_text`, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes; `mark_color`, the colour of the changed dot and a trace's marks), `test_support.rs` (headless egui helpers and the shared test designs, the torque card's among them with their one table, `card_designs()`; tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests; on the web it writes the settled design into the page's address and, at the end of each frame, takes the page's loading line away), `starting_design` (the design it opens on and hands the panel as its baseline), `address_of` (the page address that stands for a design), `run_native`, `TITLE`, `CANVAS_ID`, `LOADING_ID`, `LOADING_LINE`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the calculator's own theme, "Charcoal and crimson": the token table and the visuals built from it, forced dark), `app/fonts.rs` (the app's font: cuts of Inter Regular and SemiBold with tabular digits, ahead of egui's built-in fonts; the SemiBold draws the Heading style), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
```

Replace:

```markdown
page draws no frame. One rule, `gui::follow_focus`, called last in a scroll area's content, serves
the column, the Explain panel, the Geometry tab's callout list and the Clamp tab. It does not yet
serve the Equation panel and the Clamp tab's screw-size table (sideways scroll areas inside an
up-and-down one: egui hands a scroll to the innermost area alone) or the Results table (only its
rows in view are drawn, so only they take the focus): BL-101. egui also gives the keyboard's focus
to parts that show none, each scroll area's drag surface and scroll bar and each column's resize
edge: BL-102. The design's nothing-lost table (8.2, test 9)
is one test, `nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home`:
```

with:

```markdown
page draws no frame. One rule, `gui::follow_focus`, called last in a scroll area's content, serves
the column, the Explain panel, the check list, the Geometry tab's callout list, the Clamp tab and
the Equation panel. A sideways scroll area inside an up-and-down one, the Equation panel's formula
and the Clamp tab's screw-size table, is `gui::sideways_scroll_area` (BL-101): egui hands both
axes' scroll targets to the innermost area alone, so the inner area scrolls sideways to the widget
and the outer one is asked again for its rect once the inner area has ended; a formula wider than
the panel shows from its start. The Results table draws only the lines in view, and a line takes
the focus only while the view shows it whole: Tab passes the lines on screen and leaves the table,
never stopping on a line half seen, and the table does not scroll for it (owner question 67's
default, register row A-44; a focused line a wheel then cuts loses the focus). egui also gives the
keyboard's focus to parts that show none, each scroll area's drag surface and scroll bar and each
column's resize edge: BL-102. The design's nothing-lost table (8.2, test 9)
is one test, `nothing_is_lost_each_action_view_note_dashboard_row_input_and_assumption_has_its_home`:
```

Replace:

```markdown
  colour swatches fade while another equation's value is hovered, its terms being the ones marked
  on screen; the list scrolls sideways when the region is narrower than its rows, as the equation
  does), the breadcrumb (a long walk shows its last 8 crumbs after "…") and "used by". Clicking a
  term, in the equation (the H under a Σ included) or the list, drills
```

with:

```markdown
  colour swatches fade while another equation's value is hovered, its terms being the ones marked
  on screen; the label and what the term is share the last column and wrap in the panel's width,
  BL-086, where before they ran past its edge in a sideways scroll area; a value takes at most a
  third of the list's width, `explorer::TERM_VALUE_SHARE`, and a longer one, a text such as
  "never: steady state stays below the limit", wraps in its own column; the equation, which does
  not wrap, alone scrolls sideways, decision D-109), the breadcrumb (a long walk shows its last 8 crumbs after "…") and "used by". Clicking a
  term, in the equation (the H under a Σ included) or the list, drills
```

Replace:

```markdown
  assumptions it depends on (`Registry::term_style`, `modified_assumptions_upstream`).
- **Teaching notes** (`explorer.rs`, `diagrams.rs`, `engine/explain/notes.rs`; spec Addendum A4 and the guidance
  slice 2 spec): 21 notes, 20 shown while the clamp section is paused (`engine::clamps::PAUSED`, `notes::shown`:
```

with:

```markdown
  assumptions it depends on (`Registry::term_style`, `modified_assumptions_upstream`).
- **Teaching notes** (`notes_view.rs`, `explorer.rs`, `diagrams.rs`, `engine/explain/notes.rs`; spec Addendum A4 and the guidance
  slice 2 spec): 21 notes, 20 shown while the clamp section is paused (`engine::clamps::PAUSED`, `notes::shown`:
```

Replace:

```markdown
  scrolls to the input's row (`NoteLink`, `Explorer::follow_note_link`), or plain text with the reason where the
  input is not free on the design shown: "held by Auto", "set by the <pick> pick" (`explorer::held_reason`,
  `engine::picks::set_by_pick`), or at the top or bottom of its slider the entry's way (`SliderRange::step_from`).
```

with:

```markdown
  scrolls to the input's row (`NoteLink`, `Explorer::follow_note_link`), or plain text with the reason where the
  input is not free on the design shown: "held by Auto", "set by the <pick> pick" (`notes_view::held_reason`,
  `engine::picks::set_by_pick`), or at the top or bottom of its slider the entry's way (`SliderRange::step_from`).
```

Replace:

```markdown
  A note quotes live values of the design (`{result.path}` placeholders, `engine::explain::note_text`,
  `explorer::live_value`), "no prediction" where there is none (a value that is not a finite number, or one computed
  from an out-of-range end effect), never in its first sentence or its title. Since BL-085 (decision C-11) the
```

with:

```markdown
  A note quotes live values of the design (`{result.path}` placeholders, `engine::explain::note_text`,
  `notes_view::live_value`), "no prediction" where there is none (a value that is not a finite number, or one computed
  from an out-of-range end effect), never in its first sentence or its title. Since BL-085 (decision C-11) the
```

Replace:

```markdown
  title a link to its note. A note's entry is the links to the equations it explains ("Explains",
  `explorer::explains_ui`: one opens the Equation panel and leaves the selection, C-25) and the note
  (`explorer::note_ui`; a note shown by id goes through `explorer::reviewed_note`) with its diagram painted with egui (the square wave and
  the sum of its harmonics, the flux paths with and without back iron with each block lettered N or S, torque
```

with:

```markdown
  title a link to its note. A note's entry is the links to the equations it explains ("Explains",
  `notes_view::explains_ui`: one opens the Equation panel and leaves the selection, C-25) and the note
  (`notes_view::note_ui`; a note shown by id goes through `explorer::reviewed_note`) with its diagram painted with egui (the square wave and
  the sum of its harmonics, the flux paths with and without back iron with each block lettered N or S, torque
```

Replace:

```markdown
  each step opening its equation and selecting its note in the Explain panel (Previous, Next, Stop; spec 5.5,
  `Explorer::start` returns the step's note); the ferrite step's bar adds "(ferrite rings
  only)" (`explorer::FERRITE_ONLY`): its check reads n/a on NdFeB rings. The empty Equation panel names Start here
  (TC-29b). Under an equation no reviewed note explains, or a value with no equation, the lesson line offers the
  next best read (`explorer::note_fallback`, TC-29a, TC-30): the dashboard row's own Why note or assumption, else
```

with:

```markdown
  each step opening its equation and selecting its note in the Explain panel (Previous, Next, Stop; spec 5.5,
  `Explorer::start` returns the step's note; the step line wraps in the panel's width, BL-086); the ferrite step's bar adds "(ferrite rings
  only)" (`explorer::FERRITE_ONLY`): its check reads n/a on NdFeB rings. Every way of opening the Equation panel sets
  its trail first, so its empty text went (BL-089). Under an equation no reviewed note explains, or a value with no equation, the lesson line offers the
  next best read (`explorer::note_fallback`, TC-29a, TC-30): the dashboard row's own Why note or assumption, else
```

Replace:

```markdown
| `src/engine/clamps.rs` | The paused section hides exactly the `clamps.` and `materials.screws.` paths and only while paused; each of its 28 inputs (25 clamp inputs, 3 screw classes), at every probe design, at both slider ends or every choice, moves no result outside it. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input, a 1280 x 1024 screen): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame; the value lands on the step's decimals (1.41) and stepping back lands on the default exactly; a typed value snaps to the step and is clamped into the range; pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept until edited and flagged; the changed dot and per-field reset; the session baseline (UX overhaul slice A: a panel built `with_baseline` opens on it and Reset all returns to it, `new()` keeps the spreadsheet's values, nothing moves it; the changed dot, a row's reset and its hover, the tooltip's Default line and the Equation panel's changed tag follow it; the status line reads "Starting design" at it; Open the example design is one undo step and is not drawn on a panel whose baseline is the spreadsheet's design; the new texts keep the text rules of the design's section 8.3; the settled design follows settled edits, undo, redo and an opened link); the axial length override starts blank and enters at the ring's length; a selector switches the branch (no back iron); a text input edits the part name and its note follows; every group opens and draws every input; the dashboard's stored-3D label, markers, end-effect banner (over the dashboard and the centre region) and space-claim overshoot; the results table's search and its export requests; undo and redo (buttons and shortcuts; one step per drag, per nudge, per typed name; Ctrl+Z in a text field left to the field; reset all undoable); save and load requests, a loaded and a refused design file, the share link on the clipboard and a broken link; the host's report; Torque -> Magnets: the solved design shown, the debounce (no solve per frame, none while a button is down, a repaint after a solve), the Size tab's locked Auto slider (Overall length, Outside diameter or Magnets per ring, at the solved value; with ring-size Auto the cap OD and thread rows are locked too, following the cup), leaving the mode keeps the value as one undo step and solves a pending change first, an unreachable target shows the best value, the free-variable picker, a share link carries the sizing state, the JSON export holds the design shown, typing in the results search does not defer the solve; a held arrow key is one undo step, a slider's value box being typed in is an edit of the design (one step on Enter), the results search holds back no undo step, only this frame's rows count as design fields, a start-up share link is no undo step, a host can keep Ctrl+Z for itself; idle frames change nothing with every group open and off-grid values; the measured drag enters at the model's drag, unrounded; every drawn text and hover text has glyphs in egui's default fonts (every tab of the centre region and the geometry callouts' hover texts); the panel draws inside an `egui::Window` (M5); the Geometry tab follows the design shown (a live redraw; the sized design in Torque -> Magnets); each plot draws its plot, the design's marker at the edited pull-out in the edit's frame; the clamp tab draws the recommended clamp or drawing.py's message; a ~930 px window shows a row's label (filling its 120-point column) and value; the ordering (O-1 to O-8): every group of both orders opens and draws its rows and headings, the Advanced sections closed until their heading is clicked; the order toggle changes no input, design, share link or undo history; the filter box shows the matches alone (count, run headings; a matched row nudges; no undo step for typing; the workbook order's headings; none; blank); a focused advanced input opens its group and heading past a filter; the results by group (every heading drawn, a heading opens and closes its rows, the engine order without headings); the failing filter shows exactly the failing checks with five red and three amber badges; a click on an input's label frames exactly the dashboard rows it drives and its own row until a second click; a click on a result frames exactly the Key design rows it reads and counts the traced rows per group; the traced filter (disabled without a trace, the traced count, the headline's traced heading); a heading shows the worst level of its group's checks (the headline and the closed coupling model red, the mass none); a heading click while searching leaves its group as it was, its hover text saying why; the failing filter with a search no failing check matches says no result matches; an input whose results have no equation record traces none, says so, and the traced filter then says the trace marks nothing (Clear trace turns the filter off); reading a traced result keeps the input's trace and its traced rows, a result without a record keeps it, a result outside it replaces it; idle frames in the workflow order's filtered view change nothing; the final review's fixes: out of the end-effect range the table's headline rows carry exactly the dashboard's badges (none on the hot minimum, the screw and the gearbox check) and the failing filter leaves the hot minimum out (seven red, four amber); a search that hides the traced rows says no result matches, and a heading counts the traced rows it shows; a trace moves no input row, so a second click at the same point ends it; the locked free variable's label traces it with no edit or undo step; views stay in their own area: on a 1280 x 620 page and the laptop screen, with the Equation panel at its starting height and dragged as far up as it goes, on a 1280 x 300 page and inside an `egui::Window` (the linkage Tools window's 1100 x 700), no view (geometry, the five plots, clamp, results) paints into the panel or takes a click inside it (shapes split from the panel's by its separator line, widgets by its resize handle), the scrolling views fill the space down to it, and the window keeps its size; the panel opens at about 45 % of the centre region and its drag stops `VIEW_STRIP` short of the region's top, then follows the pointer back down; every view draws beside the panel in a tiny window; both spreadsheet buttons (the header's and the results table's) queue `magcoupling-results.xlsx`, which reads back with the four sheets, the edited face gap as a number and the share link, follows the input order shown and, in Torque -> Magnets, holds the solved length (noted) and the solve's outcome; slice 1 (BL-055): the panel opens on Size and torque under the welcome strip, which closes for the session and leaves no undo step; the dashboard's status line counts the failing checks at the default design and after a fix; the Required slider edits `metal.required_min_Nm` (one undo restores it) and the Hot-low gap follows it; the consequence lines follow an input edit; Auto locks the matching slider, re-solves after the debounce, leaves idle frames alone and reports an unreachable requirement; opening the Size tab no longer changes the sizing mode; every new string has glyphs in egui's default fonts; slice 2 (BL-056): the Show spreadsheet details toggle starts off, a click turns it on and off, its hover says what it shows, and it is in no design file, share link or undo step and survives a loaded file, undo, reset all and a share link opened as the session's start; off, the face-gap row's hover names no path or cell and the pull-out's hover on the dashboard and on the Size tab no path, cell or correction, on they do; the correction ids and their legend, the results table's cells (pinned tests switch the details on), the Equation panel's cell and corrections lines and a result-without-record's cell, and the assumptions' sources show only with the details; at 930 px the table's label and value stay on screen with the cell and marker columns and without them; the toggle's label and hover, the legend, the marker heading and scope, the original-note prefix and the example design's status start and hover pass `plain_text_violations`; every text, the details' included, has glyphs in the default fonts; one name per quantity (`size_controls.rs` tests too): the Size tab's readout labels equal the input and dashboard rows' names (Required torque, Hot-low torque; one Hot-low torque readout in the centre beside the dashboard row of that name), the Auto row and its hover name the required torque, the footnote and the Outside diameter and Overall length hovers say what they move (the ring radius; both rings' magnet length) within the help caps (the length hover within its own cap, 4 sentences and 80 words, since BL-074: it also says that the first move from a blank axial magnet length sets both rings to one length, a test seen red first), each label shows its hover, the open equation's header is the display label, and Auto's no-valid-value line names the ring radius; the Outside diameter slider's lowest value is one expression, `slider_minimum` (the user's Minimum OD, else the range's start or a loaded smaller design's own diameter, held to the Minimum OD's range), which the controls pin on a loaded smaller design's first frame (`the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown`, slice 3); each Size slider makes its design through its own mapping, `SliderMapping`, which the torque card's steps share, the Outside diameter's at the slider's own minimum (at a Minimum OD below the default floor it reaches a diameter that floor refuses), the count's refusing a value a cast would change (not finite, not whole or past what an i64 holds) and neither the gap's nor the count's checking a range (`each_size_slider_makes_its_design_through_its_own_mapping`, slice 3); slice 2, help batch A (T7): each Key design row draws its hint line under it and above the next row (once: its own group open adds none), each of the five Size-tab controls its hint line under its label and above the next control's, and at a 930-point window every hint stays inside the inputs side, at least one wrapped; the panel's own strings (the assumption reset, the order toggle, the workflow headings, the trace banners and hint, the explorer's no-equation text and tag, the end-effect banner and greyed hover, the geometry notes, the welcome strip) pass the plain-words guard and name none of the engine's internal words; with Show spreadsheet details on, the dashboard keeps its width frame after frame (each value and its correction ids flow as one wrapped text); with Auto on (Torque -> Magnets, the axial magnet length free), Start here's step-1 note draws its magnet-length entry as plain text " (held by Auto)" once, a click on that label opens no row and keeps the note open, and its live values are the design shown's: the pull-out watch-out, which moves with the magnet length, reads the solved length's numbers and not the stored inputs' (Review Focus RF-1); the shaft-clamp pause (ruling 11): the Clamp tab and the clamp inputs (the workflow order's clamp sections, the original-sheet order's Shaft clamps group, the filter's matches) show only while the section is not paused, its two assumptions leave the Assumptions view and the banner, the shipped panel starts from the switch and passes its pause to the fixes (at the clamp's limit the hot-low torque's fix is drawn only while paused), the header's banner and the trace banner (`the_panel_passes_its_pause_to_the_fixes_the_banner_and_the_trace`), a design file written before the pause and a share link keep their clamp values in a paused panel (the engine still computes the section); one test, `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`, walks `every_shown_text` (every drawn text, and the hover texts of inputs, dashboard rows, table rows, callouts and the named static hovers; the rows through the panel's own filters) once per state: every such text has glyphs paused or not, and while paused none of them, nor a shown note's fields and drawn change entries, the Start here walk and hover, a counting guide's name and Why, the shaft joints' Problem as drawn on three designs that fail them (BL-064: the inner hub slipping, over its stress limit, on an uncooled wheel shaft), a Why link's reason or a shown assumption's name, rationale and source, names the clamp (the guard; no allow-list entry; no shown note's change entry or equation leads into the section; with the section shown the same enumeration names the clamp and reaches the Clamp tab, the clamp dashboard row, a clamp input row, the Shaft clamps group and the clamp friction assumption; while paused it reaches the label and help of every shaft-joint input and result and, each whole as drawn, their Shrink fit and Fitting temperatures sections and the Shaft joints headings of the original-sheet inputs and of the results table, BL-064); panel tests draw the whole section (`Harness::of` clears the pause) unless a test pauses it; slice 2 (T15): the gearbox ripple row's Why link opens the slip note, its first sentence drawn once and no input changed; slice 2 (T19, reachability, spec 3.7): the Size tab's Why links under Magnets per ring and Outside diameter, and on the torque card's heading (slice 3), open the torque note, and those at the end of the Required and Hot-low rows the torque-band note, each on its own in the Equation panel (clicked on the text's last character, `test_support::last_characters`, since a wrapped row's text rect also covers the slider before it); with an odd magnet count the Outside diameter control and its link are not drawn and the Magnets per ring link and the torque card's are; the link under each copy of the end-effect banner (over the dashboard and the centre) opens the end-effect note, and without the banner there is none; every dashboard row opened with Explain on shows its note, the closest note, an assumption or its own reason, never `NO_NOTE` (`explain_never_shows_no_teaching_note_on_a_dashboard_row`); the closest-note link, the one copy of the title on its line left of the dashboard, opens that note (`click_the_closest_note_link`, shared with the gearbox row's Explain test); with Explain on, a value with nothing upstream to read draws `NO_NOTE` and `START_HERE_HINT` once each, and Explain off draws neither; BL-059: the gearbox row's own Why link, the copy of the torque-band title under the row's label (the hot-low, hot-minimum and cold-high rows draw it first), opens that note and changes no input; BL-064: the shaft-joint row's own Why link, the one copy of the shaft-joint note's title on the dashboard, under the row's label, opens that note and changes no input (`the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note`, sharing `row_s_why_links` and `click_opens_note` with the gearbox row's test); with Explain on the gearbox row, which has no equation or note of its own, offers its Why link's note as the closest note, and the link beside the lead opens it; at the defaults the failing filter lists five red checks and one amber (the daily-cycle screen at the entered bondline, BL-063), the gearbox check's row once with a red badge (`table_badges_on`, shared with the end-effect test); slice 3 (T5): the Size tab draws the torque card in place of the label under the readouts, its heading and each of its lines once, one under the next, a click on the heading closing and opening it as one on its arrow does (`the_size_tab_draws_the_torque_card_in_place_of_the_label`); a typed Required value reaches the card in the frame that applies it (`a_required_edit_reaches_the_card_in_its_own_frame`); under Magnets per ring two lines, the ring-size line with Auto off or on the magnet length and the fit line, the card drawing the Auto line of each Auto state, the count's steps (at 4 magnets, and at 40 where 40 blocks fit, the side past the slider reading "outside the slider's range") and following each edit; idle frames compute the card once, a gap edit, a clamp-pause toggle and a new Minimum OD once each (the drawn Outside diameter row's -1 mm side then reading "outside the slider's range"), and turning Auto off at a solved length or Auto on the count at its best computes nothing, while the design's results the cache is handed are computed once per design, none for idle frames, the pause or the Minimum OD (`idle_frames_compute_the_card_once_and_an_edit_again`); a drag of the gap slider with Auto on the magnet length computes the card, and its design's results, once for each design it draws (`a_drag_computes_the_card_once_for_each_design_it_draws`); idle frames change no input with the card open and closed, at the clamp pause; slice 3 (T6, headless, each test pausing the clamp section before the card's first frame, but for the Equation panel's check on the screen, which draws every view unpaused): at 1280 x 800 the card sits under the Pull-out readout and above the drawing, open, every line drawn and the drawing at least 160 points, at the defaults and the smoke design; at 930 x 800 it starts closed (the region under 400 points), at short magnets too, closed no taller than the label it replaced, and its Why link wraps under the heading inside the region and opens the torque note; at 1080 x 745 no back iron starts closed (its open card would take the drawing under its floor) and short magnets open; each rule opens it at its boundary, a region exactly 400 points wide or exactly as tall as the open card, the item spacing and the floor, while 399.99 points or 0.01 point less, or the Equation panel open, closes it (`the_card_s_default_opens_at_each_rule_s_boundary`); with no user choice a window resize decides it again (open at 1280 x 800, closed again at 930 x 800), a change of its height alone too (at 1080 points wide open at 800 points tall, closed at 700, open again at 800); the Equation panel closes it and its closing opens it, and no view, the card open or closed, paints into the panel, on the screen or inside an `egui::Window`; a click on the heading or the arrow wins for the session, through a design edit, a resize, the Equation panel (the drawing keeping its 160 points beside it), Reset all and a load; with no user choice Reset all and a load keep the default decided for the view (at 1080 x 800 no back iron's closed card stays closed through Reset all and a load of 8 magnets, which would open it, and the defaults' open card stays open through a load of no back iron) (`reset_all_and_a_load_keep_the_card_s_default_with_no_user_choice`); drags of the gap, the Outside diameter across the fit edge and Required across hot-low (checked on every drag frame), the assumptions banner, and a side panel narrowing the region at no back iron leave it as decided; opened at 930 x 800, and at 1280 x 800 at a design file's calibration coefficient of 1e300, whose 887-character lines rule 3 starts closed, its body scrolls (the mouse wheel brings its last row into view) and the drawing keeps 160 points; its measured height is its drawn height within a point at 1280 x 800 and 930 x 1024, and at 1280 x 1024 at that calibration coefficient, and at 930 x 800 its body is the cap (`the_card_s_measured_height_is_the_drawn_height`); the look: one line in the accent, as wide as the active tab and just under its name, which follows a click to another tab, the active tab's name in the strong text colour and another's in the quiet one (a theme's weak colour; the text colour in egui's own visuals, which set none), and one accent square of 8 points before the title's first glyph, at the title's height (`the_active_tab_is_underlined_in_the_accent_and_no_other`, `the_brand_mark_is_one_accent_square_before_the_title`); the changed dot and a trace's banner and frames are the host's link colour, not the colour it gives a selected choice's name (`the_panel_s_marks_are_the_host_s_link_colour`); a readout's equation wider than the window is set smaller so its tooltip stays inside the window, and one that fits keeps the tooltip's size (`src/gui/readouts.rs`, `an_equation_wider_than_the_window_is_set_smaller_and_another_keeps_its_size`); BL-097: on the app's starting design, Auto on until solved and Auto off, Tab from the top of the page through the Size column changes no input, no result and no Minimum OD, makes no undo step and selects nothing (`tab_through_a_length_auto_left_off_its_step_changes_nothing`, with the `app` feature); with a face gap of 1.4123 mm and a requirement past its slider's range loaded, Tab past every value box of the Size column and of the open Detailed inputs does the same (`tab_past_every_value_box_keeps_a_value_off_its_step_or_outside_its_range`); Enter on a value box with nothing typed, or with the number it shows typed again, keeps the 1.4123, while a typed value edits on Tab (one undo step, the row selected), on Enter and on a click elsewhere, and an arrow key steps it onto the step (`a_typed_value_and_an_arrow_key_edit_a_value_off_its_step_and_its_own_text_does_not`); the rule itself (`src/gui/input_ui.rs`, `a_value_box_s_own_text_is_no_typed_number`); a number typed into a tiny value's box edits, one undo step, clamped to the range as egui clamps it: the vacuum permeability, the NdFeB expansion and the cap's field integral (`a_number_typed_into_a_tiny_value_s_box_edits`); BL-100: at 1280 x 800 and 1024 x 700 with the Detailed inputs open, after each press of Tab and of Shift+Tab every focused widget of the left column is whole in its view once the scroll has settled, scrolled no further than it was cut plus egui's gap at the edge; nothing is selected or edited; a wheel then takes the focused row out of view and it stays out; an idle frame asks for no repaint; a Down arrow key's move is followed too (`the_left_column_follows_the_keyboard_s_focus_and_fights_no_wheel`); the same walk through a check's entry in the Explain panel at 1024 x 700 (`the_explain_panel_follows_the_keyboard_s_focus`) and through the Geometry tab's callout list (under the open Equation panel) and the Clamp tab (`the_centre_s_scrolling_views_follow_the_keyboard_s_focus`); each fails with its area's call of `follow_focus` taken out; the left column wheeled 300 points down stays where the wheel left it while Tab, then Shift+Tab, pass the top bar, the centre and the right column (`a_wheeled_column_holds_still_while_the_focus_is_on_no_widget_of_its_own`; it fails with `follow_focus`'s clip check taken out); BL-098: at 1024 x 700 and 1280 x 800 every shown note in turn, then the note list, a check's entry, a tile's and the orientation, leaves the right column 300 points wide (egui's `PanelState`) with its right edge where it was (`the_right_column_keeps_its_width_whichever_note_is_shown`); so does the check list's material warning on a design with no back iron and on one whose sleeve conducts better than 316L, and its Why link opens its note (`the_right_column_keeps_its_width_with_a_material_warning_shown`); in the app's theme and font, on the starting design and on the failing example design with the spreadsheet details on, so does every entry the Explain panel can show: the orientation, the note list, every note, check, tile, Size column row and input, and every result of the results table (`the_right_column_keeps_its_width_whatever_entry_is_shown_in_the_app_s_own_font`, with the `app` feature); a diagram in a place 284 or 150 points wide is the picture scaled to that width with every shape and label inside it, and is 300 x 130 where it has room (`src/gui/diagrams.rs`, `a_diagram_takes_the_width_it_is_given_scaled_as_one_picture`); the five plots share one Plots tab (BL-104, decision C-9): the tab row is Size and torque, Geometry, Plots, Clamp and Results table, on one row at 1024 x 700, the picker under the Plots tab names the five plots in their order and marks the one shown, a click on a name shows its plot, the tab keeps the plot shown on a click and reopens the plot picked last from another view, picking edits nothing, selects nothing and is no undo step, a picked plot is drawn in the frame of its click, and the tab's word passes the plain-words guard (`the_five_plots_share_one_tab_whose_picker_shows_each_plot`); by the keyboard, Tab reaches the Plots tab and walks the picker's five names in their order with no view changed, Enter on the tab opens the plot shown last, and Space or Enter on a focused name shows its plot (`the_keyboard_reaches_the_plots_tab_and_picks_a_plot_by_space_or_enter`) |
| `src/gui/geometry.rs`, `src/gui/geometry_view.rs` (feature `gui`) | The end view draws every part from the results (the pocket's corners at the pocket corner radius, the rings at the retainers' diameters, the key, the shaft joints' 3 mm, from where it crosses the bore), faceted blocks on their flats and arcs as sectors; each end-view callout's line measures the gap it names (the corner gap along face 1's normal, from the dashed circle the inner corners sweep to the outer flat); red below zero and for a clearance below its target; the side view spans the housing in effect (a 50.8 mm override: the 53.6 mm cavity, the hub centred on it, the liner and sleeve the retainer span in effect); the space-claim callouts name each overshoot, in red, with the exceeded axes' dashed lines red (every axis at once too), exactly at the claim inside and the next number below over, and a NaN claim amber; a claim far past the pieces (1e300, 1e39, -1e39) is left off with one note, drawn at `CLAIM_REACH` times the extent and not past it; the design checks show while each inconsistency holds (equality is not below); the wall-rule line unless there is no back iron (the cup wall input named by its label), asking for the wall the cup wall check compares with, so the first slider step that passes is not asked for the verdict's rounded-up suggestion; that wall rounded up at its last digit, so a wall shown equal to it passes (12 poles, a 1.23 T design flux density and a 1.95 mm wall fail the check, and the line asks for 1.951 mm); a minimum length rounds up at its last shown digit (past a power of ten too, and a length already on a digit stays), and one not finite or not positive shows as it is; pole counts from a file draw at most 200 blocks; two poles draw their four blocks in a round pocket and hub (a 2-gon's corners would fly off), three a triangular pocket; a NaN draws nothing wrong and says so (the parts dropped counted per view). Painted: both views at one scale, the largest that fits (`side_by_side`: none without room or for an extent that is not finite); the cup OD, the shaft and the face gap at their pixel lengths, the cup depth and the cap as painted rectangles; the callout texts in their colours (the default clearance red, its line too); an overshoot red; hovering a dimension shows its result's hover text; each dimension tag sits on its own plate of the drawing's background; a NaN view and a tiny screen draw; a region of no size paints no view; a huge claim keeps the pieces to scale with its note; in a short region (400 points down to none) the view ends at its foot, the drawing keeping 160 points while the list keeps three rows, then shrinking to scale (`drawing_height`); in a region of exactly `drawing_floor` (slice 3: the room the torque card leaves, pinned at 212.125 points in egui's default style, 160 over three 17.375-point list rows) the drawing keeps its 160 points, and a point less it shrinks below them; a dimension's hover and a callout's in the list under the drawing name their result's path only with the spreadsheet details; the design checks name their inputs by their labels and pass the plain-words guard; the key in both states of the shaft-clamp pause, at the shaft joints' key width (BL-064: each end of its slider; one typed wider than the bore from the axis; a width or a keyway depth that is not a number not drawn and counted; none at a keyway depth of 0 or less, nor at a key width of 0 or less), never at the paused clamp's, and the clamp-boss design check only while the shaft-clamp section is not paused |
```

with:

```markdown
| `src/engine/clamps.rs` | The paused section hides exactly the `clamps.` and `materials.screws.` paths and only while paused; each of its 28 inputs (25 clamp inputs, 3 screw classes), at every probe design, at both slider ends or every choice, moves no result outside it. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input, a 1280 x 1024 screen): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame; the value lands on the step's decimals (1.41) and stepping back lands on the default exactly; a typed value snaps to the step and is clamped into the range; pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept until edited and flagged; the changed dot and per-field reset; the session baseline (UX overhaul slice A: a panel built `with_baseline` opens on it and Reset all returns to it, `new()` keeps the spreadsheet's values, nothing moves it; the changed dot, a row's reset and its hover, the tooltip's Default line and the Equation panel's changed tag follow it; the status line reads "Starting design" at it; Open the example design is one undo step and is not drawn on a panel whose baseline is the spreadsheet's design; the new texts keep the text rules of the design's section 8.3; the settled design follows settled edits, undo, redo and an opened link); the axial length override starts blank and enters at the ring's length; a selector switches the branch (no back iron); a text input edits the part name and its note follows; every group opens and draws every input; the dashboard's stored-3D label, markers, end-effect banner (over the dashboard and the centre region) and space-claim overshoot; the results table's search and its export requests; undo and redo (buttons and shortcuts; one step per drag, per nudge, per typed name; Ctrl+Z in a text field left to the field; reset all undoable); save and load requests, a loaded and a refused design file, the share link on the clipboard and a broken link; the host's report; Torque -> Magnets: the solved design shown, the debounce (no solve per frame, none while a button is down, a repaint after a solve), the Size tab's locked Auto slider (Overall length, Outside diameter or Magnets per ring, at the solved value; with ring-size Auto the cap OD and thread rows are locked too, following the cup), leaving the mode keeps the value as one undo step and solves a pending change first, an unreachable target shows the best value, the free-variable picker, a share link carries the sizing state, the JSON export holds the design shown, typing in the results search does not defer the solve; a held arrow key is one undo step, a slider's value box being typed in is an edit of the design (one step on Enter), the results search holds back no undo step, only this frame's rows count as design fields, a start-up share link is no undo step, a host can keep Ctrl+Z for itself; idle frames change nothing with every group open and off-grid values; the measured drag enters at the model's drag, unrounded; every drawn text and hover text has glyphs in egui's default fonts (every tab of the centre region and the geometry callouts' hover texts); the panel draws inside an `egui::Window` (M5); the Geometry tab follows the design shown (a live redraw; the sized design in Torque -> Magnets); each plot draws its plot, the design's marker at the edited pull-out in the edit's frame; the clamp tab draws the recommended clamp or drawing.py's message; a ~930 px window shows a row's label (filling its 120-point column) and value; the ordering (O-1 to O-8): every group of both orders opens and draws its rows and headings, the Advanced sections closed until their heading is clicked; the order toggle changes no input, design, share link or undo history; the filter box shows the matches alone (count, run headings; a matched row nudges; no undo step for typing; the workbook order's headings; none; blank); a focused advanced input opens its group and heading past a filter; the results by group (every heading drawn, a heading opens and closes its rows, the engine order without headings); the failing filter shows exactly the failing checks with five red and three amber badges; a click on an input's label frames exactly the dashboard rows it drives and its own row until a second click; a click on a result frames exactly the Key design rows it reads and counts the traced rows per group; the traced filter (disabled without a trace, the traced count, the headline's traced heading); a heading shows the worst level of its group's checks (the headline and the closed coupling model red, the mass none); a heading click while searching leaves its group as it was, its hover text saying why; the failing filter with a search no failing check matches says no result matches; an input whose results have no equation record traces none, says so, and the traced filter then says the trace marks nothing (Clear trace turns the filter off); reading a traced result keeps the input's trace and its traced rows, a result without a record keeps it, a result outside it replaces it; idle frames in the workflow order's filtered view change nothing; the final review's fixes: out of the end-effect range the table's headline rows carry exactly the dashboard's badges (none on the hot minimum, the screw and the gearbox check) and the failing filter leaves the hot minimum out (seven red, four amber); a search that hides the traced rows says no result matches, and a heading counts the traced rows it shows; a trace moves no input row, so a second click at the same point ends it; the locked free variable's label traces it with no edit or undo step; views stay in their own area: on a 1280 x 620 page and the laptop screen, with the Equation panel at its starting height and dragged as far up as it goes, on a 1280 x 300 page and inside an `egui::Window` (the linkage Tools window's 1100 x 700), no view (geometry, the five plots, clamp, results) paints into the panel or takes a click inside it (shapes split from the panel's by its separator line, widgets by its resize handle), the scrolling views fill the space down to it, and the window keeps its size; the panel opens at about 45 % of the centre region and its drag stops `VIEW_STRIP` short of the region's top, then follows the pointer back down; every view draws beside the panel in a tiny window; both spreadsheet buttons (the header's and the results table's) queue `magcoupling-results.xlsx`, which reads back with the four sheets, the edited face gap as a number and the share link, follows the input order shown and, in Torque -> Magnets, holds the solved length (noted) and the solve's outcome; slice 1 (BL-055): the panel opens on Size and torque under the welcome strip, which closes for the session and leaves no undo step; the dashboard's status line counts the failing checks at the default design and after a fix; the Required slider edits `metal.required_min_Nm` (one undo restores it) and the Hot-low gap follows it; the consequence lines follow an input edit; Auto locks the matching slider, re-solves after the debounce, leaves idle frames alone and reports an unreachable requirement; opening the Size tab no longer changes the sizing mode; every new string has glyphs in egui's default fonts; slice 2 (BL-056): the Show spreadsheet details toggle starts off, a click turns it on and off, its hover says what it shows, and it is in no design file, share link or undo step and survives a loaded file, undo, reset all and a share link opened as the session's start; off, the face-gap row's hover names no path or cell and the pull-out's hover on the dashboard and on the Size tab no path, cell or correction, on they do; the correction ids and their legend, the results table's cells (pinned tests switch the details on), the Equation panel's cell and corrections lines and a result-without-record's cell, and the assumptions' sources show only with the details; at 930 px the table's label and value stay on screen with the cell and marker columns and without them; the toggle's label and hover, the legend, the marker heading and scope, the original-note prefix and the example design's status start and hover pass `plain_text_violations`; every text, the details' included, has glyphs in the default fonts; one name per quantity (`size_controls.rs` tests too): the Size tab's readout labels equal the input and dashboard rows' names (Required torque, Hot-low torque; one Hot-low torque readout in the centre beside the dashboard row of that name), the Auto row and its hover name the required torque, the footnote and the Outside diameter and Overall length hovers say what they move (the ring radius; both rings' magnet length) within the help caps (the length hover within its own cap, 4 sentences and 80 words, since BL-074: it also says that the first move from a blank axial magnet length sets both rings to one length, a test seen red first), each label shows its hover, the open equation's header is the display label, and Auto's no-valid-value line names the ring radius; the Outside diameter slider's lowest value is one expression, `slider_minimum` (the user's Minimum OD, else the range's start or a loaded smaller design's own diameter, held to the Minimum OD's range), which the controls pin on a loaded smaller design's first frame (`the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown`, slice 3); each Size slider makes its design through its own mapping, `SliderMapping`, which the torque card's steps share, the Outside diameter's at the slider's own minimum (at a Minimum OD below the default floor it reaches a diameter that floor refuses), the count's refusing a value a cast would change (not finite, not whole or past what an i64 holds) and neither the gap's nor the count's checking a range (`each_size_slider_makes_its_design_through_its_own_mapping`, slice 3); slice 2, help batch A (T7): each Key design row draws its hint line under it and above the next row (once: its own group open adds none), each of the five Size-tab controls its hint line under its label and above the next control's, and at a 930-point window every hint stays inside the inputs side, at least one wrapped; the panel's own strings (the assumption reset, the order toggle, the workflow headings, the trace banners and hint, the explorer's no-equation text and tag, the end-effect banner and greyed hover, the geometry notes, the welcome strip) pass the plain-words guard and name none of the engine's internal words; with Show spreadsheet details on, the dashboard keeps its width frame after frame (each value and its correction ids flow as one wrapped text); with Auto on (Torque -> Magnets, the axial magnet length free), Start here's step-1 note draws its magnet-length entry as plain text " (held by Auto)" once, a click on that label opens no row and keeps the note open, and its live values are the design shown's: the pull-out watch-out, which moves with the magnet length, reads the solved length's numbers and not the stored inputs' (Review Focus RF-1); the shaft-clamp pause (ruling 11): the Clamp tab and the clamp inputs (the workflow order's clamp sections, the original-sheet order's Shaft clamps group, the filter's matches) show only while the section is not paused, its two assumptions leave the Assumptions view and the banner, the shipped panel starts from the switch and passes its pause to the fixes (at the clamp's limit the hot-low torque's fix is drawn only while paused), the header's banner and the trace banner (`the_panel_passes_its_pause_to_the_fixes_the_banner_and_the_trace`), a design file written before the pause and a share link keep their clamp values in a paused panel (the engine still computes the section); one test, `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`, walks `every_shown_text` (every drawn text, and the hover texts of inputs, dashboard rows, table rows, callouts and the named static hovers; the rows through the panel's own filters) once per state: every such text has glyphs paused or not, and while paused none of them, nor a shown note's fields and drawn change entries, the Start here walk and hover, a counting guide's name and Why, the shaft joints' Problem as drawn on three designs that fail them (BL-064: the inner hub slipping, over its stress limit, on an uncooled wheel shaft), a Why link's reason or a shown assumption's name, rationale and source, names the clamp (the guard; no allow-list entry; no shown note's change entry or equation leads into the section; with the section shown the same enumeration names the clamp and reaches the Clamp tab, the clamp dashboard row, a clamp input row, the Shaft clamps group and the clamp friction assumption; while paused it reaches the label and help of every shaft-joint input and result and, each whole as drawn, their Shrink fit and Fitting temperatures sections and the Shaft joints headings of the original-sheet inputs and of the results table, BL-064); panel tests draw the whole section (`Harness::of` clears the pause) unless a test pauses it; slice 2 (T15): the gearbox ripple row's Why link opens the slip note, its first sentence drawn once and no input changed; slice 2 (T19, reachability, spec 3.7): the Size tab's Why links under Magnets per ring and Outside diameter, and on the torque card's heading (slice 3), open the torque note, and those at the end of the Required and Hot-low rows the torque-band note, each on its own in the Equation panel (clicked on the text's last character, `test_support::last_characters`, since a wrapped row's text rect also covers the slider before it); with an odd magnet count the Outside diameter control and its link are not drawn and the Magnets per ring link and the torque card's are; the link under each copy of the end-effect banner (over the dashboard and the centre) opens the end-effect note, and without the banner there is none; every dashboard row opened with Explain on shows its note, the closest note, an assumption or its own reason, never `NO_NOTE` (`explain_never_shows_no_teaching_note_on_a_dashboard_row`); the closest-note link, the one copy of the title on its line left of the dashboard, opens that note (`click_the_closest_note_link`, shared with the gearbox row's Explain test); with Explain on, a value with nothing upstream to read draws `NO_NOTE` and `START_HERE_HINT` once each, and Explain off draws neither; BL-059: the gearbox row's own Why link, the copy of the torque-band title under the row's label (the hot-low, hot-minimum and cold-high rows draw it first), opens that note and changes no input; BL-064: the shaft-joint row's own Why link, the one copy of the shaft-joint note's title on the dashboard, under the row's label, opens that note and changes no input (`the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note`, sharing `row_s_why_links` and `click_opens_note` with the gearbox row's test); with Explain on the gearbox row, which has no equation or note of its own, offers its Why link's note as the closest note, and the link beside the lead opens it; at the defaults the failing filter lists five red checks and one amber (the daily-cycle screen at the entered bondline, BL-063), the gearbox check's row once with a red badge (`table_badges_on`, shared with the end-effect test); slice 3 (T5): the Size tab draws the torque card in place of the label under the readouts, its heading and each of its lines once, one under the next, a click on the heading closing and opening it as one on its arrow does (`the_size_tab_draws_the_torque_card_in_place_of_the_label`); a typed Required value reaches the card in the frame that applies it (`a_required_edit_reaches_the_card_in_its_own_frame`); under Magnets per ring two lines, the ring-size line with Auto off or on the magnet length and the fit line, the card drawing the Auto line of each Auto state, the count's steps (at 4 magnets, and at 40 where 40 blocks fit, the side past the slider reading "outside the slider's range") and following each edit; idle frames compute the card once, a gap edit, a clamp-pause toggle and a new Minimum OD once each (the drawn Outside diameter row's -1 mm side then reading "outside the slider's range"), and turning Auto off at a solved length or Auto on the count at its best computes nothing, while the design's results the cache is handed are computed once per design, none for idle frames, the pause or the Minimum OD (`idle_frames_compute_the_card_once_and_an_edit_again`); a drag of the gap slider with Auto on the magnet length computes the card, and its design's results, once for each design it draws (`a_drag_computes_the_card_once_for_each_design_it_draws`); idle frames change no input with the card open and closed, at the clamp pause; slice 3 (T6, headless, each test pausing the clamp section before the card's first frame, but for the Equation panel's check on the screen, which draws every view unpaused): at 1280 x 800 the card sits under the Pull-out readout and above the drawing, open, every line drawn and the drawing at least 160 points, at the defaults and the smoke design; at 930 x 800 it starts closed (the region under 400 points), at short magnets too, closed no taller than the label it replaced, and its Why link wraps under the heading inside the region and opens the torque note; at 1080 x 745 no back iron starts closed (its open card would take the drawing under its floor) and short magnets open; each rule opens it at its boundary, a region exactly 400 points wide or exactly as tall as the open card, the item spacing and the floor, while 399.99 points or 0.01 point less, or the Equation panel open, closes it (`the_card_s_default_opens_at_each_rule_s_boundary`); with no user choice a window resize decides it again (open at 1280 x 800, closed again at 930 x 800), a change of its height alone too (at 1080 points wide open at 800 points tall, closed at 700, open again at 800); the Equation panel closes it and its closing opens it, and no view, the card open or closed, paints into the panel, on the screen or inside an `egui::Window`; a click on the heading or the arrow wins for the session, through a design edit, a resize, the Equation panel (the drawing keeping its 160 points beside it), Reset all and a load; with no user choice Reset all and a load keep the default decided for the view (at 1080 x 800 no back iron's closed card stays closed through Reset all and a load of 8 magnets, which would open it, and the defaults' open card stays open through a load of no back iron) (`reset_all_and_a_load_keep_the_card_s_default_with_no_user_choice`); drags of the gap, the Outside diameter across the fit edge and Required across hot-low (checked on every drag frame), the assumptions banner, and a side panel narrowing the region at no back iron leave it as decided; opened at 930 x 800, and at 1280 x 800 at a design file's calibration coefficient of 1e300, whose 887-character lines rule 3 starts closed, its body scrolls (the mouse wheel brings its last row into view) and the drawing keeps 160 points; its measured height is its drawn height within a point at 1280 x 800 and 930 x 1024, and at 1280 x 1024 at that calibration coefficient, and at 930 x 800 its body is the cap (`the_card_s_measured_height_is_the_drawn_height`); the look: one line in the accent, as wide as the active tab and just under its name, which follows a click to another tab, the active tab's name in the strong text colour and another's in the quiet one (a theme's weak colour; the text colour in egui's own visuals, which set none), and one accent square of 8 points before the title's first glyph, at the title's height (`the_active_tab_is_underlined_in_the_accent_and_no_other`, `the_brand_mark_is_one_accent_square_before_the_title`); the changed dot and a trace's banner and frames are the host's link colour, not the colour it gives a selected choice's name (`the_panel_s_marks_are_the_host_s_link_colour`); a readout's equation wider than the window is set smaller so its tooltip stays inside the window, and one that fits keeps the tooltip's size (`src/gui/readouts.rs`, `an_equation_wider_than_the_window_is_set_smaller_and_another_keeps_its_size`); BL-097: on the app's starting design, Auto on until solved and Auto off, Tab from the top of the page through the Size column changes no input, no result and no Minimum OD, makes no undo step and selects nothing (`tab_through_a_length_auto_left_off_its_step_changes_nothing`, with the `app` feature); with a face gap of 1.4123 mm and a requirement past its slider's range loaded, Tab past every value box of the Size column and of the open Detailed inputs does the same (`tab_past_every_value_box_keeps_a_value_off_its_step_or_outside_its_range`); Enter on a value box with nothing typed, or with the number it shows typed again, keeps the 1.4123, while a typed value edits on Tab (one undo step, the row selected), on Enter and on a click elsewhere, and an arrow key steps it onto the step (`a_typed_value_and_an_arrow_key_edit_a_value_off_its_step_and_its_own_text_does_not`); the rule itself (`src/gui/input_ui.rs`, `a_value_box_s_own_text_is_no_typed_number`); a number typed into a tiny value's box edits, one undo step, clamped to the range as egui clamps it: the vacuum permeability, the NdFeB expansion and the cap's field integral (`a_number_typed_into_a_tiny_value_s_box_edits`); BL-100: at 1280 x 800 and 1024 x 700 with the Detailed inputs open, after each press of Tab and of Shift+Tab every focused widget of the left column is whole in its view once the scroll has settled, scrolled no further than it was cut plus egui's gap at the edge; nothing is selected or edited; a wheel then takes the focused row out of view and it stays out; an idle frame asks for no repaint; a Down arrow key's move is followed too (`the_left_column_follows_the_keyboard_s_focus_and_fights_no_wheel`); the same walk through a check's entry in the Explain panel at 1024 x 700 (`the_explain_panel_follows_the_keyboard_s_focus`) and through the Geometry tab's callout list (under the open Equation panel) and the Clamp tab (`the_centre_s_scrolling_views_follow_the_keyboard_s_focus`); each fails with its area's call of `follow_focus` taken out; the left column wheeled 300 points down stays where the wheel left it while Tab, then Shift+Tab, pass the top bar, the centre and the right column (`a_wheeled_column_holds_still_while_the_focus_is_on_no_widget_of_its_own`; it fails with `follow_focus`'s clip check taken out); BL-101: at 1024 x 700 the same walk through the Equation panel, open on the hot-low torque and on the hot minimum check at its starting height and dragged to its least, every stop whole and the hot minimum check's formula, wider than the panel, from its start (`the_equation_panel_follows_the_keyboard_s_focus_into_its_terms_and_formula`; it fails with the panel's own call, with `sideways_scroll_area`'s second ask or with `follow_focus`'s content check taken out), through the Clamp tab with the screw-size table open, its cells past the table's right edge and under the view's foot among the stops (`the_screw_size_table_follows_the_keyboard_s_focus_up_down_and_sideways`), and through the right column in a window short enough for the check list to scroll (`the_check_list_follows_the_keyboard_s_focus_where_it_scrolls`; `the_explain_panel_follows_the_keyboard_s_focus` fails when `follow_focus` keeps its last focus by the content's id, which the check list and the Explain panel share); at 1024 x 700 and 1280 x 800, as the table opens (a group heading drawn past the foot) and wheeled 300 points down (at 1280 x 800 a row cut at the top), every Tab stop of the Results tab is whole as the focus arrives and the table does not move (`the_results_table_s_tab_stops_are_the_lines_it_shows_whole`; it fails with either of `line_sense`'s two callers given the plain click back); BL-086: at 1000 x 800 σ's assumption tag wraps onto two rows inside the panel (`the_term_list_wraps_inside_a_narrow_centre_region`), and every equation the registry holds, opened in the app's font in a window 1024 points wide on the starting design and on the example design with the details on, keeps the panel inside the centre region with no text cut but the formula's (`every_text_of_the_equation_panel_is_whole_on_every_equation_in_the_app_s_own_font`; it fails with a term's value laid out on one line); BL-098: at 1024 x 700 and 1280 x 800 every shown note in turn, then the note list, a check's entry, a tile's and the orientation, leaves the right column 300 points wide (egui's `PanelState`) with its right edge where it was (`the_right_column_keeps_its_width_whichever_note_is_shown`); so does the check list's material warning on a design with no back iron and on one whose sleeve conducts better than 316L, and its Why link opens its note (`the_right_column_keeps_its_width_with_a_material_warning_shown`); in the app's theme and font, on the starting design and on the failing example design with the spreadsheet details on, so does every entry the Explain panel can show: the orientation, the note list, every note, check, tile, Size column row and input, and every result of the results table (`the_right_column_keeps_its_width_whatever_entry_is_shown_in_the_app_s_own_font`, with the `app` feature); a diagram in a place 284 or 150 points wide is the picture scaled to that width with every shape and label inside it, and is 300 x 130 where it has room (`src/gui/diagrams.rs`, `a_diagram_takes_the_width_it_is_given_scaled_as_one_picture`); the five plots share one Plots tab (BL-104, decision C-9): the tab row is Size and torque, Geometry, Plots, Clamp and Results table, on one row at 1024 x 700, the picker under the Plots tab names the five plots in their order and marks the one shown, a click on a name shows its plot, the tab keeps the plot shown on a click and reopens the plot picked last from another view, picking edits nothing, selects nothing and is no undo step, a picked plot is drawn in the frame of its click, and the tab's word passes the plain-words guard (`the_five_plots_share_one_tab_whose_picker_shows_each_plot`); by the keyboard, Tab reaches the Plots tab and walks the picker's five names in their order with no view changed, Enter on the tab opens the plot shown last, and Space or Enter on a focused name shows its plot (`the_keyboard_reaches_the_plots_tab_and_picks_a_plot_by_space_or_enter`) |
| `src/gui/geometry.rs`, `src/gui/geometry_view.rs` (feature `gui`) | The end view draws every part from the results (the pocket's corners at the pocket corner radius, the rings at the retainers' diameters, the key, the shaft joints' 3 mm, from where it crosses the bore), faceted blocks on their flats and arcs as sectors; each end-view callout's line measures the gap it names (the corner gap along face 1's normal, from the dashed circle the inner corners sweep to the outer flat); red below zero and for a clearance below its target; the side view spans the housing in effect (a 50.8 mm override: the 53.6 mm cavity, the hub centred on it, the liner and sleeve the retainer span in effect); the space-claim callouts name each overshoot, in red, with the exceeded axes' dashed lines red (every axis at once too), exactly at the claim inside and the next number below over, and a NaN claim amber; a claim far past the pieces (1e300, 1e39, -1e39) is left off with one note, drawn at `CLAIM_REACH` times the extent and not past it; the design checks show while each inconsistency holds (equality is not below); the wall-rule line unless there is no back iron (the cup wall input named by its label), asking for the wall the cup wall check compares with, so the first slider step that passes is not asked for the verdict's rounded-up suggestion; that wall rounded up at its last digit, so a wall shown equal to it passes (12 poles, a 1.23 T design flux density and a 1.95 mm wall fail the check, and the line asks for 1.951 mm); a minimum length rounds up at its last shown digit (past a power of ten too, and a length already on a digit stays), and one not finite or not positive shows as it is; pole counts from a file draw at most 200 blocks; two poles draw their four blocks in a round pocket and hub (a 2-gon's corners would fly off), three a triangular pocket; a NaN draws nothing wrong and says so (the parts dropped counted per view). Painted: both views at one scale, the largest that fits (`side_by_side`: none without room or for an extent that is not finite); the cup OD, the shaft and the face gap at their pixel lengths, the cup depth and the cap as painted rectangles; the callout texts in their colours (the default clearance red, its line too); an overshoot red; hovering a dimension shows its result's hover text; each dimension tag sits on its own plate of the drawing's background; a NaN view and a tiny screen draw; a region of no size paints no view; a huge claim keeps the pieces to scale with its note; in a short region (400 points down to none) the view ends at its foot, the drawing keeping 160 points while the list keeps three rows, then shrinking to scale (`drawing_height`); in a region of exactly `drawing_floor` (slice 3: the room the torque card leaves, pinned at 212.125 points in egui's default style, 160 over three 17.375-point list rows) the drawing keeps its 160 points, and a point less it shrinks below them; a dimension's hover and a callout's in the list under the drawing name their result's path only with the spreadsheet details; the design checks name their inputs by their labels and pass the plain-words guard; the key in both states of the shaft-clamp pause, at the shaft joints' key width (BL-064: each end of its slider; one typed wider than the bore from the axis; a width or a keyway depth that is not a number not drawn and counted; none at a keyway depth of 0 or less, nor at a key width of 0 or less), never at the paused clamp's, and the clamp-boss design check only while the shaft-clamp section is not paused |
```

Replace:

```markdown
| `src/gui/torque_card.rs` (feature `gui`) | The count row is the neighbour lines that stood under Magnets per ring (decision 7A; their tests moved here from `gui/guidance.rs`, rewritten to the card's types and hot-low): each side is the design with only the count changed, or words where its blocks do not fit; at 4 magnets, and at 40 where 40 blocks fit, the side past the slider reads "outside the slider's range"; the count row skips the keyway (D-1); a count side reads a number exactly when sizing counts its design (now part of `a_side_reads_a_number_exactly_when_sizing_counts_its_design`); with no prediction on the design shown, or magnets too short for the model, the card compares nothing and draws its one line. The torque card (slice 3, BL-073): its four controls are the Size tab's, in the tab's order, each with its step, its slider's label and the Auto variable that sets it (`each_control_is_a_size_tab_slider_with_its_step`), and each row's slider is stepped as its own control at every design of the table (`each_control_s_slider_is_stepped_as_that_control`); over one table of designs, `card_designs()` in `gui/test_support.rs` (every design of its spec's section 2.1, T-14's one-flip design and R-13's boss plateau, with its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own `SizingRunner::shown`): every number is the difference of two `compute_all` runs of a design the test rebuilds through the slider's own mapping, bit for bit, the size changes against `dimensions` (none exactly where either design has none) and the calibration note against `model.f_cal` (`each_number_is_the_difference_of_two_compute_all_runs`); each side's words in the order of the tests, a step that lands within `STEP_SLACK_MM` of a slider's end a number, the jump words on every length side whose overlapping magnet length changes by anything other than the step (a cup cavity of 11.7 mm among them, whose +1 mm side lengthens the coupling by the step and leaves the overlap where it was, bit for bit, and one of 10.7 mm, whose +1 mm side moves the overlap by the step's size the other way), the cup's jump words on the boss plateau's +1 mm side, a count side whose calibration switch takes its torque to or below zero a number (to exactly zero drawn "+2 -> -0.85 N·m (-100.0 %)", spec 8.1 test 3's literal), and one whose torque is not finite "no prediction" (`the_words_follow_the_order_of_the_tests`); the design shown decides first: a magnet count below the slider's range before anything else (a design file's 3, 2, 1, 0 or -2: one line alone, where 3 and 2 would draw rows and 1, 0 and -2 the fit's line; the owner's FO-3), then the readouts' own rule, its not-finite half included for either torque (a design file's calibration coefficient of 0 at no back iron, both torques; a variation of -1e308, hot-low alone, the pull-out still 2.69 N·m), the line of a pull-out or hot-low of zero or less in words with no double negative (`the_design_shown_decides_first`); the size rows only where the Size tab draws them (`the_size_rows_follow_the_size_controls`); the diameter row steps as the slider does at the user's Minimum OD (`the_diameter_row_steps_as_the_slider_does_at_the_user_s_minimum`); a design shown past a slider's end (a design file's gap of 5.5 or 6.0 mm, an axial override of 60 mm, or of 0.5 mm under its slider with no end effect, an Overall length below its range, a ring radius of 31.0 mm past its slider, an Outside diameter above its range, 44 narrow magnets, which fit, past the magnet slider, the Minimum OD at its slider's top) reads "outside the slider's range" on both sides of that row (at (a) with the Minimum OD at 45 mm the +1 mm side too, which asks for 43.80 mm, below the slider's 45.00 mm, while the handle moves up: the words name no end, the owner's FO-1), and a side back on the slider is the slider's own design (the gap's +0.1 mm side at 0.2 mm, the count's -2 side at 42 narrow magnets) (`a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range`); moved with the N-2 and N+2 lines' rule from `gui/guidance.rs`: `count_step` keeps even counts inside 4 to 40 and never overflows (`count_step_keeps_even_counts_inside_the_slider_and_never_overflows`); modelled on the neighbour lines' test, which moved into it as its count-row part, a side reads a number exactly when sizing counts its design, at every row and through the card's count row, over the defaults, magnets too short for the model, the end effect at a step and the keyway through the hub, the keyway left out on the count row (`a_side_reads_a_number_exactly_when_sizing_counts_its_design`); the spec's numbers, and whole lines built from the constants so a reworded text flows through, every side word and the caution flip among them, with a caption that never says everything else is held, in any of the words a rewording might use for it, since at the reverse pair the length step trims the longer inner ring and no cost on that row shows it, and two jump texts that read alike, each naming the step's amount and direction, the boss plateau's Outside diameter row among the whole lines (`the_card_reads_the_spec_s_numbers`, `the_card_s_lines_read_as_drafted`); the flips are exactly the counted checks whose level differs, and each quoted change is of a check that fails on both designs, never the hot-low or the temperature check, in its result's own unit, and is that result on the stepped design less the design shown's, bit for bit (`the_flips_and_the_margins_name_the_checks_that_move`), the clamp named only while unpaused (`the_card_names_the_clamp_only_while_unpaused`); the held row and the Auto line, drawn outside the card, the held row with Auto off drawing each number side's N·m and its own percent (`the_held_row_reads_set_by_auto_and_the_card_is_the_same_design_s`); size changes and margins drawn only when they show (`size_changes_and_margins_are_drawn_only_when_they_show`), exactly where the decimals the Size tab writes them at read a digit, at any start (`a_change_is_drawn_exactly_where_its_own_decimals_read_a_digit`); a row whose two sides read alike flips Hot-low torque only on the side whose hot-low is below Required (T-14, the table's "T-14 one flip": `a_row_whose_two_sides_read_alike_flips_hot_low_only_where_it_crosses_required`); where the boss sets the Outside diameter (the table's "R-13 boss plateau") -1 mm reads "outside the slider's range" and +1 mm, which lifts the cup 37.78 -> 41.00 mm, the cup's jump words, not the +7.55 % it drew before as a slope (the owner's FO-2) (`where_the_boss_sets_the_outside_diameter_one_step_reads_the_slider_s_own_lift`); the +1 mm diameter side reads those words exactly where the shaft boss of the design shown sits above its cup and its cap, over the table, the defaults at every even count of the magnet slider and a boss just below and just above the plateau's cup (`the_diameter_row_reads_the_cup_s_jump_exactly_where_the_boss_sets_the_outside_diameter`); a step to a design without size controls (a design file's 6 magnets on a 38 mm ring, two fewer) draws its number and no size change (`a_step_to_a_design_without_size_controls_draws_no_size_change`); no number reads zero or disagrees with its percent (`no_card_number_reads_zero_or_disagrees_with_its_percent`); no card text ranks the controls, orders the torques, says the coupling protects the gearbox or names the clamp, and every one passes the plain-words guard (`the_card_texts_never_rank_the_controls`) and has its glyphs (`every_card_text_has_its_glyphs`); at finite values a design file or link can carry past every slider (a count from 0 to 1e6 and i64's ends; gaps, overrides, rings, cup depths, bosses, caps and bores at 0, negative or 1e300; a variation just under 1, at -1 or at 1e300, a magnet Br of 1e-6 or -1 T, a calibration coefficient of 1e-9, 1e-300, 1e-320 or 1e300, a Required of 1e-300 or -1, c_end 0 or 1e300, a gearbox rating of 0, a residual target of 1e300 and a hot ambient of ±1e300), at both back irons and at the Minimum OD's top, the card is drawn without a panic (the 188 cards draw 416 rows, counted), equals a second card of the same design, and every line of every Auto state passes the plain-words guard and has its glyphs, and, by the owner's FO-4 at the slice-3 STOP, no number on the card or in the readout's gap above it is wider than 12 characters and no N·m there reads zero beside a percent that reads a digit, and on a card row where one would, both of its N·m are their changes to within half a unit of the fourth significant digit (T-14; the fold-in's G-3 re-review, MF-2) (`the_card_never_panics_at_the_values_a_design_file_can_carry`); at the G-3 record's probe 13 the lines read as the record writes them ("-1 mm -> +1.690e-11 N·m (+1.9 %)" at a calibration coefficient of 1e-9 with no back iron at 8 magnets, "+2 -> +1.443 N·m (+1.612e11 %)", in four significant digits beside the row's "-5.182e-10" (MF-2), "+1 mm -> +8.401e298 N·m (+3.5 %)" and "margin +2.405e300 N·m (+9.622e301 %)" at one of 1e300, "margin +2.29 N·m (+2.285e302 %)" at a Required of 1e-300; `at_a_design_file_s_magnitudes_a_number_reads_four_significant_digits`); at slider values (both parts blank, manual blocks of 0.2 T, a gap of 3.0 mm, 4 magnets and a ring of 15.0 mm, each on its slider) the Outside diameter row, whose +1 mm side reads zero at the cap beside +0.7 %, draws both its N·m in four significant digits, "-5.033e-5" beside "+4.599e-5", never "-0.0001" (`at_slider_values_a_row_s_two_n_m_read_four_significant_digits_together`); every number is read from the card and every other word is the template's (`the_card_s_strings_are_constants_and_its_numbers_come_from_results`). |
| `src/gui/explorer.rs`, `src/engine/explain/notes.rs`, `src/engine/explain/note_text.rs`, `src/engine/explain/probes.rs`, `src/engine/picks.rs` (guidance slice 2) | A note draws its parts in order (what it is, What to change, Watch out in the warning colour, the Where this comes from fold, closed until clicked, then holding the sources); a free input's display label is a link to its row and the entry's text is not; an input Auto holds or a pick sets is plain text with the reason; an entry at the end of its slider its way says so (a count too; a choice and a blank optional number never); `held_reason` names Auto and the pick, in plain words; a note's link frames the input's row and keeps the note open; every note shows with glyphs, its sources once the fold is opened (`panel.rs`). A step reads as its word (Raise, Lower, Choose, Enter); an aim's sense holds only for a move its way (`Sense::holds`: strictly, never for an unmoved result or a NaN; the one rule the aim test and the review sheet share); the change rules name each kind of mistake (an entered number or a choice names no aim; a step names the result it moves, except a model assumption's) and hold for every note's entries; the one cap rule (`cap_problems`) names each kind of mistake, and the slice-2 caps (3 to 5 sentences of at most 25 words, 1 to 5 change entries, 0 to 2 watch-outs of at most 30 words) and the plain sentence 0 hold every note shown, on since T18 (`every_note_keeps_to_the_slice_2_caps`, `every_sentence_zero_is_plain_words`; while the clamp section is paused `clamp_preload` keeps its slice-1 text and is skipped); the before snapshot reads back and holds the 17 notes slice 2 started from. Start here follows slice 2's order (spec 3.3, `SLICE_2_ORDER`; a step whose note a later task adds is skipped until then), `torque_from_shear` first and owning the area-and-lever chain (the 2D torque, area times lever arm, gap radius, pole pitch, overlapping length), its sentence that first speaks of each pole also saying what a pole spans and its first watch-out saying what pull-out is, no text drawn before it naming pull-out (R8, `the_torque_note_glosses_each_pole_where_it_is_first_used`, `the_torque_note_glosses_pull_out_where_it_is_first_used`); the walk's first step reads "How the coupling makes torque" and the button's hover (`start_here_hover`) is plain words with that topic first; steps 2 to 6 (`harmonics`, `back_iron_factor`, `pullout_angle`, `end_effect`, `calibration`) each say what to change, `harmonics` owns the fill rows and `calibration` the prototype's modelled torque (`calibration.model_torque_Nm`); step 7, `br_temperature`, owns the three coefficient rows, and step 8, the torque band (`hot_low_cold_high`, owning the band's rows), follows it in the full order and in the order without the clamp step; step 9, the slip note (`when_slipping`, owning the gearbox ripple and the slip frequency), comes between the torque band and `slip_heating` in both orders, and `slip_heating` also owns the field frequency, the pole pairs and the slip speed in rad/s, `thermal_time_constant` the steady temperatures, the rise per slip event and the peak; step 12, `demagnetization`, also owns the two onset margins (`temperature.magnet_life.margin_onset_C` and `margin_limit_C`) and keeps the knee diagram, and step 13, `ferrite_cold_demag`, has no diagram (R7) and the only bar that adds "(ferrite rings only)" after the step count (`FERRITE_ONLY`), every bar naming its step of the steps shown and its note's title (`the_start_here_bar_names_every_step_it_shows`: 15 steps in the full order, 14 while the clamp section is paused, the shaft-joint step the 14th in both; the hover ends with its topic while paused; `the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink` pins its sentence 4 as drawn at the defaults, at library N42 parts, with mixed rings (inner N42M, outer N35: the outer ring governs, and the cap is the inner ring's own onset less the margin), and on a wheel shaft at 20 and 40 °C; `the_shaft_joint_note_keeps_the_caps_owns_the_joint_rows_and_precedes_the_clamp_step`: its 24 equations are the check's 23 numeric results and the inner ring's own single-ring onset); neither types a stored reverse field, the knee fraction or the design margin (`the_rewritten_notes_type_no_editable_default`), and no note says the coupling protects the gearbox (`no_note_says_the_coupling_protects_the_gearbox`, by `plain::mentions_gearbox_protection`, the rule the guide and overlay bans share); T15's three notes also keep to the caps and a plain sentence 0 per id, in A-15's form (`the_slip_notes_keep_the_caps`), and T16's two (`the_demagnetization_notes_keep_the_caps`). The warning notes own the rows slice 2 gives them (`the_wall_and_sleeve_notes_own_the_rows_slice_2_gives_them`): "Back-iron wall and saturation" (`a5.low_saturation`, also the cup-wall check's Why) owns the design flux density, the gap flux density, the wall the rule asks for, the suggested wall and the check; the conductivity note owns the sleeve conductivity it compares beside the two shell losses; the open-circuit note keeps the circuit in effect. At the defaults the outer library part sets the block's width and thickness, so the `harmonics` and `back_iron_factor` entries on them read "set by the Outer magnet part pick"; on custom dimensions they are links (Review Focus RF-4). With the 1018 back iron, which brings its own design flux density, the wall note's entry on that input reads "set by the Back iron material (hub, cup and boss) pick", and with 12L14 it is a link (RF-4, `the_wall_note_s_design_flux_entry_names_the_pick_that_sets_it`); with the sleeve at its slider minimum, the conductivity note's sleeve entry draws " (at the bottom of its range)" as plain text while the liner entry stays a link (`the_sleeve_entry_at_the_bottom_of_its_slider_is_plain_text_with_the_reason`). `set_by_pick`: only the library parts' manual fields at the defaults; each material pick's inputs; no input it names moves a result on any pick; every input a pick leaves without effect is named; a code outside the choices sets nothing. `SliderRange::step_from` keeps a step on the slider. Live values: `live_value` reads as the readouts and gives none where there is no prediction (a text such as "n/a", a number that is not finite, an input, no result, or a result computed from an out-of-range end effect, as for magnets at or below the end-effect length); a note draws its live values from the design shown, "no prediction" for the ferrite probe's hot onset; every note draws at every probe design with glyphs and no brace; `notes_for_review` (ignored) writes the per-note review sheet. `note_text`: text splits into literals and placeholders, each fault is named, values are rendered in place, placeholders are listed in drawing order. `probes`: the five designs are what their names say, each computes, counts for inverse sizing and draws on the Size tab; a step moves one input one slider step (a count by its step, a blank optional number from its seed) or names why it cannot (a choice or an entered number, no such input, a pick holds it, not a number, at its end). The plain-words rule for a note (`notes::plain_words_problems`, the one rule: a banned word or an audit id in the title, a source, or a text drawn with the caller's live values) names each kind of mistake, an audit id in a source or a watch-out among them, and reads a live value as drawn (`tests/explain.rs`'s `every_note_reads_in_plain_words_at_every_probe_design` runs it over every note shown at the five probe designs); the Equation panel's value line draws the cup wall's failing verdict with the wall input's label, and a passing wall as the engine gives it (ruling R9); the cup wall equation, in the Equation panel and every readout's hover, draws its verdict literal with the wall input's label while the record keeps the engine text (`typeset.rs`); the shaft-clamp pause: `START_HERE_ALL` ends with the clamp step and `start_here_for` drops exactly it while paused (`START_HERE` is the order for `clamps::PAUSED`), only `clamp_preload` is not shown while paused (`shown_when`, `shown`), the Start here hover names the clamp topic only while the section shows, and no shown equation reads a term of the section or links to it in its Used by; reachability (T19, spec 3.7): the empty Equation panel's text and Start here's hint name the Start here and Explain buttons, and Explain's own lines pass the plain-words guard with glyphs; every dashboard row without a note of its own offers a shown reviewed note, an assumption or its own reason (`note_fallback`), never `NO_NOTE`; over every explained result without a note or a Why link, the offered note is the nearest upstream (fewest steps, by the test's own level-by-level walk), an assumption only with no note upstream, and each kind of offer occurs; a row's own link comes first, and a value with no record has nothing upstream; at 400 x 800 with `pullout_angle` open alone, each change entry's label lies inside the panel and a click on the last character of the wrapped inner magnet part's label returns its `NoteLink::Input` (A-25 T-6); the paths the paused clamp section hides are skipped; the owner's sign-off sheet pairs every note shown with its text before slice 2 (T4's snapshot, `notes::before`; the three new notes say so, and the shaft-joint note, added after the slice by BL-064, says its own physics review is its record, in both switch states; the sheet is not regenerated for it), and while the clamp section is paused says why the clamp note is not on it, both switch states tested; every placeholder of every note has a value at the default design (`the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice`; `notes_signoff_sheet`, ignored, writes the sheet) |
| `src/gui/format.rs` (feature `gui`) | Four significant digits, scientific outside 1e-3 to 1e6, carries (9.99996 shows as 10.00), signed zero, `+inf`, `-inf`, `NaN` (one text for the display and both exports), integers, text, None, units (the sweep tables' "Nm" written N·m on screen; the metadata and exports keep "Nm"); `last_digit_step`, the place of the fourth significant digit, the step `mm_at_least` and the gearbox Problem's limits round by; the Size tab's one number format (slice 3, decision 10B): a change as a percentage to one decimal with its true sign (an exact zero of either sign "+0.0 %", a fall that rounds to zero "-0.0 %"), as N·m to the fewest decimals, from two, that read a digit that is not zero, at most two more (-0.0046 reads "-0.005", -0.00004 "-0.00"), a row's two N·m at one precision (-0.0063 and -0.0046 read "-0.006" and "-0.005"), the cap at any start (6e-17 from 14 decimals takes 16), the decimals search's fallback, and the readout's gap in that format (`size_controls.rs`: "short 0.29 N·m (-11.6 %)", "short 0.001 N·m (-0.0 %)"; a margin past the cap, "margin +0.00 N·m (+0.0 %)", and a hot-low that is not finite, "margin +inf N·m (+inf %)"; the word and the percent's sign agreeing at every finite gap, a hair either side of the requirement included; and its spec's section 5 table at (a), (b), (c), (e) and (e2)); and that format's bound, the owner's FO-4 at the slice-3 STOP: 999999.9 keeps its fixed form while 1e6, and 999999.996 at two decimals ("1000000.00"), reads "+1.000e6", a percent past six whole digits "+1.612e11 %", a torque change that reads zero at the cap beside a percent that reads a digit four significant digits ("+1.690e-11"), and beside a percent that reads zero, or with none drawn, its own decimals as before; a torque card row's two N·m at one precision under that bound (`shared_torque_precision`): both in four significant digits where either reads so ("-5.033e-5" beside "+4.599e-5", "+1.443" beside "-5.182e-10"), an exact zero still "+0.00", and every other row at its shared decimals; NaN and inf as before; the readout's gap the same, a row of one, a shortfall's size too ("short 1.000e300 N·m (-100.0 %)", "short 5.000e-7 N·m (-50.0 %)") |
```

with:

```markdown
| `src/gui/torque_card.rs` (feature `gui`) | The count row is the neighbour lines that stood under Magnets per ring (decision 7A; their tests moved here from `gui/guidance.rs`, rewritten to the card's types and hot-low): each side is the design with only the count changed, or words where its blocks do not fit; at 4 magnets, and at 40 where 40 blocks fit, the side past the slider reads "outside the slider's range"; the count row skips the keyway (D-1); a count side reads a number exactly when sizing counts its design (now part of `a_side_reads_a_number_exactly_when_sizing_counts_its_design`); with no prediction on the design shown, or magnets too short for the model, the card compares nothing and draws its one line. The torque card (slice 3, BL-073): its four controls are the Size tab's, in the tab's order, each with its step, its slider's label and the Auto variable that sets it (`each_control_is_a_size_tab_slider_with_its_step`), and each row's slider is stepped as its own control at every design of the table (`each_control_s_slider_is_stepped_as_that_control`); over one table of designs, `card_designs()` in `gui/test_support.rs` (every design of its spec's section 2.1, T-14's one-flip design and R-13's boss plateau, with its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own `SizingRunner::shown`): every number is the difference of two `compute_all` runs of a design the test rebuilds through the slider's own mapping, bit for bit, the size changes against `dimensions` (none exactly where either design has none) and the calibration note against `model.f_cal` (`each_number_is_the_difference_of_two_compute_all_runs`); each side's words in the order of the tests, a step that lands within `STEP_SLACK_MM` of a slider's end a number, the jump words on every length side whose overlapping magnet length changes by anything other than the step (a cup cavity of 11.7 mm among them, whose +1 mm side lengthens the coupling by the step and leaves the overlap where it was, bit for bit, and one of 10.7 mm, whose +1 mm side moves the overlap by the step's size the other way), the cup's jump words on the boss plateau's +1 mm side, a count side whose calibration switch takes its torque to or below zero a number (to exactly zero drawn "+2 -> -0.85 N·m (-100.0 %)", spec 8.1 test 3's literal), and one whose torque is not finite "no prediction" (`the_words_follow_the_order_of_the_tests`); the design shown decides first: a magnet count below the slider's range before anything else (a design file's 3, 2, 1, 0 or -2: one line alone, where 3 and 2 would draw rows and 1, 0 and -2 the fit's line; the owner's FO-3), then the readouts' own rule, its not-finite half included for either torque (a design file's calibration coefficient of 0 at no back iron, both torques; a variation of -1e308, hot-low alone, the pull-out still 2.69 N·m), the line of a pull-out or hot-low of zero or less in words with no double negative (`the_design_shown_decides_first`); the size rows only where the Size tab draws them (`the_size_rows_follow_the_size_controls`); the diameter row steps as the slider does at the user's Minimum OD (`the_diameter_row_steps_as_the_slider_does_at_the_user_s_minimum`); a design shown past a slider's end (a design file's gap of 5.5 or 6.0 mm, an axial override of 60 mm, or of 0.5 mm under its slider with no end effect, an Overall length below its range, a ring radius of 31.0 mm past its slider, an Outside diameter above its range, 44 narrow magnets, which fit, past the magnet slider, the Minimum OD at its slider's top) reads "outside the slider's range" on both sides of that row (at (a) with the Minimum OD at 45 mm the +1 mm side too, which asks for 43.80 mm, below the slider's 45.00 mm, while the handle moves up: the words name no end, the owner's FO-1), and a side back on the slider is the slider's own design (the gap's +0.1 mm side at 0.2 mm, the count's -2 side at 42 narrow magnets) (`a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range`); moved with the N-2 and N+2 lines' rule from `gui/guidance.rs`: `count_step` keeps even counts inside 4 to 40 and never overflows (`count_step_keeps_even_counts_inside_the_slider_and_never_overflows`); modelled on the neighbour lines' test, which moved into it as its count-row part, a side reads a number exactly when sizing counts its design, at every row and through the card's count row, over the defaults, magnets too short for the model, the end effect at a step and the keyway through the hub, the keyway left out on the count row (`a_side_reads_a_number_exactly_when_sizing_counts_its_design`); the spec's numbers, and whole lines built from the constants so a reworded text flows through, every side word and the caution flip among them, with a caption that never says everything else is held, in any of the words a rewording might use for it, since at the reverse pair the length step trims the longer inner ring and no cost on that row shows it, and two jump texts that read alike, each naming the step's amount and direction, the boss plateau's Outside diameter row among the whole lines (`the_card_reads_the_spec_s_numbers`, `the_card_s_lines_read_as_drafted`); the flips are exactly the counted checks whose level differs, and each quoted change is of a check that fails on both designs, never the hot-low or the temperature check, in its result's own unit, and is that result on the stepped design less the design shown's, bit for bit (`the_flips_and_the_margins_name_the_checks_that_move`), the clamp named only while unpaused (`the_card_names_the_clamp_only_while_unpaused`); the held row and the Auto line, drawn outside the card, the held row with Auto off drawing each number side's N·m and its own percent (`the_held_row_reads_set_by_auto_and_the_card_is_the_same_design_s`); size changes and margins drawn only when they show (`size_changes_and_margins_are_drawn_only_when_they_show`), exactly where the decimals the Size tab writes them at read a digit, at any start (`a_change_is_drawn_exactly_where_its_own_decimals_read_a_digit`); a row whose two sides read alike flips Hot-low torque only on the side whose hot-low is below Required (T-14, the table's "T-14 one flip": `a_row_whose_two_sides_read_alike_flips_hot_low_only_where_it_crosses_required`); where the boss sets the Outside diameter (the table's "R-13 boss plateau") -1 mm reads "outside the slider's range" and +1 mm, which lifts the cup 37.78 -> 41.00 mm, the cup's jump words, not the +7.55 % it drew before as a slope (the owner's FO-2) (`where_the_boss_sets_the_outside_diameter_one_step_reads_the_slider_s_own_lift`); the +1 mm diameter side reads those words exactly where the shaft boss of the design shown sits above its cup and its cap, over the table, the defaults at every even count of the magnet slider and a boss just below and just above the plateau's cup (`the_diameter_row_reads_the_cup_s_jump_exactly_where_the_boss_sets_the_outside_diameter`); a step to a design without size controls (a design file's 6 magnets on a 38 mm ring, two fewer) draws its number and no size change (`a_step_to_a_design_without_size_controls_draws_no_size_change`); no number reads zero or disagrees with its percent (`no_card_number_reads_zero_or_disagrees_with_its_percent`); no card text ranks the controls, orders the torques, says the coupling protects the gearbox or names the clamp, and every one passes the plain-words guard (`the_card_texts_never_rank_the_controls`) and has its glyphs (`every_card_text_has_its_glyphs`); at finite values a design file or link can carry past every slider (a count from 0 to 1e6 and i64's ends; gaps, overrides, rings, cup depths, bosses, caps and bores at 0, negative or 1e300; a variation just under 1, at -1 or at 1e300, a magnet Br of 1e-6 or -1 T, a calibration coefficient of 1e-9, 1e-300, 1e-320 or 1e300, a Required of 1e-300 or -1, c_end 0 or 1e300, a gearbox rating of 0, a residual target of 1e300 and a hot ambient of ±1e300), at both back irons and at the Minimum OD's top, the card is drawn without a panic (the 188 cards draw 416 rows, counted), equals a second card of the same design, and every line of every Auto state passes the plain-words guard and has its glyphs, and, by the owner's FO-4 at the slice-3 STOP, no number on the card or in the readout's gap above it is wider than 12 characters and no N·m there reads zero beside a percent that reads a digit, and on a card row where one would, both of its N·m are their changes to within half a unit of the fourth significant digit (T-14; the fold-in's G-3 re-review, MF-2) (`the_card_never_panics_at_the_values_a_design_file_can_carry`); at the G-3 record's probe 13 the lines read as the record writes them ("-1 mm -> +1.690e-11 N·m (+1.9 %)" at a calibration coefficient of 1e-9 with no back iron at 8 magnets, "+2 -> +1.443 N·m (+1.612e11 %)", in four significant digits beside the row's "-5.182e-10" (MF-2), "+1 mm -> +8.401e298 N·m (+3.5 %)" and "margin +2.405e300 N·m (+9.622e301 %)" at one of 1e300, "margin +2.29 N·m (+2.285e302 %)" at a Required of 1e-300; `at_a_design_file_s_magnitudes_a_number_reads_four_significant_digits`); at slider values (both parts blank, manual blocks of 0.2 T, a gap of 3.0 mm, 4 magnets and a ring of 15.0 mm, each on its slider) the Outside diameter row, whose +1 mm side reads zero at the cap beside +0.7 %, draws both its N·m in four significant digits, "-5.033e-5" beside "+4.599e-5", never "-0.0001" (`at_slider_values_a_row_s_two_n_m_read_four_significant_digits_together`); every number is read from the card and every other word is the template's (`the_card_s_strings_are_constants_and_its_numbers_come_from_results`). |
| `src/gui/notes_view.rs`, `src/engine/explain/notes.rs`, `src/engine/explain/note_text.rs`, `src/engine/explain/probes.rs`, `src/engine/picks.rs` (guidance slice 2) | A note draws its parts in order (what it is, What to change, Watch out in the warning colour, the Where this comes from fold, closed until clicked, then holding the sources); a free input's display label is a link to its row and the entry's text is not; an input Auto holds or a pick sets is plain text with the reason; an entry at the end of its slider its way says so (a count too; a choice and a blank optional number never); `held_reason` names Auto and the pick, in plain words; a note's link frames the input's row and keeps the note open; every note shows with glyphs, its sources once the fold is opened (`panel.rs`). A step reads as its word (Raise, Lower, Choose, Enter); an aim's sense holds only for a move its way (`Sense::holds`: strictly, never for an unmoved result or a NaN; the one rule the aim test and the review sheet share); the change rules name each kind of mistake (an entered number or a choice names no aim; a step names the result it moves, except a model assumption's) and hold for every note's entries; the one cap rule (`cap_problems`) names each kind of mistake, and the slice-2 caps (3 to 5 sentences of at most 25 words, 1 to 5 change entries, 0 to 2 watch-outs of at most 30 words) and the plain sentence 0 hold every note shown, on since T18 (`every_note_keeps_to_the_slice_2_caps`, `every_sentence_zero_is_plain_words`; while the clamp section is paused `clamp_preload` keeps its slice-1 text and is skipped); the before snapshot reads back and holds the 17 notes slice 2 started from. Start here follows slice 2's order (spec 3.3, `SLICE_2_ORDER`; a step whose note a later task adds is skipped until then), `torque_from_shear` first and owning the area-and-lever chain (the 2D torque, area times lever arm, gap radius, pole pitch, overlapping length), its sentence that first speaks of each pole also saying what a pole spans and its first watch-out saying what pull-out is, no text drawn before it naming pull-out (R8, `the_torque_note_glosses_each_pole_where_it_is_first_used`, `the_torque_note_glosses_pull_out_where_it_is_first_used`); the walk's first step reads "How the coupling makes torque" and the button's hover (`start_here_hover`) is plain words with that topic first; steps 2 to 6 (`harmonics`, `back_iron_factor`, `pullout_angle`, `end_effect`, `calibration`) each say what to change, `harmonics` owns the fill rows and `calibration` the prototype's modelled torque (`calibration.model_torque_Nm`); step 7, `br_temperature`, owns the three coefficient rows, and step 8, the torque band (`hot_low_cold_high`, owning the band's rows), follows it in the full order and in the order without the clamp step; step 9, the slip note (`when_slipping`, owning the gearbox ripple and the slip frequency), comes between the torque band and `slip_heating` in both orders, and `slip_heating` also owns the field frequency, the pole pairs and the slip speed in rad/s, `thermal_time_constant` the steady temperatures, the rise per slip event and the peak; step 12, `demagnetization`, also owns the two onset margins (`temperature.magnet_life.margin_onset_C` and `margin_limit_C`) and keeps the knee diagram, and step 13, `ferrite_cold_demag`, has no diagram (R7) and the only bar that adds "(ferrite rings only)" after the step count (`FERRITE_ONLY`), every bar naming its step of the steps shown and its note's title (`the_start_here_bar_names_every_step_it_shows`: 15 steps in the full order, 14 while the clamp section is paused, the shaft-joint step the 14th in both; the hover ends with its topic while paused; `the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink` pins its sentence 4 as drawn at the defaults, at library N42 parts, with mixed rings (inner N42M, outer N35: the outer ring governs, and the cap is the inner ring's own onset less the margin), and on a wheel shaft at 20 and 40 °C; `the_shaft_joint_note_keeps_the_caps_owns_the_joint_rows_and_precedes_the_clamp_step`: its 24 equations are the check's 23 numeric results and the inner ring's own single-ring onset); neither types a stored reverse field, the knee fraction or the design margin (`the_rewritten_notes_type_no_editable_default`), and no note says the coupling protects the gearbox (`no_note_says_the_coupling_protects_the_gearbox`, by `plain::mentions_gearbox_protection`, the rule the guide and overlay bans share); T15's three notes also keep to the caps and a plain sentence 0 per id, in A-15's form (`the_slip_notes_keep_the_caps`), and T16's two (`the_demagnetization_notes_keep_the_caps`). The warning notes own the rows slice 2 gives them (`the_wall_and_sleeve_notes_own_the_rows_slice_2_gives_them`): "Back-iron wall and saturation" (`a5.low_saturation`, also the cup-wall check's Why) owns the design flux density, the gap flux density, the wall the rule asks for, the suggested wall and the check; the conductivity note owns the sleeve conductivity it compares beside the two shell losses; the open-circuit note keeps the circuit in effect. At the defaults the outer library part sets the block's width and thickness, so the `harmonics` and `back_iron_factor` entries on them read "set by the Outer magnet part pick"; on custom dimensions they are links (Review Focus RF-4). With the 1018 back iron, which brings its own design flux density, the wall note's entry on that input reads "set by the Back iron material (hub, cup and boss) pick", and with 12L14 it is a link (RF-4, `the_wall_note_s_design_flux_entry_names_the_pick_that_sets_it`); with the sleeve at its slider minimum, the conductivity note's sleeve entry draws " (at the bottom of its range)" as plain text while the liner entry stays a link (`the_sleeve_entry_at_the_bottom_of_its_slider_is_plain_text_with_the_reason`). `set_by_pick`: only the library parts' manual fields at the defaults; each material pick's inputs; no input it names moves a result on any pick; every input a pick leaves without effect is named; a code outside the choices sets nothing. `SliderRange::step_from` keeps a step on the slider. Live values: `live_value` reads as the readouts and gives none where there is no prediction (a text such as "n/a", a number that is not finite, an input, no result, or a result computed from an out-of-range end effect, as for magnets at or below the end-effect length); a note draws its live values from the design shown, "no prediction" for the ferrite probe's hot onset; every note draws at every probe design with glyphs and no brace; `notes_for_review` (ignored) writes the per-note review sheet. `note_text`: text splits into literals and placeholders, each fault is named, values are rendered in place, placeholders are listed in drawing order. `probes`: the five designs are what their names say, each computes, counts for inverse sizing and draws on the Size tab; a step moves one input one slider step (a count by its step, a blank optional number from its seed) or names why it cannot (a choice or an entered number, no such input, a pick holds it, not a number, at its end). The plain-words rule for a note (`notes::plain_words_problems`, the one rule: a banned word or an audit id in the title, a source, or a text drawn with the caller's live values) names each kind of mistake, an audit id in a source or a watch-out among them, and reads a live value as drawn (`tests/explain.rs`'s `every_note_reads_in_plain_words_at_every_probe_design` runs it over every note shown at the five probe designs); the Equation panel's value line draws the cup wall's failing verdict with the wall input's label, and a passing wall as the engine gives it (ruling R9); the cup wall equation, in the Equation panel and every readout's hover, draws its verdict literal with the wall input's label while the record keeps the engine text (`typeset.rs`); the shaft-clamp pause: `START_HERE_ALL` ends with the clamp step and `start_here_for` drops exactly it while paused (`START_HERE` is the order for `clamps::PAUSED`), only `clamp_preload` is not shown while paused (`shown_when`, `shown`), the Start here hover names the clamp topic only while the section shows, and no shown equation reads a term of the section or links to it in its Used by; reachability (T19, spec 3.7): the empty Equation panel's text and Start here's hint name the Start here and Explain buttons, and Explain's own lines pass the plain-words guard with glyphs; every dashboard row without a note of its own offers a shown reviewed note, an assumption or its own reason (`note_fallback`), never `NO_NOTE`; over every explained result without a note or a Why link, the offered note is the nearest upstream (fewest steps, by the test's own level-by-level walk), an assumption only with no note upstream, and each kind of offer occurs; a row's own link comes first, and a value with no record has nothing upstream; at 400 x 800 with `pullout_angle` open alone, each change entry's label lies inside the panel and a click on the last character of the wrapped inner magnet part's label returns its `NoteLink::Input` (A-25 T-6); the paths the paused clamp section hides are skipped; the owner's sign-off sheet pairs every note shown with its text before slice 2 (T4's snapshot, `notes::before`; the three new notes say so, and the shaft-joint note, added after the slice by BL-064, says its own physics review is its record, in both switch states; the sheet is not regenerated for it), and while the clamp section is paused says why the clamp note is not on it, both switch states tested; every placeholder of every note has a value at the default design (`the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice`; `notes_signoff_sheet`, ignored, writes the sheet) |
| `src/gui/format.rs` (feature `gui`) | Four significant digits, scientific outside 1e-3 to 1e6, carries (9.99996 shows as 10.00), signed zero, `+inf`, `-inf`, `NaN` (one text for the display and both exports), integers, text, None, units (the sweep tables' "Nm" written N·m on screen; the metadata and exports keep "Nm"); `last_digit_step`, the place of the fourth significant digit, the step `mm_at_least` and the gearbox Problem's limits round by; the Size tab's one number format (slice 3, decision 10B): a change as a percentage to one decimal with its true sign (an exact zero of either sign "+0.0 %", a fall that rounds to zero "-0.0 %"), as N·m to the fewest decimals, from two, that read a digit that is not zero, at most two more (-0.0046 reads "-0.005", -0.00004 "-0.00"), a row's two N·m at one precision (-0.0063 and -0.0046 read "-0.006" and "-0.005"), the cap at any start (6e-17 from 14 decimals takes 16), the decimals search's fallback, and the readout's gap in that format (`size_controls.rs`: "short 0.29 N·m (-11.6 %)", "short 0.001 N·m (-0.0 %)"; a margin past the cap, "margin +0.00 N·m (+0.0 %)", and a hot-low that is not finite, "margin +inf N·m (+inf %)"; the word and the percent's sign agreeing at every finite gap, a hair either side of the requirement included; and its spec's section 5 table at (a), (b), (c), (e) and (e2)); and that format's bound, the owner's FO-4 at the slice-3 STOP: 999999.9 keeps its fixed form while 1e6, and 999999.996 at two decimals ("1000000.00"), reads "+1.000e6", a percent past six whole digits "+1.612e11 %", a torque change that reads zero at the cap beside a percent that reads a digit four significant digits ("+1.690e-11"), and beside a percent that reads zero, or with none drawn, its own decimals as before; a torque card row's two N·m at one precision under that bound (`shared_torque_precision`): both in four significant digits where either reads so ("-5.033e-5" beside "+4.599e-5", "+1.443" beside "-5.182e-10"), an exact zero still "+0.00", and every other row at its shared decimals; NaN and inf as before; the readout's gap the same, a row of one, a shortfall's size too ("short 1.000e300 N·m (-100.0 %)", "short 5.000e-7 N·m (-50.0 %)") |
```

Replace:

```markdown
| `src/gui/plain_text.rs` (feature `gui`) | Every overlay key is an input or a result (a hint's an input or one of the Size tab's own hint keys), once, with text; no help row repeats the engine's; the nine canonical names of spec 3.5 and the gearbox ripple row's (BL-059: Nominal slip swing at the gearbox input), and no other row, the Size tab's gap and count names equal to the input rows'; `label_for` falls back to the metadata label and never to the path, and never lengthens a label; the help mechanism on a stand-in table (the overlay, else the engine; the original is the spreadsheet's own note, kept only where it is not empty: the deviation registry's workbook text over a help a correction reworded, else the engine's help) and the public accessors over every input's own engine help; every label, help and hint passes the plain-words guard and the glyph test, calls no torque guaranteed, holds no brace and keeps its caps (help <= 30 words in 1-2 sentences, hint <= 12 words in one); no display label is another input's or result's name; no overlay row, and no engine label or help shown where the overlay has none, says the coupling protects the gearbox (`no_label_help_or_hint_says_the_coupling_protects_the_gearbox`, BL-059). Every input has help from the engine or the overlay (`every_input_has_help_from_the_engine_or_the_overlay`); `overlay_strings_for_review` prints the overlay for the physics review. Help batch A (T7, `help_tests`): every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint groups has help; hints exist exactly for the Key design inputs and the three `SIZE_TAB_HINT_KEYS` (none of them an input path); batch A help and every hint carry no digit; no hint types an input label (catalogue or display); the review sheet lists every input of its groups and every hint; `help_for_review` (ignored) writes the G-3 sheet (`HELP_REVIEW_GROUPS`, `HELP_REVIEW_OUT`). Help batch B (T8, `batch_b_tests`): the 52 empty helps of the Calibration and model, Materials, Housing and retainers, and Thermal and demagnetization groups each have a row (the set is computed from `WORKFLOW`), and 31 non-empty helps are replaced (the grade-mode density, rating and coefficient helps, the stored 3D fields, placeholder and assumed values, and helps with a cell or correction id or a stale fact), none of them Rust-only; the toggle's original note is the spreadsheet's own text (the deviation registry's for a help a correction reworded, none where that is empty); 175 direction claims hold, each a result that rises, falls or stays when a batch-B input rises one slider step on a named design (the example, no back iron, a 1018 back iron, a 304 back iron, a titanium sleeve, a 7075 cap, screw classes 10.9 and A4-70, the coercivity inputs, a measured drag, manual magnets with and without Y30 grades, library N42 parts in both rings), and so do the outside-diameter (the boss or the cap sets it only as the largest part), cap-resize, cup-wall, measured-drag, magnet-mass and temperature-check claims; a help that names the shaft joint its input feeds has its claim among them, the way its words say (`batch_b_helps_name_the_shaft_joint_they_feed`: the boss length lets the cup's fit hold more, and the single-ring reverse field can lower the inner hub's allowed fitting temperature, which falls with library N42 parts in both rings, where the onset binds, and stays at the defaults); each stored-3D, prototype, unmeasured-default and adapter cluster is exactly the help rows that end with its caveat (each caveat is typed once, as a macro the rows are built from); the stored-3D cluster is the dashboard's `STORED_3D_INPUTS` less the Rust-only ones, and its caveat is the engine's sentence (`temperature::stored_3d_sentence!`), the one literal `STORED_3D_SENTENCE` and the engine's free-space helps read too; no help repeats its input's default; no overlay row covers a Rust-only input, whose help the engine words (R1).; the joint-neutral texts (ruling 11) claim what the engine does: a stronger manual block and longer magnets raise the pull-out, the cold-high torque and its gearbox-input value (longer magnets the overall length too), and a larger bore, both shafts' diameter (both joints' holding torques move with it), thins the hub wall under the magnets and past the keyway: its help opens with both shafts and names the hub check (by its label), which the bore's slider top fails on the 8-magnet probe design, keyed or keyless, with the wall past the keyway still standing, and that wall breaks through at the slider top on the smallest ring; the keyway's help names both bores and says it caps the key's contact height: one step deeper thins the hub wall past the key and, with the contact height under the keyway, leaves both joints' key pressures bit for bit, while on a keyway shallower than the contact height (1.0 mm, the "shallow keyway" design) it lowers both; the gearbox texts claim what the engine does (BL-059, `the_gearbox_texts_claim_what_the_engine_does`): the rating's help names the gearbox check and the cold-high torque at the gearbox input, the ripple row's label and help the slip swing, the pull-out at the gearbox output at the operating temperature without the production allowance, through the ratio and efficiency, the check row's help (the engine's) both sides of the gearbox and that fatigue from repeated slips is not checked, the gear ratio's help the slip swing as the ripple row does (one name on screen), and no text says peak, warm or no check (the rating's type is open; the operating temperature can sit below the minimum); at the five probe designs the swing is the nominal pull-out divided by the ratio and efficiency, and moves when the operating temperature rises one slider step but not when the minimum temperature or the variation does; and with the rating halfway between the swing and cold-high at the gearbox input the check follows cold-high, above the swing at the defaults and below it with the operating temperature under the minimum (-40 against 20 °C, variation 0.1) |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the calculator's theme and opens the starting design (its status line, no changed dot, the example design's button, the requirement and its margin on the readouts); the starting design passes every counted check with no caution and no material warning, pinned to `compute_all` with the numbers the README quotes, changes seven inputs and its fourteen stored 3D fields on their slider steps and no assumption, stores the 3D fields of a recorded run of the reference routine for its geometry, and draws no caution on its opening tab; a link and a design file saved from the first starting design (main 2ca65ee) open that design as it was; the example design opens as one undo step and Reset all returns the starting design; the spreadsheet export still flags against the workbook (the seven inputs and fourteen 3D fields at the starting design, none at the example design); the page address is clean at the starting design and the share link elsewhere, follows settled edits, undo and redo and waits for an edit in progress; a Ctrl+Z or Ctrl+Y held for 60 frames writes the address once, at the key's release, and a write the page refuses is asked again on the next frame (the tests' page records each write and can refuse one); a link opened at start-up stays the address until the first edit, and one that cannot be read changes nothing; a link cut at 2,000 characters is refused, changes nothing and the header says why; a link stays under 4,000 characters; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link, the smoke prompt names the header's Show spreadsheet details toggle (`SHOW_DETAILS`), and its R14 steps name `WELCOME`, `explorer::WHAT_TO_CHANGE`, `explorer::WHERE_FROM`, the Geometry tab and the Magnets fit row as the page draws them, and its card step quotes `torque_card::CARD_HEADING` (the script's own `CARD_HEADING`, typed once for the first view and the card step), `CAPTION`, `AUTO_HOLDS_LENGTH` and `held_line(Control::OverallLength)` whole and names the other three rows by their starts (slice 3, test 26); the frame timer `FRAME_TIMER` is one line that holds the WebGL draw-call test (`drawElements`, `if (window.__drew)`), and the card step installs it with `page.addInitScript` (plan review 4.1); the drag step finds the gap slider by the value its box shows, the script's `SMOKE_GAP_SHOWN` (the link's 1.5 mm at the slider's step's two decimals, with its unit, typed once), and counts the sizing lines by `SIZING_LOG_PREFIX` in its code and its words; each page's console messages are collected after its last step, since the Playwright MCP returns only those since the last navigation; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the tokens are the UX overhaul spec's table value for value, the visuals are that table field for field and egui's dark visuals in every other field, and every text token is at least 4.5:1 on every surface it is drawn on, with the spec's own figures to one decimal (`the_tokens_are_the_spec_s_table`, `the_visuals_are_the_token_table`, `every_text_token_reads_on_every_surface_it_is_drawn_on`); the selection fill is at least 3:1 against a field and the panel and 2.8:1 on a tooltip, the text colour on it is at least 4.5:1, no lighter fill could give a tooltip 3:1 and keep that, it is a blue, and no other text colour reads on it (`selected_text_shows_on_every_surface_and_reads_on_its_fill`); the app's font: Inter leads the proportional family and the built-in fonts follow, the ten digits have one advance width at seven sizes and are the cut's own, the cut holds printable ASCII and the other characters the app's texts use and not a character Inter lacks, a row of the installed font is as tall as the built-in font's at eight sizes, to egui's rounding (`a_row_is_as_tall_as_the_built_in_font_s`), a Cyrillic letter and an emoji, both outside the cut, have a glyph with the font installed and none with the cut alone (`a_character_outside_the_cut_is_drawn_by_a_built_in_font`), and the app installs it while a panel in a plain context keeps egui's fonts (`src/app/fonts.rs`; `the_app_draws_its_text_in_inter`); on the app's page, at the starting design and at the failing example design, the accent is on two shapes (the brand mark and the active tab's line), on no text and on no badge (`the_accent_marks_the_brand_and_the_active_tab_and_nothing_else`); on the app's page a dashboard row's title, the heads of the Why, Problem and Fix lines and the Size tab's readout labels are drawn in the muted token and their values in the text token, at the starting design and at the failing example design, and a tab that is not the active one is muted while the active tab's name is the text token (`a_label_is_muted_and_its_value_is_in_the_text_colour`); in the app's own theme and font, at 1280 x 800 and at 1024 x 700, no text of any tab (each plot through the Plots tab's picker, since BL-104) is cut sideways by the rect egui clips it to or by the window, a plot's rotated axis label included (`no_text_is_cut_in_any_tab_in_the_app_s_own_font`: the panel's layout tests draw in egui's built-in font, which is about 13 % narrower, so only this test sees a text that no longer fits), and the widest tooltip, the Verdict row's, stays inside the window (`the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font`); BL-099: at 1024 x 700, 1280 x 800 and 1920 x 1080 the left column is 320 points wide (egui's `PanelState`) and no text is cut, on any tab or on any row of the column, with the folds shut, each fold opened and shut again, every input group and its Advanced heading opened and shut again, a filter that lists every input set and cleared, an input traced, and Auto on with the Detailed inputs open (`the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font`); every input row, as it is, changed, changed and outside its slider's range, and locked by Auto, takes no more than the width it is given, at the three widths the column gives a row, in egui's default font and in the app's (`every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font`); every input row whose label wraps at the narrowest of those widths, changed, has as many Tab stops as at a width where nothing wraps, and its reset button sits close under the label, in either font (`an_input_label_that_wraps_is_one_tab_stop_in_either_font`); BL-098: at 1280 x 800 and 1024 x 700, with a note that has a diagram shown through Learn and its title, the right column is 300 points wide and no text is cut on any tab (`the_right_column_keeps_its_width_with_a_note_shown_in_the_app_s_own_font`); the hub page's card for the calculator does not name the paused clamp section (`the_hub_page_does_not_name_the_paused_clamp_section`); the smoke script's `PAGE_BACKGROUND`, typed once, is the theme's panel colour as a browser reports it, and its clean-page step judges the look, reads the page's background and names the title and the active tab; the theme stays dark when the system turns light; BL-107: a control at rest shows its edge, by its fill or its outline, at 3:1 or more on the panel, a menu and a field, under the pointer and pressed a box keeps its fill while its outline brightens and a handle's ring stays the text colour, and an open picker keeps the resting outline (`a_control_at_rest_shows_its_edge_at_three_to_one_on_every_surface_it_sits_on`); with both folds of the left column open every rect and circle it draws, and the Design menu's checkbox, shows its edge at 3:1 or more against what is behind it (`every_control_shows_its_edge_on_the_panel_and_in_the_design_menu`); Auto's picker, opened, keeps an outline at 3:1 or more on the panel (`an_open_picker_keeps_its_edge`); BL-108: on all five plots of the starting and the example design every legend entry's dot is 3:1 or more on its disc and every disc's ring 3:1 or more on the legend, an entry clicked hidden keeps its disc with no dot and a second click shows it, the eight colours of the plots' series all drawn (`a_plot_legend_s_dots_show_at_three_to_one_on_their_discs_shown_or_hidden`); BL-109: on the first view at 1280 x 800 each tile's resting outline is the resting outline of a framed control, 3:1 or more on the panel, and a click on the Hot-low tile outlines it in the text colour at the same width with no tile moved (`a_tile_shows_its_edge_on_the_panel_as_a_framed_control_does`); the title and the left column's heading, and no other text, are Inter SemiBold, and the top bar stays one row with the heading on one line in the column, at 1024 x 700, 1280 x 800 and 1920 x 1080 (`the_title_and_the_size_heading_are_semibold_and_the_top_bar_keeps_one_row`); the SemiBold cut holds the Regular cut's characters, tabular digits and rows and is wider (`the_semibold_cut_has_the_regular_cut_s_characters_digits_and_rows`), and leads its own family with egui's Heading style in it (`inter_leads_the_proportional_family_and_the_built_in_fonts_follow`); the page names its language, the browser's colour and a dark scheme, shows `LOADING_LINE` after the canvas in the muted colour, and its script does not touch the line (`the_page_wears_the_app_s_colours_and_a_loading_line_until_the_app_draws`); the smoke script's `LOADING_ID` is the page's, and its look step reads the line gone and judges the heavier title and the controls' grey edge |

```

with:

```markdown
| `src/gui/plain_text.rs` (feature `gui`) | Every overlay key is an input or a result (a hint's an input or one of the Size tab's own hint keys), once, with text; no help row repeats the engine's; the nine canonical names of spec 3.5 and the gearbox ripple row's (BL-059: Nominal slip swing at the gearbox input), and no other row, the Size tab's gap and count names equal to the input rows'; `label_for` falls back to the metadata label and never to the path, and never lengthens a label; the help mechanism on a stand-in table (the overlay, else the engine; the original is the spreadsheet's own note, kept only where it is not empty: the deviation registry's workbook text over a help a correction reworded, else the engine's help) and the public accessors over every input's own engine help; every label, help and hint passes the plain-words guard and the glyph test, calls no torque guaranteed, holds no brace and keeps its caps (help <= 30 words in 1-2 sentences, hint <= 12 words in one); no display label is another input's or result's name; no overlay row, and no engine label or help shown where the overlay has none, says the coupling protects the gearbox (`no_label_help_or_hint_says_the_coupling_protects_the_gearbox`, BL-059). Every input has help from the engine or the overlay (`every_input_has_help_from_the_engine_or_the_overlay`); `overlay_strings_for_review` prints the overlay for the physics review. Help batch A (T7, `help_tests`): every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint groups has help; hints exist exactly for the Key design inputs and the three `SIZE_TAB_HINT_KEYS` (none of them an input path); batch A help and every hint carry no digit; no hint types an input label (catalogue or display); the review sheet lists every input of its groups and every hint; `help_for_review` (ignored) writes the G-3 sheet (`HELP_REVIEW_GROUPS`, `HELP_REVIEW_OUT`). Help batch B (T8, `batch_b_tests`): the 52 empty helps of the Calibration and model, Materials, Housing and retainers, and Thermal and demagnetization groups each have a row (the set is computed from `WORKFLOW`), and 31 non-empty helps are replaced (the grade-mode density, rating and coefficient helps, the stored 3D fields, placeholder and assumed values, and helps with a cell or correction id or a stale fact), none of them Rust-only; the toggle's original note is the spreadsheet's own text (the deviation registry's for a help a correction reworded, none where that is empty); 175 direction claims hold, each a result that rises, falls or stays when a batch-B input rises one slider step on a named design (the example, no back iron, a 1018 back iron, a 304 back iron, a titanium sleeve, a 7075 cap, screw classes 10.9 and A4-70, the coercivity inputs, a measured drag, manual magnets with and without Y30 grades, library N42 parts in both rings), and so do the outside-diameter (the boss or the cap sets it only as the largest part), cap-resize, cup-wall, measured-drag, magnet-mass and temperature-check claims; a help that names the shaft joint its input feeds has its claim among them, the way its words say (`batch_b_helps_name_the_shaft_joint_they_feed`: the boss length lets the cup's fit hold more, and the single-ring reverse field can lower the inner hub's allowed fitting temperature, which falls with library N42 parts in both rings, where the onset binds, and stays at the defaults); each stored-3D, prototype, unmeasured-default and adapter cluster is exactly the help rows that end with its caveat (each caveat is typed once, as a macro the rows are built from); the stored-3D cluster is the dashboard's `STORED_3D_INPUTS` less the Rust-only ones, and its caveat is the engine's sentence (`temperature::stored_3d_sentence!`), the one literal `STORED_3D_SENTENCE` and the engine's free-space helps read too; no help repeats its input's default; no overlay row covers a Rust-only input, whose help the engine words (R1).; the joint-neutral texts (ruling 11) claim what the engine does: a stronger manual block and longer magnets raise the pull-out, the cold-high torque and its gearbox-input value (longer magnets the overall length too), and a larger bore, both shafts' diameter (both joints' holding torques move with it), thins the hub wall under the magnets and past the keyway: its help opens with both shafts and names the hub check (by its label), which the bore's slider top fails on the 8-magnet probe design, keyed or keyless, with the wall past the keyway still standing, and that wall breaks through at the slider top on the smallest ring; the keyway's help names both bores and says it caps the key's contact height: one step deeper thins the hub wall past the key and, with the contact height under the keyway, leaves both joints' key pressures bit for bit, while on a keyway shallower than the contact height (1.0 mm, the "shallow keyway" design) it lowers both; the gearbox texts claim what the engine does (BL-059, `the_gearbox_texts_claim_what_the_engine_does`): the rating's help names the gearbox check and the cold-high torque at the gearbox input, the ripple row's label and help the slip swing, the pull-out at the gearbox output at the operating temperature without the production allowance, through the ratio and efficiency, the check row's help (the engine's) both sides of the gearbox and that fatigue from repeated slips is not checked, the gear ratio's help the slip swing as the ripple row does (one name on screen), and no text says peak, warm or no check (the rating's type is open; the operating temperature can sit below the minimum); at the five probe designs the swing is the nominal pull-out divided by the ratio and efficiency, and moves when the operating temperature rises one slider step but not when the minimum temperature or the variation does; and with the rating halfway between the swing and cold-high at the gearbox input the check follows cold-high, above the swing at the defaults and below it with the operating temperature under the minimum (-40 against 20 °C, variation 0.1) |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the calculator's theme and opens the starting design (its status line, no changed dot, the example design's button, the requirement and its margin on the readouts); the starting design passes every counted check with no caution and no material warning, pinned to `compute_all` with the numbers the README quotes, changes seven inputs and its fourteen stored 3D fields on their slider steps and no assumption, stores the 3D fields of a recorded run of the reference routine for its geometry, and draws no caution on its opening tab; a link and a design file saved from the first starting design (main 2ca65ee) open that design as it was; the example design opens as one undo step and Reset all returns the starting design; the spreadsheet export still flags against the workbook (the seven inputs and fourteen 3D fields at the starting design, none at the example design); the page address is clean at the starting design and the share link elsewhere, follows settled edits, undo and redo and waits for an edit in progress; a Ctrl+Z or Ctrl+Y held for 60 frames writes the address once, at the key's release, and a write the page refuses is asked again on the next frame (the tests' page records each write and can refuse one); a link opened at start-up stays the address until the first edit, and one that cannot be read changes nothing; a link cut at 2,000 characters is refused, changes nothing and the header says why; a link stays under 4,000 characters; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link, the smoke prompt names the header's Show spreadsheet details toggle (`SHOW_DETAILS`), and its R14 steps name `WELCOME`, `notes_view::WHAT_TO_CHANGE`, `notes_view::WHERE_FROM`, the Geometry tab and the Magnets fit row as the page draws them, and its card step quotes `torque_card::CARD_HEADING` (the script's own `CARD_HEADING`, typed once for the first view and the card step), `CAPTION`, `AUTO_HOLDS_LENGTH` and `held_line(Control::OverallLength)` whole and names the other three rows by their starts (slice 3, test 26); the frame timer `FRAME_TIMER` is one line that holds the WebGL draw-call test (`drawElements`, `if (window.__drew)`), and the card step installs it with `page.addInitScript` (plan review 4.1); the drag step finds the gap slider by the value its box shows, the script's `SMOKE_GAP_SHOWN` (the link's 1.5 mm at the slider's step's two decimals, with its unit, typed once), and counts the sizing lines by `SIZING_LOG_PREFIX` in its code and its words; each page's console messages are collected after its last step, since the Playwright MCP returns only those since the last navigation; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the tokens are the UX overhaul spec's table value for value, the visuals are that table field for field and egui's dark visuals in every other field, and every text token is at least 4.5:1 on every surface it is drawn on, with the spec's own figures to one decimal (`the_tokens_are_the_spec_s_table`, `the_visuals_are_the_token_table`, `every_text_token_reads_on_every_surface_it_is_drawn_on`); the selection fill is at least 3:1 against a field and the panel and 2.8:1 on a tooltip, the text colour on it is at least 4.5:1, no lighter fill could give a tooltip 3:1 and keep that, it is a blue, and no other text colour reads on it (`selected_text_shows_on_every_surface_and_reads_on_its_fill`); the app's font: Inter leads the proportional family and the built-in fonts follow, the ten digits have one advance width at seven sizes and are the cut's own, the cut holds printable ASCII and the other characters the app's texts use and not a character Inter lacks, a row of the installed font is as tall as the built-in font's at eight sizes, to egui's rounding (`a_row_is_as_tall_as_the_built_in_font_s`), a Cyrillic letter and an emoji, both outside the cut, have a glyph with the font installed and none with the cut alone (`a_character_outside_the_cut_is_drawn_by_a_built_in_font`), and the app installs it while a panel in a plain context keeps egui's fonts (`src/app/fonts.rs`; `the_app_draws_its_text_in_inter`); on the app's page, at the starting design and at the failing example design, the accent is on two shapes (the brand mark and the active tab's line), on no text and on no badge (`the_accent_marks_the_brand_and_the_active_tab_and_nothing_else`); on the app's page a dashboard row's title, the heads of the Why, Problem and Fix lines and the Size tab's readout labels are drawn in the muted token and their values in the text token, at the starting design and at the failing example design, and a tab that is not the active one is muted while the active tab's name is the text token (`a_label_is_muted_and_its_value_is_in_the_text_colour`); in the app's own theme and font, at 1280 x 800 and at 1024 x 700, no text of any tab (each plot through the Plots tab's picker, since BL-104) is cut sideways by the rect egui clips it to or by the window, a plot's rotated axis label included (`no_text_is_cut_in_any_tab_in_the_app_s_own_font`: the panel's layout tests draw in egui's built-in font, which is about 13 % narrower, so only this test sees a text that no longer fits), and the widest tooltip, the Verdict row's, stays inside the window (`the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font`); BL-099: at 1024 x 700, 1280 x 800 and 1920 x 1080 the left column is 320 points wide (egui's `PanelState`) and no text is cut, on any tab or on any row of the column, with the folds shut, each fold opened and shut again, every input group and its Advanced heading opened and shut again, a filter that lists every input set and cleared, an input traced, and Auto on with the Detailed inputs open (`the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font`); every input row, as it is, changed, changed and outside its slider's range, and locked by Auto, takes no more than the width it is given, at the three widths the column gives a row, in egui's default font and in the app's (`every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font`); every input row whose label wraps at the narrowest of those widths, changed, has as many Tab stops as at a width where nothing wraps, and its reset button sits close under the label, in either font (`an_input_label_that_wraps_is_one_tab_stop_in_either_font`); BL-098: at 1280 x 800 and 1024 x 700, with a note that has a diagram shown through Learn and its title, the right column is 300 points wide and no text is cut on any tab (`the_right_column_keeps_its_width_with_a_note_shown_in_the_app_s_own_font`); the hub page's card for the calculator does not name the paused clamp section (`the_hub_page_does_not_name_the_paused_clamp_section`); the smoke script's `PAGE_BACKGROUND`, typed once, is the theme's panel colour as a browser reports it, and its clean-page step judges the look, reads the page's background and names the title and the active tab; the theme stays dark when the system turns light; BL-107: a control at rest shows its edge, by its fill or its outline, at 3:1 or more on the panel, a menu and a field, under the pointer and pressed a box keeps its fill while its outline brightens and a handle's ring stays the text colour, and an open picker keeps the resting outline (`a_control_at_rest_shows_its_edge_at_three_to_one_on_every_surface_it_sits_on`); with both folds of the left column open every rect and circle it draws, and the Design menu's checkbox, shows its edge at 3:1 or more against what is behind it (`every_control_shows_its_edge_on_the_panel_and_in_the_design_menu`); Auto's picker, opened, keeps an outline at 3:1 or more on the panel (`an_open_picker_keeps_its_edge`); BL-108: on all five plots of the starting and the example design every legend entry's dot is 3:1 or more on its disc and every disc's ring 3:1 or more on the legend, an entry clicked hidden keeps its disc with no dot and a second click shows it, the eight colours of the plots' series all drawn (`a_plot_legend_s_dots_show_at_three_to_one_on_their_discs_shown_or_hidden`); BL-109: on the first view at 1280 x 800 each tile's resting outline is the resting outline of a framed control, 3:1 or more on the panel, and a click on the Hot-low tile outlines it in the text colour at the same width with no tile moved (`a_tile_shows_its_edge_on_the_panel_as_a_framed_control_does`); the title and the left column's heading, and no other text, are Inter SemiBold, and the top bar stays one row with the heading on one line in the column, at 1024 x 700, 1280 x 800 and 1920 x 1080 (`the_title_and_the_size_heading_are_semibold_and_the_top_bar_keeps_one_row`); the SemiBold cut holds the Regular cut's characters, tabular digits and rows and is wider (`the_semibold_cut_has_the_regular_cut_s_characters_digits_and_rows`), and leads its own family with egui's Heading style in it (`inter_leads_the_proportional_family_and_the_built_in_fonts_follow`); the page names its language, the browser's colour and a dark scheme, shows `LOADING_LINE` after the canvas in the muted colour, and its script does not touch the line (`the_page_wears_the_app_s_colours_and_a_loading_line_until_the_app_draws`); the smoke script's `LOADING_ID` is the page's, and its look step reads the line gone and judges the heavier title and the controls' grey edge |

```

- [ ] **Step 2: Check.** `git diff --stat` lists the nine files and no other; `python -I -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ['docs/ai/backlog.yaml', 'docs/ai/02-system.yaml', 'docs/ai/03-structure.yaml', 'docs/ai/04-memory.yaml']]"` prints nothing; `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3` still reads `1157 passed`.

- [ ] **Step 3: Commit** the nine files: `docs(magcoupler): slice B stage 2 part 2b-iii-c docs sweep: README, docs/ai, register rows A-44, A-55 and A-56, owner questions 67 and 76, the design's header, BL-086, BL-088, BL-089, BL-101 and BL-112 in progress (BL-112 T6)`.

### Task 7: Release: the full gate, gui-smoke, the reviews, one fix wave, the merge and the push

**Model:** `fable` controller; fresh-context `fable` reviewers (while `fable` is out, back 2026-12-01, both on the session model, and the report says so); `haiku` at low effort for git; the smoke workflow's own agents on `sonnet`.

**Files:**
- Modify: `docs/ai/backlog.yaml` (Step 7); create `docs/analyses/2026-11-27-magcoupler-slice-b2p2biii-c-review.md` (Steps 4, 5 and 7).

- [ ] **Step 1: Preconditions.** Tasks 1 to 6 are committed, each reviewed; `git status --short` shows only the owner's untracked press analysis; `git diff main...HEAD --stat -- magcoupling-rs/src/engine magcoupling-rs/src/app magcoupling-rs/tests reference linkage-sim-rs .claude` is empty.

- [ ] **Step 2: The full gate.** From PowerShell at the worktree's root:

  ```powershell
  & 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/gate.sh --full
  git checkout -- docs/chebyshev_lambda
  ```

  Expected: `GATE PASS`. A red gate goes back to the controller with the gate's output; nothing is merged.

- [ ] **Step 3: gui-smoke, from the branch's own script.** Build the web calculator from the branch (`linkage-sim-rs/scripts/build_magcoupling_web.sh` through Git Bash), serve `linkage-sim-rs/web/` with `python -m http.server` on a free port other than 8080, run the `gui-smoke` workflow from this branch's `.claude/workflows/gui-smoke.js` with `{"frame_check": true}`, and stop the server after. Expected: `passed: true` for the calculator and a p95 under one 60 fps frame. Keep its first screenshot.

- [ ] **Step 4: The text review, with the pictures** (fresh context; the design's 8.3). No on-screen string is new or changed; one is gone, `EMPTY_TEXT`, which no state of the app showed. It checks that no other text is new, changed or gone (`git diff main...HEAD -- magcoupling-rs/src` with the moved strings matched word for word), and renders, in a browser from the build of Step 3, the Start here walk at 1024 x 700, 1280 x 800 and 1920 x 1080 (each step's Equation panel; the term rows and tags wrapped, the step line wrapped, the panel inside the centre region), the Equation panel on the high case's rotations to the limit at 1024 x 700 (its long text value wrapped in its own column), the Equation panel on the hot minimum check at 1024 x 700 (the formula past the edge, its scroll bar under the pointer), Tab and Shift+Tab through the Equation panel, the check list (in a window short enough for it to scroll) and the Results table at 1024 x 700. The Clamp tab is hidden while the shaft-clamp section is paused, so its screw-size table is held by Task 4's test, not seen in the browser. It rules on three things and records what it decides: whether the wrapped term rows still read as a list (decision D-110), whether the formula scrolling sideways is right or should be set smaller (decision D-109, owner question 76), and whether a Results table line half seen should take the focus after all (decision D-113, owner question 67). A change the owner should weigh goes to `docs/analyses/2026-11-18-owner-questions.md` as an open question with its conservative answer, not shipped silently. Its findings join Step 5's fix wave. Record it in `docs/analyses/2026-11-27-magcoupler-slice-b2p2biii-c-review.md` (committed in Step 7).

- [ ] **Step 5: The independent whole-branch review** (fresh context, three lenses in parallel, each with a skeptic on every finding that is not a suggestion). Give each `git diff main...HEAD`, the design, this plan and Step 4's pictures, and none of the builders' notes.
  - **The Equation panel and its text.** Every text of it whole on each Start here step and with each result open, at the three window sizes in the app's font and theme, the formula alone scrolling; `EMPTY_TEXT` gone with no orphaned guard or test; no on-screen string added, changed or removed but `EMPTY_TEXT`.
  - **Keyboard reach.** Tab and Shift+Tab through the Equation panel, the screw-size table, the Results table and the check list at 1024 x 700 and 1280 x 800; each focused widget scrolled into view, the columns holding still otherwise (the BL-097 to BL-100 tests); focus alone selects nothing (BL-093); no Tab stop added (BL-102); `follow_focus`'s content check and unique slot against every caller (the left column, the Explain panel, the callout list, the Clamp tab); Esc and the trace unchanged.
  - **Correctness and code.** BL-088 a pure move (diff the moved items against `main` byte for byte, but for the three changes Task 1 names); red steps real and no test weakened (the walk helper's two widenings); `sideways_scroll_area` reused, not copied; dead code; idle frames (no repaint loop added); the gui-smoke checks still hold.
  One fix wave follows, on the decider's model: each accepted finding gets its red test first. A finding that is not a defect of this part is filed in `docs/ai/backlog.yaml` with the next free id.

- [ ] **Step 6: If Step 4 or 5 changed a file, run Steps 2 and 3 again.** The merge needs a gate and a smoke run of the tree that is merged.

- [ ] **Step 7: Close the items and merge.** In `docs/ai/backlog.yaml` (the day's date; say what the reviews found), BL-112 and the four items it took:

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Task 2):
```

with:

```yaml
  status: fixed
  notes: "FIXED <date> with BL-112 (merged). Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Task 2):
```

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Task 1):
```

with:

```yaml
  status: fixed
  notes: "FIXED <date> with BL-112 (merged). Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Task 1):
```

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Task 3):
```

with:

```yaml
  status: fixed
  notes: "FIXED <date> with BL-112 (merged). Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Task 3):
```

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Tasks 4 and 5):
```

with:

```yaml
  status: fixed
  notes: "FIXED <date> with BL-112 (merged). Built on branch magcoupling/overhaul-b2p2b-iii-c with BL-112 (plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md, Tasks 4 and 5):
```

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-c (Tasks 1 to 5 of the plan);
```

with:

```yaml
  status: fixed
  notes: "FIXED <date>: merged (<the gate, the reviews and the smoke run in a few words>). Built on branch magcoupling/overhaul-b2p2b-iii-c (Tasks 1 to 5 of the plan);
```

  Write the date and the few words in place of `<date>` and `<...>` (the five blocks apply as written; the words are then typed into the file). Commit it with the review record: `docs(magcoupler): BL-112 fixed, with BL-086, BL-088, BL-089 and BL-101: the reviews, the gate and the smoke run passed`. Then merge the branch into `main` with a merge commit (`git merge --no-ff`), its message one line of what the screen now does.

- [ ] **Step 8: Push, then report** (the owner's ruling of 2026-11-22: each stage goes live once the gate, the smoke run and the reviews pass, with a report afterwards; no stop). Push `main` (it deploys the site). When the `deploy-web.yml` run is green, run the smoke workflow from `main`'s script with `{"url": "https://colesorkness.com"}` and report the result; if it fails, say so plainly: the site is live with the change. The report to the owner holds: what changed on screen, in three sentences; two pictures (a Start here step at 1024 x 700, the hot minimum check's formula at 1024 x 700); the decisions D-106 to D-115, one line each; owner question 76 and question 67's new line; that slice B is complete; which model ran the decider's steps; what was not verified (Safari and Firefox; a touch screen; a screen reader).

## What this plan covers of the design

| The design | Where |
|---|---|
| 8.2 test 15: the layout at 1024 x 700, 1280 x 800 and 1920 x 1080 (the Equation panel during the walk) | Task 3 |
| 5.5 Start here: each step opens its equation, the bar's step line | Task 3 (the step line wraps) |
| 8.3 Text rules: no text added; one gone | Task 2 (`EMPTY_TEXT` goes; no state of the app showed it) |
| 5.2 a hover selects nothing, nor the keyboard's focus arriving (BL-093) | Tasks 4 and 5 (each walk asserts the selection unchanged) |
| 9.1 The docs and the backlog item with the slice | Task 6 |
| 9.4 Release | Task 7 |
| The code's own debt (BL-088): the note drawing where the Explain panel reads it | Task 1 |
| Not in this part | Nothing of slice B; slices C to F, and the physics items BL-083, BL-084, BL-087 and BL-067 (a later part). |
