# Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-b (the look): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ] `) syntax for tracking.

**Goal:** The look's last three items (the design's section 6; slice B stage 1's whole-branch review). Every control at rest shows its edge at 3:1 or more against what is behind it (findings E-4 and R-6): a value box, a framed button, a picker and a text box are outlined, and a checkbox's box and a slider's rail and handle filled, in one more grey, where the hairline colour drew them at 1.1:1 to 1.2:1 on the panel; an open picker keeps that outline, and every plot legend's dot sits on a grey disc, egui_plot's mark of a toggle. The title and the left column's heading "Size and torque" are drawn in Inter SemiBold (suggestion S-1), a second cut made by the committed font script from the recorded Inter 4.1 file; every other text stays Regular, the top bar stays one row and the left column 320 points wide at 1024 x 700, 1280 x 800 and 1920 x 1080. The page names its language, gives the browser the panel colour and a dark scheme, and while the wasm file loads shows one line, "Loading the calculator", in the muted colour on the charcoal page; the app takes it away in the paint of its first frame (suggestion S-4). The gui-smoke run judges all three. Nothing that is on screen today is lost (the nothing-lost table).

**Architecture:**
- **One more colour.** `app::theme::CONTROL` (`#656c76`, a grey of the tokens' hue) is the resting outline (`widgets.inactive.bg_stroke`) and the resting fill (`inactive.bg_fill`), the fill under the pointer and pressed (`hovered.bg_fill`, `active.bg_fill`) and the open outline (`open.bg_stroke`). So a box keeps its fill while its outline brightens (muted, then text), and an open picker keeps its outline (egui draws a picker's box in the open state while its list shows). egui draws a value box, a framed button, a picker and a text box with that outline, and a checkbox's box, a radio button and a slider's rail and handle with that fill; its menus drop the outline, so the Design menu's checkbox shows by its fill. A slider's handle keeps its ring in the text colour (`fg_stroke`) in every state, so it shows the pointer, a drag and the keyboard's focus by the ring's width and its size alone, as since stage 1 (D-98). egui_plot draws each legend entry, a toggle that hides or shows its line, as a disc in the resting fill and outline under the line's dot, so every plot legend dot now sits on a grey disc (D-97; the panel's plot code is not touched). Frameless widgets (the top bar's words, the tabs, menu items, an unselected name in a row of choices) and the hairlines between the columns are not touched.
- **Two faces.** `app::fonts::install` sets the whole font definitions: Inter Regular first in the proportional family, as before, and Inter SemiBold first in a family of its own (`fonts::semibold()`), egui's built-in fonts behind each; then it points every style's `TextStyle::Heading` at that family, at the size it had. The panel's two `ui.heading` texts, the title and the Size column's heading, are SemiBold; the panel itself names no font. The SemiBold cut is a binary file a plan block cannot carry: Task 2's first step makes it with `assets/fonts/subset_inter.py` from the downloaded Inter 4.1 zip and checks three SHA-256 hashes.
- **The page shell.** `linkage-sim-rs/web/tools/magcoupler/index.html` gains `lang="en"`, `theme-color` and `color-scheme` in its head and, after the canvas, the line `app::LOADING_LINE` with the id `app::LOADING_ID`, styled in the muted colour across the middle of the page. The app removes it at the end of each frame on wasm32 (`remove_loading_line`, called by `eframe::App::update`); eframe paints the frame in the same task, so the line goes in the paint that shows the app's first frame. The page's script is unchanged (`init();`).
- **The smoke run.** The prompt's `look` check also reads that the line is gone and judges the heavier title and the controls' grey edges; no new check key, so the verdict test's count stays.
- **Unchanged:** the engine, the exports, every on-screen word but the page's one line, every position (the SemiBold rows are the Regular rows), the panel's code (`magcoupling-rs/src/gui/` is not touched), the hub page and the linkage app.

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32; fontTools 4.51.0 on Python 3.12 (Task 2 Step 1 only; the build runs no font tools); the headless egui harness (the app tests' `frame_on`, `click_at`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. This part is its section 6 (6.1 the tokens, 6.2 the font, its weights and its cost, 6.3 where the theme lives), goal G-7 (contrast), test 15 of section 8.2 (the layout at 1024 x 700, 1280 x 800 and 1920 x 1080), the text rules of section 8.3 and the docs and release rules of section 9. Executors read the spec's sections 6, 8.2, 8.3 and 9 beside this plan. Part 2b-iii-a's plan (`docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md`, "Why part 2b-iii is cut, and what follows", decision D-86) sets the scope, and slice B stage 1's plan (`docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`, "What follows" items 6, 7 and 9, and its table "Declined in this stage") holds the review's four items, E-4, R-6, S-1 and S-4; the review's own record is `C:\Users\Cole\source\repos\lsim-audit\.superpowers\sdd\2026-11-18-magcoupler-overhaul-b1\final-review.json` (gitignored, in the stage 1 worktree). Backlog item: **BL-107** (`docs/ai/backlog.yaml`, filed with this plan).

**How this plan was made.** The part was built once as a prototype on a copy of `main` at `f84bc4f`, one commit per step (Task 1's red and green; Task 2's font file with the script's record, its red and its green; Tasks 3 and 4's red and green), then BL-107 filed and the docs swept. Every block below was generated from those commits' diffs. The plan's review (seven findings) changed Task 1 (the open outline, a handle's ring and the open outline in the theme test, two app tests), the docs and the plan's text; the prototype's commits were made again with those changes, and the blocks generated and replayed again as below. The font file was made by the script from a copy of the Inter 4.1 zip downloaded for stage 1, whose SHA-256 equals the one the script records; the same run of the script on `extras/ttf/Inter-Regular.ttf` gave the committed Regular cut bit for bit, so the tool on this machine is the one that made it. Then the plan was replayed with the apply tool, in this plan's order, on a fresh copy of `main` with BL-107 filed, in a target directory of its own, Task 2 Step 1 running its commands as written (the zip downloaded anew into a new, empty directory). Each block applied once; each red step failed as its "Expected" says; the result equals the prototype file for file (line endings aside; the font file byte for byte) and passes `cargo test` with and without the app feature (1561 tests with it, 656 without; 1150 lib tests with it against 1143 on the base), both native clippy runs after each task, both wasm32 clippy runs, `cargo fmt --check` and the gui-smoke verdict test (`all 85 checks hold`). Each count after a task is the replay's. Not run for this plan: the full gate (`gate.sh --full`: gate 12, the Python oracle, and gates 1 to 3 on `linkage-sim-rs`, whose `tests/web_layout.rs` reads the page's import line; the prototype passed that one test file), the release WASM build through the build script, and the smoke workflow. Task 6 runs them. The prototype's release WASM build was measured and loaded in Chrome (headless, through Playwright) for the numbers under "What the prototype showed".

**Where this part applies.** On `main` at `f84bc4f` with this plan's commit (the plan and BL-107), and nothing else that touches `magcoupling-rs/`, `linkage-sim-rs/web/tools/magcoupler/index.html`, `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` or Task 5's nine files. If a block does not match, stop and report: the controller regenerates the blocks (the planner's tools are listed under "Before Task 1"), it does not adapt a block in place.

## Why part 2b-iii-b is one plan, and what follows

Part 2b-iii-a's plan cut part 2b-iii in three (decision D-86, register row A-47) and named this part's three items: the controls' outlines, a SemiBold face and the page shell's loading line. They are one subject, the look of section 6, and small: a new theme colour with its tests, a second font file with the code that installs it, and the page's head and one line. The four code tasks run from about 150 to 610 plan lines each, Task 1 the longest (the new colour and its four tests), then Task 2 (the font step, two tests and the new `install`), and one reviewer holds each with its list under "Files". So the part is not cut again (decision D-95).

**What follows, with its own plan and backlog item:** part 2b-iii-c (the Equation panel): BL-086 (the term rows, their tags and the walk bar's step line cut at the panel's edge during the Start here walk), BL-088 (the note drawing out of `explorer.rs` into its own module, a pure move), BL-089 (`EMPTY_TEXT` can no longer be seen), and BL-101's Equation panel part (its nested sideways scroll areas do not follow the keyboard's focus) if that plan holds it.

Not touched here: BL-083, BL-084 and BL-087 (physics: they wait for the fable decider, back 2026-12-01), BL-095, BL-102, BL-103, BL-105 and BL-106.

Kept as they are, by decision, from the review's items: a slider handle's state cue, its ring's width and its size (the stage 1 review's E-4 and its plan's "What follows" item 6 named it; D-98 keeps it, and owner question 73 offers the blue handle); and the hairlines between the columns, which E-4's title counted (1.21:1 on the panel): they are no control, WCAG 1.4.11 asks 3:1 of a control's edge and of the graphics a reader needs to understand the page, not of a dividing line, so they stay hairlines (D-97).

## What the prototype showed

All in the app's own theme and font (`MagcouplingApp::new`, which applies `theme::apply` and `fonts::install`), at a pixel ratio of 1.

1. **The edges on `main`.** With both folds of the left column open (a window 20,000 points tall, so every row draws), the column draws 93 filled or outlined shapes: 27 sliders, each a rail and a handle; 32 value boxes and pickers; 3 text boxes; 3 checkboxes; 1 selected choice (the order toggle). 65 of them are under 3:1 against the panel, each at 1.21:1 (a rail, a box, an outline in the border colour); the handles show by their ring in the text colour. The Design menu draws its checkbox's box with no outline (egui's `menu_style`), the border colour on the menu's `#1e2227`, 1.1:1.
2. **The grey.** The darkest grey of the tokens' hue that is 3:1 on the lightest surface a control sits on, a menu (`#1e2227`), is `#656c76`: 3.26:1 on the panel, 3.02:1 on a menu, 3.48:1 on a field (`#121417`), and the text colour's tick on it 4.32:1. The review's "about `#626973`" is 3.12:1 on the panel and 2.88:1 on a menu. With it every one of the 93 shapes and the menu's box is 3:1 or more. A selected choice keeps its fill (`#466a90`) with the grey edge at 1.06:1 to it, so the ring around it is barely seen.
3. **The menu fades in.** egui fades a menu in: two frames after the click its checkbox's fill is drawn at alpha 252 of 255, 2.97:1 by the test's measure; six frames draw it whole. Task 1's test lets six frames pass.
4. **The SemiBold cut.** The script, run on `extras/ttf/Inter-SemiBold.ttf` of Inter 4.1, prints `435 glyphs, 386 characters, digit advance 1325` (the Regular cut: 435, 386, 1328) and writes 30,524 bytes; a second run writes the same bytes. Both cuts hold the same characters, keep the zip's copyright and licence lines (name IDs 0, 13 and 14: "Copyright 2016 The Inter Project Authors", the OFL), and Inter declares no Reserved Font Name, so `OFL.txt` stays true for both. A row of the SemiBold family is as tall as a row of the Regular one at the four sizes the test reads (12.5, 15, 18 and 22 points).
5. **The widths.** At 18 points the title is 262.1 points wide in SemiBold and 257.7 in Regular, the heading 134.6 and 133.4. At 1024 x 700, 1280 x 800 and 1920 x 1080 alike the top bar's texts stay on one row, ending at x 647.8 (643.5 in Regular) of a 1024-point window, and the heading sits on one line in the 320-point column (its right edge at x 150). The BL-099 and BL-098 width tests (the left column at 320 points at the three sizes with every fold, group and filter, the right column at 300) pass unchanged in the new face.
6. **The cost.** The release WASM file (`magcoupling-web_bg.wasm` after `wasm-bindgen`, no `wasm-opt`) is 6,820,704 bytes on `main` and after Task 1; Task 2 makes it 6,867,573 (+46,869: the cut's 30,524 and the code that installs it), Task 3 6,874,135 (+6,562: `remove_loading_line`'s calls into the page). Compressed, the whole part adds 22,302 bytes with gzip -9 (2,643,908 to 2,666,210) and 14,994 with brotli -q 11 (1,948,164 to 1,963,158); the face's own share is 20,689 and 15,815. The page grows from 1,034 to 1,709 bytes (532 to 784 gzipped).
7. **When the line goes.** In Chrome on a local server the line is on the page at `DOMContentLoaded` (about 14 ms after the navigation) and the app's first draw follows at about 90 ms. Removed by the page's script when `init()` resolved, the line left in the animation frame before the first draw, 26 ms earlier: a blank charcoal frame. Removed by the app at the end of its frame, the removal and the first draw fall in the same animation frame in each of three loads, and the line is gone before the first draw call. With the network held at 1 MB/s the line shows for about 7.4 s, then the first view; the page is charcoal throughout, with no frame of another colour.
8. **An open picker and a plot legend** (found by the plan's review, measured again here). egui draws an open picker's box in `widgets.open`, whose outline was the hairline colour: opened, Auto's picker drew its outline at 1.21:1 on the panel where every shut picker shows the grey (3.26:1); with the grey as the open outline it keeps it. egui_plot 0.33 draws each legend entry as a disc in the interact state's `bg_fill` and `bg_stroke` (radius 0.35 of the text's row) under the line's dot (0.25): on `main` that disc is the border colour, 1.29:1 on the legend's background, near invisible; now it is the grey, 3.48:1. Torque vs temperature, the longest legend, has seven entries; its Operating temperature line's muted dot is 1.67:1 on the disc, the violet, blue and amber dots 1.98:1, 2.40:1 and 2.58:1. The step from a box's resting outline to the pointer's muted one is 1.67:1 (4.48:1 on `main`, from the hairline colour).
9. **Line endings and rustfmt.** `cargo fmt` may rewrite a file it changes with LF line endings on this machine; git's `autocrlf` takes them back at the commit, and the apply tool reads each file's endings as it finds them. `.claude/workflows/gui-smoke.js` is LF (`.gitattributes`); the others are CRLF. git takes the `.ttf` file as binary.

No pictures are kept with this plan: Task 6's text review renders the page loading, the first view at the three window sizes, the Design menu, Auto's picker open, a slider focused by Tab and the Plots tab on Torque vs temperature with its legend.

## Global Constraints

- **No engine number moves** (the design's 2.3). No file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` changes.
- **The panel is not touched.** No file under `magcoupling-rs/src/gui/` changes: the panel sets no theme and no font (2.2, 6.3), and the app's theme and fonts reach it through egui's visuals and styles.
- **The linkage app and the hub are not touched** (O-1). No file under `linkage-sim-rs/src/` changes, and under `linkage-sim-rs/web/` only the calculator's page, `tools/magcoupler/index.html`.
- **One new on-screen string:** the page's loading line, "Loading the calculator" (`app::LOADING_LINE`, the review's own words). It passes the plain-words guard and the gearbox guard (Task 3's test). No other text is added, changed or removed.
- **Two colours outside the token table, no more:** `SELECTION` (register row A-19) and `CONTROL` (row A-52). **Crimson is never text, never on a value, never in the check list** (C-5), and no control's edge is the accent.
- **A font file is made, never edited by hand.** `Inter-SemiBold.subset.ttf` is the script's output from the recorded source, and its SHA-256 is checked; a different hash stops the build.
- **Downloaded files are untrusted data.** The zip goes into its own new, empty directory outside the tree; Python reads it with `-I`; the directory is removed after the cut is checked.
- **View state and the exports do not change** (5.1, 12.1).
- **The blocks.** Each Replace block's old text occurs exactly once in its file at the moment its step applies it. Apply a step's blocks in the order given, top to bottom. Keep each file's line endings. If an old text is not found, or is found twice, stop and report: do not adapt the block. Line numbers are not given because earlier steps move them. `C:\Users\Cole\source\repos\linkage_simulation\.superpowers\sdd\2026-11-22-magcoupler-overhaul-b2\apply_plan.py` (gitignored, in the main checkout; stage 1's tool, unchanged) applies one step's blocks this way: `python <it> <this plan> <task> <step>`, from the worktree's root. The replay used it. It carries no binary file: Task 2 Step 1's commands make the font.
- **TDD.** Red, then green. Every red step here fails at run time with the message its "Expected" names.
- **Before each commit**, from the repository root:
  ```bash
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
  ```
  `cargo fmt` changes no text of a task's files (the blocks are formatted); it may rewrite a file it reads with other line endings, which git's `autocrlf` undoes at the commit. After it, `git status --short` shows only the files the task names. If it shows another, stop and report.
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-27-magcoupler-overhaul-b2p2biiib/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Shell.** On this machine `bash` typed in PowerShell is WSL, which fails on the checkout's line endings. Shell commands and scripts run through Git Bash: `& 'C:\Program Files\Git\bin\bash.exe' <script>`, or Git Bash itself. Python scripts are written with the Write tool, not a here-document. Do not edit a source file with `sed -i` from Git Bash: it rewrites the file's line endings.
- **Never run `cargo fmt` inside `linkage-sim-rs`.** `magcoupling-rs` is fmt-clean; keep it so.
- **Docs ship with the code** (repository rule): Task 5 sweeps the README and `docs/ai/`, the register, the owner-question list and the design's header and 6.2.

## Model per task

CLAUDE.md section 5: `fable` is the controller and the decider. BL-107 is `risk: mechanical` (decision D-105): the part shows no new number and makes no physical claim, and its one new on-screen string is a loading line. So the tasks that transcribe the plan's exact blocks and run its exact commands run on `sonnet`, and every review and the release run on `fable`; while `fable` is out (back 2026-12-01) the decider's steps inherit the session model, and the report says which model ran them. A `sonnet` task that ends blocked, leaves the gate red or is rejected in review is redone on `fable` (escalate, never downgrade). A hash in Task 2 Step 1 that does not match is no task failure to retry: it stops the build and goes to the controller.

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | Every control shows its edge at 3:1 or more (`CONTROL`) | `sonnet` |
| 2 | The title and the left column's heading in Inter SemiBold (the cut made by the script, its hashes checked; the Heading style) | `sonnet` |
| 3 | The page shell: language, colours and the loading line the app removes | `sonnet` |
| 4 | gui-smoke's look check reads the line gone and judges the heavier title and the edges | `sonnet` |
| 5 | Docs sweep | `sonnet` |
| 6 | Release: the full gate, gui-smoke, the text review with its pictures, the independent whole-branch review, one fix wave, the merge and the push | `fable` controller (the session model while `fable` is out); fresh-context `fable` reviewers (likewise); `haiku` for git; the smoke workflow's own agents on `sonnet` |

Each task's diff is reviewed by a fresh-context reviewer before the next task starts (`sonnet` for these mechanical diffs, CLAUDE.md section 5); it checks the diff against the task's blocks and nothing else, and for Task 2 that the font file's SHA-256 is the one recorded. A task that ends blocked or leaves a test red stops the build and goes back to the controller.

## Review Focus

Thirteen inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named, or says where it is checked.

1. **A checkbox in a menu.** egui's menus draw a checkbox's box with no outline, so its fill alone must show it on the menu's surface. Task 1: `every_control_shows_its_edge_on_the_panel_and_in_the_design_menu` (the Design menu's box, after its fade) and `a_control_at_rest_shows_its_edge_at_three_to_one_on_every_surface_it_sits_on` (the fill on a menu's colour).
2. **Every kind of control, not one of each.** Task 1's app test reads every filled or outlined shape of the left column with both folds open: 27 sliders, the value boxes and pickers, the text boxes, the checkboxes and the order toggle.
3. **Under the pointer, pressed, focused and open.** A box and a handle must not go darker than at rest, and an open picker must not lose its edge. Task 1's theme test: the fill is kept, a box's outline is the muted colour, then the text colour, a handle's ring is the text colour in every state, and the open outline is the resting one; `an_open_picker_keeps_its_edge` opens Auto's picker and reads its box. A handle shows the pointer, a drag and the keyboard's focus by its ring's width and size alone, and the step from a box's resting outline to the pointer's is smaller than on `main` (1.67:1 against 4.48:1): decision D-98, recorded, not tested by look; Task 6's text review renders an open picker and a handle focused by Tab and rules on both.
4. **A selected choice.** It keeps its blue fill and gains the grey edge (1.06:1 to the fill). No test holds the look; Task 6's text review rules on it (decision D-97).
5. **A glyph in a heading.** The glyph guards ask the Regular cut (`cut_holds`). Task 2: `the_semibold_cut_has_the_regular_cut_s_characters_digits_and_rows` (the same characters up to the maths signs).
6. **The heading's row and the brand mark.** A taller row would move the column and the mark (`gui::look::mark_ui` reads the Heading style's height). Task 2: the same test (equal rows at four sizes).
7. **The widths at the three sizes.** Task 2: `the_title_and_the_size_heading_are_semibold_and_the_top_bar_keeps_one_row` (1024 x 700, 1280 x 800, 1920 x 1080), and the BL-099 and BL-098 tests, unchanged, now in the new face.
8. **No other text changes face.** Task 2: the app test (two SemiBold texts, no other; the tab of the heading's words stays Regular) and `inter_leads_the_proportional_family_and_the_built_in_fonts_follow` (every text style but Heading as egui's).
9. **A character outside the cut in a heading.** It must fall to a built-in font, not to an empty box. Task 2: the font test holds the SemiBold family's list (the cut, then egui's built-in fonts).
10. **The line's timing.** No blank frame between the line and the app, no frame of another colour. Not testable without a browser: Task 3's code (the removal in the app's frame), measured in the prototype ("What the prototype showed" item 7), and checked again in Task 6 Step 4.
11. **The wasm file fails to load.** The line stays on the page (decision D-103). Not tested; Task 6's text review weighs it.
12. **The smoke check stays one key.** Task 4: the guard in `app.rs` and the verdict test (`all 85 checks hold` before and after).
13. **A plot legend's toggles.** egui_plot draws each entry's dot on a disc in the resting fill and outline, so every legend dot gains a grey disc, and a muted dot is 1.67:1 on it, close to the look of a hidden line's empty disc. Task 1: `a_plot_legend_s_entries_show_their_dots_on_a_grey_disc` (Torque vs temperature's seven entries, each disc in the resting fill and outline at 3:1 on the legend); Task 6's text review renders that plot and rules on it (decision D-97, owner question 73).

## Decisions this plan takes

Each is a reading of the design or of the review where it was open, numbered after part 2b-iii-a's D-86 to D-94 (one stage, one list). The owner may overrule any of them; Task 6 lists them in its report. The ones that need a row in the register have one (Task 5).

| Id | Decision | Why |
|---|---|---|
| D-95 | Part 2b-iii-b is one plan: the controls' edges, the SemiBold face and the page shell. | One subject (section 6), each task short; see "Why part 2b-iii-b is one plan". |
| D-96 | One more colour outside the token table, `CONTROL` `#656c76`, for the edge of a control at rest: the resting outline and the resting fill. Register row A-52; owner question 73. | WCAG's 3:1 for a control's edge (1.4.11), the conservative side; no token does it without damage (the review's E-4). The darkest grey of the tokens' hue at 3:1 on a menu, so a button's ring is as quiet as 3:1 allows; the review's `#626973` is 2.9:1 there. |
| D-97 | Every control egui frames at rest takes the edge (value boxes, framed buttons, pickers, open or shut, text boxes, checkboxes, radio buttons, slider rails and handles), and a plot legend's entries show their dots on a disc of it (egui_plot draws the disc in the resting fill and outline); frameless ones stay frameless (the top bar's words, the tabs, menu items, an unselected name in a row of choices); a selected choice keeps its fill with the grey edge; the hairlines between the columns, which are no control, stay hairlines. Register row A-52; owner question 73. | The words of a frameless control name it, and the top bar's and the tabs' framelessness is the design's (4.2, the underlined tab). The edge comes from egui's own visuals, so no panel code changes. An open picker in the hairline colour would read as dimmed beside the shut ones. The legend's disc marks each entry as the toggle it is (WCAG 1.4.11); bare dots would need the panel's plot code to set the legend's style, which this part does not touch, and the owner may choose them (question 73). WCAG 1.4.11 asks nothing of a dividing line. |
| D-98 | A box and a handle keep the `CONTROL` fill under the pointer and pressed; a box's outline brightens (muted, then text), in the colours it had, from a lighter rest (1.67:1 from the grey to the muted colour, where it was 4.48:1 from the hairline colour). A slider's handle keeps its ring in the text colour in every state, so it shows the pointer, a drag and the keyboard's focus by the ring's width (1, 1.5 and 2 points) and its size alone, as since stage 1 (the review's E-4 named this; it stays). Register row A-52 (its answer names it); owner question 73 offers the blue handle. | With the hovered and pressed fills left at the border colour, a box would go darker under the pointer than at rest. A blue handle (`active.bg_fill` the selection fill) would turn a pressed or focused checkbox's box blue too, since egui gives both one fill: a new look no one has judged. The conservative call keeps stage 1's cue; the builder's call. |
| D-99 | The SemiBold face draws egui's Heading style: the title and the left column's heading. The right column's captions, the tiles' numbers and every other text stay Regular. Register row A-53; owner question 74. | The panel names no font (6.3), and the Heading style is the one the app can point at a face without the panel naming one; the review's S-1 named the title and the headings, and said the values wait. The smallest change. |
| D-100 | `fonts::install` sets the whole font definitions (`set_fonts`), each cut first in its family with egui's built-in fonts behind, instead of `add_font`. | `add_font` would make the SemiBold family hold the cut alone: a character outside it would draw an empty box, against 6.2 (the built-in fonts stay the fallback). The Regular family's list is the same as before (its test holds it). |
| D-101 | The page shell: `lang="en"`, `theme-color` the panel colour and `color-scheme: dark` in the head, and one line, "Loading the calculator", in the muted colour in a 14 px system font across the middle of the page, after the canvas. Register row A-54; owner question 75. | The review's S-4, in its own words; the fewest new words that say the page is working, in the app's colours. The system font because Inter is inside the wasm file the line waits for. The meta tags in the head act before any style, so nothing flashes white. |
| D-102 | The app removes the line at the end of each frame on wasm32 (`remove_loading_line` in `eframe::App::update`, a lookup that finds nothing after the first frame); the page's script only starts the app. Register row A-54. | Measured in Chrome: removed by the script when `init()` resolved, the line left one animation frame before the first draw (a blank frame); removed in the app's frame, both fall in the same frame. A lookup by id per frame costs microseconds; a flag would be a field for nothing. |
| D-103 | A wasm file that fails to load leaves the line on the page; no error text. Register row A-54. | An error line is a new string no one asked for (8.3), and the page was silent on failure before. The cost: "Loading the calculator" stays where the page used to be blank. |
| D-104 | gui-smoke's `look` check also reads that the line is gone (by its id, typed once as `LOADING_ID`) and judges the heavier title and heading and the grey edge of Auto's box and of the value boxes; no new check key. | The three items in a browser on the page the check already reads; the verdict test's rules and count stay. |
| D-105 | BL-107 is `risk: mechanical`; its build tasks run on `sonnet`, its reviews and release on `fable` (the session model while `fable` is out). | No new number or physical claim; the one new on-screen string is a loading line. |

## Nothing lost

Every text, look and action this part changes, and where it is after. Anything not listed (every text but the page's line, every position, every view, the panel's code, the hub, the linkage app) does not change. The "Test" column names the test that holds the new state.

| Today | After this part | Test |
|---|---|---|
| A control's resting edge and fill in the border colour (1.1:1 to 1.2:1) | The same controls with the grey `CONTROL` edge (3.0:1 to 3.5:1); a selected choice's blue fill kept | `a_control_at_rest_shows_its_edge_at_three_to_one_on_every_surface_it_sits_on`; `every_control_shows_its_edge_on_the_panel_and_in_the_design_menu`; `the_visuals_are_the_token_table` |
| Under the pointer a box darkened to the border colour with a muted ring | The box keeps its grey fill; the ring is muted, pressed the text colour, the colours it had, from a lighter rest (1.67:1 from rest to the pointer's ring, where it was 4.48:1); a slider's handle keeps its text-colour ring, which thickens as before | the theme test above |
| An open picker's box outlined in the hairline colour, as at rest | Outlined in the grey, as at rest | `an_open_picker_keeps_its_edge`; the theme test above |
| A plot legend's dots on a disc in the border colour, near invisible (1.29:1 on the legend) | The same dots on a grey disc (3.48:1); a muted dot 1.67:1 on it; a hidden line's entry an empty grey disc | `a_plot_legend_s_entries_show_their_dots_on_a_grey_disc`; Task 6's text review rules on the look |
| The hairlines between the columns (1.21:1) | The same | none: no control (D-97) |
| The title and "Size and torque" in Inter Regular, 18 points | Inter SemiBold, 18 points, rows as tall, 1 to 2 % wider; the top bar one row and the column 320 points at the three sizes | `the_title_and_the_size_heading_are_semibold_and_the_top_bar_keeps_one_row`; `the_semibold_cut_has_the_regular_cut_s_characters_digits_and_rows`; the BL-099 and BL-098 width tests |
| Every other text in Inter Regular, egui's built-in fonts behind | The same | the app test above (two SemiBold texts); `inter_leads_the_proportional_family_and_the_built_in_fonts_follow`; `the_app_draws_its_text_in_inter` |
| The page charcoal and empty while the wasm file loads; no `lang`, `theme-color` or `color-scheme` | The same charcoal with the muted line, gone in the paint of the app's first frame; the three in the head | `the_page_wears_the_app_s_colours_and_a_loading_line_until_the_app_draws`; the smoke run (Task 4) |
| The page's script `init();`; the canvas id; the background `#181b1f` | The same | the page test above; `the_canvas_id_matches_the_web_page`; `the_web_page_background_is_the_panel_colour`; `linkage-sim-rs/tests/web_layout.rs` |
| gui-smoke's `look`: background, brand mark, underlined tab, one face | The same, and the line gone, the heavier title and heading, the grey edges | `the_smoke_test_share_link_opens_its_design`; the verdict test |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/app/theme.rs` | `luminance` and `contrast` out of the test module, one test (1, red); `CONTROL`, four widget states, the comments, the token table's test (1, green); `visuals`' comment (2) | 1, 2 |
| `magcoupling-rs/src/app.rs` | Three tests (1); one test (2); `LOADING_ID`, `LOADING_LINE`, one test (3, red); `remove_loading_line` (3, green); the smoke prompt's guard (4) | 1, 2, 3, 4 |
| `magcoupling-rs/assets/fonts/subset_inter.py` | Its docstring records the SemiBold source, the two cuts and their hashes | 2 |
| `magcoupling-rs/assets/fonts/Inter-SemiBold.subset.ttf` | New, made by the script (binary) | 2 |
| `magcoupling-rs/src/app/fonts.rs` | `INTER_SEMIBOLD`, `SEMIBOLD_NAME`, `alone`, one test (2, red); `semibold`, `install`, the module's comment, the family test (2, green) | 2 |
| `linkage-sim-rs/web/tools/magcoupler/index.html` | `lang`, two meta tags, the line and its style, the comment | 3 |
| `.claude/workflows/gui-smoke.js` | `LOADING_ID`, the look step, the description and a comment | 4 |
| `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`, the register, the owner questions, the design's header and 6.2 | The docs sweep | 5 |
| `docs/ai/backlog.yaml` | BL-107 in progress (5); BL-107 fixed (6) | 5, 6 |

**Existing tests this plan edits, and how.** No assertion about behaviour that stays is weakened.
- **`the_visuals_are_the_token_table`** (`theme.rs`, Task 1 green). The inactive, hovered and active states' fills, and the inactive and open outlines, become `CONTROL`; every other assertion stays, and its last check (every field outside the table is egui's) still runs.
- **`luminance` and `contrast`** (`theme.rs`, Task 1 red) move out of the test module unchanged, `#[cfg(test)] pub(crate)`, so the app test measures with the theme's own formula.
- **`inter_alone`** (`fonts.rs`, Task 2 red) becomes `alone(INTER_NAME, INTER)`: the same context, built by a helper the SemiBold test shares.
- **`inter_leads_the_proportional_family_and_the_built_in_fonts_follow`** (`fonts.rs`, Task 2 green) keeps its two assertions and gains the SemiBold family's list and the text styles.
- **`the_smoke_test_share_link_opens_its_design`** (`app.rs`, Task 4 red) gains one block; nothing in it changes.
- **Unchanged and still passing:** `the_web_page_background_is_the_panel_colour`, `the_canvas_id_matches_the_web_page`, `the_app_draws_its_text_in_inter`, `the_ten_digits_have_one_advance_width`, `a_row_is_as_tall_as_the_built_in_font_s`, the BL-099 and BL-098 width tests and `no_text_is_cut_in_any_tab_in_the_app_s_own_font`, which now draw the headings in SemiBold; and `linkage-sim-rs/tests/web_layout.rs` (the page's import line), which Task 6's full gate runs.

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` after this plan has merged, create the branch `magcoupling/overhaul-b2p2b-iii-b` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  python -I -c "import fontTools, sys; print(fontTools.version, sys.version.split()[0])"
  ```

  Expected: `test result: ok. 1143 passed; 0 failed; 6 ignored`, `all 85 checks hold` and `4.51.0 3.12.3` (another 3.12 patch release is fine; another fontTools version may cut other bytes: install 4.51.0, or stop and report). Another count means `main` is not the base the blocks were checked on: stop and report.

- [ ] **Step 3: Check the base.** `git log --oneline f84bc4f..main -- magcoupling-rs linkage-sim-rs/web/tools/magcoupler .claude/workflows/gui-smoke.js linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs docs/ai docs/analyses/2026-11-10-conservative-assumptions.md docs/analyses/2026-11-18-owner-questions.md docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` lists this plan's commit (it touches `docs/ai/backlog.yaml`) and nothing else. If it lists more, stop: the controller regenerates the blocks on that `main` with the planner's tools (gitignored, in the main checkout's `.superpowers/sdd/2026-11-27-magcoupler-overhaul-b2p2biiib/`, the scripts in its `tools/` folder: `gen_blocks.py` (part 2b-iii-a's, with `.html` fenced as html), `frags.py`, `replay.py` and `finalize.py`, which read the prototype as a clone of `seq.bundle` named `proto` in the folder above them (branch `proto`); `seq.bundle`, the prototype's commits on `main` at `f84bc4f`: `s1r` and `s1` Task 1's red and green, `s2a` Task 2's font file and the script's record, `s2r` and `s2` its red and green, `s3r`, `s3`, `s4r` and `s4` Tasks 3 and 4's, `s4b` BL-107 filed, `s5` the docs; and the edit scripts that made them, `ed_lib.py`, `fmt.py`, `e1_green.py`, `e2_script.py`, `e2_red.py`, `e2_red_fix.py`, `e2_green.py`, `e2_green_b.py`, `e2_green_c.py`, `e3_red.py`, `e3_green.py`, `e4_red.py`, `e4_green.py`, `file_bl107.py` and `e5.py`; Task 1's red state, and the review's changes to Task 1 and the docs, were written by hand, so the bundle's commits are the states) before Task 1.

## Tasks

### Task 1: Every control shows its edge at 3:1 or more

**Model:** `sonnet` (BL-107 is `risk: mechanical`; the task transcribes exact blocks).

The theme gains `CONTROL`, the edge of a control at rest, and gives it to the resting outline and fill, to the fill under the pointer and pressed and to the open outline (decisions D-96 to D-98). Four tests hold it: one on the theme's numbers, three on what the app draws (the left column and the Design menu, an open picker, a plot's legend).

**Files:**
- Modify: `magcoupling-rs/src/app/theme.rs`, `magcoupling-rs/src/app.rs`

**Interfaces:**
- Produces: `app::theme::CONTROL: Color32`; `#[cfg(test)] pub(crate) fn app::theme::luminance(Color32) -> f64` and `contrast(Color32, Color32) -> f64` (moved out of the test module, unchanged).
- Changes: `visuals()`: `widgets.inactive` is (`CONTROL`, `PANEL2`, a 1-point `CONTROL` outline); `widgets.hovered` and `widgets.active` take `CONTROL` as their fill; `widgets.open`'s outline is a 1-point `CONTROL`.

- [ ] **Step 1 (red): the edges' tests.** The blocks add `every_control_shows_its_edge_on_the_panel_and_in_the_design_menu` to `app.rs` (with both folds of the left column open in a window tall enough for every row, every filled or outlined rect and circle the column draws is 3:1 or more against the panel by its fill or its outline, more than 90 of them; then, the Design menu opened and faded in, its checkbox's box is 3:1 or more against the menu), `an_open_picker_keeps_its_edge` (Auto's picker, clicked open on the laptop screen with its list showing, draws its box with an outline at 3:1 or more against the panel) and `a_plot_legend_s_entries_show_their_dots_on_a_grey_disc` (on the Plots tab's Torque vs temperature, each of the legend's seven entries draws its dot on a disc in the resting fill and outline, 3:1 or more against the legend's background) to `app.rs`, move `luminance` and `contrast` out of `theme.rs`'s test module, and add `a_control_at_rest_shows_its_edge_at_three_to_one_on_every_surface_it_sits_on` (the resting outline and fill at 3:1 on the panel, a menu and a field; the tick at 3:1 on the fill; the hovered and pressed fills equal to the resting fill, their outlines the muted and the text colour, their rings the resting ring's colour; the open outline the resting one).

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        assert_ne!(theme::BAD, ACCENT);
    }
```

with:

```rust
        assert_ne!(theme::BAD, ACCENT);
    }

    #[test]
    fn every_control_shows_its_edge_on_the_panel_and_in_the_design_menu() {
        // The stage 1 review (E-4, R-6), in the app's own theme: a checkbox's box, a slider's
        // rail and a value box at rest were 1.1:1 to 1.2:1 on the panel. With both folds of the
        // left column open, every rect and circle the column draws (the sliders' rails and
        // handles, the value boxes, the pickers, the text boxes, the checkboxes and the order
        // toggle: no other shape is filled or outlined there at rest) shows an edge, by its
        // fill or its outline, at 3:1 or more against the panel. The Design menu draws a
        // checkbox's box with no outline: its fill is 3:1 or more against the menu.
        use crate::gui::test_support::{flat_shapes, text_center, text_rects};
        use crate::gui::{ASSUMPTIONS_FOLD, DESIGN_MENU, DETAILED_INPUTS, SHOW_DETAILS};
        use theme::contrast;
        // The stronger of a shape's fill and outline against `behind`.
        let edge = |fill: egui::Color32, stroke: egui::Stroke, behind: egui::Color32| {
            let outline = if stroke.width > 0.0 && stroke.color.a() > 0 {
                contrast(stroke.color, behind)
            } else {
                1.0
            };
            let fill = if fill.a() > 0 {
                contrast(fill, behind)
            } else {
                1.0
            };
            fill.max(outline)
        };
        // A window tall enough to draw every row of the column.
        let tall = egui::vec2(1280.0, 20_000.0);
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_on(&ctx, &mut app, tall, Vec::new());
        for fold in [DETAILED_INPUTS, ASSUMPTIONS_FOLD] {
            let output = frame_on(&ctx, &mut app, tall, Vec::new());
            click_at(&ctx, &mut app, tall, text_center(&output, fold), 6);
        }
        let output = frame_on(&ctx, &mut app, tall, Vec::new());
        let column = side_rect(&ctx, "magcoupling_inputs");
        let mut faint = Vec::new();
        let mut shapes = 0;
        for shape in flat_shapes(&output) {
            let (what, ratio) = match shape {
                egui::Shape::Rect(rect)
                    if column.contains_rect(rect.rect)
                        && rect.rect != column
                        && (rect.fill.a() > 0 || rect.stroke.color.a() > 0) =>
                {
                    (
                        format!("{:?}", rect.rect),
                        edge(rect.fill, rect.stroke, theme::PANEL),
                    )
                }
                egui::Shape::Circle(circle) if column.contains(circle.center) => (
                    format!("a circle at {:?}", circle.center),
                    edge(circle.fill, circle.stroke, theme::PANEL),
                ),
                _ => continue,
            };
            shapes += 1;
            if ratio < 3.0 {
                faint.push(format!("{what}: {ratio:.2}"));
            }
        }
        // Six size controls, twenty-one more sliders, their value boxes and the rest.
        assert!(shapes > 90, "{shapes} shapes");
        assert!(
            faint.is_empty(),
            "{} of {shapes} faint: {faint:#?}",
            faint.len()
        );
        // The Design menu, opened on the laptop screen, its fading in done.
        frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        let output = frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        click_at(&ctx, &mut app, LAPTOP, text_center(&output, DESIGN_MENU), 6);
        let output = frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        let label = text_rects(&output, SHOW_DETAILS);
        let label = label.first().expect("the menu shows the checkbox");
        let boxes: Vec<f64> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(rect)
                    if rect.rect.right() <= label.left()
                        && (rect.rect.center().y - label.center().y).abs() < 2.0
                        && (rect.rect.width() - rect.rect.height()).abs() < 0.5 =>
                {
                    Some(edge(rect.fill, rect.stroke, theme::PANEL2))
                }
                _ => None,
            })
            .collect();
        assert_eq!(boxes.len(), 1, "the checkbox's box: {boxes:?}");
        assert!(boxes[0] >= 3.0, "the menu's checkbox: {:.2}", boxes[0]);
    }

    #[test]
    fn an_open_picker_keeps_its_edge() {
        // egui draws an open picker's box in the open state's colours, not the resting ones.
        // Opened, Auto's picker keeps an outline at 3:1 or more against the panel, as at rest,
        // so it does not read as dimmed or disabled while its list shows.
        use crate::engine::sizing::FreeVariable;
        use crate::gui::sizing::auto_choice;
        use crate::gui::test_support::{flat_shapes, text_rects};
        use theme::contrast;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        let output = frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        let shown = text_rects(&output, auto_choice(FreeVariable::AxialLength));
        assert_eq!(shown.len(), 1, "Auto's picker shows its size: {shown:?}");
        let output = click_at(&ctx, &mut app, LAPTOP, shown[0].center(), 6);
        // Open: its list names the other sizes.
        let listed = text_rects(&output, auto_choice(FreeVariable::RingRadius));
        assert_eq!(listed.len(), 1, "the picker's list: {listed:?}");
        let outlines: Vec<f64> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(rect)
                    if rect.rect.contains_rect(shown[0])
                        && rect.rect.height() < 3.0 * shown[0].height() =>
                {
                    Some(if rect.stroke.width > 0.0 {
                        contrast(rect.stroke.color, theme::PANEL)
                    } else {
                        1.0
                    })
                }
                _ => None,
            })
            .collect();
        assert_eq!(outlines.len(), 1, "the picker's box: {outlines:?}");
        assert!(
            outlines[0] >= 3.0,
            "the open picker's outline: {:.2}",
            outlines[0]
        );
    }

    #[test]
    fn a_plot_legend_s_entries_show_their_dots_on_a_grey_disc() {
        // egui_plot draws each legend entry, a toggle that hides or shows its line, as a disc
        // in the resting fill and outline with the line's dot on it. That disc is the edge of
        // the toggle: 3:1 or more against the legend's background (the field colour), on every
        // entry of Torque vs temperature's legend, the longest.
        use crate::gui::CentreView;
        use crate::gui::plots::PlotKind;
        use crate::gui::test_support::flat_shapes;
        use theme::contrast;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        let view = CentreView::Plot(PlotKind::TorqueTemperature);
        let output = show_view(&ctx, &mut app, LAPTOP, view, 2);
        let (left, right) = (
            side_rect(&ctx, "magcoupling_inputs").right(),
            side_rect(&ctx, "magcoupling_dashboard").left(),
        );
        let circles: Vec<&egui::epaint::CircleShape> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Circle(circle)
                    if left < circle.center.x && circle.center.x < right =>
                {
                    Some(circle)
                }
                _ => None,
            })
            .collect();
        // An entry: a filled and outlined disc (a plot's own points are one or the other), and
        // on it, at its centre, its line's smaller dot.
        let discs: Vec<&egui::epaint::CircleShape> = circles
            .iter()
            .copied()
            .filter(|disc| {
                disc.fill.a() > 0
                    && disc.stroke.width > 0.0
                    && circles
                        .iter()
                        .any(|dot| dot.center == disc.center && dot.radius < disc.radius)
            })
            .collect();
        assert_eq!(discs.len(), 7, "the legend's entries: {discs:?}");
        let rest = theme::visuals().widgets.inactive;
        for disc in discs {
            let ratio = contrast(disc.fill, theme::BG);
            assert!(ratio >= 3.0, "a legend entry's disc: {ratio:.2}");
            assert_eq!((disc.fill, disc.stroke), (rest.bg_fill, rest.bg_stroke));
        }
    }
```

**`magcoupling-rs/src/app/theme.rs`**

Replace:

```rust

#[cfg(test)]
```

with:

```rust

/// The WCAG 2 relative luminance of an opaque colour.
#[cfg(test)]
pub(crate) fn luminance(colour: Color32) -> f64 {
    let linear = |channel: u8| {
        let s = f64::from(channel) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * linear(colour.r()) + 0.7152 * linear(colour.g()) + 0.0722 * linear(colour.b())
}

/// The WCAG 2 contrast ratio of two opaque colours.
#[cfg(test)]
pub(crate) fn contrast(a: Color32, b: Color32) -> f64 {
    let (a, b) = (luminance(a), luminance(b));
    (a.max(b) + 0.05) / (a.min(b) + 0.05)
}

#[cfg(test)]
```

Replace:

```rust
    use super::*;

    /// The WCAG 2 relative luminance of an opaque colour.
    fn luminance(colour: Color32) -> f64 {
        let linear = |channel: u8| {
            let s = f64::from(channel) / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * linear(colour.r()) + 0.7152 * linear(colour.g()) + 0.0722 * linear(colour.b())
    }

    /// The WCAG 2 contrast ratio of two opaque colours.
    fn contrast(a: Color32, b: Color32) -> f64 {
        let (a, b) = (luminance(a), luminance(b));
        (a.max(b) + 0.05) / (a.min(b) + 0.05)
    }

```

with:

```rust
    use super::*;

```

Replace:

```rust
    #[test]
    fn the_theme_is_dark_whatever_the_system_prefers() {
```

with:

```rust
    #[test]
    fn a_control_at_rest_shows_its_edge_at_three_to_one_on_every_surface_it_sits_on() {
        // The stage 1 review (E-4, R-6), by WCAG 2's rule for a control's edge (1.4.11): at
        // least 3:1 against what is around it. At rest a value box, a framed button, a picker
        // and a text box show their resting outline; a checkbox's box, a radio button and a
        // slider's rail and handle show their resting fill, and in a menu, which drops the
        // outline, a checkbox's box shows by its fill alone. Both were the border colour,
        // 1.2:1 on the panel. A control sits on the panel, in a menu and on a field.
        let v = visuals();
        let rest = v.widgets.inactive;
        assert!(rest.bg_stroke.width >= 1.0);
        for (what, edge) in [("outline", rest.bg_stroke.color), ("fill", rest.bg_fill)] {
            for (surface, colour) in [("the panel", PANEL), ("a menu", PANEL2), ("a field", BG)] {
                let ratio = contrast(edge, colour);
                assert!(ratio >= 3.0, "the resting {what} on {surface}: {ratio:.2}");
            }
        }
        // A ticked box's tick and a slider handle's ring are the text colour, on that fill.
        assert!(contrast(rest.fg_stroke.color, rest.bg_fill) >= 3.0);
        // Under the pointer and pressed, a box and a handle keep their fill and a box's outline
        // brightens: the muted colour, then the text colour. A handle's ring is the text colour
        // in every state (egui only thickens it and grows the handle), so its state shows by
        // the ring's width and the handle's size alone (decision D-98).
        let hovered = v.widgets.hovered;
        let pressed = v.widgets.active;
        assert_eq!(
            (hovered.bg_fill, pressed.bg_fill),
            (rest.bg_fill, rest.bg_fill)
        );
        assert_eq!(
            (hovered.bg_stroke.color, pressed.bg_stroke.color),
            (MUTED, TEXT)
        );
        assert!(luminance(MUTED) > luminance(rest.bg_stroke.color));
        assert_eq!(
            (hovered.fg_stroke.color, pressed.fg_stroke.color),
            (rest.fg_stroke.color, rest.fg_stroke.color)
        );
        // An open picker keeps the resting outline: egui draws its box in the open state's
        // colours, and opening it must not make it look dimmed.
        assert_eq!(v.widgets.open.bg_stroke, rest.bg_stroke);
    }

    #[test]
    fn the_theme_is_dark_whatever_the_system_prefers() {
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 12
  ```

  Expected: 4 failed: `a_plot_legend_s_entries_show_their_dots_on_a_grey_disc`, `an_open_picker_keeps_its_edge`, `every_control_shows_its_edge_on_the_panel_and_in_the_design_menu`, `a_control_at_rest_shows_its_edge_at_three_to_one_on_every_surface_it_sits_on`. `a_control_at_rest_shows_its_edge_at_three_to_one_on_every_surface_it_sits_on` panics with `the resting outline on the panel: 1.21`; `every_control_shows_its_edge_on_the_panel_and_in_the_design_menu` panics with `65 of 93 faint` and lists them, each at 1.21 (the first `[[16.0 84.0] - [116.0 92.0]]: 1.21`, the Required torque slider's rail); `an_open_picker_keeps_its_edge` panics with `the open picker's outline: 1.21`; `a_plot_legend_s_entries_show_their_dots_on_a_grey_disc` panics with `a legend entry's disc: 1.29`.

- [ ] **Step 3 (green): the colour.** The blocks add `CONTROL` beside `SELECTION`, give it to the four widget states (the open state's outline, with a comment on why), update the comments that count the colours outside the table, and set the token table's test to the new states.

**`magcoupling-rs/src/app/theme.rs`**

Replace:

```rust
// The token table (spec 6.1). Every colour [`visuals`] sets is one of these, but the fill
// behind what is selected ([`SELECTION`]).

```

with:

```rust
// The token table (spec 6.1). Every colour [`visuals`] sets is one of these, but the fill
// behind what is selected ([`SELECTION`]) and the edge of a control at rest ([`CONTROL`]).

```

Replace:

```rust

/// The fill behind selected text and of a selected choice: the one colour outside the spec's
/// table (6.1), which names none that a reader can see behind text. A quiet blue, the hue of
/// [`LINK`]: 3:1 or more against a field and the panel and 2.8:1 on a tooltip, with [`TEXT`]
```

with:

```rust

/// The fill behind selected text and of a selected choice: a colour outside the spec's table
/// (6.1), which names none that a reader can see behind text. A quiet blue, the hue of
/// [`LINK`]: 3:1 or more against a field and the panel and 2.8:1 on a tooltip, with [`TEXT`]
```

Replace:

```rust

/// The corner radius of buttons, fields, menus and tooltips [points].
```

with:

```rust

/// The edge of a control at rest (the stage 1 review, E-4 and R-6): the outline of a value
/// box, a framed button, a picker (open too) and a text box, and the fill of a checkbox's box,
/// a radio button, a slider's rail and handle and the disc behind a plot legend's dot (egui_plot
/// draws it in the resting fill and outline). The other colour outside the spec's table (6.1),
/// whose border colour is 1.2:1 on the panel: a grey of the tokens' hue, 3:1 or more on the
/// panel, a menu and a field, under [`MUTED`], with the tick of a ticked box ([`TEXT`]) at
/// 4.3:1 on it (the theme's tests pin each).
pub const CONTROL: Color32 = Color32::from_rgb(0x65, 0x6c, 0x76);

/// The corner radius of buttons, fields, menus and tooltips [points].
```

Replace:

```rust

/// One widget state: its fill (a checkbox's box, a slider's rail), a button's fill and its
/// outline. Its text is in the text colour and its corners are [`RADIUS`].
fn state(state: &mut WidgetVisuals, fill: Color32, button: Color32, outline: Stroke) {
```

with:

```rust

/// One widget state: its fill (a checkbox's box, a slider's rail and handle), a button's fill
/// and its outline. Its text is in the text colour and its corners are [`RADIUS`].
fn state(state: &mut WidgetVisuals, fill: Color32, button: Color32, outline: Stroke) {
```

Replace:

```rust
/// (`gui::look::quiet_text`), so the value is the bright one of the pair. Borders are hairlines
/// and nothing casts a shadow (decision C-6).
pub fn visuals() -> egui::Visuals {
```

with:

```rust
/// (`gui::look::quiet_text`), so the value is the bright one of the pair. Borders are hairlines
/// and nothing casts a shadow (decision C-6). A control at rest shows its edge in [`CONTROL`];
/// under the pointer and pressed its fill stays and its outline brightens.
pub fn visuals() -> egui::Visuals {
```

Replace:

```rust
    state(&mut v.widgets.noninteractive, PANEL2, PANEL2, hairline);
    state(&mut v.widgets.inactive, BORDER, PANEL2, hairline);
    state(
```

with:

```rust
    state(&mut v.widgets.noninteractive, PANEL2, PANEL2, hairline);
    state(
        &mut v.widgets.inactive,
        CONTROL,
        PANEL2,
        Stroke::new(1.0, CONTROL),
    );
    state(
```

Replace:

```rust
        &mut v.widgets.hovered,
        BORDER,
        BORDER,
```

with:

```rust
        &mut v.widgets.hovered,
        CONTROL,
        BORDER,
```

Replace:

```rust
        &mut v.widgets.active,
        BORDER,
        BORDER,
```

with:

```rust
        &mut v.widgets.active,
        CONTROL,
        BORDER,
```

Replace:

```rust
    );
    state(&mut v.widgets.open, PANEL2, PANEL2, hairline);
    v
```

with:

```rust
    );
    // An open picker's box (egui draws it in this state) keeps the resting outline, so opening
    // it does not make it look dimmed; menus drop this outline (egui's `menu_style`).
    state(
        &mut v.widgets.open,
        PANEL2,
        PANEL2,
        Stroke::new(1.0, CONTROL),
    );
    v
```

Replace:

```rust
            ("noninteractive", w.noninteractive, PANEL2, PANEL2, hairline),
            ("inactive", w.inactive, BORDER, PANEL2, hairline),
            (
```

with:

```rust
            ("noninteractive", w.noninteractive, PANEL2, PANEL2, hairline),
            (
                "inactive",
                w.inactive,
                CONTROL,
                PANEL2,
                Stroke::new(1.0, CONTROL),
            ),
            (
```

Replace:

```rust
                w.hovered,
                BORDER,
                BORDER,
```

with:

```rust
                w.hovered,
                CONTROL,
                BORDER,
```

Replace:

```rust
            ),
            ("active", w.active, BORDER, BORDER, Stroke::new(1.0, TEXT)),
            ("open", w.open, PANEL2, PANEL2, hairline),
        ] {
```

with:

```rust
            ),
            ("active", w.active, CONTROL, BORDER, Stroke::new(1.0, TEXT)),
            ("open", w.open, PANEL2, PANEL2, Stroke::new(1.0, CONTROL)),
        ] {
```

Replace:

```rust
        assert_eq!(v.widgets.open.bg_stroke, rest.bg_stroke);
    }
```

with:

```rust
        assert_eq!(v.widgets.open.bg_stroke, rest.bg_stroke);
        // The figures [`CONTROL`]'s comment quotes, to one decimal; the border colour it
        // replaces at rest.
        let one_decimal = |a, b| (contrast(a, b) * 10.0).round() / 10.0;
        assert_eq!(
            (one_decimal(CONTROL, PANEL), one_decimal(CONTROL, PANEL2)),
            (3.3, 3.0)
        );
        assert_eq!(
            (one_decimal(TEXT, CONTROL), one_decimal(BORDER, PANEL)),
            (4.3, 1.2)
        );
    }
```

- [ ] **Step 4: Run the checks** of "Before each commit". Expected: `test result: ok. 1147 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text.

- [ ] **Step 5: Commit** the two files: `fix(magcoupler): every control shows its edge at 3:1 or more on the panel, a menu and a field (BL-107 T1)`.

### Task 2: The title and the left column's heading in Inter SemiBold

**Model:** `sonnet`.

A second cut of Inter, SemiBold, made by the committed script from the recorded source, draws egui's Heading style: the title and the Size column's heading (decisions D-99 and D-100). Step 1 makes the binary file and checks it; the plan's blocks cannot carry it.

**Files:**
- Create: `magcoupling-rs/assets/fonts/Inter-SemiBold.subset.ttf` (Step 1, by the script)
- Modify: `magcoupling-rs/assets/fonts/subset_inter.py` (its docstring), `magcoupling-rs/src/app/fonts.rs`, `magcoupling-rs/src/app.rs`, `magcoupling-rs/src/app/theme.rs` (one comment)

**Interfaces:**
- Produces: `app::fonts::INTER_SEMIBOLD: &[u8]`, `SEMIBOLD_NAME: &str`, `semibold() -> egui::FontFamily`; `#[cfg(test)] fn alone(name, bytes) -> egui::Context`.
- Changes: `fonts::install` sets the font definitions (`ctx.set_fonts`) and every style's `TextStyle::Heading` to `semibold()` at its size.

- [ ] **Step 1: The SemiBold cut.** First the blocks: the script's docstring records the SemiBold source and both cuts with their hashes (its code does not change). Then, in Git Bash from the worktree's root, the commands below: they download Inter 4.1 into a new, empty directory outside the tree (a download is untrusted data), check the zip's SHA-256, take its `extras/ttf/Inter-SemiBold.ttf` out with Python in isolated mode, check that file's SHA-256, cut it with the committed script (isolated mode too: the script's folder and the current folder are not on Python's path), check the cut's SHA-256 and size, and remove the directory. A hash that does not match stops the build: report it, and change nothing to make it match.

**`magcoupling-rs/assets/fonts/subset_inter.py`**

Replace:

```python
"""Builds the calculator's font: Inter Regular, cut to the characters the app draws, with
tabular digits as its default digits and the vertical metrics of egui's built-in font.

```

with:

```python
"""Builds the calculator's fonts: Inter Regular and Inter SemiBold, each cut to the characters
the app draws, with tabular digits as its default digits and the vertical metrics of egui's
built-in font.

```

Replace:

```python
    python subset_inter.py Inter-Regular.ttf Inter-Regular.subset.ttf

```

with:

```python
    python subset_inter.py Inter-Regular.ttf Inter-Regular.subset.ttf
    python subset_inter.py Inter-SemiBold.ttf Inter-SemiBold.subset.ttf

The SemiBold cut draws the title and the headings (UX overhaul spec 6.2; the stage 1 review's
S-1). Both cuts hold the same characters, digits and rows: the same list, the same metrics.

```

Replace:

```python
  its file extras/ttf/Inter-Regular.ttf
  (SHA-256 40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82).
Licence: SIL Open Font License 1.1 (OFL.txt beside this file, the zip's LICENSE.txt). Inter
  declares no Reserved Font Name, so the cut file keeps the name.
Tool: fontTools 4.51.0 on Python 3.12. The output does not depend on the date (the head table's
```

with:

```python
  its file extras/ttf/Inter-Regular.ttf
  (SHA-256 40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82), cut to
  Inter-Regular.subset.ttf (30,684 bytes, SHA-256
  8f8ed7fc1abbc5080700ec7044eeea4afe76ec1ca6d6296eb49e823fecac3421),
  and its file extras/ttf/Inter-SemiBold.ttf
  (SHA-256 78a843fade9d4612a5567302fb595b56976eb5fcebf4fea5a5912d638bafcde3), cut to
  Inter-SemiBold.subset.ttf (30,524 bytes, SHA-256
  bfce2d669a45afe425b001c6ab20048d3dceea50f4cf38d39e11d29048d1d4b8).
Licence: SIL Open Font License 1.1 (OFL.txt beside this file, the zip's LICENSE.txt), for
  every weight of the family; both files keep its copyright and licence lines. Inter declares
  no Reserved Font Name, so each cut file keeps its name.
Tool: fontTools 4.51.0 on Python 3.12. The output does not depend on the date (the head table's
```

  ```bash
  DL="$(mktemp -d)"
  ls -A "$DL" | wc -l
  curl -L --fail --silent --show-error -o "$DL/Inter-4.1.zip" https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip
  echo "9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e  $DL/Inter-4.1.zip" | sha256sum -c -
  python -I -c "import sys, zipfile; open(sys.argv[2], 'wb').write(zipfile.ZipFile(sys.argv[1]).read('extras/ttf/Inter-SemiBold.ttf'))" "$DL/Inter-4.1.zip" "$DL/Inter-SemiBold.ttf"
  echo "78a843fade9d4612a5567302fb595b56976eb5fcebf4fea5a5912d638bafcde3  $DL/Inter-SemiBold.ttf" | sha256sum -c -
  python -I magcoupling-rs/assets/fonts/subset_inter.py "$DL/Inter-SemiBold.ttf" magcoupling-rs/assets/fonts/Inter-SemiBold.subset.ttf
  echo "bfce2d669a45afe425b001c6ab20048d3dceea50f4cf38d39e11d29048d1d4b8  magcoupling-rs/assets/fonts/Inter-SemiBold.subset.ttf" | sha256sum -c -
  wc -c < magcoupling-rs/assets/fonts/Inter-SemiBold.subset.ttf
  rm -r "$DL"
  ```

  Expected: `0` (the directory is new and empty), three lines ending `OK`, between them the script's line `magcoupling-rs/assets/fonts/Inter-SemiBold.subset.ttf: 435 glyphs, 386 characters, digit advance 1325`, then `30524`. `git status --short` shows the script and the new font file and nothing else.

- [ ] **Step 2 (red): the face's tests.** The blocks add `INTER_SEMIBOLD` and `SEMIBOLD_NAME` to `fonts.rs` (no code reads them yet), make `inter_alone` a call of a new helper `alone`, and add two tests: `the_semibold_cut_has_the_regular_cut_s_characters_digits_and_rows` (`fonts.rs`: the two cuts hold the same characters up to U+22FF, the SemiBold's digits are tabular and its rows as tall as the Regular's at four sizes, and it is wider; it passes at once, since it reads Step 1's file) and `the_title_and_the_size_heading_are_semibold_and_the_top_bar_keeps_one_row` (`app.rs`: at 1024 x 700, 1280 x 800 and 1920 x 1080 the title and the left column's heading are the SemiBold texts, all of their sections and no other text's; the heading on one line inside the column; every text of the top bar on the title's row inside the window).

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        assert_ne!(first(&plain), fonts::INTER_NAME);
    }
```

with:

```rust
        assert_ne!(first(&plain), fonts::INTER_NAME);
    }

    #[test]
    fn the_title_and_the_size_heading_are_semibold_and_the_top_bar_keeps_one_row() {
        // The stage 1 review (S-1): with one weight the title and the headings were only larger
        // than the text under them. In the app egui's Heading style is Inter SemiBold, which
        // draws the title and the left column's heading and no other text (the tab of the same
        // words stays Regular). A heavier face is wider, so at the smallest window, the laptop
        // screen and a desktop one the top bar's six words stay on the title's row and the
        // heading on one line inside the 320-point column (the column's width and the texts
        // cut on every tab are the BL-099 test's, in the same font).
        use crate::gui::test_support::flat_shapes;
        let semibold = egui::FontFamily::Name(fonts::SEMIBOLD_NAME.into());
        for size in [WINDOWS[1], WINDOWS[0], egui::vec2(1920.0, 1080.0)] {
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            frame_on(&ctx, &mut app, size, Vec::new());
            let output = frame_on(&ctx, &mut app, size, Vec::new());
            let mut heavy = Vec::new();
            for shape in flat_shapes(&output) {
                if let egui::Shape::Text(text) = shape {
                    let sections = &text.galley.job.sections;
                    let semibold_sections = sections
                        .iter()
                        .filter(|s| s.format.font_id.family == semibold)
                        .count();
                    if semibold_sections > 0 {
                        assert_eq!(semibold_sections, sections.len(), "{size:?}");
                        heavy.push((text.galley.text().to_owned(), text.visual_bounding_rect()));
                    }
                }
            }
            let names: Vec<&str> = heavy.iter().map(|(text, _)| text.as_str()).collect();
            let size_heading = crate::gui::CentreView::SizeAndTorque.label();
            assert_eq!(
                names,
                ["Magnetic coupling calculator", size_heading],
                "{size:?}"
            );
            let (title, heading) = (heavy[0].1, heavy[1].1);
            let column = side_rect(&ctx, "magcoupling_inputs");
            let style = ctx.style();
            let row = ctx.fonts(|f| f.row_height(&egui::TextStyle::Heading.resolve(&style)));
            assert!(
                column.contains_rect(heading) && heading.height() <= row,
                "{size:?}: the heading {heading:?} in {column:?}, rows of {row}"
            );
            // The top bar: every text over the left column (the title and the six words) on
            // the title's row, inside the window.
            let bar: Vec<(String, egui::Rect)> = flat_shapes(&output)
                .into_iter()
                .filter_map(|shape| match shape {
                    egui::Shape::Text(text)
                        if text.visual_bounding_rect().bottom() <= column.top() =>
                    {
                        Some((text.galley.text().to_owned(), text.visual_bounding_rect()))
                    }
                    _ => None,
                })
                .collect();
            assert!(bar.len() >= 7, "{size:?}: {bar:?}");
            for (text, at) in &bar {
                assert!(
                    (at.center().y - title.center().y).abs() < 4.0 && at.right() <= size.x,
                    "{size:?}: {text:?} at {at:?}, the title at {title:?}"
                );
            }
        }
    }
```

**`magcoupling-rs/src/app/fonts.rs`**

Replace:

```rust

/// Puts Inter first in the proportional family. egui's built-in fonts stay behind it, for a
```

with:

```rust

/// The cut of Inter SemiBold, for the title and the headings: the same characters, digits and
/// rows as [`INTER`] (the same script, the same list).
pub const INTER_SEMIBOLD: &[u8] = include_bytes!("../../assets/fonts/Inter-SemiBold.subset.ttf");

/// Its name among egui's fonts, and the name of the family it leads.
pub const SEMIBOLD_NAME: &str = "Inter SemiBold";

/// Puts Inter first in the proportional family. egui's built-in fonts stay behind it, for a
```

Replace:

```rust
fn inter_alone() -> egui::Context {
    let mut definitions = egui::FontDefinitions::empty();
```

with:

```rust
fn inter_alone() -> egui::Context {
    alone(INTER_NAME, INTER)
}

/// A context whose only font is `bytes`, named `name`.
#[cfg(test)]
fn alone(name: &str, bytes: &'static [u8]) -> egui::Context {
    let mut definitions = egui::FontDefinitions::empty();
```

Replace:

```rust
    definitions.font_data.insert(
        INTER_NAME.to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(INTER)),
    );
```

with:

```rust
    definitions.font_data.insert(
        name.to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(bytes)),
    );
```

Replace:

```rust
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        definitions
            .families
            .insert(family, vec![INTER_NAME.to_owned()]);
    }
```

with:

```rust
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        definitions.families.insert(family, vec![name.to_owned()]);
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

    #[test]
    fn the_semibold_cut_has_the_regular_cut_s_characters_digits_and_rows() {
        // The title and the headings are drawn in the SemiBold cut (the stage 1 review, S-1),
        // and the panel's glyph guards ask the Regular cut (`cut_holds`): they hold for the
        // headings too because the two cuts hold the same characters (the script's one list,
        // read here over every character up to the maths signs). Its digits are tabular and
        // its rows are as tall as the Regular cut's, so a heading's row and the brand mark
        // beside the title (`gui::look::mark_ui`) do not move. It is heavier, so wider.
        let (regular, semibold) = (inter_alone(), alone(SEMIBOLD_NAME, INTER_SEMIBOLD));
        let font = FontId::proportional(14.0);
        let held = |ctx: &egui::Context, c: char| ctx.fonts(|f| f.has_glyph(&font, c));
        let differ: String = (0x20..0x2300)
            .filter_map(char::from_u32)
            .filter(|&c| held(&regular, c) != held(&semibold, c))
            .collect();
        assert_eq!(differ, "", "held by one cut and not the other");
        assert!(held(&semibold, '\u{03A9}') && !held(&semibold, '\u{0416}'));
        for size in [12.5, 15.0, 18.0, 22.0] {
            let font = FontId::proportional(size);
            let widths: Vec<f32> = "0123456789"
                .chars()
                .map(|digit| semibold.fonts(|f| f.glyph_width(&font, digit)))
                .collect();
            assert!(
                widths.iter().all(|&w| w > 0.0 && w == widths[0]),
                "{size}: {widths:?}"
            );
            let row = |ctx: &egui::Context| ctx.fonts(|f| f.row_height(&font));
            assert_eq!(row(&semibold), row(&regular), "{size}");
        }
        let width = |ctx: &egui::Context| {
            let heading = FontId::proportional(18.0);
            let words = "Size and torque".to_owned();
            ctx.fonts(|f| f.layout_no_wrap(words, heading, egui::Color32::WHITE))
                .size()
                .x
        };
        assert!(width(&semibold) > width(&regular));
    }
}
```

- [ ] **Step 3: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 12
  ```

  Expected: 1 failed: `the_title_and_the_size_heading_are_semibold_and_the_top_bar_keeps_one_row`. It fails at the first window, `[1024.0 700.0]`, with `left: []` and `right: ["Magnetic coupling calculator", "Size and torque"]`: no text is drawn in the SemiBold family yet.

- [ ] **Step 4 (green): the face.** The blocks give `fonts.rs` its `semibold` family and the new `install`, its module comment and the family test's new assertions, and update `theme.rs`'s comment on the heavier face.

**`magcoupling-rs/src/app/fonts.rs`**

Replace:

```rust
//!
//! The file is a cut of Inter Regular, made once by `assets/fonts/subset_inter.py` and
//! committed with its licence (`assets/fonts/OFL.txt`, SIL Open Font License 1.1); the build
//! runs no font tools. Two things the cut does:
//!
```

with:

```rust
//!
//! The files are cuts of Inter Regular and Inter SemiBold, made once by
//! `assets/fonts/subset_inter.py` and committed with their licence (`assets/fonts/OFL.txt`,
//! SIL Open Font License 1.1); the build runs no font tools. What each cut does:
//!
```

Replace:

```rust
//!
//! One face. egui has no bold; a heavier face for numbers and headings is added only if a
//! later slice needs it. The panel itself installs no font: in another host it is drawn in
//! that host's fonts, so the typesetter's substitutions for egui's built-in fonts
//! (`gui::typeset::glyph_safe`) stay as they are.
```

with:

```rust
//!
//! Two faces. egui has no bold, so the app points egui's Heading style at the SemiBold cut:
//! the title and the left column's heading, the panel's two `ui.heading` texts (the stage 1
//! review, S-1). Every other text is Regular; a tile's number keeps it, since the panel names
//! no font. The panel itself installs no font: in another host it is drawn in that host's
//! fonts, so the typesetter's substitutions for egui's built-in fonts
//! (`gui::typeset::glyph_safe`) stay as they are.
```

Replace:

```rust

use egui::epaint::text::{FontInsert, FontPriority, InsertFontFamily};

/// The cut of Inter Regular.
```

with:

```rust

/// The cut of Inter Regular.
```

Replace:

```rust

/// Puts Inter first in the proportional family. egui's built-in fonts stay behind it, for a
/// character outside the cut, and the monospace family is not touched.
pub fn install(ctx: &egui::Context) {
```

with:

```rust

/// The family the SemiBold cut leads: the title and the headings.
pub fn semibold() -> egui::FontFamily {
    egui::FontFamily::Name(SEMIBOLD_NAME.into())
}

/// Puts Inter Regular first in the proportional family and Inter SemiBold first in its own
/// ([`semibold`]), which egui's Heading style then draws in, at the size it had. egui's
/// built-in fonts stay behind each cut, for a character outside it, and the monospace family
/// is not touched.
pub fn install(ctx: &egui::Context) {
```

Replace:

```rust
pub fn install(ctx: &egui::Context) {
    ctx.add_font(FontInsert::new(
        INTER_NAME,
        egui::FontData::from_static(INTER),
        vec![InsertFontFamily {
            family: egui::FontFamily::Proportional,
            priority: FontPriority::Highest,
        }],
    ));
}
```

with:

```rust
pub fn install(ctx: &egui::Context) {
    let mut definitions = egui::FontDefinitions::default();
    let built_in = definitions.families[&egui::FontFamily::Proportional].clone();
    for (name, bytes, family) in [
        (INTER_NAME, INTER, egui::FontFamily::Proportional),
        (SEMIBOLD_NAME, INTER_SEMIBOLD, semibold()),
    ] {
        definitions.font_data.insert(
            name.to_owned(),
            std::sync::Arc::new(egui::FontData::from_static(bytes)),
        );
        let mut fonts = vec![name.to_owned()];
        fonts.extend(built_in.iter().cloned());
        definitions.families.insert(family, fonts);
    }
    ctx.set_fonts(definitions);
    ctx.all_styles_mut(|style| {
        let heading = egui::TextStyle::Heading;
        let size = style.text_styles[&heading].size;
        style
            .text_styles
            .insert(heading, egui::FontId::new(size, semibold()));
    });
}
```

Replace:

```rust
        );
    }
```

with:

```rust
        );
        // The SemiBold cut leads its own family, the built-in fonts behind it, and egui's
        // Heading style draws in it at the size it had; no other style changes.
        proportional[0] = SEMIBOLD_NAME.to_owned();
        assert_eq!(definitions.families[&semibold()], proportional);
        let ctx = installed();
        let plain = egui::Style::default();
        for (style, font) in &ctx.style().text_styles {
            let mut want = plain.text_styles[style].clone();
            if *style == egui::TextStyle::Heading {
                want.family = semibold();
            }
            assert_eq!(*font, want, "{style:?}");
        }
    }
```

**`magcoupling-rs/src/app/theme.rs`**

Replace:

```rust
/// Body text, values and headings share [`TEXT`]; egui has no bold, so `strong()` no longer
/// sets a value apart from the text beside it (a second, heavier face is the cure if a later
/// slice needs one). `weak()` is [`MUTED`], and the panel draws a label beside its value in it
/// (`gui::look::quiet_text`), so the value is the bright one of the pair. Borders are hairlines
/// and nothing casts a shadow (decision C-6). A control at rest shows its edge in [`CONTROL`];
/// under the pointer and pressed its fill stays and its outline brightens.
pub fn visuals() -> egui::Visuals {
```

with:

```rust
/// Body text, values and headings share [`TEXT`]; egui has no bold, so `strong()` no longer
/// sets a value apart from the text beside it (the app's heavier face draws the title and the
/// headings: `app::fonts`). `weak()` is [`MUTED`], and the panel draws a label beside its value
/// in it (`gui::look::quiet_text`), so the value is the bright one of the pair. Borders are
/// hairlines and nothing casts a shadow (decision C-6). A control at rest shows its edge in
/// [`CONTROL`]; under the pointer and pressed its fill stays and its outline brightens.
pub fn visuals() -> egui::Visuals {
```

- [ ] **Step 5: Run the checks** of "Before each commit". Expected: `test result: ok. 1149 passed; 0 failed; 6 ignored` for the lib tests with the app feature; both clippy runs clean; `cargo fmt` changes no text.

- [ ] **Step 6: Commit** the five files (the font file with them): `feat(magcoupler): the title and the left column's heading in Inter SemiBold, a second cut made by the font script (BL-107 T2)`.

### Task 3: The page shell: language, colours and the loading line the app removes

**Model:** `sonnet`.

The page names its language and gives the browser the panel colour and a dark scheme; while the wasm file loads it shows "Loading the calculator" in the muted colour, and the app removes the line at the end of its first frame (decisions D-101 to D-103).

**Files:**
- Modify: `linkage-sim-rs/web/tools/magcoupler/index.html`, `magcoupling-rs/src/app.rs`

**Interfaces:**
- Produces: `app::LOADING_ID: &str` (`magcoupling_loading`), `app::LOADING_LINE: &str`; on wasm32, `fn remove_loading_line()`, called last in `MagcouplingApp`'s `eframe::App::update`.

- [ ] **Step 1 (red): the page's test.** The blocks add `LOADING_ID` and `LOADING_LINE` beside `CANVAS_ID` (no code reads them yet) and `the_page_wears_the_app_s_colours_and_a_loading_line_until_the_app_draws`: `lang="en"`; the theme-color meta tag with the panel colour and the color-scheme meta tag in the head; the line, with its id and words, after the canvas; its style rule with the muted colour, fixed and centred; the script starting the app and not naming the line; the line through the plain-words and gearbox guards, ASCII only.

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
pub const CANVAS_ID: &str = "magcoupling_canvas";

```

with:

```rust
pub const CANVAS_ID: &str = "magcoupling_canvas";

/// Id of the page's loading line in `linkage-sim-rs/web/tools/magcoupler/index.html`, which
/// the app removes when it draws its first frame.
pub const LOADING_ID: &str = "magcoupling_loading";

/// The page's loading line, in the muted colour on the page's charcoal while the wasm file
/// loads (the stage 1 review, S-4): the one text of the page outside the canvas.
pub const LOADING_LINE: &str = "Loading the calculator";

```

Replace:

```rust
    #[test]
    fn the_canvas_id_matches_the_web_page() {
```

with:

```rust
    #[test]
    fn the_page_wears_the_app_s_colours_and_a_loading_line_until_the_app_draws() {
        // The stage 1 review (S-4): while the wasm file loads, the page was charcoal with
        // nothing to say it was working, and the browser's own bars kept their colour. The page
        // names its language, gives the browser the panel colour and a dark scheme from its
        // head (before any style, so nothing flashes white), and shows one line in the muted
        // colour across the middle of the page, after the canvas. The app removes the line at
        // the end of its first frame (`remove_loading_line`, on wasm32 only), so the browser
        // shows the app and the line's removal in one paint; the page's script only starts the
        // app (a browser run checks both).
        let page = include_str!("../../linkage-sim-rs/web/tools/magcoupler/index.html");
        let hex = |c: egui::Color32| format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b());
        assert!(
            page.contains("<html lang=\"en\">"),
            "index.html names its language"
        );
        let head = &page[..page.find("</head>").expect("a head")];
        for (what, meta) in [
            (
                "the browser's colour",
                format!(
                    "<meta name=\"theme-color\" content=\"{}\">",
                    hex(theme::PANEL)
                ),
            ),
            (
                "the dark scheme",
                "<meta name=\"color-scheme\" content=\"dark\">".to_owned(),
            ),
        ] {
            assert!(
                head.contains(&meta),
                "index.html's head gives {what}: {meta}"
            );
        }
        // The line, after the canvas, and its style: the muted colour, centred on the page.
        let line = format!("<p id=\"{LOADING_ID}\">{LOADING_LINE}</p>");
        let canvas = format!("<canvas id=\"{CANVAS_ID}\"></canvas>");
        let at = page.find(&line).expect("index.html has the loading line");
        assert!(page.find(&canvas).is_some_and(|canvas| canvas < at));
        let rule = format!("#{LOADING_ID} {{");
        let rule = &head[head.find(&rule).expect("the line's style")..];
        let rule = &rule[..rule.find('}').expect("its end")];
        for property in [
            format!("color: {};", hex(theme::MUTED)),
            "position: fixed;".to_owned(),
            "text-align: center;".to_owned(),
        ] {
            assert!(rule.contains(&property), "the line's style has {property}");
        }
        // The script starts the app and touches the line nowhere: the app removes it.
        let script = &page[page.find("<script").expect("a script")..];
        assert!(script.contains("init();") && !script.contains(LOADING_ID));
        // A new text on screen: plain words, no gearbox claim, every glyph in the page's font.
        use crate::engine::explain::plain::mentions_gearbox_protection;
        use crate::gui::guidance::plain_text_violations;
        assert_eq!(plain_text_violations(LOADING_LINE), Vec::<&str>::new());
        assert!(!mentions_gearbox_protection(LOADING_LINE));
        assert!(LOADING_LINE.is_ascii());
    }

    #[test]
    fn the_canvas_id_matches_the_web_page() {
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 12
  ```

  Expected: 1 failed: `the_page_wears_the_app_s_colours_and_a_loading_line_until_the_app_draws`. `the_page_wears_the_app_s_colours_and_a_loading_line_until_the_app_draws` panics with `index.html names its language`.

- [ ] **Step 3 (green): the page and the removal.** The blocks give the page its head, the line's style, the line and a comment that says who removes it, and `app.rs` its `remove_loading_line`, compiled for wasm32 only (the wasm32 clippy runs of Task 4 Step 4 check it; a browser run checks what it does, Task 6).

**`linkage-sim-rs/web/tools/magcoupler/index.html`**

Replace:

```html
<!DOCTYPE html>
<html>
<head>
```

with:

```html
<!DOCTYPE html>
<html lang="en">
<head>
```

Replace:

```html
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
```

with:

```html
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="theme-color" content="#181b1f">
    <meta name="color-scheme" content="dark">
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
```

Replace:

```html
        }
    </style>
```

with:

```html
        }
        #magcoupling_loading {
            position: fixed;
            top: 50%;
            left: 0;
            right: 0;
            margin: 0;
            transform: translateY(-50%);
            text-align: center;
            color: #8a919a;
            font: 14px system-ui, sans-serif;
        }
    </style>
```

Replace:

```html
    <!-- Built by scripts/build_magcoupling_web.sh. The canvas id must equal
         magcoupling::app::CANVAS_ID (a magcoupling-rs test checks). Absolute
         paths: /tools/magcoupler (no trailing slash) must not resolve the glue
         against /tools/. -->
    <canvas id="magcoupling_canvas"></canvas>
```

with:

```html
    <!-- Built by scripts/build_magcoupling_web.sh. The canvas id must equal
         magcoupling::app::CANVAS_ID, and the loading line's id and words
         magcoupling::app::LOADING_ID and LOADING_LINE (magcoupling-rs tests
         check). Absolute paths: /tools/magcoupler (no trailing slash) must not
         resolve the glue against /tools/. The line shows while the wasm file
         loads; the app removes it when it draws its first frame. -->
    <canvas id="magcoupling_canvas"></canvas>
```

Replace:

```html
    <canvas id="magcoupling_canvas"></canvas>
    <script type="module">
```

with:

```html
    <canvas id="magcoupling_canvas"></canvas>
    <p id="magcoupling_loading">Loading the calculator</p>
    <script type="module">
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        self.ui(ctx);
    }
```

with:

```rust
        self.ui(ctx);
        #[cfg(target_arch = "wasm32")]
        remove_loading_line();
    }
}

/// Removes the page's loading line ([`LOADING_ID`]) at the end of a frame: eframe paints the
/// frame in the same task, so the browser shows the app's first frame and the line's removal in
/// one paint. (Removed by the page's script when the module had loaded, the line left a blank
/// frame or two before the app's first.) After the first frame there is no line to find.
#[cfg(target_arch = "wasm32")]
fn remove_loading_line() {
    if let Some(line) = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id(LOADING_ID))
    {
        line.remove();
    }
```

- [ ] **Step 4: Run the checks** of "Before each commit", and the wasm32 build of the web binary:

  ```bash
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings
  ```

  Expected: `test result: ok. 1150 passed; 0 failed; 6 ignored` for the lib tests with the app feature; every clippy run clean; `cargo fmt` changes no text.

- [ ] **Step 5: Commit** the two files: `feat(magcoupler): the page names its language and colours and shows a loading line the app removes in its first frame (BL-107 T3)`.

### Task 4: gui-smoke's look check reads the line gone and judges the heavier title and the edges

**Model:** `sonnet`.

On the clean page the look's step also reads that the loading line is gone, by its id typed once as `LOADING_ID`, and judges in its screenshot the heavier title and heading and the grey edge of Auto's box and of the value boxes (decision D-104). `app.rs`'s test holds the script's id to the app's and the step's words; the check's key stays `look`, so the verdict test is unchanged.

**Files:**
- Modify: `.claude/workflows/gui-smoke.js`, `magcoupling-rs/src/app.rs`

**Interfaces:**
- Changes: none to the agent's schema; the prompt's look step reads one more value and judges three more things.

- [ ] **Step 1 (red): the guard.**

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        }
        // The app opens the link and the solve succeeds after its debounce.
```

with:

```rust
        }
        // UX overhaul slice B, stage 2, part 2b-iii-b (the stage 1 review's S-4, S-1, E-4 and
        // R-6): on the clean page the look's step also reads that the page's loading line is
        // gone (its id typed once, as LOADING_ID, equal to the page's), and judges the heavier
        // title and heading and the grey edge of Auto's box and of the value boxes.
        {
            assert_eq!(quoted("LOADING_ID"), LOADING_ID, "the script's LOADING_ID");
            let look = between("Then the look:", "Collect the console");
            let gone = "document.getElementById('${LOADING_ID}') === null";
            for (what, text) in [
                ("reads that the loading line is gone", gone),
                (
                    "judges the heavier title and heading",
                    "heavier than the text under them",
                ),
                ("names Auto's box", crate::gui::AUTO),
                ("judges the controls' edges", "grey edge"),
            ] {
                assert!(look.contains(text), "the look's step {what}: {text}");
            }
        }
        // The app opens the link and the solve succeeds after its debounce.
```

- [ ] **Step 2: Run the red.**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib app::tests::the_smoke_test_share_link_opens_its_design 2>&1 | tail -n 8
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1
  ```

  Expected: 1 failed: `the_smoke_test_share_link_opens_its_design`. `the_smoke_test_share_link_opens_its_design` panics with `no LOADING_ID`. The verdict test ends `all 85 checks hold` (its checks do not change).

- [ ] **Step 3 (green): the prompt.**

**`.claude/workflows/gui-smoke.js`**

Replace:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, the tab row is Size and torque, Geometry, Plots and Results table and the Plots tab\'s picker shows each plot, the left side on the Geometry tab is still the size controls over the folds Detailed inputs and Assumptions, an input tooltip shows no spreadsheet details and a click on its label shows the input\'s entry in the Explain panel, a click on a dimension tag of the drawing shows its result\'s entry there, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

with:

```js
  name: 'gui-smoke',
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design in its own look (a charcoal page with no loading line left, the brand mark, the underlined tab, the app font with its heavier title, the grey edges of the controls), the first view shows the top bar\'s six words, four tiles over the drawing and the check list over the Explain panel, the Design menu holds the details toggle and Load design, a click on a check shows its entry and its lesson link opens its note in the Explain panel, the tab row is Size and torque, Geometry, Plots and Results table and the Plots tab\'s picker shows each plot, the left side on the Geometry tab is still the size controls over the folds Detailed inputs and Assumptions, an input tooltip shows no spreadsheet details and a click on its label shows the input\'s entry in the Explain panel, a click on a dimension tag of the drawing shows its result\'s entry there, a click on the Hot-low tile shows the torque card in the Explain panel, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
  whenToUse: 'Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/, and during Phase 1 audit',
```

Replace:

```js

// The torque card's heading, which the card step looks for in the Explain panel: typed once here. app.rs's smoke test holds it equal to gui::torque_card::CARD_HEADING.
```

with:

```js

// The id of the page's loading line (slice B, stage 2, part 2b-iii-b; the stage 1 review's
// S-4), which the app removes when it draws its first frame: typed once here. app.rs's smoke
// test holds it equal to app::LOADING_ID.
const LOADING_ID = 'magcoupling_loading'

// The torque card's heading, which the card step looks for in the Explain panel: typed once here. app.rs's smoke test holds it equal to gui::torque_card::CARD_HEADING.
```

Replace:

```js
// panel colour, read from the page, and the screenshot shows the brand mark before the title,
// the line under the active tab and text in the app's font. Then
// deterministic state through its share link, checked in
```

with:

```js
// panel colour, read from the page, and the screenshot shows the brand mark before the title,
// the line under the active tab and text in the app's font; since part 2b-iii-b the loading
// line is gone from the page, the title and the left side's heading are heavier than the
// rest, and the controls' edges show in grey. Then
// deterministic state through its share link, checked in
```

Replace:

```js
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: first set the window to 1280 x 800 CSS pixels with browser_resize (the laptop size of the first view). Then the starting design: navigate to ${url}/tools/magcoupler/ (the clean address, with no share link) and wait 8 seconds (WASM init). Take a screenshot and read the address's query with browser_run_code_unsafe, code: async (page) => page.evaluate(() => location.search). starting_design=true only if the check list (the right side), under its heading "CHECKS", is topped by the line "${STARTING_STATUS}", the header row at the top of the page (it may wrap to a second line) shows the words "Undo", "Redo", "Design", "Share", "Export" and "Learn" in this order, each without a box around it, and no "Equation panel" or "Open the example design" button, and the code returned an empty string (at the starting design the address stays clean). Then the look: read the page's background with browser_run_code_unsafe, code: async (page) => page.evaluate(() => getComputedStyle(document.body).backgroundColor). look=true only if the code returned "${PAGE_BACKGROUND}" and the screenshot shows the calculator's own look: a charcoal page; at the top left a small crimson square right before the title "Magnetic coupling calculator"; in the row of tabs the selected tab, "Size and torque", marked by a crimson line under its name and not by a filled button; and every text drawn in one clean sans-serif face, with no empty box where a character should be. Collect the console messages at level "info" (this page's: the next step navigates away).
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
```

with:

```js
    `Smoke-test the WASM magnetic coupling calculator using Playwright MCP tools (${MAGCOUPLING_TOOLS}).
Steps: first set the window to 1280 x 800 CSS pixels with browser_resize (the laptop size of the first view). Then the starting design: navigate to ${url}/tools/magcoupler/ (the clean address, with no share link) and wait 8 seconds (WASM init). Take a screenshot and read the address's query with browser_run_code_unsafe, code: async (page) => page.evaluate(() => location.search). starting_design=true only if the check list (the right side), under its heading "CHECKS", is topped by the line "${STARTING_STATUS}", the header row at the top of the page (it may wrap to a second line) shows the words "Undo", "Redo", "Design", "Share", "Export" and "Learn" in this order, each without a box around it, and no "Equation panel" or "Open the example design" button, and the code returned an empty string (at the starting design the address stays clean). Then the look: read the page's background with browser_run_code_unsafe, code: async (page) => page.evaluate(() => getComputedStyle(document.body).backgroundColor), and whether the page's loading line is gone, code: async (page) => page.evaluate(() => document.getElementById('${LOADING_ID}') === null). look=true only if the first code returned "${PAGE_BACKGROUND}", the second returned true, and the screenshot shows the calculator's own look: a charcoal page; at the top left a small crimson square right before the title "Magnetic coupling calculator"; in the row of tabs the selected tab, "Size and torque", marked by a crimson line under its name and not by a filled button; the title and the heading "Size and torque" at the top of the left side heavier than the text under them; the unticked box of "Auto: smallest" and the value box at the right end of each slider each showing a grey edge; and every text drawn in one clean sans-serif typeface, with no empty box where a character should be. Collect the console messages at level "info" (this page's: the next step navigates away).
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
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

  Expected: `test result: ok. 1150 passed; 0 failed; 6 ignored` for the lib tests with the app feature (1561 tests in all with it, 656 without); every clippy run clean; `all 85 checks hold`.

- [ ] **Step 5: Commit** the two files: `test(magcoupler): gui-smoke's look check reads the loading line gone and judges the heavier title and the controls' edges (BL-107 T4)`.

### Task 5: Docs sweep

**Model:** `sonnet`.

The README's app, fonts, theme, font, web page, smoke and tests passages; `docs/ai/` (02 the system, 03 the modules, 04 the issue, 05 the tracker, the backlog item in progress); the register's rows A-52 to A-54 and its log; owner questions 73 to 75 and the list's count of part C (56); the design's status line and its 6.2's weights.

**Files:**
- Modify: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/analyses/2026-11-18-owner-questions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

- [ ] **Step 1: The docs.**

**`docs/ai/02-system.yaml`**

Replace:

```yaml
      - "Reachability (guidance slice 2, spec 3.7, S2-8): more ways to the notes without turning Explain on first. Start here opens with the torque note; the empty Equation panel names Start here and Explain; Why links (gui::dashboard::why_note_link, one 'Why: <title>' link, never the bare title, which Start here draws once) sit under Magnets per ring and Outside diameter and on the torque card's heading (guidance::SIZE_TAB_NOTE; slice 3), at the end of the Required and Hot-low rows (TORQUE_BAND_NOTE, the hot minimum check's note) and under each copy of the end-effect banner (BANNER_NOTE, the end-effect check's note; dashboard::end_effect_banner_ui); a note not reviewed draws no link, and release_notes_are_reviewed keeps the three reviewed. Under an equation without a reviewed note, or a value with no equation record, Explain shows explorer::note_fallback: the dashboard row's own Why note or assumption, else the nearest reviewed note upstream (breadth first over the registry's terms; a NoteLink::Note link that opens it), else the nearest assumption's rationale, else the row's reason or NO_NOTE, with START_HERE_HINT; NO_NOTE never shows for a dashboard row. The walk runs on each frame Explain shows it (a few hundred paths at most), no cache. G-3 physics review of the strings: PASS in round 3 after round 1's three must-fix items (START_HERE_HINT and EMPTY_TEXT say Start here walks the main notes in a suggested order; NO_NOTE) and round 2's one (NO_NOTE claims only what the panel shows: under a value with no record nothing is known upstream) (docs/analyses/2026-10-14-magcoupler-help-review.md, Reachability strings (T19))."
      - "The look (UX overhaul slice B, stage 1, BL-079; design sections 6 and 8.1 tests 6 and 7): the standalone app owns its theme and its font, and the panel still sets neither. app::theme holds the one token table (BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK, LINK) and visuals(), egui's dark visuals with the tokens in place of its colours: a test holds every field to the table and every other field to egui's, and another holds every text token to 4.5:1 on every surface it is drawn on (muted and the level colours are under it on the border colour, the fill of a hovered button and of a selected chip, whose own text is TEXT and LINK; a caption whose text is selected with the pointer is not counted). Body text, values and headings share TEXT (egui has no bold: strong() no longer sets a value apart), weak() is MUTED. The accent is never a text colour and never on a value or a check: it marks the brand mark and the active tab only (gui::look::mark_ui and tab_ui; a test counts its shapes on the app's page). ACCENT and OK are the panel's own constants (gui::look), since no egui theme has a field for them; the spreadsheet export's fills keep their own palette (xlsx::fill). app::fonts installs a cut of Inter Regular first in the proportional family (egui's built-in fonts stay behind it and stay monospace); the cut is built once by assets/fonts/subset_inter.py, its ten digits are Inter's tabular ones, and with the app feature every glyph guard also holds its texts to the cut (test_support::assert_glyphs through app::fonts::cut_holds). The spacing is unchanged, and decision M41-3 (theme matches the linkage app) is reversed for the calculator alone. Since the fix wave of the stage's whole-branch review (2026-11-22): a label beside its value, a caption's head, a tab that is not active and a plot's axes are drawn in gui::look::quiet_text, a theme's own weak colour (the app's MUTED) and the host's text colour where a host sets none (egui's own weak text is about 2.6:1); the fill behind selected text and of a selected choice is app::theme::SELECTION, a quiet blue and the one colour outside the token table (3:1 against a field and the panel, the text colour on it 4.5:1; register row A-19), with a selected choice's name in the text colour; the panel's own marks (the changed dot, a trace's banner and frames) read gui::look::mark_color, the host's link colour; the font cut carries the ascender, descender and line gap of egui's built-in font, so a row is as tall as the layout's own font made it; and no text is cut sideways in any tab at 1280 x 800 or 1024 x 700 in the app's font, which the app test no_text_is_cut_in_any_tab_in_the_app_s_own_font holds (the pole sweep's line is named in a wrapping caption over its plot, a plot readout longer than a row wraps, a tooltip's equation wider than the window is set smaller, and a plot may paint its axis label a quarter row into the centre region's margin)."
      - "The session baseline and the two named designs (UX overhaul slice A, BL-077; design docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md sections 7.1 and 7.2): DesignInputs::default() is the example design, the spreadsheet's values, and the overhaul never moves it. The standalone app opens on app::starting_design() (seven inputs off the spreadsheet's, among them the standard 1.4 mm hub keyway, and its own fourteen stored 3D fields; Required torque 1.35 N·m; its test pins it to compute_all: every counted check Good, no caution, no material warning, no assumption changed) and hands it to the panel as its baseline (MagcouplingPanel::with_baseline; new() keeps the spreadsheet's values). On screen 'the default' is the baseline: the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's 'Starting design'. The spreadsheet export's changed flag, the assumptions banner and the Assumptions view's dots stay against the workbook. Nothing moves the baseline (no undo, file, link or opened design), and it is in no file, link or undo step. Design files and links do not depend on it: they name every input, and a missing input still reads the spreadsheet's value. On the web the app writes app::address_of(the settled design) into the page's address with history.replaceState: the clean address at the starting design, the Share link elsewhere; Share always copies the full link. It writes only while no key is held (a held Ctrl+Z or Ctrl+Y settles a step at every key repeat; the run gives one write at the release), and address_design advances only when the browser took the write (Safari refuses more than about 100 in 10 seconds); a refused write is asked again on the next frame, with no back-off. If a later physics fix makes the starting design fail, it gets a new starting design, never a looser test. Owner rulings of 2026-11-22 (BL-080; docs/analyses/2026-11-10-conservative-assumptions.md rows A-17 and A-18): its hub keyway is the standard 1.4 mm (DIN 6885-1 t2 for a 3 x 3 mm key), so the wall past it is 2.55 mm and its opening tab draws no caution; its temperature check and the inner hub's fitting temperature read 3D fields computed for its own geometry (STARTING_3D_FIELDS, a run of reference/magcoupling-py/tools/starting_design_fields.py pinned by a test), which stay fixed under an edit like the example design's."
```

with:

```yaml
      - "Reachability (guidance slice 2, spec 3.7, S2-8): more ways to the notes without turning Explain on first. Start here opens with the torque note; the empty Equation panel names Start here and Explain; Why links (gui::dashboard::why_note_link, one 'Why: <title>' link, never the bare title, which Start here draws once) sit under Magnets per ring and Outside diameter and on the torque card's heading (guidance::SIZE_TAB_NOTE; slice 3), at the end of the Required and Hot-low rows (TORQUE_BAND_NOTE, the hot minimum check's note) and under each copy of the end-effect banner (BANNER_NOTE, the end-effect check's note; dashboard::end_effect_banner_ui); a note not reviewed draws no link, and release_notes_are_reviewed keeps the three reviewed. Under an equation without a reviewed note, or a value with no equation record, Explain shows explorer::note_fallback: the dashboard row's own Why note or assumption, else the nearest reviewed note upstream (breadth first over the registry's terms; a NoteLink::Note link that opens it), else the nearest assumption's rationale, else the row's reason or NO_NOTE, with START_HERE_HINT; NO_NOTE never shows for a dashboard row. The walk runs on each frame Explain shows it (a few hundred paths at most), no cache. G-3 physics review of the strings: PASS in round 3 after round 1's three must-fix items (START_HERE_HINT and EMPTY_TEXT say Start here walks the main notes in a suggested order; NO_NOTE) and round 2's one (NO_NOTE claims only what the panel shows: under a value with no record nothing is known upstream) (docs/analyses/2026-10-14-magcoupler-help-review.md, Reachability strings (T19))."
      - "The look (UX overhaul slice B, stage 1, BL-079; design sections 6 and 8.1 tests 6 and 7): the standalone app owns its theme and its font, and the panel still sets neither. app::theme holds the one token table (BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK, LINK) and visuals(), egui's dark visuals with the tokens in place of its colours: a test holds every field to the table and every other field to egui's, and another holds every text token to 4.5:1 on every surface it is drawn on (muted and the level colours are under it on the border colour, the fill of a hovered button and of a selected chip, whose own text is TEXT and LINK; a caption whose text is selected with the pointer is not counted). Body text, values and headings share TEXT (egui has no bold: strong() no longer sets a value apart), weak() is MUTED. The accent is never a text colour and never on a value or a check: it marks the brand mark and the active tab only (gui::look::mark_ui and tab_ui; a test counts its shapes on the app's page). ACCENT and OK are the panel's own constants (gui::look), since no egui theme has a field for them; the spreadsheet export's fills keep their own palette (xlsx::fill). app::fonts installs a cut of Inter Regular first in the proportional family (egui's built-in fonts stay behind it and stay monospace); the cut is built once by assets/fonts/subset_inter.py, its ten digits are Inter's tabular ones, and with the app feature every glyph guard also holds its texts to the cut (test_support::assert_glyphs through app::fonts::cut_holds). The spacing is unchanged, and decision M41-3 (theme matches the linkage app) is reversed for the calculator alone. Since the fix wave of the stage's whole-branch review (2026-11-22): a label beside its value, a caption's head, a tab that is not active and a plot's axes are drawn in gui::look::quiet_text, a theme's own weak colour (the app's MUTED) and the host's text colour where a host sets none (egui's own weak text is about 2.6:1); the fill behind selected text and of a selected choice is app::theme::SELECTION, a quiet blue outside the token table (with CONTROL since BL-107) (3:1 against a field and the panel, the text colour on it 4.5:1; register row A-19), with a selected choice's name in the text colour; the panel's own marks (the changed dot, a trace's banner and frames) read gui::look::mark_color, the host's link colour; the font cut carries the ascender, descender and line gap of egui's built-in font, so a row is as tall as the layout's own font made it; and no text is cut sideways in any tab at 1280 x 800 or 1024 x 700 in the app's font, which the app test no_text_is_cut_in_any_tab_in_the_app_s_own_font holds (the pole sweep's line is named in a wrapping caption over its plot, a plot readout longer than a row wraps, a tooltip's equation wider than the window is set smaller, and a plot may paint its axis label a quarter row into the centre region's margin)."
      - "The session baseline and the two named designs (UX overhaul slice A, BL-077; design docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md sections 7.1 and 7.2): DesignInputs::default() is the example design, the spreadsheet's values, and the overhaul never moves it. The standalone app opens on app::starting_design() (seven inputs off the spreadsheet's, among them the standard 1.4 mm hub keyway, and its own fourteen stored 3D fields; Required torque 1.35 N·m; its test pins it to compute_all: every counted check Good, no caution, no material warning, no assumption changed) and hands it to the panel as its baseline (MagcouplingPanel::with_baseline; new() keeps the spreadsheet's values). On screen 'the default' is the baseline: the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's 'Starting design'. The spreadsheet export's changed flag, the assumptions banner and the Assumptions view's dots stay against the workbook. Nothing moves the baseline (no undo, file, link or opened design), and it is in no file, link or undo step. Design files and links do not depend on it: they name every input, and a missing input still reads the spreadsheet's value. On the web the app writes app::address_of(the settled design) into the page's address with history.replaceState: the clean address at the starting design, the Share link elsewhere; Share always copies the full link. It writes only while no key is held (a held Ctrl+Z or Ctrl+Y settles a step at every key repeat; the run gives one write at the release), and address_design advances only when the browser took the write (Safari refuses more than about 100 in 10 seconds); a refused write is asked again on the next frame, with no back-off. If a later physics fix makes the starting design fail, it gets a new starting design, never a looser test. Owner rulings of 2026-11-22 (BL-080; docs/analyses/2026-11-10-conservative-assumptions.md rows A-17 and A-18): its hub keyway is the standard 1.4 mm (DIN 6885-1 t2 for a 3 x 3 mm key), so the wall past it is 2.55 mm and its opening tab draws no caution; its temperature check and the inner hub's fitting temperature read 3D fields computed for its own geometry (STARTING_3D_FIELDS, a run of reference/magcoupling-py/tools/starting_design_fields.py pinned by a test), which stay fixed under an edit like the example design's."
```

Replace:

```yaml
      - "Since BL-104 (UX overhaul slice B, stage 2, part 2b-iii-a; plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md): the five plots share one tab (design C-9). CentreView keeps a view per plot (ALL and tabs list every view, in the order the tab row reaches them); CentreView::tab is the text of the tab that shows a view (plots::PLOTS for each plot), and centre_ui draws the Plots tab once, at the first plot's place (CentreView::tab_row, the row the smoke prompt's guard reads too), opening MagcouplingPanel::plot, the plot shown last (view state). Over a plot, view_ui draws the picker: the five PlotKind labels as selectable_value in a wrapping row, the plot shown picked. plots.rs draws no red but a sweep row that does not fit (decision C-5): the requirement lines and the pull-out point in visuals.strong_text_color(), the governing limit and the time to it in VIOLET. Each axis's AxisHints::label_spacing is one point (X_TICK_SPACING 60, Y_TICK_SPACING 20, egui_plot's own lower ends), so a tick number is drawn at full strength or not at all (the stage 1 review's E-7)."
      - "Since BL-100: a scroll area follows the keyboard's focus. gui::follow_focus(ui), called last in a scroll area's content, scrolls that area by the least that shows whole (ui.scroll_to_rect(rect, None)) the widget the focus has arrived on, in the first frame the area is drawn with the focus there and never again (it keeps the focused id it saw last in egui's temporary memory, by the content ui's id), so a wheel or a drag of the scroll bar afterwards is not fought, nothing is selected or edited (BL-093) and an idle frame asks for no repaint. Tab, Shift+Tab, an Up or Down arrow key, a click and a request_focus are all arrivals. The widget is this area's when its layer is the area's and its interact_rect is its rect cut by the area's clip rect; a widget of another panel or of a list open over the area is not. Called by the left column and the Explain panel (panel.rs), the Geometry tab's callout list (geometry_view::list_ui) and the Clamp tab (clamp_drawing::clamp_ui). Not by an area nested in another (egui's ScrollArea takes both axes' scroll targets, so the outer area never hears of a widget of the inner one): the Equation panel's formula and term list, the Clamp tab's screw-size table; nor by the Results table, which draws only the rows in view (BL-101). egui's own Tab stops that show no focus are BL-102."
```

with:

```yaml
      - "Since BL-104 (UX overhaul slice B, stage 2, part 2b-iii-a; plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md): the five plots share one tab (design C-9). CentreView keeps a view per plot (ALL and tabs list every view, in the order the tab row reaches them); CentreView::tab is the text of the tab that shows a view (plots::PLOTS for each plot), and centre_ui draws the Plots tab once, at the first plot's place (CentreView::tab_row, the row the smoke prompt's guard reads too), opening MagcouplingPanel::plot, the plot shown last (view state). Over a plot, view_ui draws the picker: the five PlotKind labels as selectable_value in a wrapping row, the plot shown picked. plots.rs draws no red but a sweep row that does not fit (decision C-5): the requirement lines and the pull-out point in visuals.strong_text_color(), the governing limit and the time to it in VIOLET. Each axis's AxisHints::label_spacing is one point (X_TICK_SPACING 60, Y_TICK_SPACING 20, egui_plot's own lower ends), so a tick number is drawn at full strength or not at all (the stage 1 review's E-7)."
      - "Since BL-107 (UX overhaul slice B, stage 2, part 2b-iii-b; plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-b-the-look.md): the look's last three items (design section 6; the stage 1 review's E-4, R-6, S-1 and S-4). A control at rest shows its edge: app::theme::CONTROL (#656c76, the second colour outside the token table, register row A-52) is the resting outline (widgets.inactive.bg_stroke: a value box, a framed button, a picker, a text box) and the resting fill (inactive.bg_fill: a checkbox's box, a radio button, a slider's rail and handle), 3.3:1 on PANEL, 3.0:1 on PANEL2 (egui's menu_style drops the outline, so the Design menu's checkbox shows by its fill alone) and 3.5:1 on BG; hovered.bg_fill and active.bg_fill are CONTROL too, so a box keeps its fill and its outline brightens (MUTED, then TEXT); a slider's handle keeps its TEXT ring (fg_stroke) in every state and shows hover, drag and focus by the ring's width and its size alone (decision D-98). widgets.open.bg_stroke is CONTROL as well, so an open picker (egui's ComboBox draws its box in the open state) keeps its outline; menus drop it (menu_style). egui_plot's legend draws each entry's icon as a disc in the interact state's bg_fill and bg_stroke under the series dot, so every plot legend dot sits on a CONTROL disc (3.5:1 on BG). Frameless widgets (bar_button, tab_ui, menu items, an unselected selectable value) stay frameless; a selected one has SELECTION with the CONTROL edge. The hairlines between the columns are no control and stay BORDER. app::fonts installs the whole FontDefinitions (set_fonts): Inter Regular first in the proportional family and Inter SemiBold first in FontFamily::Name(SEMIBOLD_NAME) (fonts::semibold()), egui's built-in fonts behind each, and points every style's TextStyle::Heading at the SemiBold family at its size, so the panel's two ui.heading texts (the title, the Size column's heading) are SemiBold and no other text is (the panel names no font; register row A-53). The SemiBold cut is made by assets/fonts/subset_inter.py from Inter 4.1's extras/ttf/Inter-SemiBold.ttf (hashes in its docstring) and holds the Regular cut's characters, digits and rows, so cut_holds stays the glyph guard's question. The page (linkage-sim-rs/web/tools/magcoupler/index.html) has lang en, theme-color #181b1f and color-scheme dark in its head and the line LOADING_LINE in MUTED (14 px system font: Inter is inside the wasm file), which app::remove_loading_line (eframe::App::update, wasm32 only) takes off the page at the end of each frame by LOADING_ID; eframe paints in the same task, so the line goes in the paint of the app's first frame (register row A-54). A wasm file that fails to load leaves the line on the page."
      - "Since BL-100: a scroll area follows the keyboard's focus. gui::follow_focus(ui), called last in a scroll area's content, scrolls that area by the least that shows whole (ui.scroll_to_rect(rect, None)) the widget the focus has arrived on, in the first frame the area is drawn with the focus there and never again (it keeps the focused id it saw last in egui's temporary memory, by the content ui's id), so a wheel or a drag of the scroll bar afterwards is not fought, nothing is selected or edited (BL-093) and an idle frame asks for no repaint. Tab, Shift+Tab, an Up or Down arrow key, a click and a request_focus are all arrivals. The widget is this area's when its layer is the area's and its interact_rect is its rect cut by the area's clip rect; a widget of another panel or of a list open over the area is not. Called by the left column and the Explain panel (panel.rs), the Geometry tab's callout list (geometry_view::list_ui) and the Clamp tab (clamp_drawing::clamp_ui). Not by an area nested in another (egui's ScrollArea takes both axes' scroll targets, so the outer area never hears of a widget of the inner one): the Equation panel's formula and term list, the Clamp tab's screw-size table; nor by the Results table, which draws only the rows in view (BL-101). egui's own Tab stops that show no focus are BL-102."
```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): mod.rs (follow_focus: a scroll area scrolls, by the least that shows it whole, to the widget the keyboard's focus arrives on; called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, BL-100; wrapping_line: ui.horizontal_wrapped with one point of the width kept free, for every wrapping line of the left column, of the Explain panel and of the check list's warning links, BL-099 and BL-098), checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), select and select_over (a click asks for the result in the Explain panel: ReadoutEvents::selected, since BL-094), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds; diagram_ui scales a diagram as one picture to a place narrower than DIAGRAM_SIZE, BL-098), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel, whose left column is the Size column on every tab since BL-096 (inputs_ui, size_column_ui: the size controls over the folds DETAILED_INPUTS and ASSUMPTIONS_FOLD, the trace's banner at its foot; no InputsView): new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5, which share one tab since BL-104 (CentreView::tab: plots::PLOTS for each plot; the tab opens the panel's plot field, the plot shown last; view_ui draws the picker of the five over a plot); Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused; CentreView::tab_row(clamps_paused): those views with each run under one tab's text kept once, the row centre_ui draws and the smoke prompt's guard in app.rs reads) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input; the label is one block in the head line, so a wrapped label is one Tab stop) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider (every slider's set-up; its value box's own text, the text egui's formatter wrote in it, is no typed value: typed_number, BL-097), head_line (a row's head line, the changed dot and then the label on a line that wraps at the row's width: an input row's and an assumption's, BL-099), RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); drawing_ui (the opening tab's: the tags, and only the notes with a level, since BL-094); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PLOTS, the centre's one tab of the five plots (decision C-9, BL-104); PlotKind, the choices of its picker, with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; X_TICK_SPACING and Y_TICK_SPACING, each axis's AxisHints label_spacing as one point, so a tick number is at full strength or not drawn (E-7, BL-104); no red but a sweep row that does not fit: the requirement lines and the pull-out point in the strong text colour, the governing limit and the time to it in VIOLET (C-5, BL-104); NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the left column, on every tab since BL-096; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), side_rect (a side panel's rect as egui keeps it: the panel's and the app's width tests), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and SELECTION, the one colour outside it, the fill behind what is selected; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
    workbook_parity: "test-only, enabled by a self dev-dependency; never in a shipped build (linkage-sim-rs/scripts/magcoupling_shipped.sh, gate 10)"
```

with:

```yaml
    gui: "src/gui/ (egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (its wasm feature and js-sys on wasm32); no eframe, no I/O): mod.rs (follow_focus: a scroll area scrolls, by the least that shows it whole, to the widget the keyboard's focus arrives on; called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, BL-100; wrapping_line: ui.horizontal_wrapped with one point of the width kept free, for every wrapping line of the left column, of the Explain panel and of the check list's warning links, BL-099 and BL-098), checks.rs (the check list, UX overhaul slice B stage 2 part 1, BL-081: the status line, a row per counted check in CHECK_GUIDES order with state_mark's shape and its level's word, the material warnings and the stored-3D line; state_mark is every state's mark on the page, the dashboard's badge gone), tiles.rs (the four tiles: Tile, tiles_ui, NEEDS, gap_pieces; hot-low and the requirement quoted by guidance::hot_low_quotes), explain.rs (the Explain panel: Selection, explain_ui, the orientation, a check's entry and a tile's entry with the torque card and the check of its row, the foot links; since BL-085 a note's entry, the note list LEARN and shown_notes; since BL-090 a Size column row's entry, Selection::Input, input_ui, SPREADSHEET_CELL; since BL-092 every input's entry (input_row), TRACE_LINK and Asked::Trace, fix_selection (a fix link selects its input), cell_line (a tile's and a check's cell too)), look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active; quiet_text, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes: a theme's own weak colour, else the host's text colour; mark_color, the host's link colour, for the changed dot and a trace's banner and frames), typeset.rs (the equation typesetter: layout, layout_equation, layout_symbol -> Laid of Ink; TermColors (of, with_members, with_marked), TERM_PALETTE; glyph_safe; equation_ui, laid_ui, term_at), readouts.rs (registry (OnceLock, built once), Readouts (new(marks, show_details), show_details, show and show_over (the hover text built with the frame's show_details), select and select_over (a click asks for the result in the Explain panel: ReadoutEvents::selected, since BL-094), mark, open_note, finish -> ReadoutEvents), OPEN_HINT, MARK_WIDTH, REGISTRY_LOG_PREFIX), explorer.rs (Explorer: open, trail, hovered, focus, explain, start_here, note; open_path, drill, back_to, follow, focus_input, take_scroll, close (clears the focus), open_note, start, stop, marks, end_frame, follow_note_link; explorer_ui (with the held closure), note_ui (NoteView, NoteLink; WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM, HELD_BY_AUTO, AT_TOP, AT_BOTTOM), START_HERE_BUTTON and start_here_hover(clamps_paused) (the walk's topics in START_HERE's order, plain words, the clamp topic only while the section is not paused; explorer_ui takes the panel's clamps_paused for the hover and for used_by_shown), used_by_shown (the Used by links less the paused section's results), held_reason, live_value (a note's {result.path} as the readouts show it; none, drawn as no prediction, where there is none), term_tag, term_label, plain_symbol, reviewed_note (every note shown by id), note_fallback -> NoteFallback {Note, Assumption, Reason} (Explain's next best read under an equation without a note, or a value with no record: the dashboard row's Why link, else the nearest reviewed note upstream, else the nearest assumption, else the row's reason or NO_NOTE), EMPTY_TEXT, NO_NOTE, START_HERE_HINT, NEAREST_NOTE, NEAREST_ASSUMPTION, NoteLink::Note (the closest note's link; follow_note_link opens it), start_here_text (FERRITE_ONLY on the ferrite step); EQUATION_PANEL, panel_heights (the panel's drag range and starting height: PANEL_SHARE 0.45 of the centre region, MIN_PANEL_HEIGHT 120, its top edge stopping VIEW_STRIP 160 below the region's top), MAX_TRAIL, SHOWN_CRUMBS, ELIDED, FOCUS_WIDTH, SWATCH_DIM; its tests hold notes_signoff_sheet (ignored: the owner's sign-off sheet, docs/analyses/2026-10-14-magcoupler-notes-signoff.md, each note shown before slice 2 from notes::before's snapshot and as it ships at the default design, with its review record; signoff_sheet(clamps_paused) says why the clamp note is not on it while paused)), diagrams.rs (diagram_shapes, diagram_ui, DIAGRAM_SIZE: the six notes::Diagram kinds; diagram_ui scales a diagram as one picture to a place narrower than DIAGRAM_SIZE, BL-098), pickers.rs (MATERIAL_PICKERS, PART_PATHS, GRADE_PATHS, picker_ui, choice_hover, material_of, material_summary, material_properties, part_label, part_properties, grade_label, grade_properties), plain_text.rs (slice 2: the display-name and help overlay keyed by path; label_for, help_for, original_help, hint_for, NO_SPREADSHEET_CELL (the provenance line of an input or result without a cell), CUP_WALL_INPUT and shown_literal (R9: the cup wall verdict's text literal with the wall input's label, drawn by the typesetter and verdict_display), display_label (for callers that hold the metadata, and the catalogue while it is built); LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059; REQUIRED_TORQUE, PULL_OUT_NOMINAL and HOT_LOW_TORQUE are the three the Size tab and the plots draw too, typed once), HELP (help batch A, T7: every input of the Requirements, Magnets and rings, Gap and clearances and Shaft, key and clamps groups; help batch B, T8: the 52 empty and 25 replaced helps of the Calibration and model, Materials, Housing and retainers and Thermal and demagnetization groups, and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check; the caveat that ends each row of four clusters (stored 3D fields, bench prototype, unmeasured defaults, optional adapter) is typed once, as a macro the rows are built from with concat! (stored_3d_caveat!, prototype_caveat!, guess_caveat!, adapter_caveat!)), HINTS (keys: the Key design input paths and SIZE_TAB_HINT_KEYS), SIZE_TAB_HINT_KEYS (MINIMUM_OD_HINT_KEY, DIAMETER_HINT_KEY, LENGTH_HINT_KEY: the Size-tab controls' hint keys, which are no inputs)), panel.rs (MagcouplingPanel, whose left column is the Size column on every tab since BL-096 (inputs_ui, size_column_ui: the size controls over the folds DETAILED_INPUTS and ASSUMPTIONS_FOLD, the trace's banner at its foot; no InputsView): new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/Load design/Copy share link/Export spreadsheet/Equation panel and the Show spreadsheet details toggle (show_details, SHOW_DETAILS, SHOW_DETAILS_HINT; off by default, GUI-only), inputs SidePanel, dashboard SidePanel, CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5, which share one tab since BL-104 (CentreView::tab: plots::PLOTS for each plot; the tab opens the panel's plot field, the plot shown last; view_ui draws the picker of the five over a plot); Clamp; Results; CentreView::tabs(clamps_paused): every view but Clamp while the shaft-clamp section is paused; CentreView::tab_row(clamps_paused): those views with each run under one tab's text kept once, the row centre_ui draws and the smoke prompt's guard in app.rs reads) under the end-effect banner, the tab row and banner then the view each in a child ui confined (max rect and clip rect) to the space above the Equation panel (confined), so no view paints into the panel or takes a click inside it; fn ui(&mut self, ui) recomputing compute_all of the shown design every frame; clamps_paused (engine::clamps::PAUSED when a panel opens; a field so the tests draw either state: the tabs, the inputs side, the Assumptions view and banner, the dashboard, the results table, the geometry and the trace banner leave the section out while it is set; panel tests draw the whole section unless a test pauses it); PanelRequest (SaveFile with the file's bytes, OpenDesign) drained by take_requests; export_spreadsheet (both spreadsheet buttons: the header's and the results table's Export XLSX); assumptions_banner; load_design_file, load_share_payload, open_design, open_share_payload (the session's start: no undo step), set_share_base, set_keyboard_shortcuts, report; undo/redo shortcuts; the Key design group's Auto status line (AUTO_STATUS_START), the right column (checks_and_explain_ui, since BL-081: the check list over the Explain panel, one explain::Selection; the torque card, slice 3, is drawn in the Hot-low tile's entry, explain::card_ui, no longer a collapsing header between the readouts and the drawing; auto(), the variable Auto sets, shared with the controls; size_tab_consequences, the Size tab's one ConsequenceCache ask, the lines under the controls' and the card's (one key: the design shown, the Minimum OD and the shaft-clamp pause), handed the design's own results from size_tab_results (ResultsCache: compute_all once per design, none on an idle frame)) and run_sizing), inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, slider_range (an input's slider range from that catalogue, None without a slider: the shared spelling the torque card's gap and count, controls_ui, the guides' slider_bottom and test_support read; explorer.rs and the panel's Required slider keep their own), InputCatalogue::shown(clamps_paused) and without_hidden (what the inputs side draws and filters: every input but the paused section's, emptied sections and groups dropped, built once; get keeps every input for lookups), groups_in(InputOrder), section_of(InputOrder, path); InputGroup::plain_sections, advanced_sections and has_advanced, InputSection::heading_in (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS (re-exported from engine::api), step_decimals, outside_range, text_hint, input_tooltip (the path, the cell and ORIGINAL_NOTE only with show_details; ASSUMPTION_LINE last for a model assumption), value_text (an input's value in words: a blank, a selector's choice; the tooltip's Default line and an input's Explain entry)), input_ui.rs (input_row (a locked row keeps its label live, to select its input; the label is one block in the head line, so a wrapped label is one Tab stop) -> RowOutput {edit, widget, label_clicked (selects the input since BL-092)}, slider (every slider's set-up; its value box's own text, the text egui's formatter wrote in it, is no typed value: typed_number, BL-097), head_line (a row's head line, the changed dot and then the label on a line that wraps at the row's width: an input row's and an assumption's, BL-099), RowEdit, CHANGED_FROM_DEFAULT, OUTSIDE_RANGE_HINT, weak_line, hint_line (the weak hint line under a Key design row or a Size-tab control, from plain_text::hint_for)), dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (33 design checks), verdict_level, check_level, failing_checks (with dashboard_ui and shown_lines, the dashboard rows drawn, leaves the paused section's checks and rows out while clamps_paused), badge, END_EFFECT_ROWS, greyed_by_end_effect (out of range: no badge on a greyed check, the table's and its failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14 stored 3D inputs), STORED_3D_SENTENCE (the sentence every stored 3D input's help ends with, from engine::temperature::stored_3d_sentence!), result_info, result_tooltip (the hover hook; the path, cell and corrections only with show_details), hover_text, MARKER_LEGEND, dashboard_lines, end_effect_banner, END_EFFECT_GREYED, verdict_display, shown_text and DashboardLine::shown_value (R9: the cup wall verdict drawn with the wall input's label on the dashboard, in the results table and in the Equation panel); the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to Readouts::focus_input (surfaced as ReadoutEvents::focus), and a Why: link on every row (WHY_LINK, NO_NOTE_YET); why_note_text and why_note_link (a 'Why: <title>' link outside the rows: the warnings, the Size tab, the end-effect banner) and end_effect_banner_ui (the banner and its link, over the dashboard and the centre region)), geometry.rs (geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth or key width of 0 or less, BL-064) -> Geometry {end, side: View {pieces, dashed, callouts, min, max}, notes}; Part, Outline, Callout with tag/path/level, Note; MAX_DRAWN_POLES 200; CLAIM_REACH 10 (a claim line farther than 10 x the pieces' extent is left off with a note); wall_cautions(inputs, results) -> the Size tab's amber wall lines (the wall past the keyway under about 2.5 mm, moved from panel.rs; the cap skirt under MIN_CAP_SKIRT_MM 0.5, SKIRT_CAUTION_START, BL-082); DESIGN_CHECK, WALL_RULE, NOT_DRAWN, mm, finite), geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); drawing_ui (the opening tab's: the tags, and only the notes with a level, since BL-094); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3); Transform; side_by_side (views at one scale, None without room); arrowhead; plated_text (a text on a plate, shared with the clamp drawing); part_color; DIMENSION), plots.rs (PLOTS, the centre's one tab of the five plots (decision C-9, BL-104); PlotKind, the choices of its picker, with id() from the *_ID consts, torque_at_temperature, torque_temperature, sweep_points, slip_heating, torque_at_angle, torque_angle, plot_ui; legend names (PULL_OUT and REQUIRED are plain_text's PULL_OUT_NOMINAL and REQUIRED_TORQUE); LineName, where a sweep names its line: the legend, or for the pole sweep's sentence a caption over the plot (line_caption_ui); SWEEP_LINE_WIDTH, Y_LABEL_OVERHANG; X_TICK_SPACING and Y_TICK_SPACING, each axis's AxisHints label_spacing as one point, so a tick number is at full strength or not drawn (E-7, BL-104); no red but a sweep row that does not fit: the requirement lines and the pull-out point in the strong text colour, the governing limit and the time to it in VIOLET (C-5, BL-104); NOTHING_TO_PLOT, EMPTY_TEMPERATURE_AXIS, rows_not_plotted), clamp_drawing.rs (unchanged; its tab is offered only while the shaft-clamp section is not paused: clamp_drawing -> ClampDrawing of Mark {Circle, Area, CentreLine, Dimension, Note} per view, NO_SCREW_FITS, fmt_g, clip_to_disc, drawing_ui, clamp_ui, SUMMARY, screw_rows), corrections.rs (CorrectionIndex::get, GOLDEN_FILES, marker_text, marker_tooltip (MARKER_HEADING, each correction's title with was and now at the example design, MARKER_SCOPE; no repo path), correction_title (every title, E25's and E26's included, is the registry's plain title in the hovers and in the spreadsheet's list of corrections)), results_table.rs (table_entries (TableEntry.check), entry_index, search (never the paused section's rows while clamps_paused; ResultsTable::ui takes the pause too), ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, empty_text, CLEAR_TO_CLOSE, row_tooltip, column_widths and row_width (the cell and marker columns only with show_details), entry_level (a row's badge) and worst_level (a group heading's: the worst of its rows', the paused section's left out while clamps_paused), RUST_ONLY, TableAction (ExportCsv, ExportJson, ExportXlsx), EXPORT_XLSX), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner(clamps_paused) (at the foot of the left column, on every tab since BL-096; it counts no path of the paused section); TRACING, CLEAR_TRACE), session.rs (Design, design_to_json/design_from_json, read_design_json and read_share_payload (a LoadedDesign: the design and an optional notice), encode/decode_share_payload, encode_share_text, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement, with the notice only in Torque -> Magnets), history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100), spreadsheet.rs (the spreadsheet export's layout, decisions X-1 to X-12: sheets(&Snapshot) -> Summary, Inputs (in the order shown), Results (by physics chain), Assumptions, each a Sheet (frozen_rows, autofilter) of rows of Cell {CellValue (Empty, Number, Text, Time), CellStyle (Plain, Title, Header, Group, Section, Level)}; value_cell, level_cell; the Inputs ordered and headed by the page's own InputGroup::plain_sections/advanced_sections and InputSection::heading_in, the Results by results_table::table_lines and entry_level; INPUT_COLUMNS, RESULT_COLUMNS, DASHBOARD_COLUMNS, ASSUMPTION_COLUMNS), xlsx.rs (spreadsheet_bytes, xlsx_bytes (rust_xlsxwriter: number cells, frozen headers and autofilters where the sheet asks, widths, bold headings, GROUP_FILL on the group headings, level fills), cell_text (cut at MAX_CELL_CHARS 32,767 UTF-16 code units, as Excel counts, at a whole character), fill (the standalone page's badge colours, a fixed dark-theme palette), now_unix_s (js_sys::Date on wasm32), XLSX_FILE_NAME magcoupling-results.xlsx, XLSX_MIME), guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every guide but the clamp's while the shaft-clamp section is paused) and counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}>, each Quoted built by Quoted::new, the one constructor the table and its test 9 read (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, set by the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text, Direction {Raise, Lower, Change, and Direction::CustomHotterGrade: a ring's magnet part to Custom dimensions with a grade rated hotter than the part, two picks scored as one move}, Fixes {Static, PerTerm}, fixes and all_fixes (with clamps_paused: a guide that does not count shows no fix and holds back no other; a fix is listed only where its move helps; the fit guide drops Raise Outside diameter where no slider diameter fits, fits_on_the_slider), FixCache, within_minimum_diameter (drops Lower Outside diameter where the Size tab's Minimum OD stops the slider), guide_notes (the notes under the fixes: trade_offs, the note under two failing checks whose fixes move one input both ways; side_effects, a fix that sets back another check that is not passing, from SIDE_EFFECTS; unchecked_costs, the sleeve/liner note no check covers), NO_FIX, SIZE_TAB_ALIASES, Learn {Note, Assumption, None}, ROW_LEARN and row_learn, SIZE_TAB_NOTE, TORQUE_BAND_NOTE and BANNER_NOTE (where the Size tab's and the banner's Why links lead), DesignStatus, design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER), Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults), plain_text_violations (re-exported from engine::explain::plain): the banned-pattern scan every guidance test runs; it uses engine::temperature::CURE_MARGIN_MIN_C for the adhesive cure term), size_controls.rs (the Size tab's GUI side: since BL-090 SIZE_ROWS and size_row (the Size column's rows, the keys of their Explain entries), row_help, floor_line, MINIMUM_HOVER, touched and row_label, and controls_ui draws only labels and sliders; REQUIRED, HOT_LOW (the input's and the dashboard row's display labels: plain_text's REQUIRED_TORQUE and HOT_LOW_TORQUE), DIAMETER_HOVER, LENGTH_HOVER, LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label, NO_PREDICTION, NO_TORQUE, AUTO, SET_BY_AUTO, variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step; the count set whole, a value a cast would change (not finite, not whole or past what an i64 holds) refused; the gap and the count arm check no range, which the caller holds to the slider); ControlsOutput.note (the note a Why link under Magnets per ring or Outside diameter asked for); a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump, CupJump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownCountBelowSlider, ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown: a magnet count below the slider's range first, the owner's FO-3, then the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rules; the mapping; side_words, the validity rule of every row, with D-1 for the count; each side stepped by a Slider that names its own control (Slider::control), so a slider is never stepped as another control; the jumps, Shown::jump: words whenever the overlapping magnet length (the Overall length row) or, since the owner's FO-2, the cup's diameter (the Outside diameter row: where the shaft boss sets it) changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines that the count row replaced; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP and CUP_JUMP, CALIBRATION_CHANGE, the four one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, held_by_auto (Auto holds an input: its row locked, a note's change entry plain text), SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25); format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero (decimals_with_a_digit, the one cap rule, which the torque card's shows reads too), TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, the format's bound since the owner's FO-4 at the slice-3 STOP (MOST_WHOLE_DIGITS, six: a number whose fixed form would need more reads four significant digits with its sign, as format_value writes it, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row (T-14; the fold-in's G-3 re-review, MF-2): TorquePrecision, the one precision shared_torque_precision finds for a row, the readout's gap a row of one, which signed_torque_change draws for a torque card side and the readout's margin and torque_shortfall for its shortfall's size, drawn without a sign), and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them), format.rs (format_value, with_unit (the sweep tables' workbook unit Nm written N·m on screen; metadata and exports keep Nm), non_finite_text, search_haystack (display label, catalogue label, path, cell), search_needle), test_support.rs (cfg(test): sized_frame(_at), region_frame (a view in a child ui of a given max rect, unclipped, returning the rect it used), SCREEN, key_event, key_tap, select_all, primary_button, flat_shapes, drawn_texts, drawn_count, text_shapes (every drawn text equal to a needle with the rect egui clips it to: the one exact-text match that text_rects, last_characters, text_clip_rect, text_color and the panel tests' own text probes read), text_colors (the colour of every drawn copy of a text), cut_texts (the texts of a frame whose ink runs more than a point past the left or right edge of the rect egui clips them to or of the screen, a rotated text measured as drawn), side_rect (a side panel's rect as egui keeps it: the panel's and the app's width tests), text_rect, text_rects, text_clip_rect (the rect egui clips a drawn text to, which the torque card's layout tests read), text_center, text_color, assert_glyphs, glyph_context (a context with its fonts loaded, for assert_glyphs), tooltips_at_once, hover_frames (lay out, move the pointer to a target, show its tooltip: the frame before and the last), assert_hover_names_path_only_with_details (a hover site's spreadsheet-details test, both settings), short_magnets, the shared test edits set_parts (both rings' library parts, a blank one leaving its ring on manual blocks: the one parts edit of the designs and edits here), hot_day, small_boss, n42_parts (library N42 blocks in both rings: their magnet limit below the hot-day start, and on the example design the inner ring's single-ring onset less the cure margin caps the inner hub's fitting temperature, BL-064), mixed_rings (inner N42M, outer N35 on manual dimensions: the outer ring governs, and the inner hub's cap is still the inner ring's own onset less the cure margin, BL-064), reversed_hub_band (the inner hub's minimum interference at 20 µm, over its default 18 µm maximum: a reversed band, not judged, BL-064) and hub_overstressed (the inner hub's maximum interference at 26 µm: its bore stress over its limit and its fitting temperature over the allowed one, BL-064), and test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round; one Overall length step trims the reverse pair's inner ring by more than the step), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity, with_cup_depth(11.0): with_cup_depth(mm) the defaults with the cup cavity input at mm), temperature_edge (a hot-day ambient of 82.08 °C, Temperature limits passing; at 82.10 °C a caution), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring), with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side() (the card as the panel asks for it, which the review dump prints too), card_at, side_at, row_line (a card row's line by its label): every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto design solved and shown by the panel's own SizingRunner::shown), auto_states (Auto off, then on each variable it can hold: the states the card's lines are drawn in) and QUOTED_UNITS (the units a quoted result carries, which check_terms' test and the card's margin test read); with_magnets (a design with only its magnet count changed, FreeVariable::MagnetsPerRing.apply: the one count edit of the guides' tests, end_effect_edge and the card's count-row tests), end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups is_shaft_joint (whether a path is one of the shaft joints', BL-064), shaft_joint_input_rows and shaft_joint_result_rows (the shaft joints' input and result rows, in schema order, BL-064), workflow_group (the inputs side's WORKFLOW group by id) and slider_range (inputs::slider_range for a path the test knows has a slider, panicking otherwise), read_xlsx (calamine, a dev-dependency), xlsx_part (zip), snapshot (the spreadsheet tests' Snapshot of a design in the workflow order))"
    app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and the font and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when no key is held and the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address -> bool, whether the page took the write: history.replaceState on wasm32, nothing natively, and under test tests::replace_address, which records each write per thread and can refuse one); open_share_payload logs SHARE_LINK_LOADED, a picked design file DESIGN_FILE_LOADED; run_native shared by both bins; TITLE; CANVAS_ID = magcoupling_canvas; LOADING_ID and LOADING_LINE, the page's loading line, which remove_loading_line (wasm32, called at the end of eframe::App::update) takes off the page, BL-107), src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK, and the two colours outside it, SELECTION, the fill behind what is selected, and CONTROL, the edge of a control at rest (BL-107); visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark; luminance and contrast, the WCAG formulas, tests only), src/app/fonts.rs (INTER and INTER_SEMIBOLD, the cuts of Inter Regular and SemiBold embedded from assets/fonts/; install sets the font definitions, Inter first in the proportional family and the SemiBold first in semibold(), FontFamily::Name(SEMIBOLD_NAME), egui's built-in fonts behind each, and points TextStyle::Heading at semibold(); cut_holds, tests only), assets/fonts/ (the two cuts, subset_inter.py that built them with fontTools: the digits remapped to the tabular ones, and the ascender, descender and line gap of egui's built-in font; OFL.txt), src/app/files.rs (save of a request's bytes: rfd dialog natively (write_file), Blob download on the web; DesignPicker: rfd pick, async on the web); bins src/bin/magcoupling_app.rs (magcoupling-app, native) and src/bin/magcoupling_web.rs (magcoupling-web, wasm32 eframe WebRunner via wasm-bindgen start; reads ?m= and sets the share base to the page's address; native fallback = run_native), both required-features app"
    workbook_parity: "test-only, enabled by a self dev-dependency; never in a shipped build (linkage-sim-rs/scripts/magcoupling_shipped.sh, gate 10)"
```

Replace:

```yaml
    versions: "Cargo.toml caret ranges (egui 0.32, egui_plot 0.33, eframe 0.32, wasm-bindgen 0.2); Cargo.lock pins wasm-bindgen 0.2.114, the deploy-web.yml wasm-bindgen-cli pin, as linkage-sim-rs/Cargo.lock does (gate 11 checks both); egui/eframe need no longer match linkage-sim-rs's (the panel is no longer embedded)"
  web_bundle: "linkage-sim-rs/web/tools/magcoupler/ (index.html committed, canvas magcoupling_canvas, absolute /tools/magcoupler/ glue path; magcoupling-web.js and magcoupling-web_bg.wasm gitignored in web/.gitignore), built by linkage-sim-rs/scripts/build_magcoupling_web.sh (called by build_web.sh, which deploy-web.yml runs), served at colesorkness.com/tools/magcoupler/ (serve_web.sh [PORT] locally); vercel.json revalidates /tools/magcoupler/magcoupling-web.js and redirects the old /magcoupling/ paths there"
  manual_size_gui: "src/gui/size_controls.rs (controls_ui, the direct size/count/gap sliders and a local Minimum OD range preference); panel.rs draws these controls in the left column of every tab, over the closed Detailed inputs and Assumptions folds (the Size column, since BL-096; until then only CentreView::SizeAndTorque's side had them); CentreView::SizeAndTorque keeps live engine torque and geometry in the centre and all dashboard checks, and joins edits to existing history; since slice 1 it is the first view and the solver's home: the Required torque slider (metal.required_min_Nm), Hot-low torque and its gap (torque_gap_text), the Outside diameter and Overall length hovers (DIAMETER_HOVER, LENGTH_HOVER; since BL-074 the length hover also says that the first move from a blank axial magnet length sets both rings to one length), pullout_label, the Auto toggle (AUTO, AUTO_SUFFIX, variable_slider, change_text, SET_BY_AUTO: the matching slider disabled) and the consequence lines under each control (Consequences, one ConsequenceCache held by the panel); since slice 2 a hint line (plain_text::HINTS) under each control, above its consequence lines; since slice 3 the torque card (gui::torque_card, held in the same ConsequenceCache; since BL-081 drawn in the Hot-low tile's Explain entry, explain::card_ui, not between the readouts and the drawing), and under Magnets per ring the ring-size line and the fit line only (the lines for two magnets fewer and two more are the card's count row)"
```

with:

```yaml
    versions: "Cargo.toml caret ranges (egui 0.32, egui_plot 0.33, eframe 0.32, wasm-bindgen 0.2); Cargo.lock pins wasm-bindgen 0.2.114, the deploy-web.yml wasm-bindgen-cli pin, as linkage-sim-rs/Cargo.lock does (gate 11 checks both); egui/eframe need no longer match linkage-sim-rs's (the panel is no longer embedded)"
  web_bundle: "linkage-sim-rs/web/tools/magcoupler/ (index.html committed: lang en, theme-color and color-scheme dark in its head, canvas magcoupling_canvas, then the loading line magcoupling_loading, which the app removes in its first frame (BL-107), absolute /tools/magcoupler/ glue path; magcoupling-web.js and magcoupling-web_bg.wasm gitignored in web/.gitignore), built by linkage-sim-rs/scripts/build_magcoupling_web.sh (called by build_web.sh, which deploy-web.yml runs), served at colesorkness.com/tools/magcoupler/ (serve_web.sh [PORT] locally); vercel.json revalidates /tools/magcoupler/magcoupling-web.js and redirects the old /magcoupling/ paths there"
  manual_size_gui: "src/gui/size_controls.rs (controls_ui, the direct size/count/gap sliders and a local Minimum OD range preference); panel.rs draws these controls in the left column of every tab, over the closed Detailed inputs and Assumptions folds (the Size column, since BL-096; until then only CentreView::SizeAndTorque's side had them); CentreView::SizeAndTorque keeps live engine torque and geometry in the centre and all dashboard checks, and joins edits to existing history; since slice 1 it is the first view and the solver's home: the Required torque slider (metal.required_min_Nm), Hot-low torque and its gap (torque_gap_text), the Outside diameter and Overall length hovers (DIAMETER_HOVER, LENGTH_HOVER; since BL-074 the length hover also says that the first move from a blank axial magnet length sets both rings to one length), pullout_label, the Auto toggle (AUTO, AUTO_SUFFIX, variable_slider, change_text, SET_BY_AUTO: the matching slider disabled) and the consequence lines under each control (Consequences, one ConsequenceCache held by the panel); since slice 2 a hint line (plain_text::HINTS) under each control, above its consequence lines; since slice 3 the torque card (gui::torque_card, held in the same ConsequenceCache; since BL-081 drawn in the Hot-low tile's Explain entry, explain::card_ui, not between the readouts and the drawing), and under Magnets per ring the ring-size line and the fit line only (the lines for two magnets fewer and two more are the card's count row)"
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
  - "Magcoupler, the columns hold still (branch magcoupling/columns-hold-still, cut from main 739a732 on 2026-11-26; one commit per item, in_progress until the controller's reviews, gate and merge). BL-097: a value box left with nothing typed changes nothing. input_ui::slider hands egui's Slider one custom_parser (input_ui::typed_number): the box's own text is no typed value, so Tab, Shift+Tab, Enter or a click elsewhere with nothing typed leaves a size off its step (Auto turned off leaves 11.942 mm shown as 11.9) and a value outside its slider's range bit for bit, with no undo step, no selection and no check moved; a typed number, an arrow key and a drag edit as before. Typing the number the box already shows is no edit either (register row A-43, owner question 66: the conservative answer, not the owner's). Decision D-73's special case for the Minimum OD's slider is removed: its tests hold on the one rule and fail without it. BL-100: a scroll area follows the keyboard's focus. gui::follow_focus, called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, scrolls the area by the least that shows whole the widget the focus has arrived on (Tab, Shift+Tab, an Up or Down arrow key), once, so a wheel is not fought, nothing is selected and an idle frame asks for no repaint. Filed from it, not fixed: BL-101 (the Equation panel, the Clamp tab's screw-size table and the Results table do not follow the focus: nested scroll areas and rows drawn only in view; register row A-44, owner question 67 on whether Tab should walk every row of the Results table) and BL-102 (egui's own Tab stops that show no focus: each scroll area's drag surface and scroll bar, each column's resize edge). BL-099: the left column keeps its 320 points whatever it shows. The head line of an input row and of an assumption wraps (input_ui::head_line), so a long label takes a second line where it ran past the column's edge and widened it for good (330.1 points with the Detailed inputs open in the app's font, 443.1 with the last group open, 418.8 in egui's default font); a group's heading wraps where its traced count makes it longer than the column (panel::wrapped_title; found by the build: 336.6 points with the Required torque traced). The item's evidence blamed the part pickers; they fitted their width, and still cut their choice with an ellipsis (register row A-45, owner question 68: the conservative answer, not the owner's). BL-098: the right column keeps its 300 points whatever the Explain panel shows. A note's diagram was 300 points wide in a column that holds 284 (316 points from the first note with a diagram on); it is now the same picture scaled to the width it is given (diagrams::diagram_ui; register row A-46, owner question 69: about 5 % smaller there). The sweep of every entry (notes, checks, tiles, inputs, all results; both fonts, both designs, details off and on) found one more cause, a wrapping line's label a fraction of a point past the line (0.06 in two notes, 0.375 in one result's entry): every wrapping line of the two side columns is now gui::wrapping_line, which keeps a point free (input_ui::head_line calls it). The sweep found nothing else, so no item was filed from it. risk: mechanical. Fix wave of the controller's review (2026-11-27): BL-097's rule compares the typed number with the text egui's formatter wrote in the box (recorded through Slider::custom_formatter), not with every coarser rounding of the value, which for a tiny value (the vacuum permeability) threw a typed number away; the check list's material warning's Why link wraps (gui::wrapping_line), which widened the right column with no back iron, on main too; tests added for follow_focus's clip check (a wheeled column holds still while the focus is elsewhere) and for the left column's width with Auto on; the grade pickers' blank choice is newly cut in the app's font (A-45 and owner question 68 corrected); BL-103 filed (the widths at pixel ratios 0.9 and 1.35). Fix wave of the re-review (2026-11-27): an input's label is drawn as one block in its head line, so a label that wraps is one Tab stop that shows its focus (SK-1: egui lays a label out in a wrapping line row by row, each row a Tab stop, and only the first showed the focus; 16 input labels wrap at the narrowest width the column gives a row in egui's default font; new with BL-099's wrapping head line); after a wrapped label the note and the reset button take a line of their own; the hold-still test's comment reads its own 24.5 points (RF-1)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-a (BL-104, plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md, decisions D-86 to D-94): the five plots under one Plots tab with a picker (C-9): the picker a row of the five names (D-87, register row A-48, owner question 70), the tab opens on the plot shown last (D-88, register row A-51, the builder's call), the tab row keeps 'Results table' (D-89, A-49, question 71); in a plot red only on a sweep row that does not fit (D-90, A-50, question 72: the requirement lines and the pull-out point in the strong text colour, the limit and the time to it violet); each tick number at full strength or not drawn (D-91, E-7); owner question 56 keeps its default (D-92); gui-smoke's plots_picker (D-93). Part 2b-iii is cut in three (D-86, A-47): 2b-iii-b the look (E-4, R-6, S-1, S-4), 2b-iii-c the Equation panel (BL-086, BL-088, BL-089; BL-101's Equation panel part weighed there). BL-105 filed: at 1024 x 700 and 1280 x 800 four plots' horizontal axes carry one or two numbers or none (egui_plot's 60-point threshold, kept by D-91; on main too). risk: mechanical (D-94)."
```

with:

```yaml
  - "Magcoupler, the columns hold still (branch magcoupling/columns-hold-still, cut from main 739a732 on 2026-11-26; one commit per item, in_progress until the controller's reviews, gate and merge). BL-097: a value box left with nothing typed changes nothing. input_ui::slider hands egui's Slider one custom_parser (input_ui::typed_number): the box's own text is no typed value, so Tab, Shift+Tab, Enter or a click elsewhere with nothing typed leaves a size off its step (Auto turned off leaves 11.942 mm shown as 11.9) and a value outside its slider's range bit for bit, with no undo step, no selection and no check moved; a typed number, an arrow key and a drag edit as before. Typing the number the box already shows is no edit either (register row A-43, owner question 66: the conservative answer, not the owner's). Decision D-73's special case for the Minimum OD's slider is removed: its tests hold on the one rule and fail without it. BL-100: a scroll area follows the keyboard's focus. gui::follow_focus, called last in the content of the left column, the Explain panel, the Geometry tab's callout list and the Clamp tab, scrolls the area by the least that shows whole the widget the focus has arrived on (Tab, Shift+Tab, an Up or Down arrow key), once, so a wheel is not fought, nothing is selected and an idle frame asks for no repaint. Filed from it, not fixed: BL-101 (the Equation panel, the Clamp tab's screw-size table and the Results table do not follow the focus: nested scroll areas and rows drawn only in view; register row A-44, owner question 67 on whether Tab should walk every row of the Results table) and BL-102 (egui's own Tab stops that show no focus: each scroll area's drag surface and scroll bar, each column's resize edge). BL-099: the left column keeps its 320 points whatever it shows. The head line of an input row and of an assumption wraps (input_ui::head_line), so a long label takes a second line where it ran past the column's edge and widened it for good (330.1 points with the Detailed inputs open in the app's font, 443.1 with the last group open, 418.8 in egui's default font); a group's heading wraps where its traced count makes it longer than the column (panel::wrapped_title; found by the build: 336.6 points with the Required torque traced). The item's evidence blamed the part pickers; they fitted their width, and still cut their choice with an ellipsis (register row A-45, owner question 68: the conservative answer, not the owner's). BL-098: the right column keeps its 300 points whatever the Explain panel shows. A note's diagram was 300 points wide in a column that holds 284 (316 points from the first note with a diagram on); it is now the same picture scaled to the width it is given (diagrams::diagram_ui; register row A-46, owner question 69: about 5 % smaller there). The sweep of every entry (notes, checks, tiles, inputs, all results; both fonts, both designs, details off and on) found one more cause, a wrapping line's label a fraction of a point past the line (0.06 in two notes, 0.375 in one result's entry): every wrapping line of the two side columns is now gui::wrapping_line, which keeps a point free (input_ui::head_line calls it). The sweep found nothing else, so no item was filed from it. risk: mechanical. Fix wave of the controller's review (2026-11-27): BL-097's rule compares the typed number with the text egui's formatter wrote in the box (recorded through Slider::custom_formatter), not with every coarser rounding of the value, which for a tiny value (the vacuum permeability) threw a typed number away; the check list's material warning's Why link wraps (gui::wrapping_line), which widened the right column with no back iron, on main too; tests added for follow_focus's clip check (a wheeled column holds still while the focus is elsewhere) and for the left column's width with Auto on; the grade pickers' blank choice is newly cut in the app's font (A-45 and owner question 68 corrected); BL-103 filed (the widths at pixel ratios 0.9 and 1.35). Fix wave of the re-review (2026-11-27): an input's label is drawn as one block in its head line, so a label that wraps is one Tab stop that shows its focus (SK-1: egui lays a label out in a wrapping line row by row, each row a Tab stop, and only the first showed the focus; 16 input labels wrap at the narrowest width the column gives a row in egui's default font; new with BL-099's wrapping head line); after a wrapped label the note and the reset button take a line of their own; the hold-still test's comment reads its own 24.5 points (RF-1)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-b (BL-107, plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-b-the-look.md, decisions D-95 to D-105): the look's last three items, in one plan (D-95). The controls' edges: one more colour outside the token table, app::theme::CONTROL #656c76 (D-96, register row A-52, owner question 73), the darkest of the tokens' hue at 3:1 on a menu (the 2b-iii-a plan's 'about #626973' is 2.9:1 there); framed controls take it, an open picker keeps it, and egui_plot draws every legend dot on a disc of it; frameless ones and the hairlines between the columns stay as they were (D-97); a box and a handle keep their fill under the pointer and pressed, and a handle still shows its state by its ring's width alone (D-98). A SemiBold face for egui's Heading style: the title and the Size column's heading, no other text, tile numbers Regular (D-99, A-53, question 74), installed with the whole font definitions so its family keeps the built-in fonts behind it (D-100). The page shell: lang, theme-color, color-scheme and the line 'Loading the calculator' in the muted colour (D-101, A-54, question 75), removed by the app at the end of its first frame, where the page's script removing it on init() left a blank frame in Chrome (D-102); a wasm file that fails to load leaves the line (D-103). gui-smoke's look check reads the line gone and judges the heavier title and the controls' edges, no new check (D-104). risk: mechanical (D-105). Next: part 2b-iii-c (the Equation panel: BL-086, BL-088, BL-089, BL-101's Equation panel part)."
  - "Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-a (BL-104, plan docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md, decisions D-86 to D-94): the five plots under one Plots tab with a picker (C-9): the picker a row of the five names (D-87, register row A-48, owner question 70), the tab opens on the plot shown last (D-88, register row A-51, the builder's call), the tab row keeps 'Results table' (D-89, A-49, question 71); in a plot red only on a sweep row that does not fit (D-90, A-50, question 72: the requirement lines and the pull-out point in the strong text colour, the limit and the time to it violet); each tick number at full strength or not drawn (D-91, E-7); owner question 56 keeps its default (D-92); gui-smoke's plots_picker (D-93). Part 2b-iii is cut in three (D-86, A-47): 2b-iii-b the look (E-4, R-6, S-1, S-4), 2b-iii-c the Equation panel (BL-086, BL-088, BL-089; BL-101's Equation panel part weighed there). BL-105 filed: at 1024 x 700 and 1280 x 800 four plots' horizontal axes carry one or two numbers or none (egui_plot's 60-point threshold, kept by D-91; on main too). risk: mechanical (D-94)."
```

Replace:

```yaml
  - "ASSUMED FOR NOW (owner ruling 2026-11-10 on open questions: \"Any open questions choose the conservative answer for now and make those assumptions known somewhwre\"): open questions take the conservative answer (the side safe for the design verdict: more load, more heat, less strength, a larger envelope, a stricter check; where equally safe, the option that changes less and can be undone; the workbook-default example never moves on an assumption). The ruling was not itself an approval; on 2026-11-16 the owner approved the diffs and the answers of BL-058, BL-061, BL-062, BL-063 and BL-074 (\"Approve and yes. Stop asking for my direction and do what you think is best\"), so those five are fixed in the backlog; merging and pushing are still not done. Answers taken, each in its backlog entry: BL-061 the shaft boss counts toward the rotating diameter claim; BL-062 sintered NdFeB keeps 6.7e5 S/m, Sm2Co17 1.54e6, SmCo5 2.5e6, ferrite 1.0e2 and bonded NdFeB 2.5e4 S/m as Rust-only grade data; BL-063 report both (the workbook's daily-cycle screen kept, a second screen at the entered bondline); BL-067 option (b), the gearbox and shaft-joint checks compare the larger of cold-high and the operating pull-out with its allowance; BL-074 the Overall length slider's first-move jump is treated as intended, documented and pinned, the mapping unchanged; BL-068's SF-1 stays at option A. Of the five, BL-074 is built on branch magcoupling/off-default-fixes (2026-11-10: the Overall length hover, the docs and four pinned first moves in tests/size_controls.rs; not merged; reviewed, and GATE PASS on 2026-11-11; approved by the owner on 2026-11-16); BL-061 is built there too (2026-11-11: E25, approved by the owner on 2026-11-16, makes the space claim's rotating diameter the largest of the cup, the cap and the shaft boss; the defaults do not move; not merged; reviewed, and GATE PASS on 2026-11-11; approved by the owner on 2026-11-16); BL-063 is built there too (2026-11-11: a second, Rust-only daily-cycle screen at the entered bondline right after the workbook's, which stays cell for cell; on it the default design reads above the fatigue endurance, 4.932 MPa against 3.0 MPa, one more amber check; not merged; reviewed, and GATE PASS on 2026-11-16; approved by the owner on 2026-11-16); BL-062 is built there too (2026-11-11: E26, approved by the owner on 2026-11-16, makes each ring's half of the magnets' eddy heat read its own ring's conductivity by the family of the ring's grade, with sintered NdFeB, every library part and a magnet with no grade on the input; the defaults do not move; not merged; reviewed, and GATE PASS on 2026-11-16; approved by the owner on 2026-11-16); the last, BL-067, is not built yet. The one full list, with what changes if the owner rules otherwise: docs/analyses/2026-11-10-conservative-assumptions.md (untracked in the main checkout on 2026-11-10)."
  - "Decided (owner, 2026-11-08 to 2026-11-16; UX overhaul design sections 1.2 to 1.4, which hold the whole list; these are the ones slice A, BL-077, makes real): O-7 the passing example, the address bar and the seven fixes come first. O-8 and O-9 the opening design's requirement is 1.35 N·m ('Yes 1.35Nm'): 0.35 N·m of hot bearing drag plus 1.00 N·m for acceleration, on a design sized inside the given facts (the 0.6 N·m gearbox input rating at 5:1, the 43 mm space claim, B842SH magnets at their stock length: the design's call C-19). The design's own calls, which the owner approved with it on 2026-11-16: C-1 slice A is on today's screen; C-8 nothing is stored between visits, the address is the only memory; C-20 the opening design is named 'Starting design' and 'Example design' keeps meaning the spreadsheet's; C-21 and C-22 one baseline, fixed for the session, which 'the default' on screen follows while the spreadsheet export and the assumptions banner keep comparing with the workbook; C-23 Share always copies the full link; C-24 a reload does not keep the Size tab's Minimum OD; C-27 'Torque the coupling must carry for driving' stays at 0.7 N·m. Still open and standing on their recommendations (docs/analyses/2026-11-10-conservative-assumptions.md, rows A-2 to A-5): Q-1 the calculator's own theme, Q-3 slip heating is always judged, Q-4 slices A and B are released apart, Q-5 the long link stays. The slice A plan's own calls P-1 to P-14 are in docs/superpowers/plans/2026-11-17-magcoupler-overhaul-a-first-arrival.md; row A-17 is the one that changes what a visitor sees. Slice B's stage 1 (BL-079) makes O-4 and Q-1 real: the look is 'Charcoal and crimson' and decision M41-3 (theme matches the linkage app) is reversed for the calculator alone. Its plan's own calls L-1 to L-16 are in docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md; the ones a visitor sees: a label is muted and its value bright (L-2 as the review amended it; body text and headings share the value's colour, no bold), what is selected sits on a quiet blue fill with its text in the text colour (L-3 as the review changed it; the one colour outside the token table, register row A-19), the digits alone are tabular, the pole sweep's line is named over its plot and not in the legend (L-15), and rows are as tall as on main because the font cut carries the built-in font's vertical metrics (L-16)."
  - "NEXT (owner order, 2026-10-13, G-1; slice 2 shipped 2026-10-16; BL-059, the gearbox check, shipped 2026-10-17; BL-064, the shaft joints, shipped 2026-10-19; slice 3, the torque card, shipped 2026-10-21): guided lessons, plot captions and drawing titles; guides for the results-table checks off the dashboard. Other engine gaps split from slice 2, BL-059 and BL-064: BL-057 (E21), BL-058 (E22, written 2026-11-09 on branch magcoupling/off-default-fixes, not merged, approved by the owner on 2026-11-16; its specific-heat half, E23, is BL-076), BL-060 (the clamp boss link, paused with the clamp section, which BL-064 keeps paused), BL-061..BL-063 (verify first), BL-065 (a 3D check of the end factor with magnetic steel behind the magnets; the owner to place it; the torque card's percents re-run when it lands), BL-066 (the repeated slip swing against the gearbox's nominal rating; needs the owner's datasheet value), BL-067 (an operating temperature below the minimum, where cold-high is not the strongest pull-out; the gearbox and shaft-joint checks both read cold-high), BL-068 (the hot-minimum Problem can read two equal numbers; fixed 2026-11-09 on branch magcoupling/off-default-fixes, not merged), BL-069 (the UltraOx finish against the nickel plating model), BL-070 (fretting and fatigue of the shaft joints), BL-071 (separate shaft diameters; needs the owner's sizes), BL-072 (the temperature verdict's cure margin reads the governing ring's onset: the owner-approved E24, to land with E21). From slice 3: BL-074 (the Overall length slider's first move from a blank override sets both rings to one length; the owner decides whether that is intended; documented and pinned as intended, approved by the owner on 2026-11-16; on branch magcoupling/off-default-fixes, not merged), BL-075 (the BL-064 fold-in's review notes with substance)."
```

with:

```yaml
  - "ASSUMED FOR NOW (owner ruling 2026-11-10 on open questions: \"Any open questions choose the conservative answer for now and make those assumptions known somewhwre\"): open questions take the conservative answer (the side safe for the design verdict: more load, more heat, less strength, a larger envelope, a stricter check; where equally safe, the option that changes less and can be undone; the workbook-default example never moves on an assumption). The ruling was not itself an approval; on 2026-11-16 the owner approved the diffs and the answers of BL-058, BL-061, BL-062, BL-063 and BL-074 (\"Approve and yes. Stop asking for my direction and do what you think is best\"), so those five are fixed in the backlog; merging and pushing are still not done. Answers taken, each in its backlog entry: BL-061 the shaft boss counts toward the rotating diameter claim; BL-062 sintered NdFeB keeps 6.7e5 S/m, Sm2Co17 1.54e6, SmCo5 2.5e6, ferrite 1.0e2 and bonded NdFeB 2.5e4 S/m as Rust-only grade data; BL-063 report both (the workbook's daily-cycle screen kept, a second screen at the entered bondline); BL-067 option (b), the gearbox and shaft-joint checks compare the larger of cold-high and the operating pull-out with its allowance; BL-074 the Overall length slider's first-move jump is treated as intended, documented and pinned, the mapping unchanged; BL-068's SF-1 stays at option A. Of the five, BL-074 is built on branch magcoupling/off-default-fixes (2026-11-10: the Overall length hover, the docs and four pinned first moves in tests/size_controls.rs; not merged; reviewed, and GATE PASS on 2026-11-11; approved by the owner on 2026-11-16); BL-061 is built there too (2026-11-11: E25, approved by the owner on 2026-11-16, makes the space claim's rotating diameter the largest of the cup, the cap and the shaft boss; the defaults do not move; not merged; reviewed, and GATE PASS on 2026-11-11; approved by the owner on 2026-11-16); BL-063 is built there too (2026-11-11: a second, Rust-only daily-cycle screen at the entered bondline right after the workbook's, which stays cell for cell; on it the default design reads above the fatigue endurance, 4.932 MPa against 3.0 MPa, one more amber check; not merged; reviewed, and GATE PASS on 2026-11-16; approved by the owner on 2026-11-16); BL-062 is built there too (2026-11-11: E26, approved by the owner on 2026-11-16, makes each ring's half of the magnets' eddy heat read its own ring's conductivity by the family of the ring's grade, with sintered NdFeB, every library part and a magnet with no grade on the input; the defaults do not move; not merged; reviewed, and GATE PASS on 2026-11-16; approved by the owner on 2026-11-16); the last, BL-067, is not built yet. The one full list, with what changes if the owner rules otherwise: docs/analyses/2026-11-10-conservative-assumptions.md (untracked in the main checkout on 2026-11-10)."
  - "Decided (owner, 2026-11-08 to 2026-11-16; UX overhaul design sections 1.2 to 1.4, which hold the whole list; these are the ones slice A, BL-077, makes real): O-7 the passing example, the address bar and the seven fixes come first. O-8 and O-9 the opening design's requirement is 1.35 N·m ('Yes 1.35Nm'): 0.35 N·m of hot bearing drag plus 1.00 N·m for acceleration, on a design sized inside the given facts (the 0.6 N·m gearbox input rating at 5:1, the 43 mm space claim, B842SH magnets at their stock length: the design's call C-19). The design's own calls, which the owner approved with it on 2026-11-16: C-1 slice A is on today's screen; C-8 nothing is stored between visits, the address is the only memory; C-20 the opening design is named 'Starting design' and 'Example design' keeps meaning the spreadsheet's; C-21 and C-22 one baseline, fixed for the session, which 'the default' on screen follows while the spreadsheet export and the assumptions banner keep comparing with the workbook; C-23 Share always copies the full link; C-24 a reload does not keep the Size tab's Minimum OD; C-27 'Torque the coupling must carry for driving' stays at 0.7 N·m. Still open and standing on their recommendations (docs/analyses/2026-11-10-conservative-assumptions.md, rows A-2 to A-5): Q-1 the calculator's own theme, Q-3 slip heating is always judged, Q-4 slices A and B are released apart, Q-5 the long link stays. The slice A plan's own calls P-1 to P-14 are in docs/superpowers/plans/2026-11-17-magcoupler-overhaul-a-first-arrival.md; row A-17 is the one that changes what a visitor sees. Slice B's stage 1 (BL-079) makes O-4 and Q-1 real: the look is 'Charcoal and crimson' and decision M41-3 (theme matches the linkage app) is reversed for the calculator alone. Its plan's own calls L-1 to L-16 are in docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md; the ones a visitor sees: a label is muted and its value bright (L-2 as the review amended it; body text and headings share the value's colour, no bold), what is selected sits on a quiet blue fill with its text in the text colour (L-3 as the review changed it; then the one colour outside the token table, register row A-19), the digits alone are tabular, the pole sweep's line is named over its plot and not in the legend (L-15), and rows are as tall as on main because the font cut carries the built-in font's vertical metrics (L-16)."
  - "NEXT (owner order, 2026-10-13, G-1; slice 2 shipped 2026-10-16; BL-059, the gearbox check, shipped 2026-10-17; BL-064, the shaft joints, shipped 2026-10-19; slice 3, the torque card, shipped 2026-10-21): guided lessons, plot captions and drawing titles; guides for the results-table checks off the dashboard. Other engine gaps split from slice 2, BL-059 and BL-064: BL-057 (E21), BL-058 (E22, written 2026-11-09 on branch magcoupling/off-default-fixes, not merged, approved by the owner on 2026-11-16; its specific-heat half, E23, is BL-076), BL-060 (the clamp boss link, paused with the clamp section, which BL-064 keeps paused), BL-061..BL-063 (verify first), BL-065 (a 3D check of the end factor with magnetic steel behind the magnets; the owner to place it; the torque card's percents re-run when it lands), BL-066 (the repeated slip swing against the gearbox's nominal rating; needs the owner's datasheet value), BL-067 (an operating temperature below the minimum, where cold-high is not the strongest pull-out; the gearbox and shaft-joint checks both read cold-high), BL-068 (the hot-minimum Problem can read two equal numbers; fixed 2026-11-09 on branch magcoupling/off-default-fixes, not merged), BL-069 (the UltraOx finish against the nickel plating model), BL-070 (fretting and fatigue of the shaft joints), BL-071 (separate shaft diameters; needs the owner's sizes), BL-072 (the temperature verdict's cure margin reads the governing ring's onset: the owner-approved E24, to land with E21). From slice 3: BL-074 (the Overall length slider's first move from a blank override sets both rings to one length; the owner decides whether that is intended; documented and pinned as intended, approved by the owner on 2026-11-16; on branch magcoupling/off-default-fixes, not merged), BL-075 (the BL-064 fold-in's review notes with substance)."
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown

---

## 2026-11-27 - Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-a: the Plots tab (BL-104)
```

with:

```markdown

---

## 2026-11-27 - Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-b: the look (BL-107)
- What changed: every control at rest shows its edge at 3:1 or more (the stage 1 review's E-4 and R-6): a value box, a framed button, a picker and a text box are outlined, and a checkbox's box and a slider's rail and handle filled, in `app::theme::CONTROL` (#656c76, a second colour outside the token table), where the hairline colour was 1.1:1 to 1.2:1; under the pointer and pressed a box keeps its fill and its outline brightens, and a slider's handle still shows its state by its ring's width alone; an open picker keeps the grey outline; every plot legend dot sits on a grey disc, egui_plot's mark of a toggle. The title and the left column's heading "Size and torque" are Inter SemiBold (S-1): a second cut made by the committed script from the recorded Inter 4.1 file, which egui's Heading style draws in; every other text stays Regular. The page names its language, gives the browser the panel colour and a dark scheme, and shows "Loading the calculator" in the muted colour while the wasm file loads, which the app takes away at the end of its first frame, in the same paint (S-4). gui-smoke's look check reads the line gone and judges the heavier title and the controls' edges.
- Tests: a control's resting outline and fill at 3:1 on the panel, a menu and a field, its fill kept under the pointer and pressed, a handle's ring the text colour in every state and the open outline the resting one (theme); every rect and circle of the open left column and the Design menu's checkbox at 3:1 against what is behind it, Auto's picker opened still outlined at 3:1, the seven discs of Torque vs temperature's legend in the resting fill at 3:1 on the legend, the title and the heading the only SemiBold texts with the top bar on one row at 1024 x 700, 1280 x 800 and 1920 x 1080, the page's head, line and script, the smoke prompt's words (app); the SemiBold cut's characters, digits and rows equal to the Regular cut's, and its family and the Heading style (fonts).
- Measured: the face adds 46,869 bytes to the wasm file (6,820,704 to 6,867,573; about 21 KB gzipped, 16 KB with brotli); the page grows from 1,034 to 1,709 bytes. The title is 262.1 points wide in SemiBold (257.7 Regular), the heading 134.6 (133.4). In Chrome the line's removal and the app's first draw fall in the same animation frame.
- Not in this part: part 2b-iii-c (the Equation panel: BL-086, BL-088, BL-089, BL-101's Equation panel part).
- Docs: README, docs/ai 02, 03, 04, backlog BL-107 in_progress, register rows A-52 to A-54, owner questions 73 to 75, the design's header and its 6.2.

## 2026-11-27 - Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-a: the Plots tab (BL-104)
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  priority: 2
  status: open
  notes: "Filed 2026-11-27 with its plan (the design's header: each plan files one item). Part 2b-iii's second of three (decision D-86, register row A-47); 2b-iii-c, the Equation panel (BL-086, BL-088, BL-089, BL-101's Equation panel part), follows with its own plan and item. Not cut again (decision D-95). risk: mechanical (decision D-105): no new number and no physical claim; one new on-screen string, the page's loading line. BL-083, BL-084, BL-087, BL-095, BL-102, BL-103, BL-105 and BL-106 are not touched."

```

with:

```yaml
  priority: 2
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-b (Tasks 1 to 4 of the plan); the full gate, the smoke run and the whole-branch review are its Task 6. Filed 2026-11-27 with its plan (the design's header: each plan files one item). Part 2b-iii's second of three (decision D-86, register row A-47); 2b-iii-c, the Equation panel (BL-086, BL-088, BL-089, BL-101's Equation panel part), follows with its own plan and item. Not cut again (decision D-95). risk: mechanical (decision D-105): no new number and no physical claim; one new on-screen string, the page's loading line. BL-083, BL-084, BL-087, BL-095, BL-102, BL-103, BL-105 and BL-106 are not touched."

```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
| A-51 | C-9 and 4.4 do not say which plot the Plots tab opens on, or what a click on the tab does while a plot shows. Which? | The plot shown last (Torque vs temperature until another is picked), and a click on the tab while a plot shows keeps that plot. View state: in no design file, share link or undo step. | Going back to the plot one was reading stays one click, as its own tab was; a click on the tab of the view shown changes nothing, as on every other tab. No owner question: the builder's call, not the owner's answer. | The tab always opens on Torque vs temperature. | The plan of part 2b-iii-a (BL-104), decision D-88 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
```

with:

```markdown
| A-51 | C-9 and 4.4 do not say which plot the Plots tab opens on, or what a click on the tab does while a plot shows. Which? | The plot shown last (Torque vs temperature until another is picked), and a click on the tab while a plot shows keeps that plot. View state: in no design file, share link or undo step. | Going back to the plot one was reading stays one click, as its own tab was; a click on the tab of the view shown changes nothing, as on every other tab. No owner question: the builder's call, not the owner's answer. | The tab always opens on Torque vs temperature. | The plan of part 2b-iii-a (BL-104), decision D-88 |
| A-52 | Stage 1's review (E-4, R-6) found a checkbox's box, a slider's rail and a value box at rest at 1.1:1 to 1.2:1 on the panel, and no token gives a control's edge 3:1 without other damage (muted as a fill puts the tick at 2.6:1; muted as the outline rings every button in a bright grey). One more colour, or the edges as they are? | One more colour outside the token table, `CONTROL` `#656c76`, a grey of the tokens' hue: the resting outline of a value box, a framed button, a picker and a text box and the resting fill of a checkbox's box, a radio button and a slider's rail and handle; 3.3:1 on the panel, 3.0:1 on a menu, 3.5:1 on a field, the tick 4.3:1 on it. An open picker keeps the grey outline. A plot legend's entries (egui_plot draws each, a toggle, as a disc in the resting fill under its line's dot) show their dots on a grey disc. A box and a handle keep that fill under the pointer and pressed; a box's outline brightens to the muted colour, then the text colour, and a slider's handle shows the pointer, a drag and the keyboard's focus by its ring's width and size alone, as since stage 1. Frameless controls (the top bar's words, the tabs, menu items, the names in a row of choices) stay frameless, and the hairlines between the columns, which are no control, stay the hairline colour. | WCAG's 3:1 for a control's edge (1.4.11): a person finds the box, the rail and the field. It is the darkest grey of that hue at 3:1 on a menu, the lightest surface a control sits on (the review's "about #626973" is 2.9:1 there), so the ring around a button is as quiet as 3:1 allows. An open picker that lost its outline would read as dimmed. The cost: every framed button and a selected choice now carry a grey ring; every legend dot sits on a grey disc (Torque vs temperature's muted dot is 1.7:1 on it); the step from rest to the pointer's ring is smaller (1.7:1, where it was 4.5:1); and a second colour sits outside the design's table (with `SELECTION`, A-19). Owner question 73. | The hairline colour again (1.2:1); or a lighter grey; or the edge on the fill only, leaving buttons and value boxes unringed; or the legend's dots without the disc (the panel's plot code would set it); or a dragged or focused handle on the selection blue, as before stage 1. | The plan of part 2b-iii-b (BL-107), decisions D-96 to D-98 |
| A-53 | The design's 6.2 allows a SemiBold face "for tile numbers and headings", and the review's S-1 named the title and the headings. Which texts? | egui's Heading style: the title and the left column's heading "Size and torque", the panel's two headings. The right column's captions (CHECKS, EXPLAIN), the tiles' numbers and every other text stay Regular. | The panel names no font (the design's 6.3: the app sets the theme and the font), and the Heading style is the one the app can point at a face without the panel naming one; the smallest change that answers S-1. Owner question 74. | The tiles' numbers too (the panel would name a text style the app fills); or the captions too. | The plan of part 2b-iii-b (BL-107), decision D-99 |
| A-54 | S-4 asks for one loading line while the wasm file loads. Which words, and how does it go? | "Loading the calculator", the review's own words, in the muted colour in a 14 px system font across the middle of the charcoal page; the app takes it off the page at the end of its first frame, so it goes in the paint that shows the app. A wasm file that fails to load leaves the line. | The fewest new words that say the page is working; no colour new to the page; the line is gone exactly when the app is there (a browser run measured a blank frame when the page's script removed it). No error text: a new string no one asked for. Owner question 75. | Other words ("Loading…"), a spinner, or no line; an error line when the wasm file fails. | The plan of part 2b-iii-b (BL-107), decisions D-101 to D-103 |
| A-32 | The design's 5.2 says "a link inside the Explain panel selects what it names: a fix link selects and focuses its input". A teaching note's "What to change" entries also link to inputs. Do they select the input too? | No: a note's change-entry link shows the input's row, as now, and the note stays in the Explain panel. A check's fix link selects its input. | The design names the fix link only; a note is a lesson being read (the Start here walk selects one per step), and replacing it on a click would lose the reader's place. Owner question 54. | A note's input link selects the input as a fix link does. | The plan of part 2b-ii-a (BL-092), decision D-55 |
```

Replace:

```markdown
- 2026-11-27: A-50's cost completed by the fix wave of the branch's review (SK-1): Slip heating's legend, once the high case reaches the limit, gives the governing limit and the time to it the same violet dot, as it gave them the same red one before. The answer is unchanged.
- 2026-11-23: A-33, A-34, A-35 and A-36 added with the plan of slice B's stage 2, part 2b-ii-b (BL-094). Each is an assumption, not the owner's answer.
```

with:

```markdown
- 2026-11-27: A-50's cost completed by the fix wave of the branch's review (SK-1): Slip heating's legend, once the high case reaches the limit, gives the governing limit and the time to it the same violet dot, as it gave them the same red one before. The answer is unchanged.
- 2026-11-27: A-52, A-53 and A-54 added with the plan of slice B's stage 2, part 2b-iii-b (BL-107). Each is an assumption, not the owner's answer.
- 2026-11-23: A-33, A-34, A-35 and A-36 added with the plan of slice B's stage 2, part 2b-ii-b (BL-094). Each is an assumption, not the owner's answer.
```

**`docs/analyses/2026-11-18-owner-questions.md`**

Replace:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 53 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54, 55, 56, 57, 58 and 59 on 2026-11-23, questions 60 to 65 on 2026-11-25, questions 66 to 69 on 2026-11-26, questions 70, 71 and 72 on 2026-11-27). Part D, five that matter only if the tools stop being generic.

```

with:

```markdown

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 56 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54, 55, 56, 57, 58 and 59 on 2026-11-23, questions 60 to 65 on 2026-11-25, questions 66 to 69 on 2026-11-26, questions 70 to 75 on 2026-11-27). Part D, five that matter only if the tools stop being generic.

```

Replace:

```markdown

### Linkage simulator
```

with:

```markdown

### 73. Should every control show a grey edge?

Added 2026-11-27 (the next free number). A checkbox's box, a slider's rail and a value box were drawn in the hairline colour, so faint on the dark panel (1.2:1) that the boxes were hard to find. They now show their edge in a new grey, `#656c76`: value boxes, buttons with a frame, pickers and text boxes are outlined in it, and a checkbox's box and a slider's rail and handle are filled with it, at 3:1 against the panel and the menus (the usual rule for a control's edge). An open picker keeps its grey outline. Each name in a plot's legend, which you click to hide or show its line, now shows its coloured dot on a small grey disc; a grey dot (the Operating temperature line of Torque vs temperature) is hard to tell from its disc. The words of the top bar, the tabs and menu items stay without a frame. A selected choice keeps its blue fill with a thin grey edge. A slider's handle still shows that you point at it, drag it or reach it with Tab only by a thicker ring.

- **If you do not answer:** the grey edge.
- (a) As the default.
- (b) The faint edges as before.
- (c) A lighter grey (a brighter ring around every button).
- (d) The grey edge, but the legend's dots without the grey disc.
- (e) The grey edge, and a dragged or focused slider handle turns blue, as it did before the new look.

Source: the slice B stage 1 review, findings E-4 and R-6; A-52; BL-107 plan, decisions D-96 to D-98.

### 74. Should more than the title and the left column's heading be drawn heavier?

Added 2026-11-27 (the next free number). The title "Magnetic coupling calculator" and the heading "Size and torque" over the left column are now drawn in a heavier weight of the same typeface (Inter SemiBold), so they stand out by weight and not only by size. Everything else stays in the regular weight, including the four tiles' big numbers and the small headings CHECKS and EXPLAIN on the right. The design allowed the heavier weight for the tile numbers too.

- **If you do not answer:** the title and the heading only.
- (a) As the default.
- (b) The tiles' numbers too.
- (c) The tiles' numbers and the CHECKS and EXPLAIN headings too.

Source: UX design section 6.2; the slice B stage 1 review, suggestion S-1; A-53; BL-107 plan, decision D-99.

### 75. What should the page say while the calculator loads?

Added 2026-11-27 (the next free number). The calculator is one file of about 6.9 MB, and while it loads the page was dark and empty. It now shows one quiet grey line in the middle, "Loading the calculator", which goes as the calculator appears. If the file cannot load, the line stays.

- **If you do not answer:** "Loading the calculator".
- (a) As the default.
- (b) Other words (say which).
- (c) No line, as before.

Source: the slice B stage 1 review, suggestion S-4; A-54; BL-107 plan, decisions D-101 to D-103.

### Linkage simulator
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, its second (the drawing's selections and the word budget; BL-094) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md`, its third (the left column and the nothing-lost test; BL-096) by the plan `docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md`; part 2b-iii is built in three, and its first (the Plots tab; BL-104) by the plan `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md`; parts 2b-iii-b and 2b-iii-c and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

with:

```markdown

- Date: 2026-11-09. Base: `main` at 1048962. Status: **approved by the owner on 2026-11-16 ("Approve and yes. Stop asking for my direction and do what you think is best"); slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the second stage, sections 4 and 5, is built in two parts, and its first (the check list, the Explain panel, the tiles; BL-081) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-the-instrument.md`; the second part is built in two, and its first (the top bar and the notes; BL-085) by the plan `docs/superpowers/plans/2026-11-22-magcoupler-overhaul-b2-part2a-top-bar-and-notes.md`; part 2b is built in three, and its first (the Size column's lines in the Explain panel; BL-090) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-i-size-column.md`; part 2b-ii is built in three, and its first (every input a selection; BL-092) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-a-inputs.md`, its second (the drawing's selections and the word budget; BL-094) by the plan `docs/superpowers/plans/2026-11-23-magcoupler-overhaul-b2-part2b-ii-b-drawing.md`, its third (the left column and the nothing-lost test; BL-096) by the plan `docs/superpowers/plans/2026-11-25-magcoupler-overhaul-b2-part2b-ii-c-column.md`; part 2b-iii is built in three, and its first (the Plots tab; BL-104) by the plan `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md`, its second (the look; BL-107) by the plan `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-b-the-look.md`; part 2b-iii-c and slices C to F are not built.**
- Revised 2026-11-09 for the owner's ruling on the requirement (O-8, section 7.1) and for the builder check
```

Replace:

```markdown
  a second (SemiBold, for tile numbers and headings) is added only if the spike shows the tiles need it.
- **Cost.** About 0.1 MB per face after subsetting, against a 6.66 MB wasm file today.
```

with:

```markdown
  a second (SemiBold, for tile numbers and headings) is added only if the spike shows the tiles need it.
  (Built 2026-11-27, BL-107: a SemiBold cut, made by the same script, draws egui's Heading style, the title and the
  left column's heading; the tile numbers stay Regular, since the panel names no font: register row A-53. It adds
  46.9 KB to the wasm file.)
- **Cost.** About 0.1 MB per face after subsetting, against a 6.66 MB wasm file today.
```

**`magcoupling-rs/README.md`**

Replace:

```markdown
| `src/gui/` | Feature `gui`: `mod.rs` (`follow_focus`: a scroll area scrolls, by the least that shows it whole, to the widget the keyboard's focus arrives on, BL-100; `wrapping_line`: a line that wraps with one point of its width kept free, for every wrapping line of the left column, of the Explain panel and of the check list's warning links, BL-099 and BL-098), `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, the left column (since BL-096 the Size column on every tab, with its folds `DETAILED_INPUTS` and `ASSUMPTIONS_FOLD`), the assumptions banner, the sizing controls, the right column (`checks_and_explain_ui`: the check list over the Explain panel, one `explain::Selection`, the Hot-low entry's consequences asked of the `ConsequenceCache` through `size_tab_consequences`, the Size tab's one ask, with the design's own results computed once per design (`ResultsCache`))), `tiles.rs` (since BL-081, the four tiles over the drawing on Size and torque: `Tile`, `tiles_ui`, `MIN_TILE_WIDTH`, the Hot-low tile's `needs_lines` and `gap_pieces`), `checks.rs` (since BL-081, the check list: `checks_ui`, `state_mark`, `mark_level`, `CHECKS_SHARE`), `explain.rs` (since BL-081, the Explain panel: `explain_ui`, `Selection`, `Asked`, the orientation and each entry; the torque card sits in the Hot-low tile's entry, `card_ui`, decision C-10; an entry's "Full lesson:" link is `guidance::row_learn`'s), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::new(marks, show_details)`, `show_details`, `Readouts::show` and `show_over` (the hover text built with the frame's `show_details`), `select` and `select_over` (a click selects the result, since BL-094; outlined while focused or selected, `with_selection`), `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `note_ui` (a note drawn against a `NoteView`, returning the `NoteLink` clicked), `held_reason`, `term_tag`, `reviewed_note`, `note_fallback` (Explain's next best read under an equation without a note, `NoteFallback`), `start_here_text` with `FERRITE_ONLY` on the ferrite step; its heights: `panel_heights`, `PANEL_SHARE`, `MIN_PANEL_HEIGHT`, `VIEW_STRIP`), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`, which scales a diagram to a place narrower than its 300 points, BL-098), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `plain_text.rs` (the display-name and help overlay over the engine's metadata, keyed by path: `label_for`, `help_for`, `original_help`, `hint_for`, `display_label`; `LABELS` holds spec 3.5's canonical names, three of them constants the Size tab and the plots draw too: `REQUIRED_TORQUE`, `PULL_OUT_NOMINAL`, `HOT_LOW_TORQUE`; `HELP` and `HINTS` the slice-2 help), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`; `wall_cautions`, the Size tab's amber wall lines: the wall past the keyway and, since BL-082, the cap skirt under `MIN_CAP_SKIRT_MM`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list; `drawing_ui`, the opening tab's, without the list; `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `slider_range` (an input's slider range from the catalogue built once, the shared spelling the torque card, the Size tab's controls, the guides and the tests read; the Equation panel and the panel's Required slider keep their own), `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider` (with `typed_number`: a value box's own text is no typed value, BL-097), `head_line` (a row's head line, the changed dot and then the label, on a line that wraps at the width the row is given: an input row's and an assumption's, BL-099): one input row of any type; a click on its label traces the input; `weak_line` and `hint_line`: a weak line wrapped to its region's width (under a control, and the torque card's lines), and the hint line `plain_text::hint_for` gives a Key design row or a Size-tab control), `dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`, `shown_lines` (the dashboard rows drawn while the shaft-clamp section is paused or not; `dashboard_lines` keeps every row for the exports), `END_EFFECT_ROWS`, `greyed_by_end_effect`, `STORED_3D_ROWS`, `STORED_3D_INPUTS`, `STORED_3D_SENTENCE`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`; `why_note_text` and `why_note_link`, a "Why: <title>" link outside the rows, and `end_effect_banner_ui`, the banner with its link), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), `result_groups.rs` (`result_groups`: the headline, the nine chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`, `target_mismatch_notice`), `history.rs` (`History`: undo and redo), `guidance.rs` (the guidance layer, pure functions: `CHECK_GUIDES`, `counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `check_terms` (`CheckTerm`, `Quoted` with its one constructor `Quoted::new`; slice 3: each counted check's terms in a fixed order, the one table the fix tests and the torque card read), `SPACE_CLAIM_AXIS_SIZES`, `all_fixes`, `FixCache`, `within_minimum_diameter`, `guide_notes` (`trade_offs`, `side_effects` and the unchecked costs), `design_status`, `Consequences` (the lines under the Size tab's controls and its torque card, `card`), `ConsequenceCache` (one key: the design shown, the Minimum OD and the shaft-clamp pause), `magnet_count_line`, `SIZE_TAB_NOTE`, `TORQUE_BAND_NOTE` and `BANNER_NOTE` (where the Size tab's and the banner's Why links lead), `plain_text_violations` (re-exported from `engine::explain::plain`)), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines (since slice 3 without the N-2 and N+2 lines, which are the torque card's count row) and Why links; `slider_minimum`, the Outside diameter slider's lowest value, one expression with the torque card; `SliderMapping`, each Size slider's own mapping of a value to a design, one mapping for the controls and the torque card's steps, which sets the count whole, refusing a value a cast would change, and checks no range on the gap or the count: its caller holds the value to the slider), `torque_card.rs` (the torque card, slice 3, BL-073; pure, no egui: `card` -> `TorqueCard` (`ShownCountBelowSlider`, `ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow`, or `Rows`, one `Row` per Size control (`Control`), read by `rows`), each control one fixed step each way through its slider's own mapping (`size_controls::SliderMapping`, the one the controls apply), in the order of tests of its spec's section 3.3: a side's `Words`, or its `Change` (the stepped design's hot-low, its change in N·m and as a fraction of the design shown's, the Outside diameter and Overall length changes, whether the calibration factor changes, the counted checks whose level flips (`Flip`, over `counted_guides` at the panel's shaft-clamp pause) and the failing checks' quantities the step moves (`MarginChange`, from `guidance::check_terms`, drawn where `shows`)); `lines(auto)` draws it, a row's two N·m at one precision and the row Auto holds as `held_line`, every text a named constant (`CARD_HEADING`, `CAPTION`, the three Auto lines, the side words, `JUMP` and `CUP_JUMP`, `CALIBRATION_CHANGE`, the four one-line forms and the flip words; `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` also end the Size tab's exploration lines); `side_words` (sizing's validity rule, every row's, the keyway left out on the count row: decision D-1) and `count_step` (two magnets each way: checked, even, inside the slider), both moved from the N-2 and N+2 lines that the count row replaced; `STEP_SLACK_MM`, the one slack of the range test and the jump rules: the overlapping magnet length on the Overall length row, the cup's diameter on the Outside diameter row since the owner's FO-2), `sizing.rs` (`SizingMode`, `held_by_auto`, `SizingState` (the mode and the free variable; the target is `metal.required_min_Nm`), `SizingRunner`: debounced inverse sizing over a search range; `sizing_range`, the range Auto searches, one rule for the panel and the torque card's test table, slice 3), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell; the Size tab's one number format, slice 3 decision 10B: `fixed` (a number to a set number of decimals, with no + added), and `signed_fixed`, `signed_change` and `change_decimals` (a change to the fewest decimals, from its own (`TORQUE_DECIMALS`, two, for N·m: the readout's gap and the torque card), that keep its true sign and read a digit that is not zero, at most `CHANGE_EXTRA_DECIMALS` more: `decimals_with_a_digit`, the one cap rule, which the torque card's `shows` reads too), `shared_change_decimals` (a torque card row's two N·m at one precision), the one-decimal `signed_percent`, the format's bound since the owner's FO-4 at the slice-3 STOP (a number whose fixed form would need more than six whole digits, `MOST_WHOLE_DIGITS`, reads four significant digits with its sign, as the readouts write values, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row: `TorquePrecision`, the one precision `shared_torque_precision` finds for a row, the readout's gap a row of one, which `signed_torque_change` draws for a torque card side and the readout's margin and `torque_shortfall` for its shortfall's size, which is drawn without a sign), and `fewest_decimals_apart`, the guidance's search for more decimals), `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries; `mark_ui`, the brand mark before the title; `tab_ui`, a tab of the centre's row, underlined in the accent when it is the active one; `quiet_text`, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes; `mark_color`, the colour of the changed dot and a trace's marks), `test_support.rs` (headless egui helpers and the shared test designs, the torque card's among them with their one table, `card_designs()`; tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests; on the web it writes the settled design into the page's address), `starting_design` (the design it opens on and hands the panel as its baseline), `address_of` (the page address that stands for a design), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the calculator's own theme, "Charcoal and crimson": the token table and the visuals built from it, forced dark), `app/fonts.rs` (the app's font: a cut of Inter Regular with tabular digits, ahead of egui's built-in fonts), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
```

with:

```markdown
| `src/gui/` | Feature `gui`: `mod.rs` (`follow_focus`: a scroll area scrolls, by the least that shows it whole, to the widget the keyboard's focus arrives on, BL-100; `wrapping_line`: a line that wraps with one point of its width kept free, for every wrapping line of the left column, of the Explain panel and of the check list's warning links, BL-099 and BL-098), `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, the left column (since BL-096 the Size column on every tab, with its folds `DETAILED_INPUTS` and `ASSUMPTIONS_FOLD`), the assumptions banner, the sizing controls, the right column (`checks_and_explain_ui`: the check list over the Explain panel, one `explain::Selection`, the Hot-low entry's consequences asked of the `ConsequenceCache` through `size_tab_consequences`, the Size tab's one ask, with the design's own results computed once per design (`ResultsCache`))), `tiles.rs` (since BL-081, the four tiles over the drawing on Size and torque: `Tile`, `tiles_ui`, `MIN_TILE_WIDTH`, the Hot-low tile's `needs_lines` and `gap_pieces`), `checks.rs` (since BL-081, the check list: `checks_ui`, `state_mark`, `mark_level`, `CHECKS_SHARE`), `explain.rs` (since BL-081, the Explain panel: `explain_ui`, `Selection`, `Asked`, the orientation and each entry; the torque card sits in the Hot-low tile's entry, `card_ui`, decision C-10; an entry's "Full lesson:" link is `guidance::row_learn`'s), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::new(marks, show_details)`, `show_details`, `Readouts::show` and `show_over` (the hover text built with the frame's `show_details`), `select` and `select_over` (a click selects the result, since BL-094; outlined while focused or selected, `with_selection`), `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `note_ui` (a note drawn against a `NoteView`, returning the `NoteLink` clicked), `held_reason`, `term_tag`, `reviewed_note`, `note_fallback` (Explain's next best read under an equation without a note, `NoteFallback`), `start_here_text` with `FERRITE_ONLY` on the ferrite step; its heights: `panel_heights`, `PANEL_SHARE`, `MIN_PANEL_HEIGHT`, `VIEW_STRIP`), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`, which scales a diagram to a place narrower than its 300 points, BL-098), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `plain_text.rs` (the display-name and help overlay over the engine's metadata, keyed by path: `label_for`, `help_for`, `original_help`, `hint_for`, `display_label`; `LABELS` holds spec 3.5's canonical names, three of them constants the Size tab and the plots draw too: `REQUIRED_TORQUE`, `PULL_OUT_NOMINAL`, `HOT_LOW_TORQUE`; `HELP` and `HINTS` the slice-2 help), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`; `wall_cautions`, the Size tab's amber wall lines: the wall past the keyway and, since BL-082, the cap skirt under `MIN_CAP_SKIRT_MM`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list; `drawing_ui`, the opening tab's, without the list; `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `slider_range` (an input's slider range from the catalogue built once, the shared spelling the torque card, the Size tab's controls, the guides and the tests read; the Equation panel and the panel's Required slider keep their own), `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider` (with `typed_number`: a value box's own text is no typed value, BL-097), `head_line` (a row's head line, the changed dot and then the label, on a line that wraps at the width the row is given: an input row's and an assumption's, BL-099): one input row of any type; a click on its label traces the input; `weak_line` and `hint_line`: a weak line wrapped to its region's width (under a control, and the torque card's lines), and the hint line `plain_text::hint_for` gives a Key design row or a Size-tab control), `dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`, `shown_lines` (the dashboard rows drawn while the shaft-clamp section is paused or not; `dashboard_lines` keeps every row for the exports), `END_EFFECT_ROWS`, `greyed_by_end_effect`, `STORED_3D_ROWS`, `STORED_3D_INPUTS`, `STORED_3D_SENTENCE`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`; `why_note_text` and `why_note_link`, a "Why: <title>" link outside the rows, and `end_effect_banner_ui`, the banner with its link), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), `result_groups.rs` (`result_groups`: the headline, the nine chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`, `target_mismatch_notice`), `history.rs` (`History`: undo and redo), `guidance.rs` (the guidance layer, pure functions: `CHECK_GUIDES`, `counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `check_terms` (`CheckTerm`, `Quoted` with its one constructor `Quoted::new`; slice 3: each counted check's terms in a fixed order, the one table the fix tests and the torque card read), `SPACE_CLAIM_AXIS_SIZES`, `all_fixes`, `FixCache`, `within_minimum_diameter`, `guide_notes` (`trade_offs`, `side_effects` and the unchecked costs), `design_status`, `Consequences` (the lines under the Size tab's controls and its torque card, `card`), `ConsequenceCache` (one key: the design shown, the Minimum OD and the shaft-clamp pause), `magnet_count_line`, `SIZE_TAB_NOTE`, `TORQUE_BAND_NOTE` and `BANNER_NOTE` (where the Size tab's and the banner's Why links lead), `plain_text_violations` (re-exported from `engine::explain::plain`)), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines (since slice 3 without the N-2 and N+2 lines, which are the torque card's count row) and Why links; `slider_minimum`, the Outside diameter slider's lowest value, one expression with the torque card; `SliderMapping`, each Size slider's own mapping of a value to a design, one mapping for the controls and the torque card's steps, which sets the count whole, refusing a value a cast would change, and checks no range on the gap or the count: its caller holds the value to the slider), `torque_card.rs` (the torque card, slice 3, BL-073; pure, no egui: `card` -> `TorqueCard` (`ShownCountBelowSlider`, `ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow`, or `Rows`, one `Row` per Size control (`Control`), read by `rows`), each control one fixed step each way through its slider's own mapping (`size_controls::SliderMapping`, the one the controls apply), in the order of tests of its spec's section 3.3: a side's `Words`, or its `Change` (the stepped design's hot-low, its change in N·m and as a fraction of the design shown's, the Outside diameter and Overall length changes, whether the calibration factor changes, the counted checks whose level flips (`Flip`, over `counted_guides` at the panel's shaft-clamp pause) and the failing checks' quantities the step moves (`MarginChange`, from `guidance::check_terms`, drawn where `shows`)); `lines(auto)` draws it, a row's two N·m at one precision and the row Auto holds as `held_line`, every text a named constant (`CARD_HEADING`, `CAPTION`, the three Auto lines, the side words, `JUMP` and `CUP_JUMP`, `CALIBRATION_CHANGE`, the four one-line forms and the flip words; `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` also end the Size tab's exploration lines); `side_words` (sizing's validity rule, every row's, the keyway left out on the count row: decision D-1) and `count_step` (two magnets each way: checked, even, inside the slider), both moved from the N-2 and N+2 lines that the count row replaced; `STEP_SLACK_MM`, the one slack of the range test and the jump rules: the overlapping magnet length on the Overall length row, the cup's diameter on the Outside diameter row since the owner's FO-2), `sizing.rs` (`SizingMode`, `held_by_auto`, `SizingState` (the mode and the free variable; the target is `metal.required_min_Nm`), `SizingRunner`: debounced inverse sizing over a search range; `sizing_range`, the range Auto searches, one rule for the panel and the torque card's test table, slice 3), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell; the Size tab's one number format, slice 3 decision 10B: `fixed` (a number to a set number of decimals, with no + added), and `signed_fixed`, `signed_change` and `change_decimals` (a change to the fewest decimals, from its own (`TORQUE_DECIMALS`, two, for N·m: the readout's gap and the torque card), that keep its true sign and read a digit that is not zero, at most `CHANGE_EXTRA_DECIMALS` more: `decimals_with_a_digit`, the one cap rule, which the torque card's `shows` reads too), `shared_change_decimals` (a torque card row's two N·m at one precision), the one-decimal `signed_percent`, the format's bound since the owner's FO-4 at the slice-3 STOP (a number whose fixed form would need more than six whole digits, `MOST_WHOLE_DIGITS`, reads four significant digits with its sign, as the readouts write values, and so does a torque change that reads zero at the cap beside a percent that reads a digit, and with it the other N·m of its torque card row: `TorquePrecision`, the one precision `shared_torque_precision` finds for a row, the readout's gap a row of one, which `signed_torque_change` draws for a torque card side and the readout's margin and `torque_shortfall` for its shortfall's size, which is drawn without a sign), and `fewest_decimals_apart`, the guidance's search for more decimals), `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries; `mark_ui`, the brand mark before the title; `tab_ui`, a tab of the centre's row, underlined in the accent when it is the active one; `quiet_text`, the colour of a label beside its value, a caption's head, a tab that is not active and a plot's axes; `mark_color`, the colour of the changed dot and a trace's marks), `test_support.rs` (headless egui helpers and the shared test designs, the torque card's among them with their one table, `card_designs()`; tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests; on the web it writes the settled design into the page's address and, at the end of each frame, takes the page's loading line away), `starting_design` (the design it opens on and hands the panel as its baseline), `address_of` (the page address that stands for a design), `run_native`, `TITLE`, `CANVAS_ID`, `LOADING_ID`, `LOADING_LINE`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the calculator's own theme, "Charcoal and crimson": the token table and the visuals built from it, forced dark), `app/fonts.rs` (the app's font: cuts of Inter Regular and SemiBold with tabular digits, ahead of egui's built-in fonts; the SemiBold draws the Heading style), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
```

Replace:

```markdown
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
| `assets/fonts/` | The app's font (feature `app`): `Inter-Regular.subset.ttf`, the cut `src/app/fonts.rs` embeds; `subset_inter.py`, the command that built it (fontTools; the build does not run it); `OFL.txt`, Inter's licence |
| `tests/` | Parity, differential, metadata and registry tests (below) |
```

with:

```markdown
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
| `assets/fonts/` | The app's font (feature `app`): `Inter-Regular.subset.ttf` and `Inter-SemiBold.subset.ttf`, the cuts `src/app/fonts.rs` embeds; `subset_inter.py`, the command that built them (fontTools; the build does not run it); `OFL.txt`, Inter's licence |
| `tests/` | Parity, differential, metadata and registry tests (below) |
```

Replace:

```markdown
  egui's dark visuals with the tokens in place of its colours: hairline borders, one corner
  radius and no shadows. One colour is outside the table, the fill behind what is selected
  (`SELECTION`, `#466a90`, a quiet blue in the hue of the links): egui paints selected text and
```

with:

```markdown
  egui's dark visuals with the tokens in place of its colours: hairline borders, one corner
  radius and no shadows. Two colours are outside the table. One is the fill behind what is selected
  (`SELECTION`, `#466a90`, a quiet blue in the hue of the links): egui paints selected text and
```

Replace:

```markdown
  lighter fill would take the text under 4.5:1, which is why a tooltip stops at 2.8:1), and a
  selected choice's name is the text colour, as a focused field's outline is. The panel's own
  marks (the changed dot, a trace's banner and frames, the frame around a row an equation
```

with:

```markdown
  lighter fill would take the text under 4.5:1, which is why a tooltip stops at 2.8:1), and a
  selected choice's name is the text colour, as a focused field's outline is. The other, since
  BL-107 (the stage 1 review's E-4 and R-6), is the edge of a control at rest (`CONTROL`,
  `#656c76`, a grey of the tokens' hue): the outline of a value box, a framed button, a picker and
  a text box, and the fill of a checkbox's box, a radio button and a slider's rail and handle,
  which the hairline colour drew at 1.1:1 to 1.2:1. It is 3.3:1 on the panel, 3.0:1 on a menu
  (whose checkbox egui draws with no outline) and 3.5:1 on a field, and a ticked box's tick, in
  the text colour, is 4.3:1 on it (WCAG's 3:1 for a control's edge); under the pointer and
  pressed a box and a handle keep that fill and a box's outline brightens, to the muted colour
  (1.7:1 to the grey, where it was 4.5:1 to the hairline colour), then the text colour. A
  slider's handle keeps its ring in the text colour in every state, so it shows the pointer, a
  drag and the keyboard's focus by the ring's width and its size alone, as since stage 1. An open
  picker keeps the grey outline (egui draws it in its open state). egui_plot draws each entry of
  a plot's legend, a toggle that hides or shows its line, as a disc in the resting fill and
  outline under the line's dot, so every legend dot now sits on a grey disc (3.5:1 on the
  legend's background; Torque vs temperature's muted dot is 1.7:1 on it). Frameless controls
  stay frameless (the top bar's words, the tabs, menu items and the names in a row of choices,
  whose words name them), and a selected choice keeps its fill, edged in the grey at 1.1:1 to
  it. The hairlines between the columns are no control and stay the hairline colour. The panel's own
  marks (the changed dot, a trace's banner and frames, the frame around a row an equation
```

Replace:

```markdown
  be drawn in a built-in font, a different face in the middle of a word, and the cure is to add
  it to the script's list and build the cut again. One face: a heavier one for numbers and
  headings waits for a slice that needs it. The panel installs no font: another host draws it in
  its own.

```

with:

```markdown
  be drawn in a built-in font, a different face in the middle of a word, and the cure is to add
  it to the script's list and build the cut again. A second cut, Inter SemiBold (30,524 bytes;
  since BL-107, the stage 1 review's S-1), is made by the same script from the same Inter 4.1 zip
  with the same list and metrics; it leads a family of its own (`fonts::semibold`, egui's
  built-in fonts behind it), and the app points egui's Heading style at it at the size it had.
  So the title and the left column's heading, "Size and torque", the panel's two `ui.heading`
  texts, are SemiBold, and every other text is Regular, a tile's number too: the panel names no
  font. The cut holds the Regular cut's characters, tabular digits and rows (a test), so the
  glyph guards hold for the headings and no row moves. It is 1 to 2 % wider: the title is 262.1
  points where it was 257.7 and the heading 134.6 where it was 133.4, and the top bar stays one
  row and the left column 320 points at 1024 x 700, 1280 x 800 and 1920 x 1080. It adds 46,869
  bytes to the wasm file (6,820,704 to 6,867,573; about 21 KB more gzipped, 16 KB with
  brotli). The panel installs no font: another host draws it in its own.

```

Replace:

```markdown
glue and the wasm are gitignored build outputs). Its canvas id is
`app::CANVAS_ID`, which a test checks. `deploy-web.yml` runs
`build_web.sh`, which builds the linkage bundle and then runs `build_magcoupling_web.sh`, so both bundles ship
```

with:

```markdown
glue and the wasm are gitignored build outputs). Its canvas id is
`app::CANVAS_ID`, which a test checks. Since BL-107 (the stage 1 review's S-4) the page names
its language (`lang="en"`), gives the browser the panel colour (`theme-color`) and a dark
scheme (`color-scheme`) from its head, and while the wasm file loads shows one line,
`app::LOADING_LINE` ("Loading the calculator"), in the muted colour across the middle of the
charcoal page. The app takes it away (`remove_loading_line`, on wasm32, by `app::LOADING_ID`) at
the end of its first frame, which eframe paints in the same task, so the browser shows the app
and the line's going in one paint (measured in Chrome: the same animation frame, where the
page's script removing it when the module loaded left a blank frame first). A wasm file that
fails to load leaves the line on the page. `deploy-web.yml` runs
`build_web.sh`, which builds the linkage bundle and then runs `build_magcoupling_web.sh`, so both bundles ship
```

Replace:

```markdown
share links included) on the next push of `main`; pushing needs the user's
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it first opens `/tools/magcoupler/` at the clean address and checks the status line "Starting design: passes all 9 checks", the header's "Open the example design" button and that the address stays clean (UX overhaul slice A, `starting_design`), and on the same page that the calculator wears its own look (slice B stage 1, `look`: the page's background is the theme's panel colour, read from the page, and the screenshot shows a charcoal page, the crimson square before the title, the crimson line under the selected tab and every text in one sans-serif face with no empty box; that sees the theme's marks in a browser, and it cannot tell Inter from egui's built-in font, which is one sans-serif face too: `the_app_draws_its_text_in_inter` holds the font); then it opens
`/tools/magcoupler/` through a pinned share link and checks the canvas, the Size and torque tab with its
```

with:

```markdown
share links included) on the next push of `main`; pushing needs the user's
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it first opens `/tools/magcoupler/` at the clean address and checks the status line "Starting design: passes all 9 checks", the header's "Open the example design" button and that the address stays clean (UX overhaul slice A, `starting_design`), and on the same page that the calculator wears its own look (slice B stage 1, `look`: the page's background is the theme's panel colour, read from the page, and the screenshot shows a charcoal page, the crimson square before the title, the crimson line under the selected tab and every text in one sans-serif face with no empty box, and since BL-107 that the page's loading line is gone (by its id, `LOADING_ID`), the title and the "Size and torque" heading heavier than the text under them and a grey edge on Auto's box and on the value boxes; that sees the theme's marks in a browser, and it cannot tell Inter from egui's built-in font, which is one sans-serif face too: `the_app_draws_its_text_in_inter` holds the font); then it opens
`/tools/magcoupler/` through a pinned share link and checks the canvas, the Size and torque tab with its
```

Replace:

```markdown
| `src/gui/plain_text.rs` (feature `gui`) | Every overlay key is an input or a result (a hint's an input or one of the Size tab's own hint keys), once, with text; no help row repeats the engine's; the nine canonical names of spec 3.5 and the gearbox ripple row's (BL-059: Nominal slip swing at the gearbox input), and no other row, the Size tab's gap and count names equal to the input rows'; `label_for` falls back to the metadata label and never to the path, and never lengthens a label; the help mechanism on a stand-in table (the overlay, else the engine; the original is the spreadsheet's own note, kept only where it is not empty: the deviation registry's workbook text over a help a correction reworded, else the engine's help) and the public accessors over every input's own engine help; every label, help and hint passes the plain-words guard and the glyph test, calls no torque guaranteed, holds no brace and keeps its caps (help <= 30 words in 1-2 sentences, hint <= 12 words in one); no display label is another input's or result's name; no overlay row, and no engine label or help shown where the overlay has none, says the coupling protects the gearbox (`no_label_help_or_hint_says_the_coupling_protects_the_gearbox`, BL-059). Every input has help from the engine or the overlay (`every_input_has_help_from_the_engine_or_the_overlay`); `overlay_strings_for_review` prints the overlay for the physics review. Help batch A (T7, `help_tests`): every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint groups has help; hints exist exactly for the Key design inputs and the three `SIZE_TAB_HINT_KEYS` (none of them an input path); batch A help and every hint carry no digit; no hint types an input label (catalogue or display); the review sheet lists every input of its groups and every hint; `help_for_review` (ignored) writes the G-3 sheet (`HELP_REVIEW_GROUPS`, `HELP_REVIEW_OUT`). Help batch B (T8, `batch_b_tests`): the 52 empty helps of the Calibration and model, Materials, Housing and retainers, and Thermal and demagnetization groups each have a row (the set is computed from `WORKFLOW`), and 31 non-empty helps are replaced (the grade-mode density, rating and coefficient helps, the stored 3D fields, placeholder and assumed values, and helps with a cell or correction id or a stale fact), none of them Rust-only; the toggle's original note is the spreadsheet's own text (the deviation registry's for a help a correction reworded, none where that is empty); 175 direction claims hold, each a result that rises, falls or stays when a batch-B input rises one slider step on a named design (the example, no back iron, a 1018 back iron, a 304 back iron, a titanium sleeve, a 7075 cap, screw classes 10.9 and A4-70, the coercivity inputs, a measured drag, manual magnets with and without Y30 grades, library N42 parts in both rings), and so do the outside-diameter (the boss or the cap sets it only as the largest part), cap-resize, cup-wall, measured-drag, magnet-mass and temperature-check claims; a help that names the shaft joint its input feeds has its claim among them, the way its words say (`batch_b_helps_name_the_shaft_joint_they_feed`: the boss length lets the cup's fit hold more, and the single-ring reverse field can lower the inner hub's allowed fitting temperature, which falls with library N42 parts in both rings, where the onset binds, and stays at the defaults); each stored-3D, prototype, unmeasured-default and adapter cluster is exactly the help rows that end with its caveat (each caveat is typed once, as a macro the rows are built from); the stored-3D cluster is the dashboard's `STORED_3D_INPUTS` less the Rust-only ones, and its caveat is the engine's sentence (`temperature::stored_3d_sentence!`), the one literal `STORED_3D_SENTENCE` and the engine's free-space helps read too; no help repeats its input's default; no overlay row covers a Rust-only input, whose help the engine words (R1).; the joint-neutral texts (ruling 11) claim what the engine does: a stronger manual block and longer magnets raise the pull-out, the cold-high torque and its gearbox-input value (longer magnets the overall length too), and a larger bore, both shafts' diameter (both joints' holding torques move with it), thins the hub wall under the magnets and past the keyway: its help opens with both shafts and names the hub check (by its label), which the bore's slider top fails on the 8-magnet probe design, keyed or keyless, with the wall past the keyway still standing, and that wall breaks through at the slider top on the smallest ring; the keyway's help names both bores and says it caps the key's contact height: one step deeper thins the hub wall past the key and, with the contact height under the keyway, leaves both joints' key pressures bit for bit, while on a keyway shallower than the contact height (1.0 mm, the "shallow keyway" design) it lowers both; the gearbox texts claim what the engine does (BL-059, `the_gearbox_texts_claim_what_the_engine_does`): the rating's help names the gearbox check and the cold-high torque at the gearbox input, the ripple row's label and help the slip swing, the pull-out at the gearbox output at the operating temperature without the production allowance, through the ratio and efficiency, the check row's help (the engine's) both sides of the gearbox and that fatigue from repeated slips is not checked, the gear ratio's help the slip swing as the ripple row does (one name on screen), and no text says peak, warm or no check (the rating's type is open; the operating temperature can sit below the minimum); at the five probe designs the swing is the nominal pull-out divided by the ratio and efficiency, and moves when the operating temperature rises one slider step but not when the minimum temperature or the variation does; and with the rating halfway between the swing and cold-high at the gearbox input the check follows cold-high, above the swing at the defaults and below it with the operating temperature under the minimum (-40 against 20 °C, variation 0.1) |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the calculator's theme and opens the starting design (its status line, no changed dot, the example design's button, the requirement and its margin on the readouts); the starting design passes every counted check with no caution and no material warning, pinned to `compute_all` with the numbers the README quotes, changes seven inputs and its fourteen stored 3D fields on their slider steps and no assumption, stores the 3D fields of a recorded run of the reference routine for its geometry, and draws no caution on its opening tab; a link and a design file saved from the first starting design (main 2ca65ee) open that design as it was; the example design opens as one undo step and Reset all returns the starting design; the spreadsheet export still flags against the workbook (the seven inputs and fourteen 3D fields at the starting design, none at the example design); the page address is clean at the starting design and the share link elsewhere, follows settled edits, undo and redo and waits for an edit in progress; a Ctrl+Z or Ctrl+Y held for 60 frames writes the address once, at the key's release, and a write the page refuses is asked again on the next frame (the tests' page records each write and can refuse one); a link opened at start-up stays the address until the first edit, and one that cannot be read changes nothing; a link cut at 2,000 characters is refused, changes nothing and the header says why; a link stays under 4,000 characters; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link, the smoke prompt names the header's Show spreadsheet details toggle (`SHOW_DETAILS`), and its R14 steps name `WELCOME`, `explorer::WHAT_TO_CHANGE`, `explorer::WHERE_FROM`, the Geometry tab and the Magnets fit row as the page draws them, and its card step quotes `torque_card::CARD_HEADING` (the script's own `CARD_HEADING`, typed once for the first view and the card step), `CAPTION`, `AUTO_HOLDS_LENGTH` and `held_line(Control::OverallLength)` whole and names the other three rows by their starts (slice 3, test 26); the frame timer `FRAME_TIMER` is one line that holds the WebGL draw-call test (`drawElements`, `if (window.__drew)`), and the card step installs it with `page.addInitScript` (plan review 4.1); the drag step finds the gap slider by the value its box shows, the script's `SMOKE_GAP_SHOWN` (the link's 1.5 mm at the slider's step's two decimals, with its unit, typed once), and counts the sizing lines by `SIZING_LOG_PREFIX` in its code and its words; each page's console messages are collected after its last step, since the Playwright MCP returns only those since the last navigation; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the tokens are the UX overhaul spec's table value for value, the visuals are that table field for field and egui's dark visuals in every other field, and every text token is at least 4.5:1 on every surface it is drawn on, with the spec's own figures to one decimal (`the_tokens_are_the_spec_s_table`, `the_visuals_are_the_token_table`, `every_text_token_reads_on_every_surface_it_is_drawn_on`); the selection fill is at least 3:1 against a field and the panel and 2.8:1 on a tooltip, the text colour on it is at least 4.5:1, no lighter fill could give a tooltip 3:1 and keep that, it is a blue, and no other text colour reads on it (`selected_text_shows_on_every_surface_and_reads_on_its_fill`); the app's font: Inter leads the proportional family and the built-in fonts follow, the ten digits have one advance width at seven sizes and are the cut's own, the cut holds printable ASCII and the other characters the app's texts use and not a character Inter lacks, a row of the installed font is as tall as the built-in font's at eight sizes, to egui's rounding (`a_row_is_as_tall_as_the_built_in_font_s`), a Cyrillic letter and an emoji, both outside the cut, have a glyph with the font installed and none with the cut alone (`a_character_outside_the_cut_is_drawn_by_a_built_in_font`), and the app installs it while a panel in a plain context keeps egui's fonts (`src/app/fonts.rs`; `the_app_draws_its_text_in_inter`); on the app's page, at the starting design and at the failing example design, the accent is on two shapes (the brand mark and the active tab's line), on no text and on no badge (`the_accent_marks_the_brand_and_the_active_tab_and_nothing_else`); on the app's page a dashboard row's title, the heads of the Why, Problem and Fix lines and the Size tab's readout labels are drawn in the muted token and their values in the text token, at the starting design and at the failing example design, and a tab that is not the active one is muted while the active tab's name is the text token (`a_label_is_muted_and_its_value_is_in_the_text_colour`); in the app's own theme and font, at 1280 x 800 and at 1024 x 700, no text of any tab (each plot through the Plots tab's picker, since BL-104) is cut sideways by the rect egui clips it to or by the window, a plot's rotated axis label included (`no_text_is_cut_in_any_tab_in_the_app_s_own_font`: the panel's layout tests draw in egui's built-in font, which is about 13 % narrower, so only this test sees a text that no longer fits), and the widest tooltip, the Verdict row's, stays inside the window (`the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font`); BL-099: at 1024 x 700, 1280 x 800 and 1920 x 1080 the left column is 320 points wide (egui's `PanelState`) and no text is cut, on any tab or on any row of the column, with the folds shut, each fold opened and shut again, every input group and its Advanced heading opened and shut again, a filter that lists every input set and cleared, an input traced, and Auto on with the Detailed inputs open (`the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font`); every input row, as it is, changed, changed and outside its slider's range, and locked by Auto, takes no more than the width it is given, at the three widths the column gives a row, in egui's default font and in the app's (`every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font`); every input row whose label wraps at the narrowest of those widths, changed, has as many Tab stops as at a width where nothing wraps, and its reset button sits close under the label, in either font (`an_input_label_that_wraps_is_one_tab_stop_in_either_font`); BL-098: at 1280 x 800 and 1024 x 700, with a note that has a diagram shown through Learn and its title, the right column is 300 points wide and no text is cut on any tab (`the_right_column_keeps_its_width_with_a_note_shown_in_the_app_s_own_font`); the hub page's card for the calculator does not name the paused clamp section (`the_hub_page_does_not_name_the_paused_clamp_section`); the smoke script's `PAGE_BACKGROUND`, typed once, is the theme's panel colour as a browser reports it, and its clean-page step judges the look, reads the page's background and names the title and the active tab; the theme stays dark when the system turns light |

```

with:

```markdown
| `src/gui/plain_text.rs` (feature `gui`) | Every overlay key is an input or a result (a hint's an input or one of the Size tab's own hint keys), once, with text; no help row repeats the engine's; the nine canonical names of spec 3.5 and the gearbox ripple row's (BL-059: Nominal slip swing at the gearbox input), and no other row, the Size tab's gap and count names equal to the input rows'; `label_for` falls back to the metadata label and never to the path, and never lengthens a label; the help mechanism on a stand-in table (the overlay, else the engine; the original is the spreadsheet's own note, kept only where it is not empty: the deviation registry's workbook text over a help a correction reworded, else the engine's help) and the public accessors over every input's own engine help; every label, help and hint passes the plain-words guard and the glyph test, calls no torque guaranteed, holds no brace and keeps its caps (help <= 30 words in 1-2 sentences, hint <= 12 words in one); no display label is another input's or result's name; no overlay row, and no engine label or help shown where the overlay has none, says the coupling protects the gearbox (`no_label_help_or_hint_says_the_coupling_protects_the_gearbox`, BL-059). Every input has help from the engine or the overlay (`every_input_has_help_from_the_engine_or_the_overlay`); `overlay_strings_for_review` prints the overlay for the physics review. Help batch A (T7, `help_tests`): every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint groups has help; hints exist exactly for the Key design inputs and the three `SIZE_TAB_HINT_KEYS` (none of them an input path); batch A help and every hint carry no digit; no hint types an input label (catalogue or display); the review sheet lists every input of its groups and every hint; `help_for_review` (ignored) writes the G-3 sheet (`HELP_REVIEW_GROUPS`, `HELP_REVIEW_OUT`). Help batch B (T8, `batch_b_tests`): the 52 empty helps of the Calibration and model, Materials, Housing and retainers, and Thermal and demagnetization groups each have a row (the set is computed from `WORKFLOW`), and 31 non-empty helps are replaced (the grade-mode density, rating and coefficient helps, the stored 3D fields, placeholder and assumed values, and helps with a cell or correction id or a stale fact), none of them Rust-only; the toggle's original note is the spreadsheet's own text (the deviation registry's for a help a correction reworded, none where that is empty); 175 direction claims hold, each a result that rises, falls or stays when a batch-B input rises one slider step on a named design (the example, no back iron, a 1018 back iron, a 304 back iron, a titanium sleeve, a 7075 cap, screw classes 10.9 and A4-70, the coercivity inputs, a measured drag, manual magnets with and without Y30 grades, library N42 parts in both rings), and so do the outside-diameter (the boss or the cap sets it only as the largest part), cap-resize, cup-wall, measured-drag, magnet-mass and temperature-check claims; a help that names the shaft joint its input feeds has its claim among them, the way its words say (`batch_b_helps_name_the_shaft_joint_they_feed`: the boss length lets the cup's fit hold more, and the single-ring reverse field can lower the inner hub's allowed fitting temperature, which falls with library N42 parts in both rings, where the onset binds, and stays at the defaults); each stored-3D, prototype, unmeasured-default and adapter cluster is exactly the help rows that end with its caveat (each caveat is typed once, as a macro the rows are built from); the stored-3D cluster is the dashboard's `STORED_3D_INPUTS` less the Rust-only ones, and its caveat is the engine's sentence (`temperature::stored_3d_sentence!`), the one literal `STORED_3D_SENTENCE` and the engine's free-space helps read too; no help repeats its input's default; no overlay row covers a Rust-only input, whose help the engine words (R1).; the joint-neutral texts (ruling 11) claim what the engine does: a stronger manual block and longer magnets raise the pull-out, the cold-high torque and its gearbox-input value (longer magnets the overall length too), and a larger bore, both shafts' diameter (both joints' holding torques move with it), thins the hub wall under the magnets and past the keyway: its help opens with both shafts and names the hub check (by its label), which the bore's slider top fails on the 8-magnet probe design, keyed or keyless, with the wall past the keyway still standing, and that wall breaks through at the slider top on the smallest ring; the keyway's help names both bores and says it caps the key's contact height: one step deeper thins the hub wall past the key and, with the contact height under the keyway, leaves both joints' key pressures bit for bit, while on a keyway shallower than the contact height (1.0 mm, the "shallow keyway" design) it lowers both; the gearbox texts claim what the engine does (BL-059, `the_gearbox_texts_claim_what_the_engine_does`): the rating's help names the gearbox check and the cold-high torque at the gearbox input, the ripple row's label and help the slip swing, the pull-out at the gearbox output at the operating temperature without the production allowance, through the ratio and efficiency, the check row's help (the engine's) both sides of the gearbox and that fatigue from repeated slips is not checked, the gear ratio's help the slip swing as the ripple row does (one name on screen), and no text says peak, warm or no check (the rating's type is open; the operating temperature can sit below the minimum); at the five probe designs the swing is the nominal pull-out divided by the ratio and efficiency, and moves when the operating temperature rises one slider step but not when the minimum temperature or the variation does; and with the rating halfway between the swing and cold-high at the gearbox input the check follows cold-high, above the swing at the defaults and below it with the operating temperature under the minimum (-40 against 20 °C, variation 0.1) |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the calculator's theme and opens the starting design (its status line, no changed dot, the example design's button, the requirement and its margin on the readouts); the starting design passes every counted check with no caution and no material warning, pinned to `compute_all` with the numbers the README quotes, changes seven inputs and its fourteen stored 3D fields on their slider steps and no assumption, stores the 3D fields of a recorded run of the reference routine for its geometry, and draws no caution on its opening tab; a link and a design file saved from the first starting design (main 2ca65ee) open that design as it was; the example design opens as one undo step and Reset all returns the starting design; the spreadsheet export still flags against the workbook (the seven inputs and fourteen 3D fields at the starting design, none at the example design); the page address is clean at the starting design and the share link elsewhere, follows settled edits, undo and redo and waits for an edit in progress; a Ctrl+Z or Ctrl+Y held for 60 frames writes the address once, at the key's release, and a write the page refuses is asked again on the next frame (the tests' page records each write and can refuse one); a link opened at start-up stays the address until the first edit, and one that cannot be read changes nothing; a link cut at 2,000 characters is refused, changes nothing and the header says why; a link stays under 4,000 characters; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link, the smoke prompt names the header's Show spreadsheet details toggle (`SHOW_DETAILS`), and its R14 steps name `WELCOME`, `explorer::WHAT_TO_CHANGE`, `explorer::WHERE_FROM`, the Geometry tab and the Magnets fit row as the page draws them, and its card step quotes `torque_card::CARD_HEADING` (the script's own `CARD_HEADING`, typed once for the first view and the card step), `CAPTION`, `AUTO_HOLDS_LENGTH` and `held_line(Control::OverallLength)` whole and names the other three rows by their starts (slice 3, test 26); the frame timer `FRAME_TIMER` is one line that holds the WebGL draw-call test (`drawElements`, `if (window.__drew)`), and the card step installs it with `page.addInitScript` (plan review 4.1); the drag step finds the gap slider by the value its box shows, the script's `SMOKE_GAP_SHOWN` (the link's 1.5 mm at the slider's step's two decimals, with its unit, typed once), and counts the sizing lines by `SIZING_LOG_PREFIX` in its code and its words; each page's console messages are collected after its last step, since the Playwright MCP returns only those since the last navigation; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the tokens are the UX overhaul spec's table value for value, the visuals are that table field for field and egui's dark visuals in every other field, and every text token is at least 4.5:1 on every surface it is drawn on, with the spec's own figures to one decimal (`the_tokens_are_the_spec_s_table`, `the_visuals_are_the_token_table`, `every_text_token_reads_on_every_surface_it_is_drawn_on`); the selection fill is at least 3:1 against a field and the panel and 2.8:1 on a tooltip, the text colour on it is at least 4.5:1, no lighter fill could give a tooltip 3:1 and keep that, it is a blue, and no other text colour reads on it (`selected_text_shows_on_every_surface_and_reads_on_its_fill`); the app's font: Inter leads the proportional family and the built-in fonts follow, the ten digits have one advance width at seven sizes and are the cut's own, the cut holds printable ASCII and the other characters the app's texts use and not a character Inter lacks, a row of the installed font is as tall as the built-in font's at eight sizes, to egui's rounding (`a_row_is_as_tall_as_the_built_in_font_s`), a Cyrillic letter and an emoji, both outside the cut, have a glyph with the font installed and none with the cut alone (`a_character_outside_the_cut_is_drawn_by_a_built_in_font`), and the app installs it while a panel in a plain context keeps egui's fonts (`src/app/fonts.rs`; `the_app_draws_its_text_in_inter`); on the app's page, at the starting design and at the failing example design, the accent is on two shapes (the brand mark and the active tab's line), on no text and on no badge (`the_accent_marks_the_brand_and_the_active_tab_and_nothing_else`); on the app's page a dashboard row's title, the heads of the Why, Problem and Fix lines and the Size tab's readout labels are drawn in the muted token and their values in the text token, at the starting design and at the failing example design, and a tab that is not the active one is muted while the active tab's name is the text token (`a_label_is_muted_and_its_value_is_in_the_text_colour`); in the app's own theme and font, at 1280 x 800 and at 1024 x 700, no text of any tab (each plot through the Plots tab's picker, since BL-104) is cut sideways by the rect egui clips it to or by the window, a plot's rotated axis label included (`no_text_is_cut_in_any_tab_in_the_app_s_own_font`: the panel's layout tests draw in egui's built-in font, which is about 13 % narrower, so only this test sees a text that no longer fits), and the widest tooltip, the Verdict row's, stays inside the window (`the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font`); BL-099: at 1024 x 700, 1280 x 800 and 1920 x 1080 the left column is 320 points wide (egui's `PanelState`) and no text is cut, on any tab or on any row of the column, with the folds shut, each fold opened and shut again, every input group and its Advanced heading opened and shut again, a filter that lists every input set and cleared, an input traced, and Auto on with the Detailed inputs open (`the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font`); every input row, as it is, changed, changed and outside its slider's range, and locked by Auto, takes no more than the width it is given, at the three widths the column gives a row, in egui's default font and in the app's (`every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font`); every input row whose label wraps at the narrowest of those widths, changed, has as many Tab stops as at a width where nothing wraps, and its reset button sits close under the label, in either font (`an_input_label_that_wraps_is_one_tab_stop_in_either_font`); BL-098: at 1280 x 800 and 1024 x 700, with a note that has a diagram shown through Learn and its title, the right column is 300 points wide and no text is cut on any tab (`the_right_column_keeps_its_width_with_a_note_shown_in_the_app_s_own_font`); the hub page's card for the calculator does not name the paused clamp section (`the_hub_page_does_not_name_the_paused_clamp_section`); the smoke script's `PAGE_BACKGROUND`, typed once, is the theme's panel colour as a browser reports it, and its clean-page step judges the look, reads the page's background and names the title and the active tab; the theme stays dark when the system turns light; BL-107: a control at rest shows its edge, by its fill or its outline, at 3:1 or more on the panel, a menu and a field, under the pointer and pressed a box keeps its fill while its outline brightens and a handle's ring stays the text colour, and an open picker keeps the resting outline (`a_control_at_rest_shows_its_edge_at_three_to_one_on_every_surface_it_sits_on`); with both folds of the left column open every rect and circle it draws, and the Design menu's checkbox, shows its edge at 3:1 or more against what is behind it (`every_control_shows_its_edge_on_the_panel_and_in_the_design_menu`); Auto's picker, opened, keeps an outline at 3:1 or more on the panel (`an_open_picker_keeps_its_edge`); each of the seven entries of Torque vs temperature's legend draws its dot on a disc in the resting fill and outline, 3:1 or more on the legend (`a_plot_legend_s_entries_show_their_dots_on_a_grey_disc`); the title and the left column's heading, and no other text, are Inter SemiBold, and the top bar stays one row with the heading on one line in the column, at 1024 x 700, 1280 x 800 and 1920 x 1080 (`the_title_and_the_size_heading_are_semibold_and_the_top_bar_keeps_one_row`); the SemiBold cut holds the Regular cut's characters, tabular digits and rows and is wider (`the_semibold_cut_has_the_regular_cut_s_characters_digits_and_rows`), and leads its own family with egui's Heading style in it (`inter_leads_the_proportional_family_and_the_built_in_fonts_follow`); the page names its language, the browser's colour and a dark scheme, shows `LOADING_LINE` after the canvas in the muted colour, and its script does not touch the line (`the_page_wears_the_app_s_colours_and_a_loading_line_until_the_app_draws`); the smoke script's `LOADING_ID` is the page's, and its look step reads the line gone and judges the heavier title and the controls' grey edge |

```

- [ ] **Step 2: Check.** `git diff --stat` lists the nine files and no other; `python -I -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ['docs/ai/backlog.yaml', 'docs/ai/02-system.yaml', 'docs/ai/03-structure.yaml', 'docs/ai/04-memory.yaml']]"` prints nothing; `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app 2>&1 | grep "test result"` shows every result `ok`, 1561 passed in all, as after Task 4 (the whole crate, since `tests/explain.rs` reads the README and `03-structure.yaml` for its counts).

- [ ] **Step 3: Commit** the nine files: `docs(magcoupler): slice B stage 2 part 2b-iii-b docs sweep: README, docs/ai, the register, owner questions 73 to 75, the design's header and 6.2, BL-107 in progress (BL-107 T5)`.

### Task 6: Release: the full gate, gui-smoke, the reviews, one fix wave, the merge and the push

**Model:** `fable` controller; fresh-context `fable` reviewers (while `fable` is out, back 2026-12-01, both on the session model, and the report says so); `haiku` at low effort for git; the smoke workflow's own agents on `sonnet`.

**Files:**
- Modify: `docs/ai/backlog.yaml` (Step 7); create `docs/analyses/2026-11-27-magcoupler-slice-b2p2biii-b-review.md` (Steps 4, 5 and 7).

- [ ] **Step 1: Preconditions.** Tasks 1 to 5 are committed, each reviewed; `git status --short` shows only the owner's untracked press analysis; `git diff main...HEAD --stat -- magcoupling-rs/src/engine magcoupling-rs/src/gui magcoupling-rs/tests reference linkage-sim-rs/src` is empty, and `git diff main...HEAD --stat -- linkage-sim-rs/web` lists `linkage-sim-rs/web/tools/magcoupler/index.html` alone; `sha256sum magcoupling-rs/assets/fonts/Inter-SemiBold.subset.ttf` is the hash Task 2 Step 1 checked.

- [ ] **Step 2: The full gate.** From PowerShell at the worktree's root:

  ```powershell
  & 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/gate.sh --full
  git checkout -- docs/chebyshev_lambda
  ```

  Expected: `GATE PASS`. A red gate goes back to the controller with the gate's output; nothing is merged.

- [ ] **Step 3: gui-smoke, from the branch's own script.** Build the web calculator from the branch (`linkage-sim-rs/scripts/build_magcoupling_web.sh` through Git Bash; note the size of `magcoupling-web_bg.wasm` against `main`'s, about 6.87 MB against 6.82), serve `linkage-sim-rs/web/`, and run the `gui-smoke` workflow from this branch's `.claude/workflows/gui-smoke.js` with `{"frame_check": true}`. Expected: `passed: true` for the calculator with `look: true` (the line gone, the heavier title and heading, the grey edges) and a p95 under one 60 fps frame. Keep its first screenshot.

- [ ] **Step 4: The text review, with the pictures** (fresh context; the design's 8.3). One new on-screen string: the page's loading line, "Loading the calculator". It checks that no other text is new, changed or gone (`git diff main...HEAD -- magcoupling-rs/src/gui magcoupling-rs/src/engine` is empty), and renders, in a browser from the build of Step 3: the page loading with the network held to about 1 MB/s (a screenshot while the line shows; the page charcoal, the line muted and centred), the load timed as "What the prototype showed" item 7 times it (the line's removal and the first draw in the same animation frame; no frame between them of another colour or blank), the first view at 1024 x 700, 1280 x 800 and 1920 x 1080, the Design menu open, Auto's picker open, a slider's handle reached by Tab, and the Plots tab on Torque vs temperature with its legend, each at 1280 x 800. It rules on three things and records what it decides: whether the grey edges read as a quiet frame and not as a grid of boxes, whether a selected choice with its grey edge still reads as selected, whether an open picker and a focused handle (its ring's width alone) read clearly, and whether the legend's dots on their grey discs, the muted Operating temperature dot most of all, still read as shown lines and not as hidden ones (decisions D-96 to D-98, owner question 73); whether the SemiBold title and heading set them apart enough with the tiles' numbers still Regular (decision D-99, owner question 74); and whether "Loading the calculator" is the right line, including what it says if the wasm file fails to load (decisions D-101 and D-103, owner question 75; block the wasm file's request to see it). A change the owner should weigh goes to `docs/analyses/2026-11-18-owner-questions.md` as an open question with its conservative answer, not shipped silently. Its findings join Step 5's fix wave. Record it in `docs/analyses/2026-11-27-magcoupler-slice-b2p2biii-b-review.md` (committed in Step 7).

- [ ] **Step 5: The independent whole-branch review** (fresh context, three lenses in parallel, each with a skeptic on every finding that is not a suggestion). Give each `git diff main...HEAD`, the design, this plan and Step 4's pictures, and none of the builders' notes.
  - **Is every control's edge there, and nothing else changed?** Every kind of control the app draws (in the left column, the centre's views, the plots' legends, the Equation panel, the Results table, the menus and the tooltips) at rest, under the pointer, pressed, focused (a slider's handle reached by Tab among them) and, for a picker, open; the frameless ones still frameless; `the_visuals_are_the_token_table`'s last check (every field outside the table is egui's) still holding.
  - **Is the face right, and does it fit?** The font file against the recorded hashes and the script's record; the two families' lists; the Heading style alone changed; the widths at the three sizes (render them) and the left and right columns' width tests; the glyph guards for the headings.
  - **The page and the code.** The head and the line against the design's colours; `remove_loading_line` on every frame (cost, a page with no line, a host page with an element of that id); a failed load; the smoke prompt's new words against what a browser shows.
  One fix wave follows, on the decider's model: each accepted finding gets its red test first. A finding that is not a defect of this part is filed in `docs/ai/backlog.yaml` with the next free id.

- [ ] **Step 6: If Step 4 or 5 changed a file, run Steps 2 and 3 again.** The merge needs a gate and a smoke run of the tree that is merged.

- [ ] **Step 7: Close the item and merge.** In `docs/ai/backlog.yaml` (the day's date; say what the reviews found):

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b2p2b-iii-b (Tasks 1 to 4 of the plan);
```

with:

```yaml
  status: fixed
  notes: "FIXED <date>: merged (<the gate, the reviews and the smoke run in a few words>). Built on branch magcoupling/overhaul-b2p2b-iii-b (Tasks 1 to 4 of the plan);
```

  Commit it with the review record: `docs(magcoupler): BL-107 fixed: the reviews, the gate and the smoke run passed`. Then merge the branch into `main` with a merge commit (`git merge --no-ff`), its message one line of what the screen now does.

- [ ] **Step 8: Push, then report** (the owner's ruling of 2026-11-22: each stage goes live once the gate, the smoke run and the reviews pass, with a report afterwards; no stop). Push `main` (it deploys the site). When the `deploy-web.yml` run is green, run the smoke workflow from `main`'s script with `{"url": "https://colesorkness.com"}` and report the result; if it fails, say so plainly: the site is live with the change. The report to the owner holds: what changed on screen, in three sentences; two pictures (the first view at 1280 x 800, the page while it loads); the decisions D-95 to D-105, one line each; owner questions 73 to 75; part 2b-iii-c with its items; which model ran the decider's steps; the size the face added to the wasm file; what was not verified (Safari and Firefox; the line on a slow phone; a failed load in the live site).

## What this plan covers of the design

| The design | Where |
|---|---|
| 6.1 The tokens, and G-7 (a control's edge at 3:1, the stage 1 review's E-4 and R-6) | Task 1 (one colour outside the table, `CONTROL`, register row A-52) |
| 6.2 Font: weights ("a second, SemiBold, for tile numbers and headings") and cost (about 0.1 MB a face) | Task 2 (the headings; the tile numbers stay Regular, D-99; 46.9 KB measured) |
| 6.2 Font: "the built file, its licence and the exact command that made it are committed; the app build does not run font tools" | Task 2 Step 1 (the script, its record and the hashes) |
| 6.3 Where the theme lives: the app sets the visuals and the fonts, the panel reads them | Tasks 1 and 2 (no file under `magcoupling-rs/src/gui/` changes) |
| 6.3 The web page's background is the panel colour (and the review's S-4: the page shell) | Task 3 |
| 8.2 test 15: the layout at 1024 x 700, 1280 x 800 and 1920 x 1080 | Task 2 (the top bar and the heading at the three sizes; the width tests in the new face) |
| 8.3 Text rules: the new string through the plain-words and gearbox guards | Task 3 |
| 9.1 The docs, the backlog item and the gui-smoke checklist with the slice | Tasks 4 and 5 |
| 9.4 Release | Task 6 |
| Not in this part | Part 2b-iii-c (the Equation panel: BL-086, BL-088, BL-089, BL-101's Equation panel part). |
