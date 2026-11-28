# Magcoupler UX overhaul, slice B, part 2b-iii-c (the Equation panel): text review (BL-112, plan Task 7 Step 4)

Date: 2026-11-28. Reviewed: branch `magcoupling/overhaul-b2p2b-iii-c` at e099b26 against `main` (aa1949a).
Plan: `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-c-equation-panel.md` (Task 7 Step 4; decisions D-106 to D-115). Design: `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, sections 5.5, 8.2 (test 15) and 8.3. Owner questions 67 and 76 in `docs/analyses/2026-11-18-owner-questions.md`. Register rows A-44, A-55 and A-56 in `docs/analyses/2026-11-10-conservative-assumptions.md`. Backlog items BL-086, BL-088, BL-089, BL-101 and BL-112 in `docs/ai/backlog.yaml`.

**Who reviewed.** A reviewer with a fresh context. It ran on the session model (Opus 5.5), standing in for `fable` while `fable` is out (back 2026-12-01), as the plan's Task 7 allows. It read the plan's Task 7 Step 4 and its decisions, the design, the owner questions, the register rows and the backlog items named above, and the branch's diff. It did not read the builders' or the earlier reviewers' notes.

**What changed after the plan was written.** The branch took two review fix waves (c14e80f, 72098b6 and 79d6fd2, with their docs commits). A term's long name and its tag now wrap inside their grid cell, and a wrapped name is one Tab stop (`gui::label_block`). The closest note's link is one Tab stop however its title wraps. A Results table line takes the focus only while whole. `main` (aa1949a, the linkage batch) was merged in (e099b26). This review judged the panel as it now ships.

**How.** The branch's web build (`magcoupling-web_bg.wasm`, 6,886,494 bytes, built 2026-11-28 at 04:55) was served at `http://localhost:8080/tools/magcoupler/`. The served file is byte for byte the one in the worktree. The commits after the last code commit (79d6fd2) touch no file under `magcoupling-rs/`. It was opened in Chromium through Playwright at a pixel ratio of 1, in the app's own theme. Two designs were used: the passing starting design (the clean address) and the example design, which fails 4 of 9 checks (the share link in `.claude/workflows/gui-smoke.js`). Rendered:
- the Start here walk, all 14 steps, at 1024 x 700, 1280 x 800 and 1920 x 1080. Each step's Equation panel was shown at its top, then wheeled down to its foot (2 to 6 pictures a step);
- at 1024 x 700:
  - the high case's rotations to the limit (`N_lim,hi`), opened from its Results table line's entry;
  - the temperature verdict (`V_temp`);
  - the hot minimum check (`C_hot`): with the pointer on its formula (scroll bar shown), scrolled sideways to its end, and with the pointer away;
  - Tab and Shift+Tab through the Equation panel, on `C_hot` and on Start here step 14 (`T_hold,hub`, whose formula is taller than the panel's view);
  - the Results table, as it opens, and with Torque open and wheeled 307 points;
  - the example design's first view, and its Gearbox input torque check's Equation link;
- at 1024 x 450, a height at which the check list scrolls: Tab through the check list;
- `main`'s live build (`https://colesorkness.com/tools/magcoupler/`) at 1024 x 700, for comparison: the first view, and the hot minimum check's Equation panel with and without the pointer on the formula.

Where the keyboard's focus sat was found by diffing each picture against the one before. The branch's console showed no error and no warning.

The strings were checked by a script. It compared, as multisets, every string literal in the non-test code of `magcoupling-rs/src` at `main` and at HEAD. Test modules, `gui/test_support.rs` (compiled only for tests) and comments were left out. It also compared the literals on the diff's added and removed lines, so a moved literal cancels out. A second script compared the lines that left `explorer.rs` with the lines that arrived in `notes_view.rs` and `test_support.rs`.

Pictures (the agent's scratchpad, not in the repository): `b2p2biiic-text-review/`, 397 files. They include a Start here step at 1024 x 700 with wrapped term rows (`start-here-1024x700.png`, step 11) and the hot minimum check's formula at 1024 x 700 with its scroll bar under the pointer (`hot-min-formula-1024x700.png`).

## The texts

| Check | Result | Evidence |
|---|---|---|
| No on-screen string is added or changed; one is gone, `EMPTY_TEXT` | Holds | In the non-test code, only two literals are at `main` and not at HEAD. One is `EMPTY_TEXT`'s sentence. The other is `"magcoupling_equation_terms_scroll"`, the id of the terms' sideways scroll area that went: an egui id, never drawn. No literal is at HEAD and not at `main`. Every other literal added or removed on the diff's lines is in a test (assert messages, equation paths, the test designs' names) or in a comment. |
| The note drawing moved word for word (BL-088) | Holds | Between the lines that left `explorer.rs` and those that arrived in `notes_view.rs`, only these differ: the module comment, split between the two modules; the imports; `note_text_on` made `pub(crate)`; `NoteLink`'s two doc links, which now name the Equation panel's items by path; and the test module's header. The seven note strings and every drawn literal arrived unchanged. What else left `explorer.rs` was changed in place there: `EMPTY_TEXT`, its draw and its test's lines, the step line, the formula's area and the term grid. |
| No engine or linkage text changed | Holds | `git diff main...HEAD` is empty for `magcoupling-rs/src/engine` and for `linkage-sim-rs`. It is also empty for `magcoupling-rs/src/app`, `magcoupling-rs/tests`, `reference` and `.claude`. |
| Every text on screen is a text `main` drew | Holds | The hot minimum check's panel has the same words as on `main`'s live build (`main-live-hot-min-equation-1024x700.png` against `hot-min-formula-1024x700.png`). On `main`, the term tags are cut at the panel's edge ("click to open its", "input: click to f"). On the branch they wrap. Across the walk, the step lines, term names, tags and values are `main`'s words (the strings check); only where they wrap changed. |
| No text of the panel is cut but the formula | Holds | On every step, at the three sizes, every term name, tag and value and the step line is whole inside the panel. The panel's right edge stays inside the centre region (x 708 of 716 at 1024 wide). The formula runs past the edge on steps 6, 10, 12, 13 and 14 at 1024 x 700, and on 6, 13 and 14 at 1280 x 800. At 1920 x 1080 it never does. This is as the plan says. The long text values wrap in their own column: "never: steady state stays below the limit" takes three rows on `N_lim,hi` (`rotations-high-long-value-1024x700.png`), and "n/a (coercivity rises as the magnet cools)" takes three on `V_temp`. The closest note's title wraps under its lead on both. |
| Seen, filed before, not of this part | Noted | The header row's Start here button is drawn over a long breadcrumb on the Shaft joints and Gearbox input torque values (BL-115). In one picture the pointer, resting on the Explain column's scroll bar, re-wrapped an open note past the column (BL-116). Three of the Equation panel's Tab stops show no focus mark (BL-102, with KR-2's formula). |

## Rulings

### 1. Do the wrapped term rows still read as a list? (D-110)

**Ruling: yes. Keep them as they ship. No change.**

- **The columns hold.** On every row the swatch, the symbol and the value sit in their own columns, and the rows are striped. Down any list the eye finds the next term by its swatch and symbol (`start-here-1024x700.png`, `w1024-step14-d.png` to `-f.png`).
- **Name and tag stay apart.** The name is in the body colour and the tag in the muted grey, or amber for an assumption, so the two read apart even where they share a line.
- **Most rows are one line from 1280 wide.** At 1280 x 800 most rows are one line; only a few long names or tags take a second row. At 1920 x 1080 every row is one line.
- **What wrapping costs at 1024 x 700.** This is the cost D-110 names.
  - A row takes two to four lines.
  - The tags no longer line up in a column. A tag follows a short name on its line ("End-effect factor click to open its equation"). After a wrapped name it starts a line of its own (`gui::label_block`).
  - A tag can break mid-phrase ("Magnets per ring input:" / "click to find its row").
  - Step 14's list runs several views deep.
  - None of this hides a word. On `main` the same tags ran past the edge and were cut.
- **No owner question.** D-110 has no owner question, and none is added. Wrapping is the only way to keep the term rows inside the panel (BL-086's acceptance), and the rows read as a list at every size tried.

### 2. Should the formula scroll sideways, or be set smaller? (D-109, owner question 76)

**Ruling: keep it at its 22 points, scrolling sideways. No change in the code. Owner question 76 stays open with its default, (a).**

- **Why it scrolls.** An equation does not wrap. Set smaller to fit the 359-point view at 1024 x 700, the widest walk step's equation would fall below the 8 points the tooltips' equations stop at. Scrolling keeps one size for every equation. `main` scrolls the formula the same way, with the same bar.
- **The browser's widths.** Measured at 1920 x 1080, where each formula is whole, the ink of each walk step's formula spans about:

  | Step | Width | Size to fit at 1024 x 700 |
  |---|---|---|
  | 13, the cold check | 1,203 points | 6.6 points |
  | 6, the calibration factor | 1,009 points | 7.8 points |
  | 14, the shaft joint | 880 points | 9.0 points |
  | 12, the demagnetization onset | 538 points | 14.7 points |
  | 10, the slip loss | 489 points | 16.2 points |

  The hot minimum check's formula would fit at about 16 points.
  - The plan, A-55 and question 76 give the widest as about 1,040 points, needing 7.6 points. Both figures are under the 8-point floor, so the ruling does not change.
- **Option (b) has more merit than question 76 tells the owner.** With an 8-point floor, three of the five steps (10, 12 and 14) would show their equation whole at 9 to 16 points. Only two (6 and 13) would still scroll.
- **A cut that does not show.** At 1024 x 700 the hot minimum check's equation is cut between its outcomes and its conditions. With the pointer away, it reads as a whole brace of two quoted outcomes, and nothing shows that "if T_hot,low < T_req" and "otherwise" are out of view (`w1024-hot-min-formula-scrolled-to-end.png` shows them). `main` draws the same.
- **The scroll bar covers part of the formula.** With the pointer on the formula, egui's floating scroll bar is drawn over the lower half of its last row (`hot-min-formula-1024x700.png`). `main` draws the same (`main-live-hot-min-formula-pointer-1024x700.png`).
- **Not clearly wrong.** The formula's value line under it ("C_hot = Estimate covers hot min") is whole. The bar shows under the pointer, and the side columns can be dragged narrower. Whether the owner wants (b) is a matter of judgement, so it goes to question 76 with its conservative answer (finding F-2).

### 3. Should a Results table line half seen take the focus after all? (D-113, owner question 67)

**Ruling: no. Keep D-113 as it ships. Owner question 67 stays open with its default, (a).**

- **The table does not scroll for the focus.** That is question 67's default, so a line half seen would keep its focus outline cut by the view's edge. BL-101 removes exactly that.
- **As the table opens, at 1024 x 700.** Shift+Tab from the page's end:
  - one stop with no mark (BL-102);
  - the three group headings that the view shows whole (Demagnetization, Temperature, Torque);
  - each Headline line, from "End-effect model check" up.
- **Wheeled 307 points with Torque open.**
  - The walk starts at the last whole line ("Harmonic 3 geometry factor with bac…", y 665 to 683).
  - It climbs to "Verdict", the first whole line under the view's top (y 195 to 213).
  - Then it leaves for Export XLSX. The line cut by the view's top ("Margin above the hot-day start") took no focus.
- **The cost is already recorded.** A focused line that a wheel then cuts loses the focus, and the next Tab starts at the page's first stop. Question 67 and A-44 say so (KR-4).
- **The group headings' focus mark is faint.** A focused group heading shows its focus only by a brighter triangle. That is egui's look and `main`'s, not this part's.

## Findings

| # | Finding | Where | Blocks | What to do |
|---|---|---|---|---|
| F-1 | The walk bar's three buttons can split across rows. At 1024 x 700, step 9 puts Previous on the step line's row and Next and Stop on the next. Step 13 puts Previous and Next on the line's second row and Stop on a third. At 1280 x 800, step 12 puts Stop alone on the next row. D-111 says the buttons follow the line "on its last row or the next". Each button is whole and works. | `magcoupling-rs/src/gui/explorer.rs`, the walk bar: the step line and the three buttons are separate items of one `ui.horizontal_wrapped` | No | In the fix wave, draw Previous, Next and Stop in one `ui.horizontal` inside the wrapping line, so they wrap as one block. Add an assert to the app's walk test that the three share one row on every step at the three sizes. No text changes. If the fix wave declines it, word D-111 to what ships. |
| F-2 | Question 76 undersells its option (b). It gives only the widest equation's size (7.6 points). In the browser that equation measures about 1,203 points, so it needs 6.6. At 9 to 16 points, (b) would show three of the five long walk equations whole at 1024 x 700 (steps 10, 12 and 14), and the hot minimum check's at about 16. It also does not say that where the cut falls between two parts, nothing shows the equation goes on: the hot minimum check reads as complete without its conditions. | `docs/analyses/2026-11-18-owner-questions.md` question 76; A-55; D-109's figure | No | In the fix wave, add these facts to question 76 and A-55, docs only. Recheck the 1,040-point figure in the app's font against the browser's 1,203. The default stays (a). |
| F-3 | The Equation panel keeps its scroll offset when the open equation changes, so the new equation can open with its title and the top of its formula above the view. Seen on the walk's Next (step 2 opened at step 1's offset) and on an Equation link (`N_lim,hi` opened at `V_temp`'s offset). The panel's area has one id (`magcoupling_equation_scroll`) for every equation. Since this part, the term rows wrap at 1024 x 700, so the panel scrolls, and keeps an offset, more often. Not checked on `main`. | `magcoupling-rs/src/gui/explorer.rs`, the panel's `ScrollArea::vertical().id_salt("magcoupling_equation_scroll")` | No | File in `docs/ai/backlog.yaml` as BL-121 (the next free id) for a later part. The panel scrolls to its top when the open equation changes (Next, Previous, a term followed, an Equation link). Add a panel test: open an equation, wheel down, open another, and its title is in view. |
| F-4 | With the pointer on the formula, egui's floating sideways scroll bar is drawn over the lower half of the formula's last row ("Estimate covers hot min" on the hot minimum check). `main` draws the same. | `gui::sideways_scroll_area` (the formula's area); egui's floating scroll bar style | No | File for a later part with the next free id after F-3's. Give the formula's area room under its content for the bar, or a solid bar, so the bar never covers the formula. Add a test that the bar's rect does not meet the formula's glyphs. |
| F-5 | The code's comment (`sideways_scroll_area`) and the plan's Review Focus item 1 say that a formula wider than the view "shows from its start" when it takes the focus. That holds only for a formula not scrolled sideways. With the hot minimum check's formula scrolled to its end, Tab through the panel left it there (`w1024-kb-hotmin-tab-00.png` to `-09.png`). As egui's `scroll_area.rs` reads, its least scroll moves a target wider than the view by no more than the room left on the target's other side, so a formula scrolled to its end stays there. On step 14 the formula, taller than its view, was shown from its lower rows. The formula has no focus mark and no keyboard action (KR-2, BL-102), so a keyboard user loses nothing. | `magcoupling-rs/src/gui/mod.rs`, the comment on `sideways_scroll_area`; plan Review Focus item 1 | No | For the whole-branch review's keyboard lens: confirm in the harness. Then word the comment to what egui does (a widget wider than the view shows from its start only if it was not scrolled sideways), or leave it with the note. No behaviour change in this part. |

Nothing in this review needs a change before release. F-1 is a small layout fix for the fix wave, or a rewording of D-111. F-2 adds facts to owner question 76 and A-55. F-3 and F-4 are filed for a later part. F-5 goes to the keyboard lens.
