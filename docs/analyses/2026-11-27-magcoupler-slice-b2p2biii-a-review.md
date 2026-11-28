# Magcoupler UX overhaul, slice B, part 2b-iii-a (the Plots tab): text review (BL-104, plan Task 6 Step 4)

Date: 2026-11-27. Reviewed: branch `magcoupling/overhaul-b2p2b-iii` at 5d85c43 against `main`.
Plan: `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii.md` (Task 6 Step 4; decisions D-86 to D-94). Design: `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, decisions C-5 and C-9, sections 4.4 and 8.3. Owner questions 70, 71 and 72 in `docs/analyses/2026-11-18-owner-questions.md`.

**Who reviewed.** A reviewer with a fresh context. It ran on the session model (Opus 5.5), standing in for `fable` while `fable` is out (back 2026-12-01), as the plan's Task 6 allows. It read the plan, the design and the owner questions named above, and the branch's diff. It did not read the builders' or the earlier reviewers' notes.

**How.** The branch's web build, served at `http://localhost:8080/tools/magcoupler/`, was opened in Chromium through Playwright at 1280 x 800 and at 1024 x 700, at a pixel ratio of 1, in the app's own theme. Two designs were used: the passing starting design (the clean address) and the example design, which fails 4 of 9 checks (the share link in `.claude/workflows/gui-smoke.js`). On each design, at each size, the review took a picture of the opening tab row, then clicked the Plots tab and each of the five names in the picker. `main`'s live build (`https://colesorkness.com/tools/magcoupler/`) was opened for comparison: its tab row at both sizes, its Gap sweep and Torque vs temperature at 1280 x 800, and its Slip heating on the example design at 1024 x 700. The review also checked that, on the branch, the Plots tab opens on the plot shown last (Slip heating, then Geometry, then Plots). The branch's console showed no error.

Pictures (the agent's scratchpad, not in the repository): `b2p2biii-text-review/`, 35 files. They include the Plots tab with the picker at 1280 x 800 (`02-passing-1280x800-plots-tab-picker-torque-vs-temperature.png`) and a sweep on the passing design at 1280 x 800 (`04-passing-1280x800-gap-sweep.png`).

## The texts

| Check | Result | Evidence |
|---|---|---|
| One new on-screen string, "Plots" (`plots::PLOTS`) | Holds | The only string literal the branch adds in `magcoupling-rs/src/gui/` outside the tests is `pub const PLOTS: &str = "Plots";`. It is the design's own word (C-9) and plain. |
| Every other text of the tab row and the picker is a text `main` drew | Holds | The tab row reads Size and torque, Geometry, Plots, Results table. The picker reads Torque vs temperature, Gap sweep, Pole sweep, Slip heating, Torque vs rotation. These are the strings and the order of `main`'s eight tabs, which take two rows at 1280 x 800 and three at 1024 x 700 (pictures 20 and 50). Each plot's readout line, legend names and axis labels are the same as on `main` (pictures 21 and 22 against 04 and 02). |
| No text left the screen | Holds | At both sizes and on both designs, every tab, picker name, readout line, legend name and axis label is drawn whole. The tab row is one row at both sizes; at 1024 x 700 it ends near x 665. The picker is one row at 1280 x 800 and two at 1024 x 700 (three names, then two). |
| No engine text changed | Holds | `git diff main...HEAD -- magcoupling-rs/src/engine` is empty (0 lines). |

## Rulings

### 1. "Results table" as the fourth tab, where the design writes "Results" (D-89, owner question 71)

**Ruling: keep "Results table". No change.** The words are accurate: the view under the tab is the results table and nothing else (design 4.4: "Results. The results table, unchanged"). Results are also shown in the tiles, the check list and every plot's readout, so the bare word "Results" says less about what the tab holds. "Results table" is the text the tab has had since it was made, and the scope does not need it changed. The row Size and torque, Geometry, Plots, Results table fits on one row at 1024 x 700 with room to spare, so the longer word costs no space. Owner question 71 stays open with its conservative default, "Results table".

### 2. A row of five names reading as a picker under a tab, and not as a second tab row, at 1024 x 700 where it wraps (D-87, owner question 70)

**Ruling: it reads as a picker. No change.** The two rows use different marks and sit in different places:
- The tab row is muted text. The active tab is bright, with the crimson underline, and a hairline rule runs under the row.
- The picker sits below that rule, inside the view. It marks the plot shown with the filled blue selection fill. That is the same widget as the inputs' order toggle and the results table's order.

At 1024 x 700 the picker wraps to three names and then two (pictures 31 and 32). The second row has the same plain names and the same fill when a name on it is picked (pictures 35 and 36), so it reads as part of the same group and not as a third row of tabs. The readout line follows at once in small text, which closes the group.

One weak point, a matter of taste: a name not picked has no frame at rest and is in the bright text colour, while a tab not picked is muted. The picker's names are therefore brighter than the tabs above them. They still read as choices, because one of them is always filled.

A drop-down would hide the five names behind a click. Owner question 70 stays open with its conservative default, the row of names.

### 3. With the red gone, a requirement line in the strong text colour and a limit in violet still reading as a requirement and a limit beside their legend names, on both designs (D-90, owner question 72)

**Ruling: they read as a requirement and a limit, on both designs. No change.**
- **Dashed means reference.** Every reference line is dashed: the operating temperature, the governing limit, the required torque and the required floor. Every data series is solid or dots. The line keeps its meaning without the red.
- **Each line is named twice.** The legend names it ("Required torque", "Required floor", "Governing limit"), and the readout line above the plot states its value ("Required floor: larger of traction need and service minimum: 1.350 N·m", "Governing temperature limit: 94.94 °C").
- **The requirement line is the brightest mark on the plot.** Strong text `#e6e8eb` on the plot background `#121417` is 15.0:1, against 5.7:1 for the failing red it replaced, so it is not lost. In the torque-temperature plot its legend dot is set apart from the operating temperature's muted grey dot (2.6:1 between the two) and from every other colour there.
- **The limit is the same violet in both plots that draw it** (6.9:1 on the plot background). It is the violet the torque-temperature plot already gave its limit line, so a limit looks the same in Torque vs temperature and in Slip heating (picture 16 shows the dashed limit and its legend dot in violet).
- **The example design draws the lines exactly as the passing design does** (pictures 11 to 15 and 41 to 45 against 02 to 07 and 31 to 36). That is correct: on `main` the lines were red on both designs, so their colour never carried the verdict, and the check list still does. On the example design no plot line stands for a failing check: Hot-low torque passes at a margin of +0.00 N·m and Temperature limits passes. So option (c) of question 72 ("red only while the check fails") would draw no red here either.

The cost D-90 names stands: a requirement line no longer looks like a warning. That is what C-5 asks for. Owner question 72 stays open with its conservative default, as built.

### 4. The shared legend dots

**Ruling: keep both pairs. No change.** egui_plot draws every legend entry as a filled dot in its colour.

- **The sweeps: "Required floor" and "This design" share the bright dot** (pictures 04, 05, 12, 13, 33, 34, 42, 43). On the plot one is a dashed horizontal line and the other an open ring sitting on the curve, two shapes that cannot be confused. The names differ, and the plot draws no other bright mark. On `main` the floor shared the red dot with "Does not fit", which tied a requirement line to a failing row and was the worse pairing (picture 21).

  The plan's smallest change, the ring in `BLUE`, would make it worse in the Gap sweep. "This design" would then share the blue dot with "Pull-out torque (nominal)", whose curve is also `BLUE`. The ring would also be a blue ring on the blue curve it sits on (blue against the strong text colour is only 1.8:1, and blue on blue is no contrast at all), where the bright ring now stands out. The change trades one shared dot for another and costs the ring its visibility.
- **Slip heating: "Governing limit" and "Time to the limit (high case)" share the violet dot** (pictures 14, 16, 44). The dot sits on the limit line and marks where the high case meets that limit, so the shared colour ties together two marks of one quantity. They shared the red dot on `main` (picture 51).

Question 72 already describes both pairs and the blue-ring option, so nothing new goes to the owner.

## Findings

| # | Finding | Where | Blocks | What to do |
|---|---|---|---|---|
| F-1 | On the example design at 1024 x 700, Slip heating's legend (right top) covers the "Time to the limit (high case)" dot. The dot and the high case curve show faintly through the legend's fill (picture 44). At 1280 x 800 the curve also runs under the legend's "Estimate" row (picture 14). `main` draws the same at 1024 x 700 (picture 51): the branch's one tab row plus the picker's two rows leave the plot the height `main`'s three tab rows left it. The value is still on screen in the readout line ("time to the limit (high case): 770.4 s"). Not a defect of this part. | `magcoupling-rs/src/gui/plots.rs` (`plot`: `Legend::default().position(Corner::RightTop)` for every plot) | No | File in `docs/ai/backlog.yaml` with the next free id, for a later part: a legend corner per plot that its data leave free (for Slip heating, whose curves rise to the right, a bottom corner), with a test that no plot's legend covers a marked point. |
| F-2 | The horizontal axes are sparse: Pole sweep shows only "10" and Torque vs rotation only "0" on the passing design at both sizes, and Slip heating shows "0" and "1000". `main` draws the same. | `magcoupling-rs/src/gui/plots.rs` (`X_TICK_SPACING`, decision D-91) | No | Already filed as BL-105 (open). Nothing new. |

Nothing in this review needs a change before release, and nothing new goes to the owner question list.
