# Magcoupler R3 polish batch: browser review

Date: 2026-11-28

Reviewer: a fresh-context reviewer on the session model (Opus 5.5), standing in for fable while fable is out (back 2026-12-01). The reviewer read the backlog entries of the batch's items and section 8.3 of the UX overhaul design, and no builder or reviewer notes beyond them.

Branch: `magcoupling/r3-polish` in the `lsim-b2` worktree, tip `fc97b87` (18 commits over main `fb7557c`). Items: BL-095, BL-078, BL-102, BL-103, BL-105, BL-106, BL-110, BL-111, BL-115, BL-116, BL-119, BL-120, BL-121, BL-122, BL-123.

## How and what was rendered

- The branch's release build served at `http://localhost:8080/tools/magcoupler/` (its JS and wasm are dated 2026-11-28 16:27 UTC, after the tip commit at 16:19 UTC). main from the live site, `https://colesorkness.com/tools/magcoupler/`, for comparison.
- Chromium through the Playwright tools, at 1024 x 700 and 1280 x 800 CSS pixels and a device pixel ratio of 1. The clean address opens the starting design; the share link in `.claude/workflows/gui-smoke.js` opens the failing example design. Clicks went in by coordinates, with a wait of about a second after each.
- BL-103 was also looked at with the device pixel ratio emulated at 0.9 and 1.35 (Chrome DevTools Protocol). At 1.35 the page laid out at about 948 x 593 points.
- BL-111: a `requestAnimationFrame` hook copied the canvas after each frame while the page loaded, so the first painted frame could be looked at.
- BL-102: Tab was pressed round the whole page, with a screenshot after each press. Each frame was diffed against the one before to find the new focus mark, and contact sheets were made of the marks. Two walks were made: 1024 x 700 with the folds shut, and 1280 x 800 with Detailed inputs open and the Equation panel open on Start here.
- BL-121 and BL-122: the walk was clicked through all 14 steps at both sizes. A script found the Next button from the screen's pixels at each step.
- Console: no errors or warnings over the whole session.
- The screenshots are in the session scratchpad, `r3-visual-review` (85 files, named for what they show).

## Items

| Item | Where seen | What was seen | Verdict |
|---|---|---|---|
| BL-095 | Example design, 1024 x 700. Size and torque tab, the red Overall length tag (5) | The tag's entry (0.2813 mm, "Large-diameter stack beyond its bay") shows "Checks it touches" with a "Space claim Fails" link. A click on the link selects the Space claim check and shows its entry. Only the bay overshoot was reached in the browser; the diameter and length overshoots were not. | Pass |
| BL-078 | Starting design, 1024 x 700. Detailed inputs filtered to "keyway" | At a keyway depth of 1.45 mm (a wall of 2.50 mm) the branch draws no line, and main draws "Wall past keyway: 2.50 mm; existing guidance is about 2.5 mm." At 1.50 mm (2.45 mm) the branch draws "Wall past keyway: 2.45 mm; existing guidance is about 2.5 mm." in amber over the drawing. The slider's 0.05 mm step turns 1.46 into 1.45, so the 2.49 mm case of the red test could not be typed in. | Pass |
| BL-102 | 1024 x 700 with the folds shut. 1280 x 800 with Detailed inputs open and the Equation panel on Start here | At 1024 x 700 the cycle has 44 stops: Design, Share, Export, Learn, the size controls, the folds, the 9 checks, Start here, the 4 tabs, the 4 tiles and the 6 dimension tags. At 1280 x 800 it has 101 stops. Every press drew a focus mark: an outline, a bigger slider handle, an underline on a link or label, or a bigger fold arrow. No press landed on a scroll area, a scroll bar, a column edge or the formula. The order Close, Start here, then the crumbs is as recorded. Shift+Tab and the Plots tab were not walked (BL-124 covers the plot axes). | Pass (F-5: the folds' mark is faint, as before) |
| BL-103 | Detailed inputs, Key design and Magnets and rings groups, 1024 x 700 and 1280 x 800 | Each "Pick a part" and "Pick a grade" box starts right of its label (label x 34 to 98 or 109, box from x 104 or 115 to 320), and no text is drawn over another. The column's divider stays at x 327 with the folds shut and open (a 320-point column). At emulated ratios 0.9 and 1.35, with Calibration and model, then Advanced, open, the box "0.00000125664 T·m/A" ends inside the column and the column holds. | Pass |
| BL-105 | Plots tab, all five plots. Starting design and example design, 1024 x 700 and 1280 x 800 | Starting design at 1024: Torque vs temperature reads 0 50 100, Gap sweep 1 2 3 4, Pole sweep 6 8 10 12 14 16, Slip heating 0 500 1000, Torque vs rotation 0 20 40 60 80. At 1280: -40 to 100 by 20, 0.5 to 4 by 0.5, 6 to 16 by 2, 0 to 1200 by 200, 0 to 80 by 20. Every axis is evenly stepped, with no '14 15 16' or '40 50 60' break. Example design at 1280: all fine (0 to 1400 by 200 on Slip heating). Example design at 1024: Slip heating reads "0 500 1000 150". The 1500 mark's number is cut at the plot's right edge; main draws "0 1000" there, with nothing cut. | **Fail: F-1** |
| BL-106 | Example design, Slip heating and Pole sweep, 1024 x 700 and 1280 x 800 (and the starting design) | Both legends sit at the bottom right. The time-to-the-limit dot (770.4 s, about 93 °C) and every Pole sweep row are clear of them at both sizes. On main at 1024 the top-right legend covers the dot and the 16-magnet row. | Pass (nits F-3, F-4) |
| BL-110 | Tiles, and the Slip heating legend | The tiles' grey edges look as they do on main. A hidden legend entry (Estimate) draws a ringed disc. A host without the app's theme (`Visuals::dark()`) cannot be reached in the browser. | Pass, as far as it can be seen |
| BL-111 | Load of the clean address, 1024 x 700 | In the first painted frame the four tiles already share one height, and the second frame is the same. | Pass |
| BL-115 | Equation panel, 1024 x 700. Trail T_hot,low > T_op > T_pull > T_2D > σ; also a narrow centre with the right column widened 130 points | The crumbs wrap onto rows under the heading. None is drawn under Start here or Close at 2, 3, 4 or 5 crumbs. A click on the T_op crumb goes back to T_op. In the narrow centre the header row is clean too. | Pass |
| BL-116 | Filed case: 1024 x 700, Start here, pointer at (1006, 500) and (1008, 690), then off. Compared with main | Branch: the note's lines do not re-wrap and none is cut, with the pointer on the bar or off it, and the heading link is whole after the pointer leaves. While the pointer is on the bar, the bar (about x 997 to 1007) covers the last letter of the heading link "About this tool". main: with the pointer on the bar the body re-wraps wider and is cut ("cylinder a", "T_2D", "R_g:"), and the heading reads "About this to". | Fixed as filed (residual F-2; better than main) |
| BL-119 | 1280 x 800, Tab down the open Detailed inputs | The value boxes that take the focus at the column's foot (Operating magnet temperature, Minimum outer return ring wall, Thermal conductance) show whole, with their bottom edges in view. | Pass |
| BL-120 | 1024 x 700, right column dragged 130 points wider, V_temp opened from the Temperature limits entry | "Closest teaching note:" is followed by the title on its own rows, "Demagnetization: permanent loss / from heat and slipping", broken only between words. | Pass |
| BL-121 | Start here, all 14 steps, 1024 x 700 and 1280 x 800 | On every step, Previous, Next and Stop sit together on one row under the step line, including steps 9, 12 and 13, which split before. 1920 x 1080 was not checked. | Pass |
| BL-122 | T_2D wheeled down, then its σ term opened; Next through all 14 steps at both sizes | σ opens at its top, with its title in view. Each step opened by Next shows its title at the top of the panel. | Pass |
| BL-123 | Hot minimum check (C_hot) at 1024 x 700, a formula wider than the panel | With the pointer on the formula's sideways bar, the widened bar sits in its own strip under "Estimate covers hot min" and covers no glyph. | Pass (F-6 is a different bar) |

## On-screen strings added or changed

The text came from `git diff main...HEAD -- magcoupling-rs/src`, outside the tests. The branch adds no new words: no string constant was added or reworded. So nothing new goes through the plain-words guard or a physics review. No "clamp" and no ring name by colour appear.

| String | Where | Change | Verdict |
|---|---|---|---|
| The horizontal axis numbers of the five plots (for example "0.5", "1.5", "-40", "1200") | Plots tab | New steps (1, 2 or 5 times a power of ten, each step a multiple of the finer one), with the decimals of the mark's step and no trailing zeros | True to the code, plain and generic. One number is cut to a wrong value (F-1) |
| "Checks it touches" and the "Space claim" link with its level | Entries of `housing.diameter_overshoot_mm`, `housing.length_overshoot_mm` and `housing.bay_overshoot_mm` (and the dimension tags that share them) | Existing strings, now shown on these three entries | True: the overshoot is the space-claim check's failing measure. Plain, in the page's own words, generic. OK |
| "Wall past keyway: {wall} mm; existing guidance is about 2.5 mm." | Size and torque tab, over the drawing | Words unchanged. It is no longer drawn when the wall prints as 2.50 | True to the code: compared as printed. OK. The typed-in "2.5 mm" is main's guidance figure, unchanged here |
| "Equation panel", the crumbs, ">", the ellipsis with "{n} earlier steps" | Equation panel header | Moved into the room left of the buttons; text unchanged | OK |
| "Start here", "Close" | Equation panel header | Laid out first, at the right; text unchanged | OK |
| "Previous", "Next", "Stop" | Start here walk bar | Their own row; text unchanged | OK |
| "Closest teaching note:" and the note's title | Equation panel | The title starts its own row when the room left is short; text unchanged | OK |
| "Set by Auto; turn Auto off to edit" | Hover on a size slider that Auto has locked | Now drawn through `add_enabled_ui`; seen still showing on the example design | OK |
| The legend entries of Pole sweep and Slip heating | Plots tab | Same words, now at the bottom right, listed in reverse order | OK (nit F-3) |
| `request_discard("tile row height changed")` | Not on screen (egui's debug reason) | New | Not applicable |

## Findings

**F-1. Blocks.** BL-105: on the example design at 1024 x 700, Slip heating's horizontal axis reads "0 500 1000 150". The number of the 1500 mark is drawn centred on its tick near the plot's right edge and is clipped there, so the last number reads 150 where the value is 1500. Screenshots: `branch-1024-bl105-example-slip-heating-axis-1500-cut-to-150.png` and `branch-bl105-example-design-every-x-axis-1024-and-1280.png`. main draws "0 1000" there, with nothing cut. The item's acceptance asks for no number cut, and the branch's axis test covers the starting design only. This blocks because the page shows a wrong number that main does not show. Fix suggestion: leave out a horizontal number whose text would pass the plot's frame (keep the tick and drop the label), and add the example design to the axis test.

**F-2. Does not block.** BL-116 residual: while the pointer is on the Explain column's scroll bar, the widened bar covers the last letter of the heading link "About this tool". The heading sits right-aligned to about x 998, and the bar spans about x 997 to 1007. The note's body wraps clear of the bar. It is far better than main, where the body re-wraps and is cut. Screenshots: `branch-1024-bl116-heading-zoom.png` and `branch-1024-bl116-bar-zoom.png`. Suggested: lay the heading row out in the same width as the body, or file it.

**F-3. Nit.** The bottom-right legends list their entries in reverse order. Slip heating: main lists "Estimate, Governing limit, High case, Time to the limit (high case)", and the branch lists "Time to the limit (high case), High case, Governing limit, Estimate". Pole sweep is reversed the same way. Nothing is wrong, but the order now differs between plots. The order could be kept, or accepted as is.

**F-4. Nit.** Pole sweep, starting design, 1024 x 700: the dashed Required floor line runs under the legend from about x 546 to 703. It is a reference line, not a marked point, and main's top-right legend covered the 16-magnet point instead.

**F-5. Observation, before this branch.** The focus mark on a fold header (Detailed inputs, Assumptions, the input groups, "Where this comes from") is only a slightly larger arrow. This is egui's default for a collapsing header, and the branch does not change it (not checked on main). It still shows, but it is faint. It could be filed.

**F-6. Observation, outside the items.** While the pointer is in the Equation panel, its vertical floating scroll bar covers the right end of a formula that reaches the panel's edge (the closing quote of "Estimate covers hot min" on the hot minimum check). BL-116's notes say only the Explain panel's scroll area was changed. Not checked on main. It could be filed.

Out of scope and not judged: the Torque vs temperature high-case curve runs under that plot's top-right legend (a line, not a marked point); and "Vacuum permeability" is listed twice under Advanced (T·m/A and H/m).

## Summary

14 of the 15 items read as their acceptance asks in the browser. BL-105 fails on the example design at 1024 x 700, where Slip heating's last axis number reads "150" for 1500 (F-1, blocks). BL-116 is fixed as filed and is far better than main, with a one-letter residual under the hovered bar (F-2). There are two nits (F-3, F-4) and two observations from before the branch (F-5, F-6). No on-screen words were added or reworded.

## Fix wave

- **F-1, fixed.** The "example design" here is the one the gui-smoke share link opens. Red first: `every_plot_s_horizontal_axis_carries_three_evenly_spaced_numbers_in_the_app_s_own_font` (app.rs) now also draws the example design, the example 5 degrees hotter and the share link's design at 1024 x 700 and 1280 x 800, and asserts that no horizontal number's text passes its plot's frame. On the old code it failed on the share link's design at 1024 x 700 ("0 500 1000 1500", the 1500 cut by 9 points) and on the example and hotter example at 1280 x 800 ("1400" 1 point past the frame). The test's cut check parsed the whole `"1500" by 9 points` line and so could never fail; it now reads the quoted text. The fix is in `gui/plots.rs` (`plot`): a horizontal number whose text would pass the frame on either side is left out, and its tick stays. The grid spacer hands the formatter the axis' points per unit each frame, and the formatter measures the number in egui_plot's font. Every design and window in the test keeps at least three evenly stepped numbers; the share link's Slip heating now reads "0 500 1000" at 1024 x 700.
- **F-2, fixed.** The heading row was already laid out in the body's width. The right-aligned link ended flush with the strip kept for the bar (its text 0.2 of a point into it, natively), so the hovered bar, 10 points wide from the strip's edge, covered its last letter. `explain_ui` now puts egui's margin between contents and bar (`scroll.bar_inner_margin`, 4 points) right of the link. Test: `the_explain_heading_s_link_ends_left_of_the_scroll_bar_under_the_pointer` (panel.rs) puts the pointer on the bar and checks that the link's text ends left of the bar's widest rect. It failed first: 998.2 against 998.0.
- **F-3, F-5, F-6, filed** as BL-125, BL-126 and BL-127. **F-4** (a nit: a reference line under a legend) is not filed.
- No on-screen string was added or changed. Not re-checked in the browser: the web build was not rebuilt.
