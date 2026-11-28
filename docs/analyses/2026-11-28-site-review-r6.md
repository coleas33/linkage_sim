# Site review R6 (2026-11-28): the live colesorkness.com/tools/ before the full release

Date: 2026-11-28. Reviewed: the live site at https://colesorkness.com/tools/ (the hub), /tools/linkage/ and /tools/magcoupler/, after R1 to R5 (main 1528105). The linkage follow-up R5b (BL-117, BL-118, BL-140) was not yet live.

Who reviewed: a fresh-context reviewer on the session model (Opus 5.5), standing in for fable while fable is out (back 2026-12-01), in Chromium through Playwright at 1024 x 700, 1280 x 800 and 1920 x 1080, as an engineer would use the tools.

## Pages

| Page | Size | Verdict | Notes |
|---|---|---|---|
| Hub /tools/ | 1024 x 700 | pass | Loads clean with no console messages. Dark page titled 'Engineering Tools' with two cards, 'Linkage Simulator' and 'Magnetic Coupling Calculator', each with a one-line scope and an 'Open the ...' link. Nothing is cut. |
| Hub /tools/ | 1280 x 800 | pass | Both card links work: /tools/linkage/ and /tools/magcoupler/ each opened their app. Reads as a tools page. Nit: the subtitle says 'Two calculators', but one of the two is a simulator. |
| Hub /tools/ | 1920 x 1080 | pass | Same layout, centred, nothing cut or overlapping. |
| Linkage simulator /tools/linkage/ | 1024 x 700 | fail | Loads dark with the welcome card (Load a Sample Mechanism, Start Tutorial, New Empty Mechanism, Watch Demo). After a reload the 'Recover Unsaved Work?' dialog sits over the card, which is acceptable. The sample loads and solves. The plot tab row runs out of width and 'Actuator Force' wraps one letter per line down the right edge (L-6). Dragging the plot divider up leaves a large empty band above the plot (L-6). The verdict is fail because of blocker L-1, which does not depend on window size: it was reproduced at 1920 x 1080 and not re-run at this size. The empty-glyph boxes (L-4) show here too. |
| Linkage simulator /tools/linkage/ | 1280 x 800 | fail | FourBar Play works: the button changes to Pause, the status bar reads PLAYING and the crank angle and forces update live. Plot tab label 'Actuat or Power' wraps. By default the plot area is a thin strip at the bottom, and after dragging the divider up the plot stays at the bottom under an empty band. Fail because of L-1, which does not depend on window size; the empty-glyph boxes (L-4) also show here. |
| Linkage simulator /tools/linkage/ | 1920 x 1080 | fail | FourBar (4-Bar Crank-Rocker), 6-Bar B1 (Watt I), Slider-Crank and Triple-Rocker (4-2-5-2) all load and solve. Play works, and the Driver Torque plot zooms with the wheel, pans and resets with a double-click. Add weight works. Mounting Angle rotates the mechanism, but at 100 deg most of it leaves the view (L-7). BLOCKER L-1: Set as Driver on J4 leaves the reaction panel and status bar showing old numbers and four different driver-angle readouts. A 2 mm drag of O2 after that gives 'Solver failed to converge' and FAIL. Ctrl+Z does nothing; only Edit > Undo works (L-2). Triple-Rocker shows 'Toggle points: None' next to 'Transmission Min: 0 deg (POOR)' (L-3). Console clean. |
| Calculator /tools/magcoupler/ | 1024 x 700 | pass (minor issues) | First view fits: top bar, the 4 tiles in a 2 x 2 grid, a small drawing, 9 checks over the Explain panel, 'Starting design: passes all 9 checks'. The failing share link, the Results table and the Geometry tab read correctly, with long lines wrapping instead of being cut. With the Equation panel open the drawing and plots shrink to about 200 px. The Torque vs temperature legend covers about a third of the plot, and the hover readout 'x = 96.1 / y = 1.854' (no units) spills over the Explain text (C-4, C-5). |
| Calculator /tools/magcoupler/ | 1280 x 800 | pass (minor issues) | Every listed step was done here. Checked: the first view (top bar Undo, Redo, Design, Share, Export, Learn; 4 tiles; drawing with numbered dimension tags; check list; Explain), a check entry, the gap slider drag (the checks and the drawing update live and the address follows), Auto (sets the magnet length and greys Overall length with 'Set by Auto; turn Auto off to edit'), the Start here walk 1 to 14 (Next greys at 14, Stop works), the smoke share link (fails 4 of 9), and all four tabs. Plots checked: Torque vs temperature, Gap sweep, Pole sweep, Slip heating, Torque vs rotation. Also checked: the Results table (916 results), Detailed inputs, Assumptions, the Tab order (top bar then sliders), Share ('Share link copied to the clipboard (2687 characters)'), Export (downloads magcoupling-results.xlsx), Load design (file overlay, 'Design file loaded', checks recomputed) and Ctrl+Z. Minor issues C-1 to C-9. |
| Calculator /tools/magcoupler/ | 1920 x 1080 | pass (minor issues) | The first view uses the width well, with the tiles in one row and a large drawing. The Start here walk runs to step 14 and the long T_hold,hub piecewise equation lays out cleanly. Plots and the failing design read well. The plot summary line is still in a very small font (C-3). Console clean. |

## Console

Clean on all three pages at all sizes: 0 errors and 0 warnings for the whole session. The hub logged nothing. The linkage simulator logged only eframe/egui DEBUG and INFO startup lines (WebGL2 selected, shader version, 'event handlers installed'). The calculator logged only INFO lines: 'magcoupling explorer: 421 equations', 'loaded the design from the share link (sizing: Torque -> Magnets)', 'magcoupling sizing: Solved at 11.94 mm (hot-low torque 1.350 N.m)' and 'Solved at 14.18 mm (hot-low torque 2.500 N.m)', and 'magcoupling: loaded a design file'. Logs: C:\Users\Cole\source\repos\linkage_simulation\.playwright-mcp\console-2026-11-28T17-53-35-419Z.log (linkage) and console-2026-11-28T18-05-37-578Z.log (calculator).

## Findings

### L-1 (blocker, blocks the release)

Where: Linkage simulator: right-click a joint > Set as Driver (the Driver section says 'J1 (right-click to change)'), Joint Reactions panel, status bar, then a ground-pivot drag

Reassigning the driver to the rocker leaves wrong numbers on screen, and the next ordinary edit fails to solve. Right after Set as Driver on J4, the Joint Reactions panel still shows the crank-driven values with 'Reactions verified': Driver Torque 0.1034 N.m, J1 15 N (0,-15), J2 5 N (0,-5), J3 4 N (0,4), J4 14 N (0,14). After an Edit > Undo re-solve, the same pose reads Driver Torque -0.2894 N.m, J1 10 N, J2 10 N, J3 17 N, J4 26 N. The driver angle has four readings: the slider says 180 deg, the canvas label at J4 says 627.1 deg, the rocker link label says 87.1 deg and the status bar says theta 0.0 deg. Then dragging the ground pivot O2 down about 2 mm gives a red 'Solver failed to converge - mechanism may be over-constrained or at a toggle point'. The status bar reads FAIL, the crank is drawn detached from J1, and Joint Reactions says 'No reaction data (need driver)' while the Driver section shows J4. The same O2 move with J1 as the driver solves (Driver Torque 0.0962 N.m), and the pose is reachable (O2 to J3 is about 48.4 mm, within the 30 to 50 mm coupler-plus-crank range). The 180 deg slider value is not reachable for this rocker.

Repro: Open https://colesorkness.com/tools/linkage/ at 1920 x 1080 and click Load a Sample Mechanism (4-Bar Crank-Rocker). Right-click J4 and choose Set as Driver, then read Joint Reactions, the Crank Angle slider, the J4 label and the status bar. Drag O2 (J1) down about 30 px, so Ground Link reads Angle 3.013 deg. Screenshots: r6-site-review/linkage-1920x1080-driver-set-J4-repro.png and linkage-1920x1080-driverJ4-drag-ground-pivot.png.

### L-2 (major)

Where: Linkage simulator: keyboard undo

Ctrl+Z does nothing, although the Edit menu shows 'Undo Ctrl+Z' and Edit > Undo and the Undo History buttons both work. A listener on the page showed the keydown arriving at the canvas with ctrlKey=true, and the state did not change. In a clean state and in the FAIL state of L-1 alike, only the menu recovers. Ctrl+Z works in the calculator.

Repro: Load the 4-Bar Crank-Rocker sample, drag O2 down so it reads Y -2 mm, click the empty canvas and press Ctrl+Z: O2 stays at Y -2 mm. Edit > Undo returns it to Y 0 mm.

### L-3 (major)

Where: Linkage simulator: Properties > Mechanism Health

'Toggle points: None' shows in green next to 'Transmission Min: 0 deg (POOR)' in red. A triple rocker cannot turn fully and must pass toggle positions, and the crank trail is drawn as two arcs, so 'None' is wrong. The same contradiction appears on the FourBar once the rocker is the driver.

Repro: Samples > Triple-Rocker (4-2-5-2): Mechanism Health reads 'Grashof: Non-Grashof', 'Toggle points: None', 'Transmission Min: 0 deg (POOR)'. See r6-site-review/linkage-1920x1080-triple-rocker.png.

### L-4 (major)

Where: Linkage simulator: status bar, Mechanism Health, Link Editor, Ground Pivots, Joint Reactions

Symbols draw as empty boxes (a missing-glyph box) at every size: status bar '4-Bar Crank-Rocker (box) (box) (box) theta 0.0 deg tau 0.103 N.m (box) 3B 4J DOF=0', 'Transmission (box): Min 44 deg', 'Ground Link: O2 (box) O4', 'Pos: (10.000, 0.000) mm (box) -48.5 deg', 'B(box)C' and 'B(box)C angle' beside the link-length sliders, and '(box) Reactions verified'. To an engineer the page looks broken, and the length slider labels cannot be read.

Repro: Load any sample and look at the bottom status bar and the Link Editor rows beside the 40 mm and 0 deg sliders.

### C-1 (major)

Where: Calculator: orange 'Design check:' line under the drawing and under the Geometry list

'Design check: Cap thread nominal diameter 41.00 mm is below Steel cup body OD 41.54 mm' reports a real geometric conflict (the thread is smaller than the body it must go over). It is not one of the nine checks, the headline count ('fails 4 of 9', 'fails 3 of 9') ignores it, and nothing says whether it is a failure or what to change. It appeared on the gui-smoke share link, after dragging the face gap to 4 mm, and on the 2 mm design file. 'Steel cup body OD' is also capitalised mid-sentence.

Repro: Open the share link from .claude/workflows/gui-smoke.js (MAGCOUPLING_SMOKE_PAYLOAD) at 1280 x 800 and read the line below the drawing. See r6-site-review/magcoupler-1280x800-failing-design.png.

### L-5 (minor)

Where: Linkage simulator: Driver-change and FAIL states, the Driver Torque plot

After Edit > Undo out of the L-1 FAIL state, the Driver Torque plot x axis runs from about 180 to 540 deg ('Driver Angle (deg)'). It shows only two curve fragments, near 185 deg and past 505 deg, so the rocker's real range is hard to read.

Repro: Follow L-1, then Edit > Undo once, and open the Driver Torque plot tab.

### L-6 (minor)

Where: Linkage simulator: bottom plot panel

The plot panel opens as a thin strip, about 60 px at 1920 x 1080. Its legend is cut off ('crank.A' clipped), and Coupler Trace opens with axes of about +/-2000 mm for a 40 mm mechanism, so the trace is a dot. The tab row wraps 'Actuat or Power' at 1280 x 800, and at 1024 x 700 'Actuator Force' wraps one letter per line. When the divider is dragged up, the plot stays pinned to the bottom with a 200 px empty band above it.

Repro: Load the 4-Bar Crank-Rocker sample at 1280 x 800 and 1024 x 700, then drag the plot divider up. See r6-site-review/linkage-1280x800-plotpanel-resized.png and linkage-1024x700-fourbar.png.

### L-7 (minor)

Where: Linkage simulator: Mounting Angle

Setting the mounting angle to 100 deg rotates the grid and the mechanism, but the view does not refit: the coupler and rocker leave the visible canvas and only the crank circle stays in view. Typing 0 brings it back.

Repro: 4-Bar Crank-Rocker, open Mounting Angle and drag the slider to 100 deg. See r6-site-review/linkage-1920x1080-mounting-angle-changed.png.

### C-2 (minor)

Where: Calculator: Results table, status column

Status texts are cut with an ellipsis at 1024 and 1280 widths: 'Estimate covers ...', 'Too thin: raise M...', 'OK on temperatu...', 'Both shaft joints ...', 'Above the gearb...', 'Exceeds the spa...'. The reason for a failing check has to be read elsewhere.

Repro: Open the failing share link and the Results table tab, then look at the Headline group.

### C-3 (minor)

Where: Calculator: Plots tab, summary line above each plot

The summary line ('Pull-out torque (nominal): 2.941 N.m  Hot-low torque: 2.500 N.m  Cold-high torque ...  Governing temperature limit: 93.06 deg C', 'Thermal time constant: 291.9 s ...') uses a font of about 8 px, much smaller than the rest of the UI. It is hard to read at every size.

Repro: Open Plots and select any plot.

### C-4 (minor)

Where: Calculator: Plots at 1024 x 700

The Torque vs temperature legend (7 rows) covers about a third of the plot area and hides the left part of the High line. With the Equation panel open the plot is about 200 px tall and the legend covers most of it.

Repro: At 1024 x 700 click Start here, then the Plots tab. See r6-site-review/magcoupler-1024x700-plots-tab.png.

### C-5 (minor)

Where: Calculator: plot hover readout

The hover readout 'x = 96.1 / y = 1.854' has no units, and at 1024 x 700 it spills past the plot edge over the Explain panel text.

Repro: At 1024 x 700, Plots > Torque vs temperature, hover near the right edge of the plot. See r6-site-review/magcoupler-1024x700-plots-tab-panel-closed.png.

### C-6 (minor)

Where: Calculator: Gap sweep and Pole sweep plots

The Gap sweep x axis is 'Corner gap [mm]', while the only gap control is 'Magnet face gap'. The legend category 'Nominal: test needed' is unclear. On the Pole sweep the 'This design' marker (about 2.94 N.m at 10 magnets) sits off the sweep curve (about 2.86 N.m at 10). The caption ('each row at the smallest ring radius ...') explains this, but it will still trip a reader.

Repro: On the failing share link, Plots > Gap sweep, then Pole sweep.

### C-7 (minor)

Where: Calculator: Explain panel for a size input, 'What a change does'

The list is one dense run-on paragraph with ASCII arrows, for example 'Magnet face gap: -0.1 mm -> +0.02 N.m (+2.6 %), Outside diameter -0.2 mm, Space claim: Rotating diameter -0.22 mm; +0.1 mm -> -0.02 N.m (-2.5 %), ...'. It is hard to scan.

Repro: Drag the Magnet face gap slider and read the Explain panel.

### C-8 (minor)

Where: Calculator: left column 'Minimum OD'

'Minimum OD' looks like an input slider but moves when another input changes: dragging the face gap from 1.80 to 2.90 mm moved both Minimum OD and Outside diameter from 42.8 to 44.0 mm. It is not clear whether it is an input, a result, or what sets it.

Repro: Clean address, drag Magnet face gap right and watch Minimum OD.

### C-9 (minor)

Where: Calculator: slider keyboard focus

When Tab reaches a slider, the only sign is a slightly larger knob. Top-bar buttons get a clear outline. Keyboard users can lose their place in the left column.

Repro: Click the Size and torque tab, then press Tab 6 to 12 times.

### N-1 (nit)

Where: Calculator wording and precision

'Fixed 3D fields: starting or example design' under the checks is cryptic to a new user. Clearances show 7 significant figures ('-0.003174 mm'). Auto shows 'margin +0.00 N.m (+0.0 %)'. The T_hold,hub equation has an 'undefined' branch. The Load design overlay is a plain white browser dialog on the dark app.

Repro: Look at the first view, the failing design's Geometry tab and Start here step 14.

### N-2 (nit)

Where: Linkage simulator: number formatting and labels

An empty mechanism shows '-0.00 kg'. Reactions show negative zero ('J1: 15 N (-0, -15)'). 'Constraints: OK (res=0.0e0)' is solver jargon. Dimension labels overlap on the 6-Bar Watt I sample ('2500.0 mm 56.1 deg' over 'rocker4 2500.0 mm 97.2 deg').

Repro: New Empty Mechanism; FourBar with O2 at Y -2 mm; Samples > 6-Bar B1 (Watt I).

## Summary

The hub and the magnetic coupling calculator are ready for engineers. The linkage simulator has one blocker.

Hub: it loads, both links work and it reads as a tools page at all three sizes.

Calculator: it passes at 1024 x 700, 1280 x 800 and 1920 x 1080. Every listed step worked:
- the first view, slider drag, Auto, and the 14-step Start here walk to its end;
- the failing smoke share link (fails 4 of 9 checks);
- all four tabs, including all five plots;
- Detailed inputs, Assumptions and Tab focus;
- Share (link copied), Export (an xlsx file downloaded), Load design (the file loaded and the checks recomputed) and Ctrl+Z.

The calculator's one major issue is C-1: an orange 'Design check' line about the cap thread being smaller than the cup body. It is not one of the nine checks and the headline count ignores it. The rest is polish:
- statuses cut short in the Results table;
- a tiny font in the plot summary line;
- at 1024 x 700, the legend covers part of the plot and the hover readout has no units and spills onto the Explain panel;
- unclear wording in the gap and pole sweep plots;
- dense 'What a change does' text;
- 'Minimum OD' moves when other inputs change;
- weak keyboard focus on sliders.

Linkage simulator: it loads dark, and all four samples (FourBar, 6-Bar Watt I, Slider-Crank, Triple-Rocker) load and solve. Play, plot zoom, pan and reset, and Add weight work.
- **Blocker (L-1):** after Set as Driver on the rocker J4, the screen keeps the crank-driven reaction forces and torque under 'Reactions verified', and the driver angle has four readings (180, 627.1, 87.1 and 0.0 deg). The next 2 mm drag of the ground pivot O2 fails to solve: the status bar reads FAIL, the crank is drawn detached from J1, and the panel says 'need driver' although a driver is set. The same drag solves with the crank as driver.
- **Major:** Ctrl+Z does nothing, although the Edit menu offers it; only Edit > Undo works (L-2).
- **Major:** 'Toggle points: None' shows next to 'Transmission Min: 0 deg (POOR)' on the Triple-Rocker (L-3).
- **Major:** symbols draw as empty boxes all through the status bar and side panel (L-4).
- Minor: the plot panel opens too small and its tab labels wrap, and at a 100 deg mounting angle the mechanism leaves the view.

Both apps' consoles were clean: no errors or warnings all session.

What I did not cover:
- L-1 and L-2 were reproduced only at 1920 x 1080; they do not depend on window size.
- On the linkage simulator I did not check the export, Image and Tools menus, the Slider-Crank and six-bar beyond loading, or the tutorial and demo.
- I did not check frame timing beyond watching the screen.

Files I created:
- C:\Users\Cole\source\repos\linkage_simulation\.playwright-mcp\r6\gap-2mm.magcoupling.json, the design file fed to Load design. The .playwright-mcp folder is git-ignored and I edited no tracked file.
- C:\Users\Cole\source\repos\linkage_simulation\.playwright-mcp\magcoupling-results.xlsx, saved there by the Export download.

All screenshots are in C:\Users\Cole\AppData\Local\Temp\claude\69e43908be0ec656\c368a0e9-0dad-4f12-ad62-643ce3a7f894\scratchpad\r6-site-review. The browser is closed.
