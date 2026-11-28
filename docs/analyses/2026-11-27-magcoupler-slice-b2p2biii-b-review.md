# Magcoupler UX overhaul, slice B, part 2b-iii-b (the look): text review (BL-107, plan Task 6 Step 4)

Date: 2026-11-27. Reviewed: branch `magcoupling/overhaul-b2p2b-iii-b` at f50047a against `main` (f84bc4f).
Plan: `docs/superpowers/plans/2026-11-27-magcoupler-overhaul-b2-part2b-iii-b-the-look.md` (Task 6 Step 4; decisions D-95 to D-105). Design: `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, sections 6 and 8.3. Owner questions 73, 74 and 75 in `docs/analyses/2026-11-18-owner-questions.md`. Backlog items BL-107, BL-108 and BL-109 in `docs/ai/backlog.yaml`.

**Who reviewed.** A reviewer with a fresh context. It ran on the session model (Opus 5.5), standing in for `fable` while `fable` is out (back 2026-12-01), as the plan's Task 6 allows. It read the plan, the design, the owner questions and the backlog items named above, and the branch's diff. It did not read the builders' or the earlier reviewers' notes.

**What changed after the plan was written.** BL-108 and BL-109 were fixed on this branch (8559faa, f50047a). A plot legend's dot now sits on a dark disc (the field colour) inside the grey ring, not on a grey disc, and the four tiles over the drawing take the controls' grey edge. So `magcoupling-rs/src/gui/` now holds a diff, in `plots.rs` and `tiles.rs` only. This review judged the legend and the tiles as they now ship.

**How.** The branch's web build (`magcoupling-web_bg.wasm`, 6,874,977 bytes, built with BL-108 and BL-109's fixes), served at `http://localhost:8080/tools/magcoupler/`, was opened in Chromium through Playwright at a pixel ratio of 1, in the app's own theme. Two designs were used: the passing starting design (the clean address) and the example design, which fails 4 of 9 checks (the share link in `.claude/workflows/gui-smoke.js`). Rendered:
- the page loading, with the network held to 1 MB/s by CDP (`Network.emulateNetworkConditions`, 20 ms latency, cache off);
- the load timed in the page, once held and three times at full speed;
- a CDP screencast of the held load;
- a failed load (the `.wasm` request aborted with `page.route`);
- the first view at 1024 x 700, 1280 x 800 and 1920 x 1080;
- at 1280 x 800: the Design menu open; Auto's picker open with the pointer moved off it; the Tab key walked to the Required torque slider's handle; a value box under the pointer; the Hot-low tile selected; the Plots tab on Torque vs temperature with its legend, with Operating temperature hidden by a click and shown again by a second; and on the example design, its first view, Torque vs temperature and the Gap sweep (the red "Does not fit" dot in the legend).

Colours were read from the pictures' pixels, and contrast computed by the WCAG formula. `main`'s live build (`https://colesorkness.com/tools/magcoupler/`) was opened for comparison: its first view at 1280 x 800, its failed load and its first frames at load. The branch's console showed no error or warning, apart from the deliberately failed load.

Pictures (the agent's scratchpad, not in the repository): `b2p2biiib-text-review/`, 55 files. They include the first view at 1280 x 800 (`first-view-1280x800.png`) and the page while it loads (`loading-line.png`).

## The texts

| Check | Result | Evidence |
|---|---|---|
| One new on-screen string, "Loading the calculator" (`app::LOADING_LINE`) | Holds | Outside the tests, `app.rs`, `theme.rs` and `fonts.rs` add three string literals. Only `LOADING_LINE` reaches the screen. The other two are the line's id `magcoupling_loading` and the font family's name `Inter SemiBold`, and neither is drawn. The page carries the same words (`<p id="magcoupling_loading">Loading the calculator</p>`). The words are plain, and Task 3's test runs them through the plain-words guard and the gearbox guard. |
| `git diff main...HEAD -- magcoupling-rs/src/gui` holds only `plots.rs` and `tiles.rs`, with no string added, changed or removed | Holds | `plots.rs` adds 13 lines in `plot_ui`: a comment, and the loop that gives the inactive, hovered and active fills the field colour (BL-108). `tiles.rs` changes the resting outline to `widgets.inactive.bg_stroke`'s colour at one width, and updates its comment (BL-109). No added or removed line holds a string literal. |
| No engine text changed | Holds | `git diff main...HEAD -- magcoupling-rs/src/engine` is empty (0 lines). The same diff is empty for `magcoupling-rs/tests`, `reference` and `linkage-sim-rs/src`. Under `linkage-sim-rs/web` it lists `tools/magcoupler/index.html` alone. |
| Every other text on screen is a text `main` drew | Holds | The first view at 1280 x 800 has the same words in the same places as `main`'s (pictures `first-view-1280x800.png` and `main-live-first-view-1280x800.png`). The top bar from "Undo" on sits about 4 points further right, because the SemiBold title is wider. The Design menu, Auto's picker, the Plots tab, the legends and the example design read as on `main`. |
| No text left the screen | Holds | The top bar is one row, and "Size and torque" one line in the 320-point column, at 1024 x 700, 1280 x 800 and 1920 x 1080. |

## Rulings

### 1. The grey edges (D-96 to D-98, owner question 73)

**Ruling: keep the grey edge as it ships. No change.**

- **A quiet frame, not a grid of boxes.** Each edge is one pixel of `#656c76`, 3.26:1 on the panel. The text inside and beside it is 15:1 and the labels 5.4:1, so apart from the hairlines the edges are the faintest marks on the screen. In the first view the left column outlines six value boxes, a picker and a checkbox, and fills six rails and handles. They line up in one column and read as the inputs' fields. On `main` (1.21:1) the same boxes were close to invisible (pictures `zoom-left-column-and-tiles-1280x800.png` and `main-live-zoom-left-column-and-tiles-1280x800.png`).
- **The tiles with their new edge.** The four tiles take the same edge (3.26:1, measured at the first tile's left edge). Their faint fill (1.08:1) already set them off; with the edge they read as four cards of the same kind as the value boxes, which suits click targets. A click on Hot-low outlines it in the text colour, 14.1:1 on the panel, clearly apart from the three at rest (`zoom-hot-low-tile-selected-1280x800.png`).
- **A selected choice still reads as selected.** In the plot picker, "Torque vs temperature" is the blue fill `#466a90` (3.06:1 on the panel) with bright text. Its grey edge is 1.06:1 to the fill: the ring is barely seen and adds nothing to read. The names not chosen stay frameless (`zoom2x-plots-picker-selected-choice.png`). In the open picker's list the chosen name is the same blue fill with the grey edge, 2.84:1 on the menu.
- **An open picker reads clearly.** Opened, with the pointer off it, Auto's picker keeps its grey outline (3.26:1) above its list, so it does not look dimmed beside the shut boxes (`zoom-auto-picker-open-1280x800.png`). The Design menu's "Show spreadsheet details" box is the grey fill, 3.02:1 on the menu (`zoom-design-menu-open-1280x800.png`).
- **A focused handle reads, by its size and ring alone.**
  - Tab walks the top bar's Design, Share, Export and Learn, then reaches the Required torque slider at the fifth press. The handle grows from 16 to 20 pixels across, and its ring from 1 to 2 pixels (`zoom2x-required-torque-handle-at-rest-tab-4.png` against `zoom2x-required-torque-handle-focused-tab-5.png`). Beside the next handle at rest the difference is plain; on its own it is a small cue.
  - It is the cue `main` has (D-98), not a regression. Owner question 73 offers the blue handle as its option (e).
  - Under the pointer a box's outline steps from the grey to the muted colour: 1.67:1, where `main`'s step was 4.48:1. The step is visible but small (`zoom-value-box-under-pointer-1280x800.png`), and register row A-52 records this cost.
- **The legend's dots, as they now ship, read as shown lines.**
  - At a pixel ratio of 1 the dark disc shows as a one-pixel dark gap between the grey ring and the dot; the dot fills the rest of the ring.
  - Each dot against that gap, on Torque vs temperature (starting design): violet 6.09:1, blue 7.16:1, and the muted Operating temperature dot 5.08:1 (5.80:1 against the field colour itself). On the example design's Gap sweep the red "Does not fit" dot is 5.01:1, and the amber 7.13:1 (`zoom3x-example-gap-sweep-legend.png`).
  - A click on "Operating temperature" hides its line and leaves the empty ring with the field colour inside (`zoom3x-plots-legend-operating-temperature-hidden.png`). A second click shows the line again, and the picture is then pixel for pixel the first one.
  - The muted dot is close to its ring (1.67:1), so that entry looks like a solid grey disc. It is plainly not the empty ring: the dot is 5.80:1 against a hidden entry's inside. So the muted entry reads as a shown line.
  - BL-108's fix did what it says. In the branch's first build the muted dot was 1.67:1 on a grey disc, and a hidden entry was the bare grey disc.
- **A disabled control.** On the example design, Overall length is set by Auto. Its box outline fades to 1.74:1, which reads as switched off beside the live boxes, as it should. WCAG 1.4.11 exempts an inactive control, and the line under it says why it is off.

Owner question 73 stays open with its conservative default, (a) the grey edge. Its text matches what ships: the dark disc in the grey ring, a hidden line's entry as the empty ring, and the tiles' grey edge with the selected tile's white outline.

### 2. The SemiBold title and heading, with the tiles' numbers still Regular (D-99, owner question 74)

**Ruling: they are set apart enough. No change.**

- The title and "Size and torque" carry about 27 % and 29 % more ink than `main`'s Regular at the same size. Counting pixels over one brightness threshold in the same box, the title has 1,257 against 987 and the heading 658 against 510. A tile number's count is the same on both (455). See `zoom2x-title-heading-tiles-branch-top-main-bottom.png`: the branch is on top, `main` below.
- The heading now differs by weight from the same words in the first tab and from the labels under it, so the column reads as titled. The title leads the top bar by its weight as well as by the crimson mark.
- The tile numbers stay larger than the title and in Regular, and still read as the screen's values. In SemiBold they would compete with the title. The design (6.2) allows it, but the panel names no font, so it would need panel code (D-99).
- The fit holds: the top bar is one row and the heading one line at the three sizes, and nothing is cut.

Owner question 74 stays open with its conservative default, (a) the title and the heading only.

### 3. "Loading the calculator" as the line, and what the page says when the wasm file fails to load (D-101, D-103, owner question 75)

**Ruling: keep the line. No change in this part.**

- **The words.** The line is plain. It uses the noun of the page's title ("Magnetic Coupling Calculator"), and has no ellipsis or spinner that would promise progress. It is muted `#8a919a` on the charcoal `#181b1f`, 5.43:1, in a 14 px system font (Inter is inside the file the line waits for). Its ink is centred, at x 572 to 709 and y 396 to 409 of the 1280 x 800 page (`loading-line.png`). The page names `lang="en"`, `theme-color` `#181b1f` and `color-scheme: dark`.
- **The timing, as in "What the prototype showed" item 7.**
  - Each load was instrumented in the page: a wrap of `Element.remove` and `Node.removeChild` caught the line's removal, and a wrap of WebGL's `drawElements` and `drawArrays` caught the first draw. A requestAnimationFrame counter gave each event's frame, and a flag cleared by a microtask showed whether the two fell in one JavaScript task.
  - Three loads on the local server, with the cache off: the line was on the page at DOMContentLoaded (14 to 21 ms) and was removed at 113 to 137 ms. Each time the first draw came 5 ms later, in the same animation frame and the same task, and the line was already gone at the first draw call.
  - With the network held at 1 MB/s: the line was on the page at 164 ms and removed at 7,309 ms, and the first draw came at 7,314 ms, in the same frame and the same task.
  - A browser paints only between tasks, so no paint can show the page with the line gone and the app not yet drawn.
- **The frames.**
  - A CDP screencast of the held load, started when the navigation committed, gave four frames: the line, then the app 7.3 s later, then the app twice more. There was no blank frame and no frame of another colour (`load-screencast-last-frame-with-line.png`, `load-screencast-first-app-frame.png`).
  - A screencast started before the navigation also caught one white frame first. That frame is the new tab's own `about:blank` (its background is transparent) before the page commits; the page's own first frame is charcoal.
- **A failed load.**
  - With the `.wasm` request aborted, the line stays. It was still there after 15 s, and the picture is pixel for pixel the loading picture (`failed-load-wasm-aborted-1280x800.png`).
  - The console shows `Failed to load resource: net::ERR_FAILED` and an uncaught `TypeError: Failed to fetch`. `main`'s page with the same failure is solid charcoal (`main-live-failed-load-wasm-aborted-1280x800.png`).
  - So the line says "Loading" of a load that has stopped. That is no worse for the reader than `main`'s blank page: either way they wait, then reload.
  - A failure line would be new words, which the scope does not need (8.3, and the owner's rule to keep wording unless the scope needs new text). D-103 is the conservative answer.
  - Question 75 tells the owner that the line stays, but offers no line for a failure (finding F-1).

Owner question 75 stays open with its conservative default, (a) "Loading the calculator".

## Findings

| # | Finding | Where | Blocks | What to do |
|---|---|---|---|---|
| F-1 | On a failed load the line stays and keeps saying "Loading the calculator". The page's script `init();` has no catch, so the console shows an uncaught `TypeError: Failed to fetch`. `main`'s page was blank on the same failure. This is decision D-103, not a defect of this part. Question 75 says the line stays, but none of its options is a line saying the load failed. | `linkage-sim-rs/web/tools/magcoupler/index.html` (the script's `init();`); owner question 75 | No | In the fix wave, add an option (d) to question 75: if the file cannot load, the line says so and asks for a reload, in words the owner chooses. The default stays (a), and nothing changes in the code in this part. |
| F-2 | In the app's first painted frame the four tiles have uneven heights: the first tile ends at y 152. From the second frame they share one height (y 204). This is one frame, at the moment the line gives way to the app. `main`'s live build draws the same first frame (a screencast of each: `main-live-load-screencast-app-frame-1.png`, `branch-load-screencast-app-frame-1.png`), so it is not a defect of this part. | `magcoupling-rs/src/gui/tiles.rs` (`tiles_ui`: each row's height is read from the last frame's memory, 0 on the first, then `request_repaint`) | No | File in `docs/ai/backlog.yaml` with the next free id, at low priority, for a later part. Draw the first frame at its settled layout (for example, egui's `request_discard` in place of `request_repaint` when a row's height changes), with a test that the first frame's tiles share one height. |

Nothing in this review needs a change before release. F-1 adds an option to owner question 75 in the fix wave, and F-2 is filed for later.
