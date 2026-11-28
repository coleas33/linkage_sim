# Update Tracker

Meaningful changes to the project that future sessions should know about.
Reverse chronological (newest at top).

---

## 2026-11-28 - Magcoupler: a legend disc and a resting tile keep a visible edge under a host that sets no resting outline, and the legend test checks the disc hovered and pressed (BL-110)

- gui/tiles.rs, gui/plots.rs (noninteractive hairline fallback); tests in tiles.rs, plots.rs, app.rs; README; backlog BL-110 fixed.

## 2026-11-28 - Magcoupler: a plot's legend sits in a corner its data leave free: Pole sweep and Slip heating at the bottom right (BL-106)

- gui/plots.rs PlotKind::legend_corner (read by plot()); test in app.rs; README; backlog BL-106 fixed.

## 2026-11-28 - Magcoupler: every plot's horizontal axis carries at least three numbers at 1024 x 700 and 1280 x 800 (BL-105)

- gui/plots.rs x_grid_steps (1-2-5 grid steps, finest with ticks over 60 points apart), x_number; test in app.rs; README; backlog BL-105 fixed.

## 2026-11-28 - Magcoupler: the left column holds 320 points at pixel ratios 0.9 and 1.35 (BL-103)

- gui/input_ui.rs add_slider (slider width capped to what its value box leaves; used by input rows and gui/size_controls.rs), gui/pickers.rs one point free in the part and grade pickers; test in app.rs; README; backlog BL-103 fixed.

## 2026-11-28 - Magcoupler: the Tab key passes the stops that show no focus: scroll-area surfaces and bars, panel edges, the Equation panel's formula (BL-102)

- gui/mod.rs scroll_area, hide_resize_edge, TabWalk (panel.rs calls it at the end of ui()); explorer.rs formula Sense::CLICK; test in gui/panel.rs; backlog BL-102 fixed, BL-124 filed (plot axes).

## 2026-11-28 - Magcoupler: the keyway-wall line is compared as printed, so no line reads 2.50 mm (BL-078)

- gui/geometry.rs wall_cautions rounds the wall to 0.01 mm before the 2.5 test; test in gui/panel.rs; backlog BL-078 fixed.

## 2026-11-28 - Magcoupler: a space-claim overshoot result's entry links the space-claim check (BL-095)

- badging_check maps housing.{diameter,length,bay}_overshoot_mm to the space-claim check; test in gui/explain.rs; README and backlog BL-095 updated.

## 2026-11-28 - Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-c: the Equation panel (BL-112)
- What changed: walking Start here at 1024 x 700 no term name, tag or step line runs past the Equation panel's edge (BL-086): a term's name and what it is share the term grid's last column and wrap in the panel's width, a long text value wraps in its own column, and the walk bar's step line wraps; the panel no longer grows past the centre region, on any equation. The equation itself still scrolls sideways where it is wider than the panel (decision D-109, owner question 76). The note drawing is `gui/notes_view.rs`, moved whole out of `gui/explorer.rs` (BL-088). The panel's empty text is gone (BL-089). The keyboard's focus is followed in the Equation panel, its formula and the Clamp tab's screw-size table (`gui::sideways_scroll_area`), and in the check list; a Results table line is a Tab stop only while its view shows it whole (BL-101, owner question 67's default).
- Tests: every text of the Equation panel but its formula whole on each Start here step at the three window sizes, the panel inside the centre region (app); the same on every equation the registry holds, in the app's font at 1024 points wide (panel); a panel open with no trail draws its header row only; σ's tag wraps inside a 1000-point window; the keyboard walks through the Equation panel (two equations, the panel at its starting and its least height), the open screw-size table, the Results tab (as it opens and wheeled) and the check list (panel).
- Docs: README, docs/ai 02, 03, 04, backlog BL-086, BL-088, BL-089, BL-101 and BL-112 in progress, register rows A-44 (its cost), A-55 and A-56, owner questions 67 (its text) and 76, the design's header.
- Fix wave of the whole-branch review: a term's name and tag that wrapped started at the middle of their grid row and hung half a row into the next one, under its stripe or its text (F-1, blocking): egui's Grid centres a cell's child by the height it first asks for, and a wrapping line asks for one widget's, so the cell is now a `ui.vertical`, which starts at the row's top. A wrapped name was a Tab stop per row (KR-1): it is drawn through `gui::label_block`, the input rows' one-block label (SK-1) made a shared helper, so after a wrapped name its tag starts a line of its own. `test_support::covered_texts_in` finds a glyph under a rect painted after it or under another text, and `equation_panel_faults` with `equation_panel_in_centre` is the one check of the Start here walk and the every-equation test (F-2, CTC-1); red first, with `a_term_whose_name_wraps_is_one_tab_stop_in_the_app_s_own_font` (panel). `Readouts::select_over` takes the sense and `readouts::sensor` is private again (CTC-2). Owner question 67 and A-44 say a focused Results table line a wheel cuts loses the focus and the next Tab starts at the top (KR-4). The formula's Tab stop with no focus mark joins BL-102 (KR-2); BL-119 filed (KR-3); CTC-3 declined.
- Fix wave of the whole-branch re-review: under an equation with no note of its own, the closest note's link beside its lead was a Tab stop per row when its title wrapped (RR-1, on main too): it is drawn through `gui::label_block`, which now takes any widget; red first, with the Tab stop test renamed `a_wrapped_term_name_or_closest_note_title_is_one_tab_stop_in_the_app_s_own_font` (panel), which now also opens two equations whose closest note's title wraps and flags a stop that starts under the one before it at its left edge or further left. The header row's buttons drawn over a long breadcrumb trail filed as BL-115 (RS-1), and a web-only re-wrap of an open note with the pointer on the Explain column's scroll bar as BL-116 (RR-2); both on main too.

## 2026-11-28 - Mounting angle from a file wrapped into the slider's range (BL-114)
- Branch review F-11: the Mounting Angle slider clamps to -180..=180 and file_io.rs did not normalise a file's mounting_angle, so 270 degrees was clamped, rebuilt and changed on the first idle frame with the section open. A loaded angle outside the range is now wrapped into it (270 becomes -90, the same mounting). Test bl114_* in gui/input_panel.rs.

## 2026-11-28 - Linear driver time in one method (BL-113)
- Branch review F-5, F-6: undo on a linear-driven mechanism reset the stroke to length_0 (probe: Slider-Crank at 1.2, two mount edits, undo gave 1.15 and converged=false), and recompute_dynamics and flip_assembly_branch took their time from driver_angle. AppState::current_driver_time() (state/mod.rs) is now the one match on the driver kind; rebuild, restore_snapshot, recompute_dynamics, flip_assembly_branch and property_panel/equations.rs use it (compute_t_mech deleted). MechanismSnapshot gains driver_stroke; restore_snapshot reinstalls the driver kind before it solves. Revolute behaviour is unchanged. Tests bl113_* in gui/state/tests.rs.

## 2026-11-28 - R5 review fixes
- Branch review of linkage/r5-mechanical, small findings: F-4 committed_number skips on_hover_text for "" (egui drew an empty tooltip box on the attachment-point and modulation fields); F-7 the BL-046 label plate colour is the named LABEL_PLATE in canvas/colors.rs (draw_pill_label differs in font, alpha and text colour, so it is not reused); F-8 the mounting rotation plus Y flip is ViewTransform::world_direction_to_screen beside world_to_screen, the old primitives.rs function deleted; F-9 the Chebyshev PNG test asserts the output dir starts with CARGO_MANIFEST_DIR/target, and AGENTS.md gives the regenerate and copy commands; F-10 BL-006's note starts DEFERRED and this branch's 2026-11-28 entries in this file are in reverse commit order.
- Backlog closed after checking each claim: BL-005 (rust-test-fixer agent already general; untracked here), BL-008 (no let delta; no mixed_script_confusables warning), BL-014 (duplicate of fixed BL-037).
- .claude/workflows/fix-campaign.js: the per-item reviewer no longer uses agentType feature-dev:code-reviewer (no shell, so it rejected every fix); the default agent reviews mechanical items, fbd-math-reviewer still physics ones. This change was already in the tree at 129f3c3; no doc described the old reviewer except the historical 2026-08-08 plan.

## 2026-11-28 - Weight mass cap stays 1,000 kg, said on hover (BL-034, A-57)
- Reworked 19f8546: WEIGHT_MASS_RANGE_KG back to 0.001..=1000 (question 33's default, option (a)); the weight editor's and + Mass field's hover say "Up to 1,000 kg."; tests: typing 5000 stores 1000, a 1000 kg weight's gravity power is 400 times a 2.5 kg one's; docs and A-57 updated.

## 2026-11-28 - Keep weights on driver reassign (BL-030)
- reassign_driver rebuilds a sample from the stock builder and carries each user weight over by body id into the new blueprint and live body (a weight on a body the stock sample lacks is dropped with a status message); replaces 8d01d32's blueprint path, which could not assemble (review F-1). Tests check convergence and the weightless pose (FourBar J4, WattII J2) plus the drop case; 02-system known limitation updated.

## 2026-11-28 - WASM guide describes the browser features it ships (BL-004)
- docs/guides/WASM_DEPLOYMENT.md: Known Limitations (no dialogs/export/autosave/recents) replaced by Browser Behavior; build and cache statements already matched build_web.sh and vercel.json.

## 2026-11-28 — Defer nom 1.2.4 warning (BL-006)
- cargo tree -i nom: nom 1.2.4 <- meval 0.2.0; meval 0.2.0 is still the latest and needs nom 1.x, so deferred with that upstream blocker named. No code or lockfile change.

## 2026-11-28 - Tests stop rewriting docs/chebyshev_lambda (BL-012)
- export_chebyshev_lambda_pngs writes to linkage-sim-rs/target/chebyshev_lambda/; a test pins the directory outside docs/; AGENTS.md says to copy by hand to refresh the committed images.

## 2026-11-28 - Web build keeps the CAD dark theme (BL-013)
- apply_cad_theme selects the Dark preference before styling, so a light browser preference no longer gives a light web build; two theme tests failed before, pass after (reused from Codex 261f6b4).

## 2026-11-28 - Readable zone-force label (BL-046)
- The zone-force label draws on a dark plate above the yellow overlap highlight; two painted-output tests; reused from Codex 33c0079.

## 2026-11-28 - Mounted force drawings (BL-049)
- Reaction, component, external and zone-force arrows turn with the mounting angle and a force zone draws its four turned corners; five 30-degree tests failed before, pass after (fix reused from the parked Codex branch).

## 2026-11-28 — Undoable committed force edits (BL-045)
- update_force_element records one undo step and sets dirty; force-editor numeric fields commit once on release or focus loss (from Codex 8e6ad7d); red test plus gate green.

## 2026-11-28 - Delete with toolbar focus (BL-033)
- handle_delete_shortcut gates on focused_widget_may_edit_text (TextEdit state plus response/touch guards) instead of wants_keyboard_input; five bl033_* tests in gui/mod.rs (reused from Codex ed88854).

## 2026-11-28 — Preserve linear pose on rebuild (BL-051)
- rebuild() takes its solve time from the newly assigned driver kind (linear stroke or revolute angle), so an edit no longer moves a linear-driven mechanism to driver length 0. Four bl051 tests (Codex acb1531's three plus a Slider-Crank sample on a linear driver at stroke 1.2) pass; the three linear ones fail with the fix reverted; GATE PASS.

## 2026-11-28 - Preserve sweep plot X pan/zoom (BL-019)
- with_default_x_bounds queues a one-shot X init (first display, range change/revisit, double click) instead of default_x_bounds every frame; reused from the Codex branch fix; four bl019 tests, drag and zoom failed before.

## 2026-11-28 - Preserve idle mounting angle (BL-021)
- Removed the Mounting Angle slider step_by(0.5) that snapped an off-grid angle and rebuilt every idle frame; regression test bl021_expanded_mounting_slider_preserves_off_grid_state_on_idle_frames.

## 2026-11-27 - Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-b: the look (BL-107)
- What changed: every control at rest shows its edge at 3:1 or more (the stage 1 review's E-4 and R-6): a value box, a framed button, a picker and a text box are outlined, and a checkbox's box and a slider's rail and handle filled, in `app::theme::CONTROL` (#656c76, a second colour outside the token table), where the hairline colour was 1.1:1 to 1.2:1; under the pointer and pressed a box keeps its fill and its outline brightens, and a slider's handle still shows its state by its ring's width alone; an open picker keeps the grey outline; every plot legend entry, egui_plot's mark of a toggle, shows its dot on a dark disc inside a grey ring, and the four tiles over the drawing are outlined in the grey (BL-108, BL-109). The title and the left column's heading "Size and torque" are Inter SemiBold (S-1): a second cut made by the committed script from the recorded Inter 4.1 file, which egui's Heading style draws in; every other text stays Regular. The page names its language, gives the browser the panel colour and a dark scheme, and shows "Loading the calculator" in the muted colour while the wasm file loads, which the app takes away at the end of its first frame, in the same paint (S-4). gui-smoke's look check reads the line gone and judges the heavier title and the controls' edges.
- Tests: a control's resting outline and fill at 3:1 on the panel, a menu and a field, its fill kept under the pointer and pressed, a handle's ring the text colour in every state and the open outline the resting one (theme); every rect and circle of the open left column and the Design menu's checkbox at 3:1 against what is behind it, Auto's picker opened still outlined at 3:1, every legend dot of the five plots at 3:1 or more on its disc, shown or hidden, and every disc's ring at 3:1 on the legend (BL-108), each tile's resting outline the resting outline of a framed control at 3:1 on the panel, the selected one in the text colour with no tile moved (BL-109), the title and the heading the only SemiBold texts with the top bar on one row at 1024 x 700, 1280 x 800 and 1920 x 1080, the page's head, line and script, the smoke prompt's words (app); the SemiBold cut's characters, digits and rows equal to the Regular cut's, and its family and the Heading style (fonts).
- Measured: the face adds 46,869 bytes to the wasm file (6,820,704 to 6,867,573; about 21 KB gzipped, 16 KB with brotli), and the loading line's code 6,566 more, so the branch takes it from 6,820,704 to 6,874,139 bytes (+53,435; about 23 KB gzipped); the committed page (LF) grows from 1,000 to 1,659 bytes (518 to 770 gzipped). The title is 262.1 points wide in SemiBold (257.7 Regular), the heading 134.6 (133.4). In Chrome the line's removal and the app's first draw fall in the same animation frame.
- Not in this part: part 2b-iii-c (the Equation panel: BL-086, BL-088, BL-089, BL-101's Equation panel part).
- Docs: README, docs/ai 02, 03, 04, backlog BL-107 in_progress, register rows A-52 to A-54, owner questions 73 to 75, the design's header and its 6.2.
- Review fix wave (branch magcoupling/overhaul-b2p2b-iii-b): owner question 73, register row A-52, the README and docs/ai 02 and 04 give the legend's real cost, every coloured dot 1.6:1 to 2.6:1 on the grey disc (4.4:1 to 6.9:1 on main), the red "Does not fit" dot lowest, and the dot alone the shown or hidden cue (LOOK-1; the panel change filed as BL-108); they name the four tiles over the drawing, which keep their hairline, as a framed control the grey does not reach (LOOK-2; filed as BL-109); the Measured line gives the committed page's sizes and the branch's whole wasm cost (F-1, F-2); the SemiBold test takes `fonts::semibold()` (CC-1) and the app and theme tests share `theme::hex` (CC-2; docs/ai 03 names it).
- BL-108 and BL-109 folded in before the release (same branch): egui_plot draws a legend entry's disc in the interact state's fill, so on the grey every coloured dot was 1.6:1 to 2.6:1 and the dot alone showed a line shown or hidden; `plots::plot_ui`, in the scope where it sets the plot's visuals, gives the resting, hovered and pressed fills the field colour, so each dot sits on a dark disc inside the grey ring (3.5:1 on the legend), 5.7:1 (the red "Does not fit") to 15:1, and a hidden line's entry is the empty ring; no control outside the plots changes. The tiles (`tiles.rs`) take the resting outline's colour at rest (the grey, 3.3:1 on the panel, where the hairline was 1.2:1), one width with the selected and focused text-colour outline, so no row moves. Tests (app, the app's theme): `a_plot_legend_s_dots_show_at_three_to_one_on_their_discs_shown_or_hidden` (replaces `a_plot_legend_s_entries_show_their_dots_on_a_grey_disc`; five plots, the starting and the example design, a click hides and a second shows, the eight series colours drawn) and `a_tile_shows_its_edge_on_the_panel_as_a_framed_control_does`, each red first. Docs: README, docs/ai 02, 04, owner question 73, register row A-52, backlog BL-108 and BL-109 fixed.

## 2026-11-27 - Magcoupler UX overhaul, slice B, stage 2, part 2b-iii-a: the Plots tab (BL-104)
- What changed: the five plots share one Plots tab (design C-9): the tab row is Size and torque, Geometry, Plots and Results table (the Clamp tab before the table while that section shows), one row from 1024 x 700 up where it wrapped at 1280 x 800. Under the Plots tab a picker names the five plots in the order their tabs had and marks the one shown; the tab opens on the plot shown last. Red in a plot marks only a sweep row that does not fit (decision C-5, the stage 1 review's C-3): the required torque and the sweeps' required floor are drawn in the strong text colour, the governing limit and the time to it in violet, the pull-out point in the strong text colour. Each axis draws a tick number at full strength or not at all (the stage 1 review's E-7). gui-smoke judges the tab row and the picker (`plots_picker`).
- Tests: the tab row, the picker, its picks (each drawn in its click's frame) and the tab's memory, and the keyboard through the Plots tab and the picker (Tab changes no view, Enter on the tab, Space or Enter on a name) (panel); red only on a sweep row that does not fit in every plot on five designs, every tick number in the axes' colour at 25 window sizes in two visuals (plots); the app's text-cut and column-width tests reach each plot through the picker; the smoke prompt's words (app) and the verdict test's list (85 checks). Two tests take a new setting because the tab row is now one row: the Geometry list's focus test drags the Equation panel 40 points taller, and the underline test's wrapped row is drawn in an 860 x 800 window.
- Not in this part: part 2b-iii-b (the look: the controls' outlines, a SemiBold face, the page shell's loading line) and part 2b-iii-c (the Equation panel: BL-086, BL-088, BL-089, BL-101's Equation panel part); the sparse horizontal axes at 1024 x 700 and 1280 x 800 (BL-105, filed: egui_plot's 60-point threshold, kept by decision D-91, as on main).
- Docs: README, docs/ai 02, 03, 04, backlog BL-104 in_progress and BL-105 filed, register rows A-47 to A-51, owner questions 70 to 72, the design's header.
- Review fix wave (branch magcoupling/overhaul-b2p2b-iii): `CentreView::tab_row` is the one tab row, which `centre_ui` draws and the smoke prompt's guard reads, and the selected tab is the view shown, not a text compare (CQ-1); `the_tab_row_is_one_row_in_the_app_s_own_font` (app) pins the one row at 1024 x 700 and 1280 x 800 in Inter (T-1; it fails with the unfolded tabs); register row A-50 names the sweeps' shared legend dot (LOOK-1); BL-104's note names the real branch (D-1); doc and comment wording (Q-2, Q-3).

## 2026-11-27 - Magcoupler: fix wave of the re-review of BL-097 to BL-100 (branch magcoupling/columns-hold-still)
- SK-1: since BL-099's head line wraps, an input label that wrapped was two Tab stops and the second showed no focus (egui lays a label out in a wrapping line row by row, each row a Tab stop of its own; only the first draws the focus underline). `input_ui::input_row` now draws the label as one block, one text row high, so a label that fits sits where it did and a wrapped one is one stop with its focus shown on both lines; after a wrapped label the note and the reset button take a line of their own. Red test first: `an_input_label_that_wraps_is_one_tab_stop_in_either_font` (app; it failed with 80 stops against 64 for 16 wrapped labels).
- RF-1: the comment of `a_wheeled_column_holds_still_while_the_focus_is_on_no_widget_of_its_own` and BL-100's note now give the test's own reading with the clip check taken out (the second stop, 24.5 points), not the first review's 28 points.
- Docs: README (the column note, the app tests), docs/ai 02 (the column note and the egui lesson), 04, backlog BL-099 and BL-100 notes.

## 2026-11-27 - Magcoupler: fix wave of the controller's review of BL-097 to BL-100 (branch magcoupling/columns-hold-still)
- BL097-EP-1 (blocking): a number typed into a tiny value's box was thrown away. `typed_number` took every coarser rounding of the value for the box's own text, and egui holds any two numbers under about 1.9e-6 equal, so on the vacuum permeability (1.256637e-6), the NdFeB expansion (-8e-7) and the cap's and the web's field integrals typing 1.2566e-6, -1e-6 or 0 changed nothing where main edits. `input_ui::slider` now records the text egui's own formatter writes in the box (`Slider::custom_formatter`) and compares the typed number with that text alone. Red test first: `a_number_typed_into_a_tiny_value_s_box_edits` (panel); `a_value_box_s_own_text_is_no_typed_number` gains the tiny cases.
- W-1: the check list's material warning's Why link was on a line that did not wrap, and with no back iron (or a sleeve that conducts better than 316L) it widened the right column for good, on main too: it is now drawn through `gui::wrapping_line`. Red test first: `the_right_column_keeps_its_width_with_a_material_warning_shown` (panel).
- KF-1: a test for `follow_focus`'s clip check, the guard that keeps a column still while the focus is elsewhere (the panels share one egui layer): `a_wheeled_column_holds_still_while_the_focus_is_on_no_widget_of_its_own` (it fails with the check taken out); the code's comment says the layer check adds little.
- W-4: the app's left-column width test holds Auto on with the Detailed inputs open (it fails at 445.2 points with Auto's line made a plain row). W-5: one `test_support::side_rect` for the panel and app tests; the right column's tests use `SMALLEST`.
- W-2, docs only: in the app's font a grade picker's blank choice ("Blank: the manual Br, no rating") is newly cut, where main's column had grown wider first; README, BL-099's note, register row A-45 and owner question 68 now say so. The choice stays the owner's (question 68).
- Filed: BL-103 (W-3): at pixel ratios 0.9 and 1.35 the left column still widens past 320 points by a slider row that cannot wrap.
- Docs: README, docs/ai 02 (the DragValue lesson), 03, 04, backlog BL-097 to BL-100 notes and BL-103, register row A-45, owner question 68.

## 2026-11-26 - Magcoupler: the right column keeps its width (BL-098; branch magcoupling/columns-hold-still)
- Why: showing a teaching note with a diagram (the first is "harmonics", from Learn or any lesson link) made the right column 316 points wide where it starts at 300, and it stayed that wide after the note was left: the centre, its tiles and its drawing lost 16 points for the rest of the session.
- Root cause: a note's diagram is a fixed 300 points wide (`DIAGRAM_SIZE`) and the Explain panel's column holds 284; egui's side panel grows to what it holds and keeps the width.
- What changed: `diagrams::diagram_ui` takes the width it is given: 300 x 130 where it has room, else the same picture scaled as one (shapes, strokes and labels) to that width, 284 x 123 in the Explain panel. The sweep of every entry the Explain panel can show found a second cause of the same kind: a label in a wrapping line can stand a fraction of a point past the line (0.06 of a point in two notes' change entries, 0.375 in the entry of the result `temperature.adhesive_life.daily_screen` in the app's font), which widens the column by that much. BL-099's free point is now one shared rule, `gui::wrapping_line`, and every wrapping line of the two side columns goes through it: the left column's head lines (`input_ui::head_line`), Auto's row and its Key design line, and on the right the Explain entries' lines, a note's change entries and equation links and a check's number rows.
- Kept: every word; the check entries, the tile entries and the note list as they were (their layout tests pass unchanged); the column can still be dragged.
- Tests (red first; each fails again with its part of the fix taken out): `the_right_column_keeps_its_width_whichever_note_is_shown` (panel; 1024 x 700 and 1280 x 800; it failed at 316 points on "harmonics", and at 300.0625 with the diagram fixed and no free point), `the_right_column_keeps_its_width_whatever_entry_is_shown_in_the_app_s_own_font` (panel, `app` feature: every entry in the app's font; it failed at 300.375 on the result), `the_right_column_keeps_its_width_with_a_note_shown_in_the_app_s_own_font` (app: a note shown through Learn, the width and `cut_texts` on every tab at both sizes) and `a_diagram_takes_the_width_it_is_given_scaled_as_one_picture` (diagrams). The helper that finds a diagram by its marked points (`diagram_points`) now looks for them at the scaled radius.
- Swept and clean after the fix, so nothing was filed from the sweep: 1117 selections (the orientation, the note list, 20 notes, 10 checks, 4 tiles, 6 Size rows, every input, every result), with the spreadsheet details off and on, on the starting and the example design, in egui's default font and in the app's.
- Docs: README, docs/ai 02 (a note and a lesson, and BL-099's lesson), 03, 04, backlog BL-098 in_progress, register row A-46, owner question 69.

## 2026-11-26 - Magcoupler: the left column keeps its width (BL-099; branch magcoupling/columns-hold-still)
- Why: in the app's own theme and font the left column grew from 320 to 330.1 points when the Detailed inputs first opened, to 332.1 with the Assumptions, up to 443.1 with the last input group open, and stayed that wide after everything was shut again: the centre and its drawing shifted for the rest of the session, and text was cut on two tabs at 1024 x 700. In egui's default font an opened group made it 418.8.
- Root cause: a row's head line (the changed dot, the label, the out-of-range note, the reset button) was a `ui.horizontal`, which does not wrap, so a label longer than its row ran past the column's edge; egui's side panel grows to what it holds and keeps the width. The row that set 330.1 points is the Key design row "Thermal conductance to the housing and shafts"; 443.1 is "Opposite-ring field at an aluminium hub (fundamental, free space)". The item's evidence blamed the part pickers: they fit the width they are given under the theme's button padding (measured).
- What changed: one rule, `input_ui::head_line`, draws the head line of every input row and of every assumption on a line that wraps at the row's width, one point kept free (a label in a wrapping line can stand up to a point past it). The note and the reset button are added to that line itself (`ui.add_enabled`), not in a child ui, which would not wrap back to the row's start. Found by the build and fixed with it: a group's heading with its traced count ("Requirements and operating conditions (1 traced)") is longer than the column in the app's font and made it 336.6 points when the Required torque was traced; the heading is now a galley wrapped at the header's width (`panel::wrapped_title`).
- Kept: every on-screen word; the column can still be dragged wider or narrower; a row whose label fits is drawn where it was. The part and grade pickers cut their choice with an ellipsis where it does not fit, as before (register row A-45, owner question 68).
- Tests (red first; each fails again with its part of the fix taken out): `the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font` (the app's theme and font at 1024 x 700, 1280 x 800 and 1920 x 1080: the column's width from egui's `PanelState` and `cut_texts` on every tab, with the folds shut, each fold opened and shut again, every input group and its Advanced heading opened and shut again, a filter that lists every input set and cleared, an input traced; it failed at 330.1 points on the first opened fold, and at 336.6 on the traced heading) and `every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font` (every input row, as it is, changed, outside its range and locked, at the three widths the column gives a row, in both fonts; it fails for two rows by 0.28 of a point without the free point).
- Docs: README, docs/ai 02 (the column note and three lessons), 03, 04, backlog BL-099 in_progress, register row A-45, owner question 68.

## 2026-11-26 - Magcoupler: a scroll area follows the keyboard's focus (BL-100; branch magcoupling/columns-hold-still)
- Why: with the Detailed inputs open, most Tab stops of the left column were below the window's foot and the column did not follow (33 of 57 at 1024 x 700 in the panel tests): the focus went where nobody could see it.
- Root cause: egui scrolls no scroll area to a widget for taking the keyboard's focus.
- What changed: `gui::follow_focus(ui)`, called last in a scroll area's content, scrolls that area by the least that shows whole the widget the focus has arrived on (`scroll_to_rect(rect, None)`), in the first frame the area is drawn with the focus there and never again. One rule, four callers: the left column, the Explain panel, the Geometry tab's callout list, the Clamp tab. It keeps the focused id it saw last (egui's own `gained_focus` misses an Up or Down arrow key's move), and takes a widget for its own when the area's clip rect is what cuts it.
- Kept: the focus arriving selects nothing and edits nothing (BL-093); a wheel or a drag of the scroll bar after the arrival is not fought; an idle frame asks for no repaint.
- Probed, not fixed, filed as BL-101: the Equation panel and the Clamp tab's screw-size table (sideways scroll areas nested in an up-and-down one: egui hands a scroll to the innermost area alone, so the one rule does not reach them) and the Results table (only the rows in view are drawn; with the rule Tab would walk every row before it left the table: register row A-44, owner question 67). Found on the way, filed as BL-102: egui gives the keyboard's focus to each scroll area's drag surface and scroll bar and to each column's resize edge, none of which shows it.
- Tests (each fails with its area's call taken out, and all three with the rule's scroll taken out): `the_left_column_follows_the_keyboard_s_focus_and_fights_no_wheel` (1280 x 800 and 1024 x 700, Tab, Shift+Tab and a Down arrow walk, the least scroll, the wheel, the idle frame), `the_explain_panel_follows_the_keyboard_s_focus`, `the_centre_s_scrolling_views_follow_the_keyboard_s_focus`; `test_support::shift_tab` is shared with the trace-link test.
- Docs: README, docs/ai 02 (a note and two lessons), 04, backlog BL-100 in_progress, BL-101 and BL-102 filed, register row A-44, owner question 67.

## 2026-11-26 - Magcoupler: a value box left with nothing typed changes nothing (BL-097; branch magcoupling/columns-hold-still)
- Why: live on the public site. On the starting design, Auto on, Auto off, then Tab through the Size column rounded the adopted magnet length from 11.942 to 11.9 mm with nothing typed: an edit, an undo step, a selection, and hot-low 1.3445 N·m under the 1.35 N·m requirement (one failed check where there was none).
- Root cause: egui's value box (`DragValue`) writes its text back when the keyboard's focus leaves it, typed or not, and the text is the value at the box's decimals, clamped to the slider's range; `Slider::set_value` then snaps it to the step.
- What changed: `input_ui::slider` gives egui's `Slider` one `custom_parser`, `input_ui::typed_number`: the box's own text (the value formatted as the box shows it, at any decimal count up to the step's) is no typed number, so nothing is written back. One rule for every caller (the Size column's sliders, the Required torque, `input_row`'s two), with no state between frames. A value outside its slider's range is covered too: a Tab pass brought it to the range's end. The special case for the Minimum OD's slider (BL-096, decision D-73: the slider handed the value its box shows) is removed; its tests hold on the one rule and fail without it.
- Behaviour pinned: typing the number a box already shows is no edit either (1.4123 mm shown as 1.41 stays 1.4123 when 1.41 is typed); any other typed number, on Tab, Enter or a click elsewhere, an arrow key and a drag edit as before. Register row A-43, owner question 66.
- Tests (red first, and red again with the parser taken out): `tab_through_a_length_auto_left_off_its_step_changes_nothing` (the starting design; `app` feature), `tab_past_every_value_box_keeps_a_value_off_its_step_or_outside_its_range`, `a_typed_value_and_an_arrow_key_edit_a_value_off_its_step_and_its_own_text_does_not` (panel), `a_value_box_s_own_text_is_no_typed_number` (input_ui); the Tab loop of `tab_through_the_size_column_selects_nothing_and_sets_no_minimum_od` is now the shared `tab_past`.
- Docs: README, docs/ai 02 (the Size column note and a lesson), 03, 04, backlog BL-097 in_progress, register row A-43, owner question 66.

## 2026-11-25 - Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-c: the left column and the nothing-lost test (BL-096)
- What changed: the left column is the Size column on every tab (design 4.3): the size controls, then the folds Detailed inputs and Assumptions, which are now the only home of the input groups and of the assumptions; the tabs "Design inputs" and "Assumptions" are gone (`InputsView`). The folds are drawn with no indent, so an opened fold is no wider than the inputs side was, and scroll with the column (BL-091). A trace's banner and Clear trace sit at the column's foot on every tab, the opening one too. A fix link on the Outside diameter scrolls the column back to its slider. Tab through the Size column selects nothing and sets no Minimum OD while every size is on its slider's step. gui-smoke judges the left side on the Geometry tab (`same_column`) and opens the Detailed inputs fold for its tooltip step.
- Tests: the fold and the filtered view scroll with the column and a row asked for still scrolls into view; Tab through the Size column; the column's rows in the same place on every tab, the folds' state kept across tabs, the column 320 points wide with a fold open (egui's default font); the banner and Clear trace at the column's foot on every tab at 1024 x 700 and 1280 x 1024; the Outside diameter's fix link from a scrolled column; the table-driven nothing-lost test (the old top bar's actions, every CentreView, every shown note in Learn, every DASHBOARD row's home, every input and every assumption from every tab); the smoke prompt's words (app) and the verdict test's list. The panel tests' harness opens the Detailed inputs fold where a test drives its rows.
- Fix wave of the whole-branch review (2026-11-26): the Key design group's Open the Size tab button scrolls the left column back to Auto's toggle (it set only the centre tab, so from a scrolled column the click showed nothing new; red test first); the nothing-lost test counts the rows drawn, so an input that shares a label with a Size control or another input cannot lose its row unseen; the test harness's `fold` fails when its click shuts a fold; BL-099 records the opened-group case (443.1 points, text cut on two tabs at 1024 x 700).
- Second fix wave (2026-11-26, the re-review's two findings): with both folds open and a filter that kept its row, the dropdown of "Highest odd harmonic summed" was drawn twice under one widget id and no click or Space opened it (main's opened): an unindented fold's body has no id of its own. The Assumptions fold's body now has one (`push_id`), and the fold keeps its full width. Red test first (`a_dropdown_drawn_in_both_folds_opens_and_takes_its_pick`); the panel tests' `Harness::frame` now fails on egui's ID clash warning in every frame (`assert_no_id_clash`, shared with the window test). BL-099 records that an opened input group widens the column in egui's default font too (320 to 418.8 points).
- Not in this part: part 2b-iii (the Plots tab and the look). Filed, not fixed: BL-097 (Tab through a value box rewrites a value off its step: Auto on, Auto off, Tab), BL-098 (a teaching note widens the right column), BL-099 (in the app's own font an opened fold or a filter widens the left column), BL-100 (a row that takes the keyboard's focus is not scrolled into view).

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-b: the drawing's selections and the word budget (BL-094)
- What changed: a click on a dimension tag (its line or its number), a row of the Geometry tab's callout list or a results-table row selects its result, and the Explain panel shows its entry (its value and name, what it is, the check of its row, the stored-3D line where it applies, its lesson, its Equation link, its cell with spreadsheet details on); a check's own row selects the check. Such a click no longer opens the Equation panel; the entry's Equation link does, and those hovers lost the line that said a click opens it. The opening tab draws the drawing with its tags and no callout list (a note with a level stays under it while true); the Geometry tab keeps the list. The first view draws 200 words at 1280 x 800, held to 250. The gui-smoke run clicks tag 1 on the Geometry tab and checks the face gap's entry (`tag_entry`).
- Tests: a selecting readout by click and by Enter or Space, not on focus (readouts); any result's entry, the Equation and lesson links, a check's own result (explain); a tag, a callout and a table row select and open no equation (panel); the opening tab's drawing keeps its tags and only the notes with a level (geometry_view); the first view at most 250 words (app); the trace and Equation panel tests go through the entry's link; the smoke prompt guard and the verdict test (81 checks).
- Not in this part: the left column no longer changing with the centre tab, the table-driven nothing-lost test (part 2b-ii-c); part 2b-iii as before.

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-ii-a: every input a selection (BL-092)
- What changed: touching a Detailed inputs or Assumptions row (a click on its label, a drag of its slider, focus in its field, an edit or a reset) selects its input, and the Explain panel shows its entry (its value, what it is, its hint, the checks whose fixes name it, the "pulls two ways" sentence, its cell with spreadsheet details on). Every input's entry starts with the link "Highlight the results this drives", which traces it (a second click ends the trace); the label no longer traces, and its hover lost the line that said it did. A check's fix link selects its input as well as showing its row. With spreadsheet details on, a tile's and a check's entry end with their cell. The gui-smoke run clicks the hovered Key design label and checks the input's entry (`input_entry`).
- Tests: any input's entry, the trace link, the fix link's selection, the cell feet (explain); a label's click and a touch of the row's widget select and a hover does not, the entry's link traces (panel); the trace tests go through the link; the smoke prompt guard and the verdict test (79 checks).
- Not in this part (plan decision D-49): dimension tags, results-table rows and callout rows as selections, the callout list to the Geometry tab, the word budget of 250, gui-smoke reading a dimension tag's entry (part 2b-ii-b); the left column no longer changing with the centre tab, the table-driven nothing-lost test (part 2b-ii-c); part 2b-iii as before.

## 2026-11-23 - Magcoupler UX overhaul, slice B, stage 2, part 2b-i: the Size column's lines in the Explain panel (BL-090)
- What changed: the Size column draws each control's label and slider, then the folds Detailed inputs and Assumptions (the Assumptions view is now one click from the opening tab). Touching a control selects it, and the Explain panel shows its entry: its value, what it is, what sets it (its hint, the what-moves-what lines, Minimum OD's floor, the line on what the controls move), its torque-card row, the checks whose fixes name it, the "pulls two ways" sentence where it applies, its Why note, and with spreadsheet details on its cell. The hint lines, the what-moves-what lines, the floor line, the Why links and the footnote no longer show under the controls.
- Tests: each row's entry (explain), touching selects and a hover does not, the column draws no moved line, the Assumptions fold opens on an assumption's row; the size tab's line tests read the entries; the first view holds at most 280 words (272).
- Review fix wave (2026-11-23): the footnote on what the controls move (`LINKED_HOUSING`) is in the Outside diameter and Overall length entries only, and left out while Auto sets the magnet count, whose count its last clause says stays (PT-1; register A-28, owner question 52 on cutting it and four hint lines that repeat a hover). `size_controls::row_label` takes the row, shows `row_help` on hover (Required torque's label now has its hover too) and senses `Sense::CLICK` alone, so a label is no Tab stop. The Assumptions fold draws its rows without a scroll area of its own (the inputs side wraps `assumptions_ui` in one), so the column scrolls through them. Tests: `the_footnote_on_what_the_controls_move_is_in_the_diameter_and_length_entries_alone` (red first), `the_assumptions_fold_scrolls_with_the_size_column_to_its_last_assumption`; the touching test hovers every slider and Tabs from the length slider to the gap slider in exactly two presses.
- Not in this part (plan decision D-34): dimension tags, results-table rows, callout rows and Detailed inputs rows as selections, the callout list to the Geometry tab, the word budget of 250 (part 2b-ii); the Plots tab, the outlines, the SemiBold face, the page shell, BL-086, BL-088, BL-089 (part 2b-iii).

## 2026-11-22 - Magcoupler UX overhaul, slice B, stage 2, part 2a: the top bar and the notes (BL-085)
- What changed: the top bar is Undo, Redo, Design, Share, Export and Learn, words with no box until hovered; the Design menu holds the reset (Reset to the starting design where the baseline is the starting design), Open the example design, Save design, Load design and Show spreadsheet details; the Equation panel button left the bar. The Explain panel draws the teaching notes: Learn lists them by title, and every note link on the page (Why links, warnings' links, lesson links, the lesson line under an open equation, each Start here step) selects its note there; the Equation panel's Explain checkbox and its note view are gone. A text value takes no unit (the slip-heating readout read "never: steady state stays below the limit s").
- Tests: the top bar's and the menu's order and places, the buttons' box only under the pointer, Learn's list and a note's entry, the lesson line (drawn whole at every walk step in the app's font), the walk selecting its notes, the readout's words; the panel's note-link tests read the selection; the first view holds at most 430 words (427).
- Review fix wave (2026-11-23): a Design menu item run from the keyboard closes the menu as a click does (each item calls `ui.close()`); the reset's status line reads "Starting design restored" after Reset to the starting design ("Default design restored" stays for Reset all); one `show_note` sets a note as the selection; tests for a text value's unit in the results table and the Equation panel's value line; BL-088 (move the note drawing out of explorer.rs) and BL-089 (the Equation panel's empty text is unreachable) filed for part 2b.
- Not in this part (plan decision D-20): the Size column's folds and lines in the Explain panel, inputs, dimension tags, table rows and callout rows as selections, the Plots tab's picker and red reference lines, the word budget of 250, the controls' outlines, a SemiBold face, the page shell (part 2b); BL-083 and BL-084 (D-30).

## 2026-11-22 - Magcoupler: an amber caution for a cap skirt under 0.5 mm (BL-082; branch magcoupling/skirt-wall-caution)
- Why: owner answer 4a of 2026-11-22 (question 4 of `docs/analyses/2026-11-18-owner-questions.md`): no check judged the cap's skirt, so a 0.15 mm skirt passed all 9 checks (design section 12.4 item 1, register row A-6).
- What changed: `gui::geometry::wall_cautions` returns the Size tab's amber wall lines and `panel.rs` draws them: the wall past the keyway (moved from `panel.rs`, words unchanged) and the cap skirt, (cap outside diameter - cap thread diameter) / 2, while it is under `MIN_CAP_SKIRT_MM` = 0.5 mm compared as quoted at four significant digits: "Cap wall past its thread (skirt): 0.4900 mm; guide minimum is 0.5 mm.". It counts as no failed check; the starting design (0.55 mm) draws none and still reads "passes all 9 checks". The 0.5 mm is a conservative guide value (register row A-22).
- Tests (red first): `a_cap_skirt_under_the_minimum_draws_one_caution_naming_it`, `a_cap_skirt_at_the_minimum_draws_no_caution`, `the_skirt_caution_is_drawn_amber_on_the_opening_tab_and_counts_no_check`; `the_starting_design_draws_no_caution_on_its_opening_tab` also pins the 0.55 mm skirt and looks for the skirt line.
- Docs: README, docs/ai 02/03/04, backlog BL-082 in_progress, register rows A-6 and A-22, owner question 4 (built).
- Review fix wave: `size_tab_new_texts_have_glyphs_and_plain_words` takes a 0.49 mm skirt and checks the wall lines' glyphs and plain words (R1); the doc comment, README and A-22 call the skirt nominal (a tapped root sits a few hundredths of a mm past the nominal diameter) (P2); a skirt at or below zero filed as BL-087 (P1); the Geometry tab's missing wall lines left to slice B placement (R2).

## 2026-11-22 - Magcoupler UX overhaul, slice B, stage 2, part 1: the instrument's columns (BL-081)
- What changed: the right column is the check list (`gui::checks`: the status line, a row per counted check in one fixed order with a shape and a word for its state, the material warnings and the stored-3D line, which is now on the first view at 1280 x 800) over the Explain panel (`gui::explain`: the orientation on the first frame; a click on a check shows its Problem, Why sentence, fixes, notes and numbers; a click on a tile shows its help and, for hot-low, the torque card and the check of its row). The opening tab is four tiles over the drawing and its callout list (`gui::tiles`); Required torque leads the Size column, which scrolls in a short window, and the footnote on what the size controls move sits under the Detailed inputs fold, which is on the first view. Every state's mark on the page is a shape (`checks::state_mark`): the dashboard's filled-disc badge is gone from the entries, the material warnings and the Results tab. The dashboard's list, the welcome strip and the centre's readouts and card are gone, each text to the home the plan's nothing-lost table names.
- Tests: each module's own, red first; `the_first_view_keeps_its_three_columns_its_tiles_and_the_fixed_3d_line_in_the_app_s_own_font` (1024 x 700, 1280 x 800, 1920 x 1080 in the app's theme and font, the starting and the example design) and `the_first_view_draws_at_most_four_hundred_and_fifty_words_at_1280_by_800` (608 words before, 443 after); `a_fix_link_on_the_opening_tab_shows_its_row_inside_a_short_window`; the gui-smoke prompt reads the new layout (`check_entry` and `lesson_link_opened` replace `dashboard_scrolls` and `why_link_opened`).
- Not in this part (plan decision D-1): the top bar and Design menu, Learn, the Size column's folds and hint lines in the Explain panel, inputs and dimension tags as selections (the callout list then leaves the opening tab), the Plots tab's picker, the word budget of 250, the slip-heating readout's stray unit.

## 2026-11-22 - Magcoupler starting design: the standard 1.4 mm keyway and 3D fields of its own (BL-080; branch magcoupling/starting-design-own-fields)
- What changed: the owner's answers to register rows A-17 and A-18 (owner questions 3 and 2). `app::starting_design` sets the hub keyway depth to 1.4 mm (DIN 6885-1 t2 for a 3 x 3 mm key on a 10 mm shaft; the spreadsheet's 1.7 mm stays in the example design), so the wall past it is 2.55 mm and the Size tab's amber line is gone; and it stores the fourteen 3D fields of a run of the reference routine for its own geometry (`gui::dashboard::STARTING_3D_FIELDS`, from the new `reference/magcoupling-py/tools/starting_design_fields.py`, corrections E3, E5, E17 carried). All nine verdicts unchanged; the magnet limit 93.06 -> 94.94 C. Texts about the stored fields name both designs: the dashboard line "Fixed 3D fields: starting or example design", its hover, every stored 3D input's last help sentence ("Stored 3D value, starting or example design; it does not follow yours."), the two input section headings, five teaching-note sentences; the temperature Problem's source clauses name the design whose reverse fields are in ("example" or "starting").
- Tests: the_starting_design_draws_no_caution_on_its_opening_tab, the_standard_keyway_depth_moves_no_check_of_the_starting_design, a_link_and_a_file_saved_from_the_first_starting_design_open_it_as_it_was (fixtures in magcoupling-rs/tests/data/saved_designs/, written at main 2ca65ee), the_starting_design_stores_the_3d_fields_of_its_own_geometry, a_temperature_problem_names_the_design_whose_stored_fields_it_read; input_schema.json re-blessed. The smoke script reads only the status line, which does not change.
- Docs: owner questions 1 to 6 marked answered (2026-11-22); question 48 added (the 3D routine's pull-out torque is 0.73 of the engine's for the starting design); BL-082 filed (the skirt-wall caution at 0.5 mm, question 4); question 6's answer (Codex's branch stays parked) in 04-memory.

## 2026-11-22 - Magcoupler slice B, stage 1: the fix wave of the whole-branch review (BL-079; branch magcoupling/overhaul-b1-the-look)
- Why: three independent reviewers read the stage at 68b0dbc and a skeptic confirmed every finding that is not a suggestion (`.superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/final-review.json`, gitignored). One finding blocked the merge: with one text colour, values no longer stood out from their labels.
- What changed, each fix with its red test first where a test can fail for it: a label beside its value and a caption's head are muted (`look::quiet_text`; a host that sets no weak colour keeps its text colour for them, for the tabs that are not active and for the plots' axes); selected text shows, on a fill of its own, a quiet blue that is the one colour outside the token table (`theme::SELECTION`, register row A-19), and the panel's marks read `look::mark_color`; no text is cut sideways in any tab at 1280 x 800 or 1024 x 700 in the app's own font (the pole sweep's line is named in a wrapping caption over its plot, a plot readout longer than a row wraps, a tooltip's equation wider than the window is set smaller, a plot's axis label keeps its letters' tops); a plot's own weak marks are faded once in a host without the theme; the font cut carries the built-in font's ascender, descender and line gap, so a row is as tall as on `main` and the first view's drawing is 197 points tall at 1280 x 800, from 175 (211 on `main`); four doc sentences are true of the tip.
- Tests: `a_label_is_muted_and_its_value_is_in_the_text_colour`, `no_text_is_cut_in_any_tab_in_the_app_s_own_font` and `the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font` draw the app in its theme and font (the panel's layout tests draw in egui's built-in font, about 13 % narrower); the theme's tests pin the selection fill's contrast on each surface; `test_support::cut_texts` and `text_colors` are new. 1,056 lib tests with the app feature (1,045 at 68b0dbc).
- Not in this wave, listed in the plan's "After the whole-branch review" and, where they are stage 2's, in its "What follows": the plots' reference lines in the failing red on a passing design, the controls' faint outlines, the brand mark's shape (the owner's eye), a second font face, the top bar's buttons, the page's loading line. Not run by the wave: `gate.sh`, the browser smoke run, the release build; the merge and the push are the controller's.

## 2026-11-18 - Magcoupler UX overhaul, slice B, stage 1: the look (BL-079; branch magcoupling/overhaul-b1-the-look)
- What changed: the calculator has its own theme and font, on today's layout. `app/theme.rs` holds one token table (UX overhaul spec 6.1: charcoal surfaces, hairlines, text, muted, the crimson accent, the three level colours; links keep egui's blue) and `visuals()`, egui's dark visuals with the tokens in place of its colours; the web page's background is the panel colour. This reverses M41-3 ("Theme matches the linkage app"; the spec's question Q-1, on its recommendation): the linkage app is not touched. `app/fonts.rs` installs Inter ahead of egui's built-in fonts: a 30,684-byte cut of Inter Regular whose ten digits are its tabular ones (`assets/fonts/`, with the script that built it and the licence). The accent marks the brand (a square before the title) and the active tab (a line under it), and nothing else (`gui/look.rs`). The plots' grid, ticks and axis labels take the weak text colour. The hub page's card no longer lists "clamps".
- Tests: the tokens are the spec's table and the visuals are the tokens (this replaces the parity test with the linkage app); every text token is at least 4.5:1 on every surface it is drawn on, with the spec's own figures; the digits have one advance width; every glyph guard of the panel also holds its texts to the cut of Inter (`test_support::assert_glyphs`, with the app feature); the accent is on two shapes of the app's page and on no text or badge. The gui-smoke workflow gained `look`, and its verdict test (gate 13) the case.
- What the spike found (design 6.2): the cut holds every character the guards collect; Inter's `tnum` feature also widens spaces and punctuation, so the digits alone are remapped; Inter's rows are taller than the built-in font's (1.21 em), and the first view at 1280 x 800 still shows the whole centre; `strong()` no longer sets a value apart (one text colour, no bold): a heavier face is the cure if the tiles of stage 2 need one.
- Not in this stage: the layout (the top bar, the Size column, the tiles, the check list) and the Explain panel, which are stage 2; the fixed-3D line on the first view (register row A-18), which stage 2's check list brings; the slip-heating readout's stray unit (design 9.2).

## 2026-11-17 - Magcoupler slice A, fixes from the independent review (BL-077; branch magcoupling/overhaul-a)
- What changed: the web app no longer writes the address while a key is held. A held Ctrl+Z or Ctrl+Y undoes or redoes a step at every key repeat and each step is a settled design, so the first build sent a write per repeat (Safari refuses more than about 100 `replaceState` calls in 10 seconds); the run now gives one write at the key's release. `app::replace_address` returns whether the page took the write, and `address_design` advances only then, so a refused write is asked again on the next frame (no back-off) in place of leaving a stale address.
- Tests: `a_held_undo_or_redo_writes_the_address_once_at_its_release` and `a_refused_address_write_is_tried_again_on_the_next_frame` (both red before the fix), and `a_link_cut_at_two_thousand_characters_is_refused_and_the_header_says_why` (new coverage, green before). Under test `replace_address` is `app::tests::replace_address`, which records each write per thread and can refuse one. Record: the "independent review" addendum of `docs/analyses/2026-11-17-magcoupler-slice-a-review.md`.
- Wording (second commit): four texts called the spreadsheet's design "the default design", which on screen now means the design the session started on. They say "example design": the temperature Problem's two source clauses (`guidance.rs`: "3D results for the example NdFeB design", "slip fields from the example design"), the marker hover's last line (`corrections::MARKER_SCOPE`) and two registry texts (`deviations.rs`, E20 and E21 `corrected_formula`, which no screen or export shows). Tests: the pinned Problem strings, a new pin of `MARKER_SCOPE` and `deviations::tests::no_entry_calls_the_spreadsheet_s_design_the_default_design` (all red before the change). Left as they are: "Default design restored" after Reset all (it restores the baseline; slice B rewords it), and the code comments and test names where "default design" means `DesignInputs::default()`.
- Not verified: the write and its refusal in a real browser (Chromium's smoke run does not reach the limit; Safari and Firefox are not run).

## 2026-11-17 - Magcoupler UX overhaul, slice A: the page opens on a starting design, and the design is in the address bar (BL-077; branch magcoupling/overhaul-a)
- What changed: the panel holds a session baseline (`MagcouplingPanel::with_baseline`; `new()` keeps the spreadsheet's values). The changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow it; the spreadsheet export's flag and the assumptions banner stay against the workbook. The app opens on `app::starting_design()` (8 magnets, ring radius 9.00 mm, face gap 1.80 mm, outer return ring wall 2.20 mm, cap thread 41.7 mm, Required torque 1.35 N·m = 0.35 N·m of hot bearing drag + 1.00 N·m to accelerate the wheel), which passes all 9 checks; "Open the example design" opens the spreadsheet's values as one undo step. On the web the app rewrites the page's address with `history.replaceState` after every settled edit, undo, redo and load (`app::address_of`: clean at the starting design, the share link elsewhere). `DesignInputs::default`, the engine and every parity fixture are unchanged.
- Tests: 25 new headless tests (panel, inputs, dashboard, guidance, history, app); the gui-smoke workflow gained `starting_design` and `address_follows`, and its verdict test (gate 13) their cases. The text and physics review is `docs/analyses/2026-11-17-magcoupler-slice-a-review.md`.
- For the owner: the starting design draws one amber line on the opening tab, "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm." (register row A-17: kept visible, the approved values not moved). The review changed one word of the starting design's hover ("magnets" to "magnet part": the number per ring changes) and found that the starting design's temperature check and inner-hub fitting temperature use the example design's stored 3D fields, as every design's do (register row A-18: left as it is, now stated in the README, the doc comment and the design's 7.1). Not released until the owner's review of the diff (design 9.4).

## 2026-11-16 - Magcoupling tests: the differential JSON is parsed once per test binary (audit MT-02; branch audit/2026-11-16-speed)
- What changed: MT-02 (74b5353): `load_cases` (magcoupling-rs/tests/common/mod.rs) caches each module's parsed cases for the life of the test binary and returns a shared `&'static [Case]`; `parse_cases` is the uncached parser. deviations.rs clones the inputs it moves; `every_branch_is_reached` lost its local cache. Measured: differential + deviations + explain, debug, 43.1 s before, 41.9 s after (within noise).
- Tests: new `shared_cases_are_one_copy_equal_to_a_fresh_parse`. The data files never change during a run, so nothing to invalidate. Checks as above.

## 2026-11-16 - Magcoupling dashboard: a guide's notes are built only when its block is open (audit MGV-06; branch audit/2026-11-16-speed)
- What changed: MGV-06 (e105f84): `GuideShown.notes` (gui/dashboard.rs) is a closure that `fix_line` calls, so `guide_notes` runs only under an open block (red, or amber once opened). The level is still computed each frame (the header and the open decision need it); the fit row's second level call is left. Drawn texts unchanged. Not measured.
- Tests: new `a_guide_reads_its_notes_only_when_its_block_is_open` (reads counted at green, shut amber, red); existing guide tests pin the drawn text. No cache. cargo test, cargo test --features app, both clippy runs and fmt --check pass.

## 2026-11-16 - Magcoupling input rows: tooltips built only when hovered (audit MGP-03; branch audit/2026-11-16-speed)
- What changed: MGP-03 (c20fbd4): `input_row` (gui/input_ui.rs) builds the row tooltip, the label hover and the reset hover inside `on_hover_ui` closures, so nothing is formatted until a tooltip shows. Hover text is unchanged. Not measured.
- Tests: existing `an_input_row_s_hover_names_its_path_and_cell_only_with_the_spreadsheet_details` and the widget-hover panel tests pin the texts. No cache. Checks as above.

## 2026-11-16 - Canvas: motion ribbon poses solved once and reused (audit LGC-18; branch audit/2026-11-16-speed)
- What changed: LGC-18 (f05212a): `draw_motion_ribbon` (canvas/rendering/mod.rs) keeps its ghost poses in `AppState::motion_ribbon_cache` (a `RefCell<Option<RibbonCache>>`; the render pass borrows state immutably) and draws from them. Inputs: ghost count, driver rate and start angle, the sweep's `u_values`, the warm-start seed `last_good_q`, and the mechanism (no revision counter exists, so `ribbon_cache_is_fresh` checks coordinate and constraint counts plus the constraint residual of every cached pose). While the live pose moves (playback) the seed changes each frame, so the poses are still re-solved then; idle, hover and pan repaints reuse them. Drawn ghosts are unchanged. Measured (debug build, 20 ghosts, sample four-bar): 14.5 ms to solve vs 0.15 ms for the freshness check.
- Tests: new `ribbon_poses_match_the_original_per_frame_loop` (pin), `ribbon_cache_is_reused_while_inputs_are_unchanged`, `ribbon_cache_is_recomputed_when_any_input_changes` (ghost count, rate, start angle, u_values, seed, mechanism rebuild); ignored timing test `ribbon_timing_solve_versus_freshness_check`. cargo test --all and clippy pass.

## 2026-11-16 - Plot panel: sweep borrowed when no shift is needed (audit LGC-05; branch audit/2026-11-16-speed)
- What changed: LGC-05 (e064f5f): `sweep_in_display_frame` in plot_panel/mod.rs returns a `Cow<SweepData>`. Stroke mode and a zero driver offset borrow the stored sweep, so those frames copy nothing. An angle-mode sweep with a non-zero offset is still cloned once per draw (the plots take `&SweepData`; passing the shifted vectors separately would touch every plot function). Drawn values are unchanged. Not measured.
- Tests: new `display_frame_shift_values_are_pinned` pins the shifted `angles_deg` and `toggle_angles`, the untouched other series and stored sweep, and that stroke mode and zero offset borrow. No cache, so no invalidation test. cargo test --all and clippy pass.

## 2026-11-16 - Magcoupling speed: defaults, schema and slider ranges built once (audit MEX-01, MEC-03; branch audit/2026-11-16-consistency)
- What changed: MEX-01 (f674edc): `api::workbook_defaults()` and `api::input_meta(path)` hold the workbook defaults and the input schema in OnceLocks; `assumptions::states`, `modified`, `any_modified` and `reset_to_workbook_defaults` read them instead of rebuilding both on every call (the GUI calls them every frame). MEC-03 (7ac33da): `FreeVariable::range` and `size_controls::geometry` read slider ranges from `input_meta`; ranges are static metadata, so they do not depend on the design values. Behaviour is unchanged. The cost per call was not measured, and the gui-smoke frame timing was not run.
- Tests: new `the_cached_defaults_and_schema_equal_a_fresh_build` (tests/assumptions.rs) and `the_cached_slider_ranges_equal_a_freshly_built_input_table` (tests/sizing.rs). Both guard equality with a fresh build and cannot fail on the old code. cargo test, cargo test --features app, clippy (both feature sets) and fmt pass.

## 2026-11-16 - GUI consistency: plot tab label, reference-line span, expression editor commit (audit LGC-07, LGC-08, LGS-18; branch audit/2026-11-16-consistency)
- What changed: LGC-07 (af964ed): in stroke mode the driver-effort tab is "Stroke Force (Static)" (`driver_tab_label`); the separate "Actuator Force" tab keeps its name. LGC-08 (4f670f8): `reference_line_x_span` in plot_panel/mod.rs gives the transmission-angle thresholds and the MA = 1 line the sweep's own display x range (custom angle range, radians, stroke in mm); empty sweep falls back to a full turn. LGS-18 (6634710): the expression-driver boxes apply once when a box loses focus (`expression_edit_action`); Esc restores the blueprint's texts; an edit equal to the applied driver pushes no undo step.
- Tests: new `stroke_mode_driver_tab_label_differs_from_actuator_force_tab`, `reference_line_x_span_follows_the_sweep_x_range`, `expression_edit_applies_once_when_the_edit_ends`, `expression_driver_texts_reads_only_expression_drivers`. Each targets a new helper, so none can fail on the old code; the egui focus, Esc and drawing paths are untested. cargo test --all and clippy pass.

## 2026-11-16 - LGE-18: index_at_driver and angles_deg docs say what trajectory sweeps hold (audit; branch audit/2026-11-16-consistency)
- What changed: LGE-18 (65fa63f): doc comments only. A computed trajectory sweep fills `angles_deg` with sample times in seconds (only `empty_trajectory_sweep_data` leaves it empty); `index_at_driver` compares degrees, so it must not be called on a trajectory sweep. No guard was added: whether `current_sweep_index` callers can reach a trajectory sweep is a behaviour question left to the owner.
- Tests: none (comments only). clippy passes.

## 2026-11-16 - one world bounding-box helper for the mechanism SVG and the schematic (audit LGE-17; branch audit/2026-11-16-consistency)
- What changed: LGE-17 (a6afd9c): `padded_world_bounds` in gui/export/svg.rs builds the padded world box for both exporters (pad 15% for the SVG, 20% for the schematic, unchanged). It takes the schematic's rules: linear-driver endpoints are included and a zero-size span is widened about its centre (threshold 1e-10). Existing samples give identical output; the mechanism SVG changes only for a driver endpoint that is not an attachment point (no longer clipped) and a zero-size span (centred, not stretched one way).
- Tests: new `padded_world_bounds_include_linear_driver_endpoints_and_centre_a_zero_span`, red with the driver loop removed. cargo test --all and clippy pass.

## 2026-11-16 - one four-bar detector for the sweep and the report (audit LGE-10; branch audit/2026-11-16-consistency)
- What changed: LGE-10 (1d11528): `detect_fourbar` in gui/sweep/fourbar.rs (the report's rule: ground plus 3 moving bodies, exactly 4 revolute joints, one revolute driver, no linear driver, lengths from the joint anchors) is the only detector. `detect_fourbar_links` (transmission angle, Grashof) is a tuple view of it; the report's `detect_fourbar` adds the world joint positions at the pose. A four-bar with a coupler point or an extra ground point used to be "not a four-bar" in the sweep (no transmission angle, no Grashof result) and is now detected; lengths no longer depend on the pose.
- Tests: new `detect_fourbar_ignores_extra_attachment_points`, red with the old attachment-count rule. cargo test --all and clippy pass.

## 2026-11-16 - driver time at a sweep sample uses the shared helper (audit LGS-04 follow-up; branch audit/2026-11-16-consistency)
- What changed: LGS-04 (follow-up, df2fdcf): `sweep_time` (gui/sweep), `compute_t_mech` (gui/property_panel/equations.rs), the GIF export loop (gui/export/raster.rs) and the non-Grashof sample loader (gui/samples/special.rs) call `driver_angle_to_time` instead of writing `(angle - theta_0) / omega` out again. The helper stays `pub(crate)` in gui/state (every user already reaches it). `sweep_time` angle mode had no zero-omega guard (inf, or NaN at angle == theta_0); it now returns 0 like its stroke mode. Test-only copies with constant omega are left alone.
- Tests: new `sweep_time_zero_omega_is_zero_in_both_modes`, red on the old code (inf). Loading clamps omega to 0.01, so only direct callers of `compute_sweep_data` reach omega = 0; the GIF export rejects omega = 0 first; the equations copy already had the guard. cargo test --all and clippy (with and without --features raster) pass.

## 2026-11-16 - magcoupling GUI: status constants, unit helper, clamps_paused, one Minimum OD clear (audit MGV-05, MGG-04, MGV-04, MGP-10; branch audit/2026-11-16-dry)
- What changed: MGV-05 (3acf7c5): `STATUS_NOMINAL` and `STATUS_BELOW_MINIMUM` in engine/sweeps.rs, used by the status code there and by gui/plots.rs (strings unchanged; the Python oracle's). MGG-04 (f28a61d): `guidance::result_unit(path)` replaces the unit one-liner at seven sites (guidance.rs x5, torque_card.rs, explorer.rs). MGV-04 (0969e9f): `explorer_ui` takes `clamps_paused` for the Start here hover and the Used by links; `START_HERE_HOVER` is removed (use `start_here_hover(clamps_paused)`); the Start here walk (`notes::START_HERE`) still reads `clamps::PAUSED`; the clamp section stays paused and hidden. MGP-10 (e68ae6a): private `MagcouplingPanel::replace_design` clears `minimum_diameter_mm` for open, load and reset (undo and redo still do not clear); a load now clears before `set_design`, not after, like open and reset.
- Tests: new `result_unit_is_the_results_own_unit_and_empty_for_a_non_result`, `the_panel_s_pause_decides_the_drawn_used_by_links` and `a_new_design_clears_the_lowered_minimum_od_but_undo_and_redo_keep_it`; the start-here hover tests check both forms. cargo test, cargo test --features app, clippy with and without --features app and cargo fmt --check pass.

## 2026-11-16 - sweep CSV column list (audit LGE-13; branch audit/2026-11-16-dry)
- What changed: LGE-13 (034ffc3): `write_sweep_csv` in gui/export/csv.rs builds one list of columns (header, decimals, series) and writes the header and every row from it; the toggle column stays sparse (blank on NaN). Output is byte-identical.
- Tests: new `sweep_csv_all_columns_pinned` pins header and rows with every optional column, a short series and a toggle; it passed on the old code first. cargo test --all and cargo clippy --all-targets -- -D warnings pass.

## 2026-11-16 - menu download helper (audit LGS-09; branch audit/2026-11-16-dry)
- What changed: LGS-09 (0a115f4): the File menu's "generate, download, report" block (eleven sites) is `download_generated` in gui/menu_bar.rs, with the same file names, MIME types, filters and error messages. `download::download_text` had no caller left and is removed; `download_bytes` takes text as bytes.
- Tests: new `a_failed_generation_is_logged_with_its_prefix_and_opens_the_error_panel`; the Ok path opens a save dialog and cannot run headless. cargo test --all and cargo clippy --all-targets -- -D warnings pass; wasm32 lib check compiles the web-only JSON download.

## 2026-11-16 - sample builders use build_standard_fourbar (audit LGE-15; branch audit/2026-11-16-dry)
- What changed: LGE-15 (ed2f476): quick-return, toggle clamp, Hoeken, Roberts, figure-8 coupler curve and pantograph call `build_standard_fourbar` (now `pub(super)`; coupler point is an optional (x, y), plus an `above` flag). Left as they are because the skeleton differs: build_fourbar_with_driver, parallelogram press and actuator, both Chebyshev samples, slider-cranks, Whitworth, bell crank, six-bars. No behaviour change.
- Tests: new `standard_fourbar_samples_keep_their_coupler_point_and_branch`; golden fixture tests unchanged and green; cargo test --all and cargo clippy --all-targets -- -D warnings pass.

## 2026-11-16 - serialization error helper, joint body lookup and two-point force view (audit LFA-14, LFA-04, LFA-06; branch audit/2026-11-16-dry)
- What changed: LFA-14 (2cf46d2): `SerializationError::build(cause)` replaces the `|e| SerializationError::Build(e.to_string())` closure at ten sites in io/from_json.rs. LFA-04 (b1c80a1): `joint_bodies` in io/to_json.rs does the two body lookups of the four `joint_to_json` arms (this also covers the four to_json lookups LFA-14 listed; message unchanged). LFA-06 (3f5635f): `ForceElement::two_point_ends` and `two_point_ends_mut` name the spring, damper, gas spring and actuator once; compound.rs (three functions and a test), `resolve_named_points` and from_json.rs (two helpers) use them. `attached_body_ids` keeps its full variant list (the compiler checks it). No behaviour change; no copy differed.
- Tests: new `joint_bodies_tests` (exact messages) and a `two_point_ends` test; cargo test --all and cargo clippy --all-targets -- -D warnings pass.

## 2026-11-16 - driver angle-to-time helper and one copy of the time-modulation maths (audit LGS-04, LFA-15; branch audit/2026-11-16-dry)
- What changed: LGS-04 (f1a73d3): the driver-angle to time mapping, written out eight times under gui/state, is one driver_angle_to_time(angle, theta_0, omega) in state/solver_helpers.rs with the zero-omega guard (returns 0.0). flip_assembly_branch and the two load-time solves in file_io.rs had no guard and gave inf/NaN at omega = 0; they now match the other copies. Copies outside gui/state (sweep, export, samples, property_panel) were not touched. Follow-up df2fdcf moved those copies to the helper. LFA-15 (fa5352d): TimeModulation::factor calls compile, so the modulation maths exists once; expression warnings from `factor` now carry the compile prefix.
- Tests: new helper tests (omega = 0, sub-epsilon, 0/0) and time_modulation tests for every variant and the bad-expression cases; cargo test --all and cargo clippy --all-targets -- -D warnings pass.

## 2026-11-16 - linkage-sim-rs drops the unused `open` dependency (audit LCS-01, INF-16; branch audit/2026-11-16-cleanup)
- What changed: removed the optional `open` crate and `dep:open` from the `native` feature (64d5126; that commit's subject line wrongly reads "apply machine-applicable clippy fixes (INF-02)", its body is the INF-02 text, but the diff is only Cargo.toml and Cargo.lock). Cargo.lock loses open, is-wsl, is-docker and pathdiff. Kept `wasm-bindgen-futures` (the async `#[wasm_bindgen(start)]` in src/bin/linkage_web.rs expands to it; removing it breaks the wasm32 build) and `uuid` (the dxf crate needs its `js` feature on WASM). `meval` is used.
- Tests: cargo test --all, cargo clippy --all-targets and the gate 3 wasm32 check pass.

## 2026-11-16 - linkage-sim-rs clippy warnings cleared and gate 2 denies them (audit INF-02; branch audit/2026-11-16-cleanup)
- What changed: the crate went from about 275 clippy warnings to zero with no behaviour change: cargo clippy --fix (8722228), then hand fixes and narrow allows with a one-line reason for dead native export entry points and long parameter lists (9a03b3c). Gate 2 now runs `cargo clippy --all-targets -- -D warnings` (3998a12); BL-003 is fixed.
- Tests: cargo test --all passes; the wasm32 linkage-web check builds (it keeps seven wasm-only warnings, which gate 3's plain cargo check does not deny).

## 2026-11-16 - Dead and test-only code removed (audit LGC-01, LGE-06, MGG-06, MGV-03, MGV-08; branch audit/2026-11-16-cleanup)
- What changed: linkage-sim-rs deletes `EqKind::driver_lambda_units`, `draw_actuator_stroke_annotation` (4afc5c8) and the unused `add_plotly_multi_chart` (1d262b7; not reused because the three inline charts differ in mode, colours, per-series x, axis anchor and name escaping). magcoupling-rs marks `TermColors::is_empty`, `Laid::texts` (eda24a8), `Cell::as_text` and `package_label` (fbf0186) `#[cfg(test)]`, and drops `dep:log` and `dep:js-sys` from the `app` feature (e44603a). No behaviour change.
- Tests: both crates' full suites and clippy gates pass (magcoupling-rs with and without the `app` feature); no new tests, removal only.

## 2026-11-16 - The panel's clamp-text allowlist is asserted empty with its reason (audit MGP-11; branch audit/2026-11-16-section-c)
- What changed: in magcoupling-rs/src/gui/panel.rs the test `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` keeps its `ALLOWED` hook (ruling 11: no text may name the paused clamp) and now asserts it is empty, with a message that an entry needs the owner's say and the assertion changes with that ruling.
- Tests: shown to bite: one entry in ALLOWED fails the test with "ALLOWED holds a text that may name the paused clamp". Reverted.

## 2026-11-16 - The ignore guard's name and doc say the six files it checks (audit MT-06; branch audit/2026-11-16-section-c)
- What changed: magcoupling-rs/tests/explain.rs `only_the_review_tools_are_ignored` is now `in_the_six_files_that_hold_them_only_the_review_tools_are_ignored`, and its doc says it reads exactly the six listed sources. Scope is not widened (a directory walk would give up the compile-time `include_str!` list). The README, 03-structure.yaml and 04-memory.yaml follow the rename.
- Tests: shown to bite: a test with an ignore attribute appended to tests/explain.rs fails the guard ("ignored checks: [tests/explain.rs: zz_extra]"). Removed.

## 2026-11-16 - notes_before_snapshot is a documented write-once generator that cannot fail (audit MEX-06; branch audit/2026-11-16-section-c)
- What changed: in magcoupling-rs/src/engine/explain/notes.rs the ignored `notes_before_snapshot` asserted its output file did not exist, but the file is committed, so a hand run with `--ignored` could only fail. It now has a doc comment with the command (`cargo test --lib notes_before_snapshot -- --ignored`), the command in its ignore reason, and returns with a printed note when the file exists. It stays ignored and stays in the ignore guard's allow-list.
- Tests: shown by the hand run: `--ignored` now passes where the old body panicked on the existing file.

## 2026-11-16 - Python sweep tests fail on an unsolved or missing sample instead of skipping it (audit PY-08; branch audit/2026-11-16-section-c)
- What changed: in test_numerical_consistency.py (the 3 six-bar FD tests and 5 sweep invariant/residual tests), test_sixbar_ternary.py, test_fourbar_benchmark.py and test_textbook_benchmarks.py, every `if sample is None: continue` is now an assert, so a sweep that stops converging, or a float-key lookup that stops matching, turns the test red instead of a green no-op. All samples converge today, so nothing else changed. Left alone on purpose: test_regression_baselines.py:501,503 (deliberate near-toggle exclusions) and test_failure_modes.py:498 (already follows assert n_converged == len(angles)). The three pasted angle-to-index loops are not merged (leads.md).
- Tests: shown to bite: making position_sweep return None for step 7 fails 13 tests. Reverted.

## 2026-11-16 - The assertion-free ground-input MA test now asserts (audit PY-03; branch audit/2026-11-16-section-c)
- What changed: tests/test_mechanical_advantage.py test_ground_input_returns_zero now passes "ground" as the input body and asserts input velocity 0.0, a nonzero output velocity and MA == inf. A new test, test_same_body_input_and_output_gives_unit_ma, covers the rocker-to-rocker case the old comment promised (MA == 1).
- Tests: shown to bite: making _get_velocity return 1.0 for ground fails the ground test. Reverted. Run with PYTHONPATH set to this worktree's src: the editable install points at the main checkout.

## 2026-11-16 - Template tests stop writing to the real ~/.linkage-sim (audit LGS-14; branch audit/2026-11-16-section-c)
- What changed: gui/state/templates.rs gets storage_root(): ~/.linkage-sim normally, a per-process temp dir in unit tests. New test tests_persist_under_temp_dir_not_home. The developer's existing files ("test", "roundtrip", "a", "b", "only") were left in place.
- Tests: shown to bite: pointing storage_root() back at the home dir in test builds fails the new test. Reverted. Commit d8d3c57.

## 2026-11-16 - braindump_repro.rs header says what the file is (audit INF-11; branch audit/2026-11-16-section-c)
- What changed: the "TEMPORARY ... Not for commit" header is now "tracked repro for BL-017; keep it". Comment only. The backlog note "(untracked)" in BL-017 is still stale; left for the owner.
- Tests: none needed (comment only). Commit e700cd4.

## 2026-11-16 - Duplicate sample-count test removed; PeaucellierLipkin documented (audit LGE-20; branch audit/2026-11-16-section-c)
- What changed: samples/mod.rs sample_count_is_30 (exact copy of all_samples_listed) deleted. The PeaucellierLipkin variant builds a change-point 4-bar; it is public API so it is not renamed, a doc comment says so (the label already reads "Coupler Curve (figure-8)"). Renaming is an owner call.
- Tests: all_samples_listed still asserts 30. Commit d4d9f4e.

## 2026-11-16 - The two weak solver tests assert something that can fail (audit LCS-16; branch audit/2026-11-16-section-c)
- What changed: inverse_solve_status_constructs_all_variants (severity.rs) matches every variant and its fields; fourbar_reactions_extract (statics.rs) requires resultant == hypot(force_global) per joint and one nonzero joint reaction. The duplicated 4-bar fixture is left: no crate-wide test-support module exists (the helper is pub(crate) inside inverse_kinematics only), so sharing it is a refactor for the owner to decide.
- Tests: shown to bite: a changed Singularity field fails the severity test; resultant forced to 0.0 fails the statics test. Both reverted. Commit 5e344f9.

## 2026-11-16 - The solver property tests and the spring fixture count the cases that reach an assertion (audit INF-14; branch audit/2026-11-16-section-c)
- What changed: tests/property_tests.rs: the residual and velocity properties moved into check_residual_after_solve and check_velocity_constraint (return whether the case reached its assertions); two new tests require at least 40% of cases to reach one (63% do today); a converged position with no velocity solution now fails. tests/golden_fixtures.rs spring_affects_statics_driver_torque asserts convergence and a virtual_work_check Ok at every angle and counts them.
- Tests: shown to bite: solve_position forced to converged false fails both reach-count tests; virtual_work_check forced to Err fails the spring fixture. Both reverted. Commit 33bfce8.

## 2026-11-16 - A control target with a zero direction is an error, not a panic (audit LCS-13; branch audit/2026-11-16-section-a)
- What changed: new ControlTarget::validate (inverse_kinematics/control_target.rs) returns the new LinkageError::InvalidControlTarget for a Projection whose axis_dir is zero or NaN; workspace_probe calls it first, so solve_for_target and the trajectory sweep return an Err (Strict and Analysis alike) instead of stopping at the assert. The asserts in evaluate/gradient/hessian and the projection() constructor stay. Readers checked: evaluate/gradient/hessian are reached only after a probe (solve_for_target, sweep, trajectory_ops); no match on LinkageError is exhaustive.
- Tests: zero_direction_projection_target_is_an_error_not_a_panic (red first: panicked "Projection axis_dir is zero"). Commit 134b45d.

## 2026-11-16 - check_grashof no longer panics on a NaN link length (audit LFA-18; branch audit/2026-11-16-section-a)
- What changed: the sort and the shortest-link search in analysis/grashof.rs use unwrap_or(Ordering::Equal), as motor_sizing.rs does. A NaN length gives is_grashof false, is_change_point false and NonGrashof. Readers checked: update_grashof (blueprint_ops.rs) and crank_selection.rs read only the result fields.
- Tests: nan_link_length_is_not_grashof_and_does_not_panic (red first: panicked at the old unwrap). Commit c287553.

## 2026-11-16 - The virtual-work check names a missing driver (audit LFA-12; branch audit/2026-11-16-section-a)
- What changed: virtual_work_check (analysis/virtual_work.rs) looks for the driver body pair first and returns the new LinkageError::NoDriver ("Mechanism has no driver; add a revolute driver before this analysis.") instead of MechanismNotBuilt. The two unknown-body lookups below it still report MechanismNotBuilt. Readers checked: no match on LinkageError is exhaustive, so no other arm was needed.
- Tests: virtual_work_on_built_mechanism_without_driver_reports_no_driver (red first: got MechanismNotBuilt). Commit bd0ba5f.

## 2026-11-16 - Two samples refuse a driver joint they cannot honour and the Custom 6-bar errors when no start solves (audit LGE-14; branch audit/2026-11-16-section-a)
- What changed: build_chebyshev_lambda_actuator (samples/fourbar.rs) and build_custom_6bar (samples/special.rs) return an error for Some(joint) other than "J1" (None and "J1" build as before; both samples drive on J1). The Custom 6-bar's start search moved into custom_6bar_from_json and ends in an error when no start angle converged, instead of a log line and a zero q0. Readers checked: the only caller with a joint is AppState::reassign_driver (state/driver_ops.rs), which already logs an Err and falls back to the blueprint path; build_sample (None) is unchanged, and the Custom 6-bar test shows its embedded blueprint still solves.
- Tests: samples_with_a_fixed_driver_joint_reject_another_joint and custom_6bar_that_does_not_assemble_is_an_error (red first: both failed, the second needed the new custom_6bar_from_json seam, added before the fix). Commit 181acc5.

## 2026-11-16 - The browser build's shortcut help no longer lists Ctrl+S and Ctrl+Shift+S (audit LGS-03; branch audit/2026-11-16-section-a)
- What changed: gui/mod.rs builds the shortcut help rows in shortcut_rows(browser); the window passes cfg!(target_arch = "wasm32") and the two save rows are dropped in the browser build, where handle_keyboard_shortcuts has no handler for them (it is feature = "native"). Not changed: the File menu button labels in gui/menu_bar.rs still say Ctrl+S (outside the finding; the menu's Save is a different path in the browser).
- Tests: shortcut_help_lists_save_rows_only_outside_the_browser (red first: did not compile, no builder). The wasm32 branch itself is not run natively; the test passes browser = true. Commit d5776d7.

## 2026-11-16 - Renaming a link is one undo step and marks the file changed (audit LGC-12; branch audit/2026-11-16-section-a)
- What changed: property_panel/pending_edits.rs UpdateLabel calls push_undo before it edits the blueprint and mechanism (undo step, dirty = true); the sweep recompute stays (the weight breakdown shows the names). property_panel/mod.rs keeps the text being typed in egui memory and sends one UpdateLabel when the field loses focus with changed text (Esc drops it), as the weight name field does, so a rename is one undo step, not one per keystroke. Readers checked: UpdateLabel has one producer (the link editor label field).
- Tests: rename_link_is_one_undo_step_and_marks_dirty (red first: undo depth did not grow). Commit 54c6c43.

## 2026-11-16 - A cosine-stroke linear driver saves and loads as a cosine driver (audit LFA-01; branch audit/2026-11-16-section-a)
- What changed: LinearDriverJson (io/schema.rs) has an optional cosine_stroke [min, max] field (serde default, skipped when absent). to_json writes it for DriverMeta::CosineStroke (length_0 = initial length, velocity kept for old readers); from_json builds cosine_linear_driver when it is present and constant_velocity_linear_driver otherwise, so old files load as before. Owner decision (controller): finish the feature, not delete it. Readers checked: the only other LinearDriverJson builder is gui/state/driver_ops.rs (sets None).
- Tests: cosine_linear_driver_round_trips_through_json (red first: loaded back as LinearLength). Commit c855eb5.

## 2026-11-16 - A compound force that cannot be expanded leaves no orphan bodies (audit LFA-07; branch audit/2026-11-16-section-a)
- What changed: forces/compound.rs expand_compound_force resolves both end references and checks the body and attachment point exist before it adds the cylinder and rod, and returns an error otherwise. from_json already logs the error and adds the original force, now with no jointless bodies left behind. Readers checked: from_json is the only caller outside tests.
- Tests: failed_expansion_leaves_no_orphan_bodies (red first: force_0_cyl was left in the mechanism). Commit 7fbeb5e.

## 2026-11-16 - The Health panel shows toggle and over-80% positions the way the plots do (audit LGC-16; branch audit/2026-11-16-section-a)
- What changed: new plot_panel::format_sweep_x (reuses sweep_x_to_display) shifts a body-frame sweep x by the driver display offset in angle mode, converts to the display angle unit and labels it deg, rad or mm (stroke mode: metres to mm, no offset). Health toggle points and the Exceeding 80% range (gui/property_panel/health.rs) call it instead of printing raw values with a degree sign. Readers checked: has_toggles_in_range still compares in body frame; the plots and SweepData are unchanged.
- Tests: health_x_text_shifts_by_display_offset_and_labels_units (red first: format_sweep_x stubbed to the old format, left 120 deg, right 150 deg). Commit 5d81637.

## 2026-11-16 - The deploy runs the tests before it builds or deploys (audit INF-05; branch audit/2026-11-16-section-a)
- What changed: .github/workflows/deploy-web.yml has three new steps before the wasm-bindgen install and build: cargo test --all (linkage-sim-rs), cargo test and cargo test --features app (magcoupling-rs), gate.sh gates 1, 4 and 7 on the pinned 1.89.0 toolchain and the existing cache. Clippy, the Python oracle and the wasm-bindgen parity gates stay local. docs/guides/WASM_DEPLOYMENT.md said the deploy installs the stable toolchain; it now says 1.89.0 and lists the test step.
- Tests: none possible for CI YAML. The YAML parses (python yaml.safe_load, step order checked) and each added command exits 0 locally.

## 2026-11-16 - A failed open, save or shared-link load shows in the status line (audit LGS-10; branch audit/2026-11-16-section-a)
- What changed: new AppState::report_failure (gui/state/file_io.rs) logs and sets status_message (5 s) to "<what>: <error>". Open JSON, Recent Files, Save, Save As (menu_bar.rs), Ctrl+S, Ctrl+Shift+S and the shared-URL load (gui/mod.rs) call it instead of log::error! alone. Readers checked: recover_autosave_file and recover_wasm_autosave already set a status on failure, left as is; save_to_file/load_from_file still return Err unchanged; the status line reader is unchanged.
- Tests: failed_save_is_shown_in_the_status_line (red first: report_failure stubbed as a no-op, the status stayed empty). Commit 78f4fe4.

## 2026-11-16 - GIF export fails when no frame is written or the driver speed is zero (audit LGE-04; branch audit/2026-11-16-section-a)
- What changed: generate_mechanism_gif_bytes (gui/export/raster.rs) now returns Err when omega is zero or not finite (it divided by omega with no guard) and when no frame was written (frames that failed to solve or rasterize were skipped and an empty GIF came back as Ok). Readers checked: export_mechanism_gif (passes the error on) and the menu_bar.rs GIF export call (already shows an Err); no other callers.
- Tests: export_gif_zero_omega_returns_error, export_gif_no_frames_rendered_returns_error (red first: both guards removed, the no-frames test failed and the zero-omega run did not return).

## 2026-11-16 - Undo and redo refresh the driver display offset and the stroke range (audit LGS-16; branch audit/2026-11-16-section-a)
- What changed: AppState::restore_snapshot (gui/state/undo_ops.rs) now calls recompute_driver_display_offset() and the new shared AppState::refresh_sweep_stroke_range(&Mechanism) (gui/state/mod.rs), as rebuild and load do. Before, undo/redo kept the offset and sweep stroke range of the mechanism that was edited. The inline stroke-range loops in rebuild() (blueprint_ops.rs) and load (file_io.rs) are replaced by the same function. Readers checked: sweep_stroke_min/max and driver_display_offset readers (sweep, animation, input panel) only read; load_sample/rebuild behaviour is unchanged.
- Tests: undo_restores_driver_display_offset, undo_restores_sweep_stroke_range (red first: both failed after undo with the old values).

## 2026-11-16 - Removing a body drops its linear drivers (audit LGS-17; branch audit/2026-11-16-section-a)
- What changed: AppState::remove_body (gui/state/entity_crud.rs) now also retains only the `blueprint.linear_drivers` that touch neither end on the removed body. Before, they stayed and the rebuild failed in Mechanism::add_linear_driver on the missing body. Force elements were already dropped (`attached_body_ids` covers every variant). Readers checked: other linear_drivers users (scale, undo, file_io, driver_ops, animation export) only read or add; emove_attachment_point (point refs, not bodies) was not changed.
- Tests: remove_body_cascades_to_linear_drivers (red first: the driver still referenced the deleted body).

## 2026-11-16 - Names from a mechanism file are escaped in every export (audit LGE-07; branch audit/2026-11-16-section-a)
- What changed: new linkage-sim-rs/src/gui/export/escape.rs holds the one escaper per context. `html_escape` (HTML/SVG text and attributes: & < > " ') replaces report.rs's 4-character `html_escape` and schematic.rs's `xml_escape`, which disagreed. `js_str` (serde_json string literal, every `<` written `\u003c`) writes names inside the report's `<script>` (the reactions and coupler-trace plot names, which were `'{}'` raw). `dxf_text` turns control characters (newline, CR, tab) in a name into `_`, so a layer name cannot split a DXF code/value pair; DXF has no other markup. Now escaped where they were raw: body, attachment-point and joint names in the report's Dimensions, Mass Properties and joint-reaction tables, the body in a force element's detail line, the `\text{}` body names in the derivation (after the LaTeX escape), and body and joint labels in svg.rs. Plotly series titles and sensor names in the report were already static or escaped. Readers checked: html_escape/xml_escape callers (report.rs, schematic.rs) are unchanged in output for ids without a quote (`'` now writes `&#39;`, was `&apos;` in the schematic); the animation page already escapes its names (animation.rs).
- Tests: report_escapes_hostile_names, svg_escapes_hostile_names, dxf_layer_name_cannot_break_group_structure (red first: the raw `<script>` and the split DXF pair); escape.rs unit tests for the three helpers.

## 2026-11-16 - A force on an unknown body is a load error, not a panic (audit LFA-05; branch audit/2026-11-16-section-a)
- What changed: `ForceElement::resolve_named_points` (forces/elements/force_element.rs) indexed the body map (`bodies[&id]`), which panics on a missing key inside a function that returns a Result. It now returns the new `BodyError::UnknownBody`. The loader (io/from_json.rs) returns that as `SerializationError::Build` instead of swallowing it, and runs `validate_force_elements` (previously called only by tests) after the forces are added: a force that names an unknown body without a named point is an error too; the other validator warnings (negative stiffness and so on) are logged only. Readers checked: `resolve_named_points` has no other non-test caller; the point-name-not-found case still logs and keeps the force as before.
- Tests: `named_point_on_unknown_body_is_an_error_not_a_panic` and `unnamed_point_on_unknown_body_is_an_error` (from_json.rs; red first: both loaded Ok, with the lookup fix alone, before the loader change).

## 2026-11-16 - A failed load leaves the app unchanged (audit LGS-12; branch audit/2026-11-16-section-a)
- What changed: in linkage-sim-rs/src/gui/state/file_io.rs, load_from_json_str assigned the sweep mode, sensors, motion profile, rated force, duration, parametric settings and blueprint before the mechanism build, so a file that failed to build left the app half old and half new. The build now runs first (after parsing); the fields are assigned only when it built. Readers checked: the callers of load_from_json_str (file load, autosave recovery, templates, share URL) only act on the Err/Ok; the build reads no AppState field.
- Tests: `failed_build_leaves_settings_unchanged` (file_io.rs; red first: the rated force was overwritten by the failed file).

## 2026-11-16 — A failed autosave recovery keeps the autosave and shows the error (audit LGS-02; branch audit/2026-11-16-section-a)
- What changed: in `linkage-sim-rs/src/gui/mod.rs` the Recover button deleted the autosave (desktop file, browser localStorage) even when loading it failed, with only a log line. The desktop path now calls `AppState::recover_autosave_file` and the browser path `recover_wasm_autosave` (both in `gui/state/file_io.rs`): the autosave is deleted only after a load that succeeded; on failure it stays and `status_message` says so. Discard is unchanged. A kept desktop autosave is offered again at the next start. Readers checked: `check_autosave_recovery` and `check_wasm_autosave_recovery` only test that the autosave exists; nothing else reads `recovery_path` after the prompt.
- Tests: `recover_autosave_keeps_file_and_reports_when_load_fails` (red first: with the delete on the error path it failed) and `recover_autosave_deletes_file_after_successful_load` (gui/state/tests.rs). The browser path is wasm-only and has no native test.

## 2026-11-16 — A failed statics solve is a NaN driver torque in both sweep modes (audit LGE-02; branch audit/2026-11-16-section-a)
- What changed: in `linkage-sim-rs/src/gui/sweep/mod.rs` the forward sweep stored a failed statics solve as a driver torque of 0.0 (trajectory mode stored NaN), so a failed sample read as "no torque needed". Both modes now call `statics_sample`, which gives NaN on a failed solve. Readers of `driver_torques` checked: the plots (gaps), `compute_envelope` (filters non-finite), the CSV and report exports (NaN or null), the weight breakdown, the actuator force and power (NaN in, NaN out) and motion_profile.rs (skips non-finite) already handle NaN as trajectory mode does; `ParametricMetric::RmsDriverTorque` (state/parametric.rs) turned the whole sweep NaN, so it now takes the RMS of the solved samples.
- Tests: `failed_statics_solve_is_a_nan_driver_torque` (gui/sweep/mod.rs) and `rms_driver_torque_metric_skips_failed_samples` (gui/state/tests.rs). The statics solve only fails on an unbuilt mechanism or an SVD failure, which no sweep fixture reaches, so the test calls `statics_sample` with a failed solve.

## 2026-11-16 — Joint reaction series stay aligned with the angle axis when early sweep samples fail (audit LGE-01; branch audit/2026-11-16-section-a)
- What changed: in `linkage-sim-rs/src/gui/sweep/mod.rs` a joint's reaction series was created at the first sample that solved, so a sweep whose first samples fail left it shorter than `angles_deg` and every later reaction was plotted and exported against the wrong angle. The new `push_reaction` pads a new series with NaN up to the current sample; the forward sweep and trajectory mode both use it.
- Test: `sweep_reaction_series_stay_aligned_when_first_sample_fails` (a 4-bar whose 0 deg sample cannot assemble).

## 2026-11-16 — Owner approval of the off-default fixes (BL-058, BL-061, BL-062, BL-063, BL-074; branch magcoupling/off-default-fixes)
- What was approved: the owner answered the list of items waiting on them with "Approve and yes. Stop asking for my direction and do what you think is best". That approves the diffs of BL-058 (E22), BL-061 (E25), BL-062 (E26), BL-063 (the second daily-cycle screen) and BL-074 (the Overall length first-move documentation) on this branch, and the conservative answers BL-061, BL-062, BL-063 and BL-074 were built on (taken under the ruling of 2026-11-10; register docs/analyses/2026-11-10-conservative-assumptions.md). BL-067 is not part of it and stays open. Nothing is merged or pushed yet.
- What changed in this pass: `Approval::Assumed { item }` is now `Approval::Owner { item }` (engine/deviations.rs); E25's and E26's evidence reads "docs/ai/backlog.yaml, item BL-06x: approved by the owner on 2026-11-16" and their `corrected_formula` strings start "Approved by the owner on 2026-11-16"; `every_entry_is_approved_in_its_report` (tests/deviations.rs) now checks that the backlog entry of an `Owner` item holds the sentence "Approved by the owner on 2026-11-16". The on-screen suffix " (an assumption, not approved)" is gone (the const `ASSUMED` and `listed_title` in gui/corrections.rs are removed; the lists and hovers show the registry title; the Summary's heading "Corrections applied (every approved correction is on)" is true again). The explanation table label of `grades.magnet_sigma_S_m` reads "(E26; none for sintered NdFeB)". Docs, comments, README and tests that called these items unapproved say they were approved on 2026-11-16. docs/ai/backlog.yaml: BL-058, BL-061, BL-062, BL-063 and BL-074 are `fixed`, each note ending with the approval.
- Not changed: the engine's own "assumptions" feature (engine/assumptions.rs, the Assumptions banner) is a different thing; the older tracker entries below keep the words of their day.

## 2026-11-11 — Magcoupler E26: each ring's magnets read a conductivity by their grade's family in the eddy heat, an assumption that is not approved (BL-062, branch magcoupling/off-default-fixes)
- Why: `temperature::compute` took the eddy heat inside the magnets (Temperature design!C129, `temperature.slip_loss.magnets_W`) at the one input C113, sintered NdFeB's conductivity (6.7e5 S/m), for both rings whatever their grade, as the workbook does. Manual dimensions read 0.2278 W with every one of the 17 grades. SmCo conducts more than NdFeB, so its heat was understated, the side that is not safe; ferrite and bonded NdFeB conduct far less. The research of 2026-11-09 (`docs/analyses/2026-11-09-magcoupling-bl062-conductivity-research.md`, untracked in the main checkout and not on this branch) lists the printed values and leaves the choice to the owner. Under the owner's ruling of 2026-11-10 on open questions the choice is made for now: its options 1B, 2C, 3B, 4B and 5B, an assumption (register row A-8), not the owner's answer. The other answers: the lower printed values of each family (2A, 3A, 4A, 5A), or a higher sintered NdFeB value (1A, 1C), which moves the default example.
- What changed: correction E26 (`DeviationId::E26`; `Deviations::ALL` holds 26), held as `Approval::Assumed { item: "BL-062" }`. `GradeFamily::magnet_sigma_S_m` (engine/grades.rs) gives one value per family, stated once and cited in the code with its printed resistivity: Sm2Co17 1.54e6 S/m and SmCo5 2.5e6 S/m (Vacuumschmelze's low-resistivity ends, 0.65 and 0.4 Ωmm²/m), sintered ferrite 1.0e2 S/m (MS-Schramberg's ">10^4" in 10^-6 Ωm: a maker's upper bound on the conductivity, not a measurement) and bonded NdFeB 2.5e4 S/m (Bunting BREMAG, 4 x 10^-3 Ωcm); none for sintered NdFeB. The values are Rust-only, in no data file, and do not pass through C113's range check (1e5 to 2e6 S/m). `temperature::magnet_sigma_S_m` gives a ring its family's value, else the input, and `p_mag` reads `(sigma_inner + sigma_outer) / 2.0` where it read the input: both rings are taken as the inner block (the workbook's simplification, kept), so each ring's half at its own conductivity is the mean of the two. Two Rust-only results show what each ring reads, `temperature.slip_loss.inner_magnet_sigma_S_m` and `outer_magnet_sigma_S_m`, after the cap's conductivity. Manual dimensions with the grade on both rings, on the workbook basis: Y30 0.2278 -> 3.401e-5 W, Recoma 26 0.5237 W, Recoma 20 0.8502 W, BCN-19 0.008502 W; Y30 inside N42 0.1139 W, half of each pair's. With every correction on, the high-case steady temperature of a Recoma 20 pair goes 92.51 -> 98.74 °C and of a Y30 pair 92.51 -> 90.23 °C; no verdict text moves on the five probe designs. The Python engine is not changed.
- Not changed: the stored field inside the blocks (C122, `b_magnet_T`) and the block size, which still do not follow the grade or the ring, so the heat is exact for no grade; the total's sum; a measured drag, which still replaces the estimate (E26 then moves C129, C130 and C131 alone); the input's label "NdFeB conductivity", its symbol, its range and its empty engine help.
- Bit for bit: the default design (both rings are sintered NdFeB library parts, and `(s + s) / 2.0` is `s`), every library part, every sintered NdFeB grade, manual dimensions with no grade, and every differential case, each on both bases (E26 alone against none; all against all less E26). With E26 off every grade reads 0.22784639081300484 W. No blessed, golden, schema or differential data file changed.
- Words that follow: the equation record of `temperature.slip_loss.magnets_W` reads the mean of the two new results; each has its own record (the family's value from the grades table's new `magnet_sigma_S_m` field, else the input); all three are marked E26. The explorer's slip-heating chain lists the two (196 paths, was 194), so the results table shows them under Slip heating. The overlay help of `temperature.slip_loss.sigma_ndfeb_S_m` said "Used whatever the magnet grade, which overstates the heat in ferrite"; it now reads "Electrical conductivity of sintered NdFeB magnets: their eddy-current slip heat rises with it. A ring with no grade uses it; SmCo, ferrite and bonded NdFeB rings use their family's value." Its claims: up on the example and on manual dimensions with no grade, no move on the Y30 design (was up). `picks::set_by_pick` names the input as set where both rings have no library part and a SmCo, ferrite or bonded NdFeB grade (named for the inner ring's grade); with one such ring the input still moves its half. The hover of a marked value and the spreadsheet Summary end E26's title "(an assumption, not approved)", as E25's; the 40 cells E26's probes move carry its mark with the spreadsheet details on. The `slip_heating` and `when_slipping` notes were re-read against the new behaviour and no sentence changed: their watch-outs speak of the stored fields, which still do not follow the grade. Both stay reviewed. The comments that named E25 as the one correction not approved now name E25 and E26. Counts: 1,120 result rows (was 1,118), 421 of them explained (was 419).
- Tests: tests/magnet_conductivity.rs (new): `a_ferrite_pair_heats_far_less_than_an_ndfeb_pair_and_a_smco_pair_more` (red first on the old code, where the Y30 pair read 0.22784639081300484 W), all 17 grades, the mixed pair as the mean in either order, every library part and no grade on the input, the input's reach (an NdFeB pair in proportion, no result at all of a Y30 pair, a mixed pair by its NdFeB half), the measured drag, and the two Rust-only rows. tests/deviations.rs: E26's five probes (`each_probe_shows_its_correction`, `addendum_entries_name_every_cell_their_probes_change`), `e26_leaves_every_default_cell_bit_for_bit`, `e26_moves_only_a_ring_of_another_family` (every grade in three placements and every library part, on both bases, each moved case equal to E26 off with C113 set to the mean of the two rings' conductivities), `e26_off_gives_every_grade_the_input_s_heat` (the backlog's probe), `e26_reaches_no_result_past_the_estimate_when_a_drag_is_measured` and `e26_moves_no_differential_case`. Unit tests: engine/grades.rs `each_family_s_magnet_conductivity_is_the_one_taken_for_bl062` (the five answers, each against its printed resistivity), engine/picks.rs (the pick named, and no input a pick sets moves a result), engine/explain/tables.rs. tests/explain.rs: the drift guard holds the three records at the Y30 probe design and at the mixed-grade augmentations, and `the_slip_losses_read_fields_stored_from_the_example_design` now holds the Y30 design to the example's loss in every part but the magnets (it held the totals equal). gui tests of the help's claims, the two titles, the Summary and the trace's count. Mutation checks, each restored after with the file's hash compared: both rings reading the input again under E26 turned 13 tests red (the red test and five more in tests/magnet_conductivity.rs, `each_probe_shows_its_correction`, `e26_moves_only_a_ring_of_another_family`, `e26_reaches_no_result_past_the_estimate_when_a_drag_is_measured`, the drift guard, the stored-fields test, the picks test and the help's claims); the mixed pair reading only the inner ring's conductivity turned 5 red (`a_mixed_pair_reads_the_mean_of_its_two_rings`, `the_input_moves_an_ndfeb_pair_s_heat_and_leaves_a_ferrite_pair_s_alone`, `e26_moves_only_a_ring_of_another_family`, `each_probe_shows_its_correction` and the drift guard). `cargo test` with and without `--features app`, clippy both ways and `cargo fmt --check` pass. The builder did not run the full gate.
- Noted, not changed: the research file's 0.5232 W for Sm2Co17 is 1 / 0.65e-6 unrounded, and the code holds the register's 1.54e6 S/m, 0.5237 W; its other figures agree. Outside the hover of a marked value (with the spreadsheet details on) and the spreadsheet Summary, nothing on screen says the family values are an assumption: the two new rows have no cell and so no mark. The Summary's heading "Corrections applied (every approved correction is on)" now stands over E25 and E26 (older: the text review of E25). The grade picker's hover lists no conductivity. BL-062's evidence in the backlog cites a line of temperature.rs that has moved.
- Reviews: fresh-context general-purpose agents on fable. The physics and code review did not approve in round 1 for one stale README count (194 scope paths where the test holds 196) and asked for the wording "of the values the research kept" beside each value's "highest printed"; both were applied, and round 2 approved with its own old-against-new dump (3,720 runs, every pre-existing row byte for byte), the conversions redone by hand (vendor sites not reopened), and a skin-depth check that the widest block the sliders allow stays under the turn, so the higher conductivity stays the hot side. The G-3 text review passed in round 2 with no must-fix; its record is in `docs/analyses/2026-10-14-magcoupler-help-review.md`. Its three should-fix items are left for the owner, not applied: SF-1 ("A ring with no grade uses it" can read as the only ring on the input), SF-2 (the spreadsheet Summary heading "Corrections applied (every approved correction is on)" over E25 and E26) and SF-3 (this entry's "Noted, not changed" line leaves out the Equation panel's "Embodies corrections" hover, which also shows the "not approved" title). GATE PASS on 2026-11-16, all 13 gates, none skipped.
- Still owed before the branch merges: the owner's review of the diff (`risk: physics`, AGENTS.md) with the owner's own answer. BL-062 is in_progress, not fixed.

## 2026-11-11 — Magcoupler: a second daily-cycle bond screen at the entered bondline, an assumption that is not approved (BL-063, branch magcoupling/off-default-fixes)
- Why: the workbook's daily-cycle screen (Temperature design!C201, C202) reads the bond shear at the recommended bondline (0.1 mm by default), not at the bondline entered as the adhesive thickness under the inner magnets (`metal.bond_inner_mm`, 0.05 mm by default). A thinner entered bondline shears more and passed the screen unseen: the default design read "Below the fatigue endurance" at 2.563 MPa, while the entered bondline's shear scales to 4.932 MPa, above the 3.0 MPa limit (AA 326: lap shear 15 MPa times the 0.2 endurance fraction). The owner's question (judge the entered bondline, or report both?) is open. Under the owner's ruling of 2026-11-10 on open questions it is answered "report both" for now: an assumption (register row A-9), not the owner's answer. The other answer: one screen at the entered bondline, as a registered deviation of C201 and C202.
- What changed: two Rust-only results right after the workbook's two, with no cell: `temperature.adhesive_life.daily_entered_peak_shear_MPa` (`temperature.mismatch.peak_shear_current_MPa` times the daily swing over the worst swing, as C201 scales the recommended bondline's shear) and `temperature.adhesive_life.daily_entered_screen` (C202's strict comparison on that shear; `temperature::compute` holds the comparison once for both screens, so E22's size comparison covers both). The default design now reads "Above the fatigue endurance: qualify by thermal cycling" on the new screen and stays "Below the fatigue endurance" on the workbook's; at an entered bondline of 0.2 mm the new screen reads below (1.307 MPa). Where the worst swing is 0, both shears are 0 times the swing over 0, not a number, and both screens read above: the workbook's behaviour, not changed. In the GUI the new screen is a design check (`gui::dashboard::CHECKS` holds 33) with the workbook screen's badge rule (below green, above amber); it sits in the adhesive group of the results table and the failing filter lists it. No correction is registered and the Python engine is not changed: no ported cell moves.
- Bit for bit: C201, C202 and every other result row. A scratch dump of 7,998 runs (3,391 differential cases, every ordered pair of grades with and without back iron, 30 pairs of the two bondlines, each with the corrections on and off) is byte for byte the same before and after: C201's bits, C202's text and a hash of the other 1,116 rows of each run. No blessed, golden, schema or differential data file changed.
- Words that follow: the help of `temperature.mismatch.recommended_bondline_mm` names both screens ("The daily-cycle screen reads it, the entered-bondline screen the thickness under the inner magnets."), and the help claims gain two rows (a thicker entered bondline lowers the new shear; the recommended bondline leaves it). Counts: 1,118 result rows (was 1,116); 33 checks (was 32), 24 of them off the dashboard; the default design lists six failing checks, five red and this amber one (was five), and the defaults on manual magnets eight (was seven). Nothing was added where the workbook's screen has nothing: no guide or Problem (the dashboard's status line does not count either screen), no equation record, no teaching note, no correction marker, and the temperature verdict reads neither screen.
- Tests: tests/daily_screen.rs (red first on the old code: "no result temperature.adhesive_life.daily_entered_peak_shear_MPa"), `the_default_design_is_above_the_endurance_at_the_entered_bondline_and_below_it_at_the_recommended_one` and `each_screen_reads_its_own_bondline_over_a_spread_of_bondlines_and_swings` (60 designs: both formulas bit for bit, the recommended bondline does not move the new shear, equal bondlines give equal shears and screens, all four pairs of readings reached); engine/temperature.rs `daily_entered_screen_is_above_at_equal_shear` and `both_daily_screens_read_above_when_the_worst_swing_is_zero`; tests/deviations.rs, where the two `e22_compares_the_size_of_a_negative_shear` tests assert the new screen too; gui tests of the badge levels, the failing lists (dashboard, panel and results table), the group, the trace's count (419 of 1,118 results explained) and the help's claims. Mutation checks, each restored after: the recommended bondline's shear fed to the new screen turned three tests red (both in tests/daily_screen.rs and `daily_entered_screen_is_above_at_equal_shear`); the comparison made non-strict turned `daily_screen_is_above_at_equal_shear` and `daily_entered_screen_is_above_at_equal_shear` red. `cargo test` with and without `--features app`, clippy both ways and `cargo fmt --check` pass. The builder did not run the full gate.
- Reviews: fresh-context general-purpose agents on fable. The physics and code review approved in round 1 with its own old-against-new probe (2,812 runs beyond the defaults, every pre-existing row bit for bit) and a mutation that turned the new tests red. The G-3 text review passed with no must-fix over 39 probed designs; its record is in `docs/analyses/2026-10-14-magcoupler-help-review.md` (run 2026-11-16 after a resume, dated 2026-11-11 as prescribed). Its four should-fix items are left for the owner, not applied: SF-1 (the workbook's "Daily-cycle screen" row names no bondline and has no help), SF-2 (the changed help's second sentence drops its verb), SF-3 (one thickness has three names on screen) and SF-4 (the daily swing help says "the screen" where there are now two). Recorded too: nothing on screen marks the second screen as an assumption, and with every correction off it compares the signed shear, as the workbook's screen does. GATE PASS on 2026-11-16, all 13 gates, none skipped.
- Still owed before the branch merges: the owner's review of the diff (`risk: physics`, AGENTS.md) and the owner's own answer. BL-063 is in_progress, not fixed.

## 2026-11-11 — Magcoupler E25: the space claim's diameter judges the shaft boss too, an assumption that is not approved (BL-061, branch magcoupling/off-default-fixes)
- Why: `metal_design::compute` took the rotating diameter (Metal design!C135) as the larger of the cup and the cap, as the workbook does, while the shaft boss turns with them and the Size tab's Outside diameter counts it. BL-061's evidence design (the default at a 7 mm ring radius, a 40 mm boss, a 38 mm claim) read "Inside the space claim" with a 40.0 mm outside diameter. The owner's question (does the boss count toward the claim?) is open. Under the owner's ruling of 2026-11-10 on open questions it is answered yes for now, the larger envelope: an assumption (register row A-7), not the owner's answer. The other answers: the boss named apart from the claim, or left out as in the workbook.
- What changed: correction E25 (`DeviationId::E25`; `Deviations::ALL` holds 25). With it on, `rot_od` is the largest of the cup, the cap and the boss, and the reserve C136, `housing.diameter_overshoot_mm` and the space claim check follow. The evidence design now reads "Exceeds the space claim: diameter 2.00 mm over" (C135 36.18 -> 40.0 mm, C136 1.824 -> -2.0 mm). The registry does not call it approved: the new `Approval::Assumed { item: "BL-061" }` gives `BACKLOG` (docs/ai/backlog.yaml) as its report and "an assumption, not an approval" in its evidence. The Python engine is not changed.
- Bit for bit: the default design (boss 22 mm under the 42.8 mm cap), every differential case, and on the evidence design a boss below or equal to the cap, each on both bases (E25 alone against none; all against all less E25). Moved, where the boss is the largest: C135 and C136 and no other workbook cell. No blessed, golden or schema file changed.
- Words that follow: the equation record of `metal.rotating_od_mm` (the largest of the three, marked E25); the help of `housing.diameter_overshoot_mm`; the GUI overlay texts of `metal.max_diameter_mm` and `metal.boss_od_mm`; the guidance's diameter axis of the space claim, which now offers "Lower Shaft boss outside diameter" where the boss sets a diameter past the claim (without it that design has no fix to show); the hover of a marked value and the spreadsheet Summary's list of corrections, where E25's title ends "(an assumption, not approved)" (`gui::corrections::listed_title`); and the comments that called every correction approved, which now name E25 as not approved. The engine help of `metal.boss_od_mm` is the workbook's empty help and stays empty (a change would re-bless tests/data/input_schema.json); the overlay row carries the behaviour.
- Tests: tests/sizing.rs `a_boss_larger_than_the_cup_and_cap_and_past_the_claim_fails_the_diameter_axis` (red first on the old code: 36.17578798729891 against 40.0) and `the_size_tab_s_outside_diameter_is_the_diameter_the_space_claim_judges`; tests/deviations.rs E25's probe (`each_probe_shows_its_correction`), `e25_probe_is_the_backlog_item_s_evidence_design`, `e25_leaves_every_default_cell_bit_for_bit`, `e25_moves_only_a_boss_larger_than_the_cup_and_cap` and `e25_moves_no_differential_case`; tests/explain.rs two edge points of the drift guard on the defaults, a shaft boss of 46 mm and of 44 mm, both above the 42.8 mm cap, each asserted to be the rotating diameter (the guard reads the record's boss term only where the boss is the largest, and its two-values rule needs both points: with one, `every_record_reproduces_the_engine_everywhere` fails with "metal.rotating_od_mm reads metal.boss_od_mm only as Num(46.0)"); gui tests of the boss fix, of the Problem sentence ("Rotating diameter 46.00 mm is 3.000 mm over the 43.00 mm claim.") and of the two lists' words. Mutation checks, each restored after: the cup-and-cap maximum put back under E25 turned 10 tests red (the two in tests/sizing.rs, the probe and `e25_moves_only_a_boss_larger_than_the_cup_and_cap`, the drift guard at its 46 mm point, four in gui/guidance.rs and the plain_text claims test); the equation record put back to the cup and the cap turned the drift guard red at both points ("record Num(42.8), engine Num(46.0)" and "record Num(42.8), engine Num(44.0)"). `cargo test` with and without `--features app`, clippy both ways and `cargo fmt --check` pass. The builder did not run the full gate.
- Reviews: fresh-context general-purpose agents on fable. The physics and code review did not approve in round 1 for one test gap (nothing pinned the boss term of the equation record of `metal.rotating_od_mm`); the rework added the two edge points named above, and round 2 approved with its own probe of `compute_all` and both mutations rerun. The G-3 text review of the changed on-screen texts passed with no must-fix; its record is in `docs/analyses/2026-10-14-magcoupler-help-review.md`. Its three should-fix items are left for the owner, not applied: SF-1 (the spreadsheet Summary's heading "Corrections applied (every approved correction is on)" stands over a list that now holds E25), SF-2 (the guidance's "No listed change helps from here" line does not name an exact tie of two diameters past the claim; older than this item) and SF-3 (no test pins that the three changed helps name the shaft boss). The physics review also names two things E25 does not reach: the sweep tables (`engine/sweeps.rs` judges the cup alone against the claim, as the workbook does) and the large-diameter length of the space claim, which leaves the boss length out. GATE PASS on 2026-11-11, all 13 gates, none skipped.
- Still owed before the branch merges: the owner's review of the diff (`risk: physics`, AGENTS.md) and the owner's own answer. BL-061 is in_progress, not fixed.

## 2026-11-10 — Magcoupler Size tab: the Overall length slider's first move from a blank override is documented and pinned (BL-074, branch magcoupling/off-default-fixes)
- Why: `engine::size_controls::resize_length` sets both rings' axial override to `requested - length_range.min + axial_range.min`, and with a blank override the length range follows the outer ring the cup cavity holds. So the slider's first move sets both rings to one length: the step in the ordinary case, not the step where the ring lengths differ or the cup cavity is shorter than the outer ring. The torque card already reads words there; the slider did it unannounced. Under the owner's ruling of 2026-11-10 on open questions the jump is treated as intended: an assumption (register row A-11), not the owner's answer. The other answer is a mapping change in `engine/size_controls.rs`, chosen with the owner.
- What changed: the Overall length hover (`gui::size_controls::LENGTH_HOVER`) now says that while the axial magnet length is blank the first move sets both rings to one length, and that with rings of unequal length, or a cup cavity shorter than the outer ring, that move can change the magnet length by more than the step, or the other way, and the torque with it. The hover is 4 sentences and 75 words, over spec 3.5's help caps (2 sentences, 30 words), so the caps test gives it its own cap (4 sentences, 80 words). The README's Size-tab paragraph and docs/ai/02-system.yaml say the same with the numbers. No engine line changed (`git diff -- magcoupling-rs/src/engine` is empty) and no engine result moves.
- Tests: `tests/size_controls.rs` pins one step each way on four designs, each with both ring lengths and the hot-low torque: the default (12.7 -> 11.7 / 13.7 mm, hot-low 2.2852 -> 2.0844 / 2.4860 N·m), a B442 inner part in a B842SH outer part (both rings 11.7 / 13.7 mm, hot-low 1.0099 -> 2.0844 / 2.4860 N·m, +106.39 % / +146.16 %; the hub follows the inner ring from 13.0 to 18.35 / 20.35 mm), the reverse pair (the longer inner ring trimmed from 12.7 to 5.35 / 7.35 mm, hot-low 1.0099 -> 0.8091 / 1.2107 N·m) and an 11.0 mm cup cavity (both rings 12.7 -> 10.0 / 12.0 mm, hot-low 2.2852 -> 1.7430 / 2.1446 N·m, down both ways). The numbers equal BL-074's evidence. The hover test (`the_length_hover_says_the_first_move_from_a_blank_length_sets_both_rings_to_one_length`) failed first on the old text. Mutation check, restored after: the override counted from the shorter ring turned the mixed-pair and short-cavity pins red; 0.5 mm added to the mapping turned all four red.
- Noted, not changed: the first mutation shows that the backlog's example of the other answer (the override counted from the shorter ring), alone, would not keep the overall length at the requested value. The Outside diameter slider still does its boss-plateau jump unannounced (slice-3 spec R-13); the torque card reads words there.
- Reviews: a fresh-context physics and code review (general-purpose agent on fable) approved in round 1, with its own probe of `compute_all` and four mutations of its own. The G-3 text review of the hover (general-purpose agent on fable) passed with no must-fix; its record is in `docs/analyses/2026-10-14-magcoupler-help-review.md`. Its should-fix SF-1 (the hover could read "by more or less than the step") is left for the owner, not applied. GATE PASS on 2026-11-11, all 13 gates, none skipped.
- Still owed before the branch merges: the owner's review of the diff (`risk: physics`, AGENTS.md) and the owner's own answer. BL-074 is in_progress, not fixed.

## 2026-11-10 — Open questions answered conservatively for now (owner ruling; BL-061, BL-062, BL-063, BL-067, BL-074; branch magcoupling/off-default-fixes)
- Why: the owner ruled on 2026-11-10: "Any open questions choose the conservative answer for now and make those assumptions known somewhwre". Five backlog items waited on an owner answer before any work could start.
- How conservative is read: the side safe for the design verdict (more load, more heat, less strength, a larger envelope, a stricter check); where options are equally safe, the one that changes less and can be undone; the workbook-default example never moves on an assumption. The ruling is not an approval: merging, pushing, the owner's review of a `risk: physics` diff and approval of a design document still wait for the owner.
- Answers taken, each written into its backlog entry's acceptance and notes as an assumption the owner can overrule: BL-061 the shaft boss counts toward the rotating diameter claim (it enters `rot_od`); BL-062 the research report's options 1B, 2C, 3B, 4B and 5B (sintered NdFeB keeps 6.7e5 S/m; Sm2Co17 1.54e6, SmCo5 2.5e6, ferrite 1.0e2 and bonded NdFeB 2.5e4 S/m as Rust-only grade data); BL-063 report both (the workbook's screen at the recommended bondline kept cell for cell, a second Rust-only screen at the entered bondline); BL-067 option (b), the gearbox check and the shaft-joint check compare the larger of cold-high and the operating pull-out with its allowance; BL-074 the first-move jump is treated as intended, documented and pinned by tests, the mapping unchanged. BL-068's SF-1 stays at option A.
- Register: docs/analyses/2026-11-10-conservative-assumptions.md (untracked in the main checkout on 2026-11-10; rows A-1 to A-16) lists every assumption made under the ruling, why it is the conservative side and what changes if the owner rules otherwise. docs/ai/04-memory.yaml carries the short form.
- Docs only: no code changed and none of the five items is built yet.

## 2026-11-09 — Magcoupler correction E22: the bond screen and the expansion warning read the inner ring's own expansion (BL-058, branch magcoupling/off-default-fixes)
- Why: the owner approved the Addendum A-4 corrections on 2026-10-01 (docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md, section 5, option A on all 10 decisions); E22 had not landed. The bond's thermal-mismatch screen (C104 to C106, C201, C202) and warning rule 6 gave every magnet sintered NdFeB's expansion across the magnetization (C95, −0.8e-6 /°C), so a ferrite or SmCo inner ring read an overstated shear and a false caution. Plan: docs/superpowers/plans/2026-10-01-magcoupling-addendum-a4-corrections.md, Tasks 2 and 3, re-anchored to today's code, the CTE half only.
- E22 (decisions 4 to 7, 9 and 10): `Grade::bond_plane_cte_per_C` holds the data file's values (Y30 10.0e-6, Recoma 20 14.0e-6, Recoma 26 and Recoma 30 13.0e-6 /°C; each cited; none for sintered or bonded NdFeB). `temperature::bond_plane_cte_per_C` gives the inner ring's grade value on E22, else the C95 input, and both consumers read it: the Volkersen screen and warning rule 6 (through the new Rust-only `temperature.mismatch.magnet_cte_per_C`; the results table now has 1,116 rows, 419 explained). Only the inner ring is read: the hub bonds it, and the cup bond is not screened. Under E22 the screens C106 and C202 compare the shear's size, the signed values kept on display (decision 9); that also reads a sintered NdFeB design at one slider corner (the steel expansion input below C95, plan decision A4-4).
- Registry: E22 is `Applied` (cells C104, C105, C106, C201, C202; the report's probes E22-1, Recoma 26 on both rings: C104 64.07 -> -3.424 MPa and C106 reads Below, and E22-2, Y30: C104 64.07 -> 11.25 MPa and C202 reads Below, at full precision); C95's workbook help ("Across the magnetization.") is kept there. E23 (each ring's grade specific heat) is not in this change: it stays `Planned` and is now BL-076.
- Bit for bit: the default design, every library part, every sintered and bonded NdFeB grade, a grade on the outer ring only and every differential case, on both bases. Moved: Y30 and the three Recoma grades on the inner ring, where the result equals the workbook screen with the grade's value typed into C95.
- Words that follow: C95's help and the GUI overlay's, the help of warning rule 6 ("the inner ring magnets'"), and `picks::set_by_pick`, which now names the NdFeB expansion input as set by a ferrite or SmCo inner grade on custom dimensions (the input moves nothing there). `tests/data/input_schema.json` re-blessed (C95's help, one line). The G-3 text review of 2026-11-10 then changed four texts, each applied as the reviewer wrote it: the overlay help of `materials.steel.cte_per_C` (the one must-fix: "so higher raises it" was false of the shear's size on a SmCo inner ring over a steel hub; it now reads "the further from the magnets' expansion, the higher the bond shear from temperature swings"), the overlay help of `temperature.mismatch.ndfeb_cte_per_C` (it names "the bond shear from temperature swings" again), C95's engine help (the input is sintered NdFeB's value, used by every library part, every NdFeB grade and a manual magnet without a grade; `tests/data/input_schema.json` blessed again) and the help of `temperature.mismatch.magnet_cte_per_C` ("That ring is bonded to the hub.").
- Three notes reworded, then re-reviewed and signed off (plan decision A4-2; `Review::Reviewed`, 2026-11-09): `a5.cte_mismatch_with_magnets` (the caution fires for NdFeB on 304, 6061 or no back iron; ferrite and SmCo inner rings use their own expansion and fire it on no listed hub; the watch-out on the one NdFeB value is replaced by one on bonded NdFeB, a magnet with no grade and the negative shear), and two the plan did not name. `a5.uncoated_low_alloy_steel`: its back-iron entry said 304 "fires the expansion caution", false for a ferrite or SmCo inner ring, and now says "with NdFeB". `a5.non_ferromagnetic_back_iron` (found by the item's review): its last sentence said the expansion-mismatch caution fires with the open-circuit warning, false for the same rings, and now says "with NdFeB magnets". While the three were Draft they were hidden and these tests were red: tests/explain.rs `release_notes_are_reviewed`, `every_note_names_its_committed_review_section`, `every_note_s_review_date_is_recorded_in_its_section`; with `--features app`, gui `the_warnings_that_fire_are_listed_with_their_reviewed_notes`, `a_warning_shows_in_its_colour_and_links_to_its_note`, `the_closest_note_link_opens_that_note`, `every_teaching_note_shows_with_glyphs_in_the_default_fonts`, `the_card_never_opens_or_closes_under_a_drag` (the dashboard does not widen without a warning's Why link). The physics review then wrote a record under each note ("BL-058 E22 re-review (2026-11-09)", docs/analyses/2026-10-14-magcoupler-notes-review.md; each PASS for the note's text) and the three are marked reviewed under that date: all eight pass.
- Reviews, done. Physics: the campaign's fresh-context physics reviewer, a general-purpose agent on fable and not the fbd-math-reviewer agent that AGENTS.md names, two rounds; round 1 found the open-circuit note's sentence, and round 2 left one reason open, the help round, which the author cannot close. Notes: a record under each of the three notes, each PASS, and the sign-off. Help (G-3, docs/analyses/2026-10-14-magcoupler-help-review.md, `## BL-058 E22 help texts`): round 1 on 2026-11-10 read the four changed texts and the six unchanged overlay rows that speak of the same shear, and returned MUST_FIX, one (`materials.steel.cte_per_C`), with three should-fix; the four New texts were applied as written; round 2 on 2026-11-10, a fresh context, returned PASS, no must-fix and no should-fix.
- Recorded by the help reviewers, no change asked (each could be a backlog item of its own; none is filed): with no back iron or a 304 hub the two steel rows (`materials.steel.cte_per_C`, `materials.steel.modulus_GPa`) still say "the steel hub" and move no shear, as before this branch; no on-screen text says what a negative peak end shear means (a SmCo inner grade on any steel hub shows one, -0.7004 MPa on the default hub); no direction-table row pins "the further, the higher" past the crossing (the engine side is pinned by `e22_compares_the_size_of_a_negative_shear_on_sintered_ndfeb_too`); a grade text the table does not hold ("y30", "Ferrite") reads as no grade and the input is used; the trap list of the slice-2 plan still calls "the expansion screen uses each grade's own expansion" false, and after E22 it is true for a ferrite or SmCo inner ring and for no other. Not checked by either help round: the GUI was not run, and the grade table's expansion values were not compared with their datasheets.
- Still owed before the branch merges: the owner's review of the diff (AGENTS.md, a `risk: physics` item). Until then BL-058 is `in_progress`, not `fixed`.
- Red tests first: tests/grades.rs `bond_plane_expansion_equals_the_a4_data_file`; tests/deviations.rs `e22_leaves_every_default_cell_bit_for_bit`, `e22_moves_only_a_ferrite_or_smco_inner_ring`, `e22_reproduces_the_report_s_bond_screen_table` (the report's 16 rows), `e22_compares_the_size_of_a_negative_shear`, `e22_compares_the_size_of_a_negative_shear_on_sintered_ndfeb_too`, `e22_moves_no_differential_case`; temperature.rs `e22_reads_the_inner_ring_s_grade_cte_and_compares_the_shear_s_size`; warnings.rs `a_ferrite_or_smco_inner_ring_fires_the_expansion_rule_on_no_listed_hub` (the note's claim, over the four grades, eight back irons and both circuits), and `the_open_circuit_note_says_the_expansion_caution_fires_with_ndfeb_magnets` (the open-circuit note's last sentence, at every probe design, each back iron and both circuits; red on the old sentence).
- Left as found: the A-4 data file's status strings still read "pending" (`meta.status` and each grade's `engine_literals.status`), though the report's section 5 and docs/ai/04-memory.yaml record the approval.

## 2026-11-09 — The hot-minimum Problem never reads two equal numbers on the sliders (BL-068, branch magcoupling/off-default-fixes)
- Why: where hot-low torque was below the requirement by less than half a last digit, the Problem quoted both at four significant digits and read alike ("Hot-low torque 0.9300 N·m is below the required 0.9300 N·m": an axial magnet length of 4.74 mm, a face gap of 0.5 mm and a required torque of 0.93 N·m, each on its slider).
- Fix: `guidance::hot_min_problem` quotes hot-low through `quoted_limit` (rounded down at its last shown digit, one digit lower where that still reads equal) when the two would read the same: 0.9299 against 0.9300 N·m. Every other design reads as before (the defaults: 2.285 against 2.500 N·m); a hot-low that is not positive or not finite is quoted as it is. The mirror of the gearbox Problem fix (6cbc852).
- Tests (red before the fix): `the_hot_minimum_problem_never_reads_two_equal_numbers` and `on_the_sliders_the_hot_minimum_problem_quotes_hot_low_below_the_requirement`. README: the Check guides paragraph and the guidance.rs test row. BL-068 is fixed; its changed sentence passed the G-3 text re-review in a fresh context on 2026-11-09 with no must-fix (`docs/analyses/2026-10-14-magcoupler-help-review.md`, "Re-review after BL-068").
- Audit (read-only): the clearance and cup wall Problems read two equal numbers on no slider-step design probed (every step pair of shaft displacement by clearance target, and of axial length by cup wall); the space claim Problem quotes three numbers and read equal on none of the single-slider sweeps of its three claim sliders (it was not probed on two sliders at once); no text changed.
- Follow-up (2026-11-10, the controller's review of 1065264): `docs/ai/backlog.yaml` did not parse at 1065264 (BL-068's notes had no closing quote and pointed at "see AUDIT"); the string is closed and carries the audit. The re-review's SF-2 and SF-3 are folded in: the first test also pins that hot-low reads as its readout where the two read apart ("0.9300 ... 0.9400" at a requirement of 0.94 N·m), and `hot_min_problem` calls `apart_from` in place of a copy of its branch (the text drawn is unchanged). SF-1 (a decade edge only a file's values reach, "0.9990 ... 1.000") is the owner's choice and stays as recorded. The README says a hot-low "at or below zero" is quoted as it is (it said "or not a number", which never shows: the check is not red then). Not merged: the branch waits for the owner's OK.

---

## 2026-11-10 — Whole-project audit: 27 sure fixes on a branch, 168 findings listed for the owner (branch audit/2026-11-10-whole-project)
- Why: the owner asked: "Read this entire project and find everything that's broken, inconsistent, inefficient, or no longer needed. Fix what you're sure about and list the rest for me to review."
- How: 15 finder agents by area and 8 skeptic agents (Sonnet, read-only, no cargo or pytest) on main at 1048962. 197 findings; 2 dropped because the check could not confirm them. No physics was judged. Results: `docs/analyses/2026-11-10-whole-project-audit/` (`INDEX.md`, `<area>.md`, `<area>-verified.md`, `prepared-fixes/`).
- Fixed (27 findings, 9 commits d6f5880 to c452531, each naming what it left): `docs/ai/02-system.yaml` loads again (BL-015 fixed: seven unquoted scalars quoted, words unchanged); stale paths and lines in the living documents; gate 11 of `gate.sh` prints its message when the wasm-bindgen pin is missing (INF-03); the duplicate `meval` dev-dependency; comments that described code that no longer exists; five pieces of code that cannot run or do nothing (`SerializationError::UnknownJointType`, two stale `#[allow(dead_code)]`, a dead local, a no-op `let _`); the actuator-sizing tutorial's menu names (LGC-15); `compute_trajectory` factors the constraint Jacobian once per sample (LGE-11, same numbers, time not measured); six literals in `gui/guidance.rs` name the file's constants (MGG-03).
- What a user can see of it: the tutorial's text and gate 11's message. `gate.sh` printed GATE PASS on the tree that holds all the fixes together; it was not run again for each commit.
- Listed for the owner, not fixed (168): `docs/analyses/2026-11-10-whole-project-audit.md`, most important first. 21 give a wrong result, lose work or crash, all in the linkage simulator (LGE-01, LGE-02, LGS-02, LGS-12 and LFA-05 first); 6 are physics-suspect and need the physics review before any change (LCS-02, LCS-06, LCS-14, LFA-16, LFA-17, LGE-03); then tests that can pass without asserting, the deploy that runs no gate before publishing (INF-05), dead or unfinished code, documents to rewrite, inconsistencies, speed, repetition and the baseline tools' output. None of the 168 is filed in backlog.yaml yet.
- Not touched: CLAUDE.md, AGENTS.md and the two campaign workflows (the owner's uncommitted edits) and the git-ignored files under `.claude/`. Nothing merged, nothing pushed.

## 2026-11-04 — Magcoupler corrections E21 and E24: the rating never raises an onset, and the cure margin reads the lower ring (BL-057, BL-072, branch magcoupling/bl057-bl072-e21-e24)
- Why: the owner approved the Addendum A-4 corrections on 2026-10-01 (docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md, section 5, option A on all 10 decisions); E21 and E24 had not landed. Plan: docs/superpowers/plans/2026-10-01-magcoupling-addendum-a4-corrections.md, Tasks 1 and 5, re-anchored to today's code.
- E21 (BL-057): `temperature::ring_demag` holds each ring's rating-calibration offset at `py_max(offset, 0)` on E20 and E21, before E20 picks the governing ring. Registry entry E21 (`Approval::AddendumA4`, decisions 1 and 3; 33 cells; 4 probes at full precision); `A4_REPORT`; `decision_list`. Rust-only `temperature.demag.inner_calibration_offset_C` and `outer_calibration_offset_C`, which the per-ring magnet limit and single-ring onset records now read (the BL-064 copy of the offset terms is gone). C50's and C101's help reworded in plain words (no "clamp" on screen). The default design (offset +9.92 C) and every grade or part with an offset of 0 or more are bit for bit on every result row; the moved designs are the seven grades (N52, N50, N50M, N38UH, N35EH, Recoma 26, Recoma 30), the N52 parts on both bases and the SuperMagnetMan arcs on the registry basis only.
- E22 and E23 are registered `Planned` (no code reads them) only because ids are index-based and E24 needs its index; they stay BL-058.
- E24 (BL-072): under E20 the cure margin C24 is MIN(inner ring's own single-ring onset, outer ring's own) − cure (`py_min`, the inner on a tie), gated on E20 and E24; registry entry E24 (decision 2; cells C24 and C25; probe Recoma 30 inside, N42H outside on `NONE.with(E20)`: 169.877 -> 90.830 C, plan decision A4-1). New Rust-only `temperature.demag.outer_onset_single_ring_C` beside BL-064's inner one (record in `records/demagnetization.rs`, path in the `demagnetization` scope chain: 194 paths; the results table explains 419 of 1,115); C24's record is `min(inner, outer) - cure`; C24's help reworded. The temperature guide's cure sentence now quotes the lower ring's onset (`guidance::cure_onset`), so its printed numbers still subtract. BL-072's red test (`e24_fails_a_cure_that_reaches_the_inner_ring_s_own_onset`: inner N42H, outer Recoma 30, manual dimensions, `passing_design()`, 2214 Hi-Temp) read +70.43 C before and reads -8.17 C, CHECK, after. No verdict moves at the default sliders; identical rings and every design where the inner ring has the lower onset are bit for bit. The demagnetization note has no cure-margin sentence, so it needed no change for E24.
- Tests moved by E21: BL-064's pins of N50M's single-ring onset (98.80 -> 91.27 C) and Recoma 30's (191.88 -> 191.43 C) in tests/grades.rs, tests/shaft_joint.rs and gui/guidance.rs; the "onset above the fitting need, onset less 10 C below it" case is now N42H at a knee fraction of 0.70 (onset 100.16 C). tests/explain.rs: the watch-out test renamed `the_demagnetization_watch_out_names_the_grades_rated_above_the_model_onset`; Recoma 26 on both rings now has the magnets governing (101.7 C).
- The demagnetization note: fifth sentence ("Onsets shift down to fit the magnet's rating, never up") and second watch-out reworded, two per-ring offset equations added, and it is back to `Review::Draft` (plan decision A4-2). Until a physics reviewer signs it off, `note_for` hides it and these tests are red: tests/explain.rs `release_notes_are_reviewed`, `every_note_names_its_committed_review_section`, `every_note_s_review_date_is_recorded_in_its_section`; with `--features app`, gui `every_start_here_step_opens_an_equation_with_its_reviewed_note`, `the_start_here_bar_names_every_step_it_shows`, `every_learn_link_resolves`, `a_check_s_why_link_opens_its_note_in_the_equation_panel`, `every_start_here_step_shows_with_glyphs_in_the_default_fonts`, `every_teaching_note_shows_with_glyphs_in_the_default_fonts` (all pass with the note marked reviewed).
- Physics review fold-in (2026-11-04, third commit): an independent physics review found no blocker. E21 names C25 too (34 cells: E21 turns the verdict at a knee fraction of 0.70 and for typed N42 coercivity on the user basis, beyond report 2.4's 95 configurations; C201 moves 1 ulp in one registry case); the hold is `temperature::e21_hold`, with a unit test that -0.0 is kept bit for bit; E24 propagates a NaN onset on either ring (`py_min` alone dropped the outer ring's), as does `guidance::cure_onset`, each tested per ring; a GUI test that the quoted onset less the cure is the margin where the outer ring is lower; E24's title in plain words; C101's help says a lower limit "can shrink" the swing. The demagnetization note: sentences 2 to 4 say "NdFeB or SmCo", watch_out[1] says the seven grades are rated hotter than the model's reference onset so their limits come from the model alone, and the note is signed off again (`Review::Reviewed`, 2026-11-04; record: the `### BL-057 E21 re-review (2026-11-04)` subsection of docs/analyses/2026-10-14-magcoupler-notes-review.md). Every note test and the release gate pass again; `gate.sh` GATE PASS on the committed tree.

## 2026-10-19 — Magcoupler guidance slice 3: the torque card, one step of each size control (BL-073, branch magcoupling/slice3-torque-card)
- Why: the owner went from 10 to 8 magnets, lowered the torque target and expected the coupling to shrink; it did not, and nothing on screen said why (slice 1's trigger; decision G-1 put a torque sensitivity card third). The card shows, for each of the four Size controls, what one fixed step each way does to the hot-low torque and what the step costs. At 8 magnets on the default ring the Outside diameter row reads -1 mm -> -0.006 N·m (-0.4 %) and +1 mm -> -0.005 N·m (-0.3 %), the Overall length row -1 mm -> -0.15 N·m (-9.1 %). Spec `docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md` (revision 5, committed dae024f after five review rounds); plan `docs/superpowers/plans/2026-10-20-magcoupler-torque-card.md` (T1-T10); research and the owner's decisions (gitignored) `.superpowers/sdd/2026-10-19-magcoupler-slice3/`.
- Owner decisions (2026-10-19, rounds 1-5): **1A** the step card, live while dragging; **2A** rows for the four Size controls only (Outside diameter, Overall length, Magnet face gap, Magnets per ring), hot-low only; **3A** fixed steps of ±1 mm, ±1 mm, ±0.1 mm and ±2 magnets; **4A** between the torque readouts and the drawing in place of the fixed label, collapsible and open by default, closed while the Equation panel is open or the window is narrow, the drawing keeping its 160 pt floor; **5A** with Auto on, the card describes the design shown: the held row reads "set by Auto" and one line says Auto sizes again after a change; **6A** the controls' fixed order, each row stating its step, no sorting and no ranking words; **7A** the N±2 lines under Magnets per ring move into the card, in hot-low (this reverses slice 1's spec section 3.3 for those lines), while the first line and the smallest-diameter line stay; **8A** each side's change in Outside diameter and Overall length, the dashboard checks whose level flips and the change of each failing check's own quantity; **9A** words on the failing side only, none of the card's numbers where the design shown does not fit or has no prediction, full validity for the size and gap rows and decision D-1 for the count, and words where the overlapping magnet length jumps; **10B** one decimal of percent and N·m to the fewest decimals that keep the true sign, on the card and on the readout's gap (a change to the shipped whole-percent format; it closes slice 1's S19, percent precision, deferred in 04-memory at 5a1126a and dropped from it in 47bdcd7, so no doc held it at 69fa505); **11A** read-only rows, labels through label_for; **12A** no new note: a Why link to the torque note and one caption line, BL-065 left open; **13A** live recompute inside the one ConsequenceCache, a browser drag frame measured before release, an optimization only if it misses 60 fps; **14A** the full test set; **15A** BL-073, risk physics, released only after the STOP and the owner's OK; **16A** a step that changes the calibration factor shows the true number with a short note; **17A, 17B** BL-074 and BL-075 filed with this slice.
- The night run (2026-10-19, `owner-decisions.md`): the owner approved the spec ("Approved. Run throughout the night to implement /ultracode as needed"), and on that approval the controller ruled, each on the STOP sheet for the owner to reverse: owner question O-1 answered "widen" (the Overall length row reads words whenever the overlapping magnet length changes by anything other than the step, so a short cup cavity's +1 mm side reads words where 9A as written showed -6.2 %); O-2 answered "quote none" (no Temperature limits quantity on the card, its flips still shown, T-13); spec review round 3: the design shown decides first by the readouts' own rule, with one line in place of the rows where the magnets do not fit, the model has no prediction, or a pull-out or hot-low torque is not above zero (`ShownNoHotLow`), the card's open state keyed on the screen's size (`CardView`), the layout test's windows, and the shaft boss plateau recorded as risk R-13; round 4: the card's header row wraps, so its Why link stays inside a narrow centre region, and a count side whose calibration switch takes its torque to or below zero shows its number with the calibration note. The plan review's rulings are in the plan's head.
- Number format (T1): `gui/format.rs` gains `signed_fixed`, `signed_change`, `change_decimals` and `CHANGE_EXTRA_DECIMALS` (a change to the fewest decimals, from its unit's own up to two more, that keep its true sign and a digit that is not zero; past that cap its own decimals with its sign), `TORQUE_DECIMALS` (a torque change's own two decimals, one constant for the readout's gap and the card), and `fewest_decimals_apart` with `MOST_DECIMALS` moved there from `gui/guidance.rs`; `signed_percent` reads one decimal, an exact zero of either sign "+0.0 %", a value that rounds to zero keeping its sign; `torque_gap_text` follows the rule (at the example design "short 0.21 N·m (-8.6 %)", was "(-9 %)").
- Shared pieces (T2): `guidance::check_terms` (`CheckTerm`, `Quoted`; the production half of the fix tests' terms, which now call it), `SPACE_CLAIM_AXIS_SIZES` (the Problem's axis table and the quoted axes read one table), `size_controls::slider_minimum` (the Outside diameter slider's minimum, one expression for the controls and the card) and `size_controls::SliderMapping` (each Size slider's own mapping of a value to a design, which the controls apply on an edit and the card on each step), `CARD_RANKING_WORDS` and its whole-word matcher in `engine/explain/plain.rs`, and the card's shared test designs in `gui/test_support.rs`.
- The card (T3, T4): `gui/torque_card.rs`, pure: `card(inputs, results, minimum_diameter_mm, clamps_paused) -> TorqueCard` (`ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow` or `Rows`), each Size control stepped through its slider's own mapping and tested in the order of the spec's section 3.3, with `STEP_SLACK_MM` the one slack of the range test and the jump rule; `side_words` and `count_step` moved there from the neighbour lines (moved, not copied); flips over the counted guides and the failing checks' quantities through `check_terms`; every text a named constant set by the G-3 review; a row's two N·m at one precision (T-14); `card_designs()`, the one table of the spec's section 2 designs the card's tests loop over, T-14's one-flip design and R-13's boss plateau among them; one rule for the range Auto searches, `gui::sizing::sizing_range`, which the panel and that table read.
- Wiring and layout (T5, T6): `Consequences.card` in place of the neighbour lines, the cache's one key in three parts (the design shown, the Minimum OD, the shaft-clamp pause); `RING_SIZE_STAYS` reworded, `AUTO_RESIZES_RING` and `AUTO_RESETS_LENGTH` removed, `magnet_count_line` per the spec's section 4.3; the card drawn in the Size view in place of the label above the drawing (`torque_card_ui`, a `CollapsingState` whose wrapped header row holds the heading, which toggles it, and the Why link), its default decided only when the `CardView` changes and the user's own click winning for the session; `geometry_view::drawing_floor` and `list_row`; `test_support::drawing_rect`.
- Reviews and smoke (T7, T8): every card string set by the G-3 physics review (`docs/analyses/2026-10-14-magcoupler-help-review.md`, `## Slice 3 torque card`); gui-smoke gains the card step (the texts quoted whole, held to the code by `app.rs`'s smoke test) and a browser drag-frame timing on the gap slider that keeps only the frames the app draws (`FRAME_TIMER`), which gates `passed` only with `frame_check`.
- Docs and backlog (T9): the README (the status, the module list, the Size-tab lines and the card, the Why links, the test rows), docs/ai 02-05; BL-073 (`in_progress` until the owner's OK), BL-074 (the Overall length slider's first move from a blank override sets both rings to one length, with the shaft boss plateau of risk R-13 in its evidence), BL-075 (the BL-064 fold-in's review notes with substance) and one line in BL-065's notes (the card's percents are a consumer of its fix). Not covered by the card: the diameter row's boss plateau (R-13) and the reverse pair's trimmed inner ring (BL-074), both recorded; the card's percents read the model's end factor (BL-065).
- T10 release checks (2026-10-20, re-run 2026-10-21 after the fold-in): the whole-branch review passed after its fix wave of 2 items (re-reviewed) (record: .superpowers/sdd/2026-10-19-magcoupler-slice3/final-review.md); `gate.sh` GATE PASS with gate 12 running (the oracle venv reference/magcoupling-py/.venv); the web build carries the card (its heading is in magcoupling-web_bg.wasm, main's Auto ring-size line is not); the Size tab's two caches with the card take 3.302 ms per changed design at the example design, native release (record: .superpowers/sdd/2026-10-19-magcoupler-slice3/timing-T10.txt); a gui-smoke run passed on the local server with the card step and the browser frame check (Chrome 155: 182 drag frames, median 1.6 ms, 95th percentile 8.9 ms, and 10.3 ms at the ring design; record: .superpowers/sdd/2026-10-19-magcoupler-slice3/smoke-local.json). The owner reviewed the diff and gave the OK, with the deploy authorization, on 2026-10-20, choosing the picks FO-1 to FO-4 and the fold-in of every leftover first; merged to main and pushed 2026-10-21; the deploy run and the live smoke are checked in the session that pushed. The review leftovers fold in after the owner's answer at the STOP (3A), one commit per item: WP-A gives the Size tab's cap rule one source, `format::decimals_with_a_digit` (read by `change_decimals` and the card's `shows`; no text on screen changes), and pins the readout's gap past the cap, at a hot-low that is not finite, its word against its percent's sign, and at the spec's designs. WP-B applies the owner's word picks and the G-3 review's two should-fix texts on screen, each for its G-3 re-review: a side off its slider reads "outside the slider's range" (FO-1); the jump words name the step's amount (SF-1); the Outside diameter row reads the cup's jump words (`CUP_JUMP`) where the shaft boss sets the Outside diameter, so R-13's boss plateau is covered now (FO-2); a design shown with fewer magnets than the slider takes draws one line (`SHOWN_COUNT_BELOW_SLIDER`), checked before the magnets' fit (FO-3); the hot-low one-line form says "of zero or less" (SF-2). WP-C applies the owner's number pick on screen, for its G-3 re-review: a number whose fixed form would need more than six whole digits, or an N·m that reads zero at the cap beside a percent that reads a digit, reads four significant digits, with its sign, as the readouts write values, on the card and in the readout's gap, and a shortfall's size too, without a sign (FO-4: at a design file's calibration coefficient of 1e-9 or 1e300 or its Required of 1e-300, "+1.690e-11 N·m (+1.9 %)" and "margin +2.29 N·m (+2.285e302 %)" where it drew "+0.00" and 303 digits). The G-3 re-review of these picks (round 1) asked for two fixes, applied in one commit: the jump words name the step's amount and direction, since at a cup cavity of 10.7 mm the +1 mm side moves the overlap by the step's size the other way (MF-1); and a torque card row whose one N·m reads four significant digits beside a percent that reads a digit draws its other N·m so too, so the two keep their true ratio ("-5.033e-5" beside "+4.599e-5", not "-0.0001", at manual blocks of 0.2 T on the sliders; T-14, MF-2). The re-review's round 2 passed, with nothing for the owner (record: `docs/analyses/2026-10-14-magcoupler-help-review.md`, `## Slice 3 torque card`). WP-D tidies the panel's card code with no text on screen changed: the Size tab's two cache asks, the lines under the controls and the card, go through one method (`size_tab_consequences`) that hands the cache the design's own results, computed once per design (`ResultsCache`: none on an idle frame, where each ask ran its own `compute_all`); rule 3 and the card body's cap read one `room_over_the_floor`; rule 3 is read last, so its pre-measure runs only where rules 1 and 2 open the card, and a new test pins each rule at its boundary (a region exactly 400 points wide, or exactly as tall as the open card, the spacing and the floor, opens it); the card body's scroll id derives from `TORQUE_CARD_ID`. WP-E tidies the card's, the Size controls' and the guides' code with no text on screen changed: the Magnets per ring slider's mapping (`SliderMapping`) refuses a value a cast to the count would change (not finite, not whole or past what an i64 holds) where it truncated or saturated it, the package's one behaviour change, which no slider or card step reaches; an input's slider range has one shared spelling, `inputs::slider_range` over the catalogue built once, read by the card's gap and count rows (its last production `expect` gone, and its count range no longer rebuilt from every input row), the controls, the guides and the tests; the card's slider names its own control, so a side is never stepped as another control; the Size tab's gap and count controls and the panel's tests name the gap by guidance's one `FACE_GAP`; the test designs share one parts edit (`set_parts`), the check terms one constructor (`Quoted::new`) and the plain-words guard one whole-word rule (`says`); `side_words` and `count_step` are private to the card. The card's `compute_all` count per changed design is unchanged. WP-F strengthens and tidies the card's, the shared designs', the guides' and the Size controls' tests with no text on screen changed: the shared designs' test pins the temperature edge from both sides (a caution at 82.10 °C), the end effect out of range at 4.0 mm and the reverse pair's inner ring trimmed by more than one length step; `check_terms`' test holds the margins to the results at every guide test design, with the not-judged inner hub's four terms passing and each shaft-joint cause in its own unit; the Outside diameter mapping's test reaches a diameter below the default floor; a design shown past each slider's end (a 6.0 mm gap, a 31 mm ring, a 0.5 mm override with no end effect, 44 narrow magnets) reads the slider's range on both sides; each quoted margin change is held to `compute_all`; the card's tests pin the spec's "-0.85 N·m (-100.0 %)", the held row's numbers with Auto off and the robustness test's 416 rows, and the caption's pin bans the claim that everything else is held in any words; the card table's Auto designs come from the panel's own `SizingRunner::shown`; the tests share `with_cup_depth`, `with_magnets`, `row_line`, `auto_states`, `glyph_context`, `QUOTED_UNITS` and one number-side loop, and the count-row test moved from the guides is now part of the side-reads-a-number test modelled on it. WP-G strengthens and tidies the panel's card tests and the shared text probes with no text on screen changed: the card is checked as decided on every frame of a drag; with no user choice Reset all and a load keep the default decided for the view, and a change of the window's height alone decides it again; inside an `egui::Window` the card opened by the user paints and takes no click in the Equation panel; at a design file's calibration coefficient of 1e300 (card lines of 887 characters) rule 3 starts the card closed, its body scrolls above the drawing's floor and its measured height is its drawn height; test 25 names its capped cases, the card tests share one floor check (`keeps_its_floor`), and the floor test pins its figures by hand (212.125 points in egui's default style); the text probes find a drawn text through one match, `test_support::text_shapes`, the panel tests' `clipped_text` gave way to `text_rect` and `text_clip_rect`, and the panel tests read the card's lines through one `card_lines`. WP-H tidies the gui-smoke workflow, its docs and its tests with no text on screen changed: the README's drag sentence moves the pointer 60 times, alternately 3 px right and left of the press, and names the moved gap the release run also needs (the T8 text re-review's SF-3), and the script's `frame_check` comment names the drag's three conditions (SF-4); a `frame_check` that is not `true` or `false` stops the run before its first step, and the calculator's result records `frame_check`; the script checks the calculator's `passed` again from its agent's fields (every check its schema requires, no console error), and an agent that returns nothing leaves its step every required field, false or empty, with no drag numbers; the drag step types the gap's shown value once (`SMOKE_GAP_SHOWN`), and test 26 holds it and the drag's sizing-line prefix to the app; T8's stub is now a tracked node test (`linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`, 71 checks) that runs as gate 13 of `gate.sh` (a SKIP line without node; the Python oracle keeps gate 12, and the `--full` release builds are gate 14). WP-I corrects the docs and comments the whole-branch review's S-1 lists and adds its O-1 pin, with no text on screen changed: at the keyway through the hub only the inner hub's joint goes unjudged; the README lists the robustness test's values as the test holds them; the card's texts read as set by the slice's G-3 review, not drafts; the user's `card_choice` decides before `card_default`; the drawing's floor is the room the card leaves below it; `weak_line` names the card's lines; the README's T6 clause names the one Equation-panel check that runs unpaused; the gap's decimals stop at two more; 04-memory's teaching NEXT says slice 3 built the card as 2A and 6A narrowed it; 6A, not 14A, sets the rows' order; slice 3's public docs name private items in plain backticks (no dead links); test 4 pins the hot-low half of the readouts' rule at a design file's variation of -1e308; the card has a row only where the Size tab draws its slider, one line where nothing compares, and a side at 40 magnets only where 40 blocks fit; a dated note sits under the G-3 record's traps bullet; 02-system's known limitations record the reverse pair's trimmed inner ring (BL-074) and the owner's FO-3 and FO-4 bound on a design file's degenerate values, FO-4 reaching slider values too (the fold-in G-3 re-review's SF-1); the gitignored `check_shipped.py` reads a reworded partial quote as a wording change.

## 2026-10-17 — Magcoupler shaft joints: the keyed shrink fits that hold the rotors on their shafts (BL-064, branch magcoupling/bl064-shaft-joint)
- Why: nothing checked how the two rotors are held on their shafts once the shaft-clamp section was paused (2026-10-15). The owner's ruling of 2026-10-17: "Hub is shrink fit with key only" (no clamp screws, no adapter joint). The drive line is motor -> GAM 5:1 gearbox -> coupling -> wheel, and both rotor-to-shaft joints carry the coupling's torque: the inner hub (its magnets bonded on its ten flats) on the wheel shaft, and the cup (its integral boss) on the gearbox output shaft. Cold-high (`metal.torque_cold_high_Nm`, 3.823 N·m at the defaults) can reach each joint before the coupling slips; it is not a bound on them (backlash, resonance crossings and re-lock ring-down can exceed it on the gearbox side, and with the operating temperature below the minimum the operating pull-out is above it, BL-067). Spec `docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md` (approved by the owner 2026-10-18 at 0687034, after six review rounds); plan `docs/superpowers/plans/2026-10-18-bl064-shaft-joint.md` (T1-T10); research (gitignored) `.superpowers/sdd/2026-10-17-bl064-shaft-joint/`.
- Owner decisions (2026-10-17): **1A** both joints in one counted dashboard row, one pure joint function evaluated twice, the verdict naming the failing joint; **2A** the fit alone must hold the safety factor times cold-high at its loosest and the part must stay within yield ÷ 1.2 (`YIELD_FACTOR`) at its tightest; the key alone is a backup checked at the same torque; fit and key capacities are never added; **3A** minimum and maximum measured diametral interference as inputs, one band per joint (placeholders until the real fits are set: inner hub 13-18 µm, cup 15-22 µm); **4A** DIN 6885-1 3 × 3 key inputs, width 3.0 mm and contact height 1.2 mm, shared by both joints (the paused clamp's key inputs stay with the paused section); **5A** a safety factor input, 2.0, on slip and on the key backup; **6A** the shaft-clamp section stays paused and unchanged; **7A** "cannot be fitted" is a counted cause per joint: the part's needed temperature, 20 °C + (maximum interference + joining clearance − the shaft's shrink at its fitting temperature) ÷ (α_part · d), at or below its allowed temperature. Process: the bare cup is shrunk onto the gearbox shaft first and its magnets bonded after; the inner hub is bonded first and shrunk onto the cooled wheel shaft (wheel shaft −196 °C, liquid nitrogen; gearbox shaft 20 °C; the shrink below 265 K from NIST's Fe-2.25Ni fit, not a constant coefficient); allowed fitting temperatures 100 °C for the bonded hub, never above the adhesive's design limit, and 300 °C for the bare cup. **8A** a new reviewed note `shaft_joint`, the note cap 20 -> 21. Material 4140 with UltraOx, the bore masked (the library's 4140 yield, 415 MPa, conservative while the condition is open); both shafts Ø10 (`coupling.bore_mm`). **R-15** (after spec review 3): the bonded hub's allowed fitting temperature is also capped at the magnets' single-ring onset less `CURE_MARGIN_MIN_C` (10 °C), the temperature verdict's cure margin, so lower grades read "cannot be fitted" and the row is a stored-3D row; its **ring ruling** (after spec review 5): the onset is the inner ring's own, a new Rust-only result `temperature.demag.inner_onset_single_ring_C`, never the governing ring's. Release: the owner's review of the diff before the merge (risk: physics).
- Engine (T1, T2): `temperature.demag.inner_onset_single_ring_C` (the inner ring's own single-ring onset beside `inner_magnet_limit_C`, equal to the governing onset wherever the inner ring governs) with its record in `records/demagnetization.rs` and its path in the `demagnetization` scope chain; `PartProperties::hub()` in `material_library.rs` (the part material in effect and its yield; the warnings' hub expansion in `api.rs` reads it too; `temperature.rs` keeps its own E18 choice of the hub's expansion for the bond-mismatch screen, `temperature.rs:1075-1079`, toggled with the correction: the same rule with every correction on, so it is not folded into the helper, an observation with no change). `engine::shaft_joint` (Rust-only, no cell, like `gearbox`): `ShaftJointInputs` (15, the first top-level Rust-only input group) and `ShaftJointResults` (24), computed in `api.rs` after `gearbox::compute`: plane-stress Lamé fits, the holding with the key's arc removed (floored at zero), the bore's von Mises stress, the key flank with the contact height capped at the keyway depth over the bearing length L − b (+∞ and a fail without bearing area; no key term without a keyway); the loosest and tightest ends (the minimum magnet temperature, and the adhesive's design limit with the rotor over its shaft); `hub_fit_allowed` (NaN when any of its three values is NaN; ties name the earlier limit); the cause order (not a number; a safety factor not above zero, plan deviation PD-2; an input typed below what it can physically be, ruling C-4 extended by the whole-branch review: a friction, smoothing or joining clearance below zero, a key width at or below zero with a keyway or a shaft below absolute zero, `impossible_values`; then a gearbox shaft below its slider's bottom, `OUTPUT_SHAFT_COLDEST_C`, the coldest the check takes it since its rated range is not checked, build ruling B-5; with `cannot_be_made` holding these four for `verdict` and the guide, whose Problem then gives one sentence for each such input, or for the gearbox shaft, and no fix; the fit slips; the bore stress; cannot be fitted; the key; then no wall around the bore (a part diameter typed at or below zero included, plan deviation PD-1), then a reversed band), only a determinate joint judged; badge numbers never equal; the verdict constants `SHAFT_JOINTS_PASS` ("Both shaft joints hold and can be fitted"), `SHAFT_JOINT_UNKNOWN`, `FIT_SLIPS`, `BORE_OVER_ITS_LIMIT`, `CANNOT_BE_FITTED`, `KEY_OVER_ITS_LIMIT` and `KEY_HAS_NO_BEARING_AREA`. `tests/shaft_joint.rs` end to end; `input_schema.json` re-blessed; the default share link grows from 2,462 to 2,643 characters (guard below 2,700). No Python-ported input, result, label, help, default, range or cell changed, so gate 12's parity is unaffected. 187 inputs; 1,112 results.
- Explain layer (T3): `records/shaft_joint.rs` (23 records; the drift guard holds each to the engine at every probe design and at a design with no wall around the bore); the `shaft_joint` scope chain after `geometry`: the scope holds 193 paths and the results table 416 explained results of 1,112; `CHAIN_LABELS` 9 (the shaft joints between Geometry and Mass); four `ASSUMPTIONS` rows (`fit_friction`, `fit_smoothing`, `rotor_over_shaft`, `joining_clearance`; 18 rows), whose Rust-only sources the widened rule of `tests/assumptions.rs` accepts ("BL-064 shaft joint spec").
- Panel, drawing and exports (T2, T4, T5): the Shaft and hub joint group gains the two key rows and the sections "Shrink fit" and "Fitting temperatures", before the paused clamp's (`SECTION_LABELS` 19; T2); the key is drawn in the bore in both states from `shaft_joint.key_width_mm`, and none at a keyway depth or key width of 0 or less (T4; plan deviation PD-3); CSV, JSON and the XLSX Inputs and Results sheets carry every new input and result (T4), and the XLSX Summary lists 19 dashboard lines (T5).
- Dashboard, guide and fixes (T5, T6): `DASHBOARD` 19 (the row at index 15, after the hidden clamp row and before the gearbox row; Magnets fit is still drawn last), `CHECKS` 32, `END_EFFECT_ROWS` 9 (it reads cold-high, so it greys, uncounted, while the end-effect model is out of range), `STORED_3D_ROWS` 4 (R-15: its hover carries the "Fixed 3D fields from the example design" label, the drawn label now follows it, and `STORED_3D_NOTE` names the bonded hub's allowed fitting temperature). `SHAFT_JOINT_GUIDE` (label "Shaft joints") joins `CHECK_GUIDES` (10) between the clamp's and the gearbox's: its Problem names the joint and the cause with live numbers, the window of measured interference that would work, quoted inward (or says in numbers that there is none), and the source of the hub's allowed temperature (your limit, the adhesive's design limit, or the magnets' single-ring onset less 10 °C); `Fixes::PerTerm` per cause and joint, with three fixes held back by rule (a band step that would reverse the band, either way; the key's contact height where the key has no bearing length). The example design reads "Example design: fails 4 of 9 checks" while the shaft-clamp section is paused ("Example design: fails 4 of 10 checks" with it shown); `passing_design()` passes all 9; `only_the_joint_fails()` (the hub minimum at 8 µm) fails 1 of 9. At the defaults the inner hub holds 9.484 N·m against the 7.647 N·m it must hold and needs 93.27 °C against 100.0 °C allowed; the cup holds 16.90 N·m and needs 280.2 °C against 300.0 °C. Auto unchanged.
- Note and reviews (T7, T8): the new note `shaft_joint` ("Shaft joints: a keyed shrink fit"; its Start here step just before `clamp_preload`: 15 steps, 14 while paused; 21 notes, 20 shown), physics-reviewed (`docs/analyses/2026-10-14-magcoupler-notes-review.md#shaft_joint`); every new on-screen text G-3 reviewed (`docs/analyses/2026-10-14-magcoupler-help-review.md`, `## BL-064 shaft joints`). `CLAMPS_PAUSED_LINE` no longer says the clamp returns with BL-064; no new sign-off sheet (the owner reviews the diff).
- Docs and backlog (T9): the README, docs/ai 02-05; BL-064 (the acceptance amended to the owner's ruling; the owner decisions, the open facts and the default verdict in its notes), BL-060 (paused with the clamp section, which BL-064 keeps paused), BL-066 (its filter tests now move from four gearbox inputs to five), BL-067 (the shrink-fit check reads cold-high too; one fix serves both checks); filed BL-069 (the UltraOx finish against the nickel plating model), BL-070 (fretting and fatigue of the joints under the reversing slip swing), BL-071 (separate shaft diameters) and BL-072 (the temperature verdict's cure margin reads the governing ring's onset: the owner-approved E24 of 2026-10-01, to land with E21, BL-057). Not covered by the check (spec section 10): fretting and fatigue (BL-070), the keyway corner's stress concentration, the gearbox's own shaft, bearings, seal and heat soak, the bond strain from fitting a bonded hub, the rotor chill from a cryogenic shaft, frost on a cold shaft, and the key's shear.
- T10 release checks (2026-10-19): the whole-branch review passed after its fix wave of 1 item, re-reviewed (MF-1: a shaft-joint value typed below what it can physically be reads unknown; record: .superpowers/sdd/2026-10-18-bl064-shaft-joint/final-review.md); `gate.sh` GATE PASS with gate 12 running (the oracle venv reference/magcoupling-py/.venv); the web build carries the check (its passing verdict text is in magcoupling-web_bg.wasm, main's stored-3D hover text is not); a gui-smoke run passed (hub, linkage; magcoupler: first view, the Show spreadsheet details toggle, share link, Auto solve, design file, a Why link opening a note, an input tooltip without spreadsheet details, the dashboard scrolling to Magnets fit at 1280 x 800, a clean console). The owner reviewed the diff and gave the OK on 2026-10-19; merged to main and pushed; the deploy run and the live smoke are checked in the session that pushed.
- Review leftovers (2026-10-19, owner decision 2A): the reviews' open leftovers folded in before the merge, one commit per item, LO-1 to LO-76 (LO-66 extends the gitignored release check `check_shipped.py` and has none; the fold-in plan and the controller's ruling B-5 on the items that offered options are in `.superpowers/sdd/2026-10-18-bl064-shaft-joint/`, gitignored). Behaviour: a key width is floored only where there is a keyway (B-5); a gearbox shaft typed below its slider's bottom, -20 °C (`OUTPUT_SHAFT_COLDEST_C`), reads its own unknown, `UNKNOWN_GEARBOX_SHAFT_TOO_COLD`, with its own Problem sentence and no fix, since the gearbox's rated range is not checked (B-5), while the wheel shaft is still judged down to absolute zero; the drawing draws no key at a key width of 0 or less, as at a keyway depth of 0 or less (PD-3). On-screen texts, G-3 re-reviewed together under the help review's `## BL-064 shaft joints` (`### Re-review of the review leftovers (2026-10-19)`: round 1's two must-fix items applied, the gearbox shaft's typed value quoted so it reads below its slider's bottom, `input_text_below`, and the single-ring reverse field's help naming the allowed fitting temperature; round 2 passed): the gearbox shaft's new badge reason and its Problem sentence; the engine helps of the safety factor, the friction, the rotor-over-shaft allowance, the bonded hub's limit (input and result), the wheel shaft, the stress limit and the inner ring's single-ring onset (`input_schema.json` re-blessed for the input helps); the guide's empty window (its edges and its clearance clause) and its Why; `STORED_3D_NOTE`; the overlay helps of the bore, the keyway depth, the boss length and the single-ring reverse field. The note `shaft_joint`: sentence 1 says what each joint must hold, the first watch-out names the heat cure and the smoothing source names the surfaces the 5 µm rests on, physics re-reviewed under the notes review's `## shaft_joint` (passed in round 1; the note's review date stays 2026-10-19, the re-review's date). Refactors, one rule in one place: `api::parts_in_effect` and one `joint_links` behind `shaft_joint_links`; `has_bearing_area`; `PartMaterial::is_non_magnetic` and `Material::yield_or_nan`; the guide's one rounding rule, one decimals search and `result_number`, and its fitting source reading the engine's links; `plain::TORQUE_ORDER_WORDS`; shared test helpers in `gui::test_support` and `tests/common`. Tests: stronger pins and the edges the reviews named (no wall named before a reversed band; all six impossible-value sentences, and two at once; a key contact typed at -0.5 mm; the note's sentence 4 with mixed rings). Records and docs corrected: 02-system cites BL-067 for the operating pull-out; the owner's open facts are listed once, in 04-memory, which BL-064's, BL-069's and BL-071's notes and 02-system point to; this entry's NIST clause and cause order; BL-067's product as computed; 03-structure lists the shaft-joint note tests apart from the note release gates; the notes review's Tests bullet and its per-µm bore stress range; the help review's map of Task 2's SF-3. Kept, by B-5, and recorded in the help review (`### The review leftovers' rulings (2026-10-19)`): a friction typed at exactly 0 is judged; `NO_FIX` in a no-wall design; "outside" in `UNKNOWN_IMPOSSIBLE_VALUE`; the guide's four-digit readout of the slider bottoms.

## 2026-10-16 — Magcoupler gearbox check: cold-high at the gearbox input against its rating (BL-059, branch magcoupling/bl059-gearbox-check)
- Why: nothing compared the coupling's strongest expected torque with the gearbox it sits on. The drive line is motor -> GAM 5:1 gearbox -> coupling (on the gearbox output) -> wheel (owner, 2026-10-14): when the wheel is held the coupling passes up to its pull-out before it slips, and while it slips the gearbox carries a fully reversing swing of the same amplitude; with the minimum magnet temperature at or below the operating one, the strongest expected pull-out is cold-high (`metal.torque_cold_high_Nm`: the minimum magnet temperature, with the production allowance added; an operating temperature below the minimum is BL-067). At the defaults cold-high at the gearbox input is 0.8049 N·m against the 0.6 N·m input rating. Spec `docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md`; plan `docs/superpowers/plans/2026-10-16-bl059-gearbox-check.md` (T1-T10); research (gitignored) `.superpowers/sdd/2026-10-16-bl059-gearbox-check/`.
- Owner decisions (2026-10-16, all the recommended options): **1A** the basis is the gearbox input: `metal.cold_high_input_Nm` (cold-high divided by the ratio and the efficiency) against `coupling.gearbox_input_rating_Nm`, the same test as cold-high <= rating x ratio x efficiency = 2.85 N·m at the output (the workbook's `model.gearbox_reference_Nm`); it is never unconservative, since the forward-drive half of each slip cycle puts T/(i·η) on the input. BL-059's note of 2026-10-14 that called this the same as 3 N·m at the output is corrected. **2A** built with 0.6 now; which datasheet line it is stays open (BL-059 notes, 04-memory). **3A** counted and red. **4A** the conflict sentence in the Problem. **5A** Auto unchanged: it still sizes for the hot minimum check only. **6A** fixes Lower Axial magnet length and Raise Magnet face gap. **7A** only the texts that become false or misleading change. **8A** the repeated slip swing as a fatigue load is BL-066. Release: the owner's review of the diff before the merge (risk: physics).
- Engine (T1): `engine::gearbox`, Rust-only like `housing`: `GearboxResults { cold_high_check }` declared with `out_rust_only` (no cell) and `compute(cold_high_input_Nm, rating_Nm)`: at or under the rating `WITHIN_THE_GEARBOX_RATING`; strictly above, +∞ included, `ABOVE_THE_GEARBOX_RATING` followed by both torques to 0.01 N·m, the cold-high at least 0.01 N·m above the rating as quoted (the space claim's `quoted_overshoot` rule), so a red verdict never shows two equal numbers; a NaN in either `GEARBOX_CHECK_UNKNOWN`. `DesignResults.gearbox` is computed in `api.rs` beside `housing::compute`; `PACKAGE_LABELS` has 13 entries ("Gearbox"). No Python-ported input, default, range, label, help or result changed, so gate 12's parity is unaffected; the results table has 1087 results (392 explained).
- Dashboard and guide (T2-T4): `DASHBOARD` 18 rows, the check at index 15 after the 15 headline rows (Magnets fit is still drawn last); `CHECKS` 31; `verdict_level` Good, Bad (by the prefix) and Caution (unknown); `END_EFFECT_ROWS` 8: the check greys, uncounted, while the end-effect model is out of range (at c_end 1.5 the negative cold-high would otherwise read as a pass). `GEARBOX_GUIDE` joins `CHECK_GUIDES` (9, in the dashboard's order) and `ROW_LEARN`, with its Why link to the torque-band note (`hot_low_cold_high`). Its Problem quotes cold-high at the gearbox input and the rating; where meeting the requirement would put more than the rating on the gearbox input at the design's band (cold-high over hot-low, which no magnet size, gap or count changes), it adds the conflict sentence (`gearbox_conflict`: the band, what even just meeting the requirement at the gearbox output puts on the gearbox input, and the targets that would settle it; 1.673, 0.8806, 1.703 and 1.140 at the defaults); where the requirement alone is above the rating at the gearbox input (the ratio's or efficiency's slider bottom) it offers no band, and with the operating temperature below the minimum it is left out (plan deviation PD-1, BL-067); a NaN reads `GEARBOX_UNKNOWN` (Caution). Fixes: Lower Axial magnet length and Raise Magnet face gap; both also lower hot-low, so at the defaults the length pulls two ways (the hot-low and gearbox blocks both say so) and the face-gap side-effect line prints in the gearbox block. The example design reads "Example design: fails 4 of 8 checks" while the shaft-clamp section is paused (4 of 9 with it shown). `passing_design()` takes a gearbox input rating of 1.0 N·m (cold-high at the input 0.7070, 29 % under it); with 0.6 back it fails the gearbox check alone. The tripwire `nothing_checks_a_gearbox_torque_against_its_rating_yet` became `the_gearbox_check_is_a_dashboard_row_with_its_guide`; a dedicated test covers a design just under the gearbox limit, where the hot minimum's length fix is held back and its block shows `NO_FIX`; Auto on length at the defaults (13.770 mm) leaves the check red (5A).
- On-screen texts (T5, T6, T8): the gearbox rating's on-screen help names the check (the engine help stays the workbook's); the ripple row gets an on-screen name, "Nominal slip swing at the gearbox input", and a help that says what it is (the pull-out at the gearbox output at operating temperature, without the production allowance, divided by ratio and efficiency: 0.5660 N·m at the defaults, beside cold-high at the input, 0.8049) and that the gearbox check compares cold-high at the gearbox input instead; it claims no inequality, since with the operating temperature below the minimum the swing can exceed cold-high (BL-067); the ban on saying the coupling protects the gearbox now covers the guide and overlay texts as well as the notes. The two watch-outs that said nothing checks the gearbox (`hot_low_cold_high` watch_out[1], `when_slipping` watch_out[0]) now say what the check compares: the torque-band one adds that repeated slips also load the gearbox in fatigue, which nothing checks; the slip one says that cold the swings can reach cold-high and that the check compares only that torque with the input rating. Their sources name the check and the open datasheet question (`notes::GEARBOX_CHECK_SOURCE`). Both notes were physics re-reviewed (a `### BL-059 re-review` subsection under each note's section of `docs/analyses/2026-10-14-magcoupler-notes-review.md`; PASS at round 2, after round 1's must-fix took "peak" out of the slip watch-out, since with the operating temperature below the minimum cold-high is not the swings' peak), and every new on-screen text (the guide's label, Problem and Why, the three verdict texts, the engine label and help, the two overlay rows) passed one G-3 review (`docs/analyses/2026-10-14-magcoupler-help-review.md`, `## BL-059 gearbox check`; PASS at round 2, after round 1's three must-fix items made the conflict sentence, the ripple help and the engine help name each torque's side of the gearbox, and replaced the engine help's peak-check sentence, which could read as a peak rating, with "Fatigue from repeated slips is not checked."). No new note sign-off sheet (decision 14A: the owner reviews the diff).
- Panel and exports (T7): the gearbox row's own Why link opens the torque-band note and the row opens in Explain; CSV and JSON carry `gearbox.cold_high_check` with an empty cell column; the XLSX Summary lists 18 dashboard lines; the results table's failing filter lists 5 red checks at the defaults (4 before).
- Docs and backlog (T9): the README, docs/ai 02-05 and BL-059 describe the check (BL-059's basis note corrected to 1A; the datasheet question recorded in BL-059 and 04-memory); BL-066 filed (risk: physics): a repeated-slip check of the slip swing against the gearbox's nominal (continuous) rating, which needs the owner's datasheet value as a new input; BL-067 filed (risk: physics, priority 3): an operating temperature below the minimum, where cold-high is not the strongest pull-out (it also records that the torque-band note's sentence 2, which predates BL-059, is false there). Not covered by the check (spec section 3): the slip swing as a fatigue load (BL-066), backlash impacts at each reversal, the ring-down at re-lock, torsional resonance crossings as the slip frequency sweeps 0-167 Hz, the drive line's stored energy at a jam, and the gearbox's own temperature range (the old GAM catalog lists -10 or -20 °C; the coupling's minimum is -40 °C).
- T10 release checks (2026-10-17): the whole-branch review passed after its fix wave of 3 items (re-reviewed), and a second wave of 3 minors the re-review raised (re-reviewed) (record: .superpowers/sdd/2026-10-16-bl059-gearbox-check/final-review.md); `gate.sh` GATE PASS with gate 12 running (the oracle venv reference/magcoupling-py/.venv); the web build carries the check (its verdict text is in magcoupling-web_bg.wasm, the old torque-band watch-out is not); a gui-smoke run passed (hub, linkage; magcoupler: first view, the Show spreadsheet details toggle, share link, Auto solve, design file, a Why link opening a note, an input tooltip without spreadsheet details, the dashboard scrolling to Magnets fit at 1280 x 800, a clean console). The owner reviewed the diff and gave the OK on 2026-10-17; merged to main and pushed; the deploy run and the live smoke are checked in the session that pushed.
- Review leftovers (2026-10-17, owner decision 2A): the whole-branch review's minor items folded in before the merge. Docs and test comments: 03-structure's guide count reads 8 while the clamp section is paused, and `the_docs_count_the_dashboard_rows_and_the_checks` now holds it; "a peak check only" became "fatigue from repeated slips is not checked" (README, 03-structure, the `engine::gearbox` module doc); 02-system's gearbox limitation adds the BL-067 corner and the gearbox's own temperature range, and its rating line names the Python-locked "GAM 5:1 peak input rating." note as the one exception; the slip watch-out's "compares only that torque" is pinned; three test comments say the swing "can be" above cold-high in the BL-067 corner; dated corrections close the help review's and the notes review's BL-059 records. Two on-screen words (G-3 round 1 S-3 and S-2): the gear ratio's help says "such as the slip swing", and the gearbox Problem's band clause names "a smaller variation allowance backed by measurement", last in its list (after "SmCo magnets"), so no "or" follows "backed by measurement" (G-3 re-review of the leftovers, M-1; at the defaults "(a warmer minimum temperature, SmCo magnets or a smaller variation allowance backed by measurement narrows it)"); that re-review passed at round 2, with no text changed (help review, `## BL-059 gearbox check`). BL-068 filed (risk: mechanical, priority 3): the hot-minimum Problem can read two equal numbers while its check fails ("Hot-low torque 0.9300 N·m is below the required 0.9300 N·m"), the rounding BL-059's MF-1 fixed for the gearbox Problem.

## 2026-10-14 — Magcoupler guidance slice 2: notes that say what to change, plain words on screen (BL-056, branch magcoupling/guidance-2)
- Plain-words guard (T1, ruling R2): `BANNED` and `plain_text_violations` moved from `gui::guidance` to `engine::explain::plain` (pure std; `gui::guidance` re-exports both), so the engine's tests (gate 4) can hold the notes to it. Twelve patterns added (spec 3.4): "M1 audit", "skeptic", "Plan A-", numbered decisions, "drawing.py", `.rs` names, the three sheet names Metal design, Temperature design and Shaft clamps where they point into the spreadsheet (followed by `!`, by ` C` or ` (C` and a digit, or by ` sheet`: on their own they are input group headings, as in the fix text "Raise Boss outside diameter (Shaft clamps)"; owner-level deviation D-1), "Calibration!", "correction E" with a number, and a whole-word M-number id (M1, M43-13) that leaves the clamp's screw sizes (`SCREW_SIZES`), size designations (M41 x 0.5) and four-digit part names (M5045) alone. Known limit: an id that is also a screw size (M3, M4, M5, M6) passes. The same module holds `word_count` and `sentence_count`, the only word and sentence counters of every cap test (A-16).
- T2, display names and help overlay (`gui::plain_text`): `label_for`, `help_for`, `original_help`, `hint_for` over three path-keyed tables. `LABELS` holds spec 3.5's nine canonical names (Ring radius, Magnets per ring, Magnet face gap, Required torque, Production variation allowance; Gap radius R_g, Pull-out torque (nominal), Hot-low torque, Cold-high torque); `HELP` and `HINTS` ship empty for T7 and T8, and the Size tab's three hint keys (`SIZE_TAB_HINT_KEYS`, no inputs) are declared here for them. Wired at every label and help read: input rows and tooltips, the Size tab's gap and count hovers, the dashboard rows and every result tooltip, fix text (C-3 on display labels: still the same seven inputs carry a group heading), equation term names and the trace banner, and both searches (display label, catalogue label, path, cell; "gearbox" still finds 3 inputs). The results table and every export keep the catalogue labels; the Size tab's Outside diameter keeps `SIZE_TAB_ALIASES` in fix text. Tests: real overlay keys, the canonical names, fallbacks, the help mechanism on a stand-in table and the public accessors over every input's engine help, guard, glyphs, caps (plain.rs's counters) and no torque called guaranteed over every row, one quantity per name, a renamed Key design input drawn under its display label only; `every_input_has_help_from_the_engine_or_the_overlay` is ignored until T8 (87 inputs without help today).
- Show spreadsheet details (T3, S2-5): a header toggle, off by default and GUI-only (`MagcouplingPanel.show_details`; in no design file, share link or undo step; kept by loads, undo and reset all), reaches every view through `Readouts::new(marks, show_details)` and gates every provenance line: the input and result hover paths and cells (and the original spreadsheet note where the help overlay replaced one), the dashboard's correction ids with a one-line legend, the results table's cell and marker columns, the Equation panel's cell line, its corrections line (titles on hover) and its term-path hovers, and the assumptions' Source lines. The marker hover now reads "Differs from the original spreadsheet:" with each title and was/now values at the default design (no repo path). The status line at the defaults says "Example design" (hover: "The starting values from the original spreadsheet."; amends G-6), and the plain-words guard has no exempt phrase left. Exports and search are unchanged. gui-smoke's first view names the toggle; `app.rs` pins it. An ignored test (`every_hover_text_with_the_details_off_is_in_plain_words`) waits for T10. Review round 1: every hover site the toggle gates has a headless off-and-on test (the results-table row, the geometry callout list and the screw table added), on shared `test_support` helpers (`tooltips_at_once`, `hover_frames`, `assert_hover_names_path_only_with_details`).
- T4 (note structure): `Note` gains `change` (each `Change`: an input path, a `Step` (Raise, Lower, Choose or Enter), the result it moves and which way, the effect) and `watch_out` as a list; the 17 notes are converted with their text unchanged (still Reviewed). The Equation panel draws What to change (the input's display label a link to its row, or plain text: held by Auto, set by a part pick, at its slider end), Watch out, and a closed Where this comes from fold for the sources. New: `engine::picks::set_by_pick` (its tests nudge every input on every pick), `SliderRange::step_from` (one step for fixes and change entries), `gui::sizing::held_by_auto`. The slice-2 cap tests (one `cap_problems` rule) are written and ignored until T18; `docs/analyses/2026-10-14-magcoupler-notes-before.txt` records the notes before any rewrite, for the owner's sign-off sheet.
- T5 (live values and probes): `{result.path}` in a note's sentences, change texts and watch-outs is drawn with the design's number and unit (`engine::explain::note_text`, `gui::explorer::live_value`; "no prediction" where a result has none, or is computed from an out-of-range end effect). `engine::explain::probes` holds the five probe designs (default, 8 magnets per ring, 304 back iron, Y30 on custom dimensions, 20 mm ring radius through the Size tab's mapping) and `step_input`, one slider step of an input (a blank optional input from its seed: `OPTIONAL_SEEDS` moved to `engine::api`). `tests/explain.rs` checks every placeholder at the five designs and every change entry's aim; `notes.rs` holds `REWRITTEN` and the plain-words rule for the notes already rewritten (sources included); the ignored `notes_for_review` writes the per-note physics review sheet (`NOTES_FOR_REVIEW=id1,id2 NOTES_REVIEW_OUT=<file> cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture`). Review round 1: a live value's "here" clause matches whole words ("where" and "there" no longer count); `engine::explain::plain::audit_ids` (`TC-`, `PA-`, `CG-`) is checked over every note text by both the notes' plain-words rule and `every_note_reads_in_plain_words_at_every_probe_design`, so the audit-id check outlives `REWRITTEN` (T18 deletes only the list and its test); `notes::Sense::holds` is the one aim rule the aim test and the review sheet share.
- T6, one name per quantity on the rest of the screen: the Size tab's readouts read Required torque and Hot-low torque (the input row's and the dashboard row's names); Outside diameter and Overall length gain hovers that say what they move (the ring radius; the magnet length of both rings), and the footnote says the same (O-7); Auto offers "ring radius" and "meets the required torque"; the fit Why and Problem and the Auto-on first line say ring radius; the plots name the nominal pull-out, the required torque, the production variation allowance, Magnets per ring on the pole sweep's axis and "Below the required floor" for the sweeps' amber status (it compares the nominal pull-out with the required floor); the torque-rotation curve reads "Torque (nominal)" (only its peak is the pull-out); the pole sweep's line names the smallest ring radius its inner magnets and the keyed bore allow; the open equation's header and the plot readouts use the display labels; the production-variation assumption is named as its input. The sweep columns keep the workbook's "Nm" (locked by `tests/schema.rs`) and the screen writes N·m (`format::with_unit`). Not renamed: the geometry callout's "rotating OD" (cup and cap only, unlike the Size tab's Outside diameter, which counts the shaft boss), "magnet count" in the Auto dropdown, and the export-only "Target hot-low torque". Review round 1: the three names the Size tab and the plots draw beside `LABELS` (Required torque, Hot-low torque, Pull-out torque (nominal)) are typed once, as `plain_text::REQUIRED_TORQUE`, `HOT_LOW_TORQUE` and `PULL_OUT_NOMINAL`, which `LABELS` uses and `size_controls::REQUIRED`, `HOT_LOW`, `plots::REQUIRED` and `PULL_OUT` alias; the four Size-tab label hovers are one table-driven panel test.
- Help batch A (T7): every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft, key and clamps groups has on-screen help (35 empty helps filled, 15 workbook helps replaced on screen through `gui::plain_text::HELP`; the engine keeps the workbook text), and a weak hint line (`gui::plain_text::HINTS`, drawn by `gui::input_ui::hint_line`) sits under each Key design row and each Size-tab control (the old Minimum OD line under the controls folded into its hint; the Size tab's Minimum OD, Outside diameter and Overall length are keyed `SIZE_TAB_HINT_KEYS`, not input paths). Every direction was probed with `compute_all`, and the text follows the non-obvious results: the outer magnet thickness adds torque only up to a point and the inner one usually; fewer magnets usually lose torque and, if flat, enlarge the cup; a larger ring radius can lower the pull-out at 8 magnets, so its help claims no torque direction; a larger bore moves the clamp screws outward; the hot-day ambient lowers the hot-day pull-out, not hot-low; the length hints name the clamp demand. G-3 review PASS in round 2 after round 1's eight must-fix items (four helps, four hints; seeds S-13 and S-14 accepted, reworded where the seed's words were wrong): `docs/analyses/2026-10-14-magcoupler-help-review.md`. Tests: `plain_text::help_tests` (batch A coverage, the hint keys, no digit, no typed label, the review sheet; `help_for_review` ignored, it writes the G-3 sheet), three panel drawing tests (under its row or control, once, inside the inputs side at 930 points with wrapping); T2's all-rows test now also rejects a brace. Two panel tests whose subject fell below the 1024-point fold moved to the tall harness (the order toggle; Auto's locked free-variable row).
- T8, help batch B (`gui::plain_text::HELP`): 52 empty helps (Calibration and model, Materials, Housing and retainers, Thermal and demagnetization) and 31 overrides (the grade-mode density, rating and coefficient helps, two of them results; the stored 3D fields, each saying it is a stored value for the example design that does not follow yours; "Placeholder", "ASSUMED" and "Illustrative"; the cell id, correction ids and the unsourced "4140 at 1.5 T" on screen; the Hcj coefficient's advice to type a positive value the slider clamps away; the stale high multiplier, joint screw count and drag "None"). Every input now has help and the coverage test is on; a help key is an input or a result (T2's all-rows test; per A-6 T7's duplicate key tests were already gone, so the brief's Step 4c was dropped). The toggle's original note is the spreadsheet's own text: for a help a correction reworded, the registry's workbook text (none for the adhesive shear modulus and the Hcj coefficient); T2's stand-in test covers that branch. 172 direction claims on 12 named designs pin the helps to the engine (`batch_b_tests::CLAIMS`), with pins for the boss and the cap setting the Outside diameter only as the largest part. G-3 review PASS in round 3 after round 1's four must-fix items (seed S-17 accepted for the prototype's magnetic constant; the driving rise names the hot-day starting magnet temperature; the cap sets the Outside diameter only when larger than the cup and boss; the steel specific heat names the keys, screws and lock tabs, which keep it at a non-magnetic back-iron pick) and round 2's one (only another magnetic back-iron pick replaces the steel specific heat; the default 4140 keeps it): docs/analyses/2026-10-14-magcoupler-help-review.md, Batch B (T8). Ruling 5: no "Corrected note" toggle line; three engine gaps (the space claim leaves out the boss, ferrite eddy heat at the NdFeB conductivity, the daily-cycle screen at the recommended bondline) go to T20 as BL-061..BL-063, "verify first". Review fix round 1 (I1, DRY): the four cluster caveats (the stored 3D fields, the bench prototype, the unmeasured defaults, the optional adapter) were typed in 28 rows; each is now typed once, as a macro (`stored_3d_caveat!`, `prototype_caveat!`, `guess_caveat!`, `adapter_caveat!`) the rows are built from with `concat!`, so no string on screen changed. The cluster test now checks that each cluster is exactly the rows ending with its caveat and pins the stored-3D sentence word for word (A-7; T9 can define `STORED_3D_SENTENCE` from the same macro).
- T9 engine wording: every Rust-only input and result help (and two labels with cell ids, and the autofit label) rewritten in plain words, at most 30 words in two sentences; the help of the five results and the screw-length column whose correction already rewords it, and the sweeps' swept-variable column; `1, 3, 5 (workbook)` -> `1, 3, 5 (default)`; the stored-3D label "Fixed 3D fields from the example design" with a hover that promises no live model, and one stored-3D sentence (`STORED_3D_SENTENCE`, amendment A-7) ending every stored 3D input's help. `tests/data/input_schema.json` re-blessed (help and one choice text only). New tests: `tests/schema.rs` (4), `gui/dashboard.rs` (2), `gui/plain_text.rs` (1). G-3 physics review of the strings: PASS in round 2 after round 1's two must-fix items (seed S-15 accepted: the back-iron pick's help names the saturation caution some other steels trip, against the clean-swap trap; the inner Br coefficient used reads "which every ring without a picked grade uses", not "the same for both rings", false on a mixed design); seed S-16 rejected (past the fifth harmonic the pull-out can move far more than 2 %): docs/analyses/2026-10-14-magcoupler-help-review.md, Engine wording (T9). Task review fix round 1: the stored-3D hover opens "The temperature checks use", not "limits" (G-3 S-c: only the reverse fields set the limits; the slip-heating fields set the slip fault's peak, which the verdict compares with them); the harmonic set's help is G-3 S-a's text, "Higher ones matter most for narrow, widely spaced blocks" (input_schema.json re-blessed); the stored-3D sentence is typed once, as the engine macro `temperature::stored_3d_sentence!`; T8's stored-3D cluster is derived from `STORED_3D_INPUTS`; the sweeps' swept-variable column joins the engine-owned checks (`ENGINE_WORDED_COLUMNS`). New tests: `gui/dashboard.rs` (1), `engine/model.rs` (1). G-3 re-check of the two changed strings: PASS, no must-fix; its two should-fix items applied (the harmonic claim test runs with every correction on, as the app shows it, and with the corrections off; the record's S-a figures give the app's values, the corrections-off ones in brackets): Engine wording (T9), Re-review: code-review fix round 1.
- T10 GUI jargon and rationales: map-locks J1-J3, J7, J8, J10, J12, J13, J15 and map-jargon C-4 to C-6, C-8, C-11, A-10 reworded (each string and its replacement in the plan; the stored-field headings say "(example design)", amendment A-7); J16 (the welcome strip) unchanged and now pinned plain (R14). R9: the cup wall's failing verdict is drawn with the wall input's label and the engine's own number on the dashboard, in the results table and in the Equation panel's value line (amendment A-9; `dashboard::verdict_display`, `shown_text`, `DashboardLine::shown_value`); the engine string, the badge and the exports are unchanged (`engine::materials::CUP_WALL_TOO_THIN` names its prefix); the cup wall equation's formula draws its verdict literal the same way in the Equation panel and every readout's hover (`plain_text::shown_literal`). R10: the 8 assumption rationales with jargon rewritten (end effect: within 2 % of free-space 3D for magnets 3.2 mm and longer, optimistic at wider gaps, amendment A-8; calibration: the prototype match compares the circuit, the magnets per ring and the part only); sources keep their cells. The dashboard guide test's rationale exemption is gone; rationales and warning texts are guarded in the engine tests; T3's hover-text check is on (A-8). The longer drawn verdict exposed a layout loop: with Show spreadsheet details on, a value wrapped beside its correction ids widened the resizable dashboard every frame; each value and its ids now flow as one wrapped text. New tests: 11 (2 in `tests/assumptions.rs`, 1 in `tests/material_links.rs`, 8 GUI: 2 dashboard, 2 panel, 1 trace, 1 results table, 1 explorer, 1 typeset); T3's hover-text check, un-ignored, takes the place of the drafted inputs tooltip test (A-8). G-3 physics review of the strings: PASS in round 2 after round 1's three must-fix items (the cup wall equation's verdict literal drawn with the wall input's label; the greyed hover no longer says the pull-out is computed from the pull-out; the knee-fraction rationale scopes the onset calibration to a rated magnet's heating onsets, since a hard ferrite's cold onsets are not calibrated): docs/analyses/2026-10-14-magcoupler-help-review.md, GUI strings and rationales (T10). Task review fix round 1: the Geometry tab's wall-rule line asks for the wall the cup wall check compares with (`materials.backiron_thickness_needed_mm`, 1.919 mm at the defaults), not the verdict's suggestion rounded up to 0.1 mm, which a 1.95 mm wall (on the slider's 0.05 mm grid) passes while the line asked for 2.000 mm; the line names the input through `plain_text::CUP_WALL_INPUT`; the three plain-words tests (`tests/schema.rs`, `tests/assumptions.rs`, `tests/material_links.rs`) share one check, `tests/common::plain_problems`. New test: `gui/geometry.rs` (1). Text re-review 1 (the G-3 re-check of that line, MUST_FIX, one item): the rule's wall is rounded up at its last digit (`geometry::mm_at_least`), so a wall shown equal to it passes; rounded to the nearest digit, 19 of the 379,620 slider designs (poles x design flux density x wall) showed the wall meeting the rule while the check failed (12 poles, 1.23 T and a 1.95 mm wall: the rule's wall is 1.9501 mm, shown 1.950 mm, now 1.951 mm), and none now; the line is unchanged at the defaults (1.919 mm). New tests: `gui/geometry.rs` (2). Text re-review 2 (the G-3 re-check of the rounded-up line): PASS, no must-fix; its SF-3 (a garbled sentence in the record) is applied; SF-1 (`mm_at_least` meets its guarantee only to within a few ulps, a case no slider design reaches) and SF-2 (the cup wall guide's Problem line rounds the same need to the nearest digit, so it differs from the line in the last digit at about half the failing slider designs) stay open for the controller: docs/analyses/2026-10-14-magcoupler-help-review.md, GUI strings and rationales (T10), Re-review: code-review fix round 1 (re-check 2).
- T11: new teaching note `torque_from_shear` ("How the coupling makes torque": the shear stress on the gap cylinder, T_2D = σ·2π·R_g²·L, the ring radius set apart from the magnet count; what to change: magnet length, ring radius, magnets per ring) as Start here step 1; `START_HERE` reordered to spec 3.3 (12 steps until T14 and T15 add theirs); the step-1 title pin and the Start here hover (`explorer::START_HERE_HOVER`) follow. Its sources are plain words (amendment A-13; `REWRITTEN` holds it). Reviewed at the five probe designs (`docs/analyses/2026-10-14-magcoupler-notes-review.md#torque_from_shear`, created here): PASS in round 2 after round 1's two must-fix items ("endless-ring" glossed as "magnet ends ignored"; σ grows with "stronger magnet grades and magnetic steel", since a 304 back iron is steel but not magnetic); four should-fix items stay open there for the controller. New tests: the Start here order (`SLICE_2_ORDER`), the note's two engine claims, the note's own caps and sentence-0 check (`torque_from_shear_keeps_to_the_slice_2_caps`), and Review Focus RF-1 (amendment A-25 T-1: with Auto on, the magnet-length entry is plain text held by Auto, a click on it opens no row, and the note's live values are the design shown's, checked on the pull-out watch-out, which moves with the magnet length, against the stored inputs' copy). The aim on the blank axial override is checked by T5's aim test from its seed (amendment A-14), so no separate blank-input test was added.
- T12: rewrote `harmonics` ("Block fill and the magnet pattern (harmonics)", now owning the fill rows), `back_iron_factor` ("Back iron: why steel behind the magnets adds torque") and `pullout_angle` ("Pull-out: the most torque before the rings slip"), each with what to change and its watch-out, sources in plain words (A-13; `REWRITTEN` holds all three). Reviewed at the five probe designs (`docs/analyses/2026-10-14-magcoupler-notes-review.md`, `#harmonics`, `#back_iron_factor`, `#pullout_angle`): PASS in round 5 (controller rulings 9 and 12 raised T12's round limit) after seven must-fix items: "spaces", not "small gaps", between blocks; the outer-part entry narrowed to a wider, equally thick block that fits, naming the shaft-joint and gearbox loads and the grade's heat tolerance; "Magnetic steel" behind each ring, since 304 is steel but not magnetic; the inner-part entry narrowed to a same-size, stronger-grade block, its pull-out in proportion to Br unless calibration changes. Every trade-off is joint-neutral ("shaft-joint and gearbox loads rise", ruling 11); four should-fix items stay open there. Diagrams (R7): the torque-angle diagram draws the fundamental alone dashed with its 90° tick (TC-22; the summed curve and its marked peak stay); the flux diagram letters each block N or S and the end-fringing diagram brackets c_end·τ_p/2 at each end (TC-24); the square wave's label reads "blocks and spaces"; the heating curve's labels are T15's. New tests: the three notes' engine claims (harmonics past the fifth under 2 % at the probe designs, the low-fill third harmonic, a wider equally thick outer block against a thinner one, the free-space materials, a same-size, stronger-grade block's pull-out in proportion to its Br, and the bench factor ending on the 304 design), the torque-chain notes' what-to-change and fill ownership, Review Focus RF-4's harmonics part (A-25 T-4: at the defaults the outer block's manual width and thickness entries name the part pick), the three diagram fixes and a no-overlapping-labels check over all six diagrams. T11's caps test is now `the_rewritten_notes_keep_to_the_slice_2_caps`, driven by `REWRITTEN`, so T18 deletes it with that list, and a note task that appends its id needs no per-id `cap_problems` assert (A-15). Task review fix round 1: two harmonics fields were wrong off the probe designs. The harmonic-set entry ("adding more moved the pull-out under 2 % where checked") now reads "keep at least 1, 3, 5; more for narrow, widely spaced blocks, where higher harmonics can move pull-out": with the example's library parts, 36 of 353 fitting slider designs move by more than 2 % (up to 43.9 % at 4 magnets on a 30 mm ring). The σ line ("σ_1 = … of the total σ = …") now reads "at pull-out the fundamental gives σ_1 = …, the summed harmonics σ = …": a same-sign third harmonic subtracts at 90°, so σ_1 exceeds σ at 114 of those designs. New test: `harmonics_claims_hold_off_the_probe_designs` (tests/explain.rs). Text re-review 1 (ruling 7): the entry passed; the σ line had one must-fix, applied. σ_1, the fundamental at the summed curve's pull-out angle, set beside σ read as the harmonics' effect, with the wrong sign where the angle leaves 90° (the 20 mm design showed +10.1 % where the set lowers the pull-out by 12.09 %; 34 of 289 grid designs). The line now reads "the fundamental alone peaks at …, the summed harmonics at σ = …" (`model.amp1_Pa`, σ with the set at 1); the test pins that at the five probe designs and a 17.5 mm ring, the sign of σ − A_1 against the 2D torque's change, and σ_1 < σ < A_1 on the 20 mm design. Text re-review 2 (the re-check of that line): PASS, no must-fix; the fundamental alone peaks at A_1, and σ is the summed curve's maximum at every design the app reaches (22,200 design-sets: five variants, 4 to 40 magnets, every Size-tab ring radius, every harmonic set). Its two optional should-fix items stay open for the controller: SF-1 (the note's square-wave diagram calls the magnet pattern, a field, "the fundamental", while the line's is the shear stress; the review offers "the fundamental alone gives a peak shear stress of …") and SF-2 (the test's 2D-torque sign pin follows from its set-1 pin; pin the pull-out's sign at the probe designs instead): docs/analyses/2026-10-14-magcoupler-notes-review.md, harmonics, Re-review: code-review fix round 1 (re-check 2).
- T12b, the shaft-clamp section paused (owner 2026-10-15, ruling 11, options 1A-4A; BL-064): one switch, `engine::clamps::PAUSED` (true), hides the slotted clamp, the shaft key and the adapter-joint screws (every `clamps.` input and result and the `materials.screws.` classes only the clamp reads: `clamps::SECTION`, `clamps::hides`). Hidden while paused: the Clamp tab, the clamp input groups and sections (the workflow group is renamed Shaft and hub joint: the bore and the keyway depth stay), the dashboard's clamp rows, guide and Why link, the results table's clamp rows (search, failing filter, counts, heading badges), the two clamp assumptions (view and banner), the geometry's key and clamp-boss check, every Used by link into the section, and `clamp_preload`'s Start here step (`notes::START_HERE_ALL`, `start_here_for`, `START_HERE` = `start_here_for(PAUSED)`: 11 steps now, 13 once T14 and T15 land; `shown_when`, `shown`; the note stays in `NOTES`). The clamp checks count toward no verdict (`guidance::counts`: the status line counts 7 guides and a hidden check vetoes no other fix; `failing_checks` leaves them out). Unchanged: the engine, Python parity, design files and share links (a pre-pause file loads), and the exports (no single filter point, pinned to the full dump: the CSV, the JSON and the spreadsheet keep the section). Joint-neutral texts ("shaft-joint and gearbox loads"): `torque_from_shear` change[0], the bore, both manual Br and the shaft boss helps, the magnet-length and overall-length hints, the cap material's engine help (input schema re-blessed). Functions that branch take `clamps_paused: bool`; the panel's `clamps_paused` field starts from the switch, and the panel tests draw the whole section unless a test pauses it. New tests: both switch states for every branch, the section's inputs move nothing outside it (at every probe design, both slider ends or every choice), the panel passes its pause to the fixes, the header's banner and the trace banner, a pre-pause file and link round-trip, the exports keep the section, the claims of the new texts, and the guard, folded into the glyph test (`every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`, over `every_shown_text`): while paused no text it collects names the clamp (every drawn text, and the hover texts of inputs, dashboard rows, table rows, callouts and the named static hovers, plus the shown notes, Start here, the counted guides, the Why links' reasons and the shown assumptions). The changed reviewed texts went to the text re-review. Text re-review 1: eight of the nine changed items passed; the bore help had one must-fix, applied. It named only the wall past the keyway, but a larger bore also thins the hub wall under the magnets, which the hub check reads: on the 8-magnet probe design the hub check fails from a 15.5 mm bore, keyed or keyless (hub wall 2.350 mm against the back-iron rule's 2.399 mm), with 0.65 mm still past the keyway, and for a keyless joint no input-slider design breaks through (the wall stays at least 0.80 mm, though the Size tab's lowest Minimum OD can take it to zero). The help now reads "Diameter of the shaft in the hub bore. Larger thins the hub wall: the hub check can fail, and the wall past the keyway can break through."; `the_joint_neutral_texts_claim_what_the_engine_does` pins that it names the hub check by its label and that, on the 8-magnet probe design with the design's keyway and keyless, the bore's slider top turns the hub check from passing to failing and lowers the hub wall while the wall past the keyway stays above zero. Its two optional should-fix items stay open for the controller: SF-1 (the shaft boss help no longer tells the two bosses apart once the section returns; note it on BL-060) and SF-2 (the cap material help's "6061-T6 uses the inputs" covers its density and specific heat, not its conductivity), as does its note on the "Optional adapter and its joint" section label, whose joint inputs are hidden while paused: docs/analyses/2026-10-14-magcoupler-help-review.md, Batch A (T7), Re-review: changed by the implementer after its review. Text re-review 2 (the re-check of the bore help): PASS, no must-fix; every claim holds against the engine at the five probe designs, with the design's keyway, keyless and the deepest keyway, and Auto with the ring radius free keeps the ring where it was, so the wall still thins. Its optional SF-1 corrected the records' keyless floor (0.80 mm on the input sliders, at the 0.2 mm bondline, not 0.95 mm; zero at the Size tab's lowest Minimum OD). Its two notes stay open for the controller: the Size tab's red line "Exploration only: the keyway breaks through the hub." also shows at a keyless zero wall, where there is no keyway, and the engine's Python-locked input label "Keyed bore diameter" will read oddly beside a keyless shrink fit (BL-064): docs/analyses/2026-10-14-magcoupler-help-review.md, Batch A (T7), Re-review: changed by the implementer after its review (re-check 2). Task review fix round 1: the shared test designs moved to gui::test_support (hot_day, small_boss, only_the_clamp_fails, clamp_at_its_limit), used at every site, and 02-system names the renamed glyph test.
- T13: `end_effect` ("Magnet ends: the length lost to fringing") and `calibration` ("Calibration: what scales the torque estimate", now also owning `calibration.model_torque_Nm`) rewritten (3-5 sentences, What to change entries, watch-outs, live values; sources in plain words, A-13; `REWRITTEN` holds both) and physics-reviewed at the five probe designs (`docs/analyses/2026-10-14-magcoupler-notes-review.md`, `#end_effect`, `#calibration`): PASS in round 3 after seven must-fix items. The magnets-per-ring lever was dropped (in free-space 3D, more magnets do not cut the real end loss at fixed blocks), so the end-effect note offers longer magnets and, only with 3D or bench evidence, the end coefficient, each naming the shaft-joint and gearbox loads (ruling 11); its 3D agreement reads "within 3 % at 3.2 mm or longer" on the example design, "below 3D with fewer magnets or larger rings", and c_end is glossed as an empirical coefficient. The calibration note gives the measured factor only with no magnetic steel behind the magnets (the 304 design matches), the prototype's magnets per ring and B842SH parts; its bench entries are Enter (A-2), and the watch-out says never to edit the prototype inputs to match a design. New `tests/explain.rs` tests pin each claim: the levers' directions, one `probes::step_input` step at the five probe designs (the blank axial length seeded from the inner ring's length; cold-high and cold-high at the gearbox input with them), the constant torque per millimetre, the end-effect check's edge, the calibration rule's match, what it ignores and its step. Open should-fix items stay in the record for the controller, among them SF-C (narrower or tightly packed blocks, B822 on the example ring, read high against 3D, also below a third of a pole pitch, where the second watch-out then said low) and SF-4 (the `end_effect` and `calibration_factor` assumption rationales still say "within 2 %" and "no steel"). Task review fix round 1 (I-1) and its two text re-reviews (MUST_FIX each, MF-1 applied verbatim each): the second end-effect watch-out was "Below about a third of a pole pitch of overlap it reads well low; ...". The unscoped claim was false in the optimistic direction: with the narrower B822 block on the example ring, which fits with the end-effect check OK, free-space 3D reads the model 3.80 % high at 2.5 mm with 10 per ring and 9.29 % high at 2.0 mm with 12. The fix round scoped it to the example design (round 3's W1, "On the example design it reads low ..."), and the first re-review found that false too: the example design has magnetic steel behind both rings, and a 3D check of that circuit reads the model high below a third of a pole pitch (+12.8 % at 2.5 mm, +17.6 % at 2.9 mm) and low only near the 2 mm floor (−4.0 %). It became "Below a third of a pole pitch of overlap f_end can read well high or low; ..." (30 words). The second re-review found that boundary misleading: it read as the edge of the hazard, but on the example design as shipped (steel) the model's largest error, +21.5 % (optimistic), lies above it at 3.66 mm, and in all ten circuits it checked the largest error lies at 0.16 to 0.42 τ_p, so no fraction of a pole pitch bounds it. It is now "At short overlap f_end is least reliable and can read too high or too low; at or below c_end·τ_p there is no prediction and the end-effect check flags it." (29 words). The `tests/explain.rs` test `the_end_effect_note_s_short_overlap_watch_out_names_both_directions` (renamed from `..._is_scoped_to_the_example_design`) finds the watch-out by "short overlap" or "c_end·τ_p", pins the engine side (the example design at the magnet length slider's floor and both B822 designs fit, have a short overlap, below a third of a pole pitch by the test's own measure, have f_end above 0 and read OK) and asserts that the claim names both directions, no one design and no pole pitch. The record has the re-reviews under `### Re-review: code-review fix round 1` and `### Re-review: code-review fix round 1 (re-check 2)`. Still open for the controller: W0 and the rest of SF-C (the second re-review adds B822 reading high above a third of a pole pitch without steel), SF-1, SF-A, SF-4, SF-S (no 3D check with steel backs the steel circuit's end factor; the second re-review's ideal-steel estimate puts f_end about 20 % high from 2.9 to 4.25 mm and 6.65 % high at 12.7 mm on the example design, where sentences[4]'s steel-free "within 3 % at 3.2 mm or longer" invites trust) and the second re-review's new SF-T (pin the 304 design at the floor as the test's low side).
- T14: new note `hot_low_cold_high` ("Hot-low and cold-high: the torque band", Start here step 8): what each edge of the band uses, the shaft-joint and gearbox loads set by cold-high, the 20 °C pull-out that passes hot-low; watch-outs: hot-low is not the hot-day case, and nothing checks cold-high at the gearbox input (BL-059). `HOT_MIN_GUIDE.learn` and the cold-high `ROW_LEARN` entry now lead to it (no dashboard row links an assumption any more). `br_temperature` rewritten ("Heat weakens magnets: torque against temperature"; the alpha records; only a ring with custom dimensions and a picked grade keeps its own coefficient). Both physics-reviewed; tests pin every claim and the wiring. Review (`docs/analyses/2026-10-14-magcoupler-notes-review.md`, `#hot_low_cold_high`, `#br_temperature`): PASS in round 3 after three must-fix items, all in `br_temperature` (`hot_low_cold_high` passed in round 1 and never changed). Sentence 2 says each °C changes torque "by a share of about α_i + α_o" (seed S-5), since the bare figure read as an absolute torque. The grade entry first tied "starts weaker" to any grade nearer zero, false from the Y30 design (every other grade is nearer zero and starts stronger); then its "loses less when hot" read as fewer N·m, while the nearer-zero grade loses more N·m in 30 to 33 of 69 grade pairs. It now reads "a grade whose coefficient is nearer zero loses a smaller share when hot; SmCo, the nearest, starts weaker than sintered NdFeB. Custom dimensions only." Seeds S-6 and S-7 were rejected (S-6 double-counts heat in hot-low and is false below 20 °C; S-7's "exactly" adds nothing a user acts on). New `tests/explain.rs` tests: the two notes' levers at the five probe designs, one `probes::step_input` step each through the file-level helper `stepped` (T13's two closures folded into it), with cold-high at the gearbox input for longer magnets and a colder minimum; the band's sentences (hot-low and cold-high as the pull-out less or plus the allowance, the gearbox relation, the 20 °C pull-out that passes hot-low) and watch-outs; the coefficient input moving only rings without a picked grade, a mixed design too; and the grade entry's claims (`a_grade_nearer_zero_loses_a_smaller_share_when_hot_and_smco_starts_below_sintered_ndfeb`). New GUI tests: the three band rows and the hot-minimum guide lead to the note, and nothing checks a gearbox torque against its rating yet (fails once BL-059 lands). Should-fix items stay open in the record: SF-1 (the operating-temperature entry repeats the watch-out), SF-3 ("before it slips" to "before the coupling slips"), SF-4 and SF-6 (optional tests); SF-5's wording went into the README's `tests/explain.rs` row. Task review fix round 1: `hot_low_cold_high` sentence 0 named neither edge of the band, though spec 3.1 has it answer the rows that link the note; it now reads "Hot-low is the weakest coupling expected at the operating temperature, and cold-high the strongest at the minimum temperature, allowing for production scatter." (it keeps the production-scatter gloss of the allowance and none of S-6's heat wording). `br_temperature` change[3] takes SF-2 ("a hotter day lowers the hot-day torque and the temperature margins; the hot-day start is this ambient plus the driving rise."), since one step also lowers the magnet-limit and adhesive margins the temperature verdict reads. The sentences test pins sentence 0's words and hot-low below cold-high; the temperature levers test pins the entry's words and, at the five probe designs, the three verdict margins falling, the skipping onset's falling wherever it is finite, and no temperature margin rising. Text re-review 1 (ruling 7): both fields PASS, no must-fix. Its three optional should-fix items were applied with no on-screen text change: the sentences test also pins "operating temperature", "minimum temperature" and "expected" in sentence 0 (traps T-E and T-O); the levers test pins the hot-day start as exactly the ambient plus the driving rise; and the record says that the adhesive's hot fatigue margin, a strength ratio and not a temperature margin, rises slightly on a hotter day (the model holds the hot strength fraction fixed while the peak torque falls). The reports are under each note's section.
- T15: new note `when_slipping` ("When the coupling slips": the swing from plus to minus the pull-out per pole pair, the eddy drag, re-locking only as the slip speed falls, the gearbox ripple; it owns `model.gearbox_input_ripple_Nm` and `metal.slip_freq_Hz`), Start here step 9, and the ripple row's Why link moved from `pullout_angle`; `slip_heating` and `thermal_time_constant` rewritten (tau_th and the hot-day start as the panel shows them, closing M43-13; the peak the check uses; the stored slip fields as a watch-out); the heating diagram gains the limit and the time to reach it. Reviewed per note (`docs/analyses/2026-10-14-magcoupler-notes-review.md`). Review (`#when_slipping`, `#slip_heating`, `#thermal_time_constant`): PASS in round 2 after six must-fix items, applied verbatim. The re-lock sentence drops "near zero" (seed S-8: the re-lock slip speed depends on the inertias and the load, and nothing models it). The ripple sentence names its value as the ripple "at its input": the coupling sits on the gearbox output, which carries the full swing, so the unqualified 0.566 N·m read as the gearbox's swing beside its 3 N·m output rating. Both stored-field watch-outs say "those fields" do not follow the design, since the losses and the magnets' limits do follow it. The fault-trip entry says a shorter unbroken slip "can lower" the peak (seed S-9: once one slip event is hotter than the fault slip, a shorter trip leaves the peak unchanged). The conductance source drops its typed −2.48 °C, which no longer reproduces (−2.501 °C at today's defaults); the same figure stays in the conductance assumption's source (`assumptions.rs`). New tests: `notes.rs` (step 9 between the torque band and `slip_heating` in both orders; the rows spec 3.3 gives the three notes; the three ids in `REWRITTEN`); `tests/explain.rs` (the three notes' claims at the five probe designs: the slip frequency, the drag under a tenth of the pull-out, the ripple through the ratio and the efficiency below cold-high at the input; heat = drag × slip speed, the steel parts' speed^1.5 and the shells', cap's and magnets' speed², the other sleeve and liner materials heating less, a thinner shell widening the running clearance; a measured drag replacing the loss and its high case; the lump's start, P/G, C/G, P·t/C, the life's average rise, the peak and the "never" rule; a hot-day start at the limit keeping the peak above it; the stored slip fields as a tripwire; T14's three relative-tolerance closures folded into one file-level `close`); `diagrams.rs` (the heating curve marks the limit and the time to reach it, its five labels pinned in order); `guidance.rs` and `panel.rs` (the ripple row leads to and opens the slip note). Open should-fix items stay in the record for the controller: SF-2 (the gearbox watch-out with the mounting fact), SF-3, SF-4, and the tests SF-R2-1 (pin "can lower") and SF-R2-2 (the reverse fields in the stored-field tripwire). Task review fix round 1: `slip_heating` said "steel" where only magnetic steel is skin-limited (the engine takes the speed^1.5 form only with the magnetic-steel circuit, and a 304 back iron, steel but not magnetic, takes the whole-thickness speed² form), so sentences[2] and [4] now read "In thick magnetic steel ..." and "Without magnetic steel back iron ..."; SF-6 was applied verbatim to the two "aluminium" sources ("the weak-field, whole-thickness loss of non-magnetic body parts", "non-magnetic body parts at their own specific heat"), since a 304 pick is the body too; spec 3.1's sentence-0 rule took SF-1 (`when_slipping`: "Past pull-out the coupling slips: one ring turns against the other, and the torque between them, carried by the gearbox, swings back and forth.") and SF-5 (`thermal_time_constant`: "... mass changes only how fast, which the time constant measures.") verbatim. Tests: the claims tests pin "magnetic steel" (every "steel" in `slip_heating`'s sentences), "gearbox" and "time constant" in the two sentence 0s; new `the_non_magnetic_body_is_the_picked_metal_not_only_aluminium` (the two sources, and a non-magnetic pick's own conductivity and specific heat as the body's at the 304 design); new `notes.rs` `the_slip_notes_keep_the_caps` (A-15's per-id form, A-28c item 9, beside `REWRITTEN`). Text re-review 1 (ruling 7): all six fields PASS, no must-fix. Its optional SF-3 (a test) was applied with no on-screen text change: the slip-temperature claims test also pins "mass changes only how fast" in `thermal_time_constant`'s sentence 0, the words that keep it clear of the trap "more mass lowers the steady temperature". Its SF-1 (`slip_heating` sources[3]: "whole-thickness" names the thin-sheet form the cited report replaces) and SF-2 (sentences[4]'s unchanged "heat through their whole thickness", loose for a thick non-magnetic hub) would change on-screen text, so they stay open for the controller. The reports are under each note's section.
- T16: `demagnetization` and `ferrite_cold_demag` rewritten and reviewed: no stored reverse field or editable default quoted as a fact about the design (a test holds it), a watch-out for the grades whose onsets the rating shift raises (BL-057), the ferrite note without the NdFeB knee diagram and its Start here step labelled "(ferrite rings only)" (`explorer::FERRITE_ONLY`); a test holds every note to never saying the coupling protects the gearbox (BL-059). `clamp_preload` is not rewritten: it keeps its earlier text and review, hidden with the paused shaft-clamp section (ruling 11; its rewrite comes with BL-064). `demagnetization` ("Demagnetization: permanent loss from heat and slipping", Start here step 12) now also owns the two onset margins; `ferrite_cold_demag` ("Ferrite magnets: demagnetization risk at the cold end", step 13). Both change[2] entries enter the like-poles field (`Step::Enter`, A-2). Review (`docs/analyses/2026-10-14-magcoupler-notes-review.md#demagnetization`, `#ferrite_cold_demag`): PASS in round 3 after five must-fix items, applied verbatim. The grade entry names NdFeB and "usually" (Recoma 30, rated 250 °C, lowers the limit; bonded BCN-19 can too) and says the grades rated above N42SH are weaker. The limit sentence says "each NdFeB ring's limit" (a ferrite ring's is its rating) and that onsets are shifted to match "the magnet's" rated maximum temperature, not the grade's: the library's N50 arcs carry their vendor's 60 °C rating. For the same reason the grade-list watch-out reads "At their grade's own rating" (the arcs are shifted down, by 12.98 °C). The ferrite watch-out names the slipping field alone (the rings-aligned field's cold onset, 170.6 °C, is below Y30's 250 °C rating). New tests: `notes.rs` (the caps per id in A-15's form, the rows spec 3.3 gives, no editable default typed, the gearbox guard over every note); `explorer.rs` (only the ferrite step's bar adds the label); `tests/explain.rs` (the two notes' claims at the five probe designs and on the example design: the levers, the skipping onset as the lowest and the limit as it less the margin, the shift to the magnet's own rating, the lower ring limit governing, the onset margins; the hotter-rated grades' limit, torque and adhesive trade, on both rings and one; the grade list derived from every grade's offset; the stored fields the example's whatever the design; ferrite's positive coefficient, the Y30 hot and cold limits, the check failing at any coldest temperature and n/a on NdFeB, Y30's tested range; one ferrite ring, and an entered field; with shared helpers `slider`, `Rings` and `on_grade`). Open should-fix items stay in the record for the controller: SF-1 to SF-7 (round 1), SF-R2-1 (restated as SF-R3-1: pin the arcs' 60 °C shift) and SF-R2-2. Task review fix round 1 and its text re-review 1 (MUST_FIX, MF-1 applied verbatim): the ferrite note's Y30 entry named only the reverse move's trade (back to a library NdFeB part), not what a move to Y30 costs (spec 3.1's trade-off rule). The fix round made change[0] "custom dimensions with the Y30 grade model a ferrite ring, with far less torque; a library NdFeB part trades the cold risk for the hot." The re-review found "far less torque" true only against a part of the same size, ruling 12's defect class: "Custom dimensions (manual)" keeps the manual fields' own size (B842's by default), so from a smaller library part that fits the same move need not cut the torque (M5045: x0.910 to x1.275 of the pull-out, a rise in 7 of the 8 ring and design cases; B442 keeps 63-71 %). change[0] now reads "custom dimensions with the Y30 grade model a ferrite ring, with far less torque than same-size library NdFeB parts, which trade cold risk for hot." (25 words). At equal size Y30 keeps at most 27.75 % of a library part's pull-out (15 parts, either ring, five probe designs); at the manual defaults it leaves 25-28 % on one ring at the NdFeB probe designs and 7-8 % on both, and the hot-minimum check, already red at the example's requirement, goes deeper in. `the_ferrite_note_s_entries_hold` pins "far less torque" and "same-size"; at the four NdFeB probe designs on either ring or both, under half the pull-out and a just-passing hot-minimum check turned red; at every probe design, for every library part on either ring, Y30 at the part's own size (the model's block size asserted equal) under half its pull-out; and M5045 on the example's inner ring, which fits, not under half with Y30 at the manual defaults. The record has the re-review under `### Re-review: code-review fix round 1`. Still open for the controller: SF-A (the coercivity source, carried from round 1's SF-3 and the task review's M-1) and SF-B (with Y30 on both rings the thermal-mismatch screen turns red at the default, 8-magnet and 20 mm designs: its hot end rises to the adhesive's limit, and it uses the NdFeB expansion and modulus for a ferrite ring; engine scope, a backlog line beside BL-062, verify first). Text re-review 2 (the re-check of that entry): PASS, no must-fix. "Far less torque than same-size library NdFeB parts" holds in all 150 probe cases (at most 27.75 % of the part's pull-out and hot-low, 29.74 % of its cold-high) and on a wider grid of 80,256 cases (at most 31.88 %: −40 to 150 °C, every back iron, 4 to 20 magnets, 9 to 30 mm rings), and "which trade cold risk for hot" holds ring by ring at the Y30 design. Two new optional should-fix items join SF-A and SF-B, open for the controller: SF-C (the bench factor applies only to B842SH on both rings in the prototype's free-space circuit, so an entered bench torque near half the stored 1.8 N·m or below can reverse an on-screen cross-part torque comparison at such a design, this entry's and `demagnetization` change[0]'s alike; a verify-first backlog line and an optional test pin) and SF-D (the same-size loop could also assert hot-low and cold-high): docs/analyses/2026-10-14-magcoupler-notes-review.md, ferrite_cold_demag, Re-review: code-review fix round 1 (re-check 2).
- T17, the six warning notes: each `a5.*` note rewritten as What it is / What to change / Watch out and physics-reviewed at the five probe designs (`docs/analyses/2026-10-14-magcoupler-notes-review.md`). `a5.low_saturation` is now "Back-iron wall and saturation", the Why of the cup-wall check, and owns the wall rule's rows (`model.gap_flux_density_T`, `materials.cup_wall_suggested_mm`, `materials.cup_wall_check` beside `model.bsat_T` and `model.backiron_needed_mm`); `a5.high_conductivity_sleeve_or_liner` owns `materials.sleeve_sigma_S_m`; `a5.ferromagnetic_sleeve_or_liner` got the shortest rewrite (R13: no listed sleeve can fire it). R11: `src/engine/warnings.rs` tests pin through `compute_all` which back-iron and sleeve choices fire each rule, and that each back-iron warning's note names them. The other titles: "No magnetic steel behind the magnets", "Sleeve or liner that conducts better than 316L", "Unplated carbon or low-alloy steel", "Hub expands much more than the magnets". Review (`#a5.non_ferromagnetic_back_iron` and the five sections after it): PASS in round 2 after 13 must-fix items in three notes, applied verbatim. Restoring steel now names its costs in both open-circuit and expansion notes: the cup-wall check comes back, with plating, steel's mass and slip heat, and higher shaft-joint and gearbox loads (from 304, 4140 takes cold-high at the gearbox input from 0.508 to 0.805 N·m, past the 0.6 N·m rating, and turns the check "Too thin"). The steel torque quoted is the unsaturated one, and only no-back-iron designs get the bench calibration (seed S-10, amended: the default design has the prototype's magnets and parts but steel). The wall sentences state the calculator's rule, not physics (trap T-I), the rule rather than the check uses the design flux density (seed S-11, amended: the check reads "No back iron" at the 304 design), and the magnets-per-ring wall entry was dropped: at the default 12 magnets do not fit, yet one step turned the check OK while the real need stays above the wall. The expansion note names 304 stainless as a hub that fires and says no listed magnetic steel does (seed S-12, amended: a typed 4140 coefficient can fire it), and glosses lap-shear strength. Ruling 5's "usually" (SF-1) and SF-2 to SF-6, SF-8 and SF-9 stay open in the record for the controller. A-25: `explorer.rs` tests `the_wall_note_s_design_flux_entry_names_the_pick_that_sets_it` (1018 sets the design flux density, so the wall note's entry on it is plain text; 12L14 leaves it a link; RF-4) and `the_sleeve_entry_at_the_bottom_of_its_slider_is_plain_text_with_the_reason`; `notes.rs` `the_wall_and_sleeve_notes_own_the_rows_slice_2_gives_them`, and the six ids join `REWRITTEN`. With this task every note shown is rewritten (`clamp_preload` keeps its slice-1 text, hidden while the shaft-clamp section is paused): T4's cap tests pass with `--ignored` but for that note, which T18's gates skip while paused, and `release_notes_are_reviewed` passes (T18 un-ignores them). Task review fix round 1: the plating note's back-iron entry offered 304 without its costs (spec 3.1's trade-off rule), so change[1] now reads "416 or 17-4PH need no plating here but trip the low-saturation caution; 304 needs none but loses steel torque and fires the expansion caution." From an unplated steel at the probe designs, 304 leaves 51-63 % of the pull-out (default 2.6885 -> 1.6974 N·m) and, at four of the five, turns the bond check from below to above the lap-shear strength. `the_plating_note_s_back_iron_entry_names_the_cost_of_each_pick_it_offers` (`warnings.rs`) pins the three costs at every probe design. `each_back_iron_warning_note_names_the_library_picks_that_fire_it` now also fails when a note names a pick that does not fire its rule. The `explorer.rs` tests of unlinked note entries (A-25 T-4 and T-7, T12's and T4's) share helpers (`unlinked_on`, `unlinked_with`, `auto_off`, `slider`, `assert_entry_plain_with_reason`, and `test_support::drawn_count`). A text re-review of the field follows (ruling 7). Text re-review 1 (MUST_FIX, MF-1 applied verbatim): the fix round had also dropped "here" from the 416 and 17-4PH half, which rounds 1 and 2 had passed with it; without it, under sentence 0's "rust in humid air", the entry reads as saying 416 does not rust, and 416 is the least corrosion-resistant stainless offered. "here" is restored (the text above), and the cost test now pins "need no plating here". The 304 half passed; SF-1 (ruling 5's "usually", with the open-circuit note's sentences[4]) stays open (`#a5.uncoated_low_alloy_steel`, "Re-review: code-review fix round 1"). Text re-review 2 (the re-check of the restored entry): PASS, no must-fix. "Loses steel torque" holds beyond the probe designs: 304 keeps at most 0.637 of 4140's pull-out over 3707 fitting layouts of magnets per ring, ring radius and face gap, and at most 0.679 over every harmonic-count choice. From an unplated steel the 304 pick also turns the daily-cycle screen from below to above the fatigue endurance, which the expansion caution covers (SF-3, a record correction to re-review 1's "no other check turns"). Re-review 1's observation on the `CTE_MISMATCH_LIMIT_PER_C` comment is closed: its 17.7e-6 is 304's mismatch with the magnets, not its expansion coefficient. SF-1 and the optional daily-screen assert and comment rewrite stay open (`#a5.uncoated_low_alloy_steel`, "Re-review: code-review fix round 1 (re-check 2)").
- T18, the release gates on: the note caps (3-5 sentences of at most 25 words, 1-5 change entries, 0-2 watch-outs of at most 30 words, the sentence-0 rule), the plain-words rule over every note at the five probe designs and the help coverage test run for good; `release_notes_are_reviewed` is un-ignored and covers every note a user can open by id (Start here, warnings, and with the `gui` feature the check guides' and the dashboard rows' Why notes); `every_note_names_its_committed_review_section` ties every note's record to its section of `docs/analyses/2026-10-14-magcoupler-notes-review.md`. The gates hold the notes shown: while the shaft-clamp section is paused they skip `clamp_preload` (its slice-1 text, kept for un-pausing), and T12b's `only_the_clamp_note_is_hidden_and_only_while_the_clamp_section_is_paused` pins that to the switch. `start_here_is_fourteen_steps_and_thirteen_while_paused` and the explorer's `the_start_here_bar_names_every_step_it_shows` pin the full 14-step order and the 13 steps shown while paused; `only_the_review_tools_are_ignored` fails if a check is ignored again. Task review fix round 1: the plain-words rule is one function, `notes::plain_words_problems(note, value)`, run by the all-notes test with each probe design's live values and by its own unit test, so the test of the rule tests the gate that ships; the bar test pins the first and the ferrite bars as drawn ("Start here 13 of 13 (ferrite rings only): ..." while paused) and replaces the older ferrite-step test; the Start here order test no longer repeats the equation-cover loop that `ids_are_unique_and_every_link_resolves` runs.
- T19, reachability (spec 3.7, S2-8): "Why: <note>" links under Magnets per ring and Outside diameter (how the coupling makes torque), at the end of the Required and Hot-low rows (the hot minimum check's note) and under both copies of the end-effect banner (the end-effect note), each one link (never the bare title: F11); the empty Equation panel names Start here and Explain; under an equation without a reviewed note, or a value with no equation, Explain offers the row's own Why note or assumption, else the closest reviewed note upstream (a link), else the closest assumption's rationale, else the row's reason or "No teaching note is shown here for this value." with "Start here walks the main teaching notes in a suggested order." (`explorer::note_fallback`); no dashboard row shows "No teaching note". Headless click tests for every new link (clicked on the text's last character, `test_support::last_characters`); the narrow-panel change-entry test (A-25 T-6); the explorer tests and the review dump skip what the paused clamp section hides, and no offer leads to a note not shown (A-28g). G-3 review of the wording and of each row's offer: PASS in round 3 after round 1's three must-fix items (Start here walks the main notes in a suggested order, in the hint and the empty text; `NO_NOTE` no longer says no note "covers" the value) and round 2's one (`NO_NOTE` claims only what the panel shows, true also under the values with no equation record); `mass.total_g`'s offer, "No magnetic steel behind the magnets", was accepted (`docs/analyses/2026-10-14-magcoupler-help-review.md`, Reachability strings (T19)).
- T20, docs and smoke: the README, docs/ai and BL-056 brought in line with the slice (20 notes, 19 shown while the clamp section is paused; 14 steps, 13 while paused; the overlays, the toggle, reachability; `the_docs_count_the_notes_and_the_steps` keeps the README's and 03-structure's counts true, and while the clamp section is paused both docs' "20 notes, 19 shown while the clamp section is paused"); the gui-smoke prompt gained the R14 steps (the details toggle in the header, a Why link opens a note, an input tooltip shows no spreadsheet details, the dashboard scrolls to Magnets fit at 1280 x 800), pinned by `src/app.rs`'s smoke test (`WELCOME` is re-exported from `gui`); `notes_signoff_sheet` (ignored) writes the owner's old-against-new sheet from T4's snapshot (`notes::before::parse`), for the notes shown, with `live_value`'s values at the default design; `the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice` tests it in both switch states. Rulings 5 and 8 and A-28h: BL-061..BL-063 filed after each claim was checked against the code and `compute_all` (the space claim's diameter leaves out the shaft boss; the magnets' eddy heat uses the NdFeB conductivity for every grade; the daily-cycle screen reads the recommended bondline, not the entered one); BL-059 records the owner's 0.6 N·m input limit (3 N·m at the 5:1 output); BL-060 and BL-064 say what the pause defers and what un-pausing turns on; 04-memory's rating offsets for the SmCo and bonded grades recomputed (Recoma 26 -60.2 °C, Recoma 30 -0.45 °C, Recoma 20 +217.3 °C, BCN-19 +94.3 °C).
- T21 whole-branch review fix wave (2026-10-16; the review and its must-fix list: `.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/final-review.md`), one commit per item. MF-1 (ruling 10 a): `torque_from_shear` sentences[3] and [4] swapped, words unchanged, so the sentence that first speaks of each pole says what a pole spans (R8; `the_torque_note_glosses_each_pole_where_it_is_first_used`). MF-2 (ruling 10 b): its pull-out watch-out says what pull-out is, "The dashboard's pull-out torque, the most the rings are expected to pass before slipping, is T_2D corrected ..." ("expected" by controller ruling 13, as the note's text re-review asked: the dashboard's number is a nominal estimate, not a bound), and comes first, before the R_g² watch-out, so the term is glossed where a reader first meets it (`the_torque_note_glosses_pull_out_where_it_is_first_used`, which also checks "expected to pass"; the RF-1 panel test reads the first watch-out). MF-1 and MF-2 were re-reviewed (T21 text re-review 1; record under `## torque_from_shear` of `docs/analyses/2026-10-14-magcoupler-notes-review.md`), and the gloss as applied passed T21 text re-review 2 (same section). MF-3 (ruling 10 c): `torque_from_shear_claims_hold_on_the_engine` pins the note's engine claims at all five probe designs, computed from the engine: two fewer magnets keep R_g and grow the cup, one gap step wider lowers σ, 4140 gives a higher σ than 304, and T_2D rises with each two more magnets while the blocks fit. MF-4 (SF-S): `end_effect` sentences[4] scopes its steel-free 3D band to the example design and names the steel case, which that design ships with ("... with magnetic steel behind the magnets, the default, f_end likely reads high."; `the_end_effect_note_s_3d_band_names_the_steel_case`); BL-065 (risk: physics) asks for a 3D check of the end factor with permeable steel before any note states a steel figure. Re-reviewed: PASS, confirming watch_out[1] too (T21 text re-review 1; record under `## end_effect` of the notes review record). MF-5: the end-effect assumption's rationale gives the note's band and scope ("Checked on the example design without its steel, it is within 3 % of free-space 3D for magnets 3.2 mm and longer; with magnetic steel behind the magnets it likely reads high."; it said 2 %, which T13's 3D data exceed) and says, as the note's short-overlap watch-out does, that it is least reliable for short magnets, reading too high or too low (T21 text re-review 1, MF-B; as applied, PASS in T21 text re-review 2; G-3 records in the T10 section of `docs/analyses/2026-10-14-magcoupler-help-review.md`), and the calibration rationale says "no magnetic steel ... B842SH parts" (a 304 back iron is steel, and its design gets the measured factor); `tests/assumptions.rs` reads the band from the note's sentence (deviation D-5 from R10's literal 2 %). MF-6: BL-056's notes list the spec deviations the build makes, D-1 to D-4 (controller ruling 2, A-26a to A-26d; A-26e deferred to BL-060) and D-5.
- T21 release gate, run 1 (2026-10-16): gate 1 failed on two linkage tests of the animation export, not on the slice. Their helper `data_span` looked for `;\n` after `const DATA = `, but this checkout's `animation_template.html` has CRLF line endings (`core.autocrlf=true`; the index keeps LF; git rewrote the working copy at the 2026-10-14 checkout of main, after slice 1's gate had passed), which `include_str!` embeds as they are. The helper now takes the rest of the line with `str::lines` and drops its `;`; `the_data_is_found_whatever_the_line_endings` reads the real page with LF and with CRLF, so an LF checkout catches a regression too. The page itself was right under either ending.
- T21 release checks (2026-10-16): the whole-branch review passed after its fix wave (record: .superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/final-review.md; notes edited by the wave re-reviewed under their sections of docs/analyses/2026-10-14-magcoupler-notes-review.md); `gate.sh` GATE PASS with gate 12 running (the oracle venv reference/magcoupling-py/.venv), on run 2 after the linkage test fix above; the web build and a gui-smoke run passed (hub, linkage; magcoupler: first view, the Show spreadsheet details toggle, share link, Auto solve, design file, a Why link opening a note, an input tooltip without spreadsheet details, the dashboard scrolling to Magnets fit at 1280 x 800, a clean console). The owner's sign-off sheet is docs/analyses/2026-10-14-magcoupler-notes-signoff.md (the 19 notes shown while the shaft-clamp section is paused). The owner signed off the sheet on 2026-10-16; merged to main and pushed; the deploy run is checked in the session that pushed.

## 2026-10-13 — Magcoupler guidance slice 1: one torque number, Size-tab solver, check guides (BL-055, branch magcoupling/guidance)
- Why: the owner went from 10 to 8 magnets, lowered the torque target and expected the coupling to shrink; it did not, and nothing on screen said why. A read-only audit the same day (14 agents; 143 of 144 findings survived adversarial verification; the result is in the session scratchpad, not the repo) found eight themes. This slice takes four: the size model is invisible, two unlinked torque targets, red checks give no cause or remedy, and the teaching notes are unreachable (in part). Audience: mechanically literate engineers who are not magnetics specialists. Spec `docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md`.
- Owner decisions (2026-10-13): **G-1** order: this slice, the note rewrite with three new notes, a torque sensitivity card, then guided lessons. **G-2** one torque number: the sizing target is `metal.required_min_Nm` (reverses A2-9 and M41-9 for the target only; the mode and free variable stay GUI sizing state). **G-3** every new physics sentence gets one fresh-context session-model physics review per slice; numbers in it come from results, never literals; no Draft/Reviewed gate for hints. **G-4** the solver moves to the Size tab, magnet length stays the default free variable, the outcome names what changed. **G-5** the first view is Size and torque with a dismissible one-line welcome strip (reverses M42-2); Explain stays off by default (A4, M43-2). **G-6** defaults stay the workbook example (decision 28); the screen labels it and counts its failing checks live. **G-7** an Auto toggle locks the matching Size-tab slider and re-solves after other edits (existing `SizingRunner`, debounce and file format). **G-8** release: merge to main and deploy once `gate.sh` reaches GATE PASS and the physics and whole-branch reviews pass.
- One number: `SizingState` is the mode and the free variable; the target is `metal.required_min_Nm` (`REQUIREMENT_INPUT`), edited by the Required slider as an ordinary design edit (undo, files, links). Files and links still write `target_torque_Nm` (equal to the requirement) and keep `DESIGN_VERSION`; a file whose target differs from its requirement loads with the requirement, and in Torque -> Magnets with a notice (`target_mismatch_notice`, with units); a forward-mode file or link gets none, since before G-2 every file wrote the default target (2.5 N·m), which that mode never used (final review, 2026-10-14). A requirement of zero or less is left out of a saved file and refused by the solver.
- Size tab (now the first view, under a dismissible welcome strip): Required, Hot-low with its live gap and "Pull-out torque (nominal, at T)"; "no prediction" while the end-effect banner shows. Under the controls: the neighbour magnet counts' pull-out and cup OD (from `compute_all`, in words when a neighbour's blocks do not fit or its end effect is out of range), a first line that follows the Auto state (`magnet_count_line`: "The ring radius stays the same when you change the magnet count; the cup OD below can still change." with Auto off, `AUTO_RESIZES_RING` or `AUTO_RESETS_LENGTH` with Auto on the ring size or the magnet length, none with Auto on the count; physics review 3, N4), the smallest outside diameter that fits (with Minimum OD unset, how far it could go and that lowering Minimum OD reaches it), the cap, cup and boss that set the outside diameter, and the magnets plus housing lengths.
- Auto solver: the Auto toggle replaces the Key design group's mode switch (now a status line with a button to the Size tab); the matching slider is locked, other edits re-solve after the debounce, the outcome names the change ("Not reachable: best Y at X" when it cannot); ring size searches the Size tab's diameter range (Minimum OD included) and shows the design through the Outside diameter slider's own mapping (`with_ring_radius`), naming Outside diameter. Opening the Size tab no longer forces forward mode.
- Dashboard: a status line ("Workbook example design: fails N of 8 checks" at the defaults, "Your design: ..." otherwise), a Magnets fit row, Problem, Why and Fix with fix links under each red or amber row, and a Why: link on every row (reviewed note, assumption, or "no note yet"). A fix is listed only where its move helps and turns no passing check red; a failing check with no such fix says so; two failing checks that pull one input both ways each carry a `trade_offs` note; `guide_notes` adds `side_effects` (raising the face gap lowers the pull-out) and the sleeve/liner note no check covers. The fit guide drops "Raise Outside diameter" where no slider diameter fits (physics review 3, N3), and `within_minimum_diameter` drops "Lower Outside diameter" where Minimum OD stops the slider. While the reverse-field inputs hold their defaults, the temperature Problem says the cold and slip-onset limits come from the default design's 3D fields. `Direction::CustomHotterGrade` takes a ring's magnet part to Custom dimensions with a grade rated hotter than the part. `gui/guidance.rs` holds the guides, the status and the consequence lines as pure functions.
- Engine additions (torque model untouched): `engine::size_controls` diameter and length breakdowns, `smallest_fitting_diameter`, `ring_radius_range` and `with_ring_radius`; `sizing::solve_in` over a caller's range (`solve` is `solve_in` over the variable's range), `FreeVariable::grid_in`, `SizingError::InvalidRange`; `engine::temperature::CURE_MARGIN_MIN_C` names the adhesive cure margin the verdict already used.
- Unchanged: the engine's labels, help and verdict strings (workbook text lock, `tests/python_schema.rs`), the results table and exports, input metadata, the deviation registry and the file schema.
- Tests, written first and red before each change: guide coverage and real check paths; every fix path an input with a slider, choices or a picker; every shown fix moves each failing term toward passing and turns no passing check red; reviewed notes, existing assumptions and reasons for none; a Why link on every dashboard row; no banned pattern in any guidance or consequence text; every Why one sentence of at most 30 words; Problem lines at the default and at a passing design; consequence numbers equal `compute_all` of the neighbour and follow an edit; `smallest_fitting_diameter` (fits at the result, not 1e-3 mm below, both range-end forms, flats and arcs); the Required slider, one undo, the load notice and file and link round trips; Auto (locked slider, re-solve after the debounce, idle frames edit nothing, not reachable, ring size below the old 9 mm floor with a lowered Minimum OD, the locked Outside diameter slider showing the solved diameter); first view, welcome strip and status counts; the owner's 8-magnet case; glyphs for every new string.
- G-8 (2026-10-14): done: the fresh-context physics reviews 1-3 of every new Why, Problem and consequence sentence (G-3), the whole-branch review (its must-fix items applied), `gate.sh` GATE PASS and a gui-smoke run (first view, share link, Auto solve, design file, clean console, the dashboard scrolling to its last row at 1280x800, the locked Auto slider greyed with its value). Not yet done: merge, push and the deploy check.

## 2026-10-12 — Adjustable manual minimum OD (BL-054)
- **Minimum OD** changes the direct diameter slider's lower limit without changing the current inputs, torque or design undo history. It is a view preference; reset/load starts fresh and a loaded smaller design exposes its actual OD as a pinned starting floor. Existing files/share links preserve actual smaller dimensions without a format change.
- The explicit adapter minimum can extend below the inherited 9 mm radius range; the bore/bondline envelope, fixed boss, cap/thread bounds and existing upper radius bound remain. Existing model/metadata/inverse sizing are unchanged. The centre view flags magnet-fit failure and keyway breakthrough and displays the existing thin-wall guidance.
- Red regressions recorded for the absent control and rejection of 8 magnets at 38 mm. The 21 adapter tests cover explicit floors, fixed-part limits, invalid settings, loaded sizes and round trips; headless UI regressions cover unchanged inputs/torque/history when the minimum changes, 38 mm edits, keyboard focus, idle frames, undo/redo, load/share/reset, fit warnings and nonfinite typed input. Independent geometry and integrated review PASS. The full required gate ended **GATE PASS**, including 529 app unit tests, native/WASM checks and clippy, shipped-feature guards, all 1,159 Python oracle tests and current differential data. The user's authorization to publish this magcoupler UX work applies.

## 2026-10-11 — Direct magcoupler size exploration (BL-053)
- New **Size and torque** tab: outside diameter, overall length, magnet face gap and magnets per ring, with live expected and hot-low torque and the geometry. The detailed input groups remain available under **Detailed inputs**; results, equations, assumptions, exports, files and share links retain their existing format.
- Diameter edits move the inner back apothem, cap OD and cap thread together, retaining signed housing allowances, magnet cross sections, shaft dimensions, gaps and wall thicknesses. Overall length edits use the existing axial magnet override and housing allowance/floor rules. Existing fit, clearance, wall and cap/thread checks remain visible; no numerical torque prediction is presented outside the end-effect model's range.
- Regression first failed on the absent tab; the out-of-range headline regression also failed before its fix. The 13 adapter tests cover literal flat/arc dimensions, 8 magnets, the boss floor, housing allowances, metadata bounds, no-op/invalid requests and round trips; headless UI tests cover same-frame torque, gap, idle inputs, inverse-to-manual transition and undo/redo. Independent geometry and final integrated review PASS. The full required gate ended **GATE PASS**, including 525 app unit tests, native/WASM checks and clippy, the shipped-feature guard, all 1,159 Python oracle tests and current differential data. The user authorized publishing the completed change to update the website.

## 2026-10-06 — colesorkness.com/tools/ live; the old-link redirects made permanent
- Merged (3a324bb) and deployed. Live checks passed: `/` 307 to `/tools/`; the hub and both apps 200; old linkage.colesorkness.com links (share links included, with and without the calculator's slash) land on the new addresses with `m` once; the press share link shows its 21,506 N peak; the calculator's share link opens its design; Tools -> Magnetic coupling calculator opens `/tools/magcoupler/` in a new tab. Vercel also passes the query through, so `?tool=magcoupling` arrives at the calculator as `/tools/magcoupler/?tool=magcoupling` (harmless: the calculator reads only `m`).
- Then the old-host and `/magcoupling/` redirects became permanent (308, decision 4A); `/` to `/tools/` stays temporary (T-1). `tests/web_layout.rs` pins both.

## 2026-10-06 — colesorkness.com layout and the standalone calculator (branch linkage/url-move)
- The web output (`linkage-sim-rs/web/`) is laid out for colesorkness.com/tools/: the hub page at `/tools/` (`web/tools/index.html`; `/` redirects there), the linkage app at `/tools/linkage/` (`web/tools/linkage/`), the calculator at `/tools/magcoupler/` (`web/tools/magcoupler/`, was `web/magcoupling/`). `vercel.json` adds `trailingSlash` and temporary redirects: linkage.colesorkness.com to the new paths (share links keep `?m=` by an explicit capture; `?tool=magcoupling` goes to the calculator), `/magcoupling/` to `/tools/magcoupler/`, `/` to `/tools/`. `tests/web_layout.rs` keeps the pages, `vercel.json`, `.gitignore` and the scripts in step.
- Share links: `SHARE_URL_BASE` (`https://colesorkness.com/tools/linkage/?m=`, `gui/state/file_io.rs`) and magcoupling-rs `PUBLIC_BASE_URL` (`https://colesorkness.com/tools/magcoupler/`).
- The calculator is no longer embedded: `gui/calculator_window.rs` deleted, Tools -> Magnetic coupling calculator opens the site in a new tab (`menu_bar::MAGCOUPLER_URL`), linkage-sim-rs drops its magcoupling-rs dependency and the wasm-only rfd; gate 10 guards the calculator's builds only, gate 11 checks only wasm-bindgen against the CLI pin. BL-038 to BL-040 fixed by removal.

## 2026-10-06 — Animated HTML export (Part B, branch linkage/animation-export)
- File -> Export Animation (HTML)... saves one self-contained page that plays the mechanism through its sweep: links, shapes, the actuator, force-zone points, weights, joint reactions, readouts with lbf, an actuator force (or driver torque) chart, play/scrub controls. Desktop and browser (`export::download::download_text`).
- `gui/export/animation.rs` rebuilds each sample's pose from `SweepData::body_angles` and `coupler_traces` (a mount-point actuator's hidden bodies included) instead of solving again; reactions per frame at `sweep_time`, now shared with the sweep.
- `animation_template.html`: the page, one data slot; every `<` in the embedded JSON written `\u003c`.
- Final-review fixes: the page is drawn in the canvas's frame (turned by the mounting angle); a coupler point named like a body's first pin no longer misplaces the body (`sweep::trace_key`, shared); labels and readouts use the canvas's `format_magnitude` and zero band with lbf, share rows the Weight Breakdown's names; the item is disabled while the sweep is being recomputed; the player draws only on a new sample and no longer spreads every value into one call.

## 2026-10-06 — Round shapes and the contact point (Part A, branch linkage/round-geometry)
- `BodyGeometry.shape`: rectangle (default, left out of files) or circle (diameter in width); schema 1.2.0.
- Force zones gain a third application mode, the shape's contact point (`at_contact_point`): its extreme point against the force, found at every pose, so a wheel's load acts at its bottom. One helper, `force_zone_application`, now decides overlap and point for the evaluation, the overlap ratio, the independent check and the canvas.
- Property panel: Add rectangle / Add circle, a shape switch, Diameter; the stub Draw Geometry / Redraw buttons removed (BL-044). Force editor: Overlap centre / Locked point / Contact point; dropping the F marker locks it.
- Final-review fixes: geometry edits are one undo step each and mark the file changed; sizes must be > 0 and finite; Offset sliders unclamped.

## 2026-10-05 — Weights draw over force zones (BL-042, branch fix/weights-over-force-zones)
- Canvas weights are drawn in `render_overlays` after the force elements. A force zone's
  overlap highlight is an additive yellow, and it covered a weight on the user's press tool.
- New test `a_weight_inside_a_force_zone_draws_over_the_zone_highlight` checks the paint order.

## 2026-10-05 — Share links sweep the sender's crank range (BL-041, branch fix/share-link-sweep-offset)
- `AppState::load_from_json_str` recomputes the driver display offset before the sweep. A link
  from a mechanism whose driven link is drawn at an angle (the user's press: 44.3 deg) swept a
  range shifted by that angle until an edit recomputed it.
- New test `share_url_sweeps_the_senders_crank_range_for_a_rotated_crank` pins it.

## 2026-10-03 — Magcoupling spreadsheet export: the final review's fix wave (branch magcoupling/xlsx)
- The check-level fills are a fixed palette: the badge colours of egui's dark visuals (green
  5AC878, amber FF8F00, red FF0000), which the standalone page forces. The linkage app's
  calculator window may draw in light visuals (green 14823C, amber FF6400), and the fills do not
  follow it. The README, `gui/xlsx.rs`, decision X-11 and 03-structure now say so; the writer
  is unchanged (the values and the level words were right).
- The spreadsheet tests' `snapshot` fixture, written twice (`gui::spreadsheet`, `gui::xlsx`), is
  one helper in `gui::test_support`.
- Still open for the user before merge: the plan's Decisions header says X-2 to X-12 are not yet
  confirmed and 04-memory asks the user to confirm them (left as committed); opening a downloaded
  file in Excel or LibreOffice (neither is installed on the build machine).

## 2026-10-03 — Magcoupling GUI: spreadsheet download of the current layout (branch magcoupling/xlsx)
- "Export spreadsheet" (header) and "Export XLSX" (beside the results table's CSV and JSON exports) save `magcoupling-results.xlsx`: the design shown laid out as the page, values only (decision X-1: no formulas, no round trip). Sheets Summary (export time, version, the share link as text, the sizing state, the banners, the dashboard with badges, the material warnings, the corrections applied), Inputs (the order the inputs side shows; group, section and Advanced headings; changed, assumption, Advanced and Key design flags; every input once), Results (by physics chain with heading levels, check levels as the page's fills, the equation as plain text where a record exists; every result once) and Assumptions. `gui::spreadsheet` lays the sheets out (tested directly), `gui::xlsx` writes them with rust_xlsxwriter 0.99.1 (pinned; gate 11 checks both lock files), read back in tests with calamine and zip (dev-dependencies). The sheets' order and headings come from the page's own code (`InputGroup::plain_sections`/`advanced_sections`, `InputSection::heading_in`, `results_table::table_lines`, `entry_level`), which the inputs side and the results table call too; the table sheets freeze and filter their header row, group headings are grey, and a text past Excel's 32,767 UTF-16 units is cut at a whole character. Share links (both apps) are now compressed by flate2's zlib-rs backend (rust_xlsxwriter's zip); tests pin a link each app wrote before and check it still decodes. `PanelRequest::SaveFile.contents` is now `Vec<u8>`; both hosts write the bytes (app::files::write_file; linkage export::download::write_file through download_bytes, with an Excel workbook save filter). The plan records the decisions table and the measured web bundle growth.

## 2026-10-03 — Magcoupling GUI: views stay in their own area (branch magcoupling/overlap-fix)
- The centre views no longer paint into or take clicks in the Equation panel (the callout list ran ~35 points into it on a 1280 x 620 page; dragged tall, geometry, plots, results and clamp up to ~200): the tab row and banner, then the view, are each laid out in a child ui confined (max rect and clip rect) to the space above the panel (`panel::confined`), every view fits the height left (the geometry drawing shrinks below 160 points to leave the callout list three rows, `drawing_height`; plots lose their 150-point floor; the list, clamp and table scroll areas their 64-point one), and the panel opens at 45 % of the centre region (at least `MIN_PANEL_HEIGHT` 120) with its drag stopping `VIEW_STRIP` 160 points below the region's top (`explorer::panel_heights`, replacing `PANEL_HEIGHT` 320). Tests: `no_centre_view_paints_or_takes_clicks_in_the_equation_panel`, `inside_an_egui_window_no_centre_view_reaches_into_the_equation_panel`, `the_equation_panel_starts_at_about_45_percent_and_its_drag_stops_short_of_the_view`, the per-view `*_stays_in_a_short_region` tests, `the_panel_starts_at_45_percent_and_its_drag_leaves_the_view_its_strip`.

## 2026-10-02 — Magcoupling GUI ordering: the final review's fix wave (branch magcoupling/ordering)
- Out of the end-effect range (f_end <= 0, audit M9) the results table showed badges the
  dashboard hides: its check rows, its headings' worst level and "Failing checks only" read
  `check_level`, which had no greying rule, so the short-magnet design showed the hot minimum red
  and listed it as failing, and the recommended screw (sized from the invalid pull-out) green.
  One rule now, `dashboard::greyed_by_end_effect` (out of range and in `END_EFFECT_ROWS`), read
  by `check_level` and so by the dashboard (`dashboard_lines` takes its badge from `check_level`),
  the table's badges and headings and `failing_checks`. Wider greying (`model.verdict`, the
  hot-day checks) stays M41-11's deferred work. Tests: the dashboard's
  `out_of_range_end_effect_drops_the_pull_out_checks_from_the_table_and_the_failing_filter`,
  the panel's `out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too`.
- The trace's banner and Clear trace sit at the foot of the inputs side, laid out bottom up in
  the same frame, so starting or ending a trace moves no row above them: "a second click on the
  label ends the trace" lands on the label again (on top, the banner pushed every row down by its
  height). Test: `a_trace_moves_no_input_row_so_a_second_click_on_the_label_ends_it`.
- "Traced only": the table says the trace marks nothing (`NOTHING_TRACED`) only when the trace
  marks none of the rows the failing filter lets through (`results_table::empty_text`); a search
  that hides the traced rows says no result matches. A group heading counts the traced rows among
  those it shows, so a search no longer draws "Torque (3, 28 traced)". The locked free variable's
  label (Torque -> Magnets) traces it: `input_row` takes `locked` and disables only the reset,
  the note and the value widgets. Tests:
  `a_search_that_hides_the_traced_rows_says_so_and_the_headings_count_what_it_shows`,
  `the_locked_free_variable_s_label_still_traces_it`, `the_empty_table_names_the_filter_that_empties_it`.

## 2026-10-02 — Magcoupling GUI: smarter ordering of the inputs and the results (branch magcoupling/ordering)
- Inputs (decision O-1, O-2): the workflow order is the default, `inputs::WORKFLOW`:
  Requirements and operating conditions, Magnets and rings, Gap and clearances, Housing and
  retainers, Shaft, key and clamps, Materials, Thermal and demagnetization, Calibration and model,
  every input in exactly one section (a coverage test lists any input added without one); the
  adhesive and its bondlines share a section, the optional adapter and its joint too, and the
  fields stored from a 3D run sit in view beside the 3D reference torques; the clamp and screw
  factors, the screw classes, the optional adapter and the model constants (27 rows, O-4) sit
  under each group's closed Advanced heading. "Workbook groups"
  brings back the package groups; the order is panel state only (design files and share links are
  byte-for-byte unchanged: a test compares the design and the link across the toggle). A filter
  box (O-5) matches label, path or cell as the results search (`format::search_haystack`, shared)
  and shows the matching rows alone; a leaf term focusing a row clears it.
- Results (O-6, O-7): the table groups the results: the headline (the dashboard's 17 rows, open),
  eight chains (the six A-3 chains, each with the results of its nested groups the scope leaves
  out, `CHAIN_PREFIXES`, then adhesive and mass), then "Other results" by package
  (`result_groups.rs`), each result in exactly one group (the first that lists it); each heading
  shows the worst level of its checks, and a heading click while searching is ignored; "Engine
  order" brings back the flat table, and both exports keep the engine order. `dashboard::CHECKS`
  lists the 30 design checks (the dashboard's 7 and 23 more, with levels in `verdict_level`);
  their rows carry the dashboard's badge (none on the two greyed by f_end <= 0, as on the
  dashboard), and "Failing checks only" lists the red then the amber ones.
- Tracing (O-8, `trace.rs`): a click on an input's label frames every explained result it drives
  (the registry's `downstream`: 392 of the 1086 results have an equation record; an input that
  reaches none says so), a click on a result frames the inputs it reads (`upstream_inputs`) unless
  an input's trace marks it or it has no record (the trace then stays), in the selection colour
  through `Readouts::mark` (`TermColors::with_marked`), with a banner and a Clear trace button at
  the foot of the inputs side, per-group traced counts and a "Traced only" table filter for an
  input's trace.
- The linkage app's calculator window is unchanged (its tests pass). Docs: the magcoupling README,
  03-structure, 04-memory, the spec's M4 layout amendment.

## 2026-10-02 — Magcoupling M5: the final review's fix wave (branch magcoupling/m5)
- The calculator window keeps, frame included, to the screen below the menu bar and the band
  where egui grabs its edge, so it never covers the Tools menu that toggles it:
  `draw_menu_bar` hands `CalculatorWindow::set_menu_bar_bottom` the bar's bottom every frame (the
  window, shown first, uses the last frame's). Before, it was kept to the whole screen less its
  margins at the bottom right: a drag up (or a tap that jumps the pointer onto its X, which egui
  applies as a drag) put it over File..Tools, and on a screen shorter than the window (a laptop's
  browser) it sat over the menu bar from its first frame and its bottom 14 points were cut off:
  egui caps a window's contents at its constrain rect less the title bar but not the frame, so
  such a window is larger than the rect, and egui clips (and takes no press on) what sticks out.
  Now the window is constrained to that area with its contents capped at the area less its
  chrome (`window_chrome`: the frame's margins and the title bar, as egui lays them out), so it
  is always whole. Tests: `the_calculator_window_keeps_below_the_menu_bar` (menu_bar; 1400 x 900
  and 1400 x 650, dragged up, painted whole, Tools clickable down to its lower edge),
  `a_small_screen_keeps_the_whole_window_on_it` (fills an 800 x 600 screen exactly, pinning
  `window_chrome`, also after a corner drag), and
  `a_press_on_a_tooltip_or_egui_s_debug_layer_leaves_the_keyboard_where_it_is` (the popup rule's
  Tooltip and Debug layers, untested before). Seen in the browser at 1400 x 900 and 1400 x 650.
- Docs: the popup rule names the Tooltip and Debug layers (02-system, 03-structure, the module
  docs, the magcoupling README); the README's web build lines (both bundles through the guard,
  `wasm-bindgen-cli@0.2.114`, the browser's file chooser for Load design); WASM_DEPLOYMENT's cache
  key (both lock files).

## 2026-10-02 — Magcoupling M5: the calculator in the linkage app (branch magcoupling/m5)
- `linkage-sim-rs` depends on `magcoupling-rs` by path with feature `gui` (one egui: 0.32.3 in
  both lock files; gate 11 passes). Tools > Magnetic coupling (`gui/calculator_window.rs`)
  toggles an `egui::Window` holding one `MagcouplingPanel`, created on the first opening and kept,
  with its design and undo history, while the window is closed; nothing in it touches the model.
- Keyboard (decision M5-5): the window has the keys from its opening or a press on it (its
  resize band included) until a press on the linkage app: its panels, canvas or menu-bar buttons
  (an open menu's items and drop-down lists move nothing). Only then are the panel's undo keys
  on, and the window removes the frame's key events but egui's zoom keys, so neither the linkage
  shortcuts (undo, save, new, delete) nor the canvas keys (arrow nudge, F, Escape, Enter) run on
  them. `update` shows the window before any key reader; the shortcut block moved to
  `handle_keyboard_shortcuts` and a source test pins the order. Documented, not handled: Tab
  focus moving between the parts (BL-040) and the window covering the recovery prompt (BL-039).
- On a screen without room for its frame (a hidden browser canvas reports 0 x 0) the window skips
  the frame, so egui does not squeeze its stored size; it comes back in its place at its size.
- Files: saving through `export::download`, "Load design" through rfd natively and on the web
  (the linkage wasm build now has rfd). On the web, `?tool=magcoupling` opens the window at start
  and share links made there open `/magcoupling/` on the same server.
- BL-037 fixed (decision M5-7): the window title is sent only when it changes, never on wasm32.
- Gate 10 guards the linkage app's native and wasm32 builds too (`LINKAGE_*_ARGS` in
  `magcoupling_shipped.sh`), with a negative control per shipped build (the calculator's native
  app gained one); `build_web.sh` pipes the linkage release build through the guard and
  `deploy-web.yml` now builds both bundles through it.
- gui-smoke's linkage step opens `/?tool=magcoupling` and checks the window, the registry's log
  line and that no Title warning is logged.
- The linkage wasm bundle grows from about 10.5 MB to about 12.0 MB with the panel (rfd and the
  panel), accepted (decision M5-10); the calculator's own bundle is unchanged (5.0 MB). The
  magcoupling crate, its engine and its tests are unchanged. Nothing pushed.
- Docs: both READMEs, `docs/FEATURES.md` (Magnetic Coupling Calculator) and
  `docs/guides/WASM_DEPLOYMENT.md` (`build_web.sh` builds both bundles through the guard; the web
  build carries rfd for the picker). Backlog: BL-038 (one design file picker for both hosts),
  BL-039 and BL-040 (the documented keyboard and prompt edges).

## 2026-10-02 — Magcoupling M4-3: final review fix wave (branch magcoupling/m4-3)
- The whole-branch review found no Critical or Important issue; four Minor findings, fixed with a
  failing test first where they change behaviour:
  - The H under a Σ (and under an `arg max`) is a term: the harmonic set's selector
    `coupling.max_harmonic`, which every Σ record lists among its terms. `TermColors::of` gives it
    a palette colour at the Σ, before the body's terms (the registry's term order, so the other
    terms of the eight Σ records move one colour on), and the typesetter draws H in that colour
    with its path. Its term-list swatch is coloured, its Assumptions row ("Highest odd harmonic
    summed") is framed while a Σ is in view, and a click on the H shows and frames that row.
  - The Equation panel's term list scrolls sideways when the centre region is narrower than its
    rows; the tag column (an assumption, "depends on a modified assumption") was cut off at
    1280 x 800.
  - Closing the Equation panel (Close or the header toggle, `Explorer::close`) clears the leaf
    term's input-row frame, which stayed until the next trail (decision M43-12 now says so).
  - A selector's code reads with its choice's label after a colon (`5: 1, 3, 5 (workbook)`,
    `1: steel circuit`) instead of nested parentheses.
  - The M5 readiness test `the_panel_works_inside_an_egui_window` now opens a value, follows a
    term and a leaf assumption term inside an `egui::Window`, then checks the window's size, the
    Equation panel's lists, no egui ID clash, and an unchanged design.
- The engine, parity, differential data, registry and teaching notes are unchanged. Nothing
  pushed.

## 2026-10-01 — Magcoupling M4-3: equation explorer, assumptions, teaching notes, pickers (branch magcoupling/m4-3)
- Equation explorer (`gui/typeset.rs`, `gui/readouts.rs`, `gui/explorer.rs`): an egui typesetter for
  the A-3 markup (stacked fractions, scripts, strokes for √, ∈ and the ceiling and floor brackets,
  a large Σ, `arg max`, scaled delimiters, `cases` braces, `where` lines); every displayed value
  (dashboard, results table, geometry callouts, clamp tab, a readout line over each plot) shows its
  equation on hover with each term coloured, and the terms' values and input rows are framed in
  those colours on screen; a click opens the docked Equation panel (bottom of the centre region:
  breadcrumb, the equation large, value and corrections, term list, "used by"); clicking a term
  drills in, an input term opens its group, scrolls to and frames its row. The registry is built
  once per process. A selector compared for equality with a code shows its choice's label (an
  ordering keeps its numbers); a test checks every drawn run against the record's plain rendering.
  A long walk shows its last 8 crumbs; the term list's swatches fade while another equation's
  value is hovered.
- Assumptions: the inputs side's Assumptions view (each assumption's rows, rationale and source),
  the header banner with "Reset to workbook defaults", assumption and affected terms tagged in the
  Equation panel.
- Teaching notes: the Explain toggle shows the open equation's reviewed note with its painted
  diagram (`gui/diagrams.rs`: six kinds); "Start here" walks the suggested order.
- Pickers and warnings (`gui/pickers.rs`): material choices with their library properties; part
  and grade pickers from the A6 tables beside the text fields; the material warnings at the top of
  the dashboard in their severity's colour, each linked to its note.
- Decisions M43-1 to M43-15 (the plan's table); deferred: per-row end-effect greying (M43-15),
  symbol renames (M43-13). The engine, parity, differential data and registry are unchanged.
  Nothing pushed.

## 2026-10-01 — Magcoupling M4-2: final review fix wave (branch magcoupling/m4-2)
- The whole-branch review found no Critical or Important issue; five Minor findings, fixed with a
  failing test first where they change behaviour:
  - Sweep plots whose rows are all NaN show "Nothing to plot" (they showed an empty frame), and a
    sweep that leaves some rows out counts them in a weak note (`plots::rows_not_plotted`).
  - The torque-temperature plot names an empty axis (a minimum temperature at or past the axis end,
    which a design file can hold) instead of saying the values are not numbers
    (`plots::EMPTY_TEMPERATURE_AXIS`).
  - Two poles (a design file only) draw the pocket and the hub round: the 2-gon's corners sat about
    3e17 mm out and egui filled nothing; flats are drawn from 3 poles.
  - Cosmetic: the geometry view's dimension tags sit on a plate of the drawing's background
    (`geometry_view::plated_text`, shared with the clamp drawing's dimension texts); the clamp
    views' titles sit in a band above the views, so the notes no longer run under them, and a note
    that would run past the drawing's right edge (the flange note) moves back inside.
  - Docs: `02-system.yaml` names `geometry_view.rs` among the key files; the M4-2 entry below says
    how the 02-system conflict hunks were resolved.
- Deferred, recorded in `04-memory.yaml` ("Open (M4-2 final review)"): "Screws per clamp 1.000"
  (the shared four-significant-digit format of a workbook number, as in the results table); the
  clamp drawing's 8-point text floor below about 480 points wide; the wasm frame time and a browser
  hover of the callout list (not measured or done by the review).
- The engine, parity, differential data and registry are unchanged. Nothing pushed.

## 2026-10-01 — Magcoupling M4-2: geometry view, plots and clamp drawing (branch magcoupling/m4-2)
- `magcoupling-rs` panel: the centre region is one tab row (`CentreView`): the geometry view (the
  default), five plots, the clamp and the results table, under the end-effect banner when
  f_end <= 0 (decisions M42-1, M42-2).
- Geometry view (`gui/geometry.rs`, `gui/geometry_view.rs`): the end view (cup and pocket, both rings
  of blocks on the flats or as arcs, liner, sleeve, hub, shaft, key) and the upper half side view
  (cap, cup wall, web, boss, and the parts in the cavity, centred), both at one scale, drawn from the
  results of the design shown with the housing in effect, so they follow every edit, the length
  override and Torque -> Magnets. Tagged dimension callouts listed under the views: face gap, corner
  gap (along face 1's normal, from the circle the inner corners sweep to the outer flat), running
  clearance, and the space claim per axis; red when violated (a gap below zero, the
  clearance below its target, an axis exceeded), amber when not a number (M42-3, M42-4). The space
  claim dashed; the autofit wall as a hint; the two workbook design checks (cap thread below the cup
  body OD, boss OD 22 vs 25 mm) as notes while they hold (flag only, no engine change; M42-9). Hovering a
  dimension shows its result's hover text (`dashboard::hover_text`, the M4-3 hook). A claim line
  farther than ten times the pieces' extent (a design file's finite but huge claim) is left off
  with a note, and parts holding a number that is not finite are counted in a note.
- Plots (`gui/plots.rs`, egui_plot 0.33.0, locked to linkage-sim-rs's and added to gate 11):
  torque vs temperature with the variation band, the requirement and the limit; the gap and pole
  sweeps by row status, greyed outside the end-effect model; slip heating vs the governing limit;
  torque vs rotation with the pull-out point. Series from the same frame's results through the
  engine's closed forms, pinned by tests to the engine's torques; the torque plots grey when
  f_end <= 0 (M42-5, M42-6).
- Clamp (`gui/clamp_drawing.rs`): a painter port of drawing.py's end and top views, its texts and
  limits, cuts clipped to the boss, drawing.py's message when no screw fits; the clamp table
  (summary, machining steps, the screw sizes table with the recommended column) (M42-7).
- Results table: the label column flexes (`column_widths`; the value, cell and marker columns keep
  M4-1's 130, 130 and 60 points), so a ~930 px window shows the label and the value (M42-8). `gui-smoke` judges the geometry view in its first screenshot (no click).
- `docs/ai/02-system.yaml`: the three unresolved merge-conflict hunks the M4-1 merge (42ab370) left
  in the magcoupling block and its status line are resolved: A-3's sentence and invariants and
  M4-1's sentence, key files and invariants are kept; where both sides held the panel invariant,
  M4-1's version (HEAD's text plus slider rounding) replaces HEAD's; the status line is one line
  covering A-3, M4-1 and M4-2. The M4-1 sentence is reworded so the block parses as YAML.
- The engine, parity, differential data and registry are unchanged. Nothing pushed.

## 2026-10-01 — Magcoupling M4-1: final review fix wave
- The whole-branch review found no Critical or Important issue; no code changes.
- Deferred, recorded in `04-memory.yaml` (the "Open (M4-2, M4-3)" item): when M4-3 adds the equation text to
  `result_tooltip`, `dashboard_ui` builds its hover text lazily (`on_hover_ui`, as the results table does);
  M4-2 reworks the results table's fixed columns (a ~930 px window shows only the label column without
  scrolling).
- Checked in eframe 0.32.3's source: on an origin that is not secure (plain http, not localhost) the
  web copy of the share link logs an error and copies nothing, with no panic, while the panel's status
  still says "copied"; the deployed site is https, and the linkage app's copy takes the same path.

## 2026-10-01 — Magcoupling M4-1: GUI inputs, dashboard, results table and session (branch magcoupling/m4-1)
- `magcoupling-rs` panel (`src/gui/`): every input generated from the metadata, grouped as the package
  groups them, with the Key design group on top (the A-2 axial length override, blank by default);
  sliders with value boxes, arrow nudges, log scales, values rounded to the step's decimals, edits
  clamped to the range, per-field reset, the changed dot, tooltips; selectors, optional inputs and text
  inputs. The dashboard: the headline with green/amber/red badges from the check verdicts, the
  corrected-vs-workbook markers from the registry and the golden files, the audit M9 greying with its
  banner, the "3D values from the workbook" label, the space claim. The results table: search, CSV and
  JSON export (`+inf`/`NaN` as text).
- Session: undo/redo (one step per settled edit), reset all, design files and share links in one
  schema-versioned JSON format (every input and the sizing state; `?m=` deflate + URL-safe base64),
  loading all or nothing. The sizing mode switch (Torque -> Magnets): `sizing::solve` debounced
  (0.25 s, never per frame), the solved or best value shown and its row locked, kept on leaving.
- Edits in progress: a drag, a held key (a held arrow key is one undo step) or a text field of an input
  row; the results search is no edit. A start-up share link is the session's start (no undo step);
  leaving Torque -> Magnets solves a change still waiting for its debounce first; the JSON export holds
  the design that produced the results; a host can keep the undo keys (M5). Old files' renamed paths
  migrate through `session::PATH_MIGRATIONS`; a refused file names every problem of the inputs and
  the sizing state.
- App: the linkage app's CAD dark theme, forced dark; native file dialogs and web downloads (rfd); the
  web entry opens `?m=` links. `gui-smoke` opens `/magcoupling/` through a pinned link and loads a
  design file through rfd's web picker (zero console errors, share link loaded, sizing solved, design
  file loaded).
- Decisions M41-1 to M41-15 as recommended in the plan
  (`docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md`). The engine,
  parity, differential data and registry are unchanged. Nothing pushed: the next push of `main` deploys
  `/magcoupling/` and needs the user's go.

## 2026-10-01 — Magcoupling A4 notes: independent physics review fixes
- `magcoupling-rs/src/engine/explain/notes.rs`: seven "imprecise" findings of an independent physics
  review applied to six notes (text only; no equation, record or engine change):
  - ferrite_cold_demag: the watch-out tells a ferrite design to enter its own reverse fields; the
    defaults (354 to 863 kA/m) come from the NdFeB design and put a Y30 ring's cold limit at 336 C.
  - calibration: the 2D model reads low (about 4 % without back iron, M4; less with steel, M5), and
    the assumed 0.95 adds conservatism rather than correcting that bias; M5 added to the sources.
  - harmonics: the third harmonic is negative even with no gaps (sin(3 pi/2) = -1) and the fill
    decides which harmonics are negative (the default outer ring, fill 0.62, has a positive third);
    higher harmonics are also smaller; 1, 3, 5 is the default and the calculator sums up to 11.
  - br_temperature: the permanent loss starts at the demagnetization onset; the limit is 10 C below it.
  - demagnetization: the watch-out's +/-20 % is on the Hcj slope (M13).
  - slip_heating: the 316L skin depth (about 34 mm at the default slip) and the 0.7 end factor.
- The six notes' Review records name the independent review (2026-10-01) and its commit; the other
  eleven keep ee11bfd. All 17 stay reviewed; the release gate is green.

## 2026-10-01 — Magcoupling Addendum A-3: final review fix wave
- Drift guard: 15 more edge points, one per comparison the review named (the hot minimum and
  hot-day torque checks, each ring's rating check, which limit governs, each of the verdict's four
  terms alone, both ends of the FEA range, the time to the limit's start-at-the-limit and never
  arms), each read or searched (`exact_input`) from the defaults and asserted on its edge.
  Mutation-checked: 11 of the 15 `<`/`<=` swaps passed the guard before, all 15 fail it now.
- Registry: `corrections_upstream` is precomputed at build and returns `&[DeviationId]` (37 us to
  46 ns per path in release); `upstream` no longer rebuilds the graph map per call; the module doc
  says which calls are lookups.
- Records: the per-ring Br coefficients read Calculator C35 (`model.alpha_br_per_C`) and the
  governing cold limit reads the cold ring's skipping cold onset plus the margin, so neither is a
  hover dead end; `metal.slip_freq_Hz` is `f_{e,MD}`. Values unchanged bit for bit.
- Docs: decision G4's exception in explain/mod.rs and the README; the notes' test attribution; the
  symbol letters the chains reuse (symbols.rs header); each note's sign-off record names commit
  ee11bfd (no note text changes; all 17 stay reviewed).
- Parity, the differential data and every registry probe unchanged.

## 2026-10-01 — Magcoupling Addendum A-3: explainability (A2, A3 traceability, A4)
- `magcoupling-rs`: the engine side of the equation explorer (`src/engine/explain/`): one formula
  markup per explained result that is both what the M4 panel typesets and what the drift guard
  evaluates (no Rust closures); the registry (`Registry::build`, `equation_for`, `used_by`, the
  dependency graph, term rows and A3 styles); 392 equations over decision 31's chains, the dashboard
  and the geometry callouts (decision G1: 169 scope paths, every one drilling down to inputs).
- The drift guard proves every record against the engine at the defaults and every differential
  case under 16 augmentations (harmonic sets, back irons, grade mode, axial override, ferrite from
  the inputs at two betas and unrated, the sleeve and cap picks, E17's free-space fields), every
  `cases` arm taken and every value term of every record varied.
- A3 traceability: soundness for every input, sensitivity for the 15 assumption inputs with five
  cancellations listed; every assumption reaches and moves an explained result.
- A4: 17 teaching notes drafted from the M1 derivations, physics-reviewed and signed off; the
  start-here order; `note_for` shows reviewed notes only; the release gate is green.
- Engine: Rust-only reads only (harmonics 7-11 parts, the amplitudes and E7 angles, each ring's E20
  block, the materials in effect), plus decision G4's outer-ring block computed with E20 off;
  parity, the differential data and every registry probe unchanged.

## 2026-09-30 — Magcoupling Addendum A-2: final review fix wave
- E18's corrected_formula no longer quotes the library's 6061 at 68.3 GPa (decision A8): since A2-6 it is
  E18's 68.9 GPa. Text only; no number, probe or correction changes.
- `tests/grades.rs` `mixed_rings_each_take_their_own_alpha_and_density_either_way_round`: a library NdFeB
  ring beside a Y30 grade-mode ring, each way round, pins decision A2-7's per-ring wiring (both rings'
  factors in C8 and C155, the governing ring's E20 coefficient and onsets, the magnets' and the bond
  block's mass); mutation-checked against swapped or shared coefficients and the inner density.
- Docs: `assumptions.rs` and `tests/assumptions.rs` name three documented assumption overrides (C22 for a
  grade-mode ring beside the A-1 two); `04-memory.yaml` tells plan A-3's traceability test to exempt
  them; the crate docs (`lib.rs`) name `engine::housing`.
- Deferred, recorded in `04-memory.yaml`: `model::shear_stress` still evaluates all six harmonics when
  fewer are summed (part of the A-2 plan's measured 8.5 -> 17 us compute_all doubling, about 50x inside
  the spec's budget).

## 2026-09-30 — Magcoupling Addendum A-2: parameters and sizing (A3, A1)
- `magcoupling-rs`: one general E7 peak search for any odd harmonic set (decision 29: every root of
  dT/dx in cos^2 x by recursion on derivatives) and the harmonic set as the Rust-only assumption
  `coupling.max_harmonic` (1 to 11, default 5), summed by the Calculator, both circuit sums, the
  sweeps and the Calibration prototype (Rust-only tau7_Pa to tau11_Pa).
- The assumptions registry (`assumptions.rs`: the spec's 14 rows over 15 inputs, rationale and
  source, `modified`, `reset_to_workbook_defaults`); the audit M9 flag `end_effect_check`; the axial
  length override `coupling.magnets.axial_length_mm`; inverse sizing (`sizing.rs`: axial length,
  magnets per ring, ring radius; a 64-cell scan refined at the first crossing, each peak and each
  validity edge; counts a value only if its blocks fit (flats, or arcs that do not overlap), the
  keyway leaves hub wall and f_end > 0); the autofit wall suggestion
  `materials.cup_wall_suggested_mm` and the space claim (`housing.rs`, `DesignResults::housing`,
  overshoot per axis, at least 0.01 mm), with the hub length, cup cavity depth and retainer span
  following the length override (decision A2-8, option B: `housing::axial_housing`).
- Carry-overs: the library's 6061 takes E18's 68.9 GPa (A2-6); C45 keeps its NdFeB slider, a positive
  beta is typed (A2-5); a grade-mode ring takes its grade's alpha(Br) and density (A2-7).
- Parity (1,149 checks), the differential data and every registry probe unchanged; the default
  headline with every correction on is unchanged (2.688 N·m, 93.06 °C, M4 x 14).

## 2026-09-30 — Magcoupling M4 infrastructure (branch magcoupling/m4-infra)
- `magcoupling-rs` features: `gui` (egui 0.32) adds `gui::MagcouplingPanel`, a panel any egui app
  can host (`fn ui(&mut self, &mut egui::Ui)`). The tracer content: metadata-driven sliders for face
  gap, pole count and axial length, "Reset all", and the headline recomputed with `compute_all`
  every frame. `app` (eframe 0.32) adds `app::MagcouplingApp` and the binaries `magcoupling-app`
  (native) and `magcoupling-web` (wasm32, eframe WebRunner + wasm-bindgen). The engine stays pure
  std; every new dependency is optional.
- Versions: `magcoupling-rs/Cargo.lock` pins `linkage-sim-rs/Cargo.lock`'s egui and eframe 0.32.3
  and wasm-bindgen 0.2.114, the `wasm-bindgen-cli` version that `deploy-web.yml` installs. The
  engine's existing lock entries are unchanged.
- Second web bundle: `linkage-sim-rs/scripts/build_magcoupling_web.sh` builds into
  `linkage-sim-rs/web/magcoupling/` (index.html committed; glue and wasm gitignored).
  `build_web.sh` calls it, and `serve_web.sh [PORT]` serves it at `/magcoupling/`. Proven with
  Playwright: zero console errors, live slider click, `/magcoupling` without a slash.
  `deploy-web.yml` (committed, not pushed) runs the script and caches `magcoupling-rs/target/`, with
  a key over both lock files. `vercel.json` revalidates `/magcoupling/magcoupling-web.js`.
- workbook-parity guard: `linkage-sim-rs/scripts/magcoupling_shipped.sh` holds the shipped cargo
  arguments once and `magcoupling_assert_shipped` (cargo `--message-format=json` features per
  compiled unit). The web build pipes through it, and gate 10 runs it on the native and wasm32
  builds plus a negative control. A `compile_error!` cannot work: the self dev-dependency unifies
  the feature into the bins of `cargo test --features app` (demonstrated).
- `gate.sh` now has 12 gates: 7 `cargo test --features app`, 8 clippy `--features app` with
  `-D warnings`, 9 wasm32 clippy (gui lib, web bin), 10 the guard, 11 lock parity; the Python
  oracle moved to 12.
- Headless egui tests (`src/gui/panel.rs`, `format.rs`, `src/app.rs`): a slider change (arrow key,
  rail click) updates the headline in the same frame; reset restores the defaults; the pole count
  stays even; range ends stop; idle frames write nothing; the panel draws inside an egui::Window.
- Open for the M4 plans (`04-memory.yaml`): the axial-length input mapping, step-snapping float
  noise, the clamping policy for typed values, gui-smoke on `/magcoupling/`, theme, bundle size.

## 2026-09-30 — Magcoupling Addendum A-1: final review fix wave
- `docs/ai/03-structure.yaml` parses again (two quoted scalars) and lists `tests/material_links.rs`;
  the invariant this branch added to `02-system.yaml` is quoted (that file's older parse failures,
  from line 13 on, predate the branch and are untouched).
- `magcoupling-rs/README.md`: status no longer puts M3 before M4, the registry row names E1-E20, and
  the deviations and robustness test rows list the Addendum A tests. The E5 help on C121 and the E5
  row now say the workbook's 1.035e-5 includes the steel hub's image (free space is E17's 6.837e-6).
- `material_library::resolve`: the cap default reuses the body's workbook 6061 values (no number changes).
- `04-memory.yaml` records that E20's rating calibration gives negative offsets (raised onsets) for
  N52, N50, N50M, N38UH, N35EH and Recoma 26; a clamp at offset >= 0 is a candidate correction.

## 2026-09-30 — Magcoupling Addendum A-1: data and physics (grades, materials, E15-E20)
- `magcoupling-rs`: the A6 grade table (`grades.rs`, 17 grades cited per value) and the parts' vendor
  data (coating, magnetization, vendor page); any grade with manual dimensions (Rust-only
  `coupling.magnets.grade_inner`/`grade_outer`); the A5 materials library (`material_library.rs`, 14
  materials) with Rust-only per-part selectors `materials.parts.*` whose default is the workbook
  material (the inputs), physics links (circuit, design flux density, conductivity, density, specific
  heat, expansion and modulus; the C6 override) and six warnings (`warnings.rs`, `DesignResults::warnings`).
- Corrections E15 (heat capacity), E16 (removed web disc), E17 (aluminium eddy losses, T1 with three
  Rust-only free-space fields), E18 (aluminium hub mismatch screen), E19 (SuperMagnetMan arcs per the
  vendor grid) and E20 (each ring's own coercivity, Br and rating, the weaker ring governing; ferrite limited
  on the cold side), each with the
  report's changed-cell tables or probes as expected values. Registry: `Approval`, `depends_on`,
  `Deviations::with/without` (decision 15).
- Rust-only inputs (`param_rust_only`): no cell, skipped by the metadata tests, left out by
  gen_differential.py. Parity (1,149 checks) and the differential data are unchanged; the default
  headline with every correction on is unchanged (2.688 N·m, 93.06 °C, M4 x 14).

## 2026-09-30 — Magcoupling Addendum A verification (data tables, E15-E18, defaults)
- `docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md` and
  `...-addendum-a-data.json`: the A6 grade table (17 grades, K&J-minimum
  NdFeB basis, Arnold Recoma SmCo, ferrite Y30, bonded NdFeB), the 15-part
  mapping and the A5 materials table (14 entries), every value cited and
  re-checked by an independent skeptic; E15-E17 (plus candidate E18)
  computed audit-style at back iron = 0; a defaults-exactness check of
  grades, materials, housing autofit and the generalized harmonic search;
  the A2 scope count (418 result fields; v1 covers about 159).
- The user approved option A on all 31 decisions (2026-09-30). The spec's
  Addendum A text gains N50/N50M, the 20 mm bay in the envelope, and
  "design flux density feeds the wall check".
- Next: the Addendum A engine plans (data and physics; parameters and
  sizing; explainability), then M4.

## 2026-09-29 — Magcoupling M2: final review fix wave
- E7 (`model::peak_off_half_pitch`): the quadratic in cos² x now takes its
  roots without cancellation (q = −(qb + sign(qb)·√disc)/2, roots q/qa and
  qc/q). The textbook form lost the small root when the fifth harmonic
  vanished (|a5/a1| up to about 1.5e-13, which a ring at a fill of exactly 0.4
  or 0.8 reaches), so E7 kept half a pitch, a local minimum: at 6 poles, no
  back iron and a 0.4-pitch manual inner block the harmonic sum was 13,121 Pa
  against a true peak of 32,505 Pa (pull-out 0.162 against 0.40 N·m). Default
  cells stay bit for bit (same `None`/`Some` decisions outside the band).
  Tests: `peak_off_half_pitch_finds_the_brute_force_maximum` (3,000 random
  amplitude triples against a grid, a third of them in the band),
  `a_vanishing_fifth_harmonic_does_not_lose_the_peak`, and
  `e7_finds_the_peak_at_a_fill_of_exactly_0_4` (Calculator and Calibration).
- DRY: the E7 gate is one function, `model::peak_angle` (with `tau_at`, the
  a·sin(n x) term), used by `at_pull_out` (pull-out and sweep rows), the two
  circuit sums C95 and C96, and Calibration C40-C42. The corner radius
  √(r_face² + (w/2)²) and the Br temperature factor 1 + α (T − 20 °C) are
  `model::corner_radius` and `model::br_factor`, used by the Calculator, the
  sweeps, Calibration, Metal design and Temperature design. Bit for bit: every
  result of 1,504 seeded input sets hashed identically under NONE, ALL and
  only(E7) before and after (scratch harness, not committed).
- `headline` reads its 15 paths with the new `ResultSet::get(path)` (results!
  and rows! generate it, table rows as `table[i].field`) instead of building
  all 996 result rows: 202 µs to 0.44 µs per call in release (381 to 2.6 µs in
  debug), values bit for bit. Tests: `result_get_reads_fields_groups_and_table_rows`,
  `result_get_agrees_with_every_result_row` (meta),
  `get_and_headline_read_what_result_rows_lists` (every real result path, two
  designs); the per-frame smoke test now times `compute_all` plus `headline`.
- Never-panic (D3) pinned for non-finite values that bypass `set()`:
  `compute_all_never_panics_on_non_finite_struct_literals` writes NaN, +inf
  and −inf into `metal.face_gap_mm`, `clamps.boss_od_mm`,
  `temperature.thermal.conductance_W_K` and `metal.measured_drag_Nm`, and a
  non-finite drag with `npole = i64::MAX`, under NONE and ALL; `validate()`
  must name exactly that path as `NotFinite`.
- Cleanups (behaviour-neutral): `check_value` and `f64::from_value` share one
  non-finite check (`meta::finite`); the no-op `#[allow(non_snake_case)]` on
  `api::compute` is gone (clippy -D warnings stays clean without it); the
  `retainers` `too_many_arguments` allow stays, since E8 made it 8 parameters,
  with its comment saying so; `gen_differential.py` uses the clamps and sweeps
  names it already imports (no `py_clamps`, `py_sweeps`), and the sweep probe
  tag reads "low requirement: no row below the hot minimum" (its rows are
  nominal or outside the OD envelope); only that tag changed in
  `gap_sweep.json` and `pole_sweep.json`.
- Docs: the crate README says which tests switch the corrections off (all of
  them go through the test-only feature, not only parity and differential)
  and lists every `tests/data/` file, `static_data.json` and `deviations/`
  included. Not changed on purpose, recorded as open items in
  `docs/ai/04-memory.yaml`: E4's keyed-wall term takes the bondline input
  while the block-fit term keeps the workbook's 0.05 literal (approved
  formula; Addendum A3), a negative end factor f_end inside the slider ranges
  (audit M9; M4/A3 guard or GUI flag), and the `workbook-parity` feature
  guard (a `compile_error!` with `app` would break `cargo test --features
  app` through the self dev-dependency; M4 decides).

## 2026-09-29 — Magcoupling M2: engine port complete
- Every Python module except `fields3d` (M3) is ported to `magcoupling-rs/`,
  one Rust module per Python module, workbook-exact with the corrections off:
  `calibration` (23 result cells), `library`, `model` (73; `mass_estimate` 6),
  `metal_design` (retainers 9, sheet 49), `materials` (7), `temperature` (130),
  `clamps` (33 result cells, 25 input cells, the 165-cell screw table),
  `sweeps` (Gap sweep 338, Pole sweep 156 table cells) and the API completion
  (`headline`, `DesignInputs::validate`). Workbook parity covers all 1,149
  checks: 330 result cells, 494 sweep cells, 165 screw-table cells and 160
  default inputs (`the_port_checks_every_cell_test_parity_checks` pins the
  totals to `test_parity.py`).
- Harness additions: modules that read several input groups (`MODULES` in
  `gen_differential.py`), columnar differential data (one header, one line per
  case), table rows in the metadata model (`rows!`, `TableLayout`, synthesized
  cells), the full run (`differential/full.json` varies all 160 inputs at
  once, plus one case for each pair of selector choices), the `BRANCHES` table
  (every branch of every text result is reached, and no text result lacks a
  row), and `tests/robustness.rs` (selector codes outside their choices,
  extreme inputs, zero measured drag; `compute_all` never panics).
- Corrections E1 to E14 are all applied, one commit each, through the
  deviation registry: `compute_all` always applies them, and the workbook form
  stays reachable only through the test-only `workbook-parity` feature.
  Headline changes at the default design: pull-out at operating temperature
  2.647 to 2.688 N·m and governing temperature limit 92.55 to 93.06 °C (E3),
  clamp screw M4 x 12 to M4 x 14 (E2; it protrudes 0.34 mm, and the Rust-only
  `clamps.length_note` says so), and Temperature design C106 and C202 change
  from "Above ..." to "Below the lap-shear strength" and "Below the fatigue
  endurance" (E1). E3, E4 and E5 (246, 47 and 37 changed cells) are reviewed
  golden files, `tests/data/deviations/E<k>.json`; E7 to E13 leave every
  default cell bit for bit and carry registry probes on off-default inputs;
  E14 rewords a help text and one README sentence, no numbers.
- Decisions D1 to D7, the recommended option each: D1 N42SH Br = 1.30 T; D2 E1
  as the constant C96 = 0.107 GPa (per-adhesive modulus with Addendum A5); D3
  a selector code outside its choices never panics (NaN or "#N/A"), plus
  `validate()` at input boundaries; D4 corrections that change more than 15
  cells at defaults use golden files; D5 `clippy -D warnings` for
  `magcoupling-rs` only; D6 E2's warning is the Rust-only result
  `clamps.length_note`; D7 the harmonic set stays the workbook's 1, 3, 5 until
  the Addendum A engine plan.
- Equality-edge unit tests: one per module with threshold comparisons (`model`,
  `metal_design`, `materials`, `temperature`, `clamps`, `sweeps`), each
  comparison at exact equality with the equality asserted first; step 8 of
  "Porting a module" in the crate README.
- Open items (docs/ai/04-memory.yaml): merge main into `magcoupling/m2` before
  the Addendum A engine plan (Addendum A is on main, 7a9d2e8); M3 must apply E3
  and E5 inside `fields3d`; the Addendum A engine plan makes the harmonic set a
  parameter.

## 2026-09-29 — Magcoupling M2: engine port tracer bullet (calibration)
- New crate `magcoupling-rs/` beside `linkage-sim-rs/` (not a workspace):
  library `magcoupling`, pure std engine, wasm32-clean; features `gui`/`app`
  declared empty for M4, `workbook-parity` test-only via a self
  dev-dependency. Branch `magcoupling/m2` (from `magcoupling/m1`).
- Metadata model (`src/engine/meta.rs`): `inputs!`/`results!` declare each
  field once with its default and a const builder in Python's `param()`/
  `out()` argument order, plus slider range and the Addendum A3
  `assumption` flag; field names keep the Python spelling so dotted paths
  equal the Python `input_schema()` paths.
- `compat.rs`: Python/Excel semantics (py_min/py_max, CEILING/FLOOR with the
  1e-12 guard and no -0.0, TEXT(x,"0"), float repr with half-even ties,
  _fmt_num, f-string rounding, parity rule).
- Deviation registry: E1-E14 from the approved M1 report, all `Planned`;
  `Deviations::ALL` for users, `NONE`/`only(id)` for tests.
- Calibration sheet ported; parity (23 result cells, 16 default inputs),
  differential vs Python (300 seeded cases, every range end, both selector
  choices, inclusive span ends), helpers corpus (~1,250 values), metadata
  parity with the Python schema, schema and registry checks.
- Generator `reference/magcoupling-py/tools/gen_differential.py` reads the
  Rust slider ranges from `tests/data/input_schema.json`; `--check` guards
  staleness. The corpus found two real Python/Rust differences (repr ties,
  ceil of (-1, 0]).
- `linkage-sim-rs/scripts/gate.sh` now has 7 gates: + magcoupling-rs test,
  clippy -D warnings, wasm32 check, and the vendored Python parity suite +
  data freshness (SKIP line when no oracle venv is found).
- Guide: `magcoupling-rs/README.md` (porting pattern, translation rules).

## 2026-09-29 — Payload weights Task 10: hands-on checklist and plot-tab counts
- Payload spec: appended "Hands-on checklist (robot-lift model)", one pass
  over the finished feature. Section 0 is the gate, the WASM build/serve
  scripts and the `gui-smoke` workflow. Sections 1-6 cover placement (toolbar
  mass field, hint, W<n> names), the weight editor and Add weight, Esc/Enter
  commits, select/drag/reattach/cancel/delete, the canvas readout (arrows,
  tooltip, selected readout, label words, gold stale arrow), braking bands,
  the Weight Breakdown views, the physical-intuition checks, stored-force
  invariance, mounting angle, Nathan Mode, save/reopen/share and a
  no-actuator sample. It folds in the checklist patches from the old Task
  7/8 drafts. It drops the pre-BL-026 steps: no "set F to 0 first", and the
  Actuator Force plot no longer shows F_required - F_stored.
- A caution at the top of the checklist and a new 02-system
  known_limitations entry: on a loaded sample, driver reassignment rebuilds
  the stock sample and drops every weight. It is undoable, and a model
  reopened from a file keeps its weights.
- A throwaway probe on `test_support::swept_lift` (not committed) checked
  every number and claim in the checklist against the code. Labels:
  "7.9 kN push, motoring" at 45 deg, "1.3 kN push, braking" at 135 and
  "0.00 N" at 90. W1's share is +5.5 kN at 45 and halves at the rocker
  midpoint. Braking runs 91-269 deg, Other loads stays below 4e-7 N, Total
  equals the plotted force, and going from stored force 50 N to 0 changes
  nothing (3e-11). A 30 deg mounting angle moves gray to 60/240 deg.
  Reassign-then-undo restores the weights.
- The probe corrected two draft claims. (1) Force shares are blank only at
  56-57 deg. The 236.3 deg reversal, where the actuator is at its shortest,
  has no sample inside the 1 % band, so W1's share jumps from about +45 kN
  to -20 kN with no gap (new 02-system known_limitations entry). (2) The
  change-point glitch on this sample is at 360 deg on the plots (a
  one-sample braking band and a pull force), not a band break at 180.
- README and FEATURES: 15 plot tabs (was "10"). The list now includes
  actuator force/speed/power, weight breakdown and output force.
- 04-memory: active issue (the checklist runs before merge) and an open
  question: the gold stale-weight arrow is in the code and the checklist
  but not in the spec's canvas-readout section, which names only
  green/red/gray. The stroke-mode driver-share scope and the small wording
  gaps (weight_breakdown, "push, braking", the "-" for the reversal) were
  already settled by the spec's 2026-09-29 amendments, so they are not
  open questions.

## 2026-09-29 — Payload weights Task 9: canvas readout
- `AppState::current_sweep_index` (the sample the plot cursor marks;
  `actuator_label_force` now uses it) and `AppState::actuator_label_power`
  (required power + braking tolerance from `WeightBreakdown`).
- Actuator label (`force_render.rs actuator_label_text`,
  `format_actuator_label`): "1.2 kN push, braking" - push/pull by sign
  (positive = extension), motoring/braking by the braking-band rule;
  "(stored)" on the stored-force fallback; "(computed)" is gone. Since
  BL-026 both words derive from REQUIRED quantities: the force is
  `sweep.actuator_forces` (the plotted required force in both modes) and
  the power is `WeightBreakdown::total_power`. The drafts' BL-026 caveat
  (push/pull from F_required - F_stored) is gone, and `swept_lift` no
  longer zeroes the stored force to dodge it.
- `canvas/rendering/weights.rs`: `draw_weights` replaces the point-mass
  block of `render_mechanism`: gold marker plus a gravity-direction arrow
  (`gravity_screen_dir`, 12-40 px by mass, `WEIGHT_ARROW_*` in colors.rs)
  coloured by `classification_color` of the weight's classification at the
  current sample (`weight_at_pose`; gold until the sweep has the weight at
  its current mass and position). The permanent "2.00 kg" labels are gone:
  hovering a weight shows a tooltip (`show_weight_tooltip`, weights before
  joints) and the selected weight a card with the title, mass and force
  share ("-" near stroke reversal). `interaction::weights_interactive` is
  `pub(super)`.
- Arrowheads (DRY): `primitives::draw_arrowhead(tip, dir, head_len,
  stroke)` is the one head of every straight canvas arrow. `draw_arrow`
  (shaft plus a head clamped to the arrow length, used by the weight
  arrows) calls it, and so do `draw_force_arrow`,
  `draw_external_force_arrow`, the Fx/Fy component arrows, the actuator
  line's midpoint head and the force-zone arrows (6 px head), each with
  its old head length, so their geometry is unchanged. The torque arc's
  head (0.5 rad, 5 px) has a different angle and keeps its own code.
- Test helpers (DRY): `test_support` gains `drawn_line_colors`,
  `swept_lift`, `sample_at` and `pose_at`, and reuses
  `set_actuator_stored_force` (no second copy). The canvas
  `weight_readout` tests reuse `weight_clicks::frame` (now returns the
  frame output) and `hit_testing::tests::weight_screen`, and the
  duplicate `weight_screen` in `weight_clicks` is gone.
- Tests: `force_render::tests` (every force/power sign combination, unit
  edges, non-finite force,
  `actuator_label_words_follow_the_plotted_required_force_and_power_in_every_mode`
  at every sample with stored force 0, the sample's 50 N and one above
  every required force, stored fallback "50 N push (stored)"),
  `rendering::weights::tests` (arrow
  length, gravity direction under mounting angles, weight_at_pose vs the
  breakdown incl. stale entries, readout lines incl. reversal dash and
  driver-torque shares, share format), `gui::state::tests`
  (current_sweep_index, actuator_label_power), `gui::canvas::tests::
  weight_readout` (arrow colours at 45/135/90 deg, gold for an unswept
  weight, no arrows without gravity, no permanent labels, hover tooltip,
  selected card, actuator label words on the canvas).
- Mutation check: a label that reads F_required - F_stored fails the
  every-mode test at the sample's own 50 N ("50 N pull" at 90 deg).
- Docs: FEATURES (actuator label words, weight arrows), ENGINEERING_OUTPUTS
  (canvas readout), 02-system `canvas_readouts_read_the_current_sweep_sample`
  (BL-026 text rewritten), 03-structure (primitives, force_render,
  rendering/weights, test_support). The spec checklist lines for the
  readout land with the checklist in Task 10.

## 2026-09-29 — Payload weights Task 8: placement and the weight editor
- `+ Mass` toolbar: a mass field (`draw_place_mass_field`, shown while the
  tool is active) edits `AppState::last_point_mass_kg`, so it starts at the
  last mass used. The placement click snaps to the grid (`snapped_world`,
  shared with the weight drag), previews the snapped landing point, selects
  the new weight and consumes the click (`handle_place_mass` returns true;
  no click selection that frame). The hint names the next id and mass
  (`canvas/rendering/weights.rs place_mass_hint`, "Click to place weight
  W3 (2.5 kg) on 'coupler'"). A failed placement shows a status message.
- `property_panel/weight_editor.rs` (new): `draw_selected_weight` (Weight
  <name> header, Link: <owning link>, name, mass, body-local X/Y, Move to
  Link / Reposition / Delete) above the Link Editor; `draw_link_weights`
  replaces the Point Masses block: "Weights (n)", always shown for a moving
  link, one editor per weight, Add weight (last mass at the link's base
  CG, selected). Fields commit once per edit via `committed_number` (see
  02-system `weight_fields_commit_once_per_edit`).
- `PendingPropertyEdit::{SetPointMassLabel, AddPointMass}`; `RemovePointMass`
  clears a selection of the removed weight.
- `gui/state/display_units.rs`: `format_decimal` (6 decimals, trimmed),
  `format_mass_kg`. `analysis::gravity_breakdown`: `display_name` is pub,
  new `name_with_id` (also used by the Weight Breakdown legend) and
  `point_mass_title` ("W1" / "Robot torso (W1)").
  `canvas::WEIGHT_COLOR` is re-exported for the property panel.
- Tests: `gui::property_panel::weight_editor::tests` (typed commit on Enter
  and on Tab, clamping, no commit when leaving untouched or on Esc, drag
  commit on release, idle frames; panel shows name/mass/link/position,
  name commit, Esc drops a name, Tab to mass, stale selection, empty
  Weights section, Add weight click), `pending_edits::tests`,
  `gui::tests` (toolbar field visibility, starts at last mass, typing, idle
  frames), `gui::canvas::tests::weight_clicks` (snapped placement selects
  the weight, unsnapped placement, snap off the pointer still selects,
  hint), `display_units::tests`, `gravity_breakdown` name helpers.
- `gui/test_support.rs` (shared test helpers, DRY): adds `typed`,
  `central_panel_frame` (one headless frame of a closure in a central
  panel), `drawn_texts`, `drew_text` and `text_rect`. The new tests use
  them instead of local copies, and the property panel's `one_idle_frame`
  (Task 2) and the plot panel's `plot_panel_frame` (Task 7) now call
  `central_panel_frame` too. The toolbar field tests use
  `test_support::key_press`.
- Mutation checks: dropping `!placed_weight` from the click-selection gate
  fails `placing_a_weight_selects_it_even_when_the_snap_moves_it_off_the_pointer`
  and the snapped-placement test; dropping `edited != shown` from
  `committed_number` fails `leaving_a_field_without_typing_commits_nothing`.
- Docs: FEATURES (Place Mass tool, Weight editor), ARCHITECTURE (property
  panel weight editing), 02-system `weight_fields_commit_once_per_edit`,
  `place_mass_click_snaps_and_selects` and the DragValue release-frame
  lesson, 03-structure (weight_editor, rendering/weights, name helpers,
  test_support). The spec's hands-on checklist wording for the Weights
  section lands with the checklist in Task 10.

## 2026-09-29 — Payload weights Task 7: braking bands and the Weight Breakdown plot
- `gui/plot_panel/weights.rs` (new): `PlotTab::WeightBreakdown` ("Weight
  Breakdown" tab, enabled when `SweepData::weight_breakdown` is `Some`): one
  line per weight source (legend `"coupler (link)"`, `"Robot (W1)"`) plus
  "Other loads" and "Total", force share or power share
  (`AppState::weight_breakdown_show_power`, not persisted). Each weight's
  line is split into runs coloured per sample by
  `WeightBreakdown::classifications` (new: band computed once per weight)
  through `canvas::classification_color` (new, `canvas/colors.rs`: green
  helping / red hurting / gray neutral, distinct in Nathan Mode grayscale),
  the palette the canvas weight arrows will share. Force shares leave gaps
  near stroke reversal and at failed samples.
- Braking bands: `braking_bands(xs, braking)` (runs of braking samples,
  midpoint to midpoint, clipped at the sweep ends, zero-width dropped) and
  `band_y_extent` (data y range, never `plot_bounds()`), drawn by
  `draw_braking_bands` as "Braking" polygons behind the Actuator Force and
  Actuator Power curves. Source: `WeightBreakdown::braking` only, i.e. the
  required power. Since BL-026 the plotted actuator force and power are the
  required values in stored-force mode too, so the bands and the Total line
  follow the plotted curves in both modes; no BL-026 caveat remains.
- `plot_panel::sweep_x_to_display` (new, DRY): the one sweep-x to display-x
  conversion (m -> mm in stroke mode, degrees -> display angle unit). It
  replaces the inline copies at 9 sites: `compute_default_x_bounds`,
  `draw_toggle_markers`, `draw_range_boundary_markers`,
  `draw_angle_series_with_range`, the actuator force / power "With Inertia"
  overlays and the safety-factor overlay, the driver-torque statics
  overlay, and `draw_braking_bands`. `current_driver_display` is not a
  sweep x value (radians plus the display offset) and keeps its own branch.
- Sign wording: the Actuator Force tab tip and the LinearActuator force
  field tip now say positive = extension (push), negative = retraction
  (pull) (were "positive = tension"); the rated-force lines are "Rated
  (push)" / "Rated (pull)" (was "Rated (compression)" for the pull side).
  Both tips are constants with tests.
- Tests: `gui::plot_panel::weights::tests` (band edges, ends, NaN x,
  descending x, length mismatch, y extent, legend names, run splitting in
  both views, y labels; bands = negative plotted actuator power on the
  parallelogram in stored-force AND sizing mode, which uses
  `test_support::set_actuator_stored_force` instead of an inline loop; the
  Total line equals the plotted actuator force and power in stored-force
  mode), `gui::plot_panel::tests` (tooltip, `sweep_x_to_display`, headless
  render of the new tab and the banded tabs incl. driver basis and no
  breakdown), `gui::canvas::colors::tests`,
  `gui::property_panel::force_editor::tests`,
  `gui::sweep::weights::tests::classifications_match_classification_at_every_sample`.
  Mutation check: dropping `stored_force * rate` from `required_totals`'
  total power fails the stored-force case of the band test (91 deg) and
  the Total line test.
- Docs: 02-system `weight_breakdown_sum_and_sign_conventions`, egui_plot
  auto-bounds lesson, 15 plot tabs; 03-structure plot_panel/weights.rs,
  `sweep_x_to_display`, colors palette, `classifications()`;
  `docs/architecture/ENGINEERING_OUTPUTS.md` per-weight breakdown and
  braking section; FEATURES/README actuator sizing bullets. The stale
  "10 plot tabs" lines in README/FEATURES and the hands-on checklist are
  Task 10.

## 2026-09-29 — Payload weights Task 6: drag and drop weights
- `gui/state/types.rs`: `WeightDrag { body_id, weight_id, current_world }`;
  `AppState::weight_drag: Option<WeightDrag>` (preview only).
- `gui/canvas/interaction.rs`: `handle_weight_drag` (runs before the
  ground-pivot / force-zone drags, which it pre-empts, and suppresses pan):
  press on a weight + drag selects it and previews (pointer snapped to the
  grid when snapping is on; target link highlighted; dashed line; grabbing
  cursor); release = one `move_point_mass` (one undo step), selection
  follows the weight. `weight_drop_target`: nearest link within
  `LINK_PICK_RADIUS` that the blueprint has and is not ground (compound
  actuator bodies are skipped), else its own link. Esc or a release outside
  the canvas cancels. Rendering fades the dragged weight and shows a hint.
- `gui/canvas/hit_testing.rs`: `find_nearest_body_segment_where` (filtered
  search; `find_nearest_body_segment` delegates). `colors.rs`:
  `LINK_PICK_RADIUS` (60 px, now also used by Place Mass and Move to Link),
  `WEIGHT_COLOR`.
- `gui/mod.rs`: the Delete/Backspace block becomes
  `handle_delete_shortcut`, removes a selected (or multi-selected) weight,
  and is ignored while a widget has keyboard focus (Backspace in a text
  field used to delete the selected body or joint).
- Tests: `gui::canvas::tests::weight_clicks` (move on release with snap and
  one undo step + undo, unsnapped drop, preview leaves the blueprint alone,
  reattach at the drop point, nearest-own-link stays, mounting angle, Esc,
  release outside the canvas, empty-canvas drag still pans, no drag outside
  Select mode), `gui::canvas::interaction::tests` (drop target rules),
  `gui::canvas::hit_testing::tests::find_nearest_body_segment_where_*`,
  `gui::tests` (Delete/Backspace removes selected and multi-selected
  weights, stale selection, other keys, Backspace while typing).
- Drop rule: the nearest link within the pick radius wins, its own link
  included, so a drop nearer its own link than a neighbour's stays put. The
  spec's Track 2 section 3 already said so (the accepted-deviation amendment);
  it now also states where the weight lands and that ground and compound
  actuator bodies never take a weight.
- Cleanups: the Move to Link / Reposition cursor preview uses `WEIGHT_COLOR`
  instead of a literal gold (and follows Nathan mode like the other weight
  previews). `find_nearest_body_segment_where` is clippy-clean (let-else,
  one condition, `is_none_or`), which also clears the three warnings the old
  `find_nearest_body_segment` carried.
- `gui/test_support.rs` gains `primary_button` and `key_press`; the canvas
  drag tests and the `gui::tests` delete tests use them instead of local
  copies. The canvas interaction tests reuse the `hit_testing` tests'
  `segment(body, y)` bar fixture.
- Mutation checks done: dropping the `wants_keyboard_input` guard fails the
  Backspace-while-typing test; accepting every body as a drop target fails
  `a_weight_drop_skips_links_that_cannot_carry_a_weight`.

## 2026-09-29 — Payload weights Task 5: weight hit testing and selection
- `gui/canvas/hit_testing.rs`: `point_mass_screen_pos` (weight marker
  position at the current pose/view; `None` when not finite) and
  `find_point_mass_at(state, screen_pos, radius_px)` (nearest weight within
  the radius; ground and blank-id weights skipped; ties go to sorted body id,
  then list order).
- `gui/state/types.rs`: `SelectedEntity::Weight { body_id, weight_id }`.
- `gui/state/entity_crud.rs`: `AppState::body_local_to_world`, the inverse
  of `world_to_body_local`; Move to Link now uses it instead of an inlined
  transform.
- `gui/canvas/interaction.rs`: Select-mode clicks test weights first
  (`WEIGHT_HIT_RADIUS` 8 px), then joints, then pins; Shift+click toggles
  weights in the multi-selection. Hovering a weight draws a ring and sets
  the grab cursor (`weights_interactive`: plain Select mode, no other pick
  armed). `gui/canvas/rendering/mod.rs` draws weights through
  `point_mass_screen_pos` and rings selected weights.
- Tests: `gui::canvas::hit_testing::tests` (hit, miss, nearest, tie order,
  skipped weights, no blueprint), `gui::canvas::tests::weight_clicks`
  (click selects, empty click clears, weight beats joint on a pin,
  Shift+click toggle, grab cursor only in Select mode) and
  `body_local_to_world_*` in `gui/state/tests.rs`.
- `gui/test_support.rs` gains `primary_button_with(pos, pressed, modifiers)`;
  the canvas click tests build their pointer events with it instead of a
  local helper.
- Mutation check done: letting the joint loop run after a weight hit fails
  `a_weight_on_a_pin_wins_the_click_over_the_joint`.

## 2026-09-29 — Payload weights Task 4: per-weight breakdown in the sweep
- `gui/sweep/weights.rs` (new): `WeightBreakdown` (per source `gravity_power`,
  `force_share` = `-P_g/rate`, `power_share` = `-P_g`; `other_force`,
  `other_power` = total - sum; `total_force`, `total_power`; `braking`;
  `classification(source, sample)`), `ShareBasis::{ActuatorForce,
  DriverTorque}`, `WeightBreakdownBuilder` and `required_totals`. Re-exported
  as `gui::sweep::{WeightBreakdown, ShareBasis}`.
- `gui/sweep/mod.rs`: `SweepData::weight_breakdown: Option<WeightBreakdown>`;
  `compute_sweep_data_with_weights(.., weight_sources)` feeds one row per
  sample (NaN rows through `push_nan_row` and the velocity-failure branch;
  NaN totals where statics failed instead of the 0 torque the plot shows);
  `compute_sweep_data` keeps its signature and computes no breakdown.
  `AppState::compute_sweep` passes `weight_sources(blueprint)`.
- Totals are the REQUIRED actuator force/power. BL-026 (on main) already
  makes `actuator_forces` the required force in stored-force mode, so
  `required_totals` takes it as is. The draft's stored-force add-back is
  gone. The total power `T*omega + F_stored*dL/dt` stays finite through
  stroke reversal (02-system.yaml `weight_breakdown_total_is_required_force`).
  No actuator: driver-torque shares (N for a linear driver in stroke mode).
  Trajectory mode: `None`.
- `gui/test_support.rs` gains `set_actuator_stored_force` (shared
  stored-force setter; later tasks reuse it instead of copying the loop).
- Tests: `gui::sweep::weights::tests` cover Parallelogram and Chebyshev
  sizing with two weights (sums, totals = plotted actuator force/power,
  remainder ~ 0). `stored_force_mode_and_sizing_mode_give_identical_breakdowns`
  checks totals, shares, remainders, braking and classification on both
  samples. Also covered: known-pose and trace-derived helping/hurting;
  braking follows net gravity power; near-reversal NaN; forced solver
  failures with actuator and driver basis; FourBar driver-torque basis;
  mounting angle; builder and `required_totals` hand values.
  `sweep_gravity_power_matches_energy_change_between_adjacent_samples` now
  checks every source on both actuator samples. Positions are rebuilt from
  the traces plus body angles; the measured central-difference error is
  O(h^2): 5e-5 on the Parallelogram, 1.2e-3 on the Chebyshev.
  `breakdown_sum_invariant_holds_after_a_mass_edit_on_a_weighted_link_without_rebuild`
  guards BL-024. Also `gui::sweep::tests::stroke_mode_weight_breakdown_splits_the_linear_driver_force`.
- Mutation checks done. Restoring the add-back fails the stored/sizing test
  and the `required_totals` test. Dropping `apply_point_masses` from
  `sync_live_mass_props` fails the mass-edit test (F_other = -78 N).

## 2026-09-29 — Payload weights Task 3: gravity breakdown physics module
- New pure module `analysis/gravity_breakdown.rs` (spec Track 2, section 2):
  `weight_sources(&MechanismJson)` (link self-weights `link:<body>` at base
  mass/CG for positive finite base mass, then every point mass the loader
  applies; bodies sorted by id), `gravity_powers` (`P_g,i = m_i g . v_i`
  from `State::body_point_velocity`; 0 on ground, NaN for a body missing
  from the mechanism), `gravity_vector` (sum of the mechanism's Gravity
  elements, so the mounting angle is respected), and the named rules
  `classify` (`NEUTRAL_REL` = 0.01), `force_share` (NaN when
  `|rate| < EPS_REL_LDOT * max|rate|`, `EPS_REL_LDOT` = 0.01),
  `is_braking` (`BRAKE_TOL_REL` = 1e-6), `max_abs_finite`.
- Tests: `analysis::gravity_breakdown::tests` (single bar hand calculation
  rising / falling / horizontal; sign flip with direction; potential-energy
  finite difference on a 4-bar under 30 deg tilted gravity; sources sum to
  the built `Q_gravity . q_dot`; source ordering and skip rules; rule edge
  cases) and `tests/gravity_breakdown_reference.rs` (energy finite
  difference + sum check on ParallelogramActuator and ChebyshevLambdaActuator
  with two weights each, compound actuator bodies included).
- Mutation check done: negating `g` inside `gravity_powers` fails 4 unit
  tests and both integration tests.

## 2026-09-29 — Payload weights Task 2: id-addressed weight editing API
- `gui/state/blueprint_ops.rs`: index-based `remove_point_mass` /
  `update_point_mass` / `move_point_mass_to_body` replaced by
  `add_point_mass -> Option<String>` (sets `last_point_mass_kg`),
  `find_point_mass`, `move_point_mass` (same body = reposition, other body =
  reattach), `set_point_mass_mass`, `set_point_mass_label` (trimmed, blank
  clears), `remove_point_mass_by_id`. Each validates with
  `io::point_mass_skip_reason` and records exactly one undo entry through
  `mutate_and_rebuild`; invalid = no entry; no-op = no entry.
- Undo fidelity: `take_snapshot` now calls the extracted
  `overlay_blueprint_point_masses` (shared with `serialize_to_json_string`),
  so undo/redo restores the weight list with ids and labels instead of baking
  weights into the link mass (gap noted in the BL-025 entry below).
- Callers migrated with no visible UI change: property panel pending edits
  (`SetPointMassMass`, `SetPointMassPosition`, `RemovePointMass`,
  `ReassignPointMass`, `RepositionPointMass` carry `weight_id`), canvas Move
  to Link / Reposition (`reassigning_point_mass` / `repositioning_point_mass`
  are `(body_id, weight_id)`), Place Mass uses `last_point_mass_kg`
  (`AppState` field, default 1.0). The weight mass DragValue no longer clamps
  a loaded out-of-range mass (e.g. 1500 kg, or 0 kg) on an idle frame.
- Tests: `gui/state/tests.rs` (BL-025 tests on the id API, now also asserting
  undo restores the weight lists; add/move/label/remove, invalid targets,
  no-op edits, undo/redo list restore, repairing loader-skipped weights),
  `property_panel::tests` (idle-frame clamp), `property_panel::pending_edits::tests`,
  `canvas::tests::weight_clicks` (headless Reposition, Move to Link, Place Mass clicks).
  Their fixtures take link ids from `gui::test_support::sorted_link_ids`.

## 2026-09-29 — Payload weights Task 1: point-mass ids, labels and loader validation
- Spec: `docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md`
  (Track 2, section 1 "Data model").
- `io/schema.rs`: `PointMassJson` gains `id: String` (`#[serde(default)]`) and
  `label: Option<String>` (skipped when absent). `assign_point_mass_ids` gives
  blank/duplicate ids the smallest unused `W<n>` (bodies sorted by id, list
  order; idempotent); `next_point_mass_id` for new weights.
- `io/from_json.rs`: `point_mass_skip_reason` rejects weights on ground,
  non-positive/non-finite masses (a negative mass used to be subtracted) and
  non-finite positions; the loader skips them; `point_mass_warnings` lists them.
  `apply_point_masses` is the one place weights reach the physics: the loader
  and BL-024's `sync_live_mass_props` both call it, so a base-mass edit skips
  the same weights a rebuild skips.
- `gui/state/file_io.rs::load_from_json_str` assigns ids and pushes warnings
  into `error_log` (error panel opens). `add_point_mass` assigns the next id.
- New `gui/test_support.rs` (`#[cfg(test)]`): `sorted_link_ids`, the shared
  "sorted non-ground link ids" fixture helper for GUI module tests.
- Tests: `io::schema::point_mass_id_tests`,
  `io::from_json::point_mass_validation_tests`, and in `gui/state/tests.rs`
  old-file load gets ids, ids + labels round-trip through save and share URL,
  invalid weights skipped and reported, and
  `set_body_mass_skips_point_masses_the_loader_rejects` (mutation: applying
  every weight in `sync_live_mass_props` turns it red); BL-023 round-trip
  helper now compares ids and labels too.

## 2026-09-29 — Magcoupling M1: math audit
- Independent audit of the vendored magcoupling 1.0.0 engine
  (`reference/magcoupling-py/`, the workbook port) against re-derivations, a
  2D field model, magpylib 3D, limit and scaling laws and literature data.
  787 checks: 741 pass, 46 fail. Each failing root-cause group was judged by
  three skeptic lenses; two check bugs were fixed test-first (`0b7c162`).
- Findings: 14 engine errors (14 checks), 15 model approximations
  (29 checks), 3 placeholder inputs (3 checks); 573 engine checks confirmed
  correct (164 reference self-tests, 2 harness and 2 consistency-only
  checks excluded).
- Engine errors that change a default output: adhesive shear modulus
  (C96 is EA 9514's, not AA 326's; C106 and C202 flip to "Below"),
  clamp screw length omits the slit (M4 x 12 becomes M4 x 14, which
  protrudes from the 25 mm boss), N42SH library Br 1.29 T is below K&J's
  1.30 T minimum (clamp recommendation fails from 1.3025 T), pole-sweep hub
  wall counts the bondline, rear-web eddy loss 4x low (high-case steady
  temperature 0.04 degC under the limit), cup wall at flats counts the
  bondline. The hot-torque verdicts do not change.
- Report: `docs/analyses/2026-09-29-magcoupling-math-audit.md`; merged
  results and every skeptic verdict:
  `docs/analyses/2026-09-29-magcoupling-audit-results.json`. The Decision
  column (E1..E14) awaits the user's review before M2 planning.
- Tools added: `audit/tools/group_candidates.py`,
  `audit/tools/coverage_table.py` (run it with `PYTHONIOENCODING=utf-8` on
  Windows; the cp1252 console cannot print the docstrings).
- Final-review fixes: `audit/tests/conftest.py` now applies the family-marker
  rule only to items under `audit/`, so `pytest tests audit/tests` runs both
  trees in one session (3 conftest-rule cases added to
  `test_harness_smoke.py`: 790 checks, same 46 failures, engine coverage
  still 573). `docs/ai/03-structure.yaml` has a `reference:` block (layout,
  read-only engine, how to run the parity suite and the audit with the
  `.venv`). The report header records the library versions; E14 is tagged
  documentation-only. The check-fix commit is `0b7c162` (a message-only
  rewrite of the earlier `89e38ae`; tree unchanged).

## 2026-09-29 — BL-027: inverse and forward dynamics add the velocity-quadratic force Q_v
- Root cause: `M(q)` depends on `theta` when a CG is offset from the body
  origin, but neither dynamics solver had the Lagrange term that goes with it,
  `Q_v = [m theta_dot^2 A(theta) s_cg; 0]` per body. The "With Inertia" torque
  was wrong for bodies whose origin moves (parallelogram rocker, 50 kg at
  (0, 0.8), 1 rev/s: tau_ID - tau_statics = -3158 N*m, correct 0). Every
  pivot reaction was missing the `m omega^2 |s_cg|` centripetal load, even
  for a crank pinned at its own origin.
- Fix: new `assemble_quadratic_velocity_forces` (`solver/assembly.rs`), added
  to the RHS in `solve_inverse_dynamics` (`Phi_q^T lambda = -(Q + Q_v - M q_ddot)`)
  and `forward_dynamics::compute_rhs`. Docs: ANALYSIS_MODES.md and
  NUMERICAL_FORMULATION.md now use the code's ID sign and note that forward
  dynamics uses the opposite lambda sign.
- Tests: `quadratic_velocity_forces_hand_computed`,
  `constant_ke_parallelogram_inverse_dynamics_torque_equals_statics`,
  `inverse_dynamics_torque_minus_statics_matches_ke_rate` (energy check),
  `pivot_reaction_is_centripetal_force_for_crank_pinned_at_origin` (Newton),
  and forward `free_spin_about_off_origin_pivot_conserves_speed_and_energy`.
  Mutation: dropping Q_v from the solvers turns the constant-KE, KE-rate,
  free-spin and Newton tests red (Newton: pivot reaction [0, 0]; a negated
  Q_v gives the reversed vector).
- Golden ID fixtures come from the Python reference, which still omits Q_v.
  `tests/golden_fixtures.rs::quadratic_velocity_lambda_shift` bridges them
  until BL-028 fixes Python and regenerates them. BL-029 tracks the separate
  motion-profile alpha-term error (`gui/sweep/motion_profile.rs`).

## 2026-09-29 — BL-024: set_body_mass / set_body_izz keep point masses in the live mechanism
- Root cause: `AppState::set_body_mass` / `set_body_izz` (`gui/state/blueprint_ops.rs`)
  wrote the blueprint BASE mass/Izz straight onto the live `Mechanism` body
  without rebuilding, so the live body lost every point-mass contribution
  until the next rebuild. Gravity Q, statics and the reaction solve used the
  bare base mass (rocker 1 kg + 2 kg point mass, base set to 4 kg: live mass
  4 kg, expected 6 kg). Composite CG and Izz also depend on base mass, so
  they went stale too (base-Izz edit left the parallel-axis term wrong).
- Fix: both setters now edit the blueprint base value and call the new private
  `sync_live_mass_props`, which recomputes the composite (base + point masses)
  with `Body::add_point_mass`, the same math the loader applies, and copies
  mass/CG/Izz onto the live body. No rebuild, so animation state is untouched.
- Parametric `BodyMass`/`BodyIzz` sweep: already correct (clones the
  blueprint, sets the BASE value, rebuilds through the loader, so point masses
  stay on top). Documented on `SweepParameter` (`gui/state/parametric.rs`) and
  locked by a test. The property panel mass/Izz sliders also edit the base.
- Tests (`gui/state/tests.rs`): `set_body_mass_keeps_point_masses_in_live_mechanism_bl024`
  (live composite == new base + point mass == fresh build of the blueprint;
  gravity Q_y == -g * composite mass with no rebuild),
  `set_body_mass_to_zero_leaves_only_point_mass_in_live_mechanism_bl024`,
  `set_body_izz_keeps_point_masses_in_live_mechanism_bl024`,
  `parametric_body_mass_sweep_varies_base_and_keeps_point_masses_bl024`.
  Mutation check: scaling gravity mass by 0.9 in `evaluate_gravity` turns the
  mass test red (Q_y -52.97 vs -58.86); restored.
- Not fixed here: undo snapshots still bake point masses in (BL-023 note).


## 2026-09-28 — BL-025: point-mass edits, Reposition and Move to Link are single undo steps
- Root cause: `update_point_mass` never called `push_undo` (its doc said the
  caller should; neither caller did), so numeric mass/X/Y edits and the canvas
  Reposition click could not be undone. `remove_point_mass` pushed an undo
  entry before validating the body/index, so an invalid target left a phantom
  entry. Move to Link (canvas handler) was `remove_point_mass` +
  `add_point_mass` = two undo entries and two rebuilds.
- Fix (`gui/state/blueprint_ops.rs`): `update_point_mass` and
  `remove_point_mass` validate (private `point_mass_exists`) and only then push
  undo; new `move_point_mass_to_body` does validate, one `push_undo`, move,
  one `rebuild`, and keeps the mass in place if the destination body is
  missing. `gui/canvas/interaction.rs` Move to Link calls it. The property
  panel already emits `UpdatePointMass` only on drag-stop / typed commit
  (never per drag frame), so one committed edit = one undo entry.
- Tests (`gui/state/tests.rs`): `point_mass_{numeric_mass_edit,numeric_position_edit,
  reposition,move_to_link}_is_one_undo_step_bl025` assert exactly one new undo
  entry, a real change to the built composite mass/CG/Izz, and that one
  `undo()` restores the prior composite properties and undo depth;
  `point_mass_invalid_targets_create_no_undo_entry_bl025` covers bad
  body/index for remove, update and move (no entry, model untouched).
- Not fixed here: `add_point_mass` still pushes undo before checking the body
  exists (no UI path hits it); undo snapshots still bake point masses in and
  drop the editable list (see BL-023 note), so after an undo the composite is
  correct but the point-mass list is empty.


## 2026-09-28 — BL-023: save / autosave / share URL no longer double-count point masses
- Root cause: `AppState::serialize_to_json_string` built the body JSON from
  the live mechanism via `mechanism_to_json` (composite, point-mass-inclusive
  mass/CG/Izz) and then re-attached the blueprint `point_masses`; load applies
  the list on top of the stored values, so every point mass was counted twice
  (rocker 1 kg + 50 kg -> 101 kg after one share-URL round trip). Affected
  native save, native/WASM autosave, and share URLs (single serialization point).
- Fix: for bodies that have point masses, serialize the blueprint BASE
  mass/CG/Izz next to the list (`gui/state/file_io.rs`). Bodies without point
  masses are unchanged.
- Tests (`gui/state/tests.rs`): `point_masses_not_double_counted_by_{serialize_load,
  share_url,file_save_load}_bl023` assert built composite and blueprint base
  mass/CG/Izz plus the point-mass lists unchanged to 1e-12, incl. a second
  save/load generation and base mass after removing the point masses.
- Files saved by the buggy build (with point masses) already contain inflated
  composite mass; they are not repaired on load.
- Not fixed here: undo/redo snapshots (`take_snapshot`) still bake point masses
  into body mass and drop the editable list (physics-consistent, list lost);
  BL-024 covers `set_body_mass`/`set_body_izz` dropping point masses live.


## 2026-09-13 — BL-020: slow playback no longer frozen (crank slider write-back)
- Root cause: `egui::Slider::step_by(0.5)` snaps its bound value to the
  step grid on every frame even with no input (verified headlessly on
  egui 0.32.3); the input panel then re-solved the pose at the snapped
  value, undoing any animation step < 0.25 deg/frame (frozen below
  ~15 deg/s at 60 fps, all mechanisms, native + WASM) and quantising the
  pose to 0.5 deg while paused. Predates Phase 0 — not a batch-1 change.
- Fix: dropped `step_by(0.5)` from the crank-angle slider
  (`gui/input_panel.rs`); a non-stepped slider only writes back on real
  interaction, so the existing value-diff gate now fires only for drags
  and typed edits.
- Test: `gui::input_panel::tests::idle_frame_does_not_move_driver_angle_bl020`
  drives one idle frame through `draw_input_panel`, playing and paused.
- Same-class Mounting Angle slider logged as BL-021 (collapsed by default).

## 2026-09-12 — BL-010: canvas actuator label agrees with the actuator force plot
- Label value extracted into `AppState::actuator_label_force` returning
  `ActuatorLabelForce::{Computed, Stored}`; the canvas only formats it.
  The `la.force ≈ 0` gate is gone: stored-force mode now shows the same
  statics sample the plot draws (was +2225 N stored vs -2223 N plotted on
  ChebyshevLambdaActuator).
- New `SweepData::index_at_driver`: circular nearest-angle lookup in angle
  mode (seam-crossing range sweeps such as 200..365 and wrapped driver
  angles resolve to the same sample; 0/360 tie -> first sample), linear
  nearest-stroke in metres in stroke mode, NaN samples skipped.
- Actuator force plot no longer drops finite samples: `filter_actuator_outliers`
  (Tukey fence) removed for both the Statics and With-Inertia series; both
  now build their points via `plot_panel::finite_series` (finite-only pairing).
  Near-singular spikes now stretch the auto Y range; Y zoom persists so the
  flat part stays inspectable. A `default_y_bounds` soft view was tried and
  rejected: egui_plot 0.33 re-applies default bounds every frame and locks
  the axis — the same is already true of X via `with_default_x_bounds`
  (new BL-019).
- Tests: `tests/actuator_force_label.rs` (promoted from the BL-010 repros,
  assertions inverted), `sweep::tests::index_at_driver_*`,
  `plot_panel::tests::outlier_fence_*` (finite_series keeps every finite
  spike the old fence dropped). `tests/braindump_repro.rs` now only
  holds the BL-017 repro.

---

## 2026-08-08 — Agentic test-and-improve loop bootstrapped
- Phase 0: fixed linear_driver doctest (text fence) + 3 deny-level
  approx_constant clippy errors; cargo test --all and clippy now green.
- Added scripts/gate.sh (test + clippy + WASM check gate).
- Added docs/ai/backlog.yaml (9 seed items) as the loop queue;
  04-memory active_issues now points at it.
- Added .claude/workflows/{audit-campaign,gui-smoke,fix-campaign}.js.
- Spec: docs/superpowers/specs/2026-08-08-agentic-test-improve-loop-design.md;
  plan: docs/superpowers/plans/2026-08-08-agentic-test-improve-loop.md.
- Runtime note: the Workflow tool delivers `args` as a JSON string; all three
  workflow scripts normalize via an ARGS shim (discovered during selftest,
  fixed in 91f4fa4 and applied to the other two scripts).

---

## 2026-05-28 — validate-the-validator review: close the validator's coverage gaps

**What:** A read-only adversarial-review workflow (5 lenses, findings
adversarially verified) examined `body_equilibrium_residual` +
`compute_validation` — the trust anchor written solo. Verdict: the
validator's PHYSICS is correct, but its TESTS only exercised the force
half of a force-AND-moment check, so whole branches could regress to
garbage while the suite stayed green ("the correlated-error hole relocated
one level up — in the coverage, not the formula"). Closed every confirmed
gap with mutation-verified tests:

- **B1/B2 — moment term untested.** Zeroing `mr = net_m/char_len` passed
  the whole suite; the only residual-teeth test perturbed a force (caught
  by the force term alone). Confirmed independently: with `mr=0` the suite
  still caught the real `point_force_to_q` swap (force term + FBD values
  catch force-applied bugs) — but moment-ONLY load types (external torque,
  driver couple, fixed-joint moments) would slip. Added
  `body_equilibrium_residual_catches_moment_only_imbalance`: bumps the
  driver couple (pure moment, no force) and asserts the moment term
  catches it; its `good < 1e-9` baseline also pins the cross-product sign.
  Mutation-verified (passes with the moment term, fails with `mr=0`).
- **W1 — external force/torque handlers had zero coverage** (flipping
  either sign passed all tests). Added
  `validation_verified_with_external_force_and_torque`; mutation-verified
  both handler signs.
- **W3 — no `Failed` verdict ever asserted end-to-end.** Added
  `validation_failed_when_ill_conditioned` (feeds cond > 1e8 and non-finite
  → Failed; also pins `Failed→!is_valid` and `Unverified→is_valid`) and
  `validation_failed_on_nan_reaction`. (The driver-collapse firing side was
  pinned in the prior commit.)
- **W2 — force-zone overlap gate + centroid are SHARED with production,
  not independent.** Softened the overstated independence comment to be
  precise (independence holds for the moment-arm projection; the overlap
  decision and centroid are shared, narrow blast radius). Added
  `validation_verified_force_zone_centroid_branch` to exercise the
  previously-unreached `body_local_app_point: None` centroid path.
- **N3** — documented the body_j-uses-body_i-joint-point reuse as a
  fail-safe (only ever adds residual) in a code comment.

**Process scar:** lost the five new tests mid-session by using
`git checkout -- reactions.rs` to revert an in-main mutation — it also
wiped the uncommitted tests. Re-applied. Rule reinforced: mutation-test in
an isolated worktree (as done for the earlier campaign), never
`git checkout` a file holding uncommitted work.

**Counts:** 711 → 716 lib tests passing. Wasm32 clean.

**Spec status:** the reaction validator's test coverage now matches the
soundness it actually has. Remaining (deferred, documented): N4 global
force_scale normalisation (float-drift band only), N5 external-on-ground
silent drop (config-lint gap, not a validator defect).

---

## 2026-05-28 — mutation-test the validator; pin the driver-collapse gate

**What:**
- Ran a systematic mutation-testing campaign against the reaction
  validator (`body_equilibrium_residual` + `compute_validation`) in an
  isolated git worktree: flipped each sign/handler (gravity, the moment
  cross-product two ways, driver couple, joint Newton-3, actuator force,
  force-zone force) plus two production mutations (point_force_to_q
  moment-arm swap, actuator Newton-3 sign). Every behaviourally-distinct
  mutation was CAUGHT by the existing tests — the validator's sign logic
  is well pinned.
- Caught a harness footgun in the process: a `replace(count=1)` anchor
  `let force_on_a = -force_on_b;` matched the gas-spring element before the
  actuator, so the first "actuator" mutation was actually a no-op gas-spring
  mutation. Lesson recorded in the fbd-derive skill workflow: use
  occurrence-specific anchors for mutation testing.
- Found ONE real coverage gap: disabling the pass-2 driver-collapse gate
  in `compute_validation` was invisible to the entire suite. That gate is
  the ONLY thing that catches a wrong back-solved actuator force —
  per-body equilibrium cannot, because the reactions self-consistently
  balance whatever force was injected (residual stays ~1e-12). If a
  refactor broke the gate, wrong-physics could regress silently to a green
  badge — the exact failure mode the 2026-05-28 fix removed.
- Closed it with `driver_collapse_gate_catches_wrong_actuator_force`:
  builds pass-2 reactions from a deliberately-2× actuator force, asserts
  (a) per-body equilibrium is fooled (resid < 1e-9 — documents WHY the gate
  is needed) and (b) `compute_validation` returns `Failed`. The test was
  itself mutation-verified: it passes with the gate and fails without it.

**Counts:** 710 → 711 lib tests passing.

**Process note:** a read-only "validate-the-validator" adversarial-review
workflow ran concurrently in the main worktree while mutation testing ran
in an isolated one (no interference). Its findings, if any survive
verification, to be folded in separately.

---

## 2026-05-28 — extend independent reaction check to force zones + external forces

**What:**
- `body_equilibrium_residual` now models `ForceZone`, `ExternalForce`,
  and `ExternalTorque` in addition to gravity + a single LinearActuator.
  Each is rebuilt from element data / geometry (force-zone overlap via the
  `crate::geometry` helpers; external force/torque via `modulation.factor(t)`),
  NOT through `point_force_to_q`, preserving independence from the
  moment-arm projection.
- Threaded `t` through `body_equilibrium_residual` and `compute_validation`
  (needed for external-force/torque time modulation).
- The decoded press mechanism (gravity + sizing actuator + a 3 kN
  `ForceZone` on the output link) now reports **Verified** instead of
  Unverified.
- New regression test `validation_verified_with_force_zone`: a 4-bar with
  gravity + sizing actuator + a 1000 N force zone at an OFF-CG application
  point reaches Verified, residual < 1e-9 (exercises the zone moment term).
- Updated the GUI "Unverified" badge hover text to list the now-covered
  set (revolute + driver + gravity + single actuator + force zones +
  external forces/torques) and the still-unmodeled set (springs, dampers,
  motors, fixed/prismatic joints, multiple actuators).

**Counts:** 709 → 710 lib tests passing. Wasm32 clean.

**Spec status:** Option B independent check now covers the full element
set of the validated press class. Remaining unmodeled: rotary/velocity-
dependent elements (springs, dampers, gas spring, motor, bearing
friction, joint limit, torsion spring, rotary damper) and non-revolute
joints (fixed, prismatic, cam) → those still report Unverified.

---

## 2026-05-28 — reaction validation made non-circular (adversarial-review fix)

**What:**
- An adversarial-review workflow (degraded on a session-token limit, but
  its surviving artifacts + direct verification were conclusive) found
  that the 2026-05-21 `is_valid()` / equilibrium badge was **unsound**.
  The `residual_norm` check (‖Φ_qᵀλ + Q‖) is near-tautological: λ is
  SVD-solved to make exactly that zero, so for the square full-rank 4-bar
  Jacobian it sits at ~1e-14 regardless of whether the physics is right.
  Proven two ways: (a) a mutation test swapping the moment arm in
  `point_force_to_q` left `residual_norm` at 3.49e-14 while reactions were
  wrong; (b) a probe injecting a spurious 100 N force on the rocker left
  BOTH `is_valid()` criteria passing while J4's reaction was 2× off.
- Replaced with an INDEPENDENT per-body Cartesian Newton-Euler check
  (`solver::reactions::body_equilibrium_residual`): for each moving body,
  sum world-frame joint reactions (from `force_global`, Newton-3 signs, at
  joint geometry) + driver couple + applied forces (gravity, actuator from
  geometry — NOT via `point_force_to_q`), assert ΣF and ΣM ≈ 0 relative to
  force scale. Genuinely independent of the solve, so it catches
  sign/frame/Jacobian errors the generalised residual hides.
- New tri-state `ValidationState { Verified, Unverified, Failed }`.
  `Unverified` is returned (never a false green/red) when the mechanism
  has element/joint types the check doesn't yet model — currently it
  covers gravity + a single LinearActuator on a revolute 4-bar; force
  zones, springs, fixed/prismatic joints, multiple actuators → Unverified.
- Added a condition-number ceiling (1e8) and kept the pass-2
  driver-collapse check (relative) as a complementary gate that catches
  wrong back-solved actuator force.
- GUI badge is now tri-state and honest: green "Reactions verified",
  grey "not independently verified", red "Reaction check FAILED", with
  hover text matching what is actually checked.
- `ForceResults.equilibrium_valid: Option<bool>` →
  `reaction_validation: Option<ValidationState>`.
- Cleaned up workflow-agent test debris: removed two `#[ignore]` eprintln
  probes; kept + renamed the trajectory-path test
  (`qdot_variant_is_rate_invariant_and_matches_constant_speed`); reverted
  an orphaned mutation-test swap a review agent left in `helpers.rs`.
- New regression tests: `body_equilibrium_residual_catches_perturbed_reaction`
  (the headline — proves the check has teeth), plus Verified/Unverified
  verdict tests.

**Process note:** the review workflow's agents made uncommitted edits to
`src/` (a mutation probe + added tests). Read-only review intent was
violated; the `fbd-math-reviewer` agent definition grants Bash. Worth
tightening agent tool scopes before the next review workflow.

**Counts:** 705 → 709 lib tests passing. Wasm32 clean.

**Spec status:** Option B is now genuinely sound (was shipped unsound on
2026-05-21). Remaining: extend `body_equilibrium_residual` to force zones
+ external forces (the real press mechanism has a ForceZone, so it
currently shows "Unverified"), then fixed/prismatic joints.

---

## 2026-05-21 — validation Option B: equilibrium check + GUI badge

**What:**
- `ReactionSolveResult` gains `residual_norm: f64` (the `‖Φ_qᵀλ + Q‖`
  for the final solve) and an `is_valid()` method that combines the
  residual threshold (1e-6 N absolute) with the pass-2 driver-torque
  collapse check (1e-6 N·m when `used_two_pass`).
- Fixed a real gap: pass-2's `residual_norm` was previously hardcoded
  to `0.0` (never computed). A failed pass-2 SVD could silently ship
  wrong lambdas. Now computed and surfaced through `is_valid()`.
- `solve_reactions_with_actuator` `debug_assert!`'s `is_valid()` at
  the end of its pass-2 path. Dev builds panic if equilibrium fails;
  release builds rely on the GUI badge and FBD regression tests.
- `ForceResults` gains `equilibrium_valid: Option<bool>`, populated by
  `recompute_force_results` from the helper's `is_valid()`.
- GUI property panel: new green/red badge ("✓ Equilibrium check
  passed" / "✗ Equilibrium check FAILED") at the top of the Joint
  Reactions section, with hover text explaining the threshold and
  pointing at the FBD tests for diagnosis when it goes red.
- New test `is_valid_reports_true_for_well_conditioned_solves` covers
  all three branches (no-actuator pass-1, sizing-mode pass-2,
  known-force pass-1).

**Why it matters:**
- The FBD-validated regression tests cover 5 specific poses with
  absolute-truth comparison. The equilibrium badge layers on top of
  that: it runs on EVERY pose the GUI ever displays, catching any
  bug class that produces reactions inconsistent with the applied
  forces. Includes the previously-silent pass-2 SVD failures.

**Counts:** 704 → 705 lib tests passing. Wasm32 build clean.

**Spec status:** Option A (5 FBD-validated poses) + Option B
(equilibrium check + badge) both shipped. Option C (external tool
comparison) remains open; idle until you produce a reference.

---

## 2026-05-21 — FBD validation extended to θ_2=0.9π (near-singular)

**What:**
- New regression test `fbd_validates_pass2_reactions_near_singular`.
  At θ_2 = 0.9π the linkage is ~18° from its toggle limit at θ_2 = π
  (where coupler+rocker are collinear and BD = b + c = 5). Transmission
  angle at C ≈ 165°. Loop-closure discriminant ≈ 0.01 vs ~16 for the
  linear-term squared — the near-singular signature (two roots almost
  identical).
- `assert_pass2_matches_fbd` now takes `tol` as a parameter; the four
  existing well-conditioned poses pass `1e-4`; this one passes `1e-2`
  to absorb the residual that the SVD-based solve produces when the
  answer magnitude is large. Driver-torque tolerance scales
  proportionally (`tol * 1e-2` with a floor of `1e-6`).
- Passed first-run, which is mildly surprising — the simulator's
  numerical conditioning at this pose is better than worst-case
  expected.

**Why it matters:**
- The "test numerical conditioning" goal from the validation spec is
  now covered. If the simulator's SVD ever degrades at near-singular
  configurations, this test catches it loud and clear before the
  GUI's reaction display starts producing nonsense values near
  toggle limits.

**Counts:** 703 → 704 lib tests passing.

**Spec status:** 5 of 4–6 FBD-validated poses done. Option A is now
above the spec's recommended upper bound; declaring option A complete
unless the user wants the 5π/4 mirror too.

---

## 2026-05-21 — FBD validation extended to θ_2=π/4 (mid-stroke, generic case)

**What:**
- New regression test `fbd_validates_pass2_reactions_at_45deg`.
  Mid-stroke pose with Bx ≠ 0 AND By ≠ 0 — exercises every coefficient
  slot in `solve_fbd_pass2_for_pose`, including the Crank-M row that
  collapsed to `R_J1x + R_J2x = 0` at TDC/BDC (Bx = 0). The 60° test
  had Bx ≠ 0 too but the discriminant gave clean √3 numbers; this
  one gets √2 with √(3 + √2) in the discriminant root.
- Loop-closure quadratic: `(68 − 16√2)·Cx² + (−336 + 40√2)·Cx + 424 = 0`.
  Discriminant `768 + 256√2 = 256·(3 + √2)`. Open branch C ≈ (3.450, 1.923).
- Code computes the quadratic coefficients from `sqrt2` directly
  rather than baking high-precision literals, so the algebra is
  inspectable. Passed first-run.

**Counts:** 702 → 703 lib tests passing. Spec's lower bound of 4 poses now met.

**Spec status:** 4 of 4–6 FBD-validated poses done (π/4, π/3, π/2, 3π/2).
Remaining candidates: near-singular (numerical conditioning),
5π/4 (lower-half-plane generic, mirror of π/4).

---

## 2026-05-21 — clear long-standing tests/ crate red

**What:**
- `tests/compound_force_integration.rs`: +6 fields (`sweep_state`,
  `sensor_config`, `motion_profile`, `actuator_rated_force`,
  `simulation_duration`, `parametric_config`) added to each of 2
  `MechanismJson` literals — 12 added lines total. All defaulted to
  `None` / `0.0` per the struct's `#[serde(default)]` semantics.
- `tests/force_zone_tests.rs`: +1 field (`body_local_app_point: None`)
  added to each of 9 `ForceZoneElement` literals.
- Cleared 11 `E0063` compile errors that had blocked
  `cargo test --tests` for weeks (stale fixtures from earlier
  `MechanismJson` / `ForceZoneElement` refactors).
- Dispatched as a background `rust-test-fixer` subagent run; output
  verified independently before commit.

**Why:**
- Long-standing red on `cargo test --tests` was a continuous
  distractor and hid any real test-crate regressions. Now clean.

**Counts:** integration tests now 78/78 passing across 10 binaries
(was 0 of 78 — they wouldn't even compile). Lib tests unchanged at
702/702.

**Conservatism:** all changes are additive `: None` / `: 0.0`
field initialisers on test fixtures. No struct definitions touched.
No test intent altered.

---

## 2026-05-21 — FBD validation extended to θ_2=3π/2 (BDC) + branch-aware seed

**What:**
- New regression test `fbd_validates_pass2_reactions_at_270deg`
  (bottom dead center, crank vertical pointing down). Same quadratic
  as TDC but with `Cy = 10 − 4·Cx`, so the open branch C is at
  `((44 + 4√2)/17, (−6 − 16√2)/17) ≈ (2.921, −1.684)`. Passed
  first-run.
- `seed_pose_at_angle` now picks the rocker initial guess based on
  `angle.sin()`: upper half-plane crank → rocker θ = +π/2 (existing
  behaviour); lower half-plane → rocker θ = −π/2 (new). Without this,
  the rocker seed at +π/2 would conflict with the coupler seed (whose
  C_y has the same sign as sin(crank angle)) and Newton might pick
  the wrong branch or fail to converge. Existing π/3 and π/2 callers
  are unaffected (both have sin > 0).

**Counts:** 701 → 702 lib tests passing.

**Spec status:** 3 of 4–6 FBD-validated poses done (π/3, π/2, 3π/2).
Mid-stroke (π/4 or 5π/4) and near-singular are the remaining
candidates.

---

## 2026-05-21 — FBD validation extended to θ_2=π/2 + helper extracted

**What:**
- New regression test `solver::reactions::tests::fbd_validates_pass2_reactions_at_90deg`
  for top-dead-center (crank vertical). Open branch root from loop
  closure: `Cx = (44 + 4√2)/17 ≈ 2.921`, `Cy ≈ 1.684`. Test passed
  first-run.
- Refactored: extracted `solve_fbd_pass2_for_pose(bx, by, cx, cy) → [f64; 9]`
  + `assert_pass2_matches_fbd(mech, q, θ, expected, label)` helpers
  from the inline 60° test. Each new pose is now ~10 lines of
  derivation + a single helper call instead of ~150 lines of matrix
  construction. The 60° test still carries the full FBD derivation
  comment block as the canonical worked example.
- Generalized `seed_pose_at_60deg(mech)` → `seed_pose_at_angle(mech, θ)`.
  Three existing callers updated.

**Why:**
- Adding more poses is now ~30 min each, down from ~1 hour. Lowers
  the cost of broader validation coverage per the spec
  (`docs/superpowers/specs/2026-05-18-reaction-force-validation-design.md`).

**Counts:** 700 → 701 lib tests passing.

**Spec status:** option A is 2 of 4–6 poses in. Remaining candidates:
mid-stroke (π/4 or 5π/4), bottom-dead-center (3π/2), near-singular,
negative-y branch. Each is now a small follow-on.

---

## 2026-05-18 — FBD validation test for 4-bar + LinearActuator at θ_2=π/3

**What:**
- New regression test `solver::reactions::tests::fbd_validates_pass2_reactions_at_60deg`.
  Builds the 9×9 Cartesian equilibrium system by hand (force balance +
  moment balance per body) for the crank-rocker + sizing-mode
  LinearActuator config, solves it with `nalgebra::full_piv_lu`, and
  asserts the simulator's pass-2 reactions and back-solved F_act match
  the independent FBD solution within 1e-4 N at pose θ_2 = π/3. Also
  checks the pass-2 driver torque lambda is ~0 (the point of pass-2).
- Caught and corrected a sign-convention bug in the FBD derivation
  during the write-up: `evaluate_linear_actuator` applies
  `force_on_a = −F_act·u` (with `u = (p_b − p_a)/|·|`), so positive
  F_act = extension. My first cut had `+F_act·u` on body_a, which is
  the compression convention. Same physical force; flipped sign on
  F_act in the answer. Now matches.

**Why this matters:**
- The J3-mismatch bug (commit `492fbb7`) shipped because no test
  asserted absolute correctness — only solver self-consistency
  (residual small, lambdas finite). This is the first absolute-truth
  anchor: the simulator's `Φ_qᵀ λ = −Q` matches an independently
  written Cartesian equilibrium at a verified pose. Catches future
  sign-flipped Jacobians, pass-2 double-counts, swapped actuator
  endpoints, mass-to-Q mapping errors, etc.

**Counts:** 699 → 700 lib tests passing.

**Spec status:**
`docs/superpowers/specs/2026-05-18-reaction-force-validation-design.md`
option A is now partially in (1 of 4–6 poses). Extending to additional
poses (top-dead-center, near-singular, etc.) is incremental; the
infrastructure is in place.

---

## 2026-05-18 — focus shift + validation-design spec

**What:**
- Updated `docs/ai/01-meta.yaml` `active_focus` to reaction-force
  validation for 4-bars with a LinearActuator (was: Custom 6-Bar press
  workflow). The 6-bar workflow is no longer the user's stated focus.
- New `docs/superpowers/specs/2026-05-18-reaction-force-validation-design.md`
  documenting three approaches — closed-form FBD (A), internal cross-
  checks (B), external tool comparison (C) — plus a Hybrid path
  (A+B). Recommendation: A first, then B.
- New entry in `docs/ai/04-memory.yaml` `open_questions` tracking the
  reference-choice decision that blocks the implementation plan.

**Why:**
- The J3-mismatch bug (commit `492fbb7`) shipped because existing
  reaction tests only assert solver self-consistency, not absolute
  correctness. Validation work is the natural next focus to lock in
  trust before further hardware-deployment work.

---

## 2026-05-12 — repo hygiene: gitignore browser-automation scratch

**What:**
- Added `.playwright-mcp/`, `.superpowers/`, and `/linkage-*.png` to the
  repo-root `.gitignore`. These artifacts are produced by Playwright MCP
  sessions and ad-hoc Claude Code work poking at the deployed web app;
  they should never have been tracked but had been showing as `??`
  untracked entries across sessions.

**Why:**
- Lowers the noise floor on `git status` so real working-tree changes
  stand out. No behavioural impact.

**Known noise still in the tree (not addressed here):** the
`docs/chebyshev_lambda/*.png` files drift by 1–2 bytes when the raster
GIF tests run — committed PNG metadata is non-deterministic across
runs. Worth investigating which test is regenerating them and either
making it deterministic or routing it to a scratch path.

---

## 2026-05-01 — docs + UX: trajectory mode walkthrough

**What:**
- New `docs/guides/TRAJECTORY_MODE.md` walkthrough — concept, when-to-use,
  5-minute quick-start on the canonical 4-Bar Crank-Rocker, plot reading,
  failure modes (R/S/B/N glyphs), profile selection, correctness
  verification recipe, advanced features (comparison overlay, motion
  ribbon, playback, exports), and a troubleshooting table.
- Linked from `README.md` doc index.
- One-line UX fix in `gui/mod.rs`: switching `Sweep mode → Trajectory`
  now calls `mark_sweep_dirty()` so the trajectory plot populates
  immediately instead of leaving the user staring at "No trajectory
  data yet" until they hunt for the Compute button.
- New regression test `compute_trajectory_world_x_target_tracks_target`:
  drives WorldX of crank.B along a constant-speed profile from 0.005
  to 0.009 m, asserts achieved tracks target within 1e-6 and the per-
  sample pose snapshot's θ_crank satisfies the inverse identity
  `x = 0.01·cos(θ)`. Complements the existing trivial-Angle test by
  exercising the Newton outer loop on a non-linear u→h relationship.

**Why:** User feedback was "not obvious how to use it." Three things
needed to land together: (1) a real walkthrough that explains *why*
trajectory mode exists vs. forward sweep, (2) an in-app cue that
trajectory data appears immediately on mode switch, and (3) a
regression test that pins correctness on a non-trivial target so
future refactors don't silently break the inverse solve.

**Test results:** 669 lib tests pass (was 668 + 1 new). Native + WASM
both compile clean.

---

## 2026-05-01 — fix(build): exhaustive cfg gating for `--no-default-features`

**What:** Inline audit of the recent web-pass commits caught two
configurations where cfg gates were not exhaustive across
`(feature = "native") × (target_arch = "wasm32")`:

1. `gui/export/download.rs::download_bytes_impl` had only the
   `feature = "native"` and `target_arch = "wasm32"` arms; a
   `--no-default-features` native build hit an unresolved symbol from
   the public caller. Added a third fallback arm returning
   `DownloadOutcome::Failed`.
2. `gui/state/templates.rs` gated the `dirs::home_dir()`-using
   functions (templates_dir, persist_*, remove_persisted_*,
   load_*_native) by `not(target_arch = "wasm32")` but `dirs` is in
   the `native` feature flag. Same shape of bug — `--no-default-features`
   native build had unresolved `dirs` crate. Switched to
   `feature = "native"` and added no-op fallbacks for the persist /
   remove methods so the ungated entry points (save_as_template,
   delete_template, save_as_custom_sample, delete_custom_sample) still
   link. The `load_saved_*` dispatchers got tri-state cfgs returning
   empty `Vec` for the unsupported configuration.

**Why:** The `chrono_now` panic on wasm32 (e7006ec) was the same shape
of bug — un-gated APIs that don't compile in every supported
configuration. Audit widened the verification to all four build
configurations: native default, native --no-default-features, wasm
default, wasm + raster. All now compile clean.

**Test results:** 668 lib tests pass.

---

## 2026-04-30 — feat(web): recent mechanisms list (Pass 4)

**What:** Added a 5-entry localStorage-backed ring buffer that captures
mechanism JSON snapshots when the user drag-and-drops a JSON or clicks
"Download JSON…". Surfaced as a "Recent Mechanisms" submenu under
File (web only) — clicking an entry restores it without re-dragging.

Storage key: `linkage_recent_mechanisms` (single localStorage entry,
JSON array of `(name, json_text, unix_secs)` tuples). New helpers
`AppState::wasm_push_recent_mechanism` and
`AppState::wasm_load_recent_mechanisms`. `format_relative_time` in
menu_bar.rs renders the timestamp as "5 min ago" / "2 hr ago" /
"3 days ago".

**Why:** Native has a Recent Files list; web didn't, so a user closing
the tab and reopening lost their session even with autosave (autosave
holds only the most recent state, not history). The ring buffer is the
web-equivalent of native's recent paths — paths don't translate to web,
so we store full JSON snapshots instead. Bounded to 5 entries to stay
under typical localStorage quotas.

**Discovery during work:** Most other "missing" persistence features
(Saved Templates, Custom Samples, Autosave) already had full
localStorage backings on WASM and just weren't documented as such.
Verified Save as Template / Manage Templates menu items show on web
and autosave-recovery prompt fires on startup. The web persistence
story is now complete.

**Test results:** 668 lib tests pass (no new tests; `js_sys::Date` is
WASM-only so the helpers can't be exercised under `cargo test`).
Native + WASM both compile clean.

---

## 2026-04-30 — feat(web): drag-and-drop imports for JSON + CSV (Pass 3)

**What:** Extended the existing drag-and-drop handler in
`gui/mod.rs::update()` to dispatch on file extension:
- `.json` → `state.load_from_json_str` (mechanism)
- `.csv` → `parse_keyframes_csv_str` (keyframes; only in Trajectory mode)
- `.dxf` → DXF overlay (existing)
- everything else → background image (existing)

Refactored `parse_keyframes_csv` (path-based, native-only) to layer over
`parse_keyframes_csv_str` (cross-platform). +3 tests covering simple
parse, comment/blank skipping, and empty-input rejection.

Updated the trajectory panel: hide the native-only Import CSV button on
web and show a "Drag a .csv onto the canvas" hint instead. Updated the
File menu's web-only drag-and-drop hint to enumerate all four supported
formats.

**Why:** After Passes 1 and 2 made every export work on web, the
export-import asymmetry was the next gap. Drag-and-drop already
worked for DXF and images on web (egui `RawInput::dropped_files`
delivers bytes on both targets), so extending it to JSON and CSV was
~30 lines of dispatcher code rather than building a full `<input
type=file>` + FileReader integration. An explicit "Open JSON…" menu
button on web is deferred — the channel-based async file-picker
architecture is documented but not built.

**Test results:** 668 lib tests pass (was 665 + 3 new). Native + WASM
both compile clean.

**Limitations:** No explicit menu file picker on web — drag-and-drop
only. Recent files / saved templates / autosave on web all still
require localStorage wiring (separate effort).

---

## 2026-04-30 — feat(web): browser-side raster + CSV downloads (Pass 2A + 2B)

**Pass 2A — CSV:** Refactored `gui/export/csv.rs` to factor out
cross-platform `write_sweep_csv` and `write_coupler_csv` writers
taking `&mut impl Write`. Public surface gained
`generate_sweep_csv_string` / `generate_coupler_csv_string` companions
to the existing path-writing functions. Both Sweep CSV and Coupler CSV
menu_bar buttons now route through `download::download_text` and work
on web. +4 tests pin file/String parity. 665 lib tests pass.

**Pass 2B — PNG/GIF:** Added a new `raster` feature flag in
`Cargo.toml` (the `native` feature now depends on it). Refactored
`gui/export/raster.rs` from `cfg(feature = "native")` to
`cfg(feature = "raster")` and added `generate_mechanism_png_bytes` /
`generate_mechanism_gif_bytes` companions. Extended `download.rs`
with a `download_bytes` helper for binary content. Menu_bar PNG/GIF
buttons gated by `raster` (so they appear on any build with the
feature, including web).

**Deployment:** Updated `scripts/build_web.sh` and
`.github/workflows/deploy-web.yml` to add `--features raster` to the
WASM build. Bundle cost is ~+2.5 MB pre-wasm-bindgen
(9.25 MB → 11.73 MB; post-processing brings the final shipped
`_bg.wasm` to a smaller fraction).

**Why:** After Pass 1 lit up text exports on web, raster was the last
gap blocking parity with the native export menu. Resvg + gif compile
to wasm32 cleanly — verified via `cargo check --no-default-features
--features raster --target wasm32-unknown-unknown` — so the only
question was bundle size. 27% growth is acceptable given the user
gets PNG and GIF in exchange.

---

## 2026-04-30 — feat(web): browser-side downloads for text exports (Pass 1)

**What:** Web build can now save files. Added `gui/export/download.rs` with
`download_text(filename, mime, contents, filter)` that branches by target:
native opens an `rfd` save dialog and writes bytes; web wraps the contents
in a `Blob`, mints an `ObjectURL`, programmatically clicks a transient
`<a download>` anchor, then revokes the URL. Same call site, both
platforms. Returns a `DownloadOutcome { Saved | Cancelled | Failed }`
that maps to the AppState status / error-panel UX via `apply_download_outcome`.

Refactored five export buttons in `gui/menu_bar.rs` to route through the
helper and dropped their `#[cfg(feature = "native")]` gate:
- Export firmware (JSON) — uses new `build_firmware_json_string`
- Export SVG — `generate_svg_string`
- Export labeled schematic SVG — `generate_schematic_svg`
- Export DXF — `generate_dxf_string`
- Generate Report (HTML) — `generate_html_report`

Added a web-only "Download JSON…" button as the Save analog (the web
build has no concept of a "current open file" so Save / Save As stay
native-only). The associated `serialize_to_json_string` was promoted to
`pub(crate)`. Made the `*_string` generators platform-independent by
dropping the over-broad `#[cfg(feature = "native")]` from `report.rs`,
`schematic.rs`, `svg.rs`, and `dxf.rs` — the gating was historical, the
function bodies have always been pure-Rust.

Pass 2 (Sweep CSV / Coupler CSV / PNG / GIF / DXF import / Image import
/ keyframe CSV import) remains native-only. The CSV generators write
directly to a `Path`; PNG/GIF use native-only crates (resvg, gif).

**Why:** A full month of trajectory work landed exports for the native
build only. On the deployed web app, every export menu item was either
absent or non-functional — so the user could compute a beautiful
trajectory plus pose snapshots and have no way to extract the data.
Pass 1 lights up the highest-value text exports without touching the
raster-export code path.

**Test results:** 661 lib tests pass (no new tests; the helper is
JS-side glue). WASM target compiles clean. Native dev build compiles
clean.

**Cargo.toml:** Expanded `web-sys` features with `Blob`,
`BlobPropertyBag`, `HtmlAnchorElement`, `HtmlElement`, `Element`,
`Document` for the blob-download path.

---

## 2026-04-29 — feat(traj): trajectory diff overlay (T3)

**What:** Added `AppState::trajectory_comparison: Option<SweepData>` plus
"Save as comparison" / "Clear" buttons in the trajectory panel's
Visualization section. When a comparison is saved, the trajectory plot
overlays its target/achieved/u traces in faded dotted style alongside the
live trajectory so the user can A/B two profiles, two targets, or two
severity settings on the same axes. The overlay carries the snapshot's own
time axis (not the live one) so different durations render honestly.
Implemented in `gui/plot_panel/trajectory.rs::comparison_traces_from`.

**Why:** Users iterating on trajectory design need a quick A/B without
exporting CSVs and replotting externally. Trapezoidal vs S-curve at the
same target, or the same profile pointed at two different targets, are
the canonical "did I improve it?" workflows.

**Test results:** 661 lib tests pass (657 baseline + 4 new). New tests
cover snapshot duration vs live-duration time axis, non-trajectory mode
rejection, partial-snapshot rejection, and degenerate sample-count
rejection.

---

## 2026-04-29 — feat(traj): per-sample pose snapshot in CSV/JSON exports (T4)

**What:** Added `pose_body_order: Option<Vec<String>>` and
`pose_snapshots: Option<Vec<Vec<[f64; 3]>>>` to `SweepData`.
`compute_trajectory` populates them with each non-ground body's
`(x, y, θ_rad)` at every sample. Trajectory CSV appends
`q_x_<body>_m, q_y_<body>_m, q_theta_<body>_rad` columns per body after
the v1 columns; firmware JSON adds an optional `pose: { body_id: [x, y, θ] }`
map per sample. Both are skipped when the snapshot is absent so the v1
schema stays valid for older / non-trajectory data.

**Why:** Downstream tools (firmware test harnesses, motion previewers,
report generators) need to replay the full pose without re-running the
inverse solve. Per-body columns are additive, so existing v1 consumers
keep working.

**Test results:** 657 lib tests pass (653 baseline + 4 new). New tests
cover CSV present/absent paths, JSON present/absent paths, plus an
extension to `compute_trajectory_populates_all_trajectory_fields`
asserting θ_crank in the pose snapshot tracks the Angle target on the
directly-driven body.

---

## 2026-04-29 — feat(eq): auto-generated labeled mechanism schematic (SVG)

**What:** Added `gui/export/schematic.rs::generate_schematic_svg(mech, q)` that
produces a self-contained SVG of the loaded mechanism with constraint labels,
matching the visual style of figures 2 & 3 in
`docs/superpowers/specs/2026-04-29-linkage-equations-reference.md`. The figure
includes a title strip with DOF, ground hatching, bars (one per non-ground
body), joint clusters (coincident joints render as one circle with merged
labels), italic body labels at each body's centroid, a red driver indicator
(curved arrow for `RevoluteDriver`, cylinder/piston line for `LinearDriver`),
and a sidebar legend listing constraint rows grouped by joint kind plus the
driver row's symbolic form (delegated to `gui::eq_rendering::symbolic_form`).
Wired into `gui/menu_bar.rs` as File → "Export labeled schematic (SVG)...".

**Why:** Part of the "linkage equations reference" body of work (E-series). The
goal is to give users a publication-quality labeled figure of their own
mechanism for design docs and lab reports — generated programmatically from
the loaded mechanism rather than hand-coded per-figure as in the spec.

**Test results:** 647 lib tests pass (642 baseline + 5 new). New tests cover
the 4-bar revolute-driver path, the `ParallelogramActuator` sample, an inline
`LinearDriver`-driven 4-bar (linear-driver indicator branch), an empty
mechanism (returns Err), and structural well-formedness of the emitted SVG.

---

## 2026-04-30 — feat(traj): analytic acceleration inverse via per-variant Hessian

**What:** Replaced the placeholder zero-Hessian arms in `ControlTarget::hessian`
with closed-form Hessians for `WorldX`, `WorldY`, `Projection`, and `Distance`
(Angle is linear in q, retains zero Hessian). Added
`solver/inverse_kinematics/derivatives.rs::inverse_acceleration_analytic` —
computes `r''(u) = ∇²g(dq/du, dq/du) + ∇g·(d²q/du²)` directly via the
analytic Hessian plus a "kinematic-only γ" assembly (existing `assemble_gamma`
with `dq/du` in place of `q̇` and the `Φ_tt` driver-row term zeroed). Eliminates
the two extra forward solves per sample that `inverse_acceleration_fd` requires.

**Why:** Resolves the `04-memory.yaml` open question on analytic acceleration.
While FD was sufficient for v1 trajectory analysis, the analytic path is more
accurate (no FD truncation error) and faster (one linear solve vs two forward
solves). Available for callers to opt into; `compute_trajectory` continues to
use FD by default per spec §8.3 recommendation.

**Test results:** 635 lib tests pass (628 baseline + 4 Hessian FD-check tests +
2 analytic-vs-FD agreement tests + 1 constant-velocity test). Analytic and FD
agree to ~1e-2 (FD's 1e-3 truncation error dominates). Constant-velocity test
hits machine-zero with analytic vs `1e-3` with FD — confirming the accuracy gain.

---

## 2026-04-29 — feat(traj): JSON firmware export adapter

**What:** Added a `FirmwareAdapter` trait and a concrete `JsonAdapter` under
`linkage-sim-rs/src/gui/export/firmware/`. The adapter consumes a
post-`compute_trajectory` `SweepData` plus the active `ControlTarget`,
`Trajectory`, and input-parameter unit label and emits a versioned JSON
envelope (`schema_version = "linkage-traj-firmware-v1"`) with metadata
(`target_kind`, `target_units`, `input_units`, `n_samples`,
`duration_seconds`) and an array of per-sample objects (`t`, `target`,
`achieved`, `residual`, `u`, `u_dot`, `u_ddot`, `f_actuator_n`, `status`).

NaN/non-finite actuator forces serialize as JSON `null` via
`Option::filter(|x| x.is_finite())` so consumers see a clean sentinel
instead of an invalid `NaN` token. Solver failure modes carry diagnostic
detail in the `status` string (`"Reachability: target=… clamped=…"`,
`"Singularity: dg_du=…"`, `"BranchJump: norm=…"`,
`"NonConvergent: iter=… residual=…"`).

A new "Export firmware (JSON)..." entry appears in the File menu only when
the cached sweep is in `SweepMode::Trajectory`. The wrapper picks
`input_units` from `state.driver_kind` (revolute → `"rad"`, linear →
`"m"`, none → `"rad"` fallback) and surfaces failures via `error_log` /
`show_error_panel`; success sets a transient status message.

**Why:** Resolves the v3 firmware adapter open question in
`04-memory.yaml`. JSON is the v1 universal target — easy to consume from
any language, schema-versioned for forward compatibility. G-code, Aerotech
AeroBasic, Beckhoff TwinCAT NC PTP, and Galil DMC are planned ~150-LoC
follow-ups behind the same `FirmwareAdapter` trait, implemented on demand
once a hardware target is selected.

**Touched files:**
- `linkage-sim-rs/src/gui/export/firmware/mod.rs` — new: `FirmwareAdapter` trait + re-exports
- `linkage-sim-rs/src/gui/export/firmware/json.rs` — new: `JsonAdapter`, `FirmwareJsonEnvelope`, `FirmwareJsonSample`, schema constant + 3 unit tests
- `linkage-sim-rs/src/gui/export/mod.rs` — `pub mod firmware;`
- `linkage-sim-rs/src/gui/menu_bar.rs` — "Export firmware (JSON)..." menu entry + `export_firmware_json` helper
- `docs/ai/04-memory.yaml` — closed v3 firmware adapter open question
- `docs/ai/05-update-tracker.md` — this entry

**Test results:** 628 lib tests pass (625 baseline + 3 firmware adapter
tests: round-trip parse, NaN-actuator-force filter, empty-data rejection).
Build clean for `--features native`. Pre-existing `dirs`-crate errors in
non-native builds are unrelated to this change.

---

## 2026-04-29 — feat(traj): keyframe trajectory input + CSV import

**What:** Added a `Trajectory` enum with two variants — `Profile` (existing
analytic motion profile: ConstantSpeed / Trapezoidal / SCurve) and
`KeyframeTable` (user-defined `(t, h)` waypoints with linear interpolation) —
and migrated `SweepMode::Trajectory.profile: TrajectoryProfile` to
`SweepMode::Trajectory.trajectory: Trajectory`. Added a CSV import path so
`(t, h)` series authored externally (e.g. from a hardware capture or
spreadsheet) can be loaded as keyframes.

`KeyframeTrajectory::evaluate(t)` returns `(h, ḣ, ḧ)` where:
  - `h` is the linear interpolation between the bracketing waypoints.
  - `ḣ` is the segment slope `(h₁ - h₀) / (t₁ - t₀)`.
  - `ḧ` is `0` everywhere (piecewise-linear has zero curvature on each
    segment; the velocity step at waypoint boundaries is small at typical
    trajectory sample rates and the inverse-dynamics `ü` solve falls back to
    its FD path anyway).
Outside the table (`t < first` or `t > last`), evaluate clamps to the edge
value with `ḣ = 0`.

The trajectory_panel UI gains a top-level "Trajectory kind" dropdown
(Profile / Keyframes); switching kinds substitutes a sensible default for
the new variant. The Keyframes editor shows a row-per-waypoint table with
inline `t`/`h` drag-edits, an X button to remove rows (minimum 2 retained),
"+ Add waypoint", "Sort by t", and (native only) "Import CSV...". The
inline preview plot renders the current `h(t)` curve and overlays
waypoint markers when in Keyframes mode.

CSV format: 2 columns `t_seconds, target_value`. The first row is treated
as a header if its first cell fails to parse as a float. Blank lines and
`#`-prefixed lines are skipped. Waypoints are sorted by `t` ascending.

**Why:** Closes the long-standing "Trajectory v2: keyframe / waypoint
trajectory input" and "Trajectory v2: CSV-table trajectory import" open
questions in `04-memory.yaml`. Enables non-canonical trajectory shapes
(replays of recorded motion, hand-authored bring-up sequences, profiles
that don't fit ConstantSpeed / Trapezoidal / SCurve).

**Touched files:**
- `linkage-sim-rs/src/gui/state/types.rs` — `Trajectory` enum, `KeyframeTrajectory` struct + 3 unit tests
- `linkage-sim-rs/src/gui/state/mod.rs` — re-export `Trajectory`, `KeyframeTrajectory`
- `linkage-sim-rs/src/gui/sweep/mod.rs` — `SweepMode::Trajectory.profile` → `.trajectory`; `compute_trajectory` parameter type → `&Trajectory`; integration test updated
- `linkage-sim-rs/src/gui/state/blueprint_ops.rs` — destructure / call site updates
- `linkage-sim-rs/src/gui/mod.rs` — toolbar default constructor wraps in `Trajectory::Profile(...)`
- `linkage-sim-rs/src/gui/plot_panel/trajectory.rs` — duration / evaluate via enum
- `linkage-sim-rs/src/gui/export/csv.rs` — duration via enum + test fixture
- `linkage-sim-rs/src/gui/trajectory_panel/profile_input.rs` — full rewrite: two-level kind dropdown + analytic editor + keyframe editor + CSV import + parse_keyframes_csv test

**Test results:** 625 lib tests pass (621 baseline + 3 keyframe interp tests
+ 1 CSV parser test). Build clean for both `--features native` and the
default (WASM-friendly) profile; no new warnings beyond the pre-existing
baseline.

**Concerns / known limitations:**
- WASM CSV import is not wired up (rfd's WASM async API would need a
  separate code path; defer until a browser user actually asks for it).
  The button is `#[cfg(feature = "native")]`-gated, so the WASM build
  simply doesn't render it.
- `ḧ` reports `0` for keyframe trajectories. The inverse-dynamics path
  uses FD anyway for non-Angle targets, so this is a non-issue in
  practice; the `h_ddot` value flows into `compute_trajectory` only as a
  hint for the FD step direction (which the solver then refines).
- Switching trajectory kinds in the UI discards the prior variant's
  parameters (analytic ↔ keyframes). This is intentional — there's no
  obvious mapping between "ConstantSpeed start=0 end=1 dur=1" and a
  3-waypoint keyframe table — and matches the established switch-resets
  pattern of the shape dropdown.

---

## 2026-04-29 — feat(traj): add SCurve (jerk-limited) motion profile

**What:** Added a third variant `MotionProfile::SCurve { jerk_fraction: f64 }`
to the `MotionProfile` enum. v1 implementation uses the pure quintic
ease-in-out polynomial `s(τ) = τ³(10 − 15τ + 6τ²)` for trajectory-mode
position, so velocity and acceleration are both zero at `t=0` and
`t=duration` (the defining property of a jerk-limited profile). The
`jerk_fraction` field is reserved for a future full 7-segment formal
version and is currently unused.

In **trajectory mode**, `TrajectoryProfile::evaluate` dispatches to a new
`scurve_value(t, duration, start, end)` helper next to the existing
`trapezoidal_value` in `gui/state/types.rs`. The profile-editor UI in
`gui/trajectory_panel/profile_input.rs` exposes "SCurve" alongside
"ConstantSpeed" and "Trapezoidal" in the shape dropdown.

In **driver-side sweep mode**, SCurve falls back to `ConstantSpeed`
behaviour (sets `profile_torques / profile_omega / profile_alpha` to
`None`). The jerk-limited evaluation is meaningful only for trajectory
mode where the `(h, ḣ, ḧ)` tuple feeds the inverse-kinematics solve.
The driver-side selector in `gui/input_panel.rs` does not surface
SCurve as a selectable option but maps it to the "Constant Speed" label
when the trajectory-mode UI has switched it on.

**Why:** Closes the long-standing "Trajectory v2: S-curve" open question
in `04-memory.yaml`. Real actuator hardware uses jerk-limited profiles
to avoid mechanical shocks at start/stop; v1 quintic captures the
endpoint property without the implementation cost of the full
7-segment piecewise jerk profile.

**Touched files:**
- `linkage-sim-rs/src/gui/state/mod.rs` — new `SCurve { jerk_fraction }` variant
- `linkage-sim-rs/src/gui/state/types.rs` — `scurve_value` + dispatch + 2 tests
- `linkage-sim-rs/src/gui/trajectory_panel/profile_input.rs` — dropdown + ctor
- `linkage-sim-rs/src/gui/sweep/motion_profile.rs` — driver-side fallback to const-speed
- `linkage-sim-rs/src/gui/input_panel.rs` — exhaustive match (SCurve labelled "Constant Speed")

**Test results:** 621 lib tests pass (619 baseline + 2 new SCurve tests:
`trajectory_profile_scurve_endpoints_match_target`,
`trajectory_profile_scurve_integrates_to_span`). Build clean (no new
warnings beyond the pre-existing baseline).

---

## 2026-04-29 — R1b refactor: collapse driver scalars into DriverKind enum payload

**What:** Removed the dual-purpose `driver_omega`, `driver_theta_0`, and
`driver_stroke` scalar fields from `AppState`. Their values now live as
payload on the `DriverKind` enum:

- `DriverKind::Revolute { angle, omega, theta_0 }`
- `DriverKind::Linear { stroke, velocity, length_0 }`
- `DriverKind::None` (unit, unchanged)

Added `driver_omega() / driver_theta_0() / driver_stroke()` getters and
`set_driver_omega() / set_driver_theta_0() / set_driver_stroke()` setters
on `AppState` so flat call sites stay readable; dispatch points in
`step_animation`, `compute_sweep`, and trajectory mode pattern-match the
variant directly. Setters on the wrong variant (e.g. setting stroke on
Revolute) are silent no-ops.

Equality / pattern-match call sites updated from
`state.driver_kind == DriverKind::Linear` to
`matches!(state.driver_kind, DriverKind::Linear { .. })`.
`Eq` derive removed from `DriverKind` (f64 payload has no `Eq`); `Copy`
preserved (all-f64 payload is Copy).

**Why:** The dual-purpose scalars were the implicit-overloading remnant
flagged as R1b in `04-memory.yaml/open_questions`. Collapsing onto the
enum makes per-driver-kind semantics type-checked rather than
documented-by-convention.

**Touched call sites:** 14 files, +283 / -155 lines.

**Test results:** 619 lib tests pass (no count change vs.
`c905f28`/baseline). Release lib + `linkage-gui` binary both compile
clean.

**Subtleties handled:**
- `load_sample()`, `reassign_driver()`, `set_constant_speed_driver()`,
  `convert_actuator_to_linear_driver()`, `restore_snapshot()`, and
  blueprint `rebuild()` reorder writes to construct the kind variant
  fully before any setter call — setters are no-ops on the wrong
  variant, so write-then-transition was the failure mode to guard
  against.
- The `MechanismSnapshot` DTO in `gui/undo.rs` retains its dual-purpose
  scalar shape (it's a serialization format, not state) — `take_snapshot`
  / `restore_snapshot` translate between the DTO and the variant
  payload via the new accessors and a fresh blueprint-driven kind
  reconstruction on restore.
- Linear-driver stroke preservation across rebuilds: the old code
  preserved a finite `driver_stroke` field across `rebuild()`. The new
  blueprint_ops pattern-matches the prior `DriverKind::Linear { stroke,
  .. }` and reuses it; resets to `length_0` on Linear→Linear with NaN
  stroke or any non-Linear→Linear transition.

---

## 2026-04-30 — Trajectory-mode position control: Stage 4 (polish + CSV + docs) — feature shipped

**What:**
- CSV export wired for trajectory mode in `gui/export/csv.rs::write_trajectory_csv`. v1 superset column layout: `t_seconds, target_value, achieved_value, residual, u, u_dot, u_ddot, F_actuator_N, driver_torque_Nm, status`. Status text format like `Reachability:0.0920->0.0870`, `Singularity:1.5e-7`, etc.
- "From joint" helper in target picker (`gui/trajectory_panel/target_picker.rs::draw_point_input_with_joint_helper`) — populates body-local point coords from existing joints on the chosen body.
- Failure-band hover tooltips: rendered as a `ui.collapsing("⚠ N failure(s)")` summary below the trajectory plot, with one line per failed sample showing `t={:.3}s: <status payload>`. Cleaner than per-band tooltips per egui_plot 0.33 limitations.
- `docs/FEATURES.md` — user-facing entry under "Trajectory-mode position control" with use case, how-to, failure handling, math reference cross-link.
- `docs/ai/04-memory.yaml` — 9 deferred items added to `open_questions`: SCurve, click-to-set canvas, keyframe input, CSV-table import, firmware adapter, analytic acceleration, sweep_mode persistence, linear-driver dispatch, u_range frame conversion.

**Why:** Closes the trajectory-mode feature. v1 ships with full forward + inverse pipeline, GUI activation, CSV export, plot rendering, and click-to-scrub. Polish items deferred to follow-up tasks per the spec.

**Test results:** 619 lib tests pass (+2 from Stage 3 baseline; +9 cumulative across Stage 4: 2 CSV + 7 from earlier Stage 4 substages).

**Commits in Stage 4:**
- 0ce97af — CSV export
- 28028eb — From-joint helper
- bab31f0 — Failure-band tooltip summary
- f325484 — User-facing FEATURES.md + deferred items in 04-memory.yaml
- (plus this final wrap entry)

**Known limitations / follow-ups (tracked in 04-memory.yaml):**
- sweep_mode/trajectory_severity not persisted through save/load
- Linear-driver trajectory dispatch is TODO
- u_range display-frame vs body-frame conversion (subtle bug; needs verification on a DXF-imported mechanism)
- F_actuator_N column always NaN until compute_trajectory wires actuator force computation
- v2 features (SCurve, keyframes, CSV import) deferred until users ask

---

## 2026-04-30 — Trajectory-mode position control: Stage 3 (UI surface) shipped

**What:** Stage 3 surfaces Stage 2's `compute_trajectory` backend through the
GUI. New `Trajectory` option in the SweepMode dropdown switches the input
panel to a trajectory-specific UI (target picker + profile editor + severity
toggle), and the plot panel renders a time-axis view with click-to-scrub
that drives the canvas mechanism through the back-solved trajectory.

- **`gui/state/trajectory_ops.rs`** — new `AppState::solve_for_trajectory_target(target, h)`
  sibling to `solve_at_angle` / `solve_at_stroke`. Calls `solve_for_target`
  in `Severity::Analysis` and updates `self.q` on success, leaving state
  unchanged on failure. Used by click-to-scrub.
- **`gui/trajectory_panel/`** (3 files):
  - `mod.rs` — top-level dispatcher. Three `CollapsingHeader` sections:
    Target observable → Profile → Solve options (severity radio toggle:
    Analysis vs Strict).
  - `target_picker.rs` — `ControlTarget` variant picker (5 variants:
    `Angle`, `WorldX`, `WorldY`, `Distance`, `Projection`) with the
    relevant body / point / axis fields per variant + live `g(q)` readout
    on the current pose so the user can see where they are before
    choosing the target value.
  - `profile_input.rs` — `TrajectoryProfile` editor: shape selector
    (currently trapezoidal), `start` / `end` / `duration` /
    `accel_frac` / `decel_frac` fields, `n_samples` slider, and an
    inline `h(t)` preview plot.
- **`SweepMode` dropdown extended** in `gui/input_panel.rs` — `Angle` /
  `Stroke` / `Trajectory` options. Selecting `Trajectory` constructs a
  default `SweepMode::Trajectory { target: Angle{driver}, profile:
  default, severity: Analysis, n_samples: 64 }` and routes the input
  panel to `trajectory_panel::draw` instead of the angle/stroke
  controls.
- **`AppState` new fields** (`gui/state/mod.rs`): `sweep_mode:
  SweepMode` (replaces ad-hoc `is_stroke_sweep` checks), `trajectory_severity:
  Severity`. Both default to `Angle` / `Analysis`.
- **`gui/state/blueprint_ops.rs::compute_sweep`** — branches on
  `state.sweep_mode`. `SweepMode::Trajectory` dispatches to the
  Stage 2 `compute_trajectory` for revolute drivers (the integration
  deferred from Stage 2). Linear-driver path is TODO and falls back
  to the existing `compute_sweep_data` so users still get a sweep
  curve.
- **`gui/plot_panel/trajectory.rs`** (new) — time-axis plot rendering.
  Main plot stacks target(t) (orange) + achieved(t) (cyan) on the
  primary axis; residual(t) and u(t) are stacked in separate panels.
  Failure-band overlays paint regions where
  `inverse_solve_status != Converged`. Click-to-scrub on the main
  plot calls `state.solve_for_trajectory_target(&target, h)` to
  drive the canvas mechanism to the clicked sample's pose.
- **`gui/plot_panel/mod.rs`** — early-return dispatch when
  `sweep_mode.is_trajectory()`: `trajectory::render(state, ui)`
  bypasses the forward-sweep tabs (the trajectory X-axis is time, not
  driver angle, so the existing tabs don't apply).
- **Minor refactor:** `empty_trajectory_sweep_data` promoted to
  `pub(crate)` so `compute_sweep` in blueprint_ops can construct an
  empty `SweepData` shell before calling `compute_trajectory` to fill
  it.

**Why:** Stages 1 and 2 shipped the math and the per-sample analysis
loop, but the trajectory-mode pipeline was unreachable from the UI.
Stage 3 wires it through: dropdown → input panel → compute → plot →
click-to-scrub. The user can now define a desired output trace, see
the back-solved actuator input alongside it, and scrub the canvas
mechanism through the trajectory by clicking the time-axis plot.

**Test count:** 617 lib tests pass (unchanged from Stage 2). UI is
integration-only — existing solver tests and `TrajectoryProfile` /
`SweepData` unit tests cover the math; manual GUI verification is
deferred to user.

**Open follow-ups:**
- Persist `sweep_mode` and `trajectory_severity` through save/load.
  Both fields are in-memory only at the moment, so reopening a
  document drops the trajectory configuration.
- Linear-driver dispatch in `compute_sweep`. Currently the
  `SweepMode::Trajectory` branch checks `driver_kind == Revolute`
  and falls through to the angle/stroke `compute_sweep_data` path
  for linear-driver mechanisms.
- Display-frame vs body-frame conversion of `u_range` in the
  trajectory dispatch. The forward-sweep path subtracts
  `driver_display_offset` to get body-frame θ; the trajectory
  dispatch currently does not. Confirm correct frame.
- Per-frame body velocity / acceleration readouts. Stage 1 noted
  that `solve_velocity` / `solve_acceleration` are not currently
  called per-frame; trajectory mode might benefit from them for
  diagnostics.

---

## 2026-04-30 — Trajectory-mode position control: Stage 2 (sweep extension) shipped

**What:** Stage 2 wires the Stage 1 inverse-kinematics solver into the
sweep pipeline. Adds a `compute_trajectory` sibling to `compute_sweep_data`
that, given a `TrajectoryProfile` describing the desired output trace
`h(t)`, back-solves the actuator input `u(t)` per sample and runs the
existing per-sample force/energy/reaction analyses unchanged.

- **`TrajectoryProfile` struct** (`gui/state/types.rs`) — composes the
  existing `MotionProfile` shape (trapezoidal velocity ramp) with absolute
  units (`start`, `end`, `duration`). Implements
  `evaluate(t) -> (h, h_dot, h_ddot)` and `sample_times(n)` for uniform
  sampling across `[0, duration]`.
- **`SweepData` trajectory time-series fields** (`gui/sweep/mod.rs`) —
  6 new `Option<Vec<f64>>` fields (`target_values`, `achieved_values`,
  `tracking_residual`, `u_values`, `u_dot_values`, `u_ddot_values`)
  plus `Option<Vec<InverseSolveStatus>>` for diagnostics. All carry
  `#[serde(default, skip_serializing_if = "Option::is_none")]` so existing
  saved snapshots load cleanly. SweepData / SweepMode / InverseSolveStatus
  gain Serialize/Deserialize derives.
- **`SweepMode::Trajectory { target, profile, severity, n_samples }`
  variant** joining `Angle` and `Stroke`. `is_trajectory()` accessor
  added; existing `is_stroke()` consumers in plot panels unchanged.
- **`compute_trajectory` per-sample loop** (`gui/sweep/mod.rs`) — for each
  sample time `t_k`: evaluates the profile, calls `solve_for_target` to
  back-solve `u_k`, computes `u_dot_k` / `u_ddot_k` via
  `inverse_velocity` / `inverse_acceleration_fd`, then invokes the same
  `solve_statics` / `solve_inverse_dynamics` /
  `compute_energy_state_mech` calls used by `compute_sweep_data`. The
  driver row of `Φ_t` and `γ` is overridden with the back-solved
  `u_dot_k` / `u_ddot_k` (two lines per sample) so the constant-speed
  constraint code in `core/` remains untouched. On
  `Severity::Analysis` failures the output is re-evaluated from the
  partial `q` so time-series channels remain length-matched.
- **Refactor fold-in:** `Mechanism::driver_row()` helper centralizes the
  `n_constraints() - 1` driver-last-row assumption (Task 1.10 review
  item, applied at the Φ_t / γ override site).
- **Test count:** +1 integration test
  (`compute_trajectory_populates_all_trajectory_fields`) verifying field
  population, target tracking, and `Converged` status across all samples.
  Total lib tests: 617 pass (was 614 after Stage 1; +3 across Stage 2
  tasks).
- **Stage 2 GUI activation: NONE.** `compute_trajectory` is reachable
  from code and tests but not from the UI. `compute_sweep_data` does
  not yet dispatch to it. Stage 3 introduces `state.sweep_mode` and
  wires `AppState::compute_sweep` to branch on the variant.

**Why:** Stage 1 shipped the inverse solver in isolation. Stage 2
threads it through the per-sample analysis pipeline so a full trajectory
produces the same plottable channels as a forward sweep (joint
reactions, driver torque, energies, etc.) — without ripping up the
constant-speed-parameterized constraint code. The Φ_t / γ driver-row
override is the minimal seam.

**Migration note:** No GUI changes shipped in Stage 2; users see no
difference yet. Stage 3 adds the trajectory-mode UI surface (target
picker, profile editor, severity toggle, mode switcher) and dispatches
`AppState::compute_sweep` to the right backend.

**Test results:** 617 lib tests pass (+1 new
`compute_trajectory_populates_all_trajectory_fields`).

---

## 2026-04-30 — Trajectory-mode position control: Stage 1 (solver layer) shipped

**What:** New `solver/inverse_kinematics/` subtree (6 files) implements the
inverse-kinematics solver for trajectory-mode position control. Given a
desired output observable `g(q)` and target value `h`, back-solves the
actuator input parameter `u` such that `g(q(u)) = h`, where `q(u)` is the
forward solution from the existing kinematics solvers.

- **Module layout** (`linkage-sim-rs/src/solver/inverse_kinematics/`):
  - `mod.rs` — public surface (re-exports of types + functions).
  - `severity.rs` — `Severity::{Strict, Analysis}` + `InverseSolveStatus`
    (5 variants: `Converged`, `Reachability`, `Singularity`, `BranchJump`,
    `NonConvergent`).
  - `control_target.rs` — `ControlTarget` enum with 5 variants
    (`Angle`, `WorldX`, `WorldY`, `Distance`, `Projection`); constructors
    validate against ground.
  - `solver.rs` — `workspace_probe` + `solve_for_target` outer Newton loop;
    full failure-mode detection (reachability, singularity, branch-jump,
    non-convergence).
  - `derivatives.rs` — `inverse_velocity` (closed-form) +
    `inverse_acceleration_fd` (finite difference); shared `compute_dq_du`
    helper.
  - `test_helpers.rs` — `#[cfg(test)]` 4-bar fixture + warm-start q.
- **Severity model** — `Severity::Strict` returns `Err(LinkageError::...)`
  on any failure; `Severity::Analysis` returns
  `Ok(InverseSolveResult { status: <failure variant>, ... })` so the GUI
  can render failed samples in red rather than aborting the trajectory.
  Same Newton/FD math runs in both modes; only the return type differs
  (`classify_or_fail` helper centralizes the dispatch).
- **New `LinkageError` trajectory variants** (`error.rs`):
  `TrajectoryUnreachable`, `TrajectorySingular`, `TrajectoryBranchJump`,
  `TrajectoryNonConvergent`. `From<InverseSolveStatus> for LinkageError`
  bridges the two enums.
- **Refactor in `core/`** — new `Mechanism::driver_row()` helper centralizes
  the `n_constraints() - 1` assumption that the driver is the last
  constraint row.
- **Test count:** ~32 new tests in inverse_kinematics module + 1 new error
  test. Total lib tests: 614 pass (was 613, +1 singularity-detection test
  added by Task 1.14).
- **Stage 1 GUI integration:** NONE. Solver works in isolation. Stage 2
  starts with sweep extension (`gui/sweep/mod.rs::compute_trajectory`),
  which will override Φ_t / γ on the driver row using back-solved
  `u_dot` / `u_ddot`.

**Why:** User wants position control via trajectory — specify a desired
output (e.g. coupler-point world Y vs time) and back-solve the required
crank/stroke trajectory. Existing solvers go forward (driver → output);
this layer goes inverse (target output → required driver input).

**Migration note:** No GUI changes shipped in Stage 1; users see no
difference yet. Stage 2 will wire the solver into `compute_trajectory`,
and Stage 3 will add the trajectory-mode UI.

**Test results:** 614 lib tests pass (+1 new
`singularity_detection_near_extremum`). Pre-existing linear_driver
doctest still fails (not related). BranchJump test deferred to
stage-1.5 — engineering the failure mode requires constructing a 4-bar
near a toggle, which is a separate spec; TODO comment in place.

---

## 2026-04-24 — LinearDriver GUI feature: stroke-driven sweeps

**What:** Wired the dormant LinearDriver pipeline end-to-end so a user
can convert a `LinearActuator` force element into a kinematic
`LinearDriver` constraint and run stroke-driven analysis. Six
preparatory refactors plus the conversion UI:

- **R1 — `DriverKind` discriminant.** New enum on `AppState`
  (`None` / `Revolute` / `Linear`). Documents the per-variant meaning
  of the existing `driver_omega` / `driver_theta_0` / `driver_stroke`
  scalars (rad+rad/s vs m+m/s) and unblocks dispatch without ripping
  out 150+ usages of the four scalars. Follow-up R1b (deferred) can
  collapse them into payload on the variants.
- **R2 — `rebuild()` reads `linear_drivers`.** Sets `driver_kind =
  Linear` when the blueprint has any linear driver, populating
  velocity/length_0 and seeding `driver_stroke`. Falls back to
  Revolute if `bp.drivers` is non-empty, otherwise None.
  `load_from_json_str` applies the same detection.
- **R3 — `step_animation` dispatch.** Matches `driver_kind` and
  delegates to `step_animation_revolute` (existing) or new
  `step_animation_linear` (mirrors revolute but moves stroke in
  metres). The animation slider's "deg/s" doubles as "mm/s" in
  Linear mode for a consistent feel.
- **R4 — `SweepMode::Stroke`.** `compute_sweep_data` detects via
  `mech.n_linear_drivers()` and iterates 1 mm steps; default range
  is `length_0 ± 100 mm`. `compute_sweep` guard counts linear
  drivers; sweep-range field is interpreted as mm in Linear mode and
  converted to metres for the solver. Plot dispatcher's offset shift
  is a no-op for stroke sweeps; click-to-scrub branches between
  `solve_at_angle` and `solve_at_stroke`.
- **R5 — Driver-section UI dispatch.** `draw_input_panel` shows the
  Crank Angle section for Revolute/None, the new Actuator Stroke
  section for Linear (mm slider + mm sweep-range DragValues). The
  `sweep_angle_min/max_deg` field is repurposed for stroke (mm) when
  in Linear mode — only one driver mode is active at a time.
- **R6 — Display-offset gate.** `compute_driver_display_offset`
  returns 0 when `driver_kind != Revolute`; the angle-frame offset
  has no meaning for stroke values. Plot dispatcher's
  `sweep_in_display_frame` is also a no-op for stroke sweeps.

- **Conversion UI.** `LinearActuator` force editor gets a `Set as
  Linear Driver` button. `AppState::convert_actuator_to_linear_driver`
  removes any revolute drivers, strips the actuator force element,
  and adds a `LinearDriverJson` with `length_0 = current stroke` so
  the pose doesn't jump on takeover. Default velocity 10 mm/s.

**Why:** User wanted stroke-driven analysis for press mechanisms.
`LinearActuator` was a force element (apply force, find equilibrium),
which is the wrong abstraction for "sweep stroke and read out
required force". `LinearDriver` is the right one but had no GUI entry
point and dormant solver wiring.

**Test results:** 578 lib tests pass (+3 new:
`sweep_stroke_mode_for_linear_driver`,
`sweep_stroke_mode_with_explicit_range`,
`convert_actuator_to_linear_driver_switches_mode`).

**Migration note:** Existing files with revolute drivers continue to
work unchanged. Sample mechanisms (FourBar, ChebyshevLambdaActuator
with its LinearActuator force, etc.) all load in Revolute mode by
default; user opts into Linear mode via the Set as Linear Driver
button.

---

## 2026-04-24 — Sweep range stored in display frame (follow-up)

**What:** The initial crank-angle-offset patch shifted the Crank Angle
slider to display frame but kept `sweep_angle_min_deg` /
`sweep_angle_max_deg` as body-frame values, then added/subtracted the
offset at the UI edges. That produced a jarring slider range (e.g.
`41°..61°` for a DXF-imported crank with α=41°) and sweep-range
DragValues that couldn't accept `0` as min.

Semantic cutover: `sweep_angle_min_deg` and `sweep_angle_max_deg` are
now stored in DISPLAY frame directly.

- Slider range: clean `[0, 360]` when sweep range is off; exactly the
  user's `[min, max]` when on. No more offset-shifted ranges.
- Sweep-range DragValues accept `0..=720` display degrees, matching
  the Crank Angle slider they sit below.
- `compute_sweep` subtracts the offset to produce body-frame
  `(min, max)` for `compute_sweep_data`.
- `step_animation_revolute` subtracts the offset for the ping-pong
  bounds when sweep range is on.
- `has_toggles_in_range` (health panel) subtracts the offset for the
  body-frame comparison.

**Schema note:** `_sweep_angle_min` / `_sweep_angle_max` in share URLs
are now display-frame (matches the `_animation_speed_deg_per_sec`
semantic). Old URLs load with body-frame values that get interpreted
as display — the sweep range visually shifts by α on first load. This
is a one-time migration side-effect; subsequent saves are clean.

**Test results:** 575 lib tests pass unchanged.

---

## 2026-04-24 — Crank angle display matches visible bar orientation

**What:**
- New `AppState::driver_display_offset` (rad), recomputed on every
  `rebuild` / `load_sample` / `load_from_json_str` / `reassign_driver`.
  It equals the angle from the driver body's grounded-pivot local
  coord to its farthest other attachment point, i.e. the local A→B
  direction. Zero for sample-built mechanisms (A at (0,0), B at
  (len, 0)); nonzero for DXF imports whose sketch wasn't drawn along
  +X. (`src/gui/state/mod.rs` + rebuild / load / driver paths.)
- Every user-visible angle surface now shows `body_frame_θ + offset`:
  Crank Angle slider (`src/gui/input_panel.rs`), sweep-range min/max
  DragValues (same file), plot X-axes + current-angle cursor +
  click-to-scrub (`src/gui/plot_panel/mod.rs`), and the canvas
  crank-angle indicator arc (`src/gui/canvas/rendering/mod.rs`). The
  solver, sweep data, JSON schema, and share URLs keep body-frame θ
  as the source of truth — the offset is applied only at the display
  boundary.
- Plot dispatcher clones `SweepData` per frame with shifted
  `angles_deg` and `toggle_angles`; every individual plot function
  remained unchanged. The clone is cheap relative to the redraw and
  keeps the 20+ plot call sites consistent without modification.

**Why:** User reported the Crank Angle slider showed near 0° while the
visible link sat ~15° above horizontal (and 53° looked almost vertical
instead of at 90°). Diagnosis: the slider showed the driver body's
internal θ. For sample-built bodies that equals the visible bar
direction (because local A→B lies on +X), but DXF imports preserve
world coords as local coords so the A→B direction has an arbitrary
offset α. Display now reads `θ + α` so the slider tracks the
orientation the user actually sees on canvas.

**Test results:** 575 lib tests pass (+2 new:
`driver_display_offset_zero_for_sample_builders`,
`driver_display_offset_picks_up_rotated_crank`).

---

## 2026-04-24 — Force zone application point: visualize + draggable override

**What:**
- `ForceZoneElement` gets a new optional field
  `body_local_app_point: Option<[f64; 2]>`. When `None`, the solver
  uses the overlap centroid as before. When `Some`, the force applies
  at the pinned body-local point, letting the user model contact at a
  specific location rather than an area-averaged centroid.
  (`src/forces/elements/element_types.rs`,
  `src/forces/elements/evaluation.rs`.)
- Canvas now renders a crosshair + "F" label at the active application
  point — faint yellow for the auto-centroid, brighter orange for a
  pinned override (labelled "F (locked)"). Uses a shared helper
  `force_zone_app_point_world` in
  `src/gui/canvas/rendering/force_render.rs`.
- Left-drag on the marker in Select mode enters an app-point drag. A
  ghost crosshair follows the pointer; on release the pointer world
  position is converted to the target body's local frame and written
  to the override, committing via `update_force_element`. Escape
  cancels the drag mid-gesture.
- Force editor for `ForceZone` grows an "Application point" section
  with a `Lock to body-local point` checkbox, Local X/Y DragValues
  (mm), and a `Reset to auto (overlap centroid)` button.

**Why:** User wanted to see where the force is actually applied and
pin it to the real contact point. Overlap centroid is correct for
distributed contact but wrong for a specific contact pad — the
override lets the user enforce the physical contact location without
abandoning the zone's "trigger on contact" semantics.

**Schema compatibility:** `serde(default, skip_serializing_if =
"Option::is_none")` — old files load cleanly with
`body_local_app_point = None`, and files saved without an override
don't grow the field.

**Test results:** 573 lib tests pass unchanged. JSON round-trip and
sample-builders updated to include the new field explicitly.

---

## 2026-04-24 — Delete body also strips referencing forces + diagnostics relocated

**What:**
- `AppState::remove_body` (`src/gui/state/entity_crud.rs`) now also
  retains-out force elements whose `attached_body_ids()` reference the
  deleted body. Previously a LinearActuator (or any spring/damper/etc.)
  attached to the deleted link stayed in the blueprint, and the next
  rebuild tried to resolve its attachment points on a missing body,
  freezing the GUI. New regression test
  `remove_body_cascades_to_force_elements` loads the
  ParallelogramActuator sample (which has a crank-attached linear
  actuator), removes the crank, and asserts no force still references
  it.
- Moved the debug `dt / fps / step` readout from the top toolbar
  (where it wrapped off-screen on narrow viewports) to the bottom
  status bar. Still gated on the View → Debug Overlay toggle, but
  always visible next to the angle / torque / DOF readouts.

**Why:** User reported "Cannot delete link 1, it's the last link but
has an actuator attached at the end joint, tool freezes, delete
actuator with it if needed" — which is the cascade gap described
above. Also reported they couldn't see the diagnostics line with
Debug Overlay on, so it's been relocated to a guaranteed-visible
spot.

**Test results:** 573 lib tests pass (+1 new cascade test).

---

## 2026-04-24 — Context menu Set-Driver submenu + frame-time diagnostics

**What:**
- Right-click context menus on joints and attachment points now offer
  a `Set Driver to \u{2026}` submenu whenever *any* grounded revolute
  exists in the mechanism, not only when the clicked element is itself
  the grounded one. Previously users right-clicked a non-grounded
  joint, saw `Set as Driver` disabled with only a hover tooltip, and
  reported it as "grayed out, can't add a driver". The submenu lists
  every grounded revolute with the current driver labeled `(current)`,
  so the action is always reachable from any joint.
  (`src/gui/canvas/context_menu.rs`.)
- Top toolbar shows a compact `dt=16.7ms  fps=60  step=0.083°/frame`
  diagnostics line next to the speed slider when the Debug Overlay is
  enabled (View menu). Intended to narrow down the user's "no motion
  under 15 °/s" report on the WASM build — if the reported dt or step
  is out of whack, the issue is in the browser/egui frame pipeline, not
  the animation math.

**Why:** User said "I couldn't add a driver because the selection was
grayed out before". The previous UX required right-clicking the right
*specific* element; the submenu decouples discovery from precision.
Frame-time readout gives us an observable signal to diagnose the
low-speed animation-freeze complaint without guesswork.

**Test results:** 572 lib tests pass unchanged.

---

## 2026-04-24 — Link Editor delete button + Driver panel picker

**What:**
- `Delete` button in the Link Editor header row
  (`src/gui/property_panel/mod.rs`). Shown next to the body-label edit
  field for any non-ground body. Red text, hover text
  "Delete this link and all joints connected to it". Wired through a
  new `PendingPropertyEdit::DeleteBody` variant that calls
  `state.remove_body` and clears `selected` + `link_editor_body`.
- Driver section now shows a `ComboBox` picker when the mechanism has
  no driver but at least one grounded revolute joint exists
  (`src/gui/input_panel.rs` — new `draw_no_driver_picker`). Selecting
  a joint writes to `pending_driver_reassignment`, which the frame
  pipeline already consumes to call `reassign_driver`.
- When NO grounded revolute joint exists, the Driver section shows the
  same amber "Needs a ground pivot…" hint the right-click context
  menu uses, so the discoverability path is consistent across
  surfaces.

**Why:** User reported "We have no way to delete links…also can't add
driver even with ground points". Decoding their shared URL confirmed
two grounded revolute joints (J5, J6) and an empty drivers map — the
data was correct, but the only path to set a driver was the canvas
right-click menu, which the user wasn't discovering. A dedicated
picker in the Driver panel makes the action visible where the user
expects it. Same reasoning for Delete: the canvas right-click menu
had it, but the Link Editor (where users already look to edit a link)
didn't.

**Test results:** 572 lib tests pass unchanged.

---

## 2026-04-23 — Driver omega floor + disabled + Body ribbon button

**What:**
- New constant `MIN_DRIVER_OMEGA_ABS = 0.01` (rad/s ≈ 0.1 RPM) and helper
  `clamp_driver_omega` in `src/core/driver.rs`. Applied at three
  boundaries:
  1. `set_constant_speed_driver` (GUI Speed DragValue) — typing 0 RPM
     now snaps to 0.1 RPM instead of freezing the mechanism.
  2. `load_mechanism_unbuilt_from_json` — legacy files written with
     omega=0 are rescued at load time.
  3. `solve_at_angle` — belt-and-braces zero guard mirroring the other
     callsites (blueprint_ops, file_io, undo_ops).
- Top-ribbon "+ Body" button is now disabled with hover text redirecting
  users to the Link Editor ("To add a rigid body, use the Link Editor
  in the property panel..."). The Link Editor exposes mass, inertia,
  mount/coupler points, and geometry together, which is the desired
  canonical flow.

**Why:** User reported the animation would hang at low speeds. Root
cause was the driver closure capturing omega=0: `f(t) = theta_0 + 0*t`
freezes the constraint at `theta_0`, and `solve_at_angle` separately
divides by `driver_omega`. Every entry point that writes omega now
clamps above the floor. 0.01 rad/s ≈ 1 revolution per 10 minutes, so
anything slower is kinematically indistinguishable from a static
mechanism; users who truly want "paused" should use Play/Pause.

Separately, "+ Body" in the top ribbon was a shortcut whose flow
diverged from the Link Editor's full-featured body creation. Disabling
it pushes users to the canonical path.

**Test results:** 572 lib tests pass (+2 new:
`step_animation_advances_at_slow_speed` and
`set_constant_speed_driver_rejects_tiny_omega`).

---

## 2026-04-23 — Slower animation + share URL embeds speed and crank limits

**What:**
- Top-toolbar animation speed slider lower bound dropped from 10.0 to
  0.5 °/s (`src/gui/mod.rs:483`). The log scale previously compressed
  10–30 °/s into a sliver of screen, making the slider feel like it
  refused to slow the animation past ~15 °/s. Users can now genuinely
  crawl the kinematics for debugging.
- Share URLs now embed `_animation_speed_deg_per_sec` and the sweep
  range (`_sweep_range_enabled`, `_sweep_angle_min/_max`)
  unconditionally — previously the sweep range was only written when
  the toggle was on, so a recipient would land in 0..360 animation
  bounds even when the sender had a narrower working range configured.
  `driver_omega` rides along in the standard `drivers` JSON map (no
  dedicated override needed).
- `load_from_json_str` restores all four fields on the receiving side
  (`src/gui/state/file_io.rs`).

**Why:** User reported "anything less than 15 deg/s doesn't move" and
"URL sharing doesn't embed speed or crank-angle limits". First was a
slider floor, not an animation-math bug; second was a real
serialization gap.

**Test results:** Two new unit tests
(`share_url_round_trips_speed_and_crank_limits`,
`share_url_embeds_crank_limits_even_when_range_disabled`) verify the
round-trip including the "toggle off, limits still embedded" case. 568
lib tests pass.

---

## 2026-04-23 — Ground-pivot discoverability + DXF geometry error fix

**What:**
- Right-click context menu on a joint or attachment point now shows an
  inline amber tip ("Needs a ground pivot — use the + Ground tool...")
  directly under the disabled "Set as Driver" button whenever the
  mechanism has zero grounded revolute joints. Previously the hint was
  only visible by hovering the disabled button, which new users missed.
  Implemented in `gui/canvas/context_menu.rs` for both `show_joint_menu`
  and `show_attachment_menu`.
- DXF "Add Geometry to Link" now emits an accurate error toast when the
  chosen target body doesn't exist in the blueprint (e.g. synthetic
  compound-expansion bodies like `force_0_cyl` / `force_0_rod` which
  live on the `Mechanism` but not in `state.blueprint.bodies`).
  Previously the blueprint write silently no-op'd and the status toast
  falsely reported success. Fix in
  `gui/dxf_import.rs::convert_selected_to_rigid_geometry`.

**Why:** A user building a press mechanism from scratch couldn't find
their way to adding a driver — the mechanism had no ground pivots yet,
and the only hint was a hover tooltip on a disabled menu item. The
inline tip directs the user at the + Ground tool immediately.

Separately, while diagnosing the same report, the DXF geometry flow was
found to claim success even when the target body wasn't user-editable.
Surfacing the real error keeps the user from chasing a ghost.

**Test results:** `cargo check --bin linkage-gui` clean. Context-menu
change is render-only; DXF change is a defensive early-return that
triggers only when the blueprint lacks the target body.

---

## 2026-04-23 — Suppress blank console window on Windows release builds

**What:** Added
`#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]`
to the top of `linkage-sim-rs/src/bin/linkage_gui.rs`. In release builds
the resulting `linkage-gui.exe` is linked with the Windows GUI subsystem,
so launching it from the LinkageSuite launcher (or from Explorer) no
longer spawns an empty black console alongside the egui window.

**Why:** Rust's default bin subsystem on Windows is CONSOLE. Without the
attribute, Windows allocates a console for every GUI-only binary, which
appears as a blank terminal next to the simulator. Debug builds
intentionally keep the console so `cargo run`, `env_logger::init()`, and
panic backtraces still print to stdout.

**Test results:** `cargo check --bin linkage-gui` clean. No behavior
change to the library or WASM bin (`linkage_web.rs` is web-only and
doesn't need the attribute).

---

## 2026-04-15 — Crank angle label + canvas indicator

**What:**
- Added a small grey subtitle above the Crank Angle slider reading
  `Driver: <driven> relative to <partner> (0° = <ref> +X)`. Pulls body
  names from `mech.driver_body_pair()`. Makes explicit that the slider
  value is the driver constraint's f(t) = θⱼ − θᵢ.
- Added `draw_crank_angle_indicator` in `canvas/rendering/mod.rs`:
  renders an orange arc at the driver's revolute pivot spanning from
  the partner body's +X reference to the driver body's current
  orientation, with a perpendicular tick at the zero mark and a
  midpoint label showing the angle in the current display units.
  Skips silently when the driver isn't revolute or the connecting
  joint can't be identified.

**Why:** The slider label "Crank Angle" didn't say which body rotated
relative to what, or where 0° pointed. Now both the input panel and
the canvas answer that question at a glance.

**Test results:** 568 lib tests pass (no new tests — overlay is
render-only); native + wasm32 `cargo check` clean.

---

## 2026-04-15 — Flip-branch button + slider crash fix

**What:**
- New `AppState::flip_assembly_branch()` method reflects non-ground
  non-driver body poses across the line joining the two farthest-apart
  ground pivots (fallback: world x-axis when < 2 pivots exist), then
  re-solves at the current driver angle. On solver convergence it
  updates `q`, `last_good_q`, marks the sweep dirty, and recomputes
  forces. Status toast reports whether the flip actually changed the
  assembly or landed on the same branch.
- New `Flip Branch` button in the Crank Angle section (next to the
  Loop/Once toggle). Hover text describes the debug use-case.
- Fixed panic in `input_panel.rs`: the sweep-range slider's
  `f64::clamp(slider_min, slider_max)` panicked when the user typed a
  max below the current min mid-edit. Slider bounds are now
  defensively sorted; sweep computation still reads the raw values
  (commit-time enforcement already guarantees `max >= min`).

**Why:** Adjusting the sweep range can occasionally kick the solver
onto the other assembly branch (common with parallelogram-style or
change-point mechanisms). The Flip Branch button gives the user a
one-click way to jump back. The slider crash was a regression from
removing the auto-swap in the previous commit.

**Test results:** 568 lib tests pass (two new:
`flip_assembly_branch_lands_on_alternate_config`,
`flip_assembly_branch_noops_without_mechanism`); native + wasm32
`cargo check` clean.

---

## 2026-04-15 — Sweep range can cross the 0/360 seam

**What:**
- Sweep range min/max DragValue clamps relaxed from `0..=360` to
  `0..=720`. Users can now sweep ranges that cross the cycle seam
  (e.g. `200° → 365°`) which previously was impossible.
- `compute_sweep_data` rewritten so range-enabled sweeps iterate
  literally `min..=max` in 1° steps (e.g. 200..=365 = 166 samples with
  display angles 200, 201, …, 365). Range-disabled sweeps still cover
  0..=360 (361 samples) unchanged.
- Solver receives the display angle in radians directly — it doesn't
  require `mod 2π`, so "angle 365°" produces the same physical q as
  "angle 5°" while preserving a contiguous, monotonic plot X-axis.
- `SweepData::active_range` is now always `None` (the sweep IS the
  range). Plots render the active curve solid with no faded-context
  overlay. The field is retained to keep the plot render path stable.
- `q_at_zero` only updates when the sweep visits angle 0 (start ≈ 0).
  Custom-range sweeps leave the caller-supplied `q_at_zero` untouched
  so subsequent full sweeps re-seed from the correct angle-0 q.
- Commit-time enforcement: if user leaves max < min, max snaps up to
  min. Cleared the blueprint_ops swap-if-inverted fallback.

**Why:** User's press workflow has a stroke that spans the mechanism's
0°/360° transition. Without wrap support the "working range" of many
press linkages can't be expressed as a single contiguous sweep.

**Test results:** 566 lib tests pass (two old toggle-stability tests
updated; one new test `sweep_range_can_wrap_past_360` added). Native +
wasm32 `cargo check` clean.

**Breaking changes:** The length-361 invariant on `SweepData` vectors
no longer holds for range-enabled sweeps. Channel-length alignment
(all channels share the same length within a sweep) IS still enforced.
`02-system.yaml invariants_to_protect` updated.

---

## 2026-04-15 — DXF Add Geometry uses a link-picker popup

**What:**
- Renamed the DXF sidebar button "→ Add Geometry to Selected Link" to
  "→ Add Geometry to Link". Clicking it no longer requires pre-selecting
  a body; instead it opens a modal popup listing every non-ground body.
- Clicking a body name in the popup applies immediately (no Apply button)
  and closes the dialog. Escape or the close button cancels.
- New state fields `show_dxf_geometry_target_dialog` +
  `dxf_geometry_pending_indices` in `AppState`. The DXF entity indices
  are snapshotted when the popup opens, so later overlay edits cannot
  desync the target.
- New `DxfAction::OpenRigidGeometryTargetDialog` variant replaces the
  previous `ConvertSelectedToRigidGeometry` direct dispatch.
- `convert_selected_to_rigid_geometry` signature changed to take an
  explicit `target_body: String` (auto-resolution from `link_editor_body`
  / `state.selected` removed; the popup supplies it instead).
- New `draw_geometry_target_dialog(ctx, state)` rendered from
  `LinkageApp::update` beside the other dialog windows.

**Why:** Users had to remember to click a link on the canvas before
clicking the button; otherwise they got a toast asking them to do so.
The popup makes the flow explicit — every click of the button opens a
picker — eliminating the "nothing selected" error path entirely.

**Test results:** 565 lib tests pass; native + wasm32 `cargo check` clean.

**Breaking changes:** None at the API level. The button label changed
from "Add Geometry to Selected Link" to "Add Geometry to Link".

---

## 2026-04-12 — Modularize gui/mod.rs and gui/sweep.rs

**What:** Four refactorings, zero behavior change:

1. **Menu bar extraction:** `gui/mod.rs` (1925 lines) → extracted File/Edit/Help/View/Image
   menus + sample gallery into `gui/menu_bar.rs` (703 lines). `mod.rs` dropped to 1147 lines.

2. **Sweep submodule:** `gui/sweep.rs` (1399 lines) → `gui/sweep/` directory:
   - `mod.rs` (1003 lines): SweepData, compute_sweep_data, push_nan_row helper
   - `motion_profile.rs` (317 lines): trapezoidal velocity profile + tests
   - `fourbar.rs` (110 lines): 4-bar linkage detection

3. **Theme DRY fix:** Extracted `gui/theme.rs` (83 lines) with `cad_dark_visuals()`,
   `apply_nathan_mode()`, `restore_normal_visuals()`. Eliminates duplicated color
   constants between `LinkageApp::new()` and `restore_normal_visuals()`.

4. **push_nan_row helper:** Replaces 55 lines of channel-by-channel NaN pushes in
   the sweep solver-failure branch with a single function call.

**Test results:** 565 lib/integration tests pass. Pre-existing linear_driver doctest
still fails (not related).

**Breaking changes:** None. All public APIs re-exported from new `mod.rs` files.

---

## 2026-04-10 — Code quality: mutate_and_rebuild helper + Plotly extraction

**What:**
- Added `AppState::mutate_and_rebuild(|s| ...)` in `gui/state/undo_ops.rs`.
  Encodes the "push_undo + mutate + rebuild" invariant documented in
  `04-memory.yaml` as a single helper method. Skipping the helper is still
  possible but now there's a canonical way to get the pattern right.
- Converted all 12 mutation helpers in `gui/state/entity_crud.rs` to use it
  (update_ground_pivot_position, nudge_body, nudge_joint,
  add_attachment_point_to_body, remove_attachment_point, add_ground_pivot,
  add_body_with_points, remove_body, add_revolute_joint, add_prismatic_joint,
  add_fixed_joint, remove_joint). `blueprint_ops.rs` (10 sites) and
  `driver_ops.rs` (5 sites) still use the raw pattern — migrate later.
- Added `add_plotly_line_chart()` and `add_plotly_multi_chart()` helpers in
  `gui/export/report.rs`. Converted the 3 simple single-series chart sites
  (torque, actuator force, transmission angle with 40°/90° shapes) to use
  the helper. The multi-series plots (joint reactions, energy, coupler
  traces) still have inline loops — migrate later if patterns converge.

**Why:** Two of the "top 5 code quality issues" from the 2026-04-10 audit.
The undo helper enforces an invariant that future AI sessions are explicitly
told to protect; the Plotly helper removes ~60 lines of duplicated boilerplate.

Context: the audit also flagged "19 panics in force element setters" and a
"1475-line impl block in forces/elements/mod.rs" — both were false alarms.
The panics were all in `#[cfg(test)]` assertion code, and the 1620-line
`forces/elements/mod.rs` is almost entirely tests (production code is in
sub-modules totaling ~1087 lines). See 04-memory.yaml lessons_learned.

---

## 2026-04-10 — Modularize rendering and plot_panel

**What:** Split two of the largest GUI files:

- `src/gui/canvas/rendering.rs` (2233 lines) → `src/gui/canvas/rendering/` directory
  - `mod.rs` (1236 lines): render_mechanism, render_overlays, tooltips, grid, background image
  - `primitives.rs` (508 lines): spring/damper/arrow/arc/marker/alignment-guide drawing helpers
  - `force_render.rs` (526 lines): force element visualization + load path heat map

- `src/gui/plot_panel.rs` (1770 lines) → `src/gui/plot_panel/` directory
  - `mod.rs` (539 lines): PlotTab enum, dispatcher, shared helpers (colors, outlier filter, toggle markers, series-with-range)
  - `mechanics.rs` (324 lines): body angles, transmission, mechanical advantage, joint reactions
  - `dynamics.rs` (275 lines): driver torque, inverse dynamics, energy
  - `actuator.rs` (390 lines): force, speed, power
  - `coupler.rs` (289 lines): trace, velocity, acceleration, output force

**Why:** Both files mixed many distinct responsibilities. `rendering.rs` conflated
drawing primitives, force visualization, and mechanism render pass into one file.
`plot_panel.rs` had 13 independent plot functions that shared nothing except
helpers. The splits preserve behavior exactly (zero logic changes) but make the
files easier to navigate and reason about.

**Test results:** 644 tests pass, WASM + native builds clean, pre-existing
linear_driver doctest still fails (not related).

**Breaking changes:** None. All public APIs re-exported from the new `mod.rs`
files so external callers are unchanged.

---

## 2026-04-10 — Non-Grashof sweep support + two-pass statics

**What:**
- Sweep no longer breaks at unreachable angles. Pushes NaN across all data
  channels and continues, so non-Grashof mechanisms that only oscillate
  produce usable plots for the reachable range.
- Two-pass statics solve in sweep for actuator-driven mechanisms:
  pass 1 with force=0 → read driver torque → compute F_actuator via power
  balance → pass 2 with F_actuator injected into Q → reactions reflect the
  actuator load path.

**Why:** User's Custom 6-Bar press has a non-Grashof linkage (only oscillates)
and is actuator-driven. The previous sweep stopped at the first unreachable
angle, and joint reactions didn't show the actuator's load path.

---

## 2026-04-10 — DXF import overhaul

**What:**
- Drag-and-drop DXF import (works on web + native)
- Interactive DXF entity selection
- Conversion buttons: → Links, → Multi-joint Body, → Add Geometry to Selected
  Link, → Ground Pivots, → Linear Actuator, → Delete Selected
- Auto-create ground pivots when LinearActuator endpoint has no nearby body point
- All conversions are additive (preserve existing mechanism, push undo, rebuild)
- "Add Geometry to Selected Link" uses the Link Editor's pattern: sets body.geometry
  directly instead of creating a new body

**Why:** Imports SolidWorks sketches into the simulator with fast selection-based
workflow. Rigid geometry attachment matches the standard Link Editor behavior.

---

## 2026-04-10 — UX polish

**What:**
- `+Ground` tool snaps to unconnected link endpoints and creates revolute joint
- Joint creation handles coincident pivots correctly (excludes first-clicked point)
- Canvas joint labels show actual joint IDs (J1, J2, ...) instead of R1/R2 auto
- Driver speed editor added to input panel (RPM display, rad/s readout)
- Plot legends moved from RightTop to LeftTop to avoid blocking data
- Computed actuator force displayed on canvas when stored force=0 (sizing mode)
- Force zone creation no longer auto-creates BodyGeometry
- Sweep range editor defers recompute until drag_stopped/lost_focus to avoid
  crashes from transient inverted ranges
- `(full)` legend tooltip explains faded dashed vs solid curve when sweep range
  is enabled

**Why:** Various discoverability, correctness, and robustness fixes from user
feedback during the DXF press-mechanism workflow.

---

## Earlier history

See `docs/FEATURES.md` and `docs/history/` for the full project history including
the Python→Rust port (411→644 tests), Phase 5 GUI, sample mechanisms, export
formats, and Phase 6.8 UX overhaul.
