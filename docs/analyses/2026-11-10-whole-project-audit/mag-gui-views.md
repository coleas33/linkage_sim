# Audit findings: mag-gui-views
Scope read: app.rs, app/files.rs, app/theme.rs, bin/*, lib.rs, gui/mod.rs, gui/format.rs, gui/result_groups.rs, gui/xlsx.rs and the non-test code of gui/dashboard.rs, explorer.rs (to line 820), results_table.rs, plots.rs (195-420), geometry.rs (440-740), corrections.rs (1-135), clamp_drawing.rs (740-880); unused-item and duplicated-block scans over all of the part.

## MGV-01: The results table and the clamp tab draw the catalogue label; every other view draws the display label
- Kind: inconsistent
- Where: magcoupling-rs/src/gui/results_table.rs:771; magcoupling-rs/src/gui/clamp_drawing.rs:827, 862 (screw_rows at 776)
- Evidence: results_table.rs:771 `cell(ui, label, entry.info.meta.label, None);`; clamp_drawing.rs:827 `ui.label(egui::RichText::new(info.meta.label).weak());`. The dashboard draws `label_for(line.path)` (dashboard.rs:748), the hover text's first line is `display_label(path, info.meta.label)` (dashboard.rs:530), plots and explorer use `label_for`. plain_text.rs:71 `("model.pullout_Nm", PULL_OUT_NOMINAL)` renames results, and a results_table.rs test (946-964) asserts `label_for(path) != catalogue` for four results.
- Why it matters: the pull-out row in the table reads under its old spreadsheet name while its own tooltip, the dashboard, the plots and the Equation panel use the new on-screen name; the search was made to match both because the screen name is the new one (format.rs:42-45).
- Suggested action: draw `display_label(&entry.path, meta.label)` in the table row and the clamp tab (exports keep the catalogue label, as the dashboard test says), or record that the table deliberately shows the catalogue label.
- Fix class: owner-decision (a UX overhaul of this GUI is being designed)
- Confidence: likely; settle by checking spec 3.5 on which label the table should show (no test pins the table's drawn label)

## MGV-02: Stored-3D path lists typed twice, and the dashboard copies are used only by tests
- Kind: not needed
- Where: magcoupling-rs/src/gui/dashboard.rs:237-252 (STORED_3D_INPUTS) and 258 (STORED_3D_SENTENCE); magcoupling-rs/src/gui/inputs.rs:576-598
- Evidence: dashboard.rs:238-251 lists the 14 paths `temperature.demag.h_rev_*` and `temperature.slip_loss.*`; inputs.rs:577-597 lists the same 14 in the `calibration.stored_demag` and `calibration.stored_slip` sections. A comment-stripped scan finds `STORED_3D_INPUTS` and `STORED_3D_SENTENCE` used in no non-test code (STORED_3D_INPUTS: dashboard.rs tests 1642/1685/1725 and plain_text.rs:1531; STORED_3D_SENTENCE: tests only).
- Why it matters: a public constant in the shipped build exists only to feed tests, and its 14 paths duplicate the input workflow's two sections, so adding a stored field means editing both.
- Suggested action: move both constants into the test modules (or build the list from the two WorkflowSections) and delete the duplicate path list.
- Fix class: needs-judgment (a test of the sentence may want it public)
- Confidence: certain for "used only by tests"; the duplication is certain

## MGV-03: Two `pub` functions in shipped code are called only from tests
- Kind: not needed
- Where: magcoupling-rs/src/gui/spreadsheet.rs:196 (`Cell::as_text`; its callers start at line 599, after `#[cfg(test)]` at 572); magcoupling-rs/src/gui/result_groups.rs:136 (`package_label`; only caller 389, after `#[cfg(test)]` at 226)
- Evidence: Grep over the worktree for `as_text` and `package_label` finds the definition and only lines inside the test modules; `groups_of` iterates PACKAGE_LABELS directly (result_groups.rs:204).
- Why it matters: dead code in the shipped binary and wasm build; rustc does not warn because the items are `pub` in a library.
- Suggested action: move each into its test module (as_text into a `#[cfg(test)] impl Cell`).
- Fix class: mechanical-safe
- Confidence: certain

## MGV-04: The Equation panel reads `clamps::PAUSED` while the panel passes `clamps_paused` everywhere else
- Kind: inconsistent
- Where: magcoupling-rs/src/gui/explorer.rs:133, 745 (and `start_here_hover(clamps::PAUSED)`); magcoupling-rs/src/gui/panel.rs:415-417, 1938-1942
- Evidence: panel.rs:415 documents `clamps_paused` as "A field, not the constant, so a test draws either state"; dashboard_ui, results_table, geometry and the spreadsheet all take it as a parameter. explorer.rs:745 `let users = used_by_shown(&path, clamps::PAUSED);`; panel.rs:1939 admits "The explorer reads `clamps::PAUSED` itself ... so even this harness walks the paused Start here".
- Why it matters: with the field set to false (the test harness, or a later un-pause switch) the dashboard and table show the clamp rows while Used-by and Start here still hide them: two sources for one rule.
- Suggested action: pass `clamps_paused` into `explorer_ui` and the Start-here code instead of the constant.
- Fix class: needs-judgment
- Confidence: likely (behaviour is identical in the shipped app, where both equal the constant)

## MGV-05: plots.rs matches the sweep status by literal strings the engine owns
- Kind: inconsistent
- Where: magcoupling-rs/src/gui/plots.rs:322-326; magcoupling-rs/src/engine/sweeps.rs:173-175
- Evidence: plots.rs:323 `"nominal: test needed" => points.nominal.push(p),` and 324 `"below hot minimum" => ...`, everything else (`_`) goes to `no_fit`; sweeps.rs:173/175 returns the same two literals. Other GUI code uses the engine's constants (shaft_joint, housing and gearbox in dashboard.rs:394-406).
- Why it matters: rewording a status text in the engine silently moves every marker to "no_fit" in the plot; the sweep status texts have no constant.
- Suggested action: give the two statuses constants in engine/sweeps.rs and match on them.
- Fix class: mechanical-safe
- Confidence: likely (a plots test, 745-770, would catch a reword)

## MGV-06: Every dashboard guide recomputes its level and notes each frame, also when its block is shut
- Kind: inefficient
- Where: magcoupling-rs/src/gui/dashboard.rs:829 and 867 (fit_ui, indexed_guide_ui); 865
- Evidence: fit_ui calls `guidance::level(guide, inputs, results)` (829), then `indexed_guide_ui` calls it again for the same guide (867). indexed_guide_ui builds `guide_notes(fixes, index, inputs, results)` (865), which allocates trade_offs, side_effects (format! per effect) and unchecked costs, yet the notes are read only in `fix_line`, drawn when `open` (898-900).
- Why it matters: about ten guided rows build and drop a Vec<String> every frame while green or amber-shut, on a path the owner timed frame by frame (gui-smoke).
- Suggested action: pass the level from fit_ui, and build the notes only when the block is open.
- Fix class: mechanical-safe
- Confidence: likely (the cost per frame is small; it is allocation work that buys nothing)

## MGV-07: The same two axis lines are written out in three diagram functions
- Kind: inefficient
- Where: magcoupling-rs/src/gui/diagrams.rs:294-301, 425-432 (approx), 502-509
- Evidence: torque_angle (293-301) and heating_curve (501-509) both open the vec with `Shape::line_segment([plot.left_bottom(), plot.right_bottom()], Stroke::new(1.0, pens.weak))` then `[plot.left_bottom(), plot.left_top()]`; the same first segment is at 426. The 7-line-window duplicate scan flags 293-295 against 501-503.
- Why it matters: owner flags repetition; the axes are one drawing idea.
- Suggested action: one `axes(plot, pens) -> [Shape; 2]` helper.
- Fix class: mechanical-safe
- Confidence: certain (small win)

## MGV-08: The `app` feature lists dependencies its `gui` feature already enables
- Kind: not needed
- Where: magcoupling-rs/Cargo.toml (feature `app`: "dep:log", "dep:js-sys"; feature `gui` lists both)
- Evidence: `gui = [... "dep:log", ... "dep:js-sys"]` and `app = ["gui", "dep:eframe", "dep:log", "dep:env_logger", "dep:rfd", ..., "dep:js-sys"]`.
- Why it matters: two lists to keep in step, with no effect.
- Suggested action: drop the two repeated entries from `app`.
- Fix class: mechanical-safe
- Confidence: certain (Cargo.toml is just outside the listed files; it carries the app feature)

## Leads settled
- magcoupling-rs/src/app/theme.rs:13-43 and linkage-sim-rs/src/gui/theme.rs: not real. The copy is deliberate and a test (theme.rs `the_visuals_are_the_linkage_app_s`) compares the two function bodies via include_str!.
- magcoupling-rs/src/gui/dashboard.rs and inputs.rs (10 lines): real, the stored-3D path list: MGV-02.
- clamp_drawing at clamp_drawing.rs:285 (279 lines): not a finding; one function that mirrors drawing.py, and clamp code is deliberate (PAUSED). Not read line by line.
- the_smoke_test_share_link_opens_its_design (app.rs:212, 204 lines) and dashboard.rs:1247 (244 lines): long tests, not defects.
- panel.rs, guidance.rs, torque_card.rs, inputs.rs long items: outside my part.

## Not read
- Whole files: magcoupling-rs/src/gui/test_support.rs (test-only; scanned for unused helpers: none), geometry_view.rs, readouts.rs, spreadsheet.rs (only grepped; non-test body not read)
- Partly: geometry.rs (1-440, 740-896 not read), explorer.rs (820-1172), plots.rs (1-195, 424-680), diagrams.rs (most), clamp_drawing.rs (1-740, 880-883), corrections.rs (135-216), dashboard.rs and all other test modules
