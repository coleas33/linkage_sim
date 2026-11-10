# Audit findings: mag-gui-panel
Scope read: panel.rs lines 1-1863 (all production code) plus the test Harness (1864-2045), test-name/helper/assert-count scans of the rest of its tests (about 8,600 lines, 203 tests, not read line by line); inputs.rs 1-1055 (WORKFLOW table skimmed), input_ui.rs, pickers.rs 1-330, size_controls.rs 1-464, sizing.rs 1-404, session.rs 1-451, history.rs 1-105; tests of the smaller files skimmed by name only.

## MGP-01: The 14 stored-3D input paths are written twice, and the second copy is test-only
- Kind: not needed (duplicate list)
- Where: magcoupling-rs/src/gui/dashboard.rs:237-252 (`STORED_3D_INPUTS`); magcoupling-rs/src/gui/inputs.rs:576-598 (WORKFLOW sections `calibration.stored_demag` and `calibration.stored_slip`)
- Evidence: the 10-line identical run in leads.md is `temperature.slip_loss.b_hub_T` ... `web_integral_free_T2m2`. `git grep STORED_3D_INPUTS` finds only dashboard.rs:18 (a doc comment), :237 (the definition), :1642, :1685, :1725 (all inside dashboard `mod tests`) and plain_text.rs:1531 (inside plain_text's test module, `fn stored_3d()`). Nothing in production code reads it. No test in inputs.rs ties the two lists together.
- Why it matters: a stored-3D input added to or moved out of the WORKFLOW sections will not reach the constant (or the reverse), and the tests that use the constant then check a stale list; it is a `pub const` in production code that only tests use.
- Suggested action: derive the list from the WORKFLOW sections (or the `rust_only`/section metadata) in a `#[cfg(test)]` helper, or add one test asserting `STORED_3D_INPUTS` equals the paths of the two WORKFLOW sections and move the constant under `cfg(test)`.
- Fix class: needs-judgment
- Confidence: likely (certain that it is test-only; whether a cross-check exists elsewhere in tests/ was searched by name only)

## MGP-02: Under Auto the design shown and the solve key are rebuilt several times a frame, against the code's own "once a frame" comment
- Kind: inefficient
- Where: panel.rs:714, 1147, 1347, 1656 (`shown_inputs()`); panel.rs:544-553, 788-797, 853 (calls into `SizingRunner`); sizing.rs:219-230, 289-298, 340-342 (`SolveKey::new`); size_controls.rs:87-94 (`change_text`); panel.rs:1227, 772 (`self.design()` clones)
- Evidence: panel.rs:1144-1146 says "the card and the drawing below read this one copy, built once a frame (with Auto on the ring size each build runs two geometries)", but `shown_inputs()` is also called at 714 (`ui`), 1347 (`inputs_ui`), 1147 (`view_ui`, Size tab) and 1656 once per locked row (ring size locks three rows: radius, cap OD, cap thread). Each call is `runner.shown` which for ring radius runs `with_ring_radius` (geometry). `SolveKey::new` does `state.variable.apply(inputs, 0.0)`, a full `DesignInputs` clone, and runs in `runner.update` (every frame), `is_current` (auto_outcome) and `status` (which calls `is_current`; called from auto_outcome, sizing_ui and, on a solve, log_sizing). `value_text` in `status` runs `sized` + `dimensions` (geometry) every call for ring size. `change_text` calls `compute_all(design)` on every frame in Auto for a blank axial-length override (the default), 17 us by sizing.rs:13's own figure. `self.design()` (a clone) is built in `header_ui` and again at the end of `ui`.
- Why it matters: with Auto on, every frame (all frames of a slider drag) repeats the geometry and the whole-design clone four to six times; individually cheap, but the comment and the module docs promise one copy.
- Suggested action: compute the design shown once at the top of `ui` and pass it down (or cache it keyed on inputs, sizing state, runner solve count and range, as `ResultsCache` does); give `SizingRunner` a cheap key (hash or a solve counter) instead of cloning `DesignInputs` in `is_current`/`status`.
- Fix class: needs-judgment
- Confidence: likely (call sites read; the cost per call was not measured, no cargo run allowed)

## MGP-03: Every drawn input row builds its tooltip strings each frame, hovered or not
- Kind: inefficient
- Where: input_ui.rs:120, 131, 141-144 (`input_tooltip`, `format!("{tooltip}\n{TRACE_HINT}")`, the reset hover `format!`); inputs.rs:1015-1054 (`input_tooltip`); input_ui.rs:153 (`on_hover_text(&tooltip)`)
- Evidence: `let tooltip = input_tooltip(entry, show_details);` runs for every `input_row` call before any widget; it allocates a `Vec`, a `format!` for the slider line and the default, and `lines.join("\n")`; line 131 allocates again with `format!("{tooltip}\n{TRACE_HINT}")`; a changed row also formats the reset hover text eagerly.
- Why it matters: with the Key design group (10 rows) and any open groups drawn, a drag repaints every row's strings at frame rate for text only a hover reads.
- Suggested action: build the text inside `on_hover_ui`/`on_hover_text_at_pointer` closures (egui runs them only when hovered), or cache the string per (path, show_details).
- Fix class: needs-judgment
- Confidence: likely (egui evaluates `on_hover_text`'s argument eagerly, as the code is written)

## MGP-04: The same few lines are written five times in panel.rs (and twice in input_ui.rs)
- Kind: inefficient (repetition)
- Where: panel.rs:863-864, 1466-1467, 1546-1548, 1610-1611, 1635-1636 (`let widget = self.input_row_ui(..); self.design_widgets.push(widget);`); panel.rs:1171-1189, 1215-1219, 1253-1257 (`PanelRequest::SaveFile { file_name, mime, contents }`, with "application/json" typed twice and "text/csv" once); panel.rs:979-981 and 993-995 (`if why_note_link(ui, TORQUE_BAND_NOTE) { readouts.open_note(TORQUE_BAND_NOTE); }` twice); input_ui.rs:230-237 and 256-263 (the same slider-and-edit block for an optional input and a plain number)
- Evidence: five identical two-line pairs, `design_widgets.push` after each `input_row_ui`. `input_row_ui` returns only the id and every caller but one (the Key design loop, which also pushes to `key_widgets`) does this push.
- Why it matters: the owner flags repetition; `design_widgets` is what holds undo steps while a field is typed in, so a sixth call site that forgets the push silently splits undo steps.
- Suggested action: have `input_row_ui` push to `design_widgets` itself (returning the id for the Key design loop), add one `fn save(&mut self, file_name, mime, contents)` for the requests, and one helper for the Why link; share the optional/plain slider branch in input_ui.rs.
- Fix class: needs-judgment
- Confidence: certain (the lines are quoted above)

## MGP-05: Comments and README still describe a sizing mode switch the UI no longer has
- Kind: inconsistent
- Where: inputs.rs:27-28 ("the sizing mode switch sits above the group"); sizing.rs:48 ("The text of the mode switch"); panel.rs:133-134 (`FREE_VARIABLE_LABEL`, "The label of the free-variable picker"); panel.rs:1342 ("through the mode switch"); magcoupling-rs/README.md:33 ("the sizing mode switch")
- Evidence: panel.rs:8012 (a test comment) states "The solver's controls live on the Size tab: no mode switch, picker or target here", and panel.rs:8014-8017 asserts that `SizingMode::*.label()`, `FREE_VARIABLE_LABEL` and `TARGET_LABEL` are not drawn. `git grep` shows `FREE_VARIABLE_LABEL`/`TARGET_LABEL` are used only by spreadsheet.rs (Summary rows) and these tests; `SizingMode::label` only by the spreadsheet and the app's log line (app.rs:58).
- Why it matters: a reader looking for the mode switch or the free-variable picker will not find them; the two constants are spreadsheet labels, not UI.
- Suggested action: reword the five places to say Auto on the Size tab; document the two constants as spreadsheet labels.
- Fix class: mechanical-safe (comments and a README sentence only)
- Confidence: certain

## MGP-06: A part or grade row says the same thing twice, in two different wordings
- Kind: inconsistent
- Where: inputs.rs:984-1004 (`text_hint`); pickers.rs:51-54, 252-256, 292-296 (`CUSTOM`, `BLANK_GRADE`, the "shown" texts); input_ui.rs:154-160 (draws both, picker first)
- Evidence: `text_hint` returns the literal "Blank: the manual Br, no rating" for a blank grade, which is the text of `pickers::BLANK_GRADE`, already drawn as the combo's selected text one line above. For an unknown part, the picker shows "{x} (not a library part: manual dimensions)" and `text_hint` adds "Not a library part: the manual dimensions are used"; for an unknown grade, "(not in the grade table)" and "Not in the grade table: the manual Br, no rating". For a library part, `text_hint` adds "Library part" under a line that already shows the full part label and properties.
- Why it matters: one state, two phrasings that can drift apart (the BLANK_GRADE literal is copied, not shared), and the row shows the information twice.
- Suggested action: use `pickers::BLANK_GRADE` in `text_hint`, or drop `text_hint` for the four picker inputs and keep one wording. Both copies are pinned by tests (pickers.rs:420-431 uses `text_hint(.., "")` to find the picker inputs), so the tests need the same change.
- Fix class: needs-judgment (the UX overhaul may rewrite these rows)
- Confidence: likely

## MGP-07: `gui::torque_gap_text` is re-exported for a screen dump that is not in the repository
- Kind: not needed
- Where: magcoupling-rs/src/gui/mod.rs:44-45
- Evidence: the comment says "public for the slice-3 screen dump (`size_controls` is private)". `git grep` for `torque_gap_text` finds it in gui/ (guidance, size_controls, torque_card, format, panel, mod.rs) and documents only; no file under magcoupling-rs/src/bin, tests/ or any example names `gui::torque_gap_text`, and there is no examples/ directory. The plan (docs/superpowers/plans/2026-10-20-magcoupler-torque-card.md:200, 807) says the dump "reads it from outside the crate".
- Why it matters: the one thing keeping `size_controls::torque_gap_text` and the `mod size_controls;` privacy workaround is an external tool nobody can find in the tree.
- Suggested action: ask the owner whether the dump tool still exists; if not, remove the re-export and make the function `pub(crate)` again.
- Fix class: owner-decision
- Confidence: unsure (the dump may live outside the repo; the owner can settle it)

## MGP-08: `key_widgets` is rebuilt every frame in production code but read only by tests
- Kind: not needed (in production)
- Where: panel.rs:354 (field), 442 (init), 1544-1547 (filled every frame); readers: panel.rs:2001 (Harness::widget) and 2625 (a test)
- Evidence: `git grep key_widgets` lists no read outside `mod tests`.
- Why it matters: test scaffolding in the production struct, with a clear and a push per Key design row per frame.
- Suggested action: put it under `#[cfg(test)]`, or have the tests read the widget ids from `design_widgets` by index (the Key design rows come first in that frame order).
- Fix class: needs-judgment
- Confidence: certain that nothing but tests reads it

## MGP-09: Magic numbers and path strings typed in view code that the engine or `FreeVariable` already name
- Kind: inconsistent
- Where: panel.rs:1131-1143 (2.5 mm keyway-wall warning and the text "existing guidance is about 2.5 mm"); panel.rs:987, 999 ("metal.torque_hot_low_Nm", "model.pullout_Nm"); size_controls.rs:316, 347, 357 ("coupling.inner_back_apothem_mm", "coupling.magnets.axial_length_mm"); size_controls.rs:261 uses `FreeVariable::MagnetsPerRing.path()` in the same function family
- Evidence: `FreeVariable::{RingRadius, AxialLength}.path()` (engine/sizing.rs:88-90) return exactly those two paths. Every other keyway-wall test in the tree uses `> 0.0` (torque_card.rs:531, engine/sizing.rs:227); `git grep 2\.5` finds the 2.5 mm wall threshold only in panel.rs:1136 and in the spreadsheet help text (docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md:748 "keep >= 2.5 mm").
- Why it matters: the warning threshold lives only in the GUI, so an engine or help-text change will not move it; the path strings can drift from `FreeVariable`.
- Suggested action: name the threshold in the engine next to `hub_wall_past_key_mm` and read it here; use `FreeVariable::*.path()` in controls_ui.
- Fix class: needs-judgment
- Confidence: likely

## MGP-10: `minimum_diameter_mm` is cleared in three overlapping places
- Kind: inconsistent
- Where: panel.rs:588 (`open_design`), 633 (`load_with`), 686 (`reset`)
- Evidence: `open_share_payload` goes through `load_with(.., Self::open_design)`, so `open_design` clears it (588) and `load_with` clears it again after `apply` (633). `undo`/`redo` go through `set_design`, which does not clear it, while `reset` clears it before calling `set_design`.
- Why it matters: the rule "every replacement of the design except undo/redo clears the minimum" is spread over three sites, one of them redundant; a fourth replacement path (as `set_design` is private and called from five places) can miss it.
- Suggested action: clear it in one place (a `replace_design` wrapper used by load, open and reset; `set_design` stays for undo/redo).
- Fix class: mechanical-safe
- Confidence: certain for the double clear on the open path

## MGP-11: An allowlist in a test is permanently empty
- Kind: not needed
- Where: panel.rs:4572-4575 (`const ALLOWED: &[(&str, &str)] = &[];` and its `.any(..)` filter)
- Evidence: the comment says "Texts allowed to, each with why: none." The constant is `&[]`, so the `!ALLOWED.iter().any(..)` clause is always true.
- Why it matters: dead machinery in a test, kept for a future exception; trivial.
- Suggested action: drop the constant and the clause, or leave it as a documented hook (owner's choice).
- Fix class: owner-decision
- Confidence: certain that it is empty

## Leads settled
- `magcoupling-rs/src/gui/dashboard.rs` and `inputs.rs` share 10 lines: real, the list of slip-loss field paths; the whole list of 14 stored-3D paths is written twice (MGP-01).
- `magcoupling-rs/src/gui/panel.rs` 10,480 lines: not a defect by itself. Production code is lines 1-1863; the other 8,600 lines are 203 tests and their helpers. No dead test helper found (every helper and const in the test module is used at least once; every test name is unique).
- Candidate "files nothing names" / `docs/ai/04-memory.yaml:93 gui/calculator_window.rs`: outside my part (dead path in a document), already in the baseline.
- Longest-function candidates: none of the listed ones are in my part.

## Not read
- panel.rs lines 2045-10480 (the tests): scanned by name, helper use and assertion counts only, not read line by line; 8 tests with at most one assert were opened or sized, none was empty (the_closest_note_link_opens_that_note asserts inside its helper).
- inputs.rs 120-565 (the WORKFLOW data table, skimmed), inputs.rs tests 1056-end; pickers.rs, size_controls.rs, sizing.rs, session.rs, history.rs `mod tests` bodies (names only).
