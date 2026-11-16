# Verified: mag-gui-panel

## MGP-01: The 14 stored-3D input paths are written twice, and the second copy is test-only
- Verdict: CONFIRMED
- Checked: git-grep-style search over *.rs/js/html/sh/py/toml in the worktree for STORED_3D_INPUTS: dashboard.rs:18 (doc), :237 (def), :1642, :1685, :1725 (all in dashboard tests), plain_text.rs:1531 (test module). 6 hits, none in production code. Docs (specs) also name it. The cross-check test was searched by name only.
- Safe to apply without judgment: no, moving the const under cfg(test) or adding a test is a design choice and docs/specs name the constant.

## MGP-02: Under Auto the design shown and the solve key are rebuilt several times a frame
- Verdict: PARTLY (the call sites hold; cost not measured)
- Checked: panel.rs shown_inputs() called at 714, 1147, 1201, 1347, 1656 (production), plus 485 and 976 (compute_all of it); the comment at panel.rs:1144-1146 says "built once a frame". The multiple calls per frame are real. The cost per call, and the SolveKey clone claims, were not re-measured (no cargo run allowed).
- Safe to apply without judgment: no, a caching change alters behaviour paths.

## MGP-03: Every drawn input row builds its tooltip strings each frame, hovered or not
- Verdict: CONFIRMED
- Checked: input_ui.rs:120 (`let tooltip = input_tooltip(entry, show_details);` before any widget), :131 (`format!("{tooltip}\n{TRACE_HINT}")` passed eagerly to on_hover_text), :141-144 (eager format! for reset hover), :153 (`on_hover_text(&tooltip)`).
- Safe to apply without judgment: no, moving into on_hover_ui closures changes widget code and needs a check that tests reading hover text still pass.

## MGP-04: The same few lines are written five times in panel.rs (and twice in input_ui.rs)
- Verdict: CONFIRMED (panel.rs part read; input_ui.rs :230-263 not re-read)
- Checked: `design_widgets.push(widget)` at panel.rs:864, 1467, 1548, 1611, 1636 (plus 965, a different source). SaveFile mimes at 1173 ("text/csv"), 1178 and 1255 ("application/json"). why_note_link(ui, TORQUE_BAND_NOTE) at 979 and 993.
- Safe to apply without judgment: no, a refactor of production code.

## MGP-05: Comments and README still describe a sizing mode switch the UI no longer has
- Verdict: CONFIRMED for inputs.rs:27-28, panel.rs:133-134 and sizing.rs:48; PARTLY for panel.rs:1342 and README.md:33 (the sizing mode itself still exists, Auto sets it via set_sizing_mode, only a "switch" widget is gone)
- Checked: panel.rs:8012-8021 test asserts SizingMode labels, FREE_VARIABLE_LABEL and TARGET_LABEL are not drawn (count 0). FREE_VARIABLE_LABEL/TARGET_LABEL used elsewhere only at spreadsheet.rs:31,319,324 and tests. SizingMode::label used at app.rs:58 (log line), spreadsheet and panel tests. Read inputs.rs:24-29, panel.rs:126-137, 1336-1347, sizing.rs:48, README.md:30-36.
- Safe to apply without judgment: yes for the three comment lines below (comments only; no test reads source text).
- Exact change:
  - magcoupling-rs/src/gui/panel.rs:133: old `/// The label of the free-variable picker.` new `/// The label of the free variable in the spreadsheet's Summary (the Size tab has no picker row with this text).`
  - magcoupling-rs/src/gui/sizing.rs:48: old `/// The text of the mode switch: ASCII arrows, since egui's default fonts have no U+2192.` new `/// The text of the mode in the spreadsheet's Summary and the log line: ASCII arrows, since egui's default fonts have no U+2192.`
  - magcoupling-rs/src/gui/inputs.rs:27-28: old `the sizing mode switch sits above\n/// the group.` new `Auto's status and a button to the\n/// Size tab sit above the group.`

## MGP-06: A part or grade row says the same thing twice, in two different wordings
- Verdict: CONFIRMED
- Checked: inputs.rs:984-1004 text_hint returns the literal "Blank: the manual Br, no rating"; pickers.rs:54 `BLANK_GRADE` is the same text and is the combo's selected text (:294, :306). input_ui.rs:154-160 draws the picker then text_hint. Test dependence on text_hint was not re-read.
- Safe to apply without judgment: no, UX wording is pinned by tests and the UX overhaul may rewrite the rows.

## MGP-07: `gui::torque_gap_text` is re-exported for a screen dump that is not in the repository
- Verdict: NOT CONFIRMED (nothing in the tree uses the re-export, but whether the external dump exists cannot be proved)
- Checked: search of all *.rs/js/html/sh/py/toml for torque_gap_text: gui/mod.rs:45 is the re-export; every other use goes through size_controls:: or is inside gui. No bin/, tests/ or examples/ use `gui::torque_gap_text`. Specs and the plan (2026-10-20-magcoupler-torque-card.md:200, 807, 14106) say the dump uses it from outside the crate.
- Safe to apply without judgment: no, owner decision; docs/ai and plan name the re-export.

## MGP-08: `key_widgets` is rebuilt every frame in production code but read only by tests
- Verdict: CONFIRMED
- Checked: key_widgets at panel.rs:354 (field), 442 (init), 1544 and 1547 (clear/push in production), 2001 (Harness::widget) and 2625 (test). 6 hits, no other readers in the worktree.
- Safe to apply without judgment: no, cfg(test) on a struct field changes the struct per profile and the production code writing it needs gating too.

## MGP-09: Magic numbers and path strings typed in view code
- Verdict: CONFIRMED
- Checked: panel.rs:1136 `key_wall < 2.5` and :1140 text; panel.rs:987 "metal.torque_hot_low_Nm", :999 "model.pullout_Nm"; size_controls.rs:316, 347, 357 literal paths, :261 and :400 use FreeVariable::MagnetsPerRing.path(). engine FreeVariable path() equivalence not re-read.
- Safe to apply without judgment: no, the 2.5 threshold move is a design decision.

## MGP-10: `minimum_diameter_mm` is cleared in three overlapping places
- Verdict: CONFIRMED
- Checked: panel.rs:588 (open_design), :633 (load_with, after apply), :686 (reset). open_share_payload passes Self::open_design as apply (:597-601), so the open path clears twice. set_design (used by undo/redo, :662, :669) does not clear. open_design is pub, so removing :588 would change a direct caller.
- Safe to apply without judgment: no, open_design is public and a direct caller would lose the clear.

## MGP-11: An allowlist in a test is permanently empty
- Verdict: CONFIRMED
- Checked: panel.rs:4572 `const ALLOWED: &[(&str, &str)] = &[];` used only in the closure at :4574.
- Safe to apply without judgment: no, finder marked owner-decision (a documented hook).

## Seen while checking
- None.
