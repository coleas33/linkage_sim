# Audit findings: mag-engine-calc
Scope read: all non-test code of the 15 engine files and Cargo.toml; test modules, the magnet/material/grade data rows and library rows were skimmed or not read (see Not read)

## MEC-01: "quote the failing value one last digit past the limit" written three times
- Kind: inconsistent (repeated code)
- Where: magcoupling-rs/src/engine/shaft_joint.rs:610-634 (`quoted_above`, `quoted_below`); gearbox.rs:51-57 (inline copy of `quoted_above(value, rating, 2)`); housing.rs:91-93 (`quoted_overshoot`, the same floor with step 0.01)
- Evidence: gearbox.rs:53 `let quoted: f64 = rating.parse().unwrap_or(rating_Nm);` then `py_max(cold_high_input_Nm, quoted + 0.01)`; shaft_joint.rs:613 `let quoted: f64 = limit_text.parse().unwrap_or(limit);` then `py_max(value, quoted + step)`.
- Why it matters: the rule "a failing badge never shows two equal numbers" lives in three places; shaft_joint's helpers are private so gearbox re-derived it. A fourth check would copy it again.
- Suggested action: move `quoted_above`/`quoted_below` to engine/compat.rs (pub(super)), call `quoted_above(cold_high_input_Nm, rating_Nm, 2)` in gearbox.rs; housing's floor is a one-liner and can stay or use the same helper.
- Fix class: mechanical-safe (output strings stay identical; existing tests pin them)
- Confidence: certain

## MEC-02: Cargo.toml comments name checks and a host that no longer exist
- Kind: inconsistent (comment vs gate and vs repo)
- Where: magcoupling-rs/Cargo.toml:21-27 (feature `gui` comment: "a window in linkage-sim-rs, M5"), :56-60 (egui "so the panel embeds there (M5) ... gate.sh checks that both lock files agree"), :62-63 (egui_plot "Cargo.lock pins linkage-sim-rs's 0.33.0 and gate 11 checks it"), :74-80 (rust_xlsxwriter "gate 11 checks that both lock files agree (linkage-sim-rs carries the panel)")
- Evidence: linkage-sim-rs/scripts/gate.sh:123-132 gate 11 checks only `wasm-bindgen` (`lock_versions "$lock" wasm-bindgen`); `grep lock_versions` finds no egui, egui_plot or rust_xlsxwriter check anywhere. linkage-sim-rs/Cargo.toml has no magcoupling-rs and no rust_xlsxwriter (its Cargo.lock has none), and gate.sh:110-113 fails if linkage-sim-rs depends on magcoupling-rs again. docs/ai/backlog.yaml (the M5 item) says the panel embed was removed 2026-10-06.
- Why it matters: a reader trusts that a gate pins egui/egui_plot/rust_xlsxwriter in step with linkage-sim-rs; none does, so a version drift would pass the gate. The M5 embed is gone, so the "same 0.32 line so the panel embeds" reason is stale.
- Suggested action: reword the four comments to what is true (egui 0.32 and egui_plot 0.33 are kept equal to linkage-sim-rs by habit only; only wasm-bindgen is gated), or add the egui/egui_plot checks to gate 11 if the owner wants the claim to hold. The Cargo.toml feature comment should drop the M5 window.
- Fix class: needs-judgment (comment-only fix is safe; adding gate checks is owner-decision)
- Confidence: certain

## MEC-03: slider ranges are re-read from a freshly built input table on every call; the GUI caches the same data
- Kind: inefficient
- Where: magcoupling-rs/src/engine/sizing.rs:95-101 (`FreeVariable::range`: `input_rows(&DesignInputs::default())` then a linear find, per call); engine/size_controls.rs:123-128 (`geometry`: `input_rows(inputs)` per call, for 5 range lookups); callers per design change: gui/guidance.rs:2956-2968 (`Consequences::of` calls `smallest_fitting_diameter`, `dimensions`, `diameter_breakdown`, `length_breakdown`, each its own `geometry` = one `compute_all` + one `input_rows`), gui/sizing.rs:127-129, gui/torque_card.rs (several `dimensions_with_minimum_diameter`)
- Evidence: engine/meta.rs:701-711 `input_rows` allocates a `String` path per leaf input. gui/inputs.rs:809 caches the same metadata in a `OnceLock<InputCatalogue>` and its test (inputs.rs:1498-1509) says the range is "the input's own metadata", equal to `variable.range()`. Slider ranges are static (`.range(..)` on the `param`), independent of the design values.
- Why it matters: the Size tab recomputes these on each design change (the gap-slider drag is timed by gui-smoke); about ten table builds and about four extra `compute_all` runs per change are repeated work, and the engine has two other ways to spell one fact the GUI already caches.
- Suggested action: keep one `OnceLock` of the ranges needed (npole, inner_back_apothem_mm, cap_od_mm, cap_thread_dia_mm, axial_length_mm) in the engine and read it from `FreeVariable::range` and `geometry`; make `Consequences::of` call `geometry` once (expose a function that returns controls, diameter and length breakdown together).
- Fix class: needs-judgment (no output change, but touches the size-controls API; time it with the gui-smoke frame check before and after)
- Confidence: likely (the repeated calls are certain; the cost per call is not measured)

## MEC-04: workbook tables carried for the parity test only, with a doc comment that says the GUI uses one
- Kind: not needed (or inconsistent comment)
- Where: magcoupling-rs/src/engine/metal_design.rs:180-242 (`VALIDATION_ITEMS`, "A GUI checklist"); engine/temperature.rs:139-140 (`AdhesiveCandidate::role`, `::note`)
- Evidence: `grep VALIDATION_ITEMS` over the worktree: only the definition, README.md:143, tests/static_data.rs:95 and the Python source; no file under src/gui reads it. `.role` and `.note` of an adhesive are read only at tests/static_data.rs:131. The adhesive selector texts are separate literals (temperature.rs:204 "AA 326", ...), not `ADHESIVES[i].name`.
- Why it matters: 12 rows of text and two fields per adhesive are compiled into the web bundle with nothing showing them, and the comment promises a GUI that does not exist; the selector labels and the table can drift apart.
- Suggested action: either show the checklist and the adhesive role and note in the GUI (they are real engineering content) or move them behind `#[cfg(test)]` / delete and keep the Python-only copy; at least fix the comment.
- Fix class: owner-decision
- Confidence: likely (settled by one more search for `role`/`note` use in gui, done: none)



## Leads settled
- magcoupling-rs [dev-dependencies] `magcoupling-rs`: not real. It is the deliberate self dev-dependency that turns on the test-only `workbook-parity` feature (Cargo.toml comments; gate 10 guards that no shipped build gets it).
- Longest functions `temperature::compute` (554 lines), `model::compute` (278), `clamps::compute` (212): not findings. Each is one workbook sheet ported line by line, and splitting them is a taste matter the audit does not cover.
- Pub-item scan of my 15 files (name appears once in all .rs): only test functions came back, so no dead function, const or struct in the part. Result-count claims in the module docs (73, 130, 49, 9, 7, 25, 33, 165 cells; 26 sweep fields) were counted against the code and are all right.
- Same-run duplicates touching my part (scope.rs/notes.rs, theme.rs): not in my files.

## Not read
- No file is wholly unread. Skimmed or skipped: test modules of temperature.rs (from line 1625), model.rs (from 1320), sizing.rs (570-659), clamps.rs (564-841), materials.rs, calibration.rs, metal_design.rs (520-635); the data rows of material_library.rs (180-760), grades.rs (142-630), library.rs (260-340).