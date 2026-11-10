# Verified: mag-engine-calc

## MEC-01: "quote the failing value one last digit past the limit" written three times
- Verdict: PARTLY. Holds for gearbox.rs and shaft_joint.rs: gearbox.rs:51-57 is `quoted_above(cold_high_input_Nm, rating_Nm, 2)` written inline (same rating text, same `py_max(value, quoted + 0.01)`, same `fmt_fixed(..., 2)`), and shaft_joint.rs:610-618 is the private helper. Does not hold for housing.rs:91-93: `quoted_overshoot` takes one value, has no limit text and no parse of one, and floors at 0.01; it is a different one-line rule, not a copy of `quoted_above`.
- Checked: gearbox.rs:40-64, shaft_joint.rs:605-634, housing.rs:80-93; search for `quoted_above|quoted_below|quoted_overshoot|quoted +` over magcoupling-rs rs files (callers: shaft_joint.rs:585-601, housing.rs:156, gearbox.rs:56).
- Safe to apply without judgment: no (moves a helper across modules and edits gearbox.rs; behaviour is identical but it needs a build to check imports, which I may not run).

## MEC-02: Cargo.toml comments name checks and a host that no longer exist
- Verdict: CONFIRMED
- Checked: Cargo.toml:15 ("a window in linkage-sim-rs, M5"), :57-60, :62-63, :76-77 read; linkage-sim-rs/scripts/gate.sh:123-132 gate 11 checks only `wasm-bindgen` (`lock_versions "$lock" wasm-bindgen`); `lock_versions` is only called there; gate.sh:110-113 fails if linkage-sim-rs names magcoupling-rs; linkage-sim-rs/Cargo.toml has `egui_plot = "0.33"` at :23 and no magcoupling-rs or rust_xlsxwriter; `name = "rust_xlsxwriter"` appears only in magcoupling-rs/Cargo.lock; no test reads Cargo.toml (search of tests/). The comments "gate.sh checks that both lock files agree", "gate 11 checks it" and "gate 11 checks that both lock files agree" are all false.
- Safe to apply without judgment: no (the true wording depends on whether the owner wants the gate checks added; the finder marked it needs-judgment).

## MEC-03: slider ranges are re-read from a freshly built input table on every call; the GUI caches the same data
- Verdict: CONFIRMED (structure; cost per call not measured, as the finder said)
- Checked: sizing.rs:95-101 (`input_rows(&DesignInputs::default())` then a find, per call); size_controls.rs:118-138 (`geometry`: `input_rows(inputs)` then five range lookups, then `compute_all(inputs)`); `geometry(` is called by 11 public functions in size_controls.rs (:250-455); gui/guidance.rs:2956-2968 (`Consequences::of` calls `smallest_fitting_diameter`, `dimensions`, `diameter_breakdown`, `length_breakdown`); meta.rs:701-711 (`input_rows` allocates a `String` per leaf); gui/inputs.rs:809 `static CATALOGUE: OnceLock<InputCatalogue>` exists. Ranges come from `&'static` input metadata, so they do not depend on the design values.
- Safe to apply without judgment: no (changes the size-controls API and wants the gui-smoke frame timing).

## MEC-04: workbook tables carried for the parity test only, with a doc comment that says the GUI uses one
- Verdict: PARTLY. Holds: `VALIDATION_ITEMS` (metal_design.rs:181) is read only by tests/static_data.rs:95 (plus README.md:143 and the Python source), no gui file reads it, and its doc comment "A GUI checklist" (:180) promises a GUI that does not exist; `AdhesiveCandidate::role` and `::note` (temperature.rs:139-140) are read only at tests/static_data.rs:131. Does not hold: "compiled into the web bundle" for `VALIDATION_ITEMS`: an unreferenced `pub const` emits nothing in a build, so only the adhesive `role` and `note` strings (inside the referenced `ADHESIVES`) ride along. The finder's claim that the selector labels are separate literals is true (temperature.rs:204 lists "AA 326", ...).
- Checked: metal_design.rs:176-190; temperature.rs:128-141 and the `ADHESIVES` search (tests/static_data.rs:124-131, temperature.rs:193, :199, :1784, explain/tables.rs:21, :163-164: none read `.role` or `.note`); search `VALIDATION_ITEMS|\.role|\.note` over rs, js, html, sh, toml; search for "Prototype metrology" in rs, js, html: only metal_design.rs.
- Safe to apply without judgment: no (owner decision: show it, hide it behind cfg(test), or fix the comment).

## Seen while checking
- None.
