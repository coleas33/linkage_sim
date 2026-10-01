# magcoupling-rs

Magnetic slip coupling design calculator: the Rust port of the `magcoupling`
1.0.0 Python package, itself a port of `magnetic_coupling_torque_calculator.xlsx`.
Spec: `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` (M2).

The vendored Python package, `reference/magcoupling-py/`, is the **oracle**.
Every result must match the workbook snapshot and the Python engine, except
where an approved correction from the M1 math audit is registered in the
deviation registry.

**Status:** M2 complete: every Python module except `fields3d` (M3) is ported;
workbook parity covers all 1,149 checks; E1 to E14 applied (decisions D1 to D7
as recorded). Addendum A-1 (data and physics) complete: the A6 grade table and
the parts' vendor data, any grade with manual dimensions, the A5 materials
library with per-part selectors, physics links and six warnings, and E15 to E20
applied (Addendum A decisions, approved 2026-09-30). Addendum A-2 (parameters
and sizing) complete: the harmonic set up to 11 with one general E7 peak search,
the assumptions registry (A3), the end-effect validity flag, the axial length
override with the axial housing that follows it (hub length, cup cavity depth,
retainer span), inverse sizing (A1), the housing autofit suggestion and the space
claim, one aluminium modulus, and each grade-mode ring's own alpha and density
(decisions A2-1 to A2-9). M4 infrastructure in place: the `gui` and `app` features, a tracer panel
(`gui::MagcouplingPanel`), the native and web binaries, and the second web
bundle at `/magcoupling/` (see [Features and binaries](#features-and-binaries)).
Next: Addendum A-3 (explanations), then the M4 GUI plans.

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
[Differences from the workbook](#differences-from-the-workbook); E14 rewords help
text and README only, no number changes. Decisions D1 to D7 are recorded in
`docs/ai/04-memory.yaml` and `docs/ai/05-update-tracker.md`. M3 (`fields3d`,
after M4 in the recorded phase order) must also apply E3 and E5.

```rust
use magcoupling::{DesignInputs, compute_all, headline};
let res = compute_all(&DesignInputs::default());
for (key, value) in headline(&res) {
    println!("{key}: {value:?}"); // the dashboard numbers, in Python's order
}
```

## Invalid inputs

`compute_all` never panics, whatever the inputs hold. `InputSet::set` refuses a
wrong type, NaN, an infinity and a selector code outside its choices, but a struct
literal, a design file or a share link can hold one anyway. Then the results are
meaningless, not fatal. A selector code outside its choices gives NaN numbers or
the Excel-style text `"#N/A"` where Python would raise or silently pick another
row (the adhesive, the screw class), and the workbook's own IF fall-through where
a two-way IF decides. A typed value far outside its slider gives results that may
be inf or NaN. Call `DesignInputs::validate()` where inputs enter: it returns
every offending path, in schema order, with the reason `set` would give. The
Rust-only selectors follow the same rule: a material code outside its choices
gives NaN properties and the name `"#N/A"`, never another material; the
two-way coercivity source (E20) falls through to C44 and C45, as a workbook IF;
and a harmonic set outside its choices (`coupling.max_harmonic`) gives NaN
torques, never another set.

## Differences from the workbook

Every correction below is approved, E1 to E14 in the M1 math audit report
(`docs/analyses/2026-09-29-magcoupling-math-audit.md`, the row with the same
id) and E15 to E20 in the Addendum A verification report
(`docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`, the decisions
of its section 8 that each entry's `approval` cites), and registered in `src/engine/deviations.rs` with the cells it changes and
their workbook and corrected values (for E3, E4 and E5, which change 246, 47 and
37 cells, in the reviewed golden files `tests/data/deviations/E3.json`, `E4.json` and `E5.json`). The corrections are always on for users
(`compute_all`). Only tests switch them off, through the test-only
`workbook-parity` feature (`compute_all_with`, `Deviations::NONE`,
`Deviations::only`, `Deviations::with`, `Deviations::without`,
`DesignInputs::defaults_with`): the parity and differential
tests to compare against the workbook and the Python engine exactly, and the
registry, metadata, robustness and unit tests to isolate one correction, to
probe one on top of the corrections it refines (`depends_on`: E15 to E17 on E9,
decision 15), or to start from the workbook's defaults.

| Id | Cells | Workbook | This port | Report |
|---|---|---|---|---|
| E1 | Temperature design!C96 (feeds C104-C106, C201, C202) | 0.55 GPa (EA 9514's modulus) | 0.107 GPa (AA 326 TDS); C106 and C202 now read "Below ..." | E1 |
| E2 | Clamp screw sizes!C34:G35, Shaft clamps!C48 | length = CEILING(grip + 2d, 2); fits = length <= grip + thread | the slit (0.8 mm) is added to both; M4 x 12 becomes M4 x 14, which protrudes 0.34 mm: clamps.length_note says so | E2 |
| E3 | Magnet library N42SH rows; Calculator!C17, C21, C27, C31; Calibration!C21 | 1.29 T | 1.30 T (vendor minimum); pull-out 2.688 N·m, limit 93.06 °C, C91 "OK: 7x margin"; fields3d rerun pending (M3) | E3 |
| E4 | Pole sweep!C6:C11 | a_i = MAX(w/(2 tan(pi/N)) + 0.05, bore/2 + key + 2.5) | + inner bondline in the wall term; the 6-pole row reads "outside OD envelope" | E4 |
| E5 | Temperature design!C121 (feeds C125, C130 and the thermal rows) | 1.035e-5 T²·m² (with the steel hub's image, not doubled at the web; free space alone is 6.837e-6, E17) | 4.14e-5 T²·m² (doubled at the steel surface); total slip loss 2.751 W; fields3d part is M3 | E5 |
| E6 | Calculator!C63 | OD/2 − block-back apothem | OD/2 − (block back + outer bondline): 2.723 mm | E6 |
| E7 | Calculator!C76, C82, C88 → C89-C96; sweeps N, Q, T, U; Calibration!C40-C42 | every harmonic at half a pole pitch | the maximum of the harmonic torque-angle curve (`model::peak_off_half_pitch`: every root of dT/dx in cos² x, for any odd harmonic set up to 11, with no closed form and no scan grid; Addendum A decision 29); same at defaults; 6 poles 0.861 → 0.911 N·m | E7 |
| E8 | Calculator!C9, C57, C111; Metal design!C175 | flat-block corner geometry in arc mode | the corner radius C55 (face radius for arcs), round pocket; arc-mode pull-out 2.99 → 2.65 N·m | E8 |
| E9 | Calculator!C111, C113; Materials!C22 | steel cup and boss, back-iron wall advice even with no back iron | aluminium cup and boss, "No back iron"; total mass 156.9 → 96.8 g at backiron = 0 | E9 |
| E10 | Calculator!C103 → C104-C106; Materials!C20-C22 | (Br_i + Br_o)/2 · (t_i + t_o)/(t_i + t_o + g) | (Br_i t_i + Br_o t_o)/(t_i + t_o + g); same for identical rings | E10 |
| E11 | Temperature design!C91 | 0.2 typed into the formula | the fatigue-endurance input C195 | E11 |
| E12 | Temperature design!C19, C23, C150-C153 | negative times and drag when the start is above the limit | 0 s, 0 rev, 0 N·m | E12 |
| E13 | Temperature design!C156, C157 | #DIV/0! (Python: the whole calculation stops) | +inf; the other results are computed. JSON export must encode inf (M4) | E13 |
| E14 | Shaft clamps!C35 help; README | "At 22 mm only M3 fits; two of them need a 14.5 mm clamp." | "At 22 mm M4 no longer fits. Two M3 need a 14.5 mm clamp; three M2.5 need 18 mm, and from 18 mm up the calculator recommends M2.5 x 3." | E14 (documentation) |
| E15 | Temperature design!C141 (feeds C143, C145, C154-C161, C171, C172, C180-C182, C186, C189, C190, C192, C193, C196, C20) | cup, boss and hub at 4140's specific heat even when the mass model makes them aluminium | aluminium parts at C140 (900 J/(kg·K)): the cup and boss on E9's gate, the hub when C6 is not 1; hardware stays steel. Back iron 0 on E9: C141 45.78 → 63.02 J/K, no verdict changes | Addendum A row E15 (decisions 8, 15) |
| E16 | Metal design!C189 → C191, C148, C149 (C147, C189 help) | the disc bored out of the web for the adapter pilot at steel density | at the web's density, one source with the mass model (`model::cup_boss_density`): aluminium with no back iron under E9. Back iron 0 on E9: C189 3.453 → 1.188 g, C191 101.5 → 103.8 g | Addendum A row E16 (decisions 9, 14, 15) |
| E17 | Temperature design!C123, C124, C125 → C130-C134 and the thermal rows (41 cells); 3 Rust-only inputs | the steel skin-limited formula, 4140's σ and μr and the steel-circuit fields for the aluminium hub, cup and web | at C6 = 0 the low-Reynolds closed form T1 (end factor C114, σ = Materials!C43) with the free-space fields `temperature.slip_loss.b_hub_free_T` 0.07832 T, `b_cup_free_T` 0.08764 T, `web_integral_free_T2m2` 6.837e-6 T²·m² (4 s.f., decision 12; M3 computes them live); with E9 off the hub sees C116/2. Back iron 0 on E9: C130 2.477 → 2.918 W, C18 89.77 → 94.18 °C, C19 "never" → 440.3 s (inside the model's uncertainty: finite only for f_end ≥ 0.646) | Addendum A row E17 (decisions 10-15) |
| E18 | Temperature design!C104, C105, C201; verdicts C106, C202 | 4140's expansion coefficient and modulus (C94, C98) for a hub the mass model makes aluminium | at C6 ≠ 1 (E15's hub gate) the screen uses 6061-T6, 23.6e-6 /°C and 68.9 GPa (Alliance datasheet); C94 and C98 still show the steel inputs. M2 basis at back iron 0: C104 11.68 → 20.76 MPa, C106 and C202 read "Above ..." | Addendum A row E18 (decision 16) |
| E19 | Calculator!C22, C32 → the rating-calibrated demag cells (C47, C50, C56-C61, C7-C10, C12, C13, C15, C24, C181, C182) | M5044, M5045, M5026 rated 80, 100, 80 °C; M5045 graded N50M | the vendor's specification grid: 60 °C for all three, M5045 graded N50; Br stays 1.42 T. M5044 on both rings: C12 29.18 → 9.18 °C | Addendum A decision 2 |
| E20 | Temperature design!C44, C45 → C42, C47-C61, C7-C10, C12 and the margins | one N42SH coercivity curve (C44 1592 kA/m, C45 −0.5 %/°C) for every magnet, and only the inner ring's Br and rating against the outer blocks' reverse fields | each ring's own grade (library part, or the grade picked for manual dimensions), Br and rating, unless `temperature.demag.coercivity_source` = 0 picks C44 and C45; the ring with the lower magnet limit governs and the block shows it (`temperature.demag.demag_ring`); a positive beta (ferrite) has no knee on heating: hot onsets +inf, the grade's rating is the hot limit, and Rust-only cold onsets, cold limit and cold check (against Metal design C16, both rings) feed the verdict. A positive beta with no rating has no hot limit: C60 = +inf (the adhesive governs C12) and C61 = NaN, which exporters must handle as they handle E13's +inf. B842 on both rings: C12 23.06 → −3.52 °C; B842SH inside B842: 92.55 → −3.52 °C. N42SH keeps 1592 and −0.005 (decisions 17, 18), so defaults do not move | Addendum A decision 19 |

With the Addendum A5 part selectors (decision A7), "aluminium" in E9 and E15-E18 means the body material: the workbook's aluminium (C42, C140, Materials!C43 and E18's 6061) by default and whenever C6 = 0 overrides a steel pick, else a non-ferromagnetic back-iron pick (304, 6061) with its own library values. The rows' figures are the report's, at the workbook's aluminium.

Results can be +inf (E13); exporters must handle it.

## Layout

| Path | Contents |
|---|---|
| `src/lib.rs` | Crate docs and re-exports: `compute_all`, `headline`, `DesignInputs`, `DesignResults` |
| `src/engine/meta.rs` | Field metadata: `inputs!`/`results!`, `param`/`out` builders, `Value`, get/set/visit of inputs and get/visit of results by dotted path (a table row's field as `table[i].field`); table rows (`rows!`, `TableLayout`, `col`/`at_row` builders) with synthesized workbook cells |
| `src/engine/compat.rs` | Python and Excel semantics the port reproduces (see the translation rules below) |
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E20 (all applied: E1 to E14 from the M1 audit, E15 to E20 from Addendum A), probes, and the `Deviations` switch |
| `src/engine/constants.rs` | Physical constants (`MU0 = 1.256637e-06`, the workbook's rounded value) |
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup; a grade-mode ring reads its alpha and density (decision A2-7) |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells, plus the Rust-only `tau7_Pa` to `tau11_Pa`, `end_effect_check`, and the explorer's torque-angle amplitudes `amp1_Pa` to `amp11_Pa` and E7 angle `pullout_angle_rad`, plan A-3) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6, and the axial length override `coupling.magnets.axial_length_mm`, Addendum A1), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below, each ring's alpha(Br) and magnet density used: a grade-mode ring's grade, decision A2-7, and the equation explorer's terms, Rust-only reads, plan A-3: the wave number, amplitudes and geometry factors of harmonics 7 to 11, `k7` to `s11_free`, each harmonic's torque-angle amplitude `amp1_Pa` to `amp11_Pa`, `Harmonic::amplitude` being the one expression the shear stress and the pull-out share, and the E7 angles `pullout_angle_rad`, `iron_circuit_angle_rad` and `free_circuit_angle_rad`), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names and `cup_wall_suggested_mm`, the autofit's cup wall, decision 27) |
| `src/engine/temperature.rs` | `Temperature design` sheet: 7 input groups (37 cells), the adhesive table `ADHESIVES` with `selected_adhesive`, `TemperatureLinks`, `TemperatureResults` (130 cells) |
| `src/engine/clamps.rs` | `Shaft clamps` and `Clamp screw sizes` sheets: 25 input cells, `SCREW_SIZES`, `MACHINING_STEPS`, `ClampResults` (33 cells), the 165-cell screw table, `screw_class_name` |
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28; with the axial length override set, `axial_housing` makes the hub length, the cup cavity depth and the retainer span follow the rings they bound, each at least its ring's length: decision A2-8) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, `space_claim_check`, and the hub length, cup depth and retainer span in effect) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`, `KEY_INPUTS`), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included), `test_support.rs` (headless egui helpers, tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page), `run_native`, `TITLE`, `CANVAS_ID` |
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |

Not a Cargo workspace member: `linkage-sim-rs` will depend on it by path
(feature `gui`, M5).

## Features and binaries

| Feature | Adds | Dependencies (all optional) |
|---|---|---|
| none (default) | The engine | none: pure std, builds for wasm32 as it is |
| `gui` | `gui::MagcouplingPanel`: the design inputs and results, and `fn ui(&mut self, ui: &mut egui::Ui)`. Any egui app can host it: the standalone app as a full page, the linkage app in an `egui::Window` (M5) | egui 0.32 |
| `app` | `app::MagcouplingApp` and the binaries below | `gui`, eframe 0.32, log; env_logger (native); wasm-bindgen, wasm-bindgen-futures, web-sys (wasm32) |
| `workbook-parity` | **Test-only**: the switch that turns corrections off (see [Differences from the workbook](#differences-from-the-workbook)) | none |

The panel is the M4 tracer: sliders for face gap, pole count and axial length
(`KEY_INPUTS`), set up from the input metadata (label, unit, range, step, log
scale, help and cell in the tooltip), "Reset all", and the headline numbers,
recomputed with `compute_all` (every correction on) each frame after the inputs
are drawn. Sliders clamp edits only (`SliderClamping::Edits`): the slider, arrow
keys and typed values stay in the range, and an idle frame never rewrites a
value. The axial length is the manual inner length, which only matters when
the inner part is not a library part.

**Versions.** egui and eframe use the same 0.32 line as `linkage-sim-rs`, so
M5 embeds the panel with one egui; `Cargo.toml` has caret ranges, and
`Cargo.lock` pins the versions of `linkage-sim-rs/Cargo.lock` (egui and eframe
0.32.3, wasm-bindgen 0.2.114, which is the `wasm-bindgen-cli` version
`deploy-web.yml` installs). Gate 11 fails when the two lock files or the CLI pin
disagree; bump all three together.

| Binary | Target | Run |
|---|---|---|
| `magcoupling-app` | native | `cargo run --release --features app --bin magcoupling-app` (from `magcoupling-rs/`) |
| `magcoupling-web` | wasm32 | `bash linkage-sim-rs/scripts/build_magcoupling_web.sh` builds it into `linkage-sim-rs/web/magcoupling/`; `bash linkage-sim-rs/scripts/serve_web.sh [PORT]` serves it at `http://localhost:8080/magcoupling/`, next to the linkage app. `build_web.sh` builds both bundles. On a desktop, `cargo run --features app --bin magcoupling-web` opens the native window |

The web page is `linkage-sim-rs/web/magcoupling/index.html` (committed; the JS
glue and the wasm are gitignored build outputs). Its canvas id is
`app::CANVAS_ID`, which a test checks. `deploy-web.yml` runs
`build_magcoupling_web.sh` after the linkage build, so both bundles ship
(`linkage.colesorkness.com/magcoupling/`).

**workbook-parity never ships.** The feature reaches `cargo test` and
`cargo clippy --all-targets` through the self dev-dependency, and cargo then
unifies it into every unit of the build, binaries included, so a
`compile_error!` on `app` plus `workbook-parity` would break
`cargo test --features app`. The guard is
`linkage-sim-rs/scripts/magcoupling_shipped.sh` instead. It defines the shipped
cargo arguments once (`MAGCOUPLING_WEB_ARGS`, `MAGCOUPLING_NATIVE_ARGS`), and
`magcoupling_assert_shipped` reads cargo's `--message-format=json` record of
the units it compiled. It fails unless the shipped binary was compiled and no
magcoupling-rs unit has the feature. `build_magcoupling_web.sh` pipes its
release build through it, so the shipped path refuses such a bundle. Gate 10
runs the guard on the native and wasm32 builds, plus a negative control that
must trip.

## Tests

```bash
cargo test                      # from magcoupling-rs/: the engine
cargo test --features app       # plus the panel and app tests (headless egui)
bash linkage-sim-rs/scripts/gate.sh   # everything, both crates and the Python oracle
```

| Test | Checks |
|---|---|
| `tests/parity.rs` | Every result with a workbook cell (table values too: their cells are synthesized from the table layout) and every default input equals `tests/data/reference_values.json` (numbers 1e-9 relative, 1e-12 absolute; text exact), deviations off. Per-group cell counts are a ratchet (`PORTED_INPUTS`, `PORTED_RESULTS` with `cells` and `table_cells` in `tests/common/mod.rs`); `the_port_checks_every_cell_test_parity_checks` pins their totals to `test_parity.py`'s 1,149. |
| `tests/differential.rs` | Every result of every seeded case equals the Python engine (`tests/data/differential/<group>.json`), deviations off. The full run (`differential/full.json`, `full_run_matches_python_on_every_case`) varies all 160 inputs at once and compares every result, groups and tables, so it also catches a `MODULES` entry that forgot an input group; `every_selector_pair_is_covered_in_the_full_run` checks that every pair of selector choices across groups occurs in it, and `every_selector_choice_appears_in_every_module_file` that each module file sets every selector it varies to every choice. `every_branch_is_reached` checks the `BRANCHES` table (every branch of every text result is hit), `every_text_result_has_a_branches_entry` that no text-producing result lacks a `BRANCHES` row (a new branch cannot land unchecked), and `every_varied_input_takes_two_values` that each varied input changes; the helpers corpus checks `compat` against Python exactly. Rust-only results (`ResultMeta::rust_only`, e.g. `clamps.length_note`) have no Python counterpart and are skipped; so are Rust-only selectors in the selector-coverage tests (`InputMeta::rust_only`: the generator never passes them to Python). |
| `tests/python_schema.rs` | Every ported field carries the Python label, unit, help, cell, choices and default; no Python field of a ported group is missing; each ported table has the Python field order, row count and cell of every value (`tables_match_the_python_layout`); `headline` has Python's keys, order and values at the defaults; inputs and scalar results are listed in the Python order (the GUI's tables and CSV export follow it); Rust-only inputs and results are skipped, and no Rust-only input may share a path with a Python one (`rust_only_inputs_are_unknown_to_python`). |
| `tests/grades.rs` | The grade table equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (engine literals bit for bit; the N42SH engine beta is the workbook's); every library part resolves to a grade; a part's workbook Br and Tmax equal its grade's except the registered differences; sintered NdFeB alpha and density equal the engine constants; only ferrite has a positive beta; every part cites its vendor page for coating and magnetization; two Y30 grade rings take Y30's alpha(Br) and density and give the Calculator torques and the temperature limit of C22 typed to Y30's value by hand (`a_grade_ring_scales_with_its_own_alpha_and_weighs_at_its_density`, decision A2-7); a library NdFeB ring beside a Y30 grade-mode ring, each way round, keeps each ring's own alpha(Br) and density in the cold torques (C8, C155), the E20 block (the governing NdFeB ring's coefficient and onsets), the magnets' mass and the bond block's mass (`mixed_rings_each_take_their_own_alpha_and_density_either_way_round`). |
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for (6061-T6's engine modulus is E18's 68.9 GPa, decision A2-6); the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300, with the default back iron and with none, where E17's free-space field inputs act (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a harmonic set outside its choices (NaN torques in the Calculator, the Calibration prototype and every sweep row, never another set, with the corrections off and on; `validate()` names it: `an_invalid_harmonic_set_is_nan_not_another_set`), a coercivity source outside its choices (any code but 1 uses the Hcj and beta inputs, and `validate()` names it: `an_invalid_coercivity_source_uses_the_inputs`), a positive beta typed in for magnets with no grade and no rating (no hot limit: C60 = +inf, C61 = NaN, the adhesive governs C12: `a_positive_beta_without_a_rating_has_no_hot_limit`), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), inverse sizing on designs no slider reaches (an outcome or an error for every free variable, never a panic: `sizing_never_panics_on_extreme_designs`), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells (hand-listed, or in the correction's golden file under `tests/data/deviations/`); a correction that changes more than 15 cells uses a golden file and one that changes 15 or fewer lists them (`broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`); each applied correction's figures match the report (`e1_adhesive_shear_modulus_matches_the_report`, `e2_clamp_screw_length_matches_the_report`, `e3_library_remanence_matches_the_report`), and help a correction rewords belongs to a real input and differs from the workbook's. A correction that changes nothing at defaults (E7 to E13, E15 to E20) carries registry probes, off-default inputs from the report whose cells are checked with the correction off and on (`each_probe_shows_its_correction`); `every_applied_engine_correction_is_visible_somewhere` requires every applied engine correction to show at defaults or in a probe. A probe's workbook value can be a workbook error (`Literal::Error`, e.g. `#DIV/0!` where Python raises): that side still runs but is not compared. E7, E8, E9, E10, E11, E12 and E13 must leave every default cell bit for bit (`e7_leaves_every_default_cell_bit_for_bit`, `e8_leaves_flat_blocks_bit_for_bit`, `e9_leaves_every_default_cell_bit_for_bit`, `e10_leaves_every_default_cell_bit_for_bit`, `e11_leaves_every_default_cell_bit_for_bit`, `e12_leaves_every_default_cell_bit_for_bit`, `e13_leaves_every_default_cell_bit_for_bit`). `e7_finds_the_peak_at_a_fill_of_exactly_0_4` pins E7 where the fifth harmonic vanishes (a ring at a fill of exactly 0.4). The Addendum A entries name every cell their probes change, downstream cells included (`addendum_entries_name_every_cell_their_probes_change`); E15 to E18 have rows in the Addendum A report and E19 and E20 cite decisions only (`e15_to_e18_have_audit_rows_and_e19_e20_decisions_only`); each reproduces the report: `e15_heat_capacity_matches_the_report`, `e16_removed_disc_matches_the_report`, `e17_aluminium_eddy_losses_match_the_report` and `e15_to_e17_together_match_the_reports_headline_table` (on top of E9, decision 15), `e18_aluminium_hub_mismatch_matches_the_report`, `e19_supermagnetman_arcs_follow_the_vendor_grid`, and for E20 `e20_each_part_uses_its_own_coercivity`, `e20_the_hcj_and_beta_inputs_override_the_grade_when_selected`, `e20_ferrite_is_limited_on_the_cold_side`, `e20_mixed_rings_use_the_weaker_grade` and `e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature`; `e15_leaves_every_default_cell_bit_for_bit` to `e20_leaves_every_default_cell_bit_for_bit` pin that none moves a default cell. `all_corrections_together_give_the_reviewed_headline` pins what users see: the headline with every correction on at the default design. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame (drawn text equals `compute_all` of the edited inputs and differs from the default's); pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept; "Reset all" restores the default design and headline; the panel draws inside an `egui::Window` (M5); each key input is a numeric input with a slider range; tooltips carry help, path and cell |
| `src/gui/format.rs` (feature `gui`) | Four significant digits, scientific outside 1e-3 to 1e6, carries (9.99996 shows as 10.00), signed zero, `+inf`, `-inf`, `NaN`, integers, text, None, units |
| `src/app.rs` (feature `app`) | The app draws the whole panel as a page; `web/magcoupling/index.html` has the canvas `CANVAS_ID` and the title `TITLE` |

### Regenerating test data

The slider ranges are defined once, in Rust. Three generators write the data
files; `cargo test` then compares:

| # | Generator | Writes |
|---|---|---|
| 1 | `MAGCOUPLING_BLESS=1 cargo test --test schema` | `tests/data/input_schema.json`: every input's metadata and slider range, read by generator 2 |
| 2 | `cd reference/magcoupling-py && ./.venv/Scripts/python tools/gen_differential.py` (`.venv/bin/python` on POSIX; `--check` only verifies, and is what the gate runs) | `tests/data/python_schema.json` (the Python metadata, with the table layouts), `tests/data/static_data.json` (the static tables) and `tests/data/differential/*.json` (one file per result group, `helpers.json` for `compat`, and `full.json`) |
| 3 | `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells` | `tests/data/deviations/E<k>.json`, the golden files of the broad corrections (E3, E4, E5); see [Deviations](#deviations). It writes what this engine computes with the correction alone, so review the diff |

Run 1, then 2, when an input, a range, a choice or a table layout changes; run 3
when a broad correction changes cells. A Rust-only input (declared with
`param_rust_only`, exported with `"rust_only": true`) is left out by generator 2:
the Python engine has no such input, so every case keeps its Rust default and no
data file changes when one is added.

`MODULES` in `gen_differential.py` maps each ported **result group** to the
**input groups** each case varies: every group the results read, directly or
through upstream sheets (`TEXT_CHOICES` lists the values a text input may take,
since it has no slider). `differential/<group>.json` is columnar: the header
lists `input_groups`, `input_paths` and `result_paths` once, and each case is
one line `{"id", "tag", "inputs": [...], "results": [...]}` with values in the
header's path order. A module's cases are the fixed ones first (defaults, each
input at each range end and each choice, the `PROBES` on branch boundaries),
then random ones: as many as it takes to reach `CASES_PER_MODULE` and never
fewer than `RANDOM_MIN` (100). A data file over `MAX_FILE_BYTES` (4 MB) fails
the generator: vary fewer groups or cut cases.

`differential/full.json` (`FULL`) is not a module: its cases vary every input
group at once (`full_cases`: the workbook defaults, `FULL_RANDOM` = 100 random
sets, then one case for each pair of selector choices across groups that the
random sets missed) and compare every result group and table. A new module needs
no line for it: the full run picks up its inputs and results from the schema.

`BRANCHES` in `tests/differential.rs` lists, per text-producing result path
(`[*]` matches any table row), the numbers and texts (`Number`, `Text`,
`Prefix`) that the module's cases must reach, so each branch of the Python
source is compared at least once. Add its rows when a module lands:
`every_text_result_has_a_branches_entry` fails while a `Text` or `NumOrText`
result has none.

The gate runs `gen_differential.py --check`, so stale data fails it. The
snapshot copy must equal `reference/magcoupling-py/tests/reference_values.json`
(a test checks).

## Porting a module

1. Read the Python module whole. List its dataclasses and cells, helpers,
   f-strings, text sentinels, branches and the places Python can raise. Read
   the audit report rows that touch it (the planned deviations).
2. Inputs: one `inputs!` struct per Python input dataclass. Transcribe each
   `name: float = param(default, unit, label, help, cell, choices)` as
   `name: f64 = default => param(unit, label, help, cell).choices(..)`, byte for
   byte (the metadata-parity test catches typos). Add `.range(min, max, step)`
   from physical bounds (and `.log()` for wide ranges) and `.assumption()` for
   the Addendum A3 assumptions.
3. Results: one `results!` struct per Python result dataclass, `out()` calls
   transcribed. Type each field by the values Python actually produces (the
   annotations are not reliable): `f64`, `NumOrText` for a number or a fixed
   text sentinel, `String` for built text, `i64` for an index. Tables (the screw
   table, the sweeps): one `rows!` struct per Python row dataclass, each field
   with `col(unit, label, help, "B")` (rows run down the sheet) or
   `at_row(unit, label, help, 6)` (rows run across it), `uncelled_col` for a
   field without a cell; list it in the `results!` struct as
   `tables { name: Row => TableLayout::RowsDown { sheet, first_row } }` (or
   `ColumnsAcross { sheet, columns }`). Its values get the paths `name[i].field`
   and synthesized cells, so the parity test checks them and
   `table_columns_match_the_workbook_headers` checks each column's label and
   unit against the workbook; add the table's cells to `table_cells` in
   `tests/common/mod.rs`, and its layout to `table_layouts()` in
   `gen_differential.py` (`tables_match_the_python_layout` compares).
   `result_rows` lists a group's tables after its scalars, so the screw table
   comes after `clamps.layout_relief` (Python declares it after
   `boss_radius_mm`); the scalar order is Python's.
4. Compute: `pub fn compute(.., dev: Deviations) -> XResults`, line by line in
   Python's order with Python's local names (`#[allow(non_snake_case)]` on the
   function where Python uses capitals). Follow the translation rules.
5. Wire it into `api.rs` in the Python `compute_all` order; add the group to
   `PORTED_RESULTS` (and its input groups to `PORTED_INPUTS`) in
   `tests/common/mod.rs` and to `MODULES` (with the input groups it varies)
   and `PROBES` (a case on each side of every branch boundary) in
   `gen_differential.py`.
6. Bless the schema, regenerate the data, run `cargo test`, add the module's
   text results to `BRANCHES` in `tests/differential.rs`, and run the gate.
7. Apply deviations afterwards, one commit each, never in the port commit.
8. Add one equality-edge unit test per module with threshold comparisons: put
   each comparison at exact equality (inputs chosen so both sides are the same
   `f64`), assert the equality first, so a drifted input fails loudly instead of
   testing the wrong side, then assert the branch Python takes there. The module
   tests of `model`, `metal_design`, `materials`, `temperature`, `clamps` and
   `sweeps` are the examples (`checks_take_the_python_branch_at_exact_equality`,
   `wall_check_passes_at_equality`, `hot_day_torque_meets_the_minimum_at_equality`,
   `status_checks_are_strict_at_exact_equality`).

## Translation rules (Python to Rust)

| Python | Rust |
|---|---|
| `float` arithmetic | `f64`, same operand order: never reorder, factor or simplify a formula |
| `x ** 2`, `x ** 3` | `x.powi(2)`, `x.powi(3)`; `x ** y` (float y) is `x.powf(y)` |
| `math.sqrt/exp/log/sin/...`, `math.pi` | `f64` methods, `std::f64::consts::PI` |
| `min(a, b)`, `max(a, b)` | `compat::py_min`, `compat::py_max` (not `f64::min`/`max`) |
| `sum(...)` | a left fold from `0.0` in the same order |
| `lo <= x <= hi` | `(lo..=hi).contains(&x)` |
| `int / int` | `a as f64 / b as f64` (Python `/` is always float division) |
| `math.ceil`, `math.floor` | `.ceil()`, `.floor()`, but Python returns an int: never `-0.0` |
| `_fields.ceiling`, `_fields.floor_` | `compat::ceiling`, `compat::floor_` |
| `f"{x:.2f}"` | `compat::fmt_fixed(x, 2)` (ties to even; `"nan"`) |
| `repr(x)`, `str(x)` of a float | `compat::py_repr` (exact 17-digit ties go to the even digit) |
| `clamps._fmt_num`, `temperature._text0` | `compat::fmt_num`, `compat::text0` |
| `round(x)`, `int(x)` | `x.round_ties_even()`, `x.trunc()` |
| `None` input | `Option<f64>`; a text result sentinel is `NumOrText::Text` |
| `isinstance(x, (int, float))` | a match on `Option` or `NumOrText` |
| `if code == 1 ... else ...` on a selector | the same, catch-all `else` included |
| `{1: a, 2: b}[code]` (KeyError, or a negative index that wraps around) | a `match` whose fallback yields NaN or an Excel-style error text; never panic, never another row. Three uses: `ScrewClasses::proof` (NaN), `selected_adhesive` (`NO_ADHESIVE`: name `"#N/A"`, NaN numbers) and `screw_class_name` (`"#N/A"`) |
| an `int` from an input or a derived count (`coupling.npole`, `calibration.total_magnets`, `clamps.joint_screws`) | `i64`, but never `+`, `-` or `*` on it: Python ints cannot overflow and a debug build panics. Convert with `n as f64` for arithmetic; `x as i64` saturates (NaN gives 0) and `saturating_add` replaces Python's `+` on an int; comparisons against selector codes stay integer |
| an exception (`ZeroDivisionError`, math domain) | Rust yields inf or NaN; slider ranges keep Python in its domain, and the generator fails loudly if Python raises |
| rounded workbook constants | the same literal (`MU0 = 1.256637e-06`, not `4π·1e-7`) |

## Deviations

A correction is applied in its own commit after its module is workbook-exact:
branch at the formula with `if dev.is_on(DeviationId::Ek) { corrected } else
{ workbook }` (keep the workbook form beside it); for a corrected default,
declare the corrected default and record the workbook value in the entry's
`workbook_input_defaults`; set the entry to `Applied` and list every cell that
changes at defaults in `changes_at_defaults`. Where the correction rewords a
help text, record the workbook's text in `workbook_help` (an input path, a
scalar result path (decision 14: E16 and E17 keep the workbook's labels and
reword the help), or a table column as `group.table[*].field`): `tests/python_schema.rs` compares
Python's help against it, `tests/schema.rs` the workbook note of a table
column, and both `tests/deviations.rs` and `tests/schema.rs` require the port's
help to differ. A result the Python engine does not have (E2's
`clamps.length_note`) is declared with `out_rust_only`: it has no cell, and the
metadata-parity, order and differential tests skip it. `tests/deviations.rs`
then proves the correction changes exactly those cells, to those values; add a
test of the report's figures (`assert_report`, `assert_workbook`) and a row to
[Differences from the workbook](#differences-from-the-workbook). Physics
changes get a dedicated physics reviewer (spec, testing summary).

A broad correction, one that changes more than 15 cells at defaults (decision
D4; E3 changes 246 through Br), is not hand-listed: its entry leaves
`changes_at_defaults` empty and names a golden file in `changes_file`,
`tests/data/deviations/E<k>.json`, which maps every changed cell to
`[workbook, corrected]`. Write it from a run with the correction alone:

```bash
MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells
```

(run only this test: the rest of the `deviations` binary may run first and fail
on a missing file). Then review the file as a diff: every changed cell should
follow from the correction (for E3, Br-linear values move by 1.30/1.29, torques
by its square, safety factors by the inverse), and the report's figures stay
pinned by hand in the correction's `e<k>_..._matches_the_report` test, so a
blessed file cannot quietly move them. Without `MAGCOUPLING_BLESS` the test
compares against the file.
