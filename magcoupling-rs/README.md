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
(decisions A2-1 to A2-9). M4 infrastructure in place: the `gui` and `app` features, the panel
(`gui::MagcouplingPanel`), the native and web binaries, and the second web
bundle, served at [colesorkness.com/magcoupler](https://colesorkness.com/magcoupler/) (see [Features and binaries](#features-and-binaries)).
Addendum A-3 (explanations) complete: the engine side of the equation
explorer (`src/engine/explain/`: one evaluable markup per result, proven by the drift guard, over
decision 31's paths, the geometry callouts and the terms they need to reach inputs), the A3
traceability test over every assumption, and the 17 A4 teaching notes behind a physics-review gate.
M4-1 complete: every input generated from the metadata (the Key design group first), the
dashboard (badges, corrected-vs-workbook markers, the end-effect greying, the stored-3D label,
the space claim), the searchable results table with CSV and JSON export, undo and redo, design
files and share links, the sizing mode switch and the linkage app's theme (see
[The panel](#the-panel); decisions M41-1 to M41-15). M4-2 complete: the centre region's views,
the geometry view to scale (the default: both rings, the housing in effect, the dimension
callouts, the space claim and the design checks), five plots (egui_plot) and the clamp drawing
and table, and a results table that fits a narrow window (decisions M42-1 to M42-9). M4-3
complete: the equation explorer (an egui typesetter for the A-3 markup, every displayed value's
equation on hover with its terms coloured and marked on screen, the docked Equation panel with
its breadcrumb, term list, "used by" and leaf-to-slider links), the assumptions view and banner,
the teaching notes with their diagrams and the start-here order, the material, part and grade
pickers and the material warnings (decisions M43-1 to M43-15). The ordering of the inputs and the
results complete (plan `docs/superpowers/plans/2026-10-02-magcoupling-gui-ordering.md`): the
inputs by design workflow (the workbook's package groups one toggle away) with a filter box, the
results by physics chain with check badges and a failing filter, and tracing between an input and
the results it drives (decisions O-1 to O-8). Next: M3 (live 3D fields).

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
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names and `cup_wall_suggested_mm`, the autofit's cup wall, decision 27, and the materials in effect: the steel density, the hub, cup and boss values without back iron, the sleeve's and the cap's, plan A-3) |
| `src/engine/temperature.rs` | `Temperature design` sheet: 7 input groups (37 cells), the adhesive table `ADHESIVES` with `selected_adhesive`, `TemperatureLinks`, `TemperatureResults` (130 cells, plus the Rust-only results of E20: each ring's own Hcj, beta, magnet limit and cold limit, plan A-3) |
| `src/engine/clamps.rs` | `Shaft clamps` and `Clamp screw sizes` sheets: 25 input cells, `SCREW_SIZES`, `MACHINING_STEPS`, `ClampResults` (33 cells), the 165-cell screw table, `screw_class_name` |
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28; with the axial length override set, `axial_housing` makes the hub length, the cup cavity depth and the retainer span follow the rings they bound, each at least its ring's length: decision A2-8) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, `space_claim_check`, and the hub length, cup depth and retainer span in effect) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/engine/explain/` | Addendum A2 equation explorer, engine side (plan A-3): the formula markup (`markup.rs`: grammar, parser, parse tree), its evaluator with a trace of what a result depends on (`eval.rs`), the static tables a formula reads (`tables.rs`: magnets, grades, the part materials, aluminium alloys, adhesives, screw sizes), the authoring form (`record.rs`) and the records, one file per chain (`records/`), the registry (`registry.rs`: `Registry::build`, `equation_for`, `used_by`, the dependency graph, `term_rows`; each result's upstream corrections are precomputed at build, which a unit test checks against a walk of the graph), display symbols (`symbols.rs`), the v1 scope (`scope.rs`, decision 31) and a plain-text rendering (`render.rs`) |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, `InputsView` (the design inputs or the assumptions), the assumptions banner, the sizing controls), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::show`, `show_over`, `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `note_ui`, `term_tag`, `reviewed_note`; its heights: `panel_heights`, `PANEL_SHARE`, `MIN_PANEL_HEIGHT`, `VIEW_STRIP`), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list, `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type; a click on its label traces the input), `dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`, `END_EFFECT_ROWS`, `greyed_by_end_effect`, `STORED_3D_ROWS`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), `result_groups.rs` (`result_groups`: the headline, the eight chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`), `history.rs` (`History`: undo and redo), `sizing.rs` (`SizingMode`, `SizingState`, `SizingRunner`: debounced inverse sizing), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match), `test_support.rs` (headless egui helpers, tests only) |
| `src/app.rs` | Feature `app`: `MagcouplingApp` (the panel as a full page; does its requests), `run_native`, `TITLE`, `CANVAS_ID`, `SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`; `app/theme.rs` (the linkage app's CAD dark visuals, forced dark), `app/files.rs` (saving: a file dialog natively, a download on the web; `DesignPicker`) |
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |

Not a Cargo workspace member: `linkage-sim-rs` depends on it by path
(feature `gui`) for its Tools → Magnetic coupling window ([In the linkage app](#in-the-linkage-app-m5)).

## Features and binaries

| Feature | Adds | Dependencies (all optional) |
|---|---|---|
| none (default) | The engine | none: pure std, builds for wasm32 as it is |
| `gui` | `gui::MagcouplingPanel`: the design inputs and results, and `fn ui(&mut self, ui: &mut egui::Ui)`. Any egui app can host it: the standalone app as a full page, the linkage app in an `egui::Window` (M5) | egui 0.32, egui_plot 0.33, serde_json, base64, flate2, log, rust_xlsxwriter 0.99.1 (the spreadsheet export; its `wasm` feature and js-sys on wasm32) |
| `app` | `app::MagcouplingApp` and the binaries below | `gui`, eframe 0.32, log, rfd 0.15; env_logger (native); wasm-bindgen, wasm-bindgen-futures, web-sys, js-sys (wasm32) |
| `workbook-parity` | **Test-only**: the switch that turns corrections off (see [Differences from the workbook](#differences-from-the-workbook)) | none |

### The panel

M4-1 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md`),
M4-2 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md`) and M4-3
(plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-3-explorer-assumptions-notes-pickers.md`).
A header with the session buttons; the inputs on the left; the dashboard on the right; the
centre region between them (`CentreView`, one tab row: the geometry view, the five plots, the
clamp and the results table; decisions M42-1 and M42-2), under the end-effect banner when
f_end <= 0. Every result is recomputed with `compute_all` (every correction on) each frame after the
inputs are drawn, so the readouts show the same frame's edits. Views stay in their own area: the
tab row and the banner are laid out in a child ui confined to the space above the Equation panel
(its max rect and its clip rect), and the view in one confined to the space left under them, so
no view paints into the panel or takes a click inside it at any window size or panel height
(egui's `TopBottomPanel::show_inside` only shrinks the cursor; a widget laid out later may still
run past the panel's top and paints over it). Each view fits itself to the height left: the
geometry drawing keeps 160 points while the callout list keeps three rows under it, then shrinks
(to scale), and the list scrolls in the rest; each plot takes the height its readouts line leaves
(no 150-point floor); the clamp view and the results table scroll in what is left (egui's 64-point
scroll-area floor lowered to none).

- **Inputs** (`inputs.rs`, `input_ui.rs`): every input, generated from its metadata, in the
  workflow order by default (decision O-1, `WORKFLOW`, O-2: Requirements and operating conditions
  first, the spec a designer fixes before choosing magnets (the torque, the temperatures, the
  space claim, the drive and the duty); then Magnets and rings, Gap and clearances, Housing and
  retainers, Shaft, key and clamps, Materials (the adhesive and its bondlines in one section),
  Thermal and demagnetization, and Calibration and model (the fields stored from a 3D run beside
  the 3D reference torques, in view: they must be refreshed after a geometry change); each group's
  rarely changed rows (O-4: the bedding clearances, the optional adapter and its joint, the clamp
  and screw factors, the screw classes, the slip-loss end factor, the model constants) under a
  closed Advanced heading) or, one toggle away ("Workbook groups"), grouped as the package groups
  them, each nested group under its heading; the order is a view of the session, in no design file
  or share link. A filter box matches label, path or workbook cell as the results search does and
  shows the matching rows alone, under their group and section (O-5). The Key design group sits on
  top
  (`KEY_DESIGN`: face gap, pole count, both magnet parts, the A-2 axial length override (blank
  by default), operating temperature, back iron, cup wall, conductance, measured drag), whose
  inputs also stay in their groups. A number is a slider with a value box (live while dragging,
  arrow-key nudges by the step, a logarithmic scale where flagged, values rounded to the step's
  decimals: decision M41-1; every default sits on its step grid except the vacuum permeability's
  two, an open item); edits are clamped to the slider range, typed values included, while a value
  already outside it (from a file or a link) is kept and flagged "outside the slider range"
  (M41-2). A selector is a drop-down; an optional input a checkbox and a slider, entered at the
  result it overrides, unrounded (`OPTIONAL_SEEDS`, M41-12: the axial length override at the
  ring's length in use moves nothing; the measured drag at the model's drag switches the thermal
  summary to the measured branch); a text input a text field with a note (library part or not,
  grade or not). Each row shows a dot when changed from the default, a
  reset button, and a tooltip with help, path, workbook cell, slider range and default. A tab
  row above them switches to the **Assumptions** view (decision M43-7). A click on a row's label
  traces the input (decision O-8, `trace.rs`): every explained result its value flows into (the
  registry's `downstream`; the registry explains 392 of the 1086 results, so an input such as the
  drive torque reaches none, and its banner says so) is framed wherever it is drawn, in the
  selection colour, with a banner and a Clear trace button at the foot of the inputs side (so a
  trace moves no row above it, and a second click on the label ends it; the locked free
  variable's label traces too); a click on a result (it still opens the Equation panel) traces
  the inputs it reads (`upstream_inputs`), unless an input's trace marks it or it has no equation
  record: the trace then stays. Each group heading counts the rows the trace frames.
- **Dashboard** (`dashboard.rs`): the 15 headline numbers in Python's order, the space claim
  badge and the end-effect flag (`DASHBOARD`). Green, amber or red badges come from the check
  verdicts (`verdict_level`). A value whose workbook cell the deviation registry ties to an applied
  correction carries the ids (`E3 E7 E8`) with the corrections, the at-defaults workbook and
  corrected values and the evidence in its tooltip (`corrections.rs`, M41-10). When f_end <= 0
  (audit M9) the rows computed from the pull-out (`END_EFFECT_ROWS`) are greyed without badges
  under the banner "End-effect model out of range"; the temperature rows that read the stored 3D
  fields carry "3D values from the workbook" (M3 comes after M4). `result_tooltip` is the hover
  text of every readout, keyed by result path, built only while the row is hovered
  (`DashboardLine::tooltip`). Over the rows, the material warnings that fire (below).
- **Results table** (`results_table.rs`): every result with label, value, unit, workbook cell and
  marker; by physics chain by default (decision O-6, `result_groups.rs`: the headline, that is the
  dashboard's 17 rows, open; then the closed chain groups: the A-3 chains torque, temperature,
  demagnetization, slip heating, clamps and geometry, each with the results of its nested groups
  that the explorer's scope leaves out (`CHAIN_PREFIXES`: the cold demagnetization limits, the
  slip temperatures, the metal design's clearances and reserves), and the adhesive and mass
  chains; then "Other results" by package, each result in the first group that lists it; a heading
  opens its group and shows the worst level of its checks, so a closed group shows a failing
  check; a search opens every group it matches, and a heading click does nothing until the search
  is cleared) or in the engine's order; each design check's value carries the dashboard's badge
  (`CHECKS`, the dashboard's verdicts and the other 23 checks, screens and warnings; while f_end <=
  0 the hot minimum and the recommended screw, computed from the invalid pull-out, carry none, as
  on the dashboard: `greyed_by_end_effect`), and "Failing checks only" lists the red then the
  amber ones alone (O-7); "Traced only" keeps the rows an input's trace frames (the table says the
  trace marks nothing only when it marks none of the rows the failing filter lets through, else
  that the search matches nothing), and a heading counts the traced rows among those shown;
  search by label, path or cell; CSV (`path,label,value,unit,cell`) and JSON (with the
  design that produced the results: in Torque -> Magnets the inputs with the free variable at the
  value shown) at full precision, a non-finite number written `+inf`, `-inf` or `NaN` (M41-15);
  "Export XLSX" saves the spreadsheet (below).
  The label column flexes (`column_widths`, M42-8: whatever the value, cell and marker columns
  leave, at least 120 points), so a ~930 px window shows each row's label and value without
  scrolling.
- **Spreadsheet export** (`spreadsheet.rs`, `xlsx.rs`; plan
  `docs/superpowers/plans/2026-10-03-magcoupling-xlsx-export.md`, decisions X-1 to X-12): "Export
  spreadsheet" in the header and "Export XLSX" beside the CSV and JSON exports save the design
  shown as `magcoupling-results.xlsx`, laid out as the page. Sheets: **Summary** (the title, the
  UTC export time, the app version, the share link as text (about 2,500 characters, over Excel's
  2,080-character hyperlink limit: paste it into a browser), the sizing state, the end-effect and
  assumptions banners, the dashboard's rows with their badges, the material warnings and the
  corrections applied); **Inputs** in the order the inputs side shows (workflow or workbook
  groups, with the section and Advanced headings, drawn by the page's own `InputGroup::plain_sections`,
  `advanced_sections` and `InputSection::heading_in`; flags for changed from default, assumption,
  Advanced and Key design, whose group is not repeated, so every input is on one row; notes for a
  selector's choice, a blank optional input or empty text, and the free variable Torque -> Magnets
  sets); **Results** by physics chain as the table groups them (its own `table_lines`; each heading
  with the worst level of its checks; per result the value, unit, check level (`entry_level`),
  workbook cell, path, corrections and, where a record exists, the equation as plain text); **Assumptions** (value, unit, workbook default,
  rationale, source). Values only: no formula, and nothing reads the file back (X-1). Numbers are
  number cells at full precision, a number that is not finite is `+inf`, `-inf` or `NaN` text as in
  the CSV, and a text longer than a cell's 32,767 characters (as Excel counts them, in UTF-16 code
  units: an emoji counts two) is cut at a whole character, saying how long it was. The Inputs,
  Results and Assumptions sheets freeze their header row and carry Excel's filter buttons on it
  (the Summary, read top to bottom, does neither); the columns are sized, the headings bold (a
  group heading on a light grey, E7E6E6, apart from the section headings inside it) and each check
  level filled with the standalone page's badge colour, a fixed dark-theme palette (green 5AC878,
  amber FF8F00, red FF0000). rust_xlsxwriter 0.99.1 writes
  it; on wasm32 its `wasm` feature is on, without which the workbook's creation time calls
  `SystemTime::now()` and panics in the browser. The host gets the file as bytes
  (`PanelRequest::SaveFile.contents` is a `Vec<u8>`).
- **Geometry view** (`geometry.rs`, `geometry_view.rs`; the default view, M42-2): the end view
  (the cup and its pocket, the outer blocks, the liner, the sleeve, the inner blocks on their
  flats or as arcs, the hub, the shaft and the key) and the upper half side view through the flat
  centres (cap, cup wall, web and boss; inside the cavity, each centred on it, the outer and inner
  blocks, the liner, the sleeve and the hub), both at one scale (M42-3), from the results of the
  design shown: the axial dimensions are the housing in effect (`housing.*`, A-2 decision A2-8),
  so a length-sized design grows. Tagged dimension callouts (each tag on a plate of the drawing's
  background, readable over the blocks), listed under the views: the face
  gap, the corner gap (along face 1's normal, from the dashed circle its inner corners sweep to
  the outer block's flat), the running clearance (with the sleeve-to-liner gap and the movement), and
  per axis the space claim (the axial stack against the overall length, the large-diameter stack
  against the bay, the rotating OD against the diameter). A gap below zero, the clearance below
  its target and an exceeded axis draw red, a value that is not a number amber (M42-4). The space
  claim is dashed (the diameter as a circle in the end view; the overall length, the bay and the
  diameter in the side view), an exceeded axis red. Under the callouts: the autofit wall (the
  wall rule's `materials.cup_wall_suggested_mm`, unless there is no back iron) and the two
  design checks while they hold (the cap thread below the cup body OD; the boss OD differing
  between Metal design and Shaft clamps: flagged only, no engine change; M42-9). A dimension and
  its text are readouts: hovering shows the result's hover text and equation (below). A
  piece holding a number that is not finite is not drawn (a note counts them per view), a claim
  line farther than `CLAIM_REACH` (10) times the pieces' extent is left off with a note (a design
  file can hold any finite number), and blocks are drawn for 2 to 200 poles (below 3 the pocket
  and the hub are drawn round: a 2-gon has no corners).
- **Plots** (`plots.rs`, egui_plot 0.33; M42-5, M42-6): torque against magnet temperature (the pull-out,
  the band of the variation allowance, the required minimum, the operating temperature and the
  governing limit), the gap and pole sweeps (each row by status: green nominal, amber below the
  hot minimum, red no fit, grey outside the end-effect model; the design shown as a marker; the
  required floor; the pole sweep's line is named for its rows' smallest fitting apothem, so the
  design's own marker can sit off it), slip heating over time (the estimate and the high case
  against the governing limit, the time to the limit marked) and torque against relative rotation
  over one pole pair (the pull-out point at the E7 angle). The series come from this frame's
  results through the engine's own closed forms (tests pin them to the engine's torques), only
  for the tab shown; when f_end <= 0 the torque plots are grey. Points and reference lines that
  are not finite are left out; a plot with nothing left says "Nothing to plot" (the
  torque-temperature plot names an empty axis, a minimum temperature at or past its end, apart
  from values that are not numbers), and a sweep counts the rows it left out in a note.
- **Clamp** (`clamp_drawing.rs`; M42-7): drawing.py's end and top views of the one-piece slotted
  clamp for the recommended screw (its texts, coordinates and limits; the cuts clipped to the
  boss; the screws needed, at most 50), both at one scale in the dark theme's pens (each view's
  title in a band above it, as matplotlib puts an axes title; a note that would run past the
  drawing's right edge moved back inside), or
  drawing.py's message when no screw size
  fits; then the clamp table: the Shaft clamps summary, the machining steps and the 'Clamp screw
  sizes' table with the recommended size's column in green, each value with its hover text.
- **Equation explorer** (`typeset.rs`, `readouts.rs`, `explorer.rs`; spec Addendum A2): every
  displayed value (the dashboard rows, the results-table rows, the geometry callouts, the clamp
  summary and screw table, and a readout line over each plot, `PlotKind::readouts`, M43-11) goes
  through `Readouts::show`: hovering it shows its hover text, then its equation typeset (stacked
  fractions, scripts, √ and ∈ drawn with strokes, a large Σ, `arg max`, delimiters scaled to their
  contents, a brace per `cases`, the `where` lines; a selector compared for equality (= or ≠)
  with a code as its choice's label, an ordering with its numbers (`5 ≤ N_h`), a screw row as its
  size; factors side by side as in `render::plain`), each term in its own colour (`TERM_PALETTE`,
  in the formula tree's order, a ninth term taking the first again, M43-3; the H under a Σ or an
  `arg max` is a term too, the harmonic set's selector `coupling.max_harmonic`, taken at the Σ
  before the body's terms as the registry lists them); from the next frame every value and input
  row of a term of that equation is framed in the term's colour, a Σ's term by the harmonics the
  design sums. A click opens the value in the
  **Equation panel**, docked at the bottom of the centre region and closed until then (its header
  button toggles it; M43-1, M43-2; it opens at 45 % of the region's height, at least
  `MIN_PANEL_HEIGHT` 120 points, and its drag stops `VIEW_STRIP` 160 points below the region's
  top so the tab row and a small view always keep their strip, except in a region too short for
  both, where the panel keeps its 120 points: `explorer::panel_heights`): the equation large, its label, value, workbook cell and the
  corrections it embodies (`corrections_upstream`; the dashboard and table markers stay M4-1's,
  M43-14), the term list (colour, symbol, value in the unit the formula reads, a selector's code
  with its choice's label after a colon, `5: 1, 3, 5 (workbook)`, label, and what the term is; the
  colour swatches fade while another equation's value is hovered, its terms being the ones marked
  on screen; the list scrolls sideways when the region is narrower than its rows, as the equation
  does), the breadcrumb (a long walk shows its last 8 crumbs after "…") and "used by". Clicking a
  term, in the equation (the H under a Σ included) or the list, drills
  into it; an input term instead opens its group (or the Assumptions view), scrolls its row into
  view once and frames it until another equation or term is opened or the panel is closed
  (M43-12); a result without a record shows its label, value and cell (M43-9). The registry is
  built once per process (`readouts::registry`, M43-5) and logs
  `magcoupling explorer: N equations`. egui's default fonts lack ϑ, the superscript minus and ∝:
  they are drawn as θ, ¯ and ~ (`glyph_safe`, M43-6), the markup and the notes unchanged; a test
  lays out every equation and checks every character against the fonts, and another checks every
  text run against the record's plain rendering (a choice label only where the formula compares
  for equality).
- **Assumptions** (panel.rs; spec Addendum A3): the inputs side's Assumptions view lists the 14
  assumptions (`engine::assumptions::ASSUMPTIONS`), each with its changed-from-default dot, its
  input rows (the inputs' own rows, so an edit there is an edit like any other; the inputs also
  stay in their groups), its rationale and its source. While any differs from its workbook
  default, the header shows "Assumptions modified: ..." beside "Reset to workbook defaults"
  (`assumptions::reset_to_workbook_defaults`: the design inputs stay; the reset can be undone).
  In the Equation panel an assumption term is tagged, with the dot when changed, a result a
  modified assumption flows into is tagged too, and the open equation names the modified
  assumptions it depends on (`Registry::term_style`, `modified_assumptions_upstream`).
- **Teaching notes** (`explorer.rs`, `diagrams.rs`; spec Addendum A4): the Explain toggle (off by
  default) shows under the open equation's value its reviewed note (`notes::note_for`; a note
  shown by id, a warning's or a start-here step's, goes through `explorer::reviewed_note`), with its
  watch-out line, its diagram painted with egui (the square wave and its harmonics, the flux paths
  with and without back iron, torque against angle, the end fringing, the demagnetization knee
  with load lines, first-order heating) and its sources (M43-10). "Start here" walks
  `notes::START_HERE`, each step opening its equation with Explain on (Previous, Next, Stop).
- **Pickers and warnings** (`pickers.rs`, `dashboard.rs`; spec Addendum A5, A6; M43-4, M43-8): a
  part material's drop-down shows each choice's library row on hover and sums up the material
  picked under it; under each magnet part's text field a "Pick a part" drop-down lists "Custom
  dimensions (manual)" and the 15 library parts (with every correction on: E3's remanence, E19's
  rating and grade), and under each grade's a "Pick a grade" drop-down lists "Blank" and the 17
  grades, each with its row on hover; the text fields stay for typed names. The material
  warnings that fire show at the top of the dashboard in their severity's colour, each with a
  "Why: <note>" link that opens its reviewed note on its own in the Equation panel, with links to
  the equations it explains.
- **Session** (`session.rs`, `history.rs`): undo and redo (buttons, Ctrl+Z, Ctrl+Shift+Z, Ctrl+Y)
  of every change to the design, one step per settled edit (a drag, a typed value, a part name
  typed letter by letter), one per arrow nudge and one for a held arrow key's whole auto-repeat
  run (M41-14): an edit is in progress while a pointer button is down, a key is held down or a
  text field of an input row has focus (the results search is no edit); reset all; save and load
  a design file; a share link copied to the clipboard. A link the web app opens at start-up is
  the session's start (`open_share_payload`: the first Undo keeps it). A host with its own undo
  can keep the keys (`set_keyboard_shortcuts(false)`: M5's linkage window). One format for files and links: JSON
  `{"format": "magcoupling-design", "version": 1, "inputs": {path: value}, "sizing": {...}}` with
  every input (M41-5) and the sizing state (M41-4); a link is that JSON deflated and URL-safe
  base64 in `?m=` (the linkage tool's scheme; about 2.5 kB at the defaults). Loading is all or
  nothing (M41-6): an unknown path, a wrong type, a code outside its choices, a newer version or
  a malformed sizing state changes nothing and the panel names every problem, the inputs' and the
  sizing state's. A later version that renames or removes an input path adds a
  `session::PATH_MIGRATIONS` entry and bumps `DESIGN_VERSION`, so older files and links keep
  opening.
- **Sizing** (`sizing.rs`, Addendum A1): the mode switch tops the Key design group. In
  Torque -> Magnets the free variable (axial length, magnets per ring, ring radius) and the target
  hot-low torque (2.5 N·m to start, M41-9) feed `sizing::solve`, which runs once the design has
  been still for 0.25 s and no edit is in progress, never per frame. The panel then shows the
  design with the free variable at the solved value, "Solved at X", or at the best value it
  found, "Not reachable (best Y at X)" (M41-8); the free variable's row shows that value, locked
  (under the status line when the Key design group does not list it: the ring radius). Leaving the
  mode keeps the value, solving first a change still waiting for its debounce (M41-7).
- **No I/O in the panel.** Saving and picking a file are `PanelRequest`s the host does
  (`take_requests`, then `load_design_file` and `report`), so the M5 linkage window can host the
  panel with its own file handling; the share link base is the host's (`set_share_base`).
- **Theme** (`app/theme.rs`): the standalone app applies the linkage app's CAD dark visuals and
  spacing, dark whatever the system prefers (M41-3; a test keeps the visuals equal to
  `linkage-sim-rs/src/gui/theme.rs`), and the web page's background is the same panel colour.
  The panel sets no theme: in M5 the host's applies.

**Versions.** egui and eframe use the same 0.32 line as `linkage-sim-rs`, so
M5 embeds the panel with one egui; `Cargo.toml` has caret ranges, and
`Cargo.lock` pins the versions of `linkage-sim-rs/Cargo.lock` (egui and eframe
0.32.3, egui_plot 0.33.0, wasm-bindgen 0.2.114, which is the `wasm-bindgen-cli`
version `deploy-web.yml` installs). Gate 11 fails when the two lock files or the CLI
pin disagree; bump them together.

| Binary | Target | Run |
|---|---|---|
| `magcoupling-app` | native | `cargo run --release --features app --bin magcoupling-app` (from `magcoupling-rs/`) |
| `magcoupling-web` | wasm32 | `bash linkage-sim-rs/scripts/build_magcoupling_web.sh` builds it into `linkage-sim-rs/web/magcoupler/`; `bash linkage-sim-rs/scripts/serve_web.sh [PORT]` serves it at `http://localhost:8080/magcoupler/`, next to the linkage app at `/linkage/`. `build_web.sh` builds both bundles. On a desktop, `cargo run --features app --bin magcoupling-web` opens the native window |

The web page is `linkage-sim-rs/web/magcoupler/index.html` (committed; the JS
glue and the wasm are gitignored build outputs). Its canvas id is
`app::CANVAS_ID`, which a test checks. `deploy-web.yml` runs
`build_web.sh`, which builds the linkage bundle and then runs `build_magcoupling_web.sh`, so both bundles ship
(colesorkness.com/magcoupler/; the old linkage.colesorkness.com/magcoupling/ links redirect there,
share links included) on the next push of `main`; pushing needs the user's
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it opens
`/magcoupler/` through a pinned share link and checks the canvas, the geometry view in the first
screenshot (the default view: no click), the console lines
`magcoupling: loaded the design from the share link`, `magcoupling sizing: Solved at` and
`magcoupling explorer: ` (the equation registry built in the browser), then
clicks "Load design" once (found in a screenshot), uploads a pinned design file through rfd's web
picker and checks `magcoupling: loaded a design file`, with zero console errors (a test decodes
the pinned link and the design file). Its other step checks the hub page at `/` and the linkage
app at `/linkage/`.

**workbook-parity never ships.** The feature reaches `cargo test` and
`cargo clippy --all-targets` through the self dev-dependency, and cargo then
unifies it into every unit of the build, binaries included, so a
`compile_error!` on `app` plus `workbook-parity` would break
`cargo test --features app`. The guard is
`linkage-sim-rs/scripts/magcoupling_shipped.sh` instead. It defines the shipped
cargo arguments once (`MAGCOUPLING_WEB_ARGS`, `MAGCOUPLING_NATIVE_ARGS`, and the linkage
app's `LINKAGE_WEB_ARGS`, which has no magcoupling-rs in it), and
`magcoupling_assert_shipped` reads cargo's `--message-format=json` record of
the units it compiled. It fails unless the shipped binary was compiled and no
magcoupling-rs unit has the feature. `build_magcoupling_web.sh` pipes its
release build through it, so the shipped path refuses such a bundle. Gate 10
runs the guard on the calculator's native and wasm32 builds, plus a negative control per
shipped build that must trip.

### In the linkage app

From M5 (plan `docs/superpowers/plans/2026-10-02-magcoupling-m5-embed.md`) to 2026-10-06 the
linkage app embedded the panel in a Tools → Magnetic coupling window. Since then the calculator
is a site of its own and the linkage app's Tools → Magnetic coupling calculator opens it in a new
browser tab (`linkage-sim-rs/src/gui/menu_bar.rs`, `MAGCOUPLER_URL`); `linkage-sim-rs` no longer
depends on this crate (plan
`docs/superpowers/plans/2026-10-06-colesorkness-url-move-and-standalone-calculator.md`). The old
`?tool=magcoupling` links redirect to the calculator's site.

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
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults, at the edge points (inputs `set` accepts that pin what no case is built to reach: an undefined outer cold limit, each clamp comparison exactly on its equality edge (head seat, screw axis, wall, grip, thread, span), the dashboard's two verdicts exactly on theirs (a cup wall of exactly the back iron needed, a residual target of exactly the minimum running clearance), each temperature-chain comparison exactly on its edge, its input read or searched from the defaults and the equality asserted at the point (the hot minimum and hot-day torque checks, each ring's rating check, which limit governs, each of the verdict's four terms alone with the other three holding, both ends of the FEA interpolation range, and the time to the limit's start-at-the-limit and never arms), and six points where a clamp count lies outside i64 (a friction of 0 or 1e-30, a zero or negative safety factor, a 1e300 mm clamp): there the record's screws needed or screws that fit is ±inf, NaN or past 2^63 and the engine holds its saturating `as i64` cast, the one exemption the guard makes, asserted live at every size and no wider) and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override, the coercivity from the inputs with two ferrite betas, the sleeve and cap materials, E17's free-space fields), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); with every chain explained, each of the 15 assumption inputs reaches an explained result and moves one at some design point (the A3 test is not vacuous); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it, and the "used by" list follows the workbook's chain (Calculator C35 into each ring's coefficient, the cold ring's skipping cold onset into the cold limit). Records evaluated on inputs that bypass `set` (an invalid harmonic code, selector codes outside their choices, a NaN temperature and a NaN reserve written into the struct) give a value or an `EvalError`, never a panic (`records_never_panic_on_inputs_that_bypass_set`). Every A4 note links to records and each equation has at most one note; `release_notes_are_reviewed` (ignored) is the M4 release gate. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300, with the default back iron and with none, where E17's free-space field inputs act (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a harmonic set outside its choices (NaN torques in the Calculator, the Calibration prototype and every sweep row, never another set, with the corrections off and on; `validate()` names it: `an_invalid_harmonic_set_is_nan_not_another_set`), a coercivity source outside its choices (any code but 1 uses the Hcj and beta inputs, and `validate()` names it: `an_invalid_coercivity_source_uses_the_inputs`), a positive beta typed in for magnets with no grade and no rating (no hot limit: C60 = +inf, C61 = NaN, the adhesive governs C12: `a_positive_beta_without_a_rating_has_no_hot_limit`), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), inverse sizing on designs no slider reaches (an outcome or an error for every free variable, never a panic: `sizing_never_panics_on_extreme_designs`), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells (hand-listed, or in the correction's golden file under `tests/data/deviations/`); a correction that changes more than 15 cells uses a golden file and one that changes 15 or fewer lists them (`broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`); each applied correction's figures match the report (`e1_adhesive_shear_modulus_matches_the_report`, `e2_clamp_screw_length_matches_the_report`, `e3_library_remanence_matches_the_report`), and help a correction rewords belongs to a real input and differs from the workbook's. A correction that changes nothing at defaults (E7 to E13, E15 to E20) carries registry probes, off-default inputs from the report whose cells are checked with the correction off and on (`each_probe_shows_its_correction`); `every_applied_engine_correction_is_visible_somewhere` requires every applied engine correction to show at defaults or in a probe. A probe's workbook value can be a workbook error (`Literal::Error`, e.g. `#DIV/0!` where Python raises): that side still runs but is not compared. E7, E8, E9, E10, E11, E12 and E13 must leave every default cell bit for bit (`e7_leaves_every_default_cell_bit_for_bit`, `e8_leaves_flat_blocks_bit_for_bit`, `e9_leaves_every_default_cell_bit_for_bit`, `e10_leaves_every_default_cell_bit_for_bit`, `e11_leaves_every_default_cell_bit_for_bit`, `e12_leaves_every_default_cell_bit_for_bit`, `e13_leaves_every_default_cell_bit_for_bit`). `e7_finds_the_peak_at_a_fill_of_exactly_0_4` pins E7 where the fifth harmonic vanishes (a ring at a fill of exactly 0.4). The Addendum A entries name every cell their probes change, downstream cells included (`addendum_entries_name_every_cell_their_probes_change`); E15 to E18 have rows in the Addendum A report and E19 and E20 cite decisions only (`e15_to_e18_have_audit_rows_and_e19_e20_decisions_only`); each reproduces the report: `e15_heat_capacity_matches_the_report`, `e16_removed_disc_matches_the_report`, `e17_aluminium_eddy_losses_match_the_report` and `e15_to_e17_together_match_the_reports_headline_table` (on top of E9, decision 15), `e18_aluminium_hub_mismatch_matches_the_report`, `e19_supermagnetman_arcs_follow_the_vendor_grid`, and for E20 `e20_each_part_uses_its_own_coercivity`, `e20_the_hcj_and_beta_inputs_override_the_grade_when_selected`, `e20_ferrite_is_limited_on_the_cold_side`, `e20_mixed_rings_use_the_weaker_grade` and `e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature`; `e15_leaves_every_default_cell_bit_for_bit` to `e20_leaves_every_default_cell_bit_for_bit` pin that none moves a default cell. `all_corrections_together_give_the_reviewed_headline` pins what users see: the headline with every correction on at the default design. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input, a 1280 x 1024 screen): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame; the value lands on the step's decimals (1.41) and stepping back lands on the default exactly; a typed value snaps to the step and is clamped into the range; pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept until edited and flagged; the changed dot and per-field reset; the axial length override starts blank and enters at the ring's length; a selector switches the branch (no back iron); a text input edits the part name and its note follows; every group opens and draws every input; the dashboard's stored-3D label, markers, end-effect banner (over the dashboard and the centre region) and space-claim overshoot; the results table's search and its export requests; undo and redo (buttons and shortcuts; one step per drag, per nudge, per typed name; Ctrl+Z in a text field left to the field; reset all undoable); save and load requests, a loaded and a refused design file, the share link on the clipboard and a broken link; the host's report; Torque -> Magnets: the solved design shown, the debounce (no solve per frame, none while a button is down, a repaint after a solve), the locked free-variable row (also under the status line for the ring radius), leaving the mode keeps the value as one undo step and solves a pending change first, an unreachable target shows the best value, the free-variable picker, a share link carries the sizing state, the JSON export holds the design shown, typing in the results search does not defer the solve; a held arrow key is one undo step, a slider's value box being typed in is an edit of the design (one step on Enter), the results search holds back no undo step, only this frame's rows count as design fields, a start-up share link is no undo step, a host can keep Ctrl+Z for itself; idle frames change nothing with every group open and off-grid values; the measured drag enters at the model's drag, unrounded; every drawn text and hover text has glyphs in egui's default fonts (every tab of the centre region and the geometry callouts' hover texts); the panel draws inside an `egui::Window` (M5); the geometry view is the default and follows the design shown (a live redraw; the sized design in Torque -> Magnets); each plot tab draws its plot, the design's marker at the edited pull-out in the edit's frame; the clamp tab draws the recommended clamp or drawing.py's message; a ~930 px window shows a row's label (filling its 120-point column) and value; the ordering (O-1 to O-8): every group of both orders opens and draws its rows and headings, the Advanced sections closed until their heading is clicked; the order toggle changes no input, design, share link or undo history; the filter box shows the matches alone (count, run headings; a matched row nudges; no undo step for typing; the workbook order's headings; none; blank); a focused advanced input opens its group and heading past a filter; the results by group (every heading drawn, a heading opens and closes its rows, the engine order without headings); the failing filter shows exactly the failing checks with four red and two amber badges; a click on an input's label frames exactly the dashboard rows it drives and its own row until a second click; a click on a result frames exactly the Key design rows it reads and counts the traced rows per group; the traced filter (disabled without a trace, the traced count, the headline's traced heading); a heading shows the worst level of its group's checks (the headline and the closed coupling model red, the mass none); a heading click while searching leaves its group as it was, its hover text saying why; the failing filter with a search no failing check matches says no result matches; an input whose results have no equation record traces none, says so, and the traced filter then says the trace marks nothing (Clear trace turns the filter off); reading a traced result keeps the input's trace and its traced rows, a result without a record keeps it, a result outside it replaces it; idle frames in the workflow order's filtered view change nothing; the final review's fixes: out of the end-effect range the table's headline rows carry exactly the dashboard's badges (none on the hot minimum and the screw) and the failing filter leaves the hot minimum out (seven red, four amber); a search that hides the traced rows says no result matches, and a heading counts the traced rows it shows; a trace moves no input row, so a second click at the same point ends it; the locked free variable's label traces it with no edit or undo step; views stay in their own area: on a 1280 x 620 page and the laptop screen, with the Equation panel at its starting height and dragged as far up as it goes, on a 1280 x 300 page and inside an `egui::Window` (the linkage Tools window's 1100 x 700), no view (geometry, the five plots, clamp, results) paints into the panel or takes a click inside it (shapes split from the panel's by its separator line, widgets by its resize handle), the scrolling views fill the space down to it, and the window keeps its size; the panel opens at about 45 % of the centre region and its drag stops `VIEW_STRIP` short of the region's top, then follows the pointer back down; every view draws beside the panel in a tiny window; both spreadsheet buttons (the header's and the results table's) queue `magcoupling-results.xlsx`, which reads back with the four sheets, the edited face gap as a number and the share link, follows the input order shown and, in Torque -> Magnets, holds the solved length (noted) and the solve's outcome |
| `src/gui/geometry.rs`, `src/gui/geometry_view.rs` (feature `gui`) | The end view draws every part from the results (the pocket's corners at the pocket corner radius, the rings at the retainers' diameters, the key from where it crosses the bore), faceted blocks on their flats and arcs as sectors; each end-view callout's line measures the gap it names (the corner gap along face 1's normal, from the dashed circle the inner corners sweep to the outer flat); red below zero and for a clearance below its target; the side view spans the housing in effect (a 50.8 mm override: the 53.6 mm cavity, the hub centred on it, the liner and sleeve the retainer span in effect); the space-claim callouts name each overshoot, in red, with the exceeded axes' dashed lines red (every axis at once too), exactly at the claim inside and the next number below over, and a NaN claim amber; a claim far past the pieces (1e300, 1e39, -1e39) is left off with one note, drawn at `CLAIM_REACH` times the extent and not past it; the design checks show while each inconsistency holds (equality is not below); the autofit hint unless there is no back iron; pole counts from a file draw at most 200 blocks; two poles draw their four blocks in a round pocket and hub (a 2-gon's corners would fly off), three a triangular pocket; a NaN draws nothing wrong and says so (the parts dropped counted per view). Painted: both views at one scale, the largest that fits (`side_by_side`: none without room or for an extent that is not finite); the cup OD, the shaft and the face gap at their pixel lengths, the cup depth and the cap as painted rectangles; the callout texts in their colours (the default clearance red, its line too); an overshoot red; hovering a dimension shows its result's hover text; each dimension tag sits on its own plate of the drawing's background; a NaN view and a tiny screen draw; a region of no size paints no view; a huge claim keeps the pieces to scale with its note; in a short region (400 points down to none) the view ends at its foot, the drawing keeping 160 points while the list keeps three rows, then shrinking to scale (`drawing_height`) |
| `src/gui/plots.rs` (feature `gui`) | The torque-temperature curve meets the engine's pull-out at the operating temperature and at 20 °C, its band edges the hot-low and cold-high torques (bit for bit), a grade ring's own alpha too; sweep rows by status and end-effect range (4, 2, 7 and 1, 1, 4 at the defaults; short magnets grey every row); slip heating rises to the steady temperatures and marks the time to the limit on the curve; the torque-rotation curve peaks at the pull-out (harmonics 1 to 11 too; an invalid set draws none); f_end <= 0 greys the torque plots; each plot paints its series with their point counts and legends (an all-greyed sweep: its markers and no line; the pole sweep's line named for its smallest fitting apothems); values that are not numbers draw without panicking; a sweep with no finite row says there is nothing to plot, one with some counts them in a note; an empty temperature axis (a minimum temperature at or past its end) is named as such, apart from values that are not numbers; in a short region each plot ends at its foot, and a region a few points tall or none draws |
| `src/gui/clamp_drawing.rs` (feature `gui`) | `fmt_g` matches Python's `:g`; clipping keeps what is inside the boss; the default clamp has drawing.py's texts, the boss and bore circles, every cut inside the boss and one screw axis; no fitting screw (and an index outside the table) gives drawing.py's message; drawing.py's screws needed are drawn whatever fits, at most 50; the screw table's rows follow the sheet; the tab paints both views at one scale, the summary, the machining steps and the screw table, drawing.py's message, the one-piece note, and a narrow or tiny region without panicking, and no room with no scale; at 540, 660 and 980 points wide every note and dimension text stays inside the drawing and clear of the view titles (the flange note moved back from the right edge); in a short region (300 points down to none) the view scrolls in exactly that space |
| `src/gui/inputs.rs`, `src/gui/dashboard.rs`, `src/gui/corrections.rs`, `src/gui/results_table.rs` (feature `gui`) | Every input in exactly one section in schema order, every heading used, the Key design list, every input type covered; step decimals; every slider default on its step grid except the listed two; the optional seeds (entering the axial length at its seed moves nothing); text hints; tooltips. The dashboard starts with `HEADLINE`; every verdict each check gives has a level (designs that reach each branch); `END_EFFECT_ROWS` are exactly the headline rows that move with c_end (0 included: it flips the hot-minimum verdict) at back iron 1 and 0, `STORED_3D_ROWS` exactly those that move with the 14 stored 3D inputs at their slider ends; greying drops badges. The compiled golden files are the registry's; the headline pull-out carries E3 with the workbook's 2.647. The table lists every result once; the search; exact numbers read back bit for bit; CSV quoting; `+inf` and `NaN` in CSV and JSON (E20's positive-beta design); the JSON export holds the design; the label column flexes down to its minimum; the hover text carries the correction marks. The ordering: every input in exactly one workflow section (a test lists any input without one), the groups in design order with the advanced sections last and every Key design input in an ordinary one; the filter matches label, path and cell by section in either order; every one of the 30 checks is a text result in schema order and every result named as a check is listed; every verdict of every other check is classified (a design per branch; the does-not-apply texts give no badge); the failing checks are the red then the amber; out of the end-effect range the two checks among the greyed rows give no level and are not failing, and every dashboard row's badge is its check's level in range and out; the empty table names the filter that empties it; every result in exactly one group, the headline first, then the eight chains, then the other results, a path in two places in the first, every package with a heading, every chain prefix placing a result the scope leaves out; the levels order by severity; the lines with only the headline open, a search shows only the groups it matches, the engine order and the failing filter are flat; an input traces the results downstream of it, a result the inputs upstream, a result without a record nothing, an input no explained result reads none (it says so; 392 of the 1086 results are explained, every input traceable); a trace counts the paths it marks; marked paths take the trace's colour under the equation's own; in a short region the results table scrolls in what its search and filter rows leave; a group's plain then advanced sections and their headings, as the inputs side, the filter and the spreadsheet draw them; a row's badge is its check's level (none for a check greyed by the end effect), a heading's the worst of its rows |
| `src/gui/spreadsheet.rs`, `src/gui/xlsx.rs` (feature `gui`) | The four sheets in order, the table sheets' column headers frozen with filter buttons (the Summary neither); every input on one row with its value, unit and cell in either order, in exactly the order the page draws them, under its group, section and Advanced headings; the flags (changed, assumption, Advanced, Key design) and the notes (a choice, a blank optional input or empty text, the sized variable); every result on one row and every heading as the table's grouped lines (`table_lines`) give them, each value at full precision with its corrections; each check's level, none for a check greyed by the end effect, and each heading's worst level; numbers that are not finite as text; the equation column exactly for the explained results; the Summary (export time, version, link, sizing, both banners, the dashboard's rows, the warnings, the corrections applied); the Assumptions sheet; the file read back with calamine cell for cell for three designs; frozen panes and filters where the sheet asks, every column's width, bold headings, the three fills and the group grey, and no formula in the XML; a text over 32,767 UTF-16 units (letters or emoji) cut at a whole character, saying its length; the fill colours; the clock |
| `src/gui/session.rs`, `src/gui/history.rs`, `src/gui/sizing.rs` (feature `gui`) | A design file and a share link round-trip bit for bit (every input type, the sizing state); a file names every input; a missing path or sizing state takes the default; every problem reported, nothing loaded, the inputs' and the sizing state's together; an older file's renamed and removed paths migrate (a made-up table) and `PATH_MIGRATIONS` leads to inputs of this version; not a design, a newer or malformed version, a malformed sizing state, a broken link, a link that inflates past 1 MB are refused; the default link stays under 2,500 characters, and the one 49fbd3f wrote (before the spreadsheet export's zlib-rs backend) still loads. Undo and redo walk settled steps, coalesce an unsettled edit, drop redo on a new change, cap at 100. The runner waits for the debounce and runs once, restarts on a new change or an edit in progress, ignores the free variable's own value, solves a pending change at once on request; solved, unreachable and refused outcomes and their status lines |
| `src/gui/format.rs` (feature `gui`) | Four significant digits, scientific outside 1e-3 to 1e6, carries (9.99996 shows as 10.00), signed zero, `+inf`, `-inf`, `NaN` (one text for the display and both exports), integers, text, None, units |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the CAD dark theme; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the visuals are `linkage-sim-rs`'s, function body for body; the theme stays dark when the system turns light |

### Regenerating test data

The slider ranges are defined once, in Rust. Three generators write the data
files; `cargo test` then compares:

| # | Generator | Writes |
|---|---|---|
| 1 | `MAGCOUPLING_BLESS=1 cargo test --test schema` | `tests/data/input_schema.json`: every input's metadata and slider range, read by generator 2 |
| 2 | `cd reference/magcoupling-py && ./.venv/Scripts/python tools/gen_differential.py` (`.venv/bin/python` on POSIX; `--check` only verifies, and is what the gate runs) | `tests/data/python_schema.json` (the Python metadata, with the table layouts), `tests/data/static_data.json` (the static tables) and `tests/data/differential/*.json` (one file per result group, `helpers.json` for `compat`, and `full.json`) |
| 3 | `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells` | `tests/data/deviations/E<k>.json`, the golden files of the broad corrections (E3, E4, E5); see [Deviations](#deviations). It writes what this engine computes with the correction alone, so review the diff |

Run 1, then 2, when an input, a range, a choice or a table layout changes; run 3
when a broad correction changes cells. The `gui-smoke` share link
(`MAGCOUPLING_SMOKE_PAYLOAD` in `.claude/workflows/gui-smoke.js`) carries every input (decision
M41-5): regenerate it with `MagcouplingPanel::share_link` (the default design, face gap 1.5 mm,
Torque -> Magnets on the axial length at 2.5 N·m) when the design format changes, a default
changes, or an input path is renamed or removed; `the_smoke_test_share_link_opens_its_design`
fails until then. A Rust-only input (declared with
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

## Equation explorer (Addendum A2 to A4)

`src/engine/explain/` explains results. Each record states one result from its terms in a
small markup that is both what the M4 equation panel typesets and what the drift guard
evaluates, so the equation shown is the one that produced the number. Its terms are input
and result paths (hover keys, colours, drill-down and slider links); unit, label and
workbook cell come from the result's metadata, and where a formula needs a value the engine
computed but did not expose, the engine exposes it as a Rust-only result (a read, no formula
change; one exception, decision G4: with E20 off the outer ring's demagnetization block is
computed anew for the per-ring results, and it governs nothing and moves no workbook cell). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. Formatted verdicts are stated in the markup too (`concat`, `fmt`, `fmtnum`), with `ceilto` and `floorto` for Excel's CEILING and FLOOR and the literals `inf` and `nan`, so no record needs a Rust closure. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). The A4 teaching notes are static data (`notes.rs`), each listing the equations it explains; the panel shows a note only once a physics reviewer has signed it off (`note_for`); 17 of the 17 are signed off (plan A-3 Task 16; six of them revised after an independent physics review on 2026-10-01 and signed off again). There are 17: the spec's list (harmonics, the back-iron factor, pull-out against angle, end effect, Br(T), demagnetization, slip loss and skin depth, the thermal time constant, clamp preload, and the physics behind each of the six A5 warnings) plus ferrite's cold side and the one-point calibration; `START_HERE` opens them in the spec's order (torque chain, back iron, temperature, demagnetization, slip heating, clamps). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`. The GUI side
(plan M4-3) typesets the records, hovers and opens them, shows the notes and paints their
diagrams: see [The panel](#the-panel).

| Chain (decision 31) | Records | Status |
|---|---|---|
| torque | `records/torque.rs` | explained |
| demagnetization | `records/demagnetization.rs` | explained |
| slip heating | `records/slip_heating.rs` | explained |
| temperature | `records/temperature.rs` | explained |
| clamps | `records/clamps.rs` | explained |
| geometry callouts (decision G1) | `records/geometry.rs` | explained |
| dashboard | `records/dashboard.rs` | explained |

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
