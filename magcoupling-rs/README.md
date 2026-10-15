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
bundle, served at [colesorkness.com/tools/magcoupler](https://colesorkness.com/tools/magcoupler/) (see [Features and binaries](#features-and-binaries)).
Addendum A-3 (explanations) complete: the engine side of the equation
explorer (`src/engine/explain/`: one evaluable markup per result, proven by the drift guard, over
decision 31's paths, the geometry callouts and the terms they need to reach inputs), the A3
traceability test over every assumption, and the 20 A4 teaching notes behind a physics-review gate.
M4-1 complete: every input generated from the metadata (the Key design group first), the
dashboard (badges, corrected-vs-workbook markers, the end-effect greying, the stored-3D label,
the space claim), the searchable results table with CSV and JSON export, undo and redo, design
files and share links, the sizing mode switch and the linkage app's theme (see
[The panel](#the-panel); decisions M41-1 to M41-15). M4-2 complete: the centre region's views,
the geometry view to scale (the first view until 2026-10-13: both rings, the housing in effect, the dimension
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
the results it drives (decisions O-1 to O-8). Guidance slice 1 complete (spec `docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md`, BL-055, decisions G-1 to G-8): one torque number, the Size and torque tab as the first view with an Auto solver, check guides on the dashboard and what-moves-what lines (see [The panel](#the-panel)). Next: guidance slice 2 (the note rewrite), then M3 (live 3D fields).

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
| `src/engine/meta.rs` | Field metadata: `inputs!`/`results!`, `param`/`out` builders, `Value`, get/set/visit of inputs and get/visit of results by dotted path (a table row's field as `table[i].field`); table rows (`rows!`, `TableLayout`, `col`/`at_row` builders) with synthesized workbook cells; `SliderRange::step_from` (one slider step, kept on the slider: the step a check guide's fix and a teaching note's change entry take) |
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
| `src/engine/clamps.rs` | `Shaft clamps` and `Clamp screw sizes` sheets: 25 input cells, `SCREW_SIZES`, `MACHINING_STEPS`, `ClampResults` (33 cells), the 165-cell screw table, `screw_class_name`; `PAUSED` (slice-2 ruling 11, 2026-10-15: the panel hides the section while it is true), `SECTION` (the paused path prefixes `clamps.` and `materials.screws.`) and `hides(path, paused)` |
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `OPTIONAL_SEEDS` (the result an optional input enters at, M41-12), `DesignInputs::validate` |
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28; with the axial length override set, `axial_housing` makes the hub length, the cup cavity depth and the retainer span follow the rings they bound, each at least its ring's length: decision A2-8) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, `space_claim_check`, and the hub length, cup depth and retainer span in effect) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve_in` (the search over a range the caller gives, its `FreeVariable::grid_in(range)` the scan's grid; `solve` is `solve_in` over the variable's own slider range, and the Size tab's ring-size Auto passes its diameter range as an apothem range: a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` (with `InvalidRange`: a range with a non-finite end or min > max) |
| `src/engine/size_controls.rs` | The Size and torque tab's engine side: `dimensions` (outside diameter, axial stack and metadata-derived ranges), `resize_diameter` and `resize_length` (with the explicit-minimum variants), no-ops and refusals; slice 1 adds `diameter_breakdown` (cap, cup and boss; the largest is the outside diameter), `length_breakdown` (magnets + housing = the overall length), `smallest_fitting_diameter` (`DiameterFit`: `At`, `AtMinimum`, `Nowhere`; bisection of `model::blocks_fit` to 1e-6 mm over the diameter range), `ring_radius_range` (that range as `coupling.inner_back_apothem_mm`, for ring-size Auto) and `with_ring_radius` (a ring radius applied through the Outside diameter slider's own mapping, so the cap follows the cup). The torque model is untouched |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults`; `Assumption::shown(clamps_paused)` (not the clamp friction or the preload fraction while the shaft-clamp section is paused) |
| `src/engine/picks.rs` | Guidance slice 2: `set_by_pick`, the pick input that sets an input in place of its own value on a design: a back-iron pick other than 4140 (a steel's density, conductivity, specific heat, expansion and modulus; the circuit selector for 304 and 6061; 1018's design flux density), a sleeve-and-liner pick other than 316L (its density, conductivity and specific heat), a ring's library part (its manual dimensions, Br and grade) or a grade on custom dimensions (the manual Br); `BACK_IRON_PICK`, `SLEEVE_LINER_PICK`. Its tests nudge every input it names, on every pick, and no result moves; every other input a pick leaves without effect is named, or listed (the steel circuit's stored slip fields, unused without steel) |
| `src/engine/explain/` | Addendum A2 equation explorer, engine side (plan A-3): the formula markup (`markup.rs`: grammar, parser, parse tree), its evaluator with a trace of what a result depends on (`eval.rs`), the static tables a formula reads (`tables.rs`: magnets, grades, the part materials, aluminium alloys, adhesives, screw sizes), the authoring form (`record.rs`) and the records, one file per chain (`records/`), the registry (`registry.rs`: `Registry::build`, `equation_for`, `used_by`, the dependency graph, `term_rows`; each result's upstream corrections are precomputed at build, which a unit test checks against a walk of the graph), display symbols (`symbols.rs`), live values in note text (`note_text.rs`: `segments`, `render`, `rendered_texts`, `placeholders`, `NO_VALUE`), the five probe designs and one slider step of an input (`probes.rs`: `probe_designs`, `PROBE_NAMES`, `step_input`, `Unmoved`), the v1 scope (`scope.rs`, decision 31), a plain-text rendering (`render.rs`) and the plain-words guard over on-screen text (`plain.rs`: `BANNED`, `plain_text_violations`, and the shared `word_count` and `sentence_count`; pure std so gate 4 runs it, re-exported by `gui::guidance`; slice 2, spec 3.4) |
| `src/gui/` | Feature `gui`: `panel.rs` (`MagcouplingPanel`: the layout, the session buttons and shortcuts, `PanelRequest`, `CentreView`, `InputsView` (the design inputs or the assumptions), the assumptions banner, the welcome strip, the sizing controls), `typeset.rs` (the equation typesetter: `layout`, `layout_equation`, `layout_symbol`, `Laid`, `Ink`, `TermColors`, `TERM_PALETTE`, `glyph_safe`, `equation_ui`, `laid_ui`, `term_at`), `readouts.rs` (the readout hook: `registry`, `Readouts::new(marks, show_details)`, `show_details`, `Readouts::show` and `show_over` (the hover text built with the frame's `show_details`), `mark`, `ReadoutEvents`), `explorer.rs` (the Equation panel: `Explorer`, `explorer_ui`, `note_ui` (a note drawn against a `NoteView`, returning the `NoteLink` clicked), `held_reason`, `term_tag`, `reviewed_note`; its heights: `panel_heights`, `PANEL_SHARE`, `MIN_PANEL_HEIGHT`, `VIEW_STRIP`), `diagrams.rs` (the notes' six diagrams: `diagram_shapes`, `diagram_ui`), `pickers.rs` (the material, part and grade pickers: `MATERIAL_PICKERS`, `picker_ui`, `choice_hover`, the property texts), `plain_text.rs` (the display-name and help overlay over the engine's metadata, keyed by path: `label_for`, `help_for`, `original_help`, `hint_for`, `display_label`; `LABELS` holds spec 3.5's canonical names, three of them constants the Size tab and the plots draw too: `REQUIRED_TORQUE`, `PULL_OUT_NOMINAL`, `HOT_LOW_TORQUE`; `HELP` and `HINTS` the slice-2 help), `geometry.rs` (the geometry view's drawing in millimetres: `geometry`, `View`, `Callout`, `Note`), `geometry_view.rs` (`geometry_ui`: both views to scale, the callout list, `Transform`, `side_by_side`, `arrowhead`), `plots.rs` (`PlotKind`, the series builders, `plot_ui`), `clamp_drawing.rs` (`clamp_drawing`, the port of drawing.py, `clamp_ui` with the clamp table), `inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `InputGroup` with `plain_sections`, `advanced_sections` and `has_advanced`, `InputSection::heading_in` (how the inputs side, the filter's runs and the spreadsheet order and head a group's sections), `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`), `input_ui.rs` (`input_row`, `slider`: one input row of any type; a click on its label traces the input; `weak_line` and `hint_line`: a wrapped weak line under a control, and the hint line `plain_text::hint_for` gives a Key design row or a Size-tab control), `dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`, `shown_lines` (the dashboard rows drawn while the shaft-clamp section is paused or not; `dashboard_lines` keeps every row for the exports), `END_EFFECT_ROWS`, `greyed_by_end_effect`, `STORED_3D_ROWS`, `STORED_3D_INPUTS`, `STORED_3D_SENTENCE`, `result_info`, `result_tooltip`, the readouts' hover text, `hover_text`, its text with the correction marks, and the material warnings: `warning_lines`, `warning_note`, `severity_level`), `corrections.rs` (`CorrectionIndex`: the corrected-vs-workbook markers from the registry and the golden files), `results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `empty_text`, `results_csv`, `results_json`, `column_widths`, `entry_level` and `worst_level` (a row's and a heading's badge), `RUST_ONLY`), `spreadsheet.rs` (the spreadsheet's layout: `sheets` of a `Snapshot` -> `Sheet` rows of `Cell`s with a `CellStyle`, `value_cell`, `level_cell`, the column headers), `xlsx.rs` (`spreadsheet_bytes`, `xlsx_bytes` with rust_xlsxwriter, `cell_text`, `MAX_CELL_CHARS`, `fill`, `GROUP_FILL`, `now_unix_s`, `XLSX_FILE_NAME`, `XLSX_MIME`), `result_groups.rs` (`result_groups`: the headline, the eight chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`), `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`, `target_mismatch_notice`), `history.rs` (`History`: undo and redo), `guidance.rs` (the guidance layer, pure functions: `CHECK_GUIDES`, `counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `all_fixes`, `FixCache`, `within_minimum_diameter`, `guide_notes` (`trade_offs`, `side_effects` and the unchecked costs), `design_status`, `Consequences`, `ConsequenceCache`, `magnet_count_line`, `plain_text_violations` (re-exported from `engine::explain::plain`)), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines and consequence lines), `sizing.rs` (`SizingMode`, `held_by_auto`, `SizingState` (the mode and the free variable; the target is `metal.required_min_Nm`), `SizingRunner`: debounced inverse sizing over a search range), `format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell), `test_support.rs` (headless egui helpers, tests only) |
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

**Size and torque** is the first view (decision G-5; it replaced the geometry view, M42-2).
A one-line strip over the tabs, "Set the size -> read the torque -> open any red check to see
why and what to change.", closes for the session (it is in no file, link or undo step). Adjust **Outside diameter**,
**Overall length**, **Magnet face gap**, and **Magnets per ring** to see the existing
model's expected and hot-low torque. Each control names what it moves: Outside diameter moves
the ring radius (its hover says so; the input row is **Ring radius**), Overall length the
magnet length of both rings; the readouts are **Required torque**, **Hot-low torque** and the
nominal pull-out. Diameter edits move the ring radius, cap OD and cap
thread together while preserving their signed allowances and fixed thicknesses; length
edits use the existing axial magnet override and housing rules. Diameter includes the cup,
cap and shaft boss. Existing input ranges bound these controls, and all fit, clearance and
build checks stay in the dashboard. Torque is unavailable outside the end-effect model's
range. **Detailed inputs** keeps every original input, and all results, equations and
exports remain available. **Auto: smallest [magnet length | ring radius | magnet count] that
meets the required torque** (the hot-low torque; it does not look at the other checks)
runs the inverse sizing on this tab: it locks the matching slider (Overall length, Outside diameter, Magnets per ring) at the solved value, or at the best value
under a red "Not reachable: best Y at X", and solves again 0.25 s after any other edit. Ring
radius searches the ring radii of this tab's diameter range, Minimum OD included, and shows the
solved ring as the diameter slider would set it: the cap OD and its thread follow the cup (their
detailed-input rows are locked at those values), and the outcome line, solved or not
reachable, names the Outside diameter. Turning Auto off keeps the value shown.
Opening the tab changes no mode; direct edits use the existing undo/redo history and
file/share format.

**Required torque** is a slider on `metal.required_min_Nm`, the requirement the dashboard's checks
compare against (decision G-2: there is one torque number; the Torque -> Magnets target was a
second one until 2026-10-13). Editing it is an ordinary design edit: undo, files and share
links carry it. **Hot-low torque** (`metal.torque_hot_low_Nm`) is shown with its gap to
Required torque, "short X N·m (-P %)" in red or "margin +X N·m (+P %)" in green, and **Pull-out
torque (nominal, at T)** is the nominal figure at the live operating temperature. While the
end-effect banner shows, Hot-low torque and the gap read "no prediction (magnets too short for
the end-effect model at this pole pitch)" instead of a number.

Under each of the five controls (Minimum OD, Outside diameter, Overall length, Magnet face gap
and Magnets per ring) a weak hint line of at most 12 words says what it does
(`gui::plain_text::HINTS`, slice 2; the gap's and the magnet count's are their Key design rows'
hints, the other three are keyed `SIZE_TAB_HINT_KEYS`, which are not input paths). The Minimum
OD hint replaced the line under the controls that said the same.

Under the controls, short lines say what moves what, each read from the engine (the neighbour
magnet counts from `compute_all` of the design with only `coupling.npole` changed; the
diameters and lengths from `engine::size_controls`), never typed. For **Magnets per ring**:
a first line that follows the Auto state (`guidance::magnet_count_line`): with Auto off, "The
ring radius stays the same when you change the magnet count; the cup OD below can still
change."; with Auto on the ring radius, "Auto resizes the ring when you change the magnet count;
the lines below keep the present ring radius."; with Auto on the magnet length, "Auto resets the
magnet length when you change the magnet count; the lines below keep the present length."; and
none with Auto on the magnet count, whose slider is locked. Then the pull-out and cup OD at N-2 and N+2 inside the slider range (a neighbour whose
blocks do not fit, or whose end effect is out of range, reads in words; one that fails only the
keyway still shows its torque, since the keyway does not depend on the count), and the
smallest outside diameter that fits N magnets (where only the Minimum OD floor stops it, the
line says how far it could go and that lowering Minimum OD reaches it). For **Outside
diameter**: "Set by: cap A, cup B, boss C mm", the largest being the outside diameter. For
**Overall length**: "Magnets L mm + housing H mm".

**Minimum OD** adjusts the lower end of the diameter slider without resizing the current
design. Lower it to explore diameters below the original 9 mm mounting-radius range.
The fixed shaft bore, boss, cap/thread limits and upper radius bound still apply. Magnet-fit
failures and a keyway breaking through the hub are flagged beside the torque, along with
the existing guidance for a thin wall past the keyway. These checks remain separate from
the exploratory torque calculation. The minimum is a view preference: files and share
links save the actual dimensions, and reopening a smaller design starts the limit at its
saved diameter. Reset all restores the original range. Auto's ring-size search uses the
range the diameter slider shows; the other inverse-sizing variables keep their bounds.

M4-1 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-1-gui-inputs-dashboard-session.md`),
M4-2 (plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-2-geometry-plots-clamp.md`) and M4-3
(plan `docs/superpowers/plans/2026-10-01-magcoupling-m4-3-explorer-assumptions-notes-pickers.md`).
A header with the session buttons and the **Show spreadsheet details** toggle (slice 2, S2-5:
off by default; a view choice, in no design file, share link or undo step, kept by loads, undo
and reset all; with it on, the hover texts add the path, the original spreadsheet cell and,
where the GUI's help replaced it, the original spreadsheet note, and the corrections that change
a value; the dashboard shows the correction ids under a one-line legend, the results table its
cell and marker columns, the Equation panel the cell and corrections lines and each term's
internal name on hover, and the Assumptions view each source; exports and search are
unchanged); the inputs on the left; the dashboard on the right; the
centre region between them (`CentreView`, one tab row: Size and torque, the first view since
2026-10-13 (G-5), the geometry view, the five plots, the clamp and the results table;
decisions M42-1 and M42-2), under the end-effect banner when
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
  retainers, Shaft and hub joint, Materials (the adhesive and its bondlines in one section),
  Thermal and demagnetization, and Calibration and model (the fields stored from a 3D run beside
  the 3D reference torques, in view: they must be refreshed after a geometry change); each group's
  rarely changed rows (O-4: the bedding clearances, the optional adapter and its joint, the clamp
  and screw factors, the screw classes, the slip-loss end factor, the model constants) under a
  closed Advanced heading) or, one toggle away ("By original sheet"), grouped as the package groups
  them, each nested group under its heading; the order is a view of the session, in no design file
  or share link. A filter box matches the on-screen name, the original label, the path or the workbook cell as the results search does and
  shows the matching rows alone, under their group and section (O-5). The Key design group sits on
  top
  (`KEY_DESIGN`: face gap, pole count, both magnet parts, the A-2 axial length override (blank
  by default), operating temperature, back iron, cup wall, conductance, measured drag), whose
  inputs also stay in their groups; under each of its rows a weak hint line of at most 12 words
  says what the input trades (`gui::plain_text::HINTS`, slice 2). A number is a slider with a value box (live while dragging,
  arrow-key nudges by the step, a logarithmic scale where flagged, values rounded to the step's
  decimals: decision M41-1; every default sits on its step grid except the vacuum permeability's
  two, an open item); edits are clamped to the slider range, typed values included, while a value
  already outside it (from a file or a link) is kept and flagged "outside the slider range"
  (M41-2). A selector is a drop-down; an optional input a checkbox and a slider, entered at the
  result it overrides, unrounded (`api::OPTIONAL_SEEDS`, M41-12: the axial length override at the
  ring's length in use moves nothing; the measured drag at the model's drag switches the thermal
  summary to the measured branch); a text input a text field with a note (library part or not,
  grade or not). Each row shows a dot when changed from the default, a
  reset button, and a tooltip with help (the overlay's where it has a row, `plain_text::help_for`), slider range and default (with Show spreadsheet details on, also the original spreadsheet note where the help replaces it, the path and the workbook cell). A row's label is its on-screen name (`plain_text::label_for`, spec 3.5: Ring radius, Magnets per ring, Magnet face gap, Required torque, Production variation allowance); the exports keep the original labels. A tab
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
  verdicts (`verdict_level`). With **Show spreadsheet details** on, a value whose workbook cell the
  deviation registry ties to an applied correction carries the ids (`E3 E7 E8`) under a one-line
  legend (`MARKER_LEGEND`), and its tooltip lists them under "Differs from the original
  spreadsheet:" with each title and, where the correction alone changes the cell at the default
  design, its was and now values, with no repo path (`corrections.rs`, M41-10); off, neither
  shows. When f_end <= 0
  (audit M9) the rows computed from the pull-out (`END_EFFECT_ROWS`) are greyed without badges
  under the banner "End-effect model out of range"; the temperature rows that read the stored 3D
  fields carry "Fixed 3D fields from the example design" (the fields are 3D results for the
  example design and stay fixed when the design changes, as their hover and every stored 3D
  input's help say: `STORED_3D_SENTENCE`, typed once in the engine as
  `temperature::stored_3d_sentence!`). `result_tooltip` is the hover
  text of every readout, keyed by result path, built only while the row is hovered
  (`DashboardLine::tooltip`). Over the rows, the material warnings that fire (below).
- **Check guides** (`guidance.rs`, pure functions; decisions G-3, G-6): a status line tops the
  dashboard, "Example design: fails N of 8 checks" while the inputs are the defaults (its hover:
  "The starting values from the original spreadsheet."; slice 2 amends G-6),
  "Your design: fails N of 8 checks" or "passes all 8 checks" after an edit, with ", N cautions"
  added when amber rows exist (counted live from the guides; while the shaft-clamp section is paused the clamp's guide counts toward nothing, so the line reads "of 7 checks"). The eight guides are the
  dashboard's seven checks (the hot minimum, the clearance, the cup wall, the thermal verdict,
  the recommended clamp, the space claim and the end-effect flag) and a **Magnets fit** row
  (`model::blocks_fit`, drawn beside `DASHBOARD`: it has no engine result of its own). Under a
  red or amber row a block gives the **Problem** (built from the live results), a one-sentence
  **Why** and a **Fix**: one or more inputs by their on-screen label and direction (for example
  "Raise Minimum outer return ring wall"; "Change" for a choice or a picker, and for a ring's
  magnet part taken to Custom dimensions with a hotter-rated grade), each a link to that input's
  row. A fix is listed only where its move helps and turns no passing check red; a failing check
  with none says so, and two failing checks that pull one input both ways each carry a note
  that says so. A fix that sets back another check that is not passing carries a note too
  ("Raising Magnet face gap also lowers the pull-out, so Hot-low torque falls further.";
  the thinner sleeve and liner, at no torque cost, are listed first). A thinner
sleeve or liner carries its own note, since no check covers how securely they hold the magnets.
The Magnets fit guide leaves out "Raise Outside diameter" where no diameter on the slider fits
the magnet count (the Size tab's fit line then says it does not fit at any diameter), and
"Lower Outside diameter" is left out of every guide while the Size tab's Minimum OD already stops
the slider (`guidance::within_minimum_diameter`). While the four reverse-field inputs hold their
defaults (3D results for the default NdFeB design), the temperature Problem says so beside the
cold demagnetization limit and the slip-onset magnet limit, and asks for the design's own. A fix
can name an input Auto holds; its link then frames a locked row (known limitation). The slip-fault
  cooling fixes (a shorter trip, more conductance) show only while the hot-day magnet
  temperature is below the limit the peak reaches: the peak is never below that temperature,
  so above it only a hotter magnet or another adhesive helps. Every row, whatever its level, has a **Why:** link: a reviewed teaching note (the hot-low, hot-minimum and cold-high rows lead to "Hot-low and cold-high: the torque band", the gearbox ripple row to "When the coupling slips"), an
  assumption's rationale (no row uses one since slice 2), or "no note yet". The guide is a GUI layer: the engine's verdict
  strings and the exports keep the workbook wording (`tests/python_schema.rs`); the dashboard,
  the results table and the Equation panel draw the cup wall's failing verdict with the wall
  input's label in place of its cell (ruling R9, `dashboard::verdict_display`), and the cup wall
  equation draws its verdict literal the same way (`plain_text::shown_literal`). Every Why and
  Problem sentence passes a physics review (G-3), and
  the numbers in them always come from the results.
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
  search by on-screen name, label, path or cell; CSV (`path,label,value,unit,cell`) and JSON (with the
  design that produced the results: in Torque -> Magnets the inputs with the free variable at the
  value shown) at full precision, a non-finite number written `+inf`, `-inf` or `NaN` (M41-15);
  "Export XLSX" saves the spreadsheet (below).
  The cell and marker columns are drawn only with Show spreadsheet details on (off, their room
  goes to the label; the search still matches cells and paths). The label column flexes
  (`column_widths`, M42-8: whatever the value, cell and marker columns
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
- **Geometry view** (`geometry.rs`, `geometry_view.rs`; the Geometry tab, the first view until 2026-10-13, M42-2; the Size tab draws it too): the end view
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
  diameter in the side view), an exceeded axis red. Under the callouts: the wall-rule line (the
  wall the cup wall check compares with, `materials.backiron_thickness_needed_mm`, rounded up at
  its last digit (`mm_at_least`), so a wall shown equal to it passes; not the failing verdict's
  suggestion rounded up to 0.1 mm; beside the cup wall input, by its label, unless there is no
  back iron) and the two design checks while they hold (the cap thread below the cup
  body OD; the boss OD differing between the housing and the clamp inputs; each names its inputs
  by their labels: flagged only, no engine change; M42-9). A dimension and
  its text are readouts: hovering shows the result's hover text and equation (below). A
  piece holding a number that is not finite is not drawn (a note counts them per view), a claim
  line farther than `CLAIM_REACH` (10) times the pieces' extent is left off with a note (a design
  file can hold any finite number), and blocks are drawn for 2 to 200 poles (below 3 the pocket
  and the hub are drawn round: a 2-gon has no corners).
- **Plots** (`plots.rs`, egui_plot 0.33; M42-5, M42-6): torque against magnet temperature (the
  nominal pull-out, the band of the production variation allowance, the required torque, the
  operating temperature and the governing limit), the gap and pole sweeps (each row by status:
  green nominal, amber below the required floor, red no fit, grey outside the end-effect model;
  the design shown as a marker; the required floor; the pole sweep's line is named for its rows'
  ring radius, the smallest the inner magnets and the keyed bore allow, so the design's own marker
  can sit off it), slip heating over time (the estimate and the high case against the governing
  limit, the time to the limit marked) and torque against relative rotation over one pole pair
  (the curve "Torque (nominal)", its peak the pull-out point at the E7 angle). The axes, legends
  and readouts use the on-screen names of the dashboard and input rows (Pull-out torque (nominal),
  Magnets per ring, Required torque). The series come from this frame's results through the
  engine's own closed forms (tests pin them to the engine's torques), only for the tab shown; when
  f_end <= 0 the torque plots are grey. Points and reference lines that are not finite are left
  out; a plot with nothing left says "Nothing to plot" (the torque-temperature plot names an empty
  axis, a minimum temperature at or past its end, apart from values that are not numbers), and a
  sweep counts the rows it left out in a note.
- **Clamp** (`clamp_drawing.rs`; M42-7; not offered while the shaft-clamp section is paused, below): drawing.py's end and top views of the one-piece slotted
  clamp for the recommended screw (its texts, coordinates and limits; the cuts clipped to the
  boss; the screws needed, at most 50), both at one scale in the dark theme's pens (each view's
  title in a band above it, as matplotlib puts an axes title; a note that would run past the
  drawing's right edge moved back inside), or
  drawing.py's message when no screw size
  fits; then the clamp table: the Shaft clamps summary, the machining steps and the 'Clamp screw
  sizes' table with the recommended size's column in green, each value with its hover text.
- **Shaft-clamp section paused** (owner, 2026-10-15; slice-2 ruling 11, task T12b; BL-064): "Get rid of the clamping with screw section for now. Put all of that on pause, we will likely use permanent shrink fit or other method." One switch, `engine::clamps::PAUSED` (true), hides the slotted clamp, the shaft key and the adapter-joint screws: every input and result under `clamps.` and the screw classes (`materials.screws.`, which only the clamp's screws read; `clamps::SECTION`, `clamps::hides`). While it is set the panel offers no Clamp tab (`CentreView::tabs`) and draws no clamp input group or section (`InputCatalogue::shown`: the workflow order's Shaft and hub joint group keeps the bore and the keyway depth, the original-sheet order loses its Shaft clamps group and the Materials group its screw classes; the filter finds none), no clamp dashboard row, guide or Why link, no clamp row in the results table (its search, failing filter, counts and heading badges included), neither clamp assumption (view and banner), no key in the geometry drawing and no clamp-boss design check, no Used by link into the section and no Start here step for `clamp_preload`, which stays in `NOTES` unchanged and is not shown (the Teaching notes bullet gives the counts). The clamp checks count toward no verdict: the status line counts 7 guides, a hidden check holds back no other guide's fix (`guidance::counts`, `guidance::counted_guides`) and the failing filter leaves them out. The engine still computes the section (Python parity unchanged), design files and share links keep every clamp input (a file written before the pause loads and round-trips), and the exports keep the full dump: the CSV, the JSON and the spreadsheet (Summary, Inputs, Results, Assumptions) list the section as before, since they have no single filter point and their tests pin them to every input and result. Texts that named the clamp as a cost now say what rises in its place, the shaft-joint and gearbox loads (the bore, manual Br and shaft boss helps, the magnet-length and overall-length hints, the cap material's help and `torque_from_shear`'s magnet-length entry); while paused no text the guard collects names the clamp: every drawn text, and the hover texts of inputs, dashboard rows, table rows, callouts and the named static hovers, plus the shown notes, the Start here walk and hover, the counted guides, the Why links' reasons and the shown assumptions (`panel::tests::every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`, which also checks their glyphs in both states). The panel's `clamps_paused` field starts from the switch, and the panel tests draw the whole section unless a test pauses it. Setting `PAUSED` to false brings the section back with no other code edit.
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
  both, where the panel keeps its 120 points: `explorer::panel_heights`): the equation large, its label, value and (with Show spreadsheet details) workbook cell and the
  corrections it embodies (`corrections_upstream`, titles on hover; the dashboard and table markers stay M4-1's,
  M43-14), the term list (colour, symbol, value in the unit the formula reads, a selector's code
  with its choice's label after a colon, `5: 1, 3, 5 (default)`, label, and what the term is; the
  colour swatches fade while another equation's value is hovered, its terms being the ones marked
  on screen; the list scrolls sideways when the region is narrower than its rows, as the equation
  does), the breadcrumb (a long walk shows its last 8 crumbs after "…") and "used by". Clicking a
  term, in the equation (the H under a Σ included) or the list, drills
  into it; an input term instead opens its group (or the Assumptions view), scrolls its row into
  view once and frames it until another equation or term is opened or the panel is closed
  (M43-12); a result without a record shows its label and value, and its cell with Show spreadsheet details (M43-9). The registry is
  built once per process (`readouts::registry`, M43-5) and logs
  `magcoupling explorer: N equations`. egui's default fonts lack ϑ, the superscript minus and ∝:
  they are drawn as θ, ¯ and ~ (`glyph_safe`, M43-6), the markup and the notes unchanged; a test
  lays out every equation and checks every character against the fonts, and another checks every
  text run against the record's plain rendering (a choice label only where the formula compares
  for equality).
- **Assumptions** (panel.rs; spec Addendum A3): the inputs side's Assumptions view lists the 14
  assumptions (`engine::assumptions::ASSUMPTIONS`), each with its changed-from-default dot, its
  input rows (the inputs' own rows, so an edit there is an edit like any other; the inputs also
  stay in their groups), its rationale and, with Show spreadsheet details, its source. While any differs from its workbook
  default, the header shows "Assumptions modified: ..." beside "Reset assumptions to defaults"
  (`assumptions::reset_to_workbook_defaults`: the design inputs stay; the reset can be undone).
  In the Equation panel an assumption term is tagged, with the dot when changed, a result a
  modified assumption flows into is tagged too, and the open equation names the modified
  assumptions it depends on (`Registry::term_style`, `modified_assumptions_upstream`).
- **Teaching notes** (`explorer.rs`, `diagrams.rs`; spec Addendum A4): the Explain toggle (off by
  default) shows under the open equation's value its reviewed note (`notes::note_for`; a note
  shown by id, a warning's or a start-here step's, goes through `explorer::reviewed_note`), with its
  diagram painted with egui (the square wave and its harmonics, the flux paths with and without
  back iron with each block lettered N or S, torque against angle with the fundamental alone
  dashed at 90° beside a sum with a strong third harmonic, the end fringing with the length lost
  at each end bracketed, the demagnetization knee with load lines, first-order heating with τ_th at 63 %, a limit and the time to reach it), its
  "What to change" entries, its "Watch out" lines and its sources under
  a "Where this comes from" fold that starts closed (M43-10; guidance slice 2). An entry reads
  "<Raise, Lower, Choose or Enter> <the input's display label>: <effect and trade-off>"; the label
  is a link that frames and scrolls to the input's row (`NoteLink`, `Explorer::follow_note_link`),
  or plain text with the reason where the input is not free on the design shown: "held by Auto",
  "set by the <pick> pick" (`explorer::held_reason`, `engine::picks::set_by_pick`), or at the top
  or bottom of its slider the entry's way (`SliderRange::step_from`). "Start here" walks
  `notes::START_HERE`, each step opening its equation with Explain on (Previous, Next, Stop); `START_HERE` is `notes::start_here_for(clamps::PAUSED)`, the whole order `START_HERE_ALL` less its last step, the clamp note's, while the shaft-clamp section is paused.
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
- **Sizing** (`sizing.rs`, Addendum A1; the Size tab's Auto since slice 1): Auto on the Size tab is Torque -> Magnets (the Key design group shows
  only a status line, the solve's status and a button to the tab). In it the free variable (axial length, magnets per ring, ring radius) and the
  requirement `metal.required_min_Nm` (decision G-2: the one torque number; 2.5 N·m, the
  workbook's hot minimum, to start) feed `sizing::solve_in`, which runs once the design has
  been still for 0.25 s and no edit is in progress, never per frame. The panel then shows the
  design with the free variable at the solved value, "Solved at X", or at the best value it
  found, "Not reachable: best Y at X" (M41-8; "No valid value of <size> in range" when no value counts;
  for the ring radius X is the Outside diameter). The free variable's row shows that value, locked
  (under the status line when the Key design group does not list it: the ring radius), as do the
  cap OD and thread rows the ring radius moves (`set_by_auto`; the spreadsheet notes the same rows). Turning Auto
  off keeps the value, solving first a change still waiting for its debounce (M41-7). A file or link
  still records `sizing.target_torque_Nm`, equal to the requirement; if a file's target differs from
  its `metal.required_min_Nm` the requirement is used and the status line says so.
- **No I/O in the panel.** Saving and picking a file are `PanelRequest`s the host does
  (`take_requests`, then `load_design_file` and `report`), so a host (the standalone app; the
  linkage window from M5 to 2026-10-06) brings its own file handling; the share link base is the
  host's (`set_share_base`).
- **Theme** (`app/theme.rs`): the standalone app applies the linkage app's CAD dark visuals and
  spacing, dark whatever the system prefers (M41-3; a test keeps the visuals equal to
  `linkage-sim-rs/src/gui/theme.rs`), and the web page's background is the same panel colour.
  The panel sets no theme: the host's applies.

**Versions.** `Cargo.toml` has caret ranges (egui and eframe 0.32, egui_plot 0.33).
`Cargo.lock` pins wasm-bindgen 0.2.114, the `wasm-bindgen-cli` version `deploy-web.yml`
installs to bind both web bundles; gate 11 fails when either crate's lock or the CLI pin
disagree, so bump them together. egui and eframe no longer have to match `linkage-sim-rs`
since the linkage app stopped embedding the panel (2026-10-06).

| Binary | Target | Run |
|---|---|---|
| `magcoupling-app` | native | `cargo run --release --features app --bin magcoupling-app` (from `magcoupling-rs/`) |
| `magcoupling-web` | wasm32 | `bash linkage-sim-rs/scripts/build_magcoupling_web.sh` builds it into `linkage-sim-rs/web/tools/magcoupler/`; `bash linkage-sim-rs/scripts/serve_web.sh [PORT]` serves it at `http://localhost:8080/tools/magcoupler/`, next to the linkage app at `/tools/linkage/`. `build_web.sh` builds both bundles. On a desktop, `cargo run --features app --bin magcoupling-web` opens the native window |

The web page is `linkage-sim-rs/web/tools/magcoupler/index.html` (committed; the JS
glue and the wasm are gitignored build outputs). Its canvas id is
`app::CANVAS_ID`, which a test checks. `deploy-web.yml` runs
`build_web.sh`, which builds the linkage bundle and then runs `build_magcoupling_web.sh`, so both bundles ship
(colesorkness.com/tools/magcoupler/; the old linkage.colesorkness.com/magcoupling/ links redirect there,
share links included) on the next push of `main`; pushing needs the user's
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it opens
`/tools/magcoupler/` through a pinned share link and checks the canvas, the Size and torque tab with its
geometry drawing in the first screenshot (the first view: no click), the console lines
`magcoupling: loaded the design from the share link`, `magcoupling sizing: Solved at` and
`magcoupling explorer: ` (the equation registry built in the browser), then
clicks "Load design" once (found in a screenshot), uploads a pinned design file through rfd's web
picker and checks `magcoupling: loaded a design file`, with zero console errors (a test decodes
the pinned link and the design file). Its other step checks the hub page at `/tools/` and the linkage
app at `/tools/linkage/`.

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
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly); every warning text reads in plain words (the guard, no audit or code names). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. Every label and rationale reads in plain words (no cell, correction, audit, decision or code name; the sources keep their cells), and the end-effect and calibration rationales state what was checked (within 2 % of free-space 3D for magnets 3.2 mm and longer; the prototype match compares the circuit, the magnets per ring and the part only).; only the clamp friction and the preload fraction leave the panel, and only while the shaft-clamp section is paused (`Assumption::shown`) |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown; slice 1: `solve_in` over a caller's range equals `solve` over the variable's own range (same results), `grid_in` is the scan's grid, and a range with a non-finite end or min > max is `SizingError::InvalidRange`. |
| `tests/size_controls.rs` | The Size tab's engine adapter: the envelope mapping to existing inputs, retained allowances, bounds, the Minimum OD variants, no-ops and refusals; slice 1: the diameter and length breakdowns equal what `dimensions` reports (the largest of cap, cup and boss is the outside diameter; magnets + housing is the overall length), `smallest_fitting_diameter` fits at its result and not 1e-3 mm below (flats and arcs, the predicate monotone in the radius) and reads `AtMinimum` and `Nowhere` at the range ends, `ring_radius_range` is the diameter range as an apothem range, and `with_ring_radius` applies a ring radius through the Outside diameter slider's own mapping (the cap follows the cup; out-of-range requests return `None`). |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults, at the edge points (inputs `set` accepts that pin what no case is built to reach: an undefined outer cold limit, each clamp comparison exactly on its equality edge (head seat, screw axis, wall, grip, thread, span), the dashboard's two verdicts exactly on theirs (a cup wall of exactly the back iron needed, a residual target of exactly the minimum running clearance), each temperature-chain comparison exactly on its edge, its input read or searched from the defaults and the equality asserted at the point (the hot minimum and hot-day torque checks, each ring's rating check, which limit governs, each of the verdict's four terms alone with the other three holding, both ends of the FEA interpolation range, and the time to the limit's start-at-the-limit and never arms), and six points where a clamp count lies outside i64 (a friction of 0 or 1e-30, a zero or negative safety factor, a 1e300 mm clamp): there the record's screws needed or screws that fit is ±inf, NaN or past 2^63 and the engine holds its saturating `as i64` cast, the one exemption the guard makes, asserted live at every size and no wider) and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override, the coercivity from the inputs with two ferrite betas, the sleeve and cap materials, E17's free-space fields), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); with every chain explained, each of the 15 assumption inputs reaches an explained result and moves one at some design point (the A3 test is not vacuous); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it, and the "used by" list follows the workbook's chain (Calculator C35 into each ring's coefficient, the cold ring's skipping cold onset into the cold limit). Records evaluated on inputs that bypass `set` (an invalid harmonic code, selector codes outside their choices, a NaN temperature and a NaN reserve written into the struct) give a value or an `EvalError`, never a panic (`records_never_panic_on_inputs_that_bypass_set`). Every A4 note links to records and each equation has at most one note; `release_notes_are_reviewed` (ignored) is the M4 release gate. `review_sheet` (ignored) prints the physics reviewer's sheet. Guidance slice 2: every note placeholder names a number result, has no unit typed after it, stands in a clause with the word "here" or the words "on this design" (whole words: "where", "there" and "upon this design" do not count) and renders at the five probe designs with no brace left (a title, sentence 0 and the sources hold none), the check shown to name each kind of mistake; every change entry's aim moves its result the way it says, one slider step at each probe design where its input steps, and is checked at one design at least (the check confirmed on a smaller gap, longer magnets and a hotter magnet, and shown to catch a false direction and to skip an input a pick holds); `every_note_reads_in_plain_words_at_every_probe_design` (ignored until every note is rewritten): no banned pattern and no audit id (`plain::audit_ids`) in any text at the five designs. Slice-2 note claims on the engine: `torque_from_shear`'s (on the example design a millimetre of magnet length adds more pull-out than a millimetre of ring radius, both positive; at 8 magnets per ring a 1 mm larger ring lowers the pull-out); `harmonics`' (summing past the fifth moves the pull-out under 2 % at the five probe designs, but by more than a twentieth off them, at the fewest magnets on the probes' 20 mm ring with both fills under one half, the narrow, widely spaced blocks the harmonic-set entry names, which states no bound on the change; with the harmonic set at 1 the pull-out stress σ equals the fundamental's peak A_1 at the five probe designs and on a 17.5 mm ring, and σ − A_1 has the sign of the 2D torque's change from the set 1 to the design's set, so the sentence that draws σ draws A_1 as "the fundamental alone" and words neither as a share of the other; the fundamental's stress at the summed curve's pull-out angle, σ_1, lies below σ on the example design and above it on a 17.5 mm ring, and on the 20 mm ring radius design σ_1 < σ < A_1; at the 20 mm ring radius design, both fills under one half, the third harmonic moves the pull-out by more than a tenth and its angle off half a pitch; on the example design a wider, equally thick outer block (B862) fits and raises the outer fill, the pull-out and the cold-high torque and lowers the magnet limit, while a wider, thinner one (B861) fits and lowers the pull-out); `back_iron_factor`'s (among the back-iron materials only 304 and 6061-T6 give the free-space circuit, and with no back iron every material does); `pullout_angle`'s (on the example design a same-size, stronger-grade inner block, B842-N52 for B842SH, raises the pull-out exactly by the inner Br ratio, keeps the calibration factor, raises the cold-high torque and is rated for less heat, while a stronger grade of another size, M5026, fits and lowers the pull-out and the cold-high torque; on the 304 back-iron design the same pick ends the bench factor, so the pull-out moves by the Br ratio times the factor ratio). Slice-2 note claims (T13): the end-effect and calibration notes' levers move what their entries say at the five probe designs, one slider step each through `probes::step_input` (a blank axial length starts from the inner ring's length), the end-effect levers' trade-off included (cold-high, and cold-high at the gearbox input); every millimetre of overlap adds the same torque; the end-effect check flags an overlap at or below c_end·τ_p and not just above; the short-overlap watch-out says that at short overlap f_end is least reliable and can read too high or too low, with no pole-pitch boundary and no one design named, because the example design at the magnet length slider's floor and the narrower B822 block on both rings of the example ring (10 per ring at 2.5 mm, 12 at the floor) all have a short overlap (below a third of a pole pitch, the test's own measure) with a prediction (f_end above 0, the end-effect check OK), where 3D reads the model low (the example design at the floor) or high (B822, and the example design with its own steel at 2.5 and 2.9 mm), and the example design's largest error with its steel lies above a third of a pole pitch; a prototype match ignores the assumed calibration factor and keeps the measured one whatever its ring radius, gap and magnet length; "matching" the prototype inputs to another magnet count gives a factor more than 40 % off the bench one. The torque-band and temperature notes (T14): their levers move hot-low, cold-high, the pull-out and the hot-day torque as their entries say at the five probe designs, and a hotter day also lowers the temperature margins (the hot-day, magnet-limit and adhesive ones, and the skipping onset's wherever the magnets have one) and raises none, and the hot-day start is exactly the ambient plus the driving rise; the band note's sentence 0 names hot-low, cold-high, production scatter, the operating and minimum temperatures and "expected", and hot-low lies below cold-high; hot-low is the pull-out less the allowance and cold-high the pull-out at the minimum temperature plus it; with the requirement 1 % either side of hot-low, hot-low passes just when the 20 °C pull-out reaches the requirement taken back to 20 °C; the gearbox output carries cold-high, its input sees it through the ratio and the efficiency, and longer magnets or a colder minimum raise it there; on the example design hot-low fails while the hot-day torque meets the requirement, a hotter day leaves hot-low unchanged and cold-high at the gearbox input is above the input rating; the coefficient input moves only a ring without a picked grade (a mixed design too); a grade with a coefficient nearer zero loses a smaller share of its torque when hot (not always fewer N·m), and SmCo, the nearest, starts below sintered NdFeB (on the Y30 design it starts above Y30). The slipping, slip-heating and slip-temperature notes (T15), at the five probe designs: the slip frequency is the pole pairs times the slip revolutions per second, and slower slipping gives slower swings; the drag stays under a tenth of the pull-out; the ripple at the gearbox input is the pull-out through the ratio and the efficiency, and cold-high at the input lies above it; the heat is the drag times the slip speed, the magnetic-steel hub, cup and web losses grow as the slip speed to the power 1.5 and the sleeve's, liner's, cap's and magnets' (and, with no magnetic steel, the hub's, cup's and web's) as its square; titanium, Inconel and PEEK sleeves and liners heat less than 316L, and a thinner sleeve or liner widens the running clearance; a measured drag replaces the loss used and its high case while the rows the notes call estimated keep the estimate, and unmeasured the high case is the estimate times its allowance; the lump starts at the hot-day start and rises toward P/G with τ_th = C/G (more heat capacity slows it and leaves the steady temperature), one slip event adds P·t/C and the life the slip duty times P/G, the peak is the hotter of the fault slip and one event plus that average and the limit margins read it, a better heat path shortens τ_th, and the time to the limit reads never exactly when the high-case steady temperature stays at or below the limit; a hot-day start at the limit keeps the peak above it at the shortest fault trip; the slip losses read fields stored from the example design at every probe design (the Y30 design has the example's losses); `when_slipping`'s sentence 0 names the gearbox and `thermal_time_constant`'s the time constant with its trap guard ("mass changes only how fast"), and every "steel" in `slip_heating`'s sentences is magnetic steel (a 304 back iron is steel but not magnetic); with a non-magnetic back-iron pick the hub, cup and web take the pick's own conductivity and specific heat (304's at its probe design), so the two notes' sources name non-magnetic body parts, not aluminium. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300, with the default back iron and with none, where E17's free-space field inputs act (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a harmonic set outside its choices (NaN torques in the Calculator, the Calibration prototype and every sweep row, never another set, with the corrections off and on; `validate()` names it: `an_invalid_harmonic_set_is_nan_not_another_set`), a coercivity source outside its choices (any code but 1 uses the Hcj and beta inputs, and `validate()` names it: `an_invalid_coercivity_source_uses_the_inputs`), a positive beta typed in for magnets with no grade and no rating (no hot limit: C60 = +inf, C61 = NaN, the adhesive governs C12: `a_positive_beta_without_a_rating_has_no_hot_limit`), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), inverse sizing on designs no slider reaches (an outcome or an error for every free variable, never a panic: `sizing_never_panics_on_extreme_designs`), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. Plain words (slice 2): every result's label and help passes the plain-words guard (`every_result_label_and_help_passes_the_plain_words_guard`); the engine's own wording (every Rust-only input and result, the results and the column a correction rewords, and the sweeps' swept-variable column, whose workbook header differs by sheet: `ENGINE_WORDED_COLUMNS`, which the header check skips) names no cell, correction, audit, decision or code and keeps help to 30 words in at most two sentences (`engine_owned_help_is_plain_and_short`, with its helpers checked by `the_text_helpers_find_what_they_name`: `tests/common`'s `provenance_words`, `code_names` and `plain_problems`, the three checks together, which this test, `tests/assumptions.rs` and `tests/material_links.rs` share); the harmonic set's default choice reads "(default)". |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells (hand-listed, or in the correction's golden file under `tests/data/deviations/`); a correction that changes more than 15 cells uses a golden file and one that changes 15 or fewer lists them (`broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`); each applied correction's figures match the report (`e1_adhesive_shear_modulus_matches_the_report`, `e2_clamp_screw_length_matches_the_report`, `e3_library_remanence_matches_the_report`), and help a correction rewords belongs to a real input and differs from the workbook's. A correction that changes nothing at defaults (E7 to E13, E15 to E20) carries registry probes, off-default inputs from the report whose cells are checked with the correction off and on (`each_probe_shows_its_correction`); `every_applied_engine_correction_is_visible_somewhere` requires every applied engine correction to show at defaults or in a probe. A probe's workbook value can be a workbook error (`Literal::Error`, e.g. `#DIV/0!` where Python raises): that side still runs but is not compared. E7, E8, E9, E10, E11, E12 and E13 must leave every default cell bit for bit (`e7_leaves_every_default_cell_bit_for_bit`, `e8_leaves_flat_blocks_bit_for_bit`, `e9_leaves_every_default_cell_bit_for_bit`, `e10_leaves_every_default_cell_bit_for_bit`, `e11_leaves_every_default_cell_bit_for_bit`, `e12_leaves_every_default_cell_bit_for_bit`, `e13_leaves_every_default_cell_bit_for_bit`). `e7_finds_the_peak_at_a_fill_of_exactly_0_4` pins E7 where the fifth harmonic vanishes (a ring at a fill of exactly 0.4). The Addendum A entries name every cell their probes change, downstream cells included (`addendum_entries_name_every_cell_their_probes_change`); E15 to E18 have rows in the Addendum A report and E19 and E20 cite decisions only (`e15_to_e18_have_audit_rows_and_e19_e20_decisions_only`); each reproduces the report: `e15_heat_capacity_matches_the_report`, `e16_removed_disc_matches_the_report`, `e17_aluminium_eddy_losses_match_the_report` and `e15_to_e17_together_match_the_reports_headline_table` (on top of E9, decision 15), `e18_aluminium_hub_mismatch_matches_the_report`, `e19_supermagnetman_arcs_follow_the_vendor_grid`, and for E20 `e20_each_part_uses_its_own_coercivity`, `e20_the_hcj_and_beta_inputs_override_the_grade_when_selected`, `e20_ferrite_is_limited_on_the_cold_side`, `e20_mixed_rings_use_the_weaker_grade` and `e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature`; `e15_leaves_every_default_cell_bit_for_bit` to `e20_leaves_every_default_cell_bit_for_bit` pin that none moves a default cell. `all_corrections_together_give_the_reviewed_headline` pins what users see: the headline with every correction on at the default design. |
| `src/engine/explain/plain.rs` | The plain-words guard (slice 2, spec 3.4, R2), run by gate 4: every `BANNED` pattern has an example that reports it, a text reports each of its patterns once in `BANNED`'s order, and the slice-1 six come first; the clamp's screw sizes and every magnet part and grade name (read from `SCREW_SIZES`, `MAGNET_LIBRARY` and `GRADES`), size designations (M41 x 0.5), the input group headings ("Raise Boss outside diameter (Shaft clamps)") and words such as "skeptical" stay clean; `word_count` counts the runs between whitespace and `sentence_count` the stops that end a sentence ("0.1 mm" and "1.35e6" do not), the two counters every cap test shares (A-16); `audit_ids` names the audit-id prefixes a text holds (`TC-`, `PA-`, `CG-`, each once, in that order; not a date, the letters without the dash or in lower case), which the guard does not know and every note text is held to as well (A-13) |
| `src/engine/clamps.rs` | The paused section hides exactly the `clamps.` and `materials.screws.` paths and only while paused; each of its 28 inputs (25 clamp inputs, 3 screw classes), at every probe design, at both slider ends or every choice, moves no result outside it. |
| `src/gui/panel.rs` (feature `gui`) | Headless egui (`egui::Context::run` with injected input, a 1280 x 1024 screen): an arrow key on the face-gap slider and a click on its rail update the headline in the same frame; the value lands on the step's decimals (1.41) and stepping back lands on the default exactly; a typed value snaps to the step and is clamped into the range; pole count steps by two and stays even; range ends stop arrow keys; idle frames change no input; an input outside its slider range is kept until edited and flagged; the changed dot and per-field reset; the axial length override starts blank and enters at the ring's length; a selector switches the branch (no back iron); a text input edits the part name and its note follows; every group opens and draws every input; the dashboard's stored-3D label, markers, end-effect banner (over the dashboard and the centre region) and space-claim overshoot; the results table's search and its export requests; undo and redo (buttons and shortcuts; one step per drag, per nudge, per typed name; Ctrl+Z in a text field left to the field; reset all undoable); save and load requests, a loaded and a refused design file, the share link on the clipboard and a broken link; the host's report; Torque -> Magnets: the solved design shown, the debounce (no solve per frame, none while a button is down, a repaint after a solve), the Size tab's locked Auto slider (Overall length, Outside diameter or Magnets per ring, at the solved value; with ring-size Auto the cap OD and thread rows are locked too, following the cup), leaving the mode keeps the value as one undo step and solves a pending change first, an unreachable target shows the best value, the free-variable picker, a share link carries the sizing state, the JSON export holds the design shown, typing in the results search does not defer the solve; a held arrow key is one undo step, a slider's value box being typed in is an edit of the design (one step on Enter), the results search holds back no undo step, only this frame's rows count as design fields, a start-up share link is no undo step, a host can keep Ctrl+Z for itself; idle frames change nothing with every group open and off-grid values; the measured drag enters at the model's drag, unrounded; every drawn text and hover text has glyphs in egui's default fonts (every tab of the centre region and the geometry callouts' hover texts); the panel draws inside an `egui::Window` (M5); the Geometry tab follows the design shown (a live redraw; the sized design in Torque -> Magnets); each plot tab draws its plot, the design's marker at the edited pull-out in the edit's frame; the clamp tab draws the recommended clamp or drawing.py's message; a ~930 px window shows a row's label (filling its 120-point column) and value; the ordering (O-1 to O-8): every group of both orders opens and draws its rows and headings, the Advanced sections closed until their heading is clicked; the order toggle changes no input, design, share link or undo history; the filter box shows the matches alone (count, run headings; a matched row nudges; no undo step for typing; the workbook order's headings; none; blank); a focused advanced input opens its group and heading past a filter; the results by group (every heading drawn, a heading opens and closes its rows, the engine order without headings); the failing filter shows exactly the failing checks with four red and two amber badges; a click on an input's label frames exactly the dashboard rows it drives and its own row until a second click; a click on a result frames exactly the Key design rows it reads and counts the traced rows per group; the traced filter (disabled without a trace, the traced count, the headline's traced heading); a heading shows the worst level of its group's checks (the headline and the closed coupling model red, the mass none); a heading click while searching leaves its group as it was, its hover text saying why; the failing filter with a search no failing check matches says no result matches; an input whose results have no equation record traces none, says so, and the traced filter then says the trace marks nothing (Clear trace turns the filter off); reading a traced result keeps the input's trace and its traced rows, a result without a record keeps it, a result outside it replaces it; idle frames in the workflow order's filtered view change nothing; the final review's fixes: out of the end-effect range the table's headline rows carry exactly the dashboard's badges (none on the hot minimum and the screw) and the failing filter leaves the hot minimum out (seven red, four amber); a search that hides the traced rows says no result matches, and a heading counts the traced rows it shows; a trace moves no input row, so a second click at the same point ends it; the locked free variable's label traces it with no edit or undo step; views stay in their own area: on a 1280 x 620 page and the laptop screen, with the Equation panel at its starting height and dragged as far up as it goes, on a 1280 x 300 page and inside an `egui::Window` (the linkage Tools window's 1100 x 700), no view (geometry, the five plots, clamp, results) paints into the panel or takes a click inside it (shapes split from the panel's by its separator line, widgets by its resize handle), the scrolling views fill the space down to it, and the window keeps its size; the panel opens at about 45 % of the centre region and its drag stops `VIEW_STRIP` short of the region's top, then follows the pointer back down; every view draws beside the panel in a tiny window; both spreadsheet buttons (the header's and the results table's) queue `magcoupling-results.xlsx`, which reads back with the four sheets, the edited face gap as a number and the share link, follows the input order shown and, in Torque -> Magnets, holds the solved length (noted) and the solve's outcome; slice 1 (BL-055): the panel opens on Size and torque under the welcome strip, which closes for the session and leaves no undo step; the dashboard's status line counts the failing checks at the default design and after a fix; the Required slider edits `metal.required_min_Nm` (one undo restores it) and the Hot-low gap follows it; the consequence lines follow an input edit; Auto locks the matching slider, re-solves after the debounce, leaves idle frames alone and reports an unreachable requirement; opening the Size tab no longer changes the sizing mode; every new string has glyphs in egui's default fonts; slice 2 (BL-056): the Show spreadsheet details toggle starts off, a click turns it on and off, its hover says what it shows, and it is in no design file, share link or undo step and survives a loaded file, undo, reset all and a share link opened as the session's start; off, the face-gap row's hover names no path or cell and the pull-out's hover on the dashboard and on the Size tab no path, cell or correction, on they do; the correction ids and their legend, the results table's cells (pinned tests switch the details on), the Equation panel's cell and corrections lines and a result-without-record's cell, and the assumptions' sources show only with the details; at 930 px the table's label and value stay on screen with the cell and marker columns and without them; the toggle's label and hover, the legend, the marker heading and scope, the original-note prefix and the example design's status start and hover pass `plain_text_violations`; every text, the details' included, has glyphs in the default fonts; one name per quantity (`size_controls.rs` tests too): the Size tab's readout labels equal the input and dashboard rows' names (Required torque, Hot-low torque; one Hot-low torque readout in the centre beside the dashboard row of that name), the Auto row and its hover name the required torque, the footnote and the Outside diameter and Overall length hovers say what they move (the ring radius; both rings' magnet length) within the help caps, each label shows its hover, the open equation's header is the display label, and Auto's no-valid-value line names the ring radius; slice 2, help batch A (T7): each Key design row draws its hint line under it and above the next row (once: its own group open adds none), each of the five Size-tab controls its hint line under its label and above the next control's, and at a 930-point window every hint stays inside the inputs side, at least one wrapped; the panel's own strings (the assumption reset, the order toggle, the workflow headings, the trace banners and hint, the explorer's no-equation text and tag, the end-effect banner and greyed hover, the geometry notes, the welcome strip) pass the plain-words guard and name none of the engine's internal words; with Show spreadsheet details on, the dashboard keeps its width frame after frame (each value and its correction ids flow as one wrapped text); with Auto on (Torque -> Magnets, the axial magnet length free), Start here's step-1 note draws its magnet-length entry as plain text " (held by Auto)" once, a click on that label opens no row and keeps the note open, and its live values are the design shown's: the pull-out watch-out, which moves with the magnet length, reads the solved length's numbers and not the stored inputs' (Review Focus RF-1); the shaft-clamp pause (ruling 11): the Clamp tab and the clamp inputs (the workflow order's clamp sections, the original-sheet order's Shaft clamps group, the filter's matches) show only while the section is not paused, its two assumptions leave the Assumptions view and the banner, the shipped panel starts from the switch and passes its pause to the fixes (at the clamp's limit the hot-low torque's fix is drawn only while paused), the header's banner and the trace banner (`the_panel_passes_its_pause_to_the_fixes_the_banner_and_the_trace`), a design file written before the pause and a share link keep their clamp values in a paused panel (the engine still computes the section); one test, `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`, walks `every_shown_text` (every drawn text, and the hover texts of inputs, dashboard rows, table rows, callouts and the named static hovers; the rows through the panel's own filters) once per state: every such text has glyphs paused or not, and while paused none of them, nor a shown note's fields and drawn change entries, the Start here walk and hover, a counting guide's name and Why, a Why link's reason or a shown assumption's name, rationale and source, names the clamp (the guard; no allow-list entry; no shown note's change entry or equation leads into the section; with the section shown the same enumeration names the clamp and reaches the Clamp tab, the clamp dashboard row, a clamp input row, the Shaft clamps group and the clamp friction assumption); panel tests draw the whole section (`Harness::of` clears the pause) unless a test pauses it; slice 2 (T15): the gearbox ripple row's Why link opens the slip note, its first sentence drawn once and no input changed |
| `src/gui/geometry.rs`, `src/gui/geometry_view.rs` (feature `gui`) | The end view draws every part from the results (the pocket's corners at the pocket corner radius, the rings at the retainers' diameters, the key from where it crosses the bore), faceted blocks on their flats and arcs as sectors; each end-view callout's line measures the gap it names (the corner gap along face 1's normal, from the dashed circle the inner corners sweep to the outer flat); red below zero and for a clearance below its target; the side view spans the housing in effect (a 50.8 mm override: the 53.6 mm cavity, the hub centred on it, the liner and sleeve the retainer span in effect); the space-claim callouts name each overshoot, in red, with the exceeded axes' dashed lines red (every axis at once too), exactly at the claim inside and the next number below over, and a NaN claim amber; a claim far past the pieces (1e300, 1e39, -1e39) is left off with one note, drawn at `CLAIM_REACH` times the extent and not past it; the design checks show while each inconsistency holds (equality is not below); the wall-rule line unless there is no back iron (the cup wall input named by its label), asking for the wall the cup wall check compares with, so the first slider step that passes is not asked for the verdict's rounded-up suggestion; that wall rounded up at its last digit, so a wall shown equal to it passes (12 poles, a 1.23 T design flux density and a 1.95 mm wall fail the check, and the line asks for 1.951 mm); a minimum length rounds up at its last shown digit (past a power of ten too, and a length already on a digit stays), and one not finite or not positive shows as it is; pole counts from a file draw at most 200 blocks; two poles draw their four blocks in a round pocket and hub (a 2-gon's corners would fly off), three a triangular pocket; a NaN draws nothing wrong and says so (the parts dropped counted per view). Painted: both views at one scale, the largest that fits (`side_by_side`: none without room or for an extent that is not finite); the cup OD, the shaft and the face gap at their pixel lengths, the cup depth and the cap as painted rectangles; the callout texts in their colours (the default clearance red, its line too); an overshoot red; hovering a dimension shows its result's hover text; each dimension tag sits on its own plate of the drawing's background; a NaN view and a tiny screen draw; a region of no size paints no view; a huge claim keeps the pieces to scale with its note; in a short region (400 points down to none) the view ends at its foot, the drawing keeping 160 points while the list keeps three rows, then shrinking to scale (`drawing_height`); a dimension's hover and a callout's in the list under the drawing name their result's path only with the spreadsheet details; the design checks name their inputs by their labels and pass the plain-words guard; the key and the clamp-boss design check only while the shaft-clamp section is not paused |
| `src/gui/plots.rs` (feature `gui`) | The torque-temperature curve meets the engine's pull-out at the operating temperature and at 20 °C, its band edges the hot-low and cold-high torques (bit for bit), a grade ring's own alpha too; sweep rows by status and end-effect range (4, 2, 7 and 1, 1, 4 at the defaults; short magnets grey every row); slip heating rises to the steady temperatures and marks the time to the limit on the curve; the torque-rotation curve peaks at the pull-out (harmonics 1 to 11 too; an invalid set draws none); f_end <= 0 greys the torque plots; each plot paints its series with their point counts and legends (an all-greyed sweep: its markers and no line; the pole sweep's line named for its rows' smallest ring radius the inner magnets and the keyed bore allow; the legends and axes use the on-screen names (the pull-out and required torque as the dashboard and input rows, the torque-rotation curve "Torque (nominal)" since only its peak is the pull-out, the band as the production variation allowance, the pole sweep's axis Magnets per ring, the sweeps' amber status "Below the required floor"), each plain words with glyphs); values that are not numbers draw without panicking; a sweep with no finite row says there is nothing to plot, one with some counts them in a note; an empty temperature axis (a minimum temperature at or past its end) is named as such, apart from values that are not numbers; in a short region each plot ends at its foot, and a region a few points tall or none draws; a readout's hover names its path only with the spreadsheet details |
| `src/gui/clamp_drawing.rs` (feature `gui`) | `fmt_g` matches Python's `:g`; clipping keeps what is inside the boss; the default clamp has drawing.py's texts, the boss and bore circles, every cut inside the boss and one screw axis; no fitting screw (and an index outside the table) gives drawing.py's message; drawing.py's screws needed are drawn whatever fits, at most 50; the screw table's rows follow the sheet; the tab paints both views at one scale, the summary, the machining steps and the screw table, drawing.py's message, the one-piece note, and a narrow or tiny region without panicking, and no room with no scale; at 540, 660 and 980 points wide every note and dimension text stays inside the drawing and clear of the view titles (the flange note moved back from the right edge); in a short region (300 points down to none) the view scrolls in exactly that space; a summary line's hover and a screw table cell's name their path only with the spreadsheet details |
| `src/gui/inputs.rs`, `src/gui/dashboard.rs`, `src/gui/corrections.rs`, `src/gui/results_table.rs` (feature `gui`) | Every input in exactly one section in schema order, every heading used, the Key design list, every input type covered; step decimals; every slider default on its step grid except the listed two; the optional seeds (entering the axial length at its seed moves nothing); text hints; tooltips. The dashboard starts with `HEADLINE`; every verdict each check gives has a level (designs that reach each branch); `END_EFFECT_ROWS` are exactly the headline rows that move with c_end (0 included: it flips the hot-minimum verdict) at back iron 1 and 0, `STORED_3D_ROWS` exactly those that move with the 14 stored 3D inputs at their slider ends; greying drops badges. The compiled golden files are the registry's; the headline pull-out carries E3 with the workbook's 2.647. The table lists every result once; the search; exact numbers read back bit for bit; CSV quoting; `+inf` and `NaN` in CSV and JSON (E20's positive-beta design); the JSON export holds the design; the label column flexes down to its minimum; the hover text carries the correction marks. A tooltip opens with the overlay's help; the dashboard draws each row under its display label while its line keeps the catalogue label for the Summary export; the results search finds a result by either label; a renamed Key design input draws its display label, never its catalogue label too. The ordering: every input in exactly one workflow section (a test lists any input without one), the groups in design order with the advanced sections last and every Key design input in an ordinary one; the filter matches the display label, the catalogue label, path and cell by section in either order (and "gearbox" still exactly the three gearbox inputs); every one of the 30 checks is a text result in schema order and every result named as a check is listed; every verdict of every other check is classified (a design per branch; the does-not-apply texts give no badge); the failing checks are the red then the amber; out of the end-effect range the two checks among the greyed rows give no level and are not failing, and every dashboard row's badge is its check's level in range and out; the empty table names the filter that empties it; every result in exactly one group, the headline first, then the eight chains, then the other results, a path in two places in the first, every package with a heading, every chain prefix placing a result the scope leaves out; the levels order by severity; the lines with only the headline open, a search shows only the groups it matches, the engine order and the failing filter are flat; an input traces the results downstream of it, a result the inputs upstream, a result without a record nothing, an input no explained result reads none (it says so; 392 of the 1086 results are explained, every input traceable); a trace counts the paths it marks; marked paths take the trace's colour under the equation's own; in a short region the results table scrolls in what its search and filter rows leave; a group's plain then advanced sections and their headings, as the inputs side, the filter and the spreadsheet draw them; a row's badge is its check's level (none for a check greyed by the end effect), a heading's the worst of its rows; slice 2 (S2-5): an input's tooltip adds the original spreadsheet note, the path and the cell, and every result's hover the path, the cell and the corrections, only with the spreadsheet details (off is the same text without exactly those lines); the marker hover says "Differs from the original spreadsheet:" with each title, was and now values at the default design and what they are, and no repo path; the dashboard draws the correction ids and their legend, and the results table its cell and marker columns, only with the details, the status line staying first; a results-table row's hover names its path, cell and corrections only with the details; the readouts build every hover text with the frame's setting; the Equation panel's cell and corrections lines (titles on hover) and a term's internal name on hover show only with the details; the example design's status line says on hover where its values come from; the production-variation assumption is named as its input row (Production variation allowance); the stored-3D label and note say the fields are the example design's and stay fixed, in plain words, and every stored 3D input's help ends with that one sentence (`STORED_3D_INPUTS`, `STORED_3D_SENTENCE`); the note's sentence that names the slip-heating fields never says they set a limit, since at their slider ends they move only the temperature verdict, while the reverse fields move the governing limit and the hot-day margin; every input's, result's and dashboard row's hover text with Show spreadsheet details off passes the plain-words guard; the cup wall's failing verdict is drawn with the wall input's label and the engine's number on the dashboard and in the results table (ruling R9, amendment A-9; the badge reads the engine text and the CSV row keeps it; a malformed or non-finite number falls back to the engine text); a trace banner counts one in the singular; the shaft-clamp pause: the inputs side's catalogue keeps every other input in both orders and drops the emptied sections and groups (the shaft group keeps the bore and the keyway depth); the paused dashboard draws no clamp row, Problem or Why and its status line passes a design that fails only the clamp ("passes all 7 checks"); the failing list, the search, the counts and the heading badges leave the section out while paused (the headline's badge is red for a design failing only the clamp only while it shows); no shown row carries, and no shown equation embodies, a correction of the section alone (E2) |
| `src/gui/spreadsheet.rs`, `src/gui/xlsx.rs` (feature `gui`) | The four sheets in order, the table sheets' column headers frozen with filter buttons (the Summary neither); every input on one row with its value, unit and cell in either order, in exactly the order the page draws them, under its group, section and Advanced headings; the flags (changed, assumption, Advanced, Key design) and the notes (a choice, a blank optional input or empty text, the sized variable); every result on one row and every heading as the table's grouped lines (`table_lines`) give them, each value at full precision with its corrections; each check's level, none for a check greyed by the end effect, and each heading's worst level; numbers that are not finite as text; the equation column exactly for the explained results; the Summary (export time, version, link, sizing, both banners, the dashboard's rows, the warnings, the corrections applied); the Assumptions sheet; the file read back with calamine cell for cell for three designs; frozen panes and filters where the sheet asks, every column's width, bold headings, the three fills and the group grey, and no formula in the XML; a text over 32,767 UTF-16 units (letters or emoji) cut at a whole character, saying its length; the fill colours; the clock; every export keeps the paused shaft-clamp section (the CSV and JSON rows, the JSON design's inputs, the Inputs and Results sheets' rows and the Summary's dashboard row) |
| `src/gui/session.rs`, `src/gui/history.rs`, `src/gui/sizing.rs` (feature `gui`) | A design file and a share link round-trip bit for bit (every input type, the sizing state); a file names every input; a missing path or sizing state takes the default; every problem reported, nothing loaded, the inputs' and the sizing state's together; an older file's renamed and removed paths migrate (a made-up table) and `PATH_MIGRATIONS` leads to inputs of this version; not a design, a newer or malformed version, a malformed sizing state, a broken link, a link that inflates past 1 MB are refused; the default link stays under 2,500 characters, and the one 49fbd3f wrote (before the spreadsheet export's zlib-rs backend) still loads. Undo and redo walk settled steps, coalesce an unsettled edit, drop redo on a new change, cap at 100. The runner waits for the debounce and runs once, restarts on a new change or an edit in progress, ignores the free variable's own value, solves a pending change at once on request; solved, unreachable and refused outcomes and their status lines; slice 1 (BL-055): the sizing state holds only the mode and the free variable, the target being `metal.required_min_Nm`; a file's `target_torque_Nm` that differs from its requirement loads with the requirement, and with `target_mismatch_notice` only in Torque -> Magnets (a forward-mode file or link gets none: older files wrote the default target, which that mode never used); file and link round trips stay exact; the runner keys on the inputs and the search range; the ring radius and the magnet count read as their input rows' display labels, and Auto offers "ring radius" |
| `src/gui/guidance.rs` (feature `gui`) | `CHECK_GUIDES` covers every dashboard check and the Magnets fit row, and each guide names a real check; every fix path is an input with a slider or choices, or a picker input; every `Learn::Note` is a reviewed note, every `Learn::Assumption` exists and every `Learn::None` states its reason, and every dashboard row has a Why link; each Why is one sentence of at most 30 words; no guide or consequence text matches `plain_text_violations` (`engine::explain::plain`, re-exported here: cell references, correction ids, "workbook", "Addendum", "Rust-only", "Calculator!", and from slice 2 "M1 audit", "skeptic", "Plan A-", numbered decisions, `drawing.py` and `.rs` names, the sheet names Metal design, Temperature design and Shaft clamps where they point into the spreadsheet, "Calibration!", "correction E" with a number and whole-word M-number ids other than the clamp's screw sizes and size designations); each Problem line quotes the live results at the default design and none shows at a passing design; every shown fix moves each failing term toward passing and turns no passing check red; no cooling fix shows for a slip-fault peak whose limit the hot-day temperature already reaches, and the face-gap note shows exactly while the gap fix is shown beside a hot-low torque that is not passing (physics review 2); the sleeve and liner note shows with their fixes; "Raise Outside diameter" is not offered where no diameter on its slider fits; the cold limit names its source only while the reverse fields are the defaults (physics review 3); the consequence numbers equal `compute_all` of the neighbour design and follow an input edit; neighbours outside 4 to 40 magnets are not shown; the design status counts at the default and after a fix; no guide, fix, Problem or consequence text says "ring size" (the input row's name is Ring radius).; the shaft-clamp pause: only the clamp's guide stops counting, and only while paused (`counts`); a design that fails only the clamp passes all 7 counted checks while paused and fails 1 of 8 when not; a paused clamp offers no fix and holds back no other guide's fix (at the clamp's limit the hot-low torque's longer-magnet fix is withheld while the clamp shows and offered while it is paused); the fix cache computes again when the pause changes. Slice 2 (T14): the hot-low, hot-minimum and cold-high rows lead to the torque-band note, which explains each row's equation; no dashboard row, guide or check compares a gearbox torque with the gearbox rating (the note's watch-out says so until BL-059). Slice 2 (T15): the gearbox ripple row's Why link leads to the slip note, which owns its equation; that note's first watch-out also says nothing checks the gearbox. |
| `src/gui/explorer.rs`, `src/engine/explain/notes.rs`, `src/engine/explain/note_text.rs`, `src/engine/explain/probes.rs`, `src/engine/picks.rs` (guidance slice 2) | A note draws its parts in order (what it is, What to change, Watch out in the warning colour, the Where this comes from fold, closed until clicked, then holding the sources); a free input's display label is a link to its row and the entry's text is not; an input Auto holds or a pick sets is plain text with the reason; an entry at the end of its slider its way says so (a count too; a choice and a blank optional number never); `held_reason` names Auto and the pick, in plain words; a note's link frames the input's row and keeps the note open; every note shows with glyphs, its sources once the fold is opened (`panel.rs`). A step reads as its word (Raise, Lower, Choose, Enter); an aim's sense holds only for a move its way (`Sense::holds`: strictly, never for an unmoved result or a NaN; the one rule the aim test and the review sheet share); the change rules name each kind of mistake (an entered number or a choice names no aim; a step names the result it moves, except a model assumption's) and hold for every note's entries; the one cap rule (`cap_problems`) names each kind of mistake, and the slice-2 caps (3 to 5 sentences of at most 25 words, 1 to 5 change entries, 0 to 2 watch-outs of at most 30 words) and the plain sentence 0 are ignored until every note is rewritten (T18); the before snapshot reads back and holds the 17 notes slice 2 started from. Start here follows slice 2's order (spec 3.3, `SLICE_2_ORDER`; a step whose note a later task adds is skipped until then), `torque_from_shear` first and owning the area-and-lever chain (the 2D torque, area times lever arm, gap radius, pole pitch, overlapping length); the walk's first step reads "How the coupling makes torque" and the button's hover (`START_HERE_HOVER`) is plain words with that topic first; steps 2 to 6 (`harmonics`, `back_iron_factor`, `pullout_angle`, `end_effect`, `calibration`) each say what to change, `harmonics` owns the fill rows and `calibration` the prototype's modelled torque (`calibration.model_torque_Nm`); step 7, `br_temperature`, owns the three coefficient rows, and step 8, the torque band (`hot_low_cold_high`, owning the band's rows), follows it in the full order and in the order shown while the clamp section is paused; step 9, the slip note (`when_slipping`, owning the gearbox ripple and the slip frequency), comes between the torque band and `slip_heating` in both orders, and `slip_heating` also owns the field frequency, the pole pairs and the slip speed in rad/s, `thermal_time_constant` the steady temperatures, the rise per slip event and the peak; the notes listed in `REWRITTEN` (`torque_from_shear`, `harmonics`, `back_iron_factor`, `pullout_angle`, `end_effect`, `calibration`, `br_temperature`, `hot_low_cold_high`, `when_slipping`, `slip_heating` and `thermal_time_constant` so far; each note task adds its ids) already keep to the slice-2 caps and a plain sentence 0 (`the_rewritten_notes_keep_to_the_slice_2_caps`, every problem named at once; T15's three notes also per id, in A-15's form: `the_slip_notes_keep_the_caps`). At the defaults the outer library part sets the block's width and thickness, so the `harmonics` and `back_iron_factor` entries on them read "set by the Outer magnet part pick"; on custom dimensions they are links (Review Focus RF-4). `set_by_pick`: only the library parts' manual fields at the defaults; each material pick's inputs; no input it names moves a result on any pick; every input a pick leaves without effect is named; a code outside the choices sets nothing. `SliderRange::step_from` keeps a step on the slider. Live values: `live_value` reads as the readouts and gives none where there is no prediction (a text such as "n/a", a number that is not finite, an input, no result, or a result computed from an out-of-range end effect, as for magnets at or below the end-effect length); a note draws its live values from the design shown, "no prediction" for the ferrite probe's hot onset; every note draws at every probe design with glyphs and no brace; `notes_for_review` (ignored) writes the per-note review sheet. `note_text`: text splits into literals and placeholders, each fault is named, values are rendered in place, placeholders are listed in drawing order. `probes`: the five designs are what their names say, each computes, counts for inverse sizing and draws on the Size tab; a step moves one input one slider step (a count by its step, a blank optional number from its seed) or names why it cannot (a choice or an entered number, no such input, a pick holds it, not a number, at its end). The plain-words rule for a note (`plain_words_problems`) names each kind of mistake, an audit id in a source or a watch-out among them, and holds for the notes listed in `REWRITTEN`; the Equation panel's value line draws the cup wall's failing verdict with the wall input's label, and a passing wall as the engine gives it (ruling R9); the cup wall equation, in the Equation panel and every readout's hover, draws its verdict literal with the wall input's label while the record keeps the engine text (`typeset.rs`); the shaft-clamp pause: `START_HERE_ALL` ends with the clamp step and `start_here_for` drops exactly it while paused (`START_HERE` is the order for `clamps::PAUSED`), only `clamp_preload` is not shown while paused (`shown_when`, `shown`), the Start here hover names the clamp topic only while the section shows, and no shown equation reads a term of the section or links to it in its Used by |
| `src/gui/format.rs` (feature `gui`) | Four significant digits, scientific outside 1e-3 to 1e6, carries (9.99996 shows as 10.00), signed zero, `+inf`, `-inf`, `NaN` (one text for the display and both exports), integers, text, None, units (the sweep tables' "Nm" written N·m on screen; the metadata and exports keep "Nm") |
| `src/gui/plain_text.rs` (feature `gui`) | Every overlay key is an input or a result (a hint's an input or one of the Size tab's own hint keys), once, with text; no help row repeats the engine's; the nine canonical names of spec 3.5 and no other row, the Size tab's gap and count names equal to the input rows'; `label_for` falls back to the metadata label and never to the path, and never lengthens a label; the help mechanism on a stand-in table (the overlay, else the engine; the original is the spreadsheet's own note, kept only where it is not empty: the deviation registry's workbook text over a help a correction reworded, else the engine's help) and the public accessors over every input's own engine help; every label, help and hint passes the plain-words guard and the glyph test, calls no torque guaranteed, holds no brace and keeps its caps (help <= 30 words in 1-2 sentences, hint <= 12 words in one); no display label is another input's or result's name. Every input has help from the engine or the overlay (`every_input_has_help_from_the_engine_or_the_overlay`); `overlay_strings_for_review` prints the overlay for the physics review. Help batch A (T7, `help_tests`): every input of the Requirements and operating conditions, Magnets and rings, Gap and clearances and Shaft and hub joint groups has help; hints exist exactly for the Key design inputs and the three `SIZE_TAB_HINT_KEYS` (none of them an input path); batch A help and every hint carry no digit; no hint types an input label (catalogue or display); the review sheet lists every input of its groups and every hint; `help_for_review` (ignored) writes the G-3 sheet (`HELP_REVIEW_GROUPS`, `HELP_REVIEW_OUT`). Help batch B (T8, `batch_b_tests`): the 52 empty helps of the Calibration and model, Materials, Housing and retainers, and Thermal and demagnetization groups each have a row (the set is computed from `WORKFLOW`), and 31 non-empty helps are replaced (the grade-mode density, rating and coefficient helps, the stored 3D fields, placeholder and assumed values, and helps with a cell or correction id or a stale fact), none of them Rust-only; the toggle's original note is the spreadsheet's own text (the deviation registry's for a help a correction reworded, none where that is empty); 172 direction claims hold, each a result that rises, falls or stays when a batch-B input rises one slider step on a named design (the example, no back iron, a 1018 back iron, a 304 back iron, a titanium sleeve, a 7075 cap, screw classes 10.9 and A4-70, the coercivity inputs, a measured drag, manual magnets with and without Y30 grades), and so do the outside-diameter (the boss or the cap sets it only as the largest part), cap-resize, cup-wall, measured-drag, magnet-mass and temperature-check claims; each stored-3D, prototype, unmeasured-default and adapter cluster is exactly the help rows that end with its caveat (each caveat is typed once, as a macro the rows are built from); the stored-3D cluster is the dashboard's `STORED_3D_INPUTS` less the Rust-only ones, and its caveat is the engine's sentence (`temperature::stored_3d_sentence!`), the one literal `STORED_3D_SENTENCE` and the engine's free-space helps read too; no help repeats its input's default; no overlay row covers a Rust-only input, whose help the engine words (R1).; the joint-neutral texts (ruling 11) claim what the engine does: a stronger manual block and longer magnets raise the pull-out, the cold-high torque and its gearbox-input value (longer magnets the overall length too), and a larger bore thins the hub wall: its help names the hub check (by its label), which the bore's slider top fails on the 8-magnet probe design, keyed or keyless, with the wall past the keyway still standing, and that wall breaks through at the slider top on the smallest ring |
| `src/app.rs`, `src/app/theme.rs` (feature `app`) | The app draws the whole panel as a page in the CAD dark theme; a share link opens its design as the session's start (the first Ctrl+Z keeps it) and a broken one changes nothing; a picked design file loads and a refused one changes nothing; the `gui-smoke` workflow's pinned share link and design file decode to their designs and the app solves the link, and the smoke prompt names the header's Show spreadsheet details toggle; `web/magcoupling/index.html` has the canvas `CANVAS_ID`, the title `TITLE` and the panel colour as its background; the visuals are `linkage-sim-rs`'s, function body for body; the theme stays dark when the system turns light |

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
Auto on the axial length with the requirement at its default 2.5 N·m, which `target_torque_Nm` equals) when the design format changes, a default
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
caller: no global state. Formatted verdicts are stated in the markup too (`concat`, `fmt`, `fmtnum`), with `ceilto` and `floorto` for Excel's CEILING and FLOOR and the literals `inf` and `nan`, so no record needs a Rust closure. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). The A4 teaching notes are static data (`notes.rs`), each listing the equations it explains, with its "What to change" entries (an input path, a step and, for a numeric move, the result it moves and which way) and up to two watch-outs; the panel shows a note only once a physics reviewer has signed it off (`note_for`). There are 20, the cap: the spec's list (harmonics, the back-iron factor, pull-out against angle, end effect, Br(T), demagnetization, slip heating, the slip temperature, the shaft clamp, and the physics behind each of the six A5 warnings), ferrite's cold side, the one-point calibration, and slice 2's three (how the coupling makes torque, the hot-low and cold-high torque band, what happens when the coupling slips); 20 notes, 19 shown while the clamp section is paused (`engine::clamps::PAUSED`, `notes::shown`: the shaft-clamp note keeps its earlier text, hidden with its section). All 20 are signed off: the notes slice 2 wrote or rewrote by its per-note physics review (`docs/analyses/2026-10-14-magcoupler-notes-review.md`), the rest by plan A-3 Task 16 and the independent review of 2026-10-01. `START_HERE` opens 13 of them in order while the clamp section is paused (how torque is made, the torque chain, temperature and the torque band, slipping and its heat, demagnetization); the full order, `START_HERE_ALL`, ends with the clamp as step 14. Plan:
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
