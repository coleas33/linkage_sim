# Update Tracker

Meaningful changes to the project that future sessions should know about.
Reverse chronological (newest at top).

---

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
