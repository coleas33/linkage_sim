# Audit findings: linkage-gui-export
Scope read: gui/export (mod, download, csv, dxf, svg, raster, firmware, report.rs partly: head, plotly helpers, four-bar sections, math section tail; schematic head and escape; animation outline), gui/sweep (mod.rs main bodies, fourbar, motion_profile, weights outline), gui/samples (mod, fourbar, special tails, helper usage), gui/dxf_import.rs (parse and overlay drawing), bin/*.

## LGE-01: Per-joint reaction series are not padded, so they shift against angles_deg when early samples fail
- Kind: broken
- Where: linkage-sim-rs/src/gui/sweep/mod.rs:458, 788-802, 806-817 (push_nan_row 955-957), 1335-1348, 1353; consumers csv.rs:189-194, animation.rs:685, plot_panel/mechanics.rs:293
- Evidence: `reaction_data` starts empty and a joint's vector is created lazily with `.entry(jr.joint_id.clone()).or_insert_with(|| Vec::with_capacity(capacity)).push(jr.resultant)` (line 791-794). The failure paths only push NaN into vectors that already exist (`for values in reaction_data.values_mut() { values.push(f64::NAN) }`, lines 799 and 955). If sample 0..k fail (position solve fails, or statics Err), the vector is created at sample k+1 and holds k+1 fewer entries than `angles_deg`. CSV reads `.get(i)` by sample index.
- Why it matters: for any sweep that starts in an unreachable gap (non-Grashof, range-limited, the two-arc case the tests cover) the reaction curves are shifted left by the number of failed leading samples, in plots, CSV, report and animation. The tests do not check reaction alignment for that case.
- Suggested action: create all reaction keys up front (from the joint list) or pad new vectors with NaN to the current sample count when first created. Add a test with a first-sample failure that asserts every series length equals angles_deg length.
- Fix class: needs-judgment
- Confidence: likely (read by hand; a test of sweep_two_arc_fourbar with reactions would settle it)

## LGE-02: A failed statics solve is stored as driver torque 0.0 in the forward sweep but NaN in trajectory mode
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/sweep/mod.rs:626 and 630 (forward: `unwrap_or(0.0)`, `push(0.0)`) vs 1252 and 1258 (trajectory: `unwrap_or(0.0)` for missing torque, `push(f64::NAN)` on Err); comment at 743-744
- Evidence: forward: `Err(_) => { data.driver_torques.as_mut().unwrap().push(0.0); None }`. Trajectory: `Err(_) => { data.driver_torques.as_mut().unwrap().push(f64::NAN); }`. The weight breakdown code even says "A failed statics solve pushed a 0 driver torque above; the breakdown reports NaN totals there, not a fake zero."
- Why it matters: the torque plot, torque envelope, CSV `driver_driver_torque_Nm` column, report and the actuator force (computed from this torque at 701-705) show a made-up zero at poses where the solve failed, instead of a gap.
- Suggested action: push NaN on Err in the forward sweep (and NaN when `driver_torque` is None), then drop the `statics_ok` special case. Check envelope code ignores NaN.
- Fix class: needs-judgment
- Confidence: certain for the inconsistency; likely for the visible effect

## LGE-03: apply_motion_profile reads angles_deg as absolute 0-360 degrees and is applied to every sweep
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/sweep/motion_profile.rs:136-212 (esp. 156, 163-170); caller gui/state/blueprint_ops.rs:1405-1433
- Evidence: `let total_angle = 2.0 * PI; ... let angle_rad = data.angles_deg[i].to_radians(); angle_to_fraction_trapezoidal(angle_rad, total_angle, ...)` (the doc at line 56 says "radians, relative to the sweep start"). The caller passes any sweep: `apply_motion_profile(&mut data, omega, self.motion_profile)` after a range sweep (`sweep_range_enabled`, e.g. 200..365) or a linear-driver stroke sweep where `angles_deg` holds metres.
- Why it matters: for a custom range the fraction is clamped at 1 past 360 and starts mid-profile; in stroke mode metres are converted with to_radians, so omega/alpha/torque profile values are meaningless. theta_0 is also not used. The tests only cover a full 0..360 revolute sweep.
- Suggested action: skip the profile (leave the fields None) unless the sweep is Angle mode and a full revolution, or make the angle relative to the sweep start and the span. Decider to confirm intent.
- Fix class: physics-suspect
- Confidence: likely (an Angle sweep over a custom range with a trapezoidal profile would settle it)

## LGE-04: GIF export silently drops frames that fail and still returns Ok, possibly an empty GIF
- Kind: broken
- Where: linkage-sim-rs/src/gui/export/raster.rs:125-150
- Evidence: `match solve_position(...) { Ok(result) if result.converged => { if let Ok(svg_str) = generate_svg_string(..) { if let Ok(rgba) = rasterize_svg_to_rgba(..) {...write_frame } } q_guess = result.q; } _ => {} }`. Also `let t = (angle_rad - theta_0) / omega;` with no omega guard (sweep_time guards the stroke case only), and `width as u16`/`height as u16` truncate silently.
- Why it matters: a solver failure, SVG error or rasterize error just skips the frame (the ping-pong loop jumps), and if every frame fails the function returns `Ok(buf)` holding a GIF with no frames, so the user is told the export worked. omega 0 gives NaN time for every frame.
- Suggested action: count written frames and return Err("no frames could be rendered") when zero; surface rasterize/SVG errors; reject omega 0 and sizes above 65535.
- Fix class: needs-judgment
- Confidence: certain (read the code)

## LGE-05: Path-writing export wrappers are used only by their own tests (baseline "never used" settled)
- Kind: not needed
- Where: gui/export/csv.rs:7-15 (`export_sweep_csv`) and 357-365 (`export_coupler_csv`); dxf.rs:96-105 (`export_mechanism_dxf`); svg.rs:230-239 (`export_mechanism_svg`); raster.rs:69-81 and 155-173 (`export_mechanism_png`, `export_mechanism_gif`); re-exports gui/export/mod.rs:21-29, 34-35
- Evidence: Grep over the whole of linkage-sim-rs (src, tests, bin) finds each name only at its definition, in mod.rs `pub use`, in doc comments, and in the module's own `#[cfg(test)]` code (e.g. csv.rs:514, 654, raster.rs:191, 260, svg.rs:255). All GUI callers use `download::download_text/bytes` with the `generate_*` string/byte functions. They are `#[cfg(feature = "native")]` so wasm never builds them; on native nothing calls them.
- Why it matters: six public functions, four re-exports and about a dozen tests that write fixed-name files into the temp dir exist to test file-writing that no feature uses; the real path (download.rs) is tested separately.
- Suggested action: delete the six wrappers and the re-exports; move the useful assertions to the `generate_*` functions (many already exist). Or keep one and mark it. Owner may want a library API for scripts.
- Fix class: owner-decision
- Confidence: certain they are unused outside tests on both targets

## LGE-06: add_plotly_multi_chart is dead and its job is written three times inline in generate_html_report
- Kind: not needed
- Where: gui/export/report.rs:567-601 (dead helper); inline copies at 392-406 (reactions), 442-453 (energy), 462-480 (coupler traces); the colour list is written twice (391, 458)
- Evidence: the helper is named only at its definition (baseline rustc did not flag it as private-unused only because of leads.md; Grep finds one hit). The three charts each hand-build `Plotly.newPlot('...', [...], {title..., margin:{t:40,b:50,l:60,r:20}}, {responsive:true})`. The helper's `x_data_for_all: Option<&str>` falls back to the string `null /* per-series x provided */`, which would emit broken JS for the coupler case, so it never worked for per-series x.
- Why it matters: repetition the owner flags, plus a leftover helper that would be wrong if used.
- Suggested action: delete `add_plotly_multi_chart` or finish it (per-series x) and use it for the three charts; share the colour palette as a const.
- Fix class: mechanical-safe for deleting the helper; needs-judgment for the refactor
- Confidence: certain

## LGE-07: Body, joint, trace and sensor names go into HTML, SVG and JS unescaped in most exporters; escapers disagree
- Kind: inconsistent
- Where: report.rs:150, 164 (body ids in tables), 380-383 (joint ids), 401, 474 (names into JS string literals `name:'{}'`), 418 (force details); svg.rs:147, 221 (`{}` of body_id / joint.id() into SVG text); dxf.rs:48 (layer name = body id). Escaped correctly at report.rs:803-812, 1081, 3796 (`html_escape`) and schematic.rs:789 (`xml_escape`).
- Evidence: schematic.rs:786-788 states "escape defensively in case a user chooses an ID with `<`, `>`, `&`"; report.rs:824 does the same for `html_escape`; but the dimension table writes `body_id, na, nb` raw and the Plotly traces use `'{}'` with a joint id, so an id containing `'` ends the JS string and an id with `<` breaks the table. `html_escape` (4 chars) and `xml_escape` (5 chars, adds `&apos;`) are two copies of one job, and svg.rs (the PNG/GIF source and the report fallback) has neither.
- Why it matters: a model with an unusual id (user-imported JSON) produces a broken report or SVG; a `</script>` in an id would inject markup into a page the user shares.
- Suggested action: one shared escaper in export/mod.rs, applied at every interpolation; a JS-string escaper for the Plotly names (animation.rs already has one for its data, see its test a_name_cannot_end_the_script).
- Fix class: needs-judgment
- Confidence: likely (ids come from user JSON; I did not test an id with a quote)

## LGE-08: Parameters and variables silenced with `let _ =`, a dead computation, and a doc comment on the wrong function
- Kind: not needed
- Where: report.rs:1489-1492 and 1525 (`mechanism` param unused: `let _ = mechanism;`), 2067-2072 and 2136 (`mechanism`, `q` unused: `let _ = (mu_deg, q, mechanism, GROUND_ID); // silence unused`; also `use GROUND_ID` just for this), 1760-1772 (`touches_ground_and_crank` computed over a chain with a "Placeholder" closure and then discarded), call sites 1169, 1175; misplaced doc 2220-2223 ("Append 'Singular configurations' subsection...") sits above the doc of `write_trajectory_inverse_section` (2224 on); the real `write_singular_configs_section` (3404) has no doc.
- Evidence: quoted above; the `_ =` lines exist only to silence rustc.
- Why it matters: dead parameters and a joint scan per report that do nothing; a stale doc that describes a different function.
- Suggested action: drop the unused parameters and the dead scan, move the doc comment to `write_singular_configs_section`.
- Fix class: mechanical-safe
- Confidence: certain

## LGE-09: SweepData.active_range is always None; its writers and readers are dead
- Kind: not needed
- Where: sweep/mod.rs:221-227, 452, 827-831, 1356, 1400; readers plot_panel/coupler.rs:40, plot_panel/mod.rs:248, 577, 639 (outside my part); test constructors csv.rs:504, 804, raster.rs:329, sweep/mod.rs:1717, 1764
- Evidence: the field doc says "Always `None` under the current sweep behaviour" and line 831 is `data.active_range = None;` ("computed after sweep loop" comment at 452 is stale). Grep finds no assignment of a `Some`.
- Why it matters: a serialised field and ~4 rendering branches (faded context curve, dashed range lines) that can never run, plus a line to maintain in every `SweepData` literal.
- Suggested action: remove the field and the readers (it is serde-serialised in SweepData, so check saved files: no `serde(default)` on it means old files lacking it would not matter once removed, but files that have it deserialise fine if unknown fields are allowed).
- Fix class: owner-decision
- Confidence: certain it is never Some in this tree

## LGE-10: Two four-bar detectors with different rules
- Kind: inconsistent
- Where: gui/sweep/fourbar.rs:17-110 (`detect_fourbar_links`, used for transmission angle in sweep/mod.rs:499 and blueprint_ops.rs) vs gui/export/report.rs:864-977 (`detect_fourbar` -> `FourbarLayout`)
- Evidence: the sweep one needs 3 moving bodies and 4 revolutes and a `driver_body_pair()`; takes lengths from the distance between two body-local attachment points and needs exactly 2 on ground. The report one requires `bodies().len()==4`, `joints().len()==4`, exactly 1 driver and no linear drivers, finds the crank from the driver, rocker from a ground joint, and measures lengths from joint world positions at q. A mechanism with a linear driver or an extra attachment point is a four-bar to one and not to the other (the sweep one accepts a LinearDriver mechanism if `driver_body_pair()` returns Some).
- Why it matters: the transmission-angle plot and the report's four-bar section can disagree on whether a mechanism is a four-bar and on its link lengths (lengths from attachment points vs real joint anchors).
- Suggested action: one detector returning the layout (ids + lengths); build the tuple from it.
- Fix class: needs-judgment
- Confidence: likely

## LGE-11: compute_trajectory takes a full SVD of the same Jacobian twice per sample
- Kind: inefficient
- Where: gui/sweep/mod.rs:1181-1185 and 1189-1193
- Evidence: `phi_q.clone().svd(true, true).solve(&neg_phi_t, 1e-14)` then `phi_q.clone().svd(true, true).solve(&gamma, 1e-14)` in the per-sample loop (n_samples times, recomputed on every trajectory edit).
- Why it matters: the SVD dominates the loop cost (O(n^3) with the vector copy twice); one factorisation serves both right-hand sides.
- Suggested action: `let svd = phi_q.svd(true, true);` once, then two `svd.solve` calls. Same numbers, no behaviour change.
- Fix class: mechanical-safe
- Confidence: certain

## LGE-12: compute_sweep_data_with_weights and compute_trajectory repeat the same setup; SweepData is built long-hand three times
- Kind: inefficient
- Where: sweep/mod.rs:467-496 vs 1066-1097 (coupler key collection + trace/vel/accel vectors); 504-511 vs 1100-1110 (`ma_bodies`, identical); 349-360 vs 1054-1060 (actuator element find); 403-432 (six identical `if actuator_info.is_some() { Some(Vec::with_capacity(capacity)) } else { None }` arms); 389-454 and 1364-1403 (two full SweepData literals, plus test literals at 1693, 1740); the 50-line "all-NaN row" in push_nan_row vs trajectory per-field pushes
- Evidence: the `ma_bodies` blocks are textually identical (`mech.driver_body_pair().and_then(|(_bi, driver)| { let output = mech.body_order().iter().filter(..).last().cloned(); ... })`).
- Why it matters: the owner flags repetition; the two paths already drifted (LGE-02, actuator_lengths filled in one and not the other).
- Suggested action: extract `coupler_keys(mech, capacity, &mut data)`, `ma_bodies(mech)`, and build the forward data from `empty_trajectory_sweep_data(mode)` with `Some(Vec::with_capacity)` set via a small closure.
- Fix class: needs-judgment
- Confidence: certain for the duplication

## LGE-13: write_sweep_csv codes each of eleven optional columns twice
- Kind: inefficient
- Where: gui/export/csv.rs:51-126 (header flags) and 130-219 (row pushes)
- Evidence: for each of actuator_force, actuator_force_id, length, speed, power, power_id, output_force: `let has_x = sweep.x.is_some(); if has_x { headers.push(..) }` then later `if has_x { if let Some(ref v) = sweep.x { row.push(format!("{:.6}", v.get(i).copied().unwrap_or(f64::NAN))); } }`. The check is done twice and the energy, ma, inv_dyn columns repeat the `{:.6}`/`unwrap_or(NAN)` idiom 20+ times.
- Why it matters: adding a column means editing two places in the right order; a header/row order mismatch would shift data silently (the only guard is a test per layout).
- Suggested action: build a `Vec<(&str, Option<&[f64]>)>` of the optional series once and loop it for header and rows.
- Fix class: needs-judgment
- Confidence: certain

## LGE-14: Two samples ignore the requested driver joint, and the Custom 6-bar can return an unsolved start
- Kind: broken
- Where: gui/samples/fourbar.rs:578-579 (`build_chebyshev_lambda_actuator(_driver_joint_id)`), gui/samples/special.rs:1033-1085 (`build_custom_6bar(_driver_joint_id)`); caller gui/state/driver_ops.rs:22-52 (`reassign_driver`); special.rs:1083-1084
- Evidence: `build_sample_with_driver` is documented "Returns an error if the specified joint does not exist, is not revolute, or is not grounded", and 28 of 30 samples validate through `attach_driver_to_grounded_revolute*`. These two take `_driver_joint_id` and drop it, so `reassign_driver` takes the `Ok` arm, records `self.driver_joint_id = Some(joint_id)` and `LoadCaseManager::new_default(joint_id, ..)` while the mechanism still has its original driver. Separately `build_custom_6bar` ends with `log::warn!("... no starting angle converged, using zero q0"); Ok((mech, q0))`.
- Why it matters: the UI says the driver moved to joint X while the model is unchanged; and a failed assembly is returned as success with an invalid q0.
- Suggested action: return `Err` when `driver_joint_id` is `Some` and differs from the sample's driver (so reassign falls back to the blueprint path), and return `Err` instead of `Ok(zero q0)` when nothing converges.
- Fix class: needs-judgment
- Confidence: likely (did not run the UI; code path is clear)

## LGE-15: The four-bar skeleton is hand-copied into eleven sample builders although build_standard_fourbar exists
- Kind: inefficient
- Where: gui/samples/fourbar.rs:89 (`build_fourbar_with_driver`), 365 (press), 446 (parallelogram actuator), 504 (Chebyshev), 685 (lambda actuator), 750 (Hoeken), 811 (Roberts); special.rs:53, 98 (quick-return, toggle clamp), 638 (coupler curve); sixbar.rs:721 (pantograph). Helper at fourbar.rs:167.
- Evidence: Hoeken (715-756), Roberts (772-817), Chebyshev (464-513) and coupler curve (special.rs:600-644) each repeat make_ground / make_bar x3 / set_bar_mass x3 / add_body x4 / four `add_revolute_joint("J1".."J4")` / `attach_driver_to_grounded_revolute_with_theta0(.., 0.0)` / `build` / `fourbar_initial_q0` (the leads.md duplicates at fourbar.rs:734-750 vs 487 and 795-811 vs 487 are this). `build_standard_fourbar` only differs in that its coupler point is `(x, 0.0)`.
- Why it matters: about 30 lines times 6-7 builders; any change to the standard joint naming or driver wiring needs 11 edits.
- Suggested action: give `build_standard_fourbar` a coupler-point `(x, y)` and a post-build hook (or return the pieces) and call it from Hoeken, Roberts, coupler curve; leave truly different ones.
- Fix class: needs-judgment
- Confidence: certain

## LGE-16: DXF overlay drops lines that cross the view, and import silently skips unsupported entities
- Kind: broken
- Where: gui/dxf_import.rs:287 (`if canvas_rect.contains(p1) || canvas_rect.contains(p2)`), 206 (`_ => {} // skip unsupported entities`), 245 (`let _ = canvas_rect; // used for clipping checks below`), 306 (`let _center = ...` unused), 167-195 (LwPolyline bulges ignored)
- Evidence: a line is drawn only if one end lies inside the canvas rect, so a long segment passing through the view (zoomed in) disappears; arcs and circles have no such test. Polyline, Spline, Ellipse, Insert and Text entities are dropped with no message; `parse_dxf_drawing` returns Ok if at least one supported entity exists. The comment on line 245 is wrong (canvas_rect is used afterwards, so the `let _` does nothing).
- Why it matters: a part drawn in a CAD file imports with missing geometry and the user is not told, and zooming makes overlay lines vanish.
- Suggested action: clip with a segment-vs-rect test (or draw without culling), count skipped entities and show "N entities not imported" in the panel, delete lines 245 and 306.
- Fix class: needs-judgment
- Confidence: certain for the culling; likely for the missing user message (did not read the panel code 360-1391)

## LGE-17: svg.rs mechanism SVG and schematic.rs repeat the bounding-box and scaling block with different rules
- Kind: inconsistent
- Where: gui/export/svg.rs:19-60 vs gui/export/schematic.rs:70-141
- Evidence: both compute min/max of attachment points, guard a degenerate box and pad it, but schematic also includes linear-driver points, uses 1e-12 and re-centres (`cx - 0.5`) while svg uses 1e-10 and `max = min + 1.0`, and pads 15% vs 20%. The report prefers the schematic and falls back to svg.rs (`generate_schematic_svg(..).or_else(|_| generate_svg_string(..)).unwrap_or_default()`, report.rs:28-30), and svg.rs is what PNG and GIF use, so a mechanism with a linear driver is clipped in the PNG/GIF.
- Why it matters: the same mechanism is framed differently between exporters, and `.unwrap_or_default()` turns a failure into an empty diagram without a word.
- Suggested action: one `world_bounds(mech, q)` helper used by both; show an error note in the report when both generators fail.
- Fix class: needs-judgment
- Confidence: likely

## LGE-18: index_at_driver doc says trajectory sweeps leave angles_deg empty, but compute_trajectory fills it with sample times
- Kind: inconsistent
- Where: gui/sweep/mod.rs:243-245 (doc) vs 993 and 1211 (`data.angles_deg.push(t_k)`); also `empty_trajectory_sweep_data` (1364) vs the field doc at 95
- Evidence: doc: "`None` when ... the sweep has no finite samples (trajectory sweeps leave `angles_deg` empty)". Code: `// 6. ... X-axis carries sample time. data.angles_deg.push(t_k);`. In Trajectory mode `index_at_driver` would therefore compare seconds against a degree target.
- Why it matters: a caller trusting the doc gets a plausible but wrong sample index in trajectory mode.
- Suggested action: return None for `SweepMode::Trajectory` at the top of `index_at_driver` and fix the doc.
- Fix class: needs-judgment
- Confidence: likely (did not check every caller for a trajectory guard; gui/mod.rs:528 only guards one use)

## LGE-19: FirmwareAdapter trait has one implementor and two methods nothing calls
- Kind: not needed
- Where: gui/export/firmware/mod.rs:25-40 (`file_extension`, `display_name`), json.rs:175-181; sole production caller gui/menu_bar.rs:1007-1025 names `JsonAdapter` directly
- Evidence: Grep: `file_extension()` and `display_name()` appear only at their definitions and in json.rs tests (279-280). The trait docs name planned G-code/Aerotech/Beckhoff/Galil adapters ("~150 LoC each") that do not exist.
- Why it matters: abstraction for adapters that are not there; two methods exist only to be asserted in a test.
- Suggested action: owner decides: keep as the planned extension point (then wire `display_name`/`file_extension` into the menu) or collapse to a function.
- Fix class: owner-decision
- Confidence: certain they are unused outside tests

## LGE-20: Two identical sample-count tests, and a sample whose enum name does not match what it builds
- Kind: inconsistent
- Where: gui/samples/mod.rs:690-693 (`all_samples_listed`: `assert_eq!(all().len(), 30)`) and 935-942 (`sample_count_is_30`, same assertion); enum variant `PeaucellierLipkin` (mod.rs:44, builder special.rs:600 `build_peaucellier_lipkin`) vs its label "Coupler Curve (figure-8)" (mod.rs:79) and doc "Substitutes for the Peaucellier-Lipkin 8-bar linkage"
- Evidence: the two tests are the same check. The builder is a plain Grashof change-point four-bar; its own doc says it only stands in for the Peaucellier-Lipkin linkage. Also `all()` is a hand list that nothing ties to the enum, so a new variant not added to it passes both tests only by updating the number.
- Why it matters: one redundant test; a misleading identifier in code and saved state.
- Suggested action: delete one test; rename the variant and builder to CouplerCurve (check serialised names first).
- Fix class: mechanical-safe for the test; owner-decision for the rename
- Confidence: certain

## Leads settled
- leads.md section 2, `report.rs:567 add_plotly_multi_chart`: real, dead (LGE-06).
- Baseline rustc "never used" list `export_sweep_csv`, `export_coupler_csv`, `export_mechanism_svg`, `export_mechanism_dxf`, `export_mechanism_png`, `export_mechanism_gif`: real; not used on wasm32 (cfg native only) and not on native outside their own tests (LGE-05).
- leads.md section 4 `samples/special.rs:614-628` vs `fourbar.rs:729`, `fourbar.rs:795-811` and `734-750` vs `487`: real duplication (LGE-15).
- leads.md section 4 `gui/sweep/mod.rs` vs `tests/property_tests.rs` (11 lines): not read, test-side; not reported.
- leads.md section 4 `solver/inverse_kinematics/test_helpers.rs:17-32` vs `export/schematic.rs:813` (test fixture building a four-bar): real but test code only; not reported.
- leads.md section 6 long functions in my part (`generate_schematic_svg` 707 lines, `compute_sweep_data_with_weights` 528, `write_math_background_section` 512, `generate_html_report` 509, `write_trajectory_inverse_section` 417, `compute_trajectory` 351, `generate_svg_string` 221, `draw_dxf_panel` 213): length noted; the repeated parts are in LGE-06, 12, 17; the report sections are mostly static HTML text, not split by judgment. Not separate findings.
- leads.md section 1 `uuid`, `wasm-bindgen-futures`: outside my part (Cargo), not checked here.
- gui/samples/custom_6bar.json: used by `include_str!` at special.rs:1038 and listed in docs/ai/03-structure.yaml; not dead.
- bin/linkage_web.rs native fallback duplicates linkage_gui.rs `run_native` in a smaller form: intended for `cargo run --bin linkage-web` (comment lines 8-9); not reported.

## Not read
- gui/dxf_import.rs lines 360-1391 (panel and conversion code; only grepped)
- gui/export/report.rs: most of the static math sections (lines about 983-1480, 1548-2060, 2235-3395, 3551-3880, 3883-3980 skimmed), and the tests from 4040 on
- gui/export/schematic.rs lines 145-760 (drawing body) and tests
- gui/export/animation.rs lines 36-630 (read only the outline and the tests' names) and animation_template.html
- gui/sweep/weights.rs (outline only)
- gui/samples/sixbar.rs, most of special.rs (quick-return to strandbeest bodies), helpers.rs bodies; fourbar.rs lines 25-160, 300-460, 520-715
- all tests in csv.rs, raster.rs, svg.rs beyond what is quoted
