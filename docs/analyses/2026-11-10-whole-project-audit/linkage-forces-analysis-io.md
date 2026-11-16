# Audit findings: linkage-forces-analysis-io
Scope read: all 26 files of linkage-sim-rs/src/forces, src/analysis and src/io (non-test code in full; test modules skimmed for weak assertions and duplicated fixtures; elements/mod.rs tests 330-1425 and compound.rs/analysis tests only by test names and assertion scan).

## LFA-01: Saving a cosine-stroke linear driver writes a different driver than it loads
- Kind: inconsistent
- Where: linkage-sim-rs/src/io/to_json.rs:213-227; linkage-sim-rs/src/io/from_json.rs:301-313; linkage-sim-rs/src/core/linear_driver.rs:239 (`cosine_linear_driver`)
- Evidence: to_json.rs:214-217 `// Serialize CosineStroke as a LinearLength with velocity derived from the stroke range ... On deserialization, the sample builder will recreate the cosine form.` then `let velocity = stroke_max - stroke_min;` and `length_0: *initial_length`. from_json.rs:302 always calls `constant_velocity_linear_driver(...)` and nothing recreates the cosine form (`git grep cosine_linear_driver` finds only linear_driver.rs, its own tests). stroke_min/stroke_max are not stored, so the cosine d(t)=mid+amp*cos(2*pi*t+phase) can not come back; the file reloads as d(t)=length_0+(max-min)*t.
- Why it matters: a stroke range (m) is stored in a field that means m/s, and the comment promises a rebuild that does not exist. `cosine_linear_driver` has no non-test caller, so today the branch is unreachable from the app, but it is a pub fn and the comment will mislead whoever wires it in.
- Suggested action: either delete the CosineStroke serialization arm and `cosine_linear_driver`/`DriverMeta::CosineStroke` together (if no caller is wanted), or add the stroke fields to `LinearDriverJson` and rebuild the cosine driver on load; at minimum fix the comment.
- Fix class: owner-decision
- Confidence: likely (certain for the missing rebuild; whether the cosine driver is wanted is the owner's call)

## LFA-02: Doc comments in io say drivers are skipped on save and not restored on load; the code does both
- Kind: inconsistent
- Where: linkage-sim-rs/src/io/to_json.rs:3-4, 255; linkage-sim-rs/src/io/from_json.rs:120-122, 370-372; linkage-sim-rs/src/io/schema.rs:50-52, 358-360
- Evidence: to_json.rs:3 `**Known limitation:** Driver constraints use closures, which cannot be serialized. Drivers are skipped on save`; yet to_json.rs:165-196 writes `DriverJson::ConstantSpeed`/`Expression` and :199-232 writes linear drivers. from_json.rs:371 `Driver constraints are **not** restored` while :270-298 restores both driver kinds. schema.rs:50 `Only constant-speed revolute drivers are supported; other driver types are skipped with a warning` (Expression is also written; the skip is silent, no warning is logged at to_json.rs:195, 229).
- Why it matters: the docs tell the next reader the wrong thing about what survives a save; the "re-attach the driver" instruction is obsolete.
- Suggested action: rewrite the four doc blocks to say: ConstantSpeed, Expression and linear (meta-bearing) drivers round-trip; closure drivers without `DriverMeta` are dropped silently.
- Fix class: mechanical-safe
- Confidence: certain

## LFA-03: Dead error variant and a dead legacy joint variant in io
- Kind: not needed
- Where: linkage-sim-rs/src/io/error.rs:26-27 (`UnknownJointType`); linkage-sim-rs/src/io/schema.rs:362-370 (`JointJson::RevoluteDriver`) with its match arm linkage-sim-rs/src/io/from_json.rs:262-265
- Evidence: `git grep -n UnknownJointType` finds only the definition. `JointJson::RevoluteDriver` is never constructed anywhere (`git grep JointJson::RevoluteDriver` only the from_json.rs match arm); to_json.rs writes drivers only into the top-level `drivers` map. No tracked .json file has `"type": "revolute_driver"` in a joints map (the 60 files that contain the quoted string are Python/Rust sources and docs about the unrelated `RevoluteDriver` struct).
- Why it matters: the enum variant is accepted silently on load and does nothing; `UnknownJointType` suggests an error path that does not exist (serde rejects unknown tags itself).
- Suggested action: remove `UnknownJointType`. For `RevoluteDriver`, remove it only if no user has old files (the arm is the compatibility path for pre-`drivers` files), so ask the owner.
- Fix class: owner-decision (variant); mechanical-safe (`UnknownJointType`)
- Confidence: certain for no producer; unsure whether old user files use it

## LFA-04: Four identical copies of the body-lookup block in `joint_to_json`
- Kind: inefficient
- Where: linkage-sim-rs/src/io/to_json.rs:87-143 (Revolute, Fixed, Prismatic, CamFollower arms)
- Evidence: each arm repeats `let body_i_id = j.body_i_id(); let body_j_id = j.body_j_id(); let body_i = bodies.get(body_i_id).ok_or_else(|| SerializationError::Build(format!("body '{}' not found", body_i_id)))?; let body_j = ...` (lines 88-91, 101-104, 115-118, 130-133).
- Why it matters: the owner flags repetition; any fix to the error text or lookup has to be made four times.
- Suggested action: pull body ids, the two bodies, and the two `find_point_name` calls into one helper that takes `(body_i_id, body_j_id, point_i_local, point_j_local)` and returns the four values; the match arms then only add their own fields.
- Fix class: needs-judgment
- Confidence: certain

## LFA-05: A force that names a point on an unknown body panics the loader, and the validator that would catch it is never called
- Kind: broken
- Where: linkage-sim-rs/src/forces/elements/force_element.rs:166-206 (`resolve_named_points`); linkage-sim-rs/src/io/from_json.rs:320-326; linkage-sim-rs/src/analysis/validation.rs:223 (`validate_force_elements`); linkage-sim-rs/src/core/state.rs (`get_position`/`get_angle` use `.expect("body not registered")`)
- Evidence: force_element.rs:166 `let pt = bodies[&s.body_a].resolve_force_point(name)?;` indexes a HashMap, so a missing body is a panic, not the `Err` the signature promises; from_json.rs:322 calls it inside `unwrap_or_else(|e| { log::warn!(...); f.clone() })` which only handles `Err`. `git grep validate_force_elements` finds the definition and its own six tests only: no production caller (not `Mechanism::build`, not `compute_validation`). docs/history/improvements_3202026.md:129 lists "F4 No force element validation at build time [x] ... validate_force_elements()" as done. Meanwhile `point_force_to_q`/`body_torque_to_q` (helpers.rs:30-33, 53-56) skip an unknown body silently, but the spring, damper and actuator evaluators call `state.body_point_global`, which panics on one.
- Why it matters: a hand-edited or stale file with a force that names a point on a missing body takes the app down on load (or at the first solve) instead of showing an error; the three unknown-body behaviours (panic, silent zero, warning) disagree.
- Suggested action: use `bodies.get(..).ok_or(BodyError...)?` in `resolve_named_points`; call `validate_force_elements` from the load path (or `build`) and surface its strings; pick one unknown-body rule for the helpers.
- Fix class: needs-judgment
- Confidence: likely (certain for the index and for the missing caller; a test that loads such a file would settle the panic)

## LFA-06: The "four two-point forces" match is written out in eight places
- Kind: inefficient
- Where: linkage-sim-rs/src/forces/compound.rs:55-83 (`analyze_force`), :194-220 (`expand_compound_force`), :280-322 (`remap_force_to_compound`, four identical arms); linkage-sim-rs/src/forces/elements/force_element.rs:164-203 (`resolve_named_points`, four arms), :216-231 (`attached_body_ids`); linkage-sim-rs/src/io/from_json.rs:25-46 (`get_force_point_pos`, `get_force_body_ids`); the four evaluators in forces/elements/evaluation.rs:61-92, 112-148, 209-268, 389-443 each repeat the same 12 lines (local points, global points, delta, length, unit); more copies outside this part (gui/state/blueprint_ops.rs:928, 958, 1496; solver/reactions.rs:171)
- Evidence: compound.rs:311-319 `let mut r = a.clone(); r.body_a = cyl_id; r.point_a = [0.0, 0.0]; r.point_a_name = None; r.body_b = rod_id; ...` repeated verbatim for LinearSpring, LinearDamper, GasSpring, LinearActuator; the four structs in element_types.rs share the same six fields (`body_a, point_a, point_a_name, body_b, point_b, point_b_name`).
- Why it matters: adding a fifth two-point element means touching every one of these matches (and the compiler does not warn at the `_ =>` arms in from_json.rs:31,44 and compound.rs:82, 219, 321); the owner flags repetition.
- Suggested action: give the four structs one embedded `TwoPointEnds { body_a, point_a, point_a_name, body_b, point_b, point_b_name }` (serde `flatten` keeps the file format) or add one `ForceElement::two_point_mut()` accessor returning the six fields; then collapse the matches. Share the endpoint/length/unit step of the evaluators in one helper.
- Fix class: needs-judgment
- Confidence: certain

## LFA-07: Failed compound expansion leaves orphan bodies in the mechanism
- Kind: broken
- Where: linkage-sim-rs/src/forces/compound.rs:172-265 (`expand_compound_force`); linkage-sim-rs/src/io/from_json.rs:355-361 (fallback)
- Evidence: compound.rs:187-188 `mech.add_body(create_compound_cylinder(..))?; mech.add_body(create_compound_rod(..))?;` come first; the joints follow with `?` (230, 244, 253). When point A has no name and point B is a mount point, `point_a_name.unwrap_or_default()` (:228) is `""`, so `add_revolute_joint` fails after both bodies are registered. from_json.rs:357-359 then logs and `mech.add_force(f); // fallback: add original force as-is`, never removing `force_{i}_cyl` and `force_{i}_rod`.
- Why it matters: two massless bodies with no joints are then part of the mechanism (6 free coordinates), which a later `build` or solve sees as a singular system. No test covers the failure path (compound.rs tests only reach the happy path and pure helpers).
- Suggested action: validate the endpoint references before `add_body`, or build into a scratch list and commit only on success; add a test with one raw-coordinate end and one mount end.
- Fix class: needs-judgment
- Confidence: likely (what would settle it: load a force with `point_a_name: None` and `point_b_name` naming a mount point, then `build()`)

## LFA-08: Every force evaluation allocates several full-size vectors per element
- Kind: inefficient
- Where: linkage-sim-rs/src/forces/helpers.rs:25, 48 (`point_force_to_q`, `body_torque_to_q` each allocate `DVector::zeros(n_coords)`); linkage-sim-rs/src/forces/elements/evaluation.rs:48-55 (gravity: one allocation per body), :88-90, :144-146, :264-266, :439-441 (two helper calls plus a `total` per element); caller linkage-sim-rs/src/core/mechanism.rs:493-497 (`assemble_forces_compiled`)
- Evidence: evaluation.rs:88-90 `let mut total = DVector::zeros(state.n_coords()); total += point_force_to_q(state, &s.body_a, ...); total += point_force_to_q(state, &s.body_b, ...);`. `assemble_forces_compiled` does `total += force.evaluate_compiled(...)` per element, called from solver/forward_dynamics.rs:107 on every integration step and from statics/inverse dynamics at every sweep point. A spring costs 4 heap vectors per call, gravity costs one per body.
- Why it matters: this is the per-step hot path; the work is pure allocation (each vector is mostly zeros, with two or three non-zero entries).
- Suggested action: change the helpers to add into a caller-owned `&mut DVector<f64>` (they are only called inside `forces::elements::evaluation`, plus two integration tests), and have `evaluate` take `&mut` accumulator; keep a thin allocating wrapper if the public API must stay.
- Fix class: needs-judgment
- Confidence: likely (measure with a forward-dynamics run before and after)

## LFA-09: Actuator stroke limits need both ends set, which the field docs and one test do not say
- Kind: inconsistent
- Where: linkage-sim-rs/src/forces/elements/evaluation.rs:420 (`limits_active`); linkage-sim-rs/src/forces/elements/element_types.rs:284-289; linkage-sim-rs/src/forces/elements/mod.rs:876-900 (`actuator_stroke_max_only_limit`)
- Evidence: element_types.rs:284,287 `/// Minimum stroke length (m). 0 = no min limit.` / `/// Maximum stroke length (m). 0 = no max limit.` but evaluation.rs:420 `let limits_active = a.stroke_max > 0.0 && a.stroke_max > a.stroke_min;`, so `stroke_min = 0.2, stroke_max = 0` (a minimum only) gets no end stop at all, silently. The inner `a.stroke_max > 0.0` at :430 is then always true. The test named `actuator_stroke_max_only_limit` puts the actuator at 0.01 m with max 2.0 and asserts zero force, so it passes whether or not a max limit exists, and no test covers a min-only limit.
- Why it matters: the two doc lines promise independent limits; the code gives one-sided limits only for a max. The test of the max-only case can not fail on the behaviour its name claims.
- Suggested action: either allow a min-only limit (`limits_active = stroke_max > 0.0 || stroke_min > 0.0`, with the max branch guarded) or document the rule in the field docs; make the test place the length beyond 2.0 and assert the end-stop force.
- Fix class: needs-judgment
- Confidence: certain for the doc/code mismatch and the weak test; whether min-only is intended is the owner's call

## LFA-10: Schema fields nothing writes or reads
- Kind: not needed
- Where: linkage-sim-rs/src/io/schema.rs:28-47 (`SweepConfig`), :70 (`MechanismJson.sweep_config`); linkage-sim-rs/src/io/to_json.rs:241; linkage-sim-rs/src/gui/state/mod.rs:575; linkage-sim-rs/tests/compound_force_integration.rs:94, 179; linkage-sim-rs/src/io/schema.rs:235 (`#[serde(alias = "ch_local")]`)
- Evidence: `git grep -E "sweep_config|SweepConfig" -- src tests` finds the struct, its Default impl, the field, and three places that set it to `None`; nothing reads it (GUI sweep state travels in `sweep_state`, which is read at file_io.rs:302). `git grep ch_local` finds only the alias itself (no sample, test, or Python writer uses it).
- Why it matters: a documented file-format field ("Sweep analysis angle range configuration") that is never honoured suggests a feature that does not exist; the alias is an untested compatibility path.
- Suggested action: remove `SweepConfig` and the field (old files with the key still load: no `deny_unknown_fields` anywhere in src) and the `ch_local` alias, or keep the alias if old Python-era files are known to exist.
- Fix class: owner-decision
- Confidence: certain for no reader; the owner decides about old files

## LFA-11: Doc comments that contradict the code they sit on
- Kind: inconsistent
- Where: linkage-sim-rs/src/analysis/transmission.rs:89 (`mechanical_advantage`); docs/architecture/ARCHITECTURE.md:218; linkage-sim-rs/src/forces/compound.rs:168-171; linkage-sim-rs/src/io/mod.rs:6-9
- Evidence: transmission.rs:89 `Returns `None` if either body is ground` but `get_velocity_component` returns `Some(0.0)` for ground (:130-132) and the test `mechanical_advantage_ground_output_is_zero` (:253) asserts a result with MA 0 for an output on ground (only an input on ground gives `None`). ARCHITECTURE.md:218 says "The loader checks the version and applies migration transforms if needed"; from_json.rs:134-141 only compares major numbers, there is no migration. compound.rs:170 refers to "Task 4 adds them ... during the serialization load path" (a plan task number). io/mod.rs:6 says the old `serialization` module is kept "for its tests only" and the re-export block says "all public items from the old serialization module"; the module now holds only tests.
- Why it matters: each sends a reader to behaviour that is not there.
- Suggested action: fix the four comments (ground output gives MA 0; no migration step, major check only; name `from_json::load_mechanism_unbuilt_from_json` instead of "Task 4"; rename the test module `serialization` to say it tests `to_json`/`from_json`).
- Fix class: mechanical-safe
- Confidence: certain

## LFA-12: Virtual-work check reports "mechanism not built" when the mechanism has no driver
- Kind: inconsistent
- Where: linkage-sim-rs/src/analysis/virtual_work.rs:70-72, 79-82, 87-90
- Evidence: virtual_work.rs:70-72 `let (body_i, body_j) = mech.driver_body_pair().ok_or(LinkageError::MechanismNotBuilt)?;` and the two `get_index(..).map_err(|_| LinkageError::MechanismNotBuilt)`. The error text is "Mechanism must be built before solving." (error.rs:14).
- Why it matters: the mechanism is built; the user is told to build it. A missing driver and an unknown driver body are different faults.
- Suggested action: add a `NoDriver` (or reuse an existing "model" variant) and map the two lookups to it.
- Fix class: needs-judgment
- Confidence: certain

## LFA-13: The same 4-bar test fixture is typed out again and again in this part
- Kind: inefficient
- Where: linkage-sim-rs/src/analysis/validation.rs:301-326 and :528-552 (identical except for `mech.build()`); linkage-sim-rs/src/analysis/virtual_work.rs:137; linkage-sim-rs/src/io/serialization.rs:20 (`build_fourbar_no_driver`) plus inline copies at :349-365, :676-689, :818-831; same shape again in solver/kinematics.rs:130, solver/reactions.rs:714, solver/statics.rs:207, solver/inverse_dynamics.rs:123, core/mechanism.rs:549, gui/export/schematic.rs:812, tests/golden_fixtures.rs:62
- Evidence: leads.md section 4 lists `validation.rs:529-549` against `validation.rs:302` (19 lines) and `serialization.rs:818-831`/`676-689` against `:349` (13 lines). validation.rs:530-532 and 303-305 are the same three `make_bar` lines with the same lengths (1, 3, 2) and ground pivots (0,0)/(4,0).
- Why it matters: roughly a dozen copies of one fixture; changing a link length for one test means chasing all of them. `solver/inverse_kinematics/test_helpers.rs:16` is already a shared `build_fourbar`, used only by IK tests.
- Suggested action: one `#[cfg(test)]` fixture module (`crate::test_support::build_fourbar(driver: bool, built: bool)`) used by all in-crate tests; within this part start by making validation.rs's second helper call the first.
- Fix class: needs-judgment
- Confidence: certain

## LFA-14: `.map_err(|e| SerializationError::Build(e.to_string()))` repeated nine times in from_json and four more lookups in to_json
- Kind: inefficient
- Where: linkage-sim-rs/src/io/from_json.rs:188, 202, 220, 240, 260, 283, 295, 312, 377; linkage-sim-rs/src/io/to_json.rs:90-91, 103-104, 117-118, 132-133 (the `ok_or_else(... Build(format!("body '{}' not found")))` form)
- Evidence: from_json.rs:201-202 `mech.add_revolute_joint(...).map_err(|e| SerializationError::Build(e.to_string()))?;` and the same tail on every `add_*` and `build` call; `SerializationError::Build(String)` is the only variant they use.
- Why it matters: one conversion written thirteen times; the original error type is flattened to a string each time.
- Suggested action: `impl From<MechanismError> for SerializationError` (and for the build error type), then plain `?`. Keep the message text so tests that match on it still pass.
- Fix class: needs-judgment
- Confidence: certain

## LFA-15: The time-modulation maths is written twice
- Kind: inefficient
- Where: linkage-sim-rs/src/forces/elements/time_modulation.rs:36-77 (`factor`) and :89-133 (`compile`)
- Evidence: `Ramp` is `(t - t_start) / (t_end - t_start)` at :53 and again at :110; `Step`, `Sinusoidal`, and the expression parse/bind/non-finite-to-zero logic (:56-75 and :114-130) are each a second copy with a second pair of warning strings.
- Why it matters: the two paths must agree (the hot loop uses `compile`, tests and `evaluate` use `factor`); a fix to one leaves the other behind.
- Suggested action: define `factor(&self, t)` as `self.compile()(t)` (accepting the re-parse that its doc already warns about) or drive both from one private function.
- Fix class: needs-judgment
- Confidence: certain

## LFA-16: Gas spring force at full compression is 2e9 times the nominal force
- Kind: inconsistent
- Where: linkage-sim-rs/src/forces/elements/evaluation.rs:240-246
- Evidence: `let compression = (g.extended_length - current_length).clamp(0.0, g.stroke); let gas_column = (g.stroke - compression).max(1e-10); let force_ratio = (g.stroke / gas_column).powf(g.polytropic_exp);` At `compression == stroke` the ratio is `stroke / 1e-10`, e.g. 0.2 / 1e-10 = 2e9 for a 0.2 m stroke (to the power `polytropic_exp`).
- Why it matters: a gas spring that reaches the end of its stroke jumps to a force of order 1e9 times `initial_force`, which stiffens any integration step; the 1e-10 floor is a divide-by-zero guard, not a modelled end stop. The element has no end-stop parameters, unlike `LinearActuatorElement`.
- Suggested action: decide on a cap (a stated maximum compression ratio) or an end-stop spring and note it in the element docs.
- Fix class: physics-suspect
- Confidence: unsure (the gas law is the model; whether the floor is acceptable is a physics decision)

## LFA-17: The motor element and the motor-sizing check treat reverse speed differently
- Kind: inconsistent
- Where: linkage-sim-rs/src/forces/elements/evaluation.rs:363-368 (`evaluate_motor`); linkage-sim-rs/src/analysis/motor_sizing.rs:61-68 (`check_motor_sizing`)
- Evidence: evaluate_motor: `let speed_in_dir = omega_rel * m.direction; let torque_fraction = (1.0 - speed_in_dir / m.no_load_speed).clamp(0.0, 1.0);` so a shaft turning against the drive direction gets the full stall torque (the fraction clamps to 1). check_motor_sizing: `let speed = speeds[i].abs(); ... stall_torque * (1.0 - speed / no_load_speed)` so the same reverse speed reduces the available torque. Both docs state the same linear droop envelope.
- Why it matters: the feasibility check can call a point infeasible that the simulated motor would carry, or the reverse.
- Suggested action: decide one rule for reverse speed and make both follow it.
- Fix class: physics-suspect
- Confidence: unsure

## LFA-18: `check_grashof` panics on a NaN link length
- Kind: broken
- Where: linkage-sim-rs/src/analysis/grashof.rs:74, 87
- Evidence: `sorted_lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());` and `.min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())`. The caller `update_grashof` (gui/state/blueprint_ops.rs:719-735) feeds it lengths taken from live geometry, and runs on every rebuild.
- Why it matters: one NaN length (a point typed as NaN or a degenerate drag) aborts the app instead of returning "not classifiable"; the owner prefers handling more edge cases. `motor_sizing.rs:89` already uses `unwrap_or(Ordering::Equal)` for the same pattern.
- Suggested action: use `f64::total_cmp`, or return early when any length is not finite.
- Fix class: needs-judgment
- Confidence: unsure (depends on whether `detect_fourbar_links` can return NaN; a test with a NaN length would settle it)

## Leads settled
- Cargo `uuid` (leads section 1): outside my part (Cargo.toml); not read for a verdict. `git grep uuid -- src` finds no use, but the `js` feature may exist to enable `getrandom/js` for `dxf` on wasm32, so it needs a build check before removal.
- Cargo `wasm-bindgen-futures`: outside my part (Cargo.toml); no use found in forces/analysis/io; not settled.
- `is_cam_follower`, `set_angle`, `driver_lambda_units`, `add_plotly_multi_chart`, `draw_actuator_stroke_annotation`, `update_prismatic_axis` (section 2): none lie in forces/analysis/io; not examined.
- Section 3 (Python names): none in my part.
- Section 4, `validation.rs:529-549` vs `:302`: real, test-fixture duplication; see LFA-13.
- Section 4, `serialization.rs:818-831`, `:676-689` vs `:349`: real, same fixture; see LFA-13.
- Section 4, `validation.rs:307` vs `singular_behavior.rs`, `virtual_work.rs:143-158`, `inverse_dynamics.rs:129-144`: real, the shared 4-bar fixture; see LFA-13.
- Section 4, `serialization.rs:50` vs `tests/singular_behavior.rs:679-697`: real, same fixture family; see LFA-13.
- Section 5 (unnamed files): none in my part.
- Section 6, `load_mechanism_unbuilt_from_json` (235 lines, from_json.rs:133): not a finding on its own; the length comes from per-joint match arms plus the compound-force block. The force block is covered by LFA-07 and LFA-05, the joint arms by LFA-14.
- Baseline rustc "never used" items: none of the listed items is in my part; `forces/`, `analysis/` and `io/` have no `#[allow]` markers, no TODO, no `#[ignore]`.
- `SerializationError::UnknownJointType`, `JointJson::RevoluteDriver`: not in leads, found by search; see LFA-03.

## Not read
- None unread. Partly skimmed only: linkage-sim-rs/src/forces/elements/mod.rs test bodies (lines 330-1425; listed by test name and scanned for weak assertions), linkage-sim-rs/src/forces/compound.rs tests (326-574), linkage-sim-rs/src/analysis/*.rs `#[cfg(test)]` modules other than names and assertion scans, linkage-sim-rs/src/io/serialization.rs lines 830-end.
