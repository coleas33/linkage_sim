# Verified: linkage-forces-analysis-io

## LFA-01: Saving a cosine-stroke linear driver writes a different driver than it loads
- Verdict: CONFIRMED
- Checked: to_json.rs:213-227 writes `velocity = stroke_max - stroke_min` and `length_0 = initial_length` and the comment promises a rebuild; from_json.rs:301-313 always calls `constant_velocity_linear_driver`. git grep `cosine_linear_driver` across the worktree and magcoupling-rs/web: only linear_driver.rs (definition :239, tests :517-682); `DriverMeta::CosineStroke` otherwise appears in driver.rs:46, to_json.rs and a label at gui/canvas/rendering/mod.rs:1136. So no non-test code builds a cosine driver and the save arm is unreachable from the app today.
- Safe to apply without judgment: no. Owner decision (delete or complete the feature).

## LFA-02: Doc comments in io say drivers are skipped on save and not restored on load; the code does both
- Verdict: CONFIRMED (line numbers for schema.rs are a few lines off: the text is at schema.rs:56-57 and :362-363, not :50-52 and :358-360)
- Checked: to_json.rs:3-4 and :255 say drivers are skipped; the code at :165-232 writes ConstantSpeed, Expression and linear drivers. from_json.rs:120-122 and :371-373 say drivers are not restored; :269-313 restores all three. schema.rs:56-57 says only constant-speed drivers are supported and others are skipped with a warning; Expression is written (to_json.rs:176-186) and the skip at :195 is silent (a `continue` at :190 for linear meta, no log call anywhere in the block).
- Safe to apply without judgment: yes (comments only).
- Exact change (all in linkage-sim-rs/src/io):
  - to_json.rs lines 3-4, old `//! **Known limitation:** Driver constraints use closures, which cannot be` / `//! serialized. Drivers are skipped on save and must be re-attached after load.` new `//! **Known limitation:** Drivers built from closures without `DriverMeta` cannot be` / `//! serialized and are skipped silently on save. Constant-speed, expression and` / `//! linear (meta-bearing) drivers round-trip.`
  - to_json.rs line 255, old `/// Driver constraints are skipped -- they must be re-attached after loading.` new `/// Drivers without `DriverMeta` (general closures) are skipped silently; constant-speed, expression and linear drivers are saved.`
  - from_json.rs lines 120-122, old `/// The caller can add drivers or make other modifications before calling` / `/// `mech.build()`. Driver constraints from the JSON are skipped (closures` / `/// are not serializable).` new `/// The caller can add drivers or make other modifications before calling` / `/// `mech.build()`. Drivers stored in the JSON (constant-speed, expression and` / `/// linear) are restored.`
  - from_json.rs lines 371-373, old `/// Driver constraints are **not** restored (closures are not serializable).` / `/// If you need to add drivers before building, use [`load_mechanism_unbuilt`]` / `/// instead.` new `/// Drivers stored in the JSON are restored; closure drivers without `DriverMeta`` / `/// are never saved. To add further drivers before building, use` / `/// [`load_mechanism_unbuilt`] instead.`
  - schema.rs lines 56-57, old `/// Serialized driver constraints. Only constant-speed revolute drivers are` / `/// supported; other driver types are skipped with a warning.` new `/// Serialized revolute drivers (constant-speed and expression). Closure drivers` / `/// without `DriverMeta` are skipped silently.`
  - schema.rs lines 362-363, old `/// Marker for driver constraints. The closure is not serializable;` / `/// users must re-attach the driver function after loading.` new `/// Legacy marker for a driver stored in the joints map. Ignored on load; drivers` / `/// now live in the top-level `drivers` map.`

## LFA-03: Dead error variant and a dead legacy joint variant in io
- Verdict: PARTLY. `UnknownJointType` is dead (confirmed). `JointJson::RevoluteDriver` is never constructed (confirmed) but the finder's "only the from_json.rs match arm" is wrong: it is also matched in gui/canvas/rendering/mod.rs:538 and :1210 and gui/state/blueprint_ops.rs:40, :108, :128, so removing it needs edits in those three files.
- Checked: git grep `UnknownJointType`: only io/error.rs:27 (no match arm anywhere). git grep `JointJson::RevoluteDriver`: the six pattern hits above plus a plan document (docs/superpowers/plans/2026-03-18-multi-pivot-body-editor.md:609); no constructor. Tracked .json files: only gui/samples/custom_6bar.json and web/vercel.json, neither has a revolute_driver entry.
- Safe to apply without judgment: yes for `UnknownJointType` (no match, no test, no doc use); no for `RevoluteDriver` (owner decision on old files, and edits in three more files).
- Exact change: linkage-sim-rs/src/io/error.rs lines 25-27, delete the blank line at 25, `    #[error("Unknown joint type '{0}'")]` (26) and `    UnknownJointType(String),` (27), so `PointNameNotFound { .. },` is followed directly by the closing `}` of the enum.

## LFA-04: Four identical copies of the body-lookup block in `joint_to_json`
- Verdict: CONFIRMED
- Checked: to_json.rs:87-143; the Revolute, Fixed, Prismatic and CamFollower arms each repeat the same four lines (:88-91, :101-104, :115-118, :130-133).
- Safe to apply without judgment: no. Refactor.

## LFA-05: A force that names a point on an unknown body panics the loader, and the validator that would catch it is never called
- Verdict: CONFIRMED (the validator has nine test call sites, not six)
- Checked: force_element.rs:164-209 indexes `bodies[&s.body_a]` etc. (HashMap `Index` panics on a missing key) inside a function returning `Result`; from_json.rs:320-325 wraps it in `unwrap_or_else`, which only handles `Err`. Grep (ripgrep over the whole worktree) for `validate_force_elements`: validation.rs only (definition :223 and nine test calls) plus one mention in docs/history/improvements_3202026.md:129 that lists it as done; no caller in load, build or `compute_validation`. forces/helpers.rs:30-33 and :53-56 skip an unknown body silently; state.rs `get_position`/`get_angle` use `.expect("body not registered")`. I did not run a load of such a file.
- Safe to apply without judgment: no.

## LFA-06: The "four two-point forces" match is written out in eight places
- Verdict: CONFIRMED (the evaluator repeats were not opened)
- Checked: compound.rs:55-83, :194-220, :280-322 (four identical arms); force_element.rs:164-203 and :216-231 (`attached_body_ids` lists every variant, so the compiler does cover that one); from_json.rs:25-46 with `_ =>` arms at :31 and :44; compound.rs `_` arms at :82, :219, :321. element_types.rs has four `body_a: String` structs.
- Safe to apply without judgment: no.

## LFA-07: Failed compound expansion leaves orphan bodies in the mechanism
- Verdict: CONFIRMED (read only; no run)
- Checked: compound.rs:187-188 adds both bodies first; :230, :244, :253 add joints with `?`; with `point_a_name == None` and `mount_a == false` the reference is `""` (:228), and `add_revolute_joint` (mechanism.rs:180-197) resolves it through `get_attachment_point(...)?`, so it fails after both bodies are registered. from_json.rs:355-360 logs and adds the original force without removing `force_{i}_cyl` and `force_{i}_rod`. `analyze_force` (compound.rs:85-96) returns NeedsExpansion when only one end is a mount point, so the mixed case reaches this path. The singular-system consequence and the absence of a failure-path test were not checked exhaustively.
- Safe to apply without judgment: no.

## LFA-08: Every force evaluation allocates several full-size vectors per element
- Verdict: CONFIRMED (a spring allocates three vectors inside the evaluator: `total` plus two helper results; the finder said four)
- Checked: helpers.rs:25 and :48 each `DVector::zeros(n_coords)`; evaluation.rs:48-55 (gravity, one helper call per body), :88-90, :144-146, :264-266, :439-441; mechanism.rs:493-497; the helpers are used outside evaluation.rs only by tests/force_zone_tests.rs:309-322.
- Safe to apply without judgment: no.

## LFA-09: Actuator stroke limits need both ends set, which the field docs and one test do not say
- Verdict: CONFIRMED
- Checked: evaluation.rs:420 `limits_active = a.stroke_max > 0.0 && a.stroke_max > a.stroke_min`; element_types.rs:284-289 docs; mod.rs:875-900 test: actuator at 0.01 m with max 2.0 and asserts zero force. Every `stroke_min > 0` literal in tests has a larger `stroke_max`, so no test covers a min-only limit.
- Safe to apply without judgment: no. The same rule is duplicated in gui/canvas/rendering/force_render.rs:262, so a fix must change both.

## LFA-10: Schema fields nothing writes or reads
- Verdict: CONFIRMED
- Checked: git grep `sweep_config|SweepConfig`: struct and Default (schema.rs:31-48), field (:70), three `None` sets (gui/state/mod.rs:575, to_json.rs:241, tests/compound_force_integration.rs:94 and :179), plus a layout note in SYSTEM.md:70; no reader. `ch_local`: only the alias at schema.rs:235 in the whole worktree. No `deny_unknown_fields` in linkage-sim-rs.
- Safe to apply without judgment: no. Owner decision on old files, and tests set the field.

## LFA-11: Doc comments that contradict the code they sit on
- Verdict: CONFIRMED
- Checked: transmission.rs:89-90 says None if either body is ground; `get_velocity_component` returns `Some(0.0)` for ground (:130-132) and the test at :253-265 asserts `is_some()` with MA 0 (an input on ground gives None only through the `v_in` zero check, test :239-250). The doc at :123 on `get_velocity_component` has the same error. ARCHITECTURE.md:218 claims migration transforms; from_json.rs:134-141 only compares major versions. compound.rs:168-171 and :224 say "Task 4". io/mod.rs:6 and :10 describe a `serialization` module that is now `#[cfg(test)] mod tests` only (serialization.rs:3-4).
- Safe to apply without judgment: yes for the comment edits below. Not included: renaming the test module.
- Exact change:
  - linkage-sim-rs/src/analysis/transmission.rs lines 89-90, old `/// Returns `None` if either body is ground, a body is not found, or the input` / `/// velocity is effectively zero (|omega_in| < 1e-15).` new `/// Returns `None` if a body is not found or the input velocity is effectively` / `/// zero (|omega_in| < 1e-15), which includes an input on ground. A ground output` / `/// has velocity 0 and gives `ma = 0`.`
  - transmission.rs line 123, old `/// Returns `None` if the body is ground or not found in the state.` new `/// Returns `Some(0.0)` for ground and `None` if the body is not found in the state.`
  - linkage-sim-rs/src/forces/compound.rs lines 170-171, old `/// calling this function (Task 4 adds them from `mount_points` during the` / `/// serialization load path).` new `/// calling this function (`from_json::load_mechanism_unbuilt_from_json` adds them` / `/// from `mount_points` during the load path).` Line 224, old `    // the mount_point under the synthetic name `_force_{idx}_mount_a` (Task 4).` new `    // the mount_point under the synthetic name `_force_{idx}_mount_a`.` (the word "Task 4" is on that line in the file)
  - linkage-sim-rs/src/io/mod.rs line 6, old `// Keep the old module around for its tests only.` new `// Round-trip tests for `to_json`/`from_json` live in this test-only module.` Line 10, old `// Re-export for backward compatibility -- all public items from the old serialization module.` new `// Re-export the public items of `error`, `from_json`, `schema` and `to_json`.`
  - docs/architecture/ARCHITECTURE.md line 218, old `The loader checks the version and applies migration transforms if needed.` new `The loader compares the major version and rejects a file whose major version differs; there is no migration step.`

## LFA-12: Virtual-work check reports "mechanism not built" when the mechanism has no driver
- Verdict: CONFIRMED
- Checked: virtual_work.rs:70-72, :79-82, :87-90 map a missing driver and unknown driver bodies to `LinkageError::MechanismNotBuilt`, whose text is "Mechanism must be built before solving." (error.rs:14). `driver_body_pair` (mechanism.rs:426-428) is `self.drivers.first().map(...)`, so it is None for a mechanism with no revolute driver, built or not.
- Safe to apply without judgment: no. Adds an error variant.

## LFA-13: The same 4-bar test fixture is typed out again and again in this part
- Verdict: CONFIRMED (the other-module sites in the list were not all opened)
- Checked: validation.rs:301-326 and :528-552 are identical except the final `mech.build()`; virtual_work.rs:137, io/serialization.rs:20, mechanism.rs:549, gui/export/schematic.rs:812 and tests/golden_fixtures.rs:62 each define a 4-bar builder.
- Safe to apply without judgment: no. Test refactor.

## LFA-14: `.map_err(|e| SerializationError::Build(e.to_string()))` repeated nine times in from_json and four more lookups in to_json
- Verdict: CONFIRMED
- Checked: from_json.rs lines 188, 202, 220, 240, 260, 283, 295, 312, 377 each contain `SerializationError::Build(e.to_string())`; to_json.rs:90, 91, 103, 104, 117, 118, 132, 133 are the lookups. There is no `impl From<..> for SerializationError` (git grep `impl From` in io finds none).
- Safe to apply without judgment: no.

## LFA-15: The time-modulation maths is written twice
- Verdict: CONFIRMED
- Checked: time_modulation.rs:36-77 (`factor`) and :89-133 (`compile`): Ramp at :53 and :110, Step, Sinusoidal, and the parse/bind/non-finite logic with a second set of warning strings (:66, :71 versus :122, :127).
- Safe to apply without judgment: no.

## LFA-16: Gas spring force at full compression is 2e9 times the nominal force
- Verdict: PHYSICS-SUSPECT (described correctly)
- Checked: evaluation.rs:241-246: `compression` clamped to `[0, stroke]`, `gas_column = (stroke - compression).max(1e-10)`, `force_ratio = (stroke / gas_column).powf(polytropic_exp)`. At compression == stroke the ratio is stroke / 1e-10. `GasSpringElement` (element_types.rs:159-185) has no end-stop fields. Physics not judged.
- Safe to apply without judgment: no.

## LFA-17: The motor element and the motor-sizing check treat reverse speed differently
- Verdict: PHYSICS-SUSPECT (described correctly)
- Checked: evaluation.rs:363-368 (`speed_in_dir = omega_rel * direction`, fraction `(1 - speed_in_dir / no_load_speed).clamp(0, 1)`, so a negative `speed_in_dir` gives full stall torque); motor_sizing.rs:61-68 uses `speeds[i].abs()` and `stall_torque * (1 - speed / no_load_speed)`. Physics not judged; I did not compare the two doc comments.
- Safe to apply without judgment: no.

## LFA-18: `check_grashof` panics on a NaN link length
- Verdict: PARTLY. The two `partial_cmp(..).unwrap()` calls exist (grashof.rs:74 and :87) and panic on NaN. Not proven that a NaN length can reach them: `detect_fourbar_links` was not read.
- Checked: grashof.rs:59-92; callers `update_grashof` (gui/state/blueprint_ops.rs:719-735) and crank_selection.rs:89, :126. motor_sizing.rs:89 uses `unwrap_or(Ordering::Equal)`.
- Safe to apply without judgment: no. It changes behaviour on NaN input.

## Seen while checking
- gui/canvas/rendering/force_render.rs:262 repeats the actuator `limits_active` rule of LFA-09, so the two must change together.
- schema.rs:317 (`/// Driver joints are represented as a marker -- the closure cannot be serialized.`) carries the same stale claim as LFA-02 and was not in the finder's list.
