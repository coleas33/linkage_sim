# Verified: linkage-core-solver

## LCS-01: Cargo.toml lists dependencies the code never uses
- Verdict: PARTLY. Holds for `open`, `wasm-bindgen-futures` and the dev-dependency `meval`. Not confirmed for `uuid`.
- Checked: git grep over every .rs file for `open::`, `use open`, `wasm_bindgen_futures`, `spawn_local`, `uuid`: no hit in linkage-sim-rs (the only `wasm_bindgen_futures` hit is magcoupling-rs/src/app/files.rs:98, a different crate with its own Cargo.toml dependency). `meval` is in [dependencies] (Cargo.toml:26) and again in [dev-dependencies] (:62); tests/property_tests.rs:335 uses it. docs/ai/02-system.yaml:411 says "The dxf crate pulls uuid, which needs features = [\"js\"] for WASM", so the `uuid` line is a documented wasm pin and is not unused. Cargo.lock is tracked for both crates, and magcoupling-rs/Cargo.toml comments say a gate compares the two lock files.
- Safe to apply without judgment: no. Removing dependencies rewrites the tracked linkage-sim-rs/Cargo.lock, a gate checks the lock files against each other, and the wasm32 build cannot be checked here.

## LCS-02: Constraint projection in forward dynamics only edits the recorded samples and understates drift
- Verdict: PHYSICS-SUSPECT (described correctly)
- Checked: forward_dynamics.rs:364-397. `rk4_integrate` runs first (:367-368); then the loop at :375-397 projects `q_i` on every `project_interval`-th stored sample (`i % cfg.project_interval == 0`), pushes `drift` from the projected `q_i`, and never writes `q_i` back into `y_out`. Same block at :563-571 in simulate_with_events. GUI sets `project_interval: 10` at gui/state/mod.rs:1420. Physics not judged.
- Safe to apply without judgment: no. Physics-suspect, and the intended behaviour is an owner decision.

## LCS-03: simulate_with_events repeats simulate and rk4_integrate, and nothing outside tests calls it
- Verdict: CONFIRMED
- Checked: git grep for `simulate_with_events|detected_events|check_events|angle_limit_event|velocity_reversal_event` across the whole worktree: callers are only events.rs tests (:192-495), the definition at forward_dynamics.rs:436, `detected_events: Vec::new()` at :410, and the Python twin. No hit in tests/, gui, web, scripts or any cfg(wasm32) code. `simulate` (:327) and `simulate_with_events` (:436) share the y0 packing, rhs closure and unpack/projection/drift loop (:375-397 vs :558-579).
- Safe to apply without judgment: no. Removing events is an owner decision and merging the loops is a refactor.

## LCS-04: Stale doc comments in reactions.rs describe code that no longer exists
- Verdict: CONFIRMED
- Checked: reactions.rs:63-77 (`is_valid` doc cites a `1e-6 N` check on `residual_norm` and a pass-2 driver lambda check; body is only `!matches!(self.validation, ValidationState::Failed)`). reactions.rs:134-141 doc says None for "force zones, springs, dampers, fixed/prismatic joints, multiple actuators" while the code (:158-186, :295-325) models ForceZone, ExternalForce, ExternalTorque and returns None only for non-revolute joints, a second actuator, or any other force kind (spring, damper, gas spring, motor, friction, limits, torsion/rotary). Test `validation_verified_with_force_zone` exists at :1555.
- Safe to apply without judgment: yes (comments only).
- Exact change: linkage-sim-rs/src/solver/reactions.rs lines 71-74.
  old text:
  `    /// Tolerance: `1e-6 N` absolute on `residual_norm`, plus (when`
  `    /// `used_two_pass`) a `1e-6 N·m` check that the pass-2 driver`
  `    /// lambda has actually collapsed (the point of pass-2). On`
  `    /// well-conditioned systems both are typically ~1e-12.`
  new text:
  `    /// Returns false only when `validation` is `ValidationState::Failed`.`
  `    /// `residual_norm` is not read here; the validation state is computed`
  `    /// by `compute_validation`.`
  And lines 136-141, old text: `/// Returns `Some(relative_residual)` when every force element and joint type` through `/// never as pass or fail.` (the whole paragraph), new text:
  `/// Returns `Some(relative_residual)` when every joint is Revolute (plus the`
  `/// rotational Driver) and every force element is Gravity, at most one`
  `/// LinearActuator, a ForceZone, an ExternalForce or an ExternalTorque.`
  `/// Returns `None` for any other joint type (fixed, prismatic, cam) or force`
  `/// element (springs, dampers, gas springs, motors, friction, limits, torsion`
  `/// and rotary elements) and for more than one actuator; callers MUST treat`
  `/// `None` as "unverified", never as pass or fail.`

## LCS-05: pass-1-only result is built twice, and the transposed Jacobian SVD solve is written three times
- Verdict: CONFIRMED
- Checked: reactions.rs:534-562 (`pass1_only_result`) and :577-598 (closure `pass1_only`) build the same eight-field `ReactionSolveResult` with the same `compute_validation(...)` call; the closure takes the reaction vector, the function calls `extract_reactions` itself. SVD solve plus residual: statics.rs:50-70, inverse_dynamics.rs:67-97 (residual written `&phi_q_t * &lambdas - &rhs`, versus `+ &q_forces` in statics.rs:69), reactions.rs:637-652 (`phi_q.transpose()` at :640 and :650, `assemble_jacobian` again at :637).
- Safe to apply without judgment: no. Refactor.

## LCS-06: Reaction extraction reads a prismatic joint's two multipliers as an [Fx, Fy] force
- Verdict: PHYSICS-SUSPECT (described correctly)
- Checked: prismatic.rs:12-15 rows are `n_hat_i_global . d` and `theta_j - theta_i - delta_theta_0`. statics.rs:126-133 handles `n_eq == 2` with the comment "Revolute or prismatic: [Fx, Fy]" and builds `force_global = [lam[0], lam[1]]`, resultant from both. reactions.rs:160 returns None for non-revolute joints. Physics not judged.
- Safe to apply without judgment: no. Physics-suspect.

## LCS-07: Cam profile "Spline" is documented as cubic but is piecewise linear, and its comments and guards disagree with the code
- Verdict: CONFIRMED (one count off: the redundant ground checks are four, at cam.rs:140-141, 162-163, 224-225, 256-257, not five)
- Checked: cam.rs:22 doc says cubic spline; :81-96 `spline_eval` is linear with a `windows(2)` scan under a "Binary search" comment (:88); :98-108 derivative is the step slope, 0 outside the range; :110-113 second derivative is 0.0. `State::get_angle` returns 0.0 for ground (state.rs:123-126). cam.rs:177 and :191 use `if let Ok(..)` on `get_index`.
- Safe to apply without judgment: yes for the two comment edits only; the guards, the `if let Ok` and cubic-vs-linear are not.
- Exact change: linkage-sim-rs/src/core/constraint/cam.rs line 22, old `    /// Cubic spline through user-defined (theta, s) points.` new `    /// Piecewise-linear interpolation through user-defined (theta, s) points.` Line 88, old `    // Binary search for segment` new `    // Linear scan for the segment`.

## LCS-08: solve_for_target's doc tells callers to cache the workspace probe, but there is no way to, and the app runs 64 forward solves per trajectory sample
- Verdict: CONFIRMED
- Checked: solver.rs:113-116 doc; :138-144 `workspace_probe(... n_probe)` runs inside every call; git grep `WorkspaceProbe|probe_cache|solve_for_target`: `WorkspaceProbe` appears only in solver.rs and the mod.rs re-export, no cache anywhere. gui/sweep/mod.rs:1126-1138 calls `solve_for_target` per sample with `64`; gui/state/trajectory_ops.rs:38 is the other caller.
- Safe to apply without judgment: no.

## LCS-09: Inverse-kinematics helpers repeat the singularity guard four times and the finite-difference path repeats its own solves; the analytic path the notes call faster is unused
- Verdict: CONFIRMED
- Checked: git grep `eps_singularity`: exactly four blocks (derivatives.rs:51, :104, :158; solver.rs:200). `inverse_acceleration_fd` (:68-110) gets `r_prime_now` via `compute_r_prime` (:81, one `compute_dq_du`), then recomputes `grad` and `dq_du` at :100-101; it calls `compute_dq_du` four times in all. `inverse_acceleration_analytic` is called only from derivatives.rs tests (:255, :276), the mod.rs re-export, and mentioned in gui/export/report.rs:2615 and docs/ai/05-update-tracker.md:1998. The tracker quotes at :2007-2008 were not re-read.
- Safe to apply without judgment: no.

## LCS-10: Expression drivers re-parse their formula on every evaluation, and the DriverFn doc is stale
- Verdict: CONFIRMED
- Checked: driver.rs:255-259 parses and binds on every `eval`; :231-236 comment says parsing is sub-microsecond; :15-17 `DriverFn` doc says expression drivers come "in a future step" while `expression_driver` exists at :268. Each driver closure (:295-297) calls `eval`. mechanism.rs:471-478 `compile_force_modulations` exists for the force side. The cost claim is unmeasured.
- Safe to apply without judgment: yes for the doc line only; caching changes the `Send + Sync` design.
- Exact change: linkage-sim-rs/src/core/driver.rs lines 16-17, old `/// For serialization, the Rust port will add expression-based drivers` / `/// (meval/rhai) in a future step.` new `/// For serialization, expression-based drivers are built by` / `/// `expression_driver` (meval).`

## LCS-11: Every solver call goes through String-keyed HashMap lookups and builds full-width matrices per joint
- Verdict: PARTLY. The structure holds; the statement that hashing and allocation dominate is unmeasured.
- Checked: assembly.rs:13-94 has four copies of the same row-stacking loop; each joint returns an `n_eq x n_coords` `DMatrix` copied element by element (:42-46). `mechanism.rs:133-138` `n_constraints()` re-sums on every call. `constraint_ranges` is stored at mechanism.rs:407 and is not used by assembly.rs (it is used by `extract_reactions` in statics.rs). state.rs:82-89 `get_index` does a String-keyed `HashMap::get`. forward_dynamics.rs:117-146 element copy loops; :165-171 `a.clone().lu()` then `a.clone().svd`. The revolute-jacobian helper counts (helpers.rs:20-36) were not re-counted.
- Safe to apply without judgment: no.

## LCS-12: Dead or test-only public items, a stale allow, and config fields that nothing reads
- Verdict: CONFIRMED
- Checked: git grep (src, tests, ../web, ../scripts, ../docs, magcoupling-rs): `set_angle` only at its definition (state.rs:132); `is_cam_follower` only at its definition and in docs/history/ROADMAP_IMPLEMENTATION.md:632,644 (history notes); `extract_reactions_with_local` at statics.rs:171 and its test :316-330; `force_local_i/j` filled only there (zero at :156-157); `rtol`/`atol` only in the struct, Default (:53-54) and literals at forward_dynamics.rs:755 and tests/golden_fixtures.rs:705; `axis_local_i` field is read by `axis_local_i()` (prismatic.rs:36-37), which io/to_json.rs:124 calls, so the `#[allow(dead_code)]` at prismatic.rs:23 hides nothing.
- Safe to apply without judgment: yes for `set_angle`, `is_cam_follower` and the stale allow (no caller, no test, no script, no cfg path; the history doc line is a past-tense note). No for rtol/atol and the local-frame reactions (tests use them; owner decision).
- Exact change: linkage-sim-rs/src/core/state.rs lines 131-140, delete the doc line, the whole `pub fn set_angle` (131-139) and the blank line after it, so `make_q`'s doc comment follows the blank line at 130. linkage-sim-rs/src/core/constraint/mod.rs lines 68-71, delete the blank line at 68 and `pub fn is_cam_follower(&self) -> bool { matches!(self, Self::CamFollower(_)) }` (69-71), leaving `is_fixed` followed by the closing brace at :72. linkage-sim-rs/src/core/constraint/prismatic.rs line 23, delete `    #[allow(dead_code)]`.

## LCS-13: ControlTarget panics on bad data it can load from a file, repeats its normalisation three times, and hard-codes "ground"
- Verdict: CONFIRMED (the file-load path is plausible but not traced to a loader; the exact `Vector2::new(local_pt[0], local_pt[1])` pattern has 12 hits, not 14)
- Checked: control_target.rs:64, :124, :217 have the same `assert!(dir_norm > 1e-12, ...)` and normalise block; the enum derives `Serialize, Deserialize` (:16); `projection()` (:309-327) normalises and asserts on construction. `ControlTarget` sits inside `SweepMode::Trajectory` (gui/sweep/mod.rs:69-70), which derives `Deserialize` (:59). `"ground"` literals at :282, :290, :300, :316, :336 versus `GROUND_ID` in state.rs. I did not trace whether `SweepMode` reaches a saved file.
- Safe to apply without judgment: no.

## LCS-14: solve_for_target maps "no probe sample converged" to SvdSolveFailed, and its branch-jump threshold is a heuristic from the centre of gravity
- Verdict: PHYSICS-SUSPECT (described correctly)
- Checked: solver.rs:64-68 skips non-converged samples, so an empty `g_samples` means none converged; :142-144 returns `LinkageError::SvdSolveFailed`, whose message is "SVD solve failed." (error.rs); `PositionSolveNotConverged` exists. :185 `branch_threshold = 0.5 * max_body_length`; :263-281 uses `2*|cg_local|` per body, floored at 0.01 m, with the comment "heuristic only". The threshold's physics was not judged.
- Safe to apply without judgment: no.

## LCS-15: Small inconsistencies between sibling APIs in body.rs and mechanism.rs
- Verdict: CONFIRMED
- Checked: body.rs:209-224 `add_attachment_point` checks only `attachment_points`; :253-268 `add_mount_point` checks both maps and returns `MountPointNameCollision`; :270-273 `resolve_force_point` prefers the attachment. mechanism.rs:446-448 `remove_force` calls `Vec::remove` (panics on a bad index); :451-455 `replace_force` ignores a bad index.
- Safe to apply without judgment: no. Behaviour change.

## LCS-16: Two tests assert nothing that can fail, and a fourth copy of the same 4-bar fixture sits in each module
- Verdict: PARTLY. `inverse_solve_status_constructs_all_variants` (severity.rs:64-79) has no assertion: confirmed. The statics.rs:283 assertion `jr.resultant >= 0.0` is weak but not vacuous (a NaN resultant fails it, though the is_finite asserts at :284-285 also catch NaN). Fixture duplication confirmed at validation.rs:301 and :528, virtual_work.rs:137, mechanism.rs:549, schematic.rs:812, gravity_breakdown.rs:357; the other cited sites (kinematics.rs:130, reactions.rs:714/742, statics.rs:207, inverse_dynamics.rs:123) were not opened.
- Checked: severity.rs:56-85, statics.rs:262-287, git grep for 4-bar builder names.
- Safe to apply without judgment: no. Deleting tests conflicts with the repo's "more tests than fewer" rule, and the fixture move is a refactor.

## Seen while checking
- (none)
