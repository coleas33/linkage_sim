# Audit findings: linkage-core-solver
Scope read: linkage-sim-rs src/core (all), src/solver (all except the test modules of forward_dynamics, reactions, inverse_dynamics, events and control_target, read only in part), src/geometry/mod.rs, src/error.rs, src/lib.rs, Cargo.toml.



## LCS-01: Cargo.toml lists dependencies the code never uses
- Kind: not needed
- Where: linkage-sim-rs/Cargo.toml:35 (`open`, optional, in the `native` feature at :11), :42 (`wasm-bindgen-futures`, wasm32 table), :38 (`uuid`), :62 (`meval` in [dev-dependencies])
- Evidence: Grep for `open::`, `use open`, `wasm_bindgen_futures`, `spawn_local`, `uuid` over every .rs file in linkage-sim-rs (src, tests, bins): no hit for any of them. `meval` is already in [dependencies] (line 26) and the dev-dependency line repeats it (tests/property_tests.rs:335 uses it, which the normal dependency already covers). In Cargo.lock `uuid` is reached through `dxf` (lock line ~981) and `zbus`.
- Why it matters: `open` and `wasm-bindgen-futures` add compile time and lock entries for nothing; the dev-dependency `meval` is a redundant duplicate line.
- Suggested action: remove `open` (and `dep:open` from the `native` feature list), `wasm-bindgen-futures`, and the dev-dependency `meval`. For `uuid = { features = ["js"] }`: keep only if it exists to turn on the `js` feature for `dxf`'s transitive uuid on wasm32 (then add a one-line comment saying so); otherwise remove.
- Fix class: owner-decision for uuid; mechanical-safe for `open`, `wasm-bindgen-futures`, dev `meval` (verify with a wasm32 `cargo check` after, since eframe may need the futures crate itself, but it declares that itself)
- Confidence: certain for open, wasm-bindgen-futures and the dev meval line; unsure for uuid (a wasm32 build without the `js` feature would settle it)

## LCS-02: Constraint projection in forward dynamics only edits the recorded samples and understates drift
- Kind: broken
- Where: linkage-sim-rs/src/solver/forward_dynamics.rs:375-397 (simulate), :558-579 (simulate_with_events), :40-41 (config doc), linkage-sim-rs/src/gui/state/mod.rs:1420 (the GUI sets `project_interval: 10`)
- Evidence: the config doc says "Apply constraint projection every N steps". The code runs after integration has finished: `for (i, y) in y_out.iter().enumerate() { ... if cfg.project_interval > 0 && i > 0 && i % cfg.project_interval == 0 { q_i = project_constraints(...); project_velocity(...) } let phi = assemble_constraints(mech,&q_i,..); drift.push(phi.norm()); ...`. `i` indexes stored output samples (t_eval frames in the GUI, 60 per second), not integrator steps (max_step 0.002), and the projected `q_i` is never written back into `y`, so the integration itself is never corrected.
- Why it matters: the GUI simulation shows every 10th frame projected and the other nine raw, so the animation can jitter, and `drift` (shown to the user, also used by tests) is measured after projection on those frames, so it reads low and hides the real Baumgarte drift. The option cannot do what its name and doc say.
- Suggested action: either project inside the step loop and feed the result back into `y` (then the drift series is honest), or delete `project_interval`, `project_tol`, `max_project_iter`, `project_constraints`, `project_velocity` and the test at forward_dynamics.rs:881. Decide which is intended first.
- Fix class: physics-suspect
- Confidence: certain about the control flow (read in full); whether the owner wants real projection is the open question

## LCS-03: simulate_with_events repeats simulate and rk4_integrate, and nothing outside tests calls it
- Kind: inefficient (code written twice) and not needed
- Where: linkage-sim-rs/src/solver/forward_dynamics.rs:327-414 vs :436-598 (about 150 near-identical lines: y0 packing, rhs closure, the RK4 stages, t_eval store/interpolation block, unpack/projection/drift loop, success message); the RK4 stage + t_eval block also appears a third time in `rk4_integrate` :229-310. Production use: only `simulate` (gui/state/mod.rs:1432). `simulate_with_events`, `solver/events.rs` (`check_events`, `angle_limit_event`, `velocity_reversal_event`, `DynamicsEvent`, `EventKind`), `ForwardDynamicsResult.detected_events` are called only from events.rs's own tests (events.rs:256-472) and a `Vec::new()` at forward_dynamics.rs:410.
- Evidence: `git grep simulate_with_events` and `angle_limit_event` in src and tests: hits only in forward_dynamics.rs (definition) and events.rs tests. The python mirror `src/linkage_sim/solvers/events.py` is the only other user of the idea.
- Why it matters: any fix to the integrator (such as LCS-02) must be made in two or three places; events are a finished feature with no caller in the app.
- Suggested action: make `simulate` call one shared step loop that takes an `&[DynamicsEvent]` (empty by default), and delete the other copy. If the owner does not plan to expose events in the GUI, remove events.rs and the field too.
- Fix class: owner-decision (removing events) / needs-judgment (merging the loops)
- Confidence: certain

## LCS-04: Stale doc comments in reactions.rs describe code that no longer exists
- Kind: inconsistent
- Where: linkage-sim-rs/src/solver/reactions.rs:63-74 (`is_valid` doc), :136-141 (`body_equilibrium_residual` doc)
- Evidence: `is_valid` doc: "Tolerance: `1e-6 N` absolute on `residual_norm`, plus (when `used_two_pass`) a `1e-6 N·m` check that the pass-2 driver lambda has actually collapsed". The body is `!matches!(self.validation, ValidationState::Failed)` and never reads `residual_norm`. `body_equilibrium_residual` doc says it returns `None` for "force zones, springs, dampers, fixed/prismatic joints, multiple actuators" and lists only Gravity and one actuator as modelled; the code at :295-325 models ForceZone, ExternalForce and ExternalTorque too, and the module has a test `validation_verified_with_force_zone` (:1555).
- Why it matters: a reader trusting the docs will misjudge what "Verified" covers (the owner relies on this badge for the reaction numbers).
- Suggested action: rewrite the two doc blocks to match the code (validation-state based; modelled set = revolute joints, driver, gravity, one actuator, force zone, external force/torque).
- Fix class: mechanical-safe (comment only)
- Confidence: certain

## LCS-05: pass-1-only result is built twice, and the transposed Jacobian SVD solve is written three times
- Kind: inefficient (repetition)
- Where: linkage-sim-rs/src/solver/reactions.rs:534-562 (`pass1_only_result`) and :577-598 (closure `pass1_only` inside `solve_reactions_inner`); SVD-solve-and-residual block at statics.rs:50-70, inverse_dynamics.rs:67-97, reactions.rs:637-652
- Evidence: the two reactions.rs blocks build an identical `ReactionSolveResult` (same `compute_validation(... pass1_eff_actuator_force(mech), pass1.condition_number, false, driver_torque)` call and same eight fields); `pass1_only_result` also does `extract_reactions` itself while the closure takes the vector. The SVD blocks all do `phi_q.transpose()`, `.svd(true,true)`, `.solve(&rhs,1e-14).map_err(|_| SvdSolveFailed)`, then a `&phi_q_t * &lambdas ...` residual; pass-2 additionally calls `phi_q.transpose()` twice (:640 and :650) and re-assembles the Jacobian (`assemble_jacobian` at :637) that `solve_statics` already built.
- Why it matters: the owner flags repetition; the same numerical recipe in three places can drift (the sign of the residual already differs in form between files). Pass 2 also repeats a full SVD per reaction pose in sweeps.
- Suggested action: have `solve_reactions_inner` and `pass1_only_result` share one `fn pass1_only(mech,q,t,reactions,&pass1)`; add `solve_transposed(phi_q, rhs) -> Result<(lambdas, svd singular values, residual)>` in statics.rs and call it from the other two places.
- Fix class: needs-judgment
- Confidence: certain

## LCS-06: Reaction extraction reads a prismatic joint's two multipliers as an [Fx, Fy] force
- Kind: inconsistent
- Where: linkage-sim-rs/src/solver/statics.rs:126-133 (`2 => { // Revolute or prismatic: [Fx, Fy] ...`), constraint rows defined in core/constraint/prismatic.rs:13-15
- Evidence: prismatic rows are `Phi[0] = n_hat_i_global . d` and `Phi[1] = theta_j - theta_i - delta_theta_0`, so lambda[0] is a force along the slide normal and lambda[1] is a moment, not global Fx and Fy. `extract_reactions` still stores `force_global = [lam[0], lam[1]]` and `resultant = sqrt(fx^2+fy^2)` for every 2-equation constraint; BL-047's notes already say "A prismatic joint's multiplier is not an [Fx, Fy] force" but only for the HTML export, not for this function. `body_equilibrium_residual` (reactions.rs:159) avoids the case by returning None for non-revolute joints.
- Why it matters: the Joint Reactions plot and report list a prismatic (slider) joint with a force and resultant built from the wrong quantities (the canvas treatment is the "pre-existing" one BL-047 mentions).
- Suggested action: decide the intended meaning (normal force along the rotated axis plus a moment) and make `extract_reactions` convert per joint type, or label these rows as raw multipliers.
- Fix class: physics-suspect
- Confidence: likely (read the constraint rows and the extraction; a slider-crank reaction compared against a free-body calculation would settle it)

## LCS-07: Cam profile "Spline" is documented as cubic but is piecewise linear, and its comments and guards disagree with the code
- Kind: inconsistent
- Where: linkage-sim-rs/src/core/constraint/cam.rs:22 (`/// Cubic spline through user-defined (theta, s) points.`), :81 (`simple piecewise-linear for now`), :88 (`// Binary search for segment` above a linear `windows(2)` scan), :98-113 (derivative is a step function; second derivative fixed at 0), :140-141/:162-163/:224-225/:256-257 (`if state.is_ground(..) {0.0} else {state.get_angle(..)}`), :177/:191 (`if let Ok(bj) = state.get_index(..)`)
- Evidence: the variant doc promises a cubic spline; `spline_eval` interpolates linearly, `spline_derivative` returns `(s_{k+1}-s_k)/dx` (discontinuous at knots, 0 outside the range) and `spline_second_derivative` returns 0.0. `State::get_angle` already returns 0.0 for ground (state.rs:124), so the five ground checks are redundant. The cam Jacobian silently skips an unknown body (`if let Ok`) where every other joint calls `.unwrap()`.
- Why it matters: a user who enters a "cubic spline" cam gets a profile with a kinked velocity and zero acceleration term, so Newton convergence and acceleration results near knots differ from what the doc promises; the redundant guards and the swallowed error add noise.
- Suggested action: rename the doc to "piecewise-linear" (or implement cubic), fix the "Binary search" comment, drop the redundant ground checks, and `unwrap`/`expect` like the other joints.
- Fix class: needs-judgment (comment fixes are mechanical-safe; choosing cubic vs linear is the owner's call and the kinked derivative is physics-suspect)
- Confidence: certain for the comment/doc mismatches

## LCS-08: solve_for_target's doc tells callers to cache the workspace probe, but there is no way to, and the app runs 64 forward solves per trajectory sample
- Kind: inefficient and inconsistent
- Where: linkage-sim-rs/src/solver/inverse_kinematics/solver.rs:113-116 (doc), :138-140 (probe run on every call); callers linkage-sim-rs/src/gui/sweep/mod.rs:1126-1138 (loop over trajectory samples, `n_probe` 64), linkage-sim-rs/src/gui/state/trajectory_ops.rs:38
- Evidence: doc: "In hot paths ... callers should cache the probe externally ... The cache lives on `AppState`". `solve_for_target` takes no probe argument, and `git grep -n "WorkspaceProbe\|probe_cache"` finds no use outside solver.rs and mod.rs re-export, so no cache exists. `compute_trajectory` calls `solve_for_target(mech,&q_prev,target,h_k,severity,u_range,u_0,nominal_rate,1e-8,50,64)` once per sample.
- Why it matters: each sample pays about 64 extra position solves (each an SVD per Newton iteration) before the real solve; a 200-sample trajectory does about 12,800 needless solves.
- Suggested action: accept an optional precomputed `&WorkspaceProbe` (compute it once in `compute_trajectory`), or correct the doc if the cost is accepted.
- Fix class: needs-judgment
- Confidence: certain

## LCS-09: Inverse-kinematics helpers repeat the singularity guard four times and the finite-difference path repeats its own solves; the analytic path the notes call faster is unused
- Kind: inefficient (repetition) and not needed
- Where: linkage-sim-rs/src/solver/inverse_kinematics/solver.rs:198-204, derivatives.rs:49-54, :100-107, :156-161 (same `eps_singularity = 1e-6 * (grad_norm*dqdu_norm).max(1e-12)` block); derivatives.rs:81 vs :100-101 (`compute_r_prime` already computed dq_du and the gradient, then both are recomputed with another SVD); `inverse_acceleration_analytic` (derivatives.rs:138) is called only from derivatives.rs tests
- Evidence: docs/ai/05-update-tracker.md:2007-2008 says the analytic version is "faster (one linear solve vs two forward solves)" and "`compute_trajectory` continues to use FD by default"; gui/sweep/mod.rs:1161 calls only `inverse_acceleration_fd`, which does 2 forward solves plus 4 Jacobian SVDs per sample.
- Why it matters: a guard threshold changed in one place will not change in the others; the FD path costs several SVDs more than it needs per sample.
- Suggested action: factor `fn dg_du_checked(mech,q,target,t) -> Result<(r_prime,dq_du)>` and use it in all four places; reuse the first `dq_du` in the FD function. Owner decides whether to switch the trajectory code to the analytic helper or delete it.
- Fix class: needs-judgment (merge guard) / owner-decision (analytic helper)
- Confidence: certain

## LCS-10: Expression drivers re-parse their formula on every evaluation, and the DriverFn doc is stale
- Kind: inefficient and inconsistent
- Where: linkage-sim-rs/src/core/driver.rs:255-259 (`ExprEval::eval` parses and binds on each call), :231-236 (comment "Parsing is sub-microsecond"), :15-17 (`DriverFn` doc: "the Rust port will add expression-based drivers (meval/rhai) in a future step")
- Evidence: `fn eval(&self,t) -> f64 { let f = self.source.parse::<Expr>().unwrap().bind("t").unwrap(); f(t) }`; each of `constraint`, `phi_t`, `gamma` calls one of three such evaluators, so every residual, Jacobian-time term and gamma assembly parses up to three strings. `expression_driver` (same file, :268) is the "future step", already shipped. For force elements the code does the opposite: `Mechanism::compile_force_modulations` (mechanism.rs:477) exists precisely "so the meval string is parsed once".
- Why it matters: a per-evaluation parse allocates and tokenizes inside the Newton loop and the RK4 stages; the two code paths for the same job (force modulation vs driver expression) behave differently, and the "sub-microsecond" claim is unmeasured.
- Suggested action: cache the compiled closure (e.g. `std::rc::Rc` is the blocker; a `Mutex`/thread-local compiled cache or dropping the `Send + Sync` bound on `DriverFn` fixes it) and fix the stale doc.
- Fix class: needs-judgment (the doc fix alone is mechanical-safe)
- Confidence: certain on the re-parse; likely on cost (a benchmark of one sweep would settle it)

## LCS-11: Every solver call goes through String-keyed HashMap lookups and builds full-width matrices per joint
- Kind: inefficient
- Where: linkage-sim-rs/src/core/state.rs:82-89, :105-129 (`get_index`, `get_pose`, `get_position`, `get_angle` each hash the body id with the default SipHash), :179-192 (`body_point_global` does two lookups); linkage-sim-rs/src/solver/assembly.rs:13-94 (four copies of the same stack-the-rows loop, each joint allocating an n_eq x n_coords `DMatrix` that is then copied element by element); linkage-sim-rs/src/core/mechanism.rs:133-138 (`n_constraints()` re-sums virtual calls on every assemble); linkage-sim-rs/src/solver/forward_dynamics.rs:90-146, :165-168 (dense (n+m)^2 copy loops; `a.clone().lu()` clones the whole matrix, then `a.clone().svd` clones again on the fallback)
- Evidence: `assemble_constraints`, `assemble_jacobian`, `assemble_phi_t`, `assemble_gamma` differ only in the method called; `Mechanism` already stores `constraint_ranges` after `build()` (mechanism.rs:407) with the total row count and the row start of each constraint, but nothing in assembly uses it. A revolute `jacobian` (constraint/helpers.rs:20-36) does 2 `is_ground` string compares, 2 `get_index` hashes and 2 more through `body_point_global_derivative` per call.
- Why it matters: this is the per-Newton-iteration, per-RK4-stage, per-sweep-point path; the owner flags repetition and per-step allocation. The string hashing and allocation dominate for 4-bar sizes.
- Suggested action: (1) one generic `assemble_rows(mech, |joint| ...)` helper writing into a preallocated matrix using `constraint_ranges`; (2) have `build()` store each joint's `q_start` indices (or switch `State`'s map to a faster hasher) so the hot path does no string hashing; (3) take `n_constraints` from the stored ranges; (4) in `solve_augmented` use `a.clone().lu()` once and reuse the clone for the SVD fallback, and use `view_mut` block copies instead of the element loops.
- Fix class: needs-judgment
- Confidence: likely (read the code; a profile of one sweep would give the real shares)

## LCS-12: Dead or test-only public items, a stale allow, and config fields that nothing reads
- Kind: not needed
- Where: linkage-sim-rs/src/core/state.rs:132 (`State::set_angle`); linkage-sim-rs/src/core/constraint/mod.rs:69 (`JointConstraint::is_cam_follower`); linkage-sim-rs/src/solver/statics.rs:171 (`extract_reactions_with_local`, used only by its own test at :330; `JointReaction::force_local_i/j` are filled only there and zero elsewhere); linkage-sim-rs/src/core/constraint/prismatic.rs:23 (`#[allow(dead_code)]` on `axis_local_i`, which `axis_local_i()` reads and io/to_json.rs:124 calls); linkage-sim-rs/src/solver/forward_dynamics.rs:34-37 (`rtol`, `atol` "reserved for future use")
- Evidence: `git grep` for each name over src, tests, scripts, web: `set_angle(` has no caller (the only other hit is the Python twin), `is_cam_follower` only in docs/history, `extract_reactions_with_local` only at its definition and its test, `rtol`/`atol` only in the struct, `Default`, and struct literals in tests (golden_fixtures.rs:705, forward_dynamics.rs:755) that set them without effect.
- Why it matters: dead surface area to read and maintain; the stale `allow` hides nothing but misleads; `rtol`/`atol` in a test config give the false impression the integrator is tuned by them.
- Suggested action: delete `set_angle`, `is_cam_follower`, the `allow(dead_code)` line; delete `rtol`/`atol` and their literals; remove `extract_reactions_with_local` and the two `force_local_*` fields (or wire them to a user). Check the web build too: none of these sit behind a wasm32 cfg.
- Fix class: mechanical-safe for set_angle, is_cam_follower and the stale allow; owner-decision for rtol/atol and the local-frame reactions
- Confidence: certain

## LCS-13: ControlTarget panics on bad data it can load from a file, repeats its normalisation three times, and hard-codes "ground"
- Kind: inconsistent
- Where: linkage-sim-rs/src/solver/inverse_kinematics/control_target.rs:64, :124, :217 (`assert!(dir_norm > 1e-12, "Projection axis_dir is zero")` in `evaluate`, `gradient`, `hessian`); :59-66, :122-125, :215-218 (same normalise block); :282, :290, :300, :316, :336 (`"ground"` literals; the constant is `GROUND_ID` in core/state.rs:11); `Vector2::new(local_pt[0], local_pt[1])` repeated 14 times
- Evidence: the enum derives `Serialize, Deserialize` (line 16), so a saved file can hold `Projection { axis_dir: [0,0] }` and bypass the constructor that rejects it; the first `evaluate` call then panics instead of returning an error. The constructor already normalises `axis_dir`, so the three runtime normalisations are only a safety net.
- Why it matters: a malformed file can crash the GUI mid-solve; repeated blocks drift.
- Suggested action: validate on deserialise (or return `Result`/NaN-safe fallback in `evaluate`), add one `fn unit_axis(&self)`, and use `GROUND_ID`.
- Fix class: needs-judgment
- Confidence: likely (whether a loaded file can reach it depends on io/from_json.rs, which is outside this part)

## LCS-14: solve_for_target maps "no probe sample converged" to SvdSolveFailed, and its branch-jump threshold is a heuristic from the centre of gravity
- Kind: inconsistent
- Where: linkage-sim-rs/src/solver/inverse_kinematics/solver.rs:142-144 (`return Err(LinkageError::SvdSolveFailed)`), :185 and :263-281 (`branch_threshold = 0.5 * max_body_length(mech)`)
- Evidence: an empty probe means every forward Newton solve failed to converge; the error returned is "SVD solve failed." (error.rs:41), which no SVD produced. `max_body_length` computes `2*|cg_local|` per body and floors at 0.01 m, with the comment "heuristic only"; a body whose CG sits at its origin counts as zero length, so for a large mechanism of such bodies the threshold falls to 5 mm and ordinary steps would be flagged BranchJump.
- Why it matters: the message misleads the user and the support trail; the threshold is not tied to the mechanism's real size.
- Suggested action: return `LinkageError::PositionSolveNotConverged` (or a new variant); compute the extent from attachment points.
- Fix class: physics-suspect (the threshold) / needs-judgment (the error variant)
- Confidence: likely (the variant mismatch is certain; the false-BranchJump case depends on how the GUI builds bodies)

## LCS-15: Small inconsistencies between sibling APIs in body.rs and mechanism.rs
- Kind: inconsistent
- Where: linkage-sim-rs/src/core/body.rs:209-224 vs :253-268; linkage-sim-rs/src/core/mechanism.rs:446-455
- Evidence: `add_mount_point` rejects a name that collides with an attachment point (`MountPointNameCollision`), but `add_attachment_point` checks only other attachment points, so adding an attachment named like an existing mount point succeeds and `resolve_force_point` (:270) then silently prefers the attachment. `remove_force(idx)` panics on a bad index while `replace_force(idx, ..)` silently does nothing (`if idx < self.forces.len()`).
- Why it matters: order-dependent behaviour (mount-then-attach vs attach-then-mount) and a silently dropped edit.
- Suggested action: make `add_attachment_point` check `mount_points` too; make `replace_force` return a `Result`/bool or panic like `remove_force`.
- Fix class: needs-judgment
- Confidence: certain

## LCS-16: Two tests assert nothing that can fail, and a fourth copy of the same 4-bar fixture sits in each module
- Kind: inconsistent (weak tests) and inefficient (repetition)
- Where: linkage-sim-rs/src/solver/inverse_kinematics/severity.rs:65-79 (`inverse_solve_status_constructs_all_variants` only builds values into `_` bindings); linkage-sim-rs/src/solver/statics.rs:283 (`assert!(jr.resultant >= 0.0)` on a square root); 4-bar builders: solver/kinematics.rs:130, reactions.rs:714 and :742, statics.rs:207, inverse_dynamics.rs:123, core/mechanism.rs:549, plus analysis/validation.rs:301, analysis/virtual_work.rs:137, gui/export/schematic.rs:812 (leads.md section 4 lists the repeated runs: reactions.rs:743-761, 1564-1577, 1634-1646, 1980-1992, inverse_dynamics.rs:129-144, linear_driver.rs:474-494 vs :431, test_helpers.rs:17-32 vs schematic.rs:813)
- Evidence: the severity test has no assertion at all (a compile-time check only); `resultant = (fx*fx+fy*fy).sqrt()` is never negative. Each module builds the same 4-bar by hand because the shared helper `solver/inverse_kinematics/test_helpers.rs` is `pub(crate)` only inside the IK module.
- Why it matters: weak tests give false confidence; fixtures repeated in 8+ places must each be edited when the model changes.
- Suggested action: delete the two assertions or replace them with a real check; move the 4-bar builder to one `#[cfg(test)]` module at the crate root and reuse it.
- Fix class: mechanical-safe for the two assertions; needs-judgment for the fixture move
- Confidence: certain

## Leads settled
- `uuid` (unused dependency): unsure. No Rust file names it, but Cargo.lock shows it pulled in by `dxf`; the direct entry with `features = ["js"]` is likely a wasm32 feature pin. Kept in LCS-01 as owner-decision.
- `wasm-bindgen-futures` (unused dependency): real, see LCS-01 (no `wasm_bindgen_futures` or `spawn_local` anywhere).
- `JointConstraint::is_cam_follower` (core/constraint/mod.rs:69): real, dead, see LCS-12.
- `State::set_angle` (core/state.rs:132): real, dead, see LCS-12.
- Duplicate run core/constraint/fixed.rs vs revolute.rs (11 lines): real but not worth a finding; it is the unavoidable `Constraint` trait boilerplate (id, bodies, n_equations) plus the shared `translational_*` helper calls, already factored into helpers.rs.
- Duplicate run core/mechanism.rs vs solver/kinematics.rs (7 lines): real, test-only 4-bar builder, see LCS-16.
- Duplicate runs reactions.rs (715/718 with 743-761, 1564-1577, 1634-1646, 1868-1883, 1980-1992), inverse_dynamics.rs:129-144, linear_driver.rs:474-494, inverse_kinematics/test_helpers.rs:17-32 with gui/export/schematic.rs:813, solver/inverse_dynamics.rs and statics.rs (5 lines): real, all test fixtures, see LCS-16 (statics/inverse_dynamics non-test SVD block is LCS-05).
- Longest-function and "file nothing names" leads: none fall inside this part.
- `update_prismatic_axis`, `driver_lambda_units` and the other items of leads.md section 2 outside core/solver/geometry: not mine.

## Not read
- Test modules (about the last half) of linkage-sim-rs/src/solver/forward_dynamics.rs (from line 600), solver/reactions.rs (from line 703), solver/inverse_dynamics.rs (from line 259), solver/events.rs (from line 184, apart from the first tests), solver/inverse_kinematics/control_target.rs (from line 420 apart from 600-640), core/mechanism.rs (from line 542), core/linear_driver.rs (from line 330), core/driver.rs (from line 300), core/state.rs (from line 260), core/constraint/mod.rs (from line 330), core/constraint/cam.rs (from line 270), core/body.rs (from line 420), solver/statics.rs (from line 190, apart from grep), solver/inverse_kinematics/derivatives.rs (from line 230), solver/inverse_kinematics/test_helpers.rs
