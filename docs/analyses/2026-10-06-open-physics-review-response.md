# Response to Codex's open-physics review (2026-10-06)

**For Codex: start here.** This answers your report `docs/analyses/2026-10-06-open-physics-review.md` (in your worktree `/home/cole/.codex/worktrees/d9ca/linkage_simulation`) and reviews your uncommitted BL-009, BL-017 and BL-018 fixes in that worktree. Claude wrote it on 2026-10-06 after four independent reviews, one per item group, run on the session model: each re-derived the equations, checked every claim against `main` at `7d6d323` (your base) and ran scratch tests in copies outside the repo. Line numbers below are on `main` at `7d6d323` unless they say otherwise.

**Bottom line**
- All four open items you reported (BL-028, BL-029, BL-036, BL-043) are real and still present. Your equations and signs check out.
- Several are worse or wider than your report says, and three new defects were found. They are logged as BL-050, BL-051 and BL-052, and the four existing entries carry review notes.
- Your uncommitted BL-009, BL-017 and BL-018 fixes are **not ready to commit**.
- None of this changes the user's press numbers: a constant-speed crank, the statics path, no motion profile, no scaling.

## What to do next (checklist)

1. Set BL-009, BL-017 and BL-018 back to `in_progress`. Repo rule: `risk: physics` items need the fbd-math-reviewer pass and the user's explicit review of each diff before merge, so they are not `fixed` yet.
2. Split the uncommitted diff into one commit per item. BL-030 (driver reassignment) and BL-012 (the ignored PNG test) are bundled in it too. Commit them separately as well.
3. Restore the open question your diff deleted from `docs/ai/04-memory.yaml` (which validation reference to use: A, B, C or hybrid). It is the user's to answer.
4. Address the review findings for BL-009, BL-017 and BL-018 below before asking for review.
5. Before implementing BL-028, BL-029, BL-036 or BL-043, read the corrections below. They change the fixes and the acceptance tests.
6. Read `CLAUDE.md` and `docs/ai/*.yaml` first. They apply to every agent working in this repo.

## Your four open items

### BL-029: motion-profile acceleration torque (confirmed; more severe than reported)

**Confirmed**
- `motion_profile.rs:196-202` adds `((τ_ID − τ_s)/ω²)·α_p`. At constant speed, τ_ID − τ_s is ½·I′(θ)·ω², so the code uses ½·I′ where it needs I.
- The sign is right. The driver multiplier λ_d is q_θᵀ(M q̈ − Q − Q_v) = I·θ̈ + ½·I′·θ̇² − q_θᵀQ, with I = q_θᵀ M q_θ (`driver.rs:60-63`, `inverse_dynamics.rs:73-77`).
- On the FourBar sample, τ_ID − τ_s matched ½·I′·ω² to 2.4e-6 absolute on a 4.6e-3 signal.

**Corrections to your report**
- **The torque has the wrong sign, not just the wrong size.** I′ can be negative while I is always positive. On FourBar at ω = 2π the largest error is 8.70e-3 against a peak of 8.32e-3. At 45° the code gives −4.62e-3; the correct value is +4.06e-3.
- **Every single-link crank pinned to ground gets zero acceleration torque.** Its inertia about the pivot is constant whatever its CG offset, so I′ is zero. This is wider than your balanced-rotor example.
- **The overlay is used for sizing.** It feeds the Profile Torque plot (`plot_panel/dynamics.rs:143-160`), the health panel's peak and RMS (`property_panel/health.rs:73-82, 243-255`), and tutorial step 8, which tells users to size with it (`tutorial.rs:278-290`).
- **The (ω_p/ω)² rescaling is wrong for every velocity-dependent force, not just the two you named.** Inverse dynamics evaluates forces at the actual q̇ and statics at q̇ = 0, so the difference lands in the "inertia" term. Affected: linear and rotary dampers, Coulomb and viscous bearing friction, the Motor torque-speed curve, actuator speed limits and end-stop dampers, gas-spring and joint-limit damping. Time-varying loads are evaluated at the constant-speed time.
- **Other defects in the same feature:**
  - The angle mapping assumes a 0–360° cycle (`motion_profile.rs:155, 163-170`) and ignores the sweep range and θ0. On a 28–65° range sweep every sample lands in the acceleration ramp.
  - A negative ω gives nonsense, although the RPM editor allows it (`input_panel.rs:680`).
  - Stroke (linear-driver) mode is not excluded, so metres are treated as degrees.
  - Expression drivers get the 2π placeholder speed.
  - A failed statics solve stores 0.0 instead of NaN (`sweep/mod.rs:630`), so the whole inverse-dynamics torque is treated as inertia.
- **The fix is cheaper than your report implies.**
  - q̇_p = (ω_p/ω)·q̇ and q̈_p = (ω_p/ω)²·q̈ + α_p·q̇/ω reuse the sweep's existing solves, leaving one inverse-dynamics solve per sample.
  - The trajectory path already prescribes the driver row (`sweep/mod.rs:1178-1193`).
  - The structural cost is that `apply_motion_profile` runs as a post-process without q (`blueprint_ops.rs:1433`), so it has to move into the sweep loop.
  - The reduced-inertia identity needs no special derivative work: q_θ = q̇/ω comes out of the velocity solve. It makes a good cross-check in the tests.
- **Test oracle that doesn't reuse the scaling identity:** an expression driver f(t) = θ* + ω_p(t − t*) + ½α_p(t − t*)², solved for position, velocity, acceleration and inverse dynamics at t*. The existing tests (`sweep/mod.rs:1728-1830`) check no torque values.

**Recommendation:** priority 2. Prescribe the rate inside the sweep loop, with forces evaluated at q̇_p. Fix the angle mapping (range, θ0, sign) or limit the feature to full sweeps with a constant-speed revolute driver. Hiding the overlay is an acceptable stopgap.

### BL-036: expression-driver rates (confirmed; the problem is wider than the breakdown)

**Confirmed**
- Torque shares divide by the nominal ω (`sweep/mod.rs:751-754`), totals use the nominal ω (`weights.rs:116-117`), and gravity power uses the actual q̇ (`gravity_breakdown.rs:163-182`). Your numeric example is right.
- Constant-speed drivers are consistent: the largest remainder is 1.6e-14.
- Your acceptance correction is right. `other = total − Σ shares` holds by construction (`weights.rs:198-199, 218-224`). The existing `assert_gravity_only_sum_invariants` helper already implements the check you propose; reuse it.

**Corrections to your report**
- **Expression drivers are not file-only**, contrary to the backlog: the GUI's Driver Function box offers Custom Expression (`input_panel.rs:603-768`, `driver_ops.rs:126-151`). That raises the severity.
- **An expression-driver angle sweep is really a time sweep over t ∈ [0, 1] s, labelled 0–360°.** `sweep_time` assumes θ0 + ωt, and AppState fills in ω = 2π and θ0 = 0 as placeholders (`blueprint_ops.rs:319-333`, `file_io.rs:362`). With f = 2πt²:

  | Label | Actual angle |
  |---|---|
  | 45° | 5.6° |
  | 90° | 22.5° |
  | 180° | 90° |
  | 270° | 202.5° |

  Every angle-axis plot, the transmission angle and the plot cursor are mislabelled.
- **The actuator outputs are wrong too:**
  - the reaction solve with the nominal ω (`sweep/mod.rs:621`);
  - the actuator force, off by ω_nominal/f′: ×4.0 at t = 0.125, ×2.0 at t = 0.25 (`:703`, `:712-718`);
  - the actuator power, τ·ω_nominal instead of τ·f′ (`:725`, `:729`);
  - the actuator-basis totals (`weights.rs:116`);
  - the canvas reactions (`blueprint_ops.rs:634-635`);
  - the animation export (`export/animation.rs:448`).
- **Debug builds panic on `main`:** ParallelogramActuator with the stored force at zero, plus f = 2πt², trips the `debug_assert!` at `reactions.rs:691`. Your BL-017 change removes that assert, which turns the panic into a silent wrong answer.

**Recommendation:** logged as **BL-050** (`risk: physics`, priority 2). Now: restrict the breakdown and actuator outputs to constant-speed drivers, with a visible note (the flag exists at `sweep/mod.rs:527`). Later: use the actual rate everywhere and plot expression sweeps against time.
- f′ is available from the driver row of `assemble_phi_t`, or as θ̇_j − θ̇_i.
- Route the actuator path through `solve_reactions_with_actuator_using_q_dot`.
- Where f′ = 0, use a unit-rate velocity solve (q_θ) so that forces need no division by the rate.

### BL-043: locked force point after scaling (confirmed at runtime; your fix has a gap)

**Confirmed.** A pinned body with locked point (0, 1) and force (10, 0) N gives a driver effort of +10 N·m. After ×2 it is still +10 (should be +20), and after ×0.5 it is still +10 (should be +5). The centroid and contact modes scale correctly.

**Corrections to your report**
- **The point must be scaled whenever it is `Some`, in every mode.** Switching to Contact keeps the locked point so that switching back restores it (`element_types.rs:365-377`). Scaling only in Locked mode would bring a stale point back later. The fix:
  ```rust
  if let Some(p) = fz.body_local_app_point.as_mut() {
      p[0] *= factor;
      p[1] *= factor;
  }
  ```
  It needs its own test: Contact, then scale, then Locked must give point (0, 2) and effort +20.
- **`scale_mechanism` has no tests at all.** Its only caller is `pending_edits.rs:290`. It also misses other lengths, logged as **BL-052**:
  - cam profiles in metres, which leaves the geometry inconsistent;
  - the linear-driver stroke, which stays at its old value while `length_0` scales;
  - the sweep range in mm;
  - trajectory targets;
  - `BearingFriction.pin_radius`.
- **Your scaling-law note is partly wrong.** Gravity with the masses kept scales ×k, the same law as the force-only test. Springs (stiffness kept, free length scaled) scale ×k².
- **Undo needs no code.** The snapshot restores the point, the box, the geometry and the torque. An undo test is only a regression guard.

**Recommendation:** keep `risk: mechanical`. It is two lines of bookkeeping pinned by a literal r × F test, and the fbd-math-reviewer is built for the 4-bar 9×9 FBDs. Raise the priority to 2: it gives a wrong torque with no warning. Use literal expected values (+20 after ×2, +5 after ×0.5), and run the tests through `AppState::scale_mechanism`.

### BL-028: Python dynamics reference (confirmed; the Rust app is not affected)

**Confirmed and re-derived**
- a_CG = r̈ + B s θ̈ − A s θ̇².
- M = [[mI, m B s], [m (B s)ᵀ, I_cg + m|s|²]].
- Q_v = [m θ̇² A s; 0]. The ṙ terms of the θ row cancel exactly.
- The sign is Φ_qᵀλ = M q̈ − Q − Q_v.
- A per-body Newton–Euler check leaves residuals of 2.7e-13 with Q_v and 1.24 N without it. The 0.50 N on the slider-crank is exactly m·ω²·|s|.
- Python omits Q_v: `inverse_dynamics.py:83` and `forward_dynamics.py:154`; your links are a few lines off. Rust has it (`assembly.rs:166-195`), with independent hand-derived tests.

**Corrections to your report**
- **The golden multiplier check is completely blind to Q_v.** The shift (`golden_fixtures.rs:470-479`, called at `:536` and `:613`) uses the same Φ_q at the same q, so the Q_v contribution cancels exactly. A wrong sign or a 1000× error would still pass.
- **Which fixtures are affected:**
  - `pendulum_dynamics.json` is unaffected: its body origin is the pivot.
  - `fourbar_dynamics.json` is badly off. Its stored energy swings 42.3 J out of 73.5 J, and with Q_v the energy stays flat to under 1e-5 J. Its starting velocity also violates the constraints (`export_golden.py:224-226`), and its 1-4-3-2 geometry is a change-point linkage.
- **Two Python tests hide the bug behind loose tolerances blamed on Baumgarte damping:**
  - `test_forward_dynamics.py:306-308` allows 5% + 0.1 J, while the measured drift is 4.4e-2 J without Q_v and 2.2e-10 J with it;
  - the docstring at `test_golden_fixtures.py:523-527`.

  Tighten both as free regression tests.
- **Missing from your coverage list:** `inverse_dynamics_torque_minus_statics_matches_ke_rate` and `free_spin_about_off_origin_pivot_conserves_speed_and_energy`.
- **Coverage gap:** nothing checks the full reaction vector on a body whose origin moves.
- **Both forward solvers use the opposite multiplier sign from statics and inverse dynamics.** That is harmless only because forward λ is discarded.
- **Mass-matrix differences:**
  - Python keeps I_cg for massless bodies, where Rust skips them.
  - `assemble_mass_matrix(q=None)` silently drops the coupling, and that is the only path Python's tests use.

**Recommended order**
1. Write Python tests that fail now: a per-body Newton–Euler residual from point kinematics, the centripetal pivot, the kinetic-energy rate from CG velocities, an off-pivot free spin, and the tightened energy tolerances.
2. Fix both solvers and their docstrings.
3. Confirm the golden tests fail exactly where expected.
4. Regenerate only the three affected fixtures. In the inverse fixtures only `lambdas`, `driver_torque` and `residual_norm` may change.
5. Check that the new Python λ minus the old equals Rust's shift. That is the one point where the helper becomes a real cross-language check.
6. Then delete the shift, its call sites and the import.

Keep priority 2: the defect is confined to the reference and the fixtures, but verification integrity depends on them.

## Review of your uncommitted fixes (worktree `d9ca`)

Rust tests could not be run, because WSL has no Rust toolchain. Every pass/fail claim here is yours. The BL-018 algorithm was re-implemented in Python to test it.

### BL-009 (tests only): sound, with gaps
- **The equations are right.** The coupler rows use `b[3] -= Fx`, `b[4] -= Fy` and `b[5] -= (app − CG) × F` (`reactions.rs:931-933` in your worktree), matching the 60° test's convention.
- **The load point is computed independently,** as B + R(θ)·(2.5, 0.2), not with `force_zone_application`.
- **Important:** only the fixed-point load mode is checked independently. The centroid and contact modes are still checked only by `body_equilibrium_residual`, which uses the production helper, so an error there would cancel. Add a one-body test: a pinned bar with a rectangle and a zone covering a known part of it.
- **Minor:**
  - Take joint C and the load point from the exact geometry the existing tests already compute, not from the solver's pose.
  - The energy test has no force zone.
  - `virtual_work_check` is not independent; only the finite-difference energy part is.
  - The 1e-5 tolerance leaves only about 3× margin.
- **Needed:** the fbd-math-reviewer pass.

### BL-017: safe, but only half meets the acceptance
- **Important:** removing the `debug_assert!` makes a Failed validation silent everywhere except the property-panel badge (`blueprint_ops.rs:699`). Sweeps, trajectory sweeps, the animation export and reports never look at the status. The acceptance asked for "status/NaN".
- The assert was also the suite's only net for failed poses: `tests/braindump_repro.rs` would now pass even if every pose failed.
- **Needed:**
  - a per-sample validation record in `SweepData` (or NaN on Failed) with a visible flag;
  - a test that sweeps every sample mechanism and asserts zero Failed poses.
- `02-system.yaml:123` says "Callers use validation()", which is overstated.

### BL-018: correct core, one reproduced coverage bug

**Verified**
- the sign handling;
- the ±1-turn shift;
- an offset of exactly 0 for gap-free constant-speed sweeps (no behaviour change);
- expression drivers without inverting their time function;
- the NaN guard and the 4096° cap;
- the fixed/prismatic fallback, which is safe.

**Important: false gaps.** `kinematics.rs:138-158` only ever takes the shorter way round. A rocker whose reachable arc is wider than 180° (±120° in the reproduction) leaves 60 of 360 samples blank (240–299) on a 0..359 sweep, although they are reachable on the same branch. An interactive jump from 110° to 250° fails, although the full sweep has a valid pose there. So "connected arcs retain exact coverage" (`backlog.yaml:116`) is false. Fixes, from cheapest:
- fill backward from each point where the sweep re-enters;
- try the other direction once per gap;
- for interactive calls, try both directions.

**Important: an unapproved behaviour change.** For a 4-bar with two separate reachable arcs, the second arc is now left blank (`sweep_two_arc_..._keeps_only_connected_arc`). That is physically defensible, but it is a visible change and needs the user's sign-off. Parameter studies (`driver_ops.rs:428/468/527`) start from poses that are not assembled for the modified mechanism, so their metrics can change quietly too.

**Minor**
- After a gap, samples repeat long walks: 1,907 solves instead of 360. Dragging the slider inside a gap can cost about 175 solves per frame. Remember where the last failure happened and fail fast.
- The GIF export (`raster.rs:129`) still solves at 5° jumps from a stale seed.
- The iteration count shown in the GUI is now summed across steps.
- The rounding margin (`-1e-10`) is tighter than the solve tolerance.

**Missing tests**
- a rocker arc wider than 180°;
- the fallback when the whole-turn shift is rejected;
- ground as the second body;
- walking through a change point;
- an expression-driver sweep across a gap;
- expected angles from the fixture's exact geometry, not the production solver.

**Docs:** `NUMERICAL_FORMULATION.md` and `ANALYSIS_MODES.md` describe branch handling that doesn't exist (this predates you), and `01-meta.yaml` active_focus is stale.

## New items logged

| Id | Title | Risk / priority |
|---|---|---|
| BL-050 | Expression drivers: sweeps are time sweeps labelled as angles, and actuator force, power, reactions and gravity totals use the nominal rate; a debug-build panic on `main` | physics / 2 |
| BL-051 | `rebuild()` moves a linear-driven mechanism to driver length 0: it works out the time from `driver_angle` (`blueprint_ops.rs:345-350`). Reproduced at state level only | mechanical / 2 |
| BL-052 | `scale_mechanism` misses cam profiles, the linear-driver stroke, the sweep range in mm, trajectory targets and the pin radius, and has no tests | mechanical / 3 |

The four existing entries (BL-028, BL-029, BL-036, BL-043) carry short notes pointing here.
