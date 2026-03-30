# Phase 1 Validation Plan

## Objective

Validate correctness, robustness, and regression resistance of all Phase 1
kinematics, topology analysis, and force-helper features through independent
cross-checks, external ground truth, adversarial inputs, and failure-mode
verification.

## Validation Layers

1. **Internal numerical consistency** — solver pipeline self-consistency
2. **External analytical/textbook comparison** — published formulas and theorems
3. **Property-based invariance and adversarial cases** — symmetry, frame independence, randomized inputs
4. **Failure-mode and diagnostics validation** — correct behavior under degenerate/singular/impossible inputs
5. **Performance and regression baselines** — iteration counts, sweep success, golden snapshots

## Deliverables

| # | Artifact | Description |
|---|----------|-------------|
| D1 | `tests/test_numerical_consistency.py` | Layer 1 — full pipeline self-consistency |
| D2 | `tests/test_textbook_benchmarks.py` | Layer 2 — external analytical ground truth |
| D3 | `tests/test_invariance.py` | Layer 3a — deterministic invariance and force validation |
| D4 | `tests/test_property_stress.py` | Layer 3b — Hypothesis-driven stress tests |
| D5 | `tests/test_failure_modes.py` | Layer 4 — failure, diagnostics, degenerate inputs |
| D6 | `tests/test_regression_baselines.py` | Layer 5 — iteration counts, sweep success, golden snapshots |
| D7 | `tests/golden/fourbar_snapshot.json` | Golden data: 4-bar at 4 canonical angles |
| D8 | `tests/golden/slidercrank_snapshot.json` | Golden data: slider-crank at 4 canonical angles |
| D9 | `tests/golden/sixbar_snapshot.json` | Golden data: 6-bar at 4 canonical angles |
| D10 | Traceability matrix (this document) | Feature-to-test mapping, maintained as tests are written |
| D11 | Tolerance calibration comments | Top-of-file block in each test file documenting chosen h, epsilon, and calibration rationale |

## Acceptance Criteria

| # | Criterion | How to verify |
|---|-----------|---------------|
| AC1 | Every feature F1-F17 has at least one new validation path beyond existing tests | Traceability matrix has no empty "New validation" cells |
| AC2 | Every solver stage (position, velocity, acceleration) cross-checked by at least one independent method | L1 FD chain tests pass for all 3 benchmark mechanisms |
| AC3 | At least 2 features validated against external ground truth independent of solver implementation | Roberts' cognate + slider-crank closed-form both pass |
| AC4 | All known singular/redundant/infeasible scenarios fail predictably per the failure policy table | L4 tests pass with expected PositionSolveResult states and diagnostics |
| AC5 | No regression in convergence behavior on canonical mechanisms | L5 iteration-count and sweep-success tests pass against calibrated baselines |
| AC6 | Golden snapshots committed and reproducible | L5 snapshot tests pass at 1e-10 tolerance |
| AC7 | Force helpers validated via virtual work principle | L3a virtual work tests pass |
| AC8 | All Hypothesis tests deterministic in CI | `derandomize=True` on every `@given` test, no CI flakiness over 10 consecutive runs |
| AC9 | FD tolerances calibrated, not guessed | Each test file contains calibration comment block with empirical h-vs-error analysis |
| AC10 | All existing 395 tests still pass | `pytest` full suite green after all new tests added |

---

## Scope: Phase 1 Feature List

Every implemented feature, explicitly enumerated:

| # | Feature | Module |
|---|---------|--------|
| F1 | Revolute constraint (residual, Jacobian, gamma, phi_t) | `core/constraints.py` |
| F2 | Prismatic constraint (residual, Jacobian, gamma, phi_t) | `core/constraints.py` |
| F3 | Fixed constraint (residual, Jacobian, gamma, phi_t) | `core/constraints.py` |
| F4 | Revolute driver (residual, Jacobian, gamma, phi_t) | `core/drivers.py` |
| F5 | Newton-Raphson position solver | `solvers/kinematics.py` |
| F6 | Velocity solver (Phi_q * q_dot = -Phi_t) | `solvers/kinematics.py` |
| F7 | Acceleration solver (Phi_q * q_ddot = gamma) | `solvers/kinematics.py` |
| F8 | Coupler point position/velocity/acceleration | `analysis/coupler.py` |
| F9 | Sweep continuation | `solvers/sweep.py` |
| F10 | Gruebler DOF analysis | `analysis/validation.py` |
| F11 | Jacobian rank analysis (SVD) | `analysis/validation.py` |
| F12 | Graph connectivity (BFS) | `analysis/validation.py` |
| F13 | point_force_to_Q | `forces/helpers.py` |
| F14 | body_torque_to_Q | `forces/helpers.py` |
| F15 | gravity_to_Q | `forces/helpers.py` |
| F16 | assemble_Q | `forces/assembly.py` |
| F17 | Rotation matrices A(theta), B(theta) | `core/state.py` |
| F18 | State vector bookkeeping | `core/state.py` |

---

## Traceability Matrix

| Feature | Existing coverage | New validation (layer) | Test file | Ground truth | Priority | Status |
|---------|------------------|----------------------|-----------|-------------|----------|--------|
| F1 Revolute | FD Jacobian, 4-bar benchmark | Residual across sweep (L1), distance invariants (L1), cognate theorem (L2) | `test_numerical_consistency`, `test_textbook_benchmarks` | Internal numerical, Roberts' theorem | P1 | new |
| F2 Prismatic | FD Jacobian, slider-crank benchmark | Residual across sweep (L1), closed-form comparison (L2), frame rotation invariance (L3) | `test_numerical_consistency`, `test_textbook_benchmarks`, `test_invariance` | Internal numerical, Norton formulas | P1 | new |
| F3 Fixed | FD Jacobian, basic tests | Equivalent-mechanism modeling (L3), redundancy detection (L4) | `test_invariance`, `test_failure_modes` | Cross-model agreement | P2 | new |
| F4 Driver | FD Jacobian, hypothesis tests | FD velocity/accel chain (L1), driver-at-singularity behavior (L4) | `test_numerical_consistency`, `test_failure_modes` | Internal numerical | P1 | new |
| F5 Position solve | Convergence on benchmarks | Bad-guess recovery (L4), infeasible-assembly failure (L4), iteration count baselines (L5) | `test_failure_modes`, `test_regression_baselines` | Expected failure behavior | P1 | new |
| F6 Velocity solve | FD on benchmark mechanisms | FD on sweep positions vs solver (L1), slider-crank closed-form vel (L2) | `test_numerical_consistency`, `test_textbook_benchmarks` | Internal numerical, Norton formulas | P1 | new |
| F7 Acceleration solve | FD on benchmark mechanisms | FD on sweep velocities vs solver (L1), slider-crank closed-form accel (L2) | `test_numerical_consistency`, `test_textbook_benchmarks` | Internal numerical, Norton formulas | P1 | new |
| F8 Coupler kinematics | Basic coupler tests | FD on coupler positions to vel to accel for 4-bar and 6-bar (L1), 4-bar coupler curve shape (L2) | `test_numerical_consistency`, `test_textbook_benchmarks` | Internal numerical, known coupler geometry | P1 | new |
| F9 Sweep continuation | Sweep + branch tracking | Branch continuity test (L4), full-rotation sweep success rate (L5), step-count baseline (L5) | `test_failure_modes`, `test_regression_baselines` | Expected behavior | P2 | new |
| F10 Gruebler DOF | Basic mechanisms | Grashof classification cross-check (L2), overconstrained detection (L4), topology cross-consistency (L4) | `test_textbook_benchmarks`, `test_failure_modes` | Textbook classification | P2 | new |
| F11 Jacobian rank | Redundant constraint test | Rank drop at toggle (L1), condition number at near-singular (L4), topology cross-consistency (L4) | `test_numerical_consistency`, `test_failure_modes` | Known toggle positions | P1 | new |
| F12 Connectivity | BFS basic tests | Disconnected topology detection (L4), multi-component graph (L4), topology cross-consistency (L4) | `test_failure_modes` | Expected failure | P3 | new |
| F13 point_force_to_Q | 23 force helper tests | Frame rotation invariance (L3), virtual work consistency (L3) | `test_invariance` | Virtual work principle | P2 | new |
| F14 body_torque_to_Q | 23 force helper tests | Frame rotation invariance (L3), torque-only virtual work (L3) | `test_invariance` | Virtual work principle | P2 | new |
| F15 gravity_to_Q | 23 force helper tests | Frame rotation invariance (L3), gravity potential energy consistency (L3) | `test_invariance` | delta_W = -mg * delta_h | P2 | new |
| F16 assemble_Q | Basic assembly tests | Superposition check (L3), zero-force baseline (L3) | `test_invariance` | Linearity | P3 | new |
| F17 A(theta), B(theta) | Implicit in constraint tests | Orthogonality (L3), B = dA/dtheta via FD (L3), det(A)=1 (L3) | `test_invariance` | Matrix identities | P2 | new |
| F18 State bookkeeping | 28 tests | Covered adequately | - | - | - | existing |

**Maintenance rule:** Any new Phase 1 feature or solver behavior added after this spec must update the traceability matrix in the same PR.

---

## Global Definitions

### Branch Invariant

For a mechanism in configuration q, select three non-collinear points A, B, C
on distinct bodies in the kinematic chain (typically: ground pin, coupler
attachment, output pin). Define:

    s = sign((B - A) x (C - A))

where `x` is the 2D cross product (scalar). Branch continuity means `s` does
not change sign between consecutive sweep steps except at singular/toggle
configurations where the cross product may pass through zero.

All tests referencing "branch invariant" use this definition. The specific
point triplet is chosen per-mechanism and documented in the test fixture.

---

## Tolerance Policy

| Test category | Step size h | Tolerance epsilon | Rationale |
|--------------|------------|-------------------|-----------|
| FD Jacobian (existing) | h = 1e-7 | epsilon = 1e-5 | Standard for double-precision central differences |
| FD velocity from position sweep | h = sweep delta_t (e.g. 1 deg = 0.0175 rad) | epsilon = h^2 * C (calibrated per mechanism) | Central FD truncation is O(h^2); tolerance scales with step size |
| FD acceleration from velocity sweep | Same | Same scaling | Same reasoning |
| Constraint residual | - | epsilon = 1e-10 | Must match solver convergence criterion |
| Distance invariants | - | epsilon = 1e-10 | Exact geometric identity, limited only by solve precision |
| Textbook benchmark comparison | - | epsilon = 1e-8 | Analytical formulas computed in double precision |
| Golden snapshot comparison | - | epsilon = 1e-10 | Regression detection, not exact reproducibility |
| Property-based invariance | - | epsilon = 1e-8 | Round-trip through transforms loses ~8 digits |

**Calibration step:** Before finalizing tolerances, run each FD comparison on
the existing 4-bar benchmark at multiple h values (1e-5 through 1e-8), plot
error vs h, and pick the h that minimizes error. Document the chosen values
in a comment block at the top of the test file.

**Avoiding flakiness:** FD tests must exclude configurations within 5 degrees
of known toggle/singularity positions (where Jacobian condition number > 1e8).
Singular configurations get their own dedicated tests in Layer 4.

---

## Singularity and Failure Policy

| Scenario | Expected behavior | How to test |
|----------|------------------|-------------|
| Exact toggle (rank-deficient Phi_q) | Solver must not silently report a healthy regular solve. Acceptable outcomes: (a) `PositionSolveResult.converged == False`, (b) converge but with elevated iteration count. In either case, a follow-up `jacobian_rank_analysis()` call must report condition number > 1e8. Test asserts `converged == False` OR condition number > 1e8 via separate rank analysis. | Drive to known toggle angle, check PositionSolveResult + jacobian_rank_analysis |
| Near-singularity (condition > 1e8) | Position solver: converge but with more iterations. Separate `jacobian_rank_analysis()` call reports high condition number. No silent degradation. | Drive near toggle, check iteration count > normal, check condition via rank analysis |
| Infeasible assembly (no solution) | Position solver: return `PositionSolveResult` with `converged == False`. Residual norm remains large. | Provide impossible link lengths, assert `converged == False` |
| Redundant constraints | Rank analysis: rank < m. Gruebler may disagree with rank-based DOF (documented and expected). | Parallelogram + extra constraint, check rank deficit |
| Disconnected topology | Connectivity: report unreachable bodies | Build mechanism with isolated body, assert detection |
| Bad initial guess | If converges: `converged == True` and `residual_norm < 1e-10`. Branch detection via signed orientation invariant determines which branch was found. If branch differs from expected: logged as alternate-branch convergence (acceptable but distinct). If does not converge: `converged == False`, and returned `q` is last iterate (not garbage — can be inspected). | Provide far-from-solution guess, verify clean outcome |

**Note on solver API:** The position solver returns `PositionSolveResult(q, converged, iterations, residual_norm)` — it does not raise exceptions on failure. Condition numbers are obtained via a separate `jacobian_rank_analysis()` call. The velocity solver uses `np.linalg.lstsq` which silently returns a least-squares solution for singular systems; validate velocity results by checking constraint satisfaction `‖Phi_q * q_dot + Phi_t‖ < epsilon` rather than expecting exceptions.

---

## Layer 1: Internal Numerical Consistency

**File:** `tests/test_numerical_consistency.py`

Tests that the solver pipeline is self-consistent without external references.
Each test sweeps a benchmark mechanism and cross-checks solver stages against
each other.

| # | Test | Features validated | Mechanism | Details |
|---|------|--------------------|-----------|---------|
| 1a | FD velocity from position sweep (4-bar) | F5, F6, F4 | 4-bar crank-rocker | Sweep at 1 deg steps. At each step, compute (q(t+h) - q(t-h)) / (2h). Compare to solve_velocity(). Exclude +/-5 deg of toggle. |
| 1b | FD velocity from position sweep (slider-crank) | F5, F6, F2, F4 | Slider-crank | Same method. Validates prismatic Phi_t contribution to velocity. |
| 1c | FD velocity from position sweep (6-bar) | F5, F6, F1, F4 | Watt I 6-bar | Same method. Validates multi-loop velocity consistency. |
| 2a | FD acceleration from velocity sweep (4-bar) | F6, F7, F1, F4 | 4-bar crank-rocker | Sweep velocities at 1 deg steps. FD on q_dot vs solve_acceleration(). Cross-checks gamma terms. |
| 2b | FD acceleration from velocity sweep (slider-crank) | F6, F7, F2, F4 | Slider-crank | Same. Validates prismatic gamma. |
| 2c | FD acceleration from velocity sweep (6-bar) | F6, F7, F1, F4 | Watt I 6-bar | Same. Validates multi-loop gamma. |
| 3a | Rigid-body distance invariant (4-bar coupler) | F5, F17, F18 | 4-bar | Coupler body has 2 attachment points. Assert distance constant (+/-1e-10) across full sweep. |
| 3b | Rigid-body distance invariant (6-bar ternary) | F5, F17, F18 | Watt I 6-bar | Ternary body has 3 attachment points: 3 pairwise distances, all constant. |
| 4 | Constraint residual across sweep | F1, F2, F3, F5 | All three benchmarks | Assert residual norm < 1e-10 at every sweep step. |
| 5 | Jacobian rank drop at toggle | F11, F5 | Purpose-built non-Grashof 4-bar | Link lengths chosen so toggle occurs at known crank angle. Drive to toggle +/-0.01 deg. Assert rank drops by 1. Assert condition number > 1e8. |
| 6a | Coupler position to FD velocity (4-bar) | F8 | 4-bar with coupler point | Call `eval_coupler_point()` at adjacent angles, FD on position component vs velocity component from same function. |
| 6b | Coupler velocity to FD acceleration (4-bar) | F8 | 4-bar with coupler point | FD on velocity component of `eval_coupler_point()` vs acceleration component. |
| 6c | Coupler position to FD velocity (6-bar) | F8 | Watt I 6-bar with ternary coupler | Same FD approach on `eval_coupler_point()` for ternary body coupler point. Exercises non-trivial local coordinates. |

**Note on coupler API:** The actual function is `eval_coupler_point(state, body_id, point_local, q, q_dot, q_ddot)` which returns a tuple `(position, velocity, acceleration)` in a single call. Tests extract individual components from the tuple.

Estimated: ~22 test cases (some rows parametrized across angle ranges).

---

## Layer 2: External Analytical / Textbook Benchmarks

**File:** `tests/test_textbook_benchmarks.py`

Tests against published formulas and mathematical theorems. Priority order
reflects strength of ground truth.

| # | Test | Features validated | Ground truth source | Details |
|---|------|--------------------|-------------------|---------|
| 1 | Slider-crank closed-form position | F2, F5 | Norton "Design of Machinery" Ch. 4 | x_slider = r*cos(theta) + sqrt(l^2 - r^2*sin^2(theta)). This is the same formula already used in `test_slidercrank_benchmark.py:analytical_slider_position`. Compare at 15 deg increments through full rotation. Tolerance 1e-10. |
| 2 | Slider-crank closed-form velocity | F2, F6 | Norton Ch. 6 | Full analytical velocity expression. Compare at same angles. |
| 3 | Slider-crank closed-form acceleration | F2, F7 | Norton Ch. 7 | Full analytical acceleration expression. Compare at same angles. |
| 4a | 4-bar known positions | F1, F5 | Analytical circle-intersection (extend existing) | Currently tested at selected angles. Extend to 5 deg increments, 30-330 deg, with analytical reference at each. |
| 4b | 4-bar known velocities | F1, F6 | Analytical differentiation of circle-intersection | Derive closed-form velocity for the existing 4-bar dimensions. Compare at 15 deg increments. |
| 4c | 4-bar known accelerations | F1, F7 | Analytical second derivative | Same approach. |
| 5 | Roberts' cognate theorem | F1, F5, F8 | Mathematical identity: 3 cognate 4-bars share coupler curve | Build 3 cognate linkages from Roberts' construction for the existing 4-bar. Sweep all three. Assert coupler point paths agree within 1e-8. High-value but high-setup-complexity. Sequenced after simpler analytical checks (steps 1-4) which validate the core pipeline first. **Diagnostic strategy on failure:** (1) verify each cognate linkage independently satisfies constraints at every sweep step, (2) compare coupler points at a single known angle before full curve comparison, (3) use a worked example from a textbook (e.g., Norton Ch. 3 or Waldron & Kinzel) with published cognate dimensions to isolate construction errors from solver errors. |
| 6 | Grashof classification | F10, F9 | Grashof criterion: S+L <= P+Q | Build 4 mechanisms: crank-rocker, double-crank, rocker-rocker, change-point. Assert full-rotation sweep succeeds only for crank-rocker and double-crank. Assert rocker-rocker sweep hits toggle. |
| 7 | Transmission angle | F1, F5, F11 | mu = arccos((b^2 + c^2 - a^2 - d^2 + 2ad*cos(theta)) / (2bc)) | Compute at 15 deg increments for existing 4-bar. Compare analytical vs computed from body poses. |

Estimated: ~30-35 test cases (parametrized across angles and mechanism variants).

---

## Layer 3a: Deterministic Invariance Tests

**File:** `tests/test_invariance.py`

| # | Test | Features validated | Details |
|---|------|--------------------|---------|
| 1 | Mirror symmetry | F1, F2, F5, F6, F7 | Negate all Y-coordinates and local attachment Y-components of the 4-bar. Solve. Assert all output Y-values are negated. Catches sign errors in A(theta), B(theta). |
| 2 | Global frame rotation | F1, F2, F5, F17 | Rotate entire 4-bar by 37 deg (arbitrary non-special angle). Solve at same input. Transform results back. Assert match within 1e-8. Catches global-frame assumptions. |
| 3 | A(theta) orthogonality | F17 | At 20 angles: assert A(theta)^T * A(theta) = I, det(A(theta)) = 1. |
| 4 | B(theta) = dA/dtheta | F17 | At 20 angles: assert B(theta) approx (A(theta+h) - A(theta-h)) / 2h within 1e-8. |
| 5 | Equivalent mechanism: fixed joint vs 1 revolute + 1 revolute driver (angle locked) | F3, F1, F4 | A fixed joint removes 3 DOF (2 translational + 1 rotational). Model equivalently as: 1 revolute joint (removes 2 translational DOF) + 1 revolute driver that locks relative angle to a constant (removes 1 rotational DOF). Build same mechanism both ways. Solve both. Assert identical kinematics. |
| 6 | Force frame rotation invariance | F13, F14, F15 | Rotate BOTH the mechanism geometry (all body positions, attachment points) AND the gravity vector definition by the same angle. Assert Q vectors are related by the corresponding block-diagonal coordinate rotation: for each body's (Qx, Qy) pair, apply the 2x2 rotation; each body's Q_theta is unchanged. This tests that force helpers have no hidden global-frame assumptions, not that gravity is frame-invariant (it is not, by definition). |
| 7 | Virtual work consistency (point force) | F13 | Apply F at point P on body. Perturb q by delta_q. Compute F dot delta_r_P (actual work) and Q dot delta_q (generalized work). Assert equal within 1e-8. |
| 8 | Virtual work consistency (gravity) | F15 | Perturb q by delta_q. Compute -m*g*delta_h_cg for each body vs Q_gravity dot delta_q. Assert equal. |
| 9 | Virtual work consistency (torque) | F14 | Apply torque tau on body. Perturb theta by delta_theta. Assert tau*delta_theta = Q dot delta_q. |
| 10 | Force superposition | F16 | Apply F1 alone -> Q1. Apply F2 alone -> Q2. Apply both -> Q12. Assert Q12 = Q1 + Q2. |
| 11 | Zero-force baseline | F16 | No force elements. Assert assemble_Q returns zero vector. |

Estimated: ~15-18 test cases.

---

## Layer 3b: Property-Based Stress Tests

**File:** `tests/test_property_stress.py`

| # | Test | Features validated | Details |
|---|------|--------------------|---------|
| 1 | Random valid 4-bar: constraint satisfaction | F1, F5 | Generate random link lengths satisfying Grashof crank-rocker condition (S+L < P+Q). Assemble. Solve at random valid crank angle. Assert residual < 1e-10. Seeded, 50 examples, @settings(max_examples=50, derandomize=True). |
| 2 | Random valid 4-bar: velocity self-consistency | F1, F6 | Same generation. Solve velocity. Assert Phi_q * q_dot + Phi_t norm < 1e-10. |
| 3 | Random valid 4-bar: distance invariant | F1, F5, F17 | Same generation. Solve at 2 different crank angles. Assert coupler length constant. |
| 4 | Random prismatic slider-crank: constraint satisfaction | F2, F5 | Generate random valid slider-crank dimensions. Solve. Assert residual. |

**Generation constraints:**
- Link length bounds: 0.5 <= L <= 10.0
- Transmission angle: mu >= 15 deg (filter out near-singular geometries)
- Joint separation: min distance between any two distinct joint locations >= 0.1
- Singularity margin: crank angle >= 10 deg from computed toggle positions

**Guardrails:**
- `derandomize=True` on all Hypothesis tests (deterministic in CI)
- `database=None` to prevent cached example drift
- Generation restricted to known-valid mechanism families (Grashof crank-rocker, standard slider-crank)
- On failure: explicitly log link lengths, joint coordinates, driver angle, computed transmission angle, Jacobian condition number
- Use `@example(...)` decorator to pin any discovered failure case as a permanent regression test

**Quarantine rule:** If any test flakes twice in CI, mark it
`@pytest.mark.xfail(strict=False)` with a tracking comment and move
investigation to a follow-up. Do not leave intermittent tests in the
blocking path.

Estimated: ~8-10 test cases.

---

## Layer 4: Failure-Mode and Diagnostics Validation

**File:** `tests/test_failure_modes.py`

Verifies correct behavior under degenerate, singular, and impossible inputs.

| # | Test | Features | Scenario | Expected behavior |
|---|------|----------|----------|-------------------|
| 1 | Infeasible assembly | F5 | 4-bar with S + L > P + Q + clearance at requested angle | `PositionSolveResult.converged == False`. Residual norm remains large. |
| 2 | Bad initial guess | F5 | Valid 4-bar, initial guess 180 deg from solution | If `converged == True`: residual < 1e-10 and branch identified via branch invariant. If `converged == False`: returned `q` is last iterate (inspectable, not garbage). |
| 3 | Exact toggle position | F5, F11 | Non-Grashof 4-bar at computed toggle angle (specify: links 2, 3, 4, 5 with toggle at theta = arccos((a^2 + d^2 - (b-c)^2) / (2ad))). | Solver must not silently report healthy solve. Assert `converged == False` OR `jacobian_rank_analysis()` reports condition > 1e8. |
| 4 | Near-toggle degradation | F5, F11 | Same mechanism, 0.1 deg from toggle | Solver converges. Iteration count > baseline. Condition number elevated. |
| 5a | Redundant constraints: extra revolute | F11, F10 | Closed 4-bar + one redundant revolute at existing pin | Rank deficit = 1 reported. |
| 5b | Redundant constraints: parallelogram | F11, F10 | Parallelogram with extra diagonal closure | Rank deficit = 1 reported. |
| 5c | Redundant constraints: duplicate fixed | F11, F10 | Body with two identical fixed joints to ground at the same attachment point, same reference angle. The second joint duplicates all 3 constraint equations exactly. | Rank deficit = 3 reported. |
| 6 | Disconnected body | F12 | 4-bar + one floating body with no joints | Connectivity reports unreachable body. |
| 7 | Multi-component disconnect | F12 | Two separate 4-bars sharing no bodies | Connectivity reports two components. |
| 8 | Zero-length link | F1, F5 | Revolute where both attachment points are at body origin | Solver handles gracefully (degenerate but not invalid). |
| 9 | Singular Phi_q in velocity solve | F6, F11 | At exact toggle, attempt velocity solve | `np.linalg.lstsq` will return a least-squares solution silently. Validate by checking `‖Phi_q * q_dot + Phi_t‖` is NOT small (constraint satisfaction fails). This proves the velocity result is meaningless at singularity. |
| 10 | Branch continuity through sweep | F9 | Sweep Grashof crank-rocker through full 360 deg. At each step compute branch invariant s = sign((B-A) x (C-A)). | Invariant sign must not flip at any step away from singularity. If step-size reduction occurs, path must stay on same branch. A flip away from toggle = unintended branch jump = test failure. |

**Topology cross-consistency suite:**

| # | Mechanism | Gruebler DOF | Rank-based mobility | Connectivity | Cross-check |
|---|-----------|-------------|-------------------|--------------|-------------|
| 11a | Well-posed 4-bar + driver | 0 | 0 (full rank) | All connected | DOF, rank, connectivity all agree |
| 11b | 4-bar without driver | 1 | 1 (full rank) | All connected | Gruebler DOF = rank-based mobility = 1 |
| 11c | 4-bar (no driver) + 1 floating body | 4 (M = 3(4 bodies) - 4(2 DOF removed) = 4) | 4 (floating body adds 3 uncontrolled coords) | Floating body unreachable | Gruebler and rank agree at 4. Connectivity flags disconnect. High mobility comes from disconnected component, not underconstrained connected mechanism. |
| 11d | Parallelogram + redundant constraint | Gruebler says 1 | Rank says 2 (rank deficit = 1) | All connected | Gruebler/rank disagreement is documented and expected |
| 11e | Rigid triangle (3 bars, 3 revolutes, all grounded) | 0 | 0 | All connected | Fully constrained, no motion |

Estimated: ~20 test cases.

---

## Layer 5: Performance and Regression Baselines

**File:** `tests/test_regression_baselines.py`

Not performance optimization — early warning for regressions that manifest
as degraded convergence.

Regression baselines detect accidental numerical or convergence drift; they
must not block intentional algorithmic improvements that change outputs.
Update baselines in the same PR as the improvement.

| # | Test | What it tracks | Mechanism | Threshold |
|---|------|---------------|-----------|-----------|
| 1 | Newton-Raphson iteration count | Mean iterations across full sweep | 4-bar, 30-330 deg at 5 deg steps | Assert mean <= 4, max <= 8 |
| 2 | Newton-Raphson iteration count (slider-crank) | Same | Slider-crank, full rotation | Same thresholds |
| 3 | Sweep step success rate | Pct of steps converging without step-size reduction | 4-bar full rotation | Assert >= 98% |
| 4 | Sweep step success rate (slider-crank) | Same | Slider-crank full rotation | Assert >= 98% |
| 5 | Golden snapshot: 4-bar | Body poses, velocities, accelerations at theta = 45, 90, 135, 180 deg | 4-bar crank-rocker | Assert match to 1e-10 |
| 6 | Golden snapshot: slider-crank | Same at 4 canonical angles | Slider-crank | Same |
| 7 | Golden snapshot: 6-bar | Same at 4 canonical angles | Watt I 6-bar | Same |
| 8 | Condition number baseline | Condition number at 15 deg increments | 4-bar | Assert within 2x of stored baseline |

**Golden snapshot format:** JSON files in `tests/golden/` containing
`{angle, q, q_dot, q_ddot, coupler_pos, coupler_vel, coupler_accel}` at
selected angles. Generated once, committed, compared on every run.

**Important:** Golden snapshots are regression baselines only and do not
establish correctness. They may be updated when stronger validation (L1/L2)
reveals prior behavior was wrong. They detect drift, not prove truth.

**All thresholds are provisional**, calibrated from the current baseline on
the development machine. On first CI run, recalibrate and commit updated
thresholds. Label in code:
`# BASELINE: calibrated YYYY-MM-DD, recalibrate if platform changes`.

If exact bit-for-bit reproducibility is desired in the future, introduce a
`@pytest.mark.exact_repro` marker gated on a platform check (same OS +
NumPy + BLAS). Default CI uses 1e-10.

Estimated: ~10-12 test cases.

---

## Implementation Order

Ordered by dependency chain and priority. Each step is a committable unit.

| Step | Layer | What | Depends on | Est. tests |
|------|-------|------|-----------|-----------|
| 1 | L5 | Golden snapshot generation: run existing benchmarks at canonical angles, capture outputs, commit JSON files to tests/golden/ | Nothing (uses existing code as-is) | 0 (data only) |
| 2 | L5 | Golden snapshot comparison tests + iteration/sweep baselines: write test_regression_baselines.py | Step 1 golden data | 11 |
| 3 | L1 | Constraint residual + distance invariants: tests 3a, 3b, 4 from Layer 1 (no FD complexity yet) | Nothing | 5 |
| 4 | L1 | Tolerance calibration: run FD comparison on 4-bar at multiple h values, determine optimal h, document | Step 3 (needs working sweep) | 0 (analysis, results as comments) |
| 5 | L1 | FD velocity chain: tests 1a, 1b, 1c using calibrated h | Step 4 | 6 |
| 6 | L1 | FD acceleration chain: tests 2a, 2b, 2c | Step 5 | 6 |
| 7 | L1 | Coupler FD chain + rank drop at toggle: tests 5, 6a, 6b | Step 5 | 3 |
| 8 | L2 | Slider-crank closed-form position/velocity/acceleration: tests 1, 2, 3 | Nothing | 12 |
| 9 | L2 | 4-bar extended analytical positions/velocities/accelerations: tests 4a, 4b, 4c | Nothing | 12 |
| 10 | L2 | Roberts' cognate theorem: test 5 | Steps 8-9 (validates pipeline first) | 3 |
| 11 | L2 | Grashof classification + transmission angle: tests 6, 7 | Nothing | 8 |
| 12 | L3a | Deterministic invariance: mirror, frame rotation, A/B matrix, equivalent mechanism: tests 1-5 | Nothing | 7 |
| 13 | L3a | Force validation: frame invariance, virtual work, superposition, zero baseline: tests 6-11 | Requires force helpers (already exist) | 8 |
| 14 | L3b | Property-based stress tests: all Hypothesis tests | Steps 5-6 (need confidence in FD approach) | 9 |
| 15 | L4 | Failure modes + branch continuity + topology cross-consistency: all scenarios | Steps 3, 7 (need rank analysis established) | 20 |

---

## CI Considerations

| Concern | Mitigation |
|---------|-----------|
| Hypothesis flakiness | derandomize=True, database=None, fixed seeds |
| Golden snapshot drift | 1e-10 tolerance; any drift means code changed, investigate |
| Slow FD sweeps | Parametrize at 15 deg increments (24 points) for normal CI. Full 1 deg sweep as optional @pytest.mark.slow |
| Test isolation | Each test file builds its own mechanisms from scratch, no shared mutable fixtures |
| Fast vs slow split | Deterministic tests in normal CI; property-based/slow-sweep tests in secondary lane or marked @pytest.mark.slow |

---

## Out of Scope

- Phase 2 features (static solver, spring elements, joint reactions) — not yet implemented
- 3D extension — not in scope
- Dynamic simulation — not implemented
- Serialization round-trip validation — already well-tested (19 tests)
- Visualization correctness — subjective, already has basic tests

---

## Test Count Summary

| Layer | File | Est. tests | Priority |
|-------|------|-----------|----------|
| L1 Internal consistency | test_numerical_consistency.py | 20 | P1 |
| L2 Textbook benchmarks | test_textbook_benchmarks.py | 32 | P1 |
| L3a Deterministic invariance | test_invariance.py | 16 | P2 |
| L3b Property-based stress | test_property_stress.py | 9 | P2 |
| L4 Failure modes + branch + topology | test_failure_modes.py | 20 | P2 |
| L5 Regression baselines | test_regression_baselines.py | 11 | P3 |
| **Total new** | **6 files** | **~108** | |

Combined with existing 395 tests: ~503 total tests.
