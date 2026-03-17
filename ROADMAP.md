# Roadmap

Development phases, deliverables, exit criteria, and the recommended build order for getting to a useful engineering tool as fast as possible.

---

## Scope Control

The biggest risk for this project is not bad architecture — it is scope blow-up. The architecture is now general enough to handle arbitrary planar mechanisms, which means it is tempting to build infrastructure for months before getting a useful answer for a real mechanism.

**Guard against this.** Every phase has a narrow implementation target. The recommended build order within each phase prioritizes "produces correct numbers for the simplest useful case" over "handles every edge case."

---

## Recommended Build Order (Within Phase 1)

This is the order to write code. It gets you to a working 4-bar kinematic solver as fast as possible, then generalizes.

1. **State vector and coordinate bookkeeping** — define `q`, body-to-index mapping
2. **Body data structure** — attachment points, mass properties, ground body
3. **Revolute joint constraint** — `Phi(q)`, `Phi_q(q)`, verified against finite-difference Jacobian
4. **Fixed joint constraint** — for ground connections
5. **Mechanism assembly** — collect bodies and joints, build global `Phi` and `Phi_q`
6. **Grübler DOF count** — sanity check
7. **Jacobian rank check** — actual instantaneous mobility
8. **Kinematic position solver** — Newton-Raphson on `Phi(q,t) = 0`
9. **Revolute driver constraint** — prescribed input angle
10. **Position sweep** — solve across a range of input angles
11. **Velocity solver** — linear solve for `q_dot`
12. **Acceleration solver** — linear solve for `q_ddot` (requires `gamma` implementation)
13. **Coupler point evaluation** — position, velocity, acceleration of arbitrary body points
14. **JSON serialization / deserialization**
15. **Minimal Matplotlib viewer** — render bodies and joint locations from solved results
16. **Animation** — sweep input and render frame-by-frame
17. **Test suite** — 4-bar benchmark (position, velocity, acceleration vs. textbook)
18. **Prismatic joint constraint** — full slide-axis implementation
19. **Slider-crank benchmark** — validates prismatic joint
20. **Graph connectivity check**
21. **Ternary body test** — 6-bar Watt or Stephenson to validate multi-attachment-point bodies

After step 17, you have a working 4-bar kinematic solver. Everything after that is generalization. This ordering ensures you get useful results early and can test incrementally.

---

## Phase 1 — Core Data Model, Constraints & Kinematics

The mathematical foundation. No GUI editor. Mechanisms defined via JSON or Python API. Minimal viewer for verification.

**Deliverables:**

- Body data structure: multiple named attachment points, mass properties (`mass`, `cg_local`, `Izz_cg`), coupler points, render shape
- Ground body: fixed at origin, mass = 0, excluded from `q`
- Generalized coordinate vector `q`: 3 entries per moving body `(x, y, θ)`, with explicit index mapping
- JointConstraint types:
  - Revolute: 2 constraint equations, analytical Jacobian, `gamma` contribution
  - Prismatic: 2 constraint equations (perpendicular displacement + no rotation), slide axis definition, displacement coordinate extraction. Fully defined: axis direction in body frame, reference points, constraint on relative rotation
  - Fixed: 3 constraint equations (2 position + 1 rotation)
- Each constraint type passes finite-difference Jacobian verification test
- Driver model: adds one constraint row to `Phi` prescribing joint coordinate as `f(t)`. Driver types: constant speed, expression (safe evaluator), interpolated profile
- Mechanism assembly: build global `Phi`, `Phi_q`, and `gamma` from all joints and drivers
- Schema-versioned JSON serialization and deserialization
- Grübler DOF count (informational)
- Jacobian rank analysis at current configuration → `constraint_rank`, `instantaneous_mobility`
- Redundant constraint detection with warning
- Graph connectivity check (all bodies reachable from ground)
- Kinematic position solver: Newton-Raphson with analytical Jacobian, continuation from previous solution
- Kinematic velocity solver: linear solve `Phi_q * q_dot = -Phi_t`
- Kinematic acceleration solver: linear solve `Phi_q * q_ddot = gamma`
- Coupler point position, velocity, acceleration via body kinematics
- Assembly mode: user-provided initial guess selects configuration. No automatic branch detection in v1
- Minimal Matplotlib visualization: render bodies, joints, coupler traces
- Animation: sweep input joint, render mechanism frame-by-frame
- Test suite against benchmarks: 4-bar (Grashof crank-rocker), slider-crank, non-Grashof rocker-rocker, 6-bar with ternary link, parallelogram (redundant constraint detection)

**Exit criteria:** A user can define a 4-bar, slider-crank, or 6-bar with ternary link via JSON, solve kinematics, trace coupler points, and verify results against textbook values. Prismatic joints produce correct piston motion. Jacobian rank matches Grübler for well-posed mechanisms and correctly detects redundant constraints in the parallelogram case. All benchmark tests pass.

---

## Phase 2 — Force Elements & Static Analysis

Add the force element infrastructure and the static equilibrium solver.

**Deliverables:**

- ForceElement base class with `evaluate(state, t) → Q_contribution` interface
- Helper utilities: `point_force_to_Q`, `body_torque_to_Q`, `gravity_to_Q`, `joint_torque_to_Q`
- Built-in force element types:
  - Linear spring (stiffness, free length, preload, tension-only / compression-only / both)
  - Torsion spring at revolute joints
  - Gravity (magnitude + arbitrary direction, applied at composite CG)
  - External load (safe expression of position, velocity, time)
  - Coulomb friction at joints (regularized with `tanh` velocity smoothing, static/kinetic split)
- PointMass element: mass at arbitrary body-local position, automatic composite mass/CG/Izz recomputation via parallel axis theorem
- Generalized force vector `Q` assembly: iterate all force elements, sum contributions
- Static force solver: solve `Phi_q^T * lambda = -Q` for Lagrange multipliers
- Driver reaction extraction: `lambda_driver` = required input torque (revolute) or force (prismatic)
- Passive joint reaction extraction in all output formats: global Fx/Fy, resultant, body-local, radial/tangential
- Redundant constraint handling: minimum-norm pseudoinverse solution with warning
- Virtual work method as cross-check for input torque
- Mechanical advantage computation (user specifies input and output coordinates/directions)
- Transmission angle for user-selected body pairs
- Pressure angle at user-selected joints (with force direction specification)
- Toggle position and dead point detection via Jacobian smallest singular value monitoring
- Grashof condition check for 4-bar configurations
- Result envelopes: peak, RMS, min/max over full sweep
- Plotting: input torque vs. angle, joint reactions vs. angle, transmission angle vs. angle, spring/damper state vs. angle
- Benchmark tests: 4-bar with gravity, 4-bar with spring, point mass test, slider-crank with friction

**Exit criteria:** User attaches springs, gravity, and friction to a 4-bar, and the tool shows required input torque across the full cycle with correct values (matching hand calculation). Transmission angle warnings flag poor-geometry regions. Reaction forces are available in all output formats. Envelopes give peak and RMS loads for bearing sizing.

---

## Phase 3 — Actuators & Inverse Dynamics

Realistic actuator models and inertia-aware force analysis.

**Deliverables:**

- Additional ForceElement types:
  - Motor with linear T-ω droop
  - Linear actuator (force-stroke table or constant, with speed limit)
  - Viscous damper (translational, between two body points)
  - Rotary damper (at revolute joints)
  - Gas spring (pressure-based force + velocity-dependent damping)
- Mass matrix `M` assembly: block-diagonal from composite body properties
- Inverse dynamics solver: solve `Phi_q^T * lambda = Q - M * q_ddot` for reactions and driver effort
- Motor sizing assistant: plot required `(omega, T)` operating points over motor T-ω envelope, flag infeasible points
- Individual force element contribution breakdown vs. time (inertia vs. gravity vs. springs vs. friction — which dominates?)
- Bearing friction model beyond generic Coulomb: constant drag torque, viscous drag, optional radial-load-dependent loss
- Result envelopes for inverse dynamics: peak torque, RMS torque, peak speed, mean power
- Plotting: required torque vs. time, motor T-ω trajectory, force element contributions
- Benchmark tests: 4-bar with inertia (compare to simple-pendulum limits), slider-crank with motor, damped system

**Exit criteria:** User defines a motion profile, assigns body masses and inertias, and the tool computes required motor torque including inertial loads. Motor feasibility check correctly identifies when a motor can't follow the demanded profile. Damper energy dissipation rate is correct.

---

## Phase 4A — Forward Dynamics (Smooth)

Given applied forces/torques, simulate what the mechanism actually does over time. **This phase handles only smooth force elements — no contact, no hard stops, no stick-slip friction.**

Forward dynamics is fundamentally harder than Modes 1–3 and is split into two sub-phases to prevent scope creep. Phase 4A produces a working, validated integrator for smooth systems. Phase 4B adds discontinuous effects that change the mathematical class of the problem.

**Deliverables:**

- DAE formulation: `M * q_ddot + Phi_q^T * lambda = Q` coupled with `Phi(q,t) = 0`
- Integration approach: index reduction + Baumgarte stabilization. Keep implementation flexible — do not lock to one library or method before testing
- Constraint drift monitoring: check `||Phi(q)||` after each step, apply correction if above tolerance
- Constraint projection: Newton-Raphson correction to satisfy `Phi(q) = 0` periodically
- Viscous damping warning if no damping elements present
- Energy balance tracking: KE + PE + dissipated vs. work input at every timestep
- Transient response: step inputs, impulse response, free response from displaced position
- Natural frequency estimation: linearize about equilibrium, eigenvalue extraction
- Plotting: position/velocity/acceleration vs. time, phase portraits, energy components vs. time
- Benchmark tests: simple pendulum (known period), damped pendulum (exponential decay), 4-bar free response (energy balance closure), 4-bar step torque (steady-state reached)

**Force elements allowed in 4A:** gravity, linear springs, torsion springs, viscous dampers (translational and rotary), motor droop, gas springs, external loads (smooth expressions only). **Not allowed:** Coulomb friction, joint limits/hard stops, cables, clutches.

**Exit criteria:** Step torque on a 4-bar with springs and viscous dampers produces realistic transient oscillation decaying to steady state. Constraint drift stays below tolerance for the duration. Energy balance closes within integrator tolerance. Simple pendulum period matches analytical.

---

## Phase 4B — Forward Dynamics (Nonsmooth Effects)

Add discontinuous and switching effects to the forward dynamics integrator.

**Deliverables:**

- Contact / hard-stop handling at joint limits: penalty method (default), event detection with restitution (optional)
- Coulomb friction in forward dynamics: regularized model from Phase 2 as baseline, with monitoring for when regularization is insufficient (e.g., near-zero velocity oscillation, energy gain from regularization artifacts)
- Event detection framework: zero-crossing detection for joint limits, direction reversals, mode switches
- Penalty spring stiffness guidelines: relationship between penalty stiffness, integration step size, and integrator stability
- Benchmark tests: pendulum with hard stop (restitution coefficient verification), slider-crank with Coulomb friction (energy dissipation tracking), mechanism with joint limits (correct stop behavior)

**Exit criteria:** Mechanism with joint limits produces correct stop-and-rebound behavior. Coulomb friction dissipates the correct amount of energy over a cycle. Event detection catches zero-crossings within tolerance. Energy balance still closes.

**On exit (4A + 4B combined):** Export golden test fixtures (`--export-golden`) for all benchmark mechanisms. This is the entry gate for the Rust port. → See `docs/RUST_MIGRATION.md`

---

## Rust Port — Between Phase 4 and Phase 5

**After Phase 4 exits, port the solver kernel to Rust before building the GUI.** Phase 5 is built natively in Rust using `egui`, not in Python. The Python codebase becomes the reference implementation and test oracle.

See `docs/RUST_MIGRATION.md` for the full migration plan: port sequencing, golden test fixture strategy, Rust module mapping, crate dependencies, and coding conventions to follow during Python development for clean portability.

**Entry criteria:** All Phase 1–4 exit criteria met. Benchmark test suite passes. Golden test fixtures exported to `data/benchmarks/golden/`.

**Exit criteria:** Rust solver reproduces all golden test fixture results within tolerance. JSON mechanism files load and round-trip correctly via `serde`. All Rust tests pass.

---

## Phase 5 — Interactive GUI Editor

With the math stable, all four analysis modes validated, and the solver ported to Rust, build the interactive editor natively in Rust.

**Deliverables:**

- 2D canvas: place bodies, define attachment points, drag-and-drop
- Body editor: add/remove/move attachment points, set mass properties, define render shape
- Joint creation: click attachment point on body A → click on body B, select type
- Prismatic joint: visual axis setting
- Force element attachment: click two points or a joint, select type, set parameters via auto-generated property panel
- Point mass placement: click location on body, set mass and label
- Driver editor: select joint, define motion profile
- Validation panel: Grübler, Jacobian rank, connectivity, warnings — all live
- Real-time animation with playback controls (speed, pause, step)
- Integrated plotting panels with output selection
- Unit conversion at GUI boundary: display mm, N·mm, degrees; store SI
- Snap-to-grid, dimensioned display
- Undo/redo
- JSON save/load via file dialog
- Load case panel: define and switch between scenarios

**Exit criteria:** User can build, edit, simulate, and analyze any planar mechanism entirely through the GUI. Results match the JSON/API workflow from Phases 1–4 exactly.

---

## Phase 6 — Advanced & Quality-of-Life

Features for daily engineering use.

**Prerequisites — do not start Phase 6 until all of these are met:**

- Branch-stable sweeps: predictor-corrector continuation produces continuous results across full input range without silent branch jumps
- Singularity metrics: `σ_min`, `κ(Φ_q)`, and MA are reported and tested at known near-singular configurations
- Reliable failed-step handling: solver gracefully reduces step size, reports failure location, and does not produce partial garbage results
- Repeatable benchmarks: all benchmark mechanisms produce identical results (within tolerance) across runs
- Deterministic solver behavior: fixed tolerances, fixed step-size limits, no random initial guesses

Without these, optimization will exploit solver glitches (branch jumps, near-singular configurations, failed convergence) and return garbage that looks plausible.

**Deliverables:**

- **Synthesis tools**: find linkage dimensions for desired coupler curve or output motion (3-point, 4-point, function generation)
- **Optimization**: objective function on outputs (minimize peak torque, maximize min transmission angle, match target path) with parameter bounds. Optimizer must handle: solver failure at some parameter combinations (return penalty, not crash), branch jumps (detect and penalize or re-seed), and near-singular configurations (penalize via `σ_min` or MA thresholds)
- **Multi-DOF mechanisms**: 2+ input systems (e.g., 5-bar with two drivers)
- **Cam-follower**: cam profile as a new JointConstraint type
- **Import/export**: CAD export (DXF), animation export (GIF/MP4)
- **Parametric studies**: sweep a design variable, plot output sensitivity
- **Report generation**: auto-generated summary with diagram, dimensions, Grashof check, peak forces, transmission angle range
- **Mechanism library**: Watt, Stephenson, Roberts, quick-return, toggle clamp, slider-crank, scotch yoke as starting templates
- **Counterbalance assistant**: compute optimal spring/point-mass parameters to minimize torque variation

---

## What Is Explicitly Deferred

These are acknowledged as valuable but not in any current phase:

- Automatic assembly mode detection and branch switching (v1 uses continuation + user initial guess)
- Full stick-slip friction (v1 uses velocity regularization)
- Nonsmooth dynamics methods (complementarity solvers, Moreau time-stepping)
- Elastic / compliant bodies
- 3D visualization
- Hardware interface (encoder overlay)
- Collision detection between bodies
- Sensitivity analysis and tolerance propagation
- Monte Carlo / batch study framework (available via Python API, no dedicated GUI)
