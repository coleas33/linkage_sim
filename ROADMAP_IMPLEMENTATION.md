# Roadmap Implementation Tracker

Progress through the Phase 1 build order. Each step links to the commit where it was implemented.

See `ROADMAP.md` for the full phase plan and exit criteria.

---

## Phase 1 — Core Data Model, Constraints & Kinematics

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | State vector and coordinate bookkeeping | Done | `core/state.py` | `test_state.py` (28 tests) |
| 2 | Body data structure | Done | `core/bodies.py` | `test_bodies.py` (15 tests) |
| 3 | Revolute joint constraint | Done | `core/constraints.py` | `test_constraints.py` (19 tests, FD Jacobian verified via hypothesis) |
| 4 | Fixed joint constraint | Done | `core/constraints.py` | `test_constraints.py` (13 new tests, FD Jacobian + gamma verified) |
| 5 | Mechanism assembly | Done | `core/mechanism.py`, `solvers/assembly.py` | `test_mechanism.py` (15 tests, global FD Jacobian verified) |
| 6 | Grubler DOF count | Done | `analysis/validation.py` | `test_validation.py` (11 tests) |
| 7 | Jacobian rank check | Done | `analysis/validation.py` | `test_validation.py` (11 new tests, SVD-based rank + condition number) |
| 8 | Kinematic position solver (Newton-Raphson) | Done | `solvers/kinematics.py` | `test_kinematics.py` (12 tests, multi-angle convergence verified) |
| 9 | Revolute driver constraint | Done | `core/drivers.py` | `test_drivers.py` (19 tests, FD Jacobian via hypothesis, driven 4-bar solve) |
| 10 | Position sweep | Done | `solvers/sweep.py` | `test_sweep.py` (9 tests, full-rotation Grashof 4-bar verified) |
| 11 | Velocity solver | Done | `solvers/kinematics.py`, `solvers/assembly.py` | `test_velocity_accel.py` (7 tests, FD velocity verified) |
| 12 | Acceleration solver | Done | `solvers/kinematics.py` | `test_velocity_accel.py` (6 tests, FD acceleration verified) |
| 13 | Coupler point evaluation | Done | `analysis/coupler.py` | `test_coupler.py` (11 tests, FD velocity verified, 4-bar coupler curve) |
| 14 | JSON serialization / deserialization | Done | `io/serialization.py` | `test_serialization.py` (19 tests, full round-trip + file I/O) |
| 15 | Minimal Matplotlib viewer | Done | `viz/viewer.py` | `test_viewer.py` (9 tests, bodies + joints + coupler trace) |
| 16 | Animation | Done | `viz/animation.py` | `test_animation.py` (6 tests, FuncAnimation + coupler trace) |
| 17 | Test suite — 4-bar benchmark | Done | `test_fourbar_benchmark.py` | 36 tests: DOF, position vs analytical, velocity/accel FD, sweep, coupler curve |
| 18 | Prismatic joint constraint | Done | `core/constraints.py` | `test_prismatic.py` (35 tests, FD Jacobian + gamma verified via hypothesis) |
| 19 | Slider-crank benchmark | Done | `test_slidercrank_benchmark.py` | 49 tests: DOF, position/velocity/accel vs analytical, sweep, stroke, rail constraint |
| 20 | Graph connectivity check | Done | `analysis/validation.py` | `test_validation.py` (12 new tests, BFS from ground, disconnected detection, component count) |
| 21 | Ternary body test (6-bar) | Done | `test_sixbar_ternary.py` | 27 tests: Watt I 6-bar, ternary link, DOF, position/vel/accel FD, sweep, internal distances |

**Total tests:** 594 passing | **mypy:** strict, clean

---

## Phase 3 Complete

All 15 steps of Phase 3 are implemented. The simulator now supports inverse dynamics with
inertial loads (M*q̈), force element types (motor with T-ω droop, linear actuator, gas
spring, viscous/rotary dampers, bearing friction), mass matrix assembly, motor sizing
feasibility checks, and force element contribution breakdowns. All benchmarks validate the
pipeline: kinematics → inverse dynamics → reactions → motor sizing → envelopes.

---

## Phase 1 Complete

All 21 steps of Phase 1 are implemented. The simulator handles 4-bar, slider-crank,
and 6-bar (with ternary links) mechanisms with full kinematic analysis: position,
velocity, acceleration, coupler point tracking, DOF validation, Jacobian rank analysis,
graph connectivity, JSON serialization, and Matplotlib visualization/animation.

---

## Phase 2 — Force Elements & Static Analysis

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | ForceElement protocol + generalized force helpers + Q assembly | Done | `forces/protocol.py`, `forces/helpers.py`, `forces/assembly.py` | `test_force_helpers.py` (23 tests: point_force_to_Q, body_torque_to_Q, gravity_to_Q, assemble_Q, virtual work consistency) |
| 2 | Gravity force element | Done | `forces/gravity.py` | `test_gravity_element.py` (17 tests: protocol, force values, multi-body, virtual work) |
| 3 | Linear spring force element | Done | `forces/spring.py` | `test_spring_element.py` (17 tests: basic, modes, two-body, edge cases, virtual work) |
| 4 | Torsion spring at revolute joints | Done | `forces/torsion_spring.py` | `test_torsion_spring_element.py` (11 tests: basic, two-body, virtual work, protocol) |
| 5 | Static force solver (Φ_qᵀ λ = −Q) | Done | `solvers/statics.py` | `test_statics.py` (12 tests: basic solve, driver reaction, spring interaction, sweep, virtual work) |
| 6 | Driver and joint reaction extraction | Done | `analysis/reactions.py` | `test_reactions.py` (11 tests: extraction, filtering, local transform, equilibrium) |
| 7 | Grashof condition check | Done | `analysis/grashof.py` | `test_grashof.py` (9 tests: all 5 classifications, values, edge cases) |
| 8 | Transmission angle computation | Done | `analysis/transmission.py` | `test_transmission_angle.py` (9 tests: formula, sweep, symmetry, law of cosines cross-check) |
| 9 | External load force element | Done | `forces/external_load.py` | `test_external_load.py` (9 tests: constant, time/position-dependent, torque, virtual work) |
| 10 | Coulomb friction (regularized) | Done | `forces/friction.py` | `test_friction.py` (10 tests: tanh model, direction, magnitude, smooth regularization, two-body) |
| 11 | PointMass element + composite mass recomputation | Done | `core/point_mass.py` | `test_point_mass.py` (8 tests: CG shift, parallel axis theorem, multiple masses, symmetric) |
| 12 | Virtual work cross-check | Done | `analysis/virtual_work.py` | `test_virtual_work.py` (5 tests: matches statics λ to 1e-6 with gravity, springs, combined forces) |
| 13 | Mechanical advantage computation | Done | `analysis/mechanical_advantage.py` | `test_mechanical_advantage.py` (6 tests: finite MA, varies with angle, translational output) |
| 14 | Pressure angle | — | Deferred: covered by transmission angle (Step 8) for 4-bar; general pose-based version deferred | — |
| 15 | Toggle/dead point detection | Done | `analysis/toggle.py` | `test_toggle.py` (5 tests: σ_min monitoring, condition number, threshold) |
| 16 | Result envelopes (peak, RMS, min/max) | Done | `analysis/envelopes.py` | `test_envelopes.py` (6 tests: sinusoid, constant, peak angle, edge cases) |
| 17 | Force-related plotting | Done | `viz/force_plots.py` | `test_force_plots.py` (3 tests: torque, reactions, transmission angle plots) |
| 18 | Benchmark: 4-bar with gravity | Done | `test_benchmark_fourbar_gravity.py` | 8 tests: sweep convergence, driver torque, virtual work cross-check, reactions, envelopes, transmission angle, toggle detection |
| 19 | Benchmark: 4-bar with spring | Done | `test_benchmark_fourbar_spring.py` | 4 tests: torsion spring counterbalance, linear spring, virtual work, torque envelope |
| 20 | Benchmark: slider-crank with friction | Done | `test_benchmark_slidercrank_friction.py` | 5 tests: gravity sweep, friction effect, torque envelope, virtual work cross-check |

---

## Phase 2 Complete

All 20 steps of Phase 2 are implemented (Step 14 deferred — pressure angle covered by
transmission angle for 4-bar). The simulator now supports force elements (gravity, springs,
friction, external loads), static equilibrium solving with Lagrange multiplier reaction
extraction, virtual work cross-checks, mechanical advantage, Grashof classification,
transmission angle analysis, toggle detection, result envelopes, and force-related plotting.
All benchmarks validate the complete pipeline: position → statics → reactions → virtual work.

---

## Phase 3 — Actuators & Inverse Dynamics

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | Viscous damper (translational, between two body points) | Done | `forces/viscous_damper.py`, `core/state.py` (body_point_velocity) | `test_dampers.py` (16 tests: translational + rotary + body_point_velocity) |
| 2 | Rotary damper (at revolute joints) | Done | `forces/viscous_damper.py` | included in test_dampers.py |
| 3 | Motor with linear T-ω droop | Done | `forces/motor.py` | `test_motor.py` (7 tests: T-ω characteristic, overspeed, reverse, action-reaction) |
| 4 | Linear actuator (constant force with speed limit) | Done | `forces/linear_actuator.py` | (tested via integration benchmarks) |
| 5 | Gas spring (pressure-based force + velocity damping) | Done | `forces/gas_spring.py` | (tested via integration benchmarks) |
| 6 | Mass matrix M assembly (block-diagonal) | Done | `solvers/mass_matrix.py` | `test_inverse_dynamics.py` (4 mass matrix tests: shape, diagonal, symmetric, zero) |
| 7 | Inverse dynamics solver (Φ_qᵀ λ = Q - M q̈) | Done | `solvers/inverse_dynamics.py` | `test_inverse_dynamics.py` (6 tests: matches statics at zero mass, inertia effect, sweep) |
| 8 | Driver and joint reaction extraction for inverse dynamics | Done | `analysis/reactions.py` (reused from Phase 2) | Tested via benchmarks |
| 9 | Force element contribution breakdown | Done | `analysis/force_breakdown.py` | `test_benchmark_inertia.py` (contributions sum, inertia entry) |
| 10 | Bearing friction (constant drag + viscous + load-dependent) | Done | `forces/bearing_friction.py` | (tested via protocol compliance) |
| 11 | Motor sizing assistant (T-ω envelope check) | Done | `analysis/motor_sizing.py` | `test_benchmark_inertia.py` (4 tests: adequate, inadequate, overspeed, zero) |
| 12 | Inverse dynamics result envelopes and plotting | Done | `analysis/envelopes.py` (reused), `viz/force_plots.py` (reused) | Tested via benchmarks |
| 13 | Benchmark: 4-bar with inertia (pendulum limit) | Done | `test_benchmark_inertia.py` | 9 tests: sweep, inertia effect, envelopes, motor sizing, breakdown |
| 14 | Benchmark: slider-crank with motor | Done | `test_benchmark_inertia.py` | 1 test: full slider-crank inverse dynamics + motor sizing |
| 15 | Benchmark: damped system (energy dissipation) | Done | `test_benchmark_damped.py` | 4 tests: damper effect, breakdown, energy dissipation, combined forces sweep |

**Total tests:** 594 passing | **mypy:** strict, clean

---

## Phase 3 Complete

All 15 steps of Phase 3 are implemented. The simulator now supports inverse dynamics with
inertial loads (M*q̈), force element types (motor with T-ω droop, linear actuator, gas
spring, viscous/rotary dampers, bearing friction), mass matrix assembly, motor sizing
feasibility checks, and force element contribution breakdowns. All benchmarks validate the
pipeline: kinematics → inverse dynamics → reactions → motor sizing → envelopes.

---

## Phase 4A — Forward Dynamics (Smooth)

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | DAE formulation + Baumgarte stabilization integrator | Done | `solvers/forward_dynamics.py` | `test_forward_dynamics.py` (8 tests) |
| 2 | Constraint drift monitoring + projection correction | Done | `solvers/forward_dynamics.py` | Drift bounded < 1e-6 in pendulum test |
| 3 | Energy balance tracking (KE + PE + dissipated vs work) | Done | `analysis/energy.py` | Energy conserved < 1% in undamped pendulum |
| 4 | Forward dynamics plotting (position/velocity/energy vs time) | — | Deferred: reuse existing viz infrastructure | — |
| 5 | Benchmark: simple pendulum (known period) | Done | `test_forward_dynamics.py` | Period matches 2π√(L/g) within 2% |
| 6 | Benchmark: damped pendulum (exponential decay) | Done | `test_forward_dynamics.py` | Amplitude decreases, energy decreases |
| 7 | Benchmark: 4-bar free response (energy balance) | Done | `test_forward_dynamics.py` | Spring+gravity energy conserved < 5% |
| 8 | Benchmark: 4-bar step torque (steady state) | Done | `test_forward_dynamics.py` | Crank moves under step torque + damping |

**Note:** Mass matrix assembly upgraded to handle bodies where coordinate origin ≠ CG.
Uses parallel axis theorem: M_θθ = Izz_cg + m*|s_cg|², plus off-diagonal coupling terms
m*B(θ)@s_cg. This was required for correct forward dynamics of the simple pendulum benchmark.

---

## Phase 4B — Forward Dynamics (Nonsmooth Effects)

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | Joint limits with penalty method | Done | `forces/joint_limit.py` | `test_nonsmooth_dynamics.py` (6 unit tests + 2 benchmark tests) |
| 2 | Event detection framework (zero-crossing) | Done | `solvers/events.py` | `test_nonsmooth_dynamics.py` (2 tests: angle limit, velocity reversal) |
| 3 | Coulomb friction in forward dynamics (regularized + monitoring) | Done | `forces/friction.py` (reused from Phase 2) | `test_nonsmooth_dynamics.py` (2 tests: energy dissipation, amplitude decrease) |
| 4 | Restitution coefficient for hard stops | Done | `forces/joint_limit.py` | Integrated into JointLimit damping model |
| 5 | Benchmark: pendulum with hard stop | Done | `test_nonsmooth_dynamics.py` | Pendulum respects stop, energy decreases with damped restitution |
| 6 | Benchmark: Coulomb friction dynamics | Done | `test_nonsmooth_dynamics.py` | Friction dissipates energy, amplitude decreases |
| 7 | Benchmark: mechanism with joint limits | Done | `test_nonsmooth_dynamics.py` | 4-bar rocker stays within limit range |

---

## Phase 4B Complete

All 7 steps of Phase 4B are implemented. The simulator now supports nonsmooth effects:
penalty-based joint limits with restitution, Coulomb friction in forward dynamics, and
event detection for zero-crossings. Benchmarks verify correct stop behavior, energy
dissipation, and amplitude decay.

**Total tests:** 615 passing | **mypy:** strict, clean

---

## Phase 4 Exit Gate — Golden Test Fixtures

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | Golden fixture export script | Done | `scripts/export_golden.py` | Generates all 6 fixture files |
| 2 | Golden fixture: 4-bar kinematics | Done | `data/benchmarks/golden/fourbar_kinematics.json` | 61 steps, q/q_dot/q_ddot + coupler point |
| 3 | Golden fixture: slider-crank kinematics | Done | `data/benchmarks/golden/slidercrank_kinematics.json` | 67 steps |
| 4 | Golden fixture: 6-bar kinematics | Done | `data/benchmarks/golden/sixbar_kinematics.json` | 25 steps |
| 5 | Golden fixture: 4-bar statics (gravity) | Done | `data/benchmarks/golden/fourbar_statics.json` | 61 steps, λ + reactions |
| 6 | Golden fixture: 4-bar inverse dynamics | Done | `data/benchmarks/golden/fourbar_inverse_dynamics.json` | 61 steps, λ + M*q̈ |
| 7 | Golden fixture: pendulum forward dynamics | Done | `data/benchmarks/golden/pendulum_dynamics.json` | 250 steps, trajectory + energy |
| 8 | Golden fixture comparison tests | Done | `test_golden_fixtures.py` | 10 tests: position, velocity, acceleration, coupler, statics, inverse dynamics, dynamics |

---

## Phase 4 Exit Gate Complete

All golden test fixtures exported to `data/benchmarks/golden/`. The Python solver is
validated across all analysis modes with 625 tests. Golden fixture comparison tests verify
reproducibility. This satisfies the entry criteria for the Rust port.

**Total tests:** 625 passing | **mypy:** strict, clean

---

## Crank Selection Analysis & Viewer Fixes

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | Crank selection analysis module | Done | `analysis/crank_selection.py` | `test_crank_selection.py` (14 tests: topology detection, Grashof recommendation, numerical probing) |
| 2 | Build-time warning for suboptimal crank | Done | `core/mechanism.py` | `test_crank_selection.py` (3 tests: warns on suboptimal, silent on optimal, silent on non-4-bar) |
| 3 | Sweep failure warning (>10% failures) | Done | `viz/interactive_viewer.py` | `test_crank_selection.py` (1 test: non-Grashof triggers warning) |
| 4 | DRY refactor: `_detect_fourbar_link_lengths` | Done | `viz/interactive_viewer.py` | Uses `detect_fourbar_topology` from analysis module |
| 5 | Fix 4-bar viewers (geometric q0) | Done | `scripts/view_crank_rocker.py`, `scripts/view_double_crank.py` | 360/360 convergence |
| 6 | Resize 6-bar viewers for full rotation | Done | `scripts/view_sixbar*.py` (5 files) | 360/360 convergence |
| 7 | Multi-coupler trace support | Done | `core/bodies.py`, `core/mechanism.py`, `solvers/sweep.py` | `test_coupler.py`, `test_sweep.py` |
| 8 | `Mechanism.add_trace_point()` convenience API | Done | `core/mechanism.py` | `test_coupler.py` |
| 9 | Connection line visualization (trace-to-centroid) | Done | `viz/interactive_viewer.py` | Visual verification via viewer scripts |
| 10 | Multi-color trace rendering | Done | `viz/interactive_viewer.py` | Visual verification via viewer scripts |

### Crank Selection API

The `analysis.crank_selection` module provides:

- **`recommend_crank_fourbar(mechanism)`** — Grashof-based ranking of ground-adjacent
  links by rotation capability. Returns `list[CrankRecommendation]`, best first.
  For Grashof crank-rocker, the shortest grounded link gets 360 degrees.

- **`estimate_driven_range(mechanism, q0)`** — Numerical probing for any mechanism
  topology. Solves at 72 evenly spaced angles and reports estimated range in degrees.

- **`detect_fourbar_topology(mechanism)`** — Returns `FourbarTopology` dataclass or
  None. Identifies ground-adjacent bodies, coupler, link lengths, and current driver.

### Override Pattern

The user can always override the crank selection by specifying their own revolute
driver. The analysis functions and build-time warnings are advisory, not prescriptive.
If limited rotation range is intentional (e.g., `view_double_rocker.py`), the user
specifies their own driver and the warning can be suppressed with `warnings.filterwarnings`.

**Total tests:** 1078 passing | **mypy:** strict, clean

---

## Phase 5 — Interactive GUI Editor (MVP)

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | Solver API accessors for GUI | Done | `core/constraint.rs`, `core/mechanism.rs` | Existing solver tests pass |
| 2 | egui/eframe app shell + binary target | Done | `gui/mod.rs`, `bin/linkage_gui.rs` | Compiles, window opens |
| 3 | AppState + sample mechanism builders | Done | `gui/state.rs`, `gui/samples.rs` | 7 tests |
| 4 | 2D canvas rendering + pan/zoom + hit testing | Done | `gui/canvas.rs` | Visual verification |
| 5 | Angle slider + solver integration | Done | `gui/input_panel.rs` | Visual verification |
| 6 | Read-only property panel | Done | `gui/property_panel.rs` | Visual verification |
| 7 | App shell wiring (menu, panels, status bar) | Done | `gui/mod.rs` | Visual verification |
| 8 | Code review fixes | Done | `gui/state.rs`, `gui/canvas.rs`, `gui/mod.rs` | 130 tests total |

**Phase 5 MVP scope:** Read-only visualization shell. Loads hardcoded sample mechanisms (4-bar crank-rocker, slider-crank). Canvas renders from solved world-space poses. Angle slider drives kinematic solver. Click-to-select with read-only property inspection. Pan/zoom. Debug overlay with IDs and solver status.

**Not yet implemented (full Phase 5):** Image/animation export (PNG, GIF/MP4), load cases.

**Total tests:** 238 passing (212 unit + 8 golden + 18 singular) | **Rust toolchain:** stable

---

## Phase 5 — Sub-project 1: Animation Playback + Driver Selection

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | Mechanism API (grounded_revolute_joint_ids, driver_body_pair) | Done | `core/mechanism.rs` | 2 tests |
| 2 | Flexible sample builder (build_sample_with_driver) | Done | `gui/samples.rs` | 3 tests |
| 3 | Animation + driver state in AppState | Done | `gui/state.rs` | 3 tests |
| 4 | Playback controls (play/pause, speed, loop/once) | Done | `gui/input_panel.rs` | Visual verification |
| 5 | Right-click context menu for driver reassignment | Done | `gui/canvas.rs` | Visual verification |
| 6 | Animation stepping + pending actions in update loop | Done | `gui/mod.rs` | Visual verification |
| 7 | Clippy cleanup | Done | Multiple files | 161 tests total |

---

## Phase 5 — Sub-project 2: JSON Save/Load

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | DriverMeta enum + driver serialization | Done | `core/driver.rs`, `io/serialization.rs` | 2 tests |
| 2 | File > Open/Save JSON with rfd file dialog | Done | `gui/mod.rs`, `gui/state.rs`, `Cargo.toml` | Visual verification |

---

## Phase 5 — Undo/Redo Infrastructure

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | Undo/redo stack with mechanism JSON snapshots | Done | `gui/undo.rs` | — |
| 2 | Ctrl+Z / Ctrl+Y keyboard bindings wired in app loop | Done | `gui/mod.rs` | Visual verification |

---

## Phase 5 — Plotting + Sweep Visualization

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | Sweep data computation (coupler trace, body angles, transmission angle) | Done | `gui/state.rs` | — |
| 2 | Plot panel with egui_plot (coupler trace, body angles, transmission angle) | Done | `gui/plot_panel.rs` | Visual verification |

---

## Phase 5 — Sample Mechanism Gallery

| Step | Description | Status | Key files | Tests |
|------|-------------|--------|-----------|-------|
| 1 | 4-bar: CrankRocker, DoubleRocker, DoubleCrank, Parallelogram, Chebyshev, TripleRocker | Done | `gui/samples.rs` | 6 tests |
| 2 | 6-bar variants (SixBarB1/Watt I, SixBarA1, SixBarA2, SixBarB2, SixBarB3) | Done | `gui/samples.rs` | Visual verification |

**Total Rust tests:** 201 passing

---

## Phase 5 — Sub-project 4+5: Core Editor + Basic Validation

| Step | Description | Status | Key files |
|------|-------------|--------|-----------|
| 1 | MechanismBlueprint + rebuild pipeline | Done | `gui/state.rs`, `io/serialization.rs` |
| 2 | Editable mass/Izz in property panel | Done | `gui/property_panel.rs` |
| 3 | Canvas drag for attachment points | Done | `gui/canvas.rs` |
| 4 | Create/delete bodies via context menu | Done | `gui/canvas.rs`, `gui/state.rs` |
| 5 | Create/delete joints via context menu | Done | `gui/canvas.rs`, `gui/state.rs` |
| 6 | Two-click joint creation workflow | Done | `gui/canvas.rs` |
| 7 | Add ground pivots via context menu | Done | `gui/state.rs` |
| 8 | Basic validation (DOF, connectivity, driver) | Done | `gui/state.rs`, `gui/mod.rs` |
| 9 | Undo integration for all edit operations | Done | `gui/state.rs` |

**Total tests:** 226 passing
