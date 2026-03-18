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

**Total tests:** 580 passing | **mypy:** strict, clean

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
| 8 | Driver and joint reaction extraction for inverse dynamics | | | |
| 9 | Force element contribution breakdown | | | |
| 10 | Bearing friction (constant drag + viscous + load-dependent) | | | |
| 11 | Motor sizing assistant (T-ω envelope check) | | | |
| 12 | Inverse dynamics result envelopes and plotting | | | |
| 13 | Benchmark: 4-bar with inertia (pendulum limit) | | | |
| 14 | Benchmark: slider-crank with motor | | | |
| 15 | Benchmark: damped system (energy dissipation) | | | |
