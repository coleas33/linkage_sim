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

**Total tests:** 368 passing | **mypy:** strict, clean

---

## Phase 1 Complete

All 21 steps of Phase 1 are implemented. The simulator handles 4-bar, slider-crank,
and 6-bar (with ternary links) mechanisms with full kinematic analysis: position,
velocity, acceleration, coupler point tracking, DOF validation, Jacobian rank analysis,
graph connectivity, JSON serialization, and Matplotlib visualization/animation.
