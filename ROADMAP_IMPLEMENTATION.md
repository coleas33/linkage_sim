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
| 5 | Mechanism assembly | | | |
| 6 | Grubler DOF count | | | |
| 7 | Jacobian rank check | | | |
| 8 | Kinematic position solver (Newton-Raphson) | | | |
| 9 | Revolute driver constraint | | | |
| 10 | Position sweep | | | |
| 11 | Velocity solver | | | |
| 12 | Acceleration solver | | | |
| 13 | Coupler point evaluation | | | |
| 14 | JSON serialization / deserialization | | | |
| 15 | Minimal Matplotlib viewer | | | |
| 16 | Animation | | | |
| 17 | Test suite — 4-bar benchmark | | | |
| 18 | Prismatic joint constraint | | | |
| 19 | Slider-crank benchmark | | | |
| 20 | Graph connectivity check | | | |
| 21 | Ternary body test (6-bar) | | | |

**Total tests:** 75 passing | **mypy:** strict, clean
