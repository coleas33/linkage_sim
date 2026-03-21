# Code Review Improvements — March 20, 2026

Deep review of the Rust application (`linkage-sim-rs/`, ~30K LOC) across architecture, code quality, tests, and performance.

## Status Legend

- [ ] Pending
- [x] Complete

---

## Quick-Fix Bugs

| # | Issue | Location | Status |
|---|---|---|---|
| Q1 | `add_force_element` doesn't call `rebuild()` | `gui/state.rs:1795` | [x] Already had rebuild() — false positive |
| Q2 | `driver_joint_id` not cleared when `rebuild()` finds no driver | `gui/state.rs:1599` | [x] df07cdc |
| Q3 | `update_grashof` not called in `load_from_file` | `gui/state.rs:~1499` | [x] df07cdc |
| Q4 | `solve_augmented` silently returns zero vector on singular matrix | `solver/forward_dynamics.rs:160` | [x] df07cdc — returns Option, RK4 aborts early |
| Q5 | Schema version is exact string equality — breaks forward compat | `io/serialization.rs:388` | [x] df07cdc — semver major comparison |
| Q6 | Expression modulation fallback to `1.0` on parse error | `forces/elements.rs:88` | [x] df07cdc — fallback to 0.0 + log warning |

## Agreed Improvements

### 1. Architecture — Extract modules from `state.rs` (5,677 LOC monolith)

Extract `gui/sweep.rs` (~350 lines) and `gui/persistence.rs` (~300 lines) as the lowest-risk first step.

**Status:** [x] sweep.rs extracted (1218b40) — persistence.rs deferred

### 2. Code Quality — Fix panicking `unwrap()`/`expect()` in production paths

Replace panicking body-ID lookups with graceful error handling in:
- `forces/elements.rs` — RotaryDamper, BearingFriction, Motor, JointLimit evaluators
- `io/serialization.rs` — `joint_to_json` HashMap indexing
- `analysis/energy.rs` — `compute_kinetic_energy`

**Status:** [x] df07cdc — helpers extracted, all expect() eliminated

### 3. Tests — Cam follower gamma FD test + fix math

The `CamFollowerJoint::gamma()` omits centripetal terms from the rotating direction vector. Add a finite-difference validation test (matching existing pattern for other joints) and implement the full gamma derivation.

**Status:** [x] 1218b40 — full gamma implemented, 3 FD tests added

### 4. Performance — Hot-path optimizations

| Item | Location | Fix | Status |
|---|---|---|---|
| Expression re-parse every timestep | `forces/elements.rs:83–93` | Pre-compile at sim start | [ ] In progress |
| `all_constraints()` allocates Vec per call | `core/mechanism.rs:131` | Return `impl Iterator` with `.chain()` | [x] df07cdc |
| Undo `Vec::remove(0)` is O(n) | `gui/undo.rs:56` | Replace with `VecDeque` | [x] df07cdc |

---

## Additional Findings (Not Yet Prioritized)

### DRY Violations in `state.rs`
- Solve-position-then-update pattern copy-pasted 5 times
- Driver-joint detection logic repeated 4 times
- `save_to_file` / `write_json_to` near-duplicate
- `compute_sweep_data` is 240 lines — should be broken up

### DRY in Solver
- Constraint projection block duplicated between `simulate` and `simulate_with_events` (forward_dynamics.rs)
- Condition number computation inconsistent between statics.rs and inverse_dynamics.rs

### DRY in Tests
- `fourbar_initial_guess` duplicated across 3 test files with diverging implementations
- `build_standard_fourbar` / `build_standard_fourbar_with_gravity` near-identical

### Force Element Issues
- Gas spring formula incorrect when `stroke == 0` (silent wrong physics)
- No force element validation at build time (`validate_force_elements` doesn't exist)
- `Gravity` struct in `gravity.rs` clones entire bodies map (possibly dead code)

### GUI Issues
- `draw_canvas` is 2,014 lines in a single function
- `draw_force_elements` repeats rotary-element rendering pattern 5 times
- `export_sweep_csv` / `export_coupler_csv` inconsistent empty-data handling

### Test Coverage Gaps
- No compressed spring test
- `GrashofType::ChangePoint` has zero coverage
- Golden fixture lambda tolerance for 4-bar is 0.5 N·m vs 1e-4 for slider-crank
- Undo tests don't verify `q` survives full cycle
