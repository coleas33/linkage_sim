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
| Expression re-parse every timestep | `forces/elements.rs:83–93` | Pre-compile at sim start | [x] 73e5859 |
| `all_constraints()` allocates Vec per call | `core/mechanism.rs:131` | Return `impl Iterator` with `.chain()` | [x] df07cdc |
| Undo `Vec::remove(0)` is O(n) | `gui/undo.rs:56` | Replace with `VecDeque` | [x] df07cdc |

---

## GUI & UX Improvements (New — March 20, 2026)

### Critical UX

| # | Issue | Location | Impact |
|---|---|---|---|
| U1 | **Save/export completes silently** | `mod.rs:88–354` | [x] 6d71f01 — green toast in status bar |
| U2 | **Right-click drag (pan) conflicts with context menu** | `canvas.rs:876–884` | [x] 030069a — drag guard |
| U3 | **Force toolbar doesn't show which body it targets** | `force_toolbar.rs:142–169` | [x] 415b0d2 — "Target: body" label |

### High-Impact UX

| # | Issue | Location | Status |
|---|---|---|---|
| U4 | **Solver failure shows only a tiny red dot** | `canvas.rs` | [x] 14900bd — red banner overlay |
| U5 | **"Draw Link" hint says click empty space works, but it doesn't** | `canvas.rs:921–946` | [x] 462b337 — corrected hint text |
| U6 | **Parametric study inputs have no units and no validation** | `parametric_panel.rs:45–72` | [x] 1af4f57 — unit suffixes + validation |
| U7 | **No "Fit to View" / zoom-to-fit** | `canvas.rs, state.rs` | [x] 14900bd — press F to fit |
| U8 | **No unsaved-changes indicator** | `state.rs, mod.rs` | [x] 6d71f01 — title bar shows dirty state |
| U9 | **Clicking on plots doesn't scrub mechanism angle** | `plot_panel.rs` | [x] 8cfac1c — click-to-scrub |

### Medium-Impact UX

| # | Issue | Location | Status |
|---|---|---|---|
| U10 | **Driver joint panel says "right-click to change" with no visual cue** | `input_panel.rs:78` | [x] cff1792 — hover highlight |
| U11 | **Counterbalance assistant uses raw SI** | `parametric_panel.rs:231–247` | [x] cff1792 — display units |
| U12 | **"Add Body" tool doesn't show point count** | `canvas.rs:723–752` | [x] 7a3055f — dynamic hint |
| U13 | **Kinematic and simulation playback can run simultaneously** | `mod.rs, input_panel.rs` | [x] 7f1eed1 — mutually exclusive |
| U14 | **Load case switching doesn't push undo** | `input_panel.rs:282–284` | [x] Already implemented |
| U15 | **Expression driver only applies on focus-lost** | `input_panel.rs:376–415` | [x] 415b0d2 — applies on change |

### Visual Quality & Polish

| # | Issue | Location | Status |
|---|---|---|---|
| V1 | **Zoom step 1.04 is far too slow** | `canvas.rs:64` | [x] 462b337 — changed to 1.12 |
| V2 | **Ground line clips at x +/- 10m** | `canvas.rs:216–222` | [x] 462b337 — viewport-wide |
| V3 | **Link stroke width disagrees between preview and final** | `canvas.rs:57, 332` | [x] 462b337 — uses constant |
| V4 | **Dimension labels: 10px font** — unreadable | `canvas.rs:378–391` | [x] 0fbef3a — 11px |
| V5 | **Force magnitude labels 9px** — unreadable | `canvas.rs:1784–1894` | [x] 0fbef3a — 11px |
| V6 | **SVG export uses hairlines, not filled bars** | `export.rs:247, 285–293` | [x] 1af4f57 — polygon bars with fill |
| V7 | **PNG rasterizer fills white before dark SVG** | `export.rs:413` | [x] 462b337 — dark fill |
| V8 | **Rotary force elements render as floating text only** | `canvas.rs:1469–1621` | [x] 030069a — letter badges + dashed lines |
| V9 | **Grid vanishes when zoomed out** instead of coarsening | `canvas.rs:1967–1970` | [x] 7a3055f — doubles spacing |
| V10 | **Coupler trace dashes fixed in pixels** | `canvas.rs:255–274` | [x] 00fcc76 — zoom-scaled |
| V11 | **HTML report timestamp is raw Unix seconds** | `export.rs:834–842` | [x] 7a3055f — formatted date |
| V12 | **SVG body labels overlap joint circles** | `export.rs:296–304` | [x] 7f1eed1 — centroid placement |

---

## Additional Findings (Code Quality — Not Yet Prioritized)

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

### Test Coverage Gaps
- No compressed spring test
- `GrashofType::ChangePoint` has zero coverage
- Golden fixture lambda tolerance for 4-bar is 0.5 N·m vs 1e-4 for slider-crank
- Undo tests don't verify `q` survives full cycle
