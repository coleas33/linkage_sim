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

## Bugs Found in Second Review

| # | Issue | Status |
|---|---|---|
| B1 | `restore_snapshot` omits `compute_forces`, `update_grashof`, `mark_sweep_dirty` after undo | [x] 4a51796 |
| B2 | `restore_snapshot` divides by zero `driver_omega` — NaN propagation | [x] 4a51796 |
| B3 | Gas spring produces zero force when `stroke == 0` | [x] 4a51796 |

## Code Quality (Second Review)

| # | Issue | Status |
|---|---|---|
| D1 | `save_to_file` / `write_json_to` duplication | [x] 5356972 — save delegates to write |
| D2 | Driver-joint detection repeated 3 times | [x] 5356972 — extracted helper |
| D3 | Dead code: `forces/gravity.rs` + `forces/assembly.rs` | [x] 5356972 — deleted |
| D4 | Solve-position-then-update pattern copy-pasted 5 times | [ ] |
| D5 | `fourbar_initial_guess` diverges across 3 test files | [ ] |

## Force Validation + Tests (Second Review)

| # | Issue | Status |
|---|---|---|
| F4 | No force element validation at build time | [x] 5356972 — `validate_force_elements()` + 8 tests |
| T1 | No gas spring `stroke == 0` test | [x] 5356972 |
| T3 | `GrashofType::ChangePoint` zero coverage | [x] 5356972 |
| T4 | No compressed spring test | [ ] |

## Documentation

| # | Document | Status |
|---|---|---|
| Doc1 | WASM deployment guide | [x] b71a89f — `docs/WASM_DEPLOYMENT.md` |
| Doc3 | Sample mechanism descriptions | [x] b71a89f — `docs/SAMPLES.md` |
| Shortcuts | Keyboard & mouse reference | [x] b71a89f — `docs/SHORTCUTS.md` |
| Doc2 | Force element equations reference | [ ] |
| Doc4 | Parametric study user guide | [ ] |

## Remaining (Lower Priority)

### DRY
- Solve-position-then-update pattern (5 copies in state.rs)
- `fourbar_initial_guess` diverging test helpers
- Constraint projection duplication in forward_dynamics.rs
- Condition number computation inconsistent between statics.rs and inverse_dynamics.rs

### Tests
- Compressed spring test
- Golden fixture lambda tolerance inconsistency (4-bar 0.5 vs slider-crank 1e-4)
- Undo tests don't verify `q` survives full cycle

### Features
- Cam follower GUI (math done, UI missing)
- 8 more sample mechanisms (Watt, Stephenson, Roberts, etc.)
- WASM autosave / persistent storage
- Persistence module extraction (state.rs → persistence.rs)
