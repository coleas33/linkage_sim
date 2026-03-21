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
| U1 | **Save/export completes silently** — no toast, no title change, no feedback at all. Errors go to `log::error!()` which is invisible. | `mod.rs:88–354` | Critical |
| U2 | **Right-click drag (pan) conflicts with context menu** — short right-click opens menu, long right-click pans. No cursor change, no documented alternatives (Shift+drag, middle-click). | `canvas.rs:876–884, 1239–1390` | Critical |
| U3 | **Force toolbar doesn't show which body it targets** — user can add forces without knowing they'll attach to the wrong body. | `force_toolbar.rs:142–169` | Critical |

### High-Impact UX

| # | Issue | Location | Impact |
|---|---|---|---|
| U4 | **Solver failure shows only a tiny red dot** — no explanation of what failed or what to try. Canvas freezes at last pose. | `state.rs:957–992, canvas.rs:161` | High |
| U5 | **"Draw Link" hint says click empty space works, but it doesn't** — hint text is incorrect, confuses new users. | `canvas.rs:921–946` | High |
| U6 | **Parametric study inputs have no units and no validation** — entering 0 for mass crashes statics solver silently. | `parametric_panel.rs:45–72` | High |
| U7 | **No "Fit to View" / zoom-to-fit** — mechanism can be off-screen after load with no recovery shortcut. | `canvas.rs, mod.rs` | High |
| U8 | **No unsaved-changes indicator** — `dirty` flag tracked but never shown. Users close window and lose work. | `state.rs:722, mod.rs` | High |
| U9 | **Clicking on plots doesn't scrub mechanism angle** — plots are view-only, no bidirectional coupling. | `plot_panel.rs, input_panel.rs` | High |

### Medium-Impact UX

| # | Issue | Location | Impact |
|---|---|---|---|
| U10 | **Driver joint panel says "right-click to change" with no visual cue** for which canvas element to right-click. | `input_panel.rs:78` | Medium |
| U11 | **Counterbalance assistant uses raw SI** even when display units are mm — inconsistent with rest of UI. | `parametric_panel.rs:231–247` | Medium |
| U12 | **"Add Body" tool doesn't show point count** or minimum needed — double-click with 1 point silently does nothing. | `canvas.rs:723–752, 1091–1141` | Medium |
| U13 | **Kinematic and simulation playback can run simultaneously** — produces confusing canvas motion. Should be mutually exclusive. | `mod.rs:587–603, input_panel.rs:175–220` | Medium |
| U14 | **Load case switching doesn't push undo** — previous driver config lost with no recovery. | `input_panel.rs:282–284` | Medium |
| U15 | **Expression driver only applies on focus-lost** — typing formula then clicking Play uses stale expression. | `input_panel.rs:376–415` | Medium |

### Visual Quality & Polish

| # | Issue | Location | Impact |
|---|---|---|---|
| V1 | **Zoom step 1.04 is far too slow** — takes ~58 scroll ticks to double zoom. CAD tools use 1.10–1.15. | `canvas.rs:64` | High |
| V2 | **Ground line clips at x +/- 10m** instead of spanning viewport. Looks like a physical boundary. | `canvas.rs:216–222` | High |
| V3 | **Link stroke width disagrees between preview and final** — `BODY_STROKE_WIDTH=3.5` unused, actual polygon uses `1.5`. Link visually "snaps thinner" on drop. | `canvas.rs:57, 332` | High |
| V4 | **Dimension labels: 10px offset, no background box, 10px font** — unreadable at many zoom levels. | `canvas.rs:378–391` | High |
| V5 | **Force magnitude labels 9px with no background** — primary statics output is unreadable. | `canvas.rs:1784–1894` | High |
| V6 | **SVG export uses hairlines, not filled bars** — export looks like stick diagram, not physical links. | `export.rs:247, 285–293` | High |
| V7 | **PNG rasterizer fills white before dark SVG** — fragile, white bleeds through transparency. | `export.rs:413` | Medium |
| V8 | **Rotary force elements render as floating text only** — no line to bodies, no icon distinguishing motor from spring. | `canvas.rs:1469–1621` | Medium |
| V9 | **Grid vanishes when zoomed out** instead of coarsening to next level. | `canvas.rs:1967–1970` | Medium |
| V10 | **Coupler trace dashes fixed in pixels** — don't scale with zoom. Nearly solid when zoomed in, invisible when zoomed out. | `canvas.rs:255–274` | Medium |
| V11 | **HTML report timestamp is raw Unix seconds** — "Generated: Unix timestamp 1742478000". | `export.rs:834–842` | Medium |
| V12 | **SVG body labels overlap joint circles** — placed 8px above first attachment point instead of centroid. | `export.rs:296–304` | Low |

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
