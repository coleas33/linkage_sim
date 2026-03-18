# Parallel Workstreams: Crank Selection + Phase 5 MVP + Documentation

Three workstreams that can proceed in parallel alongside the ongoing Rust solver port.

---

## Dependencies and Ordering

```
Workstream 1 (Crank Selection)  ──────────────────────────────►  lands first
        │                                                         (solver value)
        │  public solver API stable
        ▼
Workstream 2 (Phase 5 MVP)     ──────────────────────────────►  begins in parallel
                                                                  (after core/solver/ compiles)

Workstream 3 (Documentation)   ── updates reflect landed milestones only ──►
```

- **Workstream 1** can proceed independently and should land first for solver parity.
- **Workstream 2** can begin in parallel once public solver APIs are stable enough to drive the canvas. The Rust port already has `Mechanism`, constraints, drivers, and kinematic solvers — that is sufficient.
- **Workstream 3** updates reflect actual landed milestones, not planned ones.

**Sequencing deviation:** ROADMAP.md and RUST_MIGRATION.md specify the GUI starts after the full solver port (step 10). This spec starts Phase 5 MVP after steps 1-5 (core + kinematics), in parallel with the remaining solver port (forces, statics, dynamics, analysis). This is justified because the MVP is read-only visualization — it consumes the kinematic solver API, which is already stable and ported. Workstream 3 must update ROADMAP.md and RUST_MIGRATION.md to reflect this parallelized approach.

---

## Workstream 1: Crank Selection — Finish Python + Port to Rust

### Non-goal

Workstream 1 does not attempt general mechanism classification, assembly-mode enumeration, or global optimal driver synthesis. It is scoped to grounded 4-bar linkages using Grashof heuristics and continuation-based numerical probing.

### Python: Rewrite `estimate_driven_range()` + cleanup

**Core crank selection (complete, covered by dedicated unit tests):**
- `classify_grounded_fourbar_topology()` — detect and classify standard grounded 4-bar linkages within the current mechanism representation. Not a general graph-theoretic classifier.
- `recommend_crank_fourbar()` — Grashof-based ranking of which link to drive
- `estimate_driven_range()` — numerical probing via position solver (currently simple probe-count; rewrite below)
- Build-time warning in `Mechanism.build()`
- Viewer integration with DRY refactor

**Rename `detect_fourbar_topology()` to `classify_grounded_fourbar_topology()`** in Python. The current name implies broader authority than the function delivers. It classifies common grounded 4-bar forms inside the current mechanism representation — it is not a general graph-theoretic mechanism classifier. Update all call sites: `crank_selection.py`, `mechanism.py`, `interactive_viewer.py`, `test_crank_selection.py`.

**Rewrite `estimate_driven_range()` to continuation-based sweep:**

The current implementation is a simple "count how many of N probes converge" approach with no branch tracking. This is a rewrite to a continuation-based sweep with branch-aware range detection. The return type changes from a single float (degrees) to a `DrivenRangeResult` dataclass.

**`DrivenRangeResult` definition (Python dataclass, Rust struct):**
```
DrivenRangeResult:
    range_start: float        # first reachable angle (radians)
    range_end: float          # last reachable angle on this branch (radians)
    range_degrees: float      # range_end - range_start, in degrees
    full_rotation: bool       # True if range >= 360
    n_probes: int             # total probes attempted
    n_converged: int          # probes that converged on-branch
    marginal_angles: list     # angles where sigma_min < threshold (reachable but near-singular)
    termination_reason: str   # "full_rotation" | "convergence_failure" | "branch_discontinuity" | "singular_lockup"
```

**Branch policy:**

> `estimate_driven_range()` follows a continuation-based sweep on a fixed assembly branch and reports the contiguous reachable driver interval before loss of convergence, singular lockup, or branch discontinuity.

- **Continuation from current branch:** each probe uses the previous converged state as initial guess
- **Range end:** NR fails to converge within tolerance (1e-10) and max iterations (50)
- **Failure:** residual norm exceeds tolerance at final iterate
- **Branch continuity (primary):** computed via a signed orientation invariant — the signed area of a triangle formed by three non-collinear joint locations in the coupler loop (see Appendix A). A sign flip of this invariant away from a singular configuration indicates a branch discontinuity. The range is terminated at the previous angle.
- **Branch jump (secondary guard):** if `||q_new - q_prev|| > jump_threshold` (default: max link length), the step is flagged as a suspicious jump. This catches gross failures that the orientation invariant might miss due to numerical noise, but it is not the primary branch test because `q` mixes translations and angles, and large angular wraps can look like jumps on the same branch.
- **Near-singular:** if `sigma_min(Phi_q) < singular_threshold` (default: 1e-6), the angle is flagged as reachable-but-marginal and included in the range with a warning

**Rewritten Python function signature:**
```python
def estimate_driven_range(
    mechanism: Mechanism,
    driver_joint_id: str,
    q0: NDArray[np.float64],
    step_deg: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 50,
    jump_threshold: float | None = None,   # default: max link length
    singular_threshold: float = 1e-6,
) -> DrivenRangeResult:
```

The current Python `estimate_driven_range(mechanism, q0, n_probes)` is replaced with this signature. The `driver_joint_id` parameter identifies the revolute joint where the driver is applied (matching the Rust API). The `n_probes` parameter is replaced by `step_deg` (angular step size in degrees), which determines probe count implicitly.

**Remaining Python tasks (not part of core crank selection):**
- Complete 3 six-bar script resizes (A2, B2, B3) — these are validation/examples for crank-selection outcomes, not part of the core feature

### Rust: Port to `analysis/crank_selection.rs`

**Structs:**
- `FourbarTopology` — detected topology with link lengths and role assignments
- `CrankRecommendation` — ranking with Grashof classification, recommended driver, range estimate
- `DrivenRangeResult` — contiguous range + metadata (same fields as Python definition above)

**Functions:**
- `classify_grounded_fourbar_topology(mech: &Mechanism) -> Option<FourbarTopology>`
- `recommend_crank_fourbar(mech: &Mechanism) -> Vec<CrankRecommendation>`
- `estimate_driven_range(mech: &Mechanism, driver_joint_id: &str, ...) -> DrivenRangeResult`

Note: the API takes `driver_joint_id` (the revolute joint where the driver is applied), not a body ID. The driver is associated with a specific joint in the mechanism, and using the joint ID avoids ambiguity when a body participates in multiple joints. The Python API should be updated to match.

**Dependencies:**
- `Mechanism` with body/joint graph access (ported)
- Position solver for numerical probing (ported)
- Grashof classification logic — **not yet ported**; port from `analysis/grashof.py` as a prerequisite step within this workstream

**Test fixtures (minimum 3 deterministic cases):**

| Fixture | Type | (d, a, b, c) | Expected |
|---------|------|--------------|----------|
| Grashof crank-rocker | s+l < p+q | (4.0, 2.0, 4.0, 3.0) | Full 360 range, link `a` (shortest) recommended as driver |
| Non-Grashof double-rocker | s+l > p+q | (5.0, 3.0, 4.0, 7.0) | Limited range (<360), `full_rotation == False`. Exact `termination_reason` is implementation-dependent — may be `"convergence_failure"` or `"branch_discontinuity"` depending on whether the orientation invariant trips before NR diverges. Test should verify `full_rotation == False` and `range_degrees < 360`. |
| Change-point | s+l = p+q | (3.0, 3.0, 3.0, 3.0) | Edge case: full rotation possible but marginal_angles non-empty near singular configs |

Initial guess `q0`: geometric solution at crank angle 0 using law-of-cosines (same method as `view_crank_rocker.py`).

Rust output must match Python output on all three fixtures within tolerance.

### Definition of Done — Workstream 1

- [ ] Python: `detect_fourbar_topology()` renamed to `classify_grounded_fourbar_topology()` across all 4 files
- [ ] Python: `estimate_driven_range()` rewritten to continuation-based sweep with `DrivenRangeResult` return type
- [ ] Python: branch policy implemented (orientation invariant primary, state-norm secondary, near-singular flagging)
- [ ] Python: all crank selection tests pass (existing + new tests for rewritten range estimation)
- [ ] Rust: `analysis/crank_selection.rs` compiles and exports public API
- [ ] Rust: crank selection matches Python outputs on all 3 named fixtures (exact dimensions specified above)
- [ ] Rust: range estimation behavior matches documented branch policy
- [ ] Rust: recommendation tests pass for all Grashof classifications

---

## Workstream 2: Phase 5 — egui App Shell + Canvas Prototype

### Non-goal

Workstream 2 is a visualization shell, not yet an editor. It renders and inspects mechanisms but does not create or modify them.

### MVP scope

**Sample mechanism source:** The MVP loads mechanisms from hardcoded Rust builder functions (e.g., `build_fourbar()` already exists in the test suite). No JSON file loading in this iteration — JSON round-trip in Rust depends on serde integration for mechanism serialization, which is a separate port step.

**App shell:**
- Main window with menu bar: **File > Load Sample > [4-bar, Slider-Crank], Quit**
- JSON Open/Save is a future menu item, disabled or hidden until Rust serde mechanism round-trip lands
- Left side panel for property inspection
- Central canvas area
- Bottom status bar (solver status, mechanism stats)

**Canvas (`gui/canvas.rs`):**
- Render a built `Mechanism` from **solved world-space attachment points and poses** — not from original sketch geometry
- Bodies: lines between attachment points in world space
- Revolute joints: circles at joint locations
- Prismatic joints: rendered only if the prismatic constraint is already functional in the Rust solver (it is — `PrismaticJoint` is ported in `constraint.rs`). Arrow along slide axis.
- Ground rendering rule:
  - Fixed pivot: triangle marker at grounded attachment points
  - Grounded body: hatched/shaded region or thick base line distinguishing it from moving bodies
  - Ground plane: subtle horizontal line at y=0 when relevant
- Pan and zoom from day one (egui `Painter` with world-to-screen affine transform)

**Solver integration:**
- Slider for input angle with explicit range rule:
  - If a `DrivenRangeResult` is available (from crank selection), slider uses the detected range with labeled endpoints
  - Otherwise, slider defaults to 0..360 degrees
  - If range is partial (<360), show endpoints and a warning badge indicating limited range
- **Re-solve on slider change.** Cache the last successful `q` as the initial guess for the next solve. This makes drag interaction smooth — continuation from the previous solution is both fast and branch-stable.
- If solver fails to converge at a slider position, hold the last successful pose on canvas and show failure state in status bar. Do not blank the canvas.

**Selection model:**
```rust
/// What kind of mechanism element is selected.
enum SelectedEntity {
    Body(String),
    Joint(String),
    Driver(String),
}

// Selection state in AppState:
selected: Option<SelectedEntity>
```
Define this once. `Option<SelectedEntity>` is idiomatic Rust — `None` means nothing selected, `Some(...)` means an element is selected. Click a body/joint on canvas to select. Property panel shows read-only parameters for the selected entity. Click empty space to deselect (`selected = None`).

**Debug overlay (toggleable, on by default in debug builds):**
- Body IDs at body CG positions
- Joint IDs at joint locations
- Attachment point labels (small, dimmed)
- Solver convergence status: green dot = converged, red dot = failed, yellow = marginal (near-singular)
- Residual norm in status bar

**Not in this iteration:**
- Editing, drag-and-drop, creating new bodies/joints
- Undo/redo
- Force element visualization
- Animation playback controls (play/pause/speed)
- Plotting panels
- JSON Open/Save (until Rust serde mechanism round-trip lands)

### Architecture

```
gui/
├── mod.rs          -- App struct, eframe::App impl, top-level layout
├── canvas.rs       -- 2D mechanism renderer, pan/zoom, hit testing, debug overlay
├── property_panel.rs  -- read-only display for SelectedEntity
├── state.rs        -- AppState: owns Mechanism, current q, selection, UI flags
└── input_panel.rs  -- slider for driver angle, solver status display
```

**`AppState`** owns:
- `Mechanism` (built, immutable for MVP)
- Current `q: DVector<f64>` (last converged solution)
- `selected: Option<SelectedEntity>`
- `driver_angle: f64` (slider value)
- `solver_status: SolverStatus` (converged/failed/marginal + residual)
- `show_debug_overlay: bool`
- `view_transform: ViewTransform` (pan offset + zoom scale)

**Immediate-mode pattern:** each frame reads `AppState`, renders canvas + panels. Slider change triggers solver, updates `q` and `solver_status`. Canvas always renders from the current `q`.

### Definition of Done — Workstream 2

- [ ] `egui`, `eframe` added to Cargo.toml
- [ ] App launches with menu bar, side panel, canvas area, status bar
- [ ] At least one benchmark mechanism (4-bar) loads from hardcoded builder function via File > Load Sample
- [ ] Slider changes input angle; solver updates pose live
- [ ] Canvas renders mechanism from solved world-space poses
- [ ] Selection model works: click body/joint, property panel shows read-only data
- [ ] Pan and zoom stable on canvas
- [ ] Debug overlay toggleable (IDs, solver status)
- [ ] Ground rendering follows defined rule (triangles at fixed pivots)
- [ ] Solver failure shows last good pose + failure indicator (does not blank canvas)

---

## Workstream 3: Documentation Updates

**Principle:** Docs distinguish between completed Python feature, completed Rust port, and active GUI work. No claim exceeds shipped functionality.

### ROADMAP_IMPLEMENTATION.md

Add section: **Crank Selection Analysis**

| Task | Status | Key files | Tests |
|------|--------|-----------|-------|
| Core module: topology classification + Grashof ranking | Done | `analysis/crank_selection.py` | Covered by dedicated unit tests |
| Numerical probing for driven range estimation | Done | `analysis/crank_selection.py` | Covered by dedicated unit tests |
| Build-time warning in Mechanism.build() | Done | `core/mechanism.py` | Covered by dedicated unit tests |
| Sweep failure warning + DRY refactor | Done | `viz/interactive_viewer.py` | Covered by unit test |
| Viewer scripts: geometric initial guess | Done | `scripts/view_crank_rocker.py`, `view_double_crank.py` | Manual verification |
| Six-bar resizes for full rotation | Partial | `scripts/view_sixbar.py`, `view_sixbar_A1.py` | 2 of 5 done; A2, B2, B3 remaining |

**Known limitations:**
- Classification limited to grounded 4-bar linkages within the current mechanism representation. Not a general-purpose mechanism classifier.
- Range estimation is continuation-based (single branch). Does not detect or explore alternative assembly modes.
- Recommendations are Grashof-heuristic: they rank by likelihood of full rotation, not by application-specific criteria (e.g., transmission angle, torque).
- Near-singular configurations are flagged but included in the reported range.

### ROADMAP.md

- Crank selection analysis is part of the port scope (`analysis/crank_selection.rs`)
- Phase 5 MVP shell in progress — app shell, read-only canvas, solver-driven animation. Not yet an interactive editor.
- **Sequencing update:** Phase 5 MVP begins after core solver APIs are stable (port steps 1-5), not after all analysis modules (step 9). The MVP is a visualization shell that consumes only the kinematic solver API. Full editor capabilities (Phase 5 complete) still require the full solver port.

### RUST_MIGRATION.md

Add to module mapping:
```
analysis/crank_selection.rs  ← analysis/crank_selection.py
```

Add to port sequencing (step 9, alongside other analysis modules):
- **Dependencies:** mechanism graph access, position solver continuation, Grashof classification
- **Benchmark parity:** must match Python on 3 named fixtures (crank-rocker, double-rocker, change-point) with exact dimensions specified in this spec

**Sequencing update:** Add note that Phase 5 MVP (visualization shell) begins in parallel after port steps 1-5 complete. The original linear sequencing (steps 1-9 then step 10 GUI) is revised to allow the read-only GUI to develop alongside the remaining solver port. This does not change the full Phase 5 exit criteria — the complete interactive editor still requires the full solver port.

**GUI module layout reconciliation:** Update the `gui/` module mapping to reflect the MVP layout. Add `state.rs` and `input_panel.rs` (new in MVP). Keep `animation.rs` and `plot_panel.rs` as planned-but-not-yet-implemented entries (these are post-MVP, part of full Phase 5).

### README.md

Add to capabilities list:
> **Crank selection analysis** for supported four-bar mechanisms — Grashof-based classification, driver ranking, and numerical range estimation

Not a broad promise. Scoped to what's actually stable. Do not imply JSON round-trip or GUI capabilities that are not yet shipped.

### Definition of Done — Workstream 3

- [ ] ROADMAP_IMPLEMENTATION.md has crank selection section with tasks, tests (descriptive, not exact counts), and known limitations
- [ ] ROADMAP.md reflects Phase 5 as "MVP shell in progress", not as a shipped GUI; includes sequencing update
- [ ] RUST_MIGRATION.md includes crank_selection.rs in module mapping and port sequencing with dependencies; includes parallelism note
- [ ] README.md adds crank selection as a scoped capability for four-bar mechanisms
- [ ] No doc claims exceed shipped functionality
- [ ] No doc references imply JSON round-trip or GUI editing capabilities are available when they are not

---

## Appendix A: Branch Orientation Invariant

Both `estimate_driven_range()` and the GUI sweep slider need a consistent definition of "same assembly branch." This appendix defines the canonical invariant used across both codebases.

### Definition

For a 4-bar linkage with coupler loop `O2 → A → B → O4`, define three non-collinear points on the loop:

- `P1` = joint A (crank-coupler joint, global coordinates)
- `P2` = joint B (coupler-rocker joint, global coordinates)
- `P3` = joint O4 (rocker-ground pivot, global coordinates)

The **signed area** of triangle `(P1, P2, P3)` is:

```
S = 0.5 * ((P2.x - P1.x) * (P3.y - P1.y) - (P3.x - P1.x) * (P2.y - P1.y))
```

This is equivalent to the z-component of the cross product `(P2 - P1) x (P3 - P1)`.

### Branch continuity rule

- **Sign flip of `S` away from singularity** (i.e., `|S_prev| > epsilon` and `sign(S_new) != sign(S_prev)`) → branch discontinuity. Terminate the range at the previous angle.
- **`|S|` approaching zero** → the triangle is degenerating (points becoming collinear). This is a near-singular configuration. Flag as marginal if `sigma_min(Phi_q) < singular_threshold`.
- **`|S|` passes through zero and recovers with same sign** → the mechanism passed through a singular configuration but stayed on the same branch. This is allowed if NR converged and the state-norm secondary guard was not triggered.

### Why this works

- `S` depends on mechanism geometry (world-space joint positions), not on the raw state vector `q` which mixes translations and angles
- `S` is invariant to choice of coordinate origin and orientation
- Sign of `S` directly encodes the assembly mode: the two solutions of the coupler triangle produce opposite signs
- Near singularity, `S → 0` regardless of branch, so the sign test is only applied when `|S|` is comfortably above zero

### Point selection for other mechanism types

For mechanisms beyond 4-bar, choose three non-collinear points from the primary kinematic loop that:
1. Are joint locations (not body CGs or arbitrary points)
2. Span the coupler — at least one point should be on a moving body interior to the loop
3. Are never collinear in the mechanism's regular operating range

The specific triple should be selected once at sweep start and reused for all probes.
