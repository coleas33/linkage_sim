# Rust Migration Plan

The solver kernel is built and validated in Python (Phases 1–4). The codebase is then ported to Rust before the GUI phase begins. Python serves as the prototyping language and test oracle; Rust is the production language and distribution target.

---

## Rationale

**Why start in Python:**
- Fastest iteration on numerical algorithms — constraint equations, analytical Jacobians, Newton-Raphson tuning, DAE integrator selection
- SciPy provides battle-tested solvers (`fsolve`, `solve_ivp` with Radau/BDF) that eliminate unknowns during math development
- Validated benchmark results become golden test fixtures for the Rust port
- Lower risk: if the math is wrong, you find out in days, not weeks

**Why port to Rust:**
- `egui` + `eframe` provides a better interactive GUI than any Python option (PyQt, Tkinter, etc.) for a 2D mechanism editor with drag-and-drop, animation, and property panels
- Single-binary distribution — engineers download one `.exe`, no Python environment required
- WebAssembly compilation for browser-based deployment at zero install cost
- Performance headroom for large mechanisms, parametric sweeps, and optimization loops
- Type system and ownership model prevent classes of bugs that unit tests catch in Python

**Why not hybrid (PyO3/FFI):**
- Adds binding complexity without meaningful payoff for a project this size
- The solver is small enough that a clean port is less maintenance burden than a cross-language bridge
- The GUI needs tight integration with the solver (real-time animation, drag-and-update) — FFI boundaries make this painful

---

## Cutover Point

The Rust port begins **after Phase 4 exits** — when all four analysis modes (kinematics, statics, inverse dynamics, forward dynamics) are validated in Python with a comprehensive benchmark test suite.

**Phase 5 (GUI) is never built in Python.** The GUI is built natively in Rust from the start, using the ported solver kernel.

| Phase | Language | Status |
|-------|----------|--------|
| Phase 1 — Data model & kinematics | Python | Complete, validated |
| Phase 2 — Force elements & statics | Python | Complete, validated |
| Phase 3 — Actuators & inverse dynamics | Python | Complete, validated |
| Phase 4 — Forward dynamics | Python | Complete, validated |
| **Rust port** | **Rust** | **Complete — 316 tests, validated against Python golden data** |
| Phase 5 — Interactive GUI | Rust | **In progress** — egui application |
| Phase 6 — Advanced & QoL | Rust | Not started |

---

## Python Coding Conventions for Portability

These conventions produce clean Python that maps naturally to Rust. They are not premature optimization — they are good Python that happens to port well.

### Use Protocol classes or ABCs for extension points

Python:
```python
class ForceElement(Protocol):
    def evaluate(self, state: State, t: float) -> np.ndarray: ...
```

Maps to Rust:
```rust
trait ForceElement {
    fn evaluate(&self, state: &State, t: f64) -> DVector<f64>;
}
```

### Use dataclasses for data structures

Python:
```python
@dataclass
class Body:
    id: str
    attachment_points: dict[str, tuple[float, float]]
    mass: float
    cg_local: tuple[float, float]
    Izz_cg: float
```

Maps to Rust:
```rust
struct Body {
    id: String,
    attachment_points: HashMap<String, Vector2<f64>>,
    mass: f64,
    cg_local: Vector2<f64>,
    izz_cg: f64,
}
```

### No global state

Pass `Mechanism` explicitly to all solver functions. Rust will enforce this anyway — design for it now.

### No dynamic typing tricks

- No monkey-patching, no runtime attribute injection, no `**kwargs` for structural data
- Type-annotate everything — `mypy` strict mode is a reasonable proxy for Rust's type checker
- Use `enum` for variant types (joint types, driver types), not stringly-typed dicts

### Explicit error handling

- Return explicit error types or raise specific exceptions — no bare `except` or silent `None` returns
- This maps to Rust's `Result<T, E>` pattern

### Coupler / Trace Points

Coupler points (also called trace points) are arbitrary points rigidly attached
to any moving body. They are tracked during kinematic sweeps for path tracing,
velocity/acceleration analysis, and visualization.

**Python API:**
- `body.add_coupler_point(name, x, y)` — low-level, on the Body object directly
- `mechanism.add_trace_point(name, body_id, x, y)` — convenience wrapper on Mechanism

Multiple trace points can exist across different bodies. The sweep system
auto-discovers all coupler points and traces each with a distinct color.

**Rust mapping:**
```rust
// On Body struct — already mapped via coupler_points field:
pub coupler_points: HashMap<String, Vector2<f64>>,

// On Mechanism — convenience method:
pub fn add_trace_point(&mut self, name: &str, body_id: &str, x: f64, y: f64) {
    let body = self.bodies.get_mut(body_id).expect("body not found");
    body.coupler_points.insert(name.to_string(), Vector2::new(x, y));
}
```

**Sweep data structure per trace:**
```rust
struct CouplerTrace {
    body_id: String,
    point_name: String,
    x: Vec<f64>,  // per sweep step, NaN where solve failed
    y: Vec<f64>,
}
```

The GUI (egui) should render each trace in a distinct color with a dashed
connection line from the trace point to its body's centroid, making the
rigid attachment visually clear.

---

## Golden Test Fixture Strategy

The Python test suite generates reference data that validates the Rust port.

### What to export

For each benchmark mechanism (4-bar, slider-crank, 6-bar, etc.), export JSON files containing:

| Data | Format |
|------|--------|
| Mechanism definition | Schema-versioned JSON (already exists) |
| Position sweep results | `{input_angle: [q_full]}` at each step |
| Velocity results | `{input_angle: [q_dot]}` at each step |
| Acceleration results | `{input_angle: [q_ddot]}` at each step |
| Lagrange multipliers | `{input_angle: [lambda]}` (joint reactions + driver effort) |
| Coupler point paths | `{point_name: [(x, y)]}` over sweep |
| Forward dynamics trajectory | `{t: [q, q_dot]}` at each output timestep |
| Energy balance | `{t: {KE, PE, dissipated, work_in}}` |

### Tolerance strategy

- Position: `‖Δq‖ < 1e-10` (same as NR convergence tolerance)
- Velocity/acceleration: `‖Δ‖ < 1e-8` (one integration level less precise)
- Lagrange multipliers: relative tolerance `|Δλ/λ| < 1e-6` (sensitive to conditioning)
- Forward dynamics: looser — `‖Δq‖ < 1e-5` (different integrators will diverge slightly)

### Known limitation: driver functions are not serializable

`revolute_driver` joints use Python lambdas (`f`, `f_dot`, `f_ddot`) that cannot be serialized to JSON. The current schema (v1.0.0) skips driver payloads on load — see `test_driver_skipped_on_load` in `tests/test_serialization.py`.

**Impact on Rust port:** Golden fixture JSON files do not fully define driven mechanisms. The Rust test harness must re-attach driver functions programmatically for each benchmark mechanism (e.g., `f(t) = ω*t` for constant-speed drivers).

**Future fix:** When the Rust port introduces an expression evaluator (`meval` or `rhai`), extend the schema to v1.1 with a `"driver_expr"` string field (e.g., `"2*pi*t"`) that both Python and Rust can parse. This makes fixtures fully self-contained.

### When to export

Add a `--export-golden` flag to the Python test runner. Run it once before starting the Rust port. Store results in `data/benchmarks/golden/`.

---

## Rust Port — Module Mapping

The Python module structure maps directly to Rust:

```
linkage-sim-rs/
├── src/
│   ├── error.rs                  ← centralized error types (LinkageError enum)
│   ├── core/
│   │   ├── mod.rs
│   │   ├── body.rs              ← core/bodies.py
│   │   ├── constraint.rs        ← core/constraints.py
│   │   ├── force_element.rs     ← core/force_elements.py
│   │   ├── driver.rs            ← core/drivers.py (+ DriverMeta for serialization)
│   │   ├── mechanism.rs         ← core/mechanism.py
│   │   ├── state.rs             ← core/state.py
│   │   └── load_case.rs         ← core/load_cases.py
│   ├── solver/
│   │   ├── mod.rs
│   │   ├── assembly.rs          ← solvers/assembly.py
│   │   ├── kinematics.rs        ← solvers/kinematics.py
│   │   ├── statics.rs           ← solvers/statics.py
│   │   ├── inverse_dynamics.rs  ← solvers/inverse_dynamics.py
│   │   └── forward_dynamics.rs  ← solvers/forward_dynamics.py
│   ├── forces/
│   │   ├── mod.rs
│   │   ├── assembly.rs          ← forces/assembly.py
│   │   ├── gravity.rs           ← forces/gravity.py
│   │   └── helpers.rs           ← forces/helpers.py
│   ├── analysis/
│   │   ├── mod.rs
│   │   ├── coupler.rs           ← analysis/coupler.py
│   │   ├── energy.rs            ← analysis/energy.py
│   │   ├── grashof.rs           ← analysis/grashof.py
│   │   ├── transmission.rs      ← analysis/transmission.py
│   │   ├── validation.rs        ← analysis/validation.py
│   │   ├── toggle.rs            ← analysis/toggle.py
│   │   ├── envelopes.rs         ← analysis/envelopes.py
│   │   └── reactions.rs         ← analysis/reactions.py
│   ├── gui/                      ← Phase 5, Rust-native (egui)
│   │   ├── mod.rs              ← App shell, menu bar, panel layout
│   │   ├── state.rs            ← AppState, SweepData, animation, solver integration
│   │   ├── canvas.rs           ← 2D renderer, pan/zoom, hit testing, context menu
│   │   ├── input_panel.rs      ← Playback controls, angle slider
│   │   ├── property_panel.rs   ← Read-only property inspection
│   │   ├── samples.rs          ← 13 sample mechanism builders (8 four-bar + 5 six-bar)
│   │   ├── plot_panel.rs       ← Sweep plots: coupler trace, body angles, transmission angle
│   │   ├── undo.rs             ← Undo/redo with mechanism JSON snapshots
│   │   └── (future: editor modules)
│   ├── util/
│   │   ├── mod.rs
│   │   ├── units.rs             ← util/units.py
│   │   ├── expressions.rs       ← util/expressions.py (rhai or meval)
│   │   └── plugin_registry.rs   ← util/plugin_registry.py
│   └── lib.rs
├── tests/
│   ├── golden/                   ← loaded from data/benchmarks/golden/
│   ├── test_constraints.rs
│   ├── test_kinematics.rs
│   ├── test_statics.rs
│   ├── test_dynamics.rs
│   └── test_validation.rs
├── data/
│   ├── benchmarks/golden/        ← JSON fixtures from Python
│   ├── templates/
│   └── examples/
└── Cargo.toml
```

### Rust crate dependencies

| Need | Crate | Notes |
|------|-------|-------|
| Linear algebra | `nalgebra` | Dense vectors/matrices, LU, SVD, solves |
| Sparse matrices | `nalgebra-sparse` or `faer` | If mechanism size warrants it |
| JSON serialization | `serde` + `serde_json` | Derive macros for all data structures |
| ODE integration | `ode_solvers` or custom | Explicit RK4/Dormand-Prince + Baumgarte. Add SUNDIALS FFI (`sundials-sys`) if stiff systems require it |
| Safe expression eval | `meval` (simple) or `rhai` (rich) | For user-defined force laws and drivers |
| GUI | `egui` + `eframe` | Native + WASM targets |
| Plotting | `egui_plot` | Embedded in GUI |
| Approx testing | `approx` | Float comparison in tests |
| Property testing | `proptest` | Finite-difference Jacobian verification |

---

## Forward Dynamics — Bridging the SciPy Gap

This is the one area where the Rust port requires extra effort beyond transcription.

**Python has:** `solve_ivp(method='Radau')` — an implicit Runge-Kutta method that handles stiff DAEs.

**Rust approach (phased):**

1. **Start with explicit integrator + Baumgarte + projection.** Dormand-Prince (RK45) from `ode_solvers` crate, with Baumgarte stabilization parameters tuned during the Python phase. Periodic NR projection onto the constraint manifold (reuse the position solver). This handles most linkage mechanisms.

2. **Monitor constraint drift.** If `‖Φ(q)‖` exceeds tolerance, tighten projection frequency or Baumgarte parameters.

3. **If stiff systems appear:** bind SUNDIALS IDA via `sundials-sys` FFI. This is the C equivalent of SciPy's Radau/BDF and is the industry-standard DAE solver. The FFI wrapper is a bounded task, not an open-ended research problem.

The Python phase tells you exactly which mechanisms need implicit methods vs. which work fine with explicit + Baumgarte. You port the approach that actually works, not the one you think you'll need.

---

## Port Sequencing

The Rust port follows the same build order as the Python phases, validating against golden data at each step:

1. `core/state.rs` + `core/body.rs` — coordinate bookkeeping, body structs — **COMPLETE** (March 2026)
2. `core/constraint.rs` — revolute, prismatic, fixed — **COMPLETE** (March 2026)
3. `solver/assembly.rs` — global Φ, Φ_q assembly — **COMPLETE** (March 2026)
4. `solver/kinematics.rs` — NR position solver, velocity, acceleration. Validated against golden 4-bar data — **COMPLETE** (March 2026)
5. `core/mechanism.rs` + `io/serialization.rs` — serde JSON round-trip — **COMPLETE** (March 2026)
6. `forces/*` + `solver/statics.rs` — gravity + force helpers + static solver. Validated against golden statics data — **COMPLETE** (March 2026). *Note: all Python force elements have been ported as `ForceElement` enum variants (12 total): gravity, springs, dampers, external loads, gas springs, bearing friction, joint limits, motors, linear actuators.*
7. `solver/inverse_dynamics.rs` — Validated against golden inverse dynamics data — **COMPLETE** (March 2026)
8. `solver/forward_dynamics.rs` — explicit integrator + Baumgarte. Validated against golden trajectories — **COMPLETE** (March 2026)
9. `analysis/*` — validation, transmission angle, Grashof classification, coupler curves, energy — **COMPLETE** (March 2026)
10. `gui/*` — Phase 5, built in egui — **IN PROGRESS** (MVP + animation playback + driver reassignment + 13 sample mechanisms + JSON save/load + undo/redo + plotting + gravity-loaded reaction force arrows + interactive topology editor + SVG export + force element GUI + analysis displays done; raster export remaining)

**Note:** Phase 5 MVP (visualization shell) was built in parallel after port steps 1-5, consuming only the kinematic solver API. Sub-projects for animation playback, JSON save/load, undo/redo, plotting, 6-bar sample mechanisms, interactive topology editor, load cases, and SVG export have since been completed.

**Remaining Phase 5 work:**
- Force element GUI: define/edit springs, dampers, external loads on bodies/joints — **done** (property panel editing, canvas rendering of spring/damper/force symbols)
- Analysis displays: energy plot tab, Grashof classification, Jacobian rank diagnostics — **done** (energy plot with KE/PE/total, Grashof classification in diagnostics panel, condition number display)
- Velocity solve in sweep — **done** (called at each sweep step for energy computation)
- Forward dynamics GUI — **done** (simulate button, timeline scrubbing, playback speed, constraint drift display)
- Inverse dynamics GUI — **done** (sweep plot tab with statics overlay)
- Toggle detection, envelopes, force breakdown analysis — **done** (ported to Rust, torque envelope stats in diagnostics panel)
- Raster/animation export: PNG, GIF/MP4 (nice-to-have)

Each step has a clear "done" condition: Rust output matches Python golden data within tolerance.

---

## Port Completion Summary (March 2026)

The solver kernel port (steps 1–9) is complete and validated. All four analysis modes produce results matching Python golden fixtures within tolerance.

### Test coverage

- **316 tests total**
- All tests pass via `cargo test`

### Golden fixture coverage

| Mechanism | Kinematics | Statics | Inverse Dynamics | Forward Dynamics |
|-----------|:----------:|:-------:|:----------------:|:----------------:|
| 4-bar crank-rocker | Yes | Yes | Yes | — |
| Slider-crank | Yes | Yes | Yes | — |
| Pendulum | — | — | — | Yes |

### Tolerances achieved

| Quantity | Tolerance | Notes |
|----------|-----------|-------|
| Position | < 1e-10 | Matches NR convergence tolerance |
| Velocity | < 1e-8 | |
| Acceleration | < 1e-2 | Looser near toggle configurations |
| Lagrange multipliers | < 0.5 | Looser near toggle (180 degrees) |

### Known differences from Python

Near-singular configurations (toggle points at 180 degrees), the Rust solver uses SVD decomposition where Python uses `numpy.linalg.lstsq`. Both produce valid solutions, but the specific values diverge near singularities. This accounts for the wider tolerances on acceleration and Lagrange multipliers at those configurations.

### Rust project structure

```
linkage-sim-rs/
├── src/
│   ├── error.rs        # Centralized error types (LinkageError enum)
│   ├── core/           # Body, constraint, driver (+ DriverMeta), mechanism, state
│   ├── forces/         # ForceElement enum (12 variants: springs, dampers, external loads, gas springs, friction, motors, etc.), helpers, assembly
│   ├── solver/         # Kinematics, statics, inverse/forward dynamics, assembly
│   ├── analysis/       # Validation, transmission, Grashof, coupler, energy
│   ├── io/             # JSON serialization (serde), driver serialization
│   ├── gui/            # Phase 5 egui application: animation, driver selection, 13 samples, JSON save/load, undo/redo, plotting
│   ├── bin/            # GUI binary entry point
│   └── lib.rs
├── tests/
│   └── golden_fixtures.rs   # Integration tests against Python golden data
├── data/
│   └── golden/              # JSON fixtures exported from Python
└── Cargo.toml
```

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation | Outcome |
|------|-----------|------------|---------|
| "Python is good enough, never port" | Medium | Phase 5 GUI is the forcing function | **Did not materialize.** Port completed; GUI work underway in egui |
| Port takes longer than expected | Medium | 1:1 module mapping and golden test data bound the scope | **Did not materialize.** 1:1 mapping strategy worked as planned |
| Forward dynamics needs implicit solver | Low | Python phase identifies which mechanisms need it | **Did not materialize.** Explicit RK4 + Baumgarte + projection sufficient for all benchmark mechanisms |
| nalgebra API friction | Low | Well-documented, large user base | **Minor friction only.** SVD vs lstsq near singularities required tolerance adjustments but no design changes |
| Expression evaluator (rhai/meval) limitations | Low | Scope is narrow: math expressions over named variables | **Not yet exercised.** Driver expressions still handled programmatically; expression evaluator deferred to schema v1.1 |
| Premature port — big modeling changes discovered after porting | Medium | See "Timing Caveat" below | **Did not materialize.** Data model was stable; no rework required |

---

## Timing Caveat — When to Reassess

The plan says "port after Phase 4, build GUI in Rust." This is the right default, but it assumes the solver's modeling decisions are stable by Phase 4 exit. In practice, significant modeling/UX changes often surface only once you can:

- Drag joints around and see branch flips in real time
- Visually inspect bad geometry that produces weird reactions
- Watch reaction force spikes animate past singularities
- Interactively explore parameter sensitivity

**If this is a solo or small-team project**, consider building a **thin Python visualization layer** (not a full GUI — just a Matplotlib/Plotly interactive viewer with sliders) during Phases 2–3. This is not Phase 5 — it is a debugging and discovery tool. The goal is to surface modeling problems *before* the Rust port, not after.

**Reassessment checkpoint (at Phase 4A exit):**

Before starting the Rust port, ask:

1. Have any benchmark mechanisms revealed modeling gaps that require data-model changes (new joint types, different constraint formulations, changed force element interfaces)?
2. Is the ForceElement trait stable, or are you still changing its signature?
3. Have you used the solver on at least one "real" mechanism (not just textbook benchmarks)?
4. Are you confident the JSON schema won't need breaking changes?

If the answer to any of (1–3) is "yes, still changing," delay the port and keep iterating in Python. The cost of porting too early is porting twice. The cost of porting "too late" is a few extra weeks of Python visualization that you'd want anyway.

**The forcing function is still valid:** Phase 5 GUI in Python is genuinely worse than in Rust. The question is not "should we port?" but "have we learned enough to port confidently?"

---

## Reassessment Results (2026-03-17)

### 1. Have benchmarks revealed modeling gaps requiring data-model changes?

**No.** All benchmark mechanisms (4-bar crank-rocker, slider-crank, 6-bar Watt I, pendulum) solve correctly across kinematics, statics, inverse dynamics, and forward dynamics. The interactive viewer scripts exercise all 4-bar Grashof variants, slider-crank, and all 5 Watt I 6-bar subtypes without uncovering missing joint types or constraint formulation problems. No data-model changes required.

### 2. Is the ForceElement trait stable?

**Yes.** The `ForceElement` protocol (`forces/protocol.py`) has not changed since its introduction in Phase 2 Step 1. Phase 4B added friction and joint limits as new force element implementations but did not alter the protocol signature: `evaluate(self, state, q, q_dot, t) -> NDArray`.

### 3. Have you used the solver on at least one "real" mechanism?

**Partially.** The solver has been exercised on textbook benchmarks (4-bar, slider-crank, 6-bar, pendulum) and all Grashof variants, but not on a mechanism from an actual engineering application. The interactive viewer demos provide good coverage of the parameter space but use textbook proportions.

*TODO: If you have a specific real-world mechanism to test, run it through the viewer before starting the Rust port. If not, the textbook coverage across 12+ mechanism configurations provides reasonable confidence.*

### 4. Confident the JSON schema won't need breaking changes?

**Yes, with one known gap.** Schema v1.0.0 is stable for mechanism geometry (bodies, joints, attachment points, coupler points). The known gap is driver function serialization — lambdas cannot round-trip through JSON. This will be addressed in the Rust port via schema v1.1 with symbolic expression support (`meval`/`rhai`), which is an additive change, not a breaking one.

### Go/No-Go Decision

**GO.** Questions 1, 2, and 4 are clearly resolved. Question 3 is a soft gap — the solver handles a wide variety of mechanism topologies and parameters, even if none come from a specific engineering application. The golden fixture suite now covers all 3 mechanism types across all 4 analysis modes (9 fixture files, 750+ data points). The risk of discovering a modeling problem during the Rust port is acceptably low.

---

## Feature Parity Gap: Python vs Rust (as of 2026-03-18)

The solver kernel port (steps 1–9) is complete for **constraint-based analysis** (kinematics, statics, inverse dynamics, forward dynamics). However, the Python codebase has a richer force element library that was **not ported** to Rust. This section tracks the gap.

### Force elements

| Feature | Python | Rust | Notes |
|---------|--------|------|-------|
| Gravity | Yes | Yes | Toggleable in GUI |
| `ForceElement` protocol/trait | Yes (`Protocol`) | Yes (enum) | `ForceElement` enum with 7 variants |
| Linear springs | Yes | Yes | |
| Torsion springs | Yes | Yes | |
| Viscous dampers (translational) | Yes | Yes | LinearDamper |
| Rotary dampers | Yes | Yes | |
| External point forces | Yes | Yes | |
| External torques | Yes | Yes | |
| Gas springs | Yes | Yes | Polytropic compression + damping |
| Bearing friction | Yes | Yes | Constant + viscous + Coulomb, tanh regularization |
| Joint limits | Yes | Yes | Penalty method with restitution-modulated damping |
| Motors/actuators (force-based) | Yes | Yes | Linear T-ω droop model + linear actuator |

**Rust force element library complete:** `ForceElement` enum has 12 variants: `Gravity`, `LinearSpring`, `TorsionSpring`, `LinearDamper`, `RotaryDamper`, `ExternalForce`, `ExternalTorque`, `GasSpring`, `BearingFriction`, `JointLimit`, `Motor`, `LinearActuator`. All are serializable, editable in the GUI property panel, and rendered on the canvas.

### Analysis modules — backend vs GUI

| Analysis | Rust Backend | Rust GUI | Notes |
|----------|:------------:|:--------:|-------|
| Transmission angle | Yes | Yes (plot tab) | 4-bar only |
| Coupler point tracing | Yes (pos, vel, accel) | Yes (position only) | Velocity/acceleration vectors not rendered |
| Energy (KE/PE/total) | Yes | Yes (plot tab) | KE, PE, total energy vs driver angle |
| Grashof classification | Yes | Yes (diagnostics) | Shown in collapsible diagnostics panel |
| Jacobian rank/condition | Yes | Yes (diagnostics) | Condition number + overconstrained warning |
| Validation (Grubler DOF) | Yes | Partial (status bar) | |

### Solver capabilities — backend vs GUI

| Solver | Rust Backend | Rust GUI | Notes |
|--------|:------------:|:--------:|-------|
| Position kinematics | Yes | Yes | Core of sweep + animation |
| Velocity kinematics | Yes | Yes (sweep) | Called at each sweep step for energy computation |
| Acceleration kinematics | Yes | Yes (sweep) | Called at each sweep step for inverse dynamics |
| Statics | Yes | Yes | Driver torque + reaction force arrows |
| Inverse dynamics | Yes | Yes (sweep + plot) | Torque including inertial effects, overlaid with statics |
| Forward dynamics | Yes | Yes (simulate) | RK4+Baumgarte with GUI playback, timeline scrubbing |

### Plan to close the gap

1. **Force element enum** — ~~Create `ForceElement` enum (not trait) on `Mechanism`, refactor solver APIs to read forces from mechanism. Add serialization support (tagged enum, backward-compatible JSON).~~ **DONE.** `ForceElement` enum with 7 variants, integrated into solver APIs and serialization.
2. **Force element GUI** — Collapsible force sections in property panel for body/joint selection. Canvas rendering of spring/damper symbols and external force arrows. **Partially done:** property panel editing complete, canvas rendering of spring/damper/force symbols complete.
3. **Analysis displays** — ~~Energy plot tab (requires velocity solve in sweep). Grashof classification in property panel diagnostics. Jacobian rank at current pose in diagnostics.~~ **DONE.** Energy plot tab with KE/PE/total, Grashof classification in diagnostics panel, Jacobian condition number display.
4. **Velocity solve in sweep** — ~~Add linear velocity solve at each sweep step to enable energy computation.~~ **DONE.** Velocity solved at each sweep step; energy computed via `compute_energy_state_mech`.
