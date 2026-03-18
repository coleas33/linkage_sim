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

| Phase | Language | Status at cutover |
|-------|----------|-------------------|
| Phase 1 — Data model & kinematics | Python | Complete, validated |
| Phase 2 — Force elements & statics | Python | Complete, validated |
| Phase 3 — Actuators & inverse dynamics | Python | Complete, validated |
| Phase 4 — Forward dynamics | Python | Complete, validated |
| **Rust port** | **Rust** | **Port Phases 1–4, validate against Python golden data** |
| Phase 5 — Interactive GUI | Rust | Built natively in Rust (egui) |
| Phase 6 — Advanced & QoL | Rust | Built in Rust |

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
│   ├── core/
│   │   ├── mod.rs
│   │   ├── body.rs              ← core/bodies.py
│   │   ├── constraint.rs        ← core/constraints.py
│   │   ├── force_element.rs     ← core/force_elements.py
│   │   ├── driver.rs            ← core/drivers.py
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
│   ├── analysis/
│   │   ├── mod.rs
│   │   ├── validation.rs        ← analysis/validation.py
│   │   ├── transmission.rs      ← analysis/transmission.py
│   │   ├── toggle.rs            ← analysis/toggle.py
│   │   ├── energy.rs            ← analysis/energy.py
│   │   ├── envelopes.rs         ← analysis/envelopes.py
│   │   └── reactions.rs         ← analysis/reactions.py
│   ├── gui/                      ← Phase 5, Rust-native (egui)
│   │   ├── mod.rs
│   │   ├── canvas.rs
│   │   ├── property_panel.rs
│   │   ├── animation.rs
│   │   └── plot_panel.rs
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

1. `core/state.rs` + `core/body.rs` — coordinate bookkeeping, body structs
2. `core/constraint.rs` — revolute, then prismatic, then fixed. Validate Jacobians via finite difference (`proptest`)
3. `solver/assembly.rs` — global Φ, Φ_q assembly
4. `solver/kinematics.rs` — NR position solver, velocity, acceleration. **Validate against golden 4-bar data**
5. `core/mechanism.rs` — serde JSON round-trip. Load Python's benchmark JSON files directly
6. `core/force_element.rs` + `solver/statics.rs` — force assembly, static solver. **Validate against golden statics data**
7. `solver/inverse_dynamics.rs` — **Validate against golden inverse dynamics data**
8. `solver/forward_dynamics.rs` — explicit integrator + Baumgarte. **Validate against golden trajectories** (looser tolerance)
9. `analysis/*` — validation, transmission angle, toggle detection, envelopes
10. `gui/*` — Phase 5, built fresh in egui

Each step has a clear "done" condition: Rust output matches Python golden data within tolerance.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| "Python is good enough, never port" | Medium | Phase 5 GUI is the forcing function — Python GUI options are bad enough to motivate the switch |
| Port takes longer than expected | Medium | 1:1 module mapping and golden test data bound the scope. No design decisions during port — only transcription |
| Forward dynamics needs implicit solver | Low for typical linkages | Python phase identifies which mechanisms need it. SUNDIALS FFI is a known, bounded task |
| nalgebra API friction | Low | Well-documented, large user base. Dense linear algebra coverage is solid |
| Expression evaluator (rhai/meval) limitations | Low | Scope is narrow: math expressions over named variables. Both crates handle this |
| Premature port — big modeling changes discovered after porting | Medium | See "Timing Caveat" below |

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
