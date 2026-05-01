# Planar Linkage Mechanism Simulator

A general-purpose planar multibody simulator for engineering analysis of linkage mechanisms. Handles arbitrary planar mechanisms — binary bars, ternary plates, slider blocks, bell-cranks — using a body–constraint formulation with constraint-first mathematics.

---

## What This Tool Does

Given a mechanism defined as rigid bodies connected by joints, this tool computes:

- **Kinematics**: position, velocity, and acceleration of every body and coupler point across the full range of motion
- **Static forces**: required input torque and all joint reaction forces at each configuration (all 12 force element types supported in both Python and Rust: gravity, springs, dampers, external loads, gas springs, bearing friction, joint limits, motors, linear actuators)
- **Inverse dynamics**: required actuator effort for a prescribed motion profile, including inertial loads
- **Forward dynamics**: time-domain simulation of mechanism response to applied forces
- **Crank selection analysis** for supported four-bar mechanisms — Grashof-based classification, driver ranking, and numerical range estimation

### Force element status

| Force Type | Python | Rust |
|---|---|---|
| Gravity | Yes | Yes (GUI slider, 0-100g) |
| Linear springs | Yes | Yes |
| Torsion springs | Yes | Yes |
| Viscous dampers (translational) | Yes | Yes |
| Rotary dampers | Yes | Yes |
| External point forces | Yes | Yes |
| External torques | Yes | Yes |
| Gas springs | Yes | Yes |
| Bearing friction | Yes | Yes |
| Joint limits | Yes | Yes |
| Motors (force-based) | Yes | Yes |
| Linear actuators | Yes | Yes |

The Rust solver has a `ForceElement` enum with 12 variants (`Gravity`, `LinearSpring`, `TorsionSpring`, `LinearDamper`, `RotaryDamper`, `ExternalForce`, `ExternalTorque`, `GasSpring`, `BearingFriction`, `JointLimit`, `Motor`, `LinearActuator`) covering all Python force elements. All are editable in the GUI property panel and rendered on the canvas.

The target user is a mechanical engineer sizing actuators, selecting bearings, checking transmission angles, and validating linkage geometry — not an academic researcher building a general-purpose multibody dynamics code.

---

## Architecture Summary

The simulator is built on four foundational decisions documented in detail in `docs/`:

1. **Body–constraint incidence model** — not a joint-node / link-edge graph. Bodies are first-class rigid objects with multiple attachment points. Joints are constraints between bodies. This handles ternary links, bell-cranks, and slider blocks without special cases. → `docs/architecture/ARCHITECTURE.md`

2. **Constraint-first mathematics** — all analysis modes share one backbone: generalized coordinates `q`, constraint equations `Φ(q,t) = 0`, and the constraint Jacobian `Φ_q`. Kinematics, statics, and dynamics are layers on this foundation. → `docs/architecture/NUMERICAL_FORMULATION.md`

3. **SI internally, engineering units in GUI** — solvers compute in m, kg, s, N, N·m, kg·m². The GUI converts to mm, N·mm, degrees at the display boundary. No conversion factors inside any solver.

4. **Math before GUI** — the solver kernel, test suite, and JSON workflow are built and validated before any interactive editor.

---

## Documentation

### Architecture (how the system works)

| Document | Contents |
|---|---|
| `docs/architecture/ARCHITECTURE.md` | Core data model: Body, JointConstraint, ForceElement, Mechanism. Topology, units, serialization, schema versioning |
| `docs/architecture/NUMERICAL_FORMULATION.md` | Generalized coordinates, constraint equations, Jacobian, driver treatment, force assembly, Lagrange multipliers, singularity analysis |
| `docs/architecture/ANALYSIS_MODES.md` | Kinematic, static, inverse dynamic, and forward dynamic analysis. Solver methods, inputs, outputs |
| `docs/architecture/ENGINEERING_OUTPUTS.md` | Joint reactions, input torque, mechanical advantage, transmission angle, coupler curves, result envelopes |
| `docs/architecture/VALIDATION.md` | Mechanism validation layers, constraint diagnostics, benchmark suite |
| `docs/architecture/EXTENSIBILITY.md` | Extension points for ForceElement, JointConstraint, and switching forces |

### Guides (how to use it)

| Document | Contents |
|---|---|
| `docs/guides/SAMPLES.md` | Built-in sample mechanisms and their properties |
| `docs/guides/SHORTCUTS.md` | Keyboard and mouse shortcuts reference |
| `docs/guides/PARAMETRIC_STUDIES.md` | How to run parametric sweeps and compare results |
| `docs/guides/TRAJECTORY_MODE.md` | Inverse position-control walkthrough: prescribe an output observable h(t), back-solve the actuator command u(t) |
| `docs/guides/WASM_DEPLOYMENT.md` | Building and deploying the web (WASM) version |

### Reference

| Document | Contents |
|---|---|
| `docs/reference/FORCE_ELEMENTS.md` | Equations and parameters for all 12 force element types |
| `linkage-sim-rs/SYSTEM.md` | File map of all 107 Rust source files |

### History

| Document | Contents |
|---|---|
| `docs/history/ROADMAP.md` | Original development phases and deliverables |
| `docs/history/RUST_MIGRATION.md` | Python→Rust port strategy and completion summary |

---

## Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Core solver (Phases 1–4) | Python + NumPy/SciPy | `fsolve` for constraints, `linalg` for linear systems, `solve_ivp` (Radau/BDF) for DAE |
| Core solver (production) | Rust + nalgebra | **Port complete** — validated against Python golden fixtures (644 tests). All 12 force element types ported. See `docs/history/RUST_MIGRATION.md` |
| Expression evaluator | Python: `asteval` / Rust: `meval` | **Shipped.** User-defined driver expressions (e.g., `"pi/2 * sin(3*t)"`) with GUI editor, serializable to JSON |
| GUI framework (Phase 5) | Rust: `egui` + `eframe` | 2D canvas, drag-and-drop, animation. Native + WebAssembly targets. WASM build infrastructure shipped (feature flags, web entry point) |
| Plotting (development) | Matplotlib or Plotly | Engineering-quality plots during Python development |
| Plotting (production) | `egui_plot` | Embedded in Rust GUI |
| Data persistence | JSON with schema versioning | Human-readable, diffable, version-controllable mechanism definitions. Python: `json`. Rust: `serde` |
| Unit conversion | Thin boundary layer | SI ↔ display conversion at GUI input/output only |

---

## File Structure

```
linkage-sim/
├── core/
│   ├── bodies.py              # Body, PointMass, ground body
│   ├── constraints.py         # JointConstraint base + revolute/prismatic/fixed
│   ├── force_elements.py      # ForceElement base + all built-in types
│   ├── drivers.py             # Driver types (constraint-based)
│   ├── mechanism.py           # Mechanism assembly, serialization, schema versioning
│   ├── state.py               # Generalized coordinate vector q, bookkeeping
│   └── load_cases.py          # Study/scenario manager
├── solvers/
│   ├── assembly.py            # Global Φ, Φ_q, M, Q assembly from mechanism
│   ├── kinematics.py          # Position (NR), velocity, acceleration solvers
│   ├── statics.py             # Static equilibrium solver
│   ├── inverse_dynamics.py    # Inverse dynamics solver
│   └── forward_dynamics.py    # DAE integration, constraint stabilization
├── analysis/
│   ├── validation.py          # Grübler, Jacobian rank, connectivity, Grashof
│   ├── transmission.py        # Transmission angle, pressure angle, MA
│   ├── toggle.py              # Toggle/dead-point detection via Jacobian
│   ├── energy.py              # Energy balance tracking
│   ├── envelopes.py           # Peak/RMS/min-max result extraction
│   ├── reactions.py           # Joint reaction post-processing (global, local, radial/tangential)
│   ├── synthesis.py           # Linkage synthesis (Phase 6)
│   └── optimization.py        # Parametric optimization (Phase 6)
├── gui/                        # Phase 5
│   ├── canvas.py              # 2D topology editor
│   ├── property_panel.py      # Auto-generated from element schemas
│   ├── animation.py           # Real-time mechanism animation
│   └── plot_panel.py          # Embedded plot windows
├── util/
│   ├── units.py               # SI ↔ display unit conversion
│   ├── expressions.py         # Safe math expression parser/evaluator
│   └── plugin_registry.py     # Named plugin registration for custom force laws
├── data/
│   ├── templates/             # Common mechanism templates (JSON)
│   ├── examples/              # Example mechanism files
│   └── benchmarks/            # Versioned validation cases with expected results
├── tests/
│   ├── test_constraints.py    # Constraint equations and Jacobians vs. analytical
│   ├── test_kinematics.py     # Solver vs. benchmark mechanisms
│   ├── test_statics.py        # Solver vs. hand calculations
│   ├── test_dynamics.py       # Forward dynamics vs. analytical solutions
│   ├── test_validation.py     # Grübler, rank, connectivity
│   ├── test_reactions.py      # Reaction force post-processing
│   └── test_units.py          # Unit conversion round-trips
├── docs/                       # Architecture & design documentation
└── README.md                   # This file
```

### Rust solver kernel (`linkage-sim-rs/`)

The full solver port (Phases 1-4: kinematics, statics, inverse dynamics, forward dynamics) is complete in Rust, validated against Python golden fixtures (644 tests). Phase 5 GUI built with egui/eframe. See [`docs/FEATURES.md`](docs/FEATURES.md) for the complete feature list and roadmap.

**What's shipped in the GUI:**

- **30 sample mechanisms** (11 four-bar + 7 six-bar + 12 specialty), visual sample gallery with category headers and tooltip descriptions
- **My Samples**: promote user mechanisms to the Samples dropdown
- **Full interactive editor**: create bodies, joints, and ground pivots via right-click context menu; drag ground pivots to reposition; Draw Link tool with body-aware snapping; multi-point body creation; prismatic and fixed joint creation
- **12 force element types** all editable in the property panel and rendered on the canvas, with categorized toolbar ribbon (Joint Torques / Link Forces dropdowns)
- **10 plot tabs**: coupler trace, body angles, transmission angle, driver torque, inverse dynamics, energy (KE/PE/total), mechanical advantage, joint reactions, coupler velocity, coupler acceleration
- **Actuator sizing**: actuator force plot (statics + inverse dynamics curves), stroke display in Health Report, force zone overlap diagnostics
- **Trapezoidal motion profile** with profile torque overlay for realistic acceleration/deceleration analysis
- **Parametric sweep**: full 360-degree driver rotation sweeps, parameter studies (mass, inertia, attachment points, force parameters, driver speed), save and overlay named results for side-by-side comparison
- **Forward dynamics simulation** with timeline scrubbing, playback speed control, and constraint drift display
- **Mechanism Health Report**: green/yellow/red indicators for Grashof classification, toggles, transmission angle, peak torque, peak reactions, Jacobian conditioning, convergence
- **Mounting angle**: per-mechanism gravity rotation with UI slider and canvas visualization
- **Load path visualization**: color-coded links by joint reaction force magnitude
- **Share via URL**: compress mechanism JSON (deflate + base64url), shareable link preserves crank angle
- **Image trace overlay**: import background PNG/JPEG with adjustable opacity, scale, and offset
- **Export**: PNG, SVG, GIF (ping-pong loop), DXF, CSV, HTML report with interactive Plotly charts
- **Welcome screen** with quick-start buttons; interactive tutorial (Help > Tutorial: Build a 4-Bar); demo mode auto-cycling all samples
- **Multi-select** (Shift+click), arrow-key nudge, Scale Mechanism tool, undo/redo with visual history panel
- **Autosave** (every 30s) with recovery prompt on startup; recent files menu
- **Professional dark theme** with CAD-convention canvas (major/minor grid, origin crosshair, zoom-adaptive spacing, alignment guides)
- **WebAssembly build** — all analysis, editing, and plotting features work in the browser

**Run natively:** `cd linkage-sim-rs && cargo run --bin linkage-gui`

**Run in browser (WASM):**
```bash
cd linkage-sim-rs
./scripts/build_web.sh          # Build WASM + JS bindings
./scripts/serve_web.sh          # Serve at http://localhost:8080
```
Requires: `rustup target add wasm32-unknown-unknown` and `cargo install wasm-bindgen-cli`.
Note: file dialogs, PNG/SVG/GIF/DXF export, autosave, and HTML reports are native-only. All analysis, editing, and plotting features work in the browser.

**Live deployment:** Pushes to `main` auto-deploy to [linkage.colesorkness.com](https://linkage.colesorkness.com) via GitHub Actions + Vercel. See `.github/workflows/deploy-web.yml` for the CI pipeline.

```
linkage-sim-rs/
├── src/
│   ├── core/               # body, constraint, driver, mechanism, state, linear_driver
│   ├── forces/             # elements (12 variants), gravity, helpers, assembly
│   ├── solver/             # kinematics, statics, inverse_dynamics, forward_dynamics, assembly, events, condition
│   ├── analysis/           # coupler, energy, envelopes, force_breakdown, grashof,
│   │                       #   crank_selection, motor_sizing, transmission, validation, virtual_work
│   ├── io/                 # serialization (serde JSON round-trip)
│   ├── gui/                # mod, state/, canvas/, property_panel/, samples/, export/,
│   │                       #   input_panel, plot_panel, parametric_panel, sweep,
│   │                       #   force_toolbar, tutorial, undo, error_panel
│   ├── bin/                # linkage_gui
│   └── lib.rs
├── tests/
│   ├── golden_fixtures.rs           # Integration tests against Python golden data
│   ├── property_tests.rs            # Proptest: random mechanism generation, invariant checks
│   ├── singular_behavior.rs         # Near-singularity tolerance tests
│   ├── compound_force_integration.rs # Multi-force-element compound scenarios
│   ├── force_zone_tests.rs          # Spatial force zone application and overlap
│   ├── geometry_tests.rs            # Body geometry and attachment point math
│   ├── mount_point_integration.rs   # Named mount point resolution
│   └── parallelogram_actuator_sample.rs # Actuator-driven sample validation
├── data/
│   └── golden/             # JSON fixtures exported from Python
└── Cargo.toml
```

---

## Design Principles

These are invariants. If any code violates them, it is a bug.

1. **Bodies are the truth, constraints connect them.** No special-case code for 4-bar vs. 6-bar vs. slider-crank. A ternary plate is just a body with three attachment points.

2. **Constraints are the mathematical foundation.** `Φ(q, t) = 0` and `Φ_q` are the backbone of every analysis mode from day one.

3. **Force elements are pluggable.** Adding a new smooth force element means implementing one `evaluate` method that returns a generalized force contribution. The solvers never change. Both Python (`ForceElement` protocol) and Rust (`ForceElement` enum with 12 variants) support the full force element library at parity.

4. **Smooth elements: no solver changes. Switching elements: solver changes expected.** Cables, clutches, and stick-slip friction change the mathematical class of the problem. This is acknowledged and planned for.

5. **Validate early, validate honestly.** Grübler is a sanity check, not a guarantee. Jacobian rank is the real mobility test. Redundant constraints are warned, not silently accepted.

6. **SI internally, engineering units at the boundary.** No conversion factors inside any solver. Ever.

7. **Math before GUI.** Solver kernel and test suite first. Interactive editor last.

8. **One canonical internal load representation.** All force elements return contributions to the generalized force vector `Q`. Helper utilities convert point forces, body torques, and gravity into `Q` contributions. The solver sees only `Q`.

9. **Drivers are constraints.** A motion driver adds rows to `Φ`. The associated Lagrange multiplier is the required actuator effort. This is not a force element — it is a prescribed-motion constraint.
