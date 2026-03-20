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
| Gravity | Yes | Yes (GUI toggle) |
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

1. **Body–constraint incidence model** — not a joint-node / link-edge graph. Bodies are first-class rigid objects with multiple attachment points. Joints are constraints between bodies. This handles ternary links, bell-cranks, and slider blocks without special cases. → `docs/ARCHITECTURE.md`

2. **Constraint-first mathematics** — all analysis modes share one backbone: generalized coordinates `q`, constraint equations `Φ(q,t) = 0`, and the constraint Jacobian `Φ_q`. Kinematics, statics, and dynamics are layers on this foundation. → `docs/NUMERICAL_FORMULATION.md`

3. **SI internally, engineering units in GUI** — solvers compute in m, kg, s, N, N·m, kg·m². The GUI converts to mm, N·mm, degrees at the display boundary. No conversion factors inside any solver.

4. **Math before GUI** — the solver kernel, test suite, and JSON workflow are built and validated before any interactive editor.

---

## Documentation

| Document | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | Core data model: Body, JointConstraint, ForceElement, PointMass, Mechanism. Topology design, unit system, serialization, schema versioning. Friction model tiers. Coordinate representation abstraction for future reduced-coordinate support |
| `docs/NUMERICAL_FORMULATION.md` | Generalized coordinate layout, constraint equations by joint type, prismatic joint conventions with 4 worked examples, driver treatment, force assembly into Q, Lagrange multiplier extraction, singularity analysis framework (existence/uniqueness/conditioning), reaction force post-processing. Branch management and assembly mode tracking. Dimensionless scaling strategy |
| `docs/ANALYSIS_MODES.md` | Kinematic, static, inverse dynamic, and forward dynamic analysis. Solver methods, inputs, outputs, branch management integration, singularity reporting, and known numerical challenges for each mode |
| `docs/ENGINEERING_OUTPUTS.md` | What the tool produces: joint reactions, input torque, mechanical advantage, transmission angle, coupler curves, result envelopes, load-path decomposition by source/body/joint, mechanism health panel, output coordinate frames, load case management |
| `docs/VALIDATION.md` | Mechanism validation layers (topology, constraint rank, assembly). Constraint and loop diagnostics for actionable error reporting. Versioned benchmark suite with expected results including robustness/failure cases |
| `docs/EXTENSIBILITY.md` | Three extension points (ForceElement, JointConstraint, conditional/switching). Coordinate representation extensions. Future components catalog. Design rules |
| `docs/ROADMAP.md` | Development phases with deliverables and exit criteria (forward dynamics split into 4A smooth / 4B nonsmooth). Phase 6 prerequisites. Recommended build order for fastest path to useful tool |
| `RUST_MIGRATION.md` | Python-first, Rust-second strategy. Port plan, golden test fixture strategy, module mapping, Rust crate dependencies, coding conventions for portability. Port completion summary and risk outcomes |

---

## Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Core solver (Phases 1–4) | Python + NumPy/SciPy | `fsolve` for constraints, `linalg` for linear systems, `solve_ivp` (Radau/BDF) for DAE |
| Core solver (production) | Rust + nalgebra | **Port complete** — validated against Python golden fixtures (411 tests). All 12 force element types ported. See `RUST_MIGRATION.md` |
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
├── README.md                   # This file
└── DESIGN_PRINCIPLES.md        # Short reference card of invariants
```

### Rust solver kernel (`linkage-sim-rs/`)

The full solver port (Phases 1–4: kinematics, statics, inverse dynamics, forward dynamics) is complete in Rust, validated against Python golden fixtures (411 tests). **Phase 5 GUI:** Built with egui/eframe. Features:
- 12 force element types, all editable in the property panel and rendered on the canvas
- 10 plot tabs: coupler trace, body angles, transmission angle, driver torque, inverse dynamics, energy (KE/PE/total), mechanical advantage, joint reactions, coupler velocity, coupler acceleration
- Forward dynamics simulation with timeline scrubbing, playback speed control, and constraint drift display
- PNG + SVG export (resvg-based rasterization, 1920x1080 default)
- Diagnostics panel: Grashof classification, Jacobian conditioning, crank selection, motor sizing, torque envelopes
- 13 sample mechanisms (8 four-bar + 5 six-bar), JSON save/load, undo/redo
- Animation playback with seamless 360-degree wrap (solver initial guess resets to cached angle-0 solution on wrap-around, preventing assembly-branch jumps)
- Right-click driver reassignment on any grounded revolute joint
- Gravity-loaded reaction force arrows at every joint; gravity direction indicator on canvas; both toggleable via View menu

Run with `cd linkage-sim-rs && cargo run --bin linkage-gui`.

**Interactive editor (shipped):** The GUI is now a full interactive editor, not just a visualization shell. Capabilities:
- Create bodies, joints, and ground pivots via right-click context menu on the canvas
- Drag attachment points (on ground and moving bodies) to reshape mechanism geometry in real-time, with immediate solver rebuild
- Edit mass properties (mass, Izz) via DragValue widgets in the property panel
- Delete bodies and joints with cascading cleanup of dependent elements; Delete via keyboard, Set Driver via context menu/property panel
- Two-click joint creation workflow with visual feedback (green ring highlight, hint text)
- Live validation warnings in status bar: DOF mismatch, disconnected bodies, missing driver
- MechanismBlueprint (MechanismJson) is the editable source of truth; rebuild() runs on every edit
- All edit operations push to the undo/redo stack; blueprint stays in sync with snapshots
- Multi-point body creation via + Body tool: click to place points, double-click or Enter to finish
- Add Pivot Here context menu: right-click a body edge to add attachment points, promoting binary bars to ternary plates
- Body-aware Draw Link: snapping to body segments auto-creates pivots for branching connections
- Create Joint two-click flow: right-click attachment point → Create Joint → click second point
- Modes-only toolbar: [Select] [Draw Link] [+ Body] [+ Ground]
- Closed polygon rendering for 3+ point bodies (ternary plates render as triangles)

**Phase 5 complete.** All ROADMAP deliverables shipped:
- Force element GUI — **done** (property panel editing + canvas rendering for all 12 element types)
- Analysis displays — **done** (energy, Grashof, Jacobian conditioning, mechanical advantage, torque envelopes, crank selection, motor sizing)
- Forward dynamics simulation — **done** (simulate button, timeline scrubbing, playback, constraint drift)
- Inverse dynamics — **done** (sweep plot tab with statics overlay)
- PNG + SVG export — **done** (via resvg SVG-to-PNG rasterization, 1920x1080 default)
- GIF animation export — **done** (renders sweep frames via resvg, encodes with gif crate, 800x600 @ 20fps)
- CSV export — **done** (sweep data + coupler traces with comprehensive columns)
- Link dimension annotations — **done** (toggleable via View > Link Dimensions, shows lengths in display units)
- Point mass GUI — **done** (add/remove point masses on bodies with parallel axis theorem recomputation)
- Keyboard shortcuts help — **done** (Help > Keyboard Shortcuts dialog)
- Autosave — **done** (periodic save every 30s to sibling `.autosave.json` file)
- WebAssembly build — **done** (feature flags, WASM entry point, zero-warning compilation)
- Recent files menu — **done** (tracks last 5 opened/saved files, persisted across sessions)
- Canvas hover tooltips — **done** (body/joint info on hover without clicking)
- Mechanism summary — **done** (body count, joint count, total mass always visible in property panel)
- Ctrl+S / Ctrl+Shift+S save shortcuts — **done** (quick save to last path, or Save As)
- Autosave recovery — **done** (prompt on startup when autosave file found from previous session)
- Link lengths in property panel — **done** (segment lengths shown with attachment point list)
- Prismatic + Fixed joint creation — **done** (Create Joint submenu: Revolute / Prismatic / Fixed)
- Ctrl+N new mechanism — **done** (File > New resets to empty canvas, preserves preferences)

```
linkage-sim-rs/
├── src/
│   ├── core/               # body, constraint, driver, mechanism, state
│   ├── forces/             # elements (12 variants), gravity, helpers, assembly
│   ├── solver/             # kinematics, statics, inverse_dynamics, forward_dynamics, assembly, events
│   ├── analysis/           # coupler, energy, envelopes, force_breakdown, grashof,
│   │                       #   crank_selection, motor_sizing, transmission, validation, virtual_work
│   ├── io/                 # serialization (serde JSON round-trip)
│   ├── gui/                # mod, state, canvas, input_panel, property_panel, plot_panel,
│   │                       #   samples, undo, export
│   ├── bin/                # linkage_gui
│   └── lib.rs
├── tests/
│   ├── golden_fixtures.rs   # Integration tests against Python golden data
│   ├── property_tests.rs    # Proptest: random mechanism generation, invariant checks
│   └── singular_behavior.rs # Near-singularity tolerance tests
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
