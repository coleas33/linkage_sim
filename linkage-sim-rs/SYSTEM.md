# Linkage Simulation — Rust Codebase System Guide

**Date**: 2026-03-24
**Schema Version**: 1.1.0
**Total**: 92 Rust source files, 33,103 lines

This document maps the system for AI-assisted development. Every module, file, and data flow is documented so an agent can navigate, modify, and refactor with confidence.

---

## Module Tree

```
linkage-sim-rs/src/
├── lib.rs                              (8)    Crate root — declares 8 modules
├── error.rs                           (59)    LinkageError enum
├── geometry/mod.rs                   (194)    Polygon clip, area, centroid, body_rect_to_world
│
├── core/                           (2,908 total)
│   ├── mod.rs                          (5)    Module declarations
│   ├── state.rs                      (388)    State, BodyIndex, coordinate mapping, pose helpers
│   ├── body.rs                       (446)    Body, BodyGeometry, make_bar, make_ground
│   ├── constraint/                 (1,654)    Joint constraint system
│   │   ├── mod.rs                    (793)    JointConstraint enum, delegating Constraint impl, tests
│   │   ├── trait_def.rs               (25)    Constraint trait definition
│   │   ├── helpers.rs                 (71)    translational_jacobian_block, translational_gamma
│   │   ├── revolute.rs              (112)    RevoluteJoint + make_revolute_joint
│   │   ├── fixed.rs                 (138)    FixedJoint + make_fixed_joint
│   │   ├── prismatic.rs             (208)    PrismaticJoint + make_prismatic_joint
│   │   └── cam.rs                   (307)    CamProfile, CamFollowerJoint, spline helpers
│   ├── driver.rs                     (418)    DriverConstraint (closure-based), DriverMeta
│   ├── linear_driver.rs              (441)    LinearDriver constraint: prescribed distance between two points
│   └── mechanism.rs                  (860)    Mechanism: builder, body/joint/driver/force storage
│
├── forces/                         (3,175 total)
│   ├── mod.rs                          (3)    Module declarations
│   ├── helpers.rs                    (141)    point_force_to_q, body_torque_to_q (virtual work)
│   ├── elements/                   (2,473)    Force element system
│   │   ├── mod.rs                  (1,325)    Re-exports, tests (1,300+ lines of tests)
│   │   ├── time_modulation.rs        (134)    TimeModulation enum, factor(), compile()
│   │   ├── element_types.rs          (310)    13 element data structs + serde helpers (LinearActuator has stroke limits)
│   │   ├── evaluation.rs             (550)    All evaluate_* functions, angular helpers, stroke limit penalty, force_zone_overlap_ratio
│   │   └── force_element.rs          (233)    ForceElement enum, dispatch impl
│   └── compound.rs                   (558)    Compound force expansion (mount point → bodies)
│
├── solver/                         (2,969 total)
│   ├── mod.rs                          (6)    Module declarations
│   ├── assembly.rs                   (149)    Jacobian, constraint, mass matrix assembly
│   ├── kinematics.rs                 (266)    Newton-Raphson position, velocity, acceleration
│   ├── statics.rs                    (357)    Static equilibrium via Lagrange multipliers
│   ├── inverse_dynamics.rs           (308)    Full Newton-Euler inverse dynamics
│   ├── forward_dynamics.rs         (1,379)    BDF-2 DAE integrator, adaptive time-stepping
│   └── events.rs                     (504)    Joint limit / force zone event detection
│
├── analysis/                       (2,885 total)
│   ├── mod.rs                         (10)    Module declarations
│   ├── validation.rs                 (758)    Toggle detection, DOF analysis, connectivity checks
│   ├── transmission.rs               (313)    Transmission angle, mechanical advantage
│   ├── crank_selection.rs            (305)    Which link to drive recommendation
│   ├── energy.rs                     (279)    Kinetic/potential energy computation
│   ├── virtual_work.rs               (280)    Energy-based statics validation
│   ├── motor_sizing.rs               (202)    Motor sizing validation
│   ├── force_breakdown.rs            (196)    Per-element force contribution
│   ├── grashof.rs                    (194)    Grashof classification
│   ├── coupler.rs                    (179)    Coupler point evaluation
│   └── envelopes.rs                  (169)    Min/max/RMS of sweep series
│
├── io/                             (1,598 total)
│   ├── mod.rs                         (14)    Module declarations + re-exports
│   ├── schema.rs                     (220)    MechanismJson, BodyJson, JointJson, DriverJson, LinearDriverJson, SweepConfig
│   ├── to_json.rs                    (211)    mechanism_to_json, body_to_json, joint_to_json, save
│   ├── from_json.rs                  (296)    load_mechanism_unbuilt, load_mechanism_unbuilt_from_json
│   ├── error.rs                       (28)    SerializationError enum
│   └── serialization.rs              (857)    Tests only (round-trip, schema version, edge cases)
│
├── gui/                           (15,241 total)
│   ├── mod.rs                        (971)    LinkageApp, eframe::App::update, menu bar, toolbar
│   ├── state/                      (4,823)    Application state system
│   │   ├── mod.rs                    (808)    AppState struct (44 fields), Default, core methods
│   │   ├── display_units.rs           (81)    LengthUnit, AngleUnit, DisplayUnits
│   │   ├── grid.rs                    (40)    GridSettings + snap methods
│   │   ├── view_transform.rs          (38)    ViewTransform + world_to_screen/screen_to_world
│   │   ├── load_cases.rs              (73)    LoadCase, LoadCaseManager + CRUD
│   │   ├── types.rs                  (116)    EditorTool, SelectedEntity, SolverStatus, ForceResults, etc.
│   │   ├── parametric.rs             (235)    SweepParameter, ParametricMetric, Config/Result
│   │   ├── simulation.rs              (21)    SimulationState struct
│   │   ├── blueprint_ops.rs        (1,073)    rebuild(), force/validation/sweep compute, blueprint edits
│   │   ├── entity_crud.rs            (394)    Body/joint/attachment CRUD operations
│   │   ├── driver_ops.rs             (435)    Driver assignment, load cases, parametric studies
│   │   ├── file_io.rs                (269)    save/load/autosave/recent files
│   │   ├── undo_ops.rs               (136)    push_undo, undo, redo, snapshot/restore
│   │   └── tests.rs                (2,104)    116 unit tests
│   ├── canvas/                     (3,160)    2D mechanism canvas
│   │   ├── mod.rs                    (211)    draw_canvas entry point + tests
│   │   ├── colors.rs                  (67)    Color and sizing constants
│   │   ├── hit_testing.rs             (86)    AttachmentHit, SegmentHit, projection helpers
│   │   ├── rendering.rs            (1,836)    Bodies, joints, forces, grid, tooltips, drawing primitives
│   │   ├── interaction.rs            (780+)   Pan, zoom, drag (incl. ground pivot dragging), tool modes, click selection
│   │   └── context_menu.rs           (248)    Right-click menus for joints, bodies, canvas
│   ├── property_panel/             (2,175)    Property editing panel
│   │   ├── mod.rs                    (387)    draw_property_panel main coordinator
│   │   ├── pending_edits.rs          (210)    PendingPropertyEdit enum + apply_pending
│   │   ├── health.rs                 (330)    Mechanism Health section (Grashof, toggle, torque, reactions, actuator stroke)
│   │   ├── diagnostics.rs            (360)    Diagnostics section + motor sizing + force zone feedback
│   │   └── force_editor.rs         (1,273)    Per-force-type parameter editors
│   ├── samples/                    (2,500+)   Sample mechanism builders (29 total, with thumbnails)
│   │   ├── mod.rs                    (400+)   SampleMechanism enum, build_sample dispatch, tests
│   │   ├── helpers.rs                (219)    attach_driver, fourbar_initial_q0, continuation solver
│   │   ├── fourbar.rs                (570+)   12 four-bar variants (incl. Hoeken, Roberts)
│   │   ├── sixbar.rs                 (700+)   7 six-bar variants (incl. Watt II, Pantograph)
│   │   └── special.rs              (1,000+)   Slider-crank variants, bell crank, yoke, Strandbeest walking
│   ├── export/                     (1,577)    File export
│   │   ├── mod.rs                     (20)    Re-exports
│   │   ├── csv.rs                    (454)    CSV + coupler CSV export
│   │   ├── svg.rs                    (318)    SVG string generation + file export
│   │   ├── raster.rs                 (346)    PNG rasterization + GIF animation
│   │   ├── dxf.rs                    (132)    DXF generation + export
│   │   └── report.rs                 (307)    HTML report generation
│   ├── plot_panel.rs               (1,230)    11-tab sweep data plots (incl. Actuator Force with statics + inertia curves)
│   ├── sweep.rs                      (600)    SweepData, compute_sweep_data, 4-bar detection, actuator force/ID/length
│   ├── input_panel.rs                (486)    Driver controls, animation, sweep range
│   ├── parametric_panel.rs           (410)    Parametric study + counterbalance UI
│   ├── undo.rs                       (258)    Undo/redo history stack
│   ├── force_toolbar.rs              (230)    Force creation toolbar
│   ├── error_panel.rs                (33)    Error message display
│   └── tutorial.rs                   (309)    Interactive tutorial overlay (4-bar walkthrough)
│
└── bin/
    ├── linkage_gui.rs                 (18)    Native desktop entry point
    └── linkage_web.rs                 (52)    WASM web entry point
```

---

## Data Flow

```
User Input (GUI)
    │
    ▼
AppState (gui/state/mod.rs) ◄──── Central hub: owns mechanism, q, blueprint, all UI state
    │
    ├──► Blueprint (MechanismJson) ◄──► File I/O (io/schema.rs + io/from_json.rs + io/to_json.rs)
    │       │
    │       ▼
    ├──► Mechanism (core/mechanism.rs) ◄── Bodies + Joints + Drivers + Forces
    │       │
    │       ▼
    ├──► Solver Pipeline:
    │       position (kinematics) → velocity → acceleration
    │       → statics / inverse dynamics / forward dynamics
    │       │
    │       ▼
    ├──► Sweep Data (gui/sweep.rs) ── computed for all 360° at each rebuild
    │       │
    │       ▼
    └──► Rendering:
            canvas/ (2D mechanism drawing)
            plot_panel.rs (charts)
            property_panel/ (editing)
            export/ (file output)
```

### Key Architectural Rules

1. **Bodies are truth; constraints connect them.** No special-casing for 4-bar vs 6-bar.
2. **Constraint-first math.** `Φ(q,t) = 0` and `Φ_q` are the backbone.
3. **Force elements are pluggable.** All return Q contributions via virtual work.
4. **SI internally, engineering units at GUI boundary.**
5. **Drivers are constraints, not forces.** Lagrange multiplier = required effort. Both revolute drivers (prescribe angle) and linear drivers (prescribe distance between two points) are supported.
6. **Blueprint is the source of truth.** Mechanism is rebuilt from MechanismJson on edits.
7. **Mounting angle** (`mounting_angle`, radians, default 0.0): stored on `MechanismJson` and `AppState`, synced via `rebuild()` and `file_io.rs`. Rotates the gravity vector in physics (`-g*sin(theta)`, `-g*cos(theta)`) and the canvas via `ViewTransform`. Backward-compatible in JSON (`#[serde(default)]`).

---

## Refactoring History (2026-03-24)

All 8 planned splits executed successfully. 438 tests pass, zero failures.

```
Before:  48 files,  8 files > 1,500 lines,  largest = 5,725 lines
After:   92 files,  0 files > 1,500 lines (excl. test-only),  largest impl = 1,379 lines
```

| Split | Original | New Files | Status |
|-------|----------|-----------|--------|
| `io/serialization.rs` (1,570) | 1 file | 5 files (schema, to_json, from_json, error, tests) | Done |
| `core/constraint.rs` (1,619) | 1 file | 7 files (trait, helpers, revolute, fixed, prismatic, cam, mod) | Done |
| `forces/elements.rs` (2,443) | 1 file | 5 files (time_mod, types, eval, force_element, mod) | Done |
| `gui/export.rs` (1,512) | 1 file | 6 files (csv, svg, raster, dxf, report, mod) | Done |
| `gui/samples.rs` (2,500+) | 1 file | 5 files (fourbar, sixbar, special, helpers, mod) | Done |
| `gui/property_panel.rs` (2,143) | 1 file | 4 files (pending_edits, diagnostics, force_editor, mod) | Done |
| `gui/canvas.rs` (2,880) | 1 file | 6 files (colors, hit_testing, rendering, interaction, context_menu, mod) | Done |
| `gui/state.rs` (5,725) | 1 file | 14 files (display_units, grid, view_transform, load_cases, types, parametric, simulation, blueprint_ops, entity_crud, driver_ops, file_io, undo_ops, tests, mod) | Done |

### Remaining large files (not split — at natural size)

| File | Lines | Reason |
|------|-------|--------|
| `gui/state/tests.rs` | 2,104 | Test-only file, acceptable |
| `gui/canvas/rendering.rs` | 1,836 | Drawing primitives — cohesive, all rendering |
| `solver/forward_dynamics.rs` | 1,379 | BDF-2 integrator — math-heavy, tightly coupled |
| `forces/elements/mod.rs` | 1,325 | Tests only (1,300 lines of element tests) |
| `gui/property_panel/force_editor.rs` | 1,273 | 13 force type editors — repetitive but cohesive |
| `gui/plot_panel.rs` | 1,140 | 11 plot tabs (incl. Actuator Force) — repetitive but cohesive |
| `gui/state/blueprint_ops.rs` | 1,073 | Rebuild + all blueprint mutation — tightly coupled |
