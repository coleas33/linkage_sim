# Linkage Simulation — Rust Codebase System Guide

**Date**: 2026-03-23
**Schema Version**: 1.1.0
**Total**: 48 Rust source files, 32,499 lines

This document maps the system for AI-assisted development. Every module, file, and data flow is documented so an agent can navigate, modify, and refactor with confidence.

---

## Module Tree

```
linkage-sim-rs/src/
├── lib.rs                        (8)    Crate root — declares 8 modules
├── error.rs                     (59)    LinkageError enum
├── geometry/mod.rs             (194)    Polygon clip, area, centroid, body_rect_to_world
│
├── core/                      (2,161 total)
│   ├── state.rs                (388)    State, BodyIndex, coordinate mapping, pose helpers
│   ├── body.rs                 (446)    Body, BodyGeometry, make_bar, make_ground
│   ├── constraint.rs         (1,619)    ★ Constraint trait, 4 joint types, JointConstraint enum
│   ├── driver.rs               (418)    DriverConstraint (closure-based), DriverMeta
│   └── mechanism.rs            (750)    Mechanism: builder, body/joint/driver/force storage
│
├── forces/                    (3,142 total)
│   ├── helpers.rs              (141)    point_force_to_q, body_torque_to_q (virtual work)
│   ├── elements.rs           (2,443)    ★ 13 force types + ForceElement enum + evaluators
│   └── compound.rs             (558)    Compound force expansion (mount point → bodies)
│
├── solver/                    (2,966 total)
│   ├── assembly.rs             (149)    Jacobian, constraint, mass matrix assembly
│   ├── kinematics.rs           (266)    Newton-Raphson position, velocity, acceleration
│   ├── statics.rs              (357)    Static equilibrium via Lagrange multipliers
│   ├── inverse_dynamics.rs     (308)    Full Newton-Euler inverse dynamics
│   ├── forward_dynamics.rs   (1,379)    BDF-2 DAE integrator, adaptive time-stepping
│   └── events.rs               (504)    Joint limit / force zone event detection
│
├── analysis/                  (2,706 total)
│   ├── validation.rs           (758)    Toggle detection, DOF analysis, connectivity checks
│   ├── transmission.rs         (313)    Transmission angle, mechanical advantage
│   ├── crank_selection.rs      (305)    Which link to drive recommendation
│   ├── energy.rs               (279)    Kinetic/potential energy computation
│   ├── virtual_work.rs         (280)    Energy-based statics validation
│   ├── motor_sizing.rs         (202)    Motor sizing validation
│   ├── force_breakdown.rs      (196)    Per-element force contribution
│   ├── grashof.rs              (194)    Grashof classification
│   ├── coupler.rs              (179)    Coupler point evaluation
│   └── envelopes.rs            (169)    Min/max/RMS of sweep series
│
├── io/                        (1,571 total)
│   └── serialization.rs      (1,570)    ★ JSON schema types, serialize, deserialize
│
├── gui/                      (13,331 total)
│   ├── mod.rs                  (971)    LinkageApp, eframe::App::update, menu bar, toolbar
│   ├── state.rs              (5,725)    ★★ AppState (44 fields), all mutation methods, 116 tests
│   ├── canvas.rs             (2,880)    ★ Rendering, interaction, hit testing, context menus
│   ├── property_panel.rs     (2,143)    ★ Body/joint/force property editing
│   ├── samples.rs            (1,688)    ★ 19 sample mechanism builders
│   ├── export.rs             (1,512)    ★ PNG/SVG/GIF/DXF/CSV/HTML export
│   ├── plot_panel.rs         (1,078)    10-tab sweep data plots
│   ├── sweep.rs                (558)    SweepData, compute_sweep_data, 4-bar detection
│   ├── input_panel.rs          (486)    Driver controls, animation, sweep range
│   ├── parametric_panel.rs     (410)    Parametric study + counterbalance UI
│   ├── undo.rs                 (258)    Undo/redo history stack
│   ├── force_toolbar.rs        (230)    Force creation toolbar
│   └── error_panel.rs          (33)    Error message display
│
└── bin/
    ├── linkage_gui.rs           (18)    Native desktop entry point
    └── linkage_web.rs           (52)    WASM web entry point
```

★ = over 1,500 lines, needs splitting. ★★ = critical, 5,725 lines.

---

## Data Flow

```
User Input (GUI)
    │
    ▼
AppState (gui/state.rs) ◄──── Central hub: owns mechanism, q, blueprint, all UI state
    │
    ├──► Blueprint (MechanismJson) ◄──► File I/O (io/serialization.rs)
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
            canvas.rs (2D mechanism drawing)
            plot_panel.rs (charts)
            property_panel.rs (editing)
            export.rs (file output)
```

### Key Architectural Rules

1. **Bodies are truth; constraints connect them.** No special-casing for 4-bar vs 6-bar.
2. **Constraint-first math.** `Φ(q,t) = 0` and `Φ_q` are the backbone.
3. **Force elements are pluggable.** All return Q contributions via virtual work.
4. **SI internally, engineering units at GUI boundary.**
5. **Drivers are constraints, not forces.** Lagrange multiplier = required effort.
6. **Blueprint is the source of truth.** Mechanism is rebuilt from MechanismJson on edits.

---

## The 8 Files That Need Splitting

### 1. `gui/state.rs` (5,725 lines → ~13 files)

The monolithic AppState and all its methods. Split into:

| New File | Lines | Contents |
|----------|-------|----------|
| `gui/state/mod.rs` | ~200 | AppState struct definition, Default impl, re-exports |
| `gui/state/display_units.rs` | ~80 | LengthUnit, AngleUnit, DisplayUnits |
| `gui/state/grid.rs` | ~90 | GridSettings, auto_grid_spacing, snap logic |
| `gui/state/view_transform.rs` | ~40 | ViewTransform, world_to_screen, screen_to_world |
| `gui/state/load_cases.rs` | ~140 | LoadCase, LoadCaseManager, CRUD operations |
| `gui/state/types.rs` | ~100 | EditorTool, ContextMenuTarget, SelectedEntity, SolverStatus, etc. |
| `gui/state/file_io.rs` | ~250 | save/load/autosave/recent files |
| `gui/state/blueprint_ops.rs` | ~270 | rebuild, set_link_length, set_mass, set_izz, recompute_dynamics |
| `gui/state/entity_crud.rs` | ~280 | add/remove bodies, joints, ground pivots, attachment points |
| `gui/state/driver_ops.rs` | ~220 | reassign_driver, set_expression_driver, validation |
| `gui/state/parametric.rs` | ~270 | ParametricStudyConfig/Result, run_parametric_study |
| `gui/state/simulation.rs` | ~280 | SimulationState, run_simulation, step_simulation, step_animation |
| `gui/state/undo_ops.rs` | ~140 | push_undo, undo, redo, snapshot/restore |

Tests stay alongside their module or in a `tests/` subfolder.

### 2. `gui/canvas.rs` (2,880 lines → 6 files)

| New File | Lines | Contents |
|----------|-------|----------|
| `gui/canvas/mod.rs` | ~60 | draw_canvas entry point, delegates to submodules |
| `gui/canvas/colors.rs` | ~60 | All color/sizing constants |
| `gui/canvas/hit_testing.rs` | ~80 | AttachmentHit, SegmentHit, projection helpers |
| `gui/canvas/rendering.rs` | ~660 | Grid, bodies, joints, forces, overlays, labels |
| `gui/canvas/tooltips.rs` | ~140 | Hover tooltip rendering |
| `gui/canvas/interaction.rs` | ~700 | Drag, pan, zoom, tool modes, click selection |
| `gui/canvas/context_menu.rs` | ~1,080 | Right-click menus (joint, body, canvas background) |

### 3. `forces/elements.rs` (2,443 lines → 4 files)

| New File | Lines | Contents |
|----------|-------|----------|
| `forces/time_modulation.rs` | ~130 | TimeModulation enum, factor(), compile() |
| `forces/element_types.rs` | ~280 | All 13 element data structs |
| `forces/evaluation.rs` | ~470 | All evaluate_* functions |
| `forces/force_element.rs` | ~260 | ForceElement enum, impl block (dispatch) |

Tests → `tests/force_element_tests.rs` (1,306 lines).

### 4. `gui/property_panel.rs` (2,143 lines → 4 files)

| New File | Lines | Contents |
|----------|-------|----------|
| `gui/property_panel/mod.rs` | ~370 | draw_property_panel main coordinator |
| `gui/property_panel/pending_edits.rs` | ~200 | PendingPropertyEdit enum + apply_pending |
| `gui/property_panel/diagnostics.rs` | ~290 | draw_diagnostics_section |
| `gui/property_panel/force_editor.rs` | ~1,280 | Per-force-type parameter editors |

### 5. `gui/samples.rs` (1,688 lines → 5 files)

| New File | Lines | Contents |
|----------|-------|----------|
| `gui/samples/mod.rs` | ~130 | SampleMechanism enum, dispatch, build_sample |
| `gui/samples/helpers.rs` | ~60 | attach_driver, fourbar_initial_q0 |
| `gui/samples/fourbar.rs` | ~400 | All 4-bar variants (standard, crank-rocker, parallelogram, press) |
| `gui/samples/sixbar.rs` | ~480 | All 5 six-bar variants |
| `gui/samples/special.rs` | ~230 | Quick-return, toggle clamp, scotch yoke, inverted slider-crank |

### 6. `core/constraint.rs` (1,619 lines → 7 files)

| New File | Lines | Contents |
|----------|-------|----------|
| `core/constraint/mod.rs` | ~130 | JointConstraint enum + delegating impl |
| `core/constraint/trait_def.rs` | ~20 | Constraint trait definition |
| `core/constraint/helpers.rs` | ~70 | translational_jacobian_block, translational_gamma |
| `core/constraint/revolute.rs` | ~110 | RevoluteJoint |
| `core/constraint/fixed.rs` | ~145 | FixedJoint |
| `core/constraint/prismatic.rs` | ~200 | PrismaticJoint |
| `core/constraint/cam.rs` | ~200 | CamProfile, CamFollowerJoint, spline helpers |

### 7. `io/serialization.rs` (1,570 lines → 4 files)

| New File | Lines | Contents |
|----------|-------|----------|
| `io/schema.rs` | ~210 | MechanismJson, BodyJson, JointJson, DriverJson, SweepConfig |
| `io/to_json.rs` | ~200 | mechanism_to_json, body_to_json, joint_to_json, save |
| `io/from_json.rs` | ~290 | load_mechanism_unbuilt, load_mechanism_unbuilt_from_json |
| `io/error.rs` | ~30 | SerializationError |

### 8. `gui/export.rs` (1,512 lines → 5 files)

| New File | Lines | Contents |
|----------|-------|----------|
| `gui/export/mod.rs` | ~20 | Re-exports |
| `gui/export/csv.rs` | ~170 | CSV + coupler CSV export |
| `gui/export/svg.rs` | ~290 | SVG string generation + file export |
| `gui/export/raster.rs` | ~140 | PNG rasterization + GIF animation |
| `gui/export/dxf.rs` | ~100 | DXF generation + export |
| `gui/export/report.rs` | ~260 | HTML report generation |

---

## Target State After Refactoring

```
Current:  48 files,  8 files > 1,500 lines,  largest = 5,725 lines
Target:  ~95 files,  0 files > 700 lines,  largest ≈ 500-600 lines
```

| Module | Current files | Target files | Avg lines |
|--------|:---:|:---:|:---:|
| core/ | 5 | 11 | ~200 |
| forces/ | 3 | 6 | ~350 |
| solver/ | 6 | 6 | (unchanged) |
| analysis/ | 10 | 10 | (unchanged) |
| io/ | 1 | 4 | ~250 |
| gui/ | 13 | 35 | ~350 |
| geometry/ | 1 | 1 | (unchanged) |
| other | 4 | 4 | (unchanged) |

---

## Refactoring Order (Priority)

Execute in this order to minimize conflicts and keep tests green at each step:

1. **`io/serialization.rs`** → 4 files (lowest risk — schema types are widely imported but simple to split)
2. **`core/constraint.rs`** → 7 files (self-contained math, clear per-joint-type boundaries)
3. **`forces/elements.rs`** → 4 files (clear struct/eval/enum separation)
4. **`gui/export.rs`** → 5 files (cfg-gated, isolated from other GUI)
5. **`gui/samples.rs`** → 5 files (builders are independent functions)
6. **`gui/property_panel.rs`** → 4 files (editor sections are independent)
7. **`gui/canvas.rs`** → 6 files (rendering vs interaction split)
8. **`gui/state.rs`** → 13 files (**last** — highest risk, most dependencies, needs careful import surgery)

Each split should be one commit with `cargo test` green before and after.
