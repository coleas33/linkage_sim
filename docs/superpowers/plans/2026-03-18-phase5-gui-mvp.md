# Phase 5 GUI MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only egui visualization shell that loads sample mechanisms, renders them on a 2D canvas, and drives the kinematic solver via an angle slider.

**Architecture:** Immediate-mode GUI using egui/eframe. `AppState` owns the `Mechanism` and solved `q`. Slider changes trigger position solve; canvas renders from solved world-space poses. Selection model uses `Option<SelectedEntity>`. GUI module lives in `src/gui/` within the library crate; binary target at `src/bin/linkage_gui.rs`.

**Tech Stack:** Rust, egui 0.33, eframe 0.33, nalgebra (existing), linkage-sim-rs solver (existing)

**Spec:** `docs/superpowers/specs/2026-03-18-parallel-workstreams-design.md` — Workstream 2

**eframe API note:** eframe 0.33 removed `eframe::Frame` from the `App::update` signature. The correct signature is `fn update(&mut self, ctx: &egui::Context)`. The `AppCreator` closure returns `Result<Box<dyn App>, Box<dyn Error + Send + Sync>>`. If any API call does not compile, run `cargo doc --open -p eframe` to check the local docs for the installed version.

**Known deviations from spec (deferred to future iteration):**
- Slider range hardcoded to 0..360. Spec requires `DrivenRangeResult` integration, which depends on Workstream 1 (crank selection). Add range-aware slider after Workstream 1 lands.
- Debug overlay omits yellow/marginal state. Near-singular detection requires SVD monitoring in the solver, which is not yet exposed. Green/red (converged/failed) is sufficient for MVP.
- Prismatic joints rendered as square markers. Spec calls for arrows along slide axis. Upgrade rendering after basic MVP is validated.

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `src/gui/mod.rs` | `LinkageApp` struct, `eframe::App` impl, top-level layout (menu bar, panels, canvas area, status bar) |
| `src/gui/state.rs` | `AppState`, `SolverStatus`, `SelectedEntity`, `ViewTransform` — all app state in one place |
| `src/gui/canvas.rs` | 2D mechanism renderer: world-to-screen transform, body/joint/ground drawing, hit testing, debug overlay |
| `src/gui/input_panel.rs` | Angle slider, solver status display |
| `src/gui/property_panel.rs` | Read-only property display for selected entity |
| `src/gui/samples.rs` | Hardcoded sample mechanism builders (4-bar, slider-crank) |
| `src/bin/linkage_gui.rs` | Binary entry point — `fn main()` calls `eframe::run_native()` |

### Modified files

| File | Change |
|------|--------|
| `src/lib.rs` | Add `pub mod gui;` |
| `src/core/mechanism.rs` | Add `joints()` and `drivers_meta()` public accessors |
| `src/core/constraint.rs` | Add `point_i_local()`, `point_j_local()` accessors to `JointConstraint` |
| `Cargo.toml` | Add egui/eframe deps + `[[bin]]` target |

---

## Task 1: Add GUI accessor methods to solver crate

The GUI needs to iterate joints and read their geometry. Currently `Mechanism.joints` is private and joint point coordinates have no public accessors.

**Files:**
- Modify: `linkage-sim-rs/src/core/constraint.rs`
- Modify: `linkage-sim-rs/src/core/mechanism.rs`

- [ ] **Step 1: Add point accessors to JointConstraint**

In `linkage-sim-rs/src/core/constraint.rs`, add these methods to the `JointConstraint` enum, after the existing `Constraint` impl block (after line 558). **Important:** This code must be in `constraint.rs` (same module) because the inner struct fields (`point_i_local`, `point_j_local`) are private — same-module access is required:

```rust
impl JointConstraint {
    /// Get the local attachment point on body_i.
    pub fn point_i_local(&self) -> nalgebra::Vector2<f64> {
        match self {
            Self::Revolute(j) => j.point_i_local,
            Self::Fixed(j) => j.point_i_local,
            Self::Prismatic(j) => j.point_i_local,
        }
    }

    /// Get the local attachment point on body_j.
    pub fn point_j_local(&self) -> nalgebra::Vector2<f64> {
        match self {
            Self::Revolute(j) => j.point_j_local,
            Self::Fixed(j) => j.point_j_local,
            Self::Prismatic(j) => j.point_j_local,
        }
    }

    /// True if this is a revolute joint.
    pub fn is_revolute(&self) -> bool {
        matches!(self, Self::Revolute(_))
    }

    /// True if this is a prismatic joint.
    pub fn is_prismatic(&self) -> bool {
        matches!(self, Self::Prismatic(_))
    }

    /// True if this is a fixed joint.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }
}
```

- [ ] **Step 2: Add joints() and driver accessors to Mechanism**

In `linkage-sim-rs/src/core/mechanism.rs`, add after the `bodies()` method (after line 44):

```rust
    /// Public read access to joint constraints.
    pub fn joints(&self) -> &[JointConstraint] {
        &self.joints
    }

    /// Number of driver constraints.
    pub fn n_drivers(&self) -> usize {
        self.drivers.len()
    }
```

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `cd linkage-sim-rs && cargo test`
Expected: All existing tests pass (no API was changed, only new methods added).

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/core/constraint.rs linkage-sim-rs/src/core/mechanism.rs
git commit -m "feat(rs): add GUI accessor methods to JointConstraint and Mechanism"
```

---

## Task 2: Add egui/eframe dependencies and binary target

**Files:**
- Modify: `linkage-sim-rs/Cargo.toml`
- Create: `linkage-sim-rs/src/bin/linkage_gui.rs`
- Modify: `linkage-sim-rs/src/lib.rs`

- [ ] **Step 1: Update Cargo.toml**

Add egui/eframe dependencies and binary target. In `linkage-sim-rs/Cargo.toml`:

```toml
[package]
name = "linkage-sim-rs"
version = "0.1.0"
edition = "2024"
description = "Planar linkage mechanism simulator — Rust port of the validated Python solver"

[dependencies]
nalgebra = { version = "0.33", features = ["serde-serialize"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
eframe = "0.33"
egui_extras = "0.33"
log = "0.4"
env_logger = "0.11"

[dev-dependencies]
approx = "0.5"
proptest = "1"

[[bin]]
name = "linkage-gui"
path = "src/bin/linkage_gui.rs"
```

- [ ] **Step 2: Create minimal binary entry point**

Create `linkage-sim-rs/src/bin/linkage_gui.rs`:

```rust
//! Linkage mechanism simulator — GUI application.

fn main() -> eframe::Result<()> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("Linkage Simulator"),
        ..Default::default()
    };

    eframe::run_native(
        "Linkage Simulator",
        options,
        Box::new(|cc| Ok(Box::new(linkage_sim_rs::gui::LinkageApp::new(cc)))),
    )
}
```

- [ ] **Step 3: Create gui module skeleton**

Create `linkage-sim-rs/src/gui/mod.rs`:

```rust
//! GUI module — egui-based visualization shell for the linkage simulator.

mod state;
mod canvas;
mod input_panel;
mod property_panel;
mod samples;

use eframe::egui;
pub use state::AppState;

/// Top-level application struct for eframe.
pub struct LinkageApp {
    state: AppState,
}

impl LinkageApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            state: AppState::default(),
        }
    }
}

impl eframe::App for LinkageApp {
    fn update(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Linkage Simulator");
            ui.label("GUI shell — loading...");
        });
    }
}
```

Create `linkage-sim-rs/src/gui/state.rs`:

```rust
//! Application state: mechanism, solver results, selection, view transform.

/// All mutable application state in one place.
#[derive(Default)]
pub struct AppState {}
```

Create empty stub files for the remaining modules:

`linkage-sim-rs/src/gui/canvas.rs`:
```rust
//! 2D mechanism canvas: rendering, pan/zoom, hit testing, debug overlay.
```

`linkage-sim-rs/src/gui/input_panel.rs`:
```rust
//! Angle slider and solver status display.
```

`linkage-sim-rs/src/gui/property_panel.rs`:
```rust
//! Read-only property panel for the selected entity.
```

`linkage-sim-rs/src/gui/samples.rs`:
```rust
//! Hardcoded sample mechanism builders for the GUI.
```

- [ ] **Step 4: Add gui module to lib.rs**

In `linkage-sim-rs/src/lib.rs`, add:

```rust
pub mod gui;
```

- [ ] **Step 5: Verify it compiles and the window opens**

Run: `cd linkage-sim-rs && cargo build --bin linkage-gui`
Expected: Compiles without errors.

Run: `cd linkage-sim-rs && cargo run --bin linkage-gui`
Expected: Window opens with "Linkage Simulator" heading. Close manually.

- [ ] **Step 6: Commit**

```bash
git add linkage-sim-rs/Cargo.toml linkage-sim-rs/Cargo.lock linkage-sim-rs/src/bin/linkage_gui.rs linkage-sim-rs/src/gui/ linkage-sim-rs/src/lib.rs
git commit -m "feat(gui): add egui/eframe app shell with empty window"
```

---

## Task 3: AppState and sample mechanism builders

**Files:**
- Modify: `linkage-sim-rs/src/gui/state.rs`
- Modify: `linkage-sim-rs/src/gui/samples.rs`

- [ ] **Step 1: Write test for sample mechanism builder**

Add to `linkage-sim-rs/src/gui/samples.rs`:

```rust
//! Hardcoded sample mechanism builders for the GUI.

use crate::core::body::{make_bar, make_ground};
use crate::core::mechanism::Mechanism;
use crate::solver::kinematics::solve_position;
use nalgebra::DVector;
use std::f64::consts::PI;

/// Available sample mechanisms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleMechanism {
    FourBar,
    SliderCrank,
}

impl SampleMechanism {
    pub fn label(&self) -> &'static str {
        match self {
            Self::FourBar => "4-Bar Crank-Rocker",
            Self::SliderCrank => "Slider-Crank",
        }
    }

    pub fn all() -> &'static [SampleMechanism] {
        &[SampleMechanism::FourBar, SampleMechanism::SliderCrank]
    }
}

/// Build a sample mechanism and return it with an initial guess q0.
pub fn build_sample(sample: SampleMechanism) -> (Mechanism, DVector<f64>) {
    match sample {
        SampleMechanism::FourBar => build_fourbar_sample(),
        SampleMechanism::SliderCrank => build_slider_crank_sample(),
    }
}

/// Grashof crank-rocker: ground=0.038m, crank=0.01m, coupler=0.04m, rocker=0.03m
fn build_fourbar_sample() -> (Mechanism, DVector<f64>) {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
    let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
    let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
    let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
    mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0).unwrap();

    mech.build().unwrap();

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
    state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
    state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);

    (mech, q0)
}

/// Slider-crank: crank=0.01m, coupler=0.04m, slider on x-axis
fn build_slider_crank_sample() -> (Mechanism, DVector<f64>) {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("rail", 0.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
    let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);

    let mut slider = crate::core::body::Body::new("slider");
    slider.add_attachment_point("C", 0.0, 0.0).unwrap();

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(slider).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "slider", "C").unwrap();
    mech.add_prismatic_joint(
        "P1", "ground", "rail", "slider", "C",
        nalgebra::Vector2::new(1.0, 0.0), 0.0,
    ).unwrap();
    mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0).unwrap();

    mech.build().unwrap();

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
    state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
    state.set_pose("slider", &mut q0, 0.05, 0.0, 0.0);

    (mech, q0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fourbar_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50);
        assert!(result.converged, "residual = {}", result.residual_norm);
    }

    #[test]
    fn slider_crank_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::SliderCrank);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50);
        assert!(result.converged, "residual = {}", result.residual_norm);
    }

    #[test]
    fn all_samples_listed() {
        assert_eq!(SampleMechanism::all().len(), 2);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cd linkage-sim-rs && cargo test gui::samples`
Expected: 3 tests pass.

- [ ] **Step 3: Write AppState with solver integration**

Replace `linkage-sim-rs/src/gui/state.rs` with:

```rust
//! Application state: mechanism, solver results, selection, view transform.

use nalgebra::DVector;
use crate::core::mechanism::Mechanism;
use crate::solver::kinematics::solve_position;
use super::samples::{SampleMechanism, build_sample};

/// What kind of mechanism element is selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectedEntity {
    Body(String),
    Joint(String),
    Driver(String),
}

/// Solver convergence status for display.
#[derive(Debug, Clone)]
pub struct SolverStatus {
    pub converged: bool,
    pub residual_norm: f64,
    pub iterations: usize,
}

impl Default for SolverStatus {
    fn default() -> Self {
        Self {
            converged: true,
            residual_norm: 0.0,
            iterations: 0,
        }
    }
}

/// 2D view transform: pan offset + zoom scale.
#[derive(Debug, Clone)]
pub struct ViewTransform {
    /// Offset in screen pixels (pan).
    pub offset: [f32; 2],
    /// Pixels per meter.
    pub scale: f32,
}

impl Default for ViewTransform {
    fn default() -> Self {
        Self {
            offset: [400.0, 400.0],
            scale: 5000.0, // 5000 px/m → 1mm = 5px, good for small mechanisms
        }
    }
}

impl ViewTransform {
    /// Convert world coordinates (meters) to screen pixels.
    pub fn world_to_screen(&self, wx: f64, wy: f64) -> [f32; 2] {
        [
            self.offset[0] + (wx as f32) * self.scale,
            self.offset[1] - (wy as f32) * self.scale, // flip Y: screen Y grows down
        ]
    }

    /// Convert screen pixels to world coordinates (meters).
    pub fn screen_to_world(&self, sx: f32, sy: f32) -> [f64; 2] {
        [
            ((sx - self.offset[0]) / self.scale) as f64,
            ((self.offset[1] - sy) / self.scale) as f64,
        ]
    }
}

/// All mutable application state in one place.
pub struct AppState {
    /// The loaded mechanism (None if nothing loaded).
    pub mechanism: Option<Mechanism>,
    /// Current solved generalized coordinate vector.
    pub q: DVector<f64>,
    /// Current driver angle in radians.
    pub driver_angle: f64,
    /// Last successful q (used as fallback on solver failure).
    pub last_good_q: DVector<f64>,
    /// Solver convergence status.
    pub solver_status: SolverStatus,
    /// Currently selected entity.
    pub selected: Option<SelectedEntity>,
    /// View transform (pan/zoom).
    pub view: ViewTransform,
    /// Whether to show the debug overlay.
    pub show_debug_overlay: bool,
    /// Which sample is currently loaded.
    pub current_sample: Option<SampleMechanism>,
    /// Driver angular velocity (rad/s) — stored when loading sample.
    pub driver_omega: f64,
    /// Driver initial angle offset (rad) — stored when loading sample.
    pub driver_theta_0: f64,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            mechanism: None,
            q: DVector::zeros(0),
            driver_angle: 0.0,
            last_good_q: DVector::zeros(0),
            solver_status: SolverStatus::default(),
            selected: None,
            view: ViewTransform::default(),
            show_debug_overlay: cfg!(debug_assertions),
            current_sample: None,
            driver_omega: 2.0 * std::f64::consts::PI,
            driver_theta_0: 0.0,
        }
    }
}

impl AppState {
    /// Load a sample mechanism and solve at angle 0.
    pub fn load_sample(&mut self, sample: SampleMechanism) {
        let (mech, q0) = build_sample(sample);
        self.driver_angle = 0.0;
        self.selected = None;
        // All current samples use omega=2*pi, theta_0=0.
        // Store these so solve_at_angle can compute t correctly.
        self.driver_omega = 2.0 * std::f64::consts::PI;
        self.driver_theta_0 = 0.0;

        // Solve at t=0 to get the initial configuration
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50);
        self.solver_status = SolverStatus {
            converged: result.converged,
            residual_norm: result.residual_norm,
            iterations: result.iterations,
        };
        self.q = result.q.clone();
        self.last_good_q = result.q;
        self.mechanism = Some(mech);
        self.current_sample = Some(sample);
    }

    /// Re-solve at the current driver angle.
    /// Uses last_good_q as initial guess for continuation.
    pub fn solve_at_angle(&mut self, angle_rad: f64) {
        self.driver_angle = angle_rad;

        let Some(mech) = &self.mechanism else { return };

        // Driver prescribes f(t) = omega * t + theta_0.
        // Solve for t: angle_rad = omega * t + theta_0 → t = (angle_rad - theta_0) / omega
        let t = (angle_rad - self.driver_theta_0) / self.driver_omega;

        let result = solve_position(mech, &self.last_good_q, t, 1e-10, 50);
        self.solver_status = SolverStatus {
            converged: result.converged,
            residual_norm: result.residual_norm,
            iterations: result.iterations,
        };

        if result.converged {
            self.q = result.q.clone();
            self.last_good_q = result.q;
        }
        // If not converged, keep last_good_q and q unchanged (spec: hold last good pose)
    }

    /// Whether a mechanism is loaded and ready.
    pub fn has_mechanism(&self) -> bool {
        self.mechanism.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_has_no_mechanism() {
        let state = AppState::default();
        assert!(!state.has_mechanism());
    }

    #[test]
    fn load_sample_solves_initial() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(state.has_mechanism());
        assert!(state.solver_status.converged);
        assert!(state.q.len() > 0);
    }

    #[test]
    fn solve_at_angle_updates_state() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let q_before = state.q.clone();
        state.solve_at_angle(0.5);
        // q should change because the crank moved
        assert_ne!(state.q, q_before);
        assert!(state.solver_status.converged);
    }

    #[test]
    fn world_to_screen_roundtrip() {
        let vt = ViewTransform::default();
        let screen = vt.world_to_screen(0.01, 0.02);
        let world = vt.screen_to_world(screen[0], screen[1]);
        assert!((world[0] - 0.01).abs() < 1e-6);
        assert!((world[1] - 0.02).abs() < 1e-6);
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cd linkage-sim-rs && cargo test gui`
Expected: All gui::state and gui::samples tests pass.

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/gui/state.rs linkage-sim-rs/src/gui/samples.rs
git commit -m "feat(gui): add AppState with solver integration and sample mechanisms"
```

---

## Task 4: Canvas rendering — bodies, joints, and ground

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs`

- [ ] **Step 1: Implement canvas rendering**

Replace `linkage-sim-rs/src/gui/canvas.rs` with:

```rust
//! 2D mechanism canvas: rendering, pan/zoom, hit testing, debug overlay.

use eframe::egui::{self, Color32, Pos2, Stroke, Vec2};
use crate::core::constraint::{Constraint, JointConstraint};
use crate::core::state::GROUND_ID;
use super::state::{AppState, SelectedEntity};

const BODY_COLOR: Color32 = Color32::from_rgb(60, 120, 200);
const BODY_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 180, 0);
const GROUND_COLOR: Color32 = Color32::from_rgb(100, 100, 100);
const JOINT_COLOR: Color32 = Color32::from_rgb(220, 60, 60);
const JOINT_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 220, 0);
const DEBUG_TEXT_COLOR: Color32 = Color32::from_rgb(180, 180, 180);
const JOINT_RADIUS: f32 = 5.0;
const GROUND_TRIANGLE_SIZE: f32 = 8.0;
const BODY_STROKE_WIDTH: f32 = 3.0;
const HIT_RADIUS: f32 = 10.0;

/// Draw the mechanism on the canvas and handle interaction.
pub fn draw_canvas(ui: &mut egui::Ui, state: &mut AppState) {
    let (response, painter) = ui.allocate_painter(
        ui.available_size(),
        egui::Sense::click_and_drag(),
    );
    let rect = response.rect;

    // Fill background
    painter.rect_filled(rect, 0.0, Color32::from_rgb(25, 25, 30));

    // Draw subtle y=0 ground line
    {
        let left = state.view.world_to_screen(-10.0, 0.0);
        let right = state.view.world_to_screen(10.0, 0.0);
        painter.line_segment(
            [Pos2::new(left[0], left[1]), Pos2::new(right[0], right[1])],
            Stroke::new(1.0, Color32::from_rgb(50, 50, 55)),
        );
    }

    let Some(mech) = &state.mechanism else {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No mechanism loaded.\nFile → Load Sample to begin.",
            egui::FontId::proportional(18.0),
            Color32::from_rgb(120, 120, 120),
        );
        return;
    };

    // Handle pan (drag)
    if response.dragged_by(egui::PointerButton::Middle)
        || (response.dragged_by(egui::PointerButton::Primary) && ui.input(|i| i.modifiers.shift))
    {
        let delta = response.drag_delta();
        state.view.offset[0] += delta.x;
        state.view.offset[1] += delta.y;
    }

    // Handle zoom (scroll wheel)
    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
    if scroll != 0.0 {
        let zoom_factor = 1.0 + scroll * 0.001;
        // Zoom toward mouse position
        if let Some(mouse_pos) = response.hover_pos() {
            let wx = ((mouse_pos.x - state.view.offset[0]) / state.view.scale) as f64;
            let wy = ((state.view.offset[1] - mouse_pos.y) / state.view.scale) as f64;
            state.view.scale *= zoom_factor;
            state.view.offset[0] = mouse_pos.x - (wx as f32) * state.view.scale;
            state.view.offset[1] = mouse_pos.y + (wy as f32) * state.view.scale;
        } else {
            state.view.scale *= zoom_factor;
        }
        state.view.scale = state.view.scale.clamp(100.0, 100_000.0);
    }

    let mech_state = mech.state();
    let q = &state.q;

    // Draw bodies: lines between attachment points
    for (body_id, body) in mech.bodies() {
        if body_id == GROUND_ID {
            continue; // ground drawn separately
        }

        let is_selected = state.selected == Some(SelectedEntity::Body(body_id.clone()));
        let color = if is_selected { BODY_SELECTED_COLOR } else { BODY_COLOR };

        // Collect all attachment point global positions
        let mut points: Vec<(String, [f32; 2])> = Vec::new();
        for (pt_name, pt_local) in &body.attachment_points {
            let global = mech_state.body_point_global(body_id, pt_local, q);
            let screen = state.view.world_to_screen(global.x, global.y);
            points.push((pt_name.clone(), screen));
        }

        // Draw lines between consecutive attachment points
        if points.len() >= 2 {
            // Sort by point name for consistent ordering
            points.sort_by(|a, b| a.0.cmp(&b.0));
            for i in 0..points.len() - 1 {
                painter.line_segment(
                    [Pos2::new(points[i].1[0], points[i].1[1]),
                     Pos2::new(points[i + 1].1[0], points[i + 1].1[1])],
                    Stroke::new(BODY_STROKE_WIDTH, color),
                );
            }
        }

        // Debug overlay: body ID at CG + attachment point labels
        if state.show_debug_overlay {
            let cg_global = mech_state.body_point_global(body_id, &body.cg_local, q);
            let cg_screen = state.view.world_to_screen(cg_global.x, cg_global.y);
            painter.text(
                Pos2::new(cg_screen[0], cg_screen[1] - 12.0),
                egui::Align2::CENTER_BOTTOM,
                body_id,
                egui::FontId::proportional(11.0),
                DEBUG_TEXT_COLOR,
            );

            // Attachment point labels (small, dimmed)
            let dimmed = Color32::from_rgb(120, 120, 130);
            for (pt_name, pt_local) in &body.attachment_points {
                let global = mech_state.body_point_global(body_id, pt_local, q);
                let screen = state.view.world_to_screen(global.x, global.y);
                painter.text(
                    Pos2::new(screen[0] + 6.0, screen[1] + 6.0),
                    egui::Align2::LEFT_TOP,
                    pt_name,
                    egui::FontId::proportional(9.0),
                    dimmed,
                );
            }
        }
    }

    // Draw ground markers at ground attachment points
    if let Some(ground) = mech.bodies().get(GROUND_ID) {
        for (_pt_name, pt_local) in &ground.attachment_points {
            let screen = state.view.world_to_screen(pt_local.x, pt_local.y);
            let pos = Pos2::new(screen[0], screen[1]);
            draw_ground_triangle(&painter, pos);
        }
    }

    // Draw joints
    for joint in mech.joints() {
        let joint_id = joint.id().to_string();
        let body_i = joint.body_i_id();
        let pt_i_local = joint.point_i_local();
        let global_pos = mech_state.body_point_global(body_i, &pt_i_local, q);
        let screen = state.view.world_to_screen(global_pos.x, global_pos.y);
        let pos = Pos2::new(screen[0], screen[1]);

        let is_selected = state.selected == Some(SelectedEntity::Joint(joint_id.clone()));
        let color = if is_selected { JOINT_SELECTED_COLOR } else { JOINT_COLOR };

        if joint.is_revolute() {
            painter.circle_stroke(pos, JOINT_RADIUS, Stroke::new(2.0, color));
        } else if joint.is_prismatic() {
            // Prismatic: small square
            let half = JOINT_RADIUS;
            painter.rect_stroke(
                egui::Rect::from_center_size(pos, Vec2::splat(half * 2.0)),
                0.0,
                Stroke::new(2.0, color),
            );
        } else {
            // Fixed: filled circle
            painter.circle_filled(pos, JOINT_RADIUS * 0.8, color);
        }

        // Debug overlay: joint ID
        if state.show_debug_overlay {
            painter.text(
                Pos2::new(pos.x + 8.0, pos.y - 8.0),
                egui::Align2::LEFT_BOTTOM,
                &joint_id,
                egui::FontId::proportional(10.0),
                DEBUG_TEXT_COLOR,
            );
        }
    }

    // Draw solver status indicator
    if state.show_debug_overlay {
        let status_color = if state.solver_status.converged {
            Color32::from_rgb(80, 200, 80)
        } else {
            Color32::from_rgb(200, 60, 60)
        };
        painter.circle_filled(
            Pos2::new(rect.right() - 15.0, rect.top() + 15.0),
            6.0,
            status_color,
        );
    }

    // Handle click for selection
    if response.clicked() {
        if let Some(click_pos) = response.interact_pointer_pos() {
            state.selected = hit_test(mech, mech_state, q, &state.view, click_pos);
        }
    }
}

/// Draw a ground-pivot triangle marker.
fn draw_ground_triangle(painter: &egui::Painter, center: Pos2) {
    let s = GROUND_TRIANGLE_SIZE;
    let points = vec![
        Pos2::new(center.x, center.y - s * 0.5),
        Pos2::new(center.x - s * 0.6, center.y + s * 0.5),
        Pos2::new(center.x + s * 0.6, center.y + s * 0.5),
    ];
    painter.add(egui::Shape::convex_polygon(
        points,
        GROUND_COLOR,
        Stroke::new(1.0, GROUND_COLOR),
    ));
    // Hatch lines below triangle
    let y_base = center.y + s * 0.5;
    for i in 0..4 {
        let x_off = (i as f32 - 1.5) * 4.0;
        painter.line_segment(
            [Pos2::new(center.x + x_off, y_base),
             Pos2::new(center.x + x_off - 3.0, y_base + 5.0)],
            Stroke::new(1.0, GROUND_COLOR),
        );
    }
}

/// Hit-test: find which entity (if any) is under the click position.
fn hit_test(
    mech: &crate::core::mechanism::Mechanism,
    mech_state: &crate::core::state::State,
    q: &nalgebra::DVector<f64>,
    view: &super::state::ViewTransform,
    click_pos: Pos2,
) -> Option<SelectedEntity> {
    // Check joints first (smaller targets, higher priority)
    for joint in mech.joints() {
        let body_i = joint.body_i_id();
        let pt_i_local = joint.point_i_local();
        let global = mech_state.body_point_global(body_i, &pt_i_local, q);
        let screen = view.world_to_screen(global.x, global.y);
        let dist = ((click_pos.x - screen[0]).powi(2) + (click_pos.y - screen[1]).powi(2)).sqrt();
        if dist < HIT_RADIUS {
            return Some(SelectedEntity::Joint(joint.id().to_string()));
        }
    }

    // Check bodies (check each attachment point)
    for (body_id, body) in mech.bodies() {
        if body_id == GROUND_ID { continue; }
        for (_pt_name, pt_local) in &body.attachment_points {
            let global = mech_state.body_point_global(body_id, pt_local, q);
            let screen = view.world_to_screen(global.x, global.y);
            let dist = ((click_pos.x - screen[0]).powi(2) + (click_pos.y - screen[1]).powi(2)).sqrt();
            if dist < HIT_RADIUS * 1.5 {
                return Some(SelectedEntity::Body(body_id.clone()));
            }
        }
    }

    None
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd linkage-sim-rs && cargo build --bin linkage-gui`
Expected: Compiles without errors.

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat(gui): add 2D canvas with body/joint/ground rendering and hit testing"
```

---

## Task 5: Input panel — angle slider and solver integration

**Files:**
- Modify: `linkage-sim-rs/src/gui/input_panel.rs`

- [ ] **Step 1: Implement input panel**

Replace `linkage-sim-rs/src/gui/input_panel.rs` with:

```rust
//! Angle slider and solver status display.

use eframe::egui;
use super::state::AppState;

/// Draw the input panel (bottom of canvas or left panel section).
pub fn draw_input_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if !state.has_mechanism() {
        return;
    }

    ui.separator();
    ui.heading("Driver Input");

    let mut angle_deg = state.driver_angle.to_degrees();
    let prev_angle = angle_deg;

    ui.horizontal(|ui| {
        ui.label("Crank angle:");
        ui.add(
            egui::Slider::new(&mut angle_deg, 0.0..=360.0)
                .suffix("°")
                .step_by(0.5),
        );
    });

    // Re-solve if angle changed
    if (angle_deg - prev_angle).abs() > 1e-6 {
        state.solve_at_angle(angle_deg.to_radians());
    }

    // Solver status
    ui.separator();
    let status = &state.solver_status;
    ui.horizontal(|ui| {
        let (icon, color) = if status.converged {
            ("●", egui::Color32::from_rgb(80, 200, 80))
        } else {
            ("●", egui::Color32::from_rgb(200, 60, 60))
        };
        ui.colored_label(color, icon);
        if status.converged {
            ui.label(format!(
                "Converged in {} iters (‖Φ‖ = {:.2e})",
                status.iterations, status.residual_norm
            ));
        } else {
            ui.label(format!(
                "FAILED — ‖Φ‖ = {:.2e} (showing last good pose)",
                status.residual_norm
            ));
        }
    });
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd linkage-sim-rs && cargo build --bin linkage-gui`
Expected: Compiles without errors.

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/input_panel.rs
git commit -m "feat(gui): add angle slider with solver integration"
```

---

## Task 6: Property panel — read-only display for selected entity

**Files:**
- Modify: `linkage-sim-rs/src/gui/property_panel.rs`

- [ ] **Step 1: Implement property panel**

Replace `linkage-sim-rs/src/gui/property_panel.rs` with:

```rust
//! Read-only property panel for the selected entity.

use eframe::egui;
use crate::core::constraint::Constraint;
use crate::core::state::GROUND_ID;
use super::state::{AppState, SelectedEntity};

/// Draw the property panel showing info about the selected entity.
pub fn draw_property_panel(ui: &mut egui::Ui, state: &AppState) {
    ui.heading("Properties");

    let Some(mech) = &state.mechanism else {
        ui.label("No mechanism loaded.");
        return;
    };

    let Some(selected) = &state.selected else {
        ui.label("Click a body or joint to inspect.");
        return;
    };

    match selected {
        SelectedEntity::Body(body_id) => {
            if let Some(body) = mech.bodies().get(body_id) {
                ui.strong(format!("Body: {}", body_id));
                ui.separator();

                let mech_state = mech.state();
                let q = &state.q;

                if body_id != GROUND_ID {
                    let (x, y, theta) = mech_state.get_pose(body_id, q);
                    ui.label(format!("Position: ({:.4}, {:.4}) m", x, y));
                    ui.label(format!("Angle: {:.2}°", theta.to_degrees()));
                } else {
                    ui.label("Position: (0, 0) — fixed");
                    ui.label("Angle: 0° — fixed");
                }

                ui.separator();
                ui.label(format!("Mass: {:.4} kg", body.mass));
                ui.label(format!("Izz_cg: {:.6} kg·m²", body.izz_cg));
                ui.label(format!("CG local: ({:.4}, {:.4})", body.cg_local.x, body.cg_local.y));

                ui.separator();
                ui.strong("Attachment points:");
                let mut pts: Vec<_> = body.attachment_points.iter().collect();
                pts.sort_by_key(|(name, _)| name.clone());
                for (name, local) in pts {
                    let global = mech_state.body_point_global(body_id, local, q);
                    ui.label(format!(
                        "  {} — local: ({:.4}, {:.4}), global: ({:.4}, {:.4})",
                        name, local.x, local.y, global.x, global.y
                    ));
                }
            }
        }
        SelectedEntity::Joint(joint_id) => {
            if let Some(joint) = mech.joints().iter().find(|j| j.id() == joint_id) {
                let joint_type = if joint.is_revolute() {
                    "Revolute"
                } else if joint.is_prismatic() {
                    "Prismatic"
                } else {
                    "Fixed"
                };

                ui.strong(format!("Joint: {}", joint_id));
                ui.label(format!("Type: {}", joint_type));
                ui.separator();

                ui.label(format!("Body i: {}", joint.body_i_id()));
                ui.label(format!("Body j: {}", joint.body_j_id()));
                ui.label(format!("DOF removed: {}", joint.dof_removed()));
                ui.label(format!("Equations: {}", joint.n_equations()));

                ui.separator();
                let mech_state = mech.state();
                let q = &state.q;
                let global_pos = mech_state.body_point_global(
                    joint.body_i_id(),
                    &joint.point_i_local(),
                    q,
                );
                ui.label(format!("Position: ({:.4}, {:.4}) m", global_pos.x, global_pos.y));
            }
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd linkage-sim-rs && cargo build --bin linkage-gui`
Expected: Compiles without errors.

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/property_panel.rs
git commit -m "feat(gui): add read-only property panel for selected entities"
```

---

## Task 7: App shell layout — menu bar, panels, and wiring

Wire everything together into the final app layout: menu bar with Load Sample, left panel (properties + input), central canvas, status bar.

**Files:**
- Modify: `linkage-sim-rs/src/gui/mod.rs`

- [ ] **Step 1: Replace gui/mod.rs with full app shell**

```rust
//! GUI module — egui-based visualization shell for the linkage simulator.

mod state;
mod canvas;
mod input_panel;
mod property_panel;
pub mod samples;

use eframe::egui;
pub use state::AppState;
use samples::SampleMechanism;

/// Top-level application struct for eframe.
pub struct LinkageApp {
    state: AppState,
}

impl LinkageApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            state: AppState::default(),
        }
    }
}

impl eframe::App for LinkageApp {
    fn update(&mut self, ctx: &egui::Context) {
        // --- Menu bar ---
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    ui.menu_button("Load Sample", |ui| {
                        for sample in SampleMechanism::all() {
                            if ui.button(sample.label()).clicked() {
                                self.state.load_sample(*sample);
                                ui.close_menu();
                            }
                        }
                    });
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.state.show_debug_overlay, "Debug Overlay");
                });
            });
        });

        // --- Status bar ---
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(sample) = self.state.current_sample {
                    ui.label(format!("Mechanism: {}", sample.label()));
                    ui.separator();
                }

                let status = &self.state.solver_status;
                if self.state.has_mechanism() {
                    let color = if status.converged {
                        egui::Color32::from_rgb(80, 200, 80)
                    } else {
                        egui::Color32::from_rgb(200, 60, 60)
                    };
                    ui.colored_label(color, "●");
                    ui.label(format!("‖Φ‖ = {:.2e}", status.residual_norm));
                    ui.separator();
                    ui.label(format!("θ = {:.1}°", self.state.driver_angle.to_degrees()));

                    if let Some(mech) = &self.state.mechanism {
                        ui.separator();
                        ui.label(format!(
                            "Bodies: {} | Joints: {} | DOF: {}",
                            mech.bodies().len() - 1, // exclude ground
                            mech.joints().len(),
                            mech.state().n_coords() as isize - mech.n_constraints() as isize
                        ));
                    }
                } else {
                    ui.label("No mechanism loaded");
                }
            });
        });

        // --- Left panel: properties + input ---
        egui::SidePanel::left("left_panel")
            .default_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    property_panel::draw_property_panel(ui, &self.state);
                    ui.add_space(20.0);
                    input_panel::draw_input_panel(ui, &mut self.state);
                });
            });

        // --- Central canvas ---
        egui::CentralPanel::default().show(ctx, |ui| {
            canvas::draw_canvas(ui, &mut self.state);
        });
    }
}
```

- [ ] **Step 2: Build and run the complete app**

Run: `cd linkage-sim-rs && cargo build --bin linkage-gui`
Expected: Compiles without errors.

Run: `cd linkage-sim-rs && cargo run --bin linkage-gui`
Expected:
- Window opens with menu bar, left panel, canvas, status bar
- File > Load Sample > 4-Bar Crank-Rocker loads and renders the mechanism
- Slider changes crank angle, mechanism animates
- Click on body/joint shows properties in left panel
- Middle-click-drag or Shift+drag pans, scroll wheel zooms
- View > Debug Overlay toggles ID labels
- Status bar shows solver residual and angle

- [ ] **Step 3: Test with slider-crank too**

Load File > Load Sample > Slider-Crank.
Expected: Renders with prismatic joint (square marker), slider moves along x-axis.

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/gui/mod.rs
git commit -m "feat(gui): wire app shell with menu, panels, canvas, and status bar"
```

---

## Task 8: Final verification and cleanup

- [ ] **Step 1: Run all tests**

Run: `cd linkage-sim-rs && cargo test`
Expected: All tests pass (solver tests + gui tests).

- [ ] **Step 2: Run clippy**

Run: `cd linkage-sim-rs && cargo clippy --all-targets -- -D warnings`
Expected: No warnings.

- [ ] **Step 3: Verify Definition of Done checklist**

From the spec, verify each item:

- [x] `egui`, `eframe` added to Cargo.toml
- [x] App launches with menu bar, side panel, canvas area, status bar
- [x] At least one benchmark mechanism (4-bar) loads from hardcoded builder function via File > Load Sample
- [x] Slider changes input angle; solver updates pose live
- [x] Canvas renders mechanism from solved world-space poses
- [x] Selection model works: click body/joint, property panel shows read-only data
- [x] Pan and zoom stable on canvas
- [x] Debug overlay toggleable (IDs, solver status)
- [x] Ground rendering follows defined rule (triangles at fixed pivots)
- [x] Solver failure shows last good pose + failure indicator (does not blank canvas)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(gui): Phase 5 MVP — egui visualization shell complete"
```
