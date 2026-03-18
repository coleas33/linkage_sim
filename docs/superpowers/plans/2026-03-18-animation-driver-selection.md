# Animation Playback + Revolute Driver Reassignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add continuous animation (play/pause/speed/loop) and right-click driver reassignment to the linkage simulator GUI.

**Architecture:** Animation state lives in `AppState`. Frame stepping runs in `mod.rs::update()` before panels render. Context menu on canvas uses a pending-action pattern (collect during immutable render pass, execute after). Driver reassignment rebuilds the mechanism via a flexible sample builder.

**Tech Stack:** Rust, egui 0.33 / eframe 0.33, existing linkage-sim-rs solver

**Spec:** `docs/superpowers/specs/2026-03-18-animation-driver-selection-design.md`

---

## File Structure

### Modified files

| File | Changes |
|------|---------|
| `src/core/mechanism.rs` | Add `grounded_revolute_joint_ids()`, `driver_body_pair()` |
| `src/gui/state.rs` | Add animation fields, `reassign_driver()`, `step_animation()` |
| `src/gui/samples.rs` | Add `build_sample_with_driver()` |
| `src/gui/input_panel.rs` | Replace static slider with playback controls |
| `src/gui/canvas.rs` | Add right-click context menu with pending-action pattern |
| `src/gui/mod.rs` | Add animation stepping + pending action processing in update loop |

---

## Task 1: Mechanism API — grounded revolute IDs and driver body pair

**Files:**
- Modify: `linkage-sim-rs/src/core/mechanism.rs`

- [ ] **Step 1: Add `grounded_revolute_joint_ids()` and `driver_body_pair()` to Mechanism**

In `linkage-sim-rs/src/core/mechanism.rs`, add these methods inside the `impl Mechanism` block (after `n_drivers()`):

```rust
    /// Return IDs of revolute joints where one body is ground.
    /// These are valid candidates for driver reassignment.
    pub fn grounded_revolute_joint_ids(&self) -> Vec<String> {
        self.joints
            .iter()
            .filter(|j| {
                j.is_revolute()
                    && (j.body_i_id() == GROUND_ID || j.body_j_id() == GROUND_ID)
            })
            .map(|j| j.id().to_string())
            .collect()
    }

    /// Return the (body_i, body_j) pair of the first driver, if any.
    /// Used to check if a grounded revolute joint is already the driven joint.
    pub fn driver_body_pair(&self) -> Option<(&str, &str)> {
        self.drivers.first().map(|d| (d.body_i_id(), d.body_j_id()))
    }
```

- [ ] **Step 2: Add tests**

Add these tests to the existing `#[cfg(test)] mod tests` block in mechanism.rs:

```rust
    #[test]
    fn grounded_revolute_joint_ids_fourbar() {
        let mech = build_fourbar();
        let ids = mech.grounded_revolute_joint_ids();
        // J1 connects ground-crank, J4 connects rocker-ground
        assert!(ids.contains(&"J1".to_string()));
        assert!(ids.contains(&"J4".to_string()));
        assert_eq!(ids.len(), 2);
        // J2, J3 are not grounded
        assert!(!ids.contains(&"J2".to_string()));
    }

    #[test]
    fn driver_body_pair_returns_correct_pair() {
        let mech = build_fourbar();
        let pair = mech.driver_body_pair();
        assert_eq!(pair, Some(("ground", "crank")));
    }
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs && cargo test core::mechanism`
Expected: All tests pass including 2 new ones.

- [ ] **Step 4: Commit**

```bash
cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs
git add src/core/mechanism.rs
git commit -m "feat(rs): add grounded_revolute_joint_ids and driver_body_pair to Mechanism

These accessors support GUI driver reassignment. grounded_revolute_joint_ids
returns joint IDs eligible for driving. driver_body_pair returns the current
driver's body pair for checking if a joint is already driven.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Flexible sample builder — `build_sample_with_driver`

**Files:**
- Modify: `linkage-sim-rs/src/gui/samples.rs`

- [ ] **Step 1: Refactor sample builders to separate body/joint setup from driver attachment**

The key insight: each sample builder currently hardcodes the driver. We need to split this into two steps: (1) build bodies + joints (no driver), (2) attach driver to a specified joint.

Add this function to `samples.rs`:

```rust
/// Build a sample mechanism with the driver on a specific joint.
///
/// The joint must be a grounded revolute joint. If `driver_joint_id` is None,
/// uses the default driver joint for the sample.
///
/// The driver is a constant-speed revolute driver (omega=2*pi, theta_0=0).
/// The joint-to-body-pair mapping: the non-ground body of the joint becomes
/// the driven body.
pub fn build_sample_with_driver(
    sample: SampleMechanism,
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    match sample {
        SampleMechanism::FourBar => build_fourbar_with_driver(driver_joint_id),
        SampleMechanism::SliderCrank => build_slider_crank_with_driver(driver_joint_id),
    }
}

fn build_fourbar_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
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

    // Attach driver to the specified joint (or default J1).
    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute(&mut mech, joint_id, "D1")?;

    mech.build().unwrap();

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
    state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
    state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);

    Ok((mech, q0))
}

fn build_slider_crank_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("rail", 0.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
    let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
    let mut slider = Body::new("slider");
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

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute(&mut mech, joint_id, "D1")?;

    mech.build().unwrap();

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
    state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
    state.set_pose("slider", &mut q0, 0.05, 0.0, 0.0);

    Ok((mech, q0))
}

/// Attach a constant-speed driver to a grounded revolute joint.
///
/// Finds the joint by ID, checks it is revolute and grounded, then adds
/// the driver with the non-ground body as the driven body.
fn attach_driver_to_grounded_revolute(
    mech: &mut Mechanism,
    joint_id: &str,
    driver_id: &str,
) -> Result<(), String> {
    // Find the joint in the mechanism's joints list.
    let joint = mech.joints().iter()
        .find(|j| j.id() == joint_id)
        .ok_or_else(|| format!("Joint '{}' not found", joint_id))?;

    if !joint.is_revolute() {
        return Err(format!("Joint '{}' is not revolute", joint_id));
    }

    let body_i = joint.body_i_id().to_string();
    let body_j = joint.body_j_id().to_string();

    let (ground_id, driven_id) = if body_i == GROUND_ID {
        (body_i, body_j)
    } else if body_j == GROUND_ID {
        (body_j, body_i)
    } else {
        return Err(format!("Joint '{}' is not grounded", joint_id));
    };

    mech.add_constant_speed_driver(driver_id, &ground_id, &driven_id, 2.0 * PI, 0.0)
        .map_err(|e| format!("Failed to add driver: {}", e))
}
```

Also add `use crate::core::state::GROUND_ID;` to the imports at the top of `samples.rs`.

Also update `build_sample` to delegate to `build_sample_with_driver`:

```rust
pub fn build_sample(sample: SampleMechanism) -> (Mechanism, DVector<f64>) {
    build_sample_with_driver(sample, None).expect("Default sample build should never fail")
}
```

Remove the old `build_fourbar_sample` and `build_slider_crank_sample` functions (they are replaced by the `_with_driver` versions). Update any tests that call them directly.

- [ ] **Step 2: Add tests**

```rust
    #[test]
    fn fourbar_with_alternate_driver() {
        // Drive from J4 (rocker-ground) instead of default J1 (ground-crank)
        let (mech, q0) = build_sample_with_driver(SampleMechanism::FourBar, Some("J4")).unwrap();
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(result.converged, "residual = {}", result.residual_norm);
        // Verify the driver is on ground-rocker (not ground-crank)
        assert_eq!(mech.driver_body_pair(), Some(("ground", "rocker")));
    }

    #[test]
    fn build_with_invalid_joint_errors() {
        let result = build_sample_with_driver(SampleMechanism::FourBar, Some("J2"));
        assert!(result.is_err()); // J2 is not grounded
    }

    #[test]
    fn build_with_nonexistent_joint_errors() {
        let result = build_sample_with_driver(SampleMechanism::FourBar, Some("NOPE"));
        assert!(result.is_err());
    }
```

- [ ] **Step 3: Run tests**

Run: `cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs && cargo test gui::samples`
Expected: All tests pass (original 3 + 3 new).

- [ ] **Step 4: Commit**

```bash
cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs
git add src/gui/samples.rs
git commit -m "feat(gui): add flexible sample builder with driver reassignment

build_sample_with_driver() constructs samples with driver on any grounded
revolute joint. Uses joint-to-body-pair mapping: non-ground body of the
joint becomes the driven body. Errors on non-revolute or non-grounded joints.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Animation and driver state in AppState

**Files:**
- Modify: `linkage-sim-rs/src/gui/state.rs`

- [ ] **Step 1: Add animation fields to AppState**

Add these fields to the `AppState` struct (after `driver_theta_0`):

```rust
    // ── Animation state ──────────────────────────────────────────────────────
    /// Whether the animation is currently playing.
    pub playing: bool,
    /// Animation speed in degrees per second.
    pub animation_speed_deg_per_sec: f64,
    /// True = continuous loop (ping-pong at failure), false = sweep once then stop.
    pub loop_mode: bool,
    /// Current sweep direction: +1.0 (forward) or -1.0 (backward, during ping-pong).
    pub animation_direction: f64,

    // ── Driver reassignment state ────────────────────────────────────────────
    /// Which joint ID is currently driven (cache for UI display).
    pub driver_joint_id: Option<String>,
    /// Pending driver reassignment from canvas context menu.
    /// Set by the context menu, consumed in the update loop.
    pub pending_driver_reassignment: Option<String>,
```

Update the `Default` impl to include these:

```rust
            playing: false,
            animation_speed_deg_per_sec: 90.0,
            loop_mode: true,
            animation_direction: 1.0,
            driver_joint_id: None,
            pending_driver_reassignment: None,
```

- [ ] **Step 2: Update `load_sample` to set `driver_joint_id`**

In the `load_sample` method, after `self.current_sample = Some(sample);`, add:

```rust
        // Cache the default driver joint ID.
        if let Some(mech) = &self.mechanism {
            let grounded = mech.grounded_revolute_joint_ids();
            if let Some(pair) = mech.driver_body_pair() {
                // Find the joint whose body pair matches the driver's
                self.driver_joint_id = mech.joints().iter()
                    .find(|j| {
                        j.is_revolute()
                            && ((j.body_i_id() == pair.0 && j.body_j_id() == pair.1)
                                || (j.body_i_id() == pair.1 && j.body_j_id() == pair.0))
                    })
                    .map(|j| j.id().to_string());
            }
        }
        // Reset animation state
        self.playing = false;
        self.animation_direction = 1.0;
```

- [ ] **Step 3: Add `reassign_driver` method**

```rust
    /// Rebuild the current mechanism with a different driver joint.
    /// Resets animation state and solves at angle 0.
    pub fn reassign_driver(&mut self, joint_id: &str) {
        let Some(sample) = self.current_sample else { return };

        match crate::gui::samples::build_sample_with_driver(sample, Some(joint_id)) {
            Ok((mech, q0)) => {
                self.driver_omega = 2.0 * std::f64::consts::PI;
                self.driver_theta_0 = 0.0;
                self.driver_angle = 0.0;
                self.playing = false;
                self.animation_direction = 1.0;
                self.driver_joint_id = Some(joint_id.to_string());
                self.selected = None;

                match solve_position(&mech, &q0, 0.0, 1e-10, 50) {
                    Ok(result) => {
                        self.solver_status = SolverStatus {
                            converged: result.converged,
                            residual_norm: result.residual_norm,
                            iterations: result.iterations,
                        };
                        if result.converged {
                            self.q = result.q.clone();
                            self.last_good_q = result.q;
                        } else {
                            self.q = q0.clone();
                            self.last_good_q = q0;
                        }
                    }
                    Err(_) => {
                        self.q = q0.clone();
                        self.last_good_q = q0;
                        self.solver_status = SolverStatus {
                            converged: false,
                            residual_norm: f64::NAN,
                            iterations: 0,
                        };
                    }
                }

                self.mechanism = Some(mech);
            }
            Err(msg) => {
                log::warn!("Driver reassignment failed: {}", msg);
            }
        }
    }

    /// Advance the animation by one frame. Called from the update loop.
    /// Returns true if the animation is active (caller should request repaint).
    pub fn step_animation(&mut self, dt: f64) -> bool {
        if !self.playing || !self.has_mechanism() {
            return false;
        }

        let step = self.animation_speed_deg_per_sec * dt * self.animation_direction;
        let mut new_angle = self.driver_angle.to_degrees() + step;

        // Wrap or clamp based on loop mode
        if self.loop_mode {
            if new_angle >= 360.0 {
                new_angle -= 360.0;
            } else if new_angle < 0.0 {
                new_angle += 360.0;
            }
        } else {
            // Once mode: always forward, stop at 360
            if new_angle >= 360.0 {
                new_angle = 360.0;
                self.playing = false;
            }
        }

        let prev_converged = self.solver_status.converged;
        self.solve_at_angle(new_angle.to_radians());

        // If solver failed and we're looping, reverse direction (ping-pong)
        if self.loop_mode && !self.solver_status.converged && prev_converged {
            self.animation_direction *= -1.0;
        }

        // If solver failed and we're in once mode, stop
        if !self.loop_mode && !self.solver_status.converged {
            self.playing = false;
        }

        self.playing
    }
```

- [ ] **Step 4: Add tests**

```rust
    #[test]
    fn step_animation_advances_angle() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.playing = true;
        state.animation_speed_deg_per_sec = 180.0; // 180 deg/s

        let dt = 1.0 / 60.0; // 60fps
        let angle_before = state.driver_angle;
        let still_playing = state.step_animation(dt);

        assert!(still_playing);
        assert!(state.driver_angle > angle_before);
    }

    #[test]
    fn step_animation_noop_when_paused() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.playing = false;

        let angle_before = state.driver_angle;
        let still_playing = state.step_animation(1.0 / 60.0);

        assert!(!still_playing);
        assert_eq!(state.driver_angle, angle_before);
    }

    #[test]
    fn reassign_driver_rebuilds_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));

        state.reassign_driver("J4");
        assert_eq!(state.driver_joint_id, Some("J4".to_string()));
        assert!(state.solver_status.converged);
        assert_eq!(state.driver_angle, 0.0);
        assert!(!state.playing);
    }
```

- [ ] **Step 5: Run tests**

Run: `cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs && cargo test gui::state`
Expected: All state tests pass (4 existing + 3 new).

- [ ] **Step 6: Commit**

```bash
cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs
git add src/gui/state.rs
git commit -m "feat(gui): add animation state, step_animation, and reassign_driver

AppState gains playing/speed/loop/direction fields for animation,
step_animation() for frame-by-frame advancement with ping-pong on
solver failure, and reassign_driver() to rebuild with a new driver joint.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Input panel — playback controls

**Files:**
- Modify: `linkage-sim-rs/src/gui/input_panel.rs`

- [ ] **Step 1: Replace input panel with playback controls**

Replace the entire contents of `linkage-sim-rs/src/gui/input_panel.rs`:

```rust
//! Angle slider, playback controls, and solver status display.

use eframe::egui;
use super::state::AppState;

/// Draw the input panel with animation controls.
pub fn draw_input_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if !state.has_mechanism() {
        return;
    }

    ui.separator();
    ui.heading("Driver Input");

    // ── Playback controls ────────────────────────────────────────────
    ui.horizontal(|ui| {
        // Play / Pause button
        let play_label = if state.playing { "Pause" } else { "Play" };
        if ui.button(play_label).clicked() {
            state.playing = !state.playing;
            if state.playing {
                // Reset direction when starting (unless already ping-ponging)
                if !state.loop_mode {
                    state.animation_direction = 1.0;
                }
            }
        }

        // Speed slider
        ui.label("Speed:");
        ui.add(
            egui::Slider::new(&mut state.animation_speed_deg_per_sec, 10.0..=720.0)
                .suffix(" deg/s")
                .logarithmic(true)
                .clamp_to_range(true),
        );
    });

    ui.horizontal(|ui| {
        // Loop / Once toggle
        let mode_label = if state.loop_mode { "Loop" } else { "Once" };
        if ui.button(mode_label).clicked() {
            state.loop_mode = !state.loop_mode;
            state.animation_direction = 1.0; // Reset direction on mode switch
        }

        if state.loop_mode {
            ui.label("(continuous, ping-pong at limits)");
        } else {
            ui.label("(sweep forward, stop at 360)");
        }
    });

    // ── Angle slider ─────────────────────────────────────────────────
    let mut angle_deg = state.driver_angle.to_degrees();
    let prev_angle = angle_deg;

    ui.horizontal(|ui| {
        ui.label("Crank angle:");
        let response = ui.add(
            egui::Slider::new(&mut angle_deg, 0.0..=360.0)
                .suffix("\u{00B0}")
                .step_by(0.5),
        );

        // Dragging the slider pauses animation and resets direction
        if response.dragged() && state.playing {
            state.playing = false;
            state.animation_direction = 1.0;
        }
    });

    // Re-solve if angle changed by manual slider drag
    if (angle_deg - prev_angle).abs() > 1e-6 {
        state.solve_at_angle(angle_deg.to_radians());
    }

    // ── Solver status ────────────────────────────────────────────────
    ui.separator();
    let status = &state.solver_status;
    ui.horizontal(|ui| {
        let color = if status.converged {
            egui::Color32::from_rgb(80, 200, 80)
        } else {
            egui::Color32::from_rgb(200, 60, 60)
        };
        ui.colored_label(color, "\u{25CF}"); // ●
        if status.converged {
            ui.label(format!(
                "Converged in {} iters (r = {:.2e})",
                status.iterations, status.residual_norm
            ));
        } else {
            ui.label(format!(
                "FAILED (r = {:.2e}) — last good pose",
                status.residual_norm
            ));
        }
    });

    // ── Driver info ──────────────────────────────────────────────────
    if let Some(joint_id) = &state.driver_joint_id {
        ui.label(format!("Driver: {} (right-click joint to change)", joint_id));
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs && cargo build --bin linkage-gui`

- [ ] **Step 3: Commit**

```bash
cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs
git add src/gui/input_panel.rs
git commit -m "feat(gui): add playback controls (play/pause, speed, loop/once)

Replace static slider with animation controls. Play/pause toggles
continuous animation. Speed slider (10-720 deg/s, logarithmic).
Loop/once toggle. Dragging slider pauses animation. Shows current
driver joint ID with hint to right-click for reassignment.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Canvas context menu — right-click driver reassignment

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs`

- [ ] **Step 1: Add context menu using pending-action pattern**

In `canvas.rs`, the context menu needs to be added after the immutable rendering scope but using data collected during it. Add these changes:

1. During the immutable render pass (inside the `{ ... }` scope at line 72), collect right-click target info. Add this alongside the existing `joint_hit_targets` collection:

```rust
    // Also collect data for context menu: which joints are grounded revolute, which is driven.
    let grounded_revolute_ids: Vec<String> = mech.grounded_revolute_joint_ids();
    let driver_pair = mech.driver_body_pair().map(|(a, b)| (a.to_string(), b.to_string()));
```

2. After the immutable scope ends (after line 218), before the pan/zoom interaction code, add context menu handling:

```rust
    // ── Interaction: right-click context menu ────────────────────────
    // Determine what joint (if any) was right-clicked.
    let right_click_joint: Option<String> = if response.secondary_clicked() {
        if let Some(pointer_pos) = response.interact_pointer_pos() {
            joint_hit_targets.iter()
                .find(|(screen_pos, _)| pointer_pos.distance(*screen_pos) <= HIT_RADIUS)
                .map(|(_, id)| id.clone())
        } else {
            None
        }
    } else {
        None
    };

    // Show context menu if a joint was right-clicked.
    // The context menu closure does NOT mutate state directly.
    // Instead it sets state.pending_driver_reassignment, which is
    // consumed in mod.rs after all panels render.
    if let Some(ref joint_id) = right_click_joint {
        let is_grounded_revolute = grounded_revolute_ids.contains(joint_id);
        let is_current_driver = state.driver_joint_id.as_deref() == Some(joint_id.as_str());
        let joint_id_owned = joint_id.clone();

        // Use egui's popup mechanism
        let popup_id = ui.id().with("joint_context_menu");
        ui.memory_mut(|mem| mem.open_popup(popup_id));
        egui::Area::new(popup_id)
            .order(egui::Order::Foreground)
            .fixed_pos(response.interact_pointer_pos().unwrap_or_default())
            .show(ui.ctx(), |ui| {
                ui.set_min_width(160.0);
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.label(format!("Joint: {}", joint_id_owned));
                    ui.separator();
                    if is_grounded_revolute {
                        let label = if is_current_driver {
                            "Set as Driver (current)"
                        } else {
                            "Set as Driver"
                        };
                        let btn = ui.add_enabled(!is_current_driver, egui::Button::new(label));
                        if btn.clicked() {
                            state.pending_driver_reassignment = Some(joint_id_owned.clone());
                            ui.memory_mut(|mem| mem.close_popup());
                        }
                    } else {
                        ui.label("(not a grounded revolute joint)");
                    }
                });
            });
    }
```

Note: The exact egui popup/context menu API may vary slightly. The implementer should check if `response.context_menu(|ui| { ... })` works more cleanly for this use case in egui 0.33. The key requirement is that `state.pending_driver_reassignment` is set inside the menu, and the actual rebuild happens later.

- [ ] **Step 2: Verify it compiles**

Run: `cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs && cargo build --bin linkage-gui`

If the popup API is different in egui 0.33, adapt using `response.context_menu(|ui| { ... })` which is the more idiomatic approach. The important thing is the pending-action pattern.

- [ ] **Step 3: Commit**

```bash
cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs
git add src/gui/canvas.rs
git commit -m "feat(gui): add right-click context menu for driver reassignment

Right-click a joint to see context menu. Grounded revolute joints
show 'Set as Driver' option. Current driver is grayed out. Uses
pending-action pattern: menu sets pending_driver_reassignment,
actual rebuild happens in mod.rs update loop.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Wire animation stepping and pending actions in mod.rs

**Files:**
- Modify: `linkage-sim-rs/src/gui/mod.rs`

- [ ] **Step 1: Add animation stepping and pending action processing to update()**

In `mod.rs`, add animation stepping at the **top** of `update()` (before any panels render), and pending action processing at the **bottom** (after all panels render):

Replace the entire `update` method:

```rust
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── Animation stepping (before rendering) ────────────────────
        let dt = ctx.input(|i| i.stable_dt) as f64;
        if self.state.step_animation(dt) {
            ctx.request_repaint(); // Keep animation loop running
        }

        // ── Process pending driver reassignment ──────────────────────
        if let Some(joint_id) = self.state.pending_driver_reassignment.take() {
            self.state.reassign_driver(&joint_id);
        }

        // --- Menu bar ---
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    ui.menu_button("Load Sample", |ui| {
                        for sample in SampleMechanism::all() {
                            if ui.button(sample.label()).clicked() {
                                self.state.load_sample(*sample);
                                ui.close();
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

                if self.state.has_mechanism() {
                    let status = &self.state.solver_status;
                    let color = if status.converged {
                        egui::Color32::from_rgb(80, 200, 80)
                    } else {
                        egui::Color32::from_rgb(200, 60, 60)
                    };
                    ui.colored_label(color, "\u{25CF}");
                    ui.label(format!("\u{2016}\u{03A6}\u{2016} = {:.2e}", status.residual_norm));
                    ui.separator();
                    ui.label(format!("\u{03B8} = {:.1}\u{00B0}", self.state.driver_angle.to_degrees()));

                    if self.state.playing {
                        ui.separator();
                        ui.colored_label(egui::Color32::from_rgb(80, 200, 80), "PLAYING");
                    }

                    if let Some(mech) = &self.state.mechanism {
                        ui.separator();
                        ui.label(format!(
                            "Bodies: {} | Joints: {} | DOF: {}",
                            mech.bodies().len().saturating_sub(1),
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
```

- [ ] **Step 2: Build and run**

Run: `cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs && cargo build --bin linkage-gui && cargo test`

Verify manually:
1. Load 4-bar sample
2. Click Play — mechanism animates continuously
3. Adjust speed slider — animation speeds up/slows down
4. Toggle Loop/Once — once mode sweeps and stops
5. Drag slider — pauses animation
6. Right-click J4 → Set as Driver → mechanism rebuilds, rocker becomes input
7. Right-click J1 → Set as Driver → mechanism rebuilds back to crank input
8. Load slider-crank — only J1 is grounded revolute, no reassignment possible

- [ ] **Step 3: Commit**

```bash
cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs
git add src/gui/mod.rs
git commit -m "feat(gui): wire animation stepping and driver reassignment in update loop

Animation steps each frame before rendering. Pending driver reassignment
from canvas context menu is consumed at top of update loop. Status bar
shows PLAYING indicator during animation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Final verification and cleanup

- [ ] **Step 1: Run all tests**

Run: `cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs && cargo test`
Expected: All tests pass.

- [ ] **Step 2: Run clippy**

Run: `cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs && cargo clippy --all-targets -- -D warnings`
Expected: No warnings (fix any that appear).

- [ ] **Step 3: Verify Definition of Done**

- [x] Play/Pause button toggles continuous animation
- [x] Speed slider controls animation rate (10..720 deg/s)
- [x] Loop/Once toggle: loop wraps or ping-pongs, once sweeps and stops
- [x] Manual slider pauses animation on drag
- [x] Right-click grounded revolute joint → "Set as Driver" context menu
- [x] "Set as Driver" rebuilds mechanism with new driver, resets angle
- [x] Non-grounded and non-revolute joints do not show "Set as Driver"
- [x] Already-driven joint shows the option as grayed/checked
- [x] Animation and driver selection work for both sample mechanisms
- [x] All existing tests pass + new tests

- [ ] **Step 4: Final commit**

```bash
cd C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs
git add -A
git commit -m "feat(gui): Sub-project 1 complete — animation playback + driver reassignment

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
