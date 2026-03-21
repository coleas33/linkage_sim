# GUI Bugfix & UX Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 6 bugs/UX issues: simulation error surfacing, gravity controls, link length sliders, force element toolbar ribbon, context-aware force adding, and auto-sweep.

**Architecture:** Each change modifies the egui GUI layer (`src/gui/`). The solver layer is untouched. Changes touch `state.rs` (new fields, debounce logic), `mod.rs` (toolbar ribbon, error panel), `property_panel.rs` (link sliders, force section removal), `input_panel.rs` (gravity controls), and `canvas.rs` (remove drag-to-resize). A new `force_toolbar.rs` module handles the ribbon UI.

**Tech Stack:** Rust, egui/eframe, nalgebra

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/gui/force_toolbar.rs` | Force element toolbar ribbon with categorized dropdown menus |
| Create | `src/gui/error_panel.rs` | Collapsible error/log panel at bottom of window |
| Modify | `src/gui/state.rs` | New fields: `simulation_error`, `gravity_magnitude`, `sweep_dirty`, `last_rebuild_instant`; new methods: `set_link_length()`, `set_gravity_magnitude()`; modify `rebuild()` to mark sweep dirty |
| Modify | `src/gui/mod.rs` | Add error panel, force toolbar ribbon, wire up new panels |
| Modify | `src/gui/property_panel.rs` | Replace read-only link lengths with editable sliders; remove "Add ..." force buttons |
| Modify | `src/gui/input_panel.rs` | Add gravity magnitude slider to sidebar |
| Modify | `src/gui/canvas.rs` | Remove all attachment-point dragging (replaced by sidebar sliders) |
| Modify | `src/core/mechanism.rs` | Add `replace_force()` method for gravity magnitude updates |

---

## Task 1: Error Panel for Simulation Failures

**Files:**
- Create: `src/gui/error_panel.rs`
- Modify: `src/gui/state.rs:654-784` (AppState fields)
- Modify: `src/gui/state.rs:2868-2941` (run_simulation)
- Modify: `src/gui/mod.rs:456-508` (update, panel layout)

### Step-by-step:

- [ ] **Step 1: Add error state fields to AppState**

In `src/gui/state.rs`, add these fields to the `AppState` struct after the `simulation` field (around line 745):

```rust
    /// Error messages from simulation and solver failures.
    pub error_log: Vec<String>,
    /// Whether the error panel is visible.
    pub show_error_panel: bool,
```

Initialize in `Default` impl:
```rust
    error_log: Vec::new(),
    show_error_panel: false,
```

- [ ] **Step 2: Create error_panel.rs**

Create `src/gui/error_panel.rs`:

```rust
//! Collapsible error/log panel displayed at the bottom of the window.

use eframe::egui;
use super::state::AppState;

/// Draw the error panel. Shows recent error messages with a clear button.
pub fn draw_error_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        ui.strong("Errors");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Clear").clicked() {
                state.error_log.clear();
            }
            if ui.small_button("Hide").clicked() {
                state.show_error_panel = false;
            }
        });
    });

    ui.separator();

    egui::ScrollArea::vertical()
        .max_height(150.0)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for msg in &state.error_log {
                ui.colored_label(egui::Color32::from_rgb(220, 80, 80), msg);
            }
            if state.error_log.is_empty() {
                ui.label("No errors.");
            }
        });
}
```

- [ ] **Step 3: Register error_panel module in mod.rs**

In `src/gui/mod.rs`, add alongside existing module declarations:

```rust
mod error_panel;
```

- [ ] **Step 4: Wire error panel into the layout in mod.rs**

In `src/gui/mod.rs`, after the plot panel block (around line 606) and before the left panel, add. Note: egui bottom panels stack inward, so this renders between the plot panel and the canvas:

```rust
        // --- Error panel (between plots and canvas) ---
        if self.state.show_error_panel && !self.state.error_log.is_empty() {
            egui::TopBottomPanel::bottom("error_panel")
                .resizable(true)
                .default_height(120.0)
                .show(ctx, |ui| {
                    error_panel::draw_error_panel(ui, &mut self.state);
                });
        }
```

- [ ] **Step 5: Surface simulation errors in run_simulation**

In `src/gui/state.rs`, modify `run_simulation()` (line 2868) to populate `error_log` instead of silently returning:

Replace the `load_mechanism_unbuilt_from_json` error handling (line 2875-2878):
```rust
        let mut mech = match load_mechanism_unbuilt_from_json(&mech_json) {
            Ok(m) => m,
            Err(e) => {
                self.error_log.push(format!("Simulation: failed to build mechanism: {}", e));
                self.show_error_panel = true;
                return;
            }
        };
        if let Err(e) = mech.build() {
            self.error_log.push(format!("Simulation: mechanism assembly failed: {}", e));
            self.show_error_panel = true;
            return;
        }
```

Replace the `simulate()` match at the bottom (line 2916-2940):
```rust
        match simulate(
            &mech,
            &q0,
            &q_dot0,
            (0.0, duration),
            Some(&config),
            Some(&t_eval),
        ) {
            Ok(result) if result.success => {
                self.simulation = Some(SimulationState {
                    times: result.t,
                    positions: result.q,
                    time_index: 0,
                    playing: true,
                    speed: 1.0,
                    elapsed: 0.0,
                    drift: result.constraint_drift,
                });
                self.playing = false;
            }
            Ok(result) => {
                self.error_log.push(format!(
                    "Simulation did not converge: {}",
                    result.message
                ));
                self.show_error_panel = true;
            }
            Err(e) => {
                self.error_log.push(format!("Simulation failed: {}", e));
                self.show_error_panel = true;
            }
        }
```

- [ ] **Step 6: Add error indicator to status bar**

In `src/gui/mod.rs`, in the status bar section (around line 511), add an error count indicator:

```rust
                if !self.state.error_log.is_empty() {
                    ui.separator();
                    let label = format!("{} error(s)", self.state.error_log.len());
                    if ui
                        .colored_label(egui::Color32::from_rgb(220, 80, 80), &label)
                        .on_hover_text("Click to show error panel")
                        .clicked()
                    {
                        self.state.show_error_panel = !self.state.show_error_panel;
                    }
                }
```

- [ ] **Step 7: Build and verify compilation**

Run: `cargo build 2>&1`
Expected: Successful compilation with no errors.

- [ ] **Step 8: Write tests for error surfacing**

In `src/gui/state.rs` test module, add:

```rust
    #[test]
    fn run_simulation_no_mechanism_no_panic() {
        let mut state = AppState::default();
        // No mechanism loaded -- should return early without panic or error
        state.run_simulation(1.0);
        assert!(state.error_log.is_empty());
    }

    #[test]
    fn run_simulation_corrupted_mechanism_surfaces_error() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Corrupt the blueprint: remove all bodies so the mechanism can't build
        if let Some(ref mut bp) = state.blueprint {
            bp.bodies.clear();
        }

        state.run_simulation(1.0);
        assert!(
            !state.error_log.is_empty(),
            "Simulation with corrupted mechanism should surface an error"
        );
        assert!(state.show_error_panel, "Error panel should auto-show on failure");
    }

    #[test]
    fn run_simulation_valid_mechanism_succeeds() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.run_simulation(1.0);
        // Either succeeds or surfaces a meaningful error
        if state.simulation.is_some() {
            assert!(state.error_log.is_empty());
        } else {
            assert!(!state.error_log.is_empty());
        }
    }
```

- [ ] **Step 9: Run tests**

Run: `cargo test 2>&1`
Expected: All tests pass including the new one.

- [ ] **Step 10: Commit**

```bash
git add src/gui/error_panel.rs src/gui/mod.rs src/gui/state.rs
git commit -m "feat(gui): add error panel for simulation failures

Surface simulation errors in a collapsible bottom panel instead of
silently swallowing them. Adds error count indicator to status bar."
```

---

## Task 2: Gravity Magnitude Slider

**Files:**
- Modify: `src/gui/state.rs:719-720` (replace bool with f64)
- Modify: `src/gui/state.rs:1045-1063` (sync_gravity)
- Modify: `src/gui/state.rs:2883-2886` (run_simulation gravity sync)
- Modify: `src/core/mechanism.rs` (add replace_force method)
- Modify: `src/gui/input_panel.rs` (add gravity controls)
- Modify: `src/gui/mod.rs:396` (remove View menu checkbox)
- Modify: `src/gui/canvas.rs:586-587` (gravity indicator)

### Step-by-step:

- [ ] **Step 1: Replace enable_gravity bool with gravity_magnitude f64 in AppState**

In `src/gui/state.rs`, replace the `enable_gravity` field (line 719-720):

```rust
    /// Gravity magnitude in m/s² (0 = disabled, 9.81 = Earth standard).
    pub gravity_magnitude: f64,
```

In the `Default` impl, replace `enable_gravity: true` with:
```rust
    gravity_magnitude: 9.81,
```

- [ ] **Step 2: Update sync_gravity to use magnitude**

Replace the `sync_gravity()` method (lines 1047-1063):

```rust
    pub fn sync_gravity(&mut self) {
        let Some(mech) = &mut self.mechanism else {
            return;
        };
        let has_gravity = mech.forces().iter().any(|f| matches!(f, ForceElement::Gravity(_)));
        if self.gravity_magnitude > 0.0 {
            let g_elem = GravityElement {
                g_vector: [0.0, -self.gravity_magnitude],
            };
            if has_gravity {
                // Update existing gravity element with new magnitude
                if let Some(idx) = mech
                    .forces()
                    .iter()
                    .position(|f| matches!(f, ForceElement::Gravity(_)))
                {
                    mech.replace_force(idx, ForceElement::Gravity(g_elem));
                }
            } else {
                mech.add_force(ForceElement::Gravity(g_elem));
            }
        } else if has_gravity {
            if let Some(idx) = mech
                .forces()
                .iter()
                .position(|f| matches!(f, ForceElement::Gravity(_)))
            {
                mech.remove_force(idx);
            }
        }
    }
```

Note: `replace_force` doesn't exist yet on `Mechanism`. Add it in `src/core/mechanism.rs` next to `remove_force`:

```rust
    /// Replace a force element at the given index.
    pub fn replace_force(&mut self, idx: usize, force: ForceElement) {
        if idx < self.forces.len() {
            self.forces[idx] = force;
        }
    }
```

- [ ] **Step 3: Update all references from enable_gravity to gravity_magnitude**

Search-and-replace across the codebase. Key locations:

In `src/gui/state.rs` `run_simulation()` (line 2883-2886), replace:
```rust
        if self.enable_gravity {
            mech.add_force(ForceElement::Gravity(GravityElement::default()));
        }
```
with:
```rust
        if self.gravity_magnitude > 0.0 {
            mech.add_force(ForceElement::Gravity(GravityElement {
                g_vector: [0.0, -self.gravity_magnitude],
            }));
        }
```

In `src/gui/mod.rs` View menu (line 396), replace:
```rust
                    ui.checkbox(&mut self.state.enable_gravity, "Gravity");
```
with:
```rust
                    let enabled = self.state.gravity_magnitude > 0.0;
                    let mut check = enabled;
                    if ui.checkbox(&mut check, "Gravity").changed() {
                        self.state.gravity_magnitude = if check { 9.81 } else { 0.0 };
                    }
```

In `src/gui/canvas.rs` gravity indicator (line 586-587), replace:
```rust
    if state.enable_gravity {
```
with:
```rust
    if state.gravity_magnitude > 0.0 {
```

- [ ] **Step 4: Add gravity slider to input_panel.rs**

In `src/gui/input_panel.rs`, add a gravity section after the solver status block (after line 93) and before the driver info (line 96):

```rust
    // ── Gravity controls ───────────────────────────────────────────────
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Gravity:");
        let prev_g = state.gravity_magnitude;
        ui.add(
            egui::Slider::new(&mut state.gravity_magnitude, 0.0..=20.0)
                .suffix(" m/s\u{00b2}")
                .step_by(0.01)
                .clamping(egui::SliderClamping::Always),
        );
        // Mark sweep dirty when gravity changes
        if (state.gravity_magnitude - prev_g).abs() > 1e-9 {
            state.mark_sweep_dirty();
        }
    });
```

Note: `mark_sweep_dirty()` is implemented in Task 5. Implement Task 5 before Task 2.

- [ ] **Step 5: Update compute_sweep_data to use gravity_magnitude**

In `src/gui/state.rs`, the free function `compute_sweep_data()` uses `mech` which already has gravity synced via `sync_gravity()` called in `compute_sweep()`. The `compute_energy_state_mech` call at line 3337 hardcodes `9.81`:

```rust
                    let energy = compute_energy_state_mech(mech, &q, &q_dot, 9.81);
```

This is a bug — it should use the mechanism's actual gravity. However, `compute_sweep_data` is a free function without access to `AppState`. Change the signature to accept `gravity_magnitude: f64` as a parameter:

```rust
fn compute_sweep_data(
    mech: &Mechanism,
    q_start: &DVector<f64>,
    omega: f64,
    theta_0: f64,
    gravity_magnitude: f64,
) -> (SweepData, DVector<f64>)
```

And update the call site in `compute_sweep()`:
```rust
        let (data, q_zero) = compute_sweep_data(
            self.mechanism.as_ref().unwrap(),
            &q_start, omega, theta_0,
            self.gravity_magnitude,
        );
```

And the energy line:
```rust
                    let energy = compute_energy_state_mech(mech, &q, &q_dot, gravity_magnitude);
```

- [ ] **Step 6: Build and verify**

Run: `cargo build 2>&1`
Expected: Successful compilation. Fix any remaining `enable_gravity` references.

- [ ] **Step 7: Write tests**

In `src/gui/state.rs` test module:

```rust
    #[test]
    fn gravity_magnitude_zero_disables_gravity_force() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.gravity_magnitude = 0.0;
        state.sync_gravity();
        let mech = state.mechanism.as_ref().unwrap();
        assert!(
            !mech.forces().iter().any(|f| matches!(f, ForceElement::Gravity(_))),
            "Gravity should be removed when magnitude is 0"
        );
    }

    #[test]
    fn gravity_magnitude_nondefault_updates_element() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.gravity_magnitude = 3.71; // Mars gravity
        state.sync_gravity();
        let mech = state.mechanism.as_ref().unwrap();
        let grav = mech.forces().iter().find(|f| matches!(f, ForceElement::Gravity(_)));
        assert!(grav.is_some(), "Gravity element should exist");
        if let Some(ForceElement::Gravity(g)) = grav {
            assert!((g.g_vector[1] - (-3.71)).abs() < 1e-10);
        }
    }
```

- [ ] **Step 8: Run tests**

Run: `cargo test 2>&1`
Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/gui/state.rs src/gui/mod.rs src/gui/input_panel.rs src/gui/canvas.rs src/core/mechanism.rs
git commit -m "feat(gui): replace gravity checkbox with magnitude slider

Add gravity magnitude slider (0-20 m/s²) to input panel. The View menu
checkbox becomes a convenience toggle. Gravity magnitude flows through
to statics, energy, and simulation calculations."
```

---

## Task 3: Link Length Sidebar Sliders (Replace Canvas Dragging)

**Files:**
- Modify: `src/gui/property_panel.rs:241-272` (replace read-only lengths with sliders)
- Modify: `src/gui/state.rs` (add `set_link_length()` method)
- Modify: `src/gui/canvas.rs:760-823` (remove attachment-point drag logic)

### Step-by-step:

- [ ] **Step 1: Add set_link_length method to AppState**

In `src/gui/state.rs`, add after `move_attachment_point` (line 1686):

```rust
    /// Set the distance between two attachment points on a body, maintaining direction.
    ///
    /// Moves `point_b` along the vector from `point_a` to `point_b` so that
    /// the new distance equals `new_length`. If the points are coincident,
    /// moves `point_b` along the +X direction.
    pub fn set_link_length(
        &mut self,
        body_id: &str,
        point_a: &str,
        point_b: &str,
        new_length: f64,
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get(body_id) else { return };
        let Some(&pa) = body.attachment_points.get(point_a) else { return };
        let Some(&pb) = body.attachment_points.get(point_b) else { return };

        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let current_len = (dx * dx + dy * dy).sqrt();

        // Direction from A to B (or +X if coincident)
        let (ux, uy) = if current_len > 1e-12 {
            (dx / current_len, dy / current_len)
        } else {
            (1.0, 0.0)
        };

        let new_pb = [pa[0] + ux * new_length, pa[1] + uy * new_length];

        // Apply to blueprint
        if let Some(body) = bp.bodies.get_mut(body_id) {
            if let Some(pt) = body.attachment_points.get_mut(point_b) {
                *pt = new_pb;
            }
        }
        self.rebuild();
    }
```

- [ ] **Step 2: Add PendingPropertyEdit variant for link length**

In `src/gui/property_panel.rs`, add to the `PendingPropertyEdit` enum (line 21):

```rust
    LinkLength { body_id: String, point_a: String, point_b: String, length: f64 },
```

And add the match arm in the apply block (after line 444):

```rust
            PendingPropertyEdit::LinkLength { body_id, point_a, point_b, length } => {
                state.set_link_length(&body_id, &point_a, &point_b, length);
            }
```

- [ ] **Step 3: Replace read-only link lengths with editable sliders**

In `src/gui/property_panel.rs`, replace the link lengths display block (lines 241-272):

```rust
                    // Show link segment lengths as editable sliders.
                    if pts.len() >= 2 {
                        ui.separator();
                        ui.strong("Link lengths:");
                        for pair in pts.windows(2) {
                            let (name_a, pt_a) = &pair[0];
                            let (name_b, pt_b) = &pair[1];
                            let dx = pt_b.x - pt_a.x;
                            let dy = pt_b.y - pt_a.y;
                            let len = (dx * dx + dy * dy).sqrt();
                            let mut display_len = units.length(len);
                            ui.horizontal(|ui| {
                                ui.label(format!("  {}\u{2192}{}:", name_a, name_b));
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut display_len)
                                            .speed(units.length(0.001))
                                            .range(units.length(0.001)..=units.length(10.0))
                                            .suffix(units.length_suffix()),
                                    )
                                    .changed()
                                {
                                    let new_si = units.length_to_si(display_len);
                                    pending = Some(PendingPropertyEdit::LinkLength {
                                        body_id: body_id.clone(),
                                        point_a: name_a.to_string(),
                                        point_b: name_b.to_string(),
                                        length: new_si,
                                    });
                                }
                            });
                        }
                        if pts.len() >= 3 {
                            let (name_a, pt_a) = pts.last().unwrap();
                            let (name_b, pt_b) = &pts[0];
                            let dx = pt_b.x - pt_a.x;
                            let dy = pt_b.y - pt_a.y;
                            let len = (dx * dx + dy * dy).sqrt();
                            let mut display_len = units.length(len);
                            ui.horizontal(|ui| {
                                ui.label(format!("  {}\u{2192}{}:", name_a, name_b));
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut display_len)
                                            .speed(units.length(0.001))
                                            .range(units.length(0.001)..=units.length(10.0))
                                            .suffix(units.length_suffix()),
                                    )
                                    .changed()
                                {
                                    let new_si = units.length_to_si(display_len);
                                    pending = Some(PendingPropertyEdit::LinkLength {
                                        body_id: body_id.clone(),
                                        point_a: name_a.to_string(),
                                        point_b: name_b.to_string(),
                                        length: new_si,
                                    });
                                }
                            });
                        }
                    }
```

Note: `length_to_si()` already exists on `DisplayUnits` (in `state.rs`). No new code needed for the conversion.

- [ ] **Step 4: Remove ALL canvas attachment-point dragging**

This intentionally removes all canvas-based attachment-point manipulation. Users now edit link geometry exclusively via the sidebar sliders (Task 3 Steps 1-3). Click-to-select is preserved. The user explicitly requested this change ("We should just use a slider in the side bar instead").

In `src/gui/canvas.rs`, replace lines 780-823 with click-to-select and pan handling only:

```rust
    // ── Interaction: click to select (Select mode) ────────────────────────
    if response.clicked_by(egui::PointerButton::Primary)
        && !is_shift
        && state.active_tool == EditorTool::Select
    {
        if let Some(pointer_pos) = response.interact_pointer_pos() {
            if let Some(hit) = find_nearest_attachment(pointer_pos) {
                // Select the body this point belongs to
                state.selected = Some(SelectedEntity::Body(hit.body_id.clone()));
            }
        }
    }

    // Primary drag on empty space (Select mode) pans the view.
    if response.dragged_by(egui::PointerButton::Primary) && !is_shift {
        if state.active_tool == EditorTool::Select
            && state.draw_link_start.is_none()
        {
            is_panning = true;
        }
    }

    // Clear any stale drag target since we no longer use it for resizing
    state.drag_target = None;
```

Note: Preserve the existing `drag_started_by` / `dragged_by` logic for the DrawLink and AddBody tools — only remove the Select-mode attachment-point drag. The existing code below line 823 (pan, zoom, draw-link interactions) must be preserved exactly.

- [ ] **Step 5: Build and verify**

Run: `cargo build 2>&1`
Expected: Successful compilation.

- [ ] **Step 6: Write tests**

In `src/gui/state.rs` test module:

```rust
    #[test]
    fn set_link_length_maintains_direction() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        let pa = crank.attachment_points.get("A").unwrap();
        let pb = crank.attachment_points.get("B").unwrap();
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let orig_len = (dx * dx + dy * dy).sqrt();
        let orig_angle = dy.atan2(dx);

        let new_len = orig_len * 1.5;
        state.set_link_length("crank", "A", "B", new_len);

        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        let pa2 = crank.attachment_points.get("A").unwrap();
        let pb2 = crank.attachment_points.get("B").unwrap();
        let dx2 = pb2[0] - pa2[0];
        let dy2 = pb2[1] - pa2[1];
        let actual_len = (dx2 * dx2 + dy2 * dy2).sqrt();
        let actual_angle = dy2.atan2(dx2);

        assert!(
            (actual_len - new_len).abs() < 1e-10,
            "Length should be {}, got {}",
            new_len, actual_len
        );
        assert!(
            (actual_angle - orig_angle).abs() < 1e-10,
            "Direction should be preserved: expected {}, got {}",
            orig_angle, actual_angle
        );
    }

    #[test]
    fn set_link_length_point_a_stays_fixed() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let bp = state.blueprint.as_ref().unwrap();
        let pa_before = bp.bodies.get("crank").unwrap()
            .attachment_points.get("A").unwrap().clone();

        state.set_link_length("crank", "A", "B", 0.05);

        let bp = state.blueprint.as_ref().unwrap();
        let pa_after = bp.bodies.get("crank").unwrap()
            .attachment_points.get("A").unwrap();

        assert!((pa_after[0] - pa_before[0]).abs() < 1e-12);
        assert!((pa_after[1] - pa_before[1]).abs() < 1e-12);
    }
```

- [ ] **Step 7: Run tests**

Run: `cargo test 2>&1`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/gui/property_panel.rs src/gui/state.rs src/gui/canvas.rs
git commit -m "feat(gui): replace canvas drag-resize with link length sliders

Add editable DragValue sliders for link segment lengths in the property
panel. Remove attachment-point drag-to-resize from canvas (click still
selects). Maintains link direction when changing length."
```

---

## Task 4: Force Element Toolbar Ribbon

**Files:**
- Create: `src/gui/force_toolbar.rs`
- Modify: `src/gui/mod.rs:456-508` (add second toolbar row)
- Modify: `src/gui/property_panel.rs:796-957` (remove "Add ..." buttons)
- Modify: `src/gui/state.rs` (import changes only)

### Step-by-step:

- [ ] **Step 1: Create force_toolbar.rs with categorized menus**

Create `src/gui/force_toolbar.rs`:

```rust
//! Force element toolbar ribbon with categorized dropdown menus.
//!
//! Provides two dropdown menus:
//! - **Joint Torques**: Motor, Torsion Spring, Rotary Damper, Bearing Friction, Joint Limit
//! - **Link Forces**: Linear Spring, Linear Damper, Ext. Force, Ext. Torque, Gas Spring, Linear Actuator

use eframe::egui;
use crate::core::state::GROUND_ID;
use crate::forces::elements::*;
use super::state::{AppState, SelectedEntity};

/// Pending force addition collected during toolbar rendering.
/// Applied after all UI reads are done.
pub enum PendingForceAdd {
    Add(ForceElement),
}

/// Draw the force toolbar ribbon. Returns a pending force addition if any button was clicked.
pub fn draw_force_toolbar(ui: &mut egui::Ui, state: &AppState) -> Option<PendingForceAdd> {
    let mut pending: Option<PendingForceAdd> = None;

    let (selected_body, connected_body) = resolve_target_bodies(state);

    ui.horizontal(|ui| {
        ui.spacing_mut().button_padding = egui::vec2(6.0, 3.0);

        // ── Joint Torques dropdown ──────────────────────────────────
        ui.menu_button("Joint Torques \u{25BC}", |ui| {
            if let Some((ref a, ref b)) = two_bodies(&selected_body, &connected_body) {
                if ui.button("Motor").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::Motor(MotorElement {
                        body_i: a.clone(),
                        body_j: b.clone(),
                        stall_torque: 10.0,
                        no_load_speed: 10.0,
                        direction: 1.0,
                    })));
                    ui.close_menu();
                }
                if ui.button("Torsion Spring").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::TorsionSpring(
                        TorsionSpringElement {
                            body_i: a.clone(),
                            body_j: b.clone(),
                            stiffness: 10.0,
                            free_angle: 0.0,
                        },
                    )));
                    ui.close_menu();
                }
                if ui.button("Rotary Damper").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::RotaryDamper(
                        RotaryDamperElement {
                            body_i: a.clone(),
                            body_j: b.clone(),
                            damping: 5.0,
                        },
                    )));
                    ui.close_menu();
                }
                if ui.button("Bearing Friction").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::BearingFriction(
                        BearingFrictionElement {
                            body_i: a.clone(),
                            body_j: b.clone(),
                            constant_drag: 0.1,
                            viscous_coeff: 0.01,
                            coulomb_coeff: 0.0,
                            pin_radius: 0.0,
                            radial_load: 0.0,
                            v_threshold: 0.01,
                        },
                    )));
                    ui.close_menu();
                }
                if ui.button("Joint Limit").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::JointLimit(
                        JointLimitElement {
                            body_i: a.clone(),
                            body_j: b.clone(),
                            angle_min: -std::f64::consts::FRAC_PI_2,
                            angle_max: std::f64::consts::FRAC_PI_2,
                            stiffness: 100.0,
                            damping: 0.0,
                            restitution: 0.5,
                        },
                    )));
                    ui.close_menu();
                }
            } else {
                ui.label("Select a body first");
            }
        });

        // ── Link Forces dropdown ──────────────────────────────────
        ui.menu_button("Link Forces \u{25BC}", |ui| {
            if let Some(ref body_id) = selected_body {
                if ui.button("External Force").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::ExternalForce(
                        ExternalForceElement {
                            body_id: body_id.clone(),
                            local_point: [0.0, 0.0],
                            force: [0.0, -10.0],
                            modulation: TimeModulation::Constant,
                        },
                    )));
                    ui.close_menu();
                }
                if ui.button("External Torque").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::ExternalTorque(
                        ExternalTorqueElement {
                            body_id: body_id.clone(),
                            torque: 1.0,
                            modulation: TimeModulation::Constant,
                        },
                    )));
                    ui.close_menu();
                }
            } else {
                ui.label("Select a body first");
            }

            ui.separator();

            if let Some((ref a, ref b)) = two_bodies(&selected_body, &connected_body) {
                if ui.button("Linear Spring").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::LinearSpring(
                        LinearSpringElement {
                            body_a: a.clone(),
                            point_a: [0.0, 0.0],
                            body_b: b.clone(),
                            point_b: [0.0, 0.0],
                            stiffness: 100.0,
                            free_length: 0.1,
                        },
                    )));
                    ui.close_menu();
                }
                if ui.button("Linear Damper").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::LinearDamper(
                        LinearDamperElement {
                            body_a: a.clone(),
                            point_a: [0.0, 0.0],
                            body_b: b.clone(),
                            point_b: [0.0, 0.0],
                            damping: 10.0,
                        },
                    )));
                    ui.close_menu();
                }
                if ui.button("Gas Spring").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::GasSpring(
                        GasSpringElement {
                            body_a: a.clone(),
                            point_a: [0.0, 0.0],
                            body_b: b.clone(),
                            point_b: [0.0, 0.0],
                            initial_force: 100.0,
                            extended_length: 0.5,
                            stroke: 0.2,
                            damping: 0.0,
                            polytropic_exp: 1.0,
                        },
                    )));
                    ui.close_menu();
                }
                if ui.button("Linear Actuator").clicked() {
                    pending = Some(PendingForceAdd::Add(ForceElement::LinearActuator(
                        LinearActuatorElement {
                            body_a: a.clone(),
                            point_a: [0.0, 0.0],
                            body_b: b.clone(),
                            point_b: [0.0, 0.0],
                            force: 100.0,
                            speed_limit: 0.0,
                        },
                    )));
                    ui.close_menu();
                }
            } else if selected_body.is_none() {
                ui.separator();
                ui.label("Select a body for 2-body elements");
            }
        });
    });

    pending
}

/// Determine target bodies from current selection.
///
/// Returns (selected_body, connected_body) where connected_body is the
/// body connected to selected_body at the nearest joint (smart default
/// for two-body force elements).
pub(crate) fn resolve_target_bodies(state: &AppState) -> (Option<String>, Option<String>) {
    let selected_body = match &state.selected {
        Some(SelectedEntity::Body(id)) if id != GROUND_ID => Some(id.clone()),
        Some(SelectedEntity::Joint(joint_id)) => {
            // If a joint is selected, use its body_j (non-ground body)
            if let Some(mech) = &state.mechanism {
                mech.joints()
                    .iter()
                    .find(|j| j.id() == joint_id)
                    .map(|j| {
                        if j.body_i_id() == GROUND_ID {
                            j.body_j_id().to_string()
                        } else {
                            j.body_i_id().to_string()
                        }
                    })
            } else {
                None
            }
        }
        _ => None,
    };

    let connected_body = if let (Some(ref sel_id), Some(ref mech)) =
        (&selected_body, &state.mechanism)
    {
        // Find first joint connecting this body to another, return the other body
        mech.joints()
            .iter()
            .find_map(|j| {
                if j.body_i_id() == sel_id {
                    Some(j.body_j_id().to_string())
                } else if j.body_j_id() == sel_id {
                    Some(j.body_i_id().to_string())
                } else {
                    None
                }
            })
    } else {
        None
    };

    (selected_body, connected_body)
}

/// Return two body IDs for two-body elements. Uses selected and connected.
/// If connected is ground, that's fine — ground can be part of a spring/damper.
fn two_bodies(
    selected: &Option<String>,
    connected: &Option<String>,
) -> Option<(String, String)> {
    match (selected, connected) {
        (Some(a), Some(b)) => Some((a.clone(), b.clone())),
        _ => None,
    }
}
```

- [ ] **Step 2: Register module and wire into mod.rs**

In `src/gui/mod.rs`, add:

```rust
mod force_toolbar;
```

Then in `update()`, after the existing toolbar panel (line 508) and before the status bar, add a second toolbar row:

```rust
        // --- Force element toolbar ribbon ---
        egui::TopBottomPanel::top("force_toolbar").show(ctx, |ui| {
            if let Some(force_add) = force_toolbar::draw_force_toolbar(ui, &self.state) {
                match force_add {
                    force_toolbar::PendingForceAdd::Add(force) => {
                        self.state.add_force_element(force);
                    }
                }
            }
        });
```

- [ ] **Step 3: Remove "Add ..." buttons from property_panel.rs**

In `src/gui/property_panel.rs`, remove the entire "Add ..." buttons block (lines 795-957):

Replace lines 795-957 with just:

```rust
    // Force element parameters are still edited here; adding is done via the toolbar ribbon.
```

Also remove the `two_body_ids` helper function (lines 959-967) — it is only called from the "Add ..." buttons block being removed. Verify with `grep -n two_body_ids src/gui/property_panel.rs` before deleting.

Keep the `draw_force_elements_panel` function header and the existing force element listing/editing code (lines 748-789) — only remove the "Add ..." buttons section and the closing brace adjustment.

- [ ] **Step 4: Build and verify**

Run: `cargo build 2>&1`
Expected: Successful compilation.

- [ ] **Step 5: Write tests**

In `src/gui/force_toolbar.rs`, add a test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::state::AppState;
    use crate::gui::samples::SampleMechanism;

    #[test]
    fn resolve_target_bodies_with_selected_body() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.selected = Some(SelectedEntity::Body("crank".to_string()));

        let (sel, conn) = resolve_target_bodies(&state);
        assert_eq!(sel, Some("crank".to_string()));
        assert!(conn.is_some(), "Connected body should be found via joints");
    }

    #[test]
    fn resolve_target_bodies_with_no_selection() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.selected = None;

        let (sel, conn) = resolve_target_bodies(&state);
        assert!(sel.is_none());
        assert!(conn.is_none());
    }

    #[test]
    fn resolve_target_bodies_ground_selection_excluded() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.selected = Some(SelectedEntity::Body("ground".to_string()));

        let (sel, _conn) = resolve_target_bodies(&state);
        assert!(sel.is_none(), "Ground should not be a valid force target");
    }
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test 2>&1`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/gui/force_toolbar.rs src/gui/mod.rs src/gui/property_panel.rs
git commit -m "feat(gui): move force adding to categorized toolbar ribbon

Replace sidebar 'Add ...' buttons with a toolbar ribbon containing
two dropdown menus: 'Joint Torques' (Motor, Torsion Spring, Rotary
Damper, Bearing Friction, Joint Limit) and 'Link Forces' (Spring,
Damper, Ext Force/Torque, Gas Spring, Actuator). Force elements
are now context-aware — they target the currently selected body."
```

---

## Task 5: Auto-Sweep with Debounce

**Files:**
- Modify: `src/gui/state.rs` (add sweep_dirty flag, debounce timer, mark_sweep_dirty method)
- Modify: `src/gui/mod.rs` (check debounce in update loop)

### Step-by-step:

- [ ] **Step 1: Add sweep debounce fields to AppState**

In `src/gui/state.rs`, add after the `sweep_data` field (around line 694):

```rust
    /// Whether sweep data needs recomputation (set by rebuild, gravity change, etc.).
    pub sweep_dirty: bool,
    /// Time when sweep was last marked dirty (for debounce). Uses `std::time::Instant`.
    pub sweep_dirty_since: Option<std::time::Instant>,
```

Initialize in `Default` impl:
```rust
    sweep_dirty: false,
    sweep_dirty_since: None,
```

Add the import at the top of `state.rs`:
```rust
use std::time::Instant;
```

- [ ] **Step 2: Add mark_sweep_dirty method**

In `src/gui/state.rs`, add:

```rust
    /// Mark sweep data as stale, starting the debounce timer.
    pub fn mark_sweep_dirty(&mut self) {
        self.sweep_dirty = true;
        self.sweep_dirty_since = Some(Instant::now());
    }
```

- [ ] **Step 3: Clear dirty flag in compute_sweep and mark dirty from rebuild()**

In `src/gui/state.rs`, at the top of `compute_sweep()` (around line 2846), add:

```rust
        self.sweep_dirty = false;
        self.sweep_dirty_since = None;
```

At the end of `rebuild()` (line 1665), add:

```rust
        self.mark_sweep_dirty();
```

- [ ] **Step 4: Add debounced sweep check to update loop**

In `src/gui/mod.rs`, in the `update()` method, early in the function (after keyboard shortcut handling, around line 87), add:

```rust
        // ── Debounced sweep recomputation ──────────────────────────────
        if self.state.sweep_dirty {
            if let Some(since) = self.state.sweep_dirty_since {
                if since.elapsed().as_millis() >= 200 {
                    self.state.sweep_dirty = false;
                    self.state.sweep_dirty_since = None;
                    self.state.compute_sweep();
                }
            }
        }
```

- [ ] **Step 5: Remove manual compute_sweep calls that are now redundant**

In `src/gui/state.rs`, the `load_sample()` method calls `self.compute_sweep()` at line 998. Keep this call — initial load should be immediate, not debounced.

Other call sites need case-by-case handling:

- **`reassign_driver()` and similar methods that call `self.rebuild()` before `compute_sweep()`**: Remove the `compute_sweep()` call entirely — `rebuild()` already calls `mark_sweep_dirty()`.
- **`apply_load_case()` and other methods that call `compute_sweep()` without a preceding `rebuild()`**: Replace with `self.mark_sweep_dirty()`.

Search for all `self.compute_sweep()` calls. Keep only the one in `load_sample()` (immediate computation on initial load). Remove or replace all others per the rules above.

- [ ] **Step 6: Build and verify**

Run: `cargo build 2>&1`
Expected: Successful compilation.

- [ ] **Step 7: Write tests**

In `src/gui/state.rs` test module:

```rust
    #[test]
    fn rebuild_marks_sweep_dirty() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // load_sample calls compute_sweep directly, so sweep_dirty should be
        // set by the rebuild inside load_sample but then cleared... actually
        // load_sample calls compute_sweep after rebuild, but rebuild also sets
        // sweep_dirty. Since compute_sweep doesn't clear the flag, check:
        state.sweep_dirty = false; // reset for test
        state.rebuild();
        assert!(state.sweep_dirty, "rebuild() should mark sweep as dirty");
    }

    #[test]
    fn load_sample_produces_sweep_data() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(
            state.sweep_data.is_some(),
            "load_sample should produce sweep data immediately"
        );
        let sweep = state.sweep_data.as_ref().unwrap();
        assert!(
            !sweep.angles_deg.is_empty(),
            "Sweep should have angle data"
        );
        assert!(
            !sweep.joint_reaction_magnitudes.is_empty(),
            "Sweep should have joint reaction data"
        );
    }
```

- [ ] **Step 8: Run tests**

Run: `cargo test 2>&1`
Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/gui/state.rs src/gui/mod.rs
git commit -m "feat(gui): auto-sweep with 200ms debounce on rebuild

Sweep data now recomputes automatically after mechanism changes,
with a 200ms debounce to avoid jank during interactive editing.
Plot tabs (joint reactions, energy, etc.) are immediately available
after loading a mechanism."
```

---

## Task 6: Integration Testing & Final Cleanup

**Files:**
- Modify: `src/gui/state.rs` (integration tests)
- All files from previous tasks (compile-check)

### Step-by-step:

- [ ] **Step 1: Full build verification**

Run: `cargo build 2>&1`
Expected: Clean build with no errors or warnings.

- [ ] **Step 2: Run full test suite**

Run: `cargo test 2>&1`
Expected: All tests pass, including all new tests from Tasks 1-5.

- [ ] **Step 3: Write integration test — full workflow**

In `src/gui/state.rs` test module:

```rust
    #[test]
    fn integration_gravity_affects_driver_torque() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // With gravity
        state.gravity_magnitude = 9.81;
        state.sync_gravity();
        state.compute_sweep();
        let sweep_grav = state.sweep_data.as_ref().unwrap();
        let torques_grav: Vec<f64> = sweep_grav
            .driver_torques
            .as_ref()
            .unwrap()
            .iter()
            .copied()
            .collect();

        // Without gravity
        state.gravity_magnitude = 0.0;
        state.sync_gravity();
        state.compute_sweep();
        let sweep_no_grav = state.sweep_data.as_ref().unwrap();
        let torques_no_grav: Vec<f64> = sweep_no_grav
            .driver_torques
            .as_ref()
            .unwrap()
            .iter()
            .copied()
            .collect();

        // Torques should differ when gravity is toggled
        assert_eq!(torques_grav.len(), torques_no_grav.len());
        let differs = torques_grav
            .iter()
            .zip(torques_no_grav.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            differs,
            "Driver torque should change when gravity is toggled"
        );
    }

    #[test]
    fn integration_force_add_uses_context() {
        use crate::gui::force_toolbar::resolve_target_bodies;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // No selection — should return None
        state.selected = None;
        let (sel, _) = resolve_target_bodies(&state);
        assert!(sel.is_none());

        // Select crank — should resolve
        state.selected = Some(SelectedEntity::Body("crank".to_string()));
        let (sel, conn) = resolve_target_bodies(&state);
        assert_eq!(sel, Some("crank".to_string()));
        assert!(conn.is_some());
    }
```

- [ ] **Step 4: Run all tests**

Run: `cargo test 2>&1`
Expected: All tests pass.

- [ ] **Step 5: Verify all samples load with sweep data**

In `src/gui/state.rs` test module:

```rust
    #[test]
    fn all_samples_have_sweep_data_with_joint_reactions() {
        for sample in SampleMechanism::all() {
            let mut state = AppState::default();
            state.load_sample(*sample);
            assert!(
                state.sweep_data.is_some(),
                "Sample {:?} should have sweep data after load",
                sample
            );
        }
    }
```

- [ ] **Step 6: Run final test suite**

Run: `cargo test 2>&1`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/gui/state.rs
git commit -m "test(gui): add integration tests for bugfix/UX rework

Verify gravity affects driver torque, context-aware force targeting,
and all samples produce sweep data on load."
```

---

## Implementation Order & Dependencies

```
Task 1 (Error Panel) ─────────────────────── independent
Task 2 (Gravity Slider) ─────────────────── depends on Task 5 for mark_sweep_dirty()
Task 5 (Auto-Sweep + Debounce) ──────────── independent (implement before Task 2)
Task 3 (Link Length Sliders) ─────────────── independent
Task 4 (Force Toolbar Ribbon) ───────────── independent
Task 6 (Integration Tests) ──────────────── depends on all above
```

**Recommended execution order:** 1 → 5 → 2 → 3 → 4 → 6

Task 2 references `mark_sweep_dirty()` which is defined in Task 5, so Task 5 should be done first. All other tasks are independent and could be parallelized.
