# Actuator Stroke Limits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add min/max stroke limits to LinearActuatorElement using penalty spring+damper forces at the boundaries.

**Architecture:** Add 5 new fields to the existing `LinearActuatorElement` struct with serde defaults for backward compatibility. Extend the evaluation function with penalty force logic matching `evaluate_joint_limit`'s proven pattern. Add UI controls and canvas visualization.

**Tech Stack:** Rust, egui, nalgebra, serde

**Spec:** `docs/superpowers/specs/2026-03-24-actuator-stroke-limits-design.md`

---

### Task 1: Add struct fields and serde defaults

**Files:**
- Modify: `linkage-sim-rs/src/forces/elements/element_types.rs:253-278`

- [ ] **Step 1: Add serde default helpers**

In `element_types.rs`, after the existing `default_direction()` helper (line 19), add:

```rust
fn default_end_stop_stiffness() -> f64 {
    10000.0
}
fn default_end_stop_damping() -> f64 {
    10.0
}
```

Note: `default_restitution()` already exists at line 15 — reuse it for `end_stop_restitution`.

- [ ] **Step 2: Add 5 fields to `LinearActuatorElement`**

After the `speed_limit` field (line 277), add:

```rust
    /// Minimum stroke length (m). 0 = no min limit.
    #[serde(default)]
    pub stroke_min: f64,
    /// Maximum stroke length (m). 0 = no max limit.
    #[serde(default)]
    pub stroke_max: f64,
    /// End-stop penalty spring stiffness (N/m).
    #[serde(default = "default_end_stop_stiffness")]
    pub end_stop_stiffness: f64,
    /// End-stop penalty damping (N·s/m).
    #[serde(default = "default_end_stop_damping")]
    pub end_stop_damping: f64,
    /// End-stop coefficient of restitution [0,1].
    #[serde(default = "default_restitution")]
    pub end_stop_restitution: f64,
```

- [ ] **Step 3: Fix all existing `LinearActuatorElement` struct literals in the codebase**

Every place that constructs a `LinearActuatorElement { ... }` needs the 5 new fields. Search for `LinearActuatorElement {` and add defaults. Key locations:

- `gui/force_toolbar.rs` (~line 140): add `stroke_min: 0.0, stroke_max: 0.0, end_stop_stiffness: 10000.0, end_stop_damping: 10.0, end_stop_restitution: 0.5,`
- `gui/canvas/mod.rs` (~line 193): bare struct in `fill_template_linear_actuator` test — add same defaults
- `gui/samples/fourbar.rs` (~line 410): bare struct in actuator sample builder — add same defaults
- `forces/elements/mod.rs` (test struct literals at ~lines 659, 687, 714, 887, 982): add same defaults to ALL `LinearActuatorElement` in tests

Note: `gui/canvas/rendering.rs` (~line 1664) uses `..a.clone()` spread syntax — no fixup needed there.

- [ ] **Step 4: Verify compilation**

Run: `cargo check`
Expected: Compiles with no errors related to `LinearActuatorElement`

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add stroke limit fields to LinearActuatorElement"
```

---

### Task 2: Implement penalty force evaluation (TDD)

**Files:**
- Modify: `linkage-sim-rs/src/forces/elements/evaluation.rs:376-420`
- Modify: `linkage-sim-rs/src/forces/elements/mod.rs` (tests)

- [ ] **Step 1: Write failing test — penalty zero when within limits**

Add to the test module in `forces/elements/mod.rs`, after the existing actuator tests (~line 730):

```rust
    #[test]
    fn actuator_stroke_no_penalty_within_limits() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.0, 0.0, 0.0); // length = 1.0
        let q_dot = DVector::zeros(state.n_coords());

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 50.0,
            speed_limit: 0.0,
            stroke_min: 0.5,
            stroke_max: 1.5,
            end_stop_stiffness: 10000.0,
            end_stop_damping: 10.0,
            end_stop_restitution: 0.5,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Within limits: only normal actuator force, no penalty
        assert_abs_diff_eq!(result[0], -50.0, epsilon = 1e-10); // bar1 Fx
        assert_abs_diff_eq!(result[3], 50.0, epsilon = 1e-10);  // bar2 Fx
    }
```

- [ ] **Step 2: Run test to confirm it passes (no penalty logic needed for within-limits case)**

Run: `cargo test --lib actuator_stroke_no_penalty_within_limits`
Expected: PASS (existing evaluation already returns correct actuator force within limits)

- [ ] **Step 3: Write failing test — penalty pushes apart at min stop**

```rust
    #[test]
    fn actuator_stroke_min_stop_pushes_apart() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.3, 0.0, 0.0); // length = 0.3, below min 0.5
        let q_dot = DVector::zeros(state.n_coords());

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 0.0, // no actuator force — isolate penalty
            speed_limit: 0.0,
            stroke_min: 0.5,
            stroke_max: 1.5,
            end_stop_stiffness: 10000.0,
            end_stop_damping: 0.0, // no damping for this test
            end_stop_restitution: 0.5,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Penetration = 0.5 - 0.3 = 0.2, F_penalty = +10000 * 0.2 = +2000 N (push apart)
        assert_abs_diff_eq!(result[0], -2000.0, epsilon = 1e-8); // bar1 pushed left
        assert_abs_diff_eq!(result[3], 2000.0, epsilon = 1e-8);  // bar2 pushed right
    }
```

- [ ] **Step 4: Write failing test — penalty pulls together at max stop**

```rust
    #[test]
    fn actuator_stroke_max_stop_pulls_together() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.7, 0.0, 0.0); // length = 1.7, above max 1.5
        let q_dot = DVector::zeros(state.n_coords());

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 0.0,
            speed_limit: 0.0,
            stroke_min: 0.5,
            stroke_max: 1.5,
            end_stop_stiffness: 10000.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.5,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Penetration = 1.7 - 1.5 = 0.2, F_penalty = -(10000 * 0.2) = -2000 N (pull together)
        assert_abs_diff_eq!(result[0], 2000.0, epsilon = 1e-8);  // bar1 pulled right
        assert_abs_diff_eq!(result[3], -2000.0, epsilon = 1e-8); // bar2 pulled left
    }
```

- [ ] **Step 5: Write failing test — actuator force still applied at limits**

```rust
    #[test]
    fn actuator_stroke_force_not_cut_at_limit() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.7, 0.0, 0.0); // above max
        let q_dot = DVector::zeros(state.n_coords());

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 100.0, // actuator pushing apart
            speed_limit: 0.0,
            stroke_min: 0.5,
            stroke_max: 1.5,
            end_stop_stiffness: 10000.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.5,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Actuator force (+100) + penalty (-2000) = net -1900 on bar2
        assert_abs_diff_eq!(result[3], -1900.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result[0], 1900.0, epsilon = 1e-8);
    }
```

- [ ] **Step 6: Write failing test — limits disabled when both zero**

```rust
    #[test]
    fn actuator_stroke_limits_disabled_when_both_zero() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 100.0, 0.0, 0.0); // absurdly far
        let q_dot = DVector::zeros(state.n_coords());

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 50.0,
            speed_limit: 0.0,
            stroke_min: 0.0, // disabled
            stroke_max: 0.0, // disabled
            end_stop_stiffness: 10000.0,
            end_stop_damping: 10.0,
            end_stop_restitution: 0.5,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // No penalty, just normal actuator force
        assert_abs_diff_eq!(result[0], -50.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 50.0, epsilon = 1e-10);
    }
```

- [ ] **Step 7: Write failing test — max-only limit (stroke_min=0)**

```rust
    #[test]
    fn actuator_stroke_max_only_limit() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.01, 0.0, 0.0); // very short, but no min limit
        let q_dot = DVector::zeros(state.n_coords());

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 0.0,
            speed_limit: 0.0,
            stroke_min: 0.0,  // no min limit
            stroke_max: 2.0,  // max only
            end_stop_stiffness: 10000.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.5,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // No penalty — below max, no min enforced
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 0.0, epsilon = 1e-10);
    }
```

- [ ] **Step 8: Write failing test — degenerate stroke_min == stroke_max**

```rust
    #[test]
    fn actuator_stroke_degenerate_equal_limits_inactive() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 100.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 50.0,
            speed_limit: 0.0,
            stroke_min: 1.0,
            stroke_max: 1.0, // equal to min — should be inactive
            end_stop_stiffness: 10000.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.5,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // No penalty, just normal force
        assert_abs_diff_eq!(result[0], -50.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 50.0, epsilon = 1e-10);
    }
```

- [ ] **Step 9: Write failing test — asymmetric damping**

```rust
    #[test]
    fn actuator_stroke_asymmetric_damping() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.3, 0.0, 0.0); // below min 0.5
        let mut q_dot_into = DVector::zeros(state.n_coords());
        q_dot_into[3] = -1.0; // bar2 retracting (into min stop)

        let mut q_dot_away = DVector::zeros(state.n_coords());
        q_dot_away[3] = 1.0; // bar2 extending (away from min stop)

        let act = LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            point_b_name: None,
            force: 0.0,
            speed_limit: 0.0,
            stroke_min: 0.5,
            stroke_max: 1.5,
            end_stop_stiffness: 10000.0,
            end_stop_damping: 100.0,
            end_stop_restitution: 0.5,
        };

        let fe = ForceElement::LinearActuator(act);

        let result_into = fe.evaluate(&state, &bodies, &q, &q_dot_into, 0.0);
        let result_away = fe.evaluate(&state, &bodies, &q, &q_dot_away, 0.0);

        // Moving into stop: full damping (100.0), away: reduced (100*0.5 = 50)
        // Both have same spring force. "Into" has MORE total force than "away".
        assert!(result_into[3].abs() > result_away[3].abs(),
            "Into-stop force ({}) should exceed away-from-stop force ({})",
            result_into[3], result_away[3]);
    }
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `cargo test --lib actuator_stroke`
Expected: FAIL on all penalty tests (no penalty logic implemented yet). The `within_limits` and `disabled_when_both_zero` tests may pass since they expect no penalty.

- [ ] **Step 11: Implement penalty force logic**

In `evaluation.rs`, modify `evaluate_linear_actuator`. Replace the current function body (lines 376-420) with:

```rust
pub fn evaluate_linear_actuator(
    a: &LinearActuatorElement,
    state: &State,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let pt_a_local = Vector2::new(a.point_a[0], a.point_a[1]);
    let pt_b_local = Vector2::new(a.point_b[0], a.point_b[1]);

    let pt_a_global = state.body_point_global(&a.body_a, &pt_a_local, q);
    let pt_b_global = state.body_point_global(&a.body_b, &pt_b_local, q);

    let delta = pt_b_global - pt_a_global;
    let length = delta.norm();

    if length < 1e-15 {
        return DVector::zeros(state.n_coords());
    }

    let unit = delta / length;

    // Speed limiting: ramp force to zero as speed approaches limit
    let actual_force = if a.speed_limit > 0.0 {
        let v_a = state.body_point_velocity(&a.body_a, &pt_a_local, q, q_dot);
        let v_b = state.body_point_velocity(&a.body_b, &pt_b_local, q, q_dot);
        let v_along = unit.dot(&(v_b - v_a));
        let speed_ratio = v_along.abs() / a.speed_limit;
        if speed_ratio >= 1.0 {
            0.0
        } else {
            a.force * (1.0 - speed_ratio)
        }
    } else {
        a.force
    };

    // Start with normal actuator force
    let mut net_force_along_unit = actual_force;

    // Stroke limit penalty forces
    let limits_active = a.stroke_max > 0.0 && a.stroke_max > a.stroke_min;
    if limits_active {
        // Compute relative velocity along actuator axis (positive = extending)
        let v_a = state.body_point_velocity(&a.body_a, &pt_a_local, q, q_dot);
        let v_b = state.body_point_velocity(&a.body_b, &pt_b_local, q, q_dot);
        let v_rel = unit.dot(&(v_b - v_a));

        if a.stroke_min > 0.0 && length < a.stroke_min {
            // Over-retracted: push apart (positive)
            let penetration = a.stroke_min - length;
            let damp = if v_rel < 0.0 {
                a.end_stop_damping // into stop
            } else {
                a.end_stop_damping * a.end_stop_restitution // bouncing away
            };
            net_force_along_unit += a.end_stop_stiffness * penetration - damp * v_rel;
        } else if a.stroke_max > 0.0 && length > a.stroke_max {
            // Over-extended: pull together (negative)
            let penetration = length - a.stroke_max;
            let damp = if v_rel > 0.0 {
                a.end_stop_damping // into stop
            } else {
                a.end_stop_damping * a.end_stop_restitution // bouncing away
            };
            net_force_along_unit -= a.end_stop_stiffness * penetration + damp * v_rel;
        }
    }

    let force_on_b = unit * net_force_along_unit;
    let force_on_a = -force_on_b;

    let mut total = DVector::zeros(state.n_coords());
    total += point_force_to_q(state, &a.body_a, &pt_a_local, &force_on_a, q);
    total += point_force_to_q(state, &a.body_b, &pt_b_local, &force_on_b, q);
    total
}
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `cargo test --lib actuator_stroke`
Expected: All 8 new tests PASS

- [ ] **Step 13: Run full test suite**

Run: `cargo test --lib`
Expected: All 438+ tests pass (existing actuator tests still work with new default fields)

- [ ] **Step 14: Commit**

```bash
git add -A && git commit -m "feat: implement stroke limit penalty forces for linear actuator"
```

---

### Task 3: Add sweepable parameter support

**Files:**
- Modify: `linkage-sim-rs/src/gui/state/blueprint_ops.rs:135-157`

- [ ] **Step 1: Add fields to `set_force_field`**

In `blueprint_ops.rs`, find the `ForceElement::LinearActuator(e)` match arm (~line 135). Replace:

```rust
        ForceElement::LinearActuator(e) => match field {
            "force" => { e.force = value; true }
            "speed_limit" => { e.speed_limit = value; true }
            _ => false,
        },
```

With:

```rust
        ForceElement::LinearActuator(e) => match field {
            "force" => { e.force = value; true }
            "speed_limit" => { e.speed_limit = value; true }
            "stroke_min" => { e.stroke_min = value; true }
            "stroke_max" => { e.stroke_max = value; true }
            "end_stop_stiffness" => { e.end_stop_stiffness = value; true }
            "end_stop_damping" => { e.end_stop_damping = value; true }
            "end_stop_restitution" => { e.end_stop_restitution = value; true }
            _ => false,
        },
```

- [ ] **Step 2: Add fields to `force_sweepable_fields`**

Find `ForceElement::LinearActuator(_)` in `force_sweepable_fields` (~line 157). Replace:

```rust
        ForceElement::LinearActuator(_) => vec!["force".into(), "speed_limit".into()],
```

With:

```rust
        ForceElement::LinearActuator(_) => vec![
            "force".into(), "speed_limit".into(),
            "stroke_min".into(), "stroke_max".into(),
            "end_stop_stiffness".into(), "end_stop_damping".into(), "end_stop_restitution".into(),
        ],
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check`
Expected: Compiles clean

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: add stroke limit fields to parametric sweep support"
```

---

### Task 4: Add UI controls in property panel

**Files:**
- Modify: `linkage-sim-rs/src/gui/property_panel/force_editor.rs:872-939`

- [ ] **Step 1: Add stroke limits UI section**

In `force_editor.rs`, inside the `ForceElement::LinearActuator(la)` arm, after the speed limit DragValue block (~line 913) and before the `draw_point_picker` calls (~line 916), add:

```rust
            // ── Stroke Limits ────────────────────────────────────────
            ui.separator();
            egui::CollapsingHeader::new("Stroke Limits")
                .default_open(la.stroke_max > 0.0)
                .show(ui, |ui| {
                    // Status hints
                    if la.stroke_min == 0.0 && la.stroke_max == 0.0 {
                        ui.label(egui::RichText::new("(disabled)").weak());
                    } else if la.stroke_min > 0.0 && la.stroke_max > 0.0 && la.stroke_min >= la.stroke_max {
                        ui.label(egui::RichText::new("(min \u{2265} max, limits inactive)").color(egui::Color32::from_rgb(255, 180, 40)));
                    }

                    let mut stroke_min = la.stroke_min;
                    ui.horizontal(|ui| {
                        ui.label("Min:");
                        let resp = ui.add(
                            egui::DragValue::new(&mut stroke_min)
                                .speed(0.001)
                                .range(0.0..=f64::MAX)
                                .suffix(" m"),
                        );
                        if la.stroke_min == 0.0 && la.stroke_max > 0.0 {
                            ui.label(egui::RichText::new("(inactive)").weak());
                        }
                        if resp.changed() {
                            let mut updated = la.clone();
                            updated.stroke_min = stroke_min;
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::LinearActuator(updated),
                            });
                        }
                    });

                    let mut stroke_max = la.stroke_max;
                    ui.horizontal(|ui| {
                        ui.label("Max:");
                        if ui.add(
                            egui::DragValue::new(&mut stroke_max)
                                .speed(0.001)
                                .range(0.0..=f64::MAX)
                                .suffix(" m"),
                        ).changed() {
                            let mut updated = la.clone();
                            updated.stroke_max = stroke_max;
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::LinearActuator(updated),
                            });
                        }
                    });

                    let mut stiffness = la.end_stop_stiffness;
                    ui.horizontal(|ui| {
                        ui.label("k:");
                        if ui.add(
                            egui::DragValue::new(&mut stiffness)
                                .speed(100.0)
                                .range(0.0..=f64::MAX)
                                .suffix(" N/m"),
                        ).changed() {
                            let mut updated = la.clone();
                            updated.end_stop_stiffness = stiffness;
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::LinearActuator(updated),
                            });
                        }
                    });

                    let mut damping = la.end_stop_damping;
                    ui.horizontal(|ui| {
                        ui.label("c:");
                        if ui.add(
                            egui::DragValue::new(&mut damping)
                                .speed(1.0)
                                .range(0.0..=f64::MAX)
                                .suffix(" N\u{00b7}s/m"),
                        ).changed() {
                            let mut updated = la.clone();
                            updated.end_stop_damping = damping;
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::LinearActuator(updated),
                            });
                        }
                    });

                    let mut restitution = la.end_stop_restitution;
                    ui.horizontal(|ui| {
                        ui.label("e:");
                        if ui.add(
                            egui::DragValue::new(&mut restitution)
                                .speed(0.01)
                                .range(0.0..=1.0),
                        ).changed() {
                            let mut updated = la.clone();
                            updated.end_stop_restitution = restitution;
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::LinearActuator(updated),
                            });
                        }
                    });
                });
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check`
Expected: Compiles clean

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: add stroke limits UI controls to actuator property panel"
```

---

### Task 5: Add canvas tick mark rendering

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas/rendering.rs:1067-1120`

- [ ] **Step 1: Add tick marks to actuator rendering**

In `rendering.rs`, inside the `ForceElement::LinearActuator(la)` rendering block, after the force label text (~line 1118) and before the closing `}` on line 1119, add:

```rust
                    // Draw stroke limit tick marks
                    let limits_active = la.stroke_max > 0.0 && la.stroke_max > la.stroke_min;
                    if limits_active && length > 2.0 {
                        let perp = Vec2::new(-dir.y, dir.x);
                        let tick_half = 6.0_f32;
                        let tick_stroke = Stroke::new(1.5, ACTUATOR_COLOR);

                        // World-space scale: pixels per meter
                        let scale = view.scale as f32;

                        if la.stroke_min > 0.0 {
                            let min_frac = (la.stroke_min as f32 * scale) / length;
                            if min_frac > 0.0 && min_frac < 1.0 {
                                let tick_pos = Pos2::new(
                                    start.x + delta.x * min_frac,
                                    start.y + delta.y * min_frac,
                                );
                                painter.line_segment(
                                    [
                                        Pos2::new(tick_pos.x + perp.x * tick_half, tick_pos.y + perp.y * tick_half),
                                        Pos2::new(tick_pos.x - perp.x * tick_half, tick_pos.y - perp.y * tick_half),
                                    ],
                                    tick_stroke,
                                );
                            }
                        }

                        if la.stroke_max > 0.0 {
                            let max_frac = (la.stroke_max as f32 * scale) / length;
                            if max_frac > 0.0 && max_frac < 1.0 {
                                let tick_pos = Pos2::new(
                                    start.x + delta.x * max_frac,
                                    start.y + delta.y * max_frac,
                                );
                                painter.line_segment(
                                    [
                                        Pos2::new(tick_pos.x + perp.x * tick_half, tick_pos.y + perp.y * tick_half),
                                        Pos2::new(tick_pos.x - perp.x * tick_half, tick_pos.y - perp.y * tick_half),
                                    ],
                                    tick_stroke,
                                );
                            }
                        }
                    }
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check`
Expected: Compiles clean

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "feat: render stroke limit tick marks on actuator canvas drawing"
```

---

### Task 6: Add serialization round-trip test

**Files:**
- Modify: `linkage-sim-rs/src/forces/elements/mod.rs` (tests)

- [ ] **Step 1: Add serialization test**

Add after the existing `serde_roundtrip_linear_actuator_defaults` test:

```rust
    #[test]
    fn serde_roundtrip_linear_actuator_with_stroke_limits() {
        let original = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "b1".into(),
            point_a: [0.1, 0.2],
            point_a_name: None,
            body_b: "b2".into(),
            point_b: [0.3, 0.4],
            point_b_name: None,
            force: 75.0,
            speed_limit: 1.5,
            stroke_min: 0.2,
            stroke_max: 0.8,
            end_stop_stiffness: 5000.0,
            end_stop_damping: 20.0,
            end_stop_restitution: 0.3,
        });

        let json = serde_json::to_string(&original).unwrap();
        let loaded: ForceElement = serde_json::from_str(&json).unwrap();

        match &loaded {
            ForceElement::LinearActuator(la) => {
                assert_abs_diff_eq!(la.stroke_min, 0.2, epsilon = 1e-15);
                assert_abs_diff_eq!(la.stroke_max, 0.8, epsilon = 1e-15);
                assert_abs_diff_eq!(la.end_stop_stiffness, 5000.0, epsilon = 1e-15);
                assert_abs_diff_eq!(la.end_stop_damping, 20.0, epsilon = 1e-15);
                assert_abs_diff_eq!(la.end_stop_restitution, 0.3, epsilon = 1e-15);
            }
            _ => panic!("Expected LinearActuator"),
        }
    }

    #[test]
    fn serde_old_actuator_json_loads_with_default_stroke_limits() {
        // Simulate old JSON without stroke fields (internally-tagged enum format)
        let json = r#"{
            "type": "LinearActuator",
            "body_a": "b1",
            "point_a": [0.0, 0.0],
            "body_b": "b2",
            "point_b": [1.0, 0.0],
            "force": 50.0,
            "speed_limit": 0.0
        }"#;

        let loaded: ForceElement = serde_json::from_str(json).unwrap();

        match &loaded {
            ForceElement::LinearActuator(la) => {
                assert_abs_diff_eq!(la.stroke_min, 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(la.stroke_max, 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(la.end_stop_stiffness, 10000.0, epsilon = 1e-15);
                assert_abs_diff_eq!(la.end_stop_damping, 10.0, epsilon = 1e-15);
                assert_abs_diff_eq!(la.end_stop_restitution, 0.5, epsilon = 1e-15);
            }
            _ => panic!("Expected LinearActuator"),
        }
    }
```

- [ ] **Step 2: Run serialization tests**

Run: `cargo test --lib serde_roundtrip_linear_actuator`
Expected: All pass (including old backward-compat test)

- [ ] **Step 3: Run full test suite**

Run: `cargo test --lib`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test: add stroke limit serialization round-trip tests"
```

---

### Task 7: Update SYSTEM.md

**Files:**
- Modify: `linkage-sim-rs/SYSTEM.md`

- [ ] **Step 1: Update SYSTEM.md**

No structural changes — the feature adds fields to an existing struct and modifies existing files. Add a note in the `forces/elements/` section that `LinearActuatorElement` now supports stroke limits with penalty forces.

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "docs: update SYSTEM.md with actuator stroke limits"
```
