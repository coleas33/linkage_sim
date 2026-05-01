# Actuator-Driven Linkages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LinearDriver constraint that sweeps actuator stroke (replacing crank angle), moveable ground pivots, and a new Chebyshev Lambda + Actuator sample with flipped (+y) orientation.

**Architecture:** A new `LinearDriver` struct implements the `Constraint` trait, prescribing distance between two body points as d(t). It mirrors RevoluteDriver's pattern (DriverFn closures, DriverMeta for serialization). The sweep infrastructure detects driver type and switches x-axis from angle to stroke. Ground pivot dragging is a canvas interaction addition. The new sample uses custom proportions (ground=76, crank=44.4, coupler=91.9+91.9, rocker=91.9) with alpha-beta branch for +y orientation.

**Tech Stack:** Rust, nalgebra, egui, existing Mechanism/Constraint/ForceElement APIs

**Future (noted for later):** Forward dynamics actuator mode (B) — actuator applies force, dynamics simulator computes motion.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/core/linear_driver.rs` | LinearDriver constraint implementation |
| Modify | `src/core/driver.rs` | Add LinearLength variant to DriverMeta |
| Modify | `src/core/mod.rs` | Export linear_driver module |
| Modify | `src/core/mechanism.rs` | Store and iterate LinearDrivers |
| Modify | `src/gui/sweep.rs` | Detect linear driver, adapt sweep range/labels |
| Modify | `src/gui/state/mod.rs` | Add linear driver state fields |
| Modify | `src/gui/state/blueprint_ops.rs` | Build linear driver from blueprint |
| Modify | `src/gui/canvas/interaction.rs` | Ground pivot dragging |
| Modify | `src/gui/samples/fourbar.rs` | New Chebyshev Lambda + Actuator sample |
| Modify | `src/gui/samples/mod.rs` | Register new sample |
| Modify | `src/gui/samples/helpers.rs` | Add alpha-beta branch option to fourbar_initial_q0 |
| Modify | `src/io/schema.rs` | Serialize linear driver in JSON |
| Modify | `src/io/to_json.rs` | Linear driver to JSON |
| Modify | `src/io/serialization.rs` | Linear driver round-trip |

---

### Task 1: LinearDriver constraint struct

The core solver primitive. Constrains distance between two body points as a function of time.

**Files:**
- Create: `src/core/linear_driver.rs`
- Modify: `src/core/driver.rs` (DriverMeta enum)
- Modify: `src/core/mod.rs` (module export)

**Math:**

Constraint: `Phi = |P_b - P_a| - d(t) = 0`

Where P_a, P_b are global positions of points on body_a, body_b.

Jacobian (1 x n_coords): For each body, the row has entries at (x, y, theta) indices:
```
dPhi/dr_a = -n^T          (1x2, unit direction from A to B, negated)
dPhi/dtheta_a = -n^T * B(theta_a) * s_a   (scalar)
dPhi/dr_b = n^T            (1x2)
dPhi/dtheta_b = n^T * B(theta_b) * s_b    (scalar)
```
Where n = (P_b - P_a) / L, B(theta) = [[-sin, -cos], [cos, -sin]].

Phi_t = -d'(t)

Gamma (acceleration RHS): `gamma = d''(t) - v_perp^2 / L`
Where v_perp is the velocity component perpendicular to the line of action.

- [ ] **Step 1: Write failing test for LinearDriver constraint evaluation**

In a new test module in `linear_driver.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::State;
    use crate::core::body::{make_bar, make_ground, Body};
    use crate::core::mechanism::Mechanism;
    use nalgebra::DVector;

    fn build_two_body_mech() -> (Mechanism, DVector<f64>) {
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 0.0, 0.0);
        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A").unwrap();
        mech.build().unwrap();
        let mut q = mech.state().make_q();
        mech.state().set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        (mech, q)
    }

    #[test]
    fn linear_driver_constraint_at_rest() {
        let (mech, q) = build_two_body_mech();
        let state = mech.state();
        // Driver between ground "O" at (0,0) and bar "B" at local (1,0).
        // At theta=0, bar "B" is at global (1, 0). Distance from O = 1.0.
        let driver = constant_velocity_linear_driver(
            "LD1", "ground", [0.0, 0.0], "bar", [1.0, 0.0],
            1.0, 1.0, // velocity=1, length_0=1
        );
        let phi = driver.constraint(state, &q, 0.0);
        assert!((phi[0]).abs() < 1e-10, "constraint should be zero at initial length");
    }

    #[test]
    fn linear_driver_jacobian_is_correct() {
        let (mech, q) = build_two_body_mech();
        let state = mech.state();
        let driver = constant_velocity_linear_driver(
            "LD1", "ground", [0.0, 0.0], "bar", [1.0, 0.0],
            1.0, 1.0,
        );
        let jac = driver.jacobian(state, &q, 0.0);
        // Numerical Jacobian check via finite differences
        let eps = 1e-7;
        let mut q_pert = q.clone();
        for col in 0..q.len() {
            q_pert[col] += eps;
            let phi_plus = driver.constraint(state, &q_pert, 0.0);
            q_pert[col] -= 2.0 * eps;
            let phi_minus = driver.constraint(state, &q_pert, 0.0);
            q_pert[col] += eps; // restore
            let numerical = (phi_plus[0] - phi_minus[0]) / (2.0 * eps);
            assert!(
                (jac[(0, col)] - numerical).abs() < 1e-5,
                "Jacobian column {} mismatch: analytical={}, numerical={}",
                col, jac[(0, col)], numerical
            );
        }
    }
}
```

- [ ] **Step 2: Implement LinearDriver struct and Constraint trait**

Create `src/core/linear_driver.rs` with:
- `LinearDriver` struct (id, body_a_id, point_a, body_b_id, point_b, driver_fn, meta)
- `Constraint` impl with constraint(), jacobian(), phi_t(), gamma()
- `constant_velocity_linear_driver()` factory: d(t) = length_0 + velocity * t

Add `DriverMeta::LinearLength { velocity: f64, length_0: f64 }` variant to `driver.rs`.

Add `pub mod linear_driver;` to `src/core/mod.rs`.

- [ ] **Step 3: Run tests**

Run: `cargo test --lib linear_driver`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/core/linear_driver.rs linkage-sim-rs/src/core/driver.rs linkage-sim-rs/src/core/mod.rs
git commit -m "feat: add LinearDriver constraint for actuator stroke control"
```

---

### Task 2: Integrate LinearDriver into Mechanism

**Files:**
- Modify: `src/core/mechanism.rs`

- [ ] **Step 1: Add storage and iteration**

In `Mechanism`:
```rust
linear_drivers: Vec<LinearDriver>,
```

Update `all_constraints()` to chain linear_drivers:
```rust
pub fn all_constraints(&self) -> impl Iterator<Item = &dyn Constraint> {
    self.joints.iter().map(|j| j as &dyn Constraint)
        .chain(self.drivers.iter().map(|d| d as &dyn Constraint))
        .chain(self.linear_drivers.iter().map(|d| d as &dyn Constraint))
}
```

Add `add_linear_driver()` method. Add `linear_drivers()` accessor.

Add a `driver_type()` helper that returns an enum indicating whether the mechanism uses revolute, linear, or no driver.

- [ ] **Step 2: Write test**

```rust
#[test]
fn mechanism_with_linear_driver_builds() {
    // Build a simple mechanism, add a linear driver, verify it builds and solves
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test --lib mechanism_with_linear`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: integrate LinearDriver storage and iteration in Mechanism"
```

---

### Task 3: Add alpha-beta branch option to fourbar_initial_q0

The existing `fourbar_initial_q0` always uses alpha+beta (mechanism below ground line). The new sample needs alpha-beta (+y orientation).

**Files:**
- Modify: `src/gui/samples/helpers.rs`

- [ ] **Step 1: Add `above` parameter to fourbar_initial_q0**

Add an `above: bool` parameter. When `above=true`, use `alpha - beta` instead of `alpha + beta`:

```rust
let theta_rocker = if above {
    alpha - beta + PI
} else {
    alpha + beta + PI
};
```

Update all existing call sites to pass `above: false` (preserving current behavior).

- [ ] **Step 2: Run full test suite to verify no regressions**

Run: `cargo test --lib`
Expected: all tests pass (behavior unchanged for existing samples)

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: add above/below branch selection to fourbar_initial_q0"
```

---

### Task 4: Adapt sweep for linear drivers

**Files:**
- Modify: `src/gui/sweep.rs`
- Modify: `src/gui/state/mod.rs` (add driver type state)

- [ ] **Step 1: Detect driver type in sweep**

In `compute_sweep_data()`, detect if the mechanism has a linear driver:
- If revolute driver: sweep 0-360 degrees as before
- If linear driver: sweep from stroke_min to stroke_max in the same number of steps

The linear driver's `DriverMeta::LinearLength { velocity, length_0 }` gives:
- stroke at step i = length_0 + velocity * t_i
- t_i = i * step_size / velocity

Add a `SweepMode` enum:
```rust
pub enum SweepMode {
    Angle { omega: f64, theta_0: f64 },
    Stroke { velocity: f64, length_0: f64, stroke_min: f64, stroke_max: f64 },
}
```

- [ ] **Step 2: Update sweep x-axis data**

For linear driver sweeps, `data.angles_deg` is repurposed as the x-axis values (stroke in mm instead of degrees). Add a field to `SweepData`:
```rust
pub sweep_mode: SweepMode,
```

The plot panel can use this to set axis labels ("Crank Angle (deg)" vs "Actuator Stroke (mm)").

- [ ] **Step 3: Update inverse dynamics torque extraction**

For linear drivers, the last lambda gives actuator FORCE (N), not torque (N*m). The data field name `inverse_dynamics_torques` still works but the units differ. Add to SweepData:
```rust
pub driver_force_unit: String,  // "N*m" or "N"
```

- [ ] **Step 4: Write test**

```rust
#[test]
fn sweep_with_linear_driver_produces_stroke_data() {
    // Build mechanism with linear driver, run sweep, verify x-axis is stroke values
}
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: adapt sweep infrastructure for linear driver stroke mode"
```

---

### Task 5: Moveable ground pivots in canvas

**Files:**
- Modify: `src/gui/canvas/interaction.rs`
- Modify: `src/gui/state/entity_crud.rs` (add update_ground_pivot method)

- [ ] **Step 1: Add drag handler for ground attachment points**

In `handle_interaction()`, detect when a drag starts on a ground attachment point:
1. Check `attachment_hit_targets` for hits where `body_id == "ground"`
2. On drag start: store the pivot name and start position
3. On drag: compute new world position from screen delta
4. On drag end: update the ground pivot position in the blueprint and rebuild

- [ ] **Step 2: Add AppState method to update ground pivot position**

```rust
pub fn update_ground_pivot_position(&mut self, name: &str, x: f64, y: f64) {
    self.push_undo();
    if let Some(bp) = &mut self.blueprint {
        if let Some(ground_body) = bp.bodies.get_mut("ground") {
            if let Some(point) = ground_body.attachment_points.get_mut(name) {
                point[0] = x;
                point[1] = y;
            }
        }
    }
    self.rebuild();
}
```

- [ ] **Step 3: Write test**

```rust
#[test]
fn update_ground_pivot_position_rebuilds_mechanism() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);
    let mech_before = state.mechanism.as_ref().unwrap().bodies()["ground"]
        .attachment_points["O2"].clone();
    state.update_ground_pivot_position("O2", 1.0, 1.0);
    let mech_after = state.mechanism.as_ref().unwrap().bodies()["ground"]
        .attachment_points["O2"].clone();
    assert_ne!(mech_before, mech_after);
}
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: allow dragging ground pivots in canvas"
```

---

### Task 6: New Chebyshev Lambda + Actuator sample

**Files:**
- Modify: `src/gui/samples/fourbar.rs` (new builder)
- Modify: `src/gui/samples/mod.rs` (register sample)

- [ ] **Step 1: Add ChebyshevLambdaActuator to SampleMechanism enum**

Add variant, label "Chebyshev Lambda + Actuator", add to `all()`, add match arm.

- [ ] **Step 2: Write builder function**

```rust
/// Chebyshev lambda linkage driven by a linear actuator.
///
/// Custom proportions: ground=76mm, crank=44.4mm, coupler AB=91.9mm,
/// rocker=91.9mm, extension BM=91.9mm. Flipped to +y orientation.
/// Linear actuator parallel to the straight-line trace, driving the endpoint M.
fn build_chebyshev_lambda_actuator(
    _driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    // Scale to meters: mm / 1000
    let ground_len = 0.076;
    let crank_len = 0.0444;
    let coupler_ab = 0.0919;
    let rocker_len = 0.0919;
    let extension_bm = 0.0919;
    let total_coupler = coupler_ab + extension_bm; // 0.1838

    let o2 = (0.0, 0.0);
    let o4 = (ground_len, 0.0);

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    // ... (add actuator base point on ground)

    let crank = make_bar("crank", "A", "B", crank_len, 0.0, 0.0);

    let mut coupler = make_bar("coupler", "B", "M", total_coupler, 0.0, 0.0);
    coupler.add_attachment_point("C", coupler_ab, 0.0).map_err(|e| e.to_string())?;
    coupler.add_coupler_point("M", total_coupler, 0.0).unwrap();

    let rocker = make_bar("rocker", "C", "D", rocker_len, 0.0, 0.0);

    // Build 4-bar joints (same as standard lambda)
    // ...

    // Linear actuator: ground base → coupler endpoint M
    // Position actuator base so it's parallel to the straight-line trace.
    // The straight line is at approximately y = some constant.
    // Place actuator base on ground at x = -some_offset, y = trace_y.
    // The exact position: compute M's y at the midpoint of the stroke,
    // place actuator base at the left side at the same y.

    // NO revolute driver — mechanism driven by linear driver on actuator axis.
    // Add linear driver between ground actuator base and coupler M.

    // Use alpha-beta branch (above=true) for +y orientation.
    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4,
        crank_len, coupler_ab, rocker_len,
        0.0,
        "crank", "coupler", "rocker",
        true, // above = +y
    );

    Ok((mech, q0))
}
```

- [ ] **Step 3: Write test**

```rust
#[test]
fn chebyshev_lambda_actuator_builds_and_solves() {
    let (mech, q0) = build_sample(SampleMechanism::ChebyshevLambdaActuator);
    assert!(mech.is_built());
    assert!(mech.linear_drivers().len() == 1, "should have one linear driver");
    assert!(mech.drivers().is_empty(), "should have no revolute driver");
}
```

- [ ] **Step 4: Verify the straight-line trace is in +y**

```rust
#[test]
fn chebyshev_lambda_actuator_trace_in_positive_y() {
    // Sweep and verify the M endpoint y-values are positive
    // in the straight portion
}
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add Chebyshev Lambda + Actuator sample with +y orientation"
```

---

### Task 7: JSON serialization for linear drivers

**Files:**
- Modify: `src/io/schema.rs` (add LinearDriverJson)
- Modify: `src/io/to_json.rs` (serialize)
- Modify: `src/io/serialization.rs` (deserialize + round-trip test)

- [ ] **Step 1: Add LinearDriverJson to schema**

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearDriverJson {
    pub body_a: String,
    pub point_a: [f64; 2],
    pub body_b: String,
    pub point_b: [f64; 2],
    pub velocity: f64,
    pub length_0: f64,
}
```

Add to MechanismJson:
```rust
#[serde(default, skip_serializing_if = "Vec::is_empty")]
pub linear_drivers: Vec<LinearDriverJson>,
```

- [ ] **Step 2: Implement serialization/deserialization in to_json.rs and serialization.rs**

- [ ] **Step 3: Round-trip test**

```rust
#[test]
fn linear_driver_round_trips_through_json() {
    // Build mechanism with linear driver, save, reload, verify driver preserved
}
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: serialize linear drivers in JSON schema"
```

---

### Task 8: GUI driver type selection

**Files:**
- Modify: `src/gui/input_panel.rs` (driver selector)
- Modify: `src/gui/canvas/context_menu.rs` (right-click to set linear driver)

- [ ] **Step 1: Add driver type toggle**

In the Driver section of input_panel.rs, add a selector between "Motor (Revolute)" and "Actuator (Linear)" drive modes. When Actuator is selected:
- Show actuator stroke range controls
- Hide crank angle / speed controls
- The sweep mode switches automatically

- [ ] **Step 2: Allow right-click on actuator to set as driver**

In the context menu, when right-clicking on a linear actuator force element, add "Set as Linear Driver" option that converts it to a LinearDriver constraint.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: add actuator drive mode selection in GUI"
```

---

### Task 9: Update plot labels for linear driver mode

**Files:**
- Modify: `src/gui/plot_panel.rs`

- [ ] **Step 1: Detect sweep mode and update labels**

When `sweep_data.sweep_mode` is `Stroke`:
- X-axis: "Actuator Stroke (mm)" instead of "Crank Angle (deg)"
- Driver torque plot: "Actuator Force (N)" instead of "Driver Torque (N*m)"

- [ ] **Step 2: Commit**

```bash
git commit -m "feat: update plot labels for linear driver sweep mode"
```

---

### Task 10: Integration tests and documentation

- [ ] **Step 1: Full test suite**

Run: `cargo test --lib`
Expected: all tests pass

- [ ] **Step 2: Update docs**

Add linear driver documentation to `docs/reference/FORCE_ELEMENTS.md` and `linkage-sim-rs/SYSTEM.md`.

- [ ] **Step 3: Commit and push**

```bash
git commit -m "docs: document linear driver and actuator-driven sweep"
```
