# Click-to-Place Forces + Parallelogram Actuator Sample

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users place two-point force elements (spring, damper, gas spring, actuator) by clicking two points on the canvas, and add a sample parallelogram mechanism with a linear actuator.

**Architecture:** New `EditorTool::PlaceForce` variant mirrors the existing `DrawLink` tool pattern — two-click state machine with snap-to-point. Toolbar buttons for two-point forces switch to this mode instead of immediately creating forces with defaults. Mount points are added to canvas hit targets so they're snappable. New `ParallelogramActuator` sample showcases compound force expansion.

**Tech Stack:** Rust, egui, existing `Mechanism` / `ForceElement` / `AppState` APIs

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/gui/state.rs:237-249` | Add `PlaceForce` variant to `EditorTool` |
| Modify | `src/gui/state.rs:~713` | Add `place_force_state: Option<PlaceForceState>` to `AppState` |
| Create | `src/gui/state.rs` (new structs) | `PlaceForceState`, `PlaceForceStart` structs |
| Modify | `src/gui/canvas.rs:362-373` | Include mount points in `attachment_hit_targets` |
| Modify | `src/gui/canvas.rs:~988-993` | Clear `place_force_state` on Escape |
| Modify | `src/gui/canvas.rs:~785` | Add hint text for `PlaceForce` tool |
| Modify | `src/gui/canvas.rs:1268-1271` | Exclude `PlaceForce` from selection click guard |
| Modify | `src/gui/canvas.rs:1277-1311` | Add `PlaceForce` arm to exhaustive match |
| Modify | `src/gui/canvas.rs` (new section) | Two-click interaction handler for `PlaceForce` |
| Modify | `src/gui/force_toolbar.rs:8-11` | Add `EnterPlaceMode(ForceElement)` variant to `PendingForceAdd` |
| Modify | `src/gui/force_toolbar.rs:108-140` | Change two-point force buttons to emit `EnterPlaceMode` |
| Modify | `src/gui/mod.rs:664-669` | Handle `EnterPlaceMode` → set tool + state |
| Modify | `src/gui/samples.rs:14-33` | Add `ParallelogramActuator` to `SampleMechanism` enum |
| Modify | `src/gui/samples.rs:35-77` | Add label + `all()` entry |
| Modify | `src/gui/samples.rs:90-112` | Add match arm in `build_sample_with_driver` |
| Create | `src/gui/samples.rs` (new fn) | `build_parallelogram_actuator` builder |
| Create | `tests/parallelogram_actuator_sample.rs` | Integration test for the new sample |

---

### Task 1: Add mount points to canvas hit targets

Mount points are rendered as diamonds but not included in `attachment_hit_targets`, so they can't be snapped to. Both `PlaceForce` and existing tools benefit from this.

**Files:**
- Modify: `src/gui/canvas.rs:362-373`

- [ ] **Step 1: Add mount point hit targets after rendering diamonds**

In `canvas.rs`, change the mount-point rendering loop (line ~368). Rename `_name` to `name` (remove underscore prefix) and add each mount point to `attachment_hit_targets`:

```rust
// Draw mount points as diamonds AND register as hit targets
for (name, local) in &body.mount_points {  // NOTE: was `_name` — remove underscore
    let global = mech_state.body_point_global(body_id, local, q);
    let sp = view.world_to_screen(global.x, global.y);
    let screen_pos = Pos2::new(sp[0], sp[1]);
    draw_diamond_marker(&painter, screen_pos, MOUNT_POINT_RADIUS, MOUNT_POINT_COLOR);

    attachment_hit_targets.push(AttachmentHit {
        screen_pos,
        world_pos: [global.x, global.y],
        body_id: body_id.clone(),
        point_name: name.clone(),
    });
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p linkage-sim-rs 2>&1 | tail -5`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat: include mount points in canvas snap targets"
```

---

### Task 2: Add `PlaceForce` tool state to `EditorTool` and `AppState`

**Files:**
- Modify: `src/gui/state.rs:237-249` (EditorTool enum)
- Modify: `src/gui/state.rs:~713` (AppState fields)

- [ ] **Step 1: Add state structs**

Add after `DrawLinkStart` (around line 799):

```rust
/// Tracks the state of a Place Force two-click interaction.
#[derive(Debug, Clone)]
pub struct PlaceForceState {
    /// The force element template (type + default parameters).
    /// Body IDs and point coordinates will be filled in by the clicks.
    pub force_template: ForceElement,
    /// Set after the first click.
    pub start: Option<PlaceForceStart>,
}

/// First click of a Place Force interaction.
#[derive(Debug, Clone)]
pub struct PlaceForceStart {
    /// World coordinates of point A.
    pub world_pos: [f64; 2],
    /// Body ID that point A belongs to.
    pub body_id: String,
    /// Named point (attachment or mount) if snapped, None for raw coords.
    pub point_name: Option<String>,
}
```

- [ ] **Step 2: Add `PlaceForce` variant to `EditorTool`**

```rust
pub enum EditorTool {
    Select,
    DrawLink,
    AddBody,
    AddGroundPivot,
    /// Two-click placement of a two-point force element.
    PlaceForce,
}
```

- [ ] **Step 3: Add `place_force_state` field to `AppState`**

Add alongside `draw_link_start` and `add_body_state` (around line 716):

```rust
/// Two-click force placement state. None when not in PlaceForce mode.
pub place_force_state: Option<PlaceForceState>,
```

Initialize to `None` in the `Default` impl (line ~801-913). Insert `place_force_state: None,` after the existing `add_body_state: None,` field (around line 866).

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p linkage-sim-rs 2>&1 | tail -10`
Expected: may have warnings about unused fields (ok), no errors. Fix any exhaustive match issues on `EditorTool`.

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/gui/state.rs
git commit -m "feat: add PlaceForce tool state structs"
```

---

### Task 3: Wire toolbar to enter PlaceForce mode

Change the four two-point force buttons (Linear Spring, Linear Damper, Gas Spring, Linear Actuator) to enter PlaceForce mode instead of immediately creating a force.

**Files:**
- Modify: `src/gui/force_toolbar.rs:8-11` (PendingForceAdd enum)
- Modify: `src/gui/force_toolbar.rs:108-140` (button handlers)
- Modify: `src/gui/mod.rs:664-669` (pending dispatch)

- [ ] **Step 1: Add `EnterPlaceMode` variant to `PendingForceAdd`**

```rust
pub enum PendingForceAdd {
    Add(ForceElement),
    /// Enter two-click placement mode with this force template.
    EnterPlaceMode(ForceElement),
}
```

- [ ] **Step 2: Change two-point force buttons to emit `EnterPlaceMode`**

In `force_toolbar.rs`, change the four two-point force button handlers (lines 108-140) from `PendingForceAdd::Add(...)` to `PendingForceAdd::EnterPlaceMode(...)`. Keep the same force template construction — body IDs and points will be placeholder values that get overwritten by clicks.

Example for Linear Spring:
```rust
if ui.button("Linear Spring").clicked() {
    pending = Some(PendingForceAdd::EnterPlaceMode(ForceElement::LinearSpring(LinearSpringElement {
        body_a: String::new(), point_a: [0.0, 0.0], point_a_name: None,
        body_b: String::new(), point_b: [0.0, 0.0], point_b_name: None,
        stiffness: 100.0, free_length: 0.1,
    })));
    ui.close();
}
```

Note: body IDs are now empty strings since they'll be filled by clicks. Remove the `if let Some((ref a, ref b))` guard around these four buttons — the user no longer needs to pre-select bodies. Keep the guard for single-body elements (ExternalForce, ExternalTorque) and joint torques which don't use PlaceForce mode.

- [ ] **Step 3: Handle `EnterPlaceMode` in `mod.rs`**

```rust
if let Some(force_add) = force_toolbar::draw_force_toolbar(ui, &self.state) {
    match force_add {
        force_toolbar::PendingForceAdd::Add(force) => {
            self.state.add_force_element(force);
        }
        force_toolbar::PendingForceAdd::EnterPlaceMode(template) => {
            self.state.active_tool = EditorTool::PlaceForce;
            self.state.place_force_state = Some(PlaceForceState {
                force_template: template,
                start: None,
            });
        }
    }
}
```

Add the necessary imports. Find the existing `use super::state::` import line in `mod.rs` and add `PlaceForceState` and `EditorTool` to it. For example: `use super::state::{AppState, EditorTool, PlaceForceState, ...};`

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p linkage-sim-rs 2>&1 | tail -10`
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/gui/force_toolbar.rs linkage-sim-rs/src/gui/mod.rs
git commit -m "feat: toolbar enters PlaceForce mode for two-point forces"
```

---

### Task 4: Canvas interaction handler for PlaceForce

The core two-click interaction: click 1 picks point A, preview line follows cursor, click 2 picks point B and creates the force.

**Files:**
- Modify: `src/gui/canvas.rs:~785` (hint text)
- Modify: `src/gui/canvas.rs:~988-993` (escape handler)
- Modify: `src/gui/canvas.rs` (new interaction section after DrawLink)

- [ ] **Step 1: Add hint text for PlaceForce tool**

In the hint text match (around line 766-789), add:

```rust
EditorTool::PlaceForce => {
    if state.place_force_state.as_ref().and_then(|s| s.start.as_ref()).is_some() {
        Some("Click to place force endpoint B (Esc to cancel)".to_string())
    } else {
        Some("Click to place force endpoint A (Esc to cancel)".to_string())
    }
}
```

- [ ] **Step 2: Clear `place_force_state` on Escape**

In the escape handler (line ~990-993), add:

```rust
state.place_force_state = None;
```

- [ ] **Step 2b: Exclude `PlaceForce` from selection click guard**

At `canvas.rs:1268-1271`, the guard prevents selection clicks during DrawLink/AddBody. Add PlaceForce:

```rust
    && state.active_tool != EditorTool::PlaceForce
```

- [ ] **Step 2c: Add `PlaceForce` arm to exhaustive match**

At `canvas.rs:1277-1311`, the `match state.active_tool` is exhaustive. Add:

```rust
EditorTool::PlaceForce => {
    // Handled by PlaceForce interaction section above.
}
```

- [ ] **Step 3: Add PlaceForce interaction handler**

Add a new section after the DrawLink handler (after line ~1169). This handles both clicks and preview rendering:

```rust
// ── Interaction: Place Force tool ───────────────────────────────────
if state.active_tool == EditorTool::PlaceForce {
    use crate::gui::state::PlaceForceStart;

    // Highlight all snap targets while in placement mode.
    for hit in &attachment_hit_targets {
        painter.circle_stroke(
            hit.screen_pos,
            HIT_RADIUS,
            Stroke::new(1.0, JOINT_CREATE_HIGHLIGHT.linear_multiply(0.3)),
        );
    }

    // Preview line from start to cursor after first click.
    if let Some(ref pf_state) = state.place_force_state {
        if let Some(ref start) = pf_state.start {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                let [sx, sy] = start.world_pos;
                let snap_end = find_nearest_attachment(pos);
                let (ex, ey, end_snapped) = if let Some(hit) = snap_end {
                    (hit.world_pos[0], hit.world_pos[1], true)
                } else {
                    let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                    (wx, wy, false)
                };

                let start_screen = state.view.world_to_screen(sx, sy);
                let end_screen = state.view.world_to_screen(ex, ey);

                // Dashed preview line (force, not link — use force color).
                let force_preview_color = Color32::from_rgb(255, 165, 80);
                painter.line_segment(
                    [
                        Pos2::new(start_screen[0], start_screen[1]),
                        Pos2::new(end_screen[0], end_screen[1]),
                    ],
                    Stroke::new(2.0, force_preview_color),
                );
                painter.circle_filled(
                    Pos2::new(start_screen[0], start_screen[1]),
                    JOINT_RADIUS,
                    force_preview_color,
                );
                let end_color = if end_snapped {
                    JOINT_CREATE_HIGHLIGHT
                } else {
                    force_preview_color
                };
                painter.circle_filled(
                    Pos2::new(end_screen[0], end_screen[1]),
                    JOINT_RADIUS,
                    end_color,
                );
            }
        }
    }

    // Handle clicks.
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let snap_hit = find_nearest_attachment(pos);

            let (world_pos, body_id, point_name) = if let Some(hit) = snap_hit {
                (hit.world_pos, hit.body_id.clone(), Some(hit.point_name.clone()))
            } else {
                // No snap — find nearest body via segment projection.
                let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
                if let Some(seg_hit) = find_nearest_body_segment(pos, &body_segments, 8.0) {
                    (seg_hit.world_pos, seg_hit.body_id.clone(), None)
                } else {
                    // Fallback: use ground at raw world coordinates.
                    ([wx, wy], GROUND_ID.to_string(), None)
                }
            };

            let has_start = state.place_force_state.as_ref()
                .map(|s| s.start.is_some())
                .unwrap_or(false);

            if !has_start {
                // First click — record point A.
                if let Some(ref mut pf_state) = state.place_force_state {
                    pf_state.start = Some(PlaceForceStart {
                        world_pos,
                        body_id,
                        point_name,
                    });
                }
            } else {
                // Second click — create the force and exit tool.
                if let Some(pf_state) = state.place_force_state.take() {
                    let start = pf_state.start.unwrap();

                    // Convert world positions to body-local coordinates.
                    let [la_x, la_y] = state.world_to_body_local(
                        &start.body_id, start.world_pos[0], start.world_pos[1],
                    );
                    let [lb_x, lb_y] = state.world_to_body_local(
                        &body_id, world_pos[0], world_pos[1],
                    );

                    // Fill in the force template with actual body/point info.
                    let force = fill_force_template(
                        &pf_state.force_template,
                        &start.body_id, [la_x, la_y], start.point_name,
                        &body_id, [lb_x, lb_y], point_name,
                    );

                    // Note: add_force_element() calls push_undo() internally.
                    state.add_force_element(force);
                    state.active_tool = EditorTool::Select;
                }
            }
        }
    }
}
```

- [ ] **Step 4: Add `fill_force_template` helper**

Add this helper in `canvas.rs` (or at the bottom of the PlaceForce section):

```rust
/// Fill a force element template with actual body IDs and point coordinates.
fn fill_force_template(
    template: &ForceElement,
    body_a: &str, point_a: [f64; 2], point_a_name: Option<String>,
    body_b: &str, point_b: [f64; 2], point_b_name: Option<String>,
) -> ForceElement {
    match template {
        ForceElement::LinearSpring(s) => ForceElement::LinearSpring(LinearSpringElement {
            body_a: body_a.to_string(), point_a, point_a_name,
            body_b: body_b.to_string(), point_b, point_b_name,
            ..s.clone()
        }),
        ForceElement::LinearDamper(d) => ForceElement::LinearDamper(LinearDamperElement {
            body_a: body_a.to_string(), point_a, point_a_name,
            body_b: body_b.to_string(), point_b, point_b_name,
            ..d.clone()
        }),
        ForceElement::GasSpring(g) => ForceElement::GasSpring(GasSpringElement {
            body_a: body_a.to_string(), point_a, point_a_name,
            body_b: body_b.to_string(), point_b, point_b_name,
            ..g.clone()
        }),
        ForceElement::LinearActuator(a) => ForceElement::LinearActuator(LinearActuatorElement {
            body_a: body_a.to_string(), point_a, point_a_name,
            body_b: body_b.to_string(), point_b, point_b_name,
            ..a.clone()
        }),
        other => other.clone(),
    }
}
```

- [ ] **Step 5: Fix imports at top of canvas.rs**

The existing import at line 7 is `use crate::forces::elements::ForceElement;`. Replace it with a wildcard import to get the individual struct types needed by `fill_force_template`:

```rust
use crate::forces::elements::*;  // ForceElement + LinearSpringElement, etc.
```

Also ensure `GROUND_ID` is imported:
```rust
use crate::core::state::GROUND_ID;
```

Check which are already imported and add only what's missing.

- [ ] **Step 6: Verify compilation**

Run: `cargo check -p linkage-sim-rs 2>&1 | tail -10`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat: two-click canvas interaction for placing force elements"
```

---

### Task 5: ParallelogramActuator sample mechanism

Build a parallelogram 4-bar with a linear actuator from a new ground pivot to the crank midpoint (as a mount point), showcasing compound force expansion.

**Files:**
- Modify: `src/gui/samples.rs:14-33` (enum)
- Modify: `src/gui/samples.rs:35-77` (label, all())
- Modify: `src/gui/samples.rs:90-112` (match arm)
- Create: new builder function in `src/gui/samples.rs`

- [ ] **Step 1: Add `ParallelogramActuator` to `SampleMechanism` enum**

```rust
ParallelogramActuator,
```

- [ ] **Step 2: Add label**

```rust
SampleMechanism::ParallelogramActuator => "Parallelogram + Actuator",
```

- [ ] **Step 3: Add to `all()` array**

Insert after `Parallelogram`:
```rust
SampleMechanism::ParallelogramActuator,
```

- [ ] **Step 4: Add match arm in `build_sample_with_driver`**

```rust
SampleMechanism::ParallelogramActuator => build_parallelogram_actuator(driver_joint_id),
```

- [ ] **Step 4b: Add force element imports to samples.rs**

`samples.rs` does not import from `crate::forces::elements`. Add at the top of the file:

```rust
use crate::forces::elements::{ForceElement, LinearActuatorElement};
```

- [ ] **Step 5: Write the builder function**

The existing parallelogram uses `build_standard_fourbar` with links d=4, a=2, b=4, c=2. We need to build this manually (not reuse `build_standard_fourbar`) because we need to:
- Add a mount point at the crank midpoint
- Add a new ground pivot for the actuator base
- Add the linear actuator force element

```rust
/// Parallelogram 4-bar with a linear actuator driving the crank.
///
/// Same geometry as `Parallelogram` (d=4, a=2, b=4, c=2) but with:
/// - A mount point "M" at the crank midpoint (1.0, 0.0) in crank-local coords
/// - A new ground pivot "O_act" at (-1.0, -1.5) for the actuator base
/// - A linear actuator from ground "O_act" to crank mount "M"
///
/// The actuator triggers compound force expansion (cylinder + rod bodies).
fn build_parallelogram_actuator(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (4.0_f64, 0.0_f64);

    let mut ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    // Actuator base pivot — offset below and behind the crank pivot.
    ground
        .add_attachment_point("O_act", -1.0, -1.5)
        .map_err(|e| e.to_string())?;

    let mut crank = make_bar("crank", "A", "B", 2.0, 0.0, 0.0);
    // Mount point at crank midpoint for the actuator.
    crank
        .add_mount_point("M", 1.0, 0.0)
        .map_err(|e| e.to_string())?;

    let mut coupler = make_bar("coupler", "B", "C", 4.0, 0.0, 0.0);
    coupler.add_coupler_point("P", 2.0, 0.0).unwrap();
    let rocker = make_bar("rocker", "C", "D", 2.0, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();

    // Linear actuator: ground "O_act" → crank mount "M".
    // Uses mount_point_name so compound expansion kicks in on serialization.
    mech.add_force(ForceElement::LinearActuator(LinearActuatorElement {
        body_a: "ground".to_string(),
        point_a: [-1.0, -1.5],
        point_a_name: Some("O_act".to_string()),
        body_b: "crank".to_string(),
        point_b: [1.0, 0.0],
        point_b_name: Some("M".to_string()),
        force: 50.0,
        speed_limit: 0.0,
    }));

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4, 2.0, 4.0, 2.0, 0.0,
        "crank", "coupler", "rocker",
    );

    Ok((mech, q0))
}
```

- [ ] **Step 6: Verify compilation**

Run: `cargo check -p linkage-sim-rs 2>&1 | tail -10`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/gui/samples.rs
git commit -m "feat: add Parallelogram + Actuator sample mechanism"
```

---

### Task 6: Integration test for ParallelogramActuator sample

Verify the sample builds, solves, has the expected bodies/forces, and round-trips through JSON (triggering compound expansion).

**Files:**
- Create: `tests/parallelogram_actuator_sample.rs`

- [ ] **Step 1: Write the test file**

```rust
//! Integration test: ParallelogramActuator sample builds, solves, and
//! round-trips through JSON with compound force expansion.

use linkage_sim_rs::gui::samples::{build_sample, SampleMechanism};
use linkage_sim_rs::forces::elements::ForceElement;
use linkage_sim_rs::io::serialization::{save_mechanism, load_mechanism_unbuilt};

#[test]
fn parallelogram_actuator_builds_and_solves() {
    let (mech, q0) = build_sample(SampleMechanism::ParallelogramActuator);
    assert!(mech.is_built(), "sample should build successfully");
    assert!(q0.len() > 0, "initial state vector should be non-empty");
}

#[test]
fn parallelogram_actuator_has_actuator_force() {
    let (mech, _q0) = build_sample(SampleMechanism::ParallelogramActuator);
    let actuators: Vec<_> = mech.forces().iter().filter(|f| {
        matches!(f, ForceElement::LinearActuator(_))
    }).collect();
    assert_eq!(actuators.len(), 1, "should have exactly one linear actuator");
}

#[test]
fn parallelogram_actuator_has_crank_mount_point() {
    let (mech, _q0) = build_sample(SampleMechanism::ParallelogramActuator);
    let crank = &mech.bodies()["crank"];
    assert!(
        crank.mount_points.contains_key("M"),
        "crank should have mount point 'M' at midpoint"
    );
}

#[test]
fn parallelogram_actuator_json_round_trip_expands_compound() {
    let (mech, _q0) = build_sample(SampleMechanism::ParallelogramActuator);
    let json = save_mechanism(&mech).expect("save should succeed");
    let reloaded = load_mechanism_unbuilt(&json).expect("reload should succeed");

    // Compound expansion should create cylinder + rod bodies.
    assert!(
        reloaded.bodies().contains_key("force_0_cyl"),
        "compound cylinder should be created on reload"
    );
    assert!(
        reloaded.bodies().contains_key("force_0_rod"),
        "compound rod should be created on reload"
    );
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test --test parallelogram_actuator_sample 2>&1 | tail -10`
Expected: all 4 tests pass

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/tests/parallelogram_actuator_sample.rs
git commit -m "test: integration tests for ParallelogramActuator sample"
```

---

### Task 7: Manual smoke test and cleanup

- [ ] **Step 1: Run full test suite**

Run: `cargo test --lib --test compound_force_integration --test mount_point_integration --test golden_fixtures --test singular_behavior --test parallelogram_actuator_sample 2>&1 | tail -10`
Expected: all tests pass

- [ ] **Step 2: Build the GUI and smoke test**

Run: `cargo build -p linkage-gui 2>&1 | tail -5`

Manual verification checklist (for the developer):
1. Launch GUI, load "Parallelogram + Actuator" sample — verify it renders with the actuator visible
2. Click "Linear Spring" in toolbar → cursor should show placement hint
3. Click on a body point → preview line appears
4. Click on a second body point → spring is created between them
5. Escape during placement → cancels cleanly
6. Verify mount points highlight as snap targets

- [ ] **Step 3: Commit any final fixes**

```bash
git add -A
git commit -m "fix: cleanup from smoke testing"
```
