# Compound Force Bodies & Zoom-to-Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-expand mount-point-referenced forces into physical compound bodies (cylinder + rod + revolute joints + prismatic joint), and auto zoom-to-fit on mechanism load.

**Architecture:** Forces with `point_X_name` referencing mount points are expanded at build time in `load_mechanism_unbuilt_from_json`. The expansion creates 2 massless bodies, 2 revolute joints, 1 prismatic joint, and a replacement force — all added to the mechanism before `build()`. The blueprint stores only the original force. Zoom-to-fit is called after load and includes mount points in bounding box.

**Tech Stack:** Rust, nalgebra, serde, egui/eframe

**Spec:** `docs/superpowers/specs/2026-03-21-compound-force-bodies-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `linkage-sim-rs/src/gui/state.rs` | fit_to_view mount points + call on load | Modify |
| `linkage-sim-rs/src/io/serialization.rs` | Compound expansion logic | Modify |
| `linkage-sim-rs/src/forces/compound.rs` | Compound expansion function (new module) | Create |
| `linkage-sim-rs/src/forces/mod.rs` | Export compound module | Modify |
| `linkage-sim-rs/src/gui/canvas.rs` | Compound body rendering tweaks | Modify |

---

### Task 1: Zoom-to-fit improvements

**Files:**
- Modify: `linkage-sim-rs/src/gui/state.rs`

- [ ] **Step 1: Update `fit_to_view` to include mount points in bounding box**

In `fit_to_view()` (line 943-984), after the loop over `body.attachment_points` (line 956), add a loop over `body.mount_points`:

```rust
// After the attachment_points loop, add:
for pt in body.mount_points.values() {
    let global = sim_state.body_point_global(body_id, pt, q);
    if global.x < x_min { x_min = global.x; }
    if global.x > x_max { x_max = global.x; }
    if global.y < y_min { y_min = global.y; }
    if global.y > y_max { y_max = global.y; }
}
```

- [ ] **Step 2: Call `fit_to_view` after `load_sample`**

In `load_sample()` (line 987-1050), after all setup is complete (after `compute_sweep` etc.), add a call to fit_to_view. The canvas dimensions aren't available in `load_sample` directly, so add a flag that triggers fit on next frame:

```rust
// At the end of load_sample(), add:
self.pending_fit_to_view = true;
```

Add the field to `AppState`:
```rust
pub pending_fit_to_view: bool,  // default false
```

In canvas.rs, at the start of `draw_canvas()` (near line 158 where canvas_rect is computed), check and execute:
```rust
if state.pending_fit_to_view {
    state.fit_to_view(canvas_rect.width(), canvas_rect.height());
    state.pending_fit_to_view = false;
}
```

- [ ] **Step 3: Build and verify**

Run: `cd linkage-sim-rs && cargo build 2>&1 | tail -5`
Expected: compiles cleanly.

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/gui/state.rs linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat: zoom-to-fit includes mount points and auto-fits on load"
```

---

### Task 2: Compound expansion function

**Files:**
- Create: `linkage-sim-rs/src/forces/compound.rs`
- Modify: `linkage-sim-rs/src/forces/mod.rs`

- [ ] **Step 1: Write failing test for compound detection**

Create `linkage-sim-rs/src/forces/compound.rs`:

```rust
//! Compound force expansion — auto-creates bodies and joints for
//! mount-point-referenced force elements.

use std::collections::HashMap;
use nalgebra::Vector2;

use crate::core::body::Body;
use crate::forces::elements::ForceElement;
use crate::io::serialization::BodyJson;

/// Result of analyzing a force element for compound expansion.
pub enum CompoundAnalysis {
    /// Force does not reference mount points — add as-is.
    PureForce(ForceElement),
    /// Force references mount points — needs compound expansion.
    NeedsExpansion {
        force: ForceElement,
        mount_a: bool,  // true if point_a references a mount point
        mount_b: bool,  // true if point_b references a mount point
    },
}

/// Check whether a force element references mount points and needs
/// compound expansion.
pub fn analyze_force(
    force: &ForceElement,
    bodies_json: &HashMap<String, BodyJson>,
) -> CompoundAnalysis {
    // ... implementation
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::elements::*;

    fn make_test_bodies() -> HashMap<String, BodyJson> {
        let mut bodies = HashMap::new();
        let mut ground = BodyJson {
            attachment_points: HashMap::from([("O2".to_string(), [0.0, 0.0])]),
            mass: 0.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.0,
            mount_points: HashMap::from([("M1".to_string(), [0.02, 0.01])]),
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
        };
        bodies.insert("ground".to_string(), ground);

        let mut crank = BodyJson {
            attachment_points: HashMap::from([
                ("O2".to_string(), [0.0, 0.0]),
                ("A".to_string(), [0.1, 0.0]),
            ]),
            mass: 1.0,
            cg_local: [0.05, 0.0],
            izz_cg: 0.001,
            mount_points: HashMap::from([("M2".to_string(), [0.05, 0.0])]),
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
        };
        bodies.insert("crank".to_string(), crank);
        bodies
    }

    #[test]
    fn pure_force_no_mount_points() {
        let bodies = make_test_bodies();
        let spring = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "ground".to_string(),
            point_a: [0.0, 0.0],
            point_a_name: None,
            body_b: "crank".to_string(),
            point_b: [0.1, 0.0],
            point_b_name: None,
            stiffness: 500.0,
            free_length: 0.05,
        });
        let result = analyze_force(&spring, &bodies);
        assert!(matches!(result, CompoundAnalysis::PureForce(_)));
    }

    #[test]
    fn force_with_attachment_point_name_stays_pure() {
        let bodies = make_test_bodies();
        let spring = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "ground".to_string(),
            point_a: [0.0, 0.0],
            point_a_name: Some("O2".to_string()), // attachment point, NOT mount
            body_b: "crank".to_string(),
            point_b: [0.1, 0.0],
            point_b_name: Some("A".to_string()),   // attachment point
            stiffness: 500.0,
            free_length: 0.05,
        });
        let result = analyze_force(&spring, &bodies);
        assert!(matches!(result, CompoundAnalysis::PureForce(_)));
    }

    #[test]
    fn force_with_mount_point_needs_expansion() {
        let bodies = make_test_bodies();
        let spring = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "ground".to_string(),
            point_a: [0.02, 0.01],
            point_a_name: Some("M1".to_string()), // mount point!
            body_b: "crank".to_string(),
            point_b: [0.05, 0.0],
            point_b_name: Some("M2".to_string()),  // mount point!
            stiffness: 500.0,
            free_length: 0.05,
        });
        let result = analyze_force(&spring, &bodies);
        assert!(matches!(result, CompoundAnalysis::NeedsExpansion {
            mount_a: true, mount_b: true, ..
        }));
    }

    #[test]
    fn mixed_mount_attachment_detected() {
        let bodies = make_test_bodies();
        let spring = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "ground".to_string(),
            point_a: [0.02, 0.01],
            point_a_name: Some("M1".to_string()), // mount point
            body_b: "crank".to_string(),
            point_b: [0.1, 0.0],
            point_b_name: Some("A".to_string()),   // attachment point
            stiffness: 500.0,
            free_length: 0.05,
        });
        let result = analyze_force(&spring, &bodies);
        assert!(matches!(result, CompoundAnalysis::NeedsExpansion {
            mount_a: true, mount_b: false, ..
        }));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd linkage-sim-rs && cargo test --lib forces::compound::tests -- --nocapture 2>&1 | head -20`
Expected: compile errors or test failures (todo!() panics).

- [ ] **Step 3: Implement `analyze_force`**

```rust
pub fn analyze_force(
    force: &ForceElement,
    bodies_json: &HashMap<String, BodyJson>,
) -> CompoundAnalysis {
    // Extract body IDs and point names from the force
    let (body_a_id, point_a_name, body_b_id, point_b_name) = match force {
        ForceElement::LinearSpring(s) => (&s.body_a, &s.point_a_name, &s.body_b, &s.point_b_name),
        ForceElement::LinearDamper(d) => (&d.body_a, &d.point_a_name, &d.body_b, &d.point_b_name),
        ForceElement::GasSpring(g) => (&g.body_a, &g.point_a_name, &g.body_b, &g.point_b_name),
        ForceElement::LinearActuator(a) => (&a.body_a, &a.point_a_name, &a.body_b, &a.point_b_name),
        // Single-body and rotational forces never expand
        _ => return CompoundAnalysis::PureForce(force.clone()),
    };

    // Check if each named point is a mount point (not attachment point)
    let mount_a = point_a_name.as_ref().map_or(false, |name| {
        bodies_json.get(body_a_id.as_str())
            .map_or(false, |b| b.mount_points.contains_key(name))
    });
    let mount_b = point_b_name.as_ref().map_or(false, |name| {
        bodies_json.get(body_b_id.as_str())
            .map_or(false, |b| b.mount_points.contains_key(name))
    });

    if mount_a || mount_b {
        CompoundAnalysis::NeedsExpansion {
            force: force.clone(),
            mount_a,
            mount_b,
        }
    } else {
        CompoundAnalysis::PureForce(force.clone())
    }
}
```

- [ ] **Step 4: Add module to forces/mod.rs**

In `linkage-sim-rs/src/forces/mod.rs`, add:
```rust
pub mod compound;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd linkage-sim-rs && cargo test --lib forces::compound::tests -- --nocapture`
Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add linkage-sim-rs/src/forces/compound.rs linkage-sim-rs/src/forces/mod.rs
git commit -m "feat: add compound force analysis function"
```

---

### Task 3: Compound expansion — body and joint creation

**Files:**
- Modify: `linkage-sim-rs/src/forces/compound.rs`

- [ ] **Step 1: Write failing test for expansion**

Add to compound.rs tests:

```rust
#[test]
fn expand_creates_correct_bodies_and_joints() {
    use crate::core::mechanism::Mechanism;
    use crate::core::body::{make_ground, make_bar};

    // Build a simple mechanism with a mount-point spring
    let mut mech = Mechanism::new();
    let mut ground = make_ground(&[("O2", 0.0, 0.0)]);
    ground.add_mount_point("M1", 0.05, 0.02).unwrap();
    // Promote mount point to attachment point (as the expansion will do)
    ground.add_attachment_point("_force_0_mount_a", 0.05, 0.02).unwrap();
    mech.add_body(ground).unwrap();

    let mut crank = make_bar("crank", "O2", "A", 0.1, 1.0, 0.001);
    crank.add_mount_point("M2", 0.05, 0.0).unwrap();
    crank.add_attachment_point("_force_0_mount_b", 0.05, 0.0).unwrap();
    mech.add_body(crank).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "O2").unwrap();

    // Create compound bodies
    let cyl = create_compound_cylinder(0, 0.05);
    let rod = create_compound_rod(0, 0.05);
    mech.add_body(cyl).unwrap();
    mech.add_body(rod).unwrap();

    // Verify bodies exist
    assert!(mech.bodies().contains_key("force_0_cyl"));
    assert!(mech.bodies().contains_key("force_0_rod"));
    assert_eq!(mech.bodies()["force_0_cyl"].mass, 0.0);
}
```

- [ ] **Step 2: Implement compound body creation helpers**

```rust
/// Create the cylinder body for a compound force element.
pub fn create_compound_cylinder(force_index: usize, half_len: f64) -> Body {
    let id = format!("force_{}_cyl", force_index);
    let mut body = Body::new(&id);
    body.add_attachment_point("base", 0.0, 0.0).unwrap();
    body.add_attachment_point("slide", half_len, 0.0).unwrap();
    // Mass defaults to 0.0 (massless)
    body
}

/// Create the rod body for a compound force element.
pub fn create_compound_rod(force_index: usize, half_len: f64) -> Body {
    let id = format!("force_{}_rod", force_index);
    let mut body = Body::new(&id);
    body.add_attachment_point("slide", 0.0, 0.0).unwrap();
    body.add_attachment_point("tip", half_len, 0.0).unwrap();
    // Mass defaults to 0.0 (massless)
    body
}
```

- [ ] **Step 3: Implement the full `expand_compound_force` function**

This is the core function that takes an analyzed force and adds bodies + joints + replacement force to a mechanism:

```rust
use crate::core::mechanism::Mechanism;

/// Expand a single compound force into bodies, joints, and replacement force.
///
/// Mutates the mechanism by adding 2 bodies, 2-3 joints, and a replacement
/// force element. The original force is NOT added.
///
/// `force_index` is the force's index in the blueprint force list (for naming).
/// `point_a_pos` and `point_b_pos` are the resolved global positions of the
/// two attachment points at the initial configuration.
pub fn expand_compound_force(
    mech: &mut Mechanism,
    force: &ForceElement,
    force_index: usize,
    mount_a: bool,
    mount_b: bool,
    point_a_pos: [f64; 2],
    point_b_pos: [f64; 2],
) -> Result<ForceElement, Box<dyn std::error::Error>> {
    let dx = point_b_pos[0] - point_a_pos[0];
    let dy = point_b_pos[1] - point_a_pos[1];
    let initial_length = (dx * dx + dy * dy).sqrt();
    let half_len = initial_length / 2.0;

    // Create and add compound bodies
    let cyl = create_compound_cylinder(force_index, half_len);
    let rod = create_compound_rod(force_index, half_len);
    mech.add_body(cyl)?;
    mech.add_body(rod)?;

    let cyl_id = format!("force_{}_cyl", force_index);
    let rod_id = format!("force_{}_rod", force_index);

    // Extract body IDs and point names from the force
    let (body_a_id, point_a_name, body_b_id, point_b_name) = match force {
        ForceElement::LinearSpring(s) => (s.body_a.clone(), s.point_a_name.clone(), s.body_b.clone(), s.point_b_name.clone()),
        ForceElement::LinearDamper(d) => (d.body_a.clone(), d.point_a_name.clone(), d.body_b.clone(), d.point_b_name.clone()),
        ForceElement::GasSpring(g) => (g.body_a.clone(), g.point_a_name.clone(), g.body_b.clone(), g.point_b_name.clone()),
        ForceElement::LinearActuator(a) => (a.body_a.clone(), a.point_a_name.clone(), a.body_b.clone(), a.point_b_name.clone()),
        _ => return Err("Cannot expand non-point force".into()),
    };

    // Create revolute joints at both ends
    // Point A end: connect original body to cylinder base
    let pt_a_ref = if mount_a {
        format!("_force_{}_mount_a", force_index) // promoted synthetic name
    } else {
        point_a_name.unwrap_or_default()
    };
    mech.add_revolute_joint(
        &format!("force_{}_base", force_index),
        &body_a_id, &pt_a_ref,
        &cyl_id, "base",
    )?;

    // Point B end: connect original body to rod tip
    let pt_b_ref = if mount_b {
        format!("_force_{}_mount_b", force_index) // promoted synthetic name
    } else {
        point_b_name.unwrap_or_default()
    };
    mech.add_revolute_joint(
        &format!("force_{}_tip", force_index),
        &body_b_id, &pt_b_ref,
        &rod_id, "tip",
    )?;

    // Prismatic joint between cylinder and rod
    mech.add_prismatic_joint(
        &format!("force_{}_slide", force_index),
        &cyl_id, "slide",
        &rod_id, "slide",
        Vector2::new(1.0, 0.0), // slide axis in cylinder local frame
        0.0,                     // no initial rotation offset
    )?;

    // Create replacement force referencing compound bodies
    let replacement = remap_force_to_compound(force, force_index);
    Ok(replacement)
}

/// Remap a force element to reference the compound bodies' slide points.
fn remap_force_to_compound(force: &ForceElement, idx: usize) -> ForceElement {
    let cyl_id = format!("force_{}_cyl", idx);
    let rod_id = format!("force_{}_rod", idx);
    match force {
        ForceElement::LinearSpring(s) => {
            let mut r = s.clone();
            r.body_a = cyl_id;
            r.point_a = [0.0, 0.0]; // slide point on cylinder
            r.point_a_name = None;
            r.body_b = rod_id;
            r.point_b = [0.0, 0.0]; // slide point on rod
            r.point_b_name = None;
            ForceElement::LinearSpring(r)
        }
        ForceElement::LinearDamper(d) => {
            let mut r = d.clone();
            r.body_a = cyl_id;
            r.point_a = [0.0, 0.0];
            r.point_a_name = None;
            r.body_b = rod_id;
            r.point_b = [0.0, 0.0];
            r.point_b_name = None;
            ForceElement::LinearDamper(r)
        }
        ForceElement::GasSpring(g) => {
            let mut r = g.clone();
            r.body_a = cyl_id;
            r.point_a = [0.0, 0.0];
            r.point_a_name = None;
            r.body_b = rod_id;
            r.point_b = [0.0, 0.0];
            r.point_b_name = None;
            ForceElement::GasSpring(r)
        }
        ForceElement::LinearActuator(a) => {
            let mut r = a.clone();
            r.body_a = cyl_id;
            r.point_a = [0.0, 0.0];
            r.point_a_name = None;
            r.body_b = rod_id;
            r.point_b = [0.0, 0.0];
            r.point_b_name = None;
            ForceElement::LinearActuator(r)
        }
        other => other.clone(), // shouldn't happen
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd linkage-sim-rs && cargo test --lib forces::compound::tests -- --nocapture`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/forces/compound.rs
git commit -m "feat: add compound force expansion with body/joint creation"
```

---

### Task 4: Wire expansion into serialization load path

**Files:**
- Modify: `linkage-sim-rs/src/io/serialization.rs`

- [ ] **Step 1: Add mount point promotion + compound expansion to force loop**

In `load_mechanism_unbuilt_from_json()` (line 407-582), replace the force loop
(lines 571-580) with compound-aware logic:

```rust
// ── Force elements (with compound expansion) ─────────────────────
use crate::forces::compound::{analyze_force, expand_compound_force, CompoundAnalysis};

for (i, force) in json_struct.forces.iter().enumerate() {
    // First resolve named points (existing logic)
    let resolved = force.resolve_named_points(mech.bodies())
        .unwrap_or_else(|e| {
            log::warn!("Failed to resolve force point name: {e}");
            force.clone()
        });

    // Check if force needs compound expansion
    match analyze_force(&resolved, &json_struct.bodies) {
        CompoundAnalysis::PureForce(f) => {
            mech.add_force(f);
        }
        CompoundAnalysis::NeedsExpansion { force: f, mount_a, mount_b } => {
            // Promote mount points to attachment points on original bodies
            // so that add_revolute_joint can find them
            let (body_a_id, pt_a_name, body_b_id, pt_b_name) = match &f {
                ForceElement::LinearSpring(s) => (&s.body_a, &s.point_a_name, &s.body_b, &s.point_b_name),
                ForceElement::LinearDamper(d) => (&d.body_a, &d.point_a_name, &d.body_b, &d.point_b_name),
                ForceElement::GasSpring(g) => (&g.body_a, &g.point_a_name, &g.body_b, &g.point_b_name),
                ForceElement::LinearActuator(a) => (&a.body_a, &a.point_a_name, &a.body_b, &a.point_b_name),
                _ => unreachable!(),
            };

            // Get resolved point positions
            let point_a_pos = get_point_pos(&f, true);
            let point_b_pos = get_point_pos(&f, false);

            // Promote mount points to attachment points with synthetic names
            if mount_a {
                let synthetic = format!("_force_{}_mount_a", i);
                if let Some(body) = mech.bodies_mut().get_mut(body_a_id.as_str()) {
                    let _ = body.add_attachment_point(&synthetic, point_a_pos[0], point_a_pos[1]);
                }
            }
            if mount_b {
                let synthetic = format!("_force_{}_mount_b", i);
                if let Some(body) = mech.bodies_mut().get_mut(body_b_id.as_str()) {
                    let _ = body.add_attachment_point(&synthetic, point_b_pos[0], point_b_pos[1]);
                }
            }

            // Expand compound force
            match expand_compound_force(&mut mech, &f, i, mount_a, mount_b, point_a_pos, point_b_pos) {
                Ok(replacement) => mech.add_force(replacement),
                Err(e) => {
                    log::warn!("Failed to expand compound force {i}: {e}");
                    mech.add_force(f); // fallback: add as pure force
                }
            }
        }
    }
}
```

Note: `get_point_pos` is a helper to extract the cached point_a/point_b coords:

```rust
fn get_point_pos(force: &ForceElement, is_a: bool) -> [f64; 2] {
    match force {
        ForceElement::LinearSpring(s) => if is_a { s.point_a } else { s.point_b },
        ForceElement::LinearDamper(d) => if is_a { d.point_a } else { d.point_b },
        ForceElement::GasSpring(g) => if is_a { g.point_a } else { g.point_b },
        ForceElement::LinearActuator(a) => if is_a { a.point_a } else { a.point_b },
        _ => [0.0, 0.0],
    }
}
```

- [ ] **Step 2: Check if `Mechanism` exposes `bodies_mut()`**

The expansion needs mutable access to existing bodies to promote mount points.
Check if `Mechanism` has a `bodies_mut()` method. If not, add one:

```rust
pub fn bodies_mut(&mut self) -> &mut HashMap<String, Body> {
    &mut self.bodies
}
```

- [ ] **Step 3: Set initial poses for compound bodies after build**

In `state.rs` `rebuild()` method (line 1573-1689), after `mech.build()` succeeds
(line 1591) and before the solver is called, add initial pose setting for
compound bodies. Find compound bodies by checking for `force_` prefix:

```rust
// After mech.build() at line 1591, before solving:
// Set initial poses for compound bodies
let state_ref = mech.state();
for (body_id, _body) in mech.bodies() {
    if body_id.starts_with("force_") && body_id.ends_with("_cyl") {
        // Cylinder body — extract force index and find initial geometry
        // The compound bodies were placed at origin by default;
        // the solver will find valid positions via Newton-Raphson
        // from the joint constraints. If convergence is poor,
        // set explicit poses here using the mount point positions.
    }
}
```

For now, rely on the Newton-Raphson solver to find valid positions from the
joint constraints. The compound bodies start at origin and the solver converges
because the revolute joints constrain them to the correct positions. If
convergence is poor in testing, add explicit pose initialization.

- [ ] **Step 4: Build and run full test suite**

Run: `cd linkage-sim-rs && cargo test 2>&1 | grep "test result"``
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/io/serialization.rs linkage-sim-rs/src/core/mechanism.rs linkage-sim-rs/src/gui/state.rs
git commit -m "feat: wire compound force expansion into build pipeline"
```

---

### Task 5: Integration test — compound force mechanism

**Files:**
- Create: `linkage-sim-rs/tests/compound_force_integration.rs`

- [ ] **Step 1: Write integration test**

```rust
//! Integration test: mount-point-referenced spring expands into compound
//! bodies and joints, and the mechanism builds and solves successfully.

use linkage_sim_rs::io::serialization::load_mechanism_unbuilt;

#[test]
fn compound_spring_mechanism_builds_and_solves() {
    let json_str = r#"{
        "schema_version": "1.0.0",
        "bodies": {
            "ground": {
                "attachment_points": {"O2": [0.0, 0.0], "O4": [0.038, 0.0]},
                "mount_points": {"M1": [0.02, 0.015]},
                "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0
            },
            "crank": {
                "attachment_points": {"O2": [0.0, 0.0], "A": [0.015, 0.0]},
                "mount_points": {"M2": [0.005, 0.0]},
                "mass": 0.5, "cg_local": [0.0075, 0.0], "izz_cg": 0.0001
            },
            "coupler": {
                "attachment_points": {"A": [0.0, 0.0], "B": [0.04, 0.0]},
                "mass": 0.8, "cg_local": [0.02, 0.0], "izz_cg": 0.0002
            },
            "rocker": {
                "attachment_points": {"B": [0.0, 0.0], "O4": [0.0, 0.0]},
                "mass": 0.5, "cg_local": [0.015, 0.0], "izz_cg": 0.0001
            }
        },
        "joints": {
            "J1": {"type": "revolute", "body_i": "ground", "body_j": "crank", "point_i": "O2", "point_j": "O2"},
            "J2": {"type": "revolute", "body_i": "crank", "body_j": "coupler", "point_i": "A", "point_j": "A"},
            "J3": {"type": "revolute", "body_i": "coupler", "body_j": "rocker", "point_i": "B", "point_j": "B"},
            "J4": {"type": "revolute", "body_i": "ground", "body_j": "rocker", "point_i": "O4", "point_j": "O4"}
        },
        "drivers": {
            "D1": {"type": "constant_speed", "body_i": "ground", "body_j": "crank", "omega": 1.0, "theta_0": 0.0}
        },
        "forces": [{
            "type": "LinearSpring",
            "body_a": "ground", "point_a": [0.02, 0.015], "point_a_name": "M1",
            "body_b": "crank", "point_b": [0.005, 0.0], "point_b_name": "M2",
            "stiffness": 500.0, "free_length": 0.02
        }]
    }"#;

    let mut mech = load_mechanism_unbuilt(json_str).unwrap();
    mech.build().unwrap();

    // Verify compound bodies were created
    assert!(mech.bodies().contains_key("force_0_cyl"), "cylinder body missing");
    assert!(mech.bodies().contains_key("force_0_rod"), "rod body missing");

    // Verify compound bodies are massless
    assert_eq!(mech.bodies()["force_0_cyl"].mass, 0.0);
    assert_eq!(mech.bodies()["force_0_rod"].mass, 0.0);

    // Verify total body count: 4 original + ground + 2 compound = 7
    // (ground is special, so bodies map has 7 entries)
    assert_eq!(mech.bodies().len(), 7,
        "Expected 5 original + 2 compound bodies, got {}",
        mech.bodies().len());

    // Verify joint count: 4 original + 2 revolute + 1 prismatic = 7
    assert_eq!(mech.joints().len(), 7,
        "Expected 4 original + 3 compound joints, got {}",
        mech.joints().len());
}

#[test]
fn non_mount_point_force_does_not_expand() {
    let json_str = r#"{
        "schema_version": "1.0.0",
        "bodies": {
            "ground": {
                "attachment_points": {"O2": [0.0, 0.0], "O4": [0.038, 0.0]},
                "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0
            },
            "crank": {
                "attachment_points": {"O2": [0.0, 0.0], "A": [0.015, 0.0]},
                "mass": 0.5, "cg_local": [0.0075, 0.0], "izz_cg": 0.0001
            }
        },
        "joints": {
            "J1": {"type": "revolute", "body_i": "ground", "body_j": "crank", "point_i": "O2", "point_j": "O2"}
        },
        "drivers": {
            "D1": {"type": "constant_speed", "body_i": "ground", "body_j": "crank", "omega": 1.0, "theta_0": 0.0}
        },
        "forces": [{
            "type": "LinearSpring",
            "body_a": "ground", "point_a": [0.0, 0.0],
            "body_b": "crank", "point_b": [0.015, 0.0],
            "stiffness": 500.0, "free_length": 0.01
        }]
    }"#;

    let mut mech = load_mechanism_unbuilt(json_str).unwrap();
    mech.build().unwrap();

    // No compound bodies — only 2 original + ground
    assert!(!mech.bodies().contains_key("force_0_cyl"));
    assert_eq!(mech.bodies().len(), 3); // ground + crank
}
```

- [ ] **Step 2: Run integration tests**

Run: `cd linkage-sim-rs && cargo test --test compound_force_integration -- --nocapture`
Expected: both tests pass.

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/tests/compound_force_integration.rs
git commit -m "test: integration tests for compound force expansion"
```

---

### Task 6: Final verification and cleanup

- [ ] **Step 1: Run full test suite**

Run: `cd linkage-sim-rs && cargo test 2>&1 | grep "test result"`
Expected: all tests pass.

- [ ] **Step 2: Run clippy**

Run: `cd linkage-sim-rs && cargo clippy -- -W clippy::all 2>&1 | grep -E "warning|error" | grep -v "previous errors\|could not compile\|nom v" | head -15`
Check for warnings in our new code.

- [ ] **Step 3: Manual smoke test**

Run: `cd linkage-sim-rs && cargo run`
1. Load FourBar sample — verify it auto-zooms to fit
2. Press F — verify zoom-to-fit works
3. Add mount points to ground and crank
4. Add a spring referencing both mount points
5. Verify compound bodies appear (cylinder + rod with joints)
6. Run animation — verify the compound spring telescopes correctly

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: final cleanup for compound force bodies"
```
