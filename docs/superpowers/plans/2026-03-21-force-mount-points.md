# Force Mount Points Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add named force mount points to bodies so forces can attach at independent pivot locations, with full GUI support for creating, editing, and rendering mount points.

**Architecture:** Adds `mount_points: HashMap<String, Vector2<f64>>` to `Body` (same pattern as `coupler_points`). Force elements gain optional `point_X_name: Option<String>` fields resolved at build time. GUI gets mount point CRUD, point picker dropdowns, and diamond canvas markers.

**Tech Stack:** Rust, nalgebra, serde, egui/eframe

**Spec:** `docs/superpowers/specs/2026-03-21-force-mount-points-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `linkage-sim-rs/src/core/body.rs` | Body struct, mount point methods, error variants | Modify |
| `linkage-sim-rs/src/io/serialization.rs` | BodyJson, body_to_json, load_mechanism_unbuilt_from_json, find_point_name | Modify |
| `linkage-sim-rs/src/forces/elements.rs` | Force element structs with point_X_name fields | Modify |
| `linkage-sim-rs/src/gui/property_panel.rs` | PendingPropertyEdit variants, mount point editor UI, point picker | Modify |
| `linkage-sim-rs/src/gui/state.rs` | AppState methods for mount point CRUD on blueprint | Modify |
| `linkage-sim-rs/src/gui/force_toolbar.rs` | Remove ground filter, second body picker | Modify |
| `linkage-sim-rs/src/gui/canvas.rs` | Diamond markers, mount point hit targets | Modify |

---

### Task 1: Add `mount_points` field and methods to `Body`

**Files:**
- Modify: `linkage-sim-rs/src/core/body.rs`

- [ ] **Step 1: Write failing tests for mount point operations**

Add to the `#[cfg(test)] mod tests` block (after line 230):

```rust
#[test]
fn add_mount_point_works() {
    let mut body = Body::new("test");
    body.add_mount_point("M1", 0.3, 0.1).unwrap();
    assert!(body.mount_points.contains_key("M1"));
    let pt = &body.mount_points["M1"];
    assert_abs_diff_eq!(pt.x, 0.3, epsilon = 1e-15);
    assert_abs_diff_eq!(pt.y, 0.1, epsilon = 1e-15);
}

#[test]
fn duplicate_mount_point_rejected() {
    let mut body = Body::new("test");
    body.add_mount_point("M1", 0.3, 0.1).unwrap();
    assert!(body.add_mount_point("M1", 0.5, 0.2).is_err());
}

#[test]
fn mount_point_name_collision_with_attachment_rejected() {
    let mut body = Body::new("test");
    body.add_attachment_point("A", 1.0, 2.0).unwrap();
    assert!(body.add_mount_point("A", 0.5, 0.2).is_err());
}

#[test]
fn resolve_force_point_finds_attachment() {
    let mut body = Body::new("test");
    body.add_attachment_point("A", 1.0, 2.0).unwrap();
    let pt = body.resolve_force_point("A").unwrap();
    assert_abs_diff_eq!(pt.x, 1.0, epsilon = 1e-15);
    assert_abs_diff_eq!(pt.y, 2.0, epsilon = 1e-15);
}

#[test]
fn resolve_force_point_finds_mount() {
    let mut body = Body::new("test");
    body.add_mount_point("M1", 0.3, 0.1).unwrap();
    let pt = body.resolve_force_point("M1").unwrap();
    assert_abs_diff_eq!(pt.x, 0.3, epsilon = 1e-15);
    assert_abs_diff_eq!(pt.y, 0.1, epsilon = 1e-15);
}

#[test]
fn resolve_force_point_errors_on_missing() {
    let body = Body::new("test");
    assert!(body.resolve_force_point("nope").is_err());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd linkage-sim-rs && cargo test --lib core::body::tests -- --nocapture 2>&1 | head -40`
Expected: compile errors — `mount_points` field and methods don't exist yet.

- [ ] **Step 3: Add `mount_points` field to `Body` struct and update constructors**

In `body.rs`, add field after `izz_cg` (line 29), before `coupler_points`:

```rust
    /// Named points for force element attachment (not structural joints).
    pub mount_points: HashMap<String, Vector2<f64>>,
```

Update `Body::new()` (line 38) to include `mount_points: HashMap::new()`.

Update `make_ground()` (line 131) to include `mount_points: HashMap::new()`.

Update `make_bar()` (line 155) to include `mount_points: HashMap::new()`.

- [ ] **Step 4: Add error variants to `BodyError`**

After `DuplicateCouplerPoint` (line 176), add:

```rust
    #[error("Mount point '{point}' already exists on body '{body}'")]
    DuplicateMountPoint { point: String, body: String },
    #[error("Mount point name '{point}' collides with attachment point on body '{body}'")]
    MountPointNameCollision { point: String, body: String },
    #[error(
        "Force point '{point}' not found on body '{body}'. \
         Attachment points: {available_attachment:?}, Mount points: {available_mount:?}"
    )]
    ForcePointNotFound {
        point: String,
        body: String,
        available_attachment: Vec<String>,
        available_mount: Vec<String>,
    },
```

- [ ] **Step 5: Implement `add_mount_point()` and `resolve_force_point()`**

After `add_coupler_point()` (line 118), add:

```rust
    /// Add a named mount point for force element attachment.
    ///
    /// Rejects names that collide with existing attachment_points or mount_points.
    pub fn add_mount_point(
        &mut self,
        name: &str,
        x: f64,
        y: f64,
    ) -> Result<(), BodyError> {
        if self.attachment_points.contains_key(name) {
            return Err(BodyError::MountPointNameCollision {
                point: name.to_string(),
                body: self.id.clone(),
            });
        }
        if self.mount_points.contains_key(name) {
            return Err(BodyError::DuplicateMountPoint {
                point: name.to_string(),
                body: self.id.clone(),
            });
        }
        self.mount_points
            .insert(name.to_string(), Vector2::new(x, y));
        Ok(())
    }

    /// Look up a named point from attachment_points or mount_points.
    ///
    /// Name uniqueness is enforced across both collections, so a name
    /// can only exist in one. Returns the match or an error listing
    /// available names from both collections.
    pub fn resolve_force_point(&self, name: &str) -> Result<&Vector2<f64>, BodyError> {
        self.attachment_points
            .get(name)
            .or_else(|| self.mount_points.get(name))
            .ok_or_else(|| BodyError::ForcePointNotFound {
                point: name.to_string(),
                body: self.id.clone(),
                available_attachment: self.attachment_points.keys().cloned().collect(),
                available_mount: self.mount_points.keys().cloned().collect(),
            })
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd linkage-sim-rs && cargo test --lib core::body::tests -- --nocapture`
Expected: all tests pass, including the 6 new mount point tests.

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/core/body.rs
git commit -m "feat: add mount_points field and methods to Body"
```

---

### Task 2: Add `mount_points` to serialization (BodyJson, body_to_json, load)

**Files:**
- Modify: `linkage-sim-rs/src/io/serialization.rs`

- [ ] **Step 1: Write failing round-trip test**

Add to the serialization tests section (after `fourbar_round_trip_preserves_coupler_points` ~line 708):

```rust
#[test]
fn round_trip_preserves_mount_points() {
    let json_str = r#"{
        "schema_version": "1.0.0",
        "bodies": {
            "ground": {
                "attachment_points": {"O2": [0.0, 0.0], "O4": [0.038, 0.0]},
                "mount_points": {"M1": [0.02, 0.01]},
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
        }
    }"#;
    let loaded = load_mechanism_unbuilt(json_str).unwrap();
    let ground = loaded.bodies().get("ground").unwrap();
    assert!(ground.mount_points.contains_key("M1"));
    assert_abs_diff_eq!(ground.mount_points["M1"].x, 0.02, epsilon = 1e-15);
    assert_abs_diff_eq!(ground.mount_points["M1"].y, 0.01, epsilon = 1e-15);

    // Round-trip: build first (mechanism_to_json requires built mechanism),
    // then save and reload
    let built = loaded.build().unwrap();
    let saved = mechanism_to_json(&built).unwrap();
    let reloaded = load_mechanism_unbuilt(&saved).unwrap();
    let ground2 = reloaded.bodies().get("ground").unwrap();
    assert!(ground2.mount_points.contains_key("M1"));
    assert_abs_diff_eq!(ground2.mount_points["M1"].x, 0.02, epsilon = 1e-15);
}

#[test]
fn old_json_without_mount_points_loads_with_empty() {
    let json_str = r#"{
        "schema_version": "1.0.0",
        "bodies": {
            "ground": {
                "attachment_points": {"O2": [0.0, 0.0]},
                "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0
            }
        },
        "joints": {}
    }"#;
    let loaded = load_mechanism_unbuilt(json_str).unwrap();
    let ground = loaded.bodies().get("ground").unwrap();
    assert!(ground.mount_points.is_empty());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd linkage-sim-rs && cargo test --lib io::serialization::tests -- --nocapture 2>&1 | head -40`
Expected: compile errors or test failures.

- [ ] **Step 3: Add `mount_points` to `BodyJson`**

In `serialization.rs`, add after `izz_cg` (line 106), before `coupler_points`:

```rust
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub mount_points: HashMap<String, [f64; 2]>,
```

- [ ] **Step 4: Update `body_to_json()`**

In `body_to_json()` (~line 219-243), add `mount_points` mapping alongside the existing `coupler_points` mapping. Follow the exact same pattern used for `coupler_points`:

```rust
mount_points: body.mount_points.iter().map(|(k, v)| {
    (k.clone(), [v.x, v.y])
}).collect(),
```

- [ ] **Step 5: Update `load_mechanism_unbuilt_from_json()` Body struct literal**

In the Body construction (~line 419-434), add mount_points deserialization. Follow the `coupler_points` pattern:

```rust
let mount_points: HashMap<String, Vector2<f64>> = body_json
    .mount_points
    .iter()
    .map(|(k, v)| (k.clone(), Vector2::new(v[0], v[1])))
    .collect();
```

Add `mount_points,` to the Body struct literal.

- [ ] **Step 6: Update `find_point_name()` to search mount_points**

In `find_point_name()` (~line 200-216), after searching `attachment_points`, add a search of `mount_points`:

```rust
// After the attachment_points search loop, add:
for (name, pt) in &body.mount_points {
    if (pt.x - coords.x).abs() < TOL && (pt.y - coords.y).abs() < TOL {
        return Ok(name.clone());
    }
}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd linkage-sim-rs && cargo test --lib io::serialization::tests -- --nocapture`
Expected: all tests pass including the 2 new round-trip tests.

- [ ] **Step 8: Run full test suite to check nothing broke**

Run: `cd linkage-sim-rs && cargo test 2>&1 | tail -20`
Expected: all existing tests still pass.

- [ ] **Step 9: Commit**

```bash
git add linkage-sim-rs/src/io/serialization.rs
git commit -m "feat: add mount_points to BodyJson serialization"
```

---

### Task 3: Add `point_X_name` fields to force element structs

**Files:**
- Modify: `linkage-sim-rs/src/forces/elements.rs`

- [ ] **Step 1: Write failing serde round-trip test**

Add to the tests in `elements.rs` (or create a new test module if none exists):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_spring_with_point_names_round_trips() {
        let spring = LinearSpringElement {
            body_a: "ground".to_string(),
            point_a: [0.02, 0.01],
            point_a_name: Some("M1".to_string()),
            body_b: "crank".to_string(),
            point_b: [0.0, 0.0],
            point_b_name: Some("A".to_string()),
            stiffness: 500.0,
            free_length: 0.05,
        };
        let fe = ForceElement::LinearSpring(spring);
        let json = serde_json::to_string(&fe).unwrap();
        let loaded: ForceElement = serde_json::from_str(&json).unwrap();
        if let ForceElement::LinearSpring(s) = loaded {
            assert_eq!(s.point_a_name, Some("M1".to_string()));
            assert_eq!(s.point_b_name, Some("A".to_string()));
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn linear_spring_without_point_names_defaults_to_none() {
        let json = r#"{
            "type": "LinearSpring",
            "body_a": "ground", "point_a": [0.0, 0.0],
            "body_b": "crank", "point_b": [0.1, 0.0],
            "stiffness": 500.0, "free_length": 0.05
        }"#;
        let loaded: ForceElement = serde_json::from_str(json).unwrap();
        if let ForceElement::LinearSpring(s) = loaded {
            assert_eq!(s.point_a_name, None);
            assert_eq!(s.point_b_name, None);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn external_force_with_local_point_name_round_trips() {
        let ef = ExternalForceElement {
            body_id: "crank".to_string(),
            local_point: [0.01, 0.0],
            local_point_name: Some("A".to_string()),
            force: [10.0, -5.0],
            modulation: TimeModulation::Constant,
        };
        let fe = ForceElement::ExternalForce(ef);
        let json = serde_json::to_string(&fe).unwrap();
        let loaded: ForceElement = serde_json::from_str(&json).unwrap();
        if let ForceElement::ExternalForce(e) = loaded {
            assert_eq!(e.local_point_name, Some("A".to_string()));
        } else {
            panic!("wrong variant");
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd linkage-sim-rs && cargo test --lib forces::elements::tests -- --nocapture 2>&1 | head -20`
Expected: compile errors — fields don't exist yet.

- [ ] **Step 3: Add `point_a_name` and `point_b_name` to four two-body elements**

For each of `LinearSpringElement` (line 184), `LinearDamperElement` (line 218), `GasSpringElement` (line 279), `LinearActuatorElement` (line 378), add after the `point_a` field:

```rust
    /// Named reference to a point on body A (attachment_points or mount_points).
    /// If Some, resolved at build time and cached into point_a.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_a_name: Option<String>,
```

And after the `point_b` field:

```rust
    /// Named reference to a point on body B (attachment_points or mount_points).
    /// If Some, resolved at build time and cached into point_b.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_b_name: Option<String>,
```

- [ ] **Step 4: Add `local_point_name` to `ExternalForceElement`**

After the `local_point` field (line 252), add:

```rust
    /// Named reference to a point on the body (attachment_points or mount_points).
    /// If Some, resolved at build time and cached into local_point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_point_name: Option<String>,
```

- [ ] **Step 5: Fix all compilation errors from new fields**

Search the codebase for all struct literals constructing these 5 types and add the new fields with `None` / appropriate defaults. Key locations:

- `gui/force_toolbar.rs` — all `PendingForceAdd::Add(ForceElement::LinearSpring(...))` etc. (lines 109-140). Add `point_a_name: None, point_b_name: None` to each.
- `gui/property_panel.rs` — all `make_element` closures in `draw_force_element_details()`. Each closure that clones and rebuilds a force element will need to preserve the `point_X_name` fields.
- Any test files constructing force elements.

Run: `cd linkage-sim-rs && cargo build 2>&1 | head -40` to find all remaining sites.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd linkage-sim-rs && cargo test -- --nocapture 2>&1 | tail -20`
Expected: all tests pass including the 3 new serde tests.

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/forces/elements.rs linkage-sim-rs/src/gui/force_toolbar.rs linkage-sim-rs/src/gui/property_panel.rs
git commit -m "feat: add point_X_name fields to force element structs"
```

---

### Task 4: Force point resolution at build time

**Files:**
- Modify: `linkage-sim-rs/src/forces/elements.rs` (add resolve method)
- Modify: `linkage-sim-rs/src/gui/state.rs` (call resolution on rebuild)

- [ ] **Step 1: Write failing test for resolution**

Add to `forces/elements.rs` tests:

```rust
#[test]
fn resolve_named_points_caches_coordinates() {
    use crate::core::body::Body;

    let mut ground = Body::new("ground");
    ground.add_mount_point("M1", 0.02, 0.01).unwrap();

    let mut crank = Body::new("crank");
    crank.add_attachment_point("A", 0.015, 0.0).unwrap();

    let mut bodies = HashMap::new();
    bodies.insert("ground".to_string(), ground);
    bodies.insert("crank".to_string(), crank);

    let mut spring = LinearSpringElement {
        body_a: "ground".to_string(),
        point_a: [0.0, 0.0],
        point_a_name: Some("M1".to_string()),
        body_b: "crank".to_string(),
        point_b: [0.0, 0.0],
        point_b_name: Some("A".to_string()),
        stiffness: 500.0,
        free_length: 0.05,
    };

    let fe = ForceElement::LinearSpring(spring);
    let resolved = fe.resolve_named_points(&bodies).unwrap();

    if let ForceElement::LinearSpring(s) = &resolved {
        assert_abs_diff_eq!(s.point_a[0], 0.02, epsilon = 1e-15);
        assert_abs_diff_eq!(s.point_a[1], 0.01, epsilon = 1e-15);
        assert_abs_diff_eq!(s.point_b[0], 0.015, epsilon = 1e-15);
        assert_abs_diff_eq!(s.point_b[1], 0.0, epsilon = 1e-15);
    } else {
        panic!("wrong variant");
    }
}

#[test]
fn resolve_named_points_none_preserves_raw_coords() {
    let bodies = HashMap::new();
    let spring = LinearSpringElement {
        body_a: "ground".to_string(),
        point_a: [0.05, 0.03],
        point_a_name: None,
        body_b: "crank".to_string(),
        point_b: [0.01, 0.0],
        point_b_name: None,
        stiffness: 500.0,
        free_length: 0.05,
    };
    let fe = ForceElement::LinearSpring(spring);
    let resolved = fe.resolve_named_points(&bodies).unwrap();
    if let ForceElement::LinearSpring(s) = &resolved {
        assert_abs_diff_eq!(s.point_a[0], 0.05, epsilon = 1e-15);
        assert_abs_diff_eq!(s.point_b[0], 0.01, epsilon = 1e-15);
    } else {
        panic!("wrong variant");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd linkage-sim-rs && cargo test --lib forces::elements::tests -- --nocapture 2>&1 | head -20`
Expected: compile error — `resolve_named_points` doesn't exist.

- [ ] **Step 3: Implement `resolve_named_points` on `ForceElement`**

Add to the `impl ForceElement` block:

```rust
    /// Resolve named point references to cached local coordinates.
    ///
    /// For each force element with point_X_name = Some(name), looks up the
    /// named point on the referenced body and writes the coordinates into
    /// point_X. Returns a clone with resolved coordinates.
    ///
    /// Elements without named points or without point fields are returned unchanged.
    pub fn resolve_named_points(
        &self,
        bodies: &HashMap<String, Body>,
    ) -> Result<ForceElement, crate::core::body::BodyError> {
        let mut resolved = self.clone();
        match &mut resolved {
            ForceElement::LinearSpring(s) => {
                if let Some(ref name) = s.point_a_name {
                    let pt = bodies[&s.body_a].resolve_force_point(name)?;
                    s.point_a = [pt.x, pt.y];
                }
                if let Some(ref name) = s.point_b_name {
                    let pt = bodies[&s.body_b].resolve_force_point(name)?;
                    s.point_b = [pt.x, pt.y];
                }
            }
            ForceElement::LinearDamper(d) => {
                if let Some(ref name) = d.point_a_name {
                    let pt = bodies[&d.body_a].resolve_force_point(name)?;
                    d.point_a = [pt.x, pt.y];
                }
                if let Some(ref name) = d.point_b_name {
                    let pt = bodies[&d.body_b].resolve_force_point(name)?;
                    d.point_b = [pt.x, pt.y];
                }
            }
            ForceElement::GasSpring(g) => {
                if let Some(ref name) = g.point_a_name {
                    let pt = bodies[&g.body_a].resolve_force_point(name)?;
                    g.point_a = [pt.x, pt.y];
                }
                if let Some(ref name) = g.point_b_name {
                    let pt = bodies[&g.body_b].resolve_force_point(name)?;
                    g.point_b = [pt.x, pt.y];
                }
            }
            ForceElement::LinearActuator(a) => {
                if let Some(ref name) = a.point_a_name {
                    let pt = bodies[&a.body_a].resolve_force_point(name)?;
                    a.point_a = [pt.x, pt.y];
                }
                if let Some(ref name) = a.point_b_name {
                    let pt = bodies[&a.body_b].resolve_force_point(name)?;
                    a.point_b = [pt.x, pt.y];
                }
            }
            ForceElement::ExternalForce(e) => {
                if let Some(ref name) = e.local_point_name {
                    let pt = bodies[&e.body_id].resolve_force_point(name)?;
                    e.local_point = [pt.x, pt.y];
                }
            }
            // Rotational elements and gravity have no point fields
            _ => {}
        }
        Ok(resolved)
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd linkage-sim-rs && cargo test --lib forces::elements::tests -- --nocapture`
Expected: all tests pass.

- [ ] **Step 5: Wire resolution into `load_mechanism_unbuilt_from_json`**

In `serialization.rs`, in `load_mechanism_unbuilt_from_json()`, after all bodies
are loaded but before forces are added to the mechanism (~after line 450 where
forces are iterated), resolve named points:

```rust
// After bodies are added to the mechanism, resolve force point names:
let bodies = mech.bodies().clone(); // snapshot for resolution
let resolved_forces: Vec<ForceElement> = forces_json
    .iter()
    .map(|f| f.resolve_named_points(&bodies).unwrap_or_else(|e| {
        log::warn!("Failed to resolve force point name: {e}");
        f.clone() // fallback to unresolved
    }))
    .collect();
for force in resolved_forces {
    mech.add_force(force);
}
```

Replace the existing force-add loop (which currently does
`for force in &json.forces { mech.add_force(force.clone()); }`) with this
resolved version. This ensures that at build time, all `point_X_name` references
are resolved to cached coordinates in `point_a`/`point_b`.

**Without this step, named points are silently ignored at runtime** — forces
with `point_a_name = Some("M1")` would use the unresolved `point_a = [0.0, 0.0]`
default.

- [ ] **Step 6: Run full test suite to verify**

Run: `cd linkage-sim-rs && cargo test -- --nocapture 2>&1 | tail -20`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/forces/elements.rs linkage-sim-rs/src/io/serialization.rs
git commit -m "feat: add resolve_named_points and wire into build pipeline"
```

---

### Task 5: Mount point CRUD on AppState (blueprint mutations)

**Files:**
- Modify: `linkage-sim-rs/src/gui/state.rs`
- Modify: `linkage-sim-rs/src/gui/property_panel.rs` (PendingPropertyEdit variants)

- [ ] **Step 1: Add mount point CRUD methods to AppState**

In `state.rs`, near the existing `set_body_mass` / `set_body_izz` methods (~line 1784), add:

Note: All blueprint mutations must scope the `&mut self.blueprint` borrow
inside a block so it's dropped before `self.rebuild()` is called. This avoids
the double-mutable-borrow conflict (`&mut self.blueprint` vs `&mut self` for
rebuild). Follow the pattern used by existing methods like `set_body_mass`.

```rust
    /// Add a mount point to a body in the blueprint.
    pub fn add_mount_point(&mut self, body_id: &str, name: &str, pos: [f64; 2]) {
        self.push_undo();
        {
            let Some(bp) = &mut self.blueprint else { return };
            if let Some(body) = bp.bodies.get_mut(body_id) {
                body.mount_points.entry(name.to_string()).or_insert(pos);
            }
        }
        self.rebuild();
    }

    /// Delete a mount point from a body in the blueprint.
    /// Clears point_X_name on any force that references this mount point.
    /// Returns the count of forces that had references cleared.
    pub fn delete_mount_point(&mut self, body_id: &str, name: &str) -> usize {
        self.push_undo();
        let cleared_count;
        {
            let Some(bp) = &mut self.blueprint else { return 0 };
            if let Some(body) = bp.bodies.get_mut(body_id) {
                body.mount_points.remove(name);
            }
            cleared_count = Self::clear_force_point_refs(&mut bp.forces, body_id, name);
        }
        self.rebuild();
        cleared_count
    }

    /// Rename a mount point on a body in the blueprint.
    /// Updates point_X_name on any force that references the old name.
    pub fn rename_mount_point(&mut self, body_id: &str, old_name: &str, new_name: &str) {
        self.push_undo();
        {
            let Some(bp) = &mut self.blueprint else { return };
            if let Some(body) = bp.bodies.get_mut(body_id) {
                if let Some(pos) = body.mount_points.remove(old_name) {
                    body.mount_points.insert(new_name.to_string(), pos);
                }
            }
            Self::rename_force_point_refs(&mut bp.forces, body_id, old_name, new_name);
        }
        self.rebuild();
    }

    /// Update mount point position on a body in the blueprint.
    pub fn update_mount_point_position(&mut self, body_id: &str, name: &str, pos: [f64; 2]) {
        {
            let Some(bp) = &mut self.blueprint else { return };
            if let Some(body) = bp.bodies.get_mut(body_id) {
                if let Some(pt) = body.mount_points.get_mut(name) {
                    *pt = pos;
                }
            }
        }
        self.rebuild();
    }
```

- [ ] **Step 2: Add cascade helper methods**

Add private helpers to `impl AppState`:

```rust
    /// Clear point_X_name on forces that reference a deleted point.
    /// Returns the number of forces that had references cleared.
    fn clear_force_point_refs(forces: &mut [ForceElement], body_id: &str, point_name: &str) -> usize {
        let mut count = 0usize;
        for force in forces.iter_mut() {
            match force {
                ForceElement::LinearSpring(s) => {
                    if s.body_a == body_id && s.point_a_name.as_deref() == Some(point_name) {
                        s.point_a_name = None; count += 1;
                    }
                    if s.body_b == body_id && s.point_b_name.as_deref() == Some(point_name) {
                        s.point_b_name = None; count += 1;
                    }
                }
                ForceElement::LinearDamper(d) => {
                    if d.body_a == body_id && d.point_a_name.as_deref() == Some(point_name) {
                        d.point_a_name = None; count += 1;
                    }
                    if d.body_b == body_id && d.point_b_name.as_deref() == Some(point_name) {
                        d.point_b_name = None; count += 1;
                    }
                }
                ForceElement::GasSpring(g) => {
                    if g.body_a == body_id && g.point_a_name.as_deref() == Some(point_name) {
                        g.point_a_name = None; count += 1;
                    }
                    if g.body_b == body_id && g.point_b_name.as_deref() == Some(point_name) {
                        g.point_b_name = None; count += 1;
                    }
                }
                ForceElement::LinearActuator(a) => {
                    if a.body_a == body_id && a.point_a_name.as_deref() == Some(point_name) {
                        a.point_a_name = None; count += 1;
                    }
                    if a.body_b == body_id && a.point_b_name.as_deref() == Some(point_name) {
                        a.point_b_name = None; count += 1;
                    }
                }
                ForceElement::ExternalForce(e) => {
                    if e.body_id == body_id && e.local_point_name.as_deref() == Some(point_name) {
                        e.local_point_name = None; count += 1;
                    }
                }
                _ => {}
            }
        }
        count
    }

    /// Rename point_X_name on forces that reference a renamed point.
    fn rename_force_point_refs(
        forces: &mut [ForceElement],
        body_id: &str,
        old_name: &str,
        new_name: &str,
    ) {
        for force in forces.iter_mut() {
            match force {
                ForceElement::LinearSpring(s) => {
                    if s.body_a == body_id && s.point_a_name.as_deref() == Some(old_name) {
                        s.point_a_name = Some(new_name.to_string());
                    }
                    if s.body_b == body_id && s.point_b_name.as_deref() == Some(old_name) {
                        s.point_b_name = Some(new_name.to_string());
                    }
                }
                ForceElement::LinearDamper(d) => {
                    if d.body_a == body_id && d.point_a_name.as_deref() == Some(old_name) {
                        d.point_a_name = Some(new_name.to_string());
                    }
                    if d.body_b == body_id && d.point_b_name.as_deref() == Some(old_name) {
                        d.point_b_name = Some(new_name.to_string());
                    }
                }
                ForceElement::GasSpring(g) => {
                    if g.body_a == body_id && g.point_a_name.as_deref() == Some(old_name) {
                        g.point_a_name = Some(new_name.to_string());
                    }
                    if g.body_b == body_id && g.point_b_name.as_deref() == Some(old_name) {
                        g.point_b_name = Some(new_name.to_string());
                    }
                }
                ForceElement::LinearActuator(a) => {
                    if a.body_a == body_id && a.point_a_name.as_deref() == Some(old_name) {
                        a.point_a_name = Some(new_name.to_string());
                    }
                    if a.body_b == body_id && a.point_b_name.as_deref() == Some(old_name) {
                        a.point_b_name = Some(new_name.to_string());
                    }
                }
                ForceElement::ExternalForce(e) => {
                    if e.body_id == body_id && e.local_point_name.as_deref() == Some(old_name) {
                        e.local_point_name = Some(new_name.to_string());
                    }
                }
                _ => {}
            }
        }
    }
```

- [ ] **Step 3: Add `PendingPropertyEdit` variants**

In `property_panel.rs` (line 18-25), add to the enum:

```rust
    AddMountPoint { body_id: String, name: String, position: [f64; 2] },
    DeleteMountPoint { body_id: String, name: String },
    RenameMountPoint { body_id: String, old_name: String, new_name: String },
    UpdateMountPointPosition { body_id: String, name: String, position: [f64; 2] },
```

- [ ] **Step 4: Handle new variants in `apply_pending()`**

In `apply_pending()` (line 241-264), add match arms:

```rust
            PendingPropertyEdit::AddMountPoint { body_id, name, position } => {
                state.add_mount_point(&body_id, &name, position);
            }
            PendingPropertyEdit::DeleteMountPoint { body_id, name } => {
                let cleared = state.delete_mount_point(&body_id, &name);
                if cleared > 0 {
                    log::warn!(
                        "Mount point '{}' removed — {} force ref(s) reverted to fixed coordinates",
                        name, cleared
                    );
                    // If egui toasts are available, show a warning toast here.
                    // Otherwise the log::warn is sufficient for user feedback.
                }
            }
            PendingPropertyEdit::RenameMountPoint { body_id, old_name, new_name } => {
                state.rename_mount_point(&body_id, &old_name, &new_name);
            }
            PendingPropertyEdit::UpdateMountPointPosition { body_id, name, position } => {
                state.update_mount_point_position(&body_id, &name, position);
            }
```

- [ ] **Step 5: Build and verify compilation**

Run: `cd linkage-sim-rs && cargo build 2>&1 | tail -10`
Expected: compiles cleanly.

- [ ] **Step 6: Run full test suite**

Run: `cd linkage-sim-rs && cargo test 2>&1 | tail -20`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/gui/state.rs linkage-sim-rs/src/gui/property_panel.rs
git commit -m "feat: add mount point CRUD methods to AppState"
```

---

### Task 6: Mount point editor UI in property panel

**Files:**
- Modify: `linkage-sim-rs/src/gui/property_panel.rs`

- [ ] **Step 1: Add mount point editor section**

In `draw_property_panel()`, after the attachment points / link length editing section and before force elements, add a "Mount Points" collapsible section. Follow the pattern of the existing body property editing (~lines 89-190).

The section should:
- Only show when a non-ground body is selected and blueprint exists
- Use `egui::CollapsingHeader::new("Mount Points")`
- List existing mount points from `blueprint.bodies[body_id].mount_points`
- Each row: editable name text field, x/y DragValue sliders, delete button
- "Add Mount Point" button at bottom

```rust
// Inside the body editing section, after mass/Izz/link-length editors:
if let Some(bp) = &state.blueprint {
    if let Some(body_json) = bp.bodies.get(body_id) {
        egui::CollapsingHeader::new("Mount Points")
            .default_open(true)
            .show(ui, |ui| {
                let mut sorted_names: Vec<&String> = body_json.mount_points.keys().collect();
                sorted_names.sort();
                for name in &sorted_names {
                    let pos = body_json.mount_points[*name];
                    ui.horizontal(|ui| {
                        ui.label(format!("{}", name));
                        let mut x = pos[0];
                        let mut y = pos[1];
                        if ui.add(egui::DragValue::new(&mut x).speed(0.001).prefix("x: ").suffix(" m")).changed() {
                            pending = Some(PendingPropertyEdit::UpdateMountPointPosition {
                                body_id: body_id.clone(),
                                name: name.to_string(),
                                position: [x, pos[1]],
                            });
                        }
                        if ui.add(egui::DragValue::new(&mut y).speed(0.001).prefix("y: ").suffix(" m")).changed() {
                            pending = Some(PendingPropertyEdit::UpdateMountPointPosition {
                                body_id: body_id.clone(),
                                name: name.to_string(),
                                position: [pos[0], y],
                            });
                        }
                        if ui.small_button("x").clicked() {
                            pending = Some(PendingPropertyEdit::DeleteMountPoint {
                                body_id: body_id.clone(),
                                name: name.to_string(),
                            });
                        }
                    });
                }
                if ui.button("+ Add Mount Point").clicked() {
                    // Find next unused M<N> name (handles gaps from deletion)
                    let mut next_num = 1u32;
                    while body_json.mount_points.contains_key(&format!("M{}", next_num))
                        || body_json.attachment_points.contains_key(&format!("M{}", next_num))
                    {
                        next_num += 1;
                    }
                    let new_name = format!("M{}", next_num);
                    pending = Some(PendingPropertyEdit::AddMountPoint {
                        body_id: body_id.clone(),
                        name: new_name,
                        position: [body_json.cg_local[0], body_json.cg_local[1]],
                    });
                }
            });
    }
}
```

- [ ] **Step 2: Build and verify**

Run: `cd linkage-sim-rs && cargo build 2>&1 | tail -10`
Expected: compiles cleanly.

- [ ] **Step 3: Manual smoke test**

Run: `cd linkage-sim-rs && cargo run`
- Load a sample mechanism
- Select a body
- Verify "Mount Points" section appears
- Click "Add Mount Point" — verify a point named M1 appears
- Edit x/y values — verify they update
- Click delete — verify point is removed

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/gui/property_panel.rs
git commit -m "feat: add mount point editor UI in property panel"
```

---

### Task 7: Point picker dropdown for force elements

**Files:**
- Modify: `linkage-sim-rs/src/gui/property_panel.rs`

- [ ] **Step 1: Create `draw_point_picker` helper function**

Add a new function that replaces raw x/y DragValues with a dropdown of named points:

```rust
/// Draw a point picker: dropdown of named points + custom coords fallback.
fn draw_point_picker(
    ui: &mut egui::Ui,
    label: &str,
    body_id: &str,
    current_point: &[f64; 2],
    current_name: &Option<String>,
    blueprint: &MechanismJson,
    index: usize,
    make_element: impl Fn(Option<String>, [f64; 2]) -> ForceElement,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let body_json = blueprint.bodies.get(body_id);

    // Collect available points: attachment (joint) + mount
    let mut options: Vec<(String, String, [f64; 2])> = Vec::new(); // (name, label, coords)
    if let Some(bj) = body_json {
        let mut att_names: Vec<&String> = bj.attachment_points.keys().collect();
        att_names.sort();
        for name in att_names {
            options.push((name.clone(), format!("{} (joint)", name), bj.attachment_points[name]));
        }
        let mut mt_names: Vec<&String> = bj.mount_points.keys().collect();
        mt_names.sort();
        for name in mt_names {
            options.push((name.clone(), format!("{} (mount)", name), bj.mount_points[name]));
        }
    }

    ui.horizontal(|ui| {
        ui.label(format!("{}:", label));
        let current_label = current_name
            .as_ref()
            .map(|n| n.as_str())
            .unwrap_or("custom");
        egui::ComboBox::from_id_salt(format!("{}-{}-{}", label, body_id, index))
            .selected_text(current_label)
            .show_ui(ui, |ui| {
                for (name, display, coords) in &options {
                    if ui.selectable_label(
                        current_name.as_ref() == Some(name),
                        display,
                    ).clicked() {
                        *pending = Some(PendingPropertyEdit::UpdateForce {
                            index,
                            force: make_element(Some(name.clone()), *coords),
                        });
                    }
                }
                // Custom coords option
                if ui.selectable_label(current_name.is_none(), "custom coords...").clicked() {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(None, *current_point),
                    });
                }
            });
    });

    // Show raw coords (read-only if named, editable if custom)
    if current_name.is_none() {
        draw_point_fields(ui, label, current_point, index, |pt| {
            make_element(None, pt)
        }, pending);
    }
}
```

- [ ] **Step 2: Update `draw_force_element_details` to use point picker for LinearSpring**

In the `ForceElement::LinearSpring` arm (~line 617), replace the `draw_point_fields` calls with `draw_point_picker` calls.

**Critical:** The `make_element` closure must set BOTH the name AND the coords.
Example for LinearSpring Point A:

```rust
draw_point_picker(
    ui, "Pt A", &s.body_a, &s.point_a, &s.point_a_name,
    blueprint, index,
    |name, coords| {
        let mut updated = s.clone();
        updated.point_a_name = name;
        updated.point_a = coords;
        ForceElement::LinearSpring(updated)
    },
    &mut pending,
);
draw_point_picker(
    ui, "Pt B", &s.body_b, &s.point_b, &s.point_b_name,
    blueprint, index,
    |name, coords| {
        let mut updated = s.clone();
        updated.point_b_name = name;
        updated.point_b = coords;
        ForceElement::LinearSpring(updated)
    },
    &mut pending,
);
```

If you only set `point_a` without `point_a_name`, the named reference is
silently dropped and the feature is defeated.

- [ ] **Step 3: Repeat for LinearDamper, GasSpring, LinearActuator, ExternalForce**

Apply the same pattern to each force element's draw arm. For ExternalForce,
use `local_point` / `local_point_name` instead of `point_a` / `point_a_name`.

- [ ] **Step 4: Build and verify**

Run: `cd linkage-sim-rs && cargo build 2>&1 | tail -10`
Expected: compiles cleanly.

- [ ] **Step 5: Manual smoke test**

Run: `cd linkage-sim-rs && cargo run`
- Add a mount point to a body
- Add a spring between two bodies
- Select the spring in the property panel
- Verify point picker dropdown appears with joint and mount point options
- Select a mount point — verify coords update
- Select "custom coords..." — verify raw x/y editing works

- [ ] **Step 6: Commit**

```bash
git add linkage-sim-rs/src/gui/property_panel.rs
git commit -m "feat: add point picker dropdown for force element editing"
```

---

### Task 8: Force toolbar — ground inclusion and second body picker

**Files:**
- Modify: `linkage-sim-rs/src/gui/force_toolbar.rs`

- [ ] **Step 1: Update test to expect ground as valid**

Replace `resolve_target_bodies_ground_selection_excluded` test (line 214-220):

```rust
#[test]
fn resolve_target_bodies_ground_is_valid() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);
    state.link_editor_body = Some("ground".to_string());
    let (sel, _conn) = resolve_target_bodies(&state);
    assert_eq!(sel, Some("ground".to_string()), "Ground should be a valid force target");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd linkage-sim-rs && cargo test --lib gui::force_toolbar::tests -- --nocapture`
Expected: FAIL — ground is currently filtered out.

- [ ] **Step 3: Remove ground exclusion filter**

In `resolve_target_bodies()` (line 153-154), remove the `.filter(|id| id != GROUND_ID)`:

```rust
// Before:
let selected_body = state.link_editor_body.clone()
    .filter(|id| id != GROUND_ID)
    .or_else(|| match &state.selected {
        Some(SelectedEntity::Body(id)) if id != GROUND_ID => Some(id.clone()),
        _ => None,
    });

// After:
let selected_body = state.link_editor_body.clone()
    .or_else(|| match &state.selected {
        Some(SelectedEntity::Body(id)) => Some(id.clone()),
        _ => None,
    });
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd linkage-sim-rs && cargo test --lib gui::force_toolbar::tests -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd linkage-sim-rs && cargo test 2>&1 | tail -20`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add linkage-sim-rs/src/gui/force_toolbar.rs
git commit -m "feat: allow ground as force target in toolbar"
```

---

### Task 9: Diamond markers for mount points on canvas

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs`

- [ ] **Step 1: Add mount point color constant**

Near the existing color constants (~line 30), add:

```rust
const MOUNT_POINT_COLOR: Color32 = Color32::from_rgb(224, 86, 253); // #e056fd
const MOUNT_POINT_RADIUS: f32 = 4.0;
```

- [ ] **Step 2: Add diamond drawing helper**

```rust
/// Draw a diamond marker at the given screen position.
fn draw_diamond_marker(painter: &egui::Painter, center: Pos2, radius: f32, color: Color32) {
    let points = vec![
        Pos2::new(center.x, center.y - radius),       // top
        Pos2::new(center.x + radius, center.y),        // right
        Pos2::new(center.x, center.y + radius),        // bottom
        Pos2::new(center.x - radius, center.y),        // left
    ];
    painter.add(egui::Shape::convex_polygon(
        points,
        color,
        egui::Stroke::new(1.5, Color32::WHITE),
    ));
}
```

- [ ] **Step 3: Draw mount points alongside attachment points**

In the body rendering loop, after the attachment point dots are drawn (~line 356-358), add mount point rendering. Follow the same pattern as attachment point collection (lines 300-312) but for `body.mount_points`:

```rust
// After drawing attachment point dots:
for (name, local) in &body.mount_points {
    let global = mech_state.body_point_global(body_id, local, q);
    let sp = view.world_to_screen(global.x, global.y);
    let screen_pos = Pos2::new(sp[0], sp[1]);
    draw_diamond_marker(&painter, screen_pos, MOUNT_POINT_RADIUS, MOUNT_POINT_COLOR);

    // Add to hit targets for selection
    // (reuse AttachmentHit or create MountHit)
}
```

- [ ] **Step 4: Add mount point labels on hover**

When a mount point is hovered, show its name as a tooltip. Follow the existing hover pattern for joints.

- [ ] **Step 5: Build and verify**

Run: `cd linkage-sim-rs && cargo build 2>&1 | tail -10`
Expected: compiles cleanly.

- [ ] **Step 6: Manual smoke test**

Run: `cd linkage-sim-rs && cargo run`
- Add mount points to a body
- Verify diamonds appear at correct positions
- Verify they're visually distinct from joint circles
- Verify tooltips on hover

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat: render mount points as diamond markers on canvas"
```

---

### Task 10: Integration test — mount-point-attached force produces correct results

**Files:**
- Create: `linkage-sim-rs/tests/mount_point_integration.rs` (or add to existing integration test file)

- [ ] **Step 1: Write integration test**

```rust
//! Integration test: mount-point-attached forces produce identical results
//! to equivalent raw-coordinate forces.

use linkage_sim_rs::core::body::{make_bar, make_ground};
use linkage_sim_rs::core::constraint::JointConstraint;
use linkage_sim_rs::core::mechanism::MechanismBuilder;
use linkage_sim_rs::forces::elements::{ForceElement, LinearSpringElement};
use std::collections::HashMap;

#[test]
fn mount_point_spring_matches_raw_coord_spring() {
    // Build a simple crank mechanism with a spring.
    // One uses raw coords, the other uses named mount point.
    // Both should produce identical generalized forces.

    // --- Mechanism with raw coords ---
    let mut ground_raw = make_ground(&[("O2", 0.0, 0.0)]);
    let crank = make_bar("crank", "O2", "A", 0.1, 1.0, 0.001);

    let spring_raw = LinearSpringElement {
        body_a: "ground".to_string(),
        point_a: [0.05, 0.02],
        point_a_name: None,
        body_b: "crank".to_string(),
        point_b: [0.05, 0.0],
        point_b_name: None,
        stiffness: 500.0,
        free_length: 0.03,
    };

    // --- Mechanism with mount point ---
    let mut ground_mp = make_ground(&[("O2", 0.0, 0.0)]);
    ground_mp.add_mount_point("spring_base", 0.05, 0.02).unwrap();

    let mut crank_mp = make_bar("crank", "O2", "A", 0.1, 1.0, 0.001);
    crank_mp.add_mount_point("spring_tip", 0.05, 0.0).unwrap();

    let spring_named = LinearSpringElement {
        body_a: "ground".to_string(),
        point_a: [0.0, 0.0], // will be resolved
        point_a_name: Some("spring_base".to_string()),
        body_b: "crank".to_string(),
        point_b: [0.0, 0.0], // will be resolved
        point_b_name: Some("spring_tip".to_string()),
        stiffness: 500.0,
        free_length: 0.03,
    };

    // Resolve named points
    let mut bodies_mp = HashMap::new();
    bodies_mp.insert("ground".to_string(), ground_mp.clone());
    bodies_mp.insert("crank".to_string(), crank_mp.clone());

    let fe_named = ForceElement::LinearSpring(spring_named);
    let resolved = fe_named.resolve_named_points(&bodies_mp).unwrap();

    // Verify resolved coords match raw
    if let ForceElement::LinearSpring(s) = &resolved {
        assert!((s.point_a[0] - 0.05).abs() < 1e-15);
        assert!((s.point_a[1] - 0.02).abs() < 1e-15);
        assert!((s.point_b[0] - 0.05).abs() < 1e-15);
        assert!((s.point_b[1] - 0.0).abs() < 1e-15);
    } else {
        panic!("wrong variant");
    }

    // Now verify generalized forces match: build two mechanisms and
    // evaluate forces at the same configuration.
    // This requires building actual Mechanisms with joints and drivers,
    // setting up state vectors, and comparing Q vectors.
    // If the MechanismBuilder API is available, construct both and compare:

    use nalgebra::DVector;
    use linkage_sim_rs::core::state::State;

    // Build raw-coord mechanism bodies map for force evaluation
    let mut bodies_raw = HashMap::new();
    bodies_raw.insert("ground".to_string(), make_ground(&[("O2", 0.0, 0.0)]));
    bodies_raw.insert("crank".to_string(), make_bar("crank", "O2", "A", 0.1, 1.0, 0.001));

    // Create a simple state with crank at 45 degrees
    let state = State::new(1); // 1 moving body
    let q = DVector::from_vec(vec![0.0, 0.0, std::f64::consts::FRAC_PI_4]); // x, y, theta
    let q_dot = DVector::zeros(3);

    let fe_raw = ForceElement::LinearSpring(spring_raw);
    let q_raw = fe_raw.evaluate(&state, &bodies_raw, &q, &q_dot, 0.0);
    let q_named = resolved.evaluate(&state, &bodies_mp, &q, &q_dot, 0.0);

    // Generalized forces must be identical
    assert_eq!(q_raw.len(), q_named.len());
    for i in 0..q_raw.len() {
        assert!((q_raw[i] - q_named[i]).abs() < 1e-12,
            "Q[{}] mismatch: raw={}, named={}", i, q_raw[i], q_named[i]);
    }
}
```

- [ ] **Step 2: Run integration test**

Run: `cd linkage-sim-rs && cargo test --test mount_point_integration -- --nocapture`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/tests/mount_point_integration.rs
git commit -m "test: integration test for mount-point-attached forces"
```

---

### Task 11: Final verification and cleanup

- [ ] **Step 1: Run full test suite**

Run: `cd linkage-sim-rs && cargo test 2>&1 | tail -30`
Expected: all tests pass (existing + new).

- [ ] **Step 2: Run clippy**

Run: `cd linkage-sim-rs && cargo clippy -- -D warnings 2>&1 | tail -20`
Expected: no warnings.

- [ ] **Step 3: Verify WASM build**

Run: `cd linkage-sim-rs && cargo build --target wasm32-unknown-unknown 2>&1 | tail -10`
Expected: compiles cleanly.

- [ ] **Step 4: Manual end-to-end test**

Run: `cd linkage-sim-rs && cargo run`
1. Load FourBar sample
2. Select ground body → add mount point "M1" at (0.02, 0.01)
3. Select crank body → add mount point "spring_tip" at (0.005, 0.0)
4. Add a linear spring from toolbar
5. In property panel, change spring's Point A to "M1 (mount)" via dropdown
6. Change spring's Point B to "spring_tip (mount)" via dropdown
7. Run animation — verify spring renders from M1 on ground to spring_tip on crank
8. Save mechanism → reload → verify mount points and spring references preserved
9. Undo → verify mount points restored correctly

- [ ] **Step 5: Commit any final fixes**

```bash
git add -A
git commit -m "fix: final cleanup for force mount points feature"
```
