# Multi-Pivot Body Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GUI support for creating and editing bodies with arbitrary numbers of attachment points (ternary plates, bell-cranks) via three features: Add Pivot context menu, + Body click-to-place tool, and body-aware Draw Link segment snapping.

**Architecture:** All changes are in the existing `linkage-sim-rs/src/gui/` module. The core solver and data model (`Body`, `BodyJson`, `Mechanism`) already support N-point bodies — the gap is purely in the editor tools. State mutations go through the `MechanismJson` blueprint; `rebuild()` constructs a fresh `Mechanism` from it. Undo is a stack of blueprint snapshots.

**Tech Stack:** Rust, egui/eframe, nalgebra. Tests via `cargo test`.

**Spec:** `docs/superpowers/specs/2026-03-18-multi-pivot-body-editor-design.md`

---

## File Structure

All changes are to existing files. No new files created.

| File | Responsibility | Changes |
|------|---------------|---------|
| `linkage-sim-rs/src/gui/state.rs` | Editor state, blueprint mutations, tool state | New types (AddBodyState, SegmentHit), EditorTool::AddBody, ContextMenuTarget refactor, 5 new public methods, 4 raw helpers, existing add_body replaced |
| `linkage-sim-rs/src/gui/canvas.rs` | 2D rendering, hit testing, interaction | Closed polygon rendering, segment hit detection, Add Body interaction/preview, context menu refactor, Draw Link segment snap |
| `linkage-sim-rs/src/gui/mod.rs` | Toolbar, menu bar, keyboard shortcuts | Remove Delete/Set Driver buttons, add + Body, wire Del/Backspace |
| `linkage-sim-rs/src/gui/property_panel.rs` | Property inspection panel | Set Driver button on selected joint |

---

## Task 1: Utility Methods — `world_to_body_local` and `next_attachment_point_name`

**Files:**
- Modify: `linkage-sim-rs/src/gui/state.rs:1284-1305` (near existing `next_body_id`)

These are pure functions with no side effects, so they are easy to test first.

- [ ] **Step 1: Write test for `next_attachment_point_name`**

Add to the test module at the bottom of `state.rs` (tests start at line ~2026):

```rust
#[test]
fn next_attachment_point_name_skips_existing() {
    let mut state = AppState::default();
    // Load a sample to get a blueprint with bodies
    state.load_sample(SampleMechanism::FourBar);
    // FourBar sample uses body names "crank", "coupler", "rocker" with points A/B
    // (defined in gui/samples.rs build_fourbar_at_angle)
    let name = state.next_attachment_point_name("crank");
    assert_eq!(name, "C"); // A and B exist, so next is C
}

#[test]
fn next_attachment_point_name_empty_body() {
    let mut state = AppState::default();
    // Add a body with no attachment points via blueprint
    let bp = state.blueprint.as_mut().unwrap();
    bp.bodies.insert("empty".to_string(), BodyJson {
        attachment_points: HashMap::new(),
        mass: 1.0,
        cg_local: [0.0, 0.0],
        izz_cg: 0.01,
        coupler_points: HashMap::new(),
    });
    let name = state.next_attachment_point_name("empty");
    assert_eq!(name, "A");
}

#[test]
fn next_attachment_point_name_overflow_past_z() {
    let mut state = AppState::default();
    let bp = state.blueprint.as_mut().unwrap();
    let mut pts = HashMap::new();
    // Fill A through Z
    for c in b'A'..=b'Z' {
        pts.insert(String::from(c as char), [0.0, 0.0]);
    }
    bp.bodies.insert("full".to_string(), BodyJson {
        attachment_points: pts,
        mass: 1.0,
        cg_local: [0.0, 0.0],
        izz_cg: 0.01,
        coupler_points: HashMap::new(),
    });
    let name = state.next_attachment_point_name("full");
    assert_eq!(name, "AA");
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd linkage-sim-rs && cargo test next_attachment_point_name -- --nocapture`
Expected: FAIL — method does not exist yet.

- [ ] **Step 3: Implement `next_attachment_point_name`**

Add after `next_ground_pivot_name` (around line 1305) in the `impl AppState` block:

```rust
/// Generate the next unused attachment point name for a body.
///
/// Names follow the sequence A, B, ..., Z, AA, AB, ..., AZ, BA, ...
pub fn next_attachment_point_name(&self, body_id: &str) -> String {
    let Some(bp) = &self.blueprint else {
        return "A".to_string();
    };
    let existing: std::collections::HashSet<&String> = bp
        .bodies
        .get(body_id)
        .map(|b| b.attachment_points.keys().collect())
        .unwrap_or_default();

    let mut name = String::new();
    let mut n: usize = 0;
    loop {
        // Convert n to base-26 letter sequence: 0=A, 1=B, ..., 25=Z, 26=AA, 27=AB, ...
        name.clear();
        let mut val = n;
        loop {
            name.push((b'A' + (val % 26) as u8) as char);
            val /= 26;
            if val == 0 {
                break;
            }
            val -= 1; // Adjust for 1-based (A=1, not A=0) in higher digits
        }
        // Reverse because we built it least-significant-first
        let name_rev: String = name.chars().rev().collect();
        if !existing.contains(&name_rev) {
            return name_rev;
        }
        n += 1;
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd linkage-sim-rs && cargo test next_attachment_point_name -- --nocapture`
Expected: All 3 PASS.

- [ ] **Step 5: Write test for `world_to_body_local`**

```rust
#[test]
fn world_to_body_local_identity_pose() {
    // Body at origin with theta=0: local = world
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);
    // At driver_angle=0, solve to get valid q
    // Ground body is always at (0,0,0), so world_to_body_local should be identity
    let [lx, ly] = state.world_to_body_local("ground", 0.05, 0.03);
    assert!((lx - 0.05).abs() < 1e-10);
    assert!((ly - 0.03).abs() < 1e-10);
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd linkage-sim-rs && cargo test world_to_body_local -- --nocapture`
Expected: FAIL — method does not exist yet.

- [ ] **Step 7: Implement `world_to_body_local`**

Add near `next_attachment_point_name`:

```rust
/// Convert world coordinates to body-local coordinates using the body's
/// current pose (x, y, theta) from the state vector q.
///
/// For ground (pose always 0,0,0), local = world.
pub fn world_to_body_local(&self, body_id: &str, world_x: f64, world_y: f64) -> [f64; 2] {
    if body_id == GROUND_ID {
        return [world_x, world_y];
    }
    // Extract q_start from the mechanism borrow, then drop it before
    // accessing self.q to avoid borrow checker conflict.
    let q_start = match &self.mechanism {
        Some(mech) => match mech.state().get_index(body_id) {
            Ok(idx) => idx.q_start,
            Err(_) => return [world_x, world_y],
        },
        None => return [world_x, world_y],
    };
    let bx = self.q[q_start];
    let by = self.q[q_start + 1];
    let theta = self.q[q_start + 2];
    let dx = world_x - bx;
    let dy = world_y - by;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    [cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy]
}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd linkage-sim-rs && cargo test world_to_body_local -- --nocapture`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add linkage-sim-rs/src/gui/state.rs
git commit -m "feat(gui): add world_to_body_local and next_attachment_point_name utilities"
```

---

## Task 2: Raw Blueprint Helpers

**Files:**
- Modify: `linkage-sim-rs/src/gui/state.rs`

Extract the blueprint mutation logic from existing public methods into `_raw` variants that skip undo and rebuild. This is the foundation for compound operations.

- [ ] **Step 1: Write tests for raw helpers**

```rust
#[test]
fn add_ground_pivot_raw_mutates_blueprint() {
    let mut state = AppState::default();
    state.add_ground_pivot_raw("P1", 0.1, 0.2);
    let bp = state.blueprint.as_ref().unwrap();
    let ground = bp.bodies.get("ground").unwrap();
    assert!(ground.attachment_points.contains_key("P1"));
    let pt = ground.attachment_points.get("P1").unwrap();
    assert!((pt[0] - 0.1).abs() < 1e-15);
    assert!((pt[1] - 0.2).abs() < 1e-15);
    // Should NOT have pushed undo
    assert!(!state.can_undo());
}

#[test]
fn add_revolute_joint_raw_mutates_blueprint() {
    let mut state = AppState::default();
    // Set up two bodies with attachment points
    state.add_ground_pivot_raw("O", 0.0, 0.0);
    let bp = state.blueprint.as_mut().unwrap();
    let mut pts = HashMap::new();
    pts.insert("A".to_string(), [0.0, 0.0]);
    pts.insert("B".to_string(), [0.1, 0.0]);
    bp.bodies.insert("link".to_string(), BodyJson {
        attachment_points: pts,
        mass: 1.0,
        cg_local: [0.05, 0.0],
        izz_cg: 0.01,
        coupler_points: HashMap::new(),
    });
    state.add_revolute_joint_raw("ground", "O", "link", "A");
    let bp = state.blueprint.as_ref().unwrap();
    assert_eq!(bp.joints.len(), 1);
    assert!(!state.can_undo());
}

#[test]
fn add_body_with_points_raw_first_point_is_local_origin() {
    let mut state = AppState::default();
    let points = vec![
        ("A".to_string(), [0.05, 0.03]),
        ("B".to_string(), [0.15, 0.03]),
        ("C".to_string(), [0.10, 0.08]),
    ];
    let new_body_id = state.next_body_id();
    state.add_body_with_points_raw(&new_body_id, &points);
    let bp = state.blueprint.as_ref().unwrap();
    // Should have a new body (body_1 or similar)
    let body = bp.bodies.values()
        .find(|b| b.attachment_points.contains_key("A")
            && b.attachment_points.contains_key("C"))
        .expect("body with points A, B, C");
    // A should be at local (0,0)
    let a = body.attachment_points.get("A").unwrap();
    assert!((a[0]).abs() < 1e-15);
    assert!((a[1]).abs() < 1e-15);
    // B should be offset from A: (0.15-0.05, 0.03-0.03) = (0.10, 0.00)
    let b = body.attachment_points.get("B").unwrap();
    assert!((b[0] - 0.10).abs() < 1e-15);
    assert!((b[1] - 0.00).abs() < 1e-15);
    // C should be offset from A: (0.10-0.05, 0.08-0.03) = (0.05, 0.05)
    let c = body.attachment_points.get("C").unwrap();
    assert!((c[0] - 0.05).abs() < 1e-15);
    assert!((c[1] - 0.05).abs() < 1e-15);
    // CG should be centroid of local points: A=(0,0), B=(0.10,0), C=(0.05,0.05)
    let expected_cg_x = (0.0 + 0.10 + 0.05) / 3.0;
    let expected_cg_y = (0.0 + 0.0 + 0.05) / 3.0;
    assert!((body.cg_local[0] - expected_cg_x).abs() < 1e-10);
    assert!((body.cg_local[1] - expected_cg_y).abs() < 1e-10);
    assert!(!state.can_undo());
}

#[test]
fn add_attachment_point_local_raw_adds_to_existing_body() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);
    // Crank has A and B. Add C in body-local coords.
    state.add_attachment_point_local_raw("crank", "C", 0.005, 0.003);
    let bp = state.blueprint.as_ref().unwrap();
    let crank = bp.bodies.get("crank").unwrap();
    assert!(crank.attachment_points.contains_key("C"));
    let c = crank.attachment_points.get("C").unwrap();
    assert!((c[0] - 0.005).abs() < 1e-15);
    assert!((c[1] - 0.003).abs() < 1e-15);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd linkage-sim-rs && cargo test _raw -- --nocapture`
Expected: FAIL — methods do not exist yet.

- [ ] **Step 3: Implement raw helpers**

Add to the `impl AppState` block. Refactor existing methods to call their `_raw` variants internally.

```rust
// ── Raw blueprint helpers (no undo, no rebuild) ────────────────────

/// Add a ground pivot to the blueprint without pushing undo or rebuilding.
pub(crate) fn add_ground_pivot_raw(&mut self, name: &str, x: f64, y: f64) {
    let Some(bp) = &mut self.blueprint else { return };
    let ground = bp.bodies.entry(GROUND_ID.to_string()).or_insert_with(|| BodyJson {
        attachment_points: HashMap::new(),
        mass: 0.0,
        cg_local: [0.0, 0.0],
        izz_cg: 0.0,
        coupler_points: HashMap::new(),
    });
    ground.attachment_points.insert(name.to_string(), [x, y]);
}

/// Add a revolute joint to the blueprint without pushing undo or rebuilding.
pub(crate) fn add_revolute_joint_raw(
    &mut self,
    body_i: &str,
    point_i: &str,
    body_j: &str,
    point_j: &str,
) {
    let Some(bp) = &mut self.blueprint else { return };
    let joint_id = generate_unique_id("J", &bp.joints);
    bp.joints.insert(
        joint_id,
        JointJson::Revolute {
            body_i: body_i.to_string(),
            body_j: body_j.to_string(),
            point_i: point_i.to_string(),
            point_j: point_j.to_string(),
        },
    );
}

/// Add a body with N world-coordinate points to the blueprint.
/// First point becomes local (0,0); others are offset from it.
/// Caller provides the body_id (use `next_body_id()` to generate one).
/// Does NOT push undo or rebuild.
pub(crate) fn add_body_with_points_raw(
    &mut self,
    body_id: &str,
    points: &[(String, [f64; 2])],
) {
    if points.len() < 2 {
        return;
    }
    let Some(bp) = &mut self.blueprint else { return };
    let origin = points[0].1;
    let mut attachment_points = HashMap::new();
    for (name, [wx, wy]) in points {
        let lx = wx - origin[0];
        let ly = wy - origin[1];
        attachment_points.insert(name.clone(), [lx, ly]);
    }
    // CG at centroid of local points
    let n = attachment_points.len() as f64;
    let cx: f64 = attachment_points.values().map(|p| p[0]).sum::<f64>() / n;
    let cy: f64 = attachment_points.values().map(|p| p[1]).sum::<f64>() / n;
    bp.bodies.insert(
        body_id.to_string(),
        BodyJson {
            attachment_points,
            mass: 1.0,
            cg_local: [cx, cy],
            izz_cg: 0.01,
            coupler_points: HashMap::new(),
        },
    );
}

/// Add an attachment point to an existing body in body-local coordinates.
/// Does NOT push undo or rebuild.
pub(crate) fn add_attachment_point_local_raw(
    &mut self,
    body_id: &str,
    name: &str,
    local_x: f64,
    local_y: f64,
) {
    let Some(bp) = &mut self.blueprint else { return };
    if let Some(body) = bp.bodies.get_mut(body_id) {
        body.attachment_points.insert(name.to_string(), [local_x, local_y]);
    }
}
```

- [ ] **Step 4: Refactor existing public methods to use `_raw` variants**

Replace the bodies of `add_ground_pivot`, `add_body`, and `add_revolute_joint` to delegate:

```rust
pub fn add_ground_pivot(&mut self, name: &str, x: f64, y: f64) {
    self.push_undo();
    self.add_ground_pivot_raw(name, x, y);
    self.rebuild();
}
```

Replace `add_body` entirely with `add_body_with_points`:

```rust
/// Add a new body with N attachment points specified in world coordinates.
///
/// First point becomes the body-local origin (0,0). Others are offset from it.
/// Auto-generates a unique body ID. Pushes undo and rebuilds.
pub fn add_body_with_points(&mut self, points: &[(String, [f64; 2])]) {
    self.push_undo();
    let body_id = self.next_body_id();
    self.add_body_with_points_raw(&body_id, points);
    self.rebuild();
}
```

Replace `add_revolute_joint` body:

```rust
pub fn add_revolute_joint(
    &mut self,
    body_i: &str,
    point_i: &str,
    body_j: &str,
    point_j: &str,
) {
    self.push_undo();
    self.add_revolute_joint_raw(body_i, point_i, body_j, point_j);
    self.rebuild();
}
```

- [ ] **Step 5: Update Draw Link in canvas.rs to use `_raw` pattern**

Find the Draw Link release handler (around line 647-699 in canvas.rs). Replace the individual method calls with the compound pattern:

```rust
if dist_px > 10.0 {
    state.push_undo();

    // Start connection: existing point or new ground pivot.
    let start_attach = start.attachment.clone().unwrap_or_else(|| {
        let name = state.next_ground_pivot_name();
        state.add_ground_pivot_raw(&name, sx, sy);
        (GROUND_ID.to_string(), name)
    });

    // Create the body with endpoints at exact snap positions.
    let body_id = state.next_body_id();
    let points = vec![
        ("A".to_string(), [sx, sy]),
        ("B".to_string(), [ex, ey]),
    ];
    let new_body_id = state.next_body_id();
    state.add_body_with_points_raw(&new_body_id, &points);

    // Joint at start.
    state.add_revolute_joint_raw(
        &start_attach.0,
        &start_attach.1,
        &body_id,
        "A",
    );

    // Joint at end (only if snapped to existing point).
    if let Some((end_body, end_point)) = end_attach {
        state.add_revolute_joint_raw(
            &end_body,
            &end_point,
            &body_id,
            "B",
        );
    }

    state.rebuild();
}
```

**Note:** The old `add_body` call used positional args `(body_id, ("A", sx, sy), ("B", ex, ey))`. The new `add_body_with_points_raw` takes `body_id: &str` as its first parameter so the caller controls the ID. Call `state.next_body_id()` before `add_body_with_points_raw` to get the ID needed for subsequent joint calls.

- [ ] **Step 6: Update existing tests that call `add_body`**

Find tests that call `state.add_body(...)` (search for `add_body(` in the test module). Update them to use `add_body_with_points` with the new signature.

- [ ] **Step 7: Run full test suite**

Run: `cd linkage-sim-rs && cargo test`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add linkage-sim-rs/src/gui/state.rs linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat(gui): add raw blueprint helpers and refactor undo batching

Existing mutation methods now delegate to _raw variants that skip undo
and rebuild. Draw Link refactored to use compound undo pattern (one
push_undo + N raw mutations + one rebuild)."
```

---

## Task 3: `add_attachment_point_to_body` and `remove_attachment_point`

**Files:**
- Modify: `linkage-sim-rs/src/gui/state.rs`

These are the standalone (undo-pushing) methods for Feature A (Add Pivot) and Delete Pivot.

- [ ] **Step 1: Write tests**

```rust
#[test]
fn add_attachment_point_to_body_converts_world_to_local() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);
    // Ground is at (0,0,0), so world = local for ground
    state.add_attachment_point_to_body("ground", "PX", 0.05, 0.02);
    let bp = state.blueprint.as_ref().unwrap();
    let ground = bp.bodies.get("ground").unwrap();
    let px = ground.attachment_points.get("PX").unwrap();
    assert!((px[0] - 0.05).abs() < 1e-10);
    assert!((px[1] - 0.02).abs() < 1e-10);
    assert!(state.can_undo());
}

#[test]
fn remove_attachment_point_cascades_to_joints() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);
    // Crank has points A and B. J1 connects ground:O2 to crank:A.
    // Removing crank:A should also remove J1.
    let bp = state.blueprint.as_ref().unwrap();
    let joint_count_before = bp.joints.len();
    state.remove_attachment_point("crank", "A");
    let bp = state.blueprint.as_ref().unwrap();
    assert!(!bp.bodies.get("crank").unwrap().attachment_points.contains_key("A"));
    assert!(bp.joints.len() < joint_count_before);
    assert!(state.can_undo());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd linkage-sim-rs && cargo test add_attachment_point_to_body -- --nocapture && cargo test remove_attachment_point_cascades -- --nocapture`
Expected: FAIL — methods do not exist yet.

- [ ] **Step 3: Implement both methods**

```rust
/// Add an attachment point to an existing body, converting world coordinates
/// to body-local using the body's current pose. Pushes undo, rebuilds.
pub fn add_attachment_point_to_body(
    &mut self,
    body_id: &str,
    name: &str,
    world_x: f64,
    world_y: f64,
) {
    self.push_undo();
    let [lx, ly] = self.world_to_body_local(body_id, world_x, world_y);
    self.add_attachment_point_local_raw(body_id, name, lx, ly);
    self.rebuild();
}

/// Remove an attachment point from a body. Cascades: removes any joints
/// that reference this (body_id, point_name) pair. Pushes undo, rebuilds.
pub fn remove_attachment_point(&mut self, body_id: &str, point_name: &str) {
    self.push_undo();
    let Some(bp) = &mut self.blueprint else { return };
    if let Some(body) = bp.bodies.get_mut(body_id) {
        body.attachment_points.remove(point_name);
    }
    // Cascade: remove joints referencing this point.
    bp.joints.retain(|_id, joint| {
        !joint_references_point(joint, body_id, point_name)
    });
    self.rebuild();
}
```

Also add the helper `joint_references_point` near the existing `joint_body_ids` helper:

```rust
fn joint_references_point(joint: &JointJson, body_id: &str, point_name: &str) -> bool {
    match joint {
        JointJson::Revolute { body_i, point_i, body_j, point_j, .. }
        | JointJson::Prismatic { body_i, point_i, body_j, point_j, .. }
        | JointJson::Fixed { body_i, point_i, body_j, point_j, .. } => {
            (body_i == body_id && point_i == point_name)
                || (body_j == body_id && point_j == point_name)
        }
        // RevoluteDriver has body_i/body_j but no point_i/point_j fields.
        // It references bodies, not specific attachment points, so it
        // cannot match a point removal. Return false.
        JointJson::RevoluteDriver { .. } => false,
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd linkage-sim-rs && cargo test add_attachment_point_to_body -- --nocapture && cargo test remove_attachment_point_cascades -- --nocapture`
Expected: Both PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd linkage-sim-rs && cargo test`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add linkage-sim-rs/src/gui/state.rs
git commit -m "feat(gui): add add_attachment_point_to_body and remove_attachment_point

Both push undo and rebuild. remove_attachment_point cascades to joints
that reference the removed point."
```

---

## Task 4: EditorTool::AddBody, AddBodyState, ContextMenuTarget Refactor

**Files:**
- Modify: `linkage-sim-rs/src/gui/state.rs:227-253` (EditorTool, ContextMenuTarget)
- Modify: `linkage-sim-rs/src/gui/state.rs:399-560` (AppState fields, Default impl)
- Modify: `linkage-sim-rs/src/gui/canvas.rs:750-830` (context menu hit test + rendering)

This task changes types that are used across files, so it must compile-fix both state.rs and canvas.rs together.

- [ ] **Step 1: Add `EditorTool::AddBody` variant and `AddBodyState` struct**

In state.rs, add `AddBody` to the EditorTool enum (line ~227):

```rust
pub enum EditorTool {
    Select,
    DrawLink,
    AddBody,
    AddGroundPivot,
}
```

Add the `AddBodyState` struct near `DrawLinkStart` (line ~466):

```rust
/// Tracks placement state for the Add Body tool.
#[derive(Debug, Clone)]
pub struct AddBodyState {
    /// Points placed so far: (name, world_position).
    pub points: Vec<(String, [f64; 2])>,
}
```

Add `add_body_state: Option<AddBodyState>` field to AppState (near line ~461). Initialize to `None` in Default impl (near line ~534).

- [ ] **Step 2: Refactor `ContextMenuTarget`**

Replace the struct (line ~246) with:

```rust
#[derive(Debug, Clone, Default)]
pub struct ContextMenuTarget {
    /// Joint ID if right-click landed on a joint.
    pub joint_id: Option<String>,
    /// Attachment point under cursor (body_id, point_name).
    /// Takes priority over body_area when both could match.
    pub attachment_point: Option<(String, String)>,
    /// Body area under cursor (body_id) — set only when no
    /// attachment point is within HIT_RADIUS but cursor is
    /// near a body line segment.
    pub body_area: Option<String>,
    /// World coordinates of the right-click position.
    pub world_pos: Option<[f64; 2]>,
}
```

- [ ] **Step 3: Fix canvas.rs context menu hit-test (line ~750-770)**

Replace the right-click handler to populate the new fields. The existing code at line ~757-760 finds a body attachment point. Now we need to check attachment points first, then body segments:

```rust
if response.secondary_clicked() {
    if let Some(pos) = response.interact_pointer_pos() {
        let joint_id = joint_hit_targets
            .iter()
            .find(|(screen_pos, _)| pos.distance(*screen_pos) <= HIT_RADIUS)
            .map(|(_, id)| id.clone());

        // Check attachment points first (priority over body area).
        let attachment_point = attachment_hit_targets
            .iter()
            .find(|h| h.body_id != GROUND_ID && pos.distance(h.screen_pos) <= HIT_RADIUS)
            .map(|h| (h.body_id.clone(), h.point_name.clone()));

        // Body area: only if no attachment point matched.
        // For now, use attachment_hit_targets as a proxy — if any point
        // on the body is within a larger radius, consider it a body hit.
        // (Full segment hit testing is added in Task 8.)
        let body_area = if attachment_point.is_none() {
            attachment_hit_targets
                .iter()
                .find(|h| h.body_id != GROUND_ID && pos.distance(h.screen_pos) <= HIT_RADIUS * 3.0)
                .map(|h| h.body_id.clone())
        } else {
            None
        };

        let world_pos = Some(state.view.screen_to_world(pos.x, pos.y));

        state.context_menu_target = ContextMenuTarget {
            joint_id,
            attachment_point,
            body_area,
            world_pos,
        };
    }
}
```

- [ ] **Step 4: Fix canvas.rs context menu rendering (line ~775-830)**

Replace the context_menu closure body to use the new fields:

```rust
let ctx_target = state.context_menu_target.clone();
response.context_menu(|ui| {
    if let Some(ref joint_id) = ctx_target.joint_id {
        // Joint context menu — unchanged except remove driver buttons
        // that will be re-added properly later
        ui.label(format!("Joint: {}", joint_id));
        ui.separator();

        let is_grounded_revolute = grounded_revolute_ids.contains(joint_id);
        let is_current_driver =
            current_driver_joint.as_deref() == Some(joint_id.as_str());

        if is_grounded_revolute && !is_current_driver {
            if ui.button("Set as Driver").clicked() {
                state.pending_driver_reassignment = Some(joint_id.clone());
                ui.close();
            }
        }

        if ui.button("Delete Joint").clicked() {
            state.remove_joint(joint_id);
            ui.close();
        }
    } else if let Some((ref body_id, ref point_name)) = ctx_target.attachment_point {
        // Attachment point context menu
        ui.label(format!("Point: {}:{}", body_id, point_name));
        ui.separator();

        if ui.button("Create Joint").clicked() {
            state.creating_joint = Some((body_id.clone(), point_name.clone()));
            ui.close();
        }

        if ui.button("Delete Pivot").clicked() {
            state.remove_attachment_point(body_id, point_name);
            ui.close();
        }

        // Set as Driver — only shown if this attachment point belongs
        // to a grounded revolute joint.
        if let Some(mech) = &state.mechanism {
            let grounded = mech.grounded_revolute_joint_ids();
            // Find any grounded revolute joint that references this body+point
            for joint in mech.joints() {
                if joint.is_revolute()
                    && grounded.contains(&joint.id().to_string())
                    && ((joint.body_i_id() == *body_id) || (joint.body_j_id() == *body_id))
                    && current_driver_joint.as_deref() != Some(joint.id())
                {
                    if ui.button("Set as Driver").clicked() {
                        state.pending_driver_reassignment = Some(joint.id().to_string());
                        ui.close();
                    }
                    break;
                }
            }
        }
    } else if let Some(ref body_id) = ctx_target.body_area {
        // Body area context menu
        ui.label(format!("Body: {}", body_id));
        ui.separator();

        if ui.button("Add Pivot Here").clicked() {
            if let Some([wx, wy]) = ctx_target.world_pos {
                let name = state.next_attachment_point_name(body_id);
                state.add_attachment_point_to_body(body_id, &name, wx, wy);
            }
            ui.close();
        }

        if ui.button("Delete Body").clicked() {
            state.remove_body(body_id);
            state.selected = None;
            ui.close();
        }
    } else {
        // Empty canvas context menu
        if let Some([wx, wy]) = ctx_target.world_pos {
            if ui.button("Add Ground Pivot Here").clicked() {
                let name = state.next_ground_pivot_name();
                state.add_ground_pivot(&name, wx, wy);
                ui.close();
            }
            if ui.button("Start Body Here").clicked() {
                let (sx, sy) = state.grid.snap_point(wx, wy);
                let name = state.next_attachment_point_name("__pending__");
                state.active_tool = EditorTool::AddBody;
                state.add_body_state = Some(AddBodyState {
                    points: vec![(name, [sx, sy])],
                });
                ui.close();
            }
        }
    }
});
```

- [ ] **Step 5: Fix any remaining compile errors**

Search for all uses of `ctx_target.body` or `context_menu_target.body` and update to use `attachment_point` or `body_area` as appropriate.

- [ ] **Step 6: Run full test suite**

Run: `cd linkage-sim-rs && cargo test`
Expected: All pass. Also run `cargo run --bin linkage-gui` and verify context menus work.

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/gui/state.rs linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat(gui): add EditorTool::AddBody, refactor ContextMenuTarget

ContextMenuTarget now distinguishes attachment_point from body_area hits.
Context menus show appropriate actions for each: Add Pivot Here on body
area, Create Joint / Delete Pivot on attachment points."
```

---

## Task 5: Toolbar Cleanup — Remove Delete/Set Driver, Add + Body

**Files:**
- Modify: `linkage-sim-rs/src/gui/mod.rs:223-298` (toolbar section)
- Modify: `linkage-sim-rs/src/gui/mod.rs:30-43` (keyboard shortcut handler)

- [ ] **Step 1: Replace toolbar section**

In mod.rs, replace the toolbar section (lines ~223-298). Remove Delete and Set Driver buttons. Add `[+ Body]`:

```rust
egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
    ui.horizontal(|ui| {
        ui.spacing_mut().button_padding = egui::vec2(6.0, 3.0);

        let tool = self.state.active_tool;

        if ui
            .selectable_label(tool == EditorTool::Select, "Select")
            .on_hover_text("Select and move entities")
            .clicked()
        {
            self.state.active_tool = EditorTool::Select;
            self.state.draw_link_start = None;
            self.state.add_body_state = None;
        }

        if ui
            .selectable_label(
                tool == EditorTool::DrawLink || self.state.draw_link_start.is_some(),
                "Draw Link",
            )
            .on_hover_text("Click and drag to draw a link")
            .clicked()
        {
            self.state.active_tool = EditorTool::DrawLink;
            self.state.draw_link_start = None;
            self.state.add_body_state = None;
        }

        if ui
            .selectable_label(
                tool == EditorTool::AddBody || self.state.add_body_state.is_some(),
                "+ Body",
            )
            .on_hover_text("Click to place attachment points, double-click or Enter to finish")
            .clicked()
        {
            self.state.active_tool = EditorTool::AddBody;
            self.state.draw_link_start = None;
            self.state.add_body_state = None;
        }

        if ui
            .selectable_label(tool == EditorTool::AddGroundPivot, "+ Ground")
            .on_hover_text("Click canvas to place a ground pivot")
            .clicked()
        {
            self.state.active_tool = EditorTool::AddGroundPivot;
            self.state.draw_link_start = None;
            self.state.add_body_state = None;
        }
    });
});
```

- [ ] **Step 2: Add Del/Backspace keyboard shortcut**

In the keyboard shortcuts section (around line ~32), add:

```rust
// Delete key deletes selected entity
if ctx.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)) {
    match self.state.selected.take() {
        Some(SelectedEntity::Body(id)) => {
            self.state.remove_body(&id);
        }
        Some(SelectedEntity::Joint(id)) => {
            self.state.remove_joint(&id);
        }
        other => {
            self.state.selected = other; // Put it back if not handled
        }
    }
}
```

- [ ] **Step 3: Add Set Driver to property panel**

In `property_panel.rs`, find the joint property display section (where joint properties are shown when a joint is selected). Add a "Set as Driver" button:

```rust
// After showing joint properties, if it's a grounded revolute:
if let Some(mech) = &state.mechanism {
    let grounded = mech.grounded_revolute_joint_ids();
    if grounded.contains(&joint_id.to_string()) {
        let is_current = state.driver_joint_id.as_deref() == Some(joint_id);
        if !is_current {
            if ui.button("Set as Driver").clicked() {
                state.pending_driver_reassignment = Some(joint_id.to_string());
            }
        } else {
            ui.label("(Current driver)");
        }
    }
}
```

- [ ] **Step 4: Run and verify visually**

Run: `cd linkage-sim-rs && cargo run --bin linkage-gui`
Verify: Toolbar shows [Select] [Draw Link] [+ Body] [+ Ground]. No Delete or Set Driver buttons. Del key deletes selected entity.

- [ ] **Step 5: Run test suite**

Run: `cd linkage-sim-rs && cargo test`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add linkage-sim-rs/src/gui/mod.rs linkage-sim-rs/src/gui/property_panel.rs
git commit -m "feat(gui): toolbar cleanup — modes only, actions via keyboard/menus

Remove Delete and Set Driver from toolbar. Add + Body button.
Del/Backspace key deletes selected entity. Set Driver moved to
context menu and property panel."
```

---

## Task 6: Close Polygon Rendering for 3+ Point Bodies

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs:197-203` (body rendering loop)

- [ ] **Step 1: Update body rendering to close polygons**

Find the body rendering code (around line 197-203). Currently it draws lines between consecutive sorted points but doesn't close the shape for 3+ points. Add the closing segment:

```rust
// Draw lines between consecutive attachment points.
if screen_points.len() >= 2 {
    for pair in screen_points.windows(2) {
        painter.line_segment(
            [pair[0], pair[1]],
            Stroke::new(BODY_STROKE_WIDTH, color),
        );
    }
    // Close polygon for bodies with 3+ points.
    if screen_points.len() >= 3 {
        painter.line_segment(
            [*screen_points.last().unwrap(), screen_points[0]],
            Stroke::new(BODY_STROKE_WIDTH, color),
        );
    }
}
```

- [ ] **Step 2: Verify visually**

Run: `cd linkage-sim-rs && cargo run --bin linkage-gui`
Load a 6-bar sample (e.g., "6-Bar B1 (Watt I)") that has ternary bodies. Verify the ternary body is now rendered as a closed triangle rather than an open path.

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat(gui): close polygon rendering for 3+ point bodies

Ternary and higher-order bodies now render as closed shapes instead
of open polylines."
```

---

## Task 7: Add Body Tool — Interaction and Preview

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs` (interaction handler + rendering)
- Modify: `linkage-sim-rs/src/gui/state.rs` (Escape handler cleanup)

This is the largest single task. It adds the click-to-place flow for the + Body tool.

- [ ] **Step 1: Add hint text for Add Body mode**

In the tool-mode hint text section of canvas.rs (around line 434-456), add the Add Body case:

```rust
EditorTool::AddBody => {
    if state.add_body_state.is_some() {
        Some("Click to add points, double-click or Enter to finish (Esc to cancel)")
    } else {
        Some("Click to place first point of new body (Esc to cancel)")
    }
}
```

- [ ] **Step 2: Add preview rendering for Add Body mode**

After the existing joint creation highlight rendering (around line 354), add Add Body preview:

```rust
// ── Add Body mode: render placed points and preview ─────────
if let Some(ref abs) = state.add_body_state {
    let placed_points: Vec<Pos2> = abs.points.iter().map(|(_, [wx, wy])| {
        let sp = state.view.world_to_screen(*wx, *wy);
        Pos2::new(sp[0], sp[1])
    }).collect();

    // Draw connecting lines between placed points (dashed-style: use thinner stroke).
    if placed_points.len() >= 2 {
        for pair in placed_points.windows(2) {
            painter.line_segment(
                [pair[0], pair[1]],
                Stroke::new(2.0, JOINT_CREATE_HIGHLIGHT),
            );
        }
        // Close preview polygon for 3+ points.
        if placed_points.len() >= 3 {
            let dimmer_green = Color32::from_rgba_premultiplied(60, 230, 100, 80);
            painter.line_segment(
                [*placed_points.last().unwrap(), placed_points[0]],
                Stroke::new(1.5, dimmer_green),
            );
        }
    }

    // Draw green dots at each placed point.
    for sp in &placed_points {
        painter.circle_filled(*sp, JOINT_RADIUS, JOINT_CREATE_HIGHLIGHT);
    }

    // Ghost dot at cursor position with connecting line from last placed point.
    if let Some(hover_pos) = ui.input(|i| i.pointer.hover_pos()) {
        if canvas_rect.contains(hover_pos) {
            let [gwx, gwy] = state.view.screen_to_world(hover_pos.x, hover_pos.y);
            let (sx, sy) = state.grid.snap_point(gwx, gwy);
            let ghost_screen = state.view.world_to_screen(sx, sy);
            let ghost_pos = Pos2::new(ghost_screen[0], ghost_screen[1]);

            let ghost_color = Color32::from_rgba_premultiplied(60, 230, 100, 120);
            painter.circle_filled(ghost_pos, JOINT_RADIUS * 0.7, ghost_color);

            if let Some(last) = placed_points.last() {
                painter.line_segment(
                    [*last, ghost_pos],
                    Stroke::new(1.5, ghost_color),
                );
            }
        }
    }
}
```

- [ ] **Step 3: Add Add Body click interaction**

Add a new interaction block for Add Body mode. Place it after the Draw Link section and before the selection click handler. The key detail: check `double_clicked()` BEFORE `clicked()` to avoid the double-placement bug.

```rust
// ── Interaction: Add Body tool ──────────────────────────────────
if state.active_tool == EditorTool::AddBody {
    // Enter key finishes the body.
    if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
        if let Some(ref abs) = state.add_body_state {
            if abs.points.len() >= 2 {
                state.add_body_with_points(&abs.points);
                state.add_body_state = None;
                // Stay in AddBody mode for chaining
            }
            // If < 2 points, stay in mode (user needs to add more)
        }
    }

    // Double-click finishes (same as Enter — do NOT place a new point).
    // Must check before clicked() to prevent double-fire.
    if response.double_clicked() {
        if let Some(ref abs) = state.add_body_state {
            if abs.points.len() >= 2 {
                state.add_body_with_points(&abs.points);
                state.add_body_state = None;
            }
        }
    } else if response.clicked() {
        // Single click: place a point.
        if let Some(pos) = response.interact_pointer_pos() {
            let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
            let (sx, sy) = state.grid.snap_point(wx, wy);

            if let Some(ref mut abs) = state.add_body_state {
                // Use same naming logic as next_attachment_point_name:
                // A, B, ..., Z, AA, AB, ... Gracefully handles >26 points.
                let n = abs.points.len();
                let name = if n < 26 {
                    String::from((b'A' + n as u8) as char)
                } else {
                    // Overflow: AA, AB, etc.
                    let hi = (n - 26) / 26;
                    let lo = (n - 26) % 26;
                    format!("{}{}", (b'A' + hi as u8) as char, (b'A' + lo as u8) as char)
                };
                abs.points.push((name, [sx, sy]));
            } else {
                state.add_body_state = Some(AddBodyState {
                    points: vec![("A".to_string(), [sx, sy])],
                });
            }
        }
    }
}
```

- [ ] **Step 4: Update Escape handler to clear add_body_state**

In the Escape key handler (around line 556-560), add:

```rust
if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
    state.creating_joint = None;
    state.draw_link_start = None;
    state.add_body_state = None;
    state.active_tool = EditorTool::Select;
}
```

- [ ] **Step 5: Verify visually**

Run: `cd linkage-sim-rs && cargo run --bin linkage-gui`
1. Click `[+ Body]` in toolbar
2. Click 3 points on canvas — green dots + lines should appear
3. Press Enter — body should be created (appears in blue, validation warns "Disconnected")
4. Press Esc — should cancel and return to Select
5. Test double-click to finish
6. Test "Start Body Here" from empty canvas right-click context menu

- [ ] **Step 6: Run test suite**

Run: `cd linkage-sim-rs && cargo test`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs linkage-sim-rs/src/gui/state.rs
git commit -m "feat(gui): Add Body tool — click-to-place multi-point bodies

New + Body toolbar mode: click to place attachment points, double-click
or Enter to finish. Preview shows placed points, connecting lines, and
ghost dot at cursor. Supports ternary plates and higher-order bodies."
```

---

## Task 8: Create Joint Two-Click Flow

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs` (click handler)
- Modify: `linkage-sim-rs/src/gui/state.rs` (validation helpers if needed)

The `creating_joint` state and green ring rendering already exist. This task wires the context menu trigger and the second-click handler.

- [ ] **Step 1: Add Create Joint click handler**

In the canvas click handler section, add a block that fires BEFORE the selection handler when `creating_joint` is active:

```rust
// ── Interaction: Create Joint two-click flow ────────────────────
// This must fire BEFORE the selection handler to consume the click.
if state.creating_joint.is_some() && response.clicked() {
    if let Some(pos) = response.interact_pointer_pos() {
        let second_hit = find_nearest_attachment(pos);
        if let Some(hit) = second_hit {
            let (first_body, first_point) = state.creating_joint.clone().unwrap();
            let second_body = &hit.body_id;
            let second_point = &hit.point_name;

            // Validate: not same body, not ground-ground
            if first_body == *second_body {
                // Invalid: same body
                // (Could show status message, for now just ignore)
            } else if first_body == GROUND_ID && *second_body == GROUND_ID {
                // Invalid: ground-ground
            } else {
                state.add_revolute_joint(
                    &first_body,
                    &first_point,
                    second_body,
                    second_point,
                );
                state.creating_joint = None;
            }
        }
        // If click was not near a valid point, stay in creating_joint mode.
    }
} else if state.creating_joint.is_none()
```

Update the existing selection click handler to only fire when `creating_joint.is_none()`. The existing click handler (around line 689-730) has a guard for `state.drag_target.is_none()` — add `&& state.creating_joint.is_none()` to that guard.

- [ ] **Step 2: Verify visually**

Run: `cd linkage-sim-rs && cargo run --bin linkage-gui`
1. Load a 4-bar. Right-click an attachment point → "Create Joint" appears
2. Click it — green ring appears on that point
3. Click another body's attachment point — joint is created
4. Verify Esc cancels the flow

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat(gui): wire Create Joint two-click flow from context menu

Right-click attachment point → Create Joint starts two-click flow.
Second click on a valid target creates a revolute joint. Invalid
pairings (same body, ground-ground) are silently rejected."
```

---

## Task 9: Body-Aware Draw Link — Segment Hit Detection

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs`

This adds segment hit detection to the Draw Link tool so it can snap to body edges and auto-create pivots.

- [ ] **Step 1: Add `collect_body_segments` and `find_nearest_segment` helpers**

Add these as free functions at the top of canvas.rs (after the constants):

```rust
/// Result of a body-segment hit test.
struct SegmentHit {
    body_id: String,
    /// Projected point on the segment in world coordinates.
    world_pos: [f64; 2],
    /// Screen position of the projected point.
    screen_pos: Pos2,
    /// Name of the first attachment point forming the segment.
    point_a_name: String,
    /// Name of the second attachment point forming the segment.
    point_b_name: String,
}

/// Project a point onto a line segment and return the closest point on the segment.
/// Returns None if the projection falls outside the segment.
fn project_onto_segment(point: Pos2, seg_a: Pos2, seg_b: Pos2) -> Option<(Pos2, f32)> {
    let ab = seg_b - seg_a;
    let ap = point - seg_a;
    let len_sq = ab.length_sq();
    if len_sq < 1e-10 {
        return None; // degenerate segment
    }
    let t = ab.dot(ap) / len_sq;
    if t < 0.0 || t > 1.0 {
        return None; // projection outside segment
    }
    let proj = seg_a + ab * t;
    let dist = point.distance(proj);
    Some((proj, dist))
}

/// Collected body segment for hit testing.
struct BodySegment {
    screen_a: Pos2,
    screen_b: Pos2,
    world_a: [f64; 2],
    world_b: [f64; 2],
    body_id: String,
    point_a_name: String,
    point_b_name: String,
}

/// Find the nearest body line segment to a screen point.
fn find_nearest_body_segment(
    point: Pos2,
    segments: &[BodySegment],
    max_distance: f32,
) -> Option<SegmentHit> {
    let mut best: Option<(f32, Pos2, [f64; 2], String)> = None;

    for seg in segments {
        if let Some((proj_screen, dist)) = project_onto_segment(point, seg.screen_a, seg.screen_b) {
            if dist <= max_distance {
                if best.as_ref().map_or(true, |(d, _, _, _)| dist < *d) {
                    // Interpolate world position
                    let ab_screen = seg.screen_b - seg.screen_a;
                    let ap_screen = proj_screen - seg.screen_a;
                    let t = if ab_screen.length_sq() > 1e-10 {
                        ap_screen.length() / ab_screen.length()
                    } else {
                        0.0
                    };
                    let world_x = seg.world_a[0] + t as f64 * (seg.world_b[0] - seg.world_a[0]);
                    let world_y = seg.world_a[1] + t as f64 * (seg.world_b[1] - seg.world_a[1]);

                    best = Some((dist, proj_screen, [world_x, world_y], seg.body_id.clone(),
                                 seg.point_a_name.clone(), seg.point_b_name.clone()));
                }
            }
        }
    }

    best.map(|(_, screen_pos, world_pos, body_id, point_a_name, point_b_name)| SegmentHit {
        body_id,
        world_pos,
        screen_pos,
        point_a_name,
        point_b_name,
    })
}
```

- [ ] **Step 2: Collect body segments during rendering pass**

In the body rendering loop (line ~177), collect segments into a `Vec<BodySegment>` alongside the existing `attachment_hit_targets`:

```rust
let mut body_segments: Vec<BodySegment> = Vec::new();
```

Inside the body loop, after computing `point_positions`, collect the segments:

```rust
// Collect segments for hit testing (point_names is the sorted Vec<&String>).
if point_positions.len() >= 2 {
    for i in 0..point_positions.len() - 1 {
        body_segments.push(BodySegment {
            screen_a: point_positions[i].0,
            screen_b: point_positions[i + 1].0,
            world_a: point_positions[i].1,
            world_b: point_positions[i + 1].1,
            body_id: body_id.clone(),
            point_a_name: point_names[i].clone(),
            point_b_name: point_names[i + 1].clone(),
        });
    }
    // Close polygon for 3+ points.
    if point_positions.len() >= 3 {
        body_segments.push(BodySegment {
            screen_a: point_positions.last().unwrap().0,
            screen_b: point_positions[0].0,
            world_a: point_positions.last().unwrap().1,
            world_b: point_positions[0].1,
            body_id: body_id.clone(),
            point_a_name: point_names.last().unwrap().to_string(),
            point_b_name: point_names[0].clone(),
        });
    }
}
```

- [ ] **Step 3: Integrate segment snap into Draw Link**

In the Draw Link drag-start and drag-release handlers, add segment checking at priority 2 (after point snap, before empty space). For both the start and end points, the pattern is:

```rust
let snap_hit = find_nearest_attachment(pos);
let (world_pos, attachment) = if let Some(hit) = snap_hit {
    // Priority 1: snap to existing attachment point
    (hit.world_pos, Some((hit.body_id.clone(), hit.point_name.clone())))
} else if let Some(seg_hit) = find_nearest_body_segment(pos, &body_segments, 8.0) {
    // Priority 2: snap to body segment — create new pivot
    let name = state.next_attachment_point_name(&seg_hit.body_id);
    let [lx, ly] = state.world_to_body_local(&seg_hit.body_id, seg_hit.world_pos[0], seg_hit.world_pos[1]);
    state.add_attachment_point_local_raw(&seg_hit.body_id, &name, lx, ly);
    (seg_hit.world_pos, Some((seg_hit.body_id.clone(), name)))
} else {
    // Priority 3: empty space
    let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
    let (sx, sy) = state.grid.snap_point(wx, wy);
    ([sx, sy], None)
};
```

Apply this pattern to both the drag-start (for the `DrawLinkStart` construction) and the drag-release (for the end point).

- [ ] **Step 4: Add preview indicator for segment snap**

In the Draw Link preview rendering, add a segment-snap indicator:

```rust
// Check for segment snap during preview
let snap_end = find_nearest_attachment(pos);
let segment_snap = if snap_end.is_none() {
    find_nearest_body_segment(pos, &body_segments, 8.0)
} else {
    None
};

if let Some(ref seg_hit) = segment_snap {
    // Draw rotated square (diamond) at projected point
    let center = seg_hit.screen_pos;
    let size = 5.0_f32;
    let diamond = vec![
        Pos2::new(center.x, center.y - size),
        Pos2::new(center.x + size, center.y),
        Pos2::new(center.x, center.y + size),
        Pos2::new(center.x - size, center.y),
    ];
    painter.add(egui::Shape::convex_polygon(
        diamond,
        JOINT_CREATE_HIGHLIGHT,
        Stroke::NONE,
    ));
}
```

- [ ] **Step 5: Verify visually**

Run: `cd linkage-sim-rs && cargo run --bin linkage-gui`
1. Load a 4-bar. Activate Draw Link.
2. Start a drag near the coupler body's edge (not on an endpoint) — should show green diamond
3. Release on empty space — new bar is created, new pivot is added to the coupler body
4. Verify undo undoes the entire operation in one step

- [ ] **Step 6: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat(gui): body-aware Draw Link — snap to body segments

Draw Link now detects clicks near body line segments (priority 2, after
attachment point snap). Auto-creates a pivot on the body at the nearest
segment point. Preview shows green diamond indicator for segment snaps."
```

---

## Task 10: Update Context Menu Hit Testing with Segment Detection

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs` (right-click handler)

The context menu currently uses a proximity-based body_area detection (from Task 4). Now replace it with proper segment hit testing.

- [ ] **Step 1: Update right-click handler to use segment hit detection**

Replace the `body_area` detection in the `secondary_clicked()` handler:

```rust
// Body area: only if no attachment point matched.
// Use segment hit detection for accurate body-area detection.
let body_area = if attachment_point.is_none() {
    find_nearest_body_segment(pos, &body_segments, HIT_RADIUS)
        .map(|hit| hit.body_id)
} else {
    None
};
```

- [ ] **Step 2: Verify visually**

Run: `cd linkage-sim-rs && cargo run --bin linkage-gui`
1. Right-click on a body edge (away from endpoints) — should show "Add Pivot Here" + "Delete Body"
2. Right-click directly on an attachment point — should show "Create Joint" + "Delete Pivot"
3. Right-click on empty canvas — should show "Add Ground Pivot Here" + "Start Body Here"

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat(gui): use segment hit testing for body-area context menu

Right-click context menu now uses accurate segment-distance detection
to distinguish body-area hits from attachment-point hits."
```

---

## Task 11: Integration Testing and Edge Cases

**Files:**
- Modify: `linkage-sim-rs/src/gui/state.rs` (tests)

- [ ] **Step 1: Write integration tests for the full workflow**

```rust
#[test]
fn create_ternary_body_via_add_body_with_points() {
    let mut state = AppState::default();
    let points = vec![
        ("A".to_string(), [0.0, 0.0]),
        ("B".to_string(), [0.1, 0.0]),
        ("C".to_string(), [0.05, 0.08]),
    ];
    state.add_body_with_points(&points);
    let bp = state.blueprint.as_ref().unwrap();
    // Should have ground + the new body
    assert_eq!(bp.bodies.len(), 2);
    let body = bp.bodies.iter()
        .find(|(id, _)| *id != "ground")
        .unwrap().1;
    assert_eq!(body.attachment_points.len(), 3);
    assert!(state.can_undo());
}

#[test]
fn add_pivot_then_joint_creates_ternary_mechanism() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);
    // Add a third pivot to the coupler body
    state.add_attachment_point_to_body("coupler", "P", 0.02, 0.01);
    let bp = state.blueprint.as_ref().unwrap();
    let coupler = bp.bodies.get("coupler").unwrap();
    assert_eq!(coupler.attachment_points.len(), 3); // B, C, P
}

#[test]
fn remove_attachment_point_with_min_two_points_survives() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);
    // Crank has A and B. Removing B should leave just A.
    state.remove_attachment_point("crank", "B");
    let bp = state.blueprint.as_ref().unwrap();
    let crank = bp.bodies.get("crank").unwrap();
    assert_eq!(crank.attachment_points.len(), 1);
    assert!(crank.attachment_points.contains_key("A"));
}

#[test]
fn compound_draw_link_is_single_undo_step() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::FourBar);

    let bp_before = state.blueprint.as_ref().unwrap().clone();
    let bodies_before = bp_before.bodies.len();
    let joints_before = bp_before.joints.len();

    // Simulate a compound Draw Link operation (one push_undo, N raw mutations, one rebuild)
    state.push_undo();
    state.add_ground_pivot_raw("PX", 0.1, 0.1);
    let new_body_id = state.next_body_id();
    let points = vec![
        ("A".to_string(), [0.1, 0.1]),
        ("B".to_string(), [0.2, 0.1]),
    ];
    state.add_body_with_points_raw(&new_body_id, &points);
    state.add_revolute_joint_raw("ground", "PX", &new_body_id, "A");
    state.rebuild();

    // Verify the operation added bodies/joints
    let bp_after = state.blueprint.as_ref().unwrap();
    assert!(bp_after.bodies.len() > bodies_before);
    assert!(bp_after.joints.len() > joints_before);

    // One undo should restore the entire previous state
    assert!(state.can_undo());
    state.undo();
    let bp_undone = state.blueprint.as_ref().unwrap();
    assert_eq!(bp_undone.bodies.len(), bodies_before);
    assert_eq!(bp_undone.joints.len(), joints_before);
}
```

- [ ] **Step 2: Run all tests**

Run: `cd linkage-sim-rs && cargo test`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/state.rs
git commit -m "test(gui): integration tests for multi-pivot body editing

Tests cover ternary body creation, add-pivot-then-joint workflow,
remove_attachment_point cascading, and compound undo batching."
```

---

## Task 12: Update README and RUST_MIGRATION.md

**Files:**
- Modify: `README.md`
- Modify: `RUST_MIGRATION.md`

- [ ] **Step 1: Update README interactive editor section**

Update the "Interactive editor (shipped)" section to reflect the new capabilities. Add bullet points for:
- Multi-point body creation via + Body tool (click-to-place)
- Add Pivot Here context menu for promoting bodies to ternary/quaternary
- Body-aware Draw Link segment snapping
- Modes-only toolbar (Select, Draw Link, + Body, + Ground)

- [ ] **Step 2: Update RUST_MIGRATION.md Phase 5 progress**

Update the "Remaining Phase 5 work" section and the port sequencing step 10.

- [ ] **Step 3: Commit**

```bash
git add README.md RUST_MIGRATION.md
git commit -m "docs: update for multi-pivot body editor

Document new + Body tool, Add Pivot context menu, body-aware Draw Link,
and toolbar reorganization."
```

---

## Summary

| Task | Description | Files | Dependencies |
|------|-------------|-------|-------------|
| 1 | Utility methods | state.rs | None |
| 2 | Raw blueprint helpers + undo refactor | state.rs, canvas.rs | Task 1 |
| 3 | add_attachment_point_to_body, remove_attachment_point | state.rs | Task 2 |
| 4 | EditorTool::AddBody, ContextMenuTarget refactor | state.rs, canvas.rs | Task 3 |
| 5 | Toolbar cleanup | mod.rs, property_panel.rs | Task 4 |
| 6 | Close polygon rendering | canvas.rs | None (can parallel with 1-5) |
| 7 | Add Body tool interaction | canvas.rs, state.rs | Task 4 |
| 8 | Create Joint two-click flow | canvas.rs | Task 4 |
| 9 | Body-aware Draw Link segment snap | canvas.rs | Task 6 |
| 10 | Context menu segment hit testing | canvas.rs | Task 9 |
| 11 | Integration tests | state.rs | Task 9 |
| 12 | Documentation | README.md, RUST_MIGRATION.md | Task 11 |
