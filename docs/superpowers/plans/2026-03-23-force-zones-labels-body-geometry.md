# Force Zones, Body Geometry, Labels & Crank Angle Limits — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add body geometry visualization, spatial force zones, element labels/tooltips, crank angle limits, and a Parallelogram Press sample mechanism.

**Architecture:** ForceZone is a new ForceElement variant that computes rectangle-vs-rectangle overlap using standalone polygon geometry utilities, then applies a proportional force at the overlap centroid via existing `point_force_to_q()`. BodyGeometry and labels are new fields on Body/BodyJson. SweepConfig lives on MechanismJson and controls the sweep loop range.

**Tech Stack:** Rust, nalgebra, serde, egui/eframe

**Spec:** `docs/superpowers/specs/2026-03-23-force-zones-labels-body-geometry-design.md`

---

## File Structure

### New Files
- `linkage-sim-rs/src/geometry/mod.rs` — polygon clip, area, centroid utilities
- `linkage-sim-rs/tests/geometry_tests.rs` — integration tests for polygon utilities
- `linkage-sim-rs/tests/force_zone_tests.rs` — integration tests for ForceZone force element

### Modified Files
- `linkage-sim-rs/src/lib.rs` — add `pub mod geometry;`
- `linkage-sim-rs/src/core/body.rs` — add `BodyGeometry`, `label` field, `geometry` field
- `linkage-sim-rs/src/forces/elements.rs` — add `ForceZoneElement` struct, `ForceZone` variant
- `linkage-sim-rs/src/io/serialization.rs` — add `SweepConfig`, update `BodyJson`, bump schema version
- `linkage-sim-rs/src/gui/canvas.rs` — render body geometry, force zones, labels, tooltips, overlap highlights
- `linkage-sim-rs/src/gui/state.rs` — add `show_labels` toggle, sweep config state fields
- `linkage-sim-rs/src/gui/property_panel.rs` — body geometry editing, label editing
- `linkage-sim-rs/src/gui/force_toolbar.rs` — ForceZone creation button + drag workflow
- `linkage-sim-rs/src/gui/samples.rs` — add ParallelogramPress sample
- `linkage-sim-rs/src/gui/sweep.rs` — respect SweepConfig angle range
- `linkage-sim-rs/src/gui/input_panel.rs` — sweep range min/max controls

---

## Task 1: Polygon Geometry Utilities

**Files:**
- Create: `linkage-sim-rs/src/geometry/mod.rs`
- Modify: `linkage-sim-rs/src/lib.rs`
- Create: `linkage-sim-rs/tests/geometry_tests.rs`

Three standalone pure-math functions: Sutherland-Hodgman polygon clipping against an axis-aligned rectangle, shoelace area, and centroid computation.

- [ ] **Step 1: Write failing tests for polygon_area**

Create `linkage-sim-rs/tests/geometry_tests.rs`:

```rust
use nalgebra::Vector2;

// Unit square area = 1.0
#[test]
fn polygon_area_unit_square() {
    let poly = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(1.0, 1.0),
        Vector2::new(0.0, 1.0),
    ];
    let area = linkage_sim::geometry::polygon_area(&poly);
    assert!((area - 1.0).abs() < 1e-12);
}

// Triangle area = 0.5
#[test]
fn polygon_area_triangle() {
    let poly = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(0.0, 1.0),
    ];
    let area = linkage_sim::geometry::polygon_area(&poly);
    assert!((area - 0.5).abs() < 1e-12);
}

// Empty/degenerate polygon
#[test]
fn polygon_area_degenerate() {
    assert!(linkage_sim::geometry::polygon_area(&[]).abs() < 1e-12);
    let line = vec![Vector2::new(0.0, 0.0), Vector2::new(1.0, 0.0)];
    assert!(linkage_sim::geometry::polygon_area(&line).abs() < 1e-12);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd linkage-sim-rs && cargo test --test geometry_tests -- --nocapture`
Expected: compilation error — module `geometry` doesn't exist

- [ ] **Step 3: Implement polygon_area and polygon_centroid**

Create `linkage-sim-rs/src/geometry/mod.rs`:

```rust
//! Polygon geometry utilities for overlap computations.
//!
//! Standalone pure-math functions: Sutherland-Hodgman clipping,
//! shoelace area, and centroid. Used by ForceZone.

use nalgebra::Vector2;

/// Signed area of a simple polygon using the shoelace formula.
/// Returns positive area for counter-clockwise vertices.
/// Returns absolute area regardless of winding.
pub fn polygon_area(vertices: &[Vector2<f64>]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
    }
    sum.abs() * 0.5
}

/// Centroid of a simple polygon.
/// Returns the origin if the polygon is degenerate (area ≈ 0).
pub fn polygon_centroid(vertices: &[Vector2<f64>]) -> Vector2<f64> {
    let n = vertices.len();
    if n < 3 {
        return Vector2::zeros();
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut signed_area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let cross = vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
        cx += (vertices[i].x + vertices[j].x) * cross;
        cy += (vertices[i].y + vertices[j].y) * cross;
        signed_area += cross;
    }
    if signed_area.abs() < 1e-15 {
        return Vector2::zeros();
    }
    signed_area *= 0.5;
    cx /= 6.0 * signed_area;
    cy /= 6.0 * signed_area;
    Vector2::new(cx, cy)
}
```

Add to `linkage-sim-rs/src/lib.rs`:
```rust
pub mod geometry;
```

- [ ] **Step 4: Run area/centroid tests to verify they pass**

Run: `cd linkage-sim-rs && cargo test --test geometry_tests -- --nocapture`
Expected: PASS for all 3 tests

- [ ] **Step 5: Write failing tests for polygon_clip**

Add to `linkage-sim-rs/tests/geometry_tests.rs`:

```rust
// Full overlap: body rectangle fully inside zone → clipped polygon = body
#[test]
fn polygon_clip_full_overlap() {
    let body = vec![
        Vector2::new(0.2, 0.2),
        Vector2::new(0.8, 0.2),
        Vector2::new(0.8, 0.8),
        Vector2::new(0.2, 0.8),
    ];
    let zone_min = Vector2::new(0.0, 0.0);
    let zone_max = Vector2::new(1.0, 1.0);
    let clipped = linkage_sim::geometry::clip_polygon_to_aabb(&body, &zone_min, &zone_max);
    let area = linkage_sim::geometry::polygon_area(&clipped);
    assert!((area - 0.36).abs() < 1e-10); // 0.6 * 0.6
}

// No overlap: body is entirely outside zone
#[test]
fn polygon_clip_no_overlap() {
    let body = vec![
        Vector2::new(2.0, 2.0),
        Vector2::new(3.0, 2.0),
        Vector2::new(3.0, 3.0),
        Vector2::new(2.0, 3.0),
    ];
    let zone_min = Vector2::new(0.0, 0.0);
    let zone_max = Vector2::new(1.0, 1.0);
    let clipped = linkage_sim::geometry::clip_polygon_to_aabb(&body, &zone_min, &zone_max);
    assert!(linkage_sim::geometry::polygon_area(&clipped).abs() < 1e-12);
}

// Partial overlap: half of body inside zone
#[test]
fn polygon_clip_partial_overlap() {
    let body = vec![
        Vector2::new(0.5, 0.0),
        Vector2::new(1.5, 0.0),
        Vector2::new(1.5, 1.0),
        Vector2::new(0.5, 1.0),
    ];
    let zone_min = Vector2::new(0.0, 0.0);
    let zone_max = Vector2::new(1.0, 1.0);
    let clipped = linkage_sim::geometry::clip_polygon_to_aabb(&body, &zone_min, &zone_max);
    let area = linkage_sim::geometry::polygon_area(&clipped);
    assert!((area - 0.5).abs() < 1e-10); // 0.5 * 1.0
}

// Rotated rectangle partial overlap
#[test]
fn polygon_clip_rotated_partial() {
    // 45-degree rotated square, centered at origin, side length sqrt(2)
    // so it spans from -1 to 1 on both axes (diamond shape)
    let body = vec![
        Vector2::new(1.0, 0.0),
        Vector2::new(0.0, 1.0),
        Vector2::new(-1.0, 0.0),
        Vector2::new(0.0, -1.0),
    ];
    // Clip to first quadrant
    let zone_min = Vector2::new(0.0, 0.0);
    let zone_max = Vector2::new(2.0, 2.0);
    let clipped = linkage_sim::geometry::clip_polygon_to_aabb(&body, &zone_min, &zone_max);
    let area = linkage_sim::geometry::polygon_area(&clipped);
    // Diamond has total area 2.0; first quadrant is exactly 0.5
    assert!((area - 0.5).abs() < 1e-10);
}
```

- [ ] **Step 6: Implement clip_polygon_to_aabb (Sutherland-Hodgman)**

Add to `linkage-sim-rs/src/geometry/mod.rs`:

```rust
/// Clip a convex polygon against an axis-aligned bounding box (AABB).
///
/// Uses the Sutherland-Hodgman algorithm: clip against each of the 4 AABB
/// edges in sequence. Returns the clipped polygon vertices (may be empty
/// if there is no overlap).
pub fn clip_polygon_to_aabb(
    polygon: &[Vector2<f64>],
    aabb_min: &Vector2<f64>,
    aabb_max: &Vector2<f64>,
) -> Vec<Vector2<f64>> {
    if polygon.is_empty() {
        return vec![];
    }

    let mut output = polygon.to_vec();

    // Clip against each edge: left, right, bottom, top
    // Each edge is defined by a test function (inside?) and an intersection function.
    let edges: [(fn(&Vector2<f64>, f64) -> bool, fn(&Vector2<f64>, &Vector2<f64>, f64) -> Vector2<f64>, f64); 4] = [
        (|p, val| p.x >= val, intersect_x, aabb_min.x),  // left
        (|p, val| p.x <= val, intersect_x, aabb_max.x),  // right
        (|p, val| p.y >= val, intersect_y, aabb_min.y),  // bottom
        (|p, val| p.y <= val, intersect_y, aabb_max.y),  // top
    ];

    for (inside, intersect, val) in &edges {
        if output.is_empty() {
            return vec![];
        }
        let input = output;
        output = Vec::new();

        let n = input.len();
        for i in 0..n {
            let current = &input[i];
            let next = &input[(i + 1) % n];
            let cur_inside = inside(current, *val);
            let nxt_inside = inside(next, *val);

            if cur_inside {
                output.push(*current);
                if !nxt_inside {
                    output.push(intersect(current, next, *val));
                }
            } else if nxt_inside {
                output.push(intersect(current, next, *val));
            }
        }
    }

    output
}

/// Line-edge intersection at x = val.
fn intersect_x(a: &Vector2<f64>, b: &Vector2<f64>, val: f64) -> Vector2<f64> {
    let dx = b.x - a.x;
    if dx.abs() < 1e-15 {
        return Vector2::new(val, a.y);
    }
    let t = (val - a.x) / dx;
    Vector2::new(val, a.y + t * (b.y - a.y))
}

/// Line-edge intersection at y = val.
fn intersect_y(a: &Vector2<f64>, b: &Vector2<f64>, val: f64) -> Vector2<f64> {
    let dy = b.y - a.y;
    if dy.abs() < 1e-15 {
        return Vector2::new(a.x, val);
    }
    let t = (val - a.y) / dy;
    Vector2::new(a.x + t * (b.x - a.x), val)
}
```

- [ ] **Step 7: Run all geometry tests**

Run: `cd linkage-sim-rs && cargo test --test geometry_tests -- --nocapture`
Expected: all 7 tests PASS

- [ ] **Step 8: Write centroid test and verify**

Add to `linkage-sim-rs/tests/geometry_tests.rs`:

```rust
#[test]
fn polygon_centroid_unit_square() {
    let poly = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(1.0, 1.0),
        Vector2::new(0.0, 1.0),
    ];
    let c = linkage_sim::geometry::polygon_centroid(&poly);
    assert!((c.x - 0.5).abs() < 1e-12);
    assert!((c.y - 0.5).abs() < 1e-12);
}

#[test]
fn polygon_centroid_of_clipped_region() {
    // Body half-overlapping zone: clip, then centroid should be at center of overlap
    let body = vec![
        Vector2::new(0.5, 0.0),
        Vector2::new(1.5, 0.0),
        Vector2::new(1.5, 1.0),
        Vector2::new(0.5, 1.0),
    ];
    let clipped = linkage_sim::geometry::clip_polygon_to_aabb(
        &body,
        &Vector2::new(0.0, 0.0),
        &Vector2::new(1.0, 1.0),
    );
    let c = linkage_sim::geometry::polygon_centroid(&clipped);
    assert!((c.x - 0.75).abs() < 1e-10);
    assert!((c.y - 0.5).abs() < 1e-10);
}
```

Run: `cd linkage-sim-rs && cargo test --test geometry_tests -- --nocapture`
Expected: all 9 tests PASS

- [ ] **Step 9: Add helper to transform body rectangle to world-frame polygon**

Add to `linkage-sim-rs/src/geometry/mod.rs`:

```rust
/// Transform a body-local rectangle to world-frame polygon vertices.
///
/// Given body position (x, y) and angle θ, plus rectangle dimensions
/// (width, height) and offset from body origin, returns the 4 corners
/// in world coordinates (counter-clockwise).
pub fn body_rect_to_world(
    body_x: f64,
    body_y: f64,
    body_theta: f64,
    width: f64,
    height: f64,
    offset: &Vector2<f64>,
) -> [Vector2<f64>; 4] {
    let cos_t = body_theta.cos();
    let sin_t = body_theta.sin();

    // Rectangle corners in body-local frame, centered on offset
    let half_w = width / 2.0;
    let half_h = height / 2.0;
    let local_corners = [
        Vector2::new(offset.x - half_w, offset.y - half_h),
        Vector2::new(offset.x + half_w, offset.y - half_h),
        Vector2::new(offset.x + half_w, offset.y + half_h),
        Vector2::new(offset.x - half_w, offset.y + half_h),
    ];

    let mut world_corners = [Vector2::zeros(); 4];
    for (i, lc) in local_corners.iter().enumerate() {
        world_corners[i] = Vector2::new(
            body_x + cos_t * lc.x - sin_t * lc.y,
            body_y + sin_t * lc.x + cos_t * lc.y,
        );
    }
    world_corners
}
```

- [ ] **Step 10: Test body_rect_to_world**

Add to test file:

```rust
use std::f64::consts::FRAC_PI_2;

#[test]
fn body_rect_to_world_no_rotation() {
    let corners = linkage_sim::geometry::body_rect_to_world(
        1.0, 2.0, 0.0, 0.4, 0.2, &Vector2::new(0.0, 0.0),
    );
    // Centered at (1,2), 0.4 wide, 0.2 tall
    assert!((corners[0].x - 0.8).abs() < 1e-12);
    assert!((corners[0].y - 1.9).abs() < 1e-12);
    assert!((corners[2].x - 1.2).abs() < 1e-12);
    assert!((corners[2].y - 2.1).abs() < 1e-12);
}

#[test]
fn body_rect_to_world_90deg() {
    let corners = linkage_sim::geometry::body_rect_to_world(
        0.0, 0.0, FRAC_PI_2, 1.0, 0.5, &Vector2::new(0.0, 0.0),
    );
    // Rotated 90° CCW: width along y, height along -x
    // Corner 0 was (-0.5, -0.25) local → rotated: (0.25, -0.5) world
    assert!((corners[0].x - 0.25).abs() < 1e-10);
    assert!((corners[0].y - (-0.5)).abs() < 1e-10);
}
```

Run: `cd linkage-sim-rs && cargo test --test geometry_tests -- --nocapture`
Expected: all 11 tests PASS

- [ ] **Step 11: Commit**

```bash
git add linkage-sim-rs/src/geometry/mod.rs linkage-sim-rs/src/lib.rs linkage-sim-rs/tests/geometry_tests.rs
git commit -m "feat: add polygon geometry utilities (clip, area, centroid, body_rect_to_world)"
```

---

## Task 2: BodyGeometry and Labels on Body

**Files:**
- Modify: `linkage-sim-rs/src/core/body.rs:18-47`
- Modify: `linkage-sim-rs/src/io/serialization.rs:101-116`

- [ ] **Step 1: Write failing test for BodyGeometry validation**

Add to `linkage-sim-rs/tests/geometry_tests.rs`:

```rust
use linkage_sim::core::body::BodyGeometry;

#[test]
fn body_geometry_valid() {
    let geo = BodyGeometry::new(0.06, 0.015, Vector2::new(0.0, 0.0));
    assert!(geo.is_ok());
    let geo = geo.unwrap();
    assert!((geo.width - 0.06).abs() < 1e-12);
    assert!((geo.height - 0.015).abs() < 1e-12);
}

#[test]
fn body_geometry_rejects_zero_dimension() {
    assert!(BodyGeometry::new(0.0, 0.015, Vector2::new(0.0, 0.0)).is_err());
    assert!(BodyGeometry::new(0.06, 0.0, Vector2::new(0.0, 0.0)).is_err());
    assert!(BodyGeometry::new(-0.01, 0.015, Vector2::new(0.0, 0.0)).is_err());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd linkage-sim-rs && cargo test --test geometry_tests body_geometry -- --nocapture`
Expected: FAIL — `BodyGeometry` doesn't exist

- [ ] **Step 3: Implement BodyGeometry struct and add fields to Body**

Add to `linkage-sim-rs/src/core/body.rs` (after the `use` statements, before `Body` struct):

```rust
/// Visual rectangular geometry attached to a body.
///
/// Used for force zone overlap computation and canvas rendering.
/// Dimensions are in body-local frame, centered on `offset` from body origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyGeometry {
    pub width: f64,            // meters, extent along body's local x-axis
    pub height: f64,           // meters, extent along body's local y-axis
    pub offset: Vector2<f64>,  // local-frame offset from body origin
}

impl BodyGeometry {
    /// Create a new body geometry. Width and height must be > 0.
    pub fn new(width: f64, height: f64, offset: Vector2<f64>) -> Result<Self, BodyError> {
        if width <= 0.0 || height <= 0.0 {
            return Err(BodyError::InvalidGeometry {
                reason: format!(
                    "width ({}) and height ({}) must both be > 0",
                    width, height
                ),
            });
        }
        Ok(Self { width, height, offset })
    }

    /// Total area of the rectangle (m²).
    pub fn area(&self) -> f64 {
        self.width * self.height
    }
}
```

Add new `BodyError` variant in the existing enum:
```rust
    #[error("Invalid body geometry: {reason}")]
    InvalidGeometry { reason: String },
```

Add two fields to the `Body` struct:
```rust
    /// User-editable display label.
    pub label: String,
    /// Optional visual geometry for rendering and force zone overlap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<BodyGeometry>,
```

Update `Body::new()` to initialize: `label: id.to_string()`, `geometry: None`.

Update `make_bar()` and `make_ground()` to set `label` from `id`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd linkage-sim-rs && cargo test --test geometry_tests body_geometry -- --nocapture`
Expected: PASS

- [ ] **Step 5: Update BodyJson serialization**

In `linkage-sim-rs/src/io/serialization.rs`, add to `BodyJson`:

```rust
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<crate::core::body::BodyGeometry>,
```

Update `body_to_json()` to include the new fields.
Update `mechanism_from_json()` to read label (defaulting to body ID if absent) and geometry.

- [ ] **Step 6: Run full test suite to verify nothing broke**

Run: `cd linkage-sim-rs && cargo test`
Expected: all existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/core/body.rs linkage-sim-rs/src/io/serialization.rs linkage-sim-rs/tests/geometry_tests.rs
git commit -m "feat: add BodyGeometry struct and label field to Body"
```

---

## Task 3: ForceZone ForceElement Variant

**Files:**
- Modify: `linkage-sim-rs/src/forces/elements.rs:424-492`
- Create: `linkage-sim-rs/tests/force_zone_tests.rs`

- [ ] **Step 1: Write failing test for ForceZone evaluation**

Create `linkage-sim-rs/tests/force_zone_tests.rs`:

```rust
use nalgebra::{DVector, Vector2};
use std::collections::HashMap;
use std::f64::consts::PI;

use linkage_sim::core::body::{make_bar, make_ground, Body, BodyGeometry};
use linkage_sim::core::mechanism::Mechanism;
use linkage_sim::forces::elements::{ForceElement, ForceZoneElement};

/// Build a minimal mechanism with one body that has geometry,
/// positioned so it partially overlaps a force zone.
#[test]
fn force_zone_no_overlap_produces_zero_force() {
    let (mech, q) = build_test_mechanism();
    let state = mech.state();
    let bodies = mech.bodies();

    // Zone far away from the body
    let fz = ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [10.0, 10.0],
        zone_max: [11.0, 11.0],
        force: [0.0, -500.0],
        label: None,
    };

    let q_dot = DVector::zeros(q.len());
    let contribution = ForceElement::ForceZone(fz).evaluate(state, mech.bodies(), &q, &q_dot, 0.0);
    // All entries should be zero (no overlap with far-away zone)
    assert!(contribution.iter().all(|v| v.abs() < 1e-12));
}

#[test]
fn force_zone_full_overlap_applies_full_force() {
    let (mech, q) = build_test_mechanism();
    let state = mech.state();
    let bodies = mech.bodies();

    // Zone that fully encloses the body geometry
    let fz = ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [-1.0, -1.0],
        zone_max: [1.0, 1.0],
        force: [0.0, -500.0],
        label: None,
    };

    let q_dot = DVector::zeros(q.len());
    let contribution = ForceElement::ForceZone(fz).evaluate(state, bodies, &q, &q_dot, 0.0);
    // Should have non-zero entries — full force applied
    assert!(contribution.amax() > 0.0 || contribution.amin() < 0.0);
}

fn build_test_mechanism() -> (Mechanism, DVector<f64>) {
    // Minimal: ground + one bar with geometry, connected by revolute + driver.
    // This gives a fully constrained body we can reason about.
    let ground = make_ground(&[("O", 0.0, 0.0)]);
    let mut bar = make_bar("bar", "A", "B", 0.1, 0.0, 0.0);
    bar.geometry = Some(BodyGeometry::new(0.06, 0.015, Vector2::new(0.05, 0.0)).unwrap());

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(bar).unwrap();
    mech.add_revolute_joint("J1", "ground", "O", "bar", "A").unwrap();
    // Need a driver to fully constrain
    mech.add_constant_speed_driver("D1", "ground", "bar", 0.0, 0.0).unwrap();
    mech.build().unwrap();

    let q = DVector::zeros(mech.state().n_coords());
    (mech, q)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd linkage-sim-rs && cargo test --test force_zone_tests -- --nocapture`
Expected: FAIL — `ForceZoneElement` doesn't exist

- [ ] **Step 3: Implement ForceZoneElement struct**

Add to `linkage-sim-rs/src/forces/elements.rs`, after the existing element structs:

```rust
/// A spatial force zone: applies a constant distributed force to a body
/// proportional to the overlap area between the body's geometry and the zone.
///
/// The zone is an axis-aligned rectangle in world space. The body must have
/// `BodyGeometry` set. Force is applied at the centroid of the overlap region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceZoneElement {
    /// ID of the body whose geometry is tested for overlap.
    pub body_id: String,
    /// World-space bottom-left corner of the zone (meters).
    pub zone_min: [f64; 2],
    /// World-space top-right corner of the zone (meters).
    pub zone_max: [f64; 2],
    /// Constant force vector applied at full overlap (Newtons). E.g., [0, -500].
    pub force: [f64; 2],
    /// Optional display label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}
```

Add `ForceZone(ForceZoneElement)` variant to the `ForceElement` enum.

Add the match arm in `ForceElement::evaluate()`:
```rust
ForceElement::ForceZone(fz) => evaluate_force_zone(fz, state, bodies, q),
```

And in `evaluate_compiled()` (line 501) — force zones have no time modulation, so ignore the modulation_factor and delegate:
```rust
ForceElement::ForceZone(fz) => evaluate_force_zone(fz, state, bodies, q),
```

- [ ] **Step 4: Implement evaluate_force_zone**

Add to `linkage-sim-rs/src/forces/elements.rs`:

```rust
fn evaluate_force_zone(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> DVector<f64> {
    use crate::geometry::{body_rect_to_world, clip_polygon_to_aabb, polygon_area, polygon_centroid};

    let n = state.n_coords();
    let body = match bodies.get(&fz.body_id) {
        Some(b) => b,
        None => return DVector::zeros(n),
    };
    let geo = match &body.geometry {
        Some(g) => g,
        None => return DVector::zeros(n),
    };

    // Get body position from state vector via BodyIndex
    let bi = match state.get_index(&fz.body_id) {
        Ok(idx) => idx,
        Err(_) => return DVector::zeros(n),
    };
    let bx = q[bi.x_idx()];
    let by = q[bi.y_idx()];
    let btheta = q[bi.theta_idx()];

    // Transform body rectangle to world frame
    let corners = body_rect_to_world(bx, by, btheta, geo.width, geo.height, &geo.offset);

    // Clip against zone AABB
    let zone_min = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
    let zone_max = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
    let clipped = clip_polygon_to_aabb(&corners, &zone_min, &zone_max);

    let overlap_area = polygon_area(&clipped);
    let body_area = geo.area();

    if overlap_area < 1e-15 || body_area < 1e-15 {
        return DVector::zeros(n);
    }

    let ratio = (overlap_area / body_area).min(1.0);
    let force_global = Vector2::new(fz.force[0] * ratio, fz.force[1] * ratio);

    // Apply force at the centroid of the overlap region
    let centroid_world = polygon_centroid(&clipped);

    // Convert world centroid to body-local point for point_force_to_q
    let cos_t = btheta.cos();
    let sin_t = btheta.sin();
    let dx = centroid_world.x - bx;
    let dy = centroid_world.y - by;
    let local_point = Vector2::new(
        cos_t * dx + sin_t * dy,
        -sin_t * dx + cos_t * dy,
    );

    point_force_to_q(state, &fz.body_id, &local_point, &force_global, q)
}
```

- [ ] **Step 5: Run force zone tests**

Run: `cd linkage-sim-rs && cargo test --test force_zone_tests -- --nocapture`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `cd linkage-sim-rs && cargo test`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add linkage-sim-rs/src/forces/elements.rs linkage-sim-rs/tests/force_zone_tests.rs
git commit -m "feat: add ForceZone force element variant with overlap-proportional force"
```

---

## Task 4: SweepConfig and Angle Limits

**Files:**
- Modify: `linkage-sim-rs/src/io/serialization.rs:36-52`
- Modify: `linkage-sim-rs/src/gui/sweep.rs:148`
- Modify: `linkage-sim-rs/src/gui/state.rs`

- [ ] **Step 1: Add SweepConfig to serialization**

In `linkage-sim-rs/src/io/serialization.rs`, add struct:

```rust
/// Configuration for sweep analysis angle range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepConfig {
    /// Minimum sweep angle in radians.
    pub angle_min: f64,
    /// Maximum sweep angle in radians.
    pub angle_max: f64,
    /// When false, full 360° sweep regardless of min/max.
    pub enabled: bool,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            angle_min: 0.0,
            angle_max: 2.0 * std::f64::consts::PI,
            enabled: false,
        }
    }
}
```

Add to `MechanismJson`:
```rust
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sweep_config: Option<SweepConfig>,
```

Update `mechanism_to_json()` to include sweep_config (pass through from AppState or set None if not configured).

- [ ] **Step 2: Update sweep loop to respect SweepConfig**

In `linkage-sim-rs/src/gui/sweep.rs`, modify `compute_sweep_data()` signature to accept an optional `SweepConfig`. Replace the hardcoded `for i in 0..=360` with:

```rust
    let (start_deg, end_deg) = match sweep_config {
        Some(cfg) if cfg.enabled => {
            (cfg.angle_min.to_degrees(), cfg.angle_max.to_degrees())
        }
        _ => (0.0, 360.0),
    };

    let step = 1.0_f64; // 1-degree steps
    let num_steps = ((end_deg - start_deg) / step).round() as i32;
    for i in 0..=num_steps.max(0) {
        let angle_deg = start_deg + i as f64 * step;
```

Update all callers of `compute_sweep_data()` to pass the sweep config.

- [ ] **Step 3: Add sweep config fields to AppState**

In `linkage-sim-rs/src/gui/state.rs`, add to `AppState`:

```rust
    /// Sweep angle range configuration.
    pub sweep_angle_min_deg: f64,  // degrees for GUI display
    pub sweep_angle_max_deg: f64,
    pub sweep_range_enabled: bool,
```

Initialize defaults: `0.0`, `360.0`, `false`.

- [ ] **Step 4: Run full test suite**

Run: `cd linkage-sim-rs && cargo test`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/io/serialization.rs linkage-sim-rs/src/gui/sweep.rs linkage-sim-rs/src/gui/state.rs
git commit -m "feat: add SweepConfig for crank angle limits"
```

---

## Task 5: Joint Labels in Serialization

**Files:**
- Modify: `linkage-sim-rs/src/io/serialization.rs:122-166`

- [ ] **Step 1: Add label field to JointJson variants**

Add `#[serde(default, skip_serializing_if = "Option::is_none")] label: Option<String>` to each `JointJson` variant (Revolute, Fixed, Prismatic, CamFollower, RevoluteDriver).

- [ ] **Step 2: Update joint_to_json() to include labels**

Default to `None` — labels are auto-generated at rendering time when `None`.

- [ ] **Step 3: Run full test suite**

Run: `cd linkage-sim-rs && cargo test`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/io/serialization.rs
git commit -m "feat: add optional label field to JointJson"
```

---

## Task 6: Canvas Rendering — Body Geometry

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs`

- [ ] **Step 1: Add body geometry rendering**

In the body drawing section of `draw_canvas()`, after drawing the link bar, add rendering for bodies with geometry:

```rust
// Draw body geometry rectangles
if let Some(ref geo) = body.geometry {
    let corners = linkage_sim::geometry::body_rect_to_world(
        bx, by, btheta, geo.width, geo.height, &geo.offset,
    );
    let screen_corners: Vec<egui::Pos2> = corners.iter()
        .map(|c| world_to_screen(c.x, c.y, &view))
        .collect();

    let fill = egui::Color32::from_rgba_premultiplied(64, 42, 0, 64); // orange tint
    let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 165, 0));

    let shape = egui::Shape::convex_polygon(
        screen_corners,
        fill,
        stroke,
    );
    painter.add(shape);
}
```

- [ ] **Step 2: Verify visually with cargo run**

Run: `cd linkage-sim-rs && cargo run --bin linkage_gui`
Load a mechanism, add geometry to a body via the property panel (Task 10), verify rectangle renders.
(Note: property panel not implemented yet — can test via hardcoded geometry in a sample.)

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat: render body geometry rectangles on canvas"
```

---

## Task 7: Canvas Rendering — Force Zones

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs`

- [ ] **Step 1: Add force zone rendering**

After body rendering, iterate over force elements and draw ForceZone elements:

```rust
// Draw force zones
for force in mechanism.forces() {
    if let ForceElement::ForceZone(ref fz) = force {
        let min_screen = world_to_screen(fz.zone_min[0], fz.zone_min[1], &view);
        let max_screen = world_to_screen(fz.zone_max[0], fz.zone_max[1], &view);
        let rect = egui::Rect::from_two_pos(min_screen, max_screen);

        // Zone fill + border. Note: egui doesn't support dashed strokes natively.
        // Draw the fill with painter.rect(), then draw 4 dashed line segments manually
        // for the border (use a helper that draws segments with gaps).
        let zone_fill = egui::Color32::from_rgba_premultiplied(255, 80, 80, 30);
        let zone_stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 80, 80));
        painter.rect_filled(rect, 0.0, zone_fill);
        draw_dashed_rect(&painter, rect, zone_stroke, 8.0, 4.0);

        // Force arrows inside zone
        draw_force_zone_arrows(&painter, &rect, fz, &view);

        // Label above zone
        let label_text = fz.label.as_deref().unwrap_or("Force Zone");
        let label_pos = egui::pos2(rect.center().x, rect.min.y.min(rect.max.y) - 14.0);
        painter.text(
            label_pos,
            egui::Align2::CENTER_BOTTOM,
            label_text,
            egui::FontId::monospace(10.0),
            egui::Color32::from_rgb(255, 80, 80),
        );
    }
}
```

- [ ] **Step 2: Add overlap highlight rendering**

Compute and render the overlap polygon when sweep data exists or animation is running:

```rust
// Draw overlap highlight for active force zones
if let ForceElement::ForceZone(ref fz) = force {
    if let Some(ref body) = bodies.get(&fz.body_id) {
        if let Some(ref geo) = body.geometry {
            // Get body state from current q
            let corners = geometry::body_rect_to_world(bx, by, btheta, geo.width, geo.height, &geo.offset);
            let zone_min = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
            let zone_max = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
            let clipped = geometry::clip_polygon_to_aabb(&corners, &zone_min, &zone_max);

            if clipped.len() >= 3 {
                let screen_pts: Vec<egui::Pos2> = clipped.iter()
                    .map(|c| world_to_screen(c.x, c.y, &view))
                    .collect();
                let overlap_fill = egui::Color32::from_rgba_premultiplied(255, 200, 0, 50);
                let overlap_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 204, 0));
                painter.add(egui::Shape::convex_polygon(screen_pts, overlap_fill, overlap_stroke));
            }
        }
    }
}
```

- [ ] **Step 3: Implement draw_force_zone_arrows helper**

Draw 3-5 evenly spaced arrows inside the zone rectangle, pointing in the force direction:

```rust
fn draw_force_zone_arrows(
    painter: &egui::Painter,
    rect: &egui::Rect,
    fz: &ForceZoneElement,
    view: &ViewTransform,
) {
    let force_dir = Vector2::new(fz.force[0], fz.force[1]);
    if force_dir.norm() < 1e-12 {
        return;
    }
    let dir_normalized = force_dir.normalize();
    let arrow_color = egui::Color32::from_rgb(255, 80, 80);

    let num_arrows = 3;
    let rect_sorted = egui::Rect::from_min_max(
        egui::pos2(rect.min.x.min(rect.max.x), rect.min.y.min(rect.max.y)),
        egui::pos2(rect.min.x.max(rect.max.x), rect.min.y.max(rect.max.y)),
    );

    for i in 0..num_arrows {
        let frac = (i as f32 + 1.0) / (num_arrows as f32 + 1.0);
        let x = rect_sorted.min.x + frac * rect_sorted.width();
        let y_start = rect_sorted.min.y + 0.2 * rect_sorted.height();
        let y_end = rect_sorted.min.y + 0.7 * rect_sorted.height();

        // Arrow shaft
        let start = egui::pos2(x, y_start);
        let end = egui::pos2(x, y_end);
        painter.line_segment([start, end], egui::Stroke::new(1.5, arrow_color));

        // Arrowhead
        let head_size = 4.0;
        painter.line_segment(
            [end, egui::pos2(x - head_size, y_end - head_size)],
            egui::Stroke::new(1.5, arrow_color),
        );
        painter.line_segment(
            [end, egui::pos2(x + head_size, y_end - head_size)],
            egui::Stroke::new(1.5, arrow_color),
        );
    }
}
```

Note: The arrow rendering above uses a simplified vertical-arrow approach. For arbitrary force directions, rotate the arrow start/end by the force angle. Refine during implementation.

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat: render force zones with overlap highlights and arrows on canvas"
```

---

## Task 8: Canvas Rendering — Labels and Tooltips

**Files:**
- Modify: `linkage-sim-rs/src/gui/canvas.rs`
- Modify: `linkage-sim-rs/src/gui/state.rs`

- [ ] **Step 1: Add show_labels toggle to AppState**

In `linkage-sim-rs/src/gui/state.rs`, add:
```rust
    pub show_labels: bool,
```
Initialize to `true`.

- [ ] **Step 2: Add View menu toggle**

In the View menu rendering code (search for `show_dimensions` or `show_forces` to find the pattern), add:
```rust
    ui.checkbox(&mut state.show_labels, "Show Labels");
```

- [ ] **Step 3: Render labels on canvas**

In `draw_canvas()`, after all mechanism elements are drawn:

```rust
if state.show_labels {
    let label_color = egui::Color32::from_gray(136); // #888
    let label_font = egui::FontId::monospace(10.0);

    // Body/link labels
    for (body_id, body) in mechanism.bodies() {
        if body_id == "ground" { continue; }
        if let Ok(bi) = mechanism.state().get_index(body_id) {
            let bx = q[bi.x_idx()];
            let by = q[bi.y_idx()];
            let pos = world_to_screen(bx, by, &view);
            let label_pos = egui::pos2(pos.x, pos.y - 12.0);
            painter.text(label_pos, egui::Align2::CENTER_BOTTOM, &body.label,
                label_font.clone(), label_color);
        }
    }

    // Joint labels (auto-generated from type + index)
    for (idx, joint) in mechanism.joints().iter().enumerate() {
        // Compute joint world position from connected body states
        // Use the joint's attachment point on body_i
        let joint_label = format!("{}{}", joint_type_prefix(joint), idx + 1);
        // ... position and render similarly
    }
}
```

- [ ] **Step 4: Implement hover tooltips**

Use egui's `Response::on_hover_text()` or a custom tooltip approach. For each hovered element (detected via existing hit-testing), show a tooltip with properties:

```rust
// In the interaction section of draw_canvas, after hit-testing:
if let Some(hovered_body_id) = hovered_body {
    if let Some(body) = bodies.get(hovered_body_id) {
        egui::show_tooltip_at_pointer(ui.ctx(), egui::Id::new("body_tooltip"), |ui| {
            ui.label(egui::RichText::new(&body.label).strong());
            ui.label(format!("Mass: {:.3} kg", body.mass));
            ui.label(format!("Izz: {:.6} kg·m²", body.izz_cg));
            if let Some(ref geo) = body.geometry {
                ui.label(format!("Geometry: {:.1} × {:.1} mm",
                    geo.width * 1e3, geo.height * 1e3));
            }
            // Link length from attachment points
            if body.attachment_points.len() == 2 {
                let pts: Vec<_> = body.attachment_points.values().collect();
                let length = (pts[0] - pts[1]).norm();
                ui.label(format!("Length: {:.1} mm", length * 1e3));
            }
        });
    }
}
```

Similarly for joints (show type, angle, reactions) and force zones (show force, bounds, overlap ratio).

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/gui/canvas.rs linkage-sim-rs/src/gui/state.rs
git commit -m "feat: add toggleable labels and hover tooltips on canvas"
```

---

## Task 9: GUI Controls — Body Geometry Property Panel

**Files:**
- Modify: `linkage-sim-rs/src/gui/property_panel.rs`

- [ ] **Step 1: Add PendingPropertyEdit variants for geometry**

```rust
    AddGeometry { body_id: String, width: f64, height: f64 },
    UpdateGeometry { body_id: String, geometry: BodyGeometry },
    RemoveGeometry { body_id: String },
    UpdateLabel { body_id: String, label: String },
```

- [ ] **Step 2: Add geometry section to property panel**

In the body editing section of `draw_property_panel()`, after mass/inertia controls:

```rust
// -- Geometry Section --
ui.collapsing("Geometry", |ui| {
    if let Some(ref geo) = body.geometry {
        // Width slider (mm display, m internal)
        let mut width_mm = geo.width * 1e3;
        if ui.add(egui::Slider::new(&mut width_mm, 1.0..=500.0)
            .text("Width (mm)")
            .logarithmic(true)).changed() {
            // queue UpdateGeometry edit
        }

        let mut height_mm = geo.height * 1e3;
        if ui.add(egui::Slider::new(&mut height_mm, 1.0..=500.0)
            .text("Height (mm)")
            .logarithmic(true)).changed() {
            // queue UpdateGeometry edit
        }

        // Offset sliders
        // ...

        if ui.button("Remove Geometry").clicked() {
            // queue RemoveGeometry edit
        }
    } else {
        if ui.button("Add Geometry").clicked() {
            // queue AddGeometry with sensible defaults (based on link length)
        }
    }
});
```

- [ ] **Step 3: Apply pending geometry edits**

In the edit application section, handle the new variants by modifying the blueprint and triggering mechanism rebuild.

- [ ] **Step 4: Add label editing text field**

At the top of the body property section:

```rust
// Label editing
let mut label = body.label.clone();
if ui.text_edit_singleline(&mut label).changed() {
    // queue UpdateLabel edit
}
```

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/gui/property_panel.rs
git commit -m "feat: add body geometry editing and label field in property panel"
```

---

## Task 10: GUI Controls — Force Zone Creation

**Files:**
- Modify: `linkage-sim-rs/src/gui/force_toolbar.rs`
- Modify: `linkage-sim-rs/src/gui/state.rs`
- Modify: `linkage-sim-rs/src/gui/canvas.rs`

- [ ] **Step 1: Add ForceZone button to force toolbar**

In `draw_force_toolbar()`, add under "Link Forces" section:

```rust
// -- Spatial Forces --
ui.separator();
ui.colored_label(zone_color, "Spatial:");
if ui.button("Force Zone").clicked() {
    // Enter force zone placement mode
    pending = Some(PendingForceAdd::EnterForceZoneMode);
}
```

Add `EnterForceZoneMode` variant to `PendingForceAdd` if needed, or use a new state field.

- [ ] **Step 2: Add force zone drag-to-define state**

In `linkage-sim-rs/src/gui/state.rs`, add:

```rust
pub struct ForceZoneDragState {
    pub start_world: Vector2<f64>,
}

// In AppState:
pub creating_force_zone: Option<ForceZoneDragState>,
```

- [ ] **Step 3: Implement drag-to-define interaction on canvas**

In the canvas interaction code, when `creating_force_zone` is active:
- On mouse down: record start position in world coords
- On mouse drag: draw preview rectangle
- On mouse up: create the ForceZone with a dialog/panel to set force vector and target body

- [ ] **Step 4: Add force zone property editing**

When a force zone is selected, show in property panel:
- Zone min/max coordinates (editable)
- Force X/Y components (with magnitude + angle alternate display)
- Target body dropdown (filtered to bodies with geometry)

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/gui/force_toolbar.rs linkage-sim-rs/src/gui/state.rs linkage-sim-rs/src/gui/canvas.rs
git commit -m "feat: add force zone creation via toolbar drag-to-define"
```

---

## Task 11: GUI Controls — Sweep Range

**Files:**
- Modify: `linkage-sim-rs/src/gui/input_panel.rs`

- [ ] **Step 1: Add sweep range controls to input panel**

In `draw_input_panel()`, after the crank angle slider:

```rust
// -- Sweep Range --
ui.separator();
ui.checkbox(&mut state.sweep_range_enabled, "Limit Sweep Range");
if state.sweep_range_enabled {
    ui.horizontal(|ui| {
        ui.label("Min°:");
        ui.add(egui::DragValue::new(&mut state.sweep_angle_min_deg)
            .speed(0.5)
            .range(0.0..=360.0)
            .suffix("°"));
        ui.label("Max°:");
        ui.add(egui::DragValue::new(&mut state.sweep_angle_max_deg)
            .speed(0.5)
            .range(0.0..=360.0)
            .suffix("°"));
    });
    // Clamp driver angle slider to range
    state.driver_angle = state.driver_angle.clamp(
        state.sweep_angle_min_deg.to_radians(),
        state.sweep_angle_max_deg.to_radians(),
    );
}
```

- [ ] **Step 2: Wire sweep config into compute_sweep_data calls**

Where `compute_sweep_data()` is called, construct `SweepConfig` from AppState fields:

```rust
let sweep_config = if state.sweep_range_enabled {
    Some(SweepConfig {
        angle_min: state.sweep_angle_min_deg.to_radians(),
        angle_max: state.sweep_angle_max_deg.to_radians(),
        enabled: true,
    })
} else {
    None
};
```

- [ ] **Step 3: Mark sweep dirty when range changes**

Ensure that changing min/max/enabled triggers `state.sweep_dirty = true` so the sweep recomputes.

- [ ] **Step 4: Update animation to respect range**

In animation logic, when `sweep_range_enabled`, bounce/stop at the min/max angles instead of wrapping at 360°.

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/gui/input_panel.rs
git commit -m "feat: add sweep range controls with crank angle limits"
```

---

## Task 12: Parallelogram Press Sample

**Files:**
- Modify: `linkage-sim-rs/src/gui/samples.rs:14-117`

- [ ] **Step 1: Add ParallelogramPress to SampleMechanism enum**

Add variant, label, and `all()` entry:

```rust
// In enum:
ParallelogramPress,

// In label():
SampleMechanism::ParallelogramPress => "Parallelogram Press",

// In all() — after Parallelogram:
SampleMechanism::ParallelogramPress,
```

Add match arm in `build_sample_with_driver()`:
```rust
SampleMechanism::ParallelogramPress => build_parallelogram_press(driver_joint_id),
```

- [ ] **Step 2: Implement build_parallelogram_press**

```rust
/// Parallelogram 4-bar with a rectangular press plate on the coupler
/// passing through a vertical force zone.
///
/// Demonstrates: body geometry, force zones, crank angle limits.
fn build_parallelogram_press(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (0.04_f64, 0.0_f64);
    let l_crank = 0.02_f64;
    let l_coupler = 0.04_f64;
    let l_rocker = 0.02_f64;

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);

    let mut coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
    coupler.label = "Coupler (Press Plate)".to_string();
    coupler.geometry = Some(
        BodyGeometry::new(0.06, 0.015, Vector2::new(0.02, 0.0))
            .expect("valid geometry dimensions")
    );
    // Mass for the press plate
    let plate_mass = 0.5_f64;
    let plate_w = 0.06_f64;
    let plate_h = 0.015_f64;
    coupler.mass = plate_mass;
    coupler.izz_cg = (1.0 / 12.0) * plate_mass * (plate_w * plate_w + plate_h * plate_h);
    coupler.add_coupler_point("P", 0.02, 0.0).unwrap();

    let rocker = make_bar("rocker", "C", "D", l_rocker, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();

    // Force zone: vertical downward force in the working region
    mech.add_force(ForceElement::ForceZone(ForceZoneElement {
        body_id: "coupler".to_string(),
        zone_min: [0.01, -0.005],
        zone_max: [0.03, 0.01],
        force: [0.0, -500.0],
        label: Some("Press Zone".to_string()),
    }));

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4, l_crank, l_coupler, l_rocker, 0.0,
        "crank", "coupler", "rocker",
    );

    Ok((mech, q0))
}
```

Note: The sample builder returns `(Mechanism, DVector<f64>)` per the existing pattern. The sweep angle limits (~150°–210°) must be set in `AppState` when loading this sample. In the sample-loading code (where `build_sample()` is called and AppState is populated), add:

```rust
SampleMechanism::ParallelogramPress => {
    state.sweep_range_enabled = true;
    state.sweep_angle_min_deg = 150.0;
    state.sweep_angle_max_deg = 210.0;
}
```

This keeps the builder signature consistent with all other samples while ensuring the demo shows the working stroke.

- [ ] **Step 3: Run test to verify sample builds**

Run: `cd linkage-sim-rs && cargo test -- sample --nocapture`
Expected: any existing sample tests pass; add a quick test:

```rust
#[test]
fn parallelogram_press_sample_builds() {
    let (mech, q) = build_sample(SampleMechanism::ParallelogramPress);
    assert!(mech.is_built());
    assert!(q.len() > 0);
}
```

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/gui/samples.rs
git commit -m "feat: add Parallelogram Press sample with force zone"
```

---

## Task 13: Schema Version Bump & Docs

**Files:**
- Modify: `linkage-sim-rs/src/io/serialization.rs:23`

- [ ] **Step 1: Bump schema version**

Change `SCHEMA_VERSION` from `"1.0.0"` to `"1.1.0"` (additive, backward-compatible).

- [ ] **Step 2: Run full test suite**

Run: `cd linkage-sim-rs && cargo test`
Expected: all tests PASS (schema version is checked with `semver_major()` so major=1 is compatible)

- [ ] **Step 3: Update docs**

Update the project README or relevant docs to mention:
- Body geometry feature
- Force zones
- Labels & tooltips
- Crank angle limits
- Parallelogram Press sample

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: bump schema to 1.1.0, update docs for force zones and body geometry"
```

---

## Task Dependencies

```
Task 1 (polygon utils) ──→ Task 3 (ForceZone) ──→ Task 7 (canvas force zones)
                        ╲                       ╲
                         ╲→ Task 6 (canvas geo)   ╲→ Task 10 (force zone creation)
                                                    ╲→ Task 12 (sample)
Task 2 (BodyGeometry) ──→ Task 3 (ForceZone)
                       ╲→ Task 6 (canvas geo)
                       ╲→ Task 9 (geometry panel)

Task 4 (SweepConfig) ──→ Task 11 (sweep range controls)
                      ╲→ Task 12 (sample)

Task 5 (joint labels) ──→ Task 8 (canvas labels)

Task 13 (schema + docs) depends on all above
```

**Parallelizable groups:**
- Tasks 1, 2, 4, 5 are independent of each other
- Tasks 6, 7, 8 are independent canvas work (but depend on 1-5)
- Tasks 9, 10, 11 are independent GUI work
