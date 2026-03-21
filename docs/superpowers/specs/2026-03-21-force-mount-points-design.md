# Force Mount Points — Independent Pivot & Flexible Attachment

**Date:** 2026-03-21
**Status:** Design approved, pending implementation

## Problem

Force elements (springs, dampers, gas springs, linear actuators) can only attach
at raw `[f64; 2]` local coordinates on bodies. There is no concept of a named
mount point dedicated to force attachment, no GUI workflow for placing such
points, and forces cannot easily reference existing joint locations by name. Users
need actuators and other force mechanisms to pivot at points independent of
existing joint locations, and to attach at specific named points on links.

## Requirements

1. All two-body force elements support independent pivot points and flexible
   attachment on **any body** (including ground).
2. Users define force attachment locations as **named mount points** on bodies.
3. Mount points are **visually distinct** from joint attachment points on the
   canvas (diamond vs circle markers).
4. Forces reference points **by name** — resolving to cached local coordinates at
   build time so the solver hot loop has no HashMap overhead.
5. Full backward compatibility with existing JSON files (all new fields use serde
   defaults).

## Approach

**Approach A (selected): Separate `mount_points` collection on `Body`.**

Add `mount_points: HashMap<String, Vector2<f64>>` alongside the existing
`attachment_points` and `coupler_points` collections. Force elements gain
optional `point_X_name: Option<String>` fields that resolve from either
collection at build time. Follows the established pattern in the codebase.

Rejected alternatives:
- **B: Unified `named_points` system** — cleanest long-term but requires
  touching all joint/body code and a schema migration. Overkill for this feature.
- **C: Add to existing `attachment_points`** — minimal model change but no
  data-level distinction between joint and mount points; GUI introspection is
  fragile.

## Design

### 1. Data Model Changes

#### Body struct (`core/body.rs`)

Add `mount_points` field:

```rust
pub struct Body {
    pub id: String,
    pub attachment_points: HashMap<String, Vector2<f64>>,  // joint connections
    pub coupler_points: HashMap<String, Vector2<f64>>,     // tracing points
    pub mount_points: HashMap<String, Vector2<f64>>,       // NEW: force mounts
    pub mass: f64,
    pub cg_local: Vector2<f64>,
    pub izz_cg: f64,
}
```

New methods:

```rust
impl Body {
    pub fn add_mount_point(&mut self, name: &str, x: f64, y: f64) -> Result<(), BodyError>;
    pub fn resolve_force_point(&self, name: &str) -> Result<&Vector2<f64>, BodyError>;
}
```

Update `make_ground()` and `make_bar()` to include `mount_points: HashMap::new()`.

#### BodyJson (`io/serialization.rs`)

```rust
pub struct BodyJson {
    pub attachment_points: HashMap<String, [f64; 2]>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub coupler_points: HashMap<String, [f64; 2]>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub mount_points: HashMap<String, [f64; 2]>,           // NEW
    pub mass: f64,
    pub cg_local: [f64; 2],
    pub izz_cg: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub point_masses: Vec<PointMassJson>,
}
```

#### Force elements (`forces/elements.rs`)

Add `point_X_name: Option<String>` to the 5 elements that have point fields:

| Element                 | New fields                                  |
|-------------------------|---------------------------------------------|
| `LinearSpringElement`   | `point_a_name`, `point_b_name`              |
| `LinearDamperElement`   | `point_a_name`, `point_b_name`              |
| `GasSpringElement`      | `point_a_name`, `point_b_name`              |
| `LinearActuatorElement` | `point_a_name`, `point_b_name`              |
| `ExternalForceElement`  | `local_point_name`                          |

All use `#[serde(default, skip_serializing_if = "Option::is_none")]`.

Untouched (no point fields): `TorsionSpringElement`, `RotaryDamperElement`,
`BearingFrictionElement`, `JointLimitElement`, `MotorElement`, `GravityElement`.

### 2. Point Resolution Logic

Resolution happens once at mechanism **build time**, not per-frame.

**`resolve_force_point`** searches `attachment_points` first (joint points take
priority), then `mount_points`. Returns error with both collections listed if not
found.

**Name uniqueness** is enforced across `attachment_points` AND `mount_points` on
the same body. `add_mount_point` rejects names that collide with either
collection.

**Caching:** When `point_a_name` is `Some`, the resolved coordinates are written
back to `point_a` so the solver never does a HashMap lookup in the hot loop.

**Fallback:** When `point_a_name` is `None`, raw `point_a` coordinates are used
directly (backward compatibility).

### 3. GUI Changes

#### Body editor (property_panel.rs)

New "Mount Points" collapsible section below attachment points:
- List existing mount points with name + x/y, each with delete button
- "Add Mount Point" button — auto-names `M1`, `M2`, etc., places at body CG
- Name editable inline, x/y via DragValue sliders
- Validation rejects duplicates across attachment_points + mount_points

#### Force property panel (property_panel.rs)

Replace raw x/y DragValues with **point picker dropdown**:
- Lists all attachment_points + mount_points on the body, labeled with type
- "Custom coords..." option falls back to raw x/y DragValues
- Selecting a named point sets `point_X_name = Some(...)` and caches coords
- **Body dropdown** shows all bodies in mechanism (not just joint-connected)

#### Force toolbar (force_toolbar.rs)

- New forces default to first attachment point on each body (with name set)
- "Other body" selection shows all bodies (not just joint-connected)

### 4. Canvas Rendering (canvas.rs)

- **Mount points**: diamond markers, ~6px, filled magenta/purple (`#e056fd`)
- **Joint points**: existing circle markers (unchanged)
- **Labels**: mount point names next to diamonds (same font, distinct color)
- **Selection/hover**: mount points clickable with same hit-test radius as joints
- **Force lines**: unchanged — draw from resolved global A to global B

### 5. Serialization & Backward Compatibility

Zero-migration approach — all new fields use serde defaults:

- `mount_points` missing in JSON → `HashMap::new()`
- `point_a_name` missing → `None` → uses raw `point_a` coords
- No schema version bump needed (purely additive)

### 6. Testing Strategy

**Unit tests (body.rs):**
- `add_mount_point` happy path
- Duplicate rejection within mount_points
- Cross-collection collision (name in attachment_points)
- `resolve_force_point` from attachment_points, mount_points, priority, missing

**Unit tests (resolution):**
- Named point resolves correctly
- Fallback to raw coords when name is None
- Error on invalid name

**Serialization round-trip tests:**
- Body with mount_points round-trips
- Force with point_a_name round-trips
- Old JSON without new fields loads correctly
- Invalid name produces clear build-time error

**Integration test:**
- Full mechanism with mount-point-attached forces: build, solve kinematics,
  verify generalized forces match equivalent raw-coord forces

## Files to Modify

| File | Change |
|------|--------|
| `linkage-sim-rs/src/core/body.rs` | Add `mount_points` field, `add_mount_point()`, `resolve_force_point()`, update constructors, new error variants, new tests |
| `linkage-sim-rs/src/io/serialization.rs` | Add `mount_points` to `BodyJson`, update `Body`↔`BodyJson` conversion |
| `linkage-sim-rs/src/forces/elements.rs` | Add `point_X_name` to 5 element structs |
| `linkage-sim-rs/src/forces/` (new or existing) | Force point resolution at build time |
| `linkage-sim-rs/src/gui/property_panel.rs` | Mount point editor, point picker dropdown |
| `linkage-sim-rs/src/gui/force_toolbar.rs` | Default to named points, any-body selection |
| `linkage-sim-rs/src/gui/canvas.rs` | Diamond markers, mount point labels, selection |
