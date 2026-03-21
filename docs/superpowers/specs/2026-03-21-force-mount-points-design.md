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

Add `mount_points` field (inserted after `izz_cg`, before `coupler_points` to
match existing field ordering in the actual struct):

```rust
pub struct Body {
    pub id: String,
    pub attachment_points: HashMap<String, Vector2<f64>>,  // joint connections
    pub mass: f64,
    pub cg_local: Vector2<f64>,
    pub izz_cg: f64,
    pub mount_points: HashMap<String, Vector2<f64>>,       // NEW: force mounts
    pub coupler_points: HashMap<String, Vector2<f64>>,     // tracing points
}
```

New methods:

```rust
impl Body {
    /// Add a named mount point for force attachment.
    /// Rejects names that collide with attachment_points or existing mount_points.
    pub fn add_mount_point(&mut self, name: &str, x: f64, y: f64) -> Result<(), BodyError>;

    /// Look up a named point from attachment_points or mount_points.
    /// Since name uniqueness is enforced across both collections, lookup order
    /// is irrelevant — a name can only exist in one collection. Searches both
    /// and returns the match, or errors with available names from both.
    pub fn resolve_force_point(&self, name: &str) -> Result<&Vector2<f64>, BodyError>;
}
```

Update all struct literals that construct `Body`:
- `Body::new()` — add `mount_points: HashMap::new()`
- `make_ground()` — add `mount_points: HashMap::new()`
- `make_bar()` — add `mount_points: HashMap::new()`
- `load_mechanism_unbuilt_from_json()` in `serialization.rs` (line ~419-434) —
  deserialize `mount_points` from `BodyJson` analogous to `coupler_points`

New `BodyError` variants:
- `DuplicateMountPoint { point, body }` — name already in mount_points
- `MountPointNameCollision { point, body }` — name exists in attachment_points
- `ForcePointNotFound { point, body, available_attachment, available_mount }` —
  not found in either collection

#### BodyJson (`io/serialization.rs`)

```rust
pub struct BodyJson {
    pub attachment_points: HashMap<String, [f64; 2]>,
    pub mass: f64,
    pub cg_local: [f64; 2],
    pub izz_cg: f64,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub mount_points: HashMap<String, [f64; 2]>,           // NEW (before coupler_points, matching Body field order)
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub coupler_points: HashMap<String, [f64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub point_masses: Vec<PointMassJson>,
}
```

**Critical:** `body_to_json()` (~line 219-243) must map
`body.mount_points` → `BodyJson.mount_points` (same pattern as `coupler_points`).
Without this, undo/redo snapshots (which serialize via `mechanism_to_json`) will
silently drop all mount points, creating dangling `point_X_name` references on
restore.

#### Force elements (`forces/elements.rs`)

Add `point_X_name: Option<String>` to the 5 elements that have point fields:

| Element                 | New fields                                  | Notes |
|-------------------------|---------------------------------------------|-------|
| `LinearSpringElement`   | `point_a_name`, `point_b_name`              | Caches into `point_a`, `point_b` |
| `LinearDamperElement`   | `point_a_name`, `point_b_name`              | Caches into `point_a`, `point_b` |
| `GasSpringElement`      | `point_a_name`, `point_b_name`              | Caches into `point_a`, `point_b` |
| `LinearActuatorElement` | `point_a_name`, `point_b_name`              | Caches into `point_a`, `point_b` |
| `ExternalForceElement`  | `local_point_name`                          | **Different field name** — mirrors existing `local_point`, not `point_a`. Caches into `local_point`. |

All use `#[serde(default, skip_serializing_if = "Option::is_none")]`.

Forces are serialized with their `point_X_name` field intact — no reverse
coordinate lookup is needed. The existing `find_point_name` function in
`serialization.rs` is used only for joints and is **not involved** in force
serialization. However, `find_point_name` should also be updated to search
`mount_points` for robustness.

Untouched (no point fields): `TorsionSpringElement`, `RotaryDamperElement`,
`BearingFrictionElement`, `JointLimitElement`, `MotorElement`, `GravityElement`.

### 2. Point Resolution Logic

Resolution happens once at mechanism **build time**, not per-frame.

**`resolve_force_point`** searches both `attachment_points` and `mount_points`.
Since name uniqueness is enforced across both collections, a name can only exist
in one — there is no priority ordering. Returns error listing available names
from both collections if not found.

**Name uniqueness** is enforced across `attachment_points` AND `mount_points` on
the same body. `add_mount_point` rejects names that collide with either
collection.

**Caching:** When `point_a_name` is `Some`, the resolved coordinates are written
back to `point_a` (or `local_point` for `ExternalForceElement`) so the solver
never does a HashMap lookup in the hot loop.

**Fallback:** When `point_a_name` is `None`, raw `point_a` coordinates are used
directly (backward compatibility).

### 3. GUI Changes

All GUI edits go through the `blueprint` → `rebuild()` path. Mount point edits
mutate `blueprint.bodies[body_id].mount_points` in `BodyJson` form, then trigger
mechanism rebuild — same pattern as all other property edits.

#### New `PendingPropertyEdit` variants (`gui/state.rs`)

- `AddMountPoint { body_id, name, position }` — adds to blueprint, triggers rebuild
- `DeleteMountPoint { body_id, name }` — removes from blueprint, clears any force
  `point_X_name` that references it (cascade), triggers rebuild
- `RenameMountPoint { body_id, old_name, new_name }` — renames in blueprint,
  updates any force `point_X_name` that references the old name (cascade),
  triggers rebuild
- `UpdateMountPointPosition { body_id, name, position }` — updates coords in
  blueprint, triggers rebuild

#### Mount point deletion/rename — dangling reference handling

When a mount point is **deleted**, any force element with a matching
`point_X_name` has the name **cleared to `None`** and falls back to its cached
raw coordinates (which still hold the last resolved position). A warning toast is
shown: "Mount point 'M1' removed — 2 force(s) reverted to fixed coordinates."

When a mount point is **renamed**, any force element with the old name is
**updated to the new name** automatically. No warning needed — this is seamless.

#### Body editor (property_panel.rs)

New "Mount Points" collapsible section below attachment points:
- List existing mount points with name + x/y, each with delete button
- "Add Mount Point" button — auto-names `M1`, `M2`, etc., places at body CG
- Name editable inline (triggers `RenameMountPoint` on commit)
- x/y via DragValue sliders (triggers `UpdateMountPointPosition` on release)
- Validation rejects duplicates across attachment_points + mount_points

#### Force property panel (property_panel.rs)

Replace raw x/y DragValues with **point picker dropdown**:
- Lists all attachment_points (labeled "joint") + mount_points (labeled "mount")
  on the body
- "Custom coords..." option falls back to raw x/y DragValues
- Selecting a named point sets `point_X_name = Some(...)` and caches coords
- **Body dropdown** shows all bodies in mechanism (not just joint-connected)

#### Force toolbar (force_toolbar.rs)

The current `resolve_target_bodies` derives the second body by scanning joint
topology — this is fundamentally incompatible with the "any body" requirement.
Replace `connected_body` derivation with an **explicit second body picker
dropdown** for two-body force elements:

- Primary body: from Link Editor dropdown — **must now include ground** (the
  current code filters ground out; this filter must be removed for force
  creation). A gas spring between ground and a link is a common configuration.
- Secondary body: new dropdown listing all bodies in the mechanism **including
  ground**
- Default selection: first joint-connected body (preserving current UX for common
  case), but user can pick any body including ground
- New forces default to `point_a: [0.0, 0.0]` / `point_b: [0.0, 0.0]` with
  `point_a_name: None` / `point_b_name: None` (same as current behavior).
  The user then selects named points via the property panel point picker.

### 4. Canvas Rendering (canvas.rs)

- **Mount points**: diamond markers, ~6px, filled magenta/purple (`#e056fd`)
- **Joint points**: existing circle markers (unchanged)
- **Labels**: mount point names next to diamonds (same font, distinct color)
- **Selection/hover**: mount points clickable with same hit-test radius as joints
- **Force lines**: unchanged — draw from resolved global A to global B

### 5. Serialization & Backward Compatibility

Zero-migration approach — all new fields use serde defaults:

- `mount_points` missing in JSON → `HashMap::new()` (via `#[serde(default)]`)
- `point_a_name` missing → `None` (via `#[serde(default)]`) → uses raw coords
- No schema version bump needed (purely additive)

**Save path:** `body_to_json()` must explicitly serialize `mount_points`.
`mechanism_to_json()` calls `body_to_json()`, and the undo system snapshots via
`mechanism_to_json()` — so this is the single chokepoint. Force elements are
serialized with `point_X_name` intact via serde derive (no custom logic needed).

**Round-trip invariant:** `mechanism_to_json(load_mechanism_unbuilt(json))` must
preserve all mount points and force point names exactly.

### 6. Testing Strategy

**Unit tests (body.rs):**
- `add_mount_point` happy path
- Duplicate rejection within mount_points
- Cross-collection collision (name already in attachment_points)
- `resolve_force_point` from attachment_points
- `resolve_force_point` from mount_points
- `resolve_force_point` error on missing (lists both collections)

**Unit tests (resolution):**
- Named point resolves correctly and caches to raw field
- Fallback to raw coords when name is None
- Error on invalid name with descriptive message
- `ExternalForceElement` resolution caches to `local_point` (not `point_a`)

**Serialization round-trip tests:**
- Body with mount_points round-trips through `body_to_json` → `BodyJson` → `Body`
- Force with `point_a_name` round-trips
- Old JSON without `mount_points` or `point_X_name` loads with correct defaults
- Invalid name produces clear build-time error
- Undo snapshot preserves mount points: serialize → deserialize → verify mount
  points present and force `point_X_name` references still valid

**Integration test:**
- Full mechanism with mount-point-attached forces: build, solve kinematics,
  verify generalized forces match equivalent raw-coord forces

## Files to Modify

| File | Change |
|------|--------|
| `linkage-sim-rs/src/core/body.rs` | Add `mount_points` field, `add_mount_point()`, `resolve_force_point()`, update `Body::new()`, `make_ground()`, `make_bar()` constructors, new `BodyError` variants, new tests |
| `linkage-sim-rs/src/io/serialization.rs` | Add `mount_points` to `BodyJson`, update `body_to_json()` and `load_mechanism_unbuilt_from_json()` Body struct literal, update `find_point_name()` to search mount_points |
| `linkage-sim-rs/src/forces/elements.rs` | Add `point_X_name` to 5 element structs (`local_point_name` for ExternalForce) |
| `linkage-sim-rs/src/forces/` (new or existing) | Force point resolution at build time |
| `linkage-sim-rs/src/gui/state.rs` | New `PendingPropertyEdit` variants: `AddMountPoint`, `DeleteMountPoint`, `RenameMountPoint`, `UpdateMountPointPosition`; cascade logic for delete/rename |
| `linkage-sim-rs/src/gui/property_panel.rs` | Mount point editor section, point picker dropdown for forces |
| `linkage-sim-rs/src/gui/force_toolbar.rs` | Replace `resolve_target_bodies` connected-body derivation with explicit second body picker dropdown; remove ground exclusion filter; delete or invert existing test `resolve_target_bodies_ground_selection_excluded` (it asserts ground is invalid — now contradicts the design) |
| `linkage-sim-rs/src/gui/canvas.rs` | Diamond markers for mount points, labels, selection hit-testing |
