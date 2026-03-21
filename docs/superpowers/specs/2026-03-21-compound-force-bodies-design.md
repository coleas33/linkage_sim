# Compound Force Bodies & Zoom to Fit

**Date:** 2026-03-21
**Status:** Design approved, pending implementation

## Problem

Two-body force elements (springs, dampers, gas springs, actuators) are modeled as
pure forces between points — they have no physical bodies, joints, or kinematic
constraints. When a force is mounted at an independent pivot (mount point), there
is no revolute joint at that location, and no prismatic joint to constrain the
force's telescoping motion. The actuator "floats" as a mathematical force without
physical structure.

Additionally, the canvas does not auto-fit the view when loading a mechanism,
requiring manual pan/zoom to see the linkage.

## Requirements

1. When a two-body force element references **mount points** (via `point_a_name`
   / `point_b_name`), it is auto-expanded at build time into a **compound
   structure**: 2 bodies (cylinder + rod) + 2 revolute joints + 1 prismatic
   joint + force along the prismatic axis.
2. Applies to **all two-body point force types**: LinearSpring, LinearDamper,
   GasSpring, LinearActuator.
3. Only activates when the referenced point name exists in `mount_points` — forces
   referencing `attachment_points` (joint locations) stay as pure force elements.
4. Compound bodies are **massless by default** with optional user-configurable
   mass/inertia.
5. The **blueprint** stores only the original force element — expansion is a
   build-time transformation invisible to serialization and undo/redo.
6. **Zoom to fit** on mechanism load + `F` keyboard shortcut.

## Approach

**Build-time expansion** in `load_mechanism_unbuilt_from_json` / `rebuild()`.

The blueprint (MechanismJson) is the source of truth and stores forces as-is.
During mechanism construction, forces with mount-point references are detected
and expanded into compound structures. The expansion adds bodies, joints, and a
modified force to the mechanism being built — but never touches the blueprint.

This keeps serialization, undo/redo, and the property panel simple: they only
see the original force element. The compound structure exists only in the built
`Mechanism`.

## Design

### 1. Compound Force Expansion Logic

When `load_mechanism_unbuilt_from_json` processes a two-body force element where
`point_a_name` or `point_b_name` references a **mount point** (checked by
verifying the name exists in `body.mount_points`, NOT `body.attachment_points`):

**For a force at index `i` between `body_a/point_a_name` and `body_b/point_b_name`:**

1. **Resolve mount point positions** to get global coordinates at the initial
   configuration.

2. **Compute initial geometry**:
   - `initial_length` = distance between the two resolved points
   - `half_len` = `initial_length / 2`
   - `angle` = atan2(dy, dx) from point_a to point_b

3. **Create cylinder body** (`force_{i}_cyl`):
   - Mass: 0.0 (default), Izz: 0.0
   - Attachment points: `base` at (0, 0), `slide` at (half_len, 0)
   - `render_shape`: rectangle outline for cylinder tube appearance

4. **Create rod body** (`force_{i}_rod`):
   - Mass: 0.0 (default), Izz: 0.0
   - Attachment points: `slide` at (0, 0), `tip` at (half_len, 0)
   - `render_shape`: thinner rectangle for piston rod appearance

5. **Create joints**:
   - **Revolute** (`force_{i}_base`): `body_a/point_a_name` ↔ `force_{i}_cyl/base`
   - **Revolute** (`force_{i}_tip`): `body_b/point_b_name` ↔ `force_{i}_rod/tip`
   - **Prismatic** (`force_{i}_slide`): `force_{i}_cyl/slide` ↔ `force_{i}_rod/slide`,
     axis = (1, 0) in cylinder local frame

6. **Replace the force element**: The original two-point force is replaced with
   a force applied along the prismatic joint axis. The force parameters
   (stiffness, damping, free_length, etc.) are preserved — only the attachment
   bodies/points change to reference the compound bodies.

**When only ONE end references a mount point:**
- Only that end gets a revolute joint to the compound body
- The other end (referencing an attachment point / existing joint) connects
  directly — the joint already provides the pivot

**Naming convention:**
- Bodies: `force_{i}_cyl`, `force_{i}_rod`
- Joints: `force_{i}_base`, `force_{i}_tip`, `force_{i}_slide`
- Where `i` is the force element's index in the blueprint's force list

### 2. Mixed Mount/Attachment Point Handling

When a force has one mount point and one attachment point:

- **Mount point end**: gets the compound body + revolute joint
- **Attachment point end**: force connects to the compound body, and the compound
  body's "tip" (or "base") gets a revolute joint to the existing body at the
  attachment point location

This means even with one mount point, the full compound structure (2 bodies +
prismatic) is created — the only difference is whether the revolute joint
connects to a mount point or an existing joint's attachment point.

### 3. Zoom to Fit

**Auto-fit on load:** After `load_sample()` or file load succeeds and
`rebuild()` completes, call `self.fit_to_view(canvas_width, canvas_height)`.
The `fit_to_view()` method already exists in `state.rs` (lines 942-984) — it
computes bounding box of all attachment points with 15% margin.

**Update `fit_to_view` to include mount points** in the bounding box calculation
(currently only iterates `attachment_points`).

**Keyboard shortcut:** Press `F` to trigger `fit_to_view()`. Add to the
existing keyboard handler.

### 4. Canvas Rendering

Compound force bodies are **first-class bodies and joints** in the built
mechanism, so existing rendering handles them automatically:

- Cylinder and rod bodies render via their `render_shape` polygons
- Revolute joints at both ends render as circle markers
- Prismatic joint renders as square marker

The only new rendering requirement is setting appropriate `render_shape` values
on the auto-created bodies:
- Cylinder: wider rectangle (representing the tube)
- Rod: thinner rectangle (representing the piston)

No changes to `draw_force_elements()` — the compound force is no longer a
two-point force element in the built mechanism.

### 5. Property Panel Integration

The property panel shows the **blueprint** force element — not the expanded
compound. Users edit spring stiffness, actuator force, etc. on the original
element. The compound bodies are invisible in the property panel.

**Optional mass/inertia**: If we want users to set mass on compound bodies,
we'd need new fields on the force element (e.g., `cylinder_mass`,
`rod_mass`). These would be forwarded to the compound bodies during expansion.
For now, default to zero — add mass fields later if needed.

### 6. Testing Strategy

**Unit tests:**
- Expansion produces correct body/joint/force count
- Only triggers for mount points, not attachment points
- Forces with `point_a_name: None` stay as pure force elements
- Mixed case (one mount, one attachment) handled correctly
- Massless bodies default correctly

**Serialization tests:**
- Blueprint round-trip preserves original force (not expanded form)
- Undo/redo through expansion cycles works correctly

**Integration test:**
- Four-bar + mount-point spring: build, solve kinematics, verify prismatic
  joint allows length change
- Compare compound model forces against equivalent pure-force model

**Zoom tests:**
- `fit_to_view` includes mount points in bounding box
- Keyboard shortcut triggers fit
- Auto-fit fires on sample load

## Files to Modify

| File | Change |
|------|--------|
| `linkage-sim-rs/src/io/serialization.rs` | Add compound expansion logic in `load_mechanism_unbuilt_from_json`, after bodies loaded but before build |
| `linkage-sim-rs/src/core/body.rs` | Add `render_shape` field if not present; helper to create cylinder/rod bodies |
| `linkage-sim-rs/src/gui/state.rs` | Call `fit_to_view` after load/rebuild; update `fit_to_view` to include mount_points |
| `linkage-sim-rs/src/gui/canvas.rs` | Handle `F` key for zoom-to-fit |
| `linkage-sim-rs/src/gui/mod.rs` | Wire keyboard shortcut |
