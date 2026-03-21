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

Expansion happens inside `load_mechanism_unbuilt_from_json`, **after** all
bodies and joints from the blueprint are added, but **before** `mech.build()`.
Forces are processed in a second pass: non-compound forces are added directly;
compound forces trigger the expansion.

When a two-body force element has `point_a_name` or `point_b_name` referencing a
**mount point** (checked by verifying the name exists in `body.mount_points`,
NOT `body.attachment_points`):

**For a force at index `i` between `body_a/point_a_name` and `body_b/point_b_name`:**

1. **Promote mount points to attachment points**: For each end that references a
   mount point, copy the mount point coordinates into the original body's
   `attachment_points` under a synthetic name `_force_{i}_mount_a` (or `_b`).
   This is required because `Mechanism::add_revolute_joint` only searches
   `attachment_points` — it does not search `mount_points`. The leading
   underscore marks these as auto-generated.

2. **Compute initial geometry**:
   - Resolve both point positions (from attachment_points or promoted mount points)
   - `initial_length` = distance between the two resolved points
   - `half_len` = `initial_length / 2`
   - `angle` = atan2(dy, dx) from point_a to point_b
   - `midpoint` = center between the two points

3. **Create cylinder body** (`force_{i}_cyl`):
   - Mass: 0.0 (default), Izz: 0.0
   - Attachment points: `base` at (0, 0), `slide` at (half_len, 0)
   - `render_shape`: rectangle outline for cylinder tube appearance
   - Add to mechanism via `mech.add_body()`

4. **Create rod body** (`force_{i}_rod`):
   - Mass: 0.0 (default), Izz: 0.0
   - Attachment points: `slide` at (0, 0), `tip` at (half_len, 0)
   - `render_shape`: thinner rectangle for piston rod appearance
   - Add to mechanism via `mech.add_body()`

5. **Create joints**:
   - **Revolute** (`force_{i}_base`): `body_a/_force_{i}_mount_a` ↔
     `force_{i}_cyl/base` — cylinder pivots at mount point
   - **Revolute** (`force_{i}_tip`): `body_b/_force_{i}_mount_b` ↔
     `force_{i}_rod/tip` — rod pivots at second mount/attachment point
   - **Prismatic** (`force_{i}_slide`): `force_{i}_cyl/slide` ↔
     `force_{i}_rod/slide`, axis = (1, 0) in cylinder local frame
   - The prismatic joint contributes **2 constraint equations** (perpendicular
     displacement + rotation lock), so net DOF change is: 2 bodies × 3 DOF −
     2 revolute × 2 eq − 1 prismatic × 2 eq = **0 net DOF added**

6. **Set initial poses for compound bodies**: After `mech.build()` creates the
   state vector `q0`, set poses for the compound bodies:
   - `force_{i}_cyl`: position = point_a global coords, angle = `angle`
   - `force_{i}_rod`: position = midpoint, angle = `angle`
   - This ensures the Newton-Raphson solver starts from a physically correct
     configuration rather than the origin.

7. **Replace the force element**: The original force is **not added** to
   `mech.forces`. Instead, a new force element is created referencing the
   compound bodies:
   - `body_a` = `force_{i}_cyl`, `point_a` = `slide` coordinates
   - `body_b` = `force_{i}_rod`, `point_b` = `slide` coordinates
   - All other parameters (stiffness, damping, etc.) preserved
   - `point_a_name` / `point_b_name` cleared (the compound bodies don't have
     mount points — the coordinates are direct attachment points)

**When only ONE end references a mount point** (the other references an
attachment point on an existing body):
- The full compound structure (2 bodies + prismatic) is still created
- The mount-point end gets a revolute joint connecting the compound body to
  the original body at the promoted mount point
- The attachment-point end gets a revolute joint connecting the compound body
  to the original body at the existing attachment point (which already exists
  in `attachment_points`, so no promotion needed)
- The existing joint at that attachment point is unaffected — the revolute
  joint for the compound body is a separate joint at the same location

### 2. Blueprint Snapshot Ordering

**Critical**: The blueprint snapshot must be taken **before** compound expansion.

In `load_sample()`, the current code does:
1. Build sample mechanism
2. `self.blueprint = mechanism_to_json(&mech)` — snapshots the built mechanism

This will capture expanded compound bodies in the blueprint, violating the
invariant. The fix:

- **`load_sample()`**: Currently `build_sample()` returns an already-built
  `Mechanism`. Refactor: `build_sample()` must also return (or be changed to
  return) a `MechanismJson` as the pre-expansion blueprint. Concretely, call
  `mechanism_to_json(&mech)` on the sample mechanism **before** compound
  expansion runs, and store that as `self.blueprint`. Then `rebuild()` loads
  from this clean blueprint and expansion happens inside
  `load_mechanism_unbuilt_from_json`. The simplest approach: have `load_sample`
  call `mechanism_to_json` first to capture the blueprint, then call
  `self.rebuild()` which loads from the blueprint and handles expansion.
- **`rebuild()`**: Loads from blueprint (which has no compounds), calls
  `load_mechanism_unbuilt_from_json` which handles expansion during the build
  phase. Never snapshots the expanded form.
- **Undo/redo**: Snapshots are blueprint-level — they never see compounds.

**Concrete implementation**: Move compound expansion into a new function
`expand_compound_forces(mech: &mut Mechanism, forces: &[ForceElement])` called
inside `load_mechanism_unbuilt_from_json` right before the force-add loop. The
blueprint is already captured before this function runs. For `load_sample`,
the key change is: snapshot the blueprint from the pre-expansion mechanism,
then let `rebuild()` handle expansion via the normal
`load_mechanism_unbuilt_from_json` path.

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

The original force element is **not in the built mechanism's force list** (it
was replaced by the compound-body force), so `draw_force_elements()` will not
double-draw it. No changes needed to `draw_force_elements()`.

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
- Mixed case (one mount, one attachment) creates full compound with revolute
  at the attachment point end too
- Massless bodies default correctly
- Mount point promoted to attachment_points under synthetic name
- Initial poses set correctly for compound bodies

**Serialization tests:**
- Blueprint round-trip preserves original force (not expanded form)
- `mechanism_to_json` on rebuilt mechanism does NOT contain compound bodies
  (because blueprint is snapshotted before expansion)
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
| `linkage-sim-rs/src/io/serialization.rs` | Add `expand_compound_forces()` function; call it in `load_mechanism_unbuilt_from_json` before force-add loop; promote mount points to attachment_points |
| `linkage-sim-rs/src/core/body.rs` | Ensure `render_shape` field exists; helper to create cylinder/rod bodies |
| `linkage-sim-rs/src/core/mechanism.rs` | May need to expose `set_pose` or initial q0 manipulation for compound bodies |
| `linkage-sim-rs/src/gui/state.rs` | Call `fit_to_view` after load/rebuild; update `fit_to_view` to include mount_points; ensure blueprint snapshot happens before expansion |
| `linkage-sim-rs/src/gui/canvas.rs` | Handle `F` key for zoom-to-fit |
| `linkage-sim-rs/src/gui/mod.rs` | Wire keyboard shortcut |
