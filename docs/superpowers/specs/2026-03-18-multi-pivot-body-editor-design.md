# Multi-Pivot Body Editor Design

**Date:** 2026-03-18
**Status:** Approved
**Scope:** GUI editor enhancements to support creating and editing bodies with arbitrary numbers of attachment points (ternary plates, bell-cranks, etc.)

---

## Problem

The current Draw Link tool only creates binary bars (2 attachment points). There is no GUI path to create bodies with 3+ attachment points, which means ternary plates, bell-cranks, and higher-order bodies cannot be built interactively — despite the core data model (`Body`, `BodyJson`) fully supporting them.

Design Principle #1: "Bodies are the truth, constraints connect them. A ternary plate is just a body with three attachment points." The editor must honor this.

---

## Solution: Three Complementary Features

### A) "Add Pivot Here" — context menu on existing bodies

Adds an attachment point to any existing body at the clicked position.

### B) "+ Body" tool — click-to-place multi-point bodies

New toolbar mode for creating bodies with arbitrary numbers of attachment points.

### C) Body-aware Draw Link — snap to body segments

Draw Link start/end points can land on a body's edge, auto-creating a pivot on that body.

---

## Mode vs Action Rules

The editor separates persistent interaction modes from one-shot actions. This keeps the toolbar unambiguous.

### Modes (toolbar, persistent until switched)

| Mode | Activation | Canvas behavior |
|------|-----------|----------------|
| Select | Toolbar / Esc | Click to select, drag points to move, drag empty to pan |
| Draw Link | Toolbar | Click-drag to create binary bar + auto-joints |
| Add Body | Toolbar | Click to place points, double-click/Enter to finish |
| Add Ground Pivot | Toolbar | Click to place ground pivots |

### Actions (one-shot, no mode change)

| Action | Access | Target |
|--------|--------|--------|
| Delete | Del/Backspace key, context menu | Selected entity |
| Set Driver | Context menu, property panel | Grounded revolute joint |
| Add Pivot | Context menu on body | Body under cursor |
| Create Joint | Context menu on attachment point | Two-click: point to point |
| Start Body Here | Context menu on empty canvas | Seeds Add Body mode at click position |

### Toolbar layout (modes only)

```
[Select] [Draw Link] [+ Body] [+ Ground]
```

Delete and Set Driver are removed from the toolbar. Delete is accessed via Del/Backspace keyboard shortcut and context menus. Set Driver is accessed via context menus on valid joints and the property panel.

---

## Select Mode Drag Precedence

1. **Pointer within HIT_RADIUS of attachment point** -> drag-to-move that point (undo pushed on first movement)
2. **Pointer on empty canvas** -> pan
3. **Pointer near unselected body** -> select it (no move until next drag)

Middle-mouse or Shift+drag always pans regardless of what is under cursor (existing behavior).

---

## Context Menus

### Right-click body vs. attachment point — hit-test priority

The existing `ContextMenuTarget` must be extended to distinguish two cases:

1. **Attachment point hit** (within HIT_RADIUS of a specific point) — shows the pivot menu
2. **Body area hit** (near a body line segment but NOT within HIT_RADIUS of any point) — shows the body menu

Priority: attachment point beats body area. If both match, the pivot menu is shown.

`ContextMenuTarget` gains a new field:

```rust
pub struct ContextMenuTarget {
    pub joint_id: Option<String>,
    /// Body attachment point under cursor (body_id, point_name). Takes priority over body_area.
    pub attachment_point: Option<(String, String)>,
    /// Body area under cursor (body_id) — only set when no attachment point is within HIT_RADIUS.
    pub body_area: Option<String>,
    pub world_pos: Option<[f64; 2]>,
}
```

The existing `body: Option<(String, String)>` field is replaced by the two fields above.

### Right-click body area (near body edge, not near a specific point)

- Add Pivot Here
- Delete Body

### Right-click attachment point (within HIT_RADIUS of a specific point)

- Create Joint (starts two-click flow)
- Delete Pivot (removes attachment point, cascades to joints referencing it)
- Set as Driver (only shown if this point belongs to a grounded revolute joint)

### Right-click joint

- Delete Joint
- Set as Driver (only shown/enabled for grounded revolute joints)

### Right-click empty canvas

- Add Ground Pivot Here
- Start Body Here (switches to Add Body mode, seeds first point at click position)

---

## Feature A: Add Pivot Here

### Behavior

- Adds a new named attachment point to the clicked body
- Click position (world coordinates) is converted to body-local coordinates using the body's current pose (x, y, theta) from q: `local = R(-theta) * (world - body_origin)`
- The pivot can be placed anywhere in body-local space, not restricted to the body's outline/edges (a rigid body is a 2D plate, not just its rendered edges)
- Auto-generates point name: next available letter (A, B, C, ...) not already used on that body. After Z, continues with AA, AB, ... (see "Point naming overflow" below)
- Does NOT apply to the ground body. Ground pivots are added via the existing "Add Ground Pivot Here" empty-canvas context menu action. Ground has no rendered body area to right-click on, so the "Add Pivot Here" menu entry is never shown for ground.
- Snaps to grid if grid snapping is enabled
- Pushes undo

### World-to-body-local conversion

```
dx = world_x - body_x
dy = world_y - body_y
local_x =  cos(theta) * dx + sin(theta) * dy
local_y = -sin(theta) * dx + cos(theta) * dy
```

---

## Feature B: Add Body Tool

### Entry

Click `[+ Body]` toolbar button, or right-click empty canvas -> "Start Body Here".

### Click-to-place flow

1. **First click** — places point A. Green dot appears at click position. Preview text: "Click to add points, double-click or Enter to finish".
2. **Subsequent clicks** — places points B, C, D, ... Green dots at each, green dashed polyline connecting them in placement order.
3. **Finish** — double-click or Enter finishes with current points. Minimum 2 points required; if fewer than 2 points placed and user presses Enter/double-clicks, status bar shows "Need at least 2 points" and stays in mode.
4. **Esc** — cancels entirely, discards all placed points, returns to Select.
5. **Switching tools** (clicking another toolbar button) — discards all placed points, clears `add_body_state` to `None`.

**Double-click handling (egui detail):** In egui, a double-click fires both `clicked()` and `double_clicked()` on the same frame. The event handler MUST check `double_clicked()` BEFORE `clicked()`. On a double-click frame: treat it as "finish with current points" — do NOT place an additional point. This means double-click has the same semantics as Enter: it finalizes the body using the points already placed, without adding a new one at the double-click location.

### "Start Body Here" context menu entry

Switches to Add Body mode AND seeds point A at the right-click world position. User sees one green dot already placed, continues clicking for B, C, etc.

### Point naming

Auto-generated alphabetically: A, B, C, D, ... (matching the convention used by `make_bar` and `make_ternary` in samples). After Z, continues with AA, AB, ..., AZ, BA, ... (see "Point naming overflow" below).

### Coordinate system

All placed points are stored as body-local coordinates with the first point as origin:
- Point A -> local (0, 0)
- Points B, C, ... -> local offset from A: `(world_x - A_world_x, world_y - A_world_y)`

The body's initial pose in q will be `(A_world_x, A_world_y, 0)` — body origin at A's world position, theta=0. This matches the `make_bar` convention where p1 is always at local (0,0).

### Mass properties (defaults)

- `mass`: 1.0 kg
- `cg_local`: centroid of all placed points
- `izz_cg`: 0.01 kg*m^2
- Editable via property panel after creation

### Preview rendering

- Placed points: green filled circles (same size as joint dots)
- Connecting lines: green dashed polyline in placement order (A->B->C->...)
- For 3+ points: close the polygon (draw C->A) with a lighter dashed line to preview the body shape
- Current mouse position: ghost dot showing where next point would go, with dashed line from last placed point

### Drag behavior

In Add Body mode, point placement triggers on `response.clicked()`. Accidental short drags (mouse moves < 5px between mousedown and mouseup) should be treated as clicks and place a point. Longer drags have no effect — they do NOT pan the canvas. Panning is only available in Select mode (or via middle-mouse/Shift+drag, which is always available). This prevents accidental view shifts while placing body points.

### Snap behavior

- Snaps to grid if grid snapping is enabled
- Does NOT snap to existing attachment points (those belong to other bodies; use Draw Link or Create Joint to connect)
- Does NOT snap to existing body edges (that is feature C's job)

### State tracking

```rust
pub struct AddBodyState {
    /// Points placed so far: (name, world_position).
    pub points: Vec<(String, [f64; 2])>,
}
```

Stored as `Option<AddBodyState>` on AppState, set to `None` when not in Add Body mode.

### After body creation

- Body is created disconnected (no joints). Validation warns "Disconnected: body_N".
- User connects it via Draw Link (snap to the new body's points) or Create Joint (context menu).
- Stays in Add Body mode for chaining multiple bodies. Esc to return to Select.

---

## Feature C: Body-Aware Draw Link

Enhancement to the existing Draw Link tool where start/end points can land on an existing body's edge, automatically creating a new pivot on that body.

### Hit testing priority

When the user clicks or releases during Draw Link, targets are checked in this order:

1. **Existing attachment point** (within HIT_RADIUS) -> snap to it, use existing point (current behavior)
2. **Body line segment** (within ~8px perpendicular distance) -> create new pivot on that body at nearest point on segment
3. **Empty canvas** -> create ground pivot (current behavior for start) or free endpoint (current behavior for end)

Priority 1 beats 2. If you are near an existing point, it snaps there, not to the edge.

### Line segment hit detection

For each moving body, the canvas already computes screen positions of all attachment points sorted by name. The body is drawn as line segments between consecutive sorted points. Hit detection projects the click position onto each segment and checks perpendicular distance.

For bodies with **3+ points**: test the **closed polygon edges** (A->B, B->C, C->A). For bodies with **exactly 2 points**: test only the single open segment (A->B). Do NOT close 2-point bodies into a degenerate A->B->A loop.

### What happens on a segment hit

1. Compute the nearest point on the segment in world coordinates
2. Convert to body-local coordinates using `R(-theta) * (world - body_origin)`
3. Auto-generate point name (next unused letter on that body)
4. Add the attachment point to the body's blueprint
5. Proceed with normal Draw Link: create the new bar, joint from the new pivot to the new bar's endpoint

This is functionally identical to "Add Pivot Here" + "start/end Draw Link at that pivot" compressed into one gesture.

### Preview feedback

While dragging in Draw Link mode:
- Near an existing attachment point: green highlight circle (existing behavior)
- Near a body segment: a distinct indicator at the projected segment point (e.g., rotated square drawn via `Shape::convex_polygon` — egui has no built-in diamond primitive) + highlight the body edge in green. Must be visually distinguishable from the attachment-point snap circle.
- Neither: normal cursor, no highlight

### Edge cases

- **Ground body segments:** Ground has no rendered line segments (drawn as markers at each pivot). Body-segment snapping does NOT apply to ground.
- **Minimum segment distance:** If the projected point falls outside the segment endpoints (past the tips), do not snap.
- **Body with 1 attachment point:** No line segments exist, so segment snapping is impossible. Only point snapping works.

### Undo behavior

The whole operation (add pivot to existing body + create new bar + create joints) is one undo step. This requires bypassing the per-method `push_undo()` calls in individual mutation methods. See "Undo batching for compound operations" below.

---

## Undo Batching for Compound Operations

The existing mutation methods (`add_body`, `add_revolute_joint`, `add_ground_pivot`) each call `push_undo()` and `rebuild()` independently. This works for standalone actions (context menu, etc.) but creates multiple undo steps for compound gestures like Draw Link (which calls add_ground_pivot + add_body + add_revolute_joint in sequence).

**Note:** The existing Draw Link implementation in canvas.rs currently calls each public mutation method independently, creating 2-3 undo steps per gesture. This is a pre-existing bug. Refactoring it to use the `_raw` pattern is required as part of this change, not optional.

### Solution: raw blueprint mutation helpers

Each mutation method that currently pushes undo + rebuilds gains a `_raw` variant (or internal helper) that mutates only the blueprint without pushing undo or rebuilding:

```rust
// Public API (standalone use — context menu, toolbar actions):
pub fn add_body_with_points(&mut self, points: &[(String, [f64; 2])]) {
    self.push_undo();
    self.add_body_with_points_raw(points);
    self.rebuild();
}

// Internal helper (compound use — Draw Link gesture):
fn add_body_with_points_raw(&mut self, points: &[(String, [f64; 2])]) {
    // Mutate blueprint only. No push_undo, no rebuild.
}
```

Compound operations then follow this pattern:

```
push_undo()          // one snapshot
mutate_raw(...)      // N blueprint mutations, no undo, no rebuild
mutate_raw(...)
rebuild()            // one rebuild at the end
```

This applies to:
- **Draw Link** (existing): push_undo, add_ground_pivot_raw (if needed), add_body_raw, add_revolute_joint_raw (1-2x), rebuild
- **Draw Link with segment snap** (Feature C): push_undo, add_attachment_point_local_raw, add_body_raw, add_revolute_joint_raw (1-2x), rebuild
- **Standalone actions** (Add Pivot Here, Create Joint, Delete, etc.): continue using the public methods that push undo + rebuild individually

### Which methods need `_raw` variants

| Method | Needs `_raw`? | Used in compound operations? |
|--------|:------------:|------------------------------|
| `add_body_with_points` | Yes | Draw Link |
| `add_revolute_joint` | Yes | Draw Link, Create Joint (compound only when part of Draw Link) |
| `add_ground_pivot` | Yes | Draw Link (start on empty space) |
| `add_attachment_point_to_body` (raw: `add_attachment_point_local_raw`) | Yes | Draw Link segment snap |
| `remove_body` | No | Standalone only |
| `remove_joint` | No | Standalone only |
| `remove_attachment_point` | No | Standalone only |

---

## Rebuild and State Vector Reset

All mutation operations that call `rebuild()` reset the mechanism state vector `q` to the initial pose (solved from the constraint equations at driver angle 0). This is existing behavior — it applies to all editor operations including the new ones. If the user is mid-animation when editing, the mechanism snaps to its rest configuration after the edit. This is acceptable for an editor workflow and matches the current behavior of drag-to-move, add joint, etc.

---

## Create Joint — Valid Pairings

### Valid

- Body attachment point <-> Body attachment point (different bodies)
- Body attachment point <-> Ground attachment point
- Ground attachment point <-> Body attachment point

### Invalid (blocked with feedback)

- Same body's points (self-joint is degenerate)
- Two ground points (ground-ground constraint is degenerate)
- Duplicate joint (exact same pairing already exists)

### Behavior

- First click highlights point with green ring (existing rendering code in canvas.rs)
- Second click on valid target: creates revolute joint, exits two-click mode
- Second click on invalid target: status bar message explaining why, stays in two-click mode
- Esc cancels and returns to Select mode
- Multiple joints per attachment point IS allowed (valid for 3+ bodies meeting at one pivot)

**Click priority during two-click mode:** While `creating_joint` is `Some`, primary clicks on the canvas are consumed by the joint creation handler and do NOT propagate to the selection handler. The active toolbar mode does not change — `creating_joint` is an overlay sub-state, not a toolbar mode. It can be active in any toolbar mode (typically triggered from Select mode's context menu). The joint creation handler checks for `creating_joint.is_some()` before the selection/tool handlers in the click event chain.

---

## Point Naming Overflow

Auto-generated point names use single letters A through Z. After Z is exhausted, names continue as AA, AB, ..., AZ, BA, BB, ..., ZZ, AAA, etc. This is a theoretical concern — practical mechanisms rarely exceed 6-8 attachment points — but the implementation should handle it gracefully rather than panic.

Implementation: `next_attachment_point_name` iterates through the sequence and returns the first name not already present in the body's attachment_points map.

---

## Code Changes

### state.rs — New types

```rust
/// Tracks placement state for the Add Body tool.
pub struct AddBodyState {
    /// Points placed so far: (name, world_position).
    pub points: Vec<(String, [f64; 2])>,
}

/// Result of a body-segment hit test.
pub struct SegmentHit {
    /// Body that owns the segment.
    pub body_id: String,
    /// Projected point on the segment in world coordinates.
    pub world_pos: [f64; 2],
    /// Screen position of the projected point.
    pub screen_pos: Pos2,
    /// Name of the first attachment point forming the segment.
    pub point_a_name: String,
    /// Name of the second attachment point forming the segment.
    pub point_b_name: String,
}
```

### state.rs — Modified types

`ContextMenuTarget` is extended to distinguish body-area from attachment-point hits:

```rust
pub struct ContextMenuTarget {
    pub joint_id: Option<String>,
    /// Attachment point under cursor (body_id, point_name).
    /// Takes priority over body_area when both could match.
    pub attachment_point: Option<(String, String)>,
    /// Body area under cursor (body_id) — set only when no
    /// attachment point is within HIT_RADIUS but cursor is
    /// near a body line segment.
    pub body_area: Option<String>,
    pub world_pos: Option<[f64; 2]>,
}
```

The existing `body: Option<(String, String)>` field is replaced by `attachment_point` and `body_area`.

### state.rs — New/modified fields on AppState

```rust
// EditorTool enum gains AddBody variant:
//   Select, DrawLink, AddBody, AddGroundPivot

// New field:
pub add_body_state: Option<AddBodyState>,
```

### state.rs — New methods

**Standalone methods** (push undo + rebuild individually):

| Method | Purpose |
|--------|---------|
| `add_body_with_points(&mut self, points: &[(String, [f64; 2])])` | Creates body from N world-coordinate points. First point becomes local origin, others offset from it. Computes centroid for CG. Pushes undo, rebuilds. |
| `add_attachment_point_to_body(&mut self, body_id: &str, name: &str, world_x: f64, world_y: f64)` | Converts world->body-local using current pose from q. Adds point to blueprint. Pushes undo, rebuilds. |
| `remove_attachment_point(&mut self, body_id: &str, point_name: &str)` | Removes pivot from body. Cascades: removes any joints that reference this point. Pushes undo, rebuilds. |
| `next_attachment_point_name(&self, body_id: &str) -> String` | Returns next unused letter (A, B, ..., Z, AA, AB, ...) for a body. |
| `world_to_body_local(&self, body_id: &str, world_x: f64, world_y: f64) -> [f64; 2]` | Converts world coordinates to body-local using current pose from q. |

**Raw helpers** (blueprint mutation only — no undo, no rebuild):

All `_raw` methods operate on the blueprint's coordinate system (body-local coordinates). The caller is responsible for converting world coordinates to body-local before calling `_raw` methods. This is explicit in the naming — "raw" means "direct blueprint access, no convenience conversions."

| Method | Coordinates | Purpose |
|--------|------------|---------|
| `add_body_with_points_raw(&mut self, points: &[(String, [f64; 2])])` | **World** (same as public — first point becomes local origin internally) | Blueprint-only body creation. |
| `add_revolute_joint_raw(&mut self, body_i: &str, point_i: &str, body_j: &str, point_j: &str)` | N/A (references existing points by name) | Blueprint-only joint creation. |
| `add_ground_pivot_raw(&mut self, name: &str, x: f64, y: f64)` | **World** (ground local = world) | Blueprint-only ground pivot creation. |
| `add_attachment_point_local_raw(&mut self, body_id: &str, name: &str, local_x: f64, local_y: f64)` | **Body-local** (caller must convert world->local first) | Blueprint-only attachment point addition. |

Note: `add_attachment_point_local_raw` is deliberately named with `_local_` to make the coordinate system explicit in the name, following the project preference for explicit over clever.

Compound operations (Draw Link, Draw Link with segment snap) call `push_undo()` once, then use `_raw` helpers, then call `rebuild()` once.

**Feature C compound operation example:**

```
push_undo()
let local = world_to_body_local(body_id, world_x, world_y)
add_attachment_point_local_raw(body_id, name, local[0], local[1])
add_body_with_points_raw(new_body_points)
add_revolute_joint_raw(body_id, name, new_body_id, "A")
rebuild()
```

### state.rs — Modified method

`add_body()` generalized from 2-point to N-point signature via `add_body_with_points`. The old 2-point version is removed.

### canvas.rs — Changes

| Area | Change |
|------|--------|
| Body rendering | Close polygon for 3+ point bodies (draw last->first segment). 2-point bodies remain as a single line segment. |
| Draw Link hit testing | Add segment-distance check at priority 2. New `find_nearest_segment()` helper. Segments are closed for 3+ point bodies, open for 2-point bodies. |
| Draw Link preview | Distinct indicator (rotated square via `Shape::convex_polygon`) + edge highlight when near body segment |
| Add Body mode | New rendering block: placed green dots, connecting dashed polyline, ghost dot at cursor, closed polygon preview for 3+ points |
| Add Body interaction | Click to place. `double_clicked()` checked BEFORE `clicked()` to avoid double-placement. Enter to finish. Esc to cancel. Tool switch clears state. |
| Context menu hit testing | Extended to distinguish attachment_point (within HIT_RADIUS of a point) from body_area (near a body segment but not near any point). Priority: attachment_point > body_area > empty canvas. |
| Context menu content | The existing single `else if let Some(body)` branch in `response.context_menu()` is split into two separate branches: `else if let Some(attachment_point)` showing "Create Joint", "Delete Pivot", "Set as Driver" (if valid); and `else if let Some(body_area)` showing "Add Pivot Here", "Delete Body". The joint branch and empty-canvas branch are updated to match the Context Menus section. |
| Delete/Set Driver | Remove from toolbar click handlers. Delete via Del/Backspace key handler. |

### mod.rs (toolbar) — Changes

Remove Delete and Set Driver buttons. Add `[+ Body]` selectable label. Wire Del/Backspace keyboard shortcut for delete. Switching away from Add Body mode clears `add_body_state`.

### property_panel.rs — Changes

| Change | Detail |
|--------|--------|
| Set Driver button | Show "Set as Driver" button when a grounded revolute joint is selected |
| Attachment point list | Show list of attachment points on selected body, each with coordinates |

### Canvas helpers

```rust
/// Convert world coordinates to body-local coordinates.
fn world_to_body_local(
    state: &State, body_id: &str, q: &DVector<f64>,
    world_x: f64, world_y: f64,
) -> [f64; 2]

/// Enumerate body line segments as (screen_p1, screen_p2, body_id, point_name_1, point_name_2).
/// For bodies with 3+ attachment points, includes the closing segment (last -> first).
/// For bodies with exactly 2 points, includes only the single open segment.
fn collect_body_segments(...) -> Vec<(Pos2, Pos2, String, String, String)>

/// Find the nearest body line segment to a screen point.
/// Projects the point onto each segment, rejects projections that fall
/// outside segment endpoints, and returns the closest within max_distance.
fn find_nearest_segment(
    point: Pos2,
    segments: &[(Pos2, Pos2, String, String, String)],
    max_distance: f32,
) -> Option<SegmentHit>
```
