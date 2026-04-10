# DXF Import: Snappable Overlay + Interactive Body Assignment

**Date:** 2026-04-09
**Status:** Approved

---

## Problem

Users design linkage mechanisms in SolidWorks and need to transfer them into the simulator. The current workflow is either manual JSON construction or screenshot tracing with the image overlay. Both are slow and error-prone, especially for complex mechanisms with ternary links, actuator attachment points, and force zone geometry.

SolidWorks sketch exports produce flat DXF files (all entities on one default layer) containing LINE, ARC, and CIRCLE entities. The tool needs to import these and let the user interactively assign geometry to mechanism elements.

## Solution

A two-mode DXF import pipeline:

1. **Snappable overlay (Mode C):** Import DXF geometry as a vector overlay on the canvas. Circle centers become snap targets for existing GUI tools (+Body, Draw Link, +Ground). Works immediately after import.

2. **Interactive body assignment (Mode B):** A sidebar panel for grouping DXF entities into bodies. Click entities to select them into the current body group. Circles at shared positions between bodies become joints. Then assign ground pivots and driver.

Both modes coexist: the overlay persists on canvas while the assignment panel is open. Users can mix approaches -- use existing tools for simple links, use the assignment panel for ternary bodies.

---

## Architecture

### Data Model

```
DxfOverlay {
    entities: Vec<DxfEntity>,       // parsed from file
    circles: Vec<DxfCircle>,        // extracted circle centers (snap targets)
    scale: f64,                     // mm-to-m conversion (default 0.001)
    offset: [f64; 2],               // world offset for positioning
    visible: bool,                  // toggle overlay rendering
    opacity: f32,                   // overlay opacity (0.0-1.0)
    assignments: Vec<BodyAssignment>, // Mode B groupings
    selected_entities: HashSet<usize>, // currently highlighted entities
}

DxfEntity {
    kind: DxfEntityKind,            // Line, Arc, Circle, Point
    coords: Vec<[f64; 2]>,         // start/end for lines, center for circles
    radius: Option<f64>,            // for circles/arcs
    start_angle: Option<f64>,       // for arcs
    end_angle: Option<f64>,         // for arcs
    index: usize,                   // entity index for selection
}

DxfCircle {
    center: [f64; 2],              // world coords (after scale+offset)
    radius: f64,
    entity_index: usize,           // back-reference to parent entity
}

BodyAssignment {
    name: String,                  // body ID (auto-generated or user-named)
    entity_indices: Vec<usize>,    // which DXF entities belong to this body
    circle_indices: Vec<usize>,    // which circles are attachment points
    is_ground: bool,               // true for the ground body
}
```

### DXF Parser

**Crate:** Use the `dxf` crate (https://crates.io/crates/dxf) for reading. It handles full DXF files including TABLES/HEADER sections that SolidWorks exports.

**Supported entities:**
- `LINE` -> two endpoints
- `CIRCLE` -> center + radius (joint candidates)
- `ARC` -> center + radius + start/end angles
- `LWPOLYLINE` -> vertex list (tessellate into line segments)
- `POINT` -> single coordinate

**Unit handling:** SolidWorks DXF exports are typically in millimeters. The parser applies a configurable scale factor (default 0.001 for mm->m). A "Units" dropdown in the import UI offers mm, cm, m, inches.

### Canvas Rendering

DXF overlay renders in a dedicated layer between the grid and the mechanism:

- **Lines/arcs:** thin gray strokes (distinguishable from mechanism links)
- **Circles:** gray outline + small filled dot at center (snap target)
- **Selected entities** (during assignment): highlighted in blue
- **Assigned entities:** colored per-body (same palette as mechanism bodies)
- **Snap targets:** when the user hovers near a DXF circle center while using Draw Link or +Body, the cursor snaps to the exact center position (same snap radius as grid snap)

### Snap Integration

The existing canvas interaction code has snap logic for grid points and alignment guides. Add DXF circle centers as an additional snap source:

- In the hit-testing/snap code, check proximity to each `DxfCircle.center`
- DXF snap takes priority over grid snap (exact geometry > grid)
- Visual feedback: green ring around the snap target (same as existing joint snap)

### Assignment Panel (Mode B)

A collapsible sidebar section "DXF Import" that appears when a DXF overlay is loaded:

**Controls:**
- **Scale:** dropdown (mm/cm/m/inches) + custom multiplier
- **Offset X/Y:** drag values for positioning the overlay
- **Opacity:** slider (0.0-1.0)
- **Visible:** checkbox

**Assignment workflow:**
1. Click "New Body" button -> enters selection mode
2. Click DXF entities (lines/circles) on canvas to add them to the current body group (highlighted in blue)
3. Shift+click to deselect
4. Click "Finish Body" to commit the group (auto-named "body_1", "body_2", etc.)
5. Repeat for each body
6. Click "Mark as Ground" on any committed body to flag it
7. Click "Build Mechanism" to:
   a. Create bodies from assigned entity groups
   b. For each body: circles become attachment points, lines define the body shape
   c. Circles at matching positions across two bodies become revolute joints
   d. Generate MechanismJson and load via `load_from_json_str`

**Joint detection:** Two circles from different bodies whose centers are within a tolerance (default 0.5mm after scaling) are treated as the same joint location. A revolute joint is created connecting those bodies at that point.

**Attachment point coordinates:** For each body, the first circle becomes the body origin (local 0,0). Other circles/points are stored as offsets in the body's local frame (rotated so the first-to-second circle defines the local X axis, unless the user overrides).

### File Menu Integration

Add to the File menu (next to "Open JSON..."):

```
Import DXF...    (opens file dialog, filter *.dxf)
```

On import:
1. Open file dialog with DXF filter
2. Parse DXF file
3. Create DxfOverlay and store in AppState
4. Open the "DXF Import" sidebar panel
5. Status message: "DXF loaded: {n} lines, {m} circles. Use tools to build or assign bodies."

### State Management

```rust
// In AppState:
pub dxf_overlay: Option<DxfOverlay>,
```

The overlay persists across mechanism rebuilds. Cleared explicitly via "Clear DXF Overlay" button in the assignment panel, or when loading a new JSON/sample.

---

## Scope

### In Scope
- DXF file parsing (LINE, CIRCLE, ARC, LWPOLYLINE, POINT)
- Vector overlay rendering on canvas with configurable scale/offset/opacity
- Snap-to-circle-center during Draw Link, +Body, +Ground, +Joint Point operations
- Interactive body assignment panel with entity selection
- Auto-joint detection from shared circle positions
- MechanismJson generation from assignments
- Unit selection (mm/cm/m/inches)
- Native file dialog (rfd) for DXF file selection

### Out of Scope
- WASM file import (rfd file dialogs are native-only; can add drag-drop later)
- DXF BLOCK/INSERT entity support (uncommon in sketch exports)
- Automatic body detection without user assignment
- DXF writing/re-export
- 3D DXF support (Z coordinates ignored, projected to XY plane)
- SPLINE entity support (uncommon in mechanism sketches)
- Preserving DXF layers (SolidWorks sketch exports are typically single-layer)

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `Cargo.toml` | Add `dxf` crate dependency |
| `src/gui/import/mod.rs` | New module: DXF parser + overlay data model |
| `src/gui/import/dxf_parser.rs` | New: parse DXF file into DxfOverlay |
| `src/gui/import/dxf_panel.rs` | New: assignment panel UI (sidebar) |
| `src/gui/import/dxf_render.rs` | New: canvas overlay rendering |
| `src/gui/import/dxf_snap.rs` | New: snap integration for circle centers |
| `src/gui/import/dxf_build.rs` | New: convert assignments to MechanismJson |
| `src/gui/state/mod.rs` | Add `dxf_overlay: Option<DxfOverlay>` field |
| `src/gui/mod.rs` | Add "Import DXF..." menu item, wire up panel |
| `src/gui/canvas/rendering.rs` | Call DXF overlay renderer |
| `src/gui/canvas/interaction.rs` | Add DXF snap to hit-testing |

---

## Error Handling

- **Invalid DXF file:** Status message "Failed to parse DXF: {error}". No overlay created.
- **Empty DXF:** Status message "DXF contains no geometry entities."
- **No circles found:** Warning in assignment panel "No circles found -- joint positions must be placed manually."
- **Ambiguous joint matches:** If a circle center is within tolerance of circles in 3+ bodies, flag it in the panel for user resolution.
- **Zero-length body:** If a body assignment contains only circles and no lines, warn but allow (point-mass body).

---

## Testing

- Unit tests for DXF parser: parse a minimal DXF string with LINE/CIRCLE/ARC, verify entity extraction
- Unit tests for joint detection: two circles at same position -> joint created; circles 0.3mm apart -> joint; circles 2mm apart -> no joint
- Unit tests for coordinate transform: body-local from world coordinates
- Integration test: parse a sample DXF, assign bodies, build mechanism, verify it solves
- Include a test DXF file in `linkage-sim-rs/data/test/` for regression testing
