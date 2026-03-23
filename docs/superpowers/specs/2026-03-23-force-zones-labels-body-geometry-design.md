# Force Zones, Body Geometry, Labels & Crank Angle Limits

**Date**: 2026-03-23
**Status**: Approved

## Overview

Four interconnected features for the linkage simulation GUI:

1. **Body geometry** — visual rectangular shapes attached to existing bodies/links
2. **Force zones** — world-space rectangular regions that apply constant distributed forces when a body's geometry overlaps the zone
3. **Labels & tooltips** — always-visible toggleable labels on all elements, plus hover tooltips with properties
4. **Crank angle limits** — constrain sweep analysis range and enforce physical joint limits in forward dynamics

A new default sample ("Parallelogram Press") demonstrates all features together.

## Data Model

### Body Geometry

New field on `Body`:

```rust
pub struct BodyGeometry {
    pub width: f64,            // meters, extent along body's local x-axis
    pub height: f64,           // meters, extent along body's local y-axis
    pub offset: Vector2<f64>,  // local-frame offset from body origin (default [0,0])
}

// Added to Body:
pub geometry: Option<BodyGeometry>,
pub label: String,  // user-editable display name (e.g., "Coupler", "Press Plate")
```

Bodies without geometry continue rendering as the existing blue bar. Bodies with geometry render the rectangle in addition to the bar.

### Joint/Constraint Labels

```rust
// On joint serialization:
pub label: Option<String>,  // e.g., "J1", "Crank Pivot"
```

Auto-generated defaults: revolute joints get "R1", "R2"; prismatic get "P1"; etc. User-editable.

### ForceZone — New ForceElement Variant

```rust
ForceZone {
    body_id: String,           // which body's geometry to test overlap against
    zone_min: Vector2<f64>,    // world-space bottom-left corner (meters)
    zone_max: Vector2<f64>,    // world-space top-right corner (meters)
    force: Vector2<f64>,       // constant force vector (N), e.g., [0, -500] for downward
    label: Option<String>,     // e.g., "Press Zone"
}
```

Force applied = `force * (overlap_area / body_total_area)`. Full force when body is fully inside the zone, proportional ramp on entry/exit.

Key decisions:
- `zone_min`/`zone_max` are in **world space** (fixed in the ground frame)
- `body_id` references the body whose `geometry` is tested for overlap
- Bodies without geometry cannot be targeted by a ForceZone (validation error)

### Crank Angle Limits

```rust
// On driver or mechanism-level config:
pub sweep_min: Option<f64>,  // radians
pub sweep_max: Option<f64>,  // radians
```

When set, a `JointLimit` penalty force element is auto-created on the driver joint so forward dynamics also respects the bounds.

## Solver Integration

### Overlap Computation

The ForceZone `apply()` method:

1. **Transform body geometry to world frame**: Use the body's `(x, y, θ)` from state vector `q` to compute the 4 corners of the body rectangle in world coordinates.

2. **Compute intersection area**: Clip the rotated body rectangle against the axis-aligned zone rectangle using Sutherland-Hodgman algorithm. Compute area via shoelace formula.

3. **Compute force scaling**: `ratio = intersection_area / body_total_area`. Continuous function — 0.0 outside, 1.0 fully inside, smooth ramp in between.

4. **Apply to Q**: The applied force `F = force * ratio` acts at the **centroid of the overlap region** (not the body CG). This creates physically correct moments — a body entering edge-first experiences torque. Convert to generalized forces using existing `point_force_to_q()` helper.

### Smoothness

The overlap area is continuous in `(x, y, θ)`. Derivative has kinks at corner entry/exit but these are mild enough for Newton-Raphson and the DAE integrator. No special treatment needed.

### Reaction Forces

Because ForceZone contributes to Q, inverse dynamics automatically computes increased driver torque, and statics gives updated joint reactions. No extra work — falls out of `Φ_q^T * λ = Q - M*q_ddot`.

### Crank Angle Limits — Solver Behavior

- **Sweep/analysis**: kinematics solver only evaluates from θ_min to θ_max
- **Forward dynamics**: `JointLimit` penalty force enforces bounds; animation stops/bounces at limits instead of wrapping
- **Plots**: x-axis range adjusts to the limited sweep range

## Canvas Rendering

### Body Geometry
- Filled/stroked rectangle: `rgba(255,165,0,0.25)` fill, `#ffa500` stroke
- Transforms with body's `(x, y, θ)` every frame
- Drawn behind the link bar (bar remains visible as structural element)
- Selected bodies get existing orange highlight

### Force Zone
- Dashed red rectangle border (`#ff5050`, dash `8,4`)
- Faint red fill (`rgba(255,80,80,0.12)`)
- Arrows inside matching force direction, indicating magnitude
- Label above the zone
- **Active overlap highlight**: yellow tint (`rgba(255,200,0,0.2)`) on the overlap region during sweep/animation

### Labels
- Small monospace text near each element (body labels near CG, joint labels near joint, link labels at bar midpoint)
- Color: muted gray (`#888`)
- Scale with zoom, clamped to prevent unreadable or oversized text
- **View menu toggle**: "Show Labels" (default: on)
- Auto-generated defaults: "B1"/"B2" for bodies, "R1"/"P1" by joint type+index

### Hover Tooltips
- **Body/link hover**: name, mass, inertia, geometry dimensions (if set), CG position, length, connected joints
- **Joint hover**: name, type, current angle/displacement, reaction forces (if sweep data)
- **Force zone hover**: name, force vector, zone bounds, current overlap ratio
- Tooltips appear on mouse hover, disappear on mouse leave

## GUI Controls

### Body Geometry Editing (Property Panel)
- New "Geometry" collapsible section when a body is selected
- Width/Height: logarithmic sliders (1mm–500mm display), type-to-enter for exact values
- Offset X/Y: linear sliders
- Add/Remove Geometry button
- All edits through undo/redo

### Force Zone Creation
- "Force Zone" button in force toolbar
- Click and drag on canvas to define rectangular region
- Property panel opens to set force vector and target body
- Alternative: right-click canvas → "Add Force Zone" → drag to define
- Property panel: zone corners, force X/Y (with magnitude+angle alternate input), target body dropdown (filtered to bodies with geometry)

### Label Editing
- Double-click label on canvas → inline text field
- Also editable in property panel text field at top of each element's section

### Crank Angle Limits
- "Sweep Range" section in input panel (next to driver angle slider)
- Min/Max angle inputs (degrees), default 0°–360°
- "Limit sweep range" checkbox toggle
- Driver angle slider clamps to range
- Animation stops/bounces at limits

### Serialization
- `BodyGeometry`, `ForceZone`, labels, and sweep range all serialize to JSON
- Schema version bump for new fields
- Backward compatible: missing fields default to `None` / auto-generated labels / full sweep

## Default Sample: Parallelogram Press

**Linkage geometry** (same as existing parallelogram):
- Ground pivots: O2=(0,0), O4=(0.04,0) — 40mm apart
- Link lengths: crank=20mm, coupler=40mm, rocker=20mm
- Driver on crank at 2π rad/s

**Press plate** (on coupler body):
- Geometry: 60mm wide x 15mm tall rectangle, centered on coupler midpoint
- Mass: 0.5 kg
- Inertia: auto-computed from rectangular plate formula

**Force zone**:
- Bounds: approximately x=[10mm, 30mm], y=[–5mm, 10mm] (tuned so plate enters/exits during downward stroke)
- Force: [0, –500] N (500N downward)
- Label: "Press Zone"

**Crank angle limits**:
- Sweep: ~150°–210° (working stroke where coupler descends through zone)
- Demonstrates the interesting loaded region immediately

**Sample menu name**: "Parallelogram Press" — listed after existing "Parallelogram" entry

**Expected behavior**:
- Canvas shows parallelogram with orange press plate, red force zone, all labels
- Sweeping shows plate entering zone, yellow overlap growing, torque plot spiking
- Joint reaction plots show increased forces during press stroke
