# Actuator Stroke Limits

**Date**: 2026-03-24

## Problem

LinearActuatorElement has no way to limit its extension range. Real hydraulic/pneumatic cylinders have physical stroke limits — the piston bottoms out at min retraction and max extension. Without these limits, the simulated actuator can extend/compress infinitely, producing unrealistic behavior.

## Design

### New Fields on `LinearActuatorElement`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stroke_min` | `f64` | `0.0` | Minimum distance between attachment points (m) |
| `stroke_max` | `f64` | `0.0` | Maximum distance between attachment points (m) |
| `end_stop_stiffness` | `f64` | `10000.0` | Penalty spring stiffness at limits (N/m) |
| `end_stop_damping` | `f64` | `10.0` | Penalty damper at limits (N*s/m) |
| `end_stop_restitution` | `f64` | `0.5` | Coefficient of restitution [0,1] |

**Activation rule:** Stroke limits are active when `stroke_max > 0` and `stroke_max > stroke_min`. Each limit is checked independently:
- Min stop fires only when `stroke_min > 0` and `L < stroke_min`
- Max stop fires only when `stroke_max > 0` and `L > stroke_max`

This allows valid configurations like `stroke_min = 0, stroke_max = 2.0` (max-only limit) or `stroke_min = 0.5, stroke_max = 2.0` (both limits). When both are 0.0 (default), no limits are applied — backward compatible with existing files via `#[serde(default)]`.

### Force Law

In `evaluate_linear_actuator`, after computing the normal actuator force:

1. Compute current length: `L = |P_b - P_a|` (world-frame distance between attachment points).
2. If `L < 1e-15` (degenerate zero-length): skip penalty logic, return early (same as existing early return for normal force — unit vector is undefined).
3. Compute relative velocity along actuator axis: `v_rel = unit.dot(v_b - v_a)` (positive = extending).
4. If `stroke_min > 0` and `L < stroke_min` (over-retracted):
   - Penetration: `delta = stroke_min - L`
   - Velocity is "into limit" when `v_rel < 0` (still retracting)
   - Damping coefficient: `c = end_stop_damping` when into limit, `c = end_stop_damping * end_stop_restitution` when bouncing away
   - Signed penalty force along unit vector (positive = push apart):
     `F_penalty = +(end_stop_stiffness * delta - c * v_rel)`
5. If `stroke_max > 0` and `L > stroke_max` (over-extended):
   - Penetration: `delta = L - stroke_max`
   - Velocity is "into limit" when `v_rel > 0` (still extending)
   - Damping coefficient: same asymmetric logic
   - Signed penalty force along unit vector (negative = pull together):
     `F_penalty = -(end_stop_stiffness * delta + c * v_rel)`
6. Apply penalty force on body B as `F_penalty * unit`, on body A as `-F_penalty * unit`, added to the normal actuator force contributions.

The actuator force is NOT cut at limits — it continues pushing against the mechanical stop, just as a real cylinder does. The penalty force is the only thing preventing further motion.

This matches the sign convention in `evaluate_joint_limit` where the max-limit formula uses `-(stiffness * penetration + damp * omega_rel)` with the negation baked into the expression.

### Serialization

All new fields use `#[serde(default)]` with serde default helper functions:
- `stroke_min` → `0.0` (plain serde default, no helper needed)
- `stroke_max` → `0.0` (plain serde default, no helper needed)
- `end_stop_stiffness` → new helper `default_end_stop_stiffness()` returning `10000.0`
- `end_stop_damping` → new helper `default_end_stop_damping()` returning `10.0`
- `end_stop_restitution` → reuse existing `default_restitution()` returning `0.5` (DRY)

Old JSON files without these fields load correctly with limits disabled.

### UI Changes

In `gui/property_panel/force_editor.rs`, the Linear Actuator section gains a collapsible "Stroke Limits" sub-section:

- **Min stroke** — DragValue, range [0, inf), step 0.001 m, suffix from display units
- **Max stroke** — DragValue, range [0, inf), step 0.001 m, suffix from display units
- **Stiffness** — DragValue, range [0, inf), step 100 N/m
- **Damping** — DragValue, range [0, inf), step 1.0 N*s/m
- **Restitution** — DragValue, range [0, 1], step 0.01

Validation: if the user sets `stroke_min >= stroke_max` and both are nonzero, show a warning label "(min >= max, limits inactive)" in the UI. The evaluation code handles this gracefully (each limit checked independently), so no hard clamp is needed.

When both min and max are 0, show "(disabled)" hint text. When `stroke_min == 0` but `stroke_max > 0`, show "(inactive, set > 0 to enable)" next to the min stroke field so the user knows the min stop is off.

### Canvas Rendering

When stroke limits are active, render two small tick marks perpendicular to the actuator line at the min and max stroke positions from point A. This provides visual feedback showing the allowed travel range.

### Sweepable Parameters

Add `stroke_min`, `stroke_max`, `end_stop_stiffness`, `end_stop_damping`, and `end_stop_restitution` to both `set_force_field` and `force_sweepable_fields` in `blueprint_ops.rs`. All five are useful for sensitivity analysis (e.g., "how does end-stop stiffness affect peak reaction forces?").

## Files Changed

| File | Change |
|------|--------|
| `forces/elements/element_types.rs` | Add 5 fields to `LinearActuatorElement` + serde default helpers |
| `forces/elements/evaluation.rs` | Add penalty force logic to `evaluate_linear_actuator` |
| `gui/property_panel/force_editor.rs` | Add "Stroke Limits" UI section with validation warning |
| `gui/canvas/rendering.rs` | Add tick marks at stroke limits on actuator drawing |
| `gui/state/blueprint_ops.rs` | Add 5 new fields to `set_force_field` and `force_sweepable_fields` |

## Testing

- Unit test: penalty force is zero when within limits
- Unit test: penalty force pushes apart (positive) when L < stroke_min
- Unit test: penalty force pulls together (negative) when L > stroke_max
- Unit test: actuator force still applied at limits (not cut)
- Unit test: asymmetric damping (more dissipation entering limit, less leaving)
- Unit test: limits disabled when stroke_min=0, stroke_max=0
- Unit test: limits disabled when stroke_min == stroke_max (degenerate equal case)
- Unit test: max-only limit works with stroke_min=0, stroke_max > 0
- Unit test: zero-length actuator returns zero (no penalty panic/NaN)
- Round-trip serialization test with stroke limit fields
