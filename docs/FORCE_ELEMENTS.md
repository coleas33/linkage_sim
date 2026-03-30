# Force Element Equations Reference

This document covers all 13 force element types in the linkage simulator. Each section lists the element's parameters, governing equation, and how physical forces map to generalized coordinates.

**Source files**: `linkage-sim-rs/src/forces/elements/element_types.rs`, `evaluation.rs`, `helpers.rs`

---

## Generalized Force Mapping (Virtual Work Principle)

All force elements ultimately produce a generalized force vector **Q** with three entries per body: `Q[x]`, `Q[y]`, `Q[theta]`.

### Point force on a body

A force **F** = (Fx, Fy) applied at body-local point **s** produces:

```
Q[x]     = Fx
Q[y]     = Fy
Q[theta] = B(theta) * s . F
```

where `B(theta) * s` is the partial derivative of the global attachment point position with respect to `theta` (the moment arm).

### Pure torque on a body

A torque `tau` applied to a body produces:

```
Q[x]     = 0
Q[y]     = 0
Q[theta] = tau
```

### Two-body elements

Spring, damper, and actuator elements apply equal and opposite forces/torques on both bodies (Newton's third law), summing both contributions into **Q**.

---

## 1. Gravity

Uniform gravitational field applied to every body with mass > 0.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `g_vector` | [f64; 2] | m/s^2 | Gravity vector in global coords. Default: (0, -9.81) |

**Equation:**

```
F_i = m_i * g_vector
```

Applied at each body's center of gravity (`cg_local`). Produces both translational force and a torque if the CG is offset from the body origin.

---

## 2. Linear Spring

Translational spring between two attachment points on two bodies.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_a`, `body_b` | String | -- | Body identifiers |
| `point_a`, `point_b` | [f64; 2] | m | Attachment points in body-local coords |
| `stiffness` | f64 | N/m | Spring stiffness |
| `free_length` | f64 | m | Unstretched (free) length |

**Equation:**

```
L = |P_b - P_a|               (current length)
u = (P_b - P_a) / L           (unit vector A -> B)
F = k * (L - L_0)             (scalar spring force)

F_on_A = +u * F               (toward B when extended)
F_on_B = -u * F               (toward A when extended)
```

Returns zero if `L < 1e-15` (degenerate zero-length).

---

## 3. Torsion Spring

Rotational spring between two bodies at a revolute joint.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_i`, `body_j` | String | -- | Reference body and target body |
| `stiffness` | f64 | N*m/rad | Torsional stiffness |
| `free_angle` | f64 | rad | Relative angle at zero torque |

**Equation:**

```
theta_rel = theta_j - theta_i
tau = -k * (theta_rel - theta_free)

Applied as:  +tau on body_j,  -tau on body_i
```

---

## 4. Linear Damper

Translational viscous damper between two attachment points.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_a`, `body_b` | String | -- | Body identifiers |
| `point_a`, `point_b` | [f64; 2] | m | Attachment points in body-local coords |
| `damping` | f64 | N*s/m | Damping coefficient |

**Equation:**

```
L = |P_b - P_a|
u = (P_b - P_a) / L
dL/dt = u . (v_b - v_a)       (rate of length change)
F = -c * dL/dt                (scalar damping force)

F_on_A = +u * F
F_on_B = -u * F
```

---

## 5. Rotary Damper

Rotational viscous damper between two bodies at a revolute joint.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_i`, `body_j` | String | -- | Body identifiers |
| `damping` | f64 | N*m*s/rad | Damping coefficient |

**Equation:**

```
omega_rel = theta_dot_j - theta_dot_i
tau = -c * omega_rel

Applied as:  +tau on body_j,  -tau on body_i
```

---

## 6. External Force

Point force applied at a fixed local point on a single body, with optional time modulation.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_id` | String | -- | Target body |
| `local_point` | [f64; 2] | m | Application point in body-local coords |
| `force` | [f64; 2] | N | Force vector in global coords |
| `modulation` | TimeModulation | -- | Time-dependent scaling factor (see below) |

**Equation:**

```
F(t) = force * modulation.factor(t)
```

Applied at `local_point` on the target body.

---

## 7. External Torque

Pure torque applied to a single body, with optional time modulation.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_id` | String | -- | Target body |
| `torque` | f64 | N*m | Torque magnitude (positive = CCW) |
| `modulation` | TimeModulation | -- | Time-dependent scaling factor |

**Equation:**

```
tau(t) = torque * modulation.factor(t)
```

Directly sets `Q[theta]` for the target body.

---

## 8. Gas Spring

Pressure-based force element modeling a gas spring (gas strut). Force increases with compression following a polytropic gas law, plus optional velocity-dependent damping.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_a`, `body_b` | String | -- | Body identifiers |
| `point_a`, `point_b` | [f64; 2] | m | Attachment points in body-local coords |
| `initial_force` | f64 | N | Force at the extended (nominal) length |
| `extended_length` | f64 | m | Nominal extended length |
| `stroke` | f64 | m | Maximum compression stroke |
| `damping` | f64 | N*s/m | Velocity-dependent damping (default: 0) |
| `polytropic_exp` | f64 | -- | Polytropic exponent (default: 1.0) |

**Equation:**

```
compression = clamp(extended_length - L, 0, stroke)
gas_column  = max(stroke - compression, 1e-10)

F_gas = F_initial * (stroke / gas_column)^n
F_damp = -c * dL/dt
F_total = F_gas + F_damp
```

The force acts along the line of action (pushes apart). Direction: positive = extension.

**Special behavior:**
- `n = 1.0`: isothermal compression (Boyle's law)
- `n = 1.4`: adiabatic compression (typical for fast strokes)
- `stroke <= 0`: degenerate mode, acts as a constant-force element at `initial_force`
- `gas_column` is clamped to `1e-10` to prevent division by zero at full compression

---

## 9. Bearing Friction

Multi-component friction at a revolute joint. Combines constant drag, viscous drag, and Coulomb friction with `tanh` regularization for smooth zero-velocity behavior.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_i`, `body_j` | String | -- | Body identifiers |
| `constant_drag` | f64 | N*m | Constant drag torque (e.g., seal friction) |
| `viscous_coeff` | f64 | N*m*s/rad | Viscous drag coefficient |
| `coulomb_coeff` | f64 | -- | Coulomb friction coefficient (mu) |
| `pin_radius` | f64 | m | Effective pin radius for Coulomb term |
| `radial_load` | f64 | N | Radial load for Coulomb term |
| `v_threshold` | f64 | rad/s | Regularization threshold (default: 0.01) |

**Equation:**

```
omega_rel = theta_dot_j - theta_dot_i
direction = tanh(omega_rel / v_threshold)
magnitude = T_drag + c_vis * |omega_rel| + mu * R * F_n

tau = -magnitude * direction
```

Applied as: `+tau` on body_j, `-tau` on body_i.

**Convenience constructor** `ForceElement::coulomb_friction(...)` creates a `BearingFriction` with only the Coulomb component (`constant_drag=0`, `viscous_coeff=0`).

---

## 10. Joint Limit

Penalty-based angular limit at a revolute joint. Applies a restoring spring-damper torque when the relative angle exceeds the allowed range.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_i`, `body_j` | String | -- | Body identifiers |
| `angle_min` | f64 | rad | Minimum allowed relative angle |
| `angle_max` | f64 | rad | Maximum allowed relative angle |
| `stiffness` | f64 | N*m/rad | Penalty spring stiffness |
| `damping` | f64 | N*m*s/rad | Penalty damping coefficient (default: 0) |
| `restitution` | f64 | -- | Coefficient of restitution, 0..1 (default: 0.5) |

**Equation:**

```
theta_rel = theta_j - theta_i

If theta_rel < angle_min:
    penetration = angle_min - theta_rel
    damp = damping           (if moving into stop: omega_rel < 0)
         = damping * e       (if bouncing away:    omega_rel >= 0)
    tau = k * penetration - damp * omega_rel

If theta_rel > angle_max:
    penetration = theta_rel - angle_max
    damp = damping           (if moving into stop: omega_rel > 0)
         = damping * e       (if bouncing away:    omega_rel <= 0)
    tau = -(k * penetration + damp * omega_rel)

Otherwise: tau = 0
```

Applied as: `+tau` on body_j, `-tau` on body_i.

**Special behavior:** The restitution coefficient `e` scales the damping when the joint is rebounding away from the stop. `e=0` (perfectly inelastic) applies full damping in both directions. `e=1` (perfectly elastic) applies full damping only when moving into the stop.

---

## 11. Motor

DC motor with linear torque-speed droop at a revolute joint. Models the classic T = T_stall * (1 - omega/omega_no_load) characteristic.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_i`, `body_j` | String | -- | Stator body (typically ground) and driven body |
| `stall_torque` | f64 | N*m | Maximum torque at zero speed |
| `no_load_speed` | f64 | rad/s | Speed at zero torque |
| `direction` | f64 | -- | +1.0 = CCW, -1.0 = CW (default: +1.0) |

**Equation:**

```
omega_rel = theta_dot_j - theta_dot_i
speed_in_dir = omega_rel * direction

torque_fraction = clamp(1 - speed_in_dir / omega_no_load, 0, 1)
tau = T_stall * torque_fraction * direction
```

Applied as: `+tau` on body_j, `-tau` on body_i.

**Special behavior:**
- Clamped to `[0, 1]` so the motor never produces negative torque (no regenerative braking at overspeed) and never exceeds stall torque.
- Returns zero if `no_load_speed <= 0`.

---

## 12. Linear Actuator

Force element along the line between two body points with optional speed limiting and stroke limits with end-stop penalty forces.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_a`, `body_b` | String | -- | Body identifiers |
| `point_a`, `point_b` | [f64; 2] | m | Attachment points in body-local coords |
| `force` | f64 | N | Actuator force (positive = extension/push apart) |
| `speed_limit` | f64 | m/s | Maximum extension rate; 0 = no limit (default: 0) |
| `stroke_min` | f64 | m | Minimum stroke length; 0 = no min limit (default: 0) |
| `stroke_max` | f64 | m | Maximum stroke length; 0 = no max limit (default: 0) |
| `end_stop_stiffness` | f64 | N/m | End-stop penalty stiffness (default: 10000) |
| `end_stop_damping` | f64 | N*s/m | End-stop penalty damping (default: 10) |
| `end_stop_restitution` | f64 | -- | End-stop coefficient of restitution (default: 0.5) |

**Equation:**

```
L = |P_b - P_a|
u = (P_b - P_a) / L
v_along = u . (v_b - v_a)

-- Speed limiting --
If speed_limit > 0:
    speed_ratio = |v_along| / speed_limit
    F_actual = force * (1 - speed_ratio)   if speed_ratio < 1
             = 0                            if speed_ratio >= 1
Else:
    F_actual = force

-- Stroke limit penalties (active when stroke_max > 0 and stroke_max > stroke_min) --
If L < stroke_min:
    penetration = stroke_min - L
    damp = end_stop_damping                 (if moving into stop)
         = end_stop_damping * e             (if bouncing away)
    F_actual += end_stop_stiffness * penetration - damp * v_along

If L > stroke_max:
    penetration = L - stroke_max
    damp = end_stop_damping                 (if moving into stop)
         = end_stop_damping * e             (if bouncing away)
    F_actual -= end_stop_stiffness * penetration + damp * v_along
```

Force is applied along the unit vector between attachment points.

---

## 13. Force Zone

Spatial force field defined as an axis-aligned rectangular zone. Applies a distributed force proportional to the overlap area between the zone and the body's geometry.

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| `body_id` | String | -- | Target body (must have `BodyGeometry` set) |
| `zone_min` | [f64; 2] | m | World-space bottom-left corner of zone |
| `zone_max` | [f64; 2] | m | World-space top-right corner of zone |
| `force` | [f64; 2] | N | Force vector at full overlap |
| `label` | Option | -- | Optional display label |

**Equation:**

```
overlap_area = polygon_clip(body_rect_world, zone_AABB)
ratio = min(overlap_area / body_area, 1.0)
F = force * ratio
```

Applied at the centroid of the clipped overlap polygon (converted to body-local coordinates for the generalized force mapping).

**Special behavior:**
- Body must have `BodyGeometry` (width, height, offset). Returns zero if geometry is missing.
- The body rectangle is transformed to world space, then clipped against the axis-aligned zone using Sutherland-Hodgman polygon clipping.
- Returns zero if overlap area < `1e-15`.

---

## Time Modulation (External Force / External Torque)

Both `ExternalForce` and `ExternalTorque` support a `modulation` field that scales the base force/torque by a time-dependent factor.

| Variant | Parameters | Factor |
|---------|-----------|--------|
| `Constant` | (none) | `1.0` |
| `Sinusoidal` | `omega` (rad/s), `phase` (rad) | `sin(omega * t + phase)` |
| `Step` | `t_on` (s) | `0` if `t < t_on`, else `1` |
| `Ramp` | `t_start` (s), `t_end` (s) | Linear from 0 to 1 over `[t_start, t_end]` |
| `Expression` | `expr` (string) | Arbitrary expression of `t`, e.g. `"1 - exp(-t/0.5)"` |

The `Expression` variant uses the `meval` crate. Returns `0.0` if the expression fails to parse or produces a non-finite value.
