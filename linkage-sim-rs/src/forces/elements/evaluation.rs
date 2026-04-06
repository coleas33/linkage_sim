//! Evaluation functions for all force element types.

use std::collections::HashMap;

use nalgebra::{DVector, Vector2};

use crate::core::body::Body;
use crate::core::state::{State, GROUND_ID};
use crate::forces::helpers::{body_torque_to_q, point_force_to_q};

use super::element_types::*;

// ── Angular element helpers ──────────────────────────────────────────────────

/// Get theta_dot for a body, returning 0.0 for ground and None for unknown bodies.
fn get_body_theta_dot(state: &State, body_id: &str, q_dot: &DVector<f64>) -> Option<f64> {
    if state.is_ground(body_id) {
        return Some(0.0);
    }
    state.get_index(body_id).ok().map(|idx| q_dot[idx.theta_idx()])
}

/// Get (theta, theta_dot) for a body, returning (0.0, 0.0) for ground and None for unknown bodies.
fn get_body_theta_and_dot(
    state: &State,
    body_id: &str,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> Option<(f64, f64)> {
    if state.is_ground(body_id) {
        return Some((0.0, 0.0));
    }
    state
        .get_index(body_id)
        .ok()
        .map(|idx| (q[idx.theta_idx()], q_dot[idx.theta_idx()]))
}

// ── Evaluation functions ─────────────────────────────────────────────────────

pub fn evaluate_gravity(
    g: &GravityElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> DVector<f64> {
    let g_vec = Vector2::new(g.g_vector[0], g.g_vector[1]);
    let mut total = DVector::zeros(state.n_coords());

    for (body_id, body) in bodies {
        if body_id == GROUND_ID || body.mass <= 0.0 {
            continue;
        }
        let force_global = g_vec * body.mass;
        total += point_force_to_q(state, body_id, &body.cg_local, &force_global, q);
    }

    total
}

pub fn evaluate_linear_spring(
    s: &LinearSpringElement,
    state: &State,
    q: &DVector<f64>,
) -> DVector<f64> {
    let pt_a_local = Vector2::new(s.point_a[0], s.point_a[1]);
    let pt_b_local = Vector2::new(s.point_b[0], s.point_b[1]);

    let pt_a_global = state.body_point_global(&s.body_a, &pt_a_local, q);
    let pt_b_global = state.body_point_global(&s.body_b, &pt_b_local, q);

    let delta = pt_b_global - pt_a_global;
    let length = delta.norm();

    if length < 1e-15 {
        return DVector::zeros(state.n_coords());
    }

    let unit = delta / length;
    let extension = length - s.free_length;
    let force_magnitude = s.stiffness * extension;

    // Force on body A (toward B when extended)
    let force_on_a = unit * force_magnitude;
    // Force on body B (toward A when extended — Newton's third law)
    let force_on_b = -force_on_a;

    let mut total = DVector::zeros(state.n_coords());
    total += point_force_to_q(state, &s.body_a, &pt_a_local, &force_on_a, q);
    total += point_force_to_q(state, &s.body_b, &pt_b_local, &force_on_b, q);
    total
}

pub fn evaluate_torsion_spring(
    s: &TorsionSpringElement,
    state: &State,
    q: &DVector<f64>,
) -> DVector<f64> {
    let theta_i = state.get_angle(&s.body_i, q);
    let theta_j = state.get_angle(&s.body_j, q);

    let relative_angle = theta_j - theta_i;
    let torque = -s.stiffness * (relative_angle - s.free_angle);

    // Torque on body J, reaction on body I (Newton's third law)
    let mut total = DVector::zeros(state.n_coords());
    total += body_torque_to_q(state, &s.body_j, torque);
    total += body_torque_to_q(state, &s.body_i, -torque);
    total
}

pub fn evaluate_linear_damper(
    d: &LinearDamperElement,
    state: &State,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let pt_a_local = Vector2::new(d.point_a[0], d.point_a[1]);
    let pt_b_local = Vector2::new(d.point_b[0], d.point_b[1]);

    let pt_a_global = state.body_point_global(&d.body_a, &pt_a_local, q);
    let pt_b_global = state.body_point_global(&d.body_b, &pt_b_local, q);

    let delta = pt_b_global - pt_a_global;
    let length = delta.norm();

    if length < 1e-15 {
        return DVector::zeros(state.n_coords());
    }

    let unit = delta / length;

    // Compute rate of change of length: d/dt(|P_b - P_a|) = unit · (v_b - v_a)
    let v_a = state.body_point_velocity(&d.body_a, &pt_a_local, q, q_dot);
    let v_b = state.body_point_velocity(&d.body_b, &pt_b_local, q, q_dot);
    let length_rate = unit.dot(&(v_b - v_a));

    let force_magnitude = -d.damping * length_rate;

    // Force on body A (along unit direction)
    let force_on_a = unit * force_magnitude;
    let force_on_b = -force_on_a;

    let mut total = DVector::zeros(state.n_coords());
    total += point_force_to_q(state, &d.body_a, &pt_a_local, &force_on_a, q);
    total += point_force_to_q(state, &d.body_b, &pt_b_local, &force_on_b, q);
    total
}

pub fn evaluate_rotary_damper(
    d: &RotaryDamperElement,
    state: &State,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let Some(theta_dot_i) = get_body_theta_dot(state, &d.body_i, q_dot) else {
        return DVector::zeros(q_dot.len());
    };
    let Some(theta_dot_j) = get_body_theta_dot(state, &d.body_j, q_dot) else {
        return DVector::zeros(q_dot.len());
    };

    let relative_rate = theta_dot_j - theta_dot_i;
    let torque = -d.damping * relative_rate;

    let mut total = DVector::zeros(state.n_coords());
    total += body_torque_to_q(state, &d.body_j, torque);
    total += body_torque_to_q(state, &d.body_i, -torque);
    total
}

pub fn evaluate_external_force(
    f: &ExternalForceElement,
    state: &State,
    q: &DVector<f64>,
    t: f64,
) -> DVector<f64> {
    let local_pt = Vector2::new(f.local_point[0], f.local_point[1]);
    let factor = f.modulation.factor(t);
    let force = Vector2::new(f.force[0] * factor, f.force[1] * factor);
    point_force_to_q(state, &f.body_id, &local_pt, &force, q)
}

pub fn evaluate_external_torque(te: &ExternalTorqueElement, state: &State, t: f64) -> DVector<f64> {
    let factor = te.modulation.factor(t);
    body_torque_to_q(state, &te.body_id, te.torque * factor)
}

/// Evaluate external force with a pre-computed modulation factor.
pub fn evaluate_external_force_with_factor(
    f: &ExternalForceElement,
    state: &State,
    q: &DVector<f64>,
    factor: f64,
) -> DVector<f64> {
    let local_pt = Vector2::new(f.local_point[0], f.local_point[1]);
    let force = Vector2::new(f.force[0] * factor, f.force[1] * factor);
    point_force_to_q(state, &f.body_id, &local_pt, &force, q)
}

/// Evaluate external torque with a pre-computed modulation factor.
pub fn evaluate_external_torque_with_factor(
    te: &ExternalTorqueElement,
    state: &State,
    factor: f64,
) -> DVector<f64> {
    body_torque_to_q(state, &te.body_id, te.torque * factor)
}

pub fn evaluate_gas_spring(
    g: &GasSpringElement,
    state: &State,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let pt_a_local = Vector2::new(g.point_a[0], g.point_a[1]);
    let pt_b_local = Vector2::new(g.point_b[0], g.point_b[1]);

    let pt_a_global = state.body_point_global(&g.body_a, &pt_a_local, q);
    let pt_b_global = state.body_point_global(&g.body_b, &pt_b_local, q);

    let delta = pt_b_global - pt_a_global;
    let current_length = delta.norm();

    if current_length < 1e-15 {
        return DVector::zeros(state.n_coords());
    }

    let unit = delta / current_length;

    // Degenerate gas spring with zero stroke: acts as constant-force element
    if g.stroke <= 0.0 {
        let force_on_b = unit * g.initial_force;
        let force_on_a = -force_on_b;
        let mut total = DVector::zeros(state.n_coords());
        total += point_force_to_q(state, &g.body_a, &pt_a_local, &force_on_a, q);
        total += point_force_to_q(state, &g.body_b, &pt_b_local, &force_on_b, q);
        return total;
    }

    // Compression from extended position, clamped to [0, stroke]
    let compression = (g.extended_length - current_length).clamp(0.0, g.stroke);

    // Gas force: F = F0 * (stroke / gas_column)^n
    let gas_column = (g.stroke - compression).max(1e-10);
    let force_ratio = (g.stroke / gas_column).powf(g.polytropic_exp);
    let gas_force = g.initial_force * force_ratio;

    // Velocity-dependent damping along line of action
    let damping_force = if g.damping.abs() > 0.0 {
        let v_a = state.body_point_velocity(&g.body_a, &pt_a_local, q, q_dot);
        let v_b = state.body_point_velocity(&g.body_b, &pt_b_local, q, q_dot);
        let v_along = unit.dot(&(v_b - v_a));
        -g.damping * v_along
    } else {
        0.0
    };

    let total_force = gas_force + damping_force;

    // Gas spring pushes apart (positive = extension)
    let force_on_b = unit * total_force;
    let force_on_a = -force_on_b;

    let mut total = DVector::zeros(state.n_coords());
    total += point_force_to_q(state, &g.body_a, &pt_a_local, &force_on_a, q);
    total += point_force_to_q(state, &g.body_b, &pt_b_local, &force_on_b, q);
    total
}

pub fn evaluate_bearing_friction(
    b: &BearingFrictionElement,
    state: &State,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let Some(omega_i) = get_body_theta_dot(state, &b.body_i, q_dot) else {
        return DVector::zeros(q_dot.len());
    };
    let Some(omega_j) = get_body_theta_dot(state, &b.body_j, q_dot) else {
        return DVector::zeros(q_dot.len());
    };

    let omega_rel = omega_j - omega_i;

    // Direction via tanh regularization
    let direction = (omega_rel / b.v_threshold).tanh();

    // Total friction magnitude
    let magnitude = b.constant_drag
        + b.viscous_coeff * omega_rel.abs()
        + b.coulomb_coeff * b.pin_radius * b.radial_load;

    let torque = -magnitude * direction;

    let mut total = DVector::zeros(state.n_coords());
    total += body_torque_to_q(state, &b.body_j, torque);
    total += body_torque_to_q(state, &b.body_i, -torque);
    total
}

pub fn evaluate_joint_limit(
    j: &JointLimitElement,
    state: &State,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    // Get relative angle and angular velocity
    let Some((theta_i, omega_i)) = get_body_theta_and_dot(state, &j.body_i, q, q_dot) else {
        return DVector::zeros(q_dot.len());
    };
    let Some((theta_j, omega_j)) = get_body_theta_and_dot(state, &j.body_j, q, q_dot) else {
        return DVector::zeros(q_dot.len());
    };

    let theta_rel = theta_j - theta_i;
    let omega_rel = omega_j - omega_i;

    let torque = if theta_rel < j.angle_min {
        // Below minimum -- push CCW (positive torque on j)
        let penetration = j.angle_min - theta_rel;
        // Full damping when moving into the stop, reduced by restitution when bouncing away
        let damp_factor = if omega_rel < 0.0 {
            j.damping
        } else {
            j.damping * j.restitution
        };
        j.stiffness * penetration - damp_factor * omega_rel
    } else if theta_rel > j.angle_max {
        // Above maximum -- push CW (negative torque on j)
        let penetration = theta_rel - j.angle_max;
        // Full damping when moving into the stop, reduced by restitution when bouncing away
        let damp_factor = if omega_rel > 0.0 {
            j.damping
        } else {
            j.damping * j.restitution
        };
        -(j.stiffness * penetration + damp_factor * omega_rel)
    } else {
        return DVector::zeros(state.n_coords());
    };

    let mut total = DVector::zeros(state.n_coords());
    total += body_torque_to_q(state, &j.body_j, torque);
    total += body_torque_to_q(state, &j.body_i, -torque);
    total
}

pub fn evaluate_motor(
    m: &MotorElement,
    state: &State,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    if m.no_load_speed <= 0.0 {
        return DVector::zeros(state.n_coords());
    }

    let Some(omega_i) = get_body_theta_dot(state, &m.body_i, q_dot) else {
        return DVector::zeros(q_dot.len());
    };
    let Some(omega_j) = get_body_theta_dot(state, &m.body_j, q_dot) else {
        return DVector::zeros(q_dot.len());
    };

    let omega_rel = omega_j - omega_i;
    let speed_in_dir = omega_rel * m.direction;

    // Linear droop: T = T_stall * (1 - speed / omega_no_load)
    let torque_fraction = (1.0 - speed_in_dir / m.no_load_speed).clamp(0.0, 1.0);
    let torque = m.stall_torque * torque_fraction * m.direction;

    let mut total = DVector::zeros(state.n_coords());
    total += body_torque_to_q(state, &m.body_j, torque);
    total += body_torque_to_q(state, &m.body_i, -torque);
    total
}

pub fn evaluate_linear_actuator(
    a: &LinearActuatorElement,
    state: &State,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let pt_a_local = Vector2::new(a.point_a[0], a.point_a[1]);
    let pt_b_local = Vector2::new(a.point_b[0], a.point_b[1]);
    let pt_a_global = state.body_point_global(&a.body_a, &pt_a_local, q);
    let pt_b_global = state.body_point_global(&a.body_b, &pt_b_local, q);
    let delta = pt_b_global - pt_a_global;
    let length = delta.norm();
    if length < 1e-15 {
        return DVector::zeros(state.n_coords());
    }
    let unit = delta / length;

    // Speed limiting: ramp force to zero as speed approaches limit
    let actual_force = if a.speed_limit > 0.0 {
        let v_a = state.body_point_velocity(&a.body_a, &pt_a_local, q, q_dot);
        let v_b = state.body_point_velocity(&a.body_b, &pt_b_local, q, q_dot);
        let v_along = unit.dot(&(v_b - v_a));
        let speed_ratio = v_along.abs() / a.speed_limit;
        if speed_ratio >= 1.0 { 0.0 } else { a.force * (1.0 - speed_ratio) }
    } else {
        a.force
    };

    let mut net_force_along_unit = actual_force;

    // Stroke limit penalty forces (spring + damper at end stops)
    let limits_active = a.stroke_max > 0.0 && a.stroke_max > a.stroke_min;
    if limits_active {
        let v_a = state.body_point_velocity(&a.body_a, &pt_a_local, q, q_dot);
        let v_b = state.body_point_velocity(&a.body_b, &pt_b_local, q, q_dot);
        let v_rel = unit.dot(&(v_b - v_a));

        if a.stroke_min > 0.0 && length < a.stroke_min {
            let penetration = a.stroke_min - length;
            let damp = if v_rel < 0.0 { a.end_stop_damping } else { a.end_stop_damping * a.end_stop_restitution };
            net_force_along_unit += a.end_stop_stiffness * penetration - damp * v_rel;
        } else if a.stroke_max > 0.0 && length > a.stroke_max {
            let penetration = length - a.stroke_max;
            let damp = if v_rel > 0.0 { a.end_stop_damping } else { a.end_stop_damping * a.end_stop_restitution };
            net_force_along_unit -= a.end_stop_stiffness * penetration + damp * v_rel;
        }
    }

    let force_on_b = unit * net_force_along_unit;
    let force_on_a = -force_on_b;
    let mut total = DVector::zeros(state.n_coords());
    total += point_force_to_q(state, &a.body_a, &pt_a_local, &force_on_a, q);
    total += point_force_to_q(state, &a.body_b, &pt_b_local, &force_on_b, q);
    total
}

// ── Force zone evaluation ────────────────────────────────────────────────────

pub fn evaluate_force_zone(
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

/// Compute the overlap ratio for a force zone at the current configuration.
///
/// Returns a value in [0.0, 1.0] representing the fraction of the target body's
/// geometry that lies within the force zone. Returns 0.0 if the body has no
/// geometry or is not found.
pub fn force_zone_overlap_ratio(
    fz: &ForceZoneElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> f64 {
    use crate::geometry::{body_rect_to_world, clip_polygon_to_aabb, polygon_area};

    let body = match bodies.get(&fz.body_id) {
        Some(b) => b,
        None => return 0.0,
    };
    let geo = match &body.geometry {
        Some(g) => g,
        None => return 0.0,
    };

    let bi = match state.get_index(&fz.body_id) {
        Ok(idx) => idx,
        Err(_) => return 0.0,
    };
    let bx = q[bi.x_idx()];
    let by = q[bi.y_idx()];
    let btheta = q[bi.theta_idx()];

    let corners = body_rect_to_world(bx, by, btheta, geo.width, geo.height, &geo.offset);
    let zone_min = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
    let zone_max = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
    let clipped = clip_polygon_to_aabb(&corners, &zone_min, &zone_max);

    let overlap_area = polygon_area(&clipped);
    let body_area = geo.area();

    if body_area < 1e-15 {
        return 0.0;
    }

    (overlap_area / body_area).min(1.0)
}
