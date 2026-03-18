//! Helper functions to convert physical forces/torques to generalized forces.
//!
//! Virtual work principle: a force F at point P on body i produces:
//!   Q[x_idx]     += F[0]
//!   Q[y_idx]     += F[1]
//!   Q[theta_idx] += (B(θ) · s_local) · F

use nalgebra::{DVector, Vector2};

use crate::core::state::State;

/// Convert a point force in global coordinates to generalized forces.
///
/// F_global applied at s_local on body_id produces:
///   Q[x]   += Fx
///   Q[y]   += Fy
///   Q[θ]   += (B(θ) · s_local) · F
pub fn point_force_to_q(
    state: &State,
    body_id: &str,
    local_point: &Vector2<f64>,
    force_global: &Vector2<f64>,
    q: &DVector<f64>,
) -> DVector<f64> {
    let mut result = DVector::zeros(state.n_coords());
    if state.is_ground(body_id) {
        return result;
    }

    let idx = state.get_index(body_id).unwrap();
    result[idx.x_idx()] = force_global.x;
    result[idx.y_idx()] = force_global.y;

    // Moment arm: B(θ) · s_local
    let b_s = state.body_point_global_derivative(body_id, local_point, q);
    result[idx.theta_idx()] = b_s.dot(force_global);

    result
}

/// Convert a pure torque on a body to generalized forces.
///
/// Torque τ on body_id produces Q[θ] += τ.
pub fn body_torque_to_q(state: &State, body_id: &str, torque: f64) -> DVector<f64> {
    let mut result = DVector::zeros(state.n_coords());
    if state.is_ground(body_id) {
        return result;
    }

    let idx = state.get_index(body_id).unwrap();
    result[idx.theta_idx()] = torque;

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn point_force_at_origin_body() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 1.0, 2.0, 0.0);

        let force = Vector2::new(10.0, 0.0);
        let local_pt = Vector2::new(0.0, 0.0);
        let result = point_force_to_q(&state, "bar", &local_pt, &force, &q);

        // Force at body origin (0,0 local) → no moment arm
        assert_abs_diff_eq!(result[0], 10.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-14); // no torque
    }

    #[test]
    fn point_force_with_moment_arm() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);

        // Force at (1, 0) local, force in y-direction → torque = 1 * 10 = 10
        let force = Vector2::new(0.0, 10.0);
        let local_pt = Vector2::new(1.0, 0.0);
        let result = point_force_to_q(&state, "bar", &local_pt, &force, &q);

        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result[1], 10.0, epsilon = 1e-14);
        // B(0) · (1,0) = (0, 1), dot (0, 10) = 10
        assert_abs_diff_eq!(result[2], 10.0, epsilon = 1e-14);
    }

    #[test]
    fn point_force_rotated_body() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, PI / 2.0);

        // At θ=90°, local (1,0) maps to global (0,1)
        // B(90°) · (1,0) = (-1, 0)
        let force = Vector2::new(5.0, 0.0);
        let local_pt = Vector2::new(1.0, 0.0);
        let result = point_force_to_q(&state, "bar", &local_pt, &force, &q);

        assert_abs_diff_eq!(result[0], 5.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-14);
        // B(π/2) · (1,0) = (-1, 0), dot (5, 0) = -5
        assert_abs_diff_eq!(result[2], -5.0, epsilon = 1e-14);
    }

    #[test]
    fn ground_produces_no_forces() {
        let state = State::new();
        let q = DVector::zeros(0);
        let force = Vector2::new(10.0, 20.0);
        let local_pt = Vector2::new(1.0, 0.0);
        let result = point_force_to_q(&state, "ground", &local_pt, &force, &q);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn body_torque_simple() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let result = body_torque_to_q(&state, "bar", 5.0);
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result[2], 5.0, epsilon = 1e-15);
    }
}
