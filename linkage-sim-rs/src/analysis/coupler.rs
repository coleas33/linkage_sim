//! Coupler point evaluation: position, velocity, and acceleration.
//!
//! Computes the global-frame position, velocity, and acceleration of
//! coupler points on mechanism bodies, using the kinematic solution.
//!
//! For a point P on body i with local coordinates s_local:
//!   position:     r + A(theta) * s_local
//!   velocity:     r_dot + B(theta) * s_local * theta_dot
//!   acceleration: r_ddot + B(theta) * s_local * theta_ddot - A(theta) * s_local * theta_dot^2

use nalgebra::{DVector, Vector2};

use crate::core::state::State;

/// Evaluate global position, velocity, and acceleration of a body point.
///
/// Given a point in body-local coordinates, returns its global position,
/// velocity, and acceleration using the kinematic solution.
///
/// # Arguments
/// * `state` - State vector mapping.
/// * `body_id` - Body on which the point lies.
/// * `point_local` - Local coordinates of the point on the body.
/// * `q` - Position vector.
/// * `q_dot` - Velocity vector.
/// * `q_ddot` - Acceleration vector.
///
/// # Returns
/// `(position, velocity, acceleration)` each as `Vector2<f64>`.
pub fn eval_coupler_point(
    state: &State,
    body_id: &str,
    point_local: &Vector2<f64>,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    q_ddot: &DVector<f64>,
) -> (Vector2<f64>, Vector2<f64>, Vector2<f64>) {
    // Position: r + A * s
    let pos = state.body_point_global(body_id, point_local, q);

    if state.is_ground(body_id) {
        return (pos, Vector2::zeros(), Vector2::zeros());
    }

    let idx = state.get_index(body_id).expect("body not registered");
    let theta = state.get_angle(body_id, q);
    let theta_dot = q_dot[idx.theta_idx()];
    let theta_ddot = q_ddot[idx.theta_idx()];

    // Velocity: r_dot + B(theta) * s * theta_dot
    let b_s = state.body_point_global_derivative(body_id, point_local, q);
    let x_dot = q_dot[idx.x_idx()];
    let y_dot = q_dot[idx.y_idx()];
    let vel = Vector2::new(
        x_dot + b_s.x * theta_dot,
        y_dot + b_s.y * theta_dot,
    );

    // Acceleration: r_ddot + B*s*theta_ddot - A*s*theta_dot^2
    // d^2/dt^2(A*s) = B*s*theta_ddot + dB/dtheta*s*theta_dot^2
    //               = B*s*theta_ddot - A*s*theta_dot^2
    let a_mat = State::rotation_matrix(theta);
    let a_s = a_mat * point_local;

    let x_ddot = q_ddot[idx.x_idx()];
    let y_ddot = q_ddot[idx.y_idx()];
    let accel = Vector2::new(
        x_ddot + b_s.x * theta_ddot - a_s.x * theta_dot.powi(2),
        y_ddot + b_s.y * theta_ddot - a_s.y * theta_dot.powi(2),
    );

    (pos, vel, accel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn coupler_point_ground_returns_local_coords() {
        let state = State::new();
        let q = DVector::zeros(0);
        let q_dot = DVector::zeros(0);
        let q_ddot = DVector::zeros(0);
        let local = Vector2::new(3.0, 4.0);

        let (pos, vel, accel) = eval_coupler_point(&state, "ground", &local, &q, &q_dot, &q_ddot);

        assert_abs_diff_eq!(pos.x, 3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(pos.y, 4.0, epsilon = 1e-15);
        assert_abs_diff_eq!(vel.x, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(vel.y, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(accel.x, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(accel.y, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn coupler_point_position_at_zero_angle() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 1.0, 2.0, 0.0);
        let q_dot = DVector::zeros(3);
        let q_ddot = DVector::zeros(3);

        let local = Vector2::new(0.5, 0.0);
        let (pos, vel, accel) = eval_coupler_point(&state, "bar", &local, &q, &q_dot, &q_ddot);

        // Position: (1.0 + 0.5, 2.0 + 0.0) = (1.5, 2.0)
        assert_abs_diff_eq!(pos.x, 1.5, epsilon = 1e-14);
        assert_abs_diff_eq!(pos.y, 2.0, epsilon = 1e-14);
        // No velocity or acceleration
        assert_abs_diff_eq!(vel.x, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(vel.y, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(accel.x, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(accel.y, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn coupler_point_position_rotated() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, PI / 2.0);
        let q_dot = DVector::zeros(3);
        let q_ddot = DVector::zeros(3);

        let local = Vector2::new(1.0, 0.0);
        let (pos, _vel, _accel) =
            eval_coupler_point(&state, "bar", &local, &q, &q_dot, &q_ddot);

        // At theta=90 deg, local (1,0) maps to global (0, 1)
        assert_abs_diff_eq!(pos.x, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(pos.y, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn coupler_point_velocity_from_rotation() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let mut q_dot = DVector::zeros(3);
        // theta_dot = 2.0 rad/s, no translation
        q_dot[2] = 2.0;
        let q_ddot = DVector::zeros(3);

        let local = Vector2::new(1.0, 0.0);
        let (_pos, vel, _accel) =
            eval_coupler_point(&state, "bar", &local, &q, &q_dot, &q_ddot);

        // B(0) * (1,0) = (0, 1)
        // vel = (0, 0) + (0, 1) * 2.0 = (0, 2)
        assert_abs_diff_eq!(vel.x, 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(vel.y, 2.0, epsilon = 1e-14);
    }

    #[test]
    fn coupler_point_centripetal_acceleration() {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let mut q_dot = DVector::zeros(3);
        q_dot[2] = 3.0; // theta_dot = 3 rad/s
        let q_ddot = DVector::zeros(3); // no angular acceleration

        let local = Vector2::new(1.0, 0.0);
        let (_pos, _vel, accel) =
            eval_coupler_point(&state, "bar", &local, &q, &q_dot, &q_ddot);

        // A(0) * (1,0) = (1, 0)
        // accel = (0,0) + B*s*0 - A*s*9 = -(1,0)*9 = (-9, 0)
        assert_abs_diff_eq!(accel.x, -9.0, epsilon = 1e-14);
        assert_abs_diff_eq!(accel.y, 0.0, epsilon = 1e-14);
    }
}
