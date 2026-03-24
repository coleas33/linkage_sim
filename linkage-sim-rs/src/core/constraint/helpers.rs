//! Shared translational helpers used by Revolute, Fixed, and Prismatic joints.

use nalgebra::{DMatrix, DVector, Vector2};

use crate::core::state::State;

/// Compute the 2-row translational Jacobian block for the constraint
///     Phi = r_i + A_i * s_i - r_j - A_j * s_j = 0
///
/// Writes into rows 0..2 of a pre-allocated Jacobian matrix.
pub(super) fn translational_jacobian_block(
    state: &State,
    body_i_id: &str,
    body_j_id: &str,
    point_i_local: &Vector2<f64>,
    point_j_local: &Vector2<f64>,
    q: &DVector<f64>,
    jac: &mut DMatrix<f64>,
) {
    if !state.is_ground(body_i_id) {
        let idx_i = state.get_index(body_i_id).unwrap();
        jac[(0, idx_i.x_idx())] = 1.0;
        jac[(1, idx_i.y_idx())] = 1.0;
        let b_si = state.body_point_global_derivative(body_i_id, point_i_local, q);
        jac[(0, idx_i.theta_idx())] = b_si.x;
        jac[(1, idx_i.theta_idx())] = b_si.y;
    }

    if !state.is_ground(body_j_id) {
        let idx_j = state.get_index(body_j_id).unwrap();
        jac[(0, idx_j.x_idx())] = -1.0;
        jac[(1, idx_j.y_idx())] = -1.0;
        let b_sj = state.body_point_global_derivative(body_j_id, point_j_local, q);
        jac[(0, idx_j.theta_idx())] = -b_sj.x;
        jac[(1, idx_j.theta_idx())] = -b_sj.y;
    }
}

/// Compute the 2-element translational gamma (centripetal terms) for the constraint
///     Phi = r_i + A_i * s_i - r_j - A_j * s_j = 0
///
/// gamma_trans = A_i * s_i * theta_dot_i^2 - A_j * s_j * theta_dot_j^2
pub(super) fn translational_gamma(
    state: &State,
    body_i_id: &str,
    body_j_id: &str,
    point_i_local: &Vector2<f64>,
    point_j_local: &Vector2<f64>,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> Vector2<f64> {
    let mut result = Vector2::zeros();

    if !state.is_ground(body_i_id) {
        let theta_i = state.get_angle(body_i_id, q);
        let idx_i = state.get_index(body_i_id).unwrap();
        let theta_dot_i = q_dot[idx_i.theta_idx()];
        let a_i = State::rotation_matrix(theta_i);
        result += (a_i * point_i_local) * theta_dot_i.powi(2);
    }

    if !state.is_ground(body_j_id) {
        let theta_j = state.get_angle(body_j_id, q);
        let idx_j = state.get_index(body_j_id).unwrap();
        let theta_dot_j = q_dot[idx_j.theta_idx()];
        let a_j = State::rotation_matrix(theta_j);
        result -= (a_j * point_j_local) * theta_dot_j.powi(2);
    }

    result
}
