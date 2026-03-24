//! Fixed joint: constrains all relative motion to zero.

use nalgebra::{DMatrix, DVector, Vector2};

use crate::core::state::State;

use super::helpers::{translational_gamma, translational_jacobian_block};
use super::trait_def::Constraint;

/// Fixed joint: constrains all relative motion to zero.
///
/// Removes 3 DOF (2 translational + 1 rotational).
///
/// Constraint (3 equations):
///     Phi[0:2] = r_i + A_i * s_i - r_j - A_j * s_j = 0   (coincident points)
///     Phi[2]   = theta_j - theta_i - delta_theta_0 = 0     (locked relative angle)
#[derive(Debug, Clone)]
pub struct FixedJoint {
    id_: String,
    body_i_id_: String,
    body_j_id_: String,
    point_i_local: Vector2<f64>,
    point_j_local: Vector2<f64>,
    delta_theta_0: f64,
}

impl FixedJoint {
    pub fn point_i_local(&self) -> &Vector2<f64> {
        &self.point_i_local
    }
    pub fn point_j_local(&self) -> &Vector2<f64> {
        &self.point_j_local
    }
    pub fn delta_theta_0(&self) -> f64 {
        self.delta_theta_0
    }
}

impl Constraint for FixedJoint {
    fn id(&self) -> &str {
        &self.id_
    }
    fn n_equations(&self) -> usize {
        3
    }
    fn dof_removed(&self) -> usize {
        3
    }
    fn body_i_id(&self) -> &str {
        &self.body_i_id_
    }
    fn body_j_id(&self) -> &str {
        &self.body_j_id_
    }

    fn constraint(&self, state: &State, q: &DVector<f64>, _t: f64) -> DVector<f64> {
        let global_i = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let global_j = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let diff = global_i - global_j;

        let theta_i = state.get_angle(&self.body_i_id_, q);
        let theta_j = state.get_angle(&self.body_j_id_, q);

        DVector::from_column_slice(&[diff.x, diff.y, theta_j - theta_i - self.delta_theta_0])
    }

    fn phi_t(&self, _state: &State, _q: &DVector<f64>, _t: f64) -> DVector<f64> {
        DVector::zeros(3)
    }

    fn jacobian(&self, state: &State, q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(3, n);

        // Rows 0-1: translational block (shared with revolute)
        translational_jacobian_block(
            state,
            &self.body_i_id_,
            &self.body_j_id_,
            &self.point_i_local,
            &self.point_j_local,
            q,
            &mut jac,
        );

        // Row 2: rotation lock
        if !state.is_ground(&self.body_i_id_) {
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            jac[(2, idx_i.theta_idx())] = -1.0;
        }
        if !state.is_ground(&self.body_j_id_) {
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            jac[(2, idx_j.theta_idx())] += 1.0;
        }

        jac
    }

    fn gamma(
        &self,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        _t: f64,
    ) -> DVector<f64> {
        let pos_gamma = translational_gamma(
            state,
            &self.body_i_id_,
            &self.body_j_id_,
            &self.point_i_local,
            &self.point_j_local,
            q,
            q_dot,
        );

        // gamma[2] = 0 (rotation constraint is linear in theta)
        DVector::from_column_slice(&[pos_gamma.x, pos_gamma.y, 0.0])
    }
}

/// Create a fixed joint that locks all relative motion.
pub fn make_fixed_joint(
    joint_id: &str,
    body_i_id: &str,
    point_i_local: Vector2<f64>,
    body_j_id: &str,
    point_j_local: Vector2<f64>,
    delta_theta_0: f64,
) -> FixedJoint {
    FixedJoint {
        id_: joint_id.to_string(),
        body_i_id_: body_i_id.to_string(),
        body_j_id_: body_j_id.to_string(),
        point_i_local,
        point_j_local,
        delta_theta_0,
    }
}
