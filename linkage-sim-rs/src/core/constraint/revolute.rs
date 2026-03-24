//! Revolute joint: constrains two attachment points to be coincident.

use nalgebra::{DMatrix, DVector, Vector2};

use crate::core::state::State;

use super::helpers::{translational_gamma, translational_jacobian_block};
use super::trait_def::Constraint;

/// Revolute joint: constrains two attachment points to be coincident.
///
/// Removes 2 translational DOF. Allows relative rotation.
///
/// Constraint (2 equations):
///     Phi = r_i + A_i * s_i - r_j - A_j * s_j = 0
#[derive(Debug, Clone)]
pub struct RevoluteJoint {
    id_: String,
    body_i_id_: String,
    body_j_id_: String,
    point_i_local: Vector2<f64>,
    point_j_local: Vector2<f64>,
}

impl RevoluteJoint {
    pub fn point_i_local(&self) -> &Vector2<f64> {
        &self.point_i_local
    }
    pub fn point_j_local(&self) -> &Vector2<f64> {
        &self.point_j_local
    }
}

impl Constraint for RevoluteJoint {
    fn id(&self) -> &str {
        &self.id_
    }
    fn n_equations(&self) -> usize {
        2
    }
    fn dof_removed(&self) -> usize {
        2
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
        DVector::from_column_slice(&[diff.x, diff.y])
    }

    fn phi_t(&self, _state: &State, _q: &DVector<f64>, _t: f64) -> DVector<f64> {
        DVector::zeros(2)
    }

    fn jacobian(&self, state: &State, q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(2, n);
        translational_jacobian_block(
            state,
            &self.body_i_id_,
            &self.body_j_id_,
            &self.point_i_local,
            &self.point_j_local,
            q,
            &mut jac,
        );
        jac
    }

    fn gamma(
        &self,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        _t: f64,
    ) -> DVector<f64> {
        let g = translational_gamma(
            state,
            &self.body_i_id_,
            &self.body_j_id_,
            &self.point_i_local,
            &self.point_j_local,
            q,
            q_dot,
        );
        DVector::from_column_slice(&[g.x, g.y])
    }
}

/// Create a revolute joint between two bodies at specified attachment points.
pub fn make_revolute_joint(
    joint_id: &str,
    body_i_id: &str,
    point_i_local: Vector2<f64>,
    body_j_id: &str,
    point_j_local: Vector2<f64>,
) -> RevoluteJoint {
    RevoluteJoint {
        id_: joint_id.to_string(),
        body_i_id_: body_i_id.to_string(),
        body_j_id_: body_j_id.to_string(),
        point_i_local,
        point_j_local,
    }
}
