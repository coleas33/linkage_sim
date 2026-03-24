//! Prismatic (slider) joint: allows translation along one axis, locks rotation.

use nalgebra::{DMatrix, DVector, Vector2};

use crate::core::state::State;

use super::trait_def::Constraint;

/// Prismatic (slider) joint: allows translation along one axis, locks rotation.
///
/// Removes 2 DOF (1 perpendicular translation + 1 rotation).
///
/// Constraint (2 equations):
///     Phi[0] = n_hat_i_global . d = 0        (no perpendicular displacement)
///     Phi[1] = theta_j - theta_i - delta_theta_0 = 0        (no relative rotation)
#[derive(Debug, Clone)]
pub struct PrismaticJoint {
    id_: String,
    body_i_id_: String,
    body_j_id_: String,
    point_i_local: Vector2<f64>,
    point_j_local: Vector2<f64>,
    #[allow(dead_code)]
    axis_local_i: Vector2<f64>,
    n_hat_local_i: Vector2<f64>,
    delta_theta_0: f64,
}

impl PrismaticJoint {
    pub fn point_i_local(&self) -> &Vector2<f64> {
        &self.point_i_local
    }
    pub fn point_j_local(&self) -> &Vector2<f64> {
        &self.point_j_local
    }
    pub fn axis_local_i(&self) -> &Vector2<f64> {
        &self.axis_local_i
    }
    pub fn delta_theta_0(&self) -> f64 {
        self.delta_theta_0
    }
}

impl Constraint for PrismaticJoint {
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
        let theta_i = state.get_angle(&self.body_i_id_, q);
        let a_i = State::rotation_matrix(theta_i);
        let n_hat_g = a_i * self.n_hat_local_i;

        let pt_i_g = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let pt_j_g = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let d = pt_j_g - pt_i_g;

        let theta_j = state.get_angle(&self.body_j_id_, q);

        DVector::from_column_slice(&[
            n_hat_g.dot(&d),
            theta_j - theta_i - self.delta_theta_0,
        ])
    }

    fn phi_t(&self, _state: &State, _q: &DVector<f64>, _t: f64) -> DVector<f64> {
        DVector::zeros(2)
    }

    fn jacobian(&self, state: &State, q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(2, n);

        let theta_i = state.get_angle(&self.body_i_id_, q);
        let a_i = State::rotation_matrix(theta_i);
        let b_i = State::rotation_matrix_derivative(theta_i);
        let n_hat_g = a_i * self.n_hat_local_i;
        let b_n = b_i * self.n_hat_local_i;

        let pt_i_g = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let pt_j_g = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let d = pt_j_g - pt_i_g;

        if !state.is_ground(&self.body_i_id_) {
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            // Row 0: perpendicular constraint
            jac[(0, idx_i.x_idx())] = -n_hat_g.x;
            jac[(0, idx_i.y_idx())] = -n_hat_g.y;
            let b_si = b_i * self.point_i_local;
            jac[(0, idx_i.theta_idx())] = b_n.dot(&d) - n_hat_g.dot(&b_si);
            // Row 1: rotation constraint
            jac[(1, idx_i.theta_idx())] = -1.0;
        }

        if !state.is_ground(&self.body_j_id_) {
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            let theta_j = state.get_angle(&self.body_j_id_, q);
            let b_j = State::rotation_matrix_derivative(theta_j);
            // Row 0: perpendicular constraint
            jac[(0, idx_j.x_idx())] = n_hat_g.x;
            jac[(0, idx_j.y_idx())] = n_hat_g.y;
            let b_sj = b_j * self.point_j_local;
            jac[(0, idx_j.theta_idx())] = n_hat_g.dot(&b_sj);
            // Row 1: rotation constraint
            jac[(1, idx_j.theta_idx())] += 1.0;
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
        let theta_i = state.get_angle(&self.body_i_id_, q);
        let theta_j = state.get_angle(&self.body_j_id_, q);
        let a_i = State::rotation_matrix(theta_i);
        let b_i = State::rotation_matrix_derivative(theta_i);
        let a_j = State::rotation_matrix(theta_j);
        let b_j = State::rotation_matrix_derivative(theta_j);

        let n_hat_g = a_i * self.n_hat_local_i;
        let b_n = b_i * self.n_hat_local_i;

        let pt_i_g = state.body_point_global(&self.body_i_id_, &self.point_i_local, q);
        let pt_j_g = state.body_point_global(&self.body_j_id_, &self.point_j_local, q);
        let d = pt_j_g - pt_i_g;

        // Velocities (zero for ground bodies)
        let (theta_dot_i, r_dot_i) = if !state.is_ground(&self.body_i_id_) {
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            (
                q_dot[idx_i.theta_idx()],
                Vector2::new(q_dot[idx_i.x_idx()], q_dot[idx_i.y_idx()]),
            )
        } else {
            (0.0, Vector2::zeros())
        };

        let (theta_dot_j, r_dot_j) = if !state.is_ground(&self.body_j_id_) {
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            (
                q_dot[idx_j.theta_idx()],
                Vector2::new(q_dot[idx_j.x_idx()], q_dot[idx_j.y_idx()]),
            )
        } else {
            (0.0, Vector2::zeros())
        };

        // d_dot = velocity of d vector
        let d_dot = (r_dot_j + (b_j * self.point_j_local) * theta_dot_j)
            - (r_dot_i + (b_i * self.point_i_local) * theta_dot_i);

        // gamma[0]: velocity-quadratic terms from Phi_ddot[0]
        let gamma_0 = n_hat_g.dot(&d) * theta_dot_i.powi(2)
            - 2.0 * theta_dot_i * b_n.dot(&d_dot)
            + n_hat_g.dot(&(a_j * self.point_j_local)) * theta_dot_j.powi(2)
            - n_hat_g.dot(&(a_i * self.point_i_local)) * theta_dot_i.powi(2);

        // gamma[1] = 0 (rotation constraint is linear in theta)
        DVector::from_column_slice(&[gamma_0, 0.0])
    }
}

/// Create a prismatic joint that allows sliding along one axis.
pub fn make_prismatic_joint(
    joint_id: &str,
    body_i_id: &str,
    point_i_local: Vector2<f64>,
    body_j_id: &str,
    point_j_local: Vector2<f64>,
    axis_local_i: Vector2<f64>,
    delta_theta_0: f64,
) -> Result<PrismaticJoint, &'static str> {
    let norm = axis_local_i.norm();
    if norm < 1e-12 {
        return Err("axis_local_i must be non-zero");
    }
    let axis = axis_local_i / norm;
    // Perpendicular: rotate axis by 90 deg CCW -> n_hat = (-e_y, e_x)
    let n_hat = Vector2::new(-axis.y, axis.x);

    Ok(PrismaticJoint {
        id_: joint_id.to_string(),
        body_i_id_: body_i_id.to_string(),
        body_j_id_: body_j_id.to_string(),
        point_i_local,
        point_j_local,
        axis_local_i: axis,
        n_hat_local_i: n_hat,
        delta_theta_0,
    })
}
