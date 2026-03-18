//! Joint constraint equations, Jacobians, and gamma (acceleration RHS).
//!
//! Each constraint type provides:
//!   - constraint(q, t) → residual vector Φ
//!   - jacobian(q, t)   → Jacobian matrix rows ∂Φ/∂q
//!   - gamma(q, q̇, t)  → acceleration RHS contribution
//!   - n_equations       → number of constraint rows
//!
//! Constraints connect two bodies at specified attachment points.
//! When one body is ground, its terms become constants (no q entries).

use nalgebra::{DMatrix, DVector, Vector2};

use crate::core::state::State;

/// Interface that all joint constraints must implement.
pub trait Constraint {
    fn id(&self) -> &str;
    fn n_equations(&self) -> usize;
    fn dof_removed(&self) -> usize;
    fn body_i_id(&self) -> &str;
    fn body_j_id(&self) -> &str;

    fn constraint(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64>;
    fn phi_t(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64>;
    fn jacobian(&self, state: &State, q: &DVector<f64>, t: f64) -> DMatrix<f64>;
    fn gamma(
        &self,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        t: f64,
    ) -> DVector<f64>;
}

// ---------------------------------------------------------------------------
// Revolute joint
// ---------------------------------------------------------------------------

/// Revolute joint: constrains two attachment points to be coincident.
///
/// Removes 2 translational DOF. Allows relative rotation.
///
/// Constraint (2 equations):
///     Φ = rᵢ + Aᵢ·sᵢ − rⱼ − Aⱼ·sⱼ = 0
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

    fn phi_t(&self, state: &State, _q: &DVector<f64>, _t: f64) -> DVector<f64> {
        let _ = state;
        DVector::zeros(2)
    }

    fn jacobian(&self, state: &State, q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(2, n);

        if !state.is_ground(&self.body_i_id_) {
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            jac[(0, idx_i.x_idx())] = 1.0;
            jac[(1, idx_i.y_idx())] = 1.0;
            let b_si =
                state.body_point_global_derivative(&self.body_i_id_, &self.point_i_local, q);
            jac[(0, idx_i.theta_idx())] = b_si.x;
            jac[(1, idx_i.theta_idx())] = b_si.y;
        }

        if !state.is_ground(&self.body_j_id_) {
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            jac[(0, idx_j.x_idx())] = -1.0;
            jac[(1, idx_j.y_idx())] = -1.0;
            let b_sj =
                state.body_point_global_derivative(&self.body_j_id_, &self.point_j_local, q);
            jac[(0, idx_j.theta_idx())] = -b_sj.x;
            jac[(1, idx_j.theta_idx())] = -b_sj.y;
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
        let mut result = Vector2::zeros();

        if !state.is_ground(&self.body_i_id_) {
            let theta_i = state.get_angle(&self.body_i_id_, q);
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            let theta_dot_i = q_dot[idx_i.theta_idx()];
            let a_i = State::rotation_matrix(theta_i);
            result += (a_i * self.point_i_local) * theta_dot_i.powi(2);
        }

        if !state.is_ground(&self.body_j_id_) {
            let theta_j = state.get_angle(&self.body_j_id_, q);
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            let theta_dot_j = q_dot[idx_j.theta_idx()];
            let a_j = State::rotation_matrix(theta_j);
            result -= (a_j * self.point_j_local) * theta_dot_j.powi(2);
        }

        DVector::from_column_slice(&[result.x, result.y])
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

// ---------------------------------------------------------------------------
// Fixed joint
// ---------------------------------------------------------------------------

/// Fixed joint: constrains all relative motion to zero.
///
/// Removes 3 DOF (2 translational + 1 rotational).
///
/// Constraint (3 equations):
///     Φ[0:2] = rᵢ + Aᵢ·sᵢ − rⱼ − Aⱼ·sⱼ = 0   (coincident points)
///     Φ[2]   = θⱼ − θᵢ − Δθ₀ = 0                 (locked relative angle)
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

    fn phi_t(&self, state: &State, _q: &DVector<f64>, _t: f64) -> DVector<f64> {
        let _ = state;
        DVector::zeros(3)
    }

    fn jacobian(&self, state: &State, q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(3, n);

        if !state.is_ground(&self.body_i_id_) {
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            jac[(0, idx_i.x_idx())] = 1.0;
            jac[(1, idx_i.y_idx())] = 1.0;
            let b_si =
                state.body_point_global_derivative(&self.body_i_id_, &self.point_i_local, q);
            jac[(0, idx_i.theta_idx())] = b_si.x;
            jac[(1, idx_i.theta_idx())] = b_si.y;
            jac[(2, idx_i.theta_idx())] = -1.0;
        }

        if !state.is_ground(&self.body_j_id_) {
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            jac[(0, idx_j.x_idx())] = -1.0;
            jac[(1, idx_j.y_idx())] = -1.0;
            let b_sj =
                state.body_point_global_derivative(&self.body_j_id_, &self.point_j_local, q);
            jac[(0, idx_j.theta_idx())] = -b_sj.x;
            jac[(1, idx_j.theta_idx())] = -b_sj.y;
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
        let mut pos_gamma = Vector2::zeros();

        if !state.is_ground(&self.body_i_id_) {
            let theta_i = state.get_angle(&self.body_i_id_, q);
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            let theta_dot_i = q_dot[idx_i.theta_idx()];
            let a_i = State::rotation_matrix(theta_i);
            pos_gamma += (a_i * self.point_i_local) * theta_dot_i.powi(2);
        }

        if !state.is_ground(&self.body_j_id_) {
            let theta_j = state.get_angle(&self.body_j_id_, q);
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            let theta_dot_j = q_dot[idx_j.theta_idx()];
            let a_j = State::rotation_matrix(theta_j);
            pos_gamma -= (a_j * self.point_j_local) * theta_dot_j.powi(2);
        }

        // γ[2] = 0 (rotation constraint is linear in θ)
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

// ---------------------------------------------------------------------------
// Prismatic joint
// ---------------------------------------------------------------------------

/// Prismatic (slider) joint: allows translation along one axis, locks rotation.
///
/// Removes 2 DOF (1 perpendicular translation + 1 rotation).
///
/// Constraint (2 equations):
///     Φ[0] = n̂ᵢ_global · d = 0        (no perpendicular displacement)
///     Φ[1] = θⱼ − θᵢ − Δθ₀ = 0        (no relative rotation)
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

    fn phi_t(&self, state: &State, _q: &DVector<f64>, _t: f64) -> DVector<f64> {
        let _ = state;
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

        // γ[0]: velocity-quadratic terms from Φ̈[0]
        let gamma_0 = n_hat_g.dot(&d) * theta_dot_i.powi(2)
            - 2.0 * theta_dot_i * b_n.dot(&d_dot)
            + n_hat_g.dot(&(a_j * self.point_j_local)) * theta_dot_j.powi(2)
            - n_hat_g.dot(&(a_i * self.point_i_local)) * theta_dot_i.powi(2);

        // γ[1] = 0 (rotation constraint is linear in θ)
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
    // Perpendicular: rotate axis by 90° CCW → n̂ = (−e_y, e_x)
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

/// Enum wrapper for dynamic dispatch of constraint types.
#[derive(Debug, Clone)]
pub enum JointConstraint {
    Revolute(RevoluteJoint),
    Fixed(FixedJoint),
    Prismatic(PrismaticJoint),
}

impl JointConstraint {
    pub fn point_i_local(&self) -> Vector2<f64> {
        match self {
            Self::Revolute(j) => j.point_i_local,
            Self::Fixed(j) => j.point_i_local,
            Self::Prismatic(j) => j.point_i_local,
        }
    }

    pub fn point_j_local(&self) -> Vector2<f64> {
        match self {
            Self::Revolute(j) => j.point_j_local,
            Self::Fixed(j) => j.point_j_local,
            Self::Prismatic(j) => j.point_j_local,
        }
    }

    pub fn is_revolute(&self) -> bool {
        matches!(self, Self::Revolute(_))
    }

    pub fn is_prismatic(&self) -> bool {
        matches!(self, Self::Prismatic(_))
    }

    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }
}

impl Constraint for JointConstraint {
    fn id(&self) -> &str {
        match self {
            Self::Revolute(j) => j.id(),
            Self::Fixed(j) => j.id(),
            Self::Prismatic(j) => j.id(),
        }
    }
    fn n_equations(&self) -> usize {
        match self {
            Self::Revolute(j) => j.n_equations(),
            Self::Fixed(j) => j.n_equations(),
            Self::Prismatic(j) => j.n_equations(),
        }
    }
    fn dof_removed(&self) -> usize {
        match self {
            Self::Revolute(j) => j.dof_removed(),
            Self::Fixed(j) => j.dof_removed(),
            Self::Prismatic(j) => j.dof_removed(),
        }
    }
    fn body_i_id(&self) -> &str {
        match self {
            Self::Revolute(j) => j.body_i_id(),
            Self::Fixed(j) => j.body_i_id(),
            Self::Prismatic(j) => j.body_i_id(),
        }
    }
    fn body_j_id(&self) -> &str {
        match self {
            Self::Revolute(j) => j.body_j_id(),
            Self::Fixed(j) => j.body_j_id(),
            Self::Prismatic(j) => j.body_j_id(),
        }
    }
    fn constraint(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64> {
        match self {
            Self::Revolute(j) => j.constraint(state, q, t),
            Self::Fixed(j) => j.constraint(state, q, t),
            Self::Prismatic(j) => j.constraint(state, q, t),
        }
    }
    fn phi_t(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64> {
        match self {
            Self::Revolute(j) => j.phi_t(state, q, t),
            Self::Fixed(j) => j.phi_t(state, q, t),
            Self::Prismatic(j) => j.phi_t(state, q, t),
        }
    }
    fn jacobian(&self, state: &State, q: &DVector<f64>, t: f64) -> DMatrix<f64> {
        match self {
            Self::Revolute(j) => j.jacobian(state, q, t),
            Self::Fixed(j) => j.jacobian(state, q, t),
            Self::Prismatic(j) => j.jacobian(state, q, t),
        }
    }
    fn gamma(
        &self,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        t: f64,
    ) -> DVector<f64> {
        match self {
            Self::Revolute(j) => j.gamma(state, q, q_dot, t),
            Self::Fixed(j) => j.gamma(state, q, q_dot, t),
            Self::Prismatic(j) => j.gamma(state, q, q_dot, t),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    /// Build a simple 2-body state: crank (body 0) + coupler (body 1).
    fn two_body_state() -> State {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        state.register_body("coupler").unwrap();
        state
    }

    #[test]
    fn revolute_constraint_zero_at_coincident_points() {
        let state = two_body_state();
        let mut q = state.make_q();
        // Place crank at origin, angle 0 → point (0.1, 0) in global
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.0);
        // Place coupler so its local (0,0) is at (0.1, 0) in global
        state.set_pose("coupler", &mut q, 0.1, 0.0, 0.0);

        let joint = make_revolute_joint(
            "J1",
            "crank",
            Vector2::new(0.1, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
        );

        let phi = joint.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(phi[1], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn revolute_constraint_nonzero_when_separated() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("coupler", &mut q, 0.2, 0.0, 0.0); // too far

        let joint = make_revolute_joint(
            "J1",
            "crank",
            Vector2::new(0.1, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
        );

        let phi = joint.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], -0.1, epsilon = 1e-14);
    }

    #[test]
    fn revolute_jacobian_finite_difference() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.5);
        state.set_pose("coupler", &mut q, 0.1, 0.05, 1.0);

        let joint = make_revolute_joint(
            "J1",
            "crank",
            Vector2::new(0.1, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
        );

        let jac_analytical = joint.jacobian(&state, &q, 0.0);

        // Finite-difference Jacobian
        let eps = 1e-7;
        let n = state.n_coords();
        let mut jac_fd = DMatrix::zeros(2, n);
        for col in 0..n {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[col] += eps;
            q_minus[col] -= eps;
            let phi_plus = joint.constraint(&state, &q_plus, 0.0);
            let phi_minus = joint.constraint(&state, &q_minus, 0.0);
            for row in 0..2 {
                jac_fd[(row, col)] = (phi_plus[row] - phi_minus[row]) / (2.0 * eps);
            }
        }

        for row in 0..2 {
            for col in 0..n {
                assert_abs_diff_eq!(
                    jac_analytical[(row, col)],
                    jac_fd[(row, col)],
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn revolute_ground_to_body() {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, PI / 4.0);

        let joint = make_revolute_joint(
            "J_ground",
            "ground",
            Vector2::new(0.0, 0.0),
            "crank",
            Vector2::new(0.0, 0.0),
        );

        let phi = joint.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(phi[1], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn fixed_joint_constraint_zero_at_locked_config() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("coupler", &mut q, 0.1, 0.0, 0.0);

        let joint = make_fixed_joint(
            "F1",
            "crank",
            Vector2::new(0.1, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
            0.0,
        );

        let phi = joint.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(phi[1], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(phi[2], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn fixed_joint_jacobian_finite_difference() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.5);
        state.set_pose("coupler", &mut q, 0.1, 0.05, 1.0);

        let joint = make_fixed_joint(
            "F1",
            "crank",
            Vector2::new(0.1, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
            0.5,
        );

        let jac_analytical = joint.jacobian(&state, &q, 0.0);

        let eps = 1e-7;
        let n = state.n_coords();
        let mut jac_fd = DMatrix::zeros(3, n);
        for col in 0..n {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[col] += eps;
            q_minus[col] -= eps;
            let phi_plus = joint.constraint(&state, &q_plus, 0.0);
            let phi_minus = joint.constraint(&state, &q_minus, 0.0);
            for row in 0..3 {
                jac_fd[(row, col)] = (phi_plus[row] - phi_minus[row]) / (2.0 * eps);
            }
        }

        for row in 0..3 {
            for col in 0..n {
                assert_abs_diff_eq!(
                    jac_analytical[(row, col)],
                    jac_fd[(row, col)],
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn prismatic_constraint_zero_on_axis() {
        let mut state = State::new();
        state.register_body("slider").unwrap();
        let mut q = state.make_q();
        // Slider at (0.5, 0) — on the x-axis (slide direction from ground)
        state.set_pose("slider", &mut q, 0.5, 0.0, 0.0);

        let joint = make_prismatic_joint(
            "P1",
            "ground",
            Vector2::new(0.0, 0.0),
            "slider",
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        let phi = joint.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(phi[1], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn prismatic_jacobian_finite_difference() {
        let mut state = State::new();
        state.register_body("slider").unwrap();
        let mut q = state.make_q();
        state.set_pose("slider", &mut q, 0.5, 0.01, 0.0);

        let joint = make_prismatic_joint(
            "P1",
            "ground",
            Vector2::new(0.0, 0.0),
            "slider",
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        let jac_analytical = joint.jacobian(&state, &q, 0.0);

        let eps = 1e-7;
        let n = state.n_coords();
        let mut jac_fd = DMatrix::zeros(2, n);
        for col in 0..n {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[col] += eps;
            q_minus[col] -= eps;
            let phi_plus = joint.constraint(&state, &q_plus, 0.0);
            let phi_minus = joint.constraint(&state, &q_minus, 0.0);
            for row in 0..2 {
                jac_fd[(row, col)] = (phi_plus[row] - phi_minus[row]) / (2.0 * eps);
            }
        }

        for row in 0..2 {
            for col in 0..n {
                assert_abs_diff_eq!(
                    jac_analytical[(row, col)],
                    jac_fd[(row, col)],
                    epsilon = 1e-6
                );
            }
        }
    }
}
