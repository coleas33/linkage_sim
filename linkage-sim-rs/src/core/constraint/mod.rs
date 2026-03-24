//! Joint constraint equations, Jacobians, and gamma (acceleration RHS).
//!
//! Each constraint type provides:
//!   - constraint(q, t) -> residual vector Phi
//!   - jacobian(q, t)   -> Jacobian matrix rows dPhi/dq
//!   - gamma(q, q_dot, t)  -> acceleration RHS contribution
//!   - n_equations       -> number of constraint rows
//!
//! Constraints connect two bodies at specified attachment points.
//! When one body is ground, its terms become constants (no q entries).

mod trait_def;
mod helpers;
mod revolute;
mod fixed;
mod prismatic;
mod cam;

pub use trait_def::Constraint;
pub use revolute::{RevoluteJoint, make_revolute_joint};
pub use fixed::{FixedJoint, make_fixed_joint};
pub use prismatic::{PrismaticJoint, make_prismatic_joint};
pub use cam::{CamProfile, CamFollowerJoint, make_cam_follower};

use nalgebra::{DMatrix, DVector, Vector2};

use crate::core::state::State;

/// Enum wrapper for dynamic dispatch of constraint types.
#[derive(Debug, Clone)]
pub enum JointConstraint {
    Revolute(RevoluteJoint),
    Fixed(FixedJoint),
    Prismatic(PrismaticJoint),
    CamFollower(CamFollowerJoint),
}

impl JointConstraint {
    pub fn point_i_local(&self) -> Vector2<f64> {
        match self {
            Self::Revolute(j) => *j.point_i_local(),
            Self::Fixed(j) => *j.point_i_local(),
            Self::Prismatic(j) => *j.point_i_local(),
            Self::CamFollower(j) => j.point_i_local,
        }
    }

    pub fn point_j_local(&self) -> Vector2<f64> {
        match self {
            Self::Revolute(j) => *j.point_j_local(),
            Self::Fixed(j) => *j.point_j_local(),
            Self::Prismatic(j) => *j.point_j_local(),
            Self::CamFollower(j) => j.point_j_local,
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

    pub fn is_cam_follower(&self) -> bool {
        matches!(self, Self::CamFollower(_))
    }
}

impl Constraint for JointConstraint {
    fn id(&self) -> &str {
        match self {
            Self::Revolute(j) => j.id(),
            Self::Fixed(j) => j.id(),
            Self::Prismatic(j) => j.id(),
            Self::CamFollower(j) => j.id(),
        }
    }
    fn n_equations(&self) -> usize {
        match self {
            Self::Revolute(j) => j.n_equations(),
            Self::Fixed(j) => j.n_equations(),
            Self::Prismatic(j) => j.n_equations(),
            Self::CamFollower(j) => j.n_equations(),
        }
    }
    fn dof_removed(&self) -> usize {
        match self {
            Self::Revolute(j) => j.dof_removed(),
            Self::Fixed(j) => j.dof_removed(),
            Self::Prismatic(j) => j.dof_removed(),
            Self::CamFollower(j) => j.dof_removed(),
        }
    }
    fn body_i_id(&self) -> &str {
        match self {
            Self::Revolute(j) => j.body_i_id(),
            Self::Fixed(j) => j.body_i_id(),
            Self::Prismatic(j) => j.body_i_id(),
            Self::CamFollower(j) => j.body_i_id(),
        }
    }
    fn body_j_id(&self) -> &str {
        match self {
            Self::Revolute(j) => j.body_j_id(),
            Self::Fixed(j) => j.body_j_id(),
            Self::Prismatic(j) => j.body_j_id(),
            Self::CamFollower(j) => j.body_j_id(),
        }
    }
    fn constraint(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64> {
        match self {
            Self::Revolute(j) => j.constraint(state, q, t),
            Self::Fixed(j) => j.constraint(state, q, t),
            Self::Prismatic(j) => j.constraint(state, q, t),
            Self::CamFollower(j) => j.constraint(state, q, t),
        }
    }
    fn phi_t(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64> {
        match self {
            Self::Revolute(j) => j.phi_t(state, q, t),
            Self::Fixed(j) => j.phi_t(state, q, t),
            Self::Prismatic(j) => j.phi_t(state, q, t),
            Self::CamFollower(j) => j.phi_t(state, q, t),
        }
    }
    fn jacobian(&self, state: &State, q: &DVector<f64>, t: f64) -> DMatrix<f64> {
        match self {
            Self::Revolute(j) => j.jacobian(state, q, t),
            Self::Fixed(j) => j.jacobian(state, q, t),
            Self::Prismatic(j) => j.jacobian(state, q, t),
            Self::CamFollower(j) => j.jacobian(state, q, t),
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
            Self::CamFollower(j) => j.gamma(state, q, q_dot, t),
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
        // Place crank at origin, angle 0 -> point (0.1, 0) in global
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
        // Slider at (0.5, 0) -- on the x-axis (slide direction from ground)
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

    // -----------------------------------------------------------------
    // Gamma finite-difference helper
    // -----------------------------------------------------------------

    /// Compute gamma via finite differences on phi_dot = Phi_q * q_dot + Phi_t.
    ///
    /// At time t with state (q, q_dot), advance to q_plus = q + q_dot * dt,
    /// then gamma_fd = -(phi_dot_plus - phi_dot) / dt.
    fn gamma_fd<C: Constraint>(
        joint: &C,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        t: f64,
    ) -> DVector<f64> {
        let dt = 1e-7;

        let jac = joint.jacobian(state, q, t);
        let phi_t = joint.phi_t(state, q, t);
        let phi_dot = &jac * q_dot + &phi_t;

        let q_plus = q + q_dot * dt;
        let t_plus = t + dt;
        let jac_plus = joint.jacobian(state, &q_plus, t_plus);
        let phi_t_plus = joint.phi_t(state, &q_plus, t_plus);
        let phi_dot_plus = &jac_plus * q_dot + &phi_t_plus;

        -(&phi_dot_plus - &phi_dot) / dt
    }

    fn assert_gamma_matches_fd<C: Constraint>(
        joint: &C,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        t: f64,
        tol: f64,
    ) {
        let gamma_analytical = joint.gamma(state, q, q_dot, t);
        let gamma_numerical = gamma_fd(joint, state, q, q_dot, t);
        let n_eq = joint.n_equations();
        for i in 0..n_eq {
            assert_abs_diff_eq!(
                gamma_analytical[i],
                gamma_numerical[i],
                epsilon = tol
            );
        }
    }

    // -----------------------------------------------------------------
    // Revolute gamma FD tests
    // -----------------------------------------------------------------

    #[test]
    fn revolute_gamma_fd_two_bodies() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.1, 0.2, 0.7);
        state.set_pose("coupler", &mut q, 0.3, -0.1, 1.2);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.1; // crank x_dot
        q_dot[1] = -0.2; // crank y_dot
        q_dot[2] = 3.0; // crank theta_dot
        q_dot[3] = -0.05; // coupler x_dot
        q_dot[4] = 0.15; // coupler y_dot
        q_dot[5] = -2.0; // coupler theta_dot

        let joint = make_revolute_joint(
            "J1",
            "crank",
            Vector2::new(0.1, 0.05),
            "coupler",
            Vector2::new(-0.03, 0.02),
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn revolute_gamma_fd_ground_to_body() {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 1.0);

        let mut q_dot = state.make_q();
        q_dot[2] = 5.0; // theta_dot

        let joint = make_revolute_joint(
            "J_gnd",
            "ground",
            Vector2::new(0.0, 0.0),
            "crank",
            Vector2::new(0.0, 0.0),
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    // -----------------------------------------------------------------
    // Fixed gamma FD tests
    // -----------------------------------------------------------------

    #[test]
    fn fixed_gamma_fd_two_bodies() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.1, 0.2, 0.7);
        state.set_pose("coupler", &mut q, 0.3, -0.1, 1.2);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.1;
        q_dot[1] = -0.2;
        q_dot[2] = 3.0;
        q_dot[3] = -0.05;
        q_dot[4] = 0.15;
        q_dot[5] = -2.0;

        let joint = make_fixed_joint(
            "F1",
            "crank",
            Vector2::new(0.1, 0.05),
            "coupler",
            Vector2::new(-0.03, 0.02),
            0.5,
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    // -----------------------------------------------------------------
    // Prismatic gamma FD tests
    // -----------------------------------------------------------------

    #[test]
    fn prismatic_gamma_fd_x_axis_ground_to_slider() {
        let mut state = State::new();
        state.register_body("slider").unwrap();
        let mut q = state.make_q();
        state.set_pose("slider", &mut q, 0.5, 0.01, 0.0);

        let mut q_dot = state.make_q();
        q_dot[0] = 1.0; // slider x_dot
        q_dot[1] = 0.05; // slider y_dot
        q_dot[2] = 0.0; // slider theta_dot (locked by constraint)

        let joint = make_prismatic_joint(
            "P_x",
            "ground",
            Vector2::new(0.0, 0.0),
            "slider",
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn prismatic_gamma_fd_y_axis_ground_to_slider() {
        let mut state = State::new();
        state.register_body("slider").unwrap();
        let mut q = state.make_q();
        state.set_pose("slider", &mut q, 0.02, 0.8, 0.0);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.03;
        q_dot[1] = 2.0;
        q_dot[2] = 0.0;

        let joint = make_prismatic_joint(
            "P_y",
            "ground",
            Vector2::new(0.0, 0.0),
            "slider",
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 1.0),
            0.0,
        )
        .unwrap();

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn prismatic_gamma_fd_rotated_rail() {
        // Body i (rail) is at a nonzero angle, so the axis is rotated in world space.
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, PI / 6.0); // rail at 30 deg
        state.set_pose("coupler", &mut q, 0.5, 0.3, PI / 6.0); // slider

        let mut q_dot = state.make_q();
        q_dot[0] = 0.0; // crank x_dot
        q_dot[1] = 0.0; // crank y_dot
        q_dot[2] = 0.0; // crank theta_dot (rail stationary)
        q_dot[3] = 0.5; // coupler x_dot
        q_dot[4] = 0.3; // coupler y_dot
        q_dot[5] = 0.0; // coupler theta_dot (locked)

        let joint = make_prismatic_joint(
            "P_rot",
            "crank",
            Vector2::new(0.0, 0.0),
            "coupler",
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn prismatic_gamma_fd_moving_parent_body() {
        // Body i (rail) is not ground and has translational velocity.
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.1, 0.2, 0.3);
        state.set_pose("coupler", &mut q, 0.6, 0.25, 0.3);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.5; // crank x_dot
        q_dot[1] = -0.3; // crank y_dot
        q_dot[2] = 0.0; // crank theta_dot
        q_dot[3] = 1.0; // coupler x_dot
        q_dot[4] = -0.1; // coupler y_dot
        q_dot[5] = 0.0; // coupler theta_dot

        let joint = make_prismatic_joint(
            "P_move",
            "crank",
            Vector2::new(0.05, 0.02),
            "coupler",
            Vector2::new(-0.01, 0.0),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn prismatic_gamma_fd_nonzero_rail_angular_velocity() {
        // The rail-carrying body has nonzero angular velocity -- most complex case.
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.1, 0.2, 0.8);
        state.set_pose("coupler", &mut q, 0.5, 0.4, 0.8);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.3; // crank x_dot
        q_dot[1] = -0.1; // crank y_dot
        q_dot[2] = 2.5; // crank theta_dot (nonzero angular velocity!)
        q_dot[3] = 0.8; // coupler x_dot
        q_dot[4] = 0.2; // coupler y_dot
        q_dot[5] = 2.5; // coupler theta_dot (locked, same as crank)

        let joint = make_prismatic_joint(
            "P_omega",
            "crank",
            Vector2::new(0.05, 0.03),
            "coupler",
            Vector2::new(-0.02, 0.01),
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    // -----------------------------------------------------------------
    // Cam-follower gamma FD tests
    // -----------------------------------------------------------------

    #[test]
    fn cam_follower_gamma_fd_two_bodies() {
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.1, 0.2, 0.7);
        state.set_pose("coupler", &mut q, 0.4, 0.3, 1.1);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.15; // crank x_dot
        q_dot[1] = -0.1; // crank y_dot
        q_dot[2] = 3.5; // crank theta_dot
        q_dot[3] = -0.2; // coupler x_dot
        q_dot[4] = 0.25; // coupler y_dot
        q_dot[5] = -1.5; // coupler theta_dot

        // Use a harmonic profile so s'' is nonzero everywhere
        let profile = CamProfile::Harmonic {
            amplitude: 0.05,
            frequency: 2.0,
            phase: 0.3,
            offset: 0.1,
        };

        let joint = make_cam_follower(
            "CF1",
            "crank",
            "coupler",
            Vector2::new(0.05, 0.02),
            Vector2::new(-0.03, 0.01),
            Vector2::new(1.0, 0.3),
            profile,
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn cam_follower_gamma_fd_ground_to_body() {
        let mut state = State::new();
        state.register_body("follower").unwrap();
        let mut q = state.make_q();
        state.set_pose("follower", &mut q, 0.3, 0.1, 0.5);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.4; // follower x_dot
        q_dot[1] = -0.3; // follower y_dot
        q_dot[2] = 2.0; // follower theta_dot

        // Cam body is ground (theta_i = 0, constant), follower slides
        let profile = CamProfile::Polynomial {
            coefficients: vec![0.0, 0.1, -0.05],
        };

        let joint = make_cam_follower(
            "CF_gnd",
            "ground",
            "follower",
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 1.0),
            profile,
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }

    #[test]
    fn cam_follower_gamma_fd_cam_rotating() {
        // Cam body rotates with nonzero angular velocity -- exercises all terms
        let state = two_body_state();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.4);
        state.set_pose("coupler", &mut q, 0.2, 0.15, 0.0);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.1; // crank x_dot
        q_dot[1] = 0.05; // crank y_dot
        q_dot[2] = 4.0; // crank theta_dot (large -- stresses centripetal terms)
        q_dot[3] = 0.3; // coupler x_dot
        q_dot[4] = -0.15; // coupler y_dot
        q_dot[5] = 1.0; // coupler theta_dot

        let profile = CamProfile::Harmonic {
            amplitude: 0.08,
            frequency: 3.0,
            phase: 0.0,
            offset: 0.0,
        };

        let joint = make_cam_follower(
            "CF_rot",
            "crank",
            "coupler",
            Vector2::new(0.1, 0.0),
            Vector2::new(-0.05, 0.02),
            Vector2::new(0.6, 0.8),
            profile,
        );

        assert_gamma_matches_fd(&joint, &state, &q, &q_dot, 0.0, 1e-5);
    }
}
