//! Linear driver constraint: prescribes the distance between two body points
//! as a function of time.
//!
//! Analogous to `RevoluteDriver` (which prescribes relative angle), the
//! `LinearDriver` constrains:
//!
//!     Phi = |P_b - P_a| - d(t) = 0
//!
//! where P_a, P_b are global positions of local attachment points on two bodies.
//! The Lagrange multiplier lambda gives the required actuator force along the
//! line of action.

use nalgebra::{DMatrix, DVector, Vector2};

use crate::core::constraint::Constraint;
use crate::core::driver::{DriverFn, DriverMeta};
use crate::core::state::State;

/// Linear driver: prescribes the distance between two body points as f(t).
///
/// Constraint (1 equation):
///     Phi = |P_b - P_a| - d(t) = 0
///
/// The Lagrange multiplier lambda is the required actuator force (N) along
/// the line connecting the two points.
pub struct LinearDriver {
    id_: String,
    body_a_id_: String,
    point_a: Vector2<f64>,
    body_b_id_: String,
    point_b: Vector2<f64>,
    driver_fn: DriverFn,
    /// Optional metadata for serialization.
    pub meta: Option<DriverMeta>,
}

impl std::fmt::Debug for LinearDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearDriver")
            .field("id", &self.id_)
            .field("body_a_id", &self.body_a_id_)
            .field("body_b_id", &self.body_b_id_)
            .finish()
    }
}

impl LinearDriver {
    /// Return the serialization metadata, if available.
    pub fn meta(&self) -> Option<&DriverMeta> {
        self.meta.as_ref()
    }

    /// Local-frame coordinates of point A on body A.
    pub fn point_a(&self) -> [f64; 2] {
        [self.point_a.x, self.point_a.y]
    }

    /// Local-frame coordinates of point B on body B.
    pub fn point_b(&self) -> [f64; 2] {
        [self.point_b.x, self.point_b.y]
    }
}

impl Constraint for LinearDriver {
    fn id(&self) -> &str {
        &self.id_
    }

    fn n_equations(&self) -> usize {
        1
    }

    fn dof_removed(&self) -> usize {
        1
    }

    fn body_i_id(&self) -> &str {
        &self.body_a_id_
    }

    fn body_j_id(&self) -> &str {
        &self.body_b_id_
    }

    fn constraint(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64> {
        let p_a = state.body_point_global(&self.body_a_id_, &self.point_a, q);
        let p_b = state.body_point_global(&self.body_b_id_, &self.point_b, q);
        let d_vec = p_b - p_a;
        let length = d_vec.norm();
        let d_t = (self.driver_fn.f)(t);
        DVector::from_element(1, length - d_t)
    }

    fn phi_t(&self, _state: &State, _q: &DVector<f64>, t: f64) -> DVector<f64> {
        DVector::from_element(1, -(self.driver_fn.f_dot)(t))
    }

    fn jacobian(&self, state: &State, q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(1, n);

        let p_a = state.body_point_global(&self.body_a_id_, &self.point_a, q);
        let p_b = state.body_point_global(&self.body_b_id_, &self.point_b, q);
        let d_vec = p_b - p_a;
        let length = d_vec.norm();

        // Unit direction vector from A to B.
        // Guard against zero length (degenerate configuration).
        if length < 1e-15 {
            return jac;
        }
        let n_hat = d_vec / length;

        // Body A contributions: dPhi/dr_a = -n^T, dPhi/dtheta_a = -n^T * B(theta_a) * s_a
        if !state.is_ground(&self.body_a_id_) {
            let idx_a = state.get_index(&self.body_a_id_).unwrap();
            jac[(0, idx_a.x_idx())] = -n_hat.x;
            jac[(0, idx_a.y_idx())] = -n_hat.y;
            let b_sa = state.body_point_global_derivative(&self.body_a_id_, &self.point_a, q);
            jac[(0, idx_a.theta_idx())] = -n_hat.dot(&b_sa);
        }

        // Body B contributions: dPhi/dr_b = n^T, dPhi/dtheta_b = n^T * B(theta_b) * s_b
        if !state.is_ground(&self.body_b_id_) {
            let idx_b = state.get_index(&self.body_b_id_).unwrap();
            jac[(0, idx_b.x_idx())] = n_hat.x;
            jac[(0, idx_b.y_idx())] = n_hat.y;
            let b_sb = state.body_point_global_derivative(&self.body_b_id_, &self.point_b, q);
            jac[(0, idx_b.theta_idx())] = n_hat.dot(&b_sb);
        }

        jac
    }

    fn gamma(
        &self,
        state: &State,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        t: f64,
    ) -> DVector<f64> {
        let p_a = state.body_point_global(&self.body_a_id_, &self.point_a, q);
        let p_b = state.body_point_global(&self.body_b_id_, &self.point_b, q);
        let d_vec = p_b - p_a;
        let length = d_vec.norm();

        if length < 1e-15 {
            return DVector::from_element(1, (self.driver_fn.f_ddot)(t));
        }

        // Relative velocity of the two attachment points.
        let v_a = state.body_point_velocity(&self.body_a_id_, &self.point_a, q, q_dot);
        let v_b = state.body_point_velocity(&self.body_b_id_, &self.point_b, q, q_dot);
        let d_vec_dot = v_b - v_a;

        // Velocity component along the line of action.
        let v_along = d_vec.dot(&d_vec_dot) / length;

        // Perpendicular velocity squared: |v_rel|^2 - v_along^2
        let v_perp_sq = d_vec_dot.norm_squared() - v_along * v_along;

        // Centripetal (velocity-quadratic) acceleration of the attachment points.
        // For body i: the centripetal term is -A_i * s_i * theta_dot_i^2
        // The relative centripetal acceleration is:
        //   d_vec_ddot_vel = A_a * s_a * theta_a_dot^2 - A_b * s_b * theta_b_dot^2
        //                  (note: d_vec = P_b - P_a, so P_a terms are negated)
        let mut centripetal = Vector2::zeros();
        if !state.is_ground(&self.body_a_id_) {
            let idx_a = state.get_index(&self.body_a_id_).unwrap();
            let theta_a = state.get_angle(&self.body_a_id_, q);
            let theta_a_dot = q_dot[idx_a.theta_idx()];
            let a_a = State::rotation_matrix(theta_a);
            centripetal += (a_a * &self.point_a) * theta_a_dot.powi(2);
        }
        if !state.is_ground(&self.body_b_id_) {
            let idx_b = state.get_index(&self.body_b_id_).unwrap();
            let theta_b = state.get_angle(&self.body_b_id_, q);
            let theta_b_dot = q_dot[idx_b.theta_idx()];
            let a_b = State::rotation_matrix(theta_b);
            centripetal -= (a_b * &self.point_b) * theta_b_dot.powi(2);
        }

        // gamma = d''(t) - v_perp^2 / L - (d_vec . centripetal) / L
        let gamma_val =
            (self.driver_fn.f_ddot)(t) - v_perp_sq / length - d_vec.dot(&centripetal) / length;
        DVector::from_element(1, gamma_val)
    }
}

/// Create a constant-velocity linear driver.
///
/// d(t) = length_0 + velocity * t
/// d'(t) = velocity
/// d''(t) = 0
///
/// The resulting driver stores `DriverMeta::LinearLength` so it can be
/// round-tripped through JSON serialization.
pub fn constant_velocity_linear_driver(
    id: &str,
    body_a: &str,
    point_a: [f64; 2],
    body_b: &str,
    point_b: [f64; 2],
    velocity: f64,
    length_0: f64,
) -> LinearDriver {
    LinearDriver {
        id_: id.to_string(),
        body_a_id_: body_a.to_string(),
        point_a: Vector2::new(point_a[0], point_a[1]),
        body_b_id_: body_b.to_string(),
        point_b: Vector2::new(point_b[0], point_b[1]),
        driver_fn: DriverFn {
            f: Box::new(move |t| length_0 + velocity * t),
            f_dot: Box::new(move |_t| velocity),
            f_ddot: Box::new(|_t| 0.0),
        },
        meta: Some(DriverMeta::LinearLength { velocity, length_0 }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Build a simple test setup:
    /// - Ground with point O at origin (0,0)
    /// - Bar "bar" with local points A=(0,0) and B=(1,0)
    /// - Revolute joint J1 between ground O and bar A
    /// - Bar starts at angle=0, so A is at (0,0) global, B is at (1,0) global
    fn test_state_and_q() -> (State, DVector<f64>) {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        // bar at origin, angle=0 => A(0,0)=>(0,0), B(1,0)=>(1,0)
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        (state, q)
    }

    #[test]
    fn linear_driver_constraint_at_rest() {
        let (state, q) = test_state_and_q();

        // Ground O is at (0,0), bar B is at (1,0) => distance = 1.0
        // d(t=0) = length_0 = 1.0, so Phi should be 0.
        let driver = constant_velocity_linear_driver(
            "LD1", "ground", [0.0, 0.0], "bar", [1.0, 0.0], 0.5, 1.0,
        );

        let phi = driver.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn linear_driver_constraint_nonzero_when_mismatched() {
        let (state, q) = test_state_and_q();

        // Ground O at (0,0), bar B at (1,0) => distance = 1.0
        // d(t=0) = 0.8 => Phi = 1.0 - 0.8 = 0.2
        let driver = constant_velocity_linear_driver(
            "LD1", "ground", [0.0, 0.0], "bar", [1.0, 0.0], 0.0, 0.8,
        );

        let phi = driver.constraint(&state, &q, 0.0);
        assert_abs_diff_eq!(phi[0], 0.2, epsilon = 1e-14);
    }

    #[test]
    fn linear_driver_phi_t() {
        let (state, q) = test_state_and_q();
        let velocity = 0.5;
        let driver = constant_velocity_linear_driver(
            "LD1", "ground", [0.0, 0.0], "bar", [1.0, 0.0], velocity, 1.0,
        );

        let phi_t = driver.phi_t(&state, &q, 1.0);
        assert_abs_diff_eq!(phi_t[0], -velocity, epsilon = 1e-14);
    }

    #[test]
    fn linear_driver_jacobian_is_correct() {
        // Use a non-trivial configuration: bar rotated to 45 degrees.
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        let theta = std::f64::consts::FRAC_PI_4;
        state.set_pose("bar", &mut q, 0.0, 0.0, theta);

        let driver = constant_velocity_linear_driver(
            "LD1", "ground", [0.0, 0.0], "bar", [1.0, 0.0], 0.5, 1.0,
        );

        let jac_analytical = driver.jacobian(&state, &q, 0.0);

        // Finite-difference Jacobian
        let eps = 1e-7;
        let n = state.n_coords();
        let mut jac_fd = DMatrix::zeros(1, n);
        for col in 0..n {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[col] += eps;
            q_minus[col] -= eps;
            let phi_plus = driver.constraint(&state, &q_plus, 0.0);
            let phi_minus = driver.constraint(&state, &q_minus, 0.0);
            jac_fd[(0, col)] = (phi_plus[0] - phi_minus[0]) / (2.0 * eps);
        }

        for col in 0..n {
            assert_abs_diff_eq!(
                jac_analytical[(0, col)],
                jac_fd[(0, col)],
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn linear_driver_jacobian_two_moving_bodies() {
        // Both bodies are moving (neither is ground).
        let mut state = State::new();
        state.register_body("bar_a").unwrap();
        state.register_body("bar_b").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar_a", &mut q, 0.1, 0.2, 0.5);
        state.set_pose("bar_b", &mut q, 0.8, -0.1, 1.2);

        let driver = constant_velocity_linear_driver(
            "LD2", "bar_a", [0.1, 0.05], "bar_b", [0.0, -0.03], 0.0, 1.0,
        );

        let jac_analytical = driver.jacobian(&state, &q, 0.0);

        let eps = 1e-7;
        let n = state.n_coords();
        let mut jac_fd = DMatrix::zeros(1, n);
        for col in 0..n {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[col] += eps;
            q_minus[col] -= eps;
            let phi_plus = driver.constraint(&state, &q_plus, 0.0);
            let phi_minus = driver.constraint(&state, &q_minus, 0.0);
            jac_fd[(0, col)] = (phi_plus[0] - phi_minus[0]) / (2.0 * eps);
        }

        for col in 0..n {
            assert_abs_diff_eq!(
                jac_analytical[(0, col)],
                jac_fd[(0, col)],
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn linear_driver_gamma_is_correct() {
        // Verify gamma via finite differences on phi_dot.
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.7);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.1; // bar x_dot
        q_dot[1] = -0.2; // bar y_dot
        q_dot[2] = 3.0; // bar theta_dot

        let velocity = 0.5;
        let driver = constant_velocity_linear_driver(
            "LD1", "ground", [0.0, 0.0], "bar", [1.0, 0.0], velocity, 1.0,
        );

        let gamma_analytical = driver.gamma(&state, &q, &q_dot, 0.0);

        // FD: advance state by dt, compute phi_dot at both, differentiate.
        let dt = 1e-7;
        let t = 0.0;

        let jac = driver.jacobian(&state, &q, t);
        let phi_t_val = driver.phi_t(&state, &q, t);
        let phi_dot = &jac * &q_dot + &phi_t_val;

        let q_plus = &q + &q_dot * dt;
        let t_plus = t + dt;
        let jac_plus = driver.jacobian(&state, &q_plus, t_plus);
        let phi_t_plus = driver.phi_t(&state, &q_plus, t_plus);
        let phi_dot_plus = &jac_plus * &q_dot + &phi_t_plus;

        let gamma_fd = -(&phi_dot_plus - &phi_dot) / dt;

        assert_abs_diff_eq!(gamma_analytical[0], gamma_fd[0], epsilon = 1e-5);
    }

    #[test]
    fn linear_driver_gamma_two_moving_bodies() {
        // Both bodies moving with velocities -- most complex case.
        let mut state = State::new();
        state.register_body("bar_a").unwrap();
        state.register_body("bar_b").unwrap();
        let mut q = state.make_q();
        state.set_pose("bar_a", &mut q, 0.1, 0.2, 0.8);
        state.set_pose("bar_b", &mut q, 0.7, -0.1, 1.5);

        let mut q_dot = state.make_q();
        q_dot[0] = 0.15;  // bar_a x_dot
        q_dot[1] = -0.1;  // bar_a y_dot
        q_dot[2] = 2.5;   // bar_a theta_dot
        q_dot[3] = -0.2;  // bar_b x_dot
        q_dot[4] = 0.3;   // bar_b y_dot
        q_dot[5] = -1.8;  // bar_b theta_dot

        let driver = constant_velocity_linear_driver(
            "LD2", "bar_a", [0.1, 0.05], "bar_b", [-0.03, 0.02], 0.3, 1.0,
        );

        let gamma_analytical = driver.gamma(&state, &q, &q_dot, 0.0);

        let dt = 1e-7;
        let t = 0.0;

        let jac = driver.jacobian(&state, &q, t);
        let phi_t_val = driver.phi_t(&state, &q, t);
        let phi_dot = &jac * &q_dot + &phi_t_val;

        let q_plus = &q + &q_dot * dt;
        let t_plus = t + dt;
        let jac_plus = driver.jacobian(&state, &q_plus, t_plus);
        let phi_t_plus = driver.phi_t(&state, &q_plus, t_plus);
        let phi_dot_plus = &jac_plus * &q_dot + &phi_t_plus;

        let gamma_fd = -(&phi_dot_plus - &phi_dot) / dt;

        assert_abs_diff_eq!(gamma_analytical[0], gamma_fd[0], epsilon = 1e-5);
    }

    #[test]
    fn linear_driver_meta_stores_parameters() {
        let driver = constant_velocity_linear_driver(
            "LD1", "ground", [0.0, 0.0], "bar", [1.0, 0.0], 0.5, 1.0,
        );
        match driver.meta() {
            Some(DriverMeta::LinearLength { velocity, length_0 }) => {
                assert_abs_diff_eq!(*velocity, 0.5, epsilon = 1e-15);
                assert_abs_diff_eq!(*length_0, 1.0, epsilon = 1e-15);
            }
            other => panic!("Expected LinearLength meta, got {:?}", other),
        }
    }
}
