//! Driver constraints: prescribed motion as constraint equations.
//!
//! A driver adds one constraint equation to the system, prescribing
//! a kinematic variable as a function of time. The associated Lagrange
//! multiplier gives the required actuator effort.

use nalgebra::{DMatrix, DVector};

use crate::core::constraint::Constraint;
use crate::core::state::State;

/// Driver function triple: f(t), f'(t), f''(t).
///
/// In Rust we use boxed closures instead of Python lambdas.
/// For serialization, the Rust port will add expression-based drivers
/// (meval/rhai) in a future step.
pub struct DriverFn {
    pub f: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    pub f_dot: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    pub f_ddot: Box<dyn Fn(f64) -> f64 + Send + Sync>,
}

impl std::fmt::Debug for DriverFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverFn").finish_non_exhaustive()
    }
}

/// Revolute driver: prescribes relative angle between two bodies.
///
/// Constraint (1 equation):
///     Φ = θⱼ − θᵢ − f(t) = 0
///
/// The Lagrange multiplier λ is the required input torque (N·m).
pub struct RevoluteDriver {
    id_: String,
    body_i_id_: String,
    body_j_id_: String,
    driver_fn: DriverFn,
}

impl std::fmt::Debug for RevoluteDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RevoluteDriver")
            .field("id", &self.id_)
            .field("body_i_id", &self.body_i_id_)
            .field("body_j_id", &self.body_j_id_)
            .finish()
    }
}

impl Constraint for RevoluteDriver {
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
        &self.body_i_id_
    }
    fn body_j_id(&self) -> &str {
        &self.body_j_id_
    }

    fn constraint(&self, state: &State, q: &DVector<f64>, t: f64) -> DVector<f64> {
        let theta_i = state.get_angle(&self.body_i_id_, q);
        let theta_j = state.get_angle(&self.body_j_id_, q);
        DVector::from_element(1, theta_j - theta_i - (self.driver_fn.f)(t))
    }

    fn phi_t(&self, _state: &State, _q: &DVector<f64>, t: f64) -> DVector<f64> {
        DVector::from_element(1, -(self.driver_fn.f_dot)(t))
    }

    fn jacobian(&self, state: &State, _q: &DVector<f64>, _t: f64) -> DMatrix<f64> {
        let n = state.n_coords();
        let mut jac = DMatrix::zeros(1, n);

        if !state.is_ground(&self.body_i_id_) {
            let idx_i = state.get_index(&self.body_i_id_).unwrap();
            jac[(0, idx_i.theta_idx())] = -1.0;
        }

        if !state.is_ground(&self.body_j_id_) {
            let idx_j = state.get_index(&self.body_j_id_).unwrap();
            jac[(0, idx_j.theta_idx())] = 1.0;
        }

        jac
    }

    fn gamma(
        &self,
        _state: &State,
        _q: &DVector<f64>,
        _q_dot: &DVector<f64>,
        t: f64,
    ) -> DVector<f64> {
        // γ = f̈(t) — Jacobian is constant in q, so no velocity-quadratic terms
        DVector::from_element(1, (self.driver_fn.f_ddot)(t))
    }
}

/// Create a revolute driver that prescribes relative angle vs. time.
pub fn make_revolute_driver(
    driver_id: &str,
    body_i_id: &str,
    body_j_id: &str,
    f: impl Fn(f64) -> f64 + Send + Sync + 'static,
    f_dot: impl Fn(f64) -> f64 + Send + Sync + 'static,
    f_ddot: impl Fn(f64) -> f64 + Send + Sync + 'static,
) -> RevoluteDriver {
    RevoluteDriver {
        id_: driver_id.to_string(),
        body_i_id_: body_i_id.to_string(),
        body_j_id_: body_j_id.to_string(),
        driver_fn: DriverFn {
            f: Box::new(f),
            f_dot: Box::new(f_dot),
            f_ddot: Box::new(f_ddot),
        },
    }
}

/// Create a revolute driver with constant angular velocity.
///
/// f(t) = theta_0 + omega * t
/// f'(t) = omega
/// f''(t) = 0
pub fn constant_speed_driver(
    driver_id: &str,
    body_i_id: &str,
    body_j_id: &str,
    omega: f64,
    theta_0: f64,
) -> RevoluteDriver {
    make_revolute_driver(
        driver_id,
        body_i_id,
        body_j_id,
        move |t| theta_0 + omega * t,
        move |_t| omega,
        |_t| 0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn constant_speed_driver_constraint_at_t0() {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        let mut q = state.make_q();
        state.set_pose("crank", &mut q, 0.0, 0.0, 0.5);

        let driver = constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.5);

        let phi = driver.constraint(&state, &q, 0.0);
        // θ_crank - 0 - f(0) = 0.5 - 0.5 = 0
        assert_abs_diff_eq!(phi[0], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn constant_speed_driver_phi_t() {
        let state = State::new();
        let q = DVector::zeros(0);
        let omega = 2.0 * PI;
        let driver = constant_speed_driver("D1", "ground", "ground", omega, 0.0);

        let phi_t = driver.phi_t(&state, &q, 1.0);
        assert_abs_diff_eq!(phi_t[0], -omega, epsilon = 1e-14);
    }

    #[test]
    fn constant_speed_driver_gamma() {
        let state = State::new();
        let q = DVector::zeros(0);
        let q_dot = DVector::zeros(0);
        let driver = constant_speed_driver("D1", "ground", "ground", 2.0 * PI, 0.0);

        let gam = driver.gamma(&state, &q, &q_dot, 1.0);
        assert_abs_diff_eq!(gam[0], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn driver_jacobian_structure() {
        let mut state = State::new();
        state.register_body("crank").unwrap();
        state.register_body("rocker").unwrap();
        let q = state.make_q();

        let driver = constant_speed_driver("D1", "ground", "crank", 1.0, 0.0);
        let jac = driver.jacobian(&state, &q, 0.0);

        assert_eq!(jac.nrows(), 1);
        assert_eq!(jac.ncols(), 6);
        // θ_crank is at index 2
        assert_abs_diff_eq!(jac[(0, 2)], 1.0, epsilon = 1e-15);
        // All others zero
        assert_abs_diff_eq!(jac[(0, 0)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(jac[(0, 1)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(jac[(0, 3)], 0.0, epsilon = 1e-15);
    }
}
