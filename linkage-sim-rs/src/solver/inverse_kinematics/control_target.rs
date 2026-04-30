//! `ControlTarget` enum: scalar output observables that can be controlled.
//!
//! Each variant implements `g(q)`, `∇_q g(q)`, and `∇²_q g(q)`. These feed the
//! inverse-Newton outer loop in `solver.rs` and the closed-form / FD helpers
//! in `derivatives.rs`.
//!
//! See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8
//!      docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §5.1

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::core::mechanism::Mechanism;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ControlTarget {
    /// Body angle θ_body. Linear in q (Hessian = 0).
    Angle { body_id: String },
    // Other variants added in subsequent tasks.
}

impl ControlTarget {
    /// Evaluate `g(q)` — scalar value of the observable at configuration `q`.
    pub fn evaluate(&self, mech: &Mechanism, q: &DVector<f64>) -> f64 {
        match self {
            ControlTarget::Angle { body_id } => mech.state().get_angle(body_id, q),
        }
    }

    /// Gradient `∇_q g(q)` — length n_coords.
    pub fn gradient(&self, mech: &Mechanism, q: &DVector<f64>) -> DVector<f64> {
        let state = mech.state();
        let n = state.n_coords();
        let mut grad = DVector::zeros(n);

        match self {
            ControlTarget::Angle { body_id } => {
                if !state.is_ground(body_id) {
                    let idx = state.get_index(body_id).expect("body not registered");
                    grad[idx.theta_idx()] = 1.0;
                }
            }
        }

        let _ = q; // q unused for Angle (gradient is constant)
        grad
    }

    /// Hessian `∇²_q g(q)` — n_coords × n_coords. Zero for variants linear in q.
    pub fn hessian(&self, mech: &Mechanism, q: &DVector<f64>) -> DMatrix<f64> {
        let n = mech.state().n_coords();
        let _ = q;
        match self {
            ControlTarget::Angle { .. } => DMatrix::zeros(n, n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_helpers::{build_fourbar, solve_at};
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn angle_target_evaluates_correctly() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.25); // crank at 0.25 rev = π/2
        let target = ControlTarget::Angle { body_id: "crank".into() };
        let g = target.evaluate(&mech, &q);
        assert_abs_diff_eq!(g, PI / 2.0, epsilon = 1e-8);
    }

    #[test]
    fn angle_target_gradient_is_unit_at_theta() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0);
        let target = ControlTarget::Angle { body_id: "crank".into() };
        let grad = target.gradient(&mech, &q);
        let crank_idx = mech.state().get_index("crank").unwrap();
        assert_abs_diff_eq!(grad[crank_idx.theta_idx()], 1.0, epsilon = 1e-15);
        // All other entries zero
        for i in 0..grad.len() {
            if i != crank_idx.theta_idx() {
                assert_abs_diff_eq!(grad[i], 0.0, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn angle_target_hessian_is_zero() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.0);
        let target = ControlTarget::Angle { body_id: "crank".into() };
        let h = target.hessian(&mech, &q);
        let n = mech.state().n_coords();
        assert_eq!(h.nrows(), n);
        assert_eq!(h.ncols(), n);
        for i in 0..n {
            for j in 0..n {
                assert_abs_diff_eq!(h[(i, j)], 0.0, epsilon = 1e-15);
            }
        }
    }
}
