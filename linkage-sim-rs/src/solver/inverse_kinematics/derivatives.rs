//! Closed-form inverse velocity and finite-difference inverse acceleration helpers.
//!
//! See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8.2 (velocity)
//!      docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8.3 (acceleration)

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::solver::assembly::assemble_jacobian;

use super::control_target::ControlTarget;

/// Solve for `u̇` such that `ġ(q(u)) = h_dot`, given converged `q` at the current `u`.
///
/// Closed-form: `u̇ = h_dot / r'(u)` where `r'(u) = ∇_q g · dq/du = −∇_q g · Φ_q⁻¹ Φ_u`.
///
/// Returns `Err(TrajectorySingular)` if `|r'(u)|` is below the singularity threshold.
pub fn inverse_velocity(
    mech: &Mechanism,
    q: &DVector<f64>,
    target: &ControlTarget,
    h_dot: f64,
    t_mech: f64,
) -> Result<f64, LinkageError> {
    let driver_row = mech.driver_row();
    let phi_q = assemble_jacobian(mech, q, t_mech);
    let mut phi_u = DVector::zeros(mech.n_constraints());
    phi_u[driver_row] = -1.0;
    let svd = phi_q.svd(true, true);
    let dq_du = svd.solve(&-phi_u, 1e-14)
        .map_err(|_| LinkageError::SvdSolveFailed)?;

    let grad = target.gradient(mech, q);
    let r_prime = grad.dot(&dq_du);

    let grad_norm = grad.norm();
    let dqdu_norm = dq_du.norm();
    let eps_singularity = 1e-6 * (grad_norm * dqdu_norm).max(1e-12);
    if r_prime.abs() < eps_singularity {
        return Err(LinkageError::TrajectorySingular { dg_du: r_prime });
    }

    Ok(h_dot / r_prime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::inverse_kinematics::test_helpers::{build_fourbar, solve_at};
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    fn inverse_velocity_for_angle_target_is_h_dot() {
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.1);
        // For ControlTarget::Angle on the directly-driven body, dg/du = 1.
        // So u_dot = h_dot.
        let target = ControlTarget::angle("crank");
        let h_dot = 0.5;
        let u_dot = inverse_velocity(&mech, &q, &target, h_dot, 0.1).unwrap();
        assert_abs_diff_eq!(u_dot, h_dot, epsilon = 1e-7);
    }

    #[test]
    fn inverse_velocity_consistent_with_fd() {
        // FD check: vary u by δ and compute (g_plus − g_minus)/(2δ); verify
        // that u_dot = h_dot / dg_du matches.
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.2);
        let target = ControlTarget::world_y("crank", [0.005, 0.0]);
        let h_dot = 1.0;
        let u_dot = inverse_velocity(&mech, &q, &target, h_dot, 0.2).unwrap();
        // FD verification of dg/du
        let omega = 2.0 * PI;
        let δ = 1e-6;
        let q_plus = solve_at(&mech, 0.2 + δ / omega);
        let q_minus = solve_at(&mech, 0.2 - δ / omega);
        let dg_du_fd = (target.evaluate(&mech, &q_plus) - target.evaluate(&mech, &q_minus)) / (2.0 * δ);
        // u_dot * dg_du should equal h_dot
        assert_abs_diff_eq!(u_dot * dg_du_fd, h_dot, epsilon = 1e-3);
    }
}
