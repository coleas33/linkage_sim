//! Closed-form inverse velocity and finite-difference inverse acceleration helpers.
//!
//! See: docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8.2 (velocity)
//!      docs/superpowers/specs/2026-04-29-linkage-equations-reference.md §8.3 (acceleration)

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::solver::assembly::assemble_jacobian;

use super::control_target::ControlTarget;

/// Compute `dq/du` at the given configuration: `dq/du = -Φ_q⁻¹ Φ_u`.
///
/// `Φ_u` has a single −1 entry on the driver row, 0 elsewhere (since Φ depends
/// on u only through `−u` after re-parameterization). Returns `Err(SvdSolveFailed)`
/// if the SVD pseudo-inverse fails.
pub(super) fn compute_dq_du(
    mech: &Mechanism,
    q: &DVector<f64>,
    t_mech: f64,
) -> Result<DVector<f64>, LinkageError> {
    let driver_row = mech.driver_row();
    let phi_q = assemble_jacobian(mech, q, t_mech);
    let mut phi_u = DVector::zeros(mech.n_constraints());
    phi_u[driver_row] = -1.0;
    let svd = phi_q.svd(true, true);
    svd.solve(&-phi_u, 1e-14)
        .map_err(|_| LinkageError::SvdSolveFailed)
}

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
    let dq_du = compute_dq_du(mech, q, t_mech)?;
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

/// Compute `ü` such that `g̈(q(u(t))) = h_ddot`, by finite-differencing `r'(u)`.
///
/// `ü = (h_ddot − r''(u) · u̇²) / r'(u)`, where:
///   `r'(u) = ∇_q g · dq/du`
///   `r''(u) ≈ (r'(u + δ) − r'(u − δ)) / (2δ)`
///
/// FD sidesteps the analytic `∇²_q g` term needed for option (a) in EQ §8.3.
/// Two extra forward solves per call.
#[allow(clippy::too_many_arguments)]
pub fn inverse_acceleration_fd(
    mech: &Mechanism,
    q: &DVector<f64>,
    target: &ControlTarget,
    u: f64,
    u_dot: f64,
    h_ddot: f64,
    delta: f64,
    u_0: f64,
    nominal_rate: f64,
) -> Result<f64, LinkageError> {
    use crate::solver::kinematics::solve_position;

    let r_prime_now = compute_r_prime(mech, q, target, (u - u_0) / nominal_rate)?;

    let q_plus = solve_position(
        mech, q, (u + delta - u_0) / nominal_rate, 1e-10, 50,
    )?.q;
    let r_prime_plus = compute_r_prime(
        mech, &q_plus, target, (u + delta - u_0) / nominal_rate,
    )?;

    let q_minus = solve_position(
        mech, q, (u - delta - u_0) / nominal_rate, 1e-10, 50,
    )?.q;
    let r_prime_minus = compute_r_prime(
        mech, &q_minus, target, (u - delta - u_0) / nominal_rate,
    )?;

    let r_double_prime = (r_prime_plus - r_prime_minus) / (2.0 * delta);

    // Singularity guard on r'(u_now)
    let grad = target.gradient(mech, q);
    let dq_du = compute_dq_du(mech, q, (u - u_0) / nominal_rate)?;
    let grad_norm = grad.norm();
    let dqdu_norm = dq_du.norm();
    let eps_singularity = 1e-6 * (grad_norm * dqdu_norm).max(1e-12);
    if r_prime_now.abs() < eps_singularity {
        return Err(LinkageError::TrajectorySingular { dg_du: r_prime_now });
    }

    Ok((h_ddot - r_double_prime * u_dot * u_dot) / r_prime_now)
}

fn compute_r_prime(
    mech: &Mechanism,
    q: &DVector<f64>,
    target: &ControlTarget,
    t_mech: f64,
) -> Result<f64, LinkageError> {
    let dq_du = compute_dq_du(mech, q, t_mech)?;
    let grad = target.gradient(mech, q);
    Ok(grad.dot(&dq_du))
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

    #[test]
    fn inverse_acceleration_fd_matches_zero_for_constant_velocity() {
        // For a constant-velocity trajectory, ḧ = 0 and u̇ = const, so ü should be 0.
        let mech = build_fourbar();
        let q = solve_at(&mech, 0.1);
        let target = ControlTarget::angle("crank");
        let u_0 = 0.0;
        let nominal_rate = 2.0 * PI;
        let u = 0.1 * nominal_rate; // u_k corresponding to t=0.1
        let u_dot = 0.5; // constant
        let h_ddot = 0.0;
        let δ = 1e-4;
        let u_ddot = inverse_acceleration_fd(
            &mech, &q, &target, u, u_dot, h_ddot, δ,
            u_0, nominal_rate,
        ).unwrap();
        assert_abs_diff_eq!(u_ddot, 0.0, epsilon = 1e-3);
    }
}
