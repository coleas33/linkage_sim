//! Kinematic solvers: position, velocity, and acceleration.
//!
//! Position: Newton-Raphson on Φ(q, t) = 0
//! Velocity: linear solve Φ_q · q̇ = −Φ_t
//! Acceleration: linear solve Φ_q · q̈ = γ

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::solver::assembly::{
    assemble_constraints, assemble_gamma, assemble_jacobian, assemble_phi_t,
};

/// Result of a kinematic position solve.
#[derive(Debug, Clone)]
pub struct PositionSolveResult {
    /// Converged generalized coordinate vector (or last iterate if failed).
    pub q: DVector<f64>,
    /// True if the solver converged within tolerance.
    pub converged: bool,
    /// Number of Newton-Raphson iterations performed.
    pub iterations: usize,
    /// Final ‖Φ(q, t)‖ at the returned q.
    pub residual_norm: f64,
}

/// Solve Φ(q, t) = 0 using Newton-Raphson.
///
/// At each iteration:
///     Φ_q · Δq = −Φ(q, t)
///     q ← q + Δq
///
/// Convergence when ‖Φ(q, t)‖ < tol.
pub fn solve_position(
    mech: &Mechanism,
    q0: &DVector<f64>,
    t: f64,
    tol: f64,
    max_iter: usize,
) -> PositionSolveResult {
    assert!(mech.is_built(), "Mechanism must be built before solving.");

    let mut q = q0.clone();

    for iteration in 1..=max_iter {
        let phi = assemble_constraints(mech, &q, t);
        let residual_norm = phi.norm();

        if residual_norm < tol {
            return PositionSolveResult {
                q,
                converged: true,
                iterations: iteration,
                residual_norm,
            };
        }

        let phi_q = assemble_jacobian(mech, &q, t);

        // Solve Φ_q · Δq = −Φ via least-squares (SVD decomposition)
        let neg_phi = -phi;
        let svd = phi_q.svd(true, true);
        let delta_q = svd.solve(&neg_phi, 1e-14).unwrap();
        q += delta_q;
    }

    // Final residual check
    let phi = assemble_constraints(mech, &q, t);
    let residual_norm = phi.norm();

    PositionSolveResult {
        q,
        converged: residual_norm < tol,
        iterations: max_iter,
        residual_norm,
    }
}

/// Solve for velocity: Φ_q · q̇ = −Φ_t.
///
/// Single linear solve (no iteration needed).
pub fn solve_velocity(mech: &Mechanism, q: &DVector<f64>, t: f64) -> DVector<f64> {
    assert!(mech.is_built(), "Mechanism must be built before solving.");

    let phi_q = assemble_jacobian(mech, q, t);
    let phi_t = assemble_phi_t(mech, q, t);
    let rhs = -phi_t;

    let svd = phi_q.svd(true, true);
    svd.solve(&rhs, 1e-14).unwrap()
}

/// Solve for acceleration: Φ_q · q̈ = γ.
///
/// Single linear solve (no iteration needed).
pub fn solve_acceleration(
    mech: &Mechanism,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    t: f64,
) -> DVector<f64> {
    assert!(mech.is_built(), "Mechanism must be built before solving.");

    let phi_q = assemble_jacobian(mech, q, t);
    let gamma = assemble_gamma(mech, q, q_dot, t);

    let svd = phi_q.svd(true, true);
    svd.solve(&gamma, 1e-14).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    fn build_fourbar() -> Mechanism {
        // Standard 4-bar: crank(0.01m), coupler(0.04m), rocker(0.03m), ground(0.038m)
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
        let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();

        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
            .unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
            .unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
            .unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4")
            .unwrap();
        mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0)
            .unwrap();

        mech.build().unwrap();
        mech
    }

    #[test]
    fn fourbar_position_solve_converges() {
        let mech = build_fourbar();
        let state = mech.state();

        // Set up initial guess: crank at 0 rad
        let mut q0 = state.make_q();
        // crank at angle 0: tip B at (0.01, 0)
        state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
        // coupler from (0.01, 0) to somewhere near rocker
        state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
        // rocker from coupler end to ground O4=(0.038, 0)
        state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);

        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50);

        assert!(
            result.converged,
            "NR did not converge, residual = {}",
            result.residual_norm
        );
        assert!(result.residual_norm < 1e-10);
    }

    #[test]
    fn fourbar_velocity_solve() {
        let mech = build_fourbar();
        let state = mech.state();

        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
        state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
        state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);

        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50);
        assert!(result.converged);

        let q_dot = solve_velocity(&mech, &result.q, 0.0);

        // Crank velocity: θ̇_crank should be ω = 2π (from constant speed driver)
        let crank_idx = state.get_index("crank").unwrap();
        assert_abs_diff_eq!(q_dot[crank_idx.theta_idx()], 2.0 * PI, epsilon = 1e-8);
    }

    #[test]
    fn fourbar_acceleration_solve() {
        let mech = build_fourbar();
        let state = mech.state();

        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
        state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
        state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);

        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50);
        assert!(result.converged);

        let q_dot = solve_velocity(&mech, &result.q, 0.0);
        let q_ddot = solve_acceleration(&mech, &result.q, &q_dot, 0.0);

        // Crank acceleration: θ̈_crank should be 0 (constant speed)
        let crank_idx = state.get_index("crank").unwrap();
        assert_abs_diff_eq!(q_ddot[crank_idx.theta_idx()], 0.0, epsilon = 1e-6);

        // Acceleration vector should be finite (no NaN/Inf)
        for i in 0..q_ddot.len() {
            assert!(
                q_ddot[i].is_finite(),
                "q_ddot[{}] = {} is not finite",
                i,
                q_ddot[i]
            );
        }
    }

    #[test]
    fn fourbar_position_sweep() {
        let mech = build_fourbar();
        let state = mech.state();

        let omega = 2.0 * PI;
        let n_steps = 12;

        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
        state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
        state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);

        let mut q_prev = q0;
        for i in 0..n_steps {
            let t = i as f64 / n_steps as f64;
            let result = solve_position(&mech, &q_prev, t, 1e-10, 50);
            assert!(
                result.converged,
                "Failed at step {} (t={}), residual = {}",
                i,
                t,
                result.residual_norm
            );

            // Verify crank angle matches driver
            let crank_angle = state.get_angle("crank", &result.q);
            let expected = omega * t;
            assert_abs_diff_eq!(crank_angle, expected, epsilon = 1e-8);

            q_prev = result.q;
        }
    }
}
