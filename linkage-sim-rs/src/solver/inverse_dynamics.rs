//! Inverse dynamics solver: Phi_q^T * lambda = Q - M * q_ddot
//!
//! Given a prescribed motion (q, q_dot, q_ddot) and applied forces Q,
//! solve for the constraint forces (Lagrange multipliers) that enforce
//! the constraints while accounting for inertial loads.
//!
//! This extends static analysis to include acceleration effects.
//! The multiplier for the driver constraint gives the required input
//! torque including inertial effects.

use nalgebra::{DMatrix, DVector};

use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::forces::assembly::assemble_q;
use crate::forces::gravity::Gravity;
use crate::solver::assembly::assemble_jacobian;

/// Result of an inverse dynamics solve.
#[derive(Debug, Clone)]
pub struct InverseDynamicsResult {
    /// Lagrange multiplier vector (m,).
    pub lambdas: DVector<f64>,
    /// Assembled applied generalized force vector (n_coords,).
    pub q_forces: DVector<f64>,
    /// Inertial force vector M * q_ddot (n_coords,).
    pub m_q_ddot: DVector<f64>,
    /// Residual: ||Phi_q^T * lambda - (Q - M*q_ddot)||.
    pub residual_norm: f64,
    /// Condition number of Phi_q.
    pub condition_number: f64,
}

/// Assemble the block-diagonal mass matrix M.
///
/// Each moving body contributes a 3x3 block at its coordinate indices:
///
/// ```text
///   [m,      0,      m*Bs_x ]
///   [0,      m,      m*Bs_y ]
///   [m*Bs_x, m*Bs_y, Izz_cg + m*|s_cg|^2]
/// ```
///
/// where Bs = B(theta) * s_cg is the velocity Jacobian of the CG.
/// When the body coordinate origin coincides with CG (s_cg = 0),
/// this reduces to diag(m, m, Izz_cg).
pub fn assemble_mass_matrix(mech: &Mechanism, q: &DVector<f64>) -> DMatrix<f64> {
    let state = mech.state();
    let n = state.n_coords();
    let mut m_mat = DMatrix::zeros(n, n);

    for (body_id, body) in mech.bodies() {
        if body_id == GROUND_ID {
            continue;
        }

        let idx = state.get_index(body_id).expect("body not registered");
        let mass = body.mass;
        let s_cg = &body.cg_local;

        // Diagonal mass terms
        m_mat[(idx.x_idx(), idx.x_idx())] = mass;
        m_mat[(idx.y_idx(), idx.y_idx())] = mass;

        // M_theta_theta = Izz_cg + m * |s_cg|^2 (parallel axis theorem)
        let s_cg_sq = s_cg.x * s_cg.x + s_cg.y * s_cg.y;
        m_mat[(idx.theta_idx(), idx.theta_idx())] = body.izz_cg + mass * s_cg_sq;

        // Off-diagonal coupling when CG is offset from body origin
        if mass > 0.0 && s_cg_sq > 0.0 {
            let theta = q[idx.theta_idx()];
            let (sin_t, cos_t) = theta.sin_cos();
            // B(theta) * s_cg = [-sin(theta)*sx - cos(theta)*sy, cos(theta)*sx - sin(theta)*sy]
            let bs_x = -sin_t * s_cg.x - cos_t * s_cg.y;
            let bs_y = cos_t * s_cg.x - sin_t * s_cg.y;

            m_mat[(idx.x_idx(), idx.theta_idx())] = mass * bs_x;
            m_mat[(idx.theta_idx(), idx.x_idx())] = mass * bs_x;
            m_mat[(idx.y_idx(), idx.theta_idx())] = mass * bs_y;
            m_mat[(idx.theta_idx(), idx.y_idx())] = mass * bs_y;
        }
    }

    m_mat
}

/// Solve inverse dynamics for constraint forces.
///
/// Phi_q^T * lambda = Q - M * q_ddot
///
/// The RHS includes inertial loads (M * q_ddot). The multiplier for
/// the driver constraint gives the required input torque including
/// inertial effects.
///
/// # Arguments
/// * `mech` - Built mechanism
/// * `q` - Position vector (from position solve)
/// * `q_dot` - Velocity vector (from velocity solve)
/// * `q_ddot` - Acceleration vector (from acceleration solve)
/// * `gravity` - Optional gravity force element
/// * `t` - Time parameter
pub fn solve_inverse_dynamics(
    mech: &Mechanism,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    q_ddot: &DVector<f64>,
    gravity: Option<&Gravity>,
    t: f64,
) -> InverseDynamicsResult {
    assert!(
        mech.is_built(),
        "Mechanism must be built before inverse dynamics."
    );

    let state = mech.state();

    // Assemble Phi_q and Q
    let phi_q = assemble_jacobian(mech, q, t);
    let phi_q_t = phi_q.transpose();
    let q_forces = assemble_q(state, gravity, q, q_dot, t);

    // Mass matrix and inertial term
    let m_mat = assemble_mass_matrix(mech, q);
    let m_q_ddot = &m_mat * q_ddot;

    // RHS = -(Q - M * q_ddot)  [same sign convention as Python]
    let rhs = -(&q_forces - &m_q_ddot);

    // Condition number via SVD of Phi_q
    let svd = phi_q.svd(true, true);
    let sv = &svd.singular_values;

    let condition_number = if sv.len() > 0 && sv[sv.len() - 1] > 0.0 {
        sv[0] / sv[sv.len() - 1]
    } else {
        f64::INFINITY
    };

    // Solve Phi_q^T * lambda = rhs via SVD
    let svd_t = phi_q_t.clone().svd(true, true);
    let lambdas = svd_t.solve(&rhs, 1e-14).unwrap();

    // Compute residual
    let residual = &phi_q_t * &lambdas - &rhs;
    let residual_norm = residual.norm();

    InverseDynamicsResult {
        lambdas,
        q_forces,
        m_q_ddot,
        residual_norm,
        condition_number,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use crate::forces::gravity::Gravity;
    use crate::solver::kinematics::{solve_acceleration, solve_position, solve_velocity};
    use nalgebra::Vector2;
    use std::f64::consts::PI;

    fn build_fourbar_with_gravity() -> (Mechanism, Gravity) {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
        let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);

        let mut bodies = std::collections::HashMap::new();
        bodies.insert("ground".to_string(), ground.clone());
        bodies.insert("crank".to_string(), crank.clone());
        bodies.insert("coupler".to_string(), coupler.clone());
        bodies.insert("rocker".to_string(), rocker.clone());

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
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
            .unwrap();
        mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();

        mech.build().unwrap();

        let gravity = Gravity::new(Vector2::new(0.0, -9.81), &bodies);
        (mech, gravity)
    }

    #[test]
    fn inverse_dynamics_solves() {
        let (mech, gravity) = build_fourbar_with_gravity();
        let state = mech.state();

        let angle = PI / 3.0;
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        let bx = angle.cos();
        let by = angle.sin();
        state.set_pose("coupler", &mut q0, bx, by, 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let pos = solve_position(&mech, &q0, angle, 1e-10, 50);
        assert!(pos.converged);

        let q_dot = solve_velocity(&mech, &pos.q, angle);
        let q_ddot = solve_acceleration(&mech, &pos.q, &q_dot, angle);

        let result = solve_inverse_dynamics(&mech, &pos.q, &q_dot, &q_ddot, Some(&gravity), angle);
        assert!(
            result.residual_norm < 1e-8,
            "residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn inverse_dynamics_reduces_to_statics_at_zero_acceleration() {
        // When q_dot = 0 and q_ddot = 0, inverse dynamics should give same result as statics
        let (mech, gravity) = build_fourbar_with_gravity();
        let state = mech.state();

        let angle = PI / 4.0;
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("coupler", &mut q0, angle.cos(), angle.sin(), 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let pos = solve_position(&mech, &q0, angle, 1e-10, 50);
        assert!(pos.converged);

        let n = state.n_coords();
        let q_dot_zero = DVector::zeros(n);
        let q_ddot_zero = DVector::zeros(n);

        let inv_dyn = solve_inverse_dynamics(
            &mech,
            &pos.q,
            &q_dot_zero,
            &q_ddot_zero,
            Some(&gravity),
            angle,
        );
        let statics =
            crate::solver::statics::solve_statics(&mech, &pos.q, Some(&gravity), angle);

        // M*q_ddot should be zero
        for i in 0..n {
            assert!(
                inv_dyn.m_q_ddot[i].abs() < 1e-15,
                "M*q_ddot[{}] = {} should be zero",
                i,
                inv_dyn.m_q_ddot[i]
            );
        }

        // Lambdas should match statics result
        let lam_diff = (&inv_dyn.lambdas - &statics.lambdas).norm();
        assert!(
            lam_diff < 1e-8,
            "Lambdas differ from statics: norm diff = {:e}",
            lam_diff
        );
    }

    #[test]
    fn mass_matrix_diagonal_for_cg_at_origin() {
        // Body with CG at origin should produce purely diagonal mass matrix
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let mut bar = make_bar("bar", "A", "B", 1.0, 5.0, 0.1);
        bar.cg_local = Vector2::new(0.0, 0.0); // CG at body origin

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.add_revolute_driver("D1", "ground", "bar", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();
        mech.build().unwrap();

        let state = mech.state();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, PI / 6.0);

        let m_mat = assemble_mass_matrix(&mech, &q);

        // Should be diag(5, 5, 0.1)
        assert!((m_mat[(0, 0)] - 5.0).abs() < 1e-14);
        assert!((m_mat[(1, 1)] - 5.0).abs() < 1e-14);
        assert!((m_mat[(2, 2)] - 0.1).abs() < 1e-14);

        // Off-diagonals should be zero
        assert!(m_mat[(0, 1)].abs() < 1e-14);
        assert!(m_mat[(0, 2)].abs() < 1e-14);
        assert!(m_mat[(1, 2)].abs() < 1e-14);
    }

    #[test]
    fn mass_matrix_has_coupling_for_offset_cg() {
        // Body with CG offset from origin should have off-diagonal terms
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 5.0, 0.1); // CG at (0.5, 0)

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.add_revolute_driver("D1", "ground", "bar", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();
        mech.build().unwrap();

        let state = mech.state();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, PI / 4.0);

        let m_mat = assemble_mass_matrix(&mech, &q);

        // Diagonal terms
        assert!((m_mat[(0, 0)] - 5.0).abs() < 1e-14);
        assert!((m_mat[(1, 1)] - 5.0).abs() < 1e-14);
        // M_theta_theta = Izz_cg + m * |s_cg|^2 = 0.1 + 5.0 * 0.25 = 1.35
        assert!((m_mat[(2, 2)] - 1.35).abs() < 1e-14);

        // Off-diagonal coupling should be nonzero with offset CG
        assert!(m_mat[(0, 2)].abs() > 1e-10, "Expected nonzero coupling (0,2)");
        assert!(m_mat[(1, 2)].abs() > 1e-10, "Expected nonzero coupling (1,2)");

        // Matrix should be symmetric
        assert!(
            (m_mat[(0, 2)] - m_mat[(2, 0)]).abs() < 1e-14,
            "Mass matrix not symmetric"
        );
        assert!(
            (m_mat[(1, 2)] - m_mat[(2, 1)]).abs() < 1e-14,
            "Mass matrix not symmetric"
        );
    }

    #[test]
    fn mass_matrix_skips_ground() {
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 3.0, 0.05);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.add_revolute_driver("D1", "ground", "bar", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();
        mech.build().unwrap();

        let q = mech.state().make_q();
        let m_mat = assemble_mass_matrix(&mech, &q);

        // Only 3x3 for one moving body
        assert_eq!(m_mat.nrows(), 3);
        assert_eq!(m_mat.ncols(), 3);
    }
}
