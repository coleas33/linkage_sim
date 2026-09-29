//! Inverse dynamics solver: Phi_q^T * lambda = -(Q + Q_v - M * q_ddot)
//!
//! Given a prescribed motion (q, q_dot, q_ddot) and applied forces Q,
//! solve for the constraint forces (Lagrange multipliers) that enforce
//! the constraints while accounting for inertial loads. Q_v is the
//! velocity-quadratic (centripetal) force from the configuration-dependent
//! mass matrix (see `assemble_quadratic_velocity_forces`).
//!
//! This extends static analysis to include acceleration effects.
//! The multiplier for the driver constraint gives the required input
//! torque including inertial effects.

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::solver::assembly::{
    assemble_jacobian, assemble_mass_matrix, assemble_quadratic_velocity_forces,
};
use crate::solver::condition::rank_aware_condition_number;

/// Result of an inverse dynamics solve.
#[derive(Debug, Clone)]
pub struct InverseDynamicsResult {
    /// Lagrange multiplier vector (m,).
    pub lambdas: DVector<f64>,
    /// Assembled applied generalized force vector (n_coords,).
    pub q_forces: DVector<f64>,
    /// Inertial force vector M * q_ddot (n_coords,).
    pub m_q_ddot: DVector<f64>,
    /// Residual: ||Phi_q^T * lambda + (Q + Q_v - M*q_ddot)||.
    pub residual_norm: f64,
    /// True if pseudoinverse was used (overconstrained system).
    pub is_overconstrained: bool,
    /// Condition number of Phi_q (rank-aware, filtering near-zero SVs).
    pub condition_number: f64,
}

/// Solve inverse dynamics for constraint forces.
///
/// Phi_q^T * lambda = -(Q + Q_v - M * q_ddot)
///
/// The RHS includes inertial loads (M * q_ddot and the velocity-quadratic
/// force Q_v). The multiplier for
/// the driver constraint gives the required input torque including
/// inertial effects.
///
/// # Arguments
/// * `mech` - Built mechanism
/// * `q` - Position vector (from position solve)
/// * `q_dot` - Velocity vector (from velocity solve)
/// * `q_ddot` - Acceleration vector (from acceleration solve)
/// * `t` - Time parameter
pub fn solve_inverse_dynamics(
    mech: &Mechanism,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    q_ddot: &DVector<f64>,
    t: f64,
) -> Result<InverseDynamicsResult, LinkageError> {
    if !mech.is_built() {
        return Err(LinkageError::MechanismNotBuilt);
    }

    // Assemble Phi_q and Q
    let phi_q = assemble_jacobian(mech, q, t);
    let phi_q_t = phi_q.transpose();
    let q_forces = mech.assemble_forces(q, q_dot, t);

    // Mass matrix, inertial term and velocity-quadratic force
    let m_mat = assemble_mass_matrix(mech, q);
    let m_q_ddot = &m_mat * q_ddot;
    let q_v = assemble_quadratic_velocity_forces(mech, q, q_dot);

    // RHS = -(Q + Q_v - M * q_ddot). Same sign convention as the Python
    // reference (solvers/inverse_dynamics.py), which does not include Q_v
    // yet (BL-028).
    let rhs = -(&q_forces + &q_v - &m_q_ddot);

    // Single SVD of Phi_q^T — reused for both conditioning and solve.
    // Singular values of A^T are the same as those of A, so this gives
    // the same condition number as SVD(Phi_q).
    let svd_t = phi_q_t.clone().svd(true, true);
    let sv = &svd_t.singular_values;

    let m = phi_q.nrows(); // n_constraints
    let (condition_number, is_overconstrained) =
        rank_aware_condition_number(sv.as_slice(), m);

    // Solve Phi_q^T * lambda = rhs using the already-computed SVD
    let lambdas = svd_t
        .solve(&rhs, 1e-14)
        .map_err(|_| LinkageError::SvdSolveFailed)?;

    // Compute residual
    let residual = &phi_q_t * &lambdas - &rhs;
    let residual_norm = residual.norm();

    Ok(InverseDynamicsResult {
        lambdas,
        q_forces,
        m_q_ddot,
        residual_norm,
        is_overconstrained,
        condition_number,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::energy::compute_kinetic_energy;
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use crate::forces::elements::{ForceElement, GravityElement};
    // Test-only use of the pure-geometry loop closure (no gui state involved).
    use crate::gui::samples::helpers::try_fourbar_initial_q0;
    use crate::solver::kinematics::{solve_acceleration, solve_position, solve_velocity};
    use crate::solver::statics::{extract_reactions, StaticSolveResult};
    use nalgebra::Vector2;
    use std::f64::consts::PI;

    fn build_fourbar_with_gravity() -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
        let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);

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

        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.build().unwrap();
        mech
    }

    #[test]
    fn inverse_dynamics_solves() {
        let mech = build_fourbar_with_gravity();
        let state = mech.state();

        let angle = PI / 3.0;
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        let bx = angle.cos();
        let by = angle.sin();
        state.set_pose("coupler", &mut q0, bx, by, 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(pos.converged);

        let q_dot = solve_velocity(&mech, &pos.q, angle).unwrap();
        let q_ddot = solve_acceleration(&mech, &pos.q, &q_dot, angle).unwrap();

        let result = solve_inverse_dynamics(&mech, &pos.q, &q_dot, &q_ddot, angle).unwrap();
        assert!(
            result.residual_norm < 1e-8,
            "residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn inverse_dynamics_reduces_to_statics_at_zero_acceleration() {
        // When q_dot = 0 and q_ddot = 0, inverse dynamics should give same result as statics
        let mech = build_fourbar_with_gravity();
        let state = mech.state();

        let angle = PI / 4.0;
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("coupler", &mut q0, angle.cos(), angle.sin(), 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let pos = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(pos.converged);

        let n = state.n_coords();
        let q_dot_zero = DVector::zeros(n);
        let q_ddot_zero = DVector::zeros(n);

        let inv_dyn = solve_inverse_dynamics(
            &mech,
            &pos.q,
            &q_dot_zero,
            &q_ddot_zero,
            angle,
        ).unwrap();
        let statics =
            crate::solver::statics::solve_statics(&mech, &pos.q, angle).unwrap();

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

    // ── BL-027: velocity-quadratic (centripetal) term ────────────────────────
    //
    // Energy identity used below: dotting Phi_q^T * lambda = rhs with q_dot,
    // and using Phi_q * q_dot = 0 on every scleronomic joint row, gives
    //     lambda_driver * (Phi_q * q_dot)_driver = rhs . q_dot.
    // With gravity-only (velocity-independent) Q, rhs_ID - rhs_statics =
    // M * q_ddot - Q_v, and Lagrange's energy identity gives
    // q_dot . (M * q_ddot - Q_v) = dKE/dt. The driver multipliers therefore
    // differ by exactly
    //     (tau_ID - tau_statics) * rate = dKE/dt,
    // where rate = (Phi_q * q_dot)_driver is the driver's angular rate.

    /// Driver multiplier. The driver is the last constraint (joints are
    /// stacked before drivers), which is also how the sweep reads it.
    fn driver_lambda(lambdas: &DVector<f64>) -> f64 {
        lambdas[lambdas.len() - 1]
    }

    /// Driver rate (Phi_q * q_dot)_driver from the assembled Jacobian.
    fn driver_rate(mech: &Mechanism, q: &DVector<f64>, q_dot: &DVector<f64>, t: f64) -> f64 {
        let phi_q_qdot = assemble_jacobian(mech, q, t) * q_dot;
        phi_q_qdot[phi_q_qdot.len() - 1]
    }

    /// Converged position, velocity and acceleration at time t.
    fn solve_kinematics(
        mech: &Mechanism,
        q_guess: &DVector<f64>,
        t: f64,
    ) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
        let pos = solve_position(mech, q_guess, t, 1e-13, 50).unwrap();
        assert!(pos.converged, "position solve failed at t = {}", t);
        let q_dot = solve_velocity(mech, &pos.q, t).unwrap();
        let q_ddot = solve_acceleration(mech, &pos.q, &q_dot, t).unwrap();
        (pos.q, q_dot, q_ddot)
    }

    /// Four-bar (ground O2-O4 along x) whose rocker is built C -> D, so the
    /// rocker's body origin sits at the moving coupler joint C and its
    /// ground pivot D is at body-local (l_rocker, 0) — the same frame layout
    /// as the ParallelogramActuator sample. Driver: crank angle = f(t).
    fn build_fourbar_rocker_origin_at_c(
        lengths: (f64, f64, f64, f64),
        cgs: [(f64, f64, f64, f64); 3],
        f: impl Fn(f64) -> f64 + Send + Sync + 'static,
        f_dot: impl Fn(f64) -> f64 + Send + Sync + 'static,
        f_ddot: impl Fn(f64) -> f64 + Send + Sync + 'static,
    ) -> Mechanism {
        let (l_ground, l_crank, l_coupler, l_rocker) = lengths;
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", l_ground, 0.0)]);
        let mut crank = make_bar("crank", "A", "B", l_crank, cgs[0].0, cgs[0].1);
        crank.cg_local = Vector2::new(cgs[0].2, cgs[0].3);
        let mut coupler = make_bar("coupler", "B", "C", l_coupler, cgs[1].0, cgs[1].1);
        coupler.cg_local = Vector2::new(cgs[1].2, cgs[1].3);
        let mut rocker = make_bar("rocker", "C", "D", l_rocker, cgs[2].0, cgs[2].1);
        rocker.cg_local = Vector2::new(cgs[2].2, cgs[2].3);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
        mech.add_revolute_driver("D1", "ground", "crank", f, f_dot, f_ddot).unwrap();
        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.build().unwrap();
        mech
    }

    /// Exact open-branch pose of the four-bar above at crank angle `theta`
    /// (coupler joint C on the left of the B -> O4 line), via the shared
    /// sample-convention loop closure (rocker origin at C).
    fn fourbar_pose(mech: &Mechanism, lengths: (f64, f64, f64, f64), theta: f64) -> DVector<f64> {
        let (l_ground, l_crank, l_coupler, l_rocker) = lengths;
        try_fourbar_initial_q0(
            mech.state(),
            (0.0, 0.0),
            (l_ground, 0.0),
            l_crank,
            l_coupler,
            l_rocker,
            theta,
            "crank",
            "coupler",
            "rocker",
            true,
        )
        .expect("four-bar cannot close at this crank angle")
    }

    #[test]
    fn quadratic_velocity_forces_hand_computed() {
        // "offset": m = 2, s_cg = (0, 0.8), theta = pi/2, theta_dot = 3:
        //   A(pi/2) * s = (-0.8, 0) -> Q_v = 2 * 9 * (-0.8, 0) = (-14.4, 0), theta row 0.
        // "centered": s_cg = 0 -> Q_v = 0 at any speed.
        // "massless": mass 0 with an offset CG -> skipped, Q_v = 0.
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let mut offset = make_bar("offset", "A", "B", 1.0, 2.0, 0.1);
        offset.cg_local = Vector2::new(0.0, 0.8);
        let mut centered = make_bar("centered", "A", "B", 1.0, 5.0, 0.1);
        centered.cg_local = Vector2::new(0.0, 0.0);
        let mut massless = make_bar("massless", "A", "B", 1.0, 0.0, 0.0);
        massless.cg_local = Vector2::new(0.3, 0.4);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(offset).unwrap();
        mech.add_body(centered).unwrap();
        mech.add_body(massless).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "offset", "A").unwrap();
        mech.add_revolute_joint("J2", "ground", "O", "centered", "A").unwrap();
        mech.add_revolute_joint("J3", "ground", "O", "massless", "A").unwrap();
        mech.build().unwrap();

        let state = mech.state();
        let mut q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        for id in ["offset", "centered", "massless"] {
            state.set_pose(id, &mut q, 0.0, 0.0, PI / 2.0);
            q_dot[state.get_index(id).unwrap().theta_idx()] = 3.0;
        }

        let q_v = assemble_quadratic_velocity_forces(&mech, &q, &q_dot);
        assert_eq!(q_v.len(), state.n_coords());

        let off = state.get_index("offset").unwrap();
        assert!((q_v[off.x_idx()] - (-14.4)).abs() < 1e-12, "x: {}", q_v[off.x_idx()]);
        assert!(q_v[off.y_idx()].abs() < 1e-12, "y: {}", q_v[off.y_idx()]);
        assert_eq!(q_v[off.theta_idx()], 0.0);

        for id in ["centered", "massless"] {
            let idx = state.get_index(id).unwrap();
            for i in [idx.x_idx(), idx.y_idx(), idx.theta_idx()] {
                assert_eq!(q_v[i], 0.0, "{} row {} should be zero", id, i);
            }
        }

        // Quadratic in theta_dot: zero at rest.
        let q_v_rest = assemble_quadratic_velocity_forces(&mech, &q, &DVector::zeros(state.n_coords()));
        assert_eq!(q_v_rest.norm(), 0.0);
    }

    #[test]
    fn constant_ke_parallelogram_inverse_dynamics_torque_equals_statics() {
        // BL-027 reproduction: ParallelogramActuator geometry (ground = coupler
        // = 4, crank = rocker = 2) with a 50 kg rocker mass at body-local
        // (0, 0.8), driven at a constant 1 rev/s. Crank and rocker spin at the
        // constant driver rate and the coupler translates at constant speed,
        // so KE is constant: the inertial part of the driver torque must be
        // zero and tau_ID must equal tau_statics. Missing the velocity-
        // quadratic term produced an error of 2*m*s_y*omega^2 = 3158 N*m.
        let omega = 2.0 * PI;
        let lengths = (4.0, 2.0, 4.0, 2.0);
        let mech = build_fourbar_rocker_origin_at_c(
            lengths,
            [(2.0, 0.1, 1.0, 0.0), (4.0, 0.3, 2.0, 0.0), (50.0, 0.2, 0.0, 0.8)],
            move |t| omega * t,
            move |_t| omega,
            |_t| 0.0,
        );

        let mut ke_ref: Option<f64> = None;
        for deg in [40.0_f64, 100.0, 150.0] {
            let theta = deg.to_radians();
            let t = theta / omega;
            let (q, q_dot, q_ddot) = solve_kinematics(&mech, &fourbar_pose(&mech, lengths, theta), t);

            // Premise check: KE really is constant across samples.
            let ke = compute_kinetic_energy(mech.state(), mech.bodies(), &q, &q_dot);
            let ke0 = *ke_ref.get_or_insert(ke);
            assert!(
                (ke - ke0).abs() <= 1e-9 * ke0,
                "KE not constant at {} deg: {} vs {}",
                deg, ke, ke0
            );

            let tau_id =
                driver_lambda(&solve_inverse_dynamics(&mech, &q, &q_dot, &q_ddot, t).unwrap().lambdas);
            let tau_st =
                driver_lambda(&crate::solver::statics::solve_statics(&mech, &q, t).unwrap().lambdas);
            assert!(
                (tau_id - tau_st).abs() <= 1e-9 * tau_st.abs(),
                "{} deg: tau_ID = {}, tau_statics = {}, diff = {:e} (expected 0, constant KE)",
                deg, tau_id, tau_st, tau_id - tau_st
            );
        }
    }

    #[test]
    fn inverse_dynamics_torque_minus_statics_matches_ke_rate() {
        // General (non-parallelogram) crank-rocker with off-axis CGs on every
        // link and an accelerating driver theta = 0.4 + 3 t + 2.5 t^2.
        // Energy check: (tau_ID - tau_statics) * rate == dKE/dt, with dKE/dt
        // from a central difference in time of compute_kinetic_energy (body
        // velocities, no mass matrix).
        let lengths = (4.0, 1.0, 3.5, 2.5);
        let mech = build_fourbar_rocker_origin_at_c(
            lengths,
            [(2.0, 0.01, 0.5, 0.2), (3.0, 0.05, 1.2, 0.5), (2.0, 0.02, 0.4, -0.6)],
            |t| 0.4 + 3.0 * t + 2.5 * t * t,
            |t| 3.0 + 5.0 * t,
            |_t| 5.0,
        );
        let crank_angle = |t: f64| 0.4 + 3.0 * t + 2.5 * t * t;

        for t in [0.3_f64, 1.1] {
            let (q, q_dot, q_ddot) =
                solve_kinematics(&mech, &fourbar_pose(&mech, lengths, crank_angle(t)), t);

            let tau_id =
                driver_lambda(&solve_inverse_dynamics(&mech, &q, &q_dot, &q_ddot, t).unwrap().lambdas);
            let tau_st =
                driver_lambda(&crate::solver::statics::solve_statics(&mech, &q, t).unwrap().lambdas);
            let inertial_power = (tau_id - tau_st) * driver_rate(&mech, &q, &q_dot, t);

            let h = 1e-4;
            let ke_at = |tk: f64| {
                let (qk, qk_dot, _) =
                    solve_kinematics(&mech, &fourbar_pose(&mech, lengths, crank_angle(tk)), tk);
                compute_kinetic_energy(mech.state(), mech.bodies(), &qk, &qk_dot)
            };
            let dke_dt = (ke_at(t + h) - ke_at(t - h)) / (2.0 * h);

            assert!(
                dke_dt.abs() > 1.0,
                "sample t = {} should have a non-trivial KE rate, got {}",
                t, dke_dt
            );
            assert!(
                (inertial_power - dke_dt).abs() <= 1e-6 * dke_dt.abs(),
                "t = {}: (tau_ID - tau_statics) * rate = {}, dKE/dt = {}, rel err = {:e}",
                t, inertial_power, dke_dt, (inertial_power - dke_dt).abs() / dke_dt.abs()
            );
        }
    }

    #[test]
    fn pivot_reaction_is_centripetal_force_for_crank_pinned_at_origin() {
        // BL-027 Newton check on a joint reaction (the energy tests above only
        // constrain the projection q_dot . Q_v). One bar pinned to ground at
        // its own origin, CG at body-local (0.5, 0), m = 2, constant omega, no
        // gravity. r_ddot = theta_ddot = 0, so M * q_ddot = 0 and Q_v alone
        // supplies the load. Newton on the bar: pivot force on bar =
        // m * a_cg = -m * omega^2 * A(theta) * s_cg. J1 = (ground, bar) with
        // Phi = r_ground + A s_O - r_bar - A_bar s_A, so the bar receives
        // -lambda and force_global = +m * omega^2 * A(theta) * s_cg: magnitude
        // m * omega^2 * 0.5 = 16 N along the pivot -> CG line. The code without
        // Q_v returned ~0 here. The origin is fixed, so Q_v does no work and
        // the driver torque stays 0.
        let (mass, omega, theta) = (2.0, 4.0, 0.7);
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let mut bar = make_bar("bar", "A", "B", 1.0, mass, 0.05);
        bar.cg_local = Vector2::new(0.5, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A").unwrap();
        mech.add_revolute_driver("D1", "ground", "bar", move |t| omega * t, move |_t| omega, |_t| 0.0)
            .unwrap();
        mech.build().unwrap();

        let t = theta / omega;
        let mut q_guess = mech.state().make_q();
        mech.state().set_pose("bar", &mut q_guess, 0.0, 0.0, theta);
        let (q, q_dot, q_ddot) = solve_kinematics(&mech, &q_guess, t);

        let id = solve_inverse_dynamics(&mech, &q, &q_dot, &q_ddot, t).unwrap();
        assert!(id.m_q_ddot.norm() < 1e-12, "premise: M * q_ddot = {}", id.m_q_ddot);
        let reactions = extract_reactions(
            &mech,
            &StaticSolveResult {
                lambdas: id.lambdas,
                q_forces: id.q_forces,
                residual_norm: id.residual_norm,
                is_overconstrained: id.is_overconstrained,
                condition_number: id.condition_number,
            },
        );

        let expected = Vector2::new(theta.cos(), theta.sin()) * (mass * omega * omega * 0.5);
        let pivot = reactions.iter().find(|r| r.joint_id == "J1").unwrap();
        let got = Vector2::new(pivot.force_global[0], pivot.force_global[1]);
        assert!(
            (got - expected).norm() <= 1e-9 * expected.norm(),
            "pivot reaction {:?}, expected centripetal {:?} (|F| = m omega^2 |s_cg| = {})",
            got, expected, expected.norm()
        );

        let driver = reactions.iter().find(|r| r.joint_id == "D1").unwrap();
        assert!(driver.effort.abs() < 1e-9, "driver torque = {}, expected 0", driver.effort);
    }
}
