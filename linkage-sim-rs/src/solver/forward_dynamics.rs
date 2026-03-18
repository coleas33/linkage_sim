//! Forward dynamics integrator for constrained multibody systems.
//!
//! Solves the index-3 DAE:
//!     M * q_ddot + Phi_q^T * lambda = Q
//!     Phi(q, t) = 0
//!
//! Using Baumgarte stabilization to control constraint drift:
//!     Phi_ddot + 2*alpha * Phi_dot + beta^2 * Phi = 0
//!
//! At each time step:
//! 1. Assemble M, Phi_q, Q, gamma
//! 2. Solve augmented system for [q_ddot, lambda]
//! 3. Return dy/dt = [q_dot, q_ddot]
//!
//! Integration uses a fixed-step RK4 integrator.

use nalgebra::{DMatrix, DVector};

use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::forces::assembly::assemble_q;
use crate::forces::gravity::Gravity;
use crate::solver::assembly::{
    assemble_constraints, assemble_gamma, assemble_jacobian, assemble_mass_matrix, assemble_phi_t,
};

/// Configuration for the forward dynamics integrator.
#[derive(Debug, Clone)]
pub struct ForwardDynamicsConfig {
    /// Baumgarte velocity stabilization parameter.
    pub alpha: f64,
    /// Baumgarte position stabilization parameter.
    pub beta: f64,
    /// Relative tolerance (used for adaptive step sizing, reserved for future use).
    pub rtol: f64,
    /// Absolute tolerance (used for adaptive step sizing, reserved for future use).
    pub atol: f64,
    /// Maximum / fixed step size for the RK4 integrator.
    pub max_step: f64,
    /// Apply constraint projection every N steps. 0 = no projection.
    pub project_interval: usize,
    /// Tolerance for constraint projection Newton-Raphson.
    pub project_tol: f64,
    /// Maximum iterations for constraint projection.
    pub max_project_iter: usize,
}

impl Default for ForwardDynamicsConfig {
    fn default() -> Self {
        Self {
            alpha: 5.0,
            beta: 5.0,
            rtol: 1e-8,
            atol: 1e-10,
            max_step: 0.01,
            project_interval: 0,
            project_tol: 1e-10,
            max_project_iter: 10,
        }
    }
}

/// Result of a forward dynamics simulation.
#[derive(Debug, Clone)]
pub struct ForwardDynamicsResult {
    /// Time array (N,).
    pub t: Vec<f64>,
    /// Position history: t[i] -> q[i] (each is n_coords).
    pub q: Vec<DVector<f64>>,
    /// Velocity history: t[i] -> q_dot[i] (each is n_coords).
    pub q_dot: Vec<DVector<f64>>,
    /// ||Phi(q, t)|| at each time step.
    pub constraint_drift: Vec<f64>,
    /// True if integration completed without error.
    pub success: bool,
    /// Status message.
    pub message: String,
}

/// Compute the RHS of the ODE: dy/dt = [q_dot, q_ddot].
///
/// Solves the augmented system:
///   [M,     Phi_q^T] [q_ddot]   [Q        ]
///   [Phi_q,  0     ] [lambda] = [gamma_stab]
///
/// where gamma_stab = gamma - 2*alpha*(Phi_q*q_dot + Phi_t) - beta^2*Phi
fn compute_rhs(
    mech: &Mechanism,
    q: &DVector<f64>,
    qd: &DVector<f64>,
    t: f64,
    gravity: Option<&Gravity>,
    alpha: f64,
    beta: f64,
) -> DVector<f64> {
    let n = mech.state().n_coords();
    let m = mech.n_constraints();

    let m_mat = assemble_mass_matrix(mech, q);
    let phi_q = assemble_jacobian(mech, q, t);
    let phi = assemble_constraints(mech, q, t);
    let phi_t = assemble_phi_t(mech, q, t);
    let gamma = assemble_gamma(mech, q, qd, t);
    let q_forces = assemble_q(mech.state(), gravity, q, qd, t);

    // Baumgarte-stabilized acceleration RHS:
    // gamma_stab = gamma - 2*alpha*(Phi_q*q_dot + Phi_t) - beta^2*Phi
    let phi_dot = &phi_q * qd + &phi_t;
    let gamma_stab = &gamma - 2.0 * alpha * &phi_dot - beta * beta * &phi;

    // Build augmented system
    let dim = n + m;
    let mut a_mat = DMatrix::zeros(dim, dim);
    let mut b_vec = DVector::zeros(dim);

    // Upper-left: M
    for i in 0..n {
        for j in 0..n {
            a_mat[(i, j)] = m_mat[(i, j)];
        }
    }
    // Upper-right: Phi_q^T
    for i in 0..n {
        for j in 0..m {
            a_mat[(i, n + j)] = phi_q[(j, i)];
        }
    }
    // Lower-left: Phi_q
    for i in 0..m {
        for j in 0..n {
            a_mat[(n + i, j)] = phi_q[(i, j)];
        }
    }
    // Lower-right: zero (already initialized)

    // RHS: [Q; gamma_stab]
    for i in 0..n {
        b_vec[i] = q_forces[i];
    }
    for i in 0..m {
        b_vec[n + i] = gamma_stab[i];
    }

    // Solve the augmented system
    let x = solve_augmented(&a_mat, &b_vec);

    // Extract q_ddot from solution
    let q_ddot = x.rows(0, n).clone_owned();

    // Pack dy/dt = [q_dot, q_ddot]
    let mut dy = DVector::zeros(2 * n);
    for i in 0..n {
        dy[i] = qd[i];
        dy[n + i] = q_ddot[i];
    }
    dy
}

/// Solve a linear system A*x = b, falling back to SVD if LU fails.
fn solve_augmented(a: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    // Try LU decomposition first (fast path)
    if let Some(lu) = a.clone().lu().solve(b) {
        return lu;
    }
    // Fallback: SVD least-squares
    let svd = a.clone().svd(true, true);
    svd.solve(b, 1e-14).unwrap_or_else(|_| DVector::zeros(b.len()))
}

/// Newton-Raphson constraint projection: project q back onto constraint manifold.
fn project_constraints(
    mech: &Mechanism,
    q: &DVector<f64>,
    t: f64,
    tol: f64,
    max_iter: usize,
) -> DVector<f64> {
    let mut q_proj = q.clone();
    for _ in 0..max_iter {
        let phi = assemble_constraints(mech, &q_proj, t);
        if phi.norm() < tol {
            break;
        }
        let phi_q = assemble_jacobian(mech, &q_proj, t);
        let neg_phi = -phi;
        let svd = phi_q.svd(true, true);
        match svd.solve(&neg_phi, 1e-14) {
            Ok(dq) => q_proj += dq,
            Err(_) => break,
        }
    }
    q_proj
}

/// Fixed-step RK4 integrator.
///
/// Integrates y' = f(t, y) from t_start to t_end with step size h.
/// Records solution at t_eval times (linearly interpolated from steps).
fn rk4_integrate<F>(
    f: &F,
    y0: &DVector<f64>,
    t_start: f64,
    t_end: f64,
    h: f64,
    t_eval: &[f64],
) -> (Vec<f64>, Vec<DVector<f64>>)
where
    F: Fn(f64, &DVector<f64>) -> DVector<f64>,
{
    // Pre-compute number of steps
    let n_steps = ((t_end - t_start) / h).ceil() as usize;
    let actual_h = (t_end - t_start) / n_steps as f64;

    // If t_eval is empty, store at every step
    let store_all = t_eval.is_empty();

    let mut t_out: Vec<f64> = Vec::new();
    let mut y_out: Vec<DVector<f64>> = Vec::new();

    let mut t = t_start;
    let mut y = y0.clone();
    let mut eval_idx = 0;

    // Store initial point if needed
    if store_all {
        t_out.push(t);
        y_out.push(y.clone());
    } else if eval_idx < t_eval.len() && (t_eval[eval_idx] - t).abs() < 1e-14 {
        t_out.push(t);
        y_out.push(y.clone());
        eval_idx += 1;
    }

    for _step in 0..n_steps {
        let t_next = t + actual_h;

        // RK4 stages
        let k1 = f(t, &y);
        let k2 = f(t + actual_h * 0.5, &(&y + &k1 * (actual_h * 0.5)));
        let k3 = f(t + actual_h * 0.5, &(&y + &k2 * (actual_h * 0.5)));
        let k4 = f(t_next, &(&y + &k3 * actual_h));

        let y_next = &y + (actual_h / 6.0) * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4);

        // Check if any t_eval points fall in [t, t_next]
        if store_all {
            t_out.push(t_next);
            y_out.push(y_next.clone());
        } else {
            while eval_idx < t_eval.len() && t_eval[eval_idx] <= t_next + 1e-14 {
                let te = t_eval[eval_idx];
                if (te - t_next).abs() < 1e-14 {
                    // At step boundary -- use exact value
                    t_out.push(te);
                    y_out.push(y_next.clone());
                } else if te >= t && te < t_next {
                    // Linear interpolation between y and y_next
                    let frac = (te - t) / actual_h;
                    let y_interp = &y * (1.0 - frac) + &y_next * frac;
                    t_out.push(te);
                    y_out.push(y_interp);
                }
                eval_idx += 1;
            }
        }

        t = t_next;
        y = y_next;
    }

    (t_out, y_out)
}

/// Run a forward dynamics simulation.
///
/// # Arguments
///
/// * `mech` - Built mechanism (should have DOF > 0 for free motion, i.e., no
///   driver constraint or fewer constraints than coordinates).
/// * `q0` - Initial position satisfying Phi(q0, t0) ~ 0.
/// * `q_dot0` - Initial velocity satisfying Phi_q * q_dot0 ~ -Phi_t.
/// * `t_span` - (t_start, t_end) time interval.
/// * `gravity` - Optional gravity force element.
/// * `config` - Integration parameters. Uses defaults if None.
/// * `t_eval` - Optional array of times at which to store solution.
///
/// # Returns
///
/// `ForwardDynamicsResult` with time histories.
pub fn simulate(
    mech: &Mechanism,
    q0: &DVector<f64>,
    q_dot0: &DVector<f64>,
    t_span: (f64, f64),
    gravity: Option<&Gravity>,
    config: Option<&ForwardDynamicsConfig>,
    t_eval: Option<&[f64]>,
) -> Result<ForwardDynamicsResult, LinkageError> {
    if !mech.is_built() {
        return Err(LinkageError::MechanismNotBuilt);
    }

    let default_config = ForwardDynamicsConfig::default();
    let cfg = config.unwrap_or(&default_config);

    let n = mech.state().n_coords();
    let alpha = cfg.alpha;
    let beta = cfg.beta;

    // Pack initial state: y = [q; q_dot]
    let mut y0 = DVector::zeros(2 * n);
    for i in 0..n {
        y0[i] = q0[i];
        y0[n + i] = q_dot0[i];
    }

    // Define the RHS function
    let rhs = |t: f64, y: &DVector<f64>| -> DVector<f64> {
        let q = y.rows(0, n).clone_owned();
        let qd = y.rows(n, n).clone_owned();
        compute_rhs(mech, &q, &qd, t, gravity, alpha, beta)
    };

    // Integrate using RK4
    let empty_t_eval: Vec<f64> = Vec::new();
    let eval_times = t_eval.unwrap_or(&empty_t_eval);
    let (t_out, y_out) = rk4_integrate(&rhs, &y0, t_span.0, t_span.1, cfg.max_step, eval_times);

    // Unpack results
    let mut q_out: Vec<DVector<f64>> = Vec::with_capacity(t_out.len());
    let mut qd_out: Vec<DVector<f64>> = Vec::with_capacity(t_out.len());
    let mut drift: Vec<f64> = Vec::with_capacity(t_out.len());

    for (i, y) in y_out.iter().enumerate() {
        let mut q_i = y.rows(0, n).clone_owned();
        let mut qd_i = y.rows(n, n).clone_owned();

        // Apply constraint projection if configured
        if cfg.project_interval > 0 && i > 0 && i % cfg.project_interval == 0 {
            q_i = project_constraints(
                mech,
                &q_i,
                t_out[i],
                cfg.project_tol,
                cfg.max_project_iter,
            );

            // Velocity projection: remove the constraint-violating component
            // of velocity while preserving as much of the original velocity
            // as possible. Minimum-correction approach:
            //   q_dot_new = q_dot - Phi_q^+ * (Phi_q * q_dot + Phi_t)
            let phi_q = assemble_jacobian(mech, &q_i, t_out[i]);
            let phi_t = assemble_phi_t(mech, &q_i, t_out[i]);
            let violation = &phi_q * &qd_i + &phi_t;
            let svd = phi_q.svd(true, true);
            if let Ok(correction) = svd.solve(&violation, 1e-14) {
                qd_i -= correction;
            }
        }

        let phi = assemble_constraints(mech, &q_i, t_out[i]);
        drift.push(phi.norm());

        q_out.push(q_i);
        qd_out.push(qd_i);
    }

    Ok(ForwardDynamicsResult {
        t: t_out,
        q: q_out,
        q_dot: qd_out,
        constraint_drift: drift,
        success: true,
        message: "RK4 integration completed.".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground, Body};
    use crate::core::state::GROUND_ID;
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector2;
    use std::collections::HashMap;
    use std::f64::consts::PI;

    /// Build a simple pendulum: single bar pinned to ground.
    /// CG at tip (point mass), L=1, m=1.
    fn build_pendulum() -> (Mechanism, Gravity) {
        let ground = make_ground(&[("O", 0.0, 0.0)]);

        // Bar with mass concentrated at tip: cg_local = (1, 0), Izz_cg = 0
        let mut bar = Body::new("bar");
        bar.add_attachment_point("A", 0.0, 0.0).unwrap();
        bar.mass = 1.0;
        bar.cg_local = Vector2::new(1.0, 0.0);
        bar.izz_cg = 0.0;

        let mut bodies = HashMap::new();
        bodies.insert("ground".to_string(), ground.clone());
        bodies.insert("bar".to_string(), bar.clone());

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.build().unwrap();

        let gravity = Gravity::new(Vector2::new(0.0, -9.81), &bodies);
        (mech, gravity)
    }

    fn pendulum_initial_state(
        mech: &Mechanism,
        theta0: f64,
    ) -> (DVector<f64>, DVector<f64>) {
        let state = mech.state();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, theta0);
        let q_dot = DVector::zeros(state.n_coords());
        (q, q_dot)
    }

    #[test]
    fn mass_matrix_diagonal_for_cg_at_origin() {
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        // Bar with CG at origin (attachment points don't affect CG)
        let mut bar = Body::new("bar");
        bar.add_attachment_point("A", 0.0, 0.0).unwrap();
        bar.mass = 3.0;
        bar.cg_local = Vector2::new(0.0, 0.0);
        bar.izz_cg = 0.5;

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.build().unwrap();

        let q = DVector::from_column_slice(&[0.0, 0.0, 0.0]);
        let m_mat = assemble_mass_matrix(&mech, &q);

        assert_abs_diff_eq!(m_mat[(0, 0)], 3.0, epsilon = 1e-14);
        assert_abs_diff_eq!(m_mat[(1, 1)], 3.0, epsilon = 1e-14);
        assert_abs_diff_eq!(m_mat[(2, 2)], 0.5, epsilon = 1e-14);
        // Off-diagonal should be zero
        assert_abs_diff_eq!(m_mat[(0, 2)], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(m_mat[(1, 2)], 0.0, epsilon = 1e-14);
    }

    #[test]
    fn mass_matrix_with_cg_offset() {
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 2.0, 0.01);
        // CG is at (0.5, 0) from make_bar

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.build().unwrap();

        let q = DVector::from_column_slice(&[0.0, 0.0, 0.0]);
        let m_mat = assemble_mass_matrix(&mech, &q);

        // M_theta_theta = Izz_cg + m*|s_cg|^2 = 0.01 + 2.0 * 0.25 = 0.51
        assert_abs_diff_eq!(m_mat[(2, 2)], 0.51, epsilon = 1e-14);

        // At theta=0: B(0) * (0.5, 0) = (0, 0.5)
        // Off-diag: m * Bs = 2.0 * (0, 0.5) = (0, 1.0)
        assert_abs_diff_eq!(m_mat[(0, 2)], 0.0, epsilon = 1e-14); // m * Bs_x = 0
        assert_abs_diff_eq!(m_mat[(1, 2)], 1.0, epsilon = 1e-14); // m * Bs_y = 1.0
    }

    #[test]
    fn pendulum_simulation_runs() {
        let (mech, gravity) = build_pendulum();
        let theta0 = -PI / 2.0 + 0.1;
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.005,
            ..Default::default()
        };

        let result = simulate(&mech, &q0, &qd0, (0.0, 1.0), Some(&gravity), Some(&config), None).unwrap();
        assert!(result.success);
        assert!(!result.t.is_empty());
    }

    #[test]
    fn pendulum_constraint_drift_bounded() {
        let (mech, gravity) = build_pendulum();
        let theta0 = -PI / 2.0 + 0.1;
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.002,
            ..Default::default()
        };

        let result = simulate(&mech, &q0, &qd0, (0.0, 2.0), Some(&gravity), Some(&config), None).unwrap();
        assert!(result.success);

        let max_drift = result
            .constraint_drift
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        assert!(
            max_drift < 1e-4,
            "Max constraint drift = {:e}, expected < 1e-4",
            max_drift
        );
    }

    #[test]
    fn pendulum_energy_approximately_conserved() {
        let (mech, gravity) = build_pendulum();
        let theta0 = -PI / 2.0 + 0.2;
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.001,
            rtol: 1e-10,
            atol: 1e-12,
            ..Default::default()
        };

        let t_eval: Vec<f64> = (0..=300).map(|i| i as f64 * 0.01).collect();
        let result = simulate(
            &mech,
            &q0,
            &qd0,
            (0.0, 3.0),
            Some(&gravity),
            Some(&config),
            Some(&t_eval),
        ).unwrap();
        assert!(result.success);

        // Compute energy at each step
        let state = mech.state();
        let g_mag = 9.81;
        let mut energies: Vec<f64> = Vec::new();

        for i in 0..result.t.len() {
            let q = &result.q[i];
            let qd = &result.q_dot[i];

            // KE = 0.5 * q_dot^T * M * q_dot
            let m_mat = assemble_mass_matrix(&mech, q);
            let ke = 0.5 * qd.dot(&(&m_mat * qd));

            // PE = sum m_i * g * y_cg_i
            let mut pe = 0.0;
            for (body_id, body) in mech.bodies() {
                if body_id == GROUND_ID || body.mass <= 0.0 {
                    continue;
                }
                let r_cg = state.body_point_global(body_id, &body.cg_local, q);
                pe += body.mass * g_mag * r_cg.y;
            }

            energies.push(ke + pe);
        }

        let e0 = energies[0];
        let max_deviation = energies.iter().fold(0.0_f64, |acc, &e| {
            acc.max((e - e0).abs())
        });

        // Energy should stay within 5% of initial
        assert!(
            max_deviation < e0.abs() * 0.05 + 1e-6,
            "Max energy deviation = {:e}, initial = {:e}, ratio = {:e}",
            max_deviation,
            e0,
            max_deviation / e0.abs()
        );
    }

    #[test]
    fn pendulum_period_approximately_correct() {
        let (mech, gravity) = build_pendulum();
        let theta0 = -PI / 2.0 + 0.087; // ~5 degrees from hanging
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let l = 1.0;
        let t_analytical = 2.0 * PI * (l / 9.81_f64).sqrt();
        let t_end = 3.0 * t_analytical;

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.001,
            ..Default::default()
        };

        let t_eval: Vec<f64> = (0..500)
            .map(|i| i as f64 * t_end / 499.0)
            .collect();
        let result = simulate(
            &mech,
            &q0,
            &qd0,
            (0.0, t_end),
            Some(&gravity),
            Some(&config),
            Some(&t_eval),
        ).unwrap();
        assert!(result.success);

        // Extract theta(t) and find zero crossings to measure period
        let bar_idx = mech.state().get_index("bar").unwrap();
        let equilibrium = -PI / 2.0;

        let theta_offset: Vec<f64> = result
            .q
            .iter()
            .map(|q| q[bar_idx.theta_idx()] - equilibrium)
            .collect();

        // Find positive-going zero crossings
        let mut crossings: Vec<f64> = Vec::new();
        for i in 1..theta_offset.len() {
            if theta_offset[i - 1] < 0.0 && theta_offset[i] >= 0.0 {
                let frac = -theta_offset[i - 1] / (theta_offset[i] - theta_offset[i - 1]);
                let t_cross = result.t[i - 1] + frac * (result.t[i] - result.t[i - 1]);
                crossings.push(t_cross);
            }
        }

        assert!(
            crossings.len() >= 2,
            "Found only {} crossings, need >= 2",
            crossings.len()
        );
        let measured_period = crossings[1] - crossings[0];

        // Should match within 5% (small angle approximation + numerical)
        let rel_error = (measured_period - t_analytical).abs() / t_analytical;
        assert!(
            rel_error < 0.05,
            "Period mismatch: measured={}, analytical={}, rel_error={}",
            measured_period,
            t_analytical,
            rel_error
        );
    }

    #[test]
    fn pendulum_with_constraint_projection() {
        let (mech, gravity) = build_pendulum();
        let theta0 = -PI / 2.0 + 0.2;
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 5.0,
            beta: 5.0,
            max_step: 0.005,
            project_interval: 10,
            project_tol: 1e-12,
            max_project_iter: 10,
            ..Default::default()
        };

        let result = simulate(&mech, &q0, &qd0, (0.0, 2.0), Some(&gravity), Some(&config), None).unwrap();
        assert!(result.success);

        // With projection, constraint drift should remain small
        let max_drift = result
            .constraint_drift
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        assert!(
            max_drift < 1e-4,
            "Max constraint drift with projection = {:e}",
            max_drift
        );
    }

    #[test]
    fn t_eval_stores_correct_times() {
        let (mech, gravity) = build_pendulum();
        let theta0 = -PI / 2.0 + 0.1;
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.01,
            ..Default::default()
        };

        let t_eval: Vec<f64> = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let result = simulate(
            &mech,
            &q0,
            &qd0,
            (0.0, 0.5),
            Some(&gravity),
            Some(&config),
            Some(&t_eval),
        ).unwrap();
        assert!(result.success);
        assert_eq!(result.t.len(), t_eval.len());

        for (actual, expected) in result.t.iter().zip(t_eval.iter()) {
            assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn rk4_integrates_simple_ode() {
        // Test RK4 on a known ODE: y' = -y, y(0) = 1 => y(t) = e^(-t)
        let f = |_t: f64, y: &DVector<f64>| -> DVector<f64> { -y.clone() };
        let y0 = DVector::from_column_slice(&[1.0]);

        let t_eval: Vec<f64> = vec![0.0, 0.5, 1.0, 2.0];
        let (t_out, y_out) = rk4_integrate(&f, &y0, 0.0, 2.0, 0.01, &t_eval);

        for (i, &t) in t_out.iter().enumerate() {
            let expected = (-t).exp();
            assert_abs_diff_eq!(y_out[i][0], expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn solve_augmented_system() {
        // Simple 2x2 system: [[2, 1], [1, 3]] * x = [5, 6] => x = [1.8, 1.4]
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0]);
        let b = DVector::from_column_slice(&[5.0, 6.0]);
        let x = solve_augmented(&a, &b);
        assert_abs_diff_eq!(x[0], 1.8, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 1.4, epsilon = 1e-12);
    }
}
