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
use crate::solver::assembly::{
    assemble_constraints, assemble_gamma, assemble_jacobian, assemble_mass_matrix, assemble_phi_t,
};
use crate::solver::events::{check_events, DynamicsEvent, EventOccurrence};

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
    /// Events detected during integration (empty when no events are monitored).
    pub detected_events: Vec<EventOccurrence>,
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
    alpha: f64,
    beta: f64,
    compiled_modulations: &[Box<dyn Fn(f64) -> f64>],
) -> Option<DVector<f64>> {
    let n = mech.state().n_coords();
    let m = mech.n_constraints();

    let m_mat = assemble_mass_matrix(mech, q);
    let phi_q = assemble_jacobian(mech, q, t);
    let phi = assemble_constraints(mech, q, t);
    let phi_t = assemble_phi_t(mech, q, t);
    let gamma = assemble_gamma(mech, q, qd, t);
    let q_forces = mech.assemble_forces_compiled(q, qd, t, compiled_modulations);

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
    let x = solve_augmented(&a_mat, &b_vec)?;

    // Extract q_ddot from solution
    let q_ddot = x.rows(0, n).clone_owned();

    // Pack dy/dt = [q_dot, q_ddot]
    let mut dy = DVector::zeros(2 * n);
    for i in 0..n {
        dy[i] = qd[i];
        dy[n + i] = q_ddot[i];
    }
    Some(dy)
}

/// Solve a linear system A*x = b, falling back to SVD if LU fails.
/// Returns `None` if both LU and SVD fail (singular augmented system).
fn solve_augmented(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    // Try LU decomposition first (fast path)
    if let Some(lu) = a.clone().lu().solve(b) {
        return Some(lu);
    }
    // Fallback: SVD least-squares
    let svd = a.clone().svd(true, true);
    match svd.solve(b, 1e-14) {
        Ok(x) => Some(x),
        Err(_) => {
            log::warn!("solve_augmented: both LU and SVD failed — singular augmented system");
            None
        }
    }
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
/// Returns `(times, states, solve_failed)` where `solve_failed` is true if
/// the RHS function returned `None` (singular system) at any stage.
fn rk4_integrate<F>(
    f: &F,
    y0: &DVector<f64>,
    t_start: f64,
    t_end: f64,
    h: f64,
    t_eval: &[f64],
) -> (Vec<f64>, Vec<DVector<f64>>, bool)
where
    F: Fn(f64, &DVector<f64>) -> Option<DVector<f64>>,
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

        // RK4 stages — abort integration if any stage fails
        let Some(k1) = f(t, &y) else {
            return (t_out, y_out, true);
        };
        let Some(k2) = f(t + actual_h * 0.5, &(&y + &k1 * (actual_h * 0.5))) else {
            return (t_out, y_out, true);
        };
        let Some(k3) = f(t + actual_h * 0.5, &(&y + &k2 * (actual_h * 0.5))) else {
            return (t_out, y_out, true);
        };
        let Some(k4) = f(t_next, &(&y + &k3 * actual_h)) else {
            return (t_out, y_out, true);
        };

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

    (t_out, y_out, false)
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

    // Pre-compile time modulations once (avoids re-parsing expression strings
    // on every RHS evaluation inside the integration loop).
    let compiled_modulations = mech.compile_force_modulations();

    // Pack initial state: y = [q; q_dot]
    let mut y0 = DVector::zeros(2 * n);
    for i in 0..n {
        y0[i] = q0[i];
        y0[n + i] = q_dot0[i];
    }

    // Define the RHS function
    let rhs = |t: f64, y: &DVector<f64>| -> Option<DVector<f64>> {
        let q = y.rows(0, n).clone_owned();
        let qd = y.rows(n, n).clone_owned();
        compute_rhs(mech, &q, &qd, t, alpha, beta, &compiled_modulations)
    };

    // Integrate using RK4
    let empty_t_eval: Vec<f64> = Vec::new();
    let eval_times = t_eval.unwrap_or(&empty_t_eval);
    let (t_out, y_out, solve_failed) =
        rk4_integrate(&rhs, &y0, t_span.0, t_span.1, cfg.max_step, eval_times);

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

    let (success, message) = if solve_failed {
        (false, "RK4 integration aborted: singular augmented system (both LU and SVD failed).".to_string())
    } else {
        (true, "RK4 integration completed.".to_string())
    };

    Ok(ForwardDynamicsResult {
        t: t_out,
        q: q_out,
        q_dot: qd_out,
        constraint_drift: drift,
        detected_events: Vec::new(),
        success,
        message,
    })
}

/// Run a forward dynamics simulation with event detection.
///
/// This is identical to [`simulate`] but additionally monitors a list
/// of [`DynamicsEvent`]s.  After each RK4 step, event functions are
/// evaluated and zero crossings are recorded.  If a terminal event
/// fires, integration stops early.
///
/// # Arguments
///
/// * `mech` - Built mechanism.
/// * `q0` - Initial positions satisfying Phi(q0, t0) ~ 0.
/// * `q_dot0` - Initial velocities satisfying Phi_q * q_dot0 ~ -Phi_t.
/// * `t_span` - (t_start, t_end) time interval.
/// * `config` - Integration parameters. Uses defaults if None.
/// * `t_eval` - Optional array of times at which to store solution.
/// * `events` - Slice of events to monitor.
///
/// # Returns
///
/// `ForwardDynamicsResult` with time histories and `detected_events`.
pub fn simulate_with_events(
    mech: &Mechanism,
    q0: &DVector<f64>,
    q_dot0: &DVector<f64>,
    t_span: (f64, f64),
    config: Option<&ForwardDynamicsConfig>,
    t_eval: Option<&[f64]>,
    events: &[DynamicsEvent],
) -> Result<ForwardDynamicsResult, LinkageError> {
    if !mech.is_built() {
        return Err(LinkageError::MechanismNotBuilt);
    }
    if events.is_empty() {
        return simulate(mech, q0, q_dot0, t_span, config, t_eval);
    }

    let default_config = ForwardDynamicsConfig::default();
    let cfg = config.unwrap_or(&default_config);

    let n = mech.state().n_coords();
    let alpha = cfg.alpha;
    let beta = cfg.beta;

    // Pre-compile time modulations once (avoids re-parsing expression strings
    // on every RHS evaluation inside the integration loop).
    let compiled_modulations = mech.compile_force_modulations();

    // Pack initial state: y = [q; q_dot]
    let mut y0 = DVector::zeros(2 * n);
    for i in 0..n {
        y0[i] = q0[i];
        y0[n + i] = q_dot0[i];
    }

    // RHS closure
    let rhs = |t: f64, y: &DVector<f64>| -> Option<DVector<f64>> {
        let q = y.rows(0, n).clone_owned();
        let qd = y.rows(n, n).clone_owned();
        compute_rhs(mech, &q, &qd, t, alpha, beta, &compiled_modulations)
    };

    // Manual RK4 loop with event checking
    let n_steps = ((t_span.1 - t_span.0) / cfg.max_step).ceil() as usize;
    let actual_h = (t_span.1 - t_span.0) / n_steps as f64;

    let store_all = t_eval.is_none();
    let empty_eval: Vec<f64> = Vec::new();
    let eval_times = t_eval.unwrap_or(&empty_eval);
    let mut eval_idx = 0;

    let mut t_out: Vec<f64> = Vec::new();
    let mut y_out: Vec<DVector<f64>> = Vec::new();
    let mut all_events: Vec<EventOccurrence> = Vec::new();

    let mut t = t_span.0;
    let mut y = y0;

    // Store initial point
    if store_all {
        t_out.push(t);
        y_out.push(y.clone());
    } else if eval_idx < eval_times.len() && (eval_times[eval_idx] - t).abs() < 1e-14 {
        t_out.push(t);
        y_out.push(y.clone());
        eval_idx += 1;
    }

    let mut terminated = false;
    let mut solve_failed = false;

    for _step in 0..n_steps {
        let t_next = t + actual_h;

        // RK4 stages — abort integration if any stage fails
        let Some(k1) = rhs(t, &y) else { solve_failed = true; break; };
        let Some(k2) = rhs(t + actual_h * 0.5, &(&y + &k1 * (actual_h * 0.5))) else { solve_failed = true; break; };
        let Some(k3) = rhs(t + actual_h * 0.5, &(&y + &k2 * (actual_h * 0.5))) else { solve_failed = true; break; };
        let Some(k4) = rhs(t_next, &(&y + &k3 * actual_h)) else { solve_failed = true; break; };
        let y_next = &y + (actual_h / 6.0) * (&k1 + 2.0 * &k2 + 2.0 * &k3 + &k4);

        // Check events between this step
        let occs = check_events(events, mech.state(), t, &y, t_next, &y_next);

        let has_terminal = occs.iter().any(|o| events[o.event_index].terminal);
        all_events.extend(occs);

        // Store output points
        if store_all {
            t_out.push(t_next);
            y_out.push(y_next.clone());
        } else {
            while eval_idx < eval_times.len()
                && eval_times[eval_idx] <= t_next + 1e-14
            {
                let te = eval_times[eval_idx];
                if (te - t_next).abs() < 1e-14 {
                    t_out.push(te);
                    y_out.push(y_next.clone());
                } else if te >= t && te < t_next {
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

        if has_terminal {
            terminated = true;
            break;
        }
    }

    // Unpack results
    let mut q_out: Vec<DVector<f64>> = Vec::with_capacity(t_out.len());
    let mut qd_out: Vec<DVector<f64>> = Vec::with_capacity(t_out.len());
    let mut drift: Vec<f64> = Vec::with_capacity(t_out.len());

    for (i, y_i) in y_out.iter().enumerate() {
        let mut q_i = y_i.rows(0, n).clone_owned();
        let mut qd_i = y_i.rows(n, n).clone_owned();

        // Apply constraint projection if configured
        if cfg.project_interval > 0 && i > 0 && i % cfg.project_interval == 0 {
            q_i = project_constraints(
                mech,
                &q_i,
                t_out[i],
                cfg.project_tol,
                cfg.max_project_iter,
            );
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

    let (success, message) = if solve_failed {
        (false, "RK4 integration aborted: singular augmented system (both LU and SVD failed).".to_string())
    } else if terminated {
        (true, "RK4 integration terminated by event.".to_string())
    } else {
        (true, "RK4 integration completed.".to_string())
    };

    Ok(ForwardDynamicsResult {
        t: t_out,
        q: q_out,
        q_dot: qd_out,
        constraint_drift: drift,
        detected_events: all_events,
        success,
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground, Body};
    use crate::core::state::GROUND_ID;
    use crate::forces::elements::{
        ForceElement, GravityElement, JointLimitElement, LinearSpringElement, RotaryDamperElement,
        TorsionSpringElement,
    };
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector2;
    use std::f64::consts::PI;

    /// Build a simple pendulum: single bar pinned to ground.
    /// CG at tip (point mass), L=1, m=1.
    fn build_pendulum() -> Mechanism {
        let ground = make_ground(&[("O", 0.0, 0.0)]);

        // Bar with mass concentrated at tip: cg_local = (1, 0), Izz_cg = 0
        let mut bar = Body::new("bar");
        bar.add_attachment_point("A", 0.0, 0.0).unwrap();
        bar.mass = 1.0;
        bar.cg_local = Vector2::new(1.0, 0.0);
        bar.izz_cg = 0.0;

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.build().unwrap();
        mech
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
        let mech = build_pendulum();
        let theta0 = -PI / 2.0 + 0.1;
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.005,
            ..Default::default()
        };

        let result = simulate(&mech, &q0, &qd0, (0.0, 1.0), Some(&config), None).unwrap();
        assert!(result.success);
        assert!(!result.t.is_empty());
    }

    #[test]
    fn pendulum_constraint_drift_bounded() {
        let mech = build_pendulum();
        let theta0 = -PI / 2.0 + 0.1;
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.002,
            ..Default::default()
        };

        let result = simulate(&mech, &q0, &qd0, (0.0, 2.0), Some(&config), None).unwrap();
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
        let mech = build_pendulum();
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
        let mech = build_pendulum();
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
        let mech = build_pendulum();
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

        let result = simulate(&mech, &q0, &qd0, (0.0, 2.0), Some(&config), None).unwrap();
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
        let mech = build_pendulum();
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
        let f = |_t: f64, y: &DVector<f64>| -> Option<DVector<f64>> { Some(-y.clone()) };
        let y0 = DVector::from_column_slice(&[1.0]);

        let t_eval: Vec<f64> = vec![0.0, 0.5, 1.0, 2.0];
        let (t_out, y_out, failed) = rk4_integrate(&f, &y0, 0.0, 2.0, 0.01, &t_eval);
        assert!(!failed);

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
        let x = solve_augmented(&a, &b).expect("solve should succeed");
        assert_abs_diff_eq!(x[0], 1.8, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 1.4, epsilon = 1e-12);
    }

    // ── Force-element dynamic integration tests ──────────────────────────────

    /// Build a pendulum with no gravity, just a torsion spring at the pivot.
    /// Point mass m at distance L from pivot: I = m*L^2.
    /// With a torsion spring of stiffness k and no gravity, the system is a
    /// simple harmonic oscillator: theta'' + (k/I)*theta = 0
    /// Period T = 2*pi*sqrt(I/k).
    fn build_torsion_spring_pendulum(k: f64, free_angle: f64) -> Mechanism {
        let ground = make_ground(&[("O", 0.0, 0.0)]);

        let mut bar = Body::new("bar");
        bar.add_attachment_point("A", 0.0, 0.0).unwrap();
        bar.mass = 1.0;
        bar.cg_local = Vector2::new(1.0, 0.0);
        bar.izz_cg = 0.0; // point mass at L=1

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.add_force(ForceElement::TorsionSpring(TorsionSpringElement {
            body_i: "ground".into(),
            body_j: "bar".into(),
            stiffness: k,
            free_angle,
        }));
        mech.build().unwrap();
        mech
    }

    #[test]
    fn torsion_spring_pendulum_period() {
        // Point mass pendulum with torsion spring at pivot (no gravity).
        // Simple harmonic oscillator: theta'' + (k/I)*theta = 0
        // m=1, L=1 => I = m*L^2 = 1, k=10
        // T = 2*pi*sqrt(I/k) = 2*pi*sqrt(0.1) ~ 1.987 s
        let k = 10.0;
        let free_angle = -PI / 2.0; // equilibrium at hanging
        let mech = build_torsion_spring_pendulum(k, free_angle);

        // Initial condition: 5-degree offset from equilibrium
        let theta0 = free_angle + 0.087;
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.001,
            ..Default::default()
        };

        let i_total = 1.0; // m * L^2
        let t_analytical = 2.0 * PI * (i_total / k).sqrt();
        let t_end = 3.0 * t_analytical;

        let t_eval: Vec<f64> = (0..500).map(|i| i as f64 * t_end / 499.0).collect();
        let result = simulate(
            &mech,
            &q0,
            &qd0,
            (0.0, t_end),
            Some(&config),
            Some(&t_eval),
        )
        .unwrap();
        assert!(result.success);

        // Find period from positive-going zero crossings of theta offset
        let bar_idx = mech.state().get_index("bar").unwrap();
        let theta_offset: Vec<f64> = result
            .q
            .iter()
            .map(|q| q[bar_idx.theta_idx()] - free_angle)
            .collect();

        let mut crossings: Vec<f64> = Vec::new();
        for i in 1..theta_offset.len() {
            if theta_offset[i - 1] < 0.0 && theta_offset[i] >= 0.0 {
                let frac = -theta_offset[i - 1] / (theta_offset[i] - theta_offset[i - 1]);
                crossings.push(result.t[i - 1] + frac * (result.t[i] - result.t[i - 1]));
            }
        }

        assert!(
            crossings.len() >= 2,
            "Found only {} crossings, need >= 2",
            crossings.len()
        );
        let measured_period = crossings[1] - crossings[0];
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
    fn damped_pendulum_energy_decreases() {
        // Pendulum with gravity + rotary damper at the pivot.
        // With dissipation, total energy (KE + gravitational PE) must
        // decrease monotonically (within numerical tolerance).
        let ground = make_ground(&[("O", 0.0, 0.0)]);

        let mut bar = Body::new("bar");
        bar.add_attachment_point("A", 0.0, 0.0).unwrap();
        bar.mass = 1.0;
        bar.cg_local = Vector2::new(1.0, 0.0);
        bar.izz_cg = 0.0;

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.add_force(ForceElement::RotaryDamper(RotaryDamperElement {
            body_i: "ground".into(),
            body_j: "bar".into(),
            damping: 0.5,
        }));
        mech.build().unwrap();

        // Start from a 30-degree offset from hanging
        let theta0 = -PI / 2.0 + 0.52;
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.001,
            ..Default::default()
        };

        let t_eval: Vec<f64> = (0..=400).map(|i| i as f64 * 0.01).collect();
        let result = simulate(
            &mech,
            &q0,
            &qd0,
            (0.0, 4.0),
            Some(&config),
            Some(&t_eval),
        )
        .unwrap();
        assert!(result.success);

        // Compute total energy at each step
        let state = mech.state();
        let g_mag = 9.81;
        let mut energies: Vec<f64> = Vec::new();

        for i in 0..result.t.len() {
            let q = &result.q[i];
            let qd = &result.q_dot[i];

            let m_mat = assemble_mass_matrix(&mech, q);
            let ke = 0.5 * qd.dot(&(&m_mat * qd));

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

        // Energy at end should be less than energy at start
        let e_start = energies[0];
        let e_end = *energies.last().unwrap();
        assert!(
            e_end < e_start - 0.01,
            "Damped pendulum energy did not decrease: e_start={}, e_end={}",
            e_start,
            e_end
        );

        // Energy should not increase at any step (allow small tolerance for
        // numerical integration artifacts)
        let tolerance = e_start.abs() * 1e-3 + 1e-6;
        for i in 1..energies.len() {
            assert!(
                energies[i] <= energies[i - 1] + tolerance,
                "Energy increased at step {}: E[{}]={}, E[{}]={}, diff={}",
                i,
                i - 1,
                energies[i - 1],
                i,
                energies[i],
                energies[i] - energies[i - 1],
            );
        }
    }

    #[test]
    fn spring_mass_energy_conserved() {
        // Two bars connected by a linear spring, no gravity, no damping.
        // Total energy (KE + spring PE) should be conserved.
        //
        // Setup: two point-mass bars pinned to ground, connected by a spring.
        // bar_a at ground point (-1, 0), bar_b at ground point (1, 0).
        // Spring connects the tips (at local (1, 0)) of each bar.
        let ground = make_ground(&[("P1", -1.0, 0.0), ("P2", 1.0, 0.0)]);

        let mut bar_a = Body::new("bar_a");
        bar_a.add_attachment_point("A", 0.0, 0.0).unwrap();
        bar_a.add_attachment_point("tip", 1.0, 0.0).unwrap();
        bar_a.mass = 1.0;
        bar_a.cg_local = Vector2::new(0.5, 0.0);
        bar_a.izz_cg = 0.01;

        let mut bar_b = Body::new("bar_b");
        bar_b.add_attachment_point("B", 0.0, 0.0).unwrap();
        bar_b.add_attachment_point("tip", 1.0, 0.0).unwrap();
        bar_b.mass = 1.0;
        bar_b.cg_local = Vector2::new(0.5, 0.0);
        bar_b.izz_cg = 0.01;

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar_a).unwrap();
        mech.add_body(bar_b).unwrap();
        mech.add_revolute_joint("J1", "ground", "P1", "bar_a", "A")
            .unwrap();
        mech.add_revolute_joint("J2", "ground", "P2", "bar_b", "B")
            .unwrap();

        // Spring between tips of bar_a and bar_b
        let spring_k = 50.0;
        // Free length = distance between tips when both bars hang straight down
        // At theta=-PI/2: tip of bar_a = (-1, 0) + R(-PI/2)*(1,0) = (-1, -1)
        //                  tip of bar_b = ( 1, 0) + R(-PI/2)*(1,0) = ( 1, -1)
        // Distance = 2.0
        let free_length = 2.0;
        mech.add_force(ForceElement::LinearSpring(LinearSpringElement {
            body_a: "bar_a".into(),
            point_a: [1.0, 0.0],
            body_b: "bar_b".into(),
            point_b: [1.0, 0.0],
            stiffness: spring_k,
            free_length,
        }));
        mech.build().unwrap();

        // Start with bars at different angles to load the spring.
        // bar_a at -80 deg, bar_b at -100 deg (symmetric offset from -90).
        let state = mech.state();
        let mut q0 = state.make_q();
        state.set_pose("bar_a", &mut q0, -1.0, 0.0, -80.0_f64.to_radians());
        state.set_pose("bar_b", &mut q0, 1.0, 0.0, -100.0_f64.to_radians());
        let qd0 = DVector::zeros(state.n_coords());

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.001,
            ..Default::default()
        };

        let t_eval: Vec<f64> = (0..=300).map(|i| i as f64 * 0.01).collect();
        let result = simulate(
            &mech,
            &q0,
            &qd0,
            (0.0, 3.0),
            Some(&config),
            Some(&t_eval),
        )
        .unwrap();
        assert!(result.success);

        // Compute total energy (KE + spring PE) at each step
        let mut energies: Vec<f64> = Vec::new();
        for i in 0..result.t.len() {
            let q = &result.q[i];
            let qd = &result.q_dot[i];

            // KE
            let m_mat = assemble_mass_matrix(&mech, q);
            let ke = 0.5 * qd.dot(&(&m_mat * qd));

            // Spring PE = 0.5 * k * (length - free_length)^2
            let tip_a_local = Vector2::new(1.0, 0.0);
            let tip_b_local = Vector2::new(1.0, 0.0);
            let tip_a = state.body_point_global("bar_a", &tip_a_local, q);
            let tip_b = state.body_point_global("bar_b", &tip_b_local, q);
            let length = (tip_b - tip_a).norm();
            let extension = length - free_length;
            let spring_pe = 0.5 * spring_k * extension * extension;

            energies.push(ke + spring_pe);
        }

        let e0 = energies[0];
        let max_deviation = energies
            .iter()
            .fold(0.0_f64, |acc, &e| acc.max((e - e0).abs()));

        // Energy should stay within 5% of initial (no dissipation)
        assert!(
            max_deviation < e0.abs() * 0.05 + 1e-6,
            "Spring-mass energy not conserved: max_deviation={:e}, initial={:e}, ratio={:e}",
            max_deviation,
            e0,
            max_deviation / (e0.abs() + 1e-15)
        );
    }

    #[test]
    fn joint_limit_prevents_excessive_rotation() {
        // Pendulum with gravity and joint limits. The bar should bounce
        // off the limit and never exceed [angle_min, angle_max] by more
        // than a small overshoot (penalty-based compliance).
        let ground = make_ground(&[("O", 0.0, 0.0)]);

        let mut bar = Body::new("bar");
        bar.add_attachment_point("A", 0.0, 0.0).unwrap();
        bar.mass = 1.0;
        bar.cg_local = Vector2::new(1.0, 0.0);
        bar.izz_cg = 0.0;

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
            .unwrap();
        mech.add_force(ForceElement::Gravity(GravityElement::default()));

        // Joint limits: allow rotation in [-2.5, -0.5] rad
        // (hanging is at -PI/2 ~ -1.571, so this range allows about 60 deg
        // swing on either side of hanging)
        let angle_min = -2.5;
        let angle_max = -0.5;
        mech.add_force(ForceElement::JointLimit(JointLimitElement {
            body_i: "ground".into(),
            body_j: "bar".into(),
            angle_min,
            angle_max,
            stiffness: 5000.0,
            damping: 100.0,
            restitution: 0.3,
        }));
        mech.build().unwrap();

        // Start near the upper limit so the pendulum hits it
        let theta0 = -0.6; // just inside the upper limit
        let (q0, qd0) = pendulum_initial_state(&mech, theta0);

        let config = ForwardDynamicsConfig {
            alpha: 10.0,
            beta: 10.0,
            max_step: 0.0005, // small step for stiff penalty
            ..Default::default()
        };

        let t_eval: Vec<f64> = (0..=500).map(|i| i as f64 * 0.006).collect();
        let result = simulate(
            &mech,
            &q0,
            &qd0,
            (0.0, 3.0),
            Some(&config),
            Some(&t_eval),
        )
        .unwrap();
        assert!(result.success);

        // Check that the angle stays within limits plus a small overshoot
        // allowed by the penalty method. With k=5000 and c=100, overshoot
        // should be modest.
        let bar_idx = mech.state().get_index("bar").unwrap();
        let overshoot_tolerance = 0.15; // rad ~ 8.6 deg

        for (i, q) in result.q.iter().enumerate() {
            let theta = q[bar_idx.theta_idx()];
            assert!(
                theta >= angle_min - overshoot_tolerance,
                "Angle {} rad below min limit at t={}: min={}, overshoot={}",
                theta,
                result.t[i],
                angle_min,
                angle_min - theta,
            );
            assert!(
                theta <= angle_max + overshoot_tolerance,
                "Angle {} rad above max limit at t={}: max={}, overshoot={}",
                theta,
                result.t[i],
                angle_max,
                theta - angle_max,
            );
        }

        // Additionally, verify the pendulum actually reached a limit at
        // some point (i.e., the limits were exercised, not just avoided).
        let any_near_max = result
            .q
            .iter()
            .any(|q| q[bar_idx.theta_idx()] > angle_max - 0.1);
        let any_near_min = result
            .q
            .iter()
            .any(|q| q[bar_idx.theta_idx()] < angle_min + 0.1);
        assert!(
            any_near_max || any_near_min,
            "Pendulum never approached either joint limit"
        );
    }
}
