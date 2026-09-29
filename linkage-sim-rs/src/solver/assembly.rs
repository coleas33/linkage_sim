//! Global constraint system assembly from a Mechanism.
//!
//! Assembles the global Φ, Φ_q, Φ_t, and γ vectors/matrices by stacking
//! contributions from all joints in the mechanism.
//! Also provides the global block-diagonal mass matrix M.

use nalgebra::{DMatrix, DVector};

use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;

/// Assemble global constraint residual vector Φ(q, t).
pub fn assemble_constraints(mech: &Mechanism, q: &DVector<f64>, t: f64) -> DVector<f64> {
    let m = mech.n_constraints();
    let mut phi = DVector::zeros(m);
    let state = mech.state();

    let mut row = 0;
    for joint in mech.all_constraints() {
        let n_eq = joint.n_equations();
        let c = joint.constraint(state, q, t);
        for i in 0..n_eq {
            phi[row + i] = c[i];
        }
        row += n_eq;
    }

    phi
}

/// Assemble global constraint Jacobian Φ_q(q, t).
pub fn assemble_jacobian(mech: &Mechanism, q: &DVector<f64>, t: f64) -> DMatrix<f64> {
    let m = mech.n_constraints();
    let n = mech.state().n_coords();
    let mut phi_q = DMatrix::zeros(m, n);
    let state = mech.state();

    let mut row = 0;
    for joint in mech.all_constraints() {
        let n_eq = joint.n_equations();
        let jac = joint.jacobian(state, q, t);
        for i in 0..n_eq {
            for j in 0..n {
                phi_q[(row + i, j)] = jac[(i, j)];
            }
        }
        row += n_eq;
    }

    phi_q
}

/// Assemble global time-derivative vector Φ_t(q, t).
pub fn assemble_phi_t(mech: &Mechanism, q: &DVector<f64>, t: f64) -> DVector<f64> {
    let m = mech.n_constraints();
    let mut phi_t = DVector::zeros(m);
    let state = mech.state();

    let mut row = 0;
    for joint in mech.all_constraints() {
        let n_eq = joint.n_equations();
        let pt = joint.phi_t(state, q, t);
        for i in 0..n_eq {
            phi_t[row + i] = pt[i];
        }
        row += n_eq;
    }

    phi_t
}

/// Assemble global acceleration RHS vector γ(q, q̇, t).
pub fn assemble_gamma(
    mech: &Mechanism,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    t: f64,
) -> DVector<f64> {
    let m = mech.n_constraints();
    let mut gamma = DVector::zeros(m);
    let state = mech.state();

    let mut row = 0;
    for joint in mech.all_constraints() {
        let n_eq = joint.n_equations();
        let g = joint.gamma(state, q, q_dot, t);
        for i in 0..n_eq {
            gamma[row + i] = g[i];
        }
        row += n_eq;
    }

    gamma
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
///
/// Ground bodies and massless bodies (mass <= 0.0) are skipped.
pub fn assemble_mass_matrix(mech: &Mechanism, q: &DVector<f64>) -> DMatrix<f64> {
    let state = mech.state();
    let n = state.n_coords();
    let mut m_mat = DMatrix::zeros(n, n);

    for (body_id, body) in mech.bodies() {
        if body_id == GROUND_ID || body.mass <= 0.0 {
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
        if s_cg_sq > 0.0 {
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

/// Assemble the velocity-quadratic generalized force vector Q_v(q, q̇).
///
/// Because M depends on θ when the CG is offset from the body origin, the
/// Lagrange equations carry a centripetal term: M q̈ = Q + Q_v + (constraint
/// forces). Per moving body, from a_cg = r̈ + B s θ̈ − A s θ̇²:
///
/// ```text
///   Q_v = [ m θ̇² (A(θ) s_cg)_x,  m θ̇² (A(θ) s_cg)_y,  0 ]
/// ```
///
/// The θ row is zero because the ṙ·(A s) θ̇ contributions of d/dt(∂T/∂θ̇)
/// and ∂T/∂θ cancel, and (B s)·(A s) = 0.
///
/// Zero when s_cg = 0. Ground and massless bodies (mass <= 0.0) are skipped,
/// matching `assemble_mass_matrix`.
pub fn assemble_quadratic_velocity_forces(
    mech: &Mechanism,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let state = mech.state();
    let mut q_v = DVector::zeros(state.n_coords());

    for (body_id, body) in mech.bodies() {
        if body_id == GROUND_ID || body.mass <= 0.0 {
            continue;
        }

        let idx = state.get_index(body_id).expect("body not registered");
        let s_cg = &body.cg_local;
        let theta = q[idx.theta_idx()];
        let theta_dot = q_dot[idx.theta_idx()];
        let (sin_t, cos_t) = theta.sin_cos();
        // A(theta) * s_cg = [cos(theta)*sx - sin(theta)*sy, sin(theta)*sx + cos(theta)*sy]
        let as_x = cos_t * s_cg.x - sin_t * s_cg.y;
        let as_y = sin_t * s_cg.x + cos_t * s_cg.y;
        let coeff = body.mass * theta_dot * theta_dot;

        q_v[idx.x_idx()] = coeff * as_x;
        q_v[idx.y_idx()] = coeff * as_y;
    }

    q_v
}
