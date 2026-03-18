//! Global constraint system assembly from a Mechanism.
//!
//! Assembles the global Φ, Φ_q, Φ_t, and γ vectors/matrices by stacking
//! contributions from all joints in the mechanism.

use nalgebra::{DMatrix, DVector};

use crate::core::mechanism::Mechanism;

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
