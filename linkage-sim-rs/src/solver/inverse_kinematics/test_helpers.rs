//! Shared `#[cfg(test)]` test fixtures used by `control_target`, `solver`,
//! and `derivatives` test modules. The 4-bar mechanism + warm-start `q` it
//! produces is the canonical test linkage for inverse-kinematics unit tests.

#![cfg(test)]

use nalgebra::DVector;
use std::f64::consts::PI;

use crate::core::body::{make_bar, make_ground};
use crate::core::mechanism::Mechanism;
use crate::solver::kinematics::solve_position;

/// Build a standard 4-bar (crank/coupler/rocker on ground) with constant-speed
/// driver at `omega = 2π`. Same linkage used in `solver/kinematics.rs` tests.
pub fn build_fourbar() -> Mechanism {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
    let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
    let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
    let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
    mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0).unwrap();
    mech.build().unwrap();
    mech
}

/// Solve the standard 4-bar at trajectory time `t` and return the converged `q`.
pub fn solve_at(mech: &Mechanism, t: f64) -> DVector<f64> {
    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
    state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
    state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);
    let res = solve_position(mech, &q0, t, 1e-10, 50).unwrap();
    assert!(res.converged);
    res.q
}
