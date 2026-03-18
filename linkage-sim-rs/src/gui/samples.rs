//! Hardcoded sample mechanism builders for the GUI.

use nalgebra::DVector;
use std::f64::consts::PI;

use crate::core::body::{make_bar, make_ground, Body};
use crate::core::mechanism::Mechanism;

/// Named sample mechanisms available in the GUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleMechanism {
    FourBar,
    SliderCrank,
}

impl SampleMechanism {
    pub fn label(&self) -> &'static str {
        match self {
            SampleMechanism::FourBar => "4-Bar Crank-Rocker",
            SampleMechanism::SliderCrank => "Slider-Crank",
        }
    }

    pub fn all() -> &'static [SampleMechanism] {
        &[SampleMechanism::FourBar, SampleMechanism::SliderCrank]
    }
}

/// Build and return a fully-built mechanism with an initial-guess state vector.
pub fn build_sample(sample: SampleMechanism) -> (Mechanism, DVector<f64>) {
    match sample {
        SampleMechanism::FourBar => build_fourbar_sample(),
        SampleMechanism::SliderCrank => build_slider_crank_sample(),
    }
}

/// Grashof crank-rocker 4-bar linkage.
///
/// Link lengths (meters):
/// - ground: O2=(0,0) → O4=(0.038,0)
/// - crank:   0.01 m
/// - coupler: 0.04 m
/// - rocker:  0.03 m
///
/// Satisfies Grashof condition (shortest + longest < sum of other two):
///   0.01 + 0.04 < 0.038 + 0.03  →  0.05 < 0.068  ✓
pub fn build_fourbar_sample() -> (Mechanism, DVector<f64>) {
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

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
    state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
    state.set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);

    (mech, q0)
}

/// Slider-crank linkage with a prismatic joint.
///
/// Link lengths (meters):
/// - crank:   0.01 m
/// - coupler: 0.04 m
/// - slider:  translates along X axis from ground/rail
pub fn build_slider_crank_sample() -> (Mechanism, DVector<f64>) {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("rail", 0.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
    let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);

    let mut slider = Body::new("slider");
    slider.add_attachment_point("C", 0.0, 0.0).unwrap();

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(slider).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        .unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        .unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "slider", "C")
        .unwrap();

    mech.add_prismatic_joint(
        "P1",
        "ground",
        "rail",
        "slider",
        "C",
        nalgebra::Vector2::new(1.0, 0.0),
        0.0,
    )
    .unwrap();

    mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0)
        .unwrap();

    mech.build().unwrap();

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
    state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
    state.set_pose("slider", &mut q0, 0.05, 0.0, 0.0);

    (mech, q0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::kinematics::solve_position;

    #[test]
    fn fourbar_sample_builds_and_solves() {
        let (mech, q0) = build_fourbar_sample();
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "4-bar sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn slider_crank_sample_builds_and_solves() {
        let (mech, q0) = build_slider_crank_sample();
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "slider-crank sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn all_samples_listed() {
        assert_eq!(SampleMechanism::all().len(), 2);
    }
}
