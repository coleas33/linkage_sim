//! Hardcoded sample mechanism builders for the GUI.

mod fourbar;
mod helpers;
mod sixbar;
mod special;

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;

pub use helpers::attach_driver_to_grounded_revolute;

/// Named sample mechanisms available in the GUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleMechanism {
    FourBar,
    SliderCrank,
    CrankRocker,
    DoubleRocker,
    DoubleCrank,
    Parallelogram,
    ParallelogramPress,
    ParallelogramActuator,
    Chebyshev,
    TripleRocker,
    SixBarB1,
    SixBarA1,
    SixBarA2,
    SixBarB2,
    SixBarB3,
    // Phase 6.3 additions
    QuickReturn,
    ToggleClamp,
    ScotchYoke,
    InvertedSliderCrank,
}

impl SampleMechanism {
    pub fn label(&self) -> &'static str {
        match self {
            SampleMechanism::FourBar => "4-Bar Crank-Rocker",
            SampleMechanism::SliderCrank => "Slider-Crank",
            SampleMechanism::CrankRocker => "Crank-Rocker (4-2-4-3)",
            SampleMechanism::DoubleRocker => "Double-Rocker (5-3-4-7)",
            SampleMechanism::DoubleCrank => "Double-Crank (2-4-3.5-3)",
            SampleMechanism::Parallelogram => "Parallelogram (4-2-4-2)",
            SampleMechanism::ParallelogramPress => "Parallelogram Press",
            SampleMechanism::ParallelogramActuator => "Parallelogram + Actuator",
            SampleMechanism::Chebyshev => "Chebyshev (4-2-5-5)",
            SampleMechanism::TripleRocker => "Triple-Rocker (4-2-5-2)",
            SampleMechanism::SixBarB1 => "6-Bar B1 (Watt I)",
            SampleMechanism::SixBarA1 => "6-Bar A1 (Chain A, binary ground)",
            SampleMechanism::SixBarA2 => "6-Bar A2 (Chain A, ternary ground)",
            SampleMechanism::SixBarB2 => "6-Bar B2 (Chain B, shared-binary)",
            SampleMechanism::SixBarB3 => "6-Bar B3 (Chain B, exclusive-binary)",
            SampleMechanism::QuickReturn => "Quick-Return (crank-shaper)",
            SampleMechanism::ToggleClamp => "Toggle Clamp (4-bar near-toggle)",
            SampleMechanism::ScotchYoke => "Scotch Yoke (pure sinusoidal)",
            SampleMechanism::InvertedSliderCrank => "Inverted Slider-Crank",
        }
    }

    pub fn all() -> &'static [SampleMechanism] {
        &[
            SampleMechanism::FourBar,
            SampleMechanism::SliderCrank,
            SampleMechanism::CrankRocker,
            SampleMechanism::DoubleRocker,
            SampleMechanism::DoubleCrank,
            SampleMechanism::Parallelogram,
            SampleMechanism::ParallelogramPress,
            SampleMechanism::ParallelogramActuator,
            SampleMechanism::Chebyshev,
            SampleMechanism::TripleRocker,
            SampleMechanism::SixBarB1,
            SampleMechanism::SixBarA1,
            SampleMechanism::SixBarA2,
            SampleMechanism::SixBarB2,
            SampleMechanism::SixBarB3,
            SampleMechanism::QuickReturn,
            SampleMechanism::ToggleClamp,
            SampleMechanism::ScotchYoke,
            SampleMechanism::InvertedSliderCrank,
        ]
    }
}

/// Build and return a fully-built mechanism with an initial-guess state vector.
pub fn build_sample(sample: SampleMechanism) -> (Mechanism, DVector<f64>) {
    build_sample_with_driver(sample, None).expect("built-in sample must always build successfully")
}

/// Build any sample mechanism with the driver on a specified grounded revolute joint.
///
/// If `driver_joint_id` is `None`, the default driver joint for that sample is used.
/// Returns an error if the specified joint does not exist, is not revolute, or is not grounded.
pub fn build_sample_with_driver(
    sample: SampleMechanism,
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    match sample {
        SampleMechanism::FourBar => fourbar::build_fourbar_with_driver(driver_joint_id),
        SampleMechanism::SliderCrank => fourbar::build_slider_crank_with_driver(driver_joint_id),
        SampleMechanism::CrankRocker => fourbar::build_crank_rocker_with_driver(driver_joint_id),
        SampleMechanism::DoubleRocker => fourbar::build_double_rocker_with_driver(driver_joint_id),
        SampleMechanism::DoubleCrank => fourbar::build_double_crank_with_driver(driver_joint_id),
        SampleMechanism::Parallelogram => fourbar::build_parallelogram_with_driver(driver_joint_id),
        SampleMechanism::ParallelogramPress => fourbar::build_parallelogram_press(driver_joint_id),
        SampleMechanism::ParallelogramActuator => fourbar::build_parallelogram_actuator(driver_joint_id),
        SampleMechanism::Chebyshev => fourbar::build_chebyshev_with_driver(driver_joint_id),
        SampleMechanism::TripleRocker => fourbar::build_triple_rocker_with_driver(driver_joint_id),
        SampleMechanism::SixBarB1 => sixbar::build_sixbar_b1(driver_joint_id),
        SampleMechanism::SixBarA1 => sixbar::build_sixbar_a1(driver_joint_id),
        SampleMechanism::SixBarA2 => sixbar::build_sixbar_a2(driver_joint_id),
        SampleMechanism::SixBarB2 => sixbar::build_sixbar_b2(driver_joint_id),
        SampleMechanism::SixBarB3 => sixbar::build_sixbar_b3(driver_joint_id),
        SampleMechanism::QuickReturn => special::build_quick_return(driver_joint_id),
        SampleMechanism::ToggleClamp => special::build_toggle_clamp(driver_joint_id),
        SampleMechanism::ScotchYoke => special::build_scotch_yoke(driver_joint_id),
        SampleMechanism::InvertedSliderCrank => special::build_inverted_slider_crank(driver_joint_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::kinematics::solve_position;

    #[test]
    fn fourbar_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "4-bar sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn slider_crank_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::SliderCrank);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "slider-crank sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn crank_rocker_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::CrankRocker);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "crank-rocker sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn double_rocker_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::DoubleRocker);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "double-rocker sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn double_crank_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::DoubleCrank);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "double-crank sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn parallelogram_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::Parallelogram);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "parallelogram sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn parallelogram_press_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::ParallelogramPress);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "parallelogram-press sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn chebyshev_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::Chebyshev);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "chebyshev sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn triple_rocker_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::TripleRocker);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "triple-rocker sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn all_samples_listed() {
        assert_eq!(SampleMechanism::all().len(), 19);
    }

    #[test]
    fn fourbar_with_alternate_driver() {
        let (mech, q0) = build_sample_with_driver(SampleMechanism::FourBar, Some("J4")).unwrap();
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(result.converged, "residual = {}", result.residual_norm);
        assert_eq!(mech.driver_body_pair(), Some(("ground", "rocker")));
    }

    #[test]
    fn build_with_non_grounded_joint_errors() {
        let result = build_sample_with_driver(SampleMechanism::FourBar, Some("J2"));
        assert!(result.is_err());
    }

    #[test]
    fn build_with_nonexistent_joint_errors() {
        let result = build_sample_with_driver(SampleMechanism::FourBar, Some("NOPE"));
        assert!(result.is_err());
    }

    #[test]
    fn sixbar_b1_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::SixBarB1);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 100).unwrap();
        assert!(
            result.converged,
            "6-bar B1 sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn sixbar_a1_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::SixBarA1);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 100).unwrap();
        assert!(
            result.converged,
            "6-bar A1 sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn sixbar_a2_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::SixBarA2);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 100).unwrap();
        assert!(
            result.converged,
            "6-bar A2 sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn sixbar_b2_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::SixBarB2);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 100).unwrap();
        assert!(
            result.converged,
            "6-bar B2 sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn sixbar_b3_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::SixBarB3);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 100).unwrap();
        assert!(
            result.converged,
            "6-bar B3 sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }
}
