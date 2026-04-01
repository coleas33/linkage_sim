//! Hardcoded sample mechanism builders for the GUI.

mod fourbar;
pub mod helpers;
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
    ChebyshevLambdaActuator,
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
    // Phase 7 additions (classic mechanisms)
    Hoeken,
    Roberts,
    OffsetSliderCrank,
    WhitworthQuickReturn,
    BellCrank,
    PeaucellierLipkin,
    WattII,
    Pantograph,
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
            SampleMechanism::Chebyshev => "Chebyshev Lambda (Straight-Line)",
            SampleMechanism::ChebyshevLambdaActuator => "Chebyshev Lambda + Actuator",
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
            SampleMechanism::Hoeken => "Hoeken Straight-Line",
            SampleMechanism::Roberts => "Roberts Straight-Line",
            SampleMechanism::OffsetSliderCrank => "Offset Slider-Crank",
            SampleMechanism::WhitworthQuickReturn => "Whitworth Quick-Return",
            SampleMechanism::BellCrank => "Bell Crank (90\u{00b0} redirect)",
            SampleMechanism::PeaucellierLipkin => "Coupler Curve (figure-8)",
            SampleMechanism::WattII => "6-Bar Watt II",
            SampleMechanism::Pantograph => "Pantograph (motion scaling)",
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
            SampleMechanism::ChebyshevLambdaActuator,
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
            SampleMechanism::Hoeken,
            SampleMechanism::Roberts,
            SampleMechanism::OffsetSliderCrank,
            SampleMechanism::WhitworthQuickReturn,
            SampleMechanism::BellCrank,
            SampleMechanism::PeaucellierLipkin,
            SampleMechanism::WattII,
            SampleMechanism::Pantograph,
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
        SampleMechanism::ChebyshevLambdaActuator => fourbar::build_chebyshev_lambda_actuator(driver_joint_id),
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
        SampleMechanism::Hoeken => fourbar::build_hoeken_with_driver(driver_joint_id),
        SampleMechanism::Roberts => fourbar::build_roberts_with_driver(driver_joint_id),
        SampleMechanism::OffsetSliderCrank => special::build_offset_slider_crank(driver_joint_id),
        SampleMechanism::WhitworthQuickReturn => special::build_whitworth_quick_return(driver_joint_id),
        SampleMechanism::BellCrank => special::build_bell_crank(driver_joint_id),
        SampleMechanism::PeaucellierLipkin => special::build_peaucellier_lipkin(driver_joint_id),
        SampleMechanism::WattII => sixbar::build_watt_ii(driver_joint_id),
        SampleMechanism::Pantograph => sixbar::build_pantograph(driver_joint_id),
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
    fn chebyshev_lambda_endpoint_traces_approximate_straight_line() {
        use nalgebra::Vector2;
        let (mech, q0) = build_sample(SampleMechanism::Chebyshev);
        let state = mech.state();

        // Grashof crank-rocker: sweep full 0-360°.
        let omega = 1.0;
        let theta_0 = 0.0;
        let coupler_m = Vector2::new(10.0, 0.0); // M at coupler end

        let mut trace: Vec<[f64; 2]> = Vec::new();
        let mut q = q0.clone();
        for deg in 0..=360 {
            let angle_rad = (deg as f64).to_radians();
            let t = (angle_rad - theta_0) / omega;
            match solve_position(&mech, &q, t, 1e-10, 50) {
                Ok(result) if result.converged => {
                    q = result.q.clone();
                    let global = state.body_point_global("coupler", &coupler_m, &q);
                    trace.push([global.x, global.y]);
                }
                _ => {}
            }
        }

        assert_eq!(trace.len(), 361, "Grashof mechanism should converge at all 361 angles");

        // Find the straightest 30% contiguous window (avoids near-singular
        // region at theta≈0° where M deviates from the straight line).
        let window = trace.len() * 30 / 100;
        let mut best_ratio = f64::MAX;
        for start in 0..=(trace.len() - window) {
            let w = &trace[start..start + window];
            let y_mean: f64 = w.iter().map(|p| p[1]).sum::<f64>() / w.len() as f64;
            let max_dev: f64 = w.iter().map(|p| (p[1] - y_mean).abs()).fold(0.0_f64, f64::max);
            let x_range: f64 = w.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max)
                - w.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
            if x_range > 0.01 {
                let ratio = max_dev / x_range;
                if ratio < best_ratio {
                    best_ratio = ratio;
                }
            }
        }

        // The lambda linkage is not perfectly straight — it's an approximate
        // straight-line mechanism. Accept ~20% deviation in the best window.
        assert!(
            best_ratio < 0.20,
            "Chebyshev lambda M trace: best straight section ratio={:.4} (want <0.20)",
            best_ratio,
        );
    }

    #[test]
    fn chebyshev_lambda_actuator_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::ChebyshevLambdaActuator);
        assert!(mech.is_built());
        assert_eq!(
            mech.n_drivers(), 1,
            "should have one revolute driver"
        );
        assert_eq!(
            mech.n_linear_drivers(), 0,
            "should have no linear driver"
        );
        // Verify there is exactly one LinearActuator force element.
        let actuator_count = mech.forces().iter()
            .filter(|f| matches!(f, crate::forces::elements::ForceElement::LinearActuator(_)))
            .count();
        assert_eq!(actuator_count, 1, "should have one LinearActuator force element");
        // Verify solver converges at initial state
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "should converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    /// Sweep the crank 0-360 and compute the distance from the actuator base (O_act)
    /// to coupler endpoint M at each angle. Reports the actual min/max stroke range
    /// the actuator needs to cover the full mechanism travel.
    ///
    /// Uses the same actuator base position as the builder.
    #[test]
    fn actual_stroke_range() {
        use nalgebra::Vector2;

        // Same link dimensions as build_chebyshev_lambda_actuator
        let o2 = (0.0_f64, 0.0_f64);
        let o4 = (0.076_f64, 0.0_f64);
        let l_crank = 0.0444_f64;
        let l_coupler_ab = 0.0919_f64;
        let l_rocker = 0.0919_f64;
        let l_total_coupler = 0.1838_f64;

        use crate::core::body::{make_bar, make_ground};

        let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
        let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
        let mut coupler = make_bar("coupler", "B", "M", l_total_coupler, 0.0, 0.0);
        coupler.add_attachment_point("C", l_coupler_ab, 0.0).unwrap();
        coupler.add_coupler_point("M", l_total_coupler, 0.0).unwrap();
        let rocker = make_bar("rocker", "C", "D", l_rocker, 0.0, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
        // Revolute driver on crank (1 rev/s = 2*PI rad/s), theta_0=0
        mech.add_constant_speed_driver("D1", "ground", "crank", std::f64::consts::TAU, 0.0).unwrap();
        mech.build().unwrap();

        // Initial pose (above=true for +y orientation, matching the actuator sample)
        let q0 = helpers::fourbar_initial_q0(
            mech.state(), o2, o4, l_crank, l_coupler_ab, l_rocker, 0.0,
            "crank", "coupler", "rocker", true,
        );

        let state = mech.state();
        let coupler_m_local = Vector2::new(l_total_coupler, 0.0);

        // Same actuator base as the builder: x=-0.15, y=avg_my.
        // Extract the actual base position from the built mechanism to stay in
        // sync with the builder rather than duplicating the avg_my computation.
        let (mech_check, _) = super::build_sample(SampleMechanism::ChebyshevLambdaActuator);
        let (act_base_x, act_base_y) = mech_check.forces().iter().find_map(|f| {
            if let crate::forces::elements::ForceElement::LinearActuator(act) = f {
                Some((act.point_a[0], act.point_a[1]))
            } else {
                None
            }
        }).expect("should have actuator");

        let act_base = Vector2::new(act_base_x, act_base_y);

        println!("O_act = ({:.6}, {:.6})", act_base_x, act_base_y);

        let mut min_dist = f64::MAX;
        let mut max_dist = f64::NEG_INFINITY;
        let mut min_angle = 0.0_f64;
        let mut max_angle = 0.0_f64;
        let mut q = q0.clone();

        for deg in 0..=360 {
            let angle_rad = (deg as f64).to_radians();
            // time = angle / omega, omega = 2*PI
            let t = angle_rad / std::f64::consts::TAU;
            match solve_position(&mech, &q, t, 1e-10, 100) {
                Ok(result) if result.converged => {
                    q = result.q.clone();
                    let m_global = state.body_point_global("coupler", &coupler_m_local, &q);
                    let dist = (m_global - act_base).norm();
                    if dist < min_dist {
                        min_dist = dist;
                        min_angle = deg as f64;
                    }
                    if dist > max_dist {
                        max_dist = dist;
                        max_angle = deg as f64;
                    }
                    if deg % 30 == 0 {
                        println!(
                            "  theta={:>3} deg  M=({:.6}, {:.6})  dist={:.6} m ({:.2} mm)",
                            deg, m_global.x, m_global.y, dist, dist * 1000.0,
                        );
                    }
                }
                Ok(result) => {
                    println!("  theta={} deg: did NOT converge, residual={}", deg, result.residual_norm);
                }
                Err(e) => {
                    println!("  theta={} deg: solver error: {}", deg, e);
                }
            }
        }

        println!("\n=== ACTUAL STROKE RANGE ===");
        println!("  min distance = {:.6} m ({:.2} mm) at theta = {} deg", min_dist, min_dist * 1000.0, min_angle);
        println!("  max distance = {:.6} m ({:.2} mm) at theta = {} deg", max_dist, max_dist * 1000.0, max_angle);
        println!("  stroke range = {:.6} m ({:.2} mm)", max_dist - min_dist, (max_dist - min_dist) * 1000.0);

        // Verify we got reasonable results (crank completed full revolution)
        assert!(min_dist > 0.0, "min distance should be positive");
        assert!(max_dist > min_dist, "max should exceed min");
        // Stroke range should be substantial (~180mm+), not ~6mm
        assert!(
            max_dist - min_dist > 0.10,
            "stroke range should be >100mm, got {:.2} mm",
            (max_dist - min_dist) * 1000.0,
        );
    }

    /// Verify the builder's stroke_min/stroke_max cover the full mechanism range.
    /// The actuator is now a force element only (no linear driver), driven by a
    /// standard revolute driver on J1. The stroke limits on the force element
    /// should still be correct.
    #[test]
    fn chebyshev_lambda_actuator_stroke_covers_full_range() {
        use nalgebra::Vector2;
        use crate::forces::elements::ForceElement;

        let (mech, q0) = build_sample(SampleMechanism::ChebyshevLambdaActuator);
        let state = mech.state();

        // Extract actuator stroke limits from the force element.
        let (stroke_min, stroke_max) = mech.forces().iter().find_map(|f| {
            if let ForceElement::LinearActuator(act) = f {
                Some((act.stroke_min, act.stroke_max))
            } else {
                None
            }
        }).expect("should have a LinearActuator force element");

        println!("Builder stroke_min = {:.4} mm", stroke_min * 1000.0);
        println!("Builder stroke_max = {:.4} mm", stroke_max * 1000.0);
        println!("Builder stroke range = {:.4} mm", (stroke_max - stroke_min) * 1000.0);

        // The stroke range should be substantial (>100mm), not ~6mm.
        assert!(
            stroke_max - stroke_min > 0.10,
            "stroke range should be >100mm, got {:.2} mm",
            (stroke_max - stroke_min) * 1000.0,
        );

        // Verify no linear drivers remain (revolute driver only).
        assert_eq!(mech.n_linear_drivers(), 0, "should have no linear drivers");
        assert_eq!(mech.n_drivers(), 1, "should have one revolute driver");

        // Verify solver converges at t=0.
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(result.converged, "should converge at t=0, residual={}", result.residual_norm);

        // Verify actuator base is roughly in line with M's y-level
        // (aligned with the straight-line trace, not at ground y=0).
        let coupler_m_local = Vector2::new(0.1838, 0.0);
        let m_global = state.body_point_global("coupler", &coupler_m_local, &q0);
        let act_point_a = mech.forces().iter().find_map(|f| {
            if let ForceElement::LinearActuator(act) = f {
                Some(act.point_a)
            } else {
                None
            }
        }).unwrap();

        let y_offset = (m_global.y - act_point_a[1]).abs();
        println!("Actuator base y={:.4}, M y={:.4}, offset={:.4} mm",
            act_point_a[1] * 1000.0, m_global.y * 1000.0, y_offset * 1000.0);
        // The base is at the average M y across all crank angles, so the
        // offset from M at the initial crank angle should be moderate (not
        // zero, since M varies, but well under 100mm).
        assert!(
            act_point_a[1] > 0.05,
            "actuator base should be above ground level (in line with trace), got y={:.4} mm",
            act_point_a[1] * 1000.0,
        );
        assert!(
            y_offset < 0.10,
            "actuator base should be roughly in line with M's trace, offset={:.4} mm is too large",
            y_offset * 1000.0,
        );
    }

    #[test]
    fn chebyshev_lambda_actuator_trace_in_positive_y() {
        use nalgebra::Vector2;
        let (mech, q0) = build_sample(SampleMechanism::ChebyshevLambdaActuator);
        let state = mech.state();
        let coupler_m = Vector2::new(0.1838, 0.0); // M in coupler local coords
        let m_global = state.body_point_global("coupler", &coupler_m, &q0);
        assert!(
            m_global.y > 0.0,
            "M should be in +y, got y={}",
            m_global.y
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
    fn fourbar_initial_q0_above_places_joint_in_positive_y() {
        // Build a standard Grashof 4-bar (same geometry as the FourBar sample).
        let o2 = (0.0_f64, 0.0_f64);
        let o4 = (0.038_f64, 0.0_f64);
        let l_crank = 0.01_f64;
        let l_coupler = 0.04_f64;
        let l_rocker = 0.03_f64;
        let theta_crank = 0.0_f64;

        use crate::core::body::make_bar;
        use crate::core::body::make_ground;

        let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
        let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
        let coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", l_rocker, 0.0, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
        mech.add_constant_speed_driver("D1", "ground", "crank", std::f64::consts::TAU, 0.0).unwrap();
        mech.build().unwrap();

        let state = mech.state();

        // above = false: coupler-rocker joint C should be below ground line (y < 0).
        let q0_below = helpers::fourbar_initial_q0(
            state, o2, o4, l_crank, l_coupler, l_rocker, theta_crank,
            "crank", "coupler", "rocker", false,
        );
        let rocker_idx = state.get_index("rocker").unwrap();
        let cy_below = q0_below[rocker_idx.y_idx()];
        assert!(
            cy_below < 0.0,
            "above=false should place rocker origin C below ground line, got y={}",
            cy_below,
        );

        // above = true: coupler-rocker joint C should be above ground line (y > 0).
        let q0_above = helpers::fourbar_initial_q0(
            state, o2, o4, l_crank, l_coupler, l_rocker, theta_crank,
            "crank", "coupler", "rocker", true,
        );
        let cy_above = q0_above[rocker_idx.y_idx()];
        assert!(
            cy_above > 0.0,
            "above=true should place rocker origin C above ground line, got y={}",
            cy_above,
        );
    }

    #[test]
    fn all_samples_listed() {
        assert_eq!(SampleMechanism::all().len(), 28);
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

    #[test]
    fn scotch_yoke_full_sweep_convergence() {
        use crate::gui::state::AppState;
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ScotchYoke);
        let sweep = state.sweep_data.as_ref().expect("sweep data");
        let total = sweep.angles_deg.len();
        eprintln!("ScotchYoke: {}/361 converged", total);
        assert!(total > 300, "ScotchYoke should converge for most angles, got {}", total);
    }

    #[test]
    fn inverted_slider_crank_full_sweep_convergence() {
        use crate::gui::state::AppState;
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::InvertedSliderCrank);
        let sweep = state.sweep_data.as_ref().expect("sweep data");
        let total = sweep.angles_deg.len();
        eprintln!("InvertedSliderCrank: {}/361 converged", total);
        assert!(total > 300, "InvertedSliderCrank should converge for most angles, got {}", total);
    }

    #[test]
    fn hoeken_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::Hoeken);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Hoeken sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn roberts_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::Roberts);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Roberts sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn offset_slider_crank_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::OffsetSliderCrank);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Offset slider-crank sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn whitworth_quick_return_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::WhitworthQuickReturn);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Whitworth quick-return sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn bell_crank_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::BellCrank);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 100).unwrap();
        assert!(
            result.converged,
            "Bell crank sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn peaucellier_lipkin_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::PeaucellierLipkin);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 100).unwrap();
        assert!(
            result.converged,
            "Peaucellier-Lipkin sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn watt_ii_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::WattII);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 100).unwrap();
        assert!(
            result.converged,
            "Watt II sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }

    #[test]
    fn pantograph_sample_builds_and_solves() {
        let (mech, q0) = build_sample(SampleMechanism::Pantograph);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Pantograph sample did not converge at t=0, residual = {}",
            result.residual_norm
        );
    }
}
