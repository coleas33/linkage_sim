//! Hardcoded sample mechanism builders for the GUI.

use nalgebra::DVector;
use std::f64::consts::PI;

use crate::core::body::{make_bar, make_ground, Body};
use crate::core::constraint::Constraint;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;

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
        SampleMechanism::FourBar => build_fourbar_with_driver(driver_joint_id),
        SampleMechanism::SliderCrank => build_slider_crank_with_driver(driver_joint_id),
    }
}

/// Attach a constant-speed driver to a grounded revolute joint.
///
/// Finds the joint by ID in `mech.joints()`, verifies it is revolute and grounded,
/// determines the non-ground body (the driven body), and calls
/// `mech.add_constant_speed_driver(driver_id, ground_id, driven_id, 2*PI, 0.0)`.
///
/// Returns the ID of the driven (non-ground) body on success.
pub fn attach_driver_to_grounded_revolute(
    mech: &mut Mechanism,
    joint_id: &str,
    driver_id: &str,
) -> Result<String, String> {
    attach_driver_to_grounded_revolute_with_theta0(mech, joint_id, driver_id, 0.0)
}

/// Core implementation: attach a constant-speed driver to a grounded revolute joint
/// with a caller-specified initial phase angle `theta_0`.
///
/// `theta_0` should match the initial-guess angle of the driven body so that the
/// driver constraint is satisfied at t = 0 without requiring the solver to correct it.
fn attach_driver_to_grounded_revolute_with_theta0(
    mech: &mut Mechanism,
    joint_id: &str,
    driver_id: &str,
    theta_0: f64,
) -> Result<String, String> {
    let joint = mech
        .joints()
        .iter()
        .find(|j| j.id() == joint_id)
        .ok_or_else(|| format!("joint '{}' not found in mechanism", joint_id))?;

    if !joint.is_revolute() {
        return Err(format!(
            "joint '{}' is not a revolute joint and cannot be used as a driver",
            joint_id
        ));
    }

    let body_i = joint.body_i_id().to_string();
    let body_j = joint.body_j_id().to_string();

    let driven_body = if body_i == GROUND_ID {
        body_j
    } else if body_j == GROUND_ID {
        body_i
    } else {
        return Err(format!(
            "joint '{}' connects '{}' and '{}' — neither is ground; \
             only grounded revolute joints can be used as drivers",
            joint_id, body_i, body_j
        ));
    };

    mech.add_constant_speed_driver(driver_id, GROUND_ID, &driven_body, 2.0 * PI, theta_0)
        .map_err(|e| e.to_string())?;

    Ok(driven_body)
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
fn build_fourbar_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    // Link lengths and ground pivot locations.
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (0.038_f64, 0.0_f64);
    let l_crank = 0.01_f64;
    let l_coupler = 0.04_f64;
    let l_rocker = 0.03_f64;

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    let coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
    let rocker = make_bar("rocker", "C", "D", l_rocker, 0.0, 0.0);

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

    // Determine initial crank angle. For J1-driven the crank angle is the
    // driver's theta_0. For J4-driven we compute a consistent crank angle
    // from the rocker angle via the 4-bar loop closure.
    let joint_id = driver_joint_id.unwrap_or("J1");

    let (theta_crank, theta_0) = match joint_id {
        "J1" => {
            // Default: drive crank at theta_crank = 0.
            (0.0_f64, 0.0_f64)
        }
        "J4" => {
            // Drive rocker. Compute a consistent crank angle from the
            // rocker's initial angle using the 4-bar loop closure.
            let theta_rocker = fourbar_rocker_angle_for_crank(
                o2, o4, l_crank, l_coupler, l_rocker, 0.0,
            );
            let theta_crank_for_j4 = 0.0_f64;
            (theta_crank_for_j4, theta_rocker)
        }
        _ => {
            // Let attach_driver_to_grounded_revolute_with_theta0 validate
            // and produce the proper error for unknown/non-grounded joints.
            (0.0, 0.0)
        }
    };

    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", theta_0)?;

    mech.build().map_err(|e| e.to_string())?;

    // Compute geometrically consistent initial poses from the crank angle.
    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4, l_crank, l_coupler, l_rocker, theta_crank,
    );

    Ok((mech, q0))
}

/// Compute the rocker angle for a given crank angle in the 4-bar sample.
///
/// Uses the law of cosines on the diagonal from O2+crank_tip to O4 to find
/// the rocker angle that closes the loop.
fn fourbar_rocker_angle_for_crank(
    o2: (f64, f64),
    o4: (f64, f64),
    l_crank: f64,
    l_coupler: f64,
    l_rocker: f64,
    theta_crank: f64,
) -> f64 {
    // Crank tip (point B) in world frame.
    let bx = o2.0 + l_crank * theta_crank.cos();
    let by = o2.1 + l_crank * theta_crank.sin();

    // Distance from O4 to crank tip B.
    let dx = bx - o4.0;
    let dy = by - o4.1;
    let d = (dx * dx + dy * dy).sqrt();

    // Angle from O4 to B.
    let alpha = dy.atan2(dx);

    // Law of cosines: coupler^2 = d^2 + rocker^2 - 2*d*rocker*cos(beta)
    // => cos(beta) = (d^2 + rocker^2 - coupler^2) / (2*d*rocker)
    let cos_beta = (d * d + l_rocker * l_rocker - l_coupler * l_coupler)
        / (2.0 * d * l_rocker);
    let cos_beta = cos_beta.clamp(-1.0, 1.0);
    let beta = cos_beta.acos();

    // Rocker angle: the rocker points from its origin (at point C) to
    // point D at O4. Since make_bar puts point C at local (0,0) and D at
    // local (l_rocker, 0), the body angle = direction from C to D.
    // C is at O4 + R(rocker_angle) * (l_rocker, 0) going backwards:
    // actually D is at O4, and the rocker body origin is at C.
    // rocker_angle = angle from C to D = atan2(D.y - C.y, D.x - C.x).
    //
    // But we're computing from O4's perspective: C is at
    //   O4 - R(rocker_angle) * (l_rocker, 0)
    // which means D→C direction is opposite to rocker_angle.
    //
    // The triangle at O4: the rocker extends from C to D=O4. The rocker
    // angle is the direction from C to D. From O4, point C is at angle
    // (alpha + beta) at distance l_rocker. So C→D direction = alpha + beta + PI.
    // But rocker_angle = direction from C to D = alpha + beta + PI.
    //
    // Choose the solution branch (+ beta gives the "open" configuration).
    alpha + beta + PI
}

/// Compute geometrically consistent initial poses for the 4-bar sample
/// given a crank angle.
///
/// Works by forward kinematics: place the crank at the given angle, then
/// solve the coupler/rocker positions via the loop closure triangle.
fn fourbar_initial_q0(
    state: &crate::core::state::State,
    o2: (f64, f64),
    o4: (f64, f64),
    l_crank: f64,
    l_coupler: f64,
    l_rocker: f64,
    theta_crank: f64,
) -> DVector<f64> {
    let mut q0 = state.make_q();

    // Crank: origin at A = O2, angle = theta_crank.
    // Point B = O2 + R(theta_crank) * (l_crank, 0).
    let bx = o2.0 + l_crank * theta_crank.cos();
    let by = o2.1 + l_crank * theta_crank.sin();
    state.set_pose("crank", &mut q0, o2.0, o2.1, theta_crank);

    // Rocker: D must be at O4. Rocker origin is at C.
    // Solve for C position and rocker angle using triangle B-C-O4.
    let dx = bx - o4.0;
    let dy = by - o4.1;
    let d = (dx * dx + dy * dy).sqrt();
    let alpha = dy.atan2(dx);

    let cos_beta = (d * d + l_rocker * l_rocker - l_coupler * l_coupler)
        / (2.0 * d * l_rocker);
    let cos_beta = cos_beta.clamp(-1.0, 1.0);
    let beta = cos_beta.acos();

    // Rocker angle: direction from C to D (C→D = rocker body axis).
    // C = O4 - R(rocker_angle) * (l_rocker, 0), so the rocker angle
    // points from C toward D=O4. From O4's perspective, C is at angle
    // (alpha + beta) at distance l_rocker. So rocker_angle = alpha + beta + PI.
    let theta_rocker = alpha + beta + PI;
    let cx = o4.0 - l_rocker * theta_rocker.cos();
    let cy = o4.1 - l_rocker * theta_rocker.sin();
    state.set_pose("rocker", &mut q0, cx, cy, theta_rocker);

    // Coupler: origin at B (point B in local = (0,0)), angle from B→C direction.
    let theta_coupler = (cy - by).atan2(cx - bx);
    state.set_pose("coupler", &mut q0, bx, by, theta_coupler);

    q0
}

/// Slider-crank linkage with a prismatic joint.
///
/// Link lengths (meters):
/// - crank:   0.01 m
/// - coupler: 0.04 m
/// - slider:  translates along X axis from ground/rail
fn build_slider_crank_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
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

    // Slider-crank only has J1 as grounded revolute; theta_0 = 0.0 (crank at theta=0).
    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
    state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
    state.set_pose("slider", &mut q0, 0.05, 0.0, 0.0);

    Ok((mech, q0))
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
    fn all_samples_listed() {
        assert_eq!(SampleMechanism::all().len(), 2);
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
}
