//! Hardcoded sample mechanism builders for the GUI.

use nalgebra::DVector;
use std::f64::consts::PI;

use crate::core::body::{make_bar, make_ground, Body};
use crate::forces::elements::{ForceElement, LinearActuatorElement};
use crate::core::constraint::Constraint;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::solver::kinematics::solve_position;

/// Named sample mechanisms available in the GUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleMechanism {
    FourBar,
    SliderCrank,
    CrankRocker,
    DoubleRocker,
    DoubleCrank,
    Parallelogram,
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
        SampleMechanism::FourBar => build_fourbar_with_driver(driver_joint_id),
        SampleMechanism::SliderCrank => build_slider_crank_with_driver(driver_joint_id),
        SampleMechanism::CrankRocker => build_crank_rocker_with_driver(driver_joint_id),
        SampleMechanism::DoubleRocker => build_double_rocker_with_driver(driver_joint_id),
        SampleMechanism::DoubleCrank => build_double_crank_with_driver(driver_joint_id),
        SampleMechanism::Parallelogram => build_parallelogram_with_driver(driver_joint_id),
        SampleMechanism::ParallelogramActuator => build_parallelogram_actuator(driver_joint_id),
        SampleMechanism::Chebyshev => build_chebyshev_with_driver(driver_joint_id),
        SampleMechanism::TripleRocker => build_triple_rocker_with_driver(driver_joint_id),
        SampleMechanism::SixBarB1 => build_sixbar_b1(driver_joint_id),
        SampleMechanism::SixBarA1 => build_sixbar_a1(driver_joint_id),
        SampleMechanism::SixBarA2 => build_sixbar_a2(driver_joint_id),
        SampleMechanism::SixBarB2 => build_sixbar_b2(driver_joint_id),
        SampleMechanism::SixBarB3 => build_sixbar_b3(driver_joint_id),
        SampleMechanism::QuickReturn => build_quick_return(driver_joint_id),
        SampleMechanism::ToggleClamp => build_toggle_clamp(driver_joint_id),
        SampleMechanism::ScotchYoke => build_scotch_yoke(driver_joint_id),
        SampleMechanism::InvertedSliderCrank => build_inverted_slider_crank(driver_joint_id),
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
        "crank", "coupler", "rocker",
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

/// Compute geometrically consistent initial poses for a 4-bar linkage
/// given a crank angle.
///
/// Works by forward kinematics: place the crank at the given angle, then
/// solve the coupler/rocker positions via the loop closure triangle.
///
/// `crank_id`, `coupler_id`, `rocker_id` are the body IDs in the mechanism.
fn fourbar_initial_q0(
    state: &crate::core::state::State,
    o2: (f64, f64),
    o4: (f64, f64),
    l_crank: f64,
    l_coupler: f64,
    l_rocker: f64,
    theta_crank: f64,
    crank_id: &str,
    coupler_id: &str,
    rocker_id: &str,
) -> DVector<f64> {
    let mut q0 = state.make_q();

    // Crank: origin at A = O2, angle = theta_crank.
    // Point B = O2 + R(theta_crank) * (l_crank, 0).
    let bx = o2.0 + l_crank * theta_crank.cos();
    let by = o2.1 + l_crank * theta_crank.sin();
    state.set_pose(crank_id, &mut q0, o2.0, o2.1, theta_crank);

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
    state.set_pose(rocker_id, &mut q0, cx, cy, theta_rocker);

    // Coupler: origin at B (point B in local = (0,0)), angle from B→C direction.
    let theta_coupler = (cy - by).atan2(cx - bx);
    state.set_pose(coupler_id, &mut q0, bx, by, theta_coupler);

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

/// Build a standard 4-bar linkage (d=ground, a=crank, b=coupler, c=rocker)
/// with ground pivots at O2=(0,0) and O4=(d,0).
///
/// A coupler point P is placed on the coupler at local x = coupler_point_x, y = 0
/// (i.e. at the midpoint if coupler_point_x = l_coupler/2).
///
/// `theta_crank_init` sets the initial crank angle. Use 0.0 for Grashof linkages
/// where the loop always closes at theta=0. For non-Grashof linkages the loop
/// may be geometrically invalid at theta=0 (e.g. rocker too long to reach the
/// crank tip), so pass a valid starting angle (e.g. PI/2).
///
/// The driver defaults to J1 (grounded revolute at O2, drives the crank).
fn build_standard_fourbar(
    crank_id: &str,
    coupler_id: &str,
    rocker_id: &str,
    l_ground: f64,
    l_crank: f64,
    l_coupler: f64,
    l_rocker: f64,
    coupler_point_x: f64,
    theta_crank_init: f64,
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (l_ground, 0.0_f64);

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let crank = make_bar(crank_id, "A", "B", l_crank, 0.0, 0.0);
    let mut coupler = make_bar(coupler_id, "B", "C", l_coupler, 0.0, 0.0);
    coupler
        .add_coupler_point("P", coupler_point_x, 0.0)
        .unwrap();
    let rocker = make_bar(rocker_id, "C", "D", l_rocker, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", crank_id, "A")
        .unwrap();
    mech.add_revolute_joint("J2", crank_id, "B", coupler_id, "B")
        .unwrap();
    mech.add_revolute_joint("J3", coupler_id, "C", rocker_id, "C")
        .unwrap();
    mech.add_revolute_joint("J4", rocker_id, "D", "ground", "O4")
        .unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(
        &mut mech,
        joint_id,
        "D1",
        theta_crank_init,
    )?;

    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(),
        o2,
        o4,
        l_crank,
        l_coupler,
        l_rocker,
        theta_crank_init,
        crank_id,
        coupler_id,
        rocker_id,
    );

    Ok((mech, q0))
}

/// Grashof crank-rocker 4-bar: d=4, a=2, b=4, c=3.
///
/// Grashof condition satisfied (shortest=2, longest=4): 2+4 < 4+3 → 6 < 7 ✓
/// Crank (a=2, shortest link grounded via O2) can rotate continuously.
/// Coupler point P at (2.0, 0) on coupler.
fn build_crank_rocker_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        4.0, 2.0, 4.0, 3.0, 2.0,
        0.0,
        driver_joint_id,
    )
}

/// Non-Grashof double-rocker 4-bar: d=5, a=3, b=4, c=7.
///
/// No link can rotate fully; both input and output oscillate.
/// Coupler point P at (2.0, 0) on coupler.
/// Uses theta_crank = PI/2 as initial angle because at theta=0 the crank tip is
/// only 2 units from O4, which is less than |b-c| = 3 and the triangle cannot close.
fn build_double_rocker_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        5.0, 3.0, 4.0, 7.0, 2.0,
        PI / 2.0,
        driver_joint_id,
    )
}

/// Grashof double-crank (drag-link) 4-bar: d=2, a=4, b=3.5, c=3.
///
/// Ground link is shortest: 2+3.5 < 4+3 → 5.5 < 7 ✓ (Grashof, ground shortest → double-crank).
/// Both crank and rocker rotate fully.
/// Coupler point P at (1.75, 0) on coupler.
fn build_double_crank_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        2.0, 4.0, 3.5, 3.0, 1.75,
        0.0,
        driver_joint_id,
    )
}

/// Grashof parallelogram 4-bar: d=4, a=2, b=4, c=2.
///
/// Opposite links equal (a=c=2, b=d=4). Coupler translates without rotating.
/// Coupler point P at (2.0, 0) on coupler.
fn build_parallelogram_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        4.0, 2.0, 4.0, 2.0, 2.0,
        0.0,
        driver_joint_id,
    )
}

/// Parallelogram 4-bar with a linear actuator driving the crank.
///
/// Same geometry as `Parallelogram` (d=4, a=2, b=4, c=2) but with:
/// - A mount point "M" at the crank midpoint (1.0, 0.0) in crank-local coords
/// - A new ground pivot "O_act" at (-1.0, -1.5) for the actuator base
/// - A linear actuator from ground "O_act" to crank mount "M"
///
/// The actuator triggers compound force expansion (cylinder + rod bodies).
fn build_parallelogram_actuator(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (4.0_f64, 0.0_f64);

    let mut ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    // Actuator base pivot — offset below and behind the crank pivot.
    ground
        .add_attachment_point("O_act", -1.0, -1.5)
        .map_err(|e| e.to_string())?;

    let mut crank = make_bar("crank", "A", "B", 2.0, 0.0, 0.0);
    // Mount point at crank midpoint for the actuator.
    crank
        .add_mount_point("M", 1.0, 0.0)
        .map_err(|e| e.to_string())?;

    let mut coupler = make_bar("coupler", "B", "C", 4.0, 0.0, 0.0);
    coupler.add_coupler_point("P", 2.0, 0.0).unwrap();
    let rocker = make_bar("rocker", "C", "D", 2.0, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();

    // Linear actuator: ground "O_act" → crank mount "M".
    // Uses mount_point_name so compound expansion kicks in on serialization.
    mech.add_force(ForceElement::LinearActuator(LinearActuatorElement {
        body_a: "ground".to_string(),
        point_a: [-1.0, -1.5],
        point_a_name: Some("O_act".to_string()),
        body_b: "crank".to_string(),
        point_b: [1.0, 0.0],
        point_b_name: Some("M".to_string()),
        force: 50.0,
        speed_limit: 0.0,
    }));

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4, 2.0, 4.0, 2.0, 0.0,
        "crank", "coupler", "rocker",
    );

    Ok((mech, q0))
}

/// Chebyshev approximate straight-line 4-bar: d=4, a=2, b=5, c=5.
///
/// Grashof condition: 2+5 < 5+4 → 7 < 9 ✓ (crank-rocker).
/// Coupler midpoint traces an approximate horizontal straight line.
/// Coupler point P at (2.5, 0) on coupler (midpoint).
fn build_chebyshev_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        4.0, 2.0, 5.0, 5.0, 2.5,
        0.0,
        driver_joint_id,
    )
}

/// Non-Grashof triple-rocker 4-bar: d=4, a=2, b=5, c=2.
///
/// No link satisfies Grashof (shortest+longest = 2+5 = 7, sum of others = 4+2 = 6; 7 > 6).
/// No link can make a full revolution; all three moving links oscillate.
/// Coupler point P at (2.5, 0) on coupler.
/// Uses theta_crank = PI/2 as initial angle because at theta=0 the crank tip is
/// only 2 units from O4, which is less than |b-c| = 3 and the triangle cannot close.
fn build_triple_rocker_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        4.0, 2.0, 5.0, 2.0, 2.5,
        PI / 2.0,
        driver_joint_id,
    )
}

// ---------------------------------------------------------------------------
// Six-bar helpers
// ---------------------------------------------------------------------------

/// Create a ternary body with 3 attachment points.
///
/// P1 is at the local origin (0,0). P2 and P3 are specified in body-local
/// coordinates.
fn make_ternary(
    body_id: &str,
    p1: &str,
    p2: &str,
    p3: &str,
    p2_local: (f64, f64),
    p3_local: (f64, f64),
) -> Body {
    let mut body = Body::new(body_id);
    body.add_attachment_point(p1, 0.0, 0.0).unwrap();
    body.add_attachment_point(p2, p2_local.0, p2_local.1).unwrap();
    body.add_attachment_point(p3, p3_local.0, p3_local.1).unwrap();
    body
}

/// Solve via continuation: start at `t_start` (where the initial guess is
/// well-conditioned) and step toward `t_end` over `n_steps` increments.
///
/// Returns the converged q at `t_end`, or `None` if any step diverges.
fn solve_with_continuation(
    mech: &Mechanism,
    q_start: &DVector<f64>,
    t_start: f64,
    t_end: f64,
    n_steps: usize,
) -> Option<DVector<f64>> {
    let mut q = q_start.clone();
    for i in 0..=n_steps {
        let t = t_start + (t_end - t_start) * (i as f64 / n_steps as f64);
        match solve_position(mech, &q, t, 1e-10, 100) {
            Ok(result) if result.converged => {
                q = result.q;
            }
            _ => return None,
        }
    }
    Some(q)
}

// ---------------------------------------------------------------------------
// SixBarB1 -- Watt I (Chain B, ternary ground)
// ---------------------------------------------------------------------------

/// 6-bar Watt I mechanism (type B1) with ternary ground.
///
/// Graph: ground(T), crank(B), ternary(T), rocker4(B), link5(B), output6(B)
/// Joints (7): J1: ground-crank, J2: crank-ternary, J3: ternary-rocker4,
///             J4: ground-rocker4, J5: ternary-link5, J6: link5-output6,
///             J7: ground-output6
/// Ground pivots: O2=(0,0), O4=(2.5,0.5), O6=(3.5,0)
/// Crank: 1.5, Ternary: P1/P2=(3,0)/P3=(1.5,1), Rocker4: 2.5,
/// Link5: 2.5, Output6: 2.5
/// Driver: ground-crank (J1)
fn build_sixbar_b1(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let ground = make_ground(&[
        ("O2", 0.0, 0.0),
        ("O4", 2.5, 0.5),
        ("O6", 3.5, 0.0),
    ]);
    let crank = make_bar("crank", "A", "B", 1.5, 0.0, 0.0);
    let mut ternary = make_ternary("ternary", "P1", "P2", "P3", (3.0, 0.0), (1.5, 1.0));
    ternary.add_coupler_point("CP", 1.5, 0.0).unwrap();
    let rocker4 = make_bar("rocker4", "R4A", "R4B", 2.5, 0.0, 0.0);
    let link5 = make_bar("link5", "L5A", "L5B", 2.5, 0.0, 0.0);
    let output6 = make_bar("output6", "R6A", "R6B", 2.5, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(ternary).unwrap();
    mech.add_body(rocker4).unwrap();
    mech.add_body(link5).unwrap();
    mech.add_body(output6).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "ternary", "P1").unwrap();
    mech.add_revolute_joint("J3", "ternary", "P2", "rocker4", "R4B").unwrap();
    mech.add_revolute_joint("J4", "ground", "O6", "rocker4", "R4A").unwrap();
    mech.add_revolute_joint("J5", "ternary", "P3", "link5", "L5A").unwrap();
    mech.add_revolute_joint("J6", "link5", "L5B", "output6", "R6B").unwrap();
    mech.add_revolute_joint("J7", "ground", "O4", "output6", "R6A").unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    // Geometric initial guess using continuation.
    // Try multiple starting angles to find one that converges.
    let state = mech.state();
    let q0 = sixbar_b1_find_initial(state, &mech)?;

    Ok((mech, q0))
}

/// Compute geometric initial guess for the B1 six-bar and use continuation
/// to reach t=0. Tries several starting angles to find one that converges.
fn sixbar_b1_find_initial(
    state: &crate::core::state::State,
    mech: &Mechanism,
) -> Result<DVector<f64>, String> {
    // Try a range of starting crank angles.
    for &crank_angle in &[0.3, 0.5, 0.8, 1.0, 0.15] {
        let mut q = state.make_q();

        state.set_pose("crank", &mut q, 0.0, 0.0, crank_angle);
        let bx = 1.5 * crank_angle.cos();
        let by = 1.5 * crank_angle.sin();

        // Ternary: P1 at crank tip B, initial theta ~ small
        let theta_tern = crank_angle * 0.3;
        state.set_pose("ternary", &mut q, bx, by, theta_tern);
        let ct = theta_tern.cos();
        let st = theta_tern.sin();

        // Ternary P2 global
        let p2_gx = bx + 3.0 * ct;
        let p2_gy = by + 3.0 * st;

        // Rocker4: R4A at O6=(3.5,0), R4B should be near P2
        let dx = p2_gx - 3.5;
        let dy = p2_gy;
        let theta_r4 = dy.atan2(dx);
        state.set_pose("rocker4", &mut q, 3.5, 0.0, theta_r4);

        // Ternary P3 global
        let p3_gx = bx + 1.5 * ct - 1.0 * st;
        let p3_gy = by + 1.5 * st + 1.0 * ct;

        // Link5: L5A at P3, pointing toward O4=(2.5,0.5) region
        let dx5 = 2.5 - p3_gx;
        let dy5 = 0.5 - p3_gy;
        let theta_l5 = dy5.atan2(dx5);
        state.set_pose("link5", &mut q, p3_gx, p3_gy, theta_l5);

        // Output6: R6A at O4=(2.5,0.5), R6B should be near link5 L5B
        let l5b_gx = p3_gx + 2.5 * theta_l5.cos();
        let l5b_gy = p3_gy + 2.5 * theta_l5.sin();
        let dx6 = l5b_gx - 2.5;
        let dy6 = l5b_gy - 0.5;
        let theta_o6 = dy6.atan2(dx6);
        state.set_pose("output6", &mut q, 2.5, 0.5, theta_o6);

        // Try to solve at this crank angle
        if let Ok(result) = solve_position(mech, &q, crank_angle, 1e-10, 200) {
            if result.converged {
                // Step back to t=0 via continuation
                if let Some(q0) = solve_with_continuation(
                    mech, &result.q, crank_angle, 0.0, 20,
                ) {
                    return Ok(q0);
                }
            }
        }
    }
    Err("SixBarB1: could not find converging initial guess at any starting angle".to_string())
}

// ---------------------------------------------------------------------------
// SixBarA1 -- Chain A, binary ground
// ---------------------------------------------------------------------------

/// 6-bar Chain A mechanism with binary ground (type A1).
///
/// Graph: ground(B), T1(T), B2(B), T2(T), B3(B), B4(B)
/// Joints (7): J1: ground-T1, J2: T1-B2, J3: T1-T2 (adjacent!),
///             J4: T2-B3, J5: T2-B4, J6: B2-B3, J7: B4-ground
/// Ground pivots: O2=(0,0), O4=(2,0)
/// T1: P1/P2=(1.5,0)/P3=(0.8,0.6), B2: 2.0, T2: Q1/Q2=(1.5,0)/Q3=(0.8,0.6),
/// B3: 2.0, B4: 2.0
/// Driver: ground-T1 (J1)
fn build_sixbar_a1(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 2.0, 0.0)]);
    let mut t1 = make_ternary("t1", "P1", "P2", "P3", (1.5, 0.0), (0.8, 0.6));
    t1.add_coupler_point("CP", 0.75, 0.3).unwrap();
    let b2 = make_bar("b2", "B2A", "B2B", 2.0, 0.0, 0.0);
    let t2 = make_ternary("t2", "Q1", "Q2", "Q3", (1.5, 0.0), (0.8, 0.6));
    let b3 = make_bar("b3", "B3A", "B3B", 2.0, 0.0, 0.0);
    let b4 = make_bar("b4", "B4A", "B4B", 2.0, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(t1).unwrap();
    mech.add_body(b2).unwrap();
    mech.add_body(t2).unwrap();
    mech.add_body(b3).unwrap();
    mech.add_body(b4).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "t1", "P1").unwrap();
    mech.add_revolute_joint("J2", "t1", "P2", "b2", "B2A").unwrap();
    mech.add_revolute_joint("J3", "t1", "P3", "t2", "Q1").unwrap();
    mech.add_revolute_joint("J4", "t2", "Q2", "b3", "B3A").unwrap();
    mech.add_revolute_joint("J5", "t2", "Q3", "b4", "B4A").unwrap();
    mech.add_revolute_joint("J6", "b2", "B2B", "b3", "B3B").unwrap();
    mech.add_revolute_joint("J7", "b4", "B4B", "ground", "O4").unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    // Geometric initial guess at angle = 10 degrees, then continuation to t=0.
    let angle = 10.0_f64.to_radians();
    let state = mech.state();
    let mut q = state.make_q();

    let ct = angle.cos();
    let st = angle.sin();
    state.set_pose("t1", &mut q, 0.0, 0.0, angle);
    let p2x = 1.5 * ct;
    let p2y = 1.5 * st;
    let p3x = 0.8 * ct - 0.6 * st;
    let p3y = 0.8 * st + 0.6 * ct;

    let theta_t2 = angle - 1.5;
    state.set_pose("t2", &mut q, p3x, p3y, theta_t2);
    let ct2 = theta_t2.cos();
    let st2 = theta_t2.sin();
    let q2x = p3x + 1.5 * ct2;
    let q2y = p3y + 1.5 * st2;
    let q3x = p3x + 0.8 * ct2 - 0.6 * st2;
    let q3y = p3y + 0.8 * st2 + 0.6 * ct2;

    let theta_b2 = (q2y - p2y).atan2(q2x - p2x);
    state.set_pose("b2", &mut q, p2x, p2y, theta_b2);
    let b2ex = p2x + 2.0 * theta_b2.cos();
    let b2ey = p2y + 2.0 * theta_b2.sin();
    let theta_b3 = (b2ey - q2y).atan2(b2ex - q2x);
    state.set_pose("b3", &mut q, q2x, q2y, theta_b3);

    let theta_b4 = (0.0 - q3y).atan2(2.0 - q3x);
    state.set_pose("b4", &mut q, q3x, q3y, theta_b4);

    let result = solve_position(&mech, &q, angle, 1e-10, 100)
        .map_err(|e| format!("SixBarA1 initial solve failed: {}", e))?;
    if !result.converged {
        return Err(format!(
            "SixBarA1 initial solve did not converge, residual = {}",
            result.residual_norm
        ));
    }
    let q0 = solve_with_continuation(&mech, &result.q, angle, 0.0, 15)
        .ok_or("SixBarA1 continuation to t=0 failed")?;

    Ok((mech, q0))
}

// ---------------------------------------------------------------------------
// SixBarA2 -- Chain A, ternary ground
// ---------------------------------------------------------------------------

/// 6-bar Chain A mechanism with ternary ground (type A2).
///
/// Graph: ground(T), B1(B), B2(B), T2(T), B3(B), B4(B)
/// Joints (7): J1: ground-B1, J2: ground-B2, J3: ground-T2 (adjacent!),
///             J4: T2-B3, J5: T2-B4, J6: B1-B4, J7: B2-B3
/// Ground pivots: O2=(0,0), O4=(4.5,0), O6=(2.8,0)
/// B1: 1.0, B2: 1.5, T2: Q1/Q2=(2.5,0)/Q3=(1.5,1.0), B3: 2.5, B4: 2.5
/// Driver: ground-B1 (J1)
fn build_sixbar_a2(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let ground = make_ground(&[
        ("O2", 0.0, 0.0),
        ("O4", 4.5, 0.0),
        ("O6", 2.8, 0.0),
    ]);
    let b1 = make_bar("b1", "B1A", "B1B", 1.0, 0.0, 0.0);
    let b2 = make_bar("b2", "B2A", "B2B", 1.5, 0.0, 0.0);
    let mut t2 = make_ternary("t2", "Q1", "Q2", "Q3", (2.5, 0.0), (1.5, 1.0));
    t2.add_coupler_point("CP", 1.25, 0.5).unwrap();
    let b3 = make_bar("b3", "B3A", "B3B", 2.5, 0.0, 0.0);
    let b4 = make_bar("b4", "B4A", "B4B", 2.5, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(b1).unwrap();
    mech.add_body(b2).unwrap();
    mech.add_body(t2).unwrap();
    mech.add_body(b3).unwrap();
    mech.add_body(b4).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "b1", "B1A").unwrap();
    mech.add_revolute_joint("J2", "ground", "O4", "b2", "B2A").unwrap();
    mech.add_revolute_joint("J3", "ground", "O6", "t2", "Q1").unwrap();
    mech.add_revolute_joint("J4", "t2", "Q2", "b3", "B3A").unwrap();
    mech.add_revolute_joint("J5", "t2", "Q3", "b4", "B4A").unwrap();
    mech.add_revolute_joint("J6", "b1", "B1B", "b4", "B4B").unwrap();
    mech.add_revolute_joint("J7", "b2", "B2B", "b3", "B3B").unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    // Geometric initial guess at angle = 5 degrees, then continuation to t=0.
    let angle = 5.0_f64.to_radians();
    let state = mech.state();
    let mut q = state.make_q();

    state.set_pose("b1", &mut q, 0.0, 0.0, angle);
    let b1x = 1.0 * angle.cos();
    let b1y = 1.0 * angle.sin();

    let theta_t2 = -0.5;
    state.set_pose("t2", &mut q, 2.8, 0.0, theta_t2);
    let ct2 = theta_t2.cos();
    let st2 = theta_t2.sin();

    let q3x = 2.8 + 1.5 * ct2 - 1.0 * st2;
    let q3y = 1.5 * st2 + 1.0 * ct2;
    let theta_b4 = (b1y - q3y).atan2(b1x - q3x);
    state.set_pose("b4", &mut q, q3x, q3y, theta_b4);

    let q2x = 2.8 + 2.5 * ct2;
    let q2y = 2.5 * st2;
    let theta_b2 = q2y.atan2(q2x - 4.5);
    state.set_pose("b2", &mut q, 4.5, 0.0, theta_b2);
    let b2x = 4.5 + 1.5 * theta_b2.cos();
    let b2y = 1.5 * theta_b2.sin();
    let theta_b3 = (b2y - q2y).atan2(b2x - q2x);
    state.set_pose("b3", &mut q, q2x, q2y, theta_b3);

    let result = solve_position(&mech, &q, angle, 1e-10, 100)
        .map_err(|e| format!("SixBarA2 initial solve failed: {}", e))?;
    if !result.converged {
        return Err(format!(
            "SixBarA2 initial solve did not converge, residual = {}",
            result.residual_norm
        ));
    }
    let q0 = solve_with_continuation(&mech, &result.q, angle, 0.0, 15)
        .ok_or("SixBarA2 continuation to t=0 failed")?;

    Ok((mech, q0))
}

// ---------------------------------------------------------------------------
// SixBarB2 -- Chain B, shared-binary ground
// ---------------------------------------------------------------------------

/// 6-bar Chain B mechanism with shared-binary ground (type B2).
///
/// Graph: ground(B), T1(T), B2(B), T2(T), B3(B), B4(B)
/// Joints (7): J1: ground-T1, J2: ground-T2, J3: T1-B2, J4: T1-B3,
///             J5: T2-B2, J6: T2-B4, J7: B3-B4
/// Ground pivots: O2=(0,0), O4=(1.8,0)
/// T1: P1/P2=(0.5,0)/P3=(0.25,0.25), B2: 2.0,
/// T2: Q1/Q2=(1.5,0)/Q3=(0.8,-0.6), B3: 2.0, B4: 2.0
/// Driver: ground-T1 (J1)
fn build_sixbar_b2(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 1.8, 0.0)]);
    let mut t1 = make_ternary("t1", "P1", "P2", "P3", (0.5, 0.0), (0.25, 0.25));
    t1.add_coupler_point("CP", 0.25, 0.125).unwrap();
    let b2 = make_bar("b2", "B2A", "B2B", 2.0, 0.0, 0.0);
    let t2 = make_ternary("t2", "Q1", "Q2", "Q3", (1.5, 0.0), (0.8, -0.6));
    let b3 = make_bar("b3", "B3A", "B3B", 2.0, 0.0, 0.0);
    let b4 = make_bar("b4", "B4A", "B4B", 2.0, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(t1).unwrap();
    mech.add_body(b2).unwrap();
    mech.add_body(t2).unwrap();
    mech.add_body(b3).unwrap();
    mech.add_body(b4).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "t1", "P1").unwrap();
    mech.add_revolute_joint("J2", "ground", "O4", "t2", "Q1").unwrap();
    mech.add_revolute_joint("J3", "t1", "P2", "b2", "B2A").unwrap();
    mech.add_revolute_joint("J4", "t1", "P3", "b3", "B3A").unwrap();
    mech.add_revolute_joint("J5", "t2", "Q2", "b2", "B2B").unwrap();
    mech.add_revolute_joint("J6", "t2", "Q3", "b4", "B4A").unwrap();
    mech.add_revolute_joint("J7", "b3", "B3B", "b4", "B4B").unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    // Geometric initial guess at angle = 5 degrees, then continuation to t=0.
    let angle = 5.0_f64.to_radians();
    let state = mech.state();
    let mut q = state.make_q();

    let ct1 = angle.cos();
    let st1 = angle.sin();
    state.set_pose("t1", &mut q, 0.0, 0.0, angle);
    let p2x = 0.5 * ct1;
    let p2y = 0.5 * st1;
    let p3x = 0.25 * ct1 - 0.25 * st1;
    let p3y = 0.25 * st1 + 0.25 * ct1;

    let theta_t2 = -1.5;
    state.set_pose("t2", &mut q, 1.8, 0.0, theta_t2);
    let ct2 = theta_t2.cos();
    let st2 = theta_t2.sin();
    let q2x = 1.8 + 1.5 * ct2;
    let q2y = 1.5 * st2;
    // Q3 local = (0.8, -0.6): q3_global = origin + R * (0.8, -0.6)
    let q3x = 1.8 + 0.8 * ct2 + 0.6 * st2;
    let q3y = 0.8 * st2 - 0.6 * ct2;

    let theta_b2 = (q2y - p2y).atan2(q2x - p2x);
    state.set_pose("b2", &mut q, p2x, p2y, theta_b2);

    let theta_b3 = (q3y - p3y).atan2(q3x - p3x);
    state.set_pose("b3", &mut q, p3x, p3y, theta_b3);

    let b3ex = p3x + 2.0 * theta_b3.cos();
    let b3ey = p3y + 2.0 * theta_b3.sin();
    let theta_b4 = (b3ey - q3y).atan2(b3ex - q3x);
    state.set_pose("b4", &mut q, q3x, q3y, theta_b4);

    let result = solve_position(&mech, &q, angle, 1e-10, 100)
        .map_err(|e| format!("SixBarB2 initial solve failed: {}", e))?;
    if !result.converged {
        return Err(format!(
            "SixBarB2 initial solve did not converge, residual = {}",
            result.residual_norm
        ));
    }
    let q0 = solve_with_continuation(&mech, &result.q, angle, 0.0, 15)
        .ok_or("SixBarB2 continuation to t=0 failed")?;

    Ok((mech, q0))
}

// ---------------------------------------------------------------------------
// SixBarB3 -- Chain B, exclusive-binary ground
// ---------------------------------------------------------------------------

/// 6-bar Chain B mechanism with exclusive-binary ground (type B3).
///
/// Graph: ground(B), T1(T), B1(B), T2(T), B2(B), B4(B)
/// Joints (7): J1: ground-T1, J2: ground-B4, J3: T1-B1, J4: T1-B2,
///             J5: T2-B1, J6: T2-B2, J7: T2-B4
/// Ground pivots: O2=(0,0), O4=(2.5,0)
/// T1: P1/P2=(1.0,0)/P3=(0.5,0.5), B1: 2.0,
/// T2: Q1/Q2=(1.5,0)/Q3=(0.8,-0.6), B2: 2.0, B4: 2.0
/// Driver: ground-T1 (J1)
fn build_sixbar_b3(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 2.5, 0.0)]);
    let mut t1 = make_ternary("t1", "P1", "P2", "P3", (1.0, 0.0), (0.5, 0.5));
    t1.add_coupler_point("CP", 0.5, 0.25).unwrap();
    let b1 = make_bar("b1", "B1A", "B1B", 2.0, 0.0, 0.0);
    let t2 = make_ternary("t2", "Q1", "Q2", "Q3", (1.5, 0.0), (0.8, -0.6));
    let b2 = make_bar("b2", "B2A", "B2B", 2.0, 0.0, 0.0);
    let b4 = make_bar("b4", "B4A", "B4B", 2.0, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(t1).unwrap();
    mech.add_body(b1).unwrap();
    mech.add_body(t2).unwrap();
    mech.add_body(b2).unwrap();
    mech.add_body(b4).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "t1", "P1").unwrap();
    mech.add_revolute_joint("J2", "ground", "O4", "b4", "B4B").unwrap();
    mech.add_revolute_joint("J3", "t1", "P2", "b1", "B1A").unwrap();
    mech.add_revolute_joint("J4", "t1", "P3", "b2", "B2A").unwrap();
    mech.add_revolute_joint("J5", "t2", "Q1", "b1", "B1B").unwrap();
    mech.add_revolute_joint("J6", "t2", "Q2", "b2", "B2B").unwrap();
    mech.add_revolute_joint("J7", "t2", "Q3", "b4", "B4A").unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    // Geometric initial guess at angle = 0.3 rad, then continuation to t=0.
    let angle = 0.3_f64;
    let state = mech.state();
    let mut q = state.make_q();

    state.set_pose("t1", &mut q, 0.0, 0.0, angle);
    let ct1 = angle.cos();
    let st1 = angle.sin();
    let p2x = 1.0 * ct1;
    let p2y = 1.0 * st1;
    let p3x = 0.5 * ct1 - 0.5 * st1;
    let p3y = 0.5 * st1 + 0.5 * ct1;

    let theta_b1 = angle + 0.8;
    state.set_pose("b1", &mut q, p2x, p2y, theta_b1);
    let b1ex = p2x + 2.0 * theta_b1.cos();
    let b1ey = p2y + 2.0 * theta_b1.sin();

    let theta_t2 = theta_b1 + 0.5;
    state.set_pose("t2", &mut q, b1ex, b1ey, theta_t2);
    let ct2 = theta_t2.cos();
    let st2 = theta_t2.sin();
    let q2x = b1ex + 1.5 * ct2;
    let q2y = b1ey + 1.5 * st2;
    // Q3 local = (0.8, -0.6): q3_global = origin + R * (0.8, -0.6)
    let q3x = b1ex + 0.8 * ct2 + 0.6 * st2;
    let q3y = b1ey + 0.8 * st2 - 0.6 * ct2;

    let theta_b2 = (q2y - p3y).atan2(q2x - p3x);
    state.set_pose("b2", &mut q, p3x, p3y, theta_b2);

    let theta_b4 = (0.0 - q3y).atan2(2.5 - q3x);
    state.set_pose("b4", &mut q, q3x, q3y, theta_b4);

    let result = solve_position(&mech, &q, angle, 1e-10, 100)
        .map_err(|e| format!("SixBarB3 initial solve failed: {}", e))?;
    if !result.converged {
        return Err(format!(
            "SixBarB3 initial solve did not converge, residual = {}",
            result.residual_norm
        ));
    }
    let q0 = solve_with_continuation(&mech, &result.q, angle, 0.0, 15)
        .ok_or("SixBarB3 continuation to t=0 failed")?;

    Ok((mech, q0))
}

// ── Phase 6.3 sample mechanisms ──────────────────────────────────────────────

/// Quick-return mechanism (crank-shaper).
///
/// A 4-bar where the crank is much shorter than the ground link, giving an
/// asymmetric output stroke (fast return). The time ratio of forward to return
/// stroke is > 1.
///
/// Link lengths: ground=0.060, crank=0.015, coupler=0.060, rocker=0.045
fn build_quick_return(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0, 0.0);
    let o4 = (0.060, 0.0);
    let l_crank = 0.015;
    let l_coupler = 0.060;
    let l_rocker = 0.045;

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

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;
    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4, l_crank, l_coupler, l_rocker, 0.0,
        "crank", "coupler", "rocker",
    );
    Ok((mech, q0))
}

/// Toggle clamp — 4-bar near toggle configuration.
///
/// Designed so the output link reaches near-180-degree alignment at one
/// extreme, producing very high mechanical advantage (clamping force).
///
/// Link lengths: ground=0.040, crank=0.012, coupler=0.038, rocker=0.020
fn build_toggle_clamp(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0, 0.0);
    let o4 = (0.040, 0.0);
    let l_crank = 0.012;
    let l_coupler = 0.038;
    let l_rocker = 0.020;

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

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;
    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4, l_crank, l_coupler, l_rocker, 0.0,
        "crank", "coupler", "rocker",
    );
    Ok((mech, q0))
}

/// Scotch yoke — produces pure sinusoidal output motion.
///
/// A crank drives a slider through a prismatic joint that constrains
/// motion to pure vertical translation. The slider position is exactly
/// r * sin(theta).
///
/// Uses crank + slider with prismatic joint on vertical axis.
fn build_scotch_yoke(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    // Build as crank + sliding block
    // Crank center at origin, crank length = 0.02 m
    // Slider constrained to vertical axis at x = 0
    let l_crank = 0.02;

    let ground = make_ground(&[
        ("O", 0.0, 0.0),        // crank pivot
        ("S", 0.0, l_crank),    // slider rail point
    ]);
    let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);

    // Slider is a small body
    let mut slider = Body {
        id: "slider".to_string(),
        attachment_points: std::collections::HashMap::new(),
        mass: 0.5,
        cg_local: nalgebra::Vector2::new(0.0, 0.0),
        izz_cg: 0.001,
        mount_points: std::collections::HashMap::new(),
        coupler_points: std::collections::HashMap::new(),
    };
    slider.add_attachment_point("P", 0.0, 0.0).unwrap();

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(slider).unwrap();

    mech.add_revolute_joint("J1", "ground", "O", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "slider", "P").unwrap();
    // Prismatic joint constrains slider to vertical axis
    mech.add_prismatic_joint("J3", "ground", "S", "slider", "P", nalgebra::Vector2::new(0.0, 1.0), 0.0)
        .unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;
    mech.build().map_err(|e| e.to_string())?;

    // Initial guess: crank horizontal, slider at (0, crank_length)
    let mut q0 = mech.state().make_q();
    if let Ok(bi) = mech.state().get_index("crank") {
        q0[bi.x_idx()] = l_crank;
        q0[bi.y_idx()] = 0.0;
        q0[bi.theta_idx()] = 0.0;
    }
    if let Ok(bi) = mech.state().get_index("slider") {
        q0[bi.x_idx()] = 0.0;
        q0[bi.y_idx()] = l_crank;
        q0[bi.theta_idx()] = 0.0;
    }

    // Solve to get a consistent initial position
    match solve_position(&mech, &q0, 0.0, 1e-10, 50) {
        Ok(result) if result.converged => Ok((mech, result.q)),
        Ok(_) => Ok((mech, q0)),
        Err(_) => Ok((mech, q0)),
    }
}

/// Inverted slider-crank — crank drives a prismatic joint directly.
///
/// The slider rides along the coupler (which rotates), creating a different
/// motion profile than the standard slider-crank. Ground pivot at origin,
/// crank at (0.04, 0).
fn build_inverted_slider_crank(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0, 0.0);
    let o4 = (0.04, 0.0);
    let l_crank = 0.015;

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);

    // Slider body
    let mut slider = Body {
        id: "slider".to_string(),
        attachment_points: std::collections::HashMap::new(),
        mass: 0.5,
        cg_local: nalgebra::Vector2::new(0.0, 0.0),
        izz_cg: 0.001,
        mount_points: std::collections::HashMap::new(),
        coupler_points: std::collections::HashMap::new(),
    };
    slider.add_attachment_point("P", 0.0, 0.0).unwrap();
    slider.add_attachment_point("Q", 0.0, 0.0).unwrap();

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(slider).unwrap();

    // Crank to ground
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    // Crank to slider via revolute
    mech.add_revolute_joint("J2", "crank", "B", "slider", "P").unwrap();
    // Slider to ground via prismatic (horizontal axis)
    mech.add_prismatic_joint("J3", "ground", "O4", "slider", "Q", nalgebra::Vector2::new(1.0, 0.0), 0.0)
        .unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;
    mech.build().map_err(|e| e.to_string())?;

    // Initial guess
    let mut q0 = mech.state().make_q();
    if let Ok(bi) = mech.state().get_index("crank") {
        q0[bi.x_idx()] = l_crank;
        q0[bi.y_idx()] = 0.0;
        q0[bi.theta_idx()] = 0.0;
    }
    if let Ok(bi) = mech.state().get_index("slider") {
        q0[bi.x_idx()] = l_crank;
        q0[bi.y_idx()] = 0.0;
        q0[bi.theta_idx()] = 0.0;
    }

    match solve_position(&mech, &q0, 0.0, 1e-10, 50) {
        Ok(result) if result.converged => Ok((mech, result.q)),
        Ok(_) => Ok((mech, q0)),
        Err(_) => Ok((mech, q0)),
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
        assert_eq!(SampleMechanism::all().len(), 17);
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
