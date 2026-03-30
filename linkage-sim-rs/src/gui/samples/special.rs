//! Special mechanism sample builders (quick-return, toggle clamp, scotch yoke, etc.).

use nalgebra::DVector;

use crate::core::body::{make_bar, make_ground, Body};
use crate::core::mechanism::Mechanism;
use crate::solver::kinematics::solve_position;

use super::helpers::{attach_driver_to_grounded_revolute_with_theta0, fourbar_initial_q0};

/// Quick-return mechanism (crank-shaper).
///
/// A 4-bar where the crank is much shorter than the ground link, giving an
/// asymmetric output stroke (fast return). The time ratio of forward to return
/// stroke is > 1.
///
/// Link lengths: ground=0.060, crank=0.015, coupler=0.060, rocker=0.045
pub(super) fn build_quick_return(
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
        "crank", "coupler", "rocker", false,
    );
    Ok((mech, q0))
}

/// Toggle clamp — 4-bar near toggle configuration.
///
/// Designed so the output link reaches near-180-degree alignment at one
/// extreme, producing very high mechanical advantage (clamping force).
///
/// Link lengths: ground=0.040, crank=0.012, coupler=0.038, rocker=0.020
pub(super) fn build_toggle_clamp(
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
        "crank", "coupler", "rocker", false,
    );
    Ok((mech, q0))
}

/// Scotch yoke — produces pure sinusoidal output motion.
///
/// A crank drives a slider through a yoke slot, constraining the slider
/// to pure vertical translation. The slider position is exactly
/// r * sin(theta).
///
/// Uses 3 moving bodies (crank + pin + slider) to avoid over-constraining:
///   - J1: revolute (ground → crank)          — crank pivots at origin
///   - J2: revolute (crank tip → pin)         — crank pin rides in yoke
///   - J3: prismatic (ground → slider, Y-axis) — slider on vertical rail
///   - J4: prismatic (pin → slider, X-axis)   — pin slides horizontally in yoke slot
///
/// DOF: 3×3 = 9, constraints: 2+2+2+2+1 = 9 → 0 DOF (correct).
pub(super) fn build_scotch_yoke(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let l_crank = 0.02;

    let ground = make_ground(&[
        ("O", 0.0, 0.0),       // crank pivot
        ("rail", 0.0, 0.0),    // slider rail anchor
    ]);
    let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);

    // Pin body: small intermediate body at crank tip, rides in the yoke slot
    let mut pin = Body::new("pin");
    pin.add_attachment_point("P", 0.0, 0.0).unwrap();

    // Slider body: translates vertically on the rail
    let mut slider = Body::new("slider");
    slider.add_attachment_point("S", 0.0, 0.0).unwrap();

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(pin).unwrap();
    mech.add_body(slider).unwrap();

    // J1: crank pivots on ground at origin
    mech.add_revolute_joint("J1", "ground", "O", "crank", "A").unwrap();
    // J2: crank tip connects to pin (allows rotation)
    mech.add_revolute_joint("J2", "crank", "B", "pin", "P").unwrap();
    // J3: slider constrained to vertical rail (Y-axis)
    mech.add_prismatic_joint(
        "J3", "ground", "rail", "slider", "S",
        nalgebra::Vector2::new(0.0, 1.0), 0.0,
    ).unwrap();
    // J4: pin slides horizontally in yoke slot on slider (X-axis)
    // body_i = pin, body_j = slider; axis in pin's local frame = X
    // Perpendicular constraint: pin_Y = slider_Y (same height)
    // Angle constraint: pin angle = slider angle = 0
    mech.add_prismatic_joint(
        "J4", "pin", "P", "slider", "S",
        nalgebra::Vector2::new(1.0, 0.0), 0.0,
    ).unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;
    mech.build().map_err(|e| e.to_string())?;

    // Initial guess: crank at theta=0 (horizontal right)
    // Crank body origin at (0,0), angle=0 → B at (l_crank, 0)
    // Pin at crank tip (l_crank, 0), angle=0
    // Slider at (0, 0) on the Y-axis; pin Y = slider Y = 0
    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.0, 0.0, 0.0);
    state.set_pose("pin", &mut q0, l_crank, 0.0, 0.0);
    state.set_pose("slider", &mut q0, 0.0, 0.0, 0.0);

    match solve_position(&mech, &q0, 0.0, 1e-10, 50) {
        Ok(result) if result.converged => Ok((mech, result.q)),
        Ok(_) => Ok((mech, q0)),
        Err(_) => Ok((mech, q0)),
    }
}

/// Inverted slider-crank — crank drives a slider along a rotating guide.
///
/// The crank pin rides on a guide (rocker) that pivots on a second ground
/// point, creating oscillating rotary output from rotary input.
///
/// Uses 3 moving bodies (crank + pin + guide) to avoid over-constraining:
///   - J1: revolute (ground O2 → crank)       — crank pivots at origin
///   - J2: revolute (crank tip → pin)         — crank pin connects to slider block
///   - J3: revolute (ground O4 → guide)       — guide pivots at second ground point
///   - J4: prismatic (guide → pin, along guide axis) — pin slides along guide
///
/// DOF: 3×3 = 9, constraints: 2+2+2+2+1 = 9 → 0 DOF (correct).
pub(super) fn build_inverted_slider_crank(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0, 0.0);
    let o4 = (0.04, 0.0);
    let l_crank = 0.015;
    let l_guide = 0.06; // guide long enough that the crank pin always reaches it

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);

    // Pin body: intermediate body at crank tip, slides along the guide
    let mut pin = Body::new("pin");
    pin.add_attachment_point("P", 0.0, 0.0).unwrap();

    // Guide (rocker): a bar that pivots at O4; crank pin slides along it
    let guide = make_bar("guide", "C", "D", l_guide, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(pin).unwrap();
    mech.add_body(guide).unwrap();

    // J1: crank pivots on ground at O2
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    // J2: crank tip connects to pin (allows rotation)
    mech.add_revolute_joint("J2", "crank", "B", "pin", "P").unwrap();
    // J3: guide pivots on ground at O4
    mech.add_revolute_joint("J3", "ground", "O4", "guide", "C").unwrap();
    // J4: pin slides along guide axis
    // body_i = guide (axis in guide's local frame = X = along the bar)
    // body_j = pin
    // Perpendicular constraint: pin stays on the guide line
    // Angle constraint: pin angle = guide angle (delta_theta_0 = 0)
    mech.add_prismatic_joint(
        "J4", "guide", "C", "pin", "P",
        nalgebra::Vector2::new(1.0, 0.0), 0.0,
    ).unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;
    mech.build().map_err(|e| e.to_string())?;

    // Initial guess: crank at theta=0 (horizontal right)
    // Crank: origin at O2=(0,0), angle=0 → B at (l_crank, 0)
    // Pin at crank tip (l_crank, 0)
    // Guide: origin at O4=(0.04, 0), must point toward the pin
    // Guide angle = atan2(pin_y - O4_y, pin_x - O4_x)
    //             = atan2(0 - 0, 0.015 - 0.04) = atan2(0, -0.025) = PI
    let pin_x = l_crank;
    let pin_y = 0.0;
    let guide_angle = (pin_y - o4.1).atan2(pin_x - o4.0);

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, o2.0, o2.1, 0.0);
    state.set_pose("pin", &mut q0, pin_x, pin_y, guide_angle);
    state.set_pose("guide", &mut q0, o4.0, o4.1, guide_angle);

    match solve_position(&mech, &q0, 0.0, 1e-10, 50) {
        Ok(result) if result.converged => Ok((mech, result.q)),
        Ok(_) => Ok((mech, q0)),
        Err(_) => Ok((mech, q0)),
    }
}
