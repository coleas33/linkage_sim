//! Special mechanism sample builders (quick-return, toggle clamp, scotch yoke, etc.).

use nalgebra::DVector;

use crate::core::body::{make_bar, make_ground, Body};
use crate::core::mechanism::Mechanism;
use crate::solver::kinematics::solve_position;

use super::helpers::{
    attach_driver_to_grounded_revolute_with_theta0, fourbar_initial_q0, make_ternary,
    set_bar_mass, set_slider_mass, set_ternary_mass, solve_with_continuation,
};

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
    let mut crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    let mut coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
    let mut rocker = make_bar("rocker", "C", "D", l_rocker, 0.0, 0.0);
    set_bar_mass(&mut crank, l_crank);
    set_bar_mass(&mut coupler, l_coupler);
    set_bar_mass(&mut rocker, l_rocker);

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
    let mut crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    let mut coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
    let mut rocker = make_bar("rocker", "C", "D", l_rocker, 0.0, 0.0);
    set_bar_mass(&mut crank, l_crank);
    set_bar_mass(&mut coupler, l_coupler);
    set_bar_mass(&mut rocker, l_rocker);

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
    let mut crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    set_bar_mass(&mut crank, l_crank);

    // Pin body: small intermediate body at crank tip, rides in the yoke slot
    let mut pin = Body::new("pin");
    pin.add_attachment_point("P", 0.0, 0.0).unwrap();
    set_slider_mass(&mut pin);

    // Slider body: translates vertically on the rail
    let mut slider = Body::new("slider");
    slider.add_attachment_point("S", 0.0, 0.0).unwrap();
    set_slider_mass(&mut slider);

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
    let mut crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    set_bar_mass(&mut crank, l_crank);

    // Pin body: intermediate body at crank tip, slides along the guide
    let mut pin = Body::new("pin");
    pin.add_attachment_point("P", 0.0, 0.0).unwrap();
    set_slider_mass(&mut pin);

    // Guide (rocker): a bar that pivots at O4; crank pin slides along it
    let mut guide = make_bar("guide", "C", "D", l_guide, 0.0, 0.0);
    set_bar_mass(&mut guide, l_guide);

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

/// Offset slider-crank mechanism.
///
/// Standard slider-crank with the slider rail offset from the crank center
/// by an eccentricity e. This creates asymmetric forward/return stroke timing,
/// unlike a centered slider-crank where strokes are symmetric.
///
/// Dimensions: crank=30mm, connecting rod=90mm, offset=15mm (slider rail at y=15mm).
///
/// Uses standard 4-body approach (ground + crank + coupler + slider):
///   - J1: revolute (ground O → crank)           -- crank pivots at origin
///   - J2: revolute (crank tip → coupler A)      -- connecting rod
///   - J3: revolute (coupler B → slider)         -- slider pin
///   - P1: prismatic (ground → slider, X-axis)   -- slider on offset horizontal rail
///
/// The rail anchor on ground is at (0, offset) to place the prismatic
/// constraint at the correct offset height.
pub(super) fn build_offset_slider_crank(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let l_crank = 0.030;
    let l_coupler = 0.090;
    let offset = 0.015;

    let ground = make_ground(&[
        ("O", 0.0, 0.0),         // crank pivot
        ("rail", 0.0, offset),    // slider rail anchor at offset height
    ]);
    let mut crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    let mut coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
    set_bar_mass(&mut crank, l_crank);
    set_bar_mass(&mut coupler, l_coupler);

    let mut slider = Body::new("slider");
    slider.add_attachment_point("C", 0.0, 0.0).unwrap();
    set_slider_mass(&mut slider);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(slider).unwrap();

    mech.add_revolute_joint("J1", "ground", "O", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "slider", "C").unwrap();
    // Prismatic: slider translates along X-axis at offset height
    mech.add_prismatic_joint(
        "P1", "ground", "rail", "slider", "C",
        nalgebra::Vector2::new(1.0, 0.0), 0.0,
    ).unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;
    mech.build().map_err(|e| e.to_string())?;

    // Initial guess: crank at theta=0 (horizontal right)
    // Crank tip B at (l_crank, 0) = (0.030, 0)
    // Coupler from B toward slider. Slider at y=offset, so coupler angles up.
    // Slider x = B.x + sqrt(l_coupler^2 - (offset - B.y)^2)
    let bx = l_crank;
    let by = 0.0;
    let dy = offset - by;
    let dx = (l_coupler * l_coupler - dy * dy).sqrt();
    let slider_x = bx + dx;
    let theta_coupler = dy.atan2(dx);

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.0, 0.0, 0.0);
    state.set_pose("coupler", &mut q0, bx, by, theta_coupler);
    state.set_pose("slider", &mut q0, slider_x, offset, 0.0);

    match solve_position(&mech, &q0, 0.0, 1e-10, 50) {
        Ok(result) if result.converged => Ok((mech, result.q)),
        Ok(_) => Ok((mech, q0)),
        Err(_) => Ok((mech, q0)),
    }
}

/// Whitworth quick-return mechanism.
///
/// A variant of the quick-return mechanism using a crank and slotted lever.
/// The crank drives a pin that slides in a slotted lever (rocker) pivoted
/// at a second ground point. The time ratio of forward to return stroke
/// is significantly > 1.
///
/// Uses the 3-body pin approach (like Scotch Yoke and Inverted Slider-Crank):
///   - J1: revolute (ground O2 → crank)       -- crank pivots at O2
///   - J2: revolute (crank tip → pin)         -- crank pin
///   - J3: revolute (ground O4 → lever)       -- lever pivots at O4
///   - J4: prismatic (lever → pin, along lever axis) -- pin slides in slot
///
/// Dimensions: crank=20mm, lever=60mm
/// Ground pivots: O2=(0,0), O4=(30mm, 0) -- O4 offset from O2.
pub(super) fn build_whitworth_quick_return(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0, 0.0);
    let o4 = (0.030, 0.0);
    let l_crank = 0.020;
    let l_lever = 0.060;

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let mut crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    set_bar_mass(&mut crank, l_crank);

    // Pin body: intermediate body at crank tip, slides along the lever slot
    let mut pin = Body::new("pin");
    pin.add_attachment_point("P", 0.0, 0.0).unwrap();
    set_slider_mass(&mut pin);

    // Lever (slotted rocker): pivots at O4, crank pin slides along it
    let mut lever = make_bar("lever", "C", "D", l_lever, 0.0, 0.0);
    set_bar_mass(&mut lever, l_lever);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(pin).unwrap();
    mech.add_body(lever).unwrap();

    // J1: crank pivots on ground at O2
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    // J2: crank tip connects to pin
    mech.add_revolute_joint("J2", "crank", "B", "pin", "P").unwrap();
    // J3: lever pivots on ground at O4
    mech.add_revolute_joint("J3", "ground", "O4", "lever", "C").unwrap();
    // J4: pin slides along lever axis (prismatic constraint)
    mech.add_prismatic_joint(
        "J4", "lever", "C", "pin", "P",
        nalgebra::Vector2::new(1.0, 0.0), 0.0,
    ).unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;
    mech.build().map_err(|e| e.to_string())?;

    // Initial guess: crank at theta=0 (horizontal right)
    // Crank tip B at (l_crank, 0) = (0.020, 0)
    // Lever at O4=(0.030, 0), must point toward pin at (0.020, 0)
    // Lever angle = atan2(0 - 0, 0.020 - 0.030) = atan2(0, -0.01) = PI
    let pin_x = l_crank;
    let pin_y = 0.0;
    let lever_angle = (pin_y - o4.1).atan2(pin_x - o4.0);

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, o2.0, o2.1, 0.0);
    state.set_pose("pin", &mut q0, pin_x, pin_y, lever_angle);
    state.set_pose("lever", &mut q0, o4.0, o4.1, lever_angle);

    match solve_position(&mech, &q0, 0.0, 1e-10, 50) {
        Ok(result) if result.converged => Ok((mech, result.q)),
        Ok(_) => Ok((mech, q0)),
        Err(_) => Ok((mech, q0)),
    }
}

/// Bell crank mechanism (90-degree force/motion redirection).
///
/// A classic mechanism with an L-shaped ternary lever (the "bell crank")
/// pivoted at its bend. An input crank drives one arm via a coupler,
/// and a rocker on the other arm redirects the motion by ~90 degrees.
///
/// Implemented as a standard 4-bar linkage where the coupler is replaced
/// by a ternary body (the bell crank) grounded at the bend:
///
///   ground(O_input) → input_crank → bell_crank(input_arm)
///   bell_crank(pivot=O_bell on ground) is the L-shaped lever
///   bell_crank(output_arm) → output_rocker → ground(O_output)
///
/// Bodies: ground, input_crank, bell_crank (ternary), output_rocker
/// Joints (4 revolute):
///   J1: ground(O_input) - input_crank   (driver)
///   J2: input_crank - bell_crank(P2)    (input arm tip)
///   J3: ground(O_bell) - bell_crank(P1) (bell pivot)
///   J4: bell_crank(P3) - output_rocker  (output arm tip)
///   J5: output_rocker - ground(O_output)
///
/// DOF: 3 bodies * 3 = 9, 5 revolute * 2 = 10, driver = 1 → 9-10-1 = not right...
/// Actually: 3 moving bodies * 3 = 9 DOF, 5 revolute joints * 2 = 10 constraints.
/// That gives -1 DOF (overconstrained by 1). So this is actually a constrained
/// Watt-type 6-bar with the bell crank as a shared coupler.
///
/// Correct approach: use the 6-bar formulation similar to Watt I.
/// The bell crank ternary is grounded, with the input and output 4-bar loops
/// sharing it. This gives:
///   4 moving bodies: input_crank, coupler_in, output_link, coupler_out
///   bell_crank is the ternary ground
///
/// Actually, the simplest correct bell crank:
///   ground(O_input) → crank → coupler → ternary_bell(grounded at O_bell)
///   with the ternary's other arm having a coupler point showing the 90-deg redirect.
///
/// This is a simple 4-bar with a ternary rocker (grounded at one arm, driven at
/// the other). The rocker has an L-shape so the third point shows 90-deg redirect.
///
/// 3 moving bodies (crank, coupler, bell_rocker), 4 revolute joints:
///   J1: ground(O_input) - crank
///   J2: crank - coupler
///   J3: coupler - bell_rocker(P2) (input arm tip)
///   J4: ground(O_bell) - bell_rocker(P1) (pivot at bend)
///
/// DOF: 3*3=9, 4*2=8+1(driver)=9 → 0 DOF. Correct!
pub(super) fn build_bell_crank(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    use super::helpers::make_ternary;

    // Ground pivots
    let o_input = (0.0, 0.0);   // input crank pivot
    let o_bell = (0.050, 0.0);   // bell crank pivot (at the bend)

    let l_crank = 0.015;      // input crank
    let l_coupler = 0.045;    // coupler connecting crank to bell arm

    // Bell crank: L-shaped ternary body
    // P1 = origin (0,0) = ground pivot at O_bell
    // P2 = (0.035, 0) = input arm tip (horizontal, connects to coupler)
    // P3 = (0, 0.035) = output arm tip (vertical, 90-deg redirect)
    let bell_arm_h = 0.035; // horizontal arm (input side)
    let bell_arm_v = 0.035; // vertical arm (output side, 90 degrees)

    let ground = make_ground(&[
        ("O_input", o_input.0, o_input.1),
        ("O_bell", o_bell.0, o_bell.1),
    ]);
    let mut crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    set_bar_mass(&mut crank, l_crank);
    let mut coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
    set_bar_mass(&mut coupler, l_coupler);

    let mut bell_rocker = make_ternary(
        "bell_rocker", "P1", "P2", "P3",
        (bell_arm_h, 0.0), (0.0, bell_arm_v),
    );
    // Coupler point at output arm tip to show the 90-degree redirect
    bell_rocker.add_coupler_point("CP", 0.0, bell_arm_v).unwrap();
    set_ternary_mass(&mut bell_rocker, (bell_arm_h, 0.0), (0.0, bell_arm_v));

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(bell_rocker).unwrap();

    // J1: crank pivots on ground at O_input
    mech.add_revolute_joint("J1", "ground", "O_input", "crank", "A").unwrap();
    // J2: crank-coupler
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    // J3: coupler-bell crank input arm
    mech.add_revolute_joint("J3", "coupler", "C", "bell_rocker", "P2").unwrap();
    // J4: bell crank pivot on ground
    mech.add_revolute_joint("J4", "bell_rocker", "P1", "ground", "O_bell").unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;
    mech.build().map_err(|e| e.to_string())?;

    // This is a standard 4-bar: ground(O_input→O_bell), crank, coupler,
    // bell_rocker(P2→P1). The "rocker" origin is at P2 (the coupler-rocker
    // joint), and the bar extends to P1 (at O_bell) with length = bell_arm_h.
    // But wait: the bell_rocker is a ternary, and its first attachment (P1)
    // is at (0,0), P2 at (bell_arm_h, 0). So body origin = P1 (at O_bell),
    // and P2 is at +x. The revolute J3 connects coupler-C to bell_rocker-P2.
    // J4 connects bell_rocker-P1 to ground-O_bell.
    //
    // For fourbar_initial_q0, the "rocker" origin (body origin) is at the
    // coupler-rocker joint. But here, the bell_rocker origin (P1) is at O_bell
    // (the ground pivot), and P2 (the coupler joint) is at (bell_arm_h, 0).
    // This is the OPPOSITE convention from the standard 4-bar samples where
    // the rocker origin is at C (coupler joint) and D (ground pivot) is at
    // the end. So we need custom initial pose logic.

    let state = mech.state();
    let mut q0 = state.make_q();

    // Crank at theta=0: origin at O_input, B at (l_crank, 0)
    let theta_crank = 0.0;
    state.set_pose("crank", &mut q0, o_input.0, o_input.1, theta_crank);
    let bx = o_input.0 + l_crank;
    let by = o_input.1;

    // The bell_rocker acts as a rocker with origin at P1=O_bell and P2 at
    // (bell_arm_h, 0) in local coords. The 4-bar loop closes:
    //   B → C (coupler) → P2 (bell arm) → P1 (=O_bell)
    // So: coupler length = l_coupler, "rocker" = bell_arm_h, rocker origin at O_bell.
    // The "rocker" here has origin at P1=O_bell and goes to P2 (connection point).
    // This is like a standard 4-bar but the rocker is measured from ground pivot.
    //
    // Solve triangle: B to O_bell, then use law of cosines for coupler/bell_arm.
    let dx = bx - o_bell.0;
    let dy = by - o_bell.1;
    let d = (dx * dx + dy * dy).sqrt();
    let alpha = dy.atan2(dx);

    // Law of cosines for angle at O_bell
    let cos_beta = (d * d + bell_arm_h * bell_arm_h - l_coupler * l_coupler)
        / (2.0 * d * bell_arm_h);
    let cos_beta = cos_beta.clamp(-1.0, 1.0);
    let beta = cos_beta.acos();

    // Bell rocker angle: direction from P1(O_bell) toward P2 (the coupler joint C)
    // Choose the above branch (+ y)
    let theta_bell = alpha + beta;

    // P2 in global
    let p2x = o_bell.0 + bell_arm_h * theta_bell.cos();
    let p2y = o_bell.1 + bell_arm_h * theta_bell.sin();

    state.set_pose("bell_rocker", &mut q0, o_bell.0, o_bell.1, theta_bell);

    // Coupler: from B to P2 (=C)
    let theta_coupler = (p2y - by).atan2(p2x - bx);
    state.set_pose("coupler", &mut q0, bx, by, theta_coupler);

    Ok((mech, q0))
}

/// Coupler curve mechanism (4-bar with figure-8 trace).
///
/// A 4-bar linkage with proportions chosen to produce an interesting
/// figure-8 or kidney-shaped coupler curve. The coupler point is offset
/// perpendicular to the coupler bar to maximize the curve's visual interest.
///
/// Substitutes for the Peaucellier-Lipkin 8-bar linkage, which requires
/// collocated joints that are too complex for the constraint solver.
///
/// Proportions: ground=60mm, crank=40mm, coupler=80mm, rocker=60mm.
/// Change-point Grashof: 40+80 = 60+60 = 120 (limit case).
/// Coupler point at (40mm, 26.7mm) in coupler-local coords.
///
/// Uses PI/2 as initial crank angle to avoid the change-point singularity.
pub(super) fn build_peaucellier_lipkin(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (0.060_f64, 0.0_f64); // ground = 60mm
    let l_crank = 0.040_f64;       // 40mm
    let l_coupler = 0.080_f64;     // 80mm
    let l_rocker = 0.060_f64;      // 60mm

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let mut crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    let mut coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
    // Coupler point at offset for interesting trace
    coupler.add_coupler_point("P", l_coupler / 2.0, l_coupler / 3.0).unwrap();
    let mut rocker = make_bar("rocker", "C", "D", l_rocker, 0.0, 0.0);
    set_bar_mass(&mut crank, l_crank);
    set_bar_mass(&mut coupler, l_coupler);
    set_bar_mass(&mut rocker, l_rocker);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();

    // Use PI/2 as initial crank angle since at theta=0 the geometry is at a
    // change-point singularity (crank tip to O4 distance equals |coupler-rocker|).
    let init_angle = std::f64::consts::FRAC_PI_2;
    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", init_angle)?;

    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4, l_crank, l_coupler, l_rocker, init_angle,
        "crank", "coupler", "rocker", true,
    );

    Ok((mech, q0))
}

// ---------------------------------------------------------------------------
// Strandbeest (Jansen walking linkage)
// ---------------------------------------------------------------------------

/// Compute the local coordinates of the third vertex of a triangle given
/// three side lengths, with P1 at the origin and P2 along the +x axis.
///
/// Returns `(x, y)` for the third point P3 where:
///   P1-P2 = `base`, P1-P3 = `side_a`, P2-P3 = `side_b`
///
/// The returned y is positive (above the baseline). Negate y for the
/// mirror solution below the baseline.
fn triangle_third_vertex(base: f64, side_a: f64, side_b: f64) -> (f64, f64) {
    let cos_a = (base * base + side_a * side_a - side_b * side_b) / (2.0 * base * side_a);
    let cos_a = cos_a.clamp(-1.0, 1.0);
    let sin_a = (1.0 - cos_a * cos_a).sqrt();
    (side_a * cos_a, side_a * sin_a)
}

/// Circle-circle intersection: returns the two intersection points of
/// circle(c1, r1) and circle(c2, r2), or `None` if no intersection.
fn circle_circle_intersect(
    c1: (f64, f64), r1: f64,
    c2: (f64, f64), r2: f64,
) -> Option<((f64, f64), (f64, f64))> {
    let dx = c2.0 - c1.0;
    let dy = c2.1 - c1.1;
    let d = (dx * dx + dy * dy).sqrt();
    if d > r1 + r2 + 1e-9 || d < (r1 - r2).abs() - 1e-9 || d < 1e-15 {
        return None;
    }
    let a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    let h_sq = r1 * r1 - a * a;
    let h = if h_sq > 0.0 { h_sq.sqrt() } else { 0.0 };
    let mx = c1.0 + a * dx / d;
    let my = c1.1 + a * dy / d;
    let px = -dy / d * h;
    let py = dx / d * h;
    Some(((mx + px, my + py), (mx - px, my - py)))
}

/// Strandbeest (Jansen walking linkage).
///
/// Theo Jansen's 1-DOF walking mechanism with 8 bars that converts crank
/// rotation into a foot path with approximately flat ground contact.
/// Uses Jansen's published "holy numbers" for link proportions, scaled to meters.
///
/// Topology (8 links including ground, 10 revolute joints, 1 DOF):
///
///   Ground: two fixed pivots O=(0,0) and C=(a,0).
///   Crank: O→A, length m.
///
///   Two 4-bar sub-loops share the crank and ground:
///     Upper loop: O → crank → bar_j → upper_tri(U1,U2) → bar_k → C
///     Lower loop: O → crank → bar_b → lower_tri(L1,L2) → bar_c → C
///
///   The ternaries are coupled: upper_tri.Ec ↔ lower_tri.Ec via a direct
///   revolute joint (no intermediate bar).
///
///   The foot is a coupler point on lower_tri, tracing the characteristic
///   walking path with flat ground contact.
///
/// Jansen's "holy numbers" (mm, divided by 1000 for meters):
///   a=38, b=41.5, c=39.3, d=40.1, e=55.8, f=39.4, g=36.7, h=65.7,
///   i=49.0, j=50.0, k=61.9, l=7.8, m=15.0
///
/// Dimension mapping (verified by numerical circle-circle closure):
///   - Ground O-C = a, Crank = m
///   - bar_j = j (crank tip → U1), bar_k = k (ground C → U2)
///   - bar_b = b (crank tip → L1), bar_c = c (ground C → L2)
///   - Upper ternary: U1-U2 = d, U1-Ec = i, U2-Ec = e
///   - Lower ternary: L1-L2 = f, L1-Ec = h, L2-Ec = g
pub(super) fn build_strandbeest(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    // Jansen's "holy numbers" in meters (mm / 1000).
    let a = 0.0380; // ground distance O to C
    let b = 0.0415; // bar: crank tip → lower ternary L1
    let c = 0.0393; // bar: ground C → lower ternary L2
    let d = 0.0401; // upper ternary U1-U2 distance
    let e = 0.0558; // upper ternary U2-Ec distance
    let f = 0.0394; // lower ternary L1-L2 distance
    let g = 0.0367; // lower ternary L2-Ec distance
    let h = 0.0657; // lower ternary L1-Ec distance
    let i_len = 0.0490; // upper ternary U1-Ec distance
    let j = 0.0500; // bar: crank tip → upper ternary U1
    let k = 0.0619; // bar: ground C → upper ternary U2
    let _l = 0.0078; // (not used in this parameterization)
    let m_crank = 0.0150; // crank length

    // --- Compute ternary local coordinates ---

    // Upper ternary (U1, U2, Ec): U1 at local origin, U2 along +x.
    // Sides: U1-U2 = d, U1-Ec = i_len, U2-Ec = e.
    // Ec placed below baseline (−y) to extend toward the lower mechanism.
    let (uec_x, uec_y) = triangle_third_vertex(d, i_len, e);
    let upper_p2 = (d, 0.0);
    let upper_p3 = (uec_x, -uec_y); // Ec below baseline

    // Lower ternary (L1, L2, Ec_lower): L1 at local origin, L2 along +x.
    // Sides: L1-L2 = f, L1-Ec = h, L2-Ec = g.
    // Ec placed above baseline (+y) toward the upper mechanism.
    let (lec_x, lec_y) = triangle_third_vertex(f, h, g);
    let lower_p2 = (f, 0.0);
    let lower_p3 = (lec_x, lec_y); // Ec_lower above baseline

    // Foot coupler point: mirror of Ec across the L1-L2 baseline (below).
    // This extends the lower ternary downward to trace the walking path.
    let foot_local = (lec_x, -lec_y);

    // --- Build bodies ---

    let ground = make_ground(&[("O", 0.0, 0.0), ("C", a, 0.0)]);

    let mut crank = make_bar("crank", "A", "Tip", m_crank, 0.0, 0.0);
    set_bar_mass(&mut crank, m_crank);

    let mut bar_j = make_bar("bar_j", "J1", "J2", j, 0.0, 0.0);
    set_bar_mass(&mut bar_j, j);

    let mut bar_k = make_bar("bar_k", "K1", "K2", k, 0.0, 0.0);
    set_bar_mass(&mut bar_k, k);

    let mut upper_tri = make_ternary("upper_tri", "U1", "U2", "Ec", upper_p2, upper_p3);
    set_ternary_mass(&mut upper_tri, upper_p2, upper_p3);

    let mut bar_b = make_bar("bar_b", "Bb1", "Bb2", b, 0.0, 0.0);
    set_bar_mass(&mut bar_b, b);

    let mut bar_c = make_bar("bar_c", "Bc1", "Bc2", c, 0.0, 0.0);
    set_bar_mass(&mut bar_c, c);

    let mut lower_tri = make_ternary("lower_tri", "L1", "L2", "Ec", lower_p2, lower_p3);
    lower_tri
        .add_coupler_point("FootTrace", foot_local.0, foot_local.1)
        .unwrap();
    set_ternary_mass(&mut lower_tri, lower_p2, lower_p3);

    // --- Assemble mechanism ---

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(bar_j).unwrap();
    mech.add_body(bar_k).unwrap();
    mech.add_body(upper_tri).unwrap();
    mech.add_body(bar_b).unwrap();
    mech.add_body(bar_c).unwrap();
    mech.add_body(lower_tri).unwrap();

    // 10 revolute joints for 1 DOF:
    //   DOF = 3*(8-1) - 2*10 = 21 - 20 = 1
    mech.add_revolute_joint("J01", "ground", "O", "crank", "A").unwrap();
    mech.add_revolute_joint("J02", "crank", "Tip", "bar_j", "J1").unwrap();
    mech.add_revolute_joint("J03", "crank", "Tip", "bar_b", "Bb1").unwrap();
    mech.add_revolute_joint("J04", "ground", "C", "bar_k", "K1").unwrap();
    mech.add_revolute_joint("J05", "ground", "C", "bar_c", "Bc1").unwrap();
    mech.add_revolute_joint("J06", "bar_j", "J2", "upper_tri", "U1").unwrap();
    mech.add_revolute_joint("J07", "bar_k", "K2", "upper_tri", "U2").unwrap();
    mech.add_revolute_joint("J08", "bar_b", "Bb2", "lower_tri", "L1").unwrap();
    mech.add_revolute_joint("J09", "bar_c", "Bc2", "lower_tri", "L2").unwrap();
    mech.add_revolute_joint("J10", "upper_tri", "Ec", "lower_tri", "Ec").unwrap();

    // Driver on crank pivot
    let joint_id = driver_joint_id.unwrap_or("J01");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;
    mech.build().map_err(|e| e.to_string())?;

    // --- Initial guess via continuation ---
    let state = mech.state();
    let q0 = strandbeest_find_initial(
        state, &mech, a, m_crank, j, k, b, c, d, i_len, e, f, g, h,
    )?;

    Ok((mech, q0))
}

/// Compute a converging initial guess for the Strandbeest mechanism.
///
/// Tries several starting crank angles. For each, uses circle-circle
/// intersection to compute exact sub-loop closure points, sets all body
/// poses accordingly, solves at that angle, then continues to t=0.
fn strandbeest_find_initial(
    state: &crate::core::state::State,
    mech: &Mechanism,
    a: f64,
    m_crank: f64,
    j_len: f64,
    k_len: f64,
    b_len: f64,
    c_len: f64,
    d_len: f64,
    i_len: f64,
    e_len: f64,
    f_len: f64,
    g_len: f64,
    h_len: f64,
) -> Result<DVector<f64>, String> {
    use std::f64::consts::PI;

    let params = JansenParams {
        a, m_crank, j_len, k_len, b_len, c_len,
        d_len, i_len, e_len, f_len, g_len, h_len,
    };

    // Sweep crank angles from 0 to 2*PI in small steps.
    // For each angle, try to build an exact geometric guess via
    // circle-circle intersections, then solve and continue to t=0.
    let n_angles = 72; // every 5 degrees
    for i in 0..n_angles {
        let crank_angle = 2.0 * PI * (i as f64) / (n_angles as f64);
        let t_for_angle = crank_angle / (2.0 * PI);

        if let Some(q_guess) = strandbeest_exact_guess(state, crank_angle, &params) {
            if let Ok(result) = solve_position(mech, &q_guess, t_for_angle, 1e-10, 200) {
                if result.converged {
                    // Continue from this angle to t=0
                    if let Some(q0) = solve_with_continuation(
                        mech, &result.q, t_for_angle, 0.0, 60,
                    ) {
                        return Ok(q0);
                    }
                }
            }
        }
    }
    Err("Strandbeest: could not find converging initial guess".to_string())
}

/// Parameters for the Jansen linkage geometry.
struct JansenParams {
    a: f64,
    m_crank: f64,
    j_len: f64,
    k_len: f64,
    b_len: f64,
    c_len: f64,
    d_len: f64,
    i_len: f64,
    e_len: f64,
    f_len: f64,
    g_len: f64,
    h_len: f64,
}

/// Generate an exact geometric initial guess using circle-circle intersections.
///
/// For a given crank angle, sweeps bar_j and bar_b angles to find configurations
/// where all sub-loops close (coupling gap < tolerance). Returns the first
/// valid configuration found.
fn strandbeest_exact_guess(
    state: &crate::core::state::State,
    crank_angle: f64,
    p: &JansenParams,
) -> Option<DVector<f64>> {
    let tip = (
        p.m_crank * crank_angle.cos(),
        p.m_crank * crank_angle.sin(),
    );
    let c_pt = (p.a, 0.0);

    // Sweep bar_j angle to find upper sub-loop closure
    let n_search = 180; // every 2 degrees
    let mut best_gap = f64::MAX;
    let mut best_config: Option<JansenConfig> = None;

    for i_j in 0..n_search {
        let alpha = std::f64::consts::PI * 2.0 * (i_j as f64) / (n_search as f64)
            - std::f64::consts::PI;
        let u1 = (
            tip.0 + p.j_len * alpha.cos(),
            tip.1 + p.j_len * alpha.sin(),
        );

        // U2 on circle(U1, d_len) ∩ circle(C, k_len)
        let Some((u2a, u2b)) = circle_circle_intersect(u1, p.d_len, c_pt, p.k_len) else {
            continue;
        };

        for &u2 in &[u2a, u2b] {
            // Ec on upper: circle(U1, i_len) ∩ circle(U2, e_len)
            let Some((ec_ua, ec_ub)) = circle_circle_intersect(u1, p.i_len, u2, p.e_len) else {
                continue;
            };

            for &ec_upper in &[ec_ua, ec_ub] {
                // Now sweep bar_b angle for lower sub-loop
                for i_b in 0..n_search {
                    let beta = std::f64::consts::PI * 2.0 * (i_b as f64)
                        / (n_search as f64)
                        - std::f64::consts::PI;
                    let l1 = (
                        tip.0 + p.b_len * beta.cos(),
                        tip.1 + p.b_len * beta.sin(),
                    );

                    // L2 on circle(L1, f_len) ∩ circle(C, c_len)
                    let Some((l2a, l2b)) =
                        circle_circle_intersect(l1, p.f_len, c_pt, p.c_len)
                    else {
                        continue;
                    };

                    for &l2 in &[l2a, l2b] {
                        // Ec on lower: circle(L1, h_len) ∩ circle(L2, g_len)
                        let Some((ec_la, ec_lb)) =
                            circle_circle_intersect(l1, p.h_len, l2, p.g_len)
                        else {
                            continue;
                        };

                        for &ec_lower in &[ec_la, ec_lb] {
                            let gap = ((ec_upper.0 - ec_lower.0).powi(2)
                                + (ec_upper.1 - ec_lower.1).powi(2))
                            .sqrt();
                            if gap < best_gap {
                                best_gap = gap;
                                best_config = Some(JansenConfig {
                                    tip,
                                    u1,
                                    u2,
                                    l1,
                                    l2,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Accept if coupling gap is small enough for the Newton solver to close.
    // With 2-degree angular resolution, the gap can be up to ~1mm (0.001m).
    let tolerance = 0.001; // 1mm in meters
    if best_gap > tolerance {
        return None;
    }

    let cfg = best_config?;
    let mut q = state.make_q();

    // Crank
    state.set_pose("crank", &mut q, 0.0, 0.0, crank_angle);

    // bar_j: from tip to U1
    let bar_j_angle = (cfg.u1.1 - cfg.tip.1).atan2(cfg.u1.0 - cfg.tip.0);
    state.set_pose("bar_j", &mut q, cfg.tip.0, cfg.tip.1, bar_j_angle);

    // Upper ternary: origin at U1, angle from U1→U2 direction
    let upper_angle = (cfg.u2.1 - cfg.u1.1).atan2(cfg.u2.0 - cfg.u1.0);
    state.set_pose("upper_tri", &mut q, cfg.u1.0, cfg.u1.1, upper_angle);

    // bar_k: from C to U2
    let bar_k_angle = (cfg.u2.1 - c_pt.1).atan2(cfg.u2.0 - c_pt.0);
    state.set_pose("bar_k", &mut q, c_pt.0, c_pt.1, bar_k_angle);

    // bar_b: from tip to L1
    let bar_b_angle = (cfg.l1.1 - cfg.tip.1).atan2(cfg.l1.0 - cfg.tip.0);
    state.set_pose("bar_b", &mut q, cfg.tip.0, cfg.tip.1, bar_b_angle);

    // Lower ternary: origin at L1, angle from L1→L2 direction
    let lower_angle = (cfg.l2.1 - cfg.l1.1).atan2(cfg.l2.0 - cfg.l1.0);
    state.set_pose("lower_tri", &mut q, cfg.l1.0, cfg.l1.1, lower_angle);

    // bar_c: from C to L2
    let bar_c_angle = (cfg.l2.1 - c_pt.1).atan2(cfg.l2.0 - c_pt.0);
    state.set_pose("bar_c", &mut q, c_pt.0, c_pt.1, bar_c_angle);

    Some(q)
}

/// Configuration of all key points in the Jansen linkage at a single
/// crank angle, found by circle-circle intersection.
struct JansenConfig {
    tip: (f64, f64),
    u1: (f64, f64),
    u2: (f64, f64),
    l1: (f64, f64),
    l2: (f64, f64),
}

/// Custom 6-bar press mechanism with linear actuator and force zone.
///
/// Loaded from an embedded JSON blueprint (decoded from a user's share URL).
/// The mechanism is a 6-bar linkage with 8 revolute joints, a linear actuator
/// force element, and a force zone on the output link.
pub(super) fn build_custom_6bar(
    _driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    use crate::io::load_mechanism_unbuilt;

    let json = include_str!("custom_6bar.json");

    let mut mech = load_mechanism_unbuilt(json).map_err(|e| e.to_string())?;
    mech.build().map_err(|e| e.to_string())?;

    let q0 = mech.state().make_q();
    match solve_position(&mech, &q0, 0.0, 1e-10, 100) {
        Ok(result) if result.converged => Ok((mech, result.q)),
        Ok(result) => {
            // Fall back to zero-vector initial guess if solver didn't converge
            log::warn!(
                "Custom 6-bar: initial solve did not converge (residual={}), using zero q0",
                result.residual_norm
            );
            Ok((mech, q0))
        }
        Err(e) => {
            log::warn!("Custom 6-bar: initial solve error ({}), using zero q0", e);
            Ok((mech, q0))
        }
    }
}
