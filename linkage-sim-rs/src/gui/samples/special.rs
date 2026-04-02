//! Special mechanism sample builders (quick-return, toggle clamp, scotch yoke, etc.).

use nalgebra::DVector;

use crate::core::body::{make_bar, make_ground, Body};
use crate::core::mechanism::Mechanism;
use crate::solver::kinematics::solve_position;

use super::helpers::{
    attach_driver_to_grounded_revolute_with_theta0, fourbar_initial_q0, set_bar_mass,
    set_slider_mass, set_ternary_mass,
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
