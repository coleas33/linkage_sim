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
pub(super) fn build_scotch_yoke(
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
        label: "slider".to_string(),
        geometry: None,
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
pub(super) fn build_inverted_slider_crank(
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
        label: "slider".to_string(),
        geometry: None,
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
