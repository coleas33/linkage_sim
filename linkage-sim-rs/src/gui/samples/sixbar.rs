//! Six-bar mechanism sample builders.

use nalgebra::DVector;

use crate::core::body::make_bar;
use crate::core::body::make_ground;
use crate::core::mechanism::Mechanism;
use crate::solver::kinematics::solve_position;

use super::helpers::{
    attach_driver_to_grounded_revolute_with_theta0, make_ternary, solve_with_continuation,
};

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
pub(super) fn build_sixbar_b1(
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
pub(super) fn build_sixbar_a1(
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
pub(super) fn build_sixbar_a2(
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
pub(super) fn build_sixbar_b2(
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
pub(super) fn build_sixbar_b3(
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

// ---------------------------------------------------------------------------
// Watt II -- 6-bar Watt type-II mechanism
// ---------------------------------------------------------------------------

/// 6-bar Watt type-II mechanism.
///
/// Topology: Two ternary links (T1 and T2) are each grounded, connected
/// through a binary link B1. Additional binary links B2 and B3 close
/// the kinematic chain.
///
/// Ground pivots: O2=(0,0), O4=(2.0,0)
/// T1 (grounded at O2): ternary with P1(origin), P2(0.8,0), P3(0.4,0.4)
/// T2 (grounded at O4): ternary with Q1(origin), Q2(0.8,0), Q3(0.4,0.4)
/// B1: binary link connecting T1.P2 to T2.Q2 (length 2.0)
/// B2: binary link (length 1.5)
/// B3: binary link (length 1.5)
///
/// Joints (7):
///   J1: ground-T1 (at O2)
///   J2: ground-T2 (at O4)
///   J3: T1.P2 - B1.A (ternary arm to coupler)
///   J4: B1.B - T2.Q2 (coupler to other ternary)
///   J5: T1.P3 - B2.A
///   J6: T2.Q3 - B3.A
///   J7: B2.B - B3.B (closes the chain)
///
/// Driver on J1 (ground-T1).
pub(super) fn build_watt_ii(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let ground = make_ground(&[
        ("O2", 0.0, 0.0),
        ("O4", 2.0, 0.0),
    ]);
    let t1 = make_ternary("t1", "P1", "P2", "P3", (0.8, 0.0), (0.4, 0.4));
    let t2 = make_ternary("t2", "Q1", "Q2", "Q3", (0.8, 0.0), (0.4, 0.4));
    let mut b1 = make_bar("b1", "A", "B", 2.0, 0.0, 0.0);
    b1.add_coupler_point("CP", 1.0, 0.0).unwrap();
    let b2 = make_bar("b2", "A", "B", 1.5, 0.0, 0.0);
    let b3 = make_bar("b3", "A", "B", 1.5, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(t1).unwrap();
    mech.add_body(t2).unwrap();
    mech.add_body(b1).unwrap();
    mech.add_body(b2).unwrap();
    mech.add_body(b3).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "t1", "P1").unwrap();
    mech.add_revolute_joint("J2", "ground", "O4", "t2", "Q1").unwrap();
    mech.add_revolute_joint("J3", "t1", "P2", "b1", "A").unwrap();
    mech.add_revolute_joint("J4", "b1", "B", "t2", "Q2").unwrap();
    mech.add_revolute_joint("J5", "t1", "P3", "b2", "A").unwrap();
    mech.add_revolute_joint("J6", "t2", "Q3", "b3", "A").unwrap();
    mech.add_revolute_joint("J7", "b2", "B", "b3", "B").unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    // Initial guess: T1 at small angle, then solve via continuation.
    let state = mech.state();
    let q0 = watt_ii_find_initial(state, &mech)?;

    Ok((mech, q0))
}

/// Compute geometric initial guess for the Watt II six-bar.
fn watt_ii_find_initial(
    state: &crate::core::state::State,
    mech: &Mechanism,
) -> Result<DVector<f64>, String> {
    for &angle in &[0.3, 0.5, 0.2, 0.8, 0.15, 1.0, 1.2, 0.1] {
        let mut q = state.make_q();

        // T1 at O2=(0,0), rotated by angle
        state.set_pose("t1", &mut q, 0.0, 0.0, angle);
        let ct1 = angle.cos();
        let st1 = angle.sin();

        // T1 attachment points in global
        let p2x = 0.8 * ct1;
        let p2y = 0.8 * st1;
        let p3x = 0.4 * ct1 - 0.4 * st1;
        let p3y = 0.4 * st1 + 0.4 * ct1;

        // T2 at O4=(2,0), try a small negative angle
        let t2_angle = -angle * 0.5;
        state.set_pose("t2", &mut q, 2.0, 0.0, t2_angle);
        let ct2 = t2_angle.cos();
        let st2 = t2_angle.sin();

        let q2x = 2.0 + 0.8 * ct2;
        let q2y = 0.8 * st2;
        let q3x = 2.0 + 0.4 * ct2 - 0.4 * st2;
        let q3y = 0.4 * st2 + 0.4 * ct2;

        // B1: connects P2 to Q2
        let theta_b1 = (q2y - p2y).atan2(q2x - p2x);
        state.set_pose("b1", &mut q, p2x, p2y, theta_b1);

        // B2 and B3 meet at a common point. Aim both toward the midpoint
        // of P3 and Q3 (approximately where they should meet).
        let mid_x = (p3x + q3x) / 2.0;
        let mid_y = (p3y + q3y) / 2.0 + 0.5; // offset upward for better guess
        let theta_b2 = (mid_y - p3y).atan2(mid_x - p3x);
        state.set_pose("b2", &mut q, p3x, p3y, theta_b2);

        let theta_b3 = (mid_y - q3y).atan2(mid_x - q3x);
        state.set_pose("b3", &mut q, q3x, q3y, theta_b3);

        if let Ok(result) = solve_position(mech, &q, angle, 1e-10, 200) {
            if result.converged {
                if let Some(q0) = solve_with_continuation(
                    mech, &result.q, angle, 0.0, 20,
                ) {
                    return Ok(q0);
                }
            }
        }
    }
    Err("WattII: could not find converging initial guess at any starting angle".to_string())
}

// ---------------------------------------------------------------------------
// Pantograph -- 5-bar motion-scaling mechanism
// ---------------------------------------------------------------------------

/// Pantograph mechanism (5-bar, motion scaling).
///
/// A classic mechanism used for scaling drawings and motion. The pantograph
/// uses a parallelogram sub-chain to produce an output point that traces a
/// scaled copy of the input point's path.
///
/// Topology: Ground + 4 moving bodies (binary links forming a parallelogram
/// with an extension arm).
///
/// Ground pivots: O=(0,0)
/// Link1 (input arm): grounded at O, length 2.0
/// Link2 (parallel bar): connects Link1 midpoint to Link3, length 1.5
/// Link3 (output arm): grounded at O, length 3.0 (passes through O, extends beyond)
/// Link4 (parallel bar): connects Link1 end to Link3 midpoint, length 1.5
///
/// Simplified as a 4-bar with extension:
///   Ground pivot at (0,0), second ground pivot at (0.05, 0)
///   Crank: 0.030m, Coupler: 0.050m, Rocker: 0.030m
///   Coupler point extends to (0.075, 0) for 1.5x scaling
///
/// Grashof: 0.030+0.050 < 0.050+0.030 -> 0.080 < 0.080 (change-point parallelogram).
pub(super) fn build_pantograph(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (0.050_f64, 0.0_f64);
    let l_crank = 0.030_f64;
    let l_coupler = 0.050_f64;
    let l_rocker = 0.030_f64;

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    // Coupler with extension point P at 1.5x coupler length for scaling
    let mut coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
    coupler.add_coupler_point("P", 0.075, 0.0).unwrap();
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

    use super::helpers::fourbar_initial_q0;
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4, l_crank, l_coupler, l_rocker, 0.0,
        "crank", "coupler", "rocker", false,
    );

    Ok((mech, q0))
}
