//! Shared helper functions for sample mechanism builders.

use nalgebra::{DVector, Vector2};
use std::f64::consts::PI;

use crate::core::body::Body;
use crate::core::constraint::Constraint;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::solver::kinematics::solve_position;

/// Set mass properties for a sample bar body: 1 kg uniform rod of the given length.
///
/// - `mass` = 1.0 kg
/// - `cg_local` = midpoint of the bar (length/2, 0)
/// - `izz_cg` = m * L^2 / 12  (uniform rod about its center)
pub fn set_bar_mass(body: &mut Body, length: f64) {
    body.mass = 1.0;
    body.cg_local = Vector2::new(length / 2.0, 0.0);
    body.izz_cg = length * length / 12.0; // m * L^2 / 12 with m = 1
}

/// Set mass properties for a sample ternary body: 1 kg, CG at centroid of
/// the three attachment points, Izz from the bounding-circle radius.
///
/// `p2_local` and `p3_local` are the body-local coordinates of the second
/// and third attachment points (P1 is assumed at the origin).
pub fn set_ternary_mass(body: &mut Body, p2_local: (f64, f64), p3_local: (f64, f64)) {
    let cx = (p2_local.0 + p3_local.0) / 3.0;
    let cy = (p2_local.1 + p3_local.1) / 3.0;
    // Characteristic length: max distance from origin to any attachment point.
    let d2 = (p2_local.0 * p2_local.0 + p2_local.1 * p2_local.1).sqrt();
    let d3 = (p3_local.0 * p3_local.0 + p3_local.1 * p3_local.1).sqrt();
    let l_char = d2.max(d3);
    body.mass = 1.0;
    body.cg_local = Vector2::new(cx, cy);
    body.izz_cg = l_char * l_char / 12.0; // m * L^2 / 12 with m = 1
}

/// Set mass properties for a sample slider body: 1 kg point mass at origin.
pub fn set_slider_mass(body: &mut Body) {
    body.mass = 1.0;
    // CG at origin (single attachment point), Izz ~ 0 for a compact block.
    body.cg_local = Vector2::new(0.0, 0.0);
    body.izz_cg = 0.0;
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
pub fn attach_driver_to_grounded_revolute_with_theta0(
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

/// Compute the rocker angle for a given crank angle in the 4-bar sample.
///
/// Uses the law of cosines on the diagonal from O2+crank_tip to O4 to find
/// the rocker angle that closes the loop.
pub fn fourbar_rocker_angle_for_crank(
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

/// Result of the 4-bar loop-closure geometry computation.
///
/// Contains all the computed positions and angles needed to set body poses
/// for either rocker attachment convention.
#[derive(Debug, Clone)]
pub struct FourbarClosureResult {
    /// Crank tip (point B) in world frame.
    pub bx: f64,
    pub by: f64,
    /// Coupler-rocker joint (point C) in world frame.
    pub cx: f64,
    pub cy: f64,
    /// Crank angle (same as input).
    pub theta_crank: f64,
    /// Coupler angle: direction from B to C.
    pub theta_coupler: f64,
    /// Rocker angle: direction from C to D (D = O4).
    /// Use this when the rocker body origin is at C (sample convention).
    pub theta_rocker_c_to_d: f64,
    /// Rocker angle: direction from D to C (D = O4).
    /// Use this when the rocker body origin is at D (test convention).
    pub theta_rocker_d_to_c: f64,
}

/// Solve the 4-bar loop-closure geometry for a given crank angle.
///
/// Returns `None` if the triangle inequality is violated (the coupler
/// and rocker cannot bridge from crank tip B to rocker pivot O4).
///
/// This is the pure geometry computation with no dependency on the `State`
/// type. Both `fourbar_initial_q0` and test helpers build on this.
pub fn fourbar_loop_closure(
    o2: (f64, f64),
    o4: (f64, f64),
    l_crank: f64,
    l_coupler: f64,
    l_rocker: f64,
    theta_crank: f64,
    above: bool,
) -> Option<FourbarClosureResult> {
    // Crank tip (point B) in world frame.
    let bx = o2.0 + l_crank * theta_crank.cos();
    let by = o2.1 + l_crank * theta_crank.sin();

    // Distance from O4 to crank tip B.
    let dx = bx - o4.0;
    let dy = by - o4.1;
    let d = (dx * dx + dy * dy).sqrt();

    // Triangle inequality check.
    if d > l_coupler + l_rocker || d < (l_coupler - l_rocker).abs() {
        return None;
    }

    let alpha = dy.atan2(dx);
    let cos_beta = (d * d + l_rocker * l_rocker - l_coupler * l_coupler)
        / (2.0 * d * l_rocker);
    let cos_beta = cos_beta.clamp(-1.0, 1.0);
    let beta = cos_beta.acos();

    // Rocker angle C→D: direction from C toward D=O4.
    // alpha + beta + PI places coupler below ground line (−y);
    // alpha − beta + PI places coupler above ground line (+y).
    let theta_rocker_c_to_d = if above {
        alpha - beta + PI
    } else {
        alpha + beta + PI
    };

    let cx = o4.0 - l_rocker * theta_rocker_c_to_d.cos();
    let cy = o4.1 - l_rocker * theta_rocker_c_to_d.sin();

    // D→C direction is C→D + PI (opposite direction), normalized to (-PI, PI].
    let theta_rocker_d_to_c = {
        let raw = theta_rocker_c_to_d + PI;
        let mut norm = raw % (2.0 * PI);
        if norm > PI {
            norm -= 2.0 * PI;
        } else if norm <= -PI {
            norm += 2.0 * PI;
        }
        norm
    };

    let theta_coupler = (cy - by).atan2(cx - bx);

    Some(FourbarClosureResult {
        bx,
        by,
        cx,
        cy,
        theta_crank,
        theta_coupler,
        theta_rocker_c_to_d,
        theta_rocker_d_to_c,
    })
}

/// Compute geometrically consistent initial poses for a 4-bar linkage
/// given a crank angle. Returns `None` if the triangle inequality is
/// violated (the coupler and rocker cannot reach from crank tip to O4).
///
/// This is the fallible version of [`fourbar_initial_q0`]; prefer this
/// when the link lengths come from untrusted / randomized input.
///
/// Assumes the **sample convention**: rocker body origin at C (the
/// coupler-rocker joint), with D at the ground pivot O4. For the
/// opposite convention (origin at D), use [`fourbar_loop_closure`]
/// directly.
pub fn try_fourbar_initial_q0(
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
    above: bool,
) -> Option<DVector<f64>> {
    let geom = fourbar_loop_closure(o2, o4, l_crank, l_coupler, l_rocker, theta_crank, above)?;
    let mut q0 = state.make_q();

    state.set_pose(crank_id, &mut q0, o2.0, o2.1, geom.theta_crank);
    state.set_pose(rocker_id, &mut q0, geom.cx, geom.cy, geom.theta_rocker_c_to_d);
    state.set_pose(coupler_id, &mut q0, geom.bx, geom.by, geom.theta_coupler);

    Some(q0)
}

/// Compute geometrically consistent initial poses for a 4-bar linkage
/// given a crank angle.
///
/// Works by forward kinematics: place the crank at the given angle, then
/// solve the coupler/rocker positions via the loop closure triangle.
///
/// `crank_id`, `coupler_id`, `rocker_id` are the body IDs in the mechanism.
///
/// Assumes the **sample convention**: rocker body origin at C (the
/// coupler-rocker joint), with D at the ground pivot O4. For the
/// opposite convention (origin at D), use [`fourbar_loop_closure`]
/// directly.
///
/// # Panics
/// Does not check the triangle inequality. If the geometry cannot close,
/// the result will contain `NaN` angles. Use [`try_fourbar_initial_q0`]
/// for a checked version.
pub fn fourbar_initial_q0(
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
    above: bool,
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
    // (alpha ± beta) at distance l_rocker. The sign of beta selects the
    // assembly branch: alpha + beta + PI places the coupler below the
    // ground line (−y); alpha − beta + PI places it above (+y).
    let theta_rocker = if above {
        alpha - beta + PI
    } else {
        alpha + beta + PI
    };
    let cx = o4.0 - l_rocker * theta_rocker.cos();
    let cy = o4.1 - l_rocker * theta_rocker.sin();
    state.set_pose(rocker_id, &mut q0, cx, cy, theta_rocker);

    // Coupler: origin at B (point B in local = (0,0)), angle from B→C direction.
    let theta_coupler = (cy - by).atan2(cx - bx);
    state.set_pose(coupler_id, &mut q0, bx, by, theta_coupler);

    q0
}

/// Create a ternary body with 3 attachment points.
///
/// P1 is at the local origin (0,0). P2 and P3 are specified in body-local
/// coordinates.
pub fn make_ternary(
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
pub fn solve_with_continuation(
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
