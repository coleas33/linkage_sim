//! Force-element-aware joint-reaction solve.
//!
//! Wraps the basic statics solve with a second pass when a `LinearActuator`
//! is present in sizing mode (`la.force ≈ 0`): the actuator's required force
//! is back-calculated from the power balance, re-injected into the
//! generalised force vector, and the reactions are re-extracted. The result
//! is what a strain gauge would read with the actuator as the prime mover —
//! driver torque drops to ~0 and the load flows through the actuator's
//! attachment chain.
//!
//! Without this layer, pass-1 statics treats the rotational driver as the
//! prime mover and the reactions on joints in the actuator's load path read
//! low. The previous-generation sweep had this two-pass logic inline; the
//! GUI's per-pose reaction display did not, and that was the source of the
//! GUI ↔ sweep divergence at joints in the actuator chain.
//!
//! The pass-2 step is gated on `la.force ≈ 0`. When the user has set a
//! specific actuator force, that force is already part of pass-1's
//! `q_forces` and re-applying it would double-count.

use nalgebra::{DVector, Vector2};

use crate::core::mechanism::Mechanism;
use crate::error::LinkageError;
use crate::forces::elements::{evaluate_linear_actuator, ForceElement, LinearActuatorElement};
use crate::solver::assembly::assemble_jacobian;
use crate::solver::kinematics::solve_velocity;
use crate::solver::statics::{
    extract_reactions, get_driver_reactions, solve_statics, JointReaction, StaticSolveResult,
};

/// Result of a force-element-aware reaction solve.
#[derive(Debug, Clone)]
pub struct ReactionSolveResult {
    /// Final joint reactions. Reflects pass-2 (actuator as prime mover) when
    /// `used_two_pass` is true; otherwise pass-1.
    pub reactions: Vec<JointReaction>,
    /// Driver torque from pass-1. Always pass-1 because the Driver-Torque
    /// plot shows what the rotational driver would need to deliver if it
    /// were the prime mover; pass-2 collapses this to ~0 by design.
    pub driver_torque: Option<f64>,
    /// Actuator force injected into pass-2, or `None` if pass-2 didn't run.
    pub actuator_force: Option<f64>,
    /// True when pass-2 ran and `reactions` reflects the actuator-as-prime-mover scenario.
    pub used_two_pass: bool,
    /// Pass-1 condition number (diagnostic; surfaces over-/under-constrained systems).
    pub condition_number: f64,
    /// Pass-1 over-constrained flag (diagnostic).
    pub is_overconstrained: bool,
}

/// Compute the actuator force required to drive the kinematic motion under
/// static balance, via the power-balance identity `F = τ·ω/(dL/dt)`.
///
/// Returns `None` when the actuator is at a singular configuration (length
/// rate near zero — e.g. perpendicular to motion) or when the formula
/// would yield a non-finite value.
///
/// `q_dot` must come from a velocity solve consistent with `driver_omega`
/// — typically `solve_velocity(mech, q, t)` with the same driver
/// configuration that produced `driver_torque`. The result is omega-
/// invariant (both numerator and denominator scale with omega), so any
/// consistent value works.
pub fn compute_actuator_force_from_power_balance(
    mech: &Mechanism,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    driver_torque: f64,
    driver_omega: f64,
    actuator: &LinearActuatorElement,
) -> Option<f64> {
    let mech_state = mech.state();
    let local_a = Vector2::new(actuator.point_a[0], actuator.point_a[1]);
    let local_b = Vector2::new(actuator.point_b[0], actuator.point_b[1]);
    let p_a = mech_state.body_point_global(&actuator.body_a, &local_a, q);
    let p_b = mech_state.body_point_global(&actuator.body_b, &local_b, q);
    let d = p_b - p_a;
    let length = d.norm();
    if length <= 1e-12 {
        return None;
    }
    let unit = d / length;
    let v_a = mech_state.body_point_velocity(&actuator.body_a, &local_a, q, q_dot);
    let v_b = mech_state.body_point_velocity(&actuator.body_b, &local_b, q, q_dot);
    let dl_dt = (v_b - v_a).dot(&unit);
    if dl_dt.abs() < 1e-6 {
        return None;
    }
    let f = driver_torque * driver_omega / dl_dt;
    if f.is_finite() && f.abs() > 1e-12 {
        Some(f)
    } else {
        None
    }
}

/// Solve for joint reactions, redistributing load through a `LinearActuator`
/// when one is present in sizing mode (`la.force ≈ 0`).
///
/// Returns pass-1 reactions when:
/// - no `LinearActuator` is present, or
/// - the actuator has a non-zero stored `force` (already in `q_forces`;
///   pass-2 would double-count), or
/// - the velocity solve fails, or
/// - the actuator is at a singular pose (`dL/dt ≈ 0`), or
/// - the back-calculated actuator force is not finite or is effectively zero.
///
/// Otherwise returns pass-2 reactions (actuator as prime mover, driver
/// torque collapses to ~0).
///
/// `driver_omega` must be consistent with the mechanism's driver constraint
/// — the same value the velocity solve uses internally via `Φ_t`. For
/// trajectory mode where the prescribed input rate at the current sample
/// differs from the driver's nominal omega, use
/// [`solve_reactions_with_actuator_using_q_dot`] and pass the trajectory's
/// `q_dot_k` and `u_dot_k` directly.
pub fn solve_reactions_with_actuator(
    mech: &Mechanism,
    q: &DVector<f64>,
    t: f64,
    driver_omega: f64,
) -> Result<ReactionSolveResult, LinkageError> {
    let pass1 = solve_statics(mech, q, t)?;
    // q_dot from the constraint's Φ_t (consistent with the constant-speed
    // driver). The trajectory variant injects its own q_dot below.
    match solve_velocity(mech, q, t) {
        Ok(q_dot) => solve_reactions_inner(mech, q, &q_dot, t, driver_omega, pass1),
        Err(_) => Ok(pass1_only_result(mech, &pass1)),
    }
}

/// Variant of [`solve_reactions_with_actuator`] that accepts a pre-computed
/// `q_dot`. Use this in trajectory mode where the prescribed input rate
/// (and therefore q̇) varies sample-to-sample and `solve_velocity`'s
/// constant-speed Φ_t would give the wrong scaling.
///
/// `driver_omega` must match the input rate that produced `q_dot` — for a
/// trajectory sample that's `u_dot_k`.
pub fn solve_reactions_with_actuator_using_q_dot(
    mech: &Mechanism,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    t: f64,
    driver_omega: f64,
) -> Result<ReactionSolveResult, LinkageError> {
    let pass1 = solve_statics(mech, q, t)?;
    solve_reactions_inner(mech, q, q_dot, t, driver_omega, pass1)
}

/// Build a pass-1-only result. Used when the helper short-circuits before
/// running pass-2 (no actuator, non-sizing-mode actuator, velocity solve
/// failed, singular actuator pose, etc.).
fn pass1_only_result(mech: &Mechanism, pass1: &StaticSolveResult) -> ReactionSolveResult {
    let reactions = extract_reactions(mech, pass1);
    let driver_torque = get_driver_reactions(&reactions).first().map(|r| r.effort);
    ReactionSolveResult {
        reactions,
        driver_torque,
        actuator_force: None,
        used_two_pass: false,
        condition_number: pass1.condition_number,
        is_overconstrained: pass1.is_overconstrained,
    }
}

/// Shared core: with pass-1 already solved and q_dot in hand, decide
/// whether to run pass-2 and assemble the final `ReactionSolveResult`.
fn solve_reactions_inner(
    mech: &Mechanism,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    t: f64,
    driver_omega: f64,
    pass1: StaticSolveResult,
) -> Result<ReactionSolveResult, LinkageError> {
    let reactions1 = extract_reactions(mech, &pass1);
    let driver_torque = get_driver_reactions(&reactions1).first().map(|r| r.effort);

    let pass1_only = |reacts: Vec<JointReaction>| ReactionSolveResult {
        reactions: reacts,
        driver_torque,
        actuator_force: None,
        used_two_pass: false,
        condition_number: pass1.condition_number,
        is_overconstrained: pass1.is_overconstrained,
    };

    // Find the first sizing-mode `LinearActuator`. A non-zero stored force
    // is treated as a known external load already baked into pass-1's
    // q_forces, so pass-2 is skipped to avoid double-counting.
    let actuator: Option<LinearActuatorElement> = mech.forces().iter().find_map(|f| match f {
        ForceElement::LinearActuator(act) if act.force.abs() < 1e-12 => Some(act.clone()),
        _ => None,
    });
    let act = match actuator {
        Some(a) => a,
        None => return Ok(pass1_only(reactions1)),
    };

    let drv_torque = match driver_torque {
        Some(t) => t,
        None => return Ok(pass1_only(reactions1)),
    };

    let actuator_force = match compute_actuator_force_from_power_balance(
        mech,
        q,
        q_dot,
        drv_torque,
        driver_omega,
        &act,
    ) {
        Some(f) => f,
        None => return Ok(pass1_only(reactions1)),
    };

    // Pass 2: inject the back-calculated actuator force into the
    // generalised force vector, then re-solve `Φ_qᵀ λ = -q_new`.
    let mut act_mod = act.clone();
    act_mod.force = actuator_force;
    let mech_state = mech.state();
    let q_dot_zero = DVector::zeros(mech_state.n_coords());
    let q_actuator = evaluate_linear_actuator(&act_mod, mech_state, q, &q_dot_zero);
    let q_new = &pass1.q_forces + &q_actuator;
    let phi_q = assemble_jacobian(mech, q, t);
    let rhs = -&q_new;
    let lambdas2 = phi_q
        .transpose()
        .svd(true, true)
        .solve(&rhs, 1e-14)
        .map_err(|_| LinkageError::SvdSolveFailed)?;

    let pass2 = StaticSolveResult {
        lambdas: lambdas2,
        q_forces: q_new,
        residual_norm: 0.0,
        is_overconstrained: false,
        condition_number: 0.0,
    };
    let reactions2 = extract_reactions(mech, &pass2);

    Ok(ReactionSolveResult {
        reactions: reactions2,
        driver_torque,
        actuator_force: Some(actuator_force),
        used_two_pass: true,
        condition_number: pass1.condition_number,
        is_overconstrained: pass1.is_overconstrained,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use crate::forces::elements::{ForceElement, GravityElement, LinearActuatorElement};
    use crate::solver::kinematics::solve_position;
    use std::f64::consts::PI;

    /// Build a crank-rocker 4-bar (a=1, b=4, c=3, d=4) with gravity. Used
    /// as the no-actuator baseline.
    fn build_fourbar() -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
        let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();

        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D").unwrap();
        mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();

        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.build().unwrap();
        mech
    }

    /// Same 4-bar, plus a LinearActuator between the coupler midpoint and
    /// ground midpoint with the given stored force. `force = 0.0` puts the
    /// actuator in sizing mode (helper should run pass-2). Non-zero stored
    /// force should disable pass-2 (helper guards against double-counting).
    fn build_fourbar_with_actuator(stored_force: f64) -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
        let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();

        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D").unwrap();
        mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();

        mech.add_force(ForceElement::Gravity(GravityElement::default()));

        // Actuator between mid-coupler (in coupler-local frame) and the
        // midpoint between the two ground pivots (in ground-local frame).
        // Picks an attachment that is NOT collinear with any joint axis,
        // so dl_dt is finite at the test pose and pass-2 actually runs.
        let act = LinearActuatorElement {
            body_a: "coupler".to_string(),
            point_a: [1.5, 0.0],
            point_a_name: None,
            body_b: "ground".to_string(),
            point_b: [2.0, 0.0],
            point_b_name: None,
            force: stored_force,
            speed_limit: 0.0,
            stroke_min: 0.0,
            stroke_max: 0.0,
            end_stop_stiffness: 0.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.0,
        };
        mech.add_force(ForceElement::LinearActuator(act));
        mech.build().unwrap();
        mech
    }

    /// Seed pose for the test 4-bar at θ_2 = π/3 — the same pose used by
    /// `solver::statics::tests` so we know it's a clean assembly.
    fn seed_pose_at_60deg(mech: &Mechanism) -> DVector<f64> {
        let state = mech.state();
        let angle = PI / 3.0;
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("coupler", &mut q0, angle.cos(), angle.sin(), 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);
        let pos = solve_position(mech, &q0, angle, 1e-10, 50)
            .expect("position solve");
        assert!(pos.converged);
        pos.q
    }

    #[test]
    fn no_actuator_returns_pass_one_reactions() {
        // A 4-bar without a LinearActuator: helper returns pass-1 reactions
        // and used_two_pass=false. Reactions match a direct extract_reactions.
        let mech = build_fourbar();
        let q = seed_pose_at_60deg(&mech);

        let result = solve_reactions_with_actuator(&mech, &q, PI / 3.0, 1.0)
            .expect("statics should converge");
        assert!(!result.used_two_pass, "no actuator → no pass-2");
        assert!(result.actuator_force.is_none());

        let direct = extract_reactions(&mech, &solve_statics(&mech, &q, PI / 3.0).unwrap());
        assert_eq!(result.reactions.len(), direct.len());
        for (lhs, rhs) in result.reactions.iter().zip(direct.iter()) {
            assert_eq!(lhs.joint_id, rhs.joint_id);
            assert!(
                (lhs.resultant - rhs.resultant).abs() < 1e-9,
                "{} resultant differs: {} vs {}", lhs.joint_id, lhs.resultant, rhs.resultant,
            );
        }
    }

    #[test]
    fn actuator_in_sizing_mode_runs_pass_two() {
        // LinearActuator with stored force=0: helper runs pass-2 and the
        // resulting joint reactions differ from pass-1 (the actuator now
        // carries the load, so joints in its chain see different forces).
        let mech = build_fourbar_with_actuator(0.0);
        let q = seed_pose_at_60deg(&mech);

        let result = solve_reactions_with_actuator(&mech, &q, PI / 3.0, 1.0)
            .expect("statics should converge");
        assert!(result.used_two_pass, "sizing-mode actuator → pass-2");
        let f_act = result.actuator_force.expect("actuator force computed");
        assert!(f_act.is_finite() && f_act.abs() > 1e-9, "f_act = {}", f_act);

        let pass1 = extract_reactions(&mech, &solve_statics(&mech, &q, PI / 3.0).unwrap());
        let pass1_joints: Vec<_> = pass1.iter().filter(|r| r.n_equations > 1).collect();
        let pass2_joints: Vec<_> = result.reactions.iter().filter(|r| r.n_equations > 1).collect();
        assert_eq!(pass1_joints.len(), pass2_joints.len());

        // At least one joint reaction should shift noticeably under the
        // pass-1 → pass-2 prime-mover swap. That's the whole point.
        let max_diff = pass1_joints.iter().zip(pass2_joints.iter())
            .map(|(a, b)| (a.resultant - b.resultant).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 1e-3,
            "pass-2 reactions should differ from pass-1 (max diff = {})", max_diff,
        );
    }

    #[test]
    fn actuator_with_known_force_skips_pass_two() {
        // LinearActuator with stored force ≠ 0: helper must NOT run pass-2
        // (would double-count the actuator's q_forces contribution).
        // Reactions match a direct pass-1 solve.
        let mech = build_fourbar_with_actuator(50.0);
        let q = seed_pose_at_60deg(&mech);

        let result = solve_reactions_with_actuator(&mech, &q, PI / 3.0, 1.0)
            .expect("statics should converge");
        assert!(!result.used_two_pass, "non-zero la.force → pass-2 skipped to avoid double-counting");
        assert!(result.actuator_force.is_none());

        let direct = extract_reactions(&mech, &solve_statics(&mech, &q, PI / 3.0).unwrap());
        for (lhs, rhs) in result.reactions.iter().zip(direct.iter()) {
            assert_eq!(lhs.joint_id, rhs.joint_id);
            assert!((lhs.resultant - rhs.resultant).abs() < 1e-9);
        }
    }
}
