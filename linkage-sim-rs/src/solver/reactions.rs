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

    /// FBD-validated reactions test for 4-bar + LinearActuator in sizing mode
    /// at pose θ_2 = π/3. Builds the 9×9 Cartesian equilibrium system by hand
    /// (force balance + moment balance per body), solves it via nalgebra,
    /// and asserts the simulator's pass-2 output matches.
    ///
    /// Why this catches more than the existing self-consistency tests: the
    /// matrix below is constructed from physical equilibrium equations
    /// (ΣF = 0, ΣM_CG = 0) written directly in Cartesian coordinates —
    /// not from the constraint Jacobian Φ_q the solver uses internally.
    /// A sign-flipped Jacobian row, a wrong mass-to-Q mapping, a
    /// pass-2 double-count, or a swap of actuator endpoints would produce
    /// reactions that satisfy `Φ_qᵀλ = -Q` (the solver's internal check)
    /// while violating the per-body equilibrium written here.
    ///
    /// ───── Geometry derivation ────────────────────────────────────────
    ///
    /// Config: a=1, b=3, c=2, d=4 (Grashof crank-rocker), gravity on,
    /// LinearActuator from coupler-midpoint to ground-midpoint, force=0.
    ///
    /// At θ_2 = π/3:
    ///   A = (0, 0)   (J1 location, crank-ground pivot)
    ///   B = (cos π/3, sin π/3) = (1/2, √3/2)   (J2 location)
    ///   D = (4, 0)   (J4 location, rocker-ground pivot)
    ///
    /// Loop closure (3 cos θ_c − 2 cos θ_r = 7/2, 3 sin θ_c − 2 sin θ_r = −√3/2):
    ///   Squaring & adding gives cos(θ_c − θ_r) = 0, so θ_c = θ_r − π/2
    ///   (open branch — matches the seed pose).
    ///   Substituting gives the 2×2 linear system
    ///     [−2  3][cos θ_r]   [7/2  ]
    ///     [ 3  2][sin θ_r] = [√3/2]
    ///   det = −13, so
    ///     cos θ_r = (3√3 − 14) / 26
    ///     sin θ_r = (21 + 2√3) / 26
    ///   And:
    ///     C_x = 4 + 2 cos θ_r = (38 + 3√3) / 13 ≈ 3.3228
    ///     C_y = 2 sin θ_r     = (21 + 2√3) / 13 ≈ 1.8819
    ///
    /// CGs (uniform-bar bodies, CG at midpoint):
    ///   CG_crank   = A/2 + B/2        = (1/4, √3/4)
    ///   CG_coupler = (B + C) / 2      ≈ (1.9114, 1.3739)
    ///   CG_rocker  = (C + D) / 2      ≈ (3.6614, 0.9410)
    ///
    /// Actuator endpoints in world frame:
    ///   point_a (coupler-local (1.5, 0)) = coupler midpoint = CG_coupler
    ///   point_b (ground-local (2, 0))    = (2, 0)
    ///   Δ = p_b − p_a ≈ (0.0886, −1.3739),   |Δ| ≈ 1.3767
    ///   unit u = Δ / |Δ| ≈ (0.0644, −0.9979)
    /// Important consequence: p_a = CG_coupler, so the actuator force at
    /// p_a contributes ZERO moment about the coupler CG. This drops a
    /// term from the coupler moment equation.
    ///
    /// ───── Sign convention for force_global ───────────────────────────
    ///
    /// `extract_reactions` reports the Lagrange multipliers as
    /// `force_global = (λ_x, λ_y)`. For a revolute constraint
    /// Φ = r_i − r_j = 0, the static equilibrium is Φ_qᵀλ = −Q, so the
    /// constraint imposes +λ on body_i and −λ on body_j. We adopt:
    ///   R_Jk ≡ force_global at joint Jk = force on body_i of that joint.
    ///   Force on body_j of that joint = −R_Jk (Newton's 3rd law).
    ///
    /// Joints, with body_i listed first per `add_revolute_joint(...)`:
    ///   J1: body_i = ground,   body_j = crank   →  on crank:    −R_J1
    ///   J2: body_i = crank,    body_j = coupler →  on crank:    +R_J2; on coupler: −R_J2
    ///   J3: body_i = coupler,  body_j = rocker  →  on coupler: +R_J3; on rocker:  −R_J3
    ///   J4: body_i = ground,   body_j = rocker  →  on rocker:  −R_J4
    ///
    /// ───── Equilibrium equations ──────────────────────────────────────
    ///
    /// Crank: gravity (0, −m_c·g) at CG_crank.
    ///   ΣFx: −R_J1x + R_J2x = 0
    ///   ΣFy: −R_J1y + R_J2y − m_c·g = 0  →  R_J2y − R_J1y = m_c·g
    ///   ΣM_CG: moment of (−R_J1) at A=(0,0) plus moment of (+R_J2) at B.
    ///     r_A−CG = (−1/4, −√3/4), F = −R_J1: m = (−1/4)(−R_J1y) − (−√3/4)(−R_J1x)
    ///                                          = (1/4) R_J1y − (√3/4) R_J1x
    ///     r_B−CG = (1/4, √3/4),   F =  R_J2: m = (1/4) R_J2y − (√3/4) R_J2x
    ///   Multiplying by 4:
    ///     (R_J1y + R_J2y) − √3 (R_J1x + R_J2x) = 0
    ///
    /// Coupler: gravity (0, −m_b·g) at CG_coupler. Actuator: per
    /// `evaluate_linear_actuator`, the force on body_a (the coupler here)
    /// is `−F_act·u` (with `u = (p_b − p_a)/|·|` pointing from a to b),
    /// so positive F_act = extension = pushing the bodies apart. This is
    /// the same sign convention `actuator_force` carries through the rest
    /// of the codebase.
    ///   ΣFx: −R_J2x + R_J3x − u_x F_act = 0
    ///   ΣFy: −R_J2y + R_J3y − u_y F_act − m_b·g = 0
    ///   ΣM_CG: moments of (−R_J2) at B and (+R_J3) at C.
    ///     r_B−CGc = (Bx − CGcx, By − CGcy), F = −R_J2:
    ///       m = (Bx − CGcx)(−R_J2y) − (By − CGcy)(−R_J2x)
    ///         = −(Bx − CGcx) R_J2y + (By − CGcy) R_J2x
    ///     r_C−CGc = (Cx − CGcx, Cy − CGcy), F = +R_J3:
    ///       m = (Cx − CGcx) R_J3y − (Cy − CGcy) R_J3x
    ///
    /// Rocker: gravity (0, −m_r·g) at CG_rocker.
    ///   ΣFx: −R_J3x − R_J4x = 0
    ///   ΣFy: −R_J3y − R_J4y − m_r·g = 0  →  R_J3y + R_J4y = −m_r·g
    ///   ΣM_CG: moments of (−R_J3) at C and (−R_J4) at D.
    ///     m_J3 = −(Cx − CGrx) R_J3y + (Cy − CGry) R_J3x
    ///     m_J4 = −(Dx − CGrx) R_J4y + (Dy − CGry) R_J4x
    ///
    /// ───── 9×9 linear system ──────────────────────────────────────────
    ///
    /// Unknowns x = [R_J1x, R_J1y, R_J2x, R_J2y, R_J3x, R_J3y, R_J4x, R_J4y, F_act]ᵀ
    ///
    /// The code below builds A and b from these equations and solves Ax = b.
    /// The solution is the independent FBD ground truth.
    #[test]
    fn fbd_validates_pass2_reactions_at_60deg() {
        use nalgebra::DMatrix;
        let mech = build_fourbar_with_actuator(0.0);
        let q = seed_pose_at_60deg(&mech);

        const G: f64 = 9.81;
        let m_crank = 2.0_f64;
        let m_coupler = 3.0_f64;
        let m_rocker = 2.0_f64;
        let sqrt3 = 3f64.sqrt();

        // Joint positions in world frame at θ_2 = π/3 (derived above).
        let bx = 0.5_f64;
        let by = sqrt3 / 2.0;
        let cx = (38.0 + 3.0 * sqrt3) / 13.0;
        let cy = (21.0 + 2.0 * sqrt3) / 13.0;
        let dx = 4.0_f64;
        let dy = 0.0_f64;

        // CGs (midpoints of each uniform bar).
        let cgcx = (bx + cx) * 0.5;
        let cgcy = (by + cy) * 0.5;
        let cgrx = (cx + dx) * 0.5;
        let cgry = (cy + dy) * 0.5;

        // Actuator: p_a = coupler local (1.5, 0) = coupler midpoint = CG_coupler.
        // p_b = ground local (2, 0) = (2, 0) world.
        let p_ax = cgcx;
        let p_ay = cgcy;
        let p_bx = 2.0_f64;
        let p_by = 0.0_f64;
        let dx_act = p_bx - p_ax;
        let dy_act = p_by - p_ay;
        let len_act = (dx_act * dx_act + dy_act * dy_act).sqrt();
        let ux = dx_act / len_act;
        let uy = dy_act / len_act;

        // Build the 9×9 system Ax = b.
        let mut a = DMatrix::<f64>::zeros(9, 9);
        let mut b = nalgebra::DVector::<f64>::zeros(9);

        // Column index legend:
        //   0: R_J1x  1: R_J1y  2: R_J2x  3: R_J2y
        //   4: R_J3x  5: R_J3y  6: R_J4x  7: R_J4y
        //   8: F_act

        // Row 0 — Crank ΣFx: -R_J1x + R_J2x = 0
        a[(0, 0)] = -1.0;
        a[(0, 2)] = 1.0;

        // Row 1 — Crank ΣFy: -R_J1y + R_J2y = m_crank * g
        a[(1, 1)] = -1.0;
        a[(1, 3)] = 1.0;
        b[1] = m_crank * G;

        // Row 2 — Crank ΣM_CG: (R_J1y + R_J2y) − √3·(R_J1x + R_J2x) = 0
        a[(2, 0)] = -sqrt3;
        a[(2, 1)] = 1.0;
        a[(2, 2)] = -sqrt3;
        a[(2, 3)] = 1.0;

        // Row 3 — Coupler ΣFx: -R_J2x + R_J3x − u_x F_act = 0
        // (force on body_a is `−F_act·u` per evaluate_linear_actuator)
        a[(3, 2)] = -1.0;
        a[(3, 4)] = 1.0;
        a[(3, 8)] = -ux;

        // Row 4 — Coupler ΣFy: -R_J2y + R_J3y − u_y F_act = m_coupler * g
        a[(4, 3)] = -1.0;
        a[(4, 5)] = 1.0;
        a[(4, 8)] = -uy;
        b[4] = m_coupler * G;

        // Row 5 — Coupler ΣM_CG:
        //   -(Bx − CGcx) R_J2y + (By − CGcy) R_J2x + (Cx − CGcx) R_J3y - (Cy − CGcy) R_J3x = 0
        let b_dx = bx - cgcx;
        let b_dy = by - cgcy;
        let c_dx = cx - cgcx;
        let c_dy = cy - cgcy;
        a[(5, 2)] = b_dy; // R_J2x coeff
        a[(5, 3)] = -b_dx; // R_J2y coeff
        a[(5, 4)] = -c_dy; // R_J3x coeff
        a[(5, 5)] = c_dx; // R_J3y coeff

        // Row 6 — Rocker ΣFx: -R_J3x - R_J4x = 0
        a[(6, 4)] = -1.0;
        a[(6, 6)] = -1.0;

        // Row 7 — Rocker ΣFy: -R_J3y - R_J4y = m_rocker * g
        a[(7, 5)] = -1.0;
        a[(7, 7)] = -1.0;
        b[7] = m_rocker * G;

        // Row 8 — Rocker ΣM_CG:
        //   -(Cx − CGrx) R_J3y + (Cy − CGry) R_J3x - (Dx − CGrx) R_J4y + (Dy − CGry) R_J4x = 0
        let cr_dx = cx - cgrx;
        let cr_dy = cy - cgry;
        let dr_dx = dx - cgrx;
        let dr_dy = dy - cgry;
        a[(8, 4)] = cr_dy; // R_J3x coeff
        a[(8, 5)] = -cr_dx; // R_J3y coeff
        a[(8, 6)] = dr_dy; // R_J4x coeff
        a[(8, 7)] = -dr_dx; // R_J4y coeff

        let lu = a.full_piv_lu();
        let x = lu
            .solve(&b)
            .expect("FBD linear system should be non-singular at this pose");

        let exp_r_j1 = (x[0], x[1]);
        let exp_r_j2 = (x[2], x[3]);
        let exp_r_j3 = (x[4], x[5]);
        let exp_r_j4 = (x[6], x[7]);
        let exp_f_act = x[8];

        // Now run the simulator and compare.
        let result = solve_reactions_with_actuator(&mech, &q, PI / 3.0, 1.0)
            .expect("statics should converge in sizing mode");
        assert!(result.used_two_pass, "sizing-mode actuator → pass-2");

        // Pass-2's driver torque should be ~0 by construction; if it's
        // not, the back-substituted F_act is wrong and the comparison
        // below will fail anyway, but flagging it gives a clearer
        // diagnostic.
        let drv_lambda = result
            .reactions
            .iter()
            .find(|r| r.n_equations == 1)
            .map(|r| r.effort)
            .expect("driver lambda present");
        assert!(
            drv_lambda.abs() < 1e-6,
            "pass-2 driver torque should be ~0 (actuator drives the load); got {}",
            drv_lambda,
        );

        let sim_f_act = result.actuator_force.expect("pass-2 produces F_act");
        let get = |id: &str| -> (f64, f64) {
            let r = result
                .reactions
                .iter()
                .find(|r| r.joint_id == id)
                .unwrap_or_else(|| panic!("joint {} missing in reactions", id));
            (r.force_global[0], r.force_global[1])
        };
        let sim_r_j1 = get("J1");
        let sim_r_j2 = get("J2");
        let sim_r_j3 = get("J3");
        let sim_r_j4 = get("J4");

        // Tolerance: 1e-4 N is well below any physically meaningful load
        // (gravity totals ~70 N on this mechanism). Statics solver has
        // a 1e-14 residual tolerance and the FBD here is exact to f64.
        let tol = 1e-4;
        let diff = |label: &str, sim: (f64, f64), exp: (f64, f64)| {
            assert!(
                (sim.0 - exp.0).abs() < tol && (sim.1 - exp.1).abs() < tol,
                "{} mismatch: sim = ({}, {}), FBD = ({}, {})",
                label, sim.0, sim.1, exp.0, exp.1,
            );
        };
        diff("J1", sim_r_j1, exp_r_j1);
        diff("J2", sim_r_j2, exp_r_j2);
        diff("J3", sim_r_j3, exp_r_j3);
        diff("J4", sim_r_j4, exp_r_j4);
        assert!(
            (sim_f_act - exp_f_act).abs() < tol,
            "F_act mismatch: sim = {}, FBD = {}",
            sim_f_act, exp_f_act,
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
