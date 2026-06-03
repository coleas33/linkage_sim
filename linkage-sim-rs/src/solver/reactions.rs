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
    /// Generalised-coordinate residual `‖Φ_qᵀλ + Q‖` for the final solve.
    /// DIAGNOSTIC ONLY — near-tautological because λ is solved to make it
    /// zero (see `body_equilibrium_residual`). Use `validation` for the
    /// trustworthiness verdict, not this.
    pub residual_norm: f64,
    /// Trustworthiness verdict from the INDEPENDENT per-body equilibrium
    /// check (plus condition-number gate and, for pass-2, driver-collapse).
    /// This — not `residual_norm` — is what the GUI badge and `is_valid()`
    /// report.
    pub validation: ValidationState,
}

impl ReactionSolveResult {
    /// Whether the returned reactions satisfy Cartesian equilibrium of
    /// the applied forces to a tight absolute threshold. Used by:
    /// - `solve_reactions_with_actuator`'s `debug_assert!` (catches
    ///   solver regressions in dev builds before they reach the GUI),
    /// - the GUI property panel's green/red validation badge,
    /// - any caller that wants a quick "do I trust these numbers"
    ///   check before using the reactions downstream.
    ///
    /// Tolerance: `1e-6 N` absolute on `residual_norm`, plus (when
    /// `used_two_pass`) a `1e-6 N·m` check that the pass-2 driver
    /// lambda has actually collapsed (the point of pass-2). On
    /// well-conditioned systems both are typically ~1e-12.
    pub fn is_valid(&self) -> bool {
        !matches!(self.validation, ValidationState::Failed)
    }

    /// Tri-state validation for display and assertions:
    /// - `Verified`   — independent per-body equilibrium holds (genuinely checked).
    /// - `Unverified` — the mechanism has element/joint types the independent
    ///   check does not yet model; the SVD residual converged but we make NO
    ///   physics claim. Callers must NOT present this as "passed".
    /// - `Failed`     — independent equilibrium violated, or the linear solve
    ///   didn't converge, or the pose is too ill-conditioned to trust.
    pub fn validation(&self) -> ValidationState {
        self.validation
    }
}

/// Result of the independent (non-circular) reaction validation. See
/// `body_equilibrium_residual` for why the plain `residual_norm` is not
/// sufficient on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationState {
    Verified,
    Unverified,
    Failed,
}

/// Condition-number ceiling above which a pose is treated as too
/// ill-conditioned to trust the reactions (rank-deficient / singular).
/// The near-singular sweep showed cond ~1e2 at θ=0.9π (fine) climbing to
/// ~4.7e5 at the θ=π toggle (reactions physically unbounded); 1e8 leaves
/// generous margin for legitimate near-toggle work while rejecting the
/// genuinely singular.
const CONDITION_CEILING: f64 = 1e8;

/// Relative-residual threshold for the independent per-body equilibrium
/// check. Correct solves sit at ~1e-15 (machine eps); a real sign/frame/
/// Jacobian error pushes this to O(0.1–1). 1e-6 is a safe separating line.
const EQUILIBRIUM_REL_TOL: f64 = 1e-6;

/// Independent per-body Cartesian Newton-Euler equilibrium residual — the
/// NON-circular validation of the reaction solve.
///
/// `ReactionSolveResult::residual_norm` (‖Φ_qᵀλ + Q‖) is near-tautological:
/// λ is produced by SVD-solving exactly that system, so for a square,
/// full-rank Jacobian the residual is ~machine-eps **regardless of whether
/// the applied generalised force Q — and hence the reactions — is physically
/// correct**. A mutation test (swapping the moment arm in
/// `forces::helpers::point_force_to_q`) confirmed `residual_norm` stayed at
/// ~1e-14 while the reactions were wrong; a spurious force injected on a
/// non-driver body DOF likewise left both the residual AND the pass-2
/// driver-torque-collapse check passing while a joint reaction was 2× off.
///
/// This function instead checks equilibrium in a DIFFERENT basis: for each
/// moving body it sums, in world-frame Cartesian coordinates, the joint
/// reaction forces (from `force_global`, with Newton's-3rd-law signs, at the
/// joint world positions), the driver couple, and the applied forces
/// (gravity, linear actuator) computed directly from geometry — NOT routed
/// through `point_force_to_q`. A body in static equilibrium satisfies
/// ΣF = 0 and ΣM = 0, so any sign/frame/Jacobian error the generalised
/// residual hides surfaces here as a non-zero net force or moment.
///
/// Returns `Some(relative_residual)` when every force element and joint type
/// in `mech` is modeled (currently: Revolute joints + the rotational Driver,
/// Gravity, and a single LinearActuator). Returns `None` when the mechanism
/// contains anything else (force zones, springs, dampers, fixed/prismatic
/// joints, multiple actuators) — callers MUST treat `None` as "unverified",
/// never as pass or fail.
///
/// `eff_actuator_force` is the actuator's effective axial force at this pose
/// (the back-solved value in pass-2, or stored `la.force` in pass-1); `None`
/// when no actuator acts.
pub fn body_equilibrium_residual(
    mech: &Mechanism,
    q: &DVector<f64>,
    t: f64,
    reactions: &[JointReaction],
    eff_actuator_force: Option<f64>,
) -> Option<f64> {
    use crate::core::constraint::{Constraint, JointConstraint};
    use std::collections::HashMap;

    let state = mech.state();

    // ── Bail to None on anything this check can't independently model ──
    for j in mech.joints() {
        if !matches!(j, JointConstraint::Revolute(_)) {
            return None;
        }
    }
    let mut gravity: Option<Vector2<f64>> = None;
    let mut actuator: Option<LinearActuatorElement> = None;
    for f in mech.forces() {
        match f {
            ForceElement::Gravity(g) => {
                gravity = Some(Vector2::new(g.g_vector[0], g.g_vector[1]))
            }
            ForceElement::LinearActuator(a) => {
                if actuator.is_some() {
                    return None; // >1 actuator: not modeled
                }
                actuator = Some(a.clone());
            }
            // Applied in pass 2 below (need the per-body accumulators).
            ForceElement::ForceZone(_)
            | ForceElement::ExternalForce(_)
            | ForceElement::ExternalTorque(_) => {}
            // Velocity-/state-dependent or rotary elements not yet modeled
            // by this static check: spring, damper, gas spring, motor,
            // bearing friction, joint limit, torsion spring, rotary damper.
            _ => return None,
        }
    }

    // ── Per-body accumulators (world frame) ──────────────────────────────
    let mut net_f: HashMap<&str, Vector2<f64>> = HashMap::new();
    let mut net_m: HashMap<&str, f64> = HashMap::new();
    let mut cg_world: HashMap<&str, Vector2<f64>> = HashMap::new();
    let mut force_scale = 1.0_f64;
    let mut char_len = 1e-9_f64;

    for (id, body) in mech.bodies() {
        if state.is_ground(id) {
            continue;
        }
        let (bx, by, bth) = state.get_pose(id, q);
        let (c, s) = (bth.cos(), bth.sin());
        let cg = Vector2::new(
            bx + c * body.cg_local.x - s * body.cg_local.y,
            by + s * body.cg_local.x + c * body.cg_local.y,
        );
        cg_world.insert(id.as_str(), cg);
        net_f.insert(id.as_str(), Vector2::zeros());
        net_m.insert(id.as_str(), 0.0);

        // Gravity acts at the CG (zero moment about CG) on every moving body.
        if let Some(g) = gravity {
            let fg = g * body.mass;
            *net_f.get_mut(id.as_str()).unwrap() += fg;
            force_scale = force_scale.max(fg.norm());
        }
    }

    // Nested helper: add a world-frame force `f` applied at `world_pt` to a
    // (possibly ground — then ignored) body, accumulating net force and
    // moment about that body's CG, and tracking the moment-arm scale.
    fn accum(
        net_f: &mut HashMap<&str, Vector2<f64>>,
        net_m: &mut HashMap<&str, f64>,
        cg_world: &HashMap<&str, Vector2<f64>>,
        char_len: &mut f64,
        body: &str,
        world_pt: Vector2<f64>,
        f: Vector2<f64>,
    ) {
        let Some(cg) = cg_world.get(body) else { return };
        let r = world_pt - cg;
        *char_len = char_len.max(r.norm());
        *net_f.get_mut(body).unwrap() += f;
        *net_m.get_mut(body).unwrap() += r.x * f.y - r.y * f.x;
    }

    // ── Joint reactions + driver couple ──────────────────────────────────
    for jr in reactions {
        if jr.n_equations == 1 {
            // Driver: pure couple. +effort on the driven body, −effort on
            // the reference body. Ground entries are silently dropped.
            if let Some(m) = net_m.get_mut(jr.body_j_id.as_str()) {
                *m += jr.effort;
            }
            if let Some(m) = net_m.get_mut(jr.body_i_id.as_str()) {
                *m -= jr.effort;
            }
            force_scale = force_scale.max(jr.effort.abs() / char_len.max(1e-9));
            continue;
        }
        // Joint: locate it to get its world position.
        let Some(joint) = mech.joints().iter().find(|j| j.id() == jr.joint_id) else {
            return None; // a non-joint, non-driver constraint we don't model
        };
        let pt_i = joint.point_i_local();
        let world_pt = state.body_point_global(&jr.body_i_id, &pt_i, q);
        let f = Vector2::new(jr.force_global[0], jr.force_global[1]);
        force_scale = force_scale.max(f.norm());
        // Newton's 3rd law: +f on body_i, −f on body_j. `world_pt` is body_i's
        // joint point; at a converged pose body_j's coincides, so reusing it
        // is exact. Off-manifold it would inject a spurious moment ∝ Φ_revolute,
        // but that only ADDS residual (fail-safe: never masks a wrong reaction)
        // and is negligible at the 1e-10 convergence every caller uses.
        accum(&mut net_f, &mut net_m, &cg_world, &mut char_len, &jr.body_i_id, world_pt, f);
        accum(&mut net_f, &mut net_m, &cg_world, &mut char_len, &jr.body_j_id, world_pt, -f);
    }

    // ── Linear actuator applied force (from geometry, not point_force_to_q) ──
    if let (Some(a), Some(force)) = (actuator.as_ref(), eff_actuator_force) {
        let pa_local = Vector2::new(a.point_a[0], a.point_a[1]);
        let pb_local = Vector2::new(a.point_b[0], a.point_b[1]);
        let p_a = state.body_point_global(&a.body_a, &pa_local, q);
        let p_b = state.body_point_global(&a.body_b, &pb_local, q);
        let d = p_b - p_a;
        let len = d.norm();
        if len > 1e-12 {
            let u = d / len;
            // evaluate_linear_actuator: force_on_a = −F·u, force_on_b = +F·u.
            accum(&mut net_f, &mut net_m, &cg_world, &mut char_len, &a.body_a, p_a, -force * u);
            accum(&mut net_f, &mut net_m, &cg_world, &mut char_len, &a.body_b, p_b, force * u);
            force_scale = force_scale.max(force.abs());
        }
    }

    // ── Force zones, external forces, external torques (pass 2) ──────────
    // Each is rebuilt from geometry / element data, NOT via point_force_to_q.
    // Independence is precise for the moment-arm PROJECTION: production routes
    // the moment through point_force_to_q, while this computes it directly in
    // world frame (accum's r×f), so a point_force_to_q sign/frame bug IS caught.
    // For force zones, two decisions are SHARED with production, not re-derived:
    // the binary overlap gate (polygon_area < 1e-15) and the unpinned-app-point
    // centroid (polygon_centroid). A bug in those shared geometry primitives
    // would corrupt both sides identically and cancel — narrow blast radius
    // (wrong app-POINT or overlap DECISION only, not a moment-arm projection).
    for fe in mech.forces() {
        match fe {
            ForceElement::ForceZone(fz) => {
                use crate::geometry::{
                    body_rect_to_world, clip_polygon_to_aabb, polygon_area, polygon_centroid,
                };
                let Some(body) = mech.bodies().get(&fz.body_id) else { continue };
                let Some(geo) = body.geometry.as_ref() else { continue };
                let (bx, by, bth) = state.get_pose(&fz.body_id, q);
                let corners = body_rect_to_world(bx, by, bth, geo.width, geo.height, &geo.offset);
                let zmin = Vector2::new(fz.zone_min[0], fz.zone_min[1]);
                let zmax = Vector2::new(fz.zone_max[0], fz.zone_max[1]);
                let clipped = clip_polygon_to_aabb(&corners, &zmin, &zmax);
                // Binary overlap: any contact → full force (matches
                // evaluate_force_zone).
                if polygon_area(&clipped) < 1e-15 {
                    continue;
                }
                let force = Vector2::new(fz.force[0], fz.force[1]);
                // World application point: pinned body-local override, else
                // the overlap-polygon centroid.
                let app = if let Some(lp) = fz.body_local_app_point {
                    let (c, s) = (bth.cos(), bth.sin());
                    Vector2::new(
                        bx + c * lp[0] - s * lp[1],
                        by + s * lp[0] + c * lp[1],
                    )
                } else {
                    polygon_centroid(&clipped)
                };
                accum(&mut net_f, &mut net_m, &cg_world, &mut char_len, &fz.body_id, app, force);
                force_scale = force_scale.max(force.norm());
            }
            ForceElement::ExternalForce(ef) => {
                let factor = ef.modulation.factor(t);
                let force = Vector2::new(ef.force[0] * factor, ef.force[1] * factor);
                let lp = Vector2::new(ef.local_point[0], ef.local_point[1]);
                let app = state.body_point_global(&ef.body_id, &lp, q);
                accum(&mut net_f, &mut net_m, &cg_world, &mut char_len, &ef.body_id, app, force);
                force_scale = force_scale.max(force.norm());
            }
            ForceElement::ExternalTorque(et) => {
                let factor = et.modulation.factor(t);
                if let Some(m) = net_m.get_mut(et.body_id.as_str()) {
                    *m += et.torque * factor;
                }
            }
            _ => {}
        }
    }

    // ── Worst-body relative residual ─────────────────────────────────────
    let inv_scale = 1.0 / force_scale.max(1e-9);
    let mut worst = 0.0_f64;
    for id in net_f.keys() {
        let fr = net_f[id].norm();
        let mr = net_m[id].abs() / char_len.max(1e-9);
        let r = (fr + mr) * inv_scale;
        if !r.is_finite() {
            return Some(f64::INFINITY);
        }
        worst = worst.max(r);
    }
    Some(worst)
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
        Err(_) => Ok(pass1_only_result(mech, q, t, &pass1)),
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

/// The actuator's effective axial force for a PASS-1 result: whatever the
/// (single) actuator element carries as-is (`la.force`), or `None` if no
/// actuator is present. Pass-1 applies the stored force directly, so this
/// is what balances the pass-1 reactions and what the independent check
/// must use.
fn pass1_eff_actuator_force(mech: &Mechanism) -> Option<f64> {
    mech.forces().iter().find_map(|f| match f {
        ForceElement::LinearActuator(a) => Some(a.force),
        _ => None,
    })
}

/// Compute the trustworthiness verdict from the independent per-body
/// equilibrium check, the condition-number gate, and (for pass-2) the
/// driver-collapse expectation.
fn compute_validation(
    mech: &Mechanism,
    q: &DVector<f64>,
    t: f64,
    reactions: &[JointReaction],
    eff_actuator_force: Option<f64>,
    condition_number: f64,
    expect_driver_collapse: bool,
    driver_torque_pass1: Option<f64>,
) -> ValidationState {
    if !condition_number.is_finite() || condition_number > CONDITION_CEILING {
        return ValidationState::Failed;
    }
    // Pass-2 prime-mover intent: the rotational driver torque must collapse.
    // This is the check (relative to the pass-1 torque it started from) that
    // catches a wrong back-solved actuator force — the per-body equilibrium
    // alone cannot, since the reactions self-consistently balance whatever
    // (possibly wrong) actuator force was injected.
    if expect_driver_collapse {
        if let Some(eff) = reactions.iter().find(|r| r.n_equations == 1).map(|r| r.effort) {
            let reference = driver_torque_pass1.unwrap_or(0.0).abs().max(1.0);
            if !eff.is_finite() || eff.abs() > 1e-6 * reference {
                return ValidationState::Failed;
            }
        }
    }
    match body_equilibrium_residual(mech, q, t, reactions, eff_actuator_force) {
        Some(r) if r.is_finite() && r < EQUILIBRIUM_REL_TOL => ValidationState::Verified,
        Some(_) => ValidationState::Failed,
        None => ValidationState::Unverified,
    }
}

/// Build a pass-1-only result. Used when the helper short-circuits before
/// running pass-2 (no actuator, non-sizing-mode actuator, velocity solve
/// failed, singular actuator pose, etc.).
fn pass1_only_result(
    mech: &Mechanism,
    q: &DVector<f64>,
    t: f64,
    pass1: &StaticSolveResult,
) -> ReactionSolveResult {
    let reactions = extract_reactions(mech, pass1);
    let driver_torque = get_driver_reactions(&reactions).first().map(|r| r.effort);
    let validation = compute_validation(
        mech,
        q,
        t,
        &reactions,
        pass1_eff_actuator_force(mech),
        pass1.condition_number,
        false,
        driver_torque,
    );
    ReactionSolveResult {
        reactions,
        driver_torque,
        actuator_force: None,
        used_two_pass: false,
        condition_number: pass1.condition_number,
        is_overconstrained: pass1.is_overconstrained,
        residual_norm: pass1.residual_norm,
        validation,
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

    let pass1_only = |reacts: Vec<JointReaction>| {
        let validation = compute_validation(
            mech,
            q,
            t,
            &reacts,
            pass1_eff_actuator_force(mech),
            pass1.condition_number,
            false,
            driver_torque,
        );
        ReactionSolveResult {
            reactions: reacts,
            driver_torque,
            actuator_force: None,
            used_two_pass: false,
            condition_number: pass1.condition_number,
            is_overconstrained: pass1.is_overconstrained,
            residual_norm: pass1.residual_norm,
            validation,
        }
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

    // Compute the pass-2 residual `‖Φ_qᵀ·λ_2 + q_new‖` to track whether
    // the SVD solve converged cleanly. Skipped in the previous version
    // (left as 0.0) — that meant a failed pass-2 solve could silently
    // ship wrong lambdas. With this populated, `is_valid()` and the
    // GUI badge can flag the inconsistency.
    let phi_q_t = phi_q.transpose();
    let residual = &phi_q_t * &lambdas2 + &q_new;
    let pass2_residual_norm = residual.norm();

    let pass2 = StaticSolveResult {
        lambdas: lambdas2,
        q_forces: q_new,
        residual_norm: pass2_residual_norm,
        is_overconstrained: false,
        condition_number: 0.0,
    };
    let reactions2 = extract_reactions(mech, &pass2);

    // Independent validation: per-body Cartesian equilibrium (using the
    // back-solved actuator force), condition gate, and driver-collapse.
    let validation = compute_validation(
        mech,
        q,
        t,
        &reactions2,
        Some(actuator_force),
        pass1.condition_number,
        true,
        driver_torque,
    );

    let result = ReactionSolveResult {
        reactions: reactions2,
        driver_torque,
        actuator_force: Some(actuator_force),
        used_two_pass: true,
        condition_number: pass1.condition_number,
        is_overconstrained: pass1.is_overconstrained,
        residual_norm: pass2_residual_norm,
        validation,
    };

    // Catch solver regressions in dev builds. `is_valid()` is false only on
    // a genuine `Failed` verdict (independent equilibrium violated, ill-
    // conditioned, or driver didn't collapse) — NOT on `Unverified`, so this
    // never false-fires on mechanisms outside the modeled element set.
    debug_assert!(
        result.is_valid(),
        "solve_reactions_with_actuator produced a Failed validation: \
         validation={:?}, residual_norm={}, driver_lambda={:?}",
        result.validation,
        result.residual_norm,
        result.reactions.iter().find(|r| r.n_equations == 1).map(|r| r.effort),
    );

    Ok(result)
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

    /// Seed pose for the test 4-bar at a given crank angle, with initial
    /// guesses that drive Newton-iteration toward the open-branch
    /// assembly the FBD validations assume.
    ///
    /// The rocker initial guess flips with `angle.sin()`: when the crank
    /// is in the upper half-plane (sin θ ≥ 0) the rocker seed points
    /// upward (θ_r = +π/2); when below (sin θ < 0, e.g. BDC) it points
    /// downward (θ_r = −π/2). Without this, the rocker seed at +π/2
    /// would conflict with the coupler seed at θ_c = 0 (whose C_y has
    /// the same sign as sin(crank angle)), and Newton can converge to
    /// the wrong branch or fail to converge.
    fn seed_pose_at_angle(mech: &Mechanism, angle: f64) -> DVector<f64> {
        let state = mech.state();
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("coupler", &mut q0, angle.cos(), angle.sin(), 0.0);
        let rocker_angle = if angle.sin() >= 0.0 { PI / 2.0 } else { -PI / 2.0 };
        state.set_pose("rocker", &mut q0, 4.0, 0.0, rocker_angle);
        let pos = solve_position(mech, &q0, angle, 1e-10, 50)
            .expect("position solve");
        assert!(pos.converged);
        pos.q
    }

    // ── FBD helpers (shared by the per-pose pass-2 validation tests) ───────
    //
    // The 9×9 Cartesian equilibrium system has the same structure at every
    // pose — only the joint positions, CGs, and actuator endpoint change.
    // The two helpers below isolate that machinery so each per-pose test
    // is just (a) derive the joint positions B and C from θ_2, then
    // (b) call `solve_fbd_pass2_for_pose` and `assert_pass2_matches_fbd`.
    //
    // See the full derivation comment block above
    // `fbd_validates_pass2_reactions_at_60deg` (kept in place as the
    // documented worked example).

    /// FBD constants matching `build_fourbar_with_actuator`.
    /// Changing the fixture geometry means changing these too.
    const FBD_G: f64 = 9.81;
    const FBD_M_CRANK: f64 = 2.0;
    const FBD_M_COUPLER: f64 = 3.0;
    const FBD_M_ROCKER: f64 = 2.0;
    const FBD_DX: f64 = 4.0; // rocker-ground pivot world position
    const FBD_DY: f64 = 0.0;
    const FBD_P_BX: f64 = 2.0; // actuator point_b world position (ground)
    const FBD_P_BY: f64 = 0.0;

    /// Solve the 9×9 Cartesian equilibrium system for the canonical
    /// 4-bar + LinearActuator fixture in sizing mode at the given pose.
    ///
    /// Inputs are the world-frame positions of the pin joints J2 (B) and
    /// J3 (C). A is fixed at the origin, D and the actuator's ground
    /// endpoint p_b are fixed by the fixture (see `FBD_*` constants
    /// above). p_a equals the coupler midpoint (= CG_coupler) for this
    /// fixture's `point_a = (b/2, 0)`, which zeros the actuator's moment
    /// about the coupler CG.
    ///
    /// Returns `[R_J1x, R_J1y, R_J2x, R_J2y, R_J3x, R_J3y, R_J4x, R_J4y, F_act]`.
    ///
    /// See the full derivation in the comment block above
    /// `fbd_validates_pass2_reactions_at_60deg`.
    fn solve_fbd_pass2_for_pose(bx: f64, by: f64, cx: f64, cy: f64) -> [f64; 9] {
        use nalgebra::DMatrix;

        // CGs — midpoint of each uniform bar's two attachment points.
        let cgcx = (bx + cx) * 0.5;
        let cgcy = (by + cy) * 0.5;
        let cgrx = (cx + FBD_DX) * 0.5;
        let cgry = (cy + FBD_DY) * 0.5;

        // Actuator endpoint p_a = coupler midpoint = CG_coupler for this
        // fixture. p_b is the fixed ground endpoint.
        let p_ax = cgcx;
        let p_ay = cgcy;
        let dxa = FBD_P_BX - p_ax;
        let dya = FBD_P_BY - p_ay;
        let lena = (dxa * dxa + dya * dya).sqrt();
        let ux = dxa / lena;
        let uy = dya / lena;

        let mut a = DMatrix::<f64>::zeros(9, 9);
        let mut b = DVector::<f64>::zeros(9);

        // Column legend:
        //   0: R_J1x  1: R_J1y  2: R_J2x  3: R_J2y
        //   4: R_J3x  5: R_J3y  6: R_J4x  7: R_J4y
        //   8: F_act

        // Crank ΣFx: -R_J1x + R_J2x = 0
        a[(0, 0)] = -1.0;
        a[(0, 2)] = 1.0;

        // Crank ΣFy: -R_J1y + R_J2y = m_crank·g
        a[(1, 1)] = -1.0;
        a[(1, 3)] = 1.0;
        b[1] = FBD_M_CRANK * FBD_G;

        // Crank ΣM_CG. A = (0,0), CG_crank = B/2, so r_A−CG = −B/2 and
        // r_B−CG = +B/2. Moment formula r_x·F_y − r_y·F_x then collapses
        // to: Bx·(R_J1y + R_J2y) − By·(R_J1x + R_J2x) = 0.
        a[(2, 0)] = -by;
        a[(2, 1)] = bx;
        a[(2, 2)] = -by;
        a[(2, 3)] = bx;

        // Coupler ΣFx: -R_J2x + R_J3x − u_x·F_act = 0
        // (force on body_a is −F_act·u per evaluate_linear_actuator)
        a[(3, 2)] = -1.0;
        a[(3, 4)] = 1.0;
        a[(3, 8)] = -ux;

        // Coupler ΣFy: -R_J2y + R_J3y − u_y·F_act = m_coupler·g
        a[(4, 3)] = -1.0;
        a[(4, 5)] = 1.0;
        a[(4, 8)] = -uy;
        b[4] = FBD_M_COUPLER * FBD_G;

        // Coupler ΣM_CG:
        //   −(Bx − CGcx)·R_J2y + (By − CGcy)·R_J2x
        //   + (Cx − CGcx)·R_J3y − (Cy − CGcy)·R_J3x = 0
        // Actuator term vanishes since p_a = CG_coupler.
        let b_dx = bx - cgcx;
        let b_dy = by - cgcy;
        let c_dx = cx - cgcx;
        let c_dy = cy - cgcy;
        a[(5, 2)] = b_dy;
        a[(5, 3)] = -b_dx;
        a[(5, 4)] = -c_dy;
        a[(5, 5)] = c_dx;

        // Rocker ΣFx: -R_J3x − R_J4x = 0
        a[(6, 4)] = -1.0;
        a[(6, 6)] = -1.0;

        // Rocker ΣFy: -R_J3y − R_J4y = m_rocker·g
        a[(7, 5)] = -1.0;
        a[(7, 7)] = -1.0;
        b[7] = FBD_M_ROCKER * FBD_G;

        // Rocker ΣM_CG:
        //   −(Cx − CGrx)·R_J3y + (Cy − CGry)·R_J3x
        //   − (Dx − CGrx)·R_J4y + (Dy − CGry)·R_J4x = 0
        let cr_dx = cx - cgrx;
        let cr_dy = cy - cgry;
        let dr_dx = FBD_DX - cgrx;
        let dr_dy = FBD_DY - cgry;
        a[(8, 4)] = cr_dy;
        a[(8, 5)] = -cr_dx;
        a[(8, 6)] = dr_dy;
        a[(8, 7)] = -dr_dx;

        let x = a
            .full_piv_lu()
            .solve(&b)
            .expect("FBD linear system should be non-singular");
        [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]]
    }

    /// Run `solve_reactions_with_actuator` and assert the result matches
    /// the FBD-derived expected values within `tol`. Also asserts the
    /// pass-2 driver lambda is ~0 (the whole point of pass-2: actuator
    /// carries the load, rotational driver torque collapses).
    ///
    /// `tol` is an absolute tolerance (newtons for reactions / F_act).
    /// Use `1e-4` for well-conditioned poses; loosen for near-singular
    /// poses where the SVD solve's residual scales with the magnitude
    /// of the answer (which can be in the kilonewton range when
    /// transmission angle approaches 0 or π).
    fn assert_pass2_matches_fbd(
        mech: &Mechanism,
        q: &DVector<f64>,
        crank_angle: f64,
        expected: &[f64; 9],
        pose_label: &str,
        tol: f64,
    ) {
        let result = solve_reactions_with_actuator(mech, q, crank_angle, 1.0)
            .expect("statics should converge in sizing mode");
        assert!(
            result.used_two_pass,
            "{}: sizing-mode actuator → pass-2",
            pose_label,
        );

        // Driver-torque tolerance scales with the overall force tolerance.
        // At well-conditioned poses tol=1e-4 implies driver-torque ~1e-6
        // (which is what the solver actually achieves). At near-singular
        // poses the driver-torque residual scales with the reaction
        // magnitude, so we widen this proportionally.
        let drv_lambda = result
            .reactions
            .iter()
            .find(|r| r.n_equations == 1)
            .map(|r| r.effort)
            .expect("driver lambda present");
        let drv_tol = (tol * 1e-2).max(1e-6);
        assert!(
            drv_lambda.abs() < drv_tol,
            "{}: pass-2 driver torque should be ~0 (tol {:.0e}); got {}",
            pose_label, drv_tol, drv_lambda,
        );

        let sim_f_act = result.actuator_force.expect("pass-2 produces F_act");
        let get = |id: &str| -> (f64, f64) {
            let r = result
                .reactions
                .iter()
                .find(|r| r.joint_id == id)
                .unwrap_or_else(|| panic!("{}: joint {} missing", pose_label, id));
            (r.force_global[0], r.force_global[1])
        };

        let cmp = |label: &str, sim: (f64, f64), exp: (f64, f64)| {
            assert!(
                (sim.0 - exp.0).abs() < tol && (sim.1 - exp.1).abs() < tol,
                "{}: {} mismatch (tol {:.0e}): sim = ({}, {}), FBD = ({}, {})",
                pose_label, label, tol, sim.0, sim.1, exp.0, exp.1,
            );
        };
        cmp("J1", get("J1"), (expected[0], expected[1]));
        cmp("J2", get("J2"), (expected[2], expected[3]));
        cmp("J3", get("J3"), (expected[4], expected[5]));
        cmp("J4", get("J4"), (expected[6], expected[7]));
        assert!(
            (sim_f_act - expected[8]).abs() < tol,
            "{}: F_act mismatch (tol {:.0e}): sim = {}, FBD = {}",
            pose_label, tol, sim_f_act, expected[8],
        );
    }

    #[test]
    fn no_actuator_returns_pass_one_reactions() {
        // A 4-bar without a LinearActuator: helper returns pass-1 reactions
        // and used_two_pass=false. Reactions match a direct extract_reactions.
        let mech = build_fourbar();
        let q = seed_pose_at_angle(&mech, PI / 3.0);

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
        let q = seed_pose_at_angle(&mech, PI / 3.0);

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
        // Pose θ_2 = π/3. Joint-position derivation lives in the long
        // comment block above this function (the canonical worked
        // example). All other FBD math goes through
        // `solve_fbd_pass2_for_pose` and `assert_pass2_matches_fbd`.
        let mech = build_fourbar_with_actuator(0.0);
        let q = seed_pose_at_angle(&mech, PI / 3.0);

        let sqrt3 = 3f64.sqrt();
        let bx = 0.5_f64;
        let by = sqrt3 / 2.0;
        let cx = (38.0 + 3.0 * sqrt3) / 13.0;
        let cy = (21.0 + 2.0 * sqrt3) / 13.0;

        let expected = solve_fbd_pass2_for_pose(bx, by, cx, cy);
        assert_pass2_matches_fbd(&mech, &q, PI / 3.0, &expected, "θ_2=π/3", 1e-4);
    }

    /// FBD-validated reactions test for 4-bar + LinearActuator in sizing
    /// mode at pose θ_2 = π/2 (top dead center).
    ///
    /// At θ_2 = π/2 the crank is vertical:
    ///   A = (0, 0),  B = (0, 1),  D = (4, 0)
    ///
    /// Loop closure (Cx − 0)² + (Cy − 1)² = 9 and (Cx − 4)² + Cy² = 4.
    /// Subtracting gives `4Cx − Cy = 10`, so `Cy = 4Cx − 10`.
    /// Substituting back yields `17·Cx² − 88·Cx + 112 = 0`.
    /// Discriminant: 88² − 4·17·112 = 7744 − 7616 = 128 = (8√2)².
    /// Two roots:
    ///   Cx = (44 + 4√2)/17 ≈ 2.9210,  Cy = (6 + 16√2)/17 ≈ 1.6840  (open branch — matches seed)
    ///   Cx = (44 − 4√2)/17 ≈ 2.2554,  Cy ≈ −0.9784                  (crossed branch — skipped)
    ///
    /// Open branch chosen because `seed_pose_at_angle`'s initial guesses
    /// place the coupler and rocker above the x-axis; `solve_position`
    /// Newton-iterates to the nearest consistent C.
    ///
    /// Crank ΣM_CG simplification at this pose: with B = (0, 1),
    /// `−By·(R_J1x + R_J2x) + Bx·(R_J1y + R_J2y) = 0` collapses to
    /// `R_J1x + R_J2x = 0`. The generic `solve_fbd_pass2_for_pose`
    /// handles it without special-casing.
    #[test]
    fn fbd_validates_pass2_reactions_at_90deg() {
        let mech = build_fourbar_with_actuator(0.0);
        let q = seed_pose_at_angle(&mech, PI / 2.0);

        let sqrt2 = 2f64.sqrt();
        let bx = 0.0_f64;
        let by = 1.0_f64;
        let cx = (44.0 + 4.0 * sqrt2) / 17.0;
        let cy = 4.0 * cx - 10.0;

        let expected = solve_fbd_pass2_for_pose(bx, by, cx, cy);
        assert_pass2_matches_fbd(&mech, &q, PI / 2.0, &expected, "θ_2=π/2", 1e-4);
    }

    /// FBD-validated reactions test for 4-bar + LinearActuator in sizing
    /// mode at pose θ_2 = π/4 (45°, mid-stroke). Generic case: Bx ≠ 0
    /// and By ≠ 0, so the Crank-M equation does NOT collapse to
    /// `R_J1x + R_J2x = 0` like it does at TDC/BDC. Exercises every
    /// coefficient slot in `solve_fbd_pass2_for_pose`.
    ///
    /// At θ_2 = π/4:
    ///   A = (0, 0),  B = (√2/2, √2/2),  D = (4, 0)
    ///
    /// Loop closure (Cx − Bx)² + (Cy − By)² = 9 and (Cx − 4)² + Cy² = 4.
    /// Using |B|² = 1, subtracting and rearranging gives the linear
    /// combination `(8 − √2)·Cx − √2·Cy = 20`, so
    /// `Cy = ((8 − √2)·Cx − 20) / √2`. Substituting back into
    /// `Cx² + Cy² = 8·Cx − 12` (from the rocker constraint) yields the
    /// quadratic
    ///   `(68 − 16√2)·Cx² + (−336 + 40√2)·Cx + 424 = 0`
    /// with discriminant `768 + 256·√2 = 256·(3 + √2)`, so
    /// `√Δ = 16·√(3 + √2)`. Two roots:
    ///   Cx ≈ 3.4498, Cy ≈ 1.9230  (open branch — matches seed)
    ///   Cx ≈ 2.7091, Cy ≈ −1.5260 (crossed branch — skipped)
    ///
    /// Open branch chosen because `seed_pose_at_angle` at π/4 (sin > 0)
    /// uses rocker θ_r = +π/2, placing rocker.C above the x-axis at
    /// (4, 2), close to the upper root.
    ///
    /// Code computes the quadratic coefficients from `sqrt2` directly
    /// rather than baking high-precision literals, both for derivability
    /// and so future readers can re-verify the algebra by inspection.
    #[test]
    fn fbd_validates_pass2_reactions_at_45deg() {
        let mech = build_fourbar_with_actuator(0.0);
        let angle = PI / 4.0;
        let q = seed_pose_at_angle(&mech, angle);

        let sqrt2 = 2f64.sqrt();
        let bx = sqrt2 / 2.0;
        let by = sqrt2 / 2.0;

        // Solve the quadratic for the open-branch Cx, then back out Cy
        // from the linear loop-closure combination.
        let aq = 68.0 - 16.0 * sqrt2;
        let bq = -336.0 + 40.0 * sqrt2;
        let cq = 424.0;
        let disc = bq * bq - 4.0 * aq * cq;
        let cx = (-bq + disc.sqrt()) / (2.0 * aq);
        let cy = ((8.0 - sqrt2) * cx - 20.0) / sqrt2;

        let expected = solve_fbd_pass2_for_pose(bx, by, cx, cy);
        assert_pass2_matches_fbd(&mech, &q, angle, &expected, "θ_2=π/4", 1e-4);
    }

    /// FBD-validated reactions test for 4-bar + LinearActuator in sizing
    /// mode at pose θ_2 = 3π/2 (bottom dead center — crank vertical
    /// pointing down). Mirror of the TDC test above.
    ///
    /// At θ_2 = 3π/2:
    ///   A = (0, 0),  B = (0, −1),  D = (4, 0)
    ///
    /// Loop closure (Cx − 0)² + (Cy + 1)² = 9 and (Cx − 4)² + Cy² = 4.
    /// Subtracting gives `8·Cx + 2·Cy = 20`, i.e. `Cy = 10 − 4·Cx`.
    /// Substituting yields the same quadratic as TDC: `17·Cx² − 88·Cx + 112 = 0`.
    /// Discriminant 128 = (8√2)². Two roots, with Cy = 10 − 4·Cx:
    ///   Cx = (44 + 4√2)/17 ≈ 2.9210,  Cy = (−6 − 16√2)/17 ≈ −1.6840  (open branch — matches seed)
    ///   Cx = (44 − 4√2)/17 ≈ 2.2554,  Cy ≈ 0.9784                    (other branch — skipped)
    ///
    /// Open branch chosen because at BDC `seed_pose_at_angle` flips its
    /// rocker initial guess to θ_r = −π/2 (matching the sign of
    /// `angle.sin()`), pulling Newton toward the C below the x-axis.
    /// Without that flip the seed would conflict (coupler-C below,
    /// rocker-C above) and the solve could pick either branch.
    ///
    /// Crank ΣM_CG at this pose: with B = (0, −1) and CG_crank = B/2,
    /// `−By·(R_J1x + R_J2x) + Bx·(R_J1y + R_J2y) = 0` collapses to
    /// `R_J1x + R_J2x = 0` (same simplification as TDC, since Bx = 0
    /// in both — only the sign of By differs).
    #[test]
    fn fbd_validates_pass2_reactions_at_270deg() {
        let mech = build_fourbar_with_actuator(0.0);
        let angle = 3.0 * PI / 2.0;
        let q = seed_pose_at_angle(&mech, angle);

        let sqrt2 = 2f64.sqrt();
        let bx = 0.0_f64;
        let by = -1.0_f64;
        let cx = (44.0 + 4.0 * sqrt2) / 17.0;
        let cy = 10.0 - 4.0 * cx;

        let expected = solve_fbd_pass2_for_pose(bx, by, cx, cy);
        assert_pass2_matches_fbd(&mech, &q, angle, &expected, "θ_2=3π/2", 1e-4);
    }

    /// FBD-validated reactions test for 4-bar + LinearActuator in sizing
    /// mode at pose θ_2 = 0.9π (near-singular). For this Grashof
    /// crank-rocker (a=1, b=3, c=2, d=4) the only singular configuration
    /// occurs at θ_2 = π where BD = √(17 − 8·cos π) = 5 = b + c
    /// (coupler and rocker collinear, transmission angle = 180°,
    /// Jacobian rank-deficient). At θ_2 = 0.9π we approach this within
    /// ~18°: BD = √(17 + 7.608) ≈ 4.961, transmission-angle-at-C ≈ 165°
    /// (cos via law of cosines: (3² + 2² − 4.961²) / (2·3·2) ≈ −0.967).
    ///
    /// Why include this pose: it stresses the simulator's numerical
    /// conditioning. The SVD-based static solve produces residuals that
    /// scale with the magnitude of the answer, and at near-singular
    /// poses the actuator-force magnitude grows large because the
    /// actuator has very poor mechanical advantage. The 1e-4 N
    /// tolerance used for nominal poses isn't tight enough here; we
    /// use 1e-2 N which is still 4–6 orders of magnitude below the
    /// expected force magnitudes (~kN scale at this pose).
    ///
    /// Loop closure (Cx − Bx)² + (Cy − By)² = 9 with
    /// Bx = cos(0.9π), By = sin(0.9π) and (Cx − 4)² + Cy² = 4 reduces
    /// to a linear combination
    ///   (8 − 2·Bx)·Cx − 2·By·Cy = 20
    /// (numerically `9.9021·Cx − 0.6180·Cy = 20`). Substituting back
    /// into the rocker constraint `Cx² + Cy² = 8·Cx − 12` yields a
    /// quadratic whose two roots are very close together (discriminant
    /// ≈ 0.01 vs ~16 for the linear-term squared — the near-singular
    /// signature). Open branch C ≈ (2.078, 0.930).
    ///
    /// Code computes Cx and Cy from `cos(0.9π)` / `sin(0.9π)` directly.
    /// If the test ever fails, that's likely because the simulator's
    /// conditioning has degraded — investigate before relaxing
    /// tolerance further.
    #[test]
    fn fbd_validates_pass2_reactions_near_singular() {
        let mech = build_fourbar_with_actuator(0.0);
        let angle = 0.9 * PI;
        let q = seed_pose_at_angle(&mech, angle);

        let bx = angle.cos();
        let by = angle.sin();

        // Loop-closure linear combination: (8 − 2·Bx)·Cx − 2·By·Cy = 20.
        // Solving for Cy: Cy = (p·Cx − 20)/r where p = 8 − 2·Bx, r = 2·By.
        // Substituting into the rocker constraint Cx² + Cy² = 8·Cx − 12
        // and multiplying through by r² gives the quadratic
        //   (r² + p²)·Cx² + (−40·p − 8·r²)·Cx + (400 + 12·r²) = 0.
        let p = 8.0 - 2.0 * bx;
        let r = 2.0 * by;
        let aq = r * r + p * p;
        let bq = -40.0 * p - 8.0 * r * r;
        let cq = 400.0 + 12.0 * r * r;
        let disc = bq * bq - 4.0 * aq * cq;
        assert!(
            disc > 0.0,
            "near-singular pose still has a real assembly; \
             disc = {} should be > 0",
            disc,
        );
        // Open branch (Cy > 0): take the + root.
        let cx = (-bq + disc.sqrt()) / (2.0 * aq);
        let cy = (p * cx - 20.0) / r;

        let expected = solve_fbd_pass2_for_pose(bx, by, cx, cy);
        // Tolerance widened to 1e-2 N — see docstring for rationale.
        // F_act at this pose is order kN, so 1e-2 N is still ~5 orders
        // of magnitude below the answer.
        assert_pass2_matches_fbd(&mech, &q, angle, &expected, "θ_2=0.9π (near-singular)", 1e-2);
    }

    /// Cover the `ReactionSolveResult::is_valid()` contract directly.
    /// On well-conditioned mechanisms the helper should always produce
    /// `is_valid() == true`; this test pins that across the no-actuator,
    /// sizing-mode actuator, and known-force actuator branches.
    #[test]
    fn is_valid_reports_true_for_well_conditioned_solves() {
        // No actuator → pass-1 only.
        let mech_a = build_fourbar();
        let q_a = seed_pose_at_angle(&mech_a, PI / 3.0);
        let r_a = solve_reactions_with_actuator(&mech_a, &q_a, PI / 3.0, 1.0).unwrap();
        assert!(!r_a.used_two_pass);
        assert!(
            r_a.is_valid(),
            "no-actuator pass-1 should validate; residual={}, drv_lambda={:?}",
            r_a.residual_norm,
            r_a.reactions.iter().find(|r| r.n_equations == 1).map(|r| r.effort),
        );
        assert!(
            r_a.residual_norm < 1e-10,
            "pass-1 residual should be ~1e-12; got {}", r_a.residual_norm,
        );

        // Sizing-mode actuator → pass-2.
        let mech_b = build_fourbar_with_actuator(0.0);
        let q_b = seed_pose_at_angle(&mech_b, PI / 3.0);
        let r_b = solve_reactions_with_actuator(&mech_b, &q_b, PI / 3.0, 1.0).unwrap();
        assert!(r_b.used_two_pass);
        assert!(
            r_b.is_valid(),
            "sizing-mode pass-2 should validate; residual={}, drv_lambda={:?}",
            r_b.residual_norm,
            r_b.reactions.iter().find(|r| r.n_equations == 1).map(|r| r.effort),
        );

        // Known-force actuator → pass-1 only (double-count guard fires).
        let mech_c = build_fourbar_with_actuator(50.0);
        let q_c = seed_pose_at_angle(&mech_c, PI / 3.0);
        let r_c = solve_reactions_with_actuator(&mech_c, &q_c, PI / 3.0, 1.0).unwrap();
        assert!(!r_c.used_two_pass);
        assert!(
            r_c.is_valid(),
            "known-force pass-1 should validate; residual={}",
            r_c.residual_norm,
        );
    }

    #[test]
    fn actuator_with_known_force_skips_pass_two() {
        // LinearActuator with stored force ≠ 0: helper must NOT run pass-2
        // (would double-count the actuator's q_forces contribution).
        // Reactions match a direct pass-1 solve.
        let mech = build_fourbar_with_actuator(50.0);
        let q = seed_pose_at_angle(&mech, PI / 3.0);

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

    /// Validate the trajectory `_using_q_dot` path (the one with no FBD
    /// pose test): for the same pose and load, the back-solved actuator
    /// force and all reactions must be IDENTICAL to the constant-speed
    /// path regardless of the input rate (omega cancels in the power
    /// balance F = τ·ω/(dL/dt)). Covers positive, larger, and negative
    /// rates. Originated as a workflow review probe; kept as a regression
    /// test for the otherwise-unvalidated trajectory path.
    #[test]
    fn qdot_variant_is_rate_invariant_and_matches_constant_speed() {
        use crate::solver::kinematics::solve_velocity;
        let mech = build_fourbar_with_actuator(0.0);
        let angle = PI / 3.0;
        let q = seed_pose_at_angle(&mech, angle);

        let cs = solve_reactions_with_actuator(&mech, &q, angle, 1.0).unwrap();
        assert!(cs.used_two_pass);
        let f_cs = cs.actuator_force.unwrap();

        // Same pose, q_dot scaled to several different rates.
        let qd1 = solve_velocity(&mech, &q, angle).unwrap();
        for rate in [3.7_f64, -2.1, 0.25] {
            let qd = &qd1 * rate;
            let tr = solve_reactions_with_actuator_using_q_dot(&mech, &q, &qd, angle, rate)
                .unwrap();
            assert!(tr.used_two_pass, "rate {rate}: expected pass-2");
            assert!(
                (f_cs - tr.actuator_force.unwrap()).abs() < 1e-9,
                "rate {rate}: F_act differs: cs={} tr={}",
                f_cs, tr.actuator_force.unwrap(),
            );
            for (a, b) in cs.reactions.iter().zip(tr.reactions.iter()) {
                assert_eq!(a.joint_id, b.joint_id);
                assert!(
                    (a.force_global[0] - b.force_global[0]).abs() < 1e-9
                        && (a.force_global[1] - b.force_global[1]).abs() < 1e-9,
                    "rate {rate}: {} reaction differs", a.joint_id,
                );
            }
        }
    }

    /// The canonical sizing-mode actuator pose must reach the strongest
    /// verdict: independently `Verified` (not merely `Unverified`).
    #[test]
    fn validation_verified_for_modeled_actuator_mechanism() {
        let mech = build_fourbar_with_actuator(0.0);
        let q = seed_pose_at_angle(&mech, PI / 3.0);
        let r = solve_reactions_with_actuator(&mech, &q, PI / 3.0, 1.0).unwrap();
        assert_eq!(
            r.validation(),
            ValidationState::Verified,
            "gravity + single actuator + revolute 4-bar should be independently verified",
        );
        assert!(r.is_valid());
    }

    /// THE headline regression test: the independent per-body equilibrium
    /// check must have teeth. A correct reaction set scores ~machine-eps;
    /// perturbing a single joint reaction by a physically-significant
    /// amount must blow the residual far past the tolerance. This is the
    /// class of error the near-tautological `residual_norm` could NOT
    /// catch (a mutation-test swap of the moment arm left residual_norm at
    /// ~1e-14 while reactions were wrong).
    #[test]
    fn body_equilibrium_residual_catches_perturbed_reaction() {
        let mech = build_fourbar_with_actuator(0.0);
        let angle = PI / 3.0;
        let q = seed_pose_at_angle(&mech, angle);
        let r = solve_reactions_with_actuator(&mech, &q, angle, 1.0).unwrap();
        let f_act = r.actuator_force;

        // Correct reactions → tiny residual.
        let good = body_equilibrium_residual(&mech, &q, angle, &r.reactions, f_act)
            .expect("modeled mechanism yields Some");
        assert!(good < 1e-9, "correct reactions should balance; got {good}");

        // Perturb one joint's reaction by 50 N → must be caught.
        let mut bad_reactions = r.reactions.clone();
        let j = bad_reactions
            .iter_mut()
            .find(|jr| jr.n_equations > 1)
            .expect("a joint reaction exists");
        j.force_global[0] += 50.0;
        let bad = body_equilibrium_residual(&mech, &q, angle, &bad_reactions, f_act)
            .expect("modeled mechanism yields Some");
        assert!(
            bad > EQUILIBRIUM_REL_TOL,
            "a 50 N perturbation must break equilibrium; residual stayed {bad}",
        );
    }

    /// A 4-bar with gravity + sizing actuator + a ForceZone load must now
    /// reach `Verified` (force zones are independently modeled). The zone
    /// applies a real 1000 N load at an OFF-CG application point, so the
    /// per-body moment term is exercised — a wrong zone force or moment
    /// would push the residual past tolerance and report `Failed` instead.
    #[test]
    fn validation_verified_with_force_zone() {
        use crate::core::body::BodyGeometry;
        use crate::forces::elements::ForceZoneElement;

        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let mut coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
        // Geometry spanning the coupler bar so the zone can overlap it.
        coupler.geometry = Some(BodyGeometry {
            width: 3.0,
            height: 0.4,
            offset: Vector2::new(1.5, 0.0),
        });
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
        // Sizing-mode actuator (force = 0 → pass-2 sizes it).
        mech.add_force(ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "coupler".to_string(),
            point_a: [1.5, 0.0],
            point_a_name: None,
            body_b: "ground".to_string(),
            point_b: [2.0, 0.0],
            point_b_name: None,
            force: 0.0,
            speed_limit: 0.0,
            stroke_min: 0.0,
            stroke_max: 0.0,
            end_stop_stiffness: 0.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.0,
        }));
        // Large zone guaranteed to overlap the coupler at θ_2 = π/3, with a
        // pinned OFF-CG application point so the zone moment term is real.
        mech.add_force(ForceElement::ForceZone(ForceZoneElement {
            body_id: "coupler".to_string(),
            zone_min: [0.0, 0.0],
            zone_max: [5.0, 3.0],
            force: [0.0, -1000.0],
            label: None,
            body_local_app_point: Some([2.5, 0.0]),
        }));
        mech.build().unwrap();

        let q = seed_pose_at_angle(&mech, PI / 3.0);
        let r = solve_reactions_with_actuator(&mech, &q, PI / 3.0, 1.0).unwrap();
        assert_eq!(
            r.validation(),
            ValidationState::Verified,
            "gravity + actuator + force zone should independently verify",
        );
        // The zone load really applies (1000 N) — sanity-check the
        // per-body residual is genuinely tiny, not vacuously so.
        let resid = body_equilibrium_residual(&mech, &q, PI / 3.0, &r.reactions, r.actuator_force)
            .expect("modeled");
        assert!(resid < 1e-9, "force-zone equilibrium residual {resid}");
    }

    /// A mechanism containing an element type the independent check does
    /// not model must report `Unverified` — never a false `Verified`/
    /// `Failed`. Two actuators trips the ">1 actuator" guard.
    #[test]
    fn validation_unverified_for_unmodeled_mechanism() {
        let mut mech = {
            // Rebuild the canonical fixture but add a SECOND actuator.
            let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
            let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
            let coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
            let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);
            let mut m = Mechanism::new();
            m.add_body(ground).unwrap();
            m.add_body(crank).unwrap();
            m.add_body(coupler).unwrap();
            m.add_body(rocker).unwrap();
            m.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
            m.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
            m.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
            m.add_revolute_joint("J4", "ground", "O4", "rocker", "D").unwrap();
            m.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
                .unwrap();
            m.add_force(ForceElement::Gravity(GravityElement::default()));
            m
        };
        let act = LinearActuatorElement {
            body_a: "coupler".to_string(),
            point_a: [1.5, 0.0],
            point_a_name: None,
            body_b: "ground".to_string(),
            point_b: [2.0, 0.0],
            point_b_name: None,
            force: 100.0,
            speed_limit: 0.0,
            stroke_min: 0.0,
            stroke_max: 0.0,
            end_stop_stiffness: 0.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.0,
        };
        let mut act2 = act.clone();
        act2.point_b = [2.5, 0.0];
        mech.add_force(ForceElement::LinearActuator(act));
        mech.add_force(ForceElement::LinearActuator(act2));
        mech.build().unwrap();

        let q = seed_pose_at_angle(&mech, PI / 3.0);
        let r = solve_reactions_with_actuator(&mech, &q, PI / 3.0, 1.0).unwrap();
        assert_eq!(
            r.validation(),
            ValidationState::Unverified,
            "two actuators are outside the modeled set → Unverified, not a guess",
        );
        // Unverified is NOT a failure — is_valid() stays true.
        assert!(r.is_valid());
    }

    /// Pin the pass-2 driver-collapse gate. A WRONG back-solved actuator
    /// force produces reactions that still satisfy per-body equilibrium
    /// (they self-consistently balance whatever force was injected), so the
    /// per-body check alone is fooled — ONLY the driver-collapse gate
    /// catches it. This test proves both halves, so deleting the gate makes
    /// it fail. (Found via mutation testing: disabling the gate was
    /// otherwise invisible to the whole suite.)
    #[test]
    fn driver_collapse_gate_catches_wrong_actuator_force() {
        use crate::solver::assembly::assemble_jacobian;
        use crate::solver::kinematics::solve_velocity;
        let mech = build_fourbar_with_actuator(0.0);
        let angle = PI / 3.0;
        let q = seed_pose_at_angle(&mech, angle);
        let pass1 = solve_statics(&mech, &q, angle).unwrap();
        let driver_torque = get_driver_reactions(&extract_reactions(&mech, &pass1))
            .first()
            .map(|r| r.effort);
        let act = mech
            .forces()
            .iter()
            .find_map(|f| match f {
                ForceElement::LinearActuator(a) => Some(a.clone()),
                _ => None,
            })
            .unwrap();
        let st = mech.state();
        let qz = DVector::zeros(st.n_coords());
        let phi_q = assemble_jacobian(&mech, &q, angle);

        let correct_f = compute_actuator_force_from_power_balance(
            &mech,
            &q,
            &solve_velocity(&mech, &q, angle).unwrap(),
            driver_torque.unwrap(),
            1.0,
            &act,
        )
        .unwrap();
        let wrong_f = correct_f * 2.0; // deliberately wrong

        let mut am = act.clone();
        am.force = wrong_f;
        let q_act = evaluate_linear_actuator(&am, st, &q, &qz);
        let q_new = &pass1.q_forces + &q_act;
        let lam = phi_q
            .transpose()
            .svd(true, true)
            .solve(&(-&q_new), 1e-14)
            .unwrap();
        let pass2 = StaticSolveResult {
            lambdas: lam,
            q_forces: q_new,
            residual_norm: 0.0,
            is_overconstrained: false,
            condition_number: 0.0,
        };
        let reactions = extract_reactions(&mech, &pass2);

        // Per-body equilibrium is FOOLED — the reactions self-consistently
        // balance the wrong force, so this residual is tiny. That is exactly
        // why the driver-collapse gate must exist.
        let resid = body_equilibrium_residual(&mech, &q, angle, &reactions, Some(wrong_f))
            .expect("modeled");
        assert!(
            resid < 1e-9,
            "wrong-F_act reactions are self-consistent (per-body can't catch): {resid}",
        );

        // The driver-collapse gate must catch it.
        let v = compute_validation(
            &mech,
            &q,
            angle,
            &reactions,
            Some(wrong_f),
            pass1.condition_number,
            true,
            driver_torque,
        );
        assert_eq!(
            v,
            ValidationState::Failed,
            "wrong F_act must fail validation via the driver-collapse gate",
        );
    }

    /// Pin the MOMENT half of the residual. The driver supplies a real
    /// couple (no-actuator pass-1), so bumping the driver reaction's torque
    /// breaks moment balance while touching NO force — the force term can't
    /// see it; only `mr = net_m/char_len` can. Every other residual-teeth
    /// test perturbs a force (caught by the force term regardless), so the
    /// moment machinery was otherwise untested — zeroing it passed the whole
    /// suite. The baseline `good < 1e-9` assertion also pins the cross-product
    /// SIGN. Found by the validate-the-validator review.
    #[test]
    fn body_equilibrium_residual_catches_moment_only_imbalance() {
        let mech = build_fourbar();
        let angle = PI / 3.0;
        let q = seed_pose_at_angle(&mech, angle);
        let r = solve_reactions_with_actuator(&mech, &q, angle, 1.0).unwrap();
        let good = body_equilibrium_residual(&mech, &q, angle, &r.reactions, None).unwrap();
        assert!(good < 1e-9, "baseline residual {good}");

        let mut bad = r.reactions.clone();
        bad.iter_mut().find(|x| x.n_equations == 1).unwrap().effort += 50.0;
        let resid = body_equilibrium_residual(&mech, &q, angle, &bad, None).unwrap();
        assert!(
            resid > EQUILIBRIUM_REL_TOL,
            "a pure-moment (driver-couple) imbalance must be caught; got {resid}",
        );
    }

    /// External force (off-CG) and external torque are in the modeled set; a
    /// mechanism carrying both must reach Verified with a tiny residual. Pins
    /// the ExternalForce/ExternalTorque handler signs — a flip in either
    /// breaks the residual → not Verified. (Both had zero validator coverage
    /// before; found by the review.)
    #[test]
    fn validation_verified_with_external_force_and_torque() {
        use crate::forces::elements::{
            ExternalForceElement, ExternalTorqueElement, TimeModulation,
        };
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
        mech.add_force(ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "coupler".to_string(),
            point_a: [1.5, 0.0],
            point_a_name: None,
            body_b: "ground".to_string(),
            point_b: [2.0, 0.0],
            point_b_name: None,
            force: 0.0,
            speed_limit: 0.0,
            stroke_min: 0.0,
            stroke_max: 0.0,
            end_stop_stiffness: 0.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.0,
        }));
        mech.add_force(ForceElement::ExternalForce(ExternalForceElement {
            body_id: "coupler".to_string(),
            local_point: [2.5, 0.0],
            local_point_name: None,
            force: [100.0, -50.0],
            modulation: TimeModulation::default(),
        }));
        mech.add_force(ForceElement::ExternalTorque(ExternalTorqueElement {
            body_id: "rocker".to_string(),
            torque: 30.0,
            modulation: TimeModulation::default(),
        }));
        mech.build().unwrap();

        let angle = PI / 3.0;
        let q = seed_pose_at_angle(&mech, angle);
        let r = solve_reactions_with_actuator(&mech, &q, angle, 1.0).unwrap();
        assert_eq!(
            r.validation(),
            ValidationState::Verified,
            "gravity + actuator + external force + external torque should verify",
        );
        let resid = body_equilibrium_residual(&mech, &q, angle, &r.reactions, r.actuator_force)
            .expect("modeled");
        assert!(resid < 1e-9, "external-load residual {resid}");
    }

    /// Pin the condition-number gate's FIRING side and the `Failed → !is_valid`
    /// mapping — the reject path of the trust anchor, otherwise never asserted.
    #[test]
    fn validation_failed_when_ill_conditioned() {
        let mech = build_fourbar_with_actuator(0.0);
        let angle = PI / 3.0;
        let q = seed_pose_at_angle(&mech, angle);
        let r = solve_reactions_with_actuator(&mech, &q, angle, 1.0).unwrap();
        let v = compute_validation(
            &mech, &q, angle, &r.reactions, r.actuator_force,
            CONDITION_CEILING * 10.0, true, r.driver_torque,
        );
        assert_eq!(v, ValidationState::Failed, "cond > ceiling must Fail");
        let vn = compute_validation(
            &mech, &q, angle, &r.reactions, r.actuator_force,
            f64::NAN, true, r.driver_torque,
        );
        assert_eq!(vn, ValidationState::Failed, "non-finite cond must Fail");
        let failed = ReactionSolveResult { validation: ValidationState::Failed, ..r.clone() };
        assert!(!failed.is_valid());
        let unver = ReactionSolveResult { validation: ValidationState::Unverified, ..r };
        assert!(unver.is_valid());
    }

    /// A NaN in a reaction must yield `Failed`, not a spurious `Verified`.
    #[test]
    fn validation_failed_on_nan_reaction() {
        let mech = build_fourbar_with_actuator(0.0);
        let angle = PI / 3.0;
        let q = seed_pose_at_angle(&mech, angle);
        let r = solve_reactions_with_actuator(&mech, &q, angle, 1.0).unwrap();
        let mut bad = r.reactions.clone();
        bad.iter_mut().find(|x| x.n_equations > 1).unwrap().force_global[0] = f64::NAN;
        let resid = body_equilibrium_residual(&mech, &q, angle, &bad, r.actuator_force)
            .expect("modeled");
        assert!(!(resid < EQUILIBRIUM_REL_TOL), "NaN reaction must not pass; got {resid}");
        let v = compute_validation(
            &mech, &q, angle, &bad, r.actuator_force,
            r.condition_number, true, r.driver_torque,
        );
        assert_eq!(v, ValidationState::Failed);
    }

    /// Pin the force-zone CENTROID branch (`body_local_app_point: None`), which
    /// no other test reaches — they all pin the app point. Verifies the
    /// check's centroid agrees with production's at a real overlap.
    #[test]
    fn validation_verified_force_zone_centroid_branch() {
        use crate::core::body::BodyGeometry;
        use crate::forces::elements::ForceZoneElement;
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let mut coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
        coupler.geometry = Some(BodyGeometry {
            width: 3.0,
            height: 0.4,
            offset: Vector2::new(1.5, 0.0),
        });
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
        mech.add_force(ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "coupler".to_string(),
            point_a: [1.5, 0.0],
            point_a_name: None,
            body_b: "ground".to_string(),
            point_b: [2.0, 0.0],
            point_b_name: None,
            force: 0.0,
            speed_limit: 0.0,
            stroke_min: 0.0,
            stroke_max: 0.0,
            end_stop_stiffness: 0.0,
            end_stop_damping: 0.0,
            end_stop_restitution: 0.0,
        }));
        // No body_local_app_point → both check and solver use the overlap
        // centroid. Horizontal+vertical force so the centroid moment is real.
        mech.add_force(ForceElement::ForceZone(ForceZoneElement {
            body_id: "coupler".to_string(),
            zone_min: [0.0, 0.0],
            zone_max: [5.0, 3.0],
            force: [400.0, -1000.0],
            label: None,
            body_local_app_point: None,
        }));
        mech.build().unwrap();

        let angle = PI / 3.0;
        let q = seed_pose_at_angle(&mech, angle);
        let r = solve_reactions_with_actuator(&mech, &q, angle, 1.0).unwrap();
        assert_eq!(r.validation(), ValidationState::Verified);
        let resid = body_equilibrium_residual(&mech, &q, angle, &r.reactions, r.actuator_force)
            .expect("modeled");
        assert!(resid < 1e-9, "centroid-branch residual {resid}");
    }
}
