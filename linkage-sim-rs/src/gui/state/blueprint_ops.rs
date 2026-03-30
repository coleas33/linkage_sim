//! Blueprint helper functions and AppState methods for blueprint manipulation.

use std::collections::HashMap;

use crate::analysis::grashof::check_grashof;
use crate::analysis::transmission::{mechanical_advantage, VelocityCoord};
use crate::analysis::force_breakdown::evaluate_contributions;
use crate::analysis::virtual_work::virtual_work_check;
use crate::core::constraint::Constraint;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::forces::elements::{ForceElement, GravityElement};
use crate::io::{
    load_mechanism_unbuilt_from_json,
    DriverJson, JointJson, MechanismJson,
};
use crate::solver::kinematics::solve_velocity;
use crate::solver::statics::{
    extract_reactions, get_driver_reactions, get_joint_reactions, solve_statics,
};

use nalgebra::DVector;

use super::{AppState, ForceResults, SolverStatus};
use crate::gui::sweep::{compute_sweep_data, detect_fourbar_links};

// ── Blueprint helper functions ────────────────────────────────────────────────

/// Extract the body_i and body_j IDs from a JointJson.
pub(crate) fn joint_body_ids(joint: &JointJson) -> (&str, &str) {
    match joint {
        JointJson::Revolute { body_i, body_j, .. }
        | JointJson::Fixed { body_i, body_j, .. }
        | JointJson::Prismatic { body_i, body_j, .. }
        | JointJson::CamFollower { body_i, body_j, .. }
        | JointJson::RevoluteDriver { body_i, body_j, .. } => (body_i.as_str(), body_j.as_str()),
    }
}

/// Returns true if the joint references the given (body_id, point_name) pair.
///
/// Used by `remove_attachment_point` to cascade-delete joints that depend on
/// the removed pivot. RevoluteDriver joints reference bodies but not specific
/// attachment points, so they are never matched.
pub(crate) fn joint_references_point(joint: &JointJson, body_id: &str, point_name: &str) -> bool {
    match joint {
        JointJson::Revolute { body_i, point_i, body_j, point_j, .. }
        | JointJson::Prismatic { body_i, point_i, body_j, point_j, .. }
        | JointJson::Fixed { body_i, point_i, body_j, point_j, .. }
        | JointJson::CamFollower { body_i, point_i, body_j, point_j, .. } => {
            (body_i == body_id && point_i == point_name)
                || (body_j == body_id && point_j == point_name)
        }
        // RevoluteDriver has body_i/body_j but no point_i/point_j fields.
        // It references bodies, not specific attachment points.
        JointJson::RevoluteDriver { .. } => false,
    }
}

/// Extract the body_i and body_j IDs from a DriverJson.
pub(crate) fn driver_body_ids(driver: &DriverJson) -> (&str, &str) {
    match driver {
        DriverJson::ConstantSpeed { body_i, body_j, .. }
        | DriverJson::Expression { body_i, body_j, .. } => (body_i.as_str(), body_j.as_str()),
    }
}

/// Generate a unique ID with the given prefix in a HashMap.
///
/// Tries prefix + "1", prefix + "2", ... until an unused key is found.
pub(crate) fn generate_unique_id<V>(prefix: &str, map: &HashMap<String, V>) -> String {
    let mut i = 1;
    loop {
        let id = format!("{}{}", prefix, i);
        if !map.contains_key(&id) {
            return id;
        }
        i += 1;
    }
}

// ── Force field helpers ──────────────────────────────────────────────────────

/// Set a single scalar field on a force element by field name. Returns true if successful.
pub(crate) fn set_force_field(force: &mut ForceElement, field: &str, value: f64) -> bool {
    match force {
        ForceElement::LinearSpring(e) => match field {
            "stiffness" => { e.stiffness = value; true }
            "free_length" => { e.free_length = value; true }
            _ => false,
        },
        ForceElement::TorsionSpring(e) => match field {
            "stiffness" => { e.stiffness = value; true }
            "free_angle" => { e.free_angle = value; true }
            _ => false,
        },
        ForceElement::LinearDamper(e) => match field {
            "damping" => { e.damping = value; true }
            _ => false,
        },
        ForceElement::RotaryDamper(e) => match field {
            "damping" => { e.damping = value; true }
            _ => false,
        },
        ForceElement::GasSpring(e) => match field {
            "initial_force" => { e.initial_force = value; true }
            "extended_length" => { e.extended_length = value; true }
            "stroke" => { e.stroke = value; true }
            _ => false,
        },
        ForceElement::Motor(e) => match field {
            "stall_torque" => { e.stall_torque = value; true }
            "no_load_speed" => { e.no_load_speed = value; true }
            _ => false,
        },
        ForceElement::ExternalForce(e) => match field {
            "force_x" => { e.force[0] = value; true }
            "force_y" => { e.force[1] = value; true }
            _ => false,
        },
        ForceElement::ExternalTorque(e) => match field {
            "torque" => { e.torque = value; true }
            _ => false,
        },
        ForceElement::BearingFriction(e) => match field {
            "constant_drag" => { e.constant_drag = value; true }
            "viscous_coeff" => { e.viscous_coeff = value; true }
            "coulomb_coeff" => { e.coulomb_coeff = value; true }
            _ => false,
        },
        ForceElement::JointLimit(e) => match field {
            "stiffness" => { e.stiffness = value; true }
            _ => false,
        },
        ForceElement::LinearActuator(e) => match field {
            "force" => { e.force = value; true }
            "speed_limit" => { e.speed_limit = value; true }
            "stroke_min" => { e.stroke_min = value; true }
            "stroke_max" => { e.stroke_max = value; true }
            "end_stop_stiffness" => { e.end_stop_stiffness = value; true }
            "end_stop_damping" => { e.end_stop_damping = value; true }
            "end_stop_restitution" => { e.end_stop_restitution = value; true }
            _ => false,
        },
        _ => false,
    }
}

/// List sweepable field names for a force element.
pub(crate) fn force_sweepable_fields(force: &ForceElement) -> Vec<String> {
    match force {
        ForceElement::LinearSpring(_) => vec!["stiffness".into(), "free_length".into()],
        ForceElement::TorsionSpring(_) => vec!["stiffness".into(), "free_angle".into()],
        ForceElement::LinearDamper(_) => vec!["damping".into()],
        ForceElement::RotaryDamper(_) => vec!["damping".into()],
        ForceElement::GasSpring(_) => vec!["initial_force".into(), "extended_length".into(), "stroke".into()],
        ForceElement::Motor(_) => vec!["stall_torque".into(), "no_load_speed".into()],
        ForceElement::ExternalForce(_) => vec!["force_x".into(), "force_y".into()],
        ForceElement::ExternalTorque(_) => vec!["torque".into()],
        ForceElement::BearingFriction(_) => vec!["constant_drag".into(), "viscous_coeff".into(), "coulomb_coeff".into()],
        ForceElement::JointLimit(_) => vec!["stiffness".into()],
        ForceElement::LinearActuator(_) => vec![
            "force".into(), "speed_limit".into(),
            "stroke_min".into(), "stroke_max".into(),
            "end_stop_stiffness".into(), "end_stop_damping".into(), "end_stop_restitution".into(),
        ],
        _ => Vec::new(),
    }
}

/// Find the revolute joint that connects the driver body pair, if any.
///
/// Returns the joint ID as a `String`, or `None` if there is no driver
/// or no matching revolute joint.
pub(crate) fn detect_driver_joint_id(mech: &Mechanism) -> Option<String> {
    let (a, b) = mech.driver_body_pair()?;
    mech.joints()
        .iter()
        .find(|j| {
            j.is_revolute()
                && ((j.body_i_id() == a && j.body_j_id() == b)
                    || (j.body_i_id() == b && j.body_j_id() == a))
        })
        .map(|j| j.id().to_string())
}

// ── AppState blueprint methods ───────────────────────────────────────────────

impl AppState {
    // ── Blueprint rebuild pipeline ───────────────────────────────────────

    /// Rebuild Mechanism from the current blueprint, solve at current angle.
    /// Called after every edit operation. Pauses animation to prevent the solver
    /// from fighting with mid-edit mechanism state.
    pub fn rebuild(&mut self) {
        self.playing = false;

        // Sync mounting angle to blueprint before building so it persists in saves.
        if let Some(ref mut bp) = self.blueprint {
            bp.mounting_angle = self.mounting_angle;
        }

        let Some(bp) = &self.blueprint else { return };

        // Build mechanism from blueprint
        let mut mech = match load_mechanism_unbuilt_from_json(bp) {
            Ok(m) => m,
            Err(e) => {
                log::warn!("Blueprint rebuild failed: {}", e);
                self.solver_status = SolverStatus {
                    converged: false,
                    residual_norm: f64::NAN,
                    iterations: 0,
                };
                return;
            }
        };

        if let Err(e) = mech.build() {
            log::warn!("Mechanism build failed: {}", e);
            self.solver_status = SolverStatus {
                converged: false,
                residual_norm: f64::NAN,
                iterations: 0,
            };
            return;
        }

        // Extract driver params from blueprint.
        // For revolute drivers: omega = angular velocity, theta_0 = initial angle.
        // For linear drivers: omega = velocity (m/s), theta_0 = initial length (m).
        // This dual-use allows solve_at_angle/solve_at_stroke to share the same
        // time formula: t = (value - theta_0) / omega.
        self.driver_omega = 2.0 * std::f64::consts::PI;
        self.driver_theta_0 = 0.0;
        // Check blueprint revolute drivers for actual values
        if let Some(driver) = bp.drivers.values().next() {
            match driver {
                DriverJson::ConstantSpeed { omega, theta_0, .. } => {
                    self.driver_omega = *omega;
                    self.driver_theta_0 = *theta_0;
                }
                DriverJson::Expression { .. } => {
                    // Expression drivers use f(t) directly; omega/theta_0
                    // aren't meaningful, so keep defaults for angle-slider
                    // mapping (omega=2*pi means 1 rev/s, theta_0=0).
                }
            }
        }
        // Override with linear driver params when present.
        // For constant-velocity: velocity -> omega, length_0 -> theta_0.
        // For cosine: omega=2*PI, theta_0=phase.
        if let Some(ld) = mech.linear_drivers().first() {
            use crate::core::driver::DriverMeta;
            match ld.meta() {
                Some(DriverMeta::LinearLength { velocity, length_0 }) => {
                    self.driver_omega = *velocity;
                    self.driver_theta_0 = *length_0;
                    if self.driver_stroke == 0.0 {
                        self.driver_stroke = *length_0;
                    }
                }
                Some(DriverMeta::CosineStroke { stroke_min, stroke_max, initial_length }) => {
                    let mid = (stroke_min + stroke_max) / 2.0;
                    let amp = (stroke_max - stroke_min) / 2.0;
                    let phase = if amp.abs() < 1e-15 {
                        0.0
                    } else {
                        ((initial_length - mid) / amp).clamp(-1.0, 1.0).acos()
                    };
                    self.driver_omega = 2.0 * std::f64::consts::PI;
                    self.driver_theta_0 = phase;
                    if self.driver_stroke == 0.0 {
                        self.driver_stroke = *initial_length;
                    }
                }
                _ => {}
            }
        }

        // Detect driven joint
        self.driver_joint_id = detect_driver_joint_id(&mech);

        // Solve at current position using last_good_q as initial guess.
        // For linear drivers, use driver_stroke; for revolute, use driver_angle.
        let driver_value = if mech.n_linear_drivers() > 0 {
            self.driver_stroke
        } else {
            self.driver_angle
        };
        let t = if self.driver_omega.abs() > f64::EPSILON {
            (driver_value - self.driver_theta_0) / self.driver_omega
        } else {
            0.0
        };

        // Try solving with last_good_q if it has the right dimension
        let try_q = if self.last_good_q.len() == mech.state().n_coords() {
            self.last_good_q.clone()
        } else {
            mech.state().make_q()
        };

        if !self.solve_and_update(&mech, &try_q, t, 1e-10, 50, None) {
            // First attempt failed — a NaN residual means the solver errored
            // (as opposed to converging to a loose solution), so retry from
            // a zero initial guess with more iterations.
            if self.solver_status.residual_norm.is_nan() {
                let q0 = mech.state().make_q();
                if !self.solve_and_update(&mech, &q0, t, 1e-10, 100, None) {
                    // If the retry didn't converge either, reset q to
                    // the zero guess so the display stays reasonable.
                    if !self.solver_status.residual_norm.is_nan() {
                        self.q = q0;
                    }
                }
            }
        }

        // Ensure q always matches the new mechanism's dimension.
        let n = mech.state().n_coords();
        if self.q.len() != n {
            self.q = mech.state().make_q();
            self.last_good_q = self.q.clone();
            self.q_at_zero = self.q.clone();
        }

        // Re-extract stroke range from actuator after build so the sweep
        // UI always reflects the latest stroke limits.
        self.sweep_stroke_min = 0.0;
        self.sweep_stroke_max = 0.0;
        for force in mech.forces() {
            if let ForceElement::LinearActuator(act) = force {
                if act.stroke_min > 0.0 || act.stroke_max > 0.0 {
                    self.sweep_stroke_min = act.stroke_min;
                    self.sweep_stroke_max = act.stroke_max;
                    break;
                }
            }
        }

        self.mechanism = Some(mech);
        self.compute_forces(t);
        self.update_grashof();
        self.compute_validation();
        self.mark_sweep_dirty();
    }

    // ── Blueprint edit operations ────────────────────────────────────────

    /// Move an attachment point on a body in the blueprint.
    /// This is the core drag operation for the editor.
    pub fn move_attachment_point(
        &mut self,
        body_id: &str,
        point_name: &str,
        new_x: f64,
        new_y: f64,
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(body_id) {
            if let Some(pt) = body.attachment_points.get_mut(point_name) {
                *pt = [new_x, new_y];
            }
        }
        self.rebuild();
    }

    /// Set the distance between two attachment points on a body, maintaining direction.
    ///
    /// Moves `point_b` along the vector from `point_a` to `point_b` so that
    /// the new distance equals `new_length`. If the points are coincident,
    /// moves `point_b` along the +X direction.
    pub fn set_link_length(
        &mut self,
        body_id: &str,
        point_a: &str,
        point_b: &str,
        new_length: f64,
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get(body_id) else { return };
        let Some(&pa) = body.attachment_points.get(point_a) else { return };
        let Some(&pb) = body.attachment_points.get(point_b) else { return };

        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let current_len = (dx * dx + dy * dy).sqrt();

        let (ux, uy) = if current_len > 1e-12 {
            (dx / current_len, dy / current_len)
        } else {
            (1.0, 0.0)
        };

        let new_pb = [pa[0] + ux * new_length, pa[1] + uy * new_length];

        if let Some(body) = bp.bodies.get_mut(body_id) {
            if let Some(pt) = body.attachment_points.get_mut(point_b) {
                *pt = new_pb;
            }
        }
        self.rebuild();
    }

    /// Set the orientation (angle from point_a to point_b) while preserving length.
    ///
    /// Moves `point_b` to the new angle relative to `point_a`, keeping the
    /// distance between them the same.
    pub fn set_link_orientation(
        &mut self,
        body_id: &str,
        point_a: &str,
        point_b: &str,
        new_angle_rad: f64,
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get(body_id) else { return };
        let Some(&pa) = body.attachment_points.get(point_a) else { return };
        let Some(&pb) = body.attachment_points.get(point_b) else { return };

        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let current_len = (dx * dx + dy * dy).sqrt();
        if current_len < 1e-12 {
            return;
        }

        let new_pb = [
            pa[0] + current_len * new_angle_rad.cos(),
            pa[1] + current_len * new_angle_rad.sin(),
        ];

        if let Some(body) = bp.bodies.get_mut(body_id) {
            if let Some(pt) = body.attachment_points.get_mut(point_b) {
                *pt = new_pb;
            }
        }
        self.rebuild();
    }

    /// Set mass property on a body in the blueprint.
    ///
    /// Mass does not affect the kinematic constraint equations, so a full
    /// `rebuild()` is unnecessary. We update the blueprint **and** the live
    /// mechanism directly, then recompute forces and mark the sweep dirty
    /// (force/energy curves depend on mass).
    pub fn set_body_mass(&mut self, body_id: &str, mass: f64) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(body_id) {
            body.mass = mass;
        }
        // Patch the live mechanism so we skip the full JSON roundtrip.
        if let Some(mech) = &mut self.mechanism {
            if let Some(body) = mech.body_mut(body_id) {
                body.mass = mass;
            }
        }
        self.recompute_dynamics();
    }

    /// Set moment of inertia on a body in the blueprint.
    ///
    /// Izz does not affect kinematics. Same lightweight update as `set_body_mass`.
    pub fn set_body_izz(&mut self, body_id: &str, izz: f64) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(body_id) {
            body.izz_cg = izz;
        }
        if let Some(mech) = &mut self.mechanism {
            if let Some(body) = mech.body_mut(body_id) {
                body.izz_cg = izz;
            }
        }
        self.recompute_dynamics();
    }

    /// Recompute force results and mark sweep dirty without rebuilding the
    /// mechanism. Used after mass/inertia/gravity changes that don't alter
    /// the kinematic structure.
    pub(crate) fn recompute_dynamics(&mut self) {
        let t = if self.driver_omega.abs() > f64::EPSILON {
            (self.driver_angle - self.driver_theta_0) / self.driver_omega
        } else {
            0.0
        };
        self.compute_forces(t);
        self.mark_sweep_dirty();
    }

    /// Sync gravity force element on the mechanism with the `gravity_magnitude` value.
    /// Adds or updates `ForceElement::Gravity` when magnitude > 0; removes it when 0.
    pub fn sync_gravity(&mut self) {
        let Some(mech) = &mut self.mechanism else {
            return;
        };
        let has_gravity = mech.forces().iter().any(|f| matches!(f, ForceElement::Gravity(_)));
        if self.gravity_magnitude > 0.0 {
            let g = self.gravity_magnitude;
            let theta = self.mounting_angle;
            let g_elem = GravityElement {
                g_vector: [-g * theta.sin(), -g * theta.cos()],
            };
            if has_gravity {
                if let Some(idx) = mech
                    .forces()
                    .iter()
                    .position(|f| matches!(f, ForceElement::Gravity(_)))
                {
                    mech.replace_force(idx, ForceElement::Gravity(g_elem));
                }
            } else {
                mech.add_force(ForceElement::Gravity(g_elem));
            }
        } else if has_gravity {
            if let Some(idx) = mech
                .forces()
                .iter()
                .position(|f| matches!(f, ForceElement::Gravity(_)))
            {
                mech.remove_force(idx);
            }
        }
    }

    /// Compute static force results (joint reactions + driver torque) at the
    /// current pose. Called after each successful position solve.
    ///
    /// Uses the statics solver (no inertial effects) since the GUI currently
    /// operates at quasi-static conditions. Falls back gracefully on failure,
    /// clearing force results rather than propagating errors.
    pub(crate) fn compute_forces(&mut self, t: f64) {
        self.sync_gravity();
        let Some(mech) = &self.mechanism else {
            self.force_results = ForceResults::default();
            return;
        };

        // Guard: only compute forces when the solver converged and q has the
        // correct dimension (prevents panics during partial rebuilds).
        if !self.solver_status.converged
            || self.q.len() != mech.state().n_coords()
            || mech.n_drivers() == 0
        {
            self.force_results = ForceResults::default();
            return;
        }

        let statics_result = match solve_statics(mech, &self.q, t) {
            Ok(r) => r,
            Err(_) => {
                self.force_results = ForceResults::default();
                return;
            }
        };

        let reactions = extract_reactions(mech, &statics_result);

        // Extract driver torque.
        let driver_torque = get_driver_reactions(&reactions)
            .first()
            .map(|r| r.effort);

        // Extract per-joint reaction forces.
        let mut joint_reactions = HashMap::new();
        for jr in get_joint_reactions(&reactions) {
            joint_reactions.insert(
                jr.joint_id.clone(),
                (jr.force_global[0], jr.force_global[1]),
            );
        }

        // Compute mechanical advantage via velocity solve.
        let ma = if let Ok(q_dot) = solve_velocity(mech, &self.q, t) {
            // The driver body pair gives (body_i, body_j) where body_j is
            // the driven body (crank). Find the last non-driver moving body
            // as the output body.
            if let Some((_body_i, driver_body)) = mech.driver_body_pair() {
                let output_body = mech.body_order().iter()
                    .filter(|b| b.as_str() != driver_body)
                    .last();
                if let Some(out_id) = output_body {
                    mechanical_advantage(
                        mech.state(), &q_dot,
                        driver_body, out_id,
                        VelocityCoord::Theta, VelocityCoord::Theta,
                    ).map(|r| r.ma)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Per-element force contribution breakdown.
        let n = self.q.len();
        let q_dot_zero = DVector::zeros(n);
        let mut contribs: Vec<(String, f64)> = evaluate_contributions(mech, &self.q, &q_dot_zero, t)
            .iter()
            .map(|c| (c.type_name.clone(), c.q_norm))
            .collect();
        contribs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Virtual work cross-check.
        let vw_check = if let Some(dt) = driver_torque {
            virtual_work_check(mech, &self.q, t, dt, 1e-4).ok().map(|vw| {
                (vw.input_torque, vw.lagrange_torque, vw.agrees)
            })
        } else {
            None
        };

        self.force_results = ForceResults {
            driver_torque,
            joint_reactions,
            condition_number: Some(statics_result.condition_number),
            is_overconstrained: statics_result.is_overconstrained,
            mechanical_advantage: ma,
            force_contributions: contribs,
            virtual_work_check: vw_check,
        };
    }

    /// Update cached Grashof classification and crank recommendation from
    /// the current mechanism.
    ///
    /// Only produces results for 4-bar mechanisms (3 moving bodies + ground,
    /// 4 revolute joints). Sets both `grashof_result` and
    /// `crank_recommendation` to `None` for all other mechanism topologies.
    pub(crate) fn update_grashof(&mut self) {
        let Some(mech) = &self.mechanism else {
            self.grashof_result = None;
            self.crank_recommendation = None;
            return;
        };

        match detect_fourbar_links(mech) {
            Some((crank_len, coupler_len, rocker_len, ground_len)) => {
                self.grashof_result = Some(check_grashof(
                    ground_len,
                    crank_len,
                    coupler_len,
                    rocker_len,
                    1e-10,
                ));
                self.crank_recommendation = Some(
                    crate::analysis::crank_selection::recommend_crank(
                        ground_len,
                        crank_len,
                        coupler_len,
                        rocker_len,
                    ),
                );
            }
            None => {
                self.grashof_result = None;
                self.crank_recommendation = None;
            }
        }
    }

    /// Compute validation warnings from the current mechanism state.
    ///
    /// Called after each rebuild to update `self.validation_warnings`.
    pub fn compute_validation(&mut self) {
        let mut warnings = super::ValidationWarnings::default();

        if let Some(mech) = &self.mechanism {
            // DOF check: 3 * n_moving - n_constraints should be 0
            let n_coords = mech.state().n_coords() as isize;
            let n_constraints = mech.n_constraints() as isize;
            let dof = n_coords - n_constraints;
            if dof != 0 {
                warnings.dof_warning = Some(format!(
                    "DOF = {} (coords={}, constraints={})",
                    dof, n_coords, n_constraints
                ));
            }

            // Missing driver check
            warnings.missing_driver = mech.n_drivers() == 0;

            // Disconnected body check: a body that has no joints connecting to it
            if let Some(bp) = &self.blueprint {
                for body_id in bp.bodies.keys() {
                    if body_id == GROUND_ID {
                        continue;
                    }
                    let connected = bp.joints.values().any(|j| {
                        let (bi, bj) = joint_body_ids(j);
                        bi == body_id || bj == body_id
                    });
                    if !connected {
                        warnings.disconnected_bodies.push(body_id.clone());
                    }
                }
                warnings.disconnected_bodies.sort();
            }
        }

        self.validation_warnings = warnings;
    }

    // ── Grid auto-spacing ─────────────────────────────────────────────────

    /// Set grid spacing based on the bounding box of all attachment points
    /// in the current blueprint. Picks a "clean" spacing that gives roughly
    /// 10-20 grid cells across the largest dimension.
    pub fn auto_grid_spacing(&mut self) {
        let Some(bp) = &self.blueprint else { return };

        let mut x_min = f64::INFINITY;
        let mut x_max = f64::NEG_INFINITY;
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut count = 0usize;

        for body in bp.bodies.values() {
            for pt in body.attachment_points.values() {
                x_min = x_min.min(pt[0]);
                x_max = x_max.max(pt[0]);
                y_min = y_min.min(pt[1]);
                y_max = y_max.max(pt[1]);
                count += 1;
            }
        }

        if count < 2 {
            return; // Not enough points to determine scale
        }

        let extent = (x_max - x_min).max(y_max - y_min);
        if extent <= 0.0 || !extent.is_finite() {
            return;
        }

        // Target ~10 grid cells across the largest dimension.
        // Round down to the nearest "clean" value from a fixed set.
        let raw = extent / 10.0;
        const CLEAN: [f64; 12] = [
            10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001,
        ];
        self.grid.spacing_m = CLEAN
            .iter()
            .copied()
            .find(|&c| c <= raw)
            .unwrap_or(0.001);
    }

    // ── Mount point CRUD ─────────────────────────────────────────────────

    /// Add a named mount point to a body in the blueprint.
    ///
    /// Uses `entry(...).or_insert(...)` so existing names are not overwritten.
    /// Pushes undo and rebuilds.
    pub fn add_mount_point(&mut self, body_id: &str, name: &str, pos: [f64; 2]) {
        self.push_undo();
        {
            let Some(bp) = &mut self.blueprint else { return };
            if let Some(body) = bp.bodies.get_mut(body_id) {
                body.mount_points.entry(name.to_string()).or_insert(pos);
            }
        }
        self.rebuild();
    }

    /// Remove a named mount point from a body in the blueprint.
    ///
    /// Also clears any force element references to this mount point (setting
    /// `point_X_name` to `None`). Returns the number of force references cleared.
    /// Pushes undo and rebuilds.
    pub fn delete_mount_point(&mut self, body_id: &str, name: &str) -> usize {
        self.push_undo();
        let cleared_count;
        {
            let Some(bp) = &mut self.blueprint else { return 0 };
            if let Some(body) = bp.bodies.get_mut(body_id) {
                body.mount_points.remove(name);
            }
            cleared_count = Self::clear_force_point_refs(&mut bp.forces, body_id, name);
        }
        self.rebuild();
        cleared_count
    }

    /// Rename a mount point on a body in the blueprint.
    ///
    /// Also updates all force element references that named the old point.
    /// Pushes undo and rebuilds.
    pub fn rename_mount_point(&mut self, body_id: &str, old_name: &str, new_name: &str) {
        self.push_undo();
        {
            let Some(bp) = &mut self.blueprint else { return };
            if let Some(body) = bp.bodies.get_mut(body_id) {
                if let Some(pos) = body.mount_points.remove(old_name) {
                    body.mount_points.insert(new_name.to_string(), pos);
                }
            }
            Self::rename_force_point_refs(&mut bp.forces, body_id, old_name, new_name);
        }
        self.rebuild();
    }

    /// Update the local position of a named mount point on a body.
    ///
    /// Continuous tweak — no undo snapshot pushed.
    pub fn update_mount_point_position(&mut self, body_id: &str, name: &str, pos: [f64; 2]) {
        {
            let Some(bp) = &mut self.blueprint else { return };
            if let Some(body) = bp.bodies.get_mut(body_id) {
                if let Some(pt) = body.mount_points.get_mut(name) {
                    *pt = pos;
                }
            }
        }
        self.rebuild();
    }

    // ── Mount point cascade helpers ──────────────────────────────────────

    /// Clear `point_X_name` references on all force elements that point at
    /// `(body_id, point_name)`. Returns the count of references cleared.
    fn clear_force_point_refs(forces: &mut [ForceElement], body_id: &str, point_name: &str) -> usize {
        let mut count = 0usize;
        for force in forces.iter_mut() {
            match force {
                ForceElement::LinearSpring(s) => {
                    if s.body_a == body_id && s.point_a_name.as_deref() == Some(point_name) { s.point_a_name = None; count += 1; }
                    if s.body_b == body_id && s.point_b_name.as_deref() == Some(point_name) { s.point_b_name = None; count += 1; }
                }
                ForceElement::LinearDamper(d) => {
                    if d.body_a == body_id && d.point_a_name.as_deref() == Some(point_name) { d.point_a_name = None; count += 1; }
                    if d.body_b == body_id && d.point_b_name.as_deref() == Some(point_name) { d.point_b_name = None; count += 1; }
                }
                ForceElement::GasSpring(g) => {
                    if g.body_a == body_id && g.point_a_name.as_deref() == Some(point_name) { g.point_a_name = None; count += 1; }
                    if g.body_b == body_id && g.point_b_name.as_deref() == Some(point_name) { g.point_b_name = None; count += 1; }
                }
                ForceElement::LinearActuator(a) => {
                    if a.body_a == body_id && a.point_a_name.as_deref() == Some(point_name) { a.point_a_name = None; count += 1; }
                    if a.body_b == body_id && a.point_b_name.as_deref() == Some(point_name) { a.point_b_name = None; count += 1; }
                }
                ForceElement::ExternalForce(e) => {
                    if e.body_id == body_id && e.local_point_name.as_deref() == Some(point_name) { e.local_point_name = None; count += 1; }
                }
                _ => {}
            }
        }
        count
    }

    /// Update `point_X_name` references on all force elements that point at
    /// `(body_id, old_name)` to use `new_name` instead.
    fn rename_force_point_refs(forces: &mut [ForceElement], body_id: &str, old_name: &str, new_name: &str) {
        for force in forces.iter_mut() {
            match force {
                ForceElement::LinearSpring(s) => {
                    if s.body_a == body_id && s.point_a_name.as_deref() == Some(old_name) { s.point_a_name = Some(new_name.to_string()); }
                    if s.body_b == body_id && s.point_b_name.as_deref() == Some(old_name) { s.point_b_name = Some(new_name.to_string()); }
                }
                ForceElement::LinearDamper(d) => {
                    if d.body_a == body_id && d.point_a_name.as_deref() == Some(old_name) { d.point_a_name = Some(new_name.to_string()); }
                    if d.body_b == body_id && d.point_b_name.as_deref() == Some(old_name) { d.point_b_name = Some(new_name.to_string()); }
                }
                ForceElement::GasSpring(g) => {
                    if g.body_a == body_id && g.point_a_name.as_deref() == Some(old_name) { g.point_a_name = Some(new_name.to_string()); }
                    if g.body_b == body_id && g.point_b_name.as_deref() == Some(old_name) { g.point_b_name = Some(new_name.to_string()); }
                }
                ForceElement::LinearActuator(a) => {
                    if a.body_a == body_id && a.point_a_name.as_deref() == Some(old_name) { a.point_a_name = Some(new_name.to_string()); }
                    if a.body_b == body_id && a.point_b_name.as_deref() == Some(old_name) { a.point_b_name = Some(new_name.to_string()); }
                }
                ForceElement::ExternalForce(e) => {
                    if e.body_id == body_id && e.local_point_name.as_deref() == Some(old_name) { e.local_point_name = Some(new_name.to_string()); }
                }
                _ => {}
            }
        }
    }

    /// Add a force element to the blueprint.
    ///
    /// Pushes undo, appends the element, and rebuilds.
    pub fn add_force_element(&mut self, force: ForceElement) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        bp.forces.push(force);
        self.rebuild();
    }

    /// Remove a force element from the blueprint by index.
    ///
    /// Pushes undo, removes the element, and rebuilds.
    /// No-op if `index` is out of bounds.
    pub fn remove_force_element(&mut self, index: usize) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        if index >= bp.forces.len() {
            return;
        }
        bp.forces.remove(index);
        self.rebuild();
    }

    /// Add a point mass to a body in the blueprint.
    ///
    /// Pushes undo, appends the point mass, and rebuilds (which recomputes
    /// composite mass, CG, and Izz via parallel axis theorem).
    pub fn add_point_mass(&mut self, body_id: &str, mass: f64, local_pos: [f64; 2]) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get_mut(body_id) else { return };
        body.point_masses.push(crate::io::PointMassJson {
            mass,
            local_pos,
        });
        self.rebuild();
    }

    /// Remove a point mass from a body in the blueprint by index.
    ///
    /// Pushes undo, removes the point mass, and rebuilds.
    pub fn remove_point_mass(&mut self, body_id: &str, index: usize) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get_mut(body_id) else { return };
        if index < body.point_masses.len() {
            body.point_masses.remove(index);
            self.rebuild();
        }
    }

    /// Update the slide axis of a prismatic joint in the blueprint.
    ///
    /// Continuous parameter tweak — no undo snapshot pushed.
    pub fn update_prismatic_axis(&mut self, joint_id: &str, axis: [f64; 2]) {
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(joint) = bp.joints.get_mut(joint_id) {
            if let JointJson::Prismatic { axis_local_i, .. } = joint {
                *axis_local_i = axis;
            }
        }
        self.rebuild();
    }

    /// Replace a force element in the blueprint at the given index.
    ///
    /// Intended for continuous parameter tweaks (e.g. DragValue), so no undo
    /// snapshot is pushed.
    /// No-op if `index` is out of bounds.
    pub fn update_force_element(&mut self, index: usize, force: ForceElement) {
        let Some(bp) = &mut self.blueprint else { return };
        if index >= bp.forces.len() {
            return;
        }
        bp.forces[index] = force;
        self.rebuild();
    }

    /// Enumerate all sweepable parameters from the current blueprint.
    pub fn available_parameters(&self) -> Vec<super::SweepParameter> {
        let Some(ref bp) = self.blueprint else { return Vec::new() };
        let mut params = Vec::new();

        // Body parameters (skip ground)
        for (body_id, body) in &bp.bodies {
            if body_id == GROUND_ID { continue; }
            params.push(super::SweepParameter::BodyMass(body_id.clone()));
            params.push(super::SweepParameter::BodyIzz(body_id.clone()));
            let mut pts: Vec<_> = body.attachment_points.keys().collect();
            pts.sort();
            for pt_name in pts {
                params.push(super::SweepParameter::AttachmentX(body_id.clone(), pt_name.clone()));
                params.push(super::SweepParameter::AttachmentY(body_id.clone(), pt_name.clone()));
            }
        }

        // Ground attachment point positions
        if let Some(ground) = bp.bodies.get(GROUND_ID) {
            let mut pts: Vec<_> = ground.attachment_points.keys().collect();
            pts.sort();
            for pt_name in pts {
                params.push(super::SweepParameter::AttachmentX(GROUND_ID.to_string(), pt_name.clone()));
                params.push(super::SweepParameter::AttachmentY(GROUND_ID.to_string(), pt_name.clone()));
            }
        }

        // Force element parameters
        for (idx, force) in bp.forces.iter().enumerate() {
            for field in force_sweepable_fields(force) {
                params.push(super::SweepParameter::ForceParam(idx, field));
            }
        }

        // Driver omega
        params.push(super::SweepParameter::DriverOmega);

        params
    }

    /// Apply a parameter value to a blueprint clone. Returns None if the
    /// parameter path doesn't resolve.
    pub(crate) fn set_parameter_on_blueprint(
        bp: &mut MechanismJson,
        param: &super::SweepParameter,
        value: f64,
        omega: &mut f64,
    ) -> bool {
        match param {
            super::SweepParameter::BodyMass(id) => {
                if let Some(body) = bp.bodies.get_mut(id) {
                    body.mass = value;
                    return true;
                }
            }
            super::SweepParameter::BodyIzz(id) => {
                if let Some(body) = bp.bodies.get_mut(id) {
                    body.izz_cg = value;
                    return true;
                }
            }
            super::SweepParameter::AttachmentX(body_id, point_name) => {
                if let Some(body) = bp.bodies.get_mut(body_id) {
                    if let Some(pt) = body.attachment_points.get_mut(point_name) {
                        pt[0] = value;
                        return true;
                    }
                }
            }
            super::SweepParameter::AttachmentY(body_id, point_name) => {
                if let Some(body) = bp.bodies.get_mut(body_id) {
                    if let Some(pt) = body.attachment_points.get_mut(point_name) {
                        pt[1] = value;
                        return true;
                    }
                }
            }
            super::SweepParameter::ForceParam(idx, field) => {
                if let Some(force) = bp.forces.get_mut(*idx) {
                    return set_force_field(force, field, value);
                }
            }
            super::SweepParameter::DriverOmega => {
                *omega = value;
                return true;
            }
        }
        false
    }

    /// Helper: look up attachment point local coordinates from the blueprint.
    pub(crate) fn resolve_point_coords(bp: &MechanismJson, body_id: &str, point_name: &str) -> [f64; 2] {
        bp.bodies
            .get(body_id)
            .and_then(|b| b.attachment_points.get(point_name))
            .copied()
            .unwrap_or([0.0, 0.0])
    }

    // ── Sweep computation ─────────────────────────────────────────────────

    /// Mark sweep data as stale, starting the debounce timer.
    ///
    /// The actual recomputation happens in the update loop after a 200ms
    /// debounce delay, so rapid edits don't cause jank.
    pub fn mark_sweep_dirty(&mut self) {
        self.sweep_dirty = true;
        // sweep_dirty_since is set in the update loop using egui time
        // (we can't use std::time::Instant because it panics on WASM).
        // If it's already set, keep the existing timestamp for proper debounce.
    }

    pub fn compute_sweep(&mut self) {
        self.sweep_dirty = false;
        self.sweep_dirty_since = None;

        if self.mechanism.is_none() {
            self.sweep_data = None;
            return;
        }
        // Guard: need at least one driver (revolute or linear) and one moving body.
        {
            let mech = self.mechanism.as_ref().unwrap();
            if (mech.n_drivers() == 0 && mech.n_linear_drivers() == 0)
                || mech.body_order().is_empty()
            {
                self.sweep_data = None;
                return;
            }
        }
        self.sync_gravity();

        // Detect linear driver and extract parameters for the sweep.
        // For a revolute driver, omega and theta_0 come from AppState fields.
        // For a linear driver, omega maps to velocity and theta_0 maps to length_0,
        // read directly from the driver metadata.
        let mech = self.mechanism.as_ref().unwrap();
        let (omega, theta_0) = if let Some(ld) = mech.linear_drivers().first() {
            use crate::core::driver::DriverMeta;
            match ld.meta() {
                Some(DriverMeta::LinearLength { velocity, length_0 }) => (*velocity, *length_0),
                _ => (self.driver_omega, self.driver_theta_0),
            }
        } else {
            (self.driver_omega, self.driver_theta_0)
        };

        // Always start the sweep at t=0. For angle mode, q_at_zero is the known-good
        // state at 0 degrees. For stroke mode, it's the state at the initial stroke.
        let q_start = if self.q_at_zero.len() == self.last_good_q.len() && self.q_at_zero.len() > 0 {
            self.q_at_zero.clone()
        } else {
            self.last_good_q.clone()
        };

        let has_linear_driver = mech.n_linear_drivers() > 0;
        let sweep_range = if self.sweep_range_enabled {
            if has_linear_driver {
                Some((self.sweep_stroke_min, self.sweep_stroke_max))
            } else {
                Some((self.sweep_angle_min_deg, self.sweep_angle_max_deg))
            }
        } else {
            None
        };
        let (data, q_zero) = compute_sweep_data(mech, &q_start, omega, theta_0, self.gravity_magnitude, sweep_range);
        self.sweep_data = Some(data);
        self.q_at_zero = q_zero;
    }
}
