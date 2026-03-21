//! Force element types for planar mechanisms.
//!
//! Each variant represents a force element that contributes to the
//! generalized force vector Q via the virtual work principle.

use std::collections::HashMap;

use meval;
use nalgebra::{DVector, Vector2};
use serde::{Deserialize, Serialize};

use crate::core::body::Body;
use crate::core::state::{State, GROUND_ID};
use crate::forces::helpers::{body_torque_to_q, point_force_to_q};

// ── Serde default helpers ────────────────────────────────────────────────────

fn default_polytropic_exp() -> f64 {
    1.0
}
fn default_v_threshold() -> f64 {
    0.01
}
fn default_restitution() -> f64 {
    0.5
}
fn default_direction() -> f64 {
    1.0
}

// ── Time modulation ──────────────────────────────────────────────────────────

/// Time modulation for external loads.
///
/// Multiplies the base force/torque by a time-dependent factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "modulation_type")]
pub enum TimeModulation {
    /// Constant (default, no modulation). Factor = 1.0 always.
    Constant,
    /// Sinusoidal: factor = sin(omega * t + phase).
    Sinusoidal { omega: f64, phase: f64 },
    /// Step: zero before t_on, full after t_on.
    Step { t_on: f64 },
    /// Ramp: linearly from 0 to 1 over [t_start, t_end].
    Ramp { t_start: f64, t_end: f64 },
    /// User-defined expression of time: e.g., "sin(2*pi*t)" or "1 - exp(-t/0.5)".
    /// Uses the `meval` crate for parsing and evaluation.
    Expression { expr: String },
}

impl Default for TimeModulation {
    fn default() -> Self {
        TimeModulation::Constant
    }
}

impl TimeModulation {
    /// Compute the modulation factor at time `t`.
    ///
    /// Note: for the `Expression` variant this re-parses the expression string
    /// on every call. In hot loops prefer [`compile`] to pre-parse once.
    pub fn factor(&self, t: f64) -> f64 {
        match self {
            TimeModulation::Constant => 1.0,
            TimeModulation::Sinusoidal { omega, phase } => (omega * t + phase).sin(),
            TimeModulation::Step { t_on } => {
                if t >= *t_on {
                    1.0
                } else {
                    0.0
                }
            }
            TimeModulation::Ramp { t_start, t_end } => {
                if t <= *t_start {
                    0.0
                } else if t >= *t_end {
                    1.0
                } else {
                    (t - t_start) / (t_end - t_start)
                }
            }
            TimeModulation::Expression { expr } => {
                // Parse and evaluate (re-parse each call for thread safety,
                // same pattern as ExprEval in the driver module).
                match expr.parse::<meval::Expr>() {
                    Ok(parsed) => match parsed.bind("t") {
                        Ok(f) => {
                            let val = f(t);
                            if val.is_finite() { val } else { 0.0 }
                        }
                        Err(_) => {
                            log::warn!("TimeModulation: failed to bind variable 't' in expression '{expr}' — force disabled");
                            0.0
                        }
                    },
                    Err(_) => {
                        log::warn!("TimeModulation: failed to parse expression '{expr}' — force disabled");
                        0.0
                    }
                }
            }
        }
    }

    /// Pre-compile this modulation into a closure that can be called repeatedly
    /// without re-parsing expression strings.
    ///
    /// For `Constant`, `Sinusoidal`, `Step`, and `Ramp` variants this simply
    /// captures the parameters. For `Expression` the meval string is parsed
    /// once and the bound closure is captured, eliminating the O(n) parse on
    /// every evaluation.
    ///
    /// The returned closure is **not** `Send`/`Sync` (meval closures aren't),
    /// but that is fine for single-threaded simulation loops.
    pub fn compile(&self) -> Box<dyn Fn(f64) -> f64> {
        match self {
            TimeModulation::Constant => Box::new(|_t| 1.0),
            TimeModulation::Sinusoidal { omega, phase } => {
                let omega = *omega;
                let phase = *phase;
                Box::new(move |t| (omega * t + phase).sin())
            }
            TimeModulation::Step { t_on } => {
                let t_on = *t_on;
                Box::new(move |t| if t >= t_on { 1.0 } else { 0.0 })
            }
            TimeModulation::Ramp { t_start, t_end } => {
                let t_start = *t_start;
                let t_end = *t_end;
                Box::new(move |t| {
                    if t <= t_start {
                        0.0
                    } else if t >= t_end {
                        1.0
                    } else {
                        (t - t_start) / (t_end - t_start)
                    }
                })
            }
            TimeModulation::Expression { expr } => {
                match expr.parse::<meval::Expr>() {
                    Ok(parsed) => match parsed.bind("t") {
                        Ok(f) => Box::new(move |t| {
                            let val = f(t);
                            if val.is_finite() { val } else { 0.0 }
                        }),
                        Err(_) => {
                            log::warn!("TimeModulation::compile: failed to bind 't' in expression '{expr}' — force disabled");
                            Box::new(|_t| 0.0)
                        }
                    },
                    Err(_) => {
                        log::warn!("TimeModulation::compile: failed to parse expression '{expr}' — force disabled");
                        Box::new(|_t| 0.0)
                    }
                }
            }
        }
    }
}

// ── Element data structs ─────────────────────────────────────────────────────

/// Uniform gravitational field applied to all bodies with mass > 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravityElement {
    /// Gravity vector in global coordinates (m/s²). Default: (0, -9.81).
    pub g_vector: [f64; 2],
}

impl Default for GravityElement {
    fn default() -> Self {
        Self {
            g_vector: [0.0, -9.81],
        }
    }
}

/// Linear translational spring between two points on two bodies.
///
/// F = -k * (|P_a - P_b| - free_length) along the line of action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearSpringElement {
    /// Body A identifier.
    pub body_a: String,
    /// Attachment point in body A local coordinates.
    pub point_a: [f64; 2],
    /// Body B identifier.
    pub body_b: String,
    /// Attachment point in body B local coordinates.
    pub point_b: [f64; 2],
    /// Spring stiffness (N/m).
    pub stiffness: f64,
    /// Unstretched (free) length (m).
    pub free_length: f64,
}

/// Torsion spring at a revolute joint between two bodies.
///
/// τ = -k * (θ_j - θ_i - θ_free) applied as equal and opposite torques.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TorsionSpringElement {
    /// Body I (reference) identifier.
    pub body_i: String,
    /// Body J (target) identifier.
    pub body_j: String,
    /// Torsional stiffness (N·m/rad).
    pub stiffness: f64,
    /// Free angle (rad) — the relative angle at which torque is zero.
    pub free_angle: f64,
}

/// Linear translational damper between two points on two bodies.
///
/// F = -c * d/dt(|P_a - P_b|) along the line of action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearDamperElement {
    /// Body A identifier.
    pub body_a: String,
    /// Attachment point in body A local coordinates.
    pub point_a: [f64; 2],
    /// Body B identifier.
    pub body_b: String,
    /// Attachment point in body B local coordinates.
    pub point_b: [f64; 2],
    /// Damping coefficient (N·s/m).
    pub damping: f64,
}

/// Rotary damper at a revolute joint between two bodies.
///
/// τ = -c * (θ̇_j - θ̇_i) applied as equal and opposite torques.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotaryDamperElement {
    /// Body I identifier.
    pub body_i: String,
    /// Body J identifier.
    pub body_j: String,
    /// Damping coefficient (N·m·s/rad).
    pub damping: f64,
}

/// External point force applied at a fixed local point on a body.
///
/// Force direction is in global coordinates. Optionally modulated by time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalForceElement {
    /// Body the force is applied to.
    pub body_id: String,
    /// Application point in body-local coordinates.
    pub local_point: [f64; 2],
    /// Force vector in global coordinates (N).
    pub force: [f64; 2],
    /// Time modulation applied to the force vector.
    #[serde(default)]
    pub modulation: TimeModulation,
}

/// External pure torque applied to a body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalTorqueElement {
    /// Body the torque is applied to.
    pub body_id: String,
    /// Torque magnitude (N·m). Positive = counterclockwise.
    pub torque: f64,
    /// Time modulation applied to the torque.
    #[serde(default)]
    pub modulation: TimeModulation,
}

/// Gas spring between two body points.
///
/// Models a gas spring with pressure-based force that increases with
/// compression, plus optional velocity-dependent damping.
///
/// F = F_initial * (stroke / gas_column)^n + c * dL/dt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasSpringElement {
    /// Body A identifier.
    pub body_a: String,
    /// Attachment point in body A local coordinates.
    pub point_a: [f64; 2],
    /// Body B identifier.
    pub body_b: String,
    /// Attachment point in body B local coordinates.
    pub point_b: [f64; 2],
    /// Force at the extended (nominal) length (N).
    pub initial_force: f64,
    /// Nominal extended length (m).
    pub extended_length: f64,
    /// Maximum stroke (compression) (m).
    pub stroke: f64,
    /// Velocity-dependent damping coefficient (N·s/m).
    #[serde(default)]
    pub damping: f64,
    /// Polytropic exponent (1.0=isothermal, 1.4=adiabatic).
    #[serde(default = "default_polytropic_exp")]
    pub polytropic_exp: f64,
}

/// Multi-component bearing friction at a revolute joint.
///
/// τ = -(T_drag + c_vis * |ω| + μ * R * F_n) * tanh(ω / v_thresh)
///
/// Uses tanh regularization for smooth behavior near zero velocity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BearingFrictionElement {
    /// Body I identifier.
    pub body_i: String,
    /// Body J identifier.
    pub body_j: String,
    /// Constant drag torque (N·m).
    pub constant_drag: f64,
    /// Viscous drag coefficient (N·m·s/rad).
    pub viscous_coeff: f64,
    /// Coulomb friction coefficient.
    pub coulomb_coeff: f64,
    /// Effective pin radius for Coulomb term (m).
    pub pin_radius: f64,
    /// Radial load for Coulomb term (N).
    pub radial_load: f64,
    /// Velocity regularization threshold (rad/s).
    #[serde(default = "default_v_threshold")]
    pub v_threshold: f64,
}

/// Penalty-based joint limit at a revolute joint.
///
/// Applies restoring torque when θ_rel = θ_j - θ_i goes outside
/// [angle_min, angle_max]. Includes optional restitution-based damping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLimitElement {
    /// Body I identifier.
    pub body_i: String,
    /// Body J identifier.
    pub body_j: String,
    /// Minimum allowed relative angle (rad).
    pub angle_min: f64,
    /// Maximum allowed relative angle (rad).
    pub angle_max: f64,
    /// Penalty spring stiffness (N·m/rad).
    pub stiffness: f64,
    /// Penalty damping coefficient (N·m·s/rad).
    #[serde(default)]
    pub damping: f64,
    /// Coefficient of restitution (0=perfectly inelastic, 1=perfectly elastic).
    #[serde(default = "default_restitution")]
    pub restitution: f64,
}

/// DC motor with linear torque-speed droop at a revolute joint.
///
/// T = T_stall * (1 - speed_in_dir / ω_no_load) * direction
///
/// Clamped so the motor cannot produce negative torque (overspeed)
/// or exceed stall torque.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorElement {
    /// Body I identifier (typically ground).
    pub body_i: String,
    /// Body J identifier (driven body).
    pub body_j: String,
    /// Maximum torque at zero speed (N·m).
    pub stall_torque: f64,
    /// Speed at zero torque (rad/s).
    pub no_load_speed: f64,
    /// +1.0 for CCW, -1.0 for CW drive direction.
    #[serde(default = "default_direction")]
    pub direction: f64,
}

/// Linear actuator between two body points.
///
/// Applies a constant force along the actuator line with optional
/// speed limiting. Positive force = extension (push apart).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearActuatorElement {
    /// Body A identifier.
    pub body_a: String,
    /// Attachment point in body A local coordinates.
    pub point_a: [f64; 2],
    /// Body B identifier.
    pub body_b: String,
    /// Attachment point in body B local coordinates.
    pub point_b: [f64; 2],
    /// Actuator force (N). Positive = extension/push apart.
    pub force: f64,
    /// Maximum extension rate (m/s). 0 = no limit.
    #[serde(default)]
    pub speed_limit: f64,
}

// ── ForceElement enum ────────────────────────────────────────────────────────

/// A force element attached to one or two bodies in the mechanism.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ForceElement {
    Gravity(GravityElement),
    LinearSpring(LinearSpringElement),
    TorsionSpring(TorsionSpringElement),
    LinearDamper(LinearDamperElement),
    RotaryDamper(RotaryDamperElement),
    ExternalForce(ExternalForceElement),
    ExternalTorque(ExternalTorqueElement),
    GasSpring(GasSpringElement),
    BearingFriction(BearingFrictionElement),
    JointLimit(JointLimitElement),
    Motor(MotorElement),
    LinearActuator(LinearActuatorElement),
}

impl ForceElement {
    /// Create a Coulomb friction element (pure sliding friction at a revolute joint).
    ///
    /// This is a convenience for `BearingFriction` with only the Coulomb component
    /// (constant_drag=0, viscous_coeff=0).
    ///
    /// Torque: τ = -μ * R * F_n * tanh(ω_rel / v_threshold)
    pub fn coulomb_friction(
        body_i: &str,
        body_j: &str,
        friction_coeff: f64,
        pin_radius: f64,
        radial_load: f64,
    ) -> Self {
        ForceElement::BearingFriction(BearingFrictionElement {
            body_i: body_i.to_string(),
            body_j: body_j.to_string(),
            constant_drag: 0.0,
            viscous_coeff: 0.0,
            coulomb_coeff: friction_coeff,
            pin_radius,
            radial_load,
            v_threshold: 0.01,
        })
    }

    /// Evaluate this force element's contribution to the generalized force vector Q.
    ///
    /// Uses the virtual work principle: physical forces are converted to
    /// generalized forces via `point_force_to_q` and `body_torque_to_q`.
    pub fn evaluate(
        &self,
        state: &State,
        bodies: &HashMap<String, Body>,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        _t: f64,
    ) -> DVector<f64> {
        match self {
            ForceElement::Gravity(g) => evaluate_gravity(g, state, bodies, q),
            ForceElement::LinearSpring(s) => evaluate_linear_spring(s, state, q),
            ForceElement::TorsionSpring(s) => evaluate_torsion_spring(s, state, q),
            ForceElement::LinearDamper(d) => evaluate_linear_damper(d, state, q, q_dot),
            ForceElement::RotaryDamper(d) => evaluate_rotary_damper(d, state, q_dot),
            ForceElement::ExternalForce(f) => evaluate_external_force(f, state, q, _t),
            ForceElement::ExternalTorque(t) => evaluate_external_torque(t, state, _t),
            ForceElement::GasSpring(g) => evaluate_gas_spring(g, state, q, q_dot),
            ForceElement::BearingFriction(b) => evaluate_bearing_friction(b, state, q_dot),
            ForceElement::JointLimit(j) => evaluate_joint_limit(j, state, q, q_dot),
            ForceElement::Motor(m) => evaluate_motor(m, state, q_dot),
            ForceElement::LinearActuator(a) => evaluate_linear_actuator(a, state, q, q_dot),
        }
    }

    /// Evaluate this force element using a pre-computed modulation factor.
    ///
    /// For force elements with `TimeModulation` (ExternalForce, ExternalTorque),
    /// the supplied `modulation_factor` replaces the call to `modulation.factor(t)`,
    /// avoiding expression re-parsing in the hot loop. For all other variants the
    /// `modulation_factor` is ignored and evaluation proceeds normally.
    pub fn evaluate_compiled(
        &self,
        state: &State,
        bodies: &HashMap<String, Body>,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        t: f64,
        modulation_factor: f64,
    ) -> DVector<f64> {
        match self {
            ForceElement::ExternalForce(f) => {
                evaluate_external_force_with_factor(f, state, q, modulation_factor)
            }
            ForceElement::ExternalTorque(te) => {
                evaluate_external_torque_with_factor(te, state, modulation_factor)
            }
            // All other variants have no time modulation — delegate unchanged.
            _ => self.evaluate(state, bodies, q, q_dot, t),
        }
    }

    /// Pre-compile this element's time modulation into a closure.
    ///
    /// For ExternalForce and ExternalTorque the modulation is compiled via
    /// [`TimeModulation::compile`]. For all other element types the returned
    /// closure always returns `1.0` (no modulation).
    pub fn compile_modulation(&self) -> Box<dyn Fn(f64) -> f64> {
        match self {
            ForceElement::ExternalForce(f) => f.modulation.compile(),
            ForceElement::ExternalTorque(t) => t.modulation.compile(),
            _ => Box::new(|_t| 1.0),
        }
    }

    /// Human-readable name for this force element type.
    pub fn type_name(&self) -> &'static str {
        match self {
            ForceElement::Gravity(_) => "Gravity",
            ForceElement::LinearSpring(_) => "Linear Spring",
            ForceElement::TorsionSpring(_) => "Torsion Spring",
            ForceElement::LinearDamper(_) => "Linear Damper",
            ForceElement::RotaryDamper(_) => "Rotary Damper",
            ForceElement::ExternalForce(_) => "External Force",
            ForceElement::ExternalTorque(_) => "External Torque",
            ForceElement::GasSpring(_) => "Gas Spring",
            ForceElement::BearingFriction(_) => "Bearing Friction",
            ForceElement::JointLimit(_) => "Joint Limit",
            ForceElement::Motor(_) => "Motor",
            ForceElement::LinearActuator(_) => "Linear Actuator",
        }
    }

    /// Returns body IDs this force element is attached to.
    pub fn attached_body_ids(&self) -> Vec<&str> {
        match self {
            ForceElement::Gravity(_) => vec![], // applies to all bodies
            ForceElement::LinearSpring(s) => vec![&s.body_a, &s.body_b],
            ForceElement::TorsionSpring(s) => vec![&s.body_i, &s.body_j],
            ForceElement::LinearDamper(d) => vec![&d.body_a, &d.body_b],
            ForceElement::RotaryDamper(d) => vec![&d.body_i, &d.body_j],
            ForceElement::ExternalForce(f) => vec![&f.body_id],
            ForceElement::ExternalTorque(t) => vec![&t.body_id],
            ForceElement::GasSpring(g) => vec![&g.body_a, &g.body_b],
            ForceElement::BearingFriction(b) => vec![&b.body_i, &b.body_j],
            ForceElement::JointLimit(j) => vec![&j.body_i, &j.body_j],
            ForceElement::Motor(m) => vec![&m.body_i, &m.body_j],
            ForceElement::LinearActuator(a) => vec![&a.body_a, &a.body_b],
        }
    }
}

// ── Angular element helpers ──────────────────────────────────────────────────

/// Get theta_dot for a body, returning 0.0 for ground and None for unknown bodies.
fn get_body_theta_dot(state: &State, body_id: &str, q_dot: &DVector<f64>) -> Option<f64> {
    if state.is_ground(body_id) {
        return Some(0.0);
    }
    state.get_index(body_id).ok().map(|idx| q_dot[idx.theta_idx()])
}

/// Get (theta, theta_dot) for a body, returning (0.0, 0.0) for ground and None for unknown bodies.
fn get_body_theta_and_dot(
    state: &State,
    body_id: &str,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> Option<(f64, f64)> {
    if state.is_ground(body_id) {
        return Some((0.0, 0.0));
    }
    state
        .get_index(body_id)
        .ok()
        .map(|idx| (q[idx.theta_idx()], q_dot[idx.theta_idx()]))
}

// ── Evaluation functions ─────────────────────────────────────────────────────

fn evaluate_gravity(
    g: &GravityElement,
    state: &State,
    bodies: &HashMap<String, Body>,
    q: &DVector<f64>,
) -> DVector<f64> {
    let g_vec = Vector2::new(g.g_vector[0], g.g_vector[1]);
    let mut total = DVector::zeros(state.n_coords());

    for (body_id, body) in bodies {
        if body_id == GROUND_ID || body.mass <= 0.0 {
            continue;
        }
        let force_global = g_vec * body.mass;
        total += point_force_to_q(state, body_id, &body.cg_local, &force_global, q);
    }

    total
}

fn evaluate_linear_spring(
    s: &LinearSpringElement,
    state: &State,
    q: &DVector<f64>,
) -> DVector<f64> {
    let pt_a_local = Vector2::new(s.point_a[0], s.point_a[1]);
    let pt_b_local = Vector2::new(s.point_b[0], s.point_b[1]);

    let pt_a_global = state.body_point_global(&s.body_a, &pt_a_local, q);
    let pt_b_global = state.body_point_global(&s.body_b, &pt_b_local, q);

    let delta = pt_b_global - pt_a_global;
    let length = delta.norm();

    if length < 1e-15 {
        return DVector::zeros(state.n_coords());
    }

    let unit = delta / length;
    let extension = length - s.free_length;
    let force_magnitude = s.stiffness * extension;

    // Force on body A (toward B when extended)
    let force_on_a = unit * force_magnitude;
    // Force on body B (toward A when extended — Newton's third law)
    let force_on_b = -force_on_a;

    let mut total = DVector::zeros(state.n_coords());
    total += point_force_to_q(state, &s.body_a, &pt_a_local, &force_on_a, q);
    total += point_force_to_q(state, &s.body_b, &pt_b_local, &force_on_b, q);
    total
}

fn evaluate_torsion_spring(
    s: &TorsionSpringElement,
    state: &State,
    q: &DVector<f64>,
) -> DVector<f64> {
    let theta_i = state.get_angle(&s.body_i, q);
    let theta_j = state.get_angle(&s.body_j, q);

    let relative_angle = theta_j - theta_i;
    let torque = -s.stiffness * (relative_angle - s.free_angle);

    // Torque on body J, reaction on body I (Newton's third law)
    let mut total = DVector::zeros(state.n_coords());
    total += body_torque_to_q(state, &s.body_j, torque);
    total += body_torque_to_q(state, &s.body_i, -torque);
    total
}

fn evaluate_linear_damper(
    d: &LinearDamperElement,
    state: &State,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let pt_a_local = Vector2::new(d.point_a[0], d.point_a[1]);
    let pt_b_local = Vector2::new(d.point_b[0], d.point_b[1]);

    let pt_a_global = state.body_point_global(&d.body_a, &pt_a_local, q);
    let pt_b_global = state.body_point_global(&d.body_b, &pt_b_local, q);

    let delta = pt_b_global - pt_a_global;
    let length = delta.norm();

    if length < 1e-15 {
        return DVector::zeros(state.n_coords());
    }

    let unit = delta / length;

    // Compute rate of change of length: d/dt(|P_b - P_a|) = unit · (v_b - v_a)
    let v_a = state.body_point_velocity(&d.body_a, &pt_a_local, q, q_dot);
    let v_b = state.body_point_velocity(&d.body_b, &pt_b_local, q, q_dot);
    let length_rate = unit.dot(&(v_b - v_a));

    let force_magnitude = -d.damping * length_rate;

    // Force on body A (along unit direction)
    let force_on_a = unit * force_magnitude;
    let force_on_b = -force_on_a;

    let mut total = DVector::zeros(state.n_coords());
    total += point_force_to_q(state, &d.body_a, &pt_a_local, &force_on_a, q);
    total += point_force_to_q(state, &d.body_b, &pt_b_local, &force_on_b, q);
    total
}

fn evaluate_rotary_damper(
    d: &RotaryDamperElement,
    state: &State,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let Some(theta_dot_i) = get_body_theta_dot(state, &d.body_i, q_dot) else {
        return DVector::zeros(q_dot.len());
    };
    let Some(theta_dot_j) = get_body_theta_dot(state, &d.body_j, q_dot) else {
        return DVector::zeros(q_dot.len());
    };

    let relative_rate = theta_dot_j - theta_dot_i;
    let torque = -d.damping * relative_rate;

    let mut total = DVector::zeros(state.n_coords());
    total += body_torque_to_q(state, &d.body_j, torque);
    total += body_torque_to_q(state, &d.body_i, -torque);
    total
}

fn evaluate_external_force(
    f: &ExternalForceElement,
    state: &State,
    q: &DVector<f64>,
    t: f64,
) -> DVector<f64> {
    let local_pt = Vector2::new(f.local_point[0], f.local_point[1]);
    let factor = f.modulation.factor(t);
    let force = Vector2::new(f.force[0] * factor, f.force[1] * factor);
    point_force_to_q(state, &f.body_id, &local_pt, &force, q)
}

fn evaluate_external_torque(te: &ExternalTorqueElement, state: &State, t: f64) -> DVector<f64> {
    let factor = te.modulation.factor(t);
    body_torque_to_q(state, &te.body_id, te.torque * factor)
}

/// Evaluate external force with a pre-computed modulation factor.
fn evaluate_external_force_with_factor(
    f: &ExternalForceElement,
    state: &State,
    q: &DVector<f64>,
    factor: f64,
) -> DVector<f64> {
    let local_pt = Vector2::new(f.local_point[0], f.local_point[1]);
    let force = Vector2::new(f.force[0] * factor, f.force[1] * factor);
    point_force_to_q(state, &f.body_id, &local_pt, &force, q)
}

/// Evaluate external torque with a pre-computed modulation factor.
fn evaluate_external_torque_with_factor(
    te: &ExternalTorqueElement,
    state: &State,
    factor: f64,
) -> DVector<f64> {
    body_torque_to_q(state, &te.body_id, te.torque * factor)
}

fn evaluate_gas_spring(
    g: &GasSpringElement,
    state: &State,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let pt_a_local = Vector2::new(g.point_a[0], g.point_a[1]);
    let pt_b_local = Vector2::new(g.point_b[0], g.point_b[1]);

    let pt_a_global = state.body_point_global(&g.body_a, &pt_a_local, q);
    let pt_b_global = state.body_point_global(&g.body_b, &pt_b_local, q);

    let delta = pt_b_global - pt_a_global;
    let current_length = delta.norm();

    if current_length < 1e-15 {
        return DVector::zeros(state.n_coords());
    }

    let unit = delta / current_length;

    // Degenerate gas spring with zero stroke: acts as constant-force element
    if g.stroke <= 0.0 {
        let force_on_b = unit * g.initial_force;
        let force_on_a = -force_on_b;
        let mut total = DVector::zeros(state.n_coords());
        total += point_force_to_q(state, &g.body_a, &pt_a_local, &force_on_a, q);
        total += point_force_to_q(state, &g.body_b, &pt_b_local, &force_on_b, q);
        return total;
    }

    // Compression from extended position, clamped to [0, stroke]
    let compression = (g.extended_length - current_length).clamp(0.0, g.stroke);

    // Gas force: F = F0 * (stroke / gas_column)^n
    let gas_column = (g.stroke - compression).max(1e-10);
    let force_ratio = (g.stroke / gas_column).powf(g.polytropic_exp);
    let gas_force = g.initial_force * force_ratio;

    // Velocity-dependent damping along line of action
    let damping_force = if g.damping.abs() > 0.0 {
        let v_a = state.body_point_velocity(&g.body_a, &pt_a_local, q, q_dot);
        let v_b = state.body_point_velocity(&g.body_b, &pt_b_local, q, q_dot);
        let v_along = unit.dot(&(v_b - v_a));
        -g.damping * v_along
    } else {
        0.0
    };

    let total_force = gas_force + damping_force;

    // Gas spring pushes apart (positive = extension)
    let force_on_b = unit * total_force;
    let force_on_a = -force_on_b;

    let mut total = DVector::zeros(state.n_coords());
    total += point_force_to_q(state, &g.body_a, &pt_a_local, &force_on_a, q);
    total += point_force_to_q(state, &g.body_b, &pt_b_local, &force_on_b, q);
    total
}

fn evaluate_bearing_friction(
    b: &BearingFrictionElement,
    state: &State,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let Some(omega_i) = get_body_theta_dot(state, &b.body_i, q_dot) else {
        return DVector::zeros(q_dot.len());
    };
    let Some(omega_j) = get_body_theta_dot(state, &b.body_j, q_dot) else {
        return DVector::zeros(q_dot.len());
    };

    let omega_rel = omega_j - omega_i;

    // Direction via tanh regularization
    let direction = (omega_rel / b.v_threshold).tanh();

    // Total friction magnitude
    let magnitude = b.constant_drag
        + b.viscous_coeff * omega_rel.abs()
        + b.coulomb_coeff * b.pin_radius * b.radial_load;

    let torque = -magnitude * direction;

    let mut total = DVector::zeros(state.n_coords());
    total += body_torque_to_q(state, &b.body_j, torque);
    total += body_torque_to_q(state, &b.body_i, -torque);
    total
}

fn evaluate_joint_limit(
    j: &JointLimitElement,
    state: &State,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    // Get relative angle and angular velocity
    let Some((theta_i, omega_i)) = get_body_theta_and_dot(state, &j.body_i, q, q_dot) else {
        return DVector::zeros(q_dot.len());
    };
    let Some((theta_j, omega_j)) = get_body_theta_and_dot(state, &j.body_j, q, q_dot) else {
        return DVector::zeros(q_dot.len());
    };

    let theta_rel = theta_j - theta_i;
    let omega_rel = omega_j - omega_i;

    let torque = if theta_rel < j.angle_min {
        // Below minimum -- push CCW (positive torque on j)
        let penetration = j.angle_min - theta_rel;
        // Full damping when moving into the stop, reduced by restitution when bouncing away
        let damp_factor = if omega_rel < 0.0 {
            j.damping
        } else {
            j.damping * j.restitution
        };
        j.stiffness * penetration - damp_factor * omega_rel
    } else if theta_rel > j.angle_max {
        // Above maximum -- push CW (negative torque on j)
        let penetration = theta_rel - j.angle_max;
        // Full damping when moving into the stop, reduced by restitution when bouncing away
        let damp_factor = if omega_rel > 0.0 {
            j.damping
        } else {
            j.damping * j.restitution
        };
        -(j.stiffness * penetration + damp_factor * omega_rel)
    } else {
        return DVector::zeros(state.n_coords());
    };

    let mut total = DVector::zeros(state.n_coords());
    total += body_torque_to_q(state, &j.body_j, torque);
    total += body_torque_to_q(state, &j.body_i, -torque);
    total
}

fn evaluate_motor(
    m: &MotorElement,
    state: &State,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    if m.no_load_speed <= 0.0 {
        return DVector::zeros(state.n_coords());
    }

    let Some(omega_i) = get_body_theta_dot(state, &m.body_i, q_dot) else {
        return DVector::zeros(q_dot.len());
    };
    let Some(omega_j) = get_body_theta_dot(state, &m.body_j, q_dot) else {
        return DVector::zeros(q_dot.len());
    };

    let omega_rel = omega_j - omega_i;
    let speed_in_dir = omega_rel * m.direction;

    // Linear droop: T = T_stall * (1 - speed / omega_no_load)
    let torque_fraction = (1.0 - speed_in_dir / m.no_load_speed).clamp(0.0, 1.0);
    let torque = m.stall_torque * torque_fraction * m.direction;

    let mut total = DVector::zeros(state.n_coords());
    total += body_torque_to_q(state, &m.body_j, torque);
    total += body_torque_to_q(state, &m.body_i, -torque);
    total
}

fn evaluate_linear_actuator(
    a: &LinearActuatorElement,
    state: &State,
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
) -> DVector<f64> {
    let pt_a_local = Vector2::new(a.point_a[0], a.point_a[1]);
    let pt_b_local = Vector2::new(a.point_b[0], a.point_b[1]);

    let pt_a_global = state.body_point_global(&a.body_a, &pt_a_local, q);
    let pt_b_global = state.body_point_global(&a.body_b, &pt_b_local, q);

    let delta = pt_b_global - pt_a_global;
    let length = delta.norm();

    if length < 1e-15 {
        return DVector::zeros(state.n_coords());
    }

    let unit = delta / length;

    // Speed limiting: ramp force to zero as speed approaches limit
    let actual_force = if a.speed_limit > 0.0 {
        let v_a = state.body_point_velocity(&a.body_a, &pt_a_local, q, q_dot);
        let v_b = state.body_point_velocity(&a.body_b, &pt_b_local, q, q_dot);
        let v_along = unit.dot(&(v_b - v_a));
        let speed_ratio = v_along.abs() / a.speed_limit;
        if speed_ratio >= 1.0 {
            0.0
        } else {
            a.force * (1.0 - speed_ratio)
        }
    } else {
        a.force
    };

    // Positive force = push apart (extension)
    let force_on_b = unit * actual_force;
    let force_on_a = -force_on_b;

    let mut total = DVector::zeros(state.n_coords());
    total += point_force_to_q(state, &a.body_a, &pt_a_local, &force_on_a, q);
    total += point_force_to_q(state, &a.body_b, &pt_b_local, &force_on_b, q);
    total
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    fn setup_single_bar() -> (State, HashMap<String, Body>) {
        let mut state = State::new();
        state.register_body("bar").unwrap();
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 2.0, 0.1);
        let mut bodies = HashMap::new();
        bodies.insert("ground".to_string(), ground);
        bodies.insert("bar".to_string(), bar);
        (state, bodies)
    }

    fn setup_two_bars() -> (State, HashMap<String, Body>) {
        let mut state = State::new();
        state.register_body("bar1").unwrap();
        state.register_body("bar2").unwrap();
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar1 = make_bar("bar1", "A", "B", 1.0, 2.0, 0.1);
        let bar2 = make_bar("bar2", "C", "D", 1.0, 2.0, 0.1);
        let mut bodies = HashMap::new();
        bodies.insert("ground".to_string(), ground);
        bodies.insert("bar1".to_string(), bar1);
        bodies.insert("bar2".to_string(), bar2);
        (state, bodies)
    }

    #[test]
    fn gravity_element_matches_old_gravity() {
        let (state, bodies) = setup_single_bar();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let elem = ForceElement::Gravity(GravityElement::default());
        let result = elem.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // F = mg = 2.0 * 9.81 = 19.62 downward
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result[1], -19.62, epsilon = 1e-10);
        // Torque from gravity at CG=(0.5, 0) with θ=0: B(0)·(0.5,0) = (0,0.5), dot (0,-19.62) = -9.81
        assert_abs_diff_eq!(result[2], -9.81, epsilon = 1e-10);
    }

    #[test]
    fn linear_spring_zero_at_free_length() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        // Place bars 1.0 apart (free length = 1.0 → no force)
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let spring = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "bar1".into(),
            point_a: [1.0, 0.0], // tip of bar1
            body_b: "bar2".into(),
            point_b: [0.0, 0.0], // base of bar2
            stiffness: 500.0,
            free_length: 1.0,
        });

        let result = spring.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // At free length → zero force
        for i in 0..result.len() {
            assert_abs_diff_eq!(result[i], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn linear_spring_produces_restoring_force() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        // bar1 origin at (0,0), bar2 origin at (2,0)
        // Attach spring at body origins → global distance = 2.0
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 2.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let spring = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0], // body origin
            body_b: "bar2".into(),
            point_b: [0.0, 0.0], // body origin
            stiffness: 100.0,
            free_length: 1.0,
        });

        let result = spring.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Extension = 2.0 - 1.0 = 1.0, stiffness = 100
        // Force on bar1 = +100 N in x (toward bar2)
        // Force on bar2 = -100 N in x (toward bar1)
        assert_abs_diff_eq!(result[0], 100.0, epsilon = 1e-10); // bar1 Fx
        assert_abs_diff_eq!(result[3], -100.0, epsilon = 1e-10); // bar2 Fx
    }

    #[test]
    fn torsion_spring_produces_restoring_torque() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.0, 0.0, PI / 4.0);
        let q_dot = DVector::zeros(state.n_coords());

        let spring = ForceElement::TorsionSpring(TorsionSpringElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            stiffness: 10.0,
            free_angle: 0.0,
        });

        let result = spring.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Relative angle = π/4 - 0 = π/4
        // Torque on bar2 = -10 * (π/4) (restoring)
        // Torque on bar1 = +10 * (π/4) (reaction)
        let expected = -10.0 * PI / 4.0;
        assert_abs_diff_eq!(result[5], expected, epsilon = 1e-10); // bar2 θ
        assert_abs_diff_eq!(result[2], -expected, epsilon = 1e-10); // bar1 θ
    }

    #[test]
    fn external_force_produces_correct_q() {
        let (state, bodies) = setup_single_bar();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let ext = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.5, 0.0], // CG
            force: [10.0, -5.0],
            modulation: TimeModulation::Constant,
        });

        let result = ext.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        assert_abs_diff_eq!(result[0], 10.0, epsilon = 1e-14);
        assert_abs_diff_eq!(result[1], -5.0, epsilon = 1e-14);
        // B(0) · (0.5, 0) = (0, 0.5), dot (10, -5) = -2.5
        assert_abs_diff_eq!(result[2], -2.5, epsilon = 1e-14);
    }

    #[test]
    fn external_torque_produces_correct_q() {
        let (state, bodies) = setup_single_bar();
        let q = state.make_q();
        let q_dot = DVector::zeros(state.n_coords());

        let ext = ForceElement::ExternalTorque(ExternalTorqueElement {
            body_id: "bar".into(),
            torque: 7.5,
            modulation: TimeModulation::Constant,
        });

        let result = ext.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(result[2], 7.5, epsilon = 1e-15);
    }

    #[test]
    fn rotary_damper_resists_relative_motion() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        // bar1 angular velocity = 0, bar2 angular velocity = 2 rad/s
        q_dot[5] = 2.0; // bar2 theta_dot

        let damper = ForceElement::RotaryDamper(RotaryDamperElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            damping: 5.0,
        });

        let result = damper.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Relative rate = 2.0 - 0.0 = 2.0
        // Torque on bar2 = -5 * 2 = -10 (opposes motion)
        // Torque on bar1 = +10 (reaction)
        assert_abs_diff_eq!(result[5], -10.0, epsilon = 1e-14); // bar2 θ
        assert_abs_diff_eq!(result[2], 10.0, epsilon = 1e-14); // bar1 θ
    }

    #[test]
    fn type_name_returns_correct_strings() {
        assert_eq!(
            ForceElement::Gravity(GravityElement::default()).type_name(),
            "Gravity"
        );
        assert_eq!(
            ForceElement::LinearSpring(LinearSpringElement {
                body_a: "a".into(),
                point_a: [0.0, 0.0],
                body_b: "b".into(),
                point_b: [0.0, 0.0],
                stiffness: 1.0,
                free_length: 1.0,
            })
            .type_name(),
            "Linear Spring"
        );
    }

    #[test]
    fn serde_roundtrip_gravity() {
        let elem = ForceElement::Gravity(GravityElement::default());
        let json = serde_json::to_string(&elem).unwrap();
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        assert_eq!(back.type_name(), "Gravity");
    }

    #[test]
    fn serde_roundtrip_linear_spring() {
        let elem = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "crank".into(),
            point_a: [0.1, 0.0],
            body_b: "rocker".into(),
            point_b: [-0.1, 0.0],
            stiffness: 500.0,
            free_length: 0.2,
        });
        let json = serde_json::to_string(&elem).unwrap();
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::LinearSpring(s) => {
                assert_abs_diff_eq!(s.stiffness, 500.0, epsilon = 1e-15);
                assert_eq!(s.body_a, "crank");
            }
            _ => panic!("Expected LinearSpring"),
        }
    }

    #[test]
    fn serde_tagged_format() {
        let elem = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.5, 0.0],
            force: [10.0, -5.0],
            modulation: TimeModulation::Constant,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"ExternalForce\""));
    }

    // ── Gas Spring tests ─────────────────────────────────────────────────────

    #[test]
    fn gas_spring_force_at_extended_length() {
        // At extended length, compression=0 → gas_column=stroke → force=initial_force
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        // Place bodies so attachment distance = extended_length
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.5, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let gs = ForceElement::GasSpring(GasSpringElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            initial_force: 200.0,
            extended_length: 0.5,
            stroke: 0.2,
            damping: 0.0,
            polytropic_exp: 1.0,
        });

        let result = gs.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // At extended length: compression=0, gas_column=stroke, ratio=1.0
        // force = 200.0 * 1.0 = 200.0, pushes apart along +x
        // bar1 gets pushed in -x, bar2 in +x
        assert_abs_diff_eq!(result[0], -200.0, epsilon = 1e-10); // bar1 Fx
        assert_abs_diff_eq!(result[3], 200.0, epsilon = 1e-10); // bar2 Fx
    }

    #[test]
    fn gas_spring_force_increases_with_compression() {
        // When compressed, gas column shrinks → force increases
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        // extended_length=0.5, stroke=0.2. Place bodies 0.4 apart → compression=0.1
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.4, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let gs = ForceElement::GasSpring(GasSpringElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            initial_force: 200.0,
            extended_length: 0.5,
            stroke: 0.2,
            damping: 0.0,
            polytropic_exp: 1.0,
        });

        let result = gs.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // compression = 0.5 - 0.4 = 0.1
        // gas_column = 0.2 - 0.1 = 0.1
        // ratio = (0.2 / 0.1)^1.0 = 2.0
        // force = 200.0 * 2.0 = 400.0 (pushes apart)
        assert_abs_diff_eq!(result[0], -400.0, epsilon = 1e-10); // bar1 Fx
        assert_abs_diff_eq!(result[3], 400.0, epsilon = 1e-10); // bar2 Fx
    }

    #[test]
    fn gas_spring_compression_clamped_beyond_stroke() {
        // If spring compressed beyond stroke, compression clamps to stroke
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        // extended_length=0.5, stroke=0.2. Place bodies 0.1 apart → compression would be 0.4 but clamps to 0.2
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.1, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let gs = ForceElement::GasSpring(GasSpringElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            initial_force: 200.0,
            extended_length: 0.5,
            stroke: 0.2,
            damping: 0.0,
            polytropic_exp: 1.0,
        });

        let result = gs.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // compression clamped to 0.2 (=stroke), gas_column = max(0.2-0.2, 1e-10) = 1e-10
        // ratio = (0.2 / 1e-10)^1.0 = very large
        // This tests the singularity protection (gas_column floored at 1e-10)
        assert!(result[3] > 200.0); // force should be much larger than initial
    }

    // ── Bearing Friction tests ───────────────────────────────────────────────

    #[test]
    fn bearing_friction_opposes_relative_rotation() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        // bar2 spinning at 5 rad/s, bar1 stationary
        q_dot[5] = 5.0;

        let bf = ForceElement::BearingFriction(BearingFrictionElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            constant_drag: 2.0,
            viscous_coeff: 0.5,
            coulomb_coeff: 0.0,
            pin_radius: 0.0,
            radial_load: 0.0,
            v_threshold: 0.01,
        });

        let result = bf.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // omega_rel = 5.0, tanh(5.0/0.01) ~ 1.0
        // magnitude = 2.0 + 0.5 * 5.0 = 4.5
        // torque on bar2 = -4.5 * 1.0 = -4.5 (opposes positive rotation)
        assert_abs_diff_eq!(result[5], -4.5, epsilon = 1e-3); // bar2 θ
        assert_abs_diff_eq!(result[2], 4.5, epsilon = 1e-3); // bar1 θ (reaction)
    }

    #[test]
    fn bearing_friction_smooth_near_zero_speed() {
        // Near zero speed, tanh regularization produces small, smooth torque
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        q_dot[5] = 0.001; // very slow

        let bf = ForceElement::BearingFriction(BearingFrictionElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            constant_drag: 2.0,
            viscous_coeff: 0.0,
            coulomb_coeff: 0.0,
            pin_radius: 0.0,
            radial_load: 0.0,
            v_threshold: 0.01,
        });

        let result = bf.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // omega_rel = 0.001, direction = tanh(0.001/0.01) = tanh(0.1) ~ 0.0997
        // torque = -2.0 * 0.0997 ~ -0.1993
        // Torque magnitude should be much less than constant_drag due to regularization
        assert!(result[5].abs() < 2.0);
        assert!(result[5] < 0.0); // still opposes positive motion
    }

    #[test]
    fn bearing_friction_with_coulomb_component() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        q_dot[5] = 10.0; // fast enough that tanh ~ 1.0

        let bf = ForceElement::BearingFriction(BearingFrictionElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            constant_drag: 1.0,
            viscous_coeff: 0.0,
            coulomb_coeff: 0.3,
            pin_radius: 0.01,
            radial_load: 1000.0,
            v_threshold: 0.01,
        });

        let result = bf.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // magnitude = 1.0 + 0.0 + 0.3 * 0.01 * 1000.0 = 1.0 + 3.0 = 4.0
        // torque on bar2 = -4.0 (opposes motion)
        assert_abs_diff_eq!(result[5], -4.0, epsilon = 1e-3);
    }

    // ── Joint Limit tests ────────────────────────────────────────────────────

    #[test]
    fn joint_limit_no_torque_within_range() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 0.0, 0.0, PI / 4.0);
        let q_dot = DVector::zeros(state.n_coords());

        let jl = ForceElement::JointLimit(JointLimitElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            angle_min: -PI / 2.0,
            angle_max: PI / 2.0,
            stiffness: 1000.0,
            damping: 0.0,
            restitution: 0.5,
        });

        let result = jl.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // theta_rel = π/4 is within [-π/2, π/2] → no torque
        for i in 0..result.len() {
            assert_abs_diff_eq!(result[i], 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn joint_limit_restoring_torque_above_max() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        // theta_rel = 2.0 > angle_max = 1.5
        state.set_pose("bar2", &mut q, 0.0, 0.0, 2.0);
        let q_dot = DVector::zeros(state.n_coords());

        let jl = ForceElement::JointLimit(JointLimitElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            angle_min: -1.5,
            angle_max: 1.5,
            stiffness: 1000.0,
            damping: 0.0,
            restitution: 0.5,
        });

        let result = jl.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // penetration = 2.0 - 1.5 = 0.5
        // torque on bar2 = -(1000 * 0.5) = -500 (pushes back CW)
        assert_abs_diff_eq!(result[5], -500.0, epsilon = 1e-10); // bar2 θ
        assert_abs_diff_eq!(result[2], 500.0, epsilon = 1e-10); // bar1 θ (reaction)
    }

    #[test]
    fn joint_limit_restoring_torque_below_min() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        // theta_rel = -2.0 < angle_min = -1.5
        state.set_pose("bar2", &mut q, 0.0, 0.0, -2.0);
        let q_dot = DVector::zeros(state.n_coords());

        let jl = ForceElement::JointLimit(JointLimitElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            angle_min: -1.5,
            angle_max: 1.5,
            stiffness: 1000.0,
            damping: 0.0,
            restitution: 0.5,
        });

        let result = jl.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // penetration = -1.5 - (-2.0) = 0.5
        // torque on bar2 = +1000 * 0.5 = +500 (pushes back CCW)
        assert_abs_diff_eq!(result[5], 500.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], -500.0, epsilon = 1e-10);
    }

    // ── Motor tests ──────────────────────────────────────────────────────────

    #[test]
    fn motor_stall_torque_at_zero_speed() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let q_dot = DVector::zeros(state.n_coords());

        let motor = ForceElement::Motor(MotorElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            stall_torque: 10.0,
            no_load_speed: 100.0,
            direction: 1.0,
        });

        let result = motor.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // At zero speed, torque_fraction = 1.0
        // torque = 10.0 * 1.0 * 1.0 = 10.0
        assert_abs_diff_eq!(result[5], 10.0, epsilon = 1e-14); // bar2 θ
        assert_abs_diff_eq!(result[2], -10.0, epsilon = 1e-14); // bar1 θ (reaction)
    }

    #[test]
    fn motor_zero_torque_at_no_load_speed() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        // bar2 at no-load speed
        q_dot[5] = 100.0;

        let motor = ForceElement::Motor(MotorElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            stall_torque: 10.0,
            no_load_speed: 100.0,
            direction: 1.0,
        });

        let result = motor.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // At no-load speed, torque_fraction = 0.0
        assert_abs_diff_eq!(result[5], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn motor_zero_when_no_load_speed_invalid() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let q_dot = DVector::zeros(state.n_coords());

        let motor = ForceElement::Motor(MotorElement {
            body_i: "bar1".into(),
            body_j: "bar2".into(),
            stall_torque: 10.0,
            no_load_speed: 0.0, // invalid
            direction: 1.0,
        });

        let result = motor.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        for i in 0..result.len() {
            assert_abs_diff_eq!(result[i], 0.0, epsilon = 1e-15);
        }
    }

    // ── Linear Actuator tests ────────────────────────────────────────────────

    #[test]
    fn linear_actuator_constant_force_no_speed_limit() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            force: 50.0,
            speed_limit: 0.0,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // Positive force = push apart along +x
        assert_abs_diff_eq!(result[0], -50.0, epsilon = 1e-10); // bar1 Fx (pushed left)
        assert_abs_diff_eq!(result[3], 50.0, epsilon = 1e-10); // bar2 Fx (pushed right)
    }

    #[test]
    fn linear_actuator_force_reduced_at_speed_limit() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.0, 0.0, 0.0);
        let mut q_dot = DVector::zeros(state.n_coords());
        // bar2 moving at speed_limit → force should be zero
        q_dot[3] = 2.0; // bar2 vx = speed_limit

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            force: 50.0,
            speed_limit: 2.0,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // v_along = 2.0, speed_ratio = 2.0/2.0 = 1.0 → force = 0
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[3], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn linear_actuator_force_halved_at_half_speed_limit() {
        let (state, bodies) = setup_two_bars();
        let mut q = state.make_q();
        state.set_pose("bar1", &mut q, 0.0, 0.0, 0.0);
        state.set_pose("bar2", &mut q, 1.0, 0.0, 0.0);
        let mut q_dot = DVector::zeros(state.n_coords());
        q_dot[3] = 1.0; // bar2 vx = half of speed_limit

        let act = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "bar1".into(),
            point_a: [0.0, 0.0],
            body_b: "bar2".into(),
            point_b: [0.0, 0.0],
            force: 50.0,
            speed_limit: 2.0,
        });

        let result = act.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // v_along = 1.0, speed_ratio = 0.5, force = 50 * 0.5 = 25.0
        assert_abs_diff_eq!(result[0], -25.0, epsilon = 1e-10); // bar1 pushed left
        assert_abs_diff_eq!(result[3], 25.0, epsilon = 1e-10); // bar2 pushed right
    }

    // ── Serde roundtrip tests for new elements ───────────────────────────────

    #[test]
    fn serde_roundtrip_gas_spring() {
        let elem = ForceElement::GasSpring(GasSpringElement {
            body_a: "link1".into(),
            point_a: [0.1, 0.0],
            body_b: "link2".into(),
            point_b: [-0.1, 0.0],
            initial_force: 200.0,
            extended_length: 0.5,
            stroke: 0.2,
            damping: 10.0,
            polytropic_exp: 1.3,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"GasSpring\""));
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::GasSpring(g) => {
                assert_abs_diff_eq!(g.initial_force, 200.0, epsilon = 1e-15);
                assert_abs_diff_eq!(g.stroke, 0.2, epsilon = 1e-15);
                assert_abs_diff_eq!(g.polytropic_exp, 1.3, epsilon = 1e-15);
                assert_eq!(g.body_a, "link1");
            }
            _ => panic!("Expected GasSpring"),
        }
    }

    #[test]
    fn serde_roundtrip_gas_spring_defaults() {
        // Test that default fields deserialize correctly when omitted
        let json = r#"{"type":"GasSpring","body_a":"a","point_a":[0,0],"body_b":"b","point_b":[0,0],"initial_force":100,"extended_length":0.5,"stroke":0.2}"#;
        let elem: ForceElement = serde_json::from_str(json).unwrap();
        match elem {
            ForceElement::GasSpring(g) => {
                assert_abs_diff_eq!(g.damping, 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(g.polytropic_exp, 1.0, epsilon = 1e-15);
            }
            _ => panic!("Expected GasSpring"),
        }
    }

    #[test]
    fn serde_roundtrip_bearing_friction() {
        let elem = ForceElement::BearingFriction(BearingFrictionElement {
            body_i: "arm".into(),
            body_j: "ground".into(),
            constant_drag: 1.5,
            viscous_coeff: 0.1,
            coulomb_coeff: 0.3,
            pin_radius: 0.01,
            radial_load: 500.0,
            v_threshold: 0.02,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"BearingFriction\""));
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::BearingFriction(b) => {
                assert_abs_diff_eq!(b.constant_drag, 1.5, epsilon = 1e-15);
                assert_abs_diff_eq!(b.v_threshold, 0.02, epsilon = 1e-15);
                assert_eq!(b.body_i, "arm");
            }
            _ => panic!("Expected BearingFriction"),
        }
    }

    #[test]
    fn serde_roundtrip_bearing_friction_defaults() {
        let json = r#"{"type":"BearingFriction","body_i":"a","body_j":"b","constant_drag":1,"viscous_coeff":0,"coulomb_coeff":0,"pin_radius":0,"radial_load":0}"#;
        let elem: ForceElement = serde_json::from_str(json).unwrap();
        match elem {
            ForceElement::BearingFriction(b) => {
                assert_abs_diff_eq!(b.v_threshold, 0.01, epsilon = 1e-15);
            }
            _ => panic!("Expected BearingFriction"),
        }
    }

    #[test]
    fn serde_roundtrip_joint_limit() {
        let elem = ForceElement::JointLimit(JointLimitElement {
            body_i: "crank".into(),
            body_j: "rocker".into(),
            angle_min: -1.0,
            angle_max: 1.0,
            stiffness: 5000.0,
            damping: 50.0,
            restitution: 0.3,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"JointLimit\""));
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::JointLimit(j) => {
                assert_abs_diff_eq!(j.stiffness, 5000.0, epsilon = 1e-15);
                assert_abs_diff_eq!(j.restitution, 0.3, epsilon = 1e-15);
                assert_eq!(j.body_i, "crank");
            }
            _ => panic!("Expected JointLimit"),
        }
    }

    #[test]
    fn serde_roundtrip_joint_limit_defaults() {
        let json = r#"{"type":"JointLimit","body_i":"a","body_j":"b","angle_min":-1,"angle_max":1,"stiffness":1000}"#;
        let elem: ForceElement = serde_json::from_str(json).unwrap();
        match elem {
            ForceElement::JointLimit(j) => {
                assert_abs_diff_eq!(j.damping, 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(j.restitution, 0.5, epsilon = 1e-15);
            }
            _ => panic!("Expected JointLimit"),
        }
    }

    #[test]
    fn serde_roundtrip_motor() {
        let elem = ForceElement::Motor(MotorElement {
            body_i: "ground".into(),
            body_j: "wheel".into(),
            stall_torque: 50.0,
            no_load_speed: 300.0,
            direction: -1.0,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"Motor\""));
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::Motor(m) => {
                assert_abs_diff_eq!(m.stall_torque, 50.0, epsilon = 1e-15);
                assert_abs_diff_eq!(m.direction, -1.0, epsilon = 1e-15);
                assert_eq!(m.body_j, "wheel");
            }
            _ => panic!("Expected Motor"),
        }
    }

    #[test]
    fn serde_roundtrip_motor_defaults() {
        let json = r#"{"type":"Motor","body_i":"a","body_j":"b","stall_torque":10,"no_load_speed":100}"#;
        let elem: ForceElement = serde_json::from_str(json).unwrap();
        match elem {
            ForceElement::Motor(m) => {
                assert_abs_diff_eq!(m.direction, 1.0, epsilon = 1e-15);
            }
            _ => panic!("Expected Motor"),
        }
    }

    #[test]
    fn serde_roundtrip_linear_actuator() {
        let elem = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: "piston".into(),
            point_a: [0.0, 0.0],
            body_b: "cylinder".into(),
            point_b: [0.5, 0.0],
            force: 1000.0,
            speed_limit: 0.5,
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"LinearActuator\""));
        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::LinearActuator(a) => {
                assert_abs_diff_eq!(a.force, 1000.0, epsilon = 1e-15);
                assert_abs_diff_eq!(a.speed_limit, 0.5, epsilon = 1e-15);
                assert_eq!(a.body_a, "piston");
            }
            _ => panic!("Expected LinearActuator"),
        }
    }

    #[test]
    fn serde_roundtrip_linear_actuator_defaults() {
        let json = r#"{"type":"LinearActuator","body_a":"a","point_a":[0,0],"body_b":"b","point_b":[0,0],"force":100}"#;
        let elem: ForceElement = serde_json::from_str(json).unwrap();
        match elem {
            ForceElement::LinearActuator(a) => {
                assert_abs_diff_eq!(a.speed_limit, 0.0, epsilon = 1e-15);
            }
            _ => panic!("Expected LinearActuator"),
        }
    }

    // ── Type name tests for new elements ─────────────────────────────────────

    #[test]
    fn type_names_for_new_elements() {
        assert_eq!(
            ForceElement::GasSpring(GasSpringElement {
                body_a: "a".into(),
                point_a: [0.0, 0.0],
                body_b: "b".into(),
                point_b: [0.0, 0.0],
                initial_force: 1.0,
                extended_length: 1.0,
                stroke: 0.5,
                damping: 0.0,
                polytropic_exp: 1.0,
            })
            .type_name(),
            "Gas Spring"
        );
        assert_eq!(
            ForceElement::BearingFriction(BearingFrictionElement {
                body_i: "a".into(),
                body_j: "b".into(),
                constant_drag: 0.0,
                viscous_coeff: 0.0,
                coulomb_coeff: 0.0,
                pin_radius: 0.0,
                radial_load: 0.0,
                v_threshold: 0.01,
            })
            .type_name(),
            "Bearing Friction"
        );
        assert_eq!(
            ForceElement::JointLimit(JointLimitElement {
                body_i: "a".into(),
                body_j: "b".into(),
                angle_min: -1.0,
                angle_max: 1.0,
                stiffness: 1.0,
                damping: 0.0,
                restitution: 0.5,
            })
            .type_name(),
            "Joint Limit"
        );
        assert_eq!(
            ForceElement::Motor(MotorElement {
                body_i: "a".into(),
                body_j: "b".into(),
                stall_torque: 1.0,
                no_load_speed: 1.0,
                direction: 1.0,
            })
            .type_name(),
            "Motor"
        );
        assert_eq!(
            ForceElement::LinearActuator(LinearActuatorElement {
                body_a: "a".into(),
                point_a: [0.0, 0.0],
                body_b: "b".into(),
                point_b: [0.0, 0.0],
                force: 1.0,
                speed_limit: 0.0,
            })
            .type_name(),
            "Linear Actuator"
        );
    }

    // ── Coulomb friction convenience constructor ────────────────────────────

    #[test]
    fn coulomb_friction_produces_correct_torque() {
        let (state, bodies) = setup_two_bars();
        let q = state.make_q();
        let mut q_dot = DVector::zeros(state.n_coords());
        // bar2 spinning at 10 rad/s (fast enough that tanh ~ 1.0)
        q_dot[5] = 10.0;

        let elem = ForceElement::coulomb_friction("bar1", "bar2", 0.3, 0.01, 1000.0);
        let result = elem.evaluate(&state, &bodies, &q, &q_dot, 0.0);

        // μ * R * F_n = 0.3 * 0.01 * 1000 = 3.0
        // tanh(10.0 / 0.01) ~ 1.0
        // torque on bar2 = -3.0 (opposes positive rotation)
        assert_abs_diff_eq!(result[5], -3.0, epsilon = 1e-3); // bar2 θ
        assert_abs_diff_eq!(result[2], 3.0, epsilon = 1e-3); // bar1 θ (reaction)

        // Verify this is a pure Coulomb element (no drag, no viscous)
        match &elem {
            ForceElement::BearingFriction(b) => {
                assert_abs_diff_eq!(b.constant_drag, 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(b.viscous_coeff, 0.0, epsilon = 1e-15);
                assert_abs_diff_eq!(b.coulomb_coeff, 0.3, epsilon = 1e-15);
            }
            _ => panic!("Expected BearingFriction variant"),
        }
    }

    // ── Time modulation tests ──────────────────────────────────────────────

    #[test]
    fn sinusoidal_modulation_external_force() {
        let (state, bodies) = setup_single_bar();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let elem = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.0, 0.0],
            force: [100.0, 0.0],
            modulation: TimeModulation::Sinusoidal {
                omega: PI,
                phase: 0.0,
            },
        });

        // At t=0: sin(0) = 0 → force = 0
        let r0 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.0);
        assert_abs_diff_eq!(r0[0], 0.0, epsilon = 1e-14);

        // At t=0.5: sin(π * 0.5) = 1.0 → force = 100
        let r1 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.5);
        assert_abs_diff_eq!(r1[0], 100.0, epsilon = 1e-10);

        // At t=1.0: sin(π * 1.0) ~ 0 → force ~ 0
        let r2 = elem.evaluate(&state, &bodies, &q, &q_dot, 1.0);
        assert_abs_diff_eq!(r2[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn step_modulation_external_force() {
        let (state, bodies) = setup_single_bar();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let elem = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.0, 0.0],
            force: [50.0, 0.0],
            modulation: TimeModulation::Step { t_on: 1.0 },
        });

        // Before step: t=0.5 → factor = 0
        let r0 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.5);
        assert_abs_diff_eq!(r0[0], 0.0, epsilon = 1e-15);

        // At step: t=1.0 → factor = 1
        let r1 = elem.evaluate(&state, &bodies, &q, &q_dot, 1.0);
        assert_abs_diff_eq!(r1[0], 50.0, epsilon = 1e-15);

        // After step: t=2.0 → factor = 1
        let r2 = elem.evaluate(&state, &bodies, &q, &q_dot, 2.0);
        assert_abs_diff_eq!(r2[0], 50.0, epsilon = 1e-15);
    }

    #[test]
    fn expression_modulation_evaluates_correctly() {
        let (state, bodies) = setup_single_bar();
        let mut q = state.make_q();
        state.set_pose("bar", &mut q, 0.0, 0.0, 0.0);
        let q_dot = DVector::zeros(state.n_coords());

        let elem = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.0, 0.0],
            force: [100.0, 0.0],
            modulation: TimeModulation::Expression {
                expr: "sin(2*pi*t)".into(),
            },
        });

        // At t=0: sin(0) = 0 -> force = 0
        let r0 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.0);
        assert_abs_diff_eq!(r0[0], 0.0, epsilon = 1e-10);

        // At t=0.25: sin(pi/2) = 1.0 -> force = 100
        let r1 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.25);
        assert_abs_diff_eq!(r1[0], 100.0, epsilon = 1e-10);

        // At t=0.5: sin(pi) ~ 0 -> force ~ 0
        let r2 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.5);
        assert_abs_diff_eq!(r2[0], 0.0, epsilon = 1e-10);

        // At t=0.75: sin(3*pi/2) = -1.0 -> force = -100
        let r3 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.75);
        assert_abs_diff_eq!(r3[0], -100.0, epsilon = 1e-10);
    }

    #[test]
    fn expression_modulation_invalid_expr_returns_0() {
        // An invalid expression disables the force (factor = 0.0)
        let modulation = TimeModulation::Expression {
            expr: "not_a_valid_expr!!!".into(),
        };
        assert_abs_diff_eq!(modulation.factor(1.0), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn expression_modulation_exp_decay() {
        // Test exponential decay: 1 - exp(-t/0.5)
        let modulation = TimeModulation::Expression {
            expr: "1 - exp(-t/0.5)".into(),
        };

        // At t=0: 1 - exp(0) = 0
        assert_abs_diff_eq!(modulation.factor(0.0), 0.0, epsilon = 1e-10);

        // At t=0.5: 1 - exp(-1) ~ 0.6321
        assert_abs_diff_eq!(
            modulation.factor(0.5),
            1.0 - (-1.0_f64).exp(),
            epsilon = 1e-10
        );

        // At large t: should approach 1.0
        assert_abs_diff_eq!(modulation.factor(10.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn serde_roundtrip_expression_modulation() {
        let elem = ForceElement::ExternalForce(ExternalForceElement {
            body_id: "bar".into(),
            local_point: [0.5, 0.0],
            force: [10.0, -5.0],
            modulation: TimeModulation::Expression {
                expr: "sin(2*pi*t)".into(),
            },
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"modulation_type\":\"Expression\""));
        assert!(json.contains("sin(2*pi*t)"));

        let back: ForceElement = serde_json::from_str(&json).unwrap();
        match back {
            ForceElement::ExternalForce(f) => {
                match &f.modulation {
                    TimeModulation::Expression { expr } => {
                        assert_eq!(expr, "sin(2*pi*t)");
                        // Verify the deserialized expression evaluates correctly
                        assert_abs_diff_eq!(f.modulation.factor(0.25), 1.0, epsilon = 1e-10);
                    }
                    _ => panic!("Expected Expression modulation"),
                }
            }
            _ => panic!("Expected ExternalForce"),
        }
    }

    #[test]
    fn ramp_modulation_external_torque() {
        let (state, bodies) = setup_single_bar();
        let q = state.make_q();
        let q_dot = DVector::zeros(state.n_coords());

        let elem = ForceElement::ExternalTorque(ExternalTorqueElement {
            body_id: "bar".into(),
            torque: 20.0,
            modulation: TimeModulation::Ramp {
                t_start: 1.0,
                t_end: 3.0,
            },
        });

        // Before ramp: t=0.5 → factor = 0
        let r0 = elem.evaluate(&state, &bodies, &q, &q_dot, 0.5);
        assert_abs_diff_eq!(r0[2], 0.0, epsilon = 1e-15);

        // Mid-ramp: t=2.0 → factor = (2-1)/(3-1) = 0.5 → torque = 10
        let r1 = elem.evaluate(&state, &bodies, &q, &q_dot, 2.0);
        assert_abs_diff_eq!(r1[2], 10.0, epsilon = 1e-10);

        // After ramp: t=4.0 → factor = 1.0 → torque = 20
        let r2 = elem.evaluate(&state, &bodies, &q, &q_dot, 4.0);
        assert_abs_diff_eq!(r2[2], 20.0, epsilon = 1e-15);
    }
}
