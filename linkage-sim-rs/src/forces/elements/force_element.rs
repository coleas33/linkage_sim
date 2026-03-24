//! The top-level `ForceElement` enum and its impl block.

use std::collections::HashMap;

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::core::body::Body;
use crate::core::state::State;

use super::element_types::*;
use super::evaluation::*;

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
    ForceZone(ForceZoneElement),
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
            ForceElement::ForceZone(fz) => evaluate_force_zone(fz, state, bodies, q),
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
            ForceElement::ForceZone(_) => "Force Zone",
        }
    }

    /// Resolve any named mount/attachment points and cache their coordinates.
    ///
    /// For each force element variant that carries `point_X_name` fields, this
    /// looks up the named point on the corresponding body (checking both
    /// `attachment_points` and `mount_points`) and writes the resolved
    /// coordinates into the matching `point_X` field.  Variants without named
    /// points (e.g. `Gravity`, `TorsionSpring`) are returned unchanged.
    ///
    /// Call this once at build time (e.g. inside
    /// `load_mechanism_unbuilt_from_json`) so that the resolved coordinates are
    /// baked in before simulation starts.
    pub fn resolve_named_points(
        &self,
        bodies: &HashMap<String, Body>,
    ) -> Result<ForceElement, crate::core::body::BodyError> {
        let mut resolved = self.clone();
        match &mut resolved {
            ForceElement::LinearSpring(s) => {
                if let Some(ref name) = s.point_a_name.clone() {
                    let pt = bodies[&s.body_a].resolve_force_point(name)?;
                    s.point_a = [pt.x, pt.y];
                }
                if let Some(ref name) = s.point_b_name.clone() {
                    let pt = bodies[&s.body_b].resolve_force_point(name)?;
                    s.point_b = [pt.x, pt.y];
                }
            }
            ForceElement::LinearDamper(d) => {
                if let Some(ref name) = d.point_a_name.clone() {
                    let pt = bodies[&d.body_a].resolve_force_point(name)?;
                    d.point_a = [pt.x, pt.y];
                }
                if let Some(ref name) = d.point_b_name.clone() {
                    let pt = bodies[&d.body_b].resolve_force_point(name)?;
                    d.point_b = [pt.x, pt.y];
                }
            }
            ForceElement::GasSpring(g) => {
                if let Some(ref name) = g.point_a_name.clone() {
                    let pt = bodies[&g.body_a].resolve_force_point(name)?;
                    g.point_a = [pt.x, pt.y];
                }
                if let Some(ref name) = g.point_b_name.clone() {
                    let pt = bodies[&g.body_b].resolve_force_point(name)?;
                    g.point_b = [pt.x, pt.y];
                }
            }
            ForceElement::LinearActuator(a) => {
                if let Some(ref name) = a.point_a_name.clone() {
                    let pt = bodies[&a.body_a].resolve_force_point(name)?;
                    a.point_a = [pt.x, pt.y];
                }
                if let Some(ref name) = a.point_b_name.clone() {
                    let pt = bodies[&a.body_b].resolve_force_point(name)?;
                    a.point_b = [pt.x, pt.y];
                }
            }
            ForceElement::ExternalForce(e) => {
                if let Some(ref name) = e.local_point_name.clone() {
                    let pt = bodies[&e.body_id].resolve_force_point(name)?;
                    e.local_point = [pt.x, pt.y];
                }
            }
            _ => {}
        }
        Ok(resolved)
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
            ForceElement::ForceZone(fz) => vec![&fz.body_id],
        }
    }
}
