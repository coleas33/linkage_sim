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

/// The end points shared by the four two-point forces (see
/// [`ForceElement::two_point_ends`]).
pub struct TwoPointEnds<'a> {
    pub body_a: &'a str,
    pub point_a: [f64; 2],
    pub point_a_name: Option<&'a str>,
    pub body_b: &'a str,
    pub point_b: [f64; 2],
    pub point_b_name: Option<&'a str>,
}

/// Mutable borrow of the end-point fields of a two-point force.
pub struct TwoPointEndsMut<'a> {
    pub body_a: &'a mut String,
    pub point_a: &'a mut [f64; 2],
    pub point_a_name: &'a mut Option<String>,
    pub body_b: &'a mut String,
    pub point_b: &'a mut [f64; 2],
    pub point_b_name: &'a mut Option<String>,
}

/// Look up a body for point resolution; an unknown id is an error, not a panic.
fn body_of<'a>(
    bodies: &'a HashMap<String, Body>,
    id: &str,
) -> Result<&'a Body, crate::core::body::BodyError> {
    bodies
        .get(id)
        .ok_or_else(|| crate::core::body::BodyError::UnknownBody { body: id.to_string() })
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
        if let Some(e) = resolved.two_point_ends_mut() {
            if let Some(name) = e.point_a_name.clone() {
                let pt = body_of(bodies, e.body_a)?.resolve_force_point(&name)?;
                *e.point_a = [pt.x, pt.y];
            }
            if let Some(name) = e.point_b_name.clone() {
                let pt = body_of(bodies, e.body_b)?.resolve_force_point(&name)?;
                *e.point_b = [pt.x, pt.y];
            }
        } else if let ForceElement::ExternalForce(e) = &mut resolved
            && let Some(ref name) = e.local_point_name.clone()
        {
            let pt = body_of(bodies, &e.body_id)?.resolve_force_point(name)?;
            e.local_point = [pt.x, pt.y];
        }
        Ok(resolved)
    }

    /// The two end points of a spring, damper, gas spring or linear actuator
    /// (the four elements that act between a point on body A and a point on
    /// body B); `None` for every other variant.
    pub fn two_point_ends(&self) -> Option<TwoPointEnds<'_>> {
        macro_rules! ends {
            ($e:expr) => {
                Some(TwoPointEnds {
                    body_a: &$e.body_a,
                    point_a: $e.point_a,
                    point_a_name: $e.point_a_name.as_deref(),
                    body_b: &$e.body_b,
                    point_b: $e.point_b,
                    point_b_name: $e.point_b_name.as_deref(),
                })
            };
        }
        match self {
            ForceElement::LinearSpring(e) => ends!(e),
            ForceElement::LinearDamper(e) => ends!(e),
            ForceElement::GasSpring(e) => ends!(e),
            ForceElement::LinearActuator(e) => ends!(e),
            _ => None,
        }
    }

    /// Mutable form of [`ForceElement::two_point_ends`].
    pub fn two_point_ends_mut(&mut self) -> Option<TwoPointEndsMut<'_>> {
        macro_rules! ends {
            ($e:expr) => {
                Some(TwoPointEndsMut {
                    body_a: &mut $e.body_a,
                    point_a: &mut $e.point_a,
                    point_a_name: &mut $e.point_a_name,
                    body_b: &mut $e.body_b,
                    point_b: &mut $e.point_b,
                    point_b_name: &mut $e.point_b_name,
                })
            };
        }
        match self {
            ForceElement::LinearSpring(e) => ends!(e),
            ForceElement::LinearDamper(e) => ends!(e),
            ForceElement::GasSpring(e) => ends!(e),
            ForceElement::LinearActuator(e) => ends!(e),
            _ => None,
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
            ForceElement::ForceZone(fz) => vec![&fz.body_id],
        }
    }
}
