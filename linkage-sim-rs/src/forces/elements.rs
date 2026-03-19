//! Force element types for planar mechanisms.
//!
//! Each variant represents a force element that contributes to the
//! generalized force vector Q via the virtual work principle.

use std::collections::HashMap;

use nalgebra::{DVector, Vector2};
use serde::{Deserialize, Serialize};

use crate::core::body::Body;
use crate::core::state::{State, GROUND_ID};
use crate::forces::helpers::{body_torque_to_q, point_force_to_q};

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
/// Force direction is in global coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalForceElement {
    /// Body the force is applied to.
    pub body_id: String,
    /// Application point in body-local coordinates.
    pub local_point: [f64; 2],
    /// Force vector in global coordinates (N).
    pub force: [f64; 2],
}

/// External pure torque applied to a body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalTorqueElement {
    /// Body the torque is applied to.
    pub body_id: String,
    /// Torque magnitude (N·m). Positive = counterclockwise.
    pub torque: f64,
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
}

impl ForceElement {
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
            ForceElement::ExternalForce(f) => evaluate_external_force(f, state, q),
            ForceElement::ExternalTorque(t) => evaluate_external_torque(t, state),
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
        }
    }
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
    let theta_dot_i = if state.is_ground(&d.body_i) {
        0.0
    } else {
        let idx = state.get_index(&d.body_i).expect("body not registered");
        q_dot[idx.theta_idx()]
    };
    let theta_dot_j = if state.is_ground(&d.body_j) {
        0.0
    } else {
        let idx = state.get_index(&d.body_j).expect("body not registered");
        q_dot[idx.theta_idx()]
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
) -> DVector<f64> {
    let local_pt = Vector2::new(f.local_point[0], f.local_point[1]);
    let force = Vector2::new(f.force[0], f.force[1]);
    point_force_to_q(state, &f.body_id, &local_pt, &force, q)
}

fn evaluate_external_torque(t: &ExternalTorqueElement, state: &State) -> DVector<f64> {
    body_torque_to_q(state, &t.body_id, t.torque)
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
        });
        let json = serde_json::to_string(&elem).unwrap();
        assert!(json.contains("\"type\":\"ExternalForce\""));
    }
}
