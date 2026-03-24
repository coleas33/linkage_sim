//! Element data structs for all force element types.

use serde::{Deserialize, Serialize};

use super::time_modulation::TimeModulation;

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
fn default_end_stop_stiffness() -> f64 {
    10000.0
}
fn default_end_stop_damping() -> f64 {
    10.0
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
    /// Optional named mount point reference for point A.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_a_name: Option<String>,
    /// Body B identifier.
    pub body_b: String,
    /// Attachment point in body B local coordinates.
    pub point_b: [f64; 2],
    /// Optional named mount point reference for point B.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_b_name: Option<String>,
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
    /// Optional named mount point reference for point A.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_a_name: Option<String>,
    /// Body B identifier.
    pub body_b: String,
    /// Attachment point in body B local coordinates.
    pub point_b: [f64; 2],
    /// Optional named mount point reference for point B.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_b_name: Option<String>,
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
    /// Optional named mount point reference for local_point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_point_name: Option<String>,
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
    /// Optional named mount point reference for point A.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_a_name: Option<String>,
    /// Body B identifier.
    pub body_b: String,
    /// Attachment point in body B local coordinates.
    pub point_b: [f64; 2],
    /// Optional named mount point reference for point B.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_b_name: Option<String>,
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
    /// Optional named mount point reference for point A.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_a_name: Option<String>,
    /// Body B identifier.
    pub body_b: String,
    /// Attachment point in body B local coordinates.
    pub point_b: [f64; 2],
    /// Optional named mount point reference for point B.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub point_b_name: Option<String>,
    /// Actuator force (N). Positive = extension/push apart.
    pub force: f64,
    /// Maximum extension rate (m/s). 0 = no limit.
    #[serde(default)]
    pub speed_limit: f64,
    /// Minimum stroke length (m). 0 = no min limit.
    #[serde(default)]
    pub stroke_min: f64,
    /// Maximum stroke length (m). 0 = no max limit.
    #[serde(default)]
    pub stroke_max: f64,
    /// End-stop penalty spring stiffness (N/m).
    #[serde(default = "default_end_stop_stiffness")]
    pub end_stop_stiffness: f64,
    /// End-stop penalty damping (N·s/m).
    #[serde(default = "default_end_stop_damping")]
    pub end_stop_damping: f64,
    /// End-stop coefficient of restitution [0,1].
    #[serde(default = "default_restitution")]
    pub end_stop_restitution: f64,
}

/// A spatial force zone: applies a constant distributed force to a body
/// proportional to the overlap area between the body's geometry and the zone.
///
/// The zone is an axis-aligned rectangle in world space. The body must have
/// `BodyGeometry` set. Force is applied at the centroid of the overlap region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceZoneElement {
    /// ID of the body whose geometry is tested for overlap.
    pub body_id: String,
    /// World-space bottom-left corner of the zone (meters).
    pub zone_min: [f64; 2],
    /// World-space top-right corner of the zone (meters).
    pub zone_max: [f64; 2],
    /// Constant force vector applied at full overlap (Newtons).
    pub force: [f64; 2],
    /// Optional display label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}
