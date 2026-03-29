//! JSON schema types for mechanism serialization.
//!
//! All values are in SI units.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::forces::elements::ForceElement;

/// Returns true if the value is zero — used for `skip_serializing_if` on
/// backward-compatible fields that default to 0.
fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// Current schema version for the JSON format.
pub const SCHEMA_VERSION: &str = "1.1.0";

/// Extract the major version number from a semver string (e.g., "1.2.3" -> 1).
/// Returns `None` if the string doesn't start with a valid integer.
pub(crate) fn semver_major(version: &str) -> Option<u32> {
    version.split('.').next()?.parse().ok()
}

/// Configuration for sweep analysis angle range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepConfig {
    /// Minimum sweep angle in radians.
    pub angle_min: f64,
    /// Maximum sweep angle in radians.
    pub angle_max: f64,
    /// When false, full 360deg sweep regardless of min/max.
    pub enabled: bool,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            angle_min: 0.0,
            angle_max: 2.0 * std::f64::consts::PI,
            enabled: false,
        }
    }
}

/// Top-level JSON representation of a mechanism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanismJson {
    pub schema_version: String,
    pub bodies: HashMap<String, BodyJson>,
    pub joints: HashMap<String, JointJson>,
    /// Serialized driver constraints. Only constant-speed revolute drivers are
    /// supported; other driver types are skipped with a warning.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub drivers: HashMap<String, DriverJson>,
    /// Named load cases (driver configurations) for scenario comparison.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub load_cases: Vec<LoadCaseJson>,
    /// Force elements attached to the mechanism (springs, dampers, external loads, etc.).
    /// Backward-compatible: old files without this field default to an empty list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forces: Vec<ForceElement>,
    /// Sweep analysis angle range configuration.
    /// Backward-compatible: old files without this field default to None (full 360deg).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sweep_config: Option<SweepConfig>,
    /// Mechanism mounting angle in radians. Rotates the mechanism relative to
    /// gravity (0 = horizontal, positive = counterclockwise). Backward-compatible.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub mounting_angle: f64,
}

/// JSON representation of a load case -- a named driver configuration.
///
/// Engineers use load cases to compare different operating conditions on the
/// same mechanism geometry without rebuilding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadCaseJson {
    pub name: String,
    pub driver_joint_id: String,
    pub omega: f64,   // rad/s
    pub theta_0: f64, // rad
}

/// JSON representation of a driver constraint.
///
/// Constant-speed and expression-based revolute drivers can be serialized.
/// General closure-based drivers (those without `DriverMeta`) are omitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DriverJson {
    ConstantSpeed {
        body_i: String,
        body_j: String,
        /// Angular velocity in rad/s.
        omega: f64,
        /// Initial angle offset in rad.
        theta_0: f64,
    },
    /// User-defined expression driver: f(t), f'(t), f''(t) as math strings.
    Expression {
        body_i: String,
        body_j: String,
        /// Position expression, e.g. `"2*pi*t"` or `"pi/2 * sin(3*t)"`.
        expr: String,
        /// Velocity expression (first derivative of `expr`).
        expr_dot: String,
        /// Acceleration expression (second derivative of `expr`).
        expr_ddot: String,
    },
}

/// A point mass attached to a body at a local position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointMassJson {
    pub mass: f64,
    pub local_pos: [f64; 2],
}

/// JSON representation of a rigid body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyJson {
    pub attachment_points: HashMap<String, [f64; 2]>,
    pub mass: f64,
    pub cg_local: [f64; 2],
    pub izz_cg: f64,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub mount_points: HashMap<String, [f64; 2]>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub coupler_points: HashMap<String, [f64; 2]>,
    /// Point masses attached to this body. Applied during build to update
    /// composite mass, CG, and Izz via parallel axis theorem.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub point_masses: Vec<PointMassJson>,
    /// User-editable display label (defaults to body ID if absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Optional visual geometry for rendering and force zone overlap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<crate::core::body::BodyGeometry>,
}

/// JSON representation of a joint constraint.
///
/// Uses an internally-tagged enum so the JSON has a `"type"` field.
/// Driver joints are represented as a marker -- the closure cannot be serialized.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JointJson {
    Revolute {
        body_i: String,
        body_j: String,
        point_i: String,
        point_j: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    Fixed {
        body_i: String,
        body_j: String,
        point_i: String,
        point_j: String,
        #[serde(default)]
        delta_theta_0: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    Prismatic {
        body_i: String,
        body_j: String,
        point_i: String,
        point_j: String,
        axis_local_i: [f64; 2],
        #[serde(default)]
        delta_theta_0: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    /// Cam-follower joint with profile.
    CamFollower {
        body_i: String,
        body_j: String,
        point_i: String,
        point_j: String,
        follower_direction: [f64; 2],
        #[serde(flatten)]
        profile: crate::core::constraint::CamProfile,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    /// Marker for driver constraints. The closure is not serializable;
    /// users must re-attach the driver function after loading.
    RevoluteDriver {
        body_i: String,
        body_j: String,
        #[serde(default)]
        note: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
}
