//! JSON serialization and deserialization of mechanisms.
//!
//! All values are in SI units. Schema versioning supports forward migration.
//!
//! **Known limitation:** Driver constraints use closures, which cannot be
//! serialized. Drivers are skipped on save and must be re-attached after load.

use std::collections::HashMap;

use nalgebra::Vector2;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::body::Body;
use crate::core::constraint::{Constraint, JointConstraint};
use crate::core::driver::DriverMeta;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::forces::elements::ForceElement;

/// Current schema version for the JSON format.
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Extract the major version number from a semver string (e.g., "1.2.3" → 1).
/// Returns `None` if the string doesn't start with a valid integer.
fn semver_major(version: &str) -> Option<u32> {
    version.split('.').next()?.parse().ok()
}

// ---------------------------------------------------------------------------
// JSON schema types
// ---------------------------------------------------------------------------

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
}

/// JSON representation of a load case — a named driver configuration.
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
}

/// JSON representation of a joint constraint.
///
/// Uses an internally-tagged enum so the JSON has a `"type"` field.
/// Driver joints are represented as a marker — the closure cannot be serialized.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum JointJson {
    Revolute {
        body_i: String,
        body_j: String,
        point_i: String,
        point_j: String,
    },
    Fixed {
        body_i: String,
        body_j: String,
        point_i: String,
        point_j: String,
        #[serde(default)]
        delta_theta_0: f64,
    },
    Prismatic {
        body_i: String,
        body_j: String,
        point_i: String,
        point_j: String,
        axis_local_i: [f64; 2],
        #[serde(default)]
        delta_theta_0: f64,
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
    },
    /// Marker for driver constraints. The closure is not serializable;
    /// users must re-attach the driver function after loading.
    RevoluteDriver {
        body_i: String,
        body_j: String,
        #[serde(default)]
        note: String,
    },
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum SerializationError {
    #[error("Unsupported schema version '{found}' (expected major version compatible with '{expected}').")]
    UnsupportedVersion { found: String, expected: String },

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Failed to build mechanism: {0}")]
    Build(String),

    #[error(
        "Could not find attachment point name on body '{body_id}' \
         for local coordinates [{x}, {y}]"
    )]
    PointNameNotFound {
        body_id: String,
        x: f64,
        y: f64,
    },

    #[error("Unknown joint type '{0}'")]
    UnknownJointType(String),
}

// ---------------------------------------------------------------------------
// Mechanism → JSON
// ---------------------------------------------------------------------------

/// Reverse-lookup: find the attachment-point name on a body whose local
/// coordinates match the given vector (within floating-point tolerance).
fn find_point_name(
    body: &Body,
    coords: &Vector2<f64>,
    body_id: &str,
) -> Result<String, SerializationError> {
    const TOL: f64 = 1e-12;
    for (name, pt) in &body.attachment_points {
        if (pt - coords).norm() < TOL {
            return Ok(name.clone());
        }
    }
    for (name, pt) in &body.mount_points {
        if (pt.x - coords.x).abs() < TOL && (pt.y - coords.y).abs() < TOL {
            return Ok(name.clone());
        }
    }
    Err(SerializationError::PointNameNotFound {
        body_id: body_id.to_string(),
        x: coords.x,
        y: coords.y,
    })
}

/// Convert a `Body` to its JSON representation.
fn body_to_json(body: &Body) -> BodyJson {
    let attachment_points = body
        .attachment_points
        .iter()
        .map(|(name, pt)| (name.clone(), [pt.x, pt.y]))
        .collect();

    let mount_points = body
        .mount_points
        .iter()
        .map(|(name, pt)| (name.clone(), [pt.x, pt.y]))
        .collect();

    let coupler_points = body
        .coupler_points
        .iter()
        .map(|(name, pt)| (name.clone(), [pt.x, pt.y]))
        .collect();

    BodyJson {
        attachment_points,
        mass: body.mass,
        cg_local: [body.cg_local.x, body.cg_local.y],
        izz_cg: body.izz_cg,
        mount_points,
        coupler_points,
        // Point masses are a blueprint-level concept — they modify mass/CG/Izz
        // at build time. When exporting from a built mechanism, the composite
        // properties are already baked in, so we emit an empty list.
        point_masses: Vec::new(),
    }
}

/// Convert a `JointConstraint` to its JSON representation.
///
/// Requires access to the mechanism's bodies to reverse-lookup point names.
fn joint_to_json(
    joint: &JointConstraint,
    bodies: &HashMap<String, Body>,
) -> Result<JointJson, SerializationError> {
    match joint {
        JointConstraint::Revolute(j) => {
            let body_i_id = j.body_i_id();
            let body_j_id = j.body_j_id();
            let body_i = bodies.get(body_i_id).ok_or_else(|| SerializationError::Build(format!("body '{}' not found", body_i_id)))?;
            let body_j = bodies.get(body_j_id).ok_or_else(|| SerializationError::Build(format!("body '{}' not found", body_j_id)))?;
            Ok(JointJson::Revolute {
                body_i: body_i_id.to_string(),
                body_j: body_j_id.to_string(),
                point_i: find_point_name(body_i, j.point_i_local(), body_i_id)?,
                point_j: find_point_name(body_j, j.point_j_local(), body_j_id)?,
            })
        }
        JointConstraint::Fixed(j) => {
            let body_i_id = j.body_i_id();
            let body_j_id = j.body_j_id();
            let body_i = bodies.get(body_i_id).ok_or_else(|| SerializationError::Build(format!("body '{}' not found", body_i_id)))?;
            let body_j = bodies.get(body_j_id).ok_or_else(|| SerializationError::Build(format!("body '{}' not found", body_j_id)))?;
            Ok(JointJson::Fixed {
                body_i: body_i_id.to_string(),
                body_j: body_j_id.to_string(),
                point_i: find_point_name(body_i, j.point_i_local(), body_i_id)?,
                point_j: find_point_name(body_j, j.point_j_local(), body_j_id)?,
                delta_theta_0: j.delta_theta_0(),
            })
        }
        JointConstraint::Prismatic(j) => {
            let body_i_id = j.body_i_id();
            let body_j_id = j.body_j_id();
            let body_i = bodies.get(body_i_id).ok_or_else(|| SerializationError::Build(format!("body '{}' not found", body_i_id)))?;
            let body_j = bodies.get(body_j_id).ok_or_else(|| SerializationError::Build(format!("body '{}' not found", body_j_id)))?;
            Ok(JointJson::Prismatic {
                body_i: body_i_id.to_string(),
                body_j: body_j_id.to_string(),
                point_i: find_point_name(body_i, j.point_i_local(), body_i_id)?,
                point_j: find_point_name(body_j, j.point_j_local(), body_j_id)?,
                axis_local_i: [j.axis_local_i().x, j.axis_local_i().y],
                delta_theta_0: j.delta_theta_0(),
            })
        }
        JointConstraint::CamFollower(j) => {
            let body_i_id = j.body_i_id();
            let body_j_id = j.body_j_id();
            let body_i = bodies.get(body_i_id).ok_or_else(|| SerializationError::Build(format!("body '{}' not found", body_i_id)))?;
            let body_j = bodies.get(body_j_id).ok_or_else(|| SerializationError::Build(format!("body '{}' not found", body_j_id)))?;
            Ok(JointJson::CamFollower {
                body_i: body_i_id.to_string(),
                body_j: body_j_id.to_string(),
                point_i: find_point_name(body_i, &j.point_i_local, body_i_id)?,
                point_j: find_point_name(body_j, &j.point_j_local, body_j_id)?,
                follower_direction: [j.follower_dir.x, j.follower_dir.y],
                profile: j.profile.clone(),
            })
        }
    }
}

/// Convert a `Mechanism` to its JSON-compatible struct.
///
/// Constant-speed revolute drivers are serialized with their `omega` and
/// `theta_0` parameters. General closure-based drivers (those without
/// `DriverMeta`) are silently skipped.
pub fn mechanism_to_json(mech: &Mechanism) -> Result<MechanismJson, SerializationError> {
    let bodies: HashMap<String, BodyJson> = mech
        .bodies()
        .iter()
        .map(|(id, body)| (id.clone(), body_to_json(body)))
        .collect();

    let mut joints = HashMap::new();
    for joint in mech.joints() {
        let id = joint.id().to_string();
        joints.insert(id, joint_to_json(joint, mech.bodies())?);
    }

    let mut drivers = HashMap::new();
    for driver in mech.drivers() {
        if let Some(meta) = driver.meta() {
            let id = driver.id().to_string();
            let driver_json = match meta {
                DriverMeta::ConstantSpeed { omega, theta_0 } => DriverJson::ConstantSpeed {
                    body_i: driver.body_i_id().to_string(),
                    body_j: driver.body_j_id().to_string(),
                    omega: *omega,
                    theta_0: *theta_0,
                },
                DriverMeta::Expression {
                    expr,
                    expr_dot,
                    expr_ddot,
                } => DriverJson::Expression {
                    body_i: driver.body_i_id().to_string(),
                    body_j: driver.body_j_id().to_string(),
                    expr: expr.clone(),
                    expr_dot: expr_dot.clone(),
                    expr_ddot: expr_ddot.clone(),
                },
            };
            drivers.insert(id, driver_json);
        }
        // Drivers without metadata (general closures) are silently skipped.
    }

    Ok(MechanismJson {
        schema_version: SCHEMA_VERSION.to_string(),
        bodies,
        joints,
        drivers,
        load_cases: Vec::new(),
        forces: mech.forces().to_vec(),
    })
}

/// Serialize a `Mechanism` to a JSON string.
///
/// Driver constraints are skipped — they must be re-attached after loading.
pub fn save_mechanism(mech: &Mechanism) -> Result<String, SerializationError> {
    let json_struct = mechanism_to_json(mech)?;
    let json_str = serde_json::to_string_pretty(&json_struct)?;
    Ok(json_str)
}

// ---------------------------------------------------------------------------
// JSON → Mechanism
// ---------------------------------------------------------------------------

/// Deserialize a JSON string into an **unbuilt** `Mechanism`.
///
/// The caller can add drivers or make other modifications before calling
/// `mech.build()`. Driver constraints from the JSON are skipped (closures
/// are not serializable).
pub fn load_mechanism_unbuilt(json_str: &str) -> Result<Mechanism, SerializationError> {
    let json_struct: MechanismJson = serde_json::from_str(json_str)?;
    load_mechanism_unbuilt_from_json(&json_struct)
}

/// Build an **unbuilt** `Mechanism` directly from a `MechanismJson` struct.
///
/// Same as [`load_mechanism_unbuilt`] but skips the JSON parsing step.
/// Useful when you already have a `MechanismJson` in memory (e.g., from
/// the editor blueprint).
pub fn load_mechanism_unbuilt_from_json(json_struct: &MechanismJson) -> Result<Mechanism, SerializationError> {
    let found_major = semver_major(&json_struct.schema_version);
    let expected_major = semver_major(SCHEMA_VERSION);
    if found_major != expected_major {
        return Err(SerializationError::UnsupportedVersion {
            found: json_struct.schema_version.clone(),
            expected: SCHEMA_VERSION.to_string(),
        });
    }

    let mut mech = Mechanism::new();

    // Rebuild bodies
    for (body_id, body_json) in &json_struct.bodies {
        let attachment_points: HashMap<String, Vector2<f64>> = body_json
            .attachment_points
            .iter()
            .map(|(name, coords)| (name.clone(), Vector2::new(coords[0], coords[1])))
            .collect();

        let mount_points: HashMap<String, Vector2<f64>> = body_json
            .mount_points
            .iter()
            .map(|(k, v)| (k.clone(), Vector2::new(v[0], v[1])))
            .collect();

        let coupler_points: HashMap<String, Vector2<f64>> = body_json
            .coupler_points
            .iter()
            .map(|(name, coords)| (name.clone(), Vector2::new(coords[0], coords[1])))
            .collect();

        let body = Body {
            id: body_id.clone(),
            attachment_points,
            mass: if body_id == GROUND_ID {
                0.0
            } else {
                body_json.mass
            },
            cg_local: Vector2::new(body_json.cg_local[0], body_json.cg_local[1]),
            izz_cg: if body_id == GROUND_ID {
                0.0
            } else {
                body_json.izz_cg
            },
            mount_points,
            coupler_points,
        };
        mech.add_body(body)
            .map_err(|e| SerializationError::Build(e.to_string()))?;

        // Apply point masses to update composite mass/CG/Izz
        for pm in &body_json.point_masses {
            if let Some(body_mut) = mech.body_mut(body_id) {
                body_mut.add_point_mass(pm.mass, Vector2::new(pm.local_pos[0], pm.local_pos[1]));
            }
        }
    }

    // Rebuild joints (geometric constraints only; drivers are separate)
    for (joint_id, joint_json) in &json_struct.joints {
        match joint_json {
            JointJson::Revolute {
                body_i,
                body_j,
                point_i,
                point_j,
            } => {
                mech.add_revolute_joint(joint_id, body_i, point_i, body_j, point_j)
                    .map_err(|e| SerializationError::Build(e.to_string()))?;
            }
            JointJson::Fixed {
                body_i,
                body_j,
                point_i,
                point_j,
                delta_theta_0,
            } => {
                mech.add_fixed_joint(
                    joint_id,
                    body_i,
                    point_i,
                    body_j,
                    point_j,
                    *delta_theta_0,
                )
                .map_err(|e| SerializationError::Build(e.to_string()))?;
            }
            JointJson::Prismatic {
                body_i,
                body_j,
                point_i,
                point_j,
                axis_local_i,
                delta_theta_0,
            } => {
                mech.add_prismatic_joint(
                    joint_id,
                    body_i,
                    point_i,
                    body_j,
                    point_j,
                    Vector2::new(axis_local_i[0], axis_local_i[1]),
                    *delta_theta_0,
                )
                .map_err(|e| SerializationError::Build(e.to_string()))?;
            }
            JointJson::CamFollower {
                body_i,
                body_j,
                point_i,
                point_j,
                follower_direction,
                profile,
            } => {
                mech.add_cam_follower_joint(
                    joint_id,
                    body_i,
                    point_i,
                    body_j,
                    point_j,
                    Vector2::new(follower_direction[0], follower_direction[1]),
                    profile.clone(),
                )
                .map_err(|e| SerializationError::Build(e.to_string()))?;
            }
            JointJson::RevoluteDriver { .. } => {
                // Legacy format: drivers in the joints map. Skip silently —
                // they are handled via the top-level `drivers` map now.
            }
        }
    }

    // Rebuild drivers
    for (driver_id, driver_json) in &json_struct.drivers {
        match driver_json {
            DriverJson::ConstantSpeed {
                body_i,
                body_j,
                omega,
                theta_0,
            } => {
                mech.add_constant_speed_driver(driver_id, body_i, body_j, *omega, *theta_0)
                    .map_err(|e| SerializationError::Build(e.to_string()))?;
            }
            DriverJson::Expression {
                body_i,
                body_j,
                expr,
                expr_dot,
                expr_ddot,
            } => {
                mech.add_expression_driver(
                    driver_id, body_i, body_j, expr, expr_dot, expr_ddot,
                )
                .map_err(|e| SerializationError::Build(e.to_string()))?;
            }
        }
    }

    // Restore force elements, resolving any named mount/attachment points
    // against the bodies that were just added to the mechanism.
    let bodies = mech.bodies().clone();
    let resolved_forces: Vec<ForceElement> = json_struct.forces
        .iter()
        .map(|f| f.resolve_named_points(&bodies).unwrap_or_else(|e| {
            log::warn!("Failed to resolve force point name: {e}");
            f.clone()
        }))
        .collect();
    for force in resolved_forces {
        mech.add_force(force);
    }

    Ok(mech)
}

/// Deserialize a JSON string into a **built** `Mechanism`.
///
/// Driver constraints are **not** restored (closures are not serializable).
/// If you need to add drivers before building, use [`load_mechanism_unbuilt`]
/// instead.
pub fn load_mechanism(json_str: &str) -> Result<Mechanism, SerializationError> {
    let mut mech = load_mechanism_unbuilt(json_str)?;
    mech.build()
        .map_err(|e| SerializationError::Build(e.to_string()))?;
    Ok(mech)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground, Body};
    use crate::solver::kinematics::solve_position;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    /// Build a 4-bar mechanism (no driver).
    fn build_fourbar_no_driver() -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let mut coupler = make_bar("coupler", "B", "C", 3.0, 3.0, 0.05);
        coupler.add_coupler_point("P", 1.5, 0.5).unwrap();
        let rocker = make_bar("rocker", "D", "C", 2.0, 2.0, 0.02);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();

        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
            .unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
            .unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
            .unwrap();
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
            .unwrap();

        mech.build().unwrap();
        mech
    }

    /// Build a slider-crank mechanism (no driver).
    fn build_slidercrank_no_driver() -> Mechanism {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("rail", 3.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 1.0, 0.01);
        let conrod = make_bar("conrod", "B", "C", 3.0, 2.0, 0.1);

        let mut slider = Body::new("slider");
        slider.add_attachment_point("C", 0.0, 0.0).unwrap();
        slider.mass = 0.5;

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(conrod).unwrap();
        mech.add_body(slider).unwrap();

        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
            .unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "conrod", "B")
            .unwrap();
        mech.add_revolute_joint("J3", "conrod", "C", "slider", "C")
            .unwrap();
        mech.add_prismatic_joint(
            "P1",
            "ground",
            "rail",
            "slider",
            "C",
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();

        mech.build().unwrap();
        mech
    }

    // -----------------------------------------------------------------------
    // Round-trip: 4-bar
    // -----------------------------------------------------------------------

    #[test]
    fn fourbar_round_trip_preserves_structure() {
        let mech = build_fourbar_no_driver();

        let json_str = save_mechanism(&mech).unwrap();
        let loaded = load_mechanism(&json_str).unwrap();

        // Body count matches (including ground)
        assert_eq!(loaded.bodies().len(), mech.bodies().len());
        assert_eq!(loaded.bodies().len(), 4);

        // Joint count matches
        assert_eq!(loaded.joints().len(), mech.joints().len());
        assert_eq!(loaded.joints().len(), 4);

        // Moving body count matches
        assert_eq!(loaded.state().n_moving_bodies(), 3);
        assert_eq!(loaded.state().n_coords(), 9);
    }

    #[test]
    fn fourbar_round_trip_preserves_attachment_points() {
        let mech = build_fourbar_no_driver();

        let json_str = save_mechanism(&mech).unwrap();
        let loaded = load_mechanism(&json_str).unwrap();

        // Verify each body's attachment points survived
        for (body_id, original_body) in mech.bodies() {
            let loaded_body = loaded.bodies().get(body_id).unwrap();
            assert_eq!(
                original_body.attachment_points.len(),
                loaded_body.attachment_points.len(),
                "Attachment point count mismatch for body '{}'",
                body_id,
            );
            for (pt_name, pt_coords) in &original_body.attachment_points {
                let loaded_pt = loaded_body.attachment_points.get(pt_name).unwrap();
                assert_abs_diff_eq!(loaded_pt.x, pt_coords.x, epsilon = 1e-15);
                assert_abs_diff_eq!(loaded_pt.y, pt_coords.y, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn fourbar_round_trip_preserves_coupler_points() {
        let mech = build_fourbar_no_driver();

        let json_str = save_mechanism(&mech).unwrap();
        let loaded = load_mechanism(&json_str).unwrap();

        let original_coupler = mech.bodies().get("coupler").unwrap();
        let loaded_coupler = loaded.bodies().get("coupler").unwrap();

        assert_eq!(original_coupler.coupler_points.len(), 1);
        assert_eq!(loaded_coupler.coupler_points.len(), 1);

        let orig_p = &original_coupler.coupler_points["P"];
        let load_p = &loaded_coupler.coupler_points["P"];
        assert_abs_diff_eq!(load_p.x, orig_p.x, epsilon = 1e-15);
        assert_abs_diff_eq!(load_p.y, orig_p.y, epsilon = 1e-15);
    }

    #[test]
    fn fourbar_round_trip_preserves_body_properties() {
        let mech = build_fourbar_no_driver();

        let json_str = save_mechanism(&mech).unwrap();
        let loaded = load_mechanism(&json_str).unwrap();

        for (body_id, original_body) in mech.bodies() {
            let loaded_body = loaded.bodies().get(body_id).unwrap();
            assert_abs_diff_eq!(loaded_body.mass, original_body.mass, epsilon = 1e-15);
            assert_abs_diff_eq!(loaded_body.izz_cg, original_body.izz_cg, epsilon = 1e-15);
            assert_abs_diff_eq!(
                loaded_body.cg_local.x,
                original_body.cg_local.x,
                epsilon = 1e-15
            );
            assert_abs_diff_eq!(
                loaded_body.cg_local.y,
                original_body.cg_local.y,
                epsilon = 1e-15
            );
        }
    }

    // -----------------------------------------------------------------------
    // Round-trip: slider-crank (prismatic joint)
    // -----------------------------------------------------------------------

    #[test]
    fn slidercrank_round_trip_preserves_structure() {
        let mech = build_slidercrank_no_driver();

        let json_str = save_mechanism(&mech).unwrap();
        let loaded = load_mechanism(&json_str).unwrap();

        assert_eq!(loaded.bodies().len(), 4);
        assert_eq!(loaded.joints().len(), 4); // 3 revolute + 1 prismatic

        // Verify the prismatic joint survived
        let has_prismatic = loaded.joints().iter().any(|j| {
            matches!(j, JointConstraint::Prismatic(_))
        });
        assert!(has_prismatic, "Prismatic joint not found after round-trip");
    }

    #[test]
    fn slidercrank_round_trip_preserves_prismatic_params() {
        let mech = build_slidercrank_no_driver();

        let json_str = save_mechanism(&mech).unwrap();

        // Verify the JSON contains the prismatic joint data
        let json_struct: MechanismJson = serde_json::from_str(&json_str).unwrap();
        let p1 = &json_struct.joints["P1"];
        match p1 {
            JointJson::Prismatic {
                axis_local_i,
                delta_theta_0,
                ..
            } => {
                assert_abs_diff_eq!(axis_local_i[0], 1.0, epsilon = 1e-12);
                assert_abs_diff_eq!(axis_local_i[1], 0.0, epsilon = 1e-12);
                assert_abs_diff_eq!(*delta_theta_0, 0.0, epsilon = 1e-12);
            }
            other => panic!("Expected Prismatic joint, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Round-trip + kinematic solve
    // -----------------------------------------------------------------------

    #[test]
    fn loaded_fourbar_can_solve_kinematics() {
        // Save, load as unbuilt, add driver, build, then solve.
        let mech_original = build_fourbar_no_driver();
        let json_str = save_mechanism(&mech_original).unwrap();

        let mut mech = load_mechanism_unbuilt(&json_str).unwrap();
        mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();
        mech.build().unwrap();

        let state = mech.state();
        let angle: f64 = 0.5;
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("coupler", &mut q0, angle.cos(), angle.sin(), 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);

        let result = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Kinematics solve on loaded 4-bar did not converge, residual = {}",
            result.residual_norm,
        );
        assert!(result.residual_norm < 1e-10);
    }

    #[test]
    fn loaded_slidercrank_can_solve_kinematics() {
        // Save, load as unbuilt, add driver, build, then solve.
        let mech_original = build_slidercrank_no_driver();
        let json_str = save_mechanism(&mech_original).unwrap();

        let mut mech = load_mechanism_unbuilt(&json_str).unwrap();
        mech.add_revolute_driver("D1", "ground", "crank", |t| t, |_t| 1.0, |_t| 0.0)
            .unwrap();
        mech.build().unwrap();

        let state = mech.state();
        let angle: f64 = 0.5;
        let bx = angle.cos();
        let by = angle.sin();
        let phi = (-by / 3.0).asin();
        let cx = bx + 3.0 * phi.cos();
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("conrod", &mut q0, bx, by, phi);
        state.set_pose("slider", &mut q0, cx, 0.0, 0.0);

        let result = solve_position(&mech, &q0, angle, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Slider-crank solve on loaded mechanism did not converge, residual = {}",
            result.residual_norm,
        );
        assert!(result.residual_norm < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Schema version validation
    // -----------------------------------------------------------------------

    #[test]
    fn unsupported_schema_version_rejected() {
        let json = r#"{
            "schema_version": "99.0.0",
            "bodies": {},
            "joints": {}
        }"#;

        let err = match load_mechanism(json) {
            Err(e) => e,
            Ok(_) => panic!("Expected error for unsupported version, got Ok"),
        };
        assert!(
            err.to_string().contains("99.0.0"),
            "Error should mention the bad version: {}",
            err,
        );
    }

    // -----------------------------------------------------------------------
    // JSON structure validation
    // -----------------------------------------------------------------------

    #[test]
    fn save_produces_valid_json_with_schema_version() {
        let mech = build_fourbar_no_driver();
        let json_str = save_mechanism(&mech).unwrap();

        let json_struct: MechanismJson = serde_json::from_str(&json_str).unwrap();
        assert_eq!(json_struct.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn driver_joint_serialization_marker() {
        // Build a mechanism with a driver, save it, and verify the driver
        // appears in the dedicated drivers map (not the joints map).
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
            .unwrap();
        mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0)
            .unwrap();
        mech.build().unwrap();

        let json_str = save_mechanism(&mech).unwrap();
        let json_struct: MechanismJson = serde_json::from_str(&json_str).unwrap();

        // The revolute joint is in the joints map.
        assert_eq!(json_struct.joints.len(), 1);
        assert!(json_struct.joints.contains_key("J1"));

        // The driver is in the dedicated drivers map.
        assert_eq!(json_struct.drivers.len(), 1);
        assert!(json_struct.drivers.contains_key("D1"));
    }

    #[test]
    fn constant_speed_driver_round_trips_omega_and_theta0() {
        let omega = 3.0 * PI;
        let theta_0 = PI / 4.0;

        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let coupler = make_bar("coupler", "B", "C", 3.0, 1.5, 0.05);
        let rocker = make_bar("rocker", "D", "C", 2.0, 1.5, 0.02);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D").unwrap();
        mech.add_constant_speed_driver("D1", "ground", "crank", omega, theta_0)
            .unwrap();
        mech.build().unwrap();

        // Round-trip through JSON.
        let json_str = save_mechanism(&mech).unwrap();
        let json_struct: MechanismJson = serde_json::from_str(&json_str).unwrap();

        // Verify omega and theta_0 are preserved in JSON.
        match &json_struct.drivers["D1"] {
            DriverJson::ConstantSpeed {
                omega: j_omega,
                theta_0: j_theta_0,
                body_i,
                body_j,
            } => {
                assert_abs_diff_eq!(*j_omega, omega, epsilon = 1e-15);
                assert_abs_diff_eq!(*j_theta_0, theta_0, epsilon = 1e-15);
                assert_eq!(body_i, "ground");
                assert_eq!(body_j, "crank");
            }
            other => panic!("Expected ConstantSpeed driver, got {:?}", other),
        }

        // Verify the loaded mechanism has the same driver count and can build.
        let loaded = load_mechanism(&json_str).unwrap();
        assert_eq!(loaded.n_drivers(), 1);

        // The loaded driver should produce the same f(t) as the original.
        // We verify indirectly: solve at t=0 with the original q0 and check
        // that the driver constraint is satisfied.
        use crate::solver::kinematics::solve_position;
        let state = loaded.state();
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, theta_0);
        state.set_pose("coupler", &mut q0, theta_0.cos(), theta_0.sin(), 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);
        let result = solve_position(&loaded, &q0, 0.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Loaded mechanism with non-default omega/theta_0 did not converge: residual={}",
            result.residual_norm,
        );
    }

    #[test]
    fn save_to_file_and_load_from_file_roundtrip() {
        // Write to a temp file and reload to verify the file I/O path.
        use crate::gui::AppState;
        use crate::gui::samples::SampleMechanism;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let path = std::env::temp_dir().join("linkage_test_roundtrip.json");
        state.save_to_file(&path).expect("save_to_file failed");

        let mut state2 = AppState::default();
        state2.load_from_file(&path).expect("load_from_file failed");

        // The loaded mechanism should have the same structure.
        let mech1 = state.mechanism.as_ref().unwrap();
        let mech2 = state2.mechanism.as_ref().unwrap();

        assert_eq!(mech2.bodies().len(), mech1.bodies().len());
        assert_eq!(mech2.joints().len(), mech1.joints().len());
        assert_eq!(mech2.n_drivers(), mech1.n_drivers());

        // Clean up.
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn empty_mechanism_round_trip() {
        // A mechanism with just ground and no joints
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.build().unwrap();

        let json_str = save_mechanism(&mech).unwrap();
        let loaded = load_mechanism(&json_str).unwrap();

        assert_eq!(loaded.bodies().len(), 1);
        assert_eq!(loaded.joints().len(), 0);
        assert!(loaded.is_built());
    }

    #[test]
    fn invalid_json_returns_error() {
        let result = load_mechanism("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn missing_fields_returns_error() {
        let json = r#"{"schema_version": "1.0.0"}"#;
        let result = load_mechanism(json);
        assert!(result.is_err());
    }

    #[test]
    fn ground_body_has_zero_mass_and_inertia_on_load() {
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.build().unwrap();

        let json_str = save_mechanism(&mech).unwrap();
        let loaded = load_mechanism(&json_str).unwrap();

        let ground = loaded.bodies().get(GROUND_ID).unwrap();
        assert_abs_diff_eq!(ground.mass, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(ground.izz_cg, 0.0, epsilon = 1e-15);
    }

    // -----------------------------------------------------------------------
    // load_mechanism_unbuilt_from_json
    // -----------------------------------------------------------------------

    #[test]
    fn load_unbuilt_from_json_matches_string_path() {
        let mech = build_fourbar_no_driver();
        let json_str = save_mechanism(&mech).unwrap();
        let json_struct: MechanismJson = serde_json::from_str(&json_str).unwrap();

        // Build via string path
        let mut from_str = load_mechanism_unbuilt(&json_str).unwrap();
        from_str.build().unwrap();

        // Build via struct path
        let mut from_json = load_mechanism_unbuilt_from_json(&json_struct).unwrap();
        from_json.build().unwrap();

        assert_eq!(from_str.bodies().len(), from_json.bodies().len());
        assert_eq!(from_str.joints().len(), from_json.joints().len());
        assert_eq!(from_str.state().n_coords(), from_json.state().n_coords());
    }

    // -----------------------------------------------------------------------
    // mount_points serialization
    // -----------------------------------------------------------------------

    #[test]
    fn round_trip_preserves_mount_points() {
        let json_str = r#"{
            "schema_version": "1.0.0",
            "bodies": {
                "ground": {
                    "attachment_points": {"O2": [0.0, 0.0], "O4": [0.038, 0.0]},
                    "mount_points": {"M1": [0.02, 0.01]},
                    "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0
                },
                "crank": {
                    "attachment_points": {"O2": [0.0, 0.0], "A": [0.015, 0.0]},
                    "mass": 0.5, "cg_local": [0.0075, 0.0], "izz_cg": 0.0001
                }
            },
            "joints": {
                "J1": {"type": "revolute", "body_i": "ground", "body_j": "crank", "point_i": "O2", "point_j": "O2"}
            },
            "drivers": {
                "D1": {"type": "constant_speed", "body_i": "ground", "body_j": "crank", "omega": 1.0, "theta_0": 0.0}
            }
        }"#;
        let loaded = load_mechanism_unbuilt(json_str).unwrap();
        let ground = loaded.bodies().get("ground").unwrap();
        assert!(ground.mount_points.contains_key("M1"));
        assert_abs_diff_eq!(ground.mount_points["M1"].x, 0.02, epsilon = 1e-15);
        assert_abs_diff_eq!(ground.mount_points["M1"].y, 0.01, epsilon = 1e-15);

        // Round-trip: build first (mechanism_to_json requires built mechanism)
        let mut built = loaded;
        built.build().unwrap();
        let saved = mechanism_to_json(&built).unwrap();
        let reloaded = load_mechanism_unbuilt_from_json(&saved).unwrap();
        let ground2 = reloaded.bodies().get("ground").unwrap();
        assert!(ground2.mount_points.contains_key("M1"));
        assert_abs_diff_eq!(ground2.mount_points["M1"].x, 0.02, epsilon = 1e-15);
    }

    #[test]
    fn old_json_without_mount_points_loads_with_empty() {
        let json_str = r#"{
            "schema_version": "1.0.0",
            "bodies": {
                "ground": {
                    "attachment_points": {"O2": [0.0, 0.0]},
                    "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0
                }
            },
            "joints": {}
        }"#;
        let loaded = load_mechanism_unbuilt(json_str).unwrap();
        let ground = loaded.bodies().get("ground").unwrap();
        assert!(ground.mount_points.is_empty());
    }

    // -----------------------------------------------------------------------
    // Force element serialization
    // -----------------------------------------------------------------------

    #[test]
    fn force_elements_round_trip() {
        use crate::forces::elements::*;

        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let rocker = make_bar("rocker", "C", "D", 2.0, 2.0, 0.02);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "ground", "O4", "rocker", "C").unwrap();

        // Add various force elements
        mech.add_force(ForceElement::Gravity(GravityElement::default()));
        mech.add_force(ForceElement::LinearSpring(LinearSpringElement {
            body_a: "crank".into(),
            point_a: [0.5, 0.0],
            point_a_name: None,
            body_b: "rocker".into(),
            point_b: [1.0, 0.0],
            point_b_name: None,
            stiffness: 500.0,
            free_length: 0.3,
        }));
        mech.add_force(ForceElement::ExternalForce(ExternalForceElement {
            body_id: "crank".into(),
            local_point: [0.5, 0.0],
            local_point_name: None,
            force: [10.0, -5.0],
            modulation: TimeModulation::Constant,
        }));

        mech.build().unwrap();

        let json_str = save_mechanism(&mech).unwrap();
        let loaded = load_mechanism(&json_str).unwrap();

        assert_eq!(loaded.forces().len(), 3);
        assert_eq!(loaded.forces()[0].type_name(), "Gravity");
        assert_eq!(loaded.forces()[1].type_name(), "Linear Spring");
        assert_eq!(loaded.forces()[2].type_name(), "External Force");

        // Verify spring parameters survived
        match &loaded.forces()[1] {
            ForceElement::LinearSpring(s) => {
                assert_abs_diff_eq!(s.stiffness, 500.0, epsilon = 1e-15);
                assert_abs_diff_eq!(s.free_length, 0.3, epsilon = 1e-15);
                assert_eq!(s.body_a, "crank");
                assert_eq!(s.body_b, "rocker");
            }
            _ => panic!("Expected LinearSpring"),
        }
    }

    #[test]
    fn old_json_without_forces_loads_cleanly() {
        // Simulate an old-format JSON file that doesn't have the "forces" field
        let json = r#"{
            "schema_version": "1.0.0",
            "bodies": {
                "ground": {
                    "attachment_points": {"O": [0.0, 0.0]},
                    "mass": 0.0,
                    "cg_local": [0.0, 0.0],
                    "izz_cg": 0.0
                }
            },
            "joints": {}
        }"#;

        let loaded = load_mechanism(json).unwrap();
        assert!(loaded.forces().is_empty());
        assert!(loaded.is_built());
    }

    #[test]
    fn force_elements_appear_in_json_output() {
        use crate::forces::elements::*;

        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let bar = make_bar("bar", "A", "B", 1.0, 2.0, 0.01);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(bar).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "bar", "A").unwrap();
        mech.add_force(ForceElement::ExternalTorque(ExternalTorqueElement {
            body_id: "bar".into(),
            torque: 7.5,
            modulation: TimeModulation::Constant,
        }));
        mech.build().unwrap();

        let json_str = save_mechanism(&mech).unwrap();
        assert!(json_str.contains("ExternalTorque"), "JSON should contain ExternalTorque, got: {}", json_str);
        assert!(json_str.contains("7.5"), "JSON should contain torque value 7.5");
    }

    // -----------------------------------------------------------------------
    // Expression driver serialization
    // -----------------------------------------------------------------------

    #[test]
    fn expression_driver_round_trips_expressions() {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let coupler = make_bar("coupler", "B", "C", 3.0, 1.5, 0.05);
        let rocker = make_bar("rocker", "D", "C", 2.0, 1.5, 0.02);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D").unwrap();
        mech.add_expression_driver(
            "D1",
            "ground",
            "crank",
            "2*pi*t",
            "2*pi",
            "0",
        )
        .unwrap();
        mech.build().unwrap();

        // Serialize
        let json_str = save_mechanism(&mech).unwrap();
        let json_struct: MechanismJson = serde_json::from_str(&json_str).unwrap();

        // Verify the expression driver is in JSON
        assert_eq!(json_struct.drivers.len(), 1);
        match &json_struct.drivers["D1"] {
            DriverJson::Expression {
                body_i,
                body_j,
                expr,
                expr_dot,
                expr_ddot,
            } => {
                assert_eq!(body_i, "ground");
                assert_eq!(body_j, "crank");
                assert_eq!(expr, "2*pi*t");
                assert_eq!(expr_dot, "2*pi");
                assert_eq!(expr_ddot, "0");
            }
            other => panic!("Expected Expression driver, got {:?}", other),
        }

        // Deserialize and verify the mechanism builds and solves.
        // Use t=0.1 (crank angle ~36 deg) to avoid the degenerate t=0 config.
        let loaded = load_mechanism(&json_str).unwrap();
        assert_eq!(loaded.n_drivers(), 1);

        let state = loaded.state();
        let angle = 2.0 * PI * 0.1; // f(0.1) = 0.2*pi
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("coupler", &mut q0, angle.cos(), angle.sin(), 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);
        let result = solve_position(&loaded, &q0, 0.1, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Loaded expression driver mechanism did not converge: residual={}",
            result.residual_norm,
        );
    }

    #[test]
    fn expression_driver_json_contains_type_tag() {
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_revolute_joint("J1", "ground", "O", "crank", "A").unwrap();
        mech.add_expression_driver("D1", "ground", "crank", "t", "1", "0")
            .unwrap();
        mech.build().unwrap();

        let json_str = save_mechanism(&mech).unwrap();
        assert!(
            json_str.contains("\"expression\""),
            "JSON should contain expression type tag, got: {}",
            json_str,
        );
    }

    #[test]
    fn expression_driver_invalid_expr_fails_to_load() {
        // Construct JSON with an invalid expression string
        let json = r#"{
            "schema_version": "1.0.0",
            "bodies": {
                "ground": {
                    "attachment_points": {"O": [0.0, 0.0]},
                    "mass": 0.0,
                    "cg_local": [0.0, 0.0],
                    "izz_cg": 0.0
                },
                "crank": {
                    "attachment_points": {"A": [0.0, 0.0], "B": [1.0, 0.0]},
                    "mass": 2.0,
                    "cg_local": [0.5, 0.0],
                    "izz_cg": 0.01
                }
            },
            "joints": {
                "J1": {
                    "type": "revolute",
                    "body_i": "ground",
                    "body_j": "crank",
                    "point_i": "O",
                    "point_j": "A"
                }
            },
            "drivers": {
                "D1": {
                    "type": "expression",
                    "body_i": "ground",
                    "body_j": "crank",
                    "expr": "???invalid",
                    "expr_dot": "1",
                    "expr_ddot": "0"
                }
            }
        }"#;

        let result = load_mechanism(json);
        assert!(result.is_err(), "Expected error for invalid expression driver");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Invalid") || err.contains("invalid") || err.contains("Failed"),
            "Error should mention parse failure: {}",
            err,
        );
    }

    #[test]
    fn expression_driver_with_sin_cos_solves() {
        // Expression with trigonometric functions -- verifies meval supports them.
        // f(t) = pi/4 * sin(t), so at t=1 the crank angle is ~0.675 rad.
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 4.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 1.0, 2.0, 0.01);
        let coupler = make_bar("coupler", "B", "C", 3.0, 1.5, 0.05);
        let rocker = make_bar("rocker", "D", "C", 2.0, 1.5, 0.02);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D").unwrap();
        mech.add_expression_driver(
            "D1",
            "ground",
            "crank",
            "pi/4 * sin(t)",
            "pi/4 * cos(t)",
            "-pi/4 * sin(t)",
        )
        .unwrap();
        mech.build().unwrap();

        // Solve at t=1 where f(1) = pi/4 * sin(1) ~ 0.675 rad
        let state = mech.state();
        let angle = PI / 4.0 * 1.0_f64.sin();
        let mut q0 = state.make_q();
        state.set_pose("crank", &mut q0, 0.0, 0.0, angle);
        state.set_pose("coupler", &mut q0, angle.cos(), angle.sin(), 0.0);
        state.set_pose("rocker", &mut q0, 4.0, 0.0, PI / 2.0);
        let result = solve_position(&mech, &q0, 1.0, 1e-10, 50).unwrap();
        assert!(
            result.converged,
            "Expression driver with sin/cos did not converge: residual={}",
            result.residual_norm,
        );
    }
}
