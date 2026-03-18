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
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;

/// Current schema version for the JSON format.
pub const SCHEMA_VERSION: &str = "1.0.0";

// ---------------------------------------------------------------------------
// JSON schema types
// ---------------------------------------------------------------------------

/// Top-level JSON representation of a mechanism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanismJson {
    pub schema_version: String,
    pub bodies: HashMap<String, BodyJson>,
    pub joints: HashMap<String, JointJson>,
}

/// JSON representation of a rigid body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyJson {
    pub attachment_points: HashMap<String, [f64; 2]>,
    pub mass: f64,
    pub cg_local: [f64; 2],
    pub izz_cg: f64,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub coupler_points: HashMap<String, [f64; 2]>,
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
    #[error("Unsupported schema version '{found}'. Expected '{expected}'.")]
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
        coupler_points,
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
            let body_i = &bodies[body_i_id];
            let body_j = &bodies[body_j_id];
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
            let body_i = &bodies[body_i_id];
            let body_j = &bodies[body_j_id];
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
            let body_i = &bodies[body_i_id];
            let body_j = &bodies[body_j_id];
            Ok(JointJson::Prismatic {
                body_i: body_i_id.to_string(),
                body_j: body_j_id.to_string(),
                point_i: find_point_name(body_i, j.point_i_local(), body_i_id)?,
                point_j: find_point_name(body_j, j.point_j_local(), body_j_id)?,
                axis_local_i: [j.axis_local_i().x, j.axis_local_i().y],
                delta_theta_0: j.delta_theta_0(),
            })
        }
    }
}

/// Convert a `Mechanism` to its JSON-compatible struct.
///
/// Driver constraints are **not** included — closures cannot be serialized.
/// They must be re-attached after deserialization.
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

    Ok(MechanismJson {
        schema_version: SCHEMA_VERSION.to_string(),
        bodies,
        joints,
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

    if json_struct.schema_version != SCHEMA_VERSION {
        return Err(SerializationError::UnsupportedVersion {
            found: json_struct.schema_version,
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
            coupler_points,
        };
        mech.add_body(body)
            .map_err(|e| SerializationError::Build(e.to_string()))?;
    }

    // Rebuild joints (geometric constraints only; drivers are skipped)
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
            JointJson::RevoluteDriver { .. } => {
                // Drivers cannot be deserialized (closures). Skip silently.
            }
        }
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
        // appears as a marker in the JSON.
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

        // Save — drivers are NOT included in joints (only geometric joints are)
        let json_str = save_mechanism(&mech).unwrap();
        let json_struct: MechanismJson = serde_json::from_str(&json_str).unwrap();

        // Only the revolute joint should be in the output
        assert_eq!(json_struct.joints.len(), 1);
        assert!(json_struct.joints.contains_key("J1"));
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
}
