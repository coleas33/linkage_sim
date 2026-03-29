//! Mechanism -> JSON conversion.
//!
//! **Known limitation:** Driver constraints use closures, which cannot be
//! serialized. Drivers are skipped on save and must be re-attached after load.

use std::collections::HashMap;

use nalgebra::Vector2;

use crate::core::body::Body;
use crate::core::constraint::{Constraint, JointConstraint};
use crate::core::driver::DriverMeta;
use crate::core::mechanism::Mechanism;

use super::error::SerializationError;
use super::schema::*;

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
        // Point masses are a blueprint-level concept -- they modify mass/CG/Izz
        // at build time. When exporting from a built mechanism, the composite
        // properties are already baked in, so we emit an empty list.
        point_masses: Vec::new(),
        label: Some(body.label.clone()),
        geometry: body.geometry.clone(),
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
                label: None,
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
                label: None,
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
                label: None,
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
                label: None,
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
        sweep_config: None,
        mounting_angle: 0.0,
    })
}

/// Serialize a `Mechanism` to a JSON string.
///
/// Driver constraints are skipped -- they must be re-attached after loading.
pub fn save_mechanism(mech: &Mechanism) -> Result<String, SerializationError> {
    let json_struct = mechanism_to_json(mech)?;
    let json_str = serde_json::to_string_pretty(&json_struct)?;
    Ok(json_str)
}
