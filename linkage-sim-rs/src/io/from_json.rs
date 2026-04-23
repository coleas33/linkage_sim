//! JSON -> Mechanism conversion.

use std::collections::HashMap;

use nalgebra::Vector2;

use crate::core::body::Body;
use crate::core::driver::clamp_driver_omega;
use crate::core::linear_driver::constant_velocity_linear_driver;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::forces::compound::{analyze_force, expand_compound_force, CompoundAnalysis};
use crate::forces::elements::ForceElement;

use super::error::SerializationError;
use super::schema::*;

// ---------------------------------------------------------------------------
// Compound force helpers
// ---------------------------------------------------------------------------

/// Extract the local-frame coordinates of point A or B from a two-point force element.
///
/// Returns `[0.0, 0.0]` for force variants that don't carry explicit point coordinates.
fn get_force_point_pos(force: &ForceElement, is_a: bool) -> [f64; 2] {
    match force {
        ForceElement::LinearSpring(s) => if is_a { s.point_a } else { s.point_b },
        ForceElement::LinearDamper(d) => if is_a { d.point_a } else { d.point_b },
        ForceElement::GasSpring(g) => if is_a { g.point_a } else { g.point_b },
        ForceElement::LinearActuator(a) => if is_a { a.point_a } else { a.point_b },
        _ => [0.0, 0.0],
    }
}

/// Extract the body-A and body-B IDs from a two-point force element.
///
/// Returns `None` for force variants that don't reference two specific bodies.
fn get_force_body_ids(force: &ForceElement) -> Option<(String, String)> {
    match force {
        ForceElement::LinearSpring(s) => Some((s.body_a.clone(), s.body_b.clone())),
        ForceElement::LinearDamper(d) => Some((d.body_a.clone(), d.body_b.clone())),
        ForceElement::GasSpring(g) => Some((g.body_a.clone(), g.body_b.clone())),
        ForceElement::LinearActuator(a) => Some((a.body_a.clone(), a.body_b.clone())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// JSON -> Mechanism
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
            label: body_json.label.clone().unwrap_or_else(|| body_id.clone()),
            geometry: body_json.geometry.clone(),
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
                ..
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
                ..
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
                ..
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
                ..
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
                // Legacy format: drivers in the joints map. Skip silently --
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
                // Clamp omega to the minimum magnitude so legacy files
                // saved with omega=0 (or hand-edited JSON) still produce
                // animatable mechanisms.
                let safe_omega = clamp_driver_omega(*omega);
                mech.add_constant_speed_driver(driver_id, body_i, body_j, safe_omega, *theta_0)
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

    // Rebuild linear drivers
    for ld_json in &json_struct.linear_drivers {
        let driver = constant_velocity_linear_driver(
            &ld_json.id,
            &ld_json.body_a,
            ld_json.point_a,
            &ld_json.body_b,
            ld_json.point_b,
            ld_json.velocity,
            ld_json.length_0,
        );
        mech.add_linear_driver(driver)
            .map_err(|e| SerializationError::Build(e.to_string()))?;
    }

    // Restore force elements, resolving any named mount/attachment points
    // against the bodies that were just added to the mechanism.
    // Forces that reference mount points are expanded into compound bodies
    // (cylinder + rod + prismatic joint) so the solver can handle them.
    let bodies_snapshot = mech.bodies().clone();
    let resolved_forces: Vec<ForceElement> = json_struct.forces
        .iter()
        .map(|f| f.resolve_named_points(&bodies_snapshot).unwrap_or_else(|e| {
            log::warn!("Failed to resolve force point name: {e}");
            f.clone()
        }))
        .collect();

    for (i, force) in resolved_forces.iter().enumerate() {
        match analyze_force(force, &json_struct.bodies) {
            CompoundAnalysis::PureForce(f) => {
                mech.add_force(f);
            }
            CompoundAnalysis::NeedsExpansion { force: f, mount_a, mount_b } => {
                let point_a_pos = get_force_point_pos(&f, true);
                let point_b_pos = get_force_point_pos(&f, false);

                // Promote mount points to synthetic attachment points so that
                // `add_revolute_joint` can find them on the original bodies.
                if let Some((body_a_id, body_b_id)) = get_force_body_ids(&f) {
                    if mount_a {
                        let synthetic = format!("_force_{}_mount_a", i);
                        if let Some(body) = mech.bodies_mut().get_mut(&body_a_id) {
                            let _ = body.add_attachment_point(&synthetic, point_a_pos[0], point_a_pos[1]);
                        }
                    }
                    if mount_b {
                        let synthetic = format!("_force_{}_mount_b", i);
                        if let Some(body) = mech.bodies_mut().get_mut(&body_b_id) {
                            let _ = body.add_attachment_point(&synthetic, point_b_pos[0], point_b_pos[1]);
                        }
                    }
                }

                // Expand into compound bodies + joints + replacement force.
                match expand_compound_force(&mut mech, &f, i, mount_a, mount_b, point_a_pos, point_b_pos) {
                    Ok(replacement) => mech.add_force(replacement),
                    Err(e) => {
                        log::warn!("Failed to expand compound force {i}: {e}");
                        mech.add_force(f); // fallback: add original force as-is
                    }
                }
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
