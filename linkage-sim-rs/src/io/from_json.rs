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
// Point-mass validation
// ---------------------------------------------------------------------------

/// Why the loader does not apply point mass `pm` on body `body_id`, or `None`
/// when it is applied.
///
/// A weight on ground is rejected: ground is fixed and massless, so the weight
/// could never load the mechanism and accepting it would hide a modelling
/// mistake. A mass that is not a positive finite number, or a non-finite
/// position, would corrupt the composite mass/CG/Izz (`Body::add_point_mass`
/// subtracts a negative mass). Shared by the loader, [`point_mass_warnings`],
/// the GUI editing API and anything that lists the weights the physics sees.
pub fn point_mass_skip_reason(body_id: &str, pm: &PointMassJson) -> Option<&'static str> {
    if body_id == GROUND_ID {
        Some("point masses on ground are not supported")
    } else if !(pm.mass.is_finite() && pm.mass > 0.0) {
        Some("mass must be a positive, finite number of kg")
    } else if !(pm.local_pos[0].is_finite() && pm.local_pos[1].is_finite()) {
        Some("position must be finite")
    } else {
        None
    }
}

/// One human-readable warning per point mass the loader skips (see
/// [`point_mass_skip_reason`]): bodies sorted by id, masses in list order.
/// A weight without an id is named by its 1-based list position (`#2`).
/// Empty when every point mass is applied.
pub fn point_mass_warnings(json: &MechanismJson) -> Vec<String> {
    let mut body_ids: Vec<&String> = json.bodies.keys().collect();
    body_ids.sort();
    let mut warnings = Vec::new();
    for body_id in body_ids {
        for (i, pm) in json.bodies[body_id].point_masses.iter().enumerate() {
            let Some(reason) = point_mass_skip_reason(body_id, pm) else { continue };
            let name = if is_blank_point_mass_id(&pm.id) {
                format!("#{}", i + 1)
            } else {
                format!("'{}'", pm.id)
            };
            warnings.push(format!(
                "Point mass {name} on body '{body_id}' skipped: {reason} (mass {} kg)",
                pm.mass
            ));
        }
    }
    warnings
}

/// Fold `point_masses` into `body`'s composite mass, CG and Izz, skipping the
/// ones [`point_mass_skip_reason`] rejects (named `body.id`).
///
/// The single place blueprint weights reach the physics: the loader and the
/// GUI's no-rebuild mass sync (`AppState::sync_live_mass_props`) both call it,
/// so a live body always equals a fresh build of the blueprint.
pub(crate) fn apply_point_masses(body: &mut Body, point_masses: &[PointMassJson]) {
    for pm in point_masses {
        if let Some(reason) = point_mass_skip_reason(&body.id, pm) {
            log::warn!("Skipping point mass '{}' on body '{}': {}", pm.id, body.id, reason);
            continue;
        }
        body.add_point_mass(pm.mass, Vector2::new(pm.local_pos[0], pm.local_pos[1]));
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

        let mut body = Body {
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
        // Apply point masses to update composite mass/CG/Izz, skipping the
        // ones the loader rejects (listed for the user by `point_mass_warnings`).
        apply_point_masses(&mut body, &body_json.point_masses);
        mech.add_body(body)
            .map_err(|e| SerializationError::Build(e.to_string()))?;
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

#[cfg(test)]
mod point_mass_validation_tests {
    use super::*;

    /// Ground plus one 2 kg bar (CG at x = 0.5 m, Izz 0.1). No joints are
    /// needed to check composite mass properties.
    fn ground_and_bar() -> MechanismJson {
        serde_json::from_str(
            r#"{
                "schema_version": "1.1.0",
                "bodies": {
                    "ground": {"attachment_points": {"O": [0.0, 0.0]},
                               "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0},
                    "bar": {"attachment_points": {"A": [0.0, 0.0], "B": [1.0, 0.0]},
                            "mass": 2.0, "cg_local": [0.5, 0.0], "izz_cg": 0.1}
                },
                "joints": {}
            }"#,
        )
        .unwrap()
    }

    fn pm(id: &str, mass: f64, x: f64, y: f64) -> PointMassJson {
        PointMassJson { id: id.to_string(), label: None, mass, local_pos: [x, y] }
    }

    /// (mass, cg, izz) of body `id` after loading `json`.
    fn mass_props(json: &MechanismJson, id: &str) -> (f64, [f64; 2], f64) {
        let mech = load_mechanism_unbuilt_from_json(json).unwrap();
        let b = &mech.bodies()[id];
        (b.mass, [b.cg_local.x, b.cg_local.y], b.izz_cg)
    }

    #[test]
    fn valid_point_mass_is_applied_and_not_reported() {
        let mut json = ground_and_bar();
        json.bodies.get_mut("bar").unwrap().point_masses.push(pm("W1", 2.0, 1.0, 0.0));
        let (m, cg, izz) = mass_props(&json, "bar");
        assert!((m - 4.0).abs() < 1e-12, "mass {m}");
        assert!((cg[0] - 0.75).abs() < 1e-12 && cg[1].abs() < 1e-12, "cg {cg:?}");
        // 0.1 + 2 kg * (0.25 m)^2 + 2 kg * (0.25 m)^2
        assert!((izz - 0.35).abs() < 1e-12, "izz {izz}");
        assert!(point_mass_warnings(&json).is_empty());
    }

    #[test]
    fn point_mass_on_ground_is_skipped_and_reported() {
        let mut json = ground_and_bar();
        json.bodies.get_mut("ground").unwrap().point_masses.push(pm("W1", 5.0, 0.3, 0.0));
        let (m, _, izz) = mass_props(&json, "ground");
        assert_eq!(m, 0.0, "ground must stay massless");
        assert_eq!(izz, 0.0);
        let w = point_mass_warnings(&json);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("'W1'") && w[0].contains("'ground'"), "{w:?}");
    }

    #[test]
    fn non_positive_or_non_finite_point_masses_are_skipped_and_reported() {
        let base = mass_props(&ground_and_bar(), "bar");
        let bad = [
            pm("neg", -1.0, 1.0, 0.0), // used to be subtracted from the bar
            pm("zero", 0.0, 1.0, 0.0),
            pm("nan", f64::NAN, 1.0, 0.0),
            pm("inf", f64::INFINITY, 1.0, 0.0),
            pm("far", 1.0, f64::INFINITY, 0.0),
            pm("nanpos", 1.0, 0.0, f64::NAN),
        ];
        for p in &bad {
            let mut json = ground_and_bar();
            json.bodies.get_mut("bar").unwrap().point_masses.push(p.clone());
            assert_eq!(mass_props(&json, "bar"), base, "{}: must not change the bar", p.id);
            let w = point_mass_warnings(&json);
            assert_eq!(w.len(), 1, "{}: {w:?}", p.id);
            assert!(w[0].contains(&format!("'{}'", p.id)) && w[0].contains("'bar'"), "{w:?}");
        }
    }

    #[test]
    fn invalid_point_mass_does_not_block_valid_ones_on_the_same_body() {
        let mut json = ground_and_bar();
        json.bodies
            .get_mut("bar")
            .unwrap()
            .point_masses
            .extend([pm("bad", -3.0, 1.0, 0.0), pm("good", 2.0, 1.0, 0.0)]);
        let (m, cg, _) = mass_props(&json, "bar");
        assert!((m - 4.0).abs() < 1e-12 && (cg[0] - 0.75).abs() < 1e-12, "{m} {cg:?}");
    }

    #[test]
    fn warnings_are_sorted_by_body_and_name_blank_ids_by_list_position() {
        let mut json = ground_and_bar();
        json.bodies
            .get_mut("bar")
            .unwrap()
            .point_masses
            .extend([pm("ok", 1.0, 0.0, 0.0), pm("", -2.0, 0.0, 0.0)]);
        json.bodies.get_mut("ground").unwrap().point_masses.push(pm("G", 1.0, 0.0, 0.0));
        let w = point_mass_warnings(&json);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].contains("#2") && w[0].contains("'bar'"), "{w:?}");
        assert!(w[1].contains("'G'") && w[1].contains("'ground'"), "{w:?}");
    }
}
