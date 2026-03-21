//! Compound force expansion — auto-creates bodies and joints for
//! mount-point-referenced force elements.
//!
//! When a force element references a `mount_point` on a body (rather than an
//! `attachment_point`), it cannot be directly wired up as a simple two-body
//! force.  Instead we inject a pair of massless "compound" bodies — a cylinder
//! and a rod — that model the physical housing of the force element (e.g. a
//! shock absorber or gas spring).  The cylinder is pinned at end A of the
//! force and the rod is pinned at end B; a prismatic joint allows the rod to
//! slide inside the cylinder.  The force element is then remapped to act
//! between the slide points on those compound bodies.

use std::collections::HashMap;

use nalgebra::Vector2;

use crate::core::body::Body;
use crate::core::mechanism::Mechanism;
use crate::forces::elements::ForceElement;
use crate::io::serialization::BodyJson;

// ── Analysis ─────────────────────────────────────────────────────────────────

/// Result of analyzing a force element for compound expansion.
pub enum CompoundAnalysis {
    /// Force does not reference mount points — add as-is.
    PureForce(ForceElement),
    /// Force references mount points — needs compound expansion.
    NeedsExpansion {
        force: ForceElement,
        /// Whether point A is a mount point on its body.
        mount_a: bool,
        /// Whether point B is a mount point on its body.
        mount_b: bool,
    },
}

/// Analyze a force element to determine whether it needs compound expansion.
///
/// Inspects the optional `point_a_name` / `point_b_name` fields of the force
/// and checks whether those names resolve to `mount_points` (not
/// `attachment_points`) on the referenced bodies.
///
/// Returns [`CompoundAnalysis::NeedsExpansion`] if at least one endpoint is a
/// mount point; otherwise returns [`CompoundAnalysis::PureForce`].
///
/// Force variants that carry no named point references (e.g. `Gravity`,
/// `ExternalForce`, `TorsionSpring`) always return `PureForce`.
pub fn analyze_force(
    force: &ForceElement,
    bodies_json: &HashMap<String, BodyJson>,
) -> CompoundAnalysis {
    // Extract the four identifiers we care about.  Non-point forces bail early.
    let (body_a_id, point_a_name, body_b_id, point_b_name) = match force {
        ForceElement::LinearSpring(s) => (
            s.body_a.as_str(),
            s.point_a_name.as_deref(),
            s.body_b.as_str(),
            s.point_b_name.as_deref(),
        ),
        ForceElement::LinearDamper(d) => (
            d.body_a.as_str(),
            d.point_a_name.as_deref(),
            d.body_b.as_str(),
            d.point_b_name.as_deref(),
        ),
        ForceElement::GasSpring(g) => (
            g.body_a.as_str(),
            g.point_a_name.as_deref(),
            g.body_b.as_str(),
            g.point_b_name.as_deref(),
        ),
        ForceElement::LinearActuator(a) => (
            a.body_a.as_str(),
            a.point_a_name.as_deref(),
            a.body_b.as_str(),
            a.point_b_name.as_deref(),
        ),
        // All other variants (Gravity, TorsionSpring, RotaryDamper,
        // ExternalForce, ExternalTorque, …) are pure — no named point refs.
        other => return CompoundAnalysis::PureForce(other.clone()),
    };

    let mount_a = is_mount_point(body_a_id, point_a_name, bodies_json);
    let mount_b = is_mount_point(body_b_id, point_b_name, bodies_json);

    if mount_a || mount_b {
        CompoundAnalysis::NeedsExpansion {
            force: force.clone(),
            mount_a,
            mount_b,
        }
    } else {
        CompoundAnalysis::PureForce(force.clone())
    }
}

/// Returns `true` iff `point_name` is `Some(name)` and `name` is found in the
/// `mount_points` map of `body_id` (not in `attachment_points`).
fn is_mount_point(
    body_id: &str,
    point_name: Option<&str>,
    bodies_json: &HashMap<String, BodyJson>,
) -> bool {
    let Some(name) = point_name else {
        return false;
    };
    let Some(body) = bodies_json.get(body_id) else {
        return false;
    };
    body.mount_points.contains_key(name)
}

// ── Compound body helpers ─────────────────────────────────────────────────────

/// Create the "cylinder" half of a compound force element.
///
/// The cylinder body has:
/// - `"base"` at the origin — this is where it is pinned to body A.
/// - `"slide"` at `(half_len, 0)` — the inner end where the rod slides.
pub fn create_compound_cylinder(force_index: usize, half_len: f64) -> Body {
    let id = format!("force_{}_cyl", force_index);
    let mut body = Body::new(&id);
    body.add_attachment_point("base", 0.0, 0.0)
        .expect("compound cylinder: 'base' point already exists — this is a bug");
    body.add_attachment_point("slide", half_len, 0.0)
        .expect("compound cylinder: 'slide' point already exists — this is a bug");
    body
}

/// Create the "rod" half of a compound force element.
///
/// The rod body has:
/// - `"slide"` at the origin — the guided end inside the cylinder.
/// - `"tip"` at `(half_len, 0)` — where it is pinned to body B.
pub fn create_compound_rod(force_index: usize, half_len: f64) -> Body {
    let id = format!("force_{}_rod", force_index);
    let mut body = Body::new(&id);
    body.add_attachment_point("slide", 0.0, 0.0)
        .expect("compound rod: 'slide' point already exists — this is a bug");
    body.add_attachment_point("tip", half_len, 0.0)
        .expect("compound rod: 'tip' point already exists — this is a bug");
    body
}

// ── Expansion ─────────────────────────────────────────────────────────────────

/// Expand a compound force element into bodies, joints, and a remapped force.
///
/// Given the world positions of the two force endpoints, this function:
/// 1. Computes the initial length and splits it equally into `half_len`.
/// 2. Creates and registers a cylinder body and a rod body.
/// 3. Adds three joints:
///    - Revolute at end A (cylinder `"base"` ↔ body A).
///    - Revolute at end B (rod `"tip"` ↔ body B).
///    - Prismatic between cylinder `"slide"` and rod `"slide"` along x.
/// 4. Returns a remapped [`ForceElement`] that acts between the two `"slide"`
///    points on the compound bodies instead of the original bodies.
///
/// The caller is responsible for ensuring that the named attachment points
/// referenced by `mount_a` / `mount_b` exist on the target bodies before
/// calling this function (Task 4 adds them from `mount_points` during the
/// serialization load path).
pub fn expand_compound_force(
    mech: &mut Mechanism,
    force: &ForceElement,
    force_index: usize,
    mount_a: bool,
    mount_b: bool,
    point_a_pos: [f64; 2],
    point_b_pos: [f64; 2],
) -> Result<ForceElement, Box<dyn std::error::Error>> {
    let dx = point_b_pos[0] - point_a_pos[0];
    let dy = point_b_pos[1] - point_a_pos[1];
    let initial_length = (dx * dx + dy * dy).sqrt();
    let half_len = initial_length / 2.0;

    // Create and register compound bodies.
    mech.add_body(create_compound_cylinder(force_index, half_len))?;
    mech.add_body(create_compound_rod(force_index, half_len))?;

    let cyl_id = format!("force_{}_cyl", force_index);
    let rod_id = format!("force_{}_rod", force_index);

    // Extract body IDs and point names from the original force.
    let (body_a_id, point_a_name, body_b_id, point_b_name) = match force {
        ForceElement::LinearSpring(s) => (
            s.body_a.clone(),
            s.point_a_name.clone(),
            s.body_b.clone(),
            s.point_b_name.clone(),
        ),
        ForceElement::LinearDamper(d) => (
            d.body_a.clone(),
            d.point_a_name.clone(),
            d.body_b.clone(),
            d.point_b_name.clone(),
        ),
        ForceElement::GasSpring(g) => (
            g.body_a.clone(),
            g.point_a_name.clone(),
            g.body_b.clone(),
            g.point_b_name.clone(),
        ),
        ForceElement::LinearActuator(a) => (
            a.body_a.clone(),
            a.point_a_name.clone(),
            a.body_b.clone(),
            a.point_b_name.clone(),
        ),
        _ => return Err("Cannot expand non-point force element".into()),
    };

    // Revolute joint at end A: body_a ↔ cylinder "base".
    // If mount_a is true the attachment point on body_a was pre-created from
    // the mount_point under the synthetic name `_force_{idx}_mount_a` (Task 4).
    let pt_a_ref = if mount_a {
        format!("_force_{}_mount_a", force_index)
    } else {
        point_a_name.unwrap_or_default()
    };
    mech.add_revolute_joint(
        &format!("force_{}_base", force_index),
        &body_a_id,
        &pt_a_ref,
        &cyl_id,
        "base",
    )?;

    // Revolute joint at end B: body_b ↔ rod "tip".
    let pt_b_ref = if mount_b {
        format!("_force_{}_mount_b", force_index)
    } else {
        point_b_name.unwrap_or_default()
    };
    mech.add_revolute_joint(
        &format!("force_{}_tip", force_index),
        &body_b_id,
        &pt_b_ref,
        &rod_id,
        "tip",
    )?;

    // Prismatic joint between cylinder "slide" and rod "slide" along x-axis.
    mech.add_prismatic_joint(
        &format!("force_{}_slide", force_index),
        &cyl_id,
        "slide",
        &rod_id,
        "slide",
        Vector2::new(1.0, 0.0),
        0.0,
    )?;

    // Return the force remapped to act between the compound slide points.
    Ok(remap_force_to_compound(force, force_index))
}

// ── Remap helper ──────────────────────────────────────────────────────────────

/// Remap a force element so that it acts between the `"slide"` attachment
/// points of the compound cylinder and rod bodies.
///
/// The local coordinates for both slide points are `[0.0, 0.0]` because the
/// slide points sit at the local origins of their respective compound bodies.
/// Any named point references are cleared — the compound bodies use plain
/// attachment points with no mount-point indirection.
fn remap_force_to_compound(force: &ForceElement, idx: usize) -> ForceElement {
    let cyl_id = format!("force_{}_cyl", idx);
    let rod_id = format!("force_{}_rod", idx);

    match force {
        ForceElement::LinearSpring(s) => {
            let mut r = s.clone();
            r.body_a = cyl_id;
            r.point_a = [0.0, 0.0];
            r.point_a_name = None;
            r.body_b = rod_id;
            r.point_b = [0.0, 0.0];
            r.point_b_name = None;
            ForceElement::LinearSpring(r)
        }
        ForceElement::LinearDamper(d) => {
            let mut r = d.clone();
            r.body_a = cyl_id;
            r.point_a = [0.0, 0.0];
            r.point_a_name = None;
            r.body_b = rod_id;
            r.point_b = [0.0, 0.0];
            r.point_b_name = None;
            ForceElement::LinearDamper(r)
        }
        ForceElement::GasSpring(g) => {
            let mut r = g.clone();
            r.body_a = cyl_id;
            r.point_a = [0.0, 0.0];
            r.point_a_name = None;
            r.body_b = rod_id;
            r.point_b = [0.0, 0.0];
            r.point_b_name = None;
            ForceElement::GasSpring(r)
        }
        ForceElement::LinearActuator(a) => {
            let mut r = a.clone();
            r.body_a = cyl_id;
            r.point_a = [0.0, 0.0];
            r.point_a_name = None;
            r.body_b = rod_id;
            r.point_b = [0.0, 0.0];
            r.point_b_name = None;
            ForceElement::LinearActuator(r)
        }
        other => other.clone(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::elements::{LinearSpringElement, LinearDamperElement};

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Build a minimal BodyJson with the given attachment and mount point names.
    fn make_body_json(
        attachment_names: &[&str],
        mount_names: &[&str],
    ) -> BodyJson {
        let attachment_points = attachment_names
            .iter()
            .map(|&n| (n.to_string(), [0.0_f64, 0.0_f64]))
            .collect();
        let mount_points = mount_names
            .iter()
            .map(|&n| (n.to_string(), [0.5_f64, 0.5_f64]))
            .collect();
        BodyJson {
            attachment_points,
            mass: 0.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.0,
            mount_points,
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
        }
    }

    fn make_spring(
        body_a: &str,
        point_a_name: Option<&str>,
        body_b: &str,
        point_b_name: Option<&str>,
    ) -> ForceElement {
        ForceElement::LinearSpring(LinearSpringElement {
            body_a: body_a.to_string(),
            point_a: [0.0, 0.0],
            point_a_name: point_a_name.map(|s| s.to_string()),
            body_b: body_b.to_string(),
            point_b: [0.0, 0.0],
            point_b_name: point_b_name.map(|s| s.to_string()),
            stiffness: 1000.0,
            free_length: 0.1,
        })
    }

    fn bodies_with(
        bodies: Vec<(&str, BodyJson)>,
    ) -> HashMap<String, BodyJson> {
        bodies
            .into_iter()
            .map(|(id, bj)| (id.to_string(), bj))
            .collect()
    }

    // ── Test 1: no point_a_name / point_b_name → PureForce ───────────────────

    #[test]
    fn pure_force_no_mount_points() {
        let force = make_spring("body_a", None, "body_b", None);
        let bodies = bodies_with(vec![
            ("body_a", make_body_json(&["A"], &[])),
            ("body_b", make_body_json(&["B"], &[])),
        ]);

        assert!(
            matches!(analyze_force(&force, &bodies), CompoundAnalysis::PureForce(_)),
            "force with no named points should be PureForce"
        );
    }

    // ── Test 2: names resolve to attachment_points → PureForce ───────────────

    #[test]
    fn force_with_attachment_point_name_stays_pure() {
        // "pin_a" exists only in attachment_points, not in mount_points.
        let force = make_spring("body_a", Some("pin_a"), "body_b", Some("pin_b"));
        let bodies = bodies_with(vec![
            ("body_a", make_body_json(&["pin_a"], &[])),
            ("body_b", make_body_json(&["pin_b"], &[])),
        ]);

        assert!(
            matches!(analyze_force(&force, &bodies), CompoundAnalysis::PureForce(_)),
            "names that resolve to attachment_points should yield PureForce"
        );
    }

    // ── Test 3: both names in mount_points → NeedsExpansion(both true) ────────

    #[test]
    fn force_with_mount_point_needs_expansion() {
        let force = make_spring("body_a", Some("mount_top"), "body_b", Some("mount_bot"));
        let bodies = bodies_with(vec![
            // "mount_top" is a mount_point, not an attachment_point
            ("body_a", make_body_json(&[], &["mount_top"])),
            ("body_b", make_body_json(&[], &["mount_bot"])),
        ]);

        match analyze_force(&force, &bodies) {
            CompoundAnalysis::NeedsExpansion { mount_a, mount_b, .. } => {
                assert!(mount_a, "point A should be detected as a mount point");
                assert!(mount_b, "point B should be detected as a mount point");
            }
            CompoundAnalysis::PureForce(_) => {
                panic!("expected NeedsExpansion but got PureForce");
            }
        }
    }

    // ── Test 4: one mount + one attachment → NeedsExpansion(mount_a:true, mount_b:false) ──

    #[test]
    fn mixed_mount_attachment_detected() {
        // body_a has "top" as a mount_point; body_b has "bot" as an attachment_point.
        let force = make_spring("body_a", Some("top"), "body_b", Some("bot"));
        let bodies = bodies_with(vec![
            ("body_a", make_body_json(&[], &["top"])),
            ("body_b", make_body_json(&["bot"], &[])),
        ]);

        match analyze_force(&force, &bodies) {
            CompoundAnalysis::NeedsExpansion { mount_a, mount_b, .. } => {
                assert!(mount_a, "point A should be detected as a mount point");
                assert!(!mount_b, "point B is an attachment point, not a mount point");
            }
            CompoundAnalysis::PureForce(_) => {
                panic!("expected NeedsExpansion but got PureForce");
            }
        }
    }

    // ── Test 5: compound cylinder has correct attachment points ───────────────

    #[test]
    fn create_compound_cylinder_has_correct_points() {
        let body = create_compound_cylinder(7, 0.25);
        assert_eq!(body.id, "force_7_cyl");
        assert!(
            body.attachment_points.contains_key("base"),
            "cylinder must have 'base' point"
        );
        assert!(
            body.attachment_points.contains_key("slide"),
            "cylinder must have 'slide' point"
        );
        // "base" should be at origin
        let base = body.attachment_points["base"];
        assert!((base[0] - 0.0).abs() < 1e-12 && (base[1] - 0.0).abs() < 1e-12);
        // "slide" should be at (half_len, 0)
        let slide = body.attachment_points["slide"];
        assert!((slide[0] - 0.25).abs() < 1e-12 && (slide[1] - 0.0).abs() < 1e-12);
    }

    // ── Test 6: compound rod has correct attachment points ────────────────────

    #[test]
    fn create_compound_rod_has_correct_points() {
        let body = create_compound_rod(3, 0.15);
        assert_eq!(body.id, "force_3_rod");
        assert!(
            body.attachment_points.contains_key("slide"),
            "rod must have 'slide' point"
        );
        assert!(
            body.attachment_points.contains_key("tip"),
            "rod must have 'tip' point"
        );
        // "slide" at origin
        let slide = body.attachment_points["slide"];
        assert!((slide[0] - 0.0).abs() < 1e-12 && (slide[1] - 0.0).abs() < 1e-12);
        // "tip" at (half_len, 0)
        let tip = body.attachment_points["tip"];
        assert!((tip[0] - 0.15).abs() < 1e-12 && (tip[1] - 0.0).abs() < 1e-12);
    }

    // ── Bonus test: remap_force_to_compound sets correct body IDs ─────────────

    #[test]
    fn remap_force_targets_compound_bodies() {
        let original = ForceElement::LinearSpring(LinearSpringElement {
            body_a: "chassis".to_string(),
            point_a: [0.1, 0.2],
            point_a_name: Some("mount_front".to_string()),
            body_b: "axle".to_string(),
            point_b: [0.3, 0.4],
            point_b_name: Some("mount_rear".to_string()),
            stiffness: 5000.0,
            free_length: 0.3,
        });

        let remapped = remap_force_to_compound(&original, 2);
        match remapped {
            ForceElement::LinearSpring(s) => {
                assert_eq!(s.body_a, "force_2_cyl");
                assert_eq!(s.body_b, "force_2_rod");
                assert_eq!(s.point_a, [0.0, 0.0]);
                assert_eq!(s.point_b, [0.0, 0.0]);
                assert!(s.point_a_name.is_none());
                assert!(s.point_b_name.is_none());
                // Scalar parameters should be preserved
                assert!((s.stiffness - 5000.0).abs() < 1e-12);
                assert!((s.free_length - 0.3).abs() < 1e-12);
            }
            _ => panic!("expected LinearSpring after remap"),
        }
    }

    // ── Bonus test: LinearDamper remap ────────────────────────────────────────

    #[test]
    fn remap_damper_targets_compound_bodies() {
        let original = ForceElement::LinearDamper(LinearDamperElement {
            body_a: "frame".to_string(),
            point_a: [0.0, 0.0],
            point_a_name: Some("damp_top".to_string()),
            body_b: "wheel".to_string(),
            point_b: [0.0, 0.0],
            point_b_name: Some("damp_bot".to_string()),
            damping: 200.0,
        });

        let remapped = remap_force_to_compound(&original, 5);
        match remapped {
            ForceElement::LinearDamper(d) => {
                assert_eq!(d.body_a, "force_5_cyl");
                assert_eq!(d.body_b, "force_5_rod");
                assert!(d.point_a_name.is_none());
                assert!(d.point_b_name.is_none());
                assert!((d.damping - 200.0).abs() < 1e-12);
            }
            _ => panic!("expected LinearDamper after remap"),
        }
    }
}
