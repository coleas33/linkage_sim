//! Tests for the ForceZone force element variant.

use nalgebra::{DVector, Vector2};

use linkage_sim_rs::core::body::{make_bar, make_ground, BodyGeometry};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::forces::elements::{ForceElement, ForceZoneElement};

/// Build a single-bar mechanism with BodyGeometry attached.
///
/// Bar "bar" has attachment points A at (0,0) and B at (0.1, 0).
/// A revolute joint pins bar.A to ground.O at the origin.
/// A constant-speed driver locks the angle at 0 rad (bar lies along +x).
/// BodyGeometry: width=0.06, height=0.015, offset=(0.05, 0.0) — centered
/// between A and B.
fn build_test_mechanism() -> (Mechanism, DVector<f64>) {
    let ground = make_ground(&[("O", 0.0, 0.0)]);
    let mut bar = make_bar("bar", "A", "B", 0.1, 0.0, 0.0);
    bar.geometry = Some(BodyGeometry::new(0.06, 0.015, Vector2::new(0.05, 0.0)).unwrap());

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(bar).unwrap();
    mech.add_revolute_joint("J1", "ground", "O", "bar", "A").unwrap();
    mech.add_constant_speed_driver("D1", "ground", "bar", 0.0, 0.0).unwrap();
    mech.build().unwrap();

    let q = DVector::zeros(mech.state().n_coords());
    (mech, q)
}

#[test]
fn force_zone_no_overlap_produces_zero_force() {
    let (mech, q) = build_test_mechanism();
    let state = mech.state();
    let bodies = mech.bodies();

    // Zone is far away from the bar (bar geometry is near origin)
    let fz = ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [10.0, 10.0],
        zone_max: [11.0, 11.0],
        force: [0.0, -500.0],
        label: None,
        body_local_app_point: None,
    };

    let q_dot = DVector::zeros(q.len());
    let contribution = ForceElement::ForceZone(fz).evaluate(state, bodies, &q, &q_dot, 0.0);
    assert!(
        contribution.iter().all(|v| v.abs() < 1e-12),
        "No overlap should produce zero generalized force, got: {:?}",
        contribution,
    );
}

#[test]
fn force_zone_full_overlap_applies_full_force() {
    let (mech, q) = build_test_mechanism();
    let state = mech.state();
    let bodies = mech.bodies();

    // Zone fully encloses the bar's geometry (width=0.06, height=0.015, centered at x=0.05)
    // Bar geometry spans x=[0.02, 0.08], y=[-0.0075, 0.0075]
    let fz = ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [-1.0, -1.0],
        zone_max: [1.0, 1.0],
        force: [0.0, -500.0],
        label: None,
        body_local_app_point: None,
    };

    let q_dot = DVector::zeros(q.len());
    let contribution = ForceElement::ForceZone(fz).evaluate(state, bodies, &q, &q_dot, 0.0);
    // Full overlap => ratio = 1.0, so the full -500 N force should appear
    assert!(
        contribution.amax() > 0.0 || contribution.amin() < 0.0,
        "Full overlap should produce non-zero generalized force, got: {:?}",
        contribution,
    );
}

#[test]
fn force_zone_partial_overlap_scales_force() {
    let (mech, q) = build_test_mechanism();
    let state = mech.state();
    let bodies = mech.bodies();
    let q_dot = DVector::zeros(q.len());

    // Full overlap zone
    let fz_full = ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [-1.0, -1.0],
        zone_max: [1.0, 1.0],
        force: [0.0, -500.0],
        label: None,
        body_local_app_point: None,
    };
    let full = ForceElement::ForceZone(fz_full).evaluate(state, bodies, &q, &q_dot, 0.0);

    // Partial overlap: zone covers only the right half of the body geometry.
    // Body geometry spans x=[0.02, 0.08]. Zone x starts at 0.05 => covers [0.05, 0.08]
    // which is partial overlap. With binary semantics, this still applies the full force.
    let fz_partial = ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [0.05, -1.0],
        zone_max: [1.0, 1.0],
        force: [0.0, -500.0],
        label: None,
        body_local_app_point: None,
    };
    let partial = ForceElement::ForceZone(fz_partial).evaluate(state, bodies, &q, &q_dot, 0.0);

    // Binary overlap: any contact applies full force, so partial should be
    // comparable to full (small difference from different application point).
    let full_norm = full.norm();
    let partial_norm = partial.norm();
    assert!(
        full_norm > 1e-6,
        "Full overlap force should be significant, got: {}",
        full_norm,
    );
    assert!(
        partial_norm > 1e-6,
        "Partial overlap force should be non-zero, got: {}",
        partial_norm,
    );
    // Both should apply essentially the same force magnitude (within 5%).
    let diff = (partial_norm - full_norm).abs() / full_norm;
    assert!(
        diff < 0.05,
        "Partial and full overlap forces should be similar (diff {:.1}%): partial={}, full={}",
        diff * 100.0,
        partial_norm,
        full_norm,
    );
}

#[test]
fn force_zone_missing_body_returns_zero() {
    let (mech, q) = build_test_mechanism();
    let state = mech.state();
    let bodies = mech.bodies();

    let fz = ForceZoneElement {
        body_id: "nonexistent_body".to_string(),
        zone_min: [-1.0, -1.0],
        zone_max: [1.0, 1.0],
        force: [0.0, -500.0],
        label: None,
        body_local_app_point: None,
    };

    let q_dot = DVector::zeros(q.len());
    let contribution = ForceElement::ForceZone(fz).evaluate(state, bodies, &q, &q_dot, 0.0);
    assert!(contribution.iter().all(|v| v.abs() < 1e-12));
}

#[test]
fn force_zone_body_without_geometry_returns_zero() {
    // Build mechanism without setting geometry on the bar
    let ground = make_ground(&[("O", 0.0, 0.0)]);
    let bar = make_bar("bar", "A", "B", 0.1, 0.0, 0.0); // no geometry

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(bar).unwrap();
    mech.add_revolute_joint("J1", "ground", "O", "bar", "A").unwrap();
    mech.add_constant_speed_driver("D1", "ground", "bar", 0.0, 0.0).unwrap();
    mech.build().unwrap();

    let q = DVector::zeros(mech.state().n_coords());

    let fz = ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [-1.0, -1.0],
        zone_max: [1.0, 1.0],
        force: [0.0, -500.0],
        label: None,
        body_local_app_point: None,
    };

    let q_dot = DVector::zeros(q.len());
    let contribution = ForceElement::ForceZone(fz).evaluate(mech.state(), mech.bodies(), &q, &q_dot, 0.0);
    assert!(contribution.iter().all(|v| v.abs() < 1e-12));
}

#[test]
fn force_zone_serialization_roundtrip() {
    let fz = ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [1.0, 2.0],
        zone_max: [3.0, 4.0],
        force: [10.0, -20.0],
        label: Some("test zone".to_string()),
        body_local_app_point: None,
    };
    let fe = ForceElement::ForceZone(fz);

    let json = serde_json::to_string(&fe).expect("serialize");
    let deserialized: ForceElement = serde_json::from_str(&json).expect("deserialize");

    match deserialized {
        ForceElement::ForceZone(fz2) => {
            assert_eq!(fz2.body_id, "bar");
            assert_eq!(fz2.zone_min, [1.0, 2.0]);
            assert_eq!(fz2.zone_max, [3.0, 4.0]);
            assert_eq!(fz2.force, [10.0, -20.0]);
            assert_eq!(fz2.label, Some("test zone".to_string()));
        }
        other => panic!("Expected ForceZone variant, got: {:?}", other),
    }
}

#[test]
fn force_zone_type_name() {
    let fz = ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [0.0, 0.0],
        zone_max: [1.0, 1.0],
        force: [0.0, -10.0],
        label: None,
        body_local_app_point: None,
    };
    let fe = ForceElement::ForceZone(fz);
    assert_eq!(fe.type_name(), "Force Zone");
}

#[test]
fn force_zone_attached_body_ids() {
    let fz = ForceZoneElement {
        body_id: "my_bar".to_string(),
        zone_min: [0.0, 0.0],
        zone_max: [1.0, 1.0],
        force: [0.0, -10.0],
        label: None,
        body_local_app_point: None,
    };
    let fe = ForceElement::ForceZone(fz);
    assert_eq!(fe.attached_body_ids(), vec!["my_bar"]);
}
