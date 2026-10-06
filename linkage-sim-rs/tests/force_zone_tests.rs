//! Tests for the ForceZone force element variant.

use approx::assert_relative_eq;
use nalgebra::{DVector, Vector2};

use linkage_sim_rs::core::body::{make_bar, make_ground, BodyGeometry};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::forces::elements::{
    force_zone_application, ForceElement, ForceZoneElement, ZoneAppMode,
};

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
        at_contact_point: false,
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
        at_contact_point: false,
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
        at_contact_point: false,
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
        at_contact_point: false,
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
        at_contact_point: false,
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
        at_contact_point: false,
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
        at_contact_point: false,
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
        at_contact_point: false,
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
        at_contact_point: false,
    };
    let fe = ForceElement::ForceZone(fz);
    assert_eq!(fe.attached_body_ids(), vec!["my_bar"]);
}

// ── Contact point (schema 1.2.0, decisions R-3, R-4) ─────────────────────

/// The bar of `build_test_mechanism` carrying a wheel (a 0.05 m circle centred
/// 0.1 m along it), turned to `theta`. Positions are not re-solved: the zone
/// reads the pose straight from q.
fn wheel_bar_at(theta: f64) -> (Mechanism, DVector<f64>) {
    let ground = make_ground(&[("O", 0.0, 0.0)]);
    let mut bar = make_bar("bar", "A", "B", 0.1, 0.0, 0.0);
    bar.geometry = Some(BodyGeometry::circle(0.05, Vector2::new(0.1, 0.0)).unwrap());

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(bar).unwrap();
    mech.add_revolute_joint("J1", "ground", "O", "bar", "A").unwrap();
    mech.add_constant_speed_driver("D1", "ground", "bar", 0.0, 0.0).unwrap();
    mech.build().unwrap();

    let mut q = DVector::zeros(mech.state().n_coords());
    q[mech.state().get_index("bar").unwrap().theta_idx()] = theta;
    (mech, q)
}

/// A zone over the whole wheel, its force at the contact point.
fn contact_zone(force: [f64; 2]) -> ForceZoneElement {
    ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [-1.0, -1.0],
        zone_max: [1.0, 1.0],
        force,
        label: None,
        body_local_app_point: None,
        at_contact_point: true,
    }
}

#[test]
fn the_contact_point_stays_at_the_bottom_of_the_wheel_as_it_turns() {
    for theta in [0.0, 0.6, 1.4, -0.9] {
        let (mech, q) = wheel_bar_at(theta);
        let geo = mech.bodies()["bar"].geometry.clone().unwrap();
        let pose = mech.state().get_pose("bar", &q);
        let app = force_zone_application(&contact_zone([0.0, 500.0]), &geo, pose);
        assert_eq!(app.mode, ZoneAppMode::Contact);
        assert!(app.active, "theta {theta}");
        let centre = geo.centre_world(pose.0, pose.1, pose.2);
        let point = app.point.unwrap();
        assert_relative_eq!(point.x, centre.x, epsilon = 1e-12);
        assert_relative_eq!(point.y, centre.y - 0.025, epsilon = 1e-12);
    }
}

#[test]
fn the_contact_point_loads_the_body_like_the_same_force_at_the_hub() {
    use linkage_sim_rs::forces::helpers::point_force_to_q;
    let (mech, q) = wheel_bar_at(0.6);
    let q_dot = DVector::zeros(q.len());
    let got = ForceElement::ForceZone(contact_zone([0.0, 500.0]))
        .evaluate(mech.state(), mech.bodies(), &q, &q_dot, 0.0);
    // The force at the wheel's bottom, given in the body frame.
    let (bx, by, th) = mech.state().get_pose("bar", &q);
    let geo = mech.bodies()["bar"].geometry.clone().unwrap();
    let bottom = geo.centre_world(bx, by, th) - Vector2::new(0.0, 0.025);
    let (s, c) = th.sin_cos();
    let (dx, dy) = (bottom.x - bx, bottom.y - by);
    let local = Vector2::new(c * dx + s * dy, -s * dx + c * dy);
    let force = Vector2::new(0.0, 500.0);
    let want = point_force_to_q(mech.state(), "bar", &local, &force, &q);
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(*g, *w, epsilon = 1e-12);
    }
    // A vertical force on the hub's vertical line: the same Q as at the hub.
    let at_hub = point_force_to_q(mech.state(), "bar", &Vector2::new(0.1, 0.0), &force, &q);
    for (g, h) in got.iter().zip(at_hub.iter()) {
        assert_relative_eq!(*g, *h, epsilon = 1e-9);
    }
}

#[test]
fn a_sideways_force_meets_the_side_of_the_wheel() {
    let (mech, q) = wheel_bar_at(0.0);
    let geo = mech.bodies()["bar"].geometry.clone().unwrap();
    let pose = mech.state().get_pose("bar", &q);
    let app = force_zone_application(&contact_zone([-500.0, 0.0]), &geo, pose);
    // Pushing towards -x: the surface comes from +x and meets the rightmost point.
    let p = app.point.unwrap();
    assert_relative_eq!(p.x, 0.125, epsilon = 1e-12);
    assert_relative_eq!(p.y, 0.0, epsilon = 1e-12);
    // A zero force falls back to the lowest point.
    let zero = force_zone_application(&contact_zone([0.0, 0.0]), &geo, pose);
    let p = zero.point.unwrap();
    assert_relative_eq!(p.x, 0.1, epsilon = 1e-12);
    assert_relative_eq!(p.y, -0.025, epsilon = 1e-12);
}

#[test]
fn the_contact_point_still_needs_the_shape_in_the_zone() {
    let (mech, q) = wheel_bar_at(0.0);
    let mut fz = contact_zone([0.0, 500.0]);
    fz.zone_min = [10.0, 10.0];
    fz.zone_max = [11.0, 11.0];
    let geo = mech.bodies()["bar"].geometry.clone().unwrap();
    let app = force_zone_application(&fz, &geo, mech.state().get_pose("bar", &q));
    assert!(!app.active);
    assert!(app.point.is_some(), "the canvas still marks where the force will act");
    let q_dot = DVector::zeros(q.len());
    let got = ForceElement::ForceZone(fz).evaluate(mech.state(), mech.bodies(), &q, &q_dot, 0.0);
    assert!(got.iter().all(|v| v.abs() < 1e-12), "{got:?}");
}

#[test]
fn the_contact_point_wins_over_a_locked_point_which_is_kept() {
    let mut fz = contact_zone([0.0, 500.0]);
    fz.body_local_app_point = Some([0.05, 0.0]);
    assert_eq!(fz.app_mode(), ZoneAppMode::Contact);
    fz.at_contact_point = false;
    assert_eq!(fz.app_mode(), ZoneAppMode::Locked);
    fz.body_local_app_point = None;
    assert_eq!(fz.app_mode(), ZoneAppMode::Centroid);
}

#[test]
fn with_app_mode_switches_and_keeps_the_locked_point() {
    let mut fz = contact_zone([0.0, 500.0]);
    fz.at_contact_point = false;
    // Locked with no earlier point: the seed (the panel passes the shape's centre).
    let locked = fz.with_app_mode(ZoneAppMode::Locked, [0.1, 0.0]);
    assert_eq!(locked.app_mode(), ZoneAppMode::Locked);
    assert_eq!(locked.body_local_app_point, Some([0.1, 0.0]));
    // Contact keeps the locked point; Locked again restores it, ignoring the seed.
    let contact = locked.with_app_mode(ZoneAppMode::Contact, [9.0, 9.0]);
    assert_eq!(contact.app_mode(), ZoneAppMode::Contact);
    assert_eq!(contact.body_local_app_point, Some([0.1, 0.0]));
    let back = contact.with_app_mode(ZoneAppMode::Locked, [9.0, 9.0]);
    assert_eq!(back.app_mode(), ZoneAppMode::Locked);
    assert_eq!(back.body_local_app_point, Some([0.1, 0.0]));
    // Overlap centre clears both.
    let centroid = contact.with_app_mode(ZoneAppMode::Centroid, [9.0, 9.0]);
    assert_eq!(centroid.app_mode(), ZoneAppMode::Centroid);
    assert_eq!(centroid.body_local_app_point, None);
    assert!(!centroid.at_contact_point);
}

#[test]
fn at_contact_point_is_left_out_when_false_and_defaults_to_false() {
    let mut fz = contact_zone([0.0, 500.0]);
    fz.at_contact_point = false;
    let json = serde_json::to_value(ForceElement::ForceZone(fz.clone())).unwrap();
    assert!(json.get("at_contact_point").is_none(), "{json}");
    let back: ForceElement = serde_json::from_value(json).unwrap();
    let ForceElement::ForceZone(back) = back else { panic!("a force zone") };
    assert!(!back.at_contact_point);
    fz.at_contact_point = true;
    let json = serde_json::to_value(ForceElement::ForceZone(fz)).unwrap();
    assert_eq!(json["at_contact_point"], true);
}

/// A rectangle is where the contact point changes the load: tilted, its lowest
/// corner is off the line of action through its centre, so the generalized
/// force differs from the overlap centroid's.
#[test]
fn a_tilted_rectangle_s_contact_point_is_its_lowest_corner() {
    use linkage_sim_rs::forces::helpers::point_force_to_q;
    let (mech, mut q) = build_test_mechanism();
    let theta = 0.3;
    let theta_idx = mech.state().get_index("bar").unwrap().theta_idx();
    q[theta_idx] = theta;
    // build_test_mechanism's 0.06 x 0.015 rectangle centred at (0.05, 0); the body origin at (0, 0).
    let geo = mech.bodies()["bar"].geometry.clone().unwrap();
    let mut fz = ForceZoneElement {
        body_id: "bar".to_string(),
        zone_min: [-1.0, -1.0],
        zone_max: [1.0, 1.0],
        force: [0.0, 500.0],
        label: None,
        body_local_app_point: None,
        at_contact_point: true,
    };
    let q_dot = DVector::zeros(q.len());
    let got = ForceElement::ForceZone(fz.clone()).evaluate(mech.state(), mech.bodies(), &q, &q_dot, 0.0);
    // The lowest corner, from the geometry's own outline (independent of the zone helper).
    let corners = geo.outline_world(0.0, 0.0, theta);
    let lowest = corners.iter().min_by(|a, b| a.y.total_cmp(&b.y)).unwrap();
    let (s, c) = theta.sin_cos();
    let local = Vector2::new(c * lowest.x + s * lowest.y, -s * lowest.x + c * lowest.y);
    let want = point_force_to_q(mech.state(), "bar", &local, &Vector2::new(0.0, 500.0), &q);
    for (g, w) in got.iter().zip(want.iter()) {
        assert_relative_eq!(*g, *w, epsilon = 1e-12);
    }
    // The overlap centre (here the rectangle's centre) loads the body differently.
    fz.at_contact_point = false;
    let centroid = ForceElement::ForceZone(fz).evaluate(mech.state(), mech.bodies(), &q, &q_dot, 0.0);
    assert!(
        (got[theta_idx] - centroid[theta_idx]).abs() > 0.1,
        "contact {} vs centroid {}",
        got[theta_idx],
        centroid[theta_idx]
    );
}
