//! Integration test: compound force expansion through the serialization load path.
//!
//! Verifies that a mechanism JSON with mount-point-referenced forces correctly
//! expands into compound bodies (cylinder + rod), joints (revolute + prismatic),
//! and a remapped force element during `load_mechanism_unbuilt`.

use std::collections::HashMap;

use linkage_sim_rs::core::constraint::Constraint;
use linkage_sim_rs::forces::elements::ForceElement;
use linkage_sim_rs::io::{
    load_mechanism_unbuilt, BodyJson, JointJson, MechanismJson, SCHEMA_VERSION,
};

/// Build a `MechanismJson` with two bodies connected by a revolute joint,
/// plus a spring whose endpoints reference mount points (not attachment points).
fn make_compound_spring_json() -> MechanismJson {
    let mut bodies = HashMap::new();

    // Ground body with attachment point "O" and mount point "spring_base".
    let mut ground_attach = HashMap::new();
    ground_attach.insert("O".to_string(), [0.0, 0.0]);
    let mut ground_mount = HashMap::new();
    ground_mount.insert("spring_base".to_string(), [0.05, 0.02]);
    bodies.insert(
        "ground".to_string(),
        BodyJson {
            attachment_points: ground_attach,
            mass: 0.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.0,
            mount_points: ground_mount,
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
            label: None,
            geometry: None,
        },
    );

    // Crank body with attachment point "O2" (joint pin) and mount point "spring_tip".
    let mut crank_attach = HashMap::new();
    crank_attach.insert("O2".to_string(), [0.0, 0.0]);
    let mut crank_mount = HashMap::new();
    crank_mount.insert("spring_tip".to_string(), [0.08, 0.0]);
    bodies.insert(
        "crank".to_string(),
        BodyJson {
            attachment_points: crank_attach,
            mass: 1.0,
            cg_local: [0.05, 0.0],
            izz_cg: 0.001,
            mount_points: crank_mount,
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
            label: None,
            geometry: None,
        },
    );

    // Revolute joint: ground "O" ↔ crank "O2"
    let mut joints = HashMap::new();
    joints.insert(
        "ground_crank".to_string(),
        JointJson::Revolute {
            body_i: "ground".to_string(),
            body_j: "crank".to_string(),
            point_i: "O".to_string(),
            point_j: "O2".to_string(),
            label: None,
        },
    );

    // Spring referencing mount points by name.
    let spring = ForceElement::LinearSpring(
        linkage_sim_rs::forces::elements::LinearSpringElement {
            body_a: "ground".to_string(),
            point_a: [0.05, 0.02], // resolved coordinates matching mount point
            point_a_name: Some("spring_base".to_string()),
            body_b: "crank".to_string(),
            point_b: [0.08, 0.0],
            point_b_name: Some("spring_tip".to_string()),
            stiffness: 500.0,
            free_length: 0.05,
        },
    );

    MechanismJson {
        schema_version: SCHEMA_VERSION.to_string(),
        bodies,
        joints,
        drivers: HashMap::new(),
        load_cases: Vec::new(),
        forces: vec![spring],
        sweep_config: None,
        mounting_angle: 0.0,
        linear_drivers: Vec::new(),
    }
}

/// Build a similar mechanism but with the spring using plain attachment points
/// (no mount points) so compound expansion is NOT triggered.
fn make_pure_spring_json() -> MechanismJson {
    let mut bodies = HashMap::new();

    let mut ground_attach = HashMap::new();
    ground_attach.insert("O".to_string(), [0.0, 0.0]);
    ground_attach.insert("spring_base".to_string(), [0.05, 0.02]);
    bodies.insert(
        "ground".to_string(),
        BodyJson {
            attachment_points: ground_attach,
            mass: 0.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.0,
            mount_points: HashMap::new(),
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
            label: None,
            geometry: None,
        },
    );

    let mut crank_attach = HashMap::new();
    crank_attach.insert("O2".to_string(), [0.0, 0.0]);
    crank_attach.insert("spring_tip".to_string(), [0.08, 0.0]);
    bodies.insert(
        "crank".to_string(),
        BodyJson {
            attachment_points: crank_attach,
            mass: 1.0,
            cg_local: [0.05, 0.0],
            izz_cg: 0.001,
            mount_points: HashMap::new(),
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
            label: None,
            geometry: None,
        },
    );

    let mut joints = HashMap::new();
    joints.insert(
        "ground_crank".to_string(),
        JointJson::Revolute {
            body_i: "ground".to_string(),
            body_j: "crank".to_string(),
            point_i: "O".to_string(),
            point_j: "O2".to_string(),
            label: None,
        },
    );

    let spring = ForceElement::LinearSpring(
        linkage_sim_rs::forces::elements::LinearSpringElement {
            body_a: "ground".to_string(),
            point_a: [0.05, 0.02],
            point_a_name: Some("spring_base".to_string()),
            body_b: "crank".to_string(),
            point_b: [0.08, 0.0],
            point_b_name: Some("spring_tip".to_string()),
            stiffness: 500.0,
            free_length: 0.05,
        },
    );

    MechanismJson {
        schema_version: SCHEMA_VERSION.to_string(),
        bodies,
        joints,
        drivers: HashMap::new(),
        load_cases: Vec::new(),
        forces: vec![spring],
        sweep_config: None,
        mounting_angle: 0.0,
        linear_drivers: Vec::new(),
    }
}

// ── Test 1: compound expansion creates the expected bodies ───────────────────

#[test]
fn compound_expansion_creates_cylinder_and_rod_bodies() {
    let json = make_compound_spring_json();
    let json_str = serde_json::to_string(&json).unwrap();
    let mech = load_mechanism_unbuilt(&json_str).expect("load should succeed");

    // Original bodies: ground + crank.
    // Compound bodies: force_0_cyl + force_0_rod.
    let bodies = mech.bodies();
    assert!(
        bodies.contains_key("force_0_cyl"),
        "compound cylinder body should be created"
    );
    assert!(
        bodies.contains_key("force_0_rod"),
        "compound rod body should be created"
    );
    assert_eq!(
        bodies.len(),
        4,
        "should have ground + crank + cylinder + rod = 4 bodies"
    );
}

// ── Test 2: compound bodies have correct attachment points ───────────────────

#[test]
fn compound_bodies_have_correct_attachment_points() {
    let json = make_compound_spring_json();
    let json_str = serde_json::to_string(&json).unwrap();
    let mech = load_mechanism_unbuilt(&json_str).expect("load should succeed");

    let cyl = &mech.bodies()["force_0_cyl"];
    assert!(
        cyl.attachment_points.contains_key("base"),
        "cylinder must have 'base' attachment point"
    );
    assert!(
        cyl.attachment_points.contains_key("slide"),
        "cylinder must have 'slide' attachment point"
    );

    let rod = &mech.bodies()["force_0_rod"];
    assert!(
        rod.attachment_points.contains_key("slide"),
        "rod must have 'slide' attachment point"
    );
    assert!(
        rod.attachment_points.contains_key("tip"),
        "rod must have 'tip' attachment point"
    );
}

// ── Test 3: expansion creates the expected joints ────────────────────────────

#[test]
fn compound_expansion_creates_three_joints() {
    let json = make_compound_spring_json();
    let json_str = serde_json::to_string(&json).unwrap();
    let mech = load_mechanism_unbuilt(&json_str).expect("load should succeed");

    let joints = mech.joints();
    // Original: 1 revolute (ground_crank).
    // Compound: 2 revolute (force_0_base, force_0_tip) + 1 prismatic (force_0_slide).
    assert_eq!(
        joints.len(),
        4,
        "should have 1 original + 3 compound = 4 joints"
    );

    // Verify we have the expected joint IDs.
    let joint_ids: Vec<&str> = joints.iter().map(|j| j.id()).collect();
    assert!(
        joint_ids.contains(&"force_0_base"),
        "should have revolute joint at cylinder base"
    );
    assert!(
        joint_ids.contains(&"force_0_tip"),
        "should have revolute joint at rod tip"
    );
    assert!(
        joint_ids.contains(&"force_0_slide"),
        "should have prismatic slide joint"
    );
}

// ── Test 4: the force is remapped to compound bodies ─────────────────────────

#[test]
fn force_is_remapped_to_compound_bodies() {
    let json = make_compound_spring_json();
    let json_str = serde_json::to_string(&json).unwrap();
    let mech = load_mechanism_unbuilt(&json_str).expect("load should succeed");

    assert_eq!(mech.forces().len(), 1, "should have exactly one force");

    match &mech.forces()[0] {
        ForceElement::LinearSpring(s) => {
            assert_eq!(
                s.body_a, "force_0_cyl",
                "force body_a should be remapped to cylinder"
            );
            assert_eq!(
                s.body_b, "force_0_rod",
                "force body_b should be remapped to rod"
            );
            assert_eq!(s.point_a, [0.0, 0.0], "remapped point_a should be at origin");
            assert_eq!(s.point_b, [0.0, 0.0], "remapped point_b should be at origin");
            assert!(
                s.point_a_name.is_none(),
                "named point refs should be cleared"
            );
            assert!(
                s.point_b_name.is_none(),
                "named point refs should be cleared"
            );
            // Scalar parameters preserved.
            assert!(
                (s.stiffness - 500.0).abs() < 1e-12,
                "stiffness should be preserved"
            );
            assert!(
                (s.free_length - 0.05).abs() < 1e-12,
                "free_length should be preserved"
            );
        }
        other => panic!(
            "expected LinearSpring, got {:?}",
            std::mem::discriminant(other)
        ),
    }
}

// ── Test 5: pure attachment-point force does NOT trigger expansion ────────────

#[test]
fn pure_attachment_point_force_no_expansion() {
    let json = make_pure_spring_json();
    let json_str = serde_json::to_string(&json).unwrap();
    let mech = load_mechanism_unbuilt(&json_str).expect("load should succeed");

    // No compound bodies should be created.
    assert_eq!(
        mech.bodies().len(),
        2,
        "should have only ground + crank (no compound bodies)"
    );

    // Only the original revolute joint.
    assert_eq!(mech.joints().len(), 1, "should have only the original joint");

    // Force should reference the original bodies, not compound ones.
    match &mech.forces()[0] {
        ForceElement::LinearSpring(s) => {
            assert_eq!(s.body_a, "ground");
            assert_eq!(s.body_b, "crank");
        }
        other => panic!(
            "expected LinearSpring, got {:?}",
            std::mem::discriminant(other)
        ),
    }
}

// ── Test 6: synthetic mount-point attachment points are created ───────────────

#[test]
fn synthetic_attachment_points_created_for_mount_points() {
    let json = make_compound_spring_json();
    let json_str = serde_json::to_string(&json).unwrap();
    let mech = load_mechanism_unbuilt(&json_str).expect("load should succeed");

    // The serialization path should promote mount points to synthetic
    // attachment points named `_force_{idx}_mount_{a|b}`.
    let ground = &mech.bodies()["ground"];
    assert!(
        ground
            .attachment_points
            .contains_key("_force_0_mount_a"),
        "ground should have synthetic attachment point _force_0_mount_a"
    );

    let crank = &mech.bodies()["crank"];
    assert!(
        crank
            .attachment_points
            .contains_key("_force_0_mount_b"),
        "crank should have synthetic attachment point _force_0_mount_b"
    );
}

// ── Test 7: half_len geometry is correct ─────────────────────────────────────

#[test]
fn compound_bodies_half_len_geometry() {
    let json = make_compound_spring_json();
    let json_str = serde_json::to_string(&json).unwrap();
    let mech = load_mechanism_unbuilt(&json_str).expect("load should succeed");

    // Compute expected half_len from the two mount point positions.
    // spring_base = (0.05, 0.02), spring_tip = (0.08, 0.0)
    let dx: f64 = 0.08 - 0.05;
    let dy: f64 = 0.0 - 0.02;
    let initial_length = (dx * dx + dy * dy).sqrt();
    let expected_half_len = initial_length / 2.0;

    // Cylinder: "slide" should be at (half_len, 0).
    let cyl = &mech.bodies()["force_0_cyl"];
    let slide = cyl.attachment_points["slide"];
    assert!(
        (slide[0] - expected_half_len).abs() < 1e-12,
        "cylinder slide x should be half_len ({expected_half_len}), got {}",
        slide[0]
    );
    assert!(
        slide[1].abs() < 1e-12,
        "cylinder slide y should be 0, got {}",
        slide[1]
    );

    // Rod: "tip" should be at (half_len, 0).
    let rod = &mech.bodies()["force_0_rod"];
    let tip = rod.attachment_points["tip"];
    assert!(
        (tip[0] - expected_half_len).abs() < 1e-12,
        "rod tip x should be half_len ({expected_half_len}), got {}",
        tip[0]
    );
    assert!(
        tip[1].abs() < 1e-12,
        "rod tip y should be 0, got {}",
        tip[1]
    );
}

// ── Test 8: JSON round-trip — load, serialize, reload ────────────────────────

#[test]
fn json_round_trip_preserves_compound_spring() {
    let json = make_compound_spring_json();
    let json_str = serde_json::to_string(&json).unwrap();
    let mech = load_mechanism_unbuilt(&json_str).expect("first load should succeed");

    // Save the expanded mechanism back to JSON.
    let saved_json =
        linkage_sim_rs::io::save_mechanism(&mech).expect("save should succeed");

    // Reload and verify compound bodies survived the round-trip.
    let mech2 = load_mechanism_unbuilt(&saved_json).expect("reload should succeed");
    assert!(
        mech2.bodies().contains_key("force_0_cyl"),
        "cylinder body should survive round-trip"
    );
    assert!(
        mech2.bodies().contains_key("force_0_rod"),
        "rod body should survive round-trip"
    );

    // Force count should be preserved.
    assert_eq!(
        mech2.forces().len(),
        mech.forces().len(),
        "force count should be preserved across round-trip"
    );
}

// ── Test 9: mixed — one mount endpoint, one attachment endpoint ──────────────

#[test]
fn mixed_mount_and_attachment_expands_correctly() {
    let mut json = make_compound_spring_json();

    // Move "spring_tip" from crank's mount_points to attachment_points
    // so only point A (ground's spring_base) is a mount point.
    if let Some(crank) = json.bodies.get_mut("crank") {
        let pos = crank.mount_points.remove("spring_tip").unwrap();
        crank.attachment_points.insert("spring_tip".to_string(), pos);
    }

    let json_str = serde_json::to_string(&json).unwrap();
    let mech = load_mechanism_unbuilt(&json_str).expect("load should succeed");

    // Expansion should still happen (mount_a is true, mount_b is false).
    assert!(
        mech.bodies().contains_key("force_0_cyl"),
        "compound bodies should be created for mixed mount/attachment"
    );
    assert!(
        mech.bodies().contains_key("force_0_rod"),
        "compound bodies should be created for mixed mount/attachment"
    );

    // Only ground should have the synthetic attachment point.
    assert!(
        mech.bodies()["ground"]
            .attachment_points
            .contains_key("_force_0_mount_a"),
        "ground should have synthetic point for mount_a"
    );
    // Crank should NOT have a synthetic _force_0_mount_b because point B is
    // an attachment point, not a mount point.
    assert!(
        !mech.bodies()["crank"]
            .attachment_points
            .contains_key("_force_0_mount_b"),
        "crank should NOT have synthetic point when B is an attachment point"
    );
}
