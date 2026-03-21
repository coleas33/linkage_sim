//! Integration test: mount-point-attached forces produce identical results
//! to equivalent raw-coordinate forces.
//!
//! Verifies that resolving named mount points via `resolve_named_points` and
//! then evaluating yields the same generalized force vector as a spring
//! constructed directly with raw coordinates.

use std::collections::HashMap;

use linkage_sim_rs::core::body::{make_bar, make_ground};
use linkage_sim_rs::core::state::State;
use linkage_sim_rs::forces::elements::{ForceElement, LinearSpringElement};
use nalgebra::DVector;

#[test]
fn mount_point_spring_matches_raw_coord_spring() {
    // ── Raw-coordinate spring ────────────────────────────────────────────────
    let ground_raw = make_ground(&[("O2", 0.0, 0.0)]);
    let crank_raw = make_bar("crank", "O2", "A", 0.1, 1.0, 0.001);

    let spring_raw = ForceElement::LinearSpring(LinearSpringElement {
        body_a: "ground".to_string(),
        point_a: [0.05, 0.02],
        point_a_name: None,
        body_b: "crank".to_string(),
        point_b: [0.05, 0.0],
        point_b_name: None,
        stiffness: 500.0,
        free_length: 0.03,
    });

    // ── Named-mount-point spring ─────────────────────────────────────────────
    let mut ground_mp = make_ground(&[("O2", 0.0, 0.0)]);
    ground_mp
        .add_mount_point("spring_base", 0.05, 0.02)
        .unwrap();

    let mut crank_mp = make_bar("crank", "O2", "A", 0.1, 1.0, 0.001);
    crank_mp
        .add_mount_point("spring_tip", 0.05, 0.0)
        .unwrap();

    let spring_named = ForceElement::LinearSpring(LinearSpringElement {
        body_a: "ground".to_string(),
        point_a: [0.0, 0.0], // placeholder — will be resolved
        point_a_name: Some("spring_base".to_string()),
        body_b: "crank".to_string(),
        point_b: [0.0, 0.0], // placeholder — will be resolved
        point_b_name: Some("spring_tip".to_string()),
        stiffness: 500.0,
        free_length: 0.03,
    });

    // ── Resolve named points ─────────────────────────────────────────────────
    let mut bodies_mp = HashMap::new();
    bodies_mp.insert("ground".to_string(), ground_mp.clone());
    bodies_mp.insert("crank".to_string(), crank_mp.clone());

    let resolved = spring_named.resolve_named_points(&bodies_mp).unwrap();

    // Verify coordinates resolved correctly before comparing force outputs.
    if let ForceElement::LinearSpring(s) = &resolved {
        assert!(
            (s.point_a[0] - 0.05).abs() < 1e-15,
            "point_a[0] mismatch: expected 0.05, got {}",
            s.point_a[0]
        );
        assert!(
            (s.point_a[1] - 0.02).abs() < 1e-15,
            "point_a[1] mismatch: expected 0.02, got {}",
            s.point_a[1]
        );
        assert!(
            (s.point_b[0] - 0.05).abs() < 1e-15,
            "point_b[0] mismatch: expected 0.05, got {}",
            s.point_b[0]
        );
        assert!(
            (s.point_b[1] - 0.0).abs() < 1e-15,
            "point_b[1] mismatch: expected 0.0, got {}",
            s.point_b[1]
        );
    } else {
        panic!("wrong ForceElement variant after resolution");
    }

    // ── Build shared state ───────────────────────────────────────────────────
    // Register the one moving body ("crank"); ground is excluded from q.
    let mut state = State::new();
    state.register_body("crank").unwrap();

    // q = [x_crank, y_crank, theta_crank]; set crank at origin, rotated PI/4.
    let mut q = state.make_q();
    state.set_pose("crank", &mut q, 0.0, 0.0, std::f64::consts::FRAC_PI_4);
    let q_dot = DVector::zeros(state.n_coords());

    // ── Raw body map ─────────────────────────────────────────────────────────
    let mut bodies_raw = HashMap::new();
    bodies_raw.insert("ground".to_string(), ground_raw);
    bodies_raw.insert("crank".to_string(), crank_raw);

    // ── Evaluate both springs at the same state ──────────────────────────────
    let q_force_raw = spring_raw.evaluate(&state, &bodies_raw, &q, &q_dot, 0.0);
    let q_force_named = resolved.evaluate(&state, &bodies_mp, &q, &q_dot, 0.0);

    // ── Compare ──────────────────────────────────────────────────────────────
    assert_eq!(
        q_force_raw.len(),
        q_force_named.len(),
        "Q vector length mismatch: raw={}, named={}",
        q_force_raw.len(),
        q_force_named.len()
    );

    for i in 0..q_force_raw.len() {
        assert!(
            (q_force_raw[i] - q_force_named[i]).abs() < 1e-12,
            "Q[{}] mismatch: raw={}, named={}",
            i,
            q_force_raw[i],
            q_force_named[i]
        );
    }
}
