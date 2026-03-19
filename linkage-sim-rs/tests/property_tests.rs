//! Property-based tests for the linkage simulator using proptest.
//!
//! These tests generate random valid 4-bar mechanisms and verify fundamental
//! invariants that must hold for any valid mechanism configuration.

use proptest::prelude::*;
use std::f64::consts::PI;

use linkage_sim_rs::analysis::grashof::{check_grashof, GrashofType};
use linkage_sim_rs::core::body::{make_bar, make_ground};
use linkage_sim_rs::core::constraint::Constraint;
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::io::serialization::{load_mechanism, save_mechanism};
use linkage_sim_rs::solver::assembly::{assemble_constraints, assemble_jacobian, assemble_phi_t};
use linkage_sim_rs::solver::kinematics::{solve_position, solve_velocity};

// ---------------------------------------------------------------------------
// Helper: build a 4-bar mechanism from link lengths
// ---------------------------------------------------------------------------

/// Build a 4-bar mechanism from the given link lengths.
///
/// The ground link spans from O2=(0,0) to O4=(ground_len,0).
/// All bars are massless (mass=0, izz=0) since we only test kinematics here.
/// A constant-speed driver (omega=2*PI, theta_0=0) is attached at J1.
fn build_random_fourbar(ground_len: f64, crank_len: f64, coupler_len: f64, rocker_len: f64) -> Mechanism {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", ground_len, 0.0)]);
    let crank = make_bar("crank", "A", "B", crank_len, 0.0, 0.0);
    let coupler = make_bar("coupler", "B", "C", coupler_len, 0.0, 0.0);
    let rocker = make_bar("rocker", "C", "D", rocker_len, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
    mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0).unwrap();

    mech.build().unwrap();
    mech
}

/// Compute a geometrically consistent initial guess for a 4-bar at the given
/// crank angle. Uses the same triangle-closure approach as the sample builders.
///
/// Returns `None` if the triangle cannot close (geometry invalid at this angle).
fn fourbar_initial_guess(
    mech: &Mechanism,
    ground_len: f64,
    crank_len: f64,
    coupler_len: f64,
    rocker_len: f64,
    theta_crank: f64,
) -> Option<nalgebra::DVector<f64>> {
    let state = mech.state();
    let mut q0 = state.make_q();

    // Crank tip B
    let bx = crank_len * theta_crank.cos();
    let by = crank_len * theta_crank.sin();
    state.set_pose("crank", &mut q0, 0.0, 0.0, theta_crank);

    // Distance from O4 to crank tip B
    let dx = bx - ground_len;
    let dy = by;
    let d = (dx * dx + dy * dy).sqrt();

    // Check triangle inequality: coupler and rocker must reach from B to O4
    if d > coupler_len + rocker_len || d < (coupler_len - rocker_len).abs() {
        return None;
    }

    let alpha = dy.atan2(dx);
    let cos_beta = (d * d + rocker_len * rocker_len - coupler_len * coupler_len)
        / (2.0 * d * rocker_len);
    let cos_beta = cos_beta.clamp(-1.0, 1.0);
    let beta = cos_beta.acos();

    let theta_rocker = alpha + beta + PI;
    let cx = ground_len - rocker_len * theta_rocker.cos();
    let cy = -rocker_len * theta_rocker.sin();
    state.set_pose("rocker", &mut q0, cx, cy, theta_rocker);

    let theta_coupler = (cy - by).atan2(cx - bx);
    state.set_pose("coupler", &mut q0, bx, by, theta_coupler);

    Some(q0)
}

// ---------------------------------------------------------------------------
// proptest strategies
// ---------------------------------------------------------------------------

/// Strategy to generate valid 4-bar link lengths.
/// All links between 0.5 and 5.0 meters.
fn fourbar_lengths() -> impl Strategy<Value = (f64, f64, f64, f64)> {
    (0.5..5.0_f64, 0.5..5.0_f64, 0.5..5.0_f64, 0.5..5.0_f64)
        .prop_filter("must form a valid quadrilateral", |(a, b, c, d)| {
            // Each link must be less than the sum of the other three.
            let sum = a + b + c + d;
            *a < sum - a && *b < sum - b && *c < sum - c && *d < sum - d
        })
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

proptest! {
    /// Verify that the Grubler DOF count is correct for a 4-bar with a driver.
    ///
    /// A 4-bar with 3 moving bodies (9 coords), 4 revolute joints (8 eqs),
    /// and 1 driver (1 eq) should have DOF = 9 - 9 = 0.
    ///
    /// Without the driver, DOF = 9 - 8 = 1.
    #[test]
    fn grubler_dof_matches_expected(
        (ground, crank, coupler, rocker) in fourbar_lengths(),
    ) {
        let mech = build_random_fourbar(ground, crank, coupler, rocker);
        let n = mech.state().n_coords();    // 9 for 3 moving bodies
        let m = mech.n_constraints();        // 9 for 4 revolute + 1 driver

        // With driver: fully constrained
        prop_assert_eq!(n, 9, "expected 9 coords for 3 moving bodies, got {}", n);
        prop_assert_eq!(m, 9, "expected 9 constraints (4 rev + 1 driver), got {}", m);
        prop_assert_eq!(n as isize - m as isize, 0, "DOF should be 0 with driver");

        // Verify the constraint decomposition
        let all = mech.all_constraints();
        prop_assert_eq!(all.len(), 5, "expected 5 constraints (4 joints + 1 driver)");

        // Each revolute joint contributes 2 equations, driver contributes 1
        let joint_eqs: usize = mech.joints().iter().map(|j| j.n_equations()).sum();
        prop_assert_eq!(joint_eqs, 8, "4 revolute joints should give 8 equations");
    }

    /// After a convergent position solve, the constraint residual must be near zero.
    #[test]
    fn constraint_residual_near_zero_after_solve(
        (ground, crank, coupler, rocker) in fourbar_lengths(),
        angle in 0.1_f64..6.0,
    ) {
        let mech = build_random_fourbar(ground, crank, coupler, rocker);

        // Try to get a valid initial guess at this angle
        let q0 = match fourbar_initial_guess(&mech, ground, crank, coupler, rocker, angle) {
            Some(q) => q,
            None => return Ok(()),  // Geometry doesn't close at this angle — skip
        };

        if let Ok(result) = solve_position(&mech, &q0, angle / (2.0 * PI), 1e-10, 50) {
            if result.converged {
                prop_assert!(
                    result.residual_norm < 1e-8,
                    "converged solution should have residual < 1e-8, got {}",
                    result.residual_norm
                );

                // Double-check by independently evaluating the constraints
                let phi = assemble_constraints(&mech, &result.q, angle / (2.0 * PI));
                prop_assert!(
                    phi.norm() < 1e-8,
                    "independently assembled constraint norm should be < 1e-8, got {}",
                    phi.norm()
                );
            }
        }
    }

    /// After position + velocity solve, the velocity-level constraint equation
    /// Phi_q * q_dot + Phi_t = 0 must be satisfied.
    #[test]
    fn velocity_satisfies_constraint(
        (ground, crank, coupler, rocker) in fourbar_lengths(),
        angle in 0.1_f64..6.0,
    ) {
        let mech = build_random_fourbar(ground, crank, coupler, rocker);

        let q0 = match fourbar_initial_guess(&mech, ground, crank, coupler, rocker, angle) {
            Some(q) => q,
            None => return Ok(()),
        };

        let t = angle / (2.0 * PI);
        if let Ok(pos_result) = solve_position(&mech, &q0, t, 1e-10, 50) {
            if !pos_result.converged {
                return Ok(());
            }

            if let Ok(q_dot) = solve_velocity(&mech, &pos_result.q, t) {
                // Verify: Phi_q * q_dot + Phi_t should be approximately zero
                let phi_q = assemble_jacobian(&mech, &pos_result.q, t);
                let phi_t = assemble_phi_t(&mech, &pos_result.q, t);
                let residual = &phi_q * &q_dot + &phi_t;

                prop_assert!(
                    residual.norm() < 1e-8,
                    "velocity constraint residual Phi_q*q_dot + Phi_t should be near zero, got {}",
                    residual.norm()
                );

                // q_dot should be finite (no NaN or Inf)
                for i in 0..q_dot.len() {
                    prop_assert!(
                        q_dot[i].is_finite(),
                        "q_dot[{}] = {} is not finite",
                        i,
                        q_dot[i]
                    );
                }
            }
        }
    }

    /// Grashof classification must be deterministic: same inputs always give
    /// the same result, and the classification is consistent with the link lengths.
    #[test]
    fn grashof_classification_is_deterministic_and_consistent(
        (a, b, c, d) in fourbar_lengths(),
    ) {
        let r1 = check_grashof(a, b, c, d, 1e-10);
        let r2 = check_grashof(a, b, c, d, 1e-10);

        // Determinism
        prop_assert_eq!(r1.is_grashof, r2.is_grashof,
            "Grashof flag should be deterministic");
        prop_assert_eq!(r1.classification, r2.classification,
            "Grashof classification should be deterministic");
        prop_assert_eq!(r1.is_change_point, r2.is_change_point,
            "change-point flag should be deterministic");

        // Consistency: shortest <= longest
        prop_assert!(
            r1.shortest <= r1.longest,
            "shortest ({}) should be <= longest ({})",
            r1.shortest, r1.longest
        );

        // Consistency: S+L relationship matches is_grashof flag
        if r1.grashof_sum > r1.other_sum + 1e-10 {
            prop_assert!(
                !r1.is_grashof,
                "S+L={} > P+Q={} but is_grashof=true",
                r1.grashof_sum, r1.other_sum
            );
            prop_assert_eq!(r1.classification, GrashofType::NonGrashof,
                "S+L > P+Q should classify as NonGrashof");
        }

        // If Grashof and not change-point, classification must match shortest link
        if r1.is_grashof && !r1.is_change_point {
            match r1.shortest_is {
                "ground" => prop_assert_eq!(r1.classification, GrashofType::DoubleCrank),
                "crank" | "rocker" => prop_assert_eq!(r1.classification, GrashofType::CrankRocker),
                "coupler" => prop_assert_eq!(r1.classification, GrashofType::DoubleRocker),
                _ => prop_assert!(false, "unexpected shortest_is: {}", r1.shortest_is),
            }
        }
    }

    /// JSON serialization round-trip must preserve the mechanism's structure:
    /// same body count, joint count, attachment points, and body properties.
    #[test]
    fn serialization_roundtrip_preserves_mechanism(
        (ground_len, crank_len, coupler_len, rocker_len) in fourbar_lengths(),
    ) {
        // Build without a driver since drivers with closures cannot be
        // round-tripped through JSON (the serialization format uses
        // constant-speed metadata which IS round-trippable, so we use that).
        let mech = build_random_fourbar(ground_len, crank_len, coupler_len, rocker_len);

        let json_str = save_mechanism(&mech).expect("serialization should succeed");
        let loaded = load_mechanism(&json_str).expect("deserialization should succeed");

        // Body count matches (including ground)
        prop_assert_eq!(
            loaded.bodies().len(), mech.bodies().len(),
            "body count mismatch after round-trip"
        );

        // Joint count matches
        prop_assert_eq!(
            loaded.joints().len(), mech.joints().len(),
            "joint count mismatch after round-trip"
        );

        // Driver count matches (constant-speed drivers are serializable)
        prop_assert_eq!(
            loaded.n_drivers(), mech.n_drivers(),
            "driver count mismatch after round-trip"
        );

        // Coordinate count matches
        prop_assert_eq!(
            loaded.state().n_coords(), mech.state().n_coords(),
            "coordinate count mismatch after round-trip"
        );

        // Constraint count matches
        prop_assert_eq!(
            loaded.n_constraints(), mech.n_constraints(),
            "constraint count mismatch after round-trip"
        );

        // Verify each body's attachment points survived
        for (body_id, original_body) in mech.bodies() {
            let loaded_body = loaded.bodies().get(body_id)
                .expect(&format!("body '{}' not found after round-trip", body_id));

            prop_assert_eq!(
                original_body.attachment_points.len(),
                loaded_body.attachment_points.len(),
                "attachment point count mismatch for body '{}'",
                body_id
            );

            for (pt_name, pt_coords) in &original_body.attachment_points {
                let loaded_pt = loaded_body.attachment_points.get(pt_name)
                    .expect(&format!("point '{}' on body '{}' not found after round-trip", pt_name, body_id));
                prop_assert!(
                    (loaded_pt.x - pt_coords.x).abs() < 1e-12,
                    "x-coordinate mismatch for point '{}' on body '{}'",
                    pt_name, body_id
                );
                prop_assert!(
                    (loaded_pt.y - pt_coords.y).abs() < 1e-12,
                    "y-coordinate mismatch for point '{}' on body '{}'",
                    pt_name, body_id
                );
            }
        }
    }

    /// The Jacobian matrix dimensions must always match n_constraints x n_coords.
    #[test]
    fn jacobian_dimensions_are_correct(
        (ground, crank, coupler, rocker) in fourbar_lengths(),
    ) {
        let mech = build_random_fourbar(ground, crank, coupler, rocker);
        let q = mech.state().make_q();  // zero-initialized
        let phi_q = assemble_jacobian(&mech, &q, 0.0);

        prop_assert_eq!(
            phi_q.nrows(), mech.n_constraints(),
            "Jacobian rows should equal n_constraints"
        );
        prop_assert_eq!(
            phi_q.ncols(), mech.state().n_coords(),
            "Jacobian cols should equal n_coords"
        );

        // All entries should be finite
        for i in 0..phi_q.nrows() {
            for j in 0..phi_q.ncols() {
                prop_assert!(
                    phi_q[(i, j)].is_finite(),
                    "Jacobian[{}, {}] = {} is not finite",
                    i, j, phi_q[(i, j)]
                );
            }
        }
    }

    /// Constraint ranges must partition the constraint vector without
    /// gaps or overlaps, and must cover exactly n_constraints() rows.
    #[test]
    fn constraint_ranges_partition_correctly(
        (ground, crank, coupler, rocker) in fourbar_lengths(),
    ) {
        let mech = build_random_fourbar(ground, crank, coupler, rocker);
        let ranges = mech.constraint_ranges();
        let constraints = mech.all_constraints();

        prop_assert_eq!(
            ranges.len(), constraints.len(),
            "constraint_ranges length should match all_constraints length"
        );

        let mut expected_row = 0_usize;
        for (range, constraint) in ranges.iter().zip(constraints.iter()) {
            prop_assert_eq!(
                range.constraint_id.as_str(), constraint.id(),
                "constraint ID mismatch at row {}",
                expected_row
            );
            prop_assert_eq!(
                range.row_start, expected_row,
                "row_start mismatch for constraint '{}'",
                range.constraint_id
            );
            prop_assert_eq!(
                range.n_equations, constraint.n_equations(),
                "n_equations mismatch for constraint '{}'",
                range.constraint_id
            );
            expected_row += range.n_equations;
        }

        prop_assert_eq!(
            expected_row, mech.n_constraints(),
            "total rows from ranges should equal n_constraints"
        );
    }
}
